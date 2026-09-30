"""双目标定的核心计算。

流程（每一步都是为了让结果"能投入生产"）：

  1. 各自单目标定        -> K1,D1 / K2,D2（用 calibrateCameraExtended，能拿到逐图重投影误差）
  2. 双目联合标定         -> R,T（先固定内参，只优化外参，稳）
  3. 整体精化             -> 放开内参一起优化，RMS 通常能再降一半
  4. 离群剔除 + 重算      -> 用"逐对视差误差"挑出坏数据，丢掉后重算，最多迭代几轮
  5. 立体校正             -> R1,R2,P1,P2,Q
  6. 质量验证             -> 重投影 RMS、极线误差、基线长度，给出可用性判定

第 4 步是最容易被忽略、但对最终质量影响最大的一步：
少量模糊或贴边的图会把整体误差抬高一个量级，而它们在几百个角点里"看不出来"。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .board import BoardSpec


@dataclass
class CalibConfig:
    """标定配置。默认值按"20-30 对图像、普通 USB 相机"调好的。"""

    refine_intrinsics: bool = True     # 第 3 步：放开内参整体精化
    rational_model: bool = False       # 用 8 参数畸变模型（视角很广、畸变很大时再开）
    fix_principal_point: bool = False  # 数据很少时固定主点，避免过拟合
    reject_outliers: bool = True       # 第 4 步：离群剔除
    outlier_sigma: float = 3.0         # 超过 中位数 + k*MAD 就判为坏数据
    outlier_min_px: float = 0.5        # 但至少超过这个绝对量才剔，防止误杀
    max_reject_rounds: int = 4
    min_pairs_after_reject: int = 8
    alpha: float = 0.0                 # stereoRectify 的裁剪系数：0=裁掉黑边，-1=保留全部像素

    # 可用性判定门槛（像素）
    good_rms: float = 0.35
    ok_rms: float = 0.80
    good_epipolar: float = 0.25
    ok_epipolar: float = 0.60

    expected_baseline_mm: float | None = None   # 若提供，用于检查标定是否离谱
    baseline_tolerance: float = 0.20            # 允许的相对偏差


@dataclass
class StereoCalibrationResult:
    """标定结果 + 质量报告。"""

    ok: bool = False
    K1: np.ndarray | None = None
    D1: np.ndarray | None = None
    K2: np.ndarray | None = None
    D2: np.ndarray | None = None
    R: np.ndarray | None = None
    T: np.ndarray | None = None
    E: np.ndarray | None = None
    F: np.ndarray | None = None
    R1: np.ndarray | None = None
    R2: np.ndarray | None = None
    P1: np.ndarray | None = None
    P2: np.ndarray | None = None
    Q: np.ndarray | None = None
    roi1: tuple = ()
    roi2: tuple = ()
    image_size: tuple[int, int] = (0, 0)

    n_pairs_in: int = 0
    n_pairs_used: int = 0
    rms: float = float("nan")               # stereoCalibrate 返回的整体 RMS
    err_left: float = float("nan")          # 左目重投影 RMS
    err_right: float = float("nan")
    epipolar_rms: float = float("nan")      # 校正后的极线误差（真正决定三维精度）
    epipolar_max: float = float("nan")
    baseline_mm: float = float("nan")
    per_pair_epipolar: list[float] = field(default_factory=list)
    used_index: list[int] = field(default_factory=list)
    rejected_index: list[int] = field(default_factory=list)
    verdict: str = ""
    warnings: list[str] = field(default_factory=list)
    board_desc: str = ""

    def report(self) -> str:
        lines = []
        lines.append("=" * 62)
        lines.append("双目标定结果")
        lines.append("=" * 62)
        lines.append(f"标定板        : {self.board_desc}")
        lines.append(f"图像尺寸      : {self.image_size[0]}x{self.image_size[1]}")
        lines.append(f"使用/输入图像 : {self.n_pairs_used}/{self.n_pairs_in}"
                     + (f"（剔除 {len(self.rejected_index)} 对）" if self.rejected_index else ""))
        lines.append(f"整体重投影 RMS: {self.rms:.4f} px")
        lines.append(f"左目重投影 RMS: {self.err_left:.4f} px")
        lines.append(f"右目重投影 RMS: {self.err_right:.4f} px")
        lines.append(f"极线误差 RMS  : {self.epipolar_rms:.4f} px   (最大 {self.epipolar_max:.4f})")
        lines.append(f"基线长度      : {self.baseline_mm:.2f} mm")
        lines.append("-" * 62)
        lines.append(f"判定          : {self.verdict}")
        for w in self.warnings:
            lines.append(f"  ! {w}")
        lines.append("=" * 62)
        return "\n".join(lines)


# --------------------------------------------------------------------------
def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    return float(np.sqrt(np.mean(x * x))) if x.size else float("nan")


def _calibrate_one(objp, imgp, size, cfg: CalibConfig):
    """单目标定，返回 (K, D, 逐图重投影误差)。"""
    flags = 0
    if cfg.rational_model:
        flags |= cv2.CALIB_RATIONAL_MODEL
    if cfg.fix_principal_point:
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-7)
    out = cv2.calibrateCameraExtended(objp, imgp, size, None, None, criteria=criteria, flags=flags)
    _ret, K, D, _rv, _tv, _sdi, _sde, per_view = out
    return K, D, np.asarray(per_view, dtype=np.float64).ravel()


def _stereo_once(objp, ip1, ip2, K1, D1, K2, D2, size, cfg: CalibConfig):
    """跑一次双目联合标定：先固定内参定外参，再整体精化。"""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)
    out = cv2.stereoCalibrate(
        objp, ip1, ip2, K1, D1, K2, D2, size,
        criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC,
    )
    rms, K1, D1, K2, D2, R, T, E, F = out

    if cfg.refine_intrinsics:
        flags = cv2.CALIB_USE_INTRINSIC_GUESS
        if cfg.rational_model:
            flags |= cv2.CALIB_RATIONAL_MODEL
        if cfg.fix_principal_point:
            flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        out2 = cv2.stereoCalibrate(
            objp, ip1, ip2, K1, D1, K2, D2, size, criteria=criteria, flags=flags,
        )
        rms2, K1b, D1b, K2b, D2b, R2, T2, E2, F2 = out2
        # 精化不一定总是更好，取更小的那个
        if np.isfinite(rms2) and rms2 <= rms:
            rms, K1, D1, K2, D2, R, T, E, F = rms2, K1b, D1b, K2b, D2b, R2, T2, E2, F2
    return rms, K1, D1, K2, D2, R, T, E, F


def _rectify(K1, D1, K2, D2, size, R, T, alpha: float):
    return cv2.stereoRectify(
        K1, D1, K2, D2, size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha,
    )


def _epipolar_errors(ip1, ip2, K1, D1, K2, D2, R1, R2, P1, P2) -> np.ndarray:
    """逐对图像的极线误差：校正后同名点的 y 之差。"""
    errs = []
    for c1, c2 in zip(ip1, ip2):
        p1 = cv2.undistortPoints(c1.reshape(-1, 1, 2).astype(np.float64), K1, D1, R=R1, P=P1).reshape(-1, 2)
        p2 = cv2.undistortPoints(c2.reshape(-1, 1, 2).astype(np.float64), K2, D2, R=R2, P=P2).reshape(-1, 2)
        n = min(len(p1), len(p2))
        dy = p1[:n, 1] - p2[:n, 1]
        errs.append(float(np.sqrt(np.mean(dy * dy))))
    return np.asarray(errs, dtype=np.float64)


# --------------------------------------------------------------------------
def _as_objpoints(objp, n: int) -> list[np.ndarray]:
    """把物体点规整成"每视图一个 (N,3) float32 数组"。

    允许传单个 (N,3) 数组（所有视图共用同一块标定板，本项目的常态），
    也允许传逐视图的列表（OpenCV 原生风格）。
    顺便统一转成 float32 —— OpenCV 5 对 float64 的物体点会直接报错。
    """
    if isinstance(objp, np.ndarray) and objp.ndim == 2:
        a = objp.astype(np.float32, copy=False)
        return [a] * n
    out = [np.asarray(o, np.float32).reshape(-1, 3) for o in objp]
    if len(out) != n:
        raise ValueError(f"物体点数量 {len(out)} 与视图数量 {n} 不一致")
    return out


def _as_imgpoints(pts: list[np.ndarray]) -> list[np.ndarray]:
    return [np.asarray(p, np.float32).reshape(-1, 1, 2) for p in pts]


def calibrate_stereo(
    objp,
    imgpoints1: list[np.ndarray],
    imgpoints2: list[np.ndarray],
    image_size: tuple[int, int],
    cfg: CalibConfig | None = None,
    spec: BoardSpec | None = None,
) -> StereoCalibrationResult:
    """从已配对的角点做完整双目标定（含离群剔除与质量判定）。

    objp 可以是单个 (N,3) 数组（所有视图共用一块板），也可以是逐视图列表。
    """
    cfg = cfg or CalibConfig()
    spec = spec or BoardSpec()
    res = StereoCalibrationResult(image_size=image_size, board_desc=spec.describe(),
                                  n_pairs_in=len(imgpoints1))
    if len(imgpoints1) < 4 or len(imgpoints1) != len(imgpoints2):
        res.verdict = "数据不足或左右数量不一致"
        return res

    obj_all = _as_objpoints(objp, len(imgpoints1))
    imgpoints1 = _as_imgpoints(imgpoints1)
    imgpoints2 = _as_imgpoints(imgpoints2)

    idx = list(range(len(imgpoints1)))
    best = None
    for round_i in range(cfg.max_reject_rounds if cfg.reject_outliers else 1):
        o = [obj_all[i] for i in idx]
        p1 = [imgpoints1[i] for i in idx]
        p2 = [imgpoints2[i] for i in idx]

        K1, D1, e1 = _calibrate_one(o, p1, image_size, cfg)
        K2, D2, e2 = _calibrate_one(o, p2, image_size, cfg)
        rms, K1, D1, K2, D2, R, T, E, F = _stereo_once(o, p1, p2, K1, D1, K2, D2, image_size, cfg)
        R1, R2, P1, P2, Q, roi1, roi2 = _rectify(K1, D1, K2, D2, image_size, R, T, cfg.alpha)
        epi = _epipolar_errors(p1, p2, K1, D1, K2, D2, R1, R2, P1, P2)

        # 逐对综合误差：左右重投影误差与极线误差，谁大听谁的
        per_pair = np.maximum(np.maximum(e1, e2), epi)
        med = float(np.median(per_pair))
        mad = float(np.median(np.abs(per_pair - med))) * 1.4826   # 稳健标准差
        thresh = max(med + cfg.outlier_sigma * mad, cfg.outlier_min_px)

        cur = dict(idx=list(idx), K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T, E=E, F=F,
                   R1=R1, R2=R2, P1=P1, P2=P2, Q=Q, roi1=roi1, roi2=roi2,
                   rms=float(rms), e1=float(_rms(e1)), e2=float(_rms(e2)),
                   epi=epi, per_pair=per_pair)
        if best is None or cur["rms"] <= best["rms"]:
            best = cur

        if not cfg.reject_outliers:
            break
        bad = [i for i, v in zip(idx, per_pair) if v > thresh]
        if not bad or len(idx) - len(bad) < cfg.min_pairs_after_reject:
            break
        idx = [i for i in idx if i not in set(bad)]

    b = best
    res.K1, res.D1, res.K2, res.D2 = b["K1"], b["D1"], b["K2"], b["D2"]
    res.R, res.T, res.E, res.F = b["R"], b["T"], b["E"], b["F"]
    res.R1, res.R2, res.P1, res.P2, res.Q = b["R1"], b["R2"], b["P1"], b["P2"], b["Q"]
    res.roi1, res.roi2 = b["roi1"], b["roi2"]
    res.rms, res.err_left, res.err_right = b["rms"], b["e1"], b["e2"]
    res.epipolar_rms = _rms(b["epi"])
    res.epipolar_max = float(np.max(b["epi"])) if len(b["epi"]) else float("nan")
    res.per_pair_epipolar = [float(v) for v in b["epi"]]
    res.used_index = list(b["idx"])
    res.rejected_index = [i for i in range(res.n_pairs_in) if i not in set(b["idx"])]
    res.n_pairs_used = len(b["idx"])
    res.baseline_mm = float(np.linalg.norm(b["T"])) * 1000.0

    # ---- 判定 ----
    warns: list[str] = []
    if res.n_pairs_used < 8:
        warns.append(f"有效图像只有 {res.n_pairs_used} 对，建议补到 15 对以上")
    if res.rms > cfg.ok_rms:
        warns.append(f"重投影 RMS {res.rms:.2f} px 偏大：检查标定板是否平整、图像是否模糊")
    if res.epipolar_rms > cfg.ok_epipolar:
        warns.append(f"极线误差 {res.epipolar_rms:.2f} px 偏大：左右相机的相对位姿可能不稳定")
    if cfg.expected_baseline_mm:
        rel = abs(res.baseline_mm - cfg.expected_baseline_mm) / cfg.expected_baseline_mm
        if rel > cfg.baseline_tolerance:
            warns.append(
                f"基线 {res.baseline_mm:.1f} mm 与预期 {cfg.expected_baseline_mm:.1f} mm "
                f"相差 {rel * 100:.0f}%，请核对标定板方格尺寸是否填对"
            )
    res.warnings = warns

    if res.rms <= cfg.good_rms and res.epipolar_rms <= cfg.good_epipolar:
        res.verdict = "优秀 —— 可以直接用于三角测量"
        res.ok = True
    elif res.rms <= cfg.ok_rms and res.epipolar_rms <= cfg.ok_epipolar:
        res.verdict = "可用 —— 满足三维重建要求"
        res.ok = True
    else:
        res.verdict = "不可用 —— 请重拍标定图像（换更多角度、保证清晰、铺满画面）"
        res.ok = False
    return res
