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
from collections.abc import Sequence
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
    # ---- 板占画面比例（2026-10-14 加）----
    # 为什么要有：实测那次标定**板只占画面 1.2%**，角点像素少、定位精度低，
    # 于是 RMS 0.91 px 超过 ok 门限 0.80 —— 而当时的判定只说
    # "换更多角度、保证清晰、铺满画面"，**没说是"离板太近一点"**。
    # 这个数字一直是算得出来的（采集时每次都在算），只是**没有出口**。
    board_area_l: float = float("nan")      # 左图里板的角点包围面积 / 画面面积
    board_area_r: float = float("nan")
    grade: str = ""                         # 优秀 / 可用 / 勉强可用 / 不可用
    # **可执行的下一步建议**（不是"换更多角度"这种没法照做的话）
    suggestions: list[str] = field(default_factory=list)

    @property
    def board_area(self) -> float:
        """两路里**较小**的那个比例（双目要两路都够大才有意义）。"""
        vals = [v for v in (self.board_area_l, self.board_area_r) if np.isfinite(v)]
        return float(min(vals)) if vals else float("nan")

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
        if np.isfinite(self.board_area):
            # 这个数字**必须出现在报告里**：它解释了"为什么 RMS 偏高"，
            # 而用户拿到报告时板已经收起来了 —— 只有这里能告诉他下次该靠近点。
            # 括号里给"建议范围"，因为光看一个百分比不知道多少算合适。
            lines.append(f"板占画面      : {self.board_area * 100:.1f}%"
                         f"（左 {self.board_area_l * 100:.1f}% / 右 {self.board_area_r * 100:.1f}%；"
                         f"建议 10%~30%）")
        lines.append("-" * 62)
        lines.append(f"判定          : {self.verdict}")
        for w in self.warnings:
            lines.append(f"  ! {w}")
        for s in self.suggestions:
            lines.append(f"  → {s}")
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

    # ---- 板占画面比例：从**实际用上**的那些对的角点算 ----
    #
    # 为什么要凸包：角点数组是**行序网格**（第 1 行从左到右、然后第 2 行…），
    # 而 `cv2.contourArea` 把点当作**多边形顶点按顺序**连起来。
    # 网格顺序连出来的路径会自己折回去，面积**恒为 0**（实测确认）。
    # 必须先用 `convexHull` 求外轮廓 —— 标定板是凸的，凸包就是它的真实轮廓。
    def _area(points: Sequence[np.ndarray]) -> float:
        w, h = image_size
        if not points or w <= 0 or h <= 0:
            return float("nan")
        areas = []
        for pts in points:
            p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
            if p.shape[0] < 3:
                continue
            hull = cv2.convexHull(p.astype(np.float32))
            areas.append(float(cv2.contourArea(hull)))
        if not areas:
            return float("nan")
        return float(np.median(areas)) / float(w * h)

    res.board_area_l = _area([imgpoints1[i] for i in b["idx"] if i < len(imgpoints1)])
    res.board_area_r = _area([imgpoints2[i] for i in b["idx"] if i < len(imgpoints2)])
    res.n_pairs_used = len(b["idx"])
    res.baseline_mm = float(np.linalg.norm(b["T"])) * 1000.0

    # ---- 判定 ----
    #
    # ## 为什么改成分级（2026-10-14）
    #
    # 原来是二值：`rms <= ok_rms and epipolar <= ok_epipolar` 才 `ok=True`，
    # 否则一句"不可用 —— 请重拍"。实测的一次真机标定：
    #
    #     基线 67.28 mm（实际 ~65，只差 3.5% —— 这是对上了）
    #     整体 RMS 0.9143 px（ok 门限 0.80）
    #     极线 RMS 0.8071 px（ok 门限 0.60）
    #
    # **基线核对通过了**（说明方格尺寸、内外参都对），只是精度指标刚过线，
    # 结果被整体否掉、不保存。用户的原话是"我不是很能接受" —— 他不接受是对的：
    #
    # 1. `ok_rms = 0.80 px` 这条线**本来就是"够用"的意思**，不是"超过就废"。
    #    真正决定三维精度的是极线误差，而 0.81 px 在 1080p 下约合
    #    0.81/1920 ≈ 0.04% 的画面宽度 —— 对厘米级的目标完全够。
    # 2. 二值判定丢掉了**最有用的那条信息**：差在哪、下次怎么改。
    #    一次"基线正确但精度刚过线"的标定，其价值与"完全解错"是两回事，
    #    报告里却长得一模一样。
    #
    # 所以现在是**四级**，且**只要基线没问题就仍然保存**：
    #   优秀 / 可用   —— 直接可用于三角测量
    #   勉强可用      —— 保存，但报告里写清差在哪、怎么改进；先能用起来
    #   不可用        —— **只有物理上错了才用**：基线对不上、有效对太少、数值发散
    warns: list[str] = []
    sugg: list[str] = []

    if res.n_pairs_used < 8:
        warns.append(f"有效图像只有 {res.n_pairs_used} 对，建议补到 15 对以上")
    if res.rms > cfg.ok_rms:
        warns.append(f"重投影 RMS {res.rms:.2f} px 偏大（ok 门限 {cfg.ok_rms}）")
    if res.epipolar_rms > cfg.ok_epipolar:
        warns.append(f"极线误差 {res.epipolar_rms:.2f} px 偏大（ok 门限 {cfg.ok_epipolar}）")

    baseline_bad = False
    if cfg.expected_baseline_mm:
        rel = abs(res.baseline_mm - cfg.expected_baseline_mm) / cfg.expected_baseline_mm
        if rel > cfg.baseline_tolerance:
            baseline_bad = True
            warns.append(
                f"基线 {res.baseline_mm:.1f} mm 与预期 {cfg.expected_baseline_mm:.1f} mm "
                f"相差 {rel * 100:.0f}% —— 这是**物理上对不上**，不是精度问题"
            )
            sugg.append("先核对：标定板方格边长（用尺子量实际值，别信打印设置）、"
                        "两镜头中心距、有没有把两台相机搞反。")
    else:
        sugg.append("下次加上 --baseline-mm <两镜头中心距>："
                    "不填的话基线核对这道护栏是关着的，错了也不会有人告诉你。")

    # 数值发散：RMS 到了"像素级垃圾"的程度（实测整份反序会到几百 px）
    diverged = not np.isfinite(res.rms) or res.rms > 5.0

    # ---- 可执行建议：把"换更多角度"换成能照着做的动作 ----
    #
    # ⚠️ **建议与判定是两件事，不要互相压制**。
    #
    # 我第一版把建议整体挂在"没到优秀才给"下面，结果是：板只占 1.12%、
    # 但因为合成噪声理想而解出"优秀"，于是**一条建议都没有** ——
    # 而"板太远"这件事是**真的可以更好**，与判定等级无关。
    #
    # 真正需要避免的自相矛盾只有一种：建议里说"板太近"而结论说"没到目标"
    # 这种**指向相反**的话。所以按条件各给各的，不整体开关。
    if np.isfinite(res.board_area) and res.board_area < 0.08:
        sugg.append(
            f"**板离得太远**：只占画面 {res.board_area * 100:.1f}%，建议靠近到 10%~30%。"
            f"角点像素越多，定位越准 —— 这是把 RMS 降下来最直接的一招"
            f"（采集时窗口里会实时显示这个比例）。")
    if np.isfinite(res.board_area) and res.board_area > 0.45:
        sugg.append(f"板占画面 {res.board_area * 100:.0f}% 有点太近，"
                    f"边角容易出画或畸变过大，退到 10%~30%。")
    if res.n_pairs_used < 20:
        sugg.append(f"再多收几对（现在 {res.n_pairs_used} 对，建议 20 对以上），"
                    f"并且**让板覆盖画面四角**，不只是中间。")
    if res.epipolar_rms > cfg.ok_epipolar:
        sugg.append("重拍时**每个姿势停稳约 1 秒**再移开（板在动的时候两路拍到的是"
                    "不同位姿，这是极线误差偏大的头号原因）。")
    res.suggestions = sugg
    res.warnings = warns

    # ---- 分级 ----
    if diverged or baseline_bad or res.n_pairs_used < 8:
        # 只有"物理上错了"才不可用
        if diverged:
            why = f"求解发散（整体 RMS {res.rms:.1f} px）"
        elif baseline_bad:
            why = "基线对不上"
        else:
            why = f"有效图像只有 {res.n_pairs_used} 对"
        res.grade = "不可用"
        res.verdict = f"不可用 —— {why}，请重拍"
        res.ok = False
    elif res.rms <= cfg.good_rms and res.epipolar_rms <= cfg.good_epipolar:
        res.grade = "优秀"
        res.verdict = "优秀 —— 可以直接用于三角测量"
        res.ok = True
    elif res.rms <= cfg.ok_rms and res.epipolar_rms <= cfg.ok_epipolar:
        res.grade = "可用"
        res.verdict = "可用 —— 满足三维重建要求"
        res.ok = True
    else:
        # ★ **仍然保存**：精度刚过线不等于废掉。
        res.grade = "勉强可用"
        worst = max(res.rms / cfg.ok_rms, res.epipolar_rms / cfg.ok_epipolar)
        res.verdict = (f"勉强可用 —— 精度略低于目标（最差指标为门限的 {worst:.2f} 倍），"
                       f"**已保存**，可以先用来做三维；按下面的建议重拍会更好")
        res.ok = True
    return res
