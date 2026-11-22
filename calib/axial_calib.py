"""模式 A：轴向标定（平行双目，板正对相机 + 已知沿光轴推进量 + 已知方格边长）。

## 为什么单独一个求解器

现有 `calib/stereo_calib.py` 与 `calib/capture.py` 会**主动拒收**"姿态太相似"的
数据：`capture.py` 的 `diversity_min`、`mono_calib.py` 的 fx 留一法、以及
`stereo_calib.py` 的"板占画面 >45% 太近"。而轴向标定的物理前提恰恰是
**只沿光轴前后推** —— 姿态在多样性意义上几乎不变，但成像尺度 `p` 随深度线性
变化，焦距与初始深度因此可被线性拟合唯一确定。

所以本模块走**独立口径**，不改现有求解器的任何判定分支与门限。产物与
`StereoCalibrationResult` 同构，能被 `calib/params.py::StereoRig.from_result`
直接消费。

## 数学（已在合成数据上严格自验）

约定：相机系 OpenCV（+X 右 / +Y 下 / +Z 前，光轴 +Z）。板正对相机 → 板面 ⊥ 光轴；
板只沿光轴推进 `ΔZ_i`（已知，来自位移台/刻度）；方格边长 `L = square_mm/1000` 已知。

设板中心在相机系的深度为 `Z_i = Z0 + ΔZ_i`。相邻角点在图像里的像素间距：

    p_x,i = fx · L / Z_i,        p_y,i = fy · L / Z_i

因此

    1 / p_x,i = Z0 / (fx·L) + ΔZ_i / (fx·L)        （对 ΔZ_i 是线性关系）

令斜率 `b_x = 1/(fx·L)`、截距 `a_x = Z0/(fx·L)`，则

    fx = 1 / (b_x · L),        Z0 = a_x / b_x

`fy` 同理。横向与纵向**分开拟合**：只有当 fx≈fy 时把两方向混成一个 `p` 才严格；
分开拟合既不增加假设，也能直接暴露两个方向的差异。

主点用**网格对称性**估计：板正对时，固定角点 j 的横坐标随帧满足

    u_ij = cx + c_xj / Z_i            （c_xj = fx·(x_j + tx) 与帧无关）

对每一帧 i 以 `1/Z_i` 为自变量做直线拟合，**截距就是 cx**（cy 同理）。这一步同时
把主点 `cx` 与板的横向平移 `tx` 分离开 —— 只沿光轴推进让 `1/Z_i` 变化，截距因此
可辨；若 ΔZ 全相同，斜率退化，必须报错而不是给出假数。

切向畸变 `p1/p2` 在正对时**不可观测**，锁定为 0。径向 `k1/k2` 与尺度/主点在
"正对 + 纯 Z 推进"下强耦合、可观测性很弱；本模块**不硬凑**：默认输出
`D = [0,0,0,0,0]`，并在 `AxialIntrinsics.distortion_calibrated=False` 与 `notes`
里**显式标注"未标定径向畸变"**。需要畸变请走常规多姿态标定。

## 外参

两路各自用上面的方法解出 `K/D` 后，对**同一块板同一深度**的左右角点各做一次
`solvePnP`（板中心为原点），得到 `T_camL_A`、`T_camR_A`；同一块板意味着
`T_AB = I`，于是

    T_{C_R←C_L} = T_camR_A · inv(T_camL_A)

多帧用 `relative_pose.average_T`（旋转四元数加权平均 + 平移几何中位数）聚合。

## 本模块**不做**的事

- 不改 `CalibConfig` 白名单、不注册 `calib_method`（GUI/CLI 接入留给后续 worker）。
- 不放宽/不改变现有求解器的任何门限。
- 不猜 `Z0` / `f`：退化输入直接 `AxialCalibError`。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import relative_pose
from .board import BoardSpec
from .stereo_calib import (
    CalibConfig,
    StereoCalibrationResult,
    _board_area_fraction,
    _essential_fundamental,
)

__all__ = [
    "AxialCalibError",
    "AxialIntrinsics",
    "solve_axial_intrinsics",
    "solve_axial_stereo",
]

# ΔZ 的极差低于这个值（米）就认为"没有真正推进"，斜率不可辨。
_MIN_DZ_SPAN_M = 1e-6
# 至少需要这么多帧才能拟合一条直线并留出一次自洽性检验。
_MIN_FRAMES = 3


class AxialCalibError(ValueError):
    """轴向标定的输入不成立（不是数值失败，而是物理前提不满足）。"""


@dataclass
class AxialIntrinsics:
    """单路轴向内参结果。

    Attributes
    ----------
    fx, fy, cx, cy, Z0:
        焦距（像素）、主点（像素）、板在**第一帧**（ΔZ=0）处的深度（米）。
    Z0_x, Z0_y:
        横向/纵向分别拟合出的 Z0（正常情况下两者应接近；差异大说明板没正对或有
        明显的各向异性，调用方应看这个数）。
    distortion_calibrated:
        **诚实标记**。``False`` 表示径向畸变未标定，``D`` 全是 0，不要据此声称
        已经处理了畸变。
    rms_px:
        正对模型的逐角点重投影残差 RMS（像素）。它只反映"f/Z0/主点"这套模型与
        观测的自洽程度，**不是**完整相机模型的 RMS。
    n_frames, delta_z:
        实际用了多少帧、各自的推进量（米）。
    per_frame_rms:
        逐帧残差 RMS，用来挑坏帧。
    notes:
        给人看的说明（如"未标定径向畸变"）。
    """

    fx: float
    fy: float
    cx: float
    cy: float
    Z0: float
    Z0_x: float
    Z0_y: float
    distortion_calibrated: bool
    rms_px: float
    n_frames: int
    delta_z: np.ndarray
    per_frame_rms: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def matrix_K_D(self) -> tuple[np.ndarray, np.ndarray]:
        """返回 OpenCV 口径的 `(K, D)`。D 恒为 5 元（k1,k2,p1,p2,k3）。"""
        K = np.array([[self.fx, 0.0, self.cx],
                      [0.0, self.fy, self.cy],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        D = np.zeros(5, dtype=np.float64)
        return K, D


def _as_corner_matrix(corners, spec: BoardSpec, tag: str) -> np.ndarray:
    pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] != spec.n_points:
        raise AxialCalibError(
            f"{tag}: 角点数 {pts.shape[0]} 与板规格 {spec.n_points} 不一致"
            f"（{spec.cols}x{spec.rows}）；检查 BoardSpec 与检测结果是否同一块板")
    if not np.all(np.isfinite(pts)):
        raise AxialCalibError(f"{tag}: 角点含 NaN/inf，拒绝参与标定")
    return pts


def _square_pitches(pts: np.ndarray, spec: BoardSpec) -> tuple[float, float]:
    """一个方格在图像里的横向/纵向像素尺寸（相邻角点间距中位数）。

    用**局部**相邻间距而不是"整体宽/列数"：后者在板略斜时会把梯形平均掉，
    与成像尺度的物理含义（局部放大率）对不上。
    """
    g = pts.reshape(spec.rows, spec.cols, 2)
    hg = np.linalg.norm(g[:, 1:, :] - g[:, :-1, :], axis=2).ravel()
    vg = np.linalg.norm(g[1:, :, :] - g[:-1, :, :], axis=2).ravel()
    if hg.size == 0 or vg.size == 0:
        raise AxialCalibError("板规格太小，无法提取相邻角点间距（至少 2x2 内角点）")
    px = float(np.median(hg))
    py = float(np.median(vg))
    if not (px > 0.0 and py > 0.0):
        raise AxialCalibError(f"角点间距非正（px={px}, py={py}），检测结果不可用")
    return px, py


def _fit_intrinsics(
    corners_list: list[np.ndarray], dz: np.ndarray, spec: BoardSpec,
) -> tuple[float, float, float, float, float, float, float]:
    """核心线性拟合：返回 (fx, fy, Z0, Z0_x, Z0_y, cx, cy)。

    退化（ΔZ 全相同、斜率非正、帧数不足）抛 `AxialCalibError`。
    """
    L = spec.square_mm / 1000.0
    pitches = np.array([_square_pitches(c, spec) for c in corners_list])   # (n,2)
    inv_px = 1.0 / pitches[:, 0]
    inv_py = 1.0 / pitches[:, 1]

    # 1/p = a + b·ΔZ ；b = 1/(f·L)，a = Z0/(f·L)
    bx, ax = np.polyfit(dz, inv_px, 1)
    by, ay = np.polyfit(dz, inv_py, 1)
    if not (bx > 0.0 and by > 0.0):
        raise AxialCalibError(
            f"1/p 对 ΔZ 的斜率非正（bx={bx:.3e}, by={by:.3e}）："
            "板可能没有正对相机，或推进方向与光轴相反（板在远离时变小应使 1/p 增大）")
    fx = 1.0 / (bx * L)
    fy = 1.0 / (by * L)
    Z0_x = ax / bx
    Z0_y = ay / by
    Z0 = 0.5 * (Z0_x + Z0_y)
    if not (np.isfinite(fx) and np.isfinite(fy) and Z0 > 0.0):
        raise AxialCalibError(
            f"拟合结果非法：fx={fx}, fy={fy}, Z0={Z0}（Z0 必须为正的初始深度）")

    # 主点：逐角点跨帧拟合 u_ij = cx + c_j / Z_i，截距中位数即 cx
    zs = Z0 + dz
    cxs, cys = [], []
    A = np.column_stack([np.ones(dz.size), 1.0 / zs])
    for j in range(spec.n_points):
        u = np.array([c[j, 0] for c in corners_list])
        v = np.array([c[j, 1] for c in corners_list])
        sx, *_ = np.linalg.lstsq(A, u, rcond=None)
        sy, *_ = np.linalg.lstsq(A, v, rcond=None)
        cxs.append(float(sx[0]))
        cys.append(float(sy[0]))
    cx = float(np.median(cxs))
    cy = float(np.median(cys))
    return fx, fy, Z0, Z0_x, Z0_y, cx, cy


def _residual_rms(
    corners_list: list[np.ndarray], dz: np.ndarray, spec: BoardSpec,
    fx: float, fy: float, cx: float, cy: float, Z0: float,
) -> tuple[float, list[float]]:
    """正对模型下的逐角点重投影残差（每帧只拟合横向平移 tx/ty，2 个自由度）。"""
    obj = spec.object_points()[:, :2]       # (N,2) 米
    zs = Z0 + dz
    all_sq: list[float] = []
    per_frame: list[float] = []
    for i, pts in enumerate(corners_list):
        z = zs[i]
        # u_ij - cx - fx*x_j/z = (fx/z) * tx  -> 最小二乘解 tx
        ideal_x = fx * obj[:, 0] / z
        ideal_y = fy * obj[:, 1] / z
        tx = float(np.mean((pts[:, 0] - cx - ideal_x) * z / fx))
        ty = float(np.mean((pts[:, 1] - cy - ideal_y) * z / fy))
        du = pts[:, 0] - (cx + fx * (obj[:, 0] + tx) / z)
        dv = pts[:, 1] - (cy + fy * (obj[:, 1] + ty) / z)
        sq = np.concatenate([du, dv])
        all_sq.append(sq)
        per_frame.append(float(np.sqrt(np.mean(sq ** 2))))
    flat = np.concatenate(all_sq) if all_sq else np.zeros(0)
    rms = float(np.sqrt(np.mean(flat ** 2))) if flat.size else float("nan")
    return rms, per_frame


def solve_axial_intrinsics(
    corners_list, delta_z, spec: BoardSpec | None = None, *,
    min_frames: int = _MIN_FRAMES,
) -> AxialIntrinsics:
    """单路轴向内参：由"板正对 + 已知 ΔZ + 已知 L"的多帧角点解 `fx/fy/cx/cy/Z0`。

    Parameters
    ----------
    corners_list : 序列，每项是某帧的角点（`(N,2)` 或 `(N,1,2)`），**顺序必须与
        板模型一致**（行优先，和 `BoardSpec.object_points()` 对应）。
    delta_z : 与 `corners_list` 等长的推进量（米），第一帧通常为 0。
    spec : 标定板规格；缺省用 `BoardSpec()`。
    min_frames : 最少帧数（低于它无法既拟合直线又做自洽检验）。

    Raises
    ------
    AxialCalibError
        帧数不足、ΔZ 极差≈0、角点数不符、斜率非正、拟合结果非法。
    """
    spec = spec or BoardSpec()
    dz = np.asarray(delta_z, dtype=np.float64).ravel()
    n = len(corners_list)
    if n != dz.size:
        raise AxialCalibError(
            f"角点帧数 {n} 与推进量个数 {dz.size} 不一致")
    if n < max(2, int(min_frames)):
        raise AxialCalibError(
            f"轴向标定至少需要 {min_frames} 帧，实际只有 {n} 帧；"
            f"只沿光轴推进时，少于 3 帧既无法稳健拟合也无法自检")
    if float(np.ptp(dz)) < _MIN_DZ_SPAN_M:
        raise AxialCalibError(
            "推进量 ΔZ 全相同：1/p 对 ΔZ 的斜率不可辨，f 与 Z0 在数学上无解。"
            "请确认位移台真的产生了不同的推进量，并把每帧的 ΔZ 如实传入")
    if not np.all(np.isfinite(dz)):
        raise AxialCalibError("推进量 ΔZ 含 NaN/inf")

    pts_list = [_as_corner_matrix(c, spec, f"corners_list[{i}]")
                for i, c in enumerate(corners_list)]

    fx, fy, Z0, Z0x, Z0y, cx, cy = _fit_intrinsics(pts_list, dz, spec)
    rms, per_frame = _residual_rms(pts_list, dz, spec, fx, fy, cx, cy, Z0)

    notes = [
        "径向/切向畸变未标定（D=0）：正对 + 纯 Z 推进下畸变与尺度/主点强耦合，"
        "本求解器不硬凑；需要畸变请用常规多姿态标定。",
    ]
    if not (0.5 < Z0x / max(Z0y, 1e-12) < 2.0):
        notes.append(
            f"横向/纵向拟合的 Z0 差异较大（Z0_x={Z0x:.4f}, Z0_y={Z0y:.4f}）："
            "板可能没有严格正对，或横向/纵向角点间距检测受透视影响")

    return AxialIntrinsics(
        fx=fx, fy=fy, cx=cx, cy=cy, Z0=Z0, Z0_x=Z0x, Z0_y=Z0y,
        distortion_calibrated=False, rms_px=rms, n_frames=n,
        delta_z=dz.copy(), per_frame_rms=per_frame, notes=notes,
    )


def _board_pose_center(
    corners: np.ndarray, K: np.ndarray, D: np.ndarray, objc: np.ndarray,
) -> np.ndarray:
    """以**板中心为原点**的板位姿 `T_cam_A`（4x4）。"""
    img = np.asarray(corners, dtype=np.float64).reshape(-1, 1, 2)
    ok, rvec, tvec = cv2.solvePnP(objc.astype(np.float64), img, K, D,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise AxialCalibError(
            "solvePnP 失败：角点与板模型不匹配（检查行列数/角点顺序/BoardSpec）")
    R, _ = cv2.Rodrigues(rvec)
    return relative_pose.rt_to_T(R, np.asarray(tvec, np.float64).ravel())


def solve_axial_stereo(
    corners_left, corners_right, delta_z, spec: BoardSpec | None = None,
    image_size: tuple[int, int] = (0, 0), cfg: CalibConfig | None = None,
    *, assume_parallel: bool = False,
) -> StereoCalibrationResult:
    """轴向双目标定：两路各解内参 + 用共视同一板求基线外参。

    前提：**同一块正对的板在同一深度被左右两路各拍一张**（`corners_left[i]` 与
    `corners_right[i]` 是同一帧、同一深度）。两路推进量相同（`delta_z` 共用）。

    `assume_parallel=False`（默认）：外参取自锁定内参的 `stereoCalibrate`，产物
    `rectification="stereo_rectify"`（失败降级 `none_raw`）。

    `assume_parallel=True`：机械保证两光轴平行，**强制相对旋转 R = I**（只沿光轴
    推进时 R 弱可观测、可能差 ~1°，用它做极线校正会让极线歪掉）；基线 T 仍来自
    观测（stereoCalibrate 的平移）。产物按 `none_raw` 口径：`R1=R2=I`、
    `P1=K1[I|0]`、`P2=K2[I|T]`、`Q=None`，`E/F` 由 `(I,T)` 重算。

    两种情况若给了 `cfg.expected_baseline_mm` 且基线相对偏离超过
    `cfg.baseline_tolerance`，都会**强制降级**（ok=False / 不可用），因为这通常
    意味着装反或内参/推进量有系统错误。
    """
    spec = spec or BoardSpec()
    cfg = cfg or CalibConfig()
    dz = np.asarray(delta_z, dtype=np.float64).ravel()
    n_l, n_r = len(corners_left), len(corners_right)
    if n_l != n_r:
        raise AxialCalibError(
            f"左/右帧数不一致（{n_l} vs {n_r}）：轴向模式要求两路对同一块板在同一"
            f"深度的共视帧一一对应")

    intr_l = solve_axial_intrinsics(corners_left, dz, spec)
    intr_r = solve_axial_intrinsics(corners_right, dz, spec)
    K1, D1 = intr_l.matrix_K_D()
    K2, D2 = intr_r.matrix_K_D()

    c_board = (spec.object_points().min(0) + spec.object_points().max(0)) / 2.0
    objc = (spec.object_points() - c_board).astype(np.float64)

    # 逐帧 solvePnP 链：只用来给 consistency（多帧离散度）与回退值。
    samples: list[np.ndarray] = []
    for i in range(n_l):
        cl = _as_corner_matrix(corners_left[i], spec, f"corners_left[{i}]")
        cr = _as_corner_matrix(corners_right[i], spec, f"corners_right[{i}]")
        Tl = _board_pose_center(cl, K1, D1, objc)
        Tr = _board_pose_center(cr, K2, D2, objc)
        samples.append(relative_pose.compose(Tr, relative_pose.invert(Tl)))

    T_mean, agg = relative_pose.average_T(samples)
    R = T_mean[:3, :3].copy()
    T = T_mean[:3, 3].reshape(3, 1).copy()

    # 主结果：**锁定内参**的 stereoCalibrate 联合优化 R/T。
    # 逐帧 solvePnP 链在"板正对且远离"时对 z 与横/纵位置有退化，合成无噪声下
    # 也会带来约 10 mm 的基线偏差；固定内参的联合最小二乘没有这个问题，且正是
    # 平行双目的标准做法。stereoCalibrate 失败时保留上面的链式回退值。
    stereo_calib_note = ""
    try:
        obj_f32 = spec.object_points().astype(np.float32)
        img1 = [np.asarray(c, np.float32).reshape(-1, 1, 2) for c in corners_left]
        img2 = [np.asarray(c, np.float32).reshape(-1, 1, 2) for c in corners_right]
        out = cv2.stereoCalibrate(
            [obj_f32] * n_l, img1, img2,
            K1.copy(), D1.copy(), K2.copy(), D2.copy(), image_size,
            flags=cv2.CALIB_FIX_INTRINSIC)
        R = np.asarray(out[5], dtype=np.float64).reshape(3, 3)
        T = np.asarray(out[6], dtype=np.float64).reshape(3, 1)
    except Exception as exc:                               # noqa: BLE001 - 降级要留原因
        stereo_calib_note = (
            f"stereoCalibrate（锁定内参）失败（{type(exc).__name__}: {exc}），"
            f"已回退逐帧 solvePnP 链；基线精度可能下降约 10 mm，请看 consistency")

    # assume_parallel：机械保证两光轴平行 -> 强制 R=I，基线 T 保留观测值。
    # 只沿光轴推进时 R 弱可观测（~1° 不确定度），直接用它做 stereoRectify 会
    # 把极线校正歪掉；强制 I 后校正接近恒等、匹配更稳。
    if assume_parallel:
        R = np.eye(3)

    res = StereoCalibrationResult(
        image_size=image_size, board_desc=spec.describe(),
        n_pairs_in=n_l, n_pairs_used=n_l, method="axial",
        stereo_mode="parallel",
        K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T,
        rectification=("none_raw" if assume_parallel else "stereo_rectify"),
    )
    res.E, res.F = _essential_fundamental(K1, K2, R, T)
    res.n_views_left = n_l
    res.n_views_right = n_r
    res.err_left = intr_l.rms_px
    res.err_right = intr_r.rms_px
    res.rms = float(np.sqrt(np.mean([r * r for r in
                                     (intr_l.rms_px, intr_r.rms_px)
                                     if np.isfinite(r)]))) if (
        np.isfinite(intr_l.rms_px) and np.isfinite(intr_r.rms_px)) else float("nan")
    res.baseline_mm = float(np.linalg.norm(T)) * 1000.0
    res.board_area_l = _board_area_fraction(corners_left, image_size)
    res.board_area_r = _board_area_fraction(corners_right, image_size)
    res.std1 = res.std2 = None
    res.T_AB = np.eye(4)
    res.consistency = {
        "n": int(agg["n"]),
        "used": int(agg["used"]),
        "n_outliers": int(agg["n_outliers"]),
        "rot_max_deg": float(agg.get("rot_spread_deg", 0.0)),
        "trans_max_mm": float(agg.get("trans_spread_mm", 0.0)),
        "baseline_mm": res.baseline_mm,
        "used_aggregate": bool(int(agg["n"]) >= 2),
    }

    warns: list[str] = list(intr_l.notes) + list(intr_r.notes)
    if stereo_calib_note:
        warns.append(stereo_calib_note)
    if assume_parallel:
        warns.append(
            "已按 assume_parallel 强制 R=I（适用于机械保证两光轴平行的装置）；"
            "基线 T 仍来自观测")
    else:
        # 只沿光轴推进时，板的正对姿态几乎不变 -> 相对旋转 R 的可观测性很弱
        # （纯 Z 推进对绕 x/y 的旋转几乎没有约束），stereoCalibrate 会给出一个 RMS
        # 好看但带 ~1° 不确定度的 R。这是**数据本身**的限制，不是求解器 bug。
        warns.append(
            "只沿光轴推进时相对旋转 R 弱可观测（合成真值下 R 也可能差 ~1°）："
            "f/Z0/基线可靠，R 请以机械对准为准（平行装置 R≈I，可用 assume_parallel=True）")
    if n_l < 5:
        warns.append(f"只用了 {n_l} 个深度：1/p 对 ΔZ 的直线拟合精度有限（建议 ≥5）")
    baseline_bad = False
    baseline_msg = ""
    if cfg.expected_baseline_mm:
        rel = abs(res.baseline_mm - cfg.expected_baseline_mm) / cfg.expected_baseline_mm
        if rel > cfg.baseline_tolerance:
            baseline_bad = True
            baseline_msg = (
                f"基线 {res.baseline_mm:.1f} mm 与预期 {cfg.expected_baseline_mm:.1f} mm "
                f"相差 {rel * 100:.0f}%，疑似装反或内参/推进量有系统错误")
            warns.append(baseline_msg)
    if agg["n"] >= 2 and (agg["rot_spread_deg"] > 1.5 or agg["trans_spread_mm"] > 5.0):
        warns.append(
            f"逐帧外参离散度偏大：旋转最大 {agg['rot_spread_deg']:.2f}°、"
            f"平移 {agg['trans_spread_mm']:.2f} mm —— 检查板是否真的只沿光轴移动")

    # 极线校正：assume_parallel 时按 R=I 的 none_raw 口径（P2=K2·[I|T]）；
    # 否则平行装置走 stereoRectify，失败降级 none_raw（三角化仍可用）。
    if assume_parallel:
        res.R1 = res.R2 = np.eye(3)
        res.P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
        res.P2 = K2 @ np.hstack([np.eye(3), T])
        res.Q = None
        res.roi1 = res.roi2 = ()
        res.epipolar_rms = res.epipolar_max = float("nan")
    else:
        try:
            R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
                K1, D1, K2, D2, image_size, R, T, alpha=cfg.alpha)
            res.R1, res.R2, res.P1, res.P2, res.Q = R1, R2, P1, P2, Q
            res.roi1, res.roi2 = roi1, roi2
            res.epipolar_rms = res.epipolar_max = float("nan")
        except Exception as exc:                           # noqa: BLE001 - 降级要留原因
            warns.append(
                f"stereoRectify 失败（{type(exc).__name__}: {exc}），降级为 none_raw："
                f"三角化仍可用，但极线不水平、视差与深度不是简单反比")
            res.rectification = "none_raw"
            res.R1 = res.R2 = np.eye(3)
            res.P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
            res.P2 = K2 @ np.hstack([R, T])
            res.Q = None
            res.roi1 = res.roi2 = ()
            res.epipolar_rms = res.epipolar_max = float("nan")

    # 判定：轴向内参的自洽残差是关键证据；外参看多帧离散度与基线量级。
    ok = bool(np.isfinite(res.err_left) and np.isfinite(res.err_right)
              and res.err_left <= cfg.ok_rms and res.err_right <= cfg.ok_rms)
    if ok and baseline_bad:
        # 基线量级与预期不符 = 现场唯一能自动发现"装反/尺度错"的信号，强制降级
        res.ok = False
        res.grade = "不可用"
        res.verdict = ("不可用 —— " + baseline_msg +
                       "；请先用卷尺核对基线、检查内参与 ΔZ 单位，再重拍")
    elif ok and n_l >= 2 and agg["n"] >= 2:
        good = (res.err_left <= cfg.good_rms and res.err_right <= cfg.good_rms
                and agg["rot_spread_deg"] <= 0.5 and agg["trans_spread_mm"] <= 2.0)
        res.grade = "优秀" if good else "可用"
        res.ok = True
        res.verdict = ("优秀 —— 轴向内参线性拟合自洽，逐帧外参高度一致" if good
                       else "可用 —— 轴向内参已解出；逐帧外参离散度见 consistency")
    elif ok:
        res.grade = "可用"
        res.ok = True
        res.verdict = ("可用 —— 轴向内参已解出；只有 1 个深度，外参没有重复性证据"
                       "（建议至少 2 个深度）")
    else:
        res.grade = "不可用"
        res.ok = False
        res.verdict = (f"不可用 —— 轴向内参自洽残差超门限"
                       f"（左 {res.err_left:.2f} px / 右 {res.err_right:.2f} px）；"
                       f"检查板是否真的正对、ΔZ 是否与图像尺度一致")
    res.warnings = warns
    res.cv = {"skipped": "轴向模式用线性拟合，子集交叉验证无额外信息"}
    res.suggestions = [
        "这份标定假设板始终正对相机、只沿光轴推进；偏摆/倾斜会让 f 与 Z0 的"
        "线性关系失真。",
        "D 恒为 0（未标定畸变）；若镜头畸变明显，请另做常规多姿态标定取 D。",
    ]
    return res
