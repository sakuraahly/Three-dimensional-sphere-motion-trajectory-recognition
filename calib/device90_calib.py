"""模式 B：中轴 90° 装置约束标定（同一块板绕自身中轴转 90°，分别正对互成 90° 的两相机）。

## 装置模型（用户已确认）

- 旋转轴 = 板的一条**中轴**（过板几何中心、平行板系 x 或 y 轴），不是板边。
- 板 A 绕该中轴转 90° 得到板 B → 两板**几何中心重合**。忽略板厚时，在
  **以板中心为原点的板坐标系**下

      T_AB = [R_AB | 0]

  是**纯旋转**，不需要板间距/共边长度作为输入（这是相对"绕板边"模型的关键差异）。
- `R_AB` 是绕板系中轴 `±90°` 的旋转；`axis`（"x"/"y"）与 `angle_deg` 由装置
  安装决定，本模块以参数传入（默认 `axis="y", angle_deg=90`），共 4 个候选。
- 两相机光轴互成 90° 是装置保证，它决定旋转轴方向 = `normalize(n_A × n_B)`，
  从而决定 `R_AB` 走哪个候选；它**不直接进入**下面的链式公式。

## 关键约定：板中心原点

`BoardSpec.object_points()` 的原点是**角点**（0,0,0），不是板中心。若直接用角点
原点，`T_AB` 的平移为 `(I − R_AB)·c ≠ 0`（c 为板中心，约半块板宽）。本模块内部
**统一把物体点平移到板中心**（`objc = objp − c_board`），于是 `T_AB` 是纯旋转，
`T_camA_A` / `T_camB_B` 的平移就是**板中心在两相机系的坐标** `QA` / `QB`——这正是
用户所说的 `d1` / `d2`（板中心深度）。最终 `T_{C_B←C_A}` 与原点选择无关。

## 链式解（已在合成数据上严格自验）

    T_{C_B←C_A} = T_camB_B · T_AB · inv(T_camA_A)

其中 `T_camA_A`、`T_camB_B` 是两板各自 `solvePnP`（板中心原点）的正向位姿。展开
平移（`T_camX_X = [R_X | Q_X]`，`T_AB = [R_AB | 0]`）：

    t = Q_B − (R_B · R_AB · R_A^T) · Q_A

⚠️ **不能**把 `R_B·R_AB·R_A^T` 简化成 `R_AB`：只有在 `R_A = R_B = I`（无 roll）时
才相等。自验：roll=0/180° 时简化式与严格式都精确；roll=7° 时简化式差 0.62 mm、
roll=20° 时差 10.27 mm。装置要求"板正对各自相机"，roll 实际应接近 0/180°，但
本模块**始终用严格式**，并对 roll 偏离给出诊断。

## d1 / d2 的角色

`d1`、`d2` 就是 `Z_A`、`Z_B`（板中心在两相机系的深度）。它们

- **不进入 `T_AB`**（纯旋转）；
- 只经 `t` 的平移进入相机相对位姿：`Q_A`、`Q_B` 的 z 分量被 `Z_A`、`Z_B` 覆盖。

不给 `Z_A/Z_B` 时用 `solvePnP` 反解的 z；给了则**强制覆盖**（自验：`Z_B` 加
50 mm，`t` 的对应分量正好加 50 mm——证明它真的进入结果、不是硬编码）。

## 诚实边界

- 依赖**内参 `K/D` 已知**（来自模式 A 或既有标定）。
- 若实际旋转轴走的是另一个候选（x vs y，或符号反了），链式旋转会明显偏离
  "两光轴 ≈90°"的物理事实——`roll_diagnostic` 与 `baseline` 会暴露，但本模块
  **不会**自动猜轴：静默猜错轴比明确失败更贵。
- 不改现有求解器任何分支/门限，不注册 `calib_method`（GUI/CLI 接入留给后续 worker）。
"""

from __future__ import annotations

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
    "Device90CalibError",
    "Device90Diagnostics",
    "board_center",
    "R_AB_matrix",
    "solve_device90",
]

_AXES = {"x": np.array([1.0, 0.0, 0.0]),
         "y": np.array([0.0, 1.0, 0.0])}


class Device90CalibError(ValueError):
    """90° 中轴装置的输入/前提不成立。"""


def board_center(spec: BoardSpec | None = None) -> np.ndarray:
    """板几何中心在 BoardSpec 坐标系下的坐标（米）。"""
    spec = spec or BoardSpec()
    op = spec.object_points()
    return (op.min(axis=0) + op.max(axis=0)) / 2.0


def R_AB_matrix(axis: str = "y", angle_deg: float = 90.0) -> np.ndarray:
    """板 A 系 → 板 B 系的纯旋转：绕**板系中轴** `axis` 转 `angle_deg`。

    这是"绕板中心的中轴旋转"的旋转部分；平移在中心原点下为 0，所以整个
    `T_AB = [R_AB | 0]`。`angle_deg` 保留为浮点，便于自验里把 90° 改成 80°
    观察结果变化（生产默认 90°）。
    """
    key = str(axis).lower()
    if key not in _AXES:
        raise Device90CalibError(f"axis 只能是 'x' 或 'y'，实际 {axis!r}")
    if not np.isfinite(angle_deg):
        raise Device90CalibError(f"angle_deg 非法：{angle_deg}")
    rvec = np.radians(float(angle_deg)) * _AXES[key]
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    return R


def _center_object_points(spec: BoardSpec) -> tuple[np.ndarray, np.ndarray]:
    op = spec.object_points()
    c = (op.min(axis=0) + op.max(axis=0)) / 2.0
    return (op - c).astype(np.float64), c


def _as_corners(corners, spec: BoardSpec, tag: str) -> np.ndarray:
    pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] != spec.n_points:
        raise Device90CalibError(
            f"{tag}: 角点数 {pts.shape[0]} 与板规格 {spec.n_points} 不一致")
    if not np.all(np.isfinite(pts)):
        raise Device90CalibError(f"{tag}: 角点含 NaN/inf")
    return pts


def _pnp_center(pts: np.ndarray, K: np.ndarray, D: np.ndarray,
                objc: np.ndarray) -> tuple[np.ndarray, float]:
    ok, rvec, tvec = cv2.solvePnP(
        objc.astype(np.float64), pts.reshape(-1, 1, 2), K, D,
        flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise Device90CalibError(
            "solvePnP 失败：角点与板模型不匹配（检查行列数/角点顺序/BoardSpec）")
    R, _ = cv2.Rodrigues(rvec)
    T = relative_pose.rt_to_T(R, np.asarray(tvec, np.float64).ravel())
    proj, _ = cv2.projectPoints(objc, rvec, tvec, K, D)
    rms = float(np.sqrt(np.mean(np.sum(
        (proj.reshape(-1, 2) - pts) ** 2, axis=1))))
    return T, rms


def _roll_deg_y(R: np.ndarray) -> float:
    """板 y 轴相对相机 y 轴的滚转角（度），用于诊断 roll 是否接近 0/180。

    板正对相机时板法向（板系 z）≈ 相机 z。板 y 轴在相机 xy 平面的投影与相机
    y 轴的夹角即 roll。范围 (-90, 90]。
    """
    y_axis = R[:, 1]
    ang = np.degrees(np.arctan2(y_axis[0], y_axis[1]))
    # 折到 (-90, 90]
    while ang > 90.0:
        ang -= 180.0
    while ang <= -90.0:
        ang += 180.0
    return float(ang)


@dataclass
class Device90Diagnostics:
    """给报告/排障用的中间量（不进 StereoCalibrationResult 也能复现）。"""

    Z_A: float = float("nan")            # 板 A 中心在相机 A 系的深度
    Z_B: float = float("nan")
    Q_A: np.ndarray | None = None        # 板 A 中心在相机 A 系坐标
    Q_B: np.ndarray | None = None
    roll_A_deg: float = float("nan")
    roll_B_deg: float = float("nan")
    rms_A_px: float = float("nan")
    rms_B_px: float = float("nan")
    n_A: int = 0
    n_B: int = 0
    notes: list[str] = field(default_factory=list)


def solve_device90(
    corners_A, corners_B, K1, D1, K2, D2,
    spec: BoardSpec | None = None,
    image_size: tuple[int, int] = (0, 0),
    *,
    axis: str = "y",
    angle_deg: float = 90.0,
    Z_A: float | None = None,
    Z_B: float | None = None,
    cfg: CalibConfig | None = None,
    diagnostics: Device90Diagnostics | None = None,
) -> StereoCalibrationResult:
    """中轴 90° 装置的外参求解（不需要两路共视，也不需要同框图）。

    Parameters
    ----------
    corners_A, corners_B : 每项是一帧角点（`(N,2)`/`(N,1,2)`）。`_A` 是相机 A
        拍到正对 A 的板 A；`_B` 是相机 B 拍到正对 B 的板 B（板 A 绕中轴转 90°）。
        多帧时用 `average_T` 聚合（要求每帧描述**同一物理摆放**）。
    K1,D1 / K2,D2 : 两路内参（**必须已知**）。
    spec : 板规格。
    axis, angle_deg : 装置中轴与旋转角（默认 y / 90°）。
    Z_A, Z_B : 可选的板中心深度（米）。给出则覆盖 solvePnP 的 z 分量；None 用
        solvePnP 反解。它们**只影响平移**，不影响 `T_AB`（纯旋转）。
    diagnostics : 传入一个 `Device90Diagnostics` 实例可拿到中间量（原地填充）。

    Returns
    -------
    StereoCalibrationResult，`rectification="none_raw"`、`stereo_mode="angled"`，
    `P1=K1[I|0]`、`P2=K2[R|T]`，可直接喂
    `stereo/angled_geometry.py::triangulate_pixels`。

    装反检测的灵敏度边界：合成装置实测，axis 装错只让基线模长偏约 7.8%，
    angle 反向偏约 16.9%，因此调用方必须：
      (a) 传 `expected_baseline_mm`（机械尺寸，强烈建议卷尺实测）；
      (b) 把 `cfg.baseline_tolerance` 设到 0.05 量级；
    默认 0.20 会漏掉这两种装错。
    """
    spec = spec or BoardSpec()
    cfg = cfg or CalibConfig()
    diag = diagnostics if diagnostics is not None else Device90Diagnostics()

    list_a = list(corners_A) if isinstance(corners_A, (list, tuple)) else [corners_A]
    list_b = list(corners_B) if isinstance(corners_B, (list, tuple)) else [corners_B]
    if not list_a or not list_b:
        raise Device90CalibError("至少需要各一张 A/B 板观测")

    objc, c_board = _center_object_points(spec)
    R_AB = R_AB_matrix(axis, angle_deg)
    T_AB = relative_pose.rt_to_T(R_AB, np.zeros(3))

    # Z_A/Z_B 只允许单帧覆盖（多帧时每帧深度可能不同，标量含义不明）
    if (Z_A is not None or Z_B is not None) and (len(list_a) != 1 or len(list_b) != 1):
        raise Device90CalibError(
            "Z_A/Z_B 是标量深度，只在 A、B 各 1 帧时可用；多帧请分别用列表或设 None")

    K1 = np.asarray(K1, dtype=np.float64).reshape(3, 3)
    K2 = np.asarray(K2, dtype=np.float64).reshape(3, 3)
    D1 = np.asarray(D1, dtype=np.float64).ravel() if D1 is not None else np.zeros(5)
    D2 = np.asarray(D2, dtype=np.float64).ravel() if D2 is not None else np.zeros(5)

    ta_list, tb_list = [], []
    rms_a, rms_b = [], []
    for i, ca in enumerate(list_a):
        pts = _as_corners(ca, spec, f"corners_A[{i}]")
        Ta, r = _pnp_center(pts, K1, D1, objc)
        ta_list.append(Ta); rms_a.append(r)
    for j, cb in enumerate(list_b):
        pts = _as_corners(cb, spec, f"corners_B[{j}]")
        Tb, r = _pnp_center(pts, K2, D2, objc)
        tb_list.append(Tb); rms_b.append(r)

    # 单帧：可选 Z 覆盖（Z 是板中心深度 = t 的 z 分量；板在相机前方 z>0）
    if Z_A is not None:
        if not (np.isfinite(Z_A) and Z_A > 0):
            raise Device90CalibError(f"Z_A={Z_A} 必须是正深度（米）")
        ta_list[0] = ta_list[0].copy()
        ta_list[0][2, 3] = float(Z_A)
    if Z_B is not None:
        if not (np.isfinite(Z_B) and Z_B > 0):
            raise Device90CalibError(f"Z_B={Z_B} 必须是正深度（米）")
        tb_list[0] = tb_list[0].copy()
        tb_list[0][2, 3] = float(Z_B)

    # ---- 逐帧链 + 聚合 ----
    samples: list[np.ndarray] = []
    for Ta in ta_list:
        for Tb in tb_list:
            samples.append(relative_pose.compose(Tb, T_AB, relative_pose.invert(Ta)))
    T_mean, agg = relative_pose.average_T(samples)
    R = T_mean[:3, :3].copy()
    T = T_mean[:3, 3].reshape(3, 1).copy()

    # ---- 诊断 ----
    Ta0, Tb0 = ta_list[0], tb_list[0]
    diag.Q_A = Ta0[:3, 3].copy()
    diag.Q_B = Tb0[:3, 3].copy()
    diag.Z_A = float(Ta0[2, 3])
    diag.Z_B = float(Tb0[2, 3])
    diag.roll_A_deg = _roll_deg_y(Ta0[:3, :3])
    diag.roll_B_deg = _roll_deg_y(Tb0[:3, :3])
    diag.rms_A_px = float(np.mean(rms_a))
    diag.rms_B_px = float(np.mean(rms_b))
    diag.n_A, diag.n_B = len(ta_list), len(tb_list)

    res = StereoCalibrationResult(
        image_size=image_size, board_desc=spec.describe(),
        n_pairs_in=len(samples), n_pairs_used=len(samples), method="device90",
        stereo_mode="angled",
        K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T,
        rectification="none_raw",
    )
    res.E, res.F = _essential_fundamental(K1, K2, R, T)
    res.R1 = res.R2 = np.eye(3)
    res.P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    res.P2 = K2 @ np.hstack([R, T])
    res.Q = None
    res.roi1 = res.roi2 = ()
    res.epipolar_rms = res.epipolar_max = float("nan")
    res.rms = float("nan")
    res.err_left = diag.rms_A_px
    res.err_right = diag.rms_B_px
    res.baseline_mm = float(np.linalg.norm(T)) * 1000.0
    res.board_area_l = _board_area_fraction(
        [np.asarray(c).reshape(-1, 2) for c in list_a], image_size)
    res.board_area_r = _board_area_fraction(
        [np.asarray(c).reshape(-1, 2) for c in list_b], image_size)
    res.std1 = res.std2 = None
    res.T_AB = T_AB.copy()
    res.n_views_left, res.n_views_right = len(ta_list), len(tb_list)
    res.consistency = {
        "n": int(agg["n"]),
        "used": int(agg["used"]),
        "n_outliers": int(agg["n_outliers"]),
        "rot_max_deg": float(agg.get("rot_spread_deg", 0.0)),
        "trans_max_mm": float(agg.get("trans_spread_mm", 0.0)),
        "baseline_mm": res.baseline_mm,
        "used_aggregate": bool(int(agg["n"]) >= 2),
    }

    warns: list[str] = []
    if not (np.isfinite(diag.rms_A_px) and np.isfinite(diag.rms_B_px)
            and diag.rms_A_px <= cfg.ok_rms and diag.rms_B_px <= cfg.ok_rms):
        warns.append(
            f"两板 solvePnP 残差偏大（A {diag.rms_A_px:.2f} px / B {diag.rms_B_px:.2f} px）"
            f"—— 内参不准、板没正对，或角点顺序错")
    # roll 诊断：装置要求板正对各自相机，roll 应接近 0°
    for tag, roll in (("A", diag.roll_A_deg), ("B", diag.roll_B_deg)):
        # _roll_deg_y 已把角度折到 (-90, 90]，180° 与 0° 等价，无需再单独判
        deg_off = abs(roll)
        if np.isfinite(roll) and deg_off > 5.0:
            warns.append(
                f"板 {tag} 的滚转 {roll:.1f}° 明显偏离 0°：板可能没有严格正对"
                f"相机；严格链仍成立，但若改用解析 Z 反解会失准")
    baseline_bad = False
    baseline_msg = ""
    if cfg.expected_baseline_mm:
        rel = abs(res.baseline_mm - cfg.expected_baseline_mm) / cfg.expected_baseline_mm
        if rel > cfg.baseline_tolerance:
            baseline_bad = True
            baseline_msg = (
                f"基线 {res.baseline_mm:.1f} mm 与预期 {cfg.expected_baseline_mm:.1f} mm "
                f"相差 {rel * 100:.0f}%，疑似 axis/angle 与装置不符或 A/B 接反")
            warns.append(baseline_msg)

    ok = bool(np.isfinite(diag.rms_A_px) and np.isfinite(diag.rms_B_px)
              and diag.rms_A_px <= cfg.ok_rms and diag.rms_B_px <= cfg.ok_rms
              and np.all(np.isfinite(R)) and np.all(np.isfinite(T)))
    if ok and baseline_bad:
        # 板正对时角点残差照样很小，装错 axis/angle 只会让基线的**量级**错。
        # expected_baseline_mm 是现场唯一能自动发现这件事的信号 -> 强制降级。
        res.ok = False
        res.grade = "不可用"
        res.verdict = ("不可用 —— " + baseline_msg +
                       "；请先用卷尺核对基线再重拍（板正对时残差看不出这个错）")
    elif ok:
        good = (diag.rms_A_px <= cfg.good_rms and diag.rms_B_px <= cfg.good_rms)
        res.grade = "优秀" if good else "可用"
        res.ok = True
        res.verdict = ("优秀 —— 中轴 90° 链式外参，两板拟合自洽" if good
                       else "可用 —— 中轴 90° 链式外参已解出；两板残差只到「可用」档")
    else:
        res.grade = "不可用"
        res.ok = False
        res.verdict = ("不可用 —— 中轴 90° 链式外参不可信：两板 solvePnP 残差超门限或"
                       "结果含非有限值；先查内参/角点顺序/板是否正对")
    res.warnings = warns
    res.cv = {"skipped": "device90 不需要共视，无 joint 式交叉验证"}
    res.suggestions = [
        "P1/P2 是 none_raw（未做极线校正）；三角测量可直接用 "
        "stereo/angled_geometry.py::triangulate_pixels，但依赖极线水平/视差的方法不适用。",
        "axis/angle 必须与装置实际中轴/转向一致；装错会给出基线量级明显不对的'好看'解，"
        "请用卷尺核对基线。",
    ]
    diag.notes = warns
    return res
