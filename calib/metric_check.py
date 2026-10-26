"""尺度/度量自检：**重投影 RMS 抓不到的那种错**。

## 为什么需要这个文件

`cv2.stereoCalibrate`（以及 chain 合成的外参）只最小化"重投影到左右图像的点
与观测角点一致"。这组约束只钉住**射线的方向**，钉不住**尺度**：把基线与三维点
整体放大 1.1 倍，重投影残差**一模一样**（图像坐标是归一化后的方向量，乘以 k
齐次坐标不变）。于是"RMS 0.2 px、外参整体偏 10%"是完全可以同时成立的 ——
而这个错会让后续所有三维重建的绝对尺寸系统性偏 10%，只看 RMS 永远发现不了。

本项目里唯一**与相机无关的长度真值**是标定板的方格边长（`BoardSpec.square_mm`）。

## 三种检查，其中只有一种是真检查

1. `board_scale_error()` —— **单目**。⚠️ **循环论证**：`solvePnP` 的 `t` 与
   `objp` 同尺度，`objp` 又是按 `spec.square_mm` 生成的，所以"量出来的方格边长"
   必然等于输入值，误差恒为 0。**它不能证明尺度对**。保留它只作为"角点/板模型
   一致"的入口（返回 `tautological=True`），不要拿它的 verdict 当验收。
2. `stereo_scale_consistency()` —— **双目，有效**。同一块板被左右两路各自看到，
   用 `T_{C_R←C_L}` 把两路的板原点搬到同一坐标系，看是否重合。外参若把平移
   缩放 k 倍，残差 ≈ `|k-1|·‖T‖` —— **随基线 ‖T‖（两相机平移）线性放大**，
   **与板到相机的距离无关**（实测板放 0.4/0.8/1.2 m，残差恒 10.00 mm）。
   （2026-10-26 更正：原文误写成"板到左相机的距离"，与实测相反。）
   所以它抓得住尺度。**这是本文件的核心。**
3. `square_size_check()` —— **双目三角化 + 已知方格边长**。把每对角点三角化到
   `left_camera` 系，量相邻角点间距，与 `spec.square_mm` 比。它同时检验尺度与
   内参/外参的自洽性（内参若整体缩放，三角化出的板会同步缩放）。

## 约定

- 单位：输入平移为**米**（与 `cv2`/本项目三维点一致），输出的 mm 数值为毫米。
- `verdict` 只返回字符串，**不抛异常**（数据不足也要能出报告）；
  只有"参数形状非法/数量对不上"这种编程错误才 `ValueError`。
- 阈值全部是关键字参数，不写死在函数体里。
"""

from __future__ import annotations

import numpy as np

from . import relative_pose

__all__ = [
    "stereo_scale_consistency",
    "square_size_check",
    "board_scale_error",
]

# 默认门限。数值来源：chain 链路的重复性门限同量级（`stereo_calib.CHAIN_*`）——
# 那里 2 mm / 0.5° 是"优秀"，5 mm / 1.5° 是"可用"。三角化口径的残差还含
# 角点定位噪声，所以 mm 门限放宽到 2 / 5 mm 之外不再另设一套标准。
_GOOD_TRANS_MM = 2.0
_OK_TRANS_MM = 5.0
_GOOD_ROT_DEG = 0.5
_OK_ROT_DEG = 1.5
_GOOD_SQUARE_PCT = 1.0
_OK_SQUARE_PCT = 3.0


def _as_corners(corners, n_expected: int, name: str) -> np.ndarray:
    """`(N,2)` / `(N,1,2)` -> `(N,2) float64`。"""
    a = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    if a.shape[0] != n_expected:
        raise ValueError(f"{name} 的角点数 {a.shape[0]} 与 objp 的 {n_expected} 不一致")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} 含 NaN / inf")
    return a


def _verdict(value: float, good: float, ok: float) -> str:
    if not np.isfinite(value):
        return "数据不足"
    if value <= good:
        return "优秀"
    if value <= ok:
        return "可用"
    return "不可用"


def _p95(x: np.ndarray) -> float:
    return float(np.percentile(x, 95)) if x.size else float("nan")


# --------------------------------------------------------------------------
# 1) 双目：外参尺度一致性（**真正能抓尺度的那一个**）
# --------------------------------------------------------------------------
def stereo_scale_consistency(
    T_CR_CL,
    poses_left,
    poses_right,
    *,
    good_trans_mm: float = _GOOD_TRANS_MM,
    ok_trans_mm: float = _OK_TRANS_MM,
    good_rot_deg: float = _GOOD_ROT_DEG,
    ok_rot_deg: float = _OK_ROT_DEG,
) -> dict:
    """用 `T_{C_R←C_L}` 把两路的板位姿搬到同一坐标系，看是否重合。

    参数
    ----
    T_CR_CL : 4x4，`p_R = T_{C_R←C_L} @ p_L`（即 `stereoCalibrate` 的 `(R, T)`）。
    poses_left : 序列，每项 `T_{C_L←B_i}`（左相机对第 i 次摆放的板的位姿）。
    poses_right : 序列，每项 `T_{C_R←B_i}`，与 `poses_left` **逐项对应**（同一摆放）。

    原理（为什么它能抓尺度）
    ------------------------
    对每次摆放 i，板原点在左相机系是 `t_L,i`，搬到右相机系应为
    `T_{C_R←C_L} · t_L,i`；而右路直接给出 `t_R,i`。两者之差

        r_i = ‖ R·t_L,i + T − t_R,i ‖

    在外参尺度错 k 倍时（`T' = k·T`）≈ `|k−1|·‖T‖`，即**与基线成正比**；
    而同样的错对重投影 RMS **完全没有影响**。这就是这个检查存在的唯一理由。

    返回 dict
    ---------
    n, translation_mm（均值）, median_mm, p95_mm, max_mm, per_pair_mm,
    rotation_deg（均值）, rotation_median_deg, rotation_p95_deg, per_pair_deg,
    verdict, rotation_verdict, worst_index
    """
    T = relative_pose._as_T(T_CR_CL, "T_CR_CL")
    pl = [relative_pose._as_T(p, f"poses_left[{i}]") for i, p in enumerate(poses_left)]
    pr = [relative_pose._as_T(p, f"poses_right[{i}]") for i, p in enumerate(poses_right)]
    if not pl or not pr:
        raise ValueError("stereo_scale_consistency() 需要至少一对板位姿")
    if len(pl) != len(pr):
        raise ValueError(
            f"左右板位姿数量不一致（{len(pl)} vs {len(pr)}）：它们必须逐项对应同一摆放")

    R, t = T[:3, :3], T[:3, 3]
    per_mm, per_deg = [], []
    for Tl, Tr in zip(pl, pr):
        origin_R = R @ Tl[:3, 3] + t           # 左路推出的板原点，换到右相机系
        per_mm.append(float(np.linalg.norm(origin_R - Tr[:3, 3])) * 1000.0)
        per_deg.append(relative_pose.rotation_angle_deg(R @ Tl[:3, :3], Tr[:3, :3]))

    m = np.asarray(per_mm, dtype=np.float64)
    d = np.asarray(per_deg, dtype=np.float64)
    med = float(np.median(m))
    # n=1 时"残差 = 0"是恒等式，不是证据（外参可能正是由这一对反向构造的）。
    # 本模块是公开 API，**不能**指望每个调用方都自己兜这个洞 —— 这里直接拒发合格证。
    if len(m) < 2:
        short = (f"数据不足（n={len(m)}）：少于 2 对时残差恒为 0，不构成尺度证据；"
                 f"至少要 2 张「板同一摆放」的参考图")
        v_trans = v_rot = short
    else:
        v_trans = _verdict(med, good_trans_mm, ok_trans_mm)
        v_rot = _verdict(float(np.median(d)), good_rot_deg, ok_rot_deg)
    return {
        "n": len(m),
        "translation_mm": float(m.mean()),
        "median_mm": med,
        "p95_mm": _p95(m),
        "max_mm": float(m.max()),
        "per_pair_mm": per_mm,
        "worst_index": int(np.argmax(m)),
        "rotation_deg": float(d.mean()),
        "rotation_median_deg": float(np.median(d)),
        "rotation_p95_deg": _p95(d),
        "max_rotation_deg": float(d.max()),
        "per_pair_deg": per_deg,
        "insufficient": len(m) < 2,
        "verdict": v_trans,
        "rotation_verdict": v_rot,
    }


# --------------------------------------------------------------------------
# 2) 双目三角化 + 已知方格边长
# --------------------------------------------------------------------------
def square_size_check(
    T_CR_CL,
    objp,
    corners_left,
    corners_right,
    spec,
    K_L,
    D_L,
    K_R,
    D_R,
    *,
    good_pct: float = _GOOD_SQUARE_PCT,
    ok_pct: float = _OK_SQUARE_PCT,
) -> dict:
    """三角化一对角点，量**相邻角点间距**，与 `spec.square_mm` 比。

    这是"已知长度真值回代"：棋盘方格边长是物理量，与外参/内参无关。
    若外参尺度错 k 倍，三角化出的板整体缩放 k 倍，间距同步变 k 倍 ——
    与 `stereo_scale_consistency()` 互补（后者查两路几何一致性，这条查绝对尺寸）。

    实现要点
    --------
    - 先按各自 `K, D` 去畸变到**针孔像素坐标**（`undistortPoints(..., P=K)`），
      再用 `P1 = K_L·[I|0]`、`P2 = K_R·[R|T]` 三角化 —— 与
      `stereo_calib` chain 分支产出的 `none_raw` 投影矩阵同一口径。
    - 相邻间距的取法**不依赖角点顺序**：对每个三角化点取"到其它点的最小距离"，
      整格棋盘上这个最小值就是方格边长（对角是 √2 倍）。最后取中位数，
      个别退化点（几乎共线/在相机后方）不会污染结果。

    返回：`n, square_mm, square_mm_expected, error_mm, error_pct, scale_ratio,
    per_point_mm, verdict`
    """
    T = relative_pose._as_T(T_CR_CL, "T_CR_CL")
    obj = np.asarray(objp, dtype=np.float64).reshape(-1, 3)
    n = obj.shape[0]
    cl = _as_corners(corners_left, n, "corners_left")
    cr = _as_corners(corners_right, n, "corners_right")

    R, t = T[:3, :3], T[:3, 3]
    P1 = np.asarray(K_L, np.float64) @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.asarray(K_R, np.float64) @ np.hstack([R, t.reshape(3, 1)])
    # 去畸变后再三角化：P1/P2 是针孔模型，不能再带畸变。
    u1 = _undistort_to_pinhole(cl, K_L, D_L)
    u2 = _undistort_to_pinhole(cr, K_R, D_R)

    X = _triangulate(P1, P2, u1, u2)
    dist = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    per_point = dist.min(axis=1) * 1000.0                 # 每个点到最近邻角点的距离（mm）
    square_mm = float(np.median(per_point))
    expected = float(spec.square_mm)
    err_pct = (square_mm / expected - 1.0) * 100.0 if expected > 0 else float("nan")
    return {
        "n": int(n),
        "square_mm": square_mm,
        "square_mm_expected": expected,
        "error_mm": square_mm - expected,
        "error_pct": float(err_pct),
        "scale_ratio": square_mm / expected if expected > 0 else float("nan"),
        "per_point_mm": per_point.tolist(),
        "verdict": _verdict(abs(float(err_pct)), good_pct, ok_pct),
    }


def _undistort_to_pinhole(pts: np.ndarray, K, D) -> np.ndarray:
    """去畸变但**保持像素尺度**（`P=K`），供针孔投影矩阵三角化用。"""
    import cv2

    p = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    out = cv2.undistortPoints(p, np.asarray(K, np.float64), np.asarray(D, np.float64),
                              P=np.asarray(K, np.float64))
    return out.reshape(-1, 2)


def _triangulate(P1, P2, u1, u2) -> np.ndarray:
    """`cv2.triangulatePoints` 的薄包装：`(N,2) -> (N,3)`，齐次归一化。"""
    import cv2

    X = cv2.triangulatePoints(np.asarray(P1, np.float64), np.asarray(P2, np.float64),
                              u1.T.reshape(2, -1), u2.T.reshape(2, -1))
    return (X[:3] / X[3]).T


# --------------------------------------------------------------------------
# 3) 单目（**循环论证，只作入口**）
# --------------------------------------------------------------------------
def board_scale_error(T_world_cam, objp, corners_list, spec, K, D) -> dict:
    """⚠️ **单目尺度自检是循环论证，不能证明尺度正确。**

    `solvePnP(objp, corners)` 解出的 `t` 与 `objp` 的尺度**同源**：`objp` 由
    `spec.square_mm` 生成，所以把 `objp` 缩放 k 倍，解出的 `t` 也缩放 k 倍，
    重投影完全不变。因此"用单目量方格边长 → 与 square_mm 比"必然自洽，
    误差恒为 0（本函数如实返回 `tautological=True`）。

    **保留它只是为了给调用方一个明确的入口**：想验尺度，请用
    `stereo_scale_consistency()`（查两路几何一致性）或 `square_size_check()`
    （双目三角化后量绝对尺寸）。

    返回：`n_views, reproj_rms_px, per_view_rms_px, tautological, verdict, note`
    """
    obj = np.asarray(objp, np.float64).reshape(-1, 3)
    n = obj.shape[0]
    rms = []
    for i, corners in enumerate(corners_list):
        c = _as_corners(corners, n, f"corners_list[{i}]")
        T_cam_board = relative_pose._as_T(
            _solve_pnp(obj, c, K, D, T_world_cam), f"T_cam_board[{i}]")
        # 用 PnP 位姿把板角点投回图像：这是**唯一**有信息量的量（角点/板模型一致性），
        # 而不是尺度。它随 PnP 解与观测的偏离变化，与 square_mm 无关。
        proj = _project(obj, T_cam_board, K, D)
        rms.append(float(np.sqrt(np.mean(np.sum((proj - c) ** 2, axis=1)))))
    return {
        "n_views": len(rms),
        "reproj_rms_px": float(np.mean(rms)) if rms else float("nan"),
        "per_view_rms_px": rms,
        "tautological": True,
        "verdict": "自检无效（单目 PnP 的平移与板尺寸同尺度，永远自洽）",
        "note": "尺度请用 stereo_scale_consistency() / square_size_check()；"
                "本函数只反映角点与板模型的一致性。",
    }


def _solve_pnp(obj: np.ndarray, corners: np.ndarray, K, D, T_hint=None) -> np.ndarray:
    """`solvePnP`，**取重投影最小**的那个解。

    `SOLVEPNP_ITERATIVE` 是局部优化：给一个错得远的初值（例如 `T_hint=I`）会收敛
    到坏解（合成数据上重投影 220 px）。所以先用 `SOLVEPNP_SQPNP`（无初值、全局）
    拿一个可靠初值，再把 `T_hint` 也当作一个候选初值精化，最后按重投影 RMS 取优。
    """
    import cv2

    Kw = np.asarray(K, np.float64)
    Dw = np.asarray(D, np.float64)
    img = corners.reshape(-1, 1, 2)
    seeds = []
    try:
        ok, rvec, tvec = cv2.solvePnP(obj, img, Kw, Dw, flags=cv2.SOLVEPNP_SQPNP)
        if ok:
            seeds.append((rvec, tvec))
    except cv2.error:
        pass
    if T_hint is not None:
        Th = relative_pose._as_T(T_hint, "T_world_cam")
        rvec, _ = cv2.Rodrigues(Th[:3, :3])
        ok, rvec, tvec = cv2.solvePnP(obj, img, Kw, Dw, rvec.copy(),
                                      Th[:3, 3].reshape(3, 1).copy(), True,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if ok:
            seeds.append((rvec, tvec))
    if not seeds:
        raise ValueError("solvePnP 失败：角点与板模型不匹配（检查行列数/角点顺序）")

    best, best_rms = None, float("inf")
    for rvec, tvec in seeds:
        ok, r2, t2 = cv2.solvePnP(obj, img, Kw, Dw, rvec.copy(), tvec.copy(), True,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        proj, _ = cv2.projectPoints(obj, r2, t2, Kw, Dw)
        rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - corners) ** 2, axis=1))))
        if rms < best_rms:
            best, best_rms = (r2, t2), rms
    if best is None:
        raise ValueError("solvePnP 精化失败：无可用解")
    R, _ = cv2.Rodrigues(best[0])
    return relative_pose.rt_to_T(R, np.asarray(best[1]).reshape(3))


def _project(obj: np.ndarray, T_cam_board: np.ndarray, K, D) -> np.ndarray:
    import cv2

    rvec, _ = cv2.Rodrigues(T_cam_board[:3, :3])
    pts, _ = cv2.projectPoints(obj, rvec, T_cam_board[:3, 3].reshape(3, 1),
                               np.asarray(K, np.float64), np.asarray(D, np.float64))
    return pts.reshape(-1, 2)
