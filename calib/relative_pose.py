"""刚体位姿的链式运算（纯 numpy，不依赖 OpenCV / 不读文件 / 不打印）。

## 方向约定（**全文唯一**）

    T_x_y  表示"把 **y 系**里的点变到 **x 系**"：  p_x = T_x_y @ p_y

由此得到两条推论，代码里处处用到：

    T_x_y @ T_y_z = T_x_z          （`compose(T_x_y, T_y_z)`）
    inv(T_x_y)     = T_y_x          （`invert()`，刚体逆 R^T / -R^T t）

## 90° 装置为什么需要它

`cv2.stereoCalibrate` 的前提是"同一块板**同时**出现在两路"，而 90° 装置两路
几乎不共视 —— 这条路在数学上不成立。正确做法是把外参拆成一条**不需要共视**的链：

    T_{C_R ← C_L} = T_{C_R ← B} · T_{B ← A} · T_{A ← C_L}

- `T_{C_L←A} = T_camL_A`：左相机对 A 板的位姿（单目 `solvePnP` 得到的**正向**位姿，
  即 `p_camL = T_camL_A @ p_A`）；
- `T_{C_R←B} = T_camR_B`：右相机对 B 板的位姿（同样是 `solvePnP` 的**正向**位姿，
  **不取逆**）；
- `T_{B←A} = T_AB`：A、B 两块板**刚性固定**在同一支架上的相对位姿，
  由**任一单相机同时拍到两板**的一次测量求得（`board_to_board_from_single_camera`）。

注意链上真正出现的是 `T_{A←C_L} = inv(T_camL_A)`（**逆**）与 `T_{C_R←B}`
（**正向**）—— 两段方向相反，这是全篇最容易写反的一处。逐段推导：

    P_R = T_{C_R←B} · P_B                    # 右相机 ← B 板：solvePnP 正向位姿
        = T_camR_B · T_{B←A} · P_A           # 经 T_AB 换到 A 系
        = T_camR_B · T_AB · T_{A←C_L} · P_L  # 换到左相机系：T_camL_A 取逆
        = T_camR_B · T_AB · inv(T_camL_A) · P_L

即 `T_{C_R←C_L} = T_camR_B @ T_AB @ inv(T_camL_A)`，与 `relative_extrinsics()`
第 153 行的 `compose(T_camR_B, T_AB, invert(T_camL_A))` 完全一致。

整条链只要求"每路各自能看到自己那块板"+"有一块刚性双板靶标"，
**没有任何一步要求两相机同时看到同一物体**。

⚠️ `T_AB` 这个参数名沿用计划书，语义是 **T_{B←A}（A 系点 → B 系点）**，
不是 "A 到 B 的逆"。`relative_extrinsics()` 的 docstring 里有自检说明。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "invert",
    "compose",
    "board_to_board_from_single_camera",
    "relative_extrinsics",
    "consistency",
    "rt_to_T",
    "T_to_rt",
    "rotation_angle_deg",
]

_ORTHO_ATOL = 1e-6
_ORTHO_RTOL = 1e-4


def _as_T(T, name: str = "T") -> np.ndarray:
    """校验并复制一个 4x4 刚体变换；非法就 `ValueError` 并说明原因。"""
    arr = np.asarray(T, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"{name} 必须是 4x4 齐次矩阵，实际形状 {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} 含 NaN / inf，拒绝参与链式运算")
    if not np.allclose(arr[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        raise ValueError(f"{name} 最后一行必须是 [0, 0, 0, 1]，实际 {arr[3]}")
    R = arr[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=_ORTHO_ATOL, rtol=_ORTHO_RTOL):
        raise ValueError(f"{name} 的旋转块不正交（R^T R != I），不是刚体变换")
    det = float(np.linalg.det(R))
    if abs(det - 1.0) > 1e-4:
        raise ValueError(f"{name} 的旋转块 det={det:.6f} != +1（含镜像/缩放）")
    return arr


def rt_to_T(R, t) -> np.ndarray:
    """`(R, t)` -> 4x4；采用 OpenCV 约定 `X' = R·X + t`。"""
    Rm = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tv = np.asarray(t, dtype=np.float64).reshape(3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rm
    T[:3, 3] = tv
    return T


def T_to_rt(T) -> tuple[np.ndarray, np.ndarray]:
    """4x4 -> `(R, T)`，且 `T` 形状为 `(3, 1)`（与 `cv2.stereoCalibrate` 一致）。"""
    A = _as_T(T)
    return A[:3, :3].copy(), A[:3, 3].reshape(3, 1).copy()


def invert(T) -> np.ndarray:
    """刚体逆：`inv([R|t]) = [R^T | -R^T t]`（不调用通用矩阵求逆）。"""
    A = _as_T(T)
    R = A[:3, :3]
    t = A[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def compose(*Ts) -> np.ndarray:
    """链式相乘：`compose(T_a_b, T_b_c, ...)` -> `T_a_b @ T_b_c @ ...`。

    顺序就是**从右往左作用**（与矩阵乘法一致），不额外做任何翻转。
    """
    if not Ts:
        raise ValueError("compose() 至少需要一个 4x4 变换")
    out = _as_T(Ts[0], "Ts[0]")
    for i, T in enumerate(Ts[1:], start=1):
        out = out @ _as_T(T, f"Ts[{i}]")
    return out


def board_to_board_from_single_camera(T_cam_A, T_cam_B) -> np.ndarray:
    """由**一台相机同时看到 A、B 两板**的一次测量，求两板的刚性相对位姿。

    参数
    ----
    T_cam_A, T_cam_B : 4x4
        相机分别对 A 板、B 板的位姿，即 `p_cam = T_cam_X @ p_X`（`solvePnP` 的
        `(R, t)` 直接转成 4x4 即可）。

    返回
    ----
    T_{B←A} : 4x4，`p_B = T_{B←A} @ p_A`。

    推导：`p_B = T_{B←cam} p_cam = inv(T_cam_B) · T_cam_A · p_A`。
    """
    return compose(invert(T_cam_B), T_cam_A)


def relative_extrinsics(
    T_camL_A, T_AB, T_camR_B
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按"公共参考链"求左右相机外参（**不需要双目共视**）。

    参数（全部 4x4）
    ----------------
    T_camL_A : 左相机对 A 板的位姿，`p_camL = T_camL_A @ p_A`（`solvePnP` 的 (R,t)）
    T_AB     : **T_{B←A}**（A 系 -> B 系），由 `board_to_board_from_single_camera` 得到
    T_camR_B : 右相机对 B 板的位姿，`p_camR = T_camR_B @ p_B`

    返回
    ----
    `(R, T, T_camR_camL)`：
        `T_camR_camL = T_camR_B @ T_AB @ inv(T_camL_A)`
        且 `R, T` 满足 OpenCV 约定 **`X_right = R · X_left + T`**，`T` 形状 `(3, 1)`。

    推导（注意箭头方向，这是最容易写反的一步）：

        P_R = T_{C_R←B} · P_B          # 右相机 ← B 板：就是 solvePnP 的 T_camR_B
            = T_camR_B · T_{B←A} · P_A
            = T_camR_B · T_AB · T_{A←C_L} · P_L
            = T_camR_B · T_AB · inv(T_camL_A) · P_L

    即 `T_{C_R←B}` 是 **solvePnP 的正向位姿**（不是逆），
    而 `T_{A←C_L}` 是 `T_camL_A` 的**逆**（箭头方向相反）。
    """
    chain = compose(T_camR_B, T_AB, invert(T_camL_A))
    R, T = T_to_rt(chain)

    # ---- 方向自检：把同一个点走两条路，结果必须一致 ----
    # 路 1：直接 X_right = R·X_left + T
    # 路 2：C_L -> A（取逆）-> B -> C_R（各段用入参矩阵）
    p_left = np.array([0.11, -0.07, 0.62, 1.0])
    via_chain = compose(T_camR_B, T_AB, invert(T_camL_A)) @ p_left
    direct = np.append(R @ p_left[:3] + T.ravel(), 1.0)
    if not np.allclose(via_chain, direct, atol=1e-12):
        raise AssertionError(
            "relative_extrinsics 方向自检失败：链式结果与 (R,T) 不一致 —— "
            "说明 R/T 的提取或链式顺序错了"
        )
    return R, T, chain


def rotation_angle_deg(R_a, R_b) -> float:
    """两个旋转之间的测地角（度）。"""
    Ra = np.asarray(R_a, dtype=np.float64).reshape(3, 3)
    Rb = np.asarray(R_b, dtype=np.float64).reshape(3, 3)
    c = (np.trace(Ra @ Rb.T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def consistency(T_AB_list) -> dict:
    """多次测得的 `T_{B←A}` 的重复性（**这是 chain 唯一能自查的证据**）。

    返回 dict：
        n              测量次数
        rot_max_deg    两两旋转测地角的最大值（度）
        rot_mean_deg   两两旋转测地角的均值（度）
        trans_max_mm   两两平移差的最大值（mm）
        trans_mean_mm  两两平移差的均值（mm）
        baseline_mm    各次平移长度的均值（mm，供与卷尺值对照）

    单次测量时离散度记为 0 —— 调用方**必须**同时看 `n`，
    否则会把"没测过重复性"误读成"完全一致"。
    """
    Ts = [_as_T(T, f"T_AB_list[{i}]") for i, T in enumerate(T_AB_list)]
    if not Ts:
        raise ValueError("consistency() 需要至少一次 T_AB 测量")
    n = len(Ts)
    rots, trans = [], []
    for i in range(n):
        for j in range(i + 1, n):
            rots.append(rotation_angle_deg(Ts[i][:3, :3], Ts[j][:3, :3]))
            trans.append(float(np.linalg.norm(Ts[i][:3, 3] - Ts[j][:3, 3])) * 1000.0)
    rots_arr = np.asarray(rots, dtype=np.float64)
    trans_arr = np.asarray(trans, dtype=np.float64)
    return {
        "n": n,
        "rot_max_deg": float(rots_arr.max()) if rots_arr.size else 0.0,
        "rot_mean_deg": float(rots_arr.mean()) if rots_arr.size else 0.0,
        "trans_max_mm": float(trans_arr.max()) if trans_arr.size else 0.0,
        "trans_mean_mm": float(trans_arr.mean()) if trans_arr.size else 0.0,
        "baseline_mm": float(np.mean([np.linalg.norm(T[:3, 3]) for T in Ts])) * 1000.0,
    }
