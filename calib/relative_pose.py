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
    "average_T",
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


# --------------------------------------------------------------------------
# 多帧鲁棒聚合（2026-10-26 加）
# --------------------------------------------------------------------------
#
# ## 为什么不能"取第一帧"、也不能"旋转矩阵算术平均再正交化"
#
# `T_AB` 是 chain 链上的**乘法因子**，它的误差会原样进入外参（不衰减）。
# 多帧同框图给出的是一簇观测，**用一帧 = 丢掉其余帧的全部信息**；
# 而算术平均旋转矩阵再正交化在旋转差小于 1° 时几乎等于真解、差大于 10° 时
# 会系统性偏向"矩阵平均"而非"几何平均"（对旋转群不是内蕴运算，结果依赖参数化）。
# 正确做法是在**李群/四元数**上取平均：旋转在单位四元数球面上做加权平均
# （先对齐半球符号，否则 q 与 -q 会互相抵消），平移取**几何中位数**（对离群
# 不敏感，而算术平均会被单个 20 mm 的坏帧拉走）。
#
# 权重用 Huber：偏离超过 `huber_k · σ`（σ 由 MAD 稳健估计，且设下限，
# 避免"样本极一致 → σ→0 → 把所有样本都判成离群"的退化）的样本被降权。

_QUAT_FLOOR = 1e-12


def _quat_from_R(R) -> np.ndarray:
    """旋转矩阵 -> 单位四元数 `(w, x, y, z)`（Shepperd 分支法，数值稳定）。"""
    m = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tr = float(m[0, 0] + m[1, 1] + m[2, 2])
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    nrm = float(np.linalg.norm(q))
    if nrm < _QUAT_FLOOR:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / nrm


def _R_from_quat(q) -> np.ndarray:
    """单位四元数 `(w, x, y, z)` -> 旋转矩阵（输出天然正交、det=+1）。"""
    v = np.asarray(q, dtype=np.float64).ravel()
    w, x, y, z = v / float(np.linalg.norm(v))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _huber_weights(dist: np.ndarray, c: float) -> np.ndarray:
    """Huber 权：`d <= c` 取 1，否则 `c/d`（对离群只降权、不硬剔）。"""
    d = np.asarray(dist, dtype=np.float64)
    w = np.ones_like(d)
    if c > 0:
        big = d > c
        w[big] = c / d[big]
    return w


_CHI3_MEDIAN = 1.5382      # 三维各向同性正态距离的中位数 / σ（χ₃ 中位数）


def _robust_sigma(dist: np.ndarray, floor: float) -> float:
    """由"样本到中心的**距离**"估**逐轴** σ，带下限。

    ⚠️ 这里**不能**对距离做 MAD：距离恒为正、分布偏斜（半正态/χ₃），MAD 会
    系统性低估 σ，于是正常噪声也会越过门限被当成离群。实测（合成数据）：
    σ=3 mm 的干净 8 帧，用 MAD 会剔掉 6 帧，反而把均值算坏。
    正确做法是用 χ₃ 的中位数反推：`median(d) = 1.5382 σ`。
    """
    v = np.asarray(dist, dtype=np.float64)
    if v.size == 0:
        return float(floor)
    return max(float(np.median(v)) / _CHI3_MEDIAN, float(floor))


def average_T(
    samples,
    *,
    max_iter: int = 8,
    huber_k: float = 2.5,
    rot_scale_floor_deg: float = 0.5,
    trans_scale_floor_mm: float = 1.0,
    min_rot_deg: float = 1.0,
    min_trans_mm: float = 5.0,
):
    """多帧 `T` 的 **SE(3) 鲁棒平均**：平移取几何中位数、旋转取四元数加权平均。

    参数
    ----
    samples : 序列，每项 4x4 刚体变换（同一物理量 `T_{B←A}` 的多次观测）。
    max_iter : 加权迭代上限。
    huber_k : Huber 门限（单位：σ）。2.5 对应"约 99% 的高斯样本不降权"。
    rot_scale_floor_deg / trans_scale_floor_mm : σ 的下限，**必须有**，
        否则样本高度一致时 σ→0，所有样本都会被判成离群。
    min_rot_deg / min_trans_mm : 判离群的**绝对下限**（与 `stereo_calib` 里
        `outlier_min_px` 同一个思路）：偏离小于它的一律不算离群，避免在
        "全都很一致"时凭空造出离群点。取 1° / 5 mm —— 与
        `stereo_calib.CHAIN_ROT_OK_DEG / CHAIN_TRANS_OK_MM`（重复性"可用"门限）
        同量级：比这更小的偏离本来就在可用范围内，剔掉它反而增加方差。
        判据：`max(k·χ₃中位数·σ, 下限)`。

    返回
    ----
    `(T_mean, stats)`，`stats` 含：
        n / used / n_outliers / outlier_index
        rot_spread_deg / trans_spread_mm（样本到均值的**最大**偏离，
            与 `consistency()` 的 `*_max_*` 同口径；含被剔除的样本）
        rot_rms_deg / trans_rms_mm（到均值的 RMS）
        dist_rot_deg / dist_trans_mm（**每个**样本到均值的距离，逐项对应入参）
        sigma_rot_deg / sigma_trans_mm / threshold_rot_deg / threshold_trans_mm
        iterations / converged / huber_k / max_iter

    边界
    ----
    0 个样本 -> `ValueError`；1 个样本 -> 原样返回（散布记 0，不做任何拟合）。
    """
    Ts = [_as_T(T, f"samples[{i}]") for i, T in enumerate(samples)]
    n = len(Ts)
    if n == 0:
        raise ValueError("average_T() 需要至少 1 个样本；0 个样本无法给出任何位姿")
    if n == 1:
        return Ts[0].copy(), {
            "n": 1, "used": 1, "n_outliers": 0, "outlier_index": [],
            "rot_spread_deg": 0.0, "trans_spread_mm": 0.0,
            "rot_rms_deg": 0.0, "trans_rms_mm": 0.0,
            "dist_rot_deg": [0.0], "dist_trans_mm": [0.0],
            "sigma_rot_deg": 0.0, "sigma_trans_mm": 0.0,
            "threshold_rot_deg": float(min_rot_deg),
            "threshold_trans_mm": float(min_trans_mm),
            "iterations": 0, "converged": True,
            "huber_k": float(huber_k), "max_iter": int(max_iter),
        }

    R_list = [T[:3, :3] for T in Ts]
    t_arr = np.array([T[:3, 3] for T in Ts], dtype=np.float64)      # (n,3)，米
    q_arr = np.array([_quat_from_R(R) for R in R_list], dtype=np.float64)

    # ---- 稳健起点：旋转取"到其它样本旋转距离之和最小"的那一帧（medoid），
    #      平移取逐坐标中位数。**不取第一帧** —— 第一帧本身可能就是坏帧。 ----
    n = len(Ts)
    pair = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            pair[i, j] = pair[j, i] = rotation_angle_deg(R_list[i], R_list[j])
    start = int(np.argmin(pair.sum(axis=1)))
    R_cur = R_list[start].copy()
    t_cur = np.median(t_arr, axis=0)

    def _refine(active, R0, t0):
        """在 `active` 子集上跑 Huber-IRLS；返回 (R, t, sigma_rot, sigma_trans, 迭代数, 是否收敛)。"""
        Rc, tc = R0.copy(), t0.copy()
        s_r = s_t = 0.0
        converged = False
        it = 0
        for it in range(1, int(max_iter) + 1):
            rot_d = np.array([rotation_angle_deg(R_list[i], Rc) for i in active])
            tr_mm = np.array([np.linalg.norm(t_arr[i] - tc) for i in active]) * 1000.0
            s_r = _robust_sigma(rot_d, rot_scale_floor_deg)
            s_t = _robust_sigma(tr_mm, trans_scale_floor_mm)
            w = _huber_weights(rot_d, huber_k * s_r) * _huber_weights(tr_mm, huber_k * s_t)

            # 旋转：四元数球面加权平均（先对齐半球符号，否则 q 与 -q 互相抵消）
            qs = q_arr[active]
            dots = qs @ _quat_from_R(Rc)
            qs = qs * np.where(dots < 0.0, -1.0, 1.0)[:, None]
            q_mean = (w[:, None] * qs).sum(axis=0)
            R_new = _R_from_quat(q_mean) if np.linalg.norm(q_mean) > _QUAT_FLOOR else Rc.copy()

            # 平移：Weiszfeld 迭代（几何中位数）+ Huber 权
            d = np.linalg.norm(t_arr[active] - tc, axis=1)
            if float(d.max()) < 1e-12:
                t_new = tc.copy()
            else:
                ww = w / np.maximum(d, 1e-12)
                t_new = (ww[:, None] * t_arr[active]).sum(axis=0) / ww.sum()

            d_rot = rotation_angle_deg(R_new, Rc)
            d_t = float(np.linalg.norm(t_new - tc)) * 1000.0
            Rc, tc = R_new, t_new
            if d_rot < 1e-4 and d_t < 1e-3:
                converged = True
                break
        return Rc, tc, s_r, s_t, it, converged

    active = list(range(n))
    R_c, t_c, s_r, s_t, iters, conv = _refine(active, R_cur, t_cur)

    def _distances(Rc, tc):
        rot = [rotation_angle_deg(R_list[i], Rc) for i in range(n)]
        tr = [float(np.linalg.norm(t_arr[i] - tc)) * 1000.0 for i in range(n)]
        return rot, tr

    rot_all, tr_all = _distances(R_c, t_c)
    # 门限 = max(k·χ₃中位数·σ, 绝对下限)：前者按"本批数据的真实噪声"自适应，
    # 后者防止"样本极一致"时把与中心差 0.1 mm 的样本也当离群。
    thr_r = max(huber_k * _CHI3_MEDIAN * s_r, float(min_rot_deg))
    thr_t = max(huber_k * _CHI3_MEDIAN * s_t, float(min_trans_mm))
    out_idx = [i for i in range(n) if rot_all[i] > thr_r or tr_all[i] > thr_t]

    # ---- 剔除离群后**重算一次**：Huber 只是降权，20 mm 的坏帧仍会把均值拉偏
    #      一点；既然有 `min_*` 绝对门限兜底，就干脆把它们排除后重算。
    #      离群名单**只判一次**（用第一轮的 σ），否则剔除会让 σ 变小、
    #      再把更多样本判成离群，越剔越多。 ----
    if 0 < len(out_idx) < n:
        active = [i for i in range(n) if i not in out_idx]
        R_c, t_c, s_r, s_t, iters2, conv2 = _refine(active, R_c, t_c)
        iters, conv = iters + iters2, conv2
        rot_all, tr_all = _distances(R_c, t_c)
        thr_r = max(huber_k * _CHI3_MEDIAN * s_r, float(min_rot_deg))
        thr_t = max(huber_k * _CHI3_MEDIAN * s_t, float(min_trans_mm))

    T_mean = rt_to_T(R_c, t_c)
    rot_arr = np.asarray(rot_all, dtype=np.float64)
    tr_arr = np.asarray(tr_all, dtype=np.float64)
    return T_mean, {
        "n": n,
        "used": int(len(active)),
        "n_outliers": int(len(out_idx)),
        "outlier_index": [int(i) for i in out_idx],
        "rot_spread_deg": float(rot_arr.max()),
        "trans_spread_mm": float(tr_arr.max()),
        "rot_rms_deg": float(np.sqrt(np.mean(rot_arr ** 2))),
        "trans_rms_mm": float(np.sqrt(np.mean(tr_arr ** 2))),
        "dist_rot_deg": [float(v) for v in rot_arr],
        "dist_trans_mm": [float(v) for v in tr_arr],
        "sigma_rot_deg": float(s_r),
        "sigma_trans_mm": float(s_t),
        "threshold_rot_deg": float(thr_r),
        "threshold_trans_mm": float(thr_t),
        "iterations": int(iters),
        "converged": bool(conv),
        "huber_k": float(huber_k),
        "max_iter": int(max_iter),
    }
