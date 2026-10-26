"""外参不确定度：`R, T` 到底能信到几位（数值雅可比 + delta 法）。

## 为什么需要这个文件

`calib/stereo_calib.py:depth_sigma_mm()` 的 docstring 自己写着：

    **不包含 σ_B**：cv2.stereoCalibrate 不返回 T 的方差

于是报告里的"0.60 m 处约 ±X mm"少了一项 —— 而这一项在**短基线近距**装置上
（本项目基线 65~200 mm、工作距离 0.4~1.0 m）恰恰是主导项之一：
`σ_Z` 里基线项的系数是 `Z/B`，65 mm 基线、1 m 处就是 15 倍放大。

`stereoCalibrate` 不返回 T 的协方差，但**能算**：把它当成非线性最小二乘，
在解处取残差对 6 个参数（3 旋转 + 3 平移）的雅可比，就有

    Σ_θ = σ² · (JᵀJ)⁻¹,      σ² = SSR / (m - 6)

`m` = 残差维数（2 × 角点数 × 视图数）。这是标准的非线性最小二乘协方差估计
（Hartley & Zisserman, *Multiple View Geometry*, 2nd ed., §5.3/§A.4；
Gauss-Newton 在解处的近似）。它给出的是"**其它参数都固定时** (R,T) 的方差"——
条件协方差，比联合估计（同时放开内参与各视图位姿）的方差**大**（因为把位姿误差算进去了）。
但**实测的净结果是偏低，不是偏高**：2026-10-26 用 30 次噪声实现验证，
`σ_B` 预测 / 实测散布 ≈ **0.74**（0.5 px、12 视、54 点）；基线 60/200 mm、1 px、3 视时 **0.54–0.62**；
2 m 远场（板距 1.2–2.0 m）进一步降到 **0.41** —— 远处板位姿与 `T` 强相关，条件协方差吃不到这部分。
所以对外只有一个结论：**这个 σ 是"量级估计"，不要当置信区间用**；
标定距离附近可信，远场会低估。

## 重投影残差怎么写

外参的定义（OpenCV 口径）：`x_R = R · x_L + T`，`T` 与三维点同单位（本项目 = 米）。

第 i 个视图的板位姿 `(R_i, t_i)`（板 → 左相机）由 `solvePnP(objp, 左图角点)` 给出、
**在优化中保持固定**（因为我们要的是 (R,T) 的条件不确定度，不是重标一次）。
那么右相机的位姿是 `(R·R_i, R·t_i + T)`，残差取**右目**重投影：

    r_i = project(R·R_i, R·t_i + T; K2, D2) − p2_i

**为什么不用左目残差**：左目投影完全不含 (R,T)，其对 6 个参数的雅可比列恒为 0。
把它塞进 SSR 只会让 σ² 变大（多算了不解释参数的噪声），却对 JᵀJ 毫无贡献 ——
纯粹是有偏的。所以 `m = 2·N·n_views`，只统计右目。

## 数值雅可比

步长按参数尺度取：旋转 1e-6 rad、平移 1e-6 m（中心差分）。这两个尺度下
残差变化约 `f·h/Z` ~ 1e-3 px 量级，而残差本身 ~1e2 px —— float64 的相对精度
1e-16 让"大数相减得小数"仍然有 10 个有效位，不会掉进抵消误差。

## σ_B（基线）与 σ_angle

- `σ_B = √(uᵀ Σ_t u)`，`u = T/‖T‖`：`‖T‖` 的 delta 法一阶展开
  （用完整 `Σ_t`，不是只取对角 —— `T` 的三个分量在短基线装置上强相关）。
- `σ_angle = √(tr Σ_r)`（弧度 → 度）：旋转扰动向量 `w ~ N(0, Σ_r)`，
  `E‖w‖² = tr Σ_r`，所以 `√(tr)` 是"旋转偏差向量的均方根模长"（1σ 量级）。
  比"模长的标准差"略保守，量级一致 —— 这里要的是量级。

## 深度公式（自己推，写在这里）

从 `Z = f·B/d`（`d` = 视差）出发求偏导：

    ∂Z/∂d = −f·B/d²  = −Z/d
    ∂Z/∂B =  f/d     = Z/B
    ∂Z/∂f =  B/d     = Z/f

忽略各项相关性、按一阶误差传播：

    σ_Z² = (Z/d)²·σ_d² + (Z/B)²·σ_B² (+ (Z/f)²·σ_f²)
    σ_Z  = Z·√( (σ_d/d)² + (σ_B/B)² )        ← 本函数实现的这一支

- `σ_d ≈ √2 · noise_px`：左右两路各贡献一次独立角点定位误差；
- `σ_B`：**本函数算出来的那一项**，正是 `stereo_calib.depth_sigma_mm()` 缺的；
- `σ_f`：本函数**不含**（它需要 `calibrateCameraExtended` 的
  `stdDeviationsIntrinsics`，属于内参不确定度的范畴，见
  `StereoCalibrationResult.fx_sigma()`）。两项都想要时把平方相加。

⚠️ 这是**数量级估计**：同一批角点既拟合又估噪声，`σ²` 会略偏小；
"其它参数固定"的条件假设让 `σ_t` 略偏大。两个偏差方向相反，实测比值应落在 1 附近
（`tests/selftest_calib_facts.py` 用它钉住 0.5~2.0 这个区间）。

用法::

    from calib.uncertainty import extrinsic_sigma

    u = extrinsic_sigma(objp, ip1, ip2, K1, D1, K2, D2, R, T, image_size)
    if "error" in u:
        raise RuntimeError(u["error"])          # 奇异：**不许**当 inf/nan 混过去
    print(u["sigma_baseline_mm"], u["depth_sigma_mm_at"](0.6))
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np

# 中心差分步长（按参数尺度取）：旋转 rad、平移 m
STEP_ROT = 1e-6
STEP_TRANS = 1e-6

# 判定"雅可比奇异"的列归一化条件数门限。
#
# 为什么先归一化列再算条件数：旋转列的量纲是 px/rad、平移列是 px/m，两者
# 天然差 1e5 量级 —— 不归一化时条件数永远很大，门限无从设起。
# 1e12 是"双精度下 6x6 正规方程已经解不出可信数字"的经验线。
COND_MAX = 1e12


def _obj_list(objp, n: int) -> list[np.ndarray]:
    """把物体点规整成"每视图一个 (N,3) float32"（OpenCV 5 要求 float32）。

    接受三种写法：`(N,3)`、`(N,1,3)`（OpenCV 单板常用）、`(n_views,N,3)`。
    `(N,1,3)` 与 `(n_views,N,3)` 靠 `shape[1]==1` 区分 —— 板只有一列点时
    两者会混淆，但那种板（`cols=1`）本来就不能用于标定。
    """
    if isinstance(objp, np.ndarray):
        a = np.asarray(objp, np.float32)
        if a.ndim == 2 or (a.ndim == 3 and a.shape[1] == 1):
            return [a.reshape(-1, 3)] * n
    out = [np.asarray(o, np.float32).reshape(-1, 3) for o in objp]
    if len(out) != n:
        raise ValueError(f"物体点数量 {len(out)} 与视图数量 {n} 不一致")
    return out


def _as_pts(pts, n_expected: int, what: str) -> list[np.ndarray]:
    out = [np.asarray(p, np.float64).reshape(-1, 2) for p in pts]
    for i, p in enumerate(out):
        if p.shape[0] != n_expected:
            raise ValueError(
                f"{what} 第 {i} 个视图有 {p.shape[0]} 个点，期望 {n_expected} 个"
                f"（板模型决定点数，对不上说明对应关系不可信）")
    return out


def _board_poses(obj_list: Sequence[np.ndarray], ip1: Sequence[np.ndarray],
                 K1: np.ndarray, D1: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """逐视图 solvePnP，得到板在左相机系的位姿（R_i, t_i）。"""
    poses = []
    for i, (obj, pts) in enumerate(zip(obj_list, ip1)):
        try:
            ok, rvec, tvec = cv2.solvePnP(
                obj, pts.reshape(-1, 1, 2), np.asarray(K1, np.float64),
                np.asarray(D1, np.float64), flags=cv2.SOLVEPNP_ITERATIVE)
        except cv2.error as exc:
            raise ValueError(
                f"第 {i} 个视图 solvePnP 抛错（点太少或几何退化：{pts.shape[0]} 个点）："
                f"{str(exc).splitlines()[0]}") from exc
        if not ok:
            raise ValueError(f"第 {i} 个视图 solvePnP 失败：角点与板模型不匹配")
        poses.append((cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3))[0],
                      np.asarray(tvec, np.float64).reshape(3)))
    return poses


def _residuals(params: np.ndarray, R0: np.ndarray, poses, obj_list, ip2, K2, D2) -> np.ndarray:
    """参数化重投影残差：`params = [w(3, rad), T(3, m)]`，`R = exp([w]×)·R0`。

    只统计右目（理由见模块文档：左目对 (R,T) 的雅可比恒为 0）。
    """
    dR = cv2.Rodrigues(params[:3])[0]
    R = dR @ R0
    T = params[3:6].reshape(3, 1)
    chunks = []
    for (Ri, ti), obj, p2 in zip(poses, obj_list, ip2):
        rvec = cv2.Rodrigues(R @ Ri)[0]
        tvec = R @ ti.reshape(3, 1) + T
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K2, D2)
        chunks.append(proj.reshape(-1, 2) - p2)
    return np.concatenate(chunks).ravel() if chunks else np.zeros(0)


def _jacobian(params0, R0, poses, obj_list, ip2, K2, D2) -> np.ndarray:
    """中心差分雅可比，形状 (m, 6)。"""
    m = _residuals(params0, R0, poses, obj_list, ip2, K2, D2).size
    J = np.zeros((m, 6), np.float64)
    steps = np.array([STEP_ROT] * 3 + [STEP_TRANS] * 3)
    for j in range(6):
        h = steps[j]
        pp = params0.copy()
        pm = params0.copy()
        pp[j] += h
        pm[j] -= h
        J[:, j] = (_residuals(pp, R0, poses, obj_list, ip2, K2, D2)
                   - _residuals(pm, R0, poses, obj_list, ip2, K2, D2)) / (2.0 * h)
    return J


def extrinsic_sigma(
    objp,
    imgpoints1,
    imgpoints2,
    K1,
    D1,
    K2,
    D2,
    R,
    T,
    image_size,
    *,
    noise_px: float | None = None,
) -> dict:
    """估 `(R, T)` 的 1σ 与含基线项的深度不确定度。

    参数
    ----
    objp        : (N,3) 或 逐视图列表（OpenCV 风格）
    imgpoints1/2: 左/右目角点，长度相同、每视图点数相同
    K1,D1,K2,D2 : 内参与畸变（与 `R,T` 出自同一次标定）
    R,T         : 外参；`x_R = R·x_L + T`，**`T` 单位 = 米**
    image_size  : `(w, h)`。本函数不直接用它（`projectPoints` 不需要），
                  保留是为了与 `calibrate_stereo` 的调用口径一致
    noise_px    : 角点定位噪声（像素）。缺省用 `√(SSR/(m−6))` 从残差估

    返回
    ----
    dict：`sigma_baseline_mm`, `sigma_angle_deg`, `sigma_t_mm`(3,),
    `sigma_r_deg`(3,), `noise_px`, `ssr_px2`, `dof`, `n_residuals`, `cond`,
    `baseline_mm`, `focal_px`, `depth_sigma_mm_at`(callable), `note`

    奇异时返回 `{"error": "雅可比奇异：数据不足以确定 6 个自由度", ...}` ——
    **不返回 inf/nan**（那会让下游把它当成"很大的数"继续算下去）。
    """
    R0 = np.asarray(R, np.float64).reshape(3, 3)
    T0 = np.asarray(T, np.float64).reshape(3)
    K1 = np.asarray(K1, np.float64).reshape(3, 3)
    K2 = np.asarray(K2, np.float64).reshape(3, 3)
    D1 = np.asarray(D1, np.float64).ravel()
    D2 = np.asarray(D2, np.float64).ravel()

    n_views = len(imgpoints1)
    if n_views == 0 or len(imgpoints2) != n_views:
        raise ValueError(
            f"左右视图数必须相同且非空（左 {len(imgpoints1)} / 右 {len(imgpoints2)}）")
    obj_list = _obj_list(objp, n_views)
    n_pts = obj_list[0].shape[0]
    ip1 = _as_pts(imgpoints1, n_pts, "左图角点")
    ip2 = _as_pts(imgpoints2, n_pts, "右图角点")

    m = 2 * n_pts * n_views
    dof = m - 6
    if dof <= 0:
        return {"error": f"残差维数不足：m={m}，减 6 个自由度后 ≤0，无法估计噪声",
                "n_residuals": m, "dof": dof}

    poses = _board_poses(obj_list, ip1, K1, D1)
    params0 = np.zeros(6, np.float64)
    params0[3:6] = T0

    r0 = _residuals(params0, R0, poses, obj_list, ip2, K2, D2)
    ssr = float(r0 @ r0)
    if not np.isfinite(ssr):
        return {"error": "残差非有限（R/T/内参里可能有 nan/inf）"}
    J = _jacobian(params0, R0, poses, obj_list, ip2, K2, D2)

    # ---- 奇异性检查（先按列归一化，理由见 COND_MAX 注释）----
    col = np.linalg.norm(J, axis=0)
    if not np.all(np.isfinite(col)) or np.any(col <= 0):
        dead = [i for i, c in enumerate(col) if not (c > 0)]
        return {"error": "雅可比奇异：数据不足以确定 6 个自由度",
                "dead_params": dead, "n_residuals": m, "dof": dof,
                "hint": f"第 {dead} 个参数（0-2 旋转 / 3-5 平移）对残差完全没有影响："
                        f"视图位姿过于单一，或板面与某些自由度退化"}
    Jn = J / col
    cond = float(np.linalg.cond(Jn))
    if not np.isfinite(cond) or cond > COND_MAX:
        return {"error": "雅可比奇异：数据不足以确定 6 个自由度",
                "cond": cond, "n_residuals": m, "dof": dof,
                "hint": f"列归一化后的条件数 {cond:.3e} > {COND_MAX:.0e}："
                        f"某些自由度几乎不可观测。补足位姿多样性（多角度、覆盖画面四角），"
                        f"而不是加正则化强行解出来"}

    sigma2 = ssr / dof                      # 每个残差分量的方差（px²）
    sigma_px = float(math.sqrt(sigma2)) if sigma2 > 0 else 0.0
    if noise_px is None:
        noise_px = sigma_px
    noise_px = float(noise_px)

    # Σ = σ² (JᵀJ)⁻¹。这里用 solve 而不是显式 inv，数值更稳。
    JtJ = J.T @ J
    try:
        cov = np.linalg.solve(JtJ, np.eye(6)) * sigma2
    except np.linalg.LinAlgError:
        return {"error": "雅可比奇异：数据不足以确定 6 个自由度",
                "cond": cond, "n_residuals": m, "dof": dof,
                "hint": "JᵀJ 在数值上不可逆（已通过条件数检查但 solve 仍失败）"}
    if not np.all(np.isfinite(cov)):
        return {"error": "雅可比奇异：数据不足以确定 6 个自由度",
                "cond": cond, "n_residuals": m, "dof": dof,
                "hint": "协方差矩阵出现非有限值"}

    cov_r = cov[:3, :3]          # rad²
    cov_t = cov[3:6, 3:6]        # m²
    diag_t = np.clip(np.diag(cov_t), 0.0, None)
    diag_r = np.clip(np.diag(cov_r), 0.0, None)

    baseline_m = float(np.linalg.norm(T0))
    if not (baseline_m > 0):
        return {"error": "基线为 0：T 全零，σ_B 无定义（检查外参是否解错）",
                "n_residuals": m, "dof": dof}
    u = T0 / baseline_m
    # delta 法：σ_B = |∂‖t‖/∂t · δt| = √(uᵀ Σ_t u)（用完整协方差，不只对角）
    sigma_b_m = float(math.sqrt(max(float(u @ cov_t @ u), 0.0)))
    sigma_axis_deg = np.degrees(np.sqrt(diag_r))
    sigma_angle_deg = float(np.degrees(math.sqrt(max(float(np.trace(cov_r)), 0.0))))

    f = float(K1[0, 0])

    def depth_sigma_mm_at(z_m: float) -> float:
        """距离 `z_m`（米）处的深度 1σ（mm），**含 σ_B 项**。

        公式推导见模块文档：σ_Z = Z·√((σ_d/d)² + (σ_B/B)²)，
        `d = f·B/Z`、`σ_d = √2·noise_px`。
        """
        z = float(z_m)
        if not (np.isfinite(z) and z > 0 and f > 0 and baseline_m > 0):
            return float("nan")
        d = f * baseline_m / z
        if not (np.isfinite(d) and d > 0):
            return float("nan")
        sd = math.sqrt(2.0) * noise_px
        rel = (sd / d) ** 2 + (sigma_b_m / baseline_m) ** 2
        return float(z * math.sqrt(rel) * 1000.0)

    return {
        "sigma_baseline_mm": float(sigma_b_m * 1000.0),
        "sigma_angle_deg": sigma_angle_deg,
        "sigma_t_mm": np.sqrt(diag_t) * 1000.0,
        "sigma_r_deg": sigma_axis_deg,
        "cov_t_mm2": cov_t * 1e6,
        "cov_r_rad2": cov_r,
        "noise_px": noise_px,
        "noise_px_from_residual": sigma_px,
        "ssr_px2": ssr,
        "dof": dof,
        "n_residuals": m,
        "n_views": n_views,
        "cond": cond,
        "baseline_mm": baseline_m * 1000.0,
        "focal_px": f,
        "depth_sigma_mm_at": depth_sigma_mm_at,
        "note": (
            "σ 是条件协方差（各视图板位姿、内参固定），比联合估计略保守；"
            "σ²=SSR/(m-6) 由同一批残差估得，会略偏小 —— 两个偏差方向相反。"
            "深度项**含** σ_B（旧 depth_sigma_mm 不含），但**不含** σ_f。"
        ),
    }
