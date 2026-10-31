"""三维轨迹的后处理：Savitzky-Golay 平滑 + 常速度 Kalman。

输入约定：`t` 秒（单调），`xyz` 是 (N, 3) 米（双目校正后左相机系）。
这两个函数只做数学，不碰相机/检测，方便用合成真值单独测。

为什么先做这两个
----------------
双目三角测量逐帧独立，抖动来自检测像素噪声；预测又要求"位置 + 速度"。
平滑解决前者，Kalman 同时给出后者。它们都应该在**有可信三维点之后**再跑；
检测错了，平滑只会把错误轨迹磨得更光滑。
"""

from __future__ import annotations

import numpy as np

try:                                # scipy 是规划依赖；没有就退化为滑动平均
    from scipy.signal import savgol_filter
except Exception:                   # noqa: BLE001
    savgol_filter = None

#: 可选的平滑方法名（**唯一清单**：CLI 的 `choices` 也读这里，别再手写一份）。
SMOOTH_METHODS = ("savgol", "kalman")


def savgol_trajectory(
    t: np.ndarray,
    xyz: np.ndarray,
    *,
    window: int = 11,
    polyorder: int = 2,
) -> np.ndarray:
    """对 (N,3) 三维点逐轴做 Savitzky-Golay 平滑，返回同形状数组。

    - `window` 会自动夹到"不超过 N 的奇数"；N<3 原样返回；
    - 没有 scipy 时退化为同宽度滑动平均（仍是零相位，但边缘更差）。
    """
    arr = np.asarray(xyz, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"xyz 必须是 (N,3)，现在是 {arr.shape}")
    n = len(arr)
    if n < 3:
        return arr.copy()
    w = int(window)
    if w % 2 == 0:
        w -= 1
    w = min(w, n if n % 2 else n - 1)
    if w < 3:
        return arr.copy()
    p = max(1, min(int(polyorder), w - 1))
    if savgol_filter is not None:
        return savgol_filter(arr, window_length=w, polyorder=p, axis=0)
    # 无 scipy 的兜底：中心滑动平均（不改变长度）
    pad = w // 2
    padded = np.pad(arr, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(w, dtype=np.float64) / w
    return np.column_stack([
        np.convolve(padded[:, i], kernel, mode="valid") for i in range(3)
    ])


class Kalman3D:
    """常速度模型的三维 Kalman：状态 [x,y,z,vx,vy,vz]。

    用法::

        kf = Kalman3D()
        for t, p in points:
            filtered = kf.update(t, p)
        future = kf.predict(t + 0.2)

    `update()` 返回滤波后的位置；`predict()` 不改状态。
    时间必须单调；dt<=0 时原样返回当前估计。
    """

    def __init__(self, process_var: float = 1e-4, measure_var: float = 1e-3) -> None:
        self.q = float(process_var)
        self.r = float(measure_var)
        self.x: np.ndarray | None = None
        self.P: np.ndarray | None = None
        self.t: float | None = None

    @staticmethod
    def _F(dt: float) -> np.ndarray:
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        return F

    def _Q(self, dt: float) -> np.ndarray:
        # 简化白噪声加速度模型：位置块 dt^4/4，速度块 dt^2，交叉 dt^3/2
        q = self.q
        Q = np.zeros((6, 6))
        for i in range(3):
            Q[i, i] = q * dt ** 4 / 4.0
            Q[i, i + 3] = Q[i + 3, i] = q * dt ** 3 / 2.0
            Q[i + 3, i + 3] = q * dt ** 2
        return Q

    def update(self, t: float, z: np.ndarray) -> np.ndarray | None:
        z = np.asarray(z, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(z)):
            return None
        if self.x is None:
            self.x = np.zeros(6)
            self.x[:3] = z
            self.P = np.eye(6) * 1.0
            self.P[3:, 3:] *= 10.0
            self.t = float(t)
            return z

        dt = float(t) - float(self.t)
        if dt <= 0:
            return self.x[:3].copy()
        F = self._F(dt)
        H = np.zeros((3, 6))
        H[:, :3] = np.eye(3)
        R = np.eye(3) * self.r
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
        self.t = float(t)
        return self.x[:3].copy()

    def predict(self, t: float) -> np.ndarray | None:
        if self.x is None:
            return None
        dt = float(t) - float(self.t)
        if dt <= 0:
            return self.x[:3].copy()
        return (self._F(dt) @ self.x)[:3].copy()

    @property
    def velocity(self) -> np.ndarray | None:
        return None if self.x is None else self.x[3:].copy()


def jitter_rms(xyz: np.ndarray) -> float:
    """三维抖动的**相对**指标：二阶中心差分的 RMS（米）。

    为什么能用它
    ------------
    逐帧三角测量的抖动是高频的，而真实运动在一个窗口内近似低阶多项式；
    二阶差分 `p[i+1] - 2·p[i] + p[i-1]` 压掉低频趋势、放大高频噪声，
    所以"平滑后它明显变小"就等价于"把噪声去掉了"。

    两条限制（必须知道）
    ------------------
    - 它**同时包含真实加速度**（平抛段约 `g·Δt²`），因此只能用于**同一序列
      平滑前后**的相对对比，**不是**绝对精度，**不能**跨会话/跨场景比大小；
    - 假设采样近似等间隔（不等间隔时前后同权，比值仍可用）。

    N<3（凑不出一个二阶差分）返回 0.0。
    """
    arr = np.asarray(xyz, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"xyz 必须是 (N,3)，现在是 {arr.shape}")
    if len(arr) < 3:
        return 0.0
    d2 = np.diff(arr, n=2, axis=0)
    return float(np.sqrt(np.mean(np.sum(d2 ** 2, axis=1))))


def smooth_trajectory(
    t: np.ndarray,
    xyz: np.ndarray,
    *,
    method: str = "savgol",
    window: int = 11,
    polyorder: int = 2,
    process_var: float = 1e-4,
    measure_var: float = 1e-3,
) -> tuple[np.ndarray, dict]:
    """按 `method` 平滑 (N,3) 三维点，返回 ``(smoothed, params)``。

    - ``method="savgol"``：逐轴 S-G。多项式阶 2 能**精确保留恒定加速度**，
      所以对平抛段只有去噪、没有滞后 —— 这是默认，也是数据选出来的（见
      `docs/planbooks/2026-10-31_M4轨迹平滑接主流程.md` §4 的证伪实验）。
    - ``method="kalman"``：常速度 `Kalman3D`。**只作对照**：常速度模型在平抛
      （有恒定加速度）与反弹（速度突变）处有**系统性滞后**，`q` 调大就几乎
      不滤波、调小就严重偏；不要拿"它抖动更小"当更好的证据（那是削平了真实运动）。

    `params` 是要写进 `traj3d.json` 的参数字典，便于复现。
    """
    arr = np.asarray(xyz, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"xyz 必须是 (N,3)，现在是 {arr.shape}")
    tt = np.asarray(t, dtype=np.float64).reshape(-1)
    if tt.shape[0] != arr.shape[0]:
        raise ValueError(f"t 与 xyz 长度不一致：{tt.shape[0]} vs {arr.shape[0]}")
    if method == "savgol":
        out = savgol_trajectory(tt, arr, window=window, polyorder=polyorder)
        return out, {"window": int(window), "polyorder": int(polyorder)}
    if method == "kalman":
        kf = Kalman3D(process_var=process_var, measure_var=measure_var)
        out = np.empty_like(arr)
        for i, (ti, p) in enumerate(zip(tt, arr)):
            r = kf.update(float(ti), p)
            out[i] = p if r is None else r       # 非有限值：原样保留，不编造
        return out, {"process_var": float(process_var), "measure_var": float(measure_var)}
    raise ValueError(f"未知平滑方法 {method!r}（可选：{' / '.join(SMOOTH_METHODS)}）")
