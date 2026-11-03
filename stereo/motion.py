"""轨迹平滑与求导：把"逐帧点列"变成"能看、能算"的运动量。

## 为什么单独一个模块

查看器要在右下角画平滑后的轨迹，另一个人要拿速度/加速度去做卡尔曼与 Savitzky-Golay
对比 —— 这两件事如果各写一份平滑，就会出现"图上是这条线、CSV 是另一条线"的老问题
（本项目已经因为它返工过：同一趟两个深度）。所以平滑与求导**只有这一处实现**。

## 三条硬约定

1. **丢帧必须断开，不许跨段求导**。30 fps 下 dt=33 ms，丢一帧就是 66 ms；
   跨着缺口算 `(p[i+1]-p[i])/dt` 会把"瞬移"当成"高速"，一次就能把 `|a|` 抬到几十 m/s²。
   所以所有函数都先 `split_segments()`，段内才算差分。
2. **平滑在位置上做，不在速度上做**。先平滑 p(t)，再对平滑后的位置求导；
   反过来（先算速度再平滑）会让窗口跨过拐点，把真实的加减速抹平。
3. **不许悄悄外推**。段太短（点数 < poly+2）就原样返回并标 `valid=False`，
   让上层知道"这一段没被处理"，而不是拿一条直线糊过去。

## 默认参数为什么是 0.20 s / 2 阶（实测扫出来的，不是拍的）

数据：`20261103-093820`（0.75 m、30 fps、252 个匹配点、10 段）。同一段数据只改窗口：

| 窗口 | v 中位 | v p95 | a p95 | 抖动中位 |
| ---- | -------- | ------- | -------- | -------- |
| 原始 | 0.179 | 0.691 | 15.48 | 3.6 mm |
| 0.12 s | 0.182 | 0.603 | 9.99 | 3.1 mm |
| **0.20 s** | **0.172** | **0.534** | **7.87** | **2.9 mm** |
| 0.30 s | 0.162 | 0.496 | 7.06 | 2.0 mm |
| 0.45 s | 0.137 | 0.402 | 3.45 | 1.2 mm |

0.20 s 是拐点：速度中位只降 4%，但加速度 p95 从 15.5 压到 7.9（噪声主导的那一头）；
到 0.45 s 速度中位掉 23% —— 那不是"更平滑"，是把真实运动削掉了。
二阶多项式能跟上匀加速；三阶开始把噪声也拟合进去。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "split_segments",
    "smooth_positions",
    "resample_uniform",
    "kinematics",
]

#: 一段之内允许的最大相邻时间差（秒）。超过它就算"断开了"。
#: 默认 0.1 s ≈ 30 fps 下的 3 帧；`run_stereo_video` 的配对缺口通常是 1~13 帧。
DEFAULT_MAX_GAP_S = 0.10


def split_segments(t: np.ndarray, max_gap_s: float = DEFAULT_MAX_GAP_S) -> list[np.ndarray]:
    """按时间缺口把点列切成若干连续段，返回**下标数组**的列表。

    段与段之间没有任何时间上的连续性假设（丢帧、遮挡、跟踪丢失都会产生缺口），
    所以差分、平滑、求导都只能在段内做。

    只有一个点时也返回 `[array([i])]` —— 调用方自己决定"几个点才够算"，
    这个函数不替它丢数据。
    """
    t = np.asarray(t, dtype=float).ravel()
    n = t.size
    if n == 0:
        return []
    if n == 1:
        return [np.array([0], dtype=int)]
    breaks = np.where(np.diff(t) > float(max_gap_s))[0] + 1
    return [seg for seg in np.split(np.arange(n, dtype=int), breaks) if seg.size]


def resample_uniform(t: np.ndarray, P: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """把一段（必须已连续）按固定步长重采样。

    为什么要这一步：相机给的时间戳并不严格等间隔（实测中位 33.2 ms、p95 66.4 ms），
    而 Savitzky-Golay 的系数表**假定等间隔**。不等间隔直接套，窗口在时间上会忽长忽短，
    转折处就会过冲。重采样到固定 dt 之后再滤，稳定性明显更好。

    ``dt`` 取该段的中位间隔即可（调用方一般传 `np.median(np.diff(t))`）。
    区间取 `[t[0], t[-1]]`，不外推。
    """
    t = np.asarray(t, dtype=float).ravel()
    P = np.asarray(P, dtype=float)
    if t.size < 2 or dt <= 0:
        return t.copy(), P.copy()
    n = int(np.floor((t[-1] - t[0]) / dt)) + 1
    tu = t[0] + dt * np.arange(n)
    Pu = np.column_stack([np.interp(tu, t, P[:, k]) for k in range(P.shape[1])])
    return tu, Pu


def smooth_positions(
    t: np.ndarray,
    P: np.ndarray,
    *,
    window_s: float = 0.20,
    poly: int = 2,
    max_gap_s: float = DEFAULT_MAX_GAP_S,
) -> tuple[np.ndarray, np.ndarray]:
    """对位置做 Savitzky-Golay 平滑，**逐段**进行，返回 `(P_smooth, valid)`。

    `valid[i]` 为 False 表示第 i 个点没被平滑（段太短 / 被跳过的段），
    此时 `P_smooth[i] == P[i]` —— 宁可保留原值并标记，也不要外推一条假曲线。

    窗口按**时间**给（`window_s`），不是按点数：采样率一变，按点数给的窗口
    就换了个物理尺度（本项目踩过：同一段视频抽帧后窗口等效变成 2 倍长）。
    """
    t = np.asarray(t, dtype=float).ravel()
    P = np.asarray(P, dtype=float)
    out = P.copy()
    valid = np.zeros(t.size, dtype=bool)
    if t.size == 0:
        return out, valid

    try:
        from scipy.signal import savgol_filter
    except Exception:            # scipy 缺失：不静默改变语义，直接标记"没处理"
        return out, valid

    for seg in split_segments(t, max_gap_s):
        if seg.size < poly + 2:
            continue
        dt = float(np.median(np.diff(t[seg]))) if seg.size > 1 else 0.0
        if dt <= 0:
            continue
        win = int(round(window_s / dt))
        win = max(poly + 2, win | 1)              # 必须是奇数且 > poly
        if win > seg.size:                        # 段比窗口还短：退化成整段拟合
            win = seg.size if seg.size % 2 == 1 else seg.size - 1
        if win < poly + 2:
            continue
        tu, Pu = resample_uniform(t[seg], P[seg], dt)
        if tu.size < win:
            continue
        Ps = np.column_stack([savgol_filter(Pu[:, k], win, poly, mode="interp")
                              for k in range(P.shape[1])])
        # 采样回原始时间点（平滑值本身是连续的，插值回去不会引入新的抖动）
        out[seg] = np.column_stack([np.interp(t[seg], tu, Ps[:, k])
                                    for k in range(P.shape[1])])
        valid[seg] = True
    return out, valid


def kinematics(
    t: np.ndarray,
    P: np.ndarray,
    *,
    smooth: bool = True,
    window_s: float = 0.20,
    poly: int = 2,
    max_gap_s: float = DEFAULT_MAX_GAP_S,
) -> dict:
    """位置 → 速度 → 加速度，逐段中心差分（段首尾退化为单边差分）。

    返回 dict：
        t, P, V, A           —— 与输入等长的数组（未参与计算的点填 nan）
        speed, accel         —— 模长
        segment              —— 每点所属段号（-1 = 单点段，无法求导）
        smoothed             —— 每点是否被平滑过

    **段首尾的差分永远是单边**，精度比中间低一档；需要严格一致的精度时，
    调用方应当把段首尾若干点丢掉（本函数不替你做这个决定，只如实标出来）。
    """
    t = np.asarray(t, dtype=float).ravel()
    P = np.asarray(P, dtype=float)
    n = t.size
    if smooth:
        P_used, valid = smooth_positions(t, P, window_s=window_s, poly=poly,
                                         max_gap_s=max_gap_s)
    else:
        P_used, valid = P.copy(), np.zeros(n, dtype=bool)

    V = np.full_like(P_used, np.nan, dtype=float)
    A = np.full_like(P_used, np.nan, dtype=float)
    seg_id = np.full(n, -1, dtype=int)

    for k, seg in enumerate(split_segments(t, max_gap_s)):
        if seg.size < 2:
            continue
        seg_id[seg] = k
        ts, Ps = t[seg], P_used[seg]
        # 中心差分（内部）/ 单边（端点）
        V[seg] = np.gradient(Ps, ts, axis=0, edge_order=2 if seg.size > 2 else 1)
        if seg.size >= 3:
            A[seg] = np.gradient(V[seg], ts, axis=0, edge_order=2 if seg.size > 2 else 1)

    speed = np.linalg.norm(V, axis=1)
    accel = np.linalg.norm(A, axis=1)
    return {"t": t, "P": P_used, "V": V, "A": A, "speed": speed, "accel": accel,
            "segment": seg_id, "smoothed": valid}
