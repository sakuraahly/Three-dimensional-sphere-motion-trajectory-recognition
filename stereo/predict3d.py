"""三维弹道预测：匀速 + 已知重力，以及反弹（机动）检测。

为什么三维比二维简单
--------------------
二维侧（`sphere_tracker/predict.py`）的重力被投到像平面，是一个"像素/秒²"的量
`g_px = fy * g / Z`：它同时依赖焦距 `fy` 和**深度 Z**，而 Z 本身是估计值，随球
远近漂移。所以二维必须把重力当未知量（自由二次拟合）或半未知量（`g_px` 逐帧
估），二次项和深度互相耦合，窗口一短就不稳，还要额外规定"深度轴只能用线性"。

三维里 `p(t) = p0 + v0·t + 0.5·g·t²` 的二次项是**已知常向量**：`g` 与焦距、深度、
像素全都无关。于是每条轴只剩 `(p0, v0)` 两个未知数，线性最小二乘一步解出，
不需要迭代优化器，也不存在"哪个轴能用二次"的问题。

限制（必须知道）
----------------
`g` 的方向由**坐标系**定义。本项目相机/世界系的约定是 **Y 向下**，所以默认
`g = (0, +9.81, 0)`；如果你的数据是"Y 向上"的数学系，必须显式传
`PredictConfig(g=(0, -9.81, 0))`。相机俯拍/仰拍/侧装时，真实重力在相机系里
是斜向量；拿到世界系后务必用 `world_gravity_world()` / `PredictConfig(g=...)`
覆盖，不要依赖默认值。

坐标与时基约定
--------------
- `times`：秒，单调（内部会排序；重复时间戳去重，丢弃数记在 `dropped` 里）；
- `points`：(N, 3) 米，任意固定坐标系；
- `BallisticState.p0` 是 **`t_ref` 时刻**的位置（`t_ref` = 最后一个有效样本的时刻），
  不是第一个样本；`predict` 的 horizon 也相对 `t_ref`。这样"现在"永远是最新观测。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_MIN_SPAN_S = 1e-3        # 再短就没有杠杆臂，cond 会爆
_COND_LIMIT = 1e8
_PIECE_MIN = 3            # 分段拟合每段最少点数（2 个未知数 + 1 个自由度）


@dataclass(frozen=True)
class PredictConfig:
    """预测配置。`g` 可覆盖 —— 世界系不同就换这里。"""

    # 本项目约定 Y 向下（与相机系一致）；Y 向上的数学系请显式传负号。
    g: tuple[float, float, float] = (0.0, 9.81, 0.0)
    horizons: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5)
    min_samples: int = 5
    max_gap_s: float = 0.25
    residual_gate_m: float = 0.05

    def __post_init__(self) -> None:
        gg = np.asarray(self.g, dtype=np.float64)
        if gg.shape != (3,):
            raise ValueError(f"g 必须是 3 维向量，现在是 {gg.shape}")
        if self.min_samples < 3:
            raise ValueError("min_samples 至少 3（2 个未知数 + 1 个自由度）")
        if self.residual_gate_m <= 0.0:
            raise ValueError("residual_gate_m 必须为正")
        object.__setattr__(self, "g", (float(gg[0]), float(gg[1]), float(gg[2])))


@dataclass(frozen=True)
class BallisticState:
    """一条弹道的拟合结果。`p0` 在 `t_ref` 时刻，`g` 一并带上供外推。"""

    p0: np.ndarray
    v0: np.ndarray
    g: np.ndarray
    rms_m: float
    n: int
    t_ref: float
    dropped: int = 0


def _as_arrays(times, points) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"points 必须是 (N,3)，现在是 {p.shape}")
    if t.shape[0] != p.shape[0]:
        raise ValueError(f"times 与 points 长度不一致：{t.shape[0]} vs {p.shape[0]}")
    return t, p


def _prepare(times, points, cfg: PredictConfig) -> tuple[np.ndarray, np.ndarray, int]:
    """排序 + 去重 + 丢弃非有限值 + 大间隙处截断到"最后一段连续数据"。"""
    t, p = _as_arrays(times, points)
    if t.size == 0:
        return t, p, 0

    finite = np.isfinite(t) & np.all(np.isfinite(p), axis=1)
    dropped = int(np.count_nonzero(~finite))
    t, p = t[finite], p[finite]
    if t.size == 0:
        return t, p, dropped

    order = np.argsort(t, kind="stable")
    t, p = t[order], p[order]

    keep = np.ones(t.size, dtype=bool)
    keep[1:] = np.diff(t) > 1e-9          # 重复时间戳只留第一个
    dropped += int(np.count_nonzero(~keep))
    t, p = t[keep], p[keep]

    # 大间隙（丢帧/跟踪丢失）之后的数据与之前不是同一段观测，
    # 硬拼在一起会把速度估成假的；截到最后一个间隙之后。
    if t.size > 1:
        breaks = np.nonzero(np.diff(t) > cfg.max_gap_s)[0]
        if breaks.size:
            start = int(breaks[-1]) + 1
            t, p = t[start:], p[start:]
    return t, p, dropped


def _fit_core(t: np.ndarray, p: np.ndarray, g: np.ndarray):
    """无门限的最小二乘：(p0, v0, rms)。重力项先扣掉，再解线性两参数。"""
    t_ref = float(t[-1])
    tau = t - t_ref
    y = p - 0.5 * g[None, :] * (tau ** 2)[:, None]
    A = np.stack([np.ones_like(tau), tau], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = A @ coef - y
    rms = float(np.sqrt(np.mean(resid ** 2))) if resid.size else float("inf")
    return coef[0].copy(), coef[1].copy(), rms, t_ref, A


def fit_ballistic(
    times,
    points,
    cfg: PredictConfig | None = None,
    reason: list[str] | None = None,
) -> BallisticState | None:
    """对一段三维观测做 `p(t)=p0+v0·τ+0.5·g·τ²` 的最小二乘拟合。

    只解 `(p0, v0)`（每条轴 2 个未知数），**不做迭代优化**。
    不合格时返回 `None`，绝不硬给一个解；原因写进可选的 `reason` 列表：

        n < min_samples          -> "样本不足…"
        时间跨度太短 / 矩阵退化    -> "时间跨度太短…"
        rms_m > residual_gate_m  -> "残差 … 超门限"（多半是窗口里混进了反弹/击打）
    """
    cfg = cfg or PredictConfig()
    t, p, dropped = _prepare(times, points, cfg)
    n = int(t.size)
    if n < cfg.min_samples:
        if reason is not None:
            reason.append(f"样本不足：可用 {n} < 需要 {cfg.min_samples}")
        return None

    span = float(t[-1] - t[0])
    if span < _MIN_SPAN_S:
        if reason is not None:
            reason.append(f"时间跨度太短：{span * 1e3:.3f} ms，撑不起速度估计")
        return None

    g = np.asarray(cfg.g, dtype=np.float64)
    p0, v0, rms, t_ref, A = _fit_core(t, p, g)
    if not np.all(np.isfinite(p0)) or not np.all(np.isfinite(v0)):
        if reason is not None:
            reason.append("时间跨度太短：设计矩阵退化（最小二乘无解）")
        return None
    if float(np.linalg.cond(A)) > _COND_LIMIT:
        if reason is not None:
            reason.append(f"时间跨度太短：设计矩阵条件数 {np.linalg.cond(A):.3g}")
        return None

    if rms > cfg.residual_gate_m:
        if reason is not None:
            reason.append(
                f"残差 {rms * 1e3:.1f} mm 超过门限 {cfg.residual_gate_m * 1e3:.0f} mm"
                "（窗口内可能混进了反弹/击打）"
            )
        return None

    return BallisticState(p0=p0, v0=v0, g=g, rms_m=rms, n=n, t_ref=t_ref, dropped=dropped)


def predict(state: BallisticState | None, horizons) -> dict[float, np.ndarray]:
    """解析式外推：`p(t_ref+h) = p0 + v0·h + 0.5·g·h²`。`state is None` -> `{}`。"""
    if state is None:
        return {}
    g = np.asarray(state.g, dtype=np.float64)
    p0 = np.asarray(state.p0, dtype=np.float64)
    v0 = np.asarray(state.v0, dtype=np.float64)
    out: dict[float, np.ndarray] = {}
    for h in horizons:
        hh = float(h)
        out[hh] = p0 + v0 * hh + 0.5 * g * hh * hh
    return out


def detect_bounce(
    times,
    points,
    cfg: PredictConfig | None = None,
    reason: list[str] | None = None,
) -> int | None:
    """检测"运动规律突变"（落地反弹 / 被击打），返回后段第一点的索引。

    判据分两步，缺一不可：

    1. **先看整段是否还符合同一条弹道。** 整段残差没超门限 -> 没有机动，直接
       `None`。这一步是"不许对噪声报反弹"的闸门：纯噪声仍然是同一条抛物线，
       残差只有毫米级，压根进不了搜索。
    2. 整段残差超标 -> 逐候选切分点分段拟合，要求**两段各自都**是同一条合格
       弹道（`max(rms_pre, rms_post) ≤ residual_gate_m`，且明显优于整段）。
       只有真反弹点能同时满足"前段是一段抛物线、后段是另一段抛物线"；
       切早或切晚都会让某一侧混进另一侧的数据，残差超标而被否决。
       速度沿重力方向由同向变反向（`v·g` 由正变负）作为加分项，用于在多解里挑真点。

    索引是 `_prepare` 之后（排序、去重、截断）数组里的下标；正常输入（已排序、
    无重复）下与原始下标一致。
    """
    cfg = cfg or PredictConfig()
    t, p, _ = _prepare(times, points, cfg)
    n = int(t.size)
    if n < 2 * _PIECE_MIN:
        if reason is not None:
            reason.append(f"样本不足：{n} 点不足以切成两段")
        return None

    g = np.asarray(cfg.g, dtype=np.float64)
    _, _, rms_all, _, _ = _fit_core(t, p, g)
    if rms_all <= cfg.residual_gate_m:
        if reason is not None:
            reason.append(f"整段残差 {rms_all * 1e3:.1f} mm 在门限内，无机动")
        return None

    up = int(np.argmax(np.abs(g)))           # 重力轴（默认 y）
    best: tuple[tuple[int, float], int, str] | None = None
    for i in range(_PIECE_MIN, n - _PIECE_MIN + 1):
        _, v_pre, rms_pre, _, _ = _fit_core(t[:i], p[:i], g)
        _, v_post, rms_post, _, _ = _fit_core(t[i:], p[i:], g)
        worst = max(rms_pre, rms_post)
        if worst > cfg.residual_gate_m:
            continue                          # 至少一侧不是合格弹道 -> 不是切分点
        if worst > 0.5 * rms_all:
            continue                          # 分段没带来明显改善 -> 不是真突变
        # 反弹 = 沿重力方向的速度**由同向变反向**。不能写死
        # `v_pre < 0 < v_post`：那假设 g 为负（Y 向上）；本项目默认 Y 向下，
        # 下落时 v 为正、反弹后才变负。用 `v·g` 的符号判断才与 g 无关。
        flipped = bool(v_pre[up] * g[up] > 0.0 > v_post[up] * g[up])
        score = (0 if flipped else 1, worst)
        if best is None or score < best[0]:
            note = ("速度沿重力方向由同向变反向" if flipped else "分段残差显著下降")
            best = (score, i, f"{note}：切分 {i}，前段 {rms_pre * 1e3:.1f} mm / "
                              f"后段 {rms_post * 1e3:.1f} mm（整段 {rms_all * 1e3:.1f} mm）")
    if best is None:
        if reason is not None:
            reason.append(f"整段残差 {rms_all * 1e3:.1f} mm 超标，但找不到两段各自合格的切分点")
        return None
    if reason is not None:
        reason.append(best[2])
    return int(best[1])
