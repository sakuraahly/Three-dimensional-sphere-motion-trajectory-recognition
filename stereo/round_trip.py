"""往返一致性：一次来回切成「去程 / 回程」两条腿，量同一个 X 处两条腿差多少。

为什么需要
----------
双目轨迹只在**一个方向**上有硬约束（极线），竖直 Y 与深度 Z 都靠三角测量。
球来回摆的时候，如果同一个 X 处去程与回程的 (Y, Z) 系统性错开，那说明这条
轨迹本身**不是一条线**（窄椭圆 / 空回差），而不是逐帧检测抖动 —— 这两件事
处理方式完全不同：抖动该去调检测/平滑，几何偏差该去看装置与球。

这个模块只回答"差多少"，不判断"好不好"：把数交给调用方写进 `traj3d.json`
的 `round_trip` 块，命令行再打一行结论。

口径（定死，别各自发明）
------------------------
- 只按 **X** 切段：X 是主运动方向，Y/Z 的往返偏移正是要量的东西，拿它们切段
  会自我循环；
- 切段前对 X 做 3 点滑动平均（端点保持原值）：单帧抖动会在顶点附近造出
  `+ - +` 的假换向，一条腿被劈成三条；
- 段内 X 跨度 < `min_span_m` 的丢掉：那是顶点附近的原地震动，不是一条腿；
- **一条腿只属于一对**（第 0/1 段配对、第 2/3 段配对…）。若改成
  (1,2)、(2,3) 都配，一次「回程 + 下一次去程」会被当成来回，符号必然翻，
  正常的椭圆数据会被判成 `scattered`；
- 一对腿在 X 重叠区上取 `n_grid` 个等距点，两条腿各自按 X 排序后插值；
  一段里重复的 X 只留最早那个；
- **去程 = X 随时间增大的那条腿**（另一条是回程）。相邻两段方向必然相反，
  所以一对腿里恰有一条递增，符号不会因为"从哪头开始摆"而翻；
- `Δ = 回程 - 去程`；每对取**中位数**（不是均值：抗个别坏帧）。

纯函数：只依赖 numpy，不读文件、不抛异常（任何意外都返回 `None`）。
"""

from __future__ import annotations

import numpy as np

__all__ = ["analyze_round_trip"]

#: 一对腿的 X 重叠区至少要留这么宽（米），否则两条腿"挨着"但没几处可比。
_MIN_USABLE_M = 0.02

#: 少于这么多可用点就凑不出任何一段，直接返回 None（不返回 no_pairs）。
_MIN_POINTS = 4


def _smooth3(x: np.ndarray) -> np.ndarray:
    """3 点滑动平均，端点保持原值。"""
    if x.size < 3:
        return x.copy()
    out = x.copy()
    out[1:-1] = (x[:-2] + x[1:-1] + x[2:]) / 3.0
    return out


def _segments(xs: np.ndarray, min_span_m: float) -> list[tuple[int, int]]:
    """按差分符号把 (平滑后的) X 切成单调段，返回闭区间 **[i0, i1]** 列表。

    零差分（某帧几乎没动）沿用**前一个**方向：顶点那一帧的零差分不该把一条
    腿劈成两段；开头还没动的那几个点直接并进第一段。
    """
    d = np.diff(xs)
    if d.size == 0:
        return []
    sgn = np.sign(d)
    if not np.any(sgn != 0):
        return []                                   # 完全不动：没有腿

    # 前向填充：O(n)，把 0 换成最近的已知方向（开头的 0 仍是 0）
    idx = np.where(sgn != 0, np.arange(sgn.size), -1)
    np.maximum.accumulate(idx, out=idx)
    filled = np.where(idx >= 0, sgn[np.maximum(idx, 0)], 0.0)

    starts = [0, *(int(i) + 1 for i in np.nonzero(np.diff(filled) != 0)[0])]
    ends = [*starts[1:], int(filled.size)]          # ends[k] 是差分下标（含）
    out: list[tuple[int, int]] = []
    for a, b in zip(starts, ends):
        if filled[a] == 0 or b <= a:
            continue                                # 开头还没动的一段：不是腿
        seg = xs[a:b + 1]
        if float(seg.max() - seg.min()) < min_span_m:
            continue                                # 顶点附近的原地震动
        out.append((int(a), int(b)))
    return out


def _speed_mm_s(t: np.ndarray, P: np.ndarray) -> np.ndarray:
    """整条轨迹的速度（mm/s）。时间轴不可用就退回按帧编号求梯度。"""
    try:
        if t.size >= 2 and bool(np.all(np.diff(t) > 0)):
            vel = np.gradient(P, t, axis=0)
        else:
            vel = np.gradient(P, axis=0, edge_order=1)
        return np.linalg.norm(vel, axis=1) * 1000.0
    except Exception:                               # noqa: BLE001 —— 绝不影响主流程
        return np.zeros(P.shape[0], dtype=float)


def _interp_on(grid: np.ndarray, x: np.ndarray, v: np.ndarray,
               sl: tuple[int, int]) -> np.ndarray | None:
    """把一段的 v 按 X 插值到 grid 上；这段没有两个不同的 X 就返回 None。"""
    a, b = sl
    xs = x[a:b + 1]
    vs = v[a:b + 1]
    if xs.size < 2:
        return None
    order = np.argsort(xs, kind="stable")
    xs = xs[order]
    vs = vs[order]
    keep = np.ones(xs.size, dtype=bool)
    keep[1:] = np.diff(xs) > 0                      # 重复 X 只留最早那个
    if int(keep.sum()) < 2:
        return None
    return np.interp(grid, xs[keep], vs[keep])


def _no_pairs() -> dict:
    return {
        "pairs": [],
        "n_pairs": 0,
        "dy_median_mm": None,
        "dz_median_mm": None,
        "sep_median_mm": None,
        "sign_consistent": False,
        "verdict": "no_pairs",
        "note": "轨迹里没有可配对的来回（只有一条腿或点数太少），没法量往返一致性",
    }


def _analyze(t: np.ndarray, P: np.ndarray, *, min_span_m: float,
             inner_margin_m: float, n_grid: int, retrace_mm: float) -> dict | None:
    t = np.asarray(t, dtype=float).reshape(-1)
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[1] < 3 or P.shape[0] != t.size:
        return None
    P = P[:, :3]

    good = np.isfinite(t) & np.all(np.isfinite(P), axis=1)
    if not bool(np.all(good)):
        t, P = t[good], P[good]
    if P.shape[0] < _MIN_POINTS:
        return None

    X, Y, Z = P[:, 0], P[:, 1], P[:, 2]
    segs = _segments(_smooth3(X), float(min_span_m))
    if len(segs) < 2:
        return _no_pairs()

    speed = _speed_mm_s(t, P)
    grid_n = max(2, int(n_grid))
    margin = float(inner_margin_m)
    pairs: list[dict] = []

    # 一条腿只属于一对：0↔1、2↔3 …（见模块开头的口径说明）
    for (a1, b1), (a2, b2) in zip(segs[0::2], segs[1::2]):
        xa, xb = X[a1:b1 + 1], X[a2:b2 + 1]
        lo = max(float(xa.min()), float(xb.min()))
        hi = min(float(xa.max()), float(xb.max()))
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue
        if hi - lo < 2.0 * margin + _MIN_USABLE_M:
            continue                                # 重叠区太窄，没什么可比的
        grid = np.linspace(lo + margin, hi - margin, grid_n)

        # 去程 = **X 随时间增大**的那条腿（另一条是回程）；相邻两段方向必然相反，
        # 所以一对腿里恰有一条递增。退化/同向时取时间在前的当去程。
        a_inc = float(X[b1]) > float(X[a1])
        b_inc = float(X[b2]) > float(X[a2])
        if b_inc and not a_inc:
            out_sl, ret_sl = (a2, b2), (a1, b1)
        else:
            out_sl, ret_sl = (a1, b1), (a2, b2)

        y_out = _interp_on(grid, X, Y, out_sl)
        y_ret = _interp_on(grid, X, Y, ret_sl)
        z_out = _interp_on(grid, X, Z, out_sl)
        z_ret = _interp_on(grid, X, Z, ret_sl)
        if y_out is None or y_ret is None or z_out is None or z_ret is None:
            continue

        dy = y_ret - y_out
        dz = z_ret - z_out
        sep = np.hypot(dy, dz)
        dz_med = float(np.median(dz))
        dz_med_mm = dz_med * 1000.0

        if dz_med != 0.0:
            same = float(np.mean(np.sign(dz) == np.sign(dz_med)))
        else:
            # 中位数为 0：用"非零符号是否一致"衡量，全零就是完全重合（1.0）
            nz = np.sign(dz[np.sign(dz) != 0])
            same = 1.0 if nz.size == 0 else float(np.mean(nz == np.sign(nz[0])))

        def _med_speed(sl: tuple[int, int]) -> float | None:
            sp = _interp_on(grid, X, speed, sl)
            return None if sp is None else round(float(np.median(sp)), 3)

        pairs.append({
            "x_overlap_m": [round(lo + margin, 6), round(hi - margin, 6)],
            "dy_median_mm": round(float(np.median(dy)) * 1000.0, 3),
            "dz_median_mm": round(dz_med_mm, 3),
            "sep_median_mm": round(float(np.median(sep)) * 1000.0, 3),
            "same_sign_ratio": round(same, 3),
            "speed_out_mm_s": _med_speed(out_sl),
            "speed_back_mm_s": _med_speed(ret_sl),
        })

    if not pairs:
        return _no_pairs()

    n_pairs = len(pairs)
    dy_med = round(float(np.median([p["dy_median_mm"] for p in pairs])), 3)
    dz_med = round(float(np.median([p["dz_median_mm"] for p in pairs])), 3)
    sep_med = round(float(np.median([p["sep_median_mm"] for p in pairs])), 3)
    signs = np.sign([p["dz_median_mm"] for p in pairs])
    sign_consistent = bool(np.all(signs >= 0) or np.all(signs <= 0))

    if abs(sep_med) < float(retrace_mm):
        verdict = "retrace"
        note = f"去程与回程重合在 {float(retrace_mm):g} mm 内，属于原路返回"
    elif sign_consistent:
        verdict = "ellipse"
        note = (f"去程与回程系统性不重合（ΔY {dy_med:+.1f} mm / ΔZ {dz_med:+.1f} mm，"
                f"{n_pairs} 段同号）：轨迹本身是窄椭圆，不是噪声")
    else:
        verdict = "scattered"
        note = ("来回偏移方向不固定（符号不一致），更像检测抖动而不是固定几何，"
                "建议看 quality")

    return {
        "pairs": pairs,
        "n_pairs": n_pairs,
        "dy_median_mm": dy_med,
        "dz_median_mm": dz_med,
        "sep_median_mm": sep_med,
        "sign_consistent": sign_consistent,
        "verdict": verdict,
        "note": note,
    }


def analyze_round_trip(t, P, *, min_span_m: float = 0.08,
                       inner_margin_m: float = 0.02, n_grid: int = 30,
                       retrace_mm: float = 2.0) -> dict | None:
    """量「去程 / 回程」在同一个 X 处的 Y、Z 差多少。

    参数
    ----
    t : (N,) 时间（秒），按时间顺序。
    P : (N,3) 三维点 `[X, Y, Z]`（米），与 `t` 一一对应。
    min_span_m : 一段 X 跨度小于它就算"顶点抖动"，不是一条腿。
    inner_margin_m : 重叠区两端各内缩多少（避开换向处的插值外推）。
    n_grid : 重叠区上取多少个等距点。
    retrace_mm : 中位偏移小于它就算"原路返回"。

    返回
    ----
    dict（键名见 `docs/数据标准.md` 的 `round_trip` 一节）；空输入/点数不够/
    任何意外都返回 `None` —— **本函数不抛异常**。
    """
    try:
        return _analyze(t, P, min_span_m=min_span_m,
                        inner_margin_m=inner_margin_m,
                        n_grid=n_grid, retrace_mm=retrace_mm)
    except Exception:                               # noqa: BLE001 —— 见上
        return None
