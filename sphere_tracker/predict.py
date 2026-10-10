"""轨迹预测。

两种互补的方法：

1. kalman 模式  —— 直接把卡尔曼滤波的状态外推：p(tau) = p + v*tau + 0.5*a*tau^2。
   优点是实时、平滑、不需要挑窗口；缺点是"匀加速"模型在碰撞瞬间会滞后。

2. quadratic 模式 —— 对最近一小段历史做加权二次最小二乘拟合再外推（默认）。
   对抛体/自由落体几乎是最优解；如果标定过相机、在三维米制坐标下
   还能用 use_gravity 把二次项直接固定成 g/2，只剩两个未知数，抗噪能力更强。

一个非常关键的细节：绝对不能拿整条轨迹去拟合。
球一旦落地反弹或被球拍击打，运动规律就变了，用几秒前的数据拟合出的
"未来"必然是错的。所以这里采用"自适应窗口"：

    从最长窗口（默认 0.45 s）开始试 -> 算拟合残差
    残差小于阈值 -> 说明这段数据符合同一条抛物线，直接用
    残差太大     -> 说明窗口里混进了反弹/击打，缩短窗口重试

于是自由飞行时自动用足长窗口（抗噪），刚反弹完自动退化成短窗口（跟得上）。
"""

from __future__ import annotations

import numpy as np

from .config import PredictConfig
from .types import TrailPoint, TrackState


def _points_array(trail: list[TrailPoint], dims: int) -> tuple[np.ndarray, np.ndarray]:
    """从拖尾里取出 (t, points) 数组；dims=3 时跳过没有 z 的点。"""
    ts: list[float] = []
    pts: list[list[float]] = []
    for p in trail:
        if dims == 3:
            if p.z is None:
                continue
            ts.append(p.t)
            pts.append([p.x, p.y, p.z])
        else:
            ts.append(p.t)
            pts.append([p.x, p.y])
    if not ts:
        return np.zeros(0), np.zeros((0, dims))
    return np.asarray(ts, dtype=np.float64), np.asarray(pts, dtype=np.float64)


def _smooth(points: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or points.shape[0] < window:
        return points
    kernel = np.ones(window, dtype=np.float64) / window
    out = np.empty_like(points)
    half = window // 2
    for axis in range(points.shape[1]):
        out[:, axis] = np.convolve(points[:, axis], kernel, mode="same")
        # 两端保留原值，避免卷积边界效应把最新点拉偏
        out[:half, axis] = points[:half, axis]
        out[-half:, axis] = points[-half:, axis]
    return out


def _fit_window(
    sub_t: np.ndarray,
    sub_p: np.ndarray,
    future: np.ndarray,
    cfg: PredictConfig,
    dims: int,
    gravity_axis: int,
    gravity_value: float | None,
    linear_axes: tuple[int, ...] = (),
) -> tuple[np.ndarray, float]:
    """对给定窗口内的点做逐轴拟合，返回 (预测点, 平均残差 RMS)。

    sub_t 是相对"当前时刻"的时间（<=0）。gravity_value 不为 None 时，
    重力轴上的二次项被固定为 g/2，未知数只剩 2 个。
    """
    n = sub_t.shape[0]
    w = np.linspace(0.4, 1.0, n)            # 越新的点权重越大
    out = np.empty((future.shape[0], dims), dtype=np.float64)
    total_res = 0.0
    for axis in range(dims):
        y = sub_p[:, axis]
        g = gravity_value if (gravity_value is not None and axis == gravity_axis) else None
        # 数据段太短时不要用自由二次拟合：3 个点必定能精确穿过一条抛物线，
        # 残差恒等于 0，"残差检验"就完全失效了，脏数据会被当成好数据。
        # 所以短段一律退化成线性（2 个未知数，至少还剩 1 个自由度）。
        use_quadratic = (g is None and axis not in linear_axes and n >= cfg.min_quadratic_points)
        if g is None and axis not in linear_axes and not use_quadratic:
            A = np.stack([np.ones_like(sub_t), sub_t], axis=1)
            coef, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
            fit_hist = A @ coef
            out[:, axis] = coef[0] + coef[1] * future
        elif axis in linear_axes and g is None:
            # 深度 Z 这类"没有加速度"的轴必须用线性模型。
            # 用二次拟合 Z 是灾难性的：Z 稍微有点正曲率，外推到 0.4 秒后
            # 就会变成负深度或几万米，投影回图像直接飞出太阳系。
            A = np.stack([np.ones_like(sub_t), sub_t], axis=1)
            coef, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
            fit_hist = A @ coef
            out[:, axis] = coef[0] + coef[1] * future
        elif g is not None and n >= 2:
            # y(tau) = p0 + v0*tau + 0.5*g*tau^2  -> 先扣掉已知重力项，再线性拟合
            y0 = y - 0.5 * g * sub_t ** 2
            A = np.stack([np.ones_like(sub_t), sub_t], axis=1)
            coef, *_ = np.linalg.lstsq(A * w[:, None], y0 * w, rcond=None)
            fit_hist = coef[0] + coef[1] * sub_t + 0.5 * g * sub_t ** 2
            out[:, axis] = coef[0] + coef[1] * future + 0.5 * g * future ** 2
        else:
            A = np.stack([np.ones_like(sub_t), sub_t, sub_t ** 2], axis=1)
            if n < 3:
                A = A[:, :2]
            coef, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
            fit_hist = A @ coef
            out[:, axis] = coef[0] + coef[1] * future + (
                coef[2] * future ** 2 if coef.shape[0] == 3 else 0.0
            )
        total_res += float(np.sqrt(np.mean((fit_hist - y) ** 2)))
    return out, total_res / max(dims, 1)


def predict_quadratic(
    trail: list[TrailPoint],
    cfg: PredictConfig,
    dims: int = 2,
    gravity_axis: int = 1,
    residual_tol: float | None = None,
    t_now: float | None = None,
    gravity_value: float | None = None,
    t_break: float | None = None,
) -> np.ndarray | None:
    """自适应窗口 + 自动裁剪的二次拟合外推。返回 (steps, dims) 的未来点。

    三个关键点：

    1. 时间基准是"当前帧" t_now，不是"最后一个观测点"。
       否则一旦有几帧丢失补偿，预测就会从旧位置出发，白白差出几十像素。

    2. 从最老的端点开始裁：先用最近 window_s 秒的数据拟合，
       残差超标就把最老的一部分丢掉重拟合。
       球一反弹，旧数据立刻被裁掉，模型自动跟上新的运动规律。

    3. 已知重力时（三维用 g，二维用 g_px = fy*g/Z）把二次项固定住，
       未知数从 3 个降到 2 个，刚反弹完只有四五个点也能预测准。

    residual_tol / gravity_value 不传就用配置里的值。
    """
    ts, pts = _points_array(trail, dims)
    # 已知重力时每个轴只有 2 个未知数，2-3 个点也能给出可用的速度估计；
    # 不知道重力时必须至少 4 个点，否则二次项完全不可信。
    min_pts = 3
    if pts.shape[0] < min_pts:
        return None

    # 丢失补偿（没有真实观测）的点是外推出来的，权重最低
    if dims == 2:
        observed = np.array([p.score > 0.0 for p in trail], dtype=bool)
    else:
        observed = np.array([p.score > 0.0 for p in trail if p.z is not None], dtype=bool)
    if observed.shape[0] == ts.shape[0] and observed.sum() >= 4:
        ts_f, pts_f = ts[observed], pts[observed]
    else:
        ts_f, pts_f = ts, pts
    order = np.argsort(ts_f)
    ts_f, pts_f = ts_f[order], pts_f[order]
    pts_f = _smooth(pts_f, cfg.smooth_window)

    t_ref = float(ts_f[-1]) if t_now is None else float(t_now)
    future = np.linspace(0.0, cfg.horizon_s, cfg.steps)
    tol = cfg.residual_tol if residual_tol is None else residual_tol

    # 重力值：三维用真实 g；二维用像素重力 g_px = fy*g/Z（由 pipeline 传入）
    g_val = gravity_value
    if g_val is None and dims == 3 and cfg.use_gravity:
        g_val = cfg.gravity
    if dims == 2 and not cfg.use_pixel_gravity:
        g_val = None

    # t_break：跟踪器报出来的"最近一次速度突变"时刻（落地反弹 / 被击打）。
    # 它只是个提示，真正定位分界点的是下面的"最长有效后缀"搜索。
    t_start = t_ref - max(cfg.window_s, cfg.min_window_s)
    if t_break is not None and t_break > t_start:
        t_start = float(t_break)

    sel = ts_f >= t_start
    ts_w, pts_w = ts_f[sel], pts_f[sel]
    if ts_w.shape[0] < min_pts:
        if t_break is not None:
            after = ts_f >= t_break
            if after.sum() >= 2:
                # 突变后才攒到一两个点：就用这一两个点。
                # 已知重力时每轴只有 2 个未知数，两个点也能解出速度，
                # 总比把突变前的数据混进来强得多。
                ts_w, pts_w = ts_f[after], pts_f[after]
                min_pts = 2
            else:
                return None
        else:
            # 球刚出现在画面里，数据不足一个窗口：用现有的全部点
            take = min(pts_f.shape[0], max(8, cfg.min_points * 2))
            ts_w, pts_w = ts_f[-take:], pts_f[-take:]
    if ts_w.shape[0] < min_pts:
        return None

    # 三维时 Z 轴用线性模型（相机不会前后加速）
    linear_axes: tuple[int, ...] = (2,) if dims == 3 else ()

    # ---- 最长有效后缀搜索 ----
    # 从"尽量用满历史"开始，一段段地丢掉最老的点，取第一个拟合残差合格的
    # 后缀。这等价于在问："以当前时刻结尾、且符合同一条抛物线的最长数据段
    # 是哪一段？" 答案就是"上一次运动突变之后的那一段"。
    #   自由飞行时 -> 第一段就合格，用满 0.45 s，最抗噪；
    #   刚反弹完   -> 前面几段都被否决，一直退到突变之后才通过。
    # 它不依赖跟踪器的机动标志，所以对"碰壁"这种横向的速度突变同样有效。
    best: tuple[float, np.ndarray] | None = None
    n_all = ts_w.shape[0]
    for start in range(0, n_all - min_pts + 1):
        sub_t = ts_w[start:] - t_ref
        sub_p = pts_w[start:]
        try:
            out, resid = _fit_window(
                sub_t, sub_p, future, cfg, dims, gravity_axis, g_val, linear_axes
            )
        except np.linalg.LinAlgError:
            continue
        if resid <= tol:
            return _sanitize(out, dims)
        if best is None or resid < best[0]:
            best = (resid, out)
    return _sanitize(best[1], dims) if best is not None else None


def _sanitize(points: np.ndarray, dims: int) -> np.ndarray | None:
    """把不物理的预测截断掉（深度变负、坐标变成 NaN/无穷）。"""
    if points is None or not np.all(np.isfinite(points)):
        return None
    if dims == 3:
        bad = np.nonzero(points[:, 2] <= 0.05)[0]
        if bad.size:
            keep = int(bad[0])
            if keep < 2:
                return None
            return points[:keep]
    return points


def predict_kalman(
    state: TrackState | None,
    cfg: PredictConfig,
    dims: int = 2,
) -> np.ndarray | None:
    """用卡尔曼状态做等加速外推。"""
    if state is None:
        return None
    future = np.linspace(0.0, cfg.horizon_s, cfg.steps)
    if dims == 2:
        out = np.empty((cfg.steps, 2), dtype=np.float64)
        out[:, 0] = state.x + state.vx * future + 0.5 * state.ax * future ** 2
        out[:, 1] = state.y + state.vy * future + 0.5 * state.ay * future ** 2
        return out
    if state.z is None:
        return None
    out = np.empty((cfg.steps, 3), dtype=np.float64)
    out[:, 0] = state.x + state.vx * future + 0.5 * state.ax * future ** 2
    out[:, 1] = state.y + state.vy * future + 0.5 * state.ay * future ** 2
    out[:, 2] = state.z + state.vz * future
    return out


class TrajectoryPredictor:
    """对外统一入口。"""

    def __init__(self, cfg: PredictConfig) -> None:
        self.cfg = cfg

    def predict(
        self,
        trail: list[TrailPoint],
        state: TrackState | None,
        dims: int = 2,
        residual_tol: float | None = None,
        t_now: float | None = None,
        gravity_value: float | None = None,
        t_break: float | None = None,
    ) -> np.ndarray | None:
        if not self.cfg.enable:
            return None
        if self.cfg.method == "kalman":
            return predict_kalman(state, self.cfg, dims)
        return predict_quadratic(
            trail, self.cfg, dims, residual_tol=residual_tol,
            t_now=t_now, gravity_value=gravity_value, t_break=t_break,
        )
