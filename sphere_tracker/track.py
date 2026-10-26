"""卡尔曼滤波 + 轨迹管理。

为什么必须上滤波：检测出来的圆心每帧都会抖 1-3 个像素，
直接差分求速度，噪声会被放大成几十上百像素/秒的乱跳，
预测轨迹就没法看了。卡尔曼滤波同时给出平滑的位置/速度/加速度，
"预测未来"本质上就是把这个状态往前外推。
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .config import TrackerConfig
from .plugins import plugin
from .types import Detection, TrailPoint, TrackState


class KalmanFilter:
    """线性卡尔曼滤波：状态 = [x, y, (vx, vy), (ax, ay)]。

    order=1 匀速模型（CV）：球做直线运动时最稳；
    order=2 匀加速模型（CA）：抛体/自由落体时更准，默认用它。
    支持变步长 dt（摄像头帧率不稳时很重要）。
    """

    def __init__(
        self,
        order: int = 2,
        process_noise: float = 1.5e2,
        measurement_noise: float = 2.0,
        max_q_scale: float = 64.0,
    ) -> None:
        self.order = int(order)
        self.dim = 4 if self.order == 1 else 6
        self.q_base = float(process_noise)
        self.q = self.q_base
        # 自适应过程噪声的上限倍数，来自 TrackerConfig.max_q_scale。
        # 此前这里硬编码 64.0，导致改配置不生效（死配置）；默认值保持 64.0 以免改变行为。
        self.q_max = self.q_base * float(max_q_scale)
        self.r = float(measurement_noise) ** 2
        self.x = np.zeros(self.dim, dtype=np.float64)
        self.P = np.eye(self.dim, dtype=np.float64) * 1e4
        self.H = np.zeros((2, self.dim), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.R = np.eye(2, dtype=np.float64) * self.r
        self.initialized = False

    # -- 模型矩阵 ----------------------------------------------------------
    def _F(self, dt: float) -> np.ndarray:
        F = np.eye(self.dim, dtype=np.float64)
        F[0, 2] = dt
        F[1, 3] = dt
        if self.order == 2:
            F[0, 4] = 0.5 * dt * dt
            F[1, 5] = 0.5 * dt * dt
            F[2, 4] = dt
            F[3, 5] = dt
        return F

    def _Q(self, dt: float) -> np.ndarray:
        Q = np.zeros((self.dim, self.dim), dtype=np.float64)
        if self.order == 1:
            block = np.array(
                [[dt ** 3 / 3.0, dt ** 2 / 2.0], [dt ** 2 / 2.0, dt]], dtype=np.float64
            )
            for axis in (0, 1):
                idx = [axis, axis + 2]
                Q[np.ix_(idx, idx)] = block * self.q
        else:
            block = np.array(
                [
                    [dt ** 5 / 20.0, dt ** 4 / 8.0, dt ** 3 / 6.0],
                    [dt ** 4 / 8.0, dt ** 3 / 3.0, dt ** 2 / 2.0],
                    [dt ** 3 / 6.0, dt ** 2 / 2.0, dt],
                ],
                dtype=np.float64,
            )
            for axis in (0, 1):
                idx = [axis, axis + 2, axis + 4]
                Q[np.ix_(idx, idx)] = block * self.q
        return Q

    # -- 基本操作 ----------------------------------------------------------
    def initialize(self, z: np.ndarray, velocity: np.ndarray | None = None) -> None:
        self.x[:] = 0.0
        self.x[0:2] = z
        if velocity is not None:
            self.x[2:4] = velocity
        self.P = np.eye(self.dim, dtype=np.float64)
        self.P[0, 0] = self.P[1, 1] = self.r
        self.P[2, 2] = self.P[3, 3] = 1e5   # 速度未知，先给大不确定度
        if self.order == 2:
            self.P[4, 4] = self.P[5, 5] = 1e7
        self.initialized = True

    def predict(self, dt: float) -> np.ndarray:
        if not self.initialized or dt <= 0:
            return self.x.copy()
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)
        return self.x.copy()

    def innovation(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """新息与它的协方差 S，用来算马氏距离做野点门控。"""
        y = np.asarray(z, dtype=np.float64) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        return y, S

    def mahalanobis(self, z: np.ndarray) -> float:
        y, S = self.innovation(z)
        try:
            return float(y @ np.linalg.solve(S, y))
        except np.linalg.LinAlgError:
            return float("inf")

    def update(self, z: np.ndarray) -> float:
        z = np.asarray(z, dtype=np.float64)
        if not self.initialized:
            self.initialize(z)
            return 0.0
        y, S = self.innovation(z)
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return float("inf")
        self.x = self.x + K @ y
        I = np.eye(self.dim)
        # Joseph 形式，数值上更稳定
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ self.R @ K.T
        try:
            return float(y @ np.linalg.solve(S, y))
        except np.linalg.LinAlgError:
            return float("inf")

    def adapt_noise(self, d2: float, gate: float) -> None:
        """按新息大小自适应调整过程噪声。

        新息异常大（模型跟不上）-> 临时放大 Q，让滤波器敢信观测；
        一段时间正常后再慢慢收回来，恢复平滑。
        """
        if d2 > gate:
            self.q = min(self.q * 3.0, self.q_max)
        else:
            self.q = max(self.q_base, self.q * 0.97)

    def reinitialize(self, z: np.ndarray, v: np.ndarray) -> None:
        """机动（反弹/击打）后重新初始化滤波器。

        为什么不能只改速度：检测到机动时，滤波器已经用"错误的老速度"
        把位置外推到了几十像素之外；此时协方差 P 又很小，
        一次观测更新只能拉回 5%，要十几帧才能恢复 —— 这十几帧的位置
        和轨迹全是错的。正确做法是直接把状态重置到观测量上，并放大 P，
        用"承认模型失效"换取一帧之内恢复。
        """
        self.x[:] = 0.0
        self.x[0:2] = z
        self.x[2:4] = v
        self.P = np.eye(self.dim, dtype=np.float64)
        self.P[0, 0] = self.P[1, 1] = self.r
        self.P[2, 2] = self.P[3, 3] = 1e4
        if self.order == 2:
            self.P[4, 4] = self.P[5, 5] = 1e6
        self.q = self.q_max

    @property
    def position(self) -> np.ndarray:
        return self.x[0:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:4].copy()

    @property
    def acceleration(self) -> np.ndarray:
        if self.order == 2:
            return self.x[4:6].copy()
        return np.zeros(2, dtype=np.float64)


class Track:
    """一条球体轨迹。"""

    def __init__(self, track_id: int, cfg: TrackerConfig, t: float, det: Detection) -> None:
        self.id = track_id
        self.cfg = cfg
        self.kf = KalmanFilter(cfg.order, cfg.process_noise, cfg.measurement_noise,
                               max_q_scale=cfg.max_q_scale)
        self.kf.initialize(np.array([det.x, det.y], dtype=np.float64))
        self.t = t
        self.radius = float(det.radius)
        self.score = float(det.score)
        self.hits = 1
        self.misses = 0
        self.last_hit_t = float(t)
        self.age = 1
        self.confirmed = cfg.init_hits <= 1
        self.coasting = False
        self.maneuvers = 0
        self.last_maneuver_t: float | None = None
        self.trail: deque[TrailPoint] = deque(maxlen=cfg.trail_length)
        self._init_points: list[tuple[float, float, float]] = [(t, det.x, det.y)]
        self._last_obs: tuple[float, float, float] | None = None
        self.trail.append(
            TrailPoint(t=t, x=float(det.x), y=float(det.y), radius=float(det.radius), score=float(det.score))
        )

    # -- 用最早几帧做一次多点初始化，得到初速度 ---------------------------
    def _bootstrap_velocity(self) -> None:
        pts = self._init_points[:4]
        if len(pts) < 2:
            return
        ts = np.array([p[0] for p in pts], dtype=np.float64)
        if ts[-1] - ts[0] <= 1e-6:
            return
        vx = np.polyfit(ts, np.array([p[1] for p in pts]), 1)[0]
        vy = np.polyfit(ts, np.array([p[2] for p in pts]), 1)[0]
        self.kf.x[0] = pts[-1][1]
        self.kf.x[1] = pts[-1][2]
        self.kf.x[2] = vx
        self.kf.x[3] = vy
        self.kf.P[2, 2] = self.kf.P[3, 3] = 5e4

    def predict_to(self, t: float) -> None:
        dt = max(t - self.t, 1e-6)
        self.kf.predict(dt)
        self.t = t

    def update(self, det: Detection, t: float) -> None:
        z = np.array([det.x, det.y], dtype=np.float64)

        # ---- 机动检测：速度是不是突然变了？ ----------------------------
        # 反弹/被球拍打中的那一帧，用最近两次观测算出来的速度和滤波器里的
        # 速度会差出上千像素/秒，这就是"急转弯"的信号。
        if self._last_obs is not None and self.hits >= 3:
            t_prev, x_prev, y_prev = self._last_obs
            dt = t - t_prev
            if dt > 1e-6:
                v_meas = (z - np.array([x_prev, y_prev], dtype=np.float64)) / dt
                v_kf = self.kf.velocity
                jump = float(np.linalg.norm(v_meas - v_kf))
                if jump > max(self.cfg.maneuver_min_px_s, self.cfg.maneuver_ratio * float(np.linalg.norm(v_kf))):
                    self.kf.reinitialize(z, v_meas)
                    self.maneuvers += 1
                    self.last_maneuver_t = t

        d2 = self.kf.update(z)
        self.kf.adapt_noise(d2, self.cfg.gate_chi2)
        self._last_obs = (t, float(det.x), float(det.y))
        if len(self._init_points) < 8:
            self._init_points.append((t, det.x, det.y))
        self.hits += 1
        self.misses = 0
        self.last_hit_t = float(t)      # ← 时间口径的"最后一次命中"（见 `dead`）
        self.coasting = False
        # 半径用指数滑动平均，抗单帧的半径抖动。
        # 别取太小：半径决定了单目测深的 Z，滞后会直接变成深度误差。
        alpha = 0.5
        self.radius = (1.0 - alpha) * self.radius + alpha * float(det.radius)
        self.score = float(det.score)
        if self.hits >= self.cfg.init_hits and not self.confirmed:
            self._bootstrap_velocity()
            self.confirmed = True
        if self.cfg.trail_source == "measurement":
            px, py = float(det.x), float(det.y)
        else:
            pos = self.kf.position
            px, py = float(pos[0]), float(pos[1])
        self.trail.append(
            TrailPoint(t=t, x=px, y=py, radius=float(det.radius), score=float(det.score))
        )

    def coast(self, t: float) -> None:
        """本帧没有匹配到检测：用预测位置续上（短暂遮挡/漏检时非常有用）。"""
        self.misses += 1
        self.coasting = True
        pos = self.kf.position
        self.trail.append(
            TrailPoint(t=t, x=float(pos[0]), y=float(pos[1]), radius=self.radius, score=0.0)
        )

    @property
    def dead(self) -> bool:
        """这条轨迹该删了吗？

        ## ⚠️ 2026-10-18 修：原来只数**帧数**，于是遮挡容忍度随帧率变

        `max_coast_frames = 12` 是**帧**。同一段 0.3 s 的遮挡：

            30 fps -> 9 帧  -> 轨迹**延续**（id 1->1）
            60 fps -> 18 帧 -> 轨迹**断开**（id 1->2，起了一条新轨迹）

        也就是说**帧率翻倍、遮挡容忍度减半**。本项目两路常跑 30/60 fps
        （取决于后端与分辨率），于是同一套配置在不同帧率下行为不同 ——
        而"轨迹断开"意味着拖尾/预测历史清零，跟踪连续性静默变差。

        修法：优先用**时间口径** `max_coast_s`（秒），与帧率无关；
        `max_coast_frames` 保留作兜底（老配置 / 不知道 fps 时）。
        """
        if self.cfg.max_coast_s > 0:
            return (self.t - self.last_hit_t) > self.cfg.max_coast_s
        return self.misses > self.cfg.max_coast_frames

    def state(self) -> TrackState:
        pos = self.kf.position
        vel = self.kf.velocity
        acc = self.kf.acceleration
        return TrackState(
            track_id=self.id, t=self.t,
            x=float(pos[0]), y=float(pos[1]),
            vx=float(vel[0]), vy=float(vel[1]),
            ax=float(acc[0]), ay=float(acc[1]),
            radius=float(self.radius), confirmed=self.confirmed,
            coasting=self.coasting, missed=self.misses, age=self.age, hits=self.hits,
            maneuvers=self.maneuvers, last_maneuver_t=self.last_maneuver_t,
        )


class BallTracker:
    """多目标最近邻关联 + 卡尔曼滤波。

    单球场景下你只关心 primary 轨迹；但保留多目标能力，
    可以顺便处理"镜子里还有一个球"或者"桌上还有个球"的情况。
    """

    def __init__(self, cfg: TrackerConfig) -> None:
        self.cfg = cfg
        self.tracks: list[Track] = []
        self._next_id = 1
        self.primary_id: int | None = None

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1
        self.primary_id = None

    def update(self, detections: list[Detection], t: float) -> Track | None:
        for tr in self.tracks:
            tr.predict_to(t)
            tr.age += 1

        pairs: list[tuple[float, int, int]] = []
        for ti, tr in enumerate(self.tracks):
            for di, det in enumerate(detections):
                z = np.array([det.x, det.y], dtype=np.float64)
                # 半径一致性门控：同一个球，像素半径不会突然变化 35% 以上。
                # 少了这一条，球被短暂遮挡后跟踪器很容易"跳"到画面里另一个
                # 尺寸不同的物体上（镜面高光、另一个球、屏幕上的圆形图标…）。
                if tr.radius > 1.0:
                    ratio = abs(float(det.radius) - tr.radius) / tr.radius
                    if ratio > self.cfg.radius_gate_ratio and tr.hits >= self.cfg.init_hits:
                        continue
                d2 = tr.kf.mahalanobis(z)
                if d2 > self.cfg.gate_chi2 and tr.hits >= self.cfg.init_hits:
                    continue  # 门控不过，判为野点
                dist = float(np.hypot(tr.kf.position[0] - det.x, tr.kf.position[1] - det.y))
                pairs.append((dist, ti, di))
        pairs.sort(key=lambda p: p[0])

        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for _, ti, di in pairs:
            if ti in used_tracks or di in used_dets:
                continue
            used_tracks.add(ti)
            used_dets.add(di)
            self.tracks[ti].update(detections[di], t)

        # ---- 抢救关联 ----
        # 球落地反弹或被击打的瞬间，速度方向突变，卡尔曼的"位置门控"会把
        # 真实观测当成野点拒掉。如果这里直接新建轨迹，轨迹就被打断了，
        # 后面几个点的速度/加速度全都估不准。
        # 所以再加一轮宽松关联：只要离预测位置还不算离谱，就还给原来那条轨迹。
        rescue: list[tuple[float, int, int]] = []
        for ti, tr in enumerate(self.tracks):
            if ti in used_tracks:
                continue
            for di, det in enumerate(detections):
                if di in used_dets:
                    continue
                limit = max(self.cfg.rescue_min_px, self.cfg.rescue_radius_ratio * det.radius)
                dist = float(np.hypot(tr.kf.position[0] - det.x, tr.kf.position[1] - det.y))
                if tr.radius > 1.0 and tr.hits >= self.cfg.init_hits:
                    if abs(float(det.radius) - tr.radius) / tr.radius > self.cfg.radius_gate_ratio:
                        continue
                if dist <= limit:
                    rescue.append((dist, ti, di))
        rescue.sort(key=lambda p: p[0])
        for _, ti, di in rescue:
            if ti in used_tracks or di in used_dets:
                continue
            used_tracks.add(ti)
            used_dets.add(di)
            # 关联上了，但门控确实没过 -> 说明模型跟不上，标记为可疑机动
            self.tracks[ti].kf.q = self.tracks[ti].kf.q_max
            self.tracks[ti].update(detections[di], t)

        for ti, tr in enumerate(self.tracks):
            if ti not in used_tracks:
                tr.coast(t)

        for di, det in enumerate(detections):
            if di in used_dets:
                continue
            self.tracks.append(Track(self._next_id, self.cfg, t, det))
            self._next_id += 1

        self.tracks = [tr for tr in self.tracks if not tr.dead]

        # 选择主轨迹：优先已确认、当前有观测、命中多、分数高
        def rank(tr: Track) -> tuple:
            return (tr.confirmed, not tr.coasting, tr.hits, tr.score, tr.radius)

        if self.tracks:
            best = max(self.tracks, key=rank)
            self.primary_id = best.id
            return best
        self.primary_id = None
        return None

    @property
    def primary(self) -> Track | None:
        for tr in self.tracks:
            if tr.id == self.primary_id:
                return tr
        return self.tracks[0] if self.tracks else None


# ---------------------------------------------------------------------------
# 注册表（R1：显式注册，import 本模块时执行）
#
# 为什么注册在**本模块末尾**、而不是单开 `trackers/registry.py`：
# 现在只有一种实现，独立注册表模块的唯一作用是"多一次 import 才能分派"，
# 而漏掉那次 import 的失败模式很难看出来（R1 的注解就是为它写的）。
# 真出现第二种跟踪器时再拆目录，那时分文件才有收益（与 detectors 同理）。
# ---------------------------------------------------------------------------
@plugin("tracker", "ball",
        summary="卡尔曼 + 最近邻关联（默认；支持多球、短暂遮挡后延续同一轨迹）",
        default=True)
def _build_ball_tracker(cfg: TrackerConfig) -> BallTracker:
    """`tracker/ball` 的工厂：签名统一为 `factory(cfg) -> 实现对象`。"""
    return BallTracker(cfg)
