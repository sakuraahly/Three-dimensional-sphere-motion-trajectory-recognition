"""二维像素层的多目标关联：多球跟踪 + 遮挡外推 + 半径法重识别。

## 这里**不做三维**

本模块只吃"每帧若干 (x, y) 像素检测"，只吐"这条轨迹这帧在哪"。
三角测量、深度、世界坐标变换全都不在这里 —— 那是 `stereo/` 的事。
把三维塞进来会让本模块同时依赖标定与配对逻辑，反向依赖一旦形成就拆不开
（AGENTS.md §5）。要三维，就在上层把每路的结果按 tid 对齐后再三角化。

## 三个机制的边界（都是刻意的取舍）

1. **遮挡外推**：`missing < max_missing` 时按 `const_velocity` 线性外推，
   只对外报"预测位置"，不伪造观测 —— `positions` 里缺失帧是 `None`。
   为什么只有匀速：球在二维画面里做抛物线/反弹时匀速模型会飘，
   而"不飘的模型"需要三维重力或相机内参，本模块拿不到。宁可诚实外推。

2. **半径法重识别**：轨迹丢了一段时间后，新检测落在 `reid_radius_px` 内
   就沿用原 `tid`。这**只利用了位置**，所以：

   - 两球靠近到同一个半径圈里时，**可能互相认错**（A 认成 B）；
   - 球离开画面再从别处进来、但距离 > `reid_radius_px` 时，**无法认回**；
   - 遮挡越久，外推位置越不可信，半径门限实际上在替外推误差兜底。

   **要彻底解决必须有外观特征**（颜色直方图 / 深度学习 embedding / 球面纹理），
   而本项目当前检测器只给圆心和半径，没有外观描述子。这条留在代码里，
   是为了让身份错误可被统计（`n_reid`、`swap_events`），而不是假装没有。

3. **身份互换的防护**：两球交叉时，如果"交换后总代价更低"，本模块允许交换
   分配，并把 `swap_events += 1` 记到两条轨迹上。**这不是修复，是记账。**
   二维位置信息在两球重合附近本质上是退化的（两条轨迹的预测位置几乎一样，
   谁配谁代价都相同），任何纯几何关联都无法区分 —— 我选择让最优匹配胜出
   并显式记录发生了交换，而不是用"保持原 id"掩盖一次错误的观测归属。

## 关联方式

全局最近邻 + 门限（贪心，按距离升序一对一分配），不引入 scipy / 匈牙利算法：
本场景目标数是个位数，贪心与最优解在门限外无差别，而贪心的行为可逐步复现。

用法::

    tr = MultiTracker(TrackConfig())
    for t, dets in stream:            # dets: [(x, y), ...]
        tracks = tr.update(t, dets)
    print(tr.stats())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import Sequence

__all__ = ["TrackConfig", "Track", "MultiTracker"]


@dataclass
class TrackConfig:
    """关联与生命周期参数（全部是像素 / 帧口径，与相机分辨率绑定）。"""

    gate_px: float = 40.0            # 常规关联门限（预测位置 <-> 新检测）
    max_missing: int = 15            # 连续丢失多少帧后删轨
    predict_kind: str = "const_velocity"   # 目前只实现匀速外推
    reid_radius_px: float = 120.0    # 重识别半径（大于常规门限，替外推误差兜底）
    reid_after_missing: int = 10     # 丢失达到这个帧数后，再命中就算一次重识别
    min_hits: int = 3                # 未匹配检测要连续出现几次才建轨（挡住噪声）


@dataclass
class Track:
    """一条二维轨迹。

    `positions` 是**逐帧**表：命中帧是 `(t, x, y)`，丢失帧是 `None`。
    建轨前的确认点（pending 阶段的连续检测）会回填到最前面，所以
    `len(positions)` 可能略大于"建轨后经过的帧数" —— 但它们都是真实观测。

    `predicted` 是最近一次 `update()` 后的外推位置（遮挡期间才有意义），
    **不是**观测：不要拿它当位置真值喂给下游。
    """

    tid: int
    positions: list[tuple[float, float, float] | None] = field(default_factory=list)
    state: str = "active"            # active / lost / dead
    hits: int = 0                    # 真实观测次数
    missing: int = 0                 # 连续丢失帧数
    last_seen: float = 0.0           # 最后一次命中的时刻
    velocity: tuple[float, float] = (0.0, 0.0)   # 像素/秒
    predicted: tuple[float, float] | None = None
    reid_events: int = 0             # 本条轨迹被重识别接回的次数
    swap_events: int = 0             # 本条轨迹卷入身份交换的次数

    @property
    def last_observation(self) -> tuple[float, float, float] | None:
        for p in reversed(self.positions):
            if p is not None:
                return p
        return None

    @property
    def length(self) -> int:
        """真实观测点数（不含 `None` 空洞）。"""
        return sum(1 for p in self.positions if p is not None)

    # -- 内部状态推进 ----------------------------------------------------
    def _refresh_velocity(self) -> None:
        """用最后两个观测差分速度。

        只取两点是刻意的：这是"关联用的外推速度"，不是要交付的速度估计。
        多点拟合会把遮挡前的运动混进来，反弹后反而更不准。
        代价是单帧噪声会直接进速度 —— 检测本身抖 1-2 px 时，外推 10 帧后
        会偏出 10-20 px，这正是 `reid_radius_px` 要兜住的那部分。
        """
        obs = [p for p in self.positions if p is not None]
        if len(obs) < 2:
            return
        t0, x0, y0 = obs[-2]
        t1, x1, y1 = obs[-1]
        dt = t1 - t0
        if dt <= 1e-9:
            return
        self.velocity = ((x1 - x0) / dt, (y1 - y0) / dt)

    def observe(self, t: float, xy: tuple[float, float], is_reid: bool = False) -> None:
        self.positions.append((float(t), float(xy[0]), float(xy[1])))
        self.missing = 0
        self.state = "active"
        self.last_seen = float(t)
        self.hits += 1
        if is_reid:
            self.reid_events += 1
        self._refresh_velocity()
        self.predicted = (float(xy[0]), float(xy[1]))

    def miss(self, t: float, cfg: TrackConfig) -> None:
        self.missing += 1
        self.positions.append(None)
        self.state = "dead" if self.missing >= cfg.max_missing else "lost"


@dataclass
class _Pending:
    """还没攒够 `min_hits` 的候选目标（噪声在这一层就被挡掉）。"""

    positions: list[tuple[float, float, float]] = field(default_factory=list)


class MultiTracker:
    """多目标跟踪器：一对一贪心关联 + 匀速外推 + 半径法重识别。"""

    def __init__(self, cfg: TrackConfig | None = None) -> None:
        self.cfg = cfg or TrackConfig()
        if self.cfg.predict_kind != "const_velocity":
            raise ValueError(
                f"predict_kind={self.cfg.predict_kind!r} 未实现，只支持 'const_velocity'"
            )
        self.tracks: list[Track] = []
        self._pending: list[_Pending] = []
        self._next_id = 1
        self._n_created = 0
        self._n_reid = 0
        self._n_swap = 0

    # -- 对外入口 --------------------------------------------------------
    def update(self, t: float, detections: Sequence[tuple[float, float]]) -> list[Track]:
        """喂一帧检测，返回当前存活的轨迹（按创建顺序）。"""
        cfg = self.cfg
        dets = [(float(x), float(y)) for x, y in detections]

        for tr in self.tracks:
            self._predict(tr, t)

        pairs = self._resolve_swaps(self._associate(dets), dets)
        used_t = {p[0] for p in pairs}
        used_d = {p[1] for p in pairs}

        for ti, di, _dist, is_reid in pairs:
            tr = self.tracks[ti]
            tr.observe(t, dets[di], is_reid=is_reid)
            if is_reid:
                self._n_reid += 1

        for ti, tr in enumerate(self.tracks):
            if ti not in used_t:
                tr.miss(t, cfg)

        self.tracks = [tr for tr in self.tracks if tr.state != "dead"]

        self._track_pending(t, [d for di, d in enumerate(dets) if di not in used_d])
        return list(self.tracks)

    def stats(self) -> dict[str, float]:
        """`n_tracks` 累计建轨数 / `n_active` 当前存活数 / 重识别 / 平均长度。

        `mean_track_len` 只统计**当前存活**的轨迹；已删除的轨迹长度不参与，
        因为"被删的短轨迹"会把均值拉低，反而看不出身份保持得好不好。
        """
        live = self.tracks
        return {
            "n_tracks": self._n_created,
            "n_active": len(live),
            "n_reid": self._n_reid,
            "n_swap": self._n_swap,
            "mean_track_len": (sum(tr.length for tr in live) / len(live)) if live else 0.0,
        }

    def reset(self) -> None:
        self.tracks.clear()
        self._pending.clear()
        self._next_id = 1
        self._n_created = 0
        self._n_reid = 0
        self._n_swap = 0

    # -- 预测 ------------------------------------------------------------
    def _predict(self, tr: Track, t: float) -> None:
        last = tr.last_observation
        if last is None:
            return
        t0, x0, y0 = last
        dt = max(float(t) - t0, 0.0)
        tr.predicted = (x0 + tr.velocity[0] * dt, y0 + tr.velocity[1] * dt)

    # -- 关联 ------------------------------------------------------------
    def _associate(self, dets: Sequence[tuple[float, float]]) -> list[list]:
        """按距离升序做一对一贪心分配，返回 `[ti, di, dist, is_reid]`。"""
        cfg = self.cfg
        cands: list[tuple[float, int, int, bool]] = []
        for ti, tr in enumerate(self.tracks):
            if tr.predicted is None:
                continue
            px, py = tr.predicted
            # 丢失够久 -> 走重识别半径；否则走常规门限。
            # 重识别放宽门限是必须的：外推 10 帧后匀速模型的误差
            # 常常已经超过 40 px，用常规门限等于"永远认不回来"。
            is_reid = tr.missing >= cfg.reid_after_missing
            limit = cfg.reid_radius_px if is_reid else cfg.gate_px
            for di, (x, y) in enumerate(dets):
                d = hypot(px - x, py - y)
                if d <= limit:
                    cands.append((d, ti, di, is_reid))

        cands.sort(key=lambda c: (c[0], c[1], c[2]))
        used_t: set[int] = set()
        used_d: set[int] = set()
        out: list[list] = []
        for d, ti, di, is_reid in cands:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            out.append([ti, di, d, is_reid])
        return out

    def _resolve_swaps(self, pairs: list[list], dets: Sequence[tuple[float, float]]) -> list[list]:
        """两两检查"交换是否更省"，省就换，并记账。

        为什么允许换：两球交叉时预测位置彼此靠近，直连代价与交换代价可能
        只差几个像素。坚持直连会把检测硬塞给已经飘走的那条轨迹，导致
        两条轨迹都跟不上 —— 与其自欺，不如取最优解并把 `swap_events` 暴露出来。
        **这治不了身份互换，只是让它可被观测。**
        """
        if len(pairs) < 2:
            return pairs
        cfg = self.cfg
        for i in range(len(pairs) - 1):
            for j in range(i + 1, len(pairs)):
                ti, di, d_i, reid_i = pairs[i]
                tj, dj, d_j, reid_j = pairs[j]
                pi = self.tracks[ti].predicted
                pj = self.tracks[tj].predicted
                if pi is None or pj is None:
                    continue
                cross_i = hypot(pi[0] - dets[dj][0], pi[1] - dets[dj][1])
                cross_j = hypot(pj[0] - dets[di][0], pj[1] - dets[di][1])
                # 交换后也必须各自过门限，否则等于把检测硬塞给不可信的轨迹
                lim_i = cfg.reid_radius_px if reid_i else cfg.gate_px
                lim_j = cfg.reid_radius_px if reid_j else cfg.gate_px
                if cross_i > lim_i or cross_j > lim_j:
                    continue
                if cross_i + cross_j + 1e-6 < d_i + d_j:
                    pairs[i] = [ti, dj, cross_i, reid_i]
                    pairs[j] = [tj, di, cross_j, reid_j]
                    self.tracks[ti].swap_events += 1
                    self.tracks[tj].swap_events += 1
                    self._n_swap += 1
        return pairs

    # -- 新目标确认 ------------------------------------------------------
    def _track_pending(self, t: float, leftover: Sequence[tuple[float, float]]) -> None:
        """未匹配检测先入 pending，连续出现 `min_hits` 次才转正。

        纯噪声（随机位置）几乎不可能连续 `min_hits` 帧都落在 `gate_px` 内，
        所以这一层就是"不许对噪声建轨"的实现。

        **光有"距离够近"还不够**：实测 640×480 画面、每帧 2 个均匀随机点、
        100 帧，仅靠 40 px 门限 + `min_hits=3` 会建出多条假轨（随机点
        两两相邻是常事）。所以从第 3 个点起改用"匀速外推位置"做基准，容差取
        `gate_px / 2`（20 px）——随机点要不偏不倚地落在外推点上，
        概率掉两个数量级；而真实匀速目标残差为 0，完全不受影响。

        残余风险要说清：100 个 seed 的扫描里仍有 6 个会漏建 1 条**短**假轨
        （3 帧命中后即丢失，很快被 `max_missing` 清掉）。这是纯几何确认的
        统计边界，不是实现缺陷 —— 要彻底消除得上外观特征，见模块 docstring。
        """
        cfg = self.cfg
        used: set[int] = set()
        survivors: list[_Pending] = []

        for p in self._pending:
            if not p.positions:
                continue
            px, py = p.positions[-1][1], p.positions[-1][2]
            limit = cfg.gate_px
            if len(p.positions) >= 2:
                t1, x1, y1 = p.positions[-2]
                t2, x2, y2 = p.positions[-1]
                dt1, dt2 = t2 - t1, float(t) - t2
                if dt1 > 1e-9 and dt2 > 1e-9:
                    px = x2 + (x2 - x1) / dt1 * dt2
                    py = y2 + (y2 - y1) / dt1 * dt2
                    limit = cfg.gate_px / 2.0
            best: tuple[float, int] | None = None
            for di, (x, y) in enumerate(leftover):
                if di in used:
                    continue
                d = hypot(px - x, py - y)
                if d <= limit and (best is None or d < best[0]):
                    best = (d, di)
            if best is None:
                continue          # 断一帧就作废：要的是"连续出现"
            used.add(best[1])
            x, y = leftover[best[1]]
            p.positions.append((float(t), x, y))
            survivors.append(p)

        for di, (x, y) in enumerate(leftover):
            if di not in used:
                survivors.append(_Pending(positions=[(float(t), x, y)]))

        keep: list[_Pending] = []
        for p in survivors:
            if len(p.positions) >= cfg.min_hits:
                self._spawn(p)
            else:
                keep.append(p)
        self._pending = keep

    def _spawn(self, p: _Pending) -> None:
        tr = Track(tid=self._next_id)
        self._next_id += 1
        self._n_created += 1
        tr.positions = list(p.positions)
        tr.hits = len(tr.positions)
        tr.last_seen = tr.positions[-1][0]
        tr.state = "active"
        tr._refresh_velocity()
        tr.predicted = (tr.positions[-1][1], tr.positions[-1][2])
        self.tracks.append(tr)
