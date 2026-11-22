"""尺寸门（最小/最大半径）的自动阶梯 —— "先试一档，不行再跳"。

为什么需要（2026-11-08，梨子实验）
--------------------------------
`min_radius=8 / max_radius=200` 是**针对某个拍摄距离写死的先验**。梨子凑近到
25 cm 时半径 205~212 px，刚好越过 200 → 检测器 0 检出，而它给出的理由是
「**球可能太小**」—— 方向正好相反。用户在这一轮把这件事说清楚了：
「尺寸门也是要变成自动阶梯跳变的参数吧，就像打开相机一样」。

参照物就是相机分辨率协商（`stereo/pair_source.AUTO_*`）：从保守档出发 →
每档按**可量化判据**测 → 命中就锁死 → 逐档留证据 → 走完还没命中就说清
"走完了"。这里搬的是**形式**，不是新发明一套。

判据（与那套同构，数值都可复核）
------------------------------
- **档位** `r_max = k·H`，k ∈ `SIZE_LADDER_K`。用画面高度比例而不是绝对像素：
  1080p 的 0.18 与 720p 的 0.18 是同一个视场含义。0.18·1080 = 194 ≈ 现有 200。
- **命中**：探针窗口内「这一帧有检出」的占比 ≥ `MIN_HIT_RATIO`（0.6，与
  `AUTO_MIN_PAIR_FPS = DEFAULT_FPS*0.6` 同一个"六成"口径）。
- **升档的唯一理由**：窗口里没检出，**但门外有被尺寸门单独挡住的候选**
  （`blocked_r > r_max`）。这条来自检测器的 `_diag["r_blocked"]`：
  只统计"第一道挡路的门就是半径门"的轮廓 —— 被圆度/填充率挡掉的候选
  不会让阶梯升档，因为放大窗口救不了形状。
- **锁死**：命中后整段不变。中途改门限会让同一段轨迹的物理口径前后不一致，
  比一开始选错更难查。
- **放弃**：门外也没有更大的候选（掩膜在背景上、形状全不过）→ 立即
  `no_candidate` 并说明"与尺寸门无关"；走完阶梯仍未命中 → `exhausted`。

不做的事
--------
- 不缩 `min_radius`：它是**去噪**用的，与拍摄距离无关，放宽只会放大误检。
- 不做"每帧跳档"：那是抖动，不是自适应。
- 不用 `max_radius=5000` 这种"关掉门"的做法（`train_ball_detector` 是训练场景特例）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: 阶梯档位（相对画面高度的最大半径）。
SIZE_LADDER_K: tuple[float, ...] = (0.18, 0.30, 0.45, 0.65)
#: 探针窗口内"有检出"的帧占比达到多少算命中（同相机那套的"六成"）。
MIN_HIT_RATIO = 0.6
#: 每个档位看多少帧再裁决。
PROBE_FRAMES = 12


@dataclass
class LadderAttempt:
    """一档的裁决记录（用于日志与判据）。"""

    index: int
    k: float
    r_max: float
    frames: int = 0
    hit_frames: int = 0
    blocked_max: float | None = None
    verdict: str = "probing"      # probing / hit / clipped / miss / empty

    @property
    def hit_ratio(self) -> float:
        return (self.hit_frames / self.frames) if self.frames else 0.0

    def line(self) -> str:
        blocked = "—" if self.blocked_max is None else f"{self.blocked_max:.0f}px"
        return (f"第 {self.index + 1} 档 r_max={self.r_max:.0f}px（k={self.k:.2f}）："
                f"{self.hit_frames}/{self.frames} 帧有检出"
                f"（{self.hit_ratio:.0%}），门外最大候选 {blocked} → {self.verdict}")


@dataclass
class SizeLadder:
    """尺寸门阶梯状态机。**纯逻辑、不碰 cv2**，方便单测与反向自证。"""

    image_h: float
    min_radius: float = 8.0
    probe_frames: int = PROBE_FRAMES
    steps: tuple[float, ...] = SIZE_LADDER_K
    hit_ratio: float = MIN_HIT_RATIO
    index: int = 0
    state: str = "probing"        # probing / locked / exhausted / no_candidate
    attempts: list[LadderAttempt] = field(default_factory=list)
    _hits: int = 0
    _frames: int = 0
    _blocked: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.image_h <= 0:
            raise ValueError(f"image_h 必须是正的，收到 {self.image_h!r}")
        if not self.steps:
            raise ValueError("steps 不能为空")
        self._start_attempt()

    # ---------------- 只读 ---------------- #
    @property
    def r_max(self) -> float:
        k = self.steps[min(self.index, len(self.steps) - 1)]
        return round(float(self.image_h) * k, 1)

    @property
    def window(self) -> tuple[float, float]:
        """当前档位的 `(min_radius, r_max)`，直接喂给检测器。"""
        return (float(self.min_radius), self.r_max)

    @property
    def locked(self) -> bool:
        return self.state == "locked"

    @property
    def done(self) -> bool:
        """协商结束（无论成功还是放弃）。"""
        return self.state in ("locked", "exhausted", "no_candidate")

    # ---------------- 状态机 ---------------- #
    def _start_attempt(self) -> None:
        k = self.steps[min(self.index, len(self.steps) - 1)]
        self.attempts.append(LadderAttempt(index=self.index, k=k,
                                           r_max=round(float(self.image_h) * k, 1)))

    def _current(self) -> LadderAttempt:
        return self.attempts[-1]

    def _next_step(self) -> str:
        if self.index + 1 >= len(self.steps):
            self.state = "exhausted"
            return "exhausted"
        self.index += 1
        self._hits = 0
        self._frames = 0
        self._blocked = []
        self._start_attempt()
        return "step"

    def observe(self, detected: bool, blocked_r: float | None = None) -> str:
        """喂一帧的两个事实，返回状态串。

        - ``detected``：这一帧**有没有检出**（`len(dets) > 0`）。
        - ``blocked_r``：这一帧被**尺寸门单独挡住**的最大候选半径
          （检测器的 `_diag["r_blocked"]`；没有就 None）。

        返回：``probing`` / ``step`` / ``hit`` / ``locked`` / ``exhausted`` /
        ``no_candidate``。调用方只需认这几种。
        """
        if self.done:
            return self.state

        self._frames += 1
        if detected:
            self._hits += 1
        if blocked_r is not None:
            self._blocked.append(float(blocked_r))

        cur = self._current()
        cur.frames = self._frames
        cur.hit_frames = self._hits
        cur.blocked_max = max(self._blocked) if self._blocked else None

        if self._frames < self.probe_frames:
            cur.verdict = "probing"
            return "probing"

        if cur.hit_ratio >= self.hit_ratio:
            cur.verdict = "hit"
            self.state = "locked"
            return "hit"

        # 没检出，且门外有更大的候选 → 多半是窗口太小，升一档。
        if cur.blocked_max is not None and cur.blocked_max > cur.r_max:
            cur.verdict = "clipped"
            return self._next_step()

        # 门外也没有更大的 → 不是尺寸门的事（掩膜/形状/预设），立刻收手。
        # 否则会一路放大到 0.65·H，把白墙当球。
        cur.verdict = "empty" if cur.blocked_max is None else "miss"
        self.state = "no_candidate"
        return "no_candidate"

    # ---------------- 人话 ---------------- #
    def describe(self) -> str:
        if self.state == "locked":
            a = self._current()
            return (f"锁在第 {a.index + 1}/{len(self.steps)} 档："
                    f"{self.min_radius:.0f}–{a.r_max:.0f}px（画面高 {self.image_h:.0f}px）")
        if self.state == "no_candidate":
            return ("尺寸门与本次 0 检出**无关**：门外没有更大的候选 —— "
                    "先查掩膜落在哪（预设/取色）、圆度/填充率，别指望放大半径能救")
        if self.state == "exhausted":
            return (f"{len(self.steps)} 档走完仍未命中"
                    f"（{self.steps[0]:.2f}~{self.steps[-1]:.2f}·H）："
                    "球比最后一档还大，或者根本没稳定检出")
        return f"协商中：第 {self.index + 1} 档 r_max={self.r_max:.0f}px"

    def history_lines(self) -> list[str]:
        """逐档证据，与相机那套"逐档记录"同一个用途。"""
        return [a.line() for a in self.attempts]
