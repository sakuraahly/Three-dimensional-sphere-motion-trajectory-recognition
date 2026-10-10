"""运行健康度指标：一个对象，给 HUD、日志、测试共用。

为什么要单独一个模块
--------------------
项目里"帧率"这个词同时指三件不同的事，**已经因此走错过一次方向**
（`交接文档.md` §10.10：界面上那个 `detect 19 ms/frame (52 fps)` 量的是
**检测计算耗时**，被读成"相机采集帧率"，于是整轮排查都在查错方向）：

| 指标 | 到底在量什么 | 掉的常见原因 |
| ---- | ------------ | ------------ |
| `capture_fps` | 采集线程**从驱动拿到**多少帧/秒 | USB 带宽、相机曝光、占用同一总线 |
| `loop_fps`    | 主循环**处理完**多少帧/秒 | 检测太慢（算法） |
| `detect_ms`   | 单次检测耗时（可分段） | 边缘密度、分辨率 |

三者**必须分开报**，而且**必须同名同值** —— 之前 HUD、日志、测试各算一套，
于是同一个现象在三处叫三个名字。

还有两个"延迟类"指标，它们回答的是完全不同的问题：

- `frame_age_ms`：**用户看到的这帧有多老**。相机采集比处理快时，主循环
  永远在消费几帧之前的画面（`LatestFrameCamera` 的 docstring 描述过这个现象：
  "你看到的是几百毫秒前的画面"）。
- `dropped`：本帧**在被取走之前被下一帧覆盖**的次数。
  注意现有实现（`LatestFrameCamera`）是**锁内原地覆盖、没有队列**，
  所以"丢"是"被覆盖"，不是"进队列失败"。

本模块**刻意只做数据**（全局 §10.1：先定职能与边界）：
没有 IO、不 import 本项目任何其他模块、不 print。装 handler、画到屏幕上、
写进日志，都是调用方的事。

用法::

    h = Health()
    h.note_capture(capture_ms=33.3, frame_age_ms=120.0)
    h.begin_frame()
    ...检测...
    h.end_frame(detect_ms=12.3)
    line = h.summary_line()
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

__all__ = ["Health", "StageTimes"]


@dataclass
class StageTimes:
    """分段耗时（毫秒）。由调用方喂进来，本类只存。"""

    detect_ms: float = 0.0            # 检测（多路时取最慢或求和，由调用方决定）
    track_ms: float = 0.0             # 跟踪 / 状态估计
    render_ms: float = 0.0            # 绘制

    def clear(self) -> None:
        self.detect_ms = self.track_ms = self.render_ms = 0.0

    @property
    def total_ms(self) -> float:
        return self.detect_ms + self.track_ms + self.render_ms


@dataclass
class Health:
    """一次运行的健康度累加器。**纯数据**，可安全序列化。

    `capture_fps` / `loop_fps` / `drop_rate` 是**派生量**（用属性算），
    刻意**不做成字段** —— 做成字段就会有两个真相（一个是累加出来的，
    一个是反序列化读回来的），而"同一个量在两处各算一遍"正是本模块要消除的问题。
    """

    # ---- 采集侧 ----
    captured_frames: int = 0          # 采集线程拿到的帧数
    dropped_frames: int = 0           # 被下一帧覆盖、从未被取走的帧数
    capture_ms: float = 0.0           # 最近一帧的采集间隔
    frame_age_ms: float = 0.0         # 最近一帧"从采集到被取走"的年龄

    # ---- 处理侧 ----
    processed_frames: int = 0
    process_ms: float = 0.0           # 最近一帧的整帧处理耗时
    stages: StageTimes = field(default_factory=StageTimes)

    # ---- 双目专有 ----
    skew_ms: float = 0.0              # 左右两路的时间偏差估计（>0 = 右路更晚）
    skew_ms_valid: bool = False       # 没估出来时为 False，别把 0 当真值

    # ------------------------------------------------------------------
    # 累加
    # ------------------------------------------------------------------
    def begin_frame(self) -> None:
        """一帧开始：清掉上一帧的分段耗时，避免读到陈旧值。"""
        self.stages.clear()

    def end_frame(self, detect_ms: float = 0.0, track_ms: float = 0.0,
                  render_ms: float = 0.0, process_ms: float | None = None) -> None:
        """一帧结束。`process_ms` 不给就用分段之和。"""
        self.stages.detect_ms = float(detect_ms)
        self.stages.track_ms = float(track_ms)
        self.stages.render_ms = float(render_ms)
        self.process_ms = float(process_ms) if process_ms is not None else self.stages.total_ms
        self.processed_frames += 1

    def note_capture(self, capture_ms: float, frame_age_ms: float = 0.0) -> None:
        """采集线程每拿到一帧调一次。

        **这里不动 `dropped_frames`** —— 拿到一帧不等于丢了一帧。
        覆盖发生在"还没人取走它，下一帧就来了"的时候，只有采集线程知道，
        由 `note_dropped()` 显式上报。
        """
        self.captured_frames += 1
        self.capture_ms = float(capture_ms)
        self.frame_age_ms = float(frame_age_ms)

    def note_dropped(self, n: int = 1) -> None:
        """本帧在被取走之前被下一帧覆盖 —— 只能在采集线程里加。"""
        self.dropped_frames += max(0, int(n))

    # ------------------------------------------------------------------
    # 派生量
    # ------------------------------------------------------------------
    @staticmethod
    def _rate(frames: int, ms_per_frame: float) -> float:
        if frames <= 0 or ms_per_frame <= 0:
            return 0.0
        return 1000.0 / ms_per_frame

    @property
    def capture_fps(self) -> float:
        """采集帧率。用**帧间隔**算 —— 不用总时间/总帧数，那会被启动阶段拉低。"""
        return self._rate(self.captured_frames, self.capture_ms)

    @property
    def loop_fps(self) -> float:
        """主循环处理帧率。这是 HUD 上那个数，**不是采集帧率**。"""
        return self._rate(self.processed_frames, self.process_ms)

    @property
    def drop_rate(self) -> float:
        """丢帧比例（0~1）。没有任何采集帧时为 0。"""
        total = self.captured_frames + self.dropped_frames
        if total <= 0:
            return 0.0
        return self.dropped_frames / total

    # ------------------------------------------------------------------
    # 呈现与序列化
    # ------------------------------------------------------------------
    def summary_line(self) -> str:
        """一行给人看、且**每个名字都唯一**（不再出现裸的 `FPS`）。"""
        parts = [
            f"cap {self.capture_fps:5.1f}",
            f"loop {self.loop_fps:5.1f}",
            f"detect {self.stages.detect_ms:6.1f} ms",
            f"age {self.frame_age_ms:5.1f} ms",
            f"drop {self.dropped_frames}",
        ]
        if self.skew_ms_valid:
            parts.append(f"skew {self.skew_ms:+6.1f} ms")
        return "  ".join(parts)

    def to_dict(self) -> dict:
        """快照。派生量也在里面，但那是**只读**的方便字段，`from_dict` 会丢弃它们。"""
        d = asdict(self)
        d["capture_fps"] = round(self.capture_fps, 3)
        d["loop_fps"] = round(self.loop_fps, 3)
        d["drop_rate"] = round(self.drop_rate, 4)
        d["process_ms"] = round(self.process_ms, 3)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Health":
        """与 `to_dict` 往返一致。

        ⚠️ **只认 dataclass 字段** —— 派生量（`capture_fps` / `loop_fps` / `drop_rate`）
        必须在反序列化时**丢弃**，否则它们会变成字段，从此有两个真相。
        """
        stages = StageTimes(**(data.get("stages") or {}))
        known = {f for f in cls.__dataclass_fields__}   # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in known and k != "stages"}
        return cls(stages=stages, **kwargs)
