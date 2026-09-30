"""数据容器。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FrameContext:
    """一帧的共享上下文。

    由 pipeline 每帧构造一次，传给所有检测后端，避免：
      - 每个后端各自 cvtColor 一遍灰度图（color/hough/亚像素细化都要灰度）；
      - 相机自运动被重复估计。
    检测器可以忽略它（默认 None），需要时再取。
    """

    t: float = 0.0
    gray: np.ndarray | None = None
    motion: Any = None            # camera_motion.MotionEstimate
    camera_moving: bool = False


@dataclass
class Detection:
    """单帧检测结果（像素坐标）。"""

    x: float
    y: float
    radius: float
    score: float = 1.0
    method: str = "unknown"
    # 每个后端各自的得分，便于调试融合过程
    parts: dict[str, float] = field(default_factory=dict)
    contour: np.ndarray | None = None
    mask: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x, self.y)

    @property
    def diameter(self) -> float:
        return 2.0 * self.radius


@dataclass
class TrailPoint:
    """轨迹上的一个历史点。"""

    t: float
    x: float
    y: float
    z: float | None = None
    radius: float = 0.0
    score: float = 0.0


@dataclass
class TrackState:
    """跟踪器对外输出的状态。"""

    track_id: int
    t: float
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    radius: float = 0.0
    z: float | None = None
    vz: float = 0.0
    confirmed: bool = False
    coasting: bool = False          # 本帧是"外推"出来的（没有观测）
    missed: int = 0
    age: int = 0
    hits: int = 0
    maneuvers: int = 0              # 检测到多少次速度突变（反弹/击打）
    last_maneuver_t: float | None = None   # 最近一次速度突变的时刻（秒）

    @property
    def speed_px(self) -> float:
        return float(np.hypot(self.vx, self.vy))


@dataclass
class FrameResult:
    """一帧的完整结果。"""

    t: float
    frame_index: int
    detection: Detection | None = None
    track: TrackState | None = None
    trail: list[TrailPoint] = field(default_factory=list)
    prediction: np.ndarray | None = None      # (N,2) 或 (N,3) 预测点
    prediction_3d: np.ndarray | None = None   # (N,3) 米
    position_3d: np.ndarray | None = None     # (3,) 米
    fps: float = 0.0
    detect_ms: float = 0.0
    camera_motion: Any = None     # camera_motion.MotionEstimate
    camera_moving: bool = False   # 本帧相机是否在动（幅值超过阈值）
    camera_steady: bool = False   # 相机是否在"匀速"移动（匀速不破坏像素重力约束）
