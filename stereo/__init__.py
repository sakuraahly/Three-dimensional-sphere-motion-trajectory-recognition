"""双目：同步、极线配对、三角测量。

本包里"能算的"都已经能跑了（同步、配对、三角测量），
"还没接"的是把它们串进实时主流程 —— 见 docs/计划书.md 的 M3。

    sync.py         双路视频同步（闪光/拍手对齐）
    triangulate.py  极线约束配对 + 三角测量 + 三维轨迹

分工说明：
    calib/    负责"标定出参数"（离线、一次）
    sphere_tracker/  负责"单目 2D 检测与跟踪"（逐帧）
    stereo/   负责"把两路合成三维"（逐帧，依赖上面两者）
"""

from __future__ import annotations

from .sync import SyncConfig, SyncResult, apply_offset, brightness_series, find_offset
from .triangulate import (
    MatchConfig,
    StereoPoint,
    StereoReconstructor,
    match_epipolar,
    triangulate_one,
)

__all__ = [
    "SyncConfig",
    "SyncResult",
    "find_offset",
    "brightness_series",
    "apply_offset",
    "MatchConfig",
    "StereoPoint",
    "StereoReconstructor",
    "match_epipolar",
    "triangulate_one",
]
