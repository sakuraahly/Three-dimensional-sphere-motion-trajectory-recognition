"""兼容垫片 —— 检测器已拆分到 sphere_tracker.detectors 包。

保留这个文件是为了让 from sphere_tracker.detect import build_detector
这类旧写法继续可用。新代码请直接从 sphere_tracker.detectors 导入。
"""

from __future__ import annotations

from .detectors import (
    BaseDetector,
    ColorBlobDetector,
    FusionDetector,
    HoughDetector,
    MotionDetector,
    YoloDetector,
    YoloSamDetector,
    build_detector,
    build_single_detector,
    refine_circle_in_box,
    resolve_weights,
)

__all__ = [
    "BaseDetector",
    "ColorBlobDetector",
    "HoughDetector",
    "MotionDetector",
    "FusionDetector",
    "YoloDetector",
    "YoloSamDetector",
    "build_detector",
    "build_single_detector",
    "resolve_weights",
    "refine_circle_in_box",
]
