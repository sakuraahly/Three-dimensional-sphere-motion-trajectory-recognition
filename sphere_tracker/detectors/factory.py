"""检测器工厂。

单独成文件是为了避开循环导入：fusion.py 需要 build_single_detector
来构造被投票的子检测器，而 build_detector 又要构造 FusionDetector。
把它们分开放，依赖方向就是单向的：

    factory.py （只依赖 color / hough / motion）
        ↑
    fusion.py、__init__.py
"""

from __future__ import annotations

from ..config import DetectorConfig
from .base import BaseDetector
from .color import ColorBlobDetector
from .hough import HoughDetector
from .motion import MotionDetector


def build_single_detector(name: str, cfg: DetectorConfig) -> BaseDetector | None:
    """按名字造一个"非融合"检测器。"""
    name = name.strip().lower()
    if name == "color":
        return ColorBlobDetector(cfg)
    if name == "hough":
        return HoughDetector(cfg)
    if name == "motion":
        return MotionDetector(cfg)
    return None
