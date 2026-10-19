"""检测器集合。

按"线索来源"分文件，每个文件只负责一种检测思路：

    base.py     基类 + 轮廓打分等公共工具
    factory.py  按名字造单个检测器（避开循环导入的中间层）
    color.py    颜色分割   —— 主力，不受相机运动影响
    hough.py    霍夫圆     —— 几何线索补充
    motion.py   背景差分   —— 物体与背景同色时唯一的线索（含相机运动补偿）
    fusion.py   投票融合   —— 把上面几路合起来
    neural.py   YOLO / SAM —— 可选，需要 torch；半径精度不如几何方法

对外统一从本包导入：from sphere_tracker.detectors import build_detector
"""

from __future__ import annotations

from ..config import DetectorConfig
from .base import BaseDetector, resolve_weights
from .color import ColorBlobDetector
from .factory import build_single_detector
from .fusion import FusionDetector
from .hough import HoughDetector
from .motion import MotionDetector
from .neural import YoloDetector, YoloSamDetector, refine_circle_in_box
from .specular import SpecularBallDetector

__all__ = [
    "BaseDetector",
    "ColorBlobDetector",
    "HoughDetector",
    "MotionDetector",
    "SpecularBallDetector",
    "FusionDetector",
    "YoloDetector",
    "YoloSamDetector",
    "build_detector",
    "build_single_detector",
    "resolve_weights",
    "refine_circle_in_box",
]


# --------------------------------------------------------------------------
# 工厂
# --------------------------------------------------------------------------
def build_detector(cfg: DetectorConfig) -> BaseDetector:
    """根据配置创建检测器。"""
    name = cfg.method.strip().lower()
    if name in ("color", "hough", "motion", "specular"):
        det = build_single_detector(name, cfg)
        assert det is not None
        return det
    if name == "fusion":
        return FusionDetector(cfg)
    if name == "yolo":
        return YoloDetector(cfg)
    if name in ("yolo_sam", "sam", "sam2"):
        return YoloSamDetector(cfg)
    raise ValueError(
        f"未知检测方法 {cfg.method!r}，可选：color / hough / motion / specular / "
        "fusion / yolo / yolo_sam"
    )
