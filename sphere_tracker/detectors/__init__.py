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
from ..plugins import resolve
from .base import BaseDetector, resolve_weights
from .color import ColorBlobDetector
from .factory import build_single_detector, canonical_name
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
    """根据配置创建检测器。

    名字 → 实现走注册表（`registry.py`），这里**不再有 if 链**：
    以前这份分派和 `factory.build_single_detector` 各写一遍，加一个检测器要改两处。
    未知名字抛 `UnknownPlugin`，消息里带可用名字与注册模块（R3，不 fallback）。
    """
    return resolve("detector", canonical_name(cfg.method))(cfg)
