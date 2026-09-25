"""球体运动轨迹识别 —— 检测 / 跟踪 / 三维重建 / 轨迹预测。

模块划分::

    config.py    所有可调参数（支持 JSON 存取）
    types.py     数据结构：Detection / TrackState / FrameResult
    geometry.py  亚像素圆拟合、相机模型、由球半径反推深度
    detect.py    检测后端：颜色 / 霍夫圆 / 运动 / 融合 / YOLO / SAM
    track.py     卡尔曼滤波器 + 轨迹管理（拖尾、丢失补偿）
    predict.py   轨迹预测：卡尔曼外推 / 二次多项式(重力)拟合
    viz.py       可视化：拖尾、圆、预测虚线、HUD
    sources.py   视频源：摄像头 / 视频文件 / 图片
    pipeline.py  单帧总流程编排
"""

from .config import AppConfig, load_config, save_config
from .types import Detection, FrameResult, TrackState, TrailPoint
from .pipeline import SpherePipeline

__all__ = [
    "AppConfig",
    "load_config",
    "save_config",
    "Detection",
    "FrameResult",
    "TrackState",
    "TrailPoint",
    "SpherePipeline",
]
__version__ = "0.2.0"
