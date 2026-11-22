"""检测器注册表：7 个实现的**唯一**清单（R7）。

以前这份清单写了两遍（`__init__.py` 的 if 链与 `factory.py` 的 if 链），
外面还有三份手抄（GUI `METHODS`、三个 app 的 `choices`/help），已经漂移过
（GUI 7 项 / `run_stereo_video` 5 项 / `run_all` 4 项）。现在只有这里一份：
加检测器 = 写一个模块 + 在这里加一行 `register(...)`。

`fusion` 的工厂**延迟 import**：`fusion.py` 依赖 `factory.build_single_detector`，
而 `factory.py` import 本模块，模块级 import 会成环。
"""

from __future__ import annotations

from ..config import DetectorConfig
from ..plugins import register
from .base import BaseDetector
from .color import ColorBlobDetector
from .generic import GenericBallDetector
from .hough import HoughDetector
from .motion import MotionDetector
from .neural import YoloDetector, YoloSamDetector
from .specular import SpecularBallDetector

# 默认项 = `DetectorConfig.method` 的值（`config.py`）。排第一：GUI 下拉的首项
# 就是"不传参数时真正生效的那个"，顺序与默认值不再是两件事。
register("detector", "color", ColorBlobDetector,
         summary="颜色阈值分割（最快，默认；球色明显时首选）", default=True)
register("detector", "specular", SpecularBallDetector,
         summary="高光/镜面定位（金属球，需人工 seed；半径精度依赖种子）")
register("detector", "hough", HoughDetector,
         summary="霍夫圆（几何线索，约慢 4 倍；颜色不可靠时补充）")
register("detector", "motion", MotionDetector,
         summary="背景差分（含相机运动补偿；球与背景同色时唯一的线索）")


def _fusion(cfg: DetectorConfig) -> BaseDetector:
    from .fusion import FusionDetector  # 延迟：见模块 docstring 的成环说明

    return FusionDetector(cfg)


register("detector", "fusion", _fusion,
         summary="多线索投票融合（最稳，约慢 6 倍；有同色干扰物时用）")
# YOLO / SAM 的实际依赖在首次推理时才 import（`neural.py` 的 `_ensure_model`），
# 所以这里构造不炸、GUI 能列出这一项；缺包时 detect() 给出安装命令。
register("detector", "yolo", YoloDetector,
         summary="YOLO 目标检测（需 ultralytics；零样本可识别常见球）",
         needs=("ultralytics",))
register("detector", "yolo_sam", YoloSamDetector,
         summary="YOLO 出框 + SAM 分割（精度最高、最吃算力；需 ultralytics）",
         needs=("ultralytics",))
# 不依赖颜色的可选路径：边缘 + 圆度 + 径向一致性，识别近圆水果等任意球体。
# 追加在末尾 —— 既有登记项的顺序与默认项（color）一字不动（R4/R7）。
register("detector", "generic", GenericBallDetector,
         summary="几何优先（边缘+圆度+径向一致性），不依赖颜色；近圆水果等任意球体可选")
