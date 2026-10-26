"""检测器工厂（按名字造**单个**检测器）。

单独成文件是为了避开循环导入：`fusion.py` 需要 `build_single_detector`，
而 `build_detector` 又要构造 `FusionDetector`。

清单本身已搬到 `registry.py`（R7：注册表是唯一选项来源）。本模块只剩
"把名字规整一下 → 查表 → 不认识就返回 None"这一层薄适配，
`fusion.py` / `neural.py` 依赖的 `None = 跳过这个子检测器` 语义保持不变。
"""

from __future__ import annotations

from ..config import DetectorConfig
from ..plugins import names, resolve
from . import registry  # noqa: F401  （R1：注册在 import 时发生，这里显式依赖）
from .base import BaseDetector

# 历史别名（旧 `build_detector` 接受过 sam / sam2）。这是**别名映射**，不是清单：
# 真正的名字只在 `registry.py` 里登记一次。
_ALIASES = {"sam": "yolo_sam", "sam2": "yolo_sam"}


def canonical_name(name: str) -> str:
    """规整名字：去空白 + 小写 + 历史别名归一。"""
    key = str(name).strip().lower()
    return _ALIASES.get(key, key)


def build_single_detector(name: str, cfg: DetectorConfig) -> BaseDetector | None:
    """按名字造一个检测器；注册表里没有就返回 None（调用方据此跳过）。"""
    key = canonical_name(name)
    if key not in names("detector"):
        return None
    return resolve("detector", key)(cfg)
