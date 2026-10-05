"""检测器基类与公共工具。"""

from __future__ import annotations

import numpy as np
from ..config import DetectorConfig
from ..types import Detection
from pathlib import Path


def resolve_weights(weights: str) -> str:
    """把权重路径解析成实际可用的路径。

    规则：
      1. 绝对路径 -> 原样返回（不存在就退回文件名，交给 ultralytics 处理）；
      2. 相对路径 -> 相对**项目根目录**解析（而不是当前工作目录），
         这样在 apps/ 或任意目录下运行都能找到 models/yolo11n.pt；
      3. 项目里找不到 -> 只返回文件名，让 ultralytics 自己去缓存目录找或下载。
    """
    if not weights:
        return weights
    p = Path(weights)
    if p.is_absolute():
        return str(p) if p.exists() else p.name
    root = Path(__file__).resolve().parents[1]
    candidate = root / p
    if candidate.exists():
        return str(candidate)
    return p.name


class BaseDetector:
    name = "base"

    def __init__(self, cfg: DetectorConfig) -> None:
        self.cfg = cfg

    # ctx 是可选的共享上下文（灰度图 / 相机自运动），由 pipeline 每帧构造一次。
    # 不需要它的后端直接忽略即可。
    def detect(self, frame: np.ndarray, ctx=None) -> list[Detection]:
        raise NotImplementedError

    def reset(self) -> None:
        """换视频源时调用（背景模型需要重置）。"""

    # -- 公共打分/筛选 -----------------------------------------------------
    def _score_candidate(
        self,
        radius: float,
        circularity: float = 0.0,
        fill_ratio: float = 0.0,
        radial: float = 0.0,
        edge_support: float = 0.0,
        weights: tuple[float, float, float, float] = (0.30, 0.28, 0.27, 0.15),
    ) -> float:
        """把各种形状指标合成 0-1 的置信度。

        四项指标各有侧重，缺一不可：
          圆度 circularity   排除锯齿状/细长目标
          填充率 fill        排除方块（方块只能填满外接圆的 ~61%）
          径向一致性 radial   排除方块/胶囊/多目标粘连（最有判别力的一项）
          边缘支持度 support  排除"颜色对但边缘不像球"的区域
        """
        if radius < self.cfg.min_radius or radius > self.cfg.max_radius:
            return 0.0
        if radial > 0.0 and radial < self.cfg.min_radial_consistency:
            return 0.0
        if fill_ratio > 0.0 and fill_ratio < self.cfg.min_fill_ratio:
            return 0.0
        size_score = 1.0
        # 半径贴着上下限时降权，避免"卡在阈值边缘"的误检
        if radius < self.cfg.min_radius * 1.3:
            size_score = 0.8
        if radius > self.cfg.max_radius * 0.8:
            size_score = 0.8
        w1, w2, w3, w4 = weights
        total_w = 0.0
        shape = 0.0
        for value, weight in (
            (circularity, w1), (fill_ratio, w2), (radial, w3), (edge_support, w4)
        ):
            if value <= 0.0:
                continue
            shape += value * weight
            total_w += weight
        if total_w <= 0.0:
            return 0.0
        return float(max(0.0, min(1.0, shape / total_w * size_score)))
