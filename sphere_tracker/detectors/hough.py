"""霍夫圆检测器（梯度法，作为几何线索的补充）。"""

from __future__ import annotations

import cv2
import numpy as np
from .base import BaseDetector
from ..config import DetectorConfig
from ..geometry import circle_edge_support, circle_radial_gradient_stats
from ..types import Detection


class HoughDetector(BaseDetector):
    """霍夫圆变换：不依赖颜色，但对参数和光照敏感。

    加了"圆上边缘支持度"打分，能明显压制霍夫圆常见的乱检。
    """

    name = "hough"

    def __init__(self, cfg: DetectorConfig) -> None:
        super().__init__(cfg)
        self._edges: np.ndarray | None = None

    def detect(self, frame: np.ndarray, ctx=None) -> list[Detection]:
        gray = getattr(ctx, "gray", None) if ctx is not None else None
        if gray is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        k = self.cfg.hough_blur_ksize | 1
        blur = cv2.medianBlur(gray, k)
        self._edges = cv2.Canny(blur, max(10, int(self.cfg.hough_param1 * 0.4)), int(self.cfg.hough_param1))
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=self.cfg.hough_dp,
            minDist=max(self.cfg.hough_min_dist, self.cfg.min_radius * 2.0),
            param1=self.cfg.hough_param1,
            param2=self.cfg.hough_param2,
            minRadius=int(self.cfg.min_radius),
            # ⚠️ 用 hough **自己**的上限，不是物理约束 max_radius。
            # HoughCircles 的代价随搜索半径区间线性上涨；实测 1080p 真实纹理下
            # 用 200 要 4 秒一帧（见 config.py 的注释）。80 已经够用。
            maxRadius=int(min(self.cfg.hough_max_radius, self.cfg.max_radius)),
        )
        out: list[Detection] = []
        if circles is None:
            return out
        for cx, cy, r in np.round(circles[0]).astype(float):
            # 两道校验，缺一不可：
            #   1) 圆上的梯度方向必须沿半径（纹理上凑出来的圆过不了）；
            #   2) 梯度的正负号必须统一（真球体一边全是暗->亮，假圆正负混杂）。
            grad, consistency = circle_radial_gradient_stats(blur, cx, cy, r)
            if grad < self.cfg.hough_min_support or consistency < self.cfg.hough_min_sign_consistency:
                continue
            edge_sup = circle_edge_support(self._edges, cx, cy, r, samples=90)
            # 40% 圆上边缘支持 + 60% 径向梯度一致性（`grad*consistency`）。
            # 用**关键字**写：旧版把这两个量当成位置参数传，读起来像"圆度=边缘支持"，
            # 已经让人误判过一次"这是 bug 吗"。数值行为不变。
            score = self._score_candidate(
                r, circularity=edge_sup, edge_support=grad * consistency,
                weights=(0.4, 0.0, 0.0, 0.6),
            )
            if score <= 0:
                continue
            out.append(
                Detection(x=float(cx), y=float(cy), radius=float(r), score=score,
                          method=self.name,
                          parts={"edge_support": edge_sup, "gradient_support": grad,
                                 "sign_consistency": consistency})
            )
        out.sort(key=lambda d: d.score, reverse=True)
        return out
