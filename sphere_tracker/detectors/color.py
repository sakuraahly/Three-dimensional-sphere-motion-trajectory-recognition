"""颜色分割检测器（HSV 阈值 + 形态学 + 轮廓打分）。"""

from __future__ import annotations

import cv2
import math
import numpy as np
from .base import BaseDetector
from ..config import DetectorConfig
from ..geometry import (
    circle_edge_support,
    circle_gradient_support,
    circle_radial_gradient_stats,
    contour_circularity,
    fit_circle_robust,
    radial_consistency,
    refine_circle_subpixel,
)
from ..types import Detection


class ColorBlobDetector(BaseDetector):
    """HSV 阈值分割 + 形态学 + 轮廓圆度筛选。

    这是"传统方法"里性价比最高的方案：只要球的颜色和背景能分开，
    它的速度和精度都远超 Canny / 霍夫圆，而且天然给出球的边缘轮廓。
    """

    name = "color"

    def __init__(self, cfg: DetectorConfig) -> None:
        super().__init__(cfg)
        self._last_mask: np.ndarray | None = None

    def build_mask(self, frame: np.ndarray) -> np.ndarray:
        c = self.cfg.color
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array(c.hsv_lower, dtype=np.uint8)
        upper = np.array(c.hsv_upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        if c.use_second_range:
            mask |= cv2.inRange(
                hsv,
                np.array(c.hsv_lower2, dtype=np.uint8),
                np.array(c.hsv_upper2, dtype=np.uint8),
            )
        if c.open_ksize >= 3:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c.open_ksize, c.open_ksize))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        if c.close_ksize >= 3:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c.close_ksize, c.close_ksize))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        self._last_mask = mask
        return mask

    def detect(self, frame: np.ndarray, ctx=None) -> list[Detection]:
        mask = self.build_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        out: list[Detection] = []
        min_area = math.pi * self.cfg.min_radius ** 2 * 0.5
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            # 用"最小外接圆"算填充率：方块只能填满约 61%，球体能填满 90%+
            (_, _), r_enc = cv2.minEnclosingCircle(cnt)
            if r_enc < self.cfg.min_radius or r_enc > self.cfg.max_radius:
                continue
            circ = contour_circularity(cnt)
            fill = float(area / (math.pi * r_enc * r_enc)) if r_enc > 0 else 0.0
            if circ < self.cfg.min_circularity or fill < self.cfg.min_fill_ratio:
                continue
            pts = cnt.reshape(-1, 2)
            if pts.shape[0] >= 5:
                cx, cy, r = fit_circle_robust(pts)
            else:
                (cx, cy), r = cv2.minEnclosingCircle(cnt)
                cx, cy, r = float(cx), float(cy), float(r)
            if r < self.cfg.min_radius or r > self.cfg.max_radius:
                continue
            radial = radial_consistency(pts, cx, cy, r)
            score = self._score_candidate(r, circ, fill, radial)
            if score <= 0:
                continue
            out.append(
                Detection(
                    x=float(cx), y=float(cy), radius=float(r), score=score,
                    method=self.name, contour=pts, mask=mask,
                    parts={"circularity": circ, "fill": fill, "radial": radial},
                )
            )
        out.sort(key=lambda d: d.score, reverse=True)
        if self.cfg.subpixel_refine:
            # 灰度图优先复用 pipeline 已经算好的，省一次 cvtColor
            gray = getattr(ctx, "gray", None) if ctx is not None else None
            if gray is None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for det in out[:3]:      # 只细化最有希望的几个，避免拖慢帧率
                rx, ry, rr = refine_circle_subpixel(
                    gray, det.x, det.y, det.radius, n_rays=self.cfg.subpixel_rays
                )
                if self.cfg.min_radius <= rr <= self.cfg.max_radius:
                    det.x, det.y, det.radius = float(rx), float(ry), float(rr)
                    det.parts["subpixel"] = 1.0
        return out
