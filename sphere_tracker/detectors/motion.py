"""背景差分检测器（含相机运动补偿）。"""

from __future__ import annotations

import cv2
import math
import numpy as np
from .base import BaseDetector
from ..camera_motion import warp_to_current
from ..config import DetectorConfig
from ..geometry import (
    contour_circularity,
    fit_circle_robust,
    radial_consistency,
)
from ..types import Detection


class MotionDetector(BaseDetector):
    """运动检测。相机固定时用 MOG2 背景建模；相机在动时自动切换成"对齐后再差分"。

    为什么需要两套：
      MOG2 假设背景静止，相机一平移整幅画面都变成"前景"，误检会瞬间爆炸。
      所以先估计相邻帧的全局运动（相机自运动），把上一帧对齐到当前帧再相减，
      这样即使相机手持移动，运动物体依然能被分离出来。
    """

    name = "motion"

    def __init__(self, cfg: DetectorConfig) -> None:
        super().__init__(cfg)
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=cfg.motion_history,
            varThreshold=cfg.motion_var_threshold,
            detectShadows=cfg.motion_detect_shadows,
        )
        self._frames = 0
        self._prev_frame: np.ndarray | None = None
        self._bg_stale = 0          # 相机移动期间累计的帧数（背景模型已失效）
        self.last_mode = "bg"       # 便于调试：本帧用的是哪种模式

    def reset(self) -> None:
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=self.cfg.motion_history,
            varThreshold=self.cfg.motion_var_threshold,
            detectShadows=self.cfg.motion_detect_shadows,
        )
        self._frames = 0
        self._prev_frame = None
        self._bg_stale = 0

    # ------------------------------------------------------------------
    def _compensated_mask(self, frame: np.ndarray, motion) -> np.ndarray:
        """相机移动时的替代方案：对齐上一帧后做"有符号差分"，再剔除残影。

        为什么不能直接对 |差分| 取轮廓：
          差分图里同时有"物体旧位置"和"新位置"两块，叠加起来是被切过的月牙，
          圆度和径向一致性全不合格，检不出来。
        怎么区分：
          有符号差分 curr - aligned > 0 表示"这里变亮了" = 物体刚出现在这；
          < 0 表示"这里变暗了" = 物体刚从这离开（残影）。
          但物体也可能是暗的，所以两个方向都要试；
          最终用"当前帧在该区域内部与周围是否有明显反差"来判定：
          真位置 -> 内部是物体、周围是背景 -> 反差大；
          残影   -> 内部已经是背景了       -> 反差小，丢掉。
        """
        aligned = warp_to_current(self._prev_frame, motion)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        agray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        out = np.zeros_like(gray)
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        min_area = math.pi * self.cfg.min_radius ** 2 * 0.35
        max_area = math.pi * self.cfg.max_radius ** 2 * 2.5

        for signed in (cv2.subtract(gray, agray), cv2.subtract(agray, gray)):
            thr, _ = cv2.threshold(signed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thr = max(float(thr), float(self.cfg.motion_diff_min_threshold))
            _, m = cv2.threshold(signed, thr, 255, cv2.THRESH_BINARY)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k3)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k5)
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area or area > max_area:
                    continue
                if self._is_current_position(gray, cnt):
                    cv2.drawContours(out, [cnt], -1, 255, -1)
        return out

    def _is_current_position(self, gray: np.ndarray, cnt: np.ndarray) -> bool:
        """判断这块差分区域是"物体当前所在"还是"物体刚离开的残影"。

        判据：在当前帧里，这块区域的内部与紧邻周围是否有明显反差。
        物体真在这 -> 内部是物体、外面是背景，反差大；
        残影       -> 内部早已是背景，内外接近，反差小。
        """
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        k_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        inner = cv2.erode(mask, k_in)
        outer = cv2.dilate(mask, k_out) & cv2.bitwise_not(mask)
        if cv2.countNonZero(inner) < 8 or cv2.countNonZero(outer) < 8:
            return False
        inner_mean = float(cv2.mean(gray, mask=inner)[0])
        outer_mean = float(cv2.mean(gray, mask=outer)[0])
        return abs(inner_mean - outer_mean) >= float(self.cfg.motion_contrast_min)

    def _segment_object(self, gray: np.ndarray, proposal: np.ndarray):
        """把差分区域当"区域提议"，在当前帧里分割出真正的球并返回轮廓。

        为什么不能直接用差分的轮廓：物体移动时，差分得到的是"旧位置 ∪ 新位置"
        减掉重叠部分后的两块月牙 —— 形状根本不圆，圆度/径向一致性全不合格。
        差分只擅长回答"哪里发生了变化"，形状还得回到当前帧去量。
        做法：在提议区域外扩一圈做局部 Otsu（对亮球暗球都成立），
        取其中形状最像球的那一块，再交给统一的打分流程。
        """
        x, y, w, h = cv2.boundingRect(proposal)
        m = int(max(w, h) * 0.8) + 6
        x0 = max(0, x - m); y0 = max(0, y - m)
        x1 = min(gray.shape[1], x + w + m); y1 = min(gray.shape[0], y + h + m)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        crop = gray[y0:y1, x0:x1]
        thr, _ = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, binary = cv2.threshold(crop, thr, 255, cv2.THRESH_BINARY)
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        min_area = math.pi * self.cfg.min_radius ** 2 * 0.5
        best = None
        best_area = 0.0
        for cand in (binary, cv2.bitwise_not(binary)):
            cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, k3)
            cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, k3)
            cnts, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for c in cnts:
                area = cv2.contourArea(c)
                if area < min_area:
                    continue
                (ccx, ccy), rr = cv2.minEnclosingCircle(c)
                if rr < self.cfg.min_radius or rr > self.cfg.max_radius:
                    continue
                if cv2.contourArea(c) / (math.pi * rr * rr) < self.cfg.min_fill_ratio:
                    continue
                if contour_circularity(c) < self.cfg.min_circularity:
                    continue
                # 必须落在提议区域附近，否则可能是画面里别的圆形物体
                gx, gy = ccx + x0, ccy + y0
                pad = 0.4 * max(w, h) + 6
                if not (x - pad <= gx <= x + w + pad and y - pad <= gy <= y + h + pad):
                    continue
                if area > best_area:
                    best_area = area
                    best = c + np.array([[[x0, y0]]], dtype=c.dtype)
        return best

    def detect(self, frame: np.ndarray, ctx=None) -> list[Detection]:
        self._frames += 1
        cfg = self.cfg
        motion = getattr(ctx, "motion", None) if ctx is not None else None
        moving = bool(getattr(ctx, "camera_moving", False)) if ctx is not None else False
        mode = cfg.motion_compensation
        use_comp = (
            mode == "force"
            or (mode == "auto" and moving)
        ) and self._prev_frame is not None and motion is not None and motion.has_transform

        pre_contours = None
        if use_comp:
            fg = self._compensated_mask(frame, motion)
            self.last_mode = "compensated"
            self._bg_stale += 1
            # 差分区域只当"提议"，形状回到当前帧去量
            gray_seg = getattr(ctx, "gray", None) if ctx is not None else None
            if gray_seg is None:
                gray_seg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            proposals, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            pre_contours = []
            for p in proposals:
                obj = self._segment_object(gray_seg, p)
                if obj is not None:
                    pre_contours.append(obj)
        else:
            fg = self._bg.apply(frame)
            self.last_mode = "bg"
            if cfg.motion_compensation != "off" and self._bg_stale > 0:
                # 相机刚从"移动"回到"静止"：背景模型早过期了，重学几帧，
                # 否则会吐出一大块假前景，把跟踪器带跑。
                for _ in range(3):
                    self._bg.apply(frame, learningRate=1.0)
                self._bg_stale = 0
        self._prev_frame = frame.copy()

        if cfg.motion_detect_shadows and self.last_mode == "bg":
            _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)  # 去掉阴影(127)
        # 注意：形态学"闭"会把团块胀大，直接导致半径偏大 -> 测深偏小。
        # 所以只用 1 次闭运算，半径的精度靠后面的圆拟合找回来。
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=1)
        if self._frames < 5 and pre_contours is None:  # 背景还没建好
            return []
        if pre_contours is not None:
            contours = pre_contours
        else:
            contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        out: list[Detection] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < math.pi * self.cfg.min_radius ** 2 * 0.5:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            if radius < self.cfg.min_radius or radius > self.cfg.max_radius:
                continue
            circ = contour_circularity(cnt)
            fill = float(area / (math.pi * radius * radius)) if radius > 0 else 0.0
            pts = cnt.reshape(-1, 2)
            if pts.shape[0] >= 5:
                fx_, fy_, r2 = fit_circle_robust(pts)
            else:
                fx_, fy_, r2 = float(cx), float(cy), float(radius)
            radial = radial_consistency(pts, fx_, fy_, r2)
            score = self._score_candidate(r2, circ, fill, radial)
            if score <= 0:
                continue
            out.append(
                Detection(x=float(fx_), y=float(fy_), radius=float(r2), score=score,
                          method=self.name, contour=pts, mask=fg,
                          parts={"circularity": circ, "fill": fill, "radial": radial})
            )
        out.sort(key=lambda d: d.score, reverse=True)
        return out
