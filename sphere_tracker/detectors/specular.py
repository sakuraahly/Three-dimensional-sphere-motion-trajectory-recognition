"""镜面球（金属摆锤）检测器：人工 seed + CSRT 跟踪 + 模板/运动门控。

为什么不是 Hough/颜色
--------------------
实测（1920×1080，`20261019-193809`）：

- 真球半径 **35~40 px**，不是 640 宽预览里的 8~12 px；
- 球速可达 **~80 px/帧**；
- 饱和度与背景白袋几乎一样（S≈30 vs 26），颜色阈值不可分；
- Hough 的窄半径 6~14 命中率 0%，宽半径 14~34 也只有 22.7%，
  且会锁到背景圆（右路 frame 37 起锁在 ~(645,605)）；
- **CSRT + 人工 seed** 实测 86.4%@15 px、100%@20 px、20.5 ms/帧。

所以本检测器**要求人工 seed**（`--seed x,y,r`，右路另给 `--seed-right`）。
自动初始化不可靠（Hough top-1 可偏 528 px），不假装全自动。
丢失时返回空列表，**绝不** snap 到远处的 Hough 圆。
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from ..config import DetectorConfig
from ..types import Detection
from .base import BaseDetector


class SpecularBallDetector(BaseDetector):
    name = "specular"

    def __init__(self, cfg: DetectorConfig) -> None:
        super().__init__(cfg)
        self._seed: tuple[float, float, float] | None = None
        self._seed_frame = 0
        self._frame_idx = -1
        self._tracker = None
        self._bbox: tuple[int, int, int, int] | None = None
        self._template: np.ndarray | None = None
        self._prev_patch: np.ndarray | None = None
        self._prev_center: tuple[float, float] | None = None
        self._started = False
        self._frames_since_seed = 0
        self._static_frames = 0
        self._recent = deque(maxlen=25)
        self.last_error = ""
        self._snapshot: dict = {}

    # ------------------------------------------------------------------
    def seed(self, x: float, y: float, radius: float, frame_index: int = 0) -> None:
        """人工给一个初始球心/半径；第 frame_index 帧才初始化 CSRT。"""
        self._seed = (float(x), float(y), float(radius))
        self._seed_frame = max(0, int(frame_index))
        self._frame_idx = -1
        self._tracker = None
        self._bbox = None
        self._template = None
        self._prev_patch = None
        self._prev_center = None
        self._started = False
        self._frames_since_seed = 0
        self._static_frames = 0
        self._recent.clear()
        self.last_error = ""

    def reset(self) -> None:
        """重置跟踪器，但**保留**人工 seed（换源后仍可重新初始化）。"""
        self._tracker = None
        self._bbox = None
        self._template = None
        self._prev_patch = None
        self._prev_center = None
        self._started = False
        self._frames_since_seed = 0
        self._static_frames = 0
        self._recent.clear()
        self._frame_idx = -1

    # ------------------------------------------------------------------
    @staticmethod
    def _bbox_at(x: float, y: float, side: int, shape) -> tuple[int, int, int, int]:
        h, w = shape[:2]
        side = max(8, int(side))
        x0 = int(round(x - side / 2))
        y0 = int(round(y - side / 2))
        x0 = max(0, min(w - side, x0))
        y0 = max(0, min(h - side, y0))
        return x0, y0, side, side

    @staticmethod
    def _patch(gray: np.ndarray, bbox, size: int = 64) -> np.ndarray | None:
        x, y, w, h = bbox
        if w <= 2 or h <= 2:
            return None
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(gray.shape[1], x + w), min(gray.shape[0], y + h)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        p = gray[y0:y1, x0:x1]
        return cv2.resize(p, (size, size), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _ncc(a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None or a.shape != b.shape:
            return 0.0
        aa = a.astype(np.float32).ravel()
        bb = b.astype(np.float32).ravel()
        aa -= aa.mean()
        bb -= bb.mean()
        na, nb = float(np.linalg.norm(aa)), float(np.linalg.norm(bb))
        if na < 1e-6 or nb < 1e-6:
            return 0.0
        return float(np.dot(aa, bb) / (na * nb))

    def _make_tracker(self):
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
            return cv2.legacy.TrackerCSRT_create()
        return None

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray, ctx=None) -> list[Detection]:
        c = self.cfg.specular
        # CSRT 在**彩色图**上明显更稳：实测左路灰度 1/11，BGR 10/11@15px。
        track_frame = frame
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # ctx.gray 可能是放大前的原图（pipeline 的 detect-width 坑），形状不符就不用
        if ctx is not None and getattr(ctx, "gray", None) is not None:
            if ctx.gray.shape[:2] == gray.shape[:2]:
                gray = ctx.gray

        if self._seed is None:
            self.last_error = "specular 需要人工 seed：--seed x,y,r（右路 --seed-right）"
            return []
        self._frame_idx += 1
        if self._frame_idx < self._seed_frame:
            return []

        if self._tracker is None:
            if not self._started:
                x, y, r = self._seed
                if not (c.min_radius <= r <= c.max_radius):
                    # seed 半径只做软提示，不硬拒绝（用户可能量的是外径）
                    self.last_error = (f"seed 半径 {r:.1f} 超出建议范围 "
                                       f"{c.min_radius:.0f}~{c.max_radius:.0f}，仍尝试")
                side = int(c.init_box)
                if r > 45.0:                      # 大球（>90px 直径）才放大框；36px 球仍用 64
                    side = max(side, int(round(1.8 * r)))
                side = min(side, int(c.max_box))
                bbox = self._bbox_at(x, y, side, gray.shape)
                tracker = self._make_tracker()
                if tracker is None:
                    self.last_error = "OpenCV 没有 CSRT tracker（需要 opencv-contrib）"
                    return []
                try:
                    tracker.init(track_frame, bbox)
                except Exception as exc:                    # noqa: BLE001
                    self.last_error = f"CSRT init 失败：{exc}"
                    return []
                self._tracker = tracker
                self._bbox = bbox
                self._template = self._patch(gray, bbox)
                self._prev_center = (x, y)
                self._started = True
                self._frames_since_seed = 0
                self._recent.clear()
                self._recent.append((x, y))
                self.last_error = ""
                return [Detection(x=x, y=y, radius=r, score=1.0, method=self.name,
                                  extra={"state": "seed", "bbox": bbox})]
            return []

        ok, bbox = self._tracker.update(track_frame)
        if not ok or bbox is None:
            self.last_error = "CSRT 跟丢；不 snap 到背景圆，请检查 seed 或重新录制"
            self._tracker = None
            return []
        x, y, w, h = [int(round(v)) for v in bbox]
        cx, cy = x + w / 2.0, y + h / 2.0
        radius = max(w, h) / 2.0
        self._frames_since_seed += 1

        if not (c.min_radius <= radius <= c.max_radius):
            self.last_error = f"跟踪框半径 {radius:.1f} 超出 {c.min_radius:.0f}~{c.max_radius:.0f}"
            return []

        jump = 1e9
        if self._prev_center is not None:
            jump = float(np.hypot(cx - self._prev_center[0], cy - self._prev_center[1]))
            if jump > c.max_jump_px:
                self.last_error = f"单帧跳 {jump:.1f}px > {c.max_jump_px:.0f}，判为跟丢"
                self._tracker = None
                return []

        patch = self._patch(gray, (x, y, w, h))
        ncc = self._ncc(self._template, patch) if patch is not None else 0.0
        motion = 0.0
        if self._prev_patch is not None and patch is not None:
            motion = float(cv2.absdiff(patch, self._prev_patch).mean())
        if motion < c.static_motion_eps and jump < 1.0:
            self._static_frames += 1
        else:
            self._static_frames = 0
        if self._static_frames > c.max_static_frames:
            self.last_error = (f"连续 {self._static_frames} 帧中心/局部都不动，"
                               "判为丢了（不再报背景）")
            self._tracker = None
            return []
        if c.min_ncc > 0 and ncc < c.min_ncc:
            self.last_error = f"模板 NCC {ncc:.2f} < {c.min_ncc:.2f}，判为跟丢"
            self._tracker = None
            return []
        if (c.min_motion > 0
                and self._frames_since_seed > c.motion_grace_frames
                and motion < c.min_motion and jump < 1.0):
            self.last_error = (f"局部运动 {motion:.1f} < {c.min_motion:.1f} 且中心几乎不动，"
                               "判为锁到静止背景")
            self._tracker = None
            return []

        self._bbox = (x, y, w, h)
        self._prev_patch = patch
        self._prev_center = (cx, cy)
        self._recent.append((cx, cy))
        if len(self._recent) >= self._recent.maxlen:
            xs = [p[0] for p in self._recent]
            ys = [p[1] for p in self._recent]
            if max(xs) - min(xs) < 5.0 and max(ys) - min(ys) < 5.0:
                self.last_error = (f"最近 {self._recent.maxlen} 帧中心跨度 <5px，"
                                   "判为丢了（不再报静止背景）")
                self._tracker = None
                return []
        self._snapshot = {
            "center": (round(cx, 1), round(cy, 1)), "radius": round(radius, 1),
            "ncc": round(ncc, 3), "motion": round(motion, 2), "jump": round(jump, 2),
            "bbox": (x, y, w, h), "frame": self._frames_since_seed,
        }
        self.last_error = ""
        return [Detection(
            x=cx, y=cy, radius=radius,
            score=float(max(0.0, min(1.0, 0.6 * max(0.0, ncc) + 0.4 * min(1.0, motion / 30.0)))),
            method=self.name,
            parts={"ncc": float(ncc), "motion": float(motion), "jump": float(jump)},
            extra={"state": "track", "bbox": (x, y, w, h),
                   "frames_since_seed": self._frames_since_seed},
        )]

    def debug_snapshot(self) -> dict:
        return dict(self._snapshot, error=self.last_error, has_seed=self._seed is not None)
