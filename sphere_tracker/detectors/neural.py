"""AI 后端：YOLO 与 YOLO+SAM 检测器（可选依赖 torch / ultralytics）。"""

from __future__ import annotations

import cv2
import math
import numpy as np
from .base import BaseDetector
from .base import resolve_weights
from ..config import DetectorConfig
from ..geometry import contour_circularity, fit_circle_robust
from ..types import Detection


def refine_circle_in_box(
    frame: np.ndarray, det: Detection, cfg: DetectorConfig
) -> Detection | None:
    """在检测框内用颜色/霍夫把方框细化为贴合球体的圆。

    YOLO 的框总是比球大一圈（框是外接矩形），直接拿框当球会让半径偏大、
    圆心偏，进而让测深和预测都偏。细化一下很值得。
    """
    box = det.extra.get("box")
    if box is None:
        return None
    h, w = frame.shape[:2]
    x1 = int(max(0, math.floor(box[0])))
    y1 = int(max(0, math.floor(box[1])))
    x2 = int(min(w, math.ceil(box[2])))
    y2 = int(min(h, math.ceil(box[3])))
    if x2 - x1 < 5 or y2 - y1 < 5:
        return None
    roi = frame[y1:y2, x1:x2]
    sub = DetectorConfig(**{**cfg.__dict__, "method": "color"})
    best: Detection | None = None
    for name in ("color", "hough"):
        detector = build_single_detector(name, sub)
        if detector is None:
            continue
        found = detector.detect(roi)
        if found:
            cand = found[0]
            cand = Detection(
                x=cand.x + x1, y=cand.y + y1, radius=cand.radius,
                score=min(1.0, 0.5 * det.score + 0.5 * cand.score),
                method=f"yolo+{name}", contour=None,
                parts={**cand.parts, "refine": name},
            )
            if best is None or cand.score > best.score:
                best = cand
    return best


class YoloDetector(BaseDetector):
    """YOLO 目标检测（ultralytics）。

    零样本就能用：官方 COCO 权重里有 "sports ball"（类别 32）这一类。
    对乒乓球、网球、足球、篮球这类常见球通常直接可用；
    对特殊颜色/材质的球，用几百张自己拍的图微调一下效果最好。

    安装：pip install ultralytics
    """

    name = "yolo"

    def __init__(self, cfg: DetectorConfig) -> None:
        super().__init__(cfg)
        self._model = None
        self._load_error: str | None = None

    def _ensure_model(self):
        if self._model is not None or self._load_error is not None:
            return self._model
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:  # pragma: no cover - 取决于环境
            self._load_error = (
                "未安装 ultralytics，无法使用 YOLO 后端。请运行：\n"
                "    pip install ultralytics\n"
                f"（原始错误：{exc}）"
            )
            return None
        try:
            self._model = YOLO(resolve_weights(self.cfg.yolo_weights))
        except Exception as exc:  # pragma: no cover
            self._load_error = f"加载 YOLO 权重 {self.cfg.yolo_weights!r} 失败：{exc}"
        return self._model

    def detect(self, frame: np.ndarray, ctx=None) -> list[Detection]:
        model = self._ensure_model()
        if model is None:
            raise RuntimeError(self._load_error or "YOLO 不可用")
        kwargs: dict = dict(
            conf=self.cfg.yolo_conf,
            imgsz=self.cfg.yolo_imgsz,
            device=self.cfg.yolo_device,
            verbose=False,
        )
        if self.cfg.yolo_classes:
            kwargs["classes"] = self.cfg.yolo_classes
        results = model.predict(frame, **kwargs)
        out: list[Detection] = []
        for res in results:
            boxes = getattr(res, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), conf in zip(xyxy, confs):
                w = float(x2 - x1)
                h = float(y2 - y1)
                radius = 0.5 * max(w, h)
                cx = float((x1 + x2) / 2.0)
                cy = float((y1 + y2) / 2.0)
                det = Detection(
                    x=cx, y=cy, radius=radius, score=float(conf), method=self.name,
                    parts={"box_w": w, "box_h": h},
                    extra={"box": (float(x1), float(y1), float(x2), float(y2))},
                )
                if self.cfg.yolo_refine:
                    det = refine_circle_in_box(frame, det, self.cfg) or det
                out.append(det)
        out.sort(key=lambda d: d.score, reverse=True)
        return out


class YoloSamDetector(BaseDetector):
    """YOLO 出框 + SAM 分割出像素级边缘（精度最高，也最吃算力）。

    这才是真正意义上的"AI 球体边缘识别"：
    SAM 给出的是掩膜，掩膜边界就是球的真实边缘，可以做亚像素圆拟合，
    在有遮挡、反光、阴影的情况下比阈值法稳得多。

    安装：pip install ultralytics
    权重：sam2.1_t.pt（约 40 MB，会自动下载）或 sam2.1_s.pt / sam2.1_b.pt
    """

    name = "yolo_sam"

    def __init__(self, cfg: DetectorConfig) -> None:
        super().__init__(cfg)
        self._sam = None
        self._prompter: BaseDetector | None = None
        self._load_error: str | None = None

    def _ensure(self):
        if self._sam is not None or self._load_error is not None:
            return self._sam
        try:
            from ultralytics import SAM  # type: ignore
        except ImportError as exc:  # pragma: no cover
            self._load_error = (
                "未安装 ultralytics，无法使用 SAM 后端。请运行：\n"
                "    pip install ultralytics\n"
                f"（原始错误：{exc}）"
            )
            return None
        try:
            self._sam = SAM(resolve_weights(self.cfg.sam_model))
        except Exception as exc:  # pragma: no cover
            self._load_error = f"加载 SAM 权重 {self.cfg.sam_model!r} 失败：{exc}"
            return None
        if self.cfg.sam_refine_from == "yolo":
            self._prompter = YoloDetector(self.cfg)
        else:
            self._prompter = build_single_detector(self.cfg.sam_refine_from, self.cfg)
        return self._sam

    def detect(self, frame: np.ndarray, ctx=None) -> list[Detection]:
        sam = self._ensure()
        if sam is None:
            raise RuntimeError(self._load_error or "SAM 不可用")
        if self._prompter is None:
            return []
        seeds = self._prompter.detect(frame)
        out: list[Detection] = []
        for seed in seeds[:3]:
            box = seed.extra.get("box")
            if box is None:
                r = max(seed.radius * 1.15, seed.radius + 4)
                box = (seed.x - r, seed.y - r, seed.x + r, seed.y + r)
            try:
                results = sam.predict(frame, bboxes=[list(box)], verbose=False)
            except Exception:
                continue
            for res in results:
                masks = getattr(res, "masks", None)
                if masks is None or masks.data is None or len(masks.data) == 0:
                    continue
                m = masks.data.cpu().numpy()[0]
                mask = (m > 0.5).astype(np.uint8) * 255
                if mask.shape[:2] != frame.shape[:2]:
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if not contours:
                    continue
                cnt = max(contours, key=cv2.contourArea)
                if cnt.shape[0] < 5:
                    continue
                cx, cy, r = fit_circle_robust(cnt.reshape(-1, 2))
                if r < self.cfg.min_radius or r > self.cfg.max_radius:
                    continue
                score = self._score_candidate(r, contour_circularity(cnt), 1.0, 1.0)
                out.append(
                    Detection(x=cx, y=cy, radius=r, score=max(score, seed.score * 0.9),
                              method=self.name, contour=cnt.reshape(-1, 2), mask=mask,
                              parts={"seed_score": seed.score},
                              extra={"seed_method": seed.method})
                )
        out.sort(key=lambda d: d.score, reverse=True)
        return out
