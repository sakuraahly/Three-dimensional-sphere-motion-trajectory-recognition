"""球体检测后端。

设计思想：把"找球"拆成可替换的后端，同一个接口，随便换：

    传统方法                            AI 方法
    ------------------------------      ----------------------------------
    color   HSV 分割 + 圆度筛选         yolo     YOLO 检测框（可零样本，COCO sports ball）
    hough   霍夫圆                      yolo_sam YOLO 给框 + SAM 像素级分割
    motion  背景差分（静态相机）
    fusion  上面几种投票融合（默认推荐）

所有后端都输出同样的 Detection（圆心 + 半径 + 置信度），
后面的跟踪 / 三维重建 / 预测完全不需要知道用的是哪种方法。
"""

from __future__ import annotations

import math
import time

import cv2
import numpy as np

from .config import DetectorConfig
from .geometry import (
    circle_edge_support,
    circle_gradient_support,
    circle_radial_gradient_stats,
    contour_circularity,
    fit_circle_robust,
    radial_consistency,
    refine_circle_subpixel,
)
from .types import Detection


class BaseDetector:
    name = "base"

    def __init__(self, cfg: DetectorConfig) -> None:
        self.cfg = cfg

    def detect(self, frame: np.ndarray) -> list[Detection]:
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


# --------------------------------------------------------------------------
# 1) 颜色分割
# --------------------------------------------------------------------------
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

    def detect(self, frame: np.ndarray) -> list[Detection]:
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
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for det in out[:3]:      # 只细化最有希望的几个，避免拖慢帧率
                rx, ry, rr = refine_circle_subpixel(
                    gray, det.x, det.y, det.radius, n_rays=self.cfg.subpixel_rays
                )
                if self.cfg.min_radius <= rr <= self.cfg.max_radius:
                    det.x, det.y, det.radius = float(rx), float(ry), float(rr)
                    det.parts["subpixel"] = 1.0
        return out


# --------------------------------------------------------------------------
# 2) 霍夫圆
# --------------------------------------------------------------------------
class HoughDetector(BaseDetector):
    """霍夫圆变换：不依赖颜色，但对参数和光照敏感。

    加了"圆上边缘支持度"打分，能明显压制霍夫圆常见的乱检。
    """

    name = "hough"

    def __init__(self, cfg: DetectorConfig) -> None:
        super().__init__(cfg)
        self._edges: np.ndarray | None = None

    def detect(self, frame: np.ndarray) -> list[Detection]:
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
            maxRadius=int(self.cfg.max_radius),
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
            score = self._score_candidate(
                r, edge_sup, 0.0, 0.0, grad * consistency, weights=(0.4, 0.0, 0.0, 0.6)
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


# --------------------------------------------------------------------------
# 3) 背景差分（运动）
# --------------------------------------------------------------------------
class MotionDetector(BaseDetector):
    """MOG2 背景建模 + 形态学 + 轮廓。只适合摄像头固定的场景。"""

    name = "motion"

    def __init__(self, cfg: DetectorConfig) -> None:
        super().__init__(cfg)
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=cfg.motion_history,
            varThreshold=cfg.motion_var_threshold,
            detectShadows=cfg.motion_detect_shadows,
        )
        self._frames = 0

    def reset(self) -> None:
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=self.cfg.motion_history,
            varThreshold=self.cfg.motion_var_threshold,
            detectShadows=self.cfg.motion_detect_shadows,
        )
        self._frames = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self._frames += 1
        fg = self._bg.apply(frame)
        if self.cfg.motion_detect_shadows:
            _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)  # 去掉阴影(127)
        # 注意：形态学"闭"会把团块胀大，直接导致半径偏大 -> 测深偏小。
        # 所以只用 1 次闭运算，半径的精度靠后面的圆拟合找回来。
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=1)
        if self._frames < 5:  # 背景还没建好
            return []
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


# --------------------------------------------------------------------------
# 4) 融合
# --------------------------------------------------------------------------
class FusionDetector(BaseDetector):
    """多后端投票融合。

    规则：
      1. 依次跑各个子检测器，各取最好的一个候选；
      2. 任意两个候选中心距离 < max(agree_px, agree_ratio * 半径)，算"同意"；
      3. 得票最高的候选胜出，每多一票加 0.12 分；
      4. 只有一票且分数低于 vote_single_min 时判为不可信（返回空）。
    这样既能用颜色拿精度、用霍夫拿鲁棒性，又能靠"多方一致"压掉误检。
    """

    name = "fusion"

    def __init__(self, cfg: DetectorConfig) -> None:
        super().__init__(cfg)
        self.members: dict[str, BaseDetector] = {}
        for name in cfg.fusion_members:
            det = build_single_detector(name, cfg)
            if det is not None:
                self.members[name] = det
        self.last_parts: dict[str, Detection | None] = {}

    def reset(self) -> None:
        for det in self.members.values():
            det.reset()

    def detect(self, frame: np.ndarray) -> list[Detection]:
        cands: list[tuple[str, Detection]] = []
        self.last_parts = {}
        for name, det in self.members.items():
            try:
                found = det.detect(frame)
            except cv2.error:
                found = []
            best = found[0] if found else None
            self.last_parts[name] = best
            if best is not None:
                cands.append((name, best))
        if not cands:
            return []

        # 计算两两一致 -> 投票
        votes: dict[str, int] = {name: 0 for name, _ in cands}
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                ni, di = cands[i]
                nj, dj = cands[j]
                tol = max(self.cfg.agree_px, self.cfg.agree_ratio * max(di.radius, dj.radius))
                if math.hypot(di.x - dj.x, di.y - dj.y) <= tol:
                    votes[ni] += 1
                    votes[nj] += 1

        scored: list[tuple[float, str, Detection]] = []
        for name, det in cands:
            total = min(1.0, det.score + 0.12 * votes[name])
            scored.append((total, name, det))
        scored.sort(key=lambda item: item[0], reverse=True)

        total, name, best = scored[0]
        if votes[name] == 0 and total < 0.55:
            return []  # 孤证且置信度不高 -> 宁可不报
        best = Detection(
            x=best.x, y=best.y, radius=best.radius, score=total,
            method=f"{self.name}:{name}", contour=best.contour, mask=best.mask,
            parts={**best.parts, "votes": float(votes[name])},
        )
        return [best]


def build_single_detector(name: str, cfg: DetectorConfig) -> BaseDetector | None:
    """按名字造一个"非融合"检测器。"""
    name = name.strip().lower()
    if name == "color":
        return ColorBlobDetector(cfg)
    if name == "hough":
        return HoughDetector(cfg)
    if name == "motion":
        return MotionDetector(cfg)
    return None


# --------------------------------------------------------------------------
# 5) AI 后端（可选依赖，未安装时给出清晰提示）
# --------------------------------------------------------------------------
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
            self._model = YOLO(self.cfg.yolo_weights)
        except Exception as exc:  # pragma: no cover
            self._load_error = f"加载 YOLO 权重 {self.cfg.yolo_weights!r} 失败：{exc}"
        return self._model

    def detect(self, frame: np.ndarray) -> list[Detection]:
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
            self._sam = SAM(self.cfg.sam_model)
        except Exception as exc:  # pragma: no cover
            self._load_error = f"加载 SAM 权重 {self.cfg.sam_model!r} 失败：{exc}"
            return None
        if self.cfg.sam_refine_from == "yolo":
            self._prompter = YoloDetector(self.cfg)
        else:
            self._prompter = build_single_detector(self.cfg.sam_refine_from, self.cfg)
        return self._sam

    def detect(self, frame: np.ndarray) -> list[Detection]:
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


# --------------------------------------------------------------------------
# 工厂
# --------------------------------------------------------------------------
def build_detector(cfg: DetectorConfig) -> BaseDetector:
    """根据配置创建检测器。"""
    name = cfg.method.strip().lower()
    if name in ("color", "hough", "motion"):
        det = build_single_detector(name, cfg)
        assert det is not None
        return det
    if name == "fusion":
        return FusionDetector(cfg)
    if name == "yolo":
        return YoloDetector(cfg)
    if name in ("yolo_sam", "sam", "sam2"):
        return YoloSamDetector(cfg)
    raise ValueError(
        f"未知检测方法 {cfg.method!r}，可选：color / hough / motion / fusion / yolo / yolo_sam"
    )
