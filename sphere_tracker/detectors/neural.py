"""AI 后端：YOLO 与 YOLO+SAM 检测器（可选依赖 torch / ultralytics）。"""

from __future__ import annotations

import cv2
import math
import numpy as np
from pathlib import Path
from .base import BaseDetector
from .base import resolve_weights
from ..config import DetectorConfig
from ..geometry import contour_circularity, fit_circle_robust
from ..types import Detection

#: 项目根（neural.py → detectors → sphere_tracker → 根）。
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: COCO 的类别数。只有这个规模的模型才需要按类别号过滤 —— 理由见
#: ``resolve_classes_filter`` 的 docstring。
COCO_CLASS_COUNT = 80

#: 自训单类球检测器的默认落点（``apps/tools/train_ball_detector.py --out``）。
SELF_TRAINED_WEIGHTS = "models/ball_yolo.pt"


def resolve_effective_weights(weights: str, *, root: Path | None = None) -> tuple[str, bool]:
    """决定这次到底加载哪份权重，返回 ``(权重, 是否自动换成了自训权重)``。

    为什么要自动优选（2026-10-30）：GUI 的检测器下拉只有**名字**（``yolo``），
    没有地方传权重路径。训完 ``models/ball_yolo.pt`` 之后如果必须手改配置才能用上，
    用户看到的就是"我训完了但检测还是 0 检出"，然后把原因归到"模型没学会"。

    只在权重仍是**默认值**时替换：用户显式写过路径（``--weights`` / 配置文件），
    说明他知道自己要哪份，替他换掉等于把他的选择藏起来。
    """
    root = _PROJECT_ROOT if root is None else Path(root)
    text = str(weights or "").replace("\\", "/")
    # 与配置默认值比较，而不是写死一串字面量：改了默认值这里自动跟上。
    default_name = str(DetectorConfig().yolo_weights).replace("\\", "/")
    if text not in (default_name, SELF_TRAINED_WEIGHTS):
        return weights, False
    if text == SELF_TRAINED_WEIGHTS:      # 已经明确要自训权重，不算"自动换"
        return weights, False
    candidate = root / SELF_TRAINED_WEIGHTS
    if candidate.is_file():
        return SELF_TRAINED_WEIGHTS, True
    return weights, False


def resolve_classes_filter(model_names, cfg_classes: list[int] | None) -> list[int] | None:
    """返回要传给 ``predict(classes=...)`` 的类别号；``None`` 表示不过滤。

    为什么不能无条件用 ``cfg.yolo_classes = [32]``（2026-10-30 的坑）：
    32 是 **COCO** 的 sports ball。自训模型的类别号是 **0**（单类 ball），
    拿 ``classes=[32]`` 去过滤一个只有 0 类的模型，ultralytics 会安静地返回
    0 个框 —— 现象和"模型根本没学会"一模一样，而其实一个框都没被允许输出。

    所以只在"模型确实是 COCO-80"时才按配置过滤；自训模型用它的全部类别。
    """
    count = len(model_names) if model_names is not None else 0
    if count == COCO_CLASS_COUNT and cfg_classes:
        return list(cfg_classes)
    return None


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
    # 局部 import：factory 会 import registry，registry 又 import 本模块 ——
    # 模块级 import 成环。以前这里干脆没 import，于是第一次出框就 NameError
    # （yolo_refine 默认 True，等于 YOLO 一旦检出就崩），见交接说明。
    from .factory import build_single_detector

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

    自训权重（``models/ball_yolo.pt``，由 ``apps/tools/train_ball_detector.py`` 产出）
    存在且配置仍是默认权重时**自动改用**它，并打印一行说明 —— 见
    ``resolve_effective_weights``。自训模型是单类（0=ball），
    所以类别过滤只对 COCO-80 生效（见 ``resolve_classes_filter``）。

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
        weights, auto = resolve_effective_weights(self.cfg.yolo_weights)
        # 路径解析统一走 base.resolve_weights（2026-10-30 team-lead 修好了它的项目根）。
        # 这里不再自己兜一层：两份解析逻辑迟早漂移，而漂移的代价已经在
        # "models/ball_yolo.pt 解析成裸文件名 → FileNotFoundError" 上付过一次。
        path = resolve_weights(weights)
        try:
            self._model = YOLO(path)
        except Exception as exc:  # pragma: no cover
            # 报**这次真正要加载的那份**（auto 时会和配置里的默认值不同）
            self._load_error = f"加载 YOLO 权重 {path!r} 失败：{exc}"
            return self._model
        # 不静默：自动换权重等于换了整个检测器的行为，"用哪份权重"必须写在屏幕上。
        if auto:
            print(f"[模型] 用自训权重 {SELF_TRAINED_WEIGHTS}（单类 ball）")
        elif str(weights).replace("\\", "/") != str(DetectorConfig().yolo_weights).replace("\\", "/"):
            # 显式指定的也要打一行：跑完只剩日志时，"3 epoch 的权重还是 20 epoch 的"
            # 决定了这份结果是能用的还是仅供流程验证，而日志里本来一个字都没有。
            print(f"[模型] 用显式指定的权重 {weights}")
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
        # 只有 COCO-80 才按 cfg.yolo_classes 过滤：自训模型的类别号是 0，
        # 用 [32] 过滤会让它永远返回 0 框（详见 resolve_classes_filter）。
        classes = resolve_classes_filter(getattr(model, "names", None), self.cfg.yolo_classes)
        if classes:
            kwargs["classes"] = classes
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
            # 同 refine_circle_in_box：成环，只能延迟 import。
            from .factory import build_single_detector

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
