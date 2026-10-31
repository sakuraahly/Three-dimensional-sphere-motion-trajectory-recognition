"""AI 后端：YOLO 与 YOLO+SAM 检测器（可选依赖 torch / ultralytics）。"""

from __future__ import annotations

import cv2
import json
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

#: ``refine_circle_in_box`` 里"框内细化"的搜索半径，相对**框半宽**的比例下/上限。
#:
#: 几何依据：YOLO 的框是球的**外接范围**（框总比球大一圈），所以真球半径必然落在
#: "约框半宽的 0.5〜1.15 倍"内。用**框尺寸**定窗口，而不是继承全局
#: ``hough_max_radius=80`` —— 那个 80 是给**全图** hough 定的性能上限
#: （``HoughCircles`` 代价随搜索区间线性上涨，见 config.py 注释），在框内
#: ~180×180 的小 ROI 里根本不适用：镜面球像素半径约 90 px > 80，
#: 沿用全局值 hough 必然找不到，细化就退回用框中心当圆心。
#:
#: **下界必须收紧**是这一条的关键：镜面球内部有个高光斑（实测半径约 20 px），
#: 若沿用全局下界 ``min_radius=8``，hough 会"高高兴兴地细化到高光斑上"，
#: 圆心就偏到球面一侧去了（这个失败模式本项目吃过一次，代号 B-132）。
REFINE_BOX_RADIUS_LO = 0.5
REFINE_BOX_RADIUS_HI = 1.15

#: 建议置信度阈值 sidecar 的后缀。训练结束时写在**权重旁边**（``ball_yolo.pt`` →
#: ``ball_yolo.conf.json``）。
#:
#: 为什么需要它（2026-10-30 实测）：20 epoch 的 ``models/ball_yolo.pt`` 在 val 上
#: top-1 分数**中位只有 0.055**，而 ``DetectorConfig.yolo_conf`` 默认 0.25 ——
#: 0.25 不是"滤掉误检"，是把 689 张带框图**一张不剩**地滤掉。用户看到的现象是
#: "我训完了，检测还是什么都没有"，于是判断"这个 YOLO 完全没用"。
#: 阈值该取多少**只有训练那边知道**（它手里有带标注的 val），所以由它写下来，
#: 运行时在"用户没说过该用多少"的前提下照着用。
CONF_SIDECAR_SUFFIX = ".conf.json"

#: 显式给过 ``--conf`` 的配置（``id(配置) → 配置``）。
#:
#: 为什么要外挂一份记忆：``config.py`` 是公开契约（另一支在改），加一个
#: ``yolo_conf_explicit`` 字段会把"一次运行里的输入来源"写进持久配置；
#: 而这里要回答的恰恰是"**这一次**的 conf 是人给的吗"。同理不能拿
#: ``cfg.yolo_conf != 0.25`` 去推断 —— 用户显式写 ``--conf 0.25`` 时值不变，
#: 但那仍然是他的选择，覆盖它等于把他的选择藏起来。
#: 存**强引用**而不是只存 id：只存 id 时对象被回收后 id 会被复用，
#: 下一个配置可能被误判成"显式"。
_EXPLICIT_CONF: dict[int, DetectorConfig] = {}


def mark_conf_explicit(cfg: DetectorConfig) -> None:
    """声明"这份配置的 ``yolo_conf`` 是用户显式给的"，运行时不再按 sidecar 校准。

    只有 ``apps/run_video.py`` 在 ``--conf`` 出现时调用；别的入口不给就是不标记，
    于是它们仍然享受"开箱能用"的默认校准。
    """
    _EXPLICIT_CONF[id(cfg)] = cfg


def conf_is_explicit(cfg: DetectorConfig) -> bool:
    """这份配置的 ``yolo_conf`` 是否由用户显式指定（见 ``mark_conf_explicit``）。"""
    return _EXPLICIT_CONF.get(id(cfg)) is cfg


def conf_sidecar_path(weights: str, *, root: Path | None = None) -> Path:
    """``models/ball_yolo.pt`` → ``<项目根>/models/ball_yolo.conf.json``。"""
    root = _PROJECT_ROOT if root is None else Path(root)
    path = Path(str(weights).replace("\\", "/"))
    if not path.is_absolute():
        path = root / path
    return path.with_suffix(CONF_SIDECAR_SUFFIX)


def load_recommended_conf(weights: str, *, root: Path | None = None) -> float | None:
    """读权重旁的 sidecar，返回建议阈值；缺失 / 损坏 / 越界一律返回 ``None``。

    这里**绝不抛**：sidecar 是"锦上添花"的校准信息，读不到就该退回配置值继续跑。
    为了一个可选文件让整段录像跑不起来，是拿次要的东西换掉主要的东西。
    越界（≤0 或 ≥1）也当没写：0 会把所有像素都当球，1 会滤掉一切，
    两个极端都比"用配置默认值"更糟。
    """
    path = conf_sidecar_path(weights, root=root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        value = float(payload.get("recommended_conf"))
    except (TypeError, ValueError):
        return None
    if not (0.0 < value < 1.0) or not math.isfinite(value):
        return None
    return value


def resolve_effective_conf(cfg: DetectorConfig, weights: str, *, auto: bool,
                          root: Path | None = None) -> float | None:
    """返回"这次该采用的推荐阈值"，``None`` = 不动配置。

    四个条件缺一不可，每个都对应一种"不许替用户做主"的场景：

    1. ``auto``：权重是**自动优选**出来的。用户显式 ``--weights`` 时他知道自己要哪份
       权重，也就该由他自己决定门槛；
    2. 用户没显式给 ``--conf``（见 ``mark_conf_explicit``）；
    3. 配置里的值仍是**默认值**。配置文件里写了 0.10 同样是用户的选择；
    4. sidecar 里有一个可用值，且和当前值不同（相同就没有"采用"这回事）。
    """
    if not auto or conf_is_explicit(cfg):
        return None
    default = float(DetectorConfig().yolo_conf)
    if abs(float(cfg.yolo_conf) - default) > 1e-9:
        return None
    value = load_recommended_conf(weights, root=root)
    if value is None or abs(value - float(cfg.yolo_conf)) <= 1e-9:
        return None
    return value


def conf_notice(adopted: float, default: float) -> str:
    """采用推荐阈值时的那一行。**必须打**：它解释"为什么跑出来的和配置不一样"。"""
    return (f"[阈值] 自训权重建议 conf={adopted:g}（配置默认 {default:g} 会把所有框滤掉），"
            "已采用；要覆盖请显式 --conf")


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
    # 搜索半径按**框尺寸**定，不继承全局的 hough_max_radius（见 REFINE_BOX_RADIUS_LO
    # 的注释）：框是球的外接范围，所以真球半径落在"约框半宽的 0.5〜1.15 倍"内。
    half = 0.5 * max(box[2] - box[0], box[3] - box[1])
    lo = max(cfg.min_radius, REFINE_BOX_RADIUS_LO * half)
    hi = min(cfg.max_radius, REFINE_BOX_RADIUS_HI * half)
    if hi < lo:
        hi = lo
    sub = DetectorConfig(**{**cfg.__dict__, "method": "color",
                            "min_radius": lo, "max_radius": hi,
                            "hough_max_radius": hi})
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
                parts={**cand.parts, "refine": name, "refine_window": (lo, hi)},
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

    **阈值也一起校准**：同一份权重旁边若有一份 ``*.conf.json``（训练端在 val 上
    扫出来的推荐阈值），且这次是自动优选权重的、用户又没显式给过 ``--conf``，
    就采用它并打一行 ``[阈值]`` 说明 —— 见 ``resolve_effective_conf``。理由是
    分数尺度：20 epoch 的权重在 val 上 top-1 分数中位只有 0.055，默认 0.25 会把
    所有框滤掉，"训完了还是什么都没有"就是这么来的。

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
        # 阈值校准：只在"权重是自动优选的 + 用户没说过 conf"时才用 sidecar。
        # 顺序上放在加载成功之后 —— 权重都加载不了，谈门槛没有意义。
        self._adopt_recommended_conf(weights, auto=auto)
        return self._model

    def _adopt_recommended_conf(self, weights: str, *, auto: bool,
                                root: Path | None = None) -> float | None:
        """把权重旁 sidecar 的推荐阈值装进 ``self.cfg``；返回采用的值（没采用则 ``None``）。

        抽成独立方法是为了能被**直接断言**：真正"采纳"这个动作发生在这里，
        只测 ``resolve_effective_conf`` 的话，把这一句赋值删掉测试也不会红 ——
        而那正是"看起来校准了、其实没生效"的典型形态。
        """
        adopted = resolve_effective_conf(self.cfg, weights, auto=auto, root=root)
        if adopted is None:
            return None
        default_conf = float(DetectorConfig().yolo_conf)
        self.cfg.yolo_conf = adopted      # detect() 每帧读它，改在这里就够
        print(conf_notice(adopted, default_conf))
        return adopted

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
