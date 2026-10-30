"""插件注册表自测：清单唯一（R7）、未知名字报错（R3）、默认唯一（R4）、边界（R5）。

为什么需要它
------------
"有哪些检测器"以前写在**四个地方**（GUI `METHODS`、`run_stereo_video` 的 `choices`、
`run_all` 的 `choices`、`run_realtime` 的 help），而且已经漂移：GUI 7 项、
`run_stereo_video` 5 项（缺 yolo/yolo_sam）、`run_all` 4 项。
用户在不同入口能选的实现不一样，**界面上没有任何迹象**。

现在清单只有一处（`sphere_tracker/detectors/registry.py`），本测试守住三件事：

1. 注册表自身的不变量：名字唯一、**恰好一个**默认、`summary` 非空；
2. `resolve` 拼错名字时必须**报错并列出候选**，绝不静默 fallback 到默认；
3. 没有任何地方再手写检测器名清单（R7），库内也没有反向 import `apps.`（R5）。

这两条是 AST 静态扫描 —— 散落的手写清单不会再靠"人记得"来避免。

反向自证（改了就会红）
--------------------
==========================================  ====================================  ========
改动                                        注入方式                              结果
==========================================  ====================================  ========
GUI 退回手写清单                            `METHODS = ["color", "hough", ...]`   exit=1
某个 app 又抄一份 4 项 choices               `choices=("color","hough",...)`       exit=1
注册表出现两个 default                     给 specular 加 `default=True`         exit=1
`summary` 写空串                            某项 summary=""                       exit=1
`resolve` 未知名字改回 fallback             `return default_factory`              exit=1
fusion 的成员拼错（不在注册表里）           `fusion_members += ["colour"]`        exit=1
库内 import `apps.`                         在 `stereo/` 里加 `import apps.x`     exit=1
自训模型也按 `[32]` 过滤                    `resolve_classes_filter` 恒返回配置    exit=1
自动优选覆盖显式权重                        去掉"显式路径不替换"的分支             exit=1
前置检查放行 1 个会话                        `MIN_SESSIONS = 1`                    exit=1
采纳 sidecar 的推荐阈值却不生效              删掉 `_adopt_recommended_conf` 的赋值   exit=1
推荐阈值退回"升序取第一个"（永远 0.02）       `reversed(rows)` → `rows`              exit=1
`--conf` 缺省又变回 0.25                     `default=0.25`                         exit=1
==========================================  ====================================  ========

用法::

    python tests/selftest_plugins.py
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "apps").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sphere_tracker.config import DetectorConfig                      # noqa: E402
from sphere_tracker.detectors import build_detector, build_single_detector  # noqa: E402

# 「import 即注册」：这些模块的顶层（或尾部）执行注册装饰器。
# 漏掉任何一个，对应 kind 的表就是空的，`default()` 会抛错 —— 所以放在这里显式 import，
# 而不是指望别处顺手带上（那正是"看不见的依赖"）。
import calib.export                     # noqa: E402,F401  注册 exporter
import calib.methods                    # noqa: E402,F401  注册 calib_method
import sphere_tracker.predict           # noqa: E402,F401  注册 predictor
import sphere_tracker.sources           # noqa: E402,F401  注册 source
from sphere_tracker.plugins import (                                   # noqa: E402
    UnknownPlugin, clear, default, describe, names, register, resolve,
)

# 已迁移到注册表的**全部**六个扩展点（P0/P1 detector、P2 其余四类、P3 tracker）。
# 少写一个等于少守一块：表里没有的 kind，"名字唯一/恰一默认/summary 非空"就没人在盯。
KINDS = ("detector", "predictor", "source", "calib_method", "exporter", "tracker")
# R7 要扫描的入口：任何"用户能选检测器"的地方。
ENTRY_FILES = sorted((ROOT / "apps").glob("*.py")) + [ROOT / "启动程序.py"]
# R5 没有白名单。2026-10-26 解掉 `stereo/sessions.py` 那处反向依赖后，违例数为 0 ——
# 此时**保留白名单就是永久豁免**：以后新增的反向依赖只要往表里塞一行就能"合法"，
# 而这正是本项目反复踩的坑（写着的闸门不拦人）。要豁免，先改 AGENTS §5。


def _check_registry() -> list[tuple[str, bool]]:
    """不变量：名字唯一 / 恰好一个默认 / summary 非空。"""
    checks: list[tuple[str, bool]] = []
    for kind in KINDS:
        got = names(kind)
        meta = describe(kind)
        defaults = [d["name"] for d in meta if d["is_default"]]
        checks.append((f"{kind}：注册表非空（{len(got)} 项：{' / '.join(got)}）", bool(got)))
        checks.append((f"{kind}：名字在 kind 内唯一", len(got) == len(set(got))))
        checks.append((f"{kind}：恰好一个默认（{defaults or '无'}）", len(defaults) == 1))
        checks.append((f"{kind}：每项都写了 summary（{len(meta)} 项）",
                       all(d["summary"].strip() for d in meta)))
        checks.append((f"{kind}：describe 与 names 顺序一致",
                       [d["name"] for d in meta] == got))
    return checks


def _check_lookup() -> list[tuple[str, bool]]:
    """R3：未知名字抛 UnknownPlugin，消息里有候选与来源模块；不 fallback。"""
    checks: list[tuple[str, bool]] = []
    try:
        resolve("detector", "no_such_detector")
        checks.append(("R3：未知名字必须抛 UnknownPlugin（没有 fallback）", False))
        return checks
    except UnknownPlugin as exc:
        msg = str(exc)
        avail = names("detector")
        checks.append(("R3：抛的是 UnknownPlugin（LookupError 子类）", isinstance(exc, LookupError)))
        checks.append(("R3：消息里带请求的名字", "no_such_detector" in msg))
        checks.append(("R3：消息里带 kind", "detector" in msg))
        checks.append((f"R3：消息里列全 {len(avail)} 个可用名字",
                       all(n in msg for n in avail)))
        checks.append(("R3：消息里指出注册模块（漏 import 时能查）",
                       "sphere_tracker.detectors" in msg))

    # R2 / R4：用一个临时 kind 试错，不污染 detector。
    register("selftest_kind", "a", lambda cfg: cfg, summary="临时 A")
    try:
        default("selftest_kind")
        checks.append(("R4：零个 default 时 default() 必须报错（不靠 import 顺序）", False))
    except LookupError:
        checks.append(("R4：零个 default 时 default() 报错（不靠 import 顺序）", True))
    register("selftest_kind", "b", lambda cfg: cfg, summary="临时 B", default=True)
    checks.append(("R4：单个 default 时取到它", default("selftest_kind") == "b"))
    register("selftest_kind", "c", lambda cfg: cfg, summary="临时 C", default=True)
    try:
        default("selftest_kind")
        checks.append(("R4：多个 default 时 default() 必须报错（不许二选一）", False))
    except LookupError:
        checks.append(("R4：多个 default 时 default() 报错（不许二选一）", True))
    try:
        register("selftest_kind", "a", lambda cfg: cfg, summary="重复")
        checks.append(("R2：重名注册必须抛 ValueError", False))
    except ValueError:
        checks.append(("R2：重名注册抛 ValueError（要覆盖得写 replace=True）", True))
    try:
        register("selftest_kind", "d", lambda cfg: cfg, summary="  ")
        checks.append(("R6：summary 为空串必须报错", False))
    except ValueError:
        checks.append(("R6：summary 为空串报错（不是可选项）", True))
    register("selftest_kind", "a", lambda cfg: cfg, summary="覆盖", replace=True)
    checks.append(("R2：replace=True 才允许覆盖", "a" in names("selftest_kind")))
    return checks


def _literal_name_lists(path: Path, known: set[str]) -> list[tuple[int, list[str]]]:
    """找"手写清单"：一个列表/元组/集合字面量里出现 ≥3 个检测器名（R7）。"""
    hits: list[tuple[int, list[str]]] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            continue
        values = [e.value for e in node.elts
                  if isinstance(e, ast.Constant) and isinstance(e.value, str)]
        matched = [v for v in values if v.strip().lower() in known]
        # 3 是"足以确定在抄清单"的下限：1~2 个名字常是别名或语义配置
        # （如 `fusion_members`、`sam_refine_from` 的取值），不算清单。
        if len(matched) >= 3:
            hits.append((node.lineno, matched))
    return hits


def _check_no_manual_lists() -> list[tuple[str, bool]]:
    """R7：入口脚本里不许再有手写检测器名清单。"""
    known = {n.lower() for n in names("detector")}
    bad: list[str] = []
    for path in ENTRY_FILES:
        for lineno, matched in _literal_name_lists(path, known):
            bad.append(f"{path.relative_to(ROOT).as_posix()}:{lineno} 手写清单 {matched}")
    if bad:
        print("    手写清单（应为 names(\"detector\")）：")
        for line in bad:
            print(f"      · {line}")
    return [(f"R7：{len(ENTRY_FILES)} 个入口里没有手写检测器名清单", not bad)]


def _app_imports(path: Path) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "apps" or alias.name.startswith("apps."):
                    hits.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == "apps" or node.module.startswith("apps."):
                hits.append((node.lineno, node.module))
    return hits


def _check_library_boundary() -> list[tuple[str, bool]]:
    """R5：库内（sphere_tracker / calib / stereo）不许 import `apps.`。"""
    known: list[str] = []
    new: list[str] = []
    for pkg in ("sphere_tracker", "calib", "stereo"):
        for path in sorted((ROOT / pkg).rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            for lineno, module in _app_imports(path):
                new.append(f"{rel}:{lineno} import {module}")
    for line in new:
        print(f"    [违例] 库内 import apps.：{line}")
    return [(f"R5：库内不得 import apps.（当前违例 {len(new)} 处）", not new)]


def _check_wiring() -> list[tuple[str, bool]]:
    """一致性：注册表里每个名字都能构造；默认值与 config 一致；fusion 成员存在。"""
    checks: list[tuple[str, bool]] = []
    cfg = DetectorConfig()
    broken: list[str] = []
    for name in names("detector"):
        try:
            det = build_single_detector(name, cfg)
        except Exception as exc:                                  # noqa: BLE001
            broken.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        if det is None:
            broken.append(f"{name}: 返回 None")
    checks.append((f"一致性：{len(names('detector'))} 个名字都能构造（不跑推理）", not broken))
    for line in broken:
        print(f"      · {line}")

    checks.append((f"一致性：default() == DetectorConfig().method（{default('detector')!r}）",
                   default("detector") == cfg.method))
    checks.append((f"一致性：names()[0] 就是默认（下拉首项不漂）",
                   names("detector")[0] == default("detector")))
    checks.append(("一致性：build_detector(默认配置) 造得出",
                   type(build_detector(cfg)).__name__ == "ColorBlobDetector"))
    missing = [m for m in cfg.fusion_members if m not in names("detector")]
    checks.append((f"一致性：fusion_members 全在注册表里（缺 {missing or '无'}）", not missing))

    # R6：needs 如实声明；缺依赖时构造不炸、detect 给安装提示（不是 import 期就崩）
    needs = {d["name"]: d["needs"] for d in describe("detector")}
    checks.append(("R6：yolo / yolo_sam 声明了 ultralytics",
                   all("ultralytics" in needs.get(n, []) for n in ("yolo", "yolo_sam"))))
    if importlib.util.find_spec("ultralytics") is None:
        det = build_single_detector("yolo", cfg)
        det._ensure_model()                                       # noqa: SLF001  （只探依赖，不推理）
        err = getattr(det, "_load_error", "") or ""
        checks.append(("R6：缺 ultralytics 时构造成功、错误信息带安装命令",
                       det is not None and "pip install ultralytics" in err))
    else:
        print("    [信息] 本机装了 ultralytics —— 缺依赖分支无法实测，跳过（不假装覆盖）")
    return checks


def _load_train_tool():
    """按文件路径加载 ``apps/tools/train_ball_detector.py``（``apps/tools`` 不是包）。

    必须是**路径加载**而不是 ``import apps.tools.x``：``apps/tools`` 下没有
    ``__init__.py``，包导入会失败。模块顶层刻意没 import ultralytics/torch，
    所以这里在没有 GPU、甚至没装 ultralytics 的机器上也能加载。
    """
    path = ROOT / "apps" / "tools" / "train_ball_detector.py"
    spec = importlib.util.spec_from_file_location("train_ball_detector_selftest", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"加载不了 {path}")
    module = importlib.util.module_from_spec(spec)
    # 必须先登记进 sys.modules：模块里有 @dataclass，dataclasses 会去
    # sys.modules[cls.__module__] 取命名空间，没登记就是 None.__dict__ 直接炸。
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeYolo:
    """够 YoloDetector.detect() 用的假模型：不推理、只记下传进来的 kwargs。"""

    def __init__(self, names) -> None:
        self.names = names
        self.kwargs: dict | None = None

    def predict(self, frame, **kwargs):
        self.kwargs = kwargs
        return []                      # 空结果：本测试只关心"类别过滤传了没有"


def _check_yolo_classes_filter() -> list[tuple[str, bool]]:
    """自训单类模型被 ``classes=[32]`` 过滤 = 永远 0 检出（看着像"没学会"）。"""
    import numpy as np

    from sphere_tracker.detectors.neural import YoloDetector

    checks: list[tuple[str, bool]] = []
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    cfg = DetectorConfig()             # 默认 yolo_classes == [32]
    det = YoloDetector(cfg)
    det._model = _FakeYolo({0: "ball"})   # noqa: SLF001
    det.detect(frame)
    single = dict(det._model.kwargs or {})   # noqa: SLF001
    checks.append(("自训单类模型（names 只有 0）→ 不传 classes（否则永远 0 框）",
                   "classes" not in single))

    det2 = YoloDetector(cfg)
    det2._model = _FakeYolo({i: f"c{i}" for i in range(80)})   # noqa: SLF001
    det2.detect(frame)
    coco = dict(det2._model.kwargs or {})     # noqa: SLF001
    checks.append((f"COCO-80 模型 → 仍按配置传 classes={cfg.yolo_classes}",
                   coco.get("classes") == list(cfg.yolo_classes or [])))
    return checks


def _check_weights_preference() -> list[tuple[str, bool]]:
    """默认权重 + 存在 models/ball_yolo.pt → 自动优选；显式路径不被覆盖。"""
    import tempfile
    from pathlib import Path as _Path

    from sphere_tracker.detectors.neural import (SELF_TRAINED_WEIGHTS,
                                                 resolve_effective_weights)
    from sphere_tracker.detectors.base import resolve_weights

    checks: list[tuple[str, bool]] = []
    default = DetectorConfig().yolo_weights
    with tempfile.TemporaryDirectory() as td:
        root = _Path(td)
        (root / "models").mkdir()
        self_trained = root / SELF_TRAINED_WEIGHTS
        self_trained.write_bytes(b"")            # 空文件就够：只判存在

        got, auto = resolve_effective_weights(default, root=root)
        checks.append((f"默认权重 + 自训权重存在 → 自动切到 {SELF_TRAINED_WEIGHTS}", auto and got == SELF_TRAINED_WEIGHTS))

        # base.resolve_weights 曾经把项目根算成 sphere_tracker/，于是仓库自带的
        # models/yolo11n.pt 被解析成裸文件名（→ ultralytics 去别处找/下载）。
        # 这条守住"仓库里那份权重能被真正找到"，别再退回文件名。
        resolved = _Path(resolve_weights(default))
        checks.append(("base.resolve_weights 把相对路径解析成仓库根下的绝对路径",
                       resolved.is_absolute() and resolved.is_file()))

        explicit = "models/my_ball_v2.pt"
        got2, auto2 = resolve_effective_weights(explicit, root=root)
        checks.append(("显式指定的权重不会被自动优选覆盖", (not auto2) and got2 == explicit))

        self_trained.unlink()
        got3, auto3 = resolve_effective_weights(default, root=root)
        checks.append(("没有自训权重时保持默认权重（不凭空换路径）",
                       (not auto3) and got3 == default))
    return checks


def _check_train_tool_preflight() -> list[tuple[str, bool]]:
    """训练工具的前置检查是纯函数 —— 直接喂数字，不造数据集、不训模型。"""
    import contextlib
    import io

    tool = _load_train_tool()
    checks: list[tuple[str, bool]] = []

    few_sessions = tool.preflight_problems(1, 500, True)
    checks.append(("前置检查：会话数 < 2 要给出「会话」原因",
                   len(few_sessions) == 1 and "会话" in few_sessions[0]))
    few_boxes = tool.preflight_problems(3, 10, True)
    checks.append((f"前置检查：正样本 < {tool.MIN_POSITIVE_BOXES} 要给出「框」原因",
                   len(few_boxes) == 1 and str(tool.MIN_POSITIVE_BOXES) in few_boxes[0]))
    no_dep = tool.preflight_problems(3, 500, False)
    checks.append(("前置检查：没有 ultralytics 要给出安装命令（不是 traceback）",
                   any("pip install ultralytics" in p for p in no_dep)))
    checks.append(("前置检查：都满足时没有原因",
                   tool.preflight_problems(3, 500, True) == []))

    checks.append(("会话解析：前缀取到下划线前一段",
                   tool.parse_session("s20261029_1430_f0001") == "s20261029"))
    checks.append(("会话解析：解析不出归「未知会话」（不许一图一会话，那样检查永远通过）",
                   tool.parse_session("IMG1234") == "未知会话"))

    # 分桶边界：三个桶必须首尾相接、无重叠也无缝
    checks.append(("尺寸分桶：<48 / 48~120 / >120 三档落在正确桶里",
                   [tool.size_bucket(v) for v in (20, 48, 119.9, 120, 500)]
                   == ["<48 px", "48~120 px", "48~120 px", ">120 px", ">120 px"]))

    # 报告口径：运行时阈值下 0 检出必须说"不能当种子"，不能用中位去安慰
    zero = {"headline": {"runtime_conf": 0.25, "n_val_with_boxes": 689, "n_val_images": 733,
                         "n_detected": 0, "median_px": None, "p90_px": None,
                         "recall_30px": 0.0, "recall_30px_all": 0.0}}
    zero_text = "\n".join(tool.summary_lines(zero))
    checks.append(("运行时阈值 0 检出 → 结论必须是「不能当种子」",
                   "不能当种子" in zero_text and "圆心精度" in zero_text))

    good = {"headline": {"runtime_conf": 0.25, "n_val_with_boxes": 689, "n_val_images": 733,
                         "n_detected": 689, "median_px": 5.0, "p90_px": 12.0,
                         "recall_30px": 0.95, "recall_30px_all": 0.9,
                         "radius_median_rel_error": 0.02}}
    good_text = "\n".join(tool.summary_lines(good))
    checks.append(("命中率达标 → 结论才允许说「可以当种子用」",
                   "可以当种子用" in good_text and "不能当种子" not in good_text))

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = tool.main(["--dry-run", "--data", str(ROOT / "outputs" / "_no_such_dataset_")])
    text = buf.getvalue()
    checks.append(("数据集不存在：--dry-run 退出码 2 且说明原因",
                   code == 2 and "不存在" in text and "退出码 2" in text))
    return checks


def _check_conf_calibration() -> list[tuple[str, bool]]:
    """自训权重旁的推荐阈值（sidecar）：只该在"自动权重 + 用户没表态"时生效。

    现场：20 epoch 权重在 val 上 top-1 分数**中位 0.055**，而 ``yolo_conf`` 默认 0.25，
    于是一刀切下去 689 张带框图一张不剩 —— 用户看到"训完了还是什么都没有"，
    结论就成了"这个 YOLO 完全没用"。所以训练端把"该用多少"写进权重旁边的 sidecar，
    运行时在用户没说话时照着用。

    这里守的是**四条"不许替用户做主"**（显式权重 / 显式 --conf / 配置里写了别的值 /
    sidecar 读不到），以及"采纳这个动作真的写进了配置"。
    """
    import json
    import tempfile
    from pathlib import Path as _Path

    from sphere_tracker.detectors.neural import (SELF_TRAINED_WEIGHTS, YoloDetector,
                                                 conf_is_explicit, load_recommended_conf,
                                                 mark_conf_explicit, resolve_effective_conf)

    checks: list[tuple[str, bool]] = []
    default_conf = DetectorConfig().yolo_conf
    with tempfile.TemporaryDirectory() as td:
        root = _Path(td)
        (root / "models").mkdir()
        (root / SELF_TRAINED_WEIGHTS).write_bytes(b"")      # 空文件就够：只判存在
        sidecar = root / "models" / "ball_yolo.conf.json"
        sidecar.write_text(json.dumps({"recommended_conf": 0.05,
                                       "median_top1_conf": 0.055}), encoding="utf-8")

        got = resolve_effective_conf(DetectorConfig(), SELF_TRAINED_WEIGHTS, auto=True, root=root)
        checks.append(("自动优选权重 + 用户没给 --conf → 采用 sidecar 的 0.05", got == 0.05))

        no_auto = resolve_effective_conf(DetectorConfig(), SELF_TRAINED_WEIGHTS,
                                        auto=False, root=root)
        checks.append(("显式 --weights（auto=False）→ 不采用（他自己挑的权重，门槛也归他）",
                       no_auto is None))

        explicit = DetectorConfig()
        mark_conf_explicit(explicit)
        checks.append(("显式 --conf 0.25（值恰好等于配置默认）→ 不覆盖",
                       conf_is_explicit(explicit)
                       and resolve_effective_conf(explicit, SELF_TRAINED_WEIGHTS,
                                                  auto=True, root=root) is None))

        from_file = DetectorConfig()
        from_file.yolo_conf = 0.10
        checks.append(("配置文件里写了 0.10 → 不覆盖（那也是用户的选择）",
                       resolve_effective_conf(from_file, SELF_TRAINED_WEIGHTS,
                                              auto=True, root=root) is None))

        # "采纳"这个动作本身：删掉 _adopt_recommended_conf 里的赋值，这两条立刻红
        det = YoloDetector(DetectorConfig())
        adopted = det._adopt_recommended_conf(SELF_TRAINED_WEIGHTS, auto=True, root=root)  # noqa: SLF001
        checks.append((f"采纳动作真的写进了 cfg（{det.cfg.yolo_conf:g}，detect() 每帧读它）",
                       adopted == 0.05 and abs(det.cfg.yolo_conf - 0.05) < 1e-9))

        det2 = YoloDetector(DetectorConfig())
        mark_conf_explicit(det2.cfg)
        checks.append(("显式 conf 时采纳动作什么都不做",
                       det2._adopt_recommended_conf(SELF_TRAINED_WEIGHTS,  # noqa: SLF001
                                                    auto=True, root=root) is None
                       and det2.cfg.yolo_conf == default_conf))

        sidecar.write_text("{ 这不是 json", encoding="utf-8")
        checks.append(("sidecar 损坏 → 返回 None（不抛：可选文件不许拖垮整段录像）",
                       load_recommended_conf(SELF_TRAINED_WEIGHTS, root=root) is None))
        sidecar.write_text(json.dumps({"recommended_conf": 0}), encoding="utf-8")
        checks.append(("sidecar 值越界（0）→ 当作没写",
                       load_recommended_conf(SELF_TRAINED_WEIGHTS, root=root) is None))
        sidecar.unlink()
        checks.append(("sidecar 缺失 → 退回配置默认值",
                       load_recommended_conf(SELF_TRAINED_WEIGHTS, root=root) is None
                       and resolve_effective_conf(DetectorConfig(), SELF_TRAINED_WEIGHTS,
                                                  auto=True, root=root) is None))
    return checks


def _check_conf_recommend_rule() -> list[tuple[str, bool]]:
    """训练端"推荐阈值"的规则：取**保住 80% 命中且过半图有检出**的最严门槛。

    用真实扫描表（``models/ball_yolo.report.json`` 的 20 epoch 结果）当样例：
    0.02 → 88% / 617 图，0.03 → 79% / 557，0.05 → 49% / 348，0.08 → 23% / 166。
    所以答案必须是 0.03：再往上提（0.05）召回腰斩，往下放（0.02）只多收误检。
    """
    tool = _load_train_tool()
    checks: list[tuple[str, bool]] = []

    scan = {f"{c:.2f}": {"recall_30px": r, "n_matched": m} for c, r, m in (
        (0.02, 0.8795, 617), (0.03, 0.7939, 557), (0.05, 0.4920, 348),
        (0.08, 0.2337, 166), (0.10, 0.1379, 100), (0.15, 0.0247, 18),
        (0.25, 0.0, 0))}
    rec = tool.recommend_conf(scan, n_with_boxes=689)
    rec_text = f"{rec[0]:g}" if rec else "无"
    checks.append((f"真实扫描表 → {rec_text}（应为 0.03："
                   "保住 80% 命中前提下最严，而不是命中率最高的 0.02）",
                   rec is not None and abs(rec[0] - 0.03) < 1e-9))

    # 覆盖率那条独立起作用：0.03 命中够但检出图太少，答案要往后挪
    thin = {"0.02": {"recall_30px": 0.88, "n_matched": 689},
            "0.03": {"recall_30px": 0.87, "n_matched": 300},
            "0.05": {"recall_30px": 0.85, "n_matched": 400}}
    rec_thin = tool.recommend_conf(thin, n_with_boxes=689)
    checks.append(("检出覆盖率不足（300 < 半数 344）时跳过该档 → 退到 0.05",
                   rec_thin is not None and abs(rec_thin[0] - 0.05) < 1e-9))

    dead = {f"{c:.2f}": {"recall_30px": 0.0, "n_matched": 0}
            for c in tool.RECOMMEND_CONF_GRID}
    checks.append(("全表命中率为 0（模型没学会）→ 不给推荐值（不往 sidecar 写假阈值）",
                   tool.recommend_conf(dead, n_with_boxes=689) is None))

    pair = tool.conf_sidecar_payload(
        {"headline": {"n_val_with_boxes": 689, "median_top1_conf": 0.055}, "by_conf": scan},
        dataset="outputs/ball_dataset")
    needed = ("recommended_conf", "median_top1_conf", "dataset", "created")
    checks.append(("sidecar 的四个必填键都在",
                   pair is not None and all(k in pair[0] for k in needed)))

    # 反向自证：把 RECALL_KEEP 放到 0.2，"最严"就一路爬到 0.25 —— 说明断言钉在这条规则上
    original = tool.RECALL_KEEP
    try:
        tool.RECALL_KEEP = 0.2
        loose = tool.recommend_conf(
            {"0.02": {"recall_30px": 0.90, "n_matched": 620},
             "0.25": {"recall_30px": 0.30, "n_matched": 600}}, n_with_boxes=689)
    finally:
        tool.RECALL_KEEP = original
    checks.append(("阈值语义：RECALL_KEEP 放松到 0.2 时同一张表会选 0.25（规则真的在起作用）",
                   loose is not None and abs(loose[0] - 0.25) < 1e-9
                   and original == 0.8))
    return checks


def _check_run_video_conf_flags() -> list[tuple[str, bool]]:
    """``run_video`` 的两个旋钮必须"未指定"与"指定了默认值"可区分。

    ``--conf`` 的缺省值**绝不能**是 0.25：那样"用户没说话"和"用户说了 0.25"
    在配置里长得一模一样，阈值校准会替一个明确表过态的人改门槛。
    """
    from apps import run_video as rv

    checks: list[tuple[str, bool]] = []
    old = sys.argv
    try:
        sys.argv = ["run_video.py", "x.mkv"]
        quiet = rv.parse_args()
        sys.argv = ["run_video.py", "x.mkv", "--conf", "0.25", "--imgsz", "960"]
        loud = rv.parse_args()
    finally:
        sys.argv = old
    checks.append(("run_video：--conf 默认是 None（能区分「没给」与「给了 0.25」）",
                   quiet.conf is None))
    checks.append(("run_video：--imgsz 默认是 None（不覆盖配置里的 yolo_imgsz）",
                   quiet.imgsz is None))
    checks.append(("run_video：显式 --conf / --imgsz 照收",
                   loud.conf == 0.25 and loud.imgsz == 960))
    return checks


def _check_radius_evidence() -> list[tuple[str, bool]]:
    """无种子时"检测器框半径有没有边缘证据"的判定（run_video 的 [半径] 那行）。

    合成图的构造刻意分两种：
    - **整圈**都有边（实心圆盘）：轮廓 100% 有梯度 → 必须判 ok；
    - **只有 1/3 圆弧**有边：对应现场那颗镜面球（实测下半圈暗对暗，整圈只有 25%~35%）。
      这里不用"正好一半"当弱样例：0.5 就是阈值本身，半个圆测出来 0.51，
      拿它做断言等于在阈值上跳舞，改一行无关代码就会翻。
    """
    import numpy as np
    import cv2

    from apps import run_video as rv

    checks: list[tuple[str, bool]] = []

    def full_disk() -> "np.ndarray":
        img = np.full((320, 320), 20, np.uint8)
        cv2.circle(img, (160, 160), 90, 200, -1)
        return img

    def partial_arc() -> "np.ndarray":
        img = np.full((320, 320), 20, np.uint8)
        # 200°~320° 的扇形：约 1/3 圈有边，其余与背景同灰度（真实里的"暗对暗"）
        cv2.ellipse(img, (160, 160), (90, 90), 0, 200, 320, 200, -1)
        return img

    s_full, v_full = rv.radius_evidence(full_disk(), 160, 160, 90)
    checks.append((f"整圈有边 → ok（支持度 {s_full:.2f} ≥ {rv.RADIUS_EVIDENCE_MIN_SUPPORT}）",
                   v_full == "ok" and s_full >= rv.RADIUS_EVIDENCE_MIN_SUPPORT))

    s_weak, v_weak = rv.radius_evidence(partial_arc(), 160, 160, 90)
    checks.append((f"只有 1/3 圈有边 → weak（支持度 {s_weak:.2f} < 0.5）",
                   v_weak == "weak" and s_weak < 0.5))

    # 初值偏大/偏小（框半宽常见的两种误差方向）也不能凭空变成"有证据"
    near = [rv.radius_evidence(partial_arc(), 160, 160, r)[1] for r in (76, 104)]
    checks.append(("半径偏 ±15% 时同样判 weak（不许靠别处的边凑数）",
                   near == ["weak", "weak"]))

    text_weak = "\n".join(rv.radius_evidence_lines(121.0, s_weak, "weak"))
    checks.append(("weak 文案必须写明「未经边缘验证」并给出深度影响",
                   "未经边缘验证" in text_weak and "Z = f·D/(2r)" in text_weak))
    text_ok = "\n".join(rv.radius_evidence_lines(91.0, s_full, "ok"))
    checks.append(("ok 文案不许出现「未经边缘验证」",
                   "未经边缘验证" not in text_ok))

    # 反向自证：阈值 0.0 时，"只有 1/3 圈"也必须判 ok —— 说明这条测试真的在守阈值
    original = rv.RADIUS_EVIDENCE_MIN_SUPPORT
    try:
        rv.RADIUS_EVIDENCE_MIN_SUPPORT = 0.0
        flipped = rv.radius_evidence(partial_arc(), 160, 160, 90)[1]
    finally:
        rv.RADIUS_EVIDENCE_MIN_SUPPORT = original
    checks.append(("阈值语义：MIN_SUPPORT 降到 0 时同一张图会判 ok（说明断言钉在阈值上）",
                   flipped == "ok" and original == 0.5))
    return checks


def main() -> int:
    print("=" * 70)
    print("插件注册表自测（清单唯一 / 未知名字报错 / 默认唯一 / 边界）")
    print("=" * 70)

    results: list[tuple[str, bool]] = []
    for section, fn in (
        ("1. 注册表不变量", _check_registry),
        ("2. 查找与注册规则（R2/R3/R4/R6）", _check_lookup),
        ("3. 唯一清单（R7）", _check_no_manual_lists),
        ("4. 库边界（R5）", _check_library_boundary),
        ("5. 与 config / 工厂的一致性", _check_wiring),
        ("6. YOLO 自训接入（类别过滤 / 权重优选 / 训练前置检查）",
         lambda: (_check_yolo_classes_filter()
                  + _check_weights_preference()
                  + _check_train_tool_preflight())),
        ("7. 半径证据（无种子时 [半径] 告警）", _check_radius_evidence),
        ("8. 阈值校准（sidecar / 推荐规则 / --conf 缺省值）",
         lambda: (_check_conf_calibration()
                  + _check_conf_recommend_rule()
                  + _check_run_video_conf_flags())),
    ):
        print(f"----- {section} -----")
        got = fn()
        for name, ok in got:
            print(f"  [{'通过' if ok else '不达标'}] {name}")
        results.extend(got)
        print()

    # clear() 仅供测试：清空后 names 为空（否则下一个人会以为有"持久化"）
    clear()
    results.append(("clear() 后注册表为空（只给测试用）", names("detector") == []))
    print(f"  [{'通过' if results[-1][1] else '不达标'}] {results[-1][0]}")

    ok = all(v for _n, v in results)
    bad = [n for n, v in results if not v]
    if bad:
        print(f"\n不达标 {len(bad)} 项：")
        for name in bad:
            print(f"  · {name}")
    print("\n判定结果：" + ("全部通过" if ok else "有不达标项"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
