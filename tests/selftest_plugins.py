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
框内细化继承全局 hough_max_radius（80）      `sub` 不按框尺寸定搜索窗口             exit=1
某个检测器又自己 new 一个锚点 Detection      往 color 的 `_rank` 里插一份构造        exit=1
原始录像名又写回手写元组                    在 view_3d 里写 `("left.mkv","left.mp4")`  exit=1
原始录像的 mkv 优先被打乱                    `LEFT_VIDEO_NAMES` 改成 mp4 在前      exit=1
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

from sphere_tracker.config import ColorConfig, DetectorConfig          # noqa: E402
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
ENTRY_FILES = sorted((ROOT / "apps").glob("*.py")) + [ROOT / "apps/gui_main.py"]
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


def _check_refine_in_box_window() -> list[tuple[str, bool]]:
    """``refine_circle_in_box`` 的搜索半径窗口：按**框尺寸**定，不继承全局 hough_max_radius。

    现场（2026-10-31）：自训 ``models/ball_yolo.pt`` 的细化成功率只有 11.5%，
    88.5% 的帧退回用 YOLO 框中心当圆心（框中心误差中位 4.29 px）。根因是子配置
    **原样继承**了全局 ``hough_max_radius=80``（镜面球像素半径约 90 > 80，hough
    必然找不到），而镜面球在 color 路径下又分割不出来 —— 细化必然失败。那个 80
    是给**全图** hough 定的性能上限，在框内 ~180×180 的 ROI 里不适用。

    不依赖 ultralytics：直接手造 ``Detection(extra={"box": ...})`` 再调细化函数。
    """
    import numpy as np
    import cv2

    from sphere_tracker.detectors.neural import refine_circle_in_box
    from sphere_tracker.types import Detection

    checks: list[tuple[str, bool]] = []

    def scene(radius: float, box_half: float, highlight: tuple[float, float] | None = None):
        """浅灰背景 + 深灰实心球（模拟金属/镜面球）；可选在偏移处加一个亮白高光斑。

        边缘用 ``LINE_AA`` 画（真实相机的球边缘是抗锯齿的）：硬边渲染下
        ``HoughCircles`` 在 ``param2=30`` 只会得到零星票数、检不出，那是**合成图的
        假象**而非代码缺陷 —— 用抗锯齿边更贴近现场，也才能让本用例验证到真东西。
        """
        size = int(2 * box_half) + 120
        frame = np.full((size, size, 3), 200, np.uint8)
        cx = cy = size // 2
        cv2.circle(frame, (cx, cy), int(radius), (90, 90, 90), -1, lineType=cv2.LINE_AA)
        if highlight is not None:
            off, hr = highlight
            cv2.circle(frame, (int(cx + off), cy), int(hr), (255, 255, 255), -1,
                       lineType=cv2.LINE_AA)
        det = Detection(
            x=float(cx), y=float(cy), radius=box_half, score=0.9, method="yolo",
            extra={"box": (cx - box_half, cy - box_half, cx + box_half, cy + box_half)},
        )
        return frame, det, float(cx), float(cy)

    cfg = DetectorConfig()

    # 1) 主用例：球半径 90 > 旧硬顶 80 —— 这一条就是"改前必红"的锚点。
    frame, det, cx, cy = scene(90.0, 90.0)
    out = refine_circle_in_box(frame, det, cfg)
    win = out.parts.get("refine_window") if out else None
    checks.append((f"半径 90 的球：细化窗口上界 > 80（{win}；继承全局 hough_max_radius 时为 80）",
                   win is not None and win[1] > 80.0))
    checks.append(("半径 90 的球：细化成功（method 以 yolo+ 开头，没退回框中心）",
                   out is not None and out.method.startswith("yolo+")))
    # 4) 主用例的圆心精度（相对真球心）
    if out is None:
        checks.append(("半径 90 的球：圆心误差 < 2 px", False))
    else:
        err = float(np.hypot(out.x - cx, out.y - cy))
        checks.append((f"半径 90 的球：圆心误差 {err:.2f} px < 2 px", err < 2.0))

    # 2) 高光斑陷阱：偏离球心 40 px 处有个半径 20 px 的亮白斑。
    #    下界若不收紧（沿用全局 min_radius=8），hough 会细化到高光斑上、圆心偏到球面一侧。
    frame, det, cx, cy = scene(90.0, 90.0, highlight=(40.0, 20.0))
    out = refine_circle_in_box(frame, det, cfg)
    hx, hy = cx + 40.0, cy
    if out is None:
        checks.append(("高光斑陷阱：细化成功且圆心贴真球心（未跑偏到高光斑）", False))
    else:
        d_ball = float(np.hypot(out.x - cx, out.y - cy))
        d_hi = float(np.hypot(out.x - hx, out.y - hy))
        checks.append((f"高光斑陷阱：圆心到真球心 {d_ball:.2f} px 远小于到高光斑 {d_hi:.2f} px",
                       d_ball < d_hi))
        checks.append((f"高光斑陷阱：圆心贴真球心（{d_ball:.2f} px < 3 px）", d_ball < 3.0))

    # 3) 不劣化：球半径 60 但框给半宽 200 的松框 —— 窗口 [100, 200] 框不住真球，
    #    此时必须**安静地退回框中心**（不抛异常、method 仍是 yolo）。
    frame, det, cx, cy = scene(60.0, 200.0)
    try:
        out = refine_circle_in_box(frame, det, cfg)
    except Exception:                                   # noqa: BLE001
        out = "raised"
    if out == "raised":
        checks.append(("松框：不抛异常", False))
    else:
        result = out or det                            # 复刻调用方 `or det` 的退回语义
        bx = 0.5 * (det.extra["box"][0] + det.extra["box"][2])
        by = 0.5 * (det.extra["box"][1] + det.extra["box"][3])
        checks.append(("松框（半宽 200 / 球半径 60）：不抛异常，且退回框中心（method=yolo）",
                       result.method == "yolo"
                       and abs(result.x - bx) < 1e-6 and abs(result.y - by) < 1e-6))
    return checks



def _check_deterministic_rng() -> list[tuple[str, bool]]:
    """检测结果必须可复现：CSRT 用 OpenCV 的**进程级全局 RNG**，不固定种子时
    同一段视频每次启动进程给出的检出帧数都不一样（实测 222 / 198 / 283）。
    对一个用来比精度的工具，这层噪声会盖住真实差异。
    """
    from sphere_tracker.detectors import base

    calls: list[int] = []
    orig = base.cv2.setRNGSeed
    base.cv2.setRNGSeed = lambda v: calls.append(v)      # type: ignore[assignment]
    try:
        base._rng_seeded = False
        base.ensure_deterministic_rng()
        base.ensure_deterministic_rng()                  # 第二次必须什么都不做
    finally:
        base.cv2.setRNGSeed = orig                       # type: ignore[assignment]
        base._rng_seeded = True

    from sphere_tracker.config import DetectorConfig
    return [
        ("RNG 只固定一次，且用固定常量（不是时间/随机）",
         calls == [base.DETERMINISTIC_RNG_SEED]),
        ("回捕默认关闭（会锁到日光灯管，见待办 B-163）",
         DetectorConfig().specular.reacquire is False),
    ]

def _check_detector_behaviour() -> list[tuple[str, bool]]:
    """检测器行为：点选取色 / 种子帧兜底 / 帧间一致性 / auto / 融合 / 半径锚定。

    这些是"看数字才能发现"的行为，注册表本身证明不了。用合成图是刻意的：
    真值已知，断言能钉在像素上；真实录像只能给"看起来对"。
    """
    import cv2
    import numpy as np

    checks: list[tuple[str, bool]] = []
    red = np.array([0, 0, 200], dtype=np.uint8)       # BGR → H≈0, S=255, V=200

    def scene(pts, size=(480, 640)):
        img = np.full((size[1], size[0], 3), 128, dtype=np.uint8)
        for x, y, r in pts:
            cv2.circle(img, (int(x), int(y)), int(r), red.tolist(), -1)
        return img

    # ---- 1) 预设不匹配 = 0（旧行为，留着当反证）----
    det = build_single_detector("color", DetectorConfig())
    checks.append(("color：白球预设看不见红球（已知的静默失败形态）",
                   det.detect(scene([(320, 240, 40)])) == []))

    # ---- 2) 点选之后必须看见：种子从点击处取色，覆盖预设 ----
    det = build_single_detector("color", DetectorConfig())
    det.seed(320, 240, 40)
    got = det.detect(scene([(320, 240, 40)]))
    ok = (len(got) == 1 and abs(got[0].x - 320) <= 1.5
          and abs(got[0].y - 240) <= 1.5 and abs(got[0].radius - 40) <= 1.5
          and got[0].score > 0.9)
    checks.append(("color：点选球心后，白预设也能检出红球（种子取色）", ok))

    # ---- 3) 完全无法颜色分割时，种子帧仍输出 Detection（不许说"没看见"）----
    det = build_single_detector("color", DetectorConfig())
    det.seed(200, 150, 30)
    blank = np.full((300, 400, 3), 128, dtype=np.uint8)
    got = det.detect(blank)
    anchor_ok = (len(got) == 1 and got[0].extra.get("seed_anchor") is True
                 and abs(got[0].radius - 30) < 1e-6)
    after = det.detect(blank)
    checks.append(("color：种子帧颜色分割失败 → 按种子输出（seed_anchor）", anchor_ok))
    checks.append(("color：种子帧之后不拿旧位置一直冒充观测", after == []))

    # ---- 4) auto：不说球色也检出 ----
    det = build_single_detector(
        "color", DetectorConfig(color=ColorConfig(mode="auto")))
    got = det.detect(scene([(320, 240, 40)]))
    checks.append(("color：--preset auto 不需要点选/预设也能检出",
                   len(got) == 1 and abs(got[0].x - 320) <= 1.5))

    # ---- 5) 帧间一致性：同色干扰进来后 top-1 不能换人 ----
    det = build_single_detector(
        "color", DetectorConfig(color=ColorConfig.preset("red_ball")))
    det.seed(320, 240, 40)
    first = det.detect(scene([(320, 240, 40)]))
    second = det.detect(scene([(320, 240, 40), (100, 100, 40)]))
    checks.append(("color：两个同色同分候选时，跟住上一帧那颗（不按扫描顺序）",
                   len(first) == 1 and len(second) >= 1
                   and abs(second[0].x - 320) <= 1.5 and abs(second[0].y - 240) <= 1.5))

    # ---- 6) min_score 真的接到 color（B-178）----
    det = build_single_detector(
        "color", DetectorConfig(min_score=0.99,
                                color=ColorConfig.preset("red_ball")))
    checks.append(("color：min_score 对 color 生效（调高到 0.99 → 0 候选）",
                   det.detect(scene([(320, 240, 40)])) == []))

    # ---- 7) 融合：同色干扰 + 时间连续性 ----
    det = build_single_detector(
        "fusion", DetectorConfig(color=ColorConfig.preset("red_ball")))
    det.seed(320, 240, 40)
    det.detect(scene([(320, 240, 40)]))
    got = det.detect(scene([(320, 240, 40), (100, 100, 40)]))
    checks.append(("fusion：两个同色目标时仍取上一帧那颗（聚类 + 时间连续性）",
                   len(got) == 1 and abs(got[0].x - 320) <= 2.0))

    # ---- 8) specular：半径锚在种子上，不是框的一半 ----
    det = build_single_detector("specular", DetectorConfig())
    det.seed(200, 150, 20)
    ball = np.full((300, 400, 3), 90, dtype=np.uint8)
    yy, xx = np.ogrid[:300, :400]
    inside = (xx - 200) ** 2 + (yy - 150) ** 2 <= 20 ** 2
    ball[inside & (((xx // 4 + yy // 4) % 2) == 0)] = (190, 190, 190)
    got = det.detect(ball)                  # 种子帧
    track = det.detect(ball)                # 跟踪帧
    ok = (len(got) == 1 and abs(got[0].radius - 20) < 1e-6
          and len(track) == 1 and abs(track[0].radius - 20) <= 1.0)
    checks.append(("specular：输出半径 = 种子半径（框半径只作缩放比例）", ok))

    # ---- 9) specular：CSRT 起不来时，种子帧也不许报"没看见" ----
    det = build_single_detector("specular", DetectorConfig())
    det.seed(120, 100, 12)
    orig = det._make_tracker
    det._make_tracker = lambda: None                       # type: ignore[assignment]
    try:
        got = det.detect(ball)
    finally:
        det._make_tracker = orig                           # type: ignore[assignment]
    checks.append(("specular：CSRT 不可用 → 种子帧仍输出 Detection（seed-anchor）",
                   len(got) == 1 and got[0].extra.get("state") == "seed-anchor"
                   and abs(got[0].radius - 12) < 1e-6))

    # ---- 10) 预设清单：auto 是策略，不是颜色 ----
    checks.append(("preset_names() 不含 auto（它不是一种颜色）",
                   "auto" not in ColorConfig.preset_names()))
    checks.append(("preset_choices() 含 auto，且能落到 mode=auto",
                   ColorConfig.preset_choices()[0] == "auto"
                   and ColorConfig.preset("auto").mode == "auto"))
    return checks


def _anchor_call_sites(src: str, label: str = "<src>") -> list[str]:
    """扫一段源码：``Detection(..., extra={..., "seed_anchor": ...})`` 出现在哪。

    抽成独立函数只为一件事：**判据自己要能证明抓得到**。一条永远返回空列表
    的扫描器也是"通过" —— 本项目已经吃过一次「闸门不拦人」的亏，所以下面
    用一段伪造源码做自检。
    """
    sites: list[str] = []
    for node in ast.walk(ast.parse(src, filename=label)):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name != "Detection":
            continue
        for kw in node.keywords:
            if kw.arg != "extra" or not isinstance(kw.value, ast.Dict):
                continue
            for key in kw.value.keys:
                if isinstance(key, ast.Constant) and key.value == "seed_anchor":
                    sites.append(f"{label}:{node.lineno}")
    return sites


def _check_anchor_single_source() -> list[tuple[str, bool]]:
    """锚点只能从 `BaseDetector.anchor_detection()` 一处构造。

    为什么要有这条（2026-11-07，同一个毛病第三次之后）
    --------------------------------------------------
    `color` / `generic` / `specular` 原来各写一份 `Detection(...)`：`method`
    有的带 ``:seed`` 有的不带、`extra` 有的带 `state`/`center_source` 有的不带、
    `last_error` 有的写原因有的不写。而 `pipeline` 只认 `extra["seed_anchor"]`
    —— `specular` 那一路连键都没有，锚点被当成真检测**静默**放过去，两轮
    Critic 才抓出来。B-150 也是同型（同一件事三处实现，改一处漏两处）。

    用 AST 而不是字符串匹配：注释里提到 `seed_anchor` 不该算数（字面量判据
    被自己的修复注释误伤过一次，见 `docs/交接文档.md` §75.5）。
    """
    files = sorted((ROOT / "sphere_tracker" / "detectors").glob("*.py"))
    sites: list[str] = []
    for path in files:
        sites += _anchor_call_sites(path.read_text(encoding="utf-8"), path.name)
    in_base = [s for s in sites if s.startswith("base.py:")]
    outside = [s for s in sites if not s.startswith("base.py:")]
    if outside:
        print("    绕过 BaseDetector.anchor_detection 的构造点：")
        for s in outside:
            print(f"      · {s}")

    # 自检 1：同样的扫描器喂一段「散装锚点」，必须抓得到
    fake = ("def f(self):\n"
            "    return Detection(x=1, y=2, radius=3,\n"
            "                     extra={'seed_anchor': True})\n")
    scanner_ok = len(_anchor_call_sites(fake, "fake.py")) == 1
    # 自检 2：注释里出现的 `seed_anchor` 不算（AST 不解析注释）
    commented = ("# 以前这里写过 extra={'seed_anchor': True}\n"
                 "x = 1\n")
    comment_ok = _anchor_call_sites(commented, "c.py") == []

    # 语义：note 只追加不覆盖（specular 在锁定失败前可能刚写过「半径超范围」）
    det = build_single_detector("generic", DetectorConfig())
    det.last_error = "先前的原因"
    d = det.anchor_detection(11.0, 22.0, 33.0, note="这次的原因")
    geom_ok = (d.extra.get("seed_anchor") is True and d.method.endswith(":seed")
               and abs(d.x - 11.0) < 1e-9 and abs(d.y - 22.0) < 1e-9
               and abs(d.radius - 33.0) < 1e-9 and d.score == 1.0)
    append_ok = ("先前的原因" in det.last_error and "这次的原因" in det.last_error)
    det2 = build_single_detector("generic", DetectorConfig())
    det2.last_error = "原有诊断"
    det2.anchor_detection(1.0, 2.0, 3.0)          # 不传 note
    keep_ok = det2.last_error == "原有诊断"

    return [
        (f"锚点构造只有一处（扫 {len(files)} 个检测器文件：共命中 {len(sites)} 处，"
         f"其中 base.py {len(in_base)} 处）",
         len(in_base) == 1 and not outside),
        ("判据自检：散装锚点能被扫出来（不是永远判绿）", scanner_ok),
        ("判据自检：注释里的 seed_anchor 不算（AST 不误伤注释）", comment_ok),
        ("anchor_detection：位置/半径原样、score=1.0、method 以 :seed 结尾", geom_ok),
        ("anchor_detection：note 追加而非覆盖（不丢上一条原因）", append_ok),
        ("anchor_detection：不传 note 时不动已有诊断", keep_ok),
    ]


def _check_size_ladder_and_gui_state() -> list[tuple[str, bool]]:
    """13. 尺寸门自动阶梯 + 装置档案门控（2026-11-08）。

    全是**行为判据**，不是"代码里有没有某个字"：把修复退回去，下面会真的变红。
    反向自证用的是"同一状态机、喂另一种输入"，不改源码也不改判据。
    """
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import patch

    from sphere_tracker.detectors.size_ladder import SizeLadder
    import apps._gui_app as gui

    root = Path(__file__).resolve().parents[1]
    out: list[tuple[str, bool]] = []

    # ---- A. 阶梯三态 ----
    a = SizeLadder(image_h=1080, probe_frames=4)
    for _ in range(4):
        a.observe(True)
    out.append(("阶梯：稳定检出 → 锁死在首档 194px",
                a.locked and a.index == 0 and abs(a.window[1] - 194.4) < 0.01))

    b = SizeLadder(image_h=1080, probe_frames=4)
    for _ in range(4):
        b.observe(False, 250.0)          # 门外有更大的、只被尺寸门挡住的候选
    out.append(("阶梯：门外 250px > 194px → 升到第二档 324px",
                b.index == 1 and abs(b.window[1] - 324.0) < 0.01))

    c = SizeLadder(image_h=1080, probe_frames=4)
    for _ in range(4):
        c.observe(False, None)           # 门外什么都没有：掩膜/形状的问题
    out.append(("反证：门外没有更大候选 → 绝不升档、判 no_candidate",
                c.index == 0 and c.state == "no_candidate"))

    d = SizeLadder(image_h=1080, probe_frames=4, steps=(0.18,))
    for _ in range(4):
        d.observe(False, 999.0)
    out.append(("阶梯：走完最后一档仍不行 → exhausted（不静默）",
                d.state == "exhausted"))
    out.append(("阶梯按画面高度缩放：720p 首档 = 129.6px",
                abs(SizeLadder(image_h=720).window[1] - 129.6) < 0.01))

    ladder_src = (root / "sphere_tracker/detectors/size_ladder.py").read_text(
        encoding="utf-8")
    out.append(("反证：升档条件写死在源码里（去掉就红）",
                "cur.blocked_max is not None and cur.blocked_max > cur.r_max"
                in ladder_src))

    # ---- B. 装置档案门控（mock active profile，绝不改用户 config）----
    def state(name, mode, calib):
        prof = SimpleNamespace(name=name, stereo_mode=mode,
                               calibration_path=calib)
        with patch("stereo.profiles.get_active", return_value=prof):
            return gui.AppState()

    par = state("平行装置", "parallel", "config/calib/平行装置")
    p_ok, p_txt = par.parallel_status()
    a_ok, a_txt = par.angled_status()
    out.append(("档案=平行+有平行标定 → 平行就绪、大角度不适用",
                p_ok and not a_ok and "不适用" in a_txt))
    out.append(("门厅读到的是档案自己的标定（68.8mm，不是顶层 66.3mm）",
                abs((par.baseline_mm() or 0) - 68.84) < 0.1))
    out.append(("反证：大角度卡不再念平行标定的基线数字",
                "基线" not in a_txt))

    # 2026-11-11：原用例借「实验性质平行装置1」当"配了标定但文件不在"的样本 ——
    # 而用户当天 08:34 给这套装置做了真机标定、`config/calib/实验性质平行装置1.npz`
    # 真的落地了，前提消失，三条语义判据**无辜变红**。
    # 换成测试自己的基名（没人会拿它去标定），并先断言"它确实不存在"：
    # 将来真有人建了，红的是这条前提，而不是下面三条语义断言。
    absent = ROOT / "config" / "calib" / "__selftest_未标定的装置__"
    out.append(("前提：用例的标定基名确实不存在（不拿用户真实标定当样本）",
                not any(absent.with_suffix(s).exists() for s in (".npz", ".json"))))
    empty = state("用例装置（无标定）", "parallel",
                  "config/calib/__selftest_未标定的装置__")
    out.append(("档案配了标定但文件不在 → 不许回落到顶层默认",
                empty.rig is None
                and empty._stereo_calib_path().name != "stereo_calibration"
                and any("不存在" in n for n in empty.notes)))
    out.append(("反证：该档案下大角度同样被挡（不因无标定而放行）",
                not empty.angled_status()[0]))

    conflict = state("冲突例", "angled", "config/calib/平行装置")
    out.append(("档案模式与标定 meta 冲突 → 两张卡都拒绝，不静默挑一个信",
                "别套装置" in conflict.parallel_status()[1]
                and "别套装置" in conflict.angled_status()[1]))

    out.append(("单目明确标注为全局独立（与双目档案分开管理）",
                "全局" in par.mono_status()[1]))

    hub_src = (root / "apps/_gui_app.py").read_text(encoding="utf-8")
    out.append(("反证：模式不适用/冲突时「进入」按钮真的被禁用",
                'state="normal" if allowed else "disabled"' in hub_src))
    out.append(("未标定的平行档案仍可进入（窗口里才有标定按钮，不能锁在门外）",
                empty.mode_allowed("parallel")
                and not empty.mode_allowed("angled")))
    out.append(("反证：大角度档案禁用平行卡；冲突档案两张卡都禁用",
                not state("90度装置", "angled", "config/calib/90度装置").mode_allowed("parallel")
                and not conflict.mode_allowed("parallel")
                and not conflict.mode_allowed("angled")))
    out.append(("反证：点击时按当下状态复核（不是只在刷新时判一次）",
                "def _enter_mode" in hub_src))
    return out


def _check_gui_wiring_and_mono_identity() -> list[tuple[str, bool]]:
    """14. GUI 尺寸门开关 / 主链路显式报错 / 单目设备身份（2026-11-08）。"""
    from pathlib import Path
    from unittest.mock import patch

    from apps import _actions
    from calib.mono_calib import MonoCalib
    import apps._gui_app as gui

    out: list[tuple[str, bool]] = []
    cmd_on = _actions.offline_stereo("s", method="color", radius_auto=True)
    cmd_off = _actions.offline_stereo("s", method="color", radius_auto=False)
    # :1043 保持：GUI 恒走阶梯，不加 --no-radius-auto。
    out.append(("GUI「尺寸门自动阶梯」常开 → 不加 --no-radius-auto",
                "--no-radius-auto" not in cmd_on))
    # :1045 按 2026-11-14 新行为改写：GUI 已**移除**「取消勾选」入口
    # （BallSettings 里不再创建那个 Checkbutton，radius_auto() 恒 True），
    # 所以 GUI 永远不加 --no-radius-auto；但 CLI 的 --no-radius-auto 能力
    # 保留（命令行显式给 radius_auto=False 仍收到该开关）。
    # 用 AST 读 BallSettings 源码，判据才不会因"注释里提了一句"就假绿。
    import ast as _ast
    import inspect
    import textwrap
    from apps._gui_widgets import BallSettings
    _bs_tree = _ast.parse(textwrap.dedent(inspect.getsource(BallSettings)))
    _cb_texts = [kw.value.value
                 for n in _ast.walk(_bs_tree)
                 if isinstance(n, _ast.Call)
                 and isinstance(n.func, _ast.Attribute)
                 and n.func.attr == "Checkbutton"
                 for kw in n.keywords
                 if kw.arg == "text"
                 and isinstance(kw.value, _ast.Constant)
                 and isinstance(kw.value.value, str)]
    _no_gate_cb = "尺寸门自动阶梯" not in _cb_texts
    _ra_src = inspect.getsource(BallSettings.radius_auto)
    _always_true = ("return True" in _ra_src
                    and "radius_auto_var.get" not in _ra_src)
    out.append(("GUI 已无取消入口、恒走阶梯（radius_auto() 恒 True）；"
                "CLI --no-radius-auto 仍有效",
                _no_gate_cb and _always_true
                and "--no-radius-auto" in cmd_off))

    # 单目内参：device_identity 必须能往返，且换设备时 GUI 判为不可用
    d = MonoCalib.from_dict({"fx": 1000.0, "fy": 1000.0, "cx": 960.0,
                             "cy": 540.0, "image_size": [1920, 1080],
                             "device_identity": "USB\\AAA"})
    out.append(("单目标定 device_identity 往返不丢",
                d.device_identity == "USB\\AAA"
                and d.to_dict().get("device_identity") == "USB\\AAA"))
    out.append(("反证：老标定没有 device_identity 时如实留空",
                MonoCalib.from_dict({"fx": 1.0}).device_identity == ""))

    st3 = gui.AppState()
    st3.mono = dict(st3.mono or {}, fx=1000.0, device_identity="USB\\AAA")
    with patch.object(gui.AppState, "camera_roles",
                      return_value={"left": {"identity": "USB\\OTHER"}}):
        ok1, txt1 = st3.mono_status()
    out.append(("反证：单目标定绑定另一台相机 → 判不可用，不静默沿用旧 fx",
                not ok1 and "另一台相机" in txt1))

    return out


def _check_pose_region_mapping() -> list[tuple[str, bool]]:
    """15. 姿态区域归属：九个格子都要算数（2026-11-08 真机 38/60 被丢弃）。"""
    import numpy as np

    from calib.board_coverage import region_of
    from calib.mono_calib import _region_index

    out: list[tuple[str, bool]] = []
    W, H = 1920, 1080

    def cell(col, row, fx=0.5, fy=0.5):
        """第 (col,row) 个 3×3 格子里的一点，按 fx/fy 在该格内偏移。"""
        return ((col + fx) * W / 3.0, (row + fy) * H / 3.0)

    # 五个锚点仍各归自己（口径没变）
    anchors = {(0, 0): 0, (2, 0): 1, (0, 2): 2, (2, 2): 3, (1, 1): 4}
    out.append(("五个锚点仍归各自区域",
                all(_region_index(*cell(c, r), W, H) == want
                    for (c, r), want in anchors.items())))
    # 四个边中点格子：以前全返回 -1，现在必须落到某个区域
    edges = [(1, 0), (1, 2), (0, 1), (2, 1)]
    idxs = [_region_index(*cell(c, r, fx, fy), W, H)
            for (c, r) in edges for fx, fy in ((0.2, 0.5), (0.5, 0.2), (0.8, 0.8))]
    out.append(("反证：四边中点 12 个采样位置全部有归属（不再 -1）",
                all(0 <= i <= 4 for i in idxs)))
    out.append(("归并后仍是 5 个区域（没有冒出第 6 个）",
                set(idxs) <= set(range(5))))
    # 用真机同款角点数组走公开接口，确认不是只改了私有函数
    fake = np.array([[cell(1, 0)[0] + dx, cell(1, 0)[1] + dy]
                     for dx in (-40, 0, 40) for dy in (-40, 0, 40)], np.float32)
    out.append(("公开接口 region_of 对上中格子也返回有效区域",
                0 <= region_of(fake, (H, W)) <= 4))
    return out


def _check_dual_board_ref_pair() -> list[tuple[str, bool]]:
    """16. 双板链式参考对（2026-11-09 方案 A）：护栏 / argv / GUI 四步 / 行为。

    背景：链两端位姿必须来自**同一个支架摆放**。旧双板 GUI 各取内参段
    `argmin(per_view)` 一帧（板是移动的，几乎必然不同摆放），合成真值实测
    外参错 11.86° / 90.4 mm 而程序一声不响。方案 A 加「组合装置摆好别动、
    两路同时各拍 3 张」的参考对，链两端只从它取。
    """
    import tempfile
    from types import SimpleNamespace
    from unittest.mock import patch

    import cv2
    import numpy as np

    from apps import _actions
    import apps.calibrate_stereo as cs
    from apps.tools.check_chain_inputs import check_ref_pair
    from calib.board import BoardSpec

    out: list[tuple[str, bool]] = []
    spec = BoardSpec(cols=8, rows=5, square_mm=25.0)
    objp = spec.object_points()

    # ---- ① 护栏：缺参考对就拒绝出数（不许静默给错外参）----
    base = dict(left=None, right=None, left_video=None, right_video=None,
                same_board=False, left_dir="L", right_dir=None, board_b_dir="B",
                pair_frame="P", pair_camera="left", ref_left=None, ref_right=None,
                ref_pair_left=None, ref_pair_right=None)

    def mkargs(**kw):
        return SimpleNamespace(**{**base, **kw})

    msg = cs._chain_input_error(mkargs())
    out.append(("缺参考对 → 双板模式被拦，且信息点名第 ③ 步与错数代价",
                bool(msg) and "参考对" in msg and "③" in msg and "10°" in msg))
    out.append(("反证：补上参考对 → 放行",
                cs._chain_input_error(mkargs(ref_pair_left="rl",
                                             ref_pair_right="rr")) is None))
    out.append(("单板模式不受影响（它有自己的参考段）",
                cs._chain_input_error(mkargs(
                    same_board=True, right_dir="R", ref_left="a", ref_right="b",
                    pair_frame=None, pair_camera=None)) is None))

    # ---- ② argv 与 GUI 四步真的接上 ----
    cmd = _actions.calibrate_args("chain_dual", {}, "left", "right", "angled")
    out.append(("GUI 双板标定 → argv 真下传两个参考对目录",
                "--ref-pair-left" in cmd and "--ref-pair-right" in cmd
                and any("ref_pair_left" in x for x in cmd)
                and any("ref_pair_right" in x for x in cmd)))
    out.append(("反证：单板标定不掺参考对参数",
                not any(x.startswith("--ref-pair") for x in
                        _actions.calibrate_args("chain", {}, "l", "r", "angled"))))
    step3 = _actions.capture_chain_frames(stage="ref_pair", cols="8", rows="5",
                                          square="25", target="3")
    out.append(("第 ③ 步（target=3）argv：--target 3、双路同拍、按参考段判稳",
                "--both" in step3 and step3[step3.index("--target") + 1] == "3"
                and step3[step3.index("--stage") + 1] == "ref"
                and any("ref_pair_left" in x for x in step3)
                and any("ref_pair_right" in x for x in step3)))
    # 防硬编码反证（2026-11-11 Critic 抓到的阻断缺陷）：ref_pair 分支曾经
    # **忽略 target 入参**、写死 `"--target 1"` —— 界面说 3 对、实际只拍 1 对，
    # 而当时只钉 GUI 源码，所以全绿是假象。这里换一个不同的值再拼一次：
    # 写死 1 时这条必红。
    probe = _actions.capture_chain_frames(stage="ref_pair", cols="8", rows="5",
                                          square="25", target="7")
    out.append(("反证：ref_pair 的 target 真的来自调用方（写死 1 会红）",
                probe[probe.index("--target") + 1] == "7"))
    out.append(("chain_dirs() 提供两个新目录",
                {"ref_pair_left", "ref_pair_right"} <= set(_actions.chain_dirs())))

    gui_src = Path("apps/_gui_stereo.py").read_text(encoding="utf-8")
    # 2026-11-11 用户要求：③ 参考对 1 张 → 3 张（两路同时各拍 3 张）；④ 仍是 5 张。
    # 只判 `"target=\"3\"" in gui_src` 是空转 —— 单板分支的参考段本来就有 3，
    # 所以截出 `stage="ref_pair"` 之后那一段再判。
    ref_pair_seg = (gui_src.split('stage="ref_pair"')[1][:260]
                    if 'stage="ref_pair"' in gui_src else "")
    out.append(("GUI 第 ③ 步 ref_pair 的目标张数是 3，且 ③ 标签写「各拍 3 张」",
                'target="3"' in ref_pair_seg and "各拍 3 张" in gui_src))
    out.append(("GUI 双板流程真的有第 ③ 步（stage=ref_pair）",
                'stage="ref_pair"' in gui_src))
    out.append(("GUI 四步标签齐全，且都是人话",
                all(k in gui_src for k in ("① 拍 A 板", "② 拍 B 板",
                                           "③ 组合装置", "④ 让一台相机"))))
    out.append(("GUI 标定缺图检查把参考对列为必需",
                '"ref_pair_left"' in gui_src and '"ref_pair_right"' in gui_src))
    core = Path("apps/calibrate_stereo.py").read_text(encoding="utf-8")
    out.append(("反证：旧的错误做法（各取 argmin(per_view)）已删除",
                'np.argmin(mono_l["per_view"])' not in core))
    out.append(("链两端位姿只从参考对取",
                "_ref_pair_poses(" in core
                and "T_camL_A, T_camR_B, rp_info" in core))
    cci_src = Path("apps/tools/check_chain_inputs.py").read_text(encoding="utf-8")
    out.append(("自检 dual 模式会查参考对、下一步命令也带新参数",
                "check_ref_pair(root, spec)" in cci_src
                and "--ref-pair-left" in cci_src))
    out.append(("自检：单张同框图不再只警告（与标定拒绝出数口径一致）",
                "标定会直接拒绝出数" in cci_src))
    out.append(("GUI 大角度窗把卷尺基线接进标定命令（§67 A1）",
                "var_baseline" in gui_src
                and "baseline=self.var_baseline.get()" in gui_src
                and "卷尺量两镜头中心距" in gui_src))
    out.append(("反证：卷尺基线填了才下传、留空不下传（不替用户猜数）",
                "--baseline-mm" in _actions.calibrate_args(
                    "chain_dual", {}, "l", "r", "angled", baseline="124")
                and "--baseline-mm" not in _actions.calibrate_args(
                    "chain_dual", {}, "l", "r", "angled", baseline="")))

    # ---- ④ 真实鼠标点按器的静态根因与「不假点击」加固 ----
    ct_src = Path("apps/tools/gui_clickthrough.py").read_text(encoding="utf-8")
    agents_src = Path("AGENTS.md").read_text(encoding="utf-8")
    out.append(("真实点按器：点击前用 WindowFromPoint 核对坐标归属",
                "WindowFromPoint" in ct_src
                and "raise SystemExit(8)" in ct_src
                and ct_src.count("_require_point_on_window(hwnd, x, y, tag)") == 2))
    out.append(("真实点按器：exit 5 打印窗口尺寸诊断（比例坐标已知缺陷）",
                "诊断：门厅 rect" in ct_src
                and "calibrate_hub.png" in ct_src
                and "比例坐标依赖窗口尺寸" in ct_src))
    out.append(("工具表登记 exit 8 与已知根因",
                "坐标不在目标窗口上（8）" in agents_src
                and "比例坐标依赖窗口尺寸" in agents_src))

    # ---- ③ `_ref_pair_poses` 的行为：配对 / 尺寸 / 报错 ----
    class _FakeObs:
        found = True
        corners = np.zeros((40, 2), np.float32)

    K = np.eye(3)
    D = np.zeros(5)
    T_eye = np.eye(4)

    def _call(pl, pr, size_l=(640, 480), size_r=(640, 480)):
        with patch.object(cs, "detect_board", return_value=_FakeObs()), \
             patch.object(cs, "_pnp_reproj_rms", return_value=(T_eye, 0.5)):
            return cs._ref_pair_poses(list(pl), list(pr), spec, objp,
                                      K, D, K, D, size_l, size_r)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        dl, dr = root / "l", root / "r"
        dl.mkdir()
        dr.mkdir()
        for d in (dl, dr):
            cv2.imwrite(str(d / "same.png"), np.full((480, 640), 128, np.uint8))
        Tl, Tr, info = _call([dl / "same.png"], [dr / "same.png"])
        out.append(("参考对按同名文件成对，返回两端位姿",
                    np.allclose(Tl, T_eye) and np.allclose(Tr, T_eye)
                    and info["n"] == 1 and "同名" in info["pairing"]))

        # 各只放一张、名字不同 → 自动配成一对（小白不该被文件名卡住）
        dl2, dr2 = root / "l2", root / "r2"
        dl2.mkdir()
        dr2.mkdir()
        cv2.imwrite(str(dl2 / "a.png"), np.full((480, 640), 128, np.uint8))
        cv2.imwrite(str(dr2 / "b.png"), np.full((480, 640), 128, np.uint8))
        _x, _y, info2 = _call([dl2 / "a.png"], [dr2 / "b.png"])
        out.append(("两边各 1 张但不同名 → 自动配成一对", info2["n"] == 1))

        # 尺寸与内参段不一致 → 明确报错（不能拿别档的图凑）
        dl3, dr3 = root / "l3", root / "r3"
        dl3.mkdir()
        dr3.mkdir()
        cv2.imwrite(str(dl3 / "bad.png"), np.full((360, 640), 128, np.uint8))
        cv2.imwrite(str(dr3 / "bad.png"), np.full((480, 640), 128, np.uint8))
        bad_msg = ""
        try:
            _call([dl3 / "bad.png"], [dr3 / "bad.png"])
        except ValueError as exc:
            bad_msg = str(exc)
        out.append(("参考对尺寸与内参段不一致 → 明确报错",
                    "尺寸" in bad_msg))

        # 两边各 2 张、全不同名 → 拒绝，并说清第 ③ 步怎么拍
        no_pair_msg = ""
        try:
            _call([dl2 / "a.png", dl3 / "bad.png"],
                  [dr2 / "b.png", dr3 / "bad.png"])
        except ValueError as exc:
            no_pair_msg = str(exc)
        out.append(("没有成对图 → 拒绝并提示第 ③ 步",
                    "第 ③ 步" in no_pair_msg))

        p, ok = check_ref_pair(root, spec)
        out.append(("自检：空目录判不能跑且点名第 ③ 步",
                    not ok and any("第 ③ 步" in x for x in p)))

    # ---- 2026-11-11 修「自检假绿」：参考段板动 / 单图几何质量 ----
    #
    # 真实事故：14:39 的 ③ 相机拍参考对时装置动过（标定实测参考对离散
    # 9.656°/85.49 mm），旧版自检只数"N 对、N 对检出整板"，判「全部通过」。
    # 下面用**同一份判据 + 真值可控的假角点**证明它不是空转：
    #   ① 各对位移 0 -> 不误报；② 位移 8 px -> 必须报「第 ③ 步板动了」；
    #   ③ 只有 1 对 -> 未评估、不误报；④ 单图几何判据的理想/抖动两端。
    import io
    from contextlib import redirect_stdout

    import apps.tools.check_chain_inputs as cci

    class _FakeObs:
        def __init__(self, corners):
            self.found = True
            self.corners = np.asarray(corners, np.float32).reshape(-1, 1, 2)
            self.area_ratio = 0.05
            self.focus = 5000.0
            self.note = ""

    grid = (np.stack(np.meshgrid(np.linspace(0, 7, 8), np.linspace(0, 4, 5)),
                     axis=-1).reshape(-1, 2) * 40.0 + 100.0)

    def _stable_corners(_p, _spec):
        return (480, 640), _FakeObs(grid), ""

    def _shifted_corners(p, _spec):
        bump = 8.0 if p.name.endswith("_2.png") else 0.0
        return (480, 640), _FakeObs(grid + bump), ""

    def _pair_dirs(root: Path, names: list[str]) -> None:
        for side in ("ref_pair_left", "ref_pair_right"):
            (root / side).mkdir()
            for n in names:
                cv2.imwrite(str(root / side / n), np.full((48, 64), 128, np.uint8))

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _pair_dirs(root, ["pair_1.png", "pair_2.png", "pair_3.png"])
        with patch.object(cci, "_corners", _stable_corners):
            with redirect_stdout(io.StringIO()):
                p_stable, ok_stable = check_ref_pair(root, spec)
        out.append(("自检：参考对各对之间板没动 -> 不误报「板动了」",
                    ok_stable and not any("板动了" in x for x in p_stable)))
        with patch.object(cci, "_corners", _shifted_corners):
            buf = io.StringIO()
            with redirect_stdout(buf):
                p_moved, ok_moved = check_ref_pair(root, spec)
        out.append(("自检：参考对各对位移 >5px -> 报「第 ③ 步板动了」且带数字",
                    (not ok_moved)
                    and any("板动了" in x and "第 ③ 步" in x for x in p_moved)
                    and "8.0 px" in buf.getvalue()))

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _pair_dirs(root, ["only.png"])
        with patch.object(cci, "_corners", _stable_corners):
            buf = io.StringIO()
            with redirect_stdout(buf):
                p_one, ok_one = check_ref_pair(root, spec)
        out.append(("自检：参考对只有 1 对 -> 提示「未评估」且不误报板动",
                    ok_one and "未评估" in buf.getvalue()
                    and not any("板动了" in x for x in p_one)))

    from apps.tools.check_chain_inputs import _grid_residual

    H = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    ideal = cv2.perspectiveTransform(
        spec.object_points()[:, :2].reshape(-1, 1, 2), H).reshape(-1, 2)
    r_ideal = _grid_residual(_FakeObs(ideal), spec)
    noisy = ideal + np.random.default_rng(0).normal(0.0, 1.0, ideal.shape)
    r_noisy = _grid_residual(_FakeObs(noisy), spec)
    out.append(("自检：单图几何判据理想网格≈0、1px 抖动≈1px（判据不是空转）",
                r_ideal < 0.01 and 0.7 < r_noisy < 1.5))

    cci_src2 = Path("apps/tools/check_chain_inputs.py").read_text(encoding="utf-8")
    out.append(("自检源码：参考对真的比较各对之间角点位移（reference_shift + REF_STILL_PX）",
                "reference_shift(" in cci_src2 and "REF_STILL_PX" in cci_src2
                and "参考段（第 ③ 步）板动了" in cci_src2))
    out.append(("自检源码：逐图角点质量判据（单应残差 + focus_gate）在位",
                "_grid_residual(" in cci_src2 and "focus_gate(" in cci_src2))
    return out


def _check_calib_session_isolation() -> list[tuple[str, bool]]:
    """17. 标定图按会话隔离（2026-11-09）。

    以前所有采集追加到 ``outputs/chain_frames/left`` 这类固定目录，
    ``_run_chain`` 全读 —— 真机实测 board_b 344 张里混了两批（11-04 的 304
    张 + 11-08 的 40 张），mono_calib_frames 60 张混了两天。用户明确要求
    「千万不要多次标定的图片混杂在一起」，所以一次采集 = 一个子目录，
    根层只放 ``_current.json`` 指针，**绝不回落读根层图片**。
    """
    import tempfile
    from pathlib import Path

    from apps import _actions

    out: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "chain_frames"
        d1 = _actions.new_chain_session(root=root, profile="装置A", mode="angled",
                                        board={"cols": "8", "rows": "5",
                                               "square": "25"})
        s1 = d1["left"].parent
        out.append(("一次采集 = 一个会话子目录，写 session.json 与 _current.json",
                    s1.parent == root and (s1 / "session.json").is_file()
                    and (root / "_current.json").is_file()))
        d2 = _actions.new_chain_session(root=root, profile="装置A", mode="angled")
        s2 = d2["left"].parent
        out.append(("反证：连续两次采集落在**不同**会话（不再追加同一目录）",
                    s1 != s2 and s1.is_dir() and s2.is_dir()))
        cur = _actions.current_chain_dirs(root=root)
        out.append(("当前会话指针指向最新一次，八段目录全在会话内",
                    cur is not None and cur["left"].parent == s2
                    and all(v.parent == s2 for v in cur.values())))

        # 根层故意放一张「上次标定」的图：标定的 argv 绝不能指向它
        legacy = root / "left" / "old.png"
        legacy.parent.mkdir(parents=True, exist_ok=True)
        legacy.write_bytes(b"old")
        cmd = _actions.calibrate_args("chain_dual", {}, "left", "right",
                                     "angled", dirs=cur)
        out.append(("反证：标定 argv 指向会话目录，不吃根层历史图",
                    cur is not None
                    and cmd[cmd.index("--left-dir") + 1] == str(cur["left"])
                    and all(str(legacy.parent) != x for x in cmd)
                    and "--ref-pair-left" in cmd))

        # 四步采集 + 自检 + 标定，六条 argv 必须全指向**同一个**会话
        kw = dict(cols="8", rows="5", square="25")
        chain_cmds = [
            _actions.capture_chain_frames(stage="left", target="999", dirs=cur, **kw),
            _actions.capture_chain_frames(stage="dual_b", target="999", dirs=cur, **kw),
            _actions.capture_chain_frames(stage="ref_pair", target="3", dirs=cur, **kw),
            _actions.capture_pair_frame(camera="left", dirs=cur, **kw),
            _actions.check_chain_inputs(mode="dual", dirs=cur, **kw),
            cmd,
        ]
        out.append(("四步采集 + 自检 + 标定，六条 argv 全指向同一会话",
                    all(str(s2) in " ".join(c) for c in chain_cmds)))

        # 2026-11-11 用户明确：④ 同框图**保持 5 张不动**（改成 3 张的是 ③ 参考对）。
        # 这里钉住 ④ 的 argv 与界面文案仍是 5 —— 谁再顺手改小，这条立刻红。
        pair_cmd = chain_cmds[3]
        pair_target = (pair_cmd[pair_cmd.index("--target") + 1]
                       if "--target" in pair_cmd else "")
        gui_src_now = Path("apps/_gui_stereo.py").read_text(encoding="utf-8")
        out.append(("④ 同框图仍是 5 张：argv --target 5，界面文案「拍 5 张」",
                    pair_target == "5" and "拍 5 张" in gui_src_now))

        (root / "_current.json").unlink()
        out.append(("反证：当前会话指针丢失 → None，绝不回落根层",
                    _actions.current_chain_dirs(root=root) is None))

        mroot = Path(td) / "mono_calib_frames"
        m1 = _actions.new_mono_session(root=mroot, camera="left")
        m2 = _actions.new_mono_session(root=mroot, camera="left")
        out.append(("单目：每次拍图单独会话，指针指向最新一次",
                    m1 != m2 and _actions.current_mono_dir(root=mroot) == m2
                    and (m2 / "session.json").is_file()))
        (mroot / "_current.json").unlink()
        out.append(("反证：单目指针丢失 → None",
                    _actions.current_mono_dir(root=mroot) is None))

    gui = Path("apps/_gui_stereo.py").read_text(encoding="utf-8")
    mono = Path("apps/_gui_mono.py").read_text(encoding="utf-8")
    out.append(("GUI 链式：采集建会话、自检与标定读当前会话，不再固定根目录",
                "new_chain_session(" in gui
                and "current_chain_dirs()" in gui
                and "d = chain_dirs()" not in gui
                and gui.count("dirs=d") >= 8))
    out.append(("GUI 单目：拍图建会话、标定读当前会话",
                "new_mono_session(" in mono
                and "current_mono_dir()" in mono
                and "frames_dir=CALIB_FRAME_DIR" not in mono))

    # ---- 17b. 只重拍某一步（2026-11-11）：全程临时目录，不碰真实 outputs ---- #
    # sessions_root 用 td2/chain_frames，于是 trash_root = td2/_trash 也在
    # 同一个 TemporaryDirectory 内，结束时一次清干净。
    import ast
    import json
    import shutil
    from apps import _cleanup

    def _count(d):
        return len(list(d.glob("*.png"))) if d.is_dir() else 0

    def _trash_list(troot):
        return sorted(p.name for p in troot.iterdir()) if troot.is_dir() else []

    with tempfile.TemporaryDirectory() as td2:
        root = Path(td2) / "chain_frames"
        cur = _actions.new_chain_session(root=root, mode="angled")
        for k, n in (("left", 4), ("board_b", 4), ("ref_pair_left", 3),
                     ("ref_pair_right", 3), ("pair", 5)):
            cur[k].mkdir(parents=True, exist_ok=True)
            for i in range(n):
                (cur[k] / f"{i}.png").write_bytes(b"x")
        ptr = root / "_current.json"
        before_ptr = json.loads(ptr.read_text(encoding="utf-8"))
        other_before = {k: _count(cur[k]) for k in ("left", "board_b", "pair")}

        # 统一入口：故意改坏实现让计划构造抛错时，判据**变红而不是整个脚本崩**，
        # 这样 4 条反向自证能逐条报出是哪几条红（而不是只看到一段 traceback）。
        def _reshoot(keys, **kw):
            try:
                return _actions.reshoot_chain_stage(cur, tuple(keys), **kw), True
            except Exception:                              # noqa: BLE001
                return None, False

        # --- 源码 / AST 判据（不需要窗口实例） ---
        gui_src = Path("apps/_gui_stereo.py").read_text(encoding="utf-8")
        head = gui_src.index("    def _reshoot_chain_stage(self) -> None:")
        tail = gui_src.find("\n    def _check_chain_inputs", head)
        reshoot_body = gui_src[head:tail if tail != -1 else len(gui_src)]
        start_calls = [ln for ln in reshoot_body.splitlines()
                       if "tasks.start(" in ln]
        # 判据 4
        out.append(("重拍回调源码里不含 new_chain_session(，且 tasks.start 不带 on_done",
                    "new_chain_session(" not in reshoot_body
                    and len(start_calls) == 1
                    and all("on_done" not in ln for ln in start_calls)))

        act_src = Path("apps/_actions.py").read_text(encoding="utf-8")
        ah = act_src.index("def reshoot_chain_stage(")
        at = act_src.find("\ndef ", ah + 1)
        act_body = act_src[ah:at if at != -1 else len(act_src)]
        out.append(("reshoot_chain_stage 源码确实带 allow_nested=True（去掉即计划构造抛错）",
                    "allow_nested=True" in act_body))

        tree = ast.parse(gui_src)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_chain_steps")
        # 按**内容特征**定位每一步的 steps，不按行号顺序猜（2026-11-19）：
        # `_chain_steps` 新增 device90 两步分支后多了第三个 `steps = [...]`，
        # 旧判据取"前两个"会把 device90 当成 dual、dual 当成 single，全线错位。
        # 分支条件也不好用（单板那条在 else 里、没有自己的 If）。三个分支的
        # 目录键互不相同，用它认领最稳：
        #   dual   → 含 ref_pair_left（③ 参考对）
        #   single → 含 ref_left 且不含 ref_pair_left（③ 单板参考段）
        #   device90 → 两步、无任何 ref_*（本组判据不关心它）
        all_step_lists = [
            n.value for n in ast.walk(fn)
            if (isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "steps"
                        for t in n.targets)
                and isinstance(n.value, ast.List))]

        def _list_keys(lst):
            keys = set()
            for el in lst.elts:
                if (isinstance(el, ast.Tuple) and len(el.elts) == 3
                        and isinstance(el.elts[1], ast.Tuple)):
                    keys |= {e.value for e in el.elts[1].elts
                             if isinstance(e, ast.Constant)}
            return keys

        dual_list = next((s for s in all_step_lists
                          if "ref_pair_left" in _list_keys(s)), None)
        single_list = next((s for s in all_step_lists
                            if "ref_left" in _list_keys(s)
                            and "ref_pair_left" not in _list_keys(s)), None)
        step_lists = [x for x in (dual_list, single_list) if x is not None]

        def _keys_of(lst):
            got = []
            for el in lst.elts:
                if (isinstance(el, ast.Tuple) and len(el.elts) == 3
                        and isinstance(el.elts[1], ast.Tuple)):
                    got.append(tuple(e.value for e in el.elts[1].elts
                                     if isinstance(e, ast.Constant)))
            return got

        dual_keys = _keys_of(dual_list) if dual_list is not None else []
        single_keys = _keys_of(single_list) if single_list is not None else []
        all_dirs = set(_actions.chain_dirs())
        flat_dual = {k for t in dual_keys for k in t}
        flat_single = {k for t in single_keys for k in t}
        # 判据 5
        out.append(("双板 4 步 5 目录 / 单板 3 步 4 目录，目录键集合与 chain_dirs() 一致",
                    len(dual_keys) == 4
                    and flat_dual == {"left", "board_b", "ref_pair_left",
                                      "ref_pair_right", "pair"}
                    and len(single_keys) == 3
                    and flat_single == {"left", "right", "ref_left", "ref_right"}
                    and flat_dual <= all_dirs and flat_single <= all_dirs))

        # 判据 1：重拍第 ③ 步后 _current.json 的 session/created_at 完全不变
        info3, ok3 = _reshoot(("ref_pair_left", "ref_pair_right"), label="重拍第3步")
        after_ptr = json.loads(ptr.read_text(encoding="utf-8"))
        out.append(("重拍第 ③ 步后 _current.json 的 session/created_at 完全不变",
                    ok3 and info3 is not None
                    and before_ptr == after_ptr
                    and before_ptr.get("session") == after_ptr.get("session")
                    and before_ptr.get("created_at") == after_ptr.get("created_at")))

        # 判据 2：只移走第 ③ 步，另外三段目录仍在且图片数不变
        out.append(("只移走第 ③ 步后，另外三段目录仍在且图片数不变",
                    ok3
                    and all(cur[k].is_dir() and _count(cur[k]) == other_before[k]
                            for k in ("left", "board_b", "pair"))
                    and all(cur[k].is_dir() and not any(cur[k].iterdir())
                            for k in ("ref_pair_left", "ref_pair_right"))))

        # 判据 3：manifest 只含该段相对路径，restore_batch 能原样恢复且文件数一致。
        # ⚠️ 生产硬要求「移走后立刻 mkdir 重建为空目录」与 restore_batch 的
        # 「目标已存在就抛 FileExistsError」天然冲突（calibrate_args 只在 pair/
        # 存在时才下传 --pair-frame，目录不能留空）。所以恢复测试里**先把重建
        # 的两个空目录删掉**再 restore —— 这是判据自己的处理，不是放宽实现；
        # 重建仍由判据 2 / 6 单独钉住。
        man = {"items": []}
        if ok3 and info3 is not None and info3["batch"] is not None:
            man = json.loads(
                (info3["batch"] / "manifest.json").read_text(encoding="utf-8"))
        man_paths = {str(it.get("path") or "") for it in man.get("items") or []}
        sess = cur["left"].parent.name
        rel_ok = (len(man_paths) == 2
                  and all(p.startswith(sess + "/")
                          and p.split("/")[-1] in ("ref_pair_left", "ref_pair_right")
                          for p in man_paths))
        for k in ("ref_pair_left", "ref_pair_right"):
            if cur[k].is_dir():
                shutil.rmtree(cur[k])
        restored = (_cleanup.restore_batch(root, info3["batch"])
                    if (ok3 and info3 is not None and man_paths) else 0)
        out.append(("回收站 manifest 只含该段相对路径，restore_batch 能原样恢复且文件数一致",
                    ok3 and rel_ok and restored == 2
                    and _count(cur["ref_pair_left"]) == 3
                    and _count(cur["ref_pair_right"]) == 3
                    and set(info3["moved"]) == man_paths))

        # 新增判据「③ 两目录必须成对」：用 GUI 源码里 ③ 的 keys 重拍，
        # 两个目录都必须被清空重建（只清一个会红）。
        pair_keys = next((t for t in dual_keys if "ref_pair_left" in t), ())
        _reshoot(pair_keys, label="成对")
        out.append(("③ 两目录必须成对：GUI 给的 keys 是两个，且两个目录都被清空重建",
                    len(pair_keys) == 2
                    and all(cur[k].is_dir() and not any(cur[k].iterdir())
                            for k in ("ref_pair_left", "ref_pair_right"))))

        # 判据 6：重拍第 ④ 步后 calibrate_args 仍带 --pair-frame（钉住
        # 「移走后必须 mkdir 重建」—— 否则 pair/ 不存在，argv 会缺参数）。
        ok6 = _reshoot(("pair",), label="重拍第4步")[1]
        cmd6 = _actions.calibrate_args("chain_dual", {}, "left", "right",
                                       "angled", dirs=cur)
        out.append(("重拍第 ④ 步后 calibrate_args(chain_dual, dirs=d) 仍带 --pair-frame",
                    ok6 and "--pair-frame" in cmd6 and cur["pair"].is_dir()
                    and not any(cur["pair"].iterdir())))

        # 判据 7：dry_run=True 一个字节都不写（会话目录与回收站都不变）
        ptr_bytes = ptr.read_bytes()
        snap = {k: (_count(cur[k]), cur[k].is_dir())
                for k in ("left", "board_b", "ref_pair_left",
                          "ref_pair_right", "pair")}
        trash_before = _trash_list(_cleanup.trash_root(root))
        dry, ok7 = _reshoot(("board_b",), label="dry", dry_run=True)
        out.append(("dry_run=True 不写盘（会话目录与回收站都不变）",
                    ok7 and dry is not None and dry["dry_run"] is True
                    and dry["batch"] is None
                    and ptr.read_bytes() == ptr_bytes
                    and {k: (_count(cur[k]), cur[k].is_dir())
                         for k in snap} == snap
                    and _trash_list(_cleanup.trash_root(root)) == trash_before))

        # 反向护栏：两层目录（<会话>/<段>）必须靠 allow_nested=True 才被接受。
        # 真实实现里去掉它，_reshoot 会捕获 ValueError 让上面运行时判据全红；
        # 这里把「不带 allow_nested 会抛错」也钉成一条正向判据。
        try:
            _cleanup.plan_cleanup(root, [str(cur["board_b"])], mode="session")
            nested_guard = False
        except ValueError:
            nested_guard = True
        out.append(("plan_cleanup 不带 allow_nested 时两层目录直接抛错（钉住 allow_nested=True）",
                    nested_guard))
    # ---- 17c. restore_chain_stage：专用恢复入口（2026-11-11）---------------- #
    # 数据管理的「回收站…」只看 data/_trash/、会把这批还原到错误位置；这里
    # 验证专用入口按 manifest.root 还原、预检/整批拒绝/越界都不留半恢复状态。
    # 用独立临时目录，避免与 17b 里被反复折腾的会话互相干扰。
    with tempfile.TemporaryDirectory() as td3:
        root3 = Path(td3) / "chain_frames"
        cur3 = _actions.new_chain_session(root=root3, mode="angled")

        def snapshot(d):
            return ({p.name: p.read_bytes() for p in d.iterdir()
                     if p.is_file()} if d.is_dir() else {})

        def safe_restore(b):
            try:
                return _actions.restore_chain_stage(b)
            except Exception:                              # noqa: BLE001
                return {"restored": -1, "skipped": [], "reason": "EXC",
                        "batch": b}

        # 造带内容的假 .png：文件名集合 + 内容字节都要能逐字比对
        for k in ("left", "board_b", "ref_pair_left", "ref_pair_right"):
            cur3[k].mkdir(parents=True, exist_ok=True)
            for i in range(3):
                (cur3[k] / f"{i}.png").write_bytes(bytes([65 + i, len(k), i * 7 + 1]))
        cur3["pair"].mkdir(parents=True, exist_ok=True)
        for i in range(5):
            (cur3["pair"] / f"{i}.png").write_bytes(b"pair")
        before3 = {k: snapshot(cur3[k]) for k in cur3}

        # 判据 1 + 2：reshoot → restore 后逐字相同、批次清掉、其余段一个字节没变
        info_r = _actions.reshoot_chain_stage(
            cur3, ("ref_pair_left", "ref_pair_right"), label="恢复判据")
        batch_r = info_r["batch"]
        res = safe_restore(batch_r)
        out.append(("restore_chain_stage 恢复后该步文件名与字节逐字相同、其余三段一个字节没变",
                    res["restored"] == 2 and res["reason"] == ""
                    and all(snapshot(cur3[k]) == before3[k] for k in cur3)))
        out.append(("恢复成功后 outputs/_trash 批次目录被清掉",
                    res["restored"] == 2 and not batch_r.exists()))

        # 判据 3：该步已有新图 → 整批拒绝，新图/其它段原样保留、批次仍在
        info_r2 = _actions.reshoot_chain_stage(
            cur3, ("ref_pair_left", "ref_pair_right"), label="拒绝判据")
        batch_r2 = info_r2["batch"]
        (cur3["ref_pair_left"] / "new.png").write_bytes(b"new-shot")
        before_reject = {k: snapshot(cur3[k]) for k in cur3}
        res2 = safe_restore(batch_r2)
        out.append(("该步已有新图时 restore_chain_stage 整批拒绝，新图与其它段原样保留、批次仍在",
                    res2["restored"] == 0 and bool(res2["reason"])
                    and any("ref_pair_left" in s for s in res2["skipped"])
                    and all(snapshot(cur3[k]) == before_reject[k] for k in cur3)
                    and batch_r2.exists()))

        # 判据 4：越界 manifest（../escape / 绝对路径）必须抛错且不移动任何文件。
        # 每个用例用**独立新批次**：否则故意改坏实现（不抛错）会删掉批次，
        # 下一次写 manifest 就 FileNotFoundError —— 脚本崩溃而不是判据变红。
        escaped = 0
        for bad_rel in ("../escape", str(Path(td3).anchor) + "escape"):
            info_e = _actions.reshoot_chain_stage(
                cur3, ("board_b",), label="越界判据")
            batch_e = info_e["batch"]
            man_e = json.loads(
                (batch_e / "manifest.json").read_text(encoding="utf-8"))
            man_e["items"] = [{"path": bad_rel, "bytes": 0, "files": 0}]
            (batch_e / "manifest.json").write_text(
                json.dumps(man_e, ensure_ascii=False), encoding="utf-8")
            before_e = {k: snapshot(cur3[k]) for k in cur3}
            try:
                _actions.restore_chain_stage(batch_e)
            except ValueError:
                escaped += 1
            if not all(snapshot(cur3[k]) == before_e[k] for k in cur3):
                escaped = -99          # 移动了文件 → 直接判红
        out.append(("restore_chain_stage 拒绝 ../escape 与绝对路径 manifest，且不移动任何文件",
                    escaped == 2))
        # 判据 5：dry_run 的 batch=None 传进恢复函数必须明确报错
        dry_info = _actions.reshoot_chain_stage(
            cur3, ("left",), label="dry", dry_run=True)
        try:
            _actions.restore_chain_stage(dry_info["batch"])
            dry_rejected = False
        except ValueError:
            dry_rejected = True
        out.append(("dry_run=True 的 batch=None 传进 restore_chain_stage 明确报错",
                    dry_info["batch"] is None and dry_rejected))
    return out



# ---- 18. 红色塑料球 color 三条路径（2026-11-13）---------------------------- #
# 真机会话 data/stereo/parallel/20261111-153857：球确定是红色塑料球，H 中位
# 176、约 85% 落在 H<=10 或 H>=170（跨 HSV 0/180 边界）。下面这个 64x64 BGR
# 补丁是从该会话 left.mkv 第 40 帧、以 seeds.json 的球心 (954.95,1034.58)
# 为中心裁出的**无损真机像素**（PNG + base85），球心落在 (32,32)、r≈24.57。
# 之所以内嵌而不是运行时读那 593MB 视频：/data/** 被 .gitignore 挡在版本库外，
# 判据必须自带真机像素才能在任何环境复算；本机若真有视频，末尾两条增强判据
# 还会只读抽帧跑一次完整链路当交叉证据。全程只读 data/，临时产物只写
# tempfile.mkdtemp() 并在 finally 删净。绝不改环形色相算法。
_REAL_SEED_XYRS = (954.9473684210526, 1034.5794392523364,
                   24.56502791358749, 40)
_REAL_ROI_B85 = (
    "iBL{Q4GJ0x0000DNk~Le0000$0000$2m$~A03{3Ni2wi=+DSw~RA>c-!BNZhZFU~?<2cqj&;NBl@"
    "8ng%9h1EC4uTtQh&z6S?%ARes-_xIVih&rC{}4xowiUbMU7ajc0?0&Llg=UYpBJ3D2bDK?(2W9wG"
    "OZQnRn6--RSSfzw>|nHojwm7}B)I_zp?@JdyzM!T|;tfG+?D5P<Pw7%v6^;)4N$0P*uk+Q!c#2^b"
    "u}0AT?F2oNSfe8Io-d%r{clnG);<Hh)gAO#LEzyQP-A_>5-4dR1AfPh~Bg8*Of^GE`)4S*y7l7QO"
    "{7$6{+009An2^ui}pWpi(8ZRUO&X^_%5I;3G{sE-$!N3565BvZSfboH_4TAvj!Jt9o1MwY_07%-#"
    "&yd86P+cHl0>lTw0*DU=%)k4$ep?^_5&+v{P{bed3t*ch{vHA#0SsV#AOMj7XuKFUK!f;0G-wbo@"
    "%NC#7q;;QNxTSSn*<0GAV8P^@xh?^cmLM^L=xr;h@J*O5+J^?O~5w3APE?MACdr!55xx&w($iG;)"
    "6W^0U98QpD_U9r#v7&@C5)lgh`nAK!Er_d@un1?f>&fWRL+6aMvvnAi$Y{Au=Qwu*W74B47Yx0HT"
    "7VJ>v#IQess}6i8sE3F|{({DBzTv_XI%O}x+)MUb`!HY{2sKoW*wgZLQ|{M*0#H%I~Uz@A7DCSwf"
    "S7{9<?k^TaJ2*NL87(t4pJ@(k<2apgU1!mv@KvE<C8YYGqVr+vZZ3Bc66bu41ZCC_A;-^d)KZpPK"
    "|M5Qn<_ryEm>PlxDFC)-+V;xy7XUw)GXt9neu41|Yycz(laNG%_&|cxeIJkjEJ7S&Y=EEu0w4fGB"
    ")&rcBtd{MKG=qcFZg?Z^S3!IPY@sux?#YQLIPn7JT&pa0Lg&qY1<MB(7+hBXD}dK_syRXDe%RhNl"
    "*+af<cBr8oW<|crjobFMI*<fnS*TLk3K|@OS^_{}z})2LSn=YK(0Wl!+k$55|N6lM-aZ_DoNe0E{"
    "vA40>Gco~goo@E0&(m{KlAL}skWj994%N^LU)(!|&%KuW|5%`XfPfMLP_2!i~b|NVF0Qb~s(dioF"
    "p*l75`2biESEts%CK{mR_(|rzSV57%1K0WtwH3rjto_AGs_k6(ExfjI9FLv&|B9o!@xU`Zfgdh#T"
    "5Jl6bMH2jg4+cOQANU{twg2@kdF^;y`z^G(9;vm0YR?Va!Co@vga<PsF?M@;Ts?hbX1ZBM+0Rq&_"
    "nfZhsp|XvdCol^g8&V@y<HSYMnta6z1B*mVz1b1y=|^Uh!KiFuw@7Y35Ya47%wEimp}4n{@Nq;>F"
    "xUZ7ryuPwZ4|VzSgI&9(zeB;cU&0@p(@dr^D?jSNG}Z?)&NKsYW%=`2E!TbMEu||L^bjsdL{`tq-"
    "A05D@|lfK;sfuu`e7zW#-LyRNsLkIf8bC<99+48fl<VB+t`f9_BJbxf!9xb~;iKeFRf;_=vzy+7^"
    "s^<#bgSl`=1Jr`=B)KB+a&(B?Fsxu;hIdwk&<ov-;`t$qw{JB0q>-}8@#vV@(3`r18`2zrg%Dwj9"
    "v0}envDfvu*4xT!<(07lE4>#o7y{Ywz?LW)1DhcK*+26?!VbpD+_Inl-<Bh^sEhn6;}>>*FZqvt^"
    "?0x^^~DMtxkU$D&u7?kPQ8Dw_s_WRc|PZP_w%Wlsh%1bVq{2}$^{9gjUIzE0CDY%o!8s7KV5lcUb"
    "(M@mHFw-E0uvz!%aZ6Fdon}_|N=Dzax;QD51hfC;;kNGY|Sx>Qms2^QL(OF4pGkz~fq<mii|@PPe"
    "P;`JDU4IXE}&<8#nGr=|^X<=T7a%5`ljUI6sCd#XJc2ncG0c0|VYw)fkWZ&$o+uFSodX>n)^7KZ>"
    "#nE&*D_&*0F$^)XsBaXnZb9#H;#5a!a&kxjvS$!|`75wzr-&=h6zR<Ael+VE_oO_;6pF;Ok&lq54"
    "Ue{x-%pDP2Yegg|lK>6WbYaR}Xb|gKZ(n^{pEj?Io$K1MmO~aCaL^QJ!u+TH^#2l~m{Jfk^9(ttB"
    "o}78FVF5;eI9dm-*Miv&lcWResB5B&#w}xkiqGg@$;VNecs>csZ-N011c00)}{ja5-W1Y`k=@&4Y"
    "4uhF1J$qaphxiE3UX6Z?P9vIK+TcLl_XjfAUZLHpx(Ct`O-Haw=9Z?d8rsSJ$4p>bdJ)^}ePSeMN"
    "j?>G}I#g?Fx$l32ECe9n1)ch7X6bME^-)iVHQtpS@5m6?0(Yh63n8Zh_Hok%ujh{(`ZzOB60zOLA"
    "dJF&zJ2Zu00g5UhfzYaShSFW8g_j}SA%+QvWd%1R>Pr^FS-Oop#3%%eNuetU9`xi?j#tC2KGTH5J"
    "+89^WU8kzMY<EO~4T}saVy(5W$i;}*E3buM!wluhxMIIuv2*Q}kByzk&`z$fXyJwb#IOAqo{_Y2T"
    "^X#CoRn7Tk@*lW_T^f2_Ia*)KIUAQHH|9XpD)}or%u)V<n%=@N}iPwfK1!%8NZMiOo<>x3RFbo-i"
    "o9Gp$U{{U}Zc$J+4omVsEY7*N$spCw6KtEX7U;2KbA=_E!TRND0&LpC6%Xt!w2LF8IPcra#eF-;b"
    "%?&pj1g>{?UHcRxSBzrWYH<yl0A0^TcRiy{cZfNdJ=5l9MvZArOSuC?~IXtCuXEhxlXU;ktOIASS"
    "7dvPbO#YZCJ?W;%Zguq|=*Z+zDej!+O!V{o8n|>3&%J>97rr+jVa~^ZDPtM_V^jWSx@8_I8M{vL)"
    "5WOQrNR$Ew2-_IPNE85jXoM7*xtBzQ5Mj}xQi?$C>#MI37c;mMD{(D7Hg@i}EA~=~zx=QLWf3nVW"
    "iOaj=wVlQJ_%oCd_`RDtIyk<hb!mAl%~?Hy3zMcyIU}AVdzOQ#0V)O6b5Y5xIaHb0!e^|q|A(!wi"
    "#lCG87T1%(T+`&1-9=BPL_k()HlA*H>?Gt#w`e<v;P4MDPXBIPDk{*36pTo(JaPd0?)-AFhW_PKB"
    "H9oK`(Gg&DXB$U~T(;0rS2C38ijBoaUV@sCML@)$O37!)Iuq+TK-BNZ7MW#z*4(B2a&sbYCu-dBG"
    "5glp&Z;5WbeLxCaz%ZQ#dMI@ZvyX*2irY_et_vN}==?V{~(yhLu3pPCT5EdaCGa@6DxpM8jueDZ$"
    "P8GzEY<G2?(^b>6_JToxUxXyENk1O5ue=_PbyH79#j4z$yK;Fge*3TfkfumrWal_+%*4#<euLkDP"
    "nbCG7tZ!sbJ87dxHPq<!vl|}IW0OQ0`?FP8IfyWdtZC+$RrsNYb|AfwB1!R^L)O0W~yO(EK+t;eq"
    "7#{@^W5B@r?E4dd5DnKSOtj-~P)#3<w5kdBPPQj-zK)z1eTXCrtL)zPBrVr;9F3#nkqMjTblEoDN"
    "FFA0m~xGFPq@kpX7r+Iv5)ytXn71NiZee>~lFpXtU}FpVK+?NVBS^_cy*^Lpy~tkvu7C(>t$KltN"
    "+k+47jfmS^i96*lO^wo8FuIUT2eXlvoCw$O{ZcT5`B1pjFZny*H2iXP;PzoxPN$8Oo5&3fM+)I%l"
    "K@dEr>YS>=SS0PCN^zS*VO@ND>iy}C>$CRRk3Ud6*Wx$7^1~nilgOD>_wodgn1$I>yL(S<_nwn|F"
    "Q4e6X#zRDe8LUD9s;6?0SthYL{gwYQ6VXa=>Z*)dtKM{xc1Ii-D9K2{W69vViX6_k!Cp3>+1EW+|"
    "Rsz!uY)Q`&#_wSAHlAV3IlgHoc|^hdbPtdrj@B?YgGc)apB@HH~q)7fMrv@xg>e1AfsLq(Fh9f&^"
    "iEY>Qg4R;-K^LkfVP1{mX~tSGH!SVH8)>bxrVGxz7*Kgso;i@*FY{U@Rc1IX#G=$xWQc1&IFHMQq"
    "#_cdotuRb}YY1%Qh(3~O+fCS;di;W>E1}R9OzMw>ZfJlKf33?C_ilBlDg8)ccF|nIrVw9~su}|#x"
    "*k?aJul>FjzxfybJTZVtru&Vv+{7f1GkYrg-gDN}?z5)%RP+fqJmF5aJb(d<41l%>d-NelnFJY>L"
    "J=C9lt8@LZhJ-&B`UxGK~j_zqg4z@%_zpvy0zY8*Y)-}_cIs2`D4E+3}67^zE}&D(ju3=`b0ff-`"
    "VGyTGOjfPK6tu@W>+!VA>)B2ts(VBvPax8VH6YV?dA!Nt8(Uct#RPLj%|*AqmQ67Sq%y3vrlF<}="
    "pWkN3Quk^KH2{d4q)c#$h{FH}m4T$t4-`>eX7@2hJ~?WuHYS{~W*5C$MR01*KY7C<T_PzFemda(x"
    "r$&3g^NHpM=G1%krj15dkh;<{I3k)-xaWikmJG1uVbL?j%zyHtwrqG0G2$eX(2_QTRv!~Yd?#jNe"
    "sWoR$r(4~^$fiwzuxP<S2T1_7z>q*0Bt?cq(&hjNg{X`WNT9n7fasaQ*e$|bH}-Hd0?cL{uAA|$t"
    "i0Z1zen=hKl+<+6P6+OQqhHOO{MFaxiFV|^}Xk;IeRKSF(V9ZcmRtOAvg#D0eXg{z!w8#NI?P|cz"
    "Q;psE`EW3%-N~Gak@@jV$lsrUH?zycxIh$#r7ix!)uC?H~KIuxZ1Vdtt&&JKPJir?&gyihAF3*3_"
    "Pp)9H~19K)bt&=x3LqznNB_#HqrKtKj<0|ZDBCP5-10gP?hgkcD9l5gf3vLia;RvzV3`__7oeRA>"
    "JfAP<u2R6o<UUbX|4tKh@YfoK1+4nVPO|3cU&S_zQF~>Abi~}MNAdR4Dkc3DYgbA54Wc(Bj5fEM?"
    "0>IcNhA<5h4)Fo57AvA6Zsk$Fb3J3d$F5lX!LR&z^tcB-oZfVJXoq{b*PL`c=Ip6GXH6}ioC=RT#"
    "DD_~K#m2<6e*ho0f7`iKzN}kLk0mD1W1qsAjvO;0TV+529tUWa-@zpid%V<Pwmf)XI>{3fAGiuQ}"
    "(#+X-zLoOw%KGOs%OkwdY*!J!efVpXCXUJj4)13_y+r3J^#`1Yl7jKtM1kBMccZhz}wGkOFMHuwe"
    "rbUIt=5nlVx_N_i`f@~QQ%e8#?G@dy9P4`rZjkEXIKrZBB(O)pnY?>U!y&&fG^Dm`fnmWw>{0E8`"
    "2hCl%%Xp1620)j!pInWRw0OA8F{EWddh9E(>8P2N|L$<OQN9$HTl~3iJ`_|$Qe)TWNz+gP-&bp%u"
    "9(2W2bS?MpT2p(@My=^|hap(5z%3L_8v)V?ina*Af=G~{Fn|CJWdgKKfDk}Z_+ku`LjV>rQ0G1hO"
    "HCE?XdUG{vtm7C-AaD{SN}p7FSgUY>co_%Ijug~6<up;PwzQv&h~^SV8Y>(c!rxM41qL6(GCJIDH"
    "0+;d_kH331Cc+5I}&W`~sL900dwwCRS0Fno-Kbd@^qBqjhWDN`CVv{-O-nfMfcaiY`y1E2?5nOs%"
    "Okz2+>}a;H0CdD0cGaMOexAPo`%1P~wzf~1HSgCJlK#sDBef&fVw^b0wl`C&+Cjbf;hig9Z_8MoH"
    "aelib}-~QTv#u?i)tv=herrD$E=*~XjP9N7Z=cH@R+34*~Pugh;rU4Ia5rk#S;LiXi1rRU;z`&+q"
    "<3%J4N`?RqWYUm9Lp8F(k*z4{j(lo8nVtK|bu;+=f8(#n_IRc_XU*BuYDafWM;}eehd$vGbMHBqY"
    "fmjtjuWC`4`CX@1S|~3qCf&52@@db2oOVj$7A9P$`Bw^1AziGh;|Hfq>eb!iFGTVu`1S+aWnbtzx"
    "khYfU!NC6ZOpLG@?7Y)JN0wkt=cMoYnW9x;#0(V0H_BAxv9v!18bmfFePW#)|-iAbcPNoA6IjVQ3"
    "&jWSBJ2%*JZwP$M1b&Q-DR*muN}aghA>-}<XEVAw|AIG>k0#yQR|)L|EWl9Rr3?%nt5-qYd9=|IV"
    "24`ITx0}c;70>1;0Gzd_`puiW<fITw?A%Pmo5E%vx6tm(gE|X0isd87WJN7$yE1Qeo|MmY;Gc;|}"
    ")z2R<bPU#5RqApb`mC7jJLg_~vd*4~o^T~wE_jF`EKdOPh>$H(Bq0+ov``d8kpy4>ZVv#-)IeaEA"
    "_?Fy@AY;gO%7Sg%B<Wc*5_fxDki`A_1}^Wn<iY-e-N%1XIs^(VwZdZm9FUL>XUU=cl4wyP(sr~4>"
    "7_7EIR-Z0TUnv5HKvagdk}e7yvc^nld>M6a(=ExV4_~c56WzvX!0dM4rg^Va0AGfBDz{Yq$f#fN0"
    "=dI2!9r)MU4&G$kL+9euK&aqiWfJvkL9xe3D~cfi7gEjS2z#4qq-SmOYM4<>1RFk!=tA<ZzwFAn#"
    "6yd7PZERC`vk8~6_qhhrdf9c=;EsO=w6naH-3O%Z3cXY?J+?W${v+t}s&so*klP>xe3S-$qhdW?l"
    "!b1;9@e9Csp%EUu5C#a6g26F{v|=bTK$$46r`}2zB1X}Qjy#H^sK{bYF8<=5{I58m0XRHsCZ;jN8"
    "BJ@7Zn-^2sHl5&<*9Xd)P!s1Cq&T`9(fX-vfK$124Ik&!54rSV@!Y~0Fs~}wK-&gO$`GP7F)~t=^"
    "&eGsv<gf#VW1CD8@+qrGMwQIK~$-vuD;AW4h6YK5)zJDY~Lh)>)_Xo@{H5{S)$VMm>QJSFppK1Oe"
    "lP01SZwz_APhBms~F1xA5QHCWOpEtzUY=cAEBm7+5%cWYOy*6Ii=_|3oj-_QdSwmi$7(-_N*KAeI"
    "}nCV+nF%_p)cbtiK%#q*egKkZ8DtrPR=pcj_;tPW&VTz(jfHp};hK4fLi~>j5q!pH$S+QHoP%9=^"
    "<<+q(^FYn8g5Umo|1CWp6BJr=G{qUsXbN3$5i{o*X!eQjsLmSo<W$V9Ihf{@=7cNUfeL{J@g0%|4"
    "H+T?Tl`?q&`?%1)r?V$Vn{HG+3T&fOggfa6}uuU@-{|kC4cbm|96C8dn`2OsVR-s!!Eip4cF97w5"
    "C)QZB@r~c4Z&Ufg5vjT62;ucW`6_e8C3;Fl3}55+VT_G}NG}QA}hjtuVt<am{*^mK9BovSJnED4T"
    "LFDENcl_#2oZ`~nx|$jzSUVU;|XhI{&k3ogw>b@Xs1yRwevc#3m0l|JDPl-wd{0K^9qFl44lNdyR"
    "uVXDocTH#oe+mTR0QOpxpYmGQmD@#$#Lp5U*%;dMf@!u217=9UI=vI%aRjnDg;p)E0!8E%xqaMv<"
    "H_tPsJO^`YD&~YMTy)7M4Dba%Wl~9s1Vw-b$87KnCl~T^Zsc|(q}Dpz6^o%!6r;5YDn)^#1jQfxs"
    "UN|EG5j(l=;m0}YO@<|xMvRBaC4enoZ(D*aNgC0DbzvVF%@&d%_(7I69$Q&S(*Hf435!YgJ(Fgu`"
    "aC3xv&<f)~dK#iy<pUt0N0)#wg4L#czM(M>J%>LwKeLV+@;Z+MI?5Zug;^E_&dBhkefHKgb!`F|B"
    "FU(R9qwH2T2fDZmg&5GE0n6oAHP&<sR2qm{@#xtt3t5Ybv4kI|+A5mq=Bayix@hBC-+{(~P8q#>I"
    "axZ&{_25gLJ(}OO!;i8K++=D@%d4A3rjy0p|m}Z@ruIFb>!|f@fjR8Q+${-;H5jL%W(veoS5^GH)"
    ")^f<9QaT^j#Rw`bjRIPZWu_&Q{Ken+5kc5A<OvU9*dD{im;h%u<2iQIO$_1jV2bAsH^&;YT3zg;s"
    "r&O!!UGRnGc}kOcm#_eMFtU6G8ZDD><CACEk`<np<;MH980mNuw>;htz<4s1OETwCkWvr5^li&00"
    "00<MNUMnLSTX"
)


def _real_ball_roi():
    """解码内嵌的 64x64 真机帧补丁（球心 (32,32)）。"""
    import base64
    import cv2
    import numpy as np
    raw = base64.b85decode(_REAL_ROI_B85)
    img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.shape == (64, 64, 3), "内嵌真机像素损坏"
    return img


def _check_color_red_ball_paths() -> list[tuple[str, bool]]:
    """§18 真机红塑料球：取色两段 / auto 不静默回退 / ball_kind 兼容 / 端到端。

    三条路径的复现结论：①「跨帧种子取色错位」在真实生产链路**未触发**（取色
    恰在种子帧 40、得两段红区间、score 0.889），按红线不改；② auto 锁不上静默
    回退白球、③ ball_kind="" 静默切 specular 均真触发，已修。本节钉住 ②③ 的
    修复与「① 不该被制造出来」这个事实。
    """
    import json
    import shutil
    import tempfile
    from pathlib import Path

    import cv2
    import numpy as np

    from sphere_tracker.config import ColorConfig, DetectorConfig
    from sphere_tracker.detectors import build_single_detector
    from sphere_tracker.detectors.color import hsv_ranges_from_pixels
    from apps import run_stereo_video as rsv

    out: list[tuple[str, bool]] = []
    roi = _real_ball_roi()
    cx = cy = 32
    sr = _REAL_SEED_XYRS[2]

    # ---- 1) 真机红球（跨 0/180）在种子帧取色 = 两段，且真检出 ---- #
    det = build_single_detector("color", DetectorConfig())
    det.seed(cx, cy, sr, frame_index=0)
    got = det.detect(roi)
    rng = det._seed_ranges
    seg2 = bool(rng) and len(rng) == 2
    cross = (seg2 and any(int(b[0]) >= 170 for a, b in rng)
             and any(int(a[0]) <= 10 for a, b in rng))
    real_hit = bool(got) and not (got[0].extra or {}).get("seed_anchor")
    out.append(("§18-1 真机红球(跨0/180)种子帧取色=两段且真检出 score>0.7",
                seg2 and cross and real_hit and got[0].score > 0.7))
    # 直接用真机圆环像素过一遍环形区间估计（不依赖检测器状态，证明算法未改）
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    yy, xx = np.ogrid[:64, :64]
    rr = np.hypot(xx - cx, yy - cy)
    ring = (rr >= 0.45 * sr) & (rr <= 0.80 * sr)
    pure = hsv_ranges_from_pixels(hsv[ring])
    out.append(("§18-1b 真机圆环像素直接估区间也是两段（环形色相算法未改）",
                len(pure) == 2))

    # ---- 2) auto 锁不上时不静默回退白球（②）---- #
    blank = np.full((64, 64, 3), 100, dtype=np.uint8)
    da = build_single_detector(
        "color", DetectorConfig(color=ColorConfig.preset("auto")))
    ga = da.detect(blank)
    ma = da.build_mask(blank)
    cover = cv2.countNonZero(ma) / float(ma.size)
    out.append(("§18-2 auto 锁不上不叫 fallback:config、明说未锁定、掩膜为空",
                da._auto_name != "fallback:config"
                and "未锁定" in da.last_error and cover == 0.0 and not ga))

    # ---- 3) ball_kind="" 不再静默切 specular + 兼容四条不变（③）---- #
    td = Path(tempfile.mkdtemp(prefix="selftest_color_paths_"))
    try:
        base = {"schema": "sphere-stereo-seeds", "version": 1,
                "session": "20261111-153857", "seed_frame": 40,
                "source": "gui_click", "ball_kind": "",
                "left": {"x": 954.95, "y": 1034.58, "radius": 24.57},
                "right": {"x": 866.53, "y": 1037.10, "radius": 29.28}}

        def _seeds(kind):
            d = dict(base)
            d["ball_kind"] = kind
            p = td / f"seeds_{kind or 'empty'}.json"
            p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
            return p

        empty_p = _seeds("")
        m0, n0 = rsv._pick_detector("color", False, empty_p, 0, False)
        visible = (m0 == "specular" and n0.startswith("[警告]")
                   and "ball_kind 为空" in n0 and "--method color" in n0)
        m_metal, n_metal = rsv._pick_detector("color", False, _seeds("metal"), 0, False)
        m_plastic, _ = rsv._pick_detector("color", False, _seeds("plastic"), 0, False)
        m_weird, n_weird = rsv._pick_detector("color", False, _seeds("weird"), 0, False)
        m_exp, n_exp = rsv._pick_detector("hough", True, empty_p, 0, False)
        compat = (m_metal == "specular" and n_metal.startswith("[种子]")
                  and m_plastic == "color"
                  and m_weird == "color" and n_weird.startswith("[警告]")
                  and m_exp == "hough" and n_exp == "")
        out.append(("§18-3 ball_kind='' 可见警告；metal/plastic/显式/未知四分支不变",
                    visible and compat))
        # 读 seeds 那处也不许再把空 ball_kind 静默省略（③ 的第二处可见性改动）
        src = (ROOT / "apps" / "run_stereo_video.py").read_text(encoding="utf-8")
        out.append(("§18-3b 读标准 seed 处对空 ball_kind 不再静默省略",
                    "（ball_kind 为空：未记录球种类）" in src))
    finally:
        shutil.rmtree(td, ignore_errors=True)

    # ---- 4) 端到端：真机帧序列跑完整链路（贴帧号/score）---- #
    def _shift(img, dx, dy):
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                              borderMode=cv2.BORDER_REPLICATE)

    seq = [_shift(roi, d, d // 2) for d in (0, 3, -4, 6, -2)]
    de = build_single_detector("color", DetectorConfig())
    de.seed(cx, cy, sr, frame_index=0)
    hits = []
    for i, fr in enumerate(seq):
        g = de.detect(fr)
        if g and not (g[0].extra or {}).get("seed_anchor"):
            hits.append((i, round(g[0].score, 3)))
    out.append((f"§18-4 端到端真机帧序列 {len(hits)}/5 帧真检出 {hits}",
                len(hits) >= 4))

    # ---- 5/6) 增强：本机有真机视频时，只读抽帧跑完整链路 ---- #
    video = ROOT / "data" / "stereo" / "parallel" / "20261111-153857" / "left.mkv"
    if video.exists():
        cap = cv2.VideoCapture(str(video))
        de2 = build_single_detector("color", DetectorConfig())
        de2.seed(*_REAL_SEED_XYRS[:3], frame_index=_REAL_SEED_XYRS[3])
        real_hits = []
        if cap.isOpened():
            for i in range(50):        # 覆盖种子帧 40 与其后 10 帧
                ok2, fr = cap.read()
                if not ok2:
                    break
                g = de2.detect(fr)
                if g and not (g[0].extra or {}).get("seed_anchor"):
                    real_hits.append((i, round(g[0].score, 3)))
        cap.release()
        out.append((f"§18-5 真机视频 0~49 帧完整链路 {len(real_hits)} 帧真检出 "
                    f"{real_hits[:6]}...",
                    len(real_hits) >= 8 and any(i >= 40 for i, _ in real_hits)))
        # 只读抽 0/40/151/301 到 tempfile：种子帧 40 取色两段、背景帧 0 单段
        # -> 佐证「① 取色帧错位才落背景」在真实链路里没被触发。
        tmp = Path(tempfile.mkdtemp(prefix="selftest_real_frames_"))
        try:
            cap = cv2.VideoCapture(str(video))
            for i in (0, 40, 151, 301):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok3, fr = cap.read()
                if ok3:
                    cv2.imwrite(str(tmp / f"left_{i:03d}.png"), fr)
            cap.release()

            def _seg_of(path):
                img = cv2.imread(str(path))
                if img is None:
                    return -1
                d = build_single_detector("color", DetectorConfig())
                d.seed(*_REAL_SEED_XYRS[:3], frame_index=0)
                d._init_seed_ranges(img)
                return 0 if d._seed_ranges is None else len(d._seed_ranges)

            s40 = _seg_of(tmp / "left_040.png")
            s0 = _seg_of(tmp / "left_000.png")
            out.append((f"§18-6 真机抽帧：种子帧40取色{s40}段、背景帧0取色{s0}段"
                        "（①未触发 + 背景帧已被 §21 护栏拒绝）",
                        s40 == 2 and s0 == 0))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return out


# ---- 21. 种子取色质量护栏（2026-11-15）-------------------------------------- #
def _syn_scene(kind: str | None, bg=(110, 110, 110), r: int = 26):
    """合成帧：均匀背景（带 2 灰阶噪声）+ 可选圆斑 + 中心高光。

    ``kind=None`` 就是"灰底、球不在画面"，用来造"点选落在背景"的情形。
    返回 ``(frame, (cx, cy, r))``。尺寸 640x480 是为了让默认尺寸门
    （`min_radius=18` / `max_radius=100`）覆盖 r=26。
    """
    import cv2
    import numpy as np

    w, h = 640, 480
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = bg
    rng = np.random.default_rng(7)
    img = np.clip(img.astype(np.int16)
                  + rng.normal(0, 2.0, img.shape).astype(np.int16),
                  0, 255).astype(np.uint8)
    cx, cy = w // 2, h // 2
    if kind is not None:
        color = {"white": (235, 235, 235), "orange": (40, 130, 235),
                 "green": (60, 160, 60), "blue": (200, 90, 40)}[kind]
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (int(cx - 0.25 * r), int(cy - 0.3 * r)), max(2, r // 5),
                   tuple(min(255, v + 40) for v in color), -1, cv2.LINE_AA)
    return img, (float(cx), float(cy), float(r))


def _seed_at(det, frame, frame_index: int):
    """把 `frame` 当成第 `frame_index` 帧喂一次（种子取色正好落在这一帧）。"""
    det._frame_idx = frame_index - 1
    return det.detect(frame)


def _check_seed_color_guard() -> list[tuple[str, bool]]:
    """§21 `_init_seed_ranges` 取色护栏：背景帧必须拒，白球/窄单段不许误伤。

    缺陷（实测）：以前只要圆环取到 ≥8 个像素就 `_seed_ranges = ranges`，于是把
    球心喂给**球不在画面**的帧时，背景色被当成球色 —— 同一会话
    `data/stereo/parallel/20261111-153857` 的帧 0/151/301 取到单段
    `H20~110 S≥2~4`、占整帧 65%~75%，随后 n=1、score=1.000 的**种子锚点**冒充
    检出。生产链路当前取色恰在种子帧 40（两段红、S≥41、占整帧 1.2%）所以没踩到；
    一旦帧号错位、或换一段开头球不在画面的录像，就会静默把背景当球色。

    护栏是**组合条件**（见 `color.py` 顶部 `_SEED_*` 常量的实测依据），本节钉两侧：
    真机红球必须接受；背景/无彩宽区间必须拒绝且说得清；白球（低 S 高 V）与
    橙/绿窄单段必须不被误伤；拒绝后 `detect()` 照常跑、`debug_snapshot()` 可诊断。
    """
    from sphere_tracker.config import DetectorConfig
    from sphere_tracker.detectors import build_single_detector

    out: list[tuple[str, bool]] = []
    video = ROOT / "data" / "stereo" / "parallel" / "20261111-153857" / "left.mkv"

    def _new():
        return build_single_detector("color", DetectorConfig())

    # ---- 1) 真机种子帧（内嵌 64x64 真机像素，来自 left.mkv 第 40 帧）---- #
    roi = _real_ball_roi()
    d1 = _new()
    d1.seed(32, 32, _REAL_SEED_XYRS[2], frame_index=0)
    got1 = d1.detect(roi)
    r1 = d1._seed_ranges
    out.append(("§21-1 真机种子帧（第40帧像素）取色=两段且被接受、真检出 score>0.7",
                bool(r1) and len(r1) == 2
                and bool(got1) and not (got1[0].extra or {}).get("seed_anchor")
                and got1[0].score > 0.7))

    # ---- 2) 真机背景帧（球不在画面）必须拒绝 ---- #
    if video.exists():
        import cv2

        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, bg0 = cap.read()
        cap.release()
        if ok:
            d2 = _new()
            d2.seed(*_REAL_SEED_XYRS[:3], frame_index=0)
            got2 = _seed_at(d2, bg0, 0)
            snap = d2.debug_snapshot()
            note = d2.last_error
            out.append(("§21-2 真机背景帧（第0帧，球不在）被拒绝：_seed_ranges=None、"
                        "last_error 说人话、_diag 有原因",
                        d2._seed_ranges is None
                        and snap.get("seed_ranges_rejected") == 1
                        and snap.get("seed_ranges_reject_code") == "background"
                        and ("背景" in note and "重新点选" in note)
                        and bool(got2)))     # 拒绝不等于崩：锚点照常返回

    # ---- 3) 合成"灰底无球"也必须拒绝（任何环境都能跑的那一侧）---- #
    floor, seed_f = _syn_scene(None)
    d3 = _new()
    d3.seed(*seed_f, frame_index=0)
    d3.detect(floor)
    snap3 = d3.debug_snapshot()
    out.append(("§21-3 合成灰底无球（点选落在背景）：拒绝 + 邻环覆盖率入 _diag",
                d3._seed_ranges is None
                and snap3.get("seed_ranges_rejected") == 1
                and float(snap3.get("seed_annulus_cover") or 0.0) > 0.5
                and "重新点选" in d3.last_error))

    # ---- 4) 白球不许误伤（低 S / 高 V / 单段窄）---- #
    white_ok = True
    white_notes = []
    for bgname, bg in (("灰底", (110, 110, 110)), ("亮底", (200, 200, 200))):
        img, seed = _syn_scene("white", bg=bg)
        dw = _new()
        dw.seed(*seed, frame_index=0)
        gw = _seed_at(dw, img, 0)
        hit = (dw._seed_ranges is not None and bool(gw)
               and not (gw[0].extra or {}).get("seed_anchor"))
        white_ok = white_ok and hit
        white_notes.append(f"{bgname}{'过' if hit else '误伤'}")
    out.append((f"§21-4 白球（低 S 高 V）不被误伤 {white_notes}", white_ok))

    # ---- 5) 窄单段不许误伤（橙 / 绿 / 蓝）---- #
    narrow_ok = True
    narrow_notes = []
    for kind in ("orange", "green", "blue"):
        img, seed = _syn_scene(kind)
        dn = _new()
        dn.seed(*seed, frame_index=0)
        gn = _seed_at(dn, img, 0)
        seg = 0 if dn._seed_ranges is None else len(dn._seed_ranges)
        hit = (seg == 1 and bool(gn) and not (gn[0].extra or {}).get("seed_anchor"))
        narrow_ok = narrow_ok and hit
        narrow_notes.append(f"{kind}={seg}段")
    out.append((f"§21-5 窄单段彩色球不被误伤 {narrow_notes}", narrow_ok))

    # ---- 6) 拒绝后不崩、且只试一次（不逐帧重取色）---- #
    d6 = _new()
    d6.seed(*seed_f, frame_index=0)
    no_raise = True
    try:
        _seed_at(d6, floor, 0)
        for _ in range(3):
            d6.detect(floor)
    except Exception as exc:                      # noqa: BLE001 - 自检就是要抓"崩"
        no_raise = False
        print(f"    （拒绝后 detect 抛异常：{exc!r}）")
    out.append(("§21-6 拒绝后 detect() 连跑 4 帧不抛异常、_seed_ranges 仍为 None、"
                "且不逐帧重试（_seed_rejected 一次即止）",
                no_raise and d6._seed_ranges is None and d6._seed_rejected is True))

    # ---- 7) 拒绝原因可通过 debug_snapshot() 诊断 ---- #
    snap7 = d6.debug_snapshot()
    reason = str(snap7.get("seed_ranges_reject_reason") or "")
    out.append(("§21-7 debug_snapshot() 里有 seed_ranges_rejected / reject_code / "
                "reject_reason 三个可排查字段",
                snap7.get("seed_ranges_rejected") == 1
                and snap7.get("seed_ranges_reject_code") in ("background", "overwide")
                and ("背景" in reason or "过宽" in reason)
                and snap7.get("error") == d6.last_error))
    return out



# ---------------------------------------------------------------------------
# 20. 原始左右录像文件名（唯一产地）
# ---------------------------------------------------------------------------
# 这三个文件以前各写一份 `("left.mkv", ...)` 元组：查看器按左/右分开取、数据管理
# 按钮按并集判断"有没有原始视频"、导出按并集收文件。口径已经漂过一次（按钮说有、
# 查看器打开却不在），所以第 20 节把这条事实钉在 `sphere_tracker/video_files.py`：
# 字面量只许出现在那里，调用点只许 import。
RAW_VIDEO_CALL_SITES = (
    ROOT / "apps" / "view_3d.py",
    ROOT / "apps" / "data_manager.py",
    ROOT / "apps" / "_data_export.py",
)


def _raw_video_name_literals(path: Path, names: set[str]) -> list[tuple[int, list[str]]]:
    """找"手写原始录像名清单"：列表/元组/集合字面量里出现原始录像文件名。

    只看**字面量集合**。写在注释、docstring、help 文本里的
    ``left/right.mkv``（比如"open raw videos (left/right.mkv)"）不是清单，
    不该被算成违例 —— 那会逼人删掉有用的说明文案。
    """
    hits: list[tuple[int, list[str]]] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            continue
        values = [e.value for e in node.elts
                  if isinstance(e, ast.Constant) and isinstance(e.value, str)]
        matched = [v for v in values if v in names]
        if matched:
            hits.append((node.lineno, matched))
    return hits


def _module_imports(path: Path) -> list[str]:
    """模块 import 到的模块名（``import x`` / ``from x import y`` 都算，含函数内的）。"""
    got: list[str] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            got.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            got.append(node.module)
    return got


def _check_raw_video_names() -> list[tuple[str, bool]]:
    """20：文件名唯一产地 + mkv 优先 + 只扫根目录 + 单路放行 + 与导出条目一致。"""
    import shutil
    import tempfile

    from apps._data_export import _cat_files
    from sphere_tracker.video_files import (
        ALL_RAW_VIDEO_NAMES, LEFT_VIDEO_NAMES, RAW_VIDEO_EXTENSIONS,
        RIGHT_VIDEO_NAMES, has_any_raw_video, pick_left_video, pick_right_video,
    )

    out: list[tuple[str, bool]] = []
    names = set(ALL_RAW_VIDEO_NAMES)

    out.append((f"常量：LEFT/RIGHT 各 3 个（mkv/mp4/avi）、无重复（{LEFT_VIDEO_NAMES} / "
                f"{RIGHT_VIDEO_NAMES}）",
                len(LEFT_VIDEO_NAMES) == 3 and len(RIGHT_VIDEO_NAMES) == 3
                and len(set(LEFT_VIDEO_NAMES)) == 3
                and len(set(RIGHT_VIDEO_NAMES)) == 3))
    out.append((f"常量：ALL 是左右并集（不是另抄一份），合计 {len(ALL_RAW_VIDEO_NAMES)} 个",
                names == set(LEFT_VIDEO_NAMES) | set(RIGHT_VIDEO_NAMES)
                and len(ALL_RAW_VIDEO_NAMES) == 6))
    out.append(("常量：每个名字的后缀都在 RAW_VIDEO_EXTENSIONS 里",
                all(Path(n).suffix in RAW_VIDEO_EXTENSIONS
                    for n in ALL_RAW_VIDEO_NAMES)))

    bad: list[str] = []
    for path in RAW_VIDEO_CALL_SITES:
        for lineno, matched in _raw_video_name_literals(path, names):
            bad.append(f"{path.relative_to(ROOT).as_posix()}:{lineno} 手写 {matched}")
    if bad:
        print("    手写原始录像名（应 import sphere_tracker.video_files）：")
        for line in bad:
            print(f"      · {line}")
    out.append((f"三处调用点（{len(RAW_VIDEO_CALL_SITES)} 个）里没有手写文件名清单",
                not bad))

    imports = _module_imports(ROOT / "sphere_tracker" / "video_files.py")
    heavy = sorted({m.split(".")[0] for m in imports}
                   & {"cv2", "numpy", "scipy", "stereo", "calib"})
    out.append((f"纯 stdlib：video_files.py 不 import 重依赖（{', '.join(imports)}）",
                not heavy))

    tmp_dirs: list[Path] = []

    def mk(*filenames: str) -> Path:
        d = Path(tempfile.mkdtemp(prefix="selftest_raw_video_"))
        tmp_dirs.append(d)
        for name in filenames:
            p = d / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"")
        return d

    try:
        both = mk(*ALL_RAW_VIDEO_NAMES)
        out.append(("mkv 优先：左路同时有 left.mkv / left.mp4 -> 取 left.mkv",
                    pick_left_video(both) == both / "left.mkv"))
        out.append(("mkv 优先：右路同时有 right.mkv / right.mp4 -> 取 right.mkv",
                    pick_right_video(both) == both / "right.mkv"))

        mp4_only = mk("left.mp4")
        out.append(("只有 left.mp4：取 left.mp4（mkv 缺失不是错误）",
                    pick_left_video(mp4_only) == mp4_only / "left.mp4"))

        right_only = mk("right.mp4")
        out.append(("单路放行：只有 right.mp4 时左路为 None、右路正常",
                    pick_left_video(right_only) is None
                    and pick_right_video(right_only) == right_only / "right.mp4"))
        out.append(("has_any_raw_video：只有 right.mp4 -> True（少一路也放行）",
                    has_any_raw_video(right_only) is True))

        empty = mk()
        out.append(("has_any_raw_video：六个都没有 -> False",
                    has_any_raw_video(empty) is False))
        out.append(("单路放行：空目录 pick_* 都返回 None（不抛异常）",
                    pick_left_video(empty) is None and pick_right_video(empty) is None))

        nested = mk("raw/left.mkv", "raw/right.mkv")
        out.append(("只扫会话根目录：raw/left.mkv 不算原始素材（不递归）",
                    pick_left_video(nested) is None
                    and pick_right_video(nested) is None
                    and has_any_raw_video(nested) is False))

        dir_like = mk()
        (dir_like / "left.mkv").mkdir()
        out.append(("是文件才算：目录叫 left.mkv 不骗过 has_any_raw_video",
                    pick_left_video(dir_like) is None
                    and has_any_raw_video(dir_like) is False))

        # 导出侧同一条清单：写出的条目集合与顺序都必须来自 ALL_RAW_VIDEO_NAMES
        exportable = mk(*ALL_RAW_VIDEO_NAMES)
        arcs = [arc for _p, arc in _cat_files(exportable, "raw")]
        expected = [f"videos/raw/{n}" for n in ALL_RAW_VIDEO_NAMES]
        out.append((f"_cat_files('raw') 的条目与常量一致（{len(arcs)} 个，顺序同常量）",
                    arcs == expected
                    and {Path(a).name for a in arcs} <= names))
    finally:
        for d in tmp_dirs:
            shutil.rmtree(d, ignore_errors=True)

    # ---- avi 多格式支持（用户新需求：处理外部 avi 输入）---- #
    avi_only = mk("left.avi")
    out.append(("判据1 只有 left.avi：pick_left_video 返回它",
                pick_left_video(avi_only) == avi_only / "left.avi"))
    both_mkv_avi = mk("left.mkv", "left.avi")
    out.append(("判据2 left.mkv + left.avi 同时在 -> 取 mkv（优先级不回退）",
                pick_left_video(both_mkv_avi) == both_mkv_avi / "left.mkv"))
    avi_half = mk("right.avi")
    out.append(("判据3 avi-only 会话（只有 right.avi）-> has_any_raw_video True",
                has_any_raw_video(avi_half) is True))
    # 判据4：avi-only 半成品目录能被 discover_session_dirs 枚举到。
    from sphere_tracker.discovery import discover_session_dirs
    disc_root = Path(tempfile.mkdtemp(prefix="selftest_avi_discover_"))
    tmp_dirs.append(disc_root)
    avi_sess = disc_root / "20261119-120000"
    avi_sess.mkdir()
    (avi_sess / "left.avi").write_bytes(b"")
    out.append(("判据4 avi-only 半成品目录能被 discover_session_dirs 枚举到",
                avi_sess in discover_session_dirs(disc_root)))

    # 判据5：fov_dialog 源码 AST —— 不再用 left.suffix 推右路，改用左右各自 pick_*。
    fd_tree = ast.parse((ROOT / "apps" / "fov_dialog.py").read_text(encoding="utf-8"))
    left_suffix = [n for n in ast.walk(fd_tree)
                   if isinstance(n, ast.Attribute) and n.attr == "suffix"
                   and isinstance(n.value, ast.Name) and n.value.id == "left"]
    fd_calls = {n.func.id for n in ast.walk(fd_tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    out.append(("判据5 fov_dialog AST：无 left.suffix 推路、左右各自 pick_*、无 _VIDEO_NAMES",
                not left_suffix
                and {"pick_left_video", "pick_right_video"} <= fd_calls
                and not any(isinstance(n, ast.Name) and n.id == "_VIDEO_NAMES"
                            for n in ast.walk(fd_tree))))

    # 判据6：record_stereo 源码 AST —— --container choices 仅 mkv/avi（无 mp4）、
    # 左右路径动态拼装、无 left.mkv 字面量。
    rs_tree = ast.parse((ROOT / "apps" / "record_stereo.py").read_text(encoding="utf-8"))
    container_choices: tuple[str, ...] | None = None
    for node in ast.walk(rs_tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--container"):
            for kw in node.keywords:
                if kw.arg == "choices" and isinstance(kw.value, (ast.Tuple, ast.List)):
                    container_choices = tuple(
                        e.value for e in kw.value.elts if isinstance(e, ast.Constant))
            break

    def _dynamic_path(tree: ast.AST, prefix: str) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.JoinedStr):
                literal = "".join(v.value for v in node.values
                                  if isinstance(v, ast.Constant)
                                  and isinstance(v.value, str))
                if (literal.startswith(prefix)
                        and any(isinstance(v, ast.FormattedValue) for v in node.values)):
                    return True
        return False

    left_mkv_literals = [n for n in ast.walk(rs_tree)
                         if isinstance(n, ast.Constant) and isinstance(n.value, str)
                         and "left.mkv" in n.value]
    out.append(("判据6 record_stereo AST：--container 仅 mkv/avi（无 mp4）、"
                f"左右路径动态拼装、无 left.mkv 字面量（choices={container_choices}）",
                container_choices == ("mkv", "avi")
                and _dynamic_path(rs_tree, "left.")
                and _dynamic_path(rs_tree, "right.")
                and not left_mkv_literals))
    for d in tmp_dirs:
        shutil.rmtree(d, ignore_errors=True)
    return out


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
        ("9. 框内细化窗口（YOLO 框 → 圆）", _check_refine_in_box_window),
        ("10. 可复现性（固定 RNG）与回捕默认值", _check_deterministic_rng),
        ("11. 检测器行为（点选取色 / 帧间一致性 / auto / 融合 / 半径锚）",
         _check_detector_behaviour),
        ("12. 锚点单一出口（AST 扫构造点 + note 语义）", _check_anchor_single_source),
        ("13. 尺寸门自动阶梯 + 装置档案门控（行为判据 + 反证）",
         _check_size_ladder_and_gui_state),
        ("14. GUI 尺寸门开关 / 主链路显式报错 / 单目设备身份",
         _check_gui_wiring_and_mono_identity),
        ("15. 姿态区域归属（九格全算数 + 反证）", _check_pose_region_mapping),
        ("16. 双板链式参考对（方案 A：护栏 / argv / GUI 四步 / 行为）",
         _check_dual_board_ref_pair),
        ("17. 标定图按会话隔离（不混批 / 不回落根层）",
         _check_calib_session_isolation),
        ("18. 红塑料球 color 三条路径（真机像素 / auto 未锁定 / ball_kind 兼容）",
         _check_color_red_ball_paths),
        ("20. 原始左右录像文件名（唯一产地 / mkv 优先 / 只扫根目录 / 单路放行）",
         _check_raw_video_names),
        ("21. 种子取色质量护栏（背景帧必拒 / 白球窄单段不误伤 / 可诊断）",
         _check_seed_color_guard),
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
