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
