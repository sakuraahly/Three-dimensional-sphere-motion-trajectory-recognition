"""GUI 文案提取器：**后续新增内容自动生成对应多语言版本**的那一半。

## 它解决什么

界面上加一句新提示，如果靠人维护两份语言文件，结果一定是英文那份漏掉，
而且没人发现。这里把这件事变成机械动作：

    python apps/tools/i18n_extract.py --check     # 有新文案没登记 → 退出码 1
    python apps/tools/i18n_extract.py --write     # 自动补齐所有语言的资源文件
    python apps/tools/i18n_extract.py --stats     # 覆盖率 / 待译条数

`--write` 会：

1. AST 扫描白名单里的 GUI 模块，**只取用户看得见的字符串**
   （控件 text/label/title、消息框、状态栏、帮助内容、预设名），
   不碰注释、docstring、日志格式串、路径、代码标识符；
2. 新文案只在 `i18n_seed.py` 有**整句精确译文**时填英文，否则留空；
   **不再按词拼接**（`Record与识别` 这种混排禁止进入资源表）；
3. **保留已有译文**：人工填过的条目不会被覆盖；
4. 为 `i18n.LANGS` 里除中文外的每种语言各生成一张表 ——
   以后加第三种语言，只要往 `LANGS` 加一项，再跑一次 `--write` 就有对应资源。

`--check` 是硬门禁：源码里的每条中文必须存在英文表、值非空且不含中文；
新增文案只登记不翻译，CI 会红，直到补完整句子。

产物 `apps/i18n_data.py` 是普通 `.py` 文件（不是 JSON）：这个项目用的是
白名单式 `.gitignore`，`.py` 自动放行、自动进纯净包，不用再改两份清单。

用法::

    python apps/tools/i18n_extract.py --check
    python apps/tools/i18n_extract.py --write
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import re
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "sphere_tracker").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.i18n import LANGS                                   # noqa: E402
from apps.i18n_seed import NO_AUTOFILL, SEED                 # noqa: E402

DATA_MODULE = ROOT / "apps" / "i18n_data.py"

#: 只扫这些文件：GUI 模块 + 纯内容模块。工具脚本 / 算法库 / 测试不在内。
SCAN_DIRS = ("apps",)
# 白名单，不是扫描全部 `apps/*.py`：那是为了排除 `run_video.py` / `calibrate_stereo.py`
# 这类"命令行 help 文本"，它们不该进界面词典。
#
# ⚠ 白名单的代价是**加一个界面文件就可能漏扫**（漏了不会报错，只是英文界面
# 里那一块照旧中文）。所以 `tests/selftest_launcher_params.py::_check_i18n`
# 有一条判据钉住：`apps/_gui_*.py` 与 `apps/*_dialog.py` 必须都在这里。
SCAN_NAMES = {
    "gui_main.py", "cli_main.py", "_gui_app.py", "_gui_widgets.py",
    "_gui_mono.py", "_gui_stereo.py", "_gui_bench.py", "_gui_console.py",
    "_gui_geometry.py", "help_content.py", "help_window.py",
    "data_manager.py", "view_3d.py", "junk_dialog.py", "cleanup_dialog.py",
    "profile_dialog.py", "fov_dialog.py", "seed_picker.py", "tools_dialog.py",
    "cameras_dialog.py",
}
#: 内容型模块：整个文件的字符串都是给人看的（说明正文、分区标题）。
CONTENT_ONLY = {"help_content.py"}
#: 这些常量里存的是界面上要显示的标签，不是代码内部标识。
DATA_CONSTS = {"VIEW_PRESETS", "SECTIONS", "STATE_LEGEND", "SOURCE_CHOICES",
               "SOURCE_OFFLINE", "SOURCE_LIVE", "PRESET_NONE", "HELP_SECTIONS",
               "VIDEO_SOURCE_LABELS",
               # 相机预览失败的"照着做"结论 + 全文说明（2026-11-24）：它们是
               # 模块级常量、不在任何控件调用点里，旧提取器整个漏扫 —— 英文
               # 界面下相机打不开时满屏中文。显示点在 cameras_dialog 包 t()。
               "REASON_INVALIDATED", "REASON_CANNOT_OPEN", "REASON_NO_FRAME",
               "_NO_DSHOW_HELP", "_OPEN_FIX_STEPS", "CAPTURE_UNMEASURED_TEXT",
               # 相机状态条的 tf() 模板常量（同理，赋给常量不在调用点字面量里）。
               "TPL_HOGS_FULL", "TPL_HOG_NOTE", "NOTE_TWO_FAILED",
               "TPL_FULL_TAIL", "FULL_TAIL_FALLBACK", "TPL_STATUS_ERROR",
               "TPL_WAIT_SECS", "WATCHDOG_SLOW", "TPL_BAD_FRAME",
               "TPL_FRAME_OK", "TPL_PUMP_ERROR", "PREVIEWING",
               "TPL_CODE_NEWER", "TPL_CONFLICT_CHECK", "TPL_AUTO_REFRESH",
               "TPL_OPENED_PARALLEL", "TPL_VIRTUAL_PREVIEW",
               "TPL_ALREADY_PREVIEW", "TPL_AVOIDED_HINT", "TPL_OPENING",
               "TPL_NOT_SAVED", "TPL_WRITTEN", "TPL_LIMIT_NOT_SAVED",
               "TPL_LIMIT_WRITTEN", "TPL_VIRTUAL_ROLE",
               # Treeview 单元格状态标签（values= 由 i18n._apply_tree 逐格翻）
               "ROLE_IGNORED", "ROLE_VIRTUAL", "ROLE_BUILTIN", "ROLE_NONE",
               "NO_NAME",
               # 双目窗第①②④步说明 + 动态状态行模板（2026-11-24 截图）
               "PREMISE_PARALLEL", "PREMISE_ANGLED",
               "HINT_METHOD_PARALLEL", "HINT_METHOD_ANGLED", "STEP4_HINT",
               "ADV_CAPTURE_TITLE", "ADV_CAPTURE_SUMMARY",
               "TPL_RIG_PROFILE", "TPL_RIG_MODE", "TPL_RIG_BASE",
               "TPL_RIG_ANGLE", "TPL_CALIB_EXISTING", "CALIB_NONE",
               "TPL_CALIB_LINE", "TPL_CAM_LINE", "CAM_READ_FAILED",
               "TPL_RECORD_BLOCKED",
               # 折叠区半径三句（for note in (...) 元组字面量，回溯漏扫）
               "RADIUS_NOTE_PIXEL", "RADIUS_NOTE_ANCHOR",
               "RADIUS_NOTE_LADDER", "RADIUS_ADV_TITLE",
               "RADIUS_ADV_SUMMARY", "TITLE_STEREO_PARALLEL",
               "TITLE_STEREO_ANGLED",
               # 2026-11-24 全面核对下拉/弹窗时补：
               # 这些是界面词表，字符串挂在模块常量上、values= 只看到变量名。
               # 不登记它们，`--check` 会假报一致，英文下拉仍显示中文。
               "BALL_METAL", "BALL_PLASTIC", "BALL_CUSTOM", "BALL_KINDS",
               "BALL_HINTS", "RADIUS_NOTES", "_NO_FRAMES",
               "SOURCE_OFFLINE", "SOURCE_REALTIME", "SOURCE_CHOICES",
               "DETECTOR_HINT", "CAMERA_HINT_BASE",
               "METHODS", "MODE_LABELS", "PRESETS", "STATUS_OPTIONS",
               "VIDEO_SOURCE_LABELS", "RANGES", "STEP_SPECS"}
#: 控件 / 消息框调用中表示"显示文本"的关键字参数。
TEXT_KWARGS = {"text", "label", "title", "message", "detail", "hint",
               "placeholder", "tooltip", "desc", "description", "header",
               "footer", "empty", "status", "value_text",
               # 下拉选项是用户看得见的文字；值本身可能是数据（设备名/路径），
               # 但含中文的选项必须先登记，否则切英文仍显示中文。
               "values"}
#: 位置参数里是显示文本的调用：函数名 -> 要取的参数下标。
TEXT_POSITIONAL = {
    "set_status": (0,), "showinfo": (0, 1), "showerror": (0, 1),
    "showwarning": (0, 1), "askyesno": (0, 1), "askokcancel": (0, 1),
    "card": (1, 2), "primary": (1,), "secondary": (1,),
    "accent": (1,), "chip": (1,), "section": (1,),
    # 2026-11-24（B-187 的一半）：hint(inline_note/set_status 的第二参也是
    # 显示文案。漏了它们，"检查中…"/"正在读取设备列表…" 这类状态标签
    # 永远进不了词典，英文界面照样中文。
    "hint": (1,), "inline_note": (1,), "set_status": (0, 1),
    "_say": (0,),  # 相机管理窗口的状态条工厂（同 set_status）
    "heading": (2,), "add_command": (), "title": (0,),
    # 自定义开场参数：App.open_window(key, title, factory) / 无 text= 的
    # simpledialog 前两个位置参数是弹窗标题与提示。
    "open_window": (1,),
    "askstring": (0, 1), "askinteger": (0, 1), "askfloat": (0, 1),
    "askopenfilename": (), "asksaveasfilename": (), "askdirectory": (),
    # 自绘控件工厂：这些调用不一定有 text= 关键字，但第一/第二个位置参数
    # 就是标题或说明；漏掉它们时新按钮在 --check 里看不见。
    "CollapsibleSection": (1, 2), "_hint_pack": (1,),
    "_Button": (1,), "_Step": (1, 2), "button": (1,),
    "Label": (0,), "Button": (0,), "Checkbutton": (0,),
    "Radiobutton": (0,), "LabelFrame": (0,),
    # 翻译函数本身的首参就是界面文案（含 tf 的占位符模板）。不登记这两条，
    # 状态骨架用 tf() 改写后提取器看不见，--check 会假报"一致"。
    "t": (0,), "tf": (0,),
    # notes.append("...") / infos.append("...") 存的是状态条上给人看的说明，
    # 不是代码内部标识。漏了它，状态条在英文模式下就留一句中文（2026-11-23）。
    "append": (0,),
}
_CN = re.compile(r"[\u4e00-\u9fff]")
#: 明显不是界面文字：格式串、路径、样式名、颜色、URL、纯标点片段。
_SKIP_PREFIX = ("%", "{", "http", "config/", "apps/", "docs/", "data/",
                "outputs/", "logs/", "assets/", "tests/", "#", "/", ".",
                "Hint.T", "Warn.T", "Ok.T", "TLabel", "python ")


def _docstring_ids(tree: ast.AST) -> set[int]:
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if (isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                out.add(id(first.value))
    return out


def _clean(s: str) -> bool:
    if not s or not _CN.search(s):
        return False
    # `#` 在 _SKIP_PREFIX 里是为了排除颜色码（#b91c1c）。但 "#{index} 出图了"
    # / "#N 打不开" 这类是**真界面文案**（相机卡片状态句），含中文就该收。
    # 所以含中文的字符串不受 `#` 前缀排除影响（2026-11-24）。
    prefixes = tuple(p for p in _SKIP_PREFIX if p != "#") if _CN.search(s) else _SKIP_PREFIX
    if s.startswith(prefixes) or s in NO_AUTOFILL:
        return False
    if len(s) > 600:                       # 超长的是文档段落，不是控件文案
        return False
    return True


def _const(node, docs: set[int], out: set[str]) -> None:
    if isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
        for sub in ast.iter_child_nodes(node):
            _const(sub, docs, out)
        return
    if (isinstance(node, ast.Constant) and isinstance(node.value, str)
            and id(node) not in docs and _clean(node.value)):
        out.add(node.value)


def _text_variables(tree: ast.AST) -> set[str]:
    """找出"被当作显示文本传给 UI 工厂"的局部变量名。

    为什么需要这一趟（2026-11-23 实测漏翻的根因）：
    门厅三张卡的说明不是直接写在 `card(..., desc)` 里，而是先存进局部元组
    `specs = (("mono", 标题, 说明, opener), ...)`，循环里再用变量传入。
    旧提取器只看调用点的字面量参数，变量被 `_const()` 直接跳过 —— 于是
    这三句从未进词典，`--check` 却报"与源码一致"，英文界面照旧中文。

    这是一次**点到为止的数据流追踪**：只回溯"变量名 → 它的赋值表达式"，
    不做跨函数/跨模块分析，避免把算法里的中文字符串误收进界面词典。
    """
    used: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (func.attr if isinstance(func, ast.Attribute)
                else func.id if isinstance(func, ast.Name) else "")
        args = []
        for i in TEXT_POSITIONAL.get(name, ()):
            if i < len(node.args):
                args.append(node.args[i])
        args += [kw.value for kw in node.keywords if kw.arg in TEXT_KWARGS]
        for arg in args:
            # 解开 + / f-string 里的变量名；常量在调用点已被 _const 收走
            for sub in ast.walk(arg):
                if isinstance(sub, ast.Name):
                    used.add(sub.id)

    # 第二层：循环解包。门厅写的是 `for key, title, desc, opener in specs:`
    # 再 `card(parent, title, desc)` —— 调用点只见到 title/desc，真正的中文
    # 在 specs 的赋值里。把"被解包出来的文本变量 ← 迭代源变量"接上。
    for node in ast.walk(tree):
        if not isinstance(node, ast.For) or not isinstance(node.iter, ast.Name):
            continue
        loop_vars = {t.id for t in ast.walk(node.target) if isinstance(t, ast.Name)}
        if loop_vars & used:
            used.add(node.iter.id)
    return used


def extract_strings(path: Path) -> set[str]:
    """扫一个模块，返回用户可见的中文字符串集合（供自测直接调用）。"""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return set()
    docs = _docstring_ids(tree)
    out: set[str] = set()

    if path.name in CONTENT_ONLY:
        for node in ast.walk(tree):
            _const(node, docs, out)
        return out

    text_vars = _text_variables(tree)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = (func.attr if isinstance(func, ast.Attribute)
                    else func.id if isinstance(func, ast.Name) else "")
            for kw in node.keywords:
                if kw.arg in TEXT_KWARGS:
                    _const(kw.value, docs, out)
            for i in TEXT_POSITIONAL.get(name, ()):
                if i < len(node.args):
                    _const(node.args[i], docs, out)
        # 数据常量（预设名、说明分区、状态图例）里的显示文字
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            named = {t.id for t in targets if isinstance(t, ast.Name)}
            # `*_note` = 先拼好、后面用 f-string 塞进状态条的一句话
            # （retry_note / metal_note 等）。这类文案以前**整个漏扫**：
            # 字符串挂在变量上、不在 set_status 的参数里，于是 `--check` 报
            # "与源码一致"、英文界面却照旧显示中文。
            #
            # **只收整句常量，不收 f-string**：`f"…{x}…"` 在 AST 里是碎片，
            # 逐块登记只会得到「  （你的偏好 」这种半句话的 key，英文更乱。
            # 要真正覆盖 f-string 得改成"占位符键 + 运行时 format"，那是另一件事，
            # 已登记为待办（见 docs/待办清单.md B-185）。
            note_assign = any(t.endswith("_note") for t in named)
            if named & DATA_CONSTS or note_assign:
                value = node.value
                if value is not None:
                    has_joined = any(isinstance(sub, ast.JoinedStr)
                                     for sub in ast.walk(value))
                    if not (note_assign and has_joined):
                        for sub in ast.walk(value):
                            _const(sub, docs, out)
        # 变量回溯：只回溯**数据结构字面量**赋值（元组/列表/字典）。
        # 门厅 `specs = ((标题, 说明, opener), ...)` 就是这种：循环解包后
        # 以变量传给 card()，字面量藏在赋值里。
        # 绝不回溯 `text = ...` / `head = f"..."` 这类普通变量 —— 那些是运行时
        # 拼出来的状态串，扫进来会得到几十条半句噪声 key（2026-11-23 实测
        # 一次多扫 96 条）。f-string 同样跳过：碎片不能当整句 key。
        if (isinstance(node, ast.Assign)
                and isinstance(node.value, (ast.Tuple, ast.List, ast.Dict))):
            targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if targets & text_vars:
                has_joined = any(isinstance(sub, ast.JoinedStr)
                                 for sub in ast.walk(node.value))
                if not has_joined:
                    for sub in ast.walk(node.value):
                        _const(sub, docs, out)
    return out


def scan_all() -> set[str]:
    found: set[str] = set()
    for d in SCAN_DIRS:
        base = ROOT / d
        if not base.is_dir():
            continue
        for p in sorted(base.glob("*.py")):
            if p.name in SCAN_NAMES and p.name not in ("i18n.py", "i18n_data.py",
                                                       "i18n_seed.py"):
                found |= extract_strings(p)
    return found


def auto_translate(s: str) -> str:
    """只接受种子里的**整句精确译文**；否则留空，等人工补。

    不再做子串/逐词组合：那会生成 `Record与识别`、`Calibration堆栈`
    这种比中文更难读的混排。`--check` 会让空值/含中文的译文直接失败。
    """
    value = SEED.get(s, "")
    return value if value and not _CN.search(value) else ""


def _load_existing() -> dict[str, dict[str, str]]:
    tables: dict[str, dict[str, str]] = {}
    if not DATA_MODULE.is_file():
        return tables
    try:
        spec = importlib.util.spec_from_file_location("_i18n_data_old", DATA_MODULE)
        mod = importlib.util.module_from_spec(spec)       # type: ignore[arg-type]
        assert spec.loader is not None
        spec.loader.exec_module(mod)
    except Exception:                                    # noqa: BLE001
        return tables
    for lang in LANGS:
        if lang == "zh-CN":
            continue
        code = lang.replace("-", "_").upper()
        val = getattr(mod, code, None)
        if isinstance(val, dict):
            tables[code] = {str(k): str(v) for k, v in val.items()}
    return tables


def build_tables(keys: set[str]) -> dict[str, dict[str, str]]:
    old = _load_existing()
    out: dict[str, dict[str, str]] = {}
    for lang in LANGS:
        if lang == "zh-CN":
            continue
        code = lang.replace("-", "_").upper()
        prev = old.get(code, {})
        table = {k: prev[k] for k in sorted(keys) if prev.get(k)}
        for k in sorted(keys):
            if k not in table:
                table[k] = auto_translate(k) if code == "EN_US" else ""
        out[code] = table
    return out


def render(tables: dict[str, dict[str, str]], keys: set[str]) -> str:
    lines = [
        '"""GUI 多语言资源 —— 由 apps/tools/i18n_extract.py 自动生成，请勿手改。',
        "",
        "改文案请改界面源码，然后跑：python apps/tools/i18n_extract.py --write",
        "人工润色译文请改 apps/i18n_seed.py（重新生成会保留本文件里已填的非空值）。",
        "值为空 = 待译：运行时回退中文；--check 会阻止缺译文/含中文的条目。",
        f"扫描到 {len(keys)} 条界面文案。",
        '"""',
        "from __future__ import annotations",
        "",
    ]
    for code, table in tables.items():
        done = sum(1 for v in table.values() if v)
        lines.append(f"# {code}: {done}/{len(table)} 条已有译文")
        lines.append(f"{code}: dict[str, str] = {{")
        for k in sorted(table):
            lines.append(f"    {k!r}: {table[k]!r},")
        lines.append("}")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="扫描 GUI 文案并生成双语资源")
    ap.add_argument("--check", action="store_true", help="有新文案未登记就退出 1")
    ap.add_argument("--write", action="store_true", help="生成 apps/i18n_data.py")
    ap.add_argument("--stats", action="store_true", help="打印覆盖率")
    args = ap.parse_args(argv)

    keys = scan_all()
    tables = build_tables(keys)
    existing = _load_existing()
    known = existing.get("EN_US", {})
    fresh = sorted(keys - set(known))
    missing = sorted(
        k for k in keys
        if not known.get(k) or _CN.search(str(known.get(k)))
    )

    if args.check:
        if fresh:
            print(f"[不达标] {len(fresh)} 条新界面文案还没登记，跑一次 --write：")
            for s in fresh[:10]:
                print("  · " + s[:70])
        if missing:
            print(f"[不达标] {len(missing)} 条界面文案缺完整英文（空值或含中文）：")
            for s in missing[:10]:
                print("  · " + s[:70])
        if fresh or missing:
            return 1
        print(f"多语言资源与源码一致（{len(keys)} 条文案，全部有完整英文）")
        return 0

    if args.stats or not args.write:
        for code, table in tables.items():
            done = sum(1 for v in table.values() if v)
            pct = 100.0 * done / len(table) if table else 100.0
            print(f"{code}: {done}/{len(table)} 条有译文（{pct:.1f}%），"
                  f"待译 {len(table) - done} 条")
        print(f"源码扫描到 {len(keys)} 条界面文案；新增 {len(fresh)} 条")
        if not args.write:
            return 0

    DATA_MODULE.write_text(render(tables, keys), encoding="utf-8")
    print(f"已生成 {DATA_MODULE.relative_to(ROOT).as_posix()}："
          f"{len(keys)} 条文案 / {len(tables)} 种语言")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
