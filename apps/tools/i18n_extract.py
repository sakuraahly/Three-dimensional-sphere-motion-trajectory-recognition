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
2. 对每条新文案，按 `apps/i18n_seed.py` 的领域词典自动生成英文：
   整句精确命中用高质量译文，否则用词典做短语组合；
   组合后仍有中文的记为待译（空值），运行时回退中文并继续部分英文化；
3. **保留已有译文**：人工填过的条目不会被覆盖；
4. 为 `i18n.LANGS` 里除中文外的每种语言各生成一张表 ——
   以后加第三种语言，只要往 `LANGS` 加一项，再跑一次 `--write` 就有对应资源。

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
               "VIDEO_SOURCE_LABELS"}
#: 控件 / 消息框调用中表示"显示文本"的关键字参数。
TEXT_KWARGS = {"text", "label", "title", "message", "detail", "hint",
               "placeholder", "tooltip", "desc", "description", "header",
               "footer", "empty", "status", "value_text"}
#: 位置参数里是显示文本的调用：函数名 -> 要取的参数下标。
TEXT_POSITIONAL = {
    "set_status": (0,), "showinfo": (0, 1), "showerror": (0, 1),
    "showwarning": (0, 1), "askyesno": (0, 1), "askokcancel": (0, 1),
    "card": (1, 2), "primary": (1,), "secondary": (1,),
    "accent": (1,), "chip": (1,), "section": (1,),
    "heading": (2,), "add_command": (), "title": (0,),
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
    if s.startswith(_SKIP_PREFIX) or s in NO_AUTOFILL:
        return False
    if len(s) > 600:                       # 超长的是文档段落，不是控件文案
        return False
    return True


def _const(node, docs: set[int], out: set[str]) -> None:
    if (isinstance(node, ast.Constant) and isinstance(node.value, str)
            and id(node) not in docs and _clean(node.value)):
        out.add(node.value)


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


def _phrase_replace(s: str) -> str:
    keys = sorted((k for k in SEED if len(k) >= 2 and _CN.search(k)),
                  key=len, reverse=True)
    if not keys:
        return s
    pat = re.compile("|".join(re.escape(k) for k in keys))
    return pat.sub(lambda m: SEED.get(m.group(0), m.group(0)), s)


def auto_translate(s: str) -> str:
    """整句精确命中 → 高质量译文；否则用词典自动组词生成对应版本。

    组合结果可能中英混排（长段落里的专有表述词典没覆盖），但它仍是**自动
    生成的对应语言版本**：切到英文后整屏会明显英文化，而不是只有按钮变。
    人工润色请改 `i18n_seed.py` 或本文件里已填的值 —— `--write` 会保留它们。
    词典一个词都没命中的才留空（运行时回退中文）。
    """
    if s in SEED:
        return SEED[s]
    filled = _phrase_replace(s)
    return filled if filled != s else ""


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
        "值为空 = 待译：运行时回退中文，并用种子词典做短语级自动英文化。",
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
    known = set(existing.get("EN_US", {}))
    fresh = sorted(keys - known)

    if args.check:
        if fresh:
            print(f"[不达标] {len(fresh)} 条新界面文案还没登记，跑一次 --write：")
            for s in fresh[:10]:
                print("  · " + s[:70])
            return 1
        print(f"多语言资源与源码一致（{len(keys)} 条文案）")
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
