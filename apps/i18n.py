"""GUI 多语言运行时（只管用户看得见的文字，不碰代码注释 / 日志格式 / 标识符）。

## 设计：中文原文即 key，零改造现有界面代码

项目里有约 2400 条界面文案散在十几个模块里，全部改成 ``t("some.key")``
既要动上千处源码、又会让"新增一句文案"变成"先想个 key、再改两个语言文件"
的双份劳动 —— 结果一定是有人只改中文、英文悄悄漏掉。

这里换一种做法：

- **源码里的中文原样保留**，它就是 key；
- `t(text)` 查 `apps/i18n_data.py` 生成的双语表，命中就换，没命中回退原文；
- 窗口建好后 `apply(widget)` 递归遍历控件树，把 Label / Button / 标题 /
  菜单 / Treeview 表头与条目的显示文本过一遍 `t()`；
- 新建窗口由 `install()` 挂的钩子在 `after_idle` 自动补一次，
  **所以以后新增界面、新增文案，不用改本文件任何一行**；
- `apps/tools/i18n_extract.py` 扫描源码自动把新文案登记进两份语言资源，
  并用 `i18n_seed.py` 的领域词典自动给出英文（详见该工具文档）。

## 明确的安全边界

只翻译**纯展示**文本。参与逻辑判断的值（Combobox 选项、`StringVar` 里
要被 `.get()` 比较的字符串、文件路径、命令参数）一律不碰 —— 否则
`if var.get() == "离线（当前会话）"` 会在切换语言后静默失效。
消息框的标题/正文属于纯展示，已统一挂钩。

## 语言切换

`set_language()` 保存偏好到 `config/language.txt`（本机状态，不入库），
并立即刷新所有已打开窗口。`LANGS` 加一项即可扩展第三种语言：
提取器会为每种语言生成对应资源文件，运行时按 `i18n_data.<CODE>` 取表。
"""

from __future__ import annotations

import re
import tkinter as tk
from functools import lru_cache
from pathlib import Path
from tkinter import messagebox, ttk

from apps.i18n_seed import SEED

try:
    from apps import i18n_data
except ImportError:                                   # 首次生成前也能 import
    i18n_data = None                                  # type: ignore[assignment]

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "sphere_tracker").is_dir())
LANG_FILE = ROOT / "config" / "language.txt"
LANGS = ("zh-CN", "en-US")
LANG_LABELS = {"zh-CN": "中文", "en-US": "English"}
_CN = re.compile(r"[\u4e00-\u9fff]")

_current = "zh-CN"
_installed = False


# --------------------------------------------------------------------------- #
# 资源表
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=4)
def table(lang: str) -> dict[str, str]:
    """取某个语言的映射表（带缓存：控件树遍历会调用几百次）。"""
    if lang == "zh-CN":
        return {}
    name = lang.replace("-", "_").upper()
    data = getattr(i18n_data, name, {}) if i18n_data is not None else {}
    merged = dict(SEED)
    merged.update({k: v for k, v in data.items() if v})
    return merged


@lru_cache(maxsize=1)
def _phrase_pattern() -> re.Pattern[str] | None:
    """按 key 长度降序编译一张大正则：长词先替换，避免 "左目" 被 "左" 拆掉。"""
    keys = sorted((k for k in table(_current) if len(k) >= 2 and _CN.search(k)),
                  key=len, reverse=True)
    return re.compile("|".join(re.escape(k) for k in keys)) if keys else None


def _autofill(text: str) -> str:
    """整句没命中时用领域词典自动组词，让动态拼接的状态文本也能英文化。"""
    pat = _phrase_pattern()
    # 长段落（帮助正文、模式卡说明）也要英文化，否则切到英文后整屏还是中文。
    # 一次正则扫描是线性的，2000 字以内不会成为 UI 卡顿点。
    if pat is None or len(text) > 2000:
        return text
    tbl = table(_current)
    return pat.sub(lambda m: tbl.get(m.group(0), m.group(0)), text)


def t(text):
    """翻译一段**界面文本**；非字符串、中文模式、未登记的文本原样返回。"""
    if not isinstance(text, str) or not text or _current == "zh-CN":
        return text
    hit = table(_current).get(text)
    if hit:
        return hit
    if _CN.search(text):
        filled = _autofill(text)
        if filled != text:
            return filled
    return text


def t_all(items):
    return [t(x) if isinstance(x, str) else x for x in items]


# --------------------------------------------------------------------------- #
# 语言状态
# --------------------------------------------------------------------------- #
def language() -> str:
    return _current


def set_language(lang: str) -> None:
    """切换语言：持久化 + 立刻刷新所有已打开窗口。"""
    global _current
    if lang not in LANGS:
        raise ValueError(f"不支持的语言 {lang!r}；可选 {LANGS}")
    if lang == _current:
        return
    _current = lang
    table.cache_clear()
    _phrase_pattern.cache_clear()
    try:
        LANG_FILE.parent.mkdir(parents=True, exist_ok=True)
        LANG_FILE.write_text(lang, encoding="utf-8")
    except OSError:
        pass
    refresh_all()


def label(lang: str | None = None) -> str:
    return LANG_LABELS.get(lang or _current, lang or _current)


def _load_saved() -> None:
    global _current
    try:
        saved = LANG_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return
    if saved in LANGS:
        _current = saved


# --------------------------------------------------------------------------- #
# 控件树刷新
# --------------------------------------------------------------------------- #
def _orig_map(widget) -> dict[str, str]:
    """记住每个展示属性的**中文原文**和**上次写入的显示值**。

    两个值都要记：只记原文的话，切回中文那一刻用"新语言的 t(原文)"去比对
    当前显示的英文，会误判成"程序动态改过文本"，于是把英文当成新原文存下来
    —— 界面就再也切不回中文了（实测踩过）。
    """
    d = getattr(widget, "_i18n_orig", None)
    if d is None:
        d = {}
        try:
            setattr(widget, "_i18n_orig", d)
        except (AttributeError, TypeError):
            return {}
    return d


def _original_for(d: dict[str, str], key: str, cur: str) -> str:
    """算出该拿哪段原文去翻译；能识别程序自己改过的动态文本。"""
    shown = d.get(key + ":shown")
    prev = d.get(key)
    if prev is None or (shown is not None and cur != shown and cur != prev):
        d[key] = cur
    return d[key]


def _mark_shown(d: dict[str, str], key: str, value: str) -> None:
    d[key + ":shown"] = value


def _set_translated(d: dict[str, str], key: str, cur: str, setter) -> None:
    """翻译 + 写入 + 记账，三件事一起做，避免各处各写一半。"""
    new = t(_original_for(d, key, cur))
    if new == cur:
        _mark_shown(d, key, cur)
        return
    try:
        setter(new)
        _mark_shown(d, key, new)
    except (tk.TclError, TypeError, AttributeError, KeyError):
        _mark_shown(d, key, cur)


def _swap(widget, opt: str, *, getter, setter) -> None:
    try:
        cur = getter()
    except (tk.TclError, KeyError, TypeError, AttributeError):
        return
    if not isinstance(cur, str) or not cur:
        return
    _set_translated(_orig_map(widget), opt, cur, setter)


def _apply_menu(menu, depth: int = 0) -> None:
    if depth > 3:
        return
    try:
        end = menu.index("end")
    except (tk.TclError, AttributeError):
        return
    if end is None:
        return
    remember = getattr(menu, "_i18n_menu_orig", None)
    if remember is None:
        remember = {}
        try:
            setattr(menu, "_i18n_menu_orig", remember)
        except AttributeError:
            return
    for i in range(end + 1):
        try:
            cur = menu.entrycget(i, "label")
        except tk.TclError:
            continue
        if isinstance(cur, str) and cur:
            _set_translated(remember, str(i), cur,
                            lambda v, n=i: menu.entryconfigure(n, label=v))
        try:
            child = menu.nametowidget(menu.entrycget(i, "menu"))
            _apply_menu(child, depth + 1)
        except (tk.TclError, KeyError):
            pass


def _apply_tree(widget) -> None:
    """Treeview 的表头与条目标题（帮助窗口、数据管理都靠它显示文字）。"""
    try:
        cols = widget.cget("columns")
    except (tk.TclError, KeyError):
        return
    if not cols:
        return
    d = _orig_map(widget)
    for col in ("#0", *[str(c) for c in cols]):
        try:
            cur = widget.heading(col, "text")
        except (tk.TclError, TypeError):
            continue
        if isinstance(cur, str) and cur:
            _set_translated(d, "head:" + col, cur,
                            lambda v, c=col: widget.heading(c, text=v))

    def walk(iid: str, depth: int = 0) -> None:
        if depth > 4:
            return
        for child in widget.get_children(iid):
            try:
                cur = widget.item(child, "text")
            except tk.TclError:
                continue
            if isinstance(cur, str) and cur:
                _set_translated(d, "item:" + child, cur,
                                lambda v, i=child: widget.item(i, text=v))
            walk(child, depth + 1)

    walk("")


def apply(widget) -> None:
    """递归翻译一棵控件树。可重复调用（切换语言 / 窗口刷新）。"""
    try:
        if not widget.winfo_exists():
            return
    except (tk.TclError, AttributeError):
        return
    # 复杂控件（如帮助窗口的 Text）自己知道怎么按当前语言重绘：
    # 约定一个 _i18n_refresh() 即可，不用把渲染细节塞进本模块。
    refresh = getattr(widget, "_i18n_refresh", None)
    if callable(refresh):
        try:
            refresh()
        except Exception:                                  # noqa: BLE001
            pass
        return
    for opt in ("text",):
        _swap(widget, opt,
              getter=lambda o=opt: widget.cget(o),
              setter=lambda v, o=opt: widget.configure(**{o: v}))
    if isinstance(widget, (tk.Tk, tk.Toplevel)):
        _swap(widget, "title", getter=widget.title, setter=widget.title)
        try:
            name = widget.cget("menu")
            if name:
                _apply_menu(widget.nametowidget(name))
        except (tk.TclError, KeyError, AttributeError):
            pass
    _apply_tree(widget)
    try:
        children = widget.winfo_children()
    except (tk.TclError, AttributeError):
        return
    for child in children:
        try:
            apply(child)
        except (tk.TclError, AttributeError):
            pass


def refresh_all() -> None:
    """刷新当前解释器里所有窗口（Toplevel 都是 root 的后代）。"""
    root = tk._default_root
    if root is None:
        return
    try:
        apply(root)
    except (tk.TclError, AttributeError):
        pass


# --------------------------------------------------------------------------- #
# 自动挂钩：新增窗口 / 消息框不用逐个改
# --------------------------------------------------------------------------- #
def _patch_windows() -> None:
    for cls in (tk.Tk, tk.Toplevel):
        orig = getattr(cls, "__init__")
        if getattr(orig, "_i18n_patched", False):
            continue

        def make(base):
            def init(self, *args, **kwargs):
                base(self, *args, **kwargs)
                # 中文是默认模式，连回调都不排，避免给 26 个 GUI 自测添负担。
                if language() == "zh-CN":
                    return
                try:
                    self.after_idle(lambda w=self: apply(w))
                except (tk.TclError, AttributeError):
                    pass
            init._i18n_patched = True       # type: ignore[attr-defined]
            return init

        setattr(cls, "__init__", make(orig))


def _patch_messagebox() -> None:
    names = ("showinfo", "showerror", "showwarning", "askyesno",
             "askokcancel", "askretrycancel", "askquestion", "askyesnocancel")
    for name in names:
        orig = getattr(messagebox, name, None)
        if orig is None or getattr(orig, "_i18n_patched", False):
            continue

        def make(base):
            def call(title=None, message=None, *args, **kwargs):
                return base(t(title), t(message), *args, **kwargs)
            call._i18n_patched = True       # type: ignore[attr-defined]
            return call

        setattr(messagebox, name, make(orig))


def install() -> None:
    """幂等挂接。模块被 import 时自动执行一次。"""
    global _installed
    if _installed:
        return
    _installed = True
    _load_saved()
    _patch_windows()
    _patch_messagebox()


install()

__all__ = ["LANGS", "LANG_LABELS", "apply", "install", "label", "language",
           "refresh_all", "set_language", "t", "t_all", "table"]
