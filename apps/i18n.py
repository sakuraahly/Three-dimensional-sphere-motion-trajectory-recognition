"""GUI 多语言运行时（只管用户看得见的文字，不碰代码注释 / 日志格式 / 标识符）。

## 设计：中文原文即 key，零改造现有界面代码

项目里有约 2400 条界面文案散在十几个模块里，全部改成 ``t("some.key")``
既要动上千处源码、又会让"新增一句文案"变成"先想个 key、再改两个语言文件"
的双份劳动 —— 结果一定是有人只改中文、英文悄悄漏掉。

这里换一种做法：

- **源码里的中文原样保留**，它就是 key；
- `t(text)` 查 `apps/i18n_data.py` 的**整句英文**表，命中就换，没命中回退中文；
- **不做逐词/子串自动英文化**：`Record与识别`、`Calibration堆栈` 这种混排
  比保留中文更难读；新文案必须先登记、再补完整译文；
- 窗口建好后 `apply(widget)` 递归遍历控件树，把 Label / Button / 标题 /
  菜单 / Treeview 表头与条目的显示文本过一遍 `t()`；
- 新建窗口由 `install()` 挂的钩子在 `after_idle` 自动补一次；
- `apps/tools/i18n_extract.py` 只负责登记新文案，译文在
  `i18n_data.py` 中维护（`i18n_seed.py` 只提供固定词条与整句模板的译文）。

## 明确的安全边界

只翻译**纯展示**文本。参与逻辑判断的值（`StringVar` 里要被 `.get()` 比较的
字符串、文件路径、命令参数）保持原文；下拉框做**显示与逻辑分离**：
`values` 显示英文，`Combobox.get()` / `textvariable.get()` 仍返回中文原文，
`set()` 接中文原文。这样 `if var.get() == "离线（当前会话）"` 不会因为切语言
而静默失效。消息框的标题/正文属于纯展示，已统一挂钩。

## 语言切换

`set_language()` 保存偏好到 `config/language.txt`（本机状态，不入库），
并立即刷新所有已打开窗口。`LANGS` 加一项即可扩展第三种语言：
提取器会为每种语言生成对应资源文件，运行时按 `i18n_data.<CODE>` 取表。
"""

from __future__ import annotations

import os
import re
import string
import tkinter as tk
from functools import lru_cache
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

from apps.i18n_seed import PUNCT, SEED

try:
    from apps import i18n_data
except ImportError:                                   # 首次生成前也能 import
    i18n_data = None                                  # type: ignore[assignment]

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "sphere_tracker").is_dir())
LANG_FILE = ROOT / "config" / "language.txt"
LANGS = ("zh-CN", "en-US")
LANG_LABELS = {"zh-CN": "中文", "en-US": "English"}

_current = "zh-CN"
_installed = False
#: 所有被接管的 Combobox：语言切换时要同时换下拉的**显示值**。
#: 逻辑值始终是建控件时的原文；用户/代码读到的也是原文（见 _patch_combobox）。
_choice_widgets: list = []


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
    # 人工种子**优先于**自动生成的资源：同一个 key 在两边都有时，种子是
    # 人工润色过的整句/术语，资源文件可能仍带旧的机翻措辞（2026-11-24
    # 用户报"英文模式像白痴"后改为这一口径）。
    merged = {k: v for k, v in data.items() if v}
    merged.update(SEED)
    merged.update(PUNCT)
    return merged


def t(text):
    """翻译一段**界面文本**；非字符串、中文模式、未登记的文本原样返回。"""
    if not isinstance(text, str) or not text or _current == "zh-CN":
        return text
    hit = table(_current).get(text)
    if hit:
        return hit
    # 不做逐词/子串自动英文化：中文和英文词混在一句里（如
    # “Record与识别”“Calibration堆栈”）比保留中文更难读。未登记的新文案
    # 先回退中文，等 `i18n_extract.py --write` 登记后再人工补整句英文。
    return text


def t_all(items):
    return [t(x) if isinstance(x, str) else x for x in items]


@lru_cache(maxsize=2)
def _template_matchers(lang: str) -> tuple:
    """把带 `{name}` 的整句条目编译成"锚定正则 → 英文模板"。

    只用于**运行时拼出来的整句**（状态条 / messagebox 的 f-string）。它不是
    逐词替换：必须整句命中一个已经登记的中文模板，才用对应英文模板重写；
    匹配不到就原样返回，绝不猜。这样 `相机枚举完成：4 台在线…` 这类
    "数据夹在中文骨架里"的句子也能翻，而设备名、路径等数据保持原样。
    """
    entries = []
    for key, value in table(lang).items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        if "{" not in key or "}" not in key or not re.search(r"[\u4e00-\u9fff]", key):
            continue
        try:
            parsed = list(string.Formatter().parse(key))
        except ValueError:
            continue
        fields = [field for _, field, _, _ in parsed if field is not None]
        if not fields or len(key) > 500:
            continue
        # 只接受简单的 {name} / {name:spec}，不碰 {a[b]} / {a.b}。
        if any(not str(f).isidentifier() for f in fields):
            continue
        chunks = []
        names = []
        for literal, field, spec, conv in parsed:
            chunks.append(re.escape(literal))
            if field is not None:
                name = str(field)
                idx = len(names)
                names.append(name)
                chunks.append(f"(?P<f{idx}>.*?)")
        try:
            pattern = re.compile("^" + "".join(chunks) + "$", re.S)
        except re.error:
            continue
        # 去掉英文模板里的格式说明：捕获到的是已经格式化后的显示值。
        english = re.sub(r"\{([^{}:]+):[^}]+\}", r"{\1}", value)
        entries.append((len(key), pattern, names, english))
    entries.sort(key=lambda item: item[0], reverse=True)
    return tuple(entries)


def dynamic(text):
    """翻译运行时拼好的整句：先整句精确，再模板匹配；不逐词猜。"""
    if not isinstance(text, str) or not text or _current == "zh-CN":
        return text
    hit = t(text)
    if hit != text:
        return hit
    for _size, pattern, names, english in _template_matchers(_current):
        m = pattern.match(text)
        if m is None:
            continue
        values = {}
        for i, name in enumerate(names):
            values[name] = m.group(f"f{i}")
        try:
            return english.format(**values)
        except (KeyError, IndexError, ValueError):
            continue
    return text


def tf(template: str, **kwargs) -> str:
    """带占位符的整句翻译：先翻**模板**，再 format 填动态值。

    为什么需要它（2026-11-23）：状态条/卡片上有大量"就绪：已标定 fx=1388…"
    这类句子，数据（fx、装置名、日期）不该翻译、中文骨架该翻译。整句 t()
    对拼好的动态串永远命不中词典，只能原样中文；逐词拼接又会产生
    "Calibration堆栈"那种混排（会破坏可读性，已在 d9ff001 废弃）。
    tf() 走中间路线：词典登记**中文模板**（含 `{place}`），运行时先取译文
    模板再填值。调用点必须写`字面量模板`，才能被 i18n_extract 扫到。

    注意：这些状态文本是代码显式 configure 上去的，不走 apply() 的原文
    记账，所以切语言时不会把英文结果误存成"原文"（B-185 只影响 apply 路径）。
    """
    translated = t(template)
    try:
        return translated.format(**kwargs)
    except (KeyError, IndexError, ValueError):
        return translated


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
    _template_matchers.cache_clear()
    try:
        LANG_FILE.parent.mkdir(parents=True, exist_ok=True)
        LANG_FILE.write_text(lang, encoding="utf-8")
    except OSError:
        pass
    refresh_choices()
    refresh_all()


def label(lang: str | None = None) -> str:
    return LANG_LABELS.get(lang or _current, lang or _current)


def _load_saved() -> None:
    global _current
    # 环境变量优先于本机偏好：回归测试用 SM_LANG=zh-CN 让判据与用户语言
    # 解耦（用户本机存了 en-US 时，中文关键词的帮助/窗口高度断言会假红，
    # 2026-11-24 实测）。GUI 正常启动时没有这个变量，行为不变。
    forced = os.environ.get("SM_LANG")
    if forced in LANGS:
        _current = forced
        return
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


def forget(widget) -> None:
    """清空控件的原文记账：下次 apply() 以**当前**显示值为原文。

    给"运行时拼出来的状态文本"用（StatusBar / 卡片状态标签）。这类文本
    由代码用 tf() 按当前语言现场生成，不该让记账把上一语言的显示值当原文
    —— 否则切语言时 apply() 会把新生成的英文误翻回旧中文（2026-11-23）。
    """
    try:
        delattr(widget, "_i18n_orig")
    except (AttributeError, TypeError):
        pass


def _swap(widget, opt: str, *, getter, setter) -> None:
    try:
        cur = getter()
    except (tk.TclError, KeyError, TypeError, AttributeError):
        return
    if not isinstance(cur, str) or not cur:
        return
    _set_translated(_orig_map(widget), opt, cur, setter)


# --------------------------------------------------------------------------- #
# 下拉框：显示英文，逻辑值仍是中文原文
# --------------------------------------------------------------------------- #
# 为什么不能只翻 values（2026-11-24 用户报"下拉菜单还是中文"）：
# `ttk.Combobox` 的 values 同时承担"给用户看"和"给代码判断"两种职责。
# `_gui_stereo` 用 `var_method.get()` 去 METHODS 里找 key，`_gui_mono` 用
# `var_source.get() == SOURCE_REALTIME` —— 直接把 values 换成英文，界面好看了，
# 判断却永远匹配不上，最后静默选错方法。这里把这层拆开：
#
#   * values 只放显示值（t(原文)）；
#   * Combobox.get() / textvariable.get() 反查回原文；
#   * Combobox.set() / textvariable.set() 接原文，写进控件时换成显示值。
#
# 这样既不动各窗口的逻辑，也允许语言切换时重算显示值。
def _choice_maps(values) -> tuple[dict[str, str], dict[str, str]]:
    orig = [str(v) for v in values]
    shown = [t(v) for v in orig]
    forward = dict(zip(orig, shown))
    reverse = dict(zip(shown, orig))
    return forward, reverse


def _combo_logical(widget, value):
    reverse = getattr(widget, "_i18n_choice_reverse", {})
    return reverse.get(value, value)


def _combo_display(widget, value):
    forward = getattr(widget, "_i18n_choice_forward", {})
    return forward.get(value, t(value))


def _patch_choice_var(var, widget) -> None:
    if getattr(var, "_i18n_choice_patched", False):
        return
    base_get, base_set = var.get, var.set

    def get():
        return _combo_logical(widget, base_get())

    def set(value):
        return base_set(_combo_display(widget, value))

    var.get = get
    var.set = set
    var._i18n_choice_patched = True


def _install_choice_widget(widget, textvariable=None) -> None:
    """接管一个 Combobox：翻译 values、包住 get/set、登记到切换表。"""
    try:
        raw_values = widget.cget("values")
    except (tk.TclError, TypeError):
        return
    orig = tuple(str(v) for v in (raw_values or ()))
    if not orig:
        return
    forward, reverse = _choice_maps(orig)
    widget._i18n_choice_values = orig
    widget._i18n_choice_forward = forward
    widget._i18n_choice_reverse = reverse

    # 当前值先按原文记住，再把显示值写回去（否则初始仍显示中文）。
    try:
        current = widget.get()
    except (tk.TclError, AttributeError):
        current = ""
    logical = _combo_logical(widget, current)
    try:
        widget.configure(values=[forward.get(v, t(v)) for v in orig])
    except (tk.TclError, TypeError):
        pass

    if not getattr(widget, "_i18n_choice_patched", False):
        base_get, base_set = widget.get, widget.set

        def get():
            return _combo_logical(widget, base_get())

        def set(value):
            return base_set(_combo_display(widget, value))

        widget.get = get
        widget.set = set
        widget._i18n_choice_patched = True

    if logical:
        try:
            widget.set(logical)
        except (tk.TclError, TypeError):
            pass
    # textvariable 就是调用方传进来的 tk.StringVar；直接复用，不另建第二份状态。
    if textvariable is not None and hasattr(textvariable, "get"):
        _patch_choice_var(textvariable, widget)

    if widget not in _choice_widgets:
        _choice_widgets.append(widget)


def _patch_comboboxes() -> None:
    base_init = ttk.Combobox.__init__
    if getattr(base_init, "_i18n_patched", False):
        return

    def init(self, *args, **kwargs):
        var = kwargs.get("textvariable")
        base_init(self, *args, **kwargs)
        _install_choice_widget(self, textvariable=var)

    init._i18n_patched = True       # type: ignore[attr-defined]
    ttk.Combobox.__init__ = init


def refresh_choices() -> None:
    """语言切换后重算所有下拉框的显示值，逻辑值保持不变。"""
    for widget in list(_choice_widgets):
        try:
            if not widget.winfo_exists():
                _choice_widgets.remove(widget)
                continue
            current = widget.get()          # 已被包成原文
            orig = getattr(widget, "_i18n_choice_values", ())
            forward, reverse = _choice_maps(orig)
            widget._i18n_choice_forward = forward
            widget._i18n_choice_reverse = reverse
            widget.configure(values=[forward.get(v, t(v)) for v in orig])
            widget.set(current)
        except (tk.TclError, TypeError, AttributeError):
            continue


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
                cur = ""
            if isinstance(cur, str) and cur:
                _set_translated(d, "item:" + child, cur,
                                lambda v, i=child: widget.item(i, text=v))
            # Treeview 单元格 values= 也要翻（2026-11-24 截图：相机管理表里
            # "内置"/"未实测"/"left ✓" 全是中文）。只翻**整格精确命中词典**
            # 的字符串——设备名/路径/端口这些是数据，本来就不在词典里。
            # 整行 values 当一组记账：原文 + 上次显示值，切换语言时
            # 始终从原文重翻（同 _set_translated 的双向记账口径）。
            key = "vals:" + child
            try:
                vals = tuple(widget.item(child, "values") or ())
            except tk.TclError:
                vals = ()
            if vals and all(isinstance(x, str) or x is None for x in vals):
                dval = _orig_map(widget)
                prev = dval.get(key)
                shown = dval.get(key + ":shown")
                if prev is None or (shown is not None and vals != shown
                                    and vals != tuple(prev)):
                    dval[key] = vals          # 程序改过 → 当前值当新原文
                original = dval[key]
                new_vals = tuple(t(x) if isinstance(x, str) else x
                                 for x in original)
                try:
                    widget.item(child, values=new_vals)
                    dval[key + ":shown"] = new_vals
                except tk.TclError:
                    pass
            walk(child, depth + 1)

    walk("")


def apply_self(widget) -> None:
    """只翻译 widget **自身**（text / 窗口 title / 菜单 / 表头），不递归子树。

    给定义了 `_i18n_refresh()` 钩子的人用：`apply()` 遇到钩子会直接调它并
    return，不会再翻 self —— 于是"有钩子的窗口"标题反而永远留在中文
    （2026-11-24 相机管理窗口实测：title="相机管理" 切英文仍是中文）。
    钩子内部先 `apply_self(self)` 翻自身，再 `apply_children(self)` 翻子树。
    """
    try:
        if not widget.winfo_exists():
            return
    except (tk.TclError, AttributeError):
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
    # Combobox 的 values 由 `_install_choice_widget` 单独接管：显示英文，
    # `.get()` / textvariable 仍返回原文（见模块 docstring 的安全边界）。
    # 这里不再直接碰 values，避免把逻辑值改成译文。


def apply_children(widget) -> None:
    """只翻译 widget 的**子树**，不触发 widget 自己的 _i18n_refresh。

    给"动态状态句"的容器用：容器的 _i18n_refresh 先用新语言重新生成那几句
    （它们不在词典里，apply() 的整句匹配够不着），再调本函数把其余静态控件
    照常翻一遍。直接调 apply(容器) 会再次触发钩子 → 递归。
    """
    try:
        if not widget.winfo_exists():
            return
    except (tk.TclError, AttributeError):
        return
    for child in widget.winfo_children():
        apply(child)


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
                return base(dynamic(title), dynamic(message), *args, **kwargs)
            call._i18n_patched = True       # type: ignore[attr-defined]
            return call

        setattr(messagebox, name, make(orig))


def _patch_dialogs() -> None:
    """simpledialog 的标题/提示、filedialog 的标题也走同一套翻译。

    这些窗口不是我们创建的 Toplevel（由 Tk 自己弹），`apply()` 遍历不到；
    不在这里挂钩子，英文界面里点「改名」「选择目录」仍会弹出中文标题。
    """
    for name in ("askstring", "askinteger", "askfloat"):
        orig = getattr(simpledialog, name, None)
        if orig is None or getattr(orig, "_i18n_patched", False):
            continue

        def make(base):
            def call(title, prompt, *args, **kwargs):
                return base(dynamic(title), dynamic(prompt), *args, **kwargs)
            call._i18n_patched = True       # type: ignore[attr-defined]
            return call

        setattr(simpledialog, name, make(orig))

    for name in ("askopenfilename", "asksaveasfilename", "askdirectory"):
        orig = getattr(filedialog, name, None)
        if orig is None or getattr(orig, "_i18n_patched", False):
            continue

        def make(base):
            def call(*args, **kwargs):
                if kwargs.get("title"):
                    kwargs["title"] = dynamic(kwargs["title"])
                return base(*args, **kwargs)
            call._i18n_patched = True       # type: ignore[attr-defined]
            return call

        setattr(filedialog, name, make(orig))


def install() -> None:
    """幂等挂接。模块被 import 时自动执行一次。"""
    global _installed
    if _installed:
        return
    _installed = True
    _load_saved()
    _patch_windows()
    _patch_messagebox()
    _patch_dialogs()
    _patch_comboboxes()


install()

__all__ = ["LANGS", "LANG_LABELS", "apply", "apply_self", "apply_children",
           "forget", "install", "label", "language", "refresh_all",
           "set_language", "t", "t_all", "tf", "dynamic", "table"]
