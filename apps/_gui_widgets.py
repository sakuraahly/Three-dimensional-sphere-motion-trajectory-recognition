"""新界面的基础控件：样式、卡片、参数行、按钮层级、换行回排、底部状态条。

为什么单独一个模块（2026-10-27 计划书 §2.5）
--------------------------------------------
旧界面 3055 行里，"卡片长什么样"和"标定怎么跑"写在同一个类上，115 个方法
互相看得见彼此的属性，于是改一处谁都不敢动。本模块**只做形状**：不 import
任何业务模块、不读命令行、不创建 Tk 根窗口 —— 因此可以被单目窗、双目窗、
工作台、控制台同时复用，且能被无显示器环境下的测试单独实例化。

三条从旧界面学到的硬约束（落成代码，而不是写在文档里等后来者踩）

1. **卡片说明必须换行**。不设 ``wraplength`` 时第 1 步那张卡的自然宽度实测
   1941 px（整窗 1973），屏幕只有 1600 —— 新加的控件被挤到窗口外，用户在
   反馈里说的是"控件不见了"。本模块所有说明文字一律带 ``wraplength``，
   并由 :meth:`Surface.reflow` 随视口宽度重算。
2. **行号由辅助函数分配**。旧代码让调用方手算 ``row``，而每个参数行占两行
   （标签/输入一行 + 说明一行），手工 ``grid(row=3)`` 正好落在被占的行上，
   文字与输入框互相盖住。这里每个 parent 自带游标，谁先放谁占下一行。
3. **主操作与次操作必须有视觉差异**。旧界面第 1 步 9 个按钮全是同款
   ``TButton``，用户找不到"该点哪个"；这里 :meth:`Surface.primary` 与
   :meth:`Surface.secondary` 是仅有的两个放按钮的入口，前者带 ``Go.TButton``。

文案约定：**控件上不许出现 Markdown 标记**。旧界面有 20 处 ``**加粗**``
原样印在按钮和提示里（B-96），用户以为是程序坏了。本模块不做渲染，
也不接受带标记的文案 —— 传进来什么样就显示什么样，所以传之前就写干净。
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from apps import i18n                 # 模块导入即挂接"新窗口自动翻译 / 消息框翻译"
from apps._gui_geometry import fit_height
from sphere_tracker.config import AUTO_PRESET

__all__ = [
    "BG", "ACCENT", "INK", "MUTED", "WARN", "OK", "FONT_FAMILY",
    "CARD_HINT_WRAP", "PARAM_HINT_WRAP", "INLINE_NOTE_WRAP",
    "init_style", "Surface", "ScrollPage", "StatusBar", "fit_to_content",
    "BALL_METAL", "BALL_PLASTIC", "BALL_CUSTOM", "BALL_KINDS", "BALL_HINTS",
    "METHOD_METAL", "METHOD_PLASTIC", "PRESET_NONE",
    "method_for_kind", "kind_for_method", "kind_default_for", "kind_key",
    "BallSettings",
    "preset_label", "preset_options", "detector_needs_note",
]

# 配色沿用旧界面（不换皮：换肤会把 199 个控件的视觉经验全部作废，见计划书 §8）
BG = "#eef2f8"
ACCENT = "#2563eb"
INK = "#1f2937"
MUTED = "#6b7280"
WARN = "#b45309"
OK = "#15803d"
FONT_FAMILY = "Microsoft YaHei UI"

CARD_HINT_WRAP = 820      # 初值；实际由 Surface.reflow 按视口宽度刷新
PARAM_HINT_WRAP = 720     # 初值；同上
INLINE_NOTE_WRAP = 520    # 按钮行里的原因文字，封顶（再宽就等于"一句说明撑满整卡"）


def init_style(root: tk.Misc) -> ttk.Style:
    """配置 ttk 样式。整个进程只该调用一次（App 启动时）。"""
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass                                   # 非 Windows 主题缺 clam 时不致命
    root.option_add("*Font", (FONT_FAMILY, 10))
    style.configure("Card.TLabelframe", background=BG, borderwidth=0)
    style.configure("Card.TLabelframe.Label", foreground=ACCENT,
                    font=(FONT_FAMILY, 11, "bold"), background=BG)
    style.configure("TFrame", background=BG)
    style.configure("TLabel", background=BG, foreground=INK)
    style.configure("Title.TLabel", background=BG, foreground=INK,
                    font=(FONT_FAMILY, 15, "bold"))
    style.configure("Hint.TLabel", background=BG, foreground=MUTED,
                    font=(FONT_FAMILY, 9))
    style.configure("Warn.TLabel", background=BG, foreground=WARN,
                    font=(FONT_FAMILY, 9))
    style.configure("Ok.TLabel", background=BG, foreground=OK,
                    font=(FONT_FAMILY, 9))
    # 主操作：实心 accent。clam 下必须同时给 background 与 map，否则按下态回退灰色。
    style.configure("Go.TButton", font=(FONT_FAMILY, 11, "bold"),
                    padding=(14, 8), background=ACCENT, foreground="white")
    style.map("Go.TButton",
              background=[("active", "#1d4ed8"), ("disabled", "#93a4bf")],
              foreground=[("disabled", "#eef2f8")])
    style.configure("TButton", padding=6)
    style.configure("TRadiobutton", background=BG)
    style.configure("TCheckbutton", background=BG)
    return style


class Surface:
    """一页内容的控件工厂：卡片、参数行、按钮、换行回排。

    一个窗口一个 Surface（或每页一个）。它只记住"哪些标签需要重新换行"，
    不持有任何业务状态 —— 业务状态归 ``AppState``（见 ``_gui_app``）。
    """

    def __init__(self, root: tk.Misc) -> None:
        self.root = root
        self._wrapped: list[tuple[ttk.Label, str]] = []

    # ---------- 换行回排 ----------
    def reflow(self, viewport: int) -> None:
        """视口宽度变了：把已登记的说明文字重设换行宽度。

        ``viewport`` 是内容区宽度（不是窗口宽度）。卡片左右各留 16 px padding
        再加 14 px 内边距，所以卡片说明比视口窄约 80；参数说明还要再让出
        标签列，取 160。下限 320/280 是为了窄窗口下不至于一个词一行。

        ``inline`` 是按钮行里那段原因文字：它要让出**按钮占掉的宽度**，
        所以比视口窄 360（实测一行 4 个按钮约 300 + 边距 60），上限 520。
        """
        card_wrap = max(320, int(viewport) - 80)
        param_wrap = max(280, int(viewport) - 160)
        inline_wrap = max(220, min(INLINE_NOTE_WRAP, int(viewport) - 360))
        alive: list[tuple[ttk.Label, str]] = []
        for lab, kind in self._wrapped:
            wrap = {"card": card_wrap, "param": param_wrap,
                    "inline": inline_wrap}.get(kind, param_wrap)
            try:
                lab.configure(wraplength=wrap)
                alive.append((lab, kind))
            except tk.TclError:
                continue                        # 已被销毁的标签：静默移出登记表
        self._wrapped = alive

    # ---------- 卡片 ----------
    def card(self, parent: tk.Misc, title: str, hint: str = "", *,
             pady: tuple[int, int] = (8, 0), fill: str = "x",
             expand: bool = False) -> ttk.Frame:
        """一张卡片。返回的 Frame 是内部 grid 的父容器。

        卡片间距默认 (8, 0)（2026-11-03 由 10 调到 8）：四步叠起来时，每张卡
        多 2 px 就是最后一步多 8 px —— 双目「大角度」第④步原本恰好在视口下沿，
        加一行控件后全靠这 8 px 回到预算内（`tests/selftest_window_mode.py`
        逐卡量下沿）。同时保留一点留白：贴死会让相邻步骤读成一整块。

        ``fill``/``expand`` 默认 ``"x"``/``False`` —— 参数卡按内容定高，这没问题。
        但**日志那种要吃掉剩余高度的卡片必须传 ``fill="both", expand=True``**，
        否则窗口放大之后内容还缩在左上角（2026-10-28 截图里控制台就是这样：
        窗口 900×560，内容只占 530×330，右边下面全空）。
        """
        box = ttk.LabelFrame(parent, text=title, style="Card.TLabelframe",
                             padding=(14, 8))
        box.pack(fill=fill, expand=expand, padx=16, pady=pady)
        if hint:
            lab = ttk.Label(box, text=hint, style="Hint.TLabel",
                            wraplength=CARD_HINT_WRAP)
            # columnspan 给足：卡片里最多用到第 4 列，多跨几列免得被挤成两行
            lab.grid(row=0, column=0, columnspan=9, sticky="w", pady=(0, 8))
            self._wrapped.append((lab, "card"))
            # ⚠ 说明占了 row=0，**必须把游标推到 1**，否则第一个 action_row /
            # param_* 会拿到 row=0 直接压在这段文字上。2026-10-28 截图里
            # 主按钮确实压住了卡片说明 —— 这是旧界面"文字盖住输入框"的同款 bug，
            # 换了个方向又犯了一次。所以推进游标这件事写在这里，不靠调用方记得。
            setattr(box, "_next_row", 1)
        return box

    # ---------- 参数行 ----------
    def _next_row(self, parent: tk.Misc) -> int:
        """给 parent 分配下一个空行（每次调用 +1）。

        行号存在 parent 对象上（Tk 控件允许挂任意属性），因此**同一张卡片里
        多个 Surface 方法交替调用也不会撞行** —— 这是旧界面手工算 row 撞出来的
        那个"文字盖住输入框"bug 的根治办法。
        """
        row = getattr(parent, "_next_row", 0)
        setattr(parent, "_next_row", row + 1)
        return row

    def hint(self, parent: tk.Misc, text: str, *, indent: int = 0,
             style: str = "Hint.TLabel") -> ttk.Label:
        """一行说明（跨列、限宽换行）。"""
        lab = ttk.Label(parent, text=text, style=style,
                        wraplength=PARAM_HINT_WRAP)
        lab.grid(row=self._next_row(parent), column=indent, columnspan=5,
                 sticky="w", pady=(0, 4))
        self._wrapped.append((lab, "param"))
        return lab

    def inline_note(self, row: ttk.Frame, text: str = "") -> ttk.Label:
        """按钮行里的说明（在按钮**右边**，不单独占一行）。

        为什么要有它（2026-10-31，工作台 ⑤ 被挤出首屏）
        ------------------------------------------------
        "这一步为什么不能点"原本占一行：即使**空着**，一个空 TLabel 也有
        27 px 行高 —— 5 张卡就是 135 px，正好把 ⑤ 推到首屏下面。而它解释的
        只是旁边那个按钮，放在同一行本来就更近。

        ``row`` 必须是 :meth:`action_row` 的返回值（那是一个 pack 容器）；
        宽度登记为 ``kind="inline"``，随视口重排但封顶 520 —— 再宽就等于
        "一行说明撑满整卡"，那正是旧界面文字把界面撑爆的形状。
        """
        lab = ttk.Label(row, text=text, style="Hint.TLabel",
                        wraplength=INLINE_NOTE_WRAP)
        lab.pack(side="left", padx=(10, 4))
        self._wrapped.append((lab, "inline"))
        return lab

    def param_entry(self, parent: tk.Misc, label: str, value: str = "",
                    hint: str = "", *, width: int = 10) -> ttk.Entry:
        """一行参数：``标签 [输入框]``，说明另起一行放**下面**。

        说明放下面而不是右边：那一列会按说明文字的自然宽度撑开整张卡片
        （实测某条说明单独就要 567 px）。
        """
        row = self._next_row(parent)
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w",
                                           pady=(4, 0))
        ent = ttk.Entry(parent, width=width)
        if value:
            ent.insert(0, value)
        ent.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=(4, 0))
        if hint:
            self.hint(parent, hint)
        return ent

    def param_combo(self, parent: tk.Misc, label: str, values: list[str],
                    value: str = "", hint: str = "", *,
                    width: int = 10) -> ttk.Combobox:
        """一行参数：``标签 [只读下拉]``，排版与 param_entry 一致。"""
        row = self._next_row(parent)
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w",
                                           pady=(4, 0))
        cb = ttk.Combobox(parent, values=values, width=width, state="readonly")
        if value:
            cb.set(value)
        cb.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=(4, 0))
        if hint:
            self.hint(parent, hint)
        return cb

    def param_check(self, parent: tk.Misc, text: str, variable: tk.BooleanVar,
                    hint: str = "") -> ttk.Checkbutton:
        """一行复选框。``text`` 必须是**行为描述**，不是功能名。"""
        chk = ttk.Checkbutton(parent, text=text, variable=variable)
        chk.grid(row=self._next_row(parent), column=0, columnspan=2,
                 sticky="w", pady=(4, 0))
        if hint:
            self.hint(parent, hint)
        return chk

    def readonly_row(self, parent: tk.Misc, label: str, value: str) -> ttk.Label:
        """一行"当前生效：…"—— L3 参数不放进界面，但要让人看得见值。"""
        row = self._next_row(parent)
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w",
                                           pady=(2, 0))
        lab = ttk.Label(parent, text=value, style="Hint.TLabel")
        lab.grid(row=row, column=1, columnspan=3, sticky="w",
                 padx=(8, 0), pady=(2, 0))
        return lab

    # ---------- 按钮 ----------
    def action_row(self, parent: tk.Misc) -> ttk.Frame:
        """按钮行容器。**所有按钮都放进 action_row**，不要直接 grid 到卡片上。

        旧界面第 3 步 5 个按钮同在 ``row=1``、没有分组容器，混进了测试/规范
        自查这类开发工具。有容器才谈得上"主次"。
        """
        row = ttk.Frame(parent)
        row.grid(row=self._next_row(parent), column=0, columnspan=9,
                 sticky="w", pady=(10, 2))
        return row

    def primary(self, parent: tk.Misc, text: str, command) -> ttk.Button:
        """当前步骤的**唯一**主操作（实心 accent）。一步只该有一个。"""
        btn = ttk.Button(parent, text=text, style="Go.TButton",
                         command=command)
        btn.pack(side="left", padx=(0, 8))
        return btn

    def secondary(self, parent: tk.Misc, text: str, command) -> ttk.Button:
        """次要操作（默认样式）：刷新、查看、换方法。"""
        btn = ttk.Button(parent, text=text, command=command)
        btn.pack(side="left", padx=(0, 8))
        return btn


# --------------------------------------------------------------------------- #
# 球种类与检测参数（2026-11-03 增）：三个窗口共用的一份词表 + 一块控件
# --------------------------------------------------------------------------- #
# 为什么是"球种类"而不是"检测方法"摆前面
# --------------------------------------
# 用户知道自己拿的是什么球（镀铬的／红色的），不知道 `specular` 与 `color`
# 对他那颗球意味着什么。检测方法是他这句话的**翻译结果**，所以只能跟在后面
# 显示、不能让他先做一次术语题。
#
# 这份词表是**界面语言**，不是检测器清单：方法名一律由调用方从
# `sphere_tracker/detectors/registry.py` 取（R7，界面不许手写名字），
# 这里只写"金属球对应注册表里的哪个名字"这层映射。
BALL_METAL = "镜面金属球"
BALL_PLASTIC = "纯色塑料球"
BALL_CUSTOM = "自定义"
BALL_KINDS = (BALL_METAL, BALL_PLASTIC, BALL_CUSTOM)

#: 球种类 → 检测方法（注册表里的名字）。「自定义」不在表里 = 由下拉自己决定。
METHOD_METAL = "specular"
METHOD_PLASTIC = "color"
_BALL_KIND_METHOD = {BALL_METAL: METHOD_METAL, BALL_PLASTIC: METHOD_PLASTIC}
#: 球种类 → 写进 `seeds.json` 的 `ball_kind`（英文键，给命令行读）。
#: 用它而不是把中文界面文字写进数据文件：数据契约要能跨语言/跨版本读。
_BALL_KIND_KEY = {BALL_METAL: "metal", BALL_PLASTIC: "plastic"}


def kind_key(kind: str) -> str:
    """界面上的球种类 → ``seeds.json`` 的 ``ball_kind``（``metal`` / ``plastic``）。

    「自定义」返回空串：那一档本来就不知道该用哪个检测器，写一个猜的值
    比不写更糟 —— 读取方会把"没记录"当旧文件兼容处理。
    """
    return _BALL_KIND_KEY.get(str(kind or "").strip(), "")

#: 预设下拉里的「不用」哨兵。沿用经典界面的同一个词 —— 换一个说法等于
#: 让两套界面在同一个概念上各讲一套话。
PRESET_NONE = "不使用"

#: 选完球种类**当场**要说的话。它不是装饰：选「镜面金属球」却直接点「开始识别」
#: 会得到 0 配对（本机实测这颗球 288 帧里 color 落在球上 0 帧），
#: 而那看起来就像"程序跑完了"。
BALL_HINTS = {
    BALL_METAL: "镜面金属球只有高光能定位：先「点选球心」人工点一次球心再识别，"
                "否则自动检测找不到球（实测 288 帧里 color 落在球上 0 帧）。",
    BALL_PLASTIC: "纯色塑料球靠颜色分割：球的颜色要和背景分得开（同色背景会把"
                  "整片区域当成球）。颜色预设选「自动取色」（按画面估主色，"
                  "不用知道球色）或选一个贴合的球色；不选则用配置里的白球预设。",
    BALL_CUSTOM: "自定义：检测方法直接取注册表里的实现；换一种球就换一个方法。",
}


def method_for_kind(kind: str) -> str:
    """球种类 → 检测方法；「自定义」（或认不出的值）返回空串 = 不指定。"""
    return _BALL_KIND_METHOD.get(str(kind or "").strip(), "")


def kind_for_method(method: str, current: str = "") -> str:
    """检测方法 → 界面该显示的球种类。

    只有"当前这一档本来就对应这个方法"时才保留它，其余一律显示「自定义」——
    这就是"手动改方法 → 球种类变成自定义"那条规则的**全部实现**：
    它只决定显示，不反过来改方法。
    """
    m = str(method or "").strip()
    cur = str(current or "").strip()
    if m and _BALL_KIND_METHOD.get(cur) == m:
        return cur
    return BALL_CUSTOM


def kind_default_for(method: str) -> str:
    """初始那一档（只在建控件时用一次）：specular→金属、color→纯色、其余→自定义。

    与 :func:`kind_for_method` 分开是因为语义不同：这里**允许**从方法反推种类
    （开窗时没有"用户当前那一档"可言），而联动发生时不许这样做。
    """
    m = str(method or "").strip()
    for kind, name in _BALL_KIND_METHOD.items():
        if name == m:
            return kind
    return BALL_CUSTOM


#: 预设名的**首字母 → 中文颜色字**。键是单个字母，**不是** `red_ball` 这样的
#: 完整名字 —— 所以它不是预设表的副本：`COLOR_PRESETS` 加一种球色时这里不用动
#: （加 `purple_ball` 自动得到"紫球"；字母不认识就原样显示，绝不把注册表里
#: 真实存在的预设从下拉里藏掉）。
_PRESET_INITIAL = {"r": "红", "o": "橙", "g": "绿", "b": "蓝",
                   "y": "黄", "w": "白", "k": "黑", "p": "紫", "c": "青"}

#: 预设名统一的后缀（`red_ball` → `red`）。
_PRESET_SUFFIX = "_ball"


def preset_label(name: str) -> str:
    """``red_ball`` → ``红球``；``auto`` → ``自动取色``；不认识的退回原名。

    为什么允许退回英文：能退不能藏。让一个新预设显示成 ``purple`` 只是不好看，
    而把它从下拉里悄悄丢掉，用户会以为"程序不支持这个颜色"。
    """
    text = str(name or "").strip()
    if not text or text == PRESET_NONE:
        return PRESET_NONE
    if text == "auto":
        # 它不是一种颜色（见 `ColorConfig.preset_choices`），单独给个人话名字。
        return "自动取色"
    stem = text[:-len(_PRESET_SUFFIX)] if text.endswith(_PRESET_SUFFIX) else text
    cn = _PRESET_INITIAL.get(stem[:1].lower())
    return f"{cn}球" if cn else (stem or text)


def preset_options(presets) -> dict[str, str]:
    """``{下拉显示: 真正要下传的 --preset 值}``，从预设表推出来，不手写。

    重名（例如将来同时有 ``red_ball`` 与 ``rose_ball``，都映射到"红球"）时
    **退回内部名**而不是合并：合并会让其中一项永远选不到，而界面又看不出来
    —— 与"检测器清单手抄漏一项"是同一种静默失效。
    """
    names = [str(p) for p in presets]
    labels = [preset_label(p) for p in names]
    clash = {lab for lab in labels if labels.count(lab) > 1}
    out: dict[str, str] = {}
    for name, lab in zip(names, labels):
        out[name if lab in clash else lab] = name
    return out


def _module_missing(pkg: str) -> bool:
    """这个 import 名是否**没装**（只查文件，不 import —— 不能因为界面建控件
    就把 torch 拉起来）。包名与 import 名允许不同（``segment-anything`` →
    ``segment_anything``）。"""
    import importlib.util

    for cand in (pkg, str(pkg).replace("-", "_")):
        try:
            if importlib.util.find_spec(cand) is not None:
                return False
        except (ImportError, ValueError, AttributeError):
            continue
    return True


def detector_needs_note(method: str) -> str:
    """这个检测方法缺哪个可选依赖；不缺 / 不适用 → 空串。

    为什么要有它：``yolo`` / ``yolo_sam`` 在注册表里声明了 ``needs``，而它们的
    import 发生在**首次推理**（``neural.py`` 的 ``_ensure_model``）—— 也就是说
    界面选得动、录完点下去才炸，炸在离用户这次操作很远的地方。这里提前查一次，
    把"装什么"写在状态条上。

    唯一一处函数内 import（不是模块级）：本模块的约定是"不依赖业务模块"，
    而注册表是``needs`` 的唯一来源。放在函数里，模块仍可在无依赖环境下被
    单独 import/实例化（判据就是这么跑的）。
    """
    m = str(method or "").strip()
    if not m:
        return ""
    try:
        from sphere_tracker.plugins import describe
    except Exception:                                   # noqa: BLE001
        return ""
    entry = next((d for d in describe("detector") if d.get("name") == m), None)
    if not entry:
        return ""
    missing = [pkg for pkg in entry.get("needs") or [] if _module_missing(pkg)]
    if not missing:
        return ""
    return (f"「{m}」依赖 {'、'.join(missing)}，当前没装："
            f"pip install {' '.join(missing)}；装上之前它每帧都会失败")


class BallSettings:
    """球种类 + 检测方法 + 颜色预设（+ 可选的球径 / 半径）一行控件。

    三个窗口都挂它（单目第②步、双目第④步、工作台第③步）—— 为什么必须共用
    一块而不是各摆三个下拉：球种类到检测方法是一条**联动**，写三遍就会漂三遍，
    而"界面选了 A、命令行跑的是 B"在本项目已经出过三次（球径 mm/m、
    「检查装置」一个参数都不传、录制不传 ``--preview``）。

    三条刻意的不作为
    ----------------
    1. **联动只改显示**：切换球种类只写"检测方法"这一个值（它本来就是该档的
       定义），球直径、焦距、会话路径一个字都不碰；
    2. **不猜默认值**：球直径的输入变量由调用方传进来 —— 每个窗口只有一处
       球径真相，这里再放一个默认值就是第二处；
    3. **不 import 业务模块**：方法清单由调用方从注册表取，控件只照着画。

    ``methods`` / ``presets`` 必须是调用方从唯一来源取来的（注册表 /
    ``ColorConfig.preset_names()``），控件不提供任何内置清单。
    """

    def __init__(self, surface: Surface, parent: tk.Misc, *,
                 methods, presets=(), method_var: tk.StringVar | None = None,
                 preset_var: tk.StringVar | None = None, kind: str = "",
                 diameter_var: tk.StringVar | None = None,
                 diameter_label: str = "球直径（mm）", diameter_width: int = 8,
                 on_diameter_key=None, radius_vars: tuple | None = None,
                 radius_auto_var: tk.BooleanVar | None = None,
                 anchor_diameter_var: tk.StringVar | None = None,
                 anchor_var: tk.BooleanVar | None = None,
                 on_change=None) -> None:
        self.surface = surface
        self.methods = [str(m) for m in methods]
        self.presets = [str(p) for p in presets]
        #: 下拉显示 ↔ 真正下传的预设名。**从预设表推**（`preset_options`），
        #: 不手写"红=red_ball"这类字典 —— 那会是预设表的第四份副本。
        self._preset_by_label = preset_options(self.presets)
        self._label_by_preset = {v: k for k, v in self._preset_by_label.items()}
        self.var_method = method_var if method_var is not None else tk.StringVar()
        self.var_preset = (preset_var if preset_var is not None
                           else tk.StringVar(value=PRESET_NONE))
        #: 下拉里显示的那一串（中文）。外部变量 `var_preset` 始终存**内部名**，
        #: 因为它是写进 `gui_state.json` 的东西：标签的推导规则一改，
        #: 存标签就等于存了一份会过期的值。
        self._var_preset_label = tk.StringVar()
        self.var_kind = tk.StringVar()
        self.var_diameter = diameter_var
        self.radius_vars = radius_vars
        self.radius_auto_var = radius_auto_var
        # 双目专用：物理球直径（mm，接 --anchor-diameter-mm）与半径锚点门开关。
        # 独立存，绝不与单目 diameter_var（--ball-diameter，单位米）串。
        self.anchor_diameter_var = anchor_diameter_var
        self.anchor_var = anchor_var
        self._anchor_diameter_entry = None
        self._on_change = on_change
        self._busy = False                  # 防止"改值 → 回调 → 再改值"打转
        self._kind = (kind if kind in BALL_KINDS
                      else kind_default_for(self.var_method.get()))
        # 只给了球种类、没给方法（工作台/双目那种"从零开始"的用法）：
        # 把方法补成这一档的定义值。否则 `kind()` 会反推成「自定义」——
        # 界面显示"纯色塑料球"、实际没有任何方法，两边打架。
        picked = method_for_kind(self._kind)
        if picked and not str(self.var_method.get() or "").strip():
            self.var_method.set(picked)
        # 纯色球从零开始时默认「自动取色」：不点选、也不知道球色，auto 会从画面
        # 主色估区间 —— 这比默认 white_ball（红球必然 0 检出）安全。
        # 只改**新建**控件的默认：调用方传了 preset_var（保存过的偏好）就一律不动，
        # 用户明确选过「不使用」也要保留。
        if (preset_var is None and self._kind == BALL_PLASTIC
                and str(self.var_preset.get() or "").strip() in ("", PRESET_NONE)
                and AUTO_PRESET in self.presets):
            self.var_preset.set(AUTO_PRESET)
        # 预设值可能来自旧配置文件：认不出的名字当"不用"（别把不存在的预设
        # 带进命令行 —— --preset 有 choices，会直接 argparse 退出）
        if str(self.var_preset.get() or "").strip() not in [PRESET_NONE, *self.presets]:
            self.var_preset.set(PRESET_NONE)

        row = ttk.Frame(parent)
        row.grid(row=surface._next_row(parent), column=0, columnspan=9,
                 sticky="w", pady=(0, 0))
        ttk.Label(row, text="球种类").pack(side="left")
        self.cb_kind = ttk.Combobox(row, values=list(BALL_KINDS), width=10,
                                    state="readonly", textvariable=self.var_kind)
        self.cb_kind.pack(side="left", padx=(6, 12))
        self.cb_kind.bind("<<ComboboxSelected>>", self._on_kind_selected)
        ttk.Label(row, text="检测方法").pack(side="left")
        self.cb_method = ttk.Combobox(row, values=self.methods, width=9,
                                      state="readonly",
                                      textvariable=self.var_method)
        self.cb_method.pack(side="left", padx=(6, 12))
        ttk.Label(row, text="颜色预设").pack(side="left")
        self.cb_preset = ttk.Combobox(
            row, values=[PRESET_NONE, *self._preset_by_label],
            width=11, state="readonly", textvariable=self._var_preset_label)
        self.cb_preset.pack(side="left", padx=(6, 0))
        self.cb_preset.bind("<<ComboboxSelected>>", self._on_preset_selected)
        if self.var_diameter is not None:
            ttk.Label(row, text=diameter_label).pack(side="left", padx=(14, 0))
            ent = ttk.Entry(row, textvariable=self.var_diameter,
                            width=diameter_width)
            ent.pack(side="left", padx=(6, 0))
            if callable(on_diameter_key):
                # 与旧控件同一条绑定：改数字时误差预算要跟着重算
                ent.bind("<KeyRelease>", lambda _e: on_diameter_key())
        if self.radius_vars is not None:
            # 2026-11-14：像素门限只是「筛候选」的进阶参数，不是首屏该问用户的事；
            # 物理球直径（mm）也只在勾了半径锚点门时才有意义。两者连同说明一起
            # 收进「高级」折叠区，并**嵌进上面这一行的末尾、不新增 grid 行** ——
            # ④ 卡片首屏已贴到窗口下沿（selftest_window_mode 逐卡量下沿，实测
            # 797/797、866/866），新增一行就会把按钮裁掉。
            from apps._collapsible import CollapsibleSection
            adv = CollapsibleSection(
                row, "高级：半径与直径", expanded=False,
                summary="像素门限留空=自动阶梯")
            # 收起态的标题按钮不能比同一行的 Entry/Combobox 高：④ 卡片首屏已
            # 贴到视口下沿，row 多几 px 就把按钮裁掉（selftest_window_mode 实测
            # angled ④ 会差 +7 px）。给这一个按钮单独配紧凑 padding，不碰全局
            # TButton 样式，也不改动首屏任何一行的 pady。
            _adv_style = ttk.Style()
            _adv_style.configure("BallAdv.TButton", padding=(1, 0))
            adv.header_button.configure(style="BallAdv.TButton")
            adv.pack(side="left", padx=(10, 0))
            inner = adv.content
            lo, hi = self.radius_vars
            px_line = ttk.Frame(inner)
            px_line.pack(anchor="w", pady=(4, 2))
            ttk.Label(px_line, text="像素半径门限（留空=自动阶梯）").pack(side="left")
            ttk.Entry(px_line, textvariable=lo, width=5).pack(side="left", padx=(6, 2))
            ttk.Label(px_line, text="–").pack(side="left")
            ttk.Entry(px_line, textvariable=hi, width=5).pack(side="left", padx=(2, 6))
            # 「尺寸门自动阶梯」勾选框已移除（2026-11-14 用户裁决）：阶梯是默认
            # 且唯一的路径，一个能把它关掉的勾选框反而违背意图；GUI 恒走阶梯
            # （radius_auto() 恒 True），--no-radius-auto 的 CLI 能力在
            # _actions.offline_stereo 里保留给命令行用户。
            if self.anchor_diameter_var is not None:
                mm_line = ttk.Frame(inner)
                mm_line.pack(anchor="w", pady=(2, 2))
                ttk.Label(mm_line, text="球直径（mm）").pack(side="left")
                ent = ttk.Entry(mm_line, textvariable=self.anchor_diameter_var,
                                width=8)
                ent.pack(side="left", padx=(6, 8))
                ttk.Label(mm_line,
                          text="用于按物理尺寸校验球的视半径；不勾锚点门不生效",
                          style="Hint.TLabel").pack(side="left")
                self._anchor_diameter_entry = ent
                if self.anchor_var is not None:
                    def _sync_anchor_diameter(*_args):
                        ent.configure(state=("normal"
                                             if bool(self.anchor_var.get())
                                             else "disabled"))
                    self.anchor_var.trace_add("write", _sync_anchor_diameter)
                    _sync_anchor_diameter()
            # 三句短说明常驻在折叠区里（不只进状态条）：把三类尺寸分开说清，
            # 免得「点球边的像素圆半径 / 像素门限 / 物理球直径」被混成一回事。
            for note in (
                "半径 = 点选球边得到的像素圆半径（seeds.json），和下面两个门限都不是一回事。",
                "半径锚定门：用左右点选球心 + 物理球直径校验视半径，须先勾「半径锚点门」且点过球心。",
                "尺寸门自动阶梯：按 4 档上限逐档试探找最大候选；留空像素门限就走它，物理直径不控制尺寸门。",
            ):
                ttk.Label(inner, text=note, style="Hint.TLabel",
                          wraplength=PARAM_HINT_WRAP).pack(anchor="w")
        # ⚠ **不再单起一行放说明**（2026-11-03 实测）：三个窗口的首屏高度都已
        # 顶到预算边缘（双目第④步装下时恰好 869 px），多加一行说明就整段出屏 ——
        # 而"控件被裁到窗口下沿"在本项目是比"少一句解释"严重得多的故障
        # （`tests/selftest_window_mode.py` 直接盯着每步的下沿）。
        # 说明走两条路：每档该注意什么写进卡片的说明文字（常驻），
        # 切档那一刻的提醒由调用方通过 ``on_change`` 写进状态条（见 ``hint()``）。

        # 方法被程序改掉（例如单目有了种子强制 specular）时，显示要跟着走
        self.var_method.trace_add("write", self._sync)
        # 预设：内部名 ↔ 下拉标签双向同步。外部仍只认 `var_preset`（内部名），
        # 所以旧配置文件里的 `red_ball` 照样能认；界面上一行中文。
        self._var_preset_label.set(self._label_by_preset.get(
            str(self.var_preset.get() or "").strip(), PRESET_NONE))
        self.var_preset.trace_add("write", self._sync_preset_label)
        self._sync(first=True)

    # ---------- 取值 ----------
    @staticmethod
    def resolve(kind: str, method: str = "",
                preset: str = PRESET_NONE) -> tuple[str, str]:
        """界面上的三个值 → 真要下传的 (``--method``, ``--preset``)。

        空串 = **这项不传**。纯函数，所以判据可以在没有显示器的机器上断言
        "选纯色球 + 红色 → 命令行是 color + red_ball"。

        两条规则写在**这一处**，三个窗口不用各写一遍：

        - 镜面球不传 ``--preset``：那个参数是颜色分割的 HSV 预设，
          对高光定位没有意义，传了只是让命令行看起来像"选了颜色"；
        - 纯色球**强制** ``color``：这一档的定义就是颜色分割，
          留着别的检测方法等于让用户选的档位不生效。
        """
        k = str(kind or "").strip()
        if k == BALL_METAL:
            return METHOD_METAL, ""
        if k == BALL_PLASTIC:
            return METHOD_PLASTIC, _preset_value(preset)
        m = str(method or "").strip()
        # 自定义档也可能落到 specular 上（它在注册表里是可选项）：一样不传预设。
        return m, ("" if m == METHOD_METAL else _preset_value(preset))

    def kind(self) -> str:
        """当前显示的球种类 —— **由方法反推**，因此永远与实际下传的一致。"""
        return kind_for_method(self.var_method.get(), self._kind)

    def method(self) -> str:
        return str(self.var_method.get() or "").strip()

    def preset(self) -> str:
        return BallSettings.resolve(self.kind(), self.method(),
                                    self.var_preset.get())[1]

    def min_radius(self) -> str:
        return str(self.radius_vars[0].get() or "").strip() if self.radius_vars else ""

    def max_radius(self) -> str:
        return str(self.radius_vars[1].get() or "").strip() if self.radius_vars else ""

    def radius_auto(self) -> bool:
        """尺寸门自动阶梯**恒开**（2026-11-14 用户裁决）。

        GUI 不再提供「取消勾选」入口 —— 用户明确要求阶梯是默认且唯一的路径，
        一个能把它关掉的勾选框反而违背意图。``--no-radius-auto`` 的 CLI 能力
        保留在 ``_actions.offline_stereo``（命令行用户显式给 ``radius_auto=False``
        仍可关），界面只是不再下传这个开关。
        """
        return True

    def problem(self) -> str:
        """这个区块里"填错了"的地方，一句话；空串 = 没问题。

        只说**会被怎么处理**，不静默：留空是合法的（用脚本默认），
        填了负数/乱码则说明"这一项不会被采用" —— 两者在界面上必须看得出区别。
        """
        if self.radius_vars is None:
            return ""
        lo, hi = self.min_radius(), self.max_radius()
        bad: list[str] = []
        values: dict[str, float] = {}
        for label, text in (("最小半径", lo), ("最大半径", hi)):
            if not text:
                continue
            try:
                num = float(text)
            except ValueError:
                bad.append(f"{label}「{text}」不是数字，这一项不传给命令行")
                continue
            if num <= 0:
                bad.append(f"{label}要填正数，{text} 不会被采用")
                continue
            values[label] = num
        if len(values) == 2 and values["最小半径"] > values["最大半径"]:
            bad.append(f"最小半径 {lo} 大于最大半径 {hi}，这个范围什么都选不出来")
        return "；".join(bad)

    def hint(self) -> str:
        """这一档该注意什么 —— 调用方拿它写状态条（切档那一刻要说出来）。

        两段拼起来：球种类的前提（``BALL_HINTS``）+ 方法缺可选依赖的提醒
        （``yolo`` / ``yolo_sam`` 没装时，界面选得动、跑起来才炸）。
        """
        parts = [BALL_HINTS.get(self.kind(), ""), detector_needs_note(self.method())]
        if self.radius_vars is not None:
            # 三类尺寸各一句短说明（切档那一刻写进状态条；常驻的完整版在操作
            # 说明「尺寸与半径」分区，控件旁另有一句短标签）：
            #   ① 半径 = 点球边得到的像素圆半径（seeds.json.radius）；
            #   ② 半径锚定门 = 球直径 + 左右点选球心校视半径，须先勾锚点门；
            #   ③ 尺寸门自动阶梯 = 留空像素门限就走 4 档试探，GUI 恒开。
            # 三者互不换算：物理球直径**不控制尺寸门**。
            parts.append(
                "半径 = 点选球边得到的像素圆半径（seeds.json），不是门限；"
                "「球直径（mm）」只喂半径锚定门校验视半径、须先勾锚点门且点过球心，"
                "不控制尺寸门；像素半径门限留空就走尺寸门自动阶梯（4 档试探），"
                "和球的毫米直径没有换算关系")
        return "；".join(p for p in parts if p)

    # ---------- 联动 ----------
    def set_kind(self, kind: str) -> None:
        """程序化地改这一档（例如单目发现会话里有 seed.json → 就是镜面球）。"""
        if kind not in BALL_KINDS:
            return
        self._kind = kind
        self.var_kind.set(kind)
        picked = method_for_kind(kind)
        if picked and self.var_method.get() != picked:
            self.var_method.set(picked)     # 只动"方法"，别的输入一律不碰
        self._sync()

    def _on_kind_selected(self, _evt=None) -> None:
        # ``_sync`` 里会回调 ``on_change``（窗口拿去刷状态条），这里不重复调
        self.set_kind(self.var_kind.get())

    def _on_preset_selected(self, _evt=None) -> None:
        """下拉里选的是**标签**，写回变量的是**内部名**（argv 只认内部名）。"""
        label = self._var_preset_label.get()
        self.var_preset.set(self._preset_by_label.get(label, PRESET_NONE))
        if callable(self._on_change):
            self._on_change()

    def _sync_preset_label(self, *_a) -> None:
        """内部名变了（程序改的 / 读配置读出来的）→ 下拉跟着显示对应中文。"""
        if self._busy:
            return
        name = str(self.var_preset.get() or "").strip()
        label = self._label_by_preset.get(name, PRESET_NONE)
        if label != self._var_preset_label.get():
            self._busy = True
            try:
                self._var_preset_label.set(label)
            finally:
                self._busy = False

    def _sync(self, *_a, first: bool = False) -> None:
        """把"已选的那一档"回写到控件上（**只改显示**）。"""
        if self._busy:
            return
        self._busy = True
        try:
            kind = self.kind()
            self.var_kind.set(kind)
            # 金属/纯色这两档的方法是被定义的（specular / color），下拉置灰：
            # 能改的地方只有「自定义」——否则"档位"与"方法"可以互相矛盾，
            # 而命令行只认方法，用户以为选的是档位。
            self.cb_method.configure(
                state="disabled" if kind in (BALL_METAL, BALL_PLASTIC) else "readonly")
            self.cb_preset.configure(
                state="disabled" if kind == BALL_METAL else "readonly")
            # 镜面球不传 --preset（`resolve` 已经如此），但下拉**显示**不能还停在
            # "红球"：用户会以为红色正在生效。显示成「不使用」，底层值保留 ——
            # 切回纯色球时还能恢复他原来选的颜色（`_label_by_preset` 查得到）。
            if kind == BALL_METAL:
                self._var_preset_label.set(PRESET_NONE)
            else:
                name = str(self.var_preset.get() or "").strip()
                self._var_preset_label.set(
                    self._label_by_preset.get(name, PRESET_NONE))
        finally:
            self._busy = False
        if not first and callable(self._on_change):
            self._on_change()


def _preset_value(preset) -> str:
    """界面上的预设值 → 命令行值（``不使用`` 与空白都变成空串）。"""
    text = str(preset or "").strip()
    return "" if text in ("", PRESET_NONE) else text


class ScrollPage(ttk.Frame):
    """一屏装不下时的滚动页（Canvas + 内嵌 Frame）。

    为什么还要它：分窗之后每个模式窗的"一步"仍可能有 400–600 px，屏幕矮的
    机器（900×600）一屏放不下。**但预算仍是硬的**（首屏输入 ≤8、按钮 ≤5）：
    滚动是兜底，不是把内容继续往上堆的许可。
    """

    def __init__(self, master: tk.Misc, surface: Surface) -> None:
        super().__init__(master)
        self.surface = surface
        self._canvas = tk.Canvas(self, bg=BG, highlightthickness=0, bd=0)
        bar = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=bar.set)
        self._canvas.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        self.content = ttk.Frame(self._canvas)
        self._win = self._canvas.create_window((0, 0), window=self.content,
                                               anchor="nw")
        self.content.bind("<Configure>", self._on_content)
        self._canvas.bind("<Configure>", self._on_canvas)
        # 滚轮：进入页面就绑，离开不解除（页面是长驻的），鼠标移出窗口无副作用
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self._canvas.bind_all(seq, self._on_wheel, add="+")

    def _on_content(self, _evt=None) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas(self, evt) -> None:
        self._canvas.itemconfigure(self._win, width=evt.width)
        self.surface.reflow(evt.width)

    def _on_wheel(self, evt) -> None:
        # 只在指针确实位于本页面之上时滚动，避免多窗口互相抢滚轮
        widget = self.winfo_containing(evt.x_root, evt.y_root)
        while widget is not None:
            if widget is self:
                break
            widget = getattr(widget, "master", None)
        else:
            return
        if getattr(evt, "num", None) == 4:
            self._canvas.yview_scroll(-1, "units")
        elif getattr(evt, "num", None) == 5:
            self._canvas.yview_scroll(1, "units")
        else:
            self._canvas.yview_scroll(int(-evt.delta / 120), "units")


class StatusBar(ttk.Frame):
    """窗口底部常驻状态条：一句话说人话 + 查看日志 + 中断。

    为什么每个窗口都要有它（计划书 §7 的风险清单）：分窗之后日志不再"在页面
    最底部"，某个窗口卡住时用户会以为程序死了。这里永远显示最近一条任务的状态，
    并给出**下一步该做什么**，而不是只报告"失败"。
    """

    def __init__(self, master: tk.Misc, *, on_log=None, on_stop=None,
                 help_section: str | None = None) -> None:
        super().__init__(master)
        self._label = ttk.Label(self, text="就绪", style="Hint.TLabel")
        # ⚠️ 先 pack 右侧按钮、最后 pack 左侧状态文本。
        # pack 是"先到先得"：文本先 pack 时，一条长状态（例如"正在按 parallel 解算
        # 20261024-173619 …"）会按自己的请求宽度占满，右侧三个按钮就被挤出窗口
        # （2026-11-06 走查截图 46：任务日志/中断/操作说明全被切）。
        # 反过来先给按钮固定位置，文本拿剩下的空间（太长就裁文本，不裁按钮）。
        # 「? 操作说明」放状态条：**每个窗口都有、位置固定**。散落在各页面里
        # 的按钮用户找不到，而说明又刚把长段落从界面搬走（2026-11-05）。
        if help_section is not None:
            def _jump() -> None:
                from apps.help_window import open_help
                open_help(self, section=help_section)

            # 右侧三个按钮在 880 px 的单目窗里会顶到边缘：把间距从 12 收到 4，
            # 省 24 px，最右的「? 操作说明」就不会被切掉一截。
            ttk.Button(self, text="? 操作说明", command=_jump).pack(
                side="right", padx=(0, 4))
        if on_stop is not None:
            ttk.Button(self, text="中断", command=on_stop).pack(side="right",
                                                                padx=(0, 4))
        if on_log is not None:
            # 文案用完整名字：用户明确要求「查看日志」改成这个。
            # 它在每个窗底部常驻，是进入任务表与日志的唯一入口。
            ttk.Button(self, text="任务管理器和运行日志", command=on_log).pack(
                side="right", padx=(0, 4))
        # 文本最后 pack，拿剩余空间；`anchor="w"` 保证从左边开始显示。
        self._label.pack(side="left", fill="x", expand=True,
                         padx=(16, 8), pady=6, anchor="w")

    def set_status(self, text: str, level: str = "info") -> None:
        """level ∈ info/ok/warn —— 只换颜色，话术由调用方保证"说人话"。"""
        style = {"ok": "Ok.TLabel", "warn": "Warn.TLabel"}.get(
            level, "Hint.TLabel")
        try:
            self._label.configure(text=text, style=style)
            # 状态文本由调用方用 tf() 按**当前语言**现场生成，不应被原文记账
            # 当成静态控件文案：清掉记账，避免切语言时 apply() 拿上一语言的
            # 显示值当原文、把刚生成的英文又翻回中文（2026-11-23 实测）。
            i18n.forget(self._label)
        except tk.TclError:
            pass


def fit_to_content(win: tk.Misc, page: ScrollPage, status: StatusBar, *,
                   width: int, screen_h: int, extra: int = 0) -> int:
    """把窗口高度调成"内容 + 状态条（+ 滚动页之外的固定高度）"，封顶为屏幕可用高度。

    为什么在**建完之后**量、而不是建之前算：内容高度由 ``wraplength`` 后的
    实际折行、按钮内边距、ttk 主题的行高共同决定，手算必然漂。所以先按内容
    建好（此时窗口还没显示，量到的 ``winfo_reqheight`` 就是最终值），再一次性
    把尺寸交出去 —— 用户不会看到窗口先开后缩的中间态。

    ``extra`` 是滚动页**之外**的固定高度（门厅顶部那条品牌栏就是这种：
    它不在 ScrollPage 里，但同样占客户区）。漏掉它，窗口就永远矮这么多，
    表现为"最下面一截要滚 8 px"——门厅正是这么差了 8 px。

    调用方必须在**所有卡片建完、``_reload``/``_refresh`` 之后**调它：
    状态行、原因行的文字会随数据变，量早了就是量一个还没写完的界面。

    装不下（返回高度 - 状态条 < 内容）时不报错、也不裁：``ScrollPage`` 本来
    就是为此存在的（窄屏/小屏的兜底）。这条路径不该被当成正常态 ——
    ``tests/selftest_window_mode.py`` 里有一条硬判据盯着"每一步都在视口里"。
    """
    win.update_idletasks()
    content = page.content.winfo_reqheight()
    height = fit_height(content, screen_h=screen_h,
                        status=status.winfo_reqheight() + int(extra))
    win.geometry(f"{int(width)}x{int(height)}")
    return int(height)
