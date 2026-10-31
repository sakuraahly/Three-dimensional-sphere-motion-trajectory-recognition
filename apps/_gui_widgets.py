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

from apps._gui_geometry import fit_height

__all__ = [
    "BG", "ACCENT", "INK", "MUTED", "WARN", "OK", "FONT_FAMILY",
    "CARD_HINT_WRAP", "PARAM_HINT_WRAP", "INLINE_NOTE_WRAP",
    "init_style", "Surface", "ScrollPage", "StatusBar", "fit_to_content",
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
             pady: tuple[int, int] = (10, 0), fill: str = "x",
             expand: bool = False) -> ttk.Frame:
        """一张卡片。返回的 Frame 是内部 grid 的父容器。

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

    def __init__(self, master: tk.Misc, *, on_log=None, on_stop=None) -> None:
        super().__init__(master)
        self._label = ttk.Label(self, text="就绪", style="Hint.TLabel")
        self._label.pack(side="left", padx=(16, 8), pady=6)
        if on_stop is not None:
            ttk.Button(self, text="中断", command=on_stop).pack(side="right",
                                                                padx=(0, 12))
        if on_log is not None:
            # 文案用完整名字：用户明确要求「查看日志」改成这个。
            # 它在每个窗底部常驻，是进入任务表与日志的唯一入口。
            ttk.Button(self, text="任务管理器和运行日志", command=on_log).pack(
                side="right", padx=(0, 6))

    def set_status(self, text: str, level: str = "info") -> None:
        """level ∈ info/ok/warn —— 只换颜色，话术由调用方保证"说人话"。"""
        style = {"ok": "Ok.TLabel", "warn": "Warn.TLabel"}.get(
            level, "Hint.TLabel")
        try:
            self._label.configure(text=text, style=style)
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
