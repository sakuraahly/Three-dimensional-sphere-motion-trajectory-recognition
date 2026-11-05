"""操作说明窗口：左侧分类树、右侧正文、顶部搜索。

为什么要有它（2026-11-05 用户）：

"目前程序缺少说明书，在主界面做个操作说明的入口界面，然后按照不同分区和问题
给出操作说明，同时要有搜索功能。这样很多界面的操作说明文字就可以删掉进而
不会影响界面排布了。"

也就是说它不只是"文档"，还是**给界面腾地方的手段**：那些因为放不下而被
截断、或挤占控件位置的说明文字，应该搬到这儿来。

设计取舍：
- 用 `ttk.Treeview` 做目录而不是自绘列表：分区/条目两层结构、键盘上下键、
  展开收起都是现成的；
- 正文用只读 `tk.Text` 而不是 `Label`：说明有分段和缩进，Label 的
  `wraplength` 换行算不准（那正是别处说明被截断的原因）；
- 搜索**实时**过滤（`trace_add`），不按回车也生效 —— 说明总条目只有几十条，
  没必要让用户多点一次。
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from apps.help_content import SECTIONS, iter_items, search

try:                                    # 与主界面同一套字体，缺了也能跑
    from apps._gui_widgets import FONT_FAMILY
except Exception:                       # noqa: BLE001
    FONT_FAMILY = "Microsoft YaHei UI"

#: 正文区字号。比界面常规大一号：说明是要"读"的，不是扫一眼的按钮标签。
BODY_FONT = (FONT_FAMILY, 10)
TITLE_FONT = (FONT_FAMILY, 12, "bold")
MUTED = "#64748b"


class HelpWindow(tk.Toplevel):
    """操作说明窗口。

    ``section`` 可以直接定位到某个分区（其它窗口的「?」按钮会传自己的 id，
    比如 3D 查看器传 ``"viewer"``），省得用户自己在一堆条目里找。
    """

    def __init__(self, master, section: str = "") -> None:
        super().__init__(master)
        self.title("操作说明")
        self.geometry("980x660")
        self.minsize(700, 460)
        # 非模态、**不是 transient**：transient 在 Windows 上会把窗口降成
        # 工具窗口，标题栏的最大化按钮直接灰掉。说明窗口经常要被摆到旁边
        # 一边看一边操作，它得能最大化、能最小化到任务栏、能单独 Alt+Tab
        # （2026-11-05 用户："可以全屏，关闭最小化"）。
        self.resizable(True, True)

        self._items: dict[str, tuple[str, str, str]] = {}   # iid -> (分区, 标题, 正文)
        self._query = tk.StringVar(master=self, value="")

        self._build()
        self._fill_tree()
        if section:
            self._goto_section(section)
        # Esc：全屏时先退出全屏，否则才关闭窗口 —— 直接 destroy 会让
        # "想退出全屏按 Esc" 变成"把说明窗口关了"。
        self.bind("<Escape>", self._on_escape)
        self.bind("<F11>", self._toggle_fullscreen)
        # 不绑 <Double-1>：Tk 的 bindtags 会把**子控件**的双击冒泡到 Toplevel，
        # 双击目录条目会莫名其妙全屏。标题栏双击由系统处理，不需要我们插手。

    def _on_escape(self, _event=None) -> None:
        if getattr(self, "_full", False):
            self._toggle_fullscreen()
        else:
            self.destroy()

    def _toggle_fullscreen(self, _event=None) -> None:
        """F11 在"全屏"和"窗口"之间切换。"""
        self._full = not getattr(self, "_full", False)
        try:
            self.attributes("-fullscreen", self._full)
        except tk.TclError:                       # 个别窗口管理器不认
            try:
                self.state("zoomed" if self._full else "normal")
            except tk.TclError:
                pass

    # ------------------------------------------------------------------
    def _build(self) -> None:
        top = ttk.Frame(self, padding=(12, 10, 12, 6))
        top.pack(fill="x")

        ttk.Label(top, text="搜索").pack(side="left")
        self.search_entry = ttk.Entry(top, textvariable=self._query, width=34)
        self.search_entry.pack(side="left", padx=(6, 6))
        # 实时过滤：几十条说明，边打边筛比"输完按回车"顺手
        self._query.trace_add("write", lambda *_: self._on_search())
        self.search_entry.bind("<Escape>", self._clear_search)

        ttk.Button(top, text="清除", width=6,
                   command=self._clear_search).pack(side="left")
        ttk.Label(top, text="多个词用空格分开（同时包含）",
                  foreground=MUTED).pack(side="left", padx=(10, 0))

        self.count_lab = ttk.Label(top, text="", foreground=MUTED)
        self.count_lab.pack(side="right")

        body = ttk.PanedWindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=12, pady=(0, 10))

        # ---- 左：目录 ----
        left = ttk.Frame(body)
        # 条目名最长的一条 22 个汉字（"曲线跑到坐标框外面 / 背景不动只有曲线
        # 在动"），10pt 要 300px，默认 200px 的列宽会把它砍掉半句 —— 目录看不全
        # 等于没有目录。9pt + 260px 刚好放下。
        # 注意 `ttk.Treeview` **没有 font 参数**（传了会 TclError），字体只能
        # 通过 Style 设置。
        _style = ttk.Style(self)
        _style.configure("Help.Treeview", font=(FONT_FAMILY, 9))
        self.tree = ttk.Treeview(left, show="tree", selectmode="browse",
                                 style="Help.Treeview")
        self.tree.column("#0", width=260, minwidth=190)
        sb = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        body.add(left, weight=1)

        # ---- 右：正文 ----
        right = ttk.Frame(body)
        self.text = tk.Text(right, wrap="word", padx=14, pady=12,
                            font=BODY_FONT, relief="flat", background="#ffffff",
                            spacing1=2, spacing3=6, cursor="arrow")
        tsb = ttk.Scrollbar(right, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=tsb.set)
        self.text.pack(side="left", fill="both", expand=True)
        tsb.pack(side="right", fill="y")
        # 只读，但要能选中复制 —— 用 disabled 之后仍可选中，只是不能编辑
        self.text.configure(state="disabled")
        self.text.tag_configure("title", font=TITLE_FONT, spacing3=10)
        self.text.tag_configure("section", font=(FONT_FAMILY, 9),
                                foreground=MUTED, spacing3=4)
        self.text.tag_configure("body", font=BODY_FONT)
        body.add(right, weight=3)

        self.search_entry.focus_set()

    # ------------------------------------------------------------------
    def _fill_tree(self, hits: list[tuple[str, str, str, str]] | None = None) -> None:
        """重建左侧目录。``hits`` 给了就是搜索结果（扁平），否则是全量分区树。"""
        self.tree.delete(*self.tree.get_children())
        self._items.clear()

        if hits is not None:
            for i, (sid, stitle, title, body) in enumerate(hits):
                iid = f"r{i}"
                # 搜索结果扁平静止：不分组，一眼扫完
                self.tree.insert("", "end", iid=iid, text=f"{title}",
                                 open=False)
                self._items[iid] = (f"{stitle}", title, body)
            self.count_lab.configure(text=f"搜到 {len(hits)} 条")
            if hits:
                self.tree.selection_set("r0")
                self._on_select()          # 直接渲染，见下
            else:
                self._show("没有匹配的说明", "换个词试试；也可以清空搜索看全部。", "")
            return

        n = 0
        for sid, stitle, items in SECTIONS:
            parent = self.tree.insert("", "end", iid=f"s:{sid}",
                                      text=stitle, open=True)
            for j, (title, body) in enumerate(items):
                iid = f"i:{sid}:{j}"
                self.tree.insert(parent, "end", iid=iid, text=title)
                self._items[iid] = (stitle, title, body)
                n += 1
        self.count_lab.configure(text=f"共 {n} 条说明")
        first = self.tree.get_children()
        if first:
            kids = self.tree.get_children(first[0])
            if kids:
                self.tree.selection_set(kids[0])
                # 选中不会立刻派发 `<<TreeviewSelect>>`（要等事件循环），
                # 所以窗口刚打开时右侧是空白的 —— 用户第一眼看到"点了没反应"。
                # 直接调一次渲染，别依赖事件时序。
                self._on_select()

    def _on_search(self) -> None:
        q = self._query.get().strip()
        if not q:
            self._fill_tree()
            return
        self._fill_tree(search(q))

    def _clear_search(self, _event=None) -> None:
        self._query.set("")          # trace 会触发 _on_search -> 重建全量
        self.search_entry.focus_set()

    def _on_select(self, _event=None) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        meta = self._items.get(sel[0])
        if meta is None:             # 选中的是分区节点
            return
        stitle, title, body = meta
        self._show(title, body, stitle)

    def _goto_section(self, section: str) -> None:
        """定位到某个分区的第一条（其它窗口的「?」按自己的 id 跳过来）。"""
        for i, (sid, stitle, items) in enumerate(SECTIONS):
            if sid != section:
                continue
            parent = f"s:{sid}"
            if self.tree.exists(parent):
                kids = self.tree.get_children(parent)
                if kids:
                    self.tree.selection_set(kids[0])
                    self.tree.see(kids[0])
            return

    def _show(self, title: str, body: str, section: str) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        if section:
            self.text.insert("end", section + "\n", "section")
        self.text.insert("end", title + "\n", "title")
        self.text.insert("end", body, "body")
        self.text.configure(state="disabled")
        self.text.yview_moveto(0.0)


def open_help(master, section: str = "") -> HelpWindow:
    """打开（或聚焦已有的）说明窗口。

    复用一个窗口而不是每次新建：用户经常「看一句 → 回主界面做一步 → 再看」，
    每次都弹新窗会堆一屏。
    """
    # 兼容"传进来的不是 widget"：`Viewer3D` 这类是普通类，只有 `.root`。
    # 直接 `master.winfo_toplevel()` 会 AttributeError，而且**在 tkinter 回调里
    # 不会弹窗**，用户只会看到"按钮点了没反应"（2026-11-05 踩过）。
    base = master if hasattr(master, "winfo_toplevel") else getattr(master, "root", None)
    if base is None or not hasattr(base, "winfo_toplevel"):
        raise TypeError(f"open_help 需要一个 Tk/Toplevel 或带 .root 的对象：{master!r}")
    top = base.winfo_toplevel()
    existing = getattr(top, "_help_window", None)
    if existing is not None and existing.winfo_exists():
        existing.deiconify()
        existing.lift()
        if section:
            existing._goto_section(section)
        return existing
    win = HelpWindow(top, section=section)
    try:
        top._help_window = win
    except Exception:                   # noqa: BLE001
        pass
    return win
