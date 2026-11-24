"""可折叠的设置区块（Tk/ttk），供各 GUI 设置面板复用。

本模块是**纯库**：import 时只定义类与常量，不会创建 ``Tk``、不会读命令行、
也不依赖本项目其它模块。直接运行本文件会执行内置自测（隐藏窗口）。

用法::

    from apps._collapsible import CollapsibleSection

    section = CollapsibleSection(
        parent,
        "相机参数",
        expanded=True,
        summary="2 项",
        on_toggle=lambda expanded: print("现在展开：", expanded),
    )
    section.pack(fill="x")
    ttk.Label(section.content, text="左相机").pack(anchor="w")

约定：

- 调用方只把控件放进 ``section.content``，**不要自己 pack/grid content**；
  展开/收起由本类管理，收起走 ``pack_forget()``，控件不会被销毁，输入值保留。
- 头部使用 ttk 默认 ``TButton`` 样式，没有额外的样式参数；需要统一换肤时，
  由 ``ttk.Style`` 配置 ``TButton``（或子类化后改 ``header_button``）。
- ``padding`` 只作用在 ``content`` 内部（ttk 的 padding 语义），可为整数或
  ``(水平, 垂直)`` 这类 Tk 屏幕距离；为 0 时不设置。
- 回调在状态、头部文字、内容显隐都更新完之后调用。回调抛出的异常会继续向上
  传播（不吞异常），但此时内部状态与界面已经一致，不会出现"显示收起了、
  ``expanded`` 却仍是 True"这类半更新状态。
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

__all__ = ["CollapsibleSection"]

_CHEVRON_EXPANDED = "\u25be"   # ▾
_CHEVRON_COLLAPSED = "\u25b8"  # ▸
_SUMMARY_SEPARATOR = "  \u00b7  "  # "  ·  "


class CollapsibleSection(ttk.Frame):
    """一个"标题按钮 + 可选摘要 + 可收起内容"的 ttk 区块。

    公开属性：
        header_button: 头部 ``ttk.Button``（可用键盘 Space/Return 切换）。
        content:       内容容器；调用方把设置控件放这里。
        expanded:      当前是否展开。

    公开方法：
        toggle():       在展开/收起之间切换。
        expand():       展开（已展开时无操作，不触发回调）。
        collapse():     收起（已收起时无操作，不触发回调）。
        set_summary():  更新摘要；无论当前是否收起都会刷新头部。
    """

    def __init__(
        self,
        parent,
        title: str,
        *,
        expanded: bool = True,
        on_toggle=None,
        summary: str = "",
        padding=0,
    ) -> None:
        super().__init__(parent)

        self._title = str(title)
        self._summary = "" if summary is None else str(summary)
        self._on_toggle = on_toggle
        self.expanded = bool(expanded)

        self.header_button = ttk.Button(
            self,
            text=self._header_text(),
            command=self.toggle,
            takefocus=True,
        )
        # 占满一整行；Window 上按钮默认高度也足以容纳触控目标。
        self.header_button.pack(side="top", fill="x")
        # 只在按钮上绑定 Return（不是 bind_all）。Space 由 ttk.Button 自带。
        self.header_button.bind("<Return>", self._on_return, add="+")

        # 给展开内容加一圈边框：否则用户分不清哪些控件属于这个折叠区。
        frame_kw = {"relief": "solid", "borderwidth": 1}
        if padding:
            frame_kw["padding"] = padding
        self.content = ttk.Frame(self, **frame_kw)

        if self.expanded:
            self.content.pack(side="top", fill="x")

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------
    def toggle(self) -> None:
        """切换展开/收起。"""
        self._set_expanded(not self.expanded)

    def expand(self) -> None:
        """展开。已经展开时不做事、不调用回调。"""
        self._set_expanded(True)

    def collapse(self) -> None:
        """收起。已经收起时不做事、不调用回调。"""
        self._set_expanded(False)

    def set_summary(self, text: str) -> None:
        """更新头部摘要，并立即刷新头部（收起状态下也会刷新）。"""
        self._summary = "" if text is None else str(text)
        self._refresh_header()

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------
    def _header_text(self) -> str:
        # 标题/摘要是界面文案：整句 f-string 拼出的 "▸ 高级：… · …" 命不中
        # 词典，逐块 t() 翻译、分隔符与展开箭头保留（2026-11-24 截图实测）。
        from apps import i18n                              # 延迟导入，避免循环
        chevron = _CHEVRON_EXPANDED if self.expanded else _CHEVRON_COLLAPSED
        title = i18n.t(self._title)
        summary = i18n.t(self._summary) if self._summary else ""
        text = f"{chevron} {title}"
        if summary:
            text += _SUMMARY_SEPARATOR + summary
        return text

    def _i18n_refresh(self) -> None:
        """切语言时重绘头部（apply() 的约定钩子）。"""
        from apps import i18n
        self._refresh_header()
        for child in self.winfo_children():
            i18n.apply(child)

    def _refresh_header(self) -> None:
        self.header_button.configure(text=self._header_text())

    def _set_expanded(self, expanded: bool) -> None:
        """把状态真正改到 ``expanded``；无变化则直接返回。

        先改状态、刷头部、显隐内容，最后才调回调 —— 即使回调抛异常，内部状态
        和界面也已经是同一个状态。回调不吞异常，交由调用方/Tk 的异常处理器处理。
        """
        expanded = bool(expanded)
        if expanded == self.expanded:
            return

        self.expanded = expanded
        self._refresh_header()
        if expanded:
            self.content.pack(side="top", fill="x")
        else:
            self.content.pack_forget()

        if self._on_toggle is not None:
            self._on_toggle(expanded)

    def _on_return(self, _event=None) -> str:
        """头部按钮的 Return 绑定。返回 ``break``，避免再有默认行为。"""
        self.toggle()
        return "break"


# ----------------------------------------------------------------------
# 自测：直接运行本文件时执行。导入本模块不会执行这里。
# ----------------------------------------------------------------------
def _run_selftest() -> int:
    root = tk.Tk()
    root.withdraw()  # 不弹窗；只测 Tk 的布局管理状态
    try:
        global_bindings_before = root.bind_all()

        calls_a: list[bool] = []
        section_a = CollapsibleSection(
            root,
            "甲",
            expanded=True,
            summary="两项",
            on_toggle=calls_a.append,
        )
        calls_b: list[bool] = []
        section_b = CollapsibleSection(
            root,
            "乙",
            expanded=False,
            on_toggle=calls_b.append,
        )
        section_a.pack(fill="x")
        section_b.pack(fill="x")
        root.update()

        # ---- 初始状态 / 头部 / 全宽 / 键盘绑定 ------------------------
        assert section_a.expanded is True
        assert section_a.content.winfo_manager() == "pack"
        assert section_a.header_button.cget("text") == "▾ 甲  ·  两项"
        assert section_a.header_button.pack_info().get("fill") == "x"
        assert section_b.expanded is False
        assert section_b.content.winfo_manager() == ""
        assert section_b.header_button.cget("text") == "▸ 乙"
        assert calls_a == [] and calls_b == []
        assert section_a.header_button.bind("<Return>")
        assert root.bind_all("<Return>") == ""
        assert root.bind_all() == global_bindings_before
        print("PASS 初始展开/收起、全宽头部、Return 绑定、无全局绑定")

        # ---- toggle 两次：状态、头部、内容、回调各更新一次 ------------
        section_a.toggle()
        assert section_a.expanded is False
        assert section_a.content.winfo_manager() == ""
        assert section_a.header_button.cget("text") == "▸ 甲  ·  两项"
        assert calls_a == [False]

        section_a.toggle()
        assert section_a.expanded is True
        assert section_a.content.winfo_manager() == "pack"
        assert section_a.header_button.cget("text") == "▾ 甲  ·  两项"
        assert calls_a == [False, True]
        print("PASS toggle 两次：状态、头部、内容显隐、回调各正确")

        # ---- 收起不销毁控件，Entry 的值保留 ---------------------------
        entry = ttk.Entry(section_a.content)
        entry.pack(fill="x")
        entry.insert(0, "camera-0")
        assert entry.get() == "camera-0"

        section_a.collapse()
        assert section_a.content.winfo_manager() == ""
        assert entry.get() == "camera-0"
        assert entry.winfo_exists()

        section_a.expand()
        assert section_a.content.winfo_manager() == "pack"
        assert entry.get() == "camera-0"
        assert calls_a == [False, True, False, True]
        print("PASS 收起/展开后 Entry 的值仍然保留")

        # ---- 幂等：没有状态变化时不调用回调 ---------------------------
        section_b.collapse()
        assert calls_b == []
        assert section_b.expanded is False

        section_b.expand()
        assert section_b.expanded is True
        assert section_b.content.winfo_manager() == "pack"
        assert calls_b == [True]

        section_b.expand()
        assert calls_b == [True]
        assert section_b.content.winfo_manager() == "pack"

        section_b.collapse()
        assert section_b.expanded is False
        assert section_b.content.winfo_manager() == ""
        assert calls_b == [True, False]

        section_b.collapse()
        assert calls_b == [True, False]
        print("PASS expand/collapse 幂等，且无变化时不调用回调")

        # ---- 收起状态下 set_summary 也刷新头部 ------------------------
        section_b.set_summary("三项")
        assert section_b.expanded is False
        assert section_b.header_button.cget("text") == "▸ 乙  ·  三项"
        assert calls_b == [True, False]
        print("PASS 收起状态下 set_summary 立即刷新头部")

        # ---- 回调抛异常：异常上抛，但状态和界面保持一致 ---------------
        def raising_callback(_expanded: bool) -> None:
            raise RuntimeError("selftest callback boom")

        section_c = CollapsibleSection(root, "丙", on_toggle=raising_callback)
        section_c.pack(fill="x")
        try:
            section_c.toggle()
        except RuntimeError as exc:
            assert str(exc) == "selftest callback boom"
        else:
            raise AssertionError("回调异常没有被上抛")

        assert section_c.expanded is False
        assert section_c.content.winfo_manager() == ""
        assert section_c.header_button.cget("text") == "▸ 丙"
        print("PASS 回调抛异常后状态与界面仍一致")

        print("PASS CollapsibleSection 自测全部通过")
        return 0
    finally:
        root.destroy()


if __name__ == "__main__":
    raise SystemExit(_run_selftest())
