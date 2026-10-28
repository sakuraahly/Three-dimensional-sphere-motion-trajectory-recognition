"""控制台窗口：任务表 + 运行日志（单例，任何窗口都能开）。

为什么单独一个文件（2026-10-27 计划书 §2.5）：门厅、单目窗、双目窗都要能打开它，
而它又是**唯一**能中断任务的窗口。放在 ``_gui_app`` 里会与 App 互相引用；
独立之后依赖是单向的：console → (widgets, task_runner)，不认识任何模式窗。

为什么"快测 / 全套回归 / 规范自查"也在这里（计划书 P4）：它们是开发者工具，
不属于用户的标定流程。摆在用户流程里会让人以为"跑标定之前要先点快测"。
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

from apps._gui_widgets import BG, Surface

if TYPE_CHECKING:                                   # 只为类型注解，不产生运行期依赖
    from apps._gui_app import App

__all__ = ["ConsoleWindow"]


# --------------------------------------------------------------------------- #
# 控制台：任务表 + 日志（单例，任何窗口都能开）
# --------------------------------------------------------------------------- #
class ConsoleWindow(tk.Toplevel):
    """任务表 + 日志。**唯一**能中断任务的窗口。

    为什么把"快测 / 全套回归 / 规范自查"也放这里（计划书 P4）：它们是开发者
    工具，不属于用户的标定流程。放在用户流程里会让人以为"跑标定要先点快测"。
    """

    def __init__(self, app: App) -> None:
        super().__init__(app)
        self.app = app
        self.configure(bg=BG)
        # 做大：任务表 5 行 + 日志 14 行在 560 高里挤成一条缝。
        # 屏幕逻辑 1600x1000 下 1100x780 放得下，且仍可拖拽缩放。
        self.geometry("1100x780")
        self.minsize(760, 520)
        self.surface = Surface(self)

        box = self.surface.card(
            self, "任务",
            "卡住时在这里中断：直接关窗口不会杀掉子进程，相机可能仍被占着。"
            "「中断」先试优雅退出，3 秒不退再强制结束整棵进程树。")
        # 表格要**双向**拉伸：只写 sticky="we" 时 frame 的高度按"内容请求高度"
        # 算，而 Treeview 的 height=5 与字体行高对不上时会溢出——按钮行就压在
        # 第 2 行上（2026-10-28 用户截图）。
        box.columnconfigure(0, weight=1)         # 横向铺满
        # ⚠ **必须走游标**：写死 row=1 时游标仍停在 1（card() 的 hint 占了 row=0
        # 之后只推到 1），于是紧随其后的 action_row 也拿到 row=1 ——
        # 按钮直接压在第 2 行上（2026-10-28 用户截图）。
        row_tree = self.surface._next_row(box)
        box.rowconfigure(row_tree, weight=1)     # 纵向：表格行吃掉多余空间
        frame = ttk.Frame(box)
        frame.grid(row=row_tree, column=0, columnspan=9, sticky="nsew",
                   pady=(4, 4))
        cols = ("id", "title", "pid", "elapsed", "status")
        heads = ("#", "任务", "PID", "已运行", "状态")
        # ⚠ 列宽单位是**像素**（`ttk.Treeview.column(width=)` 吃像素，不是字符数）。
        # 之前按字符数的直觉写成 (4, 34, 8, 9, 14)，结果"任务"列只有 34 px，
        # 标题被裁成"管理"两个字、行内文字还挤在一起（2026-10-28 用户截图）。
        widths = (46, 380, 84, 96, 96)
        self.tree = ttk.Treeview(frame, columns=cols, show="headings",
                                 height=6, selectmode="browse")
        for c, h, w in zip(cols, heads, widths):
            # 列头跟内容的对齐方式一致：标题/状态左对齐，其余居中
            self.tree.heading(c, text=h,
                              anchor="w" if c in ("title", "status") else "center")
            self.tree.column(c, width=w, minwidth=w,
                             # 只有标题列吸收额外宽度：窗口放大时该变宽的是
                             # "任务"（它最长），不是 PID / 已运行 这种短字段。
                             stretch=(c == "title"),
                             anchor="w" if c in ("title", "status") else "center")
        bar = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=bar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        row = self.surface.action_row(box)
        self.surface.primary(row, "中断选中", self._stop_selected)
        self.surface.secondary(row, "中断全部", self._stop_all)
        self.surface.secondary(row, "打开日志文件", self._open_log_file)

        # fill="both", expand=True：日志区必须随窗口长，否则窗口放大之后
        # 内容仍是左上角一块（2026-10-28 截图就是这个）。
        logbox = self.surface.card(self, "运行日志", "",
                                   fill="both", expand=True)
        self.text = tk.Text(logbox, height=14, wrap="none", state="disabled",
                            font=("Consolas", 9), bg="#f8fafc")
        self.text.grid(row=0, column=0, columnspan=9, sticky="nsew")
        logbox.rowconfigure(0, weight=1)
        logbox.columnconfigure(0, weight=1)     # 列也要权重，否则只长高不长宽
        bar2 = ttk.Scrollbar(logbox, orient="vertical", command=self.text.yview)
        bar2.grid(row=0, column=9, sticky="ns")
        self.text.configure(yscrollcommand=bar2.set)
        for line in app.recent_log():
            self._append(line)
        self._log_cb = self._append
        app.subscribe_log(self._log_cb)
        self.refresh()

    # ---------- 日志 ----------
    def _append(self, text: str) -> None:
        """追加一行；**只有本来就贴着底**才自动滚（用户往上翻时不许拽回来）。"""
        try:
            at_bottom = self.text.yview()[1] >= 0.999
            self.text.configure(state="normal")
            self.text.insert("end", text + "\n")
            if at_bottom:
                self.text.see("end")
            self.text.configure(state="disabled")
        except tk.TclError:
            pass

    # ---------- 任务表 ----------
    def refresh(self) -> None:
        try:
            sel = self._selected_tid()
            for item in self.tree.get_children():
                self.tree.delete(item)
            for t in self.app.tasks.tasks():
                if t.running:
                    state = "运行中"
                elif t.exit_code == 0:
                    state = "完成"
                else:
                    state = f"返回码 {t.exit_code}"
                if t.kill_note:
                    state += "（未停干净）"
                self.tree.insert("", "end", iid=str(t.tid),
                                 values=(t.tid, t.title, t.proc.pid,
                                         f"{t.elapsed():.1f}s", state))
            if sel is not None and self.tree.exists(str(sel)):
                self.tree.selection_set(str(sel))
        except tk.TclError:
            pass

    def _selected_tid(self) -> int | None:
        sel = self.tree.selection()
        return int(sel[0]) if sel else None

    def _on_select(self, _evt=None) -> None:
        tid = self._selected_tid()
        if tid is not None:
            self.app.tasks.select(tid)

    def _task_of_tid(self, tid: int | None):
        if tid is None:
            return None
        return next((t for t in self.app.tasks.tasks() if t.tid == tid), None)

    def _stop_selected(self) -> None:
        task = self._task_of_tid(self._selected_tid())
        if task is None:
            self._append("[提示] 先在表里选一个任务")
            return
        ok = self.app.tasks.stop(task, graceful=True)
        self._append(f"[任务 {task.tid}] " + ("已确认停掉" if ok
                                             else "未能确认停掉，请看上面的原因"))
        self.refresh()

    def _stop_all(self) -> None:
        ok = self.app.tasks.stop_all(graceful=True)
        self._append(f"[任务] 确认停掉 {ok} 个")
        self.refresh()

    def _open_log_file(self) -> None:
        task = self._task_of_tid(self._selected_tid())
        if task is None or task.log_path is None:
            self._append("[提示] 这个任务没有日志文件")
            return
        try:
            import os
            os.startfile(str(task.log_path))            # noqa: S606
        except Exception as exc:                        # noqa: BLE001
            self._append(f"[提示] 打不开日志文件：{exc}")

    def on_close(self) -> None:
        self.app.unsubscribe_log(self._log_cb)


