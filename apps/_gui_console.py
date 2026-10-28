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
        self.geometry("900x560")
        self.surface = Surface(self)

        box = self.surface.card(
            self, "任务",
            "卡住时在这里中断：直接关窗口不会杀掉子进程，相机可能仍被占着。"
            "「中断」先试优雅退出，3 秒不退再强制结束整棵进程树。")
        frame = ttk.Frame(box)
        frame.grid(row=1, column=0, columnspan=9, sticky="we", pady=(4, 4))
        cols = ("id", "title", "pid", "elapsed", "status")
        heads = ("#", "任务", "PID", "已运行", "状态")
        widths = (4, 34, 8, 9, 14)
        self.tree = ttk.Treeview(frame, columns=cols, show="headings",
                                 height=5, selectmode="browse")
        for c, h, w in zip(cols, heads, widths):
            self.tree.heading(c, text=h)
            self.tree.column(c, width=w, anchor="w" if c in ("title", "status") else "center")
        bar = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=bar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        row = self.surface.action_row(box)
        self.surface.primary(row, "中断选中", self._stop_selected)
        self.surface.secondary(row, "中断全部", self._stop_all)
        self.surface.secondary(row, "打开日志文件", self._open_log_file)

        logbox = self.surface.card(self, "运行日志", "")
        self.text = tk.Text(logbox, height=14, wrap="none", state="disabled",
                            font=("Consolas", 9), bg="#f8fafc")
        self.text.grid(row=1, column=0, columnspan=9, sticky="nsew")
        logbox.rowconfigure(1, weight=1)
        bar2 = ttk.Scrollbar(logbox, orient="vertical", command=self.text.yview)
        bar2.grid(row=1, column=9, sticky="ns")
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


