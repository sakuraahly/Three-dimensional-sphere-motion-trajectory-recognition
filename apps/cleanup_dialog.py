"""清理/回收站 GUI：先预览，再软删除；彻底删除只发生在回收站里。

`apps/_cleanup.py` 负责路径安全与移动，本文件只做确认和展示。
"""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk

ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps._cleanup import (  # noqa: E402
    MODES,
    CleanupPlan,
    list_trash,
    move_to_trash,
    plan_cleanup,
    purge_batch,
    restore_batch,
    trash_root,
)

WARN = "#b45309"


class CleanupDialog(tk.Toplevel):
    """一次清理：整个会话或只清 derived/，只移到回收站。"""

    def __init__(self, parent, sessions_root: str | Path,
                 targets: list[str | Path], on_done=None) -> None:
        super().__init__(parent)
        self.title("清理会话（先移到回收站）")
        self.transient(parent)
        self.grab_set()
        self.root_dir = Path(sessions_root).resolve()
        self.targets = [Path(t) for t in targets]
        self.on_done = on_done
        self.plan: CleanupPlan | None = None
        self._build()
        self._preview()

    def _build(self) -> None:
        top = ttk.Frame(self, padding=12)
        top.pack(fill="both", expand=True)
        ttk.Label(top, text="⚠ 这里只把目录移到回收站，不会直接永久删除。",
                  foreground=WARN, font=("Microsoft YaHei UI", 10, "bold")).pack(anchor="w")
        ttk.Label(top, text=f"sessions 根目录：{self.root_dir}",
                  wraplength=760).pack(anchor="w", pady=(4, 2))
        ttk.Label(top, text=f"回收站：{trash_root(self.root_dir)}",
                  wraplength=760).pack(anchor="w", pady=(0, 8))

        self.mode_var = tk.StringVar(value="session")
        box = ttk.LabelFrame(top, text="清理方式", padding=8)
        box.pack(fill="x", pady=(0, 8))
        for key, text in MODES.items():
            ttk.Radiobutton(box, text=text, value=key, variable=self.mode_var,
                            command=self._preview).pack(anchor="w", pady=1)

        keep = ttk.Frame(top)
        keep.pack(fill="x", pady=(0, 8))
        ttk.Label(keep, text="保留最新").pack(side="left")
        self.keep_var = tk.StringVar(value="0")
        ttk.Spinbox(keep, from_=0, to=max(0, len(self.targets)), width=5,
                    textvariable=self.keep_var,
                    command=self._preview).pack(side="left", padx=(4, 4))
        ttk.Label(keep, text="个（按会话名倒序；0=不保留）").pack(side="left")

        self.summary = ttk.Label(top, text="", font=("Consolas", 10))
        self.summary.pack(anchor="w", pady=(0, 4))
        paths_box = ttk.Frame(top)
        paths_box.pack(fill="both", expand=True, pady=(0, 8))
        self.paths = tk.Text(paths_box, height=10, wrap="none")
        self.paths.pack(side="left", fill="both", expand=True)
        psb = ttk.Scrollbar(paths_box, orient="vertical", command=self.paths.yview)
        psb.pack(side="right", fill="y")
        self.paths.configure(yscrollcommand=psb.set)

        bar = ttk.Frame(top)
        bar.pack(fill="x")
        ttk.Button(bar, text="刷新预览", command=self._preview).pack(side="left")
        ttk.Button(bar, text="移到回收站", command=self._move).pack(side="right")
        ttk.Button(bar, text="取消", command=self.destroy).pack(side="right", padx=(0, 6))

    def _keep_latest(self) -> int:
        try:
            return max(0, int(self.keep_var.get().strip() or "0"))
        except ValueError:
            return 0

    def _preview(self) -> None:
        try:
            self.plan = plan_cleanup(self.root_dir, self.targets,
                                     mode=self.mode_var.get(),
                                     keep_latest=self._keep_latest())
        except Exception as exc:                            # noqa: BLE001
            self.plan = None
            self.summary.configure(text=f"预览失败：{exc}")
            self.paths.delete("1.0", "end")
            return
        self.summary.configure(
            text=f"将移动 {self.plan.summary()}；跳过 {len(self.plan.skipped)} 项")
        lines = []
        for item in self.plan.items:
            lines.append(f"{item.source}  ({item.files} 文件, {item.bytes} B)")
        for note in self.plan.skipped:
            lines.append(f"[跳过] {note}")
        self.paths.delete("1.0", "end")
        self.paths.insert("end", "\n".join(lines))

    def _move(self) -> None:
        self._preview()
        if self.plan is None or not self.plan.items:
            messagebox.showinfo("清理", "没有可移动的目录。", parent=self)
            return
        if not messagebox.askyesno(
                "确认移到回收站",
                f"{self.plan.summary()}\n\n只移到回收站，可在『回收站』恢复。继续？",
                parent=self):
            return
        try:
            batch = move_to_trash(self.root_dir, self.plan, label="manual")
        except Exception as exc:                            # noqa: BLE001
            messagebox.showerror("清理失败", str(exc), parent=self)
            return
        messagebox.showinfo(
            "已移到回收站",
            f"批次：{batch}\n\n可在数据管理的『回收站』恢复或彻底删除。",
            parent=self)
        if self.on_done:
            self.on_done()
        self.destroy()


class TrashDialog(tk.Toplevel):
    """回收站：恢复，或输入 DELETE 后彻底删除。"""

    def __init__(self, parent, sessions_root: str | Path, on_done=None) -> None:
        super().__init__(parent)
        self.title("回收站")
        self.transient(parent)
        self.grab_set()
        self.root_dir = Path(sessions_root).resolve()
        self.on_done = on_done
        self.batches = []
        self._build()
        self.refresh()

    def _build(self) -> None:
        top = ttk.Frame(self, padding=12)
        top.pack(fill="both", expand=True)
        ttk.Label(top, text=f"回收站：{trash_root(self.root_dir)}",
                  wraplength=760).pack(anchor="w", pady=(0, 8))
        cols = ("batch", "created", "mode", "items", "size")
        heads = ("批次", "时间", "方式", "项数", "大小")
        widths = (210, 150, 70, 55, 110)
        self.tree = ttk.Treeview(top, columns=cols, show="headings",
                                 selectmode="browse", height=12)
        for c, h, w in zip(cols, heads, widths):
            self.tree.heading(c, text=h)
            self.tree.column(c, width=w, anchor="w", stretch=(c == "batch"))
        self.tree.pack(fill="both", expand=True, pady=(0, 8))
        bar = ttk.Frame(top)
        bar.pack(fill="x")
        ttk.Button(bar, text="刷新", command=self.refresh).pack(side="left")
        ttk.Button(bar, text="恢复选中", command=self._restore).pack(side="left", padx=(6, 0))
        ttk.Button(bar, text="彻底删除选中", command=self._purge).pack(side="left", padx=(6, 0))
        ttk.Button(bar, text="关闭", command=self.destroy).pack(side="right")

    def refresh(self) -> None:
        self.batches = list_trash(self.root_dir)
        self.tree.delete(*self.tree.get_children())
        for b in self.batches:
            self.tree.insert("", "end", iid=b.path.name,
                             values=(b.path.name, b.created, b.mode,
                                     len(b.items), f"{b.bytes / 1024 ** 2:.1f} MiB"))

    def _selected(self):
        sel = self.tree.selection()
        if not sel:
            return None
        return next((b for b in self.batches if b.path.name == sel[0]), None)

    def _restore(self) -> None:
        b = self._selected()
        if b is None:
            messagebox.showinfo("回收站", "先选一个批次。", parent=self)
            return
        if not messagebox.askyesno("恢复", f"恢复批次？\n{b.path.name}", parent=self):
            return
        try:
            n = restore_batch(self.root_dir, b.path)
        except Exception as exc:                            # noqa: BLE001
            messagebox.showerror("恢复失败", str(exc), parent=self)
            return
        messagebox.showinfo("恢复", f"已恢复 {n} 个目录。", parent=self)
        if self.on_done:
            self.on_done()
        self.refresh()

    def _purge(self) -> None:
        b = self._selected()
        if b is None:
            messagebox.showinfo("回收站", "先选一个批次。", parent=self)
            return
        if not messagebox.askyesno(
                "彻底删除", f"这会永久删除：\n{b.path}\n不可恢复。继续？", parent=self):
            return
        word = simpledialog.askstring(
            "彻底删除", "输入 DELETE 确认永久删除：", parent=self)
        if word != "DELETE":
            messagebox.showinfo("已取消", "没有删除。", parent=self)
            return
        try:
            purge_batch(self.root_dir, b.path)
        except Exception as exc:                            # noqa: BLE001
            messagebox.showerror("删除失败", str(exc), parent=self)
            return
        messagebox.showinfo("已删除", "批次已永久删除。", parent=self)
        if self.on_done:
            self.on_done()
        self.refresh()
