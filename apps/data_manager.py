"""数据管理面板：勾选会话、脱敏、按类别打成 zip。

运行：
    python apps/data_manager.py

也可以从主界面右上角『数据管理』打开。
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
from datetime import datetime, time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps._data_export import (  # noqa: E402
    CATEGORIES,
    DEFAULT_CATEGORIES,
    ExportOptions,
    ExportResult,
    SessionItem,
    estimate_bytes,
    export_zip,
    normalize_output,
    scan_items,
)

ACCENT = "#0f4c81"
MUTED = "#6b7280"


def _fmt_bytes(n: int) -> str:
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.2f} GiB"
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.1f} MiB"
    if n >= 1024:
        return f"{n / 1024:.1f} KiB"
    return f"{n} B"


def _parse_dt(text: str, end: bool = False) -> datetime | None:
    text = text.strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(text, fmt)
            if fmt == "%Y-%m-%d" and end:
                return datetime.combine(dt.date(), time(23, 59, 59))
            return dt
        except ValueError:
            continue
    return None


class DataManager(tk.Tk):
    def __init__(self, root_dir: Path) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.title("数据管理 · 脱敏导出")
        self.geometry("1280x860")
        self.minsize(1050, 700)
        self.items: list[SessionItem] = []
        self.selected: set[str] = set()
        self._anchor_id: str | None = None
        self._running = False
        self._current_out: Path | None = None
        self._cat_vars: dict[str, tk.BooleanVar] = {}
        self._redact_vars: dict[str, tk.BooleanVar] = {}
        self._ui_queue: queue.Queue = queue.Queue()
        self._build()
        self.refresh()
        self.after(100, self._poll_ui)

    # ------------------------------------------------------------------
    def _build(self) -> None:
        head = tk.Frame(self, bg=ACCENT)
        head.pack(fill="x")
        tk.Label(head, text="数据管理 · 脱敏导出", bg=ACCENT, fg="white",
                 font=("Microsoft YaHei UI", 16, "bold")).pack(
            side="left", padx=18, pady=(10, 2))
        tk.Label(head, text="勾选会话 → 选类别 → 自动去标识 → 打成有目录的 zip",
                 bg=ACCENT, fg="#dbeafe",
                 font=("Microsoft YaHei UI", 9)).pack(side="left", padx=(0, 8), pady=(10, 2))
        ttk.Button(head, text="刷新", command=self.refresh).pack(side="right", padx=12, pady=8)

        toolbar = ttk.Frame(self, padding=(12, 8))
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="全选", command=self.select_all).pack(side="left")
        ttk.Button(toolbar, text="全不选", command=self.select_none).pack(side="left", padx=4)
        ttk.Button(toolbar, text="反选", command=self.invert).pack(side="left", padx=4)
        ttk.Button(toolbar, text="选到此处", command=self.select_to_here).pack(side="left", padx=4)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Label(toolbar, text="从").pack(side="left")
        self.from_var = tk.StringVar()
        ttk.Entry(toolbar, textvariable=self.from_var, width=17).pack(side="left", padx=(2, 4))
        ttk.Label(toolbar, text="到").pack(side="left")
        self.to_var = tk.StringVar()
        ttk.Entry(toolbar, textvariable=self.to_var, width=17).pack(side="left", padx=(2, 4))
        ttk.Button(toolbar, text="按时间选择", command=self.select_time).pack(side="left")
        ttk.Button(toolbar, text="清除时间", command=self.clear_time).pack(side="left", padx=4)
        ttk.Label(toolbar, text="格式 YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS",
                  foreground=MUTED).pack(side="left", padx=8)

        body = ttk.PanedWindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=12, pady=(0, 6))

        left = ttk.Frame(body)
        right = ttk.Frame(body, width=330)
        body.add(left, weight=4)
        body.add(right, weight=1)

        cols = ("sel", "id", "created", "frames", "status", "size")
        heads = ("选", "会话", "时间", "帧", "状态", "大小")
        widths = (34, 170, 145, 75, 50, 95)
        self.tree = ttk.Treeview(left, columns=cols, show="headings",
                                 selectmode="extended", height=18)
        for c, h, w in zip(cols, heads, widths):
            self.tree.heading(c, text=h)
            self.tree.column(c, width=w, minwidth=w, anchor="w",
                             stretch=(c in {"id", "created"}))
        self.tree.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        sb.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.bind("<Button-1>", self._tree_click)
        self.tree.bind("<Double-1>", lambda _e: self.toggle_focus())

        tip = ttk.Label(left, text="点第一列勾选；Shift+点行做区间勾选；双击也可切换。",
                        foreground=MUTED)
        tip.pack(side="bottom", anchor="w", pady=(4, 0))

        self.count_label = ttk.Label(right, text="", font=("Microsoft YaHei UI", 10, "bold"))
        self.count_label.pack(anchor="w", pady=(0, 6))

        cats = ttk.LabelFrame(right, text="导出类别", padding=8)
        cats.pack(fill="x", pady=(0, 8))
        for key, label in CATEGORIES.items():
            var = tk.BooleanVar(value=key in DEFAULT_CATEGORIES)
            self._cat_vars[key] = var
            ttk.Checkbutton(cats, text=label, variable=var,
                            command=self._refresh_count).pack(anchor="w", pady=1)
        ttk.Label(cats, text="默认不含视频；视频画面未打码，勾选前确认授权。",
                  foreground="#b45309", wraplength=300).pack(anchor="w", pady=(4, 0))

        red = ttk.LabelFrame(right, text="脱敏", padding=8)
        red.pack(fill="x", pady=(0, 8))
        for key, label in (("camera", "移除相机 identity/name/port"),
                           ("labels", "移除会话 label/notes"),
                           ("platform", "移除 OS platform")):
            var = tk.BooleanVar(value=True)
            self._redact_vars[key] = var
            ttk.Checkbutton(red, text=label, variable=var).pack(anchor="w", pady=1)

        out_box = ttk.LabelFrame(right, text="输出", padding=8)
        out_box.pack(fill="x", pady=(0, 8))
        self.out_var = tk.StringVar(value=self._default_output())
        ttk.Entry(out_box, textvariable=self.out_var).pack(fill="x", pady=(0, 4))
        ttk.Button(out_box, text="选择 zip 路径…", command=self.browse_output).pack(anchor="w")

        self.export_btn = ttk.Button(right, text="开始导出", command=self.start_export)
        self.export_btn.pack(fill="x", pady=(2, 4))
        self.progress = ttk.Progressbar(right, mode="determinate")
        self.progress.pack(fill="x", pady=(0, 4))
        self.status = ttk.Label(right, text="等待选择", foreground=MUTED,
                                wraplength=310, justify="left")
        self.status.pack(anchor="w")

        log_box = ttk.LabelFrame(self, text="导出日志", padding=4)
        log_box.pack(fill="x", padx=12, pady=(0, 8))
        self.log = tk.Text(log_box, height=6, wrap="none")
        self.log.pack(side="left", fill="both", expand=True)
        lsb = ttk.Scrollbar(log_box, orient="vertical", command=self.log.yview)
        lsb.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=lsb.set)

    def _default_output(self) -> str:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return str(ROOT / "outputs" / "exports" / f"sphere-stereo-export-{stamp}.zip")

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        try:
            self.items = scan_items(self.root_dir)
        except Exception as exc:                            # noqa: BLE001
            messagebox.showerror("扫描失败", str(exc), parent=self)
            return
        valid = {it.key for it in self.items}
        self.selected &= valid
        self.tree.delete(*self.tree.get_children())
        for it in self.items:
            size = _fmt_bytes(it.size_bytes)
            self.tree.insert("", "end", iid=it.key,
                             values=("☑" if it.key in self.selected else "☐",
                                     it.id, it.created,
                                     f"{it.frames_left}/{it.frames_right}",
                                     it.status, size))
        self._refresh_count()

    def _refresh_count(self) -> None:
        chosen = [it for it in self.items if it.key in self.selected]
        try:
            opts = self._options()
            size = estimate_bytes(chosen, opts)
        except Exception:                                   # noqa: BLE001
            size = sum(it.size_bytes for it in chosen)
        self.count_label.configure(
            text=f"已选 {len(chosen)} / {len(self.items)} 个会话\n"
                 f"预计原始数据 {_fmt_bytes(size)}")
        self._set_row_checks()
        self._sync_highlight()

    def _sync_highlight(self) -> None:
        if not hasattr(self, "tree"):
            return
        try:
            ids = [it.key for it in self.items if it.key in self.selected
                   and self.tree.exists(it.key)]
            self.tree.selection_set(ids)
        except tk.TclError:
            pass

    def _set_row_checks(self) -> None:
        for it in self.items:
            if self.tree.exists(it.key):
                self.tree.set(it.key, "sel", "☑" if it.key in self.selected else "☐")

    def _options(self) -> ExportOptions:
        cats = frozenset(k for k, v in self._cat_vars.items() if v.get())
        return ExportOptions(
            categories=cats,
            redact_camera=bool(self._redact_vars["camera"].get()),
            redact_labels=bool(self._redact_vars["labels"].get()),
            redact_platform=bool(self._redact_vars["platform"].get()),
        )

    # ------------------------------------------------------------------
    def _tree_click(self, event):
        row = self.tree.identify_row(event.y)
        col = self.tree.identify_column(event.x)
        if not row:
            return None
        if col == "#1":
            self.toggle_key(row)
            self._anchor_id = row
            return "break"
        if event.state & 0x0001 and self._anchor_id:      # Shift
            self._select_range(self._anchor_id, row)
            return "break"
        self._anchor_id = row
        return None

    def toggle_focus(self) -> None:
        row = self.tree.focus()
        if row:
            self.toggle_key(row)

    def toggle_key(self, key: str) -> None:
        if key in self.selected:
            self.selected.remove(key)
        else:
            self.selected.add(key)
        self._refresh_count()

    def _select_range(self, a: str, b: str) -> None:
        keys = [it.key for it in self.items]
        try:
            i, j = keys.index(a), keys.index(b)
        except ValueError:
            return
        lo, hi = sorted((i, j))
        for key in keys[lo:hi + 1]:
            self.selected.add(key)
        self._refresh_count()

    def select_all(self) -> None:
        self.selected = {it.key for it in self.items}
        self._refresh_count()

    def select_none(self) -> None:
        self.selected.clear()
        self._refresh_count()

    def invert(self) -> None:
        all_keys = {it.key for it in self.items}
        self.selected = all_keys - self.selected
        self._refresh_count()

    def select_to_here(self) -> None:
        row = self.tree.focus()
        if not row:
            messagebox.showinfo("选到此处", "先点一行作为终点。", parent=self)
            return
        if not self._anchor_id:
            self._anchor_id = row
        self._select_range(self._anchor_id, row)

    def select_time(self) -> None:
        start = _parse_dt(self.from_var.get())
        end = _parse_dt(self.to_var.get(), end=True)
        if start is None and end is None:
            messagebox.showinfo("按时间选择", "填一个开始或结束时间，例如 2026-10-20。",
                                parent=self)
            return
        if start and end and start > end:
            messagebox.showwarning("按时间选择", "开始时间晚于结束时间。", parent=self)
            return
        chosen: set[str] = set()
        for it in self.items:
            try:
                created = datetime.strptime(it.created, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            if start and created < start:
                continue
            if end and created > end:
                continue
            chosen.add(it.key)
        self.selected = chosen
        self._refresh_count()

    def clear_time(self) -> None:
        self.from_var.set("")
        self.to_var.set("")

    # ------------------------------------------------------------------
    def browse_output(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self, title="导出 zip", defaultextension=".zip",
            filetypes=[("ZIP", "*.zip")],
            initialdir=str(Path(self.out_var.get()).parent),
            initialfile=Path(self.out_var.get()).name)
        if path:
            self.out_var.set(path)

    def _log_line(self, text: str) -> None:
        self.log.insert("end", text + "\n")
        self.log.see("end")

    def start_export(self) -> None:
        if self._running:
            return
        chosen = [it for it in self.items if it.key in self.selected]
        if not chosen:
            messagebox.showinfo("导出", "先勾选至少一个会话。", parent=self)
            return
        try:
            opts = self._options()
        except ValueError as exc:
            messagebox.showerror("导出", str(exc), parent=self)
            return
        if not opts.categories:
            messagebox.showinfo("导出", "至少选一个导出类别。", parent=self)
            return
        if ("raw" in opts.categories or "annotated" in opts.categories) and not messagebox.askyesno(
                "视频隐私", "视频画面没有自动打码，房间/人物会被一起交付。\n继续导出？",
                parent=self):
            return
        out = normalize_output(self.out_var.get())
        self.out_var.set(str(out))
        if out.exists() and not messagebox.askyesno(
                "覆盖", f"文件已存在，覆盖？\n{out}", parent=self):
            return
        if out.exists():
            try:
                out.unlink()
            except OSError as exc:
                messagebox.showerror("导出", f"删不掉旧文件：{exc}", parent=self)
                return
        self._running = True
        self._current_out = out
        self.export_btn.configure(state="disabled")
        self.progress["value"] = 0
        self.status.configure(text="导出中…")
        self._log_line(f"开始导出 {len(chosen)} 个会话 → {out}")

        def work() -> None:
            try:
                result = export_zip(
                    chosen, opts, out,
                    progress=lambda done, total, name: self._ui_queue.put(
                        ("progress", done, total, name)))
            except Exception as exc:                        # noqa: BLE001
                self._ui_queue.put(("error", exc))
            else:
                self._ui_queue.put(("done", result))

        threading.Thread(target=work, daemon=True).start()

    def _poll_ui(self) -> None:
        """后台线程不能碰 Tk；只把消息放进队列，这里在主线程消费。"""
        while True:
            try:
                msg = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            kind = msg[0]
            if kind == "progress":
                self._on_progress(msg[1], msg[2], msg[3])
            elif kind == "error":
                self._on_error(msg[1])
            elif kind == "done":
                self._on_done(msg[1])
        try:
            self.after(100, self._poll_ui)
        except tk.TclError:
            pass

    def _on_progress(self, done: int, total: int, name: str) -> None:
        self.progress["maximum"] = max(total, 1)
        self.progress["value"] = done
        self.status.configure(text=f"{done}/{total}  {name}")
        self._log_line(name)

    def _on_error(self, exc: Exception) -> None:
        self._running = False
        self.export_btn.configure(state="normal")
        if self._current_out is not None:
            try:
                if self._current_out.exists():
                    self._current_out.unlink()
            except OSError:
                pass
        self._current_out = None
        self.status.configure(text=f"失败：{exc}")
        self._log_line(f"[失败] {exc}")
        messagebox.showerror("导出失败", str(exc), parent=self)

    def _on_done(self, result: ExportResult) -> None:
        self._running = False
        self._current_out = None
        self.export_btn.configure(state="normal")
        self.progress["value"] = self.progress["maximum"]
        self.status.configure(text=f"完成：{result.output.name}  {_fmt_bytes(result.bytes_written)}")
        self._log_line(f"完成：{result.output}  {_fmt_bytes(result.bytes_written)}")
        if messagebox.askyesno("导出完成", f"{result.output}\n\n打开所在文件夹？", parent=self):
            self.open_output_folder(result.output)

    def open_output_folder(self, path: Path) -> None:
        folder = Path(path).parent
        try:
            os.startfile(str(folder))                       # type: ignore[attr-defined]
        except Exception:                                   # noqa: BLE001
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description="数据管理 · 脱敏导出")
    ap.add_argument("--root", default=str(ROOT / "data" / "sessions"))
    args = ap.parse_args()
    app = DataManager(Path(args.root))
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
