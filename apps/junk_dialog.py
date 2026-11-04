"""垃圾管理器：先看清、再勾选、才删除。

为什么做成"列表 + 勾选"而不是"一键清理"
-------------------------------------
"一键清理"的每一次误判都要用户用**不可再生的数据**付账。这里的方向相反：
先把磁盘上**可能是垃圾**的东西摊开（多大、多少文件、最新的一个是什么时候），
让用户按项勾选，再执行。默认勾中的只有"删了下次跑就有"的那一档（日志 / 缓存 /
调试产物）；数据集、标定图这类"删了要重拍"的**默认不勾**，说明里写清代价。

为什么时间范围默认"全部"、而默认勾选只有"推荐"档
----------------------------------------------
清理最容易伤到的是"昨天刚生成的、今天还要用"的文件 —— 但防它的是**勾选**，
不是时间：默认勾选的只有日志/缓存/调试产物（删了下次跑就有），"谨慎"档
（数据集、标定图）默认不勾。时间范围再筛一道就成了双保险，代价是
"打开看不到东西"（实测本项目 3.9 GiB 垃圾里 3.9 GiB 都是最近 7 天的）。
所以默认"全部"，想更保守的人自己选"30 天前"，界面会如实说"另有 X 被筛掉"。

真正的安全边界不在这个窗口，在 `apps/_junk.py`：扫描只从白名单读，
执行只允许删计划里出现过的路径。界面这边只负责"让人看清"。
"""

from __future__ import annotations

import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "sphere_tracker").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps._gui_widgets import BG, MUTED, WARN  # noqa: E402
from apps._junk import (  # noqa: E402
    DANGER_GONE,
    DANGER_SAFE,
    JunkPlan,
    clean_junk,
    human_size,
    scan_junk,
)

#: 时间范围下拉：``(给人看的标签, 天数或 None)``。
#: 默认"全部"：默认勾选的只有"删了下次跑就有"那一档，时间再筛一道会让
#: 功能看起来"什么都没有"（实测 3.9 GiB 垃圾全是最近 7 天的）。
RANGES: tuple[tuple[str, int | None], ...] = (
    ("全部（不按时间筛）", None),
    ("7 天前的", 7),
    ("30 天前的", 30),
    ("90 天前的", 90),
    ("1 年前的", 365),
)

_PICKED = "☑"
_UNPICKED = "☐"


class JunkDialog(tk.Toplevel):
    """垃圾管理器（模态）。"""

    COLS = (("pick", "", 34), ("label", "项", 250), ("kind", "档", 58),
            ("files", "文件", 62), ("size", "体积", 78),
            ("newest", "最新的一个", 120))

    def __init__(self, parent, *, root: str | Path | None = None) -> None:
        super().__init__(parent)
        self.root = Path(root) if root is not None else ROOT
        self.plan: JunkPlan = JunkPlan()
        self._picked: set[str] = set()

        self.title("垃圾管理器 · 清日志与临时产物")
        self.geometry("880x560")
        self.minsize(720, 460)
        self.configure(bg=BG)
        self.transient(parent)

        head = ttk.Frame(self, padding=(16, 12, 16, 0))
        head.pack(fill="x")
        ttk.Label(head, text="只清下面列出的项；data 里的录像、config/docs/assets 一律不碰。",
                  foreground=MUTED).pack(anchor="w")
        row = ttk.Frame(head)
        row.pack(fill="x", pady=(8, 0))
        ttk.Label(row, text="时间范围").pack(side="left")
        self.var_range = tk.StringVar(value=RANGES[0][0])
        combo = ttk.Combobox(row, values=[r[0] for r in RANGES], width=18,
                             state="readonly", textvariable=self.var_range)
        combo.pack(side="left", padx=(8, 12))
        combo.bind("<<ComboboxSelected>>", lambda _e: self._reload())
        ttk.Button(row, text="重新扫描", command=self._reload).pack(side="left")

        frame = ttk.Frame(self)
        frame.pack(fill="both", expand=True, padx=16, pady=(8, 0))
        self.tree = ttk.Treeview(frame, columns=[c[0] for c in self.COLS],
                                 show="headings", selectmode="none")
        for key, title, width in self.COLS:
            self.tree.heading(key, text=title)
            self.tree.column(
                key, width=width,
                anchor="center" if key in ("pick", "kind", "files", "size") else "w")
        bar = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=bar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        self.tree.tag_configure("safe", foreground="#111")
        self.tree.tag_configure("warn", foreground=WARN)
        self.tree.tag_configure("gone", foreground="#b91c1c")
        self.tree.bind("<Button-1>", self._on_click)
        self.tree.bind("<space>", lambda _e: self._toggle_focused())
        self.tree.bind("<Return>", lambda _e: self._toggle_focused())

        self.lbl_sum = ttk.Label(self, text="", foreground=MUTED)
        self.lbl_sum.pack(anchor="w", padx=16, pady=(6, 0))
        self.lbl_note = ttk.Label(self, text="", foreground=MUTED, wraplength=820)
        self.lbl_note.pack(anchor="w", padx=16)

        actions = ttk.Frame(self)
        actions.pack(fill="x", padx=16, pady=10)
        ttk.Button(actions, text="清理选中项", command=self._clean).pack(side="right")
        ttk.Button(actions, text="全不选", command=self._pick_none).pack(
            side="right", padx=(0, 6))
        ttk.Button(actions, text="全选推荐", command=self._pick_safe).pack(
            side="right", padx=(0, 6))
        ttk.Button(actions, text="关闭", command=self.destroy).pack(side="left")

        self._reload()
        self.grab_set()

    # ---------- 数据 ----------
    def _days(self) -> int | None:
        label = self.var_range.get()
        return next((d for name, d in RANGES if name == label), RANGES[0][1])

    def _reload(self) -> None:
        try:
            self.plan = scan_junk(self.root, older_than_days=self._days())
        except (OSError, ValueError) as exc:
            self.lbl_sum.configure(text=f"扫描失败：{exc}", foreground=WARN)
            return
        # 默认勾中"推荐"那一档；谨慎/彻底删的保持未勾。
        self._picked = {i.key for i in self.plan.items if i.danger == DANGER_SAFE}
        self._fill()

    def _fill(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        for it in self.plan.items:
            kind = {"safe": "推荐", "warn": "谨慎", "gone": "彻底删"}.get(it.danger, "")
            self.tree.insert(
                "", "end", iid=it.key,
                values=(_PICKED if it.key in self._picked else _UNPICKED,
                        it.label, kind, it.count, human_size(it.size),
                        _ago(it.newest)),
                tags=(it.danger,))
        if not self.plan.items:
            self.tree.insert("", "end", iid="__none__",
                             values=("", "（这个时间范围内没有可清理的东西）",
                                     "", "", "", ""), tags=("safe",))
        self._refresh_sum()

    def _refresh_sum(self) -> None:
        chosen = [i for i in self.plan.items if i.key in self._picked]
        total = sum(i.size for i in chosen)
        files = sum(i.count for i in chosen)
        extra = ""
        if self.plan.filtered_bytes or self.plan.filtered_files:
            extra = (f"；另有 {human_size(self.plan.filtered_bytes)}"
                     f"（{self.plan.filtered_files} 个文件）比所选时间范围新，已被筛掉")
        self.lbl_sum.configure(
            text=f"已选 {len(chosen)} 项 / {files} 个文件 / {human_size(total)}"
                 f"　（全部可清理：{human_size(self.plan.size)}{extra}）")
        gone = [i for i in chosen if i.danger == DANGER_GONE]
        if gone:
            self.lbl_note.configure(
                text="注意：选中的「回收站」是彻底删除，删了不可恢复。",
                foreground=WARN)
        else:
            self.lbl_note.configure(
                text="「推荐」= 删了下次跑就有；「谨慎」= 删了要重跑/重拍才能再生。",
                foreground=MUTED)

    # ---------- 勾选 ----------
    def _toggle(self, key: str) -> None:
        if key == "__none__" or not key:
            return
        if key in self._picked:
            self._picked.discard(key)
        else:
            self._picked.add(key)
        self.tree.set(key, "pick", _PICKED if key in self._picked else _UNPICKED)
        self._refresh_sum()

    def _toggle_focused(self) -> None:
        sel = self.tree.focus()
        if sel:
            self._toggle(sel)

    def _on_click(self, event) -> None:
        """点 ``pick`` 列 = 勾选/取消；点别的列只移动焦点（不误勾）。"""
        if self.tree.identify_region(event.x, event.y) not in ("cell", "tree"):
            return
        row = self.tree.identify_row(event.y)
        if not row:
            return
        self.tree.focus(row)
        col = self.tree.identify_column(event.x)
        if col == "#1":                      # pick 列
            self._toggle(row)

    def _pick_safe(self) -> None:
        self._picked = {i.key for i in self.plan.items if i.danger == DANGER_SAFE}
        self._fill()

    def _pick_none(self) -> None:
        self._picked.clear()
        self._fill()

    # ---------- 执行 ----------
    def _clean(self) -> None:
        chosen = [i for i in self.plan.items if i.key in self._picked]
        if not chosen:
            self.lbl_note.configure(text="先勾选要清理的项", foreground=WARN)
            return
        total = sum(i.size for i in chosen)
        files = sum(i.count for i in chosen)
        gone = [i for i in chosen if i.danger == DANGER_GONE]
        warn = [i for i in chosen if i.danger != DANGER_SAFE]

        lines = [f"将永久删除 {files} 个文件（{human_size(total)}）：", ""]
        lines += [f"· {i.label}　{human_size(i.size)}" for i in chosen[:8]]
        if len(chosen) > 8:
            lines.append(f"…另有 {len(chosen) - 8} 项")
        if gone:
            lines += ["", "⚠ 其中「回收站」里的会话删除后**无法恢复**。"]
        elif warn:
            lines += ["", "⚠ 其中「谨慎」项删了要重跑或重拍才能再生。"]
        lines += ["", "确定要删除吗？"]

        if not messagebox.askyesno("确认清理", "\n".join(lines), parent=self):
            return

        deleted, freed, failures = clean_junk(
            JunkPlan(items=chosen), self.root)
        note = f"已删除 {deleted} 个文件，释放 {human_size(freed)}"
        if failures:
            note += f"；{len(failures)} 个没删掉（多为正被占用）：" + failures[0]
        self.lbl_note.configure(
            text=note, foreground=WARN if failures else MUTED)
        self._reload()
        # _reload 会重置 lbl_note，这里把结果再写回去（用户最想看的是"删了多少"）
        self.lbl_note.configure(
            text=note, foreground=WARN if failures else MUTED)


def _ago(ts: float) -> str:
    """mtime → 「3 天前」这种粗粒度说法；空集/0 显示「—」。

    粗粒度是有意的：清理时用户判断的是"这是不是上个月那批"，不是精确到分钟 ——
    写秒数只会让人多读一行字。
    """
    if not ts:
        return "—"
    days = (time.time() - ts) / 86400.0
    if days < 1:
        return "今天"
    if days < 2:
        return "昨天"
    if days < 60:
        return f"{int(days)} 天前"
    if days < 730:
        return f"{int(days / 30)} 个月前"
    return f"{days / 365:.1f} 年前"
