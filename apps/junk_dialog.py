"""垃圾管理器：先看清、再勾选、才删除；另有三个"一键"入口。

判断依据（三层）
----------------
1. **位置白名单**：只从 `apps/_junk.py` 里声明过的位置读 ——
   `outputs/` 的一级子目录、`__pycache__`、`*.tmp`、`session.json.bak-*`、
   `data/_trash`。不在这个清单里的路径，删函数**根本拿不到**（想删
   `data/stereo` 得先改扫描器，不是在界面里多写一句 unlink）。
2. **名字**：`_`/`tmp_`/`agent_` 前缀 = 调试产物（推荐档）；
   `logs`/`test_results` 是写死认得的安全目录；其余一律"谨慎"（默认不勾）。
3. **内容**：光看名字会猜错 —— `outputs/_ab` 1.6 GiB 带 `_` 前缀，形状却是数据。
   所以推荐档还要过一遍 `looks_like_data`：文件大、或结果类后缀占六成，
   就**自动降级为谨慎**并写清原因。

目录变了怎么办
--------------
- **新增** `outputs/<新名字>`：不在白名单名字里、又不带调试前缀 → 落到"谨慎"
  （默认不勾）。不知道就先别删，这是安全的一侧。
- **改名/移走** 了白名单里声明的位置（如 `outputs/logs`）：这次扫描找不到，
  会列在"找不到的位置"里提示出来 —— 静默跳过会让人以为"本来就没有"。
- **你自己拿主意**：右键某一行可以「标为不动」，选择记在 `config/gui_state.json`，
  下次仍不默认勾它。

为什么"一键"也要弹确认
----------------------
一键清日志/缓存是**真的删文件**（不进回收站：这些东西的价值就是可再生）。
一键只省掉"逐项勾选"，不省掉"知道要删什么" —— 确认框里仍写清删多少个文件、
释放多少、以及其中有没有"谨慎"档。
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
    DANGER_WARN,
    JunkItem,
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

#: 三个"一键"的目标。键与 `_junk.scan_junk` 产出的 `JunkItem.key` 一致。
QUICK_LOGS = ("outputs/logs",)
QUICK_CACHE = ("__pycache__", "tmp_files", "session_bak")

_PICKED = "☑"
_UNPICKED = "☐"
_MUTED_TAG = "muted"


class JunkDialog(tk.Toplevel):
    """垃圾管理器（模态）。"""

    COLS = (("pick", "", 36), ("label", "项", 260), ("kind", "档", 58),
            ("files", "文件", 64), ("size", "体积", 80),
            ("newest", "最新的一个", 100))

    def __init__(self, parent, *, root: str | Path | None = None,
                 prefs=None) -> None:
        super().__init__(parent)
        self.root = Path(root) if root is not None else ROOT
        # 记住"标为不动"的选择。prefs 可以是 None（测试/独立运行时就不记）。
        self.prefs = prefs
        self.plan: JunkPlan = JunkPlan()
        self._picked: set[str] = set()
        self._muted: set[str] = self._load_muted()

        self.title("垃圾管理器 · 清日志与临时产物")
        self._place_window()
        self.configure(bg=BG)
        self.transient(parent)
        self.resizable(True, True)

        # ---------- 顶部 ----------
        head = ttk.Frame(self, padding=(16, 12, 16, 0))
        head.pack(fill="x")
        ttk.Label(
            head,
            text="只清下面列出的项。判定 = 位置白名单 + 名字 + 内容特征；"
                 "看起来像数据的一律降为「谨慎」（默认不勾）。",
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
        ttk.Button(row, text="全选推荐", command=self._pick_safe).pack(
            side="left", padx=(12, 0))
        ttk.Button(row, text="全不选", command=self._pick_none).pack(
            side="left", padx=(6, 0))
        self.btn_full = ttk.Button(row, text="全屏", command=self._toggle_full)
        self.btn_full.pack(side="right")

        # ---------- 列表 ----------
        frame = ttk.Frame(self)
        frame.pack(fill="both", expand=True, padx=16, pady=(8, 0))
        self.tree = ttk.Treeview(frame, columns=[c[0] for c in self.COLS],
                                 show="headings", selectmode="none")
        for key, title, width in self.COLS:
            self.tree.heading(key, text=title)
            self.tree.column(
                key, width=width, minwidth=width,
                # 「项」列吃掉多余宽度：全屏时不至于右边一大片空白
                stretch=(key == "label"),
                anchor="center" if key in ("pick", "kind", "files", "size") else "w")
        bar = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=bar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        self.tree.tag_configure("safe", foreground="#111")
        self.tree.tag_configure("warn", foreground=WARN)
        self.tree.tag_configure("gone", foreground="#b91c1c")
        self.tree.tag_configure(_MUTED_TAG, foreground="#6b7280")
        self.tree.bind("<Button-1>", self._on_click)
        self.tree.bind("<space>", lambda _e: self._toggle_focused())
        self.tree.bind("<Return>", lambda _e: self._toggle_focused())
        self.tree.bind("<Button-3>", self._popup_menu)

        # 右键：给"这项我拿不准 / 别默认勾"一个出口，不占按钮位
        self.menu = tk.Menu(self, tearoff=0)
        self.menu.add_command(label="标为不动（以后别默认勾选）",
                              command=lambda: self._set_muted(True))
        self.menu.add_command(label="恢复默认（可被默认勾选）",
                              command=lambda: self._set_muted(False))

        # ---------- 统计 / 说明 ----------
        self.lbl_sum = ttk.Label(self, text="", foreground=MUTED)
        self.lbl_sum.pack(anchor="w", padx=16, pady=(6, 0))
        self.lbl_note = ttk.Label(self, text="", foreground=MUTED, wraplength=980)
        self.lbl_note.pack(anchor="w", padx=16)

        # ---------- 一键区 ----------
        quick = ttk.Frame(self)
        quick.pack(fill="x", padx=16, pady=(8, 0))
        ttk.Label(quick, text="一键：", foreground=MUTED).pack(side="left")
        ttk.Button(quick, text="清日志", command=self._quick_logs).pack(
            side="left", padx=(0, 6))
        ttk.Button(quick, text="清缓存与临时",
                   command=self._quick_cache).pack(side="left", padx=(0, 6))
        ttk.Button(quick, text="清全部推荐项",
                   command=self._quick_safe).pack(side="left")

        # ---------- 底部 ----------
        actions = ttk.Frame(self)
        actions.pack(fill="x", padx=16, pady=(8, 10))
        ttk.Button(actions, text="清理选中项", command=self._clean_picked).pack(
            side="right")
        ttk.Button(actions, text="关闭", command=self.destroy).pack(side="left")

        self._reload()
        self.grab_set()

    # ---------- 窗口 ----------
    def _place_window(self) -> None:
        """初始尺寸按屏幕比例给（不再是死 880x560），并居中。"""
        sw = max(1024, self.winfo_screenwidth())
        sh = max(720, self.winfo_screenheight())
        w = min(max(900, int(sw * 0.72)), sw - 80)
        h = min(max(580, int(sh * 0.72)), sh - 80)
        x = max(0, (sw - w) // 2)
        y = max(0, (sh - h) // 2 - 24)
        self.geometry(f"{w}x{h}+{x}+{y}")
        self._normal_geometry = f"{w}x{h}+{x}+{y}"

    def _toggle_full(self) -> None:
        """全屏 ↔ 还原。用 ``state("zoomed")``（Windows 上就是最大化）。"""
        try:
            if self.state() == "zoomed":
                self.state("normal")
                self.geometry(self._normal_geometry)
                self.btn_full.configure(text="全屏")
            else:
                self._normal_geometry = self.geometry()
                self.state("zoomed")
                self.btn_full.configure(text="还原")
        except tk.TclError:
            pass                     # 某些窗口管理器不支持 zoomed，忽略即可

    # ---------- 偏好：标为不动 ----------
    def _load_muted(self) -> set[str]:
        if self.prefs is None:
            return set()
        raw = str(self.prefs.get_str("junk.muted", "") or "")
        return {k.strip() for k in raw.split(",") if k.strip()}

    def _store_muted(self) -> None:
        if self.prefs is None:
            return
        try:
            self.prefs.set("junk.muted", ",".join(sorted(self._muted)))
            self.prefs.save()
        except (OSError, ValueError):
            pass                     # 存不下不该打断清理这件事

    def _popup_menu(self, event) -> None:
        row = self.tree.identify_row(event.y)
        if not row or row == "__none__":
            return
        self.tree.focus(row)
        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()

    def _set_muted(self, muted: bool) -> None:
        key = self.tree.focus()
        if not key or key == "__none__":
            return
        if muted:
            self._muted.add(key)
            self._picked.discard(key)
        else:
            self._muted.discard(key)
        self._store_muted()
        self._fill()

    # ---------- 数据 ----------
    def _days(self) -> int | None:
        label = self.var_range.get()
        return next((d for name, d in RANGES if name == label), RANGES[0][1])

    def _reload(self) -> None:
        try:
            self.plan = scan_junk(self.root, older_than_days=self._days(),
                                  muted=self._muted)
        except (OSError, ValueError) as exc:
            self.lbl_sum.configure(text=f"扫描失败：{exc}", foreground=WARN)
            return
        # 默认勾中"推荐"档，且尊重"标为不动"
        self._picked = {i.key for i in self.plan.items
                        if i.danger == DANGER_SAFE and not i.muted}
        self._fill()

    def _fill(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        for it in self.plan.items:
            kind = {DANGER_SAFE: "推荐", DANGER_WARN: "谨慎",
                    DANGER_GONE: "彻底删"}.get(it.danger, "")
            tags = [it.danger] if not it.muted else [_MUTED_TAG, it.danger]
            self.tree.insert(
                "", "end", iid=it.key,
                values=(_PICKED if it.key in self._picked else _UNPICKED,
                        it.label + ("（不动）" if it.muted else ""), kind,
                        it.count, human_size(it.size), _ago(it.newest)),
                tags=tuple(tags))
        if not self.plan.items:
            self.tree.insert("", "end", iid="__none__",
                             values=("", "（这个时间范围内没有可清理的东西）",
                                     "", "", "", ""), tags=("safe",))
        self._refresh_sum()

    def _refresh_sum(self) -> None:
        chosen = [i for i in self.plan.items if i.key in self._picked]
        total = sum(i.size for i in chosen)
        files = sum(i.count for i in chosen)
        safe = [i for i in self.plan.items if i.danger == DANGER_SAFE]
        warn = [i for i in self.plan.items if i.danger == DANGER_WARN]
        degraded = [i for i in warn if i.note and "降为" in i.note]

        text = (f"已选 {len(chosen)} 项 / {files} 个文件 / {human_size(total)}"
                f"　｜　推荐档 {human_size(sum(i.size for i in safe))}"
                f"（{len(safe)} 项）"
                f"　谨慎档 {human_size(sum(i.size for i in warn))}（{len(warn)} 项）")
        if degraded:
            text += f"，其中 {len(degraded)} 项是「看起来像数据」被自动降级的"
        if self.plan.filtered_bytes or self.plan.filtered_files:
            text += (f"　｜　另有 {human_size(self.plan.filtered_bytes)}"
                     f"（{self.plan.filtered_files} 个文件）比所选时间范围新，已筛掉")
        self.lbl_sum.configure(text=text)

        notes: list[str] = []
        if self.plan.missing:
            notes.append("找不到的位置：" + "；".join(self.plan.missing))
        if any(i.danger == DANGER_GONE for i in chosen):
            notes.append("选中的「回收站」是彻底删除，删了不可恢复。")
        if not notes:
            notes.append("「推荐」= 删了下次跑就有；「谨慎」= 删了要重跑/重拍才能再生；"
                         "右键某行可「标为不动」。")
        self.lbl_note.configure(
            text="　".join(notes),
            foreground=WARN if self.plan.missing or any(
                i.danger == DANGER_GONE for i in chosen) else MUTED)

    # ---------- 勾选 ----------
    def _toggle(self, key: str) -> None:
        if not key or key == "__none__":
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
        """点 ``pick`` 列 = 勾选/取消；点别的列 = 看这项的说明（不误勾）。"""
        if self.tree.identify_region(event.x, event.y) not in ("cell", "tree"):
            return
        row = self.tree.identify_row(event.y)
        if not row:
            return
        self.tree.focus(row)
        if self.tree.identify_column(event.x) == "#1":
            self._toggle(row)
        else:
            self._describe(row)

    def _describe(self, key: str) -> None:
        """把这一项的 note / detail 写到说明行。

        为什么需要它：降级原因（"看起来像数据集，已降为谨慎"）只统计个数是不够的 ——
        用户点开某项时最想知道的正是"你为什么把它当谨慎"。
        """
        it = next((i for i in self.plan.items if i.key == key), None)
        if it is None:
            return
        why = it.note or it.detail
        if not why:
            return
        self.lbl_note.configure(
            text=f"{it.label}：{why}",
            foreground=WARN if it.danger != DANGER_SAFE else MUTED)

    def _pick_safe(self) -> None:
        self._picked = {i.key for i in self.plan.items
                        if i.danger == DANGER_SAFE and not i.muted}
        self._fill()

    def _pick_none(self) -> None:
        self._picked.clear()
        self._fill()

    # ---------- 一键 ----------
    def _quick_logs(self) -> None:
        self._run_quick(keys=QUICK_LOGS, title="一键清日志",
                        missing_hint="没有找到 outputs/logs —— 可能这个项目还没跑过任务，"
                                     "或它被改名/移动了")

    def _quick_cache(self) -> None:
        self._run_quick(keys=QUICK_CACHE, title="一键清缓存与临时",
                        missing_hint="没有字节码缓存、临时残留或多余的会话备份可清")

    def _quick_safe(self) -> None:
        self._run_quick(only_safe=True, title="一键清全部推荐项",
                        missing_hint="当前没有「推荐」档的东西可清")

    def _run_quick(self, *, keys: tuple[str, ...] | None = None,
                   only_safe: bool = False, title: str,
                   missing_hint: str) -> None:
        items = [
            i for i in self.plan.items
            if (keys is None or i.key in keys)
            and (not only_safe or i.danger == DANGER_SAFE)
            and not i.muted
        ]
        if not items:
            self.lbl_note.configure(text=missing_hint, foreground=MUTED)
            return
        self._execute(items, title=title)

    # ---------- 执行 ----------
    def _clean_picked(self) -> None:
        chosen = [i for i in self.plan.items if i.key in self._picked]
        if not chosen:
            self.lbl_note.configure(text="先勾选要清理的项", foreground=WARN)
            return
        self._execute(chosen, title="清理选中项")

    def _execute(self, items: list[JunkItem], *, title: str) -> None:
        total = sum(i.size for i in items)
        files = sum(i.count for i in items)
        gone = [i for i in items if i.danger == DANGER_GONE]
        risky = [i for i in items if i.danger != DANGER_SAFE]

        lines = [f"将永久删除 {files} 个文件（{human_size(total)}）：", ""]
        lines += [f"· {i.label}　{human_size(i.size)}" for i in items[:8]]
        if len(items) > 8:
            lines.append(f"…另有 {len(items) - 8} 项")
        if gone:
            lines += ["", "⚠ 其中「回收站」里的会话删除后**无法恢复**。"]
        elif risky:
            lines += ["", "⚠ 其中「谨慎」项删了要重跑或重拍才能再生。"]
        lines += ["", "这些文件不进回收站（日志/缓存/临时文件的价值就是可再生）。",
                  "确定要删除吗？"]

        if not messagebox.askyesno(title, "\n".join(lines), parent=self):
            return

        deleted, freed, failures = clean_junk(JunkPlan(items=items), self.root)
        note = f"{title}：已删除 {deleted} 个文件，释放 {human_size(freed)}"
        if failures:
            note += f"；{len(failures)} 个没删掉（多为正被占用）：{failures[0]}"
        self._reload()          # 重扫会把 lbl_note 刷掉，所以下面再写一次
        self.lbl_note.configure(text=note, foreground=WARN if failures else MUTED)


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
