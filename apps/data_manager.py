"""数据管理面板：勾选会话、脱敏、按类别打成 zip。

运行：
    python apps/data_manager.py

也可以从主界面右上角『数据管理』打开。
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
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
    export_motion_pair,
    export_zip,
    normalize_output,
    scan_items,
)
from sphere_tracker.paths import STEREO_ROOT, ensure_dirs  # noqa: E402

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


# ----------------------------------------------------------------------
# 会话元数据编辑（改名 / 改备注）
#
# 与 `apps/_cleanup.py` 同一套安全习惯：只认 sessions 根目录的**直接子目录**，
# 拒绝 `..`、绝对路径与符号链接逃逸；写之前**必须**留一份备份，
# 再写同目录临时文件 `os.replace` 原子替换。**绝不**直接 `json.dump` 覆盖。
# ----------------------------------------------------------------------
#: 面板默认打开的双目会话根。从 `sphere_tracker/paths.py` 取而不是自己拼 ——
#: 之前这里写 `ROOT / "data" / "sessions"`，与另外 7 处各存一份，改名必漏。
#: 这一处只做默认值：用户点『数据目录…』或传 `--root` 仍可打开任意根（含旧位置）。
SESSIONS_ROOT = STEREO_ROOT
BAK_PREFIX = "session.json.bak-"


def _sessions_root(sessions_root: Path | str | None) -> Path:
    root = Path(sessions_root) if sessions_root is not None else SESSIONS_ROOT
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"sessions 根目录不存在：{root}；先用『数据目录…』确认位置")
    return root


def _resolve_session_dir(session_dir: Path | str,
                         sessions_root: Path | str | None = None) -> tuple[Path, Path]:
    """校验并返回 (会话目录, session.json)；越界/符号链接一律 ValueError。"""
    root = _sessions_root(sessions_root)
    raw = Path(session_dir)
    if raw.is_symlink():
        raise ValueError(f"拒绝编辑符号链接：{raw}")
    p = raw.resolve()
    if p == root or not p.is_relative_to(root):
        raise ValueError(f"只能编辑 {root} 下的会话，收到：{p}")
    if p.parent != root:
        raise ValueError(f"只支持 sessions 根目录的直接子目录（不许嵌套/`..`）：{p}")
    if not p.is_dir():
        raise ValueError(f"会话目录不存在：{p}")
    jf = p / "session.json"
    if jf.is_symlink():
        raise ValueError(f"拒绝编辑符号链接：{jf}")
    if not jf.is_file():
        raise ValueError(f"这条会话没有 session.json：{jf}")
    return p, jf


def latest_backup(session_dir: Path | str) -> Path | None:
    """返回该会话最近一次 `session.json.bak-*`，没有就 None。"""
    try:
        cands = sorted(Path(session_dir).glob(BAK_PREFIX + "*"),
                       key=lambda q: q.stat().st_mtime)
    except OSError:
        return None
    return cands[-1] if cands else None


def _backup_file(jf: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = jf.with_name(f"{BAK_PREFIX}{stamp}")
    n = 2
    while dest.exists():
        dest = jf.with_name(f"{BAK_PREFIX}{stamp}-{n}")
        n += 1
    try:
        shutil.copy2(jf, dest)
    except OSError as exc:
        raise OSError(
            f"备份失败，文件名未改动：{dest}（{exc}）；"
            "检查磁盘空间/文件是否被占用后重试") from exc
    return dest


def update_session_meta(session_dir: Path, *, label: str | None = None,
                        notes: str | None = None,
                        sessions_root: Path | str | None = None) -> dict:
    """改 `session.json` 的 label / notes，其余键原样保留。

    - `label=None` 表示不改；`label=""`（或全是空白）**拒绝**：会话必须有名字；
    - `notes=""` 合法：清空备注；
    - 写前备份成 `session.json.bak-<YYYYmmdd-HHMMSS>`，再原子替换；
    - 失败抛 ValueError/OSError，消息里带原因与下一步。
    """
    if label is not None:
        if not isinstance(label, str):
            raise ValueError(f"label 必须是字符串，收到 {type(label).__name__}")
        if not label.strip():
            raise ValueError("会话名称不能为空；填一个能认出来的名字再保存"
                             "（比如『金属球-左路虚』）")
    if notes is not None and not isinstance(notes, str):
        raise ValueError(f"notes 必须是字符串，收到 {type(notes).__name__}")

    _dir, jf = _resolve_session_dir(session_dir, sessions_root)
    try:
        text = jf.read_text(encoding="utf-8")
    except OSError as exc:
        raise OSError(f"读不了 {jf}：{exc}；文件未改动") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{jf} 不是合法 JSON（第 {exc.lineno} 行第 {exc.colno} 列：{exc.msg}）；"
            "文件未做任何改动，先修好它或从 session.json.bak-* 恢复再改") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{jf} 顶层不是 JSON 对象，拒绝改动")

    if label is not None:
        data["label"] = label
    if notes is not None:
        data["notes"] = notes

    backup = _backup_file(jf)
    tmp = jf.with_name(f"{jf.name}.tmp-{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2, allow_nan=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, jf)
    except (OSError, ValueError) as exc:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise type(exc)(
            f"写入 {jf} 失败：{exc}；原文件还在，备份在 {backup}") from exc
    return data


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
        self._last_focus: str | None = None
        self._running = False
        self._current_out: Path | None = None
        self._worker: threading.Thread | None = None
        self._closing = False
        self._cancel = threading.Event()
        self._cat_vars: dict[str, tk.BooleanVar] = {}
        self._redact_vars: dict[str, tk.BooleanVar] = {}
        self._ui_queue: queue.Queue = queue.Queue()
        # 列表筛选：状态下拉 + 搜索框（只改显示，不动勾选状态）
        self._status_filter: str = "全部"
        self._search_text: str = ""
        self._build()
        self.refresh()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_after_id = self.after(100, self._poll_ui)

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
        # 一个按钮同时做两件事：单击=刷新当前目录，右键=换数据目录。
        # 之前「刷新」「数据目录…」两个按钮语义重叠（换目录后本来就自动刷新），
        # 合并成一个，减少"到底点哪个"的困惑。
        self.head_refresh = ttk.Button(head, text="刷新数据目录",
                                       command=self.refresh)
        self.head_refresh.pack(side="right", padx=(4, 12), pady=8)
        self.head_refresh.bind("<Button-3>", lambda _e: self.choose_root())
        # 搜索框（实时匹配会话 id / 名称 label），放头部右侧，和刷新数据目录并列
        ttk.Label(head, text="搜索", background=ACCENT, foreground="white").pack(
            side="right", padx=(4, 2), pady=8)
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *_a: self._apply_filters())
        ttk.Entry(head, textvariable=self.search_var, width=20).pack(
            side="right", pady=8)

        toolbar = ttk.Frame(self, padding=(12, 8))
        toolbar.pack(fill="x")
        # 状态筛选下拉（全部 / 各状态符号）
        ttk.Label(toolbar, text="状态").pack(side="left")
        self.status_filter_var = tk.StringVar(value="全部")
        status_options = ["全部", "✓ 有3D", "◐ 已点选", "○ 已录制", "✗ 坏", "— 空"]
        ttk.Combobox(toolbar, textvariable=self.status_filter_var,
                     values=status_options, state="readonly", width=10).pack(
            side="left", padx=(2, 10))
        self.status_filter_var.trace_add("write", lambda *_a: self._apply_filters())
        ttk.Button(toolbar, text="全选", command=self.select_all).pack(side="left")
        ttk.Button(toolbar, text="全不选", command=self.select_none).pack(side="left", padx=4)
        ttk.Button(toolbar, text="反选", command=self.invert).pack(side="left", padx=4)
        ttk.Button(toolbar, text="选到此处", command=self.select_to_here).pack(side="left", padx=4)
        ttk.Button(toolbar, text="设为起点", command=self.set_anchor).pack(side="left", padx=4)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Label(toolbar, text="从").pack(side="left")
        self.from_var = tk.StringVar()
        self.from_box = ttk.Combobox(toolbar, textvariable=self.from_var, width=16)
        self.from_box.pack(side="left", padx=(2, 4))
        ttk.Label(toolbar, text="到").pack(side="left")
        self.to_var = tk.StringVar()
        self.to_box = ttk.Combobox(toolbar, textvariable=self.to_var, width=16)
        self.to_box.pack(side="left", padx=(2, 4))
        ttk.Button(toolbar, text="按时间选择", command=self.select_time).pack(side="left")
        ttk.Button(toolbar, text="清除时间", command=self.clear_time).pack(side="left", padx=4)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Button(toolbar, text="编辑名称/备注…", command=self._edit_selected).pack(
            side="left")
        ttk.Label(toolbar, text="日期可下拉选，或手填 YYYY-MM-DD",
                  foreground=MUTED).pack(side="left", padx=8)

        # 垂直 PanedWindow：上=主区（左列表+右面板），下=导出日志。
        # 两条可拖分隔线（类似 VSCode）：主区内部那条拖左右，这条拖上下。
        outer = ttk.PanedWindow(self, orient="vertical")
        outer.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        body = ttk.PanedWindow(outer, orient="horizontal")

        left = ttk.Frame(body)
        right = ttk.Frame(body)

        cols = ("sel", "id", "name", "created", "frames", "mode", "status", "size")
        heads = ("选", "会话", "名称", "时间", "帧", "模式", "状态", "大小")
        widths = (34, 150, 150, 140, 70, 66, 46, 90)
        self.tree = ttk.Treeview(left, columns=cols, show="headings",
                                 selectmode="extended", height=18)
        for c, h, w in zip(cols, heads, widths):
            self.tree.heading(c, text=h)
            self.tree.column(c, width=w, minwidth=w, anchor="w",
                             stretch=(c in {"id", "name", "created"}))
        self.tree.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        sb.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.bind("<Button-1>", self._tree_click)
        self.tree.bind("<Double-1>", lambda _e: self._open_viewer())

        tip = ttk.Label(left, text="点第一列勾选；勾起点→点终点→『选到此处』；Shift+点行也可区间。",
                        foreground=MUTED)
        tip.pack(side="bottom", anchor="w", pady=(4, 0))

        self.count_label = ttk.Label(right, text="", font=("Microsoft YaHei UI", 10, "bold"))
        self.count_label.pack(anchor="w", pady=(0, 6))
        ttk.Label(right, text="状态：✓ 有3D   ◐ 已点选   ○ 已录制   ✗ 坏   — 空",
                  foreground=MUTED, wraplength=310).pack(anchor="w", pady=(0, 6))

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

        self.export_btn = ttk.Button(right, text="开始导出 (zip)", command=self.start_export)
        self.export_btn.pack(fill="x", pady=(2, 4))
        # 「处理前 + 处理后」一次导出（2026-11-03）：对每个**已跑过三维**的选中会话，
        # 在输出目录下成对写出 `<会话>_raw.csv`（原始轨迹，逐字节复制）与
        # `<会话>_motion.csv`（平滑 + 速度/加速度）及 `数据说明.md`。不依赖 zip 与类别勾选。
        self.export_pair_btn = ttk.Button(
            right, text="导出轨迹数据 (csv)", command=self._export_motion_pair)
        self.export_pair_btn.pack(fill="x", pady=(0, 4))
        ttk.Button(right, text="查看 3D + 视频", command=self._open_viewer).pack(
            fill="x", pady=(0, 4))
        clean_bar = ttk.Frame(right)
        clean_bar.pack(fill="x", pady=(0, 4))
        ttk.Button(clean_bar, text="清理…", command=self._clean_selected).pack(
            side="left", fill="x", expand=True)
        ttk.Button(clean_bar, text="回收站…", command=self._open_trash).pack(
            side="left", fill="x", expand=True, padx=(6, 0))
        self.progress = ttk.Progressbar(right, mode="determinate")
        self.progress.pack(fill="x", pady=(0, 4))
        self.status = ttk.Label(right, text="等待选择", foreground=MUTED,
                                wraplength=310, justify="left")
        self.status.pack(anchor="w")

        log_box = ttk.LabelFrame(outer, text="导出日志", padding=4)
        self.log = tk.Text(log_box, height=10, wrap="none")
        self.log.pack(side="left", fill="both", expand=True)
        lsb = ttk.Scrollbar(log_box, orient="vertical", command=self.log.yview)
        lsb.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=lsb.set)

        # 内容都建好之后再 add 到两个 PanedWindow（ttk 的 pane 只支持 weight，
        # 不支持 minsize；拖动由内置 sash 手柄完成）。
        body.add(left, weight=4)
        body.add(right, weight=1)
        outer.add(body, weight=4)
        outer.add(log_box, weight=1)

    def _default_output(self) -> str:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        # zip 导出：exports/detailed/<时间戳文件夹>/xxx.zip —— 每次导出自己成一层
        return str(ROOT / "outputs" / "exports" / "detailed"
                   / f"export-{stamp}" / f"sphere-stereo-export-{stamp}.zip")

    def _default_traj_output(self) -> str:
        """「导出轨迹数据 (csv)」：exports/trajectory/<时间戳文件夹>/，每次导出一层。"""
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return str(ROOT / "outputs" / "exports" / "trajectory" / f"trajectory-{stamp}")

    # ------------------------------------------------------------------
    def _status_symbol(self, it: SessionItem) -> str:
        """把状态下拉的值映射回 STATUS_SYMBOLS 的键，用于过滤。"""
        return {
            "✓ 有3D": "traj",
            "◐ 已点选": "seeded",
            "○ 已录制": "recorded",
            "✗ 坏": "bad",
            "— 空": "empty",
        }.get(self._status_filter)

    def _visible_items(self) -> list[SessionItem]:
        """当前筛选 + 搜索之后应该显示在列表里的项（顺序与 self.items 一致）。"""
        want_symbol = self._status_symbol(self._status_filter)
        q = self.search_var.get().strip().lower()
        out: list[SessionItem] = []
        for it in self.items:
            if want_symbol is not None and it.symbol != want_symbol:
                continue
            if q:
                name = self._session_name(it).lower()
                if q not in it.id.lower() and q not in name:
                    continue
            out.append(it)
        return out

    def _apply_filters(self) -> None:
        """按状态 + 搜索重画列表。只改显示，勾选状态原样保留。"""
        self._status_filter = self.status_filter_var.get()
        self._search_text = self.search_var.get()
        visible = {it.key for it in self._visible_items()}
        self.tree.delete(*self.tree.get_children())
        for it in self.items:
            if it.key not in visible:
                continue
            size = _fmt_bytes(it.size_bytes)
            self.tree.insert("", "end", iid=it.key,
                             values=("☑" if it.key in self.selected else "☐",
                                     it.id, self._session_name(it),
                                     it.created,
                                     f"{it.frames_left}/{it.frames_right}",
                                     "大角度" if it.stereo_mode == "angled" else "平行",
                                     it.symbol, size))
        self._refresh_count()

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        try:
            self.items = scan_items(self.root_dir)
        except Exception as exc:                            # noqa: BLE001
            messagebox.showerror("扫描失败", str(exc), parent=self)
            return
        valid = {it.key for it in self.items}
        self.selected &= valid
        self._fill_date_options()
        self._apply_filters()

    def _fill_date_options(self) -> None:
        """把会话里实际出现的日期填进「从/到」下拉（去重、升序），方便点选。"""
        dates: set[str] = set()
        for it in self.items:
            c = (it.created or "").strip()
            if len(c) >= 10:
                dates.add(c[:10])          # YYYY-MM-DD 部分
        ordered = sorted(dates)
        self.from_box.configure(values=ordered)
        self.to_box.configure(values=ordered)

    @staticmethod
    def _session_name(it: SessionItem) -> str:
        """列表里显示 session.json 的 label；坏文件就留空，不挡住列表。"""
        try:
            meta = json.loads((it.path / "session.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ""
        label = meta.get("label") if isinstance(meta, dict) else None
        return label if isinstance(label, str) else ""

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
        shift = bool(event.state & 0x0001)
        if shift:
            start = self._anchor_id or self._last_focus
            if start:
                self.tree.focus(row)
                self._anchor_id = start
                self._select_range(start, row)
            else:
                self._anchor_id = row
                self.tree.focus(row)
                self._last_focus = row
            return "break"
        if col == "#1":
            self.tree.focus(row)
            self._last_focus = row
            self.toggle_key(row)
            self._anchor_id = row
            return "break"
        # 普通点行只移动焦点，不改区间起点；起点由 checkbox / 『设为起点』决定
        self.tree.focus(row)
        self._last_focus = row
        return None

    def toggle_focus(self) -> None:
        row = self.tree.focus()
        if row:
            self._last_focus = row
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
        # 只作用于当前**可见**项：筛选/搜索后"全选"不该把被隐藏的也勾上
        self.selected.update(it.key for it in self._visible_items())
        self._anchor_id = None
        self._refresh_count()

    def select_none(self) -> None:
        self.selected.clear()
        self._anchor_id = None
        self._refresh_count()

    def invert(self) -> None:
        # 反选同样只针对可见项，避免"看不见的也被翻转"
        visible = {it.key for it in self._visible_items()}
        self.selected = (self.selected - visible) | (visible - self.selected)
        self._anchor_id = None
        self._refresh_count()

    def set_anchor(self) -> None:
        row = self.tree.focus()
        if not row:
            messagebox.showinfo("设为起点", "先点一行作为起点。", parent=self)
            return
        self._anchor_id = row
        self._last_focus = row
        self.status.configure(text=f"区间起点：{row}")

    def select_to_here(self) -> None:
        row = self.tree.focus()
        if not row:
            messagebox.showinfo("选到此处", "先点一行作为终点。", parent=self)
            return
        if not self._anchor_id:
            messagebox.showinfo("选到此处", "先点第一列勾起点，或先按『设为起点』。",
                                parent=self)
            return
        self._select_range(self._anchor_id, row)

    def select_time(self) -> None:
        raw_from = self.from_var.get().strip()
        raw_to = self.to_var.get().strip()
        start = _parse_dt(raw_from)
        end = _parse_dt(raw_to, end=True)
        if raw_from and start is None:
            messagebox.showwarning("按时间选择", f"开始时间解析不了：{raw_from}",
                                   parent=self)
            return
        if raw_to and end is None:
            messagebox.showwarning("按时间选择", f"结束时间解析不了：{raw_to}",
                                   parent=self)
            return
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
        self._anchor_id = None
        self._refresh_count()

    def clear_time(self) -> None:
        self.from_var.set("")
        self.to_var.set("")

    # ------------------------------------------------------------------
    def choose_root(self) -> None:
        path = filedialog.askdirectory(
            parent=self, title="选择数据目录（放会话的直接父目录，默认 data/stereo）",
            initialdir=str(self.root_dir))
        if not path:
            return
        root = Path(path)
        if not root.is_dir():
            messagebox.showerror("数据目录", f"不是目录：{root}", parent=self)
            return
        self.root_dir = root
        self.selected.clear()
        self.refresh()

    def _focused_item(self) -> SessionItem | None:
        row = self.tree.focus()
        if row:
            return next((it for it in self.items if it.key == row), None)
        if len(self.selected) == 1:
            key = next(iter(self.selected))
            return next((it for it in self.items if it.key == key), None)
        return None

    def _open_viewer(self) -> None:
        it = self._focused_item()
        if it is None:
            messagebox.showinfo("3D + 视频", "先在列表里选中一条会话。", parent=self)
            return
        viz = it.path / "derived" / "visualization.json"
        traj = it.path / "derived" / "traj3d.json"
        if not viz.exists() and not traj.exists():
            messagebox.showinfo("3D + 视频", f"这条会话还没有可读的三维数据：\n{it.path}",
                                parent=self)
            return
        annotated = it.path / "derived" / "annotated.mp4"
        args = [sys.executable, str(ROOT / "apps" / "view_3d.py"), str(it.path)]
        if not annotated.exists():
            if not messagebox.askyesno(
                    "没有标注视频", "这条会话没有 annotated.mp4。\n只打开 3D 查看器？",
                    parent=self):
                return
            args.append("--no-video")
        try:
            subprocess.Popen(args)
        except Exception as exc:                            # noqa: BLE001
            messagebox.showerror("3D + 视频", f"打不开查看器：{exc}", parent=self)
            return
        self._log_line(f"打开 3D + 视频：{it.id}")

    def _open_trash(self) -> None:
        from apps.cleanup_dialog import TrashDialog
        TrashDialog(self, self.root_dir, on_done=self.refresh)

    def _clean_selected(self) -> None:
        from apps.cleanup_dialog import CleanupDialog
        chosen = [it for it in self.items if it.key in self.selected]
        if not chosen:
            it = self._focused_item()
            chosen = [it] if it else []
        if not chosen:
            messagebox.showinfo("清理", "先勾选或选中至少一个会话。", parent=self)
            return
        CleanupDialog(self, self.root_dir, [it.path for it in chosen],
                      on_done=self.refresh)

    # ------------------------------------------------------------------
    def _edit_selected(self) -> None:
        it = self._focused_item()
        if it is None:
            self.status.configure(text="先在列表里选一个会话")
            return
        jf = it.path / "session.json"
        try:
            meta = json.loads(jf.read_text(encoding="utf-8"))
            if not isinstance(meta, dict):
                raise ValueError("session.json 顶层不是 JSON 对象")
        except (OSError, ValueError) as exc:
            self.status.configure(text=f"读不了 {it.id}/session.json：{exc}")
            messagebox.showerror("编辑名称/备注", f"{jf}\n\n{exc}", parent=self)
            return
        cur_label = meta.get("label") if isinstance(meta.get("label"), str) else ""
        cur_notes = meta.get("notes") if isinstance(meta.get("notes"), str) else ""

        dlg = tk.Toplevel(self)
        dlg.title(f"编辑名称/备注 · {it.id}")
        dlg.transient(self)
        dlg.resizable(False, False)
        body = ttk.Frame(dlg, padding=14)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text=f"会话 {it.id}（{it.path}）", foreground=MUTED,
                  wraplength=430).pack(anchor="w", pady=(0, 8))
        ttk.Label(body, text="名称（必填）").pack(anchor="w")
        label_var = tk.StringVar(value=cur_label)
        label_entry = ttk.Entry(body, textvariable=label_var, width=52)
        label_entry.pack(fill="x", pady=(2, 8))
        ttk.Label(body, text="备注（可留空）").pack(anchor="w")
        notes_var = tk.StringVar(value=cur_notes)
        ttk.Entry(body, textvariable=notes_var, width=52).pack(fill="x", pady=(2, 8))
        ttk.Label(body, text="保存前自动备份成 session.json.bak-<时间>。",
                  foreground=MUTED).pack(anchor="w", pady=(0, 8))
        bar = ttk.Frame(body)
        bar.pack(fill="x")

        def save() -> None:
            try:
                update_session_meta(it.path, label=label_var.get(),
                                    notes=notes_var.get(),
                                    sessions_root=self.root_dir)
            except (ValueError, OSError) as exc:
                # 原始原因照实显示，不吞
                self.status.configure(text=f"没改成 {it.id}：{exc}")
                messagebox.showerror("编辑名称/备注", str(exc), parent=dlg)
                return
            bak = latest_backup(it.path)
            dlg.destroy()
            self.refresh()
            self.tree.focus(it.key)
            self.status.configure(
                text=f"已更新 {it.id}（备份：{bak.name if bak else '未找到'}）")
            self._log_line(f"改元数据：{it.id}  名称={label_var.get()!r}")

        ttk.Button(bar, text="确定", command=save).pack(side="right")
        ttk.Button(bar, text="取消", command=dlg.destroy).pack(side="right", padx=(0, 6))
        label_entry.focus_set()
        dlg.grab_set()

    # ------------------------------------------------------------------
    def browse_output(self) -> None:
        # initialdir 用固定的 detailed 目录（先建好，避免被静默降级）。
        exports_dir = ROOT / "outputs" / "exports" / "detailed"
        exports_dir.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            parent=self, title="导出 zip", defaultextension=".zip",
            filetypes=[("ZIP", "*.zip")],
            initialdir=str(exports_dir),
            initialfile=Path(self.out_var.get()).name)
        if path:
            self.out_var.set(path)

    def _log_line(self, text: str) -> None:
        self.log.insert("end", text + "\n")
        self.log.see("end")

    def _export_motion_pair(self) -> None:
        """「导出轨迹数据 (csv)」：弹窗选目录，取消/失败则落默认目录。"""
        if self._running:
            self.status.configure(text="正在导出，请先等上一个任务结束")
            return
        chosen = [it for it in self.items if it.key in self.selected]
        if not chosen:
            messagebox.showinfo("导出轨迹", "先勾选至少一个会话。", parent=self)
            return
        # 默认目录（exports/trajectory/<时间戳>/），弹窗取消/异常时兜底用它。
        default_dir = Path(self._default_traj_output())
        # 弹窗 initialdir 指到固定的 trajectory 目录（父级），让用户在里面选/建文件夹
        traj_root = ROOT / "outputs" / "exports" / "trajectory"
        traj_root.mkdir(parents=True, exist_ok=True)
        out_dir = default_dir
        try:
            picked = filedialog.askdirectory(
                parent=self, title="选择轨迹数据导出目录（取消则用默认目录）",
                initialdir=str(traj_root), mustexist=False)
        except Exception:                                   # noqa: BLE001
            picked = ""          # 弹窗交互失败（无显示/后端异常）→ 走默认
        if picked:
            out_dir = Path(picked)
        else:
            self._log_line(f"未选目录，落到默认目录：{out_dir}")
        done, skipped = [], []
        for it in chosen:
            try:
                pair = export_motion_pair(it.path, out_dir)
            except Exception as exc:                     # noqa: BLE001
                self._log_line(f"[跳过] {it.id}：{exc}")
                skipped.append(it.id)
                continue
            if pair is None:
                self._log_line(f"[跳过] {it.id}：还没跑过三维（没有 traj3d.csv）")
                skipped.append(it.id)
                continue
            self._log_line(f"[导出] {it.id} → {pair[0].name} / {pair[1].name}")
            done.append(it.id)
        self.status.configure(
            text=f"已导出 {len(done)} 个会话的轨迹数据（原始+平滑 csv + 数据说明）到 {out_dir}"
                 + (f"；跳过 {len(skipped)} 个（未跑三维）" if skipped else ""))
        if done:
            self._log_line(f"共 {len(done)} 个会话，输出目录：{out_dir}")

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
                parent=self, default=messagebox.NO):
            return
        try:
            out = normalize_output(self.out_var.get())
        except ValueError as exc:
            messagebox.showerror("输出路径", str(exc), parent=self)
            return
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
        self._closing = False
        self._cancel.clear()
        self._current_out = out
        self.export_btn.configure(state="disabled")
        self.progress["value"] = 0
        self.status.configure(text="导出中…")
        self._log_line(f"开始导出 {len(chosen)} 个会话 → {out}")

        def work() -> None:
            def report(done: int, total: int, name: str) -> None:
                if self._cancel.is_set():
                    raise RuntimeError("用户取消导出")
                self._ui_queue.put(("progress", done, total, name))

            try:
                result = export_zip(chosen, opts, out, progress=report)
            except Exception as exc:                        # noqa: BLE001
                if self._cancel.is_set():
                    return
                self._ui_queue.put(("error", exc))
            else:
                if not self._cancel.is_set():
                    self._ui_queue.put(("done", result))

        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()

    def _cleanup_partial(self) -> None:
        if self._current_out is not None:
            part = self._current_out.with_name(self._current_out.name + ".part")
            try:
                if part.exists():
                    part.unlink()
            except OSError:
                pass

    def _on_close(self) -> None:
        # 先停轮询再决定关不关：`_poll_ui` 每 100 ms 自己续一次，窗口一旦 destroy，
        # 那条已经排进队列的回调就会 TclError（`invalid command name "..._poll_ui"`），
        # 而 Tk 把它打到 stderr —— `selftest_launcher_params` 会因此偶发判红。
        self._stop_polling()
        if not self._running:
            self.destroy()
            return
        if not messagebox.askyesno(
                "退出", "正在导出。退出会取消导出，等当前文件写完再关闭。继续？",
                parent=self):
            return
        self._cancel.set()
        self._closing = True
        self.export_btn.configure(state="disabled")
        self.status.configure(text="正在取消导出，等待当前文件收尾…")
        self._wait_close()

    def _wait_close(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            self.after(100, self._wait_close)
            return
        self._cleanup_partial()
        self.destroy()

    def _stop_polling(self) -> None:
        """取消尚未触发的 `_poll_ui`（幂等：重复调用无副作用）。"""
        poll_id = getattr(self, "_poll_after_id", None)
        if poll_id is None:
            return
        try:
            self.after_cancel(poll_id)
        except tk.TclError:
            pass                      # 已经执行过/窗口已销毁
        self._poll_after_id = None

    def _poll_ui(self) -> None:
        """后台线程不能碰 Tk；只把消息放进队列，这里在主线程消费。"""
        # 窗口已在关闭流程里：不再消费、也不再续（续了就是那条 TclError 的来源）。
        if self._closing or not self.winfo_exists():
            return
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
            self._poll_after_id = self.after(100, self._poll_ui)
        except tk.TclError:
            pass

    def _on_progress(self, done: int, total: int, name: str) -> None:
        self.progress["maximum"] = max(total, 1)
        self.progress["value"] = done
        self.status.configure(text=f"{done}/{total}  {name}")
        self._log_line(name)

    def _on_error(self, exc: Exception) -> None:
        self._running = False
        self._worker = None
        if self._closing:
            return
        self._cleanup_partial()
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
        self._worker = None
        if self._closing:
            return
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
    ap.add_argument("--root", default=str(STEREO_ROOT))
    args = ap.parse_args()
    # 新布局的根可能还没建（第一次跑 / 还没迁移）—— 面板不该因为"目录未创建"打不开
    ensure_dirs()
    app = DataManager(Path(args.root))
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
