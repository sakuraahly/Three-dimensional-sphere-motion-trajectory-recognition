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
import subprocess
import sys
import threading
import time as _time
from dataclasses import replace
from datetime import datetime, time
from pathlib import Path
from types import SimpleNamespace
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

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
)
from apps._open_in_explorer import (  # noqa: E402
    OpenInExplorerError,
    open_in_explorer,
)
from apps._common import (  # noqa: E402
    atomic_write_with_backup,
    backup_glob,
)
from apps._task_registry import register_end, register_start  # noqa: E402
from apps._session_index import (  # noqa: E402
    ROOT_NODE,
    UNFILED_LABEL,
    all_group_paths,
    all_node_paths,
    build_index,
    collection_map,
    collection_paths_by_node,
    group_of,
    load_collections,
    new_node_id,
    node_children,
    paths_for_root,
    save_collections,
    save_index,
)
from sphere_tracker.paths import STEREO_ROOT, ensure_dirs  # noqa: E402
# 「有没有原始视频」与查看器/导出的判据必须是同一条：文件名清单与 `is_file()`
# 口径都在 `sphere_tracker/video_files.py`，不在这里再写一份元组。
from sphere_tracker.video_files import has_any_raw_video  # noqa: E402

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


def mode_cell_text(it: SessionItem) -> str:
    """会话表「模式」列文案：冲突行追加**可见提示**（纯数据层，可单测）。

    用户裁决（2026-11-10）：`data/stereo/parallel/20261104-140552` 记录 `angled`
    却摆在 `parallel/`，**不改 `session.json`、不搬目录、不改 `index.json` 里
    `group` 的推导口径** —— 只在界面说清"目录与记录不一致"，要不要搬由用户定。

    判据与 `apps._session_index.build_index` 的 `mode_conflict` 同源
    （记录优先、且必须有记录 + 有布局才叫冲突），不另写一份口径。
    """
    from apps._stereo_layout import layout_mode_of

    layout = layout_mode_of(it.path)
    if it.stereo_mode and layout and layout != it.stereo_mode:
        return f"{it.mode_label} ⚠ 目录={layout}，记录={it.stereo_mode}"
    return it.mode_label


def name_cell_text(it: SessionItem) -> str:
    """会话表「名称」列文案：实时采集的会话加**可见前缀**（纯数据层，可单测）。

    为什么加前缀而不是加一列（2026-11-17 用户要求"实时双目也要能在数据管理里
    看到"）：`cols` 有 8 列、宽度是按这个列数分好的（见 `_build_table`），
    多一列要么挤掉「名称」要么触发横向滚动 —— 而"能看出这是实时采集的"只要
    一个字的前缀就够了。前缀也不改 `label` 本身：导出、`index.json`、分组
    与排序仍按原名走；只有列表显示和搜索（`_visible_items`）用这一格的字，
    否则"屏幕上写着「实时 ·」却搜不到「实时」"是自相矛盾的。
    """
    if it.source == "realtime":
        return f"实时 · {it.label}"
    return it.label


# ----------------------------------------------------------------------
# 会话元数据编辑（改名 / 改备注）
#
# 与 `apps/_cleanup.py` 同一套安全习惯：只认 sessions 根下的**会话目录**
# （直接子目录，或一层装置分区下的会话 —— 见 `_stereo_layout`），
# 拒绝 `..`、绝对路径与符号链接逃逸；写之前**必须**留一份备份，
# 再写同目录临时文件 `os.replace` 原子替换。**绝不**直接 `json.dump` 覆盖。
# ----------------------------------------------------------------------
#: 面板默认打开的双目会话根。从 `sphere_tracker/paths.py` 取而不是自己拼 ——
#: 之前这里写 `ROOT / "data" / "sessions"`，与另外 7 处各存一份，改名必漏。
#: 这一处只做默认值：用户点『数据目录…』或传 `--root` 仍可打开任意根（含旧位置）。
SESSIONS_ROOT = STEREO_ROOT


def _sessions_root(sessions_root: Path | str | None) -> Path:
    root = Path(sessions_root) if sessions_root is not None else SESSIONS_ROOT
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"sessions 根目录不存在：{root}；先用『数据目录…』确认位置")
    return root


def _resolve_session_dir(session_dir: Path | str,
                         sessions_root: Path | str | None = None) -> tuple[Path, Path]:
    """校验并返回 (会话目录, session.json)；越界/符号链接一律 ValueError。

    允许两种层级（2026-11-04，会话按装置分区后）：

    - ``<根>/<时间戳>``           旧扁平布局；
    - ``<根>/<模式>/<时间戳>``    新布局（``data/stereo/parallel/<时间戳>``）。

    两级都必须真的是"会话"（有 ``session.json``）。中间层用
    ``_stereo_layout.layout_mode_of`` 判定，**不靠层数** —— 否则
    ``<根>/a/b`` 这种任意嵌套也会被放进来。
    """
    root = _sessions_root(sessions_root)
    raw = Path(session_dir)
    if raw.is_symlink():
        raise ValueError(f"拒绝编辑符号链接：{raw}")
    p = raw.resolve()
    if p == root or not p.is_relative_to(root):
        raise ValueError(f"只能编辑 {root} 下的会话，收到：{p}")
    if p.parent != root:
        from apps._stereo_layout import layout_mode_of

        if p.parent.parent != root or layout_mode_of(p) is None:
            raise ValueError(
                f"只支持 {root} 的直接子目录，或一层装置分区"
                f"（parallel/angled）下的会话：{p}")
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
        # glob 格式与写备份同源（`apps._common.backup_glob`），不在这里再拼一份。
        cands = sorted(Path(session_dir).glob(
            backup_glob(Path(session_dir) / "session.json")),
            key=lambda q: q.stat().st_mtime)
    except OSError:
        return None
    return cands[-1] if cands else None


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

    # 备份 / 原子写委托给唯一实现；keep_backups=None = 写时不裁剪，
    # 回滚点统一由 apps/_junk.py 在清理阶段按 3 份保留（既有行为不变）。
    atomic_write_with_backup(jf, data, keep_backups=None)
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
        # 跨进程任务账本：本进程是独立进程，父进程的 TaskRunner 看不见这里的
        # 导出。tid 从 1001 起，避开 runner 从 1 起的编号（key 含 PID，不会撞）。
        self._ext_tid = 1000
        # 开出去的查看器：原来是 fire-and-forget，关数据管理后成了真孤儿
        # （占着资源、用户不知道）—— 这里留句柄，关窗时一并收掉。
        self._viewer_procs: list[subprocess.Popen] = []
        self._closing = False
        self._cancel = threading.Event()
        self._cat_vars: dict[str, tk.BooleanVar] = {}
        self._redact_vars: dict[str, tk.BooleanVar] = {}
        self._ui_queue: queue.Queue = queue.Queue()
        # 列表筛选：状态下拉 + 搜索框（只改显示，不动勾选状态）
        self._status_filter: str = "全部"
        self._search_text: str = ""
        # 分组过滤：None=全部；""=根层散会话；其余=逻辑路径（含子目录）
        self._filter_group: str | None = None
        self._syncing_group = False
        # 第一次建树时把一级目录默认展开；之后一律按用户的展开状态恢复
        self._group_tree_ready = False
        # 集合配置与索引放在**数据根的同级**：换根时必须跟着换（见 choose_root）
        self.collections_path, self.index_path = paths_for_root(self.root_dir)
        self.collections, _w = load_collections(self.collections_path)
        self._build()
        self.refresh()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        # 直接 `destroy()` 的路径（测试、或用户从任务栏关窗）不走 `_on_close`，
        # 那条已经排进队列的 `_poll_ui` 会在窗口没了之后触发，Tk 把
        # `invalid command name "..._poll_ui"` 打到 stderr（实测 2026-11-09
        # 跑 `selftest_launcher_params` 时就撞到，属偶发判红的来源）。
        # 挂在 `<Destroy>` 上就与关闭方式无关：窗口没了就撤掉续约。
        self.bind("<Destroy>", self._on_destroy, add="+")
        self._poll_after_id = self.after(100, self._poll_ui)

    def _on_destroy(self, event=None) -> None:
        """窗口销毁：撤掉尚未触发的轮询（幂等）。只在**本窗口**的销毁事件上动作。"""
        if event is not None and getattr(event, "widget", None) is not self:
            return
        self._stop_polling()
        self._terminate_viewers()

    def _terminate_viewers(self) -> None:
        """关数据管理时收掉它开出去的查看器。

        原来是 ``Popen`` 后 fire-and-forget：关掉数据管理，查看器还活着，
        没人知道它在占资源。这里只 terminate **还活着**的那些（已退出的
        ``poll()`` 非 None，跳过），重复调用安全。
        """
        for proc in getattr(self, "_viewer_procs", []):
            # 测试里 `subprocess.Popen` 会被打桩成 None；真实路径不会，
            # 但空句柄不该让关窗回调抛 AttributeError。
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                except OSError:
                    pass
        self._viewer_procs = []

    def _next_ext_tid(self) -> int:
        """本进程私有的任务号（1001 起）；配合 PID 才是全局唯一 key。"""
        self._ext_tid += 1
        return self._ext_tid

    def _registry_task(self, tid: int, title: str) -> SimpleNamespace:
        """账本要的最小任务视图 —— 不为了它去造一个假的 ``Task``。"""
        return SimpleNamespace(
            proc=SimpleNamespace(pid=os.getpid()), tid=tid, title=title,
            owner="data_manager", started=_time.time(), ended=None,
            exit_code=None, stop_reason="")

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
        status_options = ["全部", "✓ 有3D", "◑ 三维点过少", "◐ 已点选", "○ 已录制", "✗ 坏", "— 空"]
        # width=13：`◑ 三维点过少` 是 5 个汉字 + 符号 + 空格，ttk 里汉字算 2 个宽度，
        # 原来的 10 会把末尾裁掉（弹层仍完整，但收起时看不全就等于没有筛选项）。
        ttk.Combobox(toolbar, textvariable=self.status_filter_var,
                     values=status_options, state="readonly", width=13).pack(
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

        # 左侧再分成两栏：分组目录树 | 会话表（像资源管理器：逐层展开，
        # 右边永远是这批会话的时间倒序表）。目录树只占 ~190px，把宽度留给 8 列表格。
        left = ttk.PanedWindow(body, orient="horizontal")
        self._left_pane = left
        tree_pane = ttk.Frame(left)
        table_pane = ttk.Frame(left)
        right = ttk.Frame(body)

        # ---- 分组目录树（虚拟目录，存在 collections.json，不动磁盘）----
        tree_head = ttk.Frame(tree_pane)
        tree_head.pack(side="top", fill="x", pady=(0, 4))
        ttk.Label(tree_head, text="分组目录",
                  font=("Microsoft YaHei UI", 9, "bold")).pack(side="left")

        group_btns = ttk.Frame(tree_pane)
        group_btns.pack(side="bottom", fill="x", pady=(4, 0))
        ttk.Button(group_btns, text="＋新建", width=6,
                   command=self._group_new).pack(side="left")
        ttk.Button(group_btns, text="改名", width=5,
                   command=self._group_rename).pack(side="left", padx=(4, 0))
        ttk.Button(group_btns, text="删除", width=5,
                   command=self._group_delete).pack(side="left", padx=(4, 0))

        self.group_tree = ttk.Treeview(tree_pane, show="tree",
                                       selectmode="browse")
        self.group_tree.column("#0", width=180, minwidth=120, stretch=True)
        gsb = ttk.Scrollbar(tree_pane, orient="vertical",
                            command=self.group_tree.yview)
        gsb.pack(side="right", fill="y")
        self.group_tree.pack(side="left", fill="both", expand=True)
        self.group_tree.configure(yscrollcommand=gsb.set)
        self.group_tree.bind("<<TreeviewSelect>>", self._on_group_select)
        self.group_tree.bind("<Button-3>", self._group_right_click)
        self._build_group_menu()

        cols = ("sel", "id", "name", "created", "frames", "mode", "status", "size")
        heads = ("选", "会话", "名称", "时间", "帧", "模式", "状态", "大小")
        # 「模式」列放宽到 160：冲突行要在文案后追加「⚠ 目录=…，记录=…」这份
        # 可见提示，66px 会把 ⚠ 之后的字全裁掉，等于没提示。
        widths = (34, 150, 150, 140, 70, 160, 46, 90)
        self.tree = ttk.Treeview(table_pane, columns=cols, show="headings",
                                 selectmode="extended", height=18)
        for c, h, w in zip(cols, heads, widths):
            self.tree.heading(c, text=h)
            self.tree.column(c, width=w, minwidth=w, anchor="w",
                             stretch=(c in {"id", "name", "created"}))

        # ⚠️ pack 是**按顺序分地**的：先把表格 `expand=True` 放进去，它会把整个
        # 高度吃掉，后面 `side="bottom"` 的提示行只能拿到残渣 —— 实测文字被裁成
        # 「多选 / 区间」，「? 操作说明」按钮整个看不见（既有缺陷，2026-11-09）。
        # 所以**先** pack 底部提示行，再 pack 表格与滚动条。
        #
        # 选中方法的完整说明搬到「操作说明 · 数据管理」了（2026-11-05 用户：
        # "很多界面的操作说明文字就可以删掉进而不会影响界面排布"）。
        # 这里只留一句极短提示 + 一键入口，把宽度让给列表。
        tip_row = ttk.Frame(table_pane)
        tip_row.pack(side="bottom", fill="x", pady=(4, 0))
        ttk.Label(tip_row, text="多选 / 区间选择的方法：", foreground=MUTED).pack(
            side="left")
        ttk.Button(tip_row, text="? 操作说明", width=10,
                   command=self._open_help).pack(side="left", padx=(4, 0))

        sb = ttk.Scrollbar(table_pane, orient="vertical", command=self.tree.yview)
        sb.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.bind("<Button-1>", self._tree_click)
        self.tree.bind("<Double-1>", lambda _e: self._open_viewer())
        self.tree.pack(side="left", fill="both", expand=True)

        # ---- 右栏：内容一屏放不下，必须能滚 ----
        # 实测（1280x860，正是窗口默认尺寸）：右栏内容 reqheight=772、可用 607，
        # 差 165px —— 「开始导出 (zip)」被压到只剩 2px 高，后面的
        # 「导出轨迹数据」「清理…」「进度条」「状态」**根本没有 mapped**。
        # 最大化时看不出来，所以一直没人报（既有缺陷，2026-11-09 量出来的）。
        # 用 Canvas + 内嵌 Frame 做纵向滚动：按钮永远滚得到，而不是靠
        # "把窗口拉高一点"这种运气。
        right_canvas = tk.Canvas(right, highlightthickness=0, borderwidth=0,
                                 width=240, background=self.cget("background"))
        right_sb = ttk.Scrollbar(right, orient="vertical",
                                 command=right_canvas.yview)
        right_canvas.configure(yscrollcommand=right_sb.set)
        right_sb.pack(side="right", fill="y")
        right_canvas.pack(side="left", fill="both", expand=True)
        panel = ttk.Frame(right_canvas)
        panel_win = right_canvas.create_window((0, 0), window=panel, anchor="nw")

        def _sync_scrollregion(_event=None) -> None:
            right_canvas.configure(scrollregion=right_canvas.bbox("all"))

        def _sync_panel_width(event) -> None:
            right_canvas.itemconfigure(panel_win, width=event.width)

        panel.bind("<Configure>", _sync_scrollregion)
        right_canvas.bind("<Configure>", _sync_panel_width)

        def _wheel(event) -> str:
            # 只有**真的溢出**时才滚：否则滚轮会被吃掉却什么也不动，
            # 用户会以为界面卡了。
            if panel.winfo_reqheight() <= right_canvas.winfo_height():
                return "break"
            right_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
            return "break"

        right_canvas.bind("<MouseWheel>", _wheel)

        self.count_label = ttk.Label(panel, text="", font=("Microsoft YaHei UI", 10, "bold"))
        self.count_label.pack(anchor="w", pady=(0, 6))
        # 状态图例（2026-11-05 用户："数据管理的状态符号没有说明"）。
        # 原来挤成一行 `✓ 有3D  ◐ 已点选  ○ 已录制  ✗ 坏  — 空`：
        # ① 换行后第二行被面板高度裁掉，用户根本看不见完整的；
        # ② 光列符号不解释，`◐ 已点选` 和 `○ 已录制` 差在哪看不出来。
        # 现在把**推进顺序**写出来 —— 前三态是管线上的三个阶段，不是并列的
        # 三种类型：录完只有视频 → 点完球心 → 跑完三维。箭头比逗号有用。
        # `◑ 三维点过少` 不是独立阶段，是「跑完了但没跑出东西」的告警态：
        # 离线三维跑过、有产物，但匹配率低到画不出轨迹（阈值见
        # `_workbench.THIN_MATCH_RATIO`），介于 ◐ 与 ✓ 之间。
        self.status_legend = ttk.Label(
            panel,
            text="状态（→ 表示管线推进顺序）：\n"
                 "○ 已录制（未点球心） → ◐ 已点选（未跑三维） → ◑ 三维点过少 → ✓ 有三维\n"
                 "✗ 数据损坏（缺 session.json / 有产物却读不到帧数 / 左右差太多）\n"
                 "— 没录成（左右都是 0 帧且没有任何产物；不是数据坏了，重录即可）",
            foreground=MUTED, justify="left", wraplength=230)
        self.status_legend.pack(anchor="w", pady=(0, 6))
        # 换行宽度**跟着面板走**，不能写死：右侧面板宽 251px、这段文字需要
        # 311px，写死 wraplength 会被横向裁掉一截（用户看到的"说明不全"），
        # 窗口拉宽了又白白折行。ttk.Label 不会自己收缩到父容器宽度，
        # 只能由 `<Configure>` 告诉我们实际多宽。
        # `add="+"`：这个事件上已经挂了"重算滚动区"的回调，不能覆盖掉。
        panel.bind("<Configure>", self._on_right_resize, add="+")

        cats = ttk.LabelFrame(panel, text="导出类别", padding=8)
        cats.pack(fill="x", pady=(0, 8))
        for key, label in CATEGORIES.items():
            var = tk.BooleanVar(value=key in DEFAULT_CATEGORIES)
            self._cat_vars[key] = var
            ttk.Checkbutton(cats, text=label, variable=var,
                            command=self._refresh_count).pack(anchor="w", pady=1)
        # 折行宽度跟着面板走（见 `_on_right_resize`），这里只给初值：
        # 写死 300 会正好被右边裁掉最后一个字。
        self._cats_warn = ttk.Label(
            cats, text="默认不含视频；视频画面未打码，勾选前确认授权。",
            foreground="#b45309", wraplength=200)
        self._cats_warn.pack(anchor="w", pady=(4, 0))

        red = ttk.LabelFrame(panel, text="脱敏", padding=8)
        red.pack(fill="x", pady=(0, 8))
        for key, label in (("camera", "移除相机 identity/name/port"),
                           ("labels", "移除会话 label/notes"),
                           ("platform", "移除 OS platform")):
            var = tk.BooleanVar(value=True)
            self._redact_vars[key] = var
            ttk.Checkbutton(red, text=label, variable=var).pack(anchor="w", pady=1)

        out_box = ttk.LabelFrame(panel, text="输出", padding=8)
        out_box.pack(fill="x", pady=(0, 8))
        self.out_var = tk.StringVar(value=self._default_output())
        ttk.Entry(out_box, textvariable=self.out_var).pack(fill="x", pady=(0, 4))
        ttk.Button(out_box, text="选择 zip 路径…", command=self.browse_output).pack(anchor="w")

        self.export_btn = ttk.Button(panel, text="开始导出 (zip)", command=self.start_export)
        self.export_btn.pack(fill="x", pady=(2, 4))
        # 「处理前 + 处理后」一次导出（2026-11-03）：对每个**已跑过三维**的选中会话，
        # 在输出目录下成对写出 `<会话>_raw.csv`（原始轨迹，逐字节复制）与
        # `<会话>_motion.csv`（平滑 + 速度/加速度）及 `数据说明.md`。不依赖 zip 与类别勾选。
        self.export_pair_btn = ttk.Button(
            panel, text="导出轨迹数据 (csv)", command=self._export_motion_pair)
        self.export_pair_btn.pack(fill="x", pady=(0, 4))
        ttk.Button(panel, text="查看 3D + 视频", command=self._open_viewer).pack(
            fill="x", pady=(0, 4))
        ttk.Button(panel, text="查看原始视频",
                   command=lambda: self._open_viewer(mode="raw")).pack(
            fill="x", pady=(0, 4))
        ttk.Button(panel, text="在文件资源管理器中打开",
                   command=self._open_session_in_explorer).pack(
            fill="x", pady=(0, 4))
        clean_bar = ttk.Frame(panel)
        clean_bar.pack(fill="x", pady=(0, 4))
        ttk.Button(clean_bar, text="清理…", command=self._clean_selected).pack(
            side="left", fill="x", expand=True)
        ttk.Button(clean_bar, text="回收站…", command=self._open_trash).pack(
            side="left", fill="x", expand=True, padx=(6, 0))
        self.progress = ttk.Progressbar(panel, mode="determinate")
        self.progress.pack(fill="x", pady=(0, 4))
        self.status = ttk.Label(panel, text="等待选择", foreground=MUTED,
                                wraplength=200, justify="left")
        self.status.pack(anchor="w")
        # 滚轮绑到**面板的每一个子控件**上：鼠标停在复选框上滚，也得能滚。
        # （不用 bind_all：那会把滚轮从别的窗口抢走。）
        self._bind_wheel_tree(panel, _wheel)

        log_box = ttk.LabelFrame(outer, text="导出日志", padding=4)
        self.log = tk.Text(log_box, height=10, wrap="none")
        self.log.pack(side="left", fill="both", expand=True)
        lsb = ttk.Scrollbar(log_box, orient="vertical", command=self.log.yview)
        lsb.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=lsb.set)

        # 内容都建好之后再 add 到两个 PanedWindow（ttk 的 pane 只支持 weight，
        # 不支持 minsize；拖动由内置 sash 手柄完成）。
        left.add(tree_pane, weight=0)
        left.add(table_pane, weight=1)
        body.add(left, weight=4)
        body.add(right, weight=1)
        outer.add(body, weight=4)
        outer.add(log_box, weight=1)
        # 目录树只给一个 190px 的**初始**宽度；用户仍可拖，表格优先拿空间。
        self.after_idle(lambda: self._set_left_sash(190))

    def _set_left_sash(self, pos: int) -> None:
        """把「分组目录 | 会话表」的分隔线放到 pos 像素处（窗口映射后才行）。"""
        pane = getattr(self, "_left_pane", None)
        if pane is None:
            return
        try:
            if pane.winfo_width() > pos + 100:
                pane.sashpos(0, pos)
        except tk.TclError:
            pass

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
            "◑ 三维点过少": "traj_thin",
            "◐ 已点选": "seeded",
            "○ 已录制": "recorded",
            "✗ 坏": "bad",
            "— 空": "empty",
        }.get(self._status_filter)

    def _visible_items(self) -> list[SessionItem]:
        """当前筛选 + 搜索之后应该显示在列表里的项（顺序与 self.items 一致）。"""
        want_symbol = self._status_symbol(self._status_filter)
        q = self.search_var.get().strip().lower()
        group = self._filter_group
        out: list[SessionItem] = []
        for it in self.items:
            if want_symbol is not None and it.symbol != want_symbol:
                continue
            # 分组与状态/搜索是**与**关系：选中「实验1/梨子」时只看这一支
            if group is not None and group != "__all__":
                if group == "":
                    if it.group != "":
                        continue
                elif it.group != group and not it.group.startswith(group + "/"):
                    continue
            if q:
                # 搜索按**屏幕上那一格的字**：名称列带了「实时 ·」前缀，用户照着
                # 看见的字搜「实时」却搜不到就是自相矛盾。
                name = name_cell_text(it).lower()
                if q not in it.id.lower() and q not in name:
                    continue
            out.append(it)
        return out

    def _apply_filters(self) -> None:
        """按状态 + 搜索 + 分组重画列表。只改显示，勾选状态原样保留。"""
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
                                     it.id, name_cell_text(it),
                                     it.created,
                                     f"{it.frames_left}/{it.frames_right}",
                                     mode_cell_text(it),
                                     it.symbol, size))
        self._refresh_count()

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        # 每次刷新都重读集合配置：用户可能手改了 collections.json，或另开一个
        # 面板改过。读坏了 load_collections 会给警告并保留原文件，不覆盖。
        self.collections, warns = load_collections(self.collections_path)
        try:
            index, self.items = build_index(
                self.root_dir, recursive=True, collections=self.collections)
        except Exception as exc:                            # noqa: BLE001
            messagebox.showerror("扫描失败", str(exc), parent=self)
            return
        try:
            save_index(index, self.index_path)
        except (OSError, ValueError) as exc:
            self._log_line(f"[索引] 写不了 {self.index_path}：{exc}")
        for w in warns:
            self._log_line("[配置] " + w)
        for w in getattr(self.items, "warnings", []) or []:
            self._log_line("[扫描] " + w)
        self._refresh_group_tree()
        valid = {it.key for it in self.items}
        self.selected &= valid
        self._fill_date_options()
        self._apply_filters()
        msg = self._consistency_report()
        if msg:
            self._log_line("[一致性] " + msg)

    def _consistency_report(self) -> str:
        """所有会话的分组都必须在树上真的有节点；返回空串 = 没问题。

        为什么要有这条护栏：树和表是两份数据（一份来自 collections.json、
        一份来自扫描），任何一边少算一层，用户看到的就是"表里有、树里点不到"
        的幽灵会话。宁可往日志里写一行，也不要静默。
        """
        tree = getattr(self, "group_tree", None)
        missing: list[str] = []
        for it in self.items:
            g = it.group or ""
            if not g:
                continue
            if tree is None or not tree.exists("g:" + g):
                missing.append(g)
        if not missing:
            return ""
        uniq = sorted(set(missing))
        shown = "、".join(uniq[:6]) + ("…" if len(uniq) > 6 else "")
        return f"{len(missing)} 段会话在树上没有归属：{shown}"

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

    # ------------------------------------------------------------------
    # 分组目录树（虚拟目录）：建树、过滤、增删改、放入/移出
    # ------------------------------------------------------------------
    def _build_group_menu(self) -> None:
        m = tk.Menu(self, tearoff=0)
        m.add_command(label="新建目录…", command=self._group_new)
        m.add_command(label="改名…", command=self._group_rename)
        m.add_command(label="删除目录…", command=self._group_delete)
        m.add_separator()
        m.add_command(label="把勾选的会话放入此目录", command=self._group_assign)
        m.add_command(label="把勾选的会话移出目录", command=self._group_unassign)
        self._group_menu = m

    def _expanded_iids(self) -> set[str]:
        """递归收集当前展开的节点：重建树之后要按这些 iid 重新展开。"""
        out: set[str] = set()

        def walk(parent: str) -> None:
            for iid in self.group_tree.get_children(parent):
                if self.group_tree.item(iid, "open"):
                    out.add(iid)
                walk(iid)

        walk("")
        return out

    def _refresh_group_tree(self) -> None:
        """按「会话分组 ∪ 空目录」重建树，并保留展开状态与选中目录。

        空目录也要建：用户刚点完「新建」还没往里放东西，如果树只按会话建，
        这个目录会当场消失，用户会以为没建成。
        """
        tree = getattr(self, "group_tree", None)
        if tree is None:
            return
        expanded = self._expanded_iids()
        counts: dict[str, int] = {}
        for it in self.items:
            g = it.group or ""
            counts[g] = counts.get(g, 0) + 1
        paths = set(counts) | all_node_paths(self.collections)
        ordered = all_group_paths(paths)      # "" 在最前，按层级排序

        def count_under(path: str) -> int:
            if path == "":
                return counts.get("", 0)
            return sum(n for k, n in counts.items()
                       if k == path or k.startswith(path + "/"))

        # 树里的过滤目标失效了（目录被删/改名、换过数据根）→ 回到「全部会话」
        if self._filter_group is not None and self._filter_group not in set(ordered):
            self._filter_group = None

        self._syncing_group = True
        try:
            tree.delete(*tree.get_children())
            # 「全部会话」是**唯一**顶层节点，其余一律挂在它下面。
            # 为什么不能让一级分组和它并列：那样树上就是三个平级条目
            # （全部会话 / 散会话 / 平行双目），读起来像三个过滤器，而不是
            # "全部 → 分组 → 子目录"的层级 —— 用户要的是资源管理器那种逐层展开。
            tree.insert("", "end", iid="g:__all__",
                        text=f"全部会话 ({len(self.items)})", open=True)
            first_build = not self._group_tree_ready
            for path in ordered:
                iid = "g:" + path
                name = UNFILED_LABEL if path == "" else path.rsplit("/", 1)[-1]
                if "/" in path:
                    pid = "g:" + path.rsplit("/", 1)[0]
                else:
                    pid = "g:__all__"            # 一级分组：挂在「全部会话」下
                # 首次建树把一级目录展开（第一眼就看见两个分区）；之后按用户
                # 的展开状态恢复，用户收起的节点不会被 refresh 偷偷展开。
                is_open = iid in expanded or (first_build and path.count("/") < 1)
                tree.insert(pid, "end", iid=iid,
                            text=f"{name} ({count_under(path)})", open=is_open)
        finally:
            self._syncing_group = False
        self._group_tree_ready = True
        self._select_group_iid()

    def _select_group_iid(self) -> None:
        """让树上的选中项跟着 `_filter_group` 走（重建后 / 新建后调用）。"""
        if not hasattr(self, "group_tree"):
            return
        target = ("g:__all__" if self._filter_group is None
                  else "g:" + self._filter_group)
        if not self.group_tree.exists(target):
            return
        self._syncing_group = True
        try:
            self.group_tree.selection_set(target)
            self.group_tree.focus(target)
            self.group_tree.see(target)
        except tk.TclError:
            pass
        finally:
            self._syncing_group = False

    def _on_group_select(self, _event=None) -> None:
        if getattr(self, "_syncing_group", False):
            return
        sel = self.group_tree.selection()
        if not sel:
            return
        iid = sel[0]
        if iid == "g:__all__":
            group: str | None = None
        elif iid == "g:":
            group = ""
        elif iid.startswith("g:"):
            group = iid[2:]
        else:
            return
        if group == self._filter_group:
            return
        self._filter_group = group
        self._apply_filters()

    def _group_right_click(self, event) -> None:
        row = self.group_tree.identify_row(event.y)
        if row:
            self.group_tree.selection_set(row)
            self.group_tree.focus(row)
        try:
            self._group_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._group_menu.grab_release()

    def _group_target_node(self) -> dict | None:
        """当前选中目录对应的节点；「全部会话」与「散会话」都指向根节点。"""
        sel = self.group_tree.selection()
        iid = sel[0] if sel else "g:__all__"
        if iid in ("g:__all__", "g:"):
            nid: str | None = ROOT_NODE
        else:
            reverse = {p: n for n, p in
                       collection_paths_by_node(self.collections).items()}
            nid = reverse.get(iid[2:])
        for node in self.collections.get("nodes") or []:
            if node.get("id") == nid:
                return node
        return None

    def _selected_group_path(self) -> str | None:
        """选中节点对应的逻辑路径：根/散会话是 ``""``，没选中也是 ``""``。"""
        sel = self.group_tree.selection()
        if not sel or sel[0] in ("g:__all__", "g:"):
            return ""
        iid = sel[0]
        return iid[2:] if iid.startswith("g:") else None

    def _ensure_node_for_path(self, path: str) -> dict | None:
        """把一条逻辑路径落成真实节点（缺哪层补哪层）。

        为什么需要：``平行双目`` 这类一级分组是 ``group_of`` 从**磁盘层级**
        （``parallel/<时间戳>``）推出来的，collections.json 里并没有这个节点。
        用户选中它点「新建 / 放入」时不补节点，就会出现"树上明明有、点了却说
        找不到目录"。补出来的节点和磁盘分组同名同路径，会话归属不变。
        """
        if not path:
            return next((n for n in self.collections.get("nodes") or []
                         if n.get("id") == ROOT_NODE), None)
        for nid, p in collection_paths_by_node(self.collections).items():
            if p == path:
                return next((n for n in self.collections.get("nodes") or []
                             if n.get("id") == nid), None)
        kids = node_children(self.collections)
        taken = {str(n.get("id")) for n in self.collections.get("nodes") or []}
        parent_id = ROOT_NODE
        node: dict | None = None
        for part in [p for p in path.split("/") if p]:
            node = next((n for n in kids.get(parent_id, [])
                         if str(n.get("name")) == part), None)
            if node is None:
                nid = new_node_id(taken, part)
                taken.add(nid)
                node = {"id": nid, "name": part, "parent_id": parent_id,
                        "session_ids": []}
                self.collections.setdefault("nodes", []).append(node)
                kids.setdefault(parent_id, []).append(node)
            parent_id = str(node.get("id"))
        return node

    def _group_write_target(self) -> dict | None:
        """可写的目录节点：已存在就直接用，来自磁盘的分组先补成节点。"""
        node = self._group_target_node()
        if node is not None:
            return node
        path = self._selected_group_path()
        return self._ensure_node_for_path(path) if path is not None else None

    def _group_new(self) -> None:
        target = self._group_write_target()
        if target is None:
            messagebox.showinfo("新建目录", "先在树里选一个目录。", parent=self)
            return
        raw = simpledialog.askstring(
            "新建目录", "目录名（可用 / 一次建多级，例如 实验1/梨子）",
            parent=self)
        if raw is None:
            return
        raw = raw.strip()
        if not raw:
            messagebox.showwarning("新建目录", "目录名不能为空。", parent=self)
            return
        if raw.endswith("/"):
            messagebox.showwarning("新建目录", "目录名不能以 / 结尾。", parent=self)
            return
        parts = [p.strip() for p in raw.split("/")]
        if any(not p for p in parts) or any(p in (".", "..") for p in parts):
            messagebox.showwarning(
                "新建目录", "目录名不合法：不能有空的层级、. 或 ..", parent=self)
            return

        parent_id = str(target.get("id") or ROOT_NODE)
        taken = {str(n.get("id")) for n in self.collections.get("nodes") or []}
        kids = node_children(self.collections)
        final_id = parent_id
        for i, part in enumerate(parts):
            existing = next((n for n in kids.get(parent_id, [])
                             if str(n.get("name")) == part), None)
            if existing is not None:
                if i == len(parts) - 1:
                    messagebox.showwarning(
                        "新建目录", f"这一层已经有「{part}」了，换个名字。",
                        parent=self)
                    return
                parent_id = str(existing.get("id"))
                final_id = parent_id
                continue
            nid = new_node_id(taken, part)
            taken.add(nid)
            node = {"id": nid, "name": part, "parent_id": parent_id,
                    "session_ids": []}
            self.collections.setdefault("nodes", []).append(node)
            kids.setdefault(parent_id, []).append(node)
            parent_id = final_id = nid

        new_path = collection_paths_by_node(self.collections).get(final_id, "")
        self._after_collections_change(f"新建目录：{raw}", select_path=new_path)

    def _group_rename(self) -> None:
        sel = self.group_tree.selection()
        if not sel or sel[0] in ("g:__all__", "g:"):
            messagebox.showinfo("改名", "选一个自己建的目录再改名。", parent=self)
            return
        node = self._group_target_node()
        if node is None:
            messagebox.showinfo(
                "改名",
                "这个目录来自磁盘布局（例如 stereo/parallel），没有可改的虚拟"
                "节点；请新建一个目录把会话放进去，或直接改磁盘上的目录名。",
                parent=self)
            return
        cur = str(node.get("name") or "")
        raw = simpledialog.askstring("改名", "新目录名", initialvalue=cur,
                                     parent=self)
        if raw is None:
            return
        new = raw.strip()
        if not new:
            messagebox.showwarning("改名", "目录名不能为空。", parent=self)
            return
        if "/" in new:
            messagebox.showwarning(
                "改名", "目录名不能包含 /（要分层请新建子目录）。", parent=self)
            return
        for sib in node_children(self.collections).get(node.get("parent_id"), []):
            if sib.get("id") != node.get("id") and str(sib.get("name")) == new:
                messagebox.showwarning(
                    "改名", f"同一层已经有「{new}」了：两条路径相同会让会话归属"
                            "出现二义。换个名字。", parent=self)
                return
        if new == cur:
            return
        node["name"] = new
        new_path = collection_paths_by_node(self.collections).get(node["id"], "")
        self._after_collections_change(f"改名：{cur} → {new}", select_path=new_path)

    def _group_delete(self) -> None:
        sel = self.group_tree.selection()
        if not sel or sel[0] in ("g:__all__", "g:"):
            messagebox.showinfo("删除目录", "「全部会话」「散会话」不是可删的目录。",
                                parent=self)
            return
        node = self._group_target_node()
        if node is None:
            messagebox.showinfo(
                "删除目录",
                "这个目录来自磁盘布局（例如 stereo/parallel），树上没有可删的"
                "虚拟节点；要删请直接整理磁盘上的目录。\n会话数据不受影响。",
                parent=self)
            return
        nodes = {str(n.get("id")): n for n in self.collections.get("nodes") or []}
        doomed = {str(node.get("id"))}
        grew = True
        while grew:                     # 先收集全部后代，父子一起删
            grew = False
            for nid, n in nodes.items():
                if nid not in doomed and str(n.get("parent_id")) in doomed:
                    doomed.add(nid)
                    grew = True
        sids = [sid for nid in doomed for sid in (nodes[nid].get("session_ids") or [])]
        n_sub = len(doomed) - 1
        msg = (f"删除目录「{node.get('name')}」"
               + (f"及其 {n_sub} 个子目录" if n_sub else "")
               + f"？\n\n目录下的 {len(sids)} 段会话会回到「散会话」"
                 "（磁盘上已有分区目录的会话仍按磁盘位置显示），**不删任何数据**。")
        if not messagebox.askyesno("删除目录", msg, parent=self,
                                   default=messagebox.NO):
            return
        name = str(node.get("name") or "")
        self.collections["nodes"] = [
            n for n in self.collections.get("nodes") or []
            if str(n.get("id")) not in doomed]
        self._after_collections_change(
            f"删除目录：{name}（{len(sids)} 段会话未删）")

    def _checked_sids(self) -> set[str]:
        return {it.sid for it in self.items if it.key in self.selected}

    def _group_assign(self) -> None:
        node = self._group_write_target()
        if node is None:
            messagebox.showinfo("放入目录", "先在树里选一个目录。", parent=self)
            return
        sids = self._checked_sids()
        if not sids:
            messagebox.showinfo("放入目录", "先在右边勾选会话。", parent=self)
            return
        target_id = str(node.get("id") or ROOT_NODE)
        for n in self.collections.get("nodes") or []:
            if str(n.get("id")) == target_id:
                continue                    # 先把它从原来所在的目录移除
            n["session_ids"] = [s for s in (n.get("session_ids") or [])
                                if s not in sids]
        tgt = next(n for n in self.collections["nodes"]
                   if str(n.get("id")) == target_id)
        have = set(tgt.get("session_ids") or [])
        tgt["session_ids"] = list(tgt.get("session_ids") or []) + sorted(
            sids - have)
        self._after_collections_change(
            f"放入 {len(sids)} 段会话 → {node.get('name')}")

    def _group_unassign(self) -> None:
        sids = self._checked_sids()
        if not sids:
            messagebox.showinfo("移出目录", "先在右边勾选会话。", parent=self)
            return
        removed = 0
        for n in self.collections.get("nodes") or []:
            ids = list(n.get("session_ids") or [])
            keep = [s for s in ids if s not in sids]
            removed += len(ids) - len(keep)
            n["session_ids"] = keep
        if not removed:
            messagebox.showinfo("移出目录", "勾选的会话本来就不在任何目录里。",
                                parent=self)
            return
        self._after_collections_change(f"移出目录：{len(sids)} 段会话")

    # ---- 落盘 / 局部重算 --------------------------------------------
    def _reload_groups(self) -> None:
        """只重算每项的分组，不重新扫盘（会话的 sid / rel_path 没变）。"""
        mapping = collection_map(self.collections)
        self.items = [replace(it, group=group_of(it.sid, mapping, it.rel_path))
                      for it in self.items]

    def _persist_collections(self, what: str) -> bool:
        try:
            save_collections(self.collections, self.collections_path)
        except (OSError, ValueError) as exc:
            messagebox.showerror(
                "保存分组配置",
                f"写不了 {self.collections_path}：\n{exc}\n\n"
                "界面已从磁盘重新读取，不会显示一份没存下的状态。",
                parent=self)
            self.collections, warns = load_collections(self.collections_path)
            for w in warns:
                self._log_line("[配置] " + w)
            self._reload_groups()
            self._refresh_group_tree()
            self._apply_filters()
            return False
        self._log_line(f"[分组] {what}")
        return True

    def _after_collections_change(self, what: str,
                                  select_path: str | None = None) -> None:
        if not self._persist_collections(what):
            return
        self._reload_groups()
        if select_path is not None:
            self._filter_group = select_path
        self._refresh_group_tree()
        self._apply_filters()

    @staticmethod
    def _bind_wheel_tree(widget, handler) -> None:
        """把滚轮处理绑到 widget 与它的**全部子控件**上（递归）。

        为什么不用 `bind_all`：那会把这个窗口的滚轮语义挂到全局，切到别的
        窗口（3D 查看器、相机管理）时也跟着滚 —— Tk 里这种"图省事"的全局绑定
        是典型的互相踩。递归绑只影响这一棵子树的控件。
        """
        widget.bind("<MouseWheel>", handler, add="+")
        for child in widget.winfo_children():
            DataManager._bind_wheel_tree(child, handler)

    def _on_right_resize(self, event) -> None:
        """右侧面板宽度变了 -> 重设**所有**折行标签的 wraplength。

        `ttk.Label` 不会自己收缩到父容器宽度：写死 `wraplength` 时，
        面板 251px 而文字要 311px，右边那截直接被裁掉（用户看到的
        "状态说明不全"）；窗口拉宽又白白折行。只能由 `<Configure>` 告诉
        我们实际多宽，再回写换行宽度。

        2026-11-09：右栏改成可滚动的 Canvas 之后，面板宽度会随窗口变（实测
        217–311 之间），所以「状态 / 红色提示 / 进度状态」这三处折行宽度**一起**
        跟着走 —— 只改图例那一处，另外两处照样被裁掉最后几个字。
        """
        wrap = max(120, int(event.width) - 16)
        for attr in ("status_legend", "status", "_cats_warn"):
            lab = getattr(self, attr, None)
            if lab is None:
                continue
            try:
                if int(lab.cget("wraplength")) != wrap:
                    lab.configure(wraplength=wrap)
            except Exception:                        # noqa: BLE001
                pass

    def _open_help(self) -> None:
        """打开操作说明并定位到「数据管理」分区。"""
        from apps.help_window import open_help

        open_help(self, section="data")

    def _refresh_count(self) -> None:
        chosen = [it for it in self.items if it.key in self.selected]
        try:
            opts = self._export_options()
            size = estimate_bytes(chosen, opts)
        except Exception:                                   # noqa: BLE001
            size = sum(it.size_bytes for it in chosen)
        self.count_label.configure(
            text=f"已选 {len(chosen)} / {len(self.items)} 个会话\n"
                 f"预计原始数据 {_fmt_bytes(size)}")
        self._set_row_checks()
        self._sync_highlight()
        # 勾上视频后，橙色提示要说清「点导出会弹一次确认、需选『是』」——
        # 否则用户按了回车（默认落在「否」）会以为按钮没反应。
        # getattr 保护：_cats_warn 在 __init__ 的 473 行才建，首次 _refresh_count
        # 虽在其后，早调用也不能 AttributeError。
        warn = getattr(self, "_cats_warn", None)
        if warn is not None:
            video_on = any(
                self._cat_vars[k].get() for k in ("annotated", "raw")
                if k in self._cat_vars)
            if video_on:
                warn.configure(
                    text="已勾选视频：点「开始导出」会弹一次确认，"
                         "需选『是』才会导出。画面未打码，先确认授权。")
            else:
                warn.configure(
                    text="默认不含视频；视频画面未打码，勾选前确认授权。")

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

    def _export_options(self) -> ExportOptions:
        """导出类别 + 脱敏开关。

        ⚠️ 这个方法**不能叫 `_options`**：`tkinter.commondialog.Dialog.show()`
        会执行 `master._options(self.options)`，而 `master` 就是我们传进
        `messagebox.*(parent=self)` / `filedialog.*(parent=self)` 的这个窗口。
        Tk 想调的是 `tk.Misc._options(self, cnf, kw=None)`（把配置字典转成 Tcl
        参数），一旦被本类同名方法遮蔽，所有模态对话框都会抛
        `TypeError: DataManager._options() takes 1 positional argument but 2
        were given`，弹窗根本出不来 —— 2026-11-11 真机实测踩到。
        """
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
        # 集合配置 / 索引跟着数据根走：否则换一块盘以后界面显示的是**另一棵树**
        # 的分组，点开就是一堆找不到的会话。
        self.collections_path, self.index_path = paths_for_root(self.root_dir)
        self.collections, warns = load_collections(self.collections_path)
        for w in warns:
            self._log_line("[配置] " + w)
        self.selected.clear()
        self._filter_group = None
        self._group_tree_ready = False      # 换树后重新把一级目录默认展开
        self.refresh()

    def _focused_item(self) -> SessionItem | None:
        row = self.tree.focus()
        if row:
            return next((it for it in self.items if it.key == row), None)
        if len(self.selected) == 1:
            key = next(iter(self.selected))
            return next((it for it in self.items if it.key == key), None)
        return None

    def _open_viewer(self, mode: str = "annotated") -> None:
        it = self._focused_item()
        if it is None:
            messagebox.showinfo("3D + 视频", "先在列表里选中一条会话。", parent=self)
            return
        # 2026-11-11：**没有三维结果也允许打开** —— 查看器会降级成"空场景 +
        # 会话信息 + 为什么没有结果"，不再在这里一律拦掉。
        viz = it.path / "derived" / "visualization.json"
        traj = it.path / "derived" / "traj3d.json"
        if not viz.exists() and not traj.exists():
            self._log_line(f"打开 3D 查看器（无三维结果，降级视图）：{it.id}")
        annotated = it.path / "derived" / "annotated.mp4"
        args = [sys.executable, str(ROOT / "apps" / "view_3d.py"), str(it.path)]
        if mode == "raw":
            if not has_any_raw_video(it.path):
                messagebox.showinfo(
                    "没有原始视频",
                    "这条会话没有左右原始视频（left/right.mkv 或 .mp4），"
                    "请用「查看 3D + 视频」查看标注视频。",
                    parent=self)
                return
            args.extend(["--video-source", "raw"])
        elif not annotated.exists():
            if not messagebox.askyesno(
                    "没有标注视频", "这条会话没有 annotated.mp4。\n只打开 3D 查看器？",
                    parent=self):
                return
            args.append("--no-video")
        try:
            viewer = subprocess.Popen(args)
        except Exception as exc:                            # noqa: BLE001
            messagebox.showerror("3D + 视频", f"打不开查看器：{exc}", parent=self)
            return
        # 留句柄：关数据管理时 terminate()，不再留真孤儿进程。
        self._viewer_procs.append(viewer)
        # 只写 start、不写 end：查看器什么时候关是它自己的事。账本靠 PID
        # 探活自动把它从"正在运行"里摘掉（崩了也一样），不需要跨进程回调。
        task = self._registry_task(
            self._next_ext_tid(), "3D 查看器（数据管理打开）")
        register_start(task, source="data_manager")
        self._log_line(f"打开 3D + 视频：{it.id}")

    def _open_session_in_explorer(self) -> None:
        it = self._focused_item()
        if it is None:
            messagebox.showinfo("打开会话文件夹", "先只选一条会话。", parent=self)
            return
        try:
            open_in_explorer(it.path, allowed_root=self.root_dir)
        except OpenInExplorerError as exc:
            messagebox.showerror("打开会话文件夹", f"打不开会话文件夹：{exc}",
                                 parent=self)
            return
        self._log_line(f"已在文件资源管理器中打开：{it.id}")

    def _open_trash(self) -> None:
        from apps.cleanup_dialog import TrashDialog

        _top = self.winfo_toplevel()
        _old = getattr(_top, "_trash_window", None)
        if _old is not None and _old.winfo_exists():
            _old.deiconify()
            _old.lift()
            return
        _top._trash_window = TrashDialog(
            self, self.root_dir, on_done=self.refresh)

    def _clean_selected(self) -> None:
        from apps.cleanup_dialog import CleanupDialog
        chosen = [it for it in self.items if it.key in self.selected]
        if not chosen:
            it = self._focused_item()
            chosen = [it] if it else []
        if not chosen:
            messagebox.showinfo("清理", "先勾选或选中至少一个会话。", parent=self)
            return
        _top = self.winfo_toplevel()
        _old = getattr(_top, "_cleanup_window", None)
        if _old is not None and _old.winfo_exists():
            messagebox.showinfo(
                "清理", "上一个清理窗口还开着，先完成或关掉它。", parent=self)
            return
        _top._cleanup_window = CleanupDialog(
            self, self.root_dir, [it.path for it in chosen],
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
        # 标准窗口：最小化 / 最大化 / 关闭（AGENTS.md §7.1）。
        # 不用 transient()（Windows 上变工具窗口、只剩 ×）、不锁尺寸、不全局 grab。
        dlg.resizable(True, True)
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
        # 同步跑在 UI 线程，父进程的 TaskRunner 完全看不见 —— 额外登记进账本，
        # 控制台的"正在运行"就有这一行了。try/finally 保证异常也收尾。
        reg_task = self._registry_task(
            self._next_ext_tid(), f"导出轨迹：{out_dir.name}")
        register_start(reg_task, source="data_manager")
        done, skipped = [], []
        try:
            for it in chosen:
                try:
                    pair = export_motion_pair(it.path, out_dir)
                except Exception as exc:                 # noqa: BLE001
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
        finally:
            register_end(reg_task, source="data_manager")

    def start_export(self) -> None:
        if self._running:
            return
        chosen = [it for it in self.items if it.key in self.selected]
        if not chosen:
            messagebox.showinfo("导出", "先勾选至少一个会话。", parent=self)
            return
        try:
            opts = self._export_options()
        except ValueError as exc:
            messagebox.showerror("导出", str(exc), parent=self)
            return
        if not opts.categories:
            messagebox.showinfo("导出", "至少选一个导出类别。", parent=self)
            return
        video_cats = [k for k in ("annotated", "raw") if k in opts.categories]
        if video_cats:
            cat_short = "、".join(CATEGORIES[k].split("（")[0] for k in video_cats)
            notice = [
                f"本次导出包含视频：{cat_short}。",
                f"已选 {len(chosen)} 段会话。",
            ]
            try:
                notice.append(f"预计体积 {_fmt_bytes(estimate_bytes(chosen, opts))}。")
            except Exception:                                   # noqa: BLE001
                pass    # 体积估不出来就省掉这行，别让确认框崩掉
            notice += [
                "视频几乎压不动，导出可能要几十秒到几分钟。",
                "视频画面没有自动打码，房间/人物会被一起交付，请先确认授权。",
                "选『是』立刻开始导出；选『否』则什么都不做。",
            ]
            if not messagebox.askyesno(
                    "包含视频，请确认", "\n".join(notice),
                    parent=self, default=messagebox.NO):
                # 隐私安全默认仍是「否」：按回车/点默认 = 取消。原来这里直接
                # return，按钮不变、status 停在「等待选择」、日志一行不写，
                # 用户完全看不出发生过什么，还以为「开始导出」按钮坏了。
                self.status.configure(
                    text="已取消：视频隐私确认未通过，没有生成 zip")
                self._log_line("[取消] 视频隐私确认未通过，没有生成 zip")
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
        if video_cats:
            self.status.configure(
                text="导出中…（含视频，进度条可能几十秒不动，属正常）")
            self._log_line("含视频：视频改用纯存储不压缩，进度条可能几十秒不动，属正常")
        else:
            self.status.configure(text="导出中…")
        self._log_line(f"开始导出 {len(chosen)} 个会话 → {out}")

        # 本进程是独立进程，父进程的 TaskRunner 看不到这次导出 —— 额外登记
        # 进跨进程账本，控制台"正在运行"就有这一行。既有 _running 布尔与
        # 进度 UI 一行不动，账本只是多记一笔。
        reg_task = self._registry_task(
            self._next_ext_tid(), f"导出 zip：{out.name}")
        register_start(reg_task, source="data_manager")

        def work() -> None:
            def report(done: int, total: int, name: str) -> None:
                if self._cancel.is_set():
                    raise RuntimeError("用户取消导出")
                self._ui_queue.put(("progress", done, total, name))

            try:
                try:
                    result = export_zip(chosen, opts, out, progress=report)
                except Exception as exc:                    # noqa: BLE001
                    if self._cancel.is_set():
                        return
                    self._ui_queue.put(("error", exc))
                else:
                    if not self._cancel.is_set():
                        self._ui_queue.put(("done", result))
            finally:
                # 取消 / 异常 / 正常完成都要收尾，否则会一直挂在"正在运行"。
                register_end(reg_task, source="data_manager")

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
        self._terminate_viewers()
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
        # 导出产物落在 outputs/ 下，不在会话根里 —— allowed_root 必须留 None，
        # 传 self.root_dir 会被判越界。这里是"顺手一按"，失败静默即可。
        try:
            open_in_explorer(folder, allowed_root=None)
        except OpenInExplorerError:
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
