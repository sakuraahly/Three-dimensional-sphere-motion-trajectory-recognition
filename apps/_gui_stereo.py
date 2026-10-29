"""双目模式窗（``parallel`` / ``angled`` 共用一份代码，由 ``mode`` 参数化）。

为什么"两个窗口"不是"两套皮肤"（2026-10-27 计划书 §2.1）
--------------------------------------------------------
平行与大角度的区别不在外观，而在**几何前提与标定方法**：

======================  ====================  ==========================
                        parallel               angled
======================  ====================  ==========================
深度来源                 校正后视差            原始像素三角测量
标定方法                 joint（必须共视）      joint+跳过 FOV 或 chain
能否两路看不到同一区域    不能                   能（chain）
``--stereo-mode``        parallel               angled
======================  ====================  ==========================

所以本模块**只有一份实现**：``mode`` 只影响三处 —— 第 2 步能选的方法、
第 2 步的前提文案、第 4 步命令行里的 ``--stereo-mode``。
**窗口里没有模式下拉框**：模式 = 你进的哪个窗口（旧界面那个顶栏下拉框是
``stereo_mode`` 的第三份真相，已删除）。

用户问过的那件事（2026-10-28）：单板链式标定与双目工作台是什么关系？
答案写在 ``angled`` 窗的第 2 步：chain 的产物 meta 记 ``stereo_mode=angled``，
第 4 步直接用 —— 它们本来就是一条链路，只是旧界面把两者摆成了两张互不相干的卡。
"""

from __future__ import annotations

import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from apps._gui_widgets import BG, INK, MUTED, WARN, ScrollPage, StatusBar, Surface
from apps._collapsible import CollapsibleSection
from apps._task_runner import ROOT

__all__ = ["StereoWindow", "SessionPicker", "scan_sessions", "pick_seeds",
           "board_args", "chain_dirs", "calibrate_args", "record_args",
           "record_out_dir"]

CHAIN_ROOT = ROOT / "outputs" / "chain_frames"

#: 两条方法在界面上的说法。键与 `calibrate_stereo.py --method` 一致。
METHODS = {
    "joint": ("两路共视（joint）",
              "两路在**每个姿态**都要同时看到同一块板。适合平行或充分共视的装置。"),
    "joint_skip_fov": ("两路共视，跳过视场裁剪（joint + no FOV）",
                       "仍是共同求解，但不按重叠视场裁剪 —— 大角度装置裁完会剩太窄。"),
    "chain": ("单板链式（chain）",
              "板不必被两路同时看到：各拍一组内参图，再用一块固定参考板把两路串起来。"
              "90 度这类几乎看不到同一区域的装置用它。产物就是本窗口要的 angled。"),
}


# --------------------------------------------------------------------------- #
# 命令行拼装：界面取值 → argv，**不依赖 Tk** 的纯函数
# --------------------------------------------------------------------------- #
# 为什么抽成模块级函数（2026-10-29）
# ----------------------------------
# 这两段拼装原来内联在 `_start_calibrate` / `_start_record` 里，而
# `tests/selftest_gui_budget.py` 只守控件数量与文案 —— 也就是说**新窗口真正
# 干活的那几行没有任何判据看着**："界面上选的板参数有没有正确进到
# calibrate_stereo 的命令行"，只能靠人眼看一遍。
# 老窗口（`启动程序.py` 的 `_calibrate_args` / `_record_args`）里同一件事有
# 7 节判据守着，新窗口却是空的 —— 这才是必须补的缺口。
# 抽出来的函数不碰 Tk、不读控件，判据才能直接调它；**行为与抽之前逐字等价**，
# 抽的时候没有顺手"修正"任何默认值。
BOARD_DEFAULTS = {"cols": "8", "rows": "5", "square": "25", "target": "40"}
_BOARD_FLAGS = (("cols", "--cols"), ("rows", "--rows"),
                ("square", "--square"), ("target", "--target"))


def board_args(board: dict | None) -> list[str]:
    """棋盘参数四项（内角点列/行、方格边长、目标对数）。

    留空 = 用脚本自己的默认档（8×5 / 25mm / 40 对），**不是**非法输入：
    这几个框在界面上本来就带着默认值，用户清空只是"这一次用默认"。
    真正的坑在另一头 —— "留空也传一个空串"：`--cols ""` 会在标定程序里
    被 `int()` 直接咬死，而报错的地方离用户的操作十万八千里。
    """
    out: list[str] = []
    for key, flag in _BOARD_FLAGS:
        raw = str((board or {}).get(key) or "").strip()
        out += [flag, raw or BOARD_DEFAULTS[key]]
    return out


def chain_dirs(root: Path | None = None) -> dict[str, Path]:
    """链式标定的四个目录（左右各一组 + 参考段的双路同拍图）。"""
    base = root if root is not None else CHAIN_ROOT
    return {"left": base / "left", "right": base / "right",
            "ref_left": base / "ref_left", "ref_right": base / "ref_right"}


def calibrate_args(method: str, board: dict | None = None,
                   left: str = "left", right: str = "right",
                   mode: str = "parallel", *,
                   stationary: str = "", sharpness: str = "",
                   relaxed: bool = False, backend: str = "",
                   dirs: dict | None = None) -> list[str]:
    """双目标定的完整 argv。

    三条分支的差别**只在参数**，不在文案：

    - ``chain``：``--method chain --same-board --left-dir/--right-dir``；
      参考段真有图时才加 ``--pair-frame``（指向不存在的目录等于给后端一个
      必然失败的证据，不如不提）。
    - ``joint_skip_fov``：仍是共同求解，但**模式写死 angled** —— 按重叠视场
      裁剪正是大角度装置要跳过的那一步。
    - ``joint``：带**本窗口**的模式（平行窗 parallel / 大角度窗 angled）。

    ``--skip-fov`` 不属于这里任何一条：它只属于旧界面那个"90° 装置"专用入口。
    新窗口的等效做法是选 `joint_skip_fov`（它走 ``--stereo-mode angled``，
    后端据此不再做 FOV 对齐），把这个 flag 混进普通 joint 会同时改变裁剪与
    内参口径，属于静默改行为。

    ``backend`` 只在调用方给出**具体**后端（dshow/msmf）时才下传：
    ``auto`` 是脚本自己的默认值，替用户写死一遍等于把"没选"伪装成"选了"。
    **两台相机共用这一个参数** —— 出现两次意味着有人给左右各拼了一份，
    那会拼出"左 dshow / 右 msmf"这种直接把一路弄挂的组合。
    """
    cmd = [sys.executable, str(ROOT / "apps" / "calibrate_stereo.py")]
    if method == "chain":
        d = dirs if dirs is not None else chain_dirs()
        cmd += ["--method", "chain", "--same-board",
                "--left-dir", str(d["left"]),
                "--right-dir", str(d["right"])] + board_args(board)
        # 参考段的双路同拍图作为"同一块板"的证据
        if Path(d["ref_left"]).exists():
            cmd += ["--pair-frame", str(d["ref_left"]), "--pair-camera", "left"]
    else:
        cmd += ["--method", "joint"] + board_args(board)
        if method == "joint_skip_fov":
            cmd += ["--stereo-mode", "angled"]
        else:
            cmd += ["--stereo-mode", mode]
        cmd += ["--left", str(left or "").strip() or "left",
                "--right", str(right or "").strip() or "right"]
    # 高级项：只有用户真的填了才下传（空 = 让后端用自己的默认值）
    if str(stationary or "").strip():
        cmd += ["--stationary-px", str(stationary).strip()]
    if str(sharpness or "").strip():
        cmd += ["--min-sharpness", str(sharpness).strip()]
    if relaxed:
        cmd += ["--relaxed"]
    if str(backend or "").strip() and str(backend).strip() != "auto":
        cmd += ["--backend", str(backend).strip()]

    # `--min-pairs` **必须传**，否则"试标定档"必然失败（2026-10-29 查出的功能缺口）。
    #
    # 链条：`calibrate_stereo` 的 `--min-pairs` 默认 20；而收图一旦达到
    # 「目标对数」就停（`calib/capture.py:274` `full = count >= cfg.target_pairs`）。
    # 于是把目标改成 8 的人会看到：收 8 对 → 停 → 8 < 20 → `:278 enough` 判假
    # → 标定失败。他只想快速验证通路，却被判"对数不足"。
    #
    # 旧界面把这个值放在「高级」里可填；新窗口没有这个框，所以在这里兜住：
    # 取 `min(20, target)` —— target ≥ 20 时与脚本默认**完全一致**（不改变语义），
    # 只有试标定档才下调到 target 本身（够 target 就能过）。
    _target = str((board or {}).get("target") or "").strip() or BOARD_DEFAULTS["target"]
    try:
        _pairs = int(_target)
    except ValueError:
        _pairs = 0
    if _pairs > 0:
        cmd += ["--min-pairs", str(min(20, _pairs))]
    return cmd


def record_args(*, left: str = "left", right: str = "right", duration: str = "10",
                out: str, mode: str | None = None) -> list[str]:
    """双目录制的 argv。

    **故意不传 ``--stereo-mode``**（B-107 的教训）：录制要按标定走 ``auto``，
    模式由 `stereo.record_plan.plan_record` 从标定里读。传平行/大角度都可能
    与当前标定不符 —— 后果是相机先白开、会话目录先建好，然后在检查里才报
    "模式冲突"，白等一轮；传 ``auto`` 又只是把脚本默认值写死一遍。
    所以 ``mode`` 只在调用方给出**非 auto** 的显式值时才下传，其余一律不提。

    分辨率/帧率/后端也**不传**：录制程序会按标定校验尺寸（`validate_capture_size`），
    界面再写一份数字只会多一处漂移。要改这些，先给它们加控件 —— 那时再把
    对应的关键字参数接上。
    """
    cmd = [sys.executable, str(ROOT / "apps" / "record_stereo.py"),
           "--left", str(left or "").strip() or "left",
           "--right", str(right or "").strip() or "right",
           "--duration", str(duration or "").strip() or "10",
           "--out", str(out)]
    if mode and str(mode).strip() and str(mode).strip() != "auto":
        cmd += ["--stereo-mode", str(mode).strip()]
    return cmd


def record_out_dir(label: str, root: Path | None = None) -> Path:
    """录制输出目录：``data/<标签>``，标签留空用 ``stereo_session``。

    为什么不用 `record_stereo.py --out` 的默认值（``data/sessions``）：那样
    两段录制会落进同一个目录、互相覆盖。标签即目录名，是这里早就定下的约定。
    """
    base = root if root is not None else ROOT
    return base / "data" / (str(label or "").strip() or "stereo_session")


# --------------------------------------------------------------------------- #
# 会话选择：三态过滤（匹配 / 不匹配 / 未记录）
# --------------------------------------------------------------------------- #
def scan_sessions(base: Path | None = None) -> list[dict]:
    """读 ``data/sessions/*/session.json`` 的摘要。

    只读 JSON、**不构造 ``Session`` 对象**：几十个会话也能瞬间列完，
    而且不会因为某一份坏文件把整个列表打断。

    ``mode`` 字段可能是 ``None`` —— 那表示**这份会话没有记录装置模式**。
    老会话都缺这个字段（``record_stereo`` 写入 ``stereo_mode`` 是
    ``a137766`` 才加的），所以**不能默认成 parallel**。
    """
    root = base if base is not None else ROOT / "data" / "sessions"
    rows: list[dict] = []
    try:
        dirs = sorted((d for d in root.iterdir() if d.is_dir()),
                      key=lambda d: d.name, reverse=True)
    except OSError:
        return rows
    for d in dirs:
        sj = d / "session.json"
        if not sj.is_file():
            continue
        try:
            raw = json.loads(sj.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = {}
        if not isinstance(raw, dict):
            raw = {}
        left = raw.get("left") if isinstance(raw.get("left"), dict) else {}
        rows.append({
            "path": d,
            "id": d.name,
            "label": str(raw.get("label") or ""),
            "created": str(raw.get("created") or ""),
            "mode": raw.get("stereo_mode"),
            # 会话里帧数叫 frame_count（实测 20261024-173619: 301 帧），不是 frames
            "frames": left.get("frame_count") or "",
        })
    return rows


def pick_seeds(parent, session_dir: str | Path) -> Path | None:
    """打开点选球心窗口（复用 ``apps.seed_picker.SeedPicker``），返回保存的路径或 None。

    为什么是进程内而不是 ``tasks.start`` 起子进程：``apps/seed_picker.py`` 只有类、
    **没有 ``__main__``**，`python apps/seed_picker.py <会话>` 会立刻以 0 退出、
    什么窗口都不开 —— 这正是"点了没反应"的形状。经典界面也是
    ``SeedPicker(self, ...) + wait_window``，这里沿用同一做法，不重画第二个点选界面。

    延迟 import：它依赖 numpy/cv2，而第 ①②③④ 步的其它功能不需要。
    """
    from apps.seed_picker import SeedPicker

    picker = SeedPicker(parent, str(session_dir))
    parent.wait_window(picker)
    return picker.result


class SessionPicker(tk.Toplevel):
    """选一段会话：列出磁盘上的会话，按装置模式**三态**标注。

    为什么要有它：旧界面把"会话目录"做成输入框，用户得自己记住并粘贴路径 ——
    那是"把该由系统做的事转嫁给用户"。

    为什么是三态而不是两态：老会话没有 ``stereo_mode`` 字段。两态（匹配/不匹配）
    会把它们当成 parallel，于是 90° 装置录的会话在平行窗里看起来"可用" ——
    **比不过滤更糟**。所以第三态叫"未记录"，如实标注、让人自己确认。

    ``mode=None`` 表示**跨装置**（会话工作台用）：不按几何过滤、也不标不匹配 ——
    工作台要处理磁盘上任意一段会话，"只看本窗模式"在那里是错的过滤。
    """

    COLS = (("id", "目录", 130), ("label", "标签", 150),
            ("created", "创建时间", 150), ("mode", "模式", 100),
            ("frames", "帧数", 70))

    def __init__(self, parent, *, mode: str | None, on_pick,
                 current: str = "") -> None:
        super().__init__(parent)
        self.mode = mode
        self.on_pick = on_pick
        self.title("选择会话 · " + {
            "parallel": "平行双目", "angled": "大角度双目",
        }.get(str(mode), "任意装置"))
        self.geometry("820x560")
        self.minsize(680, 440)
        self.configure(bg=BG)
        self.surface = Surface(self)
        # 跨装置时"显示全部"没有意义：直接置真，也不放那个复选框
        self.var_all = tk.BooleanVar(value=mode is None)
        self._rows = scan_sessions()

        box = self.surface.card(
            self, "会话",
            "这里列出 data/sessions 下的会话。默认只显示与当前窗口同一种装置几何的；"
            "老会话没有记录装置模式，标为「未记录」—— 用之前请自己确认它是哪种装置录的。")
        if mode is not None:
            ttk.Checkbutton(
                box, text="显示不属于本窗口模式的会话（标灰，选中时会说明风险）",
                variable=self.var_all,
                command=self._refresh).grid(
                row=self.surface._next_row(box), column=0, columnspan=9,
                sticky="w", pady=(2, 0))

        frame = ttk.Frame(self)
        frame.pack(fill="both", expand=True, padx=16, pady=(8, 0))
        self.tree = ttk.Treeview(frame, columns=[c[0] for c in self.COLS],
                                 show="headings", selectmode="browse")
        for key, title, width in self.COLS:
            self.tree.heading(key, text=title)
            self.tree.column(key, width=width,
                             anchor="w" if key in ("id", "label", "created") else "center")
        bar = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=bar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        for tag, colour in (("ok", INK), ("mismatch", MUTED), ("unknown", WARN)):
            self.tree.tag_configure(tag, foreground=colour)
        self.tree.bind("<Double-1>", lambda _e: self._accept())
        self.tree.bind("<Return>", lambda _e: self._accept())

        actions = ttk.Frame(self)
        actions.pack(fill="x", padx=16, pady=10)
        self.surface.primary(actions, "用这个会话", self._accept)
        self.surface.secondary(actions, "取消", self.destroy)

        self._refresh()
        if current:
            try:
                self.tree.selection_set(str(Path(current)))
                self.tree.see(str(Path(current)))
            except tk.TclError:
                pass

    # ---------- 三态 ----------
    def _state_of(self, row: dict) -> tuple[str, str]:
        if self.mode is None:
            # 跨装置：不存在"不匹配"这回事，三态退回一态
            return "匹配", ""
        raw = row.get("mode")
        if raw is None or str(raw).strip() == "":
            return "未记录", "这份会话没记录装置模式，用之前请确认是哪套装置录的"
        if str(raw) == self.mode:
            return "匹配", ""
        return "不匹配", f"这段会话是按 {raw} 几何录的，用本窗解算会得到错的深度"

    def _refresh(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        shown = 0
        for row in self._rows:
            state, _why = self._state_of(row)
            if state == "不匹配" and not self.var_all.get():
                continue
            tag = {"匹配": "ok", "不匹配": "mismatch", "未记录": "unknown"}[state]
            label = row["label"] or ("（无标签）" if state != "未记录" else "（无标签）")
            mode_text = row["mode"] if row["mode"] else "未记录"
            self.tree.insert("", "end", iid=str(row["path"]),
                             values=(row["id"], label, row["created"],
                                     mode_text, row["frames"]),
                             tags=(tag,))
            shown += 1
        if shown == 0:
            self.tree.insert("", "end", iid="__none__",
                             values=("（没有会话）", "", "", "", ""), tags=("mismatch",))

    def _accept(self) -> None:
        sel = self.tree.selection()
        if not sel or sel[0] == "__none__":
            return
        row = next((r for r in self._rows if str(r["path"]) == sel[0]), None)
        if row is None:
            return
        _state, why = self._state_of(row)
        self.on_pick(sel[0], why)
        self.destroy()


class StereoWindow(tk.Toplevel):
    """双目四步窗。``mode`` ∈ {``parallel``, ``angled``}。"""

    def __init__(self, app, *, mode: str = "parallel") -> None:
        super().__init__(app)
        if mode not in ("parallel", "angled"):
            raise ValueError(f"mode 只能是 parallel/angled，收到 {mode!r}")
        self.app = app
        self.mode = mode
        self.configure(bg=BG)
        # 四步加起来超过 720 px：④ 的说明会被底部状态条压住（实测 120x6 px）。
        # 给到 800 让四步展开后仍留出状态条的位置；屏幕逻辑高 1000，放得下。
        self.geometry("920x800")
        self.minsize(800, 580)
        self.surface = Surface(self)

        # 记住上次填的值。**方法不存**：它由窗口模式决定（平行=joint、大角度=chain），
        # 存下来会让"同一窗口上次改过的方法"和"窗口代表哪种几何"打架。
        prefs = app.prefs
        default_method = "joint" if mode == "parallel" else "chain"
        self.var_method = tk.StringVar(value=METHODS[default_method][0])
        self.var_target = tk.StringVar(value=prefs.get_str("stereo.target", "40"))
        self.var_cols = tk.StringVar(value=prefs.get_str("board.cols", "8"))
        self.var_rows = tk.StringVar(value=prefs.get_str("board.rows", "5"))
        self.var_square = tk.StringVar(value=prefs.get_str("board.square_mm", "25"))
        self.var_stationary = tk.StringVar(
            value=prefs.get_str("stereo.stationary_px", "3"))
        self.var_sharpness = tk.StringVar(
            value=prefs.get_str("stereo.min_sharpness", "800"))
        self.var_relaxed = tk.BooleanVar(
            value=prefs.get_bool("stereo.relaxed", False))
        self.var_left = tk.StringVar(value="left")
        self.var_right = tk.StringVar(value="right")
        self.var_session = tk.StringVar(value="")
        self.var_label = tk.StringVar(value="")
        self.var_seconds = tk.StringVar(value="10")

        # 先占底部：pack 从容器边缘往里分配，先 pack 的优先拿到空间。
        # 反过来（先 pack 主内容区、再 pack 状态条）状态条会**盖住内容** ——
        # 2026-10-28 截图里第 4 步卡片就被盖掉了一半。
        self.status = StatusBar(self, on_log=app.open_console,
                                on_stop=self._stop_my_tasks)
        self.status.pack(fill="x", side="bottom")
        page = ScrollPage(self, self.surface)
        page.pack(fill="both", expand=True)
        body = page.content
        self._build_step1(body)
        self._build_step2(body)
        self._build_step3(body)
        self._build_step4(body)
        self._refresh_state()

    # ------------------------------------------------------------------ #
    # ① 装置检查
    # ------------------------------------------------------------------ #
    def _build_step1(self, parent) -> None:
        premise = ("平行双目的前提：两路光轴**近平行**（夹角很小），"
                   "且基线远小于工作距离。校正后左右图的行是对齐的。"
                   if self.mode == "parallel" else
                   "大角度双目的前提：两路夹角大（接近直角），不必平行、"
                   "也不必在每个姿态都看到同一块板。深度靠原始像素三角测量。")
        box = self.surface.card(parent, "① 装置检查", premise.replace("**", ""))
        self.lbl_rig = self.surface.hint(box, "检查中…")
        # 相机身份属于 L1，**不许进折叠区**：它是装置的物理身份，不是高级参数。
        # 旧界面让人手填 left/right，填错就静默用错相机 —— 这里只显示、
        # 只能通过「换相机…」改（读 config/cameras.json）。
        self.lbl_cams = self.surface.hint(box, "读取相机…")
        actions = self.surface.action_row(box)
        self.surface.primary(actions, "检查装置", self._check_setup)
        self.surface.secondary(actions, "换相机…", self._open_cameras)
        self.surface.secondary(actions, "重新读取标定", self._reload_state)

    # ------------------------------------------------------------------ #
    # ② 标定
    # ------------------------------------------------------------------ #
    def _build_step2(self, parent) -> None:
        hint = ("这一步决定第 4 步能不能用。平行装置只需要 joint；"
                "如果两路看不到同一区域，就该去「大角度双目」窗口。"
                if self.mode == "parallel" else
                "大角度有两条路：两路还能共视就用 joint（跳过视场裁剪）；"
                "完全看不到同一区域就用单板链式 —— 板固定不动，左右各拍一组，"
                "再用同一块板把两路串起来。链式产物直接能在第 4 步用。")
        box = self.surface.card(parent, "② 标定", hint)
        self.lbl_calib = self.surface.hint(box, "检查中…")

        # L1：目标对数 + 板参数（一行）
        line = ttk.Frame(box)
        line.grid(row=self.surface._next_row(box), column=0, columnspan=9,
                  sticky="w", pady=(4, 0))
        ttk.Label(line, text="目标对数").pack(side="left")
        ttk.Entry(line, textvariable=self.var_target, width=5).pack(
            side="left", padx=(8, 14))
        ttk.Label(line, text="棋盘内角点").pack(side="left")
        ttk.Entry(line, textvariable=self.var_cols, width=4).pack(
            side="left", padx=(8, 3))
        ttk.Label(line, text="×").pack(side="left")
        ttk.Entry(line, textvariable=self.var_rows, width=4).pack(
            side="left", padx=(3, 12))
        ttk.Label(line, text="方格 mm").pack(side="left")
        ttk.Entry(line, textvariable=self.var_square, width=5).pack(
            side="left", padx=(8, 0))

        # 方法：平行窗只有一种（写死说明），大角度窗给下拉
        if self.mode == "angled":
            line2 = ttk.Frame(box)
            line2.grid(row=self.surface._next_row(box), column=0, columnspan=9,
                       sticky="w", pady=(6, 0))
            ttk.Label(line2, text="标定方法").pack(side="left")
            cb = ttk.Combobox(line2, textvariable=self.var_method, width=34,
                              state="readonly",
                              values=[v[0] for v in METHODS.values()])
            cb.pack(side="left", padx=(8, 0))
            cb.bind("<<ComboboxSelected>>", lambda _e: self._refresh_method_hint())
            # 必须走 surface.hint：手工 ttk.Label 既没有 wraplength、也不在
            # 换行登记表里，长说明会被 grid 单元格直接裁掉（2026-10-28 截图里
            # 「…本窗口要的 angled」末尾的 d 就是这么丢的）。
            self.lbl_method = self.surface.hint(box, "")
        else:
            self.lbl_method = None
            self.surface.hint(
                box, "本窗口只用两路共视（joint）。若两路无法同时看到同一块板，"
                     "请回门厅进「大角度双目」——在那里用单板链式。")

        # L2：高级（默认收起）
        adv = CollapsibleSection(box, "高级：采集门限与相机", expanded=False,
                                 summary="停稳 / 清晰度 / 放宽 / 相机")
        adv.grid(row=self.surface._next_row(box), column=0, columnspan=9,
                 sticky="we", pady=(8, 0))
        inner = adv.content
        line3 = ttk.Frame(inner)
        line3.pack(anchor="w", pady=(4, 0))
        ttk.Label(line3, text="停稳上限 px").pack(side="left")
        ttk.Entry(line3, textvariable=self.var_stationary, width=5).pack(
            side="left", padx=(6, 12))
        ttk.Label(line3, text="清晰度下限").pack(side="left")
        ttk.Entry(line3, textvariable=self.var_sharpness, width=6).pack(
            side="left", padx=(6, 0))
        ttk.Checkbutton(inner, text="放宽模式（手持拍摄：只放松面积/边距/亮度四项）",
                        variable=self.var_relaxed).pack(anchor="w", pady=(6, 0))
        ttk.Label(inner,
                  text="停稳与清晰度填了就以你填的为准：放宽模式不会覆盖这两个值。",
                  style="Hint.TLabel").pack(anchor="w")
        line4 = ttk.Frame(inner)
        line4.pack(anchor="w", pady=(6, 4))
        ttk.Label(line4, text="左相机").pack(side="left")
        ttk.Entry(line4, textvariable=self.var_left, width=10).pack(
            side="left", padx=(6, 12))
        ttk.Label(line4, text="右相机").pack(side="left")
        ttk.Entry(line4, textvariable=self.var_right, width=10).pack(
            side="left", padx=(6, 0))

        actions = self.surface.action_row(box)
        self.surface.primary(actions, "开始标定", self._start_calibrate)

    # ------------------------------------------------------------------ #
    # ③ 采集
    # ------------------------------------------------------------------ #
    def _build_step3(self, parent) -> None:
        box = self.surface.card(
            parent, "③ 采集（录制一段，供第 4 步分析）",
            "录的是两路同步视频。球进入画面那一刻开始，别中途改帧率。"
            "如果实验需要同步信号，用闪光：倒计时里不要闪，听到开始再闪。")
        line = ttk.Frame(box)
        line.grid(row=self.surface._next_row(box), column=0, columnspan=9,
                  sticky="w", pady=(4, 0))
        ttk.Label(line, text="时长 秒").pack(side="left")
        ttk.Entry(line, textvariable=self.var_seconds, width=6).pack(
            side="left", padx=(8, 14))
        ttk.Label(line, text="标签").pack(side="left")
        ttk.Entry(line, textvariable=self.var_label, width=16).pack(
            side="left", padx=(8, 0))
        actions = self.surface.action_row(box)
        self.surface.primary(actions, "开始录制", self._start_record)

    # ------------------------------------------------------------------ #
    # ④ 分析
    # ------------------------------------------------------------------ #
    def _build_step4(self, parent) -> None:
        box = self.surface.card(
            parent, "④ 分析（对一段已录好的会话跑三维）",
            "这里跑的是离线重建：读会话里的两路视频，按标定解出三维轨迹。"
            "会话自己记录了它是哪种装置；本窗口会强制按 "
            + ("平行" if self.mode == "parallel" else "大角度") +
            " 解算 —— 这正是这个窗口存在的意义。")
        line = ttk.Frame(box)
        line.grid(row=self.surface._next_row(box), column=0, columnspan=9,
                  sticky="w", pady=(4, 0))
        ttk.Label(line, text="会话目录").pack(side="left")
        ttk.Entry(line, textvariable=self.var_session, width=46).pack(
            side="left", padx=(8, 8))
        ttk.Button(line, text="选择…", command=self._pick_session).pack(
            side="left")
        actions = self.surface.action_row(box)
        self.surface.primary(actions, "跑离线三维", self._start_offline)
        row2 = self.surface.action_row(box)
        self.surface.secondary(row2, "点选球心", self._start_seed)
        self.surface.secondary(row2, "打开 3D 查看器", self._open_viewer)
        # 第 ④ 步只管"当前会话"；要换一段历史会话从中间接着做，去工作台
        self.surface.secondary(row2, "历史会话…", self._open_bench)

    # ------------------------------------------------------------------ #
    # 状态
    # ------------------------------------------------------------------ #
    def _refresh_state(self) -> None:
        st = self.app.state
        ready, text = (st.parallel_status() if self.mode == "parallel"
                       else st.angled_status())
        clean = text.replace("**", "")
        self.lbl_calib.configure(
            text=("标定：" if ready else "标定：") + clean,
            style="Ok.TLabel" if ready else "Warn.TLabel")
        prof = getattr(st.profile, "name", None)
        mode = st.stereo_mode() or "未记录"
        base = st.baseline_mm()
        angle = st.angle_deg()
        bits = [f"装置档案：{prof or '未设置'}",
                f"标定模式：{mode}"]
        if base is not None:
            bits.append(f"基线 {base:.0f} mm")
        if angle is not None:
            bits.append(f"夹角 {angle:.1f}°")
        self.lbl_rig.configure(text="｜".join(bits))
        # 相机身份：只读 config/cameras.json，改它只能通过「换相机…」
        try:
            self.lbl_cams.configure(
                text="相机：" + st.camera_role_text("left")
                     + "｜" + st.camera_role_text("right"))
        except Exception:                                   # noqa: BLE001
            self.lbl_cams.configure(text="相机：读取失败（config/cameras.json）")
        if self.lbl_method is not None:
            self._refresh_method_hint()
        self.status.set_status("就绪" if ready else "第 2 步的标定还不匹配本窗口",
                               "ok" if ready else "warn")

    def _refresh_method_hint(self) -> None:
        if self.lbl_method is None:
            return
        label = self.var_method.get()
        key = next((k for k, v in METHODS.items() if v[0] == label), "chain")
        self.lbl_method.configure(text=METHODS[key][1].replace("**", ""))

    def _method_key(self) -> str:
        label = self.var_method.get()
        return next((k for k, v in METHODS.items() if v[0] == label), "joint")

    def _reload_state(self) -> None:
        self.app.state.reload()
        self._refresh_state()

    # ------------------------------------------------------------------ #
    # 动作
    # ------------------------------------------------------------------ #
    def _open_cameras(self) -> None:
        """打开相机设备管理（复用 apps/tools/cameras.py，不新写第二个界面）。"""
        self.app.tasks.start(
            [sys.executable, str(ROOT / "apps" / "tools" / "cameras.py"),
             "identify"], "相机设备管理")
        self.status.set_status(
            "已打开相机设备管理：每台最多等 30 秒（慢相机实测 ~24 秒），"
            "日志里会逐台报进度，别关窗口。只想绑左右可直接用 pin（不开相机，秒级）",
            "info")

    def _check_setup(self) -> None:
        cmd = [sys.executable, str(ROOT / "apps" / "tools" / "check_stereo_setup.py")]
        self.app.tasks.start(cmd, "双目装置检查")
        self.status.set_status("正在检查装置（只报告，不改配置），结果会打到日志",
                               "info")

    def _start_calibrate(self) -> None:
        method = self._method_key()
        cmd = calibrate_args(
            method,
            {"cols": self.var_cols.get(), "rows": self.var_rows.get(),
             "square": self.var_square.get(), "target": self.var_target.get()},
            self.var_left.get(), self.var_right.get(), self.mode,
            stationary=self.var_stationary.get(),
            sharpness=self.var_sharpness.get(),
            relaxed=bool(self.var_relaxed.get()))
        if method == "chain":
            self.status.set_status(
                f"链式标定要四个目录的图：{CHAIN_ROOT.name}\\left、right、"
                f"ref_left、ref_right。缺图会报缺哪一段", "info")
        else:
            self.status.set_status("正在标定：会打开两个取景窗，板停稳自动存图",
                                   "info")
        self.app.tasks.start(cmd, "双目标定", on_done=self._on_calibrated)

    def _on_calibrated(self, _task) -> None:
        self.app.state.reload()
        self._refresh_state()

    def _start_record(self) -> None:
        # 录制仍走 `--stereo-mode auto`（= 按标定），所以不会出现"模式选错"
        # 那种必然失败；但**标定本身与本窗模式不匹配**时录出来也是白录
        # （录的是另一套几何，第 4 步必然跑不通）。这里先拦一次，文案与
        # 第 2 步的状态栏同一来源，不另写一份判据。
        ready, why = (self.app.state.parallel_status() if self.mode == "parallel"
                      else self.app.state.angled_status())
        if not ready:
            self.status.set_status(
                f"先解决标定再录：{why.replace('**', '')}", "warn")
            return
        out = record_out_dir(self.var_label.get())
        cmd = record_args(left=self.var_left.get(), right=self.var_right.get(),
                          duration=self.var_seconds.get(), out=str(out))
        self.app.tasks.start(cmd, "双目录制")
        self.status.set_status(f"正在录制 {self.var_seconds.get() or '10'} 秒 → "
                               f"{out.name}；录完用它做第 4 步", "info")

    def _pick_session(self) -> None:
        """打开会话选择器（按本窗模式三态过滤）。"""
        SessionPicker(self, mode=self.mode, on_pick=self._set_session,
                      current=self.var_session.get().strip())

    def _set_session(self, path: str, why: str = "") -> None:
        """选中一个会话。``why`` 非空表示"它不属于本窗模式"，必须说出来。"""
        self.var_session.set(path)
        name = Path(path).name
        if why:
            self.status.set_status(f"已选 {name} —— 注意：{why}", "warn")
        else:
            self.status.set_status(f"已选会话：{name}", "ok")

    def _start_offline(self) -> None:
        session = self.var_session.get().strip()
        if not session:
            self.status.set_status("先选一个会话（点「选择…」）", "warn")
            return
        cmd = [sys.executable, str(ROOT / "apps" / "run_stereo_video.py"),
               session, "--stereo-mode", self.mode]
        self.app.tasks.start(cmd, "离线三维重建")
        self.status.set_status(f"正在按 {self.mode} 解算 {Path(session).name}…",
                               "info")

    def _start_seed(self) -> None:
        session = self.var_session.get().strip()
        if not session:
            self.status.set_status("先选一个会话（点「选择…」）", "warn")
            return
        saved = pick_seeds(self, session)
        if saved is None:
            self.status.set_status("没有保存球心（窗口被关掉或左右没点完）", "warn")
            return
        self.status.set_status(
            f"球心已保存：{Path(saved).name}；现在跑第 ④ 步会用 specular 解算", "ok")

    def _open_bench(self) -> None:
        """打开会话工作台。带上当前会话，用户接着做 ①–⑤ 而不是再选一遍。"""
        self.app.open_bench(self.var_session.get().strip() or None)

    def _open_viewer(self) -> None:
        """打开 3D 查看器 —— 必须带源：优先当前会话，其次最近一个真有结果的门录。

        view_3d 的 source 是必填位置参数，不给就 exit 2（表现为"点了没反应"）。
        """
        session = self.var_session.get().strip()
        if session:
            cand = Path(session)
            src = next((cand / rel for rel in ("derived/visualization.json",
                                               "visualization.json")
                        if (cand / rel).is_file()), None)
            if src is None:
                self.status.set_status(
                    f"这个会话还没有 visualization.json：先在上面的"
                    f"「跑离线三维」跑一次，再来查看", "warn")
                return
        else:
            src = self.app.recent_visualization()
            if src is None:
                self.status.set_status(
                    "先选一个会话并跑一次「跑离线三维」，才有三维结果可看", "warn")
                return
        self.app.tasks.start(
            [sys.executable, str(ROOT / "apps" / "view_3d.py"), str(src)],
            "3D 查看器")
        self.status.set_status(
            f"正在打开 3D 查看器：{src.parent.name}"
            f"（窗口内「数据源」可换任意会话）", "info")

    def on_close(self) -> None:
        """关窗时记住输入（会话路径与相机角色不存：前者每次新选，后者从
        ``config/cameras.json`` 读）。"""
        p = self.app.prefs
        p.set("stereo.target", self.var_target.get().strip())
        p.set("board.cols", self.var_cols.get().strip())
        p.set("board.rows", self.var_rows.get().strip())
        p.set("board.square_mm", self.var_square.get().strip())
        p.set("stereo.stationary_px", self.var_stationary.get().strip())
        p.set("stereo.min_sharpness", self.var_sharpness.get().strip())
        p.set("stereo.relaxed", bool(self.var_relaxed.get()))
        p.set("stereo.seconds", self.var_seconds.get().strip())
        p.save()

    def _stop_my_tasks(self) -> None:
        ok = self.app.tasks.stop_all(graceful=True)
        self.status.set_status(f"已确认停掉 {ok} 个任务", "ok" if ok else "warn")
