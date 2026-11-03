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
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from apps._gui_widgets import (BALL_METAL, BALL_PLASTIC, METHOD_METAL,
                               PRESET_NONE, BG, INK, MUTED, WARN, BallSettings,
                               ScrollPage, StatusBar, Surface, fit_to_content)
from apps._collapsible import CollapsibleSection
# argv 的唯一产地在 `_actions`（见该模块 docstring）：这里只 import 名字，
# 不再自己拼命令行 —— 否则同一个动作会有两份参数清单，已经因此出过三次缺陷。
from apps._actions import (CHAIN_ROOT, board_args, calibrate_args,
                           chain_dirs, check_chain_inputs, check_setup_args,
                           capture_chain_frames, offline_args, record_args)
from apps import _actions
from sphere_tracker.config import ColorConfig
from sphere_tracker.paths import STEREO_ROOT, stereo_roots
from sphere_tracker.plugins import names as plugin_names

__all__ = ["StereoWindow", "SessionPicker", "scan_sessions", "pick_seeds",
           "board_args", "chain_dirs", "calibrate_args", "record_args",
           "record_out_dir", "check_setup_args", "session_name_of",
           "detector_methods", "preset_choices"]


def detector_methods() -> tuple[str, ...]:
    """双目/工作台的检测方法候选：**注册表里的全部实现**，一个不多一个不少。

    为什么不在界面里写一份名字清单（R7）：这份清单以前在手抄的四份里漂过
    （GUI 7 项 / ``run_stereo_video`` 5 项 / ``run_all`` 4 项）。判据
    （``tests/selftest_gui_budget.py``）断言的就是"下拉项 == 注册表"，
    所以这里**只准**转发，不许过滤、不许补字面量。

    不带单目那个 ``smart``：它是单目窗自己的"0 检出换 hough 重跑"行为，
    ``run_stereo_video --method`` 的 choices 里没有它。
    """
    return tuple(plugin_names("detector"))


def preset_choices() -> tuple[str, ...]:
    """颜色预设候选：唯一来源 ``ColorConfig.preset_names()``（同 ``detector_methods``）。"""
    return tuple(ColorConfig.preset_names())


#: 方法在界面上的说法。键与 `calibrate_stereo.py --method` 一致（chain_dual 是
#: 界面专用键，底层仍是 chain，只是双板模式：--board-b-dir + --pair-frame）。
METHODS = {
    "joint": ("两路共视（joint）",
              "两路在**每个姿态**都要同时看到同一块板。适合平行或充分共视的装置。"),
    "joint_skip_fov": ("两路共视，跳过视场裁剪（joint + no FOV）",
                       "仍是共同求解，但不按重叠视场裁剪 —— 大角度装置裁完会剩太窄。"),
    "chain": ("单板链式（chain）",
              "板不必被两路同时看到：各拍一组内参图，再用一块固定参考板把两路串起来。"
              "90 度这类几乎看不到同一区域的装置用它。产物就是本窗口要的 angled。"),
    "chain_dual": ("双板链式（chain，两板各看一块）",
                   "两块刚性固定的板（如垂直双板），左相机看 A、右相机看 B，"
                   "再拍一张两板同框图求相对位姿。适合两路各看各的板的 90° 装置。"),
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










def record_out_dir(root: Path | None = None) -> Path:
    """录制的 ``--out``：会话**根目录** ``data/stereo``（时间戳子目录由录制程序建）。

    为什么是根、不是 ``data/stereo/<时间戳>``（2026-10-31 修，缺陷 #3）
    ------------------------------------------------------------------
    ``record_stereo.py`` 的 ``--out`` 是**根**：它自己调
    ``stereo.sessions.create_session(root)``，在 root 下新建
    ``YYYYMMDD-HHMMSS``（同名已存在则加 ``-01`` 后缀）。所以：

    - 传 ``data/stereo`` → 得到 ``data/stereo/20261031-181500``，会话扫描
      （``sphere_tracker.paths.stereo_roots()`` = ``data/stereo`` + ``data/sessions``）
      一眼就能列出它，第 ④ 步的「选择…」和工作台里都看得到；
    - 传 ``data/stereo/<时间戳>`` 反而会再套一层 —— 除非在这里另写一份命名与
      去重逻辑，那就成了"目录名有两处来源"。

    上一版返回 ``data/<标签>``：既不在扫描范围里（录完永远列不出来，只能手打
    路径），目录名又固定（第二段录制落进同一个目录、和第一段混在一起）。
    标签现在走 ``--label`` 进 ``session.json``，**不再参与拼路径**。
    """
    return root if root is not None else STEREO_ROOT


def session_name_of(visualization: Path | str) -> str:
    """一份 ``visualization.json`` 属于**哪一段会话**。

    为什么不能直接 ``path.parent.name``（2026-10-31 走查缺陷 #6/#7）：
    离线重建把结果写在 ``<会话>/derived/visualization.json``，于是
    ``parent.name`` 永远是 ``derived`` —— 状态条说"正在打开 3D 查看器：derived"，
    用户看不出打开的是哪一段。``derived`` 是会话里的**子目录**，不是会话。
    """
    p = Path(visualization).parent
    return p.parent.name if p.name == "derived" else p.name




# --------------------------------------------------------------------------- #
# 会话选择：三态过滤（匹配 / 不匹配 / 未记录）
# --------------------------------------------------------------------------- #
def scan_sessions(base: Path | tuple[Path, ...] | None = None) -> list[dict]:
    """读 ``data/stereo/*/session.json``（以及旧位置 ``data/sessions/*``）的摘要。

    只读 JSON、**不构造 ``Session`` 对象**：几十个会话也能瞬间列完，
    而且不会因为某一份坏文件把整个列表打断。

    ``base`` 不传 = 扫 ``stereo_roots()``（新的在前），迁移前后都列得全；
    传一个目录/tuple 只扫那些（测试要造临时会话，不能依赖这台机器录过什么）。
    同一个会话名两边都在时只留**新的那份**（先扫到的），避免迁移中途列表出现两行同名。

    ``mode`` 字段可能是 ``None`` —— 那表示**这份会话没有记录装置模式**。
    老会话都缺这个字段（``record_stereo`` 写入 ``stereo_mode`` 是
    ``a137766`` 才加的），所以**不能默认成 parallel**。
    """
    if base is None:
        roots: tuple[Path, ...] = stereo_roots()
    elif isinstance(base, (str, Path)):
        roots = (Path(base),)
    else:
        roots = tuple(Path(b) for b in base)

    rows: list[dict] = []
    seen: set[str] = set()
    for root in roots:
        for row in _scan_one_root(root):
            if row["id"] in seen:
                continue
            seen.add(row["id"])
            rows.append(row)
    rows.sort(key=lambda r: r["id"], reverse=True)
    return rows


def _scan_one_root(root: Path) -> list[dict]:
    """扫单个根的直接子目录（新→旧按目录名），返回还没去重的行。"""
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
            "这里列出 data/stereo 下的会话（旧位置 data/sessions 也一起列）。"
            "默认只显示与当前窗口同一种装置几何的；"
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

    #: 内容区宽度。高度**不写死**：建完之后按内容量（见 ``fit_to_content``）。
    WIDTH = 920
    WIDTH_MIN = 800

    def __init__(self, app, *, mode: str = "parallel") -> None:
        super().__init__(app)
        if mode not in ("parallel", "angled"):
            raise ValueError(f"mode 只能是 parallel/angled，收到 {mode!r}")
        self.app = app
        self.mode = mode
        self.configure(bg=BG)
        # 先给一个"先按内容量的意思"的初始尺寸：这只是建窗时的占位，
        # 真正的高度在 __init__ 末尾由 fit_to_content 按内容算出来。
        # **不要**再写死具体高度：写死的高度只代表"上次放得下"，
        # 内容一长（2026-10-31 实测 881/950）就把第 ④ 步裁在窗口下沿。
        self.geometry(f"{self.WIDTH}x600")
        self.minsize(self.WIDTH_MIN, 560)
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
        self._chain_step = 0     # 「采集标定板图」三段流程的游标（内参左→右→参考段）
        # 可信半径锚点（默认关）：用一次「点选球心」的左右两点定出"两路半径该差多少"，
        # 据此拒掉半径离谱的假配对。开关放在这里而不是"永远开"：它要求会话里
        # **有人点过**球心，没点过就开只会得到一条报错；默认关才符合"点了才有"。
        self.var_anchor = tk.BooleanVar(value=False)
        # 球种类与检测参数（2026-11-03）：控件摆在 ④，但值是**用户的选择**，
        # 所以关窗时和板参数一起存（不存"当前会话"，那是磁盘现状）。
        self.var_preset = tk.StringVar(
            value=prefs.get_str("stereo.preset", PRESET_NONE) or PRESET_NONE)
        self.var_min_radius = tk.StringVar(
            value=prefs.get_str("stereo.min_radius", ""))
        self.var_max_radius = tk.StringVar(
            value=prefs.get_str("stereo.max_radius", ""))
        self.var_label = tk.StringVar(value="")
        self.var_seconds = tk.StringVar(value="10")

        # 先占底部：pack 从容器边缘往里分配，先 pack 的优先拿到空间。
        # 反过来（先 pack 主内容区、再 pack 状态条）状态条会**盖住内容** ——
        # 2026-10-28 截图里第 4 步卡片就被盖掉了一半。
        self.status = StatusBar(self, on_log=app.open_console,
                                on_stop=self._stop_my_tasks)
        self.status.pack(fill="x", side="bottom")
        self._page = ScrollPage(self, self.surface)
        self._page.pack(fill="both", expand=True)
        body = self._page.content
        self._cam_text = "相机：读取中…"
        self._build_step1(body)
        self._build_step2(body)
        self._build_step3(body)
        self._build_step4(body)
        # 左/右相机一改，「检查装置」的可用性立刻跟着变：
        # 否则用户清空一边后按钮还是亮的，点下去才被告知"填上再来"。
        for _var in (self.var_left, self.var_right):
            _var.trace_add("write", lambda *_a: self._sync_check_enabled())
        self._refresh_state()
        # 最后一步：按**内容**定高度（上限 = 屏幕可用高度）。必须在
        # `_refresh_state` 之后 —— 那一步会写状态行/原因行，量早了就是量一个
        # 还没写完的界面，算出来的高度会偏小、第 ④ 步照样被裁。
        fit_to_content(self, self._page, self.status, width=self.WIDTH,
                       screen_h=self.app.work_area()[1])

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
        self.btn_check = self.surface.primary(actions, "检查装置", self._check_setup)
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
        if self.mode == "angled":
            # 链式标定（chain）的前置步骤：90° 装置缺的就是这两个入口 ——
            # 以前选 chain 只有「开始标定」，但后端要 `outputs/chain_frames/`
            # 下的四段图，GUI 没地方采，点了标定只能看"缺图"报错。
            self.surface.secondary(actions, "采集标定板图…", self._capture_chain_frames)
            self.surface.secondary(actions, "检查链式输入", self._check_chain_inputs)

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
            "离线重建：读会话的两路视频，按标定解出三维轨迹。本窗口强制按 "
            + ("平行" if self.mode == "parallel" else "大角度") + " 解算。")
        # 球种类与检测参数（2026-11-03）：以前这一段只有一个会话输入框，
        # 检测参数全靠 `run_stereo_video` 自己的默认值 —— 换了一颗纯色球的人
        # 在界面上**没有任何地方**能说"这是红球"，只能去命令行加 `--preset`。
        # 摆在第④步而不是第②③步：它只影响"解算"，不影响标定与录制。
        self.ball = BallSettings(
            self.surface, box,
            methods=detector_methods(), presets=preset_choices(),
            kind=(BALL_METAL
                  if _actions.has_seeds(self.var_session.get().strip())
                  else BALL_PLASTIC),
            preset_var=self.var_preset,
            radius_vars=(self.var_min_radius, self.var_max_radius),
            on_change=self._refresh_ball_note)
        line = ttk.Frame(box)
        line.grid(row=self.surface._next_row(box), column=0, columnspan=9,
                  sticky="w", pady=(4, 0))
        ttk.Label(line, text="会话目录").pack(side="left")
        # 输入框宽度 46 → 38：给同一行的锚点勾选框腾地方（不新增行，见下）。
        ttk.Entry(line, textvariable=self.var_session, width=38).pack(
            side="left", padx=(8, 8))
        ttk.Button(line, text="选择…", command=self._pick_session).pack(
            side="left")
        # 半径锚点开关（默认关）：勾了才传 --anchor-gate on。
        #
        # 为什么跟会话输入框**同一行**：本窗口高度按"一屏看全"定，单起一行实测会把
        # 「大角度」顶出 5 px（内容 874 > 窗口 869）；再加一行说明则是 53 px。
        # 宁可把输入框收窄，也不裁按钮、不动控件预算。
        ttk.Checkbutton(
            line, text="半径锚点门（勾了才生效）",
            variable=self.var_anchor).pack(side="left", padx=(10, 0))
        actions = self.surface.action_row(box)
        self.surface.primary(actions, "跑离线三维", self._start_offline)
        self.surface.secondary(actions, "点选球心", self._start_seed)
        self.surface.secondary(actions, "打开 3D 查看器", self._open_viewer)
        # 第 ④ 步只管"当前会话"；要换一段历史会话从中间接着做，去工作台
        self.surface.secondary(actions, "历史会话…", self._open_bench)
        # 四个按钮**同一行**（2026-10-31，缺陷 #9/#10）：
        # 原先分成两行，多出一整行按钮 + 行距。实测四步内容为 846/915 px
        # （平行/大角度），比当时的窗口 800 高 —— 第 ④ 步的按钮就被裁在
        # 窗口下沿；合并成一行后是 799/868 px。
        # 为什么不是"把次要按钮收进折叠区"：折叠区的标题本身要 28 px，
        # 只省回一小半，却把「历史会话…」藏起来（那是用户从中间接着做的
        # 唯一入口）。合并没有藏任何东西，省得也更多。
        # 一行 4 个按钮仍在预算内（L1 按钮 ≤5、主操作恰好 1）。

    # ------------------------------------------------------------------ #
    # 状态
    # ------------------------------------------------------------------ #
    def _refresh_state(self) -> None:
        st = self.app.state
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

        # 第②步是**标定**（产出动作），不是消费现有标定。三种方法在本窗口都合法
        # （joint/joint_skip_fov/chain 都会产出本窗口要的 stereo_mode），所以这里
        # **不**再用 parallel_status/angled_status 的"现有标定不匹配"来拒绝 ——
        # 那套判定是给第③步录制/第④步分析用的（它们消费现有标定）。以前把
        # angled_status 用在这里，导致大角度窗在"现有标定还是上次的 parallel"时，
        # 状态栏写"请用平行窗口"，用户想选 joint 重新标定却被误判成"不匹配"。
        existing = "尚无标定" if st.rig is None else f"当前已有标定（{mode}）"
        self.lbl_calib.configure(
            text=f"标定：{existing}。选好方法点「开始标定」即可产出本窗口需要的标定",
            style="Ok.TLabel")

        # 相机身份：只读 config/cameras.json，改它只能通过「换相机…」
        try:
            self._cam_text = ("相机：" + st.camera_role_text("left")
                              + "｜" + st.camera_role_text("right"))
        except Exception:                                   # noqa: BLE001
            self._cam_text = "相机：读取失败（config/cameras.json）"
        self._sync_check_enabled()
        if self.lbl_method is not None:
            self._refresh_method_hint()
        # 第②步本身不再有"就绪"门槛（标定随时可做）；就绪属于第③/④步。
        self.status.set_status("就绪", "ok")

    def _sync_check_enabled(self) -> None:
        """「检查装置」能不能点：只看左右相机有没有填。

        拿不到左右编号就让按钮**灰着并写出原因** —— 会让用户"点了等一个永远
        不来的结果"的按钮，比灰着的按钮更伤人（2026-10-31 缺陷 #4）。
        单独成一个方法是因为它还要跟着输入框变（见 ``__init__`` 里的 trace）。
        """
        has_roles = bool(self.var_left.get().strip()) and bool(self.var_right.get().strip())
        self.btn_check.configure(state="normal" if has_roles else "disabled")
        text = self._cam_text
        if not has_roles:
            text += "｜左/右有一边没填，「检查装置」不可用：先在「高级」里填上"
        self.lbl_cams.configure(text=text)

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
        self.app.tasks.start(_actions.cameras(), "相机设备管理")
        self.status.set_status(
            "已打开相机设备管理：每台最多等 30 秒（慢相机实测 ~24 秒），"
            "日志里会逐台报进度，别关窗口。只想绑左右可直接用 pin（不开相机，秒级）",
            "info")

    def _check_setup(self) -> None:
        """① 检查装置：带**本窗口选的**左右相机，并把结论写回状态条。

        为什么不能像上一版那样空手起进程（2026-10-31 缺陷 #4）
        --------------------------------------------------
        上一版 ``tasks.start([python, check_stereo_setup.py], "双目装置检查")``：
        一个参数都没有，``on_done`` 也没有。后果有两条，用户都看得见：

        1. 检查程序去查的相机不一定是第 ② 步要标定的那两台 —— 结论对了也是
           "另一套装置"的结论；
        2. 进程跑完（甚至退出）界面**一个字都不变**：状态条永远停在
           "正在检查装置…"，用户只能猜。

        现在：左右相机（与 ``calibrate_stereo`` 同一份取值）和棋盘参数一起
        下传，``on_done`` 按退出码回写。
        """
        left, right = self.var_left.get().strip(), self.var_right.get().strip()
        if not left or not right:
            # 正常路径下按钮这时已经是灰的（见 `_refresh_state`）；
            # 这里是"填了又清空"的兜底 —— 与其起一个必然白开的进程，不如说清原因
            self.status.set_status(
                "「检查装置」要知道检查哪两台相机：先在「高级：采集门限与相机」"
                "里填上左/右（角色名或索引）", "warn")
            return
        cmd = check_setup_args(
            left=left, right=right,
            board={"cols": self.var_cols.get(), "rows": self.var_rows.get(),
                   "square": self.var_square.get()})
        self.app.tasks.start(cmd, "双目装置检查", on_done=self._on_checked)
        self.status.set_status(
            f"正在检查装置 {left} / {right}（只报告，不改配置）：会开两路取景窗，"
            f"举着棋盘慢慢移动，按 q 可提前结束。结论会写回这里", "info")

    def _on_checked(self, task) -> None:
        """把「检查装置」的结果写回状态条 —— 成功/失败都要写，不能只有"正在…"。"""
        code = task.exit_code
        if code == 0:
            self.status.set_status(
                "装置检查通过：两路都能稳定看到棋盘，可以做第 ② 步标定"
                "（逐项数字在运行日志里）", "ok")
        elif code == 1:
            self.status.set_status(
                "装置检查没通过：运行日志里「标定前置检查结果」会写清是哪一路没看到"
                "棋盘、重叠区多大", "warn")
        elif code is None:
            self.status.set_status("装置检查被中断了，没有结论（可以再点一次）", "warn")
        else:
            self.status.set_status(
                f"装置检查没能跑起来（退出码 {code}）：多数是相机名/索引打不开，"
                f"输出在运行日志里", "warn")

    def _start_calibrate(self) -> None:
        method = self._method_key()
        # 标定写进「当前装置档案」的 calibration_path（若有），否则走默认
        # config/stereo_calibration。这样平行装置和 90° 装置各存各的标定，
        # 换装置重新标定不会覆盖另一套的数据。
        prof = getattr(self.app.state.profile, "calibration_path", None)
        cmd = calibrate_args(
            method,
            {"cols": self.var_cols.get(), "rows": self.var_rows.get(),
             "square": self.var_square.get(), "target": self.var_target.get()},
            self.var_left.get(), self.var_right.get(), self.mode,
            stationary=self.var_stationary.get(),
            sharpness=self.var_sharpness.get(),
            relaxed=bool(self.var_relaxed.get()),
            out=(prof or ""))
        if method in ("chain", "chain_dual"):
            # 先查图在不在，缺哪段直接说，别让后端跑一半才报"缺图"。
            d = chain_dirs()
            if method == "chain_dual":
                need = ("left", "board_b", "pair")   # 双板：左拍A、右拍B、同框图
            else:
                need = ("left", "right", "ref_left", "ref_right")  # 单板四段
            missing = [k for k in need if not d[k].is_dir()
                       or not any(d[k].iterdir())]
            if missing:
                self.status.set_status(
                    ("双板" if method == "chain_dual" else "链式") + "标定缺图："
                    + "、".join(missing)
                    + "。先点「采集标定板图…」拍齐，再「检查链式输入」",
                    "warn")
                return
            self.status.set_status(
                "正在链式标定（读 chain_frames 图）…", "info")
        else:
            self.status.set_status("正在标定：会打开两个取景窗，板停稳自动存图",
                                   "info")
        self.app.tasks.start(cmd, "双目标定", on_done=self._on_calibrated)

    def _capture_chain_frames(self) -> None:
        """采集链式标定的图（按当前方法分单板/双板两套流程）。

        - 单板（chain）：内参左 → 内参右 → 参考段（板不动，左右成对）；
        - 双板（chain_dual）：拍 A 板（左）→ 拍 B 板（右）→ 提示手动拍同框图。
        每段拍完回到界面继续下一段（on_done 串起来，避免取景窗抢相机）。
        """
        board_base = {"cols": self.var_cols.get(), "rows": self.var_rows.get(),
                      "square": self.var_square.get()}
        left, right = self.var_left.get(), self.var_right.get()
        d = chain_dirs()
        for p in d.values():
            p.mkdir(parents=True, exist_ok=True)

        if self._method_key() == "chain_dual":
            # 双板：A/B 两块板各拍一组（可移动），同框图需**手动**拍（capture_board
            # 只检单板，检不了同框两板）。
            steps = [
                ("双板·拍 A 板（左相机，板可移动）",
                 capture_chain_frames(stage="left", left=left, right=right,
                                      target="15", **board_base)),
                ("双板·拍 B 板（右相机，板可移动）",
                 capture_chain_frames(stage="dual_b", left=left, right=right,
                                      target="15", **board_base)),
            ]
        else:
            steps = [
                ("内参段·左（板可移动，多姿态）",
                 capture_chain_frames(stage="left", left=left, right=right,
                                      target="15", **board_base)),
                ("内参段·右（板可移动，多姿态）",
                 capture_chain_frames(stage="right", left=left, right=right,
                                      target="15", **board_base)),
                ("参考段（板**固定不动**，左右成对拍）",
                 capture_chain_frames(stage="ref", left=left, right=right,
                                      target="3", **board_base)),
            ]
        self._chain_step = 0

        def run_next(_task=None):
            if self._chain_step >= len(steps):
                if self._method_key() == "chain_dual":
                    self.status.set_status(
                        "A/B 两板图拍完。同框图需**手动**拍一张（两板同时入画）放进 "
                        f"{d['pair']}，再点「检查链式输入」「开始标定」", "info")
                else:
                    self.status.set_status(
                        "四段图拍完。点「检查链式输入」确认没问题，再点「开始标定」",
                        "info")
                return
            label, cmd = steps[self._chain_step]
            self._chain_step += 1
            self.status.set_status(f"[{self._chain_step}/{len(steps)}] {label}。"
                                   "取景窗里板停稳会自动存图；拍完关窗回到这里", "info")
            self.app.tasks.start(cmd, label, on_done=run_next)

        run_next()

    def _check_chain_inputs(self) -> None:
        """拍完先自检（姿态覆盖 / 参考段板是否移动），别等标定跑完才发现图不行。"""
        if self._method_key() == "chain_dual":
            # 双板没有对应的自检工具（check_chain_inputs 只认单板四段目录），
            # 这里只提示要手动确认同框图两板都完整入画。
            self.status.set_status(
                "双板链式没有自动自检：确认 left(A板)/board_b(B板) 各有图、"
                "pair 里有一张两板同框图，且 A/B 相对位置全程没变", "info")
            return
        cmd = check_chain_inputs(
            cols=self.var_cols.get(), rows=self.var_rows.get(),
            square=self.var_square.get())
        self.status.set_status(
            "检查链式输入（读 chain_frames 四段图，报姿态覆盖与参考段板是否移动）…",
            "info")
        self.app.tasks.start(cmd, "链式输入自检")

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
        out = record_out_dir()
        cmd = record_args(left=self.var_left.get(), right=self.var_right.get(),
                          duration=self.var_seconds.get(), out=str(out),
                          label=self.var_label.get())
        self.app.tasks.start(cmd, "双目录制")
        label = self.var_label.get().strip()
        # 落点写"根 + 新时间戳目录"：目录名由录制程序生成（它同时负责重名去重），
        # 这里不猜具体时间，但必须说清"录完在第 ④ 步能选到它" —— 上一版落进
        # `data/stereo_session`，不在扫描范围，用户录完在列表里找不到。
        self.status.set_status(
            f"预览窗口已开：确认两路都有画面后按【空格】开始（再倒计时进行录制）；"
            f"录 {self.var_seconds.get() or '10'} 秒 → data/stereo/ 下的新会话目录"
            f"（标签「{label or '空'}」）", "info")

    def _pick_session(self) -> None:
        """打开会话选择器（按本窗模式三态过滤）。"""
        SessionPicker(self, mode=self.mode, on_pick=self._set_session,
                      current=self.var_session.get().strip())

    def _set_session(self, path: str, why: str = "") -> None:
        """选中一个会话。``why`` 非空表示"它不属于本窗模式"，必须说出来。"""
        self.var_session.set(path)
        name = Path(path).name
        notes: list[str] = []
        if why:
            notes.append(why)
        # 会话里有人点过球心 = 这颗球是镜面球（只有 specular 认那些点）。
        # 替他把球种类切过去并**说出来**：不切的话下拉显示"纯色球"、
        # 命令行却因为没有种子解不出东西，用户会以为是标定坏了。
        if self.ball.kind() != BALL_METAL and _actions.has_seeds(path):
            self.ball.set_kind(BALL_METAL)
            notes.append("这段会话里有点选球心，球种类已切到「镜面金属球」"
                         "（specular 会读那些点）")
        if notes:
            self.status.set_status(f"已选 {name} —— 注意：" + "；".join(notes), "warn")
        else:
            self.status.set_status(f"已选会话：{name}", "ok")

    def _refresh_ball_note(self) -> None:
        """球种/半径改了之后的回写（只往状态条写，不新增任何行）。

        为什么不下放成一行常驻说明：第④步的下沿已经贴着视口（869 px），
        多一行就把「跑离线三维」挤出屏幕 —— 而"按钮看不见"比"少一句解释"严重。
        所以：填错 → 说清哪一项不会被采用；切档 → 说清这一档的前提。
        每档的长解释在卡片说明与 ``BallSettings`` 的 docstring 里。
        """
        problem = self.ball.problem()
        if problem:
            self.status.set_status(problem, "warn")
            return
        hint = self.ball.hint()
        if hint:
            self.status.set_status(hint, "warn")

    def _start_offline(self) -> None:
        session = self.var_session.get().strip()
        if not session:
            self.status.set_status("先选一个会话（点「选择…」）", "warn")
            return
        # 填坏了的半径范围不许静默变成"不传"：那会让用户以为限定了范围，
        # 实际是全画面在找（`run_stereo_video` 的默认是 8–200 px）。
        problem = self.ball.problem()
        if problem:
            self.status.set_status(problem, "warn")
            return
        # 勾了锚点却没点选过球心：命令行会拒绝（退出码 2），但那时用户已经等了
        # 一次进程启动。能在这里说清就不要让人等 —— 这也是"按钮看起来什么都做了"
        # 那类抱怨的来源。
        if self.var_anchor.get() and not _actions.has_seeds(session):
            self.status.set_status(
                "这个会话还没有「点选球心」，半径锚点门用不了：先点「点选球心」"
                "（左右各一次），再回来勾它；不想点就把勾去掉", "warn")
            return
        seed = Path(session) / "derived" / "seeds.json"
        method = self.ball.method()
        # 镜面球没有种子 = 必然 0 配对（specular 只在高光附近找，不会自己初始化）。
        # 与其让人等一次进程启动再看到"0 匹配"，不如在这里拦住并说清下一步。
        if self.ball.kind() == BALL_METAL and not seed.is_file():
            self.status.set_status(
                "镜面金属球要先「点选球心」（左右各点一次球心+球边）："
                "specular 靠那些点跟踪，没有它们这一段会得到 0 对匹配", "warn")
            return
        use_seed = seed.is_file() and method == METHOD_METAL
        cmd = _actions.offline_stereo(
            session, mode=self.mode, anchor=self.var_anchor.get(),
            method=method, seed_json=(seed if use_seed else ""),
            preset=self.ball.preset(),
            min_radius=self.ball.min_radius(),
            max_radius=self.ball.max_radius())
        tail = "（半径锚点门已开：先读会话里的点选中心）" if self.var_anchor.get() else ""
        if use_seed:
            tail += "（用会话里的点选球心跟踪）"
        self.status.set_status(
            f"正在按 {self.mode} 解算 {Path(session).name}（检测方法 {method}）{tail}…",
            "info")
        self.app.tasks.start(cmd, "离线三维重建")

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
            f"球心已保存：{Path(saved).name}；现在跑第 ④ 步会用 specular 解算。"
            "想同时用「半径锚点门」，把它勾上再跑（不勾就只是普通点选）", "ok")

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
                    "这个会话还没有 visualization.json：先在上面的"
                    "「跑离线三维」跑一次，再来查看", "warn")
                return
        else:
            src = self.app.recent_visualization()
            if src is None:
                self.status.set_status(
                    "先选一个会话并跑一次「跑离线三维」，才有三维结果可看", "warn")
                return
        self.app.tasks.start(
            _actions.view_3d(src),
            "3D 查看器")
        # 说**会话名**而不是 `src.parent.name`（那永远是 derived）
        self.status.set_status(
            f"正在打开 3D 查看器：{session_name_of(src)}"
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
        # 球种类与检测参数是**输入**：换了一颗红球的人不该每次开窗重填一遍。
        # （球种类本身不存：它由检测方法反推，存了就是同一件事的两份值。）
        p.set("stereo.preset", self.var_preset.get().strip() or PRESET_NONE)
        p.set("stereo.min_radius", self.var_min_radius.get().strip())
        p.set("stereo.max_radius", self.var_max_radius.get().strip())
        p.save()

    def _stop_my_tasks(self) -> None:
        ok = self.app.tasks.stop_all(graceful=True)
        self.status.set_status(f"已确认停掉 {ok} 个任务", "ok" if ok else "warn")
