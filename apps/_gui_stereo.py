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

import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from apps._gui_widgets import BG, ScrollPage, StatusBar, Surface
from apps._collapsible import CollapsibleSection
from apps._task_runner import ROOT

__all__ = ["StereoWindow"]

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


class StereoWindow(tk.Toplevel):
    """双目四步窗。``mode`` ∈ {``parallel``, ``angled``}。"""

    def __init__(self, app, *, mode: str = "parallel") -> None:
        super().__init__(app)
        if mode not in ("parallel", "angled"):
            raise ValueError(f"mode 只能是 parallel/angled，收到 {mode!r}")
        self.app = app
        self.mode = mode
        self.configure(bg=BG)
        self.geometry("920x720")
        self.minsize(800, 580)
        self.surface = Surface(self)

        default_method = "joint" if mode == "parallel" else "chain"
        self.var_method = tk.StringVar(value=METHODS[default_method][0])
        self.var_target = tk.StringVar(value="40")
        self.var_cols = tk.StringVar(value="8")
        self.var_rows = tk.StringVar(value="5")
        self.var_square = tk.StringVar(value="25")
        self.var_stationary = tk.StringVar(value="3")
        self.var_sharpness = tk.StringVar(value="800")
        self.var_relaxed = tk.BooleanVar(value=False)
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
        ttk.Button(line, text="最近的…", command=self._pick_latest).pack(
            side="left")
        actions = self.surface.action_row(box)
        self.surface.primary(actions, "跑离线三维", self._start_offline)
        row2 = self.surface.action_row(box)
        self.surface.secondary(row2, "点选球心", self._start_seed)
        self.surface.secondary(row2, "打开 3D 查看器", self._open_viewer)

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
            "相机设备管理已启动：逐个索引显示画面，确认哪台是左、哪台是右；"
            "改完回来点「重新读取标定」刷新显示", "info")

    def _check_setup(self) -> None:
        cmd = [sys.executable, str(ROOT / "apps" / "tools" / "check_stereo_setup.py")]
        self.app.tasks.start(cmd, "双目装置检查")
        self.status.set_status("正在检查装置（只报告，不改配置），结果会打到日志",
                               "info")

    def _chain_dirs(self) -> dict[str, Path]:
        return {"left": CHAIN_ROOT / "left", "right": CHAIN_ROOT / "right",
                "ref_left": CHAIN_ROOT / "ref_left",
                "ref_right": CHAIN_ROOT / "ref_right"}

    def _start_calibrate(self) -> None:
        method = self._method_key()
        board = ["--cols", self.var_cols.get().strip() or "8",
                 "--rows", self.var_rows.get().strip() or "5",
                 "--square", self.var_square.get().strip() or "25",
                 "--target", self.var_target.get().strip() or "40"]
        cmd = [sys.executable, str(ROOT / "apps" / "calibrate_stereo.py")]
        if method == "chain":
            dirs = self._chain_dirs()
            cmd += ["--method", "chain", "--same-board",
                    "--left-dir", str(dirs["left"]),
                    "--right-dir", str(dirs["right"])] + board
            # 参考段的双路同拍图作为"同一块板"的证据
            pair = dirs["ref_left"]
            if pair.exists():
                cmd += ["--pair-frame", str(pair), "--pair-camera", "left"]
            self.status.set_status(
                f"链式标定要四个目录的图：{CHAIN_ROOT.name}\\left、right、"
                f"ref_left、ref_right。缺图会报缺哪一段", "info")
        else:
            cmd += ["--method", "joint"] + board
            if method == "joint_skip_fov":
                cmd += ["--stereo-mode", "angled"]
            else:
                cmd += ["--stereo-mode", self.mode]
            cmd += ["--left", self.var_left.get().strip() or "left",
                    "--right", self.var_right.get().strip() or "right"]
            self.status.set_status("正在标定：会打开两个取景窗，板停稳自动存图",
                                   "info")
        # 高级项：只有用户真的填了才下传（空 = 让后端用自己的默认值）
        if self.var_stationary.get().strip():
            cmd += ["--stationary-px", self.var_stationary.get().strip()]
        if self.var_sharpness.get().strip():
            cmd += ["--min-sharpness", self.var_sharpness.get().strip()]
        if self.var_relaxed.get():
            cmd += ["--relaxed"]
        self.app.tasks.start(cmd, "双目标定", on_done=self._on_calibrated)

    def _on_calibrated(self, _task) -> None:
        self.app.state.reload()
        self._refresh_state()

    def _start_record(self) -> None:
        out = ROOT / "data" / (self.var_label.get().strip() or "stereo_session")
        cmd = [sys.executable, str(ROOT / "apps" / "record_stereo.py"),
               "--left", self.var_left.get().strip() or "left",
               "--right", self.var_right.get().strip() or "right",
               "--duration", self.var_seconds.get().strip() or "10",
               "--out", str(out)]
        self.app.tasks.start(cmd, "双目录制")
        self.status.set_status(f"正在录制 {self.var_seconds.get() or '10'} 秒 → "
                               f"{out.name}；录完用它做第 4 步", "info")

    def _pick_latest(self) -> None:
        """把最近一次录制目录填进来（不弹窗让人手抄路径）。"""
        base = ROOT / "data"
        try:
            dirs = [d for d in base.iterdir() if d.is_dir()]
            dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        except OSError:
            dirs = []
        if not dirs:
            self.status.set_status("data\\ 下还没有会话目录，先做第 3 步录制", "warn")
            return
        self.var_session.set(str(dirs[0]))
        self.status.set_status(f"已选最近的会话：{dirs[0].name}", "ok")

    def _start_offline(self) -> None:
        session = self.var_session.get().strip()
        if not session:
            self.status.set_status("先选一个会话目录（点「最近的…」）", "warn")
            return
        cmd = [sys.executable, str(ROOT / "apps" / "run_stereo_video.py"),
               session, "--stereo-mode", self.mode]
        self.app.tasks.start(cmd, "离线三维重建")
        self.status.set_status(f"正在按 {self.mode} 解算 {Path(session).name}…",
                               "info")

    def _start_seed(self) -> None:
        session = self.var_session.get().strip()
        if not session:
            self.status.set_status("先选一个会话目录", "warn")
            return
        cmd = [sys.executable, str(ROOT / "apps" / "seed_picker.py"), session]
        self.app.tasks.start(cmd, "点选球心")
        self.status.set_status("点选球心的窗口会单独打开；点完回到这里跑第 4 步",
                               "info")

    def _open_viewer(self) -> None:
        self.app.tasks.start(
            [sys.executable, str(ROOT / "apps" / "view_3d.py")], "3D 查看器")

    def _stop_my_tasks(self) -> None:
        ok = self.app.tasks.stop_all(graceful=True)
        self.status.set_status(f"已确认停掉 {ok} 个任务", "ok" if ok else "warn")
