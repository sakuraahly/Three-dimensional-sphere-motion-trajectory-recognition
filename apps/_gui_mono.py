"""单目模式窗：① 内参 → ② 球与量程 → ③ 开始。

为什么单目窗是第一个做出来的模式窗（2026-10-27 计划书 §4 P3）：它没有历史包袱。
双目那边的每一个控件都对应着一段踩坑史，而单目在旧界面上**只有一个"焦距 fx"
输入框**——把它做成完整三步，等于用新框架从零走一遍，能最快暴露框架本身的毛病。

三条设计约束

1. **精度必须写在脸上**。单目靠球径先验 ``Z = f·D/(2r)``，0.6 m 处误差约 ±7 mm，
   比双目差一个量级。界面给的是**算出来的误差预算**，不是"精度够用"这类软话术
   —— 让人拿单目结果去当双目用，比不给他这个功能更糟。
2. **没有标定就说没有标定**。旧界面那个「焦距fx(留空自动)」里的"自动"实际是
   ``geometry.py`` 按 60° 视场猜的（1% 焦距误差 = 6 mm @0.6 m）。这里三态分明：
   已标定 / 手动覆盖 / 未标定（会猜）。
3. **不假装有入口**。``apps/calibrate_mono.py`` 还没落地时，标定按钮明确写
   "入口待落地"并给出现阶段可用的替代品与其局限，而不是做一个点了没反应的按钮。
"""

from __future__ import annotations

import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from apps._gui_widgets import (BG, OK, WARN, ScrollPage, StatusBar, Surface)
from apps._task_runner import ROOT

__all__ = ["MonoWindow"]

MONO_CALIB_APP = ROOT / "apps" / "calibrate_mono.py"
MONO_CALIB_JSON = ROOT / "config" / "mono_calibration.json"
FOCAL_ESTIMATOR = ROOT / "apps" / "tools" / "calibrate_focal.py"
CALIB_FRAME_DIR = ROOT / "outputs" / "mono_calib_frames"

# 误差预算的假设（写在界面上，不许藏起来）
ASSUMED = {"dr_px": 0.2, "dD_mm": 0.2, "df_rel": 0.01}


def error_budget(fx: float, diameter_mm: float,
                 distances_m: tuple[float, ...] = (0.3, 0.6, 1.0)) -> list[tuple[float, float]]:
    """每个工作距离上的 Z 误差（mm，RSS 合成）。

    ``Z = f·D/(2r)`` 取对数微分：``dZ/Z = df/f + dD/D − dr/r``，
    而 ``r = f·D/(2Z)``，所以三项都能用已知量算出来。
    这是**解析传播**，不是实测噪声 —— 它回答"量级是多少"，
    不回答"这台机器上具体多少"（后者要静置 30 s 量 std(r)）。
    """
    out: list[tuple[float, float]] = []
    for z_m in distances_m:
        z = z_m * 1000.0
        r = fx * diameter_mm / (2.0 * z)                 # 像素半径
        if r <= 0:
            continue
        d_df = z * ASSUMED["df_rel"]                     # 焦距项
        d_dD = z * (ASSUMED["dD_mm"] / diameter_mm)      # 球径项
        d_dr = z * (ASSUMED["dr_px"] / r)                # 半径噪声项
        rss = (d_df ** 2 + d_dD ** 2 + d_dr ** 2) ** 0.5
        out.append((z_m, rss))
    return out


class MonoWindow(tk.Toplevel):
    """单目三步窗。关闭即回门厅（门厅一直开着）。"""

    def __init__(self, app) -> None:
        super().__init__(app)
        self.app = app
        self.configure(bg=BG)
        # 三步加起来约 712 px 高（实测 ① 10–338、② 348–512、③ 522–712），
        # 700 的窗口会把第③步的下半截切掉——用户看到的是"方框被遮挡"。
        # 给到 820 一次看全三步；屏幕逻辑高 1000，放得下。
        self.geometry("880x820")
        self.minsize(780, 560)
        self.surface = Surface(self)

        # 记住上次填的值（GuiPrefs 只存输入；派生值在那里会被拒绝）
        prefs = app.prefs
        # 默认取配置里的第一个角色。**不能默认空**：`capture_board` 的
        # `resolve_camera_arg(None)` 返回 None、不自动挑，会报"没能把 None
        # 解析成相机索引"；也不能默认 "main"（配置里没有这个角色）。
        # 2026-10-28 两个坑都踩着过。
        _roles = list(app.state.camera_roles().keys())
        _saved = prefs.get_str("mono.camera", "")
        # 存过的值可能已失效（角色被改名/删了，或早期版本存过 "main"）——
        # 失效就回落第一个角色，别把一个不存在的角色名带进命令行。
        if _saved not in _roles:
            _saved = _roles[0] if _roles else "0"
        self.var_camera = tk.StringVar(value=_saved)
        self.var_cols = tk.StringVar(value=prefs.get_str("board.cols", "8"))
        self.var_rows = tk.StringVar(value=prefs.get_str("board.rows", "5"))
        self.var_square = tk.StringVar(value=prefs.get_str("board.square_mm", "25"))
        self.var_target = tk.StringVar(value=prefs.get_str("mono.target", "15"))
        self.var_diameter = tk.StringVar(
            value=prefs.get_str("mono.diameter_mm", "40"))
        self.var_focal = tk.StringVar(
            value=prefs.get_str("mono.focal_override", ""))   # 空 = 用标定值
        self.var_record_secs = tk.StringVar(
            value=prefs.get_str("mono.record_seconds", "10"))

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
        self._refresh_state()

    # ------------------------------------------------------------------ #
    # ① 内参
    # ------------------------------------------------------------------ #
    def _build_step1(self, parent) -> None:
        box = self.surface.card(
            parent, "① 标定内参（一次就够）",
            "单目测深只用三样东西：焦距、球直径、球的像素半径。焦距必须标，"
            "不能猜 —— 1% 的焦距误差在 0.6 m 处就是 6 mm。"
            "拍图用棋盘：一张 A4 打印的棋盘，在画面里上下左右都摆一摆，"
            "至少 8 个不同姿态（挤在中央会让畸变系数失真）。")
        # 状态行
        # 状态行走 surface.hint：登记换行 + 由游标分配行号（不写死 row=1）
        self.lbl_calib = self.surface.hint(box, "检查中…")
        # 相机身份放 L1（同双目窗的道理）：没有它就没法确认用的是哪台
        self.lbl_cams = self.surface.hint(box, "读取相机…")
        # 选哪台相机：值取自 config/cameras.json 的角色，**不给手打**
        pick = ttk.Frame(box)
        pick.grid(row=self.surface._next_row(box), column=0, columnspan=9,
                  sticky="w", pady=(0, 4))
        ttk.Label(pick, text="用这台相机").pack(side="left")
        _roles = list(self.app.state.camera_roles().keys()) or ["0", "1", "2"]
        ttk.Combobox(pick, values=_roles, width=14, state="readonly",
                     textvariable=self.var_camera).pack(side="left", padx=(8, 0))

        # L1：只有板参数（三个必需值）合在一行 —— 它们是一件事（这块板长什么样），
        # 拆成三行会让首屏输入控件数超标，也让"8×5、25 mm"读起来不像一句话。
        line = ttk.Frame(box)
        line.grid(row=self.surface._next_row(box), column=0, columnspan=9,
                  sticky="w", pady=(4, 0))
        ttk.Label(line, text="棋盘内角点").pack(side="left")
        ttk.Entry(line, textvariable=self.var_cols, width=5).pack(
            side="left", padx=(8, 4))
        ttk.Label(line, text="×").pack(side="left")
        ttk.Entry(line, textvariable=self.var_rows, width=5).pack(
            side="left", padx=(4, 12))
        ttk.Label(line, text="方格边长 mm").pack(side="left")
        ttk.Entry(line, textvariable=self.var_square, width=6).pack(
            side="left", padx=(8, 0))
        self.surface.hint(box, "内角点 = 棋盘格内部交叉点个数，不是格子数。"
                               "手上那张纸写着 8×5 就填 8 和 5；方格边长按实际量。")

        actions = self.surface.action_row(box)
        self.btn_capture = self.surface.primary(actions, "拍标定图",
                                                self._start_capture)
        self.surface.secondary(actions, "换相机…", self._open_cameras)
        self.btn_calibrate = self.surface.secondary(actions, "标定内参",
                                                    self._start_calibrate)
        if not MONO_CALIB_APP.exists():
            self.btn_calibrate.state(["disabled"])
        row2 = self.surface.action_row(box)
        self.surface.secondary(row2, "打开标定图文件夹", self._open_frames_dir)
        self.btn_estimate = self.surface.secondary(
            row2, "替代：单点焦距估计", self._start_focal_estimate)

    # ------------------------------------------------------------------ #
    # ② 球与量程
    # ------------------------------------------------------------------ #
    def _build_step2(self, parent) -> None:
        box = self.surface.card(
            parent, "② 球与量程",
            "球直径是单目测深的第二个已知量：同一颗球在画面里占多少像素，"
            "就能反推它多远。直径量错 0.5%，深度就错 0.5%。"
            "下面的误差预算是按解析公式算的，用来判断这套方案够不够你的实验。")
        row = ttk.Frame(box)
        row.grid(row=1, column=0, columnspan=9, sticky="w", pady=(4, 0))
        ttk.Label(row, text="球直径 mm").pack(side="left")
        ent = ttk.Entry(row, textvariable=self.var_diameter, width=8)
        ent.pack(side="left", padx=(8, 16))
        ent.bind("<KeyRelease>", lambda _e: self._refresh_budget())
        ttk.Label(row, text="焦距 fx（留空用标定值）").pack(side="left")
        ent2 = ttk.Entry(row, textvariable=self.var_focal, width=8)
        ent2.pack(side="left", padx=(8, 0))
        ent2.bind("<KeyRelease>", lambda _e: self._refresh_budget())
        self.surface.hint(box, "手动填 fx 会覆盖标定值 —— 只在你确定相机换过、"
                               "又来不及重标时用。")

        # 误差预算行走游标（原来写死 row=20，加了控件就会撞）
        self.lbl_budget = self.surface.hint(box, "填一个球直径才能算误差预算")

    # ------------------------------------------------------------------ #
    # ③ 开始
    # ------------------------------------------------------------------ #
    def _build_step3(self, parent) -> None:
        box = self.surface.card(
            parent, "③ 录制与识别",
            "先录一段：**只写相机原图，不叠任何标注**，所以录完可以换参数反复检测。"
            "录好的文件在 data/mono_takes 下，用「离线视频」选它就能跑。"
            "「开始实时」是跳过录制、边看边测。没有内参时焦距按 60° 视场估算，"
            "状态条会写明 —— 那种结果只能看趋势，不能当测量。".replace("**", ""))
        line = ttk.Frame(box)
        line.grid(row=self.surface._next_row(box), column=0, columnspan=9,
                  sticky="w", pady=(4, 0))
        ttk.Label(line, text="录制时长 秒").pack(side="left")
        ttk.Entry(line, textvariable=self.var_record_secs, width=6).pack(
            side="left", padx=(8, 12))
        ttk.Label(line, text="（0 = 按 q 停止）", style="Hint.TLabel").pack(side="left")
        actions = self.surface.action_row(box)
        self.surface.primary(actions, "录制视频", self._start_record)
        self.surface.secondary(actions, "开始实时", self._start_realtime)
        self.surface.secondary(actions, "离线视频", self._start_video)

    def _start_record(self) -> None:
        """录制单目视频（只写原图）。录完可直接用「离线视频」检测。"""
        out_dir = ROOT / "data" / "mono_takes"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{time.strftime('%Y%m%d_%H%M%S')}.mkv"
        secs = self.var_record_secs.get().strip() or "10"
        cmd = [sys.executable, str(ROOT / "apps" / "record_mono.py"),
               "--camera", self.var_camera.get().strip() or "main",
               "--duration", secs,
               "--out", str(out)]
        self.app.tasks.start(cmd, f"单目录制 {secs}s")
        self.status.set_status(
            f"已开始录制 {secs} 秒 → {out.name}（有倒计时，看到 REC 才开始动）。"
            f"录完点「离线视频」选这个文件", "info")

    # ------------------------------------------------------------------ #
    # 状态
    # ------------------------------------------------------------------ #
    def _open_cameras(self) -> None:
        """打开相机设备管理（复用 apps/tools/cameras.py）。"""
        self.app.tasks.start(
            [sys.executable, str(ROOT / "apps" / "tools" / "cameras.py"),
             "identify"], "相机设备管理")
        self.status.set_status(
            "相机设备管理已启动：逐个索引确认；单目默认用自动挑的那台，"
            "要固定成 main 角色请在管理界面里 pin", "info")

    def _refresh_state(self) -> None:
        st = self.app.state
        try:
            cam = self.var_camera.get().strip()
            self.lbl_cams.configure(
                text="相机：" + ("自动挑（优先外接）" if not cam
                                 else st.camera_role_text(cam)))
        except Exception:                                   # noqa: BLE001
            self.lbl_cams.configure(text="相机：读取失败（config/cameras.json）")
        ready, text = st.mono_status()
        clean = text.replace("**", "")
        self.lbl_calib.configure(
            text=("内参：" if ready else "内参：") + clean,
            style="Ok.TLabel" if ready else "Warn.TLabel")
        if not MONO_CALIB_APP.exists():
            self.status.set_status(
                "标定入口 apps/calibrate_mono.py 还没落地（计划书 P0c/P1b）；"
                "现在可以先拍图，标定请用「替代：单点焦距估计」或经典界面",
                "warn")
        else:
            self.status.set_status("就绪", "ok")
        self._refresh_budget()

    def _current_fx(self) -> float | None:
        """焦距三态：手动覆盖 > 标定值 > None（会按 60° 视场猜）。"""
        manual = self.var_focal.get().strip()
        if manual:
            try:
                v = float(manual)
                return v if v > 0 else None
            except ValueError:
                return None
        fx = (self.app.state.mono or {}).get("fx")
        return float(fx) if isinstance(fx, (int, float)) and fx > 0 else None

    def _refresh_budget(self) -> None:
        try:
            d = float(self.var_diameter.get().strip() or "0")
        except ValueError:
            d = 0.0
        if d <= 0:
            self.lbl_budget.configure(text="填一个球直径才能算误差预算",
                                      style="Warn.TLabel")
            return
        fx = self._current_fx()
        if fx is None:
            # 60° 视场、1920 宽 → fx ≈ 1663 px（geometry.py 的同一个假设）
            fx = 1663.0
            src = "焦距未标定，按 60° 视场估（fx≈1663 px）"
            style = "Warn.TLabel"
        else:
            src = f"按 fx = {fx:.0f} px 算"
            style = "Hint.TLabel"
        parts = []
        for z_m, mm in error_budget(fx, d):
            parts.append(f"{z_m:.1f} m → ±{mm:.1f} mm")
        self.lbl_budget.configure(
            text=f"误差预算（半径噪声 0.2 px、球径误差 0.2 mm、焦距 1%；{src}）："
                 + "；".join(parts),
            style=style)

    # ------------------------------------------------------------------ #
    # 动作
    # ------------------------------------------------------------------ #
    def _common_camera_args(self) -> list[str]:
        """空或 auto 一律**不传** ``--camera``，让 capture_board / record_mono
        自己按"优先外接"挑一台。"""
        cam = self.var_camera.get().strip()
        if not cam or cam.lower() in ("auto", "none", "自动"):
            return ["--camera", "0"]      # 兜底：0 号相机（工具不接受空）
        return ["--camera", cam]

    def _start_capture(self) -> None:
        CALIB_FRAME_DIR.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(ROOT / "apps" / "tools" / "capture_board.py"),
               "--out", str(CALIB_FRAME_DIR),
               "--cols", self.var_cols.get().strip() or "8",
               "--rows", self.var_rows.get().strip() or "5",
               "--square", self.var_square.get().strip() or "25",
               "--target", self.var_target.get().strip() or "15",
               "--auto"] + self._common_camera_args()
        self.app.tasks.start(cmd, "拍单目标定图")
        self.status.set_status(
            f"已开始拍图：板在每个姿态停稳约 1 秒就会自动存；"
            f"存够 {self.var_target.get() or '15'} 张自动结束。图在 "
            f"{CALIB_FRAME_DIR.name}\\", "info")

    def _start_calibrate(self) -> None:
        if not MONO_CALIB_APP.exists():
            self.status.set_status(
                "标定入口还没落地：现在可以用「替代：单点焦距估计」，"
                "但它假设主点在画面中心、不产生畸变系数，球径误差会并入 fx", "warn")
            return
        cmd = [sys.executable, str(MONO_CALIB_APP),
               "--dir", str(CALIB_FRAME_DIR),
               "--cols", self.var_cols.get().strip() or "8",
               "--rows", self.var_rows.get().strip() or "5",
               "--square", self.var_square.get().strip() or "25"]
        self.app.tasks.start(cmd, "单目内参标定", on_done=self._on_calibrated)
        self.status.set_status("正在标定内参…", "info")

    def _on_calibrated(self, task) -> None:
        self.app.state.reload()
        self._refresh_state()

    def _start_focal_estimate(self) -> None:
        cmd = [sys.executable, str(FOCAL_ESTIMATOR)] + self._common_camera_args()
        self.app.tasks.start(cmd, "单点焦距估计")
        self.status.set_status(
            "单点焦距估计会打开一个取景窗：把球放在已知距离上点一下。"
            "注意它把球径误差算进 fx，再正算深度时会乘第二次", "warn")

    def _open_frames_dir(self) -> None:
        try:
            CALIB_FRAME_DIR.mkdir(parents=True, exist_ok=True)
            import os
            os.startfile(str(CALIB_FRAME_DIR))           # noqa: S606
        except Exception as exc:                        # noqa: BLE001
            self.status.set_status(f"打不开文件夹：{exc}", "warn")

    def _start_realtime(self) -> None:
        cmd = [sys.executable, str(ROOT / "apps" / "run_realtime.py"),
               "--ball-diameter", self.var_diameter.get().strip() or "40"]
        cmd += self._common_camera_args()
        fx = self._current_fx()
        if fx is not None and self.var_focal.get().strip():
            cmd += ["--focal", str(int(fx))]            # 只有手动覆盖才传
        if MONO_CALIB_JSON.exists():
            cmd += ["--config", str(MONO_CALIB_JSON)]
        self.app.tasks.start(cmd, "实时单目识别")
        self.status.set_status("已启动实时识别，画面在它自己的窗口里；"
                               "要停请到「日志与任务」里中断", "info")

    def _start_video(self) -> None:
        """离线视频：**先选文件再启动**。

        run_video.py 的 path 是必填位置参数，它自己不会弹选择框 ——
        原来那句"会提示选视频文件"是错的，盲启的结果是 exit 2、点了没反应。
        """
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="选择要处理的视频", parent=self,
            filetypes=[("视频", "*.mp4 *.avi *.mkv *.mov *.m4v"),
                       ("所有文件", "*.*")])
        if not path:
            self.status.set_status("没有选视频，已取消", "info")
            return
        cmd = [sys.executable, str(ROOT / "apps" / "run_video.py"), path,
               "--ball-diameter", self.var_diameter.get().strip() or "40"]
        self.app.tasks.start(cmd, f"离线单目：{Path(path).name}")
        self.status.set_status(f"已启动离线单目：{Path(path).name}", "info")

    def on_close(self) -> None:
        """关窗时记住输入。**只存输入**：fx/就绪度由 GuiPrefs 主动拒绝。"""
        p = self.app.prefs
        p.set("mono.camera", self.var_camera.get().strip())
        p.set("board.cols", self.var_cols.get().strip())
        p.set("board.rows", self.var_rows.get().strip())
        p.set("board.square_mm", self.var_square.get().strip())
        p.set("mono.target", self.var_target.get().strip())
        p.set("mono.diameter_mm", self.var_diameter.get().strip())
        p.set("mono.focal_override", self.var_focal.get().strip())
        p.set("mono.record_seconds", self.var_record_secs.get().strip())
        p.save()

    def _stop_my_tasks(self) -> None:
        ok = self.app.tasks.stop_all(graceful=True)
        self.status.set_status(f"已确认停掉 {ok} 个任务", "ok" if ok else "warn")
