"""球形物体三维运动轨迹识别 —— 图形化启动器（人类友好型）。

给不熟悉命令行的使用者一个双击即可用的入口，按"三步走"引导：
    第 1 步  要不要做相机标定（可跳过，已标定过就选跳过）
    第 2 步  选参数（摄像头、球直径、检测方法、颜色预设、预测视野）
    第 3 步  一键开始轨迹检测（实时摄像头 / 离线视频）

本程序不实现任何算法，只负责"收集参数 + 调用 apps/ 下已有程序 + 显示日志"，
保证界面与经过回归测试的底层链路始终一致。

交互约定（与项目其余工具一致）：
    两台相机左右的定义 —— **人面对镜头时，左手边那台为"左"**。
    第一次用请先点『相机设备管理（识别/固定）』逐个确认，之后左/右下拉选角色名
    left / right 即可 —— 角色按**设备身份**（不是编号）绑定，换 USB 口也不会用错。

运行：
    C:\\Users\\<USER>\\miniforge3\\envs\\myenv\\python.exe 启动程序.py
    （或直接双击 启动程序.bat）
"""

from __future__ import annotations

import queue
import re
import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

ROOT = Path(__file__).resolve().parent
while not (ROOT / "apps").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PY = sys.executable          # 用"当前解释器"，避免裸 python 解析到 msys64

ACCENT = "#2563eb"
INK = "#1f2937"
MUTED = "#6b7280"
BG = "#eef2f8"

METHODS = ["fusion", "color", "hough", "motion", "yolo", "yolo_sam"]
PRESETS = ["不使用", "white_ball", "orange_ball", "green_ball",
           "blue_ball", "red_ball", "yellow_ball"]
# 左右下拉的默认项：直接给**角色名**（left/right），绑定的设备身份不随 USB 口漂移。
# 数字编号仍可手选，但换一次接口就可能指向别的相机。
CAMERAS = ["left", "right", "auto", "0", "1", "2"]
# 单目相机：默认 auto（启动时自动挑最优），也可选 main 角色名或具体编号。
MONO_CAMERAS = ["auto", "main", "0", "1", "2"]
RESOLUTIONS = ["1920x1080", "1280x720", "960x540", "640x480"]


def _cam_arg(raw: str) -> str:
    """下拉里可能是 `0 | aoni webcam A25 | 6&...`，取前面那个编号；否则原样返回角色名。"""
    m = re.match(r"\s*(\d+)", raw or "")
    return m.group(1) if m else (raw or "").strip()


class Launcher(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("球形物体三维运动轨迹识别 · 启动器")
        self.configure(bg=BG)
        self.geometry("900x760")
        self.minsize(840, 700)

        self.proc: subprocess.Popen | None = None
        self.log_queue: queue.Queue = queue.Queue()

        self._init_style()
        self._build_header()
        self._build_step1()
        self._build_step2()
        self._build_step3()
        self._build_log()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(120, self._poll)
        self.after(200, self._refresh_cameras)     # 启动时就把当前设备填进下拉

    # ---------- 外观 ----------
    def _init_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        self.option_add("*Font", ("Microsoft YaHei UI", 10))
        style.configure("Card.TLabelframe", background=BG, borderwidth=0)
        style.configure("Card.TLabelframe.Label", foreground=ACCENT,
                        font=("Microsoft YaHei UI", 11, "bold"), background=BG)
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=INK)
        style.configure("Hint.TLabel", background=BG, foreground=MUTED,
                        font=("Microsoft YaHei UI", 9))
        style.configure("Go.TButton", font=("Microsoft YaHei UI", 11, "bold"), padding=10)
        style.configure("TButton", padding=6)
        style.configure("TRadiobutton", background=BG)
        style.configure("TCheckbutton", background=BG)

    def _build_header(self) -> None:
        head = tk.Frame(self, bg=ACCENT)
        head.pack(fill="x")
        tk.Label(head, text="球形物体三维运动轨迹识别",
                 bg=ACCENT, fg="white",
                 font=("Microsoft YaHei UI", 17, "bold")).pack(anchor="w", padx=20, pady=(14, 2))
        tk.Label(head, text="检测 → 跟踪 → 三维定位 → 轨迹预测    三步即可开始",
                 bg=ACCENT, fg="#dbeafe",
                 font=("Microsoft YaHei UI", 10)).pack(anchor="w", padx=22, pady=(0, 14))

    def _card(self, title: str, hint: str) -> ttk.Frame:
        box = ttk.LabelFrame(self, text=title, style="Card.TLabelframe", padding=(14, 8))
        box.pack(fill="x", padx=16, pady=(10, 0))
        ttk.Label(box, text=hint, style="Hint.TLabel").grid(
            row=0, column=0, columnspan=7, sticky="w", pady=(0, 8))
        return box

    # ---------- 第 1 步：标定 ----------
    def _build_step1(self) -> None:
        box = self._card(
            "第 1 步 · 相机标定",
            "左右约定：人面对镜头时，左手边那台为“左”。第一次用请先点"
            "『相机设备管理（识别/固定）』，程序会逐个显示画面让你确认哪台是左、哪台是右；"
            "笔记本内置 / 红外这类**不打算用**的相机，就在那个画面里按 n 标成「无关」"
            "（自动挑相机会跳过它）。确认后左右下拉选 left / right 就行，不用再管编号"
            "（编号会随插口漂移）。")
        self.calib_mode = tk.StringVar(value="skip")
        ttk.Radiobutton(box, text="跳过标定（我已完成标定 / 先快速体验）",
                        variable=self.calib_mode, value="skip").grid(
            row=1, column=0, columnspan=4, sticky="w")
        ttk.Radiobutton(box, text="执行双目标定（需要两台相机 + 打印棋盘格）",
                        variable=self.calib_mode, value="calibrate").grid(
            row=2, column=0, columnspan=4, sticky="w")
        ttk.Radiobutton(box, text="复检标定（检查装置有没有被碰动）",
                        variable=self.calib_mode, value="verify").grid(
            row=3, column=0, columnspan=4, sticky="w")

        ttk.Label(box, text="左相机").grid(row=4, column=0, sticky="e", pady=(8, 0))
        self.calib_left = ttk.Combobox(box, values=CAMERAS, width=9, state="readonly")
        self.calib_left.set("left")
        self.calib_left.grid(row=4, column=1, sticky="w", padx=(4, 12), pady=(8, 0))
        ttk.Label(box, text="右相机").grid(row=4, column=2, sticky="e", pady=(8, 0))
        self.calib_right = ttk.Combobox(box, values=CAMERAS, width=9, state="readonly")
        self.calib_right.set("right")
        self.calib_right.grid(row=4, column=3, sticky="w", padx=(4, 12), pady=(8, 0))
        ttk.Label(box, text="两路统一分辨率").grid(row=4, column=4, sticky="e", pady=(8, 0))
        self.calib_res = ttk.Combobox(box, values=RESOLUTIONS, width=10, state="readonly")
        self.calib_res.set("1920x1080")
        self.calib_res.grid(row=4, column=5, sticky="w", padx=(4, 12), pady=(8, 0))
        ttk.Label(box, text="帧率").grid(row=4, column=6, sticky="e", pady=(8, 0))
        self.calib_fps = ttk.Entry(box, width=5)
        self.calib_fps.insert(0, "60")
        self.calib_fps.grid(row=4, column=7, sticky="w", padx=(4, 12), pady=(8, 0))

        ttk.Button(box, text="运行标定 / 复检", command=self._run_calibration).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(10, 2))
        ttk.Button(box, text="标定前置检查（两路是否都看到棋盘）",
                   command=self._run_precheck).grid(
            row=5, column=2, columnspan=3, sticky="w", pady=(10, 2))
        ttk.Button(box, text="刷新设备列表", command=self._refresh_cameras).grid(
            row=5, column=5, columnspan=3, sticky="w", pady=(10, 2))
        ttk.Button(box, text="相机探测（核对两路参数是否一致）",
                   command=self._run_probe).grid(
            row=6, column=0, columnspan=3, sticky="w", pady=(2, 2))
        ttk.Button(box, text="相机设备管理（识别 / 固定 / 标无关）",
                   command=self._run_cameras_identify).grid(
            row=6, column=3, columnspan=2, sticky="w", pady=(2, 2))
        ttk.Button(box, text="检查相机映射", command=self._run_cameras_check).grid(
            row=6, column=5, columnspan=2, sticky="w", pady=(2, 2))
        ttk.Button(box, text="「无关」相机清单（自动挑选会跳过）",
                   command=self._run_cameras_ignored).grid(
            row=7, column=0, columnspan=3, sticky="w", pady=(2, 2))
        ttk.Button(box, text="清空「无关」标记", command=self._run_cameras_unignore_all).grid(
            row=7, column=3, columnspan=2, sticky="w", pady=(2, 2))

    # ---------- 第 2 步：参数 ----------
    def _build_step2(self) -> None:
        box = self._card("第 2 步 · 选择参数",
                         "不确定就保持默认。三维测距必须填对球直径——填错多少，距离就错多少。")
        ttk.Label(box, text="摄像头编号").grid(row=1, column=0, sticky="e", pady=4)
        self.camera = ttk.Combobox(box, values=MONO_CAMERAS, width=8, state="readonly")
        self.camera.set("auto")
        self.camera.grid(row=1, column=1, sticky="w", padx=(4, 20), pady=4)

        ttk.Label(box, text="球直径(米)").grid(row=1, column=2, sticky="e", pady=4)
        self.diameter = ttk.Entry(box, width=8)
        self.diameter.insert(0, "0.04")
        self.diameter.grid(row=1, column=3, sticky="w", padx=(4, 20), pady=4)

        ttk.Label(box, text="检测方法").grid(row=1, column=4, sticky="e", pady=4)
        self.method = ttk.Combobox(box, values=METHODS, width=10, state="readonly")
        self.method.set("fusion")
        self.method.grid(row=1, column=5, sticky="w", padx=(4, 4), pady=4)

        ttk.Label(box, text="颜色预设").grid(row=2, column=0, sticky="e", pady=4)
        self.preset = ttk.Combobox(box, values=PRESETS, width=5, state="readonly")
        self.preset.set("不使用")
        self.preset.grid(row=2, column=1, sticky="w", padx=(4, 20), pady=4)

        ttk.Label(box, text="预测视野(秒)").grid(row=2, column=2, sticky="e", pady=4)
        self.horizon = ttk.Entry(box, width=8)
        self.horizon.insert(0, "0.6")
        self.horizon.grid(row=2, column=3, sticky="w", padx=(4, 20), pady=4)

        ttk.Label(box, text="焦距fx(留空自动)").grid(row=2, column=4, sticky="e", pady=4)
        self.focal = ttk.Entry(box, width=10)
        self.focal.grid(row=2, column=5, sticky="w", padx=(4, 4), pady=4)

        self.show_3d = tk.BooleanVar(value=True)
        ttk.Checkbutton(box, text="启用三维估计（需填球直径）",
                        variable=self.show_3d).grid(row=3, column=0, columnspan=3,
                                                    sticky="w", pady=(6, 0))

    # ---------- 第 3 步：开始 ----------
    def _build_step3(self) -> None:
        box = self._card("第 3 步 · 开始轨迹检测",
                         "实时检测会弹出摄像头窗口；窗口内按 q 退出、r 重置轨迹、c 切换方法。")
        ttk.Button(box, text="启动实时摄像头检测", style="Go.TButton",
                   command=self._start_realtime).grid(row=1, column=0, padx=(0, 12), pady=4)
        ttk.Button(box, text="打开离线视频", style="Go.TButton",
                   command=self._start_video).grid(row=1, column=1, padx=(0, 12), pady=4)
        ttk.Button(box, text="一键自测环境", command=self._run_selftest).grid(
            row=1, column=2, pady=4)
        self.status = ttk.Label(box, text="就绪", style="Hint.TLabel")
        self.status.grid(row=1, column=3, sticky="e", padx=10)

    # ---------- 运行日志 ----------
    def _build_log(self) -> None:
        box = ttk.LabelFrame(self, text="运行日志", style="Card.TLabelframe", padding=(10, 6))
        box.pack(fill="both", expand=True, padx=16, pady=(10, 14))
        self.log = tk.Text(box, height=12, bg="#0f172a", fg="#e2e8f0",
                           insertbackground="#e2e8f0", relief="flat", wrap="word",
                           font=("Consolas", 9))
        self.log.pack(side="left", fill="both", expand=True)
        bar = ttk.Scrollbar(box, command=self.log.yview)
        bar.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=bar.set, state="disabled")
        ttk.Button(box, text="清空", command=self._clear_log, width=6).pack(
            side="bottom", anchor="e", pady=(6, 0))

    # ---------- 参数收集 ----------
    def _common_args(self) -> list[str]:
        args: list[str] = []
        preset = self.preset.get()
        if preset and preset != "不使用":
            args += ["--preset", preset]
        if self.diameter.get().strip():
            args += ["--ball-diameter", self.diameter.get().strip()]
        if self.horizon.get().strip():
            args += ["--horizon", self.horizon.get().strip()]
        if self.focal.get().strip():
            args += ["--focal", self.focal.get().strip()]
        if not self.show_3d.get():
            args += ["--no-3d"]
        return args

    # ---------- 动作 ----------
    def _start_realtime(self) -> None:
        cmd = [PY, str(ROOT / "apps" / "run_realtime.py"),
               "--camera", _cam_arg(self.camera.get()),
               "--method", self.method.get()]
        cfg = ROOT / "config" / "sphere_config.json"
        if cfg.exists():
            cmd += ["--config", str(cfg)]
        cmd += self._common_args()
        self._run(cmd, "实时摄像头检测")

    def _start_video(self) -> None:
        path = filedialog.askopenfilename(
            title="选择要分析的视频",
            initialdir=str(ROOT / "assets" / "videos"),
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")])
        if not path:
            return
        stamp = Path(path).stem
        out = ROOT / "outputs" / "videos" / f"{stamp}_annotated.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [PY, str(ROOT / "apps" / "run_video.py"), path,
               "--out", str(out),
               "--csv", str(ROOT / "outputs" / f"{stamp}_traj.csv"),
               "--method", self.method.get()]
        cmd += self._common_args()
        self._append(f"结果视频将保存到：{out}")
        self._run(cmd, "离线视频分析")

    def _confirm_stereo_cameras(self, left: str, right: str) -> bool:
        """左/右选中内置或红外相机时提醒一句：这种相机没法刚性固定在横梁上，标定会白做。

        解析失败（角色还没固定、枚举不可用等）就静默放行，不挡用户的路。
        """
        try:
            from sphere_tracker import camera_registry as cr
            devices = cr.list_devices()
        except Exception as exc:                             # noqa: BLE001
            self._append(f"[提示] 跳过内置/红外检查：{exc}")
            return True
        by_order = {d.order: d for d in devices}

        for role, raw in (("左", left), ("右", right)):
            try:
                index = cr.resolve_camera_arg(_cam_arg(raw), interactive=False)
            except RuntimeError:
                continue                                     # 角色没固定/对不上：不拦
            if index is None:
                continue
            entry = by_order.get(index)
            name = entry.name if entry else ""
            if cr.is_avoided(name):
                return messagebox.askyesno(
                    "确认用内置 / 红外相机？",
                    f"{role}相机（编号 {index}，{name or '未知名'}）看起来是笔记本内置或红外相机，"
                    "不适合做双目：它没法固定在同一根横梁上，装置一动标定结果就作废。\n\n"
                    "双目请用外接相机。确定仍要继续吗？")
        return True

    def _run_calibration(self) -> None:
        mode = self.calib_mode.get()
        if mode == "skip":
            messagebox.showinfo("已跳过标定",
                                "使用现有 / 默认参数直接检测。若三维距离不准，请先执行标定。")
            return
        left_raw, right_raw = self.calib_left.get(), self.calib_right.get()
        if not self._confirm_stereo_cameras(left_raw, right_raw):
            return
        left, right = _cam_arg(left_raw), _cam_arg(right_raw)
        width, height = self.calib_res.get().split("x")
        fps = self.calib_fps.get().strip() or "60"
        if mode == "calibrate":
            cmd = [PY, str(ROOT / "apps" / "calibrate_stereo.py"),
                   "--left", left, "--right", right,
                   "--cols", "8", "--rows", "5", "--square", "25",
                   "--width", width, "--height", height, "--fps", fps]
            self._run(cmd, f"双目标定（左{left} / 右{right}，收满约 25 对后自动标定）")
        else:
            calib = ROOT / "config" / "stereo_calibration"
            if not calib.with_suffix(".npz").exists():
                messagebox.showwarning("找不到标定文件",
                                       "还没有标定结果，请先执行一次双目标定。")
                return
            cmd = [PY, str(ROOT / "apps" / "verify_stereo.py"),
                   "--calib", str(calib), "--left", left, "--right", right,
                   "--width", width, "--height", height, "--fps", fps]
            self._run(cmd, "标定复检（判断装置是否被碰动）")

    def _run_probe(self) -> None:
        cmd = [PY, str(ROOT / "apps" / "tools" / "list_cameras.py"), "--max-index", "3"]
        self._run(cmd, "相机探测（核对两路分辨率 / 帧率是否一致）")

    def _run_precheck(self) -> None:
        left_raw, right_raw = self.calib_left.get(), self.calib_right.get()
        if not self._confirm_stereo_cameras(left_raw, right_raw):
            return
        left, right = _cam_arg(left_raw), _cam_arg(right_raw)
        width, height = self.calib_res.get().split("x")
        fps = self.calib_fps.get().strip() or "60"
        cmd = [PY, str(ROOT / "apps" / "tools" / "check_stereo_setup.py"),
               "--left", left, "--right", right,
               "--width", width, "--height", height, "--fps", fps]
        self._run(cmd, f"标定前置检查（左{left} / 右{right}：两路是否都看到棋盘）")

    def _refresh_cameras(self) -> None:
        """把下拉项换成"当前实际设备"（编号 | 名字 | 端口），并保留角色名选项。"""
        try:
            from sphere_tracker import camera_registry as cr
            devices = cr.list_devices()
            reg = cr.load_registry()
        except Exception as exc:                             # noqa: BLE001
            self._append(f"[错误] 读取设备列表失败：{exc}")
            return

        def label(d) -> str:
            # 标了「无关」的在名字后面挂个记号，免得用户纳闷"自动挑相机为什么不选它"
            mark = "  [无关]" if cr.is_device_ignored(d, reg) else ""
            return f"{d.order} | {d.name or '(无名字)'} | {d.port or '-'}{mark}"

        items = [label(d) for d in devices]
        stereo_values = ["left", "right", "auto"] + items
        mono_values = ["auto", "main"] + items
        if not devices:                                      # 非 Windows / 枚举不可用
            stereo_values += ["0", "1", "2"]
            mono_values += ["0", "1", "2"]

        self.calib_left.configure(values=stereo_values)
        self.calib_right.configure(values=stereo_values)
        self.camera.configure(values=mono_values)

        if devices:
            self._append(f"已刷新设备列表（{len(devices)} 台）：")
            self._append(cr.format_device_table(devices, reg=reg))
            self._append("左/右可直接选 left / right（按设备身份，换 USB 口也不漂）；"
                         "也可选具体编号应急。")
            if reg.ignored:
                names = "、".join((s.name or f"#{s.index}") for s in reg.ignored)
                self._append(f"已标为「无关」（自动挑相机会跳过）：{names}")
            else:
                self._append("提示：不打算用的相机（笔记本内置 / 红外）可以在"
                             "『相机设备管理』的画面里按 n 标成「无关」。")
        else:
            self._append("没有枚举到设备（非 Windows 或系统枚举不可用），"
                         "只能按编号使用。")

    def _run_cameras_identify(self) -> None:
        self._run([PY, str(ROOT / "apps" / "tools" / "cameras.py"), "identify"],
                  "相机设备管理（逐个画面确认：l=左 / r=右 / n=无关 / m=主，"
                  "结果写入 config/cameras.json）")

    def _run_cameras_check(self) -> None:
        self._run([PY, str(ROOT / "apps" / "tools" / "cameras.py"), "check"],
                  "检查相机映射（重插相机后看看有没有漂移）")

    def _run_cameras_ignored(self) -> None:
        self._run([PY, str(ROOT / "apps" / "tools" / "cameras.py"), "ignored"],
                  "「无关」相机清单（自动挑相机会跳过它们）")

    def _run_cameras_unignore_all(self) -> None:
        if not messagebox.askyesno(
                "清空「无关」标记",
                "确定要清空整份「无关」清单吗？\n\n"
                "清空后这些相机（笔记本内置 / 红外等）又会被自动挑选选中。\n"
                "如果只是想换一台，建议改用『相机设备管理』重新确认。"):
            return
        self._run([PY, str(ROOT / "apps" / "tools" / "cameras.py"),
                   "unignore", "--all"],
                  "清空「无关」清单（全部恢复可被自动挑选）")

    def _run_selftest(self) -> None:
        self._run([PY, str(ROOT / "tests" / "selftest_synthetic.py")],
                  "环境自测（无需摄像头，约 1 分钟）")

    # ---------- 子进程与日志 ----------
    def _run(self, cmd: list[str], title: str) -> None:
        if self.proc is not None and self.proc.poll() is None:
            messagebox.showwarning("正在运行", "已有程序在运行，请先等它结束。")
            return
        self._append(f"\n===== {title} =====\n> " + " ".join(cmd))
        self.status.configure(text="运行中…")
        self.proc = subprocess.Popen(
            cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1)
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()

    def _reader(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            self.log_queue.put(line.rstrip("\n"))
        proc.wait()
        self.log_queue.put(f"[进程结束] 返回码 {proc.returncode}")
        self.log_queue.put(None)

    def _poll(self) -> None:
        while True:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            if line is None:
                self.proc = None
                self.status.configure(text="就绪")
                continue
            self._append(line)
        self.after(120, self._poll)

    def _append(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _on_close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            if not messagebox.askyesno("仍在运行", "检测程序还在运行，确定退出启动器吗？"):
                return
        self.destroy()


def main() -> int:
    Launcher().mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
