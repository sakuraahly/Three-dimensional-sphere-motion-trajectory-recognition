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
# 单目相机：默认 auto（启动时自动挑最优），也可选具体编号。
# 注：曾经还有 "main"（笔记本内置）这一项，已移除 —— 内置相机被标进
# config/cameras.json 的「无关」清单，角色 main 也不再存在；留着这个选项
# 只会让用户选到一个必定报错的项。
MONO_CAMERAS = ["auto", "0", "1", "2"]
RESOLUTIONS = ["1920x1080", "1280x720", "960x540", "640x480"]

# ---- 换行宽度 ----
# 这两个数是**布局能不能站住**的关键，不是审美问题：
# 不设换行时，提示文字会按整行自然宽度把卡片撑开（实测第 1 步那张卡片
# 自然宽度 1941 px、整窗 1973 px，而屏幕只有 1600），结果新加的参数区
# 把原有控件全挤到窗口外。设了换行，卡片宽度就与**文字长度无关**。
# ⚠️ **实测结论**：当前**真正卡住宽度的是 CARD_HINT_WRAP**。
# 只把卡片说明设成 820/700/640，页面宽就跟着 873/756/704 变；
# 到 640 之后不再变 —— 约束转移到了参数说明（page 741）。
# 也就是说 PARAM_HINT_WRAP 现在是**护栏**（防止将来某条参数说明特别长
# 又把卡片撑开），而**不是**当前的约束。测试能抓住前者、抓不住后者，
# 这一点写在 selftest_launcher_params.py 里，不假装覆盖了。
CARD_HINT_WRAP = 820      # 卡片顶部那句总说明（当前是实际约束）
PARAM_HINT_WRAP = 520     # 每个参数下方那句说明（护栏）


def _cam_arg(raw: str) -> str:
    """下拉里可能是 `0 | aoni webcam A25 | 6&...`，取前面那个编号；否则原样返回角色名。"""
    m = re.match(r"\s*(\d+)", raw or "")
    return m.group(1) if m else (raw or "").strip()


class Launcher(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("球形物体三维运动轨迹识别 · 启动器")
        self.configure(bg=BG)
        self.geometry("980x780")
        # 最小宽度要**保证内容不被横向截断**：卡片内容按 wraplength 换行后
        # 实测约需 860 px，低于它就会出现横向裁切。
        self.minsize(880, 560)

        self.proc: subprocess.Popen | None = None
        self.log_queue: queue.Queue = queue.Queue()

        self._init_style()
        self._build_header()
        # 滚动页是**所有卡片与日志的父容器**（在 header 之后创建，
        # 于是它在 pack 顺序里排在 header 下面）。窗口再高也只显示一屏，
        # 其余靠滚轮 / PageUp·PageDown 浏览 —— 像浏览器一样。
        self._build_scroll_page()
        self._build_step1()
        self._build_step2()
        self._build_step3()
        self._build_log()
        self._bind_scroll()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(120, self._poll)
        self.after(200, self._refresh_cameras)     # 启动时就把当前设备填进下拉

    # ---------- 可滚动页面 ----------
    def _build_scroll_page(self) -> None:
        """把内容放进"一屏装不下也能滚"的页面里。

        ## 为什么必须这么做

        加标定参数区之后，卡片的自然宽度一度到 **1941 px**（整窗 1973 px），
        而屏幕只有 1600 —— 原有控件被挤到窗口外，用户看到的就是"被挤掉了"。

        宽度问题由 `wraplength`（见 `CARD_HINT_WRAP`）解决；
        而**高度**问题必须靠滚动：三步 + 参数区 + 日志，竖着放本来就不是一屏的量。
        Tkinter 没有现成的滚动容器，标准做法就是 `Canvas` + 内嵌 `Frame`。
        """
        wrap = ttk.Frame(self)
        wrap.pack(fill="both", expand=True)
        self._canvas = tk.Canvas(wrap, bg=BG, highlightthickness=0, bd=0)
        self._vbar = ttk.Scrollbar(wrap, orient="vertical",
                                   command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._on_scrollbar)
        self._canvas.pack(side="left", fill="both", expand=True)
        self._vbar.pack(side="right", fill="y")

        # `_page` 是真正装内容的 frame；`create_window` 把它嵌进 canvas。
        self._page = ttk.Frame(self._canvas)
        self._page_id = self._canvas.create_window((0, 0), window=self._page,
                                                   anchor="nw")
        # 内容尺寸变了 -> 更新可滚动范围
        self._page.bind("<Configure>", self._on_page_configure)
        # 画布宽度变了 -> 让内嵌 frame 跟着变宽（否则卡片不随窗口拉伸）
        self._canvas.bind("<Configure>", self._on_canvas_configure)

    def _on_page_configure(self, _evt=None) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, evt) -> None:
        self._canvas.itemconfigure(self._page_id, width=evt.width)

    def _on_scrollbar(self, first: str, last: str) -> None:
        """滚动条只在**真的需要**时出现（内容不超一屏就不显示，少一点视觉噪音）。"""
        self._vbar.set(first, last)
        if float(first) <= 0.0 and float(last) >= 1.0:
            self._vbar.pack_forget()
        else:
            self._vbar.pack(side="right", fill="y")

    def _bind_scroll(self) -> None:
        """滚轮 / PageUp·PageDown / Home·End 都能滚。

        ⚠️ `bind_all("<MouseWheel>")` 会**全局**抢滚轮事件（连输入框里滚动都算），
        所以先判断指针是不是在滚动区里，不在就不管。
        """
        def in_page(widget) -> bool:
            while widget is not None:
                if widget is self._canvas or widget is self._page:
                    return True
                widget = getattr(widget, "master", None)
            return False

        def on_wheel(evt):
            if not in_page(evt.widget):
                return
            self._canvas.yview_scroll(-1 if evt.delta > 0 else 1, "units")

        self.bind_all("<MouseWheel>", on_wheel)                 # Windows / macOS
        self.bind_all("<Button-4>", lambda e: in_page(e.widget)
                      and self._canvas.yview_scroll(-1, "units"))   # X11 上滚
        self.bind_all("<Button-5>", lambda e: in_page(e.widget)
                      and self._canvas.yview_scroll(1, "units"))    # X11 下滚
        for key, amount in (("<Next>", 1), ("<Prior>", -1)):
            self.bind(key, lambda e, a=amount: self._canvas.yview_scroll(a, "pages"))
        self.bind("<Home>", lambda e: self._canvas.yview_moveto(0.0))
        self.bind("<End>", lambda e: self._canvas.yview_moveto(1.0))

    def scroll_to_bottom(self) -> None:
        """滚到底（日志在长，用户多半想看最新那几行）。"""
        self.update_idletasks()
        self._canvas.yview_moveto(1.0)

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
        """一张卡片。**hint 必须换行**，否则它会按整行文字的自然宽度撑开整张卡片。

        ⚠️ 实测过：不设 `wraplength` 时第 1 步那张卡片的自然宽度是 **1941 px**
        （整窗 1973 px），而屏幕只有 1600 —— 于是新加的参数区把原有的控件
        全挤到窗口外面去了。**"文字不换行"就是把布局挤爆的直接原因。**
        """
        box = ttk.LabelFrame(self._page, text=title, style="Card.TLabelframe",
                             padding=(14, 8))
        box.pack(fill="x", padx=16, pady=(10, 0))
        # columnspan 给足：卡片里最多用到第 8 列（0..7），再加 1 列余量，
        # 免得标题提示只跨 7 列、被挤成两行。
        ttk.Label(box, text=hint, style="Hint.TLabel",
                  wraplength=CARD_HINT_WRAP).grid(
            row=0, column=0, columnspan=9, sticky="w", pady=(0, 8))
        return box

    # ---------- 参数行的小工具 ----------
    def _param_entry(self, parent, row: int, label: str, value: str,
                     hint: str = "", **kw) -> ttk.Entry:
        """一行参数：第一行 `标签 [输入框]`，第二行说明（换行）。

        **为什么不把说明放在右边**：那一列会按说明文字的自然宽度撑开整张卡片
        （实测某条说明单独就要 567 px）。放在下面并限宽，宽度就与说明长度无关。
        """
        ttk.Label(parent, text=label).grid(row=row * 2, column=0, sticky="e",
                                           padx=(0, 6), pady=(4, 0))
        ent = ttk.Entry(parent, width=kw.pop("width", 8), **kw)
        ent.insert(0, value)
        ent.grid(row=row * 2, column=1, sticky="w", pady=(4, 0))
        if hint:
            ttk.Label(parent, text=hint, style="Hint.TLabel",
                      wraplength=PARAM_HINT_WRAP).grid(
                row=row * 2 + 1, column=1, columnspan=4, sticky="w", pady=(0, 2))
        return ent

    def _param_path(self, parent, row: int, label: str, hint: str,
                    is_dir: bool) -> ttk.Entry:
        """路径参数：`标签 [输入框] [选择…]` + 第二行说明。留空 = 不要。"""
        ttk.Label(parent, text=label).grid(row=row * 2, column=0, sticky="e",
                                           padx=(0, 6), pady=(4, 0))
        ent = ttk.Entry(parent, width=30)
        ent.grid(row=row * 2, column=1, columnspan=2, sticky="w", pady=(4, 0))

        def pick() -> None:
            got = (filedialog.askdirectory(title=label) if is_dir
                   else filedialog.asksaveasfilename(title=label, defaultextension=".png"))
            if got:
                ent.delete(0, "end")
                ent.insert(0, got)

        ttk.Button(parent, text="选择…", command=pick, width=6).grid(
            row=row * 2, column=3, sticky="w", padx=(6, 0), pady=(4, 0))
        if hint:
            ttk.Label(parent, text=hint, style="Hint.TLabel",
                      wraplength=PARAM_HINT_WRAP).grid(
                row=row * 2 + 1, column=1, columnspan=4, sticky="w", pady=(0, 2))
        return ent

    def _calib_params(self) -> ttk.Frame:
        """标定参数区：**基本设置 / 高级设置** 两个页签。

        为什么分区（全局 §四「渐进式展示」）：标定有十几个可调项，
        全铺在第一屏会让人不知道哪些真的必须改。分成两页之后：
          · 基本 = **改错就白做**的那几项（板参数、收多少对、分辨率）；
          · 高级 = **用精度换成功率**的开关（停稳/清晰度/放宽/基线核对）。
        """
        nb = ttk.Notebook(self.param_holder)
        nb.pack(fill="x", pady=(2, 0))
        basic = ttk.Frame(nb, padding=(10, 8))
        adv = ttk.Frame(nb, padding=(10, 8))
        nb.add(basic, text="  基本设置  ")
        nb.add(adv, text="  高级设置  ")

        # ---- 基本 ----
        ttk.Label(basic, text="① 收多少对（越多越准，25 对约需举板几分钟）",
                  style="Hint.TLabel").grid(row=0, column=0, columnspan=7,
                                            sticky="w", pady=(0, 4))
        self.calib_target = self._param_entry(
            basic, 1, "目标对数", "25", "收够就自动开始标定（正式档 25）")
        self.calib_minpairs = self._param_entry(
            basic, 2, "最少对数", "10", "少于这么多不做标定；剔除离群后不足会直接拒绝")

        ttk.Label(basic, text="② 标定板（必须和手上那张纸完全一致）",
                  style="Hint.TLabel").grid(row=3, column=0, columnspan=7,
                                            sticky="w", pady=(10, 4))
        self.calib_cols = self._param_entry(
            basic, 4, "内角点·列", "8", "**内角点**数，不是方格数：8×5 内角点 = 9×6 方格")
        self.calib_rows = self._param_entry(basic, 5, "内角点·行", "5")
        self.calib_square = self._param_entry(
            basic, 6, "方格边长 mm", "25", "★ 必须用尺子量实际边长，别信打印设置")

        ttk.Label(basic, text="③ 两路采集（左右必须一致）", style="Hint.TLabel").grid(
            row=7, column=0, columnspan=7, sticky="w", pady=(10, 4))
        ttk.Label(basic, text="分辨率").grid(row=8, column=0, sticky="e", padx=(0, 6))
        self.calib_res = ttk.Combobox(basic, values=RESOLUTIONS, width=10,
                                      state="readonly")
        self.calib_res.set("1920x1080")
        self.calib_res.grid(row=8, column=1, sticky="w", pady=3)
        ttk.Label(basic, text="帧率").grid(row=8, column=2, sticky="e",
                                          padx=(16, 6))
        self.calib_fps = ttk.Entry(basic, width=6)
        self.calib_fps.insert(0, "60")
        self.calib_fps.grid(row=8, column=3, sticky="w", pady=3)
        ttk.Label(basic, text="（两台相机必须同分辨率、同后端）",
                  style="Hint.TLabel").grid(row=8, column=4, columnspan=3,
                                            sticky="w", padx=(10, 0))

        # ---- 高级 ----
        ttk.Label(adv, text="这几项**都是拿精度换成功率**，不确定就别动",
                  style="Hint.TLabel").grid(row=0, column=0, columnspan=7,
                                            sticky="w", pady=(0, 4))
        self.calib_stationary = self._param_entry(
            adv, 1, "停稳上限 px", "", "留空=默认 3。这条**同时替代左右同步**，"
                                       "调大到 10 以上会让两路拍到不同位姿、R/T 系统性偏")
        self.calib_sharp = self._param_entry(
            adv, 2, "清晰度下限", "", "留空=默认 55（Laplacian 方差）。调低会收进更糊的帧")
        self.calib_relaxed = tk.BooleanVar(value=False)
        ttk.Checkbutton(adv, text="放宽模式（手持拍摄：一次放松 6 条门限）",
                        variable=self.calib_relaxed).grid(
            row=3, column=0, columnspan=4, sticky="w", pady=(6, 2))
        ttk.Label(adv, text="用完必须复检极线误差；不达标请处理光照/支撑后关掉它重标",
                  style="Hint.TLabel").grid(row=4, column=0, columnspan=7,
                                            sticky="w", pady=(0, 6))

        ttk.Label(adv, text="④ 核对与留证（**强烈建议都填**）",
                  style="Hint.TLabel").grid(row=5, column=0, columnspan=7,
                                            sticky="w", pady=(6, 4))
        self.calib_baseline = self._param_entry(
            adv, 6, "预期基线 mm", "", "★ 两镜头中心距。**填了才会核对基线** ——"
                                       "不填这道护栏就是关着的")
        self.calib_savepairs = self._param_path(
            adv, 7, "存图像对", "存下来才能离线复查/重标（出问题时唯一能回答「哪一对错了」）",
            is_dir=True)
        self.calib_report = self._param_path(
            adv, 8, "验证图", "标定后出一张可视化验证图", is_dir=False)
        return nb

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
        ttk.Label(box, text="（分辨率 / 帧率 / 板参数在下面的『基本设置』里）",
                  style="Hint.TLabel").grid(row=4, column=4, columnspan=4,
                                            sticky="w", pady=(8, 0))

        # 参数区分两个页签。放 `param_holder` 里是为了让「运行标定」按钮
        # 排在参数**之后** —— 按阅读顺序：选相机 → 调参数 → 点运行。
        self.param_holder = ttk.Frame(box)
        self.param_holder.grid(row=5, column=0, columnspan=8, sticky="we", pady=(8, 4))
        self._calib_params()

        ttk.Button(box, text="运行标定 / 复检", command=self._run_calibration).grid(
            row=6, column=0, columnspan=2, sticky="w", pady=(10, 2))
        ttk.Button(box, text="标定前置检查（两路是否都看到棋盘）",
                   command=self._run_precheck).grid(
            row=6, column=2, columnspan=3, sticky="w", pady=(10, 2))
        ttk.Button(box, text="刷新设备列表", command=self._refresh_cameras).grid(
            row=6, column=5, columnspan=3, sticky="w", pady=(10, 2))
        ttk.Button(box, text="相机探测（核对两路参数是否一致）",
                   command=self._run_probe).grid(
            row=7, column=0, columnspan=3, sticky="w", pady=(2, 2))
        ttk.Button(box, text="相机设备管理（识别 / 固定 / 标无关）",
                   command=self._run_cameras_identify).grid(
            row=7, column=3, columnspan=2, sticky="w", pady=(2, 2))
        ttk.Button(box, text="检查相机映射", command=self._run_cameras_check).grid(
            row=7, column=5, columnspan=2, sticky="w", pady=(2, 2))
        ttk.Button(box, text="「无关」相机清单（自动挑选会跳过）",
                   command=self._run_cameras_ignored).grid(
            row=8, column=0, columnspan=3, sticky="w", pady=(2, 2))
        ttk.Button(box, text="清空「无关」标记", command=self._run_cameras_unignore_all).grid(
            row=8, column=3, columnspan=2, sticky="w", pady=(2, 2))

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
        ttk.Button(box, text="快测（十几秒）", command=self._run_tests_fast).grid(
            row=1, column=2, padx=(0, 6), pady=4)
        ttk.Button(box, text="全套回归（7 分钟）", command=self._run_tests_full).grid(
            row=1, column=3, padx=(0, 6), pady=4)
        ttk.Button(box, text="规范自查", command=self._run_standards).grid(
            row=1, column=4, pady=4)
        self.status = ttk.Label(box, text="就绪", style="Hint.TLabel")
        self.status.grid(row=2, column=0, columnspan=5, sticky="w", padx=2)

    # ---------- 运行日志 ----------
    def _build_log(self) -> None:
        """日志卡片。

        ⚠️ 两个"不能这么写"（滚动版布局踩到过）：

        1. **父子关系**：原来 `ttk.LabelFrame(self, ...)` —— 直接用窗口当父容器，
           于是它**不在滚动页里**，页面滚动时日志纹丝不动。必须挂到 `self._page`。
        2. **`expand=True`**：它在滚动页里意味着"吃掉剩下的空间"，
           而滚动页的高度由内容决定 —— 结果是日志把页面撑得更高、
           却又不会自己长，白占一屏。改成**固定高度 + 自己内部滚动**。
        """
        box = ttk.LabelFrame(self._page, text="运行日志", style="Card.TLabelframe",
                             padding=(10, 6))
        box.pack(fill="x", padx=16, pady=(10, 14))
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

    # ---------- 读取 / 校验参数面板 ----------
    def _read_int(self, entry: ttk.Entry, label: str, lo: int, hi: int) -> int | None:
        """读一个整数并校验范围。非法时弹窗并把焦点放回该输入框。"""
        raw = entry.get().strip()
        try:
            val = int(raw)
        except ValueError:
            messagebox.showwarning("参数不对", f"「{label}」要填整数，现在是「{raw}」。")
            entry.focus_set()
            return None
        if not (lo <= val <= hi):
            messagebox.showwarning("参数不对", f"「{label}」要在 {lo}~{hi} 之间，现在是 {val}。")
            entry.focus_set()
            return None
        return val

    def _read_opt_float(self, entry: ttk.Entry, label: str,
                        lo: float, hi: float) -> tuple[float | None, bool]:
        """读一个**可留空**的浮点。返回 (值, 是否合法)；空 = (None, True)。"""
        raw = entry.get().strip()
        if not raw:
            return None, True
        try:
            val = float(raw)
        except ValueError:
            messagebox.showwarning("参数不对", f"「{label}」要么留空、要么填数字，现在是「{raw}」。")
            entry.focus_set()
            return None, False
        if not (lo <= val <= hi):
            messagebox.showwarning("参数不对", f"「{label}」要在 {lo}~{hi} 之间，现在是 {val}。")
            entry.focus_set()
            return None, False
        return val, True

    def _calibrate_args(self) -> list[str] | None:
        """把参数面板拼成 `calibrate_stereo.py` 的命令行。非法则返回 None。

        ⚠️ **这里曾经是硬编码的**（`--cols 8 --rows 5 --square 25`，且完全没有
        `--target`），于是从界面走**永远停在脚本默认的试标定档**，
        而用户以为在做正式标定。现在参数由面板给，面板的默认值就是正式档。
        """
        target = self._read_int(self.calib_target, "目标对数", 4, 200)
        if target is None:
            return None
        minp = self._read_int(self.calib_minpairs, "最少对数", 3, 200)
        if minp is None:
            return None
        cols = self._read_int(self.calib_cols, "内角点·列", 3, 30)
        if cols is None:
            return None
        rows = self._read_int(self.calib_rows, "内角点·行", 3, 30)
        if rows is None:
            return None
        square = self._read_int(self.calib_square, "方格边长 mm", 5, 500)
        if square is None:
            return None
        if minp > target:
            messagebox.showwarning(
                "参数矛盾",
                f"「最少对数」({minp}) 不能大于「目标对数」({target}) ——\n"
                f"收够 {target} 对就停了，永远达不到 {minp}。")
            return None

        stationary, ok1 = self._read_opt_float(
            self.calib_stationary, "停稳上限 px", 0.5, 12.0)
        if not ok1:
            return None
        sharp, ok2 = self._read_opt_float(self.calib_sharp, "清晰度下限", 5.0, 500.0)
        if not ok2:
            return None
        baseline, ok3 = self._read_opt_float(
            self.calib_baseline, "预期基线 mm", 5.0, 2000.0)
        if not ok3:
            return None

        width, height = self.calib_res.get().split("x")
        fps = self.calib_fps.get().strip() or "60"
        args = ["--left", _cam_arg(self.calib_left.get()),
                "--right", _cam_arg(self.calib_right.get()),
                "--cols", str(cols), "--rows", str(rows), "--square", str(square),
                "--target", str(target), "--min-pairs", str(minp),
                "--width", width, "--height", height, "--fps", fps,
                # 序号在 DSHOW/MSMF 下含义不同；MSMF 与设备枚举顺序一致（见 pair_source）
                "--backend", "msmf"]
        if stationary is not None:
            args += ["--stationary-px", str(stationary)]
        if sharp is not None:
            args += ["--min-sharpness", str(sharp)]
        if baseline is not None:
            args += ["--baseline-mm", str(baseline)]
        if self.calib_relaxed.get():
            args += ["--relaxed"]
        save = self.calib_savepairs.get().strip()
        if save:
            args += ["--save-pairs", save]
        report = self.calib_report.get().strip()
        if report:
            args += ["--report-png", report]
        return args

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
            extra = self._calibrate_args()
            if extra is None:
                return                      # 校验失败，弹窗已经说清哪里不对
            cmd = [PY, str(ROOT / "apps" / "calibrate_stereo.py")] + extra
            target = self.calib_target.get().strip()
            base = self.calib_baseline.get().strip()
            note = f"收满 {target} 对后自动标定"
            if not base:
                note += "；⚠ 未填预期基线，基线核对将是关着的"
            self._run(cmd, f"双目标定（左{left} / 右{right}，{note}）")
        else:
            calib = ROOT / "config" / "stereo_calibration"
            if not calib.with_suffix(".npz").exists():
                messagebox.showwarning("找不到标定文件",
                                       "还没有标定结果，请先执行一次双目标定。")
                return
            cmd = [PY, str(ROOT / "apps" / "verify_stereo.py"),
                   "--calib", str(calib), "--left", left, "--right", right,
                   "--width", width, "--height", height, "--fps", fps,
                   # 序号在 DSHOW/MSMF 下含义不同；MSMF 与设备枚举顺序一致（见 pair_source）
                   "--backend", "msmf"]
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
        mono_values = ["auto"] + items
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

    def _run_tests_fast(self) -> None:
        """快测：走 apps/run_tests.py fast（与 `启动程序.bat test` 同一个运行器）。

        ⚠️ 这里**不自己拼测试清单**。以前这个按钮硬编码只跑
        `selftest_synthetic.py` 一个，而 `AGENTS.md` §7 要求跑 15 个 ——
        同一个缺陷在 `.bat` 和 GUI 两处各有一份，改一处不算修好。
        现在两个入口都调同一个运行器，分层依据是实测耗时。
        """
        self._run([PY, str(ROOT / "apps" / "run_tests.py"), "fast"],
                  "回归测试 · fast（跳过已知慢的，十几秒）")

    def _run_tests_full(self) -> None:
        """全套回归：15 个都跑，约 7 分钟。改代码后必跑这个。"""
        self._run([PY, str(ROOT / "apps" / "run_tests.py"), "full"],
                  "回归测试 · full（15 个全跑，约 7 分钟）")

    def _run_standards(self) -> None:
        """规范落差自查：引用 / 来源 / 计数 是否与磁盘一致。"""
        self._run([PY, str(ROOT / "apps" / "tools" / "check_standards.py")],
                  "规范落差自查（引用 / 来源 / 计数）")

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
        """往日志里追加一行，并**在合适的时候**自动滚到底。

        ## 为什么要有"是不是贴底"这个判断

        原来无条件 see("end")。加了页面滚动之后这变成一个**会打断用户**的行为：
        用户往上翻日志看早先的输出，新来一行就把他拽回底部 —— 而他现在
        还要同时跟**页面的**滚动较劲（两个滚动条）。

        所以：**只在用户本来就贴着底时才自动滚**。这条约定在终端、日志查看器里
        是通用的；用户一旦往上翻，就说明他在读历史，不该被打断。
        """
        at_bottom = self.log.yview()[1] >= 0.999
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        if at_bottom:
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
