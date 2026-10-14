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

import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
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

# ---- 换行宽度：**跟着视口动态算，不写死** ----
#
# 为什么必须动态：
#
# 1. 不设换行时，提示文字会按整行自然宽度把卡片撑开 —— 实测第 1 步那张卡片
#    自然宽度 1941 px、整窗 1973 px，而屏幕只有 1600，原有控件全被挤到窗口外。
# 2. 但**写死也不行**：写死 820 是按"窗口 900"估的；窗口改成 980 之后，
#    卡片自宽 873 而视口 966 —— 文字在卡片那一列里被切掉，右边还空一大片。
#    界面上"…在下面的『基本···"被截断就是这么来的。
#
# 所以：**每次画布尺寸变化就按实际宽度重设**（见 _reflow）。
# 这两个常量只当"第一次布局前的初值"。
CARD_HINT_WRAP = 820      # 初值；实际由 _reflow 按视口宽度刷新
PARAM_HINT_WRAP = 720     # 初值；同上
# 卡片左右各 16px 外边距 + 卡片内 14px 内边距 + 滚动条 ≈ 60px
CHROME_PX = 64

# 界面刷新周期（毫秒）。轮询日志 + 刷新任务耗时列，都用这一个值 ——
# 原来轮询是 120ms、任务列又自己排了 500ms，两套节奏叠在一起。
POLL_MS = 200


def _cam_arg(raw: str) -> str:
    """下拉里可能是 `0 | aoni webcam A25 | 6&...`，取前面那个编号；否则原样返回角色名。"""
    m = re.match(r"\s*(\d+)", raw or "")
    return m.group(1) if m else (raw or "").strip()


# --------------------------------------------------------------------------- #
# 任务：启动器跑起来的每一个外部程序
# --------------------------------------------------------------------------- #
@dataclass
class Task:
    """一个正在跑（或跑完）的外部程序。

    ## 为什么要单独一个对象，而不是像原来那样只留一个 `self.proc`

    原来只有 `self.proc` 一个句柄，于是：

    - **一次只能跑一个**（第二个会被"已有程序在运行"挡掉）；
    - **没有任何中断手段** —— 程序卡住（相机被占、等键盘输入）时，
      用户只能关掉整个启动器窗口，而那**不会杀掉子进程**：
      相机仍被占着，下一次运行打不开相机，而用户不知道为什么。

    记下 pid / 启动时刻 / 退出码，才能回答"它在跑吗、跑了多久、怎么死的"。

    **`log_path`：每个任务都把完整输出落盘一份。** 日志区是给眼睛看的，
    会被下一条任务冲掉；而"上一次到底报了什么"恰恰是出问题时唯一能查的东西
    （这次定位标定问题就是靠 `outputs/logs/calibrate_*.log`）。
    """

    tid: int
    title: str
    cmd: list[str]
    proc: subprocess.Popen
    log_path: Path | None = None
    started: float = field(default_factory=time.monotonic)
    ended: float | None = None
    exit_code: int | None = None
    stop_reason: str = ""
    lines: int = 0

    @property
    def running(self) -> bool:
        return self.proc.poll() is None

    def elapsed(self) -> float:
        end = self.ended if self.ended is not None else time.monotonic()
        return max(0.0, end - self.started)

    def cmdline(self) -> str:
        return " ".join(self.cmd)


def _child_pids(pid: int) -> list[int]:
    """某个进程的子进程 pid（拿不到就返回空）。

    为什么需要：`Popen.terminate()` **只杀直接子进程**。
    我们的子进程是 python，它可能自己还有子进程（例如 OpenCV 的辅助线程、
    或用户程序再起的进程）。**父进程死了子进程还活着**，
    于是相机仍被占着 —— 用户看到"停了但相机打不开"。
    所以停任务要杀**整棵树**。
    """
    try:
        import psutil                                       # 可选依赖
    except ImportError:
        return []
    try:
        return [c.pid for c in psutil.Process(pid).children(recursive=True)]
    except Exception:                                       # noqa: BLE001
        return []



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
        # 要跟着视口宽度重新换行的标签：(控件, "卡片说明"|"参数说明")
        self._wrapped: list[tuple[ttk.Label, str]] = []
        # 任务表：所有跑过的程序（在跑的 + 历史）。`_tid` 只增不减。
        self.tasks: list[Task] = []
        self._tid = 0
        self._sel_tid: int | None = None

        self._init_style()
        self._build_header()
        # 滚动页是**所有卡片与日志的父容器**（在 header 之后创建，
        # 于是它在 pack 顺序里排在 header 下面）。窗口再高也只显示一屏，
        # 其余靠滚轮 / PageUp·PageDown 浏览 —— 像浏览器一样。
        self._build_scroll_page()
        self._build_step1()
        self._build_step2()
        self._build_step3()
        self._build_tasks()      # 任务管理器：看得见、停得掉
        self._build_log()
        self._bind_scroll()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(POLL_MS, self._poll)
        self.after(200, self._refresh_cameras)     # 启动时就把当前设备填进下拉
        # 首帧量出真实宽度后立刻按它换行一次（<Configure> 也会触发，
        # 这里只是保证打开就已经是对的，不会先闪一下错的换行）。
        self.after(0, self._reflow)

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
        self._reflow(evt.width)

    def _reflow(self, viewport: int | None = None) -> None:
        """按**当前视口宽度**重设所有提示的换行宽度。

        ## 为什么不能写死

        写死 820 是按"窗口 900"估的。窗口改成 980 之后，卡片自宽 873 而视口 966：
        文字在卡片那一列里被切掉（界面里"…在下面的『基本···"看着像被截断），
        而右边又空出一大片。**写死的换行宽度一定会和实际窗口对不上。**

        重设 wraplength 会触发重新布局 -> 又发一次 <Configure>；
        但下一次算出的值相同、不再改动，所以会**收敛**，不会无限递归。
        """
        if viewport is None:
            viewport = self._canvas.winfo_width()
        if viewport <= 1:                      # 首帧还没量出来
            return
        avail = max(240, viewport - CHROME_PX)
        card = max(200, avail)
        # 参数说明在输入框右侧起，再让出标签列与内边距
        param = max(200, avail - 150)
        for lab, kind in self._wrapped:
            want = card if kind == "card" else param
            if int(lab.cget("wraplength")) != want:
                lab.configure(wraplength=want)

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
        lab = ttk.Label(box, text=hint, style="Hint.TLabel",
                        wraplength=CARD_HINT_WRAP)
        lab.grid(row=0, column=0, columnspan=9, sticky="w", pady=(0, 8))
        self._wrapped.append((lab, "card"))
        return box

    # ---------- 参数行的小工具 ----------
    #
    # ## 行号由**辅助函数自己分配**，调用方不再手算（2026-10-14 修）
    #
    # 原来每个 `_param_entry(parent, row, ...)` 要调用方自己给行号，而内部占
    # `row*2` 与 `row*2+1` **两行**。结果是：参数行传 1（占 2、3 行），
    # 下一个传 2（占 4、5 行），而手工 `.grid(row=3)` 的复选框正好落在
    # **被占的行上** —— 文字与输入框**互相盖住**，正是界面上看到的
    # "高级设置里都是被遮挡的文字"。
    #
    # 手算行号本身就是错的做法：加了/去掉一行就要重排所有编号，
    # 迟早再撞一次。改成每个 parent 维护一个游标，**谁先放谁占下一行**。
    def _next_row(self, parent) -> int:
        """给 parent 分配下一个空行（每次调用 +1）。"""
        row = getattr(parent, "_next_row", 0)
        setattr(parent, "_next_row", row + 1)
        return row

    def _hint(self, parent, text: str, *, indent: int = 0) -> None:
        """一行说明文字（跨列、限宽换行）。"""
        lab = ttk.Label(parent, text=text, style="Hint.TLabel",
                        wraplength=PARAM_HINT_WRAP)
        lab.grid(row=self._next_row(parent), column=indent, columnspan=4,
                 sticky="w", pady=(0, 4))
        self._wrapped.append((lab, "param"))

    def _param_entry(self, parent, label: str, value: str,
                     hint: str = "", **kw) -> ttk.Entry:
        """一行参数：`标签 [输入框]` 一行，说明一行。

        **说明放下面而不是右边**：那一列会按说明文字的自然宽度撑开整张卡片
        （实测某条说明单独就要 567 px）。
        """
        row = self._next_row(parent)
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(4, 0))
        ent = ttk.Entry(parent, width=kw.pop("width", 10), **kw)
        ent.insert(0, value)
        ent.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=(4, 0))
        if hint:
            self._hint(parent, hint)
        return ent

    def _param_check(self, parent, label: str, var, hint: str = "") -> None:
        """一行勾选项 + 说明。"""
        row = self._next_row(parent)
        ttk.Checkbutton(parent, text=label, variable=var).grid(
            row=row, column=0, columnspan=4, sticky="w", pady=(6, 0))
        if hint:
            self._hint(parent, hint)

    def _param_path(self, parent, label: str, hint: str, is_dir: bool) -> ttk.Entry:
        """路径参数：`[输入框] [选择…]` 一行 + 说明一行。留空 = 不要。"""
        row = self._next_row(parent)
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(4, 0))
        ent = ttk.Entry(parent, width=34)
        ent.grid(row=row, column=1, columnspan=2, sticky="w", padx=(8, 0), pady=(4, 0))

        def pick() -> None:
            got = (filedialog.askdirectory(title=label) if is_dir
                   else filedialog.asksaveasfilename(title=label, defaultextension=".png"))
            if got:
                ent.delete(0, "end")
                ent.insert(0, got)

        ttk.Button(parent, text="选择…", command=pick, width=6).grid(
            row=row, column=3, sticky="w", padx=(6, 0), pady=(4, 0))
        if hint:
            self._hint(parent, hint)
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
        #
        # 分组标题也走 `_hint()`：**行号统一由游标分配**，一处手算都没有。
        self._hint(basic, "① 收多少对（越多越准，25 对约需举板几分钟）")
        self.calib_target = self._param_entry(
            basic, "目标对数", "25", "收够就自动开始标定（默认 25 = 正式档）")
        self.calib_minpairs = self._param_entry(
            basic, "最少对数", "10", "少于这么多不做标定；剔除离群后不足会直接拒绝")

        self._hint(basic, "② 标定板（必须和手上那张纸完全一致）")
        self.calib_cols = self._param_entry(
            basic, "内角点·列", "8", "**内角点**数，不是方格数：8×5 内角点 = 9×6 方格")
        self.calib_rows = self._param_entry(basic, "内角点·行", "5")
        self.calib_square = self._param_entry(
            basic, "方格边长 mm", "25", "★ 必须用尺子量实际边长，别信打印设置")

        self._hint(basic, "③ 两路采集（左右必须一致）")
        # 分辨率与帧率放在**同一行**：`标签 [框] 标签 [框]`。
        # ⚠️ 第一版把它们放在 column 1 与 column 3、间距给 16 px，结果
        # "帧率"二字正好压在 60 那个框上（渲染布局图时看得出来）。
        # 现在帧率跟在分辨率**紧后面**，列号连续，不再靠间距硬撑。
        row = self._next_row(basic)
        ttk.Label(basic, text="分辨率").grid(row=row, column=0, sticky="w",
                                            pady=(4, 0))
        self.calib_res = ttk.Combobox(basic, values=RESOLUTIONS, width=10,
                                      state="readonly")
        self.calib_res.set("1920x1080")
        self.calib_res.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=(4, 0))
        ttk.Label(basic, text="帧率").grid(row=row, column=2, sticky="e",
                                          padx=(24, 6), pady=(4, 0))
        self.calib_fps = ttk.Entry(basic, width=6)
        self.calib_fps.insert(0, "60")
        self.calib_fps.grid(row=row, column=3, sticky="w", pady=(4, 0))
        self._hint(basic, "两台相机必须同分辨率、同后端，否则一对有效图像都收不到")

        # ---- 高级 ----
        self._hint(adv, "这几项**都是拿精度换成功率**，不确定就别动")
        # ⚠️ **默认值直接填进输入框**（用户要求）。原来这些框是**空的**，
        # 只在说明里写"留空=默认 3" —— 那等于让用户自己去记默认值是多少，
        # 而且"空"与"默认"在界面上分不出来。现在填上真实默认值，
        # 说明改成"清空 = 不启用这一项"（因为这些参数是**不传就不启用**的）。
        self.calib_stationary = self._param_entry(
            adv, "停稳上限 px", "3", "清空 = 用程序内置默认；这条**同时替代左右同步**，"
                                     "调大到 10 以上会让两路拍到不同位姿、R/T 系统性偏")
        self.calib_sharp = self._param_entry(
            adv, "清晰度下限", "55", "清空 = 用程序内置默认（Laplacian 方差）。"
                                     "调低会收进更糊的帧")
        self.calib_relaxed = tk.BooleanVar(value=False)
        self._param_check(adv, "放宽模式（手持拍摄：一次放松 6 条门限）",
                          self.calib_relaxed,
                          "用完必须复检极线误差；不达标请处理光照/支撑后关掉它重标")

        self._hint(adv, "④ 核对与留证（**强烈建议都填**）")
        # 基线留空是**有意的**：程序不知道你的装置中心距，编不出来。
        # 说明里必须讲清"不填 = 这道护栏关着"，而不是让人以为没差。
        self.calib_baseline = self._param_entry(
            adv, "预期基线 mm", "", "★ 两镜头中心距。**只有填了才会核对基线** ——"
                                    "不填这道护栏就是关着的（程序无法替你猜这个值）")
        self.calib_savepairs = self._param_path(
            adv, "存图像对", "存下来才能离线复查/重标（出问题时唯一能回答「哪一对错了」）",
            is_dir=True)
        self.calib_report = self._param_path(
            adv, "验证图", "标定后出一张可视化验证图", is_dir=False)
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
        """标定/复检前的相机检查。**弹窗报错就返回 False（拦住这次操作）**。

        ## 改了什么、为什么（2026-10-14）

        原来这里对解析失败**一律静默放行**（"不挡用户的路"），而且两个角色
        **各解析各的**。实测后果：两台同型号相机同名，其中一台换过 USB 口 →
        角色绑定的端口对不上 → 按名字兜底 → 两个角色**可能落到同一台相机**；
        而且那台可能只有 1280x720 而另一台是 1920x1080 ——
        **两路分辨率不一致，双目标定直接失去意义，而界面上看不出异常**。

        现在：
        1. 用 `resolve_roles()` **一起解析**，保证两个角色不撞车；
        2. 解析失败/歧义 -> **弹窗说清并返回 False**（拦住，而不是蒙着继续）。
        """
        try:
            from sphere_tracker import camera_registry as cr
            devices = cr.list_devices()
        except Exception as exc:                             # noqa: BLE001
            self._append(f"[提示] 跳过内置/红外检查：{exc}")
            return True
        by_order = {d.order: d for d in devices}
        roles = [r for r in (_cam_arg(left), _cam_arg(right)) if r in ("left", "right")]

        try:
            resolved = cr.resolve_roles(roles, devices=devices, interactive=False)
        except cr.AmbiguousMatch as exc:
            messagebox.showerror("相机指代不明，先重新固定", str(exc))
            return False
        except RuntimeError as exc:
            messagebox.showerror(
                "相机对不上，先处理再标定",
                f"{exc}\n\n"
                "标定必须用**两台确定的外接相机**：用错相机或多台相机分不清时，"
                "标定结果会系统性错误，而结果表面上看是正常的。")
            return False

        for role in ("left", "right"):
            r = resolved.get(role)
            if r is None:
                continue
            for w in (r.warnings or ()):
                self._append(f"[相机] {w}")
            entry = by_order.get(r.index)
            name = entry.name if entry else ""
            if cr.is_avoided(name):
                return messagebox.askyesno(
                    "确认用内置 / 红外相机？",
                    f"{role}相机（编号 {r.index}，{name or '未知名'}）看起来是笔记本内置或红外相机，"
                    "不适合做双目：它没法固定在同一根横梁上，装置一动标定结果就作废。\n\n"
                    "双目请用外接相机。确定仍要继续吗？")

        # 两路分辨率必须一致 —— 不一致的双目标定没有意义，而且很难看出来
        sizes = {}
        for role in ("left", "right"):
            r = resolved.get(role)
            if r is None:
                continue
            e = by_order.get(r.index)
            sizes[role] = (e.name if e else "", r.index)
        if len({v[1] for v in sizes.values()}) < len(sizes):
            messagebox.showerror(
                "左右指向了同一台相机",
                f"左 -> #{sizes.get('left', ('?', '?'))[1]}，右 -> #{sizes.get('right', ('?', '?'))[1]}\n\n"
                "标定需要**两台不同的**相机。请跑 `启动程序.bat setup` 重新确认。")
            return False
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
        """读一个**可留空**的浮点。返回 (值, 是否合法)；空 = (None, True) = 不传。

        ⚠️ **2026-10-14 起这些框默认不再是空的**（用户要求填上默认值），
        所以"留空"从"常态"变成了"用户主动清掉"。语义仍然是**空 = 不传该参数**
        （`calibrate_stereo.py` 会用它自己的默认值）。填上的默认值与脚本默认值**相同**，
        因此"不改动就走"的行为与改动前完全一致 —— `selftest_launcher_params.py`
        会断言这一点（面板默认值 == 脚本默认值）。
        """
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
        """跑一个外部程序。

        ## 与原来的区别

        原来只留一个 `self.proc`，第二个任务会被"已有程序在运行"挡掉，
        而且**没有任何中断手段**。现在：

        - **允许多个任务同时跑**（用户可能就要同时开两个窗口对比）；
          相机冲突由用户自己负责 —— 拦着不许开反而更妨碍人。
        - 每个任务都能在「任务」卡片里**单独中断**、看耗时、看返回码。
        - 关窗口时会**先把任务收干净**（原来不会，相机因此可能一直被占着）。
        """
        task = self.start_task(cmd, title)
        self.status.configure(text=f"运行中（{len(self.running_tasks())} 个任务）")
        self._refresh_tasks()
        return task

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

    # ------------------------------------------------------------------ #
    # 任务：启动 / 停止 / 查询
    # ------------------------------------------------------------------ #
    def _spawn(self, cmd: list[str]) -> subprocess.Popen:
        """起一个**可以被整树杀掉**的子进程。

        `CREATE_NEW_PROCESS_GROUP` 是"能优雅中断"的前提：只有新进程组才能收到
        `CTRL_BREAK_EVENT`。没有它，`send_signal` 只能换成硬杀 ——
        而硬杀对"正在写标定结果"这种程序是有害的。
        """
        kw: dict = dict(cwd=str(ROOT), stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True,
                        encoding="utf-8", errors="replace", bufsize=1)
        if os.name == "nt":
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kw["start_new_session"] = True
        return subprocess.Popen(cmd, **kw)

    def start_task(self, cmd: list[str], title: str) -> Task:
        """起一个新任务并登记。**允许多个同时跑**（相机冲突由用户自己负责）。"""
        self._tid += 1
        proc = self._spawn(cmd)
        log_path = self._task_log_path(self._tid, title)
        task = Task(tid=self._tid, title=title, cmd=list(cmd), proc=proc,
                    log_path=log_path)
        self.tasks.append(task)
        self._sel_tid = task.tid
        self._append(f"\n===== [{task.tid}] {title} =====\n> {task.cmdline()}")
        self._append(f"[任务 {task.tid}] 已启动，PID {proc.pid}（可在「任务」里中断）")
        if log_path:
            self._append(f"[任务 {task.tid}] 完整日志：{log_path}")
            try:
                log_path.write_text(
                    f"任务 {task.tid}：{title}\n"
                    f"启动：{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"PID ：{proc.pid}\n"
                    f"命令：{task.cmdline()}\n"
                    + "=" * 70 + "\n", encoding="utf-8")
            except OSError as exc:
                self._append(f"[任务 {task.tid}] ⚠ 日志文件写不了：{exc}")
                task.log_path = None
        threading.Thread(target=self._reader, args=(task,), daemon=True).start()
        self.after(200, self._refresh_tasks)
        return task

    @staticmethod
    def _task_log_path(tid: int, title: str) -> Path | None:
        """给每个任务一个日志文件：`outputs/logs/task<序号>_<时间>_<标题>.log`。

        **为什么要按任务分开**：同一个 `outputs/logs/` 里已经有各程序自己的日志
        （`calibrate_*.log` 等），但启动器这一层的输出（谁起的、什么时候、
        被谁停的、退出码多少）只有这里才有。分开写才追得下去。
        """
        try:
            d = ROOT / "outputs" / "logs"
            d.mkdir(parents=True, exist_ok=True)
            safe = re.sub(r'[\\/:*?"<>|\s]+', "_", title).strip("_")[:40] or "task"
            stamp = time.strftime("%Y%m%d_%H%M%S")
            return d / f"task{tid:02d}_{stamp}_{safe}.log"
        except OSError:
            return None

    def _task_log(self, task: Task, text: str) -> None:
        """往任务的日志文件追加一行（失败不打断任务本身）。"""
        if task.log_path is None:
            return
        try:
            with task.log_path.open("a", encoding="utf-8") as fh:
                fh.write(text + "\n")
            task.lines += 1
        except OSError:
            task.log_path = None                              # 只报一次，不刷屏

    def stop_task(self, task: Task, graceful: bool = True) -> bool:
        """停掉一个任务（含它的子进程树）。返回是否真的下手了。

        ## 为什么要杀整棵树，而不是 `proc.terminate()`

        子进程是 python；父进程被杀之后**它还可能活着**，于是相机仍被占着，
        下一次运行打不开相机，而用户看到的是"停了但相机用不了"——
        这正是"没有任何中断手段"之外的另一半问题。

        ## 为什么要先优雅后强杀

        `CTRL_BREAK_EVENT` 让程序自己走退出流程（例如标定程序来得及把已经
        收下的图像对写完）。直接 `taskkill /F` 会把它半路打断。
        """
        if not task.running:
            return False
        pid = task.proc.pid
        task.stop_reason = "用户中断" + ("（优雅）" if graceful else "（强制）")
        self._append(f"[任务 {task.tid}] 正在中断 PID {pid}…")
        kids = _child_pids(pid)
        if kids:
            self._append(f"[任务 {task.tid}] 它有 {len(kids)} 个子进程，将一并结束")

        # ① 先请求优雅退出
        if graceful:
            try:
                if os.name == "nt":
                    task.proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(os.getpgid(pid), signal.SIGINT)
            except Exception as exc:                        # noqa: BLE001
                self._append(f"[任务 {task.tid}] 优雅中断不可用（{type(exc).__name__}），"
                             f"将直接强制结束")
            try:
                task.proc.wait(timeout=3.0)
                self._append(f"[任务 {task.tid}] 已优雅退出")
                self._kill_leftover(kids, task)
                return True
            except subprocess.TimeoutExpired:
                self._append(f"[任务 {task.tid}] 3 秒内没退出，改为强制结束")

        # ② 强杀（含子树）
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                               capture_output=True, timeout=10)
            else:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception as exc:                            # noqa: BLE001
            self._append(f"[任务 {task.tid}] 强制结束失败：{type(exc).__name__}: {exc}")
        self._kill_leftover(kids, task)
        return True

    def _kill_leftover(self, kids: list[int], task: Task) -> None:
        """收尾：子树里还有活着的就补刀（否则相机可能仍被占着）。"""
        alive = []
        for pid in kids:
            try:
                import psutil
                if psutil.pid_exists(pid):
                    alive.append(pid)
            except Exception:                               # noqa: BLE001
                pass
        if not alive:
            return
        self._append(f"[任务 {task.tid}] 仍有 {len(alive)} 个子进程存活，补杀：{alive}")
        for pid in alive:
            try:
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"]
                               if os.name == "nt" else ["kill", "-9", str(pid)],
                               capture_output=True, timeout=10)
            except Exception:                               # noqa: BLE001
                pass

    def stop_all(self, graceful: bool = False) -> int:
        """停掉所有在跑的任务，返回停掉几个。"""
        running = [t for t in self.tasks if t.running]
        for t in running:
            self.stop_task(t, graceful=graceful)
        return len(running)

    def _reader(self, task: Task) -> None:
        proc = task.proc
        assert proc.stdout is not None
        for line in proc.stdout:
            body = line.rstrip()
            self._task_log(task, body)                 # ★ 每行都落盘
            self.log_queue.put(f"[{task.tid}] {body}")
        proc.wait()
        task.ended = time.monotonic()
        task.exit_code = proc.returncode
        why = f"（{task.stop_reason}）" if task.stop_reason else ""
        tail = (f"[任务 {task.tid} 结束] 返回码 {proc.returncode}{why}，"
                f"耗时 {task.elapsed():.1f}s")
        self._task_log(task, "=" * 70)
        self._task_log(task, tail)
        if task.log_path is not None:
            self._task_log(task, f"日志文件：{task.log_path}")
        self.log_queue.put(tail)
        self.log_queue.put(None)

    def _poll(self) -> None:
        while True:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            if line is None:
                self.proc = None
                self._refresh_tasks()
                self.status.configure(text="就绪" if not self.running_tasks()
                                      else f"运行中 {len(self.running_tasks())} 个")
                continue
            self._append(line)
        # 计时列要跟着走。**只在真的有任务在跑时刷** ——
        # 空转时每 500ms 重画一次表格纯属浪费。
        if self.running_tasks():
            self._refresh_tasks()
        self.after(POLL_MS, self._poll)

    def running_tasks(self) -> list[Task]:
        return [t for t in self.tasks if t.running]

    # ------------------------------------------------------------------ #
    # 任务列表界面
    # ------------------------------------------------------------------ #
    def _build_tasks(self) -> None:
        """任务管理器：看得到在跑什么、跑了多久，并且**能中断**。

        这一节补的是前面那个"卡死"缺口的另一半 ——
        光让程序说清"我在等什么"还不够，用户还得有**把它停掉**的手段。
        """
        box = self._card(
            "任务",
            "这里列出启动器起过的每个程序。**卡住时用它中断** ——"
            "直接关启动器窗口是不会杀掉子进程的，相机可能仍被占着。"
            "「中断」先试优雅退出（让它自己收尾），3 秒不退再强制结束整棵进程树。")
        cols = ("id", "title", "pid", "elapsed", "status")
        heads = ("#", "任务", "PID", "已运行", "状态")
        widths = (4, 34, 8, 9, 12)
        self.task_tree = ttk.Treeview(box, columns=cols, show="headings",
                                      height=5, selectmode="browse")
        for c, h, w in zip(cols, heads, widths):
            self.task_tree.heading(c, text=h)
            self.task_tree.column(c, width=w * 8, anchor="w", stretch=(c == "title"))
        self.task_tree.grid(row=1, column=0, columnspan=8, sticky="we", pady=(4, 4))
        self.task_tree.bind("<<TreeviewSelect>>", self._on_task_select)

        ttk.Button(box, text="中断选中", command=self._stop_selected).grid(
            row=2, column=0, sticky="w", pady=2)
        ttk.Button(box, text="强制结束选中", command=self._kill_selected).grid(
            row=2, column=1, sticky="w", padx=(6, 0), pady=2)
        ttk.Button(box, text="全部中断", command=self._stop_all_clicked).grid(
            row=2, column=2, sticky="w", padx=(6, 0), pady=2)
        ttk.Button(box, text="查看输出", command=self._show_selected_output).grid(
            row=2, column=3, sticky="w", padx=(6, 0), pady=2)
        ttk.Button(box, text="打开日志文件", command=self._open_selected_log).grid(
            row=2, column=4, sticky="w", padx=(6, 0), pady=2)
        ttk.Button(box, text="清理已结束", command=self._prune_tasks).grid(
            row=2, column=5, sticky="w", padx=(6, 0), pady=2)
        self.task_hint = ttk.Label(box, text="", style="Hint.TLabel",
                                   wraplength=CARD_HINT_WRAP)
        self.task_hint.grid(row=3, column=0, columnspan=9, sticky="w", pady=(4, 0))
        self._wrapped.append((self.task_hint, "card"))

    def _refresh_tasks(self) -> None:
        """重画任务表。**保留选中行**（否则每次刷新都把用户的选择弄丢）。"""
        if not hasattr(self, "task_tree"):
            return
        tree = self.task_tree
        keep = self._sel_tid
        tree.delete(*tree.get_children())
        for t in self.tasks:
            if t.running:
                status = "运行中"
            elif t.stop_reason:
                status = t.stop_reason
            else:
                status = f"已结束(码 {t.exit_code})"
            tree.insert("", "end", iid=str(t.tid),
                        values=(t.tid, t.title, t.proc.pid,
                                f"{t.elapsed():.1f}s", status))
            if keep == t.tid:
                tree.selection_set(str(t.tid))
        n_run = len(self.running_tasks())
        n_all = len(self.tasks)
        self.task_hint.configure(
            text=f"共 {n_all} 个任务，其中 {n_run} 个在运行。"
                 + ("选中一个再点「中断选中」。" if n_run else "当前没有在跑的任务。"))

    def _selected_task(self) -> Task | None:
        sel = self.task_tree.selection() if hasattr(self, "task_tree") else ()
        if not sel:
            return None
        tid = int(sel[0])
        return next((t for t in self.tasks if t.tid == tid), None)

    def _on_task_select(self, _evt=None) -> None:
        t = self._selected_task()
        self._sel_tid = t.tid if t else None

    def _stop_selected(self) -> None:
        t = self._selected_task()
        if t is None:
            messagebox.showinfo("先选一个任务", "在上面的列表里点一行，再点「中断选中」。")
            return
        if not t.running:
            messagebox.showinfo("它已经结束了", f"任务 {t.tid}「{t.title}」已经不在运行。")
            return
        self.stop_task(t, graceful=True)
        self._refresh_tasks()

    def _kill_selected(self) -> None:
        t = self._selected_task()
        if t is None:
            messagebox.showinfo("先选一个任务", "在上面的列表里点一行，再点「强制结束选中」。")
            return
        if not t.running:
            messagebox.showinfo("它已经结束了", f"任务 {t.tid}「{t.title}」已经不在运行。")
            return
        if not messagebox.askyesno(
                "强制结束？",
                f"任务 {t.tid}「{t.title}」会被**立刻杀掉**（含子进程树），"
                "它来不及保存已完成的结果。\n\n确定吗？"):
            return
        self.stop_task(t, graceful=False)
        self._refresh_tasks()

    def _stop_all_clicked(self) -> None:
        n = len(self.running_tasks())
        if not n:
            messagebox.showinfo("没有在跑的任务", "当前没有需要中断的程序。")
            return
        if not messagebox.askyesno("全部中断？", f"将中断 {n} 个正在运行的任务。确定吗？"):
            return
        self.stop_all(graceful=True)
        self._refresh_tasks()

    def _show_selected_output(self) -> None:
        """把这个任务的信息回显到日志区（方便复制）。"""
        t = self._selected_task()
        if t is None:
            messagebox.showinfo("先选一个任务", "在上面的列表里点一行。")
            return
        self._append(f"\n===== 任务 {t.tid}「{t.title}」=====")
        self._append(f"PID   : {t.proc.pid}")
        self._append(f"状态  : {'运行中' if t.running else '已结束'}"
                     + (f"（{t.stop_reason}）" if t.stop_reason else ""))
        self._append(f"耗时  : {t.elapsed():.1f}s")
        self._append(f"命令  : {t.cmdline()}")
        if t.log_path is not None:
            self._append(f"日志  : {t.log_path}（{t.lines} 行已落盘）")

    def _open_selected_log(self) -> None:
        """用系统默认程序打开这个任务的日志文件。

        **为什么要有个按钮**：查问题时第一件事就是"上次到底报了什么"，
        而日志区会被下一条任务冲掉。落盘之后必须让人**点得到**，
        否则文件存在但没人知道去哪找。
        """
        t = self._selected_task()
        if t is None:
            messagebox.showinfo("先选一个任务", "在上面的列表里点一行，再点「打开日志文件」。")
            return
        if t.log_path is None or not t.log_path.exists():
            messagebox.showinfo(
                "还没有日志文件",
                f"任务 {t.tid} 没有落盘日志（可能是目录不可写）。\n"
                "界面上的输出仍然在「运行日志」里。")
            return
        self._append(f"[任务 {t.tid}] 打开日志：{t.log_path}")
        try:
            if os.name == "nt":
                os.startfile(t.log_path)                      # noqa: S606
            else:
                subprocess.Popen(["xdg-open", str(t.log_path)])
        except Exception as exc:                              # noqa: BLE001
            messagebox.showinfo("打不开自动打开",
                                f"日志在：\n{t.log_path}\n\n（{type(exc).__name__}: {exc}）")

    def _prune_tasks(self) -> None:
        """清掉已结束的任务行（在跑的绝不动）。"""
        before = len(self.tasks)
        self.tasks = [t for t in self.tasks if t.running]
        self._refresh_tasks()
        self._append(f"[任务] 清理了 {before - len(self.tasks)} 条已结束记录")

    def _on_close(self) -> None:
        """退出前把子进程**收干净**。

        ⚠️ 原来只弹一句"确定退出吗"就 `destroy()` ——
        子进程**不会被杀**：相机仍被占着，下一次运行打不开相机。
        现在退出前先把在跑的任务停掉（这也是"任务管理器"存在的意义之一）。
        """
        running = self.running_tasks()
        if running:
            names = "、".join(f"{t.tid}「{t.title}」" for t in running[:4])
            more = f" 等 {len(running)} 个" if len(running) > 4 else ""
            if not messagebox.askyesno(
                    "仍有任务在运行",
                    f"还有 {len(running)} 个任务在跑：{names}{more}。\n\n"
                    "退出启动器会**把它们一并结束**（否则相机等资源可能一直被占着）。\n"
                    "确定退出吗？"):
                return
            self._append("[退出] 正在结束仍在运行的任务…")
            n = self.stop_all(graceful=True)
            self._append(f"[退出] 已请求结束 {n} 个任务")
        self.destroy()


def main() -> int:
    Launcher().mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
