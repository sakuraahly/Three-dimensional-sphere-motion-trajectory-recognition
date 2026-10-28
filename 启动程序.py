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
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

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

# 检测方法下拉框：**唯一来源是检测器注册表**（`sphere_tracker/detectors/registry.py`，规则 R7）。
# 以前这里是手写清单，于是四个入口已经漂移（GUI 7 项 / run_stereo_video 5 项 / run_all 4 项）；
# 现在加一个检测器只改注册表一处，界面与命令行不可能再对不上。
# 顺序 = 注册顺序；默认值 = 注册表声明的默认（= `DetectorConfig.method`），
# 不再靠"列表首项"这种隐式约定 —— 展示顺序与默认值是两件独立的事。
from sphere_tracker.config import ColorConfig as _ColorConfig  # noqa: E402
from sphere_tracker.detectors import registry as _detector_registry  # noqa: E402,F401  （import 即注册）
from sphere_tracker.plugins import default as _plugin_default  # noqa: E402
from sphere_tracker.plugins import describe as _plugin_describe  # noqa: E402
from sphere_tracker.plugins import names as _plugin_names  # noqa: E402

# 标定方法同理：唯一来源是 `calib/methods.py` 的注册表（规则 R7）。
from calib.methods import plan as _calib_method_plan  # noqa: E402
from calib.methods import single_board as _single_board_method  # noqa: E402

METHODS = _plugin_names("detector")
DEFAULT_METHOD = _plugin_default("detector")
# 双目标定的「外参方法」下拉：选项与默认值都来自注册表，界面里不再写方法名。
CALIB_METHODS = _plugin_names("calib_method")
DEFAULT_CALIB_METHOD = _plugin_default("calib_method")
# 颜色预设下拉：**唯一来源是 `sphere_tracker/config.py` 的 `COLOR_PRESETS`**（R7 同理）。
# 以前这里是手写的七项清单，同样抄了一份在 `run_realtime` 的帮助文本和
# `calibrate_color` 的按键表里 —— 加一种球色要改四处、还漏得掉。
# 「不使用」是界面自己的哨兵值（= 不传 `--preset`，用配置里的颜色），不是预设。
PRESETS = ["不使用", *_ColorConfig.preset_names()]
# 左右下拉的默认项：直接给**角色名**（left/right），绑定的设备身份不随 USB 口漂移。
# 数字编号仍可手选，但换一次接口就可能指向别的相机。
CAMERAS = ["left", "right", "auto", "0", "1", "2"]
# 单目相机：默认 auto（启动时自动挑最优），也可选具体编号。
# 注：曾经还有 "main"（笔记本内置）这一项，已移除 —— 内置相机被标进
# config/cameras.json 的「无关」清单，角色 main 也不再存在；留着这个选项
# 只会让用户选到一个必定报错的项。
MONO_CAMERAS = ["auto", "0", "1", "2"]
# 分辨率档位**只在 `sphere_tracker/video_profiles.py` 定义一次**：
# 这里再写一遍就是下一次"标定 1080p、录制 1440p"静默混用的种子。
from sphere_tracker.video_profiles import (  # noqa: E402  （要等 ROOT 进 sys.path）
    DEFAULT as DEFAULT_VIDEO_PROFILE,
    DEFAULT_FPS as DEFAULT_VIDEO_FPS,
    detection_width,
    options as _video_profile_options,
    parse as parse_video_profile,
)
RESOLUTIONS = _video_profile_options()
# `video_profiles` 是分辨率档位的唯一来源；默认档也只用它的 DEFAULT，
# 不在 GUI 另写一个 1080p。默认 2560x1440@24：真机实测两台外接相机无视
# 1920/1280 请求、恒给 2560x1440，且 1440p@30 主循环只有 24.07fps（会丢帧）。
DEFAULT_RESOLUTION = DEFAULT_VIDEO_PROFILE.label
CALIB_FPS_DEFAULT = str(DEFAULT_VIDEO_FPS)   # 唯一来源：video_profiles.DEFAULT_FPS
# 标定面板的两个对数默认值。写成常量而不是散在控件里的字符串字面量 ——
# 否则"测试只能靠 grep 字符串"来确认它们，改一下控件类型（StringVar→IntVar）就误报。
CALIB_TARGET_PAIRS = 40
CALIB_MIN_PAIRS = 20

# 标定结果：**几何的权威来源**（档案里的 B/D 只是参考估算，见 _refresh_profile_line）。
CALIB_PATH = ROOT / "config" / "stereo_calibration"


def calibration_measure() -> tuple[float, float] | None:
    """读标定**实测**的 ``(基线 mm, 光轴夹角 °)``；没有可用标定时返回 ``None``。

    为什么复用 ``calib.params.StereoRig`` 而不自己解析：基线与夹角分别由标定
    ``T`` / ``R`` 算出（``angle_deg`` 不是手填的 90.0），``StereoRig`` 已经把这两件事
    做对了，GUI 里再写第二份解析就会与 ``apps/tools/check_stereo_setup.py`` 的口径分叉。
    读不到（没标定过 / 文件坏了）**不是错误**：顶部状态条写"尚无标定"即可。
    """
    try:
        from calib.params import StereoRig

        rig = StereoRig.load(CALIB_PATH)
    except (OSError, ValueError, KeyError):
        return None
    return float(rig.baseline_mm), float(rig.angle_deg)


# --------------------------------------------------------------------------- #
# 单板链式标定（第三条路）：四段采集 + 一条标定命令
#
# 为什么需要它：joint 要求两路**充分共视**；90° 要求两路能和**同一块板同框**。
# 两路几乎看不到同一区域时这两条都不成立，只能让板**不动**、各自拍一组参考图，
# 再用同一块板的几何把两路串起来 —— 这就是 `--method chain --same-board`。
#
# 这里只放**纯函数**（不进 Tk、不读面板）：界面上点按钮和自测用的是同一份拼装，
# 免得"界面一套、测试另一套"之后悄悄分叉。
# --------------------------------------------------------------------------- #
# 三段（不是四段）：参考段改成**一次双路同拍**。
#
# 为什么合并：以前 ref_left / ref_right 分两次按，中间只要板被碰一下，
# 外参就会错而界面上完全看不出来。`--both` 让两路在同一时刻存同一块板，
# 左右之间不会再有时间差，也就不存在"中间被动过"这个失败模式。
CHAIN_STAGES = (
    ("left", "left", 15),
    ("right", "right", 15),
    ("ref", "both", 3),
)
# 每个阶段落到哪些**计数目录**（参考段有两个，分别数）
CHAIN_STAGE_DIRS = {
    "left": ("left",),
    "right": ("right",),
    "ref": ("ref_left", "ref_right"),
}
# 就绪判定 / 缺图提示按"目录"走：四个目录各有各的目标张数
CHAIN_DIR_TARGET = {"left": 15, "right": 15, "ref_left": 3, "ref_right": 3}
CHAIN_REQUIRED_DIRS = ("left", "right", "ref_left", "ref_right")
CHAIN_DIR_TEXT = {
    "left": "拍左内参图",
    "right": "拍右内参图",
    "ref_left": "拍左参考图",
    "ref_right": "拍右参考图",
}
CHAIN_STAGE_TEXT = {
    "left": ("拍左内参图",
             "只用左相机，覆盖画面四角与不同倾角；"
             "自动连拍：板停稳就会自动存，不用按键"),
    "right": ("拍右内参图",
              "只用右相机，同样覆盖四角与不同倾角；"
              "自动连拍：板停稳就会自动存，不用按键"),
    "ref": ("拍参考图（双路同时）",
            "一块板同时给左右两路拍；板固定后不要再碰，"
            "自动连拍会自己存够目标张数"),
}


def _count_shots(directory) -> int:
    """数目录里的 `*.png` 张数（不递归）。目录不存在 / 读不了都算 0。

    为什么只数 png：采集工具统一写 png；数别的会把残留的说明文件算成"拍到了"。
    """
    try:
        return sum(1 for p in Path(directory).glob("*.png") if p.is_file())
    except OSError:
        return 0


def _chain_ready(counts: dict[str, int]) -> bool:
    """四个目录**各至少 1 张**才允许开始标定（参考段少于 3 张只提示，不拦）。

    为什么门槛放在 1 张而不是 3 张：链式外参只要有参考图就能算，硬卡 3 张
    会让"只拍到 2 张但完全够用"的人点不动按钮；而**数量不足的风险必须说出来**，
    所以提示文字单独写在 `_chain_status_text` 里，不混进按钮可用性。
    """
    return all(int(counts.get(key, 0) or 0) >= 1 for key in CHAIN_REQUIRED_DIRS)


def _chain_missing(counts: dict[str, int]) -> str:
    """缺什么、缺几张 —— 按钮是灰的，必须在这里说清原因，不能让人猜。"""
    bits = []
    for key in CHAIN_REQUIRED_DIRS:
        got = int(counts.get(key, 0) or 0)
        if got < 1:
            bits.append(f"{CHAIN_DIR_TEXT[key]} 0/{CHAIN_DIR_TARGET[key]} 张")
    return "、".join(bits)


def _chain_status_text(counts: dict[str, int]) -> str:
    """按钮旁边那一行状态：缺图 / 参考段偏少 / 可以开始。"""
    if not _chain_ready(counts):
        return "还不能标定：缺 " + _chain_missing(counts)
    weak = [f"{CHAIN_DIR_TEXT[k]} {int(counts.get(k, 0) or 0)}/{CHAIN_DIR_TARGET[k]} 张"
            for k in CHAIN_REQUIRED_DIRS
            if k.startswith("ref") and int(counts.get(k, 0) or 0) < CHAIN_DIR_TARGET[k]]
    if weak:
        return "可以开始（参考段偏少，建议补满：" + "、".join(weak) + "）"
    return "三段都齐了，可以开始链式标定"


def chain_capture_cmd(root, stage: str, *, camera, width, height, fps,
                      backend: str, cols, rows, square) -> list[str]:
    """一段采集的 argv（契约由 `apps/tools/capture_board.py` 定，别在这里改语义）。

    - 内参段：`--side left|right --out <root>/left|right --auto`
    - 参考段：`--both --out-left <root>/ref_left --out-right <root>/ref_right --auto`
      —— **不带 `--side`**：那是单路标记，带上去双路就没意义了。
    """
    sub, side, target = next(s for s in CHAIN_STAGES if s[0] == stage)
    base = Path(root)
    cmd = [PY, str(ROOT / "apps" / "tools" / "capture_board.py")]
    if sub == "ref":
        cmd += ["--both",
                "--out-left", str(base / "ref_left"),
                "--out-right", str(base / "ref_right")]
    else:
        cmd += ["--out", str(base / sub), "--side", side,
                "--camera", str(camera)]
    cmd += ["--width", str(width), "--height", str(height), "--fps", str(fps),
            "--cols", str(cols), "--rows", str(rows), "--square", str(square),
            "--target", str(target), "--auto"]
    if backend and backend != "auto":
        cmd += ["--backend", str(backend)]
    return cmd


def chain_calibrate_cmd(root, *, cols, rows, square, width, height, fps,
                        backend: str, target, min_pairs,
                        method: str | None = None) -> list[str]:
    """链式标定的 argv。

    ``method`` 决定 `--method` 与要不要补 `--same-board`：默认取注册表里
    「需要单块固定板」的那个方法（`calib/methods.py` 的 `single_board()`），
    界面里没有再写一遍方法名。`--same-board` 只在该方法声明需要时追加 ——
    它是**前置条件**，不是链式流程的装饰。

    ⚠️ **不带 `--stereo-mode angled`**：那是「90° 装置」那条路的标记，
    两条路的前提不同，混在一起会让标定脚本按错误假设取画面。
    """
    base = Path(root)
    plan = _calib_method_plan(method or _single_board_method())
    cmd = [PY, str(ROOT / "apps" / "calibrate_stereo.py"),
           "--method", plan.name]
    if plan.same_board:
        cmd += ["--same-board"]
    cmd += ["--left-dir", str(base / "left"),
            "--right-dir", str(base / "right"),
            "--ref-left", str(base / "ref_left"),
            "--ref-right", str(base / "ref_right"),
            "--cols", str(cols), "--rows", str(rows), "--square", str(square),
            "--target", str(target), "--min-pairs", str(min_pairs),
            "--width", str(width), "--height", str(height), "--fps", str(fps)]
    if backend and backend != "auto":
        cmd += ["--backend", str(backend)]
    return cmd


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

# 检测窗口的显示图宽度（像素）。**与 `run_realtime.py --display-width` 同义。**
#
# 为什么要有这个默认值（2026-10-14 实测）：`cv2.imshow` 把 1920x1080 的图
# 缩进窗口显示要 **17.74 ms/帧**，而先把图缩到 960x540 再显示只要 **1.84 ms**
# （便宜 9.6 倍）。也就是每秒最多 56 帧是**花在缩放上**的，
# 而这件事我们自己做只要 0.55 ms。这是"画面卡顿"最大的单项。
DISPLAY_WIDTH = 960

# 界面刷新周期（毫秒）。轮询日志 + 刷新任务耗时列，都用这一个值 ——
# 原来轮询是 120ms、任务列又自己排了 500ms，两套节奏叠在一起。
POLL_MS = 200


def _cam_arg(raw: str) -> str:
    """下拉里可能是 `0 | aoni webcam A25 | 6&...`，取前面那个编号；否则原样返回角色名。"""
    m = re.match(r"\s*(\d+)", raw or "")
    return m.group(1) if m else (raw or "").strip()


def _resolved_entry(resolved, by_order):
    """`ResolvedCamera` 对应的 PnP 设备（`DeviceEntry`）；拿不到返回 None。

    ## 为什么必须有这一层（2026-10-23 修的真实误报）
    `ResolvedCamera.index` 现在是**采集索引**（OpenCV/MSMF 实际索引），
    而 `ResolvedCamera.entry` 才是 PnP 设备 —— 两者**可以不同**。
    本机实测：`Web Camera Left` 是采集 #2，但它的 PnP 序号是 #0。
    按采集索引去 `by_order` 里查，会拿到 PnP #2 `Integrated IR Camera`，
    于是外接相机被误报成「内置/红外」，用户差点去换相机。

    所以**优先用 `entry`**；只有 `entry is None`（索引兜底 / 手动编号）时，
    才退回按 `index` 查 —— 那是旧行为，也是唯一没有 PnP 身份可用的场景。
    """
    if resolved.entry is not None:
        return resolved.entry
    return by_order.get(resolved.index)


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
    kill_note: str = ""          # 停止失败的原因（"没杀干净"必须有出口，见 stop_task）
    lines: int = 0
    on_done: Callable[["Task"], None] | None = None

    @property
    def running(self) -> bool:
        return self.proc.poll() is None

    def elapsed(self) -> float:
        end = self.ended if self.ended is not None else time.monotonic()
        return max(0.0, end - self.started)

    def cmdline(self) -> str:
        return " ".join(self.cmd)


def _alive_pids(pids: list[int]) -> list[int]:
    """这批 pid 里**还有哪些活着**（判不出来的一律当成"还活着"）。

    ## 为什么需要它（而不是直接相信 `taskkill` 的调用没抛异常）

    `subprocess.run(["taskkill", ...])` **不抛异常不代表杀掉了**：权限不足时
    它只是返回 `1` 并把 `ERROR: Access denied` 写进 stderr，而
    `capture_output=True` 会把这个返回值与输出**一起吞掉** ——
    调用方看不到任何异常，于是"杀失败"与"杀成功"长得一模一样。

    这与 `LatestFrameCamera` 的静默重试是同一类缺陷：**失败没有出口**。
    所以这里坚持**按结果判定**：真的去问一次"还活着吗"。
    """

    alive: list[int] = []
    for pid in pids:
        try:
            import psutil                                   # 可选依赖
            if psutil.pid_exists(pid):
                alive.append(pid)
        except Exception:                                   # noqa: BLE001
            # 没有 psutil / 查询被拒：**当成还活着**。
            # 宁可多报一次"可能没杀干净"，也不要谎报成功 ——
            # 谎报的代价是相机一直被占着而用户以为停了。
            alive.append(pid)
    return alive


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
        from apps._gui_geometry import fit_window
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        screen_x = screen_y = 0
        # winfo_screenheight() 包含任务栏；用 Windows 工作区避免窗口底边被盖住。
        if sys.platform == "win32":
            try:
                import ctypes
                from ctypes import wintypes
                rect = wintypes.RECT()
                if ctypes.windll.user32.SystemParametersInfoW(
                        0x0030, 0, ctypes.byref(rect), 0):   # SPI_GETWORKAREA
                    screen_x, screen_y = int(rect.left), int(rect.top)
                    screen_w = int(rect.right - rect.left)
                    screen_h = int(rect.bottom - rect.top)
            except Exception:                               # noqa: BLE001
                pass
        win_w, win_h, win_x, win_y = fit_window(screen_w, screen_h)
        self.geometry(f"{win_w}x{win_h}+{win_x + screen_x}+{win_y + screen_y}")
        # 屏幕小于原设计时，最小尺寸也跟着缩小，绝不把窗口顶出屏幕。
        self.minsize(min(900, win_w), min(600, win_h))

        self.proc: subprocess.Popen | None = None
        self.log_queue: queue.Queue = queue.Queue()
        self.done_queue: queue.Queue = queue.Queue()
        # 要跟着视口宽度重新换行的标签：(控件, "卡片说明"|"参数说明")
        self._wrapped: list[tuple[ttk.Label, str]] = []
        # 任务表：所有跑过的程序（在跑的 + 历史）。`_tid` 只增不减。
        self.tasks: list[Task] = []
        self._tid = 0
        self._sel_tid: int | None = None
        self._wb_check_result: dict[str, bool] = {}
        self._wb_check_inflight: set[str] = set()
        self._wb_auto_after: str | None = None
        self._record_before: set[str] = set()
        self.stereo_mode_var = tk.StringVar(value="auto")

        self._init_style()
        self._build_header()
        self._build_nav()
        # 滚动页是**所有卡片与日志的父容器**（在 header 之后创建，
        # 于是它在 pack 顺序里排在 header 下面）。窗口再高也只显示一屏，
        # 其余靠滚轮 / PageUp·PageDown 浏览 —— 像浏览器一样。
        self._build_scroll_page()
        self._build_step1()
        self._build_step2()
        self._build_step3()
        self._build_workbench()
        self._build_tasks()      # 任务管理器：看得见、停得掉
        self._build_log()
        self._bind_scroll()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(POLL_MS, self._poll)
        # 启动时只**枚举**设备填进下拉，_refresh_cameras(probe=False) 不打开相机：
        # 原来这里走 probe=True，界面一出现就真开相机（亮灯 + 白等二三十秒），
        # 而用户可能只是想看一眼界面。要实测能不能出图，点「刷新并检测相机」。
        self.after(200, lambda: self._refresh_cameras(probe=False))
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
        from apps._gui_geometry import wrap_width
        avail = wrap_width(viewport, chrome=CHROME_PX, minimum=240)
        card = max(200, avail)
        # 参数说明在输入框右侧起，再让出标签列与内边距
        param = max(200, avail - 150)
        for lab, kind in self._wrapped:
            want = card if kind == "card" else param
            if int(lab.cget("wraplength") or 0) != want:
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
        brand = tk.Frame(head, bg=ACCENT)
        brand.pack(side="left", fill="x", expand=True, padx=20, pady=(12, 10))
        self._header_brand = brand
        tk.Label(brand, text="球形物体三维运动轨迹识别",
                 bg=ACCENT, fg="white",
                 font=("Microsoft YaHei UI", 17, "bold")).pack(anchor="w")
        tk.Label(brand, text="检测 → 跟踪 → 三维定位 → 轨迹预测    三步即可开始",
                 bg=ACCENT, fg="#dbeafe",
                 font=("Microsoft YaHei UI", 10)).pack(anchor="w", pady=(2, 0))

        actions = tk.Frame(head, bg=ACCENT)
        actions.pack(side="right", padx=(8, 16), pady=10)
        self._header_actions = actions
        mode_box = tk.Frame(actions, bg=ACCENT)
        mode_box.pack(anchor="e")
        tk.Label(mode_box, text="双目模式", bg=ACCENT, fg="#dbeafe",
                 font=("Microsoft YaHei UI", 9)).pack(side="left", padx=(0, 6))
        ttk.Combobox(mode_box, textvariable=self.stereo_mode_var, width=9,
                     state="readonly",
                     values=("auto", "parallel", "angled")).pack(side="left")
        tk.Label(mode_box, text="auto=按标定/会话判定",
                 bg=ACCENT, fg="#bfdbfe",
                 font=("Microsoft YaHei UI", 8)).pack(side="left", padx=(6, 0))
        tk.Button(mode_box, text="模式说明", command=self._show_mode_help,
                  bg="#1d4ed8", fg="white", relief="flat", padx=6,
                  font=("Microsoft YaHei UI", 8)).pack(side="left", padx=(6, 0))
        tk.Button(mode_box, text="基线夹角估算", command=self._show_geometry,
                  bg="#1d4ed8", fg="white", relief="flat", padx=6,
                  font=("Microsoft YaHei UI", 8)).pack(side="left", padx=(6, 0))
        tk.Button(actions, text="装置档案", command=self._open_profiles,
                  bg="white", fg=ACCENT, relief="flat", padx=14, pady=4,
                  font=("Microsoft YaHei UI", 10, "bold")).pack(
            anchor="e", pady=(6, 0))
        tk.Button(actions, text="数据管理", command=self._open_data_manager,
                  bg="white", fg=ACCENT, relief="flat", padx=14, pady=4,
                  font=("Microsoft YaHei UI", 10, "bold")).pack(
            anchor="e", pady=(6, 0))
        # 改名/改备注只在『数据管理』里做（那里能选中会话、看到当前值），
        # 启动器不再摆第二套入口 —— 这里只负责把"能做什么"说清楚。
        tk.Label(actions, text="可改名 / 改备注 · 清理 · 脱敏导出",
                 bg=ACCENT, fg="#bfdbfe", wraplength=220, justify="right",
                 font=("Microsoft YaHei UI", 8)).pack(anchor="e", pady=(2, 0))

        # ---- 常驻装置状态条（P0-1）----
        # 为什么单开一行而不是塞进标题右侧：这一行会随档案变化（名称 / 模式 /
        # 几何提示），塞进右上角那排按钮里会挤动布局，而且一长就被截断。
        # 它是**常驻**的：模式下拉选了什么、当前哪套档案生效，不必再点开弹窗确认。
        self._profile_bar = tk.Frame(self, bg="#1e40af")
        self._profile_bar.pack(fill="x")
        self._profile_line = tk.Label(
            self._profile_bar, text="当前装置：读取中…", bg="#1e40af", fg="#dbeafe",
            font=("Microsoft YaHei UI", 9), anchor="w", justify="left",
            wraplength=CARD_HINT_WRAP)
        self._profile_line.pack(side="left", padx=20, pady=4)
        self._wrapped.append((self._profile_line, "card"))
        self._refresh_profile_line()

    def _show_mode_help(self) -> None:
        """三种模式**行为差异**（P0-2）：下拉框只放英文值，含义写在这里。

        为什么不让下拉项直接写中文：`--stereo-mode` 收的是
        ``auto/parallel/angled``，把展示文案写进 `StringVar` 会把
        "平行（FOV 裁剪）" 这样的字符串原样传给命令行。
        """
        messagebox.showinfo(
            "双目模式：三者行为不同",
            "parallel（平行）\n"
            "  两光轴近乎平行；标定会做 FOV 裁剪对齐，取两路共同可见区。\n"
            "  适合同一根横梁上的并排相机（B 远小于 D）。\n\n"
            "angled（大角度 / 90°）\n"
            "  两光轴夹角大（近似直角）；各自用原始画面，不做 FOV 裁剪；\n"
            "  输出坐标系为 left_camera。适合 B ≈ 2·D 的 90° 装置。\n\n"
            "auto（自动判定）\n"
            "  按标定记录 / 会话记录推断上面哪一种；仅作输入，\n"
            "  解析结果不写回档案、不落盘。没有依据时按平行处理。\n\n"
            "提示：当前生效的档案与几何提示常驻在窗口顶部那条深蓝横条上。",
            parent=self)

    def _refresh_profile_line(self) -> None:
        """刷新顶部状态条：**以标定实测为主语**，档案的 B/D 只是后置参考。

        2026-10-24 按用户反馈改。原实现以"档案的期望值"为主语（还跟着一段
        基于期望 B/D 的几何估算），等于让一个**估出来的装置尺寸**抢了权威位；
        用户原话："我不能精确确定摄像头的位置，所以这个不该设置成强相关固定，
        相反，应该以标定计算得到结果为主。"

        现在的语序：``当前装置：<档案>·<模式> · 标定实测：基线/夹角 · 参考 B/D``。
        读不到档案**不是**错误 —— 全新机器上就是没有这份文件，但要说清下一步点哪里。
        """
        from stereo import profiles as rig_profiles
        from stereo.stereo_mode import mode_label

        measure = calibration_measure()
        if measure is None:
            actual = "标定实测：尚无标定"
        else:
            actual = f"标定实测：基线 {measure[0]:.1f} mm / 夹角 {measure[1]:.1f}°"

        try:
            profile = rig_profiles.get_active()
        except Exception as exc:                            # noqa: BLE001
            text = (f"当前装置：读取档案失败（{exc}）"
                    f" · {actual}"
                    " · 下一步：点右上『装置档案』检查 config/profiles.json")
            self._profile_line.configure(text=text)
            return
        if profile is None:
            self._profile_line.configure(
                text=f"当前装置：未设置 · {actual}"
                     " · 点右上『装置档案』新建，可保存平行 / 90° 两套参数")
            return

        ref_bits = []
        if profile.expected_baseline_mm is not None:
            ref_bits.append(f"B={profile.expected_baseline_mm:.0f} mm")
        if profile.working_distance_m is not None:
            ref_bits.append(f"D={profile.working_distance_m:.2f} m")
        ref = f" · 参考 {' / '.join(ref_bits)}（仅作参考）" if ref_bits else ""
        self._profile_line.configure(
            text=f"当前装置：{profile.name} · {mode_label(profile.stereo_mode)} · {actual}{ref}")

    def _open_profiles(self) -> None:
        """『装置档案』：新建 / 覆盖一套装置参数并设为当前（P0-1 的入口）。

        只做一件最小可用的事：名称 + 模式（**这两项就够**），B/D 可选。
        故意**不做**"批量管理"—— 多档案的增删已经在 `stereo/profiles.py`
        里，界面先保证"能建、能切换、能看见"，而不是造一套平行实现。

        B/D 的默认值从 600 / 0.30 改成了**空**：预填具体数字等于暗示"这里必须填准"，
        而用户明确说过"我不能精确确定摄像头的位置"—— 判定看标定，不看这两个框。
        """
        from stereo import profiles as rig_profiles
        dlg = tk.Toplevel(self)
        dlg.title("装置档案")
        dlg.configure(bg=BG)
        dlg.transient(self)
        body = ttk.Frame(dlg, padding=12)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="当前活动档案："
                  + (rig_profiles.active_name() or "未设置")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        rows = (("名称", "rig1"), ("模式（parallel / angled）", "angled"),
                ("期望基线 B（mm，可选，仅作参考）", ""),
                ("工作距离 D（米，可选，仅作参考）", ""))
        vars_: list[tk.StringVar] = []
        for i, (label, default) in enumerate(rows, start=1):
            ttk.Label(body, text=label).grid(row=i, column=0, sticky="e", pady=3)
            var = tk.StringVar(value=default)
            ttk.Entry(body, textvariable=var, width=24).grid(
                row=i, column=1, sticky="w", padx=(6, 0), pady=3)
            vars_.append(var)
        ttk.Label(body, text="只填名称 + 模式即可保存。B / D 可留空 —— 它们是装置的"
                             "粗略估计，几何判定以 config/stereo_calibration 的"
                             "实测基线 / 夹角为准；填了也只作参考。\n"
                             "保存会覆盖同名档案；不合格的值会被拒绝并说明原因。",
                  style="Hint.TLabel", wraplength=380).grid(
            row=len(rows) + 1, column=0, columnspan=2, sticky="w", pady=(8, 6))

        def save() -> None:
            name, mode, baseline, distance = (v.get().strip() for v in vars_)
            try:
                profile = rig_profiles.RigProfile(
                    name=name, stereo_mode=mode,
                    expected_baseline_mm=float(baseline) if baseline else None,
                    working_distance_m=float(distance) if distance else None)
                rig_profiles.upsert(profile, replace_existing=True)
                rig_profiles.set_active(profile.name)
            except Exception as exc:                        # noqa: BLE001
                messagebox.showwarning("装置档案", f"没保存：{exc}", parent=dlg)
                return
            self._refresh_profile_line()
            self._append(f"[装置] 已保存并设为当前：{profile.name}"
                         f"（{mode}）；{rig_profiles.geometry_hint(profile)}")
            dlg.destroy()

        ttk.Button(body, text="保存并设为当前", command=save).grid(
            row=len(rows) + 2, column=0, sticky="w")
        ttk.Button(body, text="取消", command=dlg.destroy).grid(
            row=len(rows) + 2, column=1, sticky="e")


    def _show_geometry(self) -> None:
        """用镜头中心距和工作距离估算两光轴夹角；不替用户假定 90°。"""
        from stereo.stereo_mode import geometry_advice
        try:
            baseline = simpledialog.askfloat(
                "用基线算夹角",
                "两镜头中心距（基线 B，毫米）：",
                initialvalue=65.0, parent=self)
            if baseline is None:
                return
            distance = simpledialog.askfloat(
                "用基线算夹角",
                "球到相机的工作距离（D，米）：",
                initialvalue=0.30, parent=self)
            if distance is None:
                return
        except Exception as exc:                            # noqa: BLE001
            messagebox.showwarning("用基线算夹角", f"输入不合法：{exc}", parent=self)
            return
        advice = geometry_advice(baseline, distance)
        if not advice["warning"]:
            messagebox.showwarning("用基线算夹角", "基线和距离要为正数。", parent=self)
            return
        messagebox.showinfo(
            "用基线算夹角",
            f"B={baseline:.1f} mm, D={distance:.3f} m\n"
            f"两光轴夹角 ≈ {advice['angle_deg']:.2f}°\n"
            f"判断：{advice['warning']}\n\n"
            "这是“两相机对称看向同一工作点”的估算。\n"
            "90° 对称双目需要 B = 2·D；当前装置是否满足，要按实测 B/D 判断。",
            parent=self)

    def _build_nav(self) -> None:
        """固定快速跳转条：页面很长，但用户不必先猜“下面还有什么”。"""
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=16, pady=(6, 0))
        ttk.Label(bar, text="快速跳转", style="Hint.TLabel").pack(
            side="left", padx=(0, 8))
        for label, prefix in (
                ("① 标定", "第 1 步"), ("② 参数", "第 2 步"),
                ("③ 开始", "第 3 步"), ("工作台", "双目工作台"),
                ("任务", "任务"), ("日志", "运行日志")):
            ttk.Button(bar, text=label, width=8,
                       command=lambda p=prefix: self._scroll_to_card(p)).pack(
                side="left", padx=(0, 4))
        ttk.Label(bar, text="页面较长：滚轮 / PageDown / 右侧滚动条",
                  style="Hint.TLabel").pack(side="right")

    def _scroll_to_card(self, prefix: str) -> None:
        self.update_idletasks()
        target = None
        for card in self._page.winfo_children():
            if card.winfo_class() == "TLabelframe" and str(
                    card.cget("text")).startswith(prefix):
                target = card
                break
        if target is None:
            return
        y = max(0, target.winfo_y())
        total = max(1, self._page.winfo_height())
        self._canvas.yview_moveto(min(1.0, y / total))
        self._canvas.focus_set()

    def _open_data_manager(self) -> None:
        self._run([PY, str(ROOT / "apps" / "data_manager.py")], "数据管理")

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

    def _param_combo(self, parent, label: str, values: list[str], value: str,
                     hint: str = "", width: int = 10) -> ttk.Combobox:
        """一行参数：`标签 [下拉框]` 一行，说明一行。

        与 `_param_entry` 同一套排版（说明放**下面**，避免撑开整张卡片），
        只是把输入框换成只读下拉 —— 有固定可选值的参数用它。
        """
        row = self._next_row(parent)
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(4, 0))
        cb = ttk.Combobox(parent, values=values, width=width, state="readonly")
        cb.set(value)
        cb.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=(4, 0))
        if hint:
            self._hint(parent, hint)
        return cb

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

    def _calib_params(self) -> tuple:
        """标定参数区：**基本设置 / 高级设置** 两个可折叠区。

        为什么分区（全局 §四「渐进式展示」）：标定有十几个可调项，
        全铺在第一屏会让人不知道哪些真的必须改。分成两块之后：
          · 基本 = **改错就白做**的那几项（板参数、收多少对、分辨率）；
          · 高级 = **用精度换成功率**的开关（停稳/清晰度/放宽/基线核对）。
        高级默认收起，需要时展开；控件不销毁，值不丢。
        """
        from apps._collapsible import CollapsibleSection
        basic_section = CollapsibleSection(
            self.param_holder, "基本设置（常用）", expanded=True, padding=(10, 8))
        basic_section.pack(fill="x", pady=(2, 4))
        adv_section = CollapsibleSection(
            self.param_holder, "高级设置（精度 / 排障）", expanded=False, padding=(10, 8))
        adv_section.pack(fill="x", pady=(0, 4))
        self._calib_sections = (basic_section, adv_section)
        basic = basic_section.content
        adv = adv_section.content

        # ---- 基本 ----
        #
        # 分组标题也走 `_hint()`：**行号统一由游标分配**，一处手算都没有。
        self._hint(basic, "① 收多少对（越多越准，40 对约需举板几分钟）")
        self.calib_target = self._param_entry(
            basic, "目标对数", str(CALIB_TARGET_PAIRS),
            f"收够就自动开始标定（默认 {CALIB_TARGET_PAIRS} = 正式档）")
        self.calib_minpairs = self._param_entry(
            basic, "最少对数", str(CALIB_MIN_PAIRS),
            f"少于 {CALIB_MIN_PAIRS} 对不做标定；剔除离群后不足会直接拒绝")

        self._hint(basic, "② 标定板（必须和手上那张纸完全一致）")
        self.calib_cols = self._param_entry(
            basic, "内角点·列", "8", "内角点数，不是方格数：8×5 内角点 = 9×6 方格")
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
        self.calib_res.set(DEFAULT_RESOLUTION)
        self.calib_res.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=(4, 0))
        ttk.Label(basic, text="帧率").grid(row=row, column=2, sticky="e",
                                          padx=(24, 6), pady=(4, 0))
        self.calib_fps = ttk.Entry(basic, width=6)
        self.calib_fps.insert(0, CALIB_FPS_DEFAULT)
        self.calib_fps.grid(row=row, column=3, sticky="w", pady=(4, 0))
        self._hint(basic, "两台相机必须同分辨率、同后端，否则一对有效图像都收不到")
        # 选到 >1080p 的档时**必须当场说清代价**，不能让人以为"相机支持就一定能录"。
        # 用单独一行而不是塞进上面的说明：1080p 用户不需要看这段噪音。
        self.res_note = ttk.Label(basic, text="", style="Hint.TLabel",
                                  wraplength=PARAM_HINT_WRAP)
        self.res_note.grid(row=self._next_row(basic), column=0, columnspan=4,
                           sticky="w", pady=(0, 4))
        self._wrapped.append((self.res_note, "param"))
        self.calib_res.bind("<<ComboboxSelected>>",
                            lambda _e: self._refresh_res_note())
        self._refresh_res_note()

        # ---- 高级 ----
        self._hint(adv, "这几项都是拿精度换成功率，不确定就别动")
        # ⚠️ **默认值直接填进输入框**（用户要求）。原来这些框是**空的**，
        # 只在说明里写"留空=默认 3" —— 那等于让用户自己去记默认值是多少，
        # 而且"空"与"默认"在界面上分不出来。现在填上真实默认值，
        # 说明改成"清空 = 不启用这一项"（因为这些参数是**不传就不启用**的）。
        self.calib_stationary = self._param_entry(
            adv, "停稳上限 px", "3", "清空 = 用程序内置默认；这条同时替代左右同步，"
                                     "调大到 10 以上会让两路拍到不同位姿、R/T 系统性偏")
        self.calib_sharp = self._param_entry(
            adv, "清晰度下限（focus）", "800",
            "800 是程序默认值，低于相机能检出棋盘的下限，所以平时不构成闸门；"
            "真正生效的是「相对本组中位数 40%」——只拒明显比同组糊的帧。"
            "只有调大才会更严，但会成片拒收（清空 = 不传，仍用程序默认 800）。")
        self.calib_relaxed = tk.BooleanVar(value=False)
        self._param_check(adv, "放宽模式（手持拍摄：放宽收图门限；停稳 / 清晰度若已填，按你填的）",
                          self.calib_relaxed,
                          "只放宽没填的那几项；停稳 / 清晰度这两格填了就以你的值为准。"
                          "用完必须复检极线误差；不达标请处理光照/支撑后关掉它重标")

        self._hint(adv, "④ 后端与核对留证")
        # 后端：**从界面必须能设**（2026-10-14 加）。
        # 为什么加：脚本默认后端取自 `sources.INDEX_BACKEND`（= MSMF），
        # 而本机实测 **MSMF 打开两路要 34.6 s，DSHOW 只要 11.8 s**（快 3 倍），
        # 且两个后端给的分辨率完全一致。标定要反复跑，「前置检查」更是每次都要等，
        # 界面里没有这个开关，从界面走的人就只能白等 —— 那正是
        # 「界面能设的东西不许与真实取值脱节」那条规矩的反面。
        # 默认仍是 auto（= 脚本默认），**不替用户改行为**，只是把选择权交出来。
        self.calib_backend = self._param_combo(
            adv, "相机后端", ["auto", "dshow", "msmf"], "auto",
            hint="auto = 脚本默认（MSMF）。MSMF 打开慢（本机实测两路 34.6 s vs "
                 "DSHOW 11.8 s）；两台相机必须用同一个后端，所以只在这里选一次")
        # 基线留空是**有意的**：程序不知道你的装置中心距，编不出来。
        # 说明里必须讲清"不填 = 这道护栏关着"，而不是让人以为没差。
        self.calib_baseline = self._param_entry(
            adv, "预期基线 mm", "", "★ 两镜头中心距。只有填了才会核对基线 ——"
                                    "不填这道护栏就是关着的（程序无法替你猜这个值）。"
                                    "每次重新摆相机都要重新量，别照抄旧数字")
        self.calib_savepairs = self._param_path(
            adv, "存图像对", "存下来才能离线复查/重标（出问题时唯一能回答「哪一对错了」）",
            is_dir=True)
        self.calib_report = self._param_path(
            adv, "验证图", "标定后出一张可视化验证图", is_dir=False)
        return basic_section, adv_section

    def _refresh_res_note(self) -> None:
        """选了 >1080p 的档时，把"1440@30 会丢帧、24 才是目标"写在界面上。

        依据（2026-10-24 真机实测，取代 2026-10-23 的短测）：`record_stereo`
        2560x1440@30 的**主循环**实测约 24.07fps（10 秒 242/300 帧/路，HFYU，
        两路一致）；**24fps 就是稳定目标**，请求 30 会丢帧 —— 裸采集能到 30
        不等于主循环能到 30。长录/其他机型未验证。
        这类差距画面上不一定看得出来，所以必须在**选择时**就说清；
        1080p 及以下不显示（没这个代价，少一点噪音）。
        """
        profile = parse_video_profile(self.calib_res.get(),
                                      default=DEFAULT_VIDEO_PROFILE)
        if profile.width > 1920:
            text = (f"⚠ {profile.label}：record_stereo 主循环实测约 24.07 fps"
                    f"（242/300 帧·10 秒·HFYU，两路一致），24 fps 是稳定目标；"
                    f"请求 30 会丢帧，帧率已默认 24。长录/其他机型未验证。")
        else:
            text = ""
        self.res_note.configure(text=text)

    # ---------- 第 1 步：标定 ----------
    def _build_step1(self) -> None:
        box = self._card(
            "第 1 步 · 相机标定",
            "左右约定：人面对镜头时，左手边那台为“左”。第一次用请先点"
            "『相机设备管理（识别/固定）』，程序会逐个显示画面让你确认哪台是左、哪台是右；"
            "笔记本内置 / 红外这类不打算用的相机，就在那个画面里按 n 标成「无关」"
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

        # 参数区放 `param_holder` 里，让「运行标定」按钮排在参数**之后**
        # 按阅读顺序：选相机 → 调参数 → 点运行。
        self.param_holder = ttk.Frame(box)
        self.param_holder.grid(row=5, column=0, columnspan=8, sticky="we", pady=(8, 4))
        self._calib_params()

        ttk.Button(box, text="运行标定 / 复检", command=self._run_calibration).grid(
            row=6, column=0, columnspan=2, sticky="w", pady=(10, 2))
        ttk.Button(box, text="标定前置检查（两路是否都看到棋盘）",
                   command=self._run_precheck).grid(
            row=6, column=2, columnspan=3, sticky="w", pady=(10, 2))
        ttk.Button(box, text="刷新并检测相机（枚举 + 实测能否出图）",
                   command=self._refresh_cameras).grid(
            row=6, column=5, columnspan=3, sticky="w", pady=(10, 2))
        ttk.Button(box, text="相机探测（核对两路参数是否一致）",
                   command=self._run_probe).grid(
            row=7, column=0, columnspan=3, sticky="w", pady=(2, 2))
        ttk.Button(box, text="相机设备管理（识别 / 固定 / 标无关）",
                   command=self._run_cameras_identify).grid(
            row=7, column=3, columnspan=2, sticky="w", pady=(2, 2))
        ttk.Button(box, text="「无关」相机清单（自动挑选会跳过）",
                   command=self._run_cameras_ignored).grid(
            row=8, column=0, columnspan=3, sticky="w", pady=(2, 2))
        ttk.Button(box, text="清空「无关」标记", command=self._run_cameras_unignore_all).grid(
            row=8, column=3, columnspan=2, sticky="w", pady=(2, 2))
        # ---- 90° 装置专用标定入口（P0-3）----
        # 为什么要有它：90° 装置与平行装置**走的是同一条标定命令**，差别只在
        # `--stereo-mode angled --skip-fov`。下拉框里选 angled 也能做到，但
        # "选错下拉"没有反馈、且 FOV 一步会白跑；这里把**前提条件**先讲清。
        ttk.Button(box, text="标定（90° 装置，跳过 FOV）",
                   command=self._run_calibration_90deg).grid(
            row=9, column=0, columnspan=3, sticky="w", pady=(6, 2))
        # 画质诊断（P1-3）：以前这个工具**只有命令行能用**，界面上完全没有入口，
        # 而"有一路画面明显更糊/更卡"恰恰是最常见的现场问题。
        ttk.Button(box, text="画质诊断（两路同开，区分带宽 / 光学）",
                   command=self._run_camera_quality).grid(
            row=9, column=3, columnspan=4, sticky="w", pady=(6, 2))
        # ---- 平行装置的「本次不做 FOV 对齐」出口（2026-10-24 加）----
        # 为什么加：FOV 估计失败时脚本会**降级继续**（不裁剪、按原始画面标定），
        # 但用户仍需要一个**主动**的选择 —— 尤其两路视场差异小、或共同纹理太少时，
        # 与其让它在启动帧上白跑一次匹配，不如直接说明"这次不做对齐"。
        # ⚠️ 它**只加 --skip-fov**，不加 `--stereo-mode angled`：模式与 FOV 是两件事，
        # 平行装置不该被这个勾选偷偷改成大角度。
        self.calib_skip_fov = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            box, variable=self.calib_skip_fov,
            text="本次不做 FOV 对齐（标定按两路各自原始画面）").grid(
            row=10, column=0, columnspan=5, sticky="w", pady=(4, 0))
        self.calib_skip_fov_note = ttk.Label(
            box, style="Hint.TLabel", wraplength=CARD_HINT_WRAP,
            text="后果：两路各自原始画面；适合两路视场差异小、或有共同纹理太少的情况。"
                 "两路像素尺度若不一致，视差/深度会有系统偏差 —— 勾了就在标定后复检极线误差。"
                 "（大角度装置请用上面那个『90° 装置』入口，那是另一件事。）")
        self.calib_skip_fov_note.grid(row=11, column=0, columnspan=9,
                                      sticky="w", pady=(0, 2))
        self._wrapped.append((self.calib_skip_fov_note, "card"))
        self.calib90_note = ttk.Label(
            box, style="Hint.TLabel", wraplength=CARD_HINT_WRAP,
            text="90° 装置：两路各用原始画面，不做 FOV 裁剪对齐；坐标系为 "
                 "left_camera。前提是两路有共同可见区（棋盘要同时出现在左右画面里），"
                 "且基线 B 应约为工作距离的 2 倍（B≈2·D）。不满足就别用这个大角度入口。")
        self.calib90_note.grid(row=12, column=0, columnspan=9, sticky="w", pady=(0, 2))
        self._wrapped.append((self.calib90_note, "card"))
        # ---- 外参方法（2026-10-26 加）----
        # 为什么要有：以前界面根本不传 `--method`，也就是**只有 joint 一条路**，
        # 而 90° 装置要的 chain 藏在下面的「单板链式标定」卡片里 —— 用户看不到
        # "这次到底用了哪种解法"，两个入口的说法也各写一份。现在选择点只有这里一个，
        # 选项与默认值都来自注册表（`calib/methods.py`）。
        ttk.Label(box, text="外参方法").grid(row=13, column=0, sticky="e", pady=(8, 0))
        self.calib_method = ttk.Combobox(
            box, values=CALIB_METHODS, width=10, state="readonly")
        self.calib_method.set(DEFAULT_CALIB_METHOD)
        self.calib_method.grid(row=13, column=1, sticky="w",
                               padx=(4, 12), pady=(8, 0))
        self.calib_method_note = ttk.Label(
            box, style="Hint.TLabel", wraplength=CARD_HINT_WRAP,
            text="；".join(f"{d['name']}={d['summary']}" for d in _plugin_describe("calib_method"))
                 + "。（chain 要走下面的『单板链式标定』：需要三段离线图，实时采集不支持）")
        self.calib_method_note.grid(row=14, column=0, columnspan=9,
                                    sticky="w", pady=(0, 2))
        self._wrapped.append((self.calib_method_note, "card"))
        self._build_chain_block()

    def _build_chain_block(self) -> None:
        """单板链式标定：三段采集（参考段双路同拍）+ 一条标定命令的引导区（第三条路）。

        为什么单独一张卡片而不是继续往第 1 步里塞：这一块是**另一套前提**
        （两路几乎无共视），和 joint / 90° 不是同一件事；混在一张卡片里，
        用户会以为它是"标定失败后重试的按钮"。卡片标题旁先写清三者区别。
        """
        box = self._card(
            "单板链式标定（两路几乎无共视 / 90° 大角度）",
            "三条路怎么选：joint = 两路有充分共视；90° = 共视很少但能与同一块板同框；"
            "单板链式 = 两路几乎看不到同一区域，靠“板不动”的参考图定外参。\n"
            "内参段板可以移动；参考段板必须固定在两路都能看全整块的位置，"
            "一次双路同拍，左右之间不会再有时间差。\n"
            "流程：按顺序拍完三段图（停稳就自动存，不用按键），再点最下面的"
            "『开始链式标定』。采集窗口里按 q 可提前结束，已拍的图会保留。")
        body = ttk.Frame(box)
        body.grid(row=1, column=0, columnspan=9, sticky="we", pady=(2, 4))

        self.chain_root = self._param_path(
            body, "工作目录",
            "三段图分别存到 <工作目录>\\left / right / ref_left / ref_right"
            "（参考段一次写两个目录）；某一段拍砸了可以直接重拍那一段"
            "（新图会追加进去）。", True)
        self.chain_root.delete(0, "end")
        self.chain_root.insert(0, str(Path("outputs") / "chain"
                                      / time.strftime("%Y%m%d-%H%M%S")))

        self.chain_count_labels: dict[str, ttk.Label] = {}
        for key, _side, target in CHAIN_STAGES:
            row = self._next_row(body)
            title, tip = CHAIN_STAGE_TEXT[key]
            lab = ttk.Label(body, text=f"{title}（目标 {target} 张）：{tip}",
                            wraplength=PARAM_HINT_WRAP)
            lab.grid(row=row, column=0, columnspan=3, sticky="w", pady=(4, 0))
            self._wrapped.append((lab, "param"))
            ttk.Button(body, text=title, width=18,
                       command=lambda k=key: self._chain_capture(k)).grid(
                row=row, column=3, sticky="w", padx=(8, 0), pady=(4, 0))
            cnt = ttk.Label(body, text="0 张", style="Hint.TLabel")
            cnt.grid(row=row, column=4, sticky="w", padx=(10, 0), pady=(4, 0))
            self.chain_count_labels[key] = cnt

        row = self._next_row(body)
        ttk.Button(body, text="刷新张数", width=10,
                   command=self._chain_refresh).grid(
            row=row, column=0, sticky="w", pady=(10, 2))
        # 四个目录都 ≥1 张才可点；不足时把**缺什么**写在下面的状态行里。
        self.chain_run_btn = ttk.Button(
            body, text="开始链式标定", command=self._chain_run)
        self.chain_run_btn.grid(row=row, column=3, columnspan=2,
                                sticky="w", pady=(10, 2))
        self.chain_status = ttk.Label(body, text="", style="Hint.TLabel",
                                      wraplength=PARAM_HINT_WRAP)
        self.chain_status.grid(row=self._next_row(body), column=0, columnspan=5,
                               sticky="w", pady=(0, 2))
        self._wrapped.append((self.chain_status, "param"))
        self._chain_refresh()

    def _chain_root_path(self) -> Path:
        """工作目录 → 绝对路径（相对路径按项目根解析，因为子进程 cwd 是 ROOT）。"""
        raw = self.chain_root.get().strip()
        p = Path(raw) if raw else Path("outputs") / "chain"
        return p if p.is_absolute() else ROOT / p

    def _chain_counts(self) -> dict[str, int]:
        root = self._chain_root_path()
        return {key: _count_shots(root / key) for key in CHAIN_REQUIRED_DIRS}

    def _chain_refresh(self) -> None:
        """重数各样目录，刷新计数标签 / 按钮可用性 / 缺图提示。"""
        counts = self._chain_counts()
        for key, _side, _target in CHAIN_STAGES:
            dirs = CHAIN_STAGE_DIRS[key]
            if len(dirs) == 1:
                text = f"{counts[dirs[0]]} 张"
            else:
                # 参考段两个目录分别数：`参考 左 3 / 右 3`
                text = "参考 左 {} / 右 {}".format(
                    counts["ref_left"], counts["ref_right"])
            self.chain_count_labels[key].configure(text=text)
        ready = _chain_ready(counts)
        self.chain_run_btn.configure(state="normal" if ready else "disabled")
        self.chain_status.configure(text=_chain_status_text(counts))

    def _chain_capture(self, stage: str) -> None:
        """拍一段（走 `_run`：可中断、有日志，不裸调 subprocess）。"""
        root = self._chain_root_path()
        dirs = CHAIN_STAGE_DIRS[stage]
        for name in dirs:
            try:
                (root / name).mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                messagebox.showwarning("目录建不了",
                                       f"没法创建 {root / name}：{exc}\n"
                                       "换个有写权限的工作目录再试。")
                return
        _sub, side, target = next(s for s in CHAIN_STAGES if s[0] == stage)
        cols = self._read_int(self.calib_cols, "内角点·列", 3, 30)
        if cols is None:
            return
        rows = self._read_int(self.calib_rows, "内角点·行", 3, 30)
        if rows is None:
            return
        square = self._read_int(self.calib_square, "方格边长 mm", 5, 500)
        if square is None:
            return
        # 参考段是双路同拍，没有"这一路用哪个相机号"的问题，两个都传不进去。
        camera = None
        if stage != "ref":
            camera = _cam_arg(self.calib_left.get() if side == "left"
                              else self.calib_right.get())
        width, height = self.calib_res.get().split("x")
        fps = self.calib_fps.get().strip() or CALIB_FPS_DEFAULT
        cmd = chain_capture_cmd(
            root, stage, camera=camera, width=width, height=height, fps=fps,
            backend=self.calib_backend.get(), cols=cols, rows=rows, square=square)
        where = " + ".join(str(root / d) for d in dirs)
        self._append(f"[链式标定] {CHAIN_STAGE_TEXT[stage][0]}：目标 {target} 张 → "
                     f"{where}（自动连拍，停稳即存）")
        self._run(cmd, f"链式标定 · {CHAIN_STAGE_TEXT[stage][0]}"
                       f"（{side}，目标 {target} 张，自动连拍）",
                  on_done=lambda t, k=stage: self._after_chain_capture(t, k))

    def _after_chain_capture(self, task: Task, stage: str) -> None:
        """一段拍完：刷新张数；提前退出（3）也算正常，失败要说下一步。"""
        self._chain_refresh()
        got = sum(_count_shots(self._chain_root_path() / d)
                  for d in CHAIN_STAGE_DIRS[stage])
        name = CHAIN_STAGE_TEXT[stage][0]
        if task.exit_code == 0:
            self._append(f"[链式标定] {name}完成，目录里共 {got} 张。")
        elif task.exit_code == 3:
            self._append(f"[链式标定] {name}提前结束（用户退出），已留下 {got} 张；"
                         "够用的话可以直接进下一段。")
        elif task.exit_code is None:
            return
        else:
            self._append(
                f"[链式标定] {name}失败（返回码 {task.exit_code}）。"
                f"日志：{task.log_path or '见上方输出'}。"
                "下一步：确认相机没被别的程序占用（先中断其他任务），"
                "或在『基本设置』里换一路相机；仍打不开就跑『相机探测』。")

    def _chain_run(self) -> None:
        """四目录齐了才拼链式标定命令；板参数/分辨率/后端全部沿用同一份面板值。"""
        counts = self._chain_counts()
        if not _chain_ready(counts):
            messagebox.showwarning("还缺图",
                                   "还不能标定：缺 " + _chain_missing(counts)
                                   + "。先把三段都拍上，再点这个按钮。")
            self._chain_refresh()
            return
        cols = self._read_int(self.calib_cols, "内角点·列", 3, 30)
        if cols is None:
            return
        rows = self._read_int(self.calib_rows, "内角点·行", 3, 30)
        if rows is None:
            return
        square = self._read_int(self.calib_square, "方格边长 mm", 5, 500)
        if square is None:
            return
        target = self._read_int(self.calib_target, "目标对数", 4, 200)
        if target is None:
            return
        minp = self._read_int(self.calib_minpairs, "最少对数", 3, 200)
        if minp is None:
            return
        width, height = self.calib_res.get().split("x")
        fps = self.calib_fps.get().strip() or CALIB_FPS_DEFAULT
        cmd = chain_calibrate_cmd(
            self._chain_root_path(), cols=cols, rows=rows, square=square,
            width=width, height=height, fps=fps,
            backend=self.calib_backend.get(), target=target, min_pairs=minp)
        self._run(cmd, "单板链式标定（chain / same-board）",
                  on_done=self._after_chain_calib)

    def _after_chain_calib(self, task: Task) -> None:
        """标定结束：成功就读标定 JSON 刷状态条（不在这里算基线/夹角）。"""
        self._chain_refresh()
        if task.exit_code != 0:
            self._append(
                f"[链式标定] 失败（返回码 {task.exit_code}）。"
                f"日志：{task.log_path or '见上方输出'}。"
                "下一步：先确认两张参考图是板没动过的（位置/角度一致），"
                "再检查每一段是否都拍到完整棋盘；改完补拍那一段重试。")
            return
        self._refresh_profile_line()
        measure = calibration_measure()
        detail = (f"基线 {measure[0]:.1f} mm / 夹角 {measure[1]:.1f}°"
                  if measure is not None else "读不到 config/stereo_calibration.json")
        self._append(f"[链式标定] 完成：{detail}。")
        messagebox.showinfo(
            "已更新标定",
            f"链式标定完成，{detail}。\n\n"
            "建议先跑『相机检查 / 复检标定』复核装置有没有被碰动："
            "第 1 步选『复检标定』→『运行标定 / 复检』。")

    # ---------- 第 2 步：参数 ----------
    def _build_step2(self) -> None:
        box = self._card(
            "第 2 步 · 检测与测距参数",
            "检测方法/颜色预设对实时、离线视频、双目会话都生效。"
            "球直径/预测视野/焦距只用于单目；双目深度来自标定，不读这些值。")
        ttk.Label(box, text="检测方法").grid(row=1, column=0, sticky="e", pady=4)
        self.method = ttk.Combobox(box, values=METHODS, width=10, state="readonly")
        self.method.set(DEFAULT_METHOD)      # 注册表声明的默认值（= 命令行/文档同一个）
        self.method.grid(row=1, column=1, sticky="w", padx=(4, 20), pady=4)

        ttk.Label(box, text="颜色预设").grid(row=1, column=2, sticky="e", pady=4)
        self.preset = ttk.Combobox(box, values=PRESETS, width=8, state="readonly")
        self.preset.set("不使用")
        self.preset.grid(row=1, column=3, sticky="w", padx=(4, 4), pady=4)

        self.step2_scope = ttk.Label(
            box, text="常用：单目实时 / 离线视频 / 双目会话都会读取",
            style="Hint.TLabel", wraplength=CARD_HINT_WRAP)
        self.step2_scope.grid(row=2, column=0, columnspan=6, sticky="w", pady=(0, 2))
        self._wrapped.append((self.step2_scope, "card"))

        from apps._collapsible import CollapsibleSection
        self._step2_advanced = CollapsibleSection(
            box, "高级（仅单目实时 / 离线视频）", expanded=False, padding=(8, 4))
        self._step2_advanced.grid(row=3, column=0, columnspan=6, sticky="we",
                                  pady=(4, 2))
        mono = self._step2_advanced.content

        ttk.Label(mono, text="摄像头编号").grid(row=0, column=0, sticky="e", pady=3)
        self.camera = ttk.Combobox(mono, values=MONO_CAMERAS, width=8,
                                   state="readonly")
        self.camera.set("auto")
        self.camera.grid(row=0, column=1, sticky="w", padx=(4, 8), pady=3)
        ttk.Label(mono, text="仅实时", style="Hint.TLabel").grid(
            row=0, column=2, sticky="w")

        ttk.Label(mono, text="球直径(米)").grid(row=1, column=0, sticky="e", pady=3)
        self.diameter = ttk.Entry(mono, width=8)
        self.diameter.insert(0, "0.04")
        self.diameter.grid(row=1, column=1, sticky="w", padx=(4, 8), pady=3)
        ttk.Label(mono, text="仅单目三维", style="Hint.TLabel").grid(
            row=1, column=2, sticky="w")

        ttk.Label(mono, text="预测视野(秒)").grid(row=2, column=0, sticky="e", pady=3)
        self.horizon = ttk.Entry(mono, width=8)
        self.horizon.insert(0, "0.6")
        self.horizon.grid(row=2, column=1, sticky="w", padx=(4, 8), pady=3)
        ttk.Label(mono, text="仅单目预测", style="Hint.TLabel").grid(
            row=2, column=2, sticky="w")

        ttk.Label(mono, text="焦距fx(留空自动)").grid(row=3, column=0, sticky="e", pady=3)
        self.focal = ttk.Entry(mono, width=10)
        self.focal.grid(row=3, column=1, sticky="w", padx=(4, 8), pady=3)
        ttk.Label(mono, text="仅单目测距", style="Hint.TLabel").grid(
            row=3, column=2, sticky="w")

        self.show_3d = tk.BooleanVar(value=True)
        ttk.Checkbutton(mono, text="启用三维估计（需填球直径）",
                        variable=self.show_3d).grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Label(mono, text="仅单目；双目始终输出标定三维",
                  style="Hint.TLabel").grid(row=5, column=0, columnspan=3,
                                            sticky="w")

        ttk.Button(box, text="这些参数用在哪？", command=self._show_param_scope).grid(
            row=4, column=0, sticky="w", pady=(4, 2))

    def _show_param_scope(self) -> None:
        messagebox.showinfo(
            "参数用在哪",
            "检测方法、颜色预设：\n"
            "  实时摄像头 / 离线视频 / 双目会话都用。\n\n"
            "摄像头编号：仅实时摄像头。\n"
            "球直径：仅单目三维估计；双目深度来自标定。\n"
            "预测视野：仅单目预测；双目预测已关闭。\n"
            "焦距 fx：仅单目测距；双目用标定内参。\n"
            "启用三维估计：仅单目；双目始终输出三维。",
            parent=self)

    # ---------- 第 3 步：开始 ----------
    def _build_step3(self) -> None:
        box = self._card("第 3 步 · 开始轨迹检测",
                         "实时检测会弹出摄像头窗口；窗口内按 q 退出、r 重置轨迹、c 切换方法。")
        ttk.Button(box, text="启动实时摄像头检测", style="Go.TButton",
                   command=self._start_realtime).grid(row=1, column=0, padx=(0, 12), pady=4)
        ttk.Button(box, text="打开离线视频", style="Go.TButton",
                   command=self._start_video).grid(row=1, column=1, padx=(0, 12), pady=4)
        ttk.Button(box, text="录制双目（无损）", command=self._start_record).grid(
            row=2, column=0, padx=(0, 12), pady=4)
        # 离线双目 / 点选球**只在「双目工作台」有一处入口**（P1-1）。
        # 原来这两个按钮在工作台（②③）和第 3 步各有一份，两处行为一致但
        # 一个带会话选择器、一个不带 —— 用户不知道点哪个才是"正确路径"。
        # 现在这里只做**跳转**，不放第二份实现。
        ttk.Button(box, text="离线双目 / 点选球 → 双目工作台（①–⑤）",
                   command=lambda: self._scroll_to_card("双目工作台")).grid(
            row=2, column=1, columnspan=2, padx=(0, 12), pady=4, sticky="w")
        ttk.Button(box, text="快测（十几秒）", command=self._run_tests_fast).grid(
            row=1, column=2, padx=(0, 6), pady=4)
        ttk.Button(box, text="全套回归（7 分钟）", command=self._run_tests_full).grid(
            row=1, column=3, padx=(0, 6), pady=4)
        ttk.Button(box, text="规范自查", command=self._run_standards).grid(
            row=1, column=4, pady=4)
        self.status = ttk.Label(box, text="就绪", style="Hint.TLabel")
        self.status.grid(row=3, column=0, columnspan=5, sticky="w", padx=2)

    # ---------- 双目工作台 ----------
    def _build_workbench(self) -> None:
        """把录制→校验→seed→三维→查看器串成一条可见的路径。"""
        box = self._card(
            "双目工作台",
            "选一条会话，按 ①→⑤ 做（校验 → 点选 → 离线双目 → 3D+视频 → 导出）。"
            "校验/离线/查看器都走「任务」管理器，可单独中断。"
            "删除/清理在右上『数据管理』里做，本页不再有第二个入口。")
        ttk.Label(box, text="最近会话").grid(row=1, column=0, sticky="e", pady=4)
        self.wb_var = tk.StringVar()
        self.wb_combo = ttk.Combobox(box, textvariable=self.wb_var, width=32,
                                     state="readonly")
        self.wb_combo.grid(row=1, column=1, columnspan=3, sticky="w", padx=(4, 8), pady=4)
        self.wb_combo.bind("<<ComboboxSelected>>", lambda _e: self._wb_show())
        ttk.Button(box, text="刷新", command=self._wb_refresh, width=6).grid(
            row=1, column=4, sticky="w", pady=4)
        ttk.Button(box, text="浏览会话…", command=self._wb_browse, width=11).grid(
            row=1, column=5, sticky="w", padx=(4, 0), pady=4)
        ttk.Button(box, text="清理（在数据管理里）", command=self._wb_clean,
                   width=13).grid(
            row=1, column=6, sticky="w", padx=(4, 0), pady=4)

        self.wb_info = ttk.Label(box, text="还没有会话", style="Hint.TLabel",
                                 justify="left", wraplength=CARD_HINT_WRAP)
        self.wb_info.grid(row=2, column=0, columnspan=6, sticky="w", pady=(2, 6))
        self._wrapped.append((self.wb_info, "card"))

        self._wb_buttons: dict[str, ttk.Button] = {}
        specs = [
            ("check", "① 校验数据", self._wb_check),
            ("seed", "② 点选球心", self._wb_seed),
            ("run", "③ 跑离线双目", self._wb_run),
            ("view", "④ 3D + 视频", self._wb_view),
            ("export", "⑤ 导出 JSON", self._wb_export),
            ("open", "打开会话目录", self._wb_open),
            ("files", "文件…", self._wb_files),
        ]
        for i, (key, text, cmd) in enumerate(specs):
            btn = ttk.Button(box, text=text, command=cmd, width=13)
            btn.grid(row=3, column=i, padx=(0, 6), pady=4, sticky="w")
            self._wb_buttons[key] = btn

        self.wb_canary = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            box, text="时间戳配对（默认；失败自动回退旧方案）",
            variable=self.wb_canary).grid(row=4, column=0, columnspan=6,
                                         sticky="w", pady=(0, 2))

        self.wb_hint = ttk.Label(box, text="", style="Hint.TLabel",
                                 wraplength=CARD_HINT_WRAP)
        self.wb_hint.grid(row=5, column=0, columnspan=6, sticky="w", pady=(2, 0))
        self._wrapped.append((self.wb_hint, "card"))
        self._wb_paths: dict[str, Path] = {}
        self._wb_custom: dict[Path, bool] = {}
        self._wb_last_first = ""
        self.after(300, self._wb_refresh)
        self.after(2500, self._wb_tick)

    def _wb_selected(self):
        from apps._workbench import summarize_session
        path = self._wb_paths.get(self.wb_var.get())
        return summarize_session(path) if path else None

    def _wb_browse(self) -> None:
        path = filedialog.askdirectory(
            parent=self, title="选择任意双目会话目录",
            initialdir=str(ROOT / "data" / "sessions"))
        if not path:
            return
        self._wb_add_custom(Path(path))

    def _wb_add_custom(self, path: Path) -> None:
        p = Path(path).resolve()
        if not (p / "session.json").is_file():
            messagebox.showerror(
                "会话", f"这个目录没有 session.json：\n{p}", parent=self)
            return
        self._wb_custom[p] = True
        self._wb_refresh()
        self._wb_select_path(p)
        self._append(f"[工作台] 已加入任意会话：{p}")

    def _wb_clean(self) -> None:
        """删除/清理**只有一个入口**：『数据管理』（P1-1）。

        原来这里的『清理…』直接开 `CleanupDialog`，而顶部『数据管理』是另一个
        窗口、也带清理 —— 同一件事两个窗口、两套勾选方式，用户不知道该信哪个。
        现在统一到数据管理：它的『清理…』用的是同一个 `CleanupDialog`，
        并且多了回收站 / 恢复。
        """
        if self._wb_selected() is not None:
            self._append("[工作台] 删除/清理已统一到『数据管理』："
                         "在里面勾选会话后点『清理…』（可恢复，见『回收站…』）。")
        self._open_data_manager()

    def _wb_select_path(self, path: Path) -> None:
        target = Path(path).resolve()
        for label, p in self._wb_paths.items():
            if p.resolve() == target:
                self.wb_var.set(label)
                self._wb_show()
                return

    def _wb_refresh(self, select_latest: bool = False) -> None:
        from apps._workbench import scan_sessions, summarize_session
        try:
            scanned = scan_sessions(ROOT / "data" / "sessions", limit=200)
        except Exception as exc:                            # noqa: BLE001
            # 只说"失败：<异常>"等于让用户自己猜。现象 + 可能原因 + 下一步。
            self.wb_info.configure(
                text=f"扫描会话失败：{exc}\n"
                     "可能原因：data/sessions 目录不存在、没有读取权限，"
                     "或某个会话的 session.json 损坏。\n"
                     "下一步：点「刷新」重试；仍失败就用右上『数据管理』打开该目录，"
                     "看是哪条会话读不出来（日志里有完整异常）。")
            self._append(f"[工作台] 扫描会话失败：{exc}")
            return
        scanned_paths = {s.path.resolve() for s in scanned}
        custom_paths = [p for p in self._wb_custom if p.exists()]
        self._wb_custom = {p: True for p in custom_paths}
        custom = [summarize_session(p) for p in custom_paths
                  if p.resolve() not in scanned_paths]
        summaries = scanned + custom
        labels: list[str] = []
        for idx, s in enumerate(summaries):
            label = s.label
            if s.path.resolve() in set(custom_paths):
                label += "  [任意]"
            if label in labels:
                label = f"{label}  [{idx}]"
            labels.append(label)
        old_label = self.wb_var.get()
        old_path = self._wb_paths.get(old_label)
        old_id = old_path.name if old_path is not None else ""
        self._wb_paths = {lab: s.path for lab, s in zip(labels, summaries)}
        self.wb_combo.configure(values=labels)
        first = summaries[0] if summaries else None
        selected = None
        if select_latest or old_path is None or old_path.resolve() not in {
                s.path.resolve() for s in summaries}:
            selected = first
        elif first is not None and first.id != self._wb_last_first and old_id == self._wb_last_first:
            # 刚录完/刚出现新会话：原来停在最新那条，就跟着切到新的
            selected = first
        else:
            selected = next((s for s in summaries
                             if s.path.resolve() == old_path.resolve()), first)
        if selected is not None:
            # 用 id 保留选择；状态标记（◐/✓/✗）变化时 label 会变，不能拿 label 当主键
            for lab, s in zip(labels, summaries):
                if s.path.resolve() == selected.path.resolve():
                    self.wb_var.set(lab)
                    break
        self._wb_last_first = first.id if first is not None else ""
        self._wb_show()

    def _wb_tick(self) -> None:
        # 任务跑完后按钮/状态要自己更新；不依赖用户手动点刷新
        try:
            self._wb_refresh()
        finally:
            self.after(2500, self._wb_tick)

    def _after_record_task(self, task: Task) -> None:
        """录制结束 -> 只对新目录自动校验（P2）。"""
        if task.exit_code != 0:
            self._append(
                f"[工作台] 录制失败（返回码 {task.exit_code}）。"
                "可能原因：相机被占用 / 标定分辨率与请求不一致 / 左或右相机没出图。"
                "下一步：点『第 1 步』的『刷新并检测相机』确认识别，"
                "或在『装置档案』里改用 90° 模式；详细报错见上方日志。"
                "（不自动校验旧会话）")
            self._wb_refresh()
            return
        from apps._workbench import scan_sessions
        now = scan_sessions(ROOT / "data" / "sessions", limit=50)
        new = [s for s in now if s.path.name not in self._record_before]
        if not new:
            self._append("[工作台] 录制结束但没有发现新会话目录；跳过自动校验。")
            self._wb_refresh()
            return
        target = new[0]
        self._wb_refresh(select_latest=False)
        self._wb_select_id(target.id)
        if self._wb_auto_after:
            try:
                self.after_cancel(self._wb_auto_after)
            except Exception:                               # noqa: BLE001
                pass
        self._wb_auto_after = self.after(
            800, lambda sid=target.id: self._wb_auto_check(sid))

    def _wb_select_id(self, sid: str) -> None:
        for label, path in self._wb_paths.items():
            if path.name == sid:
                self.wb_var.set(label)
                self._wb_show()
                return

    def _wb_auto_check(self, sid: str) -> None:
        self._wb_auto_after = None
        self._wb_select_id(sid)
        s = self._wb_selected()
        if s is not None and s.id == sid and s.can_check:
            self._append(f"[工作台] 录制完成，自动校验 {sid}。")
            self._wb_check(sid)

    def _wb_show(self) -> None:
        s = self._wb_selected()
        if s is None:
            self.wb_info.configure(text="还没有会话。先在「第 3 步」录制一条双目。")
            for b in getattr(self, "_wb_buttons", {}).values():
                b.configure(state="disabled")
            return
        means = ""
        if s.mean_left is not None and s.mean_right is not None:
            means = f"  亮度 {s.mean_left:.0f}/{s.mean_right:.0f}"
        traj = "—"
        if s.has_traj:
            traj = f"{s.n_matched}/{s.n_samples}"
        z = ""
        if s.z_min is not None and s.z_max is not None:
            z = f"  Z {s.z_min:.3f}–{s.z_max:.3f} m"
        res = f"  残差中位 {s.residual_median:.2f} px" if s.residual_median is not None else ""
        problems = ("\n问题：" + "；".join(s.problems)) if s.problems else ""
        checked = self._wb_check_result.get(s.id)
        check_text = "—" if checked is None else ("✓" if checked else "✗")
        self.wb_info.configure(
            text=(f"会话 {s.id}  {s.created}\n"
                  f"帧数 {s.frames_left}/{s.frames_right}  "
                  f"sync {s.sync_method}{s.sync_offset:+d} (置信度 {s.sync_confidence:.2f})"
                  f"{means}\n"
                  f"校验 {check_text}   seed {'✓' if s.has_seeds else '—'}   "
                  f"轨迹 {traj}{z}{res}{problems}"))
        next_step = {
            "bad": "先按上面的问题重录/修数据；坏数据不要进三维。",
            "recorded": ("校验未通过，按问题重录。" if checked is False
                         else "下一步：① 校验数据。" if checked is None
                         else "下一步：② 点选球心。"),
            "seeded": "下一步：③ 跑离线双目。",
            "traj": "下一步：④ 打开 3D 查看器；⑤ 可重新导出标准 JSON。",
        }.get(s.status, "")
        if s.sync_method == "none" and s.status != "bad":
            next_step = "未同步：优先重录并闪光；三维结果只作参考。"
        self.wb_hint.configure(text=next_step)

        specular = self.method.get() == "specular"
        from apps._workbench import button_states
        states = button_states(s, specular, checked)
        if s.id in self._wb_check_inflight:
            states["check"] = False
        for key, enabled in states.items():
            self._wb_buttons[key].configure(state="normal" if enabled else "disabled")

    def _wb_check(self, sid: str | None = None) -> None:
        if sid is None:
            s = self._wb_selected()
        else:
            s = next((x for x in (self._wb_selected(),) if x and x.id == sid), None)
            if s is None:
                self._wb_select_id(sid)
                s = self._wb_selected()
        if s is None or not s.can_check or s.id in self._wb_check_inflight:
            return
        if self._wb_auto_after:
            try:
                self.after_cancel(self._wb_auto_after)
            except Exception:                               # noqa: BLE001
                pass
            self._wb_auto_after = None
        self._wb_check_inflight.add(s.id)
        self._wb_show()
        self._run([PY, str(ROOT / "apps" / "tools" / "check_dataset.py"),
                   str(s.path), "--strict"], f"校验数据 {s.id}",
                  on_done=lambda task, xid=s.id: self._wb_check_done(task, xid))

    def _wb_check_done(self, task: Task, sid: str) -> None:
        self._wb_check_inflight.discard(sid)
        ok = task.exit_code == 0
        self._wb_check_result[sid] = ok
        self._append(f"[工作台] 校验 {sid}：{'通过' if ok else f'失败（码 {task.exit_code}）'}")
        s = self._wb_selected()
        if s is not None and s.id == sid:
            self._wb_show()

    def _wb_seed(self) -> None:
        s = self._wb_selected()
        if s is None:
            return
        # 点选球心属于金属球路径：③ 会用 specular 跑；但**不许改第 2 步的下拉框**
        # —— 那是用户填过的输入。有日志不等于有交代（B-98）。
        self._append(f"[工作台] 点选球心属于金属球路径：③ 会用 specular 运行，"
                     f"未改你的检测方法（第 2 步仍是 {self.method.get()}）。")
        self._pick_stereo_seed(str(s.path), offer_run=False, method="specular")

    def _wb_run(self) -> None:
        s = self._wb_selected()
        if s is None:
            return
        # 会话里已有 seeds.json 就必须用 specular 跑；但**不改第 2 步的下拉框**：
        # 改成"本次运行的本地覆盖 + 日志说明"（B-98）。
        if s.has_seeds:
            method = "specular"
            self._append(f"[工作台] 本次用 specular 运行（会话已有点选球心），"
                         f"未改你的检测方法（第 2 步仍是 {self.method.get()}）。")
        else:
            method = self.method.get()
            if method == "specular":
                messagebox.showinfo("跑离线双目", "还没有点选球心；先点『② 点选球心』。")
                return
        args = self._offline_stereo_args(str(s.path), method=method)
        args += ["--pairing", "canary" if self.wb_canary.get() else "index"]
        self._run(args, f"离线双目 {s.id}")
        self.after(1500, self._wb_refresh)

    def _wb_view(self) -> None:
        s = self._wb_selected()
        if s is None:
            return
        if not s.can_view:
            messagebox.showinfo("打开 3D", "这条会话还没有可用的三维轨迹。")
            return
        self._run([PY, str(ROOT / "apps" / "view_3d.py"), str(s.path)],
                  f"3D 查看器 {s.id}")

    def _wb_export(self) -> None:
        s = self._wb_selected()
        if s is None:
            return
        if not s.has_traj:
            messagebox.showinfo("导出 JSON", "还没有 traj3d.json；先跑离线双目。")
            return
        self._run([PY, str(ROOT / "apps" / "export_visualization.py"),
                   str(s.path), "--plot"], f"导出标准数据 {s.id}")
        self.after(1500, self._wb_refresh)

    def _wb_open(self) -> None:
        s = self._wb_selected()
        if s is None or not s.path.exists():
            return
        try:
            os.startfile(str(s.path))                       # type: ignore[attr-defined]
        except Exception:                                   # noqa: BLE001
            subprocess.Popen(["explorer", str(s.path)])

    def _open_path(self, path: Path) -> None:
        if not path.exists():
            messagebox.showinfo("文件", f"文件不存在：\n{path}")
            return
        try:
            os.startfile(str(path))                         # type: ignore[attr-defined]
        except Exception:                                   # noqa: BLE001
            subprocess.Popen(["explorer", str(path)])

    def _wb_files(self) -> None:
        """P4：逐文件打开/复制路径。"""
        s = self._wb_selected()
        if s is None:
            return
        derived = s.path / "derived"
        menu = tk.Menu(self, tearoff=0)
        items = [
            ("打开 seeds.json", derived / "seeds.json"),
            ("打开 traj3d.json", derived / "traj3d.json"),
            ("打开 visualization.json", derived / "visualization.json"),
            ("打开 annotated.mp4", derived / "annotated.mp4"),
            ("打开 traj3d.png", derived / "traj3d.png"),
        ]
        for label, path in items:
            if path.exists():
                menu.add_command(label=label, command=lambda p=path: self._open_path(p))
            else:
                menu.add_command(label=label + "（无）", state="disabled")
        menu.add_separator()
        menu.add_command(label="复制会话路径",
                         command=lambda: (self.clipboard_clear(),
                                          self.clipboard_append(str(s.path))))
        menu.add_command(label="复制 seeds.json 路径",
                         command=lambda: (self.clipboard_clear(),
                                          self.clipboard_append(str(derived / "seeds.json"))))
        btn = self._wb_buttons.get("files")
        if btn is not None:
            try:
                menu.tk_popup(btn.winfo_rootx(), btn.winfo_rooty() + btn.winfo_height())
            finally:
                menu.grab_release()

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
    def _common_args(self, *, for_video: bool = False) -> list[str]:
        """实时/离线共用的参数。

        `for_video=True` 时不要追加 `run_video.py` 不认识的 `--display-width`
        和 `--no-3d`；这两个只属于实时窗口。否则“打开离线视频”会直接
        argparse 报 unrecognized arguments 起不来。

        ⚠️ **`--display-width` 必须在这里传下去**（2026-10-14 加）。
        它默认 960，实测把显示耗时从 17.7 ms 降到 1.8 ms ——
        是"画面卡顿"最大的单项。**命令行直接跑的人自动享受默认值，
        而从 GUI 走的人如果这里不传，就享受不到** —— 那正是
        "界面与命令行行为不一致"的老毛病（和之前 `--target` 没传下去同一类）。
        """
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
        if not for_video:
            if not self.show_3d.get():
                args += ["--no-3d"]
            args += ["--display-width", str(DISPLAY_WIDTH)]
        return args

    # ---------- 动作 ----------
    def _realtime_args(self) -> list[str]:
        """实时窗口的采集参数：分辨率取面板的「分辨率」，高分辨率时限制检测宽度。

        **不能像原来那样让 `run_realtime.py` 吃自己的 1280x720 默认值**：
        面板上写着 2560x1440、实时却按 720p 开，等于"界面能设的东西与真实取值脱节"。

        高分辨率（宽 > 1920）才加 `--detect-width`：4K/1440p 直接做检测会成倍变慢，
        降到 1280 与 `video_profiles.detection_width()` 的规则一致；1080p 及以下
        **不传**，避免把本来就不大的图再缩小。`display-width` 仍由 `_common_args()`
        负责（那是显示，不是检测）。
        """
        profile = parse_video_profile(self.calib_res.get(),
                                      default=DEFAULT_VIDEO_PROFILE)
        args = ["--width", str(profile.width), "--height", str(profile.height),
                "--fps", (self.calib_fps.get().strip() or CALIB_FPS_DEFAULT)]
        if profile.width > 1920:
            args += ["--detect-width", str(detection_width(profile.width, 1280))]
        return args

    def _start_realtime(self) -> None:
        if self.method.get() == "specular":
            messagebox.showinfo(
                "实时检测", "specular 需要先点选球心；请录制后用『离线双目（会话）』。")
            return
        cmd = [PY, str(ROOT / "apps" / "run_realtime.py"),
               "--camera", _cam_arg(self.camera.get()),
               "--method", self.method.get()]
        cmd += self._realtime_args()
        cfg = ROOT / "config" / "sphere_config.json"
        if cfg.exists():
            cmd += ["--config", str(cfg)]
        cmd += self._common_args()
        self._run(cmd, "实时摄像头检测")

    def _start_video(self) -> None:
        if self.method.get() == "specular":
            messagebox.showinfo(
                "离线视频", "specular 只支持双目会话，请用『离线双目（会话）』。")
            return
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
        cmd += self._common_args(for_video=True)
        self._append(f"结果视频将保存到：{out}")
        self._run(cmd, "离线视频分析")

    def _record_args(self, duration: str, label: str) -> list[str]:
        """GUI「录制双目」按钮 -> record_stereo.py 的命令行。"""
        width, height = self.calib_res.get().split("x")
        return [
            PY, str(ROOT / "apps" / "record_stereo.py"),
            "--left", self.calib_left.get(),
            "--right", self.calib_right.get(),
            "--width", width, "--height", height,
            "--fps", (self.calib_fps.get().strip() or CALIB_FPS_DEFAULT),
            "--duration", str(duration),
            "--label", (label or "gui_record").strip(),
            "--stereo-mode", self.stereo_mode_var.get(),
            "--preview", "--notes", "GUI 录制",
        ] + ([] if self.calib_backend.get() == "auto"
             else ["--backend", self.calib_backend.get()])

    def _start_record(self) -> None:
        if self.calib_left.get() == "auto" or self.calib_right.get() == "auto":
            messagebox.showwarning(
                "录制双目", "录制需要明确左/右相机，不能选 auto；请在「第 1 步」选 left / right。")
            return
        raw = simpledialog.askstring(
            "录制双目", "录多久（秒，1–600）？", initialvalue="10", parent=self)
        if raw is None:
            return
        try:
            duration = int(raw.strip())
            if not 1 <= duration <= 600:
                raise ValueError
        except ValueError:
            messagebox.showwarning("参数不对", f"时长要填 1–600 的整数，现在是「{raw}」。")
            return
        label = simpledialog.askstring(
            "录制双目", "标签（可空；会写进 session.json）", initialvalue="", parent=self)
        if label is None:
            return
        cmd = self._record_args(str(duration), label)
        self._append(f"录制：{duration} 秒；开始后闪一次光，结束前再闪一次。")
        from apps._workbench import scan_sessions
        self._record_before = {s.path.name for s in scan_sessions(ROOT / "data" / "sessions", limit=50)}
        self._run(cmd, f"录制双目（{duration} 秒）",
                  on_done=self._after_record_task)

    def _offline_stereo_args(self, session_dir: str, *,
                             method: str | None = None) -> list[str]:
        """拼 `apps/run_stereo_video.py` 的命令行。

        ``method`` 是**本次运行**的检测方法覆盖：工作台②/③ 读到 seeds.json 时
        必须用 specular 跑，但**不许动第 2 步的下拉框**（那是用户填过的输入，
        B-98）。省略时用面板当前值 —— 面板仍是唯一的默认来源。
        """
        used = method or self.method.get()
        args = [PY, str(ROOT / "apps" / "run_stereo_video.py"), session_dir,
                "--method", used]
        preset = self.preset.get()
        if preset and preset != "不使用":
            args += ["--preset", preset]
        if used == "specular":
            seed = Path(session_dir) / "derived" / "seeds.json"
            if seed.exists():
                args += ["--seed-json", str(seed)]
        mode = self.stereo_mode_var.get()
        if mode != "auto":
            args += ["--stereo-mode", mode]
        return args

    def _pick_stereo_seed(self, session_dir: str | None = None,
                          offer_run: bool = True, *,
                          method: str | None = None) -> bool:
        """打开点选窗口；保存成功返回 True。

        ``method``：点选完成后「现在运行？」那一步用的检测方法覆盖（有 seed 的
        会话必须 specular），**不改第 2 步的下拉框**（B-98）。
        """
        if session_dir is None:
            session_dir = filedialog.askdirectory(
                title="选择双目会话目录（含 session.json）",
                initialdir=str(ROOT / "data" / "sessions"))
        if not session_dir:
            return False
        if not (Path(session_dir) / "session.json").exists():
            messagebox.showwarning("不是会话目录", "这个目录里没有 session.json。")
            return False
        from apps.seed_picker import SeedPicker
        picker = SeedPicker(self, session_dir)
        self.wait_window(picker)
        if picker.result is None:
            return False
        self._append(f"seed 已保存：{picker.result}")
        if offer_run and messagebox.askyesno(
                "点选完成", "seed 已保存。现在运行离线双目三维？"):
            use = method or self.method.get()
            if use != self.method.get():
                self._append(f"[工作台] 本次用 {use} 运行（seed 已保存），"
                             f"未改你的检测方法（第 2 步仍是 {self.method.get()}）。")
            self._run(self._offline_stereo_args(str(session_dir), method=use),
                      "离线双目三维")
        return True

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
                "标定必须用两台确定的外接相机：用错相机或多台相机分不清时，"
                "标定结果会系统性错误，而结果表面上看是正常的。")
            return False

        for role in ("left", "right"):
            r = resolved.get(role)
            if r is None:
                continue
            for w in (r.warnings or ()):
                self._append(f"[相机] {w}")
            entry = _resolved_entry(r, by_order)
            name = entry.name if entry else ""
            # 采集索引 ≠ PnP 序号时**两个都写出来**：只写一个会被读成
            # `Web Camera Left（采集 #2）` 就是 `PnP #2 Integrated IR Camera`。
            if entry is not None and entry.order != r.index:
                where = f"{name or '未知名'}（PnP #{entry.order}；采集 #{r.index}）"
            else:
                where = f"{name or '未知名'}（编号 {r.index}）"
            if cr.is_avoided(name):
                return messagebox.askyesno(
                    "确认用内置 / 红外相机？",
                    f"{role}相机 {where}看起来是笔记本内置或红外相机，"
                    "不适合做双目：它没法固定在同一根横梁上，装置一动标定结果就作废。\n\n"
                    "双目请用外接相机。确定仍要继续吗？")

        # 两路分辨率必须一致 —— 不一致的双目标定没有意义，而且很难看出来
        sizes = {}
        for role in ("left", "right"):
            r = resolved.get(role)
            if r is None:
                continue
            e = _resolved_entry(r, by_order)
            sizes[role] = (e.name if e else "", r.index)
        if len({v[1] for v in sizes.values()}) < len(sizes):
            messagebox.showerror(
                "左右指向了同一台相机",
                f"左 -> #{sizes.get('left', ('?', '?'))[1]}，右 -> #{sizes.get('right', ('?', '?'))[1]}\n\n"
                "标定需要两台不同的相机。请跑 `启动程序.bat setup` 重新确认。")
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

    def _calibrate_args(self, *, angled_90: bool = False) -> list[str] | None:
        """把参数面板拼成 `calibrate_stereo.py` 的命令行。非法则返回 None。

        ⚠️ **这里曾经是硬编码的**（`--cols 8 --rows 5 --square 25`，且完全没有
        `--target`），于是从界面走**永远停在脚本默认的试标定档**，
        而用户以为在做正式标定。现在参数由面板给，面板的默认值就是正式档。

        ``angled_90=True`` 是「标定（90° 装置）」按钮用的：它**复用同一份拼装**
        （不复制第二套参数），只把模式强制成 ``angled`` 并补 ``--skip-fov``。
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
        sharp, ok2 = self._read_opt_float(
            self.calib_sharp, "额外绝对下限（focus）", 50.0, 200000.0)
        if not ok2:
            return None
        baseline, ok3 = self._read_opt_float(
            self.calib_baseline, "预期基线 mm", 5.0, 2000.0)
        if not ok3:
            return None

        width, height = self.calib_res.get().split("x")
        fps = self.calib_fps.get().strip() or CALIB_FPS_DEFAULT
        # 90° 装置：模式由按钮强制为 angled，FOV 对齐那一步跳过 —— 那一步
        # 假设两路有足够重叠区可以裁剪，而 90° 装置的画面对应关系不是这样。
        mode = "angled" if angled_90 else self.stereo_mode_var.get()
        # 「本次不做 FOV 对齐」只补 --skip-fov，**不改模式**（平行装置仍是 parallel）。
        skip_fov = angled_90 or bool(self.calib_skip_fov.get())
        args = ["--left", _cam_arg(self.calib_left.get()),
                "--right", _cam_arg(self.calib_right.get()),
                "--stereo-mode", mode,
                "--cols", str(cols), "--rows", str(rows), "--square", str(square),
                "--target", str(target), "--min-pairs", str(minp),
                "--width", width, "--height", height, "--fps", fps]
        # 外参方法：名字与"要不要 --same-board"都由注册表声明决定（不比较方法名）。
        # 默认（joint）不带 --same-board —— 那是 chain 的前置条件，多带会被脚本拒绝。
        plan = _calib_method_plan(self.calib_method.get() or DEFAULT_CALIB_METHOD)
        args += ["--method", plan.name]
        if plan.same_board:
            args += ["--same-board"]
        if skip_fov:
            args += ["--skip-fov"]
        # ⚠️ 2026-10-14：这里原来**写死** `--backend msmf`（理由是"MSMF 与设备枚举
        # 顺序一致"）。但那样界面就完全没有选择权，而本机实测 **MSMF 打开两路
        # 要 34.6 s、DSHOW 只要 11.8 s**，且两个后端给的分辨率完全一致
        # （idx0=1920x1080 / idx1=640x480，实测两轮）。所以改成由面板的
        # 「相机后端」决定；`auto` 时不传参数，仍然落到脚本默认（MSMF）——
        # **不替用户改行为，只把选择权交出来**。
        args += self._backend_args()
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

    def _backend_args(self) -> list[str]:
        """面板上的「相机后端」→ 命令行参数。`auto` 返回空（落到脚本默认）。

        **为什么要单独一个函数**：标定 / 复检 / 前置检查三条命令都要它。
        原来只有标定那条写死了 `--backend msmf`，另两条根本不传 ——
        **同一个决定散在三处，改一处忘两处**（本项目反复踩的那类问题）。
        现在只有一个决定点：面板。
        """
        backend = self.calib_backend.get()
        return ["--backend", backend] if backend and backend != "auto" else []

    def _run_calibration(self, *, force_angled: bool = False) -> None:
        """跑标定 / 复检。``force_angled=True`` = 「标定（90° 装置）」按钮。

        两者**走同一条函数**：90° 只是把模式锁成 angled、跳过 FOV，其余
        （相机检查、参数校验、任务管理）完全一样 —— 复制一份命令拼装
        正是本项目反复出问题的地方。
        """
        mode = "calibrate" if force_angled else self.calib_mode.get()
        if mode == "skip":
            messagebox.showinfo("已跳过标定",
                                "使用现有 / 默认参数直接检测。若三维距离不准，请先执行标定。")
            return
        left_raw, right_raw = self.calib_left.get(), self.calib_right.get()
        if not self._confirm_stereo_cameras(left_raw, right_raw):
            return
        left, right = _cam_arg(left_raw), _cam_arg(right_raw)
        width, height = self.calib_res.get().split("x")
        fps = self.calib_fps.get().strip() or CALIB_FPS_DEFAULT
        if mode == "calibrate":
            extra = self._calibrate_args(angled_90=force_angled)
            if extra is None:
                return                      # 校验失败，弹窗已经说清哪里不对
            cmd = [PY, str(ROOT / "apps" / "calibrate_stereo.py")] + extra
            target = self.calib_target.get().strip()
            base = self.calib_baseline.get().strip()
            note = f"收满 {target} 对后自动标定"
            if force_angled:
                note = "90° 装置 / angled / 跳过 FOV；" + note
            elif self.calib_skip_fov.get():
                note = "本次不做 FOV 对齐（两路原始画面）；" + note
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
                   "--width", width, "--height", height, "--fps", fps]
            cmd += self._backend_args()      # 同样是面板决定，不再写死 msmf
            self._run(cmd, "标定复检（判断装置是否被碰动）")

    def _run_calibration_90deg(self) -> None:
        """「标定（90° 装置）」：先把前提讲清，再走**同一条**标定链路（P0-3）。"""
        if not messagebox.askokcancel(
                "标定 90° 装置（大角度）",
                "接下来会按【90° 装置】标定：\n"
                "  · 模式强制为 angled（大角度 / 近似直角）\n"
                "  · 跳过 FOV 对齐 —— 两路各用原始画面，不裁剪、不缩放\n"
                "  · 输出三维坐标系为 left_camera\n\n"
                "前提（不满足就别用这个入口）：\n"
                "  1. 两路有共同可见区 —— 棋盘要能同时出现在左右画面里；\n"
                "  2. 基线 B 约为工作距离的 2 倍（B≈2·D，例如 D=0.30 m 时 B≈600 mm）。\n\n"
                "不确定装置属于哪一类？先点『基线夹角估算』，或看顶部那条装置状态。",
                parent=self):
            return
        self._run_calibration(force_angled=True)

    def _run_probe(self) -> None:
        """真的开一次相机，回答「现在有几台能出图」。

        ## 为什么换成 cameras.py probe（2026-10-18）

        原来调 list_cameras.py，它按**单后端**逐个试开。本机 MSMF/DSHOW
        会各自卡在一台上（一台只在 MSMF 出图、另一台只在 DSHOW 出图），
        于是用户看到"只有一台"却不知道为什么。

        新命令会**逐后端 × 逐编号**实测，并把"两台各自只在不同后端下工作"
        这种状态直接说出来 —— 那正是"双目跑不起来"的真因。
        """
        cmd = [PY, str(ROOT / "apps" / "tools" / "cameras.py"),
               "probe", "--max-index", "3"]
        self._run(cmd, "相机探测（真的开一次：有几台能出图、在哪个后端）")

    def _run_precheck(self) -> None:
        left_raw, right_raw = self.calib_left.get(), self.calib_right.get()
        if not self._confirm_stereo_cameras(left_raw, right_raw):
            return
        left, right = _cam_arg(left_raw), _cam_arg(right_raw)
        width, height = self.calib_res.get().split("x")
        fps = self.calib_fps.get().strip() or CALIB_FPS_DEFAULT
        cmd = [PY, str(ROOT / "apps" / "tools" / "check_stereo_setup.py"),
               "--left", left, "--right", right,
               "--width", width, "--height", height, "--fps", fps]
        cmd += self._backend_args()
        self._run(cmd, f"标定前置检查（左{left} / 右{right}：两路是否都看到棋盘）")

    def _refresh_cameras(self, *, probe: bool = True) -> None:
        """把下拉项换成"当前实际设备"（编号 | 名字 | 端口），并保留角色名选项。

        ``probe=True``（默认）：「刷新并检测相机」按钮走这条 —— 枚举之后**再真开一次**
        相机（`cameras.py probe`），回答"它现在活着吗、有几台能出图"。
        ``probe=False``：启动时走这条 —— **只枚举设备，不打开相机**。界面一出现就
        真开相机是隐私观感问题（相机灯亮），也是每次启动的固定成本（实测 23.9 s）。
        要实测出图，用户自己点「刷新并检测相机」，日志会说清这一步没开硬件。
        """
        try:
            from sphere_tracker import camera_registry as cr
            devices = cr.list_devices()
            reg = cr.load_registry()
        except Exception as exc:                             # noqa: BLE001
            self._append(
                f"[错误] 读取设备列表失败：{exc}。"
                "可能原因：config/cameras.json 内容损坏，或 WMI/PnP 枚举被系统策略挡住。"
                "下一步：点『相机设备管理』重新识别一次；仍失败就手动选编号 "
                "0/1/2 应急（此时角色 left/right 无法按身份绑定）。")
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
            # 必须说清这个按钮的能力边界：只点「刷新并检测相机」才真开相机实测；
            # 启动时那次枚举不碰硬件，所以"相机现在活着吗"这个问题它还回答不了。
            # （用户 2026-10-18 反馈："刷新设备列表和检查映射不能保证有用"。）
            if probe:
                self._append("接下来会实测每台能不能出图（下面另起一个任务）。")
                # 枚举 ≠ 能用：立刻再跑一次真开相机的探测，让这个按钮真的回答问题。
                self._run([PY, str(ROOT / "apps" / "tools" / "cameras.py"),
                           "probe", "--max-index", "3"],
                          "相机探测（真的开一次：有几台能出图、在哪个后端）")
            else:
                self._append("本次只枚举了设备，未打开相机（相机灯不会亮）。"
                             "要逐台实测能不能出图，点『刷新并检测相机』。")
            if reg.ignored:
                names = "、".join((s.name or f"#{s.index}") for s in reg.ignored)
                self._append(f"已标为「无关」（自动挑相机会跳过）：{names}")
            else:
                self._append("提示：不打算用的相机（笔记本内置 / 红外）可以在"
                             "『相机设备管理』的画面里按 n 标成「无关」。")
        else:
            self._append("没有枚举到设备（非 Windows 或系统枚举不可用），"
                         "只能按编号使用。")

    def _run_camera_quality(self) -> None:
        """画质诊断（P1-3）：调 `apps/tools/camera_quality.py --together`。

        `--together` 是关键：单开一台测的是"这台相机的上限"，两路同开才暴露
        共用一条 USB 总线时的争抢 —— 这决定了结论是"换线换口"还是"擦镜头"。
        走 `_run`（与其它长任务同一套任务管理：可中断、有日志）。
        """
        left_raw, right_raw = self.calib_left.get(), self.calib_right.get()
        if left_raw == "auto" or right_raw == "auto":
            src: list[str] = ["--auto-cameras"]
        else:
            if not self._confirm_stereo_cameras(left_raw, right_raw):
                return
            src = ["--left", _cam_arg(left_raw), "--right", _cam_arg(right_raw)]
        width, height = self.calib_res.get().split("x")
        fps = self.calib_fps.get().strip() or CALIB_FPS_DEFAULT
        cmd = [PY, str(ROOT / "apps" / "tools" / "camera_quality.py")] + src
        cmd += ["--width", width, "--height", height, "--fps", fps,
                "--together", "--seconds", "3"]
        cmd += self._backend_args()
        self._append("画质诊断：会先后各开一次 + 两路同开，约 10 秒；"
                     "结论看「帧率/残帧」是否异常 —— 都不异常就不是带宽问题。")
        self._run(cmd, "相机画质诊断（单开 vs 两路同开）")

    def _run_cameras_identify(self) -> None:
        self._run([PY, str(ROOT / "apps" / "tools" / "cameras.py"), "identify"],
                  "相机设备管理（逐个画面确认：l=左 / r=右 / n=无关 / m=主，"
                  "结果写入 config/cameras.json）")

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
    def _run(self, cmd: list[str], title: str,
             on_done: Callable[[Task], None] | None = None) -> Task:
        """跑一个外部程序。

        ## 与原来的区别

        原来只留一个 `self.proc`，第二个任务会被"已有程序在运行"挡掉，
        而且**没有任何中断手段**。现在：

        - **允许多个任务同时跑**（用户可能就要同时开两个窗口对比）；
          相机冲突由用户自己负责 —— 拦着不许开反而更妨碍人。
        - 每个任务都能在「任务」卡片里**单独中断**、看耗时、看返回码。
        - 关窗口时会**先把任务收干净**（原来不会，相机因此可能一直被占着）。
        """
        task = self.start_task(cmd, title, on_done=on_done)
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

    def start_task(self, cmd: list[str], title: str,
                   on_done: Callable[[Task], None] | None = None) -> Task:
        """起一个新任务并登记。**允许多个同时跑**（相机冲突由用户自己负责）。"""
        self._tid += 1
        proc = self._spawn(cmd)
        log_path = self._task_log_path(self._tid, title)
        task = Task(tid=self._tid, title=title, cmd=list(cmd), proc=proc,
                    log_path=log_path, on_done=on_done)
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

    def _force_kill_tree(self, pid: int) -> str:
        """强杀一棵进程树，**返回为什么失败**（成功时返回空串）。

        ⚠️ 这里必须检查 `returncode` 与 `stderr`：走到这个函数时通常
        "优雅退出没起作用"，如果强杀也失败，**那就真的没停掉** ——
        而旧实现是 `subprocess.run(..., capture_output=True)` 之后
        **两个都不看**，于是权限不足（实测 stderr 为 `ERROR: Access denied`）
        被当成"停好了"：界面显示已停、任务却还在，相机还被占着。
        """
        cmd = (["taskkill", "/PID", str(pid), "/T", "/F"] if os.name == "nt"
               else ["kill", "-9", str(pid)])
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=10)
        except Exception as exc:                            # noqa: BLE001
            return f"{type(exc).__name__}: {exc}"
        if r.returncode == 0:
            return ""
        detail = ((r.stderr or "") + (r.stdout or "")).strip().splitlines()
        return (f"{cmd[0]} 返回码 {r.returncode}"
                + (f"：{detail[0]}" if detail else "（没有输出）"))

    def stop_task(self, task: Task, graceful: bool = True) -> bool:
        """停掉一个任务（含它的子进程树）。返回**是否确认已经停掉**。

        ## 为什么要杀整棵树，而不是 `proc.terminate()`

        子进程是 python；父进程被杀之后**它还可能活着**，于是相机仍被占着，
        下一次运行打不开相机，而用户看到的是"停了但相机用不了"——
        这正是"没有任何中断手段"之外的另一半问题。

        ## 为什么要先优雅后强杀

        `CTRL_BREAK_EVENT` 让程序自己走退出流程（例如标定程序来得及把已经
        收下的图像对写完）。直接 `taskkill /F` 会把它半路打断。

        ## 返回值的语义（2026-10-14 改）

        原来是"是否真的下手了"，**永远为真** —— 于是"没杀掉"与"杀掉了"
        返回值一模一样，调用方无从分辨。现在返回"**确认停掉了吗**"：
        强杀之后复查一次进程树，还在就返回 `False` 并把原因写进
        `task.kill_note`（界面会报警）。
        """
        if not task.running:
            return True
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
                return self._confirm_dead(task, kids)
            except subprocess.TimeoutExpired:
                self._append(f"[任务 {task.tid}] 3 秒内没退出，改为强制结束")

        # ② 强杀（含子树）
        why = self._force_kill_tree(pid)
        if why:
            self._append(f"[任务 {task.tid}] ⚠ 强制结束没成功：{why}")
            task.kill_note = f"强制结束未成功（{why}）"
        else:
            self._append(f"[任务 {task.tid}] 已强制结束整棵进程树")
        self._kill_leftover(kids, task)
        return self._confirm_dead(task, kids)

    def _confirm_dead(self, task: Task, kids: list[int], wait: float = 2.0) -> bool:
        """等一小会儿再复查：父进程与子进程**真的都没了**才算停掉。

        为什么不能杀完就返回 `True`：`taskkill` 可能返回非零、也可能"命令成功
        但进程仍卡在不可中断状态"。`_alive_pids` 查不出来时也算活着，
        所以这里**只会偏保守**：最坏情况是多报一次"可能没停干净"。
        """
        deadline = time.monotonic() + wait
        while True:
            alive = _alive_pids([task.proc.pid, *kids])
            if not alive:
                return True
            if time.monotonic() >= deadline:
                task.kill_note = (f"仍有 {len(alive)} 个进程存活：{alive}"
                                  + (f"（{task.kill_note}）" if task.kill_note else ""))
                self._append(
                    f"[任务 {task.tid}] ⚠ 停完仍有 {len(alive)} 个进程存活（{alive}）——"
                    f"相机可能仍被占着，请确认后再启动新任务")
                return False
            time.sleep(0.05)

    def _kill_leftover(self, kids: list[int], task: Task) -> None:
        """收尾：子树里还有活着的就补刀（否则相机可能仍被占着）。"""
        alive = _alive_pids(kids)
        if not alive:
            return
        self._append(f"[任务 {task.tid}] 仍有 {len(alive)} 个子进程存活，补杀：{alive}")
        for pid in alive:
            why = self._force_kill_tree(pid)
            if why:
                self._append(f"[任务 {task.tid}] ⚠ 子进程 {pid} 补杀失败：{why}")

    def stop_all(self, graceful: bool = False) -> int:
        """停掉所有在跑的任务，返回**确认停掉**的个数。

        ⚠️ 返回的是"确认停掉"而不是"尝试停掉"：没停掉的会留在任务表里
        （状态显示失败原因），而不是从计数里消失。
        """
        running = [t for t in self.tasks if t.running]
        ok = 0
        for t in running:
            if self.stop_task(t, graceful=graceful):
                ok += 1
        if ok < len(running):
            self._append(f"[任务] ⚠ {len(running)} 个任务里只有 {ok} 个确认停掉，"
                         f"其余请看上面的失败原因")
        return ok

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
        self.done_queue.put(task.tid)

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
        while True:
            try:
                tid = self.done_queue.get_nowait()
            except queue.Empty:
                break
            task = next((t for t in self.tasks if t.tid == tid), None)
            if task is not None and task.on_done is not None:
                try:
                    task.on_done(task)
                except Exception as exc:                    # noqa: BLE001
                    self._append(f"[任务 {task.tid}] 完成回调失败：{exc}")
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
            "这里列出启动器起过的每个程序。卡住时用它中断 ——"
            "直接关启动器窗口是不会杀掉子进程的，相机可能仍被占着。"
            "「中断」先试优雅退出（让它自己收尾），3 秒不退再强制结束整棵进程树。")
        cols = ("id", "title", "pid", "elapsed", "status")
        heads = ("#", "任务", "PID", "已运行", "状态")
        widths = (4, 34, 8, 9, 12)
        tree_frame = ttk.Frame(box)
        tree_frame.grid(row=1, column=0, columnspan=8, sticky="we", pady=(4, 4))
        self.task_tree = ttk.Treeview(tree_frame, columns=cols, show="headings",
                                      height=5, selectmode="browse")
        for c, h, w in zip(cols, heads, widths):
            self.task_tree.heading(c, text=h)
            self.task_tree.column(c, width=w * 8, anchor="w", stretch=(c == "title"))
        self.task_tree.pack(side="left", fill="both", expand=True)
        task_sb = ttk.Scrollbar(tree_frame, orient="vertical",
                                command=self.task_tree.yview)
        task_sb.pack(side="right", fill="y")
        self.task_tree.configure(yscrollcommand=task_sb.set)
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

        # 任务多时既能滚，也能直接拖滑块调整可见行数
        box.columnconfigure(1, weight=1)
        ttk.Label(box, text="显示行数").grid(row=3, column=0, sticky="w", pady=(4, 0))
        self.task_rows_var = tk.DoubleVar(value=5)
        ttk.Scale(box, from_=3, to=20, orient="horizontal",
                  variable=self.task_rows_var,
                  command=self._set_task_rows).grid(
            row=3, column=1, columnspan=4, sticky="we", padx=(6, 6), pady=(4, 0))
        self.task_rows_label = ttk.Label(box, text="5 行", style="Hint.TLabel")
        self.task_rows_label.grid(row=3, column=5, sticky="w", pady=(4, 0))

        self.task_hint = ttk.Label(box, text="", style="Hint.TLabel",
                                   wraplength=CARD_HINT_WRAP)
        self.task_hint.grid(row=4, column=0, columnspan=9, sticky="w", pady=(4, 0))
        self._wrapped.append((self.task_hint, "card"))

    def _set_task_rows(self, value) -> None:
        """拖动滑块调整任务表可见高度；行数越多，卡片越高。"""
        try:
            rows = int(round(float(value)))
        except (TypeError, ValueError):
            return
        rows = max(3, min(20, rows))
        self.task_tree.configure(height=rows)
        self.task_rows_label.configure(text=f"{rows} 行")

    def _refresh_tasks(self) -> None:
        """重画任务表。**保留选中行**（否则每次刷新都把用户的选择弄丢）。"""
        if not hasattr(self, "task_tree"):
            return
        tree = self.task_tree
        keep = self._sel_tid
        tree.delete(*tree.get_children())
        for t in self.tasks:
            if t.running:
                # 停过但**没停掉**的，状态里直接写明原因 ——
                # 否则它和"从没停过"看起来一样，而它正占着相机。
                status = f"⚠ {t.kill_note}" if t.kill_note else "运行中"
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
        if not self.stop_task(t, graceful=True):
            messagebox.showwarning(
                "没停下来",
                f"任务 {t.tid}「{t.title}」没能停掉：\n{t.kill_note or '原因不明'}\n\n"
                "它可能仍占着相机。可以在任务表里再点一次「中断」，"
                "或直接去看它的日志文件。")
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
                f"任务 {t.tid}「{t.title}」会被立刻杀掉（含子进程树），"
                "它来不及保存已完成的结果。\n\n确定吗？"):
            return
        if not self.stop_task(t, graceful=False):
            messagebox.showwarning(
                "没能强制结束",
                f"任务 {t.tid}「{t.title}」还在跑：\n{t.kill_note or '原因不明'}\n\n"
                "常见原因是权限不足。相机可能仍被它占着 —— "
                "请用任务管理器结束它，或注销后重试。")
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
        if t.kill_note:
            self._append(f"停止  : ⚠ {t.kill_note}")
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
                    "退出启动器会把它们一并结束（否则相机等资源可能一直被占着）。\n"
                    "确定退出吗？"):
                return
            self._append("[退出] 正在结束仍在运行的任务…")
            n = self.stop_all(graceful=True)
            self._append(f"[退出] 已确认结束 {n} 个任务")
            if n < len(running):
                # ⚠ 不能静默退出：没停掉的那些**会继续占着相机**，
                # 而用户下次运行只会看到"打不开相机"。
                left = [t for t in running if t.running]
                messagebox.showwarning(
                    "有任务没停下来",
                    f"{len(running)} 个任务里有 {len(left)} 个没能结束："
                    + "、".join(f"{t.tid}「{t.title}」(PID {t.proc.pid})" for t in left[:4])
                    + "\n\n启动器关闭后它们仍在运行，相机等资源会一直被占着。\n"
                      "请用 Windows 任务管理器结束这些 PID，或注销后重试。")
        self.destroy()


def main(argv: list[str] | None = None) -> int:
    """入口：默认新界面（门厅 + 三个模式窗）；``--legacy`` 走经典单窗口。

    为什么两个入口并存（2026-10-28）：分窗在逐窗落地（单目窗、双目窗已建，
    收集图/工作台的若干动作仍指向经典界面的等价入口）。**旧代码一行没删** ——
    删它要同步改 ``selftest_launcher_params`` / ``selftest_entrypoints`` 的判据，
    而按全局 §15.2，改测试判据必须由**非实现者** Critic 签 PASS；
    限流期间没有可用 Critic，所以这一步留给 B-102，由独立 Critic 验收。

    过渡期两条硬约束（写在这里，免得后来者以为是遗漏）：

    1. 经典界面只能从门厅的「经典界面（过渡）」或 ``--legacy`` 进入，
       不提供"两个都是主入口"的错觉；
    2. 新界面是**默认** —— 否则用户看不到变化，重做就没有意义。
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if "--legacy" not in args:
        from apps._gui_app import run as _run_new_gui
        return _run_new_gui()
    Launcher().mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
