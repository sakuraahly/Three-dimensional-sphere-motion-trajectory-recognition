"""相机管理对话框（GUI 内）：列设备、**同时预览多路**、按名字绑左右/主/无关。

## 为什么不再复用 apps/tools/cameras.py identify

2026-11-07 用户连着报了两次「GUI 的平行双目相机管理打不开」。那条路起一个
控制台子进程，逐台 `cv2.VideoCapture` + `imshow` + `waitKey` 让人按键：从 GUI 启动
时 stdin 是 DEVNULL、OpenCV 窗口不归 Tk 管、4 台各等 30 秒。命令行工具保留给
CLI，界面里改成 Tk 自己的窗口。

## 2026-11-07 傍晚：预览为什么"依旧打不开"，以及为什么改成多路

用户第三次指出：Windows「摄像机管理」里能打开的外接相机，这里打不开；并要求
"预览要可以同时开多个，不要局限于同时开一个"。实测确认**是程序问题，不是硬件**：

1. 旧实现用裸 `cv2.VideoCapture(index, CAP_MSMF)` + 请求 640x480 —— 项目
   `apps/tools/cameras.py::_try_open` 的注释早就记着，这些 aoni 在 MSMF 下
   请求低分辨率会"打开成功但一帧都不给"，必须走 `CameraSource` 的分辨率
   梯度（先 1920、逐级降档、逐档等首帧）。这里却把那套已验证的逻辑绕开了。
2. 旧版那个 6 秒看门狗到点就 `stop.set()`。本机实测 aoni #0 的
   首帧要 19–24 秒（MSMF 报 -2147483638 后降档才出图）—— 6 秒必然误判成
   "打不开"，而且把正在慢慢开的相机强停了。
3. 每开第二台，先 `stop.set()` 关掉第一台、只 `join(0.3s)` 就开新的：
   MSMF 的 `release()` 实测能拖几十秒，旧相机还没放手就抢新的，两路互相拖死。

现在：**每路一个独立 worker + 一个独立 stop**，互不影响；统一走 `CameraSource`；
看门狗只提示、不强停（慢相机继续等，用户可单独「停止预览」）。
`CameraSource` 里请求 1920x1080 是项目实测的兼容档，双路同开 1920 稳定、
2560 会让一路会话失效，所以这里不请求 2560。

**不碰硬件配置以外的东西**：写盘只有 ``config/cameras.json`` 一份。
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sphere_tracker import camera_registry as cr  # noqa: E402
from apps._gui_widgets import BG, MUTED  # noqa: E402

#: 本模块被加载进内存的时刻。用来判断「用户点的这个窗口是不是旧版」：
#: 磁盘上的 .py 比它新，就说明进程里跑的还是改动前的代码（见
#: `CamerasDialog._warn_if_code_is_newer`）。这个坑在 2026-11-07/08 让同一个
#: 问题被重复报了七八次。
_MODULE_LOADED_AT = time.time()

#: 交给 CameraSource 的首帧 / 整台预算。
#:
#: 2026-11-07 实测：aoni #0 在 MSMF **1920x1080 这一档永远拿不到帧**
#: （每 10 秒报一次 -2147483638），降档到 640 后才出图（整段 48 秒）。
#: 原来给 30 秒首帧，等于在一个确定性的坏档上白等 30 秒 —— 用户看到的就是
#: "点了半天打不开"。改成 8 秒：正常相机 1–9 秒能出（#1 实测 0.7–8.8 秒），
#: 坏档 8 秒就降级；整台 18 秒封顶。
OPEN_FIRST_FRAME_S = 7.0
OPEN_BUDGET_S = 14.0
#: 超过这个时间还没第一帧，**只提示不强停**（慢相机继续等）。
WATCHDOG_S = 45.0
#: 主线程轮询后台消息的间隔。
POLL_MS = 80
#: 设备变化自动检测间隔（pnputil 只枚举、不打开相机，单次约 0.1s）。
DEVICE_POLL_S = 1.5
#: 每格预览的最大尺寸。
PANE_W, PANE_H = 320, 240
#: 预览网格列数。
PANE_COLS = 2
#: 卡片状态行最多显示多少字。CameraSource 抛出的 RuntimeError 有好几段
#: 「可能原因」，整段塞进卡片会把卡片撑高、压住下一行（2026-11-07 用户截图）。
#: 卡片里只放摘要，完整原因写主状态条，主状态条也按同一口径截短。
BRIEF_LIMIT = 72
STATUS_LIMIT = 120


#: 打开阶段**全局串行**：这台机器上两路 MSMF 同时打开会互相抢会话
#: （实测同一时刻两路一起开会一路/两路都报 -2147483638，过一会儿单开又能开）。
#: 多路预览要的是"同时看画面"，不是"同时抢打开"——所以只把 open 这一步排队，
#: 画面读帧仍然各路并行。
_OPEN_LOCK = threading.Lock()

#: 「画面几乎全黑」判定的**预热与滞回**（2026-11-16 用户截图）。
#:
#: 旧实现只看**第一帧**：`frame.mean() < 8` 就把它钉成"画面几乎全黑"，
#: 之后永不撤销。可相机出图后的前几帧常常是黑的（自动曝光/增益还在收敛），
#: 于是卡片的画面上明明是正常的房间，下面却写着"镜头可能被遮住"，
#: 还配着「重试这一路」—— 用户看到的就是"能打开却显示打不开"。
#:
#: 现在：先放过前 `DARK_WARMUP_FRAMES` 帧与 `DARK_WARMUP_S` 秒，
#: 之后 <8 判黑、>=16 撤销（滞回，避免明暗临界时来回跳）。
DARK_WARMUP_FRAMES = 10
DARK_WARMUP_S = 0.6
DARK_ENTER = 8.0
DARK_CLEAR = 16.0

#: 显示用：没标过就写这个（与 `camera_registry.location_text()` 同一口径）。
LOCATION_UNSET_TEXT = "未标注"
#: 显示用：`capture_index` 还没实测时的写法。**不许**写 0 / #index 冒充。
CAPTURE_UNMEASURED_TEXT = "未实测"



def _brief(text, limit: int = BRIEF_LIMIT) -> str:
    """把多行长错误压成一行摘要：卡片/状态条的高度不许被文字撑爆。"""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


def _is_invalidated(err) -> bool:
    """这个错误是不是「采集会话失效」（`MF_E_VIDEO_RECORDING_DEVICE_INVALIDATED`）。

    为什么要单独判：这种失效是**设备/驱动层**的状态，重试只会再开关一次设备，
    既救不回来，又让已经卡住的相机栈更难受（这台 aoni 实测就是这样）。
    所以遇到它**不重试**，直接报错并让用户拔插/重启 —— 也把失败时间从
    两次 14 秒砍到一次。
    """
    text = str(err or "")
    return "2147483638" in text or "DEVICE_INVALIDATED" in text


def _short_reason(text) -> str:
    """把 CameraSource 的长解释翻成一句能照着做的结论。

    2026-11-07 实测：aoni #0 报 `-2147483638`，OpenCV 的原文是
    ``MF_E_VIDEO_RECORDING_DEVICE_INVALIDATED`` —— 设备在、能枚举、能"打开成功"，
    但那条采集会话已经失效，每帧都失败。项目 2026-11-05 的实测结论是
    **拔插一次就好**（1.2–1.6 s 出图）。所以这里的第一引导必须是"拔插"，
    不能笼统写成"被别的程序占着"，否则用户会去关一堆没开的软件。
    """
    flat = " ".join(str(text).split())
    if "2147483638" in flat or "DEVICE_INVALIDATED" in flat:
        return ("采集会话失效（不是相机坏）：先把这台 USB 拔插一次，"
                "再点「重试这一路」")
    if "无法打开摄像头" in flat or "打不开" in flat:
        # 实测最常见的两种，按**先能自己解决**的顺序写：
        # ① 上一路（别的程序 / 上一个本程序窗口 / 刚关掉的预览）还没真正放手，
        #    这两台 aoni 要几十秒冷却 → 等一会儿点「重试这一路」就好；
        # ② 确实被占着 → 关掉 Windows 相机 / QQ / 会议；
        # ③ 采集会话失效 → 拔插一次。
        # 卡面只有 72 字，所以最可能有效的那条放最前面。
        return ("打不开：先等 30–60 秒再点「重试这一路」（上一路刚用过要放手）；"
                "再查占用（Windows 相机 / QQ）；还不行拔插这台")
    if "没画面" in flat or "无帧" in flat or "读不到帧" in flat:
        return "打开了但没画面：先拔插这台，再查镜头遮挡/占用"
    return _brief(flat)


#: 常见会抓摄像头的程序（进程名小写 -> 用户看得懂的名字）。
#:
#: 不是"元凶名单"，只在**预览失败**时给一句"现在开着谁"，省得用户去关一堆
#: 没开的软件（旧文案就犯过这个毛病）。2026-11-08 实测这台机器上 QQ、微信、
#: 钉钉全都在跑，而当天两台相机确实同时报 `DEVICE_INVALIDATED`。
_CAMERA_HOGS = {
    "qq.exe": "QQ", "weixin.exe": "微信", "wechat.exe": "微信",
    "dingtalk.exe": "钉钉", "teams.exe": "Teams", "ms-teams.exe": "Teams",
    "zoom.exe": "Zoom", "windowscamera.exe": "Windows 相机",
    "obs64.exe": "OBS",
}
_HOGS_CACHE: tuple[float, str] | None = None


def _camera_hogs_text() -> str:
    """现在开着、且可能占着相机的程序名（没有就空串，缓存 60 秒）。"""
    global _HOGS_CACHE
    now = time.monotonic()
    if _HOGS_CACHE is not None and now - _HOGS_CACHE[0] < 60.0:
        return _HOGS_CACHE[1]
    text = ""
    if sys.platform == "win32":
        try:
            import subprocess
            out = subprocess.run(["tasklist", "/FO", "CSV", "/NH"],
                                 capture_output=True, text=True, timeout=4)
            names = set()
            for line in (out.stdout or "").splitlines():
                first = line.split(",", 1)[0].strip().strip('"').lower()
                if first in _CAMERA_HOGS:
                    names.add(_CAMERA_HOGS[first])
            if names:
                text = "、".join(sorted(names))
        except Exception:                                    # noqa: BLE001
            text = ""
    _HOGS_CACHE = (now, text)
    return text


#: 打开失败时说清「为什么不换后端试试」（2026-11-11 vivo 虚拟摄像头事件后加）。
#:
#: DSHOW 的枚举与这份设备列表**不是同一套编号**（AGENTS.md §9：索引与标签只在
#: `INDEX_BACKEND`（MSMF）内成立）。同一编号在 DSHOW 里可能是另一台设备 ——
#: 用户实测点了 #3「Web Camera Right」的 DSHOW 兜底，画面上开出来的是
#: vivo 办公套件的虚拟摄像头（它只注册在 DSHOW，PnP 列表里根本没有）。
#: 所以**不切换后端**，预览只走 MSMF；打不开就如实说，并给出能照着做的下一步。
_NO_DSHOW_HELP = (
    "为什么这里不切换后端：DSHOW 的编号与这份设备列表不是同一套，"
    "按同一个编号打开可能开到别的设备（实测出现过：点 #3 的 DSHOW 试开，"
    "画面上却是 vivo 办公套件的虚拟摄像头）。所以预览只用系统默认后端（MSMF）。"
)
_OPEN_FIX_STEPS = (
    "下一步：① 先点卡片上的「重试这一路」；② 还不行就拔插一次这台相机的 USB，"
    "并关掉可能占着相机的程序（微信 / QQ / 钉钉 / OBS / Windows 相机等）；"
    "③ 所有相机（含内置）都打不开，先查 设置 → 隐私和安全性 → 相机 → "
    "允许桌面应用访问相机；④ 仍打不开就跑 run.bat doctor 或 "
    "python apps/tools/cameras.py doctor —— 它只报告，不改配置、也不绑左右角色。"
)


class _PreviewPane:
    """一路预览：自己的控件、自己的线程、自己的停止信号。"""

    def __init__(self, parent, index: int, name: str, on_close,
                 on_retry=None, identity: str = "") -> None:
        self.index = index
        self.name = name
        self.identity = identity
        self.frame = ttk.LabelFrame(parent, text=f"#{index} {name}")
        holder = ttk.Frame(self.frame, width=PANE_W, height=PANE_H)
        holder.pack(padx=4, pady=(4, 2))
        holder.pack_propagate(False)
        self.label = ttk.Label(holder, text="正在打开…",
                               background="#1b1f24", foreground="#c9d1d9",
                               anchor="center")
        self.label.pack(fill="both", expand=True)
        # 状态行固定两行高：长错误只显示摘要，卡片总高不随文字增长。
        status_box = ttk.Frame(self.frame, height=36, width=PANE_W)
        status_box.pack(fill="x", padx=4)
        status_box.pack_propagate(False)
        self.status = ttk.Label(status_box, text="等待画面…", foreground=MUTED,
                                wraplength=PANE_W, anchor="nw", justify="left")
        self.status.pack(fill="both")
        btns = ttk.Frame(self.frame)
        btns.pack(fill="x", padx=4, pady=(0, 4))
        ttk.Button(btns, text="停止预览", command=on_close).pack(side="right")
        if on_retry is not None:
            ttk.Button(btns, text="重试这一路", command=on_retry).pack(
                side="right", padx=(0, 6))

        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.started = time.monotonic()
        self.got = False
        self.failed = False
        self.warned = False
        self.photo = None
        #: 当前阶段文案（"正在连接…" / "排队等…"），`_pump` 会在后面追加"已等 N 秒"。
        self.stage = "正在连接（逐档试分辨率）…"
        self.shown_secs = -1
        #: **每路一个**帧槽（maxsize=1）。2026-11-07 修：原来全局共用一个，
        #: 两路同时推帧时后到的会把先到的顶掉 —— 表现就是某一路永远是黑的。
        self.frames: queue.Queue = queue.Queue(maxsize=1)


class CamerasDialog(tk.Toplevel):
    """一台 Tk 窗口搞定相机管理；不开 cv2 窗口、不按键、不起子进程。"""

    def __init__(self, master, on_changed=None) -> None:
        super().__init__(master)
        self.title("相机管理")
        self.configure(bg=BG)
        # 标准窗口：最小化 / 最大化 / 关闭（AGENTS.md §7.1）。
        # 不用 transient()（Windows 上变工具窗口、只剩 ×）、不锁尺寸、不全局 grab。
        self.resizable(True, True)
        # `geometry()` 用的是**逻辑像素**：写死 1180x720 在 300% 缩放下等于
        # 3496x2133 物理像素，比屏幕还大，系统会把它压成接近全屏 ——
        # 用户看到的就是"每次都扩展到全屏、然后就固定了"（2026-11-08 反馈）。
        # 按屏幕可用尺寸取 82%，再给一个能看的底线。
        _sw = self.winfo_screenwidth()
        _sh = self.winfo_screenheight()
        _w = max(760, min(1180, int(_sw * 0.82)))
        _h = max(520, min(720, int(_sh * 0.82)))
        self.geometry(f"{_w}x{_h}")
        self.minsize(760, 520)
        self.on_changed = on_changed

        self._devices: list[cr.DeviceEntry] = []
        self._events: queue.Queue = queue.Queue()
        #: 每一路预览一条记录：key = 设备编号。多路互不干扰。
        self._previews: dict[int, _PreviewPane] = {}
        #: 完整错误原因（卡片里只放摘要，双击状态行看全文）。
        self._full_errors: dict[int, str] = {}
        self._after_id: str | None = None
        #: 界面刷新链 (`_pump`) 累计出错次数。出错**不会**再让界面冻结。
        self._pump_failures = 0
        #: 设备自动检测：签名变化就刷新列表；不靠用户手动点「刷新列表」。
        self._device_sig = None
        self._polling = False
        self._last_poll = time.monotonic()

        # 筛选门限：默认值与配置合成一次，界面显示的就是**真正生效的值**。
        # `min_pair_fps` 空 = 沿用 14.4（不把默认值伪装成用户设置）。
        _reg0 = cr.load_registry()
        _merged0 = cr.merged_settings(reg=_reg0)
        def _setting_text(key: str) -> str:
            raw = _reg0.settings.get(key, _merged0.get(key))
            return "" if raw is None else str(raw)
        self.var_min_width = tk.StringVar(
            value=_setting_text("min_width") or "1280")
        self.var_min_height = tk.StringVar(
            value=_setting_text("min_height") or "720")
        self.var_min_fps = tk.StringVar(
            value=_setting_text("min_fps") or "15")
        self.var_min_pair_fps = tk.StringVar(
            value="" if _reg0.settings.get("min_pair_fps") is None
            else str(_reg0.settings.get("min_pair_fps")))

        body = ttk.Frame(self, padding=12)
        body.pack(fill="both", expand=True)

        ttk.Label(
            body,
            text="按画面认清哪台是左/右，然后选中它点「设为左 / 设为右」。"
                 "预览可以同时开多路，用来并排比对；绑定按设备身份记，换 USB 口也不会漂。"
                 "表里「PnP 位次」是系统枚举顺序，**不是**采集索引；"
                 "「采集索引」只在实测（identify）后才有值，两种编号不可互换。",
            wraplength=1140, foreground=MUTED).pack(anchor="w", pady=(0, 8))

        # 主分区必须可拖动（AGENTS.md §7.2）：两个内容区并排时用
        # ttk.PanedWindow，别用固定 pack(side=...) —— 内容一长就互相遮挡、
        # 用户也没法把宽的那边拉宽看看。
        panes = ttk.PanedWindow(body, orient="horizontal")
        panes.pack(fill="both", expand=True)
        self.panes = panes

        self._left_width = min(380, int(_w * 0.36))
        left = ttk.Frame(panes, width=self._left_width)
        # ⚠️ 列名必须自己说清"PnP 位次 ≠ 采集索引"（2026-11-12）。
        # 用户报了三轮"程序显示的相机 ≠ 真实相机"，根因就是这一栏被读成了
        # `cv2.VideoCapture` 的编号 —— 而它只是**系统枚举顺序**。表头直接写出
        # 这件事，比放到说明文字里有效（用户看的是表）。
        cols = ("idx", "name", "port", "loc", "cap", "role")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", height=14)
        # 表头 / 按钮一律写成显式 text= 常量：放进循环里由变量带进去的话，
        # i18n 提取器只看 AST 里的关键字常量，整排文案会静默漏译（B-187）。
        self.tree.heading("idx", text="PnP 位次（非采集索引）")
        self.tree.heading("name", text="设备")
        self.tree.heading("port", text="USB 端口")
        self.tree.heading("loc", text="位置")
        self.tree.heading("cap", text="采集索引")
        self.tree.heading("role", text="当前角色")
        self.tree.column("idx", width=150, anchor="w")
        self.tree.column("name", width=160, anchor="w")
        self.tree.column("port", width=100, anchor="w")
        self.tree.column("loc", width=60, anchor="w")
        self.tree.column("cap", width=80, anchor="w")
        self.tree.column("role", width=90, anchor="w")
        self.tree.pack(side="top", fill="both", expand=True)
        sb = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        # 六列在窄分栏里会溢出：给一条横向滚动条，别让「位置 / 采集索引」被裁掉
        # （裁掉的两列恰好是这次新加的关键信息）。
        xsb = ttk.Scrollbar(left, orient="horizontal", command=self.tree.xview)
        self.tree.configure(xscrollcommand=xsb.set)
        xsb.pack(side="bottom", fill="x")

        # 右侧：可滚动的多预览网格。Canvas + 内部 Frame 是 Tk 里做滚动容器
        # 的标准写法；直接给 ttk.Frame 加滚动条是不生效的。
        right = ttk.Frame(panes)
        ttk.Label(right, text="预览（可同时开多路）",
                  foreground=MUTED).pack(anchor="w")
        self.canvas = tk.Canvas(right, borderwidth=0, highlightthickness=0,
                                background=BG)
        vsb = ttk.Scrollbar(right, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vsb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self.grid_frame = ttk.Frame(self.canvas)
        self._win_id = self.canvas.create_window((0, 0), window=self.grid_frame,
                                                 anchor="nw")
        self.grid_frame.bind(
            "<Configure>",
            lambda _e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")))
        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.itemconfigure(self._win_id, width=e.width))

        bar = ttk.Frame(body)
        bar.pack(fill="x", pady=(8, 4))
        ttk.Button(bar, text="预览这台", command=self._start_preview).pack(
            side="left", padx=(0, 6))
        ttk.Button(bar, text="预览全部外接", command=self._preview_all).pack(
            side="left", padx=(0, 6))
        ttk.Button(bar, text="停止全部预览", command=self._stop_all).pack(
            side="left", padx=(0, 6))
        ttk.Button(bar, text="设为左",
                   command=lambda: self._set_role("left")).pack(
            side="left", padx=(0, 6))
        ttk.Button(bar, text="设为右",
                   command=lambda: self._set_role("right")).pack(
            side="left", padx=(0, 6))
        ttk.Button(bar, text="设为主相机",
                   command=lambda: self._set_role("main")).pack(
            side="left", padx=(0, 6))
        ttk.Button(bar, text="标为无关", command=self._mark_ignored).pack(
            side="left", padx=(0, 6))
        ttk.Button(bar, text="取消无关", command=self._unmark_ignored).pack(
            side="left", padx=(0, 6))
        ttk.Button(bar, text="刷新列表", command=self._refresh_devices).pack(
            side="left", padx=(0, 6))
        ttk.Button(bar, text="关闭", command=self._close).pack(
            side="left", padx=(0, 6))

        # 筛选门限：自动挑相机 / 成对协商用。放在底部工具行而不是新开窗口：
        # 它只影响"没绑角色时的自动选择"，不该抢设备表/预览的主视线。
        _box = ttk.LabelFrame(
            body, text="筛选门限（自动挑相机用；不影响已绑定角色）")
        _box.pack(fill="x", pady=(6, 0))
        _row = ttk.Frame(_box)
        _row.pack(fill="x", padx=6, pady=4)
        ttk.Label(_row, text="最小宽").pack(side="left")
        ttk.Entry(_row, textvariable=self.var_min_width, width=6).pack(
            side="left", padx=(5, 10))
        ttk.Label(_row, text="最小高").pack(side="left")
        ttk.Entry(_row, textvariable=self.var_min_height, width=6).pack(
            side="left", padx=(5, 10))
        ttk.Label(_row, text="最低实测 fps").pack(side="left")
        ttk.Entry(_row, textvariable=self.var_min_fps, width=6).pack(
            side="left", padx=(5, 10))
        ttk.Label(_row, text="成对下限 fps（空=14.4）").pack(side="left")
        ttk.Entry(_row, textvariable=self.var_min_pair_fps, width=6).pack(
            side="left", padx=(5, 10))
        ttk.Button(_row, text="保存门限", command=self._save_settings).pack(
            side="left")

        self.status = ttk.Label(body, text="读取设备中…", foreground=MUTED,
                                wraplength=1140)
        self.status.pack(anchor="w", pady=(4, 0))

        # 内容都建好之后再 add：ttk.PanedWindow 的 pane 只认 weight，
        # 先 add 再往里塞控件会让初始尺寸算错。
        panes.add(left, weight=1)
        panes.add(right, weight=3)
        try:
            self.canvas.bind_all(
                "<MouseWheel>",
                lambda e: self.canvas.yview_scroll(
                    int(-1 * (e.delta / 120)), "units")
                if e.widget.winfo_toplevel() is self else None)
        except tk.TclError:
            pass
        self.after(80, self._init_sash)

        self.protocol("WM_DELETE_WINDOW", self._close)
        self._pump()
        self._refresh_devices()
        self._warn_if_code_is_newer()

    def _warn_if_code_is_newer(self) -> None:
        """磁盘上的界面代码比本进程新 → 明说「你点的是旧版，要重开」。

        2026-11-08 反复踩的坑：改完代码让用户"再点一次"，而他点的是**改动前
        就开着的那个窗口**。Python 把旧模块留在内存里，改多少都不生效，于是
        同一个问题被报了七八次。这句话能把这个死循环直接掐断。
        """
        import importlib.util

        newer: list[str] = []
        for rel in ("apps/cameras_dialog.py", "apps/_gui_app.py",
                    "apps/_gui_stereo.py", "apps/_gui_mono.py",
                    "sphere_tracker/sources.py"):
            path = ROOT / rel
            try:
                if path.stat().st_mtime > _MODULE_LOADED_AT:
                    newer.append(rel)
            except OSError:
                continue
        if not newer:
            return
        stamp = time.strftime("%H:%M", time.localtime(
            max((ROOT / r).stat().st_mtime for r in newer)))
        msg = (f"⚠ 界面代码在 {stamp} 又改过（{len(newer)} 个文件），"
               f"当前窗口还是旧版 —— **关掉程序重开**才会生效")
        try:
            self.status.configure(text=_brief(msg, STATUS_LIMIT),
                                  foreground="#b91c1c")
        except tk.TclError:
            pass
        self._write_error_log(-1, msg + "\n  更新过的文件：" + "、".join(newer))

    def _init_sash(self) -> None:
        """初始分栏位置：设备表够看名字，剩下的都给预览网格。可随时拖。"""
        try:
            self.panes.sashpos(0, self._left_width)
        except (tk.TclError, AttributeError):
            pass

    # ------------------------------------------------------------------ #
    # 状态 / 消息泵
    # ------------------------------------------------------------------ #
    def _say(self, text: str, error: bool = False) -> None:
        # 主状态条也只放一行：CameraSource 的失败原因有好几段，整段塞进来会
        # 把底部按钮托下去，看起来像"界面被挡住了"。
        # 失败时用红色（项目约定涨红跌绿是行情图，界面告警用红没问题）。
        #
        # ⚠️ 这里**必须吞掉 TclError**：窗口正在关闭时控件已经销毁，`configure`
        # 会抛异常；而 `_say` 被 `_show_frame` 的兜底分支调用，一抛就会把
        # 整个 `_pump` 刷新链带走（2026-11-08 的根因，见 `_pump` 注释）。
        try:
            self.status.configure(
                text=_brief(text, STATUS_LIMIT),
                foreground="#b91c1c" if error else MUTED)
        except tk.TclError:
            pass

    def _log_line(self, text: str) -> None:
        """往预览日志追加**一行**（成功/路径也记）。

        2026-11-08：以前日志只有失败，于是"到底走的是直连还是逐档、花了几秒"
        全靠猜，同一件事反复排查。成功也留一行，下一次直接看日志就能定位。
        """
        try:
            folder = ROOT / "outputs" / "logs"
            folder.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with (folder / "camera_preview.log").open("a",
                                                       encoding="utf-8") as fh:
                fh.write(f"\n[{stamp}] {text}\n")
        except OSError:
            pass

    def _write_error_log(self, index: int, full: str) -> str:
        """把完整原因追加到日志文件，返回相对路径（失败也不抛）。"""
        try:
            folder = ROOT / "outputs" / "logs"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / "camera_preview.log"
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with path.open("a", encoding="utf-8") as fh:
                fh.write(f"\n[{stamp}] 相机 #{index}\n{full}\n")
            return "outputs/logs/camera_preview.log"
        except OSError:
            return ""

    def _pump(self) -> None:
        """主线程唯一入口：后台线程只往队列里放东西，绝不碰 Tk 控件。

        ⚠️ **2026-11-08 用户第七次报「打不开」后加的护栏**：这是整个界面刷新的
        唯一驱动（帧、状态、错误、看门狗全靠它 `after` 自己续命）。以前它整段
        没有异常保护 —— 只要任何一个分支抛一次（最典型的是控件已销毁后
        `configure` 抛 `TclError`，或 `ImageTk` 对畸形帧报错），异常就会穿出
        Tk 回调，**而 `pythonw.exe` 没有 stderr，连一行提示都看不到**，于是
        `self.after(POLL_MS, self._pump)` 这一句永远执行不到 —— 界面从此冻结：
        新建的卡片永远停在「正在打开…」，既不出图也不报错，只有重启程序能好。
        这正好解释了「我这边全新进程 1.2 秒出图，用户那边点多少次都没画面」。

        现在：一次异常只影响当前这一条消息，`finally` 里无论如何都会续约。
        """
        try:
            self._drain()
        except BaseException:                                # noqa: BLE001
            self._on_pump_error()
        finally:
            try:
                if self.winfo_exists():
                    self._after_id = self.after(POLL_MS, self._pump)
            except tk.TclError:
                pass                     # 窗口已销毁：不必再续约

    def _on_pump_error(self) -> None:
        """刷新链出过错：写日志 + 计数，**绝不把异常再抛回去**。"""
        import traceback
        self._pump_failures += 1
        text = traceback.format_exc()
        if self._pump_failures <= 5:         # 只记前 5 次，避免刷爆日志
            try:
                folder = ROOT / "outputs" / "logs"
                folder.mkdir(parents=True, exist_ok=True)
                with (folder / "camera_preview.log").open("a",
                                                           encoding="utf-8") as fh:
                    fh.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                             f"界面刷新异常（第 {self._pump_failures} 次，"
                             f"已自动继续）\n{text}\n")
            except OSError:
                pass
        try:
            self.status.configure(
                text=_brief(f"界面刷新出过错（{self._pump_failures} 次，"
                            f"已自动恢复；详情见 outputs/logs/camera_preview.log）"),
                foreground="#b91c1c")
        except tk.TclError:
            pass

    def _drain(self) -> None:
        """真正干活的部分：取帧、取消息、看门狗。可以抛，外层会兜。"""
        # 每路各取各的帧槽：多路互不顶掉（共用一条队列时后到的会盖掉先到的）。
        for pane in list(self._previews.values()):
            try:
                img = pane.frames.get_nowait()
            except queue.Empty:
                continue
            try:
                self._show_frame(pane, img)
            except BaseException:                            # noqa: BLE001
                self._on_pump_error()       # 一路显示失败不许连累其它路
                continue
        while True:
            try:
                item = self._events.get_nowait()
            except queue.Empty:
                break
            # 一条消息出错只丢这一条：设备列表读崩了也不许把预览的刷新一起带走。
            try:
                self._handle_event(item)
            except BaseException:                            # noqa: BLE001
                self._on_pump_error()

        # 拔插/设备变化：后台只枚举，变化了才刷新；不阻塞预览刷新链。
        self._maybe_poll_devices()

        # 看门狗：慢相机（本机实测首帧 19–24 秒）**只提示，不强停**。
        # 旧版 6 秒就 stop.set()，把正在慢慢开的相机自己掐死了。
        now = time.monotonic()
        for pane in list(self._previews.values()):
            if pane.got or pane.failed:
                continue
            # 等待期必须"看得见在动"：每秒把已等秒数写进状态行。
            # 慢相机 #0 实测要 20 秒上下，只写"正在连接"会被当成卡死。
            try:
                secs = int(now - pane.started)
                if secs != pane.shown_secs:
                    pane.shown_secs = secs
                    pane.status.configure(
                        text=_brief(f"{pane.stage} 已等 {secs} 秒"))
                if (not pane.warned
                        and now - pane.started > WATCHDOG_S):
                    pane.warned = True
                    pane.status.configure(text=_brief(
                        "还没出画面：慢相机可能要 20–30 秒；"
                        "一直不出就检查占用或拔插一次"))
            except tk.TclError:
                continue                     # 这张卡片已被销毁，跳过

    def _handle_event(self, item: tuple) -> None:
        """处理一条后台消息。`_drain` 逐条兜异常，所以这里可以让它抛。"""
        kind = item[0]
        if kind == "devices":
            self._on_devices(item[1], item[2] if len(item) > 2 else None)
        elif kind == "error":
            self._say(str(item[1]))
        elif kind == "pstatus":
            pane = self._previews.get(item[1])
            if pane is not None:
                pane.stage = str(item[2])
                # 颜色也由**这条文案**决定：全黑是警告（橙），其余回到默认。
                # 以前只在"第一帧刚好是全黑"那一刻上色，于是"全黑 → 亮回来"
                # 撤销之后文字还是橙的，看起来仍像故障（2026-11-16）。
                pane.status.configure(
                    text=_brief(pane.stage),
                    foreground="#b45309" if "全黑" in pane.stage else "")
        elif kind == "perror":
            pane = self._previews.get(item[1])
            short = str(item[2])
            full = str(item[3]) if len(item) > 3 else short
            hint = str(item[4]) if len(item) > 4 else ""
            if hint:
                # 失败时说出"现在开着谁"，比笼统列五种可能更有用。
                full += f"\n\n现在开着的、可能占着相机的程序：{hint}"
            self._full_errors[item[1]] = full
            log = self._write_error_log(item[1], full)
            if pane is not None:
                pane.failed = True            # 停止"已等 N 秒"的计时显示
                pane.stage = short
                pane.status.configure(text=_brief(short),
                                      foreground="#b91c1c")
            tail = f"（全文：{log}）" if log else "（双击状态行看全文）"
            note = f"｜现在开着 {hint}，可能占着相机" if hint else ""
            # 两台**同时**失效是"系统相机栈卡住"的签名（2026-11-08 实测：
            # 两台都 isOpened=True 但一帧不给、每次读阻塞 10 秒）。这时拔插常常
            # 不够，重启电脑才彻底 —— 必须说出来，否则会一直让人反复拔插。
            failed = sum(1 for p in self._previews.values() if p.failed)
            if failed >= 2 and "拔插" in short:
                note += "｜两台都这样＝系统的相机服务卡住了：重启电脑比拔插更彻底"
            self._say(f"#{item[1]}：{_brief(short)}{note}{tail}", error=True)

    # ------------------------------------------------------------------ #
    # 枚举
    # ------------------------------------------------------------------ #
    def _refresh_devices(self) -> None:
        self._say("正在读取设备列表（只枚举，不开相机）…")
        threading.Thread(target=self._enumerate_worker, daemon=True).start()

    def _enumerate_worker(self) -> None:
        try:
            devices = cr.list_devices()
            self._events.put(("devices", devices, cr.device_signature(devices)))
        except Exception as exc:                              # noqa: BLE001
            self._events.put(("error", f"设备列表读不出来：{exc}"))

    def _maybe_poll_devices(self) -> None:
        """每 DEVICE_POLL_S 后台枚举一次；签名变了才通知主线程刷新。"""
        if self._polling:
            return
        if time.monotonic() - self._last_poll < DEVICE_POLL_S:
            return
        self._last_poll = time.monotonic()
        self._polling = True
        threading.Thread(target=self._poll_worker, daemon=True,
                         name="camera-device-poll").start()

    def _poll_worker(self) -> None:
        try:
            devices = cr.list_devices()
            sig = cr.device_signature(devices)
            if sig != self._device_sig:
                self._events.put(("devices", devices, sig))
        except Exception as exc:                              # noqa: BLE001
            self._events.put(("error", f"自动检测设备失败：{exc}"))
        finally:
            self._polling = False

    def _auto_check_text(self) -> str:
        """自动核对 left/right：当前设备名/端口/采集索引是否符合配置。"""
        try:
            reg = cr.load_registry()
        except Exception:                                     # noqa: BLE001
            return "自动核对：配置读不出来"
        parts = []
        for role in ("left", "right"):
            spec = reg.roles.get(role)
            label = (spec.location or spec.name or role) if spec else "未配置"
            h = cr.role_health(role, reg=reg, devices=self._devices)
            if h.status == "ok":
                parts.append(f"{role}={label} 一致")
            elif h.status == "unverified":
                parts.append(f"{role}={label} 采集索引未实测")
            elif h.status == "missing":
                parts.append(f"{role}={label} 不在线/身份变了")
            else:
                parts.append(f"{role}={label} {h.detail}")
        return "自动核对：" + "｜".join(parts)

    def _stop_stale_previews(self) -> None:
        """设备被拔掉/同编号换了身份：停掉旧预览，别让画面冻结在旧设备上。"""
        by_order = {d.order: d for d in self._devices}
        for idx, pane in list(self._previews.items()):
            dev = by_order.get(idx)
            changed = (pane.identity and dev is not None and dev.identity
                       and pane.identity.upper() != dev.identity.upper())
            if dev is None or changed:
                self._full_errors.pop(idx, None)
                # 走统一的停止路径：它会 join 采集线程，避免“看着停了其实还占着相机”。
                self._stop_preview(idx)

    def _on_devices(self, devices: list[cr.DeviceEntry],
                    sig: tuple | None = None) -> None:
        selected = self._selected()
        keep_id = selected.identity if selected is not None else ""
        keep = selected.order if selected is not None else None
        self._devices = list(devices)
        self._device_sig = sig or cr.device_signature(self._devices)
        self._stop_stale_previews()
        self._fill_tree()
        # 优先按身份找回同一台物理相机；PnP 位次变了也不要把高亮挪到别人身上。
        target = next((d for d in self._devices
                       if keep_id and d.identity
                       and d.identity.upper() == keep_id.upper()), None)
        if target is None and keep is not None:
            target = next((d for d in self._devices if d.order == keep), None)
        if target is not None:
            try:
                self.tree.selection_set(str(target.order))
                self.tree.focus(str(target.order))
            except tk.TclError:
                pass
        conflict = self._role_conflict()
        check = self._auto_check_text()
        if conflict:
            self._say(f"{conflict}｜{check}", error=True)
        else:
            self._say(f"自动刷新：{len(self._devices)} 台｜{check}")

    def _role_conflict(self) -> str:
        """两个角色指向同一台设备时返回一句警告，否则空串。

        这是 2026-11-08 用户报「left 和 right 是同一个相机」的直接原因：
        默认选中 #0 时连点「设为左」「设为右」，两个角色都落到 #0，
        而列表只显示第一个角色，界面上看不出任何异常，直到两路画面一模一样。
        `set_role` 现在会摘掉同一台的其它角色，这里负责**把已经坏掉的旧配置
        喊出来**（旧文件不会自己变好）。

        只管 **left 与 right**：这两个角色必须是两台不同的相机，否则双目没有意义。
        `main` 不参与判断 —— 单目和双目共用一台相机是合法配置，不该误报。
        """
        try:
            reg = cr.load_registry()
        except Exception:                                     # noqa: BLE001
            return ""
        devices = self._devices or cr.list_devices()
        seen: dict[str, str] = {}
        for role in ("left", "right"):
            spec = reg.roles.get(role)
            if spec is None:
                continue
            for d in devices:
                if not (spec.identity and d.identity
                        and spec.identity.upper() == d.identity.upper()):
                    continue
                key = f"#{d.order}"
                if key in seen:
                    return (f"⚠ {seen[key]} 和 {role} 是同一条设备"
                            f"（#{d.order} {d.name}）：两路会拿到完全一样的"
                            f"画面。请在列表里选**另一台**，点「设为 {role}」")
                seen[key] = role
                break
        return ""

    def _fill_tree(self) -> None:
        self.tree.delete(*self.tree.get_children())
        try:
            reg = cr.load_registry()
        except Exception:                                     # noqa: BLE001
            reg = cr.Registry()
        # ⚠️ `is_avoided()` / `is_virtual()` 的判据**来自 `settings`**
        # （`avoid_names` / `avoid_name_patterns` / `virtual_names`）。
        # 漏传 settings 就等于用内置默认值判定 —— 用户改了 `config/cameras.json`
        # 也不生效，这正是"对话框忽略用户 avoid 设置"那个 bug。
        settings = reg.settings
        health = {r: cr.role_health(r, reg=reg, devices=self._devices)
                  for r in ("left", "right")}
        for d in self._devices:
            roles = cr.roles_of_device(reg, d)
            is_virt = cr.is_virtual(d.name, settings)
            avoided = cr.is_avoided(d.name, settings)
            if cr.is_device_ignored(d, reg):
                role = "无关"
            elif is_virt:
                # 虚拟摄像头**不参与绑定**：它不是物理设备（PnP 里没有它），
                # 绑上去等于绑一个"换后端就指到别的东西"的编号。
                role = "虚拟(不参与绑定)"
            elif len(roles) > 1:
                # 旧配置可能把同一台绑成了两个角色（set_role 以前不清其它角色）。
                # 必须显式写出来：否则列表只显示第一个角色，看起来一切正常，
                # 而实际两路采集会拿到同一台相机的画面。
                role = "＋".join(sorted(roles)) + " ← 同一台，必须改"
            elif roles:
                role = roles[0]
                h = health.get(role)
                if h is not None:
                    role = f"{role} {'✓' if h.status == 'ok' else '⚠'}"
            elif avoided:
                # 内置/红外这两台是**被物理遮住的**（本机就是这么配的），
                # 点它们预览必然是黑屏。在列表里直接标出来，别让人以为是外接那台。
                role = "内置/红外(多半无画面)"
            else:
                role = "—"
            loc = "、".join(dict.fromkeys(
                cr.location_text(reg.roles[r]) for r in roles
                if r in reg.roles)) if roles else LOCATION_UNSET_TEXT
            caps = [reg.roles[r].capture_index for r in roles
                    if r in reg.roles and reg.roles[r].capture_index is not None]
            cap = ("、".join(f"#{c}" for c in dict.fromkeys(caps))
                   if caps else CAPTURE_UNMEASURED_TEXT)
            self.tree.insert(
                "", "end", iid=str(d.order),
                values=(f"#{d.order}", d.name or "(没名字)",
                        d.port or "—", loc or LOCATION_UNSET_TEXT, cap, role))
        # 2026-11-07 用户质疑「按钮不像真的在执行命令」——复现到了：
        # Treeview 默认**没有任何选中行**，而「预览这台 / 设为左 / 设为右」
        # 全都依赖 `_selected()`，于是点下去只回一句"先在左边选一台设备"，
        # 界面上什么都不会发生。这里默认选中第一台外接相机，让按钮一点就有反应。
        if self._devices and not self.tree.selection():
            first = self._default_entry()
            if first is not None:
                self.tree.selection_set(str(first.order))
                self.tree.focus(str(first.order))
                self.tree.see(str(first.order))

    def _default_entry(self) -> cr.DeviceEntry | None:
        """默认操作对象：**还没绑角色的第一台外接相机**，其次才是任意外接。

        为什么先挑"还没绑角色的"（2026-11-08）：用户的动作永远是"把还没配的那台
        配上"。以前一律选第一台外接，于是打开窗口时高亮的是已经绑好的 left，
        他点「设为右」就落回同一台 —— 这正是「left 和 right 是同一个相机」的来源。

        ⚠️ 虚拟摄像头**绝不**作为默认操作对象（它不能绑）：
        `is_avoided()` / `is_virtual()` 都传 `reg.settings`，否则用户改的
        `config/cameras.json` 在这个下拉里不生效。
        """
        if not self._devices:
            return None
        try:
            reg = cr.load_registry()
        except Exception:                                     # noqa: BLE001
            reg = cr.Registry()
        externals = [d for d in self._devices
                     if not cr.is_avoided(d.name, reg.settings)
                     and not cr.is_virtual(d.name, reg.settings)
                     and not cr.is_device_ignored(d, reg)]
        for d in externals:
            if not cr.roles_of_device(reg, d):
                return d
        if externals:
            return externals[0]
        real = [d for d in self._devices if not cr.is_virtual(d.name, reg.settings)]
        return real[0] if real else None

    def _selected(self) -> cr.DeviceEntry | None:
        sel = self.tree.selection()
        if not sel:
            return None
        try:
            order = int(str(sel[0]))
        except ValueError:
            return None
        return next((d for d in self._devices if d.order == order), None)

    # ------------------------------------------------------------------ #
    # 预览：多路并发
    # ------------------------------------------------------------------ #
    def _start_preview(self) -> None:
        """「预览这台」。**没有选中行也要有反应**：自动选第一台外接再开。

        用户 2026-11-07 的原话是"你不是真的靠按钮来执行的命令"。根因就是这里：
        按钮没错，但它依赖一个用户看不见的前置条件（Treeview 选中行）。默认
        不选中时点下去只有一行小字，看起来就是"按钮坏了"。现在兜底成：
        没选 → 自动选第一台外接 → 直接开预览。
        """
        entry = self._selected()
        if entry is None:
            entry = self._default_entry()
            if entry is None:
                self._say("设备列表是空的：先点「刷新列表」；还是空的就查 USB")
                return
            self.tree.selection_set(str(entry.order))
            self.tree.focus(str(entry.order))
            self.tree.see(str(entry.order))
        self._open_preview(entry)

    def _preview_all(self) -> None:
        """一次把所有**外接**相机都开成独立预览（跳过内置/红外/虚拟/已标无关）。

        ⚠️ 候选判定必须带 `reg.settings`：`is_avoided()` 的规则（`avoid_names` /
        `avoid_name_patterns`）和 `is_virtual()` 的规则（`virtual_names`）都在
        `config/cameras.json` 里。漏传 settings 就是"用户改了配置、对话框不认"。
        """
        try:
            reg = cr.load_registry()
        except Exception:                                     # noqa: BLE001
            reg = cr.Registry()
        picked = [d for d in self._devices
                  if not cr.is_avoided(d.name, reg.settings)
                  and not cr.is_virtual(d.name, reg.settings)
                  and not cr.is_device_ignored(d, reg)]
        if not picked:
            self._say("没找到外接相机（列表里只有内置/红外/虚拟，或已标无关）")
            return
        for d in picked:
            self._open_preview(d)
        self._say(f"已并行打开 {len(picked)} 路预览；每路互不干扰，"
                  "慢的那路继续等首帧，不会拖住别的路")

    def _open_preview(self, entry: cr.DeviceEntry) -> None:
        try:
            reg = cr.load_registry()
        except Exception:                                     # noqa: BLE001
            reg = cr.Registry()
        if cr.is_virtual(entry.name, reg.settings):
            # 虚拟摄像头**不给预览**：它的编号在别的后端里指到的东西完全不确定，
            # 看它的画面没有任何诊断价值，反而会让人以为"这就是那台相机"。
            self._say(f"#{entry.order}「{entry.name}」是虚拟摄像头，不参与绑定、"
                      "也不在这里预览（它只注册在别的后端，PnP 列表里没有它）",
                      error=True)
            return
        old = self._previews.get(entry.order)
        if old is not None:
            # 2026-11-08 用户连点几次都"没反应"的真 bug：上一次失败的卡片还留在
            # `_previews` 里，这里直接 return，**再点永远不会重开**。他只能先
            # 点「停止预览」再点「预览这台」——而界面上没有任何一句提示这件事。
            # 现在：卡片在跑才拦；卡片已失败 → 直接按重试处理。
            if old.failed:
                self._retry_preview(entry)
            else:
                self._say(f"#{entry.order} 已经在预览了；要重开先点卡片上的"
                          "「重试这一路」")
            return
        if cr.is_avoided(entry.name, reg.settings):
            self._say(f"提示：#{entry.order} 是内置/红外相机（通常被遮住，可能一直黑屏）；"
                      "要看外接相机请选名字里带 aoni 的那两台")
        title = entry.name or "(没名字)"
        pane = _PreviewPane(
            self.grid_frame, entry.order, title,
            lambda: self._stop_preview(entry.order),
            on_retry=lambda: self._retry_preview(entry),
            identity=entry.identity)
        pane.status.bind(
            "<Double-1>",
            lambda _e, i=entry.order: self._show_full_error(i))
        self._previews[entry.order] = pane
        self._relayout()
        pane.thread = threading.Thread(
            target=self._preview_worker,
            args=(pane, pane.stop), daemon=True)
        pane.thread.start()
        self._say(f"正在打开 #{entry.order}（首帧最坏约 30 秒）。"
                  "可以同时再开别的相机，彼此不打断。")

    def _show_full_error(self, index: int) -> None:
        """双击卡片状态行看完整原因（卡片里只放一行摘要，不撑高布局）。"""
        from tkinter import messagebox

        full = self._full_errors.get(index)
        messagebox.showinfo(
            "相机预览详细原因",
            full or f"#{index} 目前没有报错", parent=self)

    def _retry_preview(self, entry: cr.DeviceEntry) -> None:
        """关掉这一路再重开：相机常常是"上一路还没放手"而不是真坏。

        ⚠️ 必须等旧 worker **真的退出**再开：`stop.set()` 只是给它一个信号，
        它可能正卡在 `src.read()`（MSMF 一帧能卡 10 秒）。不等就重开，等于
        在同一台设备上叠两条采集会话 —— 这正是把会话打成
        `MF_E_VIDEO_RECORDING_DEVICE_INVALIDATED` 的一种方式（2026-11-07 复盘：
        连续探针/连续点重试都会这样）。最多等 2 秒，等不到也照样重开，
        但会把这点写进日志。
        """
        pane = self._previews.pop(entry.order, None)
        if pane is not None:
            pane.stop.set()
            th = pane.thread
            if th is not None and th.is_alive():
                th.join(timeout=2.0)
                if th.is_alive():
                    self._write_error_log(
                        entry.order,
                        "重试时旧采集线程 2 秒内没退出，仍按新会话重开")
            pane.frame.destroy()
        self._full_errors.pop(entry.order, None)
        self._relayout()
        self._open_preview(entry)

    def _relayout(self) -> None:
        for i, pane in enumerate(self._previews.values()):
            pane.frame.grid(row=i // PANE_COLS, column=i % PANE_COLS,
                            padx=4, pady=4, sticky="nw")
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _stop_preview(self, index: int) -> None:
        pane = self._previews.pop(index, None)
        if pane is None:
            return
        pane.stop.set()
        # 后台 join，不卡界面：等它退出（含 release），期间界面照常响应。
        # 不 join 就没人保证相机真的放手了 —— 下次开同一台会撞会话。
        pane.frame.destroy()
        self._relayout()

        def _wait() -> None:
            th = pane.thread
            if th is not None:
                th.join(timeout=5.0)

        threading.Thread(target=_wait, daemon=True).start()

    def _stop_all(self) -> None:
        for index in list(self._previews):
            self._stop_preview(index)
        self._say("已停止全部预览")

    def _wait_open_slot(self, stop: threading.Event, index: int) -> bool:
        """排队拿「打开相机」的槽位。返回 True = 真的持锁，调用方必须 release。

        **最多等 3 秒**：一台采集会话失效的相机能把一次打开拖 20 秒以上，
        让后来那台一直排队，用户看到的就是"点了全部预览，半天没画面"。
        等不到就先并行试一次（失败还有一次自动重试），坏的那台不许拖死好的。
        """
        if _OPEN_LOCK.acquire(blocking=False):
            return True
        self._events.put(("pstatus", index, "排队等前一路打开（最多 3 秒）…"))
        deadline = time.monotonic() + 3.0
        while not stop.is_set() and time.monotonic() < deadline:
            if _OPEN_LOCK.acquire(timeout=0.2):
                return True
        if not stop.is_set():
            self._events.put(("pstatus", index, "前一路还没开完，先并行试一次…"))
        return False

    def _stream(self, read, pane: _PreviewPane, stop: threading.Event,
                first_frame_timeout: float | None, tag: str = "") -> bool:
        """把帧推到这一路自己的帧槽，直到 stop。返回**是否显示过画面**。

        * 接受全黑帧：预览要回答的是"这台相机现在给什么"，镜头被遮住就该看到
          黑画面加一行说明，而不是被判成"打不开"；
        * `first_frame_timeout` 只约束**第一帧**：到点还没画面就返回 False，
          让调用方换一条路，不让人对着"正在打开"干等。
        """
        import cv2
        from PIL import Image

        shown = False
        dark_noted = False
        frames = 0
        t0 = time.monotonic()
        while not stop.is_set():
            try:
                ok, frame = read()
            except BaseException:                            # noqa: BLE001
                ok, frame = False, None
            if ok and frame is not None:
                if not shown:
                    # 成功也留一行：下次看日志就知道走的是快路还是慢路、几秒出图。
                    self._log_line(
                        f"✔ #{pane.index} {tag} 出图 "
                        f"{frame.shape[1]}x{frame.shape[0]}")
                shown = True
                frames += 1
                # 出帧但几乎全黑：镜头被盖/遮住（内置相机就是这样）。必须说，
                # 否则用户看到一块黑，只会判断成"预览打不开"。
                #
                # 但**不能拿前几帧就定案、也不能定案后永不撤销**（2026-11-16）：
                # 相机出图头几帧常是黑的（自动曝光还在收敛），旧写法一帧就
                # 钉死"几乎全黑"，画面上明明有正常的房间，标签却一直在。
                # 现在先放过预热帧，判黑后只要亮回来就撤销（滞回 8/16）。
                mean = float(frame.mean())
                warm = (frames >= DARK_WARMUP_FRAMES
                        and time.monotonic() - t0 >= DARK_WARMUP_S)
                if warm and not dark_noted and mean < DARK_ENTER:
                    dark_noted = True
                    pane.stage = "画面几乎全黑：镜头可能被遮住/盖着"
                    self._events.put(("pstatus", pane.index, pane.stage))
                elif dark_noted and mean >= DARK_CLEAR:
                    dark_noted = False
                    pane.stage = "预览中"
                    self._events.put(("pstatus", pane.index, pane.stage))
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img.thumbnail((PANE_W, PANE_H))
                # 只写**这一路自己的**帧槽：多路同时推帧时不会互相顶掉。
                try:
                    pane.frames.put_nowait(img)
                except queue.Full:
                    try:
                        pane.frames.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        pane.frames.put_nowait(img)
                    except queue.Full:
                        pass
                time.sleep(0.08)
                continue
            if (not shown and first_frame_timeout is not None
                    and time.monotonic() - t0 > first_frame_timeout):
                break               # 这一档等够了，交给调用方换路
            time.sleep(0.05)
        return shown

    def _diagnose_open_failure(self, wanted: int) -> str:
        """打开失败时扫一遍编号空间，把**事实**写进日志（只 `isOpened`，不读帧）。

        为什么需要（2026-11-08）：同一个 `pythonw.exe`、同一份代码，
        **全新进程能开（1.4 s）、用户那个进程一直 `isOpened=False`**，
        靠猜已经错了四五次。这里只问 OS 一个问题："0–7 号现在谁能打开？"
        `isOpened` 不读帧，所以不会触发坏态那种每次 10 秒的阻塞。
        """
        try:
            import cv2
            from sphere_tracker.sources import INDEX_BACKEND
        except BaseException:                                # noqa: BLE001
            return ""
        lines = [f"诊断 pid={os.getpid()} 线程={threading.current_thread().name} "
                 f"cv2={getattr(cv2, '__version__', '?')} 本次要开 #{wanted}"]
        # 把"这个进程是被谁怎么起来的"一起记下来：2026-11-08 用户实测
        # `run.bat` 能开、`launcher.exe` 不能，两条入口的解释器/控制台/stdio
        # 差别只能在**进程自己**这里看清（`launch.log` 是启动时写的，
        # 这里再记一次，保证失败记录单文件自足）。
        try:
            import ctypes
            lines.append(
                f"  启动方式 exe={os.path.basename(sys.executable)} "
                f"stdout_none={sys.stdout is None} "
                f"console={bool(ctypes.windll.kernel32.GetConsoleWindow())} "
                f"argv={' '.join(sys.argv[:2])}")
        except BaseException:                                # noqa: BLE001
            pass
        # 这次枚举到的编号（**只读枚举**，不开相机）：凡是"能打开但不在这张表里"
        # 的编号都是 phantom —— 它开出来的不是枚举到的物理相机。
        try:
            enumerated = {d.order for d in (self._devices or cr.list_devices())}
        except BaseException:                                # noqa: BLE001
            enumerated = set()
        for i in range(8):
            try:
                cap = cv2.VideoCapture(i, INDEX_BACKEND)
                ok = bool(cap.isOpened())
                cap.release()
            except BaseException as exc:                     # noqa: BLE001
                lines.append(f"  #{i} 抛异常 {type(exc).__name__}: {exc}")
                continue
            lines.append(f"  #{i} isOpened={ok}")
            if ok and i not in enumerated:
                # **phantom 索引命中**：能打开，但 PnP 枚举里没有这个号 ——
                # 它开出来的不是"枚举到的物理相机"（残留会话 / 虚拟设备 /
                # 别的后端编号空间）。必须明着写，否则用户会以为那是他插的那台。
                lines.append("    " + cr.phantom_index_warning(i))
        return "\n".join(lines)

    def _preview_worker(self, pane: _PreviewPane, stop: threading.Event) -> None:
        """一路预览：**只走 `CameraSource` 一条路**（逐档协商），失败就快速报错。

        ⚠ 2026-11-08 第二次改：上一版加过"先直连默认分辨率（快路）再逐档"的两段式，
        **已撤回**。实测代价：

        * 相机处于坏态（`MF_E_VIDEO_RECORDING_DEVICE_INVALIDATED`）时，
          `cap.read()` 每次阻塞 **10 秒**才返回失败，"4 秒快路"实际要等 10–20 秒，
          再叠上逐档的 14 秒，用户对着"正在打开"干等 30 秒 —— 比只用逐档更糟；
        * 一次点击会**多开一次设备**。这台 aoni 的 MSMF 会话本来就容易被反复开关
          打成失效态，多一次开关就是多一次伤害。

        所以回到项目实测过的那条路：`CameraSource`（分辨率梯度 + 首帧超时 + 预算）。
        健康时它本来就 1–2 秒出图（本机实测 1.3 s），快路没有带来任何收益。

        **重试 4 次（1/2/3 秒退避）**：实测这两台 aoni 在上一路关闭后有几十秒的
        "冷却期"，期间 `isOpened()` 直接返回 False（0.0 秒失败）。用户那边看到的
        就是"Windows 相机能开、这个程序打不开"——其实只是上一路还没真正放手。
        多等几次比让人去拔插有效；只有 `DEVICE_INVALIDATED` 不重试（那种要拔插）。
        """
        index = pane.index
        src = None
        last: BaseException | None = None
        try:
            from sphere_tracker.sources import CameraSource, INDEX_BACKEND

            # 预览**只走 INDEX_BACKEND（MSMF）**，不许按索引切 DSHOW ——
            # DSHOW 的编号与这份设备列表不是同一套（AGENTS.md §9），换后端等于
            # 悄悄换了一台设备（2026-11-11：点 #3 的 DSHOW 兜底，开出来的是
            # vivo 办公套件的虚拟摄像头）。
            backend = INDEX_BACKEND

            for attempt in (1, 2, 3, 4):
                if stop.is_set():
                    return
                if attempt == 1:
                    self._events.put(("pstatus", index,
                                      "正在连接（逐档试分辨率）…"))
                else:
                    self._events.put((
                        "pstatus", index,
                        f"相机可能刚被别的程序放开，等它释放"
                        f"（第 {attempt}/4 次）…"))
                held = self._wait_open_slot(stop, index)
                # 拿到槽位（或决定并行试）之后**必须把阶段文案改写一次**。
                # `_wait_open_slot` 会把状态改成"排队等前一路打开（最多 3 秒）…"，
                # 而它一返回，我们就已经在 `CameraSource(...)` 里逐档试了
                # （慢相机要 20 秒以上）。不重写，卡片会一直显示
                # "排队等…（最多 3 秒）… 已等 25 秒" —— 自相矛盾，像卡死
                # （2026-11-16 用户截图：#1 早就出图了，#3 还在"排队"）。
                self._events.put(("pstatus", index, (
                    "正在连接（逐档试分辨率）…" if attempt == 1
                    else f"正在连接（第 {attempt}/4 次）…")))
                if stop.is_set():
                    if held:
                        _OPEN_LOCK.release()
                    return
                try:
                    src = CameraSource(
                        index=index, width=1920, height=1080,
                        backend=backend, optimize=False,
                        search_budget_s=OPEN_BUDGET_S,
                        first_frame_timeout_s=OPEN_FIRST_FRAME_S)
                except BaseException as exc:                 # noqa: BLE001
                    last = exc
                    src = None
                finally:
                    if held:
                        _OPEN_LOCK.release()
                if src is not None:
                    break
                if _is_invalidated(last):
                    # 会话已失效：再开一次设备也救不回来，只会多一次开关。
                    break
                # 退避等它真正放手（1s / 2s / 3s）。
                #
                # 2026-11-08 实测的"冷却期"：另一路（含本程序上一个窗口、别的程序、
                # 或刚才的调试探针）关掉相机之后，这两台 aoni 在**几十秒内**仍然
                # 打不开（isOpened=False，0.0 秒就失败）。用户看到的就是"Windows
                # 相机能开、这个程序打不开"。多等几次比让用户去拔插有效得多。
                time.sleep(float(attempt))

            if src is None:
                detail = str(last) if last is not None else ""
                self._events.put((
                    "perror", index,
                    _short_reason(detail if detail else "打不开"),
                    "\n\n".join(x for x in (
                        detail, self._diagnose_open_failure(index),
                        _NO_DSHOW_HELP, _OPEN_FIX_STEPS) if x),
                    _camera_hogs_text()))
                return

            self._events.put(("pstatus", index, "已连接，等第一帧…"))
            self._stream(lambda: src.read()[:2], pane, stop, None,
                         tag="逐档 MSMF")
        except BaseException as exc:                          # noqa: BLE001
            self._events.put(("perror", index, _short_reason(exc), str(exc),
                              _camera_hogs_text()))
        finally:
            if src is not None:
                try:
                    src.release()
                except BaseException:                         # noqa: BLE001
                    pass

    def _show_frame(self, pane: _PreviewPane, img) -> None:
        try:
            from PIL import ImageTk
            pane.photo = ImageTk.PhotoImage(img)     # 必须留引用，否则被 GC 掉
            pane.label.configure(image=pane.photo, text="")
            if not pane.got:
                pane.got = True
                if "全黑" in pane.stage:
                    pane.status.configure(text=_brief(pane.stage),
                                          foreground="#b45309")
                else:
                    pane.status.configure(text="预览中")
                self._say(f"#{pane.index} 出图了；总共 "
                          f"{sum(1 for p in self._previews.values() if p.got)} "
                          "路画面在同时跑")
        except Exception as exc:                              # noqa: BLE001
            pane.status.configure(text=_brief(f"画面显示不出来：{exc}"))

    # ------------------------------------------------------------------ #
    # 绑定
    # ------------------------------------------------------------------ #
    def _save(self, mutate, note: str = "") -> None:
        # 同 `_start_preview`：没选中行时自动落到第一台外接，别让「设为左」
        # 这类按钮点了只回一句"先选一台"。用户能从列表高亮看出改的是哪台。
        entry = self._selected()
        if entry is None:
            entry = self._default_entry()
            if entry is None:
                self._say("设备列表是空的：先点「刷新列表」")
                return
            self.tree.selection_set(str(entry.order))
            self.tree.focus(str(entry.order))
            self.tree.see(str(entry.order))
        try:
            reg = cr.load_registry()
            mutate(reg, entry)
            path = cr.save_registry(reg)
        except (OSError, ValueError) as exc:
            self._say(f"没保存：{exc}")
            return
        self._fill_tree()
        conflict = self._role_conflict()
        if conflict:
            self._say(conflict, error=True)
        else:
            extra = f"。{note}" if note else ""
            self._say(f"已写入 {path}{extra}")
        if self.on_changed is not None:
            try:
                self.on_changed()
            except Exception:                                 # noqa: BLE001
                pass

    def _save_settings(self) -> None:
        """保存筛选门限（`min_width/min_height/min_fps/min_pair_fps`）。

        只写 `settings`，不碰角色绑定。`min_fps` 现在真的参与 `select_cameras`
        （仅当帧率是**数帧实测**的），`min_pair_fps` 参与 auto 成对协商；
        `min_pair_fps` 留空 = 删除该键并沿用 14.4，不写一个假默认值。
        """
        try:
            width = int(self.var_min_width.get().strip())
            height = int(self.var_min_height.get().strip())
            min_fps = float(self.var_min_fps.get().strip())
            pair_text = self.var_min_pair_fps.get().strip()
            pair_fps = float(pair_text) if pair_text else None
            if width <= 0 or height <= 0 or min_fps <= 0:
                raise ValueError("宽/高/最低 fps 必须 > 0")
            if pair_fps is not None and pair_fps <= 0:
                raise ValueError("成对下限必须 > 0（留空 = 14.4）")
        except ValueError as exc:
            self._say(f"门限没保存：{exc}")
            return
        try:
            reg = cr.load_registry()
            reg.settings["min_width"] = width
            reg.settings["min_height"] = height
            reg.settings["min_fps"] = min_fps
            if pair_fps is None:
                reg.settings.pop("min_pair_fps", None)
            else:
                reg.settings["min_pair_fps"] = pair_fps
            path = cr.save_registry(reg)
        except (OSError, ValueError) as exc:
            self._say(f"门限没保存：{exc}")
            return
        self._say(f"已写入 {path}。筛选门限从下一次自动挑相机/成对协商起生效")
        if self.on_changed is not None:
            try:
                self.on_changed()
            except Exception:                                 # noqa: BLE001
                pass

    def _set_role(self, role: str) -> None:
        entry = self._selected() or self._default_entry()
        if entry is None:
            self._say("设备列表是空的：先点「刷新列表」")
            return
        try:
            reg = cr.load_registry()
        except Exception:                                     # noqa: BLE001
            reg = cr.Registry()
        if cr.is_virtual(entry.name, reg.settings):
            # 数据层 `set_role()` 也会拒绝（ValueError）；这里先给一句人话，
            # 否则用户只会看到「没保存：...」而不知道该怎么办。
            self._say(f"#{entry.order}「{entry.name}」是虚拟摄像头，不能设成 {role} —— "
                      "它不是物理设备（PnP 列表里没有它），请选一台真实的相机",
                      error=True)
            return
        try:
            others = [r for r in cr.roles_of_device(reg, entry) if r != role]
        except Exception:                                     # noqa: BLE001
            others = []

        def do(reg, entry):
            cr.unmark_ignored(reg, entry=entry)
            # `set_role` 会把同一台的其它角色摘掉（数据层不变量），
            # 所以连点两次「设为左」「设为右」不会留下"两个角色同一台"。
            # ⚠️ 这里**不传 capture_index**：本对话框没有"实测过这个号"的证据
            # （预览开的号是 PnP 位次，不能证明它就是这台设备）。实测只能由
            # `cameras.py identify` 写，写错的号比没有号更危险。
            cr.set_role(reg, role, entry=entry,
                        verified=bool(entry.identity))

        note = f"#{entry.order} 现在是 {role}"
        if others:
            note += (f"；它原来是 {'、'.join(sorted(others))}，已一并摘掉"
                     f"（一台相机不能同时当两个角色）")
        self._save(do, note=note)
        # 先把选中行挪到下一台没绑的（用户下一步就是设右边）。
        self._select_next_unbound()

    def _select_next_unbound(self) -> None:
        """绑完一个角色，自动把选中行挪到「还没绑角色的外接相机」。

        用户的实际动作就是"设完左边，接着设右边"。不自动挪行，第二次点击还会
        落在同一台设备上 —— 那正是「left 和 right 是同一个相机」的由来。
        """
        try:
            reg = cr.load_registry()
        except Exception:                                     # noqa: BLE001
            return
        for d in self._devices:
            if (cr.is_avoided(d.name, reg.settings)
                    or cr.is_virtual(d.name, reg.settings)
                    or cr.is_device_ignored(d, reg)):
                continue
            if cr.roles_of_device(reg, d):
                continue
            iid = str(d.order)
            if self.tree.exists(iid):
                self.tree.selection_set(iid)
                self.tree.focus(iid)
                self.tree.see(iid)
            return

    def _mark_ignored(self) -> None:
        self._save(lambda reg, entry: cr.mark_ignored(
            reg, entry=entry, verified=bool(entry.identity)))

    def _unmark_ignored(self) -> None:
        self._save(lambda reg, entry: cr.unmark_ignored(reg, entry=entry))

    # ------------------------------------------------------------------ #
    def _close(self) -> None:
        if _SINGLETON.get("win") is self:
            _SINGLETON["win"] = None
        panes = list(self._previews.values())
        for pane in panes:
            pane.stop.set()
        self._previews.clear()
        # 关窗也要等采集线程退出（最多每路 0.8 秒）：不等就关，线程还在读，
        # 下次打开同一台会撞会话 —— 这是"越点越打不开"的另一个来源。
        stuck = []
        for pane in panes:
            th = pane.thread
            if th is not None and th.is_alive():
                th.join(timeout=0.8)
                if th.is_alive():
                    stuck.append(pane.index)
        if stuck:
            # 只是信息，不是故障：线程是 daemon，进程退出时一定回收；这里记下来
            # 是为了下次排查"重开同一台撞会话"时能看到上一路是什么时候放手的。
            self._log_line(f"关窗时这些采集线程还在退出：{stuck}")
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except tk.TclError:
                pass
        self.destroy()


#: 进程级单例：整个程序**只能有一个**相机管理窗口。
#:
#: 2026-11-08 用户截图两台都打不开、拔插多次也没用 —— 根因是单目窗和平行/大角度
#: 双目窗各自都能点「换相机…」，三个入口**各开一个对话框**，每个对话框里都有
#: 自己的预览 worker，同时抢同一台相机。门厅那份"单例检查"查的是
#: `app.winfo_children()`，而对话框挂在模式窗下面，根本查不到，等于没查。
#: 拔插后瞬间又被剩下的 worker 抢回去，所以"拔插也没用"。
_SINGLETON: dict = {"win": None}


def open_cameras(master, on_changed=None) -> CamerasDialog:
    """三个入口（门厅 / 单目 / 双目）共用**同一个**相机管理窗口。"""
    win = _SINGLETON.get("win")
    if win is not None:
        try:
            alive = bool(win.winfo_exists())
        except tk.TclError:
            alive = False
        if alive:
            win.deiconify()
            win.lift()
            win.focus_force()
            if on_changed is not None:
                win.on_changed = on_changed
            win._say("相机管理已经开着了：同一个程序只留一个，避免两处同时抢相机")
            return win
    win = CamerasDialog(master, on_changed=on_changed)
    _SINGLETON["win"] = win
    return win
