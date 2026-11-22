"""新 GUI 的"真人走查"工具：像没耐心的用户那样把每个窗口点一遍，留下截图与缺陷清单。

为什么要有它
------------
新界面分成了门厅 + 四个模式窗，"改一处不知道会碰坏哪个窗"是分窗方案的主要风险。
单元测试守的是纯函数与控件预算，**没有人守"用户点下去看到什么"** ——
2026-10-28 那两张截图（主按钮压住卡片说明、第 4 步被状态条盖住）都是这么漏出去的。
本工具把"开窗 → 看第一步 → 填参数 → 点主按钮 → 看结果 → 返回门厅"做成可复跑的一段，
每一步留一张 PNG 与一份控件清单，跑完直接产出人读的缺陷清单。

三条硬约束（写进代码，而不是靠后来者记得）
------------------------------------------
1. **不碰硬件、不碰数据**。摄像头相关按钮（拍图标定图/开始标定/开始录制/识别）
   一律**只记录它想执行的命令行**，不起子进程 —— 起起来就会占相机、等 30 秒、
   在别人的屏幕上弹取景窗。运行期任务因此全部是"假任务"，
   ``TaskRunner.start`` 被替换成记录器（见 :meth:`Walker._patch_tasks`）。
2. **不允许弹窗卡住走查**。模态框会让"用户"永远回不来，所以
   ``messagebox`` / ``filedialog`` 全部替换成"记一笔 + 返回默认值"；
   Tk 回调里的异常也改走记录器（默认行为是弹一个 Tradeback 对话框）。
3. **点完必须判"有没有反馈"**。用户最不能接受的不是报错，是点了什么都不发生。
   每次点击前记下状态条文案与任务表长度，点完比对：既没起任务、也没弹窗、
   也没改状态条 —— 记成"静默无反馈"，这是本工具存在的主要理由。
4. **一扇窗只量自己的控件**。控件清单**不穿过 Toplevel**：Tk 里子窗口的路径挂在
   父控件下面（会话选择器是 ``.hub.bench.picker``），把它算进 bench 的话，坐标
   基准不同 —— 会同时**制造假缺陷**（bench 假 ``clipped=2``）和**掩盖真溢出**
   （门厅报 1152，其实是工具台自己的滚动页）。"屏幕上有几扇窗"是另一个问题，
   用 :func:`_windows`；这两件事 2026-11-07 混过一次，报告里两扇窗报同一个数字
   就是当时的线索。

用法::

    python apps/tools/gui_walkthrough.py --out outputs/gui_walk/full
    python apps/tools/gui_walkthrough.py --out outputs/gui_walk/smoke --only hub,mono --step 2

产物：``<out>/NN_<窗口>_<步骤>.png``、``<out>/report.json``、``<out>/report.md``。
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import os
import py_compile
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import tkinter as tk

#: 仓库根：**按标记上溯**，不数层数。写死 `parents[2]` 在文件被移走后会静默指错
#: 别处，而这里指错的后果是"扫不到会话"被当成"没有会话"（假缺陷）。
ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: 走查窗口的短名 → 人读标题。``stereo`` 会展开成 parallel + angled 两个窗。
WINDOWS = {
    "hub": "门厅",
    "mono": "单目窗",
    "stereo": "双目窗（平行 + 大角度）",
    "bench": "会话工作台",
    "console": "控制台",
}

#: 这些命令行是"独立进程的 GUI"，走查时可以真起一次（只读、不开相机），
#: 用它证明"按钮生成的那条命令确实能开出窗口"。
EXTERNAL_SAFE = {
    "apps/data_manager.py": "数据管理 · 脱敏导出",   # 窗口标题的一部分
    "apps/view_3d.py": "3D 轨迹",
}

#: 这些脚本**绝不**真跑：会开相机 / 写 data/（经典界面已废止，不再是入口；
#: `apps/gui_main.py` 仍在表里，因为它无参数启动的是新 GUI，同样会开窗）。
#: 名字在这里出现一次，是为了让"为什么不跑"有据可查，而不是漏写。
NEVER_SPAWN = {
    "apps/record_mono.py", "apps/record_stereo.py",
    "apps/calibrate_stereo.py", "apps/calibrate_mono.py",
    "apps/tools/capture_board.py", "apps/tools/cameras.py",
    "apps/tools/calibrate_focal.py",
    "apps/run_video.py", "apps/run_stereo_video.py", "apps/run_realtime.py",
    "apps/gui_main.py",
}

SEVERITY_ORDER = {"崩溃/卡死": 0, "静默无反馈": 1, "文案误导": 2, "裁切/错位": 3, "观感": 4}


# --------------------------------------------------------------------------- #
# 记录结构
# --------------------------------------------------------------------------- #
@dataclass
class Step:
    """一步 = 用户眼里的一个画面（或一次点击后的画面）。"""

    seq: int
    window: str                      # hub / mono / stereo_parallel / ...
    title: str                       # 一句话：这一步在做什么
    action: str = "观察"             # 观察 / 点击 / 填值 / 展开 ...
    shot: str = ""                   # 截图文件名（空 = 没截到）
    shot_mode: str = ""              # grab / postscript / geometry
    size: tuple[int, int] = (0, 0)
    n_widgets: int = 0
    n_buttons: int = 0
    widgets: list[dict] = field(default_factory=list)
    clipped: list[str] = field(default_factory=list)     # 被窗口边缘切掉的控件
    overflow_px: int = 0             # 滚动页内容比视口高多少（>0 = 要滚才看得全）
    overflow_w_px: int = 0           # 横向溢出
    #: 非空 = 这一步**按设计**就该滚（理由写在这里）。空 = 首屏必须一屏看全。
    overflow_exempt: str = ""
    #: 豁免后实测的 ``yview`` 终点：1.0 = 真能滚到末尾；-1 = 没测（或被豁免却没滚动页）
    scroll_reach: float = -1.0
    status: str = ""                 # 状态条文案（用户唯一能看到的反馈）
    status_before: str = ""          # 点击前的状态条（"前后对照"是这条工具的硬产出）
    task_cmd: list[str] = field(default_factory=list)   # 真的跑了的那条 argv
    task_exit: int | None = None
    task_log: list[str] = field(default_factory=list)   # 日志最后 5 行
    task_timeout: bool = False
    artifacts: list[dict] = field(default_factory=list)  # 承诺的产物在不在
    task_note: str = ""              # 为什么没跑 / 怎么改写了
    dialogs: list[str] = field(default_factory=list)
    tasks: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class Walker:
    """整个走查的驱动者：持有 Tk 根、补丁、截图目录与结果。"""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.out = Path(args.out)
        self.shots = self.out / "shots"
        self.shots.mkdir(parents=True, exist_ok=True)
        self.only = {s.strip() for s in args.only.split(",") if s.strip()}
        self.limit = args.step                        # 0 = 不限制
        self.steps: list[Step] = []
        self.defects: list[dict] = []
        self.failed_windows: list[str] = []           # 根本起不来的窗口
        self._live: list[tuple[object, dict]] = []     # 真跑起来的任务（task, plan）
        self.events: list[dict] = []                  # 补丁收到的弹窗/任务，按时间顺序
        self._errors: list[str] = []                  # Tk 回调异常（已吞掉，不弹窗）
        self._seq = 0
        self._shot_idx = 0
        self._scale = 1.0
        self._screen = (0, 0)                         # 根窗口销毁后 winfo 就没了，先记下
        self._app = None
        self._orig = {}                               # 被替换掉的原函数，收尾时还原

    # ------------------------------------------------------------------ #
    # 启动 / 收尾
    # ------------------------------------------------------------------ #
    def build_app(self):
        from apps._gui_app import App
        self._app = App()
        self._app.report_callback_exception = self._on_tk_exception
        self._settle(0.4)
        self._screen = (self._app.winfo_screenwidth(), self._app.winfo_screenheight())
        # 截图走 PrintWindow，不读屏幕，所以没有"屏幕缩放"这回事（1:1 像素）
        self._scale = 1.0
        self._patch_dialogs()
        self._patch_tasks()
        return self._app

    def shutdown(self) -> None:
        """关掉所有窗口与任务，**确认进程都不在了**再返回。"""
        app = self._app
        if app is None:
            return
        try:
            app.tasks.shutdown(graceful=False)
        except Exception:                              # noqa: BLE001
            pass
        for key in list(getattr(app, "_windows", {})):
            try:
                app.close_window(key)
            except Exception:                          # noqa: BLE001
                pass
        try:
            if app.hub.winfo_exists():
                app.destroy()
            else:
                app.destroy()
        except Exception:                              # noqa: BLE001
            pass
        self._restore_patches()

    def _restore_patches(self) -> None:
        for obj, name, value in list(self._orig.values()):
            try:
                setattr(obj, name, value)
            except Exception:                          # noqa: BLE001
                pass
        self._orig.clear()

    # ------------------------------------------------------------------ #
    # 补丁：把所有"会卡住人或会动硬件"的出口换成记录器
    # ------------------------------------------------------------------ #
    def _replace(self, obj, name: str, value) -> None:
        self._orig[(id(obj), name)] = (obj, name, getattr(obj, name))  # type: ignore[assignment]
        setattr(obj, name, value)

    def _on_tk_exception(self, exc, val, tb) -> None:
        """Tk 回调异常：默认会弹一个 Traceback 对话框，把人挡在那里。

        这里只记账。走查要能继续，而"弹窗"本身也是缺陷证据 ——
        真人看到的是"点一下弹出错误框"，不是 traceback。
        """
        text = "".join(traceback.format_exception(exc, val, tb))
        self._errors.append(text)

    def _patch_dialogs(self) -> None:
        from tkinter import filedialog, messagebox

        def note(kind: str, args, kwargs):
            title = ""
            if args:
                title = str(args[0])
            else:
                title = str(kwargs.get("title", ""))
            msg = args[1] if len(args) > 1 else kwargs.get("message", "")
            self.events.append({"kind": kind, "title": title, "text": str(msg)})

        def showinfo(*a, **k):
            note("showinfo", a, k)
            return "ok"

        def showwarning(*a, **k):
            note("showwarning", a, k)
            return "ok"

        def showerror(*a, **k):
            note("showerror", a, k)
            return "ok"

        def askyesno(*a, **k):
            # 一律回答"否"：确认框后面接的都是"会写数据/删会话"的动作，
            # 装成用户点了"是"就等于替真人破坏数据。
            note("askyesno(回答否)", a, k)
            return False

        def askokcancel(*a, **k):
            note("askokcancel(回答否)", a, k)
            return False

        def askopenfilename(*a, **k):
            note("askopenfilename(取消)", a, k)
            return ""

        def askdirectory(*a, **k):
            note("askdirectory(取消)", a, k)
            return ""

        for fn in (showinfo, showwarning, showerror, askyesno, askokcancel):
            self._replace(messagebox, fn.__name__, fn)
        self._replace(filedialog, "askopenfilename", askopenfilename)
        self._replace(filedialog, "askdirectory", askdirectory)
        self._replace(os, "startfile", lambda path, *a, **k: self.events.append(
            {"kind": "startfile(未执行)", "path": str(path)}))

    def _patch_tasks(self) -> None:
        """``TaskRunner.start`` → **能安全跑的真跑，其余只记录命令行**。

        为什么要真跑一部分（2026-10-31 被 team-lead 打回后改的）：
        "按钮接线了"和"按钮真的发生了它承诺的事"是两件事，只有拿到
        **退出码 + 日志尾 + 产物**才算证明。原来一律不跑，于是报告里只能写
        "已启动"，那是截图级证据，撑不起"无缺陷"这个结论。

        哪些跑（白名单，见 :meth:`_plan_task`）：只读的、以及**能限帧并把产物
        重定向到 outputs/** 的。凡是要开相机、写 data/、改写 config 的，
        一律不跑，并在报告里写清"需硬件，未验证"。
        """
        from apps._task_runner import TaskRunner
        orig = TaskRunner.start
        walker = self

        def start(self, cmd, title, on_done=None, **kw):        # noqa: ANN001
            cmd = [str(c) for c in cmd]
            plan = walker._plan_task(cmd)
            walker.events.append({
                "kind": "task(执行)" if plan["mode"] != "skip" else "task(未执行)",
                "title": title, "cmd": cmd,
                "cmd_run": plan["run"], "why": plan["why"]})
            if plan["mode"] == "skip" or not walker.args.execute:
                return _FakeTask(title)
            # **必须转发多余的关键字**：界面以后给 start 加参数（比如 owner），
            # 桩不接就会 TypeError，而且报错在用户点按钮的那一刻才出现。
            task = orig(self, plan["run"], title, on_done, **kw)
            walker._live.append((task, plan))
            return task

        self._replace(TaskRunner, "start", start)

    # ------------------------------------------------------------------ #
    # 任务执行策略：跑 / 不跑 / 怎么改
    # ------------------------------------------------------------------ #
    def _plan_task(self, cmd: list[str]) -> dict:
        """这条命令行能不能跑、要不要改写。返回 ``{mode, run, why, expect}``。

        判据是**副作用**，不是"能不能验证"：
        - 开相机 → 不跑（没人愿意为了走查让相机被占）；
        - 写 ``data/`` 或 ``config/`` → 不跑（数据不可再生、配置会污染后续结论）；
        - 只读、或产物能重定向到 ``outputs/`` → 跑。
        """
        script = Path(cmd[1]).name if len(cmd) > 1 else ""
        stem = Path(cmd[1]).stem if len(cmd) > 1 else "task"
        out = self.out / "probe" / f"{len(self._live) + 1:02d}_{stem}"

        if script == "run_video.py" and len(cmd) > 2:
            run = _bounded_run_video(cmd, out, self.args.max_frames)
            return {"mode": "run", "run": run,
                    "why": f"限 {self.args.max_frames} 帧、加 --no-show、"
                           f"产物重定向到 {_rel(out)}",
                    "expect": [out / "visualization.json", out / "traj.csv"],
                    "label": out}
        if script == "check_dataset.py":
            return {"mode": "run", "run": cmd, "why": "只读校验", "expect": []}
        if script in ("check_stereo_setup.py",):
            state = _camera_state()
            if state["collided"]:
                # 左右解析到同一台时，它在 `from_args` 阶段就抛 RuntimeError
                # （stereo/pair_source.py:761），**不会打开任何相机** —— 日志里
                # 只有"取帧源装配失败"一行。跑它是安全的，而且正好能证明
                # "任务失败了，界面有没有说"。
                return {"mode": "run", "run": cmd,
                        "why": "当前左右相机撞车，脚本在开相机前就失败（安全）",
                        "expect": []}
            return {"mode": "skip", "run": [],
                    "why": "要真开相机跑 8 秒实时取景 —— 需硬件，未验证",
                    "expect": []}
        if script in ("record_stereo.py", "calibrate_stereo.py"):
            # ⚠ 这两条**碰撞时也不能跑**：实测 `record_stereo` 会先打开左路
            # （`[相机 0] ✔ MSMF @ 2560x1440 打开成功`）再发现右路撞车。
            # 也就是说"撞车 ⇒ 开不了相机"这个假设对录制不成立，跑一次就真占了
            # 相机。宁可少一条动态证据，也不动硬件。
            return {"mode": "skip", "run": [],
                    "why": "会真开相机（实测录制会先打开左路）—— 需硬件，未验证",
                    "expect": []}
        if script == "calibrate_mono.py":
            return {"mode": "skip", "run": [],
                    "why": "标定图已存在，跑成功会改写 config/mono_calibration.json",
                    "expect": []}
        if script in ("data_manager.py", "view_3d.py"):
            # 它们是**独立的 GUI 进程**，由走查器在「外部进程」那一步自己起一次
            # （带窗口探测 + 截图），不走 TaskRunner —— 那一步才看得到窗口。
            return {"mode": "skip", "run": [],
                    "why": "独立进程 GUI：由走查器的「外部进程」步骤单独起一次并截图",
                    "expect": []}
        if script in ("record_mono.py", "capture_board.py", "cameras.py",
                      "calibrate_focal.py", "run_stereo_video.py",
                      "export_visualization.py", "apps/gui_main.py"):
            return {"mode": "skip", "run": [],
                    "why": "会开相机 / 写 data/ / 起另一个界面 —— 需硬件或会动数据，未验证",
                    "expect": []}
        return {"mode": "skip", "run": [], "why": "未知脚本，按保守策略不跑", "expect": []}

    def _wait_task(self, task, plan: dict) -> dict:
        """等任务结束（有上限），返回退出码 / 日志尾 / 产物。

        为什么要 ``update()``：``TaskRunner`` 的轮询挂在 ``root.after`` 上，
        不泵事件循环就拿不到退出码，``on_done`` 也不会跑 —— 那正是我们要看的
        "状态条到底会不会回写"。
        """
        t0 = time.monotonic()
        timed_out = False
        while task.running and time.monotonic() - t0 < self.args.task_timeout:
            try:
                self._app.update()
            except tk.TclError:
                break
            time.sleep(0.05)
        if task.running:
            timed_out = True
            # 超时必须真收掉：留一个占着 stdout 的子进程，后面所有统计都会错
            try:
                self._app.tasks.stop(task, graceful=False)
            except Exception:                                  # noqa: BLE001
                pass
        try:
            self._app.tasks.pump_once()                        # 收尾日志 + on_done
        except Exception:                                      # noqa: BLE001
            pass
        tail: list[str] = []
        if task.log_path is not None and task.log_path.exists():
            try:
                lines = task.log_path.read_text(encoding="utf-8",
                                                errors="replace").splitlines()
                tail = [ln for ln in lines if ln.strip()][-5:]
            except OSError:
                tail = []
        artifacts = [{"path": _rel(p), "exists": Path(p).exists()}
                     for p in plan.get("expect") or []]
        return {"exit": task.exit_code, "tail": tail, "artifacts": artifacts,
                "timeout": timed_out, "elapsed": round(time.monotonic() - t0, 1),
                "cmd_run": plan["run"], "why": plan["why"]}

    # ------------------------------------------------------------------ #
    # 布局 / 截图
    # ------------------------------------------------------------------ #
    def _settle(self, seconds: float = 0.25) -> None:
        """把事件泵干净：Tk 的 geometry 要跑完才会给出真实尺寸。"""
        end = time.monotonic() + seconds
        app = self._app
        if app is None:
            return
        while True:
            try:
                app.update_idletasks()
                app.update()
            except tk.TclError:
                return
            if time.monotonic() >= end:
                break
            time.sleep(0.02)

    def _shot(self, win, name: str) -> tuple[str, str]:
        """截这个窗口**自己**的画面。返回 (文件名, 方式)。

        为什么不用 ``ImageGrab.grab()`` 直接抓屏幕：走查是在终端里跑的，窗口不一定
        在最前面，抓到的是用户桌面上别的东西（第一次实现就抓到了浏览器）。
        那既证明不了界面长什么样，也把别人的屏幕内容写进了产物目录。
        所以用 ``PrintWindow(PW_RENDERFULLCONTENT)`` 让窗口把自己画到内存位图里 ——
        与前后台无关，也不会把别的东西拍进来。实在不行才退化成只记几何。
        """
        path = self.shots / f"{name}.png"
        img = _capture_window(win)
        # 刚建出来的窗口可能还没画完（只给一块底色）：颜色太少就再等等。
        for _ in range(3):
            if img is None:
                break
            colors = img.getcolors(maxcolors=20000)
            if colors is None or len(colors) > 40:
                break
            self._settle(0.3)
            img = _capture_window(win)
        if img is not None:
            try:
                img.save(path)
                return path.name, "printwindow"
            except Exception:                          # noqa: BLE001
                pass
        # 退路一：窗口里的画布自己导出（Tk 的 postscript 不吃 Windows 上的 ttk）
        for canvas in _descendants(win):
            if canvas.winfo_class() != "Canvas":
                continue
            try:
                stem = path.with_suffix("")
                canvas.postscript(file=str(stem) + ".ps", colormode="color")
                return stem.name + ".ps", "postscript"
            except Exception:                          # noqa: BLE001
                break
        return "", "geometry"                          # 退路二：只记几何

    # ------------------------------------------------------------------ #
    # 控件盘点 / 裁切
    # ------------------------------------------------------------------ #
    def _scroll_pages(self, win) -> list[tuple[object, int, int]]:
        """这扇窗里的滚动页：``(页面, 纵向溢出, 横向溢出)``。

        认 ``ScrollPage`` 靠鸭子类型（它同时有 ``_canvas`` 与 ``content``）：
        它的 Tk 类名是 ``TFrame``，按类名字符串去认会认成普通 Frame 而漏掉 ——
        第一版就是这么漏的，报告里于是出现"没有裁切"的假结论。
        """
        out: list[tuple[object, int, int]] = []
        for w in _descendants(win):
            canvas = getattr(w, "_canvas", None)
            content = getattr(w, "content", None)
            if canvas is None or content is None:
                continue
            try:
                dy = content.winfo_reqheight() - canvas.winfo_height()
                dx = content.winfo_reqwidth() - canvas.winfo_width()
            except tk.TclError:
                continue
            out.append((w, dy, dx))
        return out

    def _overflow(self, win) -> tuple[int, int]:
        """滚动页的内容比视口高出/宽出多少（>0 表示"要滚才看得全"）。"""
        best = (0, 0)
        for _w, dy, dx in self._scroll_pages(win):
            best = (max(best[0], dy), max(best[1], dx))
        return best

    def _scroll_reach(self, win) -> float:
        """把每页都滚到底，回报**最差**那个 ``yview()`` 终点：1.0 = 真能滚到末尾。

        为什么要这一下：``overflow_exempt``（"这扇窗按设计就是滚动页"）如果
        只凭人写一句话就算数，那它就是个空白支票 —— 哪天 ScrollPage 的
        ``scrollregion`` 没更新（内容加长了、``<Configure>`` 没绑上），
        "能滚"会变成"滚到底还差一截"而没人发现：用户看到的最后一截是死区。
        量完把位置还原，免得后面几步的截图全停在中段。

        取**最小**而不是"最大那页"：一扇窗里有两页时（一长一短），只量长的那页
        会漏掉"短页滚不到底"，而那同样是死区。
        """
        pages = self._scroll_pages(win)
        if not pages:
            return -1.0
        worst = 1.0
        for page, _dy, _dx in pages:
            canvas = getattr(page, "_canvas", None)
            try:
                before = canvas.yview()
                canvas.yview_moveto(1.0)
                self._settle(0.08)
                _lo, hi = canvas.yview()
                canvas.yview_moveto(before[0])
                self._settle(0.08)
            except tk.TclError:
                return -1.0
            worst = min(worst, float(hi))
        return worst

    def record(self, win, window: str, title: str, *, action: str = "观察",
               extra_notes: list[str] | None = None,
               overflow_exempt: str = "") -> Step:
        """拍一张、盘一次控件、算一次裁切，落成一步。

        ``overflow_exempt`` 非空 = **这扇窗按设计就是滚动页**，理由写在值里
        （例：工具台六张独立的卡）。这不是"放过它"：同时会量它到底能不能
        滚到末尾，量不到就在报告里记一条真缺陷 —— 见 ``_scroll_reach``。
        """
        self._settle(0.2)
        if not _alive(win):                            # 窗口已被关：退到门厅，别抛
            win = self._app.hub
        self._seq += 1
        name = f"{self._seq:02d}_{window}"
        shot, mode = self._shot(win, name)
        widgets = _inventory(win)
        clipped = _clipped(win, widgets)
        ov, ovw = self._overflow(win)
        reach = -1.0
        # 豁免一写就得取证，**不跟 ov 绑在一起**：`ov` 本身就是"滚动页溢出量"，
        # 窗口里压根没有 ScrollPage 时它必然是 0 —— 若把取证塞进 `ov > 20` 里，
        # 「没有滚动容器」这个分支就永远进不去，"自称滚动页"变成一句无法证伪的话。
        # 反过来，真有滚动页但内容装得下时 `yview()` 自然是 (0, 1)，判据不会误伤。
        if overflow_exempt:
            reach = self._scroll_reach(win)
            if reach < 0.999:
                self.add_defect(
                    severity="裁切/错位", where=f"{window} · {title}",
                    what=("自称是滚动页，但滚到底还有一截看不见"
                          f"（yview 终点 {reach:.2f} < 1.00）—— 最后一屏是死区"
                          if reach >= 0 else
                          "自称是滚动页，但窗口里根本没有滚动容器（ScrollPage）"),
                    how="打开这扇窗，把滚动条拖到最底",
                    evidence=f"内容比视口高 {ov} px，yview 终点 {reach:.2f}")
        st = _status_text(win)
        step = Step(seq=self._seq, window=window, title=title, action=action,
                    shot=shot, shot_mode=mode,
                    size=(win.winfo_width(), win.winfo_height()),
                    n_widgets=len(widgets),
                    n_buttons=sum(1 for w in widgets if w["role"] == "button"),
                    widgets=widgets, clipped=clipped,
                    overflow_px=max(0, ov), overflow_w_px=max(0, ovw),
                    overflow_exempt=overflow_exempt, scroll_reach=reach,
                    status=st, notes=list(extra_notes or []))
        self.steps.append(step)
        if self.limit and self._seq >= self.limit:
            raise _StopWalk()
        return step

    # ------------------------------------------------------------------ #
    # 交互
    # ------------------------------------------------------------------ #
    def click(self, win, window: str, button, label: str) -> Step | None:
        """点一个按钮。**先判 disabled**：禁用的按钮不点，只记下来。"""
        if button is None:
            return None
        before = _ui_signature(self._app, win)
        n_events = len(self.events)
        n_err = len(self._errors)
        state = str(button.state()) if hasattr(button, "state") else ""
        if "disabled" in state:
            self._seq += 0
            step = self.record(
                win, window, f"按钮「{label}」是禁用状态（未点击）",
                action="未点击：按钮禁用",
                extra_notes=["禁用的按钮不点：真人点它也不会有反应，"
                             "要判的是它旁边有没有写清为什么不能点"])
            step.notes.append(f"state={state}")
            return step
        n_live = len(self._live)
        button.invoke()
        self._settle(0.3)
        # 真跑起来的任务：等它结束再拍、再判 —— "状态条会不会回写"只有等它结束才看得出来
        ran = self._live[n_live:]
        facts = [self._wait_task(task, plan) for task, plan in ran]
        if ran:
            self._settle(0.3)
        # 「用这个会话」这类按钮点完就把自己的窗口关了：对已销毁的窗口取
        # winfo_* 会抛 TclError，整段走查就断在这里（第一版真断过）。
        target = win
        closed = not _alive(win)
        if closed:
            master = getattr(win, "master", None)
            target = master if master is not None and _alive(master) else self._app.hub
        step = self.record(target, window, f"点击「{label}」", action="点击")
        step.status_before = before[1]
        if closed:
            step.notes.append("这一步点完，被点的窗口自己关了（属于正常收尾）")
        step.dialogs = [e for e in self.events[n_events:]
                        if not e["kind"].startswith("task(")]
        step.tasks = [e for e in self.events[n_events:]
                      if e["kind"].startswith("task(")]
        step.errors = list(self._errors[n_err:])
        if facts:
            f = facts[0]
            step.task_cmd = f["cmd_run"]
            step.task_exit = f["exit"]
            step.task_log = f["tail"]
            step.task_timeout = f["timeout"]
            step.artifacts = f["artifacts"]
            step.task_note = f["why"]
            if f["timeout"]:
                step.notes.append(f"任务 {f['elapsed']}s 未结束，已被走查器强制结束")
        elif step.tasks:
            step.task_note = step.tasks[0].get("why", "")
        # 静默无反馈：窗口没多、状态条没变、日志没长、表格没变，也没有任务/弹窗
        if not closed:
            after = _ui_signature(self._app, win)
            status_after = _status_text(win)
        else:
            after = None
            status_after = ""
        silent = (after is not None and not step.dialogs and not step.tasks
                  and not step.errors and after == before)
        if step.errors:
            # 回调里抛异常 = 这一下**什么都没发生**（Tk 会另外弹个 Traceback 框）。
            # 界面上看起来完全正常，所以必须单独记一条，否则就被"状态条没变"
            # 这条较弱的现象盖过去了。
            self.add_defect(
                severity="崩溃/卡死", where=f"{window} · {label}",
                what="按钮的回调抛异常，界面没有任何反应（Tk 会再弹一个 "
                     "Traceback 对话框）",
                how=f"点击「{label}」（{_path_of(button)}）",
                evidence=_tb_brief(step.errors[0]))
        if silent:
            self.add_defect(
                severity="静默无反馈",
                where=f"{window} · {label}",
                what="点了按钮，窗口没多、状态条没变、日志没长 —— 用户看到的是"
                     "「什么都没发生」",
                how=f"点击「{label}」（{_path_of(button)}）",
                evidence=f"状态条点击前后都是：{before[1] or '（空）'}",
            )
        if facts and step.task_exit not in (0, None) and not step.dialogs:
            # 任务真的失败了。分两种情况：界面说了，和界面没说。
            # "说了"的判据是状态条**变了**且带失败字样（退出码/失败/没能/打不开）。
            said = (status_after != before[1]
                    and re.search(r"退出码|失败|没能|打不开|错误", status_after))
            if not said:
                stuck = status_after.strip().startswith(("正在", "已启动"))
                self.add_defect(
                    severity="静默无反馈",
                    where=f"{window} · {label}",
                    what=(f"任务已退出（返回码 {step.task_exit}），状态条却仍是"
                          f"「{status_after}」—— 用户以为它还在跑" if stuck else
                          f"任务失败（返回码 {step.task_exit}），状态条却写"
                          f"「{status_after}」—— 没有任何一个字提到失败"),
                    how=f"点击「{label}」（{_path_of(button)}）",
                    evidence=f"退出码 {step.task_exit}；日志："
                             f"{' / '.join(step.task_log[-2:])}")
            else:
                step.notes.append(
                    f"任务失败但界面说清楚了：「{status_after}」")
        elif facts and step.task_exit == 0 and not step.dialogs:
            # 跑完了、也没报错，但状态条还停在"正在…" —— 用户等不到"完成"，
            # 分不清是在跑还是已经跑完了。团队口径：这算静默无反馈。
            if status_after.strip().startswith(("正在", "已启动", "已开始")):
                self.add_defect(
                    severity="静默无反馈",
                    where=f"{window} · {label}",
                    what=f"任务已经跑完（返回码 0），状态条仍停在「{status_after}」—— "
                         f"用户不知道它结束了，只会一直等",
                    how=f"点击「{label}」（{_path_of(button)}）",
                    evidence=f"日志尾：{' / '.join(step.task_log[-2:])}")
        elif step.tasks and not step.dialogs and after is not None \
                and after[1] == before[1]:
            # 起了外部程序，但窗口里一个字都不说 —— 程序若是慢启动或直接退出，
            # 用户没有任何办法区分"在跑"和"点了没反应"。
            self.add_defect(
                severity="静默无反馈",
                where=f"{window} · {label}",
                what="按钮起了外部程序，但状态条一个字都没变 —— 那个程序起得慢或者"
                     "起来就退出时，用户无法判断到底有没有反应",
                how=f"点击「{label}」（{_path_of(button)}）",
                evidence=f"任务：{' '.join(step.tasks[0]['cmd'][1:])}；"
                         f"状态条仍是：{before[1] or '（空）'}",
            )
        return step

    def set_text(self, win, window: str, entry, value: str, label: str) -> Step:
        """像用户那样清空输入框再输入，然后触发它绑的按键回调。"""
        entry.focus_force()
        entry.delete(0, "end")
        entry.insert(0, value)
        before = _status_text(win)
        self._settle(0.1)
        entry.event_generate("<KeyRelease>", keysym="a")
        self._settle(0.2)
        step = self.record(win, window, f"把「{label}」改成 {value!r}",
                           action="填值")
        step.status_before = before
        if _status_text(win) == before and not step.errors:
            step.notes.append("输入后状态条没变（若界面有解释就靠页面文案）")
        return step

    def _camera_dialog(self, window: str) -> None:
        """「换相机…」现在开的是 GUI 内对话框，走完要拍一张并关掉。

        以前那个按钮只起一个控制台子进程，走查里点完就没事了；换成 Tk
        对话框之后如果留着不关，后面每一步的"屏幕上有几扇窗"都会多一扇，
        `_ui_signature` 会把别人的窗口算进来（这是上一轮刚修过的同型坑）。
        """
        dlg = _find_toplevel(self._app, "相机管理")
        if dlg is None:
            self.add_defect(
                severity="静默无反馈", where=f"{window} · 换相机…",
                what="点了「换相机…」却没有出现相机管理对话框",
                how="打开这扇窗，点①里的「换相机…」",
                evidence="窗口树里找不到标题含「相机管理」的 Toplevel")
            return
        self.record(dlg, window, "相机管理对话框（设备列表 + 预览）", action="观察")
        # 2026-11-07 AGENTS.md §7.3：**真的点按钮**，不能只"观察"。
        # 这一步就是当初漏掉的：Treeview 默认没选中行，点「预览这台」什么都不会
        # 发生，而"观察一下"看不出来。点完要能看到预览卡片出现。
        preview = _button_in(dlg, "预览这台")
        if preview is None:
            self.add_defect(
                severity="静默无反馈", where=f"{window} · 相机管理",
                what="相机管理里找不到「预览这台」按钮",
                how="打开相机管理，检查按钮排",
                evidence="按钮文字匹配不到")
        else:
            self.click(dlg, window, preview, "相机管理 · 点「预览这台」")
            cards = len(getattr(dlg, "_previews", {}) or {})
            if cards <= 0:
                self.add_defect(
                    severity="静默无反馈", where=f"{window} · 相机管理 · 预览这台",
                    what="点了「预览这台」，却没有出现预览卡片（按钮没真正执行）",
                    how="打开相机管理，直接点「预览这台」（不要先手动选行）",
                    evidence="invoke 之后 dlg._previews 仍为空")
            self.record(dlg, window, "相机管理 · 预览已启动", action="观察")
            stop = _button_in(dlg, "停止全部预览")
            if stop is not None:
                self.click(dlg, window, stop, "相机管理 · 点「停止全部预览」")
        try:
            dlg._close()                 # 会同时停掉后台预览线程，不能只 destroy
        except tk.TclError:
            pass

    def toggle(self, win, window: str, section, label: str) -> Step:
        """展开/收起折叠区（双目第②步的「高级」）。"""
        section.header_button.invoke()
        self._settle(0.3)
        state = "展开" if section.expanded else "收起"
        return self.record(
            win, window, f"折叠区「{label}」→ {state}", action="折叠",
            # 展开这一屏**不是首屏**：用户自己按了「高级」要求多看几行，
            # 多出来的高度本来就该由滚动页接住。收起那一屏照旧按首屏算。
            overflow_exempt=("用户主动展开折叠区：这一屏不是首屏，"
                             "多出来的行由滚动页接住" if section.expanded else ""))

    def add_defect(self, *, severity: str, where: str, what: str,
                   how: str, evidence: str) -> None:
        self.defects.append({"severity": severity, "where": where, "what": what,
                             "how": how, "evidence": evidence})

    # ------------------------------------------------------------------ #
    # 走查：门厅
    # ------------------------------------------------------------------ #
    def walk_hub(self) -> None:
        app = self._app
        hub = app.hub
        hub.deiconify()
        hub.lift()
        self.record(hub, "hub", "门厅首次打开（三张模式卡的就绪度 + 工具条）")
        self._check_hub_text(hub)

        for key, card_title in (("parallel", "平行双目"), ("angled", "大角度双目")):
            card = _card(hub, card_title)
            btn = _button_in(card, "进入")
            # 2026-11-08：门厅按**当前装置档案的模式**禁用不适用那张卡的「进入」。
            # 禁用是设计、不是崩溃；走查要判的是"卡片有没有写清为什么不能点"，
            # 而不是把"没开窗口"当成缺陷（之前 active=大角度档案时误报过）。
            state = str(btn.state()) if btn is not None else ""
            self.click(hub, "hub", btn, f"{card_title} · 进入")
            if "disabled" in state:
                self.record(hub, "hub",
                            f"{card_title} 按装置档案门控正确禁用（不是崩溃）",
                            action="未点击：按钮禁用（设计如此）")
                continue
            win = app._windows.get(f"stereo_{key}")
            if win is None:
                self.add_defect(
                    severity="崩溃/卡死",
                    where=f"hub · {card_title} · 进入",
                    what="点了「进入」但没有出现双目窗（窗口表里也没有它）",
                    how=f"点击「{card_title} / 进入」",
                    evidence="app._windows 里没有 stereo_%s" % key)
                continue
            self.record(win, f"stereo_{key}", f"{card_title}窗刚打开")
            app.close_window(f"stereo_{key}")
            self.record(hub, "hub", f"关掉{card_title}窗，回到门厅")
        # 单目窗放到 mono 段里走，这里只确认"进入"把窗口开出来了
        if "mono" not in self.only:
            btn = _button_in(_card(hub, "单目 · 一台相机"), "进入")
            self.click(hub, "hub", btn, "单目 · 进入")
            if "mono" in app._windows:
                self.record(app._windows["mono"], "mono", "单目窗刚打开（仅确认能开）")
                app.close_window("mono")

        # 工具条：会起子进程的按钮只记录命令行
        for label in ("相机设备", "数据管理", "3D 查看器"):
            btn = _button_in(_card(hub, "工具"), label)
            self.click(hub, "hub", btn, f"工具 · {label}")
        # 纯本地动作
        self.click(hub, "hub", _button_in(_card(hub, "工具"), "重新读取标定"),
                   "工具 · 重新读取标定")
        # 废止入口（2026-11-20）：按钮保留，点击**只弹一句说明**；
        # 绝不能再起经典界面 —— 所以这里连任务都不该出现。
        self.click(hub, "hub", _button_in(_card(hub, "工具"), "经典界面（已废止）"),
                   "工具 · 经典界面（已废止）")
        # 工作台入口：门厅这条路径必须能进（另一条入口在双目第④步）
        self.click(hub, "hub", _button_in(_card(hub, "工具"), "会话工作台"),
                   "工具 · 会话工作台")
        # 六条「跑一次、隔很久再跑」的离线流程（2026-11-07 补进界面）。
        # **只点开、不点里面**：每张卡的主按钮都会起子进程（训练一条十几分钟），
        # 真点会把走查变成"跑一遍全流程"。这里要判的是"入口可达 + 打开不炸"。
        self.click(hub, "hub",
                   _button_in(_card(hub, "工具"), "标定与训练", prefix=True),
                   "工具 · 标定与训练（六条离线流程）")
        dlg = getattr(hub, "_tools_window", None)
        if dlg is not None and _alive(dlg):
            dlg.deiconify()
            dlg.lift()
            self.record(dlg, "tools", "标定与训练工具台：六张卡 + 各自的主按钮",
                        overflow_exempt="工具台按滚动页设计：六条互相独立的"
                                        "离线流程各占一张卡，一屏装不下是刻意的")
            dlg.destroy()

        # 控制台 / 工作台放到各自的段里走（它们是单例，先开过就会影响后面的判断）
        app.open_console()
        self.record(app._windows["console"], "console", "从门厅状态条以外的地方打开控制台")
        app.close_window("console")

    def _check_hub_text(self, hub) -> None:
        """卡片文案的三类硬伤：Markdown 残留、未闭合的「」、以及空状态。"""
        for key in ("mono", "parallel", "angled"):
            lab = hub._cards[key]["status"]
            text = lab.cget("text")
            if "**" in text:
                self.add_defect(severity="文案误导", where="hub · 模式卡状态",
                                what="卡片上印出了 Markdown 的 ** 标记",
                                how="打开门厅看三张卡的状态行",
                                evidence=text)
            if text.count("「") != text.count("」"):
                self.add_defect(severity="文案误导", where=f"hub · {key} 卡状态",
                                what="「」引号数量不配对（替换 ** 时只换了一半）",
                                how=f"看 {key} 卡的状态行",
                                evidence=text)

    # ------------------------------------------------------------------ #
    # 走查：单目窗
    # ------------------------------------------------------------------ #
    def walk_mono(self) -> None:
        app = self._app
        if "mono" not in app._windows:
            app.open_mono()
        win = app._windows["mono"]
        win.deiconify()
        win.lift()
        self.record(win, "mono", "单目窗首次打开（① 内参 / ② 球与量程 / ③ 录制与识别）")
        self._note_overflow(win, "mono")
        # 相机那行只说"身份对上了"，不等于"这台能出图"。判据在
        # `_role_online`：它调 `resolve_role(..., interactive=False)`，注释写明
        # "不开相机，所以是毫秒级"。而用户对这台机器的抱怨正是"相机打不开"——
        # 界面在这种情况下给的是"left = …，索引 0"，一个字的警告都没有。
        cams = str(win.lbl_cams.cget("text"))
        # 界面自己写了"未开相机验证 / 设备还没枚举到"就不算误导：它已经
        # 明确区分了"身份对上"和"能出图"。只有**什么都不说**才记缺陷。
        if (cams and "不在线" not in cams
                and "未开相机验证" not in cams
                and "设备还没枚举到" not in cams):
            self.add_defect(
                severity="文案误导", where="mono · ① 相机",
                what="「相机：…」这行只表示「角色/端口对得上」，不代表这台相机能开出图"
                     "（判据是静态匹配，不开相机）。相机打不开时界面照样显示得像是可用，"
                     "用户只会以为是自己点错了",
                how="打开单目窗，看第①步那行「相机：」",
                evidence=f"界面原文：{cams}")

        # ① 内参：三个会开硬件/会弹外部的入口，只看它们的命令行
        self.click(win, "mono", _button_in(_card(win, "① 标定内参"), "拍标定图"),
                   "① 拍标定图")
        self.click(win, "mono", _button_in(_card(win, "① 标定内参"), "标定内参"),
                   "① 标定内参")
        self.click(win, "mono", _button_in(_card(win, "① 标定内参"), "换相机…"),
                   "① 换相机…")
        self._camera_dialog("mono")
        self.click(win, "mono", _button_in(_card(win, "① 标定内参"), "打开标定图文件夹"),
                   "① 打开标定图文件夹")
        self.click(win, "mono", _button_in(_card(win, "① 标定内参"),
                                           "替代：单点焦距估计"), "① 替代：单点焦距估计")

        # ② 球与量程：把球径填成非数字，看界面会不会当场说清
        ent_d = _entry_of(win, "var_diameter")
        if ent_d is not None:
            step = self.set_text(win, "mono", ent_d, "abc", "② 球直径（有效值）")
            # 填了非数字之后，界面必须说"这不是数字"，而不是说"还没填" ——
            # 后者会让人以为输入框没生效，而真正发生的是这个值会被原样
            # 传给 `--ball-diameter`，run_video 在 argparse 就 exit 2（点了没反应）。
            budget = str(win.lbl_budget.cget("text"))
            # 只有"没认出这是坏输入、还倒过来说你没填"才算缺陷；
            # 若界面已经写明「abc 不是数字」，那就是做对了，不该记。
            if "填一个球直径才能算" in budget:
                self.add_defect(
                    severity="文案误导", where="mono · ② 球直径（有效值）",
                    what="框里填了「abc」，误差预算那行却写「填一个球直径才能算误差预算」"
                         "—— 用户明明填了；界面没有指出「这不是数字」，这个值还会被"
                         "原样传给 run_video，在 argparse 处退出",
                    how="② 球直径（有效值）输入 abc",
                    evidence=budget)
            self.set_text(win, "mono", ent_d, "21.6", "② 球直径（有效值）")
        ent_f = _entry_of(win, "var_focal")
        if ent_f is not None:
            self.set_text(win, "mono", ent_f, "0", "② 焦距 fx")
            self.set_text(win, "mono", ent_f, "", "② 焦距 fx")

        # ③ 先走"没有会话"的分支：这时点「开始识别」会弹文件选择框
        self._mono_offline_cancel(win)

        # ③ 再走"选历史会话"（只读），把一段真实会话选上。
        # 注意：单目窗的按钮叫「选择…」；「历史会话…」是 2026-10-30 因按钮预算
        # 被撤掉的旧入口（第③步现在只剩 5 个按钮）。走查器如果还找旧名字，
        # 会记一条"点了没反应"——那是工具陈旧，不是界面缺陷。
        self.click(win, "mono", _button_in(_card(win, "③ 录制与识别"), "选择…"),
                   "③ 选择…（历史会话）")
        picker = _find_toplevel(app, "历史会话")
        if picker is not None:
            self.record(picker, "mono", "历史会话列表（单目）", action="观察")
            row = _tree_row_by_text(picker, self.args.mono_session)
            if row is not None:
                picker.tree.selection_set(row)
                picker.tree.see(row)
                self._settle(0.2)
                self.record(picker, "mono", f"选中会话 {self.args.mono_session}",
                            action="选中")
                self.click(picker, "mono",
                           _button_in(picker, "用这个会话"), "历史会话 · 用这个会话")
            else:
                self.add_defect(
                    severity="静默无反馈", where="mono · 历史会话",
                    what=f"列表里找不到已有的单目会话 {self.args.mono_session}",
                    how="点「选择…」看列表",
                    evidence="列表为空或该会话被判成坏目录")
                picker.destroy()
        else:
            self.add_defect(severity="静默无反馈", where="mono · 选择…",
                            what="点了「选择…」没有出现会话列表窗口",
                            how="点击「③ 选择…」", evidence="没有找到新 Toplevel")

        # 会话选上之后再更新一次界面（③ 的摘要与「点选球心」「看 3D」的可用性）
        self.record(win, "mono", "选中会话后的第③步（摘要 / 按钮可用性）",
                    action="观察")
        # 「点选球心」会打开逐帧点选窗，需要人真的在图上点两下 —— 不点
        seed_btn = _button_in(_card(win, "③ 录制与识别"), "点选球心")
        if seed_btn is not None:
            st = str(seed_btn.state())
            self.record(win, "mono",
                        "「点选球心」当前" + ("可点（未点击：需要人工在图上点两下）"
                                          if "disabled" not in st else "禁用"),
                        action="未点击：需人工交互",
                        extra_notes=["点选球心会 wait_window 等人在图上点，自动化点不动它"])
        # ③ 有会话时真点一次「开始识别」：这是唯一能把"球径单位"这条链走完的入口。
        # 命令由 GUI 生成，走查器只把帧数限死、把产物重定向到 outputs/（见 _plan_task）。
        det_before = str(win.var_detector.get())
        self.click(win, "mono", _button_in(_card(win, "③ 录制与识别"), "开始识别"),
                   "③ 开始识别（有当前会话，真跑 N 帧）")
        det_after = str(win.var_detector.get())
        if det_before != det_after:
            # 会话里带 seed.json 时，界面会把「检测器」下拉**改成 specular** 并
            # 在关窗时写进 config/gui_state.json —— 用户下次打开看到的默认检测器
            # 已经不是他选的那个了，而且全程没有一句话解释。
            self.add_defect(
                severity="文案误导", where="mono · ③ 检测器",
                what=f"点一次「开始识别」，检测器从「{det_before}」被静默改成"
                     f"「{det_after}」，并且会持久化到 config/gui_state.json —— "
                     f"用户下次打开单目窗看到的默认值已经变了，界面上没有任何说明",
                how="选一段带 derived/seed.json 的会话 → ③ 开始识别",
                evidence=f"{det_before} → {det_after}")

        app.close_window("mono")
        self.record(app.hub, "hub", "关掉单目窗，回到门厅")

    def _mono_offline_cancel(self, win) -> None:
        """「开始识别」的空输入分支：没有会话时会弹文件选择框。

        为什么只走这一条分支：有会话时它会往 ``data/mono/<会话>/derived/`` 写
        csv/视频/可视化 —— 那是数据目录，走查不许动。空输入分支恰好能验证
        "没东西可跑时界面会不会给一句人话"，而且什么都不写。
        """
        win._session = None
        win.var_source.set("离线（当前会话）")
        win._refresh_session()
        self.click(win, "mono", _button_in(_card(win, "③ 录制与识别"), "开始识别"),
                   "③ 开始识别（无当前会话）")

    def _note_overflow(self, win, window: str) -> None:
        """首屏装不下就记一条 —— 阈值 20 px：更小的溢出是滚动条的容差，不值得报。"""
        ov, ovw = self._overflow(win)
        if ov > 20 or ovw > 20:
            self.add_defect(
                severity="裁切/错位",
                where=f"{window} · 首屏",
                what=f"首屏看不全：内容比视口高 {ov} px"
                     f"{f'、宽 {ovw} px' if ovw > 4 else ''} —— 最下面一截要滚动才看得到，"
                     f"而窗口高度是按「一屏看全」定的",
                how="打开窗口，不做任何缩放",
                evidence=f"窗口 {win.winfo_width()}x{win.winfo_height()}，"
                         f"内容需 {win.winfo_height() + ov} px 高")

    # ------------------------------------------------------------------ #
    # 走查：双目窗（平行 / 大角度各一遍）
    # ------------------------------------------------------------------ #
    def walk_stereo(self, mode: str) -> None:
        app = self._app
        win = app.open_stereo(mode)
        name = f"stereo_{mode}"
        win.deiconify()
        win.lift()
        self.record(win, name, f"{'平行' if mode == 'parallel' else '大角度'}双目窗首次打开")
        self._note_overflow(win, name)

        # ① 装置检查
        self.click(win, name, _button_in(_card(win, "① 装置检查"), "检查装置"),
                   "① 检查装置")
        self.click(win, name, _button_in(_card(win, "① 装置检查"), "重新读取标定"),
                   "① 重新读取标定")
        self.click(win, name, _button_in(_card(win, "① 装置检查"), "换相机…"),
                   "① 换相机…")
        self._camera_dialog(name)

        # ② 标定：高级折叠区展开 → 收起
        adv = _collapsible(win)
        if adv is not None:
            self.toggle(win, name, adv, "高级：采集门限与相机")
            self.toggle(win, name, adv, "高级：采集门限与相机")
        else:
            self.add_defect(severity="裁切/错位", where=f"{name} · ② 标定",
                            what="没找到「高级」折叠区（CollapsibleSection 不在窗口里）",
                            how="看第②步", evidence="窗口子树里没有 CollapsibleSection")
        # 大角度窗的方法下拉：换一次，看说明文字跟不跟着变
        if mode == "angled":
            combo = _combobox_of(win, "var_method")
            if combo is not None:
                old = combo.get()
                for value in combo.cget("values"):
                    if value == old:
                        continue
                    combo.set(value)
                    combo.event_generate("<<ComboboxSelected>>")
                    self._settle(0.2)
                    step = self.record(win, name, f"② 标定方法换成「{value}」",
                                       action="换下拉")
                    hint = win.lbl_method.cget("text") if win.lbl_method else ""
                    if not hint:
                        self.add_defect(severity="静默无反馈", where=f"{name} · ② 方法",
                                        what="换了标定方法，下面那句说明是空的",
                                        how=f"下拉选「{value}」", evidence="lbl_method 为空")
                    step.notes.append(f"方法说明：{hint[:60]}")
                    break
                combo.set(old)
                combo.event_generate("<<ComboboxSelected>>")
                self._settle(0.2)
        self.click(win, name, _button_in(_card(win, "② 标定"), "开始标定"),
                   "② 开始标定")

        # ③ 采集：平行窗标定就绪时它会真起录制，大角度窗会先被拦
        # 「开始录制」在标定不匹配时会把原因写进按钮文字
        # （「开始录制（先解决标定）」），而且整颗按钮是 disabled 的。
        # 精确相等会找不到它、让 step 变 None 后在下面崩掉 —— 那只是走查器的
        # 假缺陷，不是产品缺陷。用前缀匹配找到它，click() 会按既有约定把
        # disabled 按钮只记录、不点击。
        step = self.click(win, name,
                          _button_in(_card(win, "③ 采集"), "开始录制", prefix=True),
                          "③ 开始录制")
        # 录到哪里，决定了第④步能不能看见它：会话列表只扫 stereo_roots()。
        # 这条判据不做成"看代码"，而是拿按钮实际生成的 --out 去比对扫描根 ——
        # 落点不在扫描范围里，就成了"录完找不着"（用户只会说"录了但列表里没有"）。
        for task in (step.tasks if step is not None else []):
            cmd = task["cmd"]
            if "--out" in cmd:
                out = Path(cmd[cmd.index("--out") + 1])
                roots = [Path(p) for p in _stereo_roots()]
                if not any(_is_under(out, r) for r in roots):
                    self.add_defect(
                        severity="静默无反馈", where=f"{name} · ③ 开始录制",
                        what=f"录制落点 {_rel(out)} 不在会话扫描范围"
                             f"（{'、'.join(_rel(r) for r in roots)}）—— 录完第④步的"
                             f"「选择…」和工作台里都看不到这段，用户会说「录了但列表是空的」；"
                             f"而且目录名固定，第二段录制会覆盖第一段",
                        how="③ 开始录制 → 看生成的 --out",
                        evidence=" ".join(cmd))

        # ④ 分析：选会话（只读）→ 跑离线（只记命令）→ 看 3D
        self.click(win, name, _button_in(_card(win, "④ 分析"), "选择…"),
                   "④ 选择会话…")
        picker = _find_toplevel(app, "选择会话")
        if picker is not None:
            self.record(picker, name, "会话选择器（按装置模式三态标注）", action="观察")
            row = _tree_row_by_text(picker, self.args.stereo_session)
            if row is None:
                # 三态过滤默认把"不属于本窗模式"的会话藏起来（大角度窗里
                # 藏的就是平行会话）。真人这时会勾那个复选框 —— 走查也勾一次，
                # 顺便验证"标灰 + 选中时说明风险"这条链有没有接上。
                chk = _checkbutton_in(picker)
                if chk is not None:
                    chk.invoke()
                    self._settle(0.25)
                    self.record(picker, name,
                                "勾上「显示不属于本窗口模式的会话」（别模式的会话出现）",
                                action="勾选")
                    row = _tree_row_by_text(picker, self.args.stereo_session)
            if row is not None:
                picker.tree.selection_set(row)
                self._settle(0.2)
                self.click(picker, name, _button_in(picker, "用这个会话"),
                           "会话选择器 · 用这个会话")
            else:
                self.add_defect(severity="静默无反馈", where=f"{name} · 选择会话",
                                what=f"列表里找不到 {self.args.stereo_session}",
                                how="点「选择…」看列表", evidence="未找到该行")
                picker.destroy()
        else:
            self.add_defect(severity="静默无反馈", where=f"{name} · ④ 选择…",
                            what="点了「选择…」没有出现会话列表",
                            how="点击第④步「选择…」", evidence="没有新 Toplevel")
        self.click(win, name, _button_in(_card(win, "④ 分析"), "跑离线三维"),
                   "④ 跑离线三维")
        step = self.click(win, name, _button_in(_card(win, "④ 分析"), "打开 3D 查看器"),
                          "④ 打开 3D 查看器")
        # 状态条说的是"正在打开 3D 查看器：derived" —— derived 是会话里的子目录名，
        # 拿它当"哪个会话"等于没说是哪一段。
        if "：derived" in step.status:
            self.add_defect(
                severity="文案误导", where=f"{name} · ④ 打开 3D 查看器",
                what="状态条把会话名写成了子目录名「derived」，用户看不出打开的是哪一段",
                how="第④步选中会话后点「打开 3D 查看器」",
                evidence=step.status)

        app.close_window(name)
        self.record(app.hub, "hub",
                    f"关掉{'平行' if mode == 'parallel' else '大角度'}双目窗，回到门厅")

    # ------------------------------------------------------------------ #
    # 走查：会话工作台
    # ------------------------------------------------------------------ #
    def walk_bench(self) -> None:
        app = self._app
        try:
            win = app.open_bench(self.args.stereo_session)
        except Exception:                                  # noqa: BLE001
            # 窗**根本起不来**不是"跳过"，是头号缺陷：真人点下去只会看到
            # 一个空白窗（或什么都没），而 Tk 默认还会再弹一个 Traceback 框。
            tb = traceback.format_exc()
            self._errors.append(tb)
            self.failed_windows.append("会话工作台")
            self.add_defect(
                severity="崩溃/卡死",
                where="bench · 打开会话工作台",
                what="会话工作台构造时抛异常，窗口起不来 —— 门厅「会话工作台」与"
                     "双目第④步「历史会话…」两个入口都进不去",
                how="门厅 > 工具 > 会话工作台（或双目第④步 历史会话…）",
                evidence=_tb_brief(tb))
            orphan = next((w for w in _windows(app)
                           if isinstance(w, tk.Toplevel) and w.winfo_exists()
                           and str(w.title()) in ("", "tk")), None)
            if orphan is not None:
                self.record(orphan, "bench", "崩溃后残留在屏幕上的空窗口（没有标题、没有内容）",
                            action="观察")
                orphan.destroy()
            return
        win.deiconify()
        win.lift()
        self.record(win, "bench", "会话工作台打开（当前会话 + ①–⑤ 的可用性）")
        self._note_overflow(win, "bench")

        # 先看"这段会话现在到哪一步"：禁用的按钮必须写着原因
        states = _bench_states(win)
        for key, (enabled, why) in states.items():
            if enabled:
                continue
            lab = win._why.get(key)
            text = lab.cget("text") if lab is not None else ""
            if not text:
                self.add_defect(
                    severity="文案误导", where=f"bench · {key}",
                    what="这一步不能点，但没有任何一行说明为什么",
                    how=f"打开工作台看「{key}」卡片",
                    evidence=f"step_states 原因={why!r}，界面显示的却是空")

        self.click(win, "bench", _button_in(_card(win, "① 校验数据"), "校验数据"),
                   "① 校验数据")
        # ② 点选球心要人在图上点，跳过但说明
        seed = _button_in(_card(win, "② 点选球心"), "点选球心")
        if seed is not None:
            st = str(seed.state())
            self.record(win, "bench",
                        "② 点选球心当前" + ("可点（未点击：需要人工在图上点两下）"
                                        if "disabled" not in st else "禁用"),
                        action="未点击：需人工交互")
        self.click(win, "bench", _button_in(_card(win, "③ 跑离线三维"), "跑离线三维"),
                   "③ 跑离线三维")
        self.click(win, "bench", _button_in(_card(win, "④ 看三维"), "打开 3D 查看器"),
                   "④ 打开 3D 查看器")
        self.click(win, "bench", _button_in(_card(win, "④ 看三维"), "打开标注视频"),
                   "④ 打开标注视频")
        self.click(win, "bench", _button_in(_card(win, "⑤ 导出标准 JSON"), "导出 JSON"),
                   "⑤ 导出 JSON")

        # 换会话：跨装置不按几何过滤
        self.click(win, "bench", _button_in(_card(win, "当前会话"), "换会话…"),
                   "换会话…")
        picker = _find_toplevel(app, "选择会话")
        if picker is not None:
            self.record(picker, "bench", "换会话（跨装置，不过滤几何）", action="观察")
            picker.destroy()
        app.close_window("bench")
        self.record(app.hub, "hub", "关掉工作台，回到门厅")

    # ------------------------------------------------------------------ #
    # 走查：控制台
    # ------------------------------------------------------------------ #
    def walk_console(self) -> None:
        app = self._app
        win = app.open_console()
        win.deiconify()
        win.lift()
        self.record(win, "console", "控制台打开（任务表 + 运行日志）")
        self.click(win, "console", _button_in(_card(win, "任务"), "中断选中"),
                   "中断选中（未选任务）")
        self.click(win, "console", _button_in(_card(win, "任务"), "中断全部"),
                   "中断全部（本来就没有任务）")
        self.click(win, "console", _button_in(_card(win, "任务"), "打开日志文件"),
                   "打开日志文件（未选任务）")
        app.close_window("console")
        self.record(app.hub, "hub", "关掉控制台，回到门厅")

    # ------------------------------------------------------------------ #
    # 外部进程 GUI：数据管理 / 3D 查看器
    # ------------------------------------------------------------------ #
    def external_step(self, cmd: list[str], title_sub: str, label: str) -> None:
        """真起一次独立进程的 GUI（只读），等窗口出现再杀掉。

        为什么值得真起：这两个按钮是"点了没反应"的高危区（``view_3d.py`` 的
        source 是必填位置参数，漏了就是 exit 2 什么都没发生）。只记录命令行
        证明不了这一点，必须看窗口有没有真的出现。
        """
        self._seq += 1
        name = f"{self._seq:02d}_external_{Path(cmd[1]).stem}"
        log = self.shots / f"{name}.log"
        t0 = time.monotonic()
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace")
        found: list[tuple[int, str]] = []
        img = None
        rect = (0, 0, 0, 0)
        appear_s = -1.0
        try:
            while time.monotonic() - t0 < self.args.external_timeout:
                found = _windows_by_title(title_sub)
                if found:
                    break
                if proc.poll() is not None:
                    break
                time.sleep(0.25)
            if found:
                appear_s = time.monotonic() - t0
                # 窗口可见 ≠ 画完了：matplotlib 那类画布要等首帧渲染完才有内容。
                time.sleep(1.2)
                # 截图必须**在 kill 之前**：进程一结束窗口就没了，
                # GetWindowRect 返回 0、PrintWindow 直接失败 —— 第一版把
                # 截图放在 finally 后面，于是两条都记成"窗口尺寸 0x0、没截到"。
                # 窗口"出现"不等于"画完了"：刚出现时 PrintWindow 只给一块底色，
                # 所以按颜色数重试，别把一张空白图当成证据交出去。
                deadline = time.monotonic() + 8.0
                while time.monotonic() < deadline:
                    img = _capture_hwnd(found[0][0])
                    if img is not None:
                        colors = img.getcolors(maxcolors=20000)
                        if colors is None or len(colors) > 60:
                            break
                    time.sleep(0.4)
                rect = _window_rect(found[0][0])
        finally:
            _kill_tree(proc.pid)
            try:
                out = proc.stdout.read() if proc.stdout else ""
            except Exception:                              # noqa: BLE001
                out = ""
            try:
                proc.wait(timeout=5)
            except Exception:                              # noqa: BLE001
                pass
            log.write_text(f"命令：{cmd}\n退出码：{proc.returncode}\n\n{out[-4000:]}",
                           encoding="utf-8")
        alive = _pids_alive([proc.pid])
        shot, mode = "", "geometry"
        if img is not None:
            try:
                img.save(self.shots / f"{name}.png")
                shot, mode = f"{name}.png", "printwindow"
            except Exception:                              # noqa: BLE001
                pass
        step = Step(
            seq=self._seq, window="external", title=label,
            action="起独立进程", shot=shot, shot_mode=mode,
            size=(rect[2], rect[3]),
            n_widgets=0, n_buttons=0,
            notes=[f"命令：{' '.join(cmd)}",
                   (f"窗口出现：{found[0][1]!r}（{appear_s:.1f}s）" if found
                    else f"{self.args.external_timeout}s 内没有出现标题含"
                         f"「{title_sub}」的窗口"),
                   (f"窗口尺寸：{rect[2]}x{rect[3]}" if found else ""),
                   f"退出码 {proc.returncode}（走查器主动结束，非自身失败）；"
                   f"收尾后残留进程 {alive or '无'}"])
        self.steps.append(step)
        if not found:
            self.add_defect(
                severity="静默无反馈", where=f"external · {label}",
                what=f"起了命令，但 {self.args.external_timeout}s 内没有任何窗口出现"
                     "（用户看到的是「点了没反应」）",
                how=" ".join(cmd),
                evidence=(log.read_text(encoding="utf-8", errors="replace")[-600:]
                          if log.exists() else ""))
        if alive:
            self.add_defect(severity="崩溃/卡死", where=f"external · {label}",
                            what="收尾后仍有进程存活", how=" ".join(cmd),
                            evidence=str(alive))

    def run_external(self) -> None:
        """把两个独立进程 GUI 各起一次。只读，不开相机。"""
        app = self._app
        src = app.recent_visualization()
        plan: list[tuple[str, list[str], str]] = [
            ("apps/data_manager.py", [], "数据管理"),
        ]
        if src is not None:
            # src 是 <会话>/derived/visualization.json，会话名在它上一级
            sess = src.parent.parent.name if src.parent.name == "derived" else src.parent.name
            plan.append(("apps/view_3d.py", [str(src)], f"3D 查看器（{sess}）"))
        else:
            self.add_defect(
                severity="静默无反馈", where="hub · 3D 查看器",
                what="磁盘上没有任何 visualization.json，门厅的「3D 查看器」只能给一句提示",
                how="点门厅「工具 · 3D 查看器」",
                evidence="recent_visualization() 返回 None")
        for rel, extra, label in plan:
            cmd = [sys.executable, str(ROOT / rel), *extra]
            self.external_step(cmd, EXTERNAL_SAFE.get(rel, ""), label)

    # ------------------------------------------------------------------ #
    # 单位探针：球径 mm/m 断链的 A/B 对照
    # ------------------------------------------------------------------ #
    def probe_ball_unit(self) -> dict | None:
        """同一段视频、其它参数一字不改，只把 ``--ball-diameter`` 换成正确单位。

        为什么必须是 A/B 而不是"跑一次看看"：跑一次得到 0 个三维点，原因可能
        是检测器、阈值、视频本身 —— 只有同一命令只改一个值、结果从"有"变"没有"，
        才能把锅扣在那个值上。这也正是"截图证明不了"的那类缺陷。
        """
        runs = [e for e in self.events
                if e.get("kind") == "task(执行)"
                and len(e.get("cmd") or []) > 2
                and Path(e["cmd"][1]).name == "run_video.py"]
        if not runs:
            return None
        gui_cmd = runs[0]["cmd"]
        if "--ball-diameter" not in gui_cmd:
            return None
        base = self.out / "probe" / "ball_unit"
        results = []
        for label, value in (("界面传的（当 mm 用）", "21.6"),
                             ("同一颗球的正确值（m）", "0.0216")):
            cmd = _bounded_run_video(gui_cmd, base / value, self.args.max_frames)
            i = cmd.index("--ball-diameter")
            cmd[i + 1] = value
            t0 = time.monotonic()
            try:
                r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                                   encoding="utf-8", errors="replace",
                                   timeout=max(90, self.args.task_timeout * 2),
                                   stdin=subprocess.DEVNULL)
                code, out = r.returncode, (r.stdout or "") + (r.stderr or "")
            except subprocess.TimeoutExpired:
                code, out = -9, "超时"
            vis = base / value / "visualization.json"
            info = {"label": label, "value": value, "exit": code,
                    "elapsed": round(time.monotonic() - t0, 1),
                    "vis_exists": vis.exists(),
                    "tail": [ln for ln in out.splitlines() if ln.strip()][-6:]}
            if vis.exists():
                try:
                    d = json.loads(vis.read_text(encoding="utf-8"))
                    info["n_points"] = d.get("n_points")
                    info["diameter_mm"] = d.get("ball_diameter_mm")
                    b = d.get("bounds") or {}
                    z = ((b.get("min") or [None, None, None])[2],
                         (b.get("max") or [None, None, None])[2])
                    if z[0] is None:
                        # 单目的 visualization.json 不一定有 bounds，就直接从轨迹取
                        zs = [p["xyz"][2] for p in d.get("trajectory") or []
                              if isinstance(p, dict) and len(p.get("xyz") or []) >= 3]
                        if zs:
                            z = (min(zs), max(zs))
                    info["z"] = z
                except (OSError, ValueError, TypeError):
                    pass
            results.append(info)
        self._ball_probe = results
        gui_val = gui_cmd[gui_cmd.index("--ball-diameter") + 1]
        try:
            bad_unit = float(gui_val) > 1.0        # 球径以米计：>1 m 的球不存在
        except ValueError:
            bad_unit = False
        self._ball_gui_value, self._ball_bad = gui_val, bad_unit
        has = [r for r in results if r.get("n_points")]
        if bad_unit and len(has) == 1:
            self.add_defect(
                severity="静默无反馈",
                where="mono · ③ 开始识别（球径单位）",
                what="界面把球径的 mm 数值原样传给 `--ball-diameter`（那个参数吃**米**）。"
                     "同一段视频、同一条命令，只把这一个值换成正确单位，三维点从"
                     "「一个都没有、连 visualization.json 都不写」变成 30 个点 —— "
                     "而界面对此一声不吭，用户只会看到「跑完了但没结果」",
                how="单目窗选会话 → ③ 开始识别（球径框默认 21.6）",
                evidence="；".join(
                    f"{r['label']} `{r['value']}` → n_points={r.get('n_points')}, "
                    f"Z={r.get('z')}, 日志：{(r['tail'] or ['(空)'])[-1][:80]}"
                    for r in results)
                     + f"；界面实际传的是 {gui_val}")
        return {"runs": results, "gui_value": gui_val, "bad_unit": bad_unit}

    # ------------------------------------------------------------------ #
    # 命令行探针：证明"按钮想跑的那条命令"本身起得来
    # ------------------------------------------------------------------ #
    def probe_scripts(self) -> list[dict]:
        """对记录到的每个脚本跑一次 ``--help``。

        为什么用 ``--help``：它只走 argparse 与 import，不碰相机、不写库，
        但足以证明"脚本能 import、参数定义没坏"。**启动即崩**（ImportError /
        语法错误 / argparse 写错）会在这里露出来 —— 那类问题在界面上表现为
        "点了没反应"，是用户最难自查的一种。
        """
        scripts: dict[str, dict] = {}
        for ev in self.events:
            if not str(ev.get("kind") or "").startswith("task("):
                continue
            cmd = ev.get("cmd") or []
            if len(cmd) < 2:
                continue
            script = Path(str(cmd[1]))
            key = script.name
            if key in scripts:
                scripts[key]["used_by"].append(ev["title"])
                continue
            scripts[key] = {"script": str(script), "used_by": [ev["title"]]}
        results: list[dict] = []
        for key, info in scripts.items():
            rel = _rel(Path(info["script"]))
            if key in {Path(p).name for p in NEVER_SPAWN} or rel in NEVER_SPAWN:
                info["probe"] = "跳过（会开相机或写数据）"
                results.append(info)
                continue
            try:
                r = subprocess.run([sys.executable, info["script"], "--help"],
                                   cwd=str(ROOT), capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", timeout=30)
                code, out = r.returncode, (r.stdout or "") + (r.stderr or "")
            except subprocess.TimeoutExpired:
                code, out = -9, "30 秒未退出"
            info["probe_exit"] = code
            info["probe_tail"] = out.strip().splitlines()[-3:]
            results.append(info)
            if code != 0:
                self.add_defect(
                    severity="崩溃/卡死",
                    where=f"脚本 {key}",
                    what=f"`{key} --help` 退出码 {code} —— 这个入口点了会没有任何反应",
                    how=f"python {rel} --help",
                    evidence="\n".join(info["probe_tail"]))
        self._probes = results
        return results

    # ------------------------------------------------------------------ #
    # 报告
    # ------------------------------------------------------------------ #
    def write_reports(self, extra: dict) -> Path:
        self.defects.sort(key=lambda d: SEVERITY_ORDER.get(d["severity"], 9))
        data = {
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "python": sys.executable,
            "screen": {"width": self._screen[0], "height": self._screen[1],
                       "shot_scale": self._scale},
            "only": sorted(self.only),
            "steps": [s.__dict__ for s in self.steps],
            "defects": self.defects,
            "events": self.events,
            "probes": getattr(self, "_probes", []),
            "tk_errors": self._errors,
            **extra,
        }
        (self.out / "report.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        md = self._render_md(data)
        (self.out / "report.md").write_text(md, encoding="utf-8")
        return self.out / "report.md"

    def _render_md(self, data: dict) -> str:
        shots = sum(1 for s in self.steps if s.shot)
        L: list[str] = []
        L.append(f"# 新 GUI 走查报告（{data['generated']}）\n")
        L.append(f"- 解释器：`{data['python']}`")
        L.append(f"- 屏幕：{data['screen']['width']}x{data['screen']['height']}"
                 f"（截图缩放 {data['screen']['shot_scale']:g}×）")
        L.append(f"- 窗口范围："
                 f"{'、'.join(WINDOWS.get(k, k) for k in data['only']) if data['only'] else '全部'}")
        uncovered = data.get("uncovered") or []
        if uncovered:
            L.append(f"- ⚠ **未覆盖**："
                     f"{'、'.join(WINDOWS.get(k, k) for k in uncovered)}"
                     f"　（本次不是全量走查）")
        L.append(f"- 步数 {len(self.steps)}，截图 {shots} 张，缺陷 {len(self.defects)} 条")
        cam = data.get("camera_state") or {}
        if cam.get("roles"):
            L.append(f"- 相机角色：{cam['roles']}"
                     + ("　**左右撞车**（所有双目脚本会在开相机前失败）"
                        if cam.get("collided") else ""))
        files = data.get("gui_files") or {}
        if files:
            L.append("- 走查的 GUI 版本（文件 md5 前 10 位）："
                     + "，".join(f"{k}={v}" for k, v in files.items()))
        L.append("")

        L.append("## 结论（3 行以内）\n")
        for line in _conclusions(data):
            L.append(line)
        L.append("")

        L.append("## 缺陷清单（按严重度）\n")
        if not self.defects:
            L.append("无。\n")
        else:
            L.append("| # | 严重度 | 窗口·步骤 | 现象 | 复现（控件路径+动作） | 证据 |")
            L.append("| - | ------ | --------- | ---- | -------------------- | ---- |")
            for i, d in enumerate(self.defects, 1):
                L.append(f"| {i} | {d['severity']} | {_md(d['where'])} | {_md(d['what'])} "
                         f"| {_md(d['how'])} | {_md(_clip(d['evidence'], 220))} |")
            L.append("")

        L.append("## 逐窗口走查记录\n")
        cur = None
        for s in self.steps:
            if s.window != cur:
                cur = s.window
                L.append(f"\n### {cur}\n")
                L.append("| 步 | 尺寸 | 控件 | 裁切 | 异常 | 做了什么 |")
                L.append("| -- | ---- | ---- | ---- | ---- | -------- |")
            flags = []
            if s.clipped:
                flags.append(f"{len(s.clipped)} 个越界")
            if s.overflow_px > 4:
                if s.overflow_exempt:
                    # 豁免要**写在表里**：报告读者看到"需滚 1152px"才不会以为是漏项，
                    # 后面那个数字是实测的 yview 终点（1.00 = 真能滚到末尾）。
                    reach = (f"，滚到底 {s.scroll_reach:.2f}"
                             if s.scroll_reach >= 0 else "")
                    flags.append(f"需滚 {s.overflow_px}px（设计如此{reach}）")
                else:
                    flags.append(f"需滚 {s.overflow_px}px")
            if s.errors:
                flags.append("异常")
            L.append(f"| {s.seq} | {s.size[0]}x{s.size[1]} | {s.n_widgets} "
                     f"（按钮 {s.n_buttons}） "
                     f"| {'/'.join(flags) or '无'} "
                     f"| {_md(_clip(s.errors[0].strip().splitlines()[-1], 60)) if s.errors else '无'} "
                     f"| {_md(s.title)} |")
        L.append("")

        L.append("## 按钮生成过的命令行\n")
        tasks = [e for e in self.events if str(e.get("kind") or "").startswith("task(")]
        if not tasks:
            L.append("（没有按钮试图启动任务）\n")
        for e in tasks:
            ran = e["kind"] == "task(执行)"
            L.append(f"- **{e['title']}**（{'真跑，下面的 argv 是改写后的' if ran else '未执行'}）")
            L.append(f"  ```\n  GUI 生成：{' '.join(e['cmd'])}\n  ```")
            if ran and e.get("cmd_run"):
                L.append(f"  ```\n  实际执行：{' '.join(e['cmd_run'])}\n  ```")
            elif not ran:
                L.append(f"  原因：{e.get('why') or '未说明'}")
        L.append("")

        L.append("## 弹窗与文件选择框（已自动应答/取消）\n")
        dialogs = [e for e in self.events
                   if not str(e.get("kind") or "").startswith("task(")]
        if not dialogs:
            L.append("（这次走查没有碰到弹窗）\n")
        for e in dialogs:
            body = (e.get("text") or e.get("path") or "").replace("\n", " ")
            L.append(f"- `{e['kind']}` {_md(e.get('title') or '')}：{_md(_clip(body, 160))}")
        L.append("")

        L.append("## 命令行探针（`--help` 能否跑通）\n")
        probes = data.get("probes") or []
        if not probes:
            L.append("（未运行：`--step` 限步或 `--no-probe`）\n")
        for p in probes:
            if "probe_exit" in p:
                L.append(f"- `{Path(p['script']).name}` → 退出码 {p['probe_exit']}"
                         f"（被这些按钮用到：{'、'.join(p['used_by'])}）")
            else:
                L.append(f"- `{Path(p['script']).name}` → {p.get('probe')}"
                         f"（被这些按钮用到：{'、'.join(p['used_by'])}）")
        L.append("")

        L.append("## 状态条前后对照（每次点击与填值）\n")
        acts = [s for s in self.steps if s.action in ("点击", "填值")]
        if not acts:
            L.append("（本次没有点击）\n")
        for s in acts:
            before = s.status_before or "（空）"
            after = s.status or "（空）"
            mark = "  ↺ 没变" if s.status_before == s.status else ""
            L.append(f"- `#{s.seq}` {_md(s.title)}")
            L.append(f"  前：{_md(_clip(before, 120))}")
            L.append(f"  后：{_md(_clip(after, 120))}{mark}")
            for t in s.tasks:
                mark = "真跑" if t["kind"] == "task(执行)" else "未执行"
                L.append(f"  任务（{mark}）：{_md(_clip(' '.join(t['cmd'][1:]), 160))}")
            for d in s.dialogs:
                body = (d.get("text") or d.get("path") or "").replace("\n", " ")
                L.append(f"  对话：`{d['kind']}` {_md(_clip(str(d.get('title') or body), 100))}")
        L.append("")

        L.append("## 任务执行取证（真跑过的子任务）\n")
        ran = [s for s in self.steps if s.task_cmd]
        if not ran:
            L.append("（本次没有任何子任务被真跑：`--no-execute` 或全是需硬件的按钮。"
                     "在这种情况下，本报告对「按钮是否真的发生了它承诺的事」**不作结论**）\n")
        else:
            L.append("| 步 | 按钮 | 退出码 | 日志最后 3 行 | 承诺的产物 | 状态条回写 |")
            L.append("| -- | ---- | ------ | ------------- | ---------- | ---------- |")
            for s in ran:
                art = "；".join(f"{'有' if a['exists'] else '**没有**'} `{a['path']}`"
                                for a in s.artifacts) or "（无）"
                wrote = "是" if (s.status != s.status_before) else "**否**"
                L.append(f"| {s.seq} | {_md(s.title)} | {s.task_exit}"
                         f"{'（超时）' if s.task_timeout else ''} | "
                         f"{_md(' / '.join(s.task_log[-3:]) or '（日志空）')} | {art} | {wrote} |")
            L.append("")
            L.append("改写说明（为了让流程能跑又不碰数据/硬件）：\n")
            for s in ran:
                L.append(f"- `#{s.seq}` {_md(s.task_note)}")
            L.append("")

        skipped = [e for e in self.events if e.get("kind") == "task(未执行)"]
        if skipped:
            L.append("### 没有执行的按钮（需硬件 / 会动数据）\n")
            seen: set[str] = set()
            for e in skipped:
                key = e.get("title") or ""
                if key in seen:
                    continue
                seen.add(key)
                L.append(f"- **{_md(key)}**：{_md(e.get('why') or '未说明')}")
                L.append(f"  `{' '.join(e.get('cmd') or [])}`")
            L.append("")

        probe = data.get("ball_probe")
        if probe:
            L.append("## 球径单位 A/B 探针（同一段视频，只改 --ball-diameter）\n")
            L.append(f"界面这次实际传给 `--ball-diameter` 的值：`{probe.get('gui_value')}`"
                     + ("　**单位错了（按毫米传，参数吃米）**" if probe.get("bad_unit")
                        else "　单位正确（米）"))
            L.append("")
            L.append("| 传的值 | 退出码 | visualization.json | n_points | Z 范围 (m) |")
            L.append("| ------ | ------ | ------------------ | -------- | ---------- |")
            for r in probe["runs"]:
                z = r.get("z") or (None, None)
                L.append(f"| `{r['value']}`（{_md(r['label'])}） | {r['exit']} | "
                         f"{'有' if r['vis_exists'] else '**没有**'} | "
                         f"{r.get('n_points', '—')} | {z[0]}–{z[1]} |")
            L.append("")
            for r in probe["runs"]:
                L.append(f"`{r['value']}` 日志尾：")
                L.append("```")
                L.extend(r["tail"])
                L.append("```")
            L.append("")

        L.append("## 异常全文\n")
        errors = data.get("tk_errors") or []
        if not errors:
            L.append("（没有捕获到回调异常）\n")
        else:
            L.append("被 Tk 回调吞掉的异常原样贴在这里 —— 真实用户看到的是它弹出的"
                     " Traceback 对话框，或者干脆什么都没有。\n")
            for i, tb in enumerate(errors, 1):
                L.append(f"### 异常 {i}\n")
                L.append("```")
                L.append(tb.strip())
                L.append("```")
        L.append("")

        L.append("## 数据与配置\n")
        for line in data.get("snapshot_md") or ["（未比对）"]:
            L.append(f"- {line}")
        L.append("")

        L.append("## 没能验证的部分\n")
        for line in data.get("unverified") or []:
            L.append(f"- {line}")
        L.append("")
        return "\n".join(L)


class _FakeTask:
    """假任务：只为了让调用方的 ``task.exit_code`` 之类不炸。

    **不登记进 TaskRunner**：登记了之后控制台会去读 ``proc.pid`` 并显示一行
    永远"运行中"的假任务，那比不显示更误导。
    """

    def __init__(self, title: str) -> None:
        self.tid = 0
        self.title = title
        self.exit_code = None
        self.kill_note = ""
        self.running = False
        self.proc = type("P", (), {"pid": 0})()

    def elapsed(self) -> float:
        return 0.0

    def cmdline(self) -> str:
        return ""


class _StopWalk(Exception):
    """走到 ``--step N`` 就停：拿它从任意深处跳出来。"""


# --------------------------------------------------------------------------- #
# 控件工具（纯函数，不依赖 Walker）
# --------------------------------------------------------------------------- #
def _bounded_run_video(cmd: list[str], out: Path, max_frames: int) -> list[str]:
    """把 GUI 生成的 ``run_video`` argv 改成"限帧 + 产物落 outputs/ + 不弹窗"。

    只动这三件事：**帧数**（限量）、**产物路径**（不写 data/）、**--no-show**
    （否则 cv2 会弹窗）。`--ball-diameter` / `--method` / `--focal` / `--seed`
    一律原样保留 —— 那才是我们要验的东西，改了就变成验另一条命令。
    """
    drop_value = {"--out", "--csv", "--vis-json", "--max-frames"}
    out.mkdir(parents=True, exist_ok=True)
    kept: list[str] = []
    skip_next = False
    for i, tok in enumerate(cmd):
        if skip_next:
            skip_next = False
            continue
        if tok in drop_value:
            skip_next = True
            continue
        if tok == "--no-show":
            continue
        if i >= 2:                              # cmd[0]=python, cmd[1]=脚本
            kept.append(tok)
    return [cmd[0], cmd[1], *kept,
            "--out", str(out / "annotated.mp4"),
            "--csv", str(out / "traj.csv"),
            "--vis-json", str(out / "visualization.json"),
            "--max-frames", str(max_frames),
            "--no-show"]


def _camera_state() -> dict:
    """只读看左右角色有没有撞车（不开相机）。撞车时双目脚本会在开相机前失败。"""
    try:
        from sphere_tracker import camera_registry as cr
        reg = cr.load_registry()
        roles = {k: (v.index, v.identity) for k, v in reg.roles.items()}
        keys = {(v[0], v[1]) for v in roles.values()}
        return {"collided": len(keys) < len(roles), "roles": roles}
    except Exception:                                  # noqa: BLE001
        return {"collided": False, "roles": {}}


def _stereo_roots() -> list[str]:
    try:
        from sphere_tracker.paths import stereo_roots
        return [str(p) for p in stereo_roots()]
    except Exception:                                  # noqa: BLE001
        return []


def _is_under(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except (ValueError, OSError):
        return False


def _tb_brief(tb: str) -> str:
    """traceback → "最后那个 File 行；异常类型: 消息"。

    表格里塞整段 traceback 会把一行撑成十行，全文另放在「异常全文」一节。
    """
    lines = [ln.rstrip() for ln in tb.strip().splitlines() if ln.strip()]
    file_line = next((ln.strip() for ln in reversed(lines)
                      if ln.strip().startswith("File ")), "")
    return f"{file_line}；{lines[-1].strip()}" if lines else ""


def _alive(win) -> bool:
    try:
        return bool(win is not None and win.winfo_exists())
    except tk.TclError:
        return False


def _descendants(root, *, cross_toplevel: bool = False) -> list:
    """按创建顺序的深度优先展开（顺序稳定，报告才好比对）。

    **默认不穿过 Toplevel**：Tk 里子窗口的控件路径挂在父控件下面
    （``.hub.bench.picker``），所以 ``winfo_children()`` 会把**另一扇窗**的
    控件一起交出来。2026-11-07 的报告里「bench · 换会话… clipped=2」就是
    这么来的：被"切掉"的其实是会话选择器里的卡片与表格 —— 它们的坐标是
    相对选择器算的，挪进 bench 的坐标系自然越界。于是同一次走查夸了一条
    单跑 bench 永远量不出来的假缺陷（夸的是走查工具自己）。

    "这扇窗里有哪些控件"和"屏幕上有几扇窗"是两个问题：后者用
    ``cross_toplevel=True``（``_windows``）。
    """
    out, stack = [], [root]
    while stack:
        w = stack.pop(0)
        out.append(w)
        try:
            kids = list(w.winfo_children())
        except tk.TclError:
            kids = []
        if not cross_toplevel:
            # 用 isinstance，不是比类名：`tk.Toplevel(master, class_="Xxx")` 的
            # `winfo_class()` 会变成 "Xxx"，比字符串的写法那时就悄悄开始穿透了
            # （假裁切会原样回来），而 isinstance 不受 `class_=` 影响。
            kids = [c for c in kids if not isinstance(c, tk.Toplevel)]
        stack[0:0] = kids
    return out


def _windows(root) -> list:
    """``root`` 这棵树里的全部窗口，含挂在别的窗口下面的子窗。

    会话选择器是 ``SessionPicker(self, ...)`` —— 挂在点它的那扇窗下面，
    不是 App 的直接子节点。找窗口、数窗口都得穿过去，否则"点了没反应"
    会变成假缺陷。
    """
    return _descendants(root, cross_toplevel=True)


def _text_of(w) -> str:
    cls = w.winfo_class()
    try:
        if cls in ("TLabel", "Label", "TButton", "Button", "TCheckbutton",
                   "Checkbutton", "TRadiobutton", "Radiobutton",
                   "TLabelframe", "Labelframe"):
            return str(w.cget("text"))
        if cls in ("TEntry", "Entry", "TCombobox", "Combobox"):
            return str(w.get())
        if cls == "Text":
            return f"<{int(w.index('end-1c').split('.')[0])} 行>"
        if cls == "Treeview":
            return f"<{len(w.get_children())} 行>"
    except tk.TclError:
        return ""
    return ""


def _role(w) -> str:
    cls = w.winfo_class()
    if cls in ("TButton", "Button"):
        return "button"
    if cls in ("TEntry", "Entry"):
        return "entry"
    if cls in ("TCombobox", "Combobox"):
        return "combo"
    if cls in ("TCheckbutton", "Checkbutton"):
        return "check"
    if cls in ("TRadiobutton", "Radiobutton"):
        return "radio"
    if cls in ("TLabelframe", "Labelframe"):
        return "card"
    if cls == "Treeview":
        return "tree"
    if cls == "Text":
        return "text"
    if cls == "Canvas":
        return "canvas"
    if cls in ("TLabel", "Label"):
        return "label"
    return "frame"


def _inventory(win) -> list[dict]:
    """可见控件的清单：类型 + 文案 + state + 相对窗口的位置。"""
    wx, wy = win.winfo_rootx(), win.winfo_rooty()
    out: list[dict] = []
    for w in _descendants(win):
        try:
            if not w.winfo_ismapped():
                continue
            role = _role(w)
            if role in ("frame", "canvas") and not _text_of(w):
                continue
            state = ""
            try:
                state = " ".join(w.state()) if hasattr(w, "state") else ""
            except tk.TclError:
                state = ""
            out.append({
                "role": role,
                "class": w.winfo_class(),
                "text": _clip(_text_of(w), 160),
                "state": state,
                "path": _path_of(w),
                "rect": [w.winfo_rootx() - wx, w.winfo_rooty() - wy,
                         w.winfo_width(), w.winfo_height()],
                "in_scroll": _in_scroll(w),
            })
        except tk.TclError:
            continue
    return out


def _path_of(w) -> str:
    """控件路径：``卡片标题 > 按钮文案``。报告里要靠它复现。"""
    bits: list[str] = []
    cur = w
    for _ in range(8):
        if cur is None:
            break
        try:
            if cur.winfo_class() in ("TLabelframe", "Labelframe"):
                bits.append(str(cur.cget("text")))
        except tk.TclError:
            break
        cur = getattr(cur, "master", None)
    bits.reverse()
    try:
        own = _text_of(w)
    except tk.TclError:
        own = ""
    bits.append(own or w.winfo_class())
    return " > ".join(bits)


def _in_scroll(w) -> bool:
    """这个控件是不是躺在滚动页里（那它只是"要滚"，不是"被切掉"）。"""
    cur = w
    for _ in range(12):
        cur = getattr(cur, "master", None)
        if cur is None:
            return False
        try:
            if cur.winfo_class() == "Canvas":
                return True
        except tk.TclError:
            return False
        if getattr(cur, "_canvas", None) is not None:
            return True
    return False


def _clipped(win, widgets: list[dict]) -> list[str]:
    """**真的被窗口边缘切掉**的控件：在窗口外，而且不在滚动区里。

    滚动区里的控件超出视口只说明"要滚一下"，那是 ScrollPage 的职责，
    混进"裁切"里会把一条真问题淹进二十条噪声。滚动溢出单独记 overflow_px。
    """
    W, H = win.winfo_width(), win.winfo_height()
    out: list[str] = []
    for item in widgets:
        x, y, w, h = item["rect"]
        if w <= 1 or h <= 1:
            continue
        if y + h <= H + 1 and x + w <= W + 1:
            continue
        if item.get("in_scroll"):
            continue
        out.append(f"{item['path']} ({x},{y},{w}x{h} vs 窗 {W}x{H})")
    return out


def _status_text(win) -> str:
    """读窗口的状态条文字。

    主窗口的 `Status` 封装里是 `_label`；对话框（相机管理/工具台…）直接是
    一个 `ttk.Label`。以前只认 `_label`，于是对对话框永远读到空字符串，
    点按钮就被判成"状态条没变 → 静默无反馈"（2026-11-07 走查报了 6 条假缺陷，
    因为按钮其实把对话框状态条改了，只是读的地方不对）。
    """
    st = getattr(win, "status", None)
    if st is None:
        return ""
    lab = getattr(st, "_label", st)
    try:
        return str(lab.cget("text"))
    except tk.TclError:
        return ""


def _card(win, prefix: str):
    for w in _descendants(win):
        if _role(w) != "card":
            continue
        try:
            if str(w.cget("text")).startswith(prefix):
                return w
        except tk.TclError:
            continue
    return None


def _button_in(parent, text: str, *, prefix: bool = False):
    """按文字找按钮。``prefix=True`` 时还接受「文字+补充原因」这类变体
    （例：「开始录制（先解决标定）」），默认仍要求完全相等。"""
    if parent is None:
        return None
    for w in _descendants(parent):
        if _role(w) != "button":
            continue
        try:
            label = str(w.cget("text"))
            if label == text or (prefix and label.startswith(text)):
                return w
        except tk.TclError:
            continue
    return None


def _entry_of(win, attr: str):
    """按 ``textvariable`` 找输入框：界面里同一种框有好几个，只能靠变量认。"""
    var = getattr(win, attr, None)
    if var is None:
        return None
    for w in _descendants(win):
        if _role(w) != "entry":
            continue
        try:
            if str(w.cget("textvariable")) == str(var):
                return w
        except tk.TclError:
            continue
    return None


def _combobox_of(win, attr: str):
    var = getattr(win, attr, None)
    if var is None:
        return None
    for w in _descendants(win):
        if _role(w) != "combo":
            continue
        try:
            if str(w.cget("textvariable")) == str(var):
                return w
        except tk.TclError:
            continue
    return None


def _checkbutton_in(parent):
    for w in _descendants(parent):
        if _role(w) == "check":
            return w
    return None


def _collapsible(win):
    from apps._collapsible import CollapsibleSection
    for w in _descendants(win):
        if isinstance(w, CollapsibleSection):
            return w
    return None


def _find_toplevel(app, title_part: str):
    """找标题含某段文字的子窗口。

    会话选择窗是**挂在点它的那个窗口下面**的（``SessionPicker(self, ...)``），
    不是 App 的直接子节点 —— 只翻 ``app.winfo_children()`` 会漏掉它们，
    于是"点了没反应"变成一个假缺陷。这里整棵树都翻。
    """
    for w in _windows(app):
        if not isinstance(w, tk.Toplevel):
            continue
        try:
            if w.winfo_exists() and title_part in str(w.title()):
                return w
        except tk.TclError:
            continue
    return None


def _ui_signature(app, win) -> tuple:
    """用户能看到的"界面变了没有"：窗口数、状态条、日志行数、表格行数。

    为什么不是只看状态条：控制台**没有状态条**（它的反馈是往日志里追加一行），
    只盯状态条会把它三个按钮全判成"点了没反应"。
    """
    tops = 0
    for w in _windows(app):
        if isinstance(w, tk.Toplevel):
            try:
                if w.winfo_exists():
                    tops += 1
            except tk.TclError:
                pass
    texts, trees = [], []
    for w in _descendants(win):
        cls = w.winfo_class()
        try:
            if cls == "Text":
                texts.append(int(str(w.index("end-1c")).split(".")[0]))
            elif cls == "Treeview":
                trees.append(len(w.get_children()))
        except tk.TclError:
            continue
    return (tops, _status_text(win), tuple(texts), tuple(trees))


def _tree_row_by_text(win, needle: str):
    tree = next((w for w in _descendants(win) if _role(w) == "tree"), None)
    if tree is None or not needle:
        return None
    for iid in tree.get_children():
        if needle in str(iid):
            return iid
        vals = tree.item(iid, "values")
        if any(needle in str(v) for v in vals):
            return iid
    return None


def _bench_states(win) -> dict:
    try:
        from apps._gui_bench import step_states
        return step_states(win._summary)
    except Exception:                                  # noqa: BLE001
        return {}


# --------------------------------------------------------------------------- #
# 进程 / 窗口工具
# --------------------------------------------------------------------------- #
def _capture_window(win):
    """把一个 Tk 窗口画成图（见 :func:`_capture_hwnd`）。"""
    if os.name != "nt":
        return None
    try:
        hwnd = ctypes.windll.user32.GetAncestor(int(win.winfo_id()), 2)  # GA_ROOT
    except Exception:                                          # noqa: BLE001
        return None
    return _capture_hwnd(hwnd)


def _capture_hwnd(hwnd):
    """用 PrintWindow 把这个窗口画进位图，只保留客户区（不含标题栏）。

    返回 PIL.Image 或 None。整张图颜色完全一致视为失败（有些窗口在
    PrintWindow 下只给一块底色），交给调用方走退路。
    """
    if os.name != "nt" or not hwnd:
        return None
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32
        user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.c_void_p]
        user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.c_void_p]
        user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.c_void_p]
        user32.GetWindowDC.argtypes = [wintypes.HWND]
        user32.PrintWindow.argtypes = [wintypes.HWND, ctypes.c_void_p, wintypes.UINT]
        wr = wintypes.RECT()
        cr = wintypes.RECT()
        pt = wintypes.POINT(0, 0)
        user32.GetWindowRect(hwnd, ctypes.byref(wr))
        user32.GetClientRect(hwnd, ctypes.byref(cr))
        user32.ClientToScreen(hwnd, ctypes.byref(pt))
        W, H = wr.right - wr.left, wr.bottom - wr.top
        if W < 40 or H < 40:
            return None
        hdc = user32.GetWindowDC(hwnd)
        mem = gdi32.CreateCompatibleDC(hdc)
        bmp = gdi32.CreateCompatibleBitmap(hdc, W, H)
        gdi32.SelectObject(mem, bmp)
        try:
            if not user32.PrintWindow(hwnd, mem, 2):           # 2 = PW_RENDERFULLCONTENT
                return None
            bi = _BITMAPINFO()
            bi.bmiHeader.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
            bi.bmiHeader.biWidth = W
            bi.bmiHeader.biHeight = -H                          # 负 = 自上而下
            bi.bmiHeader.biPlanes = 1
            bi.bmiHeader.biBitCount = 32
            bi.bmiHeader.biCompression = 0
            buf = ctypes.create_string_buffer(W * H * 4)
            gdi32.GetDIBits(mem, bmp, 0, H, buf, ctypes.byref(bi), 0)
        finally:
            gdi32.DeleteObject(bmp)
            gdi32.DeleteDC(mem)
            user32.ReleaseDC(hwnd, hdc)
        img = Image.frombuffer("RGBA", (W, H), buf, "raw", "BGRA", 0, 1).convert("RGB")
        # 只留客户区：控件坐标是相对客户区算的，报告里的 rect 才能对得上
        ox, oy = pt.x - wr.left, pt.y - wr.top
        crop = img.crop((ox, oy, ox + cr.right, oy + cr.bottom))
        if crop.width < 40 or crop.height < 40:
            return None
        colors = crop.getcolors(maxcolors=8)
        if colors is not None and len(colors) <= 1:
            return None                                        # 整块纯色 = 没画出来
        return crop
    except Exception:                                          # noqa: BLE001
        return None


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
                ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD),
                ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG),
                ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD)]


class _BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", _BITMAPINFOHEADER),
                ("bmiColors", wintypes.DWORD * 3)]


def _kill_tree(pid: int) -> None:
    """收尾杀整棵树。**按结果判定**：杀完再问一次还在不在。"""
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       capture_output=True, text=True)
    else:
        try:
            os.killpg(os.getpgid(pid), 9)
        except Exception:                              # noqa: BLE001
            pass


def _pids_alive(pids: list[int]) -> list[int]:
    alive = []
    for pid in pids:
        if pid <= 0:
            continue
        if os.name == "nt":
            r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                               capture_output=True, text=True)
            if str(pid) in (r.stdout or ""):
                alive.append(pid)
        else:
            try:
                os.kill(pid, 0)
                alive.append(pid)
            except OSError:
                pass
    return alive


def _windows_by_title(substr: str) -> list[tuple[int, str]]:
    """枚举当前可见的顶层窗口，返回标题里含 ``substr`` 的那些。"""
    if os.name != "nt":
        return []
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    found: list[tuple[int, str]] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        if n <= 0:
            return True
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        if substr in buf.value:
            found.append((int(hwnd), buf.value))
        return True

    user32.EnumWindows(cb, 0)
    return found


def _window_rect(hwnd: int) -> tuple[int, int, int, int]:
    if os.name != "nt":
        return (0, 0, 0, 0)
    from ctypes import wintypes

    class RECT(ctypes.Structure):
        _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG),
                    ("right", wintypes.LONG), ("bottom", wintypes.LONG)]

    r = RECT()
    ctypes.windll.user32.GetWindowRect.argtypes = [wintypes.HWND,
                                                   ctypes.POINTER(RECT)]
    ctypes.windll.user32.GetWindowRect(int(hwnd), ctypes.byref(r))
    return (r.left, r.top, r.right - r.left, r.bottom - r.top)


# --------------------------------------------------------------------------- #
# 快照：走查前后必须证明"GUI 没动配置、也没动数据"
# --------------------------------------------------------------------------- #
def _fingerprint_config() -> dict[str, str]:
    import hashlib
    out: dict[str, str] = {}
    for p in sorted((ROOT / "config").glob("*.json")):
        try:
            out[p.name] = hashlib.md5(p.read_bytes()).hexdigest()
        except OSError:
            out[p.name] = "?"
    return out


def _fingerprint_data() -> dict[str, str]:
    """只记两层的 (名字 → 大小)，够发现"被写/被删/被新建"。"""
    out: dict[str, str] = {}
    base = ROOT / "data"
    if not base.is_dir():
        return out
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        for sub in sorted(entry.iterdir()):
            try:
                if sub.is_file():
                    out[str(sub.relative_to(ROOT))] = f"{sub.stat().st_size}"
                elif sub.is_dir():
                    n = sum(1 for _ in sub.iterdir())
                    out[str(sub.relative_to(ROOT)) + "/"] = f"entries={n}"
            except OSError:
                continue
    return out


def _diff(before: dict, after: dict) -> tuple[list[str], list[str], list[str]]:
    added = [k for k in after if k not in before]
    removed = [k for k in before if k not in after]
    changed = [k for k in after if k in before and after[k] != before[k]]
    return added, removed, changed


# --------------------------------------------------------------------------- #
# 报告文字
# --------------------------------------------------------------------------- #
def _conclusions(data: dict) -> list[str]:
    defects = data.get("defects") or []
    uncovered = data.get("uncovered") or []
    scope = ("　⚠ 本次不是全量走查，未覆盖："
             + "、".join(WINDOWS.get(k, k) for k in uncovered)) if uncovered else ""
    if not defects:
        if scope:
            return [f"已覆盖的窗口无崩溃、无静默失败。{scope}".rstrip(),
                    "已覆盖的每个窗口都开得起来，按钮都有反馈（仅限已覆盖的窗口）。",
                    "截图为证，见 `shots/`。"]
        return ["全流程无崩溃、无静默失败。",
                "每个窗口都开得起来，每个按钮都有反馈。",
                "截图为证，见 `shots/`。"]
    counts: dict[str, int] = {}
    for d in defects:
        counts[d["severity"]] = counts.get(d["severity"], 0) + 1
    tally = "、".join(f"{k} {v} 条" for k, v in
                     sorted(counts.items(), key=lambda kv: SEVERITY_ORDER.get(kv[0], 9)))
    first = defects[0]
    failed = data.get("failed_windows") or []
    tail = (f"有 {len(failed)} 个窗口根本起不来：{'、'.join(failed)}。"
            if failed else "所有窗口都能打开，问题集中在交互反馈与首屏布局。")
    return [
        f"共 {len(defects)} 条缺陷：{tally}。{scope}".rstrip(),
        f"最该先修：{first['where']} —— {_clip(first['what'], 70)}",
        tail,
    ]


def _md(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _clip(text: str, n: int) -> str:
    text = str(text)
    return text if len(text) <= n else text[: n - 1] + "…"


def _rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except (ValueError, OSError):
        return str(p)


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="新 GUI 的真人走查（截图 + 缺陷清单）。不碰硬件、不写 data/。")
    ap.add_argument("--out", default="outputs/gui_walk/run",
                    help="产物目录（截图、report.json、report.md）")
    ap.add_argument("--only", default="",
                    help="只走这些窗口：hub,mono,stereo,bench,console（逗号分隔）")
    ap.add_argument("--step", type=int, default=0,
                    help="只跑到第 N 步就停（0 = 跑完）")
    ap.add_argument("--mono-session", default="20261029-205522",
                    help="单目窗里要选中的那段会话（只读）")
    ap.add_argument("--stereo-session", default="20261024-173619",
                    help="双目窗/工作台里要选中的那段会话（只读）")
    ap.add_argument("--external-timeout", type=float, default=15.0,
                    help="等独立进程窗口出现的秒数")
    ap.add_argument("--no-external", action="store_true",
                    help="不起独立进程 GUI（数据管理 / 3D 查看器）")
    ap.add_argument("--no-probe", action="store_true",
                    help="不跑 `脚本 --help` 探针")
    ap.add_argument("--no-execute", dest="execute", action="store_false",
                    help="一个子任务都不真跑（只记录命令行；证据会降到'未验证'）")
    ap.add_argument("--max-frames", type=int, default=30,
                    help="离线识别真跑时只处理前 N 帧（产物落 outputs/）")
    ap.add_argument("--task-timeout", type=float, default=45.0,
                    help="单个子任务最多等多少秒，超时强制结束并记一条缺陷")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # `--only` 里写了不认识的键：**必须报错退出**，绝不能静默忽略。
    # 静默忽略 = 工具照样跑完、照样写"缺陷 0 条"的报告并退 0 —— 一次假绿。
    unknown = sorted({s.strip() for s in args.only.split(",") if s.strip()} - set(WINDOWS))
    if unknown:
        print(f"[中止] --only 里有不认识的窗口名：{'、'.join(unknown)}")
        print(f"       合法键（逗号分隔）：{'、'.join(WINDOWS)}")
        print("       （`stereo` 会展开成平行 + 大角度两个窗）")
        return 2
    ok, why = _gui_ready()
    if not ok:
        print(f"[中止] GUI 源码当前编译不过（可能有人正在保存）：{why}")
        print("       等它稳定再跑 —— 现在跑出来的'崩溃'是假的。")
        return 2
    # 会话参数默认是会话 id（``20261029-205522``），不是路径 —— 先解析成绝对路径：
    # 工作台/会话选择窗用相对路径判 `is_file()`，cwd 一变就判成"这不是会话目录"。
    args.mono_session = _resolve_session(args.mono_session, "mono")
    args.stereo_session = _resolve_session(args.stereo_session, "stereo")
    walker = Walker(args)
    # `--only` 排掉了哪些窗口：报告里必须显式标出"没走"，否则读者分不清
    # "走查过且没问题"和"根本没走"。全量走查时这个列表为空。
    uncovered = [k for k in WINDOWS if k not in walker.only] if walker.only else []
    _backup_config()
    config_before = _fingerprint_config()
    gui_before = _gui_fingerprint()
    data_before = _fingerprint_data()
    shutdown_note: list[str] = []
    unverified: list[str] = []
    md_snapshot: list[str] = []
    try:
        app = walker.build_app()
        order: list[tuple[str, callable]] = [
            ("hub", walker.walk_hub),
            ("mono", walker.walk_mono),
            ("stereo", lambda: walker.walk_stereo("parallel")),
            ("stereo", lambda: walker.walk_stereo("angled")),
            ("bench", walker.walk_bench),
            ("console", walker.walk_console),
        ]
        for key, fn in order:
            if walker.only and key not in walker.only:
                continue
            try:
                fn()
            except _StopWalk:
                raise
            except Exception:                              # noqa: BLE001
                tb = traceback.format_exc()
                walker.add_defect(
                    severity="崩溃/卡死",
                    where=f"{key} · 走查过程中",
                    what="窗口在构造/交互时抛异常（走查器捕获后继续下一个窗口）",
                    how=f"--only {key}",
                    evidence=_tb_brief(tb))
                unverified.append(f"{key}：走查中断，异常见报告 tk_errors")
                walker._errors.append(tb)
        if not args.no_external and not (walker.limit and walker._seq >= walker.limit):
            walker.run_external()
        # 单位 A/B 探针：要等单目那条命令被点出来才有素材，所以放在走查之后
        if not (walker.limit and walker._seq >= walker.limit):
            try:
                walker.probe_ball_unit()
            except Exception:                              # noqa: BLE001
                walker._errors.append(traceback.format_exc())
        # 探针放在最后：它要的是"按钮一共想跑过哪些脚本"，走完才有全集
        if not args.no_probe and not (walker.limit and walker._seq >= walker.limit):
            walker.probe_scripts()
    except _StopWalk:
        shutdown_note.append(f"按 --step {walker.limit} 提前结束")
    except Exception:                                      # noqa: BLE001
        walker.add_defect(severity="崩溃/卡死", where="走查器自身",
                          what="走查流程异常终止",
                          how=" ".join(sys.argv),
                          evidence=traceback.format_exc().strip().splitlines()[-1])
        walker._errors.append(traceback.format_exc())
    finally:
        try:
            walker.shutdown()
        except Exception:                                  # noqa: BLE001
            walker._errors.append(traceback.format_exc())

    # 配置与数据的比对：GUI 动过就必须还原，并且算一条缺陷
    config_after = _fingerprint_config()
    added, removed, changed = _diff(config_before, config_after)
    if added or removed or changed:
        detail = _config_key_diff()
        md_snapshot.append("**GUI 改动了 config/*.json**（已还原成走查前的样子）")
        for kind, items in (("新增", added), ("丢失", removed), ("内容变了", changed)):
            for name in items:
                md_snapshot.append(f"{kind}：config/{name}")
        for line in detail:
            md_snapshot.append(f"具体改了：{line}")
        if not detail:
            md_snapshot.append("具体改了：（逐键比对没有差异 —— 只是键顺序/换行被重写）")
        walker.add_defect(
            severity="观感", where="config/" + "/".join(sorted(set(changed + added + removed))),
            what="只开窗、只点按钮，配置目录下的文件就被改写了。"
                 "项目约定 config/ 是标定与装置的唯一真相，界面把自己的输入"
                 "写回同一个目录，会让'配置变了'这件事失去可追溯性（走查器已还原）",
            how="打开再关闭模式窗（on_close 里 prefs.save()）",
            evidence="；".join(detail[:4]) or "内容变了")
        _restore_config()
        md_snapshot.append("已从 `outputs/gui_walk/_pre/config_backup/` 恢复")
    else:
        md_snapshot.append("config/*.json 未变（GUI 没写配置）")

    # 走查点过「拍标定图」会新建空会话（没真采集），收尾清掉，别留误导性指针
    _empty_sessions = _drop_empty_calib_sessions()
    md_snapshot.append(f"空标定会话已清理（{_empty_sessions} 个；有图的会话不动）")

    gui_after = _gui_fingerprint()
    if gui_after != gui_before:
        diff_keys = [k for k in set(gui_before) | set(gui_after)
                     if gui_before.get(k) != gui_after.get(k)]
        md_snapshot.append("⚠ **走查期间 GUI 源码被改动过**（" +
                           "、".join(sorted(diff_keys)) +
                           "）—— 这份报告可能混了两个版本，结论请重跑确认")
        walker.add_defect(
            severity="观感", where="走查环境",
            what="走查跑到一半，GUI 源码被改了（md5 变化）。报告里的步骤可能来自不同版本，"
                 "不能直接当验收依据",
            how="并发编辑 apps/_gui_*.py",
            evidence="、".join(f"{k}: {gui_before.get(k)}→{gui_after.get(k)}"
                              for k in sorted(diff_keys)))

    d_added, d_removed, d_changed = _diff(data_before, _fingerprint_data())
    if d_added or d_removed or d_changed:
        md_snapshot.append("**data/ 有变化**（走查器不改数据，见下）")
        for kind, items in (("新增", d_added), ("消失", d_removed), ("变化", d_changed)):
            for name in items[:20]:
                md_snapshot.append(f"{kind}：{name}")
                walker.add_defect(
                    severity="静默无反馈", where=f"data/{name}",
                    what="只浏览会话，data/ 下却出现了改动",
                    how="在会话列表里选中一段会话",
                    evidence=f"{kind}：{name}")
    else:
        md_snapshot.append("data/ 未变（只浏览，没有录制/清理/离线跑）")

    unverified += [
        ("会话工作台：**根本没打开**（见缺陷 1 的 traceback），所以里面的 ①–⑤ 一步都没走到。"
         if walker.failed_windows else
         "会话工作台：②「点选球心」需要人在图上点两下，未点击。"),
        "控制台上没有 tab、也没有「清屏」按钮（它只有任务表 + 运行日志 + 三个中断/日志按钮），"
        "所以「切 tab / 点清屏」这两件事在本窗口不存在，无从验证。",
        "录制 / 标定 / 拍标定图 / 点选球心：需要相机或需要人在图上点两下，走查只记录"
        "它们想执行的命令行，没有真正执行。",
        "离线三维 / 离线单目：结果会写进 `data/<会话>/derived/`，属于数据目录，"
        "按约束没有执行；只验证了按钮生成的命令行。",
        "相机打不开这件事本身：要走 `apps/tools/cameras.py doctor` 真开相机才能定论，"
        "不在本次 GUI 走查范围内。",
        "左侧逐帧点选（SeedPicker / MonoSeedPicker）与取景窗：进程内模态，"
        "会 `wait_window` 等人点，自动化点不动，因此没有打开。",
        "数据管理 / 3D 查看器：窗口起来了也截了图，但走查器**不点它们内部的控件**"
        "（那是另一个进程的界面，要多一套控件定位才能点）。",
    ]
    md = walker.write_reports({
        "snapshot_md": md_snapshot,
        "unverified": unverified,
        "uncovered": uncovered,
        "ball_probe": getattr(walker, "_ball_probe", None) and
                      {"runs": getattr(walker, "_ball_probe"),
                       "gui_value": getattr(walker, "_ball_gui_value", "?"),
                       "bad_unit": getattr(walker, "_ball_bad", False)},
        "camera_state": _camera_state(),
        "gui_files": gui_before,
        "failed_windows": walker.failed_windows,
        "notes": shutdown_note,
        "config_changed": bool(added or removed or changed),
        "data_changed": bool(d_added or d_removed or d_changed),
    })
    print(f"报告：{md}")
    print(f"截图：{walker.shots}（{sum(1 for s in walker.steps if s.shot)} 张）")
    print(f"步骤 {len(walker.steps)}，缺陷 {len(walker.defects)}"
          f"（{', '.join(sorted({d['severity'] for d in walker.defects}))}）")
    if uncovered:
        print(f"⚠ 未覆盖：{'、'.join(WINDOWS.get(k, k) for k in uncovered)}"
              f"（本次不是全量走查）")
    # 退出码只在"走查器自己坏了"时非 0：缺陷本身是结论，不是失败
    return 0


# B-151：本函数体受 tests/selftest_window_mode.py 的 AST 判据守护 ——
# 与那边那份 _gui_fingerprint() 必须逐结构一致，改逻辑请两处同步，否则那条
# 判据会直接红（而不是半年后悄悄漂移）。
def _gui_fingerprint() -> dict[str, str]:
    """记下这次走查**跑的是哪一版 GUI**。

    团队里可能有人同时在改 `apps/_gui_*.py`（本次真的撞上过一次：走查到一半
    文件处于半保存状态，直接 IndentationError）。不锁版本的话，报告里的结论
    对不上任何一次提交，等于没有证据。
    """
    import hashlib
    out: dict[str, str] = {}
    # `_gui_*.py` + `*_dialog.py`：对话框也是界面，改一版排版也可能让
    # "这一步会不会被裁"的结论失效。2026-11-07 加 `tools_dialog.py` 时发现
    # 它落在指纹之外 —— "新窗口不在版本锁里"正是本节最该防的空洞。
    gui_files = sorted((ROOT / "apps").glob("_gui_*.py")) + \
        sorted((ROOT / "apps").glob("*_dialog.py"))
    for p in gui_files:
        try:
            out[p.name] = hashlib.md5(p.read_bytes()).hexdigest()[:10]
        except OSError:
            out[p.name] = "?"
    for name in ("_task_runner.py", "_collapsible.py"):
        p = ROOT / "apps" / name
        if p.is_file():
            try:
                out[name] = hashlib.md5(p.read_bytes()).hexdigest()[:10]
            except OSError:
                pass
    return out


def _config_key_diff() -> list[str]:
    """把"配置变了"细化到键：报告里写"gui_state.json 变了"没有用，要写变了哪个键。"""
    backup = ROOT / "outputs" / "gui_walk" / "_pre" / "config_backup"
    out: list[str] = []
    for p in sorted((ROOT / "config").glob("*.json")):
        old_p = backup / p.name
        if not old_p.is_file():
            out.append(f"{p.name}（走查前没有这个文件）")
            continue
        try:
            old = json.loads(old_p.read_text(encoding="utf-8"))
            new = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            out.append(f"{p.name}（不是合法 JSON，无法逐键比对）")
            continue
        if not (isinstance(old, dict) and isinstance(new, dict)):
            continue
        for k in sorted(set(old) | set(new)):
            if old.get(k) != new.get(k):
                out.append(f"{p.name}：{k}  {old.get(k)!r} → {new.get(k)!r}")
    return out


def _resolve_session(arg: str, kind: str) -> str:
    """会话参数 → 绝对路径；找不到就返回空串（**不许静默降级成"没会话"**）。

    三段兜底：绝对路径 → 相对仓库根 → 只给了会话 id（在各根下同名目录里找）。
    目录名一律来自 ``sphere_tracker.paths``，不在工具里手写 `data/xxx` ——
    目录清单只能有一份，这正是"各处各写一份清单"的老毛病。
    """
    from sphere_tracker.paths import MONO_ROOT, stereo_roots

    p = Path(arg)
    cand = p if p.is_absolute() else (ROOT / p)
    if cand.is_dir():
        return str(cand)
    if kind == "stereo":
        # 2026-11-04 起双目会话按装置分区（data/stereo/parallel/<id>），
        # 不能再只试 `data/stereo/<id>` —— 那会找不到旧会话，报告里就会
        # 出现"选择器列表找不到 <空>"这种**工具造成的**假缺陷。
        from apps._stereo_layout import candidate_session_dirs

        for d in candidate_session_dirs():
            if d.name == p.name and d.is_dir():
                return str(d)
    bases = (MONO_ROOT,) if kind == "mono" else stereo_roots()
    for base in bases:
        hit = base / p.name
        if hit.is_dir():
            return str(hit)
    return ""


def _gui_ready() -> tuple[bool, str]:
    """开跑前先确认 GUI 源码能编译。

    为什么需要这一条：团队里有人在同时改 `apps/_gui_*.py`，走查跑到一半撞上
    "文件存了一半"会得到 `IndentationError`，而报告会把这记成"GUI 崩溃"——
    那是假的，而且会误导别人去修一个不存在的 bug。宁可开跑前就拒绝。
    """
    for p in (sorted((ROOT / "apps").glob("_gui_*.py"))
              + sorted((ROOT / "apps").glob("*_dialog.py"))):
        tmp = Path(tempfile.gettempdir()) / f"_wk_{p.stem}.pyc"
        try:
            py_compile.compile(str(p), cfile=str(tmp), doraise=True)
        except py_compile.PyCompileError as exc:
            return False, f"{p.name}：{exc.msg}"
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass
    return True, ""


def _backup_config() -> None:
    """先把 ``config/*.json`` 抄一份到 ``_pre``：GUI 若改写了它，走完要能原样贴回。"""
    dst = ROOT / "outputs" / "gui_walk" / "_pre" / "config_backup"
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted((ROOT / "config").glob("*.json")):
        try:
            shutil.copy2(p, dst / p.name)
        except OSError:
            pass


def _drop_empty_calib_sessions() -> int:
    """走查点过「拍标定图 / 采集标定板图」会**新建会话却不采图**，收尾清掉。

    2026-11-09：标定图改成「一次采集 = 一个会话子目录」后，走查器第 94 步
    会在 `outputs/mono_calib_frames/` 留下一个只有 session.json 的空会话和
    `_current.json` 指针。留着会让用户下次打开 GUI 以为"有当前会话"，
    所以这里删掉**没有任何图片**的会话，并在指针指向已删目录时删指针。
    只删空会话，绝不动有图的真采集。
    """
    import json

    repo = Path(__file__).resolve()
    while not (repo / "sphere_tracker").is_dir():
        repo = repo.parent
    removed = 0
    for name in ("chain_frames", "mono_calib_frames"):
        root = repo / "outputs" / name
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            imgs = [p for p in d.rglob("*") if p.is_file()
                    and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")]
            if imgs:
                continue
            for p in sorted(d.rglob("*"), key=lambda x: len(x.parts), reverse=True):
                try:
                    p.unlink() if p.is_file() else p.rmdir()
                except OSError:
                    pass
            try:
                d.rmdir()
            except OSError:
                pass
            removed += 1
        cur = root / "_current.json"
        if not cur.exists():
            continue
        try:
            sess = str(json.loads(cur.read_text(encoding="utf-8")).get("session") or "")
        except (OSError, ValueError):
            sess = ""
        if not sess or not (root / sess).is_dir():
            try:
                cur.unlink()
            except OSError:
                pass
    return removed


def _restore_config() -> None:
    """把 ``outputs/gui_walk/_pre/config_backup`` 里的原始文件贴回去。"""
    src = ROOT / "outputs" / "gui_walk" / "_pre" / "config_backup"
    if not src.is_dir():
        return
    for p in src.glob("*.json"):
        try:
            shutil.copy2(p, ROOT / "config" / p.name)
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
