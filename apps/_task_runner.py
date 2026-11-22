"""任务进程管理：启动、读输出、逐行落盘、停整棵进程树。

为什么单独一个模块（2026-10-27 计划书 §4 P2）
--------------------------------------------
这段逻辑原来在 ``apps/gui_main.py`` 的 ``Launcher`` 里（约 560 行）。分窗之后有
5 个窗口要看同一张任务表 —— 如果每窗各管一份子进程，本项目最贵的那个反模式
（"同一件事 N 份实现"）会再犯一次。抽出来之后：

- **零 Tk 依赖**：不 import tkinter；定时轮询用注入的 ``schedule`` 函数，
  因此可以在无显示器环境（CI）里被测试；
- **一个进程一份**：``App`` 持有唯一实例，各窗口只订阅 ``on_log`` / ``on_change``；
- 对外只有 ``start / stop / stop_all / tasks / running / on_log / on_change``
  以及**统一结束回调** ``add_finish_listener / remove_finish_listener``。

搬过来的每个"为什么"注释都保留 —— 那些是这个模块唯一的价值：
``taskkill`` 返回 0 不代表杀掉了、``Popen.terminate`` 只杀直接子进程、
日志要按任务分开落盘。删掉注释等于删掉后来者避开同一个坑的机会。

内置自测：``python apps/_task_runner.py --selftest``（不需要窗口，不需要相机）。
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
from typing import Callable

__all__ = ["ROOT", "Task", "TaskRunner", "alive_pids", "child_pids", "POLL_MS",
           "TAIL_LINES"]

#: 按标记上溯（`sphere_tracker/` 在哪，根就在哪），不数层数 —— 文件被移动时
#: `parent.parent` 会静默指到别处；理由同 `sphere_tracker.paths.ROOT`。
ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "sphere_tracker").is_dir())
#: 根进 sys.path 才能 import `sphere_tracker.paths` —— 日志路径的唯一来源。
#: 没有这一段，`python apps/_task_runner.py --selftest` 会因为 sys.path[0]
#: 是 apps/ 而找不到包。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sphere_tracker.paths import log_dir as _log_dir  # noqa: E402
from apps._task_registry import (  # noqa: E402
    external_running as _external_running,
    register_end,
    register_start,
)

POLL_MS = 200

#: 任务结束时留在 ``Task.tail`` 里的最后几行输出。
#:
#: 为什么留尾巴而不是"到日志里自己翻"：失败时用户要的是**当场一句原因**。
#: 8 行的上限是实测折中 —— argparse 的报错在最后两行，OpenCV 的
#: "can't open camera"也在一屏之内，再多就只是噪声。
TAIL_LINES = 8


# --------------------------------------------------------------------------- #
# 进程树工具
# --------------------------------------------------------------------------- #
def alive_pids(pids: list[int]) -> list[int]:
    """这批 pid 里**还有哪些活着**（判不出来的一律当成"还活着"）。

    为什么需要它（而不是直接相信 ``taskkill`` 没抛异常）：
    ``subprocess.run(["taskkill", ...])`` **不抛异常不代表杀掉了** ——
    权限不足时它只是返回 1 并把 ``ERROR: Access denied`` 写进 stderr，
    而 ``capture_output=True`` 会把这个返回值与输出一起吞掉。
    于是"杀失败"与"杀成功"长得一模一样 —— **失败没有出口**。

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


def child_pids(pid: int) -> list[int]:
    """某个进程的子进程 pid（拿不到就返回空）。

    为什么需要：``Popen.terminate()`` **只杀直接子进程**。我们的子进程是
    python，它可能自己还有子进程（OpenCV 辅助线程、或程序再起的进程）。
    **父进程死了子进程还活着**，于是相机仍被占着 ——
    用户看到"停了但相机打不开"。所以停任务要杀**整棵树**。
    """
    try:
        import psutil                                       # 可选依赖
    except ImportError:
        return []
    try:
        return [c.pid for c in psutil.Process(pid).children(recursive=True)]
    except Exception:                                       # noqa: BLE001
        return []


# --------------------------------------------------------------------------- #
# 任务对象
# --------------------------------------------------------------------------- #
@dataclass
class Task:
    """一个正在跑（或跑完）的外部程序。

    为什么不只留一个 ``proc`` 句柄：那样一次只能跑一个、且没有任何中断手段 ——
    程序卡住（相机被占、等键盘输入）时用户只能关窗口，而**关窗口不会杀掉子进程**：
    相机仍被占着，下一次打不开相机，用户不知道为什么。

    ``log_path``：每个任务把完整输出落盘一份。界面日志会被下一条冲掉，
    而"上一次到底报了什么"恰恰是出问题时唯一能查的东西。
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
    kill_note: str = ""          # 停止失败的原因（"没杀干净"必须有出口）
    lines: int = 0
    stdout_lines: int = 0        # 子进程真正吐出的行数（判"启动即退且没输出"）
    on_done: Callable[["Task"], None] | None = None
    #: 谁起的任务（``"mono"`` / ``"stereo"`` …）。空串 = 没登记归属。
    #:
    #: 为什么需要它：统一结束回调是**全局**的（一个进程一份 TaskRunner），
    #: 没有归属时单目窗会收到双目窗任务的失败 —— 状态条在别人的窗口里
    #: 说别人的事，比不说更让人糊涂。按标题字符串匹配又会被"标题改一个字"
    #: 悄悄弄坏，所以归属必须是**显式写上去的**（窗口起完任务就填这一栏）。
    owner: str = ""
    #: 最后 ``TAIL_LINES`` 行输出：失败时状态条要能直接引用原因。
    tail: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """正常跑完（返回码 0，且不是被用户停掉的）。

        为什么把"被停掉"排除在成功外：``stop()`` 之后 ``proc.wait()`` 可能
        得到 0（子进程自己收拾干净退出），但那**不是任务的结果**，
        界面上写"完成"会让用户以为结果可用。
        """
        return self.exit_code == 0 and not self.stop_reason

    def tail_text(self, n: int = 3) -> str:
        """最后 ``n`` 行拼成一句（没有输出时给一句人话，不留空）。"""
        if not self.tail:
            return "（没有任何输出）"
        return " / ".join(self.tail[-n:])

    @property
    def running(self) -> bool:
        return self.proc.poll() is None

    def elapsed(self) -> float:
        end = self.ended if self.ended is not None else time.monotonic()
        return max(0.0, end - self.started)

    def cmdline(self) -> str:
        return " ".join(self.cmd)


# --------------------------------------------------------------------------- #
# 任务运行器
# --------------------------------------------------------------------------- #
class TaskRunner:
    """一个进程一份实例。所有窗口通过回调订阅，不各自持有子进程。

    ``schedule`` 默认用 ``root.after``（Tk）；无界面环境下传 ``None`` 并调用
    :meth:`pump_once` 手动驱动 —— 自测就是这么做的。
    """

    def __init__(self, root=None, *, on_log: Callable[[str], None] | None = None,
                 on_change: Callable[[], None] | None = None,
                 log_dir: Path | None = None, poll_ms: int = POLL_MS) -> None:
        self._root = root
        self._on_log = on_log
        self._on_change = on_change or (lambda: None)
        # 默认 logs/tasks/<日期>/：启动器这一层的日志单独一个抽屉，
        # 与各程序自己的日志分开（见 sphere_tracker.paths.log_dir）。
        self._log_dir = log_dir if log_dir is not None else _log_dir("tasks")
        self._poll_ms = poll_ms
        self._tasks: list[Task] = []
        self._tid = 0
        self._sel_tid: int | None = None
        self._log_queue: queue.Queue = queue.Queue()
        self._done_queue: queue.Queue = queue.Queue()
        self._finish_listeners: list[Callable[[Task], None]] = []
        self._stopping = False
        #: 上一次发给任务表的状态签名（见 :meth:`_task_state_signature`）。
        #: 只在签名变化时发 ``_on_change`` —— 否则 200ms 一次的全表重建会
        #: 把任务表拖卡（原先"只要还有任务在跑就每帧发"就是这个毛病）。
        self._last_state_signature: tuple | None = None
        if root is not None:
            root.after(poll_ms, self._poll)

    # ---------- 统一结束回调 ----------
    def add_finish_listener(self, cb: Callable[[Task], None]) -> None:
        """订阅"**任何一个任务结束**"（成功与失败都来）。

        为什么需要它（2026-10-31 现场）：窗口里"点完按钮就再无下文"是用户
        最不能接受的状态。以前每个按钮要各自传 ``on_done``，漏传一个
        （拍标定图 / 焦距估计 / 实时识别全都漏了）就永远停在"已开始…"，
        而失败时更是连一句"失败了"都没有 —— **失败没有出口**。
        这里给一条统一出口：谁关心结局就订一次，而不是每加一个按钮再写一份
        回调（"同一件事 N 份实现"正是本项目最贵的反模式）。

        回调在**主线程**执行（由 ``pump_once`` 从 Tk 的 ``after`` 驱动），
        所以可以直接碰控件；回调抛异常只记账，不影响其它订阅者。
        """
        if cb not in self._finish_listeners:
            self._finish_listeners.append(cb)

    def remove_finish_listener(self, cb: Callable[[Task], None]) -> None:
        """退订。窗口销毁前必须调 —— 否则回调会在已销毁的窗口上碰控件。"""
        if cb in self._finish_listeners:
            self._finish_listeners.remove(cb)

    def _notify_finished(self, task: Task) -> None:
        for cb in list(self._finish_listeners):
            try:
                cb(task)
            except Exception as exc:                        # noqa: BLE001
                self._emit(f"[任务 {task.tid}] 结束回调失败：{exc}")


    # ---------- 查询 ----------
    def tasks(self) -> list[Task]:
        return list(self._tasks)

    def running(self) -> list[Task]:
        return [t for t in self._tasks if t.running]

    def external_running(self) -> list[dict]:
        """别的进程登记进来、且 PID 仍活着的任务（见 ``apps/_task_registry``）。

        控制台把 ``running()`` 与这一份并成"正在运行"列表 —— 用户报的
        "很多任务没进管理器"里，数据管理导出/查看器就是走这条路径。
        """
        return _external_running()

    @property
    def selected_tid(self) -> int | None:
        return self._sel_tid

    def select(self, tid: int) -> None:
        self._sel_tid = tid

    # ---------- 启动 ----------
    def _spawn(self, cmd: list[str]) -> subprocess.Popen:
        """起一个**可以被整树杀掉**的子进程。

        ``CREATE_NEW_PROCESS_GROUP`` 是"能优雅中断"的前提：只有新进程组才能
        收到 ``CTRL_BREAK_EVENT``。没有它，中断只能换成硬杀 ——
        而硬杀对"正在写标定结果"的程序是有害的。

        ``stdin=DEVNULL`` 是**兜底**，不是洁癖：子进程继承控制台 stdin 时，
        任何一句 ``input()``（实测 ``stereo/pair_source.py`` 在"没有 tty"的
        路径上就会等一个回车）会**无限期挂住** —— 任务永远不结束、状态条
        永远停在"已开始…"、父进程还拿着相机。库层已经把那条交互改成非交互了，
        但这里是**所有**子进程的出口：以后谁再写一句 ``input()`` 也只会立刻
        拿到 EOF 并报错退出，而不是把界面吊死。
        """
        kw: dict = dict(cwd=str(ROOT), stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                        text=True, encoding="utf-8", errors="replace", bufsize=1)
        if os.name == "nt":
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kw["start_new_session"] = True
        return subprocess.Popen(cmd, **kw)

    def start(self, cmd: list[str], title: str,
              on_done: Callable[[Task], None] | None = None,
              owner: str = "") -> Task:
        """起一个新任务并登记。**允许多个同时跑**（相机冲突由用户自己负责）。

        ``owner`` 是起任务那个窗口的归属键（``"mono"`` / ``"stereo_parallel"`` …）。
        默认值空串 = 不登记归属 —— 所有既有调用点都不传，行为逐字不变；
        需要"关这个窗口就停它的任务"的调用点才显式传它（配 :meth:`stop_by_owner`）。
        """
        self._tid += 1
        proc = self._spawn(cmd)
        log_path = self._task_log_path(self._tid, title)
        task = Task(tid=self._tid, title=title, cmd=list(cmd), proc=proc,
                    log_path=log_path, on_done=on_done, owner=owner)
        self._tasks.append(task)
        self._sel_tid = task.tid
        # 跨进程账本：控制台的"正在运行"也包含这里登记的任务。
        # 必须在 _spawn 之后（要有 pid），失败也不许影响任务本身。
        register_start(task, source="runner")
        self._emit(f"\n===== [{task.tid}] {title} =====\n> {task.cmdline()}")
        self._emit(f"[任务 {task.tid}] 已启动，PID {proc.pid}"
                   f"（可在「任务」里中断）")
        if log_path:
            self._emit(f"[任务 {task.tid}] 完整日志：{log_path}")
            try:
                log_path.write_text(
                    f"任务 {task.tid}：{title}\n"
                    f"启动：{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"PID ：{proc.pid}\n"
                    f"命令：{task.cmdline()}\n"
                    + "=" * 70 + "\n", encoding="utf-8")
            except OSError as exc:
                self._emit(f"[任务 {task.tid}] 警告：日志文件写不了：{exc}")
                task.log_path = None
        threading.Thread(target=self._reader, args=(task,), daemon=True).start()
        self._on_change()
        return task

    # ---------- 日志 ----------
    def _emit(self, text: str) -> None:
        if self._on_log is not None:
            try:
                self._on_log(text)
            except Exception:                               # noqa: BLE001
                pass                                        # UI 挂了不该拖死任务

    def _task_log_path(self, tid: int, title: str) -> Path | None:
        """给每个任务一个日志文件：``logs/tasks/<日期>/task<序号>_<时间>_<标题>.log``。

        **为什么按任务分开**：同一个 ``logs/`` 下已经有各程序自己的日志
        （``calibrate_*.log`` 等），但启动器这一层的输出（谁起的、什么时候、
        被谁停的、退出码多少）只有这里才有。
        """
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            safe = re.sub(r'[\\/:*?"<>|\s]+', "_", title).strip("_")[:40] or "task"
            stamp = time.strftime("%Y%m%d_%H%M%S")
            return self._log_dir / f"task{tid:02d}_{stamp}_{safe}.log"
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
            task.log_path = None                            # 只报一次，不刷屏

    # ---------- 停止 ----------
    def _force_kill_tree(self, pid: int) -> str:
        """强杀一棵进程树，**返回为什么失败**（成功时返回空串）。

        这里必须检查 ``returncode`` 与 ``stderr``：走到这个函数时通常
        "优雅退出没起作用"，如果强杀也失败，**那就真的没停掉** ——
        而旧实现是 ``subprocess.run(..., capture_output=True)`` 之后
        两个都不看，于是权限不足（实测 stderr 为 ``ERROR: Access denied``）
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

    def stop(self, task: Task, graceful: bool = True) -> bool:
        """停掉一个任务（含子进程树）。返回**是否确认已经停掉**。

        返回值语义：不是"是否下手了"（那样永远为真、无从分辨），
        而是"**确认停掉了吗**" —— 强杀之后复查一次进程树，还在就返回 False
        并把原因写进 ``task.kill_note``。
        """
        if not task.running:
            # 进程已经没了（读线程可能还没来得及写 end）——补一笔，
            # 免得它一直挂在"正在运行"账本上靠 PID 探活兜底。
            register_end(task, source="runner")
            return True
        pid = task.proc.pid
        task.stop_reason = "用户中断" + ("（优雅）" if graceful else "（强制）")
        self._emit(f"[任务 {task.tid}] 正在中断 PID {pid}…")
        kids = child_pids(pid)
        if kids:
            self._emit(f"[任务 {task.tid}] 它有 {len(kids)} 个子进程，将一并结束")

        # ① 先请求优雅退出
        if graceful:
            try:
                if os.name == "nt":
                    task.proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(os.getpgid(pid), signal.SIGINT)
            except Exception as exc:                        # noqa: BLE001
                self._emit(f"[任务 {task.tid}] 优雅中断不可用"
                           f"（{type(exc).__name__}），将直接强制结束")
            try:
                task.proc.wait(timeout=3.0)
                self._emit(f"[任务 {task.tid}] 已优雅退出")
                self._kill_leftover(kids, task)
                ok = self._confirm_dead(task, kids)
                if ok:
                    register_end(task, source="runner")
                return ok
            except subprocess.TimeoutExpired:
                self._emit(f"[任务 {task.tid}] 3 秒内没退出，改为强制结束")

        # ② 强杀（含子树）
        why = self._force_kill_tree(pid)
        if why:
            self._emit(f"[任务 {task.tid}] 警告：强制结束没成功：{why}")
            task.kill_note = f"强制结束未成功（{why}）"
        else:
            self._emit(f"[任务 {task.tid}] 已强制结束整棵进程树")
        self._kill_leftover(kids, task)
        ok = self._confirm_dead(task, kids)
        if ok:
            register_end(task, source="runner")
        return ok

    def _confirm_dead(self, task: Task, kids: list[int],
                      wait: float = 2.0) -> bool:
        """等一小会儿再复查：父进程与子进程**真的都没了**才算停掉。

        为什么不能杀完就返回 True：``taskkill`` 可能返回非零、也可能
        "命令成功但进程仍卡在不可中断状态"。``alive_pids`` 查不出来时也算活着，
        所以这里**只会偏保守**：最坏情况是多报一次"可能没停干净"。
        """
        deadline = time.monotonic() + wait
        while True:
            alive = alive_pids([task.proc.pid, *kids])
            if not alive:
                return True
            if time.monotonic() >= deadline:
                task.kill_note = (f"仍有 {len(alive)} 个进程存活：{alive}"
                                  + (f"（{task.kill_note}）" if task.kill_note else ""))
                self._emit(
                    f"[任务 {task.tid}] 警告：停完仍有 {len(alive)} 个进程存活"
                    f"（{alive}）—— 相机可能仍被占着，请确认后再启动新任务")
                return False
            time.sleep(0.05)

    def _kill_leftover(self, kids: list[int], task: Task) -> None:
        """收尾：子树里还有活着的就补刀（否则相机可能仍被占着）。"""
        alive = alive_pids(kids)
        if not alive:
            return
        self._emit(f"[任务 {task.tid}] 仍有 {len(alive)} 个子进程存活，"
                   f"补杀：{alive}")
        for pid in alive:
            why = self._force_kill_tree(pid)
            if why:
                self._emit(f"[任务 {task.tid}] 警告：子进程 {pid} 补杀失败：{why}")

    def stop_all(self, graceful: bool = False) -> int:
        """停掉所有在跑的任务，返回**确认停掉**的个数。

        返回"确认停掉"而不是"尝试停掉"：没停掉的会留在任务表里
        （状态显示失败原因），而不是从计数里消失。
        """
        running = self.running()
        ok = 0
        for t in running:
            if self.stop(t, graceful=graceful):
                ok += 1
        if ok < len(running):
            self._emit(f"[任务] 警告：{len(running)} 个任务里只有 {ok} 个确认停掉，"
                       f"其余请看上面的失败原因")
        return ok

    def stop_by_owner(self, owner: str, graceful: bool = True) -> int:
        """只停 ``owner`` 属于这个窗口的任务；返回**确认停掉**的个数。

        为什么不能复用 stop_all：关一个查看器会把别人的离线重建也杀掉。
        三个窗口的 ``_stop_my_tasks`` 原来是全局停，用户的典型场景是
        "单目窗开着、双目离线重建在后台跑"，关单目窗却把双目任务一起杀了。
        归属必须精确 —— 所以这里按 :attr:`Task.owner` 逐个筛，再逐个 :meth:`stop`。

        空 owner 直接返回 0：没登记归属的任务不属于任何窗口，绝不能被
        "按归属停"误伤（例如控制台那条空 owner 的兜底路径）。
        """
        if not owner:
            return 0
        targets = [t for t in self.running()
                   if getattr(t, "owner", "") == owner]
        n = 0
        for t in targets:
            if self.stop(t, graceful=graceful):
                n += 1
        return n

    # ---------- 读输出与轮询 ----------
    def _reader(self, task: Task) -> None:
        proc = task.proc
        assert proc.stdout is not None
        for line in proc.stdout:
            body = line.rstrip()
            task.stdout_lines += 1
            task.tail.append(body)
            if len(task.tail) > TAIL_LINES:                 # 只要尾巴，别把整段留在内存
                del task.tail[0]
            self._task_log(task, body)                      # 每行都落盘
            self._log_queue.put(f"[{task.tid}] {body}")
        proc.wait()
        task.ended = time.monotonic()
        task.exit_code = proc.returncode
        register_end(task, source="runner")
        why = f"（{task.stop_reason}）" if task.stop_reason else ""
        tail = (f"[任务 {task.tid} 结束] 返回码 {proc.returncode}{why}，"
                f"耗时 {task.elapsed():.1f}s")
        self._task_log(task, "=" * 70)
        self._task_log(task, tail)
        if task.log_path is not None:
            self._task_log(task, f"日志文件：{task.log_path}")
        self._log_queue.put(tail)
        self._done_queue.put(task.tid)

    def _report_failure(self, task: Task) -> None:
        """非 0 退出时说人话；**启动即退**的要直接指出"多半是参数不对"。

        为什么必须有这一条：界面上点了按钮之后"什么都没发生"是用户最不能接受的
        反馈（2026-10-28 用户报"3D 查看器点了没反应"）。根因是子进程 argparse
        直接 exit 2，而错误只落在日志里、没人会去翻。
        这一条是**机制层兜底**：下次谁再漏传一个必填参数，也会被它抓住，
        不用等用户发现"点了没反应"。
        """
        if task.exit_code in (0, None):
            return
        if task.elapsed() < 5.0 and task.stdout_lines <= 2:
            self._emit(
                f"[任务 {task.tid}] 启动即失败（返回码 {task.exit_code}，几乎没有输出）"
                f" —— 多半是参数不对，不是程序坏了。命令：{task.cmdline()}")
        else:
            self._emit(
                f"[任务 {task.tid}] 失败：返回码 {task.exit_code}；"
                f"完整输出：{task.log_path or '（未落盘）'}")

    def _task_state_signature(self) -> tuple:
        """任务表真正关心的状态签名。

        包含哪些在跑（``running``）与每个任务的结束/中断字段
        （``ended`` / ``exit_code`` / ``stop_reason`` / ``kill_note``）。
        ``stop()`` 是**同步**写好 ``stop_reason`` 的，而进程可能还要几秒才真正
        退出 —— 签名把这些字段也纳入，用户点"中断"的那一刻任务表就刷新，
        不必等进程死的那一刻（这正是"关窗停任务后状态不更新"的根因）。

        还在跑的任务额外放秒级耗时 ``int(t.elapsed())``：用户要"观看实时任务栏"，
        "已运行"列必须走秒。取秒而不是毫秒 —— 签名每变一次就会触发
        console.refresh() 全表重建，按秒变化正好兼顾走秒与不卡。
        已结束的任务 elapsed 不再变，不放入（避免无意义地反复比同一串值）。
        """
        return tuple(
            (t.tid, t.running, t.ended, t.exit_code, t.stop_reason, t.kill_note,
             int(t.elapsed()) if t.running else 0)
            for t in self._tasks
        )

    def pump_once(self) -> None:
        """把队列里的日志与完成事件排空一次（可手动驱动，供无界面测试）。"""
        # 基线必须是**上一次 pump 结束时**的签名，不能在本帧开头重算：
        # stop() 是被 Tk 回调在两次 pump 之间调用的，若本帧开头重算基线，
        # stop_reason 的突变就已经"算进基线"了 → 刷新永远不发生。
        before = self._last_state_signature
        while True:
            try:
                line = self._log_queue.get_nowait()
            except queue.Empty:
                break
            self._emit(line)
        changed = False
        while True:
            try:
                tid = self._done_queue.get_nowait()
            except queue.Empty:
                break
            task = next((t for t in self._tasks if t.tid == tid), None)
            changed = True
            if task is not None:
                self._report_failure(task)
                # 顺序有意为之：先统一出口（"失败 + 日志路径"），再跑这个任务
                # 专门的 on_done —— 后者知道更多上下文，允许把话术说得更准
                # （例如"一帧都没检出，换 hough"比"返回码 2"有用）。
                self._notify_finished(task)
            if task is not None and task.on_done is not None:
                try:
                    task.on_done(task)
                except Exception as exc:                    # noqa: BLE001
                    self._emit(f"[任务 {task.tid}] 完成回调失败：{exc}")
        after = self._task_state_signature()
        # 有完成事件、或状态签名变了才发一次。不再"只要还有任务在跑就每帧发"：
        # 那会让 console.refresh() 在 200ms 的 after 里反复全表重建。
        should_emit = changed or after != before
        self._last_state_signature = after   # 无条件推进基线（含首帧）
        if should_emit:
            self._on_change()

    def _poll(self) -> None:
        self.pump_once()
        if self._root is not None and not self._stopping:
            self._root.after(self._poll_ms, self._poll)

    def shutdown(self, *, graceful: bool = False, timeout: float = 6.0) -> int:
        """关窗口时调用：停掉全部任务并停止轮询。返回确认停掉的个数。"""
        self._stopping = True
        ok = self.stop_all(graceful=graceful)
        deadline = time.monotonic() + timeout
        while self.running() and time.monotonic() < deadline:
            time.sleep(0.05)
        return ok


# --------------------------------------------------------------------------- #
# 内置自测（无窗口、无相机；不产任何文件到仓库里 —— 日志目录用临时目录）
# --------------------------------------------------------------------------- #
def _run_selftest() -> int:
    import tempfile

    print("=" * 70)
    print("_task_runner 自测：启动 → 收到 stdout → 退出码 → 停树杀干净")
    print("=" * 70)
    lines: list[str] = []
    changes: list[int] = []
    with tempfile.TemporaryDirectory() as td:
        runner = TaskRunner(root=None, on_log=lines.append,
                            on_change=lambda: changes.append(1),
                            log_dir=Path(td))
        py = sys.executable
        # ① 正常跑完：stdout 要能收到，退出码要能拿到
        t = runner.start([py, "-c", "print('hello-42'); print('bye')"], "自测-正常")
        deadline = time.monotonic() + 20
        while t.running and time.monotonic() < deadline:
            runner.pump_once()
            time.sleep(0.05)
        runner.pump_once()
        got = "\n".join(lines)
        checks = [
            ("收到子进程 stdout", "hello-42" in got),
            ("退出码 = 0", t.exit_code == 0),
            ("日志文件已落盘", t.log_path is not None and t.log_path.exists()),
            ("on_change 被调用过", len(changes) > 0),
        ]
        # ② 停一棵树：父进程再起一个长睡子进程，stop 之后两个都得没
        code = ("import subprocess,sys,time;"
                "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
                "print('parent-up',p.pid,flush=True);time.sleep(60)")
        t2 = runner.start([py, "-c", code], "自测-需要中断")
        deadline = time.monotonic() + 20
        while t2.running and time.monotonic() < deadline:
            runner.pump_once()
            time.sleep(0.05)
        kids_before = child_pids(t2.proc.pid)
        stopped = runner.stop(t2, graceful=True)
        runner.pump_once()
        checks.append(("停任务返回确认停掉", stopped))
        checks.append(("停完父进程真的没了", not alive_pids([t2.proc.pid])))
        if kids_before:
            checks.append(("停完子进程也没了", not alive_pids(kids_before)))
        else:
            print("  [跳过] 没观察到子进程（可能起得太快），仅验父进程")
    ok = 0
    for name, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {name}")
        ok += 1 if passed else 0
    print(f"----- {ok}/{len(checks)} 通过 -----")
    return 0 if ok == len(checks) else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_run_selftest())
    print("这是库模块（只被 import）。要自测请加 --selftest。")
