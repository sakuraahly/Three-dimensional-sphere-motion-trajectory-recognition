"""任务进程管理：启动、读输出、逐行落盘、停整棵进程树。

为什么单独一个模块（2026-10-27 计划书 §4 P2）
--------------------------------------------
这段逻辑原来在 ``启动程序.py`` 的 ``Launcher`` 里（约 560 行）。分窗之后有
5 个窗口要看同一张任务表 —— 如果每窗各管一份子进程，本项目最贵的那个反模式
（"同一件事 N 份实现"）会再犯一次。抽出来之后：

- **零 Tk 依赖**：不 import tkinter；定时轮询用注入的 ``schedule`` 函数，
  因此可以在无显示器环境（CI）里被测试；
- **一个进程一份**：``App`` 持有唯一实例，各窗口只订阅 ``on_log`` / ``on_change``；
- 对外只有 ``start / stop / stop_all / tasks / running / on_log / on_change``。

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

__all__ = ["ROOT", "Task", "TaskRunner", "alive_pids", "child_pids", "POLL_MS"]

ROOT = Path(__file__).resolve().parent.parent
POLL_MS = 200


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
    on_done: Callable[["Task"], None] | None = None

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
        self._log_dir = log_dir if log_dir is not None else ROOT / "outputs" / "logs"
        self._poll_ms = poll_ms
        self._tasks: list[Task] = []
        self._tid = 0
        self._sel_tid: int | None = None
        self._log_queue: queue.Queue = queue.Queue()
        self._done_queue: queue.Queue = queue.Queue()
        self._stopping = False
        if root is not None:
            root.after(poll_ms, self._poll)

    # ---------- 查询 ----------
    def tasks(self) -> list[Task]:
        return list(self._tasks)

    def running(self) -> list[Task]:
        return [t for t in self._tasks if t.running]

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
        """
        kw: dict = dict(cwd=str(ROOT), stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True,
                        encoding="utf-8", errors="replace", bufsize=1)
        if os.name == "nt":
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kw["start_new_session"] = True
        return subprocess.Popen(cmd, **kw)

    def start(self, cmd: list[str], title: str,
              on_done: Callable[[Task], None] | None = None) -> Task:
        """起一个新任务并登记。**允许多个同时跑**（相机冲突由用户自己负责）。"""
        self._tid += 1
        proc = self._spawn(cmd)
        log_path = self._task_log_path(self._tid, title)
        task = Task(tid=self._tid, title=title, cmd=list(cmd), proc=proc,
                    log_path=log_path, on_done=on_done)
        self._tasks.append(task)
        self._sel_tid = task.tid
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
        """给每个任务一个日志文件：``outputs/logs/task<序号>_<时间>_<标题>.log``。

        **为什么按任务分开**：同一个 ``outputs/logs/`` 里已经有各程序自己的日志
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
                return self._confirm_dead(task, kids)
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
        return self._confirm_dead(task, kids)

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

    # ---------- 读输出与轮询 ----------
    def _reader(self, task: Task) -> None:
        proc = task.proc
        assert proc.stdout is not None
        for line in proc.stdout:
            body = line.rstrip()
            self._task_log(task, body)                      # 每行都落盘
            self._log_queue.put(f"[{task.tid}] {body}")
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
        self._log_queue.put(tail)
        self._done_queue.put(task.tid)

    def pump_once(self) -> None:
        """把队列里的日志与完成事件排空一次（可手动驱动，供无界面测试）。"""
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
            if task is not None and task.on_done is not None:
                try:
                    task.on_done(task)
                except Exception as exc:                    # noqa: BLE001
                    self._emit(f"[任务 {task.tid}] 完成回调失败：{exc}")
        if changed or self.running():
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
