"""跨进程任务注册表：把"谁在跑"写到一个大家都能读的地方。

为什么需要它（2026-11-18 用户报"很多任务在执行的时候没有进入任务管理器"）
------------------------------------------------------------------------------
新 GUI 的 ``TaskRunner`` 只登记**它自己 spawn 的子进程**。但有一类任务根本
不经过它：

- ``apps/data_manager.py`` 是**独立进程**（由门厅 ``tasks.start`` 起的），
  它内部的 zip 导出在线程里跑、轨迹导出同步跑在 UI 线程 —— 父进程的
  ``TaskRunner`` 对这两件事**一无所知**；
- 数据管理里开的 3D 查看器是 ``Popen`` 后 fire-and-forget，关窗后是真孤儿。

项目里没有 socket / 共享内存 / 任务注册表，``logs/tasks/<日期>/*.log`` 是
**只写不读**的被动日志，不能当权威源。于是补这一份极小的 append-only 账本：
每个任务写一行 ``start``、结束时再写一行 ``end``；控制台把"仍 start 且 PID
还活着"的记录并进"正在运行"列表。

设计约束（与 ``apps/_task_runner.py`` 一致）
--------------------------------------------
- **零 Tk 依赖**：不 import tkinter；可以在无显示器环境（CI）里测试；
- **纯 stdlib**：``json`` / ``ctypes`` / ``os`` / ``time`` / ``pathlib``；
- **坏行不炸面板**：单行 JSON 解析失败就跳过，一条脏记录不该让整张表空白；
- **崩溃自动消失**：不靠"一定要写 end"。任务所在进程崩了，PID 探活自然
  判死，"正在运行"里就没有它；因此查看器那种 fire-and-forget 只要写
  ``start`` 就够了。

路径为什么是 ``logs/tasks/_registry.jsonl``
--------------------------------------------
唯一来源是 ``sphere_tracker.paths.log_dir("tasks")``，它按
``_LOG_GRANULARITY["tasks"] = "%Y-%m-%d"`` 返回 ``logs/tasks/<日期>/``。
注册表必须**跨日期共用一份**（否则零点一过，昨天还在跑的任务就从面板上
消失），所以取它的 ``parent`` 再接文件名，并**不调用 mkdir 到日期子目录**
—— 写的是 ``logs/tasks/_registry.jsonl``，不会污染 ``logs/tasks/<日期>/``
里的任务日志。

``key`` 为什么必须含 PID
------------------------
``tid`` 每次进程启动都从 1 重开，纯 tid 会让两个不同进程的"任务 1"撞车。
``"<pid>-<tid>"`` 才是进程生命周期内唯一、且跨进程不重名。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

__all__ = ["REGISTRY_PATH", "register_start", "register_end",
           "external_running", "pid_alive"]

#: 按标记上溯（``sphere_tracker/`` 在哪，根就在哪），不数层数 —— 理由同
#: ``apps/_task_runner.py`` 的 ROOT。
ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "sphere_tracker").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sphere_tracker.paths import log_dir as _log_dir  # noqa: E402


def _default_registry_path() -> Path:
    """``logs/tasks/<日期>/`` 的父目录下放一份跨日期共用的账本。

    ``SPHERE_TASK_REGISTRY`` 环境变量优先：它会**被子进程继承**，所以测试
    harness 用一次环境变量注入就能同时覆盖自己与它 spawn 的 Python 进程
    （父进程里 monkeypatch 模块常量对子进程无效 —— 那是全新的解释器）。
    生产环境不设这个变量，路径仍唯一来源于 ``sphere_tracker.paths``。
    """
    override = os.environ.get("SPHERE_TASK_REGISTRY")
    if override:
        return Path(override)
    return _log_dir("tasks").parent / "_registry.jsonl"


#: 账本路径。进程内可 monkeypatch 这个常量（对已 import 的调用方即时生效）；
#: 跨进程用 ``SPHERE_TASK_REGISTRY`` 环境变量（子进程 import 时读到）。
REGISTRY_PATH: Path = _default_registry_path()


def _key(pid: int, tid: int) -> str:
    return f"{pid}-{tid}"


def _append(record: dict[str, Any]) -> None:
    """追加一行 JSON 并 flush；写不进去就静默跳过。

    为什么不让写失败冒泡：注册表是**额外登记**（用户要的是"看见任务"），
    不能因为磁盘满/权限不足把本来能跑的导出/任务一起搞挂 —— 那才是真的
    "任务没进管理器"。
    """
    try:
        REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with REGISTRY_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
    except (OSError, TypeError, ValueError):
        pass


def register_start(task: Any, *, source: str = "") -> None:
    """登记一个任务开始。``task`` 只需有 ``proc.pid`` / ``tid`` / ``title``。

    ``started`` 用 **wall clock**（``time.time()``），不是 ``Task.started``
    的 ``time.monotonic()``：账本跨进程读，别的进程拿到的 monotonic 没法
    直接算"已运行多久"；而 ``data_manager`` 的线程任务压根没有 ``Task``。
    """
    try:
        pid = int(task.proc.pid)
        tid = int(task.tid)
        title = str(task.title)
    except (AttributeError, TypeError, ValueError):
        return
    _append({
        "key": _key(pid, tid),
        "op": "start",
        "pid": pid,
        "tid": tid,
        "title": title,
        "owner": str(getattr(task, "owner", "") or ""),
        "started": time.time(),
        "ended": None,
        "exit_code": None,
        "stop_reason": None,
        "source": str(source or ""),
    })


def register_end(task: Any, *, source: str = "") -> None:
    """登记一个任务结束；**同一个 key 只保留第一条 end**。

    为什么必须去重：`stop()` 确认停掉时写一次 end，紧接着读线程发现进程
    退出又写一次 —— 同一个 key 两条 end，账本无限膨胀，而且"被停了两次"
    是假语义（读侧虽按 key 取最新、面板不受影响）。

    选"写前扫账本"而不是给 Task 加标志：无状态、跨进程也安全（两个进程
    先后写 end 同样只留一条）；end 次数远少于 start，O(n) 扫文件可接受。
    """
    try:
        pid = int(task.proc.pid)
        tid = int(task.tid)
        title = str(task.title)
    except (AttributeError, TypeError, ValueError):
        return
    key = _key(pid, tid)
    try:
        rows = REGISTRY_PATH.read_text(encoding="utf-8",
                                       errors="replace").splitlines()
    except (OSError, ValueError):
        rows = []
    for row in rows:
        try:
            old = json.loads(row)
        except (ValueError, TypeError):
            continue
        if (isinstance(old, dict) and old.get("key") == key
                and old.get("op") == "end"):
            return
    ended = getattr(task, "ended", None)
    reason = str(getattr(task, "stop_reason", "") or "")
    _append({
        "key": key,
        "op": "end",
        "pid": pid,
        "tid": tid,
        "title": title,
        "owner": str(getattr(task, "owner", "") or ""),
        # 必须与 start 行同语义：wall clock。Task.started 是 monotonic，
        # 两种时钟混在同一个字段里，日后按 started 排序/算耗时的代码会静默错。
        "started": time.time(),
        "ended": float(ended) if ended is not None else time.time(),
        "exit_code": getattr(task, "exit_code", None),
        "stop_reason": reason or None,
        "source": str(source or ""),
    })


def pid_alive(pid: int) -> bool:
    """这个 PID 还活着吗？**判不出来一律算活着**（宁可多显示，不可漏显示）。

    Windows 用 ``OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION)`` +
    ``GetExitCodeProcess``；259 是 ``STILL_ACTIVE``。必须显式声明
    ``restype = c_void_p`` —— 64 位 Python 默认把 HANDLE 当 32 位 int，
    句柄会被截断，查询必然失败，于是"活着的任务全被误判成死了"。

    非 Windows 用 ``os.kill(pid, 0)``：只探活、不发信号。

    已知限制：PID 被 OS 回收再分配给别的进程时，旧账本行会被误判成活着。
    注册表的记录量很小、进程重启很快，这个窗口在实际使用里可忽略；要彻底
    解决得给每行加启动时间并和进程创建时间比对，此处不做。
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes

            k = ctypes.windll.kernel32
            k.OpenProcess.restype = ctypes.c_void_p
            k.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int,
                                      ctypes.c_ulong]
            k.GetExitCodeProcess.restype = ctypes.c_int
            k.GetExitCodeProcess.argtypes = [ctypes.c_void_p,
                                             ctypes.POINTER(ctypes.c_ulong)]
            k.CloseHandle.argtypes = [ctypes.c_void_p]
            handle = k.OpenProcess(0x1000, False, pid)
            if not handle:
                return False
            try:
                code = ctypes.c_ulong()
                if k.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return code.value == 259          # STILL_ACTIVE
                return True                            # 查不到 → 算活着
            finally:
                k.CloseHandle(handle)
        except Exception:                              # noqa: BLE001
            return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except Exception:                                  # noqa: BLE001
        return True


def external_running() -> list[dict]:
    """当前仍在跑的外部任务记录（``op == "start"`` 且 PID 还活着）。

    顺序读所有行 → 按 ``key`` 取**最后一条** → 筛出仍 start 且 PID 活着。
    单行坏 JSON / 非 dict / 缺字段一律跳过，绝不因为一条脏记录让面板炸掉。
    """
    try:
        lines = REGISTRY_PATH.read_text(encoding="utf-8",
                                        errors="replace").splitlines()
    except (OSError, ValueError):
        return []
    latest: dict[str, dict] = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(rec, dict):
            continue
        key = rec.get("key")
        if isinstance(key, str) and key:
            latest[key] = rec
    out: list[dict] = []
    for rec in latest.values():
        if rec.get("op") != "start":
            continue
        try:
            pid = int(rec.get("pid", 0))
        except (TypeError, ValueError):
            continue
        if not pid_alive(pid):
            continue
        out.append(rec)
    return out
