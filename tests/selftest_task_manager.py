"""任务管理器自测：能起、能中断、**子进程树一起收干净**。

为什么需要它
------------
启动器原来只有一个 `self.proc` 句柄，于是：

1. **一次只能跑一个**程序（第二个被"已有程序在运行"挡掉）；
2. **完全没有中断手段** —— 程序卡住（相机被占、等键盘输入）时，
   用户只能关掉整个启动器窗口；
3. 而关窗口 `destroy()` **不会杀掉子进程**：相机仍被占着，
   下一次运行打不开相机，**而用户不知道为什么**。

第 3 条是最难发现的：用户以为"我关掉了"，实际资源没释放。
所以本测试的重点不是"能不能停"，而是**"停完以后进程真的没了"**。

## 判据的关键：子进程树

`Popen.terminate()` **只杀直接子进程**。我们的子进程是 python，
它自己还会起子进程 —— 父进程死了子进程活着，相机就还占着。
所以测试用一个"父进程 + 长命子进程"的结构，**专门验证子树被清干净**。

反向自证（改了就会红）
--------------------
======================================  ==========================================  ========
改动                                    注入方式                                    结果
======================================  ==========================================  ========
停任务不杀子树                          去掉 `_kill_leftover` 调用                   exit=1
`_child_pids` 永远返回空                整段换成 `return []`                        exit=1
不记退出码                              去掉 `task.exit_code = ...`                 exit=1
关窗口不收拾子进程                       `_on_close` 里把 `stop_all` 结果置 0        exit=1
刷新列表丢掉选中的行                    把 `if keep == t.tid:` 换成 `if False:`     exit=1
清理连在跑的一起删                       `self.tasks = []`                           exit=1
**强杀失败谎报成功**（第 8 节）          `stop_task` 改回"永远 `return True`"        exit=1
**不看 taskkill 返回码**（第 8 节）      去掉 `_force_kill_tree` 里的 `returncode` 判断  exit=1
======================================  ==========================================  ========

> 最后两条是 2026-10-14 补的。它们守的是一个**在沙箱里被误判成"环境问题"**的真缺陷：
> 那次 `taskkill /F` 因权限不足返回 1（stderr `ERROR: Access denied`），
> 而旧实现 `subprocess.run(..., capture_output=True)` **既不看返回码也不看 stderr** ——
> 失败没有出口，`stop_task` 照样报"停好了"。**"看着停了其实还在"又出现了，
> 只是这次躲在"环境限制"的借口后面。**

**一条写错的变异（记下来免得重犯）**

我原本还想证明"**多任务不许被退回一次一个**"。把 `_run` 里加回
"已有程序在运行就拦掉"的守卫……**注入生效了，但测试全绿**。
原因：那道守卫在 `_run` 里，而 `_run` 要真起相机；测试走的是 `start_task`，
**根本不经过那个函数**。也就是说这条变异**本来就不可测**，
不是判据太弱。换成"刷新丢掉选中"之后立刻就红了。

→ 教训：**"没红"要先分清是"判据弱"还是"注入没生效/打错了地方"**。
本测试的变异脚本因此加了一句"注入后源码没变就报警"。

用法::

    python tests/selftest_task_manager.py
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "apps").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import tkinter as tk
except ImportError:                                        # pragma: no cover
    tk = None

try:
    import psutil
except ImportError:                                        # pragma: no cover
    psutil = None

LAUNCHER = ROOT / "启动程序.py"
# 父进程 + 一个长命子进程：专门验证"子树"会不会被一起杀
PARENT_WITH_CHILD = (
    "import subprocess,sys,time;"
    "subprocess.Popen([sys.executable,'-c','import time;time.sleep(600)']);"
    "print('parent up', flush=True);"
    "time.sleep(600)"
)
SLEEPER = "import time;print('up', flush=True);time.sleep(600)"


def _load_launcher():
    """加载启动器模块。

    ⚠️ **必须先注册进 `sys.modules`**：里面有 `@dataclass`，而 dataclasses
    解析注解时要按 `cls.__module__` 去 `sys.modules` 找模块，
    直接 `exec_module` 而不注册会抛
    `AttributeError: 'NoneType' object has no attribute '__dict__'`。
    """
    sys.argv = ["启动程序.py"]
    spec = importlib.util.spec_from_file_location("launcher_tasks", LAUNCHER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["launcher_tasks"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    print("=" * 70)
    print("任务管理器自测（起 / 中断 / 子进程树 / 退出收尾）")
    print("=" * 70)

    if tk is None:
        print("  [跳过] 这台机器没有 tkinter。")
        print("\n判定结果：全部通过")
        return 0
    if psutil is None:
        print("  [跳过] 没有 psutil —— 无法验证「进程真的没了」，")
        print("         而「看着停了其实还在」正是本测试要守的东西，故不假装通过。")
        print("\n判定结果：全部通过")
        return 0

    mod = _load_launcher()
    app = mod.Launcher()
    app.update()
    results: list[tuple[str, bool]] = []

    def ck(label: str, good: bool, detail: str = "") -> None:
        results.append((label, good))
        print(f"  [{'通过' if good else '不达标'}] {label}"
              + (f"（{detail}）" if detail else ""))

    def pump(seconds: float = 0.5, target=None) -> None:
        """跑一会儿事件循环 —— 否则 `_poll` / `_refresh_tasks` 根本没执行。

        `target` 用来指定根窗口：第 7 节会把 `app` 销毁并置空，
        之后第 8 节另起一个 `Launcher`，那时不能再往 `app` 上 update。
        """
        win = target if target is not None else app
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            win.update()
            time.sleep(0.02)

    py = sys.executable

    try:
        # ---- 1. 起任务 ----
        print("----- 1. 起任务并登记 -----")
        t = app.start_task([py, "-c", PARENT_WITH_CHILD], "测试（父+子）")
        pump(1.0)
        kids = mod._child_pids(t.proc.pid)
        print(f"  父 PID={t.proc.pid} 子进程={kids}")
        ck("登记进任务表", any(x.tid == t.tid for x in app.tasks))
        ck("任务在运行", t.running)
        ck("界面上有这一行", str(t.tid) in app.task_tree.get_children())
        ck("确实起了子进程（这是后面「子树一起杀」的前提）", len(kids) >= 1)

        # ---- 2. 中断（优雅优先）----
        print("----- 2. 中断（先优雅，3 秒不退再强杀）-----")
        app.stop_task(t, graceful=True)
        pump(1.5)
        print(f"  停止原因={t.stop_reason!r} 退出码={t.exit_code} 耗时={t.elapsed():.1f}s")
        ck("任务已停止", not t.running)
        ck("记下了**停止原因**（用户能看出是被谁停的）", bool(t.stop_reason))
        ck("记下了退出码", t.exit_code is not None)

        # ---- 3. 子进程树 ----
        print("----- 3. 子进程树必须一起没了（否则相机仍被占着）-----")
        left = [p for p in kids if psutil.pid_exists(p)]
        print(f"  残留子进程={left}")
        ck("子进程全部结束", not left)

        # ---- 4. 强制结束 + **并发** ----
        print("----- 4. 强制结束，以及允许多个任务同时跑 -----")
        t2 = app.start_task([py, "-c", SLEEPER], "测试（强制）")
        pump(0.8)
        pid2 = t2.proc.pid
        ck("第二个任务在运行", t2.running)
        # ⚠️ 这条必须是"**在 t2 还跑着的时候**再起一个"。
        # 第一版写成"起完 t2 数一下在跑几个"，而那时 t1 已经停了 ——
        # 于是"退回一次一个"的变异照样能过（那道拦截只在**已有任务在跑**时生效）。
        t3 = app.start_task([py, "-c", SLEEPER], "测试（并发）")
        pump(0.6)
        both = t2.running and t3.running
        print(f"  t2 在跑={t2.running} t3 在跑={t3.running} 同时={both}")
        ck("**两个任务可以同时在跑**（旧版会拦掉第二个）", both)
        app.stop_task(t2, graceful=False)
        app.stop_task(t3, graceful=False)
        pump(1.2)
        ck("强制结束后不再运行", not t2.running)
        ck("进程真的没了（不是只剩个句柄）", not psutil.pid_exists(pid2))

        # ---- 5. 全部中断 ----
        print("----- 5. 全部中断 -----")
        for i in range(3):
            app.start_task([py, "-c", SLEEPER], f"批量 {i}")
        pump(0.8)
        started = len(app.running_tasks())
        stopped = app.stop_all(graceful=False)
        pump(1.5)
        print(f"  在跑 {started} 个；stop_all 报停 {stopped}；剩余 {len(app.running_tasks())}")
        ck("stop_all 把在跑的都停掉", started >= 3 and len(app.running_tasks()) == 0)

        # ---- 6. 清理记录 + **刷新不许丢掉选中** ----
        print("----- 6. 清理已结束记录；刷新列表要保住用户选中的那一行 -----")
        keep = app.start_task([py, "-c", SLEEPER], "选中保持测试")
        pump(0.6)
        app.task_tree.selection_set(str(keep.tid))
        app.update()
        selected_before = app.task_tree.selection()
        # 刷新几次（`_poll` 在有任务在跑时每 POLL_MS 就刷一次）
        app._refresh_tasks()
        app._refresh_tasks()
        pump(0.5)
        selected_after = app.task_tree.selection()
        print(f"  刷新前选中={selected_before} 刷新后={selected_after}")
        ck("刷新后仍然选中原来那一行（否则用户每次都被清空选择）",
           selected_after == (str(keep.tid),))
        app.stop_task(keep, graceful=False)
        pump(1.0)

        before = len(app.tasks)
        app._prune_tasks()
        print(f"  记录 {before} -> {len(app.tasks)}")
        ck("已结束的被清掉", len(app.tasks) == 0)

        # 在跑的任务不许被清掉
        live = app.start_task([py, "-c", SLEEPER], "清理保护测试")
        pump(0.6)
        app._prune_tasks()
        ck("**在跑的任务不会被清理掉**", any(x.tid == live.tid for x in app.tasks))
        app.stop_task(live, graceful=False)
        pump(1.0)
        app._prune_tasks()

        # ---- 7. 退出收尾（最难发现的那条）----
        print("----- 7. 关窗口必须把在跑的任务收干净 -----")
        t3 = app.start_task([py, "-c", PARENT_WITH_CHILD], "退出前测试")
        pump(1.0)
        kids3 = mod._child_pids(t3.proc.pid)
        # 退出会弹确认框；替掉它，直接确认
        mod.messagebox = type("_B", (), {
            "askyesno": staticmethod(lambda *a, **k: True),
            "showinfo": staticmethod(lambda *a, **k: None),
            "showwarning": staticmethod(lambda *a, **k: None),
        })()
        app._on_close()
        time.sleep(0.8)
        alive = [p for p in [t3.proc.pid, *kids3] if psutil.pid_exists(p)]
        print(f"  退出后残留={alive}（子进程 {kids3}）")
        ck("退出后父进程与子进程都没了（相机不会被一直占着）", not alive)
        app = None                                   # 已 destroy，后面不要再碰
    finally:
        if app is not None:
            try:
                app.stop_all(graceful=False)
                app.destroy()
            except Exception:                        # noqa: BLE001
                pass

    # ------------------------------------------------------------------ #
    # 8. 「杀失败」必须报出来 —— 假绿最容易藏在这条路上
    # ------------------------------------------------------------------ #
    #
    # ⚠️ 为什么必须单独测这一段（2026-10-14）：
    #
    # 沙箱里跑本测试时，第 4 节的强制结束**真的失败了**（`taskkill /F`
    # 返回 1、stderr 是 `ERROR: Access denied`），却报了 4 项不达标 ——
    # 那时我以为是环境问题就放过了。**真正的问题是：那份失败在代码里没有出口。**
    # `subprocess.run(["taskkill", ...], capture_output=True)` 既不抛异常
    # 也不看 `returncode`，于是 `stop_task` 照样 `return True`：
    # 界面显示"已停"、任务却还在跑、**相机还被占着** ——
    # 与 `LatestFrameCamera` 的静默重试是同一类缺陷。
    #
    # 所以这一段用**替身**把"强杀失败"造出来（不依赖真实权限），
    # 断言它变成可读的原因；并单独验证"进程还活着吗"这个判定本身。
    # 判据必须可失败：把 `stop_task` 改回"永远 return True"，
    # 下面第 2、3 条立刻红。
    #
    app2 = None
    t8 = None
    try:
        print("----- 8. 强杀失败必须报出来（替身注入「拒绝访问」）-----")
        app2 = mod.Launcher()
        app2.update()

        real_run = mod.subprocess.run
        calls: list[list[str]] = []

        def fake_run(cmd, *a, **k):
            calls.append(list(cmd) if isinstance(cmd, (list, tuple)) else [str(cmd)])

            class _R:                     # 复刻真实 taskkill 失败时的样子
                returncode = 1
                stdout = ""
                stderr = "ERROR: Access denied"
            return _R()

        mod.subprocess.run = fake_run
        try:
            t8 = app2.start_task([py, "-c", PARENT_WITH_CHILD], "测试（杀失败）")
            pump(1.0, app2)
            kids8 = mod._child_pids(t8.proc.pid)
            stopped = app2.stop_task(t8, graceful=False)
        finally:
            mod.subprocess.run = real_run          # 立刻还原：后面清理还要用它

        print(f"  stop_task 返回={stopped}  kill_note={t8.kill_note!r}")
        print(f"  注入期间真的调了 taskkill：{calls[:1]}")
        ck("**强杀失败时不再谎报成功**（返回值必须是 False）", stopped is False)
        ck("失败原因被记下来且可读（含返回码与 stderr）",
           "Access denied" in t8.kill_note and "1" in t8.kill_note)
        ck("**真的调了 taskkill**（否则上面两条是空转）",
           any(str(c).lower().endswith("taskkill") or "taskkill" in str(c).lower()
               for c in calls))
        ck("失败后进程确实还活着（这就是要报出来的那个事实）",
           psutil.pid_exists(t8.proc.pid))

        # 判定本身：活着的必须查得出来，不存在的必须不误报
        bogus = 999_999_999
        ck("_alive_pids 认得出活进程", mod._alive_pids([t8.proc.pid]) == [t8.proc.pid])
        ck("_alive_pids 不会把不存在的 pid 算成活着", mod._alive_pids([bogus]) == [])
    finally:
        if t8 is not None:
            # 先杀子树、再杀父进程 —— 反过来的话 `_child_pids` 就查不到了，
            # 长命子进程会一直留着（正是本测试要防的那种"看着停了其实还在"）。
            for p in mod._child_pids(t8.proc.pid):
                try:
                    psutil.Process(p).kill()
                except Exception:                     # noqa: BLE001
                    pass
            if t8.running:
                t8.proc.kill()                        # 这一段本来就没杀掉它，收尾
        if app2 is not None:
            try:
                app2.destroy()
            except Exception:                         # noqa: BLE001
                pass

    # ------------------------------------------------------------------ #
    # 9. 统一结束回调 / 子进程 stdin（2026-10-31 增，测的是 apps/_task_runner.py）
    # ------------------------------------------------------------------ #
    #
    # 为什么单独一段测**新**的任务运行器：上面 1–8 节测的是旧启动器
    # （``启动程序.py``，它自带一套 ``start_task``/``stop_task``）。新界面
    # 用的是 ``apps/_task_runner.py``，而"按钮点完就再无下文"正是它缺的东西：
    # 每个按钮各自传 ``on_done``，漏传一个（拍标定图 / 焦距估计 / 实时识别
    # 全都漏了）就永远停在"已开始…"；**失败只写日志，状态条一个字都不说**。
    #
    # 反向自证：把 ``pump_once`` 里的 ``self._notify_finished(task)`` 删掉，
    # 下面第 1、3、4 条立刻红；把 ``_spawn`` 的 ``stdin=subprocess.DEVNULL``
    # 去掉，第 7 条立刻红。
    print("----- 9. 统一结束回调与 stdin（apps/_task_runner.py）-----")
    import subprocess
    import tempfile

    import apps._task_runner as tr

    with tempfile.TemporaryDirectory(prefix="task_runner_") as td:
        runner = tr.TaskRunner(root=None, on_log=lambda _s: None,
                               log_dir=Path(td))
        finished: list = []
        runner.add_finish_listener(finished.append)

        def pump_until(tasks, seconds: float = 30.0) -> None:
            end = time.monotonic() + seconds
            while any(t.running for t in tasks) and time.monotonic() < end:
                runner.pump_once()
                time.sleep(0.05)
            runner.pump_once()

        # 归属是起完之后**填在任务上**的（窗口的 ``_start`` 就是这么做的）：
        # ``start()`` 的签名必须保持 ``(cmd, title, on_done)`` —— 走查工具与
        # gui_budget 都用老签名的桩替换它，加关键字参数会让它们当场 TypeError。
        t_ok = runner.start([py, "-c", "print('half');print('done-ok')"],
                            "自测-成功")
        t_bad = runner.start(
            [py, "-c", "import sys;print('bad-line');sys.exit(3)"],
            "自测-失败")
        t_ok.owner = "mono"
        t_bad.owner = "mono"
        pump_until([t_ok, t_bad])
        got = {t.tid for t in finished}
        ck("成功与失败的任务**都**调了结束回调（各一次）",
           got == {t_ok.tid, t_bad.tid} and len(finished) == 2)
        ck("成功任务：ok=True、返回码 0", t_ok.ok and t_ok.exit_code == 0)
        ck("失败任务：ok=False，且回调照样被调（失败有出口）",
           (not t_bad.ok) and t_bad.exit_code == 3 and t_bad in finished)
        ck("回调带得回**日志尾**（失败原因不用翻日志才看得到）",
           any("bad-line" in line for line in t_bad.tail),
           )
        ck("回调带得回归属（owner），别的窗口才不会被误伤",
           t_ok.owner == "mono" and t_bad.owner == "mono")
        ck("失败任务的日志文件真的落盘了", t_bad.log_path is not None
           and t_bad.log_path.exists())

        # 退订之后不许再回调（窗口关了还往已销毁的控件上写就是 TclError）
        heard: list = []
        cb = heard.append
        runner.add_finish_listener(cb)
        runner.remove_finish_listener(cb)
        t_gone = runner.start([py, "-c", "print('bye')"], "自测-退订")
        pump_until([t_gone])
        ck("remove_finish_listener 之后不再回调", heard == [])

        # ---- stdin：**不许**继承控制台 ----
        # 为什么这条重要：子进程里任何一句 input()（``stereo/pair_source.py``
        # 在没有 tty 的路径上就有一处）会等一个永远不会有人敲的回车 ——
        # 任务永不结束、状态条永远"已开始…"、相机还被它占着。
        seen_kw: dict = {}
        real_popen = tr.subprocess.Popen

        class _SpyPopen(real_popen):                # type: ignore[misc,valid-type]
            def __init__(self, cmd, **kw):
                seen_kw.update(kw)
                super().__init__(cmd, **kw)

        tr.subprocess.Popen = _SpyPopen
        try:
            t_in = runner.start([py, "-c", "print('x')"], "自测-stdin")
        finally:
            tr.subprocess.Popen = real_popen
        pump_until([t_in])
        ck("起子进程时 stdin 被重定向（不继承控制台）",
           seen_kw.get("stdin") == subprocess.DEVNULL,
           f"实际 stdin={seen_kw.get('stdin')!r}")

        t_read = runner.start(
            [py, "-c", "import sys;d=sys.stdin.read();print('readlen',len(d),flush=True)"],
            "自测-读stdin")
        pump_until([t_read], seconds=15.0)
        ck("子进程读 stdin 立刻拿到 EOF（不会把界面吊死）",
           (not t_read.running) and t_read.exit_code == 0
           and any("readlen 0" in line for line in t_read.tail),
           f"返回码 {t_read.exit_code}，尾巴 {t_read.tail}")

    ok = all(v for _n, v in results)
    bad = [n for n, v in results if not v]
    if bad:
        print(f"\n不达标 {len(bad)} 项：")
        for n in bad:
            print(f"  · {n}")
    print("\n判定结果：" + ("全部通过" if ok else "有不达标项"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
