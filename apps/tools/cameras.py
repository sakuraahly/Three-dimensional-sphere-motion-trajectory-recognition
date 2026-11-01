"""摄像头设备管理：列出设备 / 目视确认角色 / 固定身份 / 标"无关" / 检查漂移。

**索引会漂移，身份不会。** `cv2.VideoCapture(0)` 的 0 由系统枚举顺序决定，
换一次 USB 口就可能从"外接相机"变成"笔记本内置"——启动器里写死 `--left 0`
就会悄悄用错设备、标定白做。这个工具把"哪台是左、哪台是右"记成**设备身份**
（Windows 上取 PnP InstanceId / USB 端口），之后用**角色名**启动即可。

六个子命令：

    python apps/tools/cameras.py list                # 列设备（不开相机，任何机器都能跑）
    python apps/tools/cameras.py list --probe        # 顺带真开一下，报实际分辨率/是否全黑
    python apps/tools/cameras.py identify            # 逐个索引显示画面，让你确认哪个是左/右/无关
    python apps/tools/cameras.py pin left port:Port_#0001.Hub_#0003
    python apps/tools/cameras.py ignore 2            # 把某台标成「无关」（自动挑相机时跳过它）
    python apps/tools/cameras.py unignore --all      # 取消「无关」标记
    python apps/tools/cameras.py check               # 重插相机后比对：有没有漂移/被别的设备占用

`identify` 的按键：窗口里

    l = 记为左相机      r = 记为右相机      n = 标为「无关」（这台不要用）
    m = 记为单目主相机   s = 跳过这台（不改配置）   q = 退出

「左 / 右 / 无关」三者是**互斥**的：把一台相机改标成别的，旧标记会自动摘掉
（一台相机不能既是左相机又是「无关」，否则自动挑相机时两边打架）。
结果写入 `config/cameras.json`（本机相关，不入库）。
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sphere_tracker import camera_registry as cr  # noqa: E402
from sphere_tracker import video_profiles  # noqa: E402
from apps._window import WindowManager  # noqa: E402

from sphere_tracker.sources import (  # noqa: E402
    CameraSource,
    quiet_repeated_grab_warnings,
)


# --------------------------------------------------------------------------- #
def _first_line(exc: BaseException) -> str:
    return (str(exc).splitlines() or [type(exc).__name__])[0]


def parse_indices(text: str | None, devices: list[cr.DeviceEntry],
                  max_index: int) -> list[int]:
    """`--indices 0,1,2` -> [0,1,2]；不给就优先用枚举到的设备，再补 0..max。"""
    if text:
        out = []
        for piece in text.split(","):
            piece = piece.strip()
            if piece.isdigit():
                out.append(int(piece))
        return out
    if devices:
        return sorted({d.order for d in devices})
    return list(range(max_index + 1))


def positive_seconds(text: str) -> float:
    """argparse 的 `type=`：接受有限正数，其余一律报错。

    为什么不能让 argparse 直接 `type=float`：`float("nan")` / `float("inf")`
    都能过，而它们进到预算计算里会得到"永远等"或"立刻超时"两种极端 ——
    前者看着就是"卡死"（正是本次要修的病症），后者把好相机判死。
    0 / 负数同理：`budget = timeout + 10` 会退化成极小值。
    这里**报错退出**而不是静默回退默认值：用户明确传了数，就该知道自己传错了。
    """
    try:
        value = float(text)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"不是数字：{text!r}") from None
    if not (0.0 < value < 1e6):
        raise argparse.ArgumentTypeError(
            f"必须是 (0, 1000000) 秒内的有限正数，收到 {text!r}")
    return value


# --------------------------------------------------------------------------- #
# list
# --------------------------------------------------------------------------- #
def cmd_list(args) -> int:
    devices = cr.list_devices()
    max_index = args.max_index

    probes: dict[int, cr.ProbeResult] = {}
    if args.probe:
        indices = sorted({d.order for d in devices} | set(range(max_index + 1)))
        for i in indices:
            # 用带回退的版本：MSMF 在本机**间歇性**给不出帧（实测一个下午里
            # 好→坏→坏→又好），而 DSHOW 一直正常。没有回退时这里会等 33 秒
            # 然后报"一帧都拿不到"，用户会以为相机坏了。理由详见
            # `camera_registry.probe_index_with_fallback()` 的文档串。
            probes[i] = cr.probe_index_with_fallback(i)

    if args.json:
        payload = {
            "platform": cr.current_platform(),
            "devices": [asdict(d) for d in devices],
            "probes": {str(k): asdict(v) for k, v in probes.items()},
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(f"平台：{cr.current_platform()}")
    if cr.current_platform() != "windows":
        print("（非 Windows：不做身份识别，功能退化为按索引使用）")
    print("检测到的相机（索引 = 枚举位次，是启发式；请用 identify 目视确认）：")
    print(cr.format_device_table(devices, probes, settings=None, reg=cr.load_registry()))

    if args.probe:
        extra = [i for i in probes if i not in {d.order for d in devices}]
        for i in sorted(extra):
            r = probes[i]
            tail = (f"实测 {r.width}x{r.height}@{r.fps:.1f}" if r.ok
                    else f"实测失败({r.error})")
            print(f"  #{i}  (不在设备枚举里)  {tail}")

    if not devices and not args.probe:
        print("\n提示：加 --probe 可以不管枚举、直接逐个索引试开一次。")

    # 末尾固定放一句"更快的路"：`identify` 要**真的开相机**（慢相机 24 秒/台），
    # 而大多数人只是想"把左/右绑到这两台名字上"—— 那条路 `pin` 秒级完成。
    # 放在末尾是因为用户看完设备表，正好在决定"下一步做什么"。
    print()
    for line in _pin_hint_lines(
            devices, "只想把左/右绑到某台设备、不打算逐台确认画面？"
                     "按名字绑身份即可（秒级，不开相机、不碰编号）："):
        print(line)
    return 0


# --------------------------------------------------------------------------- #
# identify
# --------------------------------------------------------------------------- #
# 「无关」不是一个"角色"，但它是用户在识别画面时能做的第三种选择，
# 所以和其它三个放在同一张键位表里，由 cmd_identify 分流处理。
IGNORE_TOKEN = "ignore"
_ASSIGN_KEYS = {ord("l"): "left", ord("r"): "right",
                ord("m"): "main", ord("n"): IGNORE_TOKEN}
_KEY_HINT = "l=左相机  r=右相机  n=无关(这台不要用)  m=单目主相机  s=跳过  q=退出"
# 每个索引的预览**实际用上了哪个后端**（`_preview_and_assign` 填）。
# 为什么值得记：标定/检测要用同一个后端才稳，而这个后端不一定等于偏好后端 ——
# 打不开时会回退。不告诉用户，他标定时就得再踩一次同样的坑。
_backend_used: dict[int, str] = {}


def _open_preview(index: int, backend: int, *, enumerated: bool = False,
                  timeout_s: float | None = None):
    """按**给定的那一个后端**打开预览；返回 (src, 后端名)，打不开返回 (None, None)。

    `timeout_s=None` → 用模块默认 `PREVIEW_FIRST_FRAME_S`（首帧），整台上限
    派生为 `timeout_s + 10`（保持现 30→40 的关系）；给了值就按值走（P6：
    默认值**不变**，只有显式传入才会改）。

    ## 为什么**绝不**在这里换后端（2026-10-18 第三次修正，用户实测反馈）

    教训链（两次都是我改坏的）：

    1. f8bb30 加了"按索引换后端" → 用户看到 **aoni 被标成 Integrated**
       （DSHOW 的编号与 PnP 错位）；
    2. 33b7164 把回退整个删掉 → MSMF 打不开的那台 **从界面消失**
       （"只能检测到一个"）；
    3. 我又改成"回退只为显示、不许绑定" → 用户看到 **同一台 aoni left 出现两次**
       而 right 不见 —— 因为回退让 DSHOW #2 与 MSMF #0 **可能是同一台**。

    **结论：按索引回退在"显示"这一层也站不住** —— 它会**重复**同一台物理相机。
    PnP 与各后端的编号是**互不相容的三套命名空间**，任何"混着用"的方案都会
    要么贴错标签、要么重复、要么遗漏。

    所以这里只做一件事：**在同一个后端内**开一个编号。
    想确认"另一台在别的后端下能不能出图"，用 cameras.py probe（它逐后端
    分别列出，本来就**不下结论说"哪台是哪台"**），或显式 --backend dshow。

    ## 2026-10-29：开之前先说"要等多久"，开完说"用了多久"

    用户报"相机管理卡死"，实测是**每台最坏等 30 秒**、6 台连着 = 3 分钟一个
    窗口都不出现，而这段时间**一个字都不打** —— 只能被理解成卡死。
    所以这里必须把等待**显式说出来**：开头一行、结果一行（带耗时）、
    等待期间每 5 秒一行（`_WaitTicker`）。

    失败时还会区分 `enumerated`：**"枚举不到这个编号"** 和 **"在清单里却开不出图"**
    是两件不同的事（前者查线/查编号漂移，后者查占用/拔插/换后端），
    而旧文案对两者都只说"打不开"。
    """
    from sphere_tracker.sources import backend_name

    used_name = backend_name(backend)
    # `timeout_s` 为 None 时用模块默认（现值不变）；给了值就按值派生整台上限。
    first_timeout = timeout_s if timeout_s else PREVIEW_FIRST_FRAME_S
    budget = (timeout_s + 10.0) if timeout_s else PREVIEW_BUDGET_S
    # 为什么把"整台上限"也写出来：首帧 30 秒只是**一个分辨率**的上限，
    # 出不了图时会再降分辨率试一次，整台最多 `budget` 秒。
    # 只说 30 秒，用户在第 35 秒会以为"说好的 30 秒呢、是不是又卡死了"。
    print(f"  正在打开 #{index}（{used_name}）…最多等 "
          f"{first_timeout:.0f} 秒（首帧；整台上限 "
          f"{budget:.0f} 秒）", flush=True)
    tick = _WaitTicker(index, used_name, budget_s=budget)
    tick.start()
    t0 = time.perf_counter()
    try:
        src = _try_open(index, backend, first_frame_timeout=first_timeout,
                        budget=budget)
    except RuntimeError as exc:
        took = time.perf_counter() - t0
        tick.stop()
        # 为什么把"枚举不到"和"在线却开不出图"分成两句话：
        # 原来的 `✘ 打不开` 两种情况同一个词，而**处置完全不同** ——
        # 前者是没插/编号漂（去 list、pin），后者是设备在但没帧（拔插、查占用、
        # 换后端）。本机实测 #0 就是后者（在线、MSMF 30.8 s 打不开），
        # 用户当时把这种"慢失败"一律读成了"程序卡死/相机坏了"。
        why = ("在线但开不出图" if enumerated else "设备清单里没有这台")
        print(f"  ✘ #{index}（{used_name}）{why}，用了 {took:.1f} s："
              f"{_first_line(exc)}", flush=True)
        return None, None
    finally:
        tick.stop()
    took = time.perf_counter() - t0
    print(f"  ✔ #{index}（{used_name}）出图了，用了 {took:.1f} s", flush=True)
    _backend_used[index] = used_name
    return src, used_name


def _try_open(index: int, api: int, *, first_frame_timeout: float | None = None,
              budget: float | None = None):
    """按给定后端开一路预览；失败**抛 RuntimeError**，由调用方决定是否回退。

    optimize=False + 短预算（2026-10-18，用户反馈"卡顿明显"）：
    设备识别只需要"把画面抓出来让人按键"，**不需要**挑最优分辨率/帧率。
    之前用默认参数时：optimize=True 会跑模式搜索（冷缓存最坏 10~40 s），
    search_budget_s 默认 60 s —— 遇到打不开的相机会干等一分钟。两者叠加
    就是用户看到的"卡顿"。

    `first_frame_timeout` / `budget` 为 None 时用模块默认
    （`PREVIEW_FIRST_FRAME_S` / `PREVIEW_BUDGET_S`，**现值不变**）。
    """
    # ⚠️ **请求 1920x1080（兼容档）而不是 640x480**。
    #
    # 2026-10-18 实测这台 aoni：MSMF 下请求 640x480 / 1280x720 会"打开成功但
    # 一帧都不给"（10.6 s 后判死），而 1920x1080 能出帧（约 23 s；另一台约 2 s）。
    # 2026-10-24 复测：两台外接相机**都无视 1920/1280 请求、恒给 2560x1440**，
    # 所以这里请求 1920 只是兼容旧机型；实际分辨率以 `src.info` 为准，不假设。
    return CameraSource(index=index, width=1920, height=1080, backend=api,
                        optimize=False,
                        search_budget_s=(budget if budget is not None
                                         else PREVIEW_BUDGET_S),
                        first_frame_timeout_s=(first_frame_timeout
                                               if first_frame_timeout is not None
                                               else PREVIEW_FIRST_FRAME_S))


def _open_preview_interruptible(index: int, backend: int, *, enumerated: bool = False,
                                timeout_s: float | None = None, key_poll=None,
                                skipped_out: list[int] | None = None):
    """在后台线程里开预览；主线程轮询键盘，Esc/s 立即放弃（不阻塞等待）。

    返回 `(src, 后端名)`；被跳过 / 打不开 → `(None, None)`。

    `timeout_s=None` → 用模块默认（`PREVIEW_FIRST_FRAME_S`，派生
    budget = timeout+10）。`key_poll=None` → 用 `_console_key`。

    `skipped_out`：传一个 list 进来，**用户主动跳过**的索引会写进去。
    返回值把"跳过"和"打不开"合并成 `(None, None)`（形状不变，第 21 节守着），
    所以调用方要区分两者就只能靠这个出口 —— 它决定收尾文案该不该说"打不开"。

    ## 为什么必须另起线程（P4，2026-10-31）

    `CameraSource(...)` 是**在构造函数里开相机 + 等首帧**的，整段发生在主线程、
    没有任何钩子能插进去（最多能等 `PREVIEW_FIRST_FRAME_S`=30 秒）。旧代码里
    按键循环在 `_preview_and_assign` 的**首帧之后**才跑，所以这 30 秒内按 Esc
    完全无效 —— 用户看到的就是"卡死"。这里把"开相机"搬进后台线程，
    主线程只做"轮询键盘 + 等一小步"，于是等待期也能按键放弃。

    ## 不留悬空相机（资源安全）

    若用户跳过，**后台线程可能仍在尝试打开**。一旦它最终把 `src` 开出来，
    由 worker 的 `finally` 负责 `release()`（见 `box["skipped"]` 分支），
    所以**不会留一台没人管的相机**。若线程永不返回，它只是 `daemon` 线程，
    不影响进程退出（放弃即返回，主线程不再等它）。

    ## 无控制台时退化为"不可中断"（P5）

    `key_poll` 恒返回 None（如 `_console_key` 在没有真控制台时）→ 本函数就只是
    "在后台开、主线程等它结束"，行为等价于旧的阻塞版，但**绝不崩**。
    """
    key_poll = key_poll or _console_key
    box = {"src": None, "bname": None, "done": False, "skipped": False}
    lock = threading.Lock()

    def _store_or_drop(src, bname) -> None:
        """收尾：已被跳过就**就地释放**（不留悬空相机），否则把结果交给主线程。

        为什么读 `skipped` 与写 `src` 必须在**同一把锁**里：主线程可能在 worker
        赋值的**同一瞬间**按下 Esc。旧写法（先置 `done`、再查 `skipped`）会留下
        一个窗口 —— 主线程已判定"跳过"并返回 `(None, None)`，worker 稍后才把
        相机开出来，于是**没人 release**（独立 Critic 实测到过，窗口长度 = 一次
        `key_poll()`）。现在两侧共用这一把锁：任一侧先到都能正确收尾，且
        `box["src"]` 只会被置空一次，不会重复 `release()`。
        """
        with lock:
            if box["skipped"]:
                drop, box["src"] = src, None
            else:
                box["src"], box["bname"] = src, bname
                drop = None
            box["done"] = True
        if drop is not None:
            try:
                drop.release()
            except Exception:                            # noqa: BLE001
                pass

    def _worker() -> None:
        src = bname = None
        try:
            src, bname = _open_preview(index, backend, enumerated=enumerated,
                                       timeout_s=timeout_s)
        except BaseException as exc:                     # noqa: BLE001
            # 后台线程里抛异常若不接住，`box["done"]` 会永远为 False、主线程
            # 空转 —— 比"打印一行"糟得多。原因照旧打出来，不静默。
            print(f"  ✘ #{index} 后台打开时异常：{type(exc).__name__}: "
                  f"{_first_line(exc)}", flush=True)
        _store_or_drop(src, bname)

    th = threading.Thread(target=_worker, daemon=True)
    th.start()
    while not box["done"]:
        k = key_poll()
        if k in (27, ord("s"), ord("S")):
            with lock:
                box["skipped"] = True
                drop, box["src"] = box["src"], None
            if drop is not None:
                try:
                    drop.release()
                except Exception:                        # noqa: BLE001
                    pass
            if skipped_out is not None and index not in skipped_out:
                skipped_out.append(index)
            print(f"  ⏭ 已跳过 #{index}（后台仍在尝试打开；一旦打开会自动释放）",
                  flush=True)
            return None, None
        th.join(timeout=0.05)
    return box["src"], box["bname"]


# 预览里"等第一帧"的上限（秒）。**慢相机实测要 ~24 秒**才给第一帧
#（aoni #0 + MSMF）；3 秒会把它误判成"一帧都拿不到"，于是它在界面上彻底消失。
# 预览窗口的**显示宽度**（超过就等比降采样）。识别只需看清是哪台，
# 而 1920x1080 每帧约 6 MB，直接 imshow 在核显上会明显掉帧。
PREVIEW_DISPLAY_W = 960

PREVIEW_FIRST_FRAME_S = 30.0

# 单台预览的**总**预算：要能容纳上面那次首帧等待 + 一次分辨率降级。
PREVIEW_BUDGET_S = 40.0

# 等相机出图时，每隔几秒说一句"还在等"。
#
# 为什么是 5 秒：用户点完「换相机…」之后，如果 3 分钟里**一个字都没有**，
# 他只能判断成"卡死"并去关窗口 —— 而这 3 分钟其实一直在正常等待
#（每台最多 `PREVIEW_FIRST_FRAME_S` 秒，慢相机实测 ~24 秒/台）。
# 5 秒一行既够密（看得出程序还活着），又不会把屏幕刷满。
_WAIT_TICK_S = 5.0


class _WaitTicker:
    """阻塞等待期间定时打印"已等 X 秒"的守护线程。

    ## 为什么要另起一条线程说话

    `CameraSource(...)` 是**在构造函数里开相机**的：从构造到返回，主线程完全
    被占住，没有任何回调/钩子能插进去（2026-10-29 查过 `sphere_tracker/sources.py`
    —— 它只有 `_log()` 打自己的日志，而"等第一帧"那个循环里**一句话都不打**）。
    用户等 24 秒看不到输出，就认为程序死了。

    所以这里只做一件事：**读时钟、打印**。它不碰 `CameraSource`、不碰 cv2、
    不改任何状态，因此和正在开相机的那条线程**没有共享状态**，不需要锁。
    `daemon=True` 保证用户 Ctrl+C 时它不会拖住进程。

    为什么不用 `\\r` 原地覆盖：`CameraSource` 自己也会往 stdout 打日志
    （`试 MSMF @ 1920x1080…`、`✔ … 打开成功`），两股输出抢同一行会糊成一片。
    """

    def __init__(self, index: int, backend_name: str,
                 tick_s: float = _WAIT_TICK_S,
                 budget_s: float = PREVIEW_BUDGET_S) -> None:
        self._index, self._backend = index, backend_name
        self._tick_s, self._budget_s = tick_s, budget_s
        self._t0 = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._t0 = time.perf_counter()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.wait(self._tick_s):
            el = time.perf_counter() - self._t0
            print(f"    … 还在等 #{self._index}（{self._backend}）出第一帧："
                  f"已等 {el:.0f} 秒（整台上限 {self._budget_s:.0f} 秒）"
                  f"（慢相机实测 ~24 秒才有画面，别关窗口）", flush=True)


def _console_keys_available() -> bool:
    """当前进程能否读控制台按键（Windows msvcrt）。无控制台 → False。

    ## 为什么必须**兜底成不可用**（P5）

    GUI 用 `tasks.start([...cameras.py...])` 起的是**子进程**，它的 stdin
    不一定是真控制台（可能被重定向 / 根本没有）。此时 `msvcrt.kbhit()` 会抛
    （`OSError` / `ValueError`）或恒假。**这条判据只用于决定是否提示"可按 Esc"**，
    绝不能因为读不到键盘而崩掉整个 identify。

    所以整体 try/except `(ImportError, OSError, ValueError)` → False，**绝不抛**。
    """
    try:
        import msvcrt
        msvcrt.kbhit()          # 真探一次：有控制台才会正常返回 bool
        return True
    except (ImportError, OSError, ValueError):
        return False


def _console_key() -> int | None:
    """非阻塞读一个按键（返回码位；无键 / 不可用 → None）。**绝不抛**。

    `msvcrt.getwch()` 返回 str（普通键是单个字符，特殊键是 `'\\x00'` / `'\\xe0'`
    前缀 + 一个字节），统一用 `ord(ch)` 取码位。扩展键原样返回其前缀码位即可，
    调用方只关心 Esc(27) / s / S —— 不需要为特殊键做特殊解析。
    """
    try:
        import msvcrt
        if not msvcrt.kbhit():
            return None
        ch = msvcrt.getwch()
        return ord(ch) if ch else None
    except (ImportError, OSError, ValueError):
        return None


def _preview_label(index: int, entry: cr.DeviceEntry) -> str:
    """预览窗口上的标签。**必须写明"推测"** —— 这不是事实，是我们猜的。

    ## 为什么必须标"推测"（2026-10-18，用户第二次纠正）

    用户问："你的相机管理的视频画面的内容是不是根本不是通过相机的名字来设置的?
    为什么我的命名为 right 的相机却在画面上是 left?"

    **他说得对，而且这条比上一轮更根本。**

    `cv2.VideoCapture(N, backend)` **只能按编号打开** —— 名字根本不参与。
    窗口上那行 `#N <名字>` 是"编号 N 对应设备列表里第 N 台"这个**假设**的产物。

    我的证据缺口在这里：

    | 我验证过的 | 结论 |
    | --- | --- |
    | aoni vs Integrated（MSMF `#2` 黑 = PnP `#2` 内置） | 成立 |
    | **两台一模一样的 aoni 谁在 `#0`、谁在 `#1`** | **从未验证** |

    拿"`#2` 是内置"外推到"整个顺序都对"，是**用一条弱证据支撑了一个强结论**。
    两台 aoni 同 VID/PID、同分辨率、同画面特征 —— **在软件层面不可区分**，
    所以这个顺序**原理上无法由 OpenCV 证明**。

    于是：这条标签**永远只能是推测**。把推测写成事实，用户就会按错的标签绑定角色
    （他说"命名为 right 的相机在画面上是 left"就是这么来的）。

    正确做法：**标签照实说它是推测**，绑定走 `cameras.py pin <role> name:"<完整名字>"`
    —— 那条路径按**用户赋予的名字**解析到身份，**完全不开相机、也不碰编号**。
    """
    return f"#{index}  推测:{entry.name or '(无名字)'}  端口 {entry.port or '-'}"


def _preview_and_assign(index: int, entry: cr.DeviceEntry, status: str = "",
                        *, backend: int, first_frame_timeout: float | None = None,
                        key_poll=None,
                        failed_out: list[int] | None = None) -> str | None:
    """开一台相机显示实时画面，返回用户按下的角色（None=跳过，''=退出）。

    `failed_out`：传一个 list 进来，**"这个后端打不开这台"**的索引会写进去。
    它把"**用户主动跳过**（Esc/s）"与"**后端开不出来**"分开 —— 两者都返回
    `None`，但收尾时说的是两件事：主动跳过不该被报成"没能打开、去换个后端试试"。
    （独立 Critic 指出的误导：用户按 Esc 跳过 #2 后，收尾却提示去 `probe` 查后端。）

    ⚠️ `_open_preview` **必须**返回 `(src, 后端名)` 两元组 —— 两个分支形状一致。
    这里崩过一次（2026-10-15）：早退分支写成 `return None`，解包时抛
    `TypeError: cannot unpack non-iterable NoneType object`，整个 identify 崩掉。
    判据放在 `tests/selftest_camera_registry.py` 第 14 节（**穷举返回路径**验形状）。

    ⚠️ `backend` 是**整个 identify 会话共用的**，不允许按索引换（见 `_open_preview`
    里的长注释与那张实测表）。`keyword-only` 是为了让每个调用点都必须显式交代它。
    """
    label = _preview_label(index, entry)
    skipped: list[int] = []
    try:
        # `enumerated` 决定失败文案：编号在 PnP 清单里 = "在线却开不出图"，
        # 不在清单里 = "枚举不到这个编号"。没有名字的一律按"没枚举到"算
        # （`cmd_identify` 对查不到的索引塞的是一个空白 DeviceEntry）。
        src, backend_used = _open_preview_interruptible(
            index, backend, enumerated=bool(entry.name),
            timeout_s=first_frame_timeout, key_poll=key_poll,
            skipped_out=skipped)
    except RuntimeError as exc:
        print(f"  #{index} 打不开：{_first_line(exc)}")
        if failed_out is not None and index not in failed_out:
            failed_out.append(index)
        return None
    if src is None:
        # 这个后端出不了图 -> 跳过这台（**不崩、也不换后端**）。
        # ⚠️ 只有**不是**用户主动跳过时才算"打不开"：两者返回值都是 `None`，
        # 靠 `skipped` 区分（`_open_preview_interruptible` 的 `skipped_out`）。
        if not skipped and failed_out is not None and index not in failed_out:
            failed_out.append(index)
        return None
    _backend_used[index] = backend_used

    win = "相机识别（l=左 r=右 n=无关 m=主 s=跳过 q=退出 / f 全屏）"
    wins = WindowManager()
    # 这窗口画着大字提示与状态，**不主动改尺寸**（靠内容决定大小最稳）
    wins.keep_size(win)
    gui_ok = True
    try:
        import cv2
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    except Exception:                                    # noqa: BLE001
        gui_ok = False

    try:
        if gui_ok:
            print(f"  正在显示 #{index}：{label}")
            if status:
                print(f"  当前状态：{status}（改选会覆盖它）")
            print(f"  {_KEY_HINT}")
            # ⚠️ **相机掉线必须能自己退出来**（2026-10-14 修）。
            # 原来这里 `while True` 里对 `read()` 失败**不做任何处理** ——
            # 相机被别的程序抢走、或者 USB 掉线之后，循环照转，
            # 画面冻在最后一帧、按键也没有反馈，用户看到的就是"卡死了"，
            # 只能去关整个窗口。**这正是"启动器卡死"报告的一个来源。**
            miss = 0
            while True:
                ok, frame, _t = src.read()
                if ok and frame is not None:
                    miss = 0
                    import cv2
                    # ⚠️ **显示前先降采样**（2026-10-18，用户反馈"画面依然很卡顿"）。
                    #
                    # 外接相机实际给 2560x1440（请求被无视）；而识别只需要
                    # "能看出这是哪台"。每帧直接 imshow 1440p 在中低端核显上会
                    # 明显掉帧，再加上 putText 也在全尺寸上做，就更慢。
                    # 降到 960 宽后，imshow 的拷贝量降到 1/4。
                    disp = frame
                    if frame.shape[1] > PREVIEW_DISPLAY_W:
                        ratio = PREVIEW_DISPLAY_W / float(frame.shape[1])
                        disp = cv2.resize(
                            frame,
                            (PREVIEW_DISPLAY_W, max(1, int(frame.shape[0] * ratio))),
                            interpolation=cv2.INTER_AREA)
                    cv2.putText(disp, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 255), 2, cv2.LINE_AA)
                    if status:
                        cv2.putText(disp, f"now: {status}", (8, 54),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                                    cv2.LINE_AA)
                    try:
                        wins.show(win, disp)
                    except cv2.error:
                        gui_ok = False
                        break
                else:
                    miss += 1
                    if miss == 30:                    # 约 1 秒
                        print(f"  ⚠ #{index} 读不到画面了…（镜头被挡 / 被别的程序抢走 / "
                              f"USB 掉线？）再等 1 秒，还不行就跳过这台")
                    if miss >= 60:                    # 约 2 秒
                        print(f"  ✘ #{index} 连续 2 秒读不到画面，跳过这台相机。")
                        # ⚠️ 排查顺序**先问"这台是哪台"**（2026-10-15 加，坑 37）：
                        # 本机的**笔记本内置相机被用户有意遮住**，它必然读不到画面 ——
                        # 那是**预期行为**。原来的提示只写"是不是别的程序占着"，
                        # 会把内置相机这种正常情况也引向"去查占用/拔插"。
                        print(f"     排查（按顺序）：① 这台是不是**笔记本内置相机**"
                              f"（名字含 `Integrated`）？本机它是**有意被遮住**的，"
                              f"读不到画面属正常 → 按 `n` 标「无关」即可；")
                        print(f"     ② 若是外接那台：镜头是否被挡/盖 → 别的程序"
                              f"（QQ/会议/浏览器）是否占着 → 拔插 USB。")
                        return None
                key = cv2.waitKey(1) & 0xFF
                if wins.handle_key(win, key, disp):
                    continue
                if key == ord("q") or key == 27:
                    return ""
                if key in _ASSIGN_KEYS:
                    return _ASSIGN_KEYS[key]
        if not gui_ok:
            # 没有图形界面：打印设备信息，用命令行输入角色名
            print(f"  #{index}：{label}")
            if status:
                print(f"    当前状态：{status}")
            try:
                ans = input(f"  记为哪个？[left/right/main/none(无关)，回车=跳过，q=退出] "
                            ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                return ""
            if ans in ("q", "quit", "exit"):
                return ""
            if ans in ("n", "no", "none", "ignore", "无关", "无"):
                return IGNORE_TOKEN
            return ans or None
        return None
    finally:
        src.release()
        try:
            import cv2
            cv2.destroyWindow(win)
        except Exception:                                # noqa: BLE001
            pass


def _describe_status(reg: cr.Registry, index: int,
                     entry: cr.DeviceEntry) -> str:
    """这台设备当前是什么状态（left / right / main / 无关 / 未设置）。"""
    device = entry if entry.identity or entry.port or entry.name \
        else cr.DeviceEntry(order=index)
    role = cr.role_of_device(reg, device)
    if role:
        return role
    if cr.is_device_ignored(device, reg):
        return "无关"
    return ""



def cmd_probe(args) -> int:
    """回答一个**不需要知道"哪台是哪台"**的问题：**现在有几台相机真的能出图？**

    ## 为什么必须有这个命令（2026-10-18 用户的真实困惑）

    用户反馈：「刷新设备列表和检查映射不能保证有用」+「aoni 摄像头又是只能检测到一个」。

    前半句是**对的**，因为那两个功能都**只读配置/枚举，根本不打开相机**：

    | 入口 | 实际做了什么 | 能不能回答"相机活着吗" |
    | ---- | ------------ | ----------------------- |
    | 刷新设备列表 | list_devices()（读注册表/PnP） | ❌ 只能列出**存在**的设备 |
    | 检查相机映射 | cameras.py check（比对配置 vs 枚举） | ❌ 只看配置有没有漂移 |

    而后半句的机制是本机 **MSMF 间歇性变慢/打不开**（实测 MSMF #0 一次要 20 s 以上，
    另一次直接超时），于是"单个后端试着开"的界面就只剩一台。

    ## 这个命令的口径（**只说它敢保证的事**）

    对**每个后端**、每个编号，真的开一次、读几帧，报「有图 / 纯黑 / 打不开 / 读不到帧」。
    于是"两台 aoni 是不是都活着"有了确定答案。

    ⚠️ **它不能用来绑角色**：两个后端的编号**不是同一台设备**
    （实测 DSHOW #1 = 笔记本内置、#2 才是外接）。本命令只回答"有几台能出图"，
    绑角色请用 pin <role> name:"<完整名字>" —— 那条路不开相机、不碰编号。
    """
    from sphere_tracker.sources import backend_name

    import cv2

    devices = cr.list_devices()
    names = {d.order: d.name for d in devices}
    max_index = args.max_index
    which = args.backends
    backends = []
    if which in ("both", "msmf"):
        backends.append(cv2.CAP_MSMF)
    if which in ("both", "dshow"):
        backends.append(cv2.CAP_DSHOW)

    print("=" * 68)
    print("相机能不能出图（真的开一次，逐后端 × 逐编号）")
    print("=" * 68)
    if devices:
        print(f"系统里存在 {len(devices)} 台：")
        for d in devices:
            print(f"    #{d.order}  {d.name or '(无名字)'}")
    else:
        print("（枚举不到设备 —— 只能按编号盲探）")
    print()
    print("⚠️ **不同后端的编号不是同一台设备**，所以下面只回答"
          "「这个后端下这个编号能不能出图」，")
    print("   **不能**用来决定哪台是 left / 哪台是 right（绑角色请用 pin name:）。")
    print()

    # "能打开"不等于"有可用画面"：**每帧几乎恒定的纯色**（std 很小）说明镜头被挡/
    # 被遮住/没曝光。本机那台内置相机就是被有意遮住的 —— 若把它算进"能出图"，
    # 用户会得到"两台都在"的错误结论，而这正是他这次困惑的来源之一。
    DARK_STD = 5.0

    counts: dict[str, int] = {}
    working: dict[str, set[int]] = {}          # 每个后端下"真正出图"的编号
    for api in backends:
        nm = backend_name(api)
        ok_n = 0
        dark_n = 0
        working[nm] = set()
        cells = []
        for idx in range(max_index + 1):
            r = cr.probe_index(idx, width=640, height=480, fps=30.0, backend=api)
            if r.ok and r.std >= DARK_STD:
                ok_n += 1
                working[nm].add(idx)
                cells.append(f"#{idx} ✅ 有图 {r.width}x{r.height} std{r.std:.0f}")
            elif r.ok:
                dark_n += 1
                cells.append(f"#{idx} ⚠ 近乎纯黑（std{r.std:.1f}，std<{DARK_STD:.0f}）"
                             f" —— 可能是镜头被挡/被遮住")
            else:
                first = (r.error or "").splitlines()[0][:26]
                cells.append(f"#{idx} ✘ {first}")
        counts[nm] = ok_n
        print(f"  {nm:5} **真正出图 {ok_n} 台**" +
              (f"（另有 {dark_n} 台能打开但画面近乎纯黑）" if dark_n else ""))
        for c in cells:
            print(f"        {c}")
        print()

    best = max(counts.values()) if counts else 0
    print("-" * 68)
    print(f"结论：**最多 {best} 台**能真正出图"
          f"（{'、'.join(f'{k} {v} 台' for k, v in counts.items())}）。")
    if best >= 2:
        print("  → 有一个后端能同时开出两台，可用来做双目。"
              "标定时请**显式指定**该后端（--backend dshow|msmf）。")
    elif len(working) >= 2 and sum(len(v) for v in working.values()) >= 2 \
            and not (set.intersection(*working.values()) if working else set()):
        # 每个后端各只有一台、且**编号互不相同** —— 本机实测就是这种状态
        print("  → ⚠️ **两台各自只在不同的后端下工作，没有任何后端能同时开出两台。**")
        for nm, idxs in working.items():
            print(f"        {nm}：编号 {sorted(idxs)} 能出图")
        print("     这不是识别逻辑的问题，是**后端/驱动层面的状态**。"
              "双目要求同一个后端同时开两路，所以现在**标定跑不起来**。")
        print("     按顺序试：① 两台都拔掉、等 5 秒、插回（重置驱动状态）；"
              "② 换到机身不同的 USB 口；")
        print("     ③ 关掉可能占用相机的程序；④ 再跑一次本命令，"
              "看是否变成「两台都在同一个后端」。")
    else:
        print("  → 只有一台能出图。按顺序查：① 是否有一台**镜头被挡/被遮住**"
              "（近乎纯黑就是它）；")
        print("     ② 拔插换 USB 口；③ 是否被别的程序（会议/浏览器）占着。")
    return 0 if best >= 1 else 1


def _pin_example_lines(devices: list[cr.DeviceEntry]) -> list[str]:
    """按**当前实际设备**生成 `pin` 示例命令（不写死任何型号名）。

    ## 为什么改成动态生成（2026-10-18）

    原来这两行是写死的：

        python apps/tools/cameras.py pin left  name:"aoni webcam A25 left"
        python apps/tools/cameras.py pin right name:"aoni webcam A25 right"

    换一台相机、或用户改了名，这段提示就变成**错的/看不懂的** ——
    而它正是"名字不可靠时该怎么办"的关键指引，错不起。
    现在从 `list_devices()` 取**真实存在**、且未被标「无关」/未被判为内置的设备。
    """
    usable = _usable_for_pin(devices)
    if len(usable) < 2:
        return ["     python apps/tools/cameras.py list   # 先看有哪些设备名可用"]
    a, b = usable[0], usable[1]
    return [
        f'     python apps/tools/cameras.py pin left  name:"{a.name}"',
        f'     python apps/tools/cameras.py pin right name:"{b.name}"',
    ]


def _usable_for_pin(devices: list[cr.DeviceEntry]) -> list[cr.DeviceEntry]:
    """能拿来当 `pin` 示例的设备：有名字、没被标「无关」、不是内置/红外。

    单独抽出来是为了让 `_pin_hint_lines()` 能判断"示例够不够两台" ——
    提示语和示例必须**同一套判据**，否则会出现"提示说可以照抄"、
    下面却只给了一行 `list` 的自相矛盾。
    """
    reg = cr.load_registry()
    return [d for d in devices
            if d.name
            and not cr.is_device_ignored(d, reg)
            and not cr.is_avoided(d.name, reg.settings)]


def _pin_hint_lines(devices: list[cr.DeviceEntry], lead: str) -> list[str]:
    """一条"跳过开相机、按名字绑身份"的提示块（含可复制的两行命令）。

    为什么要专门做成提示块（2026-10-29）：`pin` 是**秒级**的快路，
    却只藏在 `--help` 里 —— 用户在 identify 里为每台等 30 秒时，
    完全不知道还有这条路。提示词**必须**和实际示例一起打，
    光说"用 pin"没人会去翻帮助找参数格式。

    示例一律走 `_pin_example_lines()`（从**当前真实设备名**生成），
    不另拼一份 —— 两份示例迟早对不上。
    """
    lines = [lead, *_pin_example_lines(devices)]
    n = len(_usable_for_pin(devices))
    if n < 2:
        # 少于两台时上面只能给一行 `list` 提示，读起来像"没有示例可用" ——
        # 说明白**为什么**没有，比让用户自己去猜"是不是工具坏了"强。
        lines.append(f"     （当前只枚举到 {n} 台可用设备：另一台没插、被标成「无关」")
        lines.append("       或被判为内置/红外；插好后重跑 list，上面就是现成的两行）")
    return lines


# 头部先报这几个角色（其余自定义角色排在后面）—— left/right/main 是启动器
# 实际会用的三个，先报它们才能对上用户的直觉。
_HEADER_ROLES = ("left", "right", "main")


def _role_expected(spec: cr.RoleSpec) -> str:
    """把 `RoleSpec` 拼成"期望 名字 … / 端口 …"（有哪个写哪个）。

    不写死任何值、也不复用 `resolve_role` 的报错文本 —— 那段报错很详细，
    但它是给"解析失败"用的整块提示，塞进一行头部会淹没其它信息。

    只有名字和端口**都没有**时才写身份：InstanceId 又把端口编在里面、
    又特别长，三者一起打会把这一行拖成两屏（实测那行 90+ 字）。
    """
    parts = []
    if spec.name:
        parts.append(f"名字 {spec.name}")
    if spec.port:
        parts.append(f"端口 {spec.port}")
    if not parts and spec.identity:
        parts.append(f"身份 {spec.identity}")
    return " / ".join(parts) if parts else "配置里没记任何身份信息"


def _role_status_lines(devices: list[cr.DeviceEntry]) -> list[str]:
    """每个已配置角色**现在指向哪台**：✔ 身份对上 / ⚠ 只是旧索引兜底 / ✘ 不在线。

    ## 为什么放在 identify 的最前面（2026-10-29）

    用户点「换相机…」时的处境经常是"配置先错了"：本机实测 `left` 绑的那台
    **枚举不到**，而 `right` 绑的恰好是在线的那台（左右绑反 + left 离线）。
    这时逐台开相机只会得到一串 ✘ —— 用户看到的是"程序有问题"，
    真正该做的是重新 `pin`。先把角色指向报出来，"配置错"和"开不出图"
    才分得开（前者不用开相机就能看见）。

    全程**不开相机**：`resolve_role(interactive=False)` 只做 identity/port/name
    静态匹配，毫秒级。

    返回空列表 = 配置里一个角色都没有（此时整段不打，免得开局就吓人）。
    """
    reg = cr.load_registry()
    if not reg.roles:
        return []
    order = [r for r in _HEADER_ROLES if r in reg.roles]
    order += [r for r in sorted(reg.roles) if r not in order]

    out: list[str] = []
    for role in order:
        spec = reg.roles[role]
        try:
            got = cr.resolve_role(role, reg=reg, devices=devices, interactive=False)
        except RuntimeError:
            out.append(f"  ✘ 角色 {role} 当前不在线（期望 {_role_expected(spec)}）")
            continue
        if got.how == "index-fallback":
            # ⚠ 千万别把它写成 ✔：`index-fallback` = 身份/端口都没对上，
            # 只是蒙了配置里的旧编号。换过 USB 口 / 换过机器时这个编号会漂，
            # 而**它照样能把画面开出来** —— 用户会以为绑定是对的。
            out.append(f"  ⚠ 角色 {role} 身份/端口没对上，只是按旧索引 "
                       f"#{got.index} 兜底 —— 换过 USB 口/换过机器时索引会漂，"
                       f"先确认这一台是不是它")
            continue
        how_cn = {"identity": "身份", "port": "端口", "name": "名字",
                  "manual": "手动指定"}.get(got.how, got.how)
        shown = f"#{got.index}"
        if got.entry is not None and got.entry.name:
            shown += f" {got.entry.name}"
        out.append(f"  ✔ 角色 {role} → {shown}（按{how_cn}对上）")

    out.append("  （以上只是身份匹配，不开相机：✔/⚠ 只代表\"这台在设备清单里\"，"
               "能不能出图看下面逐台的 ✔/✘）")
    return out


def cmd_identify(args) -> int:
    devices = cr.list_devices()
    by_order = {d.order: d for d in devices}
    indices = parse_indices(args.indices, devices, args.max_index)
    if not indices:
        print("没有可识别的索引。先跑 python apps/tools/cameras.py list 看看设备。")
        return 1

    # ---- "要等多久"必须在**动手之前**说清（2026-10-29，用户报"卡死"）----
    #
    # 用户点「换相机…」后 2~3 分钟一个窗口都不出现，就判定卡死。真实原因是
    # 逐台开、每台最坏 `PREVIEW_FIRST_FRAME_S` 秒（慢相机实测 ~24 秒/台），
    # 而这段时间**一句话都不打**。所以：先报总预算，再报"还能只挑几台"，
    # 最后给出**不用开相机**的快路（pin）。

    # ---- 等待期能不能按键跳过，先在开头说一声（2026-10-31，P4/P5）--------
    #
    # 旧版等待期**按什么键都无效**（按键循环在首帧之后）。现在等待期可中断，
    # 但前提是**能读到控制台键盘** —— 从 GUI 子进程启动时读不到，必须如实告知
    # "跳不了，要中止请 Ctrl+C"，而不是让用户以为按了没反应。
    if _console_keys_available():
        print("等待期间可按 Esc 或 s 跳过当前这台（不留悬空相机）")
    else:
        print("（从非控制台启动，无法按键跳过；要中止请 Ctrl+C）")

    n = len(indices)
    timeout = args.first_frame_timeout
    worst = n * timeout
    extra = f"（约 {worst / 60:.0f} 分钟）" if worst >= 60 else ""
    print(f"共 {n} 台待识别，最坏约 {n}×{timeout:.0f} 秒{extra}"
          f"；慢相机实测 ~24 秒/台。")
    if not args.indices:
        # 只在**用户没指定**时提示：已经写了 --indices 还提示一次，读起来像没生效。
        print(f"  只想确认某几台：加 --indices "
              f"{','.join(str(i) for i in indices[:2])}（逗号分隔，其余整台跳过）")
    for line in _pin_hint_lines(
            devices, "想跳过这 30 秒/台的等待？按名字绑身份即可（不开相机）："):
        print(line)
    print()

    # ---- 角色现在指向哪台（纯静态匹配，**不开相机**，毫秒级）--------------
    #
    # 为什么要在开相机之前先报：本机 2026-10-29 的真实状态是"left 绑的那台
    # 枚举不到、right 绑的倒是在线"—— 也就是说用户点「换相机…」之前，
    # 配置已经错了。不先说出来，用户只会看到一堆窗口、并把"配置错"当成"程序卡"。
    #
    # 边界：**只提示、不阻止**。`--indices` 是用户显式指定的编号，绝不能拿
    # 角色解析去否决它（否则就是"我写了 --indices 0 它却不给我看"）。
    role_lines = _role_status_lines(devices)
    for line in role_lines:
        print(line)
    if role_lines:
        print()

    # ---- 选定**本次会话唯一**的后端（绝不在索引之间切换）-----------------
    #
    # 为什么必须在会话级定死：`#N` 的标签来自 PnP 枚举，而 cv2 用的是**后端自己的
    # 枚举**。只有 MSMF 实测与 PnP 位次一致；DSHOW 整体错位（#1=内置、#2=aoni）。
    # 所以换后端 = 标签与画面脱钩 = 用户把角色绑到错的相机上（2026-10-18 的真 bug）。
    from sphere_tracker.sources import INDEX_BACKEND, INDEX_BACKEND_NAME, backend_name

    backend = INDEX_BACKEND if not getattr(args, "backend", None) else {
        "msmf": __import__("cv2").CAP_MSMF,
        "dshow": __import__("cv2").CAP_DSHOW,
    }[args.backend]
    used = backend_name(backend)
    print(f"后端：**{used}**（整场只用这一个，不会中途换）")
    if backend != INDEX_BACKEND:
        print("=" * 70)
        print(f"⚠️ 你指定了 {used}。**它的枚举顺序与本机设备列表不一致**：")
        print("   实测本机 DSHOW 的编号整体错位（`#1` 是笔记本内置、"
              "`#2` 才是外接那台），")
        print(f"   而 {INDEX_BACKEND_NAME} 与设备列表一致。")
        print(f"   用 {used} 识别，**可能把画面上的相机贴成另一台的名字**。")
        print("   除非你已经确认过这个后端的编号，否则建议：按 Ctrl+C 退出，改用")
        for line in _pin_example_lines(devices):
            print(line)
        print("   —— 按名字直接绑，**根本不用开相机**，也就没有错位问题。")
        print("=" * 70)

    reg = cr.load_registry()
    # 🔴 这段警告是 2026-10-18 用户第二次纠正后加的，**不要删**。
    #
    # 用户问："视频画面的内容是不是根本不是通过相机的名字来设置的?
    #          为什么我的命名为 right 的相机却在画面上是 left?"
    #
    # 答案：**是，画面只按编号打开，名字根本不参与。**
    # 窗口上 `#N 推测:<名字>` 里的名字，是"编号 N ↔ 设备列表第 N 台"这个**假设**的产物。
    # 而这条假设 **只在"两台 aoni 谁在 #0/#1"这一层上从未被验证过** ——
    # 两台同 VID/PID、同分辨率、同画面特征，**软件层面不可区分**。
    # 所以标签可能是反的，**必须以你眼睛看到的画面为准**；
    # 要"名字说了算"，请用 `pin name:`（那条路径不开相机、不碰编号）。
    print("⚠️ 下面每台窗口的标签写着「推测:<名字>」——**名字是猜的，不是事实。**")
    print("   `cv2.VideoCapture` 只能按**编号**打开，名字不参与；"
          "而编号与设备列表的对应关系")
    print("   在两台同型号相机之间**无法由程序证明**（它们同 VID/PID、同分辨率）。")
    print("   所以：**请以画面为准**按键。若你已用名字区分了左右，更可靠的做法是")
    for line in _pin_example_lines(devices):
        print(line)
    print("   —— 按名字绑身份，**不开相机、不碰编号**，也就不存在左右颠倒。")
    print("\n逐个索引显示画面，请按提示确认哪台是左、哪台是右、哪台「无关」：")
    print(f"（按键：{_KEY_HINT}）\n")

    # `_unopened` 只收"这个后端打不开"的编号（由 `_preview_and_assign` 的
    # `failed_out` 写进来）；**用户主动按 Esc/s 跳过的不算** —— 收尾文案说的是
    # "没能打开、去换个后端试试"，把主动跳过混进去会误导用户（2026-10-31 修）。
    _unopened: list[int] = []
    for idx in indices:
        entry = by_order.get(idx) or cr.DeviceEntry(order=idx)
        status = _describe_status(reg, idx, entry)
        role = _preview_and_assign(idx, entry, status, backend=backend,
                                   first_frame_timeout=timeout,
                                   failed_out=_unopened)
        if role == "":                                   # 退出
            break
        if not role:                                     # 跳过/打不开：不改配置
            continue

        verified = bool(entry.identity)
        if role == IGNORE_TOKEN:
            # 标「无关」：顺手把它从 left/right/main 里摘掉（mark_ignored 内部做）
            cr.mark_ignored(reg, entry=entry, index=idx, verified=verified)
            where = f"#{idx} {entry.name}" if entry.name else f"#{idx}"
            print(f"  [无关] <- {where}（自动挑相机时会跳过它）")
            continue

        if role not in ("left", "right", "main"):
            print(f"  忽略未知角色「{role}」（只支持 left/right/main/无关）")
            continue
        # 反过来：一旦成为角色，就不再是「无关」
        cr.unmark_ignored(reg, entry=entry, index=idx)
        cr.set_role(reg, role, entry=entry, index=idx, verified=verified)
        if verified:
            print(f"  [{role}] <- #{idx} {entry.name}（身份已记录）")
        else:
            print(f"  [{role}] <- #{idx}（**拿不到设备身份**，只记了索引；"
                  f"重插 USB 后可能漂移，请务必先用 identify 确认）")

    try:
        path = cr.save_registry(reg)
    except OSError as exc:
        print(f"\n[错误] 写配置失败：{exc}")
        return 1
    print(f"\n已写入 {path}")
    print("之后启动标定/检测可直接用角色名：--left left --right right")
    # ⚠️ 角色指向同一台相机 = 配置自相矛盾（2026-10-15 加）。
    # 这次就是这么坏的：相机 #0 用 MSMF 打不开 → 用户只能给 #1 按键 →
    # 先后按了 r 和 l → left 与 right 记成同一台，**而工具一声不响**。
    # 双目拿这个配置去标定必然失败，且失败点离原因很远。
    by_identity: dict[str, list[str]] = {}
    for role, spec in reg.roles.items():
        key = spec.identity or f"idx:{spec.index}"
        by_identity.setdefault(key, []).append(role)
    clash = {k: v for k, v in by_identity.items() if len(v) > 1}
    if _unopened:
        print()
        print(f"以下编号在 {used} 下**没能打开**（不是没插，是这个后端打不开）：{_unopened}")
        print("  想确认它们在**别的后端**下是否可用：")
        print(f"     python apps/tools/cameras.py probe --max-index {max(_unopened)}")
        print("  （probe 逐后端分别列出 —— **不同后端的编号不是同一台**，它也不下结论说哪台是哪台）")
    if clash:
        print("\n⚠ **配置有问题**：以下角色指向的是**同一台**相机 ——")
        for key, roles in clash.items():
            print(f"    {' / '.join(roles)}  ->  {key}")
        print("    双目要求左右是**两台不同的相机**，现在这份配置没法用来标定。")
        print("    原因通常是**有一台没能打开**，于是你只能给另一台按键 ——")
        print("    于是两个角色记到了同一台上。改法有两种：")
        print("      · 重跑 identify，确保两台各按一次（要目视确认画面）；")
        print("      · 或者**不开相机**直接改绑（推荐：不受后端/编号影响）：")
        print("          python apps/tools/cameras.py pin left  port:<左路那台的端口>")
        print("          python apps/tools/cameras.py pin right port:<另一台的端口>")
    if _backend_used:
        used = "、".join(f"#{i}->{b}" for i, b in sorted(_backend_used.items()))
        print(f"预览实际用的后端：{used}")
        names = {b.lower() for b in _backend_used.values()}
        if len(names) > 1:
            print("⚠ 两台用的后端**不一样**。标定/检测要用同一个后端，"
                  "请显式指定其中一个（建议 `--backend dshow`）。")
    if reg.ignored:
        names = "、".join((s.name or f"#{s.index}") for s in reg.ignored)
        print(f"已标为「无关」的相机（自动挑相机时跳过）：{names}")
    return 0


# --------------------------------------------------------------------------- #
# pin / ignore / unignore 共用的取值解析
# --------------------------------------------------------------------------- #
def _resolve_device_value(
    value: str, devices: list[cr.DeviceEntry]
) -> tuple[cr.DeviceEntry | None, int | None]:
    """`0` / `port:xx` / `id:xx` / `name:xx` -> (设备, 编号)。认不出来返回 (None, None)。

    纯数字即使枚举不到设备也认（返回的 entry 为 None），这样"离线清记录"也能做。
    """
    low = value.lower()
    if value.isdigit():
        idx = int(value)
        return next((d for d in devices if d.order == idx), None), idx
    for prefix, attr in (("port:", "port"), ("id:", "identity"), ("name:", "name")):
        if low.startswith(prefix):
            needle = value[len(prefix):].strip().lower()
            if not needle:
                return None, None
            entry = next((d for d in devices
                          if needle in str(getattr(d, attr)).lower()), None)
            return (entry, entry.order) if entry else (None, None)
    return None, None


_VALUE_HINT = "取值要写成 索引(0) / port:xxx / id:xxx / name:xxx"


# --------------------------------------------------------------------------- #
# pin
# --------------------------------------------------------------------------- #
def cmd_pin(args) -> int:
    role, value = args.role, args.value
    devices = cr.list_devices()
    entry, index = _resolve_device_value(value, devices)

    if index is None:
        print(f"没有匹配 {value} 的相机。\n{_VALUE_HINT}")
        if devices:
            print(f"检测到的相机：\n{cr.format_device_table(devices)}")
        return 1

    reg = cr.load_registry()
    verified = bool(entry and entry.identity)
    old = cr.unmark_ignored(reg, entry=entry, index=index)
    cr.set_role(reg, role, entry=entry, index=index, verified=verified)
    try:
        path = cr.save_registry(reg)
    except OSError as exc:
        print(f"[错误] 写配置失败：{exc}")
        return 1

    where = f"#{index} {entry.name}" if entry else f"#{index}"
    print(f"[{role}] <- {where}  已写入 {path}")
    if old:
        print("（它原来被标为「无关」，已自动摘掉 —— 一台相机不能既是角色又是无关）")
    if not verified:
        print("提醒：这台拿不到设备身份，只记了索引；建议改用 identify 目视确认，"
              "否则重插 USB 后索引可能漂移。")
    return 0


# --------------------------------------------------------------------------- #
# ignore / unignore / ignored ——「无关」清单
# --------------------------------------------------------------------------- #
def cmd_ignore(args) -> int:
    devices = cr.list_devices()
    entry, index = _resolve_device_value(args.value, devices)
    if index is None:
        print(f"没有匹配 {args.value} 的相机。\n{_VALUE_HINT}")
        if devices:
            print(f"检测到的相机：\n{cr.format_device_table(devices)}")
        return 1

    reg = cr.load_registry()
    verified = bool(entry and entry.identity)
    old_role = cr.role_of_device(reg, entry) if entry else None
    cr.mark_ignored(reg, entry=entry, index=index, verified=verified)
    try:
        path = cr.save_registry(reg)
    except OSError as exc:
        print(f"[错误] 写配置失败：{exc}")
        return 1

    where = f"#{index} {entry.name}" if entry else f"#{index}"
    print(f"[无关] <- {where}  已写入 {path}")
    if old_role:
        print(f"（它原来是「{old_role}」，已一并摘掉）")
    print("自动挑相机（--auto-cameras / 默认）以后会跳过它。")
    if not verified:
        print("提醒：这台拿不到设备身份，只记了索引；重插 USB 后标记可能失效。")
    return 0


def cmd_unignore(args) -> int:
    reg = cr.load_registry()
    if args.all:
        n = cr.clear_ignored(reg)
        if not n:
            print("「无关」清单本来就是空的，不用清。")
            return 0
        try:
            path = cr.save_registry(reg)
        except OSError as exc:
            print(f"[错误] 写配置失败：{exc}")
            return 1
        print(f"已清空「无关」清单（{n} 条），写入 {path}")
        return 0

    if not args.value:
        print("要么给一个取值，要么用 --all 清空整份清单。\n" + _VALUE_HINT)
        return 1

    devices = cr.list_devices()
    entry, index = _resolve_device_value(args.value, devices)
    if index is None:
        print(f"没有匹配 {args.value} 的相机。\n{_VALUE_HINT}")
        return 1
    n = cr.unmark_ignored(reg, entry=entry, index=index)
    if not n:
        print(f"#{index} 本来就不在「无关」清单里，没改动。")
        return 0
    try:
        path = cr.save_registry(reg)
    except OSError as exc:
        print(f"[错误] 写配置失败：{exc}")
        return 1
    where = f"#{index} {entry.name}" if entry else f"#{index}"
    print(f"已取消「无关」标记：{where}（{n} 条），写入 {path}")
    print("现在自动挑相机又可以选它了。要固定角色请跑 identify。")
    return 0


def cmd_ignored(args) -> int:
    """打印「无关」清单，并标注每台当前在不在线。"""
    reg = cr.load_registry()
    if not reg.ignored:
        print("「无关」清单是空的 —— 没有任何相机被排除在自动挑选之外。")
        return 0
    devices = cr.list_devices()
    print(f"「无关」清单（{len(reg.ignored)} 台）—— 自动挑相机会跳过它们：")
    for i, spec in enumerate(reg.ignored, 1):
        who = spec.name or spec.identity or spec.port or "(没记名字)"
        # 当前是否在线：用**严格匹配**（不按名字），否则两台同型号会互相认错
        entry = cr.find_ignored_device(spec, devices)
        if entry is not None:
            state = f"在线 #{entry.order}"
        elif not spec.verified and spec.index is not None:
            now = next((d for d in devices if d.order == spec.index), None)
            state = (f"按索引 #{spec.index}（未确认身份，可能是「{now.name}」）"
                     if now is not None else f"离线（原索引 #{spec.index}）")
        else:
            state = "离线（没插 / 换过口）"
        print(f"  {i}. {who:<24} {state}")
    print("\n取消某一台：python apps/tools/cameras.py unignore <编号>")
    print("全部取消：  python apps/tools/cameras.py unignore --all")
    return 0


# --------------------------------------------------------------------------- #
# check
# --------------------------------------------------------------------------- #
def _pin_value_of(entry: cr.DeviceEntry | None, spec: cr.RoleSpec) -> str:
    """给 `pin` 命令用的取值：优先 port（每台唯一），其次 id，最后 name。

    为什么不首选 `name:`：本机两台同型号，`identify` 记下的名字可能**完全一样**
    （2026-10-31 的现状就是 left 与 right 都记成 "aoni webcam A25 left"），
    而 `pin name:` 的匹配是"子串命中的第一台" —— 拿同一个名字 pin 两个角色，
    只会**又绑回同一台**。port 才是每台唯一的那个。
    """
    if entry is not None:
        if entry.port:
            return f"port:{entry.port}"
        if entry.identity:
            return f"id:{entry.identity}"
    if spec.port:
        return f"port:{spec.port}"
    if spec.identity:
        return f"id:{spec.identity}"
    if spec.name:
        return f'name:"{spec.name}"'
    return "<这台的 port，见 list>"


def _role_collision_lines(
    reg: cr.Registry, devices: list[cr.DeviceEntry]
) -> tuple[bool, list[str]]:
    """「两个角色解析到同一台相机」的判定与说明。返回 (是否危险, 输出行)。

    ## 判据只调一次 `camera_registry.resolve_roles()`

    它才是"两个角色不许落到同一台相机"的**唯一实现**（按角色依次解析，把已认领的
    索引排除掉，撞车就抛 RuntimeError）。这里刻意**不**自己比 identity/index ——
    那是第二份判据，迟早与它漂移。

    而漂移的代价已经发生过一次（2026-10-31，用户被卡在死循环里）：

        `pair_source.py` 报「左右解析到同一个相机」→ 用户照报错跑 `check`
        → `check` **逐角色**比对，两个角色各自都"一致" →
        输出「未发现危险项」、**退出码 0** → 用户被弹回原点，且被告知"没事"。

    ## 危险由 resolve_roles 发起，是否"撞车"按**索引事实**复核

    撞车后逐个角色**单独**解析一次（单个角色之间不存在撞车），拿到"谁落在哪台、
    另一台是哪一个"。这不是第二份判据，而是**唯一一处**把异常翻译成结论的地方：
    只要有两个角色落在同一个索引，就是撞车。

    ⚠ 判据**不许看异常文案**。上一版写的是 `if "冲突" not in str(exc): return False`
    —— 那么 `camera_registry` 里那句话改一个字，`check` 就静默退回
    「未发现危险项 / EXIT 0」，**今天这个 bug 原样复发，而且没有测试会红**。
    所以这里按事实复核；也不碰任何相机。
    """
    roles = [r for r in ("left", "right", "main") if r in reg.roles]
    if len(roles) < 2:
        return False, []
    try:
        cr.resolve_roles(roles, reg=reg, devices=devices, interactive=False)
    except RuntimeError:
        # 可能撞车，也可能只是某一路解析失败 —— 下面按索引事实区分，
        # 不看异常文本（文案一改就静默变绿，那正是要防的）。
        pass
    else:
        return False, []

    landed: dict[int, list[str]] = {}
    entries: dict[int, cr.DeviceEntry | None] = {}
    for role in roles:
        try:
            res = cr.resolve_role(role, reg=reg, devices=devices, interactive=False)
        except RuntimeError:
            continue
        if res.index is None:
            continue
        landed.setdefault(int(res.index), []).append(role)
        entries.setdefault(int(res.index), res.entry)

    ignored = cr.ignored_orders(reg, devices)
    lines: list[str] = []
    for index, group in landed.items():
        if len(group) < 2:
            continue
        entry = entries.get(index)
        who = f"#{index} {entry.name}" if entry is not None and entry.name else f"#{index}"
        lines.append(f"[危险] {' 与 '.join(group)} 解析到同一台相机（{who}）")
        lines.append("       两台相机是双目的前提；请分别 pin（**不用开相机**）：")
        # 另一半的候选：在线、不是「无关」、不是撞车的这一台；内置/红外排后面
        others = [d for d in devices
                  if d.present and d.order not in ignored
                  and (entry is None or d.identity.upper() != entry.identity.upper())]
        others.sort(key=lambda d: cr.is_avoided(d.name))
        if len(group) == 2:
            a, b = group
            lines.append(f"         python apps/tools/cameras.py pin {a:<5} "
                         f"{_pin_value_of(entry, reg.roles[a])}")
            lines.append(f"         python apps/tools/cameras.py pin {b:<5} "
                         + (_pin_value_of(others[0], reg.roles[b]) if others
                            else "<另一台的 port，见 list>"))
        else:
            for role in group:
                lines.append(f"         python apps/tools/cameras.py pin {role:<5} "
                             f"{_pin_value_of(entry, reg.roles[role])}")
        if others:
            cand = others[0]
            lines.append(f"       另一台现在在线：#{cand.order} {cand.name or '(无名字)'}"
                         f"（端口 {cand.port or '-'}）")
            lines.append("       ⚠ 名字区分不了两台（本机同型号、可能连名字都一样），"
                         "所以用 port；")
            lines.append("         哪台是左、哪台是右只能你**目视**决定 —— "
                         "反了就把两条 pin 的取值对调。")
        elif not devices:
            lines.append("       （当前枚举不到任何相机设备，给不出候选 —— 先跑 list 看看）")
        else:
            lines.append("       除这一台外**没有别的在线相机**（没插好/没上电/被标成「无关」），")
            lines.append("       所以现在无论怎么 pin 都凑不出两台 —— 先把另一台插好。")
    if not lines:
        # 没有两个角色落在同一个索引 -> **不是撞车**（例如某一路只是解析失败，
        # 那条由下面的逐角色循环报成 [危险]）。这里必须返回"不危险"，
        # 否则每个解析失败都会被当成撞车，判据就失去意义。
        return False, []
    return True, lines


def cmd_check(args) -> int:
    reg = cr.load_registry()
    if not reg.roles and not reg.ignored:
        print("还没有固定任何相机（config/cameras.json 为空）。")
        print("第一次使用请先跑：python apps/tools/cameras.py identify")
        return 0

    devices = cr.list_devices()
    by_order = {d.order: d for d in devices}
    if not devices:
        print("（当前枚举不到任何相机设备；非 Windows 或系统枚举不可用 —— 只能按索引用）")

    danger = False

    # ---- 先判「左右撞车」：它比任何单台问题都根本 ----
    #
    # 放在逐角色循环**之前**，因为逐角色看每台都"一致"（它们各自确实解析成功），
    # 而问题恰恰是"两台角色指向同一台" —— 不说在前面，用户读完一堆「[一致]」
    # 只会以为没事。
    collide, collision_lines = _role_collision_lines(reg, devices)
    if collision_lines:
        for line in collision_lines:
            print(line)
        print("       （下面每个角色**单看**都是「一致」—— 问题不在某一台，"
              "而在这两个角色指向同一台。）")
        print()
        danger = True

    for role in sorted(reg.roles):
        spec = reg.roles[role]
        tag = "已确认" if spec.verified else "未确认"
        try:
            res = cr.resolve_role(role, reg, devices, interactive=False)
        except RuntimeError as exc:
            # ⚠️ **解析失败必须算危险项**（2026-10-15 修）。
            # 这里原来只 `continue`，**没有置 danger** —— 于是"角色解析失败"
            # 却返回退出码 0，而本文档与用户手册都承诺「有 [危险] 项时退出码为 2」。
            # 用户 2026-10-15 换 USB 口后跑 check 就是这样：报了一行失败、退出码 0，
            # 看起来像"没事"。
            danger = True
            print(f"[危险] 「{role}」({tag}) **解析失败**：{_first_line(exc)}")
            # 光说"不在线"没用 —— 换 USB 口也会这样，而那是本工具最容易让人困惑的一步。
            # 补一句"当前有没有同名的设备"（**只报告，不自动改绑**）。
            hint = cr.describe_moved_device(spec, devices, role)
            if hint:
                for line in hint.splitlines():
                    print(line)
            continue

        if res.how == "identity":
            status = "一致" if res.index == spec.index else f"漂移 -> 按身份纠正到 #{res.index}"
        elif res.how == "port":
            status = "一致（按端口）" if res.index == spec.index else f"漂移 -> 按端口纠正到 #{res.index}"
        elif res.how == "index-fallback":
            status = f"索引兜底 #{res.index}（未确认身份，可能用错）"
        else:
            status = f"手动指定 #{res.index}"
        print(f"[{'一致' if res.index == spec.index else '漂移'}] 「{role}」({tag}) {status}")

        for w in res.warnings:
            print(f"        [警告] {w}")

        # 危险：配置里记的那个索引，现在正被**另一台**设备占着
        if spec.identity and spec.index is not None:
            now = by_order.get(spec.index)
            if now is not None and now.identity and now.identity.upper() != spec.identity.upper():
                danger = True
                print(f"        [危险] 索引 #{spec.index} 现在指向的是「{now.name or now.identity}」，"
                      f"不是「{role}」原来那台！若用死索引启动就会用错相机。")
        # 危险：身份对上但设备名不符（同型号换了一台 / 记录过期）
        if spec.name and res.entry is not None and res.entry.name \
                and spec.name.lower() not in res.entry.name.lower():
            danger = True
            print(f"        [危险] 期望设备名「{spec.name}」，实际是「{res.entry.name}」。")

    # 「无关」清单：正常是信息，但"角色和无关指向同一台"属于配置自相矛盾
    if reg.ignored:
        print()
        for spec in reg.ignored:
            who = spec.name or spec.identity or spec.port or "(没记名字)"
            entry = cr.find_ignored_device(spec, devices)
            if entry is not None:
                print(f"[无关] 「{who}」在线 #{entry.order}（自动挑相机会跳过它）")
                clash = cr.role_of_device(reg, entry)
                if clash:
                    danger = True
                    print(f"        [危险] 它同时又是角色「{clash}」—— 配置自相矛盾，"
                          f"请重跑 identify 重新确认。")
            else:
                print(f"[无关] 「{who}」不在线（没插 / 换过口）—— 记录保留，插回来仍然生效")

    if danger:
        if collide:
            print("\n存在 [危险] 项：按上面的 pin 命令把左右分开（不用开相机）；"
                  "或重跑 python apps/tools/cameras.py identify 逐台目视确认。")
        else:
            print("\n存在 [危险] 项：请重跑 python apps/tools/cameras.py identify 重新确认。")
        return 2
    print("\n未发现危险项。若要每次开机都安心，可把这条命令放进启动脚本。")
    return 0


# --------------------------------------------------------------------------- #
# doctor：一次把"能不能跑双目"全查完
# --------------------------------------------------------------------------- #
# 为什么要有它（2026-08 ~ 10 的教训都指向同一件事）：
#
# 这台机器上"相机能不能用"是**浮动**的，而且失败时会以**五六种不同的面孔**出现：
# 打不开 / 能打开但无帧 / 出帧但恒为纯黑 / 只有某个后端行 / 两路分辨率不一致 /
# 配置指向已换口的旧身份。这些我这几轮**一条一条手跑命令去查**，
# 每次都写一段临时 Python —— 按项目规矩（重复 3 遍就固化）该收进工具里。
#
# 与 `check` 的分工：
#   `check`  = 配置 vs 设备（**不开相机**，秒级）—— 查"配置还对不对"
#   `doctor` = **真开相机**，查"现在到底能不能出图、能不能凑成一对"
# 两者都**只报告不改配置**。


# 「双路**同时**取帧」检查（S5，2026-10-31）的默认值。
#
# 为什么单列一组常量：双目是**两台同时**开的工况，与 `identify` 的**逐台串行**
# 不是一回事 —— 这里要同时持有两台相机句柄、各自读若干帧，耗时可以与单台预览叠加。
PAIR_FIRST_FRAME_S = 25.0    # 每台等首帧的上限（慢相机实测 ~24 s 才给第一帧）
PAIR_FRAMES = 15             # 每台读多少帧（够判断"有没有画面"+估帧率，不无限等）
PAIR_READ_BUDGET_S = 8.0     # 每台读帧的**总**预算，到点即收（把十几分钟级卡顿截断）
PAIR_SEARCH_BUDGET_S = 25.0  # 打开时的搜索总预算（坏相机到这里就如实收场）


def _simultaneous_grab_check(
    left_index: int, right_index: int, backend: int, *,
    width: int, height: int,
    frames: int = PAIR_FRAMES,
    first_frame_timeout: float = PAIR_FIRST_FRAME_S,
) -> bool:
    """两台**同时**打开、各自读若干帧 —— 与双目同工况（同一后端 + 同一分辨率）。

    返回 True = 双路同时取帧**失败或分辨率不一致**（调用方据此判 fail）；
    False = 两台能同时出图且实际分辨率一致。全程打印进度与可判定结论。

    ## 为什么必须与 `identify` 分开做（这是本段存在的唯一理由）
    `identify` 是**逐台串行**的：开一台、读完就在 `finally` 里 `release()`，
    同一时刻只有一台相机活着。而双目要的是两台**同时**开着。
    设备级失效（`MF_E_VIDEO_RECORDING_DEVICE_INVALIDATED`）在串行下测不出来，
    只有"两台同时"才复现 —— 这就是 2026-10-31「identify 说两台都好、标定却说
    取不到成对帧」那个"自相矛盾"的来源。详见
    docs/planbooks/2026-10-31_双目标定取帧失败与分辨率档.md §2 / §4.1。
    """
    from sphere_tracker.sources import backend_name, explain_grab_error

    frames = max(1, int(frames))
    bname = backend_name(backend)
    print("\n双路**同时**取帧（双目同工况：同一后端 + 同一分辨率）：")

    if left_index == right_index:
        print(f"    ✘ 左右解析到**同一台**（都是 #{left_index}）—— 配置自相矛盾，"
              f"双目标定必然失败。")
        print("      改法：重跑 `python apps/tools/cameras.py identify`（两台各按一次），"
              "或用 `pin` 按名字绑。")
        return True

    print(f"    正在同时打开 left(#{left_index}) 与 right(#{right_index})…")
    print(f"    （后端 {bname}，请求 {width}x{height}；每台最坏等 ~{first_frame_timeout:.0f}s "
          f"首帧，再各读 {frames} 帧 / {PAIR_READ_BUDGET_S:.0f}s 超时）")

    srcs: dict[str, object] = {}
    stats: dict[str, dict] = {}
    # 两台**同时存活**：先把左开出来（构造里含"等首帧"）并**不释放**，
    # 右紧接着打开 —— 于是右打开期间左仍占着设备句柄，与双目启动一致。
    for role, idx in (("left", left_index), ("right", right_index)):
        t0 = time.perf_counter()
        try:
            srcs[role] = CameraSource(
                index=idx, width=width, height=height, backend=backend,
                optimize=False, search_budget_s=PAIR_SEARCH_BUDGET_S,
                first_frame_timeout_s=first_frame_timeout)
            print(f"      ✔ {role}(#{idx}) 已打开（{time.perf_counter() - t0:.1f}s）")
        except Exception as exc:                          # noqa: BLE001
            stats[role] = {"opened": False, "frames": 0, "w": 0, "h": 0, "fps": 0.0}
            print(f"      ✘ {role}(#{idx}) **打不开**（{time.perf_counter() - t0:.1f}s）："
                  f"{_first_line(exc)}")

    # 各自读帧（**两路都还开着**的时候读）。
    for role in ("left", "right"):
        src = srcs.get(role)
        if src is None:
            continue
        n_ok = 0
        t0 = time.perf_counter()
        for _ in range(frames):
            try:
                ok, frame, _t = src.read()
            except Exception:                             # noqa: BLE001
                ok, frame = False, None
            if ok and frame is not None:
                n_ok += 1
            if time.perf_counter() - t0 > PAIR_READ_BUDGET_S:
                break
        took = time.perf_counter() - t0
        info = getattr(src, "info", None)
        info_fps = float(getattr(info, "fps", 0.0) or 0.0)
        fps = (n_ok / took) if (n_ok and took > 0) else info_fps
        stats[role] = {
            "opened": True, "frames": n_ok,
            "w": int(getattr(info, "width", 0) or 0),
            "h": int(getattr(info, "height", 0) or 0), "fps": fps,
        }

    # 收尾：两台一起释放（无论成败都不留悬空相机）。
    for src in srcs.values():
        try:
            src.release()
        except Exception:                                 # noqa: BLE001
            pass

    lstat, rstat = stats["left"], stats["right"]
    print(f"    实读：left(#{left_index}) {lstat['frames']}/{frames} 帧，"
          f"right(#{right_index}) {rstat['frames']}/{frames} 帧")

    if lstat["frames"] > 0 and rstat["frames"] > 0:
        same = (lstat["w"], lstat["h"]) == (rstat["w"], rstat["h"])
        note = (f"左 {lstat['w']}x{lstat['h']} @ {lstat['fps']:.1f} fps，"
                f"右 {rstat['w']}x{rstat['h']} @ {rstat['fps']:.1f} fps")
        if same:
            print(f"    ✅ 两台可同时取帧（{note}，后端 {bname}）")
            return False
        print(f"    ⚠ 两台分辨率不一致（{note}）→ 双目标定会失败，需同一档")
        return True

    # 至少一路 0 帧或打不开 → 失败。
    print("    ❌ 双路同时取帧失败：")
    for role, idx in (("left", left_index), ("right", right_index)):
        st = stats[role]
        if st["frames"] > 0:
            continue
        if not st["opened"]:
            print(f"       · {role}(#{idx}) 在 {bname}@请求{width}x{height} 下**打不开**"
                  f"（构造即失败，内部已逐档试过）")
        else:
            print(f"       · {role}(#{idx}) 在 {bname}@请求{width}x{height} 下 **0 帧**"
                  f"（实读 {st['frames']}/{frames}，实际 {st['w']}x{st['h']}）")
    print("       （两路同时取帧失败最常见的错误码及含义：）")
    for ln in explain_grab_error().splitlines():
        print(f"       {ln}")
    print("       → 建议：**物理拔插**上面这路相机（设备级失效软件救不回来）；"
          "仍不行，换一个")
    print("         **直接插在机身上**的 USB 口（别经 HUB / 扩展坞）。")
    return True


def cmd_doctor(args) -> int:
    """打开每台相机实测一遍，回答"现在能不能跑双目"。"""
    import tempfile

    from sphere_tracker import camera_registry as cr
    from sphere_tracker.sources import backend_name

    reg = cr.load_registry()
    devices = cr.list_devices()
    want_w, want_h = (args.width, args.height)

    print("=" * 68)
    print("相机体检（真开相机；只报告，不改配置）")
    print("=" * 68)
    print(f"目标采集档：{want_w}x{want_h}（来自 video_profiles.DEFAULT）")
    if not devices:
        print("枚举不到任何相机设备（非 Windows / 注册表不可读 / 真的没插）。")
        return 1

    # ---- ① 枚举 ----
    print(f"\n① 枚举到 {len(devices)} 台：")
    for d in devices:
        mark = "  [无关]" if cr.is_device_ignored(d, reg) else ""
        print(f"    #{d.order}  {d.name or '(无名字)':22} 端口 {d.port or '-'}{mark}")

    # ---- ② 角色解析（不开相机）----
    print("\n② 角色解析：")
    resolved: dict[str, int | None] = {}
    fail = False
    for role in ("left", "right"):
        spec = reg.roles.get(role)
        if spec is None:
            print(f"    「{role}」：配置里没有这个角色 → 先跑 identify")
            resolved[role] = None
            fail = True
            continue
        try:
            r = cr.resolve_role(role, reg, devices, interactive=False)
            resolved[role] = r.index
            print(f"    「{role}」-> #{r.index}（how={r.how}）")
        except RuntimeError as exc:
            resolved[role] = None
            fail = True
            print(f"    「{role}」-> **解析失败**：{_first_line(exc)}")
            for line in cr.describe_moved_device(spec, devices, role).splitlines():
                print(line)

    # ---- ③ 每台实测：**在唯一可信的后端上**真能出图吗 ----
    #
    # ⚠️ 2026-10-18 修：原来用 `pick_backend_for_device()` 给每个索引单独挑后端 ——
    # 那是错的。`resolved[role]` 是 **PnP 位次**，只有 MSMF 与它一致；
    # 换成 DSHOW 后同一个编号指向**另一台物理相机**（实测 #1=内置、#2=aoni），
    # 于是"实测这一路"其实测的是别的相机，结论整个失真。
    # 现在：**整场只用一个后端**，与 `identify` 同一条纪律。
    from sphere_tracker.sources import INDEX_BACKEND, INDEX_BACKEND_NAME

    probe_backend = INDEX_BACKEND
    print(f"\n③ 逐台实测（后端固定为 **{INDEX_BACKEND_NAME}** —— 唯一与设备位次一致的后端；"
          f"判据是「真能出图」）：")
    usable: dict[str, tuple[int, int, int]] = {}      # role -> (index, api, std)
    for role in ("left", "right"):
        idx = resolved.get(role)
        if idx is None:
            continue
        probe = cr.probe_index(idx, width=want_w, height=want_h,
                               fps=30.0, backend=probe_backend)
        if not probe.ok:
            fail = True
            print(f"    「{role}」#{idx}：**出不了图** —— {probe.error}")
            print(f"       （不换后端试：换后端后 #{idx} 会指到**另一台**相机，"
                  f"测出来的不是这一台）")
            continue
        usable[role] = (idx, probe_backend, int(probe.std))
        print(f"    「{role}」#{idx}：{probe.width}x{probe.height}，"
              f"std {probe.std:.1f} ✅")

    # ---- ④ 两路能不能凑成一对（同后端 + 同分辨率）----
    print(f"\n④ 两路一致性（双目硬要求：**同一个后端 + 同一个分辨率**）：")
    if len(usable) < 2:
        fail = True
        print("    ✘ 不足两路可用，没法组双目。")
    else:
        (li, la, _), (ri, ra, _) = usable["left"], usable["right"]
        if la != ra:
            fail = True
            print(f"    ✘ 两路**后端不同**（left={backend_name(la)} / "
                  f"right={backend_name(ra)}）—— 同一进程混用会挂一路。")
        else:
            print(f"    ✓ 两路同后端（{backend_name(la)}）")
        # 用**临时缓存**做一致性测试，绝不污染真缓存（"缓存把分辨率钉死"
        # 正是踩过的坑：`config/camera_modes.json` 里记着 640x480，
        # 于是显式请求 1920x1080 也被盖过）。
        import sphere_tracker.sources as _src

        saved_cache = _src._MODE_CACHE
        res: dict[str, tuple[int, int] | None] = {}
        try:
            with tempfile.TemporaryDirectory() as td:
                _src._MODE_CACHE = Path(td) / "probe_modes.json"
                for role in ("left", "right"):
                    # ⚠️ 这里也**只测同一个后端**（同一条纪律），不按索引换后端。
                    probe = cr.probe_index(usable[role][0], width=want_w,
                                           height=want_h, fps=30.0,
                                           backend=usable[role][1])
                    res[role] = (probe.width, probe.height) if probe.ok else None
        finally:
            _src._MODE_CACHE = saved_cache
        print(f"    实测分辨率：left={res.get('left')}  right={res.get('right')}")
        if res.get("left") and res.get("left") == res.get("right"):
            print(f"    ✓ 两路一致 {res['left']} —— 满足标定要求")
        else:
            fail = True
            print(f"    ✘ 两路**不一致** —— 标定会直接失败。"
                  f"本机实测：两台外接相机都无视请求、恒给 2560x1440"
                  f"（请求 1920/1280 也一样）；不一致先查是不是混进了内置/红外。")

    # ---- 双路**同时**取帧（双目同工况；S5，2026-10-31）----
    #
    # 与上面 ③/④ 的区别：③/④ 走 `probe_index`，是**逐台**试开（开一台、读一帧、
    # 立刻 release），同一时刻只有一台相机活着；而双目要的是**两台同时**开着、
    # 同一后端同一分辨率同时取帧。设备级失效（-2147483638 / DEVICE_INVALIDATED）
    # **逐台串行测不出来**，只有"两台同时"才暴露 —— 这就是「identify 说两台都好、
    # 标定却取不到成对帧」那个"自相矛盾"的来源（planbook §2/§4.1）。故补这一段。
    if resolved.get("left") is None or resolved.get("right") is None:
        print("\n双路**同时**取帧（双目同工况）：")
        print("    未配置 left/right 角色（缺一个或没解析出来）→ **跳过**双路同时检查。")
        print("    先跑 `python apps/tools/cameras.py identify` 把左右固定下来。")
    elif _simultaneous_grab_check(
            resolved["left"], resolved["right"], probe_backend,
            width=want_w, height=want_h,
            frames=args.pair_frames, first_frame_timeout=args.pair_timeout):
        fail = True

    # ---- ⑤ 结论 ----
    print("\n" + "=" * 68)
    if fail:
        print("结论：**还不能跑双目。** 按上面 ✘ 那几行的指引处理；")
        print("     换过 USB 口先跑 `identify` 重新确认（InstanceId 里编着端口号）。")
        return 2
    print("结论：**可以跑双目。** 下一步：")
    print(f"    启动程序.bat precheck      （举棋盘确认两路都看得到）")
    print(f"    启动程序.bat calibrate     （标定；基线核对记得填你量到的中心距）")
    return 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="摄像头设备管理（列表 / 确认 / 固定 / 标无关 / 检查）",
        epilog="「无关」= 明确标记这台不要用；自动挑相机时会跳过它。",
    )
    sub = p.add_subparsers(dest="cmd")

    pl = sub.add_parser("list", help="列出摄像头设备（默认不开相机）")
    pl.add_argument("--probe", action="store_true", help="真的开一下，报实际分辨率/是否全黑")
    pl.add_argument("--json", action="store_true", help="以 JSON 输出（供脚本消费）")
    pl.add_argument("--max-index", type=int, default=6, help="探测的最大索引（默认 6）")
    pl.set_defaults(func=cmd_list)

    pi = sub.add_parser("identify", help="逐个索引显示画面，确认哪个是左/右/无关")
    pi.add_argument("--indices", default=None, help="只识别这些索引，如 0,2")
    pi.add_argument("--max-index", type=int, default=6, help="没给 --indices 时补探到第几个索引")
    pi.add_argument(
        "--first-frame-timeout", type=positive_seconds, default=PREVIEW_FIRST_FRAME_S,
        help="每台相机等第一帧的上限（秒，默认 30）。"
             "**实测慢相机 ~24s 才有画面，调小会让它在界面上消失**")
    pi.add_argument(
        "--backend", choices=("msmf", "dshow"), default=None,
        help="整场用哪个后端。**默认 msmf** —— 只有它与系统设备列表的顺序一致；"
             "dshow 会整体错位，用它识别可能把标签贴到别的相机上")
    pi.set_defaults(func=cmd_identify)

    pp = sub.add_parser("pin", help="非交互地把某角色固定到某设备")
    pp.add_argument("role", help="角色名，如 left / right / main")
    pp.add_argument("value", help="0 | port:Port_#0001.Hub_#0003 | id:USB\\... | name:<名字片段>")
    pp.set_defaults(func=cmd_pin)

    pig = sub.add_parser("ignore", help="把某台相机标为「无关」（自动挑相机时跳过）")
    pig.add_argument("value", help="0 | port:Port_#0001.Hub_#0003 | id:USB\\... | name:<名字片段>")
    pig.set_defaults(func=cmd_ignore)

    pu = sub.add_parser("unignore", help="取消「无关」标记（单台，或 --all 清空整份清单）")
    pu.add_argument("value", nargs="?", default=None, help="同 ignore 的取值；与 --all 二选一")
    pu.add_argument("--all", action="store_true", help="清空整份「无关」清单")
    pu.set_defaults(func=cmd_unignore)

    pg = sub.add_parser("ignored", help="列出当前「无关」清单（并标注在不在线）")
    pg.set_defaults(func=cmd_ignored)

    pc = sub.add_parser("check", help="比对配置与当前设备：有没有漂移/被占用")
    pc.set_defaults(func=cmd_check)

    ppr = sub.add_parser(
        "probe", help="真的开一次：逐后端 × 逐编号报「能不能出图」（只诊断，不绑角色）")
    ppr.add_argument("--max-index", type=int, default=3, help="探测到第几个编号（默认 3）")
    ppr.add_argument("--backends", choices=("both", "msmf", "dshow"), default="both",
                     help="探测哪些后端（默认 both）")
    ppr.set_defaults(func=cmd_probe)

    pd = sub.add_parser(
        "doctor", help="**真开相机**体检一遍：现在能不能跑双目（只报告，不改配置）")
    pd.add_argument("--width", type=int, default=video_profiles.DEFAULT.width,
                    help=f"目标宽度（默认档 {video_profiles.DEFAULT.label}）")
    pd.add_argument("--height", type=int, default=video_profiles.DEFAULT.height,
                    help=f"目标高度（默认档 {video_profiles.DEFAULT.label}）")
    pd.add_argument(
        "--pair-timeout", type=positive_seconds, default=PAIR_FIRST_FRAME_S,
        help=f"双路同时取帧检查里每台等首帧的上限（秒，默认 {PAIR_FIRST_FRAME_S:.0f}）")
    pd.add_argument(
        "--pair-frames", type=int, default=PAIR_FRAMES,
        help=f"双路同时取帧检查里每台读多少帧（默认 {PAIR_FRAMES}）")
    pd.set_defaults(func=cmd_doctor)
    return p


def main() -> int:
    quiet_repeated_grab_warnings()
    p = build_parser()
    args = p.parse_args()
    if not getattr(args, "cmd", None):
        p.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
