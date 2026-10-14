"""视频源：摄像头 / 视频文件 / 单张图片。

Windows 上打开摄像头的坑都在这里处理掉了：
  - 后端优先级 DSHOW -> MSMF -> 默认；
  - 设置 MJPG 才能拿到高分辨率高帧率（很多摄像头默认 YUY2 只能 5-10 fps）；
  - BUFFERSIZE=1 降低延迟（否则你看到的是几百毫秒前的画面）。
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

WINDOWS_BACKENDS = [
    ("DSHOW", cv2.CAP_DSHOW),
    ("MSMF", cv2.CAP_MSMF),
    ("ANY", cv2.CAP_ANY),
]

# 「这一帧是不是全黑/全平」的判据 —— **全项目唯一的答案在这里**。
#
# 语义：帧的灰度标准差 ≤ 这个数，就认为它没有可用信息（拔线、驱动刚起来、
# 曝光还没稳定时会是这种帧），此时**不当作有效帧**。
#
# 为什么要有这个常量：这个 `1.0` 原本在**两个层、6 处**各写了一遍
# （`sources.py` 5 处 + `apps/tools/camera_quality.py` 1 处）。
# 改一处忘一处不会有任何提示，而它决定了"这帧算不算数"。
BLACK_FRAME_STD = 1.0

# ---------------------------------------------------------------------------
# 「用编号开相机」必须配哪个后端 —— 全项目唯一的答案在这里
# ---------------------------------------------------------------------------
#
# 背景：`camera_registry.list_devices()` 给出的编号来自 **Windows PnP 枚举**，
# 而 `cv2.VideoCapture(index, backend)` 用的是**各后端自己的枚举**。两者不是同一套。
#
# 本机实测（同一时刻、同一批设备；用户盖住了内置相机，它在哪一路全黑就是哪一路）：
#
#     PnP   : 0=aoni  1=aoni  2=内置(黑)  3=内置红外
#     DSHOW : 0=aoni  1=内置(黑)  2=aoni  3=打不开     ← 与 PnP **错位**
#     MSMF  : 0=aoni  1=aoni  2=内置(黑)  3=打不开     ← 与 PnP **一致**
#
# 所以：**凡是拿 list_devices() / 角色解析出来的编号去开相机，都必须用 MSMF**，
# 否则编号指向的是另一台相机 —— 而且是静默的：预览窗口有画面、标签有名字，
# 只是名字和画面对不上。踩过两次：
#   1. 标定把 `right` 开成了笔记本内置（用户盖住了它，画面全黑，像相机坏了）；
#   2. `cameras.py identify` 里用户看着 A 的画面按 r，配置记到了 B 上。
#
# 用编号的地方都要引用这个常量，不要再各写各的 —— 写两处就会漂移成两种行为。
INDEX_BACKEND_NAME = "MSMF"
INDEX_BACKEND = cv2.CAP_MSMF


def backend_name(api: int | None) -> str:
    """把 cv2 后端常量翻译成名字（打印/日志用）。"""
    for name, value in WINDOWS_BACKENDS:
        if value == api:
            return name
    return f"backend#{api}"


def quiet_repeated_grab_warnings() -> None:
    """把 OpenCV 那条**每 10 秒刷一次**的抓帧警告压下去。

    ## 为什么必须压

    相机掉线时 MSMF 会反复打印：

        [ WARN:0@48.612] global cap_msmf.cpp:1816 CvCapture_MSMF::grabFrame
        videoio(MSMF): can't grab frame. Error: -2147483638

    `-2147483638` = `MF_E_VIDEO_RECORDING_DEVICE_INVALIDATED`（设备被别的程序
    抢走 / USB 掉线）。实测它**每 10 秒一条、无限刷**，把真正有用的信息
    （收到第几对、为什么拒收）淹掉。用户来问"日志里全是这条"就是这个。

    ## 为什么只压这一条，而不是整体静音

    整体调高 `OPENCV_LOG_LEVEL` 会把**别的**警告一起吞掉 ——
    而"报错不要吞"是硬规矩。所以按**模块**分别设等级：
    只把 `videoio` 调到 ERROR（丢弃它自己的 WARN），其余模块保持 WARNING。

    ⚠️ 必须在**任何 `cv2.VideoCapture` 之前**调用（OpenCV 只在初始化时读一次）。
    """
    for mod in ("videoio", "videoio_msmf"):
        os.environ.setdefault(f"OPENCV_LOG_LEVEL_{mod}", "ERROR")


@dataclass
class SourceInfo:
    kind: str
    width: int
    height: int
    fps: float
    backend: str = ""
    path: str = ""


class VideoSource:
    """统一的帧源接口：read() -> (ok, frame, timestamp)。"""

    info: SourceInfo

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        raise NotImplementedError

    def release(self) -> None:
        pass

    @property
    def is_live(self) -> bool:
        return False


# 自动模式探测用的候选分辨率。
# 为什么需要这个：USB 摄像头的"支持分辨率列表"和"实际帧率"关系很反直觉 ——
# 实测 aoni webcam A25：请求 1280x720 只有 10 fps，请求 1024x768 反而拿到
# 1280x720@30fps（DSHOW 在不同请求下选了不同的内部模式）；
# 1920x1080 更是只有 4.3 fps（YUY2 未压缩，USB 带宽不够）。
# 不探测就只能靠猜。
CANDIDATE_MODES: tuple[tuple[int, int], ...] = (
    (1920, 1080), (1280, 720), (1024, 768), (960, 540), (800, 600), (640, 480),
)
_MODE_CACHE = Path(__file__).resolve().parents[1] / "config" / "camera_modes.json"


def _load_mode_cache() -> dict:
    try:
        return json.loads(_MODE_CACHE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_mode_cache(cache: dict) -> None:
    try:
        _MODE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _MODE_CACHE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except OSError:
        pass


class LatestFrameCamera:
    """后台线程持续采集，主循环永远拿"最新一帧"。

    为什么必须这样：在一个线程里顺序读两路摄像头，每读一路都要**等**它的下一帧
    （1080p30 就是 33 ms），两路串起来 66 ms —— 主循环再快也被拖到 15 fps。
    更要命的是，如果处理速度跟不上采集，帧会在驱动缓冲里越积越多，
    你看到的是几百毫秒前的画面（标定时表现为"明明停稳了却总说在移动"）。

    这里让采集在自己的线程里跑，主循环用 latest() 取当前帧：
    处理得完就一帧不落，处理不完就自动丢掉旧帧 —— 永远看最新画面。
    """

    def __init__(self, source: CameraSource) -> None:
        self.source = source
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0
        self._stop = False
        self.dropped = 0
        # ── 采集健康度：为什么必须有这两个计数器 ──────────────────────────
        #
        # 改之前这里是个**纯静默重试**：`except Exception: sleep(0.05); continue`。
        # 后果（2026-10-10 审查定性，本文件最严重的一条）：
        # 相机被拔掉 / USB 掉带宽 / 驱动崩掉之后，采集线程**每 50 ms 静默重试到进程结束**，
        # `self._seq` 不再增长，而主循环照常从 `latest()` 拿到**冻结的旧帧**继续跑 ——
        # 跟踪、预测、HUD 全都在动，操作员只看到画面卡住，**程序不报错、不退出、日志无一行**。
        # 输出的三维位置是静止的，而人会以为"球没动"。
        #
        # 现在这两个计数器把"采集出问题了"变成**可观测**的：
        #   `read_errors`  连续读失败次数（成功一次就清零）
        #   `last_ok_t`    最后一次成功取到帧的时刻（time.monotonic）
        # 调用方拿 `stalled()` 一问就知道：**是"球没动"还是"相机没了"**。
        self.read_errors = 0
        self.last_ok_t = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="camera-reader")
        self._thread.start()

    def _run(self) -> None:
        while not self._stop:
            try:
                ok, frame, _t = self.source.read()
            except Exception:
                self.read_errors += 1
                time.sleep(0.05)
                continue
            if not ok or frame is None:
                # 读「成功但没帧」也算一次失败：拔线之后 `read()` 往往返回 (False, None)
                # 而不是抛异常，两条路径都要计入，否则拔线仍然看不出来。
                self.read_errors += 1
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame = frame
                self._seq += 1
                self.read_errors = 0
                self.last_ok_t = time.monotonic()

    def latest(self, last_seq: int = -1):
        """返回 (seq, frame)。frame 可能和上次相同（采集还没出新帧）。"""
        with self._lock:
            return self._seq, self._frame

    def stalled(self, timeout: float = 1.0) -> bool:
        """采集是否已经"停住"：连续读失败、或超过 `timeout` 秒没有新帧。

        调用方用这个区分**"球没动"与"相机没了"** —— 在改之前这两者长得一模一样。
        """
        return self.read_errors > 0 and (time.monotonic() - self.last_ok_t) > timeout

    @property
    def info(self):
        return self.source.info

    def wait_first(self, timeout: float = 10.0):
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            seq, frame = self.latest()
            if frame is not None:
                return frame
            time.sleep(0.02)
        return None

    def stop(self) -> None:
        self._stop = True
        if self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self.source.release()


class CameraSource(VideoSource):
    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 60,
        fourcc: str = "MJPG",
        backend: int | None = None,
        optimize: bool = True,
        target_fps: float = 25.0,
    ) -> None:
        self.cap = None
        self.backend_name = ""
        self.resolution_fallback = False
        self._index, self._api = index, backend
        self._fps_req, self._fourcc = fps, fourcc
        self.mode_probed = False
        candidates = (
            [(backend_name(backend), backend)] if backend is not None
            else [(n, b) for n, b in WINDOWS_BACKENDS]
        )
        # 分辨率候选：先试用户要的，失败再降级。
        # 为什么必须降级：不少 USB 摄像头对不支持的分辨率不会报错，
        # 而是"打开成功但每帧都返回全黑"，代码不检查就会一直拿到黑屏。
        res_candidates: list[tuple[int, int]] = []
        for cand in ((width, height), (1280, 720), (640, 480), (0, 0)):
            if cand not in res_candidates:
                res_candidates.append(cand)

        self._cached_hit = False
        # 先看缓存：上次探测出的"后端 + 分辨率"直接拿来打开，
        # 省掉整个后端/分辨率搜索（MSMF 每次打开要 7 秒，不能白白重开）。
        # 指定后端时缓存**只在其后端一致时才用** —— 否则会把用户指定的后端
        # 覆盖成别的（双目两路混用不同后端会直接抓不到帧）。
        if optimize:
            cached = _load_mode_cache().get(str(index))
            cached_api = (getattr(cv2, f"CAP_{cached['backend']}", None)
                          if cached and cached.get("backend") else None)
            # v2 之前的缓存存的是"实际分辨率"而不是"请求分辨率"，会导致重开掉速，一律作废
            if cached and cached.get("v") == 2 and cached_api is not None \
                    and (backend is None or cached_api == backend):
                api = cached_api
                # ⚠️ **这条路也可能要等几十秒**（实测同一型号两台相机：
                # 0.63s vs 9.7s，负载高时 40.7s）。缓存命中**不等于快**，
                # 所以这里必须先说"我要开始开了、用的是记住的模式"，
                # 否则用户点完按钮之后是一片死寂 —— 和卡死没有区别。
                print(f"[相机 {index}] 用上次记住的模式打开："
                      f"{cached['backend']} {cached.get('width')}x{cached.get('height')}"
                      f"（已记住，不必重新搜索；但**MSMF 本身可能就要十几秒**）",
                      flush=True)
                t_c = time.perf_counter()
                cap = self._raw_open(api, int(cached.get("width", width)),
                                     int(cached.get("height", height)))
                if cap is not None:
                    self.cap = cap
                    self.backend_name = str(cached["backend"])
                    self._api = api
                    self._cached_hit = True
                    print(f"[相机 {index}] ✔ 打开完成（{time.perf_counter() - t_c:.1f}s）",
                          flush=True)
                else:
                    print(f"[相机 {index}] ✘ 记住的模式这次打不开"
                          f"（{time.perf_counter() - t_c:.1f}s），改为逐个尝试",
                          flush=True)

        errors: list[str] = []
        # ---- 打开过程必须**可见**（2026-10-14 修的真缺陷）----
        #
        # 下面这个双层循环是 `for 后端 × for 分辨率`，而 MSMF 实测**单次打开要 22.9 秒**
        # （DSHOW 0.7 秒）。全程一个字都不打印 —— 用户看到的就是"卡死了"，
        # 于是去点关闭，或者以为程序坏了。
        #
        # **"慢"和"卡死"在用户眼里是同一件事，除非程序说出来它在干什么。**
        # 所以每一步都报：现在试哪个后端、哪个分辨率、已经等了多久。
        # 用的是**同一个计时基准**，这样"已等 Xs"是可比的。
        t_start = time.perf_counter()

        def _log(msg: str) -> None:
            took = time.perf_counter() - t_start
            # 刚开始那几毫秒报"已等 0.0s"是噪音；只在真的等了一会儿之后才报时长
            suffix = f"（已等 {took:.1f}s）" if took >= 1.0 else ""
            print(f"[相机 {index}] {msg}{suffix}", flush=True)

        if self.cap is None and len(candidates) * len(res_candidates) > 1:
            _log(f"开始逐个尝试打开：{len(candidates)} 个后端 × "
                 f"{len(res_candidates)} 个分辨率。"
                 f"**MSMF 后端单次可能要 20 秒以上，请耐心等**")
        # ---- **总时间预算**（2026-10-14 加）----
        #
        # 实测：相机出不了帧时，MSMF 的**每一次** `cap.read()` 都要 **10 秒**才返回失败
        # （内部等 MF 的抓帧超时）。于是 12 次读 = 120 秒，4 个分辨率 = **8 分钟以上**，
        # 再乘后端数。用户看到的就是"点了运行，然后什么都没有"。
        #
        # 搜索本身是**为了更好**，不是为了**卡住**。给它一个上限：
        # 到点就带着"已经试过哪些、为什么都不行"收场，让上层报错。
        # 60 秒足够正常相机试完（正常打开 1~2 秒），也不会让人干等。
        search_budget_s = 60.0
        budget_hit = False
        # 第一档就**一帧都没有** -> 剩下的分辨率不必试。
        # 分辨率不对的表现是"能出帧但全黑"，而"一帧都读不到"是**设备层面**的问题
        # （掉线/被独占/后端不工作），换个分辨率救不回来。
        # 实测证据：MSMF 对这台相机每个分辨率都要 10 秒才失败，4 档 = 40 秒纯浪费。
        no_frame_at_all = False
        for name, api in ([] if self.cap is not None else candidates):
            for rw, rh in res_candidates:
                if no_frame_at_all:
                    errors.append(f"{name}: 第一档就一帧都读不到，跳过其余分辨率")
                    _log(f"⏭ {name} 一帧都读不到（设备层面问题），"
                         f"不再试其它分辨率")
                    break
                if time.perf_counter() - t_start > search_budget_s:
                    budget_hit = True
                    errors.append(
                        f"搜索超时（已试 {search_budget_s:.0f}s）：相机持续给不出画面，"
                        f"不再继续试剩下的组合")
                    _log(f"⏱ 已超 {search_budget_s:.0f}s，停止搜索"
                         f"（相机一直给不出画面，继续试只是让人干等）")
                    break
                _log(f"试 {name} @ {rw or '原始'}x{rh or '原始'}…")
                cap = cv2.VideoCapture(index, api)
                if not cap.isOpened():
                    cap.release()
                    errors.append(f"{name}: 打不开")
                    _log(f"{name} 整个后端打不开，换下一个")
                    break                      # 该后端整体不可用，换下一个后端
                if fourcc:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*fourcc))
                if rw and rh:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, rw)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, rh)
                cap.set(cv2.CAP_PROP_FPS, fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                frame = None
                # ⚠️ **每次读也要有上限**：正常情况下 `cap.read()` 是**毫秒级**，
                # 而设备不响应时 MSMF 一次要 **10 秒**才返回失败。
                # 12 次 × 10 秒 = 120 秒，光"等它自己承认失败"就两分钟。
                # 3 秒足够正常相机出首帧（实测 1~2 秒），超出就是设备不给帧。
                t_read = time.perf_counter()
                for _ in range(12):
                    if time.perf_counter() - t_read > 3.0:
                        break                  # 这个组合不值得再等
                    ret, f = cap.read()
                    if ret and f is not None:
                        frame = f
                        # 真实传感器一定有噪声：整帧完全不变 = 这条流是假的
                        if float(f.std()) > BLACK_FRAME_STD:
                            break
                    else:
                        time.sleep(0.05)
                if frame is not None and float(frame.std()) > BLACK_FRAME_STD:
                    self.cap = cap
                    self.backend_name = name
                    self.resolution_fallback = bool(rw and (rw, rh) != (width, height))
                    got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    _log(f"✔ {name} @ {rw or '原始'}x{rh or '原始'} 打开成功"
                         f"（实际 {got_w}x{got_h}）")
                    if self.resolution_fallback:
                        print(
                            f"[提示] 摄像头 {index} 不支持 {width}x{height}（该分辨率下输出全黑），"
                            f"已自动降到 {got_w}x{got_h}。"
                        )
                    break
                cap.release()
                why = "读不到画面" if frame is None else "输出全黑（分辨率不支持）"
                errors.append(f"{name} @ {rw or '默认'}x{rh or ''}: {why}")
                _log(f"✘ {name} @ {rw or '原始'}x{rh or '原始'}：{why}")
                if frame is None:
                    # 12 次读里**一次都没拿到帧**（不是"拿到但全黑"）——
                    # 这是设备层面不给帧，换分辨率没用。见上面 `no_frame_at_all`。
                    no_frame_at_all = True
            if self.cap is not None:
                break
            # ⚠️ **这里刻意不做"自动换后端"**（2026-10-14 实测后撤回）。
            #
            # 第一版我让它自动换：MSMF 出不了帧就改用 DSHOW。**结果更糟** ——
            # 换完之后 `optimize_mode` 在**混合后端的进程**里测量，
            # 相机变成 0.0 fps，于是又去做那个几十秒的模式搜索，
            # 最后把分辨率降到 1280x720 并写进缓存。
            # 而**单独测同一台相机的 MSMF 是 10/10 帧全好的** —— 是这次"回退"弄坏的。
            #
            # 根本原因在代码里早有记录（`stereo/pair_source.py`）：
            # **同一进程里混用 DSHOW + MSMF，MSMF 那路会"打开成功但抓不到帧"**。
            # 所以"自动换后端"是在亲手制造那个已知的坏状态。
            #
            # 正确做法：**说清怎么办，让人显式选**。用户可以接受一次重新指定，
            # 而"自动换完偷偷降分辨率"他完全看不见。
            if no_frame_at_all and backend is not None:
                others = "、".join(n for n, a in WINDOWS_BACKENDS if a != backend)
                raise RuntimeError(
                    f"摄像头 {index} 用 {backend_name(backend)} 能打开，"
                    f"但**一帧都拿不到**（读 12 次全空）。\n"
                    f"  ⚠ 本程序**不会自动换后端** —— 同一进程里混用 DSHOW + MSMF\n"
                    f"     会让 MSMF 那路「打开成功但抓不到帧」（本项目已实测过），\n"
                    f"     所以自动切换会把好的一路也弄坏。\n"
                    f"  怎么办（按顺序试）：\n"
                    f"    ① **拔掉相机再插回** —— 报 -2147483638 时这一步最有效；\n"
                    f"    ② 确认没有别的程序占着它（会议软件 / 浏览器 / Windows Hello）；\n"
                    f"    ③ 换一个直接插在机身上的 USB 口；\n"
                    f"    ④ 仍不行就换后端 —— **两路必须同后端**"
                    f"（只换一路，双目直接取不到帧）：\n"
                    f"       `--backend {others.split('、')[0].lower()}`"
                )
        if self.cap is None:
            raise RuntimeError(
                f"无法打开摄像头 {index}，已尝试全部后端与分辨率。\n  " + "\n  ".join(errors) +
                "\n可能原因：\n"
                "  1) 被别的程序占用（QQ / 微信 / 腾讯会议 / 浏览器 / Windows Hello）；\n"
                "  2) 是红外摄像头（IR，无彩色输出）；\n"
                "  3) 镜头上有物理遮挡（隐私滑块 / 盖子合上）；\n"
                "  4) USB 供电或带宽不足（换一个直接插在机身上的 USB 口试试）。"
            )
        # ---- 指定后端但一帧都拿不到：**报错，不自动换后端**（2026-10-14 实测后撤回）----
        #
        # 实测现象：MSMF 对相机 **打开成功**（有时要 60 秒才返回），
        # 而之后 **一帧都读不到**，只有每 10 秒一条
        # `can't grab frame. Error: -2147483638`（= MF_E_VIDEO_RECORDING_DEVICE_INVALIDATED）。
        #
        # ⚠️ 我第一版让它**自动改用 DSHOW**，结果**更糟**：
        #   · 换完之后 `optimize_mode` 在**混合后端的进程**里测量，读到 0.0 fps；
        #   · 于是又去做几十秒的模式搜索，最后把分辨率降到 1280x720 写进缓存；
        #   · 而**单独测同一台相机的 MSMF 是 10/10 帧全好的** —— 是这次回退弄坏的。
        #
        # 根因在 `stereo/pair_source.py` 早有记录：
        # **同一进程里混用 DSHOW + MSMF，MSMF 那路会"打开成功但抓不到帧"**。
        # 所以"自动换后端"是在亲手制造那个已知的坏状态。
        #
        # 正确做法：**抛错、说清怎么办、让人显式选** —— 用户能接受一次重新指定，
        # 而"自动换完偷偷降分辨率"他完全看不见。
        #
        # 真正的失败收场（跳过其余分辨率、换后端的说明）在上面的搜索循环里，
        # 那里才有 `no_frame_at_all` 这个状态。
        if optimize and not self._cached_hit:
            self.optimize_mode(target_fps=target_fps)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        real_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.fps_measured = False
        # CAP_PROP_FPS 在很多摄像头上返回的是"你请求的值"而不是真实帧率
        # （实测请求 60、实际 29.8）。用它去写录像会让播放速度错一倍，
        # 所以优先用探测/缓存里实测到的值。
        if getattr(self, "mode_fps", None):
            real_fps = float(self.mode_fps)
            self.fps_measured = True
        elif self._cached_hit:
            _c = _load_mode_cache().get(str(index), {})
            if _c.get("fps"):
                real_fps = float(_c["fps"])
                self.fps_measured = True
        if not real_fps or real_fps <= 1.0 or real_fps != real_fps:
            # 很多 USB 摄像头（尤其是走 DSHOW 的）CAP_PROP_FPS 会返回 -1 或 0，
            # 直接拿这个值会让录像文件帧率变成 -1 -> 写出来是坏文件。
            # 这里实测一段时间的真实帧率兜底。
            real_fps = self.measure_fps(duration=1.0, warmup=3)
            self.fps_measured = True
        self.info = SourceInfo("camera", w, h, real_fps, self.backend_name, str(index))
        self._interval = 1.0 / real_fps if real_fps and real_fps > 1 else 0.0
        self._emit_interval = 0.0
        self._next_t = 0.0

    def measure_fps(self, duration: float = 1.0, warmup: int = 3) -> float:
        """实测摄像头真实帧率（读若干帧计时）。

        注意：暗光下摄像头会自动拉长曝光时间，帧率可能掉到个位数 ——
        这时的低帧率是光照问题，不是代码问题。
        """
        if self.cap is None:
            return 30.0
        for _ in range(warmup):
            self.cap.read()
        count = 0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < duration:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                count += 1
            else:
                break
        elapsed = time.perf_counter() - t0
        if count < 2 or elapsed <= 0:
            return 30.0
        fps = count / elapsed
        return float(fps) if fps > 1.0 else 30.0

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        assert self.cap is not None
        ret, frame = self.cap.read()
        t = time.perf_counter()
        if not ret or frame is None:
            return False, None, t
        return True, frame, t

    # ------------------------------------------------------------------
    def _open_attempt(self, width: int, height: int, api: int | None = None):
        """按给定分辨率重开一次，返回可用的 cap（画面非全黑）或 None。"""
        use_api = api if api is not None else (self._api if self._api is not None else cv2.CAP_ANY)
        cap = cv2.VideoCapture(self._index, use_api)
        if not cap.isOpened():
            cap.release()
            return None
        if self._fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*self._fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, self._fps_req)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        frame = None
        for _ in range(12):
            ret, f = cap.read()
            if ret and f is not None:
                frame = f
                if float(f.std()) > BLACK_FRAME_STD:
                    break
            else:
                time.sleep(0.03)
        if frame is None or float(frame.std()) <= BLACK_FRAME_STD:
            cap.release()
            return None
        return cap

    @staticmethod
    def _measure_rate(cap, budget: float = 0.25, warmup: int = 2) -> float:
        for _ in range(warmup):
            cap.read()
        n = 0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < budget:
            ok, _f = cap.read()
            if ok:
                n += 1
        dt = max(time.perf_counter() - t0, 1e-6)
        return n / dt


    def _raw_open(self, api: int, width: int, height: int):
        """探测用的快速打开：只等首帧，不做全黑多帧检查。

        ⚠️ **这里也会慢，而且慢得很隐蔽**（2026-10-14 实测）：
        缓存命中时走的就是这条路径，看上去"只是开一下"。但实测**同一型号的两台相机**
        在 MSMF 下差别巨大：

            相机 0：0.63 ~ 2.0 s
            相机 1：9.7 s（负载高时到过 40.7 s）

        所以它不能是静默的 —— 用户点「标定前置检查」之后，
        **两路都开好之前一个画面都没有**（实测累计 44.9 s），
        而屏幕上什么都不说。那与"卡死"在用户眼里没有区别。

        调用方（`__init__` 的缓存分支 / `optimize_mode`）负责打印上下文，
        这里只保证**打开动作本身**可被观测：先报"开始打开"，成功/失败各报一次。
        """
        t0 = time.perf_counter()
        cap = cv2.VideoCapture(self._index, api)
        if not cap.isOpened():
            cap.release()
            return None
        if self._fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*self._fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, self._fps_req)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(3):
            ret, f = cap.read()
            if ret and f is not None and float(f.std()) > BLACK_FRAME_STD:
                took = time.perf_counter() - t0
                # 只有**真的等了一会儿**才值得报一句（<2s 属于正常，报了是噪音）
                if took >= 2.0:
                    print(f"[相机 {self._index}] 打开用了 {took:.1f}s（{backend_name(api)} "
                          f"请求 {width}x{height}）", flush=True)
                return cap
        cap.release()
        return None

    def optimize_mode(self, target_fps: float = 25.0, budget: float = 0.25) -> None:
        """自动挑出"这个摄像头跑得最顺"的后端与分辨率，并缓存结果。

        为什么要挑后端：实测同一台 USB 摄像头
            DSHOW  打开 0.7 s，1080p 只有 4.3 fps
            MSMF   打开 6.8 s，1080p 有 32.5 fps
        差别是 7 倍。但 MSMF 打开很慢，所以绝不能每个分辨率都开一次 ——
        **先只比后端（2 次打开），只有最好的后端还达不到目标帧率时才去试分辨率。**
        结果缓存到 config/camera_modes.json，之后启动直接用它打开。
        """
        cache = _load_mode_cache()
        key = str(self._index)
        api = self._api if self._api is not None else cv2.CAP_DSHOW
        name = self.backend_name or "DSHOW"
        cw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ch = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cur_rate = self._measure_rate(self.cap, budget)

        # ⚠️ **刚打开时帧率可能测成 0.0，必须等它稳下来再判**（2026-10-14 修）
        #
        # 实测（相机 1，MSMF）：打开成功后立刻测，得到 **0.0 fps**，
        # 而同一台相机**缓存里记的是 25.08 fps**。于是这里误判"太慢"，
        # 触发下面那个**每个分辨率都要重开 30 秒**的搜索 ——
        # 5 个候选 = **150 秒纯浪费**，最后还绕回一个更差的模式。
        #
        # 刚打开的相机还没开始出帧（或首帧还没到）时，测出来必然是 0 ——
        # 那是**测量时机的问题，不是设备慢**。所以先等首帧、再测；
        # 仍为 0 就再等一会儿重测，最多 ~6 秒。**"测不准"不能当成"设备不行"。**
        if cur_rate < 0.1:
            t_wait = time.perf_counter()
            while time.perf_counter() - t_wait < 6.0:
                time.sleep(0.25)
                cur_rate = self._measure_rate(self.cap, 0.4)
                if cur_rate >= 0.1:
                    break
            if cur_rate < 0.1:
                print(f"[相机 {self._index}] 等了 6 秒仍测不到帧"
                      f"（相机可能没真的在出帧）—— 按 0 fps 处理", flush=True)
            else:
                print(f"[相机 {self._index}] 刚打开时测不到帧，等了 "
                      f"{time.perf_counter() - t_wait:.1f}s 后测到 {cur_rate:.1f} fps",
                      flush=True)

        # 快路径：已经够快就什么都不做（内置摄像头 1080p 有 29 fps，走这条）。
        # 但仍要把实测帧率记下来 —— CAP_PROP_FPS 会返回"请求值 60"，
        # 拿它去写录像会让播放速度错一倍。
        if cur_rate >= target_fps * 0.9:
            self.mode_fps = cur_rate
            self.mode_probed = True
            cache[key] = {"v": 2, "backend": name, "width": cw, "height": ch,
                          "actual_width": cw, "actual_height": ch, "fps": round(cur_rate, 2)}
            _save_mode_cache(cache)
            return

        # 真的慢（不是"没测到"）才走搜索。而且**只要缓存里已经有一个可用的模式，
        # 就先别搜** —— 缓存的模式是"上次实测够快"才写进去的，重新搜一轮
        # 每个候选都要 30 秒，代价远大于收益。
        cached = cache.get(key)
        if cached and cached.get("v") == 2 and cached.get("fps", 0) >= target_fps * 0.9:
            print(f"[相机 {self._index}] 当前 {cw}x{ch} 只有 {cur_rate:.1f} fps，"
                  f"但缓存里有上次实测 {cached.get('fps')} fps 的模式"
                  f"（{cached.get('backend')} {cached.get('width')}x{cached.get('height')}）——"
                  f"**沿用缓存，不做昂贵搜索**", flush=True)
            self.cap.release()
            cap2 = self._raw_open(api, int(cached["width"]), int(cached["height"]))
            if cap2 is not None:
                self.cap = cap2
                self.backend_name = str(cached["backend"])
                self._api = api
                self.mode_fps = float(cached.get("fps") or 0.0)
                self.mode_probed = True
                return
            print(f"[相机 {self._index}] 缓存的模式这次打不开，改为搜索", flush=True)

        self.cap.release()
        self.cap = None
        print(f"[相机 {self._index}] 当前 {cw}x{ch} 只有 {cur_rate:.1f} fps，"
              f"正在寻找更合适的模式（约 10-40 秒，只需一次）…", flush=True)
        # best 里存的是**请求值**（rw, rh）—— 必须用它重开。
        # 真实例：请求 1024x768 能得到 1280x720@29fps，
        # 但按"实际值 1280x720"重开只有 10 fps（DSHOW 选了不同的内部模式）。
        best = (name, api, cw, ch, cw, ch, cur_rate)
        # 同一后端下逐个分辨率试。DSHOW 打开只要 0.7 秒，试几个可以接受；
        # MSMF 打开要 7 秒且每次 set 都要重协商，所以**不做后端对比**
        # （实测那样启动要 30-60 秒，比卡顿更难受）。
        for (rw, rh) in CANDIDATE_MODES:
            if (rw, rh) == (cw, ch):
                continue
            cap = self._raw_open(api, rw, rh)
            if cap is None:
                continue
            gw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            gh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            rate = self._measure_rate(cap, budget)
            cap.release()
            print(f"    请求 {rw}x{rh} -> 实际 {gw}x{gh} @ {rate:.1f} fps", flush=True)
            # 评分 = 像素数 x min(帧率, 目标)：既不牺牲流畅度，也不白白降分辨率
            if (gw * gh * min(rate, target_fps)) > (best[4] * best[5] * min(best[6], target_fps)):
                best = (name, api, rw, rh, gw, gh, rate)

        name, api, bw, bh, aw, ah, brate = best
        self.cap = self._open_attempt(bw, bh, api=api)
        if self.cap is None:
            self.cap = self._open_attempt(cw, ch, api=api)
            if self.cap is None:
                raise RuntimeError(f"摄像头 {self._index} 重新打开失败")
            return
        self.mode_probed = True
        self.mode_fps = brate
        self.backend_name = name
        self._api = api
        gw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        gh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 重开之后必须复测一次：请求值 → 实际模式 的映射不是一一对应的，
        # 不复测就可能"报告 29 fps、实际 10 fps"
        verify = self._measure_rate(self.cap, budget)
        if verify < brate * 0.6:
            print(f"[相机 {self._index}] 警告：重开后实测只有 {verify:.1f} fps"
                  f"（探测时 {brate:.1f}），已按实测值记录", flush=True)
            brate = verify
        self.mode_fps = brate
        if (gw, gh) != (cw, ch) or brate > cur_rate * 1.3:
            print(f"[相机 {self._index}] 自动改到 请求{bw}x{bh} -> 实际 {gw}x{gh} "
                  f"@ {brate:.1f} fps（原 {cw}x{ch} @ {cur_rate:.1f} fps）")
        cache[key] = {"v": 2, "backend": name, "width": bw, "height": bh,
                      "actual_width": gw, "actual_height": gh, "fps": round(brate, 2)}
        _save_mode_cache(cache)

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    @property
    def is_live(self) -> bool:
        return True


class FileSource(VideoSource):
    """视频文件。时间戳用文件里的帧号 / fps，保证预测的时间轴正确。"""

    def __init__(self, path: str | Path) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"视频文件不存在：{p}")
        self.cap = cv2.VideoCapture(str(p))
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频：{p}")
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.info = SourceInfo("video", w, h, fps, "file", str(p))
        self.path = p
        self._idx = 0
        self._fps = fps if fps > 0 else 30.0

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        ret, frame = self.cap.read()
        t = self._idx / self._fps
        if ret and frame is not None:
            self._idx += 1
        return ret, frame, t

    def release(self) -> None:
        self.cap.release()


class ImageSource(VideoSource):
    """单张图片（可以喂一个文件夹，当序列处理）。"""

    def __init__(self, paths: Sequence[str | Path], fps: float = 30.0) -> None:
        self.paths = [Path(p) for p in paths]
        if not self.paths:
            raise ValueError("没有可用的图片")
        first = cv2.imread(str(self.paths[0]), cv2.IMREAD_COLOR)
        if first is None:
            raise RuntimeError(f"无法读取图片：{self.paths[0]}")
        h, w = first.shape[:2]
        self.info = SourceInfo("image", w, h, fps, "file", str(self.paths[0]))
        self._fps = fps
        self._idx = 0

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        if self._idx >= len(self.paths):
            return False, None, self._idx / self._fps
        p = self.paths[self._idx]
        frame = cv2.imread(str(p), cv2.IMREAD_COLOR)
        t = self._idx / self._fps
        self._idx += 1
        if frame is None:
            return False, None, t
        return True, frame, t


def open_source(path: str | None = None, camera: int | None = None, **kwargs) -> VideoSource:
    """按路径或摄像头编号打开帧源。"""
    if path:
        p = Path(path)
        if p.is_dir():
            exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
            files = sorted(f for f in p.iterdir() if f.suffix.lower() in exts)
            return ImageSource(files, **kwargs)
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}:
            return ImageSource([p], **kwargs)
        return FileSource(p)
    return CameraSource(index=0 if camera is None else camera, **kwargs)
