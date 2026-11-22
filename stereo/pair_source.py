"""成对帧源：把"左路 + 右路"当成一个整体来取帧。

为什么它属于 `stereo/` 而不是某个 app
------------------------------------
"按同一时刻取到左右两帧"是**双目共用的取帧逻辑**，不是标定专用的：

    双目标定    要用（收板停稳的帧对）
    标定复检    要用（拍几帧板看极线误差变化）
    以后的实时双目 pipeline 也要用

所以它放在库里。以前它是 `apps/calibrate_stereo.py` 里的一个类，
`apps/verify_stereo.py` 靠 `sys.path` 去 import **另一个 app 的内部实现** ——
`calibrate_stereo.py` 一旦改名/搬走，复检工具会在运行时报
`ModuleNotFoundError: calibrate_stereo`，指不到问题根源。

三种帧源（构造时按参数自动判断）
--------------------------------

    PairSource(camera=(0, 1))                       # 两台实时相机
    PairSource(video=("cam_l.mp4", "cam_r.mp4"))    # 两段视频
    PairSource(directory=("imgs/left", "imgs/right"))  # 两个图片文件夹

关于同步偏移 `sync_offset`
-------------------------
两台独立相机拍到的帧**不是同一时刻**的。`stereo/sync.py` 能算出偏移，但算出来
之后必须有人**按偏移取帧**，否则三角测量会把"左相机 t1 的球"和"右相机 t2 的球"
当成同一个点，算出来的三维坐标错得很隐蔽。这个类就是那个"有人"：

    offset = find_offset(sl, sr).offset          # 左路第 i 帧 对应 右路第 i+offset 帧
    src = PairSource(video=(l, r), sync_offset=offset)

语义与 `stereo.sync.apply_offset` 完全一致（它就是"取帧"那句话的机器可读版本）：
**返回的左路第 i 帧与右路第 i+offset 帧是同一物理时刻。**

注意
----
- **标定不需要同步**（只收静止帧），所以标定/复检传 `sync_offset=0`（默认）即可；
- **实时相机不支持偏移**：两路各在自己的线程里跑到最新帧，帧号没有共同基准，
  只能靠硬件同步或"等两路都出新帧"降低（而非消除）时间差。传了会给出警告并忽略；
- 图片文件夹逐张配成"时刻"，文件夹内的整段偏移同样支持。

配对策略（2026-10-20 加）
------------------------
- `pairing="index"`：旧行为，左 i 配右 i+sync_offset；
- `pairing="timestamp"`：读 `*_times.csv`，按时间戳一对一配对；掉帧处跳过，
  不再让后续帧整体错位；
- `pairing="canary"`（2026-10-21 转正为默认）：先算时间戳计划；覆盖率 ≥75%
  且残差在容差内才接管，否则回退 index。仍可用 `--pairing index` 强制旧行为。

时间戳配对不改变 `read()` 的返回形状；它只改变“取哪两帧”。

用法::

    from stereo import PairSource

    src = PairSource(directory=("imgs/left", "imgs/right"))
    while True:
        ok, left, right = src.read()
        if not ok:
            break
        ...
    src.release()
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

from sphere_tracker.camera_registry import (
    AUTO_TOKEN,
    RoleSpec,
    load_registry,
    resolve_camera_arg,
    select_cameras,
)
from sphere_tracker.sources import (
    IMAGE_EXTENSIONS,
    CameraSource,
    LatestFrameCamera,
    infer_source_kind,
    open_camera_source,
)
from sphere_tracker.video_profiles import DEFAULT_FPS, probe_order

from .timestamp_pair import (
    canary_safe,
    plan_timestamp_pairs,
    read_time_sidecar,
    sidecar_for_video,
)

CAMERA_WAIT_TIMEOUT = 5.0        # 等相机出新帧的上限（秒），超时就用已有帧继续
CAMERA_POLL_INTERVAL = 0.005     # 轮询间隔，比 30 fps 的帧间隔（33 ms）小得多
# 实时两路各自的采集线程是独立跑的，严格说没有共同帧号。
# 让右路至少前进这么多帧，就认为"两路都已经刷新过"，时间差被压到一帧以内。
# （这是减小时间差，不是消除 —— 真正的同步要靠硬件触发。）
CAMERA_ALIGN_STEPS = 2

#: 并行打开两路相机的**整体**等待上限（秒）。它是"驱动层挂死"的兜底，
#: 不是"正常慢相机"的上限 —— 取足够宽，宁晚判、不误杀。
#: `CameraSource.search_budget_s` 默认 60；而 `_open` 在打不开时会**再试一次**
#: `backup_camera`（见下面 `_open` 里那次回退），故正常路径最坏 ≈ 2×60。
#: 取 150 = 2×60 + 30 余量（首帧等待 + 分辨率回退）。低于它会把一台
#: "正常但慢 + 走了备用相机"的相机误判成超时。
OPEN_JOIN_TIMEOUT_S = 150.0

#: auto（`width<=0` 或 `height<=0`）= 自适应协商的**总**预算（秒）。
#: 逐档探测每档要开两路 + 短测帧率，最坏走完 `probe_order(None)` 的 6 个档
#: （+ 原生档）。取 90 与 `OPEN_JOIN_TIMEOUT_S` 同量级：到点就带着"试过哪些、
#: 为什么都不行"收场，而不是无限期地一档一档等下去。
AUTO_PAIR_DEADLINE_S = 90.0
#: 成对帧率的**下限**：低于它就降档。= `DEFAULT_FPS * 0.6`（当前 14.4）。
#: 为什么必须有（范式审计 A2）：4K 也能"出帧"，只是每秒 2 帧 —— 只判"能不能出图"
#: 会把一个"清晰但卡成幻灯片"的档当成好档。
AUTO_MIN_PAIR_FPS = DEFAULT_FPS * 0.6


def resolve_min_pair_fps(value: float | None = None,
                         settings: dict | None = None) -> float:
    """成对帧率下限：显式值 > `config/cameras.json` 的 `min_pair_fps` > 14.4。

    默认值刻意保持 `AUTO_MIN_PAIR_FPS`（=14.4），旧行为一分不变。配置键为
    `None`/空/非法/<=0 时同样回默认；**不**把 0 解释成"关闭下限" —— 关闭
    下限会让 4K@2fps 这种"清晰但卡成幻灯片"的档被接受，那是已知的坑。
    """
    if value is not None:
        try:
            v = float(value)
        except (TypeError, ValueError):
            v = float("nan")
        if v == v and v > 0 and v != float("inf"):  # 非 nan / 正 / 非 inf
            return v
    if settings is None:
        try:
            settings = load_registry().settings
        except Exception:  # noqa: BLE001 - 读配置失败不该让取帧起不来
            settings = {}
    try:
        raw = (settings or {}).get("min_pair_fps")
        v = float(raw) if raw not in (None, "") else 0.0
    except (TypeError, ValueError):
        v = 0.0
    if v == v and v > 0 and v != float("inf"):
        return v
    return float(AUTO_MIN_PAIR_FPS)


#: 刚打开时测不到帧率（<0.1 fps）的"等首帧再测"上限（秒）。
#: 对齐 `sources.optimize_mode` 的做法：刚打开测成 0 是**测量时机**问题，不是设备慢。
AUTO_FIRST_FRAME_WAIT_S = 6.0
#: 短测成对帧率的采样窗口（秒）。
#:
#: 为什么是 **0.6** 而不是更短的 0.3（2026-10-31 独立 Critic 的中等项）：
#: 帧率 = (窗口内帧数) / 时长，帧数是整数 —— 0.3s 窗口在 13~20 fps 附近的量化
#: 误差约 ±1.7 fps（12 与 20 fps 只差 3 帧）。而帧率下限 `AUTO_MIN_PAIR_FPS`
#: 恰是 14.4，于是"贴线"档的判决会**跨次运行翻转**（真机实测：同一个档一次测成
#: 13.1 被拒、另一次 19.7 被接受）。窗口加倍到 0.6s，量化误差压到约 ±0.8 fps。
#: 代价：每档多 0.3s；最坏走完 7 个档 ≈ 多 2s，相对 `AUTO_PAIR_DEADLINE_S=90` 可忽略。
AUTO_PAIR_FPS_BUDGET = 0.6
#: 从两路各取**一帧真帧**（`wait_first`）的等待上限（秒），用于 A3 的"真帧尺寸"复核。
#: 为什么短：`CameraSource` 在**打开时**就已经成功读到过帧（否则会 raise），
#: 所以后台采集线程的首帧应在一帧间隔内到达；3s 足够，且不啃协商总预算。
AUTO_FRAME_PROBE_TIMEOUT_S = 3.0

_BACKEND_BY_CLI = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}


def _profile_label(rw: int, rh: int) -> str:
    """候选档的显示名（原生档 ``(0, 0)`` 说成"原生"）。"""
    return "原生" if (rw <= 0 or rh <= 0) else f"{rw}x{rh}"


def _wh(size: tuple[int, int]) -> str:
    return f"{size[0]}x{size[1]}"


def _pair_fps_sample(cams: list["LatestFrameCamera"], budget: float) -> float:
    """采样 `budget` 秒，返回**成对**帧率 = min(两路 seq 增量) / 时长。

    为什么取 min：双目要的是"两路都能跟上"的节奏，某台快、另一台慢时，
    成对取帧的实际上限是**慢的那台**（`read()` 会等两路都出新帧）。
    `seq` 是采集线程每存一帧 +1，停止出帧就不再增长 —— 所以这个数如实反映"真在出帧"。
    """
    base = [c.latest()[0] for c in cams]
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < budget:
        time.sleep(0.02)
    dt = max(time.perf_counter() - t0, 1e-6)
    now = [c.latest()[0] for c in cams]
    return min(n - b for n, b in zip(now, base)) / dt


def _measure_pair_fps(
    cams: list["LatestFrameCamera"],
    budget: float = AUTO_PAIR_FPS_BUDGET,
    settle_max: float = AUTO_FIRST_FRAME_WAIT_S,
) -> float:
    """测成对帧率；刚打开测成 0 时**先等首帧再测**（A5），最多 `settle_max` 秒。

    "测不准"不能当成"设备不行"—— 刚打开的相机还没开始出帧时测出来必然是 0，
    那是测量时机的问题（与 `sources.optimize_mode` 同一处理）。
    """
    rate = _pair_fps_sample(cams, budget)
    if rate >= 0.1:
        return rate
    t_wait = time.perf_counter()
    while time.perf_counter() - t_wait < settle_max:
        for c in cams:
            c.wait_first(timeout=0.25)
        rate = _pair_fps_sample(cams, budget)
        if rate >= 0.1:
            return rate
    return rate


def _probe_real_frames(cams: list["LatestFrameCamera"],
                       timeout: float = AUTO_FRAME_PROBE_TIMEOUT_S,
                       ) -> tuple[list[np.ndarray] | None, str]:
    """从两路各取**一帧真帧**，返回 `(帧列表, "")` 或 `(None, 失败原因)`。

    为什么要拿真帧（范式审计 A3，2026-10-31 独立 Critic 定出的高）：
    `cam.info.width/height` 是 `cap.get(CAP_PROP_FRAME_WIDTH/HEIGHT)` 的**上报值**，
    驱动可能上报一套、实际给另一套 —— 本文件 `read()` 里就写着实测
    （内置相机上报 1920x1080、实际给 1280x1080）。A3 的契约是"两路**实际**
    分辨率必须一致"，只比上报值 = 拿**弱证据**下**强结论**：两台上报同为
    1920x1080、真帧一个 1280x1080、一个 1920x1080 时会被**静默接受**一对待
    不一致的相机 —— 而喂给标定器的正是真帧尺寸。
    """
    out: list[np.ndarray] = []
    for i, c in enumerate(cams):
        f = c.wait_first(timeout=timeout)
        if f is None:
            side = ("左", "右")[i] if i < 2 else str(i)
            return None, f"{side}路 {timeout:.0f}s 内取不到真帧"
        out.append(f)
    return out, ""


def _join_all(threads: list[threading.Thread],
              timeout_s: float) -> list[threading.Thread]:
    """逐个 join；返回**超时后仍存活**的线程列表。**不抛异常。**

    用**整体预算（deadline）**而不是"每线程各等 `timeout_s`"：
    `deadline = perf_counter() + timeout_s`，对每个线程
    `th.join(timeout=max(0.0, deadline - perf_counter()))`。

    为什么是整体而不是每线程各自的上限：两路是并行打开的，
    用户感知的等待上限取决于**整体**耗时，不是每路各自的上限 ——
    否则"每路各等 90s"在最坏情况下会变成 180s。

    调用方据此把"主线程被卡死"变成"可控失败"（见 `CameraPairReader.__init__`）。
    """
    deadline = time.perf_counter() + timeout_s
    stuck: list[threading.Thread] = []
    for th in threads:
        th.join(timeout=max(0.0, deadline - time.perf_counter()))
        if th.is_alive():
            stuck.append(th)
    return stuck


def resolve_backend(value: str | None) -> int | None:
    """把命令行 `--backend` 的值翻成 cv2 后端常量；auto/空 = 自动成对挑一个。"""
    if value in (None, "", "auto"):
        return None
    name = str(value).lower()
    if name not in _BACKEND_BY_CLI:
        raise RuntimeError(f"未知后端 {value!r}，可选：auto / dshow / msmf")
    return _BACKEND_BY_CLI[name]


def describe_info(info) -> str:
    """把 SourceInfo 打成人话（报告的 fps 可能不是实测值，见 sources.py 的坑）。"""
    fps = f"{info.fps:.1f} fps" if info.fps and info.fps == info.fps else "帧率未知"
    return f"{info.width}x{info.height} @ {fps}"


# --------------------------------------------------------------------------
# 单路帧源：只做"按序号取帧"，好让同步偏移能用"跳过 / 定位"来表达
# --------------------------------------------------------------------------
class FrameReader(ABC):
    """单路帧源。`read()` 连续取帧，`seek(i)` 定位到第 i 帧。

    约定（成对取帧依赖它，实现者必须遵守）：

    - `seek(i)` 返回的**就是第 i 帧**，`read()` 返回的是下一帧；
    - 帧号从 0 开始，只增不减 —— 倒退就抛 `ValueError`
      （成对取帧永远单调向前，倒退说明偏移的正负号搞反了）；
    - 停在原地（`seek(当前帧)`）直接给手上的帧，不重新解码。
    """

    kind = "?"
    width = 0
    height = 0
    label = ""

    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray | None]:
        """取下一帧。返回 (ok, frame)。"""

    @abstractmethod
    def seek(self, index: int) -> tuple[bool, np.ndarray | None]:
        """定位到第 index 帧并返回它（index >= 当前帧号）。"""

    def release(self) -> None:
        pass


class VideoFrameReader(FrameReader):
    """一段视频。seek 用 `CAP_PROP_POS_FRAMES`，跳多远都不用解码中间那些帧。

    这里的状态比看上去讲究，两个变量各管一件事：

        _cur   当前手上这一帧是第几帧（-1 = 还没有帧）
        _next  解码器下一次 read 会吐出来的帧号

    `read()` 取新帧、`seek(i)` 定位到第 i 帧；两者都保证"返回的帧就是它声明的帧号"。
    `seek()` 允许停在原地（重复要同一帧不重新解码），但**不允许倒退** ——
    成对取帧只会单调往前，真倒退了说明调用方算错了偏移方向。
    """

    kind = "video"

    def __init__(self, path: str | Path) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"视频文件不存在：{p}")
        self.path = p
        self.cap = cv2.VideoCapture(str(p))
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频：{p}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.label = str(p)
        self._frame: np.ndarray | None = None
        self._cur = -1
        self._next = 0

    # --- 内部：把解码器挪到 pos 并读出一帧 -------------------------------
    def _fetch(self, pos: int) -> tuple[bool, np.ndarray | None]:
        """读第 pos 帧（pos 必须 >= _next）。成功后 _cur=pos、_next=pos+1。"""
        if pos > self._next:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            self._next = pos
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self._next = pos          # 这一帧读不到，位置标记别往前挪
            return False, None
        self._frame, self._cur = frame, pos
        self._next = pos + 1
        return True, frame

    def read(self) -> tuple[bool, np.ndarray | None]:
        return self._fetch(self._next)

    def seek(self, index: int) -> tuple[bool, np.ndarray | None]:
        if index < self._cur:
            raise ValueError(
                f"不能倒退：手上是第 {self._cur} 帧，要求回到第 {index} 帧。"
                f"（成对取帧只会往前走，倒退了说明同步偏移的正负号搞反了）")
        if index == self._cur:
            return True, self._frame
        return self._fetch(index)

    def frame_count(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def release(self) -> None:
        self.cap.release()


class ImageDirFrameReader(FrameReader):
    """一个图片文件夹（按文件名排序，一张 = 一个时刻）。

    图片和视频不同：可以**随机访问**，跳帧就是换个文件读，没有代价。
    """

    kind = "dir"

    def __init__(self, directory: str | Path) -> None:
        d = Path(directory)
        if not d.is_dir():
            raise NotADirectoryError(f"不是文件夹：{d}")
        self.dir = d
        self.files = sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not self.files:
            raise RuntimeError(f"文件夹里没有图片（支持 {sorted(IMAGE_EXTENSIONS)}）：{d}")
        first = cv2.imread(str(self.files[0]), cv2.IMREAD_COLOR)
        if first is None:
            raise RuntimeError(f"无法读取图片：{self.files[0]}")
        self.height, self.width = first.shape[:2]
        self.label = str(d)
        self._frame: np.ndarray | None = first
        self._cur = 0                 # 手上这一帧是第几张（-1 = 还没有）

    def read(self) -> tuple[bool, np.ndarray | None]:
        return self.seek(self._cur + 1)

    def seek(self, index: int) -> tuple[bool, np.ndarray | None]:
        if index < 0:
            raise ValueError(f"帧号不能是负数：{index}")
        if index < self._cur:
            raise ValueError(
                f"不能倒退：手上是第 {self._cur} 张，要求回到第 {index} 张。"
                f"（成对取帧只会往前走，倒退了说明同步偏移的正负号搞反了）")
        if index >= len(self.files):
            return False, None
        if index == self._cur:
            return True, self._frame
        frame = cv2.imread(str(self.files[index]), cv2.IMREAD_COLOR)
        if frame is None:
            return False, None
        self._frame, self._cur = frame, index
        return True, frame

    def frame_count(self) -> int:
        return len(self.files)

    def release(self) -> None:
        pass


class CameraPairReader(FrameReader):
    """两台实时相机（各自一个后台采集线程，取"最新帧"）。

    这里**不支持偏移**：两个采集线程各跑各的，帧号之间没有共同的 0 点，
    想"让右路退回去 8 帧"是做不到的（帧早没了）。所以只做一件事 ——
    等两路都出新帧，把时间差压到一帧以内。

    ## 分辨率：显式档 vs 自适应（auto）

    - **显式**给 `width/height`（>0）：按该档成对打开，行为与以前完全一致；
    - **自适应**（`width<=0` 或 `height<=0`）：逐档协商（见 `_negotiate`），
      定案后把 `requested_size="auto"` / `resolved_size` / `actual_sizes` /
      `pair_fps` 暴露给上层（标定脚本要写进 meta）。
    """

    kind = "camera"

    def __init__(
        self,
        left_index: int,
        right_index: int,
        width: int = 1920,
        height: int = 1080,
        fps: float = 60,
        backup_camera: int | None = None,
        backend: int | None = None,
        max_skew_ms: float | None = None,
        min_pair_fps: float | None = None,
        left_role: str = "",
        right_role: str = "",
        capture_api: str = "opencv",
    ) -> None:
        if left_index == right_index:
            raise RuntimeError(f"左右相机不能是同一个索引（都是 {left_index}）")
        self.label = f"camera {left_index} / {right_index}"
        self._left_index, self._right_index = int(left_index), int(right_index)
        self._left_role = str(left_role or "").strip()
        self._right_role = str(right_role or "").strip()
        self._capture_api = str(capture_api or "auto").strip().lower()
        if self._capture_api not in ("auto", "mf", "opencv"):
            raise ValueError(f"capture_api 只能是 auto/mf/opencv，收到 {capture_api!r}")
        # **不截断成整数**：29.97 被 int() 截成 29 后，录 1 分钟就少 1.8 帧，
        # 而且相机侧收到的请求也从 29.97 变成 29 —— 错得很安静。
        self._fps = float(fps)
        self._backup_camera = backup_camera
        # auto（含 0）：宽或高任一边 <= 0 就进自适应协商；否则按该显式档打开。
        self._auto = (width <= 0 or height <= 0)
        self.requested_size: str | tuple[int, int] = (
            "auto" if self._auto else (int(width), int(height)))
        self.resolved_size: tuple[int, int] | None = None
        self.actual_sizes: list[tuple[int, int]] = []
        self.pair_fps: float | None = None

        # ---- 双目实时用：时间差（2026-11-16，B-191）------------------------
        # 没有硬件触发线时，两路**不可能**同时曝光：这里只能"把时间差压到一帧
        # 以内"，不能消除它（见本类开头 :399-401）。所以上层需要的不是"保证同步"，
        # 而是"这一对差了多少毫秒" —— 有它才能决定这一帧的三维点可不可信。
        self.max_skew_ms = (None if max_skew_ms is None
                            else float(max_skew_ms))
        # auto 协商的帧率下限；None -> 读 config/cameras.json，再退 14.4。
        self.min_pair_fps = resolve_min_pair_fps(min_pair_fps)
        #: 最近一对帧的时间差（ms，正数 = 右路更晚）；只由 `read_pair()` 写。
        self.last_skew_ms: float | None = None
        #: `read_pair()` 因时间差超限而丢弃的**对数**（不是帧数）。
        self.dropped_pairs: int = 0

        # 两路必须用同一个后端，且要**成对地**挑：一个后端能同时打开两路才算数。
        # 原因（2026-10 实测）：同一进程里混用 DSHOW + MSMF 时，MSMF 那路会
        # "打开成功但抓不到帧"（报 can't grab frame），双目直接废掉；
        # 而且两台相机在两种后端下的索引空间不一定一致，混用还会张冠李戴。
        # MF 模式按设备身份打开，不再试 OpenCV 的后端枚举；`auto` 也先走 MF，
        # 失败由 `open_camera_source` 回退 OpenCV（打印警告，不静默）。
        mf_only = (self._capture_api in ("auto", "mf")
                   and bool(self._left_role) and bool(self._right_role))
        backends = ([("MF", None)] if mf_only else
                    [("指定", backend)] if backend is not None
                    else [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)])
        errors: list[str] = []
        self.cams: list[LatestFrameCamera] = []

        if self._auto:
            self._negotiate(backends, errors)
        else:
            for name, api in backends:
                opened = self._try_open_pair(name, api, int(width), int(height),
                                             errors)
                if opened is not None:
                    self.cams = opened
                    break

        if not self.cams:
            raise RuntimeError(self._no_camera_message(errors))

        self._primed = False
        # ---- 定案尺寸一律取**真帧**（上报值可能不准）----
        # auto 已在 `_negotiate` 里按真帧尺寸定案；显式档在这里补取一次真帧。
        # `width/height`（= `image_size`）要喂给标定器，必须是真帧尺寸，否则
        # 标定 image_size / 录制 capture_size 会与真实帧不符。上报值只在
        # **取不到真帧时**兜底，且明确告警（不静默）。
        if not self._auto:
            frames, why = _probe_real_frames(self.cams)
            if frames is None:
                print(f"[警告] 显式档 {int(width)}x{int(height)} 取不到真帧（{why}）；"
                      f"按上报值继续，标定尺寸可能不准", flush=True)
                self.actual_sizes = [(c.info.width, c.info.height) for c in self.cams]
            else:
                self.actual_sizes = [(f.shape[1], f.shape[0]) for f in frames]
                for i, f in enumerate(frames):
                    rep = (self.cams[i].info.width, self.cams[i].info.height)
                    if self.actual_sizes[i] != rep:
                        print(f"[警告] 相机 {i} 上报 {_wh(rep)}、实际给帧 "
                              f"{_wh(self.actual_sizes[i])}；按**实际帧尺寸**继续",
                              flush=True)
            self.resolved_size = self.actual_sizes[0]
        self.width = self.actual_sizes[0][0]
        self.height = self.actual_sizes[0][1]
        if not self._auto:                # auto 的定案行已在 _negotiate 里打过
            for i, idx in enumerate((left_index, right_index)):
                print(f"相机 {idx}: {describe_info(self.cams[i].info)}")

    # ------------------------------------------------------------------
    def _open_one(self, role: str, role_name: str, idx: int, api: int,
                  rw: int, rh: int) -> LatestFrameCamera:
        """打开一台相机（`optimize=False`）；打不开且配了备用相机时顶上。

        ⚠️ **`optimize=False` 是必须的，不是省事**（2026-10-14 修）：
        `CameraSource` 默认 `optimize=True`，它会**为这一台相机**搜索
        "最适合自己"的后端 + 分辨率。但双目要求**两路分辨率完全一致**
        （`image_size`、标定、三角测量全都建立在这个前提上），
        于是各自优化必然把两路拉开。受控实测（连续两轮，结果完全一致）：

            请求 1920x1080  optimize=False -> idx1=1920x1080 idx0=1920x1080  一致
            请求 1920x1080  optimize=True  -> idx1=1280x720  idx0=1920x1080  不一致

        不一致的表现是标定**直接失败**，而用户看到的是"我明明设了 1920x1080"。
        关掉优化 ≠ "不管分辨率"：给不出帧时 `CameraSource` 仍会回退到候选分辨率，
        那时两路是否一致由上层**报出来**（A3），是可见的，而不是被静默优化掉。
        """
        spec = _spec_of_role(role_name) if role_name else None
        if self._capture_api in ("auto", "mf"):
            if spec is not None and (spec.identity or spec.port):
                try:
                    src = open_camera_source(
                        index=idx, identity=spec.identity, port=spec.port,
                        width=rw, height=rh, fps=self._fps, backend=api,
                        optimize=False, capture_api=self._capture_api)
                    print(f"[{role}] 采集后端：{src.info.backend} "
                          f"({spec.port or spec.identity})", flush=True)
                    return LatestFrameCamera(src)
                except Exception as exc:                 # noqa: BLE001
                    if self._capture_api == "mf":
                        raise
                    print(f"[警告] {role} MF 打不开（{exc}）；"
                          f"回退 OpenCV #{idx}", flush=True)
            elif self._capture_api == "mf":
                raise RuntimeError(
                    f"{role} 要求 capture_api=mf，但 --{role_name or '?'} "
                    "不是已绑定的角色（没有 identity/port）；"
                    "显式数字编号请用 --capture-api opencv")
        try:
            return LatestFrameCamera(CameraSource(
                index=idx, width=rw, height=rh, fps=self._fps, backend=api,
                optimize=False))
        except RuntimeError:
            backup = self._backup_camera
            if backup is None or backup == idx:
                raise
            print(f"[警告] {role}相机 {idx} 打不开，改用备用相机 {backup}")
            print("       注意：备用相机与被替换的那台型号/视野都不同，"
                  "标定结果只对当前这个组合有效")
            return LatestFrameCamera(CameraSource(
                index=backup, width=rw, height=rh, fps=self._fps,
                backend=api, optimize=False))

    def _try_open_pair(self, name: str, api: int, rw: int, rh: int,
                       errors: list[str], deadline: float | None = None,
                       ) -> list[LatestFrameCamera] | None:
        """按 (name, api, rw, rh) **并行**打开两路；成功返回 [左, 右]，失败返回 None。

        失败时把原因写进 `errors` 并**当场打印** —— `errors` 只在"所有组合都
        失败"时才被收进最终报错，若只 append 就等于**静默**（独立 Critic 的 Q2b
        实测：DSHOW 超时 + MSMF 成功 → 终端一个字都没提超时）。
        """
        # ---- 两路**并行**打开（2026-10-14 改）----
        # 原来是**串行**：实测本机一台相机在 MSMF 下要 29.8 ~ 51.4 秒
        # （同型号另一台只要 0.9 秒）。串行意味着总等待是**两者之和**，并行只有较慢那个。
        results: list[object] = [None] * 2
        failures: list[BaseException] = []

        def _open_slot(slot: int, role: str, role_name: str, idx: int) -> None:
            try:
                results[slot] = self._open_one(role, role_name, idx, api, rw, rh)
            except BaseException as exc:            # noqa: BLE001
                failures.append(exc)                # 交给主线程重新抛

        threads = [
            threading.Thread(target=_open_slot,
                             args=(0, "左", self._left_role, self._left_index),
                             daemon=True, name="open-left"),
            threading.Thread(target=_open_slot,
                             args=(1, "右", self._right_role, self._right_index),
                             daemon=True, name="open-right"),
        ]
        print(f"[{name}] 并行打开两路相机（左 #{self._left_index} / "
              f"右 #{self._right_index}）请求 {_profile_label(rw, rh)}…", flush=True)
        t_open = time.perf_counter()
        for th in threads:
            th.start()
        # ⚠️ 原来是裸 `th.join()`，**没有 timeout**（2026-10-31 修）：
        # `cv2.VideoCapture` 可能在驱动层挂住，主线程会**永久等待**；
        # `daemon=True` 只保证进程能退出，不保证主线程不被卡。
        # auto 还要服从协商总预算：把本次等待压到"预算之内"。
        budget = OPEN_JOIN_TIMEOUT_S
        if deadline is not None:
            budget = max(1.0, min(OPEN_JOIN_TIMEOUT_S,
                                  deadline - time.perf_counter()))
        stuck = _join_all(threads, budget)
        opened = [c for c in results if c is not None]
        if stuck:
            for c in opened:
                c.stop()
            msg = (f"{name} 请求 {_profile_label(rw, rh)}：并行打开相机超时 "
                   f"{budget:.0f}s 仍未返回（{'/'.join(t.name for t in stuck)}）")
            errors.append(msg)
            print(f"[警告] {msg}", flush=True)
            return None
        if failures:
            for c in opened:
                c.stop()
            first_line = (str(failures[0]).splitlines() or ["打开失败"])[0]
            msg = f"{name} 请求 {_profile_label(rw, rh)}：{first_line}"
            errors.append(msg)
            print(f"[警告] {msg}", flush=True)
            return None
        if len(opened) != 2:
            for c in opened:
                c.stop()
            errors.append(f"{name} 请求 {_profile_label(rw, rh)}："
                          f"只打开了 {len(opened)}/2 路")
            return None
        print(f"[{name}] 两路都已打开，用时 {time.perf_counter() - t_open:.1f}s",
              flush=True)
        return opened                          # type: ignore[return-value]

    def _negotiate(self, backends: list[tuple[str, int]],
                   errors: list[str]) -> None:
        """auto：逐档协商，第一个满足「两路一致 + 帧率下限」的档即定案。

        控制流一句话：外层**成对挑后端**（同一后端同时开两路），内层按
        `video_profiles.probe_order(None)` **从高到低**逐档试，每档两道守卫
        —— A3 比对两路分辨率（**先**比 `cap.get()` 上报值、**再**比真帧尺寸，
        任一不等就降档；只比上报值会让"上报一致、真帧不一致"被静默接受）、
        A2 短测**成对**帧率（低于 `AUTO_MIN_PAIR_FPS` 就降档；刚打开测不到
        先等首帧再测，A5）。定案尺寸记**真帧**尺寸（喂给标定器的就是它）。
        全档都不满足就带着**逐档记录**收场。
        """
        deadline = time.perf_counter() + AUTO_PAIR_DEADLINE_S
        # 兼容测试里的 `_StubReader`（它只构造了 _negotiate 需要的最小状态）。
        min_fps = float(getattr(self, "min_pair_fps", AUTO_MIN_PAIR_FPS))
        attempts: list[str] = []
        detail: list[str] = []
        for name, api in backends:
            for (rw, rh) in probe_order(None):
                if time.perf_counter() >= deadline:
                    attempts.append(f"总预算 {AUTO_PAIR_DEADLINE_S:.0f}s 用尽")
                    self._finish_negotiate_failure(errors, attempts, detail, min_fps)
                    return
                label = _profile_label(rw, rh)
                print(f"[成对自适应] 试 {label}（后端 {name}）…", flush=True)
                opened = self._try_open_pair(name, api, rw, rh, detail,
                                             deadline=deadline)
                if opened is None:
                    attempts.append(f"{label}✘ 打不开（见上面警告）")
                    continue
                # ---- A3(1)：两路**上报**分辨率必须一致（cap.get()）----
                reported = [(c.info.width, c.info.height) for c in opened]
                if reported[0] != reported[1]:
                    attempts.append(f"{label}✘ 两路上报不一致"
                                    f"（左 {_wh(reported[0])} / 右 {_wh(reported[1])}）")
                    for c in opened:
                        c.stop()
                    continue
                # ---- A3(2)：上报值不可信，必须拿**真帧**尺寸再比一次 ----
                # `cap.get()` 是上报值、可能与实际给帧不符（见 `read()` 里的实测：
                # 内置相机上报 1920x1080、实际 1280x1080）。只比上报值 = 拿弱证据
                # 下强结论：两台上报 1920x1080、真帧 1280x1080 + 1920x1080 会被
                # **静默接受**，而标定吃的是真帧尺寸。
                timeout = AUTO_FRAME_PROBE_TIMEOUT_S
                if deadline is not None:
                    timeout = max(0.5, min(timeout, deadline - time.perf_counter()))
                frames, why = _probe_real_frames(opened, timeout)
                if frames is None:
                    attempts.append(f"{label}✘ {why}")
                    for c in opened:
                        c.stop()
                    continue
                real = [(f.shape[1], f.shape[0]) for f in frames]
                if real[0] != real[1]:
                    attempts.append(f"{label}✘ 两路**实际帧尺寸**不一致"
                                    f"（左 {_wh(real[0])} / 右 {_wh(real[1])}）")
                    for c in opened:
                        c.stop()
                    continue
                if real[0] != reported[0]:
                    print(f"[成对自适应] 注意：{label} 上报 {_wh(reported[0])}、"
                          f"实际给帧 {_wh(real[0])}；按**实际帧尺寸**继续", flush=True)
                # ---- A5 + A2：短测**成对**帧率（刚打开测不到先等首帧再测）----
                rate = _measure_pair_fps(opened)
                if rate < min_fps:
                    attempts.append(
                        f"{label}✘ 成对 {rate:.1f} fps < 下限 "
                        f"{min_fps:.1f}（实际 {_wh(real[0])}）")
                    for c in opened:
                        c.stop()
                    continue
                # ---- 定案（记**真帧**尺寸，那才是喂给标定器的尺寸）----
                self.cams = opened
                self.resolved_size = real[0]
                self.actual_sizes = real
                self.pair_fps = rate
                history = " / ".join(attempts) if attempts else "首档即成"
                print(f"[成对自适应] 定案 {_wh(real[0])}"
                      f"（后端 {name}，左 {_wh(real[0])}，右 {_wh(real[1])}，"
                      f"成对 {rate:.1f} fps；试过 {history}）")
                return
        self._finish_negotiate_failure(errors, attempts, detail, min_fps)

    @staticmethod
    def _finish_negotiate_failure(errors: list[str], attempts: list[str],
                                  detail: list[str],
                                  min_fps: float = AUTO_MIN_PAIR_FPS) -> None:
        errors.append(
            "自适应协商失败：候选分辨率档没有一个同时满足「两路实际一致」"
            f"与「成对帧率 ≥ {min_fps:.1f}」。逐档记录：\n    "
            + "\n    ".join(attempts or ["（没有走完任何一档）"]))
        if detail:
            errors.append("逐组合明细：\n    " + "\n    ".join(detail))

    def _no_camera_message(self, errors: list[str]) -> str:
        """相机没凑成一对时的报错正文（含"按代价从小到大"的定位步骤）。"""
        # ⚠️ 原文案把**所有**失败都归给"USB 带宽/集线器"并建议"换一个后端"，
        # 而真实原因常常完全不是那个（用户实测：MSMF 读 12 帧全空、DSHOW 正常）。
        # 这里是**诚实**的定位步骤：先物理拔插（设备级失效软件救不回），再看逐路报错。
        hint = (
            "\n怎么定位（按代价从小到大）：\n"
            "  ① **物理拔插**那台失败的相机：设备被系统置为无效"
            "（MSMF 报 -2147483638 / DEVICE_INVALIDATED）时软件重开会一直失败 —— "
            "只有拔下来重插（或换一个**直插机身**的 USB 口，别走集线器/扩展坞）才恢复。\n"
            "  ② 看上面每一路的报错：**两台都失败**多半是后端/驱动；"
            "**只有一台失败**先怀疑那一台（换线、换口、单独打开试试）。\n"
            "  ③ 两路分辨率不一致：自适应会在**同一个后端**下逐档重试；"
            "若某台相机只支持某个档，请用 `--width/--height` 指定**两台都支持**的档。\n"
            "  ④ 跑 `python apps/tools/cameras.py doctor` 看**「双路同时取帧」**那段"
            "（它真开两路、与双目同工况，能区分「逐台都好」与「同时取帧失败」）。")
        if self._auto:
            head = (f"左右相机（{self._left_index} / {self._right_index}）"
                    f"自适应协商失败：\n  ")
        else:
            head = (f"左右相机（{self._left_index} / {self._right_index}）"
                    f"无法用同一个后端同时打开：\n  ")
        return head + "\n  ".join(errors) + hint

    @property
    def info(self):
        """两路的 SourceInfo（左, 右）。分辨率可能不同，别假设一样。"""
        return self.cams[0].info, self.cams[1].info

    def _wait_fresh_pair(self) -> None:
        """等两路都出新帧。**`read()` 与 `read_pair()` 共用这一份**。

        分开写两份迟早会漂：哪天调了 `CAMERA_ALIGN_STEPS` 只改一边，
        两条路对"什么叫一对新帧"的定义就不一样了。
        """
        if not self._primed:
            first = [c.wait_first() for c in self.cams]
            self._primed = True
            # 报告的分辨率和实际帧尺寸可能不一致（实测内置相机
            # 报告 1920x1080、实际给 1280x1080），所以按真实帧再说一次
            for i, f in enumerate(first):
                if f is not None:
                    real = f"{f.shape[1]}x{f.shape[0]}"
                    rep = f"{self.cams[i].info.width}x{self.cams[i].info.height}"
                    if real != rep:
                        print(f"  （相机 {i} 实际输出 {real}，与报告的 {rep} 不一致，"
                              f"已按实际情况处理）")

        snap = [c.latest() for c in self.cams]
        deadline = time.perf_counter() + CAMERA_WAIT_TIMEOUT
        while True:
            # 左路必须出新帧（否则主循环会拿到同一帧空转）；
            # 右路要连续走 CAMERA_ALIGN_STEPS 帧，才说明它的时间也确实往前了 ——
            # 两路采集线程独立跑，右路只动一下不代表它追上了左路。
            # （`latest()` 现在返回 `(seq, frame, t_capture)`，这里只用 seq。）
            left_fresh = self.cams[0].latest()[0] > snap[0][0]
            right_fresh = all(c.latest()[0] > s[0] + CAMERA_ALIGN_STEPS - 1
                              for c, s in zip(self.cams[1:], snap[1:]))
            if (left_fresh and right_fresh) or time.perf_counter() >= deadline:
                return
            time.sleep(CAMERA_POLL_INTERVAL)

    def read(self) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
        self._wait_fresh_pair()
        _s1, a, _t1 = self.cams[0].latest()
        _s2, b, _t2 = self.cams[1].latest()
        return (a is not None and b is not None), a, b

    def read_pair(self, *, max_skew_ms: float | None = None
                  ) -> tuple[bool, np.ndarray | None, np.ndarray | None,
                             float, float, float | None, int, int]:
        """取一对**带时间戳**的帧，并可丢弃时间差过大的那一对。

        返回 ``(ok, left, right, t_l, t_r, skew_ms, seq_l, seq_r)``；``skew_ms``
        为正表示**右路更晚**。

        ## 为什么必须另开一个入口而不是改 `read()`

        `read()` 在 :735-736 把 `latest()` 带的 `t_capture` 取出来又扔了，
        于是上层拿不到"这两帧差多少毫秒"。而双目实时没有硬件同步，skew 是
        **唯一**能说明"这帧三维点可不可信"的量 —— 丢了它，误差只能靠猜。
        改 `read()` 的签名要动 9 处 + 18 个调用点（`docs/待办清单.md` B-01），
        风险与本收益不成比例，所以只新增、不动老签名。

        ## 两种 `ok=False`

        - 任一路没帧（相机断了 / 还没出第一帧）→ 帧也返回 None；
        - 给了 `max_skew_ms` 且直到 deadline 都超限 → 帧返回 None、`dropped_pairs += 1`。
          调用方据此**不产出三维点**（拿冻结帧硬算出来的点比没有点更糟）。

        `t_l` / `t_r` 都是同一进程的 `perf_counter`（`sources.py:792`），可直接相减。
        """
        limit = self.max_skew_ms if max_skew_ms is None else float(max_skew_ms)
        deadline = time.perf_counter() + CAMERA_WAIT_TIMEOUT
        while True:
            self._wait_fresh_pair()
            s1, a, t1 = self.cams[0].latest()
            s2, b, t2 = self.cams[1].latest()
            if a is None or b is None:
                return False, None, None, float(t1), float(t2), None, int(s1), int(s2)
            skew = (float(t2) - float(t1)) * 1000.0
            if limit is None or abs(skew) <= limit:
                self.last_skew_ms = skew
                return True, a, b, float(t1), float(t2), skew, int(s1), int(s2)
            self.dropped_pairs += 1
            self.last_skew_ms = skew
            if time.perf_counter() >= deadline:
                return False, None, None, float(t1), float(t2), skew, int(s1), int(s2)

    def stalled_sides(self, timeout: float = 1.0) -> tuple[bool, bool]:
        """(左路停住, 右路停住)。`LatestFrameCamera.stalled()` 的直接透传。

        为什么必须由上层问这个：一路掉线后 `latest()` 会**继续返回冻结的旧帧**
        （`sources.py:301-303`），`read()`/`read_pair()` 的 `ok` 仍然可能为真。
        不查它，就会"画面冻住但程序照跑"，还会拿冻结帧算出假的三维点。
        """
        return (self.cams[0].stalled(timeout=timeout),
                self.cams[1].stalled(timeout=timeout))


    def seek(self, index: int) -> tuple[bool, np.ndarray | None]:
        raise NotImplementedError("实时相机不能按帧号定位（帧号没有共同基准）")

    def release(self) -> None:
        for c in self.cams:
            c.stop()


def _make_reader(value, what: str) -> FrameReader:
    """造一路帧源：类型判定**委托** `sources.infer_source_kind`（全项目唯一一份）。

    这里只负责"上位类型 -> 哪个 FrameReader"：`PairSource` 要 `seek()` /
    `frame_count()`，所以不能用 `sources` 的 `FileSource` / `ImageSource`
    —— 那是"顺序读"的形状。判定本身照样只有一处，改后缀清单只改 `sources`。
    """
    kind = infer_source_kind(value)
    if kind != "images":
        return VideoFrameReader(value)
    try:
        return ImageDirFrameReader(value)
    except NotADirectoryError:
        # 单张图片做不了双目（一帧没有视差）—— 这条提示比 "不是文件夹" 有用
        raise RuntimeError(f"{what} 给的是单张图片（{value}），做不了双目；"
                           f"请给文件夹（directory=）或视频（video=）") from None


# --------------------------------------------------------------------------
# 成对帧源
# --------------------------------------------------------------------------
class PairSource:
    """左路 + 右路当整体取帧，支持同步偏移。

    三种构造方式（只能选一种）::

        PairSource(camera=(0, 1))
        PairSource(video=("cam_l.mp4", "cam_r.mp4"))
        PairSource(directory=("imgs/left", "imgs/right"))

    同步偏移::

        PairSource(directory=(l, r), sync_offset=+8)
        # 左路第 i 帧 与 右路第 i+8 帧 配对（右路时间更靠前 -> 右路先跳过 8 帧）

    偏移量可以直接来自 `stereo.find_offset(...).offset`；
    也可以用 `stereo.apply_offset()` 把同一个数翻译成人话核对一遍。
    """

    def __init__(
        self,
        camera: tuple[int, int] | None = None,
        video: tuple[str | Path, str | Path] | None = None,
        directory: tuple[str | Path, str | Path] | None = None,
        sync_offset: int = 0,
        width: int = 1920,
        height: int = 1080,
        fps: float = 60,
        backup_camera: int | None = None,
        backend: int | None = None,
        pairing: str = "canary",
        # 视频文件时间戳配对的**时长**容差（秒），固定 0.033。
        # **不要按帧率派生**：帧周期来自视频容器，构造函数的 ``fps=`` 是
        # **相机请求帧率**，视频路径下它与容器帧率毫无关系。真要为视频派生，
        # 正确来源在 `_build_timestamp_plan` 里读容器 fps，那是另一处改动。
        timestamp_tolerance_s: float = 0.033,
        max_skew_ms: float | None = None,
        min_pair_fps: float | None = None,
        left_role: str = "",
        right_role: str = "",
        capture_api: str = "opencv",
    ) -> None:
        given = [name for name, v in
                 (("camera", camera), ("video", video), ("directory", directory)) if v]
        if len(given) != 1:
            raise RuntimeError(
                "camera= / video= / directory= 只能且必须给一个："
                f"现在给了 {given or '（什么都没给）'}")

        self.sync_offset = int(sync_offset)
        self.kind = ""
        self.n = 0                    # 已经返回了多少对
        self.warnings: list[str] = []
        # 相机协商结果（只有 camera 分支会填；其余帧源保持 None）——
        # 标定脚本据此把"请求了什么 / 实际定案成什么"写进标定 meta。
        self.requested_size: str | tuple[int, int] | None = None
        self.resolved_size: tuple[int, int] | None = None
        self.actual_sizes: list[tuple[int, int]] = []
        self.pair_fps: float | None = None
        self.pairing_policy = pairing
        self._pair_plan = None
        self._pair_pos = 0
        self._left: FrameReader
        self._right: FrameReader
        if pairing not in {"index", "timestamp", "canary"}:
            raise ValueError(f"pairing 必须是 index/timestamp/canary，收到 {pairing!r}")

        if camera is not None:
            if self.sync_offset:
                self.warnings.append(
                    f"实时相机不支持 sync_offset（给了 {self.sync_offset:+d}），已忽略："
                    "两路各在自己的线程里跑到最新帧，帧号没有共同基准。"
                    "要真正同步得靠硬件触发，或先用 stereo.sync 标定偏移再录成视频处理。")
                self.sync_offset = 0
            self._left = self._right = CameraPairReader(
                int(camera[0]), int(camera[1]), width=width, height=height, fps=fps,
                backup_camera=backup_camera, backend=backend,
                max_skew_ms=max_skew_ms, min_pair_fps=min_pair_fps,
                left_role=left_role, right_role=right_role,
                capture_api=capture_api)
            self.kind = "camera"
            # 把协商结果暴露出来（显式档也有：resolved == 实际值）。
            self.requested_size = self._left.requested_size
            self.resolved_size = self._left.resolved_size
            self.actual_sizes = list(self._left.actual_sizes)
            self.pair_fps = self._left.pair_fps
        elif video is not None:
            self._left = _make_reader(video[0], "左路")
            self._right = _make_reader(video[1], "右路")
            self.kind = "video"
            n_l, n_r = self._left.frame_count(), self._right.frame_count()
            print(f"视频：{self._left.label} + {self._right.label}，各 {n_l} / {n_r} 帧")
            if n_l and n_r and abs(n_l - n_r) > 1:
                self.warnings.append(
                    f"两段视频帧数不一致（{n_l} vs {n_r}）：会一直取到短的那路结束为止")
            self._build_timestamp_plan(video[0], video[1], float(timestamp_tolerance_s))
        else:
            self._left = _make_reader(directory[0], "左路")
            self._right = _make_reader(directory[1], "右路")
            self.kind = "dir"
            n_l, n_r = self._left.frame_count(), self._right.frame_count()
            if n_l != n_r:
                raise RuntimeError(f"两个文件夹图片数不一致：{n_l} vs {n_r}")
            print(f"图片：左右各 {n_l} 张（按文件名排序一一对应）")

        if self.sync_offset and self.kind != "camera":
            self._describe_offset()

    # ------------------------------------------------------------------
    def _build_timestamp_plan(self, left_video, right_video, tolerance_s: float) -> None:
        """金丝雀 / 时间戳配对的计划构建；失败只告警，不改变旧方案。"""
        if self.pairing_policy == "index":
            return
        lc = sidecar_for_video(left_video)
        rc = sidecar_for_video(right_video)
        if lc is None or rc is None:
            self.warnings.append(
                f"时间戳配对需要 {Path(left_video).stem}_times.csv / "
                f"{Path(right_video).stem}_times.csv；缺文件，回退索引配对")
            return
        try:
            lt = read_time_sidecar(lc)
            rt = read_time_sidecar(rc)
            plan = plan_timestamp_pairs(lt, rt, tolerance_s)
        except Exception as exc:                            # noqa: BLE001
            self.warnings.append(f"时间戳配对失败（{exc}），回退索引配对")
            return
        print(f"[时间戳配对] {plan.summary()}")
        if self.pairing_policy == "canary":
            safe, why = canary_safe(plan)
            if not safe:
                self.warnings.append(f"金丝雀未接管：{why}；回退索引配对")
                return
            print(f"[金丝雀] 接管：{why}")
        elif not plan.pairs:
            self.warnings.append("时间戳配对为空，回退索引配对")
            return
        self._pair_plan = plan
        if self.sync_offset:
            self.warnings.append(
                f"时间戳配对已启用，忽略常数 sync_offset={self.sync_offset:+d} 帧")
            self.sync_offset = 0

    def _describe_offset(self) -> None:
        """把偏移量翻译成人话打出来 —— 正负号最容易搞反，必须可见。

        语义与 `stereo.sync.apply_offset` 一致，只是补上"这意味着哪一路先开拍"：
        "右路第 i+offset 帧" = 右路的**时间跑在前面**，即右路更早开拍。
        """
        off = self.sync_offset
        if off > 0:
            how = (f"右路时间更靠前（右路更早开拍）：右路先跳过 {off} 帧，"
                   f"再与左路逐帧配对")
        else:
            how = (f"左路时间更靠前（左路更早开拍）：左路先跳过 {-off} 帧，"
                   f"再与右路逐帧配对")
        print(f"同步偏移 {off:+d} 帧：{how}")

    @property
    def is_live(self) -> bool:
        return self.kind == "camera"

    @property
    def image_size(self) -> tuple[int, int]:
        """(宽, 高)。两路分辨率不同时取左路 —— 标定要求两路一致，会另外告警。"""
        return self._left.width, self._left.height

    def read_pair(self, *, max_skew_ms: float | None = None
                  ) -> tuple[bool, np.ndarray | None, np.ndarray | None,
                             float, float, float | None, int, int]:
        """带时间戳的取帧：``(ok, left, right, t_l, t_r, skew_ms, seq_l, seq_r)``。

        只有**实时相机**那一条路有真实 `skew_ms`。视频 / 图片目录是逐帧对齐的，
        没有"两路时间差"这个概念 —— 那里返回 ``skew_ms=None``（明说不知道），
        **不要**把 `0.0` 当成"完全同步"来用。
        """
        if self.kind == "camera":
            return self._left.read_pair(max_skew_ms=max_skew_ms)  # type: ignore[attr-defined]
        ok, left, right = self.read()
        return ok, left, right, 0.0, 0.0, None, self.n, self.n

    @property
    def skew_ms(self) -> float | None:
        """最近一对帧的时间差（ms）；非实时源恒为 None。"""
        if self.kind == "camera":
            return self._left.last_skew_ms        # type: ignore[attr-defined]
        return None

    @property
    def dropped_pairs(self) -> int:
        """因 skew 超限被丢弃的**对数**；非实时源恒为 0。"""
        if self.kind == "camera":
            return self._left.dropped_pairs       # type: ignore[attr-defined]
        return 0

    def stalled_sides(self, timeout: float = 1.0) -> tuple[bool, bool]:
        """(左路停住, 右路停住)；非实时源恒为 (False, False)。"""
        if self.kind == "camera":
            return self._left.stalled_sides(timeout=timeout)  # type: ignore[attr-defined]
        return (False, False)

    def read(self) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
        """返回 (ok, left_bgr, right_bgr)。

        每次调用都返回**新的一对**：左路第 n 帧配右路第 n+offset 帧（n 从 0 数起）。
        偏移在第一次调用时一次性吸收掉（右路 offset>0 就先跳过 offset 帧，
        左路 offset<0 就先跳过 -offset 帧），之后两路逐帧同步前进 —— 所以
        视频那条路只在开头定位一次，中间不做任何 seek。
        """
        if self.kind == "camera":
            # 相机分支：_left 和 _right 是同一个对象，两路由它一起取
            return self._left.read()                    # type: ignore[return-value]

        if self._pair_plan is not None:
            while self._pair_pos < len(self._pair_plan.pairs):
                li, ri = self._pair_plan.pairs[self._pair_pos]
                self._pair_pos += 1
                ok, left = self._left.seek(li)
                if not ok or left is None:
                    return False, None, None
                ok, right = self._right.seek(ri)
                if not ok or right is None:
                    continue
                self.n += 1
                return True, left, right
            return False, None, None

        li = self.n + max(0, -self.sync_offset)         # 本对应给的左路帧号
        ri = self.n + max(0, self.sync_offset)          # 本对应给的右路帧号
        ok, left = self._left.seek(li)
        if not ok or left is None:
            return False, None, None
        ok, right = self._right.seek(ri)
        if not ok or right is None:
            if self.n == 0 and ri > 0:
                self.warnings.append(
                    f"右路不足 {ri} 帧，装不下 {self.sync_offset:+d} 的同步偏移")
            return False, None, None
        self.n += 1
        return True, left, right

    def describe(self) -> str:
        """一行说清当前帧源（打印用）。"""
        if self.kind == "camera":
            return f"两台实时相机：{self._left.label}（不支持同步偏移）"
        if self._pair_plan is not None:
            return (f"{self.kind}：{self._left.label} + {self._right.label}"
                    f"（时间戳配对，{len(self._pair_plan.pairs)} 对）")
        off = f"，同步偏移 {self.sync_offset:+d} 帧" if self.sync_offset else ""
        return f"{self.kind}：{self._left.label} + {self._right.label}{off}"

    def report_warnings(self) -> None:
        """把构造期攒下的告警打出来（不致命，但会影响结果可信度）。"""
        for w in self.warnings:
            print(f"[警告] {w}")

    def release(self) -> None:
        # 相机分支下 _left 和 _right 是同一个对象，只能释放一次
        for r in dict.fromkeys((self._left, self._right)):
            r.release()


def _resolve_arg_noninteractive(raw, side: str) -> int | None:
    """解析 `--left/--right` 的取值，**绝不弹交互选择**。

    ## 为什么必须禁止交互（2026-10-31 在 `apps/_task_runner.py` 里查到根因）

    `resolve_camera_arg(..., interactive=True)` 在某角色"按身份/端口/名字都找不到"
    且 `verified=True` 时会走 `input()`。而任务运行器起的是**子进程**，并且
    **没有重定向 stdin** —— 子进程仍挂在那台控制台上，于是 `input()` 会一直等
    一个**永远不会有人敲**的回车。

    用户看到的现象是：点了"开始标定"，一直显示"正在标定"；日志里只有一句
    "请选择…"，取景窗永远不出现，进程也永远不退出。全项目**唯一**能无限期挂住
    的地方就在这里，所以这两处必须显式 `interactive=False`。

    解析不了就**当场报错**，并把"下一步"说全（配置错 vs 设备不在线），
    让错误出现在日志里 —— 而不是等一个不存在的按键。
    """
    try:
        return resolve_camera_arg(raw, interactive=False)
    except RuntimeError as exc:
        first = (str(exc).splitlines() or [type(exc).__name__])[0]
        raise RuntimeError(
            f"{side} 相机（--{side} {raw}）解析失败。\n"
            f"  ⚠ 这里**不会**弹出「请选择哪个编号」——命令跑在后台/GUI 子进程里，"
            f"没有人能敲那个回车，等它等于永远挂住。\n"
            f"  原因：{first}\n"
            f"  怎么办（都不用开相机）：\n"
            f"    - 看当前设备与端口：python apps/tools/cameras.py list\n"
            f"    - 按端口重绑角色：  python apps/tools/cameras.py pin {side} port:<端口>\n"
            f"    - 或显式给编号：    --{side} <编号>"
            f"（编号必须是**同一后端**的编号）"
        ) from exc


def _pin_value_for(spec: RoleSpec | None) -> str:
    """给 `pin` 用的取值：优先 port（每台唯一），其次 id，最后 name。"""
    if spec is None:
        return ""
    if spec.port:
        return f"port:{spec.port}"
    if spec.identity:
        return f"id:{spec.identity}"
    if spec.name:
        return f'name:"{spec.name}"'
    return ""


def _spec_of_role(raw) -> RoleSpec | None:
    """`--left right` 这种**角色名** -> 配置里的 RoleSpec；其它取值返回 None。

    为什么要区分：这里只有"角色名"能查到两个角色各自记的是哪台设备；
    数字编号 / `port:xxx` / `auto` 都是用户当场指定的，没有配置可比。
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.isdigit() or text == AUTO_TOKEN or ":" in text:
        return None
    try:
        return load_registry().roles.get(text)
    except Exception:                                    # noqa: BLE001
        return None


def explain_same_camera(left_i: int, right_i: int, *,
                        left_spec: RoleSpec | None = None,
                        right_spec: RoleSpec | None = None) -> str:
    """左右两路落到**同一台相机**时的报错正文（**纯函数**：不读配置、不开相机）。

    ## 为什么要分成两种说法（2026-10-31 用户实测）

    以前这里一律说"角色绑定的那台相机不在线（换过 USB 口 / 重插过）"，
    并把用户引向 `--left 0 --right 1 --backend dshow`。在本机这两条**都是错的**：

      · 两台相机**都在线**（`cameras.py list` 两台都枚举得到，一台 1.2 s 出图）；
      · DSHOW 的 `#1` 是**笔记本内置**、`#2` 才是外接 aoni —— 照它给的两个编号会
        **开到另一台相机**，而且两路照样报同样的分辨率，界面上完全看不出异常。
        静默错位比"打不开"危险得多，所以那句话必须删掉。

    真相是 `config/cameras.json` 里 left 与 right 记的是**同一台**
    （identity / port / name / index 逐字相同）—— 这是**配置错**，
    改法也**不用开相机**（pin 按端口/身份绑），所以先说这一种。

    只有两个角色记得的**不是同一台**时，才轮到"解析退化 / 索引漂移"
    ——那时换过口、重插过、退化到同一个兜底索引才是合理解释。
    """
    head = (f"左右两路解析到了**同一个相机**（索引 #{left_i}）—— 标定无法进行。\n")
    same_identity = bool(
        left_spec is not None and right_spec is not None
        and left_spec.identity and right_spec.identity
        and left_spec.identity.upper() == right_spec.identity.upper()
    )
    if same_identity:
        lv, rv = _pin_value_for(left_spec), _pin_value_for(right_spec)
        if lv == rv:
            # 两个角色记的端口/身份**逐字相同** = 它们描述的就是同一台，
            # 照抄这个值 pin 两次只会又绑回同一台。必须从"另一台"取。
            lv, rv = "<这一台的 port，见 list>", "<另一台的 port，见 list>"
        return (
            head
            + "  ⚠ 这是**配置错**，**不是相机掉线**：\n"
            + f"    config/cameras.json 里 left 与 right 记的是**同一台**"
              f"（identity 都是 {left_spec.identity}）。\n"
            + "    两台相机是双目的前提。改法**不用开相机**（按端口绑，不碰编号）：\n"
            + f"      python apps/tools/cameras.py pin left  {lv}\n"
            + f"      python apps/tools/cameras.py pin right {rv}\n"
            + "    ⚠ 两个值**不能填同一个**；名字（name:）也区分不了两台 —— "
              "本机同型号，\n"
              + "      现有配置里连名字都一模一样，而按名字匹配的是「命中的第一台」。\n"
            + "    端口用 python apps/tools/cameras.py list 看（不开相机，毫秒级）；\n"
            + "    哪台是左、哪台是右只能你目视决定。改完复核：\n"
            + "      python apps/tools/cameras.py check"
              "   ← 它现在会直接报「左右撞车」并返回非零退出码。"
        )
    return (
        head
        + "  两个角色记的**不是同一台**（identity 不同或拿不到），"
          "所以更像是**解析退化 / 索引漂移**：\n"
        + "    1) 角色绑定的那台相机当前解析不到（换过 USB 口 / 重插过 / 被占用），\n"
        + "       两个角色于是退化到同一个索引；\n"
        + "    2) 显式给了同一个索引（如 --left 0 --right 0）；\n"
        + "    3) **后端枚举错位**：`list_devices()` 的 order 来自 Windows PnP 枚举，\n"
        + "       而 `cv2.VideoCapture(index, backend)` 用的是各后端自己的枚举，"
          "两者不保证一致。\n"
        + "  怎么办：\n"
        + "    - python apps/tools/cameras.py check    看角色有没有漂移"
          "（会直接报「左右撞车」）\n"
        + "    - python apps/tools/cameras.py list     看当前设备、编号与端口\n"
        + "    - 要显式给编号，就给**同一后端**下的两个**不同**编号，"
          "并且只用该后端\n"
          + "      自己的编号（别把 `list` 的 PnP 编号当后端编号用；"
          "本机 DSHOW 的 #1 是内置相机）\n"
        + "    - 开跑后先目视确认：两路必须是**同一场景的左右两个视角**。"
    )


def from_args(args) -> PairSource:
    """按命令行参数造一个 `PairSource`（`calibrate_stereo` / `verify_stereo` 共用）。

    两个 app 的取帧参数是**同一套**（`--left/--right`、`--left-video/--right-video`、
    `--left-dir/--right-dir`、`--width/--height/--fps`、`--backup-camera`），
    所以映射逻辑放在库里共用，不各写一份 —— 各写一份就会出现
    "标定能用图片文件夹、复检不能用"这类不一致。

    `--left/--right` 的取值可以是**索引**（`0`）、**角色名**（`left`，查
    `config/cameras.json`）、或 `auto`；`--auto-cameras` 则两台都自动挑。
    之所以支持角色名：相机索引会随 USB 口/连接顺序漂移，角色名绑定的是
    **设备身份**，重插后仍能解析到正确的那台（见 `sphere_tracker/camera_registry.py`）。
    """
    left_raw = getattr(args, "left", None)
    right_raw = getattr(args, "right", None)
    left_spec = _spec_of_role(left_raw)
    right_spec = _spec_of_role(right_raw)
    capture_api = (str(getattr(args, "capture_api", "") or "").strip().lower()
                   or "opencv")
    auto = (bool(getattr(args, "auto_cameras", False))
            or left_raw == AUTO_TOKEN or right_raw == AUTO_TOKEN)

    if auto or (left_raw is not None and right_raw is not None):
        if auto:
            picks = select_cameras(need=2)
            if len(picks) < 2:
                raise RuntimeError(
                    "自动选择失败：没找到两台可用的相机。请跑 "
                    "python apps/tools/cameras.py list 看当前设备，"
                    "或用 --left 0 --right 2 手动指定。")
            left_i, right_i = picks[0].index, picks[1].index
            print(f"[自动选择] 左 #{left_i}｜右 #{right_i}")
        else:
            left_i = _resolve_arg_noninteractive(left_raw, "left")
            right_i = _resolve_arg_noninteractive(right_raw, "right")


        # 左右必须落在**两台不同的相机**上。此前没有这道检查，于是
        # "两个角色解析到同一个索引"会静默继续：标定窗口里两路其实是同一台相机，
        # 用户要到一对有效图像都收不到时才发现，白白浪费一轮。
        #
        # 报错正文由纯函数 `explain_same_camera()` 生成：它按"两个角色记的是不是
        # 同一台"分成**配置错**与**解析退化**两种说法 —— 这两种的下一步完全不同，
        # 混成一句就会把用户引向"去查 USB 口"（本次用户遇到的是配置错，两台都在线）。
        if left_i == right_i:
            raise RuntimeError(explain_same_camera(
                left_i, right_i,
                left_spec=left_spec, right_spec=right_spec))

        # 相机源默认固定 **MSMF**。
        #
        # 为什么必须是 MSMF：`list_devices()` 的序号来自 Windows PnP 枚举，
        # 而 `cv2.VideoCapture(index, backend)` 用的是各后端自己的枚举顺序。
        # 本机实测（同一时刻、同一批设备）：
        #
        #     PnP    : 0=aoni  1=aoni  2=内置  3=内置红外
        #     DSHOW  : 0=aoni  1=内置  2=aoni  3=打不开     ← 错位
        #     MSMF   : 0=aoni  1=aoni  2=内置  3=打不开     ← 与 PnP 一致
        #
        # 判据：用户盖住了内置相机，它在哪一路全黑，那一路就是内置。
        # 用 DSHOW 时序号 1 会开到内置（黑屏，看起来像相机坏了）；
        # 用 MSMF 则序号与 PnP 对齐，解析结果可以直接拿去开。
        #
        # 顺带：MSMF 在 1080p 下能跑 30fps，DSHOW 会协商成 YUY2 未压缩掉到
        # 4.3fps（见 sources.py 里 optimize_mode 的注释），所以 MSMF 两边都更优。
        #
        # ⚠️ **2026-10-14 修：`auto` 原来被折叠成 `INDEX_BACKEND`（= MSMF）**，
        # 于是 `PairSource` 里那份 `[("DSHOW", ...), ("MSMF", ...)]` 候选顺序
        # **从来没被走到过** —— 函数名叫 `auto`，行为却是"强制 MSMF"。
        # 实测后果（用户真机跑标定，连续两轮一致）：
        #
        #     MSMF : idx1=None（读 12 帧全空）  idx0=1920x1080
        #     DSHOW: idx1=1920x1080            idx0=1920x1080
        #
        # 即 **MSMF 对其中一台根本不给帧**，而 `auto` 必选它 → 标定跑不起来，
        # 还打出一条把人引向"换 USB 口"的错误建议（那台相机在 DSHOW 上好好的）。
        #
        # 现在 `auto` 传 `None`，让 `PairSource` 的**成对挑选**真正生效：
        # 先把两路都用 DSHOW 开，哪个开不了就**整对**换 MSMF ——
        # 依然绝不会两路混用不同后端（那正是原注释担心的）。
        backend_arg = getattr(args, "backend", None)
        api = resolve_backend(backend_arg)

        # ---- 索引来自「PnP 位次」时，后端**只能**是 INDEX_BACKEND ----------------
        #
        # 🔴 2026-10-18 修的真缺陷（用户报"aoni 被认成 Integrated"时发现）：
        # `left_i/right_i` 来自 `resolve_camera_arg()` / `select_cameras()`，
        # 也就是 **Windows PnP 枚举位次**。而 `cv2.VideoCapture(index, backend)`
        # 用的是**后端自己的枚举顺序** —— 实测本机（`probe_index_map.py`）：
        #
        #     PnP 标签：#0 aoni-left   #1 aoni-right   #2 Integrated   #3 Integrated IR
        #     MSMF     ：#0 有图       #1 有图        #2 纯黑(被遮)   #3 打不开   ← 一致
        #     DSHOW    ：#0 有图       #1 **纯黑**    #2 **有图**     #3 打不开   ← 错位
        #
        # 即 **DSHOW #1 是笔记本内置、#2 才是 aoni**。所以拿 PnP 位次 + DSHOW
        # 开会**开到另一台相机**：角色 `right` 实际拿到的是**被遮住的内置相机**，
        # 两路都报 1920x1080、界面上完全看不出，**标定直接废掉**。
        #
        # 这比"打不开"危险得多 —— 打不开会报错，错位是**静默**的。
        #
        # 因此：只要有一路的索引是**按角色名/auto 解析**出来的，就锁死
        # `INDEX_BACKEND`；想用别的后端必须**显式给数字编号**（那样索引就在
        # 那个后端的编号空间里，是用户自己确认过的）。
        def _explicit_index(v) -> bool:
            return v is not None and str(v).strip().isdigit()

        from sphere_tracker.sources import INDEX_BACKEND, INDEX_BACKEND_NAME

        pnp_derived = auto or not (_explicit_index(left_raw)
                                   and _explicit_index(right_raw))
        if pnp_derived:
            if api is None:
                api = INDEX_BACKEND
                print(f"[后端] 索引来自角色/auto（PnP 位次），锁定 "
                      f"{INDEX_BACKEND_NAME} —— 只有它与 PnP 位次一致。"
                      f"（想用别的后端请显式给数字编号）")
            elif api != INDEX_BACKEND:
                raise RuntimeError(
                    f"`--backend {backend_arg}` 与「按角色名解析索引」**不能一起用**。\n"
                    f"  原因：角色名解析出的是 **PnP 位次**，而各后端的编号空间不同。\n"
                    f"        本机实测：DSHOW 的 #1 是笔记本内置相机，#2 才是外接 aoni；\n"
                    f"        而 MSMF 与 PnP 一致。用 DSHOW 打开会把**另一台**相机当成\n"
                    f"        这一路，两路照样报 1920x1080，**界面上看不出任何异常**。\n"
                    f"  怎么办（二选一）：\n"
                    f"    1) 去掉 `--backend`，用默认的 {INDEX_BACKEND_NAME}；\n"
                    f"    2) 确认过 DSHOW 的编号后，**显式给数字**：\n"
                    f"         --left 0 --right 2 --backend {backend_arg}\n"
                    f"       （数字是 DSHOW 的编号，不是 `cameras.py list` 里的那个）")
        else:
            print(f"[后端] 用显式数字编号 + {backend_arg or 'auto'}："
                  f"编号按**该后端自己的**顺序解释，请自行确认对应关系")

        return PairSource(camera=(left_i, right_i), width=args.width,
                          height=args.height, fps=args.fps,
                          backup_camera=getattr(args, "backup_camera", None),
                          backend=api,
                          left_role=str(left_raw) if left_spec is not None else "",
                          right_role=str(right_raw) if right_spec is not None else "",
                          capture_api=capture_api)
    if getattr(args, "left_video", None) and getattr(args, "right_video", None):
        return PairSource(video=(args.left_video, args.right_video),
                          pairing=getattr(args, "pairing", "canary"),
                          # 兜底 0.033 与类默认一致。原这里是 0.020 —— 同一个
                          # "视频时间戳容差"两个默认值，走 from_args 比直接构造紧 40%。
                          # 不按帧率派生：见 `PairSource.__init__` 里那条注释。
                          timestamp_tolerance_s=float(
                              getattr(args, "timestamp_tolerance", 0.033)))
    if getattr(args, "left_dir", None) and getattr(args, "right_dir", None):
        return PairSource(directory=(args.left_dir, args.right_dir))
    raise RuntimeError("请指定 --left/--right 或 --left-video/--right-video "
                       "或 --left-dir/--right-dir")


__all__ = [
    "PairSource",
    "from_args",
    "resolve_backend",
    "FrameReader",
    "VideoFrameReader",
    "ImageDirFrameReader",
    "CameraPairReader",
    "describe_info",
]
