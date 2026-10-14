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
    resolve_camera_arg,
    select_cameras,
)
from sphere_tracker.sources import CameraSource, LatestFrameCamera

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
CAMERA_WAIT_TIMEOUT = 5.0        # 等相机出新帧的上限（秒），超时就用已有帧继续
CAMERA_POLL_INTERVAL = 0.005     # 轮询间隔，比 30 fps 的帧间隔（33 ms）小得多
# 实时两路各自的采集线程是独立跑的，严格说没有共同帧号。
# 让右路至少前进这么多帧，就认为"两路都已经刷新过"，时间差被压到一帧以内。
# （这是减小时间差，不是消除 —— 真正的同步要靠硬件触发。）
CAMERA_ALIGN_STEPS = 2

_BACKEND_BY_CLI = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}


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
        self.files = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)
        if not self.files:
            raise RuntimeError(f"文件夹里没有图片（支持 {sorted(IMG_EXT)}）：{d}")
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
    """

    kind = "camera"

    def __init__(
        self,
        left_index: int,
        right_index: int,
        width: int = 1920,
        height: int = 1080,
        fps: int = 60,
        backup_camera: int | None = None,
        backend: int | None = None,
    ) -> None:
        if left_index == right_index:
            raise RuntimeError(f"左右相机不能是同一个索引（都是 {left_index}）")
        self.label = f"camera {left_index} / {right_index}"

        def _open(role: str, idx: int, api: int) -> LatestFrameCamera:
            try:
                return LatestFrameCamera(CameraSource(
                    index=idx, width=width, height=height, fps=fps, backend=api))
            except RuntimeError:
                if backup_camera is None or backup_camera == idx:
                    raise
                print(f"[警告] {role}相机 {idx} 打不开，改用备用相机 {backup_camera}")
                print("       注意：备用相机与被替换的那台型号/视野都不同，"
                      "标定结果只对当前这个组合有效")
                return LatestFrameCamera(CameraSource(
                    index=backup_camera, width=width, height=height, fps=fps, backend=api))

        # 两路必须用同一个后端，且要**成对地**挑：一个后端能同时打开两路才算数。
        # 原因（2026-10 实测）：同一进程里混用 DSHOW + MSMF 时，MSMF 那路会
        # "打开成功但抓不到帧"（报 can't grab frame），双目直接废掉；
        # 而且两台相机在两种后端下的索引空间不一定一致，混用还会张冠李戴。
        backends = ([("指定", backend)] if backend is not None
                    else [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)])
        errors: list[str] = []
        self.cams: list[LatestFrameCamera] = []
        for name, api in backends:
            opened: list[LatestFrameCamera] = []
            # ---- 两路**并行**打开（2026-10-14 改）----
            #
            # ⚠️ 原来是**串行**：先开左、开好再开右。而实测本机的一台相机
            # 在 MSMF 下要 **29.8 ~ 51.4 秒**（同型号的另一台只要 0.9 秒；
            # 且单独开也这么慢 —— 已排除带宽竞争）。串行意味着总等待是
            # **两者之和**（1.0 + 37.4 = 38.4s），并行则只有**较慢那个**。
            #
            # 这个差别对用户是直接的：点「标定前置检查」后到看见画面
            # 从 38s 降到 37s 看着不多，但两路都慢时就是**减半**。
            # 而且并行不改变任何语义 —— 两路本来就要用同一个后端，
            # 失败判定与错误信息都保持不变。
            results: list[object] = [None] * 2
            failures: list[BaseException] = []

            def _open_slot(slot: int, role: str, idx: int) -> None:
                try:
                    results[slot] = _open(role, idx, api)
                except BaseException as exc:            # noqa: BLE001
                    failures.append(exc)                # 交给主线程重新抛

            threads = [
                threading.Thread(target=_open_slot, args=(0, "左", left_index),
                                 daemon=True, name="open-left"),
                threading.Thread(target=_open_slot, args=(1, "右", right_index),
                                 daemon=True, name="open-right"),
            ]
            print(f"[{name}] 并行打开两路相机（左 #{left_index} / 右 #{right_index}）…",
                  flush=True)
            t_open = time.perf_counter()
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            opened = [c for c in results if c is not None]
            if failures:
                for c in opened:
                    c.stop()
                exc = failures[0]
                first_line = (str(exc).splitlines() or ["打开失败"])[0]
                errors.append(f"{name}: {first_line}")
                continue
            print(f"[{name}] 两路都已打开，用时 {time.perf_counter() - t_open:.1f}s",
                  flush=True)
            # 每路一个后台采集线程：主循环不再被"等下一帧"拖住，
            # 处理得慢就自动丢帧，永远处理最新画面
            self.cams = opened                          # type: ignore[assignment]
            break
        if not self.cams:
            raise RuntimeError(
                f"左右相机（{left_index} / {right_index}）无法用同一个后端同时打开：\n  "
                + "\n  ".join(errors) +
                "\n最常见的原因：两台型号相同的外接相机挂在同一个 USB 集线器上，"
                "带宽/枚举受限。\n"
                "建议：把两台相机分别插到笔记本机身上**不同的 USB 口**"
                "（不要走同一个集线器），然后再试。")

        self._primed = False
        self.width = self.cams[0].info.width
        self.height = self.cams[0].info.height
        for i, idx in enumerate((left_index, right_index)):
            print(f"相机 {idx}: {describe_info(self.cams[i].info)}")

    @property
    def info(self):
        """两路的 SourceInfo（左, 右）。分辨率可能不同，别假设一样。"""
        return self.cams[0].info, self.cams[1].info

    def read(self) -> tuple[bool, np.ndarray | None]:
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
            left_fresh = self.cams[0].latest()[0] > snap[0][0]
            right_fresh = all(c.latest()[0] > seq + CAMERA_ALIGN_STEPS - 1
                              for c, (seq, _f) in zip(self.cams[1:], snap[1:]))
            if (left_fresh and right_fresh) or time.perf_counter() >= deadline:
                break
            time.sleep(CAMERA_POLL_INTERVAL)

        _s1, a = self.cams[0].latest()
        _s2, b = self.cams[1].latest()
        return (a is not None and b is not None), a, b

    def seek(self, index: int) -> tuple[bool, np.ndarray | None]:
        raise NotImplementedError("实时相机不能按帧号定位（帧号没有共同基准）")

    def release(self) -> None:
        for c in self.cams:
            c.stop()


def _make_reader(value, what: str) -> FrameReader:
    """造一路帧源：文件夹 = 图片序列，其余当视频文件。"""
    p = Path(value)
    if p.is_dir():
        return ImageDirFrameReader(p)
    if p.suffix.lower() in IMG_EXT:
        raise RuntimeError(f"{what} 给的是单张图片（{p}），做不了双目；"
                           f"请给文件夹（directory=）或视频（video=）")
    return VideoFrameReader(p)


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
        fps: int = 60,
        backup_camera: int | None = None,
        backend: int | None = None,
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
        self._left: FrameReader
        self._right: FrameReader

        if camera is not None:
            if self.sync_offset:
                self.warnings.append(
                    f"实时相机不支持 sync_offset（给了 {self.sync_offset:+d}），已忽略："
                    "两路各在自己的线程里跑到最新帧，帧号没有共同基准。"
                    "要真正同步得靠硬件触发，或先用 stereo.sync 标定偏移再录成视频处理。")
                self.sync_offset = 0
            self._left = self._right = CameraPairReader(
                int(camera[0]), int(camera[1]), width=width, height=height, fps=fps,
                backup_camera=backup_camera, backend=backend)
            self.kind = "camera"
        elif video is not None:
            self._left = _make_reader(video[0], "左路")
            self._right = _make_reader(video[1], "右路")
            self.kind = "video"
            n_l, n_r = self._left.frame_count(), self._right.frame_count()
            print(f"视频：{self._left.label} + {self._right.label}，各 {n_l} / {n_r} 帧")
            if n_l and n_r and abs(n_l - n_r) > 1:
                self.warnings.append(
                    f"两段视频帧数不一致（{n_l} vs {n_r}）：会一直取到短的那路结束为止")
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
            left_i = resolve_camera_arg(left_raw)
            right_i = resolve_camera_arg(right_raw)

        # 左右必须落在**两台不同的相机**上。此前没有这道检查，于是
        # "两个角色解析到同一个索引"会静默继续：标定窗口里两路其实是同一台相机，
        # 用户要到一对有效图像都收不到时才发现，白白浪费一轮。
        if left_i == right_i:
            raise RuntimeError(
                f"左右两路解析到了**同一个相机**（索引 #{left_i}）—— 标定无法进行。\n"
                f"  常见原因：\n"
                f"    1) 角色绑定的那台相机不在线（换过 USB 口 / 重插过），"
                f"解析退化到了同一个索引；\n"
                f"    2) 显式给了同一个索引（如 --left 0 --right 0）；\n"
                f"    3) **后端枚举错位**：list_devices() 的 order 来自 PnP 枚举，"
                f"而 cv2.VideoCapture(index) 用的是 OpenCV 各后端自己的枚举，"
                f"两者不保证一致；DSHOW 与 MSMF 之间还会整体错位。\n"
                f"  怎么办：\n"
                f"    - python apps/tools/cameras.py check    看角色有没有漂移\n"
                f"    - python apps/tools/cameras.py list     看当前设备与索引\n"
                f"    - 明确给索引 + 固定同一个后端："
                f"--left 0 --right 1 --backend dshow\n"
                f"    - 开跑后先目视确认：两路必须是**同一场景的左右两个视角**。"
            )

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
        backend_arg = getattr(args, "backend", None)
        api = resolve_backend(backend_arg)
        if api is None:
            # 默认后端取自 sources.INDEX_BACKEND —— 全项目只此一份答案，
            # 免得这里写 MSMF、别处写 DSHOW 又漂移成两种行为。
            from sphere_tracker.sources import INDEX_BACKEND, INDEX_BACKEND_NAME

            api = INDEX_BACKEND
            print(f"[后端] 未指定 --backend，相机源默认用 {INDEX_BACKEND_NAME}"
                  "（序号在 DSHOW/MSMF 下含义不同；MSMF 与设备枚举顺序一致）")
        return PairSource(camera=(left_i, right_i), width=args.width,
                          height=args.height, fps=args.fps,
                          backup_camera=getattr(args, "backup_camera", None),
                          backend=api)
    if getattr(args, "left_video", None) and getattr(args, "right_video", None):
        return PairSource(video=(args.left_video, args.right_video))
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
