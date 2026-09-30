"""视频源：摄像头 / 视频文件 / 单张图片。

Windows 上打开摄像头的坑都在这里处理掉了：
  - 后端优先级 DSHOW -> MSMF -> 默认；
  - 设置 MJPG 才能拿到高分辨率高帧率（很多摄像头默认 YUY2 只能 5-10 fps）；
  - BUFFERSIZE=1 降低延迟（否则你看到的是几百毫秒前的画面）。
"""

from __future__ import annotations

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


class CameraSource(VideoSource):
    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 60,
        fourcc: str = "MJPG",
        backend: int | None = None,
    ) -> None:
        self.cap = None
        self.backend_name = ""
        self.resolution_fallback = False
        candidates = (
            [("given", backend)] if backend is not None else [(n, b) for n, b in WINDOWS_BACKENDS]
        )
        # 分辨率候选：先试用户要的，失败再降级。
        # 为什么必须降级：不少 USB 摄像头对不支持的分辨率不会报错，
        # 而是"打开成功但每帧都返回全黑"，代码不检查就会一直拿到黑屏。
        res_candidates: list[tuple[int, int]] = []
        for cand in ((width, height), (1280, 720), (640, 480), (0, 0)):
            if cand not in res_candidates:
                res_candidates.append(cand)

        errors: list[str] = []
        for name, api in candidates:
            for rw, rh in res_candidates:
                cap = cv2.VideoCapture(index, api)
                if not cap.isOpened():
                    cap.release()
                    errors.append(f"{name}: 打不开")
                    break                      # 该后端整体不可用，换下一个后端
                if fourcc:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*fourcc))
                if rw and rh:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, rw)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, rh)
                cap.set(cv2.CAP_PROP_FPS, fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                frame = None
                for _ in range(12):
                    ret, f = cap.read()
                    if ret and f is not None:
                        frame = f
                        # 真实传感器一定有噪声：整帧完全不变 = 这条流是假的
                        if float(f.std()) > 1.0:
                            break
                    else:
                        time.sleep(0.05)
                if frame is not None and float(frame.std()) > 1.0:
                    self.cap = cap
                    self.backend_name = name
                    self.resolution_fallback = bool(rw and (rw, rh) != (width, height))
                    if self.resolution_fallback:
                        got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        print(
                            f"[提示] 摄像头 {index} 不支持 {width}x{height}（该分辨率下输出全黑），"
                            f"已自动降到 {got_w}x{got_h}。"
                        )
                    break
                cap.release()
                errors.append(
                    f"{name} @ {rw or '默认'}x{rh or ''}: "
                    + ("读不到画面" if frame is None else "输出全黑（分辨率不支持）")
                )
            if self.cap is not None:
                break
        if self.cap is None:
            raise RuntimeError(
                f"无法打开摄像头 {index}，已尝试全部后端与分辨率。\n  " + "\n  ".join(errors) +
                "\n可能原因：\n"
                "  1) 被别的程序占用（QQ / 微信 / 腾讯会议 / 浏览器 / Windows Hello）；\n"
                "  2) 是红外摄像头（IR，无彩色输出）；\n"
                "  3) 镜头上有物理遮挡（隐私滑块 / 盖子合上）；\n"
                "  4) USB 供电或带宽不足（换一个直接插在机身上的 USB 口试试）。"
            )
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        real_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.fps_measured = False
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


def list_cameras(max_index: int = 5, warmup: int = 3) -> list[tuple[int, int, int, float]]:
    """扫描可用摄像头，返回 [(索引, 宽, 高, fps), ...]。"""
    found: list[tuple[int, int, int, float]] = []
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            time.sleep(0.05)
            continue
        ok = False
        for _ in range(warmup):
            ret, frame = cap.read()
            if ret and frame is not None:
                ok = True
                break
            time.sleep(0.05)
        if ok:
            found.append(
                (
                    idx,
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    float(cap.get(cv2.CAP_PROP_FPS)),
                )
            )
        cap.release()
        time.sleep(0.1)
    return found


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
