"""视频源：摄像头 / 视频文件 / 单张图片。

Windows 上打开摄像头的坑都在这里处理掉了：
  - 后端优先级 DSHOW -> MSMF -> 默认；
  - 设置 MJPG 才能拿到高分辨率高帧率（很多摄像头默认 YUY2 只能 5-10 fps）；
  - BUFFERSIZE=1 降低延迟（否则你看到的是几百毫秒前的画面）。
"""

from __future__ import annotations

import time
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
        candidates = (
            [("given", backend)] if backend is not None else [(n, b) for n, b in WINDOWS_BACKENDS]
        )
        errors: list[str] = []
        for name, api in candidates:
            cap = cv2.VideoCapture(index, api)
            if not cap.isOpened():
                cap.release()
                errors.append(f"{name}: 打不开")
                continue
            if fourcc:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok = False
            for _ in range(10):
                ret, frame = cap.read()
                if ret and frame is not None:
                    ok = True
                    break
                time.sleep(0.05)
            if ok:
                self.cap = cap
                self.backend_name = name
                break
            cap.release()
            errors.append(f"{name}: 打开但读不到画面")
        if self.cap is None:
            raise RuntimeError(
                f"无法打开摄像头 {index}。\n  " + "\n  ".join(errors) +
                "\n提示：确认没有别的程序（QQ/微信/腾讯会议/浏览器）正在占用摄像头。"
            )
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        real_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.info = SourceInfo("camera", w, h, real_fps, self.backend_name, str(index))
        self._interval = 1.0 / real_fps if real_fps and real_fps > 1 else 0.0
        self._emit_interval = 0.0
        self._next_t = 0.0

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

    def __init__(self, paths: list[str | Path], fps: float = 30.0) -> None:
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
