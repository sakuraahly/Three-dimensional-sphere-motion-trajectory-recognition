r"""Windows Media Foundation 采集后端：按设备 symbolic link 打开，不依赖 OpenCV 编号。

## 为什么要有它

`cv2.VideoCapture(N)` 只认整数编号；PnP 位次、MSMF 编号、DSHOW 编号是三套空间，
拔插后会漂。两台同型号相机在软件层面又不可区分，于是"名字对、画面错"很容易发生。

Media Foundation 的 `IMFActivate` 自带
`MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK`，里面直接包含 PnP
设备实例/端口，例如：

    \\?\usb#vid_1bcf&pid_2a25&mi_00#6&212acdc8&0&0000#{e532...}\global

因此可以按 **symlink** 精确打开指定设备，完全绕开 OpenCV 的编号映射。

## 边界

- 只支持 Windows（非 Windows 下 `list_devices()` 返回空、打开抛错）。
- 不按 FriendlyName 猜设备；名字可重复/可改，只作显示。
- 换 USB 口后 symlink 会变；这时应由 `cameras.py check` 报漂移并重新绑定，
  本模块不自动按名字找回。
- 这是**新增路径**；`sphere_tracker.sources.CameraSource` 的 OpenCV 路径保留。

## 依赖

只用标准库 + `ctypes` + OpenCV（解码 MJPG）；不使用 comtypes/pygrabber。
"""

from __future__ import annotations

import atexit
import ctypes
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from .sources import SourceInfo, VideoSource

_IS_WINDOWS = sys.platform.startswith("win")
HRESULT = ctypes.c_int32
MF_VERSION = 0x00020070
COINIT_MULTITHREADED = 0x0
RPC_E_CHANGED_MODE = 0x80010106
FIRST_VIDEO = 0xFFFFFFFC
MF_SOURCE_READERF_STREAMTICK = 0x100
MF_SOURCE_READERF_ENDOFSTREAM = 0x2
MF_SOURCE_READERF_ERROR = 0x1

#: MF 原生 subtype 的 Data1（GUID 前 4 字节，小端显示即 FourCC）。
SUBTYPE_RGB32 = 0x00000016
SUBTYPE_NV12 = 0x3231564E
SUBTYPE_MJPG = 0x47504A4D
SUBTYPE_YUY2 = 0x32595559

# 两台 USB 2.0 相机同时 1080p30 时，未压缩 NV12/YUY2 会先吃满带宽；
# MJPG 压缩流才是双目能同时跑到 25–30 fps 的关键（OpenCV 路径同样默认 MJPG）。
_SUBTYPE_PRIORITY = {
    SUBTYPE_MJPG: 0,
    SUBTYPE_NV12: 1,
    SUBTYPE_YUY2: 2,
    SUBTYPE_RGB32: 3,
}


class MFError(RuntimeError):
    """Media Foundation 枚举/打开/读帧失败。"""


class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    ]

    @classmethod
    def parse(cls, text: str) -> "GUID":
        u = uuid.UUID(text)
        out = cls()
        out.Data1, out.Data2, out.Data3 = u.fields[0], u.fields[1], u.fields[2]
        for i, b in enumerate(u.bytes[8:]):
            out.Data4[i] = b
        return out


@dataclass(frozen=True)
class MFDevice:
    """一台 MF 视频设备（不打开相机）。"""

    index: int
    name: str
    symbolic_link: str
    identity: str = ""
    port: str = ""
    serial: str = ""


@dataclass(frozen=True)
class MediaType:
    """一个 native media type 的关键字段。"""

    subtype: int
    width: int
    height: int
    fps_num: int
    fps_den: int

    @property
    def fps(self) -> float:
        return (float(self.fps_num) / float(self.fps_den)
                if self.fps_den else 0.0)

    @property
    def label(self) -> str:
        return f"{self.width}x{self.height}@{self.fps:g} subtype=0x{self.subtype:08x}"


# --------------------------------------------------------------------------- #
# GUID / 函数原型
# --------------------------------------------------------------------------- #
G_SOURCE_TYPE = GUID.parse("c60ac5fe-252a-478f-a0ef-bc8fa5f7cad3")
G_VIDCAP = GUID.parse("8ac3587a-4ae7-42d8-99e0-0a6013eef90f")
G_FRIENDLY = GUID.parse("60d0e559-52f8-4fa2-bbce-acdb8c772500")
G_SYMLINK = GUID.parse("58f0aad8-22bf-4f8a-bb3d-d2c4978c6e2f")
G_IID_MEDIASOURCE = GUID.parse("279a808d-aec7-40c8-9c6b-a6b492c78a66")
G_MT_SUBTYPE = GUID.parse("f7e34c9a-42e8-4714-b74b-cb29d72c35e5")
G_FRAME_SIZE = GUID.parse("1652c33d-d6b2-4012-b834-72030849a37d")
G_FRAME_RATE = GUID.parse("c459a2e8-3d2c-4e44-b132-fee5156c7bb0")
G_ENABLE_PROCESSING = GUID.parse("fb394f3d-ccf1-42ee-bbb3-f9b845d5681d")

_ole32 = None
_mfplat = None
_mf = None
_mfreadwrite = None

if _IS_WINDOWS:
    _ole32 = ctypes.WinDLL("ole32")
    _mfplat = ctypes.WinDLL("mfplat")
    _mf = ctypes.WinDLL("mf")
    _mfreadwrite = ctypes.WinDLL("mfreadwrite")

    _ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    _ole32.CoInitializeEx.restype = HRESULT
    _ole32.CoUninitialize.argtypes = []
    _ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]
    _ole32.CoTaskMemFree.restype = None
    _mfplat.MFStartup.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
    _mfplat.MFStartup.restype = HRESULT
    _mfplat.MFShutdown.argtypes = []
    _mfplat.MFShutdown.restype = HRESULT
    _mfplat.MFCreateAttributes.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                           ctypes.c_uint32]
    _mfplat.MFCreateAttributes.restype = HRESULT
    _mf.MFEnumDeviceSources.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)),
        ctypes.POINTER(ctypes.c_uint32)]
    _mf.MFEnumDeviceSources.restype = HRESULT
    _mfreadwrite.MFCreateSourceReaderFromMediaSource.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    _mfreadwrite.MFCreateSourceReaderFromMediaSource.restype = HRESULT

_mf_lock = threading.Lock()
_mf_ready = False
_mf_shutdown_registered = False


def _check(hr: int, what: str) -> None:
    if hr < 0:
        raise MFError(f"{what} 失败：HRESULT 0x{hr & 0xFFFFFFFF:08X}")


def _ensure_mf() -> None:
    global _mf_ready, _mf_shutdown_registered
    if not _IS_WINDOWS:
        raise MFError("Media Foundation 只在 Windows 上可用")
    with _mf_lock:
        if _mf_ready:
            return
        hr = _ole32.CoInitializeEx(None, COINIT_MULTITHREADED)
        # S_FALSE / RPC_E_CHANGED_MODE 都表示 COM 已经被初始化；MF 仍可用。
        if hr < 0 and hr != RPC_E_CHANGED_MODE:
            raise MFError(f"CoInitializeEx 失败：0x{hr & 0xFFFFFFFF:08X}")
        hr = _mfplat.MFStartup(MF_VERSION, 0)
        if hr < 0:
            raise MFError(f"MFStartup 失败：0x{hr & 0xFFFFFFFF:08X}")
        _mf_ready = True
        if not _mf_shutdown_registered:
            atexit.register(_shutdown_mf)
            _mf_shutdown_registered = True


def _shutdown_mf() -> None:
    global _mf_ready
    with _mf_lock:
        if not _mf_ready:
            return
        try:
            _mfplat.MFShutdown()
        except Exception:                                    # noqa: BLE001
            pass
        try:
            _ole32.CoUninitialize()
        except Exception:                                    # noqa: BLE001
            pass
        _mf_ready = False


def _co_init_thread() -> None:
    """每个调用 MF 的线程都要有 COM 初始化（MTA；重复调用无害）。"""
    if not _IS_WINDOWS:
        return
    hr = _ole32.CoInitializeEx(None, COINIT_MULTITHREADED)
    if hr < 0 and hr != RPC_E_CHANGED_MODE:
        raise MFError(f"线程 CoInitializeEx 失败：0x{hr & 0xFFFFFFFF:08X}")


def _vt(ptr, index: int, restype, *argtypes):
    table = ctypes.cast(ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(table[index])


def _release(ptr) -> None:
    if ptr:
        try:
            _vt(ptr, 2, ctypes.c_ulong)(ptr)
        except Exception:                                    # noqa: BLE001
            pass


def _get_string(ptr, key: GUID) -> str:
    value = ctypes.c_wchar_p()
    length = ctypes.c_uint32()
    hr = _vt(ptr, 13, HRESULT, ctypes.POINTER(GUID),
             ctypes.POINTER(ctypes.c_wchar_p),
             ctypes.POINTER(ctypes.c_uint32))(
        ptr, ctypes.byref(key), ctypes.byref(value), ctypes.byref(length))
    if hr < 0 or not value.value:
        return ""
    text = value.value
    _ole32.CoTaskMemFree(value)
    return text


def _set_guid(ptr, key: GUID, value: GUID) -> None:
    _check(_vt(ptr, 24, HRESULT, ctypes.POINTER(GUID), ctypes.POINTER(GUID))(
        ptr, ctypes.byref(key), ctypes.byref(value)), "SetGUID")


def _set_u32(ptr, key: GUID, value: int) -> None:
    _check(_vt(ptr, 21, HRESULT, ctypes.POINTER(GUID), ctypes.c_uint32)(
        ptr, ctypes.byref(key), value), "SetUINT32")


def _attr_guid(ptr, key: GUID) -> int:
    out = GUID()
    hr = _vt(ptr, 10, HRESULT, ctypes.POINTER(GUID), ctypes.POINTER(GUID))(
        ptr, ctypes.byref(key), ctypes.byref(out))
    if hr < 0:
        return 0
    return int(out.Data1)


def _attr_u64(ptr, key: GUID) -> int:
    out = ctypes.c_uint64()
    hr = _vt(ptr, 8, HRESULT, ctypes.POINTER(GUID),
             ctypes.POINTER(ctypes.c_uint64))(
        ptr, ctypes.byref(key), ctypes.byref(out))
    return int(out.value) if hr >= 0 else 0


def _current_media_type(reader) -> int:
    """返回当前位置的 subtype（Data1）；取不到返回 0。"""
    mt = ctypes.c_void_p()
    hr = _vt(reader, 6, HRESULT, ctypes.c_uint32,
             ctypes.POINTER(ctypes.c_void_p))(
        reader, FIRST_VIDEO, ctypes.byref(mt))
    if hr < 0 or not mt:
        return 0
    try:
        return _attr_guid(mt, G_MT_SUBTYPE)
    finally:
        _release(mt)


# --------------------------------------------------------------------------- #
# symlink <-> identity/port
# --------------------------------------------------------------------------- #
def parse_symbolic_link(link: str) -> tuple[str, str, str]:
    r"""`\\?\usb#vid_..&pid_..&mi_00#<port>#{...}\global` -> (identity, port, serial).

    解析失败返回 ("", "", "")；不猜。
    """
    text = str(link or "").strip()
    if not text:
        return "", "", ""
    body = text
    for prefix in ("\\\\?\\", "\\??\\"):
        if body.startswith(prefix):
            body = body[len(prefix):]
            break
    pieces = body.split("#")
    if len(pieces) < 3 or not pieces[0].lower().startswith("usb"):
        return "", "", ""
    vidpid = pieces[1].strip()
    port = pieces[2].strip()
    identity = f"USB\\{vidpid.upper()}\\{port.upper()}"
    serial = ""
    # 有些设备 third piece 是序列号；aoni 这里是 6&... 端口，保留为空。
    if port and not port.lower().startswith("6&"):
        serial = port
    return identity, port, serial


def _enumerate() -> list[tuple[MFDevice, ctypes.c_void_p]]:
    """枚举 MF 设备，返回 `(MFDevice, IMFActivate*)`；调用方负责 release。"""
    _ensure_mf()
    attrs = ctypes.c_void_p()
    _check(_mfplat.MFCreateAttributes(ctypes.byref(attrs), 1), "MFCreateAttributes")
    try:
        _set_guid(attrs, G_SOURCE_TYPE, G_VIDCAP)
        devices = ctypes.POINTER(ctypes.c_void_p)()
        count = ctypes.c_uint32()
        _check(_mf.MFEnumDeviceSources(
            attrs, ctypes.byref(devices), ctypes.byref(count)),
            "MFEnumDeviceSources")
        out: list[tuple[MFDevice, ctypes.c_void_p]] = []
        try:
            for i in range(int(count.value)):
                act = devices[i]
                link = _get_string(act, G_SYMLINK)
                name = _get_string(act, G_FRIENDLY)
                identity, port, serial = parse_symbolic_link(link)
                out.append((MFDevice(index=i, name=name, symbolic_link=link,
                                     identity=identity, port=port, serial=serial),
                            act))
        except Exception:
            for _, act in out:
                _release(act)
            raise
        finally:
            _ole32.CoTaskMemFree(devices)
        return out
    finally:
        _release(attrs)


def list_devices() -> list[MFDevice]:
    """列出 MF 视频设备（不打开相机）。"""
    pairs = _enumerate()
    try:
        return [dev for dev, _ in pairs]
    finally:
        for _, act in pairs:
            _release(act)


def find_device(*, identity: str = "", port: str = "",
                symbolic_link: str = "",
                devices: Iterable[MFDevice] | None = None) -> MFDevice:
    """按 symlink/端口/identity 精确匹配；0 个或多个命中都抛错。"""
    devs = list(devices) if devices is not None else list_devices()
    link = str(symbolic_link or "").strip().casefold()
    ident = str(identity or "").strip().casefold()
    prt = str(port or "").strip().casefold()
    hits: list[MFDevice] = []
    for dev in devs:
        if link and dev.symbolic_link.casefold() == link:
            hits.append(dev)
            continue
        if prt and (dev.port.casefold() == prt
                    or prt in dev.symbolic_link.casefold()):
            hits.append(dev)
            continue
        if ident and dev.identity.casefold() == ident:
            hits.append(dev)
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise MFError(
            "按当前 identity/端口匹配到多台 MF 设备，拒绝猜："
            + "；".join(f"#{d.index} {d.symbolic_link}" for d in hits))
    raise MFError(
        "没有匹配到 MF 设备："
        + (f"identity={identity!r} port={port!r} link={symbolic_link!r}；" if
           (identity or port or symbolic_link) else "")
        + "当前可用："
        + ("；".join(f"#{d.index} {d.symbolic_link}" for d in devs)
           or "（MF 枚举为空）"))


# --------------------------------------------------------------------------- #
# native type 选择 / 解码（纯函数，可单测）
# --------------------------------------------------------------------------- #
def select_native_type(types: Iterable[MediaType], width: int,
                       height: int, fps: float = 0.0) -> MediaType:
    """选最接近请求的 native type；分辨率优先，其次 fps，再按 subtype 偏好。"""
    items = [t for t in types if t.width > 0 and t.height > 0]
    if not items:
        raise MFError("MF 没有可用的 native media type")
    target_px = max(1, int(width) * int(height)) if width > 0 and height > 0 else 0
    exact = [t for t in items if (t.width, t.height) == (int(width), int(height))]
    pool = exact or items
    if not exact and target_px:
        best_px = min(abs(t.width * t.height - target_px) for t in pool)
        pool = [t for t in pool if abs(t.width * t.height - target_px) == best_px]
    target_fps = float(fps or 0.0)
    if target_fps > 0:
        best_dt = min(abs(t.fps - target_fps) for t in pool)
        pool = [t for t in pool if abs(t.fps - target_fps) == best_dt]
    return min(pool, key=lambda t: (
        -t.width * t.height,
        -t.fps,
        _SUBTYPE_PRIORITY.get(t.subtype, 99),
    ))


def decode_frame(raw: bytes | np.ndarray, subtype: int, width: int,
                 height: int) -> np.ndarray:
    """把 MF sample 解成 BGR `np.ndarray`；未知格式抛 `MFError`。"""
    w, h = int(width), int(height)
    if w <= 0 or h <= 0:
        raise MFError(f"非法帧尺寸 {w}x{h}")
    arr = np.frombuffer(raw, dtype=np.uint8) if isinstance(raw, (bytes, bytearray)) \
        else np.asarray(raw, dtype=np.uint8)
    if subtype == SUBTYPE_MJPG:
        out = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if out is None:
            raise MFError("MJPG 解码失败")
        return out
    if subtype == SUBTYPE_NV12:
        need = w * h * 3 // 2
        if arr.size < need:
            raise MFError(f"NV12 数据不足：{arr.size} < {need}")
        stride = arr.size // (h * 3 // 2)
        if stride == w:
            yuv = arr[:need].reshape(h * 3 // 2, w)
        else:
            yuv = np.empty((h * 3 // 2, w), np.uint8)
            y = arr[:stride * h].reshape(h, stride)[:, :w]
            uv = arr[stride * h:stride * h + stride * (h // 2)].reshape(
                h // 2, stride)[:, :w]
            yuv[:h] = y
            yuv[h:] = uv
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
    if subtype == SUBTYPE_YUY2:
        need = w * h * 2
        if arr.size < need:
            raise MFError(f"YUY2 数据不足：{arr.size} < {need}")
        stride = arr.size // h
        packed = arr[:stride * h].reshape(h, stride // 2, 2)[:, :w]
        return cv2.cvtColor(packed, cv2.COLOR_YUV2BGR_YUY2)
    if subtype == SUBTYPE_RGB32:
        need = w * h * 4
        if arr.size < need:
            raise MFError(f"RGB32 数据不足：{arr.size} < {need}")
        stride = arr.size // h
        return arr[:stride * h].reshape(h, stride // 4, 4)[:, :w, :3].copy()
    raise MFError(f"不支持的 MF subtype 0x{subtype:08X}")


# --------------------------------------------------------------------------- #
# VideoSource
# --------------------------------------------------------------------------- #
class MFCameraSource(VideoSource):
    """按 symbolic link 打开的 MF 相机（首次 read 时才真正打开）。"""

    def __init__(
        self,
        *,
        identity: str = "",
        port: str = "",
        symbolic_link: str = "",
        width: int = 1920,
        height: int = 1080,
        fps: float = 30.0,
    ) -> None:
        if not (identity or port or symbolic_link):
            raise MFError("MFCameraSource 需要 identity / port / symbolic_link 之一")
        self._identity = identity
        self._port = port
        self._symbolic_link = symbolic_link
        self._req_w, self._req_h = int(width), int(height)
        self._req_fps = float(fps or 0.0)
        self.info = SourceInfo(
            kind="camera", width=self._req_w, height=self._req_h,
            fps=self._req_fps, backend="MF", path=symbolic_link)
        self.resolution_fallback = False
        self._opened = False
        self._failed = False
        self._reader = None
        self._reader_attrs = None
        self._source = None
        self._subtype = 0
        self._w = self._req_w
        self._h = self._req_h
        self._native_fps = self._req_fps
        self._lock = threading.Lock()
        # 构造期只做**枚举/匹配**，不打开相机：这样 identity/端口写错会在
        # 创建时立刻报出来，auto 才能决定是否回退 OpenCV。真正打开在 read()。
        self._device = find_device(
            identity=identity, port=port, symbolic_link=symbolic_link)

    # ------------------------------------------------------------------
    def _open(self) -> None:
        _co_init_thread()
        pairs = _enumerate()
        target = None
        try:
            target = find_device(
                identity=self._identity, port=self._port,
                symbolic_link=self._symbolic_link,
                devices=[dev for dev, _ in pairs])
            # 找到对应的 activate 指针
            act = next(a for d, a in pairs if d == target)
        except Exception:
            for _, a in pairs:
                _release(a)
            raise
        for d, a in pairs:
            if a is not act:
                _release(a)
        source = ctypes.c_void_p()
        try:
            _check(_vt(act, 33, HRESULT, ctypes.POINTER(GUID),
                       ctypes.POINTER(ctypes.c_void_p))(
                act, ctypes.byref(G_IID_MEDIASOURCE), ctypes.byref(source)),
                "ActivateObject")
        finally:
            _release(act)
        reader_attrs = ctypes.c_void_p()
        _check(_mfplat.MFCreateAttributes(ctypes.byref(reader_attrs), 1),
               "MFCreateAttributes(reader)")
        _set_u32(reader_attrs, G_ENABLE_PROCESSING, 1)
        reader = ctypes.c_void_p()
        try:
            _check(_mfreadwrite.MFCreateSourceReaderFromMediaSource(
                source, reader_attrs, ctypes.byref(reader)),
                "MFCreateSourceReaderFromMediaSource")
        except Exception:
            _release(reader_attrs)
            _release(source)
            raise
        types: list[tuple[MediaType, ctypes.c_void_p]] = []
        try:
            idx = 0
            while True:
                mt = ctypes.c_void_p()
                hr = _vt(reader, 5, HRESULT, ctypes.c_uint32, ctypes.c_uint32,
                         ctypes.POINTER(ctypes.c_void_p))(
                    reader, FIRST_VIDEO, idx, ctypes.byref(mt))
                if hr < 0 or not mt:
                    break
                size = _attr_u64(mt, G_FRAME_SIZE)
                rate = _attr_u64(mt, G_FRAME_RATE)
                types.append((MediaType(
                    subtype=_attr_guid(mt, G_MT_SUBTYPE),
                    width=size >> 32, height=size & 0xFFFFFFFF,
                    fps_num=rate >> 32, fps_den=rate & 0xFFFFFFFF,
                ), mt))
                idx += 1
            if not types:
                raise MFError("MF 没有 native media type")
            chosen = select_native_type(
                [t for t, _ in types], self._req_w, self._req_h, self._req_fps)
            chosen_ptr = next(p for t, p in types if t == chosen)
            hr = _vt(reader, 7, HRESULT, ctypes.c_uint32,
                     ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p)(
                reader, FIRST_VIDEO, None, chosen_ptr)
            _check(hr, f"SetCurrentMediaType({chosen.label})")
            self._subtype = chosen.subtype
            self._w, self._h = chosen.width, chosen.height
            self._native_fps = chosen.fps
            self.resolution_fallback = (
                (chosen.width, chosen.height) != (self._req_w, self._req_h))
            self.info = SourceInfo(
                kind="camera", width=self._w, height=self._h,
                fps=self._native_fps, backend="MF",
                path=target.symbolic_link)
            self._reader = reader
            self._reader_attrs = reader_attrs
            self._source = source
            self._opened = True
            print(f"[MF] {target.port or target.symbolic_link} -> {chosen.label}"
                  + ("（请求档被回退）" if self.resolution_fallback else ""),
                  flush=True)
        except Exception:
            _release(reader)
            _release(reader_attrs)
            _release(source)
            raise
        finally:
            for _t, ptr in types:
                _release(ptr)

    # ------------------------------------------------------------------
    def read(self) -> tuple[bool, np.ndarray | None, float]:
        _co_init_thread()
        with self._lock:
            if not self._opened:
                if self._failed:
                    return False, None, time.perf_counter()
                try:
                    self._open()
                except Exception as exc:                    # noqa: BLE001
                    self._failed = True
                    print(f"[MF] 打开失败：{exc}", flush=True)
                    return False, None, time.perf_counter()
        actual = ctypes.c_uint32()
        flags = ctypes.c_uint32()
        ts = ctypes.c_longlong()
        sample = ctypes.c_void_p()
        hr = _vt(self._reader, 9, HRESULT, ctypes.c_uint32, ctypes.c_uint32,
                 ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
                 ctypes.POINTER(ctypes.c_longlong),
                 ctypes.POINTER(ctypes.c_void_p))(
            self._reader, FIRST_VIDEO, 0, ctypes.byref(actual), ctypes.byref(flags),
            ctypes.byref(ts), ctypes.byref(sample))
        if hr < 0:
            return False, None, time.perf_counter()
        if not sample:
            return False, None, time.perf_counter()
        if int(flags.value) & MF_SOURCE_READERF_STREAMTICK:
            _release(sample)
            return False, None, time.perf_counter()
        buf = ctypes.c_void_p()
        try:
            _check(_vt(sample, 41, HRESULT, ctypes.POINTER(ctypes.c_void_p))(
                sample, ctypes.byref(buf)), "ConvertToContiguousBuffer")
            data = ctypes.POINTER(ctypes.c_ubyte)()
            max_len = ctypes.c_uint32()
            cur_len = ctypes.c_uint32()
            _check(_vt(buf, 3, HRESULT,
                       ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
                       ctypes.POINTER(ctypes.c_uint32),
                       ctypes.POINTER(ctypes.c_uint32))(
                buf, ctypes.byref(data), ctypes.byref(max_len),
                ctypes.byref(cur_len)), "IMFMediaBuffer.Lock")
            try:
                raw = np.ctypeslib.as_array(
                    data, shape=(int(cur_len.value),)).copy()
            finally:
                _vt(buf, 4, HRESULT)(buf)
        finally:
            _release(buf)
            _release(sample)
        try:
            frame = decode_frame(raw, self._subtype, self._w, self._h)
        except Exception as exc:                            # noqa: BLE001
            print(f"[MF] 解码失败：{exc}", flush=True)
            return False, None, time.perf_counter()
        return True, frame, time.perf_counter()

    # ------------------------------------------------------------------
    def release(self) -> None:
        with self._lock:
            for ptr in (self._reader, self._reader_attrs, self._source):
                _release(ptr)
            self._reader = None
            self._reader_attrs = None
            self._source = None
            self._opened = False


__all__ = [
    "MFDevice", "MFError", "MFCameraSource", "MediaType",
    "decode_frame", "find_device", "list_devices",
    "parse_symbolic_link", "select_native_type",
]
