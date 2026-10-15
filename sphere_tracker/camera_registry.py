"""摄像头设备管理：把"相机身份"从"索引"里解耦出来。

## 为什么需要这个模块

`cv2.VideoCapture(index)` 的编号由操作系统枚举顺序决定，**会漂移**：
换一次 USB 口、多插一个设备，编号 1 可能从"外接相机"变成"笔记本内置"。
启动器如果把 `--left 0 --right 2` 写死，索引一变就会**悄悄用错相机**。

本模块把"相机身份"记下来（Windows 上是 PnP `InstanceId` / USB 端口），
每次启动再解析成**当前**索引，从而不受连接顺序影响。

## 必须说清楚的边界（不要把它当真理）

- **可靠**：某个索引能不能打开、画面是不是全黑常量流、真实分辨率（`probe_index` 实测）；
  Windows 上设备清单的 `Present` / `InstanceId` / 端口（默认走 `pnputil`，未装/被拒时兜底 PowerShell）。
- **启发式**：`DeviceEntry.order`（枚举位次）**被当作** cv2 索引 ——
  OpenCV 不提供"索引 N 是哪个物理设备"的查询接口，只能靠枚举顺序去推。
  所以有 `apps/tools/cameras.py identify`（目视确认后写死身份）和 `check`（检测漂移）来兜底。
- 非 Windows 平台 `list_devices()` 返回空列表，整套功能退化为"按索引"（不报错）。

## 性能

`list_devices()` 每次启动都会跑。采集设备清单用 `pnputil /enum-devices /connected`
（原生命令，本机实测 ~0.1 s）；`Get-PnpDevice`（PowerShell）在本机要 **~10 s**，
只在 pnputil 不可用时兜底 —— 绝不让用户每次启动等 10 秒。

## 配置

`config/cameras.json`（本机相关，已被 `.gitignore` 覆盖）：

    {
      "v": 1,
      "roles": {
        "left":  {"identity": "USB\\VID_1BCF&PID_2A25&MI_00\\6&31324E2C&0&0000",
                  "port": "Port_#0001.Hub_#0003", "name": "aoni webcam A25",
                  "index": 0, "verified": true}
      },
      "ignored": [
        {"identity": "USB\\VID_30C9&PID_00EC&MI_00\\...", "port": "...",
         "name": "Integrated Camera", "index": 2, "verified": true}
      ],
      "settings": {"prefer_external": true, "avoid_names": ["Integrated Camera"],
                   "min_width": 1280, "min_height": 720, "min_fps": 15.0, "max_index": 6}
    }

`ignored` 是**「无关」清单**：用户明确标记"这台不要用"（典型是笔记本内置相机与红外相机）。
它和 `roles` 一样**记身份而不是编号**，所以重插 USB 之后依然有效。语义：

- `select_cameras()` / `rank_candidates()` 把标了「无关」的相机排到最后，
  只有"其它相机都不够用"时才动用，并且**一定会打印警告**（静默用错相机比报错更糟）；
- 一台相机不能**既**是左/右/主**又**是「无关」：`mark_ignored()` 会顺手把它从角色里摘掉，
  反过来 `set_role()` 之后旧的无关联记录也要清（由调用方 `cameras.py` 负责）；
- `format_device_table(..., reg=reg)` 会打出 `[无关]` 标记，选相机时看得见。
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

_CFG = Path(__file__).resolve().parents[1] / "config" / "cameras.json"

# DirectShow 的"视频捕获设备"类 —— DSHOW 后端就是按这个键的顺序枚举设备的
DEVICE_CLASS_KEY = (
    r"SYSTEM\CurrentControlSet\Control\DeviceClasses"
    r"\{e5323777-f976-4f5b-9b55-b94699c46e44}"
)

AUTO_TOKEN = "auto"

DEFAULT_SETTINGS: dict = {
    "prefer_external": True,
    "avoid_names": ["Integrated Camera", "Integrated IR Camera"],
    "min_width": 1280,
    "min_height": 720,
    "min_fps": 15.0,
    "max_index": 6,
}


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DeviceEntry:
    """一台相机（系统视角）。order 就是我们"相信"的 cv2 索引，见模块 docstring。"""

    order: int
    name: str = ""
    identity: str = ""
    port: str = ""
    present: bool = True
    source: str = "index"  # system（pnputil 列表）/ registry（只读到注册表）/ index


@dataclass
class RoleSpec:
    """配置里一个角色（left / right / main …）期望用哪台相机。"""

    identity: str = ""
    port: str = ""
    name: str = ""
    index: int | None = None
    verified: bool = False


@dataclass
class Registry:
    v: int = 1
    roles: dict[str, RoleSpec] = field(default_factory=dict)
    # 「无关」清单：明确标记"这台不要用"。内置 / 红外相机标一次就再也不用管，
    # 自动挑相机（select_cameras）会跳过它们，避免标定/实时偷偷用到错的设备。
    # 结构与 roles 一样（记身份而不是编号），所以重插 USB 也不会失效。
    ignored: list[RoleSpec] = field(default_factory=list)
    settings: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ProbeResult:
    index: int
    ok: bool
    width: int = 0
    height: int = 0
    fps: float = 0.0
    std: float = 0.0
    error: str = ""
    entry: DeviceEntry | None = None


@dataclass(frozen=True)
class ResolvedCamera:
    role: str
    index: int
    how: str  # identity / port / name / manual / index-fallback
    entry: DeviceEntry | None = None
    warnings: tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# 平台
# --------------------------------------------------------------------------- #
def current_platform() -> str:
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform.startswith("darwin"):
        return "darwin"
    if sys.platform.startswith("linux"):
        return "linux"
    return "other"


def _call(fn, *args, default=None, **kwargs):
    """调用外部采集器，任何异常都吞掉退回默认值（身份识别不该拖垮主流程）。"""
    try:
        return fn(*args, **kwargs)
    except Exception:
        return default


# --------------------------------------------------------------------------- #
# Windows：枚举真实摄像头设备
# --------------------------------------------------------------------------- #
def parse_windows_instance(instance: str) -> tuple[str, str]:
    """把 PnP InstanceId（或 DeviceClasses 子键名）归一化成 (identity, 端口段)。

        'USB\\VID_1BCF&PID_2A25&MI_00\\6&31324E2C&0&0000'
            -> ('USB\\VID_1BCF&PID_2A25&MI_00\\6&31324E2C&0&0000', '6&31324E2C&0&0000')
        '##?#USB#VID_1BCF&PID_2A25&MI_00#6&31324e2c&0&0000#{e5323777-...}'
            -> (同上，大小写归一，'6&31324e2c&0&0000')

    端口段就是最后那节：同型号相机插在不同 USB 口上，这一节不同 —— 这就是"按端口识别"的依据。
    """
    text = str(instance or "").strip()
    if not text:
        return "", ""
    text = re.sub(r"#\{[0-9a-fA-F-]{36}\}\s*$", "", text)  # 去掉结尾的 {类 GUID}
    text = text.replace("#", "\\")
    text = re.sub(r"^\\+\?\\*", "", text)  # 去掉 \\?\ / ##?# 前缀
    text = text.strip("\\")
    parts = [p for p in text.split("\\") if p]
    port = parts[-1] if len(parts) >= 2 else ""
    return text, port


def read_windows_registry_instances() -> list[str]:
    """按 DeviceClasses 子键顺序返回设备实例列表（DSHOW 的枚举顺序）。读不到返回 []。

    只收**像设备实例**的子键（一定带结尾的 `#{类 GUID}`）——
    这个类键下还有一个叫 `Properties` 的普通子键，不过滤会被当成一台相机。
    """
    try:
        import winreg  # type: ignore
    except ImportError:
        return []

    out: list[str] = []
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, DEVICE_CLASS_KEY) as key:
            for i in range(winreg.QueryInfoKey(key)[0]):
                sub = winreg.EnumKey(key, i)
                if "#{" not in sub:          # 设备实例才带类 GUID；Properties 之类直接跳过
                    continue
                ident, _port = parse_windows_instance(sub)
                if ident:
                    out.append(ident)
    except OSError:
        return []
    return out


def _run_command(args: list[str], timeout: float = 8.0) -> tuple[int, str] | None:
    """跑一个外部命令拿 (returncode, stdout)；没有该命令/超时/被拒一律 None，绝不抛。"""
    try:
        proc = subprocess.run(
            args, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.returncode, proc.stdout


# `pnputil /enum-devices /connected` 只列**当前在线**的设备，且是原生命令（~0.1 s）；
# 相比 `Get-PnpDevice`（本机实测 ~10 s，WMI 太慢）快两个数量级，
# 所以作为默认采集方式，PowerShell 只在 pnputil 不可用时兜底。
_PNPUTIL_ARGS = ["pnputil", "/enum-devices", "/connected", "/class", "Camera"]


def _parse_pnputil(text: str) -> list[dict]:
    """解析 pnputil 输出。按空行分块，每块字段顺序固定（Instance ID、Device Description、…），
    所以只取每块**前两个字段** —— **不认字段名**，因为非英文 Windows 会把标签翻译掉。"""
    out: list[dict] = []

    def flush(fields: list[str]) -> None:
        if not fields:
            return
        ident, _port = parse_windows_instance(fields[0])
        if ident:
            out.append({"InstanceId": ident,
                        "FriendlyName": fields[1] if len(fields) > 1 else "",
                        "Present": True, "LocationInformation": ""})

    fields: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            flush(fields)
            fields = []
            continue
        if ":" in line:
            fields.append(line.split(":", 1)[1].strip())
    flush(fields)
    return out


def read_windows_pnputil() -> list[dict]:
    """本机在线的摄像头设备（pnputil）。拿不到返回 []。"""
    res = _run_command(_PNPUTIL_ARGS)
    if res is None:
        return []
    code, out = res
    if code != 0 or not out.strip():
        return []
    return _parse_pnputil(out)


_PNP_SCRIPT = (
    "[Console]::OutputEncoding=[Text.Encoding]::UTF8;"
    "$ErrorActionPreference='SilentlyContinue';"
    "$d=$null;"
    "try{$d=Get-PnpDevice -Class Camera}catch{};"
    "if(-not $d){try{$d=Get-CimInstance Win32_PnPEntity |"
    " Where-Object {$_.PNPClass -eq 'Camera'}}catch{}};"
    "$d | Select-Object InstanceId,FriendlyName,Status,Present,LocationInformation |"
    " ConvertTo-Json -Compress"
)


def _run_powershell(script: str, timeout: float = 8.0) -> str | None:
    """跑一次 PowerShell，拿 stdout；超时/被拒/没有 powershell 一律返回 None，绝不抛。"""
    exe = "powershell"
    for candidate in ("powershell", "pwsh"):
        from shutil import which

        if which(candidate):
            exe = candidate
            break
    try:
        proc = subprocess.run(
            [exe, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
             "-Command", script],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    return proc.stdout


def read_windows_pnp(*, runner=_run_powershell) -> list[dict]:
    """问系统要摄像头设备清单（含 Present / LocationInformation）。失败返回 []。

    这是 **pnputil 不可用时的兜底**（本机实测 `Get-PnpDevice` 要 ~10 s，太慢）。
    """
    raw = runner(_PNP_SCRIPT)
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []
    return [d for d in data if isinstance(d, dict)]


def read_windows_devices_info() -> list[dict]:
    """摄像头设备信息：优先 pnputil（快），不行再 PowerShell（慢但兜底）。"""
    info = read_windows_pnputil()
    if info:
        return info
    return read_windows_pnp()


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes")


def windows_devices(
    *,
    registry_reader=read_windows_registry_instances,
    pnp_reader=read_windows_devices_info,
) -> list[DeviceEntry]:
    """合并"注册表枚举顺序"与"设备信息"，得到 index -> 设备 的对应。

    顺序取自注册表（DSHOW 的枚举来源）；设备信息只用来**过滤幽灵记录**和补充名字。
    采集不到时不做过滤（`present` 全 True，`source='registry'`），不会崩。
    """
    instances = _call(registry_reader, default=[]) or []
    pnp = _call(pnp_reader, default=[]) or []

    pnp_by_id: dict[str, dict] = {}
    for item in pnp:
        ident, _port = parse_windows_instance(str(item.get("InstanceId", "")))
        if ident:
            pnp_by_id[ident.upper()] = item

    has_pnp = bool(pnp_by_id)
    entries: list[DeviceEntry] = []
    seen: set[str] = set()
    for raw in instances:
        ident, port_from_id = parse_windows_instance(raw)
        if not ident or ident.upper() in seen:
            continue
        info = pnp_by_id.get(ident.upper())
        if has_pnp:
            # 拿到设备清单时，只有**在线**的才留下：
            #   - 清单里根本没有它 -> 幽灵记录（DSHOW 也枚举不到它）
            #   - 清单里有但 Present=False -> 同样不算在线
            if info is None or not _truthy(info.get("Present", True)):
                continue
        seen.add(ident.upper())
        entries.append(DeviceEntry(
            order=len(entries),
            name=str((info or {}).get("FriendlyName") or ""),
            identity=ident,
            port=str((info or {}).get("LocationInformation") or port_from_id),
            present=True,
            source="system" if has_pnp else "registry",
        ))

    if not entries and has_pnp:
        # 注册表读不到（权限/被锁）时退而用系统列表的顺序，至少身份是准的
        for item in pnp:
            ident, port_from_id = parse_windows_instance(str(item.get("InstanceId", "")))
            if not ident or ident.upper() in seen:
                continue
            if not _truthy(item.get("Present", True)):
                continue
            seen.add(ident.upper())
            entries.append(DeviceEntry(
                order=len(entries),
                name=str(item.get("FriendlyName") or ""),
                identity=ident,
                port=str(item.get("LocationInformation") or port_from_id),
                present=True,
                source="system",
            ))

    return entries


def list_devices() -> list[DeviceEntry]:
    """当前平台的摄像头设备清单。非 Windows 返回 []（整套功能退化为按索引）。"""
    if current_platform() != "windows":
        return []
    return windows_devices()


# --------------------------------------------------------------------------- #
# 探测与自动选择
# --------------------------------------------------------------------------- #
def _default_opener(index: int, width: int, height: int, fps: int,
                    backend: int | None = None):
    from .sources import INDEX_BACKEND, CameraSource  # 惰性导入：不探测就用不到 cv2

    # **必须固定后端**：传进来的 index 是 PnP 枚举的位次（`DeviceEntry.order`），
    # 而 cv2 各后端有自己的枚举顺序 —— 用 auto 会读到"另一台相机"的属性，
    # 于是自动挑相机、画质会诊、identify 全都在评错对象。
    # 完整实测对照见 sources.py 的 INDEX_BACKEND 注释。
    #
    # `backend=None` = 用 `INDEX_BACKEND`；显式给值时**只开那一个**，
    # 供 `probe_index_with_fallback()` 在上面做受控回退（理由见那个函数）。
    return CameraSource(index=index, width=width, height=height, fps=fps,
                        backend=INDEX_BACKEND if backend is None else backend,
                        optimize=False)


def probe_index(
    index: int,
    *,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    opener=None,
    backend: int | None = None,
) -> ProbeResult:
    """打开某个索引、读一帧，判断"能不能用"。任何异常都变成 ok=False，不抛。

    `backend=None` 走 `_default_opener`（= `INDEX_BACKEND`，一次尝试）。
    显式给后端时**只试那一个**（供测试与 `--backend` 覆盖用）。
    """
    open_fn = opener or _default_opener
    kwargs = {} if opener is not None else {"backend": backend}
    try:
        src = open_fn(index, width, height, fps, **kwargs)
    except Exception as exc:  # noqa: BLE001 - 打不开的原因五花八门，统一收成一句话
        first = (str(exc).splitlines() or ["打不开"])[0]
        return ProbeResult(index=index, ok=False, error=first)

    try:
        frame = None
        for _ in range(20):
            ok, f, _t = src.read()
            if ok and f is not None:
                frame = f
                break
        if frame is None:
            return ProbeResult(index=index, ok=False, error="读不到画面")
        std = float(frame.std())
        info = getattr(src, "info", None)
        w = int(getattr(info, "width", 0) or 0)
        h = int(getattr(info, "height", 0) or 0)
        rate = float(getattr(info, "fps", 0.0) or 0.0)
        if std <= 1.0:
            return ProbeResult(index=index, ok=False, width=w, height=h, fps=rate,
                               std=std, error="画面是全黑常量流")
        return ProbeResult(index=index, ok=True, width=w, height=h, fps=rate, std=std)
    except Exception as exc:  # noqa: BLE001
        return ProbeResult(index=index, ok=False, error=str(exc).splitlines()[0])
    finally:
        _call(getattr(src, "release", lambda: None), default=None)


def probe_index_with_fallback(
    index: int,
    *,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
) -> ProbeResult:
    """探测一台相机，主后端给不出帧时**受控回退**到另一个后端。

    ## 为什么探测路径可以有回退，而实时取帧路径刻意没有

    `sources.py` 里有一段长注释说明"**不做自动换后端**"：同一进程里混用
    DSHOW + MSMF 会让 MSMF 那路"打开成功但抓不到帧"，所以实时取帧**宁可
    说清让人显式选**（那次实测：自动换完更糟，还把分辨率降到 1280x720 写进缓存）。

    那条结论针对的是**同时开着两路、并长期持有**的场景。
    而这里（探测）是**短命的一次试开**：开一台、读一帧、立刻 `release()`，
    不存在"MSMF 那路一直挂着"。所以回退是安全的 —— 而且必要：

    `_default_opener` 用的是 `INDEX_BACKEND`（MSMF，2026-10-10 的实测选择），
    而 MSMF 在本机是**间歇性**坏的。2026-10-14 一个下午就见到它来回变：

        MSMF idx=1:  好 9.2s / 好（640x480）→ 坏 34.3s → 坏 33.5s → 又好

    而**同一台相机在 DSHOW 上每次都秒开、15/15 帧、同样能给 1920x1080**。
    坏的时候 `probe_index` 要等 **33 秒**才失败，于是
    `cameras.py list --probe` / 画质会诊 / 自动挑相机**全都报"一帧都拿不到"**，
    用户会以为相机坏了。

    回退**不改变探测语义**：`DeviceEntry.order` 是 PnP 位次，而本机实测
    **DSHOW 与 MSMF 的分辨率枚举一致**（idx 0 = 1920x1080、idx 1 = 640x480，
    两个后端给的是同一组值），所以"判定这台能不能用"用哪个后端都一样，
    只有**设备身份**才依赖后端枚举顺序 —— 那件事由 `cameras.py identify`
    目视确认 + 记 `identity`/`port` 解决，不靠这里。
    """
    from .sources import (INDEX_BACKEND, WINDOWS_BACKENDS,  # 惰性：不探测用不到 cv2
                          backend_name)

    r = probe_index(index, width=width, height=height, fps=fps, backend=INDEX_BACKEND)
    if r.ok:
        return r
    if current_platform() != "windows":
        return r          # 非 Windows 没有第二套后端可回退（设计上退化为纯索引）
    for name, api in WINDOWS_BACKENDS:
        if api == INDEX_BACKEND or name == "ANY":
            continue
        alt = probe_index(index, width=width, height=height, fps=fps, backend=api)
        if alt.ok:
            print(f"[相机 {index}] {backend_name(INDEX_BACKEND)} 给不出帧，"
                  f"但 {name} 正常（{alt.width}x{alt.height} std {alt.std:.1f}）——"
                  f"这次判定用 {name} 的结果", flush=True)
            return ProbeResult(index=index, ok=True, width=alt.width, height=alt.height,
                               fps=alt.fps, std=alt.std,
                               error=f"（{backend_name(INDEX_BACKEND)} 无帧，改用 {name} 探到）")
    return r


def pick_backend_for_device(
    index: int,
    *,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    prefer: int | None = None,
) -> tuple[int | None, ProbeResult]:
    """给一台相机挑一个**真能出图**的后端，返回 (后端常量, 探测结果)。

    ## 为什么需要它（2026-10-15 实测的真问题）

    这台机器上"哪个后端能用"是**浮动**的，而且在两个后端之间来回翻。
    同一次会话里两次探测，结论**正好相反**：

        （前一天）  MSMF: idx1=None        DSHOW: idx1=1920x1080
        （当天）    #0  DSHOW:**纯黑**     MSMF :#0 正常 mean=116.4 std=64.4
                    #1  DSHOW:正常        MSMF :#1 正常

    于是任何"写死一个后端"的做法都必然在某个时刻坏掉 ——
    `cameras.py identify` 原来写死 MSMF，用户就在 MSMF 坏的那一刻跑，
    结果相机 0 打不开、他只能给相机 1 按键，`config/cameras.json` 被写成
    **两个角色指向同一台**。

    ## 判据为什么看"画面质量"而不只是"能不能打开"

    "打开成功但**每一帧都是纯黑**"是这里最常见的坏状态（实测 `min=max=0`）。
    `probe_index` 已把这种判为 `ok=False`（`std <= 1.0` → "画面是全黑常量流"），
    所以**只用 `ok` 就够** —— 这也是本函数不自己读帧、而是复用 `probe_index`
    的原因：**同一个事实只留一个实现**。

    `prefer`：两路都能出图时优先用哪个（None = 用 `INDEX_BACKEND`）。
    返回 `(None, 最后一次结果)` 表示两个后端都不可用。
    """
    from .sources import INDEX_BACKEND, WINDOWS_BACKENDS, backend_name

    first = INDEX_BACKEND if prefer is None else prefer
    order: list[int] = [first]
    if current_platform() == "windows":
        for _name, api in WINDOWS_BACKENDS:
            if api is not None and api not in order:
                order.append(api)

    last: ProbeResult | None = None
    for api in order:
        r = probe_index(index, width=width, height=height, fps=fps, backend=api)
        if r.ok:
            if api != first and last is not None:
                print(f"[相机 {index}] {backend_name(first)} 不可用"
                      f"（{last.error}），**改用 {backend_name(api)}**"
                      f"（{r.width}x{r.height} std {r.std:.1f}）", flush=True)
            return api, r
        last = r
        print(f"[相机 {index}] {backend_name(api)} 不可用：{r.error}", flush=True)
    return None, (last or ProbeResult(index=index, ok=False, error="没有可用后端"))


def is_avoided(name: str, settings: dict | None = None) -> bool:
    """名字看着像内置/红外？这类相机不适合做双目（没法刚性固定在横梁上）。"""
    merged = {**DEFAULT_SETTINGS, **(settings or {})}
    low = (name or "").lower()
    if not low:
        return False
    if any(str(p).lower() in low for p in merged.get("avoid_names") or []):
        return True
    return "infrared" in low or re.search(r"\bir\b", low) is not None


def rank_candidates(
    devices: list[DeviceEntry],
    *,
    reg: Registry | None = None,
    settings: dict | None = None,
) -> list[DeviceEntry]:
    """按"更可能是我们要的那台"排序：配置里 pin 过的 > 外接 > 其它；内置/红外降权。

    标了「无关」的一律沉到最后（比"内置/红外"的降权还狠），
    这样自动挑相机永远先挑别人 —— 用户明确说过不要的，就别再自作聪明。
    """
    merged = {**DEFAULT_SETTINGS, **(settings or {})}
    reg = reg or Registry()
    pinned_ids: set[str] = set()
    pinned_ports: set[str] = set()
    verified_ids: set[str] = set()
    for spec in reg.roles.values():
        if spec.identity:
            pinned_ids.add(spec.identity.upper())
            if spec.verified:
                verified_ids.add(spec.identity.upper())
        if spec.port:
            pinned_ports.add(spec.port.lower())

    def score(d: DeviceEntry) -> float:
        s = 0.0
        if d.identity and d.identity.upper() in pinned_ids:
            s += 100.0
            if d.identity.upper() in verified_ids:
                s += 50.0
        if d.port and d.port.lower() in pinned_ports:
            s += 80.0
        if merged.get("prefer_external", True) and d.identity.upper().startswith("USB\\"):
            s += 20.0
        if is_avoided(d.name, merged):
            s -= 200.0
        if is_device_ignored(d, reg):
            s -= 1000.0
        return s
    return sorted(devices, key=lambda d: (-score(d), d.order))


@dataclass
class Availability:
    """「现在能不能用相机」的集中判定结果。

    ## 为什么需要它

    没有相机时，四个入口给出**四种不同的失败**（2026-10-10 实测）：

    | 入口 | 表现 |
    | ---- | ---- |
    | `run_realtime.py` | 5.1 s，先刷一堆 OpenCV 原始 `VIDEOIO` 警告，最后才说"没找到可用的摄像头" |
    | `calibrate_stereo.py` | 只说"取帧源装配失败，退出"（**没说是相机**） |
    | `verify_stereo.py` | 报"找不到标定文件"（真因往往正是**没有相机所以从没标定成功过**） |
    | `check_stereo_setup.py` | 退出码 2，提示要先给 `--left/--right` |

    没有任何一处**集中**回答"到底有没有相机、以及该怎么办"。
    用户（尤其是不接相机时）会被引到错误的方向上排查。

    这个类把判定集中到一个地方，并给出**可执行的下一步**。

    ⚠️ 它判的是"**现在**能不能用"，不是"机器上有没有相机" ——
    被占用、驱动刚崩、USB 掉带宽都会让一台存在的相机不可用，
    这时 `device_count > 0` 但 `usable_count == 0`，`reason` 必须说清是这种情况。
    """

    ok: bool = False
    device_count: int = 0        # 枚举到的设备数（含不可用的）
    usable_count: int = 0        # 真的能开、能读到非黑帧的台数
    need: int = 1                # 调用方需要几台（`ok` 是"够不够"，不是"有没有"）
    probed: list[int] = field(default_factory=list)   # 试开过哪些索引
    reason: str = ""             # 一句话说清"为什么不行"
    remedy: str = ""             # 可执行的下一步


def camera_availability(
    *,
    need: int = 1,
    devices: list[DeviceEntry] | None = None,
    probe=probe_index_with_fallback,
    reg: Registry | None = None,
    settings: dict | None = None,
) -> Availability:
    """现在能用几台相机？不能的话为什么、怎么办。

    ## ⚠️ 它**委托** `select_cameras`，不自己判"合格"

    第一版在这里把"能开 + 能读到帧 + 不是全黑 + 分辨率达标"**又实现了一遍**，
    于是同一台机器上两个函数给出**相反的结论**（2026-10-10 实测）：

    ```
    select_cameras_detailed(need=1)  -> ok=True  picks=[#0 1920x1080]
    camera_availability(need=1)      -> ok=False usable=0
                                        "#0 画面是全黑常量流"
    ```

    两个都对，只是判据在细节上不一致（黑帧阈值的比法不同）。
    **同一个事实实现两遍，迟早会漂移** —— 这正是本项目反复出现的头号问题，
    所以这里只保留**一个**实现：挑选用 `select_cameras`，本函数只负责
    "把失败翻译成人话 + 给出下一步"。

    所有外部依赖（`devices` / `probe` / `reg`）可注入，
    因此没有相机的机器上也能把每条分支走一遍（见 `selftest_camera_presence.py`）。
    """
    merged = {**DEFAULT_SETTINGS, **((reg.settings if reg else {}) or {})}
    merged.update(settings or {})

    devices = list_devices() if devices is None else devices
    sel = select_cameras_detailed(need=need, devices=devices, probe=probe,
                                  reg=reg, settings=merged)

    av = Availability(ok=sel.ok, device_count=len(devices),
                      usable_count=len(sel.picks))
    av.probed = list(range(sel.probed))
    av.need = int(need)

    if av.ok:
        return av

    # ---- 说清是"没有相机"还是"有但都不可用"：排查方向完全不同 ----
    if len(devices) == 0:
        av.reason = f"没有检测到任何摄像头（试开了 {sel.probed} 个索引）"
        if sel.probe_errors:
            av.reason += "；" + "；".join(sel.probe_errors[:3])
        av.remedy = (
            "① 确认相机插好（换一个 USB 口，优先直插主板而不是扩展坞）；"
            "② 跑 `启动程序.bat check` 看设备列表；"
            "③ 只是想试功能的话可以喂视频："
            "`python apps/run_video.py --source <视频文件>`"
        )
    elif av.usable_count == 0:
        av.reason = (f"检测到 {len(devices)} 个设备，但能用的只有 0 个"
                     f"（需要 {need} 个，试开了 {sel.probed} 个索引）")
        if sel.probe_errors:
            av.reason += "；" + "；".join(sel.probe_errors[:4])
        av.remedy = (
            "① 相机可能被别的程序占着（关掉相机 App / 会议软件 / 另一个本程序实例）；"
            "② 跑 `启动程序.bat diag` 看是不是带宽不足或画面全黑；"
            "③ 跑 `启动程序.bat setup` 确认「左 / 右 / 无关」的分配"
        )
    else:
        # 挑到了一些，但**不够用** —— 这条必须与"一台都没有"分开说，
        # 否则用户会以为相机全坏了，实际只是台数不够。
        av.reason = (f"检测到 {len(devices)} 个设备，但能用的只有 {av.usable_count} 个"
                     f"（需要 {need} 个，试开了 {sel.probed} 个索引）")
        av.remedy = (
            "① 第二台相机没插好 / 被占用 / 分辨率不达标（双目需要两台**同分辨率**的）；"
            "② 跑 `启动程序.bat check` 逐台确认；"
            "③ 只想试单目的话用 `启动程序.bat run`，或喂视频："
            "`python apps/run_video.py --source <视频文件>`"
        )
    return av


# --------------------------------------------------------------------------- #
# 选相机：**唯一的判定者**
# --------------------------------------------------------------------------- #
@dataclass
class Selection:
    """一次"要几台相机"的完整结论：挑到了什么、以及要告诉用户的话。

    ## 为什么要有这个结果对象（而不是让调用方自己判断）

    改之前有两套判定并存，它们**对同一台机器给出相反的结论**（2026-10-10 实测）：

    | 谁 | 对同一台笔记本怎么说 |
    | -- | -------------------- |
    | `camera_availability(need=1)` | `ok=True`（#0 确实能开、能读到帧） |
    | `select_cameras(need=1)` | 挑到了，但**告警**：只能动用被标为「无关」的相机 |

    两个都对，只是**判据不同**：后者把"用户明确标了无关"当成降级。
    于是 `run_realtime` 报出"可用性判定通过，但挑选失败" ——
    一句**自相矛盾**的话，因为它在拿两套判定的结果对撞。

    → 修法：**只留一个判定者**。`select_cameras()` 仍然是权威，
    但它把"挑到了什么"与"有什么话要说"一起交出来，调用方不再自己判断。
    """

    picks: list[ProbeResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)   # 要原样打给用户的话
    need: int = 1
    probed: int = 0                                  # 实际试开了几次
    probe_errors: list[str] = field(default_factory=list)  # 每台失败的原因

    @property
    def ok(self) -> bool:
        """**够不够用**，不是"有没有挑到"。

        ⚠️ 这里曾经只看"挑到没挑到"，于是 `need=2` 只挑到 1 台时
        `ok=True` —— 调用方（双目标定）会拿到**数量不足**的结果并继续往下跑，
        在更远的地方以更难懂的方式失败。**"够不够"才是调用方要问的问题。**
        （由 `selftest_camera_presence.py` 第 6/8 节的对照断言逼出来。）
        """
        return len(self.picks) >= max(1, int(self.need))


def select_cameras_detailed(
    *,
    need: int = 1,
    devices: list[DeviceEntry] | None = None,
    probe=probe_index_with_fallback,
    reg: Registry | None = None,
    settings: dict | None = None,
    indices: list[int] | None = None,
) -> Selection:
    """`select_cameras` 的"带说明"版本：挑到的相机 + 该说的话。

    识别到的告警（原来直接 `print` 在库层，调用方既接不住也没法落盘）
    现在随结果返回 —— 这是计划书 P1-4「库层 print 禁令」的一个真实样本。
    """
    import contextlib
    import io

    counter = {"n": 0}
    errors: list[str] = []

    def counting_probe(i: int):
        counter["n"] += 1
        return probe(i)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        picks = select_cameras(need=need, devices=devices, probe=counting_probe,
                               reg=reg, settings=settings, indices=indices,
                               _errors_out=errors)
    notes = [buf.getvalue().strip()] if buf.getvalue().strip() else []
    return Selection(picks=list(picks), notes=notes, need=int(need),
                     probed=counter["n"], probe_errors=list(errors))


def _failed_probe(entry: DeviceEntry, error: str) -> ProbeResult:
    """构造一个"试开失败"的 `ProbeResult`（保住设备信息，便于报给用户）。"""
    return ProbeResult(entry.order, False, 0, 0, 0.0, 0.0, error, entry)


def select_cameras(
    *,
    need: int = 1,
    devices: list[DeviceEntry] | None = None,
    probe=probe_index_with_fallback,
    reg: Registry | None = None,
    settings: dict | None = None,
    indices: list[int] | None = None,
    _errors_out: list[str] | None = None,
) -> list[ProbeResult]:
    """自动挑 need 台"能用"的相机。挑不满就返回实际的几台，**不抛异常**。

    合格 = 能打开 + 读到帧 + 不是全黑常量流 + 分辨率达到 min_width/min_height。
    双目时会优先挑两台实际分辨率一致的一对。

    ⚠️ **它是"能不能用相机"的唯一判定者**。想连说明一起拿到，
    用 `select_cameras_detailed()`（它不会再往终端打印）。

    `_errors_out`：传一个 list 进来，试开失败的原因会写进去。
    **刻意用私有名** —— 它是给 `select_cameras_detailed` 用的通道，
    公开 API 请走那个版本。
    """
    merged = {**DEFAULT_SETTINGS, **((reg.settings if reg else {}) or {})}
    merged.update(settings or {})
    if devices is None:
        devices = list_devices()

    if indices is not None:
        cands = [DeviceEntry(order=int(i)) for i in indices]
    else:
        ranked = rank_candidates(devices, reg=reg, settings=merged)
        known = {d.order for d in ranked}
        cands = list(ranked) + [
            DeviceEntry(order=i)
            for i in range(int(merged["max_index"]) + 1)
            if i not in known
        ]

    # 标了「无关」的排到最后：只有"其它相机都不够用"时才轮到它们，
    # 并且要显式告警 —— 用户明确说过不要的相机，静默用上比报错更糟。
    ignored_used: list[DeviceEntry] = []
    if indices is None and cands:
        keep, drop = [], []
        for d in cands:
            (drop if is_device_ignored(d, reg) else keep).append(d)
        if drop:
            cands = keep + drop
            ignored_used = drop

    min_w = int(merged["min_width"])
    min_h = int(merged["min_height"])
    limit = 1 if need <= 1 else need + 2

    picked: list[ProbeResult] = []
    probe_errors: list[str] = []
    if _errors_out is not None:
        _errors_out.clear()
    for entry in cands:
        # ⚠️ 试开必须**自己接住异常**：`probe_index` 会去开真实设备，
        # 驱动抽风 / 设备被热拔 / USB 控制器报错都可能抛出来。
        # 改之前这里不接，一个索引抛异常会让**整次挑选崩掉** ——
        # 用户看到的是一个 traceback，而不是"这台不行，试下一台"。
        # （这个缺陷是 `selftest_camera_presence.py` 的"试开抛异常"一节逼出来的：
        #   先写测试、测试崩了，才发现是库层的锅。）
        try:
            res = probe(entry.order)
        except Exception as exc:                     # noqa: BLE001
            # ⚠️ **失败的试开不许进 `picked`**。第一版把它 append 了进去，
            # 于是"试开抛异常"被当成"挑到了一台相机"（`ok=False` 的
            # `ProbeResult` 也算数）—— `usable_count` 报 1、`Selection.ok` 报 True。
            # 失败信息只进 `probe_errors`，供报错文本使用。
            probe_errors.append(f"#{entry.order} 试开异常：{type(exc).__name__}: {exc}")
            continue
        if not isinstance(res, ProbeResult):
            continue
        res = ProbeResult(res.index, res.ok, res.width, res.height, res.fps,
                          res.std, res.error, entry)
        if not res.ok:
            probe_errors.append(f"#{entry.order} {res.error or '打不开'}")
            continue
        if res.width and res.height and (res.width < min_w or res.height < min_h):
            probe_errors.append(f"#{entry.order} 分辨率 {res.width}x{res.height} "
                                f"低于要求 {min_w}x{min_h}")
            continue
        picked.append(res)
        if len(picked) >= limit:
            break

    if need <= 1:
        result = picked[:max(need, 0)]
    else:
        result = None
        for i in range(len(picked)):
            for j in range(i + 1, len(picked)):
                if (picked[i].width, picked[i].height) == (picked[j].width, picked[j].height):
                    result = [picked[i], picked[j]][:need]
                    break
            if result is not None:
                break
        if result is None:
            result = picked[:need]

    # 真挑到了「无关」的相机 -> 说清楚，别让它静默混进来
    used_ignored = [r for r in result if r.entry is not None
                    and is_device_ignored(r.entry, reg)]
    if used_ignored:
        names = "、".join(f"#{r.index} {r.entry.name or '未知名'}"
                          for r in used_ignored if r.entry)
        print(f"[警告] 可用的相机不够，只能动用被标为「无关」的相机：{names}\n"
              f"        如果这是误标，跑 python apps/tools/cameras.py unignore --all 清掉；"
              f"如果想换一台，检查相机是不是没插好。")
    if _errors_out is not None:
        _errors_out.extend(probe_errors)
    return result


# --------------------------------------------------------------------------- #
# 配置读写
# --------------------------------------------------------------------------- #
def load_registry(path: str | Path | None = None) -> Registry:
    """读 `config/cameras.json`。文件不存在/损坏一律返回空 Registry（不抛）。"""
    p = Path(path) if path else _CFG
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return Registry()
    if not isinstance(data, dict):
        return Registry()

    roles: dict[str, RoleSpec] = {}
    raw_roles = data.get("roles")
    if isinstance(raw_roles, dict):
        for name, spec in raw_roles.items():
            if not isinstance(spec, dict):
                continue
            idx = spec.get("index")
            roles[str(name)] = RoleSpec(
                identity=str(spec.get("identity") or ""),
                port=str(spec.get("port") or ""),
                name=str(spec.get("name") or ""),
                index=int(idx) if isinstance(idx, (int, float)) else None,
                verified=bool(spec.get("verified", False)),
            )
    settings = data.get("settings") if isinstance(data.get("settings"), dict) else {}
    return Registry(v=int(data.get("v") or 1), roles=roles,
                   ignored=_parse_spec_list(data.get("ignored")), settings=dict(settings))


def _parse_spec_list(raw) -> list[RoleSpec]:
    """把配置里的 `ignored`（以及将来的同类列表）解析成 RoleSpec 列表。

    容错优先：非列表 / 元素不是 dict / 字段类型不对都跳过，绝不让配置把程序弄崩。
    """
    out: list[RoleSpec] = []
    if not isinstance(raw, list):
        return out
    for spec in raw:
        if not isinstance(spec, dict):
            continue
        idx = spec.get("index")
        out.append(RoleSpec(
            identity=str(spec.get("identity") or ""),
            port=str(spec.get("port") or ""),
            name=str(spec.get("name") or ""),
            index=int(idx) if isinstance(idx, (int, float)) else None,
            verified=bool(spec.get("verified", False)),
        ))
    return out


def save_registry(reg: Registry, path: str | Path | None = None) -> Path:
    """写 `config/cameras.json`（这是用户显式动作，写不进去就让它报错）。"""
    p = Path(path) if path else _CFG
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "v": reg.v,
        "roles": {k: asdict(v) for k, v in reg.roles.items()},
        "ignored": [asdict(s) for s in reg.ignored],
        "settings": reg.settings,
    }
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def set_role(
    reg: Registry,
    role: str,
    *,
    entry: DeviceEntry | None = None,
    index: int | None = None,
    verified: bool = False,
) -> RoleSpec:
    """把一个角色固定到某台设备（`cameras.py pin/identify` 共用）。

    这里**自己**维护"角色与「无关」互斥"这条不变量，不指望调用方：

    原来注释写的是"反过来 `set_role()` 之后旧的无关联记录也要清（由调用方
    `cameras.py` 负责）"。真机上就漏了 —— 配置里出现同一台设备
    **既是 `main` 又在「无关」清单**，而两处代码各自看都"没错"：
    调用方以为自己调了 `unmark_ignored`，而它按 identity 匹配，
    遇到没有身份的旧记录就匹配不上，于是那条记录活了下来。

    **不变量要留在拥有数据的模块里。** 这里按 identity/port/编号三种口径
    都清一遍，任何调用顺序都不会留下矛盾状态。
    """
    spec = RoleSpec(
        identity=entry.identity if entry else "",
        port=entry.port if entry else "",
        name=entry.name if entry else "",
        index=index if index is not None else (entry.order if entry else None),
        verified=verified,
    )
    reg.roles[str(role)] = spec

    # 同一台设备不能同时是角色和「无关」——三种口径都清，避免哪种匹配不上就漏。
    dropped = 0
    if entry is not None:
        dropped = unmark_ignored(reg, entry=entry)
    if dropped == 0 and spec.index is not None:
        dropped = unmark_ignored(reg, index=spec.index)
    return spec


# --------------------------------------------------------------------------- #
# 「无关」清单：明确标记"这台不要用"
# --------------------------------------------------------------------------- #
def _spec_matches_device(spec: RoleSpec, d: DeviceEntry) -> bool:
    """这份记录描述的是这台设备吗？

    **只认 identity / port，绝不退到按名字匹配。** 这一点与 `_match_entry()` 不同，
    是故意的：两台同型号相机的名字**一模一样**（本项目两台 aoni 都叫
    "aoni webcam A25"），按名字匹配会把"我只想排除这一台"变成"两台一起排除"。
    角色解析可以容忍这个（最终只取一台，且 identity/port 优先），
    但「无关」清单不行 —— 静默多标一台，用户下次就找不到相机了。

    真拿不到身份时（非 Windows / 枚举不可用）退到编号，并且只认
    `verified=False` 的记录 —— 与 `resolve_role()` 的兜底口径一致：
    **没被目视确认过的记录才允许按编号走。**
    """
    if spec.identity and d.identity:
        return spec.identity.upper() == d.identity.upper()
    if spec.port and d.port:
        return spec.port.lower() in d.port.lower()
    if spec.identity or spec.port:
        # 有身份/端口却没对上 -> 就是另一台，不要再看名字
        return False
    return (not spec.verified and spec.index is not None and spec.index == d.order)


def is_device_ignored(d: DeviceEntry, reg: Registry | None = None) -> bool:
    """这台设备在「无关」清单里吗？"""
    reg = reg if reg is not None else load_registry()
    return any(_spec_matches_device(s, d) for s in reg.ignored)


def find_ignored_device(
    spec: RoleSpec, devices: list[DeviceEntry]
) -> DeviceEntry | None:
    """在设备列表里找这份「无关」记录指向的那台（严格匹配，不因重名误伤）。"""
    for d in devices:
        if _spec_matches_device(spec, d):
            return d
    return None


def ignored_orders(reg: Registry, devices: list[DeviceEntry]) -> set[int]:
    """把「无关」清单解析成"当前这些设备里，哪些编号是无关的"。"""
    out: set[int] = set()
    for d in devices:
        if is_device_ignored(d, reg):
            out.add(d.order)
    return out


def role_of_device(reg: Registry, d: DeviceEntry) -> str | None:
    """这台设备当前挂在哪个角色上（没有就 None）。标记「无关」时用来摘掉旧角色。"""
    for role, spec in reg.roles.items():
        if _spec_matches_device(spec, d):
            return role
    return None


def mark_ignored(
    reg: Registry,
    *,
    entry: DeviceEntry | None = None,
    index: int | None = None,
    verified: bool = False,
) -> RoleSpec:
    """把一台设备标成「无关」。

    同时把"它曾经是左/右/主"这件事摘掉 —— 一台相机不能既是左相机又是无关，
    否则下次自动挑相机时两边打架（谁是真理说不清）。返回值就是新的记录。
    """
    order = index if index is not None else (entry.order if entry else None)
    if entry is not None:
        old_role = role_of_device(reg, entry)
        if old_role:
            reg.roles.pop(old_role, None)
    # 先去掉同一台设备的旧记录，避免重复堆叠
    if entry is not None:
        reg.ignored = [s for s in reg.ignored if not _spec_matches_device(s, entry)]
    elif order is not None:
        reg.ignored = [s for s in reg.ignored
                       if not (not s.verified and s.index == order)]
    spec = RoleSpec(
        identity=entry.identity if entry else "",
        port=entry.port if entry else "",
        name=entry.name if entry else "",
        index=order,
        verified=verified,
    )
    reg.ignored.append(spec)
    return spec


def unmark_ignored(
    reg: Registry,
    *,
    entry: DeviceEntry | None = None,
    index: int | None = None,
) -> int:
    """取消一台设备的「无关」标记，返回摘掉了几条记录。"""
    before = len(reg.ignored)
    if entry is not None:
        reg.ignored = [s for s in reg.ignored if not _spec_matches_device(s, entry)]
    elif index is not None:
        reg.ignored = [s for s in reg.ignored if s.index != index]
    return before - len(reg.ignored)


def clear_ignored(reg: Registry) -> int:
    """清空整份「无关」清单，返回清掉了几条。"""
    n = len(reg.ignored)
    reg.ignored = []
    return n


# --------------------------------------------------------------------------- #
# 解析：角色 -> 当前索引
# --------------------------------------------------------------------------- #
def format_device_table(
    devices: list[DeviceEntry],
    probes: dict[int, ProbeResult] | None = None,
    settings: dict | None = None,
    reg: Registry | None = None,
) -> str:
    """打印用设备清单（工具与交互提示共用）。

    传 `reg` 时会把「无关」标记也打出来 —— 选相机时看得见"这台我已经标过不用了"，
    比事后纳闷"为什么自动挑相机不选它"要好。
    """
    if not devices:
        return "  （没有检测到任何摄像头设备；非 Windows 或系统枚举不可用）"
    probes = probes or {}
    lines = []
    for d in devices:
        probe = probes.get(d.order)
        tail = ""
        if probe is not None:
            tail = (f"  实测 {probe.width}x{probe.height}@{probe.fps:.1f}"
                    if probe.ok else f"  实测失败({probe.error})")
        mark = ""
        if reg is not None and is_device_ignored(d, reg):
            mark = "  [无关]"
        elif is_avoided(d.name, settings):
            mark = "  [内置/红外,不推荐]"
        lines.append(
            f"  #{d.order}  {d.name or '(无名字)':<20} "
            f"端口 {d.port or '-':<24} id {d.identity or '-'}{tail}{mark}"
        )
    return "\n".join(lines)


def _stdin_is_tty() -> bool:
    try:
        return bool(sys.stdin and sys.stdin.isatty())
    except (ValueError, OSError):
        return False


def _match_entry(
    spec: RoleSpec, devices: list[DeviceEntry], *, taken: set[int] | None = None
) -> tuple[DeviceEntry | None, str]:
    """按 identity -> port -> name 找设备。

    ## ⚠️ 两条硬规则（都是 2026-10-14 实测踩出来的）

    ### 1. 名字兜底**拒绝歧义**

    实测：两台同型号相机都叫 `aoni webcam A25`。其中一台换了 USB 口，
    于是它绑定的 **port 对不上**，`name` 兜底是**子串匹配 + 取第一个命中**，
    结果 `right -> #0`（任意抓的同名那台），而它绑定的那台根本不在线。

    危害是双重的：**用错相机**（左右可能互换），并且 #0 当时只有 1280x720
    而 #1 是 1920x1080 —— **两路分辨率不一致，双目标定失去意义**，
    界面上看不出任何异常。

    ### 2. 已目视确认过的角色，**身份/端口对不上就不许退回名字**

    这条更要紧：`verified=True` 意味着用户**亲眼确认过**哪台是左、哪台是右。
    此时身份对不上说明**那台相机不在线（或被枚举到别处）**，
    而按名字抓一台同型号的顶替，恰好会把左右**换掉** ——
    用户看到"识别出来的左右跟我标的反了"，正是这个。

    所以：有身份信息却对不上 -> 返回失败，让 `resolve_role` 报错并给下一步；
    只有角色**从没被确认过**（也就没绑定身份）时才允许纯按名字找。
    """
    taken = taken or set()

    def usable(d: DeviceEntry) -> bool:
        return d.present and d.order not in taken

    bound = bool(spec.identity or spec.port)

    # ① 身份：唯一，精确匹配
    if spec.identity:
        want = spec.identity.upper()
        for d in devices:
            if d.identity.upper() == want:
                return (d, "identity") if usable(d) else (None, "")

    # ② 端口：唯一
    if spec.port:
        want = spec.port.lower()
        for d in devices:
            if want in d.port.lower():
                return (d, "port") if usable(d) else (None, "")

    # ③ 名字：可能不唯一，且**绑过身份的绝不用它顶替**
    if spec.name and not bound:
        want = spec.name.lower()
        hits = [d for d in devices if want in d.name.lower() and usable(d)]
        if len(hits) == 1:
            return hits[0], "name"
        if len(hits) > 1:
            orders = "、".join(f"#{d.order}" for d in hits)
            raise AmbiguousMatch(
                f"配置里的名字「{spec.name}」在**当前设备里匹配到 {len(hits)} 台**"
                f"（{orders}），无法确定是哪一台。\n"
                f"  常见原因：同型号同名相机。\n"
                f"  怎么办：跑 python apps/tools/cameras.py identify **重新固定**，"
                f"程序会逐个显示画面让你确认哪台是左、哪台是右。")
    return None, ""


class AmbiguousMatch(RuntimeError):
    """按名字匹配到多台设备 —— 不能替用户猜，必须让用户确认。"""


def _prompt_pick(
    role: str, spec: RoleSpec, devices: list[DeviceEntry], why: str
) -> int | None:
    """配置对不上时，把设备清单摆出来让用户自己选（不替用户猜）。"""
    print(f"配置里的「{role}」当前对不上：{why}")
    print("检测到的相机：")
    print(format_device_table(devices))
    default = spec.index
    hint = f"回车={'用配置里记的索引 ' + str(default) if default is not None else '放弃'}"
    while True:
        try:
            ans = input(f"「{role}」用哪个编号？[输入编号；{hint}；q=放弃] ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if ans.lower() in ("q", "quit", "exit"):
            return None
        if ans == "":
            return default
        if ans.isdigit():
            return int(ans)
        print("  请输入编号（如 0）或回车。")


def resolve_roles(
    roles: list[str],
    reg: Registry | None = None,
    devices: list[DeviceEntry] | None = None,
    *,
    interactive: bool = True,
) -> dict[str, ResolvedCamera]:
    """一次解析**多个**角色，并保证它们**不会落到同一台相机**上。

    ## 为什么必须一起解析（而不是各解析各的）

    实测的真缺陷（2026-10-14）：两台同型号相机，其中一台换了 USB 口 →
    它绑定的端口对不上 → 按**名字**兜底 → 名字相同 → 任意抓了第一个。
    结果 `left` 与 `right` 都可能指向同一台；而"标定窗口里两路其实是同一台"
    这件事，用户要到**一对有效图像都收不到**时才发现，白浪费一轮。

    所以这里按角色**依次**解析，并把已经认领的索引传给下一个角色排除掉。
    """
    reg = reg if reg is not None else load_registry()
    if devices is None:
        devices = list_devices()
    out: dict[str, ResolvedCamera] = {}
    taken: set[int] = set()
    for role in roles:
        spec = reg.roles.get(str(role))
        if spec is None:
            continue
        entry, how = _match_entry(spec, devices, taken=taken)
        if entry is not None:
            taken.add(entry.order)
            warns = ()
            if is_device_ignored(entry, reg):
                warns = (f"「{role}」指向的 #{entry.order} {entry.name or ''} 已被标为「无关」——"
                         f"配置自相矛盾，跑 python apps/tools/cameras.py identify 重新确认。",)
            out[str(role)] = ResolvedCamera(role=str(role), index=entry.order, how=how,
                                            entry=entry, warnings=warns)
        else:
            out[str(role)] = resolve_role(role, reg=reg, devices=devices,
                                          interactive=interactive)
            if out[str(role)].index is not None:
                taken.add(int(out[str(role)].index))
    return out


def resolve_role(
    role: str,
    reg: Registry | None = None,
    devices: list[DeviceEntry] | None = None,
    *,
    interactive: bool = True,
    probe=None,
) -> ResolvedCamera:
    """把一个角色解析成**当前**索引。

    优先级：identity -> port -> name（都对不上时：可交互就让你选，否则报错并附设备清单）。

    ⚠️ **要同时解析多个角色时请用 `resolve_roles()`** —— 它会保证两个角色
    不落到同一台相机上；单独调用本函数没有这个保证。
    """
    reg = reg if reg is not None else load_registry()
    if devices is None:
        devices = list_devices()
    spec = reg.roles.get(str(role))
    if spec is None:
        known = "、".join(sorted(reg.roles)) or "（一个都没配）"
        raise RuntimeError(
            f"配置里没有角色「{role}」。已配置的角色：{known}\n"
            f"先跑 python apps/tools/cameras.py identify 固定相机。"
        )

    entry, how = _match_entry(spec, devices)
    if entry is not None:
        warns = ()
        if is_device_ignored(entry, reg):
            warns = (f"「{role}」指向的 #{entry.order} {entry.name or ''} 已被标为「无关」——"
                     f"配置自相矛盾，跑 python apps/tools/cameras.py identify 重新确认。",)
        return ResolvedCamera(role=str(role), index=entry.order, how=how, entry=entry,
                              warnings=warns)

    expected = []
    if spec.identity:
        expected.append(f"身份 {spec.identity}")
    if spec.port:
        expected.append(f"端口 {spec.port}")
    if spec.name:
        expected.append(f"名字 {spec.name}")
    why = "期望 " + "；".join(expected) if expected else "配置里没记任何身份信息"

    # 索引兜底只在**没目视确认过**的角色上允许：这种角色本来就没绑定身份，
    # 按索引走至少是"上次用过的编号"。反过来，identify 确认过的角色若身份对不上，
    # 说明相机真的换了/漂移了，此时**绝不能**蒙着用配置里那个旧索引
    # （那正是"左边窗口是外接、右边却是笔记本内置"的成因），必须让用户重新确认。
    if not spec.verified and spec.index is not None:
        warn = (f"「{role}」按身份/端口都没找到（{why}），"
                f"已退回配置里的索引 #{spec.index} —— 索引会漂移，可能用错相机，请核对！")
        return ResolvedCamera(role=str(role), index=spec.index, how="index-fallback",
                              warnings=(warn,))

    if interactive and _stdin_is_tty() and devices:
        chosen = _prompt_pick(str(role), spec, devices, why)
        if chosen is None:
            raise RuntimeError(f"没有为「{role}」选定相机，已中止。")
        hit = next((d for d in devices if d.order == chosen), None)
        warns = (f"「{role}」本次改用手动指定 #{chosen}（配置未改）",
                 "如需永久生效，请重跑 python apps/tools/cameras.py identify")
        return ResolvedCamera(role=str(role), index=chosen, how="manual",
                              entry=hit, warnings=warns)

    raise RuntimeError(
        f"「{role}」指定的相机当前不在线（{why}）。\n"
        f"检测到的相机：\n{format_device_table(devices)}\n"
        f"请先跑 python apps/tools/cameras.py check 看是不是端口变了。"
    )


def resolve_camera_arg(
    value,
    *,
    reg: Registry | None = None,
    devices: list[DeviceEntry] | None = None,
    interactive: bool = True,
) -> int | None:
    """`--left/--right/--camera` 的取值 -> 索引。

    None/'' -> None；int -> 原样；'0' -> 0；'port:xx'/'id:xx'/'name:xx' -> 直接匹配设备；
    其它字符串 -> 当作角色名查 `config/cameras.json`。
    """
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    text = str(value).strip()
    if text == "":
        return None
    low = text.lower()
    if low == AUTO_TOKEN:
        raise RuntimeError(
            "「auto」需要在命令行上用 --auto-cameras 表达（或在 --left/--right 里写 auto）。"
        )
    if text.isdigit():
        return int(text)

    for prefix, attr in (("port:", "port"), ("id:", "identity"), ("name:", "name")):
        if low.startswith(prefix):
            needle = text[len(prefix):].strip().lower()
            if not needle:
                raise RuntimeError(f"{prefix} 后面要写具体的值，例如 {prefix}Port_#0001")
            pool = devices if devices is not None else list_devices()
            for d in pool:
                if needle in str(getattr(d, attr)).lower():
                    return d.order
            raise RuntimeError(
                f"没找到匹配 {text} 的相机。\n检测到的相机：\n{format_device_table(pool)}"
            )

    resolved = resolve_role(text, reg, devices, interactive=interactive)
    for warn in resolved.warnings:
        print(f"[警告] {warn}")
    return resolved.index
