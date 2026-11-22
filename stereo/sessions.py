"""双目采集会话：一个目录 = 一次采集，`session.json` 是唯一权威元数据。

为什么需要这一层
----------------
录制数据既不是"可再生的 `outputs/`"，也不是"入库的 `assets/`"：
它体积大、不可再生、丢了就没了。所以给它单独的 `data/` 根，
并用一个 schema 把"这次到底录了什么、用什么参数、标定是哪一份"固定下来。
否则一周后你只有两段 mkv，不知道分辨率 / 基线 / 同步偏移，数据等于废的。

用法::

    from stereo.sessions import create_session, save_session, load_session, validate_session

    session_dir, s = create_session("data/stereo", label="ball01")
    s.left = CameraInfo(role="left", video="left.mkv", width=1920, height=1080, frame_count=300)
    ...
    save_session(s, session_dir)
    errors, warnings = validate_session(session_dir)

约定
----
- 目录名 = `YYYYMMDD-HHMMSS`（重名加 `-01`），`id` 与目录名一致；
- 视频路径默认相对会话目录；也接受绝对路径；
- `calib/` 里放**本次使用**的标定快照（npz + json），并用 sha256 固定；
- `derived/` 放可再生的 CSV / 图 / 标注视频，删了不影响原始数据；
- 基线：`baseline_mm` 是标定算法结果（几何参数用它），
  `baseline_measured_mm` 是物理测量值（只作 10% / 20% 交叉核对，不覆盖算法值）。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2

from calib.params import StereoRig
from sphere_tracker.quality import frames_quality
#: 会话文件名与「一个根下面有哪些会话」都只有一个产地：``sphere_tracker/discovery.py``
#: （2026-11-09 起；以前这里自己写一个字面量 + 一行只扫一层的 glob）。
from sphere_tracker.discovery import SESSION_FILE, discover_session_dirs
from .stereo_mode import optical_axes_angle_deg

SCHEMA = "sphere-stereo-session"
VERSION = 1
ROOT = Path(__file__).resolve().parents[1]


class SessionError(RuntimeError):
    """会话文件缺失 / 版本不认 / JSON 坏了。"""


# --------------------------------------------------------------------------
# 数据模型
# --------------------------------------------------------------------------
@dataclass
class CameraInfo:
    role: str                       # "left" / "right"
    video: str                      # 相对会话目录的文件名
    width: int
    height: int
    frame_count: int = 0
    fps_requested: float = 0.0
    fps_measured: float = 0.0
    backend: str = ""               # "MSMF" / "DSHOW" / ""
    identity: str = ""              # PnP 设备身份（换口不变）
    name: str = ""                  # Windows FriendlyName
    port: str = ""                  # 端口（换口会变）


@dataclass
class CalibrationInfo:
    path: str                       # 相对会话目录，如 "calib/stereo_calibration.npz"
    sha256: str = ""
    image_size: list[int] = field(default_factory=list)
    baseline_mm: float = 0.0        # 算法结果（几何参数以它为准）
    baseline_measured_mm: float | None = None   # 物理测量（只作核对）
    created: str = ""
    verdict: str = ""


@dataclass
class SyncInfo:
    method: str = "none"            # "flash" / "clap" / "manual" / "none"
    offset: int = 0                 # left[i] <-> right[i+offset]
    confidence: float = 0.0
    note: str = ""


@dataclass
class Session:
    id: str
    created: str
    label: str = ""
    notes: str = ""
    left: CameraInfo | None = None
    right: CameraInfo | None = None
    calibration: CalibrationInfo | None = None
    stereo_mode: str = "parallel"
    rig: dict[str, Any] = field(default_factory=dict)
    sync: SyncInfo = field(default_factory=SyncInfo)
    codec: str = "FFV1"
    container: str = "mkv"
    lossless: bool = True
    environment: dict[str, Any] = field(default_factory=dict)
    checks: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------
def new_session_id(now: float | None = None) -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime(now))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT), stderr=subprocess.DEVNULL, timeout=5,
        )
        return out.decode("ascii", "replace").strip()
    except Exception:                       # 不在 git 仓库/没装 git 都不该让采集失败
        return ""


def environment_snapshot() -> dict[str, Any]:
    return {
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "platform": platform.platform(),
    }


def _from_dict(cls, data: dict | None):
    if not data:
        return None
    allowed = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in allowed})


# --------------------------------------------------------------------------
# 创建 / 读写
# --------------------------------------------------------------------------
def create_session(
    root: str | Path,
    *,
    label: str = "",
    notes: str = "",
) -> tuple[Path, Session]:
    """在 `root` 下新建一个会话目录（含 `calib/`、`derived/`），返回目录与空会话。

    只建目录和模板 `session.json`；相机/视频/标定由调用方填。
    这样即使录制中途失败，目录里也有一个能说明"本来要录什么"的文件。
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    sid = new_session_id()
    session_dir = root / sid
    suffix = 1
    while session_dir.exists():
        session_dir = root / f"{sid}-{suffix:02d}"
        suffix += 1
    for sub in ("calib", "derived"):
        (session_dir / sub).mkdir(parents=True, exist_ok=True)
    session = Session(
        id=session_dir.name,
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
        label=label,
        notes=notes,
        environment=environment_snapshot(),
    )
    save_session(session, session_dir)
    return session_dir, session


def save_session(session: Session, session_dir: str | Path) -> Path:
    """原子写 `session.json`。"""
    d = Path(session_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / SESSION_FILE
    tmp = path.with_suffix(".json.tmp")
    data = {"schema": SCHEMA, "version": VERSION, **asdict(session)}
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)
    return path


def load_session(path: str | Path) -> tuple[Path, Session]:
    """读 `session.json`（可给目录或文件），返回 `(会话目录, Session)`。"""
    p = Path(path)
    if p.is_dir():
        p = p / SESSION_FILE
    if not p.exists():
        raise SessionError(f"找不到会话文件：{p}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SessionError(f"session.json 不是合法 JSON：{p}（{exc}）") from exc
    if raw.get("schema") != SCHEMA:
        raise SessionError(
            f"不是本项目的会话文件：schema={raw.get('schema')!r}，期望 {SCHEMA!r}")
    version = raw.get("version")
    if not isinstance(version, int) or version > VERSION:
        raise SessionError(
            f"会话版本 {version!r} 高于本代码支持的 {VERSION}；请升级代码或另存副本")
    session = Session(
        id=str(raw.get("id") or p.parent.name),
        created=str(raw.get("created") or ""),
        label=str(raw.get("label") or ""),
        notes=str(raw.get("notes") or ""),
        left=_from_dict(CameraInfo, raw.get("left")),
        right=_from_dict(CameraInfo, raw.get("right")),
        calibration=_from_dict(CalibrationInfo, raw.get("calibration")),
        stereo_mode=str(raw.get("stereo_mode") or "parallel"),
        rig=dict(raw.get("rig") or {}),
        sync=_from_dict(SyncInfo, raw.get("sync")) or SyncInfo(),
        codec=str(raw.get("codec") or "FFV1"),
        container=str(raw.get("container") or "mkv"),
        lossless=bool(raw.get("lossless", True)),
        environment=dict(raw.get("environment") or {}),
        checks=list(raw.get("checks") or []),
    )
    return p.parent, session


def _session_relative(session_dir: str | Path, rel: str | Path) -> Path:
    """``session.json`` 里的相对路径按会话目录解析；绝对路径原样返回。

    「相对就拼、绝对就原样」这条语义在本模块**只有这一份实现**：
    ``session_calibration_file``（要路径，缺了抛错）与 ``validate_session``
    （要路径，缺了记 error 不抛）共用它。两处各写一遍的话，哪天一处把
    绝对/相对判反了，另一处仍然"看着对"，问题只在某类会话上出现。

    注：``pathlib`` 的 ``/`` 运算符本来就会让绝对路径覆盖左侧，所以
    ``p if p.is_absolute() else ...`` 与裸 ``session_dir / p`` **结果相同**
    （实测 4 类输入：相对 / 带盘符绝对 / POSIX 绝对 / ``..`` 相对）。
    这里显式写出来是为了让"绝对路径原样"这条**约定**看得见 —— 不是 bug 修复，
    也不要因为"看着冗余"就删掉：``session.json`` 的契约是"相对会话目录"，
    绝对路径只是容忍形式，写在代码里比藏在 pathlib 的运算符语义里可靠。
    """
    p = Path(rel)
    return p if p.is_absolute() else Path(session_dir) / p


def session_calibration_file(session_dir: str | Path) -> Path:
    """会话使用的**标定文件路径**的唯一产地（其它模块不许再手写文件名）。

    ``session.json`` 的会话契约就在本模块（``CalibrationInfo.path``）。
    标定文件名是**跟着激活装置档案动态生成**的：录制时
    ``apps/record_stereo.py::_copy_calibration`` 按
    ``calib_base.with_suffix(".npz").name`` 把标定拷进 ``calib/``，
    于是老会话写的是 ``calib/stereo_calibration.npz``，新会话写的是
    ``calib/<装置档案名>.npz``。两者都得从 ``session.json`` 里读 ——
    把文件名写死成 ``stereo_calibration.npz`` 就会让新会话断链
    （找得到目录、找不到文件，或更糟：静默读到别套装置的几何）。

    只做**纯 JSON 读取**，不调 ``load_session``：后者会做 schema/版本校验，
    而这里只要一个路径字段，不该把视频能不能读之类的校验牵扯进来。
    ``calibration.path`` 是相对会话目录的路径（录制时这么写），相对就拼
    ``session_dir``，绝对就原样返回。

    缺 ``session.json``、缺 ``calibration``、缺 ``path``、JSON 坏了时抛
    ``SessionError`` 并说明原因 —— **明确报错**，不抛裸 ``KeyError``、
    不静默返回 ``None``（调用方要"没标定就降级"得自己 catch 并说清原因）。

    代价：本函数在 ``stereo`` 包内，import 会带 cv2/scipy。收敛到的 5 个
    消费方都是 CLI 工具（本来就要读视频/标定），可接受；**不要**把它塞进
    GUI 热路径。
    """
    p = Path(session_dir)
    if p.is_dir():
        p = p / SESSION_FILE
    if not p.is_file():
        raise SessionError(f"找不到会话文件：{p}（无法确定会话标定路径）")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SessionError(f"session.json 不是合法 JSON：{p}（{exc}）") from exc
    cal = raw.get("calibration") if isinstance(raw, dict) else None
    if not isinstance(cal, dict) or not cal.get("path"):
        raise SessionError(
            f"会话文件缺少 calibration.path：{p}（标定文件名跟随装置档案动态生成，"
            "不能写死默认名；这个会话可能没录完）")
    cal_path = _session_relative(p.parent, str(cal["path"]))
    return cal_path


def list_sessions(root: str | Path) -> list[Path]:
    """列出 ``root`` 下**所有**含 ``session.json`` 的会话目录，按名字排序。

    **递归**：除了 ``<根>/<时间戳>`` 与 ``<根>/<模式>/<时间戳>``，用户自己建的
    嵌套目录（``<根>/实验1/梨子/<时间戳>``）也算 —— 只要目录里有
    ``session.json``。判据是严的（只认 ``session.json``）：这里列出来的目录
    要拿去 ``load_session`` / 跑算法，没有元数据跑不了；"半成品也列出来"是
    数据管理的口径，走 ``discover_session_dirs(strict=False)``。

    以前这里是一行 ``glob(f"*/{SESSION_FILE}")``，**只扫一层**，于是
    ``data/stereo/parallel/`` 与 ``data/stereo/angled/`` 里的会话全漏
    （实测 29 段 vs 实际 44 段）：数据管理看得见、会话列表看不见。现在与
    数据管理共用 ``sphere_tracker.discovery`` 同一份枚举，只有一份实现。
    """
    return sorted(discover_session_dirs(root, recursive=True, strict=True))


# --------------------------------------------------------------------------
# 校验
# --------------------------------------------------------------------------
def _video_facts(path: Path) -> tuple[bool, int, int, int, float]:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return False, 0, 0, 0, 0.0
        return (
            True,
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            float(cap.get(cv2.CAP_PROP_FPS)),
        )
    finally:
        cap.release()


def _video_brightness_mean(path: Path, samples: int = 5) -> float:
    """抽几帧算平均亮度；打不开/没帧返回 0。"""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return 0.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0:
            return 0.0
        vals = []
        for k in range(max(1, samples)):
            idx = int((n - 1) * k / max(1, samples - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok and frame is not None:
                vals.append(float(frame.mean()))
        return sum(vals) / len(vals) if vals else 0.0
    finally:
        cap.release()


def _sidecar_fps(
    video_path: Path, frame_count: int,
) -> tuple[float | None, str, bool]:
    """用逐帧时间戳算**真实帧率**；返回 `(fps 或 None, 说明, sidecar 是否坏了)`。

    第三项区分"**没有** sidecar"（旧会话，合法）与"**有但坏**"（不完整/不单调，
    必须独立上报）：前者只是无法独立校准，后者说明这份 sidecar 本身不能信。

    为什么不能只看容器 fps：容器头里的 fps 是写入时给的值（曾经直接抄 ``--fps``
    默认 24），而相机实际跑 29.99。播放器按容器 fps 换算秒数就会系统性偏慢。
    sidecar 是逐帧的主机时间，是唯一能独立校准时间基的证据 ——
    但它必须**完整**（行数对上视频帧数、时间戳严格递增），否则不能当时间基用。
    """
    from .timestamp_pair import read_time_sidecar, sidecar_for_video

    sidecar = sidecar_for_video(video_path)
    if sidecar is None:
        return None, "缺 sidecar，无法独立校准时间基", False
    try:
        times = read_time_sidecar(sidecar)
    except Exception as exc:                    # noqa: BLE001 —— 坏 sidecar 只降级，不许抛
        return None, f"sidecar 不可用（{exc}），时间基不可靠", True
    if times.size < 2:
        return None, f"sidecar 不完整（{times.size} 行 / {frame_count} 帧），时间基不可靠", True
    if frame_count > 0 and abs(int(times.size) - frame_count) > 1:
        return None, (f"sidecar 不完整（{times.size} 行 / {frame_count} 帧），时间基不可靠"
                      "（按容器 fps 分析会算错秒数）"), True
    dts = [float(b - a) for a, b in zip(times[:-1], times[1:])]
    span = float(times[-1] - times[0])
    if span <= 0 or statistics.median(dts) <= 0:
        return None, "sidecar 时间戳没有推进，时间基不可靠", True
    # 用**总跨度**算持续帧率，而不是 1/中位间隔：USB 抖动不对称（偶发长间隔多、
    # 短的重复读少），中位数会把 29.99 压成 29.3；而"播放会慢多少"取决于总时长，
    # 正好就是 (帧数-1)/跨度。
    return (len(times) - 1) / span, "", False


def _sharpness(
    video_path: Path, *, warmup: int = 30, shots: int = 8,
) -> tuple[float, float] | None:
    """算 `(Laplacian 方差中位数, 平均亮度)`；读不到返回 None。

    口径**完全对齐** `apps/tools/camera_quality.py::measure()`：共用库层的同一个
    `frames_quality()`、同一 25–75% 中心裁切、同一"跳过预热再连取若干帧"的采样
    （那里是预热 30 帧、留最多 8 张；这里照抄这两个数）。差别只有输入：
    本函数吃录像，那个吃实时相机 —— 所以**数值只能在同分辨率/同场景下互相比较**，
    换个分辨率（Live 2560x1440 vs 录像 1718x972）这个绝对值没有可比性。
    只有"读不到帧"才返回 None；画质计算本身不再有兜底 —— 依赖是库内正常 import，
    出问题就该响亮地失败，不该静默降级。
    """
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return None
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0:
            return None
        start = min(max(0, warmup), max(0, n - 1))
        if start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        grays = []
        while len(grays) < max(1, shots):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if not grays:
            return None
    finally:
        cap.release()
    m = frames_quality(grays)
    return float(m.sharp), float(m.bright)


def validate_session(
    path: str | Path, *, with_notes: bool = False,
) -> tuple[list[str], list[str]] | tuple[list[str], list[str], list[str]]:
    """校验一个会话，返回 `(errors, warnings[, notes])`。

    **不抛异常**：坏会话要能一次列出所有问题，而不是停在第一个。
    是否把 warning 当失败由调用方决定（CLI 的 `--strict`）。

    三档语义：
    - `errors`：硬失败，数据不可用；
    - `warnings`：可疑，`--strict` 下算失败；
    - `notes`：只是告知（例如容器标称 fps 不准但 sidecar 齐全），**永不导致失败**，
      要 `with_notes=True` 才返回（保持老调用方的二元组不变）。
    """
    errors: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []
    try:
        session_dir, s = load_session(path)
    except SessionError as exc:
        bad = [str(exc)]
        return (bad, warnings, notes) if with_notes else (bad, warnings)

    if s.left is None:
        errors.append("缺少 left 相机信息")
    if s.right is None:
        errors.append("缺少 right 相机信息")

    frame_counts: dict[str, int] = {}
    sharpness: dict[str, tuple[float, float]] = {}
    for side, cam in (("left", s.left), ("right", s.right)):
        if cam is None:
            continue
        if not cam.video:
            errors.append(f"{side}: 没有 video 字段")
            continue
        video_path = Path(cam.video)
        if not video_path.is_absolute():
            video_path = session_dir / video_path
        if not video_path.exists():
            errors.append(f"{side}: 视频不存在：{video_path}")
            continue
        ok, w, h, n, fps = _video_facts(video_path)
        if not ok:
            errors.append(f"{side}: OpenCV 打不开视频：{video_path}")
            continue
        frame_counts[side] = n
        if (w, h) != (cam.width, cam.height):
            errors.append(
                f"{side}: 实际 {w}x{h} 与 session 记录的 {cam.width}x{cam.height} 不一致")
        if cam.frame_count and abs(n - cam.frame_count) > 1:
            errors.append(
                f"{side}: 实际 {n} 帧，session 记录 {cam.frame_count} 帧（差 >1）")
        # 容器 fps 是录制时写的标称值（两路取协商值的 min，见 record_stereo）；
        # 两路差 2% 只影响播放器观感，真正的时间基是 sidecar，所以这里只跟它比。
        real_fps, why, bad_sidecar = _sidecar_fps(video_path, cam.frame_count or n)
        if bad_sidecar:
            # sidecar 存在但坏了：独立上报，不依赖 fps_measured 是否存在
            # （否则 fps_measured 缺失时这条会完全静默）
            warnings.append(f"{side}: {why}")
        if real_fps is not None:
            # sidecar 齐全 = 有自己的时间基，容器标称不准只影响播放器观感
            if fps > 0:
                ratio = abs(fps - real_fps) / real_fps
                speed = (real_fps / fps - 1.0) * 100.0
                word = "慢" if speed > 0 else "快"
                if ratio > 0.25:
                    # 差这么多不可能是"标称记法"，是容器头真的写错了时间基，
                    # 下游按容器换算秒数会系统性出错，必须拦
                    warnings.append(
                        f"{side}: 容器 fps={fps:.2f} 与 sidecar 实测 {real_fps:.2f} 差 "
                        f"{ratio*100:.0f}%（>25% 不是标称偏差，是时间基错误：容器头写错，"
                        "播放器/FileSource 会按错误秒数换算）")
                elif ratio > 0.10:
                    notes.append(
                        f"{side}: 容器标称 {fps:.1f} fps，sidecar 实测 {real_fps:.2f} fps"
                        f"（差 {ratio*100:.0f}%，标称偏差；按 sidecar 分析即可，"
                        f"用播放器看会{word} {abs(speed):.0f}%）")
            if cam.fps_measured:
                drift = abs(cam.fps_measured - real_fps) / real_fps
                if drift > 0.05:
                    notes.append(
                        f"{side}: 主循环实测 {cam.fps_measured:.2f} fps 与 sidecar "
                        f"{real_fps:.2f} fps 差 {drift*100:.0f}%"
                        "（主循环节奏与写盘节奏不一致，分析以 sidecar 为准）")
        elif cam.fps_measured and fps > 0 and not bad_sidecar:
            # 没有 sidecar（不是坏了）时，容器 fps 就是唯一时间基，标称错会真的算错秒数。
            ratio = abs(fps - cam.fps_measured) / max(cam.fps_measured, 1e-9)
            if ratio > 0.10:
                warnings.append(
                    f"{side}: 容器 fps={fps:.2f} 与实测 {cam.fps_measured:.2f} 差 "
                    f"{ratio*100:.0f}%（{why}）")
        mean_brightness = _video_brightness_mean(video_path)
        if mean_brightness < 20.0:
            warnings.append(
                f"{side}: 视频几乎全黑（抽样平均亮度 {mean_brightness:.1f} < 20）"
                "—— 检查镜头盖/灯光/曝光；黑视频不能做检测")
        sharp = _sharpness(video_path)
        if sharp is not None:
            sharpness[side] = sharp

    if len(sharpness) == 2:
        (l_sharp, l_bright), (r_sharp, r_bright) = sharpness["left"], sharpness["right"]
        ratio = max(l_sharp, r_sharp) / max(min(l_sharp, r_sharp), 1e-6)
        res = f"{s.left.width}x{s.left.height}" if s.left else "录像原生分辨率"
        line = (f"清晰度（Laplacian 方差中位数，中心 25–75% 区，{res}）"
                f"left={l_sharp:.1f}（亮度 {l_bright:.0f}） "
                f"right={r_sharp:.1f}（亮度 {r_bright:.0f}）（比值 {ratio:.2f}")
        # 只做提示、不进 warnings：`check_dataset` 是**数据有效性**闸门，清晰度差
        # 不等于数据无效（三维精度由下游实测），而且这条判据会让已入库的旧会话被
        # 追溯定罪。清晰度的权威判据留在 `camera_quality.py`（它自己的入口）。
        if ratio >= 1.8:
            notes.append(
                line + "，≥1.8 建议查对焦/曝光，详见 apps\\tools\\camera_quality.py）")
        elif ratio >= 1.3:
            notes.append(line + "，1.3–1.8 属同型号个体差异，不影响录球）")
        else:
            notes.append(line + "，<1.3 视为两路相当）")

    if len(frame_counts) == 2 and abs(frame_counts["left"] - frame_counts["right"]) > 1:
        warnings.append(
            f"左右帧数不一致：{frame_counts['left']} vs {frame_counts['right']}"
            "（同步偏移只能吸收常数差，掉帧会累积）")

    cal = s.calibration
    if cal is None:
        errors.append("缺少 calibration 信息")
    else:
        cal_path = _session_relative(session_dir, cal.path)
        if not cal_path.exists():
            errors.append(f"标定文件不存在：{cal_path}")
        else:
            if cal.sha256:
                actual = sha256_file(cal_path)
                if actual != cal.sha256:
                    errors.append(
                        f"标定文件 sha256 不匹配：记录 {cal.sha256[:12]}…，实际 {actual[:12]}…")
            try:
                rig = StereoRig.load(cal_path.with_suffix("") if cal_path.suffix else cal_path)
            except Exception as exc:        # noqa: BLE001 —— 坏标定要报出来，不能让采集崩
                errors.append(f"标定文件读不了：{exc}")
            else:
                if cal.image_size and tuple(rig.image_size) != tuple(cal.image_size):
                    errors.append(
                        f"标定 image_size={rig.image_size} 与 session 记录 {cal.image_size} 不一致")
                if cal.baseline_mm > 0:
                    ratio = abs(rig.baseline_mm - cal.baseline_mm) / cal.baseline_mm
                    if ratio > 0.01:
                        warnings.append(
                            f"标定文件基线 {rig.baseline_mm:.2f} mm 与 session 记录 "
                            f"{cal.baseline_mm:.2f} mm 差 {ratio*100:.1f}%")
                if (cal.baseline_measured_mm is not None
                        and cal.baseline_measured_mm > 0
                        and rig.baseline_mm > 0):
                    ratio = abs(cal.baseline_measured_mm - rig.baseline_mm) / rig.baseline_mm
                    if ratio > 0.20:
                        errors.append(
                            f"物理基线 {cal.baseline_measured_mm:.2f} mm 与算法 "
                            f"{rig.baseline_mm:.2f} mm 差 {ratio*100:.0f}%（>20%，先核对方格/左右）")
                    elif ratio > 0.10:
                        warnings.append(
                            f"物理基线 {cal.baseline_measured_mm:.2f} mm 与算法 "
                            f"{rig.baseline_mm:.2f} mm 差 {ratio*100:.0f}%（>10%）")
                if str(getattr(s, "stereo_mode", "parallel")) != rig.stereo_mode:
                    warnings.append(
                        f"会话模式 {getattr(s, 'stereo_mode', 'parallel')} 与标定 meta "
                        f"{rig.stereo_mode} 不一致；分析前请核对")
                angle = optical_axes_angle_deg(rig.R)
                if getattr(s, "stereo_mode", "parallel") == "parallel" and angle >= 40.0:
                    warnings.append(
                        f"标定光轴夹角 {angle:.1f}° 很大，但会话按 parallel 记录；"
                        "如果这是大角度装置，请用 angled 模式重录/显式覆盖")
                # 标定质量里的 epipolar_rms 永远是 parallel 口径（校正后 y 差）。
                # 大角度装置下校正图像被极端拉伸，这个数不代表几何精度，
                # 不能拿来判"标定可用"，也不能与会话复检的其他口径相减比较。
                if rig.stereo_mode == "angled":
                    try:
                        q_epi = float(rig.quality.get("epipolar_rms"))
                    except (TypeError, ValueError):
                        q_epi = float("nan")
                    if q_epi == q_epi:
                        warnings.append(
                            f"标定质量 epipolar_rms={q_epi:.3f} px 是 parallel 口径"
                            "（校正后 y 差），angled 装置下不代表几何精度；"
                            "复检请用 StereoRig.epipolar_error_by_mode('angled', ...) "
                            "的口径另行判读")

    if abs(s.sync.offset) > 180:
        warnings.append(f"同步偏移 {s.sync.offset:+d} 超过默认搜索上限 ±180 帧")
    if s.sync.method == "none":
        warnings.append(
            "未同步（sync.method=none）：没有闪光/拍手偏移，三维轨迹不能直接使用"
            "（重录并在“开始”后闪光，或用 --sync-offset 手动指定）")
    if s.sync.method != "none" and s.sync.confidence <= 0:
        warnings.append(f"同步方法={s.sync.method} 但置信度={s.sync.confidence}（未真正算出偏移？）")
    if not s.lossless:
        warnings.append(f"会话标记为有损录制（codec={s.codec}）；分析前请确认这是有意的")
    return (errors, warnings, notes) if with_notes else (errors, warnings)


def describe(session: Session) -> str:
    """一行到三行的人读摘要（给 CLI / 日志用）。"""
    lines = [f"会话 {session.id}  {session.created}  label={session.label or '-'}"]
    for side, cam in (("左", session.left), ("右", session.right)):
        if cam is None:
            lines.append(f"  {side}: （缺失）")
        else:
            fps = cam.fps_measured or cam.fps_requested
            lines.append(
                f"  {side}: {cam.width}x{cam.height} @ {fps:.1f} fps  "
                f"{cam.frame_count} 帧  {cam.video}")
    if session.calibration is not None:
        cal = session.calibration
        extra = ""
        if cal.baseline_measured_mm is not None:
            extra = f"（物理 {cal.baseline_measured_mm:.2f} mm）"
        lines.append(f"  标定: 基线 {cal.baseline_mm:.2f} mm{extra}  {cal.path}")
    lines.append(
        f"  同步: {session.sync.method} {session.sync.offset:+d} 帧  "
        f"置信度 {session.sync.confidence:.3f}")
    return "\n".join(lines)
