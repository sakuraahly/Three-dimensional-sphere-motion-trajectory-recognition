"""双目采集会话：一个目录 = 一次采集，`session.json` 是唯一权威元数据。

为什么需要这一层
----------------
录制数据既不是"可再生的 `outputs/`"，也不是"入库的 `assets/`"：
它体积大、不可再生、丢了就没了。所以给它单独的 `data/` 根，
并用一个 schema 把"这次到底录了什么、用什么参数、标定是哪一份"固定下来。
否则一周后你只有两段 mkv，不知道分辨率 / 基线 / 同步偏移，数据等于废的。

用法::

    from stereo.sessions import create_session, save_session, load_session, validate_session

    session_dir, s = create_session("data/sessions", label="ball01")
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
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2

from calib.params import StereoRig

SCHEMA = "sphere-stereo-session"
VERSION = 1
SESSION_FILE = "session.json"
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
        sync=_from_dict(SyncInfo, raw.get("sync")) or SyncInfo(),
        codec=str(raw.get("codec") or "FFV1"),
        container=str(raw.get("container") or "mkv"),
        lossless=bool(raw.get("lossless", True)),
        environment=dict(raw.get("environment") or {}),
        checks=list(raw.get("checks") or []),
    )
    return p.parent, session


def list_sessions(root: str | Path) -> list[Path]:
    """列出 `root` 下所有含 `session.json` 的目录，按名字排序。"""
    return sorted(p.parent for p in Path(root).glob(f"*/{SESSION_FILE}"))


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


def validate_session(path: str | Path) -> tuple[list[str], list[str]]:
    """校验一个会话，返回 `(errors, warnings)`。

    **不抛异常**：坏会话要能一次列出所有问题，而不是停在第一个。
    是否把 warning 当失败由调用方决定（CLI 的 `--strict`）。
    """
    errors: list[str] = []
    warnings: list[str] = []
    try:
        session_dir, s = load_session(path)
    except SessionError as exc:
        return [str(exc)], warnings

    if s.left is None:
        errors.append("缺少 left 相机信息")
    if s.right is None:
        errors.append("缺少 right 相机信息")

    frame_counts: dict[str, int] = {}
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
        if cam.fps_measured and fps > 0:
            ratio = abs(fps - cam.fps_measured) / max(cam.fps_measured, 1e-9)
            if ratio > 0.10:
                warnings.append(
                    f"{side}: 容器 fps={fps:.2f} 与实测 {cam.fps_measured:.2f} 差 {ratio*100:.0f}%"
                    "（容器 fps 可能只是标称值，不一定是错）")

    if len(frame_counts) == 2 and abs(frame_counts["left"] - frame_counts["right"]) > 1:
        warnings.append(
            f"左右帧数不一致：{frame_counts['left']} vs {frame_counts['right']}"
            "（同步偏移只能吸收常数差，掉帧会累积）")

    cal = s.calibration
    if cal is None:
        errors.append("缺少 calibration 信息")
    else:
        cal_path = Path(cal.path)
        if not cal_path.is_absolute():
            cal_path = session_dir / cal_path
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
    return errors, warnings


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
