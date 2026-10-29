"""单目会话：一个目录 = 一次录制 + 它派生出的检测结果。

为什么需要这一层（2026-10-29 用户："单目的会话管理也还没有实现"）
------------------------------------------------------------------
双目早就有 ``data/sessions/<YYYYMMDD-HHMMSS>/session.json``；单目只有
``apps/record_mono.py`` 写下的一对散文件（``<out>.mkv`` + ``<out>.mkv.json``）。
散文件的后果和双目那边一模一样：一周后只剩两段视频，不知道是谁录的、
录完有没有跑过检测、结果在哪。所以单目也给它"目录 = 一次会话"的形状，
**与双目同构**（同样 ``data/<模式>/<时间戳>/`` + ``session.json`` + ``derived/``），
让人一眼看得出是同类东西。

目录形态::

    data/mono/<YYYYMMDD-HHMMSS>/
        clip.mkv          # record_mono 的产物（编码器以它为准）
        clip.mkv.json     # record_mono 写的自证元数据（分辨率/fps/帧数/编码器）
        session.json      # 本模块维护：这次录了什么
        derived/          # 检测结果（traj.csv / annotated.mp4），由跑检测的一方建

为什么**不复用** ``stereo/sessions.py``
--------------------------------------
那份 schema 的核心是 ``left``/``right``/``calibration``/``sync`` —— 单目一个都
用不上，硬套只会得到一半字段永远为空的对象，还要连坐 import ``calib.params``。
这里自带 schema，但**字段名对齐**（``camera``/``width``/``height``/``fps``/
``frame_count``/``duration_s``/``encoder``/``created``）：读代码的人不需要
学第二套词汇。

为什么**不改** ``apps/record_mono.py``
-------------------------------------
录制器的职责是"开相机、写视频、写自证元数据"，它不该知道"会话"这件事。
它已经有 ``--out``，调用方传 ``<会话目录>/clip.mkv`` 即可，那对 sidecar 会自动
落在同目录 —— 目录形态由调用方决定，这是干净的职责线。

公开接口（纯逻辑，不 import tkinter，可单测）
-------------------------------------------
``scan_mono_sessions`` 扫磁盘 / ``MonoSession`` 摘要 /
``create_mono_session`` 建目录 / ``delete_mono_session`` 软删除；
另有 ``find_clip`` 与 ``sync_from_take`` 两个小工具 —— 录制结束要把 sidecar 的
实测值写回 ``session.json``，这段解析**只能有一份实现**，不能复制到窗口里。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

SCHEMA = "sphere-mono-session"
VERSION = 1
SESSION_FILE = "session.json"
CLIP_STEM = "clip"
#: 认作"原始视频"的扩展名。录制器默认 HuffYUV+mkv，mp4v 时是 mp4。
VIDEO_SUFFIXES = (".mkv", ".mp4", ".avi", ".mov")
#: ``record_mono`` 自证元数据的 schema（只读它，不写）。
TAKE_SCHEMA = "mono-take/1"
TRASH_LABEL = "mono"

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "data" / "mono"


# --------------------------------------------------------------------------
# 数值兜底
# --------------------------------------------------------------------------
def _num(value, cast, default):
    """宽松取值：``session.json`` 可能被手改过（``"25"`` 与 ``25`` 都收）。

    取不出来就用默认值 —— 摘要里少一个数字远好过整个列表崩掉。
    """
    try:
        return cast(value)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------
# 磁盘上的东西
# --------------------------------------------------------------------------
def find_clip(session_dir: str | Path) -> Path | None:
    """会话里那段原始视频；没有就返回 ``None``。

    先认 ``clip.<ext>``（窗口约定的名字），认不到再退回"目录里的视频"——
    老的单目散文件可以直接拷进会话目录，那时名字是时间戳。
    **只看会话目录顶层**：``derived/`` 里是标注后的成品，拿它再跑一遍检测
    等于在叠加层上再叠一层（这正是 ``record_mono`` 存在的理由）。
    """
    d = Path(session_dir)
    for suffix in VIDEO_SUFFIXES:
        p = d / f"{CLIP_STEM}{suffix}"
        if p.is_file():
            return p
    try:
        clips = sorted(p for p in d.iterdir()
                       if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)
    except OSError:
        return None
    return clips[0] if clips else None


def _take_sidecar(session_dir: Path) -> tuple[Path | None, dict]:
    """``record_mono`` 写的自证文件 → ``(路径, 内容)``；没有则 ``(None, {})``。

    先按视频名找（``clip.mkv.json``，约定），找不到再扫顶层 ``*.json`` 认
    ``schema`` —— 视频被改名/挪走时，"这段本来录成什么样"仍然读得回来。
    坏 JSON 静默跳过：扫描不能因为一个坏文件中断。
    """
    d = Path(session_dir)
    candidates: list[Path] = []
    clip = find_clip(d)
    if clip is not None:
        candidates.append(clip.with_suffix(clip.suffix + ".json"))
    try:
        candidates += [p for p in sorted(d.glob("*.json"))
                       if p.name != SESSION_FILE]
    except OSError:
        pass
    for p in candidates:
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        # schema 对得上就用；老格式没有 schema 时按字段形状认（两个都必需）
        if isinstance(raw, dict) and (raw.get("schema") == TAKE_SCHEMA
                                      or ("frames" in raw and "actual" in raw)):
            return p, raw
    return None, {}


def _facts_from_take(raw: dict) -> dict:
    """把录制器的 meta 翻成会话字段（只取本模块认的键）。"""
    actual = raw.get("actual") if isinstance(raw.get("actual"), dict) else {}
    camera = raw.get("camera") if isinstance(raw.get("camera"), dict) else {}
    role = str(camera.get("requested") or "")
    if not role and camera.get("index") is not None:
        role = f"#{camera.get('index')}"
    elif role.isdigit():
        # 命令行里是 `--camera 0`（纯索引），显示成 `#0` —— 与界面下拉同一套写法，
        # 否则会话摘要里会出现一个光秃秃的 "0"，看不出那到底是索引还是角色名
        role = f"#{role}"
    return {
        "camera": role,
        "width": _num(actual.get("width"), int, 0),
        "height": _num(actual.get("height"), int, 0),
        "fps": _num(actual.get("fps"), float, 0.0),
        "frame_count": _num(raw.get("frames"), int, 0),
        "duration_s": _num(raw.get("seconds"), float, 0.0),
        "encoder": str(raw.get("codec") or ""),
        "recorded_at": str(raw.get("created") or ""),
        "video": str(raw.get("video") or ""),
    }


def _has_result(derived: Path) -> bool:
    """``derived/`` 里有没有"检测跑过一轮"的证据。

    只认轨迹 CSV 与视频：目录里躺着别的中途产物（png、日志）不能算跑完 ——
    否则"已检测"这个状态会骗人。
    """
    if not derived.is_dir():
        return False
    try:
        names = [p.name.lower() for p in derived.iterdir() if p.is_file()]
    except OSError:
        return False
    return any(n.endswith(".csv") for n in names) or \
        any(n.endswith(VIDEO_SUFFIXES) for n in names)


# --------------------------------------------------------------------------
# 摘要
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class MonoSession:
    """一段单目会话的摘要（**只读**：构造它不碰磁盘）。

    字段名刻意与双目 ``session.json`` 对齐，让人一眼看出是同类东西；
    但**没有** ``left``/``right``/``calibration`` 这些双目前提 ——
    单目的深度靠球径先验，不需要第二台相机。
    """

    path: Path
    id: str
    created: str
    label: str = ""
    camera: str = ""
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frame_count: int = 0
    duration_s: float = 0.0
    encoder: str = ""
    video: str = ""            # 相对会话目录的视频名；空 = 还没录
    has_derived: bool = False
    has_result: bool = False
    error: str = ""            # 坏目录/坏 json 的说明；正常为空
    note: str = ""             # 不致命但要说的事（例如"缺 session.json"）

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def state(self) -> str:
        """四态，直接印在界面上：坏会话 / 未录制 / 已录未检测 / 已检测。"""
        if self.error:
            return "坏会话"
        if not self.video:
            return "未录制"
        return "已检测" if self.has_result else "已录，未检测"

    def summary(self) -> str:
        """一行中文摘要：什么时间、哪台相机、什么规格、有没有跑过检测。"""
        if self.error:
            return f"{self.id} —— {self.state}：{self.error}"
        bits: list[str] = []
        if self.width and self.height:
            bits.append(f"{self.width}x{self.height}")
        if self.fps > 0:
            bits.append(f"@{self.fps:.1f} fps")
        if self.frame_count:
            bits.append(f"{self.frame_count} 帧")
        if self.duration_s > 0:
            bits.append(f"{self.duration_s:.1f} s")
        if self.encoder:
            bits.append(self.encoder)
        if self.camera:
            bits.append(f"相机 {self.camera}")
        line = f"{self.id}  {self.created or '（无时间）'}"
        if bits:
            line += "  " + "  ".join(bits)
        line += f" —— {self.state}"
        if self.note:
            line += f"｜{self.note}"
        return line


def _summarize(session_dir: Path) -> MonoSession:
    """一个目录 → 一个 ``MonoSession``；**任何情况下都返回**（坏的带 ``error``）。

    三种目录要分清（2026-10-29 起 ``record_mono`` 默认也落成会话形态了）：
    - 有 ``session.json``：正常会话，``session.json`` 是权威；
    - 只有 ``clip.mkv`` + ``clip.mkv.json``：**CLI 直录**的会话 —— 合法，
      只是还没人给它写 ``session.json``。值从录制器元数据兜底，``note`` 里说明
      缺什么（界面选中它时会自动补成正式会话）；
    - 什么都没有：才是 ``error``（手建的杂目录）。
    """
    raw: dict = {}
    error = ""
    note = ""
    take: dict = {}
    session_file = session_dir / SESSION_FILE
    if not session_file.is_file():
        _side, take = _take_sidecar(session_dir)
        if not take:
            error = "没有 session.json（不是单目会话目录，或文件被删了）"
        else:
            note = "只有录制器写的元数据，还没有 session.json（CLI 直接录的）"
    else:
        try:
            loaded = json.loads(session_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            error = f"session.json 读不了：{exc}"
        else:
            if not isinstance(loaded, dict):
                error = "session.json 不是 JSON 对象（结构不对）"
            elif loaded.get("schema") != SCHEMA:
                error = (f"不是单目会话：schema={loaded.get('schema')!r}"
                         f"（期望 {SCHEMA}）")
            else:
                raw = loaded
                _side, take = _take_sidecar(session_dir)

    clip = find_clip(session_dir)
    facts = _facts_from_take(take) if take else {}

    derived = session_dir / "derived"
    return MonoSession(
        path=session_dir,
        id=session_dir.name,
        created=str(raw.get("created") or "") or facts.get("recorded_at", ""),
        label=str(raw.get("label") or ""),
        camera=str(raw.get("camera") or "") or facts.get("camera", ""),
        # session.json 是权威；它还没被补（录制器直接跑、窗口没接上）时用 sidecar 兜底
        width=_num(raw.get("width"), int, 0) or facts.get("width", 0),
        height=_num(raw.get("height"), int, 0) or facts.get("height", 0),
        fps=_num(raw.get("fps"), float, 0.0) or facts.get("fps", 0.0),
        frame_count=_num(raw.get("frame_count"), int, 0) or facts.get("frame_count", 0),
        duration_s=_num(raw.get("duration_s"), float, 0.0) or facts.get("duration_s", 0.0),
        encoder=str(raw.get("encoder") or "") or facts.get("encoder", ""),
        video=(clip.name if clip is not None else
               (str(raw.get("video") or "") or facts.get("video", ""))),
        has_derived=derived.is_dir(),
        has_result=_has_result(derived),
        error=error,
        note=note,
    )


def _sort_key(session_dir: Path) -> tuple[int, str, float]:
    """倒序排序键：目录名里的 ``YYYYMMDD-HHMMSS`` 优先，**不用 mtime**。

    为什么不用 mtime：跑一次检测会写 ``derived/``，把旧会话的目录 mtime 刷成
    今天，"昨天那段"于是跳到最前面、默认选中就漂了。这个坑
    ``apps/_workbench.scan_sessions`` 已经吃过一次，这里不重复。
    名字不是标准时间戳的目录排在后面（它们多半是手建/坏目录）。
    """
    m = re.match(r"(\d{8}-\d{6})", session_dir.name)
    if m:
        return (1, m.group(1), 0.0)
    try:
        mtime = session_dir.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (0, session_dir.name, mtime)


def scan_mono_sessions(root: str | Path | None = None) -> list[MonoSession]:
    """扫 ``data/mono/*/``，按会话时间倒序；**坏目录也返回**（带 ``error``）。

    为什么坏目录也要回来：静默跳过会让用户对着"明明录了却没出现在列表里"
    干瞪眼，而 ``data/`` 下的东西又是不可再生的 —— 宁可显示一条红的，
    也不要假装它不存在（与 ``_workbench.scan_sessions`` 同一理由）。
    """
    base = Path(root) if root is not None else DEFAULT_ROOT
    if not base.is_dir():
        return []
    try:
        dirs = [p for p in base.iterdir() if p.is_dir()]
    except OSError:
        return []
    dirs.sort(key=_sort_key, reverse=True)
    return [_summarize(d) for d in dirs]


# --------------------------------------------------------------------------
# 建 / 写 / 软删除
# --------------------------------------------------------------------------
def _template(session_id: str, created: str, camera: str,
              label: str, notes: str) -> dict:
    """新建会话时的 ``session.json`` 骨架（数值全 0 = 还不知道）。"""
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "id": session_id,
        "created": created,
        "label": label,
        "notes": notes,
        "camera": camera,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "frame_count": 0,
        "duration_s": 0.0,
        "encoder": "",
        "video": "",
        "note": ("这个文件记的是「本来要录什么」；实测分辨率/帧数由 "
                 "apps/record_mono.py 的 sidecar 补回（sync_from_take）"),
    }


def _write_session(session_dir: Path, data: dict) -> Path:
    """原子写 ``session.json``（tmp + replace）：中途断电不会留半个文件。"""
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / SESSION_FILE
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    tmp.replace(path)
    return path


def create_mono_session(root: str | Path | None = None, *, camera: str = "",
                        label: str = "", notes: str = "",
                        now: float | None = None) -> Path:
    """建一个会话目录并写 ``session.json``，返回目录。

    **不建 ``derived/``**：先建的话"有没有派生结果"就永远为真，那个字段也就
    没用了。``derived/`` 由跑检测的一方按需创建（``run_video.py --out`` 会自己
    mkdir）。

    目录里此刻还没有视频 —— 视频由 ``record_mono.py --out <dir>/clip.mkv`` 写，
    实测值随后由 :func:`sync_from_take` 补回来。这样即使录制中途失败，
    目录里也有一个能说明"本来要录什么"的文件（与双目 ``create_session`` 同理）。
    """
    base = Path(root) if root is not None else DEFAULT_ROOT
    base.mkdir(parents=True, exist_ok=True)
    session_id = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    session_dir = base / session_id
    suffix = 1
    while session_dir.exists():
        session_dir = base / f"{session_id}-{suffix:02d}"
        suffix += 1
    session_dir.mkdir()
    _write_session(session_dir, _template(
        session_dir.name,
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
        camera, label, notes))
    return session_dir


def _created_from_dirname(name: str) -> str:
    """``20261029-184500`` → ``2026-10-29 18:45:00``；名字不标准就返回空串。"""
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S",
                             time.strptime(name[:15], "%Y%m%d-%H%M%S"))
    except ValueError:
        return ""


def sync_from_take(session_dir: str | Path) -> bool:
    """把录制 sidecar 的**实测值**写回 ``session.json``；没有 sidecar 返回 False。

    为什么要写回而不是让下游每次现读 sidecar：``session.json`` 是"这次录了什么"
    的唯一权威（双目同一条约定），下游不该被要求也懂录制器的格式。
    窗口在录制任务结束时调它；直接用 CLI 录的会话，``scan_mono_sessions`` 会
    用 sidecar 兜底**显示**，但磁盘上仍是空的 —— 那时这里一调就补齐。
    """
    d = Path(session_dir)
    sidecar, take = _take_sidecar(d)
    if not take:
        return False
    facts = _facts_from_take(take)
    try:
        raw = json.loads((d / SESSION_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = {}
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
        # 目录名就是录制时间，sidecar 里的 created 更准；都没有才用"现在" ——
        # 否则 CLI 录的会话会显示成"刚才创建的"，跟目录名对不上。
        created = (facts.get("recorded_at") or _created_from_dirname(d.name)
                   or time.strftime("%Y-%m-%d %H:%M:%S"))
        raw = _template(d.name, created, "", "", "")
    for key in ("width", "height", "fps", "frame_count", "duration_s", "encoder"):
        if facts.get(key):
            raw[key] = facts[key]
    clip = find_clip(d)
    raw["video"] = clip.name if clip is not None else facts.get("video", "")
    # 相机角色是窗口选的（录制器只是照抄），已有值就别被 "--camera 0" 覆盖掉
    raw["camera"] = str(raw.get("camera") or "") or facts.get("camera", "")
    raw["recorded_at"] = facts.get("recorded_at", "")
    if sidecar is not None:
        raw["take_meta"] = sidecar.name
    _write_session(d, raw)
    return True


def delete_mono_session(session_dir: str | Path,
                        root: str | Path | None = None) -> Path:
    """**软删除**：整个会话移到 ``data/_trash/<时间>_mono/``，返回那一批的路径。

    为什么不 ``rmtree``：录像是不可再生的。项目已经有回收站语义
    （``apps/_cleanup.py``：预览 → 移动 → 可恢复/可彻底删），这里**复用**它，
    而不是再写一套移动逻辑 —— 恢复与彻底删除也就自动共用同一份实现。
    """
    # 延迟 import：扫描/创建是纯读，没必要让每次 import 都背上回收站的细节
    from apps._cleanup import move_to_trash, plan_cleanup

    d = Path(session_dir)
    base = Path(root) if root is not None else d.parent
    plan = plan_cleanup(base, [d], mode="session")
    if not plan.items:
        raise ValueError("没有可删除的会话：" + ("；".join(plan.skipped) or str(d)))
    return move_to_trash(base, plan, label=TRASH_LABEL)
