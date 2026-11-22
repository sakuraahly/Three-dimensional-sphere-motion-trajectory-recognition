"""数据管理面板的纯导出层：选择会话、脱敏、按类别打包。

放在 `apps/` 而不是 `stereo/`：这是“给人导出交付包”的编排逻辑，不是双目库。
GUI 只负责勾选和展示，真正的文件收集/脱敏/zip 都在这里，便于测试。
"""

from __future__ import annotations

import json
import os
import re
import zipfile
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

# 原始录像的文件名与收集次序**只有** `sphere_tracker/video_files.py` 一份：
# 这里按它的顺序把文件收进 `videos/raw/`，不许再抄一遍元组。
from sphere_tracker.video_files import ALL_RAW_VIDEO_NAMES

from ._workbench import STATUS_SYMBOLS, summarize_session

CATEGORIES: dict[str, str] = {
    "metadata": "会话元数据（session.json + 时间戳）",
    "derived": "标准派生数据（seeds / traj3d / visualization / CSV）",
    "plots": "派生图（traj3d.png / sync_brightness.png）",
    "annotated": "标注视频（有画面，含隐私风险）",
    "raw": "原始双目视频（很大，有画面，含隐私风险）",
    "calibration": "该会话的标定参数（JSON/NPZ）",
}
DEFAULT_CATEGORIES = frozenset({"metadata", "derived", "plots", "calibration"})

REDACT_KEYS_CAMERA = ("identity", "name", "port")
REDACT_KEYS_TOP = ("label", "notes")


class ItemList(list):
    """``scan_items`` 的返回值：一个普通 list，外加**扫描时发现的异常**。

    为什么不用另一个 dataclass 包一层：``export_zip`` / ``estimate_bytes`` /
    调用方的 ``for it in items`` 全都按 list 用，换类型要改十几处。
    继承 ``list`` 就零成本 —— 但异常**必须能被调用方看见**：sid 撞车时
    自动加后缀是"修好了"，可用户得知道有两个会话本来同一个 id。
    """

    warnings: list[str]

    def __init__(self, iterable=(), warnings: list[str] | None = None) -> None:
        super().__init__(iterable)
        self.warnings = list(warnings or [])


@dataclass(frozen=True)
class SessionItem:
    path: Path
    id: str
    created: str
    frames_left: int
    frames_right: int
    n_matched: int
    status: str
    size_bytes: int
    #: 记录的模式；**空串 = 没记录**（不是 parallel）。老会话缺这个字段，
    #: 默认成 parallel 会让 90° 装置录的会话看起来可用 —— 用错几何解算得到
    #: 系统性错的深度，而且全程不报错。空串在界面显示为「未记录」。
    stereo_mode: str = "parallel"
    error: str = ""
    #: 稳定身份（`_session_index.sid_of`）。搬目录不变，勾选/分组都靠它。
    sid: str = ""
    #: 相对数据根的路径（`parallel/20261104-140552`），用于去重与显示。
    rel_path: str = ""
    #: 逻辑分组路径（`实验1/梨子`），由 `_session_index.group_of` 唯一产出。
    group: str = ""
    #: `session.json` 的 label / notes —— 带上它们，界面就不必每次重绘都
    #: 重新读一遍所有 session.json（原来每刷一次列表读 44 次文件）。
    label: str = ""
    notes: str = ""
    #: 全部问题（不只是坏会话的）：帧数差、未同步、无 seeds 之类也要能显示。
    problems: tuple[str, ...] = ()
    #: `session.json` 的 `source`（2026-11-17 新增）。``"realtime"`` = 实时双目
    #: 退出时自动落的会话：有 derived/ 三样产物，但**没有原始左右录像**，
    #: 所以不能重跑离线三维。老会话没有这个键 → 空串（= 普通录制会话）。
    #: 放在**最后一个字段**：位置参数是老调用方的顺序，插在中间会让它们错位。
    source: str = ""

    @property
    def key(self) -> str:
        """列表/选择用的键：**相对路径**，全根唯一。

        以前是目录名（``path.name``），而目录名不保证唯一：同一段时间戳
        可能同时存在于 ``stereo/<ts>``（旧扁平层）和 ``stereo/parallel/<ts>``
        （新分区）—— Treeview 的 iid 一撞，第二个会话直接插不进去，用户看到
        的是"少了一段"，没有任何报错。
        """
        return self.rel_path or self.path.name

    @property
    def symbol(self) -> str:
        return STATUS_SYMBOLS.get(self.status, "?")

    @property
    def mode_label(self) -> str:
        """给界面看的模式文案：平行 / 大角度 / **未记录**。"""
        return {"parallel": "平行", "angled": "大角度"}.get(self.stereo_mode, "未记录")


@dataclass(frozen=True)
class ExportOptions:
    categories: frozenset[str] = DEFAULT_CATEGORIES
    redact_camera: bool = True
    redact_labels: bool = True
    redact_platform: bool = True

    def normalized(self) -> "ExportOptions":
        unknown = set(self.categories) - set(CATEGORIES)
        if unknown:
            raise ValueError(f"未知类别：{sorted(unknown)}")
        return replace(self, categories=frozenset(self.categories))


@dataclass
class ExportResult:
    output: Path
    n_sessions: int
    n_files: int
    bytes_written: int
    sessions: list[str] = field(default_factory=list)


def _read_session(path: Path) -> dict:
    try:
        data = json.loads((path / "session.json").read_text(encoding="utf-8"))
        # [] / "oops" / 42 都是合法 JSON，但不是会话对象；不能让 .get 崩掉扫描
        return data if isinstance(data, dict) else {}
    except Exception:                                       # noqa: BLE001
        return {}


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def scan_items(root: str | Path, *, recursive: bool = True,
               collections: dict | None = None,
               progress: Callable[[int, int, str], None] | None = None) -> ItemList:
    """按会话时间倒序返回可导出的会话（坏会话也列出，但由 UI 提示）。

    枚举**只此一份**（``sphere_tracker.discovery.discover_session_dirs``；
    ``apps/_session_index`` 只是转发，``apps/_stereo_layout.candidate_session_dirs``
    只是多根包装）。以前数据管理、会话列表、``check_dataset``、
    ``export_ball_dataset`` 各写一套扫描口径，结果就有两套口径**现状就漏了
    15 个 parallel 会话**。

    ``recursive=True``（默认）递归发现：用户自己建 ``data/stereo/实验1/<时间戳>``
    也能看见，``parallel/<时间戳>`` / ``angled/<时间戳>`` 更不用说。
    ``recursive=False`` 是**旧的两层口径**（根的直接子目录 + 分区下一层），
    给"只想要那两层"的调用方留的开关，不是"行为一字不改"—— 两者都**只返回
    命中判据的目录**（宽判据：有 ``session.json`` 或视频/derived/标定之类），
    以前那个不过滤、连空目录都返回的实现已经删了，因为调用方各自再过滤一遍
    就是"漏一个就看不见一段会话"的来源。

    ``collections`` 传入虚拟集合配置时，每项带上 ``group``（逻辑分组路径）；
    不传就全按磁盘层级推断。分组定义只有 ``_session_index.group_of`` 一个产地。
    """
    from apps._session_index import (
        collection_map, discover_session_dirs, group_of, rel_key, sid_of)

    root = Path(root)
    if not root.is_dir():
        return ItemList()
    dirs = discover_session_dirs(root, recursive=recursive)
    mapping = collection_map(collections) if collections else {}

    # sid 撞车：按**相对路径**排序后先到先得，其余加 ~2/~3。
    # 必须是确定性的 —— 否则同一份数据扫两次，第二个会话的 sid 可能不一样，
    # 集合里存的分组就会突然指到别人身上。
    ordered = sorted(dirs, key=lambda p: rel_key(root, p))
    by_rel: dict[str, str] = {}        # rel_path -> 最终 sid（供下面回查）
    taken: set[str] = set()            # 已被占用的 sid（撞车判据必须是它，不是 by_rel）
    warnings: list[str] = []
    for p in ordered:
        rel = rel_key(root, p)
        base = sid_of(p)
        sid, n = base, 2
        while sid in taken:
            sid = f"{base}~{n}"
            n += 1
        if sid != base:
            warnings.append(f"会话 id 撞车：{base}（{rel}）"
                            f"已改用 {sid}；建议改 session.json 的 id")
        taken.add(sid)
        by_rel[rel] = sid

    def key(p: Path):
        m = re.match(r"(\d{8}-\d{6})", p.name)
        try:
            return (m.group(1) if m else "", p.stat().st_mtime)
        except OSError:
            return (m.group(1) if m else "", 0.0)

    dirs = sorted(dirs, key=key, reverse=True)
    items: list[SessionItem] = []
    total = len(dirs)
    for i, p in enumerate(dirs):
        rel = rel_key(root, p)
        sid = by_rel.get(rel) or sid_of(p)
        # 状态判定只保留一份：和工作台共用 summarize_session / STATUS_SYMBOLS
        summary = summarize_session(p)
        meta = _read_session(p)
        error = "；".join(summary.problems) if summary.status == "bad" else ""
        # 模式**只取 session.json 的记录**（含 rig 兜底），不默认 parallel。
        from apps._stereo_layout import session_mode_of

        items.append(SessionItem(
            path=p, id=summary.id or p.name,
            created=summary.created,
            frames_left=summary.frames_left,
            frames_right=summary.frames_right,
            n_matched=summary.n_matched, status=summary.status,
            size_bytes=_dir_size(p), stereo_mode=session_mode_of(p) or "",
            error=error, sid=sid, rel_path=rel,
            group=group_of(sid, mapping, rel),
            label=_as_text(meta.get("label")), notes=_as_text(meta.get("notes")),
            # 来源标记直接读 session.json（`stereo/sessions.py` 的 Session 没有
            # 这个字段、load_session 会静默丢掉未知键，所以不能指望它）。
            source=_as_text(meta.get("source")),
            problems=tuple(summary.problems)))
        if progress:
            progress(i + 1, total, rel)
    return ItemList(items, warnings)


def _as_text(value: object) -> str:
    """session.json 里的 label / notes 可能是数字/None/嵌套对象 —— 一律当文本。"""
    return value if isinstance(value, str) else ("" if value is None else str(value))


def redact_session(data: dict, options: ExportOptions) -> dict:
    """只去标识，不改技术字段；返回深拷贝。"""
    out = json.loads(json.dumps(data, ensure_ascii=False))
    if options.redact_labels:
        for k in REDACT_KEYS_TOP:
            out.pop(k, None)
    if options.redact_camera:
        for side in ("left", "right"):
            side_data = out.get(side)
            if isinstance(side_data, dict):
                for k in REDACT_KEYS_CAMERA:
                    side_data.pop(k, None)
    if options.redact_platform:
        env = out.get("environment")
        if isinstance(env, dict):
            env.pop("platform", None)
    removed = []
    if options.redact_camera:
        removed.append("相机 identity/name/port")
    if options.redact_labels:
        removed.append("会话 label/notes")
    if options.redact_platform:
        removed.append("OS platform")
    out["redacted"] = {
        "camera_identity": bool(options.redact_camera),
        "labels_notes": bool(options.redact_labels),
        "platform": bool(options.redact_platform),
        "note": ("已移除：" + "、".join(removed)) if removed else "未移除标识字段",
    }
    return out


def _cat_files(session: Path, category: str) -> Iterable[tuple[Path, str]]:
    """返回该类别下 (源文件, 包内相对路径) 列表。

    除了标准文件，也要带上 `_first<N>` 的部分结果和 `*_brightness.npy`，
    否则“只跑过部分帧”的会话在导出包里会看起来像空会话。
    """
    derived = session / "derived"
    if category == "derived":
        names = ("seeds.json", "traj3d.json", "visualization.json",
                 "traj3d.csv", "traj3d_smoothed.csv",
                 "left_brightness.npy", "right_brightness.npy")
        for name in names:
            p = derived / name
            if p.is_file():
                yield p, f"derived/{name}"
        for p in sorted(derived.glob("traj3d_first*.csv")):
            if p.is_file():
                yield p, f"derived/{p.name}"
        for p in sorted(derived.glob("traj3d_smoothed_first*.csv")):
            if p.is_file():
                yield p, f"derived/{p.name}"
        for p in sorted(derived.glob("traj3d_first*.json")):
            if p.is_file():
                yield p, f"derived/{p.name}"
    elif category == "plots":
        names = ("traj3d.png", "sync_brightness.png")
        for name in names:
            p = derived / name
            if p.is_file():
                yield p, f"derived/plots/{name}"
        for p in sorted(derived.glob("traj3d_first*.png")):
            if p.is_file():
                yield p, f"derived/plots/{p.name}"
    elif category == "annotated":
        # 不能简单改成只认 `annotated.mp4`：合法的分段产物
        # `annotated_first20.mp4` / `annotated_first120.mp4` 同样是交付内容，
        # 写死就把它们丢了。反过来原先用 `annotated*.mp4` 又会把正在录制的
        # 半成品 `annotated.tmp.mp4` 收进交付包。判据：stem 按 `.` 切开，
        # 任一段（小写后）落在半成品关键字就跳过；隐藏文件也跳过。
        partial = {"tmp", "part", "partial", "incomplete"}
        for p in sorted(derived.glob("annotated*.mp4")):
            if p.name.startswith("."):
                continue
            if any(seg.lower() in partial for seg in p.stem.split(".")):
                continue
            if p.is_file():
                yield p, f"videos/{p.name}"
    elif category == "raw":
        for name in ALL_RAW_VIDEO_NAMES:
            p = session / name
            if p.is_file():
                yield p, f"videos/raw/{name}"
    elif category == "calibration":
        calib = session / "calib"
        if calib.is_dir():
            for p in sorted(calib.iterdir()):
                if p.is_file():
                    yield p, f"calibration/{p.name}"


def estimate_bytes(items: Iterable[SessionItem], options: ExportOptions) -> int:
    options = options.normalized()
    total = 0
    for it in items:
        if "metadata" in options.categories:
            total += (it.path / "session.json").stat().st_size if (it.path / "session.json").exists() else 0
            for name in ("left_times.csv", "right_times.csv"):
                p = it.path / name
                if p.is_file():
                    total += p.stat().st_size
        for cat in options.categories:
            if cat in {"metadata", "annotated", "raw"}:
                continue
            for p, _arc in _cat_files(it.path, cat):
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
        for cat in ("annotated", "raw"):
            if cat in options.categories:
                for p, _arc in _cat_files(it.path, cat):
                    try:
                        total += p.stat().st_size
                    except OSError:
                        pass
    return total


def _manifest(items: list[SessionItem], options: ExportOptions,
              files: list[dict], root: str,
              folders: dict[str, str] | None = None) -> dict:
    folders = folders or {}
    return {
        "schema": "sphere-stereo-export",
        "version": 1,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "root": root,
        "n_sessions": len(items),
        "categories": sorted(options.categories),
        "redaction": {
            "camera_identity": bool(options.redact_camera),
            "labels_notes": bool(options.redact_labels),
            "platform": bool(options.redact_platform),
        },
        "sessions": [
            {"id": s.id, "sid": s.sid, "folder": folders.get(s.key, s.key),
             "group": s.group or None,
             "created": s.created,
             "frames_left": s.frames_left, "frames_right": s.frames_right,
             # 没记录就写 null。**不写 "parallel"** —— 收包的人会把它当真值，
            # 拿平行几何去解 90° 录的数据，深度系统性偏掉还不报错。
             "stereo_mode": s.stereo_mode or None,
             "mode_recorded": bool(s.stereo_mode),
             "n_matched": s.n_matched}
            for s in items
        ],
        "files": files,
        "note": "仅包含所选类别；原始视频/标注视频是否包含由导出选项决定。",
    }


def _readme(items: list[SessionItem], options: ExportOptions) -> str:
    cats = "\n".join(f"- `{k}`：{CATEGORIES[k]}" for k in sorted(options.categories))
    meta = ("脱敏 session.json" if (options.redact_camera or options.redact_labels
                                     or options.redact_platform) else "session.json")
    modes = sorted({s.stereo_mode for s in items if s.stereo_mode})
    n_unrecorded = sum(1 for s in items if not s.stereo_mode)
    mode_line = ", ".join(modes) or "-"
    mode_note = ""
    if n_unrecorded:
        # 前面空一行：Markdown 里紧贴上一行的 `>` 会被当成续行，看着像排版坏了。
        mode_note = (f"\n\n> ⚠️ 其中 {n_unrecorded} 段会话的 `stereo_mode` **没有记录**"
                     f"（manifest 里是 `null`）。这些是 2026-11-04 分区改造之前录的，"
                     f"导出时**不替它猜**；要用它们解算请先确认当时的装置几何。")
    return f"""# 球形双目轨迹数据包

导出时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
会话数：{len(items)}
模式：{mode_line}（parallel=平行，angled=大角度）{mode_note}

## 目录

```text
sessions/<会话ID>/
  metadata/      {meta} + 左右时间戳
  derived/       标准 JSON / CSV
  derived/plots/ 轨迹图 / 同步图
  calibration/   标定 JSON / NPZ（若勾选）
  videos/        标注视频（若勾选）
  videos/raw/    原始双目视频（若勾选）
manifest.json    会话清单、类别、文件大小
```

## 本次包含的类别

{cats}

## 脱敏说明

- 相机 `identity` / `name` / `port`：{'已移除' if options.redact_camera else '保留'}
- 会话 `label` / `notes`：{'已移除' if options.redact_labels else '保留'}
- `environment.platform`：{'已移除' if options.redact_platform else '保留'}
- 时间、帧数、标定参数、轨迹数值保留，用于复现分析。

> **视频画面没有做自动打码。** 勾选 `annotated` 或 `raw` 时，
> 画面里的房间、人物、屏幕内容会一起交付；请自行确认授权。

{VARIABLE_REFERENCE}
"""


#: 数据变量说明（简明）。`export_zip` 的 README 与「处理前+处理后」导出的
#: `数据说明.md` **共用这一份**，不各写各的——口径改了只改这一处。
VARIABLE_REFERENCE = """## 数据变量说明（简明）

### 坐标与单位
- 三维点坐标单位**米**，坐标系**左相机系**：X 右、Y 下、Z 前（未标定时）；
- 时间单位**秒**，`t=0` 为该段起始；速度 `m/s`，加速度 `m/s²`；
- 像素坐标单位为**像素**，图像尺寸以各自文件头/标定为准。

### `traj3d.csv` / `*_raw.csv`（三维轨迹，核心）
| 列 | 含义 |
| --- | --- |
| `frame` | 帧号（左路） |
| `t` | 时间（秒） |
| `matched` | 1=该帧左右配对成功，0=失败（0 的行 X/Y/Z 为空） |
| `X, Y, Z` | 球心三维坐标（米，左相机系） |
| `u_left, v_left, u_right, v_right` | 左右图像的像素坐标 |
| `epipolar_px` | 极线误差（像素），越小越好 |
| `residual_px` | 交会残差（像素） |
| `confident` | 该帧是否高置信 |
| `method_left, method_right` | 左右各自用的检测/跟踪方法 |

### `motion.csv` / `*_motion.csv`（速度/加速度，可选产物）
| 列 | 含义 |
| --- | --- |
| `x, y, z` | 平滑后的位置（米） |
| `vx, vy, vz, speed` | 速度分量与模（m/s） |
| `ax, ay, az, accel` | 加速度分量与模（m/s²），**需先平滑位置再求导，原始二阶差分噪声很大** |
| `segment` | 段号；数据有丢帧，**跨段不能连续差分/滤波，滤波器按段重置** |
| `smoothed` | true=该点参与了平滑 |

### `traj3d.json` / `visualization.json`
- `trajectory[]`：`{frame, t, xyz, confident}`，`xyz` 为 `[X, Y, Z]`（米）；
- `visualization.json` 另有 `style.axes` 说明轴向、`bounds` 说明范围。

### `metadata/session.json`
- `stereo_mode`：`parallel`（平行）/ `angled`（大角度）；
- `left/right.frame_count`、`fps`、`sync`（同步方法/偏移/置信度）；
- `rig`（标定基线、内外参，若存在）。

> 详细口径见仓库 `docs/数据标准.md` 与 `docs/运动数据交付说明.md`。
"""


def normalize_output(output: str | Path, *, directory: bool = False) -> Path:
    """把用户填的输出路径收敛成**唯一的那个路径**。两种模式语义不同：

    - ``directory=False``（默认，历史行为**一字不改**）：输出是一个 **zip 文件**，
      所以统一补 ``.zip``；空串 / 结尾带分隔符 / 指向已存在目录的**一律拒绝**。
    - ``directory=True``（P2 自包含交付包的目录形态）：输出是一个 **目录**。
      去掉结尾的 ``/`` 或反斜杠；空串、``.``、``..`` 拒绝；**后缀是 ``.zip`` 直接
      报错** —— 不悄悄改名、也不自动去后缀。选了目录输出却填 zip 名字，几乎
      一定点错了模式，替它改名只会把错误藏起来；已存在且是目录**允许**
      （要不要覆盖由 ``export_package`` 决定），已存在且是文件拒绝。

    两种模式都保证同一件事：**GUI 的 overwrite 检查必须和导出用同一条路径** ——
    调用方先拿本函数的返回值判 ``exists()``，导出层拿到的就是同一个路径，
    不存在"检查 a、写 b"。
    """
    raw = str(output).strip()
    if not raw:
        raise ValueError("输出路径为空")
    if directory:
        trimmed = raw.rstrip("/\\").strip()
        if not trimmed:
            raise ValueError(f"输出路径为空：{raw}")
        out = Path(trimmed)
        if out.name in ("", ".", ".."):
            raise ValueError(f"输出路径不是目录名：{raw}")
        if out.suffix.lower() == ".zip":
            raise ValueError(
                f"选了目录输出，却给了 .zip 路径：{raw}（目录包不要 .zip 后缀）")
        if out.exists() and not out.is_dir():
            raise ValueError(f"输出路径已存在且不是目录：{out}")
        return out
    if raw.endswith(("/", "\\")):
        raise ValueError(f"输出路径是目录，不是 zip：{raw}")
    out = Path(raw)
    if out.name in ("", ".", ".."):
        raise ValueError(f"输出路径不是文件名：{raw}")
    if out.exists() and out.is_dir():
        raise ValueError(f"输出路径是目录，不是 zip：{out}")
    return out.with_suffix(".zip") if out.suffix.lower() != ".zip" else out


def _plan_files(items: Iterable[SessionItem], options: ExportOptions
                ) -> list[tuple[SessionItem, list[tuple[Path, str]]]]:
    """按会话列出 ``(源文件, 包内相对路径)``。

    **zip 包与目录包共用这一份**。各写一份的后果不是"多几行代码"，而是两边
    交出的内容会悄悄分叉（某天只给 zip 补了 ``derived/plots``，目录包就少一类
    文件，还没有任何报错）。``options`` 必须已经 ``normalized()``。
    """
    planned: list[tuple[SessionItem, list[tuple[Path, str]]]] = []
    for it in items:
        rows: list[tuple[Path, str]] = []
        if "metadata" in options.categories:
            rows.append((it.path / "session.json", "metadata/session.json"))
            for name in ("left_times.csv", "right_times.csv"):
                p = it.path / name
                if p.is_file():
                    rows.append((p, f"metadata/{name}"))
        for cat in options.categories:
            if cat in {"metadata", "annotated", "raw"}:
                continue
            for p, rel in _cat_files(it.path, cat):
                rows.append((p, rel))
        for cat in ("annotated", "raw"):
            if cat in options.categories:
                for p, rel in _cat_files(it.path, cat):
                    rows.append((p, rel))
        planned.append((it, rows))
    return planned


def _unique_folders(items: Iterable[SessionItem]) -> dict[str, str]:
    """会话 → 包内目录名：清洗非法字符 + 去重（zip 包与目录包共用）。

    目录名可能相同（两个同名时间戳）或**清洗后**相同（中文被替换成 ``_``），
    所以不能只依赖 ``id``：撞了就让后一个覆盖前一个，接收方少一段且没人报错。
    """
    folders: dict[str, str] = {}
    used: set[str] = set()
    for it in items:
        base = re.sub(r"[^0-9A-Za-z._\-]+", "_", it.path.name) or "session"
        name = base
        n = 2
        while name in used:
            name = f"{base}_{n}"
            n += 1
        used.add(name)
        folders[it.key] = name
    return folders


def export_zip(items: list[SessionItem], options: ExportOptions, output: str | Path,
               progress: Callable[[int, int, str], None] | None = None) -> ExportResult:
    """把选中的会话写成一个带顶层目录的 zip。"""
    options = options.normalized()
    if not items:
        raise ValueError("没有选择任何会话")
    out = normalize_output(output)
    if out.exists():
        raise FileExistsError(f"输出已存在：{out}")
    root = re.sub(r"[^0-9A-Za-z._\-\u4e00-\u9fff]+", "_", out.stem) or "export"

    if "metadata" in options.categories:
        bad = [it.key for it in items if not _read_session(it.path)]
        if bad:
            raise ValueError(
                "这些会话缺/坏 session.json：" + "、".join(bad) +
                "；取消 metadata 类别或先修好会话")

    # 文件规划与目录名去重**与目录包共用**（_plan_files / _unique_folders）
    planned = _plan_files(items, options)
    folders = _unique_folders(items)

    file_count = sum(len(rows) for _it, rows in planned) + 2
    written = 0
    total = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    if tmp.exists():
        tmp.unlink()
    manifest_files: list[dict] = []

    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6) as zf:
            # README
            readme = _readme(items, options).encode("utf-8")
            zf.writestr(f"{root}/README.md", readme)
            total += len(readme)
            written += 1
            if progress:
                progress(written, file_count, "README.md")

            for it, rows in planned:
                sid = folders[it.key]
                prefix = f"{root}/sessions/{sid}"
                for src, rel in rows:
                    if src == it.path / "session.json":
                        if "metadata" not in options.categories:
                            continue
                        data = _read_session(it.path)
                        if not data:
                            raise ValueError(f"{it.key} 的 session.json 不可读")
                        payload = json.dumps(redact_session(data, options),
                                             ensure_ascii=False, indent=2).encode("utf-8")
                        arc = f"{prefix}/metadata/session.json"
                        zf.writestr(arc, payload)
                        total += len(payload)
                        written += 1
                        manifest_files.append({"path": arc, "bytes": len(payload)})
                        if progress:
                            progress(written, file_count, arc)
                        continue
                    if not src.is_file():
                        continue
                    arc = f"{prefix}/{rel}"
                    # 实测（HFYU 无损视频）：DEFLATE 压缩率 0.987–1.0003，
                    # 单个 800 MB 文件要 ~20 秒却几乎不缩小；而 progress 只在
                    # 整个文件写完后才回调，于是出现两次 20 秒级进度条冻结。
                    # 视频改用 ZIP_STORED 纯存储，其它文件仍走 DEFLATED。
                    video = src.suffix.lower() in {".mkv", ".mp4", ".avi", ".mov"}
                    zf.write(src, arc,
                             compress_type=zipfile.ZIP_STORED if video
                             else zipfile.ZIP_DEFLATED)
                    size = src.stat().st_size
                    total += size
                    written += 1
                    manifest_files.append({"path": arc, "bytes": size})
                    if progress:
                        progress(written, file_count, arc)

            manifest = _manifest(items, options, manifest_files, root, folders)
            zf.writestr(f"{root}/manifest.json",
                        json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
            written += 1
            if progress:
                progress(written, file_count, "manifest.json")
        os.replace(tmp, out)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise

    archive_size = out.stat().st_size if out.exists() else total
    return ExportResult(output=out, n_sessions=len(items), n_files=file_count,
                        bytes_written=archive_size, sessions=[s.id for s in items])


# --------------------------------------------------------------------------- #
# "处理前 + 处理后" 一次导出（2026-11-03）
# --------------------------------------------------------------------------- #
def export_motion_pair(session: str | Path, out_dir: str | Path) -> tuple[Path, Path] | None:
    """把一个会话的**原始轨迹**与**平滑/求导后轨迹**一起导出，返回两个文件路径。

    为什么要成对导出：用户要在同一次操作里拿到"管线直接产出的"与"给滤波/分析用的"，
    否则两个文件分别导出、命名还各写各的，很容易把平滑过的当成原始数据（或者反过来）。
    命名固定为 ``<会话>_raw.csv`` / ``<会话>_motion.csv``，一眼能分。

    `*_raw.csv` 是**逐字节复制** `derived/traj3d.csv`（不做任何重排/改列）：
    原始数据一旦被"顺手规整"，就没法和旧结果对账了。

    每次导出还会在**输出目录**写一份 ``数据说明.md``（`VARIABLE_REFERENCE`，与
    `export_zip` 的 README 共用同一份口径），让接收方一眼看懂列含义/单位/坐标系。

    会话没有 `derived/traj3d.csv` 时返回 ``None``（调用方据此告诉用户"这段还没跑过"）。
    """
    import shutil

    from apps.tools.export_motion import read_csv_traj, write_motion_csv
    from stereo.motion import kinematics

    sess = Path(session)
    src = sess / "derived" / "traj3d.csv"
    if not src.exists():
        return None
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = out / f"{sess.name}_raw.csv"
    motion = out / f"{sess.name}_motion.csv"
    shutil.copy2(src, raw)

    frame, t, P = read_csv_traj(src)          # 注意返回顺序是 (frame, t, P)
    if t.size == 0:
        return None
    kin = kinematics(t, P)
    write_motion_csv(motion, frame, kin)

    # 附一份说明（幂等覆盖：多次导出同一目录时保持最新口径）
    note = out / "数据说明.md"
    note.write_text(
        f"# 双目运动数据说明\n\n导出时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"会话：{sess.name}\n\n"
        f"本目录含该会话的两份轨迹：\n\n"
        f"- `{raw.name}`：原始三维轨迹（逐字节复制 traj3d.csv）\n"
        f"- `{motion.name}`：平滑后的位置 + 速度 + 加速度\n\n"
        f"{VARIABLE_REFERENCE}",
        encoding="utf-8")
    return raw, motion
