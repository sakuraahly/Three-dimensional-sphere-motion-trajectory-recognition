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
    error: str = ""

    @property
    def key(self) -> str:
        """目录名唯一；session.json 的 id 不保证唯一，UI/选择必须用它。"""
        return self.path.name

    @property
    def symbol(self) -> str:
        return STATUS_SYMBOLS.get(self.status, "?")

    @property
    def label(self) -> str:
        return f"{self.symbol} {self.id}  {self.created}  {self.frames_left}/{self.frames_right}帧"


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


def scan_items(root: str | Path) -> list[SessionItem]:
    """按会话时间倒序返回可导出的会话（坏会话也列出，但由 UI 提示）。"""
    root = Path(root)
    if not root.is_dir():
        return []
    dirs = [p for p in root.iterdir() if p.is_dir()]

    def key(p: Path):
        m = re.match(r"(\d{8}-\d{6})", p.name)
        return (m.group(1) if m else "", p.stat().st_mtime)

    dirs.sort(key=key, reverse=True)
    items: list[SessionItem] = []
    for p in dirs:
        # 状态判定只保留一份：和工作台共用 summarize_session / STATUS_SYMBOLS
        summary = summarize_session(p)
        error = "；".join(summary.problems) if summary.status == "bad" else ""
        items.append(SessionItem(
            path=p, id=summary.id or p.name,
            created=summary.created,
            frames_left=summary.frames_left,
            frames_right=summary.frames_right,
            n_matched=summary.n_matched, status=summary.status,
            size_bytes=_dir_size(p), error=error))
    return items


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
        for p in sorted(derived.glob("annotated*.mp4")):
            if p.is_file():
                yield p, f"videos/{p.name}"
    elif category == "raw":
        for name in ("left.mkv", "right.mkv", "left.mp4", "right.mp4"):
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
            {"id": s.id, "folder": folders.get(s.key, s.key), "created": s.created,
             "frames_left": s.frames_left, "frames_right": s.frames_right,
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
    return f"""# 球形双目轨迹数据包

导出时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
会话数：{len(items)}

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
"""


def normalize_output(output: str | Path) -> Path:
    """统一补 .zip；GUI 的 overwrite 检查必须和 export_zip 用同一条路径。"""
    raw = str(output).strip()
    if not raw:
        raise ValueError("输出路径为空")
    if raw.endswith(("/", "\\")):
        raise ValueError(f"输出路径是目录，不是 zip：{raw}")
    out = Path(raw)
    if out.name in ("", ".", ".."):
        raise ValueError(f"输出路径不是文件名：{raw}")
    if out.exists() and out.is_dir():
        raise ValueError(f"输出路径是目录，不是 zip：{out}")
    return out.with_suffix(".zip") if out.suffix.lower() != ".zip" else out


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

    # 先按会话列出文件，便于进度与 manifest
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

    # 目录名可能相同/清洗后相同；保证 zip 内 folder 唯一，不能只依赖 id
    folders: dict[str, str] = {}
    used_folders: set[str] = set()
    for it in items:
        base = re.sub(r"[^0-9A-Za-z._\-]+", "_", it.path.name) or "session"
        name = base
        n = 2
        while name in used_folders:
            name = f"{base}_{n}"
            n += 1
        used_folders.add(name)
        folders[it.key] = name

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
                    zf.write(src, arc)
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
