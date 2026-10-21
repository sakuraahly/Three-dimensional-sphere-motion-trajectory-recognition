"""安全清理会话：先预览，再移到 `_trash/`；恢复/彻底删除分开操作。

设计原则
--------
- **不直接 `rmtree` 用户数据**：第一阶段只做“移到回收站”；
- 回收站里再恢复或确认后彻底删除；
- 只允许操作 sessions 根目录的**直接子目录**（或其 `derived/`），
  拒绝符号链接、拒绝 `..` 逃逸；
- 所有 UI 都必须先看到“将移动哪些路径、多少字节”的预览。

这不是双目库，属于管理工具的纯逻辑层，所以放 `apps/_cleanup.py`。
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

TRASH_NAME = "_trash"
MODES = {"session": "整个会话（含原始视频）", "derived": "只清派生结果（derived/）"}


@dataclass(frozen=True)
class CleanupItem:
    session: Path
    source: Path
    bytes: int
    files: int


@dataclass
class CleanupPlan:
    root: Path
    mode: str
    items: list[CleanupItem] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(i.bytes for i in self.items)

    @property
    def total_files(self) -> int:
        return sum(i.files for i in self.items)

    def summary(self) -> str:
        return (f"{len(self.items)} 个会话 / {self.total_files} 个文件 / "
                f"{_fmt_bytes(self.total_bytes)}")


def _fmt_bytes(n: int) -> str:
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.2f} GiB"
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.1f} MiB"
    if n >= 1024:
        return f"{n / 1024:.1f} KiB"
    return f"{n} B"


def _tree_size(path: Path) -> tuple[int, int]:
    total = 0
    files = 0
    for p in path.rglob("*"):
        if p.is_file() and not p.is_symlink():
            try:
                total += p.stat().st_size
                files += 1
            except OSError:
                pass
    return total, files


def trash_root(sessions_root: str | Path) -> Path:
    root = Path(sessions_root).resolve()
    return root.parent / TRASH_NAME


def _resolve_target(root: Path, target: str | Path) -> Path:
    raw = Path(target)
    if not raw.is_absolute():
        raw = root / raw
    if raw.is_symlink():
        raise ValueError(f"拒绝清理符号链接：{raw}")
    p = raw.resolve()
    if p.parent != root:
        raise ValueError(f"只能清理 sessions 根目录的直接子目录：{p}")
    if p == root or p.name == TRASH_NAME:
        raise ValueError(f"拒绝清理根目录/回收站：{p}")
    if p.is_symlink():
        raise ValueError(f"拒绝清理符号链接：{p}")
    return p


def plan_cleanup(sessions_root: str | Path, targets: list[str | Path],
                 mode: str = "session", keep_latest: int = 0) -> CleanupPlan:
    """生成清理计划；不移动任何文件。"""
    root = Path(sessions_root).resolve()
    if not root.is_dir():
        raise ValueError(f"sessions 根目录不存在：{root}")
    if mode not in MODES:
        raise ValueError(f"mode 必须是 session/derived，收到 {mode!r}")
    plan = CleanupPlan(root=root, mode=mode)
    unique: list[Path] = []
    seen: set[Path] = set()
    for t in targets:
        p = _resolve_target(root, t)
        if p not in seen:
            seen.add(p)
            unique.append(p)
    # 保留最新 N 个（按目录名/时间倒序）
    if keep_latest > 0:
        newest = sorted(unique, key=lambda p: p.name, reverse=True)[:keep_latest]
        for p in newest:
            plan.skipped.append(f"{p.name}（保留最新）")
        unique = [p for p in unique if p not in newest]
    for p in unique:
        if not p.is_dir():
            plan.skipped.append(f"{p.name}（目录不存在）")
            continue
        source = p if mode == "session" else p / "derived"
        if not source.exists():
            plan.skipped.append(f"{p.name}（没有 {'derived/' if mode == 'derived' else '会话目录'}）")
            continue
        if source.is_symlink():
            plan.skipped.append(f"{p.name}（符号链接）")
            continue
        size, files = _tree_size(source)
        plan.items.append(CleanupItem(session=p, source=source, bytes=size, files=files))
    return plan


def move_to_trash(sessions_root: str | Path, plan: CleanupPlan,
                  label: str = "cleanup") -> Path:
    """把计划中的目录移动到 `_trash/<时间>_<label>/`，返回 batch 路径。"""
    root = Path(sessions_root).resolve()
    if plan.root != root:
        raise ValueError("清理计划与 sessions 根目录不一致")
    if not plan.items:
        raise ValueError("没有可移动的会话")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_label = "".join(c for c in label if c.isalnum() or c in "-_")[:30] or "cleanup"
    batch = trash_root(root) / f"{stamp}_{safe_label}"
    n = 2
    while batch.exists():
        batch = trash_root(root) / f"{stamp}_{safe_label}_{n}"
        n += 1
    batch.mkdir(parents=True)
    moved: list[dict] = []
    try:
        for item in plan.items:
            rel = item.source.relative_to(root)
            dest = batch / "sessions" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(item.source), str(dest))
            moved.append({"path": str(rel).replace("\\", "/"),
                          "bytes": item.bytes, "files": item.files})
    finally:
        manifest = {
            "schema": "sphere-stereo-trash",
            "version": 1,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "root": str(root),
            "mode": plan.mode,
            "items": moved,
        }
        try:
            (batch / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass
    return batch


@dataclass(frozen=True)
class TrashBatch:
    path: Path
    created: str
    mode: str
    items: list[dict]
    bytes: int

    @property
    def label(self) -> str:
        return f"{self.path.name}  {self.created}  {self.mode}  {_fmt_bytes(self.bytes)}"


def list_trash(sessions_root: str | Path) -> list[TrashBatch]:
    root = Path(sessions_root).resolve()
    base = trash_root(root)
    if not base.is_dir():
        return []
    out: list[TrashBatch] = []
    for d in sorted(base.iterdir(), key=lambda p: p.name, reverse=True):
        if not d.is_dir():
            continue
        try:
            man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        except Exception:                                   # noqa: BLE001
            man = {}
        items = man.get("items") or []
        total = sum(int(i.get("bytes") or 0) for i in items if isinstance(i, dict))
        out.append(TrashBatch(path=d, created=str(man.get("created") or ""),
                              mode=str(man.get("mode") or "?"),
                              items=items, bytes=total))
    return out


def _safe_batch(sessions_root: str | Path, batch: str | Path) -> Path:
    root = Path(sessions_root).resolve()
    b = Path(batch)
    if not b.is_absolute():
        b = trash_root(root) / b
    b = b.resolve()
    if not b.is_relative_to(trash_root(root)) or b == trash_root(root):
        raise ValueError(f"不是本根目录的回收站批次：{b}")
    return b


def restore_batch(sessions_root: str | Path, batch: str | Path) -> int:
    root = Path(sessions_root).resolve()
    b = _safe_batch(root, batch)
    man_path = b / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    restored = 0
    for item in man.get("items") or []:
        rel = str(item.get("path") or "")
        if not rel:
            continue
        src = (b / "sessions" / rel).resolve()
        dst = (root / rel).resolve()
        if not src.is_relative_to(b.resolve()) or not dst.is_relative_to(root):
            raise ValueError(f"回收站 manifest 路径越界：{rel}")
        if not src.exists():
            continue
        if dst.exists():
            raise FileExistsError(f"恢复目标已存在，先处理冲突：{dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        restored += 1
    # 全部恢复后把空批次也清掉，避免回收站里留下一堆空目录
    if restored and restored == len(man.get("items") or []):
        try:
            shutil.rmtree(b)
        except OSError:
            pass
    return restored


def purge_batch(sessions_root: str | Path, batch: str | Path) -> None:
    root = Path(sessions_root).resolve()
    b = _safe_batch(root, batch)
    shutil.rmtree(b)
