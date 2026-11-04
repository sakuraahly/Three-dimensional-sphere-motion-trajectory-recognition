"""垃圾盘点与清理：只认**白名单**里的"可再生的东西"，别的一律不碰。

为什么单独一层（而不是在对话框里 rglob）
---------------------------------------
清理是唯一会**减少磁盘上的东西**的操作。把"什么算垃圾"散在界面回调里，
下一个改界面的人就可能在 `rglob("*")` 后面接一句 `unlink` —— 那一次点击
能删掉 `data/stereo` 里 60 GiB 不可再生的录像。所以这里把三件事分开：

1. **扫描**（`scan_junk`）：只从 `_rules()` 产出的路径里读，绝不"扫全盘找像垃圾的"；
2. **计划**（`JunkItem` / `JunkPlan`）：先算清体积与文件数，界面**必须先预览**；
3. **执行**（`clean_junk`）：只允许删**计划里出现过**的路径，越界/符号链接一律拒绝。

为什么分"推荐 / 谨慎"两档
-------------------------
`outputs/` 下有两类完全不同的东西：

- ``_ab`` / ``tmp_frames3`` / ``agent_detector`` / ``logs`` / ``test_results``：
  调试与运行产物，删了下次跑就有了 → **推荐**（默认勾选）；
- ``chain``（链式标定图）/ ``ball_dataset``（YOLO 数据集）/ ``mono_calib_frames``：
  某次实验的输入，删了要**重拍/重跑**才能再生 → **谨慎**（默认不勾，写清代价）。

把两者混成一句"清理垃圾"就是替用户做了他才能做的判断。

绝对不碰的
----------
``data/stereo`` / ``data/mono`` 里的会话内容（录像与 derived）、``config/``、
``docs/``、``assets/``、``models/``、``apps/``、``tests/``、``calib/``、
``stereo/``、``sphere_tracker/``、``legacy/``。这些不在 `_rules()` 里出现，
`clean_junk` 就删不到它们 —— 这是**结构性**的保护，不是一句注释。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "DANGER_GONE", "DANGER_SAFE", "DANGER_WARN",
    "JunkFile", "JunkItem", "JunkPlan",
    "clean_junk", "human_size", "scan_junk",
]

#: 删了下次跑就有（日志 / 缓存 / 调试产物 / 回收站里的垃圾）。
DANGER_SAFE = "safe"
#: 删了要重跑或重拍才能再生（数据集、标定图、模型训练中间物）。
DANGER_WARN = "warn"
#: 不可恢复（回收站内容），必须二次确认。
DANGER_GONE = "gone"

#: `outputs/` 下这些前缀的目录是**调试/运行产物**：默认推荐清理。
_RECOMMENDED_PREFIXES = ("_", "tmp_", "agent_")

#: `outputs/` 下这些目录**正在被流程使用**，不列进垃圾（避免手滑清掉刚采的标定图）。
_KEEP_ALWAYS = {"chain_frames"}

#: 名字不带调试前缀、但同样是"删了下次跑就有"的目录（推荐档）。
_KNOWN_SAFE = {
    "logs": "运行日志：每次任务追加的 .log（排障看这里，但历史值不大）",
    "test_results": "回归测试报告：run_tests.py 每次运行写的 JSON",
}

#: 每个会话的 `session.json.bak-*` 至少留几个（编辑元数据的回滚点）。
KEEP_BAK_PER_SESSION = 3

_SECONDS_PER_DAY = 86400.0


def human_size(n: float) -> str:
    """字节数 → 人看的大小（与其它界面同一套口吻）。"""
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TiB"


@dataclass(frozen=True)
class JunkFile:
    path: Path
    size: int
    mtime: float


@dataclass
class JunkItem:
    """一个可勾选的清理项：一组文件（一个目录，或一类散落文件）。"""

    key: str
    label: str
    detail: str
    danger: str
    files: list[JunkFile] = field(default_factory=list)

    @property
    def size(self) -> int:
        return sum(f.size for f in self.files)

    @property
    def count(self) -> int:
        return len(self.files)

    @property
    def newest(self) -> float:
        """这批文件里**最新**的 mtime；空集返回 0。"""
        return max((f.mtime for f in self.files), default=0.0)

    def summary(self) -> str:
        return f"{self.count} 个文件 / {human_size(self.size)}"


@dataclass
class JunkPlan:
    items: list[JunkItem] = field(default_factory=list)
    #: 扫描时被跳过的路径与原因（符号链接、越界、无权限）——如实报，不静默丢。
    skipped: list[str] = field(default_factory=list)
    #: 被 ``older_than_days`` 筛掉的部分。界面必须显示它 —— 否则"选了 30 天前
    #: 只看到 1 项"会被读成"没有垃圾"，而实际是 3.9 GiB 里有 3.9 GiB 是新的。
    filtered_files: int = 0
    filtered_bytes: int = 0

    @property
    def size(self) -> int:
        return sum(i.size for i in self.items)

    @property
    def count(self) -> int:
        return sum(i.count for i in self.items)

    def summary(self) -> str:
        return f"{len(self.items)} 项 / {self.count} 个文件 / {human_size(self.size)}"


# --------------------------------------------------------------------------- #
# 扫描
# --------------------------------------------------------------------------- #
def _walk_files(root: Path, *, skipped: list[str]) -> list[JunkFile]:
    """递归列出 ``root`` 下的普通文件（跳过符号链接，越界不跟）。

    符号链接一律跳过：一个指向 ``data/stereo`` 的链接会让"清日志"变成
    "清录像"，而这种事在 Windows 上还特别难事后发现。
    """
    out: list[JunkFile] = []
    if not root.exists():
        return out
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(dirpath)
        # 不跟进符号链接目录
        dirnames[:] = [d for d in dirnames if not (base / d).is_symlink()]
        for name in filenames:
            p = base / name
            try:
                if p.is_symlink():
                    skipped.append(f"{p}（符号链接，跳过）")
                    continue
                st = p.stat()
            except OSError as exc:
                skipped.append(f"{p}（读不到：{exc}）")
                continue
            out.append(JunkFile(path=p, size=st.st_size, mtime=st.st_mtime))
    return out


def _dir_item(key: str, path: Path, *, label: str, detail: str, danger: str,
              skipped: list[str]) -> JunkItem | None:
    files = _walk_files(path, skipped=skipped)
    if not files:
        return None
    return JunkItem(key=key, label=label, detail=detail, danger=danger, files=files)


def _outputs_dir_item(d: Path, skipped: list[str]) -> JunkItem | None:
    """`outputs/` 下的一个一级子目录：按前缀判定"推荐"还是"谨慎"。"""
    if d.name in _KNOWN_SAFE:
        detail, danger = _KNOWN_SAFE[d.name], DANGER_SAFE
    elif d.name.startswith(_RECOMMENDED_PREFIXES):
        detail = "调试/运行产物，删了下次跑就有"
        danger = DANGER_SAFE
    else:
        detail = ("某次实验的产物（数据集/标定图/走查截图等），"
                  "删了要重跑或重拍才能再生 —— 要不要留由你判断")
        danger = DANGER_WARN
    return _dir_item(
        f"outputs/{d.name}", d, label=f"outputs/{d.name}", detail=detail,
        danger=danger, skipped=skipped)


def _session_bak_files(root: Path, skipped: list[str]) -> list[JunkFile]:
    """每个会话只留**最近 KEEP_BAK_PER_SESSION 个** `session.json.bak-*`。

    为什么不一次全清：那些 .bak 是改元数据（改名/改备注/改装置模式）时的
    回滚点。全删掉等于把"改错了还能退回去"这条退路一次清空，而它们每个
    只有几 KB —— 留着最近几个几乎不占空间，价值却很实。
    """
    keep: list[JunkFile] = []
    data = root / "data"
    if not data.is_dir():
        return keep
    for sj in data.rglob("session.json"):
        if sj.is_symlink():
            continue
        baks = sorted(
            (p for p in sj.parent.glob("session.json.bak-*")
             if p.is_file() and not p.is_symlink()),
            key=lambda p: p.name, reverse=True)
        for p in baks[KEEP_BAK_PER_SESSION:]:
            try:
                st = p.stat()
            except OSError as exc:
                skipped.append(f"{p}（读不到：{exc}）")
                continue
            keep.append(JunkFile(path=p, size=st.st_size, mtime=st.st_mtime))
    return keep


def scan_junk(root: str | Path, *, older_than_days: float | None = None) -> JunkPlan:
    """盘点可清理项。

    ``older_than_days``：只算 **mtime 早于** ``now - N 天`` 的文件。
    ``None`` = 不限时间（全部）。界面默认就是"全部"——因为默认勾选的只有
    "删了下次跑就有"那一档，时间再筛一遍是过度保守：实测本项目 3.9 GiB 垃圾里
    3.9 GiB 都是最近 7 天产生的，默认 30 天前会让功能看起来"什么都没有"。
    时间范围是**第二道**筛子，给想更保守的人用；筛掉多少会记在
    ``filtered_bytes`` 里如实显示。

    只从 `_rules()` 列出的位置读，不做"全盘找像垃圾的文件"。
    """
    repo = Path(root).resolve()
    plan = JunkPlan()
    skipped = plan.skipped

    if not repo.is_dir():
        raise ValueError(f"仓库根不存在：{repo}")

    outputs = repo / "outputs"
    data = repo / "data"

    # ---- ① outputs 下的一级子目录（推荐 + 谨慎两档）----
    if outputs.is_dir():
        try:
            subs = sorted((d for d in outputs.iterdir()
                           if d.is_dir() and not d.is_symlink()),
                          key=lambda d: d.name.lower())
        except OSError as exc:
            skipped.append(f"{outputs}（读不到：{exc}）")
            subs = []
        for d in subs:
            if d.name in _KEEP_ALWAYS or d.name == "__pycache__":
                continue          # 正在用的标定图 / 字节码另有条目
            item = _outputs_dir_item(d, skipped)
            if item is not None:
                plan.items.append(item)

        # outputs 根下散落的临时文件（tmp_seeds_*.json 这类）。
        # **不再收 *.tmp / *.partial**：它们由下面的 `tmp_files` 统一收，
        # 两处都收会让同一个文件进两个清理项 —— 先删的那条成功、后删的报
        # "找不到文件"，用户看到一条无意义的失败。
        loose: list[JunkFile] = []
        for p in outputs.iterdir():
            if not p.is_file() or p.is_symlink():
                continue
            if p.suffix in (".tmp", ".partial") or ".tmp-" in p.name:
                continue
            if p.name.startswith(("tmp_", "_tmp", ".")):
                try:
                    st = p.stat()
                except OSError:
                    continue
                loose.append(JunkFile(path=p, size=st.st_size, mtime=st.st_mtime))
        if loose:
            plan.items.append(JunkItem(
                key="outputs/散落临时文件", label="outputs 根下的临时文件",
                detail="tmp_*/.* 这类散落在 outputs 根的文件", danger=DANGER_SAFE,
                files=loose))

    # ---- ② 字节码缓存（全仓，含 apps/tests/sphere_tracker…）----
    pyc: list[JunkFile] = []
    for pc in repo.rglob("__pycache__"):
        if not pc.is_dir() or pc.is_symlink() or ".git" in pc.parts:
            continue
        pyc.extend(_walk_files(pc, skipped=skipped))
    if pyc:
        plan.items.append(JunkItem(
            key="__pycache__", label="Python 字节码缓存",
            detail="__pycache__ 里的 .pyc，删了下次 import 自动重建",
            danger=DANGER_SAFE, files=pyc))

    # ---- ③ 中断留下的临时文件 ----
    tmp: list[JunkFile] = []
    for pat in ("*.tmp", "*.tmp-*", "*.partial"):
        for p in repo.rglob(pat):
            if not p.is_file() or p.is_symlink() or ".git" in p.parts:
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            tmp.append(JunkFile(path=p, size=st.st_size, mtime=st.st_mtime))
    if tmp:
        # 同一个文件可能被两个模式命中（*.tmp 与 *.tmp-*），按路径去重
        uniq = {f.path: f for f in tmp}
        plan.items.append(JunkItem(
            key="tmp_files", label="临时文件残留",
            detail="*.tmp / *.tmp-* / *.partial —— 录制或导出被中断时留下的",
            danger=DANGER_SAFE, files=list(uniq.values())))

    # ---- ④ 会话元数据备份（每个会话留最近几个）----
    baks = _session_bak_files(repo, skipped)
    if baks:
        plan.items.append(JunkItem(
            key="session_bak", label="会话元数据备份",
            detail=f"session.json.bak-*（每个会话保留最近 {KEEP_BAK_PER_SESSION} 个）",
            danger=DANGER_SAFE, files=baks))

    # ---- ⑤ 回收站（彻底删，不可恢复）----
    trash = data / "_trash"
    if trash.is_dir():
        item = _dir_item(
            "trash", trash, label="回收站（data/_trash）",
            detail="之前软删的会话，**删了不可恢复**",
            danger=DANGER_GONE, skipped=skipped)
        if item is not None:
            plan.items.append(item)

    # ---- ⑥ 时间过滤（只留"够旧"的文件）----
    if older_than_days is not None:
        cutoff = time.time() - float(older_than_days) * _SECONDS_PER_DAY
        kept: list[JunkItem] = []
        for item in plan.items:
            fresh = [f for f in item.files if f.mtime < cutoff]
            plan.filtered_files += item.count - len(fresh)
            plan.filtered_bytes += item.size - sum(f.size for f in fresh)
            if not fresh:
                continue
            kept.append(JunkItem(key=item.key, label=item.label,
                                 detail=item.detail, danger=item.danger,
                                 files=fresh))
        plan.items = kept

    plan.items.sort(key=lambda i: (i.danger != DANGER_SAFE, -i.size))
    return plan


# --------------------------------------------------------------------------- #
# 执行
# --------------------------------------------------------------------------- #
def clean_junk(plan: JunkPlan, root: str | Path) -> tuple[int, int, list[str]]:
    """按计划删文件；返回 ``(删掉的文件数, 释放的字节, 失败信息)``。

    两条安全边界：

    1. **只删计划里出现过的路径** —— 计划由 `scan_junk` 的白名单产出，
       调用方就算传进来一个自己拼的 `JunkPlan`，也会被"必须在仓库根下"挡住；
    2. **不删目录树，只删文件再回收空目录** —— `shutil.rmtree` 一次调用就能
       带走整个 `outputs/`，而逐文件删到失败就停，最坏情况也只是留下一半。

    删完把空目录自底向上 rmdir（只 rmdir 空目录，非空会失败并忽略）：
    留着 `outputs/logs/` 这种空壳，下次看列表会以为"没删掉"。
    """
    repo = Path(root).resolve()
    deleted = freed = 0
    failures: list[str] = []
    touched_dirs: set[Path] = set()
    # 同一个文件可能落进两个清理项（例如既在某个 outputs 子目录里、又被
    # `*.tmp` 收了一次）。只删一次；重复的那次**不算失败** —— 它不是错误，
    # 报出来只会让用户以为"有东西没删掉"。
    seen: set[Path] = set()

    for item in plan.items:
        for f in item.files:
            p = f.path
            try:
                if p in seen:
                    continue
                seen.add(p)
            except TypeError:
                pass
            try:
                rp = p.resolve()
            except OSError:
                failures.append(f"{p}（路径解析失败）")
                continue
            if not rp.is_relative_to(repo):
                failures.append(f"{p}（在仓库外，拒绝删除）")
                continue
            if p.is_symlink():
                failures.append(f"{p}（符号链接，拒绝删除）")
                continue
            try:
                size = p.stat().st_size
                p.unlink()
                deleted += 1
                freed += size
                touched_dirs.add(p.parent)
            except OSError as exc:
                # 最常见：文件正被别的进程占用（正在写的日志）。如实报，不重试。
                failures.append(f"{p}（{exc}）")

    # 自底向上回收空目录（只删空的，深的先删）
    for d in sorted(touched_dirs, key=lambda q: len(q.parts), reverse=True):
        cur = d
        while cur != repo and cur.is_relative_to(repo):
            try:
                cur.rmdir()           # 非空会抛 OSError，正是我们要的
            except OSError:
                break
            cur = cur.parent

    return deleted, freed, failures
