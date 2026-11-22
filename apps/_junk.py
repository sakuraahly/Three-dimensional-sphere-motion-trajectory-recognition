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
`outputs/` 下（以及 2026-11-04 之后项目根的 `logs/`）有两类完全不同的东西：

- ``_ab`` / ``tmp_frames3`` / ``agent_detector`` / ``test_results`` / 根目录的 ``logs/``：
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

from apps._common import SESSION_KEEP_BACKUPS, backup_glob

__all__ = [
    "DANGER_GONE", "DANGER_SAFE", "DANGER_WARN",
    "JunkFile", "JunkItem", "JunkPlan", "PROTECTED_OUTPUTS",
    "clean_junk", "human_size", "is_protected_output", "looks_like_data",
    "looks_like_research_artifact", "scan_junk",
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
#:
#: ⚠️ `logs` 2026-11-04 起已搬到**项目根**（`<repo>/logs`），不再在 `outputs/` 下，
#: 所以从这张表里移出去了 —— 但它仍是推荐档，由 `scan_junk` 单独收（见 ⓪ 段）。
#: `outputs/logs` 若还存在，只当"旧位置迁移残留"列出，且**不报 missing**：
#: 新装/已迁移的仓库里它本来就不该存在。
_KNOWN_SAFE = {
    # 这条要写清代价：`timings.json` 是 `fast` 档"哪些测试太慢、该排除"的依据，
    # 删了下次 `fast` 会退化成跑全套（约慢 7 分钟）。不是不能清，是要知道清了会怎样。
    "test_results": "回归测试报告（run_tests.py 每次写的 JSON）。"
                    "注意 timings.json 是 fast 档的快/慢依据，删了下次 fast 会退化成全套",
}

#: 用户裁决（2026-11-10）**原地保留、不可再生**的 22 份交付物。
#:
#: 与 `docs/outputs保留清单.md` 一一对应。它们不迁 `docs/`：文档/报告里的引用
#: 按 `outputs/` 路径写，迁移会断链；而本模块的清理白名单也按 `outputs/` 路径
#: 点名，移走反而失去保护。这里**不另起一套判定**：`looks_like_research_artifact`
#: 先认它们，`clean_junk` 再兜一道"即使被勾选也拒绝删除"。
PROTECTED_OUTPUTS: frozenset[str] = frozenset({
    "outputs/_handoff_read/A4_相机identify交接.md",
    "outputs/_handoff_read/pit39.md",
    "outputs/_handoff_read/pit40.md",
    "outputs/agent_detector/report.md",
    "outputs/agent_detector_design/DESIGN_specular_detector.md",
    "outputs/agent_gui_audit/doc_evidence.md",
    "outputs/agent_gui_audit/human_review.md",
    "outputs/agent_gui_audit/mono_calib_gap.md",
    "outputs/agent_gui_audit/structure.md",
    "outputs/agent_gui_fix/critic/baseline.md",
    "outputs/agent_gui_fix/critic/verdict.md",
    "outputs/audit_critic_75b/report.md",
    "outputs/audit_critic_75c/report.md",
    "outputs/audit_critic_75d/report.md",
    "outputs/cam/doctor_report.md",
    "outputs/cam/focus_report.md",
    "outputs/cam_diag/report.md",
    "outputs/chain_diagnosis_2026-11-08.md",
    "outputs/calib_cleanup_20261109.md",
    "outputs/gui_audit/report.md",
    "outputs/joint_refine/report.md",
    "outputs/motion_export_20261103/README.md",
})

#: 每个会话的 `session.json.bak-*` 至少留几个（编辑元数据的回滚点）。
#: 与 `apps._common.SESSION_KEEP_BACKUPS` 同一份策略 —— 这里是**清理阶段**的
#: 保留数，写备份时不立刻裁剪（见 `atomic_write_with_backup` 的 keep_backups=None）。
KEEP_BAK_PER_SESSION = SESSION_KEEP_BACKUPS


def is_protected_output(path: str | Path) -> bool:
    """这份文件是不是用户裁决原地保留的交付物？

    按**相对 `outputs/` 的路径**匹配，不认绝对路径：测试与其它机器上用临时仓库
    造同样的目录结构也能被保护，不依赖仓库装在哪块盘。
    """
    parts = Path(path).parts
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "outputs":
            key = "/".join(("outputs", *parts[i + 1:]))
            return key in PROTECTED_OUTPUTS
    return False


def _protected_reason(files: list[JunkFile]) -> str | None:
    """这批文件里含用户裁决的交付物？返回一句"不能删"的理由，否则 None。"""
    keep = [f for f in files if is_protected_output(f.path)]
    if not keep:
        return None
    return (f"含 {len(keep)} 份用户裁决原地保留的不可再生交付物"
            f"（如 {keep[0].path.name}）—— 清单见 docs/outputs保留清单.md，不能删")

#: "结果类"后缀：出现得多说明这个目录更像数据/素材，而不是调试输出。
_DATA_SUFFIXES = frozenset({
    ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff",
    ".mp4", ".mkv", ".avi", ".mov", ".webm",
    ".npy", ".npz", ".csv", ".json", ".pt", ".onnx",
})
_BIG_FILE_BYTES = 5 * 1024 * 1024       # 单个文件算"大"的门槛
_BIG_FILE_COUNT = 3                     # 有这么多大文件就先按数据看
_DATA_TOTAL_BYTES = 100 * 1024 * 1024   # 总量到这个数才看后缀比例
_DATA_RATIO = 0.6                       # 结果类后缀占六成 → 像数据集

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
    #: 降级/升级的原因（"看起来像数据，已降级为谨慎"这类）。空串 = 无特别说明。
    note: str = ""
    #: 用户在界面上标了"这项不动"。它仍会列出来（不能藏），但不默认勾选。
    muted: bool = False

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
    #: 声明过、但这次扫描**没找到**的位置（目录被改名/移动/删掉了）。
    #: 静默跳过会让用户以为"这里本来就没东西"，所以要能说出来。
    missing: list[str] = field(default_factory=list)

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


def looks_like_data(files: list[JunkFile]) -> str | None:
    """这批文件看起来像**数据/素材**而不是调试产物？返回原因，否则 None。

    为什么需要它（2026-11-04）：原来只看目录名 —— `outputs/_ab` 带 `_` 前缀，
    被判"推荐"（默认勾选），可它是 1.6 GiB / 15 个文件，形状更像数据。
    一条"按名字猜"的规则总有猜错的时候，猜错的代价是删掉不能再生的东西，
    所以再加一道**看内容**的判据，命中就降级。

    只判"是否像数据"，不判"是不是垃圾" —— 降级的语义是"我不确定，你自己勾"，
    不是"这是垃圾"。
    """
    if not files:
        return None
    total = sum(f.size for f in files)
    big = sum(1 for f in files if f.size >= _BIG_FILE_BYTES)
    if big >= _BIG_FILE_COUNT:
        return (f"有 {big} 个超过 5 MiB 的文件（共 {human_size(total)}），"
                "形状更像数据/素材")
    if total >= _DATA_TOTAL_BYTES:
        n_data = sum(1 for f in files if f.path.suffix.lower() in _DATA_SUFFIXES)
        ratio = n_data / len(files)
        if ratio >= _DATA_RATIO:
            return (f"共 {human_size(total)}，其中 {ratio:.0%} 是图片/视频/数组这类"
                    "结果文件，更像数据集")
    return None


#: 文件名里出现这些词 = 人工留下的实验产物，不是随手写的调试输出。
_RESEARCH_NAME_MARKS = (
    "calibration", "seeds", "report", "verdict", "evidence", "baseline",
    "review", "design", "交接", "结论",
)


def looks_like_research_artifact(files: list[JunkFile]) -> str | None:
    """目录里含**人工结论 / 标定结果 / 成组实验记录**？返回原因，否则 None。

    为什么必须补这一条（2026-11-05 逐项校验 53 个候选后的发现）：

    原来只有 `looks_like_data`（按大文件数 + 100 MiB 总量阈值）一道闸，
    结果下面这些**明显不能删**的目录全被判成「推荐」并默认勾选：

      outputs/_cvcheck   stereo_calibration.json / .npz   —— 标定结果（才 15 KiB）
      outputs/_diag      43 张诊断图 + 10 个 .txt 结论
      outputs/_jr        60 个对比 CSV + 11 个 .txt
      outputs/_handoff_read  A4_相机identify交接.md 等 44 份文档
      outputs/agent_gui_audit  doc_evidence.md / human_review.md
      outputs/tmp_frames* 从真实会话抽的 jpg 帧序列

    它们的共同点是**小**（没到 100 MiB）或**文档多**（不是数据后缀），
    恰好躲过了按体积/后缀比例的判据。清理器一旦默认勾上，用户点一下
    "清理"就会把实验证据和标定结果一起送走。

    所以再加三道**看语义**的判据：人写的文档、标定/seeds 这类命名、
    成组的结果文件（哪怕总量很小）。命中一律降级为「谨慎」——
    降级的语义是"我不确定，你自己勾"，不是"这是垃圾"。
    """
    if not files:
        return None
    # 第一道：用户**明确点名**的 22 份交付物。按体积/后缀/文档数量去"猜"总会有
    # 猜错的时候，而这里的裁决是确定的 —— 命中即"不能删"，不再往下判。
    protected = _protected_reason(files)
    if protected:
        return protected
    docs = [f for f in files if f.path.suffix.lower() == ".md"]
    if docs:
        return f"含 {len(docs)} 份人工文档（.md，如 {docs[0].path.name}）"
    txt = [f for f in files if f.path.suffix.lower() == ".txt"]
    if len(txt) >= 3:
        return f"含 {len(txt)} 个文字记录（.txt），可能是实验结论"
    for f in files:
        low = f.path.name.lower()
        if any(mark in low for mark in _RESEARCH_NAME_MARKS):
            return f"含实验产物：{f.path.name}"
    n_data = sum(1 for f in files
                 if f.path.suffix.lower() in _DATA_SUFFIXES)
    ratio = n_data / len(files)
    if n_data >= 5 and ratio >= 0.5:
        return (f"{n_data} 个结果文件占 {ratio:.0%}，"
                "是一组成组的实验记录（哪怕总量很小）")
    return None


def _dir_item(key: str, path: Path, *, label: str, detail: str, danger: str,
              skipped: list[str]) -> JunkItem | None:
    files = _walk_files(path, skipped=skipped)
    if not files:
        return None
    return JunkItem(key=key, label=label, detail=detail, danger=danger, files=files)


def _outputs_dir_item(d: Path, skipped: list[str]) -> JunkItem | None:
    """`outputs/` 下的一个一级子目录：先按名字给初判，再看内容是否要降级。

    两级判据的顺序是有意的：

    1. `_KNOWN_SAFE`（明确写死的名字）→ 直接推荐，**不做内容检查** ——
       `logs` 里也可能有 .json，但它是运行日志这件事是确定的；
    2. 前缀（`_`/`tmp_`/`agent_`）→ 初判推荐，**再过一遍 `looks_like_data`**：
       命中就降级成谨慎并写清原因（`outputs/_ab` 1.6 GiB 就是这么拦下来的）。

    新增一个 `outputs/xxx` 目录时会怎样：既不在 `_KNOWN_SAFE`、也不带调试
    前缀的，一律落到"谨慎"档（默认不勾）—— 这是安全的一侧：不知道就先别删。
    """
    explicit_safe = d.name in _KNOWN_SAFE
    if d.name == "logs":
        # 旧日志位置（搬家前是 outputs/logs）。还在 = 迁移没做完/旧版本写的残留；
        # 不存在是正常状态，所以它不参与下面的 missing 检查。
        return _dir_item(
            "outputs/logs", d, label="outputs/logs（旧日志位置）",
            detail="搬到项目根 logs/ 之前的残留 —— 删了不影响现在的日志",
            danger=DANGER_SAFE, skipped=skipped)
    if explicit_safe:
        detail, danger = _KNOWN_SAFE[d.name], DANGER_SAFE
    elif d.name.startswith(_RECOMMENDED_PREFIXES):
        detail = "调试/运行产物，删了下次跑就有"
        danger = DANGER_SAFE
    else:
        detail = ("某次实验的产物（数据集/标定图/走查截图等），"
                  "删了要重跑或重拍才能再生 —— 要不要留由你判断")
        danger = DANGER_WARN

    item = _dir_item(
        f"outputs/{d.name}", d, label=f"outputs/{d.name}", detail=detail,
        danger=danger, skipped=skipped)
    if item is None:
        return None

    if danger == DANGER_SAFE and not explicit_safe:
        why = looks_like_data(item.files) or looks_like_research_artifact(item.files)
        if why:
            item.danger = DANGER_WARN
            item.note = f"{why} —— 已从「推荐」降为「谨慎」，要清请手动勾选"
            item.detail = item.note
    elif danger == DANGER_WARN:
        # 本来就是「谨慎」的目录也要把裁决写清楚：靠"用途不明先别删"含糊过去，
        # 下一个人还是会把它当成可以勾的垃圾。
        why = _protected_reason(item.files)
        if why:
            item.note = why
            item.detail = why
    return item


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
        # glob 格式取自 `apps._common.backup_glob`（与写备份同源），不在这里再拼一份。
        baks = sorted(
            (p for p in sj.parent.glob(backup_glob(sj))
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


def scan_junk(root: str | Path, *, older_than_days: float | None = None,
              muted: object = ()) -> JunkPlan:
    """盘点可清理项。

    ``older_than_days``：只算 **mtime 早于** ``now - N 天`` 的文件。
    ``None`` = 不限时间（全部）。界面默认就是"全部"——因为默认勾选的只有
    "删了下次跑就有"那一档，时间再筛一遍是过度保守：实测本项目 3.9 GiB 垃圾里
    3.9 GiB 都是最近 7 天产生的，默认 30 天前会让功能看起来"什么都没有"。
    时间范围是**第二道**筛子，给想更保守的人用；筛掉多少会记在
    ``filtered_bytes`` 里如实显示。

    只从 `_rules()` 列出的位置读，不做"全盘找像垃圾的文件"。

    ``muted``：用户标过"这项不动"的 key 集合。它们**仍会被扫描并列出**
    （藏起来会让人以为清理器漏了东西），只是不默认勾选。
    """
    repo = Path(root).resolve()
    plan = JunkPlan()
    skipped = plan.skipped
    muted_set = {str(m) for m in (muted or ())}

    if not repo.is_dir():
        raise ValueError(f"仓库根不存在：{repo}")

    outputs = repo / "outputs"
    data = repo / "data"

    # ---- ⓪ 项目根的 logs/（推荐档）----
    # 日志是排障时**唯一能回看的东西**，所以既不能混进 outputs/ 被整类清掉，
    # 也不能因为搬了家就从"一键清日志"里消失。
    logs_root = repo / "logs"
    if logs_root.is_dir():
        item = _dir_item(
            "logs", logs_root, label="logs/",
            detail="运行日志：按来源（tasks/calib/capture/realtime）与月份分目录",
            danger=DANGER_SAFE, skipped=skipped)
        if item is not None:
            plan.items.append(item)

    # ---- ① outputs 下的一级子目录（推荐 + 谨慎两档）----
    # 声明过的位置如果**不存在**了，要说出来（改名/移动/删掉都算），
    # 否则界面上"运行日志"这一项直接消失，用户会以为本来就没有。
    for _name in _KNOWN_SAFE:
        if not (outputs / _name).is_dir():
            plan.missing.append(
                f"outputs/{_name}（这次没找到 —— 目录可能被改名/移动，已跳过）")

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

        # outputs 根下**其余**的散落文件（截图 / 导出包 / 一次性脚本…）。
        # 不归进来的话它们永远不出现在列表里 —— 用户看着有东西却清不到，
        # 而这正是"清理器看起来不全"的来源。形状各异，判不了用途，
        # 所以一律"谨慎"：让用户自己看一眼再勾。
        stray: list[JunkFile] = []
        for p in outputs.iterdir():
            if not p.is_file() or p.is_symlink():
                continue
            if p.suffix in (".tmp", ".partial") or ".tmp-" in p.name:
                continue
            if p.name.startswith(("tmp_", "_tmp", ".")):
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            stray.append(JunkFile(path=p, size=st.st_size, mtime=st.st_mtime))
        if stray:
            # 22 份里有 2 份是散落在 outputs 根的 .md（诊断/清理清单快照），
            # 它们不在任何子目录里，`_outputs_dir_item` 管不到，所以这里补一句。
            keep_why = _protected_reason(stray)
            plan.items.append(JunkItem(
                key="outputs/散落其他文件", label="outputs 根下的其他散落文件",
                detail=keep_why or
                       "截图 / 导出包 / 一次性脚本这类散落在 outputs 根的东西 —— "
                       "不属于任何子目录，用途只有你知道，所以默认不勾",
                danger=DANGER_WARN, files=stray,
                note=keep_why or ""))

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

    if muted_set:
        for item in plan.items:
            if item.key in muted_set:
                item.muted = True
                if not item.note:
                    item.note = "你标了「不动」—— 取消勾选会记住，下次仍不勾"

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
    留着 `logs/` 这种空壳，下次看列表会以为"没删掉"。
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
            # 用户裁决的 22 份交付物：即使被手动勾选也**拒绝删除**。这是第三道
            # 安全边界（另两条是"只删计划里的路径"与"越界/符号链接拒绝"）——
            # 「谨慎」档只保证默认不勾，真正"不能删"要在执行处兜住。
            if is_protected_output(p):
                failures.append(f"{p}（用户裁决：不可再生的交付物，拒绝删除）")
                continue
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
