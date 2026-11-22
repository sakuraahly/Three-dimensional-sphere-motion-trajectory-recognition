"""双目会话的管理动作：移到回收站、改装置模式（含搬到对应分区）。

为什么要单独一个模块
--------------------
这两个动作都要**动用户数据**，而且各有各的坑：

- 删除：录像是不可再生的，只能软删（复用 ``apps/_cleanup`` 的回收站语义）；
- 改装置模式：要同时改 ``session.json`` 的字段**并且**把目录搬到对应分区，
  否则"记录说 angled、却还躺在 parallel/ 下" —— 那正是这次要消灭的状态。

把它们放在界面代码里（``_gui_stereo.SessionPicker`` 的方法）会让"搬目录 +
改 JSON + 备份"和 Treeview 的选中状态缠在一起，也很难单独测。这里保持
**纯函数**：拿路径、返回路径或抛 ValueError，界面只负责确认对话框和刷新列表。

单目为什么不需要这一层：``apps/_mono_sessions.py`` 本来就有
``delete_mono_session``，而单目没有"装置几何"这个维度，不存在改模式。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from apps._cleanup import move_to_trash, plan_cleanup
from apps._common import atomic_write_with_backup
from apps._stereo_layout import MODE_LABELS, is_legacy_flat
from sphere_tracker.paths import STEREO_MODES, STEREO_ROOT, stereo_mode_root

__all__ = [
    "TRASH_LABEL", "delete_stereo_session", "describe_layout", "mode_display",
    "set_session_mode",
]

#: 回收站批次名后缀（与单目的 `mono` 区分，方便在数据管理里认出是谁删的）。
TRASH_LABEL = "stereo"


def mode_display(mode: str | None) -> str:
    """``parallel`` → 「平行双目」；没记录就是「未记录」。"""
    return MODE_LABELS.get(str(mode or "").strip().lower(), "未记录")


def delete_stereo_session(session_dir: str | Path,
                          root: str | Path | None = None) -> Path:
    """**软删除**：整个会话移到 ``data/_trash/<时间>_stereo/``，返回批次路径。

    为什么不 ``rmtree``：录像是不可再生的。项目已有回收站语义
    （``apps/_cleanup.py``：预览 → 移动 → 可恢复/可彻底删），这里**复用**它，
    而不是再写一套移动逻辑 —— 恢复与彻底删除也就自动共用同一份实现。

    ``allow_nested=True`` 是这次新加的（2026-11-04）：会话现在按装置分区，
    躺在 ``data/stereo/<mode>/<时间戳>``，不放宽就只能删旧扁平层里的会话。

    ``root`` 默认取 ``data/stereo``（**不是** ``data/stereo/<mode>``）：
    回收站按 ``root.parent/_trash`` 定位，传分区根会把回收站建到
    ``data/stereo/_trash``，和单目/数据管理看到的 ``data/_trash`` 不一致。
    """
    from apps._stereo_layout import is_mode_root

    d = Path(session_dir)
    if is_mode_root(d):
        # 分区根长得和旧扁平层的会话一模一样（都是 data/stereo 的直接子目录）。
        # 不挡这一下，一次"删会话"会把整个分区的会话全递归移进回收站。
        raise ValueError(
            f"{d.name}/ 是装置分区根，不是一段会话；"
            "要删里面的会话请逐条选中（或到数据管理里批量选）")
    base = Path(root) if root is not None else STEREO_ROOT
    plan = plan_cleanup(base, [d], mode="session", allow_nested=True)
    if not plan.items:
        raise ValueError("没有可删除的会话：" + ("；".join(plan.skipped) or str(d)))
    return move_to_trash(base, plan, label=TRASH_LABEL)


def _rewrite_json(jf: Path, data: dict) -> None:
    """备份 + 原子替换，委托 `apps._common.atomic_write_with_backup`（唯一实现）。

    `keep_backups=None` = 写时不裁剪，回滚点由 `apps/_junk.py` 统一按 3 份保留；
    与 `data_manager.update_session_meta` 同一手法、同一份格式串。
    """
    atomic_write_with_backup(jf, data, keep_backups=None)


def _free_target(dest: Path) -> Path:
    """目标目录已存在时加 ``-01``/``-02`` 后缀（与录制器建会话同一规则）。"""
    if not dest.exists():
        return dest
    n = 1
    while True:
        cand = dest.with_name(f"{dest.name}-{n:02d}")
        if not cand.exists():
            return cand
        n += 1


def set_session_mode(session_dir: str | Path, mode: str,
                     root: str | Path | None = None) -> Path:
    """改这段会话的**装置几何**，并把它搬到对应分区，返回新路径。

    为什么需要手工改（而不是"录的时候写对就行了"）
    ------------------------------------------------
    磁盘上已有的 43 段会话里，**29 段没有 ``stereo_mode`` 字段**（字段是
    ``a137766`` 才加的）。没有字段 = 扫描时既不算平行也不算大角度，两个窗口
    都不敢认，隔离就永远差一截。这个动作让人**一次性把它们归位**。

    顺序很重要：**先改 JSON、再搬目录**。

    - 先改 JSON：万一搬目录失败（目标被占用、跨盘权限），记录已经是新的，
      只是"躺在旧分区" —— 下次打开列表会看到"位置与记录不符"，能再点一次；
    - 反过来（先搬后改）会留下"记录说旧模式、目录在新分区"的会话，
      而扫描按记录过滤 → 它在新窗口里**看不见**，在旧窗口里点开会跨越分区，
      症状是"会话莫名其妙消失"。

    搬动目标固定为 ``data/stereo/<mode>/``；已经在那儿就只改字段（幂等）。
    旧位置 ``data/sessions/`` 里的会话也会被搬进新布局 —— 那是**写入**，
    符合"新写入一律走分区根"的约定。
    """
    m = str(mode or "").strip().lower()
    if m not in STEREO_MODES:
        raise ValueError(f"装置模式只能是 {STEREO_MODES}，收到 {mode!r}")

    src = Path(session_dir)
    jf = src / "session.json"
    if not jf.is_file():
        raise ValueError(f"这条会话没有 session.json：{src}")
    try:
        data = json.loads(jf.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"读不了 {jf}：{exc}；文件未改动") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{jf} 顶层不是 JSON 对象，拒绝改动")

    dest_root = stereo_mode_root(m)
    dest_root.mkdir(parents=True, exist_ok=True)
    already = src.parent.resolve() == dest_root.resolve()

    # 已经全对就直接返回：**不要**再写一遍 JSON。重复点「改装置模式」不该
    # 每次都留一个 session.json.bak-*（那些备份会越攒越多，把会话目录弄乱）。
    rig = data.get("rig")
    rig_ok = not (isinstance(rig, dict) and "stereo_mode" in rig) or \
        str(rig.get("stereo_mode") or "").strip().lower() == m
    if already and str(data.get("stereo_mode") or "").strip().lower() == m and rig_ok:
        return src

    # ① 先改记录（含 rig.stereo_mode —— 两个字段各存一份，改一个会自相矛盾）
    data["stereo_mode"] = m
    if isinstance(rig, dict) and "stereo_mode" in rig:
        rig["stereo_mode"] = m
    if "id" in data and str(data.get("id")) == src.name:
        # 目录名可能因重名加后缀而变化，先按旧名写；真改名了下面再修一次
        data["id"] = src.name
    _rewrite_json(jf, data)

    if already:
        return src

    # ② 再搬目录
    dest = _free_target(dest_root / src.name)
    try:
        shutil.move(str(src), str(dest))
    except OSError as exc:
        raise OSError(
            f"模式已改为「{mode_display(m)}」，但目录没搬成（{exc}）。\n"
            f"它现在还在 {src.parent}，列表里会显示「位置与记录不符」；"
            "关掉占用该目录的程序后再点一次「改装置模式」即可。") from exc

    if dest.name != src.name:
        # 重名加了后缀：把 id 一并改成新目录名（扫描用的是目录名，id 只是冗余，
        # 但两处不一致会让"按 id 找会话"的逻辑多一个特例）
        try:
            moved = json.loads((dest / "session.json").read_text(encoding="utf-8"))
            if isinstance(moved, dict) and "id" in moved:
                moved["id"] = dest.name
                _rewrite_json(dest / "session.json", moved)
        except (OSError, ValueError):
            pass
    return dest


def describe_layout(session_dir: str | Path) -> str:
    """给人看的一句话：这段会话摆在哪、记录是什么（不一致时明说）。

    会话管理器用它填状态条 —— 用户点一行就能看到"它属于哪套装置、
    是不是还躺在旧扁平层"，不用自己去翻 session.json。
    """
    from apps._stereo_layout import layout_mode_of, session_mode_of

    d = Path(session_dir)
    rec, lay = session_mode_of(d), layout_mode_of(d)
    if rec and lay and rec != lay:
        return (f"记录是{mode_display(rec)}，却摆在 {lay}/ 分区里 —— "
                f"位置与记录不符，点「改装置模式」可修正")
    if is_legacy_flat(d):
        return (f"记录是{mode_display(rec)}，但还在旧扁平目录 "
                f"（data/stereo/ 下没分区）—— 点「改装置模式」可搬进分区")
    return f"装置：{mode_display(rec)}"
