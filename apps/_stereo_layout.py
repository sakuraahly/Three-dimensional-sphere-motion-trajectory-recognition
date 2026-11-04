"""双目会话的**磁盘布局**：按装置几何分区之后，怎么找、怎么判定一段会话属于哪套装置。

背景
----
2026-11-04 用户报「大角度双目和平行双目的会话没有完全隔离开」。查下来是的：

- 两种装置的会话都直接落在 ``data/stereo/<时间戳>``，物理上混在一起；
- 隔离全靠 ``session.json`` 的 ``stereo_mode`` 在列表里过滤，而 **29/43 段
  老会话根本没有这个字段** —— 于是它们在同两个窗口里都列得出来。

现在写入按装置分区（``data/stereo/parallel/…``、``data/stereo/angled/…``），
但**读**必须同时认三种位置，否则迁移没做的仓库、或别人 clone 的数据就"消失"了：

1. ``data/stereo/<mode>/<时间戳>``  ← 新布局（**唯一写入位置**）
2. ``data/stereo/<时间戳>``          ← 旧扁平布局（只读兼容）
3. ``data/sessions/<时间戳>``        ← 更早的位置（只读兼容）

这个模块只回答三件事，都**不碰 session.json 的内容**（那是 `_gui_stereo`
的扫描与 `_stereo_sessions` 的编辑各自的事）：

- 有哪些目录**可能**是会话（`candidate_session_dirs`）；
- 这段会话**记录**的模式是什么（`session_mode_of`，可能 None）；
- 这段会话**摆在哪**（`layout_mode_of`，按目录结构推断）。

为什么"记录的模式"和"摆的位置"要分开
------------------------------------
两者不一致时必须能**看出来**，而不是悄悄取其中一个：

- 分区目录里躺着一段标 ``stereo_mode=parallel`` 的会话（有人手工挪过、
  或迁移时按错的元数据归类）→ 摆的位置说 angled、记录说 parallel；
- 旧扁平层里一段标 ``angled`` 的会话 → 位置不表态、记录说 angled。

`effective_mode()` 的规则是「**记录优先**，记录没有才看位置」——因为
``record_stereo`` 写 ``stereo_mode`` 时已经做过「模式 vs 标定」的一致性体检
（``stereo/record_plan.plan_record``：不一致直接拒绝录），字段比目录名可信。
目录名是给人看的，字段是给程序用的。
"""

from __future__ import annotations

import json
from pathlib import Path

from sphere_tracker.paths import (
    LEGACY_STEREO_ROOT,
    STEREO_MODES,
    STEREO_ROOT,
    stereo_mode_root,
)
from sphere_tracker.paths import stereo_roots as _stereo_roots

__all__ = [
    "MODE_LABELS", "candidate_session_dirs", "effective_mode", "ensure_mode_root",
    "is_legacy_flat", "is_mode_root", "layout_mode_of", "looks_like_session",
    "mode_label", "mode_root_or_none", "session_mode_of",
]

#: 模式给人看的说法（界面文案与日志共用一套，避免"平行/parallel"各写一半）。
MODE_LABELS = {"parallel": "平行双目", "angled": "大角度双目"}


def mode_label(mode: str | None) -> str:
    """``parallel`` → 「平行双目」；没记录就是「未记录」。"""
    return MODE_LABELS.get(str(mode or "").strip().lower(), "未记录")


def mode_root_or_none(mode: str | None) -> Path | None:
    """模式对应的分区根；非法/空模式返回 None（不抛：扫描路径要能容错）。"""
    m = str(mode or "").strip().lower()
    return stereo_mode_root(m) if m in STEREO_MODES else None


def ensure_mode_root(mode: str) -> Path:
    """建好分区根并返回它（录制前调用，幂等）。"""
    root = stereo_mode_root(mode)
    root.mkdir(parents=True, exist_ok=True)
    return root


def looks_like_session(path: Path) -> bool:
    """这个目录**是不是**一段会话：只认 ``session.json`` 的存在，不解析内容。

    为什么不看 ``derived/`` 或视频：CLI 直录的会话可能只有录制器边车、
    也可能视频还在写；``session.json`` 是录制器**先落**的那份（见
    ``record_stereo`` 的落盘顺序），拿它当存在性判据最稳。
    """
    return (Path(path) / "session.json").is_file()


def candidate_session_dirs(
    roots: tuple[Path, ...] | list[Path] | None = None,
) -> list[Path]:
    """列出**所有可能**是会话的目录，新布局在前、各自按目录名倒序。

    规则（只认结构，不读内容）：

    - 根的直接子目录里，名字是 ``parallel``/``angled`` 的 → 当作**分区**，
      再往里看一层（那才是会话）；
    - 其余直接子目录 → 当作旧扁平布局的会话。

    返回的目录**不保证**都有 ``session.json``：调用方要自己 ``looks_like_session``
    或解析元数据 —— 有些扫描要"把坏目录也列出来给人看"（数据管理的口径），
    有些只要好会话（会话列表的口径）。在这里过滤掉就等于替调用方做了决定。
    """
    out: list[Path] = []
    seen: set[Path] = set()

    def push(p: Path) -> None:
        try:
            resolved = p.resolve()
        except OSError:
            return
        if resolved in seen:
            return
        seen.add(resolved)
        out.append(p)

    src = tuple(roots) if roots is not None else _stereo_roots()
    # 先走分区（新布局），再走扁平（旧布局）：调用方按顺序去重时"新的那份优先"。
    for root in src:
        base = Path(root)
        if not base.is_dir():
            continue
        try:
            subs = sorted((d for d in base.iterdir() if d.is_dir()),
                          key=lambda d: d.name, reverse=True)
        except OSError:
            continue
        for d in subs:
            if d.name.lower() in STEREO_MODES:
                try:
                    inner = sorted((x for x in d.iterdir() if x.is_dir()),
                                   key=lambda x: x.name, reverse=True)
                except OSError:
                    continue
                for x in inner:
                    push(x)
            else:
                push(d)
    return out


def session_mode_of(session_dir: str | Path) -> str | None:
    """读 ``session.json`` 里记录的模式；读不出/没记录就 None（**不猜**）。

    为什么不像有些扫描那样"缺字段就默认 parallel"：老会话缺这个字段，
    默认成 parallel 会让 90° 装置录的会话在平行窗里看起来**可用** ——
    比不显示更糟（用错几何解算得到系统性错的深度，而且不报错）。
    """
    try:
        raw = json.loads((Path(session_dir) / "session.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    mode = raw.get("stereo_mode")
    if mode is None:
        rig = raw.get("rig")
        mode = rig.get("stereo_mode") if isinstance(rig, dict) else None
    m = str(mode or "").strip().lower()
    return m if m in STEREO_MODES else None


def layout_mode_of(session_dir: str | Path) -> str | None:
    """按**目录结构**推断模式：父目录名是 ``parallel``/``angled`` 就是它。

    只对"分区布局"成立；旧扁平层（``data/stereo/<时间戳>``）的父目录是
    ``stereo``，返回 None —— 这正确表达了"位置不表态，只能看元数据"。
    """
    parent = Path(session_dir).parent.name.lower()
    return parent if parent in STEREO_MODES else None


def is_mode_root(path: str | Path) -> bool:
    """这个目录是不是**分区根本身**（``data/stereo/parallel``）而不是会话。

    为什么必须单独判（真实缺口，2026-11-04）：分区根是 ``data/stereo`` 的
    **直接子目录**，长得和旧扁平层的会话一模一样。清理逻辑只认"直接子目录"
    时会把它当成一段会话 —— 于是一删就把整个分区的会话全递归移走。
    数据管理的列表不会列出它（`candidate_session_dirs` 往里多走了一层），
    但会话管理器的入参来自用户选中项，多一道显式检查才安全。

    判据是「**名字是模式名，而且自己不是会话（没有 session.json）**」，
    不比对全局 ``STEREO_ROOT`` —— 那样在临时根（测试、或用户用『数据目录…』
    指向别的盘）上会失效，保护就静默关掉了。
    """
    p = Path(path)
    return p.name.lower() in STEREO_MODES and not looks_like_session(p)


def effective_mode(session_dir: str | Path) -> str | None:
    """这段会话**属于哪套装置**：记录优先，记录没有才看它摆在哪。

    两者不一致时**以记录为准**（``record_stereo`` 落盘前已核对过模式与标定），
    但调用方若想知道"摆错位置"这件事，用 ``session_mode_of`` 与
    ``layout_mode_of`` 各取一次对比即可（会话管理器就靠这个提示"位置与记录不符"）。
    """
    return session_mode_of(session_dir) or layout_mode_of(session_dir)


def is_legacy_flat(session_dir: str | Path) -> bool:
    """这段会话是不是躺在**旧扁平层**里（没进任何分区）。

    会话管理器用它决定"要不要提示整理"，也决定「改装置模式」是"只改字段"
    还是"改字段 + 搬目录"。
    """
    p = Path(session_dir)
    return layout_mode_of(p) is None and p.parent in (STEREO_ROOT, LEGACY_STEREO_ROOT)
