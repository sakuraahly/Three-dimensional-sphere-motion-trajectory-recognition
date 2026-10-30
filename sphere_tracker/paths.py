"""数据目录的**唯一来源**。

以前 `data/sessions` 这个字面量散在 8 处（`data_manager` / `record_stereo` /
`check_dataset` / `view_3d` / `_gui_app` / `_gui_bench` / `_gui_stereo` / 文档），
而单目又另起了一个 `data/mono` —— 于是"会话"这个词在磁盘上指两个不同东西，
谁也说不清 `data/sessions` 里到底是不是全部会话。2026-10-29 用户原话：
"data 下面 session 和 mono 命名有歧义"。

现在按**类型**分目录，名字自解释：

    data/stereo/<YYYYMMDD-HHMMSS>/   双目会话：left/right 视频 + session.json + derived/
    data/mono/<YYYYMMDD-HHMMSS>/     单目会话：clip.mkv + session.json + derived/

"会话"仍然是概念（一个目录 = 一次采集），但**目录名不再承担这个歧义**。

兼容：`data/sessions/` 是旧位置，`stereo_roots()` 会把它一起返回（只读），
所以老数据、老脚本、别人 clone 下来的仓库都还能用；新写入一律走 `STEREO_ROOT`。

⚠️ 这里只放**路径**，不放"怎么扫会话"。扫目录的规则各有各的判据
（双目要 `session.json` 的 schema，单目要容忍"只有录制器边车"的 CLI 直录），
塞进来会变成一个大而全又谁都不敢改的函数。
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "ROOT", "DATA_ROOT", "STEREO_ROOT", "MONO_ROOT", "LEGACY_STEREO_ROOT",
    "TRASH_ROOT", "CALIB_PAIRS_ROOT", "stereo_roots", "ensure_dirs",
]

#: 仓库根。按标记上溯，不数层数（文件被移动时会静默指错别处）。
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "sphere_tracker").is_dir())

DATA_ROOT = ROOT / "data"
STEREO_ROOT = DATA_ROOT / "stereo"
MONO_ROOT = DATA_ROOT / "mono"
TRASH_ROOT = DATA_ROOT / "_trash"
CALIB_PAIRS_ROOT = DATA_ROOT / "calib_pairs"

#: 旧的双目会话根（`data/sessions/`）。**只读兼容**，不再往里写。
LEGACY_STEREO_ROOT = DATA_ROOT / "sessions"


def stereo_roots() -> tuple[Path, ...]:
    """扫双目会话时要看的根目录，**新的在前**。

    保留旧位置是为了"迁移没做/别人 clone 的仓库"这两种情况仍能列出会话；
    写入路径一律用 `STEREO_ROOT`，不要用这个函数的结果。
    """
    return (STEREO_ROOT, LEGACY_STEREO_ROOT)


def ensure_dirs() -> None:
    """建好需要存在的目录（幂等）。录制/会话创建前调用。"""
    for p in (STEREO_ROOT, MONO_ROOT):
        p.mkdir(parents=True, exist_ok=True)
