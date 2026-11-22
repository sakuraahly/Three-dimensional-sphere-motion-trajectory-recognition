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

from datetime import datetime
from pathlib import Path

__all__ = [
    "ROOT", "DATA_ROOT", "STEREO_ROOT", "MONO_ROOT", "LEGACY_STEREO_ROOT",
    "TRASH_ROOT", "CALIB_PAIRS_ROOT", "STEREO_MODES", "stereo_roots",
    "stereo_mode_root", "stereo_mode_roots", "ensure_dirs",
    "LOGS_ROOT", "LOG_CATEGORIES", "LOG_CATEGORY_LABELS", "log_dir",
    "new_log_path", "legacy_log_dirs",
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

#: 双目装置的两种几何。**同时也是会话分区的目录名**（见 `stereo_mode_root`）。
#:
#: 为什么把分区名定成这两个词而不是"平行/大角度"：`session.json` 里的
#: `stereo_mode` 就是这两个值，目录名与元数据同一套词汇，才不会出现
#: "目录写 parallel、字段写 90°"这种两处各说一套的情况。
STEREO_MODES = ("parallel", "angled")


def stereo_mode_root(mode: str) -> Path:
    """某种装置几何的会话**写入**根：``data/stereo/<mode>``。

    为什么按装置分目录（2026-11-04，用户报"平行和大角度的会话没有完全隔离"）
    ---------------------------------------------------------------------
    以前两种装置的会话都直接落在 ``data/stereo/<时间戳>``，物理上混在一起，
    只能靠 ``session.json`` 的 ``stereo_mode`` 在列表里过滤。这有两个洞：

    1. **未记录的老会话**没有该字段，于是两个窗口都列得出来 —— 用户分不清
       哪段是哪种装置录的，用错几何解算会得到系统性错的深度且不报错；
    2. 列表过滤是**约定**，不是**结构**：任何人写一个扫描脚本就会漏掉这条规则。

    分成 ``data/stereo/<mode>/`` 之后，"哪段会话属于哪套装置"成了目录结构本身，
    磁盘上一眼可辨，写错的成本也高得多。旧布局仍然能读（见
    `apps/_stereo_layout.py` 的候选枚举），新写入一律走这里。
    """
    m = str(mode or "").strip().lower()
    if m not in STEREO_MODES:
        raise ValueError(f"stereo_mode 只能是 {STEREO_MODES}，收到 {mode!r}")
    return STEREO_ROOT / m


def stereo_mode_roots(mode: str) -> tuple[Path, ...]:
    """扫**某种**装置会话时要看的根，新的在前。

    = 该模式的分区根 + 旧扁平根 ``data/stereo`` + 旧位置 ``data/sessions``。
    后两个是**读**兼容：迁移没做、或别人 clone 的仓库里，会话还在扁平层。
    扫描方必须再按 ``session.json`` 的 ``stereo_mode`` 过滤一遍 ——
    旧的扁平根里混着两种装置的会话，光看目录分不出来。
    """
    return (stereo_mode_root(mode), STEREO_ROOT, LEGACY_STEREO_ROOT)


def stereo_roots() -> tuple[Path, ...]:
    """扫双目会话时要看的根目录，**新的在前**。

    保留旧位置是为了"迁移没做/别人 clone 的仓库"这两种情况仍能列出会话；
    写入路径一律用 `stereo_mode_root()`（按装置分区），不要用这个函数的结果。

    注意：返回的 ``data/stereo`` 下面**既有**会话目录（``<时间戳>``，旧布局），
    **也有**分区目录（``parallel`` / ``angled``）。枚举会话要用
    ``sphere_tracker.discovery.discover_session_dirs()``（``apps/_stereo_layout
    .candidate_session_dirs()`` 是它的多根包装）—— 它递归往下走，认得
    这两种位置，也认得用户自己建的嵌套目录。
    """
    return (STEREO_ROOT, LEGACY_STEREO_ROOT)


#: 日志根：**项目根**下的 ``logs/``。
#:
#: 2026-11-04 用户原话："我们没有专门小的日志文件夹吗？只有 logs 这大的文件夹？
#: 这很不应该。……把 logs 文件夹放在项目根目录。"
#:
#: 两条都在说同一件事：日志的**位置**和**结构**都该自解释。
#:
#: - 位置：以前在 ``outputs/logs/``。可 ``outputs/`` 的定义是"删了能重跑"，
#:   垃圾管理器会把它当一整类可清产物看待；而日志恰恰是**出问题后唯一能回看的
#:   东西**，排在 `outputs/` 第三层里，找人要先把"输出"这层语义猜对。放到根目录
#:   与 `data/` `config/` 平级，一眼可见，也不必先理解 outputs 的分层。
#: - 结构：以前 9196 个文件全平铺在一层，任何文件管理器滚起来都卡，
#:   "找上周那次标定"只能靠文件名排序硬找。现在按来源分目录、再按时间分片。
LOGS_ROOT = ROOT / "logs"

#: 日志按**来源**分区。一个来源一个子目录，名字就是"谁写的"。
#:
#: 为什么按来源而不是按级别/按日期：排障时人的第一问是"哪个程序出的事"
#: （标定？录制？实时？），不是"哪天的"。日期已经是第二层（``<YYYY-MM>``）。
#: 按级别分更不行 —— 一次运行本来就会同时产出 INFO 和 DEBUG，拆开反而对不上。
LOG_CATEGORIES = ("tasks", "calib", "capture", "realtime", "misc")

#: 每个分区的含义（文档、界面、垃圾管理器共用一套说法）。
LOG_CATEGORY_LABELS = {
    "tasks": "启动器里每个任务的完整输出（谁起的、跑了多久、退出码）",
    "calib": "标定：双目标定 / 单目标定 / 采板图",
    "capture": "录像：双目 / 单目采集",
    "realtime": "实时检测",
    "misc": "其他：训练、一次性脚本、手工重定向的测试输出",
}

#: 第二层的分片粒度**按量级定**，不是一刀切。
#:
#: 实测（2026-11-04 搬家时）：启动器任务一个月 6876 个文件，按月分仍然是
#: 一个大目录；标定一个月 139、录制 61、实时 22，按日分会碎成"一天一个
#: 只放几个文件的空壳"。所以任务按日，其余按月 —— 判据是"打开目录能一屏
#: 看完"，不是"所有目录长得一样"。
_LOG_GRANULARITY = {"tasks": "%Y-%m-%d"}
_DEFAULT_GRANULARITY = "%Y-%m"


def log_dir(category: str = "misc", *, when: datetime | None = None) -> Path:
    """本次日志该写进的目录：``logs/<category>/<时间分片>/``。

    分两层而不是一层：第一层回答"哪个程序"，第二层把单个目录的文件数压到
    "打开能一屏看完"的量级。第二层的粒度见 `_LOG_GRANULARITY`（任务按日、
    其余按月），不搞一刀切。

    ``category`` 必须在 `LOG_CATEGORIES` 里 —— 拼错会静默造出一个没人知道的
    野生目录，那种目录既不在文档里，也不会被"一键清日志"覆盖。宁可直接报错。

    ``when`` 只给测试与迁移用；正常运行取当前时间。
    """
    c = str(category or "").strip().lower()
    if c not in LOG_CATEGORIES:
        raise ValueError(
            f"未知日志分区 {category!r}；可选：{', '.join(LOG_CATEGORIES)}"
            "（新增分区请同时更新 LOG_CATEGORIES 与本函数的文档）")
    fmt = _LOG_GRANULARITY.get(c, _DEFAULT_GRANULARITY)
    slot = (when or datetime.now()).strftime(fmt)
    return LOGS_ROOT / c / slot


def new_log_path(category: str, tag: str, *,
                 when: datetime | None = None) -> Path:
    """分配一个日志文件路径，并把它的目录建好。

    命名沿用旧的 ``<tag>_<YYYYMMDD_HHMMSS>.log``：改结构不该改文件名，
    否则按名字搜索老日志的习惯会断掉。
    """
    stamp = (when or datetime.now()).strftime("%Y%m%d_%H%M%S")
    directory = log_dir(category, when=when)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{tag}_{stamp}.log"


def legacy_log_dirs() -> tuple[Path, ...]:
    """旧日志位置，**只读**。

    两个用途：迁移脚本按它找老文件；垃圾管理器把它当"可能还有残留"的
    兼容位置列出来（迁移做完了它就是空的，不该再报"目录丢了"）。
    """
    return (ROOT / "outputs" / "logs",)


def ensure_dirs() -> None:
    """建好需要存在的目录（幂等）。录制/会话创建前调用。"""
    for p in (STEREO_ROOT, MONO_ROOT, *(STEREO_ROOT / m for m in STEREO_MODES)):
        p.mkdir(parents=True, exist_ok=True)
