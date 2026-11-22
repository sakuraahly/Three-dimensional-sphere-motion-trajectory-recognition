"""会话目录的**唯一**枚举口径：一个根下面到底有哪些会话。

为什么单独一层（2026-11-09）
----------------------------
这个仓库曾经有**三套**扫描器，同一份磁盘给出三个不同的数：

- ``stereo.sessions.list_sessions``：``Path(root).glob("*/session.json")``，
  只扫一层，实测 **29** 段（``parallel/`` 里那 15 段全漏）；
- ``apps/_stereo_layout.candidate_session_dirs``：只认两层，且不检查有没有
  ``session.json``（空目录也返回）—— 用户建 ``data/stereo/实验1/<时间戳>``
  之后，那段会话在所有走它的面板里静默消失；
- ``apps/_session_index.discover_session_dirs``：递归、命中即剪枝，实测 **44** 段。

数据管理面板用第三套，所以它看得见三层嵌套；会话列表、标定入口、``view_3d``
的默认打开用前两套，于是"看得见用不了"。根因不是某个函数写错，而是同一个事实
有三个产地 —— 迟早对不上，这次就对不上了。所以枚举收归此处，**只此一份**。

为什么放在 ``sphere_tracker/``（同一天搬过来的）
------------------------------------------------
分层是 ``sphere_tracker → calib → stereo → apps``，这里是最底层：
``sphere_tracker`` **不依赖** ``stereo`` / ``calib`` / ``apps`` 任何一个（实测
按顶层 import 扫一遍，零命中），所以上面每一层都能放心 import 它，不会被它
反过来拖起来。

第一版放在 ``stereo/`` 包里（该文件已删除）：当时的理由只是"``stereo/`` 不许
import ``apps/``"，结果任何 import 它的进程都要先执行 ``stereo/__init__.py`` ——
那会把 ``pair_source``（cv2）和 ``sync``（scipy.signal）全拉起来。实测
数据管理窗口的启动从 0.06s 涨到 **1.56s**，全花在枚举根本用不到的
重依赖上。一个纯 stdlib 的目录枚举不该背这 1.5 秒的账，所以移到最底层。

现在的关系：

- ``stereo.sessions.list_sessions`` 直接调 ``discover_session_dirs``；
- ``apps/_stereo_layout.candidate_session_dirs`` 只是多根包装；
- ``apps/_session_index`` 只是转发导入。

两条判据（宽 / 严），别合并
---------------------------
- **宽**：``looks_like_session_dir`` —— 有 ``session.json`` **或**有视频 /
  ``derived`` / 标定之类的东西。数据管理用它：录制崩在写 ``session.json``
  之前、只剩 ``left.mkv`` 的半成品必须列出来给人修，不能视而不见。
- **严**：``strict=True`` —— 只认 ``session.json``。会话列表 / 标定 / 校验用
  它：没有元数据的目录跑不了算法，列出来只会让人白点一次。

怎么保证不慢
------------
**命中即剪枝**：认出是会话就不再往里走（会话内部的 ``derived/`` 可能有上千个
文件），再跳过 ``SKIP_DIR_NAMES`` 里的已知非会话目录，深度默认上限 5 层。
仓库吃过"全树无剪枝 ``rglob`` 扫 15.9 秒"的教训（``check_standards.py``），
这里绝不用 ``rglob``。

``recursive=False`` 是什么
--------------------------
旧的两层口径：根的直接子目录 + 分区再往下一层。留着它是为了让"就只想看一层"
的调用方有地方表达，而不是各自回去写一份 ``glob`` —— 那正是三套扫描器的起点。
它和 ``recursive=True`` 共用同一个循环，只把深度上限压到 2（根记为第 0 层，
所以"根 → 分区 → 会话"三层正好够）；**不**再判断目录名是不是
``parallel``/``angled``，那种写法正是"用户建 ``实验1/<时间戳>`` 就静默消失"的来源。

中文路径原样返回，不做任何转码/规范化 —— ``Path`` 拿到的就是磁盘上的名字。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from sphere_tracker.video_files import LEFT_VIDEO_NAMES, RIGHT_VIDEO_NAMES

__all__ = [
    "SESSION_FILE", "SESSION_MARKERS", "SKIP_DIR_NAMES",
    "discover_session_dirs", "looks_like_session_dir", "rel_key", "sid_of",
]

#: 会话元数据文件名。录制器**先落**这一份（见 ``record_stereo`` 的落盘顺序），
#: 所以"这段会话存在吗"以它为准最稳。``stereo/sessions.py`` 从这里取，不再写字面量。
SESSION_FILE = "session.json"

#: 递归枚举时**不进**的目录名。``derived`` / ``calib`` 是会话内部目录：
#: 一旦某个会话的 ``session.json`` 丢了，没有这条排除就会把
#: ``<会话>/derived`` 当成一段会话列出来（状态全是 ✗ 坏）。
SKIP_DIR_NAMES = frozenset({
    "_trash", "derived", "calib", "outputs", "logs", ".git", "__pycache__",
    ".ipynb_checkpoints", "node_modules", ".venv", "assets", "exports",
})

#: sid 里不能出现的字符：``/`` 是分组路径的分隔符，其余是 Windows 非法字符。
_SID_BAD = re.compile(r"[\\/:*?\"<>|\x00-\x1f]")

#: "这看起来是一段会话"的标记文件/目录。用 ``SESSION_FILE`` **以外**的证据，
#: 是为了让**录制中途崩掉、还没写 session.json** 的半成品也能列出来 ——
#: 那是用户最需要看见的东西（数据管理原来就承诺"坏目录也列出来给人看"），
#: 递归扫描如果只认 session.json，这些半成品会当场消失。
#:
#: 录像名**从 video_files 派生**（不另抄一份）：zip 左右两路，顺序仍是
#: left.mkv/right.mkv/left.mp4/right.mp4/[left.avi/right.avi]，加 avi 前与
#: 历史字面量逐项一致。这样"avi 会话能被枚举"与"avi 能被拾取"共用同一个
#: 事实源 —— 哪天 video_files 漏改，判据 4 会跟判据 1/3 一起红。
SESSION_MARKERS = (
    tuple(name for pair in zip(LEFT_VIDEO_NAMES, RIGHT_VIDEO_NAMES) for name in pair)
    + ("clip.mkv", "left_times.csv", "right_times.csv", "derived", "calib")
)

#: ``recursive=False``（旧两层口径）的深度上限：根记为第 0 层，「根 → 分区 →
#: 会话」正好是第 1、2 层，所以上限 2 就够，且**看不到**用户自己套的第三层。
_LEGACY_DEPTH = 2


# ----------------------------------------------------------------------
# 身份：sid 与相对路径
# ----------------------------------------------------------------------
def sid_of(session_dir: str | Path) -> str:
    """会话的**稳定身份**：``session.json`` 的 ``id``，没有就用目录名。

    为什么不能直接用目录名当身份：目录名在整理时会变（用户把会话拖进
    ``parallel/`` 不影响名字，但改名叫"梨子-1"就变了），而 ``id`` 是录制时
    写进 ``session.json`` 的、跟着文件走。

    读不出 ``session.json`` 时退回目录名 **不报错**：数据管理要能显示坏目录
    （用户得看见它才能修），不能因为一个坏文件让整个列表打不开。
    """
    p = Path(session_dir)
    raw: object = None
    try:
        raw = json.loads((p / SESSION_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = None
    if isinstance(raw, dict):
        cand = raw.get("id")
        if isinstance(cand, str) and cand.strip():
            return _SID_BAD.sub("_", cand.strip())
    return _SID_BAD.sub("_", p.name)


def rel_key(root: str | Path, session_dir: str | Path) -> str:
    """会话相对根的路径（``/`` 分隔，可直接给人看）。不在根下面就用目录名。

    相对路径**不参与身份**，只用于：去重（同一个物理目录只出现一次）、
    显示"它躺在哪个分区"、以及 ``index.json`` 里方便人肉定位。
    """
    r = Path(root)
    p = Path(session_dir)
    try:
        return p.resolve().relative_to(r.resolve()).as_posix()
    except (OSError, ValueError):
        return p.name


# ----------------------------------------------------------------------
# 枚举：一个根下面有哪些会话
# ----------------------------------------------------------------------
def looks_like_session_dir(path: str | Path) -> bool:
    """这个目录**像不像**一段会话（不解析内容，只看有没有会话该有的东西）。

    判据是"有 ``session.json``" **或**"有视频/标定/derived 之类的东西"。
    后者不是猜内容，而是为了**半成品**：录制器崩在写 ``session.json`` 之前时，
    目录里有 ``left.mkv`` 但没有元数据 —— 用户必须在列表里看见它才能修，
    不能因为它"不是合法会话"就视而不见。
    """
    p = Path(path)
    if (p / SESSION_FILE).is_file():
        return True
    for name in SESSION_MARKERS:
        try:
            if (p / name).exists():
                return True
        except OSError:
            continue
    return False


def discover_session_dirs(
    root: str | Path, *, recursive: bool = True, max_depth: int = 5,
    strict: bool = False,
) -> list[Path]:
    """列出根下面**所有**会话目录，按目录名倒序（= 时间倒序）。

    这是全仓库唯一的枚举实现：数据管理、会话列表、标定入口、``view_3d``
    的默认打开都走它。以前各写一套的后果是同一份磁盘数出 29 / 44 两个数，
    现在只有一个。

    ``recursive=True``（默认）：从根往下走，**命中即剪枝**（认出是会话就不
    往里走）、跳过 ``SKIP_DIR_NAMES``、深度上限 ``max_depth``（根记第 0 层）。

    ``recursive=False``：旧的两层口径，同一个循环把深度压到 2（根 → 分区 →
    会话）。保留它只是让"只看一层"的调用方有地方表达，不再有第二套判断逻辑。

    ``strict=False``（默认）：宽判据 ``looks_like_session_dir``，半成品也算
    （数据管理的口径：坏目录也要列出来给人修）。
    ``strict=True``：只认 ``SESSION_FILE``（会话列表 / 标定 / 校验的口径）。

    **返回值只含命中判据的目录**。这一条改变了
    ``apps/_stereo_layout.candidate_session_dirs`` 的旧行为（它以前会把
    ``实验1`` 这种空目录也返回，让调用方自己过滤），是收敛不是倒退：
    调用方本来就要过滤，过滤规则写 N 遍的结果就是有人漏了。

    排序键 ``(目录名, 相对路径)`` 倒序：目录名是时间戳，所以时间新的在前；
    不同分区下同名会话（``parallel/20261021-120000`` 与
    ``angled/20261021-120000`` 是两段不同的会话）靠相对路径保持稳定顺序。
    """
    base = Path(root)
    if not base.is_dir():
        return []

    limit = max_depth if recursive else _LEGACY_DEPTH

    def hit(p: Path) -> bool:
        """严判据只认 session.json，宽判据还认"半成品"的证据。"""
        if strict:
            return (p / SESSION_FILE).is_file()
        return looks_like_session_dir(p)

    out: list[Path] = []
    seen: set[Path] = set()

    def push(p: Path) -> None:
        try:
            key = p.resolve()
        except OSError:
            return
        if key in seen:
            return
        seen.add(key)
        out.append(p)

    stack: list[tuple[Path, int]] = [(base, 0)]
    while stack:
        here, depth = stack.pop()
        try:
            subs = [x for x in here.iterdir() if x.is_dir()]
        except OSError:
            continue
        for x in subs:
            if x.name.lower() in SKIP_DIR_NAMES:
                continue
            if hit(x):
                push(x)                      # 认出即剪枝：会话内部不再往下看
                continue
            if depth + 1 < limit:
                stack.append((x, depth + 1))

    out.sort(key=lambda p: (p.name, rel_key(base, p)), reverse=True)
    return out
