"""原始左右录像文件名的**唯一产地**：``left/right.mkv|mp4``。

为什么单独一层（2026-11-15）
----------------------------
"会话里的原始录像叫什么"以前写在**三处**，各自手写元组：

- ``apps/view_3d.py``：左/右**分开**取，``("left.mkv", "left.mp4")`` 与
  ``("right.mkv", "right.mp4")``，mkv 优先 —— 决定 raw 模式到底打开哪个文件；
- ``apps/data_manager.py``：**并集** ``("left.mkv", "right.mkv", "left.mp4",
  "right.mp4")`` —— 只回答"这段会话有没有原始视频"，决定「查看 3D + 视频」
  能不能进 raw 模式；
- ``apps/_data_export.py``：**同一份并集** —— 按它把文件收进交付包的
  ``videos/raw/``。

三处口径**已经漂过一次**：按钮检查的是 4 个名字的并集、查看器按左/右分开取，
一度靠两边"口头对齐"才没出事。根因不是某个函数写错，而是同一个事实有三个
产地 —— 加一种扩展名时漏改任何一处，界面会告诉你"有原始视频"，打开却一个
都不在。所以收归此处，**只此一份**。

四条不许动的行为（都是现场踩出来的）
------------------------------------
1. **mkv 优先**：``left.mkv`` 与 ``left.mp4`` 同时存在时取 mkv（录制器先落
   mkv，mp4 是后转的、可能还是半成品）。顺序写进 ``LEFT_VIDEO_NAMES`` /
   ``RIGHT_VIDEO_NAMES`` **本身**，调用方不许自己再排一遍；
2. **只看会话根目录**：``(p / name).is_file()``，**不递归**。``raw/left.mkv``
   这种"看着像"的布局不认 —— 录制器写根层、导出包解回来也是根层，多认一层
   就等于又造一套"到底哪个才是原始素材"；
3. **单路放行**：只录了一路（只有左或只有右）不是错误，``pick_*`` 各自返回
   路径或 ``None``，**不抛异常**。查看器要能在画面上明说"右路没有"；
4. **是文件才算**：用 ``is_file()`` 而不是 ``exists()`` —— 目录叫
   ``left.mkv`` 不该骗过"有没有原始视频"。三处判断在这点上也必须一致，否则
   又是一个新的双口径。

为什么放在 ``sphere_tracker/`` 而不是 ``stereo/``
--------------------------------------------------
和 ``discovery.py`` 同一个理由（搬家记录见该模块的文档串）：
``stereo/__init__.py`` 会把 ``pair_source``（cv2）与 ``sync``（scipy.signal）
全拉起来，实测数据管理窗口的启动从 0.06s 涨到 **1.56s**。本模块只用
``pathlib``，纯 stdlib，谁都能放心 import，不会让"查个文件名"背上 1.5 秒。

``RAW_VIDEO_EXTENSIONS`` 与 ``ALL_RAW_VIDEO_NAMES`` 不是一回事
--------------------------------------------------------------
前者是**扩展名**集合，给"这后缀算不算录像"这类判断用；后者是**文件名**清单。
绝不按扩展名去扫目录认素材：录制中的临时文件（``left.tmp.mp4``）也带合法
扩展名，扫出来就会被当成可导出的原始素材。
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "RAW_VIDEO_EXTENSIONS", "LEFT_VIDEO_NAMES", "RIGHT_VIDEO_NAMES",
    "ALL_RAW_VIDEO_NAMES", "pick_left_video", "pick_right_video",
    "has_any_raw_video",
]

#: 原始录像认得的扩展名。**不是** ``ALL_RAW_VIDEO_NAMES`` 的来源（见模块文档串）。
#: 顺序 mkv > mp4 > avi：既保既有测试与"mkv 优先"硬约束，也覆盖外部 avi 输入。
RAW_VIDEO_EXTENSIONS = (".mkv", ".mp4", ".avi")

#: 左路素材名，**顺序即优先级**：mkv > mp4 > avi。
LEFT_VIDEO_NAMES = ("left.mkv", "left.mp4", "left.avi")

#: 右路素材名，同上。
RIGHT_VIDEO_NAMES = ("right.mkv", "right.mp4", "right.avi")

#: 并集，用于"有没有原始视频"与导出时的收集/写出顺序。
ALL_RAW_VIDEO_NAMES = LEFT_VIDEO_NAMES + RIGHT_VIDEO_NAMES


def _pick(session_dir: str | Path, names: tuple[str, ...]) -> Path | None:
    """按 ``names`` 的顺序在会话**根目录**里找第一个存在的文件。"""
    base = Path(session_dir)
    for name in names:
        cand = base / name
        if cand.is_file():
            return cand
    return None


def pick_left_video(session_dir: str | Path) -> Path | None:
    """左路原始录像（mkv 优先）；只有 mp4 时返回 mp4，都没有返回 ``None``。

    ``None`` 是合法结果（单路会话），调用方不许当成错误处理。
    """
    return _pick(session_dir, LEFT_VIDEO_NAMES)


def pick_right_video(session_dir: str | Path) -> Path | None:
    """右路原始录像，语义与 :func:`pick_left_video` 完全对称。"""
    return _pick(session_dir, RIGHT_VIDEO_NAMES)


def has_any_raw_video(session_dir: str | Path) -> bool:
    """这段会话**至少有一路**原始录像吗（``LEFT`` + ``RIGHT`` 并集）。

    数据管理用它决定「查看 3D + 视频」能不能进 raw 模式：只要有一路就不拦，
    少一路由查看器自己说明。判据是 ``is_file()``，与 :func:`pick_left_video`
    一致 —— "按钮说有"和"查看器能打开"必须同时成立。
    """
    base = Path(session_dir)
    for name in ALL_RAW_VIDEO_NAMES:
        if (base / name).is_file():
            return True
    return False
