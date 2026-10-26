"""标定方法注册表：`joint` / `chain` 的**唯一**清单（规则 R7）。

以前这两个名字散在 `stereo_calib.py` 的 `__post_init__` 白名单、`calibrate_stereo()`
的 if 分支、`apps/calibrate_stereo.py` 的 `choices` 与三处互斥校验、以及 GUI 里
硬写的 `--method chain --same-board` 上。加第三种方法要改五个文件、漏一处就静默
不一致；现在只有这里一份，其余地方一律 `names()` / `describe()` / `plan()`。

每个方法登记的不是"一段计算"，而是一份**行为声明**（`MethodPlan`）：
调用方按字段决定走哪条路，不再比较方法名的字符串。这样"新方法要不要共视、
能不能实时采集"是新方法自己的事，不用回头改调用方。

`chain` 的实现延迟 import（`_calibrate_chain` 在 `stereo_calib.py` 里，
而 `stereo_calib.py` 要 import 本模块）——模块级 import 会成环。
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

from sphere_tracker.plugins import default, names, plugin, resolve

__all__ = ["MethodPlan", "plan", "single_board"]


@dataclasses.dataclass(frozen=True)
class MethodPlan:
    """一种外参解法的行为声明。字段都**恰好被用在一处**，不是通用配置。"""

    name: str
    # 产物口径：`stereo_rectify` = 做了极线校正；`none_raw` = 只去畸变、不校正。
    # 决定能否标成 parallel 模式（parallel 重建依赖校正后的极线与视差）。
    rectification: str
    # 是否要求"每个位姿两路都同时看到板"。joint 是；chain 不是（这正是它的用途）。
    needs_covisibility: bool
    # 是否只能在"已经拍好的目录"上跑（实时采集走不了这条路）。
    offline_only: bool
    # 是否需要 `--same-board` 这个前置条件（单块固定板当公共参考）。
    same_board: bool
    # 真正的实现；None = 主路径（joint）内联在 `calibrate_stereo()` 里，
    # 不在这里再包一层 —— 那 300 行是数值主路径，搬动只会增加漂移风险。
    run: Callable[..., Any] | None = None


@plugin(
    "calib_method", "joint",
    summary="一次联合求解内外参：两路在每个位姿都要同时看到标定板（平行 / 充分共视装置）",
    default=True,
)
def _joint() -> MethodPlan:
    return MethodPlan(name="joint", rectification="stereo_rectify",
                      needs_covisibility=True, offline_only=False, same_board=False)


@plugin(
    "calib_method", "chain",
    summary="单目先标内参、再用公共参考串外参：板不必每个位姿都被两路同时看到（正交 / 大角度装置）",
    needs=("同摆放参考图",),   # 前置条件：一块参考板在两路里都拍到，且期间不动
)
def _chain() -> MethodPlan:
    from .stereo_calib import _calibrate_chain  # 延迟：见模块 docstring 的成环说明

    return MethodPlan(name="chain", rectification="none_raw",
                      needs_covisibility=False, offline_only=True, same_board=True,
                      run=_calibrate_chain)


def plan(name: str | None = None) -> MethodPlan:
    """按名字取行为声明；`None` = 注册表声明的默认方法（R4，不硬编码名字）。"""
    return resolve("calib_method", name or default("calib_method"))()


def single_board() -> str:
    """需要 `--same-board` 的那个方法的名字（唯一）。

    GUI 的「单板链式标定」卡片靠它决定拼哪条命令：卡片的前提（板不动、两路各自
    拍参考图）就是该方法的 `same_board`，所以不必在界面里再写一遍方法名。
    多于一个 / 一个都没有都报错 —— 那说明声明本身自相矛盾，不该猜。
    """
    hits = [n for n in names("calib_method") if plan(n).same_board]
    if len(hits) != 1:
        raise LookupError(
            f"声明了 same_board 的标定方法应恰好 1 个，实际 {len(hits)} 个：{hits or '无'}"
        )
    return hits[0]
