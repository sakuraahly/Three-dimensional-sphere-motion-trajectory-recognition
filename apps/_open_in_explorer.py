"""在系统文件资源管理器里**打开 / 定位**一个路径：apps 层的共用薄层。

为什么单独一个模块
------------------
「在文件资源管理器中打开会话文件夹」不是数据管理专属：数据管理要开会话目录，
导出完成后的「打开所在文件夹」要开产物目录，以后别的界面还可能要"选中某个
文件"（:func:`reveal_in_explorer`）。把「打开哪个」与「怎么打开」分开：
界面层只负责挑路径，本模块只负责安全校验 + 跨平台拉起文件管理器。
不 import tkinter、不弹窗、不认任何 apps GUI 类型 —— 命令行脚本、
GUI 自动走查工具、将来的别的窗口都能直接复用。

为什么放在 apps/ 而不是 sphere_tracker/
--------------------------------------
``sphere_tracker/`` 刻意保持纯 stdlib 且**不许依赖 apps**；而"拉起系统文件
管理器"是桌面端的事，跟算法库无关。放 ``apps/_*.py`` 私有模块层，与
``_cleanup.py`` / ``_fov_store.py`` / ``_common.py`` 同类。反过来若下沉到
``sphere_tracker/``，任何 import 它的进程都得先付 GUI / 系统层的账
（``docs/文件结构.md`` 记着 ``import apps.data_manager`` 让启动从 0.29s
涨到 1.56s 的教训）。

为什么失败一律抛异常、不返回 False
----------------------------------
调用点对"打不开"的处理完全不同：数据管理要弹 ``showerror``，导出完成后的
"打开所在文件夹"只是顺手一按、静默即可。若这里返回 ``False``，调用方一句
``if not ok: pass`` 就会把错误吞掉。所以**只报成功（``True``）或抛
:class:`OpenInExplorerError`**，怎么呈现交给调用方。

Windows 首选 os.startfile，但必须运行时取属性
--------------------------------------------
``os.startfile`` 只在 Windows 存在；且 ``apps/tools/gui_walkthrough.py`` 的
GUI 自动走查会把它**替换成只记事件的 lambda**（不真弹窗）。所以必须
``getattr(os, "startfile", None)`` 在调用时才取：既不能在模块顶层
``from os import startfile``，也不能把结果缓存到模块级变量 —— 否则打桩
拿不到，走查时会真的弹出资源管理器窗口打扰用户。只有它缺失或抛异常时，
才降级到 ``explorer``。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

__all__ = [
    "OpenInExplorerError", "open_in_explorer", "reveal_in_explorer",
]


class OpenInExplorerError(RuntimeError):
    """路径非法（不存在 / 符号链接 / 越界）或系统无可用文件管理器。"""


def _resolve(raw: str | Path, allowed_root: str | Path | None) -> Path:
    """把输入收敛成一个**存在、非符号链接、不越界**的绝对路径。

    安全惯例与 ``apps/_cleanup.py`` 一致：先看 ``raw.is_symlink()`` 直接拒，
    再 ``resolve()``（顺带把 ``..`` 规范化，挡掉路径逃逸），最后
    ``is_relative_to`` 守边界。``allowed_root=None`` 表示不做边界限制 ——
    导出产物落在 ``outputs/`` 下，本来就不在会话根里，传会话根反而会被
    误判成越界。
    """
    raw = Path(raw)
    if raw.is_symlink():
        raise OpenInExplorerError(f"拒绝符号链接：{raw}")
    p = raw.resolve()
    if not p.exists():
        raise OpenInExplorerError(f"路径不存在：{p}")
    if allowed_root is not None:
        root = Path(allowed_root).resolve()
        if not p.is_relative_to(root):
            raise OpenInExplorerError(f"路径越界（不在 {root} 内）：{p}")
    return p


def open_in_explorer(path: str | Path, *,
                     allowed_root: str | Path | None = None) -> bool:
    """打开 ``path`` 所在的文件夹；传进来是文件则自动取其父目录。

    成功返回 ``True``；路径非法或系统没有可用文件管理器一律抛
    :class:`OpenInExplorerError`（见模块文档串，不返回 ``False`` 让人吞掉）。
    """
    p = _resolve(path, allowed_root)
    folder = p.parent if p.is_file() else p

    if os.name == "nt":
        # 运行时取属性：GUI 自动走查把 os.startfile 换成记录事件的 lambda，
        # 这里必须命中它，绝不能走到下面的 explorer Popen（会真弹窗）。
        startfile = getattr(os, "startfile", None)
        if startfile is not None:
            try:
                startfile(str(folder))
                return True
            except OSError:
                pass  # 只有它抛异常时才降级
        try:
            subprocess.Popen(["explorer", str(folder)])
        except OSError as exc:
            raise OpenInExplorerError(f"打不开文件资源管理器：{exc}") from exc
        return True

    if sys.platform == "darwin":
        try:
            subprocess.Popen(["open", str(folder)])
        except OSError as exc:
            raise OpenInExplorerError(f"打不开 Finder：{exc}") from exc
        return True

    try:
        subprocess.Popen(["xdg-open", str(folder)])
    except OSError as exc:
        raise OpenInExplorerError(f"打不开文件管理器：{exc}") from exc
    return True


def reveal_in_explorer(file: str | Path, *,
                       allowed_root: str | Path | None = None) -> bool:
    """在文件资源管理器里**选中**某个文件；Linux 退回打开它的父目录。

    Windows 用 ``explorer /select,<绝对路径>``：整串作为**一个**参数、
    ``shell=False``、不手工加引号 —— 中文 / 空格交给 CreateProcessW 处理，
    手工拼引号反而会在带空格的路径上断错。本次数据管理按钮不用它，
    只作为"别的地方可能要用"的模块能力保留。
    """
    p = _resolve(file, allowed_root)

    if os.name == "nt":
        try:
            subprocess.Popen(["explorer", f"/select,{p}"])
        except OSError as exc:
            raise OpenInExplorerError(f"无法定位文件：{exc}") from exc
        return True

    if sys.platform == "darwin":
        try:
            subprocess.Popen(["open", "-R", str(p)])
        except OSError as exc:
            raise OpenInExplorerError(f"无法在 Finder 中显示：{exc}") from exc
        return True

    return open_in_explorer(p.parent, allowed_root=allowed_root)
