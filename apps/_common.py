"""apps/ 下各命令行脚本共用的样板。

放在 ``apps/`` 而不是 ``sphere_tracker/``：这里装的是**应用层的取参约定**
（参数名、默认值、失败时的退出码），属于"怎么调库"，不该混进算法库。

抽出来的原因：``calibrate_stereo`` / ``verify_stereo`` / ``check_stereo_setup``
三个脚本的取帧参数是**同一套**，且都直接喂给 ``stereo.from_args()``。三份各写
一遍，迟早会出现"标定支持了 ``--left-video``、复检却不支持"这类不一致。

用法（放在各脚本自己的 ROOT 引导之后）::

    from apps._common import add_stereo_source_args, open_pair_source

    ap = argparse.ArgumentParser(description="...")
    add_stereo_source_args(ap)                 # 补上 --left/--right/--width/...
    args = ap.parse_args()
    src = open_pair_source(args)               # 失败时已打印原因，返回 None
    if src is None:
        return 1
"""

from __future__ import annotations

import argparse


def add_stereo_source_args(
    ap: argparse.ArgumentParser,
    *,
    width: int = 1920,
    height: int = 1080,
    fps: int = 60,
) -> argparse.ArgumentParser:
    """补上双目取帧参数 —— 字段名与 ``stereo.from_args()`` 读的一一对应。

    参数名/默认值集中在这里，三个双目 app 就永远同步，不会出现
    "这个脚本能用、那个不能"。
    """
    ap.add_argument("--left", default=None,
                    help="左相机：索引(0) / 角色名(left，查 config/cameras.json) / auto")
    ap.add_argument("--right", default=None,
                    help="右相机：索引(2) / 角色名(right，查 config/cameras.json) / auto")
    ap.add_argument("--auto-cameras", action="store_true",
                    help="自动挑两台可用相机（优先外接、避开内置/红外、非全黑）")
    ap.add_argument("--backup-camera", type=int, default=None,
                    help="备用相机索引：左右任一路打不开时自动顶上")
    ap.add_argument("--backend", choices=["auto", "dshow", "msmf"], default="auto",
                    help="相机后端；两台相机必须用同一个后端，auto 会自动成对挑一个")
    ap.add_argument("--left-video", default=None, help="左路视频文件（与右路成对）")
    ap.add_argument("--right-video", default=None, help="右路视频文件")
    ap.add_argument("--left-dir", default=None, help="左路图片文件夹")
    ap.add_argument("--right-dir", default=None, help="右路图片文件夹")
    ap.add_argument("--width", type=int, default=width,
                    help="相机采集宽度（两路一致）")
    ap.add_argument("--height", type=int, default=height,
                    help="相机采集高度（两路一致）")
    ap.add_argument("--fps", type=int, default=fps, help="相机采集帧率")
    return ap


def open_pair_source(args, *, exit_code: int = 1):
    """按参数装配 ``stereo.PairSource``；失败时打印原因并返回 ``None``。

    约定：返回 ``None`` 就由调用方 ``return exit_code``。三个双目脚本原本各自
    写了一遍同样的 try/except，收进这里保证提示口径一致（``check_stereo_setup``
    的退出码是 2，其余是 1，所以 ``exit_code`` 留成参数）。
    """
    from stereo import from_args

    try:
        return from_args(args)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"\n[错误] {exc}")
        return None


__all__ = ["add_stereo_source_args", "open_pair_source"]
