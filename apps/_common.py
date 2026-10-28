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

from sphere_tracker.video_profiles import DEFAULT_FPS

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# apps/_common.py 的上一级就是项目根（apps/ 在根目录下）。
ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "outputs" / "logs"

from sphere_tracker import video_profiles  # noqa: E402  (ROOT 已就绪)

_LOG_FORMAT = "%(asctime)s %(levelname)-5s %(message)s"
_LOG_DATEFMT = "%H:%M:%S"


class _MillisecondFormatter(logging.Formatter):
    """时间戳带毫秒。

    为什么要毫秒：标定的价值在**帧间时序** —— "第 137 帧板停稳了、下一帧又判在动"
    这种问题，秒级时间戳看不出来。
    """

    def formatTime(self, record, datefmt=None):  # noqa: N802 (标准库命名)
        dt = datetime.fromtimestamp(record.created)
        return dt.strftime("%H:%M:%S") + f".{int(record.msecs):03d}"


def setup_logging(
    tag: str,
    *,
    log_dir: Path | None = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> Path:
    """把日志**同时**写到终端和 ``outputs/logs/<tag>_<时间戳>.log``，返回文件路径。

    为什么要落盘：本项目最贵的一类失败是**静默失败** —— 标定收了 8 对就不动了，
    进程一退，拒绝原因的计数就没了，只能靠人在窗口前反复试。落盘之后，
    "为什么没收满"变成一个可以回看、可以贴给别人、可以跨次比对的文件。

    分工（照 STC 的"库不配日志、app 才配"）：
      - **库层**（``calib/`` ``stereo/`` …）只 ``getLogger(__name__)`` 并写记录，
        不装 handler、不决定写哪儿；
      - **app 层**（本函数）负责装 handler、定路径、定级别。
    这样库可以被测试直接 import 而不产生日志文件。

    级别分工：文件收 ``DEBUG``（每帧判定都留），终端只显示 ``INFO`` ——
    一帧一行打到终端会把有用信息淹掉。
    """
    directory = Path(log_dir) if log_dir is not None else LOG_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    root = logging.getLogger()
    # 只配置一次：同一个进程里被调用两次（或测试里反复跑）不应重复叠加 handler。
    if not getattr(root, "_dsh_configured", False):
        root.setLevel(logging.DEBUG)

        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setLevel(file_level)
        fh.setFormatter(_MillisecondFormatter(_LOG_FORMAT, _LOG_DATEFMT))
        root.addHandler(fh)

        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(console_level)
        sh.setFormatter(_MillisecondFormatter(_LOG_FORMAT, _LOG_DATEFMT))
        root.addHandler(sh)

        root._dsh_configured = True  # type: ignore[attr-defined]

    return path


def log_environment(log: logging.Logger) -> None:
    """把"这次是在什么环境里跑的"记进日志开头。

    为什么值得记：同一个标定失败，在 OpenCV 4.5 和 5.0 上原因可能完全不同，
    没有版本信息就只能重跑一遍去问。
    """
    import platform

    log.info("Python %s  (%s)", platform.python_version(), sys.executable)
    log.info("平台 %s", platform.platform())
    try:
        import cv2

        log.info("OpenCV %s", cv2.__version__)
    except Exception as exc:  # 环境自检会另外报，这里不该因为它中断
        log.warning("读不到 OpenCV 版本：%s", exc)


def add_stereo_source_args(
    ap: argparse.ArgumentParser,
    *,
    width: int = video_profiles.DEFAULT.width,
    height: int = video_profiles.DEFAULT.height,
    fps: int = DEFAULT_FPS,
) -> argparse.ArgumentParser:
    """补上双目取帧参数 —— 字段名与 ``stereo.from_args()`` 读的一一对应。

    参数名/默认值集中在这里，三个双目 app 就永远同步，不会出现
    "这个脚本能用、那个不能"。

    默认帧率 **24**（2026-10-24 真机实测）：**1440p 全尺寸写盘**时
    ``record_stereo`` 主循环约 24.07fps，请求 30 会丢帧；
    但 **FOV 对齐后写盘 1718x972（1.67MP）时 30fps 稳定**（301 帧/10s 实测）。
    两者不是一回事，别把 24 当成"这台机器只能 24"。
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
                    help=f"相机采集宽度（两路一致；默认档 {video_profiles.DEFAULT.label}）")
    ap.add_argument("--height", type=int, default=height,
                    help=f"相机采集高度（两路一致；默认档 {video_profiles.DEFAULT.label}）")
    ap.add_argument("--fps", type=int, default=fps,
                    help=f"相机采集帧率（默认 {fps}）。1440p 全尺寸写盘约 24fps；"
                         "FOV 对齐后写盘 1718x972 时 30fps 实测稳定")
    return ap


def apply_mono_calibration(cfg, args) -> str:
    """把单目内参（含畸变）填进 ``cfg.camera``，返回一句**来源说明**。

    优先级：``--focal``（显式临时覆盖）> 标定文件 > ``hfov_deg`` 估算。
    返回的这句话由调用方打印 —— 单目测深的绝对尺度完全取决于焦距，
    "这个 fx 是标出来的还是猜的"必须每次都写在脸上（计划书 §3.4）。

    没有标定文件**不是错误**：那是"还没标"，程序照旧跑，只是精度按估算走。
    """
    from calib.mono_calib import apply_to, load_mono

    path = getattr(args, "mono_calib", None)
    calib = None
    if not getattr(args, "no_mono_calib", False):
        calib = load_mono(path) if path else None

    if calib is not None:
        apply_to(calib, cfg.camera)
        note = (f"内参：{Path(path).name}  fx={calib.fx:.1f}  "
                f"主点=({calib.cx:.0f},{calib.cy:.0f})  "
                f"畸变={'有' if calib.has_distortion else '无'}  "
                f"RMS={calib.rms_px:.2f} px")
    elif getattr(args, "no_mono_calib", False):
        note = (f"内参：按 --no-mono-calib 忽略标定，"
                f"焦距按 {cfg.camera.hfov_deg:g}° 视场估算")
    else:
        note = (f"内参：没有标定文件（{path}）—— "
                f"焦距按 {cfg.camera.hfov_deg:g}° 视场估算，"
                f"1% 焦距误差 = 0.6 m 处 6 mm")

    if getattr(args, "focal", None):
        cfg.camera.fx = args.focal
        note += f"；--focal 又覆盖为 {args.focal:.0f}"
    return note


def add_mono_pipeline_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """补上**单目**链路的八个公共参数。

    ============  ==========================================================
    `--config`    读取 JSON 配置（标定结果）
    `--method`    检测方法 —— 选项/默认/说明全部来自 detector 注册表
    `--preset`    颜色预设 —— 选项来自 `ColorConfig.preset_names()`
    `--ball-diameter` / `--focal`   单目测距所需的两个物理先验
    `--predict` / `--horizon`       预测方法与其视野
    `--gravity-fit`                 三维预测时用已知重力约束
    ============  ==========================================================

    ## 为什么要有它（2026-10-27）

    单目有两个入口：``run_realtime.py``（相机）与 ``run_video.py``（离线文件 /
    图片序列）。它们跑的是**同一条流水线**（检测 → 跟踪 → 预测），这八个参数
    必须同名、同默认、同说明。原来是两份手抄，实测已经漂了：

    - ``--config`` 实时那边写了"读取 JSON 配置（标定结果）"，离线那边一个字没有；
    - ``--ball-diameter`` / ``--focal`` / ``--horizon`` / ``--gravity-fit``
      的说明同样只在实时那边有；
    - ``--preset`` 的说明里**手抄了一份预设名单** —— 那是预设表的第三份副本
      （另外两份在 `calibrate_color.py` 和 GUI 下拉里），加一种球色要改四处。

    漂移的后果不是崩溃，而是"同一个参数在两个入口里解释不一样"，用户只能靠猜。

    ## 说明文字一律派生，不手写清单（R7）

    检测方法来自 `plugins.describe("detector")`、预测方法来自 predictor 注册表、
    颜色预设来自 `ColorConfig.preset_names()`。**加一种实现/一种球色，两个入口的
    帮助同时变**，这个函数不用动。

    ## 前置条件：注册表得已经填好

    本函数在**调用时**读注册表，所以调用方必须先 import 注册模块
    （各入口第二段就做：``from sphere_tracker.detectors import registry``，
    import 即注册，R1）。忘了会在下面直接抛出一条指明修法的错误，
    而不是给出一份空的可选值。
    """
    from sphere_tracker.config import ColorConfig
    from sphere_tracker.plugins import default as default_name
    from sphere_tracker.plugins import describe, names

    if not names("detector"):
        raise RuntimeError(
            "add_mono_pipeline_args() 被调用时 detector 注册表还是空的 —— "
            "请在同一模块的顶部先 `from sphere_tracker.detectors import registry`"
            "（import 即注册，R1），再构造 ArgumentParser。"
            "否则 --method 的可选值会是空的，而且报错位置离原因很远。"
        )

    ap.add_argument("--config", default=None, help="读取 JSON 配置（标定结果）")
    ap.add_argument(
        "--method", default=None, choices=names("detector"),
        help="检测方法（不传则取配置，默认 %s）。可选：%s" % (
            default_name("detector"),
            "；".join(f"{d['name']}={d['summary']}" for d in describe("detector"))),
    )
    ap.add_argument(
        "--preset", default=None, choices=ColorConfig.preset_names(),
        help="颜色预设（球色），不传则用配置里的颜色。可选：%s" % (
            " / ".join(ColorConfig.preset_names())),
    )
    ap.add_argument("--ball-diameter", type=float, default=None,
                    help="球直径，单位米（用于三维测距）")
    ap.add_argument("--focal", type=float, default=None,
                    help="相机像素焦距 fx（标定过就填）")
    ap.add_argument(
        "--predict", default=None, choices=names("predictor"),
        help="预测方法（不传则取配置，默认 %s）。可选：%s" % (
            default_name("predictor"),
            "；".join(f"{d['name']}={d['summary']}" for d in describe("predictor"))),
    )
    ap.add_argument("--horizon", type=float, default=None, help="预测未来多少秒")
    ap.add_argument("--gravity-fit", action="store_true",
                    help="三维预测时使用已知重力约束")
    ap.add_argument("--mono-calib",
                    default=str(ROOT / "config" / "mono_calibration.json"),
                    help="单目内参标定文件（默认 config/mono_calibration.json）。"
                         "存在就自动使用，**含畸变系数**；没有则按 hfov 估算焦距")
    ap.add_argument("--no-mono-calib", action="store_true",
                    help="忽略标定文件、按 hfov 估算焦距（用来对比标定前后的差别）")
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


__all__ = [
    "ROOT",
    "LOG_DIR",
    "add_mono_pipeline_args",
    "add_stereo_source_args",
    "open_pair_source",
    "setup_logging",
    "log_environment",
]
