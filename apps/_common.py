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

from sphere_tracker.paths import LOGS_ROOT, new_log_path
from sphere_tracker.video_profiles import DEFAULT_FPS, fps_type

import argparse
import json
import logging
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

# apps/_common.py 的上一级就是项目根（apps/ 在根目录下）。
ROOT = Path(__file__).resolve().parent.parent
#: 旧名保留：日志根已上移到 ``<项目根>/logs``（见 `sphere_tracker.paths.LOGS_ROOT`）。
#: 不再有 ``outputs/logs`` 这个写入位置，`LOG_DIR` 现在指向新的根。
LOG_DIR = LOGS_ROOT

_LOG_FORMAT = "%(asctime)s %(levelname)-5s %(message)s"
_LOG_DATEFMT = "%H:%M:%S"

#: 日志 tag → 分区目录（`logs/<分区>/<年月>/`）。
#:
#: tag 是"哪段代码在写"，分区是"排障时该去哪个抽屉找"——两者不总同名：
#: `capture_board` 采的是标定板图，归 `calib` 而不是 `capture`。
#:
#: **新增带日志的入口时在这里登记一行**；漏登记不会丢日志（落到 `misc`），
#: 但会落到"其他"里，跟一次性脚本混在一起，下次排障就得先猜。
LOG_TAGS = {
    "calibrate": "calib",
    "calibrate_mono": "calib",
    "capture_board": "calib",
    "record": "capture",
    "realtime": "realtime",
    # 一键环境安装器是**零依赖独立入口**（跑它时可能还没装 cv2/numpy），
    # 不走 setup_logger，自己拼路径；来源仍在这里登记，排障时不用猜。
    # 形状由 selftest_launcher_params 与 paths.log_dir("misc") 对账。
    "setup_env": "misc",
}


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
    category: str | None = None,
    log_dir: Path | None = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> Path:
    """把日志**同时**写到终端和 ``logs/<分区>/<年月>/<tag>_<时间戳>.log``。

    返回值就是这个文件路径，各入口会把它打印出来（"本次日志：..."）——
    用户报问题时第一件事就是贴那行路径，所以路径必须真实、可点开。

    为什么要落盘：本项目最贵的一类失败是**静默失败** —— 标定收了 8 对就不动了，
    进程一退，拒绝原因的计数就没了，只能靠人在窗口前反复试。落盘之后，
    "为什么没收满"变成一个可以回看、可以贴给别人、可以跨次比对的文件。

    为什么分区（2026-11-04）：原先 9196 个日志平铺在 ``outputs/logs/`` 一层，
    找"上次标定为什么失败"要在一屏几百行里按文件名猜。现在第一层是来源
    （`calib` / `capture` / `realtime` / `tasks`），第二层是时间片
    （任务按日、其余按月，见 `paths._LOG_GRANULARITY`），
    打开目录就能定位到"这个月、这个程序"。

    ``category`` 不给时按 `LOG_TAGS` 从 ``tag`` 推；两者都没有就落 `misc`。
    ``log_dir`` 是给测试用的显式覆盖（临时目录），优先级最高。

    分工（照 STC 的"库不配日志、app 才配"）：
      - **库层**（``calib/`` ``stereo/`` …）只 ``getLogger(__name__)`` 并写记录，
        不装 handler、不决定写哪儿；
      - **app 层**（本函数）负责装 handler、定路径、定级别。
    这样库可以被测试直接 import 而不产生日志文件。

    级别分工：文件收 ``DEBUG``（每帧判定都留），终端只显示 ``INFO`` ——
    一帧一行打到终端会把有用信息淹掉。
    """
    if log_dir is not None:
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    else:
        path = new_log_path(category or LOG_TAGS.get(tag, "misc"), tag)

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
    width: int = 0,
    height: int = 0,
    fps: float = DEFAULT_FPS,
) -> argparse.ArgumentParser:
    """补上双目取帧参数 —— 字段名与 ``stereo.from_args()`` 读的一一对应。

    参数名/默认值集中在这里，三个双目 app 就永远同步，不会出现
    "这个脚本能用、那个不能"。

    ⚠️ **2026-10-31：``--width/--height`` 默认由 ``video_profiles.DEFAULT``
    改为 ``0``（= 自适应）。** 原因（planbook
    ``2026-10-31_双目标定取帧失败与分辨率档.md`` §8.3 S2）：本机 aoni
    **不支持 2560x1440**，请求它被**静默降级** 1920x1080 —— 写死一个"名义
    默认档"既拿不到、又掩盖了相机实际能力。改成 ``0`` 后由
    ``stereo.CameraPairReader`` **逐档协商**（清晰度优先 + 帧率下限 + 两路一致），
    不再被一个固定值限制死。**显式给数则跳过协商，行为与以前一致。**

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
                    help="相机采集宽度（两路一致）。**默认 0 = 自适应**"
                         "（逐档探测，清晰度优先 + 帧率下限）；给具体值则按该档打开")
    ap.add_argument("--height", type=int, default=height,
                    help="相机采集高度（两路一致）。**默认 0 = 自适应**（说明见 --width）")
    ap.add_argument("--fps", type=fps_type, default=fps,
                    help=f"相机采集帧率（默认 {fps}，可填小数，如 29.97）。"
                         "1440p 全尺寸写盘约 24fps；"
                         "FOV 对齐后写盘 1718x972 时 30fps 实测稳定")
    ap.add_argument(
        "--capture-api", choices=("auto", "mf", "opencv"), default="opencv",
        help="相机采集后端：opencv=默认，原有编号路径；"
             "mf=按设备 symlink 打开（必须绑定角色）；"
             "auto=角色有 identity/port 时先 MF、失败回退 OpenCV。"
             "显式数字编号只能用 opencv。")
    return ap


def calib_capture_size(calib_base) -> tuple[int, int] | None:
    """读标定 JSON 里记的**采集档**（``capture_size``，形如 ``[宽, 高]``）。

    为什么要它：标定是在某个**采集分辨率**下做的（FOV 裁剪之前）。录制 / 复检 /
    前置检查要复现**同一档**，否则写盘尺寸与标定 ``image_size`` 对不上
    （``record_stereo`` 的「实际 == 请求」校验会直接失败），FOV 裁剪也可能越界。

    返回 ``None`` 表示**没有这条记录**（旧标定）—— 调用方必须**明确说清**、
    退回默认档，不许静默假定。文件缺失 / 损坏 / 字段非法同样返回 ``None``。
    """
    p = Path(calib_base)
    if p.suffix != ".json":
        p = p.with_suffix(".json")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    cs = data.get("capture_size") if isinstance(data, dict) else None
    if isinstance(cs, (list, tuple)) and len(cs) == 2:
        try:
            w, h = int(cs[0]), int(cs[1])
        except (TypeError, ValueError):
            return None
        if w > 0 and h > 0:
            return (w, h)
    return None


def capture_size_from_profile(profile: object | None = None) -> tuple[int, int, str]:
    """采集档 ``(宽, 高, 来源说明)``：**装置档案优先**，没填就用 1080 安全档。

    采集档是**装置档案的属性**（``RigProfile.width/height``），不是标定文件
    ``calib/xxx.json`` 里 ``capture_size`` 的属性。以前从标定读：档案配了
    ``calibration_path`` 但还没标定时，标定 json 尚不存在 →
    ``calib_capture_size`` 返回 ``None`` → 调用方回落到
    ``video_profiles.DEFAULT`` = 2560x1440。而本机实测**双路同时开 2560 会让
    一路 MSMF 会话持久失效、必须物理拔插**（见 ``apps/tools/cameras.py``
    doctor 段与 ``apps/_gui_stereo.py::_capture_chain_frames`` 的注释），
    1920x1080 才稳。于是"从标定读采集档"会把一个未标定的新装置直接送进
    这个硬件坑。

    所以这里**不读标定**：档案填了宽高就用档案值；没填（或没有激活档案、
    档案读不了）就显式给 1080 安全档，并在返回的说明里写清来源 ——
    不许谎称"标定记录的采集档 / 与录制同档 / 相机最高档"。

    安全档取 ``sphere_tracker.video_profiles`` 的 1080 档对象：1920/1080
    这些数字的**唯一来源**在那个模块，不在这里写第二份字面量。
    """
    from sphere_tracker import video_profiles

    w = getattr(profile, "width", None)
    h = getattr(profile, "height", None)
    if isinstance(w, int) and w > 0 and isinstance(h, int) and h > 0:
        return w, h, f"装置档案采集档 {w}x{h}"
    safe = video_profiles.parse("1920x1080")
    return safe.width, safe.height, (
        f"装置档案未填采集档，用安全档 {safe.label}"
        f"（双路同时开 {video_profiles.DEFAULT.label} 会让一路 MSMF 会话"
        "持久失效、需物理拔插）")


def current_calibration(*, need_existing: bool = True) -> tuple[Path, str]:
    """当前装置档案该用的**标定基名**（无扩展名）+ 问题说明（空串 = 可以直接用）。

    2026-11-10 用户裁决 B5：CLI / GUI / 工具链**一律按当前运行档案判定**，
    不再出现"GUI 按档案、CLI 按顶层"两套口径。以前各调用点自己拼
    ``ROOT / "config" / "stereo_calibration"``，后果是：标定已经写进
    ``config/calib/<档案名>.npz``，复检按钮却去顶层找 → 报"还没有标定结果"；
    更糟的是录制前的模式校验用顶层几何去判"模式选错"。

    ``need_existing`` 分两类（语义与 ``stereo.profiles.resolve_calibration_base``
    的 ``strict`` 一致，这里只是把"抛异常"翻译成"人话原因"，免得每个调用点
    各写一份 try/except 与提示文案）：

    - ``True``（复检 / 录制前校验 / 前置检查）：**必须有现存标定**。
      "档案指定了但文件不在"与"档案没指定、顶层也没有"都返回非空说明。
    - ``False``（决定新标定往哪写）：允许文件还不存在。

    为什么在函数内 import：``apps/_common.py`` 被 GUI/CLI 大量 import，
    顶层 import ``stereo.profiles`` 会把 ``stereo/__init__`` 的 cv2 + scipy
    拖进冷启动（2026-11-10 已经踩过一次：``import apps.data_manager``
    从 0.06s 涨到 1.56s）。
    """
    from stereo.profiles import (
        ProfileError,
        calibration_files_exist,
        resolve_calibration_base,
    )

    if not need_existing:
        return resolve_calibration_base(strict=False), ""
    try:
        base = resolve_calibration_base(strict=True)
    except ProfileError as exc:
        # 档案配了标定路径但文件不在。resolve 的报错已经写清是哪套装置、
        # 期望哪个文件、为什么拒绝回落顶层 —— 原样往上给，不在这里改写第二份。
        # strict=False 不会再抛，拿到的是档案自己的 cand（"该写到哪"也有用）。
        return resolve_calibration_base(strict=False), str(exc)
    if calibration_files_exist(base):
        return base, ""
    # 走到这里只剩一种情况：档案没配路径 → 回落顶层 → 顶层也没有标定。
    return base, (
        f"找不到标定文件：{base}（.npz / .json 都不在）—— "
        "请先给当前装置做一次双目标定。")


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
        "--preset", default=None, choices=ColorConfig.preset_choices(),
        help="颜色取色方式：auto=按画面自动估主色（不需要知道球色），其余是固定预设；"
             "不传则用配置里的颜色。可选：%s" % (
                 " / ".join(ColorConfig.preset_choices())),
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


# --------------------------------------------------------------------------- #
# 备份 + 原子写：AGENTS.md §4 第 7 条「备份文件命名」的全仓库唯一实现
# --------------------------------------------------------------------------- #
#: 默认保留份数（集合配置）。会话元数据的回滚点是**显式策略差异**：写时不裁剪，
#: 由 `apps/_junk.py` 在清理阶段按 SESSION_KEEP_BACKUPS 统一保留 —— 不抹平。
DEFAULT_KEEP_BACKUPS = 5
SESSION_KEEP_BACKUPS = 3

#: 格式串只在这里出现一次；四处调用方不再各存一份。
_STAMP_FMT = "%Y%m%d-%H%M%S"
_BACKUP_FMT = "{name}.bak-{stamp}"
_BACKUP_COLLISION_FMT = "{name}.bak-{stamp}-{n}"
_TMP_FMT = "{name}.tmp-{pid}"

#: 从备份名解析 ``(时间戳, 撞车序号)``；基础名没有序号（按 0 算）。
_BACKUP_NAME_RE = re.compile(r"^.+\.bak-(\d{8}-\d{6})(?:-(\d+))?$")


def _backup_sort_key(path: str | Path) -> tuple[str, int]:
    """备份的**真实创建顺序**键：``(stamp, n)``，``n`` 按数值比、缺省 0。

    基础名 = n=0、``-2`` = n=2、``-10`` = n=10 —— 只有按数值才不会把 ``-10``
    排到 ``-2`` 前面（文件名字典序会）。``stamp`` 是零填充定长（``YYYYmmdd-HHMMSS``），
    字典序天然等于时间序。

    解析不出来的名字（历史手工备份之类）排最后、当作最新：宁可多留一份，
    也别把看不懂的东西误删。
    """
    m = _BACKUP_NAME_RE.match(Path(path).name)
    if m is None:
        return ("\uffff", 0)
    return (m.group(1), int(m.group(2) or 0))


def backup_glob(path: str | Path) -> str:
    """``<原名>.bak-*``：查找 / 清理备份时用的 glob，与写备份同源。"""
    return f"{Path(path).name}.bak-*"


def tmp_name(path: str | Path) -> str:
    """``<原名>.tmp-<pid>``：同目录临时文件名，写完 `os.replace` 原子替换。"""
    return _TMP_FMT.format(name=Path(path).name, pid=os.getpid())


def atomic_write_with_backup(path: str | Path, data, *,
                             keep_backups: int | None = DEFAULT_KEEP_BACKUPS) -> Path | None:
    """备份 → 同目录临时文件 → ``os.replace``；返回备份路径（没备份则 None）。

    这是 ``AGENTS.md`` §4 第 7 条的**唯一实现**：格式串、同秒撞车、保留份数、
    临时文件 + ``os.replace`` 都只写在这里。四处调用方
    （``_session_index.save_collections`` / ``data_manager.update_session_meta`` /
    ``_stereo_sessions._rewrite_json`` / ``_junk`` 的清理策略）不再各留一份格式串。

    ``data`` 是 JSON 可序列化对象，按四处原先一致的写法落 UTF-8 文本：
    ``json.dump(..., ensure_ascii=False, indent=2, allow_nan=False)`` + 末尾换行 +
    ``fsync``。备份是**原文件的逐字节副本**（``shutil.copy2``），不重新序列化 ——
    改坏了能按原样退回。

    ``keep_backups``：``>0`` 时只留**真实创建顺序**里最近的 N 份；``None`` 表示
    **写时不裁剪**（``session.json`` 的回滚点由 ``apps/_junk.py`` 在清理阶段统一裁剪
    —— 写时立刻删会改变既有行为）。

    排序为什么不能按文件名字典序（2026-11-10 实测抓到的真 bug）
    ------------------------------------------------------------
    基础名 ``<name>.bak-<stamp>`` 是 ``<name>.bak-<stamp>-2`` 的前缀，字典序排最前。
    若按 ``q.name`` 排序再删 ``olds[:-N]``，**每次裁剪都先删基础名**；基础名一没，
    下一次 ``while bak.exists()`` 就判为不撞车、又用基础名新建 —— 于是「刚创建的
    备份被自己这次的裁剪立刻删掉」形成死循环，磁盘永远停在最早两次写的旧内容，
    ``AGENTS.md`` §4 第 7 条写的「保留最近 N 份」变成「保留最旧 N 份」。
    同源的第二个坑：``-10`` 字典序排在 ``-2`` 前面，同秒第 10 次写也会排错。

    所以这里用 ``_backup_sort_key`` 解析成 ``(stamp, n)``（``n`` 按数值、缺省 0），
    **并且显式保护刚创建的 backup 不参与删除** —— 两道独立护栏，排序键以后再被
    改坏也不会复发。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    backup: Path | None = None

    if p.is_file():
        stamp = datetime.now().strftime(_STAMP_FMT)
        bak = p.with_name(_BACKUP_FMT.format(name=p.name, stamp=stamp))
        n = 2
        while bak.exists():
            bak = p.with_name(
                _BACKUP_COLLISION_FMT.format(name=p.name, stamp=stamp, n=n))
            n += 1
        try:
            shutil.copy2(p, bak)
        except OSError as exc:
            raise OSError(f"备份失败，原文件未改动：{bak}（{exc}）") from exc
        backup = bak
        if keep_backups is not None and keep_backups > 0:
            # 按 (stamp, n) 数值序排，不按文件名字典序（原因见上面的 docstring）。
            olds = sorted(p.parent.glob(backup_glob(p)), key=_backup_sort_key)
            # 刚创建的备份**绝不参与删除**：哪怕排序键以后被改坏，这也是一道
            # 独立护栏 —— 否则新备份会被自己这次的裁剪立刻删掉（死循环）。
            removable = [q for q in olds if q != backup]
            excess = max(0, len(olds) - keep_backups)
            for old in removable[:excess]:
                try:
                    old.unlink()
                except OSError:
                    pass

    tmp = p.with_name(tmp_name(p))
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2, allow_nan=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)
    except (OSError, ValueError) as exc:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        suffix = f"；原文件还在，备份在 {backup}" if backup else ""
        raise type(exc)(f"写入 {p} 失败：{exc}{suffix}") from exc
    return backup


__all__ = [
    "ROOT",
    "LOG_DIR",
    "DEFAULT_KEEP_BACKUPS",
    "SESSION_KEEP_BACKUPS",
    "add_mono_pipeline_args",
    "add_stereo_source_args",
    "atomic_write_with_backup",
    "backup_glob",
    "calib_capture_size",
    "capture_size_from_profile",
    "open_pair_source",
    "setup_logging",
    "log_environment",
    "tmp_name",
]
