"""视频分辨率档位：项目里 1920/1080/720 这些数字的**唯一来源**。

## 为什么要这个模块

分辨率原本散落在 GUI 的 ``RESOLUTIONS``、实时入口的 1280x720、录制默认值、
检测显示宽度等至少四处。改一处忘一处不会报错，最后表现为
"标定 1080p、录制 4K、离线再缩放"这类**静默混用** —— 双目标定直接失效，
而画面上看不出异常（见 ``docs/决策记录.md`` 2026-10-23）。

把"有哪些档、默认哪档、实际回了哪档"收进一个纯逻辑模块后：

- GUI / 录制 / 标定 / 检测都 import 同一份 ``PROFILES``，不再各写一遍；
- 相机实际给的分辨率可能不在档位表里（例如 1024x768），
  ``get()`` 会如实构造一个 profile，而不是谎称它是某个已知档；
- 请求 4K、相机只给 1080p 时，``resolve()`` 明确返回 ``fallback=True``，
  让调用方提示用户，不静默改。

模块不 import cv2 / tkinter / 本项目任何模块，可在无相机、无显示器的环境里自测。

## 默认档

``DEFAULT = 2560x1440``，``RECORD_DEFAULT = 2560x1440``。

依据（2026-10-24 真机验证，取代 2026-10-23 的「默认 1080」）：

- 两台外接相机**无视 1920/1280 请求，恒给 2560x1440** —— 默认请求 1080 只会让
  ``record_stereo`` 的「实际分辨率 == 请求」校验直接失败（默认值根本跑不起来）；
- ``record_stereo`` 在 2560x1440@30 下的**主循环**实测 **24.07 fps**
  （242/300 帧/路，10 秒，HFYU，两路一致）；**24 fps 是稳定默认**，
  请求 30 会丢帧（裸采集能到 30 ≠ 主循环能到 30）；
- 长录与其他机型仍未验证。

1920x1080 / 1280x720 仍是**合法可选档**：旧 1080 标定、旧会话、离线数据都能照常读，
显式 1080 档也仍能解析（相机实际给不给 1080 由硬件决定，不由本模块改）。
4K 仍不默认 —— 带宽/存储都还没验证。以后要改默认值，必须同步 GUI/CLI/录制与测试。

## 超高档：8K / 5K（2026-11-18）

``PROFILES`` 顶部新增 7680x4320（8K）与 5120x2880（5K）两档，供 ≥500 fps 的
高帧率相机**显式选用**（本机没有那种相机，所以只做档位表与解析，不做采集后端）。
为「加档不影响现有挡位」，三件事刻意没跟着变：

- ``DEFAULT`` / ``RECORD_DEFAULT`` 仍指向 2560x1440 的**同一个对象**，且改成
  按对象引用而不是 ``PROFILES[1]``（下标会随加档漂移，2026-11-18 已踩过）；
- ``probe_order(None)``（自动协商）上限仍是 4K（``PROBE_CEILING``）：每多试一档
  就多一次相机打开尝试，个别后端单次 20 s+，从 8K 起试会拖慢所有现有相机；
- ``high_res`` 阈值仍是 2560 宽（5K / 8K 自然为 True）。

``FPS_TIERS`` / ``parse_fps`` 是**帧率建议档与解析的唯一产地**：``FPS_TIERS`` 只给
GUI/CLI 下拉做建议值，**不限制输入**；``parse_fps`` 保留小数（``29.97`` 不许被
截成 29），超范围返回 ``None``，由调用方决定回退还是报错。
"""

from __future__ import annotations

import sys


def _drop_script_dir_from_path() -> None:
    """直接运行本文件时，别让 ``sphere_tracker/types.py`` 遮蔽标准库 ``types``。

    根因：``python sphere_tracker/video_profiles.py`` 会把脚本所在目录
    ``sphere_tracker/`` 放到 ``sys.path[0]``；标准库 ``dataclasses`` / ``re``
    在导入时会找 ``types``，于是拿到项目里的 ``types.py``，报
    ``cannot import name 'GenericAlias'``。这是**本文件直接自测**才有的路径
    问题（正常 ``import sphere_tracker.video_profiles`` 时 sys.path 是项目根，
    不会遮蔽）。只在 sys.path[0] 恰好就是本文件目录时移除它，不碰其他情况。
    """
    if not sys.path:
        return
    first = sys.path[0].replace("\\", "/").rstrip("/").casefold()
    here = __file__.replace("\\", "/").rsplit("/", 1)[0].casefold()
    if first and first == here:
        del sys.path[0]


if __name__ == "__main__":
    # 只修"直接自测"这一条路径；正常 import 时不动 sys.path。
    _drop_script_dir_from_path()

import argparse  # noqa: E402  （必须放在清遮蔽之后：本文件直接自测时 sys.path[0]
                 # 是 sphere_tracker/，argparse 间接导入的模块同样可能撞上它）
import math  # noqa: E402
import re  # noqa: E402  （必须先清掉遮蔽，见上）
from dataclasses import dataclass  # noqa: E402


@dataclass(frozen=True)
class VideoProfile:
    """一个分辨率档位。"""

    width: int
    height: int

    @property
    def label(self) -> str:
        """``"WxH"`` 形式（小写 x，可直接喂给 ``parse``）。"""
        return f"{self.width}x{self.height}"

    @property
    def pixels(self) -> int:
        """总像素数（4K 是 1080p 的 4 倍，用来解释带宽/耗时）。"""
        return self.width * self.height

    @property
    def high_res(self) -> bool:
        """是否高分辨率档。阈值 2560 宽：1440p / 4K 为 True，1080p 为 False。"""
        return self.width >= 2560


# 每档先各自命名，再组表：默认值**按对象引用**取，不再按下标取。
# ⚠️ 2026-11-18：在顶部加 8K/5K 时，``PROFILES[1]`` 从 1440 静默漂成了 5K ——
# 下标会随加档漂移，已踩过；以后加档只改这张表，别碰下面的对象引用。
P8K: VideoProfile = VideoProfile(7680, 4320)     # 8K UHD：显式可选，不进自动协商
P5K: VideoProfile = VideoProfile(5120, 2880)     # 5K：显式可选，不进自动协商
P4K: VideoProfile = VideoProfile(3840, 2160)     # 4K：可选、不默认；自动协商的上限
P1440: VideoProfile = VideoProfile(2560, 1440)   # 默认（真机恒给这一档；主循环 24fps 稳定）
P1080: VideoProfile = VideoProfile(1920, 1080)   # 旧默认；仍合法、旧标定/旧会话可读
P720: VideoProfile = VideoProfile(1280, 720)
P540: VideoProfile = VideoProfile(960, 540)
P480: VideoProfile = VideoProfile(640, 480)

# 顺序即像素降序：GUI 下拉 / CLI choices 的顺序就是它。
PROFILES: tuple[VideoProfile, ...] = (
    P8K, P5K, P4K, P1440, P1080, P720, P540, P480,
)

# 默认档直接引用上面那个对象，避免出现"两个 2560x1440"。
DEFAULT: VideoProfile = P1440

# 默认帧率。**全项目只此一处**（2026-10-26）：此前 GUI 的 CALIB_FPS_DEFAULT、
# `apps/_common.py` 的 `fps: int = 24`、`apps/record_stereo.py` 的 `--fps default=24`、
# 命令行入口的 `"60"` 各写一遍 —— 四个数字里已经有一个是错的（60），
# 而用户看到的是"同一个程序，两个入口默认帧率不同"。
DEFAULT_FPS = 24

# 帧率建议档：**只作 GUI/CLI 下拉的建议值，不限制输入**。
# 高帧率相机（≥500 fps）的档位常年是这些整数；需要 29.97 这类值就直接填，
# 由 `parse_fps` 负责校验（它不会把小数截掉）。
FPS_TIERS: tuple[int, ...] = (24, 30, 60, 120, 240, 500, 1000)

# 帧率的合法边界（供调用方复用，别在别处再写一遍）。
FPS_MAX = 1000.0
FPS_MIN = 0.1

# 录制默认档：**独立于 DEFAULT**。当前两者都指向 1440 的**同一对象**
# （对象引用，不是下标；下标会随加档漂移），名字保留独立，免得以后改 DEFAULT
# 时录制静默跟过去（录制还受吞吐/存储约束）。
RECORD_DEFAULT: VideoProfile = P1440

# 自动协商（`probe_order(None)`）的上限：4K。
# 5K/8K 只在用户显式选时才会进候选 —— 自动协商每多试一档就多一次相机打开
# 尝试（个别后端单次 20 s+），从 8K 起试会拖慢所有现有相机。
PROBE_CEILING: VideoProfile = P4K

# 接受 "1920x1080" / "1920 X 1080" / " 1920 x 1080 "。
# 文档里常写 "1920×1080"，顺手也认这个乘号，避免用户从手册复制后解析失败。
_PATTERN = re.compile(r"^\s*(\d+)\s*[xX×]\s*(\d+)\s*$")


def parse(text: str, default: VideoProfile = DEFAULT) -> VideoProfile:
    """解析 ``"WxH"``（x 不分大小写，允许空格）；非法输入返回 ``default``。

    **不认识的 WxH 但语法合法时照样返回它**（如 ``"1024x768"``）：
    相机实测可能出现非标准档，parse 不该把它误判成非法输入。
    ``"0x0"``、负数、缺一维、多个 x 等视为非法。
    """
    if not isinstance(text, str):
        return default
    match = _PATTERN.match(text)
    if match is None:
        return default
    width, height = int(match.group(1)), int(match.group(2))
    if width <= 0 or height <= 0:
        return default
    return get(width, height)


def parse_fps(text: str | int | float | None) -> float | None:
    """解析帧率文本，**保留小数**；非法 / 超范围返回 ``None``。

    合法：``"24"`` / ``"29.97"`` / ``" 500 "`` / ``"500.0"``（也接受 int/float）。
    非法：非数字、``<= 0``、``> FPS_MAX``（1000）、``nan`` / ``inf``、``None``。

    **不做静默截断**：``"29.97"`` 必须回 ``29.97``，不能像
    ``stereo/pair_source.py`` 曾用 ``int(fps)`` 那样被截成 29 —— 29.97 与 30
    差的那 0.03 会以"录 1 分钟少 1.8 帧"的形式累积。合法区间取 ``(0, FPS_MAX]``，
    ``FPS_MIN`` 是留给调用方的建议下限，本函数不拿它判非法（避免把 0.5 fps 这类
    合法慢速采集堵死）。
    """
    if isinstance(text, bool):
        return None
    if isinstance(text, (int, float)):
        value = float(text)
    elif isinstance(text, str):
        try:
            value = float(text.strip())
        except ValueError:
            return None
    else:
        return None
    if not math.isfinite(value):
        return None
    if value <= 0 or value > FPS_MAX:
        return None
    return value


def fps_type(text: str) -> float:
    """``argparse`` 的 ``type=``：合法返回 float，非法抛 ``ArgumentTypeError``。

    与 ``parse_fps`` 的分工：``parse_fps`` 是纯解析（非法返回 ``None``，适合代码里
    判空），本函数只负责把"非法"翻译成 argparse 认识的异常，让命令行报用法错误
    而不是抛裸 ``ValueError`` 或静默取默认值。
    """
    value = parse_fps(text)
    if value is None:
        raise argparse.ArgumentTypeError(
            f"帧率需为 0 < fps <= {FPS_MAX:g} 的数字，收到 {text!r}")
    return value


def options() -> list[str]:
    """GUI 下拉 / CLI choices 用的规范值列表（顺序即 ``PROFILES`` 顺序）。"""
    return [profile.label for profile in PROFILES]


def labels() -> list[str]:
    """GUI 显示文本列表。

    目前与 ``options()`` 相同，因为 ``"WxH"`` 本身既可读、又能被 ``parse()``
    原样解析。保留单独的函数名，是为了以后要加 "4K" 之类的展示后缀时，
    只改这里、不动取值；**展示文本不要变成 parse 不了的另一套格式**。
    """
    return [profile.label for profile in PROFILES]


def get(width: int, height: int) -> VideoProfile:
    """按宽高取 profile：命中档位表就返回表里的对象，否则如实构造一个。

    相机的实际分辨率可能是 1024x768 这类非标准档；返回构造的 profile
    比偷偷映射成 1280x720 更诚实（后续校验/提示才能说清实际值）。
    """
    w, h = int(width), int(height)
    for profile in PROFILES:
        if profile.width == w and profile.height == h:
            return profile
    return VideoProfile(w, h)


#: 「原生档」哨兵：``(0, 0)`` 表示"不指定宽高、用相机的原生档"。
#:
#: ⚠️ **语义未验证**：MSMF 下 ``cap.set(CAP_PROP_FRAME_WIDTH, 0)`` 到底怎么解释，
#: 本项目从未证实（planbook ``2026-10-31_双目标定取帧失败与分辨率档.md`` §3 的 P5）。
#: 所以它只作**最后兜底**，且**不进**模式搜索（``sources.CANDIDATE_MODES`` 会剔除它）。
NATIVE: tuple[int, int] = (0, 0)


def probe_order(start: VideoProfile | None = None,
                ceiling: VideoProfile = PROBE_CEILING) -> list[tuple[int, int]]:
    """自适应候选分辨率，**按像素降序**，末尾恒为原生档 ``NATIVE``。

    - ``start=None``：**自动协商**，只试 ``ceiling`` 及以下的档（默认
      ``PROBE_CEILING`` = 4K）一路降到原生档。5K/8K **刻意不进自动协商**：
      每多试一档就多一次相机打开尝试（个别后端单次 20 s+），从 8K 起试会
      拖慢所有现有相机；用户显式选 8K/5K 时才用（见下一条）；
    - ``start=某档``：**从该档开始往下**（显式档**不越级往上**，保持可控），
      末尾同样跟 ``NATIVE``。``start`` 是 8K/5K 时以它自身开头，**不受**
      ``ceiling`` 限制。``start`` **不在** ``PROFILES`` 里也照样含它自身
      （如 ``1024x768``），再跟 ``PROFILES`` 中像素更小的档。

    ``ceiling`` 只约束 ``start=None`` 这条自动协商路径，供调用方临时抬/降上限；
    日常调用不要传，保持"自动协商上限 4K"这一条不变。

    这是"分辨率降档梯子"的**唯一来源**：``sources.CameraSource`` 的
    ``res_candidates`` 与 ``CANDIDATE_MODES`` 都从这里取，不再各写一份手写清单。
    此前 ``sources.py`` 里那份硬编码梯子 ``[(申请),(1280,720),(640,480),(原生)]``
    **漏了 1920x1080** —— 本机唯一稳定可用的档 —— 却一步跳到了已知无效的 640x480
    （planbook 2026-10-31 §2 / §8.1）。

    保证：无重复项；``start`` 恰好是 ``PROFILES`` 某档时区间**含** ``start`` 自身。
    """
    order: list[tuple[int, int]] = []

    def _add(w: int, h: int) -> None:
        if (w, h) not in order:
            order.append((w, h))

    if start is None:
        for profile in PROFILES:
            if profile.pixels <= ceiling.pixels:
                _add(profile.width, profile.height)
    else:
        _add(start.width, start.height)
        for profile in PROFILES:
            if profile.pixels < start.pixels:
                _add(profile.width, profile.height)
    _add(NATIVE[0], NATIVE[1])
    return order


def resolve(requested: VideoProfile, actual_w: int, actual_h: int,
            ) -> tuple[VideoProfile, bool]:
    """把"请求的档"与"相机实际给的宽高"对齐，返回 ``(实际档, 是否回退)``。

    - 实际值任一边 ``<= 0``（相机没给出有效尺寸）：返回 ``(requested, False)``；
    - 实际值 == 请求值：返回 ``(requested, False)``；
    - 否则返回 ``(get(actual_w, actual_h), True)``，调用方必须把
      "请求 4K、实际 1080p" 明确告诉用户，不能静默继续。
    """
    aw, ah = int(actual_w), int(actual_h)
    if aw <= 0 or ah <= 0:
        return requested, False
    if aw == requested.width and ah == requested.height:
        return requested, False
    return get(aw, ah), True


def validate_pair(left: VideoProfile, right: VideoProfile) -> None:
    """两路分辨率必须一致；不一致抛 ``ValueError``（双目硬要求）。"""
    if left.width == right.width and left.height == right.height:
        return
    raise ValueError(
        f"左右分辨率不一致：{left.label} vs {right.label}；"
        "双目标定、录制、离线分析必须使用同一分辨率"
    )


def detection_width(width: int, target: int = 1280) -> int:
    """检测输入宽度：高分辨率降采样到 ``target``，但**绝不放大**。

    返回 ``min(width, target)``。例如 3840 -> 1280、1920 -> 1280、
    960 -> 960（不放大）。``target <= 0`` 是配置错误，直接抛 ``ValueError``。
    """
    tgt = int(target)
    if tgt <= 0:
        raise ValueError(f"detection_width 的 target 必须 > 0，实际 {target!r}")
    return min(int(width), tgt)


# ---------------------------------------------------------------------------
# 自测：无相机、无 GUI、无 pytest，直接跑本文件即可
# ---------------------------------------------------------------------------
def _expect_value_error(fn, label: str) -> None:
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(f"{label}: 应当抛 ValueError，实际没有抛")


def _selftest() -> int:
    p8k, p5k, p4k, p1440, p1080, p720, p540, p480 = PROFILES
    expected_labels = [
        "7680x4320", "5120x2880", "3840x2160", "2560x1440",
        "1920x1080", "1280x720", "960x540", "640x480",
    ]

    # ---- 档位 / properties ----
    assert len(PROFILES) == 8
    assert DEFAULT == VideoProfile(2560, 1440)
    # 加档不漂移默认值：两个名字仍是表里同一个对象，label 仍是 2560x1440。
    assert DEFAULT is p1440 and RECORD_DEFAULT is p1440
    assert DEFAULT is RECORD_DEFAULT and DEFAULT.label == "2560x1440"
    assert p1080 in PROFILES and parse("1920x1080") is p1080   # 旧 1080 档仍可读
    assert p8k.label == "7680x4320" and p8k.pixels == 7680 * 4320
    assert p5k.label == "5120x2880" and p5k.pixels == 5120 * 2880
    assert p8k.pixels > p5k.pixels > p4k.pixels > p1440.pixels
    assert p4k.label == "3840x2160" and p4k.pixels == 3840 * 2160
    assert p8k.high_res and p5k.high_res and p4k.high_res and p1440.high_res
    assert not p1080.high_res and not p720.high_res
    assert p1080.pixels == 1920 * 1080
    print("PASS profiles: 8 档有序（8K/5K 在顶部），DEFAULT=RECORD_DEFAULT=2560x1440，旧 1080 仍可读")

    # ---- options / labels ----
    assert options() == expected_labels
    assert labels() == expected_labels
    assert isinstance(options(), list) and isinstance(labels(), list)
    print("PASS options/labels: 顺序与 PROFILES 一致")

    # ---- parse：合法 / 大小写 / 空格 / 非标准档 ----
    assert parse("3840x2160") is p4k
    assert parse("2560x1440") is p1440
    assert parse(" 1280 X 720 ") is p720
    assert parse(" 960 x 540 ") is p540
    assert parse("1024x768") == VideoProfile(1024, 768)
    assert parse("1024x768") not in PROFILES
    assert parse("1920×1080") is p1080          # 文档里的乘号也认
    print("PASS parse: 合法输入、大小写、空格、非标准档")

    # ---- parse：非法 -> default（含自定义 default） ----
    for bad in ("", "   ", "abc", "1920", "x1080", "1920x", "1920x1080x2",
                "1920*1080", "0x0", "-1920x1080", None):
        assert parse(bad) is DEFAULT, f"parse({bad!r}) 应回退 DEFAULT"
    assert parse("bad", default=p720) is p720
    print("PASS parse invalid: 非法输入全部回退默认档")

    # ---- resolve：4K / 默认 1440 掉到 1080p 必须标记回退；默认 1440 精确命中 ----
    got, fallback = resolve(p4k, 2560, 1440)
    assert got is p1440 and fallback is True
    got, fallback = resolve(p4k, 1920, 1080)
    assert got is p1080 and fallback is True
    got, fallback = resolve(DEFAULT, 2560, 1440)
    assert got is p1440 and fallback is False
    # 旧 1080 标定/会话：显式 1080 仍精确命中，不会被悄悄当成默认档
    got, fallback = resolve(p1080, 1920, 1080)
    assert got is p1080 and fallback is False
    got, fallback = resolve(p4k, 3840, 2160)
    assert got is p4k and fallback is False
    got, fallback = resolve(p1080, 1024, 768)
    assert (got.width, got.height) == (1024, 768) and fallback is True
    print("PASS resolve: 4K->1080/1440 fallback=True，默认 1440 与旧 1080 均精确命中")

    # ---- resolve：实际尺寸为 0 / 负数 -> requested, False ----
    for aw, ah in ((0, 0), (0, 1080), (1920, 0), (-1, 1080)):
        got, fallback = resolve(p4k, aw, ah)
        assert got is p4k and fallback is False, (aw, ah, got, fallback)
    print("PASS resolve actual=0: 保留请求档且不误报回退")

    # ---- validate_pair ----
    assert validate_pair(p1080, VideoProfile(1920, 1080)) is None
    _expect_value_error(lambda: validate_pair(p4k, p1080), "validate_pair mismatch")
    print("PASS validate_pair: 同分辨率通过，不一致抛 ValueError")

    # ---- detection_width ----
    cases = (
        (3840, 1280, 1280), (2560, 1280, 1280), (1920, 1280, 1280),
        (1280, 1280, 1280), (960, 1280, 960), (640, 1280, 640),
        (3840, 960, 960), (3840, 2560, 2560), (800, 1280, 800),
        (3840, 5000, 3840),                     # 绝不放大
    )
    for width, target, expected in cases:
        actual = detection_width(width, target=target)
        assert actual == expected, (width, target, actual, expected)
        assert type(actual) is int
    _expect_value_error(lambda: detection_width(3840, target=0), "target=0")
    _expect_value_error(lambda: detection_width(3840, target=-1), "target=-1")
    print("PASS detection_width: 只缩不放，target<=0 抛 ValueError")

    # ---- 帧率档位 / parse_fps（唯一产地；不许静默截断小数） ----
    assert FPS_TIERS == (24, 30, 60, 120, 240, 500, 1000)
    assert DEFAULT_FPS == 24 and FPS_MIN == 0.1 and FPS_MAX == 1000.0
    assert parse_fps("24") == 24.0
    assert parse_fps("29.97") == 29.97            # 关键：不被 int() 截成 29
    assert parse_fps(" 500 ") == 500.0
    assert parse_fps("500.0") == 500.0
    assert parse_fps("1000") == 1000.0
    assert isinstance(parse_fps("24"), float)
    for bad in ("0", "-1", "abc", "1500", "", "nan", "inf", None):
        assert parse_fps(bad) is None, f"parse_fps({bad!r}) 应返回 None"
    print("PASS parse_fps/FPS_TIERS: 29.97 不截断，0/-1/abc/1500 返回 None")

    # ---- fps_type：argparse 的 type= —— 合法给 float，非法变用法错误（不静默取默认） ----
    assert fps_type("29.97") == 29.97            # 关键：29.97 而不是 29
    assert fps_type("500") == 500.0
    assert isinstance(fps_type("24"), float)
    for bad in ("abc", "1001", "0", "-1", ""):
        try:
            fps_type(bad)
        except argparse.ArgumentTypeError as exc:
            assert bad in str(exc), (bad, str(exc))
        else:
            raise AssertionError(f"fps_type({bad!r}) 应当抛 ArgumentTypeError")
    print("PASS fps_type: 29.97 保留小数；abc/1001/0/-1 抛 ArgumentTypeError 且带收到的原文")

    # ---- probe_order：自适应降档梯子（唯一来源；本机修复点是必须含 1080） ----
    assert NATIVE == (0, 0)
    # 自动协商（start=None）**不受 8K/5K 影响**：仍从 4K 起、仍只这 7 项。
    assert PROBE_CEILING is p4k
    assert probe_order(None) == [
        (3840, 2160), (2560, 1440), (1920, 1080),
        (1280, 720), (960, 540), (640, 480), NATIVE,
    ]
    assert (7680, 4320) not in probe_order(None)
    assert (5120, 2880) not in probe_order(None)
    # 显式选 8K：以它自身开头，照样往下含 4K / 1080p。
    order_8k = probe_order(parse("7680x4320"))
    assert parse("7680x4320") is p8k
    assert order_8k[0] == (7680, 4320)
    assert (3840, 2160) in order_8k and (1920, 1080) in order_8k
    assert order_8k == [
        (7680, 4320), (5120, 2880), (3840, 2160), (2560, 1440),
        (1920, 1080), (1280, 720), (960, 540), (640, 480), NATIVE,
    ]
    assert probe_order(p1080) == [
        (1920, 1080), (1280, 720), (960, 540), (640, 480), NATIVE,
    ]
    assert (1920, 1080) in probe_order(None)      # 修复点：旧梯子缺这一档
    assert (1920, 1080) in probe_order(p1080)
    # 显式档**不越级往上**：p1080 的结果里不含 1440 / 4K
    assert (2560, 1440) not in probe_order(p1080)
    assert (3840, 2160) not in probe_order(p1080)
    # 非标准显式档：含自身，再跟 PROFILES 中像素更小的档
    assert probe_order(parse("1024x768")) == [
        (1024, 768), (960, 540), (640, 480), NATIVE,
    ]
    # 「自动协商不含 8K/5K」与「显式 8K 以自身开头」互不影响
    assert order_8k[0] != probe_order(None)[0]
    # 无重复项 + 末尾恒为原生档
    for order in (probe_order(None), probe_order(p1080),
                  probe_order(parse("1024x768")), order_8k):
        assert len(order) == len(set(order)), order
        assert order[-1] == NATIVE
    print("PASS probe_order: 自动协商仍 4K 起 7 项（不含 5K/8K）；显式 8K 以自身开头；末尾原生档")

    print("PASS 全部自测通过（video_profiles）")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
