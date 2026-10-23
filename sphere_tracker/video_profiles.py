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

``DEFAULT = 1920x1080``。在真机两路 1440p/4K 稳定之前不把 4K 设为默认
（决策记录 2026-10-23 决策 3）。
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


PROFILES: tuple[VideoProfile, ...] = (
    VideoProfile(3840, 2160),   # 4K，可选、不默认
    VideoProfile(2560, 1440),   # 2.5K
    VideoProfile(1920, 1080),   # 默认
    VideoProfile(1280, 720),
    VideoProfile(960, 540),
    VideoProfile(640, 480),
)

# 默认档直接引用 PROFILES 里的同一个对象，避免出现"两个 1920x1080"。
DEFAULT: VideoProfile = PROFILES[2]

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
    p4k, p1440, p1080, p720, p540, p480 = PROFILES
    expected_labels = [
        "3840x2160", "2560x1440", "1920x1080",
        "1280x720", "960x540", "640x480",
    ]

    # ---- 档位 / properties ----
    assert DEFAULT == VideoProfile(1920, 1080)
    assert DEFAULT is p1080
    assert p4k.label == "3840x2160" and p4k.pixels == 3840 * 2160
    assert p4k.high_res and p1440.high_res
    assert not p1080.high_res and not p720.high_res
    assert p1080.pixels == 1920 * 1080
    print("PASS profiles: 6 档有序，DEFAULT=1920x1080，high_res 阈值正确")

    # ---- options / labels ----
    assert options() == expected_labels
    assert labels() == expected_labels
    assert isinstance(options(), list) and isinstance(labels(), list)
    print("PASS options/labels: 顺序与 PROFILES 一致")

    # ---- parse：合法 / 大小写 / 空格 / 非标准档 ----
    assert parse("3840x2160") is p4k
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

    # ---- resolve：4K -> 1080p 必须标记回退 ----
    got, fallback = resolve(p4k, 1920, 1080)
    assert got is DEFAULT and fallback is True
    got, fallback = resolve(p4k, 3840, 2160)
    assert got is p4k and fallback is False
    got, fallback = resolve(p1080, 1024, 768)
    assert (got.width, got.height) == (1024, 768) and fallback is True
    print("PASS resolve: 4K->1080p fallback=True，命中请求 fallback=False")

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

    print("PASS 全部自测通过（video_profiles）")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
