"""录制参数体检：这份标定 + 这个模式 + 这个尺寸，到底能不能录。

为什么要有这一层
----------------
2026-10-28 真机：界面里把「双目模式」从 ``auto`` 改成 ``angled``，分辨率仍是
2560x1440，点录制后得到::

    [FOV] 标定记录里带裁剪/缩放参数，但模式为 angled：忽略它。
    [错误] 标定内参基于 1718x972，当前录制请求 2560x1440；请用同一分辨率重标

链路本身没错 —— ``angled`` 忽略 FOV 是**有意的**（两路光轴近乎垂直，单应裁剪
没有物理意义）。错在**三个互相独立的控件**（双目模式 / 分辨率 / 磁盘上的标定）
被允许组合成一个从定义上就不成立的方案，而且判定发生在**相机已经打开、会话
目录已经建好之后**：用户付出的代价是一次白开的相机，加一段要读三遍才懂的技术
报错，而真正该做的事（把模式改回 auto，或者先去重标）一个字都没提。

判定只需要标定文件与两个界面参数，不需要相机。所以把它提成纯函数：

- :func:`plan_record` 回答"能不能录、为什么不能、该怎么办"；
- :func:`output_size` 回答"实际写盘多大"。

``apps/record_stereo.py`` 与本模块**共用**这两个结论，界面层只负责把
:meth:`RecordPlan.describe` 显示出来 —— 判据只有一份，不会出现"界面说能录、
后端说不能"。

三种成立组合（其余都是错的）
----------------------------
============================  ==================  ==================
模式                          相机采集            写盘
============================  ==================  ==================
parallel + 带 FOV 的标定      ``--width/--height``  ``fov.target_size``
parallel + 无 FOV 的标定      ``--width/--height``  同上（= 采集尺寸）
angled + angled 标定          ``--width/--height``  同上
============================  ==================  ==================

关键：**模式必须与标定一致**。``parallel`` 标定解出的 R/T 假设两路光轴近似平行；
把它套到 90° 装置上，就算尺寸凑对了，几何也是错的 —— 而且不会报错。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from stereo.fov_align import FovAlignment
from stereo.stereo_mode import resolve_auto

if TYPE_CHECKING:                                        # pragma: no cover
    from calib.params import StereoRig


def fov_of(rig: Any) -> FovAlignment | None:
    """标定 ``meta["fov"]`` -> :class:`FovAlignment`；缺失 / ``None`` 返回 ``None``。

    ``{}`` / ``[]`` / 非 mapping 是**损坏的记录**，由 ``from_dict`` 抛
    ``ValueError`` —— 调用方必须按损坏处理，不能当成"旧标定没有 FOV"静默放过：
    那样会按原生画面录制，尺度错了却不报错。
    """
    meta = getattr(rig, "meta", None)
    raw = meta.get("fov") if isinstance(meta, dict) else None
    if raw is None:
        return None
    return FovAlignment.from_dict(raw)                    # type: ignore[arg-type]


def output_size(mode: str, fov: FovAlignment | None,
                width: int, height: int) -> tuple[int, int]:
    """实际写盘尺寸（**唯一来源**，``record_stereo`` 与体检共用）。

    - ``angled``：两路各自的原始画面；``fov`` 即使存在也不看（显式忽略，
      而不是"碰巧没传"）；
    - ``parallel`` + 生效的 FOV：``fov.target_size``（两路共同、不放大）；
    - 其余：``width x height``。
    """
    if mode != "angled" and fov is not None and fov.active:
        return (int(fov.target_size[0]), int(fov.target_size[1]))
    return (int(width), int(height))


@dataclass(frozen=True)
class RecordPlan:
    """一次录制参数体检的结论。``ok=False`` 时**不该开相机**。"""

    ok: bool
    mode: str                       # 实际会用的模式（auto 已解析）
    calib_mode: str                 # 标定记录里的模式
    capture_size: tuple[int, int]   # 传给相机的 --width/--height
    save_size: tuple[int, int]      # 实际写盘（= 内参必须匹配的尺寸）
    calib_size: tuple[int, int]     # 标定 image_size
    headline: str                   # 一句话结论
    detail: tuple[str, ...] = ()    # 为什么
    fixes: tuple[str, ...] = ()     # 可执行的一步

    @property
    def mode_conflict(self) -> bool:
        """主矛盾是不是"模式选错了"（界面据此决定是否提供一键改 auto）。"""
        return self.mode != self.calib_mode

    def describe(self) -> str:
        """多行文本：结论 -> 原因 -> 出路。顺序固定，弹窗与日志共用。"""
        lines = [self.headline]
        lines += [f"  · {d}" for d in self.detail]
        if self.capture_size != self.save_size:
            lines.append(f"  · 相机按 {self.capture_size[0]}x{self.capture_size[1]} 采集，"
                         f"写盘 {self.save_size[0]}x{self.save_size[1]}")
        if self.fixes:
            lines.append("可以这样改：")
            lines += [f"  {i}) {f}" for i, f in enumerate(self.fixes, 1)]
        return "\n".join(lines)


def _fail(mode: str, calib_mode: str, capture: tuple[int, int],
          save: tuple[int, int], calib_size: tuple[int, int],
          headline: str, detail: list[str], fixes: list[str]) -> RecordPlan:
    return RecordPlan(False, mode, calib_mode, capture, save, calib_size,
                      headline, tuple(detail), tuple(fixes))


def plan_record(rig: "StereoRig | Any", *, mode_choice: str = "auto",
                width: int, height: int) -> RecordPlan:
    """体检一次录制参数。只读标定对象，**不碰相机、不写盘**。

    ``mode_choice`` 是界面/命令行的原始值（可能是 ``"auto"``）；
    ``width/height`` 是要传给相机的采集尺寸（``--width/--height``），
    不是写盘尺寸 —— 写盘由 :func:`output_size` 决定。
    """
    width, height = int(width), int(height)
    capture = (width, height)
    calib_size = tuple(int(v) for v in (getattr(rig, "image_size", ()) or ()))
    calib_mode = str(getattr(rig, "stereo_mode", "parallel") or "parallel").lower()
    mode = resolve_auto(mode_choice, calibration=getattr(rig, "meta", None))

    if len(calib_size) != 2 or min(calib_size) <= 0:
        return _fail(mode, calib_mode, capture, capture, calib_size,
                     "这份标定没有 image_size，判断不了能不能按当前参数录",
                     ["没有 image_size 就没法核对写盘尺寸 —— 录出来的米制尺度会是错的。"],
                     ["重新标定一次（标定会写入 image_size）"])

    try:
        fov = fov_of(rig)
    except ValueError as exc:
        return _fail(mode, calib_mode, capture, capture, calib_size,
                     "这份标定的 FOV 记录损坏，不能录",
                     [f"meta.fov 解析失败：{exc}",
                      "旧标定本来没有该字段；有值却解析不出来，说明记录坏了。"],
                     ["重新标定后再录"])

    save = output_size(mode, fov, width, height)

    # ① 模式必须与标定一致。这条**先于**尺寸判：尺寸凑对了但模式错，
    #    几何仍然是错的，而且不会报错 —— 那是更坏的情况。
    if mode != calib_mode:
        if calib_mode == "parallel":
            return _fail(
                mode, calib_mode, capture, save, calib_size,
                f"这份标定是 parallel（正对装置），不能按 angled（90°）录",
                [f"标定内参按 {calib_size[0]}x{calib_size[1]} 算出来"
                 "（parallel 路径，可能还带 FOV 裁剪）",
                 f"angled 会忽略 FOV、按 {capture[0]}x{capture[1]} 原样写盘，"
                 "内参与画面尺寸对不上",
                 "而且 parallel 标定解出的 R/T 只对「两路光轴近似平行」成立，"
                 "套到 90° 装置上几何是错的"],
                ["把「双目模式」改回 auto（= 按标定，会解析成 parallel），再点录制",
                 "真要用 90°：先按 90° 装置重标一次（标定会记 stereo_mode=angled），"
                 "再回来录"])
        return _fail(
            mode, calib_mode, capture, save, calib_size,
            f"这份标定是 angled（90° 装置），不能按 parallel 录",
            [f"标定内参按 {calib_size[0]}x{calib_size[1]} 算出来（90° 装置，不做 FOV 对齐）",
             "parallel 路径会去找 FOV 裁剪、并假设两路光轴近似平行 —— "
             "与这份标定不是一回事"],
            ["把「双目模式」改回 auto（= 按标定，会解析成 angled），再点录制"])

    # ② 写盘尺寸必须与内参一致（混用会让米制尺度系统性偏，而且不报错）
    if save != calib_size:
        fixes: list[str] = []
        if mode != "angled" and fov is not None and fov.active:
            need_w = fov.crop_rect[0] + fov.crop_rect[2]
            need_h = fov.crop_rect[1] + fov.crop_rect[3]
            fixes.append(f"把「分辨率」提到至少 {need_w}x{need_h}"
                         f"（写盘会自动回到标定的 {calib_size[0]}x{calib_size[1]}）")
        else:
            fixes.append(f"把「分辨率」改成 {calib_size[0]}x{calib_size[1]}"
                         "（= 标定内参，不要混用）")
        fixes.append("或者重新标定一次，让内参与你要用的分辨率一致")
        return _fail(mode, calib_mode, capture, save, calib_size,
                     "写盘尺寸与标定内参对不上，录出来尺度会错",
                     [f"标定内参基于 {calib_size[0]}x{calib_size[1]}",
                      f"当前参数会写 {save[0]}x{save[1]}"],
                     fixes)

    # ③ parallel + FOV：裁剪区必须落在本次采集帧内（否则裁剪会越界）
    if mode != "angled" and fov is not None and fov.active:
        cx, cy, cw, ch = fov.crop_rect
        if cx < 0 or cy < 0 or cx + cw > width or cy + ch > height:
            return _fail(mode, calib_mode, capture, save, calib_size,
                         f"当前采集尺寸装不下这份标定的裁剪区",
                         [f"标定要裁 {fov.crop_side} 路的 ({cx}, {cy}, {cw}, {ch})",
                          f"但采集帧只有 {width}x{height}"],
                         [f"把「分辨率」提到至少 {cx + cw}x{cy + ch}"])

    return RecordPlan(True, mode, calib_mode, capture, save, calib_size,
                      f"可以录：{mode} 模式，写盘 {save[0]}x{save[1]}"
                      f"（与标定内参一致）")
