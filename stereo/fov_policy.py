"""装置级 FOV 应用策略：把「是否应用到当前摄像头」收敛成唯一判定点。

## 概念（用户原话，别改写成别的）
「启用」= 把这份 FOV 参数应用到当前装置的两路相机画面上，是一个**装置级**
事实，不是「标定时用不用」「录制时用不用」两个流程开关。标定与录制只是它的
下游后果，都服从同一个 :func:`resolve_fov_apply`。

## 「开关」管的是装置档案里那份记录要不要成为权威
装置档案里的 ``fov`` 是量好的一份记录；``fov_apply`` 决定它要不要成为权威。
标定 JSON 若自己带着 FOV，说明那次标定的内参是在**裁剪帧**上算的 —— 录制必须
跟着裁才几何自洽。所以开关关闭、但标定自带 FOV 时，走 ``legacy_follow`` 分支：
**沿用标定自带的 FOV 裁剪（旧流程行为不变）**，只给一句醒目提示，建议把记录
读入档案并打开开关，而不是强制按原始画面写盘（那会让内参与画面口径错配）。

## 为什么必须有一个唯一判定点
FOV 改变的是**喂给标定的像素**。用「关 FOV 标出来的内参」去解「开 FOV 录出来
的画面」，会得到系统性错的深度，而且屏幕上完全看不出来 —— 这是本项目最贵的
那类静默错配。所以「启用了但标定不是在 FOV 对齐下做的」或「标定 fov 与档案
不一致」时，一律**报错拒绝**，绝不静默回退成「照旧录制」。

## 边界
- 纯函数：只依赖标准库 + 本包 :mod:`stereo.fov_align` 类型；不读文件、不 print、
  不起 GUI、不读 ``sys.argv``。可被 GUI / CLI / 测试直接 import。
- ``fov`` 一律返回 **dict**（``FovAlignment.from_dict().to_dict()`` 归一化），
  函数内不向调用方抛异常：解析失败包成 ``error`` 返回。
"""

from __future__ import annotations

from typing import Any

__all__ = ["resolve_fov_apply"]


def _result(apply: bool, fov: dict[str, Any] | None, reason: str,
            error: str | None, *, legacy_follow: bool = False,
            warning: str = "") -> dict[str, Any]:
    """统一返回形状，避免各分支各拼一份 dict 时漏键。

    ``legacy_follow`` / ``warning`` 是「开关关、但标定自带 FOV」那一支专用的
    提示通道；**其余所有分支走默认值**（False / ""），所以每个返回 dict 都带
    这两个键，下游 ``dec.get("legacy_follow")`` / ``dec["warning"]`` 不会 KeyError。
    """
    return {"apply": bool(apply), "fov": fov, "reason": reason,
            "error": error, "legacy_follow": legacy_follow,
            "warning": warning}


def _normalized_fov(raw: object) -> dict[str, Any]:
    """档案 / 标定里的 fov 归一成 dict；非法值由调用方包成 ``error``。"""
    from .fov_align import FovAlignment

    return FovAlignment.from_dict(raw).to_dict()      # type: ignore[arg-type]


def resolve_fov_apply(profile, *, mode: str,
                      calib_fov: dict[str, Any] | None,
                      profile_fov_apply: bool | None = None) -> dict[str, Any]:
    """判定这份 FOV 是否要应用到当前装置的左右相机画面。

    返回 ``{"apply": bool, "fov": dict|None, "reason": str, "error": str|None,
    "legacy_follow": bool, "warning": str}``，按顺序判定（先命中先返回）：

    1. ``apply_flag`` = 显式 ``profile_fov_apply``（非 None 时）或档案的
       ``profile.fov_apply``（缺字段 / 非 bool 由存储层回落成 False）；
    2. 未启用 → 分两支：标定自带 FOV（非空 dict 且非 angled）则
       ``legacy_follow=True``、沿用标定裁剪并给 ``warning``（旧流程行为不变）；
       其余（标定 fov 空 / None，或 angled）两路按原始画面、无 warning；
    3. 启用了但档案 ``fov`` 为空 → 没有记录，无法应用；
    4. ``mode == "angled"`` → 两路几乎不共视，不做裁剪（记录只作诊断）；
    5. 启用 + parallel + ``calib_fov`` 为 null / 空 → **error**，拒绝录制；
       绝不回退成「照旧录制」；
    6. 启用 + parallel + ``calib_fov`` 与档案 ``fov`` 的 ``scale`` / ``crop_rect`` /
       ``target_size`` 任一不一致 → **error**，先重新标定；
    7. 全通过 → 应用，返回档案那份 ``fov``（dict）。
    """
    # 1. 装置级开关（唯一来源：档案里的 fov_apply；测试可显式覆盖）
    if profile_fov_apply is not None:
        apply_flag = bool(profile_fov_apply)
    else:
        apply_flag = bool(getattr(profile, "fov_apply", False))

    profile_fov = getattr(profile, "fov", None)

    # 2. 未启用 —— 反向自证锚点：删掉这一整段，判据 2 / 8 必须变红
    if not apply_flag:
        # 2a. 开关关、但标定 JSON 自带 FOV：那次标定的内参是在裁剪帧上算的，
        #     录制必须跟着裁才几何自洽 —— 沿用标定自带的 FOV（旧流程行为不变），
        #     并给一句醒目提示，建议统一到装置档案。angled 不走这一支（见 2b）。
        if calib_fov and mode != "angled":
            return _result(
                False, None,
                "装置档案未启用「应用到当前摄像头」：本次沿用标定自带的 FOV",
                None,
                legacy_follow=True,
                warning="标定 JSON 带 FOV，但装置开关是关的：本次沿用标定自带的 "
                        "FOV 裁剪（几何自洽）；要统一请在 FOV 管理器里打开"
                        "「应用到当前摄像头」并读入这份记录",
            )
        # 2b. 其余情况（标定 fov 空 / None，或 angled）：两路按原始画面，无 warning。
        return _result(
            False, None,
            "装置档案未启用「应用到当前摄像头」：两路按原始画面",
            None,
        )

    # 3. 启用但档案没有 FOV 记录
    if not profile_fov:
        # 标定 JSON 却带着 fov：说明这份标定的内参是在**裁剪过的帧**上算的，
        # 档案却没把参数记下来。此时若按「没有记录」静默放行，录制会按原始
        # 画面写盘 —— 内参（裁剪帧）与画面（原始帧）口径不一致，深度系统性错
        # 且屏幕上看不出来。这是与第 5/6 条同一类的静默错配，必须报错拒绝。
        if calib_fov:
            return _result(
                False, None, "标定是在 FOV 对齐下做的，但装置档案里没有记录",
                "标定 JSON 带 fov（内参是在 FOV 对齐后的裁剪帧上算的），但装置"
                "档案里没有 FOV 记录：要么在 FOV 管理器里用「从当前标定读入」"
                "把记录补进档案，要么关掉「应用到当前摄像头」开关并重新标定"
                "（必须关 FOV 重新标定，否则内参仍是裁剪帧上的，与原始画面不符）",
            )
        # 标定本来就没用 FOV（fov 为 null / 空）：不裁是对的，benign 提示即可。
        return _result(False, None, "档案里没有 FOV 记录，无法应用", None)

    # fov 一律 dict；解析失败不抛给调用方，包成 error
    try:
        fov_dict = _normalized_fov(profile_fov)
    except (ValueError, TypeError, KeyError) as exc:
        return _result(
            False, None, "档案里的 FOV 记录无法解析",
            f"装置档案的 FOV 记录不合法：{exc}；请重新量一次或清除记录",
        )

    # 4. 大角度装置：两路视场几乎不共视，裁剪没有物理意义
    if mode == "angled":
        return _result(
            False, None,
            "大角度装置两路几乎不共视，不做裁剪（这份记录只作诊断）",
            None,
        )

    # 5. 启用了、parallel、但当前标定不是在 FOV 对齐下做的
    if not calib_fov:
        return _result(
            False, None,
            "当前标定不是在 FOV 对齐下做的，无法应用",
            "当前标定不是在 FOV 对齐下做的（标定 JSON 的 fov 为 null）："
            "先按这套 FOV 重新标定再录制",
        )

    # 6. 标定 fov 与档案 fov 必须同源：三个像素量纲字段逐一核对
    try:
        calib_norm = _normalized_fov(calib_fov)
    except (ValueError, TypeError, KeyError) as exc:
        return _result(
            False, None, "标定里的 FOV 无法解析",
            f"标定 JSON 的 fov 无法解析：{exc}；先重新标定",
        )
    for key in ("scale", "crop_rect", "target_size"):
        a, b = fov_dict.get(key), calib_norm.get(key)
        if key == "scale":
            same = float(a) == float(b)               # type: ignore[arg-type]
        else:
            same = ([int(v) for v in a]               # type: ignore[union-attr]
                    == [int(v) for v in b])           # type: ignore[union-attr]
        if not same:
            return _result(
                False, None, "标定里的 FOV 与装置档案不一致",
                "标定里的 FOV 与装置档案不一致：先重新标定",
            )

    # 7. 全通过
    return _result(
        True, fov_dict,
        "已应用装置档案的 FOV 对齐到当前装置的左右相机（标定与录制都服从）",
        None,
    )
