"""双目模式：平行 / 大角度（近似直角）的归一化、推断与几何提示。

这一层只回答三个问题，不依赖项目内其他模块：

1. 配置里的字符串到底是哪种模式？（``normalize_mode`` / ``infer_mode``）
2. 给人的界面该显示什么？（``mode_label``）
3. 当前基线与工作距离在几何上意味着什么？（``geometry_advice``）

``"auto"`` 不是第三种模式，而是**入口层**的"帮我挑一个"约定：只有
:func:`resolve_auto` 能解析它；核心层 :func:`normalize_mode` 收到 ``auto``
会直接抛 ``ValueError``（以前把它当 ``angled`` 的别名，导致
``normalize_mode("auto")`` 与入口实际行为相反）。

兼容原则
--------
- 旧 session / calibration JSON **没有** ``stereo_mode`` 字段，必须按平行模式解释；
- 新数据可以写在 ``session["stereo_mode"]``、``session["rig"]["stereo_mode"]``、
  ``calibration["stereo_mode"]`` 或 ``calibration["meta"]["stereo_mode"]``；
- 任何字段都无法识别时都退回 ``parallel``，不抛异常 —— 旧数据不能因为新代码打不开。

几何提示只是**建议**，不是门禁：``angled`` 表示“大角度 / 近似直角”，
不表示标定得到的夹角一定是 90.0°。实际夹角请用标定 R 计算
:func:`optical_axes_angle_deg`，或从基线/工作距离估算
:func:`geometry_advice`。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

MODES = ("parallel", "angled")

__all__ = [
    "MODES",
    "normalize_mode",
    "resolve_auto",
    "infer_mode",
    "mode_label",
    "optical_axes_angle_deg",
    "geometry_advice",
]

_AUTO = "auto"

_MODE_ALIASES = {
    "parallel": "parallel",
    "angled": "angled",
    "90": "angled",
    "90deg": "angled",
    "orthogonal": "angled",
    "angle": "angled",
}

_MODE_LABELS = {
    "parallel": "平行",
    "angled": "大角度（近似直角）",
}
# ``auto`` 不是模式，但**界面上会出现**（下拉框的默认项）。它必须有自己的
# 标签：返回"未知"会让用户以为选错了，而"自动判定"正是它的真实含义。
_AUTO_LABEL = "自动判定（按标定）"

_MISSING = object()


def _canonical_mode(value: Any) -> str | None:
    """把字符串归一化成 canonical 名称；无法识别返回 None。"""
    if not isinstance(value, str):
        return None
    return _MODE_ALIASES.get(value.strip().casefold())


def _is_auto(value: Any) -> bool:
    """``"auto"`` 是入口层的"帮我挑"约定，不是模式本身。"""
    return isinstance(value, str) and value.strip().casefold() == _AUTO


def normalize_mode(value: Any, default: str = "parallel") -> str:
    """将模式值归一化为 ``"parallel"`` 或 ``"angled"``。

    接受 canonical 名称和别名（大小写、首尾空白不敏感）：

    - ``parallel`` -> ``parallel``
    - ``angled`` / ``90`` / ``90deg`` / ``orthogonal`` / ``angle`` -> ``angled``

    ``None`` 和无法识别的显式值**不会抛异常**，而是返回 ``default``
    （默认 ``parallel``）。``default`` 本身也会被归一化；若它也无效，
    最终退回 ``parallel``，保证返回值始终是 :data:`MODES` 之一。

    ``"auto"`` **例外**：它是合法用户输入，但不能出现在核心层 ——
    推断出什么依赖标定 meta / 会话 / 旧输出，只有 :func:`resolve_auto`
    知道这些上下文。这里收到 ``auto``（含大小写、空白变体）一律抛
    ``ValueError``，而不是悄悄取 ``default`` 或别名成 ``angled``。
    """
    fallback = _canonical_mode(default) or "parallel"
    if _is_auto(value):
        raise ValueError(
            "'auto' 只能在入口层解析（按标定 meta / 会话 / 旧输出选择 "
            "parallel 或 angled），核心层不接受 auto："
            "调用方先 resolve_stereo_mode()"
        )
    return _canonical_mode(value) or fallback


def resolve_auto(
    explicit: Any = None,
    *,
    session: Mapping | None = None,
    calibration: Mapping | None = None,
    default: str = "parallel",
) -> str:
    """入口层唯一解析点：把 ``"auto"`` 落成 ``parallel`` / ``angled``。

    优先级（高到低）：

    1. ``explicit``：命令行/界面显式给的模式（``None`` 或 ``auto`` 表示没给）；
    2. 标定记录：``calibration["stereo_mode"]`` -> ``calibration["meta"]["stereo_mode"]``；
    3. 会话记录：``session["stereo_mode"]`` -> ``session["rig"]["stereo_mode"]``；
    4. ``default``（默认 ``parallel``，旧数据没有模式字段）。

    某一层"存在但不是 auto"就立即采用，不再往下找；值无法识别时交给
    :func:`normalize_mode` 落到 ``default``（与旧语义一致：拼错不跨层猜）。
    任何一层里写着 ``auto`` 都只是"没定"，会继续往下找 —— 这样旧会话里的
    ``auto`` 不会把整条链路弄崩，也不会被误解成某个具体模式。
    """
    fallback = _canonical_mode(default) or "parallel"
    if explicit is not None and not _is_auto(explicit):
        return normalize_mode(explicit, default=fallback)

    candidates = (
        (calibration, ("stereo_mode",)),
        (calibration, ("meta", "stereo_mode")),
        (session, ("stereo_mode",)),
        (session, ("rig", "stereo_mode")),
    )
    for data, path in candidates:
        value = _lookup(data, *path)
        if value is _MISSING or _is_auto(value):
            continue
        return normalize_mode(value, default=fallback)
    return fallback


def _lookup(data: Any, *path: str) -> Any:
    """按路径取值；不存在或中间层不是 mapping 时返回 ``_MISSING``。"""
    current = data
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return _MISSING
        current = current[key]
    return current


def infer_mode(
    session: dict | None = None,
    calibration: dict | None = None,
) -> str:
    """从旧 / 新会话与标定数据中推断双目模式。

    优先级从高到低：

    1. ``session["stereo_mode"]``
    2. ``session["rig"]["stereo_mode"]``
    3. ``calibration["stereo_mode"]``
    4. ``calibration["meta"]["stereo_mode"]``
    5. ``parallel``

    只认最靠前、且**确实存在**的字段。字段存在但值无法识别时，交给
    :func:`normalize_mode` 落到 ``parallel``，不再回退到下一层；只有字段缺失
    （旧 JSON）或中间层不是对象时才继续往下找。这样"字段存在但拼错"不会
    被静默解释成另一份标定文件的模式。

    字段写成 ``"auto"``（例如旧入口把未解析的取值直接落了盘）视为"没定"，
    继续往下一层找，最后由 ``parallel`` 兜底 —— 本函数不做 auto 解析，
    那需要入口上下文，见 :func:`resolve_auto`。
    """
    candidates = (
        (session, ("stereo_mode",)),
        (session, ("rig", "stereo_mode")),
        (calibration, ("stereo_mode",)),
        (calibration, ("meta", "stereo_mode")),
    )
    for data, path in candidates:
        value = _lookup(data, *path)
        if value is _MISSING or _is_auto(value):
            continue
        return normalize_mode(value)
    return "parallel"


def mode_label(mode: Any) -> str:
    """返回面向界面的中文标签；无法识别返回 ``"未知"``。

    canonical 名称和别名都会先归一化，因此 ``mode_label("90")`` 与
    ``mode_label("angled")`` 相同。``"auto"`` 单独给标签
    （``"自动判定（按标定）"``）—— 它是界面下拉里真实存在的一项，
    回 ``"未知"`` 会让人以为选项非法。
    """
    if _is_auto(mode):
        return _AUTO_LABEL
    canonical = _canonical_mode(mode)
    if canonical is None:
        return "未知"
    return _MODE_LABELS[canonical]


def optical_axes_angle_deg(R: Any) -> float:
    """由标定旋转矩阵 ``R`` 计算两光轴夹角（度）。

    定义：左相机 +Z 轴与 ``R.T @ [0, 0, 1]`` 的夹角，即右相机 +Z 轴在
    左相机系中的方向与左光轴的夹角。``R`` 必须是 3x3；计算前会归一化向量。

    无效输入（形状不对、非数值、含 NaN/inf、零向量）返回 ``0.0``，
    不抛异常。合法输入范围 ``[0, 180]``。
    """
    try:
        matrix = np.asarray(R, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return 0.0

    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return 0.0

    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right_axis = matrix.T @ z_axis
    norm = float(np.linalg.norm(right_axis))
    if not math.isfinite(norm) or norm <= 0.0:
        return 0.0

    cosine = float(np.dot(z_axis, right_axis) / norm)
    cosine = min(1.0, max(-1.0, cosine))
    angle = math.degrees(math.acos(cosine))
    return angle if math.isfinite(angle) else 0.0


def _positive_finite(value: Any) -> float | None:
    """转成正的有限 float；不合法返回 None，不抛异常。"""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def geometry_advice(baseline_mm: float, working_distance_m: float) -> dict:
    """按基线与工作距离给出对称交会角与重叠提示。

    假设两相机**对称**朝向同一个工作点，则光轴夹角为：

    .. code::

        angle = 2 * atan((B / 2) / D)

    其中 ``B`` 与 ``D`` 使用同一单位。本函数入参是 ``baseline_mm`` 和
    ``working_distance_m``，内部统一换算成米；``ratio = B / D``（无量纲）。

    返回字典固定包含：

    - ``angle_deg``：对称交会角，单位度；
    - ``ratio``：``B / D``；
    - ``warning``：纯建议文案。

    警告阈值：``angle < 15`` -> ``"接近平行"``；
    ``15 <= angle <= 120`` -> ``"大角度"``；
    ``angle > 120`` -> ``"接近 180° 无重叠风险"``。

    这不是硬门禁，不会因为 B/D 不匹配 90° 而拒绝调用。无效输入
    （None、非数值、NaN/inf、<= 0）返回 ``angle_deg=0.0``、``ratio=0.0``、
    ``warning=""``；这些 0 是“未计算”的哨兵值，不是测得的 0°。
    """
    baseline = _positive_finite(baseline_mm)
    distance = _positive_finite(working_distance_m)
    if baseline is None or distance is None:
        return {"angle_deg": 0.0, "ratio": 0.0, "warning": ""}

    baseline_m = baseline / 1000.0
    ratio = baseline_m / distance

    # 等价于 2*atan((B/2)/D)，用 atan2 避免中间量的边界问题。
    half_angle = math.atan2(baseline_m, 2.0 * distance)
    angle_deg = math.degrees(2.0 * half_angle)

    if angle_deg < 15.0:
        warning = "接近平行"
    elif angle_deg <= 120.0:
        warning = "大角度"
    else:
        warning = "接近 180° 无重叠风险"

    return {"angle_deg": angle_deg, "ratio": ratio, "warning": warning}


# ---------------------------------------------------------------------------
# 自检：直接运行 `python stereo/stereo_mode.py`，只依赖 numpy，退出码 0 即通过。
# ---------------------------------------------------------------------------
def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _run_selftest() -> None:
    # 1) canonical 名称、别名、旧数据默认值。
    _require(MODES == ("parallel", "angled"), "MODES 定义不对")
    _require(normalize_mode("parallel") == "parallel", "parallel 归一化失败")
    for alias in ("angled", "90", "90deg", "orthogonal", "angle"):
        _require(
            normalize_mode(alias) == "angled",
            f"别名 {alias!r} 应归一化为 angled",
        )
    for sentinel in ("auto", "AUTO", " Auto "):
        try:
            normalize_mode(sentinel)                          # noqa: B018
        except ValueError:
            pass
        else:
            raise AssertionError(f"normalize_mode({sentinel!r}) 必须抛 ValueError")
    _require(normalize_mode(None) == "parallel", "None 应退回 parallel")
    _require(normalize_mode("not-a-mode") == "parallel", "未知值应退回 parallel")
    _require(
        normalize_mode("not-a-mode", default="angled") == "angled",
        "未知值应退回显式 default",
    )
    _require(
        normalize_mode("not-a-mode", default="also-bad") == "parallel",
        "无效 default 应最终退回 parallel",
    )
    print("PASS modes: aliases normalize to parallel/angled, invalid -> parallel")

    # 2) 优先级、旧 JSON（无字段）与标签。
    _require(infer_mode() == "parallel", "无数据应推断为 parallel")
    _require(infer_mode({}, {}) == "parallel", "旧 JSON 应推断为 parallel")
    _require(
        infer_mode({"stereo_mode": "angled"}, {"stereo_mode": "parallel"})
        == "angled",
        "session 顶层字段优先级最高",
    )
    _require(
        infer_mode({"rig": {"stereo_mode": "90"}}) == "angled",
        "session.rig 别名推断失败",
    )
    _require(
        infer_mode({}, {"stereo_mode": "angled"}) == "angled",
        "calibration 顶层字段推断失败",
    )
    _require(
        infer_mode({}, {"meta": {"stereo_mode": "angled"}}) == "angled",
        "calibration.meta 字段推断失败",
    )
    _require(
        infer_mode({"rig": None}, {"stereo_mode": "angled"}) == "angled",
        "嵌套层缺失时应继续回退",
    )
    _require(
        infer_mode({"stereo_mode": "auto"}, {"stereo_mode": "angled"}) == "angled",
        "auto 字段应视为未定，继续回退",
    )
    _require(
        resolve_auto("angled", calibration={"meta": {"stereo_mode": "parallel"}})
        == "angled",
        "显式模式优先于标定",
    )
    _require(
        resolve_auto("auto", session={"stereo_mode": "parallel"},
                     calibration={"meta": {"stereo_mode": "angled"}}) == "angled",
        "标定 meta 优先于会话",
    )
    _require(
        resolve_auto(None, calibration={"meta": {"stereo_mode": "auto"}},
                     session={"rig": {"stereo_mode": "angled"}}) == "angled",
        "auto 视为未定，会话应接上",
    )
    _require(resolve_auto("auto") == "parallel", "无依据应退回 parallel")
    _require(resolve_auto("auto", default="angled") == "angled", "default 应生效")
    _require(mode_label("parallel") == "平行", "parallel 标签不对")
    _require(mode_label("angled") == "大角度（近似直角）", "angled 标签不对")
    _require(mode_label("90") == "大角度（近似直角）", "别名标签不对")
    _require(mode_label("???") == "未知", "未知标签不对")
    print("PASS inference: precedence + legacy JSON + labels")

    # 3) 光轴夹角：单位阵 0°，绕 Y 轴 90° 为 90°。
    angle_identity = optical_axes_angle_deg(np.eye(3))
    _require(abs(angle_identity - 0.0) < 1e-9, f"单位阵应为 0°，实际 {angle_identity}")

    r_y_90 = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    angle_90 = optical_axes_angle_deg(r_y_90)
    _require(abs(angle_90 - 90.0) < 1e-9, f"绕 Y 90° 应为 90°，实际 {angle_90}")
    _require(optical_axes_angle_deg(np.eye(2)) == 0.0, "非 3x3 应返回 0")
    _require(
        optical_axes_angle_deg(np.full((3, 3), np.nan)) == 0.0,
        "非有限 R 应返回 0",
    )
    print(
        "PASS optical axes: identity=0.00 deg, +90 about Y=90.00 deg, invalid=0"
    )

    # 4) 几何：B=600 mm / D=300 mm -> 90°；B=64.6 mm / D=300 mm -> 12.3°。
    advice_90 = geometry_advice(600.0, 0.30)
    _require(
        abs(advice_90["angle_deg"] - 90.0) < 1e-9,
        f"B=600mm/D=300mm 应为 90°，实际 {advice_90['angle_deg']}",
    )
    _require(abs(advice_90["ratio"] - 2.0) < 1e-12, "B/D 比值应为 2.0")
    _require(advice_90["warning"] == "大角度", "90° 的警告应为大角度")
    print(
        "PASS geometry: B=600 mm, D=300 mm -> "
        f"{advice_90['angle_deg']:.2f} deg ({advice_90['warning']})"
    )

    advice_par = geometry_advice(64.6, 0.30)
    expected_par = math.degrees(
        2.0 * math.atan((64.6 / 2.0) / (0.30 * 1000.0))
    )
    _require(
        abs(advice_par["angle_deg"] - expected_par) < 1e-9,
        f"B=64.6mm/D=300mm 计算偏差：{advice_par['angle_deg']} != {expected_par}",
    )
    _require(abs(advice_par["angle_deg"] - 12.3) < 0.1, "B=64.6mm 应约为 12.3°")
    _require(advice_par["warning"] == "接近平行", "12.3° 的警告应为接近平行")
    print(
        "PASS geometry: B=64.6 mm, D=300 mm -> "
        f"{advice_par['angle_deg']:.2f} deg ({advice_par['warning']})"
    )

    # 5) 警告边界、无效输入与“建议而非门禁”。
    advice_wide = geometry_advice(1000.0, 0.10)  # ~157.4°
    _require(
        advice_wide["warning"] == "接近 180° 无重叠风险",
        "大交会角警告不对",
    )
    for bad_baseline, bad_distance in (
        (0.0, 0.30),
        (-1.0, 0.30),
        (600.0, 0.0),
        (600.0, -0.30),
        (float("nan"), 0.30),
        (600.0, float("inf")),
        (None, 0.30),
        ("bad", 0.30),
    ):
        advice_bad = geometry_advice(bad_baseline, bad_distance)
        _require(advice_bad["warning"] == "", "无效输入的 warning 应为空")
        _require(advice_bad["angle_deg"] == 0.0, "无效输入 angle 应为 0")
        _require(advice_bad["ratio"] == 0.0, "无效输入 ratio 应为 0")
    _require(geometry_advice(600.0, 0.30)["warning"] == "大角度", "建议不应拒绝 90°")
    print("PASS geometry: warning thresholds + invalid inputs + advisory only")
    print("SELFTEST PASS: 6/6 groups")


if __name__ == "__main__":
    _run_selftest()
    raise SystemExit(0)
