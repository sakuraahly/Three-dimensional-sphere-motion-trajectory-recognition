"""装置档案：把"一套双目装置 = 一组参数 + 一份标定 + 几何预期"固化下来。

## 为什么需要这一层

同一台机器上要维护**多套**双目装置（现在的近平行装置、将来的 90° 装置），但：

- GUI 里的分辨率 / 帧率**重开就回到硬编码默认**（`启动程序.py` 里没有落盘）；
- 标定只有一份 ``config/stereo_calibration.json``，没有"这套参数配这份标定"的概念；
- 90° 装置的期望基线、工作距离只存在于人的脑子里。

装置档案把这三件事写进同一个记录，供 GUI / CLI 读取，从而能回答
"现在用的是哪套装置、它该有多少基线、它配的是哪份标定"。

## 放在 ``stereo/`` 而不是 ``sphere_tracker/``

档案要校验 ``stereo_mode`` 的取值，而唯一真相是同层的
:mod:`stereo.stereo_mode`。若放到 ``sphere_tracker/``，就会让单目库反向依赖
双目层，违反 ``AGENTS.md`` §5 的依赖方向（``calib → stereo``，库不被双目依赖）。

## 三条硬约束

1. **不新增模式取值**：只允许 ``parallel`` / ``angled``，校验复用
   :func:`stereo.stereo_mode.normalize_mode`（``auto`` 会被它拒绝）。
   别名（``90`` / ``orthogonal`` …）在入口层才允许；档案是**存储层**，
   写别名等于给未来的自己埋歧义，所以这里一律拒绝，只存 canonical 名。
2. **不存手填的 ``angle_deg``**：``docs/数据标准.md`` 规定角度由标定 ``R``
   计算。档案只存 ``expected_baseline_mm`` / ``working_distance_m``，
   需要角度时由 :func:`geometry_hint` **估算并标注"非标定值"**。
3. **``calibration_path`` 必须是仓库内相对路径**：拒绝绝对路径与 ``..``
   逃逸，与清理功能的路径口径保持一致（防御性，不是功能需求）。

## 落盘位置

``config/profiles.json``（``config/`` 已被 ``.gitignore`` 整目录忽略，
属运行期产物；**不放** ``data/``，那里是不可再生数据）。

存储格式（UTF-8，无 NaN / Infinity，无值写 ``null``）::

    {
      "schema": "sphere-rig-profiles",
      "version": 1,
      "active": "parallel-desk" | null,
      "profiles": { "<name>": { ... }, ... }
    }

本模块只做**纯文件 I/O + 纯计算**：不 print、不读 ``sys.argv``、不起 GUI，
可被 GUI / CLI / 测试直接 import。
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from .stereo_mode import geometry_advice, mode_label, normalize_mode

__all__ = [
    "RigProfile",
    "ProfileError",
    "STORE_SCHEMA",
    "STORE_VERSION",
    "DEFAULT_PATH",
    "empty_store",
    "load_store",
    "save_store",
    "list_profiles",
    "get_profile",
    "upsert",
    "delete",
    "set_active",
    "get_active",
    "active_name",
    "resolve_calibration_base",
    "validate_profile",
    "geometry_hint",
    "mode_label",
]

STORE_SCHEMA = "sphere-rig-profiles"
STORE_VERSION = 1

#: 仓库根（``stereo/`` 的上一级），用于定位运行期配置与校验相对路径。
ROOT = Path(__file__).resolve().parents[1]
#: 默认存储路径。测试通过 ``path=`` 参数改用临时目录，不污染真 ``config/``。
DEFAULT_PATH = ROOT / "config" / "profiles.json"

_MAX_NAME = 64
# 相对路径里的 Windows 盘符，例如 ``C:/x``。``PurePosixPath`` 不认它。
_DRIVE_RE = re.compile(r"^[A-Za-z]:")


class ProfileError(ValueError):
    """档案非法（名字 / 模式 / 数值 / 路径）或存储文件损坏。"""


# ---------------------------------------------------------------------------
# 基础校验
# ---------------------------------------------------------------------------
def _check_name(value: Any) -> str:
    """档案名：非空、无首尾空白、长度受限。作为 dict 的 key，必须稳定。"""
    if not isinstance(value, str):
        raise ProfileError(f"档案名必须是字符串，收到 {type(value).__name__}")
    name = value.strip()
    if not name:
        raise ProfileError("档案名不能为空")
    if len(name) > _MAX_NAME:
        raise ProfileError(f"档案名太长（{len(name)} > {_MAX_NAME}）")
    if any(ch in name for ch in "\r\n\t"):
        raise ProfileError("档案名不能含换行/制表符")
    return name


def _strict_mode(value: Any) -> str:
    """只接受 canonical ``parallel`` / ``angled``；拒绝 ``auto`` 与一切别名。

    取值集合的唯一真相是 :data:`stereo.stereo_mode.MODES`，这里不重写一份；
    但存储层额外要求"原样就是 canonical 名"，避免 ``orthogonal`` / ``90``
    这类别名在不同版本间产生歧义。``auto`` 由 ``normalize_mode`` 直接拒绝。
    """
    if not isinstance(value, str) or not value.strip():
        raise ProfileError("stereo_mode 必须是 'parallel' 或 'angled'")
    try:
        canonical = normalize_mode(value)
    except ValueError as exc:          # "auto"：核心层不接受
        raise ProfileError(f"stereo_mode 不接受 'auto'（入口层才会解析 auto）：{exc}") from exc
    if value.strip().casefold() != canonical:
        raise ProfileError(
            f"stereo_mode 只接受 'parallel' / 'angled'，不接受别名 {value!r}"
            f"（会归一化成 {canonical!r}，请直接写 canonical 名）"
        )
    return canonical


def _opt_positive_float(value: Any, field_name: str) -> float | None:
    """``None`` 或有限正数；其余（bool / 0 / 负数 / NaN / inf）一律拒绝。"""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ProfileError(f"{field_name} 必须是正数，不能是布尔值")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProfileError(f"{field_name} 必须是正数，收到 {value!r}") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ProfileError(f"{field_name} 必须是有限正数，收到 {value!r}")
    return number


def _opt_positive_int(value: Any, field_name: str) -> int | None:
    """``None`` 或正整数（拒绝 bool / 浮点 / 0 / 负数）。"""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProfileError(f"{field_name} 必须是整数或 None，收到 {value!r}")
    if value <= 0:
        raise ProfileError(f"{field_name} 必须是正整数，收到 {value!r}")
    return value


def _check_calibration_path(value: Any) -> str | None:
    """仓库内相对路径（POSIX 分隔符）。拒绝绝对路径、盘符与 ``..`` 逃逸。"""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ProfileError("calibration_path 必须是字符串或 None")
    raw = value.strip().replace("\\", "/")
    if not raw:
        return None
    if raw.startswith("/") or _DRIVE_RE.match(raw) or PurePosixPath(raw).is_absolute():
        raise ProfileError(
            f"calibration_path 必须是仓库内相对路径，不能是绝对路径：{value!r}"
        )
    parts = PurePosixPath(raw).parts
    if ".." in parts:
        raise ProfileError(
            f"calibration_path 不能含 '..'（会逃出仓库）：{value!r}"
        )
    if not parts:
        return None
    return str(PurePosixPath(*parts))


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


#: `fov` 记录除 `FovAlignment` 那 8 个字段之外，允许携带的元信息。
#: 它们的用途是**判断这份记录还能不能信**：换了相机 / 动了分辨率之后，
#: 旧的 scale / crop_rect 就不再适用（那些是像素量纲，随分辨率变）。
_FOV_META_KEYS = ("left_size", "right_size", "measured_at", "source")


def _check_fov(value: Any) -> dict[str, Any] | None:
    """校验并归一化装置档案里的 ``fov`` 记录。

    校验**复用 `FovAlignment` 自己的 `__post_init__`**（reference_side /
    crop_side / crop_rect / target_size / scale / inliers / quality 的全部规则），
    不在这里重写一遍 —— 两份校验迟早会漂：档案放行了标定层会拒绝的值，
    用户看到的就是"档案存进去了，一录就报错"。

    额外元信息（`left_size` / `right_size` / `measured_at` / `source`）单独校验，
    未知键**丢弃**（与 `RigProfile.from_dict` 的向前兼容口径一致）。
    """
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ProfileError(f"fov 必须是对象或 None，收到 {type(value).__name__}")

    from .fov_align import FovAlignment          # 同层，避免模块级循环

    try:
        FovAlignment.from_dict(value)            # 只用于校验，不保留实例
    except ValueError as exc:
        raise ProfileError(f"fov 记录不合法：{exc}") from exc

    out: dict[str, Any] = {}
    for key in ("reference_side", "crop_side", "crop_rect", "target_size",
                "scale", "method", "inliers", "quality"):
        out[key] = value[key]
    # 归一化成 JSON 友好类型（tuple -> list；数值统一 python 原生类型）
    out["crop_rect"] = [int(v) for v in value["crop_rect"]]
    out["target_size"] = [int(v) for v in value["target_size"]]
    out["scale"] = float(value["scale"])
    out["inliers"] = int(value["inliers"])
    out["quality"] = float(value["quality"])

    for size_key in ("left_size", "right_size"):
        if size_key not in value or value[size_key] is None:
            continue
        raw = value[size_key]
        try:
            size = [int(v) for v in raw]
        except (TypeError, ValueError) as exc:
            raise ProfileError(
                f"fov.{size_key} 必须是 [宽, 高]，收到 {raw!r}") from exc
        if len(size) != 2 or size[0] <= 0 or size[1] <= 0:
            raise ProfileError(
                f"fov.{size_key} 必须是 [宽>0, 高>0]，收到 {raw!r}")
        out[size_key] = size

    for text_key in ("measured_at", "source"):
        if text_key not in value or value[text_key] is None:
            continue
        if not isinstance(value[text_key], str):
            raise ProfileError(f"fov.{text_key} 必须是字符串")
        out[text_key] = value[text_key].strip()

    return out


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------
@dataclass
class RigProfile:
    """一套装置档案。

    只有 ``name`` 与 ``stereo_mode`` 是必填；几何预期（基线 / 工作距离）
    可以为空 —— 为空时 :func:`geometry_hint` 会明说"无法估算"，而不是猜。
    """

    name: str
    stereo_mode: str = "parallel"
    expected_baseline_mm: float | None = None
    working_distance_m: float | None = None
    width: int | None = None
    height: int | None = None
    fps: int | None = None
    calibration_path: str | None = None
    notes: str = ""
    created: str = field(default_factory=_now)
    #: 两路相机的**固有视场差**（`stereo.fov_align.FovAlignment` 的字典 +
    #: 测量时两路分辨率 / 时间 / 来源）。为什么不只放在某次标定的 meta 里：
    #: 这是**装置属性**（镜头出厂焦距不同，实测左路比右路放大 ~1.5 倍），
    #: 换一次标定不会变；而 meta 里的那份会随"重新标定"消失或变成 null。
    #: 存进档案后，平行装置可以复用同一份裁剪参数，单目也能查到"这台机器
    #: 两路差多少"，不必每次重新估。
    #:
    #: ⚠️ **90° 装置不做裁剪**：两路光轴近乎垂直、视场几乎不重叠，
    #: 单应裁剪没有物理意义（`record_stereo` 在 angled 下明确忽略 fov）。
    #: 档案里存它，是为了记录"这台机器的两路视场差"这个事实本身，
    #: 以及给单目提供参考——不是让 90° 也去裁剪。
    fov: dict[str, Any] | None = None

    # -- 序列化 ------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """转成可 JSON 落盘的 dict：无值一律 ``None``（= JSON ``null``）。"""
        return {
            "name": self.name,
            "stereo_mode": self.stereo_mode,
            "expected_baseline_mm": self.expected_baseline_mm,
            "working_distance_m": self.working_distance_m,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "calibration_path": self.calibration_path,
            "notes": self.notes,
            "created": self.created,
            "fov": self.fov,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RigProfile":
        """从存储记录构造；未知键忽略（向前兼容），缺省值由 dataclass 给。"""
        if not isinstance(data, Mapping):
            raise ProfileError(f"档案记录必须是对象，收到 {type(data).__name__}")
        known = {
            "name", "stereo_mode", "expected_baseline_mm", "working_distance_m",
            "width", "height", "fps", "calibration_path", "notes", "created",
            "fov",
        }
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)


def validate_profile(
    profile: RigProfile,
    *,
    existing_names: Iterable[str] = (),
    allow_existing: bool = False,
) -> RigProfile:
    """校验并把可归一化的字段归一化，返回新对象（不修改入参）。

    ``existing_names`` 给出"已占用"的名字集合；``allow_existing=False``
    （默认）时重名直接拒绝，``True`` 时允许覆盖同名档案（``upsert(replace=True)``）。
    """
    if not isinstance(profile, RigProfile):
        raise ProfileError(f"需要 RigProfile，收到 {type(profile).__name__}")

    name = _check_name(profile.name)
    taken = {_check_name(n) for n in existing_names}
    if name in taken and not allow_existing:
        raise ProfileError(f"档案名已存在：{name!r}（换个名字，或显式覆盖）")

    notes = profile.notes if isinstance(profile.notes, str) else None
    if notes is None:
        raise ProfileError("notes 必须是字符串")
    created = profile.created.strip() if isinstance(profile.created, str) else ""
    if not created:
        created = _now()

    # 采集档：宽高**必须成对**（2026-11-05）。
    # 只给宽会让采集变成「指定的宽 × 相机自己的默认高」—— 那个高没人能预期，
    # 而它会被记进标定的 capture_size、决定之后所有录制请求。宁可在这里拒绝，
    # 也不要存一个半对的档（界面上也拦一道，但底线在库里）。
    width = _opt_positive_int(profile.width, "width")
    height = _opt_positive_int(profile.height, "height")
    if (width is None) != (height is None):
        raise ProfileError(
            "采集档的宽和高必须**一起**给（或都不给）：只给一个会变成"
            "「指定宽 × 相机默认高」，那个尺寸没人能预期，还会被写进标定")

    return replace(
        profile,
        name=name,
        stereo_mode=_strict_mode(profile.stereo_mode),
        expected_baseline_mm=_opt_positive_float(
            profile.expected_baseline_mm, "expected_baseline_mm"),
        working_distance_m=_opt_positive_float(
            profile.working_distance_m, "working_distance_m"),
        width=width,
        height=height,
        fps=_opt_positive_int(profile.fps, "fps"),
        calibration_path=_check_calibration_path(profile.calibration_path),
        notes=notes,
        created=created,
        fov=_check_fov(profile.fov),
    )


# ---------------------------------------------------------------------------
# 几何提示（纯函数，角度只是**估算**）
# ---------------------------------------------------------------------------
def geometry_hint(profile: RigProfile) -> str:
    """按期望基线与工作距离给出**参考**提示（单行，可直接 print）。

    ## 语气：一律是"参考估算"，不是判定（2026-10-24 改）

    用户原话："我不能精确确定摄像头的位置，所以这个不该设置成强相关固定，
    相反，应该以标定计算得到结果为主。"

    所以这里的话术必须让读者一眼看出两件事：

    1. 括号里的角度是**由 B/D 粗估**的，不是标定值；
    2. 真实夹角**以标定 ``R`` 计算为准**（``docs/数据标准.md``：``angle_deg``
       不是手填的 90.0）。

    因此本函数**不返回任何"错误/不支持"的判定语义** —— 它只是字符串。
    估算偏离 90° 只说明"档案里填的 B/D 与 90° 装置对不上"，而不是"这个组合非法"：
    用户的 B/D 本来就是估计值，错的是数字，不是装置。
    """
    baseline = profile.expected_baseline_mm
    distance = profile.working_distance_m
    if baseline is None or distance is None:
        missing = "、".join(
            name for name, value in (
                ("expected_baseline_mm", baseline),
                ("working_distance_m", distance),
            ) if value is None
        )
        return (f"参考估算：档案未填 {missing}（两项都是可选的，不影响判定）；"
                "实际夹角以标定结果为准")

    angle = float(geometry_advice(baseline, distance)["angle_deg"])
    parts = [
        f"参考估算：B={baseline:.0f} mm / D={distance:.2f} m ≈ {angle:.1f}°"
        f"（由 B/D 粗估，非标定值；实际夹角以标定 R 计算为准）"
    ]
    if profile.stereo_mode == "angled":
        if abs(angle - 90.0) > 10.0:
            need = 2.0 * distance * 1000.0
            parts.append(
                f"按此布置看不到 90° 所需的共同视场（对称 90° 需 B≈2·D≈{need:.0f} mm）"
                f"—— 若装置是 90°，说明档案里的 B/D（当前 {baseline:.0f} mm）与实际不符，"
                "请按标定结果更新"
            )
        else:
            parts.append("与本档案估算的 90° 相符（±10° 内，仅作参考）")
    elif angle >= 15.0:
        parts.append("若实际装置是大角度/近似直角，请把 stereo_mode 设为 angled（估算仅作参考）")
    else:
        parts.append("与平行/近平行装置的参考估算相符")
    return "；".join(parts)


# ---------------------------------------------------------------------------
# 存储
# ---------------------------------------------------------------------------
def empty_store() -> dict[str, Any]:
    """空存储（文件不存在时的返回值，不是报错）。"""
    return {"schema": STORE_SCHEMA, "version": STORE_VERSION,
            "active": None, "profiles": {}}


def _path(path: str | os.PathLike[str] | None) -> Path:
    return Path(path) if path is not None else DEFAULT_PATH


def load_store(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """读取存储；**文件不存在返回空 store**（首次使用不报错）。

    文件存在但内容损坏（非 JSON / 顶层不是对象 / ``profiles`` 不是对象）
    抛 :class:`ProfileError`：静默当成空会让用户以为档案丢了。
    """
    target = _path(path)
    if not target.exists():
        return empty_store()
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProfileError(f"档案文件读不了：{target}（{exc}）") from exc
    if not isinstance(raw, dict):
        raise ProfileError(f"档案文件顶层必须是对象：{target}")

    profiles = raw.get("profiles", {})
    if not isinstance(profiles, Mapping):
        raise ProfileError(f"档案文件的 profiles 必须是对象：{target}")

    store = empty_store()
    store["profiles"] = dict(profiles)
    active = raw.get("active")
    # active 指向不存在的档案时按"未设置"处理：删档案不该留下悬空指针。
    store["active"] = active if isinstance(active, str) and active in store["profiles"] else None
    if isinstance(raw.get("schema"), str) and raw["schema"]:
        store["schema"] = raw["schema"]
    if isinstance(raw.get("version"), int) and not isinstance(raw.get("version"), bool):
        store["version"] = raw["version"]
    return store


def save_store(store: Mapping[str, Any],
               path: str | os.PathLike[str] | None = None) -> Path:
    """原子写盘（先写 ``.tmp`` 再 ``os.replace``），UTF-8，禁止 NaN/Infinity。"""
    if not isinstance(store, Mapping):
        raise ProfileError("save_store 需要字典")
    target = _path(path)
    payload = {
        "schema": store.get("schema", STORE_SCHEMA),
        "version": store.get("version", STORE_VERSION),
        "active": store.get("active"),
        "profiles": dict(store.get("profiles") or {}),
    }
    try:
        text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    except (TypeError, ValueError) as exc:
        raise ProfileError(f"档案里有无法序列化的值（NaN/Infinity？）：{exc}") from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return target


def _decode(name: str, record: Any) -> RigProfile:
    try:
        profile = RigProfile.from_dict(record)
        return validate_profile(profile)
    except ProfileError as exc:
        raise ProfileError(f"档案 {name!r} 无效：{exc}") from exc


def list_profiles(path: str | os.PathLike[str] | None = None) -> list[RigProfile]:
    """列出所有**合法**档案（非法记录跳过，避免一条坏记录毁掉整个列表）。"""
    store = load_store(path)
    out: list[RigProfile] = []
    for name, record in store["profiles"].items():
        try:
            out.append(_decode(name, record))
        except ProfileError:
            continue
    return out


def get_profile(name: str,
                path: str | os.PathLike[str] | None = None) -> RigProfile | None:
    """按名字取档案；不存在返回 ``None``。记录本身非法时抛 :class:`ProfileError`。"""
    key = _check_name(name)
    record = load_store(path)["profiles"].get(key)
    if record is None:
        return None
    return _decode(key, record)


def upsert(profile: RigProfile, *, replace_existing: bool = False,
           path: str | os.PathLike[str] | None = None) -> RigProfile:
    """写入档案（新增或覆盖）。

    ``replace_existing=False``（默认）时同名视为**错误** —— 覆盖一次装好的
    档案应当是有意的动作；``True`` 才允许覆盖。
    """
    store = load_store(path)
    existing = set(store["profiles"])
    clean = validate_profile(
        profile,
        existing_names=existing,
        allow_existing=replace_existing,
    )
    store["profiles"][clean.name] = clean.to_dict()
    save_store(store, path)
    return clean


def delete(name: str, path: str | os.PathLike[str] | None = None) -> bool:
    """删除档案；返回是否真的删掉。删的是 active 时把 ``active`` 一起清掉。"""
    key = _check_name(name)
    store = load_store(path)
    if key not in store["profiles"]:
        return False
    del store["profiles"][key]
    if store["active"] == key:
        store["active"] = None
    save_store(store, path)
    return True


def set_active(name: str | None,
               path: str | os.PathLike[str] | None = None) -> RigProfile | None:
    """设置当前装置；``None`` 表示清除。指向不存在的档案直接报错。"""
    store = load_store(path)
    if name is None:
        store["active"] = None
    else:
        key = _check_name(name)
        if key not in store["profiles"]:
            raise ProfileError(f"没有这个档案：{key!r}")
        store["active"] = key
    save_store(store, path)
    return get_active(path)


def resolve_calibration_base(
    path: str | os.PathLike[str] | None = None,
    *,
    fallback: str | os.PathLike[str] | None = None,
) -> Path:
    """解析"当前该用哪份标定"：激活档案配了 ``calibration_path`` 就用它，否则回落。

    这是**装置档案接入主链路的唯一入口**（2026-11-04 前档案是半成品）：
    录制 / 重建 / 标定三个链路本都硬编码顶层 ``config/stereo_calibration``，
    于是"在档案里切换装置"是个假动作——切了档案，标定还是顶层那一份，
    两套装置共用一份标定，几何必然对不上。

    规则（优先级从高到低）：

    1. 激活档案配了 ``calibration_path``，**且它对应的 ``.npz`` 确实存在** →
       解析成绝对路径（不带扩展名的前缀，``.npz``/``.json`` 由调用方补）。
    2. 档案配了但文件不存在、档案没配、或 ``profiles.json`` 缺失/损坏 →
       回落 ``fallback``（默认顶层 ``config/stereo_calibration``）。
       **"配了但不存在"必须回落**：半成品档案的典型状态就是路径指向一个
       从没标定过的 ``config/calib/<装置>``（2026-11-04 现场：切到"90度装置"
       后路径指向不存在的文件，若无条件跟随，录制会直接报"找不到标定"，
       而顶层那份真实标定被白白忽略）。

    调用方若需要区分"跟随了档案"还是"回落了"，看返回路径是否等于 fallback。
    """
    fb = Path(fallback) if fallback is not None else ROOT / "config" / "stereo_calibration"
    try:
        active = get_active(path)
    except ProfileError:
        active = None
    if active is not None and active.calibration_path:
        cand = ROOT / active.calibration_path
        if cand.with_suffix(".npz").exists():
            return cand
        # 档案路径是空壳：回落到顶层那份真实存在的标定
        return fb
    return fb


def active_name(path: str | os.PathLike[str] | None = None) -> str | None:
    """当前装置名；未设置返回 ``None``。"""
    return load_store(path)["active"]


def get_active(path: str | os.PathLike[str] | None = None) -> RigProfile | None:
    """当前装置档案；未设置返回 ``None``。"""
    name = active_name(path)
    return None if name is None else get_profile(name, path)
