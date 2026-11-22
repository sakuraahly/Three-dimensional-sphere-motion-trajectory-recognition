"""装置档案：把"一套双目装置 = 一组参数 + 一份标定"固化下来。

## 为什么需要这一层

同一台机器上要维护**多套**双目装置（现在的近平行装置、将来的 90° 装置），但：

- GUI 里的分辨率 / 帧率**重开就回到硬编码默认**（`apps/gui_main.py` 里没有落盘）；
- 标定只有一份 ``config/stereo_calibration.json``，没有"这套参数配这份标定"的概念。

装置档案把这两件事写进同一个记录，供 GUI / CLI 读取，从而能回答
"现在用的是哪套装置、它配的是哪份标定"。

**不存几何预期**（2026-11-07 用户要求删）：期望基线 / 工作距离是卷尺量不准、
又会被人当成"标定结果"的猜测值。真实基线与夹角一律由标定 ``R``/``T`` 算出，
档案只存实测或配置值。标定时的物理核对另走 CLI ``--baseline-mm``，那是**这一次
测量**的锚，不是装置属性。

## 放在 ``stereo/`` 而不是 ``sphere_tracker/``

档案要校验 ``stereo_mode`` 的取值，而唯一真相是同层的
:mod:`stereo.stereo_mode`。若放到 ``sphere_tracker/``，就会让单目库反向依赖
双目层，违反 ``AGENTS.md`` §5 的依赖方向（``calib → stereo``，库不被双目依赖）。

## 三条硬约束

1. **不新增模式取值**：只允许 ``parallel`` / ``angled``，校验复用
   :func:`stereo.stereo_mode.normalize_mode`（``auto`` 会被它拒绝）。
   别名（``90`` / ``orthogonal`` …）在入口层才允许；档案是**存储层**，
   写别名等于给未来的自己埋歧义，所以这里一律拒绝，只存 canonical 名。
2. **不存手填的 ``angle_deg``，也不存几何预期**：``docs/数据标准.md`` 规定角度
   由标定 ``R`` 计算；基线由标定 ``T`` 计算。档案不再存
   ``expected_baseline_mm`` / ``working_distance_m``，也不再由它们估角度。
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
import os
import re
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, NamedTuple

from .stereo_mode import mode_label, normalize_mode

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
    "mode_label",
    "calibration_files_exist",
    "CalibrationStatus",
    "profile_calibration_status",
    "CALIB_STATUS_CALIBRATED",
    "CALIB_STATUS_UNCALIBRATED",
    "CALIB_STATUS_PATH_MISMATCH",
    "CALIB_STATUS_MODE_CONFLICT",
    "CALIBRATION_STATUS_LABELS",
    "calibration_identity",
    "IdentityStatus",
    "IDENTITY_MATCHED",
    "IDENTITY_MISMATCH",
    "IDENTITY_UNREGISTERED",
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
    """一套装置档案：只有 ``name`` 与 ``stereo_mode`` 是必填。

    几何量（基线 / 夹角）**不在这里**：它们由标定结果算出，手填的预期值
    2026-11-07 起已从档案里删除（见模块头）。
    """

    name: str
    stereo_mode: str = "parallel"
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
    #: 是否把 ``fov`` **应用到当前装置的两路相机画面**（装置级事实，默认关）。
    #: 「启用」= 应用：标定（复用这份参数、不传 --skip-fov）与录制（按它裁剪两路）
    #: 都服从它，而不是两个流程开关。关 = 两路按原始画面。90° 装置两路几乎不共视，
    #: 即使启用也不裁剪（由 :func:`stereo.fov_policy.resolve_fov_apply` 统一判定）。
    #: 旧档案缺这个字段一律回落 False（见 ``from_dict`` 的白名单清洗）。
    fov_apply: bool = False

    # -- 序列化 ------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """转成可 JSON 落盘的 dict：无值一律 ``None``（= JSON ``null``）。"""
        return {
            "name": self.name,
            "stereo_mode": self.stereo_mode,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "calibration_path": self.calibration_path,
            "notes": self.notes,
            "created": self.created,
            "fov": self.fov,
            "fov_apply": self.fov_apply,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RigProfile":
        """从存储记录构造；未知键忽略（向前兼容），缺省值由 dataclass 给。"""
        if not isinstance(data, Mapping):
            raise ProfileError(f"档案记录必须是对象，收到 {type(data).__name__}")
        known = {
            "name", "stereo_mode",
            "width", "height", "fps", "calibration_path", "notes", "created",
            "fov", "fov_apply",
        }
        # 旧档案里可能还留着 expected_baseline_mm / working_distance_m：
        # 不在 known 里就会被忽略（向前兼容，不因旧字段报错）。
        kwargs = {k: v for k, v in data.items() if k in known}
        # fov_apply 只允许 bool：字符串 "false" / 数字 1 一律回落 False
        # （bool("false") == True 是个静默陷阱，旧文件缺字段也回落 False）。
        # 注意 fov_apply=True 但 fov 为空时**不**在这里改值：存储层不拒旧档案，
        # 「没有记录，无法应用」由 resolve_fov_apply() 给出。
        raw_apply = kwargs.get("fov_apply", False)
        kwargs["fov_apply"] = raw_apply if isinstance(raw_apply, bool) else False
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
        width=width,
        height=height,
        fps=_opt_positive_int(profile.fps, "fps"),
        calibration_path=_check_calibration_path(profile.calibration_path),
        notes=notes,
        created=created,
        fov=_check_fov(profile.fov),
        # 任何路径落盘都保证 fov_apply 是 bool（from_dict 已清洗，这里是第二道）：
        # 直接 new 一个 RigProfile(fov_apply="false") 再 upsert 也不写脏值。
        fov_apply=(profile.fov_apply
                   if isinstance(profile.fov_apply, bool) else False),
    )


# ---------------------------------------------------------------------------
# 几何提示：已删除（2026-11-07）
# ---------------------------------------------------------------------------
# 原 geometry_hint() 拿档案里的"期望基线 / 工作距离"粗估夹角。用户要求档案不再
# 存这两个猜测值，函数随之删除：真实基线由标定 T 算、真实夹角由标定 R 算，
# 界面与检查脚本都不再打印"参考估算"那一行。
# 需要**这一次**的物理核对时走 CLI `--baseline-mm`（CalibConfig 的锚），
# 它不是装置属性，不进档案。


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


# ---------------------------------------------------------------------------
# 标定状态判定（2026-11-10 修复 B1 / C2）
# ---------------------------------------------------------------------------
# 以前「档案 ↔ 标定文件存在性」有三份判定：resolve_calibration_base（本文件）、
# apps/_gui_app.py::_stereo_calib_path / _read_rig、apps/tools/check_stereo_setup.py。
# 口径不一致，尤其 check_stereo_setup 只判无扩展名基名，对配了 calibration_path
# 的档案**必然误报**「标定文件不存在」。这里收敛成唯一入口：
#   - calibration_files_exist(base) 判 .npz / .json / 基名三者任一；
#   - profile_calibration_status(profile) 给四态（档案列表 / 状态条用）；
#   - calibration_identity(meta, active_name) 给三态（读取侧自检用，C2）。
# 全是纯文件 I/O + 纯计算，不 print / 不读 argv / 不起 GUI。

#: 装置档案标定状态四态。
CALIB_STATUS_CALIBRATED = "calibrated"
CALIB_STATUS_UNCALIBRATED = "uncalibrated"
CALIB_STATUS_PATH_MISMATCH = "path_mismatch"
CALIB_STATUS_MODE_CONFLICT = "mode_conflict"

#: 四态的中文短标签（档案列表「标定」列用；路径另放，别在 GUI 里重写判定）。
CALIBRATION_STATUS_LABELS = {
    CALIB_STATUS_CALIBRATED: "已标定",
    CALIB_STATUS_UNCALIBRATED: "未标定",
    CALIB_STATUS_PATH_MISMATCH: "路径不符",
    CALIB_STATUS_MODE_CONFLICT: "模式冲突",
}

#: 标定 meta.profile 与当前激活档案的对照三态（C2）。
IDENTITY_MATCHED = "matched"
IDENTITY_MISMATCH = "mismatch"
IDENTITY_UNREGISTERED = "unregistered"


class CalibrationStatus(NamedTuple):
    """``profile_calibration_status`` 的返回：状态 + 一句给人看的原因。"""

    state: str
    reason: str


class IdentityStatus(NamedTuple):
    """``calibration_identity`` 的返回：三态 + 一句给人看的话（matched 时为空）。"""

    state: str
    message: str


def calibration_files_exist(base: str | os.PathLike[str] | Path) -> bool:
    """标定基名对应的 ``.npz`` / ``.json`` / 无扩展名基名**三者任一**存在。

    这是「档案 ↔ 标定文件存在性」的**唯一判定**（2026-11-10 修复 B1）。
    ``StereoRig.load`` 收的是无扩展名基名、自己补 .json/.npz，所以只判
    ``base.exists()`` 会漏掉所有把产物写成 ``<base>.npz`` / ``<base>.json``
    的情形 —— ``check_stereo_setup.py:198`` 的必然误报就是这么来的。
    """
    p = Path(base)
    return (p.exists()
            or p.with_suffix(".npz").exists()
            or p.with_suffix(".json").exists())


def _read_calibration_meta(base: Path) -> dict[str, Any] | None:
    """读 ``<base>.json`` 的 meta（平铺 dict）。坏 JSON / 非对象返回 ``None``；
    没有 .json（只有 .npz / 基名）返回 ``{}`` —— 那种老产物没有可比对的身份/模式，
    仍算已标定，不误报。"""
    j = base.with_suffix(".json")
    if not j.exists():
        return {}
    try:
        data = json.loads(j.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


class CalibrationSpecs(NamedTuple):
    """标定实测的**结构化摘要**（给人看的一组数，不含判定逻辑）。

    ``state`` 与 :class:`CalibrationStatus` 的 ``state`` 同源。其余字段缺就是
    ``None`` / 空串 —— **不编默认值**：老标定没有 ``capture_size``、``fov`` 可能是
    ``null``，显示成 0 或 1920 都是在撒谎（2026-11-11 用户任务 B-189 的底线）。
    """

    state: str
    path: Path | None
    baseline_mm: float | None = None
    angle_deg: float | None = None
    rms_px: float | None = None
    epipolar_rms_px: float | None = None
    epipolar_max_px: float | None = None
    n_pairs_used: int | None = None
    n_pairs_in: int | None = None
    grade: str = ""
    created: str = ""
    capture_size: tuple[int, int] | None = None
    depth_sigma_mm_0_6m: float | None = None
    bound_profile: str = ""
    board: str = ""


def _as_float(v: object) -> float | None:
    """收成 float；``bool`` 不算（``True`` 当 1.0 是错的）。"""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def _as_int(v: object) -> int | None:
    if isinstance(v, bool) or not isinstance(v, int):
        return None
    return v


def calibration_specs(profile: RigProfile) -> CalibrationSpecs:
    """把一套装置档案的标定 meta 收成 :class:`CalibrationSpecs`。

    这是**数值展示的唯一产地**（2026-11-11 用户任务 B-189：装置档案要能看到
    标定后的具体信息）。GUI / CLI 都不许自己 ``json.loads`` 那份 meta 再挑字段。

    只读 ``<base>.json`` 的 meta（标定时写的人读副本），**不读 ``.npz``**：
    后者要 numpy + 二进制解析，只为显示几个数不值得，而且 meta 的字段名更稳定。

    没有 ``.json``（只有 ``.npz`` 的老产物）时 meta 是 ``{}`` —— 此时各数值字段
    全为 ``None``，调用方显示"已标定，但这份产物没有详情文件"，**不是**"未标定"。
    这两件事不能混：前者能用、只是不知道参数，后者根本不能用。
    """
    st = profile_calibration_status(profile)
    try:
        base: Path | None = _calibration_base_of(profile)
    except ProfileError:
        base = None
    meta = (_read_calibration_meta(base) if base is not None else None) or {}
    q = meta.get("quality")
    q = q if isinstance(q, dict) else {}
    cap = meta.get("capture_size")
    cap_size: tuple[int, int] | None = None
    if isinstance(cap, (list, tuple)) and len(cap) == 2:
        cw, ch = _as_int(cap[0]), _as_int(cap[1])
        if cw is not None and ch is not None:
            cap_size = (cw, ch)
    return CalibrationSpecs(
        state=st.state,
        path=base,
        baseline_mm=_as_float(meta.get("baseline_mm")),
        angle_deg=_as_float(meta.get("angle_deg")),
        rms_px=_as_float(q.get("rms")),
        epipolar_rms_px=_as_float(q.get("epipolar_rms")),
        epipolar_max_px=_as_float(q.get("epipolar_max")),
        n_pairs_used=_as_int(q.get("n_pairs_used")),
        n_pairs_in=_as_int(q.get("n_pairs_in")),
        grade=str(q.get("grade") or ""),
        created=str(meta.get("created") or ""),
        capture_size=cap_size,
        depth_sigma_mm_0_6m=_as_float(q.get("depth_sigma_mm_at_0_6m")),
        bound_profile=str(meta.get("profile") or ""),
        board=str(meta.get("board") or ""),
    )


def format_specs_inline(s: CalibrationSpecs) -> str:
    """一行极简版（给列表列用）：``68.8 mm / 0.7° / 0.71 px``。

    只放**最关键的三个数**（基线 / 夹角 / 极线 RMS）——列宽有限，完整指标走
    :func:`format_specs_tooltip`。缺一个就少一段；已标定但一个数都没有时
    明说"无详情"，不显示成"未标定"。
    """
    if s.state != CALIB_STATUS_CALIBRATED:
        return "—"
    parts: list[str] = []
    if s.baseline_mm is not None:
        parts.append(f"{s.baseline_mm:.1f} mm")
    if s.angle_deg is not None:
        parts.append(f"{s.angle_deg:.1f}°")
    if s.epipolar_rms_px is not None:
        parts.append(f"{s.epipolar_rms_px:.2f} px")
    return " / ".join(parts) if parts else "已标定（无详情）"


def format_specs_tooltip(s: CalibrationSpecs, *, include_path: bool = True) -> str:
    """多行详情（给悬停提示用）。**缺的项不出现**，不写"未知"占位。

    ``include_path=False``：调用方自己已经在别处显示了标定文件（装置档案对话框
    首行就是"状态 · 路径"），别在末尾再贴一遍绝对路径。
    """
    if s.state != CALIB_STATUS_CALIBRATED:
        return ""
    lines: list[str] = []
    if s.baseline_mm is not None:
        lines.append(f"基线：{s.baseline_mm:.2f} mm")
    if s.angle_deg is not None:
        lines.append(f"光轴夹角：{s.angle_deg:.3f}°")
    if s.epipolar_rms_px is not None:
        extra = (f"（最大 {s.epipolar_max_px:.2f} px）"
                 if s.epipolar_max_px is not None else "")
        lines.append(f"极线 RMS：{s.epipolar_rms_px:.3f} px{extra}")
    if s.rms_px is not None:
        lines.append(f"整体 RMS：{s.rms_px:.3f} px")
    if s.n_pairs_used is not None:
        tot = f"/{s.n_pairs_in}" if s.n_pairs_in is not None else ""
        lines.append(f"配对：{s.n_pairs_used}{tot} 对")
    if s.depth_sigma_mm_0_6m is not None:
        lines.append(f"深度不确定度 @0.6 m：{s.depth_sigma_mm_0_6m:.2f} mm")
    if s.capture_size is not None:
        lines.append(f"采集档：{s.capture_size[0]}x{s.capture_size[1]}")
    if s.board:
        lines.append(f"标定板：{s.board}")
    if s.created:
        lines.append(f"标定时间：{s.created}")
    if s.grade:
        lines.append(f"判定：{s.grade}")
    lines.append(f"登记装置：{s.bound_profile}" if s.bound_profile
                 else "登记装置：（未登记来源 —— 这份标定早于该字段）")
    if include_path and s.path is not None:
        lines.append(f"文件：{s.path}")
    return "\n".join(lines)


def calibration_identity(meta: Mapping[str, Any] | None,
                         active_name: str | None) -> IdentityStatus:
    """标定 meta 的 ``profile`` 与当前激活档案名对照（C2，严格三态）。

    - ``matched``：meta 有 ``profile`` 且 == 当前档案名 → 正常，``message=""``；
    - ``mismatch``：meta 有 ``profile`` 但 != 当前档案名（或当前未设档案）
      → 明确警告「这份标定属于档案 X，当前是 Y」；
    - ``unregistered``：meta **没有** ``profile``（所有老标定文件）→ 状态
      「未登记来源」，给一句**说明**，绝不误报成「属于别套装置」，也不阻断使用。

    这是纯计算：调用方（``apps/_gui_app.py`` 的 ``AppState``）自行决定把哪态
    写进 warn 级 notes、哪态写进 info 级 infos。老文件一律 unregistered，
    保证升级后门厅不会冒出一堆假警告。
    """
    bound = str((meta or {}).get("profile") or "").strip()
    cur = str(active_name or "").strip()
    if not bound:
        return IdentityStatus(
            IDENTITY_UNREGISTERED,
            "标定未登记来源装置（老文件），不校验档案身份")
    if cur and bound == cur:
        return IdentityStatus(IDENTITY_MATCHED, "")
    if not cur:
        return IdentityStatus(
            IDENTITY_MISMATCH,
            f"这份标定登记属于装置档案「{bound}」，但当前未设置装置档案")
    return IdentityStatus(
        IDENTITY_MISMATCH,
        f"这份标定属于装置档案「{bound}」，当前是「{cur}」—— "
        "请确认用对了标定，或给当前装置重新标定")


def _calibration_base_of(profile: RigProfile) -> Path:
    """这套档案该用的标定基名（无扩展名）；``calibration_path`` 非法时抛 ``ProfileError``。

    "档案指向哪份标定"**只有这一份实现**：``profile_calibration_status``（判状态）、
    ``calibration_specs``（取数值）共用。两处各拼一遍的话，一处改了路径校验、
    另一处不改，就会出现"状态说已标定、详情却读另一份文件"这种对不上的情况。

    档案没配路径 → 回落顶层 ``config/stereo_calibration``（与
    ``resolve_calibration_base`` 的 fallback 同一语义）。
    """
    raw = profile.calibration_path
    if raw is None:
        return ROOT / "config" / "stereo_calibration"
    return ROOT / _check_calibration_path(raw)


def profile_calibration_status(profile: RigProfile) -> CalibrationStatus:
    """一套装置档案的标定状态（B1 纯函数），返回 :class:`CalibrationStatus`。

    四态语义（按顺序判定，先命中先返回）：

    - ``path_mismatch``「路径指向别处或配置文件非法」：``calibration_path``
      不是仓库内合法相对路径（绝对 / 含 ``..`` / 越界），或 ``<base>.json``
      存在但内容坏 / 非对象，或标定 meta 登记的 ``profile`` 指向**另一套档案**
      （C1 起新标定会写这个字段；老文件没有，不触发）。
    - ``uncalibrated``「未标定」：``.npz`` / ``.json`` / 无扩展名基名**三者都不存在**。
      注意「档案名对得上、但文件还没标」属于这一类，不是 path_mismatch ——
      当前激活档案「实验性质平行装置1」就是这种，必须返回 uncalibrated。
    - ``mode_conflict``「模式冲突」：标定 meta 的 ``stereo_mode`` 与档案
      ``stereo_mode`` 不一致（与 ``AppState.mode_conflict()`` 同口径）。
    - ``calibrated``「已标定」：文件存在、可读、身份与模式都对得上。

    与 :func:`resolve_calibration_base` 同源，共用 :func:`calibration_files_exist`
    这一份存在性判定；不 print / 不读 argv / 不起 GUI，可被 GUI / CLI / 测试直接调。
    """
    if not isinstance(profile, RigProfile):
        return CalibrationStatus(CALIB_STATUS_PATH_MISMATCH, "档案对象非法")
    try:
        base = _calibration_base_of(profile)
    except ProfileError as exc:
        return CalibrationStatus(CALIB_STATUS_PATH_MISMATCH,
                                 f"标定路径非法：{exc}")
    if not calibration_files_exist(base):
        return CalibrationStatus(
            CALIB_STATUS_UNCALIBRATED,
            f"未标定：{base} 的 .npz / .json / 无扩展名基名都不存在")
    meta = _read_calibration_meta(base)
    if meta is None:
        return CalibrationStatus(
            CALIB_STATUS_PATH_MISMATCH,
            f"标定配置非法：{base.with_suffix('.json')} 读不成对象")
    ident = calibration_identity(meta, profile.name)
    if ident.state == IDENTITY_MISMATCH:
        return CalibrationStatus(CALIB_STATUS_PATH_MISMATCH, ident.message)
    cm = str(meta.get("stereo_mode") or "").strip()
    pm = str(profile.stereo_mode or "").strip()
    if cm and pm and cm != pm:
        return CalibrationStatus(
            CALIB_STATUS_MODE_CONFLICT,
            f"档案声明「{pm}」，标定文件却记为「{cm}」—— 很可能是别套装置的标定")
    return CalibrationStatus(CALIB_STATUS_CALIBRATED, "已标定")


def resolve_calibration_base(
    path: str | os.PathLike[str] | None = None,
    *,
    fallback: str | os.PathLike[str] | None = None,
    strict: bool = True,
) -> Path:
    """解析"当前该用哪份标定"：激活档案配了 ``calibration_path`` 就用它，否则回落。

    这是**装置档案接入主链路的唯一入口**（2026-11-04 前档案是半成品）：
    录制 / 重建 / 标定三个链路本都硬编码顶层 ``config/stereo_calibration``，
    于是"在档案里切换装置"是个假动作——切了档案，标定还是顶层那一份，
    两套装置共用一份标定，几何必然对不上。

    规则（优先级从高到低）：

    1. 激活档案配了 ``calibration_path``，**且它对应的 ``.npz`` 确实存在** →
       解析成绝对路径（不带扩展名的前缀，``.npz``/``.json`` 由调用方补）。
    2. 档案配了、但标定文件不存在 → **默认直接报错**（2026-11-08 改）。
       以前这里静默回落顶层，理由是"半成品档案也能录制"；代价是切了装置却
       拿别套装置的基线/夹角在跑，界面还显示"已标定"。用户实测 20px 极线
       残差才把这件事翻出来。宁可让人先标定，也不许冒充。
    3. 档案没配路径、没有激活档案、或档案文件损坏 → 回落 ``fallback``。

    **谁该调它**：``apps/`` 下的 GUI / CLI / 工具**不要直接调**，走
    ``apps._common.current_calibration`` —— 它把"抛 ``ProfileError``"翻译成
    ``(基名, 人话原因)``，调用点只判一个 ``why`` 就够，不必各写一份 try/except
    与提示文案（2026-11-10 用户裁决 B5 之后，GUI/CLI/工具链共十几处都要用同一份
    口径，各写一遍必然漂）。本函数留给 ``stereo`` 包内与需要区分
    "跟随了档案 / 回落了"的调用方。

    ``strict=False`` 的语义是"**不报错 ≠ 换几何**"：仍返回档案自己的 ``cand``
    （文件可以还不存在），绝不悄悄换成顶层那套装置的几何。它只给
    **本来就不需要现存标定**的调用方 —— 目前是
    ``apps/calibrate_stereo.py`` 不带 ``--out`` 时决定新标定写到哪
    （输出路径不要求文件已存在，但必须落在档案指定的位置），以及
    ``current_calibration(need_existing=False)`` / 它的 except 分支
    （要拿"该写到哪"的候选路径给人看）。

    调用方若需要区分"跟随了档案"还是"回落了"，看返回路径是否等于 fallback
    （``strict=False`` 跟随档案时返回 ``cand``，同样 ``!= fallback``）。
    """
    fb = Path(fallback) if fallback is not None else ROOT / "config" / "stereo_calibration"
    try:
        active = get_active(path)
    except ProfileError:
        active = None
    if active is not None and active.calibration_path:
        cand = ROOT / active.calibration_path
        if calibration_files_exist(cand):
            return cand
        if strict:
            raise ProfileError(
                f"装置档案「{active.name}」指定的标定文件不存在：{cand}"
                "（.npz / .json 都没有）。拒绝回落到顶层 "
                "config/stereo_calibration —— 那是别套装置的几何。"
                "请先给这套装置做标定，或在装置档案里清空标定路径。")
        return cand
    return fb


def active_name(path: str | os.PathLike[str] | None = None) -> str | None:
    """当前装置名；未设置返回 ``None``。"""
    return load_store(path)["active"]


def get_active(path: str | os.PathLike[str] | None = None) -> RigProfile | None:
    """当前装置档案；未设置返回 ``None``。"""
    name = active_name(path)
    return None if name is None else get_profile(name, path)
