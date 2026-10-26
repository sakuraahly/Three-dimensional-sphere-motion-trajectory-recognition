"""把双目标定结果导出成**外部工具也能读**的参数文件。

为什么单独一层
--------------
`StereoRig.save()` 存的是给本项目自己读的格式：
`*.npz`（全矩阵）+ `*.json`（只有质量报告）。
但"拿到参数去 MATLAB / Unity / 另一个 Python 工程里用"需要的是
**分相机、带单位与坐标系说明、纯文本**的文件 —— 这是导出，不是保存。

三种格式，各有各的理由：

- **JSON**：stdlib 自带，任何语言都能读；
- **YAML**：人看最舒服，OpenCV/Python 生态常用；
- **TOML**：用户明确要的备用格式。Python 3.11+ 的 `tomllib` **只能读不能写**，
  所以这里自带一个最小序列化器（只覆盖本 schema 用到的类型），
  不引入 `tomli-w` 依赖。

单位与坐标系**必须写进文件**：三维点是"校正后左相机系、米"，
`T` 也是米。这两个约定不写，下一个人一定会按毫米去用。
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from sphere_tracker.plugins import names, plugin, resolve

from .params import StereoRig

SCHEMA_CAMERA = "sphere-camera"
SCHEMA_RIG = "sphere-stereo-rig"
VERSION = 1
# `FORMATS` 在文件后段由注册表生成（见"导出器注册表"一节），这里不写第二份清单。

_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


# --------------------------------------------------------------------------
# 数据整形
# --------------------------------------------------------------------------
def _list(value: Any) -> list | None:
    """numpy 数组 -> 嵌套 list（JSON/YAML/TOML 都能吃）。"""
    if value is None:
        return None
    return np.asarray(value, dtype=float).tolist()


def _clean(value: Any) -> Any:
    """递归转成纯 Python 类型；非有限浮点 -> None。

    为什么不保留 NaN：JSON 标准不允许 NaN（`json.dumps` 默认会写出非法 JSON），
    TOML 也不接受 `null`。统一成 `None`，三种格式的语义才一致。
    真丢了哪个质量指标，文件里会显示 `null`，比写一个 `NaN` 更诚实。
    """
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        v = float(value)
        return v if math.isfinite(v) else None
    return value


def build_camera(rig: StereoRig, side: str) -> dict:
    """单相机内参（左或右）。`D` 拉平成一维，顺序是 OpenCV 的 k1,k2,p1,p2,k3。"""
    left = side == "left"
    K = rig.K1 if left else rig.K2
    D = rig.D1 if left else rig.D2
    err_key = "err_left" if left else "err_right"
    return _clean({
        "schema": SCHEMA_CAMERA,
        "version": VERSION,
        "name": side,
        "image_size": [int(rig.image_size[0]), int(rig.image_size[1])],
        "model": "pinhole",
        "distortion_model": "opencv-k1k2p1p2k3",
        "K": _list(K),
        "D": _list(np.asarray(D, dtype=float).ravel()),
        "calibration": {
            "created": rig.meta.get("created"),
            "source": rig.meta.get("source"),
            "rms_px": rig.quality.get(err_key),
        },
    })


def build_rig(rig: StereoRig) -> dict:
    """双目整体参数（外参 + 校正 + 质量）。"""
    return _clean({
        "schema": SCHEMA_RIG,
        "version": VERSION,
        "image_size": [int(rig.image_size[0]), int(rig.image_size[1])],
        "units": {"translation": "m", "rotation": "rad", "image": "px"},
        "frame": ("3D points are expressed in the rectified left camera frame: "
                  "+X right, +Y down, +Z forward"),
        "extrinsics_convention": ("X_right = R @ X_left + T (OpenCV stereoCalibrate); "
                                  "T is the left camera origin in the right camera frame, in meters"),
        "baseline_mm": float(rig.baseline_mm),
        "baseline_side": rig.baseline_side(),
        "disparity_sign": int(rig.disparity_sign()),
        "R": _list(rig.R),
        "T": _list(np.asarray(rig.T, dtype=float).ravel()),
        "E": _list(rig.E),
        "F": _list(rig.F),
        "R1": _list(rig.R1),
        "R2": _list(rig.R2),
        "P1": _list(rig.P1),
        "P2": _list(rig.P2),
        "Q": _list(rig.Q),
        "quality": rig.quality,
        "meta": rig.meta,
    })


def build_export(rig: StereoRig) -> dict[str, dict]:
    """返回三份数据：`{"left": ..., "right": ..., "rig": ...}`。"""
    return {
        "left": build_camera(rig, "left"),
        "right": build_camera(rig, "right"),
        "rig": build_rig(rig),
    }


# --------------------------------------------------------------------------
# TOML（stdlib 只读，写由这里补）
# --------------------------------------------------------------------------
def _toml_key(key: str) -> str:
    return key if _BARE_KEY.match(key) else json.dumps(str(key), ensure_ascii=False)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return repr(value)
    if isinstance(value, str):
        # JSON 的字符串转义是 TOML basic string 的子集（含 \" \\ \n \t \uXXXX）
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        if any(v is None for v in value):
            raise ValueError("TOML 数组里不能有 null；导出前应把非有限值转成 None 并省略该键")
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(f"TOML 导出不支持的类型：{type(value).__name__}")


def toml_dumps(data: dict) -> str:
    """把本 schema 的 dict 写成 TOML。

    只支持：标量、字符串、数字数组（含嵌套）、嵌套表。
    本项目的标定 schema 正好只用到这些 —— 不为"通用 TOML 库"重造轮子。
    """
    lines: list[str] = []

    def emit(prefix: str, table: dict) -> None:
        # 先写标量，再写子表：TOML 里表头之后出现的键属于该表，
        # 顺序反了会把顶层键塞进上一张表。
        scalars = {k: v for k, v in table.items() if not isinstance(v, dict)}
        tables = {k: v for k, v in table.items() if isinstance(v, dict)}
        for key, value in scalars.items():
            if value is None:
                continue                      # TOML 没有 null
            lines.append(f"{_toml_key(key)} = {_toml_value(value)}")
        for key, value in tables.items():
            name = f"{prefix}{_toml_key(key)}"
            lines.append(f"[{name}]")
            emit(prefix + _toml_key(key) + ".", value)

    emit("", data)
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# 导出器注册表（`exporter`，R1/R7）
#
# 三个序列化器都定义在本模块里，所以注册也放在这里：单开
# `exporters/registry.py` 会变成「注册表 import export，export import 注册表」，
# 只能再引入延迟 import —— 为一个 3 项的查表多一个失败模式，不划算。
#
# 统一签名 `(data) -> str`：`write_export()` 只管"选哪个"，不再关心"怎么序列化"。
# 默认项 = 迁移前 `else` 分支的行为（toml）。
# --------------------------------------------------------------------------
@plugin("exporter", "json", summary="标准 JSON（stdlib 写出，任何语言都能读）")
def _dump_json(data: dict) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


@plugin("exporter", "yaml",
        summary="YAML（人读最舒服；需 PyYAML，缺包时导出前直接报错）",
        needs=("pyyaml",))
def _dump_yaml(data: dict) -> str:
    import yaml  # 依赖由 `write_export()` 预先检查，缺包时一个文件都不写

    return yaml.safe_dump(
        data, allow_unicode=True, sort_keys=False, default_flow_style=False,
    )


@plugin("exporter", "toml",
        summary="TOML（默认；内置最小序列化器，无需额外依赖）", default=True)
def _dump_toml(data: dict) -> str:
    return toml_dumps(data)


# 唯一清单：外部（`export_calibration.py`、help）都从这里取，不再手写。
FORMATS = tuple(names("exporter"))


# --------------------------------------------------------------------------
# 写文件
# --------------------------------------------------------------------------
def _summary(rig: StereoRig, files: list[str]) -> str:
    head = [
        "双目标定参数导出",
        "=" * 40,
        rig.describe(),
        "",
        "约定：",
        "  三维点 = 校正后左相机系，+X 右 / +Y 下 / +Z 前，单位米",
        "  T = 米；R 为无量纲旋转矩阵",
        "",
        "文件：",
    ]
    return "\n".join(head + [f"  {name}" for name in files]) + "\n"


def write_export(
    rig: StereoRig,
    out_dir: str | Path,
    formats: tuple[str, ...] | list[str] = FORMATS,
) -> list[Path]:
    """把三份数据按 `formats` 写到 `out_dir`，返回所有产物路径。

    先校验依赖再动手：请求了 YAML 但没装 PyYAML 时，**一个文件都不写**，
    免得留下半套导出物让人误以为完整。

    格式由 `exporter` 注册表分派（默认 toml）。**2026-10-26 起未知格式报错，
    不再静默降级为 toml** —— 把拼错的格式悄悄写成 TOML，正是本项目反复吃亏的
    "看起来成功了"。
    """
    fmts = [str(f).strip().lower() for f in formats if str(f).strip()]
    unknown = [f for f in fmts if f not in FORMATS]
    if unknown:
        raise ValueError(f"不支持的格式：{unknown}（可选 {FORMATS}）")
    if "yaml" in fmts:
        try:
            import yaml  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "导出 YAML 需要 PyYAML（当前环境未安装）。\n"
                "  安装：pip install pyyaml\n"
                "  或者只用：--formats json,toml"
            ) from exc

    data = build_export(rig)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    stems = {"left": "left_camera", "right": "right_camera", "rig": "stereo_rig"}
    for fmt in fmts:
        for key in ("left", "right", "rig"):
            path = out / f"{stems[key]}.{fmt}"
            text = resolve("exporter", fmt)(data[key])
            path.write_text(text, encoding="utf-8")
            written.append(path)

    summary = out / "summary.txt"
    summary.write_text(
        _summary(rig, [p.name for p in written]), encoding="utf-8"
    )
    written.append(summary)
    return written
