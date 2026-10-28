"""单目内参标定：一组棋盘图 → K/D → **带质量判定**的结果对象。

为什么单独一个模块（2026-10-27 计划书 §3.2）
--------------------------------------------
库里其实早就有单目内参标定的算法 —— ``calib/stereo_calib.py:641
calibrate_mono_points``（``cv2.calibrateCameraExtended``），chain 方法的左右两支
一直在用它。缺的是三样东西：

1. **对外的契约**：那个函数返回裸 ``dict``，没有质量判定，谁调都得自己判"这组图够不够"；
2. **可保存**：没有 ``MonoCalib`` 这种带 schema 的对象，也就没有文件格式；
3. **可解释**：内参标定最常见的失败是"图不够/板没摆开"，而裸 ``dict`` 只会给一个
   好看的 RMS —— 这正是 ``AGENTS.md`` 反复警告的"**RMS 好看就通过**"。

所以本模块的产出不是"一个 K、D"，是**一个敢不敢用的结论**。

质量判定（缺一不可，全在这里定义，界面与报告都读同一份）
--------------------------------------------------------
====================================  =========  ==========================================
判据                                  阈值       为什么
====================================  =========  ==========================================
``rms_px``                            ≤ 0.5 px   内参标定的正常水平；>1 px 说明图糊或角点错
有效视角数                            ≥ 8        少于 8 个姿态时 ``fy`` 与畸变系数不可辨
姿态覆盖（板中心落在 5 个区域里的数）  ≥ 4/5      全挤在画面中央会让畸变系数失真
每张的板占比                          ≥ 3%       太小 = 角点定位不准
====================================  =========  ==========================================

**不够就抛 ``MonoCalibError``，不返回"勉强能用"的结果** —— 沿用项目既有的
"不可用不得保存"口径（``calib/stereo_calib.py`` 的 verdict 一脉）。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from .board import BoardSpec, detect_board
from .capture import CaptureConfig
from .stereo_calib import CalibConfig, calibrate_mono_points

__all__ = [
    "MonoCalib", "MonoCalibError", "QUALITY",
    "calibrate_mono", "save_mono", "load_mono", "report_text", "apply_to",
    "SCHEMA", "VERSION",
]

SCHEMA = "sphere-mono-calibration"
VERSION = 1

#: 质量门限的**唯一来源**（界面显示、报告文字、测试断言都从这里读）。
QUALITY = {
    "rms_px_max": 0.5,
    "min_views": 8,
    # 板面积门限与采集端**同源**（``CaptureConfig.min_area_ratio`` = 1.2%）。
    # 一开始这里自己写了 3%，结果采集端明明收下的图在标定端被判"板太小" ——
    # 同一概念两处门限，正是项目最忌讳的那种"第二份真相"。
    "min_area_ratio": float(CaptureConfig.min_area_ratio),
    "min_regions": 4,
    "regions_total": 5,
}


class MonoCalibError(ValueError):
    """这组图**不可用**（不是"程序跑挂了"）。

    携带 ``rejected``（逐图拒收原因），这样界面能直接告诉用户"哪几张不行、为什么"，
    而不是只说一句"标定失败"。
    """

    def __init__(self, message: str, rejected: Sequence[tuple[str, str]] = ()) -> None:
        super().__init__(message)
        self.rejected = list(rejected)


@dataclass(frozen=True)
class MonoCalib:
    """一次单目内参标定的结果。

    字段与 ``config/mono_calibration.json`` 一一对应；``K``/``D`` 是派生属性，
    不重复存（避免"文件里一份、对象里一份"两处真相）。
    """

    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    rms_px: float = 0.0
    image_size: tuple[int, int] = (0, 0)
    board: dict = field(default_factory=dict)
    per_view_rms: tuple[float, ...] = ()
    pose_coverage: dict = field(default_factory=dict)
    views: int = 0
    #: 被剔除的离群观测（文件名, 该张的 RMS）。**必须写进报告**：
    #: "剔了几张、为什么"是标定结果的一部分，不能悄悄丢数据。
    outliers: tuple[tuple[str, float], ...] = ()
    created_at: str = ""
    source_dir: str = ""

    # ---------- 派生 ----------
    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    @property
    def D(self) -> np.ndarray:
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3],
                        dtype=np.float64)

    @property
    def has_distortion(self) -> bool:
        """五个系数全 0 等于"没有畸变"，此时去畸变是空操作（不必白算）。"""
        return bool(np.any(np.abs(self.D) > 1e-12))

    def report(self) -> str:
        """人可读报告。也提供模块级 :func:`report_text` —— 两种写法都对，
        因为调用方直觉上会写 ``calib.report()``。"""
        return report_text(self)

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "version": VERSION,
            "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
            "k1": self.k1, "k2": self.k2, "p1": self.p1, "p2": self.p2,
            "k3": self.k3,
            "rms_px": self.rms_px,
            "image_size": list(self.image_size),
            "board": dict(self.board),
            "per_view_rms": list(self.per_view_rms),
            "pose_coverage": dict(self.pose_coverage),
            "views": self.views,
            "outliers": [{"name": n, "rms_px": r} for n, r in self.outliers],
            "created_at": self.created_at,
            "source_dir": self.source_dir,
            "quality": dict(QUALITY),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "MonoCalib":
        if not isinstance(raw, dict):
            raise MonoCalibError(f"标定文件必须是对象，收到 {type(raw).__name__}")
        if raw.get("schema") not in (None, SCHEMA):
            raise MonoCalibError(f"不是单目标定文件（schema={raw.get('schema')!r}）")

        def _f(key: str, default: float = 0.0) -> float:
            v = raw.get(key, default)
            return float(v) if isinstance(v, (int, float)) else default

        size = raw.get("image_size") or [0, 0]
        return cls(
            fx=_f("fx"), fy=_f("fy"), cx=_f("cx"), cy=_f("cy"),
            k1=_f("k1"), k2=_f("k2"), p1=_f("p1"), p2=_f("p2"), k3=_f("k3"),
            rms_px=_f("rms_px"),
            image_size=(int(size[0]), int(size[1])) if len(size) >= 2 else (0, 0),
            board=dict(raw.get("board") or {}),
            per_view_rms=tuple(float(x) for x in (raw.get("per_view_rms") or [])
                               if isinstance(x, (int, float))),
            pose_coverage=dict(raw.get("pose_coverage") or {}),
            views=int(raw.get("views") or 0),
            outliers=tuple((str(o.get("name") or ""), float(o.get("rms_px") or 0.0))
                           for o in (raw.get("outliers") or [])
                           if isinstance(o, dict)),
            created_at=str(raw.get("created_at") or ""),
            source_dir=str(raw.get("source_dir") or ""),
        )


# --------------------------------------------------------------------------- #
# 几何小工具
# --------------------------------------------------------------------------- #
def _object_points(spec: BoardSpec) -> np.ndarray:
    """棋盘角点的 3D 坐标（z=0 平面，单位任意 —— 内参标定里它只定尺度）。

    用**毫米**与 ``square_mm`` 一致；单目内参本身与尺度无关，但保持与双目一致
    能省掉一次"这里单位是什么"的疑问。
    """
    xs, ys = np.meshgrid(np.arange(spec.cols, dtype=np.float32),
                         np.arange(spec.rows, dtype=np.float32))
    pts = np.zeros((spec.cols * spec.rows, 3), np.float32)
    pts[:, 0] = xs.ravel() * spec.square_mm
    pts[:, 1] = ys.ravel() * spec.square_mm
    return pts


def _region_index(cx: float, cy: float, w: int, h: int) -> int:
    """板中心落在 5 个区域里的哪一个（0..4；都不沾边返回 -1）。

    取 3×3 网格的**四角 + 中央**：那 5 个位置能覆盖"畸变可辨"所需的姿态多样性。
    """
    if w <= 0 or h <= 0:
        return -1
    third_x, third_y = w / 3.0, h / 3.0
    col = 0 if cx < third_x else (2 if cx >= 2 * third_x else 1)
    row = 0 if cy < third_y else (2 if cy >= 2 * third_y else 1)
    if (row, col) == (1, 1):
        return 4                                   # 中央
    return {(0, 0): 0, (0, 2): 1, (2, 0): 2,
            (2, 2): 3}.get((row, col), -1)         # 只有四角算数；四边中点不算


# --------------------------------------------------------------------------- #
# 标定
# --------------------------------------------------------------------------- #
def calibrate_mono(image_paths: Iterable[str | Path], spec: BoardSpec, *,
                   fix_principal_point: bool = False,
                   rational_model: bool = False,
                   min_views: int | None = None,
                   rms_max: float | None = None,
                   min_regions: int | None = None) -> MonoCalib:
    """从一组棋盘图标定单目内参。

    参数都给了默认值（来自 :data:`QUALITY`），测试时可以收紧/放开来验证判定本身
    是否真的会红 —— 这是"判据可自证"的要求，不是给生产用的旋钮。

    抛 :class:`MonoCalibError`（带 ``rejected`` 明细）表示**这组图不可用**；
    其他异常（读图 OSError 之类）原样上抛。
    """
    paths = [Path(p) for p in image_paths]
    if not paths:
        raise MonoCalibError("没有图片")
    need = QUALITY["min_views"] if min_views is None else int(min_views)
    rms_limit = QUALITY["rms_px_max"] if rms_max is None else float(rms_max)
    regions_need = (QUALITY["min_regions"] if min_regions is None
                    else int(min_regions))

    objp = _object_points(spec)
    imgpoints: list[np.ndarray] = []
    meta: list[dict] = []
    rejected: list[tuple[str, str]] = []
    image_size: tuple[int, int] | None = None

    for p in paths:
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            rejected.append((p.name, "读不出来"))
            continue
        size = (int(gray.shape[1]), int(gray.shape[0]))
        if image_size is None:
            image_size = size
        elif size != image_size:
            rejected.append((p.name, f"尺寸不一致 {size[0]}×{size[1]}"))
            continue
        obs = detect_board(gray, spec)
        if not obs.found or obs.corners is None:
            rejected.append((p.name, f"没检出棋盘（{obs.note or '未知原因'}）"))
            continue
        if obs.area_ratio < QUALITY["min_area_ratio"]:
            rejected.append(
                (p.name, f"板太小（占画面 {obs.area_ratio:.1%}，"
                         f"至少要 {QUALITY['min_area_ratio']:.0%}）"))
            continue
        pts = obs.corners.reshape(-1, 2).astype(np.float64)
        imgpoints.append(pts)
        meta.append({"name": p.name, "area_ratio": float(obs.area_ratio),
                     "focus": float(obs.focus), "detector": obs.detector,
                     "center": (float(pts[:, 0].mean()), float(pts[:, 1].mean()))})

    if image_size is None:
        raise MonoCalibError("一张图都读不出来", rejected)
    if len(imgpoints) < need:
        detail = "；".join(f"{n}：{why}" for n, why in rejected[:5])
        raise MonoCalibError(
            f"有效视角只有 {len(imgpoints)} 个，至少需要 {need} 个"
            + (f"（前几个拒收原因：{detail}）" if detail else ""), rejected)

    res = calibrate_mono_points(
        objp, imgpoints, image_size,
        CalibConfig(fix_principal_point=fix_principal_point,
                    rational_model=rational_model))
    def _unpack(r: dict):
        per = np.asarray(r["per_view"], np.float64).ravel()
        return (np.asarray(r["K"], np.float64).reshape(3, 3),
                np.asarray(r["D"], np.float64).ravel(),
                float(r["rms"]), per)

    K, D, rms, per_view_arr = _unpack(res)

    # ---- 剔除离群观测后重标 ------------------------------------------------
    # 为什么必须做：RMS 是**所有点一起去平均**，一张 5 px 的坏图（角点检错/
    # 轻微遮挡）就能把 24 张 0.3 px 的好图拉到 1.1 px，于是整组被判"不可用"。
    # 实测 2026-10-28：30 张中位数 0.32 px、最差一张 5.24 px，全用 RMS=1.10，
    # 剔掉 6 张后 RMS=0.271。**这不是放宽门限**（门限仍是 0.5），
    # 而是先把明显错的观测拿掉再重新求解 —— 标定的标准做法。
    # 剔除必须**报告**（outliers 字段 + 报告文字），不许悄悄丢数据。
    outliers: list[tuple[str, float]] = []
    med = float(np.median(per_view_arr)) if per_view_arr.size else 0.0
    limit = max(1.0, 4.0 * med)          # 至少 1.0 px 才考虑剔，避免好数据互剔
    bad_idx = [i for i, r in enumerate(per_view_arr) if r > limit]
    if bad_idx and (len(imgpoints) - len(bad_idx)) >= need:
        keep_idx = [i for i in range(len(imgpoints)) if i not in set(bad_idx)]
        outliers = [(meta[i]["name"], float(per_view_arr[i])) for i in bad_idx]
        res2 = calibrate_mono_points(
            objp, [imgpoints[i] for i in keep_idx], image_size,
            CalibConfig(fix_principal_point=fix_principal_point,
                        rational_model=rational_model))
        K, D, rms, per_view_arr = _unpack(res2)
        meta = [meta[i] for i in keep_idx]
        kept_paths = [paths[i] for i in keep_idx]
    else:
        kept_paths = paths
    per_view = tuple(float(x) for x in per_view_arr)

    if rms > rms_limit:
        raise MonoCalibError(
            f"重投影 RMS {rms:.2f} px 超过 {rms_limit} px —— "
            f"多半是图糊、板太小或角点检错。别用这组结果", rejected)

    # 姿态覆盖：板中心落在哪些区域
    regions: dict[int, int] = {}
    for m in meta:
        idx = _region_index(m["center"][0], m["center"][1],
                            image_size[0], image_size[1])
        if idx >= 0:
            regions[idx] = regions.get(idx, 0) + 1
    if len(regions) < regions_need:
        raise MonoCalibError(
            f"姿态只覆盖了 {len(regions)}/5 个区域（至少 {regions_need} 个）—— "
            f"板要摆到画面四角与中央，全挤在中间会让畸变系数失真", rejected)

    d5 = np.zeros(5, np.float64)
    d5[:min(5, D.size)] = D[:5]
    return MonoCalib(
        fx=float(K[0, 0]), fy=float(K[1, 1]),
        cx=float(K[0, 2]), cy=float(K[1, 2]),
        k1=float(d5[0]), k2=float(d5[1]), p1=float(d5[2]), p2=float(d5[3]),
        k3=float(d5[4]),
        rms_px=rms,
        image_size=image_size,
        board={"cols": spec.cols, "rows": spec.rows,
               "square_mm": spec.square_mm},
        per_view_rms=per_view,
        pose_coverage={"regions": sorted(regions), "count": len(regions),
                       "total": QUALITY["regions_total"]},
        views=len(per_view),
        outliers=tuple(outliers),
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        source_dir=str(kept_paths[0].parent) if kept_paths else "",
    )


# --------------------------------------------------------------------------- #
# 存 / 读 / 报告
# --------------------------------------------------------------------------- #
def save_mono(calib: MonoCalib, path: str | Path) -> Path:
    """写 ``config/mono_calibration.json``。父目录不存在就建。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(calib.to_dict(), ensure_ascii=False, indent=2) + "\n",
                 encoding="utf-8")
    return p


def load_mono(path: str | Path) -> MonoCalib | None:
    """读回来；文件不存在/坏掉返回 ``None``（**不抛** —— 调用方只想知道"有没有"）。"""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return MonoCalib.from_dict(raw)
    except (OSError, ValueError, MonoCalibError, KeyError, TypeError):
        return None


def apply_to(calib: MonoCalib, camera) -> None:
    """把标定写进任意带 ``fx/fy/cx/cy`` 属性的对象（``CameraConfig`` 等）。

    用 duck typing 而不是 import ``sphere_tracker.config``：``calib`` 是下层，
    不该反过来依赖上层配置类 —— 那会让"标定库"只能给这一个上层用。
    ``distortion`` 只在目标对象确实有该属性时才写，兼容老配置对象。
    """
    camera.fx = float(calib.fx)
    camera.fy = float(calib.fy)
    camera.cx = float(calib.cx)
    camera.cy = float(calib.cy)
    if hasattr(camera, "distortion"):
        camera.distortion = tuple(float(x) for x in calib.D)


def report_text(calib: MonoCalib) -> str:
    """人可读报告：一行结论 + 关键数字 + 覆盖情况。"""
    cov = calib.pose_coverage or {}
    lines = [
        f"单目内参标定 · {calib.views} 个有效视角，RMS {calib.rms_px:.3f} px",
        f"  fx = {calib.fx:.1f} px，fy = {calib.fy:.1f} px",
        f"  主点 = ({calib.cx:.1f}, {calib.cy:.1f})，图 {calib.image_size[0]}×{calib.image_size[1]}",
        f"  畸变 k1={calib.k1:+.4f} k2={calib.k2:+.4f} p1={calib.p1:+.4f} "
        f"p2={calib.p2:+.4f} k3={calib.k3:+.4f}"
        + ("（无畸变，去畸变是空操作）" if not calib.has_distortion else ""),
        *([f"  已剔除离群观测 {len(calib.outliers)} 张："
           + "、".join(f"{n}({r:.1f}px)" for n, r in calib.outliers[:4])
           + (f" 等 {len(calib.outliers)} 张" if len(calib.outliers) > 4 else "")]
          if calib.outliers else []),
        f"  姿态覆盖 {cov.get('count', '?')}/{cov.get('total', QUALITY['regions_total'])} 个区域"
        f"，板 {calib.board.get('cols', '?')}×{calib.board.get('rows', '?')}"
        f"，方格 {calib.board.get('square_mm', '?')} mm",
        "  判定：可用 —— 满足三维重建对单目的要求" if calib.rms_px <= QUALITY["rms_px_max"]
        else "  判定：不可用",
    ]
    return "\n".join(lines)
