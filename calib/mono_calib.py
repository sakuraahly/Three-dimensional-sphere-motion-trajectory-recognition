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
========================================  =========  ======================================
判据                                      阈值       为什么
========================================  =========  ======================================
``rms_px``                                ≤ 0.5 px   正常水平，超过就在报告里警告一行
``rms_px``                                ≤ 0.8 px   硬上限，与双目 ``ok_rms`` 同一个数
留一法 ``fx`` 极差 ÷ 中位 ``fx``          ≤ 3%       深度误差 1:1 跟着 fx 走，飘了别做定量
留一法 ``fx`` 极差 ÷ 中位 ``fx``          ≤ 8%       硬上限，超过判"焦距定不出来"
有效视角数                                ≥ 8        少于 8 个姿态时 ``fy`` 与畸变系数不可辨
姿态覆盖（板中心落在 5 个区域里的数）      ≥ 4/5      全挤在画面中央会让畸变系数失真
每张的板占比                              ≥ 1.2%     太小 = 角点定位不准（与采集端同源）
========================================  =========  ======================================

**判定分两条线，各自回答一个不同的问题**（2026-10-29 重定，见下方"为什么 RMS 让位"）：

1. **RMS 回答"角点对齐得好不好"** —— 它只说明重投影残差，**不说明焦距能不能定出来**。
   所以它只分三档：≤0.5 干净、0.5~0.8 报告里警告、>0.8 拒。
2. **留一法 fx 极差回答"这组图能不能定出焦距"** —— 这才是单目真正该守的线：
   ``Z = f·D/(2r)``，fx 的相对误差 **1:1** 变成测深误差。
   所有姿态都近似正对时，fx 与距离互相抵消，**RMS 可以低到 0.5 而 fx 错一倍**。

**不够就抛 ``MonoCalibError``，不返回"勉强能用"的结果** —— 沿用项目既有的
"不可用不得保存"口径（``calib/stereo_calib.py`` 的 verdict 一脉）。
"""

from __future__ import annotations

import json
import re
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
    "session_label", "session_id", "camera_note", "CAMERA_SOURCES",
    "SCHEMA", "VERSION",
]

SCHEMA = "sphere-mono-calibration"
VERSION = 1

#: ``camera_source`` 的取值集合（机器可读枚举）：写进 JSON 供脚本比较，
#: 报告再把它翻成人话（见 :func:`report_text`）。不在集合里的一律落 "unknown"。
CAMERA_SOURCES = ("identity", "role-only", "unknown")

#: 质量门限的**唯一来源**（界面显示、报告文字、测试断言都从这里读）。
QUALITY = {
    # RMS 只分档，不单独否决。上限 0.8 与双目路径的 ``CalibConfig.ok_rms = 0.80``
    # **对齐同一个数**：单目凭什么比双目更严？双目都在 0.8 收工，单目原先把 0.5
    # 当否决线，纯粹是抄来的数字，没有依据（实测批次 4：RMS 0.587 但留一法 fx
    # 只飘 2.8%，比"通过"的批次 2 的 5.1% 还稳 —— 这条门限一直在误杀）。
    "rms_px_warn": 0.5,
    "rms_px_max": 0.8,
    # 单目真正该守的硬判据：留一法 fx 极差 ÷ 中位 fx。
    # 3% 是从"深度误差预算"反推的：0.6 m 处 ±7 mm 的预算 = 1.2%，3% 已经吃掉
    # 两倍半；8% 是"再差就不该拿它做定量"的线（8% 在 0.6 m 就是 48 mm）。
    "fx_spread_warn": 0.03,
    "fx_spread_max": 0.08,
    "min_views": 8,
    # 板面积门限与采集端**同源**（``CaptureConfig.min_area_ratio`` = 1.2%）。
    # 一开始这里自己写了 3%，结果采集端明明收下的图在标定端被判"板太小" ——
    # 同一概念两处门限，正是项目最忌讳的那种"第二份真相"。
    "min_area_ratio": float(CaptureConfig.min_area_ratio),
    "min_regions": 4,
    "regions_total": 5,
}

#: 留一法的时间预算。**判据再好，不能把一次标定从 5 秒拖成 5 分钟。**
#: 依据（2026-10-29 实测，15 张 1920×1080）：读图+检测角点约 5.4 s（357 ms/张，
#: 全分辨率 SB 检测是主要成本），单次求解约 50 ms。所以逐张留一（15 次求解）
#: 只多 0.7 s。真正的风险是"图越收越多"：60 张时单次求解 145 ms、检测 21 s，
#: 60 次求解要 8.4 s —— 仍可接受，但再翻几倍就会逼近 30 s，因此留两条闸。
LOO_BUDGET_S = 12.0        # 留一法自己的绝对上限（总耗时仍要 < 30 s）
LOO_EXTRA_RATIO = 2.0      # 相对上限：额外耗时 ≤ 原耗时×2（总 ≤ 原来的 3 倍）

#: 会话切分阈值：按文件名毫秒时间戳排序后，相邻两张间隔**超过**它就算两次采集。
#: 依据（2026-10-29 实测本机 4 批数据）：
#:   · 一次采集 15 张的时间跨度 = 34~57 s（四批分别 57/38/36/34 s）；
#:   · 两次采集之间人要挪板、改参数、重开程序，实测间隔 134 s / 183 s；
#: 60 s 落在"同一趟最慢也不超过"与"两趟最快也不低于"之间。
#: **为什么不用 `_capture.json` 判会话**：它每次采集都被覆盖，目录里只剩最后一次，
#: 拿它切会话等于把历史抹掉 —— 而"这个目录混了几次采集"恰恰是要查清的事。
SESSION_GAP_S = 60.0


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
    #: 留一法 fx 极差 ÷ 中位 fx（**小数**，0.028 = 2.8%）。判据直接读它。
    #: ``< 0`` 表示"没算"（视角太少，去掉一张就不够解算）。
    #: 老 json 里没有这个键 —— 默认 0.0 会读成"非常稳"，所以 from_dict 里
    #: 缺键时显式落 -1（见 ``_missing``）。
    fx_spread: float = 0.0
    #: 留一法是**逐张**还是**按区域**做的（"每张" / "按区域" / ""）。
    #: 两种模式算的是同一个量（fx 极差 ÷ 中位 fx），但"按区域"看不到区域内
    #: 单张坏图的影响，所以这个标识必须写进报告，不能让人以为是逐张。
    fx_spread_mode: str = ""
    #: 这批图是哪台相机拍的，**只放标识**（机器可读，配置里要能 diff/比较）：
    #: ``"aoni webcam A25 left | 0"`` 或 ``"left"`` 或 ``""``。
    #: **角色名不是身份**：``config/cameras.json`` 里 ``left`` 指向的设备可以换、
    #: 也可以离线后回退到别的索引（实测同一批 61 张 fx 在 1268~1457 之间差 15%，
    #: 却查不出是哪台拍的 —— 就是边车只记了 "left" 的代价）。
    camera: str = ""
    #: ``camera`` 的**可信度**（枚举，机器可读）：``identity`` = 有设备名+设备号，
    #: 能核对到物理设备；``role-only`` = 只有角色名，查不到是哪台；
    #: ``unknown`` = 连角色名都没有。
    #: 中文说明只进报告、不进配置 —— 配置存事实，报告讲人话。
    camera_source: str = "unknown"
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
            "fx_spread": self.fx_spread,
            "fx_spread_mode": self.fx_spread_mode,
            "camera": self.camera,
            "camera_source": self.camera_source,
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
            # 缺键必须落 -1（"没算过"）。落 0.0 等于谎报"fx 一点都没飘"，
            # 而老 json 恰恰全都没有这个键 —— 这种"默认值恰好是最乐观值"的坑，
            # 本项目踩过一次（`s²` 那次），所以这里显式区分"没算"与"算出来是 0"。
            fx_spread=(_f("fx_spread")
                       if isinstance(raw.get("fx_spread"), (int, float)) else -1.0),
            fx_spread_mode=str(raw.get("fx_spread_mode") or ""),
            camera=str(raw.get("camera") or ""),
            # 缺键落 "unknown"（不是 ""）：老 json 里既没有 camera 也没有来源，
            # 说"不知道"才是真话；落 "" 会变成一个既非枚举值又像个合法空值的坑。
            camera_source=(str(raw.get("camera_source") or "unknown")
                           if raw.get("camera_source") in CAMERA_SOURCES
                           else "unknown"),
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
# 会话切分（"这个目录是几次采集？"）
# --------------------------------------------------------------------------- #
#: 文件名里的 13 位毫秒时间戳（``shot_03_1793272978741.png``）。要求前后不是数字，
#: 免得把 15 位的纳秒串或 10 位秒级串截错。
_TS_RE = re.compile(r"(?<!\d)(\d{13})(?!\d)")


def _view_timestamp(path: Path) -> int | None:
    """从文件名里取毫秒时间戳；取不到返回 None（= 归入"未知"组）。"""
    m = _TS_RE.search(path.stem)
    return int(m.group(1)) if m else None


def session_label(ts_ms: int) -> str:
    """会话的**人读**标签：``2026-10-28 14:54``。"""
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts_ms / 1000.0))


def session_id(ts_ms: int) -> str:
    """会话的**可复制**编号（``--session`` 的取值）：``20261028-1454``。"""
    return time.strftime("%Y%m%d-%H%M", time.localtime(ts_ms / 1000.0))


def _split_sessions(paths: Sequence[Path]) -> tuple[list[dict], list[Path]]:
    """按毫秒时间戳把图分成"采集会话"，返回 (会话列表, 没有时间戳的图)。

    会话 = 按时间排序后相邻间隔 ≤ :data:`SESSION_GAP_S` 的一段。每条记录
    ``{start_ts, last_ts, paths}``，路径保持时间升序（标定结果不该依赖文件名顺序）。
    """
    stamped: list[tuple[int, Path]] = []
    unknown: list[Path] = []
    for p in paths:
        t = _view_timestamp(p)
        if t is None:
            unknown.append(p)
        else:
            stamped.append((t, p))
    stamped.sort(key=lambda tp: tp[0])
    sessions: list[dict] = []
    for t, p in stamped:
        if sessions and (t - sessions[-1]["last_ts"]) <= SESSION_GAP_S * 1000.0:
            sessions[-1]["paths"].append(p)
            sessions[-1]["last_ts"] = t
        else:
            sessions.append({"start_ts": t, "last_ts": t, "paths": [p]})
    return sessions, unknown


def _session_table(sessions: Sequence[dict], per_line: int = 2) -> str:
    """会话清单（每行 ``per_line`` 条），供拒绝信息与命令行报告共用。"""
    items = [f"{session_label(s['start_ts'])}  {len(s['paths']):2d} 张"
             for s in sessions]
    return "\n".join("  " + "    ".join(items[i:i + per_line])
                     for i in range(0, len(items), per_line))


def _pick_session(paths: Sequence[Path], session: str | None) -> list[Path]:
    """决定**这次标定用哪些图**：多会话默认拒，``--session`` 显式选一次。

    为什么默认拒（2026-10-29 实测，这不是理论风险）：``outputs/mono_calib_frames``
    里躺着 4 次采集、跨两天。60 张混着标 → fx=1369、留一法只有 **1.8%**（"看起来
    很稳"），而分批真实值是 1268/1268/1380/1388 —— **同一台机器上差 15%**。
    混标之所以"稳"，是因为它把 4 组不同的位姿混成一个大样本，抖动被平均掉了；
    这种"稳"是假的，它掩盖的正是要查的差异。

    没有时间戳的图归入"未知"组，任何选择都带上它 —— 它们无法归属，静默丢掉
    等于悄悄少算数据（本项目对"悄悄丢数据"的规矩是必须报告）。
    """
    sessions, unknown = _split_sessions(paths)
    if not sessions:                          # 全都解析不出时间戳：无从判断
        return list(paths)

    if session is None:
        if len(sessions) == 1:
            return list(sessions[0]["paths"]) + unknown
        extra = (f"\n  另外还有 {len(unknown)} 张没时间戳的图（会一起算进去）"
                 if unknown else "")
        raise MonoCalibError(
            f"这个目录里有 {len(sessions)} 次采集（不是一次）：\n"
            + _session_table(sessions) + extra + "\n"
            "  混在一起标会得到一个「看起来稳」的焦距（本机实测：60 张混标 "
            "fx=1369、留一法只有 1.8%），\n"
            "  但它们是不同时间、可能不同相机拍的，分批真实值在 1268~1457 之间差 15%。\n"
            f"  请选一次来标：--session last（最近一次）或 "
            f"--session {session_id(sessions[-1]['start_ts'])}")

    want = str(session).strip().lower()
    if want in ("last", "latest", "最近", "最新"):
        return list(sessions[-1]["paths"]) + unknown
    for s in sessions:
        sid = session_id(s["start_ts"])
        full = time.strftime("%Y%m%d-%H%M%S",
                             time.localtime(s["start_ts"] / 1000.0))
        if want == sid or full.startswith(want) or sid.startswith(want):
            return list(s["paths"]) + unknown
    raise MonoCalibError(
        f"没找到会话 {session!r}。这个目录里有 {len(sessions)} 次采集：\n"
        + _session_table(sessions)
        + f"\n  可用 --session last 或 --session {session_id(sessions[-1]['start_ts'])}")


# --------------------------------------------------------------------------- #
# 标定
# --------------------------------------------------------------------------- #
def _leave_one_out_fx(objp: np.ndarray, imgpoints: Sequence[np.ndarray],
                      image_size: tuple[int, int], cfg: CalibConfig, *,
                      region_of_view: Sequence[int], names: Sequence[str],
                      elapsed: float, t_solve: float
                      ) -> tuple[float, str, float, float]:
    """留一法：每次去掉一组视图重标，返回 fx 的（极差÷中位 fx, 模式, 最小, 最大）。

    为什么是这条而不是 RMS
    ----------------------
    单目测深 ``Z = f·D/(2r)``：**fx 的相对误差 1:1 就是深度误差**。而 RMS 是
    "所有角点的重投影残差"，它与 fx 能不能定出来**几乎不相关** —— 实测 2026-10-29
    的 4 批真实数据里，RMS 0.587 那批 fx 只飘 2.8%（比被判"通过"的 0.269 批的
    5.1% 还稳），RMS 0.269 那批反而飘 5.1%。拿 RMS 当否决线，挡的是错的人。

    为什么是"去掉一张"而不是只用残差
    --------------------------------
    焦距-距离歧义（所有姿态近似正对时 f 与 z 互相抵消）**在残差上完全看不出来**：
    残差小、内参可以错一倍。去掉一张重标，把"同一物理量在数据轻微变化下的抖动"
    暴露出来 —— 这才是"定得住 / 定不住"的直接证据。

    两种模式，同一个定义
    --------------------
    极差 ÷ 中位 fx，逐张与按区域**算的是同一个量**，不许出现两种口径。
    退化到按区域时少看到的是："某个区域内单张坏图"的影响会被同区域其它图摊平，
    所以它对"局部坏图"不敏感（对"整体姿态太相似"仍然敏感，那才是主要目标）。

    时间预算见 :data:`LOO_BUDGET_S` / :data:`LOO_EXTRA_RATIO`：角点只检测一次、
    这里只重复**求解**，所以 15 张的额外成本约 15×50 ms。
    """
    n = len(imgpoints)
    if n < 4:                       # 去掉一张后还要 ≥3 个视图才可解
        return -1.0, "", float("nan"), float("nan")

    budget = min(LOO_BUDGET_S, LOO_EXTRA_RATIO * max(elapsed, 1e-3))
    folds: list[list[int]]
    mode: str
    if max(t_solve, 1e-3) * n <= budget:
        folds = [[j for j in range(n) if j != i] for i in range(n)]
        mode = "逐张"
        fold_desc = [names[i] for i in range(n)]
    else:
        # 退化的原因只有一个：图太多，逐张重标会让"5 秒的标定"变成"几分钟"，
        # 而用户要的是判据不是跑分。按区域折 = 最多 5 次拟合，成本恒定。
        by_region: dict[int, list[int]] = {}
        for i, r in enumerate(region_of_view):
            if r >= 0:
                by_region.setdefault(int(r), []).append(i)
        folds = []
        fold_desc = []
        for r, idxs in sorted(by_region.items()):
            drop = set(idxs)
            rest = [j for j in range(n) if j not in drop]
            if len(rest) >= 4:      # 剩不到 4 个视图的折没有判别力，跳过
                folds.append(rest)
                fold_desc.append(f"区域 {r}（{len(idxs)} 张）")
        mode = "按区域"

    fxs: list[float] = []
    for k, sub in enumerate(folds):
        try:
            r = calibrate_mono_points(objp, [imgpoints[i] for i in sub],
                                      image_size, cfg)
        except cv2.error as exc:
            # 去掉这一组就解不出来 —— 这本身就是"焦距定不住"的最硬证据，
            # 不能当成"这次留一法没算"跳过（那会把最坏的情况报成"没问题"）。
            raise MonoCalibError(
                f"焦距不稳定：去掉{fold_desc[k]}后标定直接失败"
                f"（{str(exc).splitlines()[0][:80] if str(exc) else 'cv2.error'}）—— "
                f"这组图撑不住任何一张的变化，别用这组结果") from exc
        fx = float(np.asarray(r["K"], np.float64).reshape(3, 3)[0, 0])
        if np.isfinite(fx) and fx > 0:
            fxs.append(fx)

    if len(fxs) < 2:
        return -1.0, "", float("nan"), float("nan")
    med = float(np.median(fxs))
    if med <= 0:
        return -1.0, "", float("nan"), float("nan")
    return (max(fxs) - min(fxs)) / med, mode, min(fxs), max(fxs)


def calibrate_mono(image_paths: Iterable[str | Path], spec: BoardSpec, *,
                   fix_principal_point: bool = False,
                   rational_model: bool = False,
                   min_views: int | None = None,
                   rms_max: float | None = None,
                   min_regions: int | None = None,
                   fx_spread_max: float | None = None,
                   session: str | None = None,
                   camera: str = "",
                   camera_source: str = "unknown") -> MonoCalib:
    """从一组棋盘图标定单目内参。

    参数都给了默认值（来自 :data:`QUALITY`），测试时可以收紧/放开来验证判定本身
    是否真的会红 —— 这是"判据可自证"的要求，不是给生产用的旋钮。
    ``fx_spread_max=99`` 等于**关掉**焦距判据（反向自证用，见
    ``tests/selftest_mono_calib.py``）。
    ``camera`` 只做记录（"这批是谁拍的"），不参与判定。

    ``session``：目录里混了多次采集时选哪一次 —— ``"last"`` 取最近一次，
    ``"20261029-1922"`` 按起始时间匹配（可只给到分钟）。不给且检出 ≥2 次采集
    → 直接拒（理由见 :func:`_pick_session`）。

    抛 :class:`MonoCalibError`（带 ``rejected`` 明细）表示**这组图不可用**；
    其他异常（读图 OSError 之类）原样上抛。
    """
    t_start = time.perf_counter()
    paths = [Path(p) for p in image_paths]
    if not paths:
        raise MonoCalibError("没有图片")
    # 会话守卫在读图**之前**：混标一次要白跑 27 秒，而且会给出一个假的"稳定"。
    paths = _pick_session(paths, session)
    need = QUALITY["min_views"] if min_views is None else int(min_views)
    rms_limit = QUALITY["rms_px_max"] if rms_max is None else float(rms_max)
    regions_need = (QUALITY["min_regions"] if min_regions is None
                    else int(min_regions))
    fx_limit = (QUALITY["fx_spread_max"] if fx_spread_max is None
                else float(fx_spread_max))

    objp = _object_points(spec)
    imgpoints: list[np.ndarray] = []
    meta: list[dict] = []
    rejected: list[tuple[str, str]] = []
    image_size: tuple[int, int] | None = None

    # ---- 检测一次，之后复用 ------------------------------------------------
    # 留一法只重复**求解**、不重复 imread+detect_board：全分辨率 SB 检测约
    # 357 ms/张（2026-10-29 实测 60 张 21.4 s），15 张重测一遍就是 5 s，
    # 而重复求解只要 50 ms/次。把"检测"和"求解"分开是这条判据能落地的前提。
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

    cfg = CalibConfig(fix_principal_point=fix_principal_point,
                      rational_model=rational_model)

    def _unpack(r: dict):
        per = np.asarray(r["per_view"], np.float64).ravel()
        return (np.asarray(r["K"], np.float64).reshape(3, 3),
                np.asarray(r["D"], np.float64).ravel(),
                float(r["rms"]), per)

    t0 = time.perf_counter()
    K, D, rms, per_view_arr = _unpack(calibrate_mono_points(
        objp, imgpoints, image_size, cfg))
    t_solve = time.perf_counter() - t0

    # ---- 剔除离群观测后重标 ------------------------------------------------
    # 为什么必须做：RMS 是**所有点一起去平均**，一张 5 px 的坏图（角点检错/
    # 轻微遮挡）就能把 24 张 0.3 px 的好图拉到 1.1 px，于是整组被判"不可用"。
    # 实测 2026-10-28：30 张中位数 0.32 px、最差一张 5.24 px，全用 RMS=1.10，
    # 剔掉 6 张后 RMS=0.271。**这不是放宽门限**（门限仍是 0.8），
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
            objp, [imgpoints[i] for i in keep_idx], image_size, cfg)
        K, D, rms, per_view_arr = _unpack(res2)
        meta = [meta[i] for i in keep_idx]
        imgpoints = [imgpoints[i] for i in keep_idx]
        kept_paths = [paths[i] for i in keep_idx]
    else:
        kept_paths = paths
    per_view = tuple(float(x) for x in per_view_arr)

    # 姿态覆盖：板中心落在哪些区域（留一法的"按区域"折也要用这份映射）
    region_of_view = [_region_index(m["center"][0], m["center"][1],
                                    image_size[0], image_size[1])
                      for m in meta]
    regions: dict[int, int] = {}
    for idx in region_of_view:
        if idx >= 0:
            regions[idx] = regions.get(idx, 0) + 1

    # ---- 留一法：fx 到底定不定得住 -----------------------------------------
    # 放在覆盖检查之前算：即使这组图因其它原因被拒，"fx 飘多少"也要能报出来
    # （诊断表要的是全部数字，不是第一道闸的数字）。成本约 n×50 ms。
    fx_spread, fx_mode, fx_lo, fx_hi = _leave_one_out_fx(
        objp, imgpoints, image_size, cfg,
        region_of_view=region_of_view,
        names=[m["name"] for m in meta],
        elapsed=time.perf_counter() - t_start,
        t_solve=t_solve)

    # ---- 判定：按"实际失败的那一条"说话 -----------------------------------
    if len(regions) < regions_need:
        raise MonoCalibError(
            f"姿态只覆盖了 {len(regions)}/{QUALITY['regions_total']} 个区域"
            f"（至少 {regions_need} 个）—— "
            f"板要摆到画面四角与中央，全挤在中间会让畸变系数失真", rejected)

    if rms > rms_limit:
        # 具体点出最差的三张：只说"图糊、板太小或角点检错"等于让用户自己猜。
        worst = sorted(zip(per_view, [m["name"] for m in meta]))[::-1][:3]
        detail = "、".join(f"{n} {r:.2f} px" for r, n in worst) or "（无逐图数据）"
        raise MonoCalibError(
            f"重投影 RMS {rms:.2f} px 超过 {rms_limit:.2f} px。\n"
            f"  最差的 3 张：{detail}\n"
            f"  · 先看这几张：有没有拖影、反光、板边被裁掉、棋盘被手挡住\n"
            f"  · RMS 是「所有角点一起平均」的残差，它高只说明有点没对齐；\n"
            f"    焦距稳不稳另有一条判据（留一法 fx 极差），这里先修角点",
            rejected)

    if fx_spread >= 0 and fx_spread > fx_limit:
        verb = ("去掉任意一张" if fx_mode == "逐张"
                else "每次去掉一个区域")
        raise MonoCalibError(
            f"焦距不稳定：{verb}后 fx 在 {fx_lo:.0f}~{fx_hi:.0f} 之间"
            f"（{fx_spread:.1%}），超过 {fx_limit:.0%}\n"
            f"  · 说明这组图的姿态太相似（多半都接近正对），fx 与距离互相抵消\n"
            f"  · 建议：同一距离下拍 3~5 张明显倾斜的（±20° 以上），而不是只平移\n"
            f"  · 单目测深 Z=f·D/(2r)，fx 飘 {fx_spread:.1%} 就是深度飘 "
            f"{fx_spread:.1%}，这组结果拿去做定量会系统性偏",
            rejected)

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
        fx_spread=fx_spread,
        fx_spread_mode=fx_mode,
        camera=str(camera),
        camera_source=(str(camera_source) if camera_source in CAMERA_SOURCES
                       else "unknown"),
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


def camera_note(camera: str, source: str) -> str:
    """``camera`` + ``camera_source`` → **人话**（报告与命令行共用同一句）。

    这两个东西的可信度差一个量级，措辞必须一样，否则命令行说"是这台"、
    报告说"不知道"——"同一件事两处各写一份"正是本项目最忌讳的那种矛盾。
    """
    if source == "identity" and camera:
        return f"{camera}（设备身份可核对）"
    if source == "role-only":
        return (f"{camera or '?'}（旧边车只记了角色名，"
                f"无法核对是哪台物理设备）")
    if camera:
        return f"{camera}（来源不明）"
    return "未知（没有 _capture.json，或边车没记身份）"


def _camera_line(calib: MonoCalib) -> str:
    """报告里那一行：配置存事实，报告讲人话。"""
    return "  相机：" + camera_note(calib.camera, calib.camera_source)


def report_text(calib: MonoCalib) -> str:
    """人可读报告：一行结论 + 关键数字 + 覆盖情况 + **两条判据各自的档位**。"""
    cov = calib.pose_coverage or {}
    rms_warn = float(QUALITY["rms_px_warn"])
    rms_max = float(QUALITY["rms_px_max"])
    fx_warn = float(QUALITY["fx_spread_warn"])
    fx_max = float(QUALITY["fx_spread_max"])
    lines = [
        f"单目内参标定 · {calib.views} 个有效视角，RMS {calib.rms_px:.3f} px",
        f"  fx = {calib.fx:.1f} px，fy = {calib.fy:.1f} px",
        f"  主点 = ({calib.cx:.1f}, {calib.cy:.1f})，图 {calib.image_size[0]}×{calib.image_size[1]}",
        f"  畸变 k1={calib.k1:+.4f} k2={calib.k2:+.4f} p1={calib.p1:+.4f} "
        f"p2={calib.p2:+.4f} k3={calib.k3:+.4f}"
        + ("（无畸变，去畸变是空操作）" if not calib.has_distortion else ""),
        _camera_line(calib),
        *([f"  已剔除离群观测 {len(calib.outliers)} 张："
           + "、".join(f"{n}({r:.1f}px)" for n, r in calib.outliers[:4])
           + (f" 等 {len(calib.outliers)} 张" if len(calib.outliers) > 4 else "")]
          if calib.outliers else []),
        f"  姿态覆盖 {cov.get('count', '?')}/{cov.get('total', QUALITY['regions_total'])} 个区域"
        f"，板 {calib.board.get('cols', '?')}×{calib.board.get('rows', '?')}"
        f"，方格 {calib.board.get('square_mm', '?')} mm",
    ]
    if calib.fx_spread >= 0:
        lines.append(f"  焦距稳定性：留一法 fx 极差 {calib.fx_spread:.1%}"
                     + (f"（{calib.fx_spread_mode}）" if calib.fx_spread_mode else "")
                     + f"，门限 {fx_warn:.0%} 警告 / {fx_max:.0%} 拒")
    else:
        lines.append("  焦距稳定性：留一法未计算（视角太少，去掉一组就不够解算）")
    # 0.5~0.8 不拒，但必须留一行痕迹 —— "能用但有瑕疵"和"好用"是两件事。
    if rms_warn < calib.rms_px <= rms_max:
        if calib.fx_spread < 0:
            lines.append(f"  [注意] RMS {calib.rms_px:.2f} px 偏软（> {rms_warn:.1f}），"
                         f"焦距稳定性没算出来，可用但别做定量")
        elif calib.fx_spread <= fx_warn:
            lines.append(f"  [注意] RMS {calib.rms_px:.2f} px 偏软（> {rms_warn:.1f}），"
                         f"但焦距稳定（{calib.fx_spread:.1%}），可用")
        else:
            lines.append(f"  [注意] RMS {calib.rms_px:.2f} px 偏软，"
                         f"且焦距也在飘（{calib.fx_spread:.1%}），可用但别做定量")
    elif calib.fx_spread > fx_warn:
        lines.append(f"  [注意] 焦距偏飘（{calib.fx_spread:.1%} > {fx_warn:.0%}），"
                     f"深度误差会同比例放大，定量用途加倾斜重拍")
    ok = calib.rms_px <= rms_max and (calib.fx_spread < 0 or calib.fx_spread <= fx_max)
    lines.append("  判定：可用 —— 满足三维重建对单目的要求" if ok
                 else "  判定：不可用（多半是老判据下存的旧文件，建议重标）")
    return "\n".join(lines)
