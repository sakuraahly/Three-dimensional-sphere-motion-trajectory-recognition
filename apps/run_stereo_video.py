"""离线双目：读一个 `data/sessions/<id>`，输出三维轨迹 CSV + 标注视频。

它不是"新的视觉算法"，而是把已有的三块拼起来：

    左路 SpherePipeline（检测 + 跟踪）
    右路 SpherePipeline（检测 + 跟踪）
    StereoReconstructor（平行：极线配对 + 三角测量）
    或 AngledReconstructor（大角度：F 极线 + 原始射线交会）

为什么先写离线入口
------------------
实时双目受"两路没有硬件同步"限制，而且相机可用性会漂；离线录制已经把
帧固定下来，可以反复调检测/配对参数、检查残差，是当前最该先跑通的一步。

用法::

    python apps/run_stereo_video.py data/sessions/20261019-192530
    python apps/run_stereo_video.py data/sessions/20261019-192530 --method hough \
        --min-radius 6 --max-radius 15 --detect-width 960

输出（会话目录下）::

    derived/traj3d.csv        逐帧：matched, t, X/Y/Z, 左右像素, 极线/交会残差
    derived/annotated.mp4     左右并排 + 检测圈（--no-video 可关）
    derived/traj3d_world.csv  同上列名，坐标换成世界系（有 world_frame.json 时）
    derived/predictions.csv   t_ref,horizon_s,x,y,z（弹道外推；--no-predict 可关）

⚠️ 默认检测器是 `color`，适合**哑光纯色球**。金属/高光球（例如摆锤）用
`color` 会锁到背景高光；可试 `--method hough --min-radius 6 --max-radius 15`，
但真实纹理背景下 Hough 仍有误检 —— 这是检测问题，不是三角测量问题。

例外（会自动处理）：会话里有 `derived/seeds.json`（GUI『点选球心』留下的种子）时，
只要**没显式给 `--method`**，就自动改用 `specular` + 该种子。镜面球在 `color` 下
恒为 0 检出，若不自动接管，用户看到的会是"0 配对 + 装置夹角"这种指错方向的结论。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "sphere_tracker").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calib import StereoRig, WorldFrame
from calib.params import is_uncorrected                                  # noqa: E402
from sphere_tracker.config import AppConfig, ColorConfig     # noqa: E402
from sphere_tracker.detectors import registry                # noqa: E402,F401  （import 即注册）
from sphere_tracker.geometry import CameraModel              # noqa: E402
from sphere_tracker.pipeline import SpherePipeline           # noqa: E402
from sphere_tracker.plugins import default as default_method, names as method_names  # noqa: E402
from sphere_tracker.plugins import describe                  # noqa: E402
from sphere_tracker.types import Detection                   # noqa: E402
from stereo import (  # noqa: E402
    SMOOTH_METHODS, MatchConfig, PairSource, PredictConfig, StereoReconstructor,
    fit_ballistic, geometry_verdict, jitter_rms, predict, smooth_trajectory,
)
from stereo.stereo_mode import (  # noqa: E402
    mode_label, optical_axes_angle_deg, resolve_auto,
)
from stereo.triangulate import (  # noqa: E402
    EPI_GATE_K,
    EPI_GATE_MAX_PX,
    EPI_GATE_MIN_PX,
    GEOM_MIN_SAMPLES,       # 几何守卫的样本下限
    anchor_problem,         # 人工锚点自身可信吗
    build_radius_anchor,    # 一次人工点选 → 左右半径比期望值
    describe_anchor,
    epipolar_gate_px,       # `--max-epipolar auto`：按标定自报极线 RMS 定门限
)
from stereo.visualization import (  # noqa: E402
    export_session_visualization,
    load_seeds,
    write_seeds,
)
from stereo.sessions import (  # noqa: E402
    load_session, save_session, validate_session,
)


def _as_detection(res) -> Detection | None:
    """优先用跟踪后的圆心；跟踪在滑行时退回原始检测。"""
    state = res.track
    if state is not None and not state.coasting:
        r = res.detection.radius if res.detection is not None else state.radius
        return Detection(x=state.x, y=state.y, radius=r, score=1.0, method="track")
    return res.detection


def _make_pipeline(cfg: AppConfig, rig: StereoRig, side: str) -> SpherePipeline:
    pipe = SpherePipeline(cfg)
    K = rig.K1 if side == "left" else rig.K2
    pipe.camera = CameraModel(
        width=int(rig.image_size[0]), height=int(rig.image_size[1]),
        fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]),
        # 与 `AppConfig.camera.ball_diameter_m` 同口径：有效直径（名义 25 mm 的球
        # 在 specular 下反推约 21.6 mm，见交接文档 §35.2）。这个分支不参与三角化，
        # 但两处填不同的数会让"单目读数"随调用路径变，属于最难查的那类漂移。
        ball_diameter_m=0.0216,
    )
    return pipe


def _mode_guard(mode: str, rectification: str) -> tuple[str, bool]:
    """**兼容旧/错 meta**：记成 parallel 但标定其实是 none_raw（未校正）时改判 angled。

    为什么需要：chain 产物（`R1=R2=I`、`P1=K1·[I|0]`、`P2=K2·[R|T]`）从没做过
    极线校正，而 parallel 重建的 dy 门限与视差范围都建立在校正图上 —— 90° 装置
    原始像素的 |dy| 可达几十~几百 px，结果是**全部候选被拒、配对率 0、不报错**。
    历史标定文件已经这么写过，所以这里按标定产出的口径改判，而不是信 meta。

    返回 ``(实际使用的模式, 是否发生了改判)``。纯函数，便于自测。
    """
    if mode == "parallel" and is_uncorrected(rectification):
        return "angled", True
    return mode, False


def _resolve_max_epipolar(text, rig, mode: str) -> tuple[float, str]:
    """把 ``--max-epipolar`` 解析成像素门限，返回 ``(门限, 要打印的一行说明或 "")``。

    **纯函数**（除 ``rig.quality`` 之外不读任何东西）。三种输入：

    - ``None``：给模式默认值（parallel 2.5 / angled 8），不打说明；
    - ``"auto"``：只有 parallel（已极线校正）才有意义 —— 那时左右对应点的 y 差
      本该是亚像素级，门限取"标定自报极线 RMS × ``EPI_GATE_K``"是有依据的。
      angled 用**原始像素**、极线不水平，这个口径不成立，所以照旧用默认值并说明。
      拿到了 ``auto`` 但标定没给可用 RMS 时，:func:`stereo.triangulate.
      epipolar_gate_px` 会回退 2.5 —— 这里把"到底是算出来的还是回退的"讲清楚，
      否则用户以为门限随标定走了，实际是常量。
    - 数字或数字字符串：原样用（显式值必须被尊重）；读不懂就回退默认值并警告，
      **不抛** —— 一个笔误不该让整段录像跑不起来。
    """
    default = 8.0 if mode == "angled" else 2.5
    if text is None:
        return default, ""
    if isinstance(text, str) and text.strip().lower() == "auto":
        if mode != "parallel":
            return default, (
                f"[门限] --max-epipolar auto 只在 parallel（已极线校正）下有意义；"
                f"{mode} 用原始像素、极线不水平 → 仍用 {default:g}px")
        quality = getattr(rig, "quality", None)
        gate = epipolar_gate_px(quality)
        rms = quality.get("epipolar_rms") if isinstance(quality, dict) else None
        if isinstance(rms, (int, float)) and math.isfinite(rms) and rms > 0:
            src = f"标定自报极线 RMS {rms:g}px × {EPI_GATE_K:g}"
        else:
            src = "标定没给出可用的 epipolar_rms → 回退 2.5px"
        return gate, f"[门限] --max-epipolar auto → {gate:g}px（{src}）"
    try:
        return float(text), ""
    except (TypeError, ValueError):
        return default, f"[警告] --max-epipolar 读不懂：{text!r} → 用默认 {default:g}px"


def _radius_anchor_from_session(args, seeds_path, rig):
    """读会话里那一对**人工点选**，算出 ``RadiusAnchor``。返回 ``(anchor, 原因)``。

    失败一律返回 ``None`` + 一句人话，由调用处决定退出码 —— 这里不抛异常，
    是因为"没点选过"是最常见的情况，用 traceback 告诉用户等于没说。

    三种拒绝都必须是硬拒绝（不是警告）：锚点一旦用了错的数据，它给的比值会
    **静默地**把后面每一帧的配对都筛一遍，结果"看着能跑、其实一直在丢真目标"，
    这正是本项目最怕的失效形态。
    """
    path = Path(seeds_path)
    if not path.is_file():
        return None, (
            f"没有种子文件：{path}\n"
            "  锚点要用**人工点选**的那一对球心。做法：GUI 双目窗 →「点选球心」"
            "（左右各点一次）→ 再回来跑。\n"
            "  不想点选就别开这个门：去掉 --anchor-gate on（默认就是 off）。")
    try:
        seeds = load_seeds(path)
    except (FileNotFoundError, ValueError) as exc:
        return None, f"读不了种子文件：{exc}"

    src = str(seeds.get("source") or "")
    if src == "auto-yolo":
        return None, (
            "这份 seeds.json 是 --auto-seed 自动定位的（source=auto-yolo），"
            "它的半径来自 YOLO 框、**不可信**（见待办 B-130）。\n"
            "  锚点只能用人点的那一份：GUI「点选球心」。")

    d_mm = float(args.anchor_diameter_mm)
    d_note = "命令行 --anchor-diameter-mm"
    if not (d_mm > 0):
        d_mm = 21.6
        d_note = "内置默认 21.6mm（没给 --anchor-diameter-mm）"
    try:
        anchor = build_radius_anchor(
            rig,
            (seeds["left"]["x"], seeds["left"]["y"]),
            (seeds["right"]["x"], seeds["right"]["y"]),
            d_mm / 1000.0)
    except ValueError as exc:
        return None, f"锚点算不出来：{exc}"

    prob = anchor_problem(anchor)
    if prob:
        return None, prob
    print(f"[锚点] 球径用 {d_mm:g}mm（{d_note}）；种子来源 {src or '未知'}")
    return anchor, ""


def _guard_step(armed: bool, guard_at: int, n_pairs: int, dy_samples, quality):
    """几何守卫的一步判定。**纯函数、无副作用**：只回答"现在能不能判、判到什么"。

    返回 ``(是否仍在等待判读, 判定或 None)``：

    - 没武装 / 还没跑够 ``guard_at`` 对 → ``(armed, None)``（保持原状，不消耗机会）；
    - 跑够了，但 :func:`stereo.triangulate.geometry_verdict` 返回 ``None``
      （样本不足 ``GEOM_MIN_SAMPLES``）→ ``(True, None)``：**不消耗这次机会**，
      下一帧接着看 —— 原来的 ``n_pairs == guard_at`` 写法在丢帧多时会一次也判不成，
      然后永久放弃，那是一条假阴性路径；
    - 判出来了 → ``(False, v)``，``v.suspect`` 由调用处决定是否早停。
    """
    if not armed or n_pairs < guard_at:
        return armed, None
    v = geometry_verdict(dy_samples, quality)
    if v is None:
        return True, None
    return False, v


def _auto_seed_pair(cfg: AppConfig, src: PairSource, max_try: int = 30):
    """不点选：用 YOLO 在最早的若干**成对**帧上找球，返回 ``(seed_l, seed_r, 帧号)``。

    **成对**是关键：种子必须左右两路来自**同一时刻**，否则 specular 的 CSRT 会在
    左右各初始化到不同时刻球所在的位置上，三角化出来的点从一开始就错位 —— 而且
    错得不像 bug，像"标定不准"。所以这里用 ``PairSource.read()``（它保证两路配对），
    不用分别拿两个 VideoCapture 各读一帧。

    找不到就返回 ``None`` 并**说明卡在哪**（没球 / 权重加载不了 / 越过视频末尾）——
    调用处据此返回码 2，而不是拿一个编出来的圆心硬跑。

    返回的半径来自 YOLO 框（框是外接矩形），**位置可信、半径不可信**；调用处会据此
    写 seed 并把"半径仍来自框"讲给用户。
    """
    from sphere_tracker.detectors.factory import build_single_detector

    det = build_single_detector("yolo", cfg.detector)
    if det is None:
        print("[自动种子] 没有可用的 yolo 检测器实现")
        return None
    for i in range(max_try):
        ok, fl, fr = src.read()
        if not ok or fl is None or fr is None:
            print(f"[自动种子] 只读到 {i} 对就越过视频末尾，没找到球")
            return None
        cand_l, cand_r = det.detect(fl), det.detect(fr)
        if not cand_l or not cand_r:
            continue
        a, b = cand_l[0], cand_r[0]
        print(f"[自动种子] 第 {i} 对命中：左 {a.method} {a.score:.2f} "
              f"({a.x:.1f},{a.y:.1f},r={a.radius:.1f})；"
              f"右 {b.method} {b.score:.2f} ({b.x:.1f},{b.y:.1f},r={b.radius:.1f})")
        if a.method == "yolo" or b.method == "yolo":
            print("  ⚠ 至少一路的圆心**没钉到轮廓上**（method=yolo = 退回框中心）："
                  "位置可用于初始化，但偏差比轮廓圆心大；半径来自框，不可信。")
        return (a.x, a.y, a.radius), (b.x, b.y, b.radius), i
    err = getattr(det, "last_error", None)
    print(f"[自动种子] 前 {max_try} 对都没在两路同时找到球："
          "确认球在画面里、且 yolo 权重能被加载（pip install ultralytics）。"
          + (f"\n  检测器最后报错：{err}" if err else ""))
    return None


def _pick_detector(method: str, method_given: bool, seeds_path: Path,
                   detect_width: int, auto_seed: bool = False) -> tuple[str, str]:
    """决定这次用哪个检测器，返回 ``(方法, 要打印的一行说明)``。**纯函数**（只读种子文件）。

    为什么"有种子就换 specular"必须做在这里、而且判据是"用户有没有显式给 --method"：

    - 镜面球在 `color` / `hough` 下恒为 0 检出（交接文档 §39.3：40 帧 0 检出），
      而会话里就存着 GUI 点选过的 `derived/seeds.json`。不接管的话，用户得到的
      结论是"0 配对 + 装置夹角可能太大" —— 方向完全是错的。
    - 判据必须是"用户没给 `--method`"，不能写成"值等于默认值"：argparse 里两者
      无法区分（`default="color"` 与显式 `--method color` 得到同一个值）。所以
      参数定义用 `default=argparse.SUPPRESS`，让属性只在显式给定时才存在。
      这比扫 `sys.argv` 可靠 —— argparse 允许前缀缩写（`--meth color`）与
      `--method=color` 两种写法，手写扫描得自己补全，容易漏。
    - 显式 `--method color` 时 **不换**：用户明确指定了就照做，悄悄换掉会让人
      怀疑"我明明说了用 color"。
    - `detect_width` 非 0 时不换：specular 的种子是原图坐标，缩放会错位，
      换了只会一路撞到后面"specular 暂不支持 --detect-width"的报错。
    - 种子文件坏了**不抛**：退回原方法并说明。一个坏 json 不该让整条链路崩掉，
      而且此时用户需要的是"种子坏了"这句人话，不是一段 traceback。
    """
    if auto_seed and not method_given and not detect_width:
        return "specular", ("[自动种子] --auto-seed → 用 specular + 自动定位的球心"
                            "（不点选；半径仍来自 YOLO 框，不可信）")
    if method_given or detect_width or not seeds_path.exists():
        return method, ""
    try:
        payload = load_seeds(seeds_path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return method, f"[警告] {seeds_path} 读不出来（{exc}），继续用 {method} 检测器"
    # 半径出身要在**每次**接管时复述，不能只在 --auto-seed 那一次说（独立 Critic 指出）：
    # 自动种子把"半径来自 YOLO 框、不可信"写进了 seeds.json，但重跑时走的是这条
    # 分支、原先一个字都不提 —— 那是一次静默的告知退化，而"不可信半径"恰好是
    # 本项目最容易被当成"标定不准"的东西。
    shown = ""
    if isinstance(payload, dict) and payload.get("source") == "auto-yolo":
        shown = "；⚠ 这份种子是 --auto-seed 自动定位的，半径来自 YOLO 框、不可信"
    return "specular", (
        "[种子] 会话里有 derived/seeds.json → 自动改用 specular + 该种子"
        f"（镜面球用 color/hough 检不到；要覆盖请显式 --method）{shown}")


def _world_block(world: WorldFrame | None, arr, base_frame: str) -> tuple[np.ndarray | None, dict]:
    """左相机系 -> 世界系。**纯函数**：不读文件、不猜模式，坐标系由调用处给出。

    `base_frame` 是"没有世界系时这次的点到底在哪个系"：
    ``parallel`` -> ``"rectified_left_camera"``（校正后左相机系），
    ``angled`` -> ``"left_camera"``（原始左相机系）。写错比不写更危险 ——
    下游会按 `left_camera` 的 K/D 去解释校正系的点。

    返回 ``(坐标已换成世界系的数组或 None, 写进 traj3d.json 的坐标系块)``：

    - ``world is None``：数组返回 ``None``，块里如实写 ``base_frame``，
      **不冒充世界系**；
    - ``world`` 给了：``traj3d.csv`` 那份坐标不动，这里只产出
      用于 ``derived/traj3d_world.csv`` 的副本，坐标系记 ``"world"``。

    输入 ``arr`` 是 ``(N,4)`` 的 ``[t, X, Y, Z]``（``rec.as_array()`` 的布局）或
    ``(N,3)`` 的纯点；时间列原样保留。
    """
    if world is None:
        return None, {"coordinate_system": base_frame, "world_frame": None}
    block = {
        "coordinate_system": "world",
        "world_frame": {"source": world.source, "note": world.note,
                        "created_at": world.created_at},
    }
    if arr is None:
        return None, block
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] not in (3, 4):
        raise ValueError(f"_world_block 需要 (N,3)/(N,4) 的点，实际 shape={a.shape}")
    out = a.copy()
    if a.shape[0]:
        if a.shape[1] == 4:
            out[:, 1:4] = world.apply(a[:, 1:4])
        else:
            out[:, 0:3] = world.apply(a)
    return out, block


def _predict_block(arr, cfg: PredictConfig) -> tuple[list[dict], dict]:
    """对一条三维轨迹做弹道拟合 + 解析外推。**纯函数**：不读文件。

    返回 ``(rows, block)``：``rows`` 每个 horizon 一条
    ``{t_ref, horizon_s, x, y, z}``，直接写 ``derived/predictions.csv``；
    ``block`` 是 ``traj3d.json`` 里的 ``predictions`` 内容。

    拟合不合格（样本不足 / 残差超门限）时 ``rows`` 为空、``block`` 只带
    ``skipped`` 原因 —— **绝不返回硬凑的预测值**。
    """
    a = (np.zeros((0, 4), dtype=np.float64) if arr is None
         else np.asarray(arr, dtype=np.float64))
    if a.ndim != 2 or a.shape[1] != 4:
        raise ValueError(f"_predict_block 需要 (N,4) 的 [t,X,Y,Z]，实际 shape={a.shape}")
    if a.shape[0] == 0:
        return [], {"skipped": "没有可用的三维点"}
    why: list[str] = []
    state = fit_ballistic(a[:, 0], a[:, 1:4], cfg, why)
    if state is None:
        return [], {"skipped": why[0] if why else "弹道拟合失败"}
    pred = predict(state, cfg.horizons)
    rows: list[dict] = []
    for h in cfg.horizons:
        p = pred[float(h)]
        rows.append({"t_ref": float(state.t_ref), "horizon_s": float(h),
                     "x": float(p[0]), "y": float(p[1]), "z": float(p[2])})
    block = {
        "t_ref": float(state.t_ref),
        "state": {"p0": [float(v) for v in state.p0],
                  "v0": [float(v) for v in state.v0],
                  "rms_m": float(state.rms_m), "n": int(state.n)},
        "points": [{"horizon_s": r["horizon_s"], "x": r["x"],
                    "y": r["y"], "z": r["z"]} for r in rows],
    }
    return rows, block


def _project_left(rig, xyz, mode: str = "parallel") -> tuple[int, int] | None:
    """把左相机系的三维点投到**原始左图**；mode 决定是否用 R1。"""
    try:
        p = np.asarray(xyz, np.float64).reshape(1, 3)
        if mode == "angled":
            rvec = np.zeros((3, 1))          # 原始左相机系，不做 rectified 旋转
        else:
            rvec, _ = cv2.Rodrigues(rig.R1.T)
        uv, _ = cv2.projectPoints(p, rvec, np.zeros(3), rig.K1, rig.D1)
        return int(round(float(uv[0, 0, 0]))), int(round(float(uv[0, 0, 1])))
    except Exception:                                       # noqa: BLE001
        return None


def _draw_reference(frame: np.ndarray, rig, origin: tuple[float, float, float],
                    axis_m: float = 0.10, mode: str = "parallel") -> None:
    """在左图标出坐标原点 + X/Y/Z 轴（原点默认 = 相机光心投影）。"""
    o = _project_left(rig, origin, mode)
    if o is None:
        return
    h, w = frame.shape[:2]
    if not (0 <= o[0] < w and 0 <= o[1] < h):
        return
    axes = [
        ((origin[0] + axis_m, origin[1], origin[2]), (0, 0, 255), "X"),
        ((origin[0], origin[1] + axis_m, origin[2]), (0, 255, 0), "Y"),
        ((origin[0], origin[1], origin[2] + axis_m), (255, 0, 0), "Z"),
    ]
    cv2.circle(frame, o, 7, (0, 255, 255), -1)
    cv2.putText(frame, "O left-cam", (o[0] + 8, o[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    for xyz, color, name in axes:
        p = _project_left(rig, xyz, mode)
        if p is None:
            continue
        # 相机光心附近的坐标轴端点可能投到画面外/数值爆炸；跳过，别画到画布外
        if not (0 <= p[0] < w and 0 <= p[1] < h):
            continue
        cv2.arrowedLine(frame, o, p, color, 3, tipLength=0.18)
        cv2.putText(frame, name, (p[0] + 4, p[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


def _draw_xyz(frame: np.ndarray, res, xyz, label: str) -> None:
    d = res.detection
    if d is None:
        return
    if xyz is None:
        text = f"{label} no 3D"
        color = (0, 0, 255)
    else:
        text = f"{label} X={xyz[0]:.3f} Y={xyz[1]:.3f} Z={xyz[2]:.3f} m"
        color = (0, 255, 255)
    org = (int(round(d.x)) + 12, int(round(d.y)) - 12)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)


def _draw(frame: np.ndarray, res, label: str, trail: list[tuple[float, float]] | None = None,
          *, xyz=None, rig=None, origin=None, draw_reference: bool = False,
          mode: str = "parallel") -> np.ndarray:
    out = frame.copy()
    if draw_reference and rig is not None and origin is not None:
        _draw_reference(out, rig, origin, mode=mode)
    if trail and len(trail) >= 2:
        pts = np.array([[int(round(x)), int(round(y))] for x, y in trail], np.int32)
        cv2.polylines(out, [pts], False, (0, 255, 255), 2)
    d = res.detection
    if d is not None:
        cv2.circle(out, (int(round(d.x)), int(round(d.y))), int(round(d.radius)),
                   (0, 0, 255), 2)
        cv2.putText(out, f"{label} r={d.radius:.1f} {d.method}", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        _draw_xyz(out, res, xyz, label)
    else:
        cv2.putText(out, f"{label} no detection", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return out


def _plot_trajectory(arr: np.ndarray, path: Path, smoothed: np.ndarray | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(arr[:, 1], arr[:, 2], arr[:, 3], ".-", color="#2563eb",
            markersize=2, linewidth=1, label="raw")
    if smoothed is not None:
        ax.plot(smoothed[:, 0], smoothed[:, 1], smoothed[:, 2], "-",
                color="#f97316", linewidth=1.6, label="S-G")
    ax.set_xlabel("X right (m)")
    ax.set_ylabel("Y down (m)")
    ax.set_zlabel("Z forward (m)")
    ax.set_title("Stereo 3D trajectory")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


#: `traj3d.csv` 的列顺序（**唯一来源**）。写入时必须每一行都等长，
#: 否则 `csv.DictReader` 会把后面的列整体左移而不报错 —— 2026-10-30 实测过：
#: 少写一个 `confident`，读出来就是 `method_left` 的字符串（"track"），
#: 于是"按 confident==0 过滤"永远得到 0 行，且没人能看出问题在哪。
CSV_HEADER = ["frame", "t", "matched", "X", "Y", "Z", "u_left", "v_left",
              "u_right", "v_right", "epipolar_px", "residual_px", "confident",
              "method_left", "method_right"]


def row_unmatched(frame: int, t: float, det_l, det_r) -> list:
    """未配对帧的一行 CSV。**长度必须等于 `CSV_HEADER`**（有判据守着）。"""
    return [frame, f"{t:.4f}", 0, "", "", "", "", "", "", "", "", 0, 0,
            "" if det_l is None else det_l.method,
            "" if det_r is None else det_r.method]


def main() -> int:
    ap = argparse.ArgumentParser(description="离线双目三维轨迹（读会话）")
    ap.add_argument("session", help="data/sessions/<id> 目录或 session.json")
    # `default=argparse.SUPPRESS`：不传时 **属性不存在**，这样才能区分
    # "用户显式 --method color" 与 "什么也没说，落到默认 color"。
    # 理由见 `_pick_detector`：这个区别决定要不要自动接管成 specular + 种子。
    ap.add_argument("--method", default=argparse.SUPPRESS,
                    choices=method_names("detector"),
                    help="检测方法（默认 %s）。可选：%s" % (
                        default_method("detector"),
                        "；".join(f"{d['name']}={d['summary']}"
                                  for d in describe("detector"))))
    ap.add_argument("--preset", default=None,
                    choices=ColorConfig.preset_names(),
                    help="color 预设（球色）；不传=用配置里的颜色。可选：%s"
                         % (" / ".join(ColorConfig.preset_names())))
    ap.add_argument("--origin", default="0,0,0",
                    help="视频里标出的参照原点（parallel=校正左相机系，"
                         "angled=原始左相机系；x,y,z 米；默认相机光心投影）")
    ap.add_argument("--seed", default="",
                    help="specular 左路人工种子 x,y,r（1920 下 r 约 32~40）")
    ap.add_argument("--seed-right", default="",
                    help="specular 右路人工种子 x,y,r（不能用左路 seed 代替）")
    ap.add_argument("--seed-json", default="",
                    help="specular 的 seeds.json（默认自动用 <session>/derived/seeds.json）")
    ap.add_argument("--seed-frame", type=int, default=0,
                    help="specular 在第几帧初始化（默认 0；种子坐标要对应这一帧）")
    ap.add_argument("--auto-seed", action="store_true",
                    help="不点选：自动改用 specular，并用 YOLO 在最前面若干**成对**帧里"
                         "定位球心当种子（写进 <session>/derived/seeds.json）。"
                         "只在没显式给 --method / --detect-width 时生效；"
                         "半径仍来自 YOLO 框，不可信")
    ap.add_argument("--anchor-gate", choices=("off", "on"), default="off",
                    help="用**一次人工点选**（左右各点一次球心，存在会话的 "
                         "derived/seeds.json）定出「左右半径该差多少」，配对时据此"
                         "拒掉假目标。默认 off；开之前必须先点选过（--auto-seed 的"
                         "种子不算，它的半径来自 YOLO 框）")
    ap.add_argument("--anchor-diameter-mm", type=float, default=0.0,
                    help="锚点换算要用的真实球径（毫米）。0 = 用内置默认值（21.6），"
                         "并在输出里写明用的是它 —— 换算错了整段的半径都在同一档上偏")
    ap.add_argument("--out", default=None, metavar="DIR",
                    help="把产物写到这个目录（traj3d.csv / annotated.mp4 / 预测 / 世界系）。"
                         "不给就写回 <session>/derived —— 做对照实验时用它，"
                         "免得每次覆盖上一次的结果")
    ap.add_argument("--detect-width", type=int, default=0,
                    help="检测缩到多宽（0=原图；真实纹理下 hough 建议 960）")
    ap.add_argument("--min-radius", type=float, default=8.0)
    ap.add_argument("--max-radius", type=float, default=200.0)
    ap.add_argument("--hough-max-radius", type=float, default=80.0)
    ap.add_argument("--max-epipolar", default=None, metavar="PX|auto",
                    help="极线门限（px）；缺省平行 2.5 / angled 8，显式值不被覆盖。"
                         "auto=按标定自报极线 RMS×%g 自适应（见 stereo.triangulate."
                         "epipolar_gate_px）" % EPI_GATE_K)
    ap.add_argument("--sync-offset", type=int, default=None,
                    help="覆盖 session 的同步偏移（默认用 session.sync.offset）")
    ap.add_argument("--pairing", choices=("index", "timestamp", "canary"),
                    default="canary",
                    help="配对策略：canary=默认（时间戳安全接管，失败回退）；"
                         "index=旧常数偏移；timestamp=强制时间戳")
    ap.add_argument("--timestamp-tolerance", type=float, default=0.033,
                    help="时间戳配对容差（秒），默认 0.033（约一帧@30fps）")
    ap.add_argument("--stereo-mode", choices=("auto", "parallel", "angled"),
                    default="auto",
                    help="auto=读 session/标定；angled=大角度（近似直角）")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--geometry-guard", type=int, default=30, metavar="N",
                    help="跑满 N 对后先判一次『录像几何 vs 标定』：对不上就早停"
                         "（不白跑整段，也不产出看似正常的错数字）；0=关。默认 30。"
                         "判据见 stereo.triangulate.geometry_verdict")
    ap.add_argument("--trail", type=int, default=300, help="轨迹线保留多少个匹配点（0=不画）")
    ap.add_argument("--smooth", action="store_true",
                    help="开启三维平滑：另存 traj3d_smoothed.csv，并把平滑结果与抖动指标"
                         "写进 traj3d.json 的 smoothing 块（默认不开，保持旧行为）")
    ap.add_argument("--smooth-method", choices=SMOOTH_METHODS, default="savgol",
                    help="平滑方法（默认 savgol）。savgol 保留恒定加速度、无滞后；"
                         "kalman=常速度 Kalman3D，对平抛/反弹段有系统滞后，仅作对照")
    ap.add_argument("--world-frame", default=str(ROOT / "config" / "world_frame.json"),
                    help="世界系文件（默认 config/world_frame.json）；存在且合法时额外写 "
                         "derived/traj3d_world.csv，traj3d.csv 仍是左相机系")
    ap.add_argument("--no-world", action="store_true",
                    help="强制不做世界系变换（即使配置文件存在）；traj3d.json 记为 left_camera")
    ap.add_argument("--predict-horizons", default="0.1,0.2,0.3,0.5",
                    help="三维预测的外推时刻（秒，逗号分隔；默认 0.1,0.2,0.3,0.5）")
    ap.add_argument("--no-predict", action="store_true", help="不做三维弹道预测")
    ap.add_argument("--no-plot", action="store_true", help="不生成 traj3d.png")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    session_dir, s = load_session(args.session)
    # ---- 检测方法：用户显式给了就照做，否则有种子就自动接管 ----
    method_given = hasattr(args, "method")
    method = getattr(args, "method", default_method("detector"))
    seeds_path = (Path(args.seed_json) if args.seed_json
                  else session_dir / "derived" / "seeds.json")
    method, method_note = _pick_detector(method, method_given, seeds_path,
                                         args.detect_width, args.auto_seed)
    if method_note:
        print(method_note)
    try:
        origin = tuple(float(v) for v in args.origin.split(","))
        if len(origin) != 3:
            raise ValueError
    except ValueError:
        print(f"[错误] --origin 要写成 x,y,z 三个数，现在是 {args.origin!r}")
        return 2
    try:
        horizons = tuple(float(v) for v in args.predict_horizons.split(",") if v.strip())
        if not horizons or any(h <= 0.0 for h in horizons):
            raise ValueError
    except ValueError:
        print(f"[错误] --predict-horizons 要写成正数秒（逗号分隔），"
              f"现在是 {args.predict_horizons!r}")
        return 2
    # ---- 世界系：先读先校验，坏文件在这里就退（不要跑完视频才发现）----
    world_path = Path(args.world_frame)
    world: WorldFrame | None = None
    if args.no_world:
        print("[坐标] --no-world：不做世界系变换，坐标为左相机系")
    elif not world_path.exists():
        print(f"[坐标] 未设置世界系（{world_path} 不存在）：只输出相机系坐标，"
              "具体是哪个相机系由下面的双目模式决定")
    else:
        try:
            world = WorldFrame.load(world_path)
        except (ValueError, OSError) as exc:
            # 明确失败：绝不回退成单位矩阵、绝不静默忽略
            print(f"[错误] 世界系文件不可用，已停止：{exc}")
            return 1
        print(f"[坐标] {world.describe()}（{world_path}）")
    errors, warnings = validate_session(session_dir)
    print(f"会话：{session_dir}")
    for w in warnings:
        print(f"  [警告] {w}")
    if errors:
        for e in errors:
            print(f"  [错误] {e}")
        return 1
    if s.left is None or s.right is None or s.calibration is None:
        print("[错误] session 缺 left/right/calibration")
        return 1

    calib_path = Path(s.calibration.path)
    if not calib_path.is_absolute():
        calib_path = session_dir / calib_path
    rig = StereoRig.load(calib_path)
    if args.stereo_mode == "auto":
        # ---- 双目模式：入口层唯一解析点 ----
        # auto 只在这里由 resolve_auto() 落定（显式 > 标定 meta > 会话 > parallel），
        # 与 calibrate_stereo / record_stereo 同一套优先级与同一份实现；
        # 核心层 normalize_mode() 收到 auto 会直接抛 ValueError。
        # 只有 session.json 真的写了 stereo_mode / rig.stereo_mode 才作为候选传入：
        # Session 对象的默认值 "parallel" 不是"会话里写了"，不能拿它冒充用户意图。
        try:
            raw_session = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
        except Exception:                                   # noqa: BLE001
            raw_session = {}
        session_hint: dict = {}
        if isinstance(raw_session, dict):
            if "stereo_mode" in raw_session:
                session_hint["stereo_mode"] = raw_session["stereo_mode"]
            rig_hint = raw_session.get("rig")
            if isinstance(rig_hint, dict) and "stereo_mode" in rig_hint:
                session_hint["rig"] = {"stereo_mode": rig_hint["stereo_mode"]}
        mode = resolve_auto(args.stereo_mode, session=session_hint,
                            calibration={"meta": rig.meta})
    else:
        # 显式给了 parallel/angled；仍然走同一入口，避免这里再长出一套归一化。
        mode = resolve_auto(args.stereo_mode)
    # ---- 兼容旧数据：parallel + none_raw -> angled ----
    # 必须在下面"实测夹角 vs 模式"复核之前：否则 90° 标定 + parallel 会直接
    # 以退出码 2 结束，而不是按可用的 angled 路径重建。
    mode, switched = _mode_guard(mode, rig.rectification)
    if switched:
        print("[模式] 标定是 none_raw（链式/未校正）但会话记为 parallel：已按 angled 重建。"
              "可用 --stereo-mode angled 消掉这条提示。")
    angle = optical_axes_angle_deg(rig.R)
    print(f"双目模式：{mode_label(mode)}（{mode}）；标定实测光轴夹角 {angle:.1f}°；"
          f"基线 {rig.baseline_mm:.1f} mm")
    if mode == "angled" and angle < 15.0:
        print("[警告] 标定夹角 <15°：这份参数更像平行双目，"
              "请确认 --stereo-mode 与标定文件是否对应。")
    elif mode == "parallel" and angle >= 40.0:
        print(f"[警告] 标定夹角 {angle:.1f}° 已经很大，但当前按 parallel 跑；"
              "如果这是大角度装置，请在会话/标定里标 angled，或显式 --stereo-mode angled。")
    # 保留"矛盾即拒跑(2)"：这是**解析之后**的一致性复核，不是第二套推断 ——
    # auto 沿用的模式（来自标定 meta 或会话）若与本次标定实测夹角明显打架，
    # 说明数据自相矛盾（例如拿平行标定去跑 90° 装置），静默继续只会产出错的三维点。
    # calibrate_stereo 对同一情形也是"标完就拒"，这里保持同一口径。
    if args.stereo_mode == "auto" and (
            (mode == "parallel" and angle >= 40.0)
            or (mode == "angled" and angle < 15.0)):
        print("[错误] auto 解析出的模式与标定实测夹角矛盾，拒绝静默继续；"
              "请显式 --stereo-mode parallel 或 angled。")
        return 2
    # ---- 本次实际使用的坐标系：由**已解析的** mode 决定，不在别处再猜 ----
    # parallel 的三角化点在校正后的左相机系；angled 走原始射线，才是原始左相机系。
    # 两者都叫"左相机系"是错的：世界系/查看器要按对应的 K/D 去解释这些点。
    base_frame = "rectified_left_camera" if mode == "parallel" else "left_camera"
    if world is not None and mode == "parallel":
        # 为什么拒绝：T_world_cam 按**原始**左相机系定义（K1/D1、原始图 PnP），
        # 而 parallel 的点在**校正后**系里，两者之间还差 R1（p_rect = R1·p_cam）。
        # 直接拿 T_world_cam 乘校正系的点，等于把一个坐标系当成另一个用 ——
        # 数字看着正常、实际整体歪掉，且从结果里看不出错。宁可不跑。
        print("[错误] 世界系变换暂不支持 parallel 模式：校正系与原始系之间还需额外变换，"
              "请用 angled 标定后重跑。")
        return 1
    # 把本次实际使用的模式写回 session，避免之后 export_visualization 按旧字段
    # 把 angled 的 traj3d.json 又改回 rectified_left_camera
    if getattr(s, "stereo_mode", "parallel") != mode:
        s.stereo_mode = mode
        rig_info = dict(getattr(s, "rig", {}) or {})
        rig_info.update({"stereo_mode": mode, "angle_deg": angle,
                         "baseline_mm": float(rig.baseline_mm)})
        s.rig = rig_info
        save_session(s, session_dir)
        print(f"[会话] 已记录实际双目模式：{mode}（{mode_label(mode)}）")
    left_video = session_dir / s.left.video
    right_video = session_dir / s.right.video
    fps = float(s.left.fps_measured or s.left.fps_requested or 30.0)

    offset = args.sync_offset if args.sync_offset is not None else int(s.sync.offset)
    if args.sync_offset is None and s.sync.method == "none":
        print("[警告] session 的同步方法是 none（没检测到闪光/拍手）—— 3D 可能错位；"
              "可先用 --sync-offset 手动试。")
    print(f"同步偏移：{offset:+d} 帧；fps={fps:.2f}")
    print(f"配对策略：{args.pairing}")
    print(f"视频：{left_video.name} + {right_video.name}")

    src = PairSource(video=(left_video, right_video), sync_offset=offset,
                     pairing=args.pairing,
                     timestamp_tolerance_s=args.timestamp_tolerance)
    src.report_warnings()
    if args.pairing != "index":
        print(f"实际配对：{src.describe()}")
    if tuple(src.image_size) != tuple(rig.image_size):
        print(f"[错误] 视频 {src.image_size} 与标定 {rig.image_size} 不一致")
        src.release()
        return 1

    cfg = AppConfig()
    cfg.detector.method = method
    if method == "specular" and args.max_epipolar is None:
        args.max_epipolar = 8.0
        # 8 px 本身也是个**经验值**，不是从几何推出来的：它只放宽了"能不能配上"，
        # 不改善任何精度。2026-10-30 实测那段会话 dy 中位 12.4 px（标定自报 0.56 px），
        # 8 px 也只放过 1/30 —— 那种情况是"录像和标定对不上"，不要靠继续往上调来救，
        # 去看程序打出的 [诊断] 几何一致性那几行，然后重标定。
        print("[specular] 极线门限 2.5px 对人工 seed/CSRT 太紧，默认放宽到 8px")
    cfg.detector.min_radius = float(args.min_radius)
    cfg.detector.max_radius = float(args.max_radius)
    cfg.detector.hough_max_radius = float(args.hough_max_radius)
    cfg.predict.enable = False
    if args.preset:
        cfg.detector.color = ColorConfig.preset(args.preset)
    pipe_l = _make_pipeline(cfg, rig, "left")
    pipe_r = _make_pipeline(cfg, rig, "right")
    seed_frame = 0        # 种子落在第几帧；后面要拿它给重建器设"预热期"
    if method == "specular":
        def _seed_arg(text: str, what: str):
            try:
                parts = [float(v) for v in text.split(",")]
            except ValueError:
                parts = []
            if len(parts) != 3 or parts[2] <= 0:
                print(f"[错误] {what} 要写成 x,y,r 三个数，例如 1190,300,36；现在是 {text!r}")
                return None
            return tuple(parts)

        if args.detect_width:
            print("[错误] --method specular 暂不支持 --detect-width："
                  "种子坐标是原图坐标，缩放会错位。")
            return 2
        if args.auto_seed and seeds_path.exists():
            # 用户的显式意图不能被静默忽略：已有种子更好（是人工点选的），所以照用它，
            # 但必须说一声"你给的 --auto-seed 这次没生效、原因是什么"。
            print("[自动种子] 会话里已有 derived/seeds.json → 直接用它"
                  "（--auto-seed 不再自动定位）；要强制重新定位请先删掉它，"
                  "或用 --seed/--seed-right 覆盖")
        seed_json = None
        if seeds_path.exists():
            try:
                seed_json = load_seeds(seeds_path)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                # 走到这里说明用户是**显式**要 specular 的（自动接管那条路已在
                # `_pick_detector` 里退回原检测器）。此时静默换回 color 只会得到
                # "0 检出"这个更难查的现象，所以停下来说清楚是哪一步坏了。
                print(f"[错误] 种子文件不可用：{seeds_path}（{exc}）")
                print("  重开『② 点选球心』重新保存，或改用 --seed/--seed-right。")
                return 2
        elif args.seed_json:
            print(f"[错误] --seed-json 不存在：{seeds_path}")
            return 2
        if args.seed or args.seed_right:
            if not args.seed or not args.seed_right:
                print("[错误] --seed 和 --seed-right 必须同时给。")
                return 2
            seed_l = _seed_arg(args.seed, "--seed")
            seed_r = _seed_arg(args.seed_right, "--seed-right")
            if seed_l is None or seed_r is None:
                return 2
            seed_frame = int(args.seed_frame)
            try:
                saved = write_seeds(session_dir, seed_l, seed_r, seed_frame=seed_frame,
                                    image_size=tuple(src.image_size), source="cli",
                                    note="命令行 --seed/--seed-right")
                print(f"[specular] 已写标准 seed：{saved}")
            except Exception as exc:                        # noqa: BLE001
                print(f"[警告] seed 写入失败（继续跑）：{exc}")
        elif seed_json is not None:
            seed_l = (float(seed_json["left"]["x"]), float(seed_json["left"]["y"]),
                      float(seed_json["left"]["radius"]))
            seed_r = (float(seed_json["right"]["x"]), float(seed_json["right"]["y"]),
                      float(seed_json["right"]["radius"]))
            seed_frame = int(seed_json.get("seed_frame", args.seed_frame))
            print(f"[specular] 读标准 seed：{seeds_path}")
        elif args.auto_seed:
            found = _auto_seed_pair(cfg, src)
            if found is None:
                return 2
            seed_l, seed_r, hit_index = found
            # 搜索已经消费掉前 hit_index 对，"种子生效帧"要相对**剩下的**帧数：
            # specular 的 seed_frame 是它自己 detect() 的调用次数，第一帧 = 0。
            seed_frame = 0
            if hit_index:
                print(f"[自动种子] 搜索用掉了前 {hit_index} 对（不进产物）；"
                      "种子在第一次处理时生效，可能滞后一帧以内")
            try:
                saved = write_seeds(session_dir, seed_l, seed_r, seed_frame=seed_frame,
                                    image_size=tuple(src.image_size), source="auto-yolo",
                                    note=f"YOLO 在第 {hit_index} 对自动定位（--auto-seed）")
                print(f"[自动种子] 已写标准 seed：{saved}")
            except Exception as exc:                        # noqa: BLE001
                print(f"[警告] seed 写入失败（继续跑）：{exc}")
        else:
            print("[错误] --method specular 需要球心种子：")
            print("  GUI：第 3 步「点选球（双目 seed）」；")
            print("  或 CLI：--seed 左x,左y,左r --seed-right 右x,右y,右r")
            print("  也可用 --seed-json <seeds.json>；不点选就用 --auto-seed"
                  "（YOLO 自动定位，半径来自框、不可信）。")
            return 2
        pipe_l.detector.seed(*seed_l, frame_index=seed_frame)
        pipe_r.detector.seed(*seed_r, frame_index=seed_frame)
        print(f"[specular] 左 seed={seed_l} 右 seed={seed_r} 初始化帧={seed_frame}")
        if seed_frame > 0:
            print(f"[提示] seed_frame={seed_frame}：前 {seed_frame} 对按设计不会初始化检测；"
                  "除非第 0 帧看不清球，否则建议改成 0")

    elif args.seed or args.seed_right:
        print("[警告] 只有 --method specular 使用 --seed/--seed-right；已忽略。")
    if args.detect_width:
        pipe_l.set_detect_width(args.detect_width)
        pipe_r.set_detect_width(args.detect_width)

    epi, epi_note = _resolve_max_epipolar(args.max_epipolar, rig, mode)
    if epi_note:
        print(epi_note)
    # 人工锚点门（默认关，见 --anchor-gate 的 help）。放在**建重建器之前**：
    # 它要注入 MatchConfig，而不是等跑完再解释结果。
    anchor = None
    if args.anchor_gate == "on":
        if mode == "angled":
            print("[锚点] 只对 parallel（已极线校正）有意义；angled 下不启用")
        else:
            anchor, why = _radius_anchor_from_session(args, seeds_path, rig)
            if anchor is None:
                print(f"[锚点][错误] {why}")
                return 2
            print(f"[锚点] {describe_anchor(anchor)}")
            print("[锚点] 半径门已开：配对时按「左右半径比」拒掉假目标"
                  "（--anchor-gate off 可关）")
    if mode == "angled":
        from stereo.angled import AngledMatchConfig, AngledReconstructor
        if args.max_epipolar is None:
            print("[angled] 极线门限默认放宽到 8px（大角度下候选像素偏差更大）")
        rec = AngledReconstructor(rig, AngledMatchConfig(max_epipolar_px=epi))
    else:
        # `source_hint` 只进"0 配对"的报错文案：那句话要能直接告诉用户
        # `--seed-json <会话>/derived/seeds.json` 长什么样，重建器自己不知道会话在哪。
        # 用 as_posix()：文案里再拼 "/derived/seeds.json"，混着反斜杠会像坏路径。
        rec = StereoReconstructor(
            rig,
            MatchConfig(max_epipolar_px=epi,
                        radius_ratio_expected=(anchor.radius_ratio if anchor else 0.0),
                        radius_ratio_band=(anchor.band if anchor else 0.25)),
            source_hint=session_dir.as_posix())
    # 种子落在第 N 帧时，前 N 帧**按设计**不会有检测。不告诉重建器的话，它的
    # 零配对门禁会在第 20 帧判"检测器坏了"并抛错 —— 2026-11-01 真实踩到：
    # 用户在 GUI 第 80 帧点选球心，跑离线三维直接报错，看起来像"点选没用"。
    if seed_frame > 0 and hasattr(rec, "warmup_frames"):
        rec.warmup_frames = int(seed_frame)
        print(f"[提示] 前 {seed_frame} 帧记为预热期：零配对判据从第 "
              f"{seed_frame + 20} 帧之后才生效，配对率分母也扣掉这段")
    # 录前几何守卫（`B-131`）：跑满 N 对就先判一次"录像几何 vs 标定"，对不上就早停。
    # 放在这里而不是库层默认：`mono_vs_stereo` / `profile_stages` 会在同一批坏会话上
    # 跑，默认拦下来会让它们从"出数"变成"崩"；那是入口层的决定，不该由库替它做。
    guard_at = args.geometry_guard if args.geometry_guard > 0 else 0
    guard_armed = bool(guard_at and mode == "parallel")
    if guard_armed:
        print(f"[几何守卫] 跑满 {guard_at} 对后先判一次极线几何，与标定对不上就早停"
              "（--geometry-guard 0 可关）")
    # `--out` 让产物**不落在会话目录里**（2026-10-27 补）。
    #
    # 为什么需要：做对照实验时（换方法 / 换 seed / 换门限）会连着跑好几次，
    # 每次都覆盖 `<session>/derived/`，上一次的结果就没了 —— 要么手工 `cp` 到别处
    # （实测就是这么干的，完事还得记着删），要么反复跑同一份数据。
    # 现在的约定：**不给 `--out` 就还是老行为**（写回会话目录），给了就全写进去。
    out_dir = Path(args.out) if args.out else (session_dir / "derived")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.out:
        print(f"[输出] 产物写到 {out_dir}（会话目录保持干净）")
    # 部分帧试跑不要把正式的 traj3d.csv / annotated.mp4 覆盖成 0.17 s
    partial = f"_first{args.max_frames}" if args.max_frames else ""
    csv_path = out_dir / f"traj3d{partial}.csv"
    video_path = out_dir / f"annotated{partial}.mp4"

    if args.max_frames:
        print(f"[部分帧] 只处理前 {args.max_frames} 对；输出加 {partial} 后缀，"
              "不会覆盖正式文件。")

    writer = None
    if not args.no_video:
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 540))
        if not writer.isOpened():
            print("[警告] 打不开标注视频 writer，继续只写 CSV")
            writer = None

    rows: list[list] = []
    trail_l: list[tuple[float, float]] = []
    trail_r: list[tuple[float, float]] = []
    n_pairs = 0
    center_src = {"contour": 0, "box": 0}   # 只看这两种；其它（非 specular）不计数
    t0 = time.perf_counter()
    try:
        while True:
            ok, left_frame, right_frame = src.read()
            if not ok or left_frame is None or right_frame is None:
                break
            if args.max_frames and n_pairs >= args.max_frames:
                break
            t = n_pairs / fps
            res_l = pipe_l.process(left_frame, t)
            res_r = pipe_r.process(right_frame, t)
            det_l = _as_detection(res_l)
            det_r = _as_detection(res_r)
            pt = rec.update(t, det_l, [det_r] if det_r is not None else [])
            n_pairs += 1

            # 圆心口径的来源（S2a）：`center_source` 由**检测器**给，不随 `TrackState`
            # 往下传，所以只在这里从流水线结果里取。用途只有一个 —— 几何守卫判读前
            # 分清"几何坏了"还是"圆心没钉到轮廓上"（后者 y 差也会错开 ≈6 px）。
            for _r in (res_l, res_r):
                _d = getattr(_r, "detection", None)
                _src = (_d.extra or {}).get("center_source") if _d is not None else None
                if _src in center_src:
                    center_src[_src] += 1

            # 几何守卫（只在 parallel）：跑到 `--geometry-guard N` 对之后判一次。
            # `dy_samples` 只在两路都检出的帧里攒，所以样本可能少于 N 对；攒够之前
            # `geometry_verdict` 返回 None —— 那是"还没法判"，**不消耗这次机会**，
            # 下一帧接着看。原来写的是 `n_pairs == guard_at`（只判一次），丢帧多时会
            # 一次也没判成、然后永久放弃 —— 那是一条假阴性路径。
            # 判出可疑就 `return 2`：**拒绝出数**。放宽门限能"跑通"，但那些点的
            # 残差与交会角本来就是错的（`B-131`：dy 中位 12.4 px，残差 3.9~7.2 px）。
            if guard_armed and n_pairs >= guard_at:
                guard_armed, v = _guard_step(guard_armed, guard_at, n_pairs,
                                             rec.dy_samples, getattr(rig, "quality", None))
                if v is not None and v.suspect:
                    _tot = center_src["contour"] + center_src["box"]
                    _box = (center_src["box"] / _tot) if _tot else None
                    print()
                    print(rec.geometry_failure_message(v, _box))
                    return 2

            if pt is None:
                # ⚠️ 这一行的字段数**必须**和 header 一样（15 个）。
                # 2026-10-30 实测：这里原来只有 14 个（少了 `confident`），于是
                # 所有未匹配行的 `confident` 列读出来是 `method_left`（"track"/"specular"），
                # 后面每一列都整体左移一格。`csv.DictReader` 不会报错，只会给错值 ——
                # 这正是本项目最贵的那类"看起来成功"。任何按 `confident == 0` 过滤的
                # 分析都会得到 0 行，且找不到原因。
                rows.append(row_unmatched(n_pairs - 1, t, det_l, det_r))
            else:
                trail_l.append(pt.u_left)
                trail_r.append(pt.u_right)
                if args.trail and len(trail_l) > args.trail:
                    trail_l = trail_l[-args.trail:]
                    trail_r = trail_r[-args.trail:]
                rows.append([
                    n_pairs - 1, f"{t:.4f}", 1,
                    f"{pt.xyz[0]:.6f}", f"{pt.xyz[1]:.6f}", f"{pt.xyz[2]:.6f}",
                    f"{pt.u_left[0]:.2f}", f"{pt.u_left[1]:.2f}",
                    f"{pt.u_right[0]:.2f}", f"{pt.u_right[1]:.2f}",
                    f"{pt.epipolar_px:.3f}", f"{pt.residual_px:.3f}",
                    (1 if mode == "parallel"
                     else int(bool(getattr(pt, "confident", pt.residual_px <= 2.0)))),
                    "" if det_l is None else det_l.method,
                    "" if det_r is None else det_r.method,
                ])

            if writer is not None:
                vl = cv2.resize(_draw(left_frame, res_l, "L", trail_l,
                                      xyz=None if pt is None else pt.xyz,
                                      rig=rig, origin=origin, draw_reference=True,
                                      mode=mode), (960, 540))
                vr = cv2.resize(_draw(right_frame, res_r, "R", trail_r,
                                      xyz=None if pt is None else pt.xyz), (960, 540))
                cv2.putText(vr, f"f={n_pairs-1} matched={rec.n_matched}", (12, 500),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                writer.write(np.hstack([vl, vr]))
    finally:
        src.release()
        if writer is not None:
            writer.release()

    # 守卫开着却没判成：说清是"没样本"，不是"没问题"。不说的话，这一段会**静默地
    # 跳过**几何判读 —— 用户以为守过了，其实一次都没判。
    if guard_armed:
        print(f"[几何守卫] 没能判读：两路同时检出的帧只有 {len(rec.dy_samples)} 个，"
              f"不足 {GEOM_MIN_SAMPLES} 个。这是**没样本**，不是**没问题** —— "
              "别拿它当几何 OK 的证据。")

    header = CSV_HEADER
    # 每一行都必须与表头等长：列错位不会报错，只会让下游读到错值。
    bad = [(i, len(r)) for i, r in enumerate(rows) if len(r) != len(header)]
    if bad:
        raise SystemExit(
            f"[错误] rows 里有 {len(bad)} 行列数与表头不一致（表头 {len(header)} 列），"
            f"前 3 个：{bad[:3]}。写出去会是一份列错位的 CSV，宁可现在失败。")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    elapsed = time.perf_counter() - t0
    print()
    print(f"处理：{n_pairs} 对，{elapsed:.1f}s（{n_pairs/max(elapsed,1e-6):.1f} 对/s）")
    # 括号里写清楚"被什么拒的"：parallel 在**配对**阶段只有极线/视差两条门，
    # 交会角与残差根本不参与筛选（它们是三角化之后才有的量）。原来一律写
    # "极线/交会角/残差"，会让人以为角度也被判过，从而去调不存在的东西。
    reject_why = ("极线/视差条件" if mode == "parallel" else "极线/交会角/残差")
    print(f"匹配：{rec.n_matched}/{n_pairs}（{rec.match_rate*100:.1f}%）  "
          f"无左={rec.n_no_left} 无右={rec.n_no_right} "
          f"拒绝={rec.n_rejected}（{reject_why}）")
    # 配对率低时把"几何对不对"量出来：用户看到的 3.3% 背后常常是"这段录像和
    # 它的标定对不上"，不说出来，下一个人还会去调 --max-epipolar（而那不是原因）。
    # 0 配对的情形走的是异常那条路，诊断已经拼在报错里。
    for line in rec.low_match_diagnostic():
        print(line)
    if n_pairs and rec.n_matched == 0:
        print("[诊断] 0 匹配，先别继续调 pairing；按下面顺序查：")
        if method == "specular" and "seed_l" in locals() and "seed_r" in locals():
            print(f"  1) seed 半径：左 {seed_l[2]:.1f} / 右 {seed_r[2]:.1f}；"
                  "CSRT 框会按 1.8×半径放大，过大时框进衣服/背景")
            print("     重开『② 点选球心』，点球心后滚轮把圆贴到球边缘；"
                  "1080p 下这颗球通常 r≈45~70，不是 130")
        if rec.n_no_left > rec.n_no_right:
            print(f"  2) 左路跟丢为主；左检测器："
                  f"{getattr(pipe_l.detector, 'last_error', '') or '(无错误信息)'}")
        elif rec.n_no_right > rec.n_no_left:
            print(f"  2) 右路跟丢为主；右检测器："
                  f"{getattr(pipe_r.detector, 'last_error', '') or '(无错误信息)'}")
        else:
            print(f"  2) 左右都检测到了，但被{reject_why}拒绝："
                  "检查两路 seed 是否都在球心、是否同一时刻；"
                  "可先用 --pairing index 对照")
        print("  3) 当前会话若继续 0 匹配，建议重录并在第 0 帧重新点选，"
              "不要靠放大 --max-epipolar 硬吃错配")
    smoothed = None
    smoothing_block: dict | None = None
    smoothing_skip: str | None = None
    arr = rec.as_array() if rec.trajectory else np.zeros((0, 4), dtype=np.float64)
    if rec.trajectory:
        print(f"三维范围：X {arr[:,1].min():.3f}~{arr[:,1].max():.3f} m  "
              f"Y {arr[:,2].min():.3f}~{arr[:,2].max():.3f} m  "
              f"Z {arr[:,3].min():.3f}~{arr[:,3].max():.3f} m")
        res = np.array([p.residual_px for p in rec.trajectory])
        print(f"交会残差：中位 {np.median(res):.3f} px，最大 {res.max():.3f} px")
        if args.smooth:
            if args.smooth_method == "kalman":
                print("[平滑] 警告：kalman 是**常速度**模型，对平抛（恒定加速度）/反弹"
                      "（速度突变）段有系统性滞后 —— 只作对照，正常出数请用默认 savgol。")
            if len(arr) < 3:
                smoothing_skip = f"可用三维点只有 {len(arr)} 个（<3），凑不出平滑窗口"
                print(f"[平滑] 跳过：{smoothing_skip}")
            else:
                smoothed, sm_params = smooth_trajectory(
                    arr[:, 0], arr[:, 1:4], method=args.smooth_method)
                jit_before = jitter_rms(arr[:, 1:4])
                jit_after = jitter_rms(smoothed)
                # 抖动是"同一序列前后对比"的相对量；前值为 0（如三点共线）时不给百分比，
                # 免得用 0/0 编出一个好看的数字。
                reduction = (None if jit_before <= 0.0
                             else (1.0 - jit_after / jit_before) * 100.0)
                sm_path = out_dir / f"traj3d_smoothed{partial}.csv"
                with open(sm_path, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["t", "X", "Y", "Z"])
                    for ti, x, y, z in zip(arr[:, 0], smoothed[:, 0],
                                           smoothed[:, 1], smoothed[:, 2]):
                        w.writerow([f"{ti:.4f}", f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])
                print(f"平滑 CSV：{sm_path}")
                print(f"[平滑] 方法={args.smooth_method} 参数={sm_params}；"
                      f"抖动 RMS 前 {jit_before * 1e3:.2f} mm → 后 {jit_after * 1e3:.2f} mm"
                      + ("" if reduction is None else f"（下降 {reduction:.1f}%）"))
                # frame 与 arr 同序（rec.as_array 的追加顺序 = matched 行的顺序，
                # 与下面 world CSV 用的是同一个对齐假设）；对不上就写 null，不编造。
                frames: list[int | None] = [int(r[0]) for r in rows if int(r[2]) == 1]
                if len(frames) != len(arr):
                    frames = [None] * len(arr)
                smoothing_block = {
                    "method": args.smooth_method,
                    "params": sm_params,
                    "jitter_rms_before_m": jit_before,
                    "jitter_rms_after_m": jit_after,
                    "jitter_reduction_pct": reduction,
                    "source_csv": f"derived/{sm_path.name}",
                    # points 与 samples 同坐标系（本块不做世界系变换）
                    "points": [{"frame": fr, "t": float(ti),
                                "xyz": [float(x), float(y), float(z)]}
                               for fr, ti, x, y, z in zip(
                                   frames, arr[:, 0], smoothed[:, 0],
                                   smoothed[:, 1], smoothed[:, 2])],
                }
        if not args.no_plot:
            plot_path = out_dir / f"traj3d{partial}.png"
            _plot_trajectory(arr, plot_path, smoothed)
            print(f"三维图：{plot_path}")
    # ---- 世界系：另写一份坐标，traj3d.csv 的左相机系语义不动 ----
    world_arr, world_block = _world_block(world, arr, base_frame)
    if world_arr is not None:
        # arr 与 rows 里的 matched 行同序（rec.as_array 逐点顺序 = 追加顺序）
        wrows = [list(r) for r in rows]
        j = 0
        for r in wrows:
            if int(r[2]) == 1 and j < len(world_arr):
                r[3] = f"{world_arr[j, 1]:.6f}"
                r[4] = f"{world_arr[j, 2]:.6f}"
                r[5] = f"{world_arr[j, 3]:.6f}"
                j += 1
        world_csv = out_dir / f"traj3d_world{partial}.csv"
        with open(world_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(wrows)
        print(f"世界系 CSV：{world_csv}")
    # ---- 三维预测：世界系存在时先把重力换到相机系，输出仍与 traj3d.csv 同系 ----
    # 默认假设相机系 Y 朝下（与 StereoRig/K 的像素轴一致），所以重力沿 +Y；
    # 拟合在哪个系里做，g 就必须写成那个系的向量（predict3d 的文档要求）。
    g_cam = (0.0, 9.81, 0.0)
    if world is not None:
        R_wc = np.asarray(world.T_world_cam, dtype=np.float64)[:3, :3]
        # p_cam = R^T·(p_world - t)，所以世界系重力换到相机系是 R^T·g_world
        g_cam = tuple(float(v) for v in (R_wc.T @ np.array([0.0, -9.81, 0.0])))
    fit_arr = np.column_stack([arr[:, 0], smoothed]) if smoothed is not None else arr
    if args.no_predict:
        pred_rows: list[dict] = []
        pred_block: dict = {"skipped": "已用 --no-predict 关闭"}
    else:
        pred_rows, pred_block = _predict_block(
            fit_arr, PredictConfig(g=g_cam, horizons=horizons))
    if pred_rows:
        pred_path = out_dir / f"predictions{partial}.csv"
        with open(pred_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_ref", "horizon_s", "x", "y", "z"])
            for r in pred_rows:
                w.writerow([f"{r['t_ref']:.6f}", f"{r['horizon_s']:.3f}",
                            f"{r['x']:.6f}", f"{r['y']:.6f}", f"{r['z']:.6f}"])
        pick = min(pred_rows, key=lambda r: abs(r["horizon_s"] - 0.3))
        print(f"预测：t_ref={pick['t_ref']:.3f} s，{pick['horizon_s']:.1f} s 后 "
              f"({pick['x']:.3f}, {pick['y']:.3f}, {pick['z']:.3f}) m；{pred_path}")
    else:
        print(f"预测跳过：{pred_block.get('skipped', '原因未知')}")
    print(f"CSV：{csv_path}")
    if writer is not None:
        print(f"标注视频：{video_path}")
    if not partial:
        # --no-world 时把 export 的世界系入口指到一个不存在的路径（它按"无配置"处理）
        vis_world_path = (world_path if world is not None
                          else world_path.with_name(world_path.name + ".disabled"))
        try:
            paths = export_session_visualization(
                session_dir, csv_name=csv_path.name, stereo_mode=mode,
                world_frame_path=vis_world_path)
            print(f"标准轨迹 JSON：{paths['trajectory']}")
            print(f"可视化 JSON：{paths['visualization']}")
            # 坐标系与预测块写进 traj3d.json：export 刚重建过它，必须在之后补
            traj_path = paths["trajectory"]
            payload = json.loads(traj_path.read_text(encoding="utf-8"))
            payload["coordinate_system"] = world_block["coordinate_system"]
            payload["world_frame"] = world_block["world_frame"]
            if pred_rows:
                payload["predictions"] = {k: pred_block[k]
                                          for k in ("t_ref", "state", "points")}
            else:
                payload["predictions"] = None
                payload["predictions_skipped"] = pred_block.get("skipped", "原因未知")
            # 平滑块：**总是**写键（没开就写 null），这样"旧文件没这个键"与
            # "这次没开平滑"能分开；points 与 samples 同坐标系，不做世界系变换。
            payload["smoothing"] = smoothing_block
            if smoothing_block is None and smoothing_skip:
                payload["smoothing_skipped"] = smoothing_skip
            traj_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        except Exception as exc:                            # noqa: BLE001
            print(f"[警告] 标准 JSON 导出失败：{exc}")
    return 0 if rec.n_matched else 2


if __name__ == "__main__":
    raise SystemExit(main())
