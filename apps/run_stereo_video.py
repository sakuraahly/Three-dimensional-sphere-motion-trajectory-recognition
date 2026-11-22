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
    derived/traj3d.json       标准轨迹 JSON：没开世界系时 xyz 与 traj3d.csv 同系；
                              有世界系时 `samples` / `bounds` 也换成世界系
                              （`coordinate_system="world"`、`world_csv` 指回上面那份）
                              —— JSON 里的坐标**永远**是 `coordinate_system` 说的那个系
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
import os
import re
import statistics
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
    classify_guard,         # 几何守卫：把「检出质量差」与「几何不一致」分开报
)
from stereo.visualization import (  # noqa: E402
    export_session_visualization,
    load_seeds,
    write_seeds,
)
from stereo.sessions import (  # noqa: E402
    load_session, save_session, session_calibration_file, validate_session,
)
from stereo.angled_geometry import (  # noqa: E402
    MIN_TRIANGULATION_ANGLE_DEG,   # 几何核数值下限，CLI 文案要说它
)


def _as_detection(res) -> Detection | None:
    """优先用跟踪后的圆心；跟踪在滑行时退回原始检测。"""
    state = res.track
    if state is not None and not state.coasting:
        r = res.detection.radius if res.detection is not None else state.radius
        return Detection(x=state.x, y=state.y, radius=r, score=1.0, method="track")
    return res.detection


def _gray(frame: np.ndarray) -> np.ndarray:
    """取单通道灰度。**每帧每路只该调一次**（联合细化要吃灰度图）。"""
    # 单通道输入直接返回：`cvtColor(..., BGR2GRAY)` 对 1 通道图会抛，而视频源
    # 理论上可能是灰度（录制端只存 BGR，但入口不该假定上游一定守约）。
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _joint_detection(det: Detection, xy, applied: bool) -> Detection:
    """用联合细化给出的圆心重建 `Detection`，其余字段原样保留。

    为什么只换 ``x/y``：半径、``score``、``method`` 描述的是"目标多大、它是被
    哪条路找到的"，联合细化只回答"圆心该落在哪"，它没有能力也不该改这些量。
    ``contour`` / ``mask`` 同样保留 —— 那是候选轮廓，圆心挪几个像素并不会让它
    失效，丢掉反而让下游少一份可用信息。
    ``extra["joint_refine"]`` 记下本帧**是否真的被采纳**：排查时最常见的疑问是
    "这帧圆心为什么和检测器输出的不一样"，没有这个标记只能靠比对数值去猜。
    """
    extra = dict(det.extra or {})
    extra["joint_refine"] = bool(applied)
    return Detection(x=float(xy[0]), y=float(xy[1]), radius=det.radius,
                     score=det.score, method=det.method, parts=dict(det.parts or {}),
                     contour=det.contour, mask=det.mask, extra=extra)


#: reason 里随帧变化的数字（"v 差 44px"、"12px"）在归类前要压平，否则每一对
#: 都是一个独立的键，"其它 35（多为…）"永远只数到 1。
_JUNK_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _support_retention(res) -> float | None:
    """支撑保留率 = **每侧各自**的 `共享处 / 独立最优` 里较差的那个。

    ## 为什么必须用"保留率"而不是 `dv_after` 来汇报

    共享 v 的定义就是"把两侧的校正后 v 压成同一个数"，所以 `dv_after` 恒接近 0
    —— 那是**同义反复，不是成绩**。把它印成"27.9 → 0.8 px"会让人以为测量精度
    提高了，而实际上它只是"我按定义让两个数相等了"。真正付了代价的是轮廓：
    把两侧拉到同一个 v，会离开各自独立最优点，支撑率必然下降。保留率（1.00 =
    与独立最优持平）才是这个方法的**成本**，也是唯一能拿来判断"值不值"的独立量。

    ## 口径：逐侧比值取最小，不是"最小的当前 ÷ 最大的参照"

    两种算法在最坏情况下能差一倍以上。逐侧算 `sup/own` 再取 min，语义是
    "**哪一侧掉得最多**"，与模块内部的采纳判据（`keep_l`、`keep_r` 都要过线）
    完全同口径 —— 汇报的数和把关的数必须是同一个，否则日志会和行为打起来。

    ``own`` 侧支撑缺失时返回 None（老版本模块只有 ``support_left/right``，
    没有参照系就无从算比值）—— 宁可这行不打，也不要拿两个不可比的数硬除。
    """
    own_l = getattr(res, "support_left_own", None)
    own_r = getattr(res, "support_right_own", None)
    if own_l is None or own_r is None:
        return None
    try:
        keep_l = float(res.support_left) / float(own_l) if float(own_l) > 1e-6 else 0.0
        keep_r = float(res.support_right) / float(own_r) if float(own_r) > 1e-6 else 0.0
        # 截到 [0,1]：`own` 是在"环相关打分表"的 argmax 处量的支撑，而分子是
        # 共享 v 处那一点的支撑 —— 两个打分器不同（一个是幅值卷积，一个是射线
        # 径向+符号一致性），所以比值**可能 >1**。>1 不代表"比最优还好"，只代表
        # "参照点不是同一个打分器的最优"，按 1.00 记（= 没掉）才不会误导。
        val = min(1.0, max(0.0, min(keep_l, keep_r)))
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


def _joint_refine_summary(stats: dict, keep_floor: float | None = None) -> str:
    """联合细化的摘要（**纯函数**，只拼字符串，便于自测）。

    `stats` 的键：`applied` / `skipped` / `support_short` / `no_gain` / `other`
    是计数，`dv_before` / `retention` 是样本列表（调用处已滤掉 NaN）。
    `keep_floor` 是模块判"代价能不能接受"的保留率下限（`DEFAULT_MIN_SUPPORT_KEEP`）；
    取不到就只解释这个比值怎么读，不编一个门限。

    ## 刻意**不打印** `dv_after`

    见 :func:`_support_retention`：共享 v 让 `dv_after` 恒≈0，"校正后 v 差中位
    0.8 px"这种句子读起来像精度提升，实际什么都没证明。`dv_before` 反倒值得
    打 —— 它说明"现状有多差、有多少可修的量"，那是输入属性，不是自证。

    ## 保留率 >1 不算好消息

    共享 v 处的支撑高于该侧独立最优，说明"环上梯度最强处"和"打分表最认可处"
    在这一帧上互相不同意（球内反射纹理比轮廓还强时就长这样），采纳可能把圆心
    从对的挪到错的。所以中位数超过 1.05 时要把这层意思放在同一行里，不能让
    "1.24 > 1.00"读成"比独立最优还好"。
    """
    total = stats["applied"] + stats["skipped"]
    if not total:
        return "[联合细化] 没有可用的成对检测"

    reject = [f"支撑不足 {stats['support_short']}", f"无改善 {stats['no_gain']}"]
    if stats["other"]:
        # 第三类只在真的出现时才占版面：`半径太小 / 未做极线校正 / 贴边` 归进
        # "无改善"会把"根本没法算"说成"算了但没变好"，那是另一句假话。
        label = f"其它 {stats['other']}"
        if stats.get("other_reasons"):
            # 顺手把这一类里最常见的那句原样带出来（截断到第一层括号前）。
            # 模块以后新增拒绝理由（例如 v 超出搜索范围）时，这条会自动变成
            # 可读的文字，而不是埋在"其它 7"里没人知道那 7 对是怎么被拒的。
            top = max(stats["other_reasons"].items(), key=lambda kv: kv[1])[0]
            label += f"（多为「{top}」）"
        reject.append(label)
    head = (f"[联合细化] 应用 {stats['applied']}/{total} 对"
            f"（拒 {stats['skipped']}：{' / '.join(reject)}）")

    tail = []
    if stats["retention"]:
        med = statistics.median(stats["retention"])
        if keep_floor is not None and math.isfinite(keep_floor):
            # 口径一致：模块的 0.85 是逐侧 support/own 的门限，这行报的也是逐侧比值
            # 取最小 —— 汇报的数和把关的数必须是同一个，否则用户会以为模块自相矛盾。
            note = f"逐侧 min(sup/own)，与采纳门限 {keep_floor:.2f} 同口径"
        else:
            note = "1.00=与独立最优持平"
        tail.append(f"共享 v 处支撑保留率 中位 {med:.2f}（{note}）")
        # 阈值取 1.05 而不是 1.00：1.01~1.05 属于打分噪声，喊狼来了会把真正
        # "支撑与打分表指向两个地方"的帧（实测见过 1.24）淹掉。
        if med > 1.05:
            tail.append("保留率>1：该帧支撑与打分表互相不同意，采纳可能是改错")
    elif stats["applied"]:
        tail.append("支撑保留率无法统计（模块未提供 support_*_own）")
    if stats["dv_before"]:
        tail.append(f"校正前 v 差中位 {statistics.median(stats['dv_before']):.1f} px")
    if not tail:
        return head
    return head + "\n[联合细化] " + "；".join(tail)


def _make_pipeline(cfg: AppConfig, rig: StereoRig, side: str) -> SpherePipeline:
    pipe = SpherePipeline(cfg)
    K = rig.K1 if side == "left" else rig.K2
    # 球径**没有默认值**（2026-11-03 改）：双目的 Z 来自视差，本来就不需要它；
    # 而写死一个 21.6 mm 会让"单目读数"随入口漂移 —— 换成不同大小的塑料球之后，
    # 那个数更是纯属编造。只在用户真的给了球径时才建 CameraModel，
    # 否则保持 `pipe.camera = None`（`_rebuild_3d` 会直接跳过，不影响三角化）。
    if cfg.camera.ball_diameter_m:
        pipe.camera = CameraModel(
            width=int(rig.image_size[0]), height=int(rig.image_size[1]),
            fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]),
            ball_diameter_m=float(cfg.camera.ball_diameter_m),
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

    ## 2026-11-06 改：种子不再等于"必须用 specular"

    旧规则是"会话里有 seeds.json → 强制 specular"，理由是镜面球在 color 下
    恒为 0 检出。那条理由只对**镜面球**成立，对纯色塑料球是反的：color 本来
    就能在整幅画面里找到它，硬换成 specular 反而把"半径来自种子"这个优势丢了。

    现在按 ``seeds.json`` 里记的 ``ball_kind`` 决定：

    - ``plastic`` → 保留 **color**，种子交给它取色/定半径；
    - ``metal`` 或**旧文件没记种类** → specular（兼容旧会话）；
    - 用户显式给了 ``--method`` → 一律照做（哪怕和 ball_kind 不一致，
      只在说明里写一句"按你说的来"）。

    `--auto-seed` 仍是"不用点选、用 YOLO 定位后交给 specular"的金属球路径；
    纯色球不需要它（color 自己找得到）。
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

    kind = str(payload.get("ball_kind") or "")
    # 半径出身要在**每次**接管时复述，不能只在 --auto-seed 那一次说（独立 Critic 指出）：
    # 自动种子把"半径来自 YOLO 框、不可信"写进了 seeds.json，但重跑时走的是这条
    # 分支、原先一个字都不提 —— 那是一次静默的告知退化，而"不可信半径"恰好是
    # 本项目最容易被当成"标定不准"的东西。
    shown = ""
    if str(payload.get("source") or "") == "auto-yolo":
        shown = "；⚠ 这份种子是 --auto-seed 自动定位的，半径来自 YOLO 框、不可信"

    if kind == "plastic":
        return "color", (
            "[种子] 会话里有纯色球的 derived/seeds.json → 保留 color，"
            f"用种子取色并锚定半径{shown}")
    if kind and kind != "metal":
        return method, (f"[警告] seeds.json 的 ball_kind={kind!r} 不认识，"
                        f"未自动接管；继续用 {method} 检测器")
    if not kind:
        # ball_kind 为空 = GUI 点选时没记球种类（也可能是 2026-11-06 前的旧会话）。
        # 返回 specular 是**刻意保留的旧会话兼容**（见 docstring），不许粗暴改
        # 成默认 color —— 那会让全部历史镜面球会话失效。但以前只返回一行
        # `[种子]` 开头的说明，混在输出里等于静默：用户明明是纯色塑料球，
        # 却完全没用上 color 而毫无察觉。这里升级成可见的 [警告]，把后果
        # 和覆盖办法写明，兼容行为一字不动。
        return "specular", (
            "[警告] [种子] seeds.json 的 ball_kind 为空（未记录球种类）：按旧会话兼容"
            "仍自动改用 specular + 该种子 —— 纯色塑料球会因此完全不走颜色检测。"
            "若这是纯色球，请显式 --method color（或重新点选时在球种类里选"
            f"『纯色塑料球』）；要覆盖请显式 --method{shown}")
    return "specular", (
        "[种子] 会话里有 derived/seeds.json → 自动改用 specular + 该种子"
        f"（镜面球用 color/hough 检不到；要覆盖请显式 --method）{shown}")


def world_gravity_world(ground_axis: str) -> tuple[float, float, float]:
    """世界系里的重力向量。

    世界系的竖直轴是 **Z**，不是 Y。``ground_axis="z"`` 时 +Z 朝下
    （板法向背离相机），重力沿 +Z；``"-z"`` 时 +Z 朝上，重力沿 -Z。
    这个出口只回答"世界系里 g 是哪一向量"，换到相机系由调用方乘 R^T。
    """
    return (0.0, 0.0, 9.81 if str(ground_axis or "z") == "z" else -9.81)


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


def _apply_world_to_trajectory_payload(
    payload: dict,
    world_arr,
    block: dict,
    base_frame: str,
    world_csv_name: str = "",
) -> tuple[dict, str | None]:
    """把**已经算好**的世界系坐标真的写进 `traj3d.json` 的 payload。

    为什么需要（B-134，2026-11-15 修）：此前这里只写 `coordinate_system` /
    `world_frame` 两个**标签**，`samples[*]["xyz"]` 仍是 `traj3d.csv` 的相机系坐标
    —— 标签说 world、数据说 camera。下游按标签解释就得到错数，而且**看不出错**。
    `world_arr` 在 `_world_block()` 里已经算好了（`(N,4)`，列 0 是 t、列 1-3 是 XYZ），
    这里只是把这份坐标落到 payload，让标签说的话成为事实。

    返回 ``(payload, 跳过原因或 None)``，三条分支：

    - ``world_arr is None``：**主路径**（没开世界系）。如实写 `base_frame`、
      `world_frame=None`，xyz 与 bounds 一字不动；
    - 数量对得上：逐点换 matched 的 xyz、**重算 bounds**（口径同
      `build_trajectory_payload()`）、宣称 `world`、附 `world_csv`；
    - 数量对不上：**不宣称世界系**（宁可退成相机系并说明，也不产出标签与数据
      不一致的文件），原因写进 `world_skipped` 交给调用方打印 —— 走到这一步
      CSV / JSON / 视频都已落地，"少一次换系"不是错误，不许抛。

    纯函数：不读文件、不写文件、不抛异常（只改传进来的 `payload`）。
    """
    samples = payload.get("samples") or []
    if world_arr is None:
        payload["coordinate_system"] = base_frame
        payload["world_frame"] = None
        return payload, None

    idx = [i for i, s in enumerate(samples) if s.get("matched") and s.get("xyz")]
    n_world = len(world_arr)
    if len(idx) != n_world:
        reason = (f"matched 样本数 {len(idx)} 与世界系行数 {n_world} 不一致，"
                  "坐标未换系（世界系见 traj3d_world.csv）")
        payload["coordinate_system"] = base_frame
        payload["world_frame"] = None
        payload["world_skipped"] = reason
        return payload, reason

    pts: list[list[float]] = []
    for j, i in enumerate(idx):
        row = world_arr[j]
        samples[i]["xyz"] = [float(row[1]), float(row[2]), float(row[3])]
        pts.append(samples[i]["xyz"])
    payload["bounds"] = None if not pts else {
        "min": [min(p[k] for p in pts) for k in range(3)],
        "max": [max(p[k] for p in pts) for k in range(3)],
    }
    payload["coordinate_system"] = "world"
    payload["world_frame"] = block.get("world_frame")
    if world_csv_name:
        payload["world_csv"] = world_csv_name
    return payload, None


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
                    choices=ColorConfig.preset_choices(),
                    help="color 取色方式：auto=按画面自动估主色（不需要知道球色），"
                         "其余是固定颜色预设；不传=用配置里的颜色。可选：%s"
                         % (" / ".join(ColorConfig.preset_choices())))
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
    ap.add_argument("--min-radius", type=float, default=None,
                    help="球的最小半径（px）。缺省=自动：下限 8px，上限走尺寸门阶梯")
    ap.add_argument("--max-radius", type=float, default=None,
                    help="球的最大半径（px）。给了它就**关掉**尺寸门自动阶梯（手工优先）")
    ap.add_argument("--no-radius-auto", action="store_true",
                    help="关掉尺寸门自动阶梯（与给 --max-radius 等价；用于复现老行为）")
    ap.add_argument("--hough-max-radius", type=float, default=80.0)
    ap.add_argument("--max-epipolar", default=None, metavar="PX|auto",
                    help="极线门限（px）；缺省平行 2.5 / angled 8，显式值不被覆盖。"
                         "auto=按标定自报极线 RMS×%g 自适应（见 stereo.triangulate."
                         "epipolar_gate_px）" % EPI_GATE_K)
    ap.add_argument("--min-angle-deg", type=float, default=None, metavar="DEG",
                    help="angled 专用：最小交会角（度），默认 5.0（见 "
                         "stereo.angled.AngledMatchConfig）。按本次录像的交会角分布调门；"
                         f"低于几何核数值下限 {MIN_TRIANGULATION_ANGLE_DEG:g}° 的部分不生效"
                         "（近平行时三角化会被 1/深度² 放大，核层先拒）；parallel 下忽略")
    ap.add_argument("--max-reproj-px", type=float, default=None, metavar="PX",
                    help="angled 专用：最大重投影误差（px），默认 3.0（见 "
                         "stereo.angled.AngledMatchConfig）。parallel 下忽略")
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
                    help="跑满 N 对后判一次『录像几何 vs 标定』：判出可疑时分开报"
                         "『检出质量差』或『几何不一致』，但两种情况都**照常出结果、"
                         "不中止**（质量只作标注，结论写进 traj3d.json 的 quality 块）；"
                         "0=关。默认 30。判据见 stereo.triangulate.geometry_verdict")
    ap.add_argument("--joint-refine", choices=("off", "on"), default="off",
                    help="跨视图联合细化：两路共享极线 v、各自解 u，用来修"
                         "『两路独立细化出的圆心各偏一边』的问题"
                         "（实测 v 差中位 -28.1px，导致极线门 0%% 通过）。"
                         "默认 off：这条路径还没有 A/B 实测数据支撑，"
                         "开了会改变圆心，先按 opt-in 放出来")
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

    # ---- 跨视图联合细化（B-166）：只解析开关，模块按需导入 ----
    # 为什么要延迟 import：off 是默认路径，那条路上不该新增一次模块加载，而且模块
    # 缺失时 off 必须照旧能跑 —— 一个没打开的功能不该让整段录像跑不起来。
    joint_refine_on = args.joint_refine == "on"
    jr_keep_floor = None
    if joint_refine_on:
        try:
            from stereo.joint_refine import describe_result, refine_pair
            import stereo.joint_refine as _jr_mod
        except ImportError as exc:
            print(f"[联合细化][错误] 模块不可用（{exc}）；需要 stereo/joint_refine.py")
            return 2
        # 摘要里要报"保留率达到多少才算划算"，这个门限属于模块的判据而不是我的
        # 判断，所以从模块读，取不到就如实不打（不在这里复制一个会漂的常数）。
        _keep = getattr(_jr_mod, "DEFAULT_MIN_SUPPORT_KEEP", None)
        jr_keep_floor = float(_keep) if _keep is not None else None
        # 开就明说"它会改写圆心"，否则拿 on/off 两份 CSV 比深度的人不会知道
        # 差异从哪来，只会以为是标定或检测器不稳定。
        print("[联合细化] 已开启：共享极线 v、各解 u 的联合细化（会改写左右圆心）")

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

    calib_path = session_calibration_file(session_dir)
    rig = StereoRig.load(calib_path)
    # A2（离线侧标记，不拦）：录制链用 is_forced_unusable() 拦下强制保存的坏标定，
    # 但离线链以前完全没读这个字段 —— 2026-11-11 就是坏标定录制被拦、离线照跑。
    # 这里只**标记**：写进 traj3d.json["quality"] + 打横幅，绝不 return/raise/改
    # 退出码。用户原则「质量差也要出结果，只有错误才紧急」在这里同样成立。
    from stereo.record_plan import is_forced_unusable
    forced_unusable = bool(is_forced_unusable(rig))
    if forced_unusable:
        print("=" * 70)
        print("[仅链路测试] 当前 rig 是强制保存的不可用标定：本次离线结果只能用于"
              "链路测试，不得用于数据结论（不中止，产物照常输出）。")
        print("=" * 70)
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
    # 半径门（2026-11-08）：**手工优先**。显式给了 --min/--max-radius 就按老规矩写死；
    # 都不给时下限仍是 8px（去噪与距离无关），上限交给 `SizeLadder` 逐档协商 ——
    # 起因是梨子 r≈205px 被写死的 200 砍掉，而提示还说"球可能太小"。
    ladder = None
    if args.min_radius is not None:
        cfg.detector.min_radius = float(args.min_radius)
    if args.max_radius is not None:
        cfg.detector.max_radius = float(args.max_radius)
    if (args.min_radius is None and args.max_radius is None
            and not args.no_radius_auto):
        from sphere_tracker.detectors.size_ladder import PROBE_FRAMES, SizeLadder

        ladder = SizeLadder(image_h=float(src.image_size[1]),
                            min_radius=float(cfg.detector.min_radius),
                            probe_frames=PROBE_FRAMES)
        print(f"[半径门] 走自动阶梯：{len(ladder.steps)} 档 "
              f"(k={', '.join(f'{k:.2f}' for k in ladder.steps)} × 画面高 "
              f"{src.image_size[1]}px)，每档看 {PROBE_FRAMES} 帧；"
              f"想写死就加 --max-radius")
    cfg.detector.hough_max_radius = float(args.hough_max_radius)
    cfg.predict.enable = False
    if args.preset:
        cfg.detector.color = ColorConfig.preset(args.preset)
    pipe_l = _make_pipeline(cfg, rig, "left")
    pipe_r = _make_pipeline(cfg, rig, "right")
    seed_frame = 0        # 种子落在第几帧；后面要拿它给重建器设"预热期"
    seed_l = seed_r = None
    # 哪些方法**接受**种子：specular 用它做 CSRT 初始化；color 用它取色 + 定半径。
    # 其余方法（hough/motion/fusion/yolo）没有种子的语义，传了只会被忽略。
    takes_seed = method in ("specular", "color")
    seed_asked = bool(args.seed or args.seed_right or args.seed_json
                      or args.auto_seed or seeds_path.exists())
    if takes_seed and seed_asked:
        def _seed_arg(text: str, what: str):
            try:
                parts = [float(v) for v in text.split(",")]
            except ValueError:
                parts = []
            if len(parts) != 3 or parts[2] <= 0:
                print(f"[错误] {what} 要写成 x,y,r 三个数，例如 1190,300,36；现在是 {text!r}")
                return None
            return tuple(parts)

        if args.detect_width and (args.seed or args.seed_right):
            print("[错误] 种子坐标是原图坐标，--detect-width 会让它错位："
                  "给了 --seed/--seed-right 就不要再缩图。")
            return 2
        # 会话里的 seeds.json 也是原图坐标：specular 没种子跑不了 → 直接停；
        # color 可以不用种子（按预设/auto 取色）→ 忽略文件并说明，别让用户
        # 以为"种子生效了"而实际坐标错位。
        ignore_file_seed = False
        if args.detect_width and seeds_path.exists():
            if method == "specular":
                print("[错误] specular 的种子坐标是原图坐标，不能与 --detect-width 同用；"
                      "去掉 --detect-width 再跑。")
                return 2
            print("[提示] --detect-width 生效时 seeds.json 的坐标会错位："
                  "本次忽略种子，按 --preset / 自动取色检测")
            ignore_file_seed = True
        if args.auto_seed and method == "color":
            # color 不需要种子 —— 它在整幅画面里按颜色找球。静默忽略会让用户以为
            # "自动定位生效了"，所以明确说一句。
            print("[自动种子] --method color 不需要 --auto-seed（颜色分割自己找得到球）；"
                  "本次忽略它，球色用 --preset 或种子取色决定")
        elif args.auto_seed and seeds_path.exists():
            # 用户的显式意图不能被静默忽略：已有种子更好（是人工点选的），所以照用它，
            # 但必须说一声"你给的 --auto-seed 这次没生效、原因是什么"。
            print("[自动种子] 会话里已有 derived/seeds.json → 直接用它"
                  "（--auto-seed 不再自动定位）；要强制重新定位请先删掉它，"
                  "或用 --seed/--seed-right 覆盖")
        seed_json = None
        if seeds_path.exists() and not ignore_file_seed:
            try:
                seed_json = load_seeds(seeds_path)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                # specular 没有种子就跑不了，必须停；color 没有种子仍能按颜色检测，
                # 此时停掉等于把"能跑"变成"不能跑"，所以只警告并继续。
                if method == "specular":
                    print(f"[错误] 种子文件不可用：{seeds_path}（{exc}）")
                    print("  重开『② 点选球心』重新保存，或改用 --seed/--seed-right。")
                    return 2
                print(f"[警告] 种子文件不可用：{seeds_path}（{exc}）；"
                      "color 会继续按预设/自动取色检测，但半径没有种子锚")
        elif args.seed_json and not seeds_path.exists():
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
                                    note="命令行 --seed/--seed-right",
                                    ball_kind=("plastic" if method == "color" else "metal"))
                print(f"[{method}] 已写标准 seed：{saved}")
            except Exception as exc:                        # noqa: BLE001
                print(f"[警告] seed 写入失败（继续跑）：{exc}")
        elif seed_json is not None:
            seed_l = (float(seed_json["left"]["x"]), float(seed_json["left"]["y"]),
                      float(seed_json["left"]["radius"]))
            seed_r = (float(seed_json["right"]["x"]), float(seed_json["right"]["y"]),
                      float(seed_json["right"]["radius"]))
            seed_frame = int(seed_json.get("seed_frame", args.seed_frame))
            kind = str(seed_json.get("ball_kind") or "")
            print(f"[{method}] 读标准 seed：{seeds_path}"
                  + (f"（ball_kind={kind}）" if kind
                     else "（ball_kind 为空：未记录球种类）"))
            if method == "color" and kind == "metal":
                print("[提示] 这份种子记的是镜面金属球；color 仍会用它取色，"
                      "但金属球建议 --method specular")
        elif args.auto_seed and method == "specular":
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
                                    note=f"YOLO 在第 {hit_index} 对自动定位（--auto-seed）",
                                    ball_kind="metal")
                print(f"[自动种子] 已写标准 seed：{saved}")
            except Exception as exc:                        # noqa: BLE001
                print(f"[警告] seed 写入失败（继续跑）：{exc}")
        elif method == "specular":
            print("[错误] --method specular 需要球心种子：")
            print("  GUI：第 3 步「点选球（双目 seed）」；")
            print("  或 CLI：--seed 左x,左y,左r --seed-right 右x,右y,右r")
            print("  也可用 --seed-json <seeds.json>；不点选就用 --auto-seed"
                  "（YOLO 自动定位，半径来自框、不可信）。")
            return 2

        if seed_l is not None and seed_r is not None:
            for label, pipe, sd in (("左", pipe_l, seed_l), ("右", pipe_r, seed_r)):
                fn = getattr(pipe.detector, "seed", None)
                if callable(fn):
                    fn(*sd, frame_index=seed_frame)
                else:                                       # 保险：注册表变了也不会静默
                    print(f"[警告] {method} 不接受种子，本次忽略")
                    break
            print(f"[{method}] 左 seed={seed_l} 右 seed={seed_r} 初始化帧={seed_frame}")
            if method == "color":
                print("[color] 种子用途：从点击点周围取 HSV 区间、半径锚在 seed_r 上；"
                      "种子帧若颜色分割无候选，直接按种子输出 Detection")
            if seed_frame > 0:
                print(f"[提示] seed_frame={seed_frame}：前 {seed_frame} 对按设计不会初始化检测；"
                      "除非第 0 帧看不清球，否则建议改成 0")

    elif args.seed or args.seed_right:
        print(f"[警告] {method} 不接受 --seed/--seed-right"
              "（只有 specular / color 用）；已忽略。")
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
        # A5：两个新门限只在 angled 下生效；不传时取 AngledMatchConfig 的
        # 5.0 / 3.0 默认值，老行为逐字不变。
        _min_angle = (5.0 if args.min_angle_deg is None
                      else float(args.min_angle_deg))
        _max_reproj = (3.0 if args.max_reproj_px is None
                       else float(args.max_reproj_px))
        angled_cfg = AngledMatchConfig(max_epipolar_px=epi,
                                       min_angle_deg=_min_angle,
                                       max_reproj_px=_max_reproj)
        rec = AngledReconstructor(rig, angled_cfg)
        # 交会角的**实际**生效值 = max(用户设的, 几何核数值下限)。
        # 不这么说清楚，用户把 --min-angle-deg 调到 3 会以为设置没生效
        # （核层先按 5° 拒了）——2026-11-18 两个 worker 同时提出这个歧义。
        _eff_min_angle = max(float(_min_angle), MIN_TRIANGULATION_ANGLE_DEG)
        print(f"[门限] 实际生效三阈值（angled）：极线 {epi:g}px / "
              f"交会角 {_eff_min_angle:g}° / 重投影 {_max_reproj:g}px")
        if float(_min_angle) < MIN_TRIANGULATION_ANGLE_DEG:
            print(f"       注意：交会角设的是 {_min_angle:g}°，但几何核数值下限是 "
                  f"{MIN_TRIANGULATION_ANGLE_DEG:g}°，{_min_angle:g}~"
                  f"{MIN_TRIANGULATION_ANGLE_DEG:g}° 的点仍会被拒")
    else:
        if args.min_angle_deg is not None or args.max_reproj_px is not None:
            print("[警告] --min-angle-deg / --max-reproj-px 只对 angled 生效；"
                  "当前是 parallel，已忽略")
        # `source_hint` 只进"0 配对"的报错文案：那句话要能直接告诉用户
        # `--seed-json <会话>/derived/seeds.json` 长什么样，重建器自己不知道会话在哪。
        # 用 as_posix()：文案里再拼 "/derived/seeds.json"，混着反斜杠会像坏路径。
        # 离线入口按新原则「凡是质量差也照常出结果」：显式放行 0 配对门禁，
        # 由入口自己打印诊断 + 写 traj3d.json["quality"] 标记。库层默认仍抛，
        # 保护 mono_vs_stereo / profile_stages 等其它调用方的静默失效防呆。
        # 只给 parallel 分支加：angled 用 AngledReconstructor，没有这个门禁。
        rec = StereoReconstructor(
            rig,
            MatchConfig(max_epipolar_px=epi,
                        radius_ratio_expected=(anchor.radius_ratio if anchor else 0.0),
                        radius_ratio_band=(anchor.band if anchor else 0.25)),
            source_hint=session_dir.as_posix(),
            allow_zero_match=True)
    # 种子落在第 N 帧时，前 N 帧**按设计**不会有检测。不告诉重建器的话，它的
    # 零配对门禁会在第 20 帧判"检测器坏了"并抛错 —— 2026-11-01 真实踩到：
    # 用户在 GUI 第 80 帧点选球心，跑离线三维直接报错，看起来像"点选没用"。
    if seed_frame > 0 and hasattr(rec, "warmup_frames"):
        rec.warmup_frames = int(seed_frame)
        print(f"[提示] 前 {seed_frame} 帧记为预热期：零配对判据从第 "
              f"{seed_frame + 20} 帧之后才生效，配对率分母也扣掉这段")
    # 录前几何守卫（`B-131`）：跑满 N 对判一次"录像几何 vs 标定"。2026-11-13 起
    # 判出可疑**不再早停**：分开报「检出质量差 / 几何不一致」，两种都照常出结果
    # （只有错误才紧急），在 stdout 打一条结构化标记，结论写进 traj3d.json["quality"]。
    # 放在入口层而不是库层默认：`mono_vs_stereo` / `profile_stages` 会在同一批坏会话
    # 上跑，默认拦下来会让它们从"出数"变成"崩"；那是入口层的决定，不该由库替它做。
    guard_at = args.geometry_guard if args.geometry_guard > 0 else 0
    guard_armed = bool(guard_at and mode == "parallel")
    if guard_armed:
        print(f"[几何守卫] 跑满 {guard_at} 对后判一次极线几何；与标定对不上时"
              "分开报『检出质量差 / 几何不一致』，都照常出结果、不中止"
              "（--geometry-guard 0 可关）")
    # `--out` 让产物**不落在会话目录里**（2026-10-27 补）。
    #
    # 为什么需要：做对照实验时（换方法 / 换 seed / 换门限）会连着跑好几次，
    # 每次都覆盖 `<session>/derived/`，上一次的结果就没了 —— 要么手工 `cp` 到别处
    # （实测就是这么干的，完事还得记着删），要么反复跑同一份数据。
    # 现在的约定：**不给 `--out` 就还是老行为**（写回会话目录），给了就全写进去。
    # B-161/172：CSV / 视频 / PNG / **标准 JSON** 全部从**同一个落点变量** out_dir
    # 派生。旧版标准 JSON 由 export_session_visualization(session_dir, ...) 写死在
    # session_dir/derived，--out 改不动它（见下方导出处的同一变量注释）。
    out_dir = Path(args.out) if args.out else (session_dir / "derived")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[输出] 全部产物（含标准 JSON）写到 {out_dir}"
          + ("（会话目录保持干净）" if args.out else "（会话目录内）"))
    # 部分帧试跑不要把正式的 traj3d.csv / annotated.mp4 覆盖成 0.17 s
    partial = f"_first{args.max_frames}" if args.max_frames else ""
    csv_path = out_dir / f"traj3d{partial}.csv"
    video_path = out_dir / f"annotated{partial}.mp4"

    if args.max_frames:
        print(f"[部分帧] 只处理前 {args.max_frames} 对；输出加 {partial} 后缀，"
              "不会覆盖正式文件。")

    writer = None
    video_tmp: Path | None = None
    if not args.no_video:
        # 写临时文件，收尾成功再原子改名成正式名：mp4v 的 moov atom 只在 release()
        # 落盘，进程被强杀会留下"存在但打不开"的 annotated.mp4（moov atom not
        # found），查看器 .exists() 判它存在、cv2 却打不开 → 视频黑屏。写 .tmp
        # 后中断只留垃圾，正式名永远是"完整"或"不存在"。
        # 临时名**结尾必须仍是 .mp4**：cv2.VideoWriter 靠后缀认容器，否则
        # writer.isOpened() 返回 False。
        video_tmp = video_path.with_name(
            video_path.stem + ".tmp" + video_path.suffix)
        writer = cv2.VideoWriter(
            str(video_tmp),
            cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 540))
        if not writer.isOpened():
            print("[警告] 打不开标注视频 writer，继续只写 CSV")
            writer = None

    rows: list[list] = []
    trail_l: list[tuple[float, float]] = []
    trail_r: list[tuple[float, float]] = []
    n_pairs = 0
    center_src = {"contour": 0, "box": 0, "seed": 0}   # 圆心口径来源（含种子锚点）
    # 检出质量信号：逐帧累加。debug_snapshot() 每帧覆盖（color.py:712 返回
    # dict(self._diag, ...)），不能收尾时取最后一帧。供守卫把「检出质量差」与
    # 「几何不一致」分开报，结论也写进 traj3d.json["quality"]。
    quality_sig = {
        "gates": {"radius": 0, "circ": 0, "fill": 0, "radial": 0, "score": 0},
        "candidates": 0,
        "biggest_gate_frames": 0,
        "seed_anchor_frames": 0,
        "frames": 0,
    }
    guard_quality: dict | None = None   # 守卫判读结论（没判过 / 不可疑时保持 None）
    # 联合细化的计数与样本（off 时是一个没人读的空容器，不产生任何输出）。
    # 计数和样本分开：dv / 保留率会被 NaN 或"算不出来"过滤掉，计数不能因此少报。
    jr_stats = {
        "applied": 0, "skipped": 0,
        "support_short": 0, "no_gain": 0, "other": 0,
        "other_reasons": {},  # "其它"桶里各句 reason 的首段原文计数（便于看清那几对为何被拒）
        "dv_before": [],     # 校正前 v 差 —— "现状有多差"，是输入属性
        "retention": [],     # 共享 v 处的支撑保留率 —— 这个方法的成本
    }
    jr_first_result = None
    t0 = time.perf_counter()
    try:
        while True:
            ok, left_frame, right_frame = src.read()
            if not ok or left_frame is None or right_frame is None:
                break
            if args.max_frames and n_pairs >= args.max_frames:
                break
            t = n_pairs / fps
            # 尺寸门阶梯（2026-11-08）：每帧前把当前档位的窗口写回 cfg（检测器读的是
            # `self.cfg.min_radius/max_radius`，两路共用同一个 cfg 对象，所以一次就够）。
            if ladder is not None and not ladder.done:
                _lo, _hi = ladder.window
                cfg.detector.min_radius, cfg.detector.max_radius = _lo, _hi
            res_l = pipe_l.process(left_frame, t)
            res_r = pipe_r.process(right_frame, t)
            det_l = _as_detection(res_l)
            det_r = _as_detection(res_r)
            if ladder is not None and not ladder.done:
                # `r_blocked` = 被**尺寸门单独**挡住的最大候选半径。只认它，
                # 不认"被圆度挡掉"的候选 —— 放大窗口救不了形状。
                _snap = getattr(pipe_l.detector, "debug_snapshot", dict)()
                # 点选种子的兜底输出**不算检出**：那是"把点选位置原样返回"，
                # 拿它当命中会让阶梯在第一档就锁死（梨子那次实测 12/12 假命中）。
                # 注意要读 `res.detection` 而不是 `_as_detection()` 的返回值 ——
                # 后者在跟踪生效时会**新建**一个 Detection，把 extra 丢了。
                _raw = getattr(res_l, "detection", None)
                _anchor = bool((getattr(_raw, "extra", None) or {}).get("seed_anchor"))
                _real = _raw is not None and not _anchor
                _st = ladder.observe(_real, _snap.get("r_blocked") or None)
                if _st == "step":
                    print(f"[半径门] {ladder.attempts[-2].line()} → 升档")
                elif _st in ("hit", "no_candidate", "exhausted"):
                    for _line in ladder.history_lines():
                        print(f"[半径门] {_line}")
                    print(f"[半径门] {ladder.describe()}")
            # 跨视图联合细化（B-166，默认 off）：只有两路都有检测时才成对做，
            # 单边缺检测没有任何"跨视图"可对齐。灰度**每帧每路只转一次**：cvtColor
            # 是这段里最容易被忘掉的重复开销（一帧转两遍等于白烧一遍）。
            if joint_refine_on and det_l is not None and det_r is not None:
                jr = refine_pair(_gray(left_frame), _gray(right_frame),
                                 det_l, det_r, rig, search_px=None)
                if jr_first_result is None:
                    jr_first_result = jr
                if jr.applied:
                    jr_stats["applied"] += 1
                else:
                    jr_stats["skipped"] += 1
                    # 只按 reason 的字面分桶（含"支撑"→支撑不足，含"改善"→无改善，
                    # 其余→其它）。不去读模块的私有常量：reason 是它对外的契约，
                    # 加个新判据时这里最多分错桶，不会跟着崩。
                    why = str(jr.reason)
                    if "支撑" in why:
                        jr_stats["support_short"] += 1
                    elif "改善" in why:
                        jr_stats["no_gain"] += 1
                    else:
                        jr_stats["other"] += 1
                        # 归一化后再聚：reason 里带每次的数值（"v 差 44px"、"12px"），
                        # 原样当键会让每一对各自成一类，"多为"永远只数到 1 —— 那就
                        # 失去了"这一类到底以什么原因为主"的意义。数字一律压成 #。
                        key = _JUNK_NUMBER.sub("#", why.split("（")[0])[:24]
                        jr_stats["other_reasons"][key] = (
                            jr_stats["other_reasons"].get(key, 0) + 1)
                # 模块在"半径太小 / 未校正 / 尺寸不一致"这几条**早退**路径上会把
                # dv 填成 NaN。NaN 混进中位数会把整行变成 nan，所以只收有限样本。
                if math.isfinite(jr.dv_before):
                    jr_stats["dv_before"].append(float(jr.dv_before))
                # 保留率只统计**被采纳**的那些：共享 v 只有被采纳时才真正作用于
                # 输出；被拒的对拿来算"在共享 v 上还剩多少支撑"没有对象。
                ret = _support_retention(jr)
                if ret is not None and jr.applied:
                    jr_stats["retention"].append(ret)
                # 不采纳时 res.left/res.right 就是输入原值，所以这里无条件重建：
                # extra 里的 joint_refine 标记要覆盖每一对，而不只是成功的那批。
                det_l = _joint_detection(det_l, jr.left, jr.applied)
                det_r = _joint_detection(det_r, jr.right, jr.applied)
            pt = rec.update(t, det_l, [det_r] if det_r is not None else [])
            n_pairs += 1

            # ---- 检出质量信号：逐帧累加（debug_snapshot 每帧覆盖，不能收尾取一次）----
            # color.py:712 的 debug_snapshot() 返回 dict(self._diag, ...)，_diag 每帧
            # 被 _diagnose 覆盖，所以这里必须自己累加 gates / biggest / anchor。
            # 非 color 检测器用 getattr 兜底，缺方法 / 缺键都不许让主循环崩。
            quality_sig["frames"] += 1
            _frame_anchor = False
            _frame_big_gate = False
            for _res, _pipe in ((res_l, pipe_l), (res_r, pipe_r)):
                _snap = getattr(getattr(_pipe, "detector", None),
                                "debug_snapshot", lambda: {})()
                if isinstance(_snap, dict):
                    _g = _snap.get("gates") or {}
                    if isinstance(_g, dict):
                        for _k in quality_sig["gates"]:
                            _v = _g.get(_k, 0)
                            if isinstance(_v, (int, float)):
                                quality_sig["gates"][_k] += int(_v)
                                quality_sig["candidates"] += int(_v)
                    _big = _snap.get("biggest") or {}
                    if (isinstance(_big, dict)
                            and _big.get("gate") in ("circ", "fill", "radial", "score")):
                        _frame_big_gate = True
                _det = getattr(_res, "detection", None)
                _extra = getattr(_det, "extra", None) or {}
                if _extra.get("seed_anchor"):
                    _frame_anchor = True
                elif _det is not None:
                    quality_sig["candidates"] += 1     # 过门被采纳的候选
            if _frame_anchor:
                quality_sig["seed_anchor_frames"] += 1
            if _frame_big_gate:
                quality_sig["biggest_gate_frames"] += 1

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
            # 判出可疑**不再早停**：分开报「检出质量差 / 几何不一致」，两种都照常
            # 出结果。放宽门限能"跑通"但那些点本来就是错的（B-131），所以结论照给、
            # 只是不中止；只有错误才紧急。结构化标记行供 GUI 后续消费。
            if guard_armed and n_pairs >= guard_at:
                guard_armed, v = _guard_step(guard_armed, guard_at, n_pairs,
                                             rec.dy_samples, getattr(rig, "quality", None))
                if v is not None and v.suspect:
                    _tot = center_src["contour"] + center_src["box"]
                    _box = (center_src["box"] / _tot) if _tot else None
                    _frames = max(quality_sig["frames"], 1)
                    sig = {
                        "gates": dict(quality_sig["gates"]),
                        "candidates": quality_sig["candidates"],
                        "biggest_gate_share": quality_sig["biggest_gate_frames"] / _frames,
                        "center_box_share": _box,
                        "both_detected_rate": len(rec.dy_samples) / max(n_pairs, 1),
                        "seed_anchor_share": quality_sig["seed_anchor_frames"] / _frames,
                    }
                    verdict = classify_guard(v, sig)
                    print()
                    if verdict == "detection_quality":
                        # 检出质量差：新文案（不含「重新标定」，指向形状门/预设/检测器）
                        print(rec.geometry_failure_message(
                            v, _box, cause="detection_quality", sig=sig))
                        _note = "检出质量差，结果仅供参考"
                    else:
                        # 几何不一致：原文案一字不动（含第 1/2/3 步与「重新标定」）
                        print(rec.geometry_failure_message(v, _box))
                        _note = "录像几何与标定不一致，结果仅供参考"
                    guard_quality = {
                        "verdict": verdict,
                        "grade": "degraded",
                        "degraded": True,
                        "degraded_reason": _note,
                        "warnings": list(getattr(rec, "warnings", [])),
                        "dy_med_px": v.med,
                        "epipolar_rms_px": v.epipolar_rms,
                        "ratio": v.ratio,
                        "match_rate": rec.match_rate,
                        "both_detected_rate": sig["both_detected_rate"],
                        "gate_rejects": dict(quality_sig["gates"]),
                        "center_box_share": _box,
                        "note": _note,
                    }
                    # 结构化、可被 GUI 消费的一行标记（GUI 状态通道统一接这里）。
                    print(f"[质量] verdict={verdict} dy_med={v.med:.2f} "
                          f"ratio={v.ratio:.1f} 结果仅供参考"
                          "（详见 traj3d.json[\"quality\"]）")

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

    # 守卫收尾：按最终分类分别说清「没样本 / 检出质量差 / 几何不一致」。
    # guard_armed 仍为 True = 武装过但没判成（样本不足）；guard_quality 非 None
    # = 判出可疑（已分开报、且照常出结果，没有中止）。两者都 False/None 时不插话。
    if guard_armed:
        print(f"[几何守卫] 没能判读：两路同时检出的帧只有 {len(rec.dy_samples)} 个，"
              f"不足 {GEOM_MIN_SAMPLES} 个。这是**没样本**，不是**没问题** —— "
              "别拿它当几何 OK 的证据。")
    elif guard_quality is not None:
        print(f"[几何守卫] 已判读：{guard_quality['verdict']} —— "
              f"{guard_quality['note']}（结果已照常输出，未中止）")

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

    # release 已把 moov atom 写进 .partial，现在原子改名成正式名。只有跑到这里
    # （读完整段 + 写完 CSV）才会发生，所以 annotated.mp4 要么完整、要么不存在，
    # 不再有"存在但 moov 缺失"的坏文件骗过 .exists()。
    if video_tmp is not None and video_tmp.exists():
        os.replace(video_tmp, video_path)

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
    # A1：一行可读的分项归因统计。口径直接取重建器的计数器与配置（与写进
    # traj3d.json["quality"] 的值同源），不在入口重算一遍。
    _q_cfg = getattr(rec, "cfg", None)
    print(f"[归因] {mode_label(mode)} 分项：n_matched={rec.n_matched} "
          f"n_no_left={rec.n_no_left} n_no_right={rec.n_no_right} "
          f"n_rejected={rec.n_rejected}；F={getattr(rec, 'f_source', 'n/a')}；"
          f"有效门限 极线={epi:g}px "
          f"交会角={getattr(_q_cfg, 'min_angle_deg', 'n/a')}° "
          f"重投影={getattr(_q_cfg, 'max_reproj_px', 'n/a')}px")
    # 联合细化只在 on 时打这一行：off 的 stdout 必须与改动前逐字一致，否则
    # "同一段输入、只有开关不同"的对照实验就没法直接 diff 了。
    if joint_refine_on:
        print(_joint_refine_summary(jr_stats, jr_keep_floor))
        if jr_stats["applied"] == 0 and jr_first_result is not None:
            # 一对都没采纳时把**第一对的原文**打出来：这是"为什么没采纳"的唯一
            # 证据。绝不去调阈值凑采纳数 —— 采纳数为 0 本身就是有价值的结论
            # （轮廓支撑太弱），把它抹平只会让人以为这个方法不可行/可行都无从判断。
            print(f"[联合细化] 一对都没采纳；首对：{describe_result(jr_first_result)}")
    # 配对率低时把"几何对不对"量出来：用户看到的 3.3% 背后常常是"这段录像和
    # 它的标定对不上"，不说出来，下一个人还会去调 --max-epipolar（而那不是原因）。
    # 0 配对现在**不再抛异常**（入口 allow_zero_match=True 放行），会走到这里；
    # 下面这段分项诊断 + zero_match 质量标记就是「不中止但不静默」的落点。
    for line in rec.low_match_diagnostic():
        print(line)
    if n_pairs and rec.n_matched == 0:
        # 库层默认 0 配对会抛 _zero_match_message()（其它调用方的防呆）；离线入口
        # 放行了门禁，这里自己把「为什么 0 匹配 / 下一步查什么」说清楚，别把
        # 「不中止」退化成「什么都不说」。A 守卫若已给出结论则沿用，不覆盖。
        if guard_quality is None:
            guard_quality = {
                "verdict": "zero_match",
                "grade": "degraded",
                "degraded": True,
                "degraded_reason": "零配对：整段没有配对成功的三维点",
                "warnings": list(getattr(rec, "warnings", [])),
                "dy_med_px": None,
                "epipolar_rms_px": None,
                "ratio": None,
                "match_rate": rec.match_rate,
                "both_detected_rate": len(rec.dy_samples) / max(n_pairs, 1),
                "gate_rejects": dict(quality_sig["gates"]),
                "center_box_share": None,
                "note": "整段 0 配对，结果仅供参考（无可用三维点）",
            }
            print(f"[质量] verdict=zero_match match_rate={rec.match_rate:.2f} "
                  "结果仅供参考（详见 traj3d.json[\"quality\"]）")
        print("[诊断] 0 匹配：结果照常输出，但这段没有可用三维点；"
              "先别继续调 pairing，按下面顺序查：")
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
    world_csv: Path | None = None
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
        # 世界系的竖直轴是 **Z**，不是 Y。ground_axis="z" 时 +Z 朝下（重力 +Z），
        # "-z" 时 +Z 朝上（重力 -Z）。原来写死 (0,-9.81,0) 等价于"世界 Y 朝上"，
        # 与本项目两种 ground_axis 都不符 —— 预测的重力项会整体指错。
        g_cam = tuple(float(v) for v in (
            R_wc.T @ np.asarray(
                world_gravity_world(getattr(world, "ground_axis", "z")))))
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
            # B-161/172：export 内部硬编码读写 `<第一参数>/derived`，为了让它与
            # 其它产物落在**同一个 out_dir**：csv_name 传绝对路径（pathlib 的 `/`
            # 遇绝对路径会覆盖前缀 → export 读到 out_dir 的新 csv，不会误读会话
            # 里残留的旧 csv）；再把 export 写的两个 JSON 搬到 out_dir，会话目录
            # 保持干净。不给 --out 时 out_dir == session_dir/derived，paths 已
            # 指向 out_dir，下面整段等价于原地不动（向后兼容）。
            paths = export_session_visualization(
                session_dir, csv_name=str(csv_path.resolve()), stereo_mode=mode,
                world_frame_path=vis_world_path)
            traj_path = out_dir / "traj3d.json"
            vis_json_path = out_dir / "visualization.json"
            if paths["trajectory"] != traj_path:
                traj_path.write_text(paths["trajectory"].read_text(encoding="utf-8"),
                                     encoding="utf-8")
                vis_json_path.write_text(paths["visualization"].read_text(encoding="utf-8"),
                                         encoding="utf-8")
                paths["trajectory"].unlink(missing_ok=True)
                paths["visualization"].unlink(missing_ok=True)
            print(f"[输出] 标准轨迹 JSON：{traj_path}")
            print(f"[输出] 可视化 JSON：{vis_json_path}")
            # 坐标系与预测块写进 traj3d.json：export 刚重建过它，必须在之后补
            payload = json.loads(traj_path.read_text(encoding="utf-8"))
            payload["source_csv"] = csv_path.name   # export 记的是绝对路径，改回会话相对名
            # 世界系结论（B-134，2026-11-15）：以前这里只写 `coordinate_system` /
            # `world_frame` 两个**标签**，samples 还是 traj3d.csv 的相机系坐标 ——
            # 标签说 world、数据说 camera。`world_arr` 上一段早就算好了，现在把这份
            # 坐标真的写进 payload（口径见 `_apply_world_to_trajectory_payload`）。
            payload, world_skip = _apply_world_to_trajectory_payload(
                payload, world_arr, world_block, base_frame,
                world_csv_name=("" if world_csv is None else world_csv.name))
            if world_skip:
                print(f"[警告] {world_skip}")
            if pred_rows:
                payload["predictions"] = {k: pred_block[k]
                                          for k in ("t_ref", "state", "points")}
            else:
                payload["predictions"] = None
                payload["predictions_skipped"] = pred_block.get("skipped", "原因未知")
            # 几何守卫结论（2026-11-13）：分开报「检出质量差 / 几何不一致」，两者都
            # 照常出结果，这里把结论写进标准 JSON 供 GUI / 下游消费。加一层不冲突：
            # 现有键都是后补的，selftest_stereo_pipeline.py 只断言 unit/n_matched/samples。
            if guard_quality is None:
                payload["quality"] = {
                    "verdict": "ok", "grade": "ok", "degraded": False,
                    "degraded_reason": None, "warnings": list(getattr(rec, "warnings", [])),
                    "dy_med_px": None, "epipolar_rms_px": None,
                    "ratio": None, "match_rate": rec.match_rate,
                    "both_detected_rate": None, "gate_rejects": {},
                    "center_box_share": None,
                    "note": "几何守卫未判出可疑（或未启用）",
                }
            else:
                _q = dict(guard_quality)
                # 没有标定自报值时 ratio / epipolar_rms 会是 nan：写成 null，免得
                # 下游（非 Python 消费者）解析 NaN 失败。
                for _kk in ("dy_med_px", "epipolar_rms_px", "ratio"):
                    if isinstance(_q.get(_kk), float) and not math.isfinite(_q[_kk]):
                        _q[_kk] = None
                payload["quality"] = _q
            # A1+A2：离线产物可归因。只从重建器 counter / cfg 与本次门限取值，
            # 不在这里重算任何统计量。**只补键、不删键**（尤其 guard_quality 的
            # verdict / degraded，以及另一路刚加的 round_trip 都在 payload 里）。
            _q = payload["quality"]
            _q["n_no_left"] = int(getattr(rec, "n_no_left", 0) or 0)
            _q["n_no_right"] = int(getattr(rec, "n_no_right", 0) or 0)
            _q["n_rejected"] = int(getattr(rec, "n_rejected", 0) or 0)
            _q["f_source"] = str(getattr(rec, "f_source", "n/a"))
            _q["max_epipolar_px"] = float(epi)
            _q["min_angle_deg"] = getattr(_q_cfg, "min_angle_deg", None)
            _q["max_reproj_px"] = getattr(_q_cfg, "max_reproj_px", None)
            _q["calibration_forced_unusable"] = forced_unusable
            # 平滑块：**总是**写键（没开就写 null），这样"旧文件没这个键"与
            # "这次没开平滑"能分开；points 与 samples 同坐标系，不做世界系变换。
            payload["smoothing"] = smoothing_block
            if smoothing_block is None and smoothing_skip:
                payload["smoothing_skipped"] = smoothing_skip
            traj_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
            # 往返一致性（2026-11-15）：`build_trajectory_payload()` 已经把它算进
            # payload["round_trip"]（几何属性，每次导出重算，不靠 carried_over）。
            # 这里只打**一行**结论；没有来回的数据（no_pairs）不刷噪音。
            # 打印失败不影响主流程 —— 走到这里 CSV / 两个 JSON / 视频 都已落地。
            try:
                rt = payload.get("round_trip") or {}
                rt_verdict = rt.get("verdict")
                dy_mm, dz_mm = rt.get("dy_median_mm"), rt.get("dz_median_mm")
                if (rt_verdict and rt_verdict != "no_pairs"
                        and dy_mm is not None and dz_mm is not None):
                    same = ("全部同号" if rt.get("sign_consistent")
                            else "符号不一致")
                    tail = {
                        "ellipse": "轨迹是窄椭圆，不是噪声",
                        "retrace": "去程与回程重合，属于原路返回",
                        "scattered": "偏移方向不固定，更像检测抖动（详见 quality）",
                    }.get(rt_verdict, str(rt.get("note") or ""))
                    print(f"[往返] {int(rt.get('n_pairs') or 0)} 段："
                          f"ΔY 中位 {dy_mm:+.1f} mm、ΔZ 中位 {dz_mm:+.1f} mm"
                          f"（{same}）→ {tail}"
                          f"（详见 traj3d.json[\"round_trip\"]）")
            except Exception as rt_exc:                     # noqa: BLE001
                print(f"[警告] 往返一致性摘要跳过：{rt_exc}")
            # 2026-11-15：同一个 quality dict 也给 visualization.json 带一份（GUI /
            # 3D 查看器只读这个文件，不该为了"这段结果可不可信"再去隔壁翻 traj3d）。
            # **不另造 schema**：直接复用上面那个 payload["quality"]；只补顶层键，
            # 原有字段一字不动。读不出来就跳过并留一行日志（不许抛 —— 走到这里
            # CSV / traj3d.json / 视频 都已经落地了，"少一个标记"不是错误）。
            try:
                vis_payload = json.loads(vis_json_path.read_text(encoding="utf-8"))
                vis_payload["quality"] = payload["quality"]
                vis_json_path.write_text(
                    json.dumps(vis_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")
            except Exception as vis_exc:                    # noqa: BLE001
                print(f"[警告] visualization.json 补写 quality 跳过：{vis_exc}")
        except Exception as exc:                            # noqa: BLE001
            print(f"[警告] 标准 JSON 导出失败：{exc}")
    # 新原则「只有错误才紧急」：0 配对时 CSV / 两个 JSON / 正式 mp4 已全部写完，
    # quality.verdict="zero_match" 标记齐全，返回 0 让 _gui_console / run_all 判完成；
    # 返回 2 会被显示成失败，与「照常出结果」冲突。
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
