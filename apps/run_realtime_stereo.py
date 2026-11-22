"""双目实时：两路相机 -> 逐帧检测 -> 极线配对 -> 三角测量 -> 三维轨迹。

**两种装置都支持**（2026-11-16 用户裁决："平行双目和大角度双目各有一个"）：

| 标定标称 | 重建器 | 像素口径 |
| --- | --- | --- |
| ``parallel`` + ``stereo_rectify`` | `StereoReconstructor` | 极线校正后 |
| ``angled``（含 ``none_raw``） | `AngledReconstructor` | **原始**像素 + 基础矩阵 F |

分发**只有一处**（`make_reconstructor()`），主循环不按模式写两个分支。
``none_raw`` 却标着 ``parallel`` 的标定仍然拒绝 —— 那是"文件写错"，不是
"自动猜成另一个模式"能解决的（理由见该函数的 docstring）。

退出时会**落成一个会话**（默认开，`--no-save-session` 可关）：
``data/stereo/<模式>/<时间戳>/``，里面是 `session.json` + ``derived/`` 下的
``traj3d.csv`` / ``seeds.json`` / ``traj3d.json`` / ``visualization.json`` /
``annotated.mp4``（纯并排）。于是实时跑出来的轨迹在「数据管理」里看得见、
在「3D 查看器」里打得开。三维点少于 `MIN_SESSION_POINTS` 不留条目。

用法::

    conda activate myenv
    python apps/run_realtime_stereo.py --left 0 --right 1
    python apps/run_realtime_stereo.py 0 1 --method color --preset white_ball
    python apps/run_realtime_stereo.py --auto-cameras --detect-width 960
    python apps/run_realtime_stereo.py --calib config/calib/实验性质大角度1

快捷键（与单目 `run_realtime` 同一套，另加 ``3`` = 三维小窗）：

    q / ESC  退出
    空格     暂停 / 继续
    r        重置两路轨迹 + 三维点
    s        存快照（并排标注图 + 两路原图）
    3        开/关三维小窗（画布右侧独立一列）
    c        在 color / motion 之间切换检测方法（**两路一起换**）
    v        开始 / 停止录制
    f        全屏（全屏时 Esc 先退全屏）

为什么另开一个文件，而不是给 `run_realtime.py` 加一个 ``--right``
---------------------------------------------------------------
`SpherePipeline` 有状态（`history3d` / `_prev_gray` / tracker，`pipeline.py:85-100`），
**一个实例只能服务一台相机**：两路共用一个实例，右路的帧会被当成左路的下一帧，
跟踪速度、相机自运动、三维历史全部互相污染 —— 而这种错**不会报错**，只是数字慢慢变歪。
所以这里是"两个 pipeline + 两个 Health + 一个 PairSource"。

三个口径必须写死（都是踩过的坑）
--------------------------------
1. **时间轴用采集时刻**（`PairSource.read_pair()` 给的 `t_l` / `t_r`），
   不在循环里现取 `perf_counter()`。后者是"我开始处理它的时刻"，
   随负载后移且抖动，而预测器吃的就是这条轴（`run_realtime.py:298-303`，
   交接文档 §P0-1：滞后 1 帧 -> 三维误差 440 mm）。
2. **`ok=False` 时绝不产出三维点**。一路掉线时 `LatestFrameCamera.latest()`
   仍返回**冻结的旧帧**，拿它硬算出来的点比没有点更糟（`sources.py:301-303`）。
3. **skew 一阶补偿 + 超限丢弃**：没有硬件触发线就不可能同时曝光，
   只能"测出时间差 -> 按像素速度退回较早时刻 -> 超限就丢这一对并计数"。
   `skew` / `res` 常驻 HUD，**绝不静默**。

质量只显示、不中止
------------------
`GeometryVerdict.suspect`（录像几何与标定对不上）在这里只标一行 + 计数，
不 `return` 非 0：实时窗口里"少一个点"和"整段停下来"是两种代价，
用户要的是**能看见问题还继续**（同 `run_stereo_video` 的「质量差也出结果」）。

可测性
------
判定逻辑全部抽成不碰相机的纯函数：`skew_correct` / `skew_dt_for_side` /
`should_drop` / `compose_hud` / `compose_canvas`，主循环 `run_loop()` 允许注入
假帧源与 `max_frames`，于是 `tests/selftest_stereo_branch.py` 能真跑一遍。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import shutil
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# 允许直接 python apps/xxx.py 运行（把项目根目录加进搜索路径）
# --- 让脚本在任意目录深度都能找到项目根 ---
ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from calib import StereoRig
from calib.params import is_uncorrected
from sphere_tracker.config import AppConfig, ColorConfig, load_config
from sphere_tracker.detectors import build_detector
from sphere_tracker.detectors import registry  # noqa: F401  import 即注册（R1）
from sphere_tracker.geometry import CameraModel
from sphere_tracker.paths import stereo_mode_root
from sphere_tracker.pipeline import SpherePipeline
from sphere_tracker.plugins import names as method_names
from sphere_tracker.sources import quiet_repeated_grab_warnings
from sphere_tracker.stats import Health
from sphere_tracker.types import Detection
from sphere_tracker.viz import draw_3d_panel
from stereo import MatchConfig, StereoReconstructor
from stereo.angled import AngledMatchConfig, AngledReconstructor
from stereo.stereo_mode import mode_label, normalize_mode
from stereo.triangulate import epipolar_gate_px, geometry_verdict
from apps._common import (
    add_stereo_source_args,
    current_calibration,
    log_environment,
    open_pair_source,
    setup_logging,
)
from apps._window import WindowManager

_log = logging.getLogger("realtime_stereo")

WINDOW = "sphere tracker (stereo real-time)"

#: 实时只允许这两个检测方法。依据是**实测**（`pipeline.py:190-193`，1080p 真实纹理）：
#: `color` 17.8 ms/路、`motion` 15.5 ms/路；而 `hough` **4384.9 ms**、`fusion`
#: **3740.5 ms** 一帧 —— 两路串起来就是 8 秒一帧，不叫实时（B-16 已定性）。
REALTIME_METHODS = ("color", "motion")
#: `c` 键的轮换表：**从注册表里过滤**，而不是手写一份（R7：清单只有一处产地）。
METHOD_CYCLE = tuple(n for n in method_names("detector") if n in REALTIME_METHODS)

#: 两路时间差的默认门限（毫秒）。超过就丢这一对：33 ms 是 30 fps 的一帧，
#: 60 ms 已经是"两帧"，一阶补偿的残差会超过它本来要消掉的量。
#:
#: ⚠️ 这个数字的语义是"**30 fps 下的两帧**"，不是绝对毫秒 —— 高帧率相机要用
#: :func:`default_max_skew_ms` 按帧率换算，别直接拿去用（500 fps 下 60 ms 是 30 帧）。
DEFAULT_MAX_SKEW_MS = 60.0


def default_max_skew_ms(fps: float) -> float:
    """两路时间差门限，按 ``DEFAULT_MAX_SKEW_MS`` 的原始语义（30 fps 下 60 ms）随帧率缩放。

    ``fps <= 30`` 时**原样返回 60.0** —— 现有 24/30 fps 相机的行为逐字节不变。
    高帧率下按 ``60 * 30 / fps`` 收紧（500 fps → 3.6 ms），并设 2 ms 下限：
    主机读帧时间戳本身的抖动就有 1~2 ms，再紧就变成"随机丢一半的帧对"。
    """
    f = float(fps)
    if f <= 0 or f <= 30.0:
        return DEFAULT_MAX_SKEW_MS
    return max(DEFAULT_MAX_SKEW_MS * 30.0 / f, 2.0)


#: 补偿门禁：跟踪器命中不足 3 次时速度还不可信（前两帧的速度基本是"首帧差"），
#: 拿它去外推等于把噪声乘上 33 ms。
SKEW_MIN_HITS = 3
#: 检测默认宽度：全分辨率 color 17.8 ms/路，两路 35.6 ms 达不到 >=20 fps 的目标；
#: 960 宽像素只有 1/4，实测 4~5 ms/路（`pipeline.py:197-202` 口径）。
DEFAULT_DETECT_WIDTH = 960
#: 显示宽度（**整幅并排图**的宽度，对半给每路）。imshow 一张 1920x1080 实测 17.74 ms，
#: 先缩到 960x540 只要 1.84 ms —— 这是单目实时里最有效的一步（见 `run_realtime.py:110-118`）。
DEFAULT_DISPLAY_WIDTH = 1600
#: 连续取不到成对帧这么多对之后放弃。相机真掉线时 `read_pair()` 每次内部要等 5 s，
#: 50 次 = 4 分钟，足够人反应过来；而单纯"这次没新帧"不会连续累积（成功一次就清零）。
MAX_CONSECUTIVE_READ_FAILS = 50
#: 大角度模式的极线门限（原始像素）。与 `AngledMatchConfig` 的默认值同源，
#: 也与离线 `run_stereo_video` 给 angled 的默认值一致（parallel 是 2.5）。
ANGLED_MAX_EPIPOLAR_PX = 8.0
#: 落会话的最低三维点数。低于它**不建目录**：实时窗口可能只是"打开看一眼"，
#: 每次退出都在数据管理里留一条空会话，几天后那个列表就没法用了。
MIN_SESSION_POINTS = 30
#: 几何判读只看最近这么多条 dy 样本（HUD 是"现在准不准"，不是"历史平均"）。
GEOM_TAIL = 300
#: 三维小窗只画最近这么多点（再多也看不清，画线是 O(n)）。
PANEL_POINTS = 300

#: 顶栏第二行常驻的操作提示（2026-11-17 用户：底部状态条放长句会被截断，
#: 而"按哪个键干什么"是每时每刻都要用的信息，应该出现在画面上方）。
#: 只写**键 + 动作**，不写解释 —— 解释在「操作说明」里。
REALTIME_HINT = ("q 退出 · 空格 暂停 · 3 三维小窗 · v 录制画面 · b 录制(画面+原始) · "
                 "s 快照 · c 切方法 · f 全屏 · r 重置")

# ---------------------------------------------------------------------------
# 纯函数：判定逻辑（不碰相机、不碰窗口，判据直接调它们）
# ---------------------------------------------------------------------------
def skew_correct(u: float, v: float, vx: float, vy: float,
                 dt_s: float) -> tuple[float, float]:
    """把某时刻的圆心按像素速度**线性外推**到另一时刻。

    ``dt_s`` 是**有符号**的时间差 = 目标时刻 - 这一帧自己的时刻（秒）：

    - ``dt_s > 0``：往前推（预测）；
    - ``dt_s < 0``：往回退（**补偿较晚那一路**走的就是这条，见 :func:`skew_dt_for_side`）；
    - ``dt_s == 0``：原样返回，一个浮点噪声都不加。

    为什么只做到一阶：没有硬同步，两路的时间差只能靠速度消。匀速段这一项
    精确，急停/急转段残差变大 —— 那是这套方案的**边界**，不是 bug（见 HUD 的 res）。
    """
    if dt_s == 0.0:
        return float(u), float(v)
    return float(u) + float(vx) * float(dt_s), float(v) + float(vy) * float(dt_s)


def skew_dt_for_side(skew_ms: float | None, side: str) -> float:
    """这一路要外推多少秒。**只有较晚的那一路需要动**，较早的那一路返回 0。

    ``skew_ms > 0`` = 右路更晚 -> 右路退回 ``-skew/1000`` 秒；
    ``skew_ms < 0`` = 左路更晚 -> 左路退回；``None`` / 0 -> 都不动。

    符号是最容易搞反的地方：**目标时刻在更早**，所以 dt 是负数。
    搞反的后果不是崩溃，而是"补偿以后误差更大"—— 判据 5（补偿前后 RMS）
    就是钉这个的。
    """
    if skew_ms is None or skew_ms == 0.0:
        return 0.0
    late = "right" if skew_ms > 0 else "left"
    return -abs(float(skew_ms)) / 1000.0 if side == late else 0.0


def should_drop(skew_ms: float | None, stalled_l: bool, stalled_r: bool,
                max_skew_ms: float | None = DEFAULT_MAX_SKEW_MS
                ) -> tuple[bool, str]:
    """这一对帧该不该丢。返回 ``(是否丢, 原因)``，原因可直接进日志/HUD。

    两条**独立**的门禁，顺序刻意是"先 stalled"：

    1. 任一路采集停住 -> 丢。``latest()`` 会继续返回冻结的旧帧，不拦就会用
       "上一秒的画面 + 这一秒的画面"算出假三维点（`pair_source.py:799-807`）。
    2. ``|skew| > 门限`` -> 丢。一阶补偿只吃匀速，超过两帧的错位连匀速都补不回来。
       ``max_skew_ms=None`` = 不设门限（与 `PairSource` 的 ``max_skew_ms`` 同口径）。
    """
    # ⚠ 反向自证锚点：整段删掉（到下面 `if skew_ms is None` 之前），
    # `tests/selftest_stereo_branch.py` 第 ⑧ 条必须变红（stalled 不再丢帧）。
    if stalled_l or stalled_r:
        sides = "/".join(name for name, bad in
                         (("left", stalled_l), ("right", stalled_r)) if bad)
        return True, f"stalled({sides})"
    if skew_ms is None:
        return True, "skew unknown"
    limit = None if max_skew_ms is None else float(max_skew_ms)
    if limit is not None and abs(float(skew_ms)) > limit:
        return True, f"|skew| {abs(float(skew_ms)):.1f} ms > {limit:g} ms"
    return False, ""


def compose_hud(*, match_rate: float | None = None,
                residual_px: float | None = None,
                disparity_px: float | None = None,
                skew_ms: float | None = None,
                dropped: int = 0,
                note: str = "") -> str:
    """拼双目那一行 HUD（**纯函数**，只格式化）。

    为什么单独一行、不用 `viz.hud_lines`：那个函数拿的是 `FrameResult`，
    **看不到 `StereoPoint`** —— 匹配率、交会残差、视差都是**一对帧**的属性，
    单路的 result 里根本没有。所以左右各保留一份单目 HUD，另加这一行。

    五个键固定出现（`match` / `res` / `disp` / `skew` / `drop`），
    取不到真值时写 ``n/a`` —— 宁可显示"不知道"，也不要显示 0 让人以为测到了。
    """
    match = "  n/a" if match_rate is None else f"{match_rate * 100:3.0f}%"
    res = " n/a" if residual_px is None else f"{residual_px:4.1f} px"
    disp = "  n/a" if disparity_px is None else f"{disparity_px:5.1f} px"
    skew = "   n/a" if skew_ms is None else f"{skew_ms:+6.1f} ms"
    line = (f"stereo  match {match}  res {res}  disp {disp}  "
            f"skew {skew}  drop {int(dropped)}")
    return line + (f"  {note}" if note else "")


def compose_canvas(vis_l: np.ndarray, vis_r: np.ndarray, hud_line: str,
                   panel: np.ndarray | None = None, alert: bool = False,
                   hint: str = "", badge: str = "") -> np.ndarray:
    """左右并排 + 顶部双目 HUD（可带一行操作提示）+ 右侧**独立一列**三维小窗。

    两个教训（2026-11-17 用户截图）：

    * **小窗不许叠在相机画面上**。旧版把它贴在 `w - pw`，也就是右相机画面的
      右上角 —— 4D 轨迹一旦跳起来就盖住画面，看起来像"画面自己遮挡自己"。
      现在画布宽度是 `两路之和 + 小窗宽`，小窗独占最右一列。
    * **提示与指标必须在同一行**（2026-11-17 用户）：先前把按键提示放到第二行，
      结果录制中的「● 录制中 N 帧」把那一行占掉，按键提示直接被顶没 ——
      越是在录制（最需要知道按哪个键停）的时候越看不到。现在：
      指标靠左、按键提示**右对齐**贴在同一个条里，谁也不挤谁。

    * 瞬时回显（按键/录制帧数）走 `badge`：画在**画面左上角**的一块深色角标上，
      两秒后自己消失。它不占顶栏、不顶掉提示，也不改画布高度。
    """
    if vis_l.shape[:2] != vis_r.shape[:2]:
        # 两路分辨率理论上一致（`CameraPairReader` 成对协商）；万一不一致，
        # 缩到左路尺寸继续 —— 这里崩掉等于"相机开起来了一帧就退"，代价更大。
        vis_r = cv2.resize(vis_r, (vis_l.shape[1], vis_l.shape[0]),
                           interpolation=cv2.INTER_AREA)
    body = np.hstack([vis_l, vis_r])
    h, w = body.shape[:2]
    bar = max(22, int(round(h * 0.035)))         # **始终一行**，不再为提示加高
    pw = panel.shape[1] if panel is not None else 0
    canvas = np.zeros((h + bar, w + pw, 3), np.uint8)
    canvas[:] = (24, 24, 24)
    color = (60, 60, 255) if alert else (150, 255, 170)
    fs = max(0.42, min(0.62, bar / 40.0))
    base = bar - max(4, bar // 5)
    cv2.putText(canvas, hud_line, (8, base),
                cv2.FONT_HERSHEY_SIMPLEX, fs, color, 1, cv2.LINE_AA)
    if hint:
        # 右对齐贴在同一行。先量指标占多宽 —— 放不下就**不画**提示（宁可少一行字，
        # 也不要把指标压掉：指标是这一帧的数，提示是随时可查的键位）。
        hfs = max(0.36, fs * 0.84)
        (mw, _), _ = cv2.getTextSize(hud_line, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        (tw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, hfs, 1)
        x = canvas.shape[1] - tw - 8
        if x > mw + 24:
            cv2.putText(canvas, hint, (x, base),
                        cv2.FONT_HERSHEY_SIMPLEX, hfs, (170, 170, 170),
                        1, cv2.LINE_AA)
    canvas[bar:, :w] = body
    if badge:
        # 角标画在**左路画面**左上角：不占顶栏、不顶掉按键提示，也不改画布高度。
        bfs = max(0.42, fs * 0.92)
        (bw, bh), bbase = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, bfs, 1)
        x0, y0 = 8, bar + 6
        cv2.rectangle(canvas, (x0 - 4, y0 - 2),
                      (x0 + bw + 6, y0 + bh + bbase + 4), (30, 30, 30), -1)
        cv2.putText(canvas, badge, (x0, y0 + bh),
                    cv2.FONT_HERSHEY_SIMPLEX, bfs, (60, 210, 255), 1, cv2.LINE_AA)
    if panel is not None:
        ph, pwid = panel.shape[:2]
        # 小窗在**它自己那一列**里垂直居中：贴顶会在下方留一条突兀的黑边，
        # 看着像"没画完"。列宽就是 pw，所以横向不会挤到画面里去。
        hh = min(ph, canvas.shape[0] - bar)
        ww = min(pwid, pw)
        y0 = bar + max(0, (h - hh) // 2)
        canvas[y0:y0 + hh, w:w + ww] = panel[:hh, :ww]
    return canvas


def _as_detection(res) -> Detection | None:
    """优先用跟踪后的圆心；跟踪在滑行时退回原始检测。

    口径与 `apps/run_stereo_video.py:98-105 _as_detection` **逐字一致** ——
    两处不一致的话，实时与离线喂给三角测量的就不是同一个点，
    同一个球在两边会给出不同的三维坐标，而且没人能解释为什么。
    """
    state = res.track
    if state is not None and not state.coasting:
        r = res.detection.radius if res.detection is not None else state.radius
        return Detection(x=state.x, y=state.y, radius=r, score=1.0, method="track")
    return res.detection


def compensate_detections(det_l: Detection | None, trk_l, det_r: Detection | None,
                          trk_r, skew_ms: float | None, *, enable: bool = True,
                          min_hits: int = SKEW_MIN_HITS
                          ) -> tuple[Detection | None, Detection | None, str]:
    """把**较晚那一路**的圆心退回较早时刻。返回 ``(左, 右, HUD 备注)``。

    备注只有三种：``""``（补了或本来就不用补）、``"raw"``（门禁没过，
    **这一帧按原始点算**，HUD 上必须写出来）、``"off"``（``--no-skew-fix``）。

    门禁为什么是 ``hits >= 3``：跟踪器前两帧的速度是"两点差 / 两点时间差"，
    噪声与真速度同量级；拿它乘 33 ms 等于把噪声搬进圆心。宁可这一帧不补，
    也不能补出一个看起来更准的数。
    """
    if not enable:
        return det_l, det_r, "off"
    if skew_ms is None or abs(float(skew_ms)) < 0.5:
        # 亚毫秒的差补不出东西，反而每次都要新建 Detection —— 直接放过。
        return det_l, det_r, ""
    side = "right" if skew_ms > 0 else "left"
    det = det_r if side == "right" else det_l
    trk = trk_r if side == "right" else trk_l
    if det is None or trk is None or getattr(trk, "hits", 0) < min_hits:
        return det_l, det_r, "raw"
    dt = skew_dt_for_side(skew_ms, side)
    u2, v2 = skew_correct(det.x, det.y, trk.vx, trk.vy, dt)
    fixed = Detection(x=u2, y=v2, radius=det.radius, score=det.score,
                      method=f"{det.method}+skew", contour=det.contour,
                      extra=dict(det.extra or {}, skew_corrected=True))
    if side == "right":
        return det_l, fixed, ""
    return fixed, det_r, ""


# ---------------------------------------------------------------------------
# 主循环（可注入假帧源；不碰窗口 —— 显示交给 on_frame 回调）
# ---------------------------------------------------------------------------
@dataclass
class LoopStats:
    """一次运行的累加器（**纯数据**，可序列化，收尾打印/判据都读它）。"""

    pairs: int = 0                       # 成功处理的**帧对**数
    failed_reads: int = 0                # 连续取不到成对帧的次数（成功即清零）
    dropped: int = 0                     # 因 stalled / skew 超限丢掉的对数
    drop_reasons: Counter = field(default_factory=Counter)
    points: int = 0                      # 真的产出了三维点的对数
    match_rate: float | None = None      # 最近一次读到的配对率（没开三维时 None）
    mode: str = "parallel"               # 重建模式：parallel / angled（见 make_reconstructor）
    #: **视频帧序号**：每处理完一对 +1（不是每个匹配点 +1）。
    #: 它与 `pairs` 数值相同，但语义不同 —— 落盘的 `traj3d.csv` 的 `frame` 列、
    #: `annotated.mp4` 的帧序号、查看器的 seek 三者都以它为准，所以单独留名字。
    frame_index: int = 0
    #: 可落盘的 matched 行（列序 = `apps.run_stereo_video.CSV_HEADER`）。
    session_rows: list[list] = field(default_factory=list)
    #: 第一对匹配点的 `(u_l, v_l, r_l, u_r, v_r, r_r, frame)`，写 `derived/seeds.json` 用。
    seed: tuple | None = None
    disparities: list[float] = field(default_factory=list)
    epipolars: list[float] = field(default_factory=list)
    angles: list[float] = field(default_factory=list)
    residuals: list[float] = field(default_factory=list)
    skews: list[float] = field(default_factory=list)
    process_ms: list[float] = field(default_factory=list)
    verdict: object | None = None        # GeometryVerdict（样本不足时 None）
    verdict_suspect: bool = False
    last_vis: np.ndarray | None = None
    last_left: np.ndarray | None = None
    last_right: np.ndarray | None = None
    stop_reason: str = ""

    def note(self, values: list[float], value: float, cap: int = 300) -> None:
        """追加一个样本并**裁剪**。HUD 只看最近一段，但总量不能无限涨。"""
        values.append(float(value))
        if len(values) > cap:
            del values[: len(values) - cap]

    @property
    def loop_fps(self) -> float:
        if not self.process_ms:
            return 0.0
        avg = statistics.fmean(self.process_ms)
        return 1000.0 / avg if avg > 0 else 0.0

    @property
    def skew_abs_median(self) -> float | None:
        return statistics.median([abs(s) for s in self.skews]) if self.skews else None

    @property
    def skew_abs_max(self) -> float | None:
        return max(abs(s) for s in self.skews) if self.skews else None

    def summary_lines(self) -> list[str]:
        """收尾那几行（判据/日志共用同一份口径，避免两处各算一遍）。"""
        med = self.skew_abs_median
        mx = self.skew_abs_max
        if self.process_ms:
            head = (f"帧对 {self.pairs}  平均每对处理 "
                    f"{statistics.fmean(self.process_ms):.1f} ms"
                    f"（约 {self.loop_fps:.1f} FPS）")
        else:
            head = f"帧对 {self.pairs}"
        rate = ("配对率 n/a" if self.match_rate is None
                else f"配对率 {self.match_rate * 100:.1f}%")
        lines = [
            head,
            f"重建模式 {self.mode}  三维点 {self.points}  {rate}",
            f"丢弃 {self.dropped} 对"
            + (f"（{', '.join(f'{k}:{v}' for k, v in self.drop_reasons.items())}）"
               if self.drop_reasons else ""),
            ("|skew| 中位 %.1f ms / 最大 %.1f ms" % (med, mx)
             if med is not None else "|skew| 没有样本（非实时源？）"),
        ]
        # 大角度没有"视差"这个概念（原始像素的 x 差不与深度成反比），
        # 但它有两个**自己的**量：极线距离与交会角。名字分开写，别混。
        if self.mode == "angled":
            if self.epipolars:
                lines.append("极线距离 中位 %.2f px / 最大 %.2f px"
                             % (statistics.median(self.epipolars),
                                max(self.epipolars)))
            if self.angles:
                lines.append("交会角 %.1f–%.1f°" % (min(self.angles),
                                                  max(self.angles)))
        if self.verdict_suspect:
            lines.append("⚠ 几何判读：录像几何与标定对不上（suspect）—— 结果仅供参考")
        return lines


def _feed_health(health: Health, t_abs: float, seq: int, *,
                 detect_ms: float, process_ms: float,
                 prev_t: float, prev_seq: int) -> None:
    """把**一路**的采集/处理健康度喂给自己的 `Health`（两路各一个实例，别混）。

    `Health` 的采集侧只有一套字段（`stats.py:78-79`），两路混用就会得到
    "cap 是两台相加"这种没人能解释的数。所以左路一个、右路一个。
    """
    now = time.perf_counter()
    capture_ms = (t_abs - prev_t) * 1000.0 if prev_t > 0 else 0.0
    age_ms = max(0.0, (now - t_abs) * 1000.0)
    health.begin_frame()
    health.note_capture(capture_ms=capture_ms, frame_age_ms=age_ms)
    if prev_seq >= 0 and seq - prev_seq > 1:
        health.note_dropped(seq - prev_seq - 1)
    health.end_frame(detect_ms=detect_ms, process_ms=process_ms)


def run_loop(source, pipe_l: SpherePipeline, pipe_r: SpherePipeline,
             rec, *,
             stats: LoopStats | None = None,
             health_l: Health | None = None, health_r: Health | None = None,
             skew_fix: bool = True, max_skew_ms: float | None = DEFAULT_MAX_SKEW_MS,
             min_hits: int = SKEW_MIN_HITS, max_frames: int = 0,
             display_width: int | None = None, panel_on=True,
             hint_fn=None, badge_fn=None,
             on_frame=None, log_every: float = 5.0,
             frame_sink=None, collect_rows: bool = True) -> LoopStats:
    """逐对帧跑完整条双目链路。**不打开任何窗口**，显示交给 ``on_frame``。

    返回累加器；``on_frame(canvas, stats)`` 返回 ``False`` 时立即收尾（按 q）。
    这样主程序与判据跑的是**同一段循环** —— 判据不需要相机、也不需要显示器。

    ``panel_on`` 既可以是 bool，也可以是**返回 bool 的零参函数**：
    运行中按 ``3`` 要能立刻生效，而 bool 在进入循环那一刻就被固化了
    （这个坑是"按 3 没反应"的唯一来源）。

    ``source`` 可以直接给帧源实例，也可以给**零参工厂**（`source_factory`）——
    判据要在同一进程里跑好几轮（每轮一对新帧），工厂省掉"谁负责重建"的争论。

    ``frame_sink(vis_l, vis_r)``：每处理完一对帧调一次（**含被丢弃的对**），
    用来边跑边写 `annotated.mp4`。写与不写都不影响帧序号 ——
    序号由 `stats.frame_index` 单独记，`traj3d.csv` 的 `frame` 列用的就是它。
    ``collect_rows``：是否把 matched 行攒进 `stats.session_rows`（不落盘时省内存）。
    """
    if callable(source) and not hasattr(source, "read_pair"):
        source = source()
    stats = stats or LoopStats()
    # 模式只在这里落一次：`make_reconstructor` 决定用哪个重建器，循环只**读**
    # 这个结论（`disp` 只对 parallel 有意义，见下面记录那一段）。
    angled = isinstance(rec, AngledReconstructor)
    stats.mode = "off" if rec is None else ("angled" if angled else "parallel")
    health_l = health_l or Health()
    health_r = health_r or Health()
    t_base: float | None = None
    prev_t = [0.0, 0.0]
    prev_seq = [-1, -1]
    next_report = time.perf_counter() + max(float(log_every), 0.0)

    while True:
        if max_frames and stats.pairs >= max_frames:
            stats.stop_reason = f"max_frames={max_frames}"
            break
        ok, left, right, t_l, t_r, skew, seq_l, seq_r = source.read_pair()
        if not ok or left is None or right is None:
            # 取不到成对帧**不产出三维点**（冻结帧硬算出来的点比没有点更糟）。
            stats.failed_reads += 1
            if on_frame is None or stats.failed_reads >= MAX_CONSECUTIVE_READ_FAILS:
                stats.stop_reason = "取不到成对帧（相机掉线/被占用？）"
                break
            time.sleep(0.002)
            continue
        stats.failed_reads = 0

        stalled_l, stalled_r = source.stalled_sides()
        drop, drop_reason = should_drop(skew, stalled_l, stalled_r, max_skew_ms)
        if t_base is None:
            t_base = min(t_l, t_r)
        t_l_rel = t_l - t_base
        t_r_rel = t_r - t_base
        # 三角化取**较晚**那个时刻（planbook B-193 的口径）。代价要说清：补偿后的
        # 点是**较早时刻**的位置，标在较晚时刻等于给它加了 <=1 帧（<=60 ms）的
        # **标签**滞后；三维值本身没变（物理上它就是 t_early 的位置）。
        # 换来的是时间轴单调、不随 skew 的正负跳来跳去。
        t_pair = max(t_l_rel, t_r_rel)

        t_a = time.perf_counter()
        res_l = pipe_l.process(left, t_l_rel)
        proc_l = (time.perf_counter() - t_a) * 1000.0
        t_a = time.perf_counter()
        res_r = pipe_r.process(right, t_r_rel)
        proc_r = (time.perf_counter() - t_a) * 1000.0

        det_l, det_r = _as_detection(res_l), _as_detection(res_r)
        det_l2, det_r2, fix_note = compensate_detections(
            det_l, res_l.track, det_r, res_r.track, skew,
            enable=skew_fix, min_hits=min_hits)

        pt = None
        if rec is not None and not drop:
            # 丢掉的那一对**不喂给重建器**：它连帧都没有可信的三维意义，
            # 喂进去只会让配对率的分母把"本来就不该算"的帧也算进去。
            pt = rec.update(t_pair, det_l2,
                            [det_r2] if det_r2 is not None else [])
            stats.match_rate = rec.match_rate
            if pt is not None:
                stats.points += 1
                if collect_rows:
                    # 落盘用的那一行：列序 = `CSV_HEADER`（唯一产地，别抄）。
                    # `frame` 取**这对帧的序号**（不是点的序号）—— 查看器按它 seek
                    # 视频，而视频是一对一帧，两者必须同一个数。
                    stats.session_rows.append([
                        stats.frame_index, f"{pt.t:.4f}", 1,
                        f"{pt.xyz[0]:.6f}", f"{pt.xyz[1]:.6f}", f"{pt.xyz[2]:.6f}",
                        f"{pt.u_left[0]:.3f}", f"{pt.u_left[1]:.3f}",
                        f"{pt.u_right[0]:.3f}", f"{pt.u_right[1]:.3f}",
                        f"{getattr(pt, 'epipolar_px', float('nan')):.3f}",
                        f"{pt.residual_px:.3f}",
                        1 if pt.residual_px <= 2.0 else 0,   # 口径同 StereoPoint.confident
                        "", ""])
                    if stats.seed is None and det_l2 is not None and det_r2 is not None:
                        stats.seed = (float(det_l2.x), float(det_l2.y),
                                      float(det_l2.radius),
                                      float(det_r2.x), float(det_r2.y),
                                      float(det_r2.radius), stats.frame_index)
                stats.note(stats.residuals, pt.residual_px)
                # 极线距离两种模式都有（一个在校正图上、一个在原始像素上），
                # 名字统一叫 epi；它是"配对有多勉强"的度量。
                epi = getattr(pt, "epipolar_px", None)
                if epi is not None and math.isfinite(float(epi)):
                    stats.note(stats.epipolars, float(epi))
                # **视差只对 parallel 有意义**：大角度下原始像素的 x 差不与深度
                # 成反比（两光轴差 88°），把这个数叫"视差"是错的语义 ——
                # HUD 上它写 n/a，另用"交会角"表达同一件事（深度有多良态）。
                if not angled:
                    stats.note(stats.disparities,
                               abs(pt.u_left[0] - pt.u_right[0]))
                else:
                    ang = getattr(pt, "angle_deg", None)
                    if ang is not None and math.isfinite(float(ang)):
                        stats.note(stats.angles, float(ang))
        if drop:
            stats.dropped += 1
            stats.drop_reasons[drop_reason] += 1

        _feed_health(health_l, t_l, seq_l, detect_ms=res_l.detect_ms,
                     process_ms=proc_l, prev_t=prev_t[0], prev_seq=prev_seq[0])
        _feed_health(health_r, t_r, seq_r, detect_ms=res_r.detect_ms,
                     process_ms=proc_r, prev_t=prev_t[1], prev_seq=prev_seq[1])
        prev_t[0], prev_t[1] = t_l, t_r
        prev_seq[0], prev_seq[1] = seq_l, seq_r
        for h in (health_l, health_r):
            h.skew_ms = float(skew) if skew is not None else 0.0
            h.skew_ms_valid = skew is not None
        stats.pairs += 1
        stats.note(stats.process_ms, proc_l + proc_r)
        if skew is not None:
            stats.note(stats.skews, skew)

        # 几何判读每 15 对算一次：`geometry_verdict` 要对全部样本取中位/分位，
        # 每帧都算等于把 O(n) 摊进主循环（跑十分钟后 n 是两万）。
        # 只有 parallel 有 dy 样本（"校正后 y 差"在未校正像素上没有几何意义），
        # 所以大角度不做这一步 —— 它的等价诊断在 `low_match_diagnostic()` 里。
        if not angled and rec is not None and stats.pairs % 15 == 0:
            v = geometry_verdict(getattr(rec, "dy_samples", [])[-GEOM_TAIL:],
                                 getattr(rec.rig, "quality", None))
            if v is not None:
                stats.verdict = v
                stats.verdict_suspect = bool(v.suspect)

        stats.last_left, stats.last_right = left, right
        if not display_width:
            half = None
            pscale = 1.0
        else:
            dw = int(display_width)
            # 三维小窗现在**独占一列**，所以要先把它从总宽里扣掉再对半给两路。
            # 不扣的话窗口总宽会比用户设的 `--display-width` 多出一列，窄屏上直接出屏
            # （2026-11-17：小窗原来叠在画面上，宽度"免费"；挪出去以后就不免费了）。
            _ps = 1.0 if max(2, dw // 2) >= 700 else 0.65
            half = max(2, (dw - int(320 * _ps)) // 2)
            pscale = 1.0 if half >= 700 else 0.65
        vis_l = pipe_l.render(left, res_l, health=health_l, display_width=half)
        vis_r = pipe_r.render(right, res_r, health=health_r, display_width=half)
        # 会话视频：**一对帧写一帧**，写的是纯并排（不带顶栏/小窗）——
        # 查看器按 2:1 从中间切成左右两路，带顶栏或右侧小窗的合成画布切出来是歪的。
        # 序号与 `session_rows` 的 `frame` 列共用 `stats.frame_index`。
        if frame_sink is not None:
            frame_sink(vis_l, vis_r)
        # 小窗的尺寸/字号用**固定档**，不跟显示缩放走：`draw_3d_panel` 的 `scale`
        # 同时管画布与字号，跟到 0.4 倍时里面的字只有 6 px 高，等于没有。
        # `pscale` 已经在上面按每路宽度定过一次，这里不再重算（重算会与预留的
        # 那一列对不上，总宽就又不守恒了）。
        panel = None
        want_panel = panel_on() if callable(panel_on) else panel_on
        if want_panel and rec is not None and rec.trajectory:
            hist = np.array([p.xyz for p in rec.trajectory[-PANEL_POINTS:]],
                            dtype=np.float64)
            panel = draw_3d_panel(
                size=(int(320 * pscale), int(240 * pscale)), history=hist,
                title=("3D angled (raw pixels)" if angled
                       else "3D stereo (triangulated)"), scale=pscale)
        # 大角度没有"视差"，但有两个自己的量：极线距离与交会角。
        # 它们**另起名字** (epi / ang)，不塞进 disp —— 那是不同的几何量，
        # 共用一个名字会让人以为"视差 42 px" 在大角度下也有同样含义。
        notes = [fix_note]
        if angled:
            if stats.epipolars:
                notes.append("epi %.1f px" % statistics.median(stats.epipolars))
            if stats.angles:
                notes.append("ang %.0f°" % statistics.median(stats.angles))
        hud = compose_hud(
            match_rate=stats.match_rate,
            residual_px=(statistics.median(stats.residuals)
                         if stats.residuals else None),
            disparity_px=(None if angled else
                          (statistics.median(stats.disparities)
                           if stats.disparities else None)),
            skew_ms=skew, dropped=stats.dropped,
            # `pts` 是**已攒的三维点数**：落盘门槛 30，到了才存会话。
            # 不显示它，用户只能在退出后才发现"什么都没留下"（2026-11-17）。
            note="  ".join([*notes, f"pts {stats.points}"]))
        # 顶栏：指标靠左、按键提示右对齐（**同一行**，谁也别顶掉谁）。
        # 瞬时回显走左上角那块角标 —— 它不占顶栏，录制时也不会把提示挤没。
        # 两处都只是"体验"，坏了不许拖垮主循环。
        _hint, _badge = REALTIME_HINT, ""
        if hint_fn is not None:
            try:
                _hint = hint_fn() or REALTIME_HINT
            except BaseException:                        # noqa: BLE001
                pass
        if badge_fn is not None:
            try:
                _badge = badge_fn() or ""
            except BaseException:                        # noqa: BLE001
                pass
        stats.last_vis = compose_canvas(vis_l, vis_r, hud, panel=panel,
                                        alert=drop, hint=_hint, badge=_badge)

        if log_every and time.perf_counter() >= next_report:
            next_report = time.perf_counter() + float(log_every)
            _log.info("健康度 L %s | R %s | %s", health_l.summary_line(),
                      health_r.summary_line(), hud)

        # 帧序号在**每对**处理完（含被丢弃、含没出点）都 +1：视频一帧对一帧，
        # CSV 的 `frame` 就是它。放到 `if pt is not None` 里面就会变成"按点计数"，
        # 一旦有帧没配对上，视频与轨迹就整体错位（查看器 seek 到错的画面）。
        stats.frame_index += 1
        if on_frame is not None and on_frame(stats.last_vis, stats) is False:
            stats.stop_reason = "用户退出"
            break
    return stats


# ---------------------------------------------------------------------------
# 配置 / 参数
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="双目实时球体检测：两路相机 -> 三角测量 -> 三维轨迹",
        epilog="检测方法只有 color / motion 能跟上实时"
               "（hough 4.4 s/帧、fusion 3.7 s/帧，B-16 已定性）。")
    # 取帧参数与标定/复检/录制**同一套**（`_common.add_stereo_source_args`）：
    # 字段名必须与 `stereo.from_args()` 读的逐字对应，否则 `open_pair_source` 会忽略。
    add_stereo_source_args(p)
    p.add_argument("cams", nargs="*", metavar="IDX",
                   help="也可以直接给两个编号：run_realtime_stereo.py 0 2")
    p.add_argument("--config", default=None, help="读取 JSON 配置（检测/预测参数）")
    p.add_argument("--method", default=None, choices=REALTIME_METHODS,
                   help="检测方法：只支持 color / motion（其余达不到实时）")
    p.add_argument("--preset", default=None, choices=ColorConfig.preset_choices(),
                   help="颜色预设；不传则用配置里的颜色")
    p.add_argument("--predict", default=None,
                   choices=method_names("predictor"), help="预测方法")
    p.add_argument("--horizon", type=float, default=None, help="预测未来多少秒")
    p.add_argument("--gravity-fit", action="store_true",
                   help="三维预测时使用已知重力约束（需要球径才有三维历史）")
    p.add_argument("--calib", default=None,
                   help="双目标定基名（.npz/.json 可不写后缀）；"
                        "不给就按**当前装置档案**解析")
    p.add_argument("--detect-width", type=int, default=DEFAULT_DETECT_WIDTH,
                   help=f"检测跑在多宽的图上（默认 {DEFAULT_DETECT_WIDTH}；0=原图）。"
                        "全分辨率 color 17.8ms/路，两路 35.6ms 达不到 >=20fps")
    p.add_argument("--display-width", type=int, default=DEFAULT_DISPLAY_WIDTH,
                   help=f"**并排整幅**的显示宽度（默认 {DEFAULT_DISPLAY_WIDTH}，"
                        "对半给每路；0=不缩放）")
    p.add_argument("--max-skew-ms", type=float, default=None,
                   help=f"两路时间差门限（默认按 --fps 派生：30 fps 及以下为 "
                        f"{DEFAULT_MAX_SKEW_MS:g} ms，更高帧率按 60*30/fps 收紧、"
                        "不低于 2 ms）；超过就丢这一对并计数")
    # default=None：未显式给就按 --fps 派生（见 default_max_skew_ms）
    p.add_argument("--no-skew-fix", action="store_true",
                   help="关闭 skew 一阶补偿（用来做 A/B 对比与排障）")
    p.add_argument("--no-3d", action="store_true",
                   help="只显示检测/跟踪，不做三角测量")
    p.add_argument("--record", default=None, metavar="PATH",
                   help="按 v 时把并排标注画面录到这个 mp4（默认 outputs/videos/）")
    p.add_argument("--save-session", dest="save_session", action="store_true",
                   default=True,
                   help="退出时把这次运行落成一个会话（默认开）："
                        "data/stereo/<模式>/<时间戳>/，含 session.json / "
                        "derived/{traj3d.csv,seeds.json,traj3d.json,"
                        "visualization.json,annotated.mp4}。"
                        f"三维点 < {MIN_SESSION_POINTS} 时不落盘")
    p.add_argument("--no-save-session", dest="save_session", action="store_false",
                   help="不落会话（只是看一眼，不给数据管理留条目）")
    p.add_argument("--max-frames", type=int, default=0,
                   help="处理这么多**对**就退出（0=一直跑）。用于无窗冒烟")
    p.add_argument("--log-every", type=float, default=5.0,
                   help="每隔几秒往日志写一行健康度（0=不写）")
    args = p.parse_args(argv)

    # --max-skew-ms 的 None 必须**在这里**填成实际数字：`should_drop()` 里
    # `max_skew_ms=None` 的语义是"不设门限"，绝不会是想表达"用默认值"。
    # 让它漏进 run_loop，高帧率下就会一对都不丢 —— 比门限偏松危险得多。
    if args.max_skew_ms is None:
        args.max_skew_ms = default_max_skew_ms(args.fps)

    if args.cams:
        if len(args.cams) != 2:
            p.error("位置参数只接受**两个**相机编号（左 右）；"
                    "要按角色名请用 --left/--right")
        if args.left is None:
            args.left = args.cams[0]
        if args.right is None:
            args.right = args.cams[1]

    # ---- 一个相机参数都没给 -> 按"成对自动挑"跑 -------------------------
    # 为什么必须有这一条（2026-11-16）：GUI 的「实时 3D（不落盘）」按钮调
    # `_actions.stereo_realtime_args()`，**全部用默认值**（不带 --left/--right）。
    # 没有这个默认，`stereo.from_args()` 会走到"请指定 --left/--right 或
    # --left-video ..."并让进程退出 1 —— 用户点一次按钮只看到一句参数错误，
    # 那就又是一个"假入口"。自动挑两台是可解释的默认（与 --auto-cameras 同一条
    # 代码路径，挑完会打印 `[自动选择] 左 #x｜右 #y`），不是静默兜底。
    if not any((args.left, args.right, args.auto_cameras,
                args.left_video, args.right_video,
                args.left_dir, args.right_dir)):
        args.auto_cameras = True
    return args


def build_config(args: argparse.Namespace) -> AppConfig:
    cfg = load_config(args.config) if args.config else AppConfig()
    if args.method:
        cfg.detector.method = args.method
    if args.preset:
        cfg.detector.color = ColorConfig.preset(args.preset)
    if args.predict:
        cfg.predict.method = args.predict
    if args.horizon:
        cfg.predict.horizon_s = args.horizon
    if args.gravity_fit:
        cfg.predict.use_gravity = True
    if args.no_3d:
        cfg.camera.enable_3d = False
    # 三维小窗由本入口自己画（数据是 `StereoPoint.xyz`，不是单目半径反推的深度）。
    # 关掉 pipeline 自己那块，否则并排图里会出现**两个**三维面板、且数据来源不同。
    cfg.viz.show_axes_3d = False
    return cfg


class ReconstructionModeError(RuntimeError):
    """这套标定不能按它**标称的模式**做三维重建（要么重标，要么改模式）。

    为什么用独立异常类型而不是返回 ``None``：调用方（`main`）必须能把它与
    "标定文件读不出来"分开处理 —— 前者的修法是"重标 / 换模式"，后者是
    "把文件找回来"。两者都退出 2，但对用户说的话完全不同。
    """


def rig_stereo_mode(rig) -> str:
    """这套标定**标称**是哪种双目模式（``"parallel"`` / ``"angled"``）。

    权威来源是标定文件自己：`StereoRig.load` 把 JSON 顶层原样放进 `meta`，
    而 `StereoRig.stereo_mode` / `rectification` 都从那里读。这里**不**根据
    光轴夹角去猜 —— 夹角 88° 但文件写着 parallel 是"文件错"，该报出来让人
    改文件，不该由程序替它翻译成另一个模式（猜错就静默换了一套几何）。

    没有 `stereo_mode` 属性/字段的对象（判据里的假 rig）按 `meta` 找，
    再退回 ``"parallel"`` —— 与 `calib.params.StereoRig.stereo_mode` 同一默认。
    """
    raw = getattr(rig, "stereo_mode", None)
    if raw:
        try:
            return normalize_mode(raw)
        except ValueError:                      # 标成 "auto"：没有上下文可解析
            pass
    meta = getattr(rig, "meta", None) or {}
    try:
        return normalize_mode(meta.get("stereo_mode"))
    except ValueError:
        return "parallel"


def make_reconstructor(rig, *, max_epipolar_px: float | None = None,
                       allow_zero_match: bool = True):
    """按标定选重建器 —— **全入口唯一分发点**。

    三种组合（判据 1 逐个钉住）：

    =========================  ===================  ==========================
    标称模式 / 校正状态          返回                 说明
    =========================  ===================  ==========================
    ``angled``（含 none_raw）    ``AngledReconstructor``  原始像素 + 基础矩阵
    ``parallel`` + 已校正        ``StereoReconstructor``  校正后像素 + y 差门限
    ``parallel`` + ``none_raw``  **抛** `ReconstructionModeError`  历史错标定
    =========================  ===================  ==========================

    为什么 `parallel + none_raw` 必须拒绝（不是"自动改走 angled"）：离线入口
    `run_stereo_video._mode_guard` 会这么改判，那是**为了救历史数据**（chain
    产物曾把 `stereo_mode` 写成 parallel）。实时这条链没有会话可写、也没有
    "本次用了哪个模式"的落盘记录，替用户改判等于把一个错标定**静默**当成
    可用 —— 而 `triangulate.py:564-575` 的护栏说明过后果：dy 门限与视差范围
    建立在校正图上，用错几何不会报错，只会配对率归零。
    所以这里报出来，让人去重标或把标定的 `stereo_mode` 改成 angled。

    `max_epipolar_px=None` 时按模式取默认：parallel 用标定自报极线 RMS 推
    （`epipolar_gate_px`，4×RMS、限 1~6 px），angled 用 8 px（大角度下候选
    在原始像素上的偏差本来就更大，`AngledMatchConfig` 的默认值）。
    """
    mode = rig_stereo_mode(rig)
    rect = str(getattr(rig, "rectification", "") or "")
    uncorrected = is_uncorrected(rect)

    # ⚠ 反向自证锚点：整段删掉（到下面 `if uncorrected:` 之前），angled 标定就
    # 拿不到大角度重建器 —— `tests/selftest_stereo_branch.py` 第 ⑩/⑫ 条必须变红。
    if mode == "angled":
        gate = (ANGLED_MAX_EPIPOLAR_PX if max_epipolar_px is None
                else float(max_epipolar_px))
        return AngledReconstructor(rig, AngledMatchConfig(max_epipolar_px=gate))

    if uncorrected:
        raise ReconstructionModeError(
            f"标定写的是 parallel，但这套参数**没做极线校正**"
            f"（rectification={rect!r}，链式/大角度装置的产物）。\n"
            "  parallel 重建的 y 差门限与视差范围建立在校正图上，拿未校正的"
            "原始像素去配，结果是**全部候选被拒、配对率 0，而且不报错**。\n"
            "  怎么办（两条，选一条）：\n"
            "    1) 这份标定是**大角度/链式**的：把它的 stereo_mode 改成 angled"
            "（或重新按大角度标一次），实时双目会走大角度重建；\n"
            "    2) 重标一次平行双目（apps/calibrate_stereo.py），产出"
            " stereo_rectify 的标定。")

    gate = (epipolar_gate_px(getattr(rig, "quality", None))
            if max_epipolar_px is None else float(max_epipolar_px))
    return StereoReconstructor(rig, MatchConfig(max_epipolar_px=gate),
                               allow_zero_match=allow_zero_match)


def make_pipeline(cfg: AppConfig, rig: StereoRig, side: str) -> SpherePipeline:
    """装配一路 pipeline（内参取**该路**的 K，不用 hfov 估算）。

    为什么单目这块也要动：配置里若填了球径，`SpherePipeline._ensure_camera` 会用
    hfov 估一个 fx 去算单目深度，HUD 上就会出现一排"看起来是三维、其实是估算"的
    数字，与旁边真正的三角测量结果并列 —— 用户没法分辨哪个能信。
    所以有球径时直接把该路的标定内参塞进去（口径与
    `apps/run_stereo_video.py:238-253 _make_pipeline` 一致）。
    """
    pipe = SpherePipeline(cfg)
    K = rig.K1 if side == "left" else rig.K2
    if cfg.camera.ball_diameter_m:
        pipe.camera = CameraModel(
            width=int(rig.image_size[0]), height=int(rig.image_size[1]),
            fx=float(K[0, 0]), fy=float(K[1, 1]),
            cx=float(K[0, 2]), cy=float(K[1, 2]),
            ball_diameter_m=float(cfg.camera.ball_diameter_m),
        )
    return pipe


def resolve_rig(arg: str | None) -> tuple[StereoRig | None, str]:
    """解析双目标定，返回 ``(rig, 失败说明)``；失败说明非空时 rig 为 None。

    路径解析**一律委托** `_common.current_calibration` / `stereo.profiles`，
    不在这里拼 `ROOT / "config" / "stereo_calibration"` —— 那正是 2026-11-10
    统一掉的那套"GUI 按档案、CLI 按顶层"的双口径。
    """
    if arg:
        base = Path(arg)
        # 基名（无后缀）也算找得到：`StereoRig.load` 自己会补 .npz/.json。
        # 用字符串拼接而不是 `with_suffix()` —— 后者会把名字里的点当扩展名替换掉
        # （"rig.v2" -> "rig.json"），那是个很难查的路径 bug。
        found = any(Path(f"{base}{suf}").exists() for suf in ("", ".npz", ".json"))
        problem = "" if found else f"找不到标定文件：{base}（.npz / .json 都不在）"
    else:
        base, problem = current_calibration(need_existing=True)
    if problem:
        return None, problem
    try:
        rig = StereoRig.load(base)
    except Exception as exc:                              # noqa: BLE001
        return None, f"标定读不出来：{base}（{exc}）"
    # ⚠ **不在这里判模式**：这套标定是 parallel 还是 angled、能不能用，
    # 由 `make_reconstructor()` 一处决定（含 none_raw + parallel 的拒绝）。
    # 两处各判一次正是本项目反复踩的"同一件事两个判据互相打架"。
    return rig, ""


class _Ui:
    """OpenCV 窗口交互：显示 / 按键 / 录制 / 快照。**不碰相机、不碰检测**。"""

    def __init__(self, args: argparse.Namespace, cfg: AppConfig,
                 pipe_l: SpherePipeline, pipe_r: SpherePipeline,
                 rec, wins: WindowManager,
                 fps: float) -> None:
        self.args, self.cfg = args, cfg
        self.pipe_l, self.pipe_r, self.rec = pipe_l, pipe_r, rec
        self.wins = wins
        self.fps = float(fps) if fps and fps > 1.0 else 30.0
        self.paused = False
        self.panel_on = not args.no_3d
        self.record_wanted = False
        self.writer: cv2.VideoWriter | None = None
        self.record_path: Path | None = None
        #: 录制模式：""（没录）/ "canvas"（只检测画面）/ "both"（+原始两路）。
        self.record_mode = ""
        #: 原始两路的 writer（只在 "both" 模式里非空）。
        self.raw_writers: list[cv2.VideoWriter] = []
        self.raw_paths: tuple[Path, Path] = ()
        # 按键回显（2026-11-17 用户："按下了没有提示很可恶"）。
        self._toast = ""
        self._toast_until = 0.0
        self._rec_frames = 0
        #: 会话记录器（`main` 建好后挂上来），按 v 时告诉它"这段要留住"。
        self.recorder = None

    def toast(self, text: str, ttl: float = 2.2, log: bool = True) -> None:
        """在顶栏第二行亮一条短提示，`ttl` 秒后自动退回按键说明。

        为什么不能只 `print`：用户盯的是 OpenCV 窗口，控制台在后面（甚至被
        `pythonw` 藏起来）。按 v 录制、s 快照、c 换方法之后**画面上什么都没变**，
        唯一合理的结论就是"按键坏了"。
        """
        self._toast = str(text)
        self._toast_until = time.perf_counter() + float(ttl)
        if log:                      # 录制帧数每 15 帧刷一次，别把日志淹了
            _log.info("提示：%s", text)

    def hint_line(self) -> str:
        """顶栏右侧那排按键提示 —— **常驻**，不会被回显顶掉。"""
        return REALTIME_HINT

    def badge_line(self) -> str:
        """左上角角标要显示的瞬时回显；没有就返回空串。"""
        if self._toast and time.perf_counter() < self._toast_until:
            return "▸ " + self._toast
        return ""

    # -- 显示 ----------------------------------------------------------
    def show(self, canvas: np.ndarray | None, stats: LoopStats) -> bool:
        """显示一帧并处理按键。返回 ``False`` = 退出程序。"""
        if canvas is None:
            return True
        if self.paused:
            return self._pause_loop(canvas, stats)
        self.wins.show(WINDOW, canvas)
        key = cv2.waitKey(1) & 0xFF
        if self.wins.handle_key(WINDOW, key, canvas):
            return True
        return self._handle_key(key, canvas, stats)

    def _pause_loop(self, canvas: np.ndarray, stats: LoopStats) -> bool:
        """暂停时**阻塞在这里**：不处理帧，相机照常采集，恢复后直接取最新帧。

        为什么不在主循环里 `continue`：那样仍会一对一对地取帧（`read_pair` 会等
        两路各出新帧），暂停只是"不显示"，CPU 照烧、`cap` 照掉 ——
        用户看到的"暂停"应该是画面真的停住。
        """
        while self.paused:
            self.wins.show(WINDOW, canvas)
            key = cv2.waitKey(30) & 0xFF
            if self.wins.handle_key(WINDOW, key, canvas):
                continue
            if not self._handle_key(key, canvas, stats):
                return False
        return True

    def _handle_key(self, key: int, canvas: np.ndarray, stats: LoopStats) -> bool:
        if key in (ord("q"), 27):
            _log.info("用户按 q 退出")
            return False
        if key == ord(" "):
            self.paused = not self.paused
            print("已暂停" if self.paused else "继续")
            _log.info("用户%s", "暂停" if self.paused else "继续")
            self.toast("已暂停（画面停住，相机仍在采集）" if self.paused
                       else "继续")
        elif key == ord("r"):
            self.pipe_l.reset()
            self.pipe_r.reset()
            if self.rec is not None:
                self.rec.trajectory.clear()
                # `dy_samples` 只有 parallel 重建器有（大角度没有"校正后 y 差"）。
                samples = getattr(self.rec, "dy_samples", None)
                if samples is not None:
                    samples.clear()
            print("两路轨迹与三维点已重置")
            _log.info("用户重置两路轨迹")
            self.toast("轨迹与三维点已重置")
        elif key == ord("3"):
            self.panel_on = not self.panel_on
            print(f"三维小窗：{'开' if self.panel_on else '关'}")
            _log.info("用户切换三维小窗：%s", self.panel_on)
            self.toast(f"三维小窗 {'开' if self.panel_on else '关'}")
        elif key == ord("c"):
            self._cycle_method()
        elif key == ord("s"):
            self._snapshot(canvas, stats)
        elif key == ord("v"):
            self._toggle_record("canvas")
        elif key == ord("b"):
            # 同时录检测画面 + 两路原始画面（2026-11-17 用户要求）。
            # 不能并到 v 里轮档：轮档没人记得住当前是哪档。
            self._toggle_record("both")
        return True

    def _cycle_method(self) -> None:
        """在允许实时的方法之间轮换，**两个实例一起换**。

        只换一个的话，两路会用不同检测器：颜色门限/掩膜都不一样，
        左右圆心系统性偏移（镜面球实测差 40%），三角测量直接把偏差算成深度。
        """
        cur = self.cfg.detector.method
        i = METHOD_CYCLE.index(cur) if cur in METHOD_CYCLE else 0
        self.cfg.detector.method = METHOD_CYCLE[(i + 1) % len(METHOD_CYCLE)]
        self.pipe_l.detector = build_detector(self.cfg.detector)
        self.pipe_r.detector = build_detector(self.cfg.detector)
        print(f"检测方法切换为：{self.cfg.detector.method}（两路同时）")
        self.toast(f"检测方法 → {self.cfg.detector.method}（两路同时）")
        _log.info("用户切换检测方法为 %s（两路同时）", self.cfg.detector.method)

    # -- 录制 ----------------------------------------------------------
    def _raw_paths(self, base: Path) -> tuple[Path, Path]:
        """原始两路视频的落点：跟着检测视频同名前缀，后缀标 left/right。"""
        return (base.with_name(base.stem + "_raw_left.mp4"),
                base.with_name(base.stem + "_raw_right.mp4"))

    def _toggle_record(self, mode: str = "canvas") -> None:
        """按一次开录、再按一次停。``mode``：``"canvas"``=只录检测画面，``"both"``=同时录原始两路。

        为什么要两个键而不是一个（2026-11-17 用户）：只录检测画面是常态（体积小、
        编码便宜）；但要做**证据留存**或拿原始画面二次分析，就得有未经绘制的两路。
        一个键轮三档会让人记不住当前是哪档，所以拆成 `v` 与 `b`。
        """
        if self.record_wanted:
            self.record_wanted = False
            self.record_mode = ""
            self._close_writer()
            n = self._rec_frames
            print(f"录制已停止（{n} 帧）")
            _log.info("录制已停止，%d 帧", n)
            self.toast(f"录制已停止（{n} 帧）")
            return
        base = Path(self.args.record) if self.args.record else (
            ROOT / "outputs" / "videos"
            / f"stereo_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
        self.record_path = base
        self.record_mode = "both" if mode == "both" else "canvas"
        self.raw_paths = self._raw_paths(base) if self.record_mode == "both" else ()
        self._rec_frames = 0
        self.record_wanted = True
        if self.recorder is not None:
            self.recorder.keep_anyway = True
        # **writer 尺寸必须取实际写入帧的 shape**，所以这里不能先开：
        # `run_realtime.py:403-419` 那个老坑是"writer 用 960 小图尺寸，写入全分辨率图"，
        # `VideoWriter.write` 对尺寸不符只丢弃 + 警告，默认参数下录出来是**空文件**。
        # 现在改为"下一帧真拿到画面时按它的 shape 开"，从结构上不可能对不上。
        if self.record_mode == "both":
            print(f"开始录制（检测 + 原始两路）：{self.record_path}；"
                  f"{self.raw_paths[0].name} / {self.raw_paths[1].name}")
            _log.info("开始录制（both）：%s + %s", self.record_path, self.raw_paths)
            self.toast("● 录制中：检测 + 原始两路（多两路 1080p 编码，帧率会掉）",
                       ttl=3.0)
        else:
            print(f"开始录制：{self.record_path}（尺寸取自实际画面）")
            _log.info("开始录制：%s", self.record_path)
            self.toast(f"● 录制画面 → {self.record_path.name}（不是会话）")

    def _close_writer(self) -> None:
        for w in (self.writer, *self.raw_writers):
            if w is not None:
                w.release()
        self.writer = None
        self.raw_writers = []

    def _open_raw(self, side: int, frame: np.ndarray) -> cv2.VideoWriter | None:
        """懒开原始那一路的 writer —— 尺寸一律取**这一路真实帧**的 shape。"""
        path = self.raw_paths[side]
        path.parent.mkdir(parents=True, exist_ok=True)
        h, w = frame.shape[:2]
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"),
                             self.fps, (w, h))
        if not vw.isOpened():
            print(f"[警告] 原始视频打不开编码器（{path.name}）—— 这一路不录了")
            _log.warning("原始录制失败：%s", path)
            return None
        _log.info("原始视频 %s（%dx%d @ %.1f fps）", path, w, h, self.fps)
        return vw

    def write_frame(self, canvas: np.ndarray, stats: LoopStats | None = None) -> None:
        """写一帧录制。``both`` 模式下额外写两路**原始**画面（未绘制）。"""
        if not self.record_wanted or canvas is None:
            return
        if self.writer is None:
            if self.record_path is None:
                return
            self.record_path.parent.mkdir(parents=True, exist_ok=True)
            h, w = canvas.shape[:2]
            self.writer = cv2.VideoWriter(
                str(self.record_path), cv2.VideoWriter.fourcc(*"mp4v"),
                self.fps, (w, h))
            if not self.writer.isOpened():
                print("[警告] 录制打不开编码器（mp4v）—— 已停止录制")
                _log.warning("录制失败：编码器不可用")
                self.writer = None
                self.record_wanted = False
                return
            _log.info("录制画布 %dx%d @ %.1f fps", w, h, self.fps)
        self.writer.write(canvas)

        if self.record_mode == "both" and stats is not None:
            raws = (getattr(stats, "last_left", None),
                    getattr(stats, "last_right", None))
            for side, frame in enumerate(raws):
                if frame is None:
                    continue
                if side >= len(self.raw_writers):
                    self.raw_writers.append(self._open_raw(side, frame))
                vw = self.raw_writers[side]
                if vw is not None:
                    vw.write(frame)

        self._rec_frames += 1
        if self._rec_frames % 15 == 1:
            tag = "检测+原始" if self.record_mode == "both" else "画面"
            self.toast(f"● 录制{tag} {self._rec_frames} 帧", ttl=1.2, log=False)

    # -- 快照 ----------------------------------------------------------
    def _snapshot(self, canvas: np.ndarray, stats: LoopStats) -> None:
        """存快照：并排标注图 + 两路原图。**只在按键时写盘**（outputs 其余不动）。"""
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = ROOT / "outputs" / "images"
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"snapshot_{stamp}.png"), canvas)
        if stats.last_left is not None:
            cv2.imwrite(str(out_dir / f"raw_{stamp}_left.png"), stats.last_left)
        if stats.last_right is not None:
            cv2.imwrite(str(out_dir / f"raw_{stamp}_right.png"), stats.last_right)
        print(f"已保存快照：outputs/images/snapshot_{stamp}.png"
              f"（+ raw_{stamp}_left/right.png）")
        _log.info("用户保存快照 %s", stamp)
        self.toast(f"已存快照 snapshot_{stamp}.png（+ 两路原图）")

    def close(self) -> None:
        self._close_writer()


def _display_path(p: str | Path) -> str:
    """给人看的路径：在仓库里就写相对路径，不在就原样（判据用临时目录）。"""
    try:
        return Path(p).relative_to(ROOT).as_posix()
    except ValueError:
        return Path(p).as_posix()


# ---------------------------------------------------------------------------
# 会话落盘：退出时把这次实时运行变成一个"数据管理里看得见、3D 查看器打得开"的会话
# ---------------------------------------------------------------------------
class _SessionRecorder:
    """一次实时运行 -> `data/stereo/<mode>/<YYYYMMDD-HHMMSS>/`。

    为什么边跑边写视频，而不是退出时再"补一段"：实时**没有原始录像**，
    只有当下这一刻的画面；退出后再想写就没有帧了。所以 `write_frame()`
    每一对帧调一次，帧序号与 `traj3d.csv` 的 `frame` 列**同一个数**
    （查看器按它 seek 视频，错一位就"点开是别的画面"）。

    为什么要 `MIN_SESSION_POINTS` 门槛：实时窗口可能只是"打开看一眼"。
    每次都留一条空会话，数据管理那个列表几天就没法用了。低于门槛连目录都不留
    （边跑边写的目录会被删掉 —— 那是**本次自己建的**，不是用户数据）。
    """

    def __init__(self, *, root: str | Path, mode: str,
                 image_size: tuple[int, int], fps: float,
                 calib_base: str = "", rig=None, stamp: str | None = None) -> None:
        self.root = Path(root)
        self.mode = str(mode)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.fps = float(fps) if fps and fps > 1.0 else 30.0
        self.calib_base = str(calib_base or "")
        self.rig = rig
        now = time.localtime()
        self.stamp = stamp or time.strftime("%Y%m%d-%H%M%S", now)
        # 跑一次可能几分钟：`created` / `label` 记**开始**那一刻，不是落盘那一刻。
        self.created = time.strftime("%Y-%m-%d %H:%M:%S", now)
        self.label = time.strftime("实时 %H:%M:%S", now)
        self.dir: Path | None = None
        self.writer: cv2.VideoWriter | None = None
        self.video_frames = 0
        self.video_failed = False
        #: 用户按过 `v`（手动录画面）。他既然明确要留东西，就不该因为
        #: "三维点不够 30" 把整段会话连同录像一起删掉（2026-11-17 用户：
        #: "我刚刚录制了，数据管理里根本没有"）。
        self.keep_anyway = False

    # -- 目录 / 视频 ----------------------------------------------------
    def _ensure_dir(self) -> Path:
        """第一次真的要用时才建目录（`--no-3d` / 点数不够时不留痕迹）。"""
        if self.dir is not None:
            return self.dir
        base = self.root / self.stamp
        d, n = base, 1
        while d.exists():                       # 同一秒跑两次：加 -01，与 recorder 同口径
            d = self.root / f"{self.stamp}-{n:02d}"
            n += 1
        (d / "derived").mkdir(parents=True, exist_ok=True)
        self.dir = d
        return d

    def write_frame(self, vis_l: np.ndarray, vis_r: np.ndarray) -> None:
        """写一帧**纯并排**（不带顶栏、不带三维小窗）。

        查看器要求左右并排并从中间切（`view_3d.py` 的 2:1 校验），
        `compose_canvas` 那种带顶栏+右侧小窗的画布切出来是歪的，所以这里
        只用两路 `pipe.render` 出来的图。
        """
        if self.video_failed:
            return
        d = self._ensure_dir()
        if vis_l.shape[:2] != vis_r.shape[:2]:
            vis_r = cv2.resize(vis_r, (vis_l.shape[1], vis_l.shape[0]),
                               interpolation=cv2.INTER_AREA)
        body = np.hstack([vis_l, vis_r])
        if self.writer is None:
            h, w = body.shape[:2]
            path = d / "derived" / "annotated.mp4"
            self.writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter.fourcc(*"mp4v"), self.fps, (w, h))
            if not self.writer.isOpened():
                # 编码器不可用不该让整次运行白跑：会话照落，只是没有视频。
                print("[会话][警告] 打不开 annotated.mp4 编码器（mp4v）——"
                      "本次只存 session.json 与轨迹，不存视频")
                self.writer = None
                self.video_failed = True
                return
            _log.info("会话视频 %s（%dx%d @ %.1f fps）", path, w, h, self.fps)
        self.writer.write(body)
        self.video_frames += 1

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()          # 不 release 就是坏 mp4（moov 没落盘）
            self.writer = None

    # -- 落盘 ------------------------------------------------------------
    def save(self, stats: LoopStats, *,
             min_points: int = MIN_SESSION_POINTS) -> Path | None:
        """写 session.json / traj3d.csv / seeds.json /（导出）traj3d.json+visualization.json。

        返回会话目录；点数不够 / 写不了时返回 ``None``。
        """
        self.close()
        if stats.points < int(min_points) and not self.keep_anyway:
            self._discard()
            print(f"[会话] 三维点 {stats.points} < {min_points}，本次不落盘"
                  "（避免在数据管理里留一堆空会话；想留档就多跑一会儿）")
            return None
        d = self._ensure_dir()
        try:
            (d / "session.json").write_text(
                json.dumps(self._payload(stats), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
            self._write_csv(d, stats)
            self._write_seeds(d, stats)
        except OSError as exc:
            print(f"[会话][错误] 会话写不进去：{exc}")
            return None
        # 导出轨迹 JSON：**放在最后**且单独兜住 —— 它读 session.json + CSV，
        # 出问题只该让"3D 查看器降级"，不该把已经写好的元数据/视频一起废掉。
        try:
            from stereo.visualization import export_session_visualization
            export_session_visualization(
                d, stereo_mode=self.mode,
                # 实时入口**不做**世界系：这里显式给一个不存在的路径，让导出
                # 保持相机系。不传参就会去读 config/world_frame.json ——
                # 用户一旦建了那个文件，parallel 的导出会直接抛错（校正系不能乘
                # T_world_cam），那时整条会话就白存了。
                world_frame_path=d / "derived" / "__no_world__.json")
        except Exception as exc:                        # noqa: BLE001
            print(f"[会话][警告] 轨迹 JSON 没生成：{exc}"
                  "（session.json 与 CSV 还在；3D 查看器会降级成空场景）")
        return d

    def _discard(self) -> None:
        """删掉**本次自己建**的目录（点数不够 / 中途失败）。别人的数据不碰。"""
        self.close()
        if self.dir is not None:
            shutil.rmtree(self.dir, ignore_errors=True)
            self.dir = None

    def _payload(self, stats: LoopStats) -> dict:
        w, h = self.image_size
        q = getattr(self.rig, "quality", None) or {}
        meta = getattr(self.rig, "meta", None) or {}
        try:
            baseline = float(getattr(self.rig, "baseline_mm", 0.0) or 0.0)
            angle = float(getattr(self.rig, "angle_deg", 0.0) or 0.0)
        except (TypeError, ValueError):
            baseline = angle = 0.0
        frames = int(stats.frame_index)
        return {
            "schema": "sphere-stereo-session",
            "version": 1,
            "id": self.dir.name,
            "created": self.created,
            "label": self.label,
            "notes": "实时双目采集（无原始录像，不能重跑）",
            "stereo_mode": self.mode,
            "source": "realtime",
            # 两路的 width/height 是**必填**：`load_session` 重建 CameraInfo 时
            # 少了它们直接 TypeError（不是 SessionError），宁可少写字段也别写半个对象。
            "left": {"role": "left", "video": "", "width": w, "height": h,
                     "frame_count": frames, "fps_measured": round(self.fps, 3)},
            "right": {"role": "right", "video": "", "width": w, "height": h,
                      "frame_count": frames, "fps_measured": round(self.fps, 3)},
            "calibration": {"path": self.calib_base, "image_size": [w, h],
                            "baseline_mm": baseline,
                            "created": str(meta.get("created") or ""),
                            "verdict": str(q.get("verdict") or "")},
            "rig": {"stereo_mode": self.mode, "baseline_mm": baseline,
                    "angle_deg": angle},
            "sync": {"method": "none", "offset": 0, "confidence": 0},
            "codec": "mp4v",
            "container": "mp4",
            "lossless": False,
            "environment": {},
            "checks": [],
        }

    def _write_csv(self, d: Path, stats: LoopStats) -> None:
        from apps.run_stereo_video import CSV_HEADER
        rows = stats.session_rows
        bad = [i for i, r in enumerate(rows) if len(r) != len(CSV_HEADER)]
        if bad:
            # 列错位不会报错，只会让下游读到错值 —— 宁可不写这一份。
            print(f"[会话][错误] {len(bad)} 行列数与表头不一致（{len(CSV_HEADER)} 列），"
                  "跳过 traj3d.csv")
            return
        with open(d / "derived" / "traj3d.csv", "w", newline="",
                  encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)
            writer.writerows(rows)

    def _write_seeds(self, d: Path, stats: LoopStats) -> None:
        if stats.seed is None:
            # 静默跳过 = 这段会话在数据管理里**永远**显示 ○「未点选球心」，
            # 而三维点其实已经算出来了（`has_traj=True` 也没用，✓ 要求 seeds）。
            # 2026-11-17 联合验证时发现的：两个半边各自的判据都绿，接缝处却是哑的。
            print("[会话][警告] 没取到第一对匹配点的种子 → 数据管理里会显示 "
                  "○「未点选球心」（三维点仍在，3D 查看器照样能看）")
            return
        from stereo.visualization import write_seeds
        ul, vl, rl, ur, vr, rr, frame = stats.seed
        write_seeds(d, (ul, vl, rl), (ur, vr, rr), seed_frame=int(frame),
                    image_size=self.image_size, source="realtime",
                    note="实时自动取第一对匹配点，不是人工点选")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    quiet_repeated_grab_warnings()
    args = parse_args(argv)
    cfg = build_config(args)

    if cfg.detector.method not in REALTIME_METHODS:
        # 配置里可能是 hough/fusion（其它入口允许）。**不静默替换**：
        # 换成 color 会让人以为"配置生效了"，而实际跑的是另一个检测器。
        print(f"[错误] 检测方法 {cfg.detector.method!r} 不支持实时双目。")
        print(f"  可用项：{' / '.join(REALTIME_METHODS)}"
              "（hough 4384.9 ms/帧、fusion 3740.5 ms/帧，见 pipeline.py:190-193）")
        print(f"  怎么办：加 --method color，或改 {args.config or '配置文件'} "
              "里的 detector.method")
        return 2

    log_path = setup_logging("realtime")
    log_environment(_log)
    print(f"日志：{log_path}")

    rig, problem = resolve_rig(args.calib)
    if rig is None:
        print(f"[错误] 双目标定不可用：{problem}")
        print("[怎么办] 先给当前装置做一次双目标定：python apps/calibrate_stereo.py；"
              "或用 --calib 指定标定基名（.npz/.json 可不写后缀）")
        print("[退出码] 2（标定缺失/不可用）")
        return 2
    print(f"[标定] {rig.image_size[0]}x{rig.image_size[1]}，"
          f"基线 {rig.baseline_mm:.1f} mm，光轴夹角 {rig.angle_deg:.1f}°")
    quality = getattr(rig, "quality", None) or {}
    if quality.get("forced_unusable") or str(quality.get("verdict", "")).startswith(
            "不可用"):
        # 不拦：实时窗口的价值就是"现场能看见"。但必须说出来 —— 这份标定
        # 自己的 verdict 已经写着"不可用/仅链路测试"，静默用它会让人把
        # 链路测出来的数当成测量结果。
        print(f"[警告] 这份标定自身判为不可用（{quality.get('verdict', '?')}）"
              "—— 三维结果仅供参考，不要当测量数据用")

    # 重建器**在开相机之前**建：模式不对是静态问题，不该让人先等 150 s 相机超时。
    rec = None
    if not args.no_3d:
        try:
            rec = make_reconstructor(rig)
        except ReconstructionModeError as exc:
            print(f"[错误] {exc}")
            print("[退出码] 2（标定不可用）")
            return 2
        if isinstance(rec, AngledReconstructor):
            print(f"[三维] 大角度重建（原始像素 + 基础矩阵 F，来源 {rec.f_source}）；"
                  f"极线门限 {rec.cfg.max_epipolar_px:g} px；"
                  f"交会角下限 {rec.cfg.min_angle_deg:g}°；坐标系 = 原始左相机系")
        else:
            print(f"[三维] 平行重建（校正后像素）；极线门限 "
                  f"{rec.cfg.max_epipolar_px:g} px")
    else:
        print("[三维] --no-3d：只显示检测/跟踪，不做三角测量")
    print(f"[模式] {rig_stereo_mode(rig)}（{mode_label(rig_stereo_mode(rig))}），"
          f"校正状态 {rig.rectification}")

    wins = WindowManager()
    src = open_pair_source(args)
    if src is None:
        print("[怎么办] 用 --left/--right 指定两台相机（cameras.py list 看编号），"
              "或 --auto-cameras 自动挑")
        return 1
    if not getattr(src, "is_live", False):
        print("[错误] 这个入口只吃**实时相机**（--left/--right），"
              "视频/图片请用 apps/run_stereo_video.py")
        src.release()
        return 2
    src.report_warnings()

    size = getattr(src, "resolved_size", None) or getattr(src, "image_size", None)
    if size and tuple(rig.image_size) != tuple(size):
        # 标定内参只对**标定时的分辨率**成立。分辨率对不上时三角测量仍然"能出数"，
        # 只是每个点的深度都按比例错 —— 这种错看不出、也查不出，必须当场拦。
        print(f"[错误] 相机给的是 {size[0]}x{size[1]}，标定是 "
              f"{rig.image_size[0]}x{rig.image_size[1]} —— 内参对不上。")
        print(f"  怎么办：加 --width {rig.image_size[0]} --height {rig.image_size[1]} "
              "让相机按标定档开；或按当前分辨率重新标定。")
        print("[退出码] 2（标定不可用）")
        src.release()
        return 2

    pipe_l = make_pipeline(cfg, rig, "left")
    pipe_r = make_pipeline(cfg, rig, "right")
    if args.detect_width:
        for pipe in (pipe_l, pipe_r):
            pipe.set_detect_width(args.detect_width)
        print(f"检测跑在 {args.detect_width}px 宽的图上（坐标会自动换算回原图）。"
              f"\n  代价：小目标的定位精度会下降；换来的是两路都能留在实时。"
              f"（全分辨率 color 实测 17.8 ms/路，两路 35.6 ms 达不到 >=20 fps）")
    if args.display_width:
        print(f"显示按 {args.display_width}px 宽输出（对半给每路）—— 实测这一步把"
              f"显示耗时从 ~17.7 ms 降到 ~1.8 ms，是流畅度最大的一项")

    ui = _Ui(args, cfg, pipe_l, pipe_r, rec, wins,
             fps=getattr(src, "pair_fps", None) or args.fps)
    print(f"检测方法：{cfg.detector.method}    预测：{cfg.predict.method}   "
          f"视野 {cfg.predict.horizon_s}s    "
          f"skew 补偿：{'关（--no-skew-fix）' if args.no_skew_fix else '开'}"
          f"（门限 {args.max_skew_ms:g} ms，超限丢对并计数）")
    # 「一个球都找不到」最常见的根因是**颜色阈值和球不匹配**（默认是白/浅色球），
    # 而界面上此前只有一行 `method color`，看不出它在找什么颜色 —— 2026-11-17
    # 用户实测"塑料球和金属球都检不出来"，原因之一就是这个（预设没被传下来）。
    # 把**实际生效**的 HSV 区间打出来，一眼就能判断方向对不对。
    if cfg.detector.method == "color":
        c = cfg.detector.color
        rng = (f"H {c.hsv_lower[0]}–{c.hsv_upper[0]}  "
               f"S {c.hsv_lower[1]}–{c.hsv_upper[1]}  "
               f"V {c.hsv_lower[2]}–{c.hsv_upper[2]}")
        if getattr(c, "use_second_range", False):
            rng += (f"  ＋H {c.hsv_lower2[0]}–{c.hsv_upper2[0]}"
                    f"（红色跨 0 度用双区间）")
        src_txt = (f"预设 {args.preset}" if args.preset
                   else "未给预设 → 默认白/浅色球（V>175、S<80）")
        print(f"[颜色] {src_txt}；生效阈值 {rng}")
        print("       球找不到时先看这一行：球的颜色要落在这个区间里。"
              "彩色球请在第①步选对球色（或用 --preset），"
              "白色/浅色球才用默认档。")
    print("提示：按 c 切换检测方法（color/motion），空格暂停，按 r 重置，"
          "按 3 开/关三维小窗，按 v 录检测画面、按 b 录检测+原始两路，按 s 存快照，"
          f"{wins.fullscreen_hint()}，按 q 退出。")

    def _on_frame(canvas: np.ndarray, stats: LoopStats) -> bool:
        ui.write_frame(canvas)
        return ui.show(canvas, stats)

    # ---- 会话落盘：边跑边写视频，退出时补元数据/轨迹 ----
    # 只有"开了三维 + 默认 save"才建：`--no-3d` 没有点可存，`--no-save-session`
    # 用户明确不要条目，都不该付"每帧多一次 hstack + 编码"的钱。
    recorder = None
    if rec is not None and args.save_session:
        try:
            root = stereo_mode_root(rig_stereo_mode(rig))
        except ValueError:                      # 模式值不合法：不落盘，但照常跑
            root = None
            print(f"[会话][警告] 无法确定会话根目录（stereo_mode="
                  f"{rig_stereo_mode(rig)!r}），本次不落盘")
        if root is not None:
            base = str(args.calib) if args.calib else str(
                current_calibration(need_existing=True)[0])
            recorder = _SessionRecorder(
                root=root, mode=rig_stereo_mode(rig), image_size=rig.image_size,
                fps=getattr(src, "pair_fps", None) or args.fps,
                calib_base=base, rig=rig)
            print(f"[会话] 退出时会存到 {_display_path(root)}/"
                  f"<时间戳>/（三维点 < {MIN_SESSION_POINTS} 不落盘）")
    elif args.save_session and rec is None:
        print("[会话] --no-3d：没有三维点，本次不落会话")
    else:
        print("[会话] --no-save-session：本次不落会话")
    collect_rows = recorder is not None
    # `recorder` 可能为 None（--no-3d / --no-save-session）：
    # 挂上去只为让 `v` 能说一句"这段留住"，None 也要能安全挂。
    ui.recorder = recorder

    stats = LoopStats()
    health_l, health_r = Health(), Health()
    try:
        stats = run_loop(src, pipe_l, pipe_r, rec, stats=stats,
                         health_l=health_l, health_r=health_r,
                         skew_fix=not args.no_skew_fix,
                         max_skew_ms=args.max_skew_ms,
                         max_frames=args.max_frames,
                         display_width=args.display_width,
                         panel_on=lambda: ui.panel_on,
                         hint_fn=ui.hint_line, badge_fn=ui.badge_line,
                         on_frame=_on_frame, log_every=args.log_every,
                         frame_sink=(recorder.write_frame
                                     if recorder is not None else None),
                         collect_rows=collect_rows)
    except KeyboardInterrupt:
        stats.stop_reason = "用户中断（Ctrl-C）"
    finally:
        src.release()
        ui.close()
        if recorder is not None:
            recorder.close()        # 中途异常也要 release，否则 mp4 是坏的
        cv2.destroyAllWindows()

    print(f"\n[收尾] 结束原因：{stats.stop_reason or '正常退出'}")
    for line in stats.summary_lines():
        print(f"[收尾] {line}")
    print(f"[收尾] 健康度 左：{health_l.summary_line()}")
    print(f"[收尾] 健康度 右：{health_r.summary_line()}")
    print(f"[收尾] PairSource 层丢弃的对数（skew 门禁在库层触发时为真）："
          f"{getattr(src, 'dropped_pairs', 0)}")
    if rec is not None:
        # `summary()` 两种重建器都有，且各自报**自己的**等价量：
        # parallel 报交会残差/深度，angled 报极线距离/交会角/深度。
        print(f"[收尾] 重建器（{stats.mode}）：{rec.summary()}")
        print(f"[收尾] 配对 {rec.n_matched}/{rec.n_frames}，"
              f"左缺 {rec.n_no_left} / 右无候选 {rec.n_no_right} / "
              f"被门限拒 {rec.n_rejected}")
        # `warnings` / `dy_samples` 只有 parallel 重建器有；大角度用
        # `low_match_diagnostic()` 说同一件事（配对率低时该看什么）。
        for w in getattr(rec, "warnings", []):
            print(f"[收尾][警告] {w}")
        for line in rec.low_match_diagnostic():
            print(f"[收尾] {line}")
        if rec.n_matched == 0 and rec.n_frames >= 20:
            print("[收尾][诊断] 一个三维点都没配上 —— 先看上面 HUD 的 detect/hits，"
                  "再确认两路是**同一场景的左右两个视角**（不是两台拍到同一侧）")
            if isinstance(rec, AngledReconstructor):
                print("[收尾][诊断] 大角度下还要看两点：① 标定的 stereo_mode / "
                      "光轴夹角是否与实物一致；② 两路是否真的共视（大角度下重叠区小）")
        geom = stats.verdict
        if geom is not None:
            print(f"[收尾] 极线几何：y 差中位 {geom.med:.2f} px（p95 {geom.p95:.2f}，"
                  f"n={geom.n}）"
                  + (f"，标定自报 {geom.epipolar_rms:.2f} px" if geom.has_ref else "")
                  + ("  -> suspect：与标定对不上，结果仅供参考"
                     if geom.suspect else "  -> 与标定相符"))
            # 完整的三步建议（`geometry_lines` 里已按"几何 vs 检出质量"分开说）
            for line in getattr(rec, "geometry_lines", list)():
                print(f"[收尾] {line}")

    # ---- 落会话（放在最后：上面那些统计先落到终端，用户已看到结果）----
    if recorder is not None:
        saved = recorder.save(stats)
        if saved is not None:
            print(f"[会话] 已存 {_display_path(saved)}（三维点 {stats.points} / 帧 "
                  f"{stats.frame_index}"
                  + (f" / 视频 {recorder.video_frames} 帧"
                     if recorder.video_frames else " / 无视频")
                  + "，可在数据管理里打开，3D 查看器可看轨迹）")
            print("[会话] 没有原始录像：数据管理里那一步「重跑」点不了"
                  "（没有 left/right.mkv），只能看这条实时轨迹")
        _log.info("会话落盘：%s", saved)
    _log.info("运行结束。%s", " | ".join(stats.summary_lines()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
