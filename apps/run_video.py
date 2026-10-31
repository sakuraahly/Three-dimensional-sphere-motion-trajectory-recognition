"""离线处理视频/图片：标注轨迹 + 预测 + 导出 CSV + 单目三维可视化。

用法::

    python apps/run_video.py assets/videos/ball.mp4
    python apps/run_video.py assets/videos/ball.mp4 --out outputs/videos/annotated.mp4 --csv outputs/traj.csv
    python apps/run_video.py assets/images/ --frames-dir          # 整个文件夹当序列
    python apps/run_video.py clip.mkv --method hough --vis-json outputs/vis.json

## 为什么"0 检出"必须失败（2026-10-29）

现场：一段 288 帧的单目录像，默认 ``color`` 检测器 40 帧里检出 **0 帧**，
而进程照样 exit 0、照样写出 CSV 与"标注视频"。用户看到的是"跑完了"，
拿到手的是一个 x/y 全空的表格 —— 这就是"这个功能有什么用"的答案。

所以这里把"什么都没测到"和"测到了一点"分成两种结局：

- ``hits == 0``        → 打印诊断（当前检测器 + 该换哪个）+ **退出码 2**，
  并且**不写** CSV / visualization.json / 标注视频（写了也全是空值）；
- ``0 < hits/total < 10%`` → 一行警告，但仍 exit 0：确实有数据，
  不该假装失败，可这么低的比率不足以当测量用。

理由一句话：本项目最贵的一类 bug 是"看起来成功"。

## "有输出"不等于"量到了球"（2026-10-30）

上一版把"检测器给了东西"就叫检出，于是同一段录像上 ``hough`` 报 **96.7%**，
而人眼核对只有 **1.4%** 落在球上：那 96% 是书架/椅子上的小圆（r≈11~13 px），
半径稳定、位置稳定，看起来完全像一条正常轨迹。球本身半径约 91 px，
``hough_max_radius=80`` 比它还小 —— 球根本不在搜索区间里。

所以本版把"检出"换成 **有效帧**：检测器给出候选之后，还要

1. 用 ``refine_circle_subpixel`` 从**边缘**重新量一次半径（检测器的 radius 是
   跟踪框的一半，回答的是"框多大"，不是"球多大"）；
2. 过一致性门（逐帧跳变 ≤8%、与最近 30 个已接受半径的中位相差 ≤25%）。

只有过了门的帧才算有效帧，退出码与可视化文件里的点数都以它为准。
`--seed x,y,r` 让人指定"就是这颗球"，因为自动检测**无法**判断一个自洽的圆
到底是不是球 —— 这一段 `hough` 的数据就是反例（详见 `[尺度]` 那行）。

**门必须锚在人给的尺度上**（同日修正）：给了种子就锚种子半径
（`seed_r × [0.75, 1.35]`），没给种子才退回滚动中位，并且报告里明说
"半径未经确认、绝对深度不可信"。滚动中位是**自证**的 —— 旧
`derived/traj.csv` 里 277 帧半径全是 11 px（真值 91），中位就是 11，
每帧都"在带内"，于是全过。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

# --- 让脚本在任意目录深度都能找到项目根 ---
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "sphere_tracker").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import csv

import cv2
import numpy as np

from sphere_tracker.config import AppConfig, ColorConfig, load_config
from sphere_tracker.detectors import registry  # noqa: F401  （import 即注册：R1）
from sphere_tracker.detectors.neural import mark_conf_explicit
from sphere_tracker.geometry import refine_circle_subpixel
from sphere_tracker.pipeline import SpherePipeline
from sphere_tracker.sources import open_source
from apps._common import add_mono_pipeline_args, apply_mono_calibration
from apps._window import WindowManager

WINDOW = "run_video"

#: 单目三维可视化的 schema。**刻意与双目不同名**：两者字段长得像，
#: 但尺度来源完全不同（单目靠球径先验、双目靠视差），混用一个 schema 会让
#: 下游"以为自己在读视差测量"。
MONO_VIS_SCHEMA = "sphere-mono-visualization"
MONO_VIS_VERSION = 1

#: 检出率低于它就说"结果多半不可用"（但不判失败，见模块 docstring）。
LOW_HIT_RATE = 0.10

#: 亚像素半径细化的搜索半宽（像素）。与 `apps/tools/mono_vs_stereo.py` 的
#: ``_REFINE_SEARCH`` 同值同理由：35 fps 下球的像素半径逐帧只变 <1.5%，
#: 细化只需要在上一帧半径附近 ±3 px 找边缘。**不能放宽** —— 镜面球的边缘
#: 与书架纹理只隔几个像素，窗口一宽，射线会先抓到背景边缘。
REFINE_SEARCH_PX = 3.0

#: 逐帧半径跳变容忍度。与 ``mono_vs_stereo._MAX_RADIUS_JUMP`` 同口径（8%）：
#: 实测可测帧的半径逐帧变化中位 0.33%、p95 0.87%、最大 1.40%，8% 是近 6 倍余量；
#: 超过它只可能是细化射线锁到了背景边缘，此时宁可沿用上一帧半径。
MAX_RADIUS_JUMP = 0.08

#: 半径带：与"最近 N 个已接受半径的中位"相差超过这一比例就整帧不认（**没给种子时**用）。
#: 窗口 30 帧 ≈ 0.86 s @35 fps：比这短的波动（抖动）交给跳变护栏，
#: 比这长的漂移（球真的靠近/远离）必须允许，所以窗口不能再短。
#: ±25% 的依据：实测可测帧半径的 IQR 只有 ±2.6%（89.2~93.9 px），25% 是近 10 倍
#: 余量 —— 它挡不住随机抖动（那是跳变护栏的事），只挡"锁到了另一个圆"这种量级错误。
RADIUS_WINDOW = 30
RADIUS_BAND = 0.25

#: 给了 ``--seed`` 时，锚点换成**种子半径**，带是 ``seed_r × [1-下界, 1+上界]``。
#:
#: 为什么必须锚种子、不能锚滚动中位：中位是**自证**的。旧 `derived/traj.csv`
#: （GUI 默认 hough）288 帧里 277 帧 detected=1、半径中位 11 px（真值 91），
#: 全都≈11 ⟹ 中位就是 11 ⟹ 每一帧都"在 ±25% 内"，于是全过。滚动中位只能
#: 说明"这几十帧是同一类圆"，不能说明"这就是用户点的那颗球"。
#:
#: 上界 35%（不是 25%）的依据：这段录像里半径从 89.2 变到 93.9（+5%），
#: 球被甩近时还会再大一点，35% 留够余量又不会放过 2 倍大的背景圆；
#: 下界 25% 用来挡"锁到小球"（旧结果的 11 px 相对 88 px 是 -87.5%）。
SEED_BAND_LO = 0.25
SEED_BAND_HI = 0.35

#: 0 有效帧时的换法建议：顺序 = 最可能有效的排在前面。
#: 每条都带"什么情况下该用它"，因为用户此刻唯一的信息就是"球在画面里"。
METHOD_HINTS = (
    ("hough", "球颜色不突出、与背景接近（几何线索；颜色不可靠时的第一选择）"),
    ("fusion", "有同色干扰物、或单一线索不稳（多线索投票，最稳但最慢）"),
    ("specular", "球是金属/高光（颜色阈值抓不到高光球，它量的是高光斑）"),
    ("motion", "球在动、背景基本静止（背景差分；球与背景同色时唯一的线索）"),
)


def parse_seed(text: str) -> tuple[float, float, float]:
    """把 ``--seed x,y,r`` 解析成三元组；写错就报错退出（不静默用默认值）。

    为什么不接受"只给一部分"：seed 的三个数分别决定"初始化框放在哪"
    与"框多大"。缺任何一个都只能靠猜，而猜错的后果是**看起来在跟踪**
    （CSRT 会老实地跟住框里的东西，哪怕那是书架）。
    """
    parts = [s.strip() for s in str(text).replace("，", ",").split(",")]
    if len(parts) != 3:
        raise SystemExit(
            f"[错误] --seed 要写成 x,y,r（像素），收到 {text!r}。"
            "例如：--seed 895,718,88")
    try:
        x, y, r = (float(s) for s in parts)
    except ValueError:
        raise SystemExit(
            f"[错误] --seed 的三个数都必须是数字，收到 {text!r}") from None
    if r <= 0:
        raise SystemExit(f"[错误] --seed 的半径必须 > 0，收到 r={r:g}")
    return x, y, r


class RadiusGate:
    """逐帧半径一致性门，同时负责统计。抽成类是为了能被测（判定与打印分开）。

    门有两级，管的是两件不同的事：

    - **跳变护栏**（``MAX_RADIUS_JUMP``）：这一帧的半径相对上一帧跳了 8% 以上，
      说明细化射线抓到了别的东西。**只回退半径**（沿用上一帧），帧仍然算有效 ——
      球心与轨迹是这一帧量到的，只有半径没量到。
    - **半径带**：整帧不认。锚点分两种，**这是本轮最要紧的一处**：
        * 给了 ``--seed``：锚 ``seed_r``，范围 ``[0.75, 1.35]×seed_r``。
          种子是用户"就这个球"的声明，也就是这一趟的**尺度基准**；
          拿它当锚，才不会像滚动中位那样自证（见 `SEED_BAND_LO` 的注释）；
        * 没给种子：只能锚最近 30 帧的滚动中位（±25%）。它**证明不了**
          "这就是那颗球"，所以报告里必须同时打 `[尺度]` 那条警告 ——
          门的强度不许大于它实际能保证的东西。

    分开的理由：跳变是"这一帧没量好"，带越界是"跟错目标了"，前者可以修补，
    后者只能丢弃 —— 混成一个门就分不出该重测还是该重捕。
    """

    def __init__(self, seed_radius: float | None = None) -> None:
        self.seed_radius = float(seed_radius) if seed_radius is not None else None
        self.window: deque[float] = deque(maxlen=RADIUS_WINDOW)
        self.r_prev = self.seed_radius
        if self.r_prev is not None:
            # 种子半径就是"第一份已接受半径"：之后没给种子时才用的滚动中位
            # 也以它为起点，免得第一帧细化失败把锚定在背景上。
            self.window.append(self.r_prev)
        self.jumps = 0
        self.inconsistent = 0
        self.jump_ratios: list[float] = []
        self.radii: list[float] = []      # 全部已接受半径（统计用，不受窗口长度限制）

    def band(self) -> tuple[float, float] | None:
        """本门此刻的接受区间；没给种子且样本不足时返回 ``None``（只跳变护栏生效）。"""
        if self.seed_radius is not None:
            return (self.seed_radius * (1.0 - SEED_BAND_LO),
                    self.seed_radius * (1.0 + SEED_BAND_HI))
        if len(self.window) >= 3:
            med = float(np.median(self.window))
            return (med * (1.0 - RADIUS_BAND), med * (1.0 + RADIUS_BAND))
        return None

    def accept(self, r: float) -> tuple[float | None, str]:
        """返回 ``(本帧采用的半径 或 None, 判定)``；判定 ∈ ok / jump / inconsistent。

        **顺序要紧**：带必须先判、且判在**这一帧量到的** ``r`` 上。
        如果先跑跳变护栏，它会先把 ``r`` 拉回上一帧（8% 以内），而带至少 25% ——
        被拉回的值永远进不了带外，带就成了永不触发的死代码。
        """
        band = self.band()
        if band is not None and not (band[0] <= r <= band[1]):
            self.inconsistent += 1
            return None, "inconsistent"
        verdict = "ok"
        if self.r_prev is not None and self.r_prev > 1e-6:
            ratio = abs(r - self.r_prev) / self.r_prev
            self.jump_ratios.append(ratio)
            if ratio > MAX_RADIUS_JUMP:
                verdict = "jump"
                self.jumps += 1
                r = self.r_prev          # 沿用上一帧；见类 docstring
        self.window.append(float(r))
        self.radii.append(float(r))
        self.r_prev = float(r)
        return float(r), verdict


def loss_segments(bad_frames: list[int]) -> list[tuple[int, int]]:
    """把"没量到"的帧号列表压成连续段 ``[(起, 止), ...]``（含两端）。

    为什么必须按段报：逐帧一行会把"球出画了 0.3 s"淹没在 112 行里，
    而这段数据真正要紧的信息就是"它是一段连续的丢失，不是随机掉帧"。
    """
    segs: list[tuple[int, int]] = []
    for f in bad_frames:
        if segs and f == segs[-1][1] + 1:
            segs[-1] = (segs[-1][0], f)
        else:
            segs.append((f, f))
    return segs


def session_name(path: str | Path) -> str:
    """可视化文件里记的 ``session``：视频所在目录名，取不到就用文件名。

    为什么不用文件名的 stem：会话的语义单位是**目录**（一次采集），
    而查看器是"一个会话一条轨迹"。用文件名会得到 ``clip`` 这种毫无信息量的名字。
    """
    p = Path(path)
    return (p if p.is_dir() else p.parent).name or p.name


def hit_report(hits: int, total: int, method: str,
               detector_error: str = "") -> tuple[int, list[str]]:
    """按**有效帧**给出 (退出码, 要打印的行)。抽成纯函数是为了能被测。

    这里的 ``hits`` 是过了半径一致性门的帧数，不是"检测器给了东西"的帧数 ——
    这一改动的现场见模块 docstring（hough 96.7% 有输出、1.4% 在球上）。

    - 0 有效帧 → 2 + 诊断。**必须是非 0**：退出码是脚本唯一的"结论"，
      0 意味着"成功"，而这里恰恰什么都没测到。
    - 低有效帧 → 0 + 一行警告。有数据就不该假装失败，但也不能装作够用。
    """
    rate = hits / total if total else 0.0
    if total == 0:
        return 2, [
            "[失败] 一帧都没读到（视频打不开 / 文件是空的 / 路径给错了）。",
            "       退出码 2 —— 不要把这个当成「跑完了」。",
        ]
    if hits == 0:
        lines = [
            f"[失败] {total} 帧里没有一帧通过一致性门 —— 这次什么都没测到，结果不可用"
            "（后面不会写出 CSV / 三维可视化：写了也只是一张空表）。",
            f"       当前检测器：{method}",
            "       退出码 2（不是 0）：本项目最贵的一类 bug 是「看起来成功」——",
            "       「跑完了」和「什么都没测到」在屏幕上长得一样，只能靠退出码分开。",
        ]
        if detector_error:
            lines.append(f"       检测器自述原因：{detector_error}")
        lines.append("       换个检测器再跑（按可能性排序，命令行加 --method <名字>）：")
        lines += [f"         --method {name:<9} {why}" for name, why in METHOD_HINTS]
        lines.append("       命令里补上「就这颗球」的信息（像素坐标与半径）：")
        lines.append("         --seed x,y,r    金属/镜面球**必须**给：自动检测分不出"
                     "「半径自洽的圆」与「球」，它只会给你一个自洽的错答案。")
        return 2, lines
    if rate < LOW_HIT_RATE:
        return 0, [
            f"[警告] 有效帧只有 {rate * 100:.1f}%（{hits}/{total} 帧），"
            f"结果多半不可用；试试 --method hough / fusion，金属球加 --seed",
            "       仍然按退出码 0 结束：确实有数据，不该假装失败。"
            "但这么低的比率不足以当测量用。",
        ]
    return 0, []


def consistency_lines(*, total: int, accepted: int, no_detection: int,
                      inconsistent: int, jumps: int, radii: list[float],
                      jump_ratios: list[float], bad_frames: list[int],
                      fps: float,
                      seed: tuple[float, float, float] | None) -> list[str]:
    """四段报告：有效帧 / 一致性 / 丢失 / 尺度。抽成纯函数是为了能被测。

    口径（本项目刻意不让"检出率"单独出现在屏幕上）：

    - ``[有效帧]`` 的分母是**读到的帧数**，分子是过了半径门的帧数；
      括号里把剩下的帧按原因拆干净，且**加得起来**：无检测 + 半径不一致
      = 总数 − 有效帧。跳变护栏回退的帧仍在有效帧里（只回退半径，帧没丢），
      所以它单列次数而不是帧数 —— 混进拆分会变成一个加不平的账。
    - ``[一致性]`` 给半径的中位与四分位，以及逐帧跳变的分位数：
      这三个数一起看才知道"这条轨迹稳不稳"，只看中位数看不出逐帧抖动。
    - ``[丢失]`` 按**连续段**报：一段连续的丢失通常是球出画 / 运动模糊 /
      被遮挡（球没给出可测的影像），而散落的单帧丢失才是检测不稳。
    - ``[尺度]`` 说清"半径是谁定的"。没有种子时**必须**明说绝对深度不可信：
      自动检测只能证明"有个圆被连续跟住"，不能证明它是那颗球 ——
      旧 `derived/traj.csv` 就是 277 帧自洽的 11 px 背景圆，深度 1.29 m（真 0.164）。
    """
    rate = accepted / total if total else 0.0
    lines = [
        f"[有效帧] {accepted}/{total} = {rate * 100:.1f}%"
        f"（其余 {total - accepted} 帧：无检测 {no_detection}、"
        f"半径不一致 {inconsistent}；跳变护栏触发 {jumps} 次，只回退半径不丢帧）"
    ]

    if radii:
        med = float(np.median(radii))
        q1, q3 = (float(v) for v in np.percentile(radii, [25.0, 75.0]))
        if jump_ratios:
            p50, p95 = (float(v) for v in np.percentile(jump_ratios, [50.0, 95.0]))
            jump_txt = (f"逐帧跳变 中位 {p50 * 100:.2f}% / p95 {p95 * 100:.2f}% / "
                        f"最大 {max(jump_ratios) * 100:.2f}%")
        else:
            jump_txt = "逐帧跳变 无数据（只有 1 帧过门）"
        lines.append(f"[一致性] 半径 {med:.1f} px（IQR {q1:.1f}~{q3:.1f}），{jump_txt}")
    else:
        lines.append("[一致性] 没有一帧通过半径门，半径统计无数据")

    segs = loss_segments(bad_frames)
    if accepted == 0:
        # 一帧都没测到 ≠ "连续丢失一段"：前者是从头就没跟上（原因在诊断里），
        # 后者是"跟住过、后来丢了"。把两者写成同一句话会把根因藏起来。
        lines.append(f"[丢失]   一帧都没测到（{len(bad_frames)} 帧全空）—— "
                     "不是「跟丢了一段」，是从来没跟上，原因见下面的诊断")
    elif not segs:
        lines.append("[丢失]   无")
    else:
        fps_ok = fps if fps and fps > 1.0 else 0.0
        parts = []
        for start, end in segs:
            n = end - start + 1
            dur = f"，{n / fps_ok:.1f} s" if fps_ok else ""
            parts.append(f"帧 {start}..{end}（{n} 帧{dur}）")
        lines.append(f"[丢失]   连续丢失 {len(segs)} 段：" + "；".join(parts)
                     + " —— 球出画或运动模糊时是正常的，不是检测不稳")

    if seed is None:
        # 这行不是免责声明，是**结论的一半**：门只证明了"同一个圆被连续跟住"，
        # 没有证明"这个圆是那颗球"。旧 traj.csv 的 277 帧就是自洽的背景小圆。
        lines.append("[尺度] 无人工种子：半径未经确认，绝对深度不可信"
                     "（自动检测只能给出「有个圆」，不能证明是这个球）")
    else:
        lo = seed[2] * (1.0 - SEED_BAND_LO)
        hi = seed[2] * (1.0 + SEED_BAND_HI)
        lines.append(f"[尺度] 人工种子 {seed[2]:g} px 锚定：接受半径 "
                     f"{lo:.1f}~{hi:.1f} px（−{SEED_BAND_LO:.0%} / +{SEED_BAND_HI:.0%}）")
    return lines


def build_vis(rows: list[dict], *, session: str, source: str,
              ball_diameter_mm: float, fx: float,
              created: str | None = None,
              seed: tuple[float, float, float] | None = None,
              accepted_frames: int | None = None,
              rejected_frames: int | None = None) -> dict | None:
    """把逐帧结果整理成单目可视化文件；没有三维点就返回 ``None``（不写空文件）。

    单目**本来就有三维**：``Z = f·D/(2r)``（``CameraModel.depth_from_radius``），
    只是以前从来没落盘 —— 于是"主界面只能看双目"看起来像"单目没有 3D"。

    两个量从**实际生效的配置**取，不写字面量：
    ``ball_diameter_mm`` 与 ``fx`` 都是 Z 的直接输入，写死一份等于让文件里
    记的尺度和真正算出来的尺度成为两个数。

    ``seed`` / ``scale_unverified`` / ``accepted_frames`` / ``rejected_frames``
    是**溯源信息**：半年后打开这份 json，必须一眼能判断"这条轨迹是人工点的
    种子跟出来的（尺度有人背书）"还是"自动检测的（尺度未确认）"，以及有多少帧
    被一致性门挡掉。没有它们，两者长得一样。
    """
    pts = []
    for row in rows:
        if "X_m" not in row:            # 只有 position_3d 非空的行才有这一列
            continue
        pts.append({
            "frame": int(row["frame"]),
            "t": float(row["t"]),
            "xyz": [row["X_m"], row["Y_m"], row["Z_m"]],
            # coasting = 这一帧没有检测，位置是跟踪器外推的
            "confident": not bool(row.get("coasting")),
        })
    if not pts:
        return None
    return {
        "schema": MONO_VIS_SCHEMA,
        "version": MONO_VIS_VERSION,
        "session": session,
        "created": created or time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "unit": "m",
        "mode": "mono",
        "coordinate_system": "mono-camera",
        "ball_diameter_mm": round(float(ball_diameter_mm), 3),
        "fx": round(float(fx), 3),
        "n_points": len(pts),
        "trajectory": pts,
        "seed": None if seed is None else [round(float(v), 3) for v in seed],
        # 没有种子 ⟹ 半径是自动检测来的 ⟹ 绝对深度只是"某个圆的深度"。
        # 这一位必须落盘：否则半年后的读者会把这两类轨迹当成同一种测量。
        "scale_unverified": seed is None,
        "accepted_frames": int(accepted_frames if accepted_frames is not None else len(pts)),
        "rejected_frames": int(rejected_frames or 0),
        "notes": ("深度由球径先验 Z=f·D/(2r) 得到，尺度依赖 D；"
                  "不是视差测量，误差约 2%（见交接文档 §35.2）"),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="离线视频/图片序列的球体轨迹识别")
    p.add_argument("path", help="视频文件、图片文件或图片文件夹")
    p.add_argument("--out", default=None, help="输出标注视频（.mp4）")
    p.add_argument("--csv", default=None, help="导出轨迹 CSV")
    p.add_argument(
        "--vis-json", default=None, metavar="PATH",
        help="导出单目三维可视化（sphere-mono-visualization），"
             "供 apps/view_3d.py 打开；只写 position_3d 非空的行")
    p.add_argument("--frames-dir", action="store_true", help="输入是图片文件夹时，输出逐帧 PNG")
    # 八个流水线参数与 `run_realtime.py` **同源**（`apps/_common.py`）。
    # 这里原来手抄了一份，实测已经和实时那边漂了：--config 没说明、
    # --ball-diameter/--focal/--horizon/--gravity-fit 全都没有说明、
    # 而 --preset 的说明里抄了一份预设名单。详见该函数的 docstring。
    add_mono_pipeline_args(p)
    p.add_argument(
        "--seed", default=None, metavar="X,Y,R",
        help="人工种子：球心像素坐标与像素半径，如 --seed 895,718,88。"
             "给了它就会自动切到 specular（只有它对镜面球有效）；"
             "没有它，脚本只能报「半径自洽的圆」，不能保证那是球")
    p.add_argument(
        "--weights", default=None, metavar="PATH",
        help="YOLO 权重路径，覆盖配置里的 yolo_weights（只对 --method yolo / yolo_sam 生效）。"
             "不传时：配置仍是默认权重且 models/ball_yolo.pt 存在 → 自动用自训权重")
    p.add_argument(
        "--classes", default=None, metavar="0,32",
        help="只保留这些类别号（逗号分隔），覆盖配置里的 yolo_classes。"
             "只有 COCO-80 权重需要它；自训单类球模型（0=ball）一律用全部类别，传了也不生效")
    p.add_argument(
        "--conf", type=float, default=None, metavar="0.25",
        help="YOLO 置信度门（覆盖配置里的 yolo_conf，只对 --method yolo / yolo_sam 生效）。"
             "**不传就按自训权重旁的建议阈值走**（models/ball_yolo.conf.json，训练时在 val 上标定）；"
             "显式给了就一律不动")
    p.add_argument(
        "--imgsz", type=int, default=None, metavar="960",
        help="YOLO 推理分辨率（覆盖配置里的 yolo_imgsz，默认 640）。"
             "球在画面里很小（几十像素）时才需要调大，代价是逐帧推理变慢")
    p.add_argument("--max-frames", type=int, default=0, help=">0 时只处理前 N 帧")
    p.add_argument("--no-show", action="store_true", help="不弹窗（批处理用）")
    return p.parse_args()


#: 无人工种子时，判断"检测器给的半径到底有没有边缘证据"的三个门（2026-10-30）。
#:
#: 为什么需要它：YOLO 给的是**框半宽**，镜面球的框通常比球大一圈；而
#: ``refine_circle_subpixel`` 只在 ±3 px 内找边缘，够不到真轮廓 —— 半径就停在
#: 框半宽上，``Z=f·D/(2r)`` 直接错同一个比例（实测 49.8 vs 91.2 px = 深度差 1.8 倍），
#: 而屏幕上还是「100% 有效帧」。门只能证伪跳变，证伪不了起点。
#:
#: 为什么不是"宽搜找到真半径"：在真实帧上量过，这颗镜面球下半圈是暗对暗、
#: 整圈只有 25%~35% 有梯度；扫一遍取最高分永远挑到球内部反射的亮斑
#: （r≈27~34 的分数比真轮廓高一倍）。**没有证据时猜一个半径，比不猜更危险**，
#: 所以这里只回答"能不能验证"，不负责"猜一个更好的"。
RADIUS_EVIDENCE_MIN_SUPPORT = 0.5   # 轮廓上至少一半采样点要有沿半径的梯度
RADIUS_EVIDENCE_MIN_DELTA = 0.05    # 且要比 ±15% 的两圈明显更像边缘（否则是"到处都有边"）
RADIUS_EVIDENCE_NEIGHBOR = 0.15


def radius_evidence(gray: np.ndarray, cx: float, cy: float,
                    r: float) -> tuple[float, str]:
    """返回 ``(轮廓支持度, 判定)``，判定 ∈ ``{"ok", "weak"}``。

    判定"这个半径有没有边缘证据"的办法是**对照**，不是绝对值：
    在 r 和它的 ±15% 两圈上各算一次"沿半径方向梯度占比"，
    - 支持度 ≥ 0.5（球轮廓至少有半圈有边），且
    - 与两圈的差 ≥ 0.05（说明 r 这一圈确实比里外更像边缘）
    才算 ok。只比绝对值的坑：满画面纹理时每一圈都是 0.5，那不代表找到了轮廓。

    纯函数：不读视频、不看全局状态，可以直接喂合成图断言。
    """
    from sphere_tracker.geometry import circle_radial_gradient_stats

    if gray is None or r is None or r < 3.0:
        return 0.0, "weak"
    support = float(circle_radial_gradient_stats(gray, float(cx), float(cy), float(r))[0])
    lo = float(circle_radial_gradient_stats(
        gray, float(cx), float(cy), float(r) * (1.0 - RADIUS_EVIDENCE_NEIGHBOR))[0])
    hi = float(circle_radial_gradient_stats(
        gray, float(cx), float(cy), float(r) * (1.0 + RADIUS_EVIDENCE_NEIGHBOR))[0])
    delta = max(abs(support - lo), abs(support - hi))
    verdict = "ok" if (support >= RADIUS_EVIDENCE_MIN_SUPPORT
                       and delta >= RADIUS_EVIDENCE_MIN_DELTA) else "weak"
    return support, verdict


def radius_evidence_lines(radius: float, support: float, verdict: str) -> list[str]:
    """把判定翻成人话。抽成纯函数是为了能断言"weak 必须说未经验证"。

    不拒绝出结果：有中心、有近似半径，比 0 帧有用；但必须说清它是**未验证**的，
    否则下一个人会把 1.8 倍的深度当成测量值。有种子时**不打印**（用户点的半径就是基准）。
    """
    if verdict == "ok":
        return [f"[半径] {radius:.1f} px（来自检测器框半宽；轮廓 {support * 100:.0f}% 有梯度，"
                "边缘验证通过）"]
    return [
        f"[半径] {radius:.1f} px（来自检测器框半宽，**未经边缘验证**："
        f"球轮廓只有 {support * 100:.0f}% 有梯度）",
        "  · 影响：Z = f·D/(2r)，半径偏 15% 就是深度偏 15%",
        "  · 要准确深度：点一次「点选球心」，或换背景更干净的一段",
    ]


def parse_classes(text: str) -> list[int]:
    """把 ``--classes 0,1`` 解析成 ``[0, 1]``；写错就报错退出（不静默用默认值）。

    为什么不接受"解析失败就当没给"：类别过滤写错的后果是**永远 0 检出**，
    而 0 检出在屏幕上和"模型没学会"长得一模一样。宁可当场停下来说清楚格式。
    顺带把中文逗号也当分隔符 —— 中文输入法下出这个字符太常见了。
    """
    parts = [s.strip() for s in str(text).replace("，", ",").split(",") if s.strip()]
    if not parts:
        raise SystemExit(
            f"[错误] --classes 至少要有一个类别号，收到 {text!r}。例如：--classes 0 或 --classes 0,32")
    out: list[int] = []
    for part in parts:
        try:
            value = int(part)
        except ValueError:
            raise SystemExit(
                f"[错误] --classes 里每项都要是整数类别号，收到 {part!r}。"
                "例如：--classes 0,32（自训单类球模型是 0）") from None
        if value < 0:
            raise SystemExit(f"[错误] --classes 的类别号不能是负数，收到 {value}")
        out.append(value)
    return out


def main() -> int:
    args = parse_args()
    seed = parse_seed(args.seed) if args.seed else None
    cfg = load_config(args.config) if args.config else AppConfig()
    if args.method:
        cfg.detector.method = args.method
    # 权重与类别号在**方法确定之后**覆盖：两者只对 yolo / yolo_sam 有意义，
    # 覆盖的是同一个 DetectorConfig，不影响 color/hough。
    if args.weights:
        cfg.detector.yolo_weights = args.weights
    if args.classes is not None:
        cfg.detector.yolo_classes = parse_classes(args.classes)
    if args.conf is not None:
        # 为什么需要这个旋钮（2026-10-30 实测）：刚训完 3 个 epoch 的模型最高分只有
        # 0.04，而默认 yolo_conf=0.25 会让整段录像 0 检出 —— 现象和"权重没接上"
        # 一模一样。没有这个旋钮，用户只能去改配置文件才能判断到底是哪一种。
        cfg.detector.yolo_conf = args.conf
        # 同时声明"这是人给的"：YoloDetector 会按权重旁的 sidecar 校准阈值，
        # 但那只在用户没表态时做。不标记的话 `--conf 0.25`（恰好等于配置默认值）
        # 会被悄悄换成 0.05 —— 用户的显式选择被藏起来，事后无从解释。
        mark_conf_explicit(cfg.detector)
    if args.imgsz is not None:
        if args.imgsz < 64:
            # 太小不是"更快"，是"球只有几个像素"，只会得到 0 检出。
            raise SystemExit(f"[错误] --imgsz 至少 64，收到 {args.imgsz}")
        cfg.detector.yolo_imgsz = args.imgsz
    # 给了种子就等于说"我指的就是这颗球，跟住它"。而 color/hough 不接受种子，
    # 会继续在整幅画面里找自己那一票 —— 本仓库那段录像上它锁定的是书架上的小圆
    # （r≈11~13 px，而球约 91 px）。所以自动切到唯一吃种子的 specular，并说明原因：
    # 静默切换会让"我明明 --method hough"的自我怀疑无处安放。
    if seed is not None and cfg.detector.method != "specular":
        print(f"[说明] 给了 --seed，检测器从 {cfg.detector.method} 切到 specular："
              "人工种子的语义是「就这颗球」，而 specular 是唯一接受种子的检测器"
              "（color/hough 会继续在背景里锁自己的圆）")
        cfg.detector.method = "specular"
    if args.preset:
        cfg.detector.color = ColorConfig.preset(args.preset)
    if args.ball_diameter:
        cfg.camera.ball_diameter_m = args.ball_diameter
    # 内参：标定文件（含畸变）优先，--focal 再覆盖；**来源每次都打印**
    print(f"[{apply_mono_calibration(cfg, args)}]")
    if args.predict:
        cfg.predict.method = args.predict
    if args.horizon:
        cfg.predict.horizon_s = args.horizon
    if args.gravity_fit:
        cfg.predict.use_gravity = True

    src = open_source(args.path)
    src_fps = float(src.info.fps or 0.0)
    print(f"输入：{src.info.kind} {src.info.width}x{src.info.height} @ {src.info.fps:.1f} fps")
    pipe = SpherePipeline(cfg)
    if seed is not None:
        if hasattr(pipe.detector, "seed"):
            print(f"[种子] 球心 ({seed[0]:g},{seed[1]:g}) 半径 {seed[2]:g} px —— "
                  "之后每帧的球心由它跟出来，半径每帧从边缘重量")
            pipe.detector.seed(seed[0], seed[1], seed[2], frame_index=0)
        else:
            # 不静默忽略：用户以为脚本在跟那颗球，实际它在自己找 —— 这两件事
            # 长得一样，只有这句话能分开。
            print(f"[注意] 当前检测器 {cfg.detector.method} 不支持 --seed，已忽略；"
                  "本次结果是自动检测的，不是从种子跟出来的")
    wins = WindowManager()

    writer = None
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        # fps 可能是 -1（某些摄像头/异常文件），必须是正数才能写出可播放的视频
        out_fps = src.info.fps if (src.info.fps and src.info.fps > 1.0) else 30.0
        writer = cv2.VideoWriter(args.out, fourcc, out_fps, (src.info.width, src.info.height))
        if not writer.isOpened():
            print(f"[警告] 无法创建输出视频 {args.out}")
            writer = None

    frames_dir = None
    if args.frames_dir:
        frames_dir = ROOT / "outputs" / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    gate = RadiusGate(seed[2] if seed is not None else None)
    hits = 0          # 有效帧：过了半径一致性门的帧
    total = 0
    coast = 0
    no_detection = 0  # 检测器本帧没给候选
    radius_note_done = False   # "[半径]" 那行只打一次：288 行同样的警告等于没警告
    bad_frames: list[int] = []   # 没通过的帧号（按连续段统计丢失用）
    try:
        while True:
            ok, frame, t = src.read()
            if not ok or frame is None:
                break
            vis, result = pipe.process_and_render(frame, t)
            total += 1
            if result.track is not None and result.track.coasting:
                coast += 1

            # ---- 半径：不信检测器给的数，从边缘重新量一次 ----
            # specular 的 radius 是 CSRT 框的 max(w,h)/2：初始化时是 init_box/2，
            # 之后只随框缩放抖动。它回答"框多大"，而 Z=f·D/(2r) 要的是"球多大"。
            # 用框半径算深度，等于把深度锁死成一个常数 —— 那条"看起来很好"的
            # 轨迹其实是常数半径画出来的。
            accepted_r: float | None = None
            verdict = "no_detection"
            det = result.detection
            if det is not None:
                # pipeline 内部也算过灰度图，但那是它的局部变量；不碰
                # sphere_tracker/* 的前提下只能在这里再算一次（开销主要在 Sobel）
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # 判一次"这个半径有没有边缘证据"。只在无种子、且半径来自**检测框**
                # 时做：有种子时用户点的半径就是基准；color/hough 的半径本来就
                # 来自轮廓，不需要再问一遍"它是不是轮廓"。
                if not radius_note_done and seed is None and det.extra.get("box") is not None:
                    support, evidence = radius_evidence(gray, det.x, det.y, det.radius)
                    for line in radius_evidence_lines(det.radius, support, evidence):
                        print(line)
                    radius_note_done = True
                r_anchor = gate.r_prev if gate.r_prev is not None else float(det.radius)
                _rx, _ry, r_ref = refine_circle_subpixel(
                    gray, float(det.x), float(det.y), float(r_anchor),
                    n_rays=int(cfg.detector.subpixel_rays), search=REFINE_SEARCH_PX)
                accepted_r, verdict = gate.accept(float(r_ref))

            if accepted_r is not None:
                hits += 1
            else:
                bad_frames.append(result.frame_index)
                if det is None:
                    no_detection += 1

            if writer is not None:
                writer.write(vis)
            if frames_dir is not None:
                cv2.imwrite(str(frames_dir / f"frame_{total:06d}.png"), vis)
            row = {
                "frame": result.frame_index,
                "t": round(result.t, 5),
                "detected": int(result.detection is not None),
                "x": round(result.track.x, 3) if result.track else "",
                "y": round(result.track.y, 3) if result.track else "",
                "radius_px": round(result.track.radius, 3) if result.track else "",
                # radius_refined_px 才是"这一帧从边缘量到的半径"，其他列里的
                # radius_px 是跟踪器平滑后的框半径（口径不同，别混着用）
                "radius_refined_px": round(accepted_r, 3) if accepted_r is not None else "",
                "gate": verdict,
                "vx": round(result.track.vx, 3) if result.track else "",
                "vy": round(result.track.vy, 3) if result.track else "",
                "ax": round(result.track.ax, 3) if result.track else "",
                "ay": round(result.track.ay, 3) if result.track else "",
                "coasting": int(result.track.coasting) if result.track else "",
            }
            # 没通过门的帧**不进轨迹**（也就没有 X_m/Y_m/Z_m，也不进可视化文件）：
            # 一帧被拒，它的半径就是不可信的，而 Z=f·D/(2r) 会把这份不可信
            # 1:1 地带进深度。留着它等于把被拒的那一帧又偷偷塞回结果里。
            # 三维点必须用**过门的那个半径**重算。pipeline 的 position_3d 是用
            # 检测器给的半径算的 —— specular 下那就是 CSRT 框的 max(w,h)/2，
            # 它回答"框多大"，不是"球多大"。拿它算 Z 等于把深度锁成一个常数。
            #
            # 为什么不能"打印用细化的、落盘用框半径的"（2026-10-30 实测）：
            # 同一趟跑出两个深度 —— 报告说 91.2 px / Z 0.164 m，
            # 可视化文件里是 70.8 px / Z 0.212 m（差 29%）。一个产物里只允许
            # 有一个深度，否则下一个人会拿错的那个去做结论。
            if accepted_r is not None and pipe.camera is not None and result.track is not None:
                z = pipe.camera.depth_from_radius(
                    float(accepted_r), float(result.track.x), float(result.track.y))
                if np.isfinite(z) and 0.01 < z < 100.0:
                    p = pipe.camera.backproject(float(result.track.x), float(result.track.y), z)
                    row.update({"X_m": round(float(p[0]), 5), "Y_m": round(float(p[1]), 5),
                                "Z_m": round(float(p[2]), 5)})
            if result.prediction is not None and len(result.prediction):
                n = len(result.prediction) - 1
                row.update({
                    "pred_x_end": round(float(result.prediction[n][0]), 3),
                    "pred_y_end": round(float(result.prediction[n][1]), 3),
                })
            rows.append(row)
            if not args.no_show:
                wins.show(WINDOW, vis)
                key = cv2.waitKey(1) & 0xFF
                # 全屏键优先：Esc 在全屏时是"退出全屏"，不该顺手把程序关掉
                if wins.handle_key(WINDOW, key, vis):
                    continue
                if key in (ord("q"), 27):
                    break
            if args.max_frames and total >= args.max_frames:
                break
    finally:
        src.release()
        if writer is not None:
            writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()

    method = cfg.detector.method
    print(f"处理完成：{total} 帧，有效 {hits} 帧，丢失补偿 {coast} 帧，检测器 {method}"
          + (f"，人工种子 ({seed[0]:g},{seed[1]:g},{seed[2]:g})"
             if seed is not None else "，无种子（自动检测）"))
    for line in consistency_lines(
            total=total, accepted=hits, no_detection=no_detection,
            inconsistent=gate.inconsistent, jumps=gate.jumps,
            radii=gate.radii, jump_ratios=gate.jump_ratios, bad_frames=bad_frames,
            fps=src_fps, seed=seed):
        print(line)
    code, lines = hit_report(hits, total, method,
                             detector_error=getattr(pipe.detector, "last_error", ""))
    for line in lines:
        print(line)

    if code != 0:
        # 0 有效帧的产物**一律不写**：一张 x/y 全空的 CSV + 一段没有标注的视频
        # 正是用户报的那个坑（"看起来跑完了"）。写之前先想清楚这份文件
        # 会不会让人以为有结果 —— 会，就不写。
        if args.csv:
            print(f"[跳过] 没有一帧通过一致性门，不写 CSV：{args.csv}"
                  "（写了也是一张空表，只会让人以为有结果）")
        if args.vis_json:
            print(f"[跳过] 没有一帧通过一致性门，不写三维可视化：{args.vis_json}"
                  "（没有三维点，写了也打不开）")
        if args.out and writer is not None and Path(args.out).exists():
            try:
                Path(args.out).unlink()
                print(f"[跳过] 标注视频已删除：{args.out}"
                      "（一帧有效帧都没有，它就是原视频的拷贝，留着等于包装了一个假结果）")
            except OSError as exc:
                print(f"[警告] 标注视频没删掉（{exc}），但它是没有标注的：{args.out}")
        return code

    if args.csv and rows:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        # 各帧的字段不一定一样（比如有 3D/预测结果时才会多出几列），
        # 所以列名要取所有行的并集，否则 DictWriter 会报 "fields not in fieldnames"
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, restval="")
            w.writeheader()
            w.writerows(rows)
        print(f"轨迹已导出：{args.csv}")
    if args.out:
        print(f"标注视频已保存：{args.out}")

    if args.vis_json:
        # 球径与 fx 从**实际生效的**配置取（不是字面量）：它们直接乘进 Z，
        # 文件里记的尺度必须就是算出来那个尺度。
        diameter_mm = float(cfg.camera.ball_diameter_m) * 1000.0
        fx = float(pipe.camera.fx) if pipe.camera is not None else float(cfg.camera.fx or 0.0)
        vis = build_vis(rows, session=session_name(args.path),
                        source=Path(args.path).name,
                        ball_diameter_mm=diameter_mm, fx=fx,
                        seed=seed, accepted_frames=hits,
                        rejected_frames=max(0, total - hits))
        if vis is None:
            print(f"[跳过] 没有三维点（position_3d 全为空），不写 {args.vis_json}；"
                  "空文件比没有文件更糟 —— 它会让查看器以为这个会话有结果。")
        else:
            out = Path(args.vis_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(vis, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
            zs = [p["xyz"][2] for p in vis["trajectory"]]
            print(f"三维可视化已导出：{out}"
                  f"（{vis['n_points']} 点，Z {min(zs):.3f}~{max(zs):.3f} m，"
                  f"D={vis['ball_diameter_mm']:g} mm，fx={vis['fx']:g} px）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
