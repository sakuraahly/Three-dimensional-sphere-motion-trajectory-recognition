"""离线处理视频/图片：标注轨迹 + 预测 + 导出 CSV + 单目三维可视化。

用法::

    python apps/run_video.py assets/videos/ball.mp4
    python apps/run_video.py assets/videos/ball.mp4 --out outputs/videos/annotated.mp4 --csv outputs/traj.csv
    python apps/run_video.py assets/images/ --frames-dir          # 整个文件夹当序列
    python apps/run_video.py clip.mkv --method hough --vis-json outputs/vis.json

## "0 检出"的处置史：从"不写产物"到"写产物 + 显眼降级标记"

**2026-10-29**：一段 288 帧的单目录像，默认 ``color`` 检测器 40 帧里检出
**0 帧**，而进程照样 exit 0、照样写出 CSV 与"标注视频"。用户看到的是
"跑完了"，拿到手的是一个 x/y 全空的表格。当时的对策是把"什么都没测到"
和"测到了一点"分成两种结局，``hits == 0`` 就**不写**三份产物、退出码 2 ——
理由一句话：**本项目最贵的一类 bug 是"看起来成功"**。

**2026-11-14（新原则）**：用户要求"质量差也要能出结果，而不是自己中止程序，
只有错误才是紧急的"，举的例子正是"之前的 3D 查看就不该因为质量差而不生成"。
所以这里把**防假结果的手段换掉、不是删掉**：

- ``hits == 0``（0 检出、但帧读到了）→ 退出码 **0**（不是错误、不中止），
  三份产物**照常写**，但每份都带显眼的降级语义：
    · visualization.json 顶层 ``quality``（``degraded=True`` /
      ``grade="degraded"`` / ``verdict="zero_hit"`` / ``degraded_reason`` /
      ``warnings``），查看器按降级视图打开；
    · CSV 是 x/y 全空的空表、标注视频没有任何标注 —— 每写一份都在 stdout
      用 ``[降级]`` 醒目说明"结果照常输出，但这段没有可用的三维点/检出数为 0"；
    · ``hit_report`` 打印诊断（当前检测器 + 该换哪个），**不静默**。
- ``0 < hits/total < 10%`` → 一行警告，但仍 exit 0：确实有数据，
  不该假装失败，可这么低的比率不足以当测量用。
- ``total == 0``（一帧都没读到）/ 没给球径 → **退出码 2**，不写产物：
  这两类是参数/文件的真错误，不是"质量差"。

让"有结果"和"这结果空"**同时成立**，而不是把文件藏起来让人以为没跑。

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
import os
import sys
import time
from pathlib import Path

# --- 让脚本在任意目录深度都能找到项目根 ---
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "sphere_tracker").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import csv

import cv2
import numpy as np

from sphere_tracker.config import AppConfig, ColorConfig, TrackerConfig, load_config
from sphere_tracker.detectors import registry  # noqa: F401  （import 即注册：R1）
from sphere_tracker.detectors.neural import mark_conf_explicit
from sphere_tracker.geometry import refine_circle_subpixel
from sphere_tracker.pipeline import SpherePipeline
from sphere_tracker.sources import open_source
# 帧层半径一致性门在**库层**（`sphere_tracker.track`），门限单点在 `TrackerConfig`。
# 这里不再自带一份实现 —— 两份实现必然漂移。
from sphere_tracker.track import RadiusGate
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

#: 帧层半径门的五个门限（跳变护栏 / 半径带 / 窗口）现在**单点**在 `TrackerConfig`：
#: `radius_max_jump` / `radius_band` / `radius_window` / `radius_seed_band_lo` /
#: `radius_seed_band_hi`；门本身是 `sphere_tracker.track.RadiusGate`。
#: （此前它们是散在这里的模块常量，2026-10-31 S1d′ 归位，值不变。）

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

    - 一帧都没读到（``total == 0``）→ 2 + 诊断：这是文件/参数的**真错误**，
      退出码必须非 0，也不写任何产物。
    - 0 有效帧但帧读到了（``hits == 0``）→ **0 + 降级诊断**：质量差不是错误，
      照常出结果（产物都带 degraded 标记，见模块 docstring），但绝不静默。
    - 低有效帧 → 0 + 一行警告。有数据就不该假装失败，但也不能装作够用。

    ``code`` 只有 ``main()`` 消费：``code != 0`` 现在**只剩真错误**那一条路。
    """
    rate = hits / total if total else 0.0
    if total == 0:
        return 2, [
            "[失败] 一帧都没读到（视频打不开 / 文件是空的 / 路径给错了）。",
            "       退出码 2 —— 不要把这个当成「跑完了」。",
        ]
    if hits == 0:
        lines = [
            f"[降级] {total} 帧里没有一帧通过一致性门 —— 这段没有可用的三维点"
            "（检出数为 0），结果为空，请勿当作有效测量用。",
            "       结果**照常输出**（CSV / 三维可视化 / 标注视频都写），"
            "但每份都带显眼的降级语义，让「有结果」和「这结果空」同时成立：",
            "       · 三维可视化顶层 quality.degraded=True / grade=degraded / "
            "verdict=zero_hit，查看器按降级视图打开；",
            "       · CSV 是 x/y 全空的空表、标注视频没有任何标注；"
            "下面每写一份都会再打一行 [降级]。",
            "       退出码 0：质量差不是错误、不中止程序；但「看起来成功」仍是"
            "本项目最贵的 bug —— 现在用显眼标记 + 醒目日志把空结果说清，"
            "而不是靠不写文件把它藏起来。",
            f"       当前检测器：{method}",
        ]
        if detector_error:
            lines.append(f"       检测器自述原因：{detector_error}")
        lines.append("       换个检测器再跑（按可能性排序，命令行加 --method <名字>）：")
        lines += [f"         --method {name:<9} {why}" for name, why in METHOD_HINTS]
        lines.append("       命令里补上「就这颗球」的信息（像素坐标与半径）：")
        lines.append("         --seed x,y,r    点选球心：镜面球（specular）用它跟住；"
                     "纯色球（--method color）用它取色并锚定半径")
        lines.append("       不知道球色时先试 --preset auto（按画面自动估主色），"
                     "不用点选也能起步。")
        return 0, lines
    if rate < LOW_HIT_RATE:
        return 0, [
            f"[警告] 有效帧只有 {rate * 100:.1f}%（{hits}/{total} 帧），"
            f"结果多半不可用；试试 --method hough / fusion，"
            f"球色/预设不确定时加 --preset auto，金属球加 --seed",
            "       仍然按退出码 0 结束：确实有数据，不该假装失败。"
            "但这么低的比率不足以当测量用。",
        ]
    return 0, []


def consistency_lines(*, total: int, accepted: int, no_detection: int,
                      inconsistent: int, jumps: int, radii: list[float],
                      jump_ratios: list[float], bad_frames: list[int],
                      fps: float,
                      seed: tuple[float, float, float] | None,
                      tracker: TrackerConfig | None = None) -> list[str]:
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
        # 带的下/上界必须与 `RadiusGate` 实际用的**同一份**门限，否则这行报告
        # 和门的行为会各说各话。门限单点在 `TrackerConfig`，两者都读它。
        tcfg = tracker if tracker is not None else TrackerConfig()
        lo = seed[2] * (1.0 - tcfg.radius_seed_band_lo)
        hi = seed[2] * (1.0 + tcfg.radius_seed_band_hi)
        lines.append(f"[尺度] 人工种子 {seed[2]:g} px 锚定：接受半径 "
                     f"{lo:.1f}~{hi:.1f} px（−{tcfg.radius_seed_band_lo:.0%} / "
                     f"+{tcfg.radius_seed_band_hi:.0%}）")
    return lines


def build_vis(rows: list[dict], *, session: str, source: str,
              ball_diameter_mm: float, fx: float,
              created: str | None = None,
              seed: tuple[float, float, float] | None = None,
              accepted_frames: int | None = None,
              rejected_frames: int | None = None) -> dict:
    """把逐帧结果整理成单目可视化文件；**0 个三维点也返回完整 payload**。

    2026-11-14 新原则：0 检出时 ``--vis-json`` 照常写文件（以前这里返回
    ``None``，等于"因为质量差而不生成"，正是用户举的 3D 查看那个坑）。
    点数为 0 时 ``trajectory`` 留空，但顶层挂 ``quality.degraded=True`` 的
    同构对象 + ``warnings``，让查看器按降级视图打开 —— 防假结果靠显眼标记，
    不靠不写文件。字段与双目 ``traj3d.json`` 的 ``payload["quality"]`` 同构：
    ``verdict`` / ``grade`` / ``degraded`` / ``degraded_reason`` / ``warnings`` / ``note``。

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
    # 0 个三维点**不再 return None**：照常构造完整 payload，只把 trajectory
    # 留空并挂上顶层 quality.degraded 标记。这样 --vis-json 一定能写出文件、
    # 查看器能按降级视图打开；「防假结果」靠标记而非不写文件（见模块 docstring）。
    payload = {
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
    if pts:
        payload["quality"] = {
            "verdict": "ok",
            "grade": "ok",
            "degraded": False,
            "degraded_reason": "",
            "warnings": [],
            "note": "",
        }
    else:
        reason = (f"0 个三维点：{len(rows)} 行里没有一行有 position_3d —— "
                  "这段没有可用的三维位置，检出数为 0")
        payload["quality"] = {
            "verdict": "zero_hit",
            "grade": "degraded",
            "degraded": True,
            "degraded_reason": reason,
            "warnings": [
                reason,
                "结果照常输出，但 trajectory 为空：请勿当作有效测量使用",
                "换个检测器 / 补 --seed x,y,r / --preset auto 后重跑",
            ],
            "note": ("与双目 traj3d.json 顶层 quality 同构；查看器识别 "
                     "degraded=True 后按降级视图打开（空场景 + 原因）"),
        }
    return payload


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
             "没显式给 --method 时自动切到 specular（镜面球语义）；"
             "显式给 --method color 时，种子用于从点击处取 HSV 并锚定半径")
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


def _consume_session_camera(video_path) -> None:
    """B-124：单目会话边车记的相机身份**必须被消费**，不能是装饰字段。

    record_mono 把相机角色/索引/设备身份写进 ``<clip>.json``；_mono_sessions
    又把角色折进 ``session.json`` 的 ``camera``。旧版 run_video 一个字都不读 ——
    标定 fx 来自「当时叫 left」的那台相机，事后换了相机根本查不出来。
    这里做**只读消费**：把会话记的角色与 config/cameras.json 当前绑定对一次，
    对不上就给可执行提示；不拦截检测（离线视频不依赖相机在线）。
    """
    import json
    from pathlib import Path
    sess = Path(video_path).resolve().parent / "session.json"
    if not sess.is_file():
        return
    try:
        raw = json.loads(sess.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(raw, dict):
        return
    role = str(raw.get("camera") or "").strip()
    if not role:
        print("[会话相机] session.json 没记相机身份（camera 为空）；"
              "这段若做过单目标定，fx 是否对应同一台相机无法核对。", flush=True)
        return
    try:
        from sphere_tracker import camera_registry as cr
        known = role in cr.load_registry().roles
    except Exception:                                       # noqa: BLE001
        return                      # 读不到注册表：不猜、不拦
    if not role.startswith("#") and not known:
        print(f"[会话相机] 这段录在角色 {role!r} 上，但 config/cameras.json 现在"
              "没有这个角色 —— 标定 fx 可能来自另一台相机。先跑 "
              "python apps/tools/cameras.py identify 重新 pin 再取 Z。", flush=True)
    else:
        print(f"[会话相机] {role!r} 与当前配置对得上（身份已消费）；"
              "单目 Z 依赖的 fx 须仍对应这台相机。", flush=True)


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
    # 种子的语义是"我指的就是这颗球，跟住它"。哪些检测器**接受**种子：
    # specular 用它初始化 CSRT，color 用它取色 + 锚定半径（2026-11-06）。
    #
    # 判据必须是"用户有没有显式给 --method"，不能用"值等于默认值"：
    # `add_mono_pipeline_args` 里 default=None，所以 `args.method is None`
    # 才是"什么也没说"。旧版只要给了 --seed 就切 specular —— 那对金属球是对的，
    # 对纯色塑料球是反的（它能被 color 稳定检出，切成 specular 反而丢掉轮廓精度）。
    if seed is not None:
        explicit = args.method is not None
        current = str(cfg.detector.method or "")
        if not explicit and current != "specular":
            print(f"[说明] 给了 --seed 且没显式 --method，检测器从 {current} 切到 specular："
                  "没点选信息时只有 specular 能可靠地跟住人工指定的那颗球"
                  "（镜面球语义）。纯色球请显式 --method color，种子会用于取色")
            cfg.detector.method = "specular"
        elif explicit and current not in ("specular", "color"):
            print(f"[注意] {current} 不接受种子；--seed 本次会被忽略。"
                  "要跟住这颗球请用 --method color（纯色球）或 specular（镜面球）")
    if args.preset:
        cfg.detector.color = ColorConfig.preset(args.preset)
    if args.ball_diameter:
        cfg.camera.ball_diameter_m = args.ball_diameter
    # 球径**没有默认值**：课题的测试球从几毫米到几十毫米都有，而单目测深
    # `Z = f·D/(2r)` 对 D 是 1:1 的 —— 替你猜一个数，就是把"我不知道球多大"
    # 悄悄变成"深度偏了 N%"（`CameraModel` 的注释里记着历史上三份默认值打架的事）。
    # 所以这里**停在启动前**，并把"怎么给"和"会给成什么单位"一次说清。
    if cfg.camera.ball_diameter_m is None:
        print("[错误] 没有球径，单目无法估深：Z = f·D/(2r)，D 错 10% 深度就错 10%。\n"
              "       用 --ball-diameter 给**米**（如 25 mm 的球写 0.025），\n"
              "       或先量好再跑；本项目不再提供任何默认球径。", flush=True)
        return 2
    # 内参：标定文件（含畸变）优先，--focal 再覆盖；**来源每次都打印**
    print(f"[{apply_mono_calibration(cfg, args)}]")
    if args.predict:
        cfg.predict.method = args.predict
    if args.horizon:
        cfg.predict.horizon_s = args.horizon
    if args.gravity_fit:
        cfg.predict.use_gravity = True

    # B-124：启动即只读消费会话边车记的相机身份（不是装饰字段）
    _consume_session_camera(args.path)
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
    out_partial: Path | None = None
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        # fps 可能是 -1（某些摄像头/异常文件），必须是正数才能写出可播放的视频
        out_fps = src.info.fps if (src.info.fps and src.info.fps > 1.0) else 30.0
        # 写临时文件，收尾成功后再原子改名成正式名：mp4v 的 moov atom 只在
        # release() 时落盘，进程被强杀（任务管理器 3 秒优雅不退就 taskkill /F）
        # 会留下一个"存在但打不开"的 annotated.mp4（moov atom not found），
        # 查看器用 .exists() 判它存在、cv2 却打不开，表现为"视频黑屏/不显示"。
        # 写 .tmp 后，中断只留 .tmp 垃圾，正式名永远是"完整"或"不存在"。
        # 注意临时名**结尾必须仍是 .mp4**：cv2.VideoWriter 靠后缀认容器格式，
        # 后缀改成 .partial 会让 writer.isOpened() 返回 False（打不开）。
        out_partial = Path(args.out).with_name(
            Path(args.out).stem + ".tmp" + Path(args.out).suffix)
        writer = cv2.VideoWriter(str(out_partial), fourcc, out_fps,
                                 (src.info.width, src.info.height))
        if not writer.isOpened():
            print(f"[警告] 无法创建输出视频 {out_partial}")
            writer = None

    frames_dir = None
    if args.frames_dir:
        frames_dir = ROOT / "outputs" / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    # 门限单点在 cfg.tracker（`TrackerConfig` 的帧层五个字段），门本身在库层。
    gate = RadiusGate(seed[2] if seed is not None else None, cfg=cfg.tracker)
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
            # 先 `process`（检测 + 跟踪 + 默认按检测器半径算一次三维），等门判完、
            # 深度定死之后再 `render` —— 这样视频 HUD 读到的 `position_3d` 就是
            # 下面"唯一一处"按过门半径重算的那个值（否则 HUD 会报框半径的深度、
            # CSV 报细化半径的深度，同一趟两个深度）。
            result = pipe.process(frame, t)
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
            # ---- 三维点：一个产物里只允许有一个深度 ----
            # 唯一的"哪个半径进 Z"的决定点就是这里：过门的细化半径 accepted_r。
            # 它**同时**决定 CSV 行（X_m/Y_m/Z_m）与 `result.position_3d`
            # （视频 HUD / 3D 面板读的就是它），两者必然是同一个数。
            #
            # 为什么过门的半径才是权威：帧层门（`RadiusGate`）的 accepted_r 是
            # "这一帧真的从边缘量到、且过门"的半径；而 pipeline 默认用的
            # `best.radius` 在 specular 下是 CSRT 框的 `max(w,h)/2`，回答"框多大"、
            # 不是"球多大"（拿它算 Z 等于把深度锁成一个常数）。
            #
            # 为什么不能"打印用细化的、落盘用框半径的"（2026-10-30 实测）：
            # 同一趟跑出两个深度 —— 报告 91.2 px / Z 0.164 m，可视化文件里是
            # 70.8 px / Z 0.212 m（差 29%）。一个产物里只允许有一个深度，
            # 否则下一个人会拿错的那个去做结论。
            #
            # "半径 → 深度"的算法只此一处（`pipe.apply_depth`），不在 app 里另写。
            # 没过门的帧**没有可信深度**：连 `position_3d` 一起清掉 —— 别让深度
            # 只从 CSV 里消失、却还挂在视频画面上（那正是"两个深度"的来源）。
            if accepted_r is not None and pipe.camera is not None and result.track is not None:
                if not pipe.apply_depth(result, float(accepted_r),
                                        float(result.track.x), float(result.track.y)):
                    result.position_3d = None
            else:
                result.position_3d = None
            if result.position_3d is not None:
                # 行与 HUD 取**同一个**三元组（CSV 的 5 位精度就是画面上显示的数）
                p = np.round(np.asarray(result.position_3d, dtype=np.float64), 5)
                row.update({"X_m": float(p[0]), "Y_m": float(p[1]), "Z_m": float(p[2])})
                result.position_3d = p
            if result.prediction is not None and len(result.prediction):
                n = len(result.prediction) - 1
                row.update({
                    "pred_x_end": round(float(result.prediction[n][0]), 3),
                    "pred_y_end": round(float(result.prediction[n][1]), 3),
                })
            rows.append(row)

            # 深度已定死，现在才渲染（HUD 与 CSV 用同一个 depth）。
            vis = pipe.render(frame, result)
            if writer is not None:
                writer.write(vis)
            if frames_dir is not None:
                cv2.imwrite(str(frames_dir / f"frame_{total:06d}.png"), vis)
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
            fps=src_fps, seed=seed, tracker=cfg.tracker):
        print(line)
    code, lines = hit_report(hits, total, method,
                             detector_error=getattr(pipe.detector, "last_error", ""))
    for line in lines:
        print(line)

    # 0 有效帧（hits==0）按新原则判 code=0，照常走下面的产物写出；这里用
    # degraded 把"结果为空"贯穿到 CSV / 视频 / vis-json 三处的醒目日志。
    # code != 0 现在只可能是 total==0 的真错误，那条路在下面提前 return、
    # 不写产物也清掉临时文件。
    degraded = hits == 0 and code == 0

    if code != 0:
        # 这里现在**只剩真错误**：total == 0（一帧都没读到）。hits == 0（0 检出）
        # 已按新原则改判 code=0，照常走下面的产物写出（带 degraded 标记）——
        # 上面那段「0 有效帧的产物一律不写」的判断没有删，而是换成了
        # 「照常写，但每份带显眼降级标记 + stdout 醒目日志」，见模块 docstring。
        # 真错误不写任何产物，也清掉临时文件：绝不留一个"存在但打不开"的
        # .tmp.mp4 冒充正式视频（这是 .tmp + os.replace 原子改名的另一半语义）。
        if args.csv:
            print(f"[错误] 一帧都没读到，不写 CSV：{args.csv}")
        if args.vis_json:
            print(f"[错误] 一帧都没读到，不写三维可视化：{args.vis_json}")
        if out_partial is not None and out_partial.exists():
            try:
                out_partial.unlink()
                print(f"[清理] 真错误，临时视频已删：{out_partial}")
            except OSError as exc:
                print(f"[警告] 临时视频没删掉（{exc}）：{out_partial}")
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
        if degraded:
            # 不加列（表头有硬校验）：空结果这件事靠 stdout 醒目日志 +
            # 同趟的 visualization.json 顶层 quality.degraded 来标记。
            print(f"[降级] 这份 CSV 是空结果（{total} 帧里 0 帧过门，x/y 全空）—— "
                  "结果照常输出，但请勿当作有效测量用；"
                  "对应的 visualization.json 里带 quality.degraded=True。")
    if args.out:
        # release 已把 moov atom 写进 .tmp.mp4，现在原子改名成正式名。这一步
        # 只会在"跑完（含 0 有效帧的降级结果）"后发生，所以 annotated.mp4
        # 要么完整、要么不存在，不会再出现"存在但 moov 缺失"的坏文件。
        # 真错误（code!=0）在上面已提前 return 并清掉临时文件，不会走到这里。
        if out_partial is not None and out_partial.exists():
            os.replace(out_partial, args.out)
        print(f"标注视频已保存：{args.out}")
        if degraded:
            print("[降级] 这段标注视频没有任何检测/三维标注，是空结果 —— "
                  "文件照常输出（原子改名已发生），但请勿当作有效测量用。")

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
        # build_vis 现在永不返回 None：0 个三维点也返回完整 payload（带
        # quality.degraded）。写文件这一步无条件发生 —— 这是"质量差也出结果"
        # 的落点；下面按 quality 分开打印，避免对空 trajectory 求 min/max。
        out = Path(args.vis_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(vis, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        quality = vis["quality"]
        if quality["degraded"]:
            print(f"[降级] 三维可视化已导出但结果为空：{out}"
                  f"（{quality['degraded_reason']}）—— "
                  "文件顶层带 quality.degraded=True，查看器按降级视图打开，"
                  "请勿当作有效测量用。")
            for warn in quality["warnings"]:
                print(f"  · {warn}")
        else:
            zs = [p["xyz"][2] for p in vis["trajectory"]]
            print(f"三维可视化已导出：{out}"
                  f"（{vis['n_points']} 点，Z {min(zs):.3f}~{max(zs):.3f} m，"
                  f"D={vis['ball_diameter_mm']:g} mm，fx={vis['fx']:g} px）")
    if degraded:
        # 与双目 run_stereo_video.py / run_all.py 的 quality_verdict 同一行格式：
        # 固定前缀 [质量] + 键 verdict=<非 ok>，GUI/CLI 侧自动识别并显示状态，
        # 无需各自改解析（run_all.py 由别的 worker 维护，这里不碰）。
        # zero_hit = 单目 0 检出专用值，与 vis-json 顶层 quality.verdict 同值。
        print('[质量] verdict=zero_hit 0 检出 / 没有可用的三维点：结果照常输出，'
              '但为空，结果仅供参考（详见 visualization.json["quality"]）')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
