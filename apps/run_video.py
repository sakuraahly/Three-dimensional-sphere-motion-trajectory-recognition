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
"""
from __future__ import annotations

import argparse
import json
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

from sphere_tracker.config import AppConfig, ColorConfig, load_config
from sphere_tracker.detectors import registry  # noqa: F401  （import 即注册：R1）
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

#: 0 检出时的换法建议：顺序 = 最可能有效的排在前面。
#: 每条都带"什么情况下该用它"，因为用户此刻唯一的信息就是"球在画面里"。
METHOD_HINTS = (
    ("hough", "球颜色不突出、与背景接近（几何线索；颜色不可靠时的第一选择）"),
    ("fusion", "有同色干扰物、或单一线索不稳（多线索投票，最稳但最慢）"),
    ("specular", "球是金属/高光（颜色阈值抓不到高光球，它量的是高光斑）"),
    ("motion", "球在动、背景基本静止（背景差分；球与背景同色时唯一的线索）"),
)


def session_name(path: str | Path) -> str:
    """可视化文件里记的 ``session``：视频所在目录名，取不到就用文件名。

    为什么不用文件名的 stem：会话的语义单位是**目录**（一次采集），
    而查看器是"一个会话一条轨迹"。用文件名会得到 ``clip`` 这种毫无信息量的名字。
    """
    p = Path(path)
    return (p if p.is_dir() else p.parent).name or p.name


def hit_report(hits: int, total: int, method: str) -> tuple[int, list[str]]:
    """按检出情况给出 (退出码, 要打印的行)。抽成纯函数是为了能被测。

    - 0 检出 → 2 + 诊断。**必须是非 0**：退出码是脚本唯一的"结论"，
      0 意味着"成功"，而这里恰恰什么都没测到。
    - 低检出 → 0 + 一行警告。有数据就不该假装失败，但也不能装作够用。
    """
    rate = hits / total if total else 0.0
    if total == 0:
        return 2, [
            "[失败] 一帧都没读到（视频打不开 / 文件是空的 / 路径给错了）。",
            "       退出码 2 —— 不要把这个当成「跑完了」。",
        ]
    if hits == 0:
        lines = [
            f"[失败] {total} 帧里一帧都没找到球 —— 这次什么都没测到，结果不可用"
            f"（CSV 的 x/y 会全是空值）。",
            f"       当前检测器：{method}",
            "       退出码 2（不是 0）：本项目最贵的一类 bug 是「看起来成功」——",
            "       「跑完了」和「什么都没测到」在屏幕上长得一样，只能靠退出码分开。",
            "       换个检测器再跑（按可能性排序，命令行加 --method <名字>）：",
        ]
        lines += [f"         --method {name:<9} {why}" for name, why in METHOD_HINTS]
        lines.append("       注意：金属球/高光球优先 specular；颜色不突出时优先 hough/fusion。")
        return 2, lines
    if rate < LOW_HIT_RATE:
        return 0, [
            f"[警告] 检出率只有 {rate * 100:.1f}%（{hits}/{total} 帧），"
            f"结果多半不可用；试试 --method hough / fusion",
            "       仍然按退出码 0 结束：确实有数据，不该假装失败。"
            "但这么低的比率不足以当测量用。",
        ]
    return 0, []


def build_vis(rows: list[dict], *, session: str, source: str,
              ball_diameter_mm: float, fx: float,
              created: str | None = None) -> dict | None:
    """把逐帧结果整理成单目可视化文件；没有三维点就返回 ``None``（不写空文件）。

    单目**本来就有三维**：``Z = f·D/(2r)``（``CameraModel.depth_from_radius``），
    只是以前从来没落盘 —— 于是"主界面只能看双目"看起来像"单目没有 3D"。

    两个量从**实际生效的配置**取，不写字面量：
    ``ball_diameter_mm`` 与 ``fx`` 都是 Z 的直接输入，写死一份等于让文件里
    记的尺度和真正算出来的尺度成为两个数。
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
    p.add_argument("--max-frames", type=int, default=0, help=">0 时只处理前 N 帧")
    p.add_argument("--no-show", action="store_true", help="不弹窗（批处理用）")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config) if args.config else AppConfig()
    if args.method:
        cfg.detector.method = args.method
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
    print(f"输入：{src.info.kind} {src.info.width}x{src.info.height} @ {src.info.fps:.1f} fps")
    pipe = SpherePipeline(cfg)
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
    hits = 0
    total = 0
    coast = 0
    try:
        while True:
            ok, frame, t = src.read()
            if not ok or frame is None:
                break
            vis, result = pipe.process_and_render(frame, t)
            total += 1
            if result.track is not None and not result.track.coasting:
                hits += 1
            if result.track is not None and result.track.coasting:
                coast += 1
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
                "vx": round(result.track.vx, 3) if result.track else "",
                "vy": round(result.track.vy, 3) if result.track else "",
                "ax": round(result.track.ax, 3) if result.track else "",
                "ay": round(result.track.ay, 3) if result.track else "",
                "coasting": int(result.track.coasting) if result.track else "",
            }
            if result.position_3d is not None:
                p = result.position_3d
                row.update({"X_m": round(float(p[0]), 5), "Y_m": round(float(p[1]), 5), "Z_m": round(float(p[2]), 5)})
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
    print(f"处理完成：{total} 帧，检出 {hits} 帧（{100.0 * hits / max(1, total):.1f}%），"
          f"丢失补偿 {coast} 帧，检测器 {method}")
    code, lines = hit_report(hits, total, method)
    for line in lines:
        print(line)

    if code != 0:
        # 0 检出的产物**一律不写**：一张 x/y 全空的 CSV + 一段没有标注的视频
        # 正是用户报的那个坑（"看起来跑完了"）。写之前先想清楚这份文件
        # 会不会让人以为有结果 —— 会，就不写。
        if args.csv:
            print(f"[跳过] 没有检出任何一帧，不写 CSV：{args.csv}"
                  "（写了也是一张空表，只会让人以为有结果）")
        if args.vis_json:
            print(f"[跳过] 没有检出任何一帧，不写三维可视化：{args.vis_json}"
                  "（没有三维点，写了也打不开）")
        if args.out and writer is not None and Path(args.out).exists():
            try:
                Path(args.out).unlink()
                print(f"[跳过] 标注视频已删除：{args.out}"
                      "（一帧没检出，它就是原视频的拷贝，留着等于包装了一个假结果）")
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
                        ball_diameter_mm=diameter_mm, fx=fx)
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
