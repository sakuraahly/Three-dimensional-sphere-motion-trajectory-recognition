"""离线处理视频/图片：标注轨迹 + 预测 + 导出 CSV。

用法::

    python apps/run_video.py assets/videos/ball.mp4
    python apps/run_video.py assets/videos/ball.mp4 --out outputs/videos/annotated.mp4 --csv outputs/traj.csv
    python apps/run_video.py assets/images/ --frames-dir          # 整个文件夹当序列
"""
from __future__ import annotations

import argparse
import sys
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
from sphere_tracker.plugins import default as default_method
from sphere_tracker.plugins import describe, names as method_names
from sphere_tracker.sources import open_source
from apps._window import WindowManager

WINDOW = "run_video"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="离线视频/图片序列的球体轨迹识别")
    p.add_argument("path", help="视频文件、图片文件或图片文件夹")
    p.add_argument("--out", default=None, help="输出标注视频（.mp4）")
    p.add_argument("--csv", default=None, help="导出轨迹 CSV")
    p.add_argument("--frames-dir", action="store_true", help="输入是图片文件夹时，输出逐帧 PNG")
    p.add_argument("--config", default=None)
    p.add_argument("--method", default=None, choices=method_names("detector"),
                   help="检测方法（不传则取配置，默认 %s）。可选：%s" % (
                       default_method("detector"),
                       "；".join(f"{d['name']}={d['summary']}"
                                 for d in describe("detector"))))
    p.add_argument("--preset", default=None)
    p.add_argument("--ball-diameter", type=float, default=None)
    p.add_argument("--focal", type=float, default=None)
    p.add_argument("--predict", default=None, choices=["kalman", "quadratic"])
    p.add_argument("--horizon", type=float, default=None)
    p.add_argument("--gravity-fit", action="store_true")
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
    if args.focal:
        cfg.camera.fx = args.focal
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

    print(f"处理完成：{total} 帧，检出 {hits} 帧（{100.0 * hits / max(1, total):.1f}%），丢失补偿 {coast} 帧")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
