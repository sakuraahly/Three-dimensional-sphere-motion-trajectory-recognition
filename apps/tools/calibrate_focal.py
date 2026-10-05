"""标定像素焦距 fx（三维测距必须要）。

原理：球在图像上的像素半径 r 与球心距离 Z 满足 r = f * R / Z。
把球放在一个量好的距离上（比如 Z = 1.000 m），量出像素半径，
就能解出 f。这个"单点标定"对测距来说够用了。

用法::

    # 用实时画面：把球放在尺子量好的 1 米处，等检测框稳定后按 s
    python apps/tools/calibrate_focal.py --camera 0 --distance 1.0 --ball-diameter 0.04

    # 用一张照片
    python apps/tools/calibrate_focal.py --image assets/images/ball_at_1m.png --distance 1.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# --- 让脚本在任意目录深度都能找到项目根 ---
ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time

import cv2
import numpy as np

from sphere_tracker.config import AppConfig, ColorConfig, load_config, save_config
from sphere_tracker.detect import build_detector
from sphere_tracker.geometry import estimate_focal_from_known_ball
from sphere_tracker.sources import open_source


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="单点焦距标定")
    p.add_argument("--camera", type=int, default=None)
    p.add_argument("--image", default=None)
    p.add_argument("--video", default=None)
    p.add_argument("--distance", type=float, required=True, help="球心到相机的实际距离（米）")
    p.add_argument("--ball-diameter", type=float, default=0.04, help="球直径（米）")
    p.add_argument("--method", default="fusion")
    p.add_argument("--preset", default="white_ball")
    p.add_argument("--config", default=None)
    p.add_argument("--save", default="config/sphere_config.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config) if args.config else AppConfig()
    cfg.detector.method = args.method
    cfg.detector.color = ColorConfig.preset(args.preset)
    cfg.camera.ball_diameter_m = args.ball_diameter
    detector = build_detector(cfg.detector)

    src = open_source(args.image or args.video, camera=args.camera)
    WINDOW = "calibrate focal"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    radii: list[float] = []
    print("把球放在量好的距离上，画面里能看到绿圈后按 s 保存；按 q 退出。")

    while True:
        ok, frame, _ = src.read()
        if not ok or frame is None:
            if src.is_live:
                time.sleep(0.01)
                continue
            break
        found = detector.detect(frame)
        vis = frame.copy()
        if found:
            det = found[0]
            radii.append(det.radius)
            if len(radii) > 30:
                radii.pop(0)
            cv2.circle(vis, (int(det.x), int(det.y)), int(det.radius), (0, 255, 0), 2, cv2.LINE_AA)
            med = float(np.median(radii))
            f = estimate_focal_from_known_ball(med, args.distance, args.ball_diameter)
            cv2.putText(
                vis,
                f"r={det.radius:.2f}px  median={med:.2f}px  ->  fx={f:.1f}px   [s]=save",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
            )
        else:
            cv2.putText(vis, "no ball detected - adjust --preset / --method",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow(WINDOW, vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s") and radii:
            med = float(np.median(radii))
            cfg.camera.fx = estimate_focal_from_known_ball(med, args.distance, args.ball_diameter)
            cfg.camera.fy = cfg.camera.fx
            cfg.camera.cx = frame.shape[1] / 2.0
            cfg.camera.cy = frame.shape[0] / 2.0
            p = save_config(cfg, args.save)
            print(f"像素半径中位数 {med:.2f} px  ->  fx = {cfg.camera.fx:.1f} px，已保存到 {p}")
    src.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
