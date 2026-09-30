"""复检双目标定：装置有没有被碰动？

标定一次可以用很久 —— 前提是**两台相机之间的相对位置没变**。
云台转动、支架被撞、相机被取下再装回，都会让 R,T 失效，
但表面上看不出任何异常（检测照样出结果，三维却系统性偏）。

这个工具用同一块标定板拍几帧，重新算**极线误差**并与标定时记录的值比较：

    python apps/verify_stereo.py --calib config/stereo_calibration --left 0 --right 1
    python apps/verify_stereo.py --calib config/stereo_calibration \
        --left-dir check/left --right-dir check/right

判定：极线误差仍在阈值内且与标定时的差异不大 -> 装置未变动；
否则说明相对位姿变了，需要重新标定。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APPS = Path(__file__).resolve().parent
for p in (str(ROOT), str(APPS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2
import numpy as np

from calib import BoardSpec, StereoRig
from calib.board import detect_board
from calibrate_stereo import PairSource


def main() -> int:
    ap = argparse.ArgumentParser(description="复检双目标定（装置是否被碰动）")
    ap.add_argument("--calib", default="config/stereo_calibration",
                    help="标定文件前缀（不带扩展名）")
    ap.add_argument("--left", type=int, default=None)
    ap.add_argument("--right", type=int, default=None)
    ap.add_argument("--left-video", default=None)
    ap.add_argument("--right-video", default=None)
    ap.add_argument("--left-dir", default=None)
    ap.add_argument("--right-dir", default=None)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--square", type=float, default=25.0)
    ap.add_argument("--samples", type=int, default=8, help="采集多少帧参与复检")
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--tolerance", type=float, default=0.15,
                    help="允许的极线误差增量（像素）")
    args = ap.parse_args()

    rig = StereoRig.load(args.calib)
    print(rig.describe())
    ref = float(rig.quality.get("epipolar_rms", float("nan")))
    board_desc = rig.meta.get("board", "")
    print(f"标定时记录的极线误差：{ref:.4f} px")
    print()

    spec = BoardSpec(cols=args.cols, rows=args.rows, square_mm=args.square)
    if board_desc and f"{args.cols}x{args.rows}" not in board_desc:
        print(f"[提示] 标定时用的是「{board_desc}」，"
              f"现在按 {args.cols}x{args.rows} 检测，请确认参数一致")

    src = PairSource(args)
    errs: list[float] = []
    frames = 0
    try:
        while len(errs) < args.samples and frames < args.max_frames:
            ok, left, right = src.read()
            frames += 1
            if not ok or left is None or right is None:
                if src.is_live:
                    continue
                break
            gl = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            gr = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
            ol = detect_board(gl, spec)
            orr = detect_board(gr, spec)
            if not (ol.found and orr.found):
                continue
            if ol.sharpness < 40 or orr.sharpness < 40:
                continue
            errs.append(rig.epipolar_error(ol.corners, orr.corners))
    finally:
        src.release()

    if not errs:
        print("[失败] 没有采到有效的标定板图像（两路都要能看到完整的板）")
        return 1

    now = float(np.sqrt(np.mean(np.square(errs))))
    worst = float(np.max(errs))
    print(f"本次复检：{len(errs)} 帧，极线误差 RMS {now:.4f} px（最大 {worst:.4f} px）")
    print(f"标定当时：{ref:.4f} px")
    delta = now - ref
    print(f"变化量：{delta:+.4f} px")
    print()

    ok = True
    if now > 0.6:
        print(f"[不通过] 极线误差 {now:.3f} px 已经偏大，左右相机的相对位姿很可能变了。")
        ok = False
    elif delta > args.tolerance:
        print(f"[不通过] 极线误差比标定时增大 {delta:.3f} px，超过容差 {args.tolerance} px。")
        ok = False

    if ok:
        print("[通过] 装置未发生明显变动，可以继续使用现有标定参数。")
        return 0
    print("建议：重新跑 apps/calibrate_stereo.py 标定一次。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
