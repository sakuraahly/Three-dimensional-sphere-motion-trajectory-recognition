"""HSV 阈值标定工具：拖动滑块，实时看掩膜和检测效果。

用法::

    python apps/tools/calibrate_color.py --camera 0
    python apps/tools/calibrate_color.py --image assets/images/ball.png
    python apps/tools/calibrate_color.py --video assets/videos/ball.mp4

按键：
    1-6    套用颜色预设（白/橙/绿/蓝/红/黄）
    f      冻结/解冻画面（冻结后调参数不会动，方便慢慢调）
    s      保存到 config/sphere_config.json
    q/ESC  退出
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

from apps._window import WindowManager  # noqa: E402
from sphere_tracker.config import AppConfig, ColorConfig, save_config
from sphere_tracker.detectors import ColorBlobDetector
from sphere_tracker.sources import open_source

WINDOW = "calibrate color"
PRESETS = ["white_ball", "orange_ball", "green_ball", "blue_ball", "red_ball", "yellow_ball"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HSV 标定")
    p.add_argument("--camera", type=int, default=None)
    p.add_argument("--image", default=None)
    p.add_argument("--video", default=None)
    p.add_argument("--config", default=None, help="初始配置")
    p.add_argument("--save", default="config/sphere_config.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = AppConfig()
    if args.config:
        from sphere_tracker.config import load_config

        cfg = load_config(args.config)

    path = args.image or args.video
    src = open_source(path, camera=args.camera)
    print(f"输入：{src.info.kind} {src.info.width}x{src.info.height} @ {src.info.fps:.1f} fps")

    def nothing(_: int) -> None:
        pass

    wins = WindowManager()
    wins.keep_size(WINDOW)      # trackbar 挂在这个窗口上，不主动改尺寸
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    c = cfg.detector.color
    bars = {
        "H_min": (0, 179, c.hsv_lower[0]),
        "H_max": (0, 179, c.hsv_upper[0]),
        "S_min": (0, 255, c.hsv_lower[1]),
        "S_max": (0, 255, c.hsv_upper[1]),
        "V_min": (0, 255, c.hsv_lower[2]),
        "V_max": (0, 255, c.hsv_upper[2]),
        "open": (1, 15, c.open_ksize),
        "close": (1, 21, c.close_ksize),
        "r_min": (2, 200, int(cfg.detector.min_radius)),
        "r_max": (5, 400, int(cfg.detector.max_radius)),
        "circ_x100": (10, 100, int(cfg.detector.min_circularity * 100)),
        "fill_x100": (10, 100, int(cfg.detector.min_fill_ratio * 100)),
    }
    for name, (lo, hi, val) in bars.items():
        cv2.createTrackbar(name, WINDOW, int(np.clip(val, lo, hi)), hi, nothing)

    frozen: np.ndarray | None = None
    print("拖动滑块调阈值；按 1-6 套预设，f 冻结，s 保存，q 退出。")

    while True:
        if frozen is not None:
            frame = frozen
        else:
            ok, frame, _ = src.read()
            if not ok or frame is None:
                if src.is_live:
                    time.sleep(0.01)
                    continue
                break

        c.hsv_lower = [
            cv2.getTrackbarPos("H_min", WINDOW),
            cv2.getTrackbarPos("S_min", WINDOW),
            cv2.getTrackbarPos("V_min", WINDOW),
        ]
        c.hsv_upper = [
            max(cv2.getTrackbarPos("H_max", WINDOW), c.hsv_lower[0]),
            max(cv2.getTrackbarPos("S_max", WINDOW), c.hsv_lower[1]),
            max(cv2.getTrackbarPos("V_max", WINDOW), c.hsv_lower[2]),
        ]
        c.open_ksize = max(1, cv2.getTrackbarPos("open", WINDOW) | 1)
        c.close_ksize = max(1, cv2.getTrackbarPos("close", WINDOW) | 1)
        cfg.detector.min_radius = float(cv2.getTrackbarPos("r_min", WINDOW))
        cfg.detector.max_radius = float(cv2.getTrackbarPos("r_max", WINDOW))
        cfg.detector.min_circularity = cv2.getTrackbarPos("circ_x100", WINDOW) / 100.0
        cfg.detector.min_fill_ratio = cv2.getTrackbarPos("fill_x100", WINDOW) / 100.0

        detector = ColorBlobDetector(cfg.detector)
        mask = detector.build_mask(frame)
        found = detector.detect(frame)

        vis = frame.copy()
        for det in found[:5]:
            cv2.circle(vis, (int(det.x), int(det.y)), int(det.radius), (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(vis, f"{det.score:.2f}", (int(det.x) + 8, int(det.y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([vis, mask_bgr])
        info = (
            f"H[{c.hsv_lower[0]},{c.hsv_upper[0]}] S[{c.hsv_lower[1]},{c.hsv_upper[1]}] "
            f"V[{c.hsv_lower[2]},{c.hsv_upper[2]}]  found={len(found)}"
            + ("   [FROZEN]" if frozen is not None else "")
        )
        cv2.putText(combined, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        wins.show(WINDOW, combined)

        key = cv2.waitKey(1) & 0xFF
        # ⚠️ **只留 F11，不要 f** —— 这个程序里 `f` 已经是"冻结画面"，
        # 全屏若抢走它就会把一个既有功能弄坏（抢键比没有全屏更糟）。
        # Esc 仍然优先"退出全屏"，只有不在全屏时才轮到下面的"退出程序"。
        if wins.handle_key(WINDOW, key, None, toggle_keys=(122,)):
            continue
        if key in (ord("q"), 27):
            break
        if key == ord("f"):
            frozen = None if frozen is not None else frame.copy()
        if key == ord("s"):
            p = save_config(cfg, args.save)
            print(f"已保存配置：{p}")
        if ord("1") <= key <= ord("6"):
            name = PRESETS[key - ord("1")]
            c = ColorConfig.preset(name)
            cfg.detector.color = c
            cv2.setTrackbarPos("H_min", WINDOW, c.hsv_lower[0])
            cv2.setTrackbarPos("H_max", WINDOW, c.hsv_upper[0])
            cv2.setTrackbarPos("S_min", WINDOW, c.hsv_lower[1])
            cv2.setTrackbarPos("S_max", WINDOW, c.hsv_upper[1])
            cv2.setTrackbarPos("V_min", WINDOW, c.hsv_lower[2])
            cv2.setTrackbarPos("V_max", WINDOW, c.hsv_upper[2])
            print(f"已套用预设：{name}")

    src.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
