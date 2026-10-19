"""离线双目：读一个 `data/sessions/<id>`，输出三维轨迹 CSV + 标注视频。

它不是"新的视觉算法"，而是把已有的三块拼起来：

    左路 SpherePipeline（检测 + 跟踪）
    右路 SpherePipeline（检测 + 跟踪）
    StereoReconstructor（极线配对 + 三角测量）

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

⚠️ 默认检测器是 `color`，适合**哑光纯色球**。金属/高光球（例如摆锤）用
`color` 会锁到背景高光；可试 `--method hough --min-radius 6 --max-radius 15`，
但真实纹理背景下 Hough 仍有误检 —— 这是检测问题，不是三角测量问题。
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "sphere_tracker").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calib import StereoRig                                  # noqa: E402
from sphere_tracker.config import AppConfig, ColorConfig     # noqa: E402
from sphere_tracker.geometry import CameraModel              # noqa: E402
from sphere_tracker.pipeline import SpherePipeline           # noqa: E402
from sphere_tracker.types import Detection                   # noqa: E402
from stereo import MatchConfig, PairSource, StereoReconstructor  # noqa: E402
from stereo.sessions import load_session, validate_session   # noqa: E402


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
        ball_diameter_m=0.04,
    )
    return pipe


def _draw(frame: np.ndarray, res, label: str) -> np.ndarray:
    out = frame.copy()
    d = res.detection
    if d is not None:
        cv2.circle(out, (int(round(d.x)), int(round(d.y))), int(round(d.radius)),
                   (0, 0, 255), 2)
        cv2.putText(out, f"{label} r={d.radius:.1f} {d.method}", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.putText(out, f"{label} no detection", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="离线双目三维轨迹（读会话）")
    ap.add_argument("session", help="data/sessions/<id> 目录或 session.json")
    ap.add_argument("--method", default="color",
                    choices=("color", "hough", "fusion", "motion"))
    ap.add_argument("--preset", default="", help="color 预设，如 white_ball；空=默认")
    ap.add_argument("--detect-width", type=int, default=0,
                    help="检测缩到多宽（0=原图；真实纹理下 hough 建议 960）")
    ap.add_argument("--min-radius", type=float, default=8.0)
    ap.add_argument("--max-radius", type=float, default=200.0)
    ap.add_argument("--hough-max-radius", type=float, default=80.0)
    ap.add_argument("--max-epipolar", type=float, default=2.5)
    ap.add_argument("--sync-offset", type=int, default=None,
                    help="覆盖 session 的同步偏移（默认用 session.sync.offset）")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    session_dir, s = load_session(args.session)
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
    left_video = session_dir / s.left.video
    right_video = session_dir / s.right.video
    fps = float(s.left.fps_measured or s.left.fps_requested or 30.0)

    offset = args.sync_offset if args.sync_offset is not None else int(s.sync.offset)
    if args.sync_offset is None and s.sync.method == "none":
        print("[警告] session 的同步方法是 none（没检测到闪光/拍手）—— 3D 可能错位；"
              "可先用 --sync-offset 手动试。")
    print(f"同步偏移：{offset:+d} 帧；fps={fps:.2f}")
    print(f"视频：{left_video.name} + {right_video.name}")

    src = PairSource(video=(left_video, right_video), sync_offset=offset)
    if tuple(src.image_size) != tuple(rig.image_size):
        print(f"[错误] 视频 {src.image_size} 与标定 {rig.image_size} 不一致")
        src.release()
        return 1

    cfg = AppConfig()
    cfg.detector.method = args.method
    cfg.detector.min_radius = float(args.min_radius)
    cfg.detector.max_radius = float(args.max_radius)
    cfg.detector.hough_max_radius = float(args.hough_max_radius)
    cfg.predict.enable = False
    if args.preset:
        cfg.detector.color = ColorConfig.preset(args.preset)
    pipe_l = _make_pipeline(cfg, rig, "left")
    pipe_r = _make_pipeline(cfg, rig, "right")
    if args.detect_width:
        pipe_l.set_detect_width(args.detect_width)
        pipe_r.set_detect_width(args.detect_width)

    rec = StereoReconstructor(rig, MatchConfig(max_epipolar_px=args.max_epipolar))
    out_dir = session_dir / "derived"
    out_dir.mkdir(parents=True, exist_ok=True)
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
    n_pairs = 0
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

            if pt is None:
                rows.append([n_pairs - 1, f"{t:.4f}", 0, "", "", "", "", "", "", "", "", 0,
                             "" if det_l is None else det_l.method,
                             "" if det_r is None else det_r.method])
            else:
                rows.append([
                    n_pairs - 1, f"{t:.4f}", 1,
                    f"{pt.xyz[0]:.6f}", f"{pt.xyz[1]:.6f}", f"{pt.xyz[2]:.6f}",
                    f"{pt.u_left[0]:.2f}", f"{pt.u_left[1]:.2f}",
                    f"{pt.u_right[0]:.2f}", f"{pt.u_right[1]:.2f}",
                    f"{pt.epipolar_px:.3f}", f"{pt.residual_px:.3f}",
                    1, "" if det_l is None else det_l.method,
                    "" if det_r is None else det_r.method,
                ])

            if writer is not None:
                vl = cv2.resize(_draw(left_frame, res_l, "L"), (960, 540))
                vr = cv2.resize(_draw(right_frame, res_r, "R"), (960, 540))
                cv2.putText(vr, f"f={n_pairs-1} matched={rec.n_matched}", (12, 500),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                writer.write(np.hstack([vl, vr]))
    finally:
        src.release()
        if writer is not None:
            writer.release()

    header = ["frame", "t", "matched", "X", "Y", "Z", "u_left", "v_left",
              "u_right", "v_right", "epipolar_px", "residual_px", "confident",
              "method_left", "method_right"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    elapsed = time.perf_counter() - t0
    print()
    print(f"处理：{n_pairs} 对，{elapsed:.1f}s（{n_pairs/max(elapsed,1e-6):.1f} 对/s）")
    print(f"匹配：{rec.n_matched}/{n_pairs}（{rec.match_rate*100:.1f}%）  "
          f"无左={rec.n_no_left} 无右={rec.n_no_right} 极线拒={rec.n_rejected}")
    if rec.trajectory:
        arr = rec.as_array()
        print(f"三维范围：X {arr[:,1].min():.3f}~{arr[:,1].max():.3f} m  "
              f"Y {arr[:,2].min():.3f}~{arr[:,2].max():.3f} m  "
              f"Z {arr[:,3].min():.3f}~{arr[:,3].max():.3f} m")
        res = np.array([p.residual_px for p in rec.trajectory])
        print(f"交会残差：中位 {np.median(res):.3f} px，最大 {res.max():.3f} px")
    print(f"CSV：{csv_path}")
    if writer is not None:
        print(f"标注视频：{video_path}")
    return 0 if rec.n_matched else 2


if __name__ == "__main__":
    raise SystemExit(main())
