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
from stereo import MatchConfig, PairSource, StereoReconstructor, savgol_trajectory  # noqa: E402
from stereo.visualization import (  # noqa: E402
    export_session_visualization,
    load_seeds,
    write_seeds,
)
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


def _project_left(rig, xyz) -> tuple[int, int] | None:
    """把校正左相机系的三维点投到**原始左图**。"""
    try:
        p = np.asarray(xyz, np.float64).reshape(1, 3)
        rvec, _ = cv2.Rodrigues(rig.R1.T)
        uv, _ = cv2.projectPoints(p, rvec, np.zeros(3), rig.K1, rig.D1)
        return int(round(float(uv[0, 0, 0]))), int(round(float(uv[0, 0, 1])))
    except Exception:                                       # noqa: BLE001
        return None


def _draw_reference(frame: np.ndarray, rig, origin: tuple[float, float, float],
                    axis_m: float = 0.10) -> None:
    """在左图标出坐标原点 + X/Y/Z 轴（原点默认 = 校正左相机光心投影）。"""
    o = _project_left(rig, origin)
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
        p = _project_left(rig, xyz)
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
          *, xyz=None, rig=None, origin=None, draw_reference: bool = False) -> np.ndarray:
    out = frame.copy()
    if draw_reference and rig is not None and origin is not None:
        _draw_reference(out, rig, origin)
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
    ax.set_ylabel("Y forward (m)")
    ax.set_zlabel("Z up (m)")
    ax.set_title("Stereo 3D trajectory")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="离线双目三维轨迹（读会话）")
    ap.add_argument("session", help="data/sessions/<id> 目录或 session.json")
    ap.add_argument("--method", default="color",
                    choices=("color", "hough", "fusion", "motion", "specular"))
    ap.add_argument("--preset", default="", help="color 预设，如 white_ball；空=默认")
    ap.add_argument("--origin", default="0,0,0",
                    help="视频里标出的参照原点（校正左相机系，x,y,z 米；默认相机光心投影）")
    ap.add_argument("--seed", default="",
                    help="specular 左路人工种子 x,y,r（1920 下 r 约 32~40）")
    ap.add_argument("--seed-right", default="",
                    help="specular 右路人工种子 x,y,r（不能用左路 seed 代替）")
    ap.add_argument("--seed-json", default="",
                    help="specular 的 seeds.json（默认自动用 <session>/derived/seeds.json）")
    ap.add_argument("--seed-frame", type=int, default=0,
                    help="specular 在第几帧初始化（默认 0；种子坐标要对应这一帧）")
    ap.add_argument("--detect-width", type=int, default=0,
                    help="检测缩到多宽（0=原图；真实纹理下 hough 建议 960）")
    ap.add_argument("--min-radius", type=float, default=8.0)
    ap.add_argument("--max-radius", type=float, default=200.0)
    ap.add_argument("--hough-max-radius", type=float, default=80.0)
    ap.add_argument("--max-epipolar", type=float, default=2.5)
    ap.add_argument("--sync-offset", type=int, default=None,
                    help="覆盖 session 的同步偏移（默认用 session.sync.offset）")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--trail", type=int, default=300, help="轨迹线保留多少个匹配点（0=不画）")
    ap.add_argument("--smooth", action="store_true", help="另存 Savitzky-Golay 平滑后的三维 CSV")
    ap.add_argument("--no-plot", action="store_true", help="不生成 traj3d.png")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    session_dir, s = load_session(args.session)
    try:
        origin = tuple(float(v) for v in args.origin.split(","))
        if len(origin) != 3:
            raise ValueError
    except ValueError:
        print(f"[错误] --origin 要写成 x,y,z 三个数，现在是 {args.origin!r}")
        return 2
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
    if args.method == "specular" and abs(args.max_epipolar - 2.5) < 1e-9:
        args.max_epipolar = 8.0
        print("[specular] 极线门限 2.5px 对人工 seed/CSRT 太紧，默认放宽到 8px")
    cfg.detector.min_radius = float(args.min_radius)
    cfg.detector.max_radius = float(args.max_radius)
    cfg.detector.hough_max_radius = float(args.hough_max_radius)
    cfg.predict.enable = False
    if args.preset:
        cfg.detector.color = ColorConfig.preset(args.preset)
    pipe_l = _make_pipeline(cfg, rig, "left")
    pipe_r = _make_pipeline(cfg, rig, "right")
    if args.method == "specular":
        def _seed_arg(text: str, what: str):
            try:
                parts = [float(v) for v in text.split(",")]
            except ValueError:
                parts = []
            if len(parts) != 3 or parts[2] <= 0:
                print(f"[错误] {what} 要写成 x,y,r 三个数，例如 1190,300,36；现在是 {text!r}")
                return None
            return tuple(parts)

        if args.detect_width:
            print("[错误] --method specular 暂不支持 --detect-width："
                  "种子坐标是原图坐标，缩放会错位。")
            return 2
        seed_path = Path(args.seed_json) if args.seed_json else (session_dir / "derived" / "seeds.json")
        seed_json = None
        if seed_path.exists():
            seed_json = load_seeds(seed_path)
        elif args.seed_json:
            print(f"[错误] --seed-json 不存在：{seed_path}")
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
                                    note="命令行 --seed/--seed-right")
                print(f"[specular] 已写标准 seed：{saved}")
            except Exception as exc:                        # noqa: BLE001
                print(f"[警告] seed 写入失败（继续跑）：{exc}")
        elif seed_json is not None:
            seed_l = (float(seed_json["left"]["x"]), float(seed_json["left"]["y"]),
                      float(seed_json["left"]["radius"]))
            seed_r = (float(seed_json["right"]["x"]), float(seed_json["right"]["y"]),
                      float(seed_json["right"]["radius"]))
            seed_frame = int(seed_json.get("seed_frame", args.seed_frame))
            print(f"[specular] 读标准 seed：{seed_path}")
        else:
            print("[错误] --method specular 需要球心种子：")
            print("  GUI：第 3 步「点选球（双目 seed）」；")
            print("  或 CLI：--seed 左x,左y,左r --seed-right 右x,右y,右r")
            print("  也可用 --seed-json <seeds.json>。自动初始化不可靠，故意不猜。")
            return 2
        pipe_l.detector.seed(*seed_l, frame_index=seed_frame)
        pipe_r.detector.seed(*seed_r, frame_index=seed_frame)
        print(f"[specular] 左 seed={seed_l} 右 seed={seed_r} 初始化帧={seed_frame}")
    elif args.seed or args.seed_right:
        print("[警告] 只有 --method specular 使用 --seed/--seed-right；已忽略。")
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
    trail_l: list[tuple[float, float]] = []
    trail_r: list[tuple[float, float]] = []
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
                    1, "" if det_l is None else det_l.method,
                    "" if det_r is None else det_r.method,
                ])

            if writer is not None:
                vl = cv2.resize(_draw(left_frame, res_l, "L", trail_l,
                                      xyz=None if pt is None else pt.xyz,
                                      rig=rig, origin=origin, draw_reference=True), (960, 540))
                vr = cv2.resize(_draw(right_frame, res_r, "R", trail_r,
                                      xyz=None if pt is None else pt.xyz), (960, 540))
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
    smoothed = None
    if rec.trajectory:
        arr = rec.as_array()
        print(f"三维范围：X {arr[:,1].min():.3f}~{arr[:,1].max():.3f} m  "
              f"Y {arr[:,2].min():.3f}~{arr[:,2].max():.3f} m  "
              f"Z {arr[:,3].min():.3f}~{arr[:,3].max():.3f} m")
        res = np.array([p.residual_px for p in rec.trajectory])
        print(f"交会残差：中位 {np.median(res):.3f} px，最大 {res.max():.3f} px")
        if args.smooth and len(arr) >= 3:
            smoothed = savgol_trajectory(arr[:, 0], arr[:, 1:4], window=11, polyorder=2)
            sm_path = out_dir / f"traj3d_smoothed{partial}.csv"
            with open(sm_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["t", "X", "Y", "Z"])
                for ti, x, y, z in zip(arr[:, 0], smoothed[:, 0], smoothed[:, 1], smoothed[:, 2]):
                    w.writerow([f"{ti:.4f}", f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])
            print(f"平滑 CSV：{sm_path}")
        if not args.no_plot:
            plot_path = out_dir / f"traj3d{partial}.png"
            _plot_trajectory(arr, plot_path, smoothed)
            print(f"三维图：{plot_path}")
    print(f"CSV：{csv_path}")
    if writer is not None:
        print(f"标注视频：{video_path}")
    if not partial:
        try:
            paths = export_session_visualization(session_dir, csv_name=csv_path.name)
            print(f"标准轨迹 JSON：{paths['trajectory']}")
            print(f"可视化 JSON：{paths['visualization']}")
        except Exception as exc:                            # noqa: BLE001
            print(f"[警告] 标准 JSON 导出失败：{exc}")
    return 0 if rec.n_matched else 2


if __name__ == "__main__":
    raise SystemExit(main())
