"""实时摄像头：检测球体 -> 标记轨迹 -> 预测未来轨迹。

用法示例::

    conda activate myenv
    python apps/run_realtime.py                     # 默认摄像头，融合检测
    python apps/run_realtime.py --method color --preset orange_ball
    python apps/run_realtime.py --camera 1 --width 1920 --height 1080 --fps 60
    python apps/run_realtime.py --record outputs/videos/session.mp4

快捷键：
    q / ESC  退出
    r        重置轨迹（球飞出去以后按一下重新开始）
    s        保存当前标注帧 + 当前配置
    p        开/关轨迹预测
    m        开/关掩膜小窗（调阈值时看这个）
    3        开/关三维视图小面板
    c        在 color / hough / motion / fusion 之间切换检测方法
    v        开始/停止录制
    空格     暂停/继续
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许直接 python apps/xxx.py 运行（把项目根目录加进搜索路径）
# --- 让脚本在任意目录深度都能找到项目根 ---
ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time

import cv2
import numpy as np

from sphere_tracker import camera_registry as cr
from sphere_tracker.config import AppConfig, ColorConfig, load_config, save_config
from sphere_tracker.detect import build_detector
from sphere_tracker.pipeline import SpherePipeline
from sphere_tracker.sources import (
    INDEX_BACKEND,
    CameraSource,
    LatestFrameCamera,
    quiet_repeated_grab_warnings,
)
from sphere_tracker.stats import FramePump, Health

WINDOW = "sphere tracker (real-time)"
METHOD_CYCLE = ["fusion", "color", "hough", "motion", "yolo", "yolo_sam"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="实时球体轨迹识别与预测")
    p.add_argument("--camera", default=None,
                   help="摄像头：索引(0) / 角色名(main，查 config/cameras.json)；"
                        "默认自动挑最优（优先外接、避开内置/红外、非全黑）")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument(
        "--method",
        default=None,
        help="检测方法：fusion / color / hough / motion / yolo / yolo_sam",
    )
    p.add_argument(
        "--preset",
        default=None,
        help="颜色预设：white_ball / orange_ball / green_ball / blue_ball / red_ball / yellow_ball",
    )
    p.add_argument("--config", default=None, help="读取 JSON 配置（标定结果）")
    p.add_argument(
        "--save-config", default="config/sphere_config.json", help="配置保存路径"
    )
    p.add_argument(
        "--ball-diameter",
        type=float,
        default=None,
        help="球直径，单位米（用于三维测距）",
    )
    p.add_argument(
        "--focal", type=float, default=None, help="相机像素焦距 fx（标定过就填）"
    )
    p.add_argument("--horizon", type=float, default=None, help="预测未来多少秒")
    p.add_argument(
        "--predict", default=None, choices=["kalman", "quadratic"], help="预测方法"
    )
    p.add_argument(
        "--gravity-fit", action="store_true", help="三维预测时使用已知重力约束"
    )
    p.add_argument("--no-3d", action="store_true", help="关闭三维估计")
    p.add_argument("--record", default=None, help="把标注后的画面录制成 mp4")
    p.add_argument("--scale", type=float, default=1.0, help="显示缩放")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> AppConfig:
    cfg = load_config(args.config) if args.config else AppConfig()
    if args.method:
        cfg.detector.method = args.method
    if args.preset:
        cfg.detector.color = ColorConfig.preset(args.preset)
    if args.ball_diameter:
        cfg.camera.ball_diameter_m = args.ball_diameter
    if args.focal:
        cfg.camera.fx = args.focal
    if args.horizon:
        cfg.predict.horizon_s = args.horizon
    if args.predict:
        cfg.predict.method = args.predict
    if args.gravity_fit:
        cfg.predict.use_gravity = True
    if args.no_3d:
        cfg.camera.enable_3d = False
    cfg.viz.scale = args.scale
    return cfg


def main() -> int:
    quiet_repeated_grab_warnings()
    args = parse_args()
    cfg = build_config(args)

    arg = args.camera
    if arg is not None and str(arg).strip().lower() == cr.AUTO_TOKEN:
        arg = None                      # --camera auto 等同"不指定"，走下面的自动选择
    try:
        index = cr.resolve_camera_arg(arg)
    except RuntimeError as exc:
        print(f"[错误] {exc}")
        return 1
    if index is None:
        # **只有一个判定者**：`select_cameras_detailed()`。
        # 改之前这里先问 `camera_availability()` 再问 `select_cameras()`，
        # 两套判据对同一台机器结论相反（前者说 ok、后者说"只能动用无关相机"），
        # 于是报出"可用性判定通过，但挑选失败"这种自相矛盾的话。
        sel = cr.select_cameras_detailed(need=1)
        for note in sel.notes:
            print(note)
        if not sel.ok:
            av = cr.camera_availability(need=1)
            print(f"[错误] {av.reason}")
            print(f"[怎么办] {av.remedy}")
            return 1
        index = sel.picks[0].index
        print(f"[自动选择] 使用摄像头 #{index}")

    try:
        # 固定 MSMF：上面的 index 可能来自 select_cameras()（PnP 位次）或角色解析，
        # 两者都按 PnP 枚举编号；用 auto 后端会让编号指向另一台相机。
        # 见 sphere_tracker/sources.py 的 INDEX_BACKEND 注释。
        src = CameraSource(
            index=index, width=args.width, height=args.height, fps=args.fps,
            backend=INDEX_BACKEND,
        )
    except RuntimeError as exc:
        print(f"[错误] {exc}")
        return 1

    info = src.info
    print(
        f"摄像头 {index} 已打开：{info.width}x{info.height} @ {info.fps:.1f} fps（后端 {info.backend}）"
    )
    print(
        f"检测方法：{cfg.detector.method}    预测：{cfg.predict.method}  视野 {cfg.predict.horizon_s}s"
    )
    print("提示：按 c 切换检测方法，按 m 看掩膜，按 s 存快照，按 q 退出。")

    pipe = SpherePipeline(cfg)
    writer: cv2.VideoWriter | None = None
    recording = False
    paused = False
    last_frame: np.ndarray | None = None
    last_vis: np.ndarray | None = None
    frame_times: list[float] = []

    # ── 采集侧：改成后台线程 + 记健康度 ────────────────────────────────────
    #
    # **为什么不再在主循环里直接 `src.read()`**：
    # 直接读的话，"相机只给了 15 fps" 与 "我们处理得慢所以只跑 15 fps"
    # 在画面上完全一样，优化方向只能靠猜；而且相机掉线时 `read()` 返回空，
    # 主循环只是 `continue` —— **画面冻住但程序照跑**。
    #
    # `LatestFrameCamera` 把采集挪到自己的线程，于是能分别回答：
    #   采集给了多少帧（cap）、我们处理了多少帧（loop）、
    #   有没有帧被覆盖掉（dropped）、当前这帧是多久以前抓的（age）、
    #   以及**采集是不是已经停住**（stalled）。
    cam_thread = LatestFrameCamera(src)
    health = Health()
    pump = FramePump(cam_thread, health, stalled_after_s=1.0)
    t_start = time.perf_counter()
    print("采集已在后台线程运行（HUD 会分别显示 cap / loop / detect）")

    try:
        while True:
            if not paused:
                got = pump.next()
                if got.frame is None or not got.is_new:
                    # 没有新帧：睡一下让出 CPU。"等多久"是这里的策略，
                    # `FramePump` 刻意不管（带上时序就难测了）。
                    time.sleep(0.002)
                    continue
                frame = got.frame
                t0 = time.perf_counter()
                result = pipe.process(frame, time.perf_counter() - t_start)
                result.stalled = got.stalled
                vis = pipe.render(frame, result, health=health)
                process_ms = (time.perf_counter() - t0) * 1000.0
                frame_times.append(process_ms / 1000.0)
                if len(frame_times) > 60:
                    frame_times.pop(0)
                health.end_frame(detect_ms=result.detect_ms, track_ms=0.0, render_ms=0.0,
                                 process_ms=process_ms)
                last_frame, last_vis = frame, vis
                if recording and writer is not None:
                    writer.write(vis)
            else:
                vis = (
                    last_vis
                    if last_vis is not None
                    else np.zeros((480, 640, 3), np.uint8)
                )

            cv2.imshow(WINDOW, vis)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
                print("已暂停" if paused else "继续")
            elif key == ord("r"):
                pipe.reset()
                print("轨迹已重置")
            elif key == ord("p"):
                cfg.predict.enable = not cfg.predict.enable
                print(f"预测：{'开' if cfg.predict.enable else '关'}")
            elif key == ord("m"):
                cfg.viz.show_mask = not cfg.viz.show_mask
            elif key == ord("3"):
                cfg.viz.show_axes_3d = not cfg.viz.show_axes_3d
            elif key == ord("c"):
                i = (
                    METHOD_CYCLE.index(cfg.detector.method)
                    if cfg.detector.method in METHOD_CYCLE
                    else 0
                )
                cfg.detector.method = METHOD_CYCLE[(i + 1) % len(METHOD_CYCLE)]
                pipe.detector = build_detector(cfg.detector)
                print(f"检测方法切换为：{cfg.detector.method}")
            elif key == ord("s"):
                stamp = time.strftime("%Y%m%d_%H%M%S")
                out_dir = ROOT / "outputs" / "images"
                out_dir.mkdir(parents=True, exist_ok=True)
                if last_frame is not None and last_vis is not None:
                    cv2.imwrite(str(out_dir / f"snapshot_{stamp}.png"), last_vis)
                    cv2.imwrite(str(out_dir / f"raw_{stamp}.png"), last_frame)
                    print(f"已保存快照：outputs/images/snapshot_{stamp}.png")
                save_config(cfg, args.save_config)
                print(f"配置已保存：{args.save_config}")
            elif key == ord("v"):
                if not recording:
                    path = args.record or str(
                        ROOT
                        / "outputs"
                        / "videos"
                        / f"session_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                    )
                    Path(path).parent.mkdir(parents=True, exist_ok=True)
                    h, w = vis.shape[:2]
                    rec_fps = (
                        src.info.fps if (src.info.fps and src.info.fps > 1.0) else 30.0
                    )
                    writer = cv2.VideoWriter(
                        path, cv2.VideoWriter.fourcc(*"mp4v"), rec_fps, (w, h)
                    )
                    recording = writer.isOpened()
                    print(
                        f"{'开始录制：' + path if recording else '录制失败（编码器不可用）'}"
                    )
                else:
                    recording = False
                    if writer is not None:
                        writer.release()
                        writer = None
                    print("录制已停止")
    finally:
        # 先停采集线程再放相机 —— 反了的话采集线程会在已释放的源上继续 read
        cam_thread._stop = True
        cam_thread._thread.join(timeout=1.0)
        if writer is not None:
            writer.release()
        src.release()
        cv2.destroyAllWindows()
        if frame_times:
            avg = sum(frame_times) / len(frame_times)
            print(
                f"平均每帧处理耗时 {avg * 1000:.1f} ms（约 {1.0 / avg:.1f} FPS 的处理能力）"
            )
        # 结尾把三个口径一起打出来：cap 与 loop 差得远 = 瓶颈在检测，不是相机
        print(f"本次运行健康度：{health.summary_line()}")
        if health.dropped_frames:
            print(f"  其中 {health.dropped_frames} 帧在取走前被覆盖 —— "
                  f"说明主循环跟不上采集，看的是跳帧后的画面")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
