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
import logging
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
from sphere_tracker import video_profiles
from sphere_tracker.config import AppConfig, ColorConfig, load_config, save_config
from sphere_tracker.detectors import build_detector
from sphere_tracker.pipeline import SpherePipeline
from sphere_tracker.sources import (
    INDEX_BACKEND,
    CameraSource,
    LatestFrameCamera,
    quiet_repeated_grab_warnings,
)
from sphere_tracker.stats import FramePump, Health
from apps._common import log_environment, setup_logging
from apps._window import WindowManager

_log = logging.getLogger("realtime")


def _describe(info) -> str:
    """把 `SourceInfo` 说成一行。

    **为什么不用 `stereo.pair_source.describe_info`**：那是双目模块的函数，
    而实时检测是**单目** app。为了打一行日志去 import 双目模块，
    会造出 `apps → stereo` 的依赖（分层上 stereo 在 apps 下面，
    但单目这条链本来就与双目无关）。几个字段直接格式化更干净。
    """
    bits = [f"{info.width}x{info.height}"]
    if info.fps and info.fps > 0:
        bits.append(f"{info.fps:.1f} fps")
    if info.backend:
        bits.append(f"后端 {info.backend}")
    if info.kind:
        bits.append(info.kind)
    return "，".join(bits)

WINDOW = "sphere tracker (real-time)"
METHOD_CYCLE = ["fusion", "color", "hough", "motion", "yolo", "yolo_sam"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="实时球体轨迹识别与预测")
    p.add_argument("--camera", default=None,
                   help="摄像头：索引(0) / 角色名(main，查 config/cameras.json)；"
                        "默认自动挑最优（优先外接、避开内置/红外、非全黑）")
    p.add_argument("--width", type=int, default=video_profiles.DEFAULT.width,
                   help=f"采集宽度（默认档 {video_profiles.DEFAULT.label}）")
    p.add_argument("--height", type=int, default=video_profiles.DEFAULT.height,
                   help=f"采集高度（默认档 {video_profiles.DEFAULT.label}）")
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
    p.add_argument("--scale", type=float, default=1.0, help="显示缩放（放大整张图，>1 才用）")
    # ---- 显示分辨率：流畅度的**最大单项**（2026-10-14 实测）----
    #
    # `cv2.imshow` 把 1920x1080 的图缩进窗口显示，实测要 **17.74 ms**；
    # 而先把图 resize 到 960x540 再 imshow 只要 **1.84 ms**（便宜 9.6 倍）。
    # 也就是说每秒最多 56 帧是**花在缩放上**的，而这件事我们自己做只要 0.55 ms。
    #
    # 默认 960：1280 也顺（11.7ms），但 960 在"看得清"和"顺"之间更稳。
    # 想看原生细节就用 `--display-width 0`（= 不缩，回到旧行为）。
    p.add_argument("--display-width", type=int, default=960,
                   help="显示图宽度（像素）。默认 960，明显更顺；0 = 不缩放（原生分辨率）")
    p.add_argument("--log-every", type=float, default=5.0,
                   help="每隔几秒往日志写一行健康度（0 = 不写）")
    # ---- 检测分辨率：真实纹理下这是**唯一有效**的提速杠杆 ----
    #
    # 实测（1080p，书架那种真实纹理）：color 17.8ms、hough **4384.9ms**、
    # fusion **3740.5ms**。而同样的画面若是纯色合成图，hough 只要 132ms ——
    # HoughCircles 的 Canny 边缘一多，累加器搜索就是指数级的。
    #
    # 默认 0 = 不缩（保持既有精度，不悄悄改变行为）。
    # 想跑 fusion/hough 就**必须**开这个，否则 3.7 秒一帧。
    p.add_argument("--detect-width", type=int, default=0,
                   help="检测跑在多宽的图上（0=原始分辨率）。fusion/hough 在真实"
                        "纹理下要几秒一帧，给 960 会快一个量级")
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

    # ---- 日志：实时检测过去**一份都不落盘**（2026-10-14 升级）----
    #
    # 其它 app 都调了 `setup_logging`，只有实时检测没有 —— 只有终端输出，
    # 进程一退就全没了。而"卡顿 / 丢帧 / 相机掉线"这类问题恰恰是**过后才想查**的：
    # 用户看到的是"卡"，真正的数字（cap / loop / detect / drop / age / skew）
    # 必须留下来才能定位，否则只能靠重跑一次复现。
    #
    # 级别分工沿用 `_common.setup_logging`：文件收 DEBUG（每帧都留），终端只 INFO。
    log_path = setup_logging("realtime")
    log_environment(_log)
    print(f"日志：{log_path}")

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
    # 窗口管理器**要在打印提示之前就建好** —— 提示语里要用它的按键说明。
    # （第一版把它建在下面 `pipe = ...` 旁边，于是这行提示引用了还没赋值的
    #  `wins`，`UnboundLocalError`。冒烟跑一遍才发现，静态检查看不出来。）
    wins = WindowManager()
    print(
        f"摄像头 {index} 已打开：{info.width}x{info.height} @ {info.fps:.1f} fps（后端 {info.backend}）"
    )
    print(
        f"检测方法：{cfg.detector.method}    预测：{cfg.predict.method}  视野 {cfg.predict.horizon_s}s"
    )
    print("提示：按 c 切换检测方法，按 m 看掩膜，按 s 存快照，"
          f"{wins.fullscreen_hint()}，按 q 退出。")

    pipe = SpherePipeline(cfg)
    if args.detect_width:
        pipe.set_detect_width(args.detect_width)
        print(f"检测跑在 {args.detect_width}px 宽的图上（坐标会自动换算回原图）。"
              f"\\n  代价：小目标的定位精度会下降；换来的是 hough/fusion 从几秒一帧"
              f"降到可用的量级。")
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
    _log.info("相机 %s 已打开：%s", index, _describe(info))
    _log.info("检测方法 %s；显示宽度 %s；预测 %s",
              cfg.detector.method,
              args.display_width or f"{info.width}（不缩放）", cfg.predict.method)
    print("采集已在后台线程运行（HUD 会分别显示 cap / loop / detect）")
    if args.display_width:
        print(f"显示按 {args.display_width}px 宽输出 —— 实测这一步把显示耗时"
              f"从 ~17.7ms 降到 ~1.8ms（这是流畅度最大的一项）")

    # ---- 周期性健康度进日志（本次升级的重点）----
    #
    # 原来实时检测**一份日志都不落盘**（其它 app 都调 setup_logging，只有它没有）：
    # 进程一退，"当时到底多少帧、丢了多少帧、检测多慢"就全没了。
    # 而"卡顿"恰恰是**过后才想查**的问题 —— 用户看到的只是"卡"，
    # 真正的数字（cap / loop / detect / drop / age）必须留下来才能定位。
    next_report = time.perf_counter() + max(args.log_every, 0.0)

    def _report_health(force: bool = False) -> None:
        """写一行健康度到日志（`--log-every 0` 可关，force 用于收尾）。"""
        nonlocal next_report
        if args.log_every <= 0 and not force:
            return
        now = time.perf_counter()
        if not force and now < next_report:
            return
        next_report = now + args.log_every
        _log.info("健康度 %s", health.summary_line())
        # 掉帧比例明显时**顺带说一句人话**：光看 drop 3 不知道严重不严重
        if health.drop_rate > 0.25 and health.processed_frames > 30:
            _log.warning(
                "丢帧 %.0f%%（采集 %.1f fps，主循环只有 %.1f fps）——"
                "画面会顿。可调小窗口或 --display-width，或换更快的方法（color）",
                health.drop_rate * 100, health.capture_fps, health.loop_fps)

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
                # ---- 逐帧 DEBUG 记录（**只有文件收**，终端不受影响）----
                #
                # 为什么值得每帧写一行：事后想回答的问题几乎都是逐帧量的 ——
                # "球在第几帧丢的""丢那一下检测耗时是不是飙了""坐标是不是跳了"。
                # 只在 INFO 记汇总的话，这些**永远查不出来**，只能重跑一次复现
                # （而现场往往已经变了）。`_common.setup_logging` 的分工就是
                # 文件收 DEBUG、终端只 INFO，所以这里写 DEBUG 不会刷屏。
                det = result.detection
                _log.debug(
                    "帧 %d  seq=%s  detect %.1fms  found=%s  r=%s  c=(%s,%s)  "
                    "trail=%d  age=%.1fms  stalled=%s",
                    health.processed_frames,
                    getattr(got, "seq", "?"),
                    result.detect_ms,
                    det is not None,
                    f"{det.radius:.1f}" if det is not None else "-",
                    f"{det.center[0]:.0f}" if det is not None else "-",
                    f"{det.center[1]:.0f}" if det is not None else "-",
                    len(result.trail or ()),
                    health.frame_age_ms,
                    got.stalled,
                )
                vis = pipe.render(frame, result, health=health,
                                  display_width=args.display_width)
                process_ms = (time.perf_counter() - t0) * 1000.0
                frame_times.append(process_ms / 1000.0)
                if len(frame_times) > 60:
                    frame_times.pop(0)
                health.end_frame(detect_ms=result.detect_ms, track_ms=0.0, render_ms=0.0,
                                 process_ms=process_ms)
                last_frame, last_vis = frame, vis
                if recording and writer is not None:
                    # ⚠️ 录制**不能**录那张 960 的小图 —— 显示要小（省 imshow 开销），
                    # 但存档要好。所以录制时按原始分辨率另渲一张（只有录制才付这份钱）。
                    if args.display_width:
                        rec_vis = pipe.render(frame, result, health=health,
                                              render_scale=1.0)
                    else:
                        rec_vis = vis
                    writer.write(rec_vis)
            else:
                vis = (
                    last_vis
                    if last_vis is not None
                    else np.zeros((480, 640, 3), np.uint8)
                )

            wins.show(WINDOW, vis)
            key = cv2.waitKey(1) & 0xFF
            # 全屏相关的键先处理：Esc 在全屏时是退出全屏，只有不在全屏时
            # 才轮到下面的退出程序。否则用户全屏之后一按 Esc 就把程序关了。
            if wins.handle_key(WINDOW, key, vis):
                continue
            _report_health()

            if key in (ord("q"), 27):
                _log.info("用户按 q 退出")
                break
            if key == ord(" "):
                paused = not paused
                print("已暂停" if paused else "继续")
                _log.info("用户%s", "暂停" if paused else "继续")
            elif key == ord("r"):
                pipe.reset()
                print("轨迹已重置")
                _log.info("用户重置轨迹")
            elif key == ord("p"):
                cfg.predict.enable = not cfg.predict.enable
                print(f"预测：{'开' if cfg.predict.enable else '关'}")
                _log.info("用户切换预测：%s", cfg.predict.enable)
            elif key == ord("m"):
                cfg.viz.show_mask = not cfg.viz.show_mask
                _log.info("用户切换掩膜显示：%s", cfg.viz.show_mask)
            elif key == ord("3"):
                cfg.viz.show_axes_3d = not cfg.viz.show_axes_3d
                _log.info("用户切换三维坐标轴：%s", cfg.viz.show_axes_3d)
            elif key == ord("c"):
                i = (
                    METHOD_CYCLE.index(cfg.detector.method)
                    if cfg.detector.method in METHOD_CYCLE
                    else 0
                )
                cfg.detector.method = METHOD_CYCLE[(i + 1) % len(METHOD_CYCLE)]
                pipe.detector = build_detector(cfg.detector)
                print(f"检测方法切换为：{cfg.detector.method}")
                _log.info("用户切换检测方法为 %s", cfg.detector.method)
            elif key == ord("s"):
                stamp = time.strftime("%Y%m%d_%H%M%S")
                out_dir = ROOT / "outputs" / "images"
                out_dir.mkdir(parents=True, exist_ok=True)
                if last_frame is not None and last_vis is not None:
                    cv2.imwrite(str(out_dir / f"snapshot_{stamp}.png"), last_vis)
                    cv2.imwrite(str(out_dir / f"raw_{stamp}.png"), last_frame)
                    print(f"已保存快照：outputs/images/snapshot_{stamp}.png")
                    _log.info("用户保存快照 %s", stamp)
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
                    _log.info("录制%s：%s", "开始" if recording else "失败", path)
                else:
                    recording = False
                    if writer is not None:
                        writer.release()
                        writer = None
                    print("录制已停止")
                    _log.info("录制已停止")
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
            _log.info("平均每帧处理耗时 %.1f ms（约 %.1f FPS 的处理能力）",
                      avg * 1000, 1.0 / avg)
        # 结尾把三个口径一起打出来：cap 与 loop 差得远 = 瓶颈在检测，不是相机
        print(f"本次运行健康度：{health.summary_line()}")
        _report_health(force=True)
        _log.info("运行结束。本次健康度 %s", health.summary_line())
        # 收尾时把完整快照也落一份 —— 事后想画曲线/比对两次运行就不必再解析文本
        try:
            import json
            _log.info("健康度快照 %s", json.dumps(health.to_dict(), ensure_ascii=False))
        except Exception as exc:                              # noqa: BLE001
            _log.warning("健康度快照写不了：%s", exc)
        if health.dropped_frames:
            print(f"  其中 {health.dropped_frames} 帧在取走前被覆盖 —— "
                  f"说明主循环跟不上采集，看的是跳帧后的画面")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
