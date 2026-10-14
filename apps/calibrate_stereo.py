"""自动双目标定：把标定板在两台相机前挥一遍就行。

用法::

    # 1) 两台实时相机（推荐：能实时看到收帧进度）
    #    用**角色名**最省心：绑定的是设备身份，插拔 USB 口也不受影响。
    python apps/calibrate_stereo.py --left left --right right --cols 8 --rows 5 --square 25

    #    要用编号也行，但**编号的含义随后端变化**：解析用的设备顺序是 PnP 枚举，
    #    而 cv2 各后端的顺序不一样（本机实测 DSHOW 的 #1 是笔记本内置，
    #    MSMF 的 #2 才是）。本程序默认固定 MSMF，所以这里给的是 MSMF 口径的编号。
    #    不确定就先用角色名，或跑 python apps/tools/cameras.py list 看当前设备。

    # 2) 两段视频（指导文档里的流程：先分别录，再拿来标定）
    python apps/calibrate_stereo.py --left-video cam1.mp4 --right-video cam2.mp4 \
        --cols 8 --rows 5 --square 25

    # 3) 两个图片文件夹（每张都是独立位姿；此时不要求"板静止"）
    python apps/calibrate_stereo.py --left-dir imgs/left --right-dir imgs/right

程序会：
  - 自动判断每一帧该不该收（清晰度、大小、贴边、曝光、位姿多样性、覆盖均匀）
  - 收够之后自动标定，并用**极线误差**等指标判定能不能用
  - 把结果存成 config/stereo_calibration.npz + .json（含人读的质量报告）

关于同步：**标定时不需要左右视频同步**。程序只收"板停稳"的帧，
静止目标在任意时刻看起来都一样，所以时间偏移不影响标定。
（真正需要同步的是后续"拍运动的小球"，那是另一回事。）

帧源（摄像头 / 两段视频 / 两个图片文件夹）统一由 `stereo.PairSource` 提供 ——
复检工具与以后的双目实时流程共用同一份取帧逻辑，不再各写一份。
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# --- 让脚本在任意目录深度都能找到项目根 ---
for _parent in Path(__file__).resolve().parents:
    if (_parent / "sphere_tracker").is_dir():
        sys.path.insert(0, str(_parent))
        break

import cv2
import numpy as np

from apps._window import WindowManager
from apps._common import (
    add_stereo_source_args,
    log_environment,
    open_pair_source,
    setup_logging,
)
from calib import AutoPairCollector, BoardSpec, CalibConfig, StereoRig, calibrate_stereo
from sphere_tracker.sources import quiet_repeated_grab_warnings

WINDOW = "stereo calibration"
PREVIEW_W = 720          # 单路预览宽度的**上限**（实际按屏幕与文字条带反推，见主循环）
FOOTER_LINE_H = 19       # 文字条带每行高度（像素）


def _screen_size() -> tuple[int, int]:
    """当前主屏分辨率；取不到就退回 1920x1080。

    为什么要它：面板比屏幕高的时候，``cv2.imshow`` 会把底部切掉 ——
    而被切掉的恰好是"为什么收不满"的那几条统计。所以布局必须知道屏幕多大，
    宁可把图像缩小，也不能让文字消失。

    用 ctypes 直接问系统，不引入第三方依赖；非 Windows 或调用失败时退回默认值。
    """
    try:
        import ctypes

        user32 = ctypes.windll.user32
        w, h = int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    return 1920, 1080


# --------------------------------------------------------------------------
def build_report_figure(rig: StereoRig, pairs, spec: BoardSpec, out_path: Path) -> None:
    """画一张验证图：把左右两路的角点放到同一张画布上（左半 / 右半）。

    极线误差是个抽象像素数，这张图让它变得可见：
    校正正确时，**左右对应的角点应该落在同一条水平线上**，
    所以这里把每一对对应角点用横线连起来 —— 线是水平的就说明校正对了。
    """
    if not pairs:
        return
    panels = []
    for k, pair in enumerate(pairs[:3]):
        pl, pr = rig.rectify_points(pair.corners_left, pair.corners_right)
        h, w = rig.image_size[1], rig.image_size[0]
        canvas = np.zeros((h, w * 2 + 10, 3), np.uint8)
        for y in range(0, h, 60):
            cv2.line(canvas, (0, y), (canvas.shape[1], y), (60, 60, 66), 1)
        n = min(len(pl), len(pr))
        max_dy = 0.0
        for i in range(n):
            x1, y1 = pl[i]
            x2, y2 = pr[i]
            max_dy = max(max_dy, abs(y1 - y2))
            cv2.line(canvas, (int(x1), int(y1)), (int(x2) + w + 10, int(y2)), (90, 200, 90), 1)
        for pts, off in ((pl, 0), (pr, w + 10)):
            for x, y in pts:
                cv2.circle(canvas, (int(x) + off, int(y)), 3, (0, 220, 255), -1)
        cv2.putText(canvas, f"pair {k + 1}: left | right   max |dy| = {max_dy:.3f} px",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1, cv2.LINE_AA)
        panels.append(canvas)
    fig = np.vstack(panels) if len(panels) > 1 else panels[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), fig)
    print(f"验证图已保存：{out_path}（左右对应角点连成的线越水平，校正越准）")


def main() -> int:
    quiet_repeated_grab_warnings()
    ap = argparse.ArgumentParser(description="自动双目标定")
    add_stereo_source_args(ap)
    ap.add_argument("--cols", type=int, default=8, help="棋盘格内角点列数")
    ap.add_argument("--rows", type=int, default=5, help="棋盘格内角点行数")
    ap.add_argument("--square", type=float, default=25.0, help="方格实际边长（毫米）")
    # 默认值 = **试标定档（8 对）**，由用户 2026-10-10 指定，用于先跑通通路。
    #
    # 代价必须说清：`CalibConfig` 的注释写着"默认值按 20-30 对图像调好的"，
    # 重投影/极线门槛（good 0.35 px、ok 0.80 px）也是按那个量级定的。
    # 8 对能解出外参（内参是固定的），但：
    #   · 离群剔除会让剩余低于 8 时**停止剔除**（stereo_calib.py 的门槛），
    #     所以坏数据可能留在里面，RMS 偏高；
    #   · 姿态覆盖必然不足，边角区域的畸变约束不住 —— 三维在画面边角会偏。
    # **拿它验证"通路能不能跑通"可以，拿它出正式三维结果不行。**
    #
    # ⚠️ **默认值已从 8/8 改为 25/10**（2026-10-13，用户要求）。
    # 原来的 8/8 是"试标定档"，但用户从 GUI 走时根本不会去加参数，
    # 于是**永远停在试标定档**却以为在做正式标定 —— 那比没有默认值更糟。
    # 现在默认就是正式档；只想验证通路请显式给 `--target 8`。
    ap.add_argument("--target", type=int, default=25,
                    help="目标收多少对（默认 25 = 正式；只想验证通路可给 8）")
    ap.add_argument("--min-pairs", type=int, default=10,
                    help="最少收多少对才尝试标定（默认 10）")
    ap.add_argument("--baseline-mm", type=float, default=None, help="预期基线（毫米），用于核对")
    ap.add_argument("--out", default="config/stereo_calibration", help="输出前缀")
    ap.add_argument("--save-pairs", default=None, help="把收下的图像对另存到这个目录（便于复查）")
    ap.add_argument("--report-png", default=None, help="输出验证图路径")
    ap.add_argument("--no-show", action="store_true", help="不弹窗（批处理）")
    ap.add_argument("--max-frames", type=int, default=0, help=">0 时最多处理这么多帧")
    ap.add_argument("--stationary-px", type=float, default=None,
                    help="判定'板停稳'的帧间位移上限（像素，默认 3）。"
                         "手持板够不着就调大，但别超过 ~12：这条门限同时替代了左右同步，"
                         "放太松会让两路拍到不同位姿，解出的 R/T 会系统性偏")
    ap.add_argument("--min-sharpness", type=float, default=None,
                    help="棋盘区域清晰度下限（Laplacian 方差，默认 55）。调低会收进更糊的帧")
    ap.add_argument("--relaxed", action="store_true",
                    help="手持拍摄的宽松预设：停稳上限 10 px、清晰度下限 28、"
                         "允许更小的板。**换来过检风险**，精度不满意就退回默认值重标")
    args = ap.parse_args()

    # 日志要在**做任何事之前**起来：标定失败时最有价值的证据（拒绝了什么、
    # 各拒了多少帧）现在只在内存里，进程一退就没了。
    log_path = setup_logging("calibrate")
    log = logging.getLogger("calibrate")
    print(f"本次日志：{log_path}")
    log.info("=== 双目标定开始 ===")
    log.info("参数：left=%s right=%s backend=%s %dx%d@%d 目标 %d 对（最少 %d）",
             args.left or args.left_video or args.left_dir,
             args.right or args.right_video or args.right_dir,
             args.backend, args.width, args.height, args.fps,
             args.target, args.min_pairs)
    log_environment(log)
    if args.relaxed:
        log.warning("启用了 --relaxed：门限放松，用精度换成功率")

    spec = BoardSpec(cols=args.cols, rows=args.rows, square_mm=args.square)
    print(f"标定板：{spec.describe()}")
    if args.target < 15:
        # 低目标数必须**显式可见**，不能悄悄用：否则以后看到一份差标定，
        # 没人知道是因为只收了几对。
        print(f"⚠ 试标定档：只收 {args.target} 对（默认/正式档 25 对）。")
        print("   内参是固定的，少几对也能解出外参，但姿态覆盖不足、离群剔除会被下限卡住 ——")
        print("   **够验证通路，不够出正式三维结果**。要正式标定去掉 --target（默认 25）")
        print()
        log.warning("试标定档：target=%d（正式档 25）", args.target)
    log.info("标定板：%s", spec.describe())
    print("提示：把板举到**两路都能看到**的位置，慢慢移动，每个姿势停一下（程序只收停稳的帧）")
    print()

    src = open_pair_source(args)
    if src is None:
        log.error("取帧源装配失败，退出")
        return 1
    print(src.describe())
    log.info("取帧源：%s", src.describe())
    src.report_warnings()
    for w in getattr(src, "warnings", []) or []:
        log.warning("取帧源告警：%s", w)
    from calib import CaptureConfig
    cap_kw: dict = dict(
        target_pairs=args.target,
        min_pairs=args.min_pairs,
        # 独立静帧场景不要求"板静止"（帧间本来就是不同位姿）
        require_stationary=(src.kind != "dir"),
    )
    if args.relaxed:
        # 手持预设。**每一条都是"用精度换成功率"**，所以逐条说清代价：
        cap_kw.update(
            stationary_max_px=10.0,   # 允许手更抖；代价=两路位姿差变大
            min_sharpness=28.0,       # 允许更糊；代价=角点定位变差
            min_area_ratio=0.008,     # 允许板更远；代价=单像素误差占比变大
            min_margin_px=6.0,        # 允许更贴边；代价=边缘畸变没被约束住
            min_brightness=25.0,
            max_brightness=245.0,
        )
        print("⚠ 已启用 --relaxed（手持预设）：门限都放松了，**这是用精度换成功率**。")
        print("   如果标定完 极线误差 > 0.3 px，或者代进去三维抖得厉害，")
        print("   请处理好光照与支撑、去掉 --relaxed 重标一次。")
        print()
    if args.stationary_px is not None:
        cap_kw["stationary_max_px"] = float(args.stationary_px)
    if args.min_sharpness is not None:
        cap_kw["min_sharpness"] = float(args.min_sharpness)
    cap_cfg = CaptureConfig(**cap_kw)
    collector = AutoPairCollector(spec, cap_cfg)
    # 全屏/窗口尺寸统一由 apps/_window.py 管（各程序口径一致）
    wins = WindowManager()

    cov_cache = None
    cov_cache_n = -1
    det_ms = 0.0
    t_prev = time.perf_counter()
    fps_frames = 0
    pairs_dir: Path | None = Path(args.save_pairs) if args.save_pairs else None
    if pairs_dir:
        (pairs_dir / "left").mkdir(parents=True, exist_ok=True)
        (pairs_dir / "right").mkdir(parents=True, exist_ok=True)

    frames = 0
    # ⚠️ **读不到帧必须能自己收场**（2026-10-14 修的真缺陷）。
    #
    # 原来（`if src.is_live: continue`）在相机掉线时会**永远转下去**，
    # 而且**一个字都不打印** —— 用户看到的就是"标定程序卡死了"，
    # 只能去关启动器（而关窗口并不会杀掉子进程）。
    #
    # 相机掉线的信号很明确（`-2147483638` = MF_E_VIDEO_RECORDING_DEVICE_INVALIDATED，
    # 实测每 10 秒一次），所以：先报，再等，超时就带着已经收到的对数退出，
    # 让**已经收下的图像对不白收**。
    miss = 0
    miss_reported = False
    try:
        while True:
            ok, left, right = src.read()
            if not ok or left is None or right is None:
                if not src.is_live:
                    break
                miss += 1
                if miss == 30 and not miss_reported:
                    miss_reported = True
                    why = ""
                    health = getattr(src, "health_line", None)
                    if callable(health):
                        why = f"（{health()}）"
                    print(f"\n⚠ 读不到画面了{why}。已收 {collector.count} 对。", flush=True)
                    print("   常见原因：相机被别的程序占用 / USB 掉线 / 掉电。"
                          "拔插一次 USB 往往就好。", flush=True)
                    print("   再等 5 秒；还是不行就保存已收的对并退出。", flush=True)
                if miss >= 150:                      # 约 5 秒（30fps）
                    print(f"\n✘ 连续 5 秒读不到画面，停止采集。"
                          f"已收 {collector.count} 对（目标 {args.target}）。", flush=True)
                    break
                time.sleep(0.02)
                continue
            miss = 0
            miss_reported = False
            frames += 1
            if args.max_frames and frames > args.max_frames:
                break

            t0 = time.perf_counter()
            gl = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            gr = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
            accepted, reason = collector.offer(gl, gr)
            det_ms = 0.9 * det_ms + 0.1 * (time.perf_counter() - t0) * 1000.0
            fps_frames += 1

            if accepted and pairs_dir:
                i = collector.count
                cv2.imwrite(str(pairs_dir / "left" / f"{i:03d}.png"), left)
                cv2.imwrite(str(pairs_dir / "right" / f"{i:03d}.png"), right)

            if not args.no_show:
                # 直接复用粗检结果画框 —— 之前这里又检测了一遍，
                # 每帧 4 次全图精确检测（约 1.2 秒）就是卡顿的根源。
                #
                # ---- 布局：先算文字，图像用剩下的空间 ----
                #
                # 为什么这么排：原来状态文字压在图像上（逐路状态在右上角、
                # 拒绝统计在左下角），而且面板比窗口高，底部那几条**被裁掉** ——
                # 被裁掉的恰好是"为什么收不满"的答案。
                # 所以现在把两路图像 + 覆盖率图放上面，所有文字放**下面独立的条带**里，
                # 并且反过来让图像去适应文字，而不是让文字去挤图像。
                footer_lines = (
                    collector.status_lines()
                    + collector.side_lines()
                    + [f"detect {det_ms:.0f} ms/frame  ({1000.0 / max(det_ms, 1e-3):.0f} fps)"
                       f"   [q] 退出  [r] 清空重收  [f] 全屏（Esc 退出全屏）"]
                )
                footer_h = len(footer_lines) * FOOTER_LINE_H + 10

                sw, sh = _screen_size()
                cov_w = 260          # 覆盖率面板宽度：要放得下宫格占用数字与棋盘四边形
                per_cam_w = max(240, (sw - 60 - cov_w) // 2)
                # 取两路里更"瘦"的宽高比，保证两路都放得下（两路分辨率可能不同）
                ar = min(left.shape[1] / left.shape[0], right.shape[1] / right.shape[0])
                img_h = int(min(PREVIEW_W / ar, per_cam_w / ar, sh - 90 - footer_h))
                img_h = max(200, img_h)

                def _prep(img, target_h):
                    s = target_h / float(img.shape[0])
                    out = cv2.resize(
                        img,
                        (max(1, int(round(img.shape[1] * s))), target_h),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    return out, s

                a, scale_l = _prep(left, img_h)
                b, scale_r = _prep(right, img_h)
                # 宽高比不同时补黑边，而不是拉伸（拉伸会让角点坐标与图上对不上）
                for img in (a, b):
                    if img.shape[0] != img_h:
                        img = np.vstack([img, np.zeros((img_h - img.shape[0], img.shape[1], 3), np.uint8)])
                for img, obs, sc in ((a, collector.last_obs[0], scale_l),
                                     (b, collector.last_obs[1], scale_r)):
                    if obs is None or not obs.found or obs.corners is None:
                        continue
                    colour = (0, 220, 0) if collector.last_accepted else (0, 140, 255)
                    quad = obs.quad()
                    if quad is not None:
                        cv2.polylines(img, [(quad * sc).astype(np.int32)], True, colour, 2)
                    pts = obs.corners.reshape(-1, 2) * sc
                    for x, y in pts:
                        cv2.circle(img, (int(x), int(y)), 2, (0, 255, 255), -1)

                # 在两半上各打一个 LEFT / RIGHT 标记（含指定的相机编号/角色名）。
                # 为什么要标：这个窗口是用户**判断左右有没有接反**的唯一现场依据 ——
                # 他会在某台摄像头前晃手，看哪一半跟着动。没有标记就只能靠猜
                # "左半边到底是不是 left"。
                for img, tag in ((a, f"LEFT  ({args.left})"), (b, f"RIGHT ({args.right})")):
                    (tw, th), _base = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(img, (0, 0), (tw + 16, th + 14), (0, 0, 0), -1)
                    cv2.putText(img, tag, (8, th + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 255, 0), 2, cv2.LINE_AA)

                # 覆盖率图只在收帧数变化时重画（原来每帧都重建一次画布）
                if collector.count != cov_cache_n:
                    # **给输出画布尺寸**（不是源图尺寸）—— 这样位姿一定会被缩放并
                    # 居中画进面板里。曾经把源尺寸当成输出尺寸传，导致姿态画到画布外，
                    # 面板看起来是空的（用户："看不到已经标定的姿态和位置"）。
                    cov_cache = collector.coverage_image(out_size=(cov_w, img_h))
                    cov_cache_n = collector.count
                cov = cov_cache
                if cov is None or cov.shape[:2] != (img_h, cov_w):
                    cov = collector.coverage_image(out_size=(cov_w, img_h))
                row = np.hstack([a, b, cov])

                # 文字条带：整幅宽度，画在图像**下方**，永远不会压住画面
                footer = np.zeros((footer_h, row.shape[1], 3), np.uint8)
                for i, line in enumerate(footer_lines):
                    cv2.putText(footer, line, (10, 18 + i * FOOTER_LINE_H),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)
                panel = np.vstack([row, footer])
                wins.show(WINDOW, panel)
                key = cv2.waitKey(1) & 0xFF
                if wins.handle_key(WINDOW, key, panel):
                    continue
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    collector.reset()
                    print("已清空重新收集")

            if collector.full:
                print(f"已收满 {collector.count} 对，开始标定…")
                break
            if src.is_live and frames % 60 == 0:
                now = time.perf_counter()
                fps = fps_frames / max(now - t_prev, 1e-6)
                t_prev = now
                fps_frames = 0
                print(f"  已收 {collector.count}/{args.target}  {fps:5.1f} fps  "
                      f"检测 {det_ms:.0f} ms/帧  ({collector.last_reason})")
    finally:
        src.release()
        if not args.no_show:
            cv2.destroyAllWindows()

    print()
    print(f"共处理 {frames} 帧，收下 {collector.count} 对")
    # 结束时的**完整**统计：不再只印前三条，也不再只留在内存里。
    # 用户卡在"收了 8 对就不动"时，能看到的证据就是这里。
    print()
    for line in collector.summary_lines():
        print(line)
    collector.log_summary()
    log.info("采集结束：%d 对（目标 %d，最少 %d）", collector.count, args.target, args.min_pairs)

    if not collector.enough:
        hint = collector.next_step_hint()
        print(f"\n[失败] 只收到 {collector.count} 对，少于 {args.min_pairs} 对，无法标定。")
        if hint:
            print(f"下一步：{hint}")
        print(f"（完整逐帧记录见日志：{log_path}）")
        return 1

    objp = spec.object_points()
    ip1 = [p.corners_left for p in collector.pairs]
    ip2 = [p.corners_right for p in collector.pairs]
    img_size = collector.image_size

    cfg = CalibConfig(expected_baseline_mm=args.baseline_mm)
    res = calibrate_stereo(objp, ip1, ip2, img_size, cfg, spec)
    print(res.report())
    log.info("标定结果：\n%s", res.report())

    if args.report_png:
        try:
            rig_tmp = StereoRig.from_result(res)
            build_report_figure(rig_tmp, collector.pairs, spec, Path(args.report_png))
        except ValueError:
            pass

    if not res.ok:
        print("标定未通过质量判定，未保存结果。请按上面的提示重拍。")
        log.warning("标定未通过质量判定，未保存")
        return 1

    rig = StereoRig.from_result(res, meta={
        "board": spec.describe(),
        "source": src.kind,
        "left": args.left if args.left is not None else (args.left_video or args.left_dir),
        "right": args.right if args.right is not None else (args.right_video or args.right_dir),
        "image_size": list(img_size),
    })
    npz, js = rig.save(args.out)
    print(f"已保存：{npz}")
    print(f"        {js}")
    print()
    print("下一步：")
    print("  - 三角测量时用 calib.StereoRig.load('config/stereo_calibration') 读取参数")
    print("  - 装置如果被碰动，用 apps/verify_stereo.py 复检极线误差")
    log.info("标定成功，已保存 %s", npz)
    print(f"（本次日志：{log_path}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
