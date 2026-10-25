"""按空格收一张标定板照片 —— 给"单板链式标定"的 4 段引导流程用。

为什么单独做一个工具
--------------------
链式标定要按固定顺序拍 4 组图（左内参 / 右内参 / 左参考 / 右参考），
每组都得**当场确认**"整块板进画面了、手停了、没糊"再按空格。GUI 需要一个
可被 subprocess 调用、能用退出码表达结果的收图器，而不是复制一份标定窗口。

判据沿用 ``calib.board`` 与 ``calib.capture`` 的口径，不另起一套：
  * 整块棋盘检出 —— ``calib.board.detect_board``（SB，绝不混用经典检测器）；
  * 静止 —— 相对上一帧角点的**最大**位移 ≤ ``--stationary-px``；
  * 清晰度 —— 棋盘区域 Laplacian 方差 ≥ ``--min-sharpness``（同 record_stereo）。

逐帧做的是**粗检**（降分辨率、不开 EXHAUSTIVE，约几十毫秒）；
真正要落盘的那一帧再全分辨率精确复核 —— 和标定窗口的做法一致，
否则 2560x1440 上每帧 300+ ms，窗口会卡到没法对准。

用法::

    python apps/tools/capture_board.py --out data/calib/L/intr --side left
    python apps/tools/capture_board.py --out /tmp/x --side left \\
        --source video:tmp/board.mp4 --no-show --target 3

退出码：0=收够 target；3=用户提前退出 / 视频放完没到 target；1=打不开设备或目录；2=参数错。
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "apps"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from _common import log_environment, setup_logging  # noqa: E402
from calib.board import BoardObservation, BoardSpec, detect_board  # noqa: E402
from sphere_tracker import camera_registry as cr  # noqa: E402
from sphere_tracker.sources import CameraSource, FileSource  # noqa: E402

log = logging.getLogger("capture_board")

# 粗检用的宽度：960 宽在 1440p 上约 30~50 ms，够实时对准；精确复核仍用全分辨率。
GATE_WORK_WIDTH = 960


def judge_capture(found: bool, still_px: float, sharpness: float,
                  stationary_px: float, min_sharpness: float) -> tuple[bool, str]:
    """这一帧能不能存？返回 (可存, 不可存的原因)。

    纯函数：不碰相机、不建窗口、不读文件 —— GUI 与自测都直接调它。
    阈值必须由入参给（``--stationary-px`` / ``--min-sharpness``），这里不写死，
    否则命令行改了参数而判据没跟着变，是最难查的一类不一致。

    顺序有意：**先报"没检出"再报"在动"再报"糊"**。
    板都没看见时说"太糊"会把人引去擦镜头，实际只是板不在画面里。
    """
    if not found:
        return False, "没检出整块棋盘"
    if still_px > stationary_px:
        return False, f"还在动 {still_px:.1f} px"
    if sharpness < min_sharpness:
        return False, f"太糊 {sharpness:.1f}"
    return True, ""


def corner_shift_px(prev: np.ndarray | None, cur: np.ndarray | None) -> float:
    """两张图的同名角点最大位移（像素）。任一缺失返回 ``inf``（视为"没法判静止"）。

    用**最大**而不是均值：均值会被 40 个几乎不动的角点稀释，
    板一端翘起来（手在转）也照样能过门限 —— 那正是最该拦下的帧。
    """
    if prev is None or cur is None:
        return float("inf")
    a = np.asarray(prev, np.float32).reshape(-1, 2)
    b = np.asarray(cur, np.float32).reshape(-1, 2)
    if a.shape != b.shape or len(a) == 0:
        return float("inf")
    return float(np.max(np.linalg.norm(a - b, axis=1)))


def _backend_const(name: str):
    """CLI 的后端名 -> cv2 常量；``auto`` 交给 CameraSource 自己成对挑。"""
    if not name or name == "auto":
        return None
    return {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}.get(name.lower())


def _open_source(args):
    """按 ``--source`` / ``--camera`` / ``--side`` 打开一路输入。

    失败抛 RuntimeError/FileNotFoundError，由 ``main()`` 统一转成退出码 1。
    """
    if args.source:
        raw = str(args.source)
        path = raw[len("video:"):] if raw.lower().startswith("video:") else raw
        return FileSource(path)

    index = cr.resolve_camera_arg(args.camera or args.side, interactive=False)
    if index is None:
        raise RuntimeError(f"没能把 {args.camera or args.side!r} 解析成相机索引；"
                           f"先跑 python apps/tools/cameras.py list --probe")
    return CameraSource(index=index, width=args.width, height=args.height,
                        fps=args.fps, backend=_backend_const(args.backend))


def _draw_overlay(frame: np.ndarray, spec: BoardSpec, obs: BoardObservation,
                  count: int, target: int, reason: str, ok: bool) -> np.ndarray:
    """把角点、进度、当前原因画在画面上（窗口模式用；不改原帧）。"""
    view = frame.copy()
    if obs.found and obs.corners is not None:
        cv2.drawChessboardCorners(view, spec.pattern_size, obs.corners, True)
    color = (80, 220, 120) if ok else (60, 180, 255)
    cv2.putText(view, f"{count}/{target}", (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, color, 2, cv2.LINE_AA)
    cv2.putText(view, (reason or "可以保存：按空格"), (12, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.putText(view, "空格 保存   q/ESC 退出", (12, view.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    return view


def main() -> int:
    ap = argparse.ArgumentParser(
        description="按空格收标定板照片（单板链式标定的 4 段引导流程用）")
    ap.add_argument("--out", required=True, help="保存目录（不存在则创建）")
    ap.add_argument("--side", required=True, choices=["left", "right"],
                    help="用哪一路相机（left/right 查 config/cameras.json 的角色）")
    ap.add_argument("--camera", default=None,
                    help="显式相机：索引 / 角色名 / name:...；默认取 --side 的角色")
    ap.add_argument("--backend", choices=["auto", "dshow", "msmf"], default="auto",
                    help="相机后端（默认 auto，由 CameraSource 成对挑）")
    ap.add_argument("--width", type=int, default=2560, help="采集宽度（默认 2560）")
    ap.add_argument("--height", type=int, default=1440, help="采集高度（默认 1440）")
    ap.add_argument("--fps", type=int, default=30, help="采集帧率（默认 30）")
    ap.add_argument("--cols", type=int, default=8, help="标定板内角点列数（默认 8）")
    ap.add_argument("--rows", type=int, default=5, help="标定板内角点行数（默认 5）")
    ap.add_argument("--square", type=float, default=25.0, help="方格边长 mm（默认 25）")
    ap.add_argument("--target", type=int, default=15, help="存够这么多张就退出（默认 15）")
    ap.add_argument("--stationary-px", type=float, default=3.0,
                    help="连续两帧角点最大位移上限（px，默认 3.0）")
    ap.add_argument("--min-sharpness", type=float, default=40.0,
                    help="棋盘区域 Laplacian 方差下限（默认 40.0）")
    ap.add_argument("--source", default=None,
                    help="视频源：video:<路径> 或直接给路径；只读视频、不建窗口")
    ap.add_argument("--timeout", type=float, default=0.0,
                    help="最多跑多少秒（0=不限；视频源下按秒计时）")
    ap.add_argument("--no-show", action="store_true", help="不建窗口（自测/后台）")
    args = ap.parse_args()

    spec = BoardSpec(cols=args.cols, rows=args.rows, square_mm=args.square)
    out_dir = Path(args.out).expanduser().resolve()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"[错误] 保存目录不可用：{out_dir}（{exc}）")
        return 1

    setup_logging("capture_board")
    log_environment(log)
    log.info("输出目录 %s；板 %s；target=%d", out_dir, spec.describe(), args.target)

    # 有 --source 一律不建窗口：视频没有键盘交互，保存只能自动触发。
    # 无窗口（--no-show / 视频源）时**判据通过即自动存**，否则没有任何保存途径。
    show = not (args.no_show or args.source)

    try:
        src = _open_source(args)
    except Exception as exc:                                     # noqa: BLE001
        print(f"[错误] 打不开相机/视频：{(str(exc).splitlines() or [type(exc).__name__])[0]}")
        return 1
    log.info("输入 %s %dx%d fps=%.0f %s %s", src.info.kind, src.info.width,
             src.info.height, src.info.fps, src.info.backend, src.info.path)

    saved: list[Path] = []
    prev_corners: np.ndarray | None = None
    last_cat = ""
    last_print = 0.0
    t0 = time.perf_counter()

    try:
        while True:
            if args.timeout > 0 and time.perf_counter() - t0 >= args.timeout:
                log.info("到 --timeout %.1f s，停止", args.timeout)
                break
            ok, frame, _ts = src.read()
            if not ok or frame is None:
                log.info("输入结束（视频放完或相机断流）")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 粗检：降分辨率 + 不开 EXHAUSTIVE，保证窗口跟得上手
            obs = detect_board(gray, spec, work_width=GATE_WORK_WIDTH, exhaustive=False)
            still = corner_shift_px(prev_corners, obs.corners if obs.found else None)
            can, reason = judge_capture(obs.found, still, obs.sharpness,
                                        args.stationary_px, args.min_sharpness)
            if obs.found:
                # 只有检出时才更新基准帧：板短暂丢失（手挡了一下）不该把
                # 下一帧判成"还在动"，否则用户会莫名其妙地被卡住。
                prev_corners = obs.corners.copy()

            if not can:
                now = time.perf_counter()
                # 不刷屏：原因类别变了、或距上次打印超过 0.5 s 才输出一行
                cat = reason.split()[0]
                if cat != last_cat or now - last_print > 0.5:
                    log.info("跳过：%s", reason)
                    last_cat = cat
                    last_print = now

            pressed = -1
            if show:
                view = _draw_overlay(frame, spec, obs, len(saved), args.target,
                                     "" if can else reason, can)
                cv2.imshow("capture_board", view)
                pressed = cv2.waitKey(1) & 0xFF
                if pressed in (ord("q"), 27):
                    log.info("用户退出（q/ESC）")
                    break
            if not can:
                continue

            # 真要落盘：全分辨率精确复核（这一步才贵，所以只在候选帧上做）
            if show and pressed != 32:
                continue
            acc = detect_board(gray, spec, exhaustive=True)
            if not acc.found:
                log.info("跳过：精确复核没检出整块棋盘")
                continue
            can_acc, reason_acc = judge_capture(True, still, acc.sharpness,
                                                args.stationary_px, args.min_sharpness)
            if not can_acc:
                log.info("跳过：精确复核 %s", reason_acc)
                continue

            name = f"shot_{len(saved) + 1:02d}_{int(time.time() * 1000)}.png"
            path = out_dir / name
            if not cv2.imwrite(str(path), frame):
                print(f"[错误] 写不进文件：{path}")
                return 1
            saved.append(path)
            print(f"已保存 {len(saved)}/{args.target}  {name}  "
                  f"（清晰度 {acc.sharpness:.1f}，位移 {still:.1f} px）")
            log.info("保存 %s（清晰度 %.1f，位移 %.1f px）", path, acc.sharpness, still)
            if len(saved) >= args.target:
                break
    finally:
        src.release()
        if show:
            cv2.destroyAllWindows()

    print(f"共保存 {len(saved)} 张 → {out_dir}")
    if len(saved) < 3:
        print("至少需要 3 张才能标定，建议 15 张覆盖画面四角")
    return 0 if len(saved) >= args.target else 3


if __name__ == "__main__":
    raise SystemExit(main())
