"""收标定板照片 —— 给"单板链式标定"的 4 段引导流程用。

为什么单独做一个工具
--------------------
链式标定要按固定顺序拍 4 组图（左内参 / 右内参 / 左参考 / 右参考），
每组都得**当场确认**"整块板进画面了、手停了、没糊"再按空格。GUI 需要一个
可被 subprocess 调用、能用退出码表达结果的收图器，而不是复制一份标定窗口。

判据沿用 ``calib.board`` 与 ``calib.capture`` 的口径，不另起一套：
  * 整块棋盘检出 —— ``calib.board.detect_board``（SB，绝不混用经典检测器）；
  * 静止 —— 相对上一帧角点的**最大**位移 ≤ ``--stationary-px``；
  * 清晰度 —— **尺度无关的 focus**，判据用 ``calib.capture.focus_gate``
    （与 calibrate_stereo 同一条规则，不是照抄一份）：
    ``focus = 棋盘 ROI 的 Laplacian 方差 × 格子像素数``；只拒**明显比本组中位数糊**
    的帧（默认低于中位数 40%）与低于绝对地板的帧。旧口径（裸 Laplacian 方差）
    会随棋盘占的像素数漂移，板一大就被误判成"糊"，已弃用。

逐帧做的是**粗检**（降分辨率、不开 EXHAUSTIVE，约几十毫秒）；
真正要落盘的那一帧再全分辨率精确复核 —— 和标定窗口的做法一致，
否则 2560x1440 上每帧 300+ ms，窗口会卡到没法对准。

三种用法
--------
1. 单路按键（原来那条路，GUI 的 4 段引导用它）::

    python apps/tools/capture_board.py --out data/calib/L/intr --side left

2. 单路自动（板举着不用按键；判据连稳 N 帧才存，存完冷却一段时间）::

    python apps/tools/capture_board.py --out data/calib/L/intr --side left --auto

3. 双路成对（左右各存一张，同一时间戳；两路都检出整块板才存）::

    python apps/tools/capture_board.py --both --out-left data/L --out-right data/R

自测（视频代替相机，不建窗口）::

    python apps/tools/capture_board.py --out /tmp/x --side left \\
        --source video:tmp/board.mp4 --no-show --target 3
    python apps/tools/capture_board.py --both --out-left /tmp/l --out-right /tmp/r \\
        --source-left video:tmp/board.mp4 --source-right video:tmp/board.mp4 --no-show

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
from calib.capture import CaptureConfig, focus_gate  # noqa: E402
from sphere_tracker import camera_registry as cr  # noqa: E402
from sphere_tracker.sources import CameraSource, FileSource  # noqa: E402

log = logging.getLogger("capture_board")

# 清晰度判据的默认参数**取自 CaptureConfig**，不在这里另写一份数字 ——
# 否则标定窗口与这个收图器会慢慢漂成两套阈值。
_FOCUS_DEFAULTS = CaptureConfig()

# 粗检用的宽度：960 宽在 1440p 上约 30~50 ms，够实时对准；精确复核仍用全分辨率。
GATE_WORK_WIDTH = 960

# 双路并排窗口里每一路的高度（两路缩放后 hstack，窗口别撑爆屏幕）。
BOTH_VIEW_HEIGHT = 720


def judge_capture(found: bool, still_px: float, focus: float,
                  stationary_px: float, min_sharpness: float | None,
                  median_focus: float = float("nan"), n_samples: int = 0,
                  min_ratio: float | None = None, floor: float | None = None,
                  min_samples: int | None = None) -> tuple[bool, str]:
    """这一帧能不能存？返回 (可存, 不可存的原因)。

    纯函数：不碰相机、不建窗口、不读文件 —— GUI 与自测都直接调它。
    阈值必须由入参给（``--stationary-px`` / ``--min-sharpness``），这里不写死，
    否则命令行改了参数而判据没跟着变，是最难查的一类不一致。

    清晰度**不是**这里自己判的：直接调 ``calib.capture.focus_gate``，
    与标定窗口**同一条规则**（绝对地板 + 相对中位数）。两处各写一套，
    早晚会分叉成"采集放行、标定拒收"。

    ``focus`` 是**尺度无关**的清晰度（``BoardObservation.focus`` =
    Laplacian 方差 × 格子像素数）。别塞 ``sharpness`` 进来 ——
    那个量随棋盘在画面里的大小变化，板一大就会被误判成"糊"。

    顺序有意：**先报"没检出"再报"在动"再报"糊"**。
    板都没看见时说"太糊"会把人引去擦镜头，实际只是板不在画面里。
    """
    if not found:
        return False, "没检出整块棋盘"
    if still_px > stationary_px:
        return False, f"还在动 {still_px:.1f} px"
    return focus_gate(
        focus, median_focus=median_focus, n_samples=n_samples,
        min_ratio=_FOCUS_DEFAULTS.min_focus_ratio if min_ratio is None else min_ratio,
        floor=_FOCUS_DEFAULTS.focus_floor if floor is None else floor,
        min_samples=(_FOCUS_DEFAULTS.focus_min_samples if min_samples is None
                     else min_samples),
        abs_min=min_sharpness,
    )


def should_autosave(stable_frames: int, auto_hold: int, since_last_s: float,
                    auto_gap: float) -> tuple[bool, str]:
    """自动模式：这一帧该自动存吗？返回 (该存, 不该存的原因)。

    纯函数，阈值全来自入参 —— ``--auto-hold`` / ``--auto-gap`` 改了要能被测出来。

    两道闸门各有各的用处，缺一不可：
      * ``auto_hold``：判据"连续"满足多少帧。单帧判据会被**手在等的时候的
        微小抖动**骗过去（板端部抖 2 px 但瞬时停在门限内），连稳 N 帧才是真停稳；
      * ``auto_gap``：两张之间的最小间隔。没有它，同一个姿势连稳 100 帧就存 100 张 ——
        用户看着"存够了"，但全是同一视角，标定退化。

    等于阈值算通过（``>=``）：用户说 6 帧就是"第 6 帧起可以存"。
    """
    need = max(1, int(auto_hold))
    if stable_frames < need:
        return False, f"稳定 {stable_frames}/{need} 帧"
    gap = max(0.0, float(auto_gap))
    if since_last_s < gap:
        return False, f"冷却中 {gap - since_last_s:.1f} s"
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


def gate_frame(gray: np.ndarray, spec: BoardSpec, prev_corners: np.ndarray | None,
               stationary_px: float, min_sharpness: float | None,
               focus_hist: list[float] | None = None
               ) -> tuple[BoardObservation, float, bool, str, np.ndarray | None]:
    """粗检一帧并给出判据：返回 (观测, 位移, 可存, 原因, 新基准帧)。

    只有检出时才更新基准帧：板短暂丢失（手挡了一下）不该把下一帧判成"还在动"，
    否则用户会莫名其妙地被卡住。

    ``focus_hist`` 是**已存下**各帧的 focus（相对判据的基准）。只看已存下的帧：
    被拒的帧不该把中位数拉低，否则"越来越糊"的一组会互相掩护着全进来。
    """
    hist = focus_hist if focus_hist is not None else []
    obs = detect_board(gray, spec, work_width=GATE_WORK_WIDTH, exhaustive=False)
    still = corner_shift_px(prev_corners, obs.corners if obs.found else None)
    can, reason = judge_capture(
        obs.found, still, obs.focus, stationary_px, min_sharpness,
        median_focus=float(np.median(hist)) if hist else float("nan"),
        n_samples=len(hist))
    if obs.found:
        prev_corners = obs.corners.copy()
    return obs, still, can, reason, prev_corners


def _backend_const(name: str):
    """CLI 的后端名 -> cv2 常量；``auto`` 交给 CameraSource 自己成对挑。"""
    if not name or name == "auto":
        return None
    return {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}.get(name.lower())


def _open_source(source: str | None, camera: str | None, args,
                 side: str | None = None):
    """按 ``video:``/``--source*`` / ``--camera*`` / 角色打开一路输入。

    失败抛 RuntimeError/FileNotFoundError，由调用方统一转成退出码 1。
    """
    if source:
        raw = str(source)
        path = raw[len("video:"):] if raw.lower().startswith("video:") else raw
        return FileSource(path)

    want = camera or side
    index = cr.resolve_camera_arg(want, interactive=False)
    if index is None:
        raise RuntimeError(f"没能把 {want!r} 解析成相机索引；"
                           f"先跑 python apps/tools/cameras.py list --probe")
    return CameraSource(index=index, width=args.width, height=args.height,
                        fps=args.fps, backend=_backend_const(args.backend))


def _draw_overlay(frame: np.ndarray, spec: BoardSpec, obs: BoardObservation,
                  count: int, target: int, reason: str, ok: bool,
                  hint: str = "空格 保存   q/ESC 退出") -> np.ndarray:
    """把角点、进度、当前原因画在画面上（窗口模式用；不改原帧）。"""
    view = frame.copy()
    if obs.found and obs.corners is not None:
        cv2.drawChessboardCorners(view, spec.pattern_size, obs.corners, True)
    color = (80, 220, 120) if ok else (60, 180, 255)
    cv2.putText(view, f"{count}/{target}", (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, color, 2, cv2.LINE_AA)
    cv2.putText(view, (reason or "可以保存"), (12, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.putText(view, hint, (12, view.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    return view


def _side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """两路画面缩到同一高度后左右拼接（窗口里一眼看出是哪一路不满足）。"""
    def fit(img: np.ndarray) -> np.ndarray:
        h = img.shape[0]
        scale = BOTH_VIEW_HEIGHT / float(h) if h else 1.0
        return cv2.resize(img, (int(round(img.shape[1] * scale)), BOTH_VIEW_HEIGHT))
    return np.hstack([fit(left), fit(right)])


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="收标定板照片（单板链式标定的 4 段引导流程用）")
    ap.add_argument("--out", default=None, help="保存目录（不存在则创建）；--both 时改用 --out-left/--out-right")
    ap.add_argument("--side", default=None, choices=["left", "right"],
                    help="用哪一路相机（left/right 查 config/cameras.json 的角色）；--both 时忽略")
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
    ap.add_argument("--target", type=int, default=15,
                    help="存够这么多张就退出（--both 下按对数算，默认 15）")
    ap.add_argument("--stationary-px", type=float, default=3.0,
                    help="连续两帧角点最大位移上限（px，默认 3.0）")
    ap.add_argument("--min-sharpness", type=float, default=800.0,
                    help="额外的**绝对**清晰度下限，比的是尺度无关的 focus"
                         "（=Laplacian 方差 × 格子像素数，默认 800.0）。"
                         "绝对阈值没有刻度，真正起作用的是相对判据：focus 低于"
                         "已存帧中位数 40%% 才判糊（阈值取自 CaptureConfig）；越小越宽松")
    ap.add_argument("--auto", action="store_true",
                    help="停稳自动存（不用按空格）")
    ap.add_argument("--auto-hold", type=int, default=6,
                    help="判据连续满足这么多帧才自动存（默认 6）")
    ap.add_argument("--auto-gap", type=float, default=1.2,
                    help="两次自动存的最小间隔秒数（默认 1.2）")
    ap.add_argument("--both", action="store_true",
                    help="双路同时采集：两路都检出整块板才成对保存（需 --out-left/--out-right）")
    ap.add_argument("--out-left", default=None, help="--both：左路保存目录")
    ap.add_argument("--out-right", default=None, help="--both：右路保存目录")
    ap.add_argument("--camera-left", default=None, help="--both：左路相机（默认取 left 角色）")
    ap.add_argument("--camera-right", default=None, help="--both：右路相机（默认取 right 角色）")
    ap.add_argument("--source", default=None,
                    help="视频源：video:<路径> 或直接给路径；只读视频（--both 时用 --source-left/--source-right）")
    ap.add_argument("--source-left", default=None, help="--both 自测：左路视频源")
    ap.add_argument("--source-right", default=None, help="--both 自测：右路视频源")
    ap.add_argument("--timeout", type=float, default=0.0,
                    help="最多跑多少秒（0=不限；视频源下按秒计时）")
    ap.add_argument("--no-show", action="store_true", help="不建窗口（自测/后台）")
    return ap


def _check_args(ap: argparse.ArgumentParser, args) -> None:
    """参数组合的合法性 —— 全部走 ``ap.error``（退出码 2）。"""
    if args.both:
        if args.out:
            ap.error("--both 用 --out-left/--out-right 两条目录，不能再给 --out")
        if args.source:
            ap.error("--both 与 --source 互斥；双路自测请用 --source-left/--source-right")
        if not (args.out_left and args.out_right):
            ap.error("--both 需要 --out-left 和 --out-right")
        if (args.source_left is None) != (args.source_right is None):
            ap.error("--source-left 与 --source-right 要一起给（两路都读视频）")
        return
    if args.out_left or args.out_right:
        ap.error("--out-left/--out-right 只用于 --both")
    if args.source_left or args.source_right:
        ap.error("--source-left/--source-right 只用于 --both 自测")
    if not args.out:
        ap.error("缺 --out（单路）；双路请用 --both --out-left … --out-right …")
    if not args.side:
        ap.error("缺 --side（left/right）")


def _auto_settings(args) -> tuple[int, float]:
    return max(1, int(args.auto_hold)), max(0.0, float(args.auto_gap))


def _make_dir(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path).expanduser().resolve()
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"[错误] 保存目录不可用：{p}（{exc}）")
        return None
    return p


def _run_single(args, spec: BoardSpec, out_dir: Path) -> int:
    # 有 --source 一律不建窗口；无窗口时**只能自动存**，否则没有任何保存途径。
    show = not (args.no_show or args.source)
    auto = bool(args.auto) or not show
    hold, gap = _auto_settings(args)

    try:
        src = _open_source(args.source, args.camera, args, side=args.side)
    except Exception as exc:                                     # noqa: BLE001
        print(f"[错误] 打不开相机/视频：{(str(exc).splitlines() or [type(exc).__name__])[0]}")
        return 1
    log.info("输入 %s %dx%d fps=%.0f %s %s", src.info.kind, src.info.width,
             src.info.height, src.info.fps, src.info.backend, src.info.path)
    if auto:
        log.info("自动保存：连稳 %d 帧、间隔 %.1f s（空格仍可随时存）", hold, gap)

    saved: list[Path] = []
    prev_corners: np.ndarray | None = None
    focus_hist: list[float] = []      # 已存帧的 focus（相对判据的基准）
    stable = 0
    last_save_ts: float | None = None
    last_cat = ""
    last_print = 0.0
    t0 = time.perf_counter()

    try:
        while True:
            if args.timeout > 0 and time.perf_counter() - t0 >= args.timeout:
                log.info("到 --timeout %.1f s，停止", args.timeout)
                break
            ok, frame, ts = src.read()
            if not ok or frame is None:
                log.info("输入结束（视频放完或相机断流）")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            obs, still, can, reason, prev_corners = gate_frame(
                gray, spec, prev_corners, args.stationary_px, args.min_sharpness,
                focus_hist)
            stable = stable + 1 if can else 0

            since = float("inf") if last_save_ts is None else ts - last_save_ts
            go, auto_reason = (should_autosave(stable, hold, since, gap)
                               if (auto and can) else (False, reason))

            pressed = -1
            if show:
                hint = (f"自动：连稳 {hold} 帧 / 间隔 {gap:.1f} s   空格 立即存   q/ESC 退出"
                        if auto else "空格 保存   q/ESC 退出")
                view = _draw_overlay(frame, spec, obs, len(saved), args.target,
                                     "" if can else reason, can, hint=hint)
                cv2.imshow("capture_board", view)
                pressed = cv2.waitKey(1) & 0xFF
                if pressed in (ord("q"), 27):
                    log.info("用户退出（q/ESC）")
                    break

            if not can:
                # 不刷屏：原因类别变了、或距上次打印超过 0.5 s 才输出一行
                now = time.perf_counter()
                cat = reason.split()[0]
                if cat != last_cat or now - last_print > 0.5:
                    log.info("跳过：%s", reason)
                    last_cat = cat
                    last_print = now
                continue

            manual = show and pressed == 32
            if not (go or manual):
                now = time.perf_counter()
                # 「冷却中 x.x s」必须打出来：用户看到"板停住了却没存"，
                # 唯一的解释就在这一行里。
                cat = auto_reason.split()[0]
                if cat != last_cat or now - last_print > 0.5:
                    log.info("不存：%s", auto_reason)
                    last_cat = cat
                    last_print = now
                continue

            # 真要落盘：全分辨率精确复核（这一步才贵，所以只在候选帧上做）
            acc = detect_board(gray, spec, exhaustive=True)
            if not acc.found:
                log.info("跳过：精确复核没检出整块棋盘")
                continue
            can_acc, reason_acc = judge_capture(
                True, still, acc.focus, args.stationary_px, args.min_sharpness,
                median_focus=float(np.median(focus_hist)) if focus_hist else float("nan"),
                n_samples=len(focus_hist))
            if not can_acc:
                log.info("跳过：精确复核 %s", reason_acc)
                continue

            name = f"shot_{len(saved) + 1:02d}_{int(time.time() * 1000)}.png"
            path = out_dir / name
            if not cv2.imwrite(str(path), frame):
                print(f"[错误] 写不进文件：{path}")
                return 1
            saved.append(path)
            focus_hist.append(float(acc.focus))
            verb = "自动保存" if not manual else "已保存"
            print(f"{verb} {len(saved)}/{args.target}  {name}  "
                  f"（focus {acc.focus:.0f}，原始清晰度 {acc.sharpness:.0f}，位移 {still:.1f} px）")
            log.info("保存 %s（focus %.0f，原始清晰度 %.0f，格子 %.1fpx，位移 %.1f px）",
                     path, acc.focus, acc.sharpness, acc.square_px, still)
            stable = 0
            last_save_ts = ts
            if len(saved) >= args.target:
                break
    finally:
        src.release()
        if show:
            cv2.destroyAllWindows()

    print(f"共保存 {len(saved)} 张 → {out_dir}")
    if 0 < len(saved) < 3:
        print("至少需要 3 张才能标定，建议 15 张覆盖画面四角")
    return 0 if len(saved) >= args.target else 3


def _run_both(args, spec: BoardSpec, out_left: Path, out_right: Path) -> int:
    """双路成对采集：两路都满足判据才存，文件名共用同一毫秒时间戳。

    为什么要"同时"：外参只在**同一时刻**的一对图之间成立。
    分别拍两段再对，只要中间把板/相机碰了一下，标定就会悄悄错 —— 而且看不出来。
    """
    show = not args.no_show
    hold, gap = _auto_settings(args)
    src_l = src_r = None
    try:
        src_l = _open_source(args.source_left, args.camera_left, args, side="left")
        src_r = _open_source(args.source_right, args.camera_right, args, side="right")
    except Exception as exc:                                     # noqa: BLE001
        print(f"[错误] 打不开相机/视频：{(str(exc).splitlines() or [type(exc).__name__])[0]}")
        if src_l is not None:
            src_l.release()
        return 1
    log.info("左路 %s %dx%d；右路 %s %dx%d", src_l.info.kind, src_l.info.width,
             src_l.info.height, src_r.info.kind, src_r.info.width, src_r.info.height)
    if (src_l.info.width, src_l.info.height) != (src_r.info.width, src_r.info.height):
        log.warning("两路分辨率不一致（%dx%d vs %dx%d）—— 标定要求一致",
                    src_l.info.width, src_l.info.height,
                    src_r.info.width, src_r.info.height)
    log.info("双路自动保存：连稳 %d 帧、间隔 %.1f s，目标 %d 对", hold, gap, args.target)

    saved = 0
    prev_l: np.ndarray | None = None
    prev_r: np.ndarray | None = None
    focus_hist_l: list[float] = []      # 两路各记各的（两路画质可以差很多）
    focus_hist_r: list[float] = []
    stable = 0
    last_save_ts: float | None = None
    last_cat = ""
    last_print = 0.0
    t0 = time.perf_counter()

    try:
        while True:
            if args.timeout > 0 and time.perf_counter() - t0 >= args.timeout:
                log.info("到 --timeout %.1f s，停止", args.timeout)
                break
            ok_l, fl, ts_l = src_l.read()
            ok_r, fr, ts_r = src_r.read()
            if not ok_l or fl is None or not ok_r or fr is None:
                log.info("输入结束（有一路放完或断流）")
                break

            gl = cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY)
            gr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            obs_l, still_l, can_l, why_l, prev_l = gate_frame(
                gl, spec, prev_l, args.stationary_px, args.min_sharpness, focus_hist_l)
            obs_r, still_r, can_r, why_r, prev_r = gate_frame(
                gr, spec, prev_r, args.stationary_px, args.min_sharpness, focus_hist_r)
            both_ok = can_l and can_r
            stable = stable + 1 if both_ok else 0
            ts = max(ts_l, ts_r)
            since = float("inf") if last_save_ts is None else ts - last_save_ts
            go, auto_reason = (should_autosave(stable, hold, since, gap)
                               if both_ok else (False, why_l or why_r))

            pressed = -1
            if show:
                vl = _draw_overlay(fl, spec, obs_l, saved, args.target,
                                   "左 OK" if can_l else f"左：{why_l}", can_l,
                                   hint="空格 立即存一对   q/ESC 退出")
                vr = _draw_overlay(fr, spec, obs_r, saved, args.target,
                                   "右 OK" if can_r else f"右：{why_r}", can_r,
                                   hint="空格 立即存一对   q/ESC 退出")
                cv2.imshow("capture_board (both)", _side_by_side(vl, vr))
                pressed = cv2.waitKey(1) & 0xFF
                if pressed in (ord("q"), 27):
                    log.info("用户退出（q/ESC）")
                    break

            if not both_ok:
                now = time.perf_counter()
                cat = (why_l or why_r).split()[0]
                if cat != last_cat or now - last_print > 0.5:
                    log.info("跳过：左 %s｜右 %s", why_l or "OK", why_r or "OK")
                    last_cat = cat
                    last_print = now
                continue

            manual = show and pressed == 32
            if not (go or manual):
                now = time.perf_counter()
                cat = auto_reason.split()[0]
                if cat != last_cat or now - last_print > 0.5:
                    log.info("不存：%s", auto_reason)
                    last_cat = cat
                    last_print = now
                continue

            # 两路都精确复核：任一路不过就不存这一对（宁可少一对，不要错一对）
            acc_l = detect_board(gl, spec, exhaustive=True)
            acc_r = detect_board(gr, spec, exhaustive=True)
            ok_l2, why_l2 = judge_capture(
                acc_l.found, still_l, acc_l.focus, args.stationary_px, args.min_sharpness,
                median_focus=float(np.median(focus_hist_l)) if focus_hist_l else float("nan"),
                n_samples=len(focus_hist_l))
            ok_r2, why_r2 = judge_capture(
                acc_r.found, still_r, acc_r.focus, args.stationary_px, args.min_sharpness,
                median_focus=float(np.median(focus_hist_r)) if focus_hist_r else float("nan"),
                n_samples=len(focus_hist_r))
            if not (ok_l2 and ok_r2):
                log.info("跳过：精确复核 左 %s｜右 %s", why_l2 or "OK", why_r2 or "OK")
                continue

            n = saved + 1
            ms = int(time.time() * 1000)
            name = f"pair_{n:02d}_{ms}.png"
            pl, pr = out_left / name, out_right / name
            if not (cv2.imwrite(str(pl), fl) and cv2.imwrite(str(pr), fr)):
                print(f"[错误] 写不进文件：{pl} 或 {pr}")
                return 1
            saved = n
            focus_hist_l.append(float(acc_l.focus))
            focus_hist_r.append(float(acc_r.focus))
            print(f"已存第 {n} 对（左 {n} / 右 {n}）  {name}  "
                  f"（focus 左 {acc_l.focus:.0f} / 右 {acc_r.focus:.0f}；"
                  f"原始清晰度 {acc_l.sharpness:.0f} / {acc_r.sharpness:.0f}）")
            log.info("保存一对 %s", name)
            stable = 0
            last_save_ts = ts
            if saved >= args.target:
                break
    finally:
        src_l.release()
        src_r.release()
        if show:
            cv2.destroyAllWindows()

    print(f"共保存 {saved} 对 → {out_left} ; {out_right}")
    return 0 if saved >= args.target else 3


def main() -> int:
    ap = _build_parser()
    args = ap.parse_args()
    _check_args(ap, args)

    spec = BoardSpec(cols=args.cols, rows=args.rows, square_mm=args.square)

    if args.both:
        out_left = _make_dir(args.out_left)
        out_right = _make_dir(args.out_right)
        if out_left is None or out_right is None:
            return 1
        setup_logging("capture_board")
        log_environment(log)
        log.info("左目录 %s；右目录 %s；板 %s；target=%d 对",
                 out_left, out_right, spec.describe(), args.target)
        return _run_both(args, spec, out_left, out_right)

    out_dir = _make_dir(args.out)
    if out_dir is None:
        return 1
    setup_logging("capture_board")
    log_environment(log)
    log.info("输出目录 %s；板 %s；target=%d", out_dir, spec.describe(), args.target)
    return _run_single(args, spec, out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
