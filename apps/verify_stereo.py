"""复检双目标定：装置有没有被碰动？

标定一次可以用很久 —— 前提是**两台相机之间的相对位置没变**。
云台转动、支架被撞、相机被取下再装回，都会让 R,T 失效，
但表面上看不出任何异常（检测照样出结果，三维却系统性偏）。

这个工具用同一块标定板拍几帧，重新算**极线误差**并与标定时记录的值比较：

    python apps/verify_stereo.py --calib config/stereo_calibration --left 0 --right 2
    python apps/verify_stereo.py --calib config/stereo_calibration \
        --left-dir check/left --right-dir check/right

判定：极线误差仍在阈值内且与标定时的差异不大 -> 装置未变动；
否则说明相对位姿变了，需要重新标定。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# --- 让脚本在任意目录深度都能找到项目根 ---
for _parent in Path(__file__).resolve().parents:
    if (_parent / "sphere_tracker").is_dir():
        sys.path.insert(0, str(_parent))
        break

import cv2
import numpy as np

from apps._common import (
    add_stereo_source_args,
    calib_capture_size,
    open_pair_source,
)
from calib import BoardSpec, StereoRig
from calib.board import detect_board
from calib.capture import CaptureConfig, focus_gate
from stereo.fov_align import FovAlignment, estimate_fov_alignment

# 视频 / 图片输入是否已按标定 FOV 归一化：归一化后两路应接近同一视场，
# 首对帧重新估计时 scale 应接近 1。原生画面（1.5x 异视场）约 0.667 / 1.5。
_NORMALIZED_SCALE_MIN = 0.90
_NORMALIZED_SCALE_MAX = 1.10


def _check_normalized_input(left, right, *, assume_normalized: bool) -> int:
    """校验视频 / 图片首对帧是否已经是标定 FOV 归一化后的画面。

    返回 0 表示可以继续复检，2 表示输入不可信（调用方直接退出 2）。

    为什么不能只看尺寸：原生画面与归一化画面**尺寸完全一样**，肉眼与
    ``shape`` 都分不出来。这里对首对帧重估一次：归一化后两路应接近同视场、
    不需要主动裁剪；如果仍需要（或纹理不足估不出来），就失败关闭 ——
    不静默假定，也不对视频 / 图片二次裁剪（那会二次缩放）。
    """
    if assume_normalized:
        print("[FOV] --fov-input-normalized：按已归一化处理，跳过首对帧校验。")
        return 0
    try:
        probe = estimate_fov_alignment(left, right)
    except ValueError as exc:
        print(f"[错误] 无法验证视频/图片输入是否已按标定 FOV 归一化：{exc}")
        print("       首对帧共同纹理/特征不足，不能静默假定已归一化。若确认")
        print("       输入已归一化（例如用同一标定经 record_stereo.py 录制），")
        print("       请加 --fov-input-normalized。")
        return 2
    if not probe.active:
        print(f"[FOV] 视频/图片输入已归一化（首对帧 scale={probe.scale:.3f}，"
              "无需裁剪），不再二次裁剪。")
        return 0
    if not (_NORMALIZED_SCALE_MIN <= probe.scale <= _NORMALIZED_SCALE_MAX):
        print(f"[错误] 视频/图片输入看起来是未归一化的原生画面：首对帧仍需要"
              f"主动对齐（crop={probe.crop_side}，scale={probe.scale:.3f}，"
              f"应在 {_NORMALIZED_SCALE_MIN:.1f}–{_NORMALIZED_SCALE_MAX:.1f} 内）。")
        print("       本次复检不会自动裁剪视频/图片；请改用 record_stereo.py 以")
        print("       同一标定录制的归一化视频，或先离线归一化。")
        print("       若确认输入已经归一化，请加 --fov-input-normalized。")
        return 2
    print("[错误] 无法确认视频/图片输入已按标定 FOV 归一化：首对帧仍被判为"
          f"需要主动对齐（crop={probe.crop_side}，scale={probe.scale:.3f}，"
          f"裁剪区={probe.crop_rect}）。")
    print("       若确认输入已经归一化，请加 --fov-input-normalized；"
          "否则请重新录制。")
    return 2


def main() -> int:
    ap = argparse.ArgumentParser(description="复检双目标定（装置是否被碰动）")
    ap.add_argument("--calib", default="config/stereo_calibration",
                    help="标定文件前缀（不带扩展名）")
    add_stereo_source_args(ap)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--square", type=float, default=25.0)
    ap.add_argument("--samples", type=int, default=8, help="采集多少帧参与复检")
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--tolerance", type=float, default=0.15,
                    help="允许的极线误差增量（像素）")
    ap.add_argument("--min-sharpness", type=float, default=None,
                    help="额外的**绝对**清晰度下限（口径同标定：尺度无关的 focus）。"
                         "默认不启用 —— 绝对阈值没有刻度，真正起作用的判据在 "
                         "calib.capture.focus_gate：只跳过明显低于本组中位数 40%% 的帧")
    ap.add_argument("--fov-input-normalized", action="store_true",
                    help="视频/图片输入已按标定 FOV 归一化：跳过首对帧校验。"
                         "默认会验证；低纹理等无法验证时若确认输入已归一化再加此开关")
    args = ap.parse_args()

    # ---- 采集档：未显式给 --width/--height 时，**优先用标定记下的采集档** ----
    # 复检要套用标定的 FOV 裁剪，而裁剪区是相对**标定采集帧**定义的；采集帧
    # 比裁剪区小就会越界。所以这里优先按标定记录的采集档打开，与标定同档。
    if args.width <= 0 or args.height <= 0:
        _cs = calib_capture_size(args.calib)
        if _cs is not None:
            args.width, args.height = _cs
            print(f"[采集档] 未显式给 --width/--height，采用标定记录的采集档 "
                  f"{_cs[0]}x{_cs[1]}（与标定同一档才能正确套用 FOV 裁剪）")
        else:
            print("[采集档] 标定里没有采集档记录（旧标定），按相机自适应；"
                  "若复检异常请显式给 --width/--height")

    rig = StereoRig.load(args.calib)

    # 新标定在 meta 里记录了 FOV 对齐参数：复检必须先把两路归一到同一视场。
    # 左相机比右相机放大 ~1.5x，直接按原生画面找同名角点，像素尺度不一致，
    # 极线误差会被系统性算大（甚至检不出完整的板）。旧标定没有该键 ->
    # fov=None，走原来的路径，行为完全不变。
    fov: FovAlignment | None = None
    raw_fov = rig.meta.get("fov") if isinstance(rig.meta, dict) else None
    if raw_fov is not None:
        try:
            fov = FovAlignment.from_dict(raw_fov)
        except ValueError as exc:
            print(f"[错误] 标定 meta.fov 无法解析：{exc}")
            print("       这份标定的 FOV 记录已损坏，按错误视场复检会得到错误的"
                  "极线误差；已停止，请重新标定。")
            return 2
        if fov.active and tuple(fov.target_size) != tuple(rig.image_size):
            print(f"[错误] 标定 meta.fov.target_size {fov.target_size} 与 "
                  f"image_size {rig.image_size} 不一致，拒绝复检。")
            print("       这份标定的 FOV 记录与内参不匹配，请重新标定。")
            return 2
        if fov.active:
            mag = max(fov.scale, 1.0 / fov.scale)
            print(f"[FOV] 沿用标定对齐：参考 {fov.reference_side} / "
                  f"裁剪 {fov.crop_side} / 裁剪区 {fov.crop_rect} / "
                  f"放大 ~{mag:.2f}x -> {fov.target_size[0]}x{fov.target_size[1]}")
        else:
            print("[FOV] 标定记录两路视场一致（crop_side=none），不裁剪")
    else:
        print("[FOV] 旧标定没有 fov 记录：按原生画面复检（行为与以前一致）")

    print(rig.describe())
    # 两种模式的极线误差**不是同一把尺**（见 StereoRig.epipolar_metric_name）：
    # parallel 是校正后 y 差，angled 是 F 矩阵点线距。报告里必须写明口径，
    # 并且阈值不能混用 —— 否则 90° 装置会拿到一个好看但无意义的数字。
    metric = rig.epipolar_metric_name(rig.stereo_mode)
    print(f"复检口径：{metric}")
    ref = float(rig.quality.get("epipolar_rms", float("nan")))
    board_desc = rig.meta.get("board", "")
    print(f"标定时记录的极线误差：{ref:.4f} px"
          f"（口径：parallel 校正后 y 差 RMS，标定产物固定用该口径）")
    print()

    spec = BoardSpec(cols=args.cols, rows=args.rows, square_mm=args.square)
    if board_desc and f"{args.cols}x{args.rows}" not in board_desc:
        print(f"[提示] 标定时用的是「{board_desc}」，"
              f"现在按 {args.cols}x{args.rows} 检测，请确认参数一致")

    src = open_pair_source(args)
    if src is None:
        return 1
    errs: list[float] = []
    frames = 0
    checked_fov_input = False
    # 与标定收帧**同一套**清晰度判据（calib.capture.focus_gate）：
    # 绝对阈值没有刻度（同一组数据里 focus 随对比度成倍变化），所以这里也
    # 用"组内相对"——只跳过明显比本组中位数糊的帧。同一份数据不会出现
    # "采集放行、复检拒收"。
    cap_cfg = CaptureConfig()
    focus_hist = {"左": [], "右": []}
    focus_skips = 0
    last_focus_reason = ""
    try:
        while len(errs) < args.samples and frames < args.max_frames:
            ok, left, right = src.read()
            frames += 1
            if not ok or left is None or right is None:
                if src.is_live:
                    continue
                break
            if fov is not None and src.kind == "camera":
                try:
                    left, right = fov.apply(left, right)
                except ValueError as exc:
                    print(f"[错误] 无法应用标定记录的 FOV 对齐：{exc}")
                    print("       实时相机画面无法按标定 FOV 归一化；请重新标定。")
                    return 1
            elif fov is not None and fov.active and not checked_fov_input:
                # 视频 / 图片在首对帧上一次性证明“已经是归一化画面”；
                # 验证不了或仍需要主动对齐都退出 2，不静默假定、不二次裁剪。
                checked_fov_input = True
                code = _check_normalized_input(
                    left, right, assume_normalized=args.fov_input_normalized)
                if code != 0:
                    return code
            gl = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            gr = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
            ol = detect_board(gl, spec)
            orr = detect_board(gr, spec)
            if not (ol.found and orr.found):
                continue
            # 尺度无关的清晰度（focus = 棋盘 ROI 的 Laplacian 方差 × 格子像素数），
            # 与标定收帧同一口径；不要用 sharpness —— 板一大就会被误判成糊。
            ok_focus = True
            for name, obs in (("左", ol), ("右", orr)):
                hist = focus_hist[name]
                ok_one, why_one = focus_gate(
                    obs.focus, label=name,
                    median_focus=float(np.median(hist)) if hist else float("nan"),
                    n_samples=len(hist),
                    min_ratio=cap_cfg.min_focus_ratio,
                    floor=cap_cfg.focus_floor,
                    min_samples=cap_cfg.focus_min_samples,
                    abs_min=args.min_sharpness,
                )
                if not ok_one:
                    ok_focus = False
                    last_focus_reason = why_one
                    break
            if not ok_focus:
                focus_skips += 1
                continue
            focus_hist["左"].append(float(ol.focus))
            focus_hist["右"].append(float(orr.focus))
            errs.append(rig.epipolar_error_by_mode(rig.stereo_mode, ol.corners, orr.corners))
    finally:
        src.release()

    if focus_skips:
        print(f"[提示] 因清晰度跳过 {focus_skips} 帧（例：{last_focus_reason}）")

    if not errs:
        print("[失败] 没有采到有效的标定板图像（两路都要能看到完整的板）")
        return 1

    now = float(np.sqrt(np.mean(np.square(errs))))
    worst = float(np.max(errs))
    print(f"本次复检：{len(errs)} 帧，{metric} RMS {now:.4f} px（最大 {worst:.4f} px）")

    if rig.stereo_mode == "angled":
        # 没有任何依据说明 angled 点线距 "多少 px 算装置被碰动"：
        # 它随标定板到相机的距离、视场、畸变残差一起变，和 parallel 的 0.6 px
        # 不是一回事，也**不能**与标定时记录的 y 差 RMS 相减比较。
        # 所以只报数、不判通过/不通过（宁可不判，也不瞎判）。
        print("标定当时的数字（parallel 口径）与本口径不可比，不计算变化量。")
        print("[不判定] angled 模式：本工具只报告极线点线距，不做通过/不通过判定。")
        print("         如需确认装置是否移动，请重新采集标定板并重标定后对比 rms。")
        return 0

    print(f"标定当时：{ref:.4f} px")
    delta = now - ref
    print(f"变化量：{delta:+.4f} px")
    print()

    ok = True
    print("[说明] 下面的 0.6 px 与 --tolerance 都是按 parallel 的 y 差口径定的阈值，"
          "只在本口径下有效。")
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
