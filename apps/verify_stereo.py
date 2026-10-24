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

from apps._common import add_stereo_source_args, open_pair_source
from calib import BoardSpec, StereoRig
from calib.board import detect_board
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
    ap.add_argument("--fov-input-normalized", action="store_true",
                    help="视频/图片输入已按标定 FOV 归一化：跳过首对帧校验。"
                         "默认会验证；低纹理等无法验证时若确认输入已归一化再加此开关")
    args = ap.parse_args()

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
    ref = float(rig.quality.get("epipolar_rms", float("nan")))
    board_desc = rig.meta.get("board", "")
    print(f"标定时记录的极线误差：{ref:.4f} px")
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
