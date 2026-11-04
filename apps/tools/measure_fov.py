"""量一次两路的**固有视场差**，存进装置档案，之后各处复用。

为什么需要这个工具
------------------
两台同型号相机装上去，出厂焦距并不一样 —— 本机实测左路比右路放大 ~1.5 倍
（`stereo/fov_align.py` 的模块注释记着这个数）。这个差异是**装置属性**：

- 它不会因为换一块标定板、换一次场景而改变；
- 但它是**像素量纲**的（scale / crop_rect / target_size），换分辨率就失效；
- 而它现在只活在"某一次双目标定"的 `meta.fov` 里 —— 重新标定就没了，
  标定时特征不够就变成 `null`，90° 装置和单目更是完全看不到它。

所以把这个测量**从标定里拆出来**：先量一次、存进装置档案，之后
平行标定/录制可以直接复用，不必每次重新猜。

三种输入（都要求两路是**同一时刻的同一场景**，否则单应没有意义）
--------------------------------------------------------------
    # 1) 一次双目录制的同一帧（最实用：left.mkv / right.mkv 本来就是同步的）
    python apps/tools/measure_fov.py --left-video data/stereo/<会话>/left.mkv \
        --right-video data/stereo/<会话>/right.mkv --frame 30

    # 2) 两张图（同一次拍摄的左右各一张）
    python apps/tools/measure_fov.py --left L.png --right R.png

    # 3) 量完直接写进装置档案（不写就只打印 + 可选 --json）
    python apps/tools/measure_fov.py --left-video ... --right-video ... \
        --write-profile 平行装置

退出码：0 = 估计成功；1 = 估计失败（纹理不足/重叠不够，见打印的原因）；
2 = 参数或文件不对。

⚠️ 90° 装置不用它做裁剪：两路光轴近乎垂直、视场几乎不重叠，单应裁剪在那种
几何下没有物理意义（`apps/record_stereo.py` 在 angled 下明确忽略 fov）。
量它仍然有用 —— 它回答"这台机器的两路差多少"，单目推 fx、判断有没有换过
镜头时都要这个数。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2

# 按**标记**上溯到含 `sphere_tracker/` 的祖先，不数层数（`selftest_entrypoints`
# 对这条是硬的：数层数在被移动/加包装目录时会静默指错）。
ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "sphere_tracker").is_dir())
sys.path.insert(0, str(ROOT))

from stereo.fov_align import (                            # noqa: E402
    MIN_FOV_INLIERS, MIN_FOV_QUALITY, estimate_fov_alignment,
    fov_confidence_problem)

SCHEMA = "sphere-fov-alignment"

#: 写进档案 `fov` 的键（与 `stereo/profiles._check_fov` 认的集合一致）。
_PROFILE_KEYS = ("reference_side", "crop_side", "crop_rect", "target_size",
                 "scale", "method", "inliers", "quality",
                 "left_size", "right_size", "measured_at", "source")


def _read_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"图像读不出来：{path}")
    return img


def _read_video_frame(path: Path, frame: int):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"视频打不开：{path}")
    try:
        if frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame))
        ok, img = cap.read()
        if not ok or img is None:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            raise ValueError(
                f"读不出第 {frame} 帧（{path}，该视频共 {total} 帧）"
                + ("；把 --frame 改小一点" if total and frame >= total else ""))
        return img
    finally:
        cap.release()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="量两路相机的固有视场差（scale / 裁剪参数），可存进装置档案")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--left", help="左路图像文件")
    src.add_argument("--left-video", help="左路视频（与 --right-video 配套）")
    ap.add_argument("--right", help="右路图像文件（与 --left 配套）")
    ap.add_argument("--right-video", help="右路视频（与 --left-video 配套）")
    ap.add_argument("--frame", type=int, default=30,
                    help="从视频取第几帧（默认 30，避开开头的黑帧/自动曝光）")
    ap.add_argument("--json", default="", help="把结果写成 JSON（可选）")
    ap.add_argument("--write-profile", default="",
                    help="把结果写进这个装置档案的 fov 字段（名字要已存在）")
    ap.add_argument("--profiles", default="",
                    help="档案文件路径（默认 config/profiles.json）")
    ap.add_argument("--min-inliers", type=int, default=MIN_FOV_INLIERS,
                    help=f"RANSAC 内点下限（默认 {MIN_FOV_INLIERS}）")
    args = ap.parse_args(argv)

    # ---- 取两路帧 ----------------------------------------------------------
    try:
        if args.left:
            if not args.right:
                print("[参数错误] 给了 --left 就要给 --right")
                return 2
            left = _read_image(Path(args.left))
            right = _read_image(Path(args.right))
            source = f"images:{Path(args.left).name}+{Path(args.right).name}"
        else:
            if not args.right_video:
                print("[参数错误] 给了 --left-video 就要给 --right-video")
                return 2
            left = _read_video_frame(Path(args.left_video), args.frame)
            right = _read_video_frame(Path(args.right_video), args.frame)
            source = f"video:{Path(args.left_video).parent.name}@{args.frame}"
    except ValueError as exc:
        print(f"[错误] {exc}")
        return 2

    h_l, w_l = left.shape[:2]
    h_r, w_r = right.shape[:2]
    print("=" * 72)
    print("两路视场差测量")
    print("=" * 72)
    print(f"  左 {w_l}x{h_l}   右 {w_r}x{h_r}   来源 {source}")

    # ---- 估计 --------------------------------------------------------------
    try:
        fov = estimate_fov_alignment(left, right, min_inliers=args.min_inliers)
    except ValueError as exc:
        print(f"\n[错误] 估计失败：{exc}")
        print("\n可能的原因（按常见程度）：")
        print("  1) 两路不是同一时刻的画面 —— 手会被拍成两个位置，任何单应都拟合不上")
        print("  2) 场景太平（白墙、桌面）—— 没有可匹配的特征点")
        print("  3) 两路几乎没有共同视场 —— 那本来就不该做 FOV 对齐")
        print("  4) 画面太模糊/过曝")
        print("\n换一帧（--frame）或换个有纹理的场景再量一次。")
        return 1

    payload = fov.to_dict()
    payload.update({
        "left_size": [w_l, h_l],
        "right_size": [w_r, h_r],
        "measured_at": datetime.now().isoformat(timespec="seconds"),
        "source": source,
    })

    print()
    print(f"  参考路      : {fov.reference_side}（较窄/较放大的一路）")
    print(f"  被裁剪路    : {fov.crop_side}" + ("" if fov.active else "（无需对齐）"))
    print(f"  尺度 left→right: {fov.scale:.4f}"
          + (f"（右图 = 左图 × {fov.scale:.3f}，即左路放大 {1 / fov.scale:.2f}×）"
             if abs(fov.scale - 1.0) > 0.01 else "（两路视场基本一致）"))
    print(f"  裁剪矩形    : {tuple(fov.crop_rect)}（在 {fov.crop_side} 的像素系里）")
    print(f"  共同输出尺寸: {tuple(fov.target_size)}")
    print(f"  估计质量    : {fov.method} / inliers {fov.inliers} / quality {fov.quality:.3f}")

    problem = fov_confidence_problem(fov)
    if problem:
        print(f"\n[警告] 可信度偏低：{problem}")
        print(f"       下限是 quality≥{MIN_FOV_QUALITY} / inliers≥{MIN_FOV_INLIERS}。"
              "这个结果只适合参考，不建议直接写进档案。")
    else:
        print("\n  可信度：达到下限（可以写进档案）")

    # ---- 落盘 --------------------------------------------------------------
    result = {"schema": SCHEMA, "version": 1, **payload}
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        print(f"  JSON 已写：{out}")

    if args.write_profile:
        import dataclasses

        from stereo import profiles
        path = Path(args.profiles) if args.profiles else None
        try:
            current = profiles.get_profile(args.write_profile, path)
        except profiles.ProfileError as exc:
            print(f"\n[错误] 档案 {args.write_profile!r} 记录损坏：{exc}")
            return 2
        if current is None:
            print(f"\n[错误] 档案 {args.write_profile!r} 不存在。")
            print("       先在门厅「装置档案」里建这条档案，"
                  "或去掉 --write-profile 只打印。")
            return 2
        updated = dataclasses.replace(
            current, fov={k: payload[k] for k in _PROFILE_KEYS if k in payload})
        try:
            profiles.upsert(updated, replace_existing=True, path=path)
        except profiles.ProfileError as exc:
            print(f"\n[错误] 写档案失败：{exc}")
            return 2
        print(f"  已写入档案「{args.write_profile}」的 fov 字段")
        print("       之后该装置标定/录制会优先用它（不用每次重新估）")

    if not args.json and not args.write_profile:
        print("\n（只打印，没有落盘：加 --json 出文件，或 --write-profile 存进档案）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
