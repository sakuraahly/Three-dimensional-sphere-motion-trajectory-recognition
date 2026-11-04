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


def session_fov_record(video_path: Path) -> dict | None:
    """同目录 ``session.json`` 里记录的 ``rig.fov``（录制时抄下来的标定参数）。

    为什么它比自己量更可信：那段记录是**标定当时在原始画面上**算的；
    而录像本身可能已经按它裁剪过——在裁剪后的画面上再量一次，量到的是
    "残余视场差"，是个看起来质量很高、实际完全错的数（实测 quality 0.988）。
    """
    js = Path(video_path).parent / "session.json"
    try:
        data = json.loads(js.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    rig = data.get("rig")
    fov = rig.get("fov") if isinstance(rig, dict) else None
    return fov if isinstance(fov, dict) and fov else None


def already_cropped_problem(video_path: Path, size: tuple[int, int]) -> str:
    """这段录像是不是**已经按 FOV 裁剪过**？是则返回一句可执行的原因。

    判据：``session.json`` 的 ``rig.fov.target_size`` 与视频分辨率相同 ——
    `apps/record_stereo.py` 就是用 `fov.target_size` 作为写盘尺寸的，所以
    相等就意味着"这段画面是裁完的"。

    在这种画面上再量一次单应，得到的是**二次裁剪**的残余差：数字自洽、
    质量指标还很高，但把它写进档案、下次标定拿去裁原始画面，裁剪框会落在
    完全错误的位置。所以这里**拒绝**，而不是给个警告继续算。
    """
    rec = session_fov_record(video_path)
    if not rec:
        return ""
    target = rec.get("target_size")
    try:
        tgt = tuple(int(v) for v in target) if target else ()
    except (TypeError, ValueError):
        return ""
    if len(tgt) == 2 and tgt == tuple(int(v) for v in size):
        return (
            f"这段录像是**已经按 FOV 裁剪过**的 {size[0]}x{size[1]}"
            f"（session.json 记录 target_size={list(tgt)}）。\n"
            f"  在裁剪后的画面上再量一次得到的是**二次裁剪**的残余差：\n"
            f"  数字自洽、质量指标还很高，但拿去裁原始画面会让裁剪框落错位置。\n"
            f"  下一步：要么直接用 session.json 里那份记录（它是标定当时在原始\n"
            f"        画面上算的），要么重新标定一次（标定时会现场估并自动记录）。")
    return ""


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


def _expected_raw_size(profile) -> tuple[int, int] | None:
    """从档案配的那份标定里取 ``image_size`` —— ``crop_rect`` 的坐标系就是它。

    为什么必须拿它来核对：`crop_rect` 是**原始画面**上的像素坐标，而
    session.json / 部分标定记录里只留了裁剪后的尺寸。2026-11-04 实测：
    `20261027-232108` 记着 `crop_rect=[404,195,1718,973]`，`404+1718=2122`，
    而当前标定是 1920 宽 —— 那份记录来自旧分辨率（约 2560x1440），
    直接套上去裁剪框会**越界**。写进档案时就要拦住，不能等标定那天才炸。
    """
    base = getattr(profile, "calibration_path", None)
    if not base:
        return None
    p = Path(base)
    if not p.is_absolute():
        p = ROOT / p
    js = p if p.suffix.lower() == ".json" else p.with_suffix(".json")
    try:
        data = json.loads(js.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    size = data.get("image_size") if isinstance(data, dict) else None
    try:
        w, h = int(size[0]), int(size[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    return (w, h) if w > 0 and h > 0 else None


def _persist(payload: dict, args) -> int:
    """把结果落盘（`--json` / `--write-profile`）并打一句收尾。返回退出码。"""
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

        # 越界校验：这份 crop_rect 必须是**当前标定**那套画面上的坐标
        size = _expected_raw_size(current)
        rect = payload.get("crop_rect") or []
        if size and len(rect) == 4:
            x, y, w, h = (int(v) for v in rect)
            if x + w > size[0] or y + h > size[1]:
                print(f"\n[错误] 这份 FOV 记录放不进当前标定的画面：")
                print(f"       裁剪矩形 ({x}, {y}, {w}, {h}) 需要至少 "
                      f"{x + w}x{y + h} 的画面，")
                print(f"       而档案「{args.write_profile}」配的标定是 "
                      f"{size[0]}x{size[1]}。")
                print(f"       说明这份记录来自**另一个分辨率**（换过相机/改过输出档，")
                print(f"       或者读的是旧会话）。把它写进档案，下次标定会因为裁剪框")
                print(f"       越界直接失败。")
                print(f"       下一步：在**当前分辨率**下重新标定一次"
                      f"（标定时现场估并自动记录），或先用 --json 只导出、")
                print(f"       确认分辨率一致后再手工写入。")
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="量两路相机的固有视场差（scale / 裁剪参数），可存进装置档案")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--left", help="左路图像文件")
    src.add_argument("--left-video", help="左路视频（与 --right-video 配套）")
    src.add_argument("--from-session", default="", metavar="DIR",
                     help="直接读这个会话 session.json 里记录的 rig.fov"
                          "（标定当时在原始画面上算的），不重新估。"
                          "这是三条来源里唯一在「录像已裁剪」时仍然正确的")
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

    # ---- 来源 A：直接读会话记录（唯一在"录像已裁剪"时仍然正确的一条）----
    if args.from_session:
        sess = Path(args.from_session)
        js = sess / "session.json"
        if not js.is_file():
            print(f"[错误] 这个目录里没有 session.json：{sess}")
            print("       要的是双目会话目录（data/stereo/<时间戳>）")
            return 2
        try:
            data = json.loads(js.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"[错误] session.json 读不出来：{exc}")
            return 2
        if not isinstance(data, dict):
            print("[错误] session.json 不是一个对象")
            return 2
        rig = data.get("rig")
        raw = rig.get("fov") if isinstance(rig, dict) else None
        if not isinstance(raw, dict) or not raw:
            print(f"[错误] {sess.name} 的 session.json 里没有 FOV 记录。")
            print("       说明这段录像录制时用的标定没有 FOV 参数"
                  "（旧标定 / --skip-fov / 估计降级）。")
            print("       若这段录像**未裁剪**，可改用 --left-video/--right-video 现场量一次。")
            return 1
        payload = {k: raw[k] for k in _PROFILE_KEYS if k in raw}
        if "target_size" not in payload or "scale" not in payload:
            print(f"[错误] {sess.name} 的 FOV 记录不完整（缺 target_size / scale）")
            return 1
        # 原始分辨率**没有记录**（crop_rect 的坐标系是未裁剪的相机画面，而
        # session 里只有裁剪后的尺寸）。所以 left_size/right_size 故意不写：
        # 让后端跳过分辨率核对；万一裁剪框越界，标定/录制会明确报错。
        created = data.get("created")
        payload["measured_at"] = (str(created) if created
                                  else datetime.now().isoformat(timespec="seconds"))
        payload["source"] = f"session:{sess.name}"

        print("=" * 72)
        print("两路视场差（读自会话记录，不重新估计）")
        print("=" * 72)
        print(f"  会话        : {sess.name}（录制于 {created or '未知'}）")
        print(f"  参考路      : {raw.get('reference_side')}")
        print(f"  被裁剪路    : {raw.get('crop_side')}")
        try:
            scale_v = float(raw.get("scale"))
        except (TypeError, ValueError):
            scale_v = float("nan")
        print(f"  尺度 left→right: {scale_v:.4f}")
        print(f"  裁剪矩形    : {tuple(raw.get('crop_rect') or [])}"
              f"（在 {raw.get('crop_side')} 的**原始**像素系里）")
        print(f"  共同输出尺寸: {tuple(raw.get('target_size') or [])}")
        print(f"  当时质量    : {raw.get('method')} / inliers {raw.get('inliers')}"
              f" / quality {raw.get('quality')}")
        print()
        print("  ⚠️ crop_rect 是**未裁剪画面**上的坐标，而这份 session.json 没有")
        print("     记原始分辨率。所以写入前会拿它与档案配的标定的 image_size 核对：")
        print("     放不进去就**拒绝写入**（不会静默沿用一套越界的参数）。")
        return _persist(payload, args)

    # ---- 取两路帧 ----------------------------------------------------------
    video_left: Path | None = None
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
            video_left = Path(args.left_video)
            left = _read_video_frame(video_left, args.frame)
            right = _read_video_frame(Path(args.right_video), args.frame)
            source = f"video:{video_left.parent.name}@{args.frame}"
    except ValueError as exc:
        print(f"[错误] {exc}")
        return 2

    h_l, w_l = left.shape[:2]
    h_r, w_r = right.shape[:2]

    # 已裁剪的录像不能再量（会得到自洽但错误的"二次裁剪"结果）
    if video_left is not None:
        problem = already_cropped_problem(video_left, (w_l, h_l))
        if problem:
            print("=" * 72)
            print("两路视场差测量")
            print("=" * 72)
            print(f"\n[拒绝] {problem}")
            return 1

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
    return _persist(payload, args)


if __name__ == "__main__":
    raise SystemExit(main())
