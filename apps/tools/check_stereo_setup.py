"""标定前置检查：两路是否都看得到棋盘、视野是否重叠。

标定能成功的**第一个前提**是"棋盘同时落在两个相机的视野里"。这个工具在标定
之前跑一遍，用实时左右画面直接回答：

    1. 左相机看到棋盘了吗？          (逐帧检出率)
    2. 右相机看到棋盘了吗？
    3. 两路分辨率 / 帧率一致吗？      (标定要求两路一致)

它不标定、不改任何参数，只做"能不能开始"的判断。

用法::

    python apps/tools/check_stereo_setup.py --left 0 --right 2
    python apps/tools/check_stereo_setup.py --left 0 --right 2 --seconds 6
    python apps/tools/check_stereo_setup.py --left-video l.mp4 --right-video r.mp4 --no-show

窗口里按 q 提前结束。建议举着棋盘慢慢在画面里上下左右移动，
让它在两路中都出现过，再看结论。
"""

from __future__ import annotations

import argparse
import json
import math
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

from apps._common import add_stereo_source_args, open_pair_source
from apps._window import WindowManager
from calib import BoardSpec, detect_board
from stereo import profiles as rig_profiles

WINDOW = "stereo setup check"
PREVIEW_W = 620          # 单路预览宽度

REPO_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_PATH = REPO_ROOT / "config" / "stereo_calibration.json"

# 与 stereo/sessions.py 的基线交叉核对口径一致：>10% 提醒、>30% 报警。
BASELINE_WARN_RATIO = 0.10
BASELINE_ERROR_RATIO = 0.30


# --------------------------------------------------------------------------
def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _fit(img: np.ndarray, width: int):
    """按宽度缩放，返回 (缩放图, 缩放系数)。"""
    s = width / float(img.shape[1])
    out = cv2.resize(img, (width, max(1, int(round(img.shape[0] * s)))),
                     interpolation=cv2.INTER_AREA)
    return out, s


def _pad_bottom(img: np.ndarray, h: int) -> np.ndarray:
    if img.shape[0] >= h:
        return img
    pad = np.zeros((h - img.shape[0], img.shape[1], 3), np.uint8)
    return np.vstack([img, pad])


def _draw(img: np.ndarray, obs, scale: float) -> None:
    """把棋盘轮廓与角点画到预览图上。"""
    if not obs.found or obs.corners is None:
        return
    quad = obs.quad()
    if quad is not None:
        cv2.polylines(img, [(quad * scale).astype(np.int32)], True, (0, 220, 0), 2)
    for x, y in obs.corners.reshape(-1, 2) * scale:
        cv2.circle(img, (int(x), int(y)), 2, (0, 255, 255), -1)


def _center_norm(obs, w: int, h: int):
    """棋盘角点中心的归一化位置（0..1）。"""
    if obs.corners is None:
        return None
    pts = obs.corners.reshape(-1, 2)
    return float(pts[:, 0].mean() / w), float(pts[:, 1].mean() / h)


def _mean(points) -> tuple[float, float]:
    a = np.asarray(points, dtype=np.float64)
    return float(a[:, 0].mean()), float(a[:, 1].mean())


# --------------------------------------------------------------------------
def _calibration_baseline(path: Path | None = None) -> float | None:
    """读实际标定文件的算法基线（mm）；读不到返回 ``None``（不抛）。"""
    target = Path(path) if path is not None else CALIBRATION_PATH
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    value = data.get("baseline_mm")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number > 0 else None


def _print_profile_section() -> tuple[list[str], list[str]]:
    """打印"当前装置档案"，返回 (warnings, errors)。

    档案是可选项：没设置就一行提示，不改变 precheck 的判定；设置了才做
    "期望基线 vs 实际标定基线"的交叉核对（>10% 警告、>30% 错误）。
    """
    print("-" * 62)
    print("当前装置档案")
    warnings: list[str] = []
    errors: list[str] = []
    try:
        active = rig_profiles.get_active()
    except rig_profiles.ProfileError as exc:
        print(f"  读不了档案        : {exc}")
        errors.append(f"装置档案读不了：{exc}")
        return warnings, errors

    if active is None:
        print("  未设置装置档案（可选）：配置后可让 precheck/录制带出正确几何预期")
        return warnings, errors

    label = rig_profiles.mode_label(active.stereo_mode)
    print(f"  名称              : {active.name}")
    print(f"  双目模式          : {label}（{active.stereo_mode}）")
    if active.width and active.height:
        fps = f"@{active.fps}" if active.fps else ""
        print(f"  分辨率 / 帧率     : {active.width}x{active.height}{fps}")

    if active.expected_baseline_mm is None:
        print("  期望基线          : 未填（无法与实际标定交叉核对）")
    else:
        print(f"  期望基线          : {active.expected_baseline_mm:.2f} mm")
        actual = _calibration_baseline()
        if actual is None:
            print("  实际标定基线      : 读不到"
                  "（config/stereo_calibration.json 缺失或无 baseline_mm）")
        else:
            ratio = abs(active.expected_baseline_mm - actual) / actual
            print(f"  实际标定基线      : {actual:.2f} mm（差 {100 * ratio:.1f}%）")
            if ratio > BASELINE_ERROR_RATIO:
                errors.append(
                    f"装置档案期望基线 {active.expected_baseline_mm:.2f} mm 与标定 "
                    f"{actual:.2f} mm 差 {100 * ratio:.0f}%（>30%：多半不是同一套装置，"
                    "先核对左右相机与标定文件）")
            elif ratio > BASELINE_WARN_RATIO:
                warnings.append(
                    f"装置档案期望基线 {active.expected_baseline_mm:.2f} mm 与标定 "
                    f"{actual:.2f} mm 差 {100 * ratio:.0f}%（>10%）")

    if active.working_distance_m is not None:
        print(f"  工作距离          : {active.working_distance_m:.2f} m")
    print(f"  几何提示          : {rig_profiles.geometry_hint(active)}")

    if active.calibration_path:
        print(f"  配套标定          : {active.calibration_path}")
        if not (REPO_ROOT / active.calibration_path).exists():
            warnings.append(
                f"档案指向的标定文件不存在：{active.calibration_path}"
                "（可能已被清理；标定后请重新设置档案）")
    if active.stereo_mode == "angled":
        print("  写盘/坐标系       : 按 docs/数据标准.md §1.5 为原始 left_camera，"
              "不做 FOV 裁剪")

    for w in warnings:
        print(f"  ! {w}")
    for e in errors:
        print(f"  × {e}")
    return warnings, errors


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="标定前置检查（两路是否都看到棋盘）")
    add_stereo_source_args(ap)
    ap.add_argument("--cols", type=int, default=8, help="棋盘格内角点列数")
    ap.add_argument("--rows", type=int, default=5, help="棋盘格内角点行数")
    ap.add_argument("--square", type=float, default=25.0, help="方格边长（毫米）")
    ap.add_argument("--seconds", type=float, default=8.0, help="实时采集多少秒后给结论")
    ap.add_argument("--work-width", type=int, default=1280, help="检测时缩放到的宽度")
    ap.add_argument("--exhaustive", action="store_true", help="更慢但更不易漏检")
    ap.add_argument("--min-both-ratio", type=float, default=0.5,
                    help="两路同时看到棋盘的最低帧比例，达到才判为通过")
    ap.add_argument("--no-show", action="store_true", help="不弹窗（批处理）")
    ap.add_argument("--profiles-only", action="store_true",
                    help="只打印当前装置档案（不开相机，便于核对参数/标定配对）")
    args = ap.parse_args()

    if args.profiles_only:
        _warnings, errors = _print_profile_section()
        print("=" * 62)
        return 1 if errors else 0

    spec = BoardSpec(cols=args.cols, rows=args.rows, square_mm=args.square)
    print(f"标定板：{spec.describe()}")
    src = open_pair_source(args)
    if src is None:
        return 2
    print(src.describe())
    src.report_warnings()
    print("提示：举起棋盘，让它同时落在两路画面里；缓慢上下左右移动以检验重叠范围。")
    print("窗口：按 f 全屏（看得更清），Esc 退出全屏，q 提前结束。")
    if src.is_live:
        print(f"采集约 {args.seconds:.0f} 秒后自动给结论（窗口里按 q 可提前结束）。")
    print()

    wins = WindowManager()
    total = n_l = n_r = n_both = 0
    area_l = area_r = 0.0
    sharp_l = sharp_r = 0.0
    centers_l: list = []
    centers_r: list = []
    L_wh = R_wh = None
    t0 = time.perf_counter()

    try:
        while True:
            # 到点就收工：即使一帧都没取到也必须退出，否则"相机取不到帧"时
            # 会在这里死循环（read() 一直失败 -> continue 永远不检查时间）。
            if src.is_live and (time.perf_counter() - t0) >= args.seconds:
                break
            ok, left, right = src.read()
            if not ok or left is None or right is None:
                if src.is_live:
                    continue
                break
            total += 1
            if L_wh is None:
                L_wh = (left.shape[1], left.shape[0])
                R_wh = (right.shape[1], right.shape[0])

            obs_l = detect_board(_to_gray(left), spec,
                                 work_width=args.work_width, exhaustive=args.exhaustive)
            obs_r = detect_board(_to_gray(right), spec,
                                 work_width=args.work_width, exhaustive=args.exhaustive)
            n_l += int(obs_l.found)
            n_r += int(obs_r.found)
            n_both += int(obs_l.found and obs_r.found)
            if obs_l.found:
                area_l += obs_l.area_ratio
                sharp_l += obs_l.sharpness
                c = _center_norm(obs_l, *L_wh)
                if c:
                    centers_l.append(c)
            if obs_r.found:
                area_r += obs_r.area_ratio
                sharp_r += obs_r.sharpness
                c = _center_norm(obs_r, *R_wh)
                if c:
                    centers_r.append(c)

            if not args.no_show:
                a, sa = _fit(left, PREVIEW_W)
                b, sb = _fit(right, PREVIEW_W)
                _draw(a, obs_l, sa)
                _draw(b, obs_r, sb)
                hh = max(a.shape[0], b.shape[0])
                a, b = _pad_bottom(a, hh), _pad_bottom(b, hh)
                panel = np.hstack([a, b])
                lbl_l = f"LEFT  cam#{args.left}" if args.left is not None else "LEFT"
                lbl_r = f"RIGHT cam#{args.right}" if args.right is not None else "RIGHT"
                cv2.putText(panel, f"{lbl_l}  board:{'YES' if obs_l.found else 'no'}",
                            (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 220, 0) if obs_l.found else (0, 120, 255), 2, cv2.LINE_AA)
                cv2.putText(panel, f"{lbl_r}  board:{'YES' if obs_r.found else 'no'}",
                            (PREVIEW_W + 10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 220, 0) if obs_r.found else (0, 120, 255), 2, cv2.LINE_AA)
                cv2.putText(panel, f"frames {total}  both {n_both}  (q to finish / f fullscreen)",
                            (10, hh - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (230, 230, 230), 1, cv2.LINE_AA)
                wins.show(WINDOW, panel)
                key = cv2.waitKey(1) & 0xFF
                # 全屏键优先：Esc 在全屏时是"退出全屏"，不该顺手结束检查
                if wins.handle_key(WINDOW, key, panel):
                    continue
                if key in (ord("q"), 27):
                    break

            if src.is_live and (time.perf_counter() - t0) >= args.seconds:
                break
    finally:
        src.release()
        if not args.no_show:
            cv2.destroyAllWindows()

    return _report(args, total, n_l, n_r, n_both, area_l, area_r, sharp_l, sharp_r,
                   centers_l, centers_r, L_wh, R_wh, spec)


def _report(args, total, n_l, n_r, n_both, area_l, area_r, sharp_l, sharp_r,
            centers_l, centers_r, L_wh, R_wh, spec) -> int:
    print()
    print("=" * 62)
    print("标定前置检查结果")
    print("=" * 62)
    print(f"标定板            : {spec.describe()}")
    if total == 0:
        print("没有取到任何帧。请检查相机是否被占用 / 视频是否可读。")
        return 1

    r_l, r_r, r_both = n_l / total, n_r / total, n_both / total
    print(f"总帧数            : {total}")
    print(f"左路看到棋盘      : {n_l}/{total}  ({100 * r_l:.0f}%)")
    print(f"右路看到棋盘      : {n_r}/{total}  ({100 * r_r:.0f}%)")
    print(f"两路同时看到      : {n_both}/{total}  ({100 * r_both:.0f}%)")
    print(f"左 / 右 分辨率    : {L_wh} / {R_wh}")
    if r_l > 0:
        print(f"左路棋盘面积占比  : {100 * area_l / n_l:.1f}%   清晰度 {sharp_l / n_l:.0f}")
    if r_r > 0:
        print(f"右路棋盘面积占比  : {100 * area_r / n_r:.1f}%   清晰度 {sharp_r / n_r:.0f}")
    if centers_l and centers_r:
        cl, cr = _mean(centers_l), _mean(centers_r)
        print(f"棋盘平均位置(0~1): 左 ({cl[0]:.2f}, {cl[1]:.2f})   右 ({cr[0]:.2f}, {cr[1]:.2f})")
    print("-" * 62)

    warnings: list[str] = []
    if L_wh and R_wh and L_wh != R_wh:
        warnings.append(f"两路分辨率不一致（{L_wh} vs {R_wh}）：标定要求两路完全一致，"
                        f"请把两台相机设成同一分辨率。")

    if r_both >= args.min_both_ratio:
        print("[通过] 两路都能稳定看到棋盘 —— 视野重叠良好，可以开始标定。")
        if r_both < 1.0:
            print(f"       提示：仍有 {100 * (1 - r_both):.0f}% 的帧只有一路看到，"
                  f"说明重叠区边缘较窄；拍标定帧时让板尽量落在重叠区中心。")
        code = 0
    elif r_l < 0.15 and r_r < 0.15:
        print("[失败] 两路都看不到棋盘。")
        print("       检查：棋盘是否在两路视野内 / 光线是否充足 / 板面是否平整清晰。")
        code = 1
    elif r_l < 0.15:
        print("[失败] 只有右路能看到棋盘，左路看不到。")
        print("       把板移到左相机画面内，或调整左相机朝向，让两路视野重叠。")
        code = 1
    elif r_r < 0.15:
        print("[失败] 只有左路能看到棋盘，右路看不到。")
        print("       把板移到右相机画面内，或调整右相机朝向，让两路视野重叠。")
        code = 1
    else:
        print("[警告] 只有部分时刻两路同时可见 —— 重叠区偏小，或板未停稳。")
        print("       建议：把两相机夹角调小让视野重叠更大；拍摄时让板停稳约 1 秒。")
        code = 1

    for w in warnings:
        print(f"  ! {w}")
    print()
    _prof_warnings, prof_errors = _print_profile_section()
    print("=" * 62)
    return 1 if prof_errors else code


if __name__ == "__main__":
    raise SystemExit(main())
