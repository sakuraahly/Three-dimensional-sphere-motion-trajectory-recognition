"""中轴 90° 装置标定 CLI（模式 B）：只吃 A/B 板正对图 + 已知内参。

用法::

    python apps/calibrate_device90.py \
        --left-dir sessions/d90/a_board   # 相机 1 拍，板正对相机 1（板 A）
        --right-dir sessions/d90/b_board  # 相机 2 拍，板正对相机 2（板 B）
        --intrinsics config/calib/<rig>   # StereoRig.load 能读的基名（.npz/.json）
        --rot-axis y --rot-angle 90 \
        --expected-baseline-mm 707 --baseline-tolerance 0.05

装置模型（见 `calib/device90_calib.py`）：同一块板绕自身中轴转 90°，两板中心重合；
`T_AB=[R_AB|0]` 是纯旋转；外参 `T_{C_B←C_A}=T_camB_B·T_AB·inv(T_camA_A)`，不需要
两路共视、也不需要同框图。

本脚本**独立可跑**，不改 `apps/calibrate_stereo.py` / `apps/_actions.py` /
`apps/_gui_stereo.py`，也不注册 `calib/methods.py` 插件。

退出码：0 成功；1 质量不可用且未 `--force-save`；2 输入/参数错误。
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

import numpy as np

from apps._common import current_calibration
from apps.calibrate_axial import (          # 复用同一份只读 helper（不碰 calibrate_stereo 私有函数）
    _backup_before_force,
    _best_frame,
    _bypass_out,
    _list_images,
    _save_res,
    _write_report,
)
from calib import BoardSpec, CalibConfig, StereoRig
from calib.device90_calib import solve_device90

__all__ = ["build_parser", "main"]


def _load_intrinsics(path: str):
    """读 StereoRig 内参；缺 K1/K2 或 D1/D2 时返回 (None, 原因)，**不做 D=0 兜底**。"""
    try:
        rig = StereoRig.load(path)
    except FileNotFoundError:
        return None, f"找不到内参文件：{path}（.npz 必须存在，.json 可选）"
    except Exception as exc:                               # noqa: BLE001
        return None, f"读取内参失败（{type(exc).__name__}: {exc}）：{path}"
    if rig.K1 is None or rig.K2 is None:
        return None, f"内参文件缺 K1/K2：{path}"
    if rig.D1 is None or rig.D2 is None or np.size(rig.D1) < 4 or np.size(rig.D2) < 4:
        return None, (f"内参文件缺 D1/D2（或长度不足 4）：{path}；本入口不做 D=0 兜底，"
                      f"请用模式 A 或常规双目标定产出一份带畸变的标定文件")
    return (np.asarray(rig.K1, np.float64), np.asarray(rig.D1, np.float64),
            np.asarray(rig.K2, np.float64), np.asarray(rig.D2, np.float64)), ""


def _check_intrinsics_hard(K1, D1, K2, D2) -> str:
    """读入内参后的**硬错误**（不继续）：K 非 3x3、fx/fy/cx/cy 非有限或焦距非正。

    返回原因字符串；空串 = 通过。device90 完全依赖外部内参，这里守住"根本不能用"
    的情况，避免 `solvePnP` 拿到 NaN/0 焦距后给出看似收敛的错解。
    """
    for name, K in (("K1", K1), ("K2", K2)):
        arr = np.asarray(K, np.float64)
        if arr.shape != (3, 3):
            return f"{name} 不是 3x3（实际 {arr.shape}）"
        fx, fy, cx, cy = arr[0, 0], arr[1, 1], arr[0, 2], arr[1, 2]
        vals = np.array([fx, fy, cx, cy], np.float64)
        if not np.all(np.isfinite(vals)):
            return f"{name} 含 NaN/inf（fx={fx}, fy={fy}, cx={cx}, cy={cy}）"
        if fx <= 0.0 or fy <= 0.0:
            return f"{name} 焦距非正（fx={fx}, fy={fy}）"
    return ""


def _intrinsics_sanity_reasons(K1, D1, K2, D2, width=None) -> list[str]:
    """内参**软告警**（不阻止求解）：返回可疑项说明列表，空 = 无可疑。

    只进 warnings / 报告 / meta，**不影响 ok**（内参可疑不等于外参几何错）。
    真实产物实测：`实验性质大角度1.npz` 的 K1 fx=2304、K2 fx=1541（差 33%），
    而两份平行装置只差 0.7% / 0.26%。左右相差过大通常意味着两台相机型号不同，
    或内参标定本身有误 —— device90 吃外部内参，外参会直接受其影响。
    """
    reasons: list[str] = []
    fx1, fx2 = float(np.asarray(K1)[0, 0]), float(np.asarray(K2)[0, 0])
    denom = max(abs(fx1), abs(fx2), 1e-12)
    rel = abs(fx1 - fx2) / denom
    if rel > 0.30:
        reasons.append(
            f"左右相机 fx 相差 {rel * 100:.1f}%（K1={fx1:.2f}, K2={fx2:.2f}）："
            f"可能两台相机型号不同，或内参标定有误；device90 完全依赖外部内参，"
            f"结果会直接受其影响")
    dmax = 0.0
    for D in (D1, D2):
        d = np.asarray(D, np.float64).ravel()
        if d.size:
            dmax = max(dmax, float(np.max(np.abs(d))))
    if dmax > 1.0:
        reasons.append(f"畸变系数绝对值超过 1.0（max|D|={dmax:.3f}），异常，请先复核内参")
    if width:
        lim = 2.0 * float(width)
        for name, K in (("K1", K1), ("K2", K2)):
            cx, cy = float(np.asarray(K)[0, 2]), float(np.asarray(K)[1, 2])
            if not (0.0 <= cx <= lim and 0.0 <= cy <= lim):
                reasons.append(
                    f"{name} 主点 ({cx:.1f}, {cy:.1f}) 超出 [0, {lim:.0f}]（width={width}）")
    return reasons


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="中轴 90° 装置标定（模式 B）：A/B 板正对图 + 已知内参，不共视、无同框图")
    ap.add_argument("--left-dir", required=True,
                    help="A 板图目录（相机 1 拍，板正对相机 1）")
    ap.add_argument("--right-dir", required=True,
                    help="B 板图目录（相机 2 拍，板正对相机 2）")
    ap.add_argument("--intrinsics", required=True,
                    help="内参来源（StereoRig 基名，含 K1/D1/K2/D2；模式 A 的产物或现有标定）")
    ap.add_argument("--rot-axis", default="y", choices=("x", "y"),
                    help="板中轴方向（默认 y）")
    ap.add_argument("--rot-angle", type=float, default=90.0,
                    help="绕中轴旋转角，度（默认 90）")
    ap.add_argument("--z-a", type=float, default=None,
                    help="板 A 中心深度 mm（可选；给了就覆盖 solvePnP 的 z）")
    ap.add_argument("--z-b", type=float, default=None,
                    help="板 B 中心深度 mm（可选；给了就覆盖 solvePnP 的 z）")
    ap.add_argument("--cols", type=int, default=8, help="内角点列数（默认 8）")
    ap.add_argument("--rows", type=int, default=5, help="内角点行数（默认 5）")
    ap.add_argument("--square", type=float, default=25.0, help="方格边长 mm（默认 25）")
    ap.add_argument("--out", default=None,
                    help="输出基名（无扩展名）；缺省用当前装置档案的标定路径")
    ap.add_argument("--expected-baseline-mm", type=float, default=None,
                    help="机械基线 mm（强烈建议给：装反/尺度错的唯一自动信号）")
    ap.add_argument("--baseline-tolerance", type=float, default=0.05,
                    help="基线相对容差（默认 0.05；注意 joint/chain 全局默认是 0.20）")
    ap.add_argument("--force-save", action="store_true",
                    help="质量不可用时仍写正式路径（先备份旧标定；仅链路测试）")
    ap.add_argument("--report", default=None, help="写一份人读 JSON 摘要到该路径")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    # ---- 1. 读内参 ----
    loaded, reason = _load_intrinsics(args.intrinsics)
    if loaded is None:
        print(f"\n[错误] {reason}")
        return 2
    K1, D1, K2, D2 = loaded
    hard = _check_intrinsics_hard(K1, D1, K2, D2)
    if hard:
        print(f"\n[错误] 内参非法：{hard}")
        return 2

    try:
        spec = BoardSpec(cols=args.cols, rows=args.rows, square_mm=args.square)
    except ValueError as exc:
        print(f"\n[错误] 板规格非法：{exc}")
        return 2

    # ---- 2. 读 A/B 目录各取最佳帧 ----
    a_paths = _list_images(Path(args.left_dir))
    b_paths = _list_images(Path(args.right_dir))
    if not a_paths:
        print(f"\n[错误] --left-dir 里没有图像：{args.left_dir}")
        return 2
    if not b_paths:
        print(f"\n[错误] --right-dir 里没有图像：{args.right_dir}")
        return 2
    fa = _best_frame(a_paths, spec)
    fb = _best_frame(b_paths, spec)
    if fa is None:
        print(f"\n[错误] --left-dir 里检不出完整棋盘（{spec.describe()}）：{args.left_dir}")
        return 2
    if fb is None:
        print(f"\n[错误] --right-dir 里检不出完整棋盘（{spec.describe()}）：{args.right_dir}")
        return 2
    corners_a, size_a, path_a, focus_a = fa
    corners_b, size_b, path_b, focus_b = fb
    print(f"[读图] A: {path_a.name}（{size_a[0]}x{size_a[1]}, focus={focus_a:.0f}）")
    print(f"[读图] B: {path_b.name}（{size_b[0]}x{size_b[1]}, focus={focus_b:.0f}）")
    intr_reasons = _intrinsics_sanity_reasons(K1, D1, K2, D2, width=size_a[0])

    # ---- 3. 求解 ----
    cfg = CalibConfig(expected_baseline_mm=args.expected_baseline_mm,
                      baseline_tolerance=float(args.baseline_tolerance))
    Z_A = None if args.z_a is None else float(args.z_a) / 1000.0
    Z_B = None if args.z_b is None else float(args.z_b) / 1000.0
    try:
        res = solve_device90(corners_a, corners_b, K1, D1, K2, D2, spec, size_a,
                             axis=args.rot_axis, angle_deg=float(args.rot_angle),
                             Z_A=Z_A, Z_B=Z_B, cfg=cfg)
    except Exception as exc:                               # noqa: BLE001 - 输入/数值错误
        print(f"\n[错误] 中轴 90° 求解失败（{type(exc).__name__}: {exc}）")
        return 2

    # 内参软告警：附加到 res.warnings（**不改 ok**：几何是否成立由 solver 判）
    if intr_reasons:
        res.warnings = list(res.warnings) + intr_reasons

    print(f"\n[中轴90°] 基线={res.baseline_mm:.2f} mm  判定={res.grade}  "
          f"axis={args.rot_axis} angle={args.rot_angle:g}°  "
          f"rectification={res.rectification}")
    for w in res.warnings:
        print(f"  [注意] {w}")

    # ---- 4. 落盘 ----
    out = args.out or str(current_calibration(need_existing=False)[0])
    meta = {
        "method": "device90",
        "calibration_mode": "device90_midaxis",
        "board": spec.describe(),
        "board_spec": {"cols": int(spec.cols), "rows": int(spec.rows),
                       "square_mm": float(spec.square_mm)},
        "rot_axis": args.rot_axis,
        "rot_angle_deg": float(args.rot_angle),
        "z_a_mm": (None if args.z_a is None else float(args.z_a)),
        "z_b_mm": (None if args.z_b is None else float(args.z_b)),
        "expected_baseline_mm": (None if args.expected_baseline_mm is None
                                 else float(args.expected_baseline_mm)),
        "baseline_tolerance": float(args.baseline_tolerance),
        "baseline_mm": float(res.baseline_mm),
        "intrinsics_source": str(args.intrinsics),
        "intrinsics_suspect": bool(intr_reasons),
        "intrinsics_suspect_reason": "；".join(intr_reasons),
        "chosen_frames": {"A": path_a.name, "B": path_b.name},
    }
    saved = _save_res(res, meta, out, bool(args.force_save))
    if saved is None:
        return 2
    npz, js, bypassed = saved
    print(f"[保存] {npz}\n[保存] {js}")

    if args.report:
        report = {
            "method": "device90",
            "calibration_mode": "device90_midaxis",
            "baseline_mm": float(res.baseline_mm),
            "grade": res.grade,
            "ok": bool(res.ok),
            "verdict": res.verdict,
            "rectification": res.rectification,
            "rot_axis": args.rot_axis,
            "rot_angle_deg": float(args.rot_angle),
            "z_a_mm": meta["z_a_mm"],
            "z_b_mm": meta["z_b_mm"],
            "expected_baseline_mm": meta["expected_baseline_mm"],
            "baseline_tolerance": float(args.baseline_tolerance),
            "intrinsics_suspect": bool(intr_reasons),
            "intrinsics_suspect_reason": "；".join(intr_reasons),
            "chosen_frames": meta["chosen_frames"],
            "warnings": list(res.warnings),
            "bypassed": bool(bypassed),
            "saved_npz": str(npz),
            "saved_json": str(js),
        }
        _write_report(args.report, report)
        print(f"[报告] {args.report}")

    if (not res.ok) and (not args.force_save):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
