"""轴向标定 CLI（模式 A）：平行双目 + 大板正对 + 已知沿光轴推进量。

用法::

    # 深度子目录按名字/单位自动解析 ΔZ（z000mm / z010mm / 0mm / 10 ...）
    python apps/calibrate_axial.py --root sessions/axial_run

    # 或者显式给 ΔZ（mm，逗号分隔，按名字排序映射到深度子目录）
    python apps/calibrate_axial.py --root sessions/axial_run --delta-z 0,10,20,30

目录约定::

    <root>/
      z000mm/left/*.png  z000mm/right/*.png
      z010mm/left/*.png  z010mm/right/*.png
      ...

每个深度子目录下 `left/` 与 `right/` 各拍**同一块正对的板**（同深度共视）；
一个目录有多张时取清晰度最高（detect_board 的 focus，尺度无关）的一张，
并在摘要里报告用了哪张。

本脚本**独立可跑**，不改 `apps/calibrate_stereo.py` 的任何现有逻辑，也不注册
`calib/methods.py` 插件。求解后端是 `calib/axial_calib.py`。

退出码：0 成功；1 质量不可用且未 `--force-save`；2 输入/参数错误。
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
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

from apps._common import current_calibration
from calib import BoardSpec, CalibConfig, StereoRig, detect_board
from calib.axial_calib import solve_axial_intrinsics, solve_axial_stereo

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# --------------------------------------------------------------------------
# 目录 / ΔZ 解析
# --------------------------------------------------------------------------
def _list_images(path: Path) -> list[Path]:
    """列出目录下的图像（按名字排序）；传单个文件则只返回它。"""
    if path.is_file():
        return [path] if path.suffix.lower() in _IMAGE_SUFFIXES else []
    if not path.is_dir():
        return []
    return sorted(p for p in path.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)


_NUM = r"[+-]?\d+(?:\.\d+)?"
_UNIT_RE = re.compile(rf"({_NUM})\s*(mm|cm|m)(?![a-z])", re.IGNORECASE)
_ZPREFIX_RE = re.compile(rf"z\s*({_NUM})", re.IGNORECASE)
_BARE_RE = re.compile(_NUM)


def _parse_dz(name: str) -> float | None:
    """从深度子目录名解析 ΔZ，返回**米**；解析不出返回 None。

    优先级（越具体的越先）：
      1. 带单位 `mm`/`cm`/`m` 的数字（如 ``z010mm``、``0mm``、``1.5cm``）；
      2. `z` 前缀的数字（如 ``z000``、``z-05``），单位默认 mm；
      3. 裸数字（如 ``10``），单位默认 mm。
    """
    s = name.strip()
    m = _UNIT_RE.search(s)
    if m:
        val, unit = float(m.group(1)), m.group(2).lower()
        return val / 1000.0 if unit == "mm" else (val / 100.0 if unit == "cm" else val)
    m = _ZPREFIX_RE.search(s)
    if m:
        return float(m.group(1)) / 1000.0
    m = _BARE_RE.search(s)
    if m:
        return float(m.group(0)) / 1000.0
    return None


def _split_delta_mm(text: str) -> list[float]:
    """把 `--delta-z` 的逗号分隔字符串转成米制列表（输入单位 mm）。"""
    try:
        vals = [float(x) for x in str(text).replace("，", ",").split(",") if x.strip()]
    except ValueError as exc:
        raise ValueError(f"--delta-z 不是合法的逗号分隔数字：{text!r}（{exc}）") from exc
    if not vals:
        raise ValueError("--delta-z 为空")
    return [v / 1000.0 for v in vals]


def _scan_depth_dirs(
    root: Path, left_sub: str, right_sub: str,
    delta_z_m: list[float] | None = None,
) -> list[dict]:
    """扫描 `root` 下的深度子目录，返回按 ΔZ 升序的条目列表。

    每项：``{"dz_m": float, "name": str, "left_dir": Path, "right_dir": Path}``。

    规则：
      - 无 ``delta_z_m``：从子目录名解析 ΔZ；解析失败 / 数量 < 3 / 值重复 -> ValueError；
      - 有 ``delta_z_m``：按**名字排序**的深度子目录逐一映射到给定序列；
        数量不一致 / 值重复 / 数量 < 3 -> ValueError；
      - 每个深度目录下必须存在 `left_sub` 与 `right_sub` 子目录。
    """
    if not root.is_dir():
        raise ValueError(f"--root 不是目录：{root}")
    subdirs = sorted([d for d in root.iterdir() if d.is_dir()], key=lambda d: d.name)
    if not subdirs:
        raise ValueError(f"--root 下没有任何子目录：{root}")

    if delta_z_m is not None:
        if len(subdirs) != len(delta_z_m):
            raise ValueError(
                f"--delta-z 给了 {len(delta_z_m)} 个值，但 --root 下有 {len(subdirs)} 个"
                f"深度子目录：{[d.name for d in subdirs]}")
        pairs = list(zip(delta_z_m, subdirs))
    else:
        parsed, bad = [], []
        for d in subdirs:
            dz = _parse_dz(d.name)
            if dz is None:
                bad.append(d.name)
            else:
                parsed.append((dz, d))
        if bad:
            raise ValueError(
                "以下深度子目录名解析不出 ΔZ（支持 z000/z010mm/z-05mm/0mm/10 等）："
                + "、".join(bad)
                + "；请改名或改用 --delta-z 显式指定（mm，逗号分隔）")
        pairs = parsed

    pairs.sort(key=lambda x: x[0])   # 按数值升序，不是字符串排序
    if len(pairs) < 3:
        raise ValueError(
            f"轴向标定至少需要 3 个深度，实际只解析到 {len(pairs)} 个："
            f"{[p[1].name for p in pairs]}")
    dzs = [dz for dz, _d in pairs]
    if len(set(round(z, 12) for z in dzs)) != len(dzs):
        raise ValueError(f"深度值重复（ΔZ = {[round(z * 1000, 3) for z in dzs]} mm）："
                         f"重复或解析到同一深度的目录会让 1/p 与 ΔZ 退化")

    items = []
    for dz, d in pairs:
        ldir, rdir = d / left_sub, d / right_sub
        if not ldir.is_dir():
            raise ValueError(f"深度目录 {d.name} 下缺少左图子目录 {left_sub}/")
        if not rdir.is_dir():
            raise ValueError(f"深度目录 {d.name} 下缺少右图子目录 {right_sub}/")
        items.append({"dz_m": float(dz), "name": d.name,
                      "left_dir": ldir, "right_dir": rdir})
    return items


def _canonical_order(corners: np.ndarray, spec: BoardSpec) -> np.ndarray:
    """把检测到的角点重排成 `BoardSpec.object_points()` 的顺序（行主序、x 向右、y 向下）。

    `findChessboardCornersSB` 返回的角点顺序取决于棋盘的起始色，可能是整体反序
    （诊断实测：与理想投影正序 RMS 278 px、反序 0.66 px）。`solvePnP` 需要
    `corners[j]` 与 `object_points()[j]` 一一对应，否则板位姿全错。这里按几何
    方向校正：行内 x 反向就翻列、行间 y 反向就翻行。
    """
    g = np.asarray(corners, np.float32).reshape(spec.rows, spec.cols, 2).copy()
    if g[0, -1, 0] - g[0, 0, 0] < 0.0:      # 行内 x 向左 -> 翻列
        g = g[:, ::-1]
    if g[-1, 0, 1] - g[0, 0, 1] < 0.0:      # 行间 y 向上 -> 翻行
        g = g[::-1]
    return g.reshape(-1, 2)


def _best_frame(paths: list[Path], spec: BoardSpec):
    """从一个目录里挑检测最好的一帧。

    返回 ``(corners(N,2) float32, (w,h), path, focus)`` 或 ``None``。
    “最好”用 `detect_board` 的 `focus`（清晰度 × 格子像素数，尺度无关），
    与项目里判“糊不糊”的口径一致。角点已按 `_canonical_order` 规范化。
    """
    best = None
    for p in paths:
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        obs = detect_board(gray, spec, exhaustive=True)
        if not obs.found or obs.corners is None:
            continue
        score = float(getattr(obs, "focus", 0.0) or 0.0)
        cand = (score, obs.corners.reshape(-1, 2).astype(np.float32),
                (int(gray.shape[1]), int(gray.shape[0])), p)
        if best is None or cand[0] > best[0]:
            best = cand
    if best is None:
        return None
    score, corners, size, path = best
    return _canonical_order(corners, spec), size, path, score


# --------------------------------------------------------------------------
# 落盘（质量差默认旁路；--force-save 才写正式路径）
# --------------------------------------------------------------------------
def _bypass_out(out: str) -> str:
    p = Path(out)
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
    return str(p.with_name(f"{p.stem}_unusable_{stamp}{p.suffix}"))


def _backup_before_force(out: str | Path) -> list[Path]:
    """强制覆盖前把旧 `.npz`/`.json` 复制到同目录带时间戳备份（绝不覆盖旧备份）。"""
    p = Path(out)
    existing = [(q, q.suffix) for q in (p.with_suffix(".npz"), p.with_suffix(".json"))
                if q.is_file()]
    if not existing:
        return []
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
    dests: list[Path] = []
    for src, ext in existing:
        dest = p.with_name(f"{p.stem}_before_force_{stamp}{ext}")
        n = 1
        while dest.exists():
            dest = p.with_name(f"{p.stem}_before_force_{stamp}_{n}{ext}")
            n += 1
        shutil.copy2(src, dest)
        dests.append(dest)
    return dests


def _save_res(res, meta: dict, out: str, force_save: bool):
    """保存 StereoRig。返回 `(npz, json, bypassed)`；备份失败返回 None。"""
    unusable = not bool(res.ok)
    rig = StereoRig.from_result(res, meta=meta, allow_unusable=unusable)
    if unusable:
        rig.quality["forced_unusable"] = True
        rig.quality["verdict"] = (
            f"{rig.quality.get('verdict') or '不可用'} 【仅链路测试，不得用于数据结论】")
        if force_save:
            try:
                backups = _backup_before_force(out)
            except Exception as exc:                       # noqa: BLE001
                print(f"\n[错误] 强制保存前备份旧标定失败（{type(exc).__name__}: {exc}）；"
                      f"为保护旧标定，本次不写正式路径 {out}")
                return None
            if backups:
                print(f"  强制覆盖前已备份旧标定：{', '.join(str(b) for b in backups)}")
            print("\n" + "!" * 66)
            print("⚠ 用户强制保存不可用标定（仅链路测试，写入正式 --out 路径）")
            print(f"  判定：{rig.quality.get('verdict')}")
            print("!" * 66)
        else:
            out = _bypass_out(out)
            print("\n" + "!" * 66)
            print("⚠ 标定质量不合格（仅质量差，非程序错误，不中止）")
            print(f"  结果另存为旁路文件供检查：{out}")
            print("  激活装置档案未改动（正式标定路径保持原样）。")
            print(f"  判定：{rig.quality.get('verdict')}")
            print("!" * 66)
    npz, js = rig.save(out)
    return npz, js, (unusable and not force_save)


def _write_report(path: str, data: dict) -> None:
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="轴向标定（模式 A）：平行双目 + 板正对 + 已知沿光轴推进量")
    ap.add_argument("--root", required=True,
                    help="采集根目录，其下是按深度命名的子目录（z000mm/z010mm/...）")
    ap.add_argument("--left-sub", default="left",
                    help="每个深度子目录下的左图子目录名（默认 left）")
    ap.add_argument("--right-sub", default="right",
                    help="每个深度子目录下的右图子目录名（默认 right）")
    ap.add_argument("--delta-z", default=None,
                    help="显式 ΔZ 列表（mm，逗号分隔，如 0,10,20,30）；给了就按此顺序"
                         "映射到按名字排序的深度子目录，不再解析目录名")
    ap.add_argument("--cols", type=int, default=8, help="内角点列数（默认 8）")
    ap.add_argument("--rows", type=int, default=5, help="内角点行数（默认 5）")
    ap.add_argument("--square", type=float, default=25.0, help="方格边长 mm（默认 25）")
    ap.add_argument("--out", default=None,
                    help="输出基名（无扩展名）；缺省用当前装置档案的标定路径")
    ap.add_argument("--expected-baseline-mm", type=float, default=None,
                    help="机械基线 mm（强烈建议给：装反/尺度错的唯一自动信号）")
    ap.add_argument("--baseline-tolerance", type=float, default=0.05,
                    help="基线相对容差（默认 0.05；注意 joint/chain 全局默认是 0.20）")
    ap.add_argument("--assume-parallel", action="store_true",
                    help="机械保证两光轴平行时强制 R=I（基线 T 仍来自观测）")
    ap.add_argument("--force-save", action="store_true",
                    help="质量不可用时仍写正式路径（先备份旧标定；仅链路测试）")
    ap.add_argument("--report", default=None, help="写一份人读 JSON 摘要到该路径")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    # ---- 1. 扫描深度目录 ----
    try:
        delta_z_arg = _split_delta_mm(args.delta_z) if args.delta_z else None
        items = _scan_depth_dirs(Path(args.root), args.left_sub, args.right_sub,
                                 delta_z_m=delta_z_arg)
    except ValueError as exc:
        print(f"\n[错误] {exc}")
        return 2

    try:
        spec = BoardSpec(cols=args.cols, rows=args.rows, square_mm=args.square)
    except ValueError as exc:
        print(f"\n[错误] 板规格非法：{exc}")
        return 2

    # ---- 2. 每个深度读图取最佳帧 ----
    corners_l, corners_r, dz_used, chosen, size = [], [], [], [], None
    for it in items:
        bl = _best_frame(_list_images(it["left_dir"]), spec)
        br = _best_frame(_list_images(it["right_dir"]), spec)
        if bl is None or br is None:
            which = "/".join(n for n, v in (("左", bl), ("右", br)) if v is None)
            print(f"\n[错误] 深度 {it['name']}：{which} 图里检不出完整棋盘"
                  f"（{spec.describe()}；检查图是否清晰、板是否完整、BoardSpec 是否对）")
            return 2
        if size is None:
            size = bl[1]
        if bl[1] != size or br[1] != size:
            print(f"\n[错误] 深度 {it['name']} 图像尺寸不一致：左 {bl[1]} / 右 {br[1]} / "
                  f"首个 {size}；轴向模式要求所有帧同尺寸")
            return 2
        corners_l.append(bl[0])
        corners_r.append(br[0])
        dz_used.append(it["dz_m"])
        chosen.append({"depth": it["name"], "left": bl[2].name, "right": br[2].name,
                       "left_focus": round(bl[3], 1), "right_focus": round(br[3], 1)})
        print(f"[读图] {it['name']}: ΔZ={it['dz_m'] * 1000:.1f} mm  "
              f"左 {bl[2].name} / 右 {br[2].name}")

    # ---- 3. 求解 ----
    cfg = CalibConfig(expected_baseline_mm=args.expected_baseline_mm,
                      baseline_tolerance=float(args.baseline_tolerance))
    try:
        res = solve_axial_stereo(corners_l, corners_r, np.asarray(dz_used), spec, size,
                                 cfg=cfg, assume_parallel=bool(args.assume_parallel))
    except Exception as exc:                               # noqa: BLE001 - 输入/数值错误
        print(f"\n[错误] 轴向求解失败（{type(exc).__name__}: {exc}）")
        return 2

    # distortion_calibrated 从 AxialIntrinsics 取（solve_axial_stereo 不返回它）
    try:
        intr_l = solve_axial_intrinsics(corners_l, np.asarray(dz_used), spec)
        distortion_calibrated = bool(intr_l.distortion_calibrated)
        fx, fy = float(intr_l.fx), float(intr_l.fy)
    except Exception:                                      # noqa: BLE001
        distortion_calibrated, fx, fy = False, float("nan"), float("nan")

    print(f"\n[轴向] fx={fx:.2f}  fy={fy:.2f}  基线={res.baseline_mm:.2f} mm  "
          f"判定={res.grade}  rectification={res.rectification}")
    for w in res.warnings:
        print(f"  [注意] {w}")

    # ---- 4. 落盘 ----
    out = args.out or str(current_calibration(need_existing=False)[0])
    meta = {
        "method": "axial",
        "calibration_mode": "axial_depth",
        "board": spec.describe(),
        "board_spec": {"cols": int(spec.cols), "rows": int(spec.rows),
                       "square_mm": float(spec.square_mm)},
        "delta_z_mm": [round(z * 1000.0, 6) for z in dz_used],
        "assume_parallel": bool(args.assume_parallel),
        "expected_baseline_mm": (None if args.expected_baseline_mm is None
                                 else float(args.expected_baseline_mm)),
        "baseline_tolerance": float(args.baseline_tolerance),
        "distortion_calibrated": distortion_calibrated,
        "chosen_frames": chosen,
    }
    saved = _save_res(res, meta, out, bool(args.force_save))
    if saved is None:
        return 2
    npz, js, bypassed = saved
    print(f"[保存] {npz}\n[保存] {js}")

    if args.report:
        report = {
            "method": "axial",
            "calibration_mode": "axial_depth",
            "fx": fx,
            "fy": fy,
            "baseline_mm": float(res.baseline_mm),
            "grade": res.grade,
            "ok": bool(res.ok),
            "verdict": res.verdict,
            "rectification": res.rectification,
            "baseline_tolerance": float(args.baseline_tolerance),
            "assume_parallel": bool(args.assume_parallel),
            "distortion_calibrated": distortion_calibrated,
            "delta_z_mm": meta["delta_z_mm"],
            "chosen_frames": chosen,
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
