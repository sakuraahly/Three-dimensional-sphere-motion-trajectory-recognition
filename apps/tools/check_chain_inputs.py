"""chain 标定的输入自检：**拍完先跑这个**，别等标定跑完才发现图不行。

为什么需要它（2026-10-28 的真实事故）
------------------------------------
当天用 90° 装置跑 chain，等到标定结束才看到「左目 RMS 2.18 px 超标 / 参考图
位姿离散 11.83°」—— 那意味着**十几分钟的拍摄全部作废**。而这两个问题在
拍完的那一刻就能查出来：

1. **内参段的姿态覆盖**：板一直放在画面中央时，15 张图只覆盖 5 个区域里的
   1 个。畸变系数不可辨，内参与主点/畸变互相耦合，RMS 必然大。
2. **参考段的板有没有动**：单板模式的前提是"参考图期间板**不动**"（T_AB=I）。
   板挪了 11° 时两路的"同一个位置"其实不是同一个位置，链出来的外参是错的。

本工具只读图片、不写任何文件、不需要相机，也不改标定。

用法::

    python apps/tools/check_chain_inputs.py --root outputs/chain/20261028-135416
    python apps/tools/check_chain_inputs.py --root <目录> --cols 8 --rows 5 --square 25

退出码：0 = 可以跑 chain；1 = 有必须先修的问题；2 = 目录/参数不对。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# 按**标记**上溯到含 `sphere_tracker/` 的祖先，不数层数：
# 数层数（parent.parent.parent）在文件被移动、或有人加一层包装目录时会静默指错，
# 而 `selftest_entrypoints` 对这条是硬的（本项目明确的规矩）。
ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "sphere_tracker").is_dir())
sys.path.insert(0, str(ROOT))

from calib.board import BoardSpec, detect_board          # noqa: E402
from calib.mono_calib import QUALITY, _region_index      # noqa: E402

#: 参考段"板没动"的判据：同一角点在不同张之间的最大位移（像素）。
#: 实测：板不动时 < 2 px（亚像素抖动）；板挪 11° 时几十上百 px。
REF_STILL_PX = 5.0
#: 内参段每路至少几张（与 calibrate_stereo 的 chain 门槛一致）
MIN_VIEWS_PER_SIDE = 15
#: 内参段建议的倾斜：只用角点外接框的"短边/长边"比粗略判断有没有斜
#: （不做 PnP，因为这里还没有内参可依赖）


def _load_gray(path: Path):
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def _corners(path: Path, spec: BoardSpec):
    g = _load_gray(path)
    if g is None:
        return None, None, "读不出"
    obs = detect_board(g, spec)
    if not obs.found or obs.corners is None:
        return None, None, f"未检出（{obs.note or '未知'}）"
    return g.shape, obs, ""


def check_intrinsics(d: Path, spec: BoardSpec, side: str) -> tuple[list[str], bool]:
    """检查一路内参图：张数、检出、覆盖、坏图。返回（问题列表, 是否可接受）。"""
    problems: list[str] = []
    if not d.is_dir():
        return [f"{side}：目录不存在 {d}"], False
    files = sorted(list(d.glob("*.png")) + list(d.glob("*.jpg")))
    if len(files) < MIN_VIEWS_PER_SIDE:
        problems.append(f"{side}：只有 {len(files)} 张，至少 {MIN_VIEWS_PER_SIDE} 张")

    regions: dict[int, int] = {}
    bad: list[str] = []
    areas: list[float] = []
    for p in files:
        shape, obs, why = _corners(p, spec)
        if obs is None:
            bad.append(f"{p.name}（{why}）")
            continue
        c = obs.corners.reshape(-1, 2)
        cx, cy = float(c[:, 0].mean()), float(c[:, 1].mean())
        idx = _region_index(cx, cy, shape[1], shape[0])
        if idx >= 0:
            regions[idx] = regions.get(idx, 0) + 1
        areas.append(obs.area_ratio)
        if obs.area_ratio < QUALITY["min_area_ratio"]:
            bad.append(f"{p.name}（板太小 {obs.area_ratio:.1%}）")

    n_ok = len(files) - len(bad)
    print(f"  {side}: {len(files)} 张，可用 {n_ok}，覆盖 {len(regions)}/5 个区域 "
          f"{sorted(regions)}，板占画面 {min(areas):.1%}~{max(areas):.1%}"
          if areas else f"  {side}: {len(files)} 张，全部不可用")
    if len(regions) < QUALITY["min_regions"]:
        problems.append(
            f"{side}：姿态只覆盖 {len(regions)}/5 个区域（至少 {QUALITY['min_regions']}）"
            f" —— 板要摆到画面四角与中央，不能一直放中间")
    if bad:
        problems.append(f"{side}：{len(bad)} 张不可用：" + "、".join(bad[:4])
                        + ("…" if len(bad) > 4 else ""))
    return problems, not problems


def check_reference(d: Path, spec: BoardSpec, side: str) -> tuple[list[str], bool]:
    """检查参考段：板**必须没动**（角点位移 < REF_STILL_PX）。"""
    problems: list[str] = []
    if not d.is_dir():
        return [f"{side}：参考目录不存在 {d}"], False
    files = sorted(list(d.glob("*.png")) + list(d.glob("*.jpg")))
    if not files:
        return [f"{side}：参考目录是空的 {d}"], False
    corners: list[np.ndarray] = []
    for p in files:
        _shape, obs, why = _corners(p, spec)
        if obs is None:
            problems.append(f"{side}：{p.name} {why}")
            continue
        corners.append(obs.corners.reshape(-1, 2))
    if len(corners) < 2:
        problems.append(f"{side}：可用参考图只有 {len(corners)} 张（至少 2 张）")
        return problems, False
    base = corners[0]
    spread = max(float(np.abs(c - base).max()) for c in corners[1:])
    print(f"  {side}: {len(corners)} 张，角点最大位移 {spread:.1f} px "
          f"（门限 {REF_STILL_PX:.0f}）")
    if spread > REF_STILL_PX:
        problems.append(
            f"{side}：参考图之间板动了（角点最大位移 {spread:.0f} px > "
            f"{REF_STILL_PX:.0f}）—— 单板模式要求板在参考段**完全不动**")
    return problems, not problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="chain 标定输入自检（不写文件、不开相机）")
    ap.add_argument("--root", required=True, help="四段目录的父目录")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--square", type=float, default=25.0)
    args = ap.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        print(f"[错误] 目录不存在：{root}")
        return 2
    try:
        spec = BoardSpec(cols=args.cols, rows=args.rows, square_mm=args.square)
    except ValueError as exc:
        print(f"[参数错误] {exc}")
        return 2

    print("=" * 72)
    print(f"chain 输入自检：{root}")
    print("=" * 72)
    print("-- 内参段（板应摆满画面四角+中央，每个姿态倾斜 30–45°）--")
    p1, ok1 = check_intrinsics(root / "left", spec, "left")
    p2, ok2 = check_intrinsics(root / "right", spec, "right")
    print("-- 参考段（板在这几张之间必须完全不动）--")
    p3, ok3 = check_reference(root / "ref_left", spec, "ref_left")
    p4, ok4 = check_reference(root / "ref_right", spec, "ref_right")

    problems = p1 + p2 + p3 + p4
    print()
    if problems:
        print("必须先修的问题：")
        for i, msg in enumerate(problems, 1):
            print(f"  {i}) {msg}")
        print("\n判定结果：不能跑 chain（修完重拍后再自检一次）")
        return 1
    print("全部通过：可以跑 chain 了")
    print("下一步：")
    print(f"  启动程序.bat test 之外，手动跑：")
    print(f"  python apps/calibrate_stereo.py --method chain --same-board "
          f"--left-dir {root / 'left'} --right-dir {root / 'right'} "
          f"--ref-left {root / 'ref_left'} --ref-right {root / 'ref_right'} "
          f"--cols {args.cols} --rows {args.rows} --square {args.square:g}")
    print("\n判定结果：可以跑 chain")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
