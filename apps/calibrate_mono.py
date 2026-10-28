"""单目内参标定入口：一组棋盘图 → `config/mono_calibration.json`。

为什么它必须是一个独立入口（2026-10-27 计划书 §3.3 / B-100）
------------------------------------------------------------
``calib/stereo_calib.py:641`` 早就有单目内参算法，chain 标定的左右两支一直在用它，
但它**没有对外的命令行入口** —— 于是想标单目的人只能去读源码、或者误以为
"单目不用标定，填个 fx 就行"。事实是：按 60° 视场猜出来的 fx 误差 1% 就是
0.6 m 处 6 mm，而单目本身在 0.6 m 的预算只有 ±7 mm。

本入口只做三件事：读图、调 :func:`calib.mono_calib.calibrate_mono`、存结果。
**判定逻辑全在库里**（RMS / 视角数 / 姿态覆盖 / 板面积），这里不重复实现；
``--check`` 时连存都不存，只报告。

用法::

    python apps/calibrate_mono.py --dir outputs/mono_calib_frames
    python apps/calibrate_mono.py --dir <目录> --check          # 只校验
    python apps/calibrate_mono.py --dir <目录> --cols 9 --rows 6 --square 20

退出码：0 = 标定完成（或 --check 通过）；1 = 这组图不可用（带原因）；2 = 参数错。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from calib.board import BoardSpec                            # noqa: E402
from calib.mono_calib import (MonoCalibError, QUALITY,       # noqa: E402
                              calibrate_mono, save_mono)

DEFAULT_DIR = ROOT / "outputs" / "mono_calib_frames"
DEFAULT_OUT = ROOT / "config" / "mono_calibration.json"

LOG = logging.getLogger("calibrate_mono")


def _setup_logging() -> Path:
    """日志落到 ``outputs/logs/calibrate_mono_<时间>.log``（与其它入口一致）。"""
    d = ROOT / "outputs" / "logs"
    d.mkdir(parents=True, exist_ok=True)
    import time
    path = d / f"calibrate_mono_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(path, encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
    )
    return path


def _image_paths(d: Path) -> list[Path]:
    """目录里的图片（png/jpg，不递归，按名字排序）。"""
    items = [p for p in d.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    return sorted(items)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="单目内参标定（棋盘图 → config/mono_calibration.json）")
    ap.add_argument("--dir", default=str(DEFAULT_DIR),
                    help=f"标定图目录（默认 {DEFAULT_DIR.relative_to(ROOT)}）")
    ap.add_argument("--cols", type=int, default=8, help="棋盘内角点列数（默认 8）")
    ap.add_argument("--rows", type=int, default=5, help="棋盘内角点行数（默认 5）")
    ap.add_argument("--square", type=float, default=25.0,
                    help="方格边长 mm（默认 25）")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help=f"输出文件（默认 {DEFAULT_OUT.relative_to(ROOT)}）")
    ap.add_argument("--check", action="store_true",
                    help="只校验并报告，不写文件")
    ap.add_argument("--fix-principal-point", action="store_true",
                    help="标定时固定主点在图像中心（图少时更稳，但可能引入偏差）")
    args = ap.parse_args(argv)

    log_path = _setup_logging()
    print(f"本次日志：{log_path}")

    d = Path(args.dir)
    if not d.is_dir():
        print(f"[错误] 目录不存在：{d}")
        print("       先拍图：在单目窗第①步点「拍标定图」，或用手册里的采集命令")
        return 2
    try:
        spec = BoardSpec(cols=args.cols, rows=args.rows, square_mm=args.square)
    except ValueError as exc:
        print(f"[参数错误] {exc}")
        return 2

    paths = _image_paths(d)
    print(f"[输入] {d}")
    print(f"[输入] {len(paths)} 张图，板 {spec.cols}×{spec.rows}，方格 {spec.square_mm:g} mm")
    if not paths:
        print("[错误] 目录里没有图片")
        return 2
    LOG.info("开始标定：%d 张，板 %dx%d，方格 %g mm",
             len(paths), spec.cols, spec.rows, spec.square_mm)

    try:
        calib = calibrate_mono(paths, spec,
                               fix_principal_point=args.fix_principal_point)
    except MonoCalibError as exc:
        print()
        print("[不可用] " + str(exc))
        if exc.rejected:
            print(f"[逐图] 共 {len(exc.rejected)} 张被拒：")
            for name, why in exc.rejected[:12]:
                print(f"        · {name}：{why}")
            if len(exc.rejected) > 12:
                print(f"        …… 其余 {len(exc.rejected) - 12} 张略（见日志）")
        print()
        print("[下一步] 按上面的原因重拍。最常见的两条：")
        print(f"        · 板要摆到画面四角与中央（现在需要覆盖 ≥{QUALITY['min_regions']}/5 个区域）")
        print("        · 每个姿态倾斜 30–45°，不要一直平放在画面中间")
        LOG.error("标定不可用：%s", exc)
        return 1

    print()
    print(calib.report())
    LOG.info("标定完成：fx=%.2f fy=%.2f RMS=%.3f 视角=%d",
             calib.fx, calib.fy, calib.rms_px, calib.views)

    if args.check:
        print()
        print("[--check] 只校验，不写文件。去掉 --check 即保存。")
        return 0

    out = Path(args.out)
    try:
        written = save_mono(calib, out)
    except OSError as exc:
        print(f"[错误] 写不进 {out}：{exc}")
        return 1
    print()
    print(f"[已保存] {written}")
    print("[下一步] 运行期会自动读它（run_realtime 会按标定值算焦距，不再按 60° 视场猜）")
    LOG.info("已保存：%s", written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
