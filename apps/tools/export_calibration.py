"""把双目标定参数导出成 JSON / YAML / TOML（分相机 + 双目整体）。

用法::

    python apps/tools/export_calibration.py
    python apps/tools/export_calibration.py --calib config/stereo_calibration --out outputs/calib_export
    python apps/tools/export_calibration.py --formats json,toml

产物（每种格式各三份）::

    left_camera.<ext>    左相机 K / D / image_size / 畸变模型
    right_camera.<ext>   右相机同上
    stereo_rig.<ext>     R / T / E / F / R1 / R2 / P1 / P2 / Q / 基线 / 质量
    summary.txt          人读摘要 + 文件清单

为什么单独一个工具而不是塞进 `calibrate_stereo.py`：
标定一次、导出多次；导出只读文件，不碰相机，也不该让标定程序多一个失败点。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "sphere_tracker").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calib import StereoRig                      # noqa: E402
from calib.export import FORMATS, write_export    # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="导出双目标定参数（分相机 + 整体）")
    ap.add_argument("--calib", default="config/stereo_calibration",
                    help="标定文件前缀（不带扩展名）")
    ap.add_argument("--out", default="outputs/calib_export",
                    help="输出目录（不存在会创建）")
    ap.add_argument("--formats", default="json,yaml,toml",
                    help=f"逗号分隔，可选：{','.join(FORMATS)}")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    try:
        rig = StereoRig.load(args.calib)
    except FileNotFoundError as exc:
        print(f"[错误] {exc}")
        print("  先跑双目标定：apps/calibrate_stereo.py --left left --right right")
        return 1

    print(rig.describe())
    print()
    try:
        paths = write_export(rig, args.out, formats)
    except (ValueError, RuntimeError) as exc:
        print(f"[错误] {exc}")
        return 1

    print(f"已导出 {len(paths)} 个文件到：{args.out}")
    for path in paths:
        print(f"  {path.name}")
    print()
    print("读取示例（stereo_rig 同理）：")
    print("  JSON : json.load(open('left_camera.json', encoding='utf-8'))")
    print("  YAML : yaml.safe_load(open('left_camera.yaml', encoding='utf-8'))")
    print("  TOML : tomllib.load(open('left_camera.toml', 'rb'))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
