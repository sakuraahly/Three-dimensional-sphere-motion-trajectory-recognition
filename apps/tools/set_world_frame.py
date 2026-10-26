"""设置世界坐标系：用一张"棋盘平放在地面上"的左相机图求 `T_world_cam`。

为什么单独一个工具
------------------
世界系是**一次性离线标定**（和双目标定同理）：拍一张、算一次、落盘，
运行期只读 `config/world_frame.json`。放进实时链路会让它多一个失败点，
而且没有人会为了跑一段视频去蹲下摆棋盘。

用法::

    python apps/tools/set_world_frame.py --image data/world_frame/board.png
    python apps/tools/set_world_frame.py --image board.png --dry-run
    python apps/tools/set_world_frame.py --image board.png --note "地面，球桌左侧"

退出码：0 成功 / 1 失败（打不开图、没标定、检不到板、rmse 超限）/ 2 参数错。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "sphere_tracker").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calib import BoardSpec, StereoRig                     # noqa: E402
from calib.world_frame import MAX_BOARD_RMSE_PX, from_board  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="用棋盘 PnP 求世界系（左相机 -> 地面），写 config/world_frame.json")
    ap.add_argument("--image", required=True,
                    help="棋盘平放在地面（或已知平面）上，用左相机拍的一张图")
    ap.add_argument("--board-cols", type=int, default=8, help="内角点列数（默认 8）")
    ap.add_argument("--board-rows", type=int, default=5, help="内角点行数（默认 5）")
    ap.add_argument("--board-square", type=float, default=25.0,
                    help="方格边长，毫米（默认 25）")
    ap.add_argument("--rig", default="config/stereo_calibration",
                    help="双目标定文件前缀（默认 config/stereo_calibration）")
    ap.add_argument("--out", default="config/world_frame.json",
                    help="输出文件（默认 config/world_frame.json）")
    ap.add_argument("--note", default="", help="写进文件的人读备注")
    ap.add_argument("--ground-axis", default="z", choices=("z", "-z"),
                    help="z=棋盘法向（默认）；-z=反向（板平放时 Z 朝上）")
    ap.add_argument("--dry-run", action="store_true", help="只算不写")
    args = ap.parse_args()
    if args.board_cols < 2 or args.board_rows < 2:
        ap.error("--board-cols / --board-rows 必须 >= 2（内角点数）")
    if args.board_square <= 0:
        ap.error("--board-square 必须 > 0（毫米）")
    return args


def main() -> int:
    args = parse_args()

    # 前置校验 1：没有标定就谈不上世界系（K1/D1 是 PnP 的输入）
    try:
        rig = StereoRig.load(args.rig)
    except FileNotFoundError:
        print(f"[错误] 找不到标定文件：{args.rig}.npz")
        print("  先做双目标定：启动程序.bat calibrate")
        return 1
    except ValueError as exc:
        print(f"[错误] 标定文件读不出来：{args.rig}（{exc}）")
        return 1

    # 前置校验 2：图必须存在（cv2.imread 失败只会返回 None，不给原因）
    if not Path(args.image).is_file():
        print(f"[错误] 找不到图片：{args.image}")
        return 1

    spec = BoardSpec(args.board_cols, args.board_rows, args.board_square)
    try:
        wf = from_board(args.image, spec, rig, ground_axis=args.ground_axis)
    except (ValueError, TypeError) as exc:
        print(f"[错误] {exc}")
        return 1

    rmse = "—" if wf.board_rmse_px is None else f"{wf.board_rmse_px:.2f} px"
    if args.dry_run:
        print("（--dry-run：只算不写）")
        print(wf.describe())
        print(f"棋盘重投影 rmse = {rmse}（上限 {MAX_BOARD_RMSE_PX:.2f} px）")
        return 0

    try:
        path = wf.save(args.out)
    except OSError as exc:
        print(f"[错误] 写不进去：{args.out}（{exc}）")
        return 1

    print(f"已写入 {path}")
    print(wf.describe())
    print(f"棋盘重投影 rmse = {rmse}（上限 {MAX_BOARD_RMSE_PX:.2f} px）")
    print("请确认棋盘是平放在地面上的，否则世界系的 z 轴不是竖直方向")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
