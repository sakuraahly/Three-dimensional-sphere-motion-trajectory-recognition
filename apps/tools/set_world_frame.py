"""设置世界坐标系：用一张"棋盘平放在地面上"的左相机图求 `T_world_cam`。

为什么单独一个工具
------------------
世界系是**一次性离线标定**（和双目标定同理）：拍一张、算一次、落盘，
运行期只读 `config/world_frame.json`。放进实时链路会让它多一个失败点，
而且没有人会为了跑一段视频去蹲下摆棋盘。

用法::

    python apps/tools/set_world_frame.py --image data/world_frame/board.png
    python apps/tools/set_world_frame.py --image board.png --dry-run
    python apps/tools/set_world_frame.py --image board.png --ground-axis -z  # Z 朝上
    python apps/tools/set_world_frame.py --image board.png --note "地面，球桌左侧"

拍图有两条硬要求（脚本会自己校验，不合格就退出 1，不给"凑合能用"的世界系）：

1. **分辨率必须与 `--rig` 那份标定一致**（内参是按标定分辨率标出来的）；
2. 棋盘要平放在地面、完整落在画面内、对焦清晰，重投影 rmse ≤
   `MAX_BOARD_RMSE_PX`（2.0 px）。

`--ground-axis` 决定 Z 的符号：``z``（默认）与棋盘法向同向 —— 板平放、图案
朝上拍时 **+Z 指向地面以下**；``-z`` 反过来（+Z 朝上），代价是 **+Y 同时反向**
（右手系）。两种取值都会由 `describe()` 打印出来，`--dry-run` 也一样。

退出码：0 成功 / 1 失败（打不开图、没标定、检不到板、分辨率不一致、rmse 超限）/ 2 参数错。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "sphere_tracker").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps._common import current_calibration              # noqa: E402
from calib import BoardSpec, StereoRig                     # noqa: E402
from calib.world_frame import MAX_BOARD_RMSE_PX, from_board  # noqa: E402


#: `--ground-axis` 的取值。`-z` 以短横线开头，见 `_join_axis_value()`。
_AXIS_VALUES = ("z", "-z")


def _join_axis_value(argv: list[str]) -> list[str]:
    """把 ``--ground-axis -z`` 拼成 ``--ground-axis=-z``。

    **这不是洁癖，是必需的**：argparse 看到单独的 ``-z`` 会当成"另一个选项"，
    于是 ``--ground-axis -z`` 直接报 ``expected one argument`` 并退出 2 ——
    而 ``-z`` 恰恰是这个参数最常用的取值。界面（`apps/_actions.py` 的
    ``_flag``）拼出来的就是两段 ``["--ground-axis", "-z"]``，不修这一处，
    「Z 轴朝上」在 GUI 里点了就是一个退出码 2 的失败进程。

    只拼 ``-z``：其它以 ``-`` 开头的 token 原样交给 argparse 报错，
    免得把"漏了取值"这种真错误静默吞成"取值是这个开关"。
    """
    out: list[str] = []
    i = 0
    while i < len(argv):
        if (argv[i] == "--ground-axis" and i + 1 < len(argv)
                and argv[i + 1] == "-z"):
            out.append("--ground-axis=-z")
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="用棋盘 PnP 求世界系（左相机 -> 地面），写 config/world_frame.json")
    ap.add_argument("--image", required=True,
                    help="棋盘平放在地面（或已知平面）上，用左相机拍的一张图")
    ap.add_argument("--board-cols", type=int, default=8, help="内角点列数（默认 8）")
    ap.add_argument("--board-rows", type=int, default=5, help="内角点行数（默认 5）")
    ap.add_argument("--board-square", type=float, default=25.0,
                    help="方格边长，毫米（默认 25）")
    # 默认跟当前装置档案的标定（2026-11-10 裁决 B5：世界系必须与运行几何同源）
    ap.add_argument("--rig", default=str(current_calibration(need_existing=False)[0]),
                    help="双目标定文件前缀（默认 = 当前装置档案的标定，未配则顶层）")
    ap.add_argument("--out", default="config/world_frame.json",
                    help="输出文件（默认 config/world_frame.json）")
    ap.add_argument("--note", default="", help="写进文件的人读备注")
    ap.add_argument("--ground-axis", default="z", choices=_AXIS_VALUES,
                    help="z=与棋盘法向同向（默认；板平放、图案朝上拍时 +Z 朝地下）；"
                         "-z=反向（+Z 朝上，代价是 +Y 同时反向）。"
                         "两种写法都行：--ground-axis -z / --ground-axis=-z")
    ap.add_argument("--dry-run", action="store_true", help="只算不写")
    args = ap.parse_args(_join_axis_value(list(sys.argv[1:] if argv is None else argv)))
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
        print("  先做双目标定：run.bat calibrate")
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
        # rig_path 只用于报错文案（"哪份标定"）；分辨率校验在 from_board() 里。
        wf = from_board(args.image, spec, rig, ground_axis=args.ground_axis,
                        rig_path=args.rig)
    except (ValueError, TypeError) as exc:
        print(f"[错误] {exc}")
        return 1

    rmse = "—" if wf.board_rmse_px is None else f"{wf.board_rmse_px:.2f} px"
    if args.dry_run:
        print("（--dry-run：只算不写）")
        print(wf.describe())
        print(f"棋盘重投影 rmse = {rmse}（上限 {MAX_BOARD_RMSE_PX:.2f} px）")
        print("以上是世界系的完整定义（含 ground_axis 与 Z/Y 的符号）；"
              "确认无误后去掉 --dry-run 再跑一次才会写文件")
        return 0

    try:
        path = wf.save(args.out)
    except OSError as exc:
        print(f"[错误] 写不进去：{args.out}（{exc}）")
        return 1

    print(f"已写入 {path}")
    print(wf.describe())
    print(f"棋盘重投影 rmse = {rmse}（上限 {MAX_BOARD_RMSE_PX:.2f} px）")
    print("请确认棋盘是平放在地面上的，否则世界系的 Z 轴不是竖直方向；"
          "Z 朝上/朝下以上面那行 ground_axis 为准")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
