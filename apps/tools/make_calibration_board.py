"""生成可打印的棋盘格标定板（PDF + PNG）。

用法::

    python apps/tools/make_calibration_board.py
    python apps/tools/make_calibration_board.py --cols 8 --rows 5 --square 25 --page a4-landscape

生成到 assets/boards/。**打印时必须选"实际大小 / 100%"，不要选"适应页面"**，
打印完用尺子量一下方格边长 —— 量出来是多少，标定时 --square 就填多少。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# --- 让脚本在任意目录深度都能找到项目根 ---
ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw

MM_PER_INCH = 25.4
DPI = 300
PAGES_MM = {
    "a4-landscape": (297.0, 210.0),
    "a4-portrait": (210.0, 297.0),
    "a3-landscape": (420.0, 297.0),
    "letter-landscape": (279.4, 215.9),
}


def mm2px(mm: float) -> int:
    return int(round(mm / MM_PER_INCH * DPI))


def make_board(cols: int, rows: int, square_mm: float, page: str, out_dir: Path) -> list[Path]:
    """cols/rows 是内角点数；方格数是 cols+1 乘 rows+1。"""
    pw_mm, ph_mm = PAGES_MM[page]
    n_x, n_y = cols + 1, rows + 1
    board_w_mm = n_x * square_mm
    board_h_mm = n_y * square_mm
    margin_mm = min(20.0, (pw_mm - board_w_mm) / 2 - 2, (ph_mm - board_h_mm) / 2 - 2)
    if margin_mm < 5:
        raise ValueError(
            f"棋盘 {board_w_mm:.0f}x{board_h_mm:.0f} mm 放不进 {page} ({pw_mm:.0f}x{ph_mm:.0f} mm)，"
            f"请把 --square 调小或换更大的纸"
        )

    W, H = mm2px(pw_mm), mm2px(ph_mm)
    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)

    x0 = (pw_mm - board_w_mm) / 2.0
    y0 = (ph_mm - board_h_mm) / 2.0 - 4.0
    sq = mm2px(square_mm)
    for i in range(n_x):
        for j in range(n_y):
            if (i + j) % 2 == 0:      # 左上角为黑格
                px0 = mm2px(x0 + i * square_mm)
                py0 = mm2px(y0 + j * square_mm)
                d.rectangle([px0, py0, px0 + sq - 1, py0 + sq - 1], fill=0)

    # 让检测更稳：在棋盘外再围一圈白边（打印纸本身是白的，这里只是保证边界干净）
    d.rectangle(
        [mm2px(x0) - 2, mm2px(y0) - 2, mm2px(x0 + board_w_mm) + 1, mm2px(y0 + board_h_mm) + 1],
        outline=255, width=1,
    )

    caption = (f"{cols}x{rows} inner corners | {square_mm:g} mm squares | print at 100% (actual size)"
               f" | then pass: --cols {cols} --rows {rows} --square {square_mm:g}")
    d.text((mm2px(10), mm2px(y0 + board_h_mm + 8)), caption, fill=0)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"chessboard_{cols}x{rows}_{square_mm:g}mm_{page}"
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    img.save(png, dpi=(DPI, DPI))
    img.convert("L").save(pdf, "PDF", resolution=DPI)
    print(f"  {board_w_mm:.0f} x {board_h_mm:.0f} mm 棋盘，{page} 纸张")
    print(f"    {png}")
    print(f"    {pdf}")
    return [png, pdf]


def main() -> int:
    ap = argparse.ArgumentParser(description="生成可打印的棋盘格标定板")
    ap.add_argument("--cols", type=int, default=8, help="内角点列数")
    ap.add_argument("--rows", type=int, default=5, help="内角点行数")
    ap.add_argument("--square", type=float, default=25.0, help="方格边长（毫米）")
    ap.add_argument("--page", default="a4-landscape", choices=sorted(PAGES_MM))
    ap.add_argument("--out", default="assets/boards")
    ap.add_argument("--all", action="store_true", help="生成常用组合")
    args = ap.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    print("生成标定板：")
    if args.all:
        make_board(8, 5, 25.0, "a4-landscape", out)
        make_board(8, 5, 20.0, "a4-portrait", out)
    else:
        make_board(args.cols, args.rows, args.square, args.page, out)
    print()
    print("打印与核对：")
    print("  1) 打印时选『实际大小 / 100%』，不要选『适应页面 / 缩放』")
    print("  2) 打印后用尺子量方格边长：量出来是多少，--square 就填多少")
    print("  3) 贴在硬纸板/亚克力上 —— 板不平会让标定白做")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
