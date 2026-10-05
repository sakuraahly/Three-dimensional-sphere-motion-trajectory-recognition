"""用 PIL 把生成的 PPT 近似渲染成 PNG，便于肉眼检查排版。

为什么不用 PowerPoint 导出：pywin32 在当前环境装不上（wheel 文件被权限拒绝），
PowerPoint COM 走不通。所以这里自己画：用 python-pptx 读每个形状的位置、填充、
边框、字号，再用 PIL 画出来。

**这不是精确渲染**，不能替代 PowerPoint（主题、阴影、字体度量都会略有差异）。
它的用途是检查：有没有重叠、有没有出界、文字有没有被挤爆。
正式定稿前建议还是在 PowerPoint 里翻一遍。

    python apps/tools/preview_opening_ppt.py
    python apps/tools/preview_opening_ppt.py docs/开题报告_v1.pptx --out outputs/ppt_preview
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.util import Emu

ROOT = Path(__file__).resolve().parents[2]
SCALE = 120                                   # 每英寸像素 → 13.33*120 = 1600 px 宽
FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"      # 微软雅黑
FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"


def px(inches: float) -> float:
    return inches * SCALE


def get_font(size_pt: float, bold: bool):
    path = FONT_BOLD if bold else FONT_PATH
    px_size = max(6, int(round(size_pt * SCALE / 72.0)))
    try:
        return ImageFont.truetype(path, px_size)
    except Exception:
        try:
            return ImageFont.truetype(FONT_PATH, px_size)
        except Exception:
            return ImageFont.load_default()


def solid_rgb(fill):
    try:
        if fill.type == 1:                     # MSO_FILL.SOLID
            c = fill.fore_color.rgb
            return (c[0], c[1], c[2])
    except Exception:
        pass
    return None


def line_rgb(shape):
    try:
        c = shape.line.color.rgb
        return (c[0], c[1], c[2])
    except Exception:
        return None


def wrap_text(text: str, font, max_w: float) -> list[str]:
    """按像素宽度折行；中文逐字断行。"""
    lines: list[str] = []
    for para in text.split("\n"):
        if not para:
            lines.append("")
            continue
        cur = ""
        for ch in para:
            if font.getlength(cur + ch) <= max_w or not cur:
                cur += ch
            else:
                lines.append(cur)
                cur = ch
        lines.append(cur)
    return lines


def _run_style(runs, default_size=12.0):
    size = next((r.font.size.pt for r in runs if r.font.size), default_size)
    bold = bool(runs and runs[0].font.bold)
    col = (30, 40, 55)
    try:
        if runs and runs[0].font.color and runs[0].font.color.rgb:
            c = runs[0].font.color.rgb
            col = (c[0], c[1], c[2])
    except Exception:
        pass
    return size, bold, col


def draw_slide(prs, no: int, out_dir: Path) -> Path:
    slide = prs.slides[no - 1]
    W = int(round(Emu(prs.slide_width).inches * SCALE))
    H = int(round(Emu(prs.slide_height).inches * SCALE))
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)

    for sh in slide.shapes:
        try:
            x, y = px(Emu(sh.left).inches), px(Emu(sh.top).inches)
            w, h = px(Emu(sh.width).inches), px(Emu(sh.height).inches)
        except Exception:
            continue
        st = str(sh.shape_type)
        border = line_rgb(sh)

        # --- 表格 ---
        if sh.has_table:
            tbl = sh.table
            cy = y
            for ri, row in enumerate(tbl.rows):
                cx = x
                rh = px(Emu(row.height).inches)
                for ci, cell in enumerate(row.cells):
                    cw = px(Emu(tbl.columns[ci].width).inches)
                    bg = solid_rgb(cell.fill)
                    d.rectangle([cx, cy, cx + cw, cy + rh], fill=bg,
                                outline=(200, 210, 220))
                    if cell.text:
                        runs = [r for p in cell.text_frame.paragraphs for r in p.runs]
                        size, bold, col = _run_style(runs, 10.0)
                        f = get_font(size, bold)
                        lines = wrap_text(cell.text, f, cw - 8)
                        lh = size * SCALE / 72.0 * 1.2
                        ty = cy + max(1, (rh - len(lines) * lh) / 2)
                        for ln in lines:
                            d.text((cx + 5, ty), ln, font=f, fill=col)
                            ty += lh
                    cx += cw
                cy += rh
            continue

        # --- 连接线 ---
        if "CONNECTOR" in st or not hasattr(sh, "fill"):
            continue

        # --- 自选图形 / 文本框 ---
        fill = solid_rgb(sh.fill)
        if fill or border:
            d.rectangle([x, y, x + w, y + h], fill=fill, outline=border,
                        width=2 if border else 0)

        if not sh.has_text_frame:
            continue
        ty = y + 2
        for p in sh.text_frame.paragraphs:
            txt = "".join(r.text for r in p.runs)
            if not txt:
                continue
            size, bold, col = _run_style(p.runs)
            f = get_font(size, bold)
            ls = p.line_spacing if isinstance(p.line_spacing, float) else 1.15
            align = str(p.alignment)
            for ln in wrap_text(txt, f, w):
                tx = x
                if "CENTER" in align:
                    tx = x + (w - f.getlength(ln)) / 2
                d.text((tx, ty), ln, font=f, fill=col)
                ty += size * SCALE / 72.0 * 1.25 * ls
            if p.space_after:
                ty += p.space_after.pt * SCALE / 72.0

    dst = out_dir / f"slide{no}.png"
    img.save(dst)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description="近似渲染 PPT 四页成 PNG")
    ap.add_argument("pptx", nargs="?",
                    default=str(ROOT / "docs" / "开题报告_v1.pptx"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "ppt_preview"))
    args = ap.parse_args()

    src = Path(args.pptx)
    if not src.exists():
        print(f"[失败] 找不到 {src}")
        return 1
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    prs = Presentation(str(src))
    for no in (11, 12, 13, 14):
        dst = draw_slide(prs, no, out_dir)
        print(f"  {dst}")
    print(f"\n预览目录：{out_dir}")
    print("注意：这是近似渲染，不能替代在 PowerPoint 里过一遍。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
