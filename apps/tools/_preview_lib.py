"""近似渲染库：用 python-pptx 读形状 + PIL 绘制。

从 apps/tools/preview_opening_ppt.py 抽出来复用（那份是给四页文稿用的）。
不能替代 PowerPoint，只用于查重叠/出界/文字溢出。
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pptx.util import Emu

SCALE = 120
FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"
CJK = 0.55


def px(v: float) -> float:
    return v * SCALE


def _font(pt: float, bold: bool):
    path = FONT_BOLD if bold else FONT_PATH
    try:
        return ImageFont.truetype(path, max(6, int(round(pt * SCALE / 72.0))))
    except Exception:
        try:
            return ImageFont.truetype(FONT_PATH, max(6, int(round(pt * SCALE / 72.0))))
        except Exception:
            return ImageFont.load_default()


def _solid(fill):
    try:
        if fill.type == 1:
            c = fill.fore_color.rgb
            return (c[0], c[1], c[2])
    except Exception:
        pass
    return None


def _line(shape):
    try:
        c = shape.line.color.rgb
        return (c[0], c[1], c[2])
    except Exception:
        return None


def _wrap(text: str, font, max_w: float) -> list[str]:
    lines = []
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


def _style(runs, dflt=12.0):
    size = next((r.font.size.pt for r in runs if r.font.size), dflt)
    bold = bool(runs and runs[0].font.bold)
    col = (30, 40, 55)
    try:
        if runs and runs[0].font.color and runs[0].font.color.type is not None:
            c = runs[0].font.color.rgb
            col = (c[0], c[1], c[2])
    except Exception:
        pass
    return size, bold, col


def render_slide(prs, no: int, out_dir: Path) -> Path:
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

        if sh.has_table:
            tbl = sh.table
            cy = y
            for row in tbl.rows:
                cx = x
                rh = px(Emu(row.height).inches)
                for ci, cell in enumerate(row.cells):
                    cw = px(Emu(tbl.columns[ci].width).inches)
                    bg = _solid(cell.fill)
                    d.rectangle([cx, cy, cx + cw, cy + rh], fill=bg,
                                outline=(205, 214, 224))
                    if cell.text:
                        runs = [r for p in cell.text_frame.paragraphs for r in p.runs]
                        size, bold, col = _style(runs, 10.0)
                        f = _font(size, bold)
                        lines = _wrap(cell.text, f, cw - 8)
                        lh = size * SCALE / 72.0 * 1.22
                        ty = cy + max(1, (rh - len(lines) * lh) / 2)
                        for ln in lines:
                            lw = f.getlength(ln)
                            tx = cx + 5
                            if cell.text_frame.paragraphs:
                                al = str(cell.text_frame.paragraphs[0].alignment)
                                if "CENTER" in al:
                                    tx = cx + (cw - lw) / 2
                            d.text((tx, ty), ln, font=f, fill=col)
                            ty += lh
                    cx += cw
                cy += rh
            continue

        if "CONNECTOR" in st or not hasattr(sh, "fill"):
            # 连接线：几何存的是 off+ext 加可选 flipH/flipV，
            # 不能只看 left/top/width/height（那会漏掉翻转）。
            try:
                off = sh._element.spPr.find(
                    "{http://schemas.openxmlformats.org/drawingml/2006/main}xfrm")
                if off is not None:
                    o = off.find("{http://schemas.openxmlformats.org/drawingml/2006/main}off")
                    e = off.find("{http://schemas.openxmlformats.org/drawingml/2006/main}ext")
                    fh = off.get("flipH") == "1"
                    fv = off.get("flipV") == "1"
                    x0 = px(int(o.get("x")) / 914400)
                    y0 = px(int(o.get("y")) / 914400)
                    x1 = x0 + px(int(e.get("cx")) / 914400)
                    y1 = y0 + px(int(e.get("cy")) / 914400)
                    if fh:
                        x0, x1 = x1, x0
                    if fv:
                        y0, y1 = y1, y0
                    col = border or (150, 160, 175)
                    d.line([x0, y0, x1, y1], fill=col, width=2)
            except Exception:
                pass
            continue

        fill = _solid(sh.fill)
        border = _line(sh)
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
            size, bold, col = _style(p.runs)
            f = _font(size, bold)
            ls = p.line_spacing if isinstance(p.line_spacing, float) else 1.15
            align = str(p.alignment)
            marL = 0
            pPr = p._p.find(
                "{http://schemas.openxmlformats.org/drawingml/2006/main}pPr")
            if pPr is not None and pPr.get("marL"):
                marL = px(int(pPr.get("marL")) / 914400)
            for ln in _wrap(txt, f, w - marL):
                tx = x + marL
                if "CENTER" in align:
                    tx = x + (w - f.getlength(ln)) / 2
                d.text((tx, ty), ln, font=f, fill=col)
                ty += size * SCALE / 72.0 * 1.25 * ls
            if p.space_after:
                ty += p.space_after.pt * SCALE / 72.0

    dst = out_dir / f"slide{no}.png"
    img.save(dst)
    return dst
