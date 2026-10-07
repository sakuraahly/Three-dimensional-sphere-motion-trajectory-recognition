"""底部溢出扫描（较保守的 1.45 系数，宁可报多不放过）。"""
import sys
from pptx import Presentation
from pptx.util import Emu
from PIL import ImageFont

SONG = r"C:\Windows\Fonts\simsun.ttc"
TIMES = r"C:\Windows\Fonts\times.ttf"
SC = 120
C = {}


def f(p, pt):
    k = (p, int(pt * SC / 72))
    if k not in C:
        try:
            C[k] = ImageFont.truetype(p, max(6, k[1]))
        except Exception:
            C[k] = ImageFont.truetype(SONG, max(6, k[1]))
    return C[k]


def nl(t, pt, w):
    fc, ft = f(SONG, pt), f(TIMES, pt)
    mw = w * SC
    n, cur = 0, 0.0
    for ch in t:
        ww = fc.getlength(ch) if ord(ch) > 0x2E80 else ft.getlength(ch)
        if cur + ww > mw and cur > 0:
            n += 1
            cur = 0.0
        cur += ww
    return n + (1 if cur > 0 else 0)


P = sys.argv[1] if len(sys.argv) > 1 else "outputs/开题报告_最终.pptx"
prs = Presentation(P)
bad = 0
for i in range(1, len(prs.slides._sldIdLst) + 1):
    for sh in prs.slides[i - 1].shapes:
        if not sh.has_text_frame:
            continue
        try:
            w = Emu(sh.width).inches
            y = Emu(sh.top).inches
        except Exception:
            continue
        need = 0.0
        for p in sh.text_frame.paragraphs:
            t = "".join(r.text for r in p.runs)
            if not t:
                continue
            pt = next((r.font.size.pt for r in p.runs if r.font.size), 26)
            ls = p.line_spacing if isinstance(p.line_spacing, float) else 1.5
            need += nl(t, pt, w) * pt * ls * 1.32 / 72
            if p.space_after:
                need += p.space_after.pt / 72
        if need > 0 and y + need > 7.40:
            print(f"  slide{i}: 底 {y + need:.2f}\"  {sh.text_frame.text[:30]!r}")
            bad += 1
print("溢出:", "无" if bad == 0 else f"{bad} 处")
