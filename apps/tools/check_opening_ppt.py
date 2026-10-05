"""几何自检：检查生成的 PPT 有没有出界、压页码、文字装不下。

为什么需要它：python-pptx 只管把形状写进去，**不会告诉你文字溢出了没有**。
PowerPoint 打开时才会暴露，或者更糟 —— 自动缩小字号把版面搞乱。
所以这里按字号和框宽估算所需行数，和框高比对，提前把问题找出来。

    python apps/tools/check_opening_ppt.py
    python apps/tools/check_opening_ppt.py docs/开题报告_v1.pptx
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from pptx import Presentation
from pptx.util import Emu

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "docs" / "开题报告_v1.pptx"

SW, SH = 13.333, 7.5
PAGE_NO_Y = 6.82          # 页码占位符上沿
CJK_RATIO = 0.55          # ASCII 字宽约等于全角字的 0.55 倍


def text_units(s: str) -> float:
    """把字符串折算成"等效全角字数"。"""
    return sum(1.0 if ord(ch) > 0x2E80 else CJK_RATIO for ch in s)


def est_lines(text: str, size_pt: float, box_w_in: float) -> int:
    if not text.strip():
        return 0
    cap_units = max(1.0, (box_w_in * 72.0) / size_pt)   # 一行能放多少全角字
    return sum(max(1, math.ceil(text_units(p) / cap_units)) for p in text.split("\n"))


def main() -> int:
    pptx = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    if not pptx.exists():
        print(f"[失败] 找不到 {pptx}")
        return 1
    prs = Presentation(str(pptx))
    problems = 0
    print(f"检查 {pptx}   ({SW:.2f}x{SH:.2f} in)\n")

    for idx, slide in enumerate(prs.slides, start=1):
        if idx not in (11, 12, 13, 14):
            continue
        print(f"===== slide{idx} =====")
        for sh in slide.shapes:
            try:
                x, y = Emu(sh.left).inches, Emu(sh.top).inches
                w, h = Emu(sh.width).inches, Emu(sh.height).inches
            except Exception:
                continue
            name = sh.name
            if x < 0.2 or y < 0.15 or x + w > SW - 0.2 or y + h > SH - 0.1:
                print(f"  [出界] {name}: {x:.2f},{y:.2f} {w:.2f}x{h:.2f}")
                problems += 1
            if "编号" not in name and "sldNum" not in name and y + h > PAGE_NO_Y + 0.02:
                print(f"  [压页码区] {name}: 底边 {y + h:.2f} > {PAGE_NO_Y}")
                problems += 1
            if not sh.has_text_frame:
                continue
            need_h = 0.0
            for p in sh.text_frame.paragraphs:
                txt = "".join(r.text for r in p.runs)
                size = next((r.font.size.pt for r in p.runs if r.font.size), 12.0)
                ls = p.line_spacing if isinstance(p.line_spacing, float) else 1.15
                need_h += est_lines(txt, size, w) * size * ls * 1.30 / 72.0
                if p.space_after is not None:
                    need_h += p.space_after.pt / 72.0
            if need_h > h + 0.06:
                txt = sh.text_frame.text.replace("\n", " / ")[:46]
                print(f"  [可能溢出] {name}: 需≈{need_h:.2f}\" 框高 {h:.2f}\"「{txt}…」")
                problems += 1
        print()

    print("=" * 60)
    if problems:
        print(f"发现 {problems} 处需要留意（见上）")
    else:
        print("几何自检通过：无出界、无压页码、无明显文字溢出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
