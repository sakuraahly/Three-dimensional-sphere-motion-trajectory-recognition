"""开题报告 PPT 合规自检（对照《PPT字号等.docx》+ GB/T 引用规范）。

    python apps/tools/check_opening_report.py outputs/开题报告_最终.pptx

检查项
  [1] 空白内容占位符 —— 会显示成"单击此处添加文本"和插图图标的元凶
  [2] 字号下限 —— 正文/表格不小于 20 号，参考文献 22 号
  [3] 字体 —— 中文宋体、英文数字 Times New Roman
  [4] 正文行距 —— 26 号及以上的**多行**内容须为 1.5 倍
      （单行小标题用 1.0 是合理的，不计入）
  [5] 页面数

内容页与模板页分开统计：模板自带的部分（封面小字、目录大字）不计入违规，
但仍会列出来供人工判断。
"""

from __future__ import annotations

import sys
from pathlib import Path

from pptx import Presentation
from pptx.util import Emu

NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
SONG, TIMES = "宋体", "Times New Roman"

# 本次生成的内容从第几页开始（1 基）；之前的页是用户自己的模板/正文
CONTENT_FROM = 17


def runs_of(slide):
    for sh in slide.shapes:
        tfs = []
        if sh.has_text_frame:
            tfs.append(sh.text_frame)
        if sh.has_table:
            tfs += [c.text_frame for r in sh.table.rows for c in r.cells]
        for tf in tfs:
            for p in tf.paragraphs:
                for r in p.runs:
                    if r.text.strip():
                        yield sh, p, r


def n_lines(p, width_in: float) -> int:
    """保守估行数：CJK 按 1.0 em、其他按 0.55 em。"""
    total = 0.0
    for r in p.runs:
        pt = r.font.size.pt if r.font.size else 18.0
        for ch in r.text:
            total += pt * (1.0 if ord(ch) > 0x2E80 else 0.55)
    usable = max(1.0, width_in * 72.0)
    return max(1, int(total / usable) + (1 if total % usable else 0))


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "outputs/开题报告_最终.pptx"
    if not Path(path).exists():
        print(f"[失败] 找不到要检查的文件：{path}")
        print("       请先用 build_opening_report.py 生成，例如：")
        print("         python apps/tools/build_opening_report.py "
              "--src <你的开题报告.pptx> --out outputs/开题报告_最终.pptx")
        print("       （生成方式见 docs/PPT生成流程.md）")
        return 2
    prs = Presentation(path)

    def is_content(i: int) -> bool:
        return i >= CONTENT_FROM

    # [1] 空白内容占位符
    ph = []
    for i, s in enumerate(prs.slides, 1):
        for sh in s.shapes:
            try:
                if not sh.is_placeholder:
                    continue
                t = str(sh.placeholder_format.type)
                if (t.startswith("OBJECT") or t.startswith("BODY")) \
                        and sh.has_text_frame and not sh.text_frame.text.strip():
                    ph.append((i, sh.placeholder_format.idx))
            except Exception:
                pass
    print(f"[1] 空白内容占位符 : {len(ph)}  {'OK' if not ph else ph[:5]}")

    # [2] 字号下限
    small_c, small_t = [], []
    for i, s in enumerate(prs.slides, 1):
        for sh, p, r in runs_of(s):
            if r.font.size and r.font.size.pt < 20:
                (small_c if is_content(i) else small_t).append(
                    (i, round(r.font.size.pt, 1), r.text[:20]))
    print(f"[2] 小于 20 号     : 内容页 {len(small_c)}  "
          f"{'OK' if not small_c else small_c[:4]}")
    if small_t:
        print(f"                    模板页 {len(small_t)}（用户原有，建议自行确认）："
              f"{small_t[:4]}")

    # [3] 字体
    tot = wrong_c = wrong_t = 0
    samp = []
    for i, s in enumerate(prs.slides, 1):
        for sh, p, r in runs_of(s):
            tot += 1
            rPr = r._r.find(NS + "rPr")
            ea = la = None
            if rPr is not None:
                e, l = rPr.find(NS + "ea"), rPr.find(NS + "latin")
                ea = e.get("typeface") if e is not None else None
                la = l.get("typeface") if l is not None else None
            if ea != SONG or la != TIMES:
                if is_content(i):
                    wrong_c += 1
                    if len(samp) < 4:
                        samp.append((i, ea, la, r.text[:16]))
                else:
                    wrong_t += 1
    print(f"[3] 字体不符       : 内容页 {wrong_c}/{tot}  "
          f"{'OK' if wrong_c == 0 else samp}")
    if wrong_t:
        print(f"                    模板页 {wrong_t}（用户原有）")

    # [4] 正文行距（只查多行的 26 号以上正文）
    ls_c, ls_t = [], []
    for i, s in enumerate(prs.slides, 1):
        for sh in s.shapes:
            if not sh.has_text_frame:
                continue
            try:
                w = Emu(sh.width).inches
            except Exception:
                continue
            for p in sh.text_frame.paragraphs:
                if not p.runs:
                    continue
                sz = next((r.font.size.pt for r in p.runs if r.font.size), None)
                ls = p.line_spacing
                if not sz or sz < 26 or not isinstance(ls, float):
                    continue
                if abs(ls - 1.5) < 0.01 or n_lines(p, w) <= 1:
                    continue
                txt = "".join(r.text for r in p.runs)[:24]
                (ls_c if is_content(i) else ls_t).append((i, sz, ls, txt))
    print(f"[4] 正文行距非 1.5 : 内容页 {len(ls_c)}  "
          f"{'OK' if not ls_c else ls_c[:4]}")
    if ls_t:
        print(f"                    模板页 {len(ls_t)}（用户原有）")

    print(f"[5] 总页数         : {len(prs.slides._sldIdLst)}")
    ok = not ph and not small_c and wrong_c == 0 and not ls_c
    print(f"\n结论：{'全部通过' if ok else '有需处理项（见上）'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
