"""近似渲染指定页，便于肉眼检查排版。

    python apps/tools/preview_pptx.py outputs/开题报告_最终.pptx --slides 17,18,20

输出 PNG 到 --out（默认 outputs/preview/）。这是 python-pptx + PIL 的**近似**渲染，
不能替代 PowerPoint，但足以发现文字出界、元素重叠、内容被切。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _preview_lib import render_slide  # noqa: E402

from pptx import Presentation  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pptx")
    ap.add_argument("--slides", default="1",
                    help="逗号分隔的页码，从 1 开始，例如 17,18,20")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "preview"))
    args = ap.parse_args()

    if not Path(args.pptx).exists():
        print(f"[失败] 找不到文件：{args.pptx}")
        print("       生成方式见 docs/PPT生成流程.md")
        return 2

    prs = Presentation(args.pptx)
    total = len(prs.slides._sldIdLst)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for tok in args.slides.split(","):
        no = int(tok.strip())
        if not 1 <= no <= total:
            print(f"  [跳过] slide{no} 超出范围（共 {total} 页）")
            continue
        dst = render_slide(prs, no, out)
        print(f"  {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
