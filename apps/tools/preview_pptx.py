"""近似渲染指定页，便于肉眼检查排版。"""
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
    ap.add_argument("--slides", default="1")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "preview"))
    args = ap.parse_args()

    prs = Presentation(args.pptx)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for tok in args.slides.split(","):
        no = int(tok.strip())
        dst = render_slide(prs, no, out)
        print(f"  {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
