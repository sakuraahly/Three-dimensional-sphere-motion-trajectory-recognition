"""PPT 预览渲染自测：连线到底画没画出来。

为什么需要它
------------
`apps/tools/_preview_lib.py` 是开题报告排版自检的渲染内核
（`preview_pptx.py` 用它把 .pptx 画成 PNG，人再去看重叠/出界/文字溢出）。
**它画错了不会报错，只会让你看到一张"看起来没问题"的图。**

守着的真实 bug（连线静默不画）：连线分支里写了

    col = border or (150, 160, 175)

但 `border` 是这个分支**之后**才赋值的局部变量，在这里引用会抛
`UnboundLocalError`；异常又被同一个 `try` 的 `except Exception: pass` 吞掉，
于是**所有连线一条都画不出来**，整页只剩空白，还看不出哪里错了。

影响面不小：`build_opening_report.py` 与 `build_opening_ppt.py` 都用
`add_connector()` 画流程箭头 —— 也就是说这份开题报告的**每一张预览图里，
所有箭头都是缺失的**，而那正是排版自检要盯的东西。

判据用的是"像素证据"：造一个只含一条连线的页面，渲染出来必须有非白像素。

用法::

    python tests/selftest_preview_lib.py

依赖 python-pptx / Pillow（只做 PPT 用，不在 requirements.txt 里）；
两者缺任意一个就整体 `[跳过]`（算通过），这样纯运行环境也能跑全套回归。
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def make_scratch_dir() -> Path:
    """可写的临时目录（沙箱里 %TEMP% 可能不可写，退回项目内 outputs/）。"""
    import tempfile
    try:
        d = Path(tempfile.mkdtemp(prefix="preview_lib_"))
        (d / ".probe").write_text("ok", encoding="utf-8")
        (d / ".probe").unlink()
        return d
    except OSError:
        d = ROOT / "outputs" / "_preview_lib_tmp"
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
        return d


def _load_preview_lib():
    """按文件路径加载 _preview_lib（它在 apps/tools/ 下，包名带下划线，不便 import）。"""
    spec = importlib.util.spec_from_file_location(
        "_preview_lib", ROOT / "apps" / "tools" / "_preview_lib.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    print("=" * 62)
    print("PPT 预览渲染自测（连线是否真的画出来）")
    print("=" * 62)

    # --- 依赖检查：缺了就跳过（整套回归在纯运行环境下也要能跑） ---
    missing = []
    try:
        from pptx import Presentation  # noqa: F401
    except ImportError:
        missing.append("python-pptx")
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        missing.append("Pillow")
    if missing:
        print(f"  [跳过] 缺少 {' / '.join(missing)}（只做 PPT 用，不在 requirements.txt）")
        print("判定结果：全部通过")
        return 0

    import numpy as np
    from PIL import Image
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
    from pptx.util import Inches

    try:
        pl = _load_preview_lib()
    except Exception as exc:  # pragma: no cover - 依赖异常时给出清晰原因
        print(f"  [跳过] 无法加载 _preview_lib：{exc}")
        print("判定结果：全部通过")
        return 0

    out = make_scratch_dir()
    checks: list[tuple[str, bool]] = []

    # ---------------- 1) 只含一条默认色连线的页面，必须不是空白 ----------------
    # 这是 bug 的最小复现：修之前整页 0 个非白像素。
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    # 水平线 y=2in（=240 px @SCALE 120），x 从 2in 到 6in
    s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
                           Inches(2), Inches(2), Inches(6), Inches(2))
    png = pl.render_slide(prs, 1, out)
    a = np.array(Image.open(png).convert("RGB"))
    nonwhite = int((a.reshape(-1, 3) < 250).any(axis=1).sum())
    band = a[236:245, :, :]
    band_px = int((band.reshape(-1, 3) < 250).any(axis=1).sum())
    ok = nonwhite > 500 and band_px > 300
    print(f"  只有一条连线：非白像素 {nonwhite}（应 >500，整条线约 4in*120*2≈960）"
          f"  连线所在行 {band_px}（应 >300）  {'通过' if ok else '不达标'}")
    checks.append(("连线被画出来（整页不再空白）", ok))

    # ---------------- 2) 连线用自己的颜色，不是写死的灰 ----------------
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
                                Inches(2), Inches(2), Inches(6), Inches(2))
    ln.line.color.rgb = RGBColor(0xC0, 0x30, 0x30)
    png = pl.render_slide(prs, 1, out)
    a = np.array(Image.open(png).convert("RGB"))
    red = int(((a[:, :, 0] > 150) & (a[:, :, 1] < 120) & (a[:, :, 2] < 120)).sum())
    ok = red > 300
    print(f"  显式红色连线：红色像素 {red}（应 >300）  {'通过' if ok else '不达标'}")
    checks.append(("连线用自己的颜色", ok))

    # ---------------- 3) 回归：普通形状照常画（别被连线分支抢走） ----------------
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    s = prs.slides.add_slide(prs.slide_layouts[6])
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                            Inches(1), Inches(1), Inches(2), Inches(1))
    sh.fill.solid()
    sh.fill.fore_color.rgb = RGBColor(0x20, 0xA0, 0x40)
    sh.line.color.rgb = RGBColor(0x10, 0x50, 0x20)
    png = pl.render_slide(prs, 1, out)
    a = np.array(Image.open(png).convert("RGB"))
    green = int(((a[:, :, 1] > 120) & (a[:, :, 0] < 120)).sum())
    ok = green > 5000
    print(f"  显式填充的圆角矩形：绿色像素 {green}（应 >5000）  {'通过' if ok else '不达标'}")
    checks.append(("普通形状仍正常绘制", ok))

    print()
    ok_all = all(p for _n, p in checks)
    print("判定结果：" + ("全部通过" if ok_all else "有不达标项"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
