"""生成开题报告 PPT 的"研究思路"四页（slide 11–14：装置搭建 / 软件控制 / 数据采集 / 测量精度）。

用法::

    # 从模板生成 docs/开题报告_v1.pptx（默认）
    python apps/tools/build_opening_ppt.py

    # 另存到别处
    python apps/tools/build_opening_ppt.py --out D:/某处/开题报告_v2.pptx

    # 模板换位置了
    python apps/tools/build_opening_ppt.py --src D:/.../开题报告模版.pptx

设计原则
--------
1. **只填 slide 11–14**，其余 15 页原样保留（背景 / 理论 / 时间规划归组内其他人）。
2. 一律走 python-pptx，**不手改 pptx 里的 XML** —— 由库负责关系与内容类型，
   产出一定是合法 pptx，也方便之后在 PowerPoint 里继续编辑。
3. 生成方式是"读模板 -> 只清空 11–14 的正文形状 -> 写新内容 -> 另存新文件"，
   母版、版式、页码占位符都保留；**绝不就地覆盖模板**。
4. 排版对齐模板已有的惯例（用 python-pptx 实测得到，非猜测）：
   - 标题占位符：宋体 32pt，x=0.79" y=0.22"；
   - 正文可用区：y 从 1.50" 到 6.62"（页码占位符在 6.82"）；
   - 正文左边界 x=0.79"，满宽 11.86"；
   - 正文字体统一 **微软雅黑**（避免宋体小字号笔画糊在一起），
     强调色语义固定：蓝=主题、红=风险/待验证、绿=已达成。
5. 图都用**基础形状**拼（矩形/箭头/折线），每个形状都能在 PowerPoint 里单独选中美化。

配套工具（改完建议依次跑一遍）::

    python apps/tools/check_opening_ppt.py      # 几何自检：出界/压页码/文字溢出
    python apps/tools/preview_opening_ppt.py    # 用 PIL 近似渲染成 PNG，便于肉眼检查
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]        # 项目根（本脚本在 apps/tools/ 下）
# 模板 pptx 是**个人交付物、不进版本库**，所以这里不写任何人的本机目录。
# 默认先找项目内 assets/templates/；实际用 --src 指到自己的文件
# （见 docs/PPT生成流程.md），换机器不需要改源码。
SRC_TEMPLATE = ROOT / "assets" / "templates" / "开题报告模版.pptx"

# 配色（对齐模板的蓝灰基调）
INK = RGBColor(0x1F, 0x28, 0x37)         # 正文近黑
DIM = RGBColor(0x5B, 0x66, 0x77)         # 次要灰
BLUE = RGBColor(0x1F, 0x4E, 0x79)        # 主蓝（标题条 / 表头）
BLUE_L = RGBColor(0xDE, 0xEA, 0xF6)      # 浅蓝底
RED = RGBColor(0xC0, 0x39, 0x2B)         # 警告红
RED_L = RGBColor(0xFC, 0xE8, 0xE6)       # 浅红底
GREEN = RGBColor(0x1E, 0x7A, 0x46)       # 通过绿
GREEN_L = RGBColor(0xE3, 0xF3, 0xE8)
GRAY_L = RGBColor(0xF2, 0xF4, 0xF7)      # 浅灰底
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BORDER = RGBColor(0xBF, 0xCF, 0xE0)

FONT = "微软雅黑"

# 正文可用区
Y0 = 1.50
Y1 = 6.62
ML = 0.79
CW = 11.86
# 小节标题的框高。**必须留够**：框高不够时 PowerPoint 会自动缩小字号，
# 或者文字视觉上溢出压到下一块。量算出来的经验值（15.5pt 一行约需 0.31"）。
H_TITLE = 0.34
H_TEXT = 0.26

# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------
def _tf(shape, wrap: bool = True):
    tf = shape.text_frame
    tf.word_wrap = wrap
    tf.margin_left = Emu(0)
    tf.margin_right = Emu(0)
    tf.margin_top = Emu(0)
    tf.margin_bottom = Emu(0)
    return tf


def add_box(slide, x, y, w, h, fill=None, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
            line_w: float = 0.75, adj: float | None = None):
    sh = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(line_w)
    sh.shadow.inherit = False
    if adj is not None and shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        try:
            sh.adjustments[0] = adj
        except Exception:
            pass
    return sh


def add_text(slide, x, y, w, h, lines, *, size=14, color=INK, bold=False,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, line_spacing=1.16,
             space_after=3, shrink=False):
    """lines: 字符串，或 [(文本, {覆盖属性}), ...]。"""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = _tf(tb)
    tf.vertical_anchor = anchor
    if shrink:
        from pptx.enum.text import MSO_AUTO_SIZE
        tf.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
    if isinstance(lines, str):
        lines = [lines]
    first = True
    for item in lines:
        opts = {}
        if isinstance(item, tuple):
            item, opts = item
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = opts.get("align", align)
        p.line_spacing = opts.get("line_spacing", line_spacing)
        p.space_after = Pt(opts.get("space_after", space_after))
        if opts.get("space_before"):
            p.space_before = Pt(opts["space_before"])
        r = p.add_run()
        r.text = item
        f = r.font
        f.name = opts.get("font", FONT)
        f.size = Pt(opts.get("size", size))
        f.bold = opts.get("bold", bold)
        f.color.rgb = opts.get("color", color)
    return tb


def add_rich(slide, x, y, w, h, runs, *, size=14, color=INK, align=PP_ALIGN.LEFT,
             line_spacing=1.16, space_after=3):
    """单个段落里混排多种样式：runs = [(文本, {属性}), ...]。"""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = _tf(tb)
    p = tf.paragraphs[0]
    p.alignment = align
    p.line_spacing = line_spacing
    p.space_after = Pt(space_after)
    for text, opts in runs:
        r = p.add_run()
        r.text = text
        f = r.font
        f.name = opts.get("font", FONT)
        f.size = Pt(opts.get("size", size))
        f.bold = opts.get("bold", False)
        f.color.rgb = opts.get("color", color)
    return tb


def add_bullets(slide, x, y, w, h, items, *, size=13, gap=6, line_spacing=1.14):
    """items: [(文本, 样式dict)]，每条前加"▪ "。样式里可给 color/bold。"""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = _tf(tb)
    first = True
    for text, opts in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.line_spacing = line_spacing
        p.space_after = Pt(opts.get("space_after", gap))
        r = p.add_run()
        r.text = "▪ " + text
        f = r.font
        f.name = FONT
        f.size = Pt(opts.get("size", size))
        f.bold = opts.get("bold", False)
        f.color.rgb = opts.get("color", INK)
    return tb


def add_line(slide, x1, y1, x2, y2, color=BORDER, width=1.25, dash=False):
    ln = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1),
                                    Inches(x2), Inches(y2))
    ln.line.color.rgb = color
    ln.line.width = Pt(width)
    if dash:
        from pptx.enum.dml import MSO_LINE_DASH_STYLE
        ln.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    return ln


def add_arrow(slide, x, y, w, h, color=BLUE, shape=MSO_SHAPE.DOWN_ARROW):
    sh = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def style_table(tbl, *, header_fill=BLUE, header_color=WHITE, size=11,
                header_size=11.5, band=GRAY_L, row_h=0.3):
    """统一表格样式：表头深蓝白字、正文斑马纹。"""
    for ri, row in enumerate(tbl.rows):
        row.height = Inches(row_h)
        for cell in row.cells:
            cell.margin_left = Inches(0.05)
            cell.margin_right = Inches(0.04)
            cell.margin_top = Inches(0.01)
            cell.margin_bottom = Inches(0.01)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            if ri == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = header_fill
            elif band is not None and ri % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = band
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = WHITE
            for p in cell.text_frame.paragraphs:
                p.line_spacing = 1.0
                p.space_after = Pt(0)
                for r in p.runs:
                    r.font.name = FONT
                    r.font.size = Pt(header_size if ri == 0 else size)
                    r.font.bold = (ri == 0)
                    r.font.color.rgb = header_color if ri == 0 else INK


def set_cell(tbl, r, c, text, *, size=None, bold=None, color=None,
             align=PP_ALIGN.LEFT):
    cell = tbl.cell(r, c)
    cell.text = text
    for p in cell.text_frame.paragraphs:
        p.alignment = align
        for run in p.runs:
            run.font.name = FONT
            if size:
                run.font.size = Pt(size)
            if bold is not None:
                run.font.bold = bold
            if color is not None:
                run.font.color.rgb = color


def add_table(slide, x, y, w, rows, cols, *, row_h=0.3, col_w=None):
    shp = slide.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(w),
                                 Inches(row_h * rows))
    tbl = shp.table
    if col_w:
        total = sum(col_w)
        for i, cw in enumerate(col_w):
            tbl.columns[i].width = Emu(int(Inches(w) * cw / total))
    return tbl


# --------------------------------------------------------------------------
# slide 11 · 装置搭建
# --------------------------------------------------------------------------
def build_slide11(slide):
    # ---- 左栏：硬件清单 ----
    add_rich(slide, ML, Y0, 6.1, H_TITLE,
             [("硬件清单", {"bold": True, "size": 15.5, "color": BLUE})])
    rows = [
        ("相机 ×2", "aoni A25，同型号", "已有"),
        ("支架", "硬质横梁，刚性固定", "待搭建"),
        ("基线 B", "待实验标定（见下方说明）", "待定"),
        ("标定板", "8×5 内角点 / 25 mm，贴硬纸板", "图已生成"),
        ("测试用球", "实验室单摆实验铁球", "已有"),
        ("通用性验证", "2–9 cm，多颜色 / 材质 / 尺寸", "待准备"),
        ("同步源", "拍手 / 闪光", "待搭建"),
        ("计算", "笔记本，纯 CPU 实时", "已有"),
        ("（可选）云台", "PID 闭环验证", "可选"),
    ]
    tbl = add_table(slide, ML, Y0 + 0.36, 6.1, len(rows) + 1, 3, row_h=0.26,
                    col_w=[1.25, 3.5, 1.0])
    for c, h in enumerate(("部件", "选定方案", "状态")):
        set_cell(tbl, 0, c, h, align=PP_ALIGN.CENTER if c != 1 else PP_ALIGN.LEFT)
    for i, (a, b, c) in enumerate(rows, start=1):
        set_cell(tbl, i, 0, a, bold=True)
        set_cell(tbl, i, 1, b)
        col = DIM if c in ("可选", "待定") else GREEN
        set_cell(tbl, i, 2, c, color=col, align=PP_ALIGN.CENTER)
    style_table(tbl, row_h=0.26, size=10.5, header_size=11)

    # ---- 左栏下部：两条硬约束 ----
    yc = Y1 - 1.72
    add_rich(slide, ML, yc, 6.1, H_TITLE,
             [("两条硬约束", {"bold": True, "size": 15.5, "color": RED})])
    add_bullets(slide, ML, yc + 0.36, 6.1, 1.42, [
        ("两台相机必须刚性固定：整体移动没关系（R, T 是常数），"
         "两台各自移动则三角测量在原理上失效。", {"color": INK}),
        ("同型号相机索引会在插拔/重启后互换：画面完全正常，但标定的 R, T 失效。"
         "对策：贴 L/R 标签 + 拍摄前复检（未动 0.08 px vs 碰动 27.9 px，差 250 倍）。",
         {"color": INK}),
    ], size=11.5, gap=3, line_spacing=1.10)

    # ---- 右栏：装置示意图 ----
    DX, DW = 7.15, 5.5
    add_rich(slide, DX, Y0, DW, H_TITLE,
             [("装置示意图", {"bold": True, "size": 15.5, "color": BLUE})])

    # 横梁
    add_box(slide, DX + 0.35, Y0 + 0.46, DW - 0.7, 0.20, fill=RGBColor(0x8A, 0x93, 0xA3),
            shape=MSO_SHAPE.RECTANGLE)
    # 两台相机
    add_box(slide, DX + 0.30, Y0 + 0.66, 1.0, 0.42, fill=BLUE,
            shape=MSO_SHAPE.RECTANGLE)
    add_box(slide, DX + DW - 1.30, Y0 + 0.66, 1.0, 0.42, fill=BLUE,
            shape=MSO_SHAPE.RECTANGLE)
    add_text(slide, DX + 0.30, Y0 + 0.74, 1.0, H_TEXT, ["左相机 L"], size=10.5,
             color=WHITE, align=PP_ALIGN.CENTER)
    add_text(slide, DX + DW - 1.30, Y0 + 0.74, 1.0, H_TEXT, ["右相机 R"], size=10.5,
             color=WHITE, align=PP_ALIGN.CENTER)

    # 基线标注（整行放得下，别让它折行；压在相机框下方，避免视觉重叠）
    # 注意不要用 "↔"：部分字体缺这个字形，会渲染成方块。用 ／ 代替更稳。
    ybase = Y0 + 1.16
    add_line(slide, DX + 1.30, ybase, DX + DW - 1.30, ybase, color=RED, width=1.5)
    add_text(slide, DX + 0.10, ybase + 0.08, DW - 0.2, H_TEXT,
             ["基线 B（深度精度 ／ 视场重叠 的折中，待实测）"], size=9.5, color=RED,
             align=PP_ALIGN.CENTER)

    # 两路视场（梯形用折线近似）+ 底边
    ytop, ybot = Y0 + 1.50, Y0 + 2.55
    add_line(slide, DX + 0.80, ytop, DX + 1.85, ybot, color=BLUE, width=1.0)
    add_line(slide, DX + 0.80, ytop, DX + DW - 0.80, ybot, color=BLUE, width=1.0, dash=True)
    add_line(slide, DX + DW - 0.80, ytop, DX + DW - 1.85, ybot, color=BLUE, width=1.0)
    add_line(slide, DX + DW - 0.80, ytop, DX + 0.80, ybot, color=BLUE, width=1.0, dash=True)
    add_line(slide, DX + 1.85, ybot, DX + DW - 1.85, ybot, color=BLUE, width=1.0)

    # 公共可见区
    add_box(slide, DX + 1.95, Y0 + 2.20, DW - 3.90, 0.68, fill=BLUE_L, line=BLUE,
            shape=MSO_SHAPE.RECTANGLE, line_w=1.0)
    add_text(slide, DX + 1.95, Y0 + 2.27, DW - 3.90, H_TEXT, ["两路公共可见区"],
             size=11, color=BLUE, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, DX + 1.95, Y0 + 2.55, DW - 3.90, H_TEXT,
             ["球的运动必须全落在这里"], size=9.5, color=DIM, align=PP_ALIGN.CENTER)

    # ---- 右栏下部：待办 ----
    yt = Y1 - 1.72
    add_rich(slide, DX, yt, DW, H_TITLE,
             [("待办清单", {"bold": True, "size": 15.5, "color": BLUE})])
    add_bullets(slide, DX, yt + 0.36, DW, 1.32, [
        ("打印标定板（选“实际大小 100%”），打印后用尺子量方格边长", {}),
        ("贴在硬纸板/亚克力上——板不平，标定白做", {}),
        ("两台相机装到同一根横梁，刚性固定、对准同一片区域", {}),
        ("实测确定基线 B（按深度精度与视场重叠的折中定）", {}),
        ("游标卡尺实测铁球直径并记录（单目对照与长度基准都要用）", {}),
        ("索引 0 那台换到 USB 3.0 口（现在 USB 2.0，白白损失分辨率）", {}),
    ], size=10.5, gap=0.5, line_spacing=1.0)


# --------------------------------------------------------------------------
# slide 12 · 软件控制
# --------------------------------------------------------------------------
def build_slide12(slide):
    # ---- 上：三层模块图（三个横条）----
    bands = [
        ("装置层", "两台相机各自后台取“最新帧” → 成对取帧（按同步偏移对齐）", BLUE, BLUE_L),
        ("算法层", "sphere_tracker/ 单目感知库 ×2 路　│　calib/ 双目标定　│　"
                   "stereo/ 同步·配对·三角测量", RGBColor(0x2E, 0x6D, 0xA4),
         RGBColor(0xE8, 0xF1, 0xF9)),
        ("应用层", "标定程序 │ 复检程序 │ 实时程序 │ 离线批处理 │ 回归测试 ×5",
         RGBColor(0x3F, 0x7C, 0xB0), GRAY_L),
    ]
    y = Y0
    for name, desc, bar, bg in bands:
        add_box(slide, ML, y, 0.68, 0.68, fill=bg, line=BORDER, shape=MSO_SHAPE.RECTANGLE)
        add_box(slide, ML, y, 1.35, 0.68, fill=bar, shape=MSO_SHAPE.RECTANGLE)
        add_text(slide, ML, y + 0.22, 1.35, 0.3, [name], size=13.5, color=WHITE,
                 bold=True, align=PP_ALIGN.CENTER)
        add_text(slide, ML + 1.55, y + 0.10, CW - 1.75, 0.5, [desc], size=12.5,
                 color=INK, anchor=MSO_ANCHOR.MIDDLE)
        y += 0.76
    # 可选层
    add_box(slide, ML, y, CW, 0.42, fill=WHITE, line=BORDER,
            shape=MSO_SHAPE.RECTANGLE)
    add_line(slide, ML, y, ML + CW, y, color=BORDER, width=1.0, dash=True)
    add_rich(slide, ML + 0.18, y + 0.09, CW - 0.36, H_TEXT,
             [("（可选）", {"color": DIM, "size": 11.5}),
              ("PID 云台闭环 —— 执行层验证，不是核心算法；时间紧可以不做，不影响主线交付",
               {"color": DIM, "size": 11.5})])

    # ---- 下左：一次开发、两路复用 ----
    yb = y + 0.66
    add_rich(slide, ML, yb, 6.05, H_TITLE,
             [("一次开发、两路复用", {"bold": True, "size": 14.5, "color": BLUE})])
    add_text(slide, ML, yb + 0.36, 6.05, 1.60, [
        "单目感知库对每一路是同一套代码，并行调用两次即可；双目部分只负责把两路合成三维。",
        "这是“工作量可控”的直接证据——不是为双目重写一遍。",
    ], size=11.5, line_spacing=1.20, space_after=8)

    # ---- 下右：三个工程取舍 ----
    add_rich(slide, 7.15, yb, 5.5, H_TITLE,
             [("三个工程取舍（体现“不是调库”）", {"bold": True, "size": 14.5, "color": BLUE})])
    add_bullets(slide, 7.15, yb + 0.36, 5.5, 1.80, [
        ("标定两级检测：粗检 12 ms 判断值不值得收，值得收的那帧才做全分辨率精检 40 ms"
         "（穷举模式单帧要 302 ms，绝不能每帧开）", {}),
        ("自动挑帧七道门限：清晰度 / 面积 / 贴边 / 曝光 / 静止 / 位姿多样性 / 覆盖均匀", {}),
        ("相机运动估计多数派优先：特征点+RANSAC → 网格点+LK → 相位相关只兜底"
         "（相位相关会锁到运动物体上）", {}),
    ], size=10.5, gap=3, line_spacing=1.10)

    # ---- 底部：性能余量（这段文字会占两行，框要留够高）----
    add_box(slide, ML, Y1 - 0.62, CW, 0.62, fill=GREEN_L, line=None,
            shape=MSO_SHAPE.RECTANGLE)
    add_rich(slide, ML + 0.15, Y1 - 0.54, CW - 0.3, 0.56,
             [("性能余量　", {"bold": True, "size": 12, "color": GREEN}),
              ("单路 6 ms/帧，两路串行 12 ms；60 fps 预算 16.7 ms，且两路可并行。"
               "实测单目整链路 34 fps（验收线 25 fps）——实时性是余量，不是约束。",
               {"size": 11.5, "color": INK})], line_spacing=1.05)


# --------------------------------------------------------------------------
# slide 13 · 数据采集
# --------------------------------------------------------------------------
def build_slide13(slide):
    # ---- 上：数据流流程条 ----
    steps = [("① 标定数据", "15–30 组\n棋盘位姿"),
             ("② 时间同步", "拍手/闪光\n求帧偏移"),
             ("③ 运动录制", "左右各一路\n参数全固定"),
             ("④ 对齐取帧", "按偏移配对\n七道门限"),
             ("⑤ 存档", "npz / csv\n+质量报告")]
    n = len(steps)
    gap = 0.20
    bw = (CW - gap * (n - 1)) / n
    for i, (name, desc) in enumerate(steps):
        x = ML + i * (bw + gap)
        add_box(slide, x, Y0, bw, 0.86, fill=BLUE_L, line=BLUE,
                shape=MSO_SHAPE.RECTANGLE, line_w=1.0)
        add_text(slide, x, Y0 + 0.09, bw, 0.26, [name], size=12, color=BLUE,
                 bold=True, align=PP_ALIGN.CENTER)
        add_text(slide, x, Y0 + 0.37, bw, 0.44, desc.split("\n"), size=9.5,
                 color=DIM, align=PP_ALIGN.CENTER, line_spacing=1.0, space_after=0)
        if i < n - 1:
            add_arrow(slide, x + bw + 0.02, Y0 + 0.30, gap - 0.04, 0.26,
                      color=BLUE, shape=MSO_SHAPE.RIGHT_ARROW)

    # ---- 主体：四块 ----
    yL = Y0 + 1.00
    # 左：标定数据的采集
    add_rich(slide, ML, yL, 6.05, H_TITLE,
             [("标定数据的采集", {"bold": True, "size": 13.5, "color": BLUE})])
    add_bullets(slide, ML, yL + 0.34, 6.05, 1.20, [
        ("两台相机同时拍棋盘格，15–30 组位姿，覆盖视场各区域并含倾斜姿态", {}),
        ("程序自动挑帧（七道门限），剔除重投影误差偏大的图", {}),
        ("标定不需要时间同步：只收板停稳的帧，静止目标任意时刻看起来都一样", {}),
    ], size=10.5, gap=1.5, line_spacing=1.05)

    # 左：存档格式
    ys = yL + 1.48
    add_rich(slide, ML, ys, 6.05, H_TITLE,
             [("存档格式", {"bold": True, "size": 13.5, "color": BLUE})])
    fr = add_table(slide, ML, ys + 0.34, 6.05, 5, 2, row_h=0.24, col_w=[1.0, 2.6])
    for c, h in enumerate(("类型", "格式")):
        set_cell(fr, 0, c, h)
    for i, (a, b) in enumerate([
        ("标定参数", ".npz（K1,D1,K2,D2,R,T,P1,P2,Q）+ .json 质量报告"),
        ("轨迹", ".csv（frame, t, x1y1, x2y2, XYZ, 平滑后, 预测值）"),
        ("视频", "左右各一份 .mp4"),
        ("图表", ".png / Plotly .html"),
    ], start=1):
        set_cell(fr, i, 0, a, bold=True)
        set_cell(fr, i, 1, b)
    style_table(fr, row_h=0.24, size=10, header_size=10.5)

    # 右：运动数据的采集
    add_rich(slide, 7.15, yL, 5.5, H_TITLE,
             [("运动数据的采集", {"bold": True, "size": 13.5, "color": BLUE})])
    mp = add_table(slide, 7.15, yL + 0.34, 5.5, 6, 2, row_h=0.245, col_w=[1.5, 4.0])
    for c, h in enumerate(("参数", "取值")):
        set_cell(mp, 0, c, h)
    for i, (a, b) in enumerate([
        ("分辨率/帧率", "1920×1080 @ 30 fps（相机实测上限）"),
        ("曝光/白平衡/焦距", "全部固定，否则颜色线索漂移"),
        ("相机距目标", "1.5–2.5 m，球成像半径 20–45 px"),
        ("运动类型", "自由落体 / 斜抛 / 落地弹跳 / 斜面滚动"),
        ("重复次数", "每种 运动×光照×球体，5–10 次"),
    ], start=1):
        set_cell(mp, i, 0, a, bold=True)
        set_cell(mp, i, 1, b)
    style_table(mp, row_h=0.245, size=10, header_size=10.5)

    # 右下：同步的具体做法
    yn = yL + 1.90
    add_rich(slide, 7.15, yn, 5.5, H_TITLE,
             [("时间同步的具体做法（可复现的操作）", {"bold": True, "size": 13.5,
                                          "color": RED})])
    add_box(slide, 7.15, yn + 0.36, 5.5, 1.36, fill=RED_L, line=None,
            shape=MSO_SHAPE.RECTANGLE)
    add_rich(slide, 7.30, yn + 0.44, 5.2, H_TEXT,
             [("开录前", {"size": 10.5, "color": INK}),
              ("拍一下手 / 闪一下灯", {"size": 10.5, "bold": True, "color": RED}),
              (" → 找出两路该事件的帧号", {"size": 10.5, "color": INK})])
    add_rich(slide, 7.30, yn + 0.68, 5.2, H_TEXT,
             [("例：左 143 帧、右 151 帧 → ", {"size": 10.5, "color": INK}),
              ("偏移 +8 帧", {"size": 10.5, "bold": True, "color": RED})])
    add_rich(slide, 7.30, yn + 0.98, 5.2, 0.60,
             [("程序按偏移自动配对取帧。已实测：人为平移 0 / +8 / −11 / +37 帧"
               "全部找回，置信度 1.000，", {"size": 10, "color": DIM}),
              ("不需要手工挪视频", {"size": 10, "bold": True, "color": DIM}),
              ("。记录表须登记“本次同步偏移”与“本次标定复检值”。",
               {"size": 10, "color": DIM})], line_spacing=1.05)


# --------------------------------------------------------------------------
# slide 14 · 测量精度
# --------------------------------------------------------------------------
def build_slide14(slide):
    # ---- 上：误差传导 + 核心公式 ----
    add_rich(slide, ML, Y0 - 0.02, 6.6, H_TITLE,
             [("误差是怎么从像素传到三维的", {"bold": True, "size": 13.5, "color": BLUE})])
    yf = Y0 + 0.38
    chain = [("2D 检测误差", "δd（像素）"),
             ("视差误差", "δd（像素）"),
             ("三维深度误差", "ε_Z")]
    bx, bw2, gp = ML, 1.85, 0.28
    for i, (a, b) in enumerate(chain):
        x = bx + i * (bw2 + gp)
        add_box(slide, x, yf, bw2, 0.68, fill=BLUE_L, line=BLUE,
                shape=MSO_SHAPE.RECTANGLE, line_w=1.0)
        add_text(slide, x, yf + 0.08, bw2, H_TEXT, [a], size=11, color=BLUE,
                 bold=True, align=PP_ALIGN.CENTER)
        add_text(slide, x, yf + 0.36, bw2, H_TEXT, [b], size=11, color=INK,
                 align=PP_ALIGN.CENTER)
        if i < 2:
            add_arrow(slide, x + bw2 + 0.03, yf + 0.23, gp - 0.06, 0.22, color=BLUE,
                      shape=MSO_SHAPE.RIGHT_ARROW)
    add_text(slide, ML, yf + 0.76, 6.6, H_TEXT,
             ["亚像素细化把检测误差压到 0.06 px；但误差传到深度时被 Z² 放大"],
             size=10, color=DIM)

    # 公式块（右）
    add_box(slide, 7.15, Y0 - 0.02, 5.5, 1.34, fill=RGBColor(0xFF, 0xF7, 0xE6),
            line=RGBColor(0xE0, 0xA8, 0x30), shape=MSO_SHAPE.RECTANGLE, line_w=1.25)
    add_text(slide, 7.15, Y0 + 0.06, 5.5, H_TEXT, ["核心误差公式"], size=11,
             color=RGBColor(0x9C, 0x66, 0x00), bold=True, align=PP_ALIGN.CENTER)
    add_rich(slide, 7.15, Y0 + 0.34, 5.5, 0.4,
             [("ε_Z  ≈  Z² · δd / (f · B)", {"size": 19, "bold": True,
                                              "color": RGBColor(0x9C, 0x66, 0x00),
                                              "font": "Cambria"})],
             align=PP_ALIGN.CENTER)
    add_text(slide, 7.15, Y0 + 0.82, 5.5, 0.48,
             ["本装置 f≈889 px、B=150 mm → 1 px 视差误差 = 7.5 mm 深度误差 @ 3.7 m",
              "“4 m 处 30 mm” 与 “1 m 处 10 mm” 不是一回事，比较必须折算到同一距离"],
             size=9.5, color=INK, align=PP_ALIGN.CENTER, line_spacing=1.0, space_after=1)

    # ---- 下左：合成真值自测（5 个回归测试）----
    yt = Y0 + 1.50
    add_rich(slide, ML, yt, 6.6, H_TITLE,
             [("判据：合成真值自测（5 个回归测试，改一行代码即可复跑）",
               {"bold": True, "size": 13.5, "color": BLUE})])
    rows = [
        ("单目全链路", "检测/跟踪/预测", "检出率 100%，半径 0.06 px"),
        ("双目标定", "K, D, R, T", "极线 0.067 px，基线 0.006%"),
        ("同步+配对+三角测量", "几何链路", "配对 100%，三维 1.31 mm"),
        ("成对取帧", "同步偏移正负号", "0/±8/±37 帧全部找回"),
        ("双目整条链路", "真渲染+两路真检测", "配对 100%，三维 RMSE 15.6 mm"),
    ]
    rt = add_table(slide, ML, yt + 0.36, 6.6, len(rows) + 1, 3, row_h=0.245,
                   col_w=[1.75, 1.85, 3.0])
    for c, h in enumerate(("回归测试", "验证对象", "实测结果（合成）")):
        set_cell(rt, 0, c, h)
    for i, (a, b, c) in enumerate(rows, start=1):
        set_cell(rt, i, 0, a, bold=True)
        set_cell(rt, i, 1, b)
        set_cell(rt, i, 2, c, color=GREEN)
    style_table(rt, row_h=0.245, size=9.5, header_size=10)

    # ---- 下右：真机验证方案 ----
    add_rich(slide, 7.15, yt, 5.5, H_TITLE,
             [("真机验证方案", {"bold": True, "size": 13.5, "color": RED})])
    add_box(slide, 7.15, yt + 0.36, 5.5, 1.50, fill=RED_L, line=None,
            shape=MSO_SHAPE.RECTANGLE)
    add_rich(slide, 7.28, yt + 0.44, 5.25, H_TEXT,
             [("诚实说明：以上全部是合成真值，真机标定尚未执行。",
               {"size": 10.5, "bold": True, "color": RED})])
    add_bullets(slide, 7.28, yt + 0.72, 5.25, 1.10, [
        ("用标定板的已知方格边长当长度基准：把板放在量好的 1.0 / 2.0 m 处，"
         "比较重建距离与实际距离，直接给出毫米级绝对误差", {}),
        ("用铁球实测直径作第二基准（兼作单目对照 Z=fR/r 的真值输入）", {}),
        ("判据：极线误差 ≤ 0.3 px、1 m 处三维误差 ≤ 10 mm", {"bold": True}),
    ], size=10, gap=1, line_spacing=1.0)

    # ---- 底部：误差源与对策 ----
    ye = Y1 - 1.46
    add_rich(slide, ML, ye, 6.6, H_TITLE,
             [("误差源与对策（按量级排序）", {"bold": True, "size": 13.5, "color": BLUE})])
    es = add_table(slide, ML, ye + 0.36, 6.6, 4, 3, row_h=0.245,
                   col_w=[1.5, 2.4, 2.7])
    for c, h in enumerate(("误差源", "量级（实测）", "对策")):
        set_cell(es, 0, c, h)
    for i, (a, b, c) in enumerate([
        ("两路同步误差", "滞后 1 帧 → 误差涨 17–34 倍", "拍手/闪光求偏移 + 按偏移取帧"),
        ("2D 检测误差", "0.5–0.9 px", "亚像素边缘细化"),
        ("标定误差", "基线 0.006%（合成）", "多姿态多图 + 拍摄前复检"),
    ], start=1):
        set_cell(es, i, 0, a, bold=True, color=RED if i == 1 else INK)
        set_cell(es, i, 1, b, color=RED if i == 1 else INK)
        set_cell(es, i, 2, c)
    style_table(es, row_h=0.245, size=9.5, header_size=10)

    # 右下角：验收口径
    add_box(slide, 7.15, ye, 5.5, 1.10, fill=GREEN_L, line=None,
            shape=MSO_SHAPE.RECTANGLE)
    add_rich(slide, 7.28, ye + 0.10, 5.25, H_TEXT,
             [("验收口径：1 m 处三维误差 ≤ 10 mm", {"size": 11.5, "bold": True,
                                              "color": GREEN})])
    add_text(slide, 7.28, ye + 0.40, 5.25, 0.62,
             ["实测折算到 1 m 处 = 1.13 mm，比验收线宽约 9 倍 ——",
              "但这都是合成真值，真机尚未验证。"],
             size=10, color=INK, line_spacing=1.08, space_after=1)


# --------------------------------------------------------------------------
BUILDERS = {11: build_slide11, 12: build_slide12, 13: build_slide13, 14: build_slide14}


def main() -> int:
    ap = argparse.ArgumentParser(description="生成开题报告研究思路四页")
    ap.add_argument("--src", default=str(SRC_TEMPLATE), help="模板 pptx 路径")
    ap.add_argument("--out", default=str(ROOT / "docs" / "开题报告_v1.pptx"),
                    help="输出 pptx 路径（不要指向模板本身）")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    if not src.exists():
        print(f"[失败] 找不到模板：{src}")
        print("       模板不进版本库，请用 --src 指向你自己的 pptx")
        return 1
    if src.resolve() == out.resolve():
        print("[失败] 拒绝就地覆盖模板，请另指定 --out")
        return 1

    try:
        prs = Presentation(str(src))          # 读模板（被 PowerPoint 打开时仍可读）
    except Exception as exc:                  # noqa: BLE001
        print(f"[失败] 打不开模板：{exc}")
        print("       如果提示被占用，请先在 PowerPoint 里关闭它。")
        return 1

    for no, fn in BUILDERS.items():
        slide = prs.slides[no - 1]
        # 清掉可能残留的正文形状（保留标题与页码占位符）
        for sh in list(slide.shapes):
            if sh.is_placeholder:
                continue
            sh._element.getparent().remove(sh._element)
        fn(slide)
        print(f"  已写入 slide{no}")

    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        prs.save(str(out))
    except PermissionError:
        print(f"[失败] 写不进去：{out}（文件被占用？请关闭后重试）")
        return 1
    print(f"\n完成：{out}")
    print(f"      共 {len(prs.slides._sldIdLst)} 页，只改动了 slide 11–14")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
