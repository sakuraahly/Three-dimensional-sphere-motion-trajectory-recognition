"""一键生成最终开题报告 PPT（从原始 pptx 一次跑完，不叠加中间产物）。

按《PPT字号等.docx》+《参考文献引用操作方法》两份规范：

  字体      中文宋体 / 英文数字 Times New Roman（全篇）
  正文      26 号、行间距 1.5 倍
  页小标题  32 号
  表格      ≥ 20 号
  参考文献  22 号、行距 1 倍、国标 GB/T 7714 书写
  物理量    符号斜体、下标用真下标

因内容量超出单页承载，对"研究思路"与"参考文献""时间规划"按内容拆页（页数不限）。

**关键坑（已踩过两次，务必保留处理）**
  1. `add_slide(layout)` 会把版式的**全部占位符**克隆到新页；只删非占位符形状不够，
     内容占位符（idx=13/14 的 OBJECT）会显示成"单击此处添加文本"和插图图标 → 必须删净。
  2. `prs.slides` 是**即时**从 XML 算出来的，新增页之后索引会变；重排要先用
     rId 建映射、一次性重排，不要反复"移动一页"。

用法::

    python apps/tools/build_opening_report.py --dry-run
    python apps/tools/build_opening_report.py --src "…/开题报告.pptx" --out outputs/开题报告_最终.pptx
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
# 输入 pptx 是个人交付物、不进版本库：默认路径只给项目内的占位，
# 实际请用 --src 指到你自己的文件（见 docs/PPT生成流程.md）。
DEFAULT_SRC = ROOT / "assets" / "templates" / "开题报告.pptx"
DEFAULT_OUT = ROOT / "outputs" / "开题报告_最终.pptx"

SONG, TIMES = "宋体", "Times New Roman"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
EA, LATIN = f"{{{A_NS}}}ea", f"{{{A_NS}}}latin"

INK = RGBColor(0x1F, 0x28, 0x37)
DIM = RGBColor(0x5B, 0x66, 0x77)
BLUE = RGBColor(0x1F, 0x4E, 0x79)
BLUE_L = RGBColor(0xDE, 0xEA, 0xF6)
GREEN = RGBColor(0x1E, 0x7A, 0x46)
GREEN_L = RGBColor(0xE3, 0xF3, 0xE8)
RED = RGBColor(0xC0, 0x39, 0x2B)
RED_L = RGBColor(0xFC, 0xE8, 0xE6)
AMBER = RGBColor(0x9C, 0x66, 0x00)
AMBER_L = RGBColor(0xFF, 0xF7, 0xE6)
AMBER_B = RGBColor(0xE0, 0xA8, 0x30)
GRAY_L = RGBColor(0xF2, 0xF4, 0xF7)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BORDER = RGBColor(0xBF, 0xCF, 0xE0)
GRAY_BAR = RGBColor(0x8A, 0x93, 0xA3)

ML, CW = 0.79, 11.86
BODY, TBL, TITLE, LS = 26, 20, 32, 1.5
REF_PT = 22


# ─────────────────────────── 基础工具 ───────────────────────────
def patch(run, size=None, bold=None, color=None, font=None, sub=False):
    rPr = run._r.get_or_add_rPr()
    for tag, face in ((EA, SONG), (LATIN, font or TIMES)):
        el = rPr.find(tag)
        if el is None:
            el = etree.SubElement(rPr, tag)
        el.set("typeface", face)
    rPr.set("lang", "zh-CN")
    run.font.name = font or TIMES
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color
    if sub:
        rPr.set("baseline", "-25000")


def _tf(sh):
    tf = sh.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Emu(0)
    return tf


def para(tf, text, *, size=BODY, bold=False, color=INK, ls=LS, first=False,
         align=PP_ALIGN.LEFT, space_after=10, font=None, sub=False):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.line_spacing = ls
    p.space_after = Pt(space_after)
    r = p.add_run()
    r.text = text
    patch(r, size=size, bold=bold, color=color, font=font, sub=sub)
    return p


def textbox(slide, x, y, w, h, items, **kw):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = _tf(tb)
    for i, (t, o) in enumerate(items):
        para(tf, t, first=(i == 0), **{**kw, **o})
    return tb


def rich(slide, x, y, w, h, runs, *, align=PP_ALIGN.CENTER, ls=1.0, space_after=0):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = _tf(tb)
    p = tf.paragraphs[0]
    p.alignment = align
    p.line_spacing = ls
    p.space_after = Pt(space_after)
    for text, o in runs:
        r = p.add_run()
        r.text = text
        patch(r, size=o.get("size", BODY), bold=o.get("bold", False),
              color=o.get("color", INK), font=o.get("font"), sub=o.get("sub", False))
    return tb


def bullets(slide, x, y, w, h, items, **kw):
    return textbox(slide, x, y, w, h, [("▪ " + t, o) for t, o in items], **kw)


def heading(slide, x, y, w, text, *, size=BODY, color=BLUE, h=0.56):
    return textbox(slide, x, y, w, h,
                   [(text, {"size": size, "bold": True, "color": color,
                            "ls": 1.0, "space_after": 0})])


def box(slide, x, y, w, h, fill=None, line=None, lw=1.0):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                                Inches(w), Inches(h))
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(lw)
    sh.shadow.inherit = False
    return sh


def line(slide, x1, y1, x2, y2, color=BLUE, w=1.75):
    ln = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1),
                                    Inches(x2), Inches(y2))
    ln.line.color.rgb = color
    ln.line.width = Pt(w)
    return ln


def table(slide, x, y, w, rows, cols, col_w, row_h, data):
    t = slide.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(w),
                               Inches(row_h * rows)).table
    tot = sum(col_w)
    for i, cw in enumerate(col_w):
        t.columns[i].width = Emu(int(Inches(w) * cw / tot))
    for ri, row in enumerate(data):
        t.rows[ri].height = Inches(row_h)
        for ci, val in enumerate(row):
            cell = t.cell(ri, ci)
            cell.text = val
            cell.margin_left = Inches(0.06)
            cell.margin_right = Inches(0.05)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = (BLUE if ri == 0
                                        else (GRAY_L if ri % 2 == 0 else WHITE))
            for p in cell.text_frame.paragraphs:
                p.line_spacing = 1.0
                p.space_after = Pt(0)
                p.alignment = PP_ALIGN.CENTER if (ri > 0 and ci == 0) else PP_ALIGN.LEFT
                for r in p.runs:
                    patch(r, size=TBL, bold=(ri == 0),
                          color=WHITE if ri == 0 else INK)
    return t


def strip_placeholders(slide, keep_title=True):
    """删掉除标题外的所有占位符 —— 治"单击此处添加文本"的根。"""
    n = 0
    for sh in list(slide.shapes):
        try:
            if keep_title and sh.is_placeholder and sh.placeholder_format.idx == 0:
                continue
            if sh.is_placeholder:
                sh._element.getparent().remove(sh._element)
                n += 1
            elif keep_title is False:
                sh._element.getparent().remove(sh._element)
                n += 1
        except Exception:
            pass
    return n


def clear_body(slide):
    """保留标题，清掉其余一切（重建页用）。"""
    for sh in list(slide.shapes):
        try:
            if sh.is_placeholder and sh.placeholder_format.idx == 0:
                continue
        except Exception:
            pass
        sh._element.getparent().remove(sh._element)


def normalize_fonts(prs) -> int:
    n = 0
    for slide in prs.slides:
        for sh in slide.shapes:
            frames = []
            if sh.has_text_frame:
                frames.append(sh.text_frame)
            if sh.has_table:
                frames += [c.text_frame for row in sh.table.rows for c in row.cells]
            for tf in frames:
                for p in tf.paragraphs:
                    for r in p.runs:
                        if r.text.strip():
                            rPr = r._r.get_or_add_rPr()
                            for tag, face in ((EA, SONG), (LATIN, TIMES)):
                                el = rPr.find(tag)
                                if el is None:
                                    el = etree.SubElement(rPr, tag)
                                el.set("typeface", face)
                            r.font.name = TIMES
                            n += 1
    return n


def clone_after(prs, src_slide, title):
    """按版式新建一页，**只留标题占位符**（内容占位符必须删净）。"""
    new = prs.slides.add_slide(src_slide.slide_layout)
    strip_placeholders(new, keep_title=True)
    for sh in new.shapes:
        try:
            if sh.is_placeholder and sh.placeholder_format.idx == 0:
                sh.text_frame.text = title
                for p in sh.text_frame.paragraphs:
                    for r in p.runs:
                        patch(r, size=TITLE)
        except Exception:
            pass
    return new


def reorder(prs, desired):
    lst = prs.slides._sldIdLst
    entries = {e.rId: e for e in list(lst.sldId_lst)}
    rid_of = {id(prs.part.related_slide(e.rId)): e.rId for e in list(lst.sldId_lst)}
    ordered = [entries[rid_of[id(s)]] for s in desired]
    for e in list(lst):
        lst.remove(e)
    for e in ordered:
        lst.append(e)


# ─────────────────────── 研究思路：装置搭建 ───────────────────────
def p17a(s):
    heading(s, ML, 1.42, CW, "硬件清单")
    table(s, ML, 2.02, CW, 8, 3, [2.4, 6.6, 2.0], 0.62, [
        ("部件", "选定方案", "状态"),
        ("相机 ×2", "aoni A25，同型号", "已有"),
        ("支架或固定件", "硬质横梁等，刚性固定", "待搭建"),
        ("基线 B", "待实验标定", "待定"),
        ("标定板", "8×5 内角点 / 25 mm，贴硬纸板", "图已生成"),
        ("测试用球", "实验室单摆实验铁球", "已有"),
        ("同步源", "拍手 / 闪光", "待搭建"),
        ("（可选）云台", "PID 闭环验证", "可选")])


def p17b(s):
    heading(s, ML, 1.42, 6.4, "装置示意图")
    dx, dw = ML, 6.4
    box(s, dx + 0.2, 2.10, dw - 0.4, 0.22, fill=GRAY_BAR)
    box(s, dx + 0.1, 2.36, 1.5, 0.56, fill=BLUE)
    box(s, dx + dw - 1.6, 2.36, 1.5, 0.56, fill=BLUE)
    textbox(s, dx + 0.1, 2.46, 1.5, 0.4, [("左相机 L", {"size": TBL, "color": WHITE,
             "align": PP_ALIGN.CENTER, "ls": 1.0, "space_after": 0})])
    textbox(s, dx + dw - 1.6, 2.46, 1.5, 0.4, [("右相机 R", {"size": TBL,
             "color": WHITE, "align": PP_ALIGN.CENTER, "ls": 1.0, "space_after": 0})])
    textbox(s, dx - 0.3, 3.00, dw + 0.6, 0.44, [("基线 B（待实验标定）",
             {"size": TBL, "color": RED, "align": PP_ALIGN.CENTER, "ls": 1.0,
              "space_after": 0})])
    line(s, dx + 0.80, 3.52, dx + 1.90, 4.74)
    line(s, dx + 0.80, 3.52, dx + dw - 0.80, 4.74)
    line(s, dx + dw - 0.80, 3.52, dx + dw - 1.90, 4.74)
    line(s, dx + dw - 0.80, 3.52, dx + 0.80, 4.74)
    box(s, dx + 1.45, 4.26, dw - 2.90, 0.92, fill=BLUE_L, line=BLUE, lw=1.75)
    textbox(s, dx + 1.45, 4.36, dw - 2.90, 0.76,
            [("两路公共可见区", {"size": TBL, "bold": True, "color": BLUE,
                               "align": PP_ALIGN.CENTER, "ls": 1.1, "space_after": 0}),
             ("球必须始终在区内", {"size": TBL, "color": DIM,
                                "align": PP_ALIGN.CENTER, "ls": 1.1, "space_after": 0})])
    heading(s, 7.60, 1.42, 5.05, "待办清单")
    bullets(s, 7.60, 2.20, 5.05, 3.4, [
        ("两台相机刚性固定、对准同一区域", {"size": TBL}),
        ("实测确定基线 B", {"size": TBL}),
        ("游标卡尺实测铁球直径并记录", {"size": TBL})], size=TBL, space_after=16)


def p17c(s):
    heading(s, ML, 1.42, CW, "两条硬约束", color=RED)
    textbox(s, ML, 2.16, CW, 4.2, [
        ("一、两台相机必须刚性固定。", {"bold": True}),
        ("整体移动没关系（R、T 是常数）；但两台各自移动则相对位姿每帧都在变，"
         "三角测量在原理上失效。", {}),
        ("二、同型号相机的索引会在插拔/重启后互换。", {"bold": True}),
        ("此时画面完全正常，但标定的 R、T 已失效。对策：贴 L/R 标签，"
         "并在每次拍摄前运行复检程序比对极线误差。", {})])


def p18(s):
    heading(s, ML, 1.30, CW, "三层软件结构")
    bands = [("装置层", "两台相机各自后台取最新帧，再按同步偏移成对取帧", BLUE, BLUE_L),
             ("算法层", "sphere_tracker 单目感知库（两路各调一次）＋ calib 双目标定"
                        "＋ stereo 同步·配对·三角测量",
              RGBColor(0x2E, 0x6D, 0xA4), RGBColor(0xE8, 0xF1, 0xF9)),
             ("应用层", "标定程序 │ 复检程序 │ 实时程序 │ 离线批处理 │ 回归测试",
              RGBColor(0x3F, 0x7C, 0xB0), GRAY_L)]
    y = 1.86
    for name, desc, bar, bg in bands:
        box(s, ML, y, CW, 1.26, fill=bg, line=BORDER)
        box(s, ML, y, 1.72, 1.26, fill=bar)
        textbox(s, ML, y + 0.40, 1.72, 0.5, [(name, {"bold": True, "color": WHITE,
                 "align": PP_ALIGN.CENTER, "ls": 1.0, "space_after": 0})])
        textbox(s, ML + 1.94, y + 0.06, CW - 2.10, 1.14,
                [(desc, {"ls": LS, "space_after": 0})])
        y += 1.34
    # 底部只留一句结论，避免 26 号/1.5 行距下贴到页面底边
    heading(s, ML, y + 0.10, CW, "一套代码、两路复用（双目部分只负责合成三维）")


def p19a(s):
    heading(s, ML, 1.42, CW, "数据采集流程")
    steps = ["① 标定数据", "② 时间同步", "③ 运动录制", "④ 对齐取帧", "⑤ 存档"]
    n, gap = len(steps), 0.20
    bw = (CW - gap * (n - 1)) / n
    for i, t in enumerate(steps):
        x = ML + i * (bw + gap)
        box(s, x, 2.02, bw, 0.72, fill=BLUE_L, line=BLUE)
        textbox(s, x, 2.16, bw, 0.5, [(t, {"size": TBL, "bold": True, "color": BLUE,
                 "align": PP_ALIGN.CENTER, "ls": 1.0, "space_after": 0})])
    heading(s, ML, 3.00, CW, "标定数据的采集")
    bullets(s, ML, 3.66, CW, 2.9, [
        ("两台相机同时拍棋盘格，15–30 组位姿，覆盖视场各区域并含倾斜姿态", {}),
        ("程序自动挑帧（七道门限），剔除重投影误差偏大的图", {}),
        ("标定不需要时间同步：只收板停稳的帧", {})], space_after=16)


def p19b(s):
    heading(s, ML, 1.42, CW, "运动数据的采集")
    table(s, ML, 2.02, CW, 6, 2, [3.6, 8.26], 0.60, [
        ("参数", "取值"),
        ("分辨率 / 帧率", "1920×1080 @ 30 fps（相机实测上限）"),
        ("曝光 / 白平衡 / 焦距", "全部固定，否则颜色线索漂移"),
        ("相机距目标", "1.5–2.5 m，球成像半径 20–45 px"),
        ("运动类型", "自由落体 / 斜抛 / 落地弹跳 / 斜面滚动"),
        ("重复次数", "每种运动 × 光照 × 球体，5–10 次")])
    heading(s, ML, 4.90, CW, "时间同步的做法", color=RED)
    bullets(s, ML, 5.48, CW, 2.0, [
        ("开录前拍一下手或闪一下灯，分别找出两路该事件的帧号", {}),
        ("例：左 143 帧、右 151 帧 → 偏移 +8 帧，程序按偏移自动配对取帧", {})],
        space_after=10)


def p19c(s):
    heading(s, ML, 1.42, CW, "存档格式")
    table(s, ML, 2.02, CW, 5, 2, [3.0, 8.86], 0.62, [
        ("类型", "格式"),
        ("标定参数", ".npz（K1, D1, K2, D2, R, T, P1, P2, Q）+ .json 质量报告"),
        ("轨迹", ".csv（frame, t, x1y1, x2y2, XYZ, 平滑后坐标, 预测值）"),
        ("视频", "左右各一份 .mp4"),
        ("图表", ".png / Plotly .html")])
    heading(s, ML, 4.62, CW, "记录表必须登记两项", color=RED)
    bullets(s, ML, 5.20, CW, 1.9, [("本次同步偏移（帧）", {}),
                                   ("本次标定复检值（与标定当时对比）", {})],
            space_after=14)


def p20a(s):
    heading(s, ML, 1.36, CW, "核心误差公式")
    box(s, ML, 2.00, CW, 1.66, fill=AMBER_L, line=AMBER_B, lw=1.5)
    rich(s, ML, 2.28, CW, 1.1, [
        ("ε", {"size": 40, "bold": True, "color": AMBER, "font": TIMES}),
        ("Z", {"size": 40, "bold": True, "color": AMBER, "font": TIMES, "sub": True}),
        ("  ≈  Z² · δd / (f · B)",
         {"size": 40, "bold": True, "color": AMBER, "font": TIMES})])
    textbox(s, ML, 3.96, CW, 2.4, [
        ("ε_Z 为深度误差，δd 为视差误差，f 为焦距，B 为基线。", {}),
        ("深度误差随距离【平方】增长，反比于焦距与基线。", {"bold": True}),
        ("因此比较精度必须折算到同一距离。", {})])


def p20b(s):
    heading(s, ML, 1.42, CW, "误差源与对策")
    table(s, ML, 2.02, CW, 5, 2, [4.2, 7.66], 0.70, [
        ("误差源", "对策"),
        ("两路同步误差（首要）", "拍手 / 闪光求帧偏移，按偏移自动配对取帧"),
        ("检测误差", "亚像素边缘细化；多线索投票降级"),
        ("标定误差", "多姿态多图 + 拍摄前复检"),
        ("运动模糊 / 视频压缩", "提高快门、增加照明、固定曝光")])


def p20c(s):
    heading(s, ML, 1.42, CW, "真机验证方案")
    bullets(s, ML, 2.10, CW, 4.4, [
        ("用标定板的已知方格边长当长度基准：把板放在量好的 1.0 / 2.0 m 处，"
         "比较重建距离与实际距离", {}),
        ("用铁球实测直径作第二基准，兼作单目对照 Z = fR / r 的真值输入", {}),
        ("判据：极线误差 ≤ 0.3 px，1 m 处三维误差 ≤ 10 mm", {"bold": True})],
        space_after=18)


# ─────────────────────── 时间规划 ───────────────────────
def p_tl_a(s):
    heading(s, ML, 1.42, CW, "阶段计划（10 月 – 11 月；按 12 月 30 日截止倒排）")
    table(s, ML, 2.02, CW, 7, 3, [3.3, 6.0, 2.56], 0.62, [
        ("时间", "任务", "产出"),
        ("10 月上旬", "装置搭建：相机刚性固定、标定板打印与实测方格", "装置就位"),
        ("10 月中旬", "真机双目标定与复检；录制带同步信号的双路视频", "参数 + 视频"),
        ("10 月下旬", "两路检测与同步联调，打通三角测量，产出三维轨迹", "三维数据"),
        ("11 月上旬", "三维平滑（S-G / 卡尔曼）与轨迹预测", "平滑与预测"),
        ("11 月下旬", "三维可视化；精度评估；对比与误差实验", "图表与数据"),
        ("12 月上旬", "补齐实验、整理数据与图表", "结果定型")])


def p_tl_b(s):
    heading(s, ML, 1.42, CW, "阶段计划（12 月；硬截止 12 月 30 日）")
    table(s, ML, 2.02, CW, 3, 3, [3.3, 6.0, 2.56], 0.62, [
        ("时间", "任务", "产出"),
        ("12 月中旬", "撰写期末大论文；按意见修改，形成缩改论文", "论文初稿"),
        ("12 月下旬", "论文定稿与查重；答辩 PPT 与预答辩演练", "全部交付")])
    # 注也按"全文不小于 20 号"，并压紧到底边之上
    textbox(s, ML, 5.85, CW, 1.4,
            [("注：本页为参考性进度安排，仅供参考；实际节点随实验进度与教师要求调整。",
              {"size": TBL, "color": DIM, "ls": LS, "space_after": 0})])


# ─────────────────────── 参考文献 ───────────────────────
REFS_A = [
    "1. Bradski G. The OpenCV Library[J]. Dr. Dobb's Journal of Software Tools, 2000.",
    "2. Zhang Z. A Flexible New Technique for Camera Calibration[J]. IEEE Transactions on "
    "Pattern Analysis and Machine Intelligence, 2000, 22(11): 1330-1334.",
    "3. Hartley R, Zisserman A. Multiple View Geometry in Computer Vision[M]. 2nd ed. "
    "Cambridge: Cambridge University Press, 2004.",
    "4. Kalman R E. A New Approach to Linear Filtering and Prediction Problems[J]. "
    "Journal of Basic Engineering, 1960, 82(1): 35-45.",
    "5. Welch G, Bishop G. An Introduction to the Kalman Filter[R]. Chapel Hill: "
    "Department of Computer Science, University of North Carolina at Chapel Hill, 2006.",
    "6. Hirschmüller H. Stereo Processing by Semiglobal Matching and Mutual Information[J]. "
    "IEEE Transactions on Pattern Analysis and Machine Intelligence, 2008, 30(2): 328-341.",
]
REFS_B = [
    "7. Savitzky A, Golay M J E. Smoothing and Differentiation of Data by Simplified Least "
    "Squares Procedures[J]. Analytical Chemistry, 1964, 36(8): 1627-1639.",
    "8. Liu Y, Liu L. Accurate Real-time Ball Trajectory Estimation with Onboard Stereo "
    "Camera System for Humanoid Ping-pong Robot[J]. Robotics and Autonomous Systems, 2018.",
    "9. Wu Y, et al. Multi-camera 3D Ball Tracking Framework for Sports Video[J]. "
    "IET Image Processing, 2020, 14(15).",
    "10. Van Zandycke G, et al. 3D Ball Localization From A Single Calibrated Image[C]//"
    "Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition "
    "Workshops (CVSports), 2022. arXiv:2204.00003.",
]
REFS_C = [
    "11. Ponglertnapakorn P, Suwajanakorn S. Where Is The Ball: 3D Ball Trajectory "
    "Estimation From 2D Monocular Tracking[C]//Proceedings of the IEEE/CVF Conference on "
    "Computer Vision and Pattern Recognition Workshops (CVSports), 2025. arXiv:2506.05763.",
]
REF_NOTE = "注：文献清单仅供参考；格式按国标 GB/T 7714，编号需与正文引用对应。"


def p_ref(s, items, with_note=True):
    textbox(s, ML, 1.52, CW, 4.7,
            [(t, {"size": REF_PT, "ls": 1.0, "space_after": 0}) for t in items])
    if with_note:
        # 注同样按"不小于 20 号"
        textbox(s, ML, 6.40, CW, 0.9,
                [(REF_NOTE, {"size": TBL, "color": DIM, "ls": LS, "space_after": 0})])


# ─────────────────────────── 主流程 ───────────────────────────
def find_slide(prs, title: str):
    """按标题占位符文本找页（找不到返回 None）——比硬编码索引稳。"""
    for s in prs.slides:
        for sh in s.shapes:
            try:
                if sh.is_placeholder and sh.placeholder_format.idx == 0:
                    if sh.text_frame.text.strip() == title:
                        return s
            except Exception:
                pass
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    if not src.exists():
        print(f"[失败] 找不到输入：{src}")
        print("       输入 pptx 不进版本库，请用 --src 指向你自己的文件")
        return 1
    prs = Presentation(str(src))
    original = list(prs.slides)          # ← 快照，新增页后索引会变
    log = [f"读入 {src.name}：{len(original)} 页"]

    # 定位关键页（按标题，不靠索引）
    anchors = {}
    for t in ("装置搭建", "软件控制", "数据采集", "测量精度", "时间规划", "参考文献"):
        s = find_slide(prs, t)
        if s is None:
            print(f"[失败] 找不到标题为「{t}」的页")
            return 1
        anchors[t] = s
    log.append("已按标题定位：" + "、".join(anchors))

    # 1) 研究思路：装置搭建 3 页 / 软件控制 1 页 / 数据采集 3 页 / 测量精度 3 页
    new_pages: dict[int, list] = {}

    def rebuild(key, builders):
        base = anchors[key]
        idx = original.index(base)
        clear_body(base)
        builders[0](base)
        made = []
        for fn in builders[1:]:
            pg = clone_after(prs, base, key)
            fn(pg)
            made.append(pg)
        new_pages[idx] = made
        log.append(f"{key}：1 页 -> {len(builders)} 页")

    rebuild("装置搭建", [p17a, p17b, p17c])
    rebuild("软件控制", [p18])
    rebuild("数据采集", [p19a, p19b, p19c])
    rebuild("测量精度", [p20a, p20b, p20c])

    # 2) 时间规划 2 页
    tl = anchors["时间规划"]
    i_tl = original.index(tl)
    clear_body(tl)
    p_tl_a(tl)
    tl2 = clone_after(prs, tl, "时间规划")
    p_tl_b(tl2)
    new_pages[i_tl] = [tl2]
    log.append("时间规划：1 页 -> 2 页")

    # 3) 参考文献 3 页
    rf = anchors["参考文献"]
    i_rf = original.index(rf)
    clear_body(rf)
    p_ref(rf, REFS_A)
    rf2 = clone_after(prs, rf, "参考文献")
    p_ref(rf2, REFS_B)
    rf3 = clone_after(prs, rf, "参考文献")
    p_ref(rf3, REFS_C, with_note=False)
    new_pages[i_rf] = [rf2, rf3]
    log.append("参考文献：1 页 -> 3 页")

    # 4) 一次性重排（rId 映射，不用反复移动）
    desired = []
    for i, s in enumerate(original):
        desired.append(s)
        desired.extend(new_pages.get(i, []))
    reorder(prs, desired)
    log.append(f"重排完成，共 {len(desired)} 页")

    # 5) 全篇字体 + 兜底清理占位符
    nf = normalize_fonts(prs)
    log.append(f"全篇字体：{nf} 处设为 宋体 / Times New Roman")
    left = 0
    for s in prs.slides:
        for sh in list(s.shapes):
            try:
                if (sh.is_placeholder
                        and str(sh.placeholder_format.type).startswith(("OBJECT", "BODY"))
                        and sh.has_text_frame and not sh.text_frame.text.strip()):
                    sh._element.getparent().remove(sh._element)
                    left += 1
            except Exception:
                pass
    log.append(f"兜底清掉 {left} 个空白内容占位符（防「单击此处添加文本」）")

    print("即将执行：")
    for x in log:
        print(f"  · {x}")
    if args.dry_run:
        print("\n（--dry-run，未写出）")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        prs.save(str(out))
    except PermissionError:
        print(f"\n[失败] 写不进去：{out}（文件被占用？请关闭 PowerPoint / WPS）")
        return 1
    print(f"\n已写入 {out}\n页数 {len(prs.slides._sldIdLst)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
