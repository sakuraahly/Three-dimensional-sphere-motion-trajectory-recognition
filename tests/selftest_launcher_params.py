"""启动器标定参数面板自测：读取、校验、以及"默认就是正式档"。

为什么需要它
------------
用户要求的"标定要收足够多对 + UI 里能设标定参数"落地时，暴露了一个**真缺陷**：

`apps/gui_main.py` 原来把板参数**硬编码**在命令里（`--cols 8 --rows 5 --square 25`），
而且**完全没有 `--target`** —— 于是从界面走标定，**永远用脚本默认的那一档**。
用户以为在做正式标定，实际收到的是试标定档的结果，而**界面上没有任何迹象**。

这与 `AGENTS.md` §6"代码与规范冲突 = 缺陷"是同一类：
**界面上能设的东西，不许是一份与真实取值无关的装饰。**

本测试钉四件事：

1. **默认值就是正式档**（`--target 40 --min-pairs 20`）；
2. **可选项留空 = 不传该参数**（不留空才传，尤其 `--baseline-mm` 是基线护栏的开关）；
3. **非法输入与参数矛盾必须被拦住**，并且给出可读的原因；
4. 低目标档仍然可用（只是会在标定程序里打印显式警告）。

反向自证（改了就会红）
--------------------
======================================  ==========================================  ========
改动                                    注入方式                                    结果
======================================  ==========================================  ========
默认值退回 8/8                          面板默认值改回 "8" / "8"                     exit=1
留空也传参数（变成空串）                去掉 `if x is not None:` 判断                exit=1
不校验就拼命令                          去掉 `_read_int` 的范围判断                  exit=1
不查"最少>目标"                         去掉 `minp > target` 那段                    exit=1
高级项接了不传                          去掉 `args += ["--baseline-mm", ...]`        exit=1
**卡片**说明不换行                      去掉 `wraplength=CARD_HINT_WRAP`             exit=1
日志挂回窗口（不进滚动页）              改 `LabelFrame(self._page, ...)` 为 `(self,)`  exit=1
日志无条件滚到底                        去掉 `if at_bottom:` 判断                    exit=1
滚动条永不显示                          把 `_vbar.pack(...)` 换成 `pass`             exit=1
======================================  ==========================================  ========

**本测试没覆盖的（如实写明）**

- **参数说明的换行**（`PARAM_HINT_WRAP`）：**不会红**。实测过 —— 当前真正卡住
  宽度的是**卡片**说明那条换行（把它设成 820/700/640，页宽跟着 873/756/704 变；
  到 640 之后不再变，约束才转移到参数说明）。所以参数说明的换行现在是**护栏**，
  不是当前约束。它的作用是"将来某条参数说明特别长时别把卡片撑开"，
  而那件事还没发生，测试自然抓不住。**不假装覆盖了。**
- **滚轮的可用性**：绑的是 `bind_all("<MouseWheel>")`，无头环境发不了滚轮事件。
  只断言了"滚动条在需要时出现"与 `scroll_to_bottom()` 生效。
- **日志卡片的 `expand=True`**：把 `fill="x"` 改回 `fill="both", expand=True`
  **不会红**。原因是页面高度本来就由内容决定，`expand` 在多出来的空间里
  没有可测的效果 —— 它只影响"看起来占多高"，而那个量没有稳的判据。
  已经把日志挂进滚动页（这一条**有**护栏），所以不假装覆盖这个次要项。

用法::

    python tests/selftest_launcher_params.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "apps").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import tkinter as tk
except ImportError:                                        # pragma: no cover
    tk = None

LAUNCHER = ROOT / "apps/gui_main.py"


def _load_launcher():
    """把启动器作为模块加载（不执行 mainloop）。

    ⚠️ **必须先注册进 `sys.modules`**：启动器里有 `@dataclass`，
    而 dataclasses 解析注解时要按 `cls.__module__` 去 `sys.modules` 里找模块，
    找不到就抛 `AttributeError: 'NoneType' object has no attribute '__dict__'`。
    直接 `exec_module` 而不注册，就是这个错。
    """
    sys.argv = ["apps/gui_main.py"]
    spec = importlib.util.spec_from_file_location("launcher_params", LAUNCHER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["launcher_params"] = mod          # ← 关键这一行
    spec.loader.exec_module(mod)
    return mod


class _Box:
    """替掉 messagebox：记录弹窗内容，避免模态框把测试卡住。"""

    def __init__(self) -> None:
        self.msgs: list[str] = []

    def showwarning(self, title, msg, **kw) -> None:        # noqa: ANN001
        self.msgs.append(msg)

    def showinfo(self, title, msg, **kw) -> None:           # noqa: ANN001
        self.msgs.append(msg)


def _set(entry, val: str) -> None:
    entry.delete(0, "end")
    if val != "":
        entry.insert(0, val)


def _check_view_3d_switch() -> list[tuple[str, bool]]:
    """8d. 3D 查看器能换任意会话（2026-10-28）。

    用户原话："3D 查看器打开后自动选最近会话，且不能改"。

    换源本身看着简单（再 `load_view` 一次），但有两个**不报错**的坑：
    ① `_left_raw/_right_raw` 是上一个会话的解码帧，新会话若没有视频，
    `_update_video` 会把它俩原样画出来 —— 标题换了、画面还是上一条；
    ② 坏源若"先改状态再加载"，会把正在看的会话弄没。

    全部用**临时会话**：判据不能依赖"这台机器录过什么"（B-27 的教训：
    干净 checkout 上会假红）。
    """
    import cv2
    import numpy as np
    import tkinter as tk

    from apps.view_3d import (VIDEO_SOURCE_LABELS, VIDEO_SOURCE_ORDER,
                              Viewer3D, load_view, scan_sources)

    def mk(base: Path, name: str, *, n: int = 3, video: bool = True) -> Path:
        sess = base / name
        (sess / "derived").mkdir(parents=True, exist_ok=True)
        pts = [{"xyz": [0.01 * i, 0.0, 1.0 + 0.001 * i], "frame": i, "t": 0.01 * i}
               for i in range(n)]
        (sess / "derived" / "visualization.json").write_text(json.dumps({
            "schema": "sphere-stereo-visualization", "version": 1,
            "session": name, "trajectory": pts,
            "coordinate_system": "left_camera", "unit": "m",
            "stereo_mode": "parallel"}, ensure_ascii=False), encoding="utf-8")
        if video:
            # 文件名必须是 annotated.mp4（`_resolve_video` 只认它）；
            # 项目"别用 mp4v"那条针对录制/叠加，这里的合成帧没有标记。
            dst = sess / "derived" / "annotated.mp4"
            vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"),
                                 5.0, (128, 64))
            if not vw.isOpened():
                raise RuntimeError(f"视频写不了：{dst}")
            for i in range(6):
                vw.write(np.full((64, 128, 3), 40 + i * 25, np.uint8))
            vw.release()
        return sess

    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        old = mk(base, "20260101-000000", video=False)
        new = mk(base, "20260202-000000", video=False)
        # 故意反着设：名字旧的 mtime 是"刚刚"、名字新的是一天前。
        # 按 mtime 排会得到 [旧, 新]，正确顺序是 [新, 旧]。
        now = time.time()
        os.utime(old / "derived" / "visualization.json", (now, now))
        os.utime(new / "derived" / "visualization.json",
                 (now - 86400, now - 86400))
        got = [f.parent.parent.name for f in scan_sources(roots=(base,))]
        checks.append((f"会话列表按目录名倒序、不被 mtime 带跑（{got}）",
                       got == ["20260202-000000", "20260101-000000"]))

    # 分区布局 `data/stereo/parallel/<会话>` 必须也能扫到。
    # 2026-11-05 的现场 bug：`scan_sources` 自己 `glob("*/derived/*")` 只扫一层，
    # 而会话按装置分到 parallel/angled 之后多了一层 —— 新布局的会话**一个都
    # 扫不到**，表现为"数据源里没有最新那次实验，默认会话落到更老的目录"。
    # 上面那个用例造的全是扁平布局，所以一直没抓到。
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        part = base / "parallel" / "20260303-000000"
        (part / "derived").mkdir(parents=True)
        (part / "derived" / "visualization.json").write_text(
            json.dumps({"schema": "sphere-stereo-visualization",
                        "version": 1, "trajectory": []}), encoding="utf-8")
        mk(base, "20260101-000000", video=False)
        got2 = [f.parent.parent.name for f in scan_sources(roots=(base,))]
        checks.append((f"分区布局（parallel/会话）也能扫到（{got2}）",
                       "20260303-000000" in got2 and "20260101-000000" in got2))

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        a = mk(base, "20260101-000000", n=5, video=True)
        b = mk(base, "20260202-000000", n=9, video=True)
        root = tk.Tk()
        root.withdraw()
        v = None
        try:
            v = Viewer3D(root, load_view(a / "derived" / "visualization.json"))
            n0, title0 = v.n, root.title()
            v.idx = 3
            v.open_source(b / "derived" / "visualization.json")
            checks += [
                ("换会话后点数换成新的（5 → 9）", v.n == 9 and n0 == 5),
                ("游标归零（不沿用上一个会话的帧号）", v.idx == 0),
                ("标题与滑杆上限都跟着换",
                 root.title() != title0 and "20260202-000000" in root.title()
                 and int(float(v.slider.cget("to"))) == 8),
            ]
            # 「平滑」开关（2026-11-03 加）：默认必须**开**（用户要的是去掉毛刺），
            # 且取消勾选后主图与小图**同时**退回原始点 —— 两条路径共用一个
            # `_smooth_active()`，这里量的是那个出口，不是按钮外观。
            checks.append(("「平滑」勾选框默认是开，文字是「平滑」",
                           bool(v.use_smooth.get())
                           and v.smooth_chk.cget("text") == "平滑"
                           and v._smooth_active() is True))
            v.use_smooth.set(False)
            v._on_toggle_smooth()
            checks.append(("取消勾选后不再用平滑点（主图/小图同一条判据）",
                           v._smooth_active() is False))
            v.use_smooth.set(True)
            v._on_toggle_smooth()

            # ---- 8d-2. 视角：CAD 方向预设 / 任意滚转 / 重画不弹回（2026-11-04）----
            # 用户报"旋转自由度不够、要像 CAD 那样选方向"。这里钉三件事：
            # ① 预设真的改到 ax（不是只改下拉文字）；
            # ② 滚转（第三个自由度）生效；
            # ③ **换帧重画不会把视角打回默认** —— 老实现每帧 view_init(22,-62)，
            #    播放时根本转不动，这才是"自由度不够"的真因。
            import numpy as _np
            v.view_box.set("俯视（从头顶）")
            v._on_view_preset()
            checks.append(("选「俯视」后 ax 的仰角/方位角真的变了",
                           abs(v.ax.elev - 0.0) < 1e-6
                           and abs(v.ax.azim + 90.0) < 1e-6))
            v.view_box.set("前视（相机看到的）")
            v._on_view_preset()
            checks.append(("「前视」是相机视角（elev=-90，屏幕 X 右 / Y 下）",
                           abs(v.ax.elev + 90.0) < 1e-6))
            v.view.update(elev=12.0, azim=34.0, roll=56.0)
            v._redraw()
            checks.append(("换帧重画后视角不被弹回（旋转留得住）",
                           abs(v.ax.elev - 12.0) < 1e-6
                           and abs(v.ax.azim - 34.0) < 1e-6
                           and abs(getattr(v.ax, "roll", 0.0) - 56.0) < 1e-6))
            v._set_roll(-30.0)
            checks.append(("滚转滑块生效（roll 是第三个自由度）",
                           abs(v.ax.roll + 30.0) < 1e-6
                           and abs(v.roll_scale.get() + 30.0) < 1e-6))
            n_base = len(v.ax.collections) + len(v.ax.lines)
            v.show_planes["XY"].set(True)
            v._on_plane_toggle()
            n_xy = len(v.ax.collections) + len(v.ax.lines)
            checks.append(("勾选 XY 平面后多出图元（面 + 网格线）", n_xy > n_base))
            v.show_planes["XZ"].set(True)
            v.show_planes["YZ"].set(True)
            v._on_plane_toggle()
            checks.append(("三个平面能同时开",
                           len(v.ax.collections) + len(v.ax.lines) > n_xy))
            for _name in v.show_planes:
                v.show_planes[_name].set(False)
            v._on_plane_toggle()
            checks.append(("全部取消后回到原来的图元数",
                           len(v.ax.collections) + len(v.ax.lines) == n_base))
            p_proj = v.ax.get_proj().copy()
            v.ortho.set(False)
            v._on_proj_toggle()
            checks.append(("正交 / 透视可切换（投影矩阵确实不同）",
                           not _np.allclose(p_proj, v.ax.get_proj())))
            v.ortho.set(True)

            # ---- 8d-3. 缩放/平移 + 可拖动四宫格（2026-11-04）----
            # 用户："放大缩小只有线动、坐标轴没自动适配，不能指定放大区域，
            # 不能前后左右移动（现在移动的是坐标轴）。"
            # 钉三件事：① 范围是**持久状态**——原生 pan/zoom 改了 ax 的 lim 后，
            # `_redraw`（播放下一帧）必须保留，不能弹回；② 绕点缩放真的以给定
            # 点为锚；③ 原生三键（左旋转/中平移/右缩放）映射在位。
            import numpy as _np2
            from tkinter import ttk as _ttk
            v.ax.set_xlim(-3, 3); v.ax.set_ylim(-3, 3); v.ax.set_zlim(-3, 3)
            v._redraw()
            lo, hi = v._current_lims()
            checks.append(("原生平移后 _redraw 保留范围（不弹回）",
                           [round(x, 1) for x in lo] == [-3.0, -3.0, -3.0]
                           and [round(x, 1) for x in hi] == [3.0, 3.0, 3.0]))
            el_before, az_before = v.ax.elev, v.ax.azim
            v._zoom_around(0.5, 0.0, 0.0, 0.0)
            lo_z, hi_z = v._current_lims()
            checks.append(("绕点缩放以给定点为锚（绕原点缩 0.5 → [-1.5,1.5]）",
                           abs(lo_z[0] + 1.5) < 1e-6 and abs(hi_z[0] - 1.5) < 1e-6
                           and abs(v.ax.elev - el_before) < 1e-6
                           and abs(v.ax.azim - az_before) < 1e-6))
            # 右键 = 平移（2026-11-05 用户："假设观察者是一个人，右键不是转动
            # 视角，而是人前后左右的走"）。但**不能交给原生 `_pan_btn`** ——
            # `drag_pan` 无边界，拖两下轨迹就跑出画布。平移改由
            # `_on_press/_on_motion` 接管（带 `_clamp_pan`），所以这里断言
            # 原生 pan 列表为空、缩放也不抢键位。
            checks.append(("三键映射：左键旋转、平移自己接管、右键不缩放",
                           v.ax._rotate_btn == [1] and v.ax._pan_btn == []
                           and v.ax._zoom_btn == []))

            # 主图铺满画布（2026-11-05 用户："全屏不是让窗口全屏，而是让坐标轴
            # 应用于全局画面，你自己看是不是很多留白"）。留白有两层来源：
            # ① subplots_adjust 的边距（3D 标签在 axes 内，边距纯浪费）；
            # ② Axes3D.apply_aspect 里 `box_aspect=1` 写死，无条件把容器压成
            #    正方形 —— 16:10 画布里 3D 容器就只剩 65% 宽。
            # 断言默认 position 宽度接近 1，就等于同时守住这两条不被改回去。
            _b = v.ax.get_position().bounds
            checks.append(("主图铺满画布（宽度≈100%，不被压成正方形）",
                           _b[2] > 0.95))
            # 「保持立方体比例」勾上要退回**物理正方形**（左右留白，但不变形）
            v.keep_cube.set(True)
            v._apply_layout()
            _b2 = v.ax.get_position().bounds
            _fw, _fh = v.fig.get_size_inches()
            checks.append(("「保持立方体比例」-> 容器退回物理正方形",
                           abs(_b2[2] * _fw - _b2[3] * _fh) < 1e-6
                           and _b2[2] < 0.95))
            v.keep_cube.set(False)
            v._apply_layout()
            checks.append(("切回默认 -> 重新铺满",
                           v.ax.get_position().bounds[2] > 0.95))

            # 真实采样点 + 点击查坐标（2026-11-05 用户："在曲线上增加我们的
            # 拍摄的真实采样点，点击这些点就可以看到我们的三维坐标值"）
            from apps.view_3d import PAN_MARGIN, PICK_RADIUS_PX, TICK_GRID_CELLS

            # 先重画一次让缓存与当前范围一致（上面 `_zoom_around` 改过 lims），
            # 再拍快照 —— `_pick_at` 内部会 `_redraw` 并按新范围重算 `_pt_px`，
            # 拿重算后的坐标去比旧点击位置会假失败。
            v._redraw()
            v.canvas.draw()
            checks.append(("采样点投影缓存 = 真实轨迹点数",
                           v._pt_px is not None
                           and len(v._pt_px) == len(v.P_raw)))
            _snap = v._pt_px.copy()
            _mid = len(_snap) // 2         # 这个用例只有 9 个点，不能写死下标
            _px, _py = float(_snap[_mid][0]), float(_snap[_mid][1])
            v._pick_at(_px + 1.0, _py - 1.0)          # 故意偏 1px
            _i = v.sel_idx
            _ok = _i is not None and (
                ((_snap[_i][0] - _px) ** 2 + (_snap[_i][1] - _py) ** 2) ** 0.5
                <= PICK_RADIUS_PX)
            checks.append(("点中采样点 -> 选中它（偏 1px 仍命中）", _ok))
            v._pick_at(-9999.0, -9999.0)
            checks.append(("点空白 -> 取消选中", v.sel_idx is None))

            # 平移必须有边界。原生 `drag_pan` 直接改三轴 lim 且无 clamp，
            # 拖两下轨迹就整体跑出画布（用户截图："线超出被挡住"）。
            v.ax.set_xlim(v.hi[0] + 9.0,
                          v.hi[0] + 9.0 + (v.hi[0] - v.lo[0]))
            v._clamp_pan()
            _c = sum(v.ax.get_xlim()) / 2.0
            checks.append(("平移被夹在数据附近（不会把轨迹拖出画布）",
                           _c <= v.hi[0] + PAN_MARGIN * (v.hi[0] - v.lo[0])
                           + 1e-9))

            # 智能最佳视角 + 坐标极值（2026-11-05 用户："智能回到最适合的
            # 观察角度和缩放比例"、"自动计算并告知每个方向的坐标极值"）
            import math as _m
            from apps.view_3d import BEST_MAX_AXIS_DOT

            _e, _a = v._best_view_angles()
            _er, _ar = _m.radians(_e), _m.radians(_a)
            _fwd = (_m.cos(_er) * _m.cos(_ar), _m.cos(_er) * _m.sin(_ar),
                    _m.sin(_er))
            # 视线不能贴着任何一根轴：贴着谁，谁就在屏幕上退化成一点，
            # 刻度会全堆到画布角落（实测 azim=90 就是这个下场）。
            checks.append(("最佳视角：视线不贴任何轴（轴不会被压成一点）",
                           max(abs(c) for c in _fwd) <= BEST_MAX_AXIS_DOT + 1e-6))
            checks.append(("最佳视角：仰角保留立体感（15°~60°）",
                           15.0 <= _e <= 60.0))

            v._smart_view()
            v.canvas.draw()
            _tb = v.ax.get_tightbbox(v.canvas.get_renderer())
            _bx = v.ax.bbox
            # 容差 8%：tightbbox 会把标题算进高度（标题常驻，约 103%）
            checks.append(("最佳视角：内容（含轴标签）不越界",
                           _tb.width <= _bx.width * 1.08
                           and _tb.height <= _bx.height * 1.08))
            checks.append(("最佳视角：缩放被反解成有限值",
                           0.35 <= v.box_zoom <= 6.0))

            _ext = [t.get_text() for t in v.ax.texts
                    if "坐标极值" in t.get_text()]
            _P = v.P_raw
            checks.append(("坐标极值面板常驻且数值正确",
                           bool(_ext)
                           and f"{_P[:, 0].min():+.4f}" in _ext[0]
                           and f"{_P[:, 2].max():+.4f}" in _ext[0]))

            # 六视图方向（2026-11-05 用户："左视和右视应该是错的，显然 Z 轴
            # 才是横轴"）。坐标约定是 X 右 / Y 下 / Z 前（相机光轴朝前），
            # 所以侧视的**横轴必须是 Z**、纵轴是 Y 且向下增大。
            # mplot3d 里 z 轴默认永远竖直，必须靠 roll 转 90° 才放得平 ——
            # 下面的断言就是钉住这件事，改角度前先跑它。
            import numpy as _npv
            from mpl_toolkits.mplot3d import proj3d as _pj
            from apps.view_3d import VIEW_PRESETS as _VP

            def _axis_dirs(e, a, r):
                v.ax.view_init(elev=e, azim=a, roll=r)
                v.ax.set_proj_type("ortho")
                _M = v.ax.get_proj()
                _o = _npv.array(_pj.proj_transform(0, 0, 0, _M)[:2])
                _res = {}
                for _k, _q in (("X", (1, 0, 0)), ("Y", (0, 1, 0)),
                               ("Z", (0, 0, 1))):
                    _w = _npv.array(_pj.proj_transform(*_q, _M)[:2]) - _o
                    _n = float(_npv.linalg.norm(_w))
                    _res[_k] = (_w / _n) if _n > 1e-9 else _npv.zeros(2)
                return _res

            _d = _axis_dirs(*_VP["右视"])
            checks.append(("右视：横轴 Z（向右）、纵轴 Y（向下增大）",
                           _d["Z"][0] > 0.9 and _d["Y"][1] < -0.9))
            _d = _axis_dirs(*_VP["左视"])
            checks.append(("左视：横轴 Z（向左）、纵轴 Y（向下增大）",
                           _d["Z"][0] < -0.9 and _d["Y"][1] < -0.9))
            _d = _axis_dirs(*_VP["前视（相机看到的）"])
            checks.append(("前视：横轴 X（向右）、纵轴 Y（向下增大）",
                           _d["X"][0] > 0.9 and _d["Y"][1] < -0.9))
            _d = _axis_dirs(*_VP["俯视（从头顶）"])
            checks.append(("俯视：横轴 X（向右）、Z 竖直",
                           _d["X"][0] > 0.9 and abs(_d["Z"][1]) > 0.9))

            # 帧步进（2026-11-05："播放器那里增加前进/后退 x 帧的功能"）
            v.idx = 5
            v._step_frames(-3)
            _b1 = v.idx == 2
            v._step_frames(-999)
            _b2 = v.idx == 0
            v._step_frames(99999)
            _b3 = v.idx == v.n - 1
            checks.append((f"帧步进：±delta 生效且夹在 [0, n-1]"
                           f"（{_b1}/{_b2}/{_b3}）", _b1 and _b2 and _b3))

            # 左上角两个信息面板不能重叠（2026-11-05 用户截图：选中点的
            # Z 那一行被「坐标极值」压掉一半）。根因是偏移用了 1.0 当基准
            # （面板顶在 0.985）、且高度用常量估（实际 67.9 写成 64）。
            v._redraw()
            v.canvas.draw()
            v._pick_at(*v._pt_px[len(v._pt_px) // 2])
            v.canvas.draw()
            _rr = v.canvas.get_renderer()
            _ext_a = _sel_a = None
            for _t in v.ax.texts:
                _s = _t.get_text()
                if "坐标极值" in _s:
                    _ext_a = _t
                elif "frame" in _s:
                    _sel_a = _t
            _no_overlap = (
                _ext_a is not None and _sel_a is not None
                and _ext_a.get_window_extent(_rr).y0
                >= _sel_a.get_window_extent(_rr).y1)
            checks.append(("左上角「坐标极值」与选中点信息不重叠", _no_overlap))

            # 分度值可调（2026-11-05 用户："补充上手动调整三轴分度值的功能"）：
            # 填 0.05 就该每 5 cm 一根网格线。
            #
            # **语义在 2026-11-15 换了**（用户："修改分度值，确实坐标轴会变，但是
            # 图线不会自动的匹配放大和缩小"）：分度值 = 「每格多少米」，改它要按
            # **固定 8 格**重设该轴范围，和示波器的 V/格 同构。所以这里改成走
            # `_on_tick_edit()`（范围 + 刻度一起），不再只调 locator。
             #
            # 注意：上面 `_zoom_around` 已把范围改成 [-1.5, 1.5]（跨度 3），
            # 而这段合成数据的 X 跨度只有 0.08 m（`_limits()` 加留白后 0.0992），
            # 所以 0.5 × 8 = 4 远超数据跨度 → 窗口会被**夹回数据范围**；
            # 但刻度间隔仍然是 0.5，判据照样成立。
            v.tick_vars["x"].set("0.5")
            v.tick_vars["y"].set("")
            v.tick_vars["z"].set("")
            v._on_tick_edit()
            xt = [round(float(t), 4) for t in v.ax.get_xticks()]
            gaps = {round(b - a, 4) for a, b in zip(xt, xt[1:])}
            checks.append(("分度值 0.5 -> X 轴刻度间隔都是 0.5",
                           bool(xt) and gaps == {0.5}))
            # 旧判据是「填 0.0001 → 退回自动、刻度 ≤ 10」。那条现在**必然不成立**，
            # 因为新语义下 0.0001 是合法的放大：8 格 = 0.8 mm。这不是"为了绿而改
            # 判据"，是模型换了 —— 旧行为（静默退回自动档）本身就是这次要修的缺陷。
            v.tick_vars["x"].set("0.0001")
            v._on_tick_edit()
            _x_span = v.lims[1][0] - v.lims[0][0]
            checks.append((f"分度值 0.0001 -> 不再退回自动，按 8 格放大"
                           f"（X 跨度 {_x_span:.6f} ≈ 0.0008）",
                           abs(_x_span - 0.0001 * TICK_GRID_CELLS) < 1e-9))
            v.tick_vars["x"].set("")
            v._on_tick_edit()

            # 滚轮默认 = **整幅缩放**（2026-11-05 用户："缩放只有曲线缩放，
            # 没有背景坐标图缩放，很别扭"）。守的是：默认滚轮不许再只改 lims
            # —— 盒子大小固定、网格间距不变，看起来就是"背景一动不动"。
            class _Wheel:
                inaxes = v.ax
                x = 120
                y = 120
                button = "up"
                key = None

            z_before = v.box_zoom
            lim_before = [list(a) for a in v._current_lims()]
            v._on_scroll(_Wheel())
            checks.append(("默认滚轮改整幅缩放（网格/刻度一起放大），不动范围",
                           v.box_zoom > z_before
                           and [round(q, 6) for q in v._current_lims()[0]]
                           == [round(q, 6) for q in lim_before[0]]))

            # Shift+滚轮 = 定点放大（改 lims，不改整幅缩放）
            z_mid = v.box_zoom
            lim_mid = [list(a) for a in v._current_lims()]

            class _WheelShift(_Wheel):
                key = "shift"

            v._on_scroll(_WheelShift())
            checks.append(("Shift+滚轮改范围（定点放大），不动整幅缩放",
                           abs(v.box_zoom - z_mid) < 1e-9
                           and v._current_lims()[0][0] > lim_mid[0][0]))
            v._reset_view()
            checks.append(("重置后范围重新 fit（lims 置空再按包围盒算）",
                           v.lims is None or
                           v._current_lims()[0][0] <= v.lo[0]))
            checks.append(("四宫格外层是可拖动的 PanedWindow",
                           isinstance(v._grid, _ttk.PanedWindow)))
            # 初始视频区不能塌成 0（2026-11-04 用户报"视频显示范围为 0，得
            # 让人拖出来"）。根因是首帧时 `sashpos` 读到 winfo_height()==1。
            # 这里不依赖真实窗口映射（本测试窗口是 withdraw 的，paned 高度
            # 恒为 1、sashpos 会被 clamp 到 0），改为直接验 `_seed_video_sash`
            # 的分支逻辑：用一个假 sashpos 记录它要设的值。
            calls: list[int] = []
            v._paned_seeded = False
            _real_sashpos = v._grid.sashpos
            _real_h = v._grid.winfo_height
            v._grid.winfo_height = lambda: 500
            v._grid.sashpos = lambda i, pos: calls.append(pos)
            try:
                v._seed_video_sash()            # 高度够 → 应设 500*0.4=200
                v._seed_video_sash()            # 已 seeded → 不应再设
            finally:
                v._grid.winfo_height = _real_h
                v._grid.sashpos = _real_sashpos
            checks.append(("视频区 sash 首次定位到非 0（约总高 40%）",
                           calls == [200]))
            # 高度不足时不动：重置门，高度 1 → 不设、且门仍开着
            v._paned_seeded = False
            calls.clear()
            v._grid.winfo_height = lambda: 1
            try:
                v._seed_video_sash()
            finally:
                v._grid.winfo_height = _real_h
            checks.append(("首帧高度不足时不设 sash（避免设到 0）且门仍开",
                           calls == [] and v._paned_seeded is False))

            # ---- 8d-4. 视频来源窗口内切换（2026-11-20）----
            # 老界面把 raw 藏在数据管理里，新查看器此前只能按启动参数走。
            # 这里钉三件事：下拉三档与 CLI 一致、切「不看」真的关 cap 清帧、
            # 切回「标注」真的重开且不丢当前帧。
            checks.append(("视频来源下拉三档与 --video-source 一致",
                           list(v.video_box.cget("values"))
                           == [VIDEO_SOURCE_LABELS[k] for k in VIDEO_SOURCE_ORDER]))
            had_cap = v.cap is not None
            v.idx = min(2, v.n - 1)
            keep_idx = v.idx
            v.video_box.set(VIDEO_SOURCE_LABELS["none"])
            v._on_pick_video_source()
            checks += [
                ("视频来源切「不看」：video_source=none",
                 v.video_source == "none"),
                ("视频来源切「不看」：cap 关掉、旧帧清空",
                 v.cap is None and v._left_raw is None and v._right_raw is None),
            ]
            v.video_box.set(VIDEO_SOURCE_LABELS["annotated"])
            v._on_pick_video_source()
            checks += [
                ("切回「标注」：重新打开 annotated.mp4",
                 had_cap and v.cap is not None),
                ("切来源不丢当前帧（idx 保留）", v.idx == keep_idx),
            ]

            # ---- 8d-5. 轴标签跟着数据走（计划书 M6 世界系参考）----
            # `visualization.py` 应用 world_frame 后会把 `style.axes` 换成
            # World X/Y/Z；查看器若仍写死相机系标签，就是"数据换了系、标签
            # 在说谎"。默认（无 style）必须保持原来的相机系措辞。
            default_x = v.ax.get_xlabel()
            v.data["style"] = {
                "axes": {"x": "World X (board columns)",
                         "y": "World Y (board rows)",
                         "z": "World Z (ground normal)"},
            }
            v._redraw()
            checks += [
                ("默认轴标签是相机系（X right）", default_x.startswith("X right")),
                ("世界系数据用 World X/Y/Z 标签（标签不说谎）",
                 v.ax.get_xlabel() == "World X (board columns)"
                 and v.ax.get_ylabel() == "World Y (board rows)"
                 and v.ax.get_zlabel() == "World Z (ground normal)"),
            ]
            v.data.pop("style", None)
            v._redraw()
        finally:
            if v is not None:
                v.close()       # 释放 cap，否则临时目录删不掉（WinError 32）
            else:
                root.destroy()

    # 切到"没有视频"的会话：必须清掉上一段的解码帧
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        a = mk(base, "20260101-000000", n=4, video=True)
        b = mk(base, "20260202-000000", n=4, video=False)
        root = tk.Tk()
        root.withdraw()
        v = None
        try:
            v = Viewer3D(root, load_view(a / "derived" / "visualization.json"))
            had = v._left_raw is not None and v._right_raw is not None
            v.open_source(b / "derived" / "visualization.json")
            checks += [
                ("切换前确实解码到了帧（否则这条是空跑）", had),
                ("新会话没视频 → 两路都是 None（不是旧帧）",
                 v._left_raw is None and v._right_raw is None),
                ("cap 也关掉了（不占文件句柄）", v.cap is None),
            ]
        finally:
            if v is not None:
                v.close()
            else:
                root.destroy()

    # 坏源：抛错，但不能把正在看的会话弄没
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        a = mk(base, "20260101-000000", n=6, video=False)
        empty = base / "20260303-000000"
        empty.mkdir()
        root = tk.Tk()
        root.withdraw()
        v = None
        try:
            v = Viewer3D(root, load_view(a / "derived" / "visualization.json"))
            before_path, before_n = str(v.data.get("_path")), v.n
            raised = False
            try:
                v.open_source(empty)
            except Exception:                               # noqa: BLE001
                raised = True
            checks += [
                ("坏源抛错（不是静默忽略）", raised),
                # 用 .get：坏实现可能已把 data 清掉，"路径没了"也要算红，
                # 而不是让测试自己 KeyError 崩掉（崩掉看不出是哪一条）
                ("当前视图没变（路径与点数都还是原来那个）",
                 str(v.data.get("_path")) == before_path and v.n == before_n),
            ]
        finally:
            if v is not None:
                v.close()
            else:
                root.destroy()

    return checks


def _main_impl() -> int:
    print("=" * 70)
    print("启动器标定参数面板自测（读取 / 校验 / 默认正式档）")
    print("=" * 70)

    if tk is None:
        print("  [跳过] 这台机器没有 tkinter —— 界面测试无法进行。")
        print("         （`run.bat gui` 在这种情况下会自动改用命令行界面。）")
        print("\n判定结果：全部通过")
        return 0

    mod = _load_launcher()
    box = _Box()
    mod.messagebox = box                          # type: ignore[assignment]
    app = mod.Launcher()
    app.update_idletasks()
    app.update()
    res_default = app.calib_res.get()      # 建窗后立刻记下默认档，后面 7b 要断言
    screen_ok = (
        0 <= app.winfo_x() and 0 <= app.winfo_y()
        and app.winfo_x() + app.winfo_width() <= app.winfo_screenwidth()
        and app.winfo_y() + app.winfo_height() <= app.winfo_screenheight()
    )
    head_ok = (app._header_brand.winfo_x() + app._header_brand.winfo_width()
               <= app._header_actions.winfo_x() + 1)

    results: list[tuple[str, bool]] = []

    # ---- 界面默认值必须与命令行/文档一致（2026-10-14 加）-----------------
    # 为什么要有这一条：这里曾经是 `METHODS[0] = "fusion"` + `set("fusion")`，
    # 而 `run_realtime.py --method` 默认是 `color`、`docs/用户手册.md` §3.1
    # 也写着"`color`（默认，最快）"。于是**从界面走的人必然吃上 fusion**
    # 在真实纹理下的 265 ms/帧（实测 loop 3.7 fps、丢帧 91%），
    # 而界面上没有任何迹象说明这个选项慢在哪 ——
    # 与 `--target` 当年没传下去是同一类缺陷（见 `交接文档.md` §13.1）。
    # 所以钉两件：界面默认 == 文档默认，且与脚本默认**逐字相同**。
    print("----- 0. 检测方法的默认值必须与命令行一致 -----")
    # `run_realtime.py` 的 `--method` 默认是 `None`，不传时沿用
    # `DetectorConfig.method`。所以"脚本默认"要从**真正生效的那处**取 ——
    # 直接用 `AppConfig()` 读，而不是去 grep 源码字符串
    # （我第一版就是 grep `"color" in rt_src`，写成了一条会被正确实现判红的坏断言）。
    from sphere_tracker.config import AppConfig as _AppConfig
    from sphere_tracker.detectors import registry as _detector_registry  # noqa: F401  （import 即注册）
    from sphere_tracker.plugins import default as _plugin_default
    from sphere_tracker.plugins import names as _plugin_names
    script_default = _AppConfig().detector.method
    got_default = app.method.get()
    print(f"  界面默认 = {got_default!r}；"
          f"DetectorConfig 的默认（= 不传 --method 时生效）= {script_default!r}")
    # 断言"注册表首项 == 注册表默认 == config 默认"，而不是"列表第 0 项的字面量"：
    # 清单顺序是展示细节，默认值才是契约（`docs/插件化架构.md` §5）。
    checks_default = [
        (f"界面默认检测方法与脚本默认逐字相同（{script_default}）",
         got_default == script_default),
        ("下拉首项 == default(\"detector\") == DetectorConfig().method",
         _plugin_names("detector")[0] == _plugin_default("detector") == script_default),
        ("下拉框内容就是注册表（界面没有手写清单）",
         list(app.method["values"]) == _plugin_names("detector")),
        ("color 排在 fusion 之前（默认项不该埋在列表中间）",
         _plugin_names("detector").index("color")
         < _plugin_names("detector").index("fusion")),
    ]
    for name, ok in checks_default:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
        results.append((name, ok))
    print()

    def expect(label: str, ok_expected: bool, contains: str = "",
               absent: str = "") -> None:
        box.msgs.clear()
        args = app._calibrate_args()
        got_ok = args is not None
        joined = " ".join(args) if args else ""
        good = got_ok == ok_expected
        if good and got_ok:
            if contains and contains not in joined:
                good = False
            if absent and absent in joined:
                good = False
        # 被拒时必须给出原因 —— 静默拒绝等于让用户猜
        if good and not ok_expected and not box.msgs:
            good = False
        print(f"  [{'通过' if good else '不达标'}] {label}")
        if not good:
            print(f"      期望 ok={ok_expected} 含{contains!r} 不含{absent!r}")
            print(f"      实际 args={joined or None}")
            if box.msgs:
                print(f"      弹窗：{box.msgs[0][:96]}")
        results.append((label, good))

    # ---- 1. 默认就是正式档 ----
    print("----- 1. 默认值 = 正式档（这条是本测试存在的主要理由）-----")
    expect("默认传 --target 40 --min-pairs 20", True,
           contains="--target 40 --min-pairs 20")

    # ---- 2. 板参数真的来自面板 ----
    print("----- 2. 板参数来自面板，不是硬编码 -----")
    _set(app.calib_cols, "7")
    _set(app.calib_rows, "5")
    _set(app.calib_square, "30")
    expect("改面板 -> 命令行跟着变（7×5 / 30mm）", True,
           contains="--cols 7 --rows 5 --square 30")
    _set(app.calib_cols, "8")
    _set(app.calib_rows, "5")
    _set(app.calib_square, "25")

    # ---- 2.5 相机后端必须**从面板传下去**（2026-10-14 加）----
    #
    # 为什么要有这一条：这里原来把 `--backend msmf` **写死**在 `_calibrate_args` 里，
    # 而「复检」与「前置检查」两条命令**根本不传**。后果是界面完全没有选择权，
    # 而本机实测 **MSMF 打开两路要 34.6 s、DSHOW 只要 11.8 s**（两个后端给的
    # 分辨率完全一致，实测两轮）—— 从界面走的人只能白等半分钟，
    # 而且"同一个决定散在三处"正是本项目反复踩的那类问题。
    # 现在只有一个决定点（面板），三条命令都走 `_backend_args()`。
    print("----- 2.5 相机后端来自面板，且三条命令用的是同一个决定 -----")
    app.calib_backend.set("dshow")
    expect("选 dshow -> 标定命令带上 --backend dshow", True, contains="--backend dshow")
    app.calib_backend.set("auto")
    expect("选 auto -> **不传** --backend（落到脚本默认，不替用户改行为）",
           True, absent="--backend")
    app.calib_backend.set("msmf")
    expect("选 msmf -> 带上 --backend msmf", True, contains="--backend msmf")
    # 三条命令共用同一个决定点：源码级确认没有第二份内联写法
    launcher_src = (ROOT / "apps/gui_main.py").read_text(encoding="utf-8")
    inline = launcher_src.count('args += ["--backend", backend]')
    uses_helper = launcher_src.count("self._backend_args()")
    print(f"  源码：内联写 backend 参数 {inline} 处；调用 _backend_args() {uses_helper} 处")
    results.append(("后端只有一个决定点（没有第二份内联复制）",
                    inline == 0 and uses_helper >= 3))
    print(f"  [{'通过' if inline == 0 and uses_helper >= 3 else '不达标'}] "
          f"后端只有一个决定点（内联 0 处、helper 调用 ≥3 处）")
    app.calib_backend.set("auto")

    # ---- 3. 面板默认值必须与脚本/CaptureConfig 的默认值**一致** ----
    #
    # ⚠️ **2026-10-14 改了契约**：原来这些框是**空的**，断言"留空 = 不传"。
    # 用户要求"填入默认的设置值"，所以现在框里**填着真实默认值**。
    # 于是"留空"从常态变成"用户主动清掉"，其语义仍是**不传该参数**
    # （`calibrate_stereo.py` 会退回自己的默认值）。
    #
    # 这里钉的是**两处默认值必须相同**：面板填 3 / 脚本默认也是 3，
    # 所以"不改任何东西直接点运行"的行为与改动前**完全一致**。
    # 若哪天脚本把默认值改成 4 而面板还写 3，这条就会红 —— 那正是要防的漂移。
    print("----- 3. 面板默认值 == 脚本/CaptureConfig 默认值 -----")
    import re as _re2
    calib_src = (ROOT / "apps" / "calibrate_stereo.py").read_text(encoding="utf-8")
    from calib.capture import CaptureConfig as _CC
    _cc = _CC()
    # argparse 的 default=None 表示"不传就用 CaptureConfig"；这里取后者为准
    expect_stationary = f"{_cc.stationary_max_px:g}"
    expect_sharp = f"{_cc.min_sharpness:g}"
    got_stationary = app.calib_stationary.get().strip()
    got_sharp = app.calib_sharp.get().strip()
    print(f"  停稳上限：面板 {got_stationary!r} / CaptureConfig {expect_stationary}")
    print(f"  清晰度  ：面板 {got_sharp!r} / CaptureConfig {expect_sharp}")
    for label, got, want in (("停稳上限 px", got_stationary, expect_stationary),
                             ("清晰度下限", got_sharp, expect_sharp),
                             ("目标对数", app.calib_target.get().strip(), "40"),
                             ("最少对数", app.calib_minpairs.get().strip(), "20"),
                             ("内角点·列", app.calib_cols.get().strip(), "8"),
                             ("内角点·行", app.calib_rows.get().strip(), "5"),
                             ("方格边长 mm", app.calib_square.get().strip(), "25")):
        good = (float(got) == float(want)) if got else False
        print(f"  [{'通过' if good else '不达标'}] {label}: 面板 {got!r} == 脚本 {want}")
        results.append((f"面板默认值 == 脚本默认值（{label}）", good))

    # 基线**故意**留空：程序无法替你猜装置中心距。这条也要钉住，
    # 免得有人"顺手"填个数字进去 —— 那会让基线核对用错误的锚。
    print(f"  预期基线：面板 {app.calib_baseline.get()!r}（**应当为空**）")
    good = app.calib_baseline.get().strip() == ""
    print(f"  [{'通过' if good else '不达标'}] 预期基线**故意留空**（程序猜不出来）")
    results.append(("预期基线故意留空（程序无法替你猜中心距）", good))

    # ---- 3b. 清空 = 不传该参数（原来的语义仍然成立）----
    print("----- 3b. 清空某个框 = 不传该参数 -----")
    _set(app.calib_stationary, "")
    expect("清空停稳上限 -> 不传 --stationary-px", True, absent="--stationary-px")
    _set(app.calib_stationary, "3")
    expect("填回 3 -> 又传下去", True, contains="--stationary-px 3.0")
    expect("未勾放宽不传 --relaxed", True, absent="--relaxed")

    # ---- 4. 填了就传 ----
    print("----- 4. 高级项填了就真的传下去 -----")
    _set(app.calib_baseline, "65")
    expect("填 65 -> 传 --baseline-mm 65（开启基线核对）", True,
           contains="--baseline-mm 65")
    _set(app.calib_baseline, "")
    _set(app.calib_stationary, "5.5")
    expect("填 5.5 -> 传 --stationary-px 5.5", True,
           contains="--stationary-px 5.5")
    _set(app.calib_stationary, "")
    _set(app.calib_sharp, "1200")
    expect("填 1200 -> 传 --min-sharpness 1200（口径是尺度无关的 focus）", True,
           contains="--min-sharpness 1200")
    _set(app.calib_sharp, "")
    app.calib_relaxed.set(True)
    expect("勾放宽 -> 传 --relaxed", True, contains="--relaxed")
    app.calib_relaxed.set(False)
    _set(app.calib_savepairs, str(ROOT / "outputs" / "tmp_pairs"))
    expect("填存图目录 -> 传 --save-pairs", True, contains="--save-pairs")
    _set(app.calib_savepairs, "")

    # ---- 5. 非法输入 / 矛盾必须被拦住 ----
    print("----- 5. 非法输入与参数矛盾必须被拦住（且说明原因）-----")
    for bad, label in (("abc", "非数字"), ("0", "越界(0)"), ("999", "越界(999)")):
        _set(app.calib_target, bad)
        expect(f"目标对数={bad}（{label}）-> 拒绝", False)
        _set(app.calib_target, "40")
    _set(app.calib_baseline, "xyz")
    expect("预期基线=xyz -> 拒绝", False)
    _set(app.calib_baseline, "")
    _set(app.calib_stationary, "99")
    expect("停稳上限=99 -> 拒绝（>12 会让两路拍到不同位姿）", False)
    _set(app.calib_stationary, "")
    _set(app.calib_target, "5")
    _set(app.calib_minpairs, "10")
    expect("最少对数 > 目标对数 -> 拒绝（否则永远达不到下限）", False)
    _set(app.calib_target, "40")
    _set(app.calib_minpairs, "20")

    # ---- 6. 试标定档仍合法 ----
    print("----- 6. 低目标档仍合法（只是标定程序会显式警告）-----")
    _set(app.calib_target, "8")
    _set(app.calib_minpairs, "8")
    expect("target=8 -> 允许（验证通信用）", True, contains="--target 8")
    _set(app.calib_target, "40")
    _set(app.calib_minpairs, "20")

    # ---- 7. 分辨率/帧率 ----
    print("----- 7. 分辨率 / 帧率仍然生效 -----")
    app.calib_res.set("1280x720")
    expect("1280x720 -> --width 1280 --height 720", True,
           contains="--width 1280 --height 720")
    app.calib_res.set("1920x1080")
    expect("默认帧率 24 -> --fps 24", True, contains="--fps 24")

    # ---- 7b. 分辨率档位来自 video_profiles（唯一来源），默认 2560x1440@24 ----
    # 守的是"GUI 与录制/标定/实时共用同一份档位表"：
    # 原来 RESOLUTIONS 与实时入口的 1280x720 默认值是两份独立数字，
    # 面板选 1080p 而实时按 720p 开，界面上看不出来。
    # 默认**就是** video_profiles.DEFAULT（当前 2560x1440）：真机实测两台外接
    # 相机无视 1920/1280 请求、恒给 2560x1440；record_stereo 1440p@30 主循环
    # 只有 24.07fps，所以帧率默认 24。旧 1080 档仍在表里、仍可显式选择。
    print("----- 7b. 分辨率唯一来源 / 默认 1440p@24 / 实时检测宽度 -----")
    from types import SimpleNamespace
    from apps import record_stereo as record_app
    from sphere_tracker.video_profiles import RECORD_DEFAULT
    from sphere_tracker.video_profiles import options as video_options
    record_src = (ROOT / "apps" / "record_stereo.py").read_text(encoding="utf-8")
    fake_rig = SimpleNamespace(image_size=[1920, 1080])
    try:
        record_app.validate_capture_size(fake_rig, 1920, 1080)
        record_same_ok = True
    except ValueError:
        record_same_ok = False
    try:
        record_app.validate_capture_size(fake_rig, 2560, 1440)
        record_mismatch_blocked = False
    except ValueError:
        record_mismatch_blocked = True
    app.calib_res.set("2560x1440")
    rec_hd = " ".join(app._record_args("10", "gui_test"))
    rt_hd = " ".join(app._realtime_args())
    app._refresh_res_note()
    note_hd = app.res_note.cget("text")
    app.calib_res.set("1920x1080")
    rt_1080 = " ".join(app._realtime_args())
    rec_1080 = " ".join(app._record_args("10", "gui_test"))
    app._refresh_res_note()
    note_1080 = app.res_note.cget("text")
    app.calib_res.set("1280x720")
    rt_720 = " ".join(app._realtime_args())
    app.calib_res.set(res_default)
    app._refresh_res_note()
    mod_src = (ROOT / "apps/gui_main.py").read_text(encoding="utf-8")
    checks_res = [
        ("默认分辨率 = 2560x1440（video_profiles.DEFAULT；不随某台相机实际能力变）",
         res_default == mod.DEFAULT_RESOLUTION == "2560x1440"),
        # ⚠️ **2026-10-31 改了这条断言的前提（不是删判据）**：录制侧的
        # `--width/--height` 默认从 `RECORD_DEFAULT` 改成了 `None`（未显式给时
        # 优先读标定 meta 的 `capture_size`，与标定同一档；没有才退回
        # `RECORD_DEFAULT`）。判据**守的真实意图没变** —— "录制档不写死 1080、
        # 且真的会去读标定采集档、兜底数值仍来自 `RECORD_DEFAULT`"。
        # 断言**具体那两行**默认值（而不是宽泛的 `'default=None'`，`record_stereo.py`
        # 里还有别的 `default=None`）以及**真的调用了** `calib_capture_size(`。
        # 依据：`apps/record_stereo.py` 的采集档分支；planbook
        # `2026-10-31_双目标定取帧失败与分辨率档.md` §8.3 S2。
        ("录制档不写死：--width/--height 默认 None，且真的读标定 capture_size"
         "（兜底数值仍来自 video_profiles.RECORD_DEFAULT）",
         RECORD_DEFAULT.label == "2560x1440"
         and '"--width", type=int, default=None' in record_src
         and '"--height", type=int, default=None' in record_src
         and "calib_capture_size(" in record_src
         and "RECORD_DEFAULT.width" in record_src
         and "RECORD_DEFAULT.height" in record_src),
        ("录制前校验标定分辨率：旧 1080 标定 + 1080 录制仍放行（旧会话可读）",
         record_same_ok),
        ("录制前校验标定分辨率：1080 标定 + 1440 录制报错（不静默混用）",
         record_mismatch_blocked),
        ("档位表来自 video_profiles.options()（含 2560x1440 / 1080p / 720p）",
         list(mod.RESOLUTIONS) == list(video_options())
         and "2560x1440" in mod.RESOLUTIONS
         and "1920x1080" in mod.RESOLUTIONS and "1280x720" in mod.RESOLUTIONS),
        ("录制在 2560x1440 档传 --width 2560 --height 1440（与标定同源）",
         "--width 2560 --height 1440" in rec_hd),
        ("旧 1080 档仍可显式选择：录制传 --width 1920 --height 1080",
         "--width 1920 --height 1080" in rec_1080),
        ("实时在 2560x1440 档传 --width 2560 --height 1440（不再固定 720p）",
         "--width 2560 --height 1440" in rt_hd),
        ("实时在 1080 档传 --width 1920 --height 1080",
         "--width 1920 --height 1080" in rt_1080),
        ("实时传面板帧率（默认 24，不再让 run_realtime 用 60）",
         app.calib_fps.get().strip() == "24" and "--fps 24" in rt_1080),
        ("高分辨率实时加 --detect-width 1280",
         "--detect-width 1280" in rt_hd),
        ("1080p 不加 --detect-width（不强制降采样）",
         "--detect-width" not in rt_1080),
        ("720p 不加 --detect-width", "--detect-width" not in rt_720),
        ("选 2560 时界面提示主循环 24.07fps 是目标、30 会丢帧（不静默承诺 30fps）",
         "24.07" in note_hd and "丢帧" in note_hd and "242/300" in note_hd
         and "24 fps 是稳定目标" in note_hd),
        ("1080p 不显示吞吐警告（少噪音）", note_1080 == ""),
        ("_start_realtime 复用 _realtime_args（不是两套拼法）",
         "cmd += self._realtime_args()" in mod_src),
    ]
    for label, good in checks_res:
        print(f"  [{'通过' if good else '不达标'}] {label}")
        results.append((label, good))
    # ⚠️ **2026-10-14 改了这条断言**：它原来是
    #   `expect("两台相机必须同后端（backend 固定 msmf）", True, contains="--backend msmf")`
    # 也就是把"写死 msmf"当成契约。现在后端由面板决定（见第 2.5 节），
    # 但**它要守的真实意图没有变**：`--backend` 是**一个**参数、两台相机共用，
    # 拿不到"左路 dshow、右路 msmf"这种会直接把 MSMF 那路弄挂的组合。
    # 所以断言改成"不多不少恰好一个 --backend"。
    app.calib_backend.set("dshow")
    args_b = app._calibrate_args() or []
    n_be = args_b.count("--backend")
    print(f"  选中 dshow 时：--backend 出现 {n_be} 次（应恰好 1 —— 两台共用同一个）")
    results.append(("两台相机同后端：--backend 只出现一次", n_be == 1))
    print(f"  [{'通过' if n_be == 1 else '不达标'}] 两台相机同后端：--backend 只出现一次")
    app.calib_backend.set("auto")

    # ---- 8. GUI 录制按钮 ----
    print("----- 8. GUI「录制双目」按钮的参数 -----")
    rec = app._record_args("10", "gui_test")
    rec_s = " ".join(rec)
    checks_rec = [
        ("调用 apps/record_stereo.py", "record_stereo.py" in rec_s),
        ("带 --duration 10", "--duration 10" in rec_s),
        ("带 --preview", "--preview" in rec_s),
        ("带 --label gui_test", "--label gui_test" in rec_s),
        ("带 --left left --right right", "--left left --right right" in rec_s),
        ("带 --fps 24", "--fps 24" in rec_s),
        ("默认双目模式 auto", "--stereo-mode auto" in rec_s),
    ]
    for label, good in checks_rec:
        print(f"  [{'通过' if good else '不达标'}] {label}")
        results.append((label, good))

    # 判据7：GUI 录制把「录制格式」下拉透传给 record_stereo 的 --container。
    # 反向自证：删掉 _record_args 里的 --container 行，这条必红。
    app.record_container.set("avi")
    rec_container_avi = " ".join(app._record_args("10", "gui_test"))
    app.record_container.set("mkv")
    rec_container_mkv = " ".join(app._record_args("10", "gui_test"))
    container_ok = ("--container avi" in rec_container_avi
                    and "--container mkv" in rec_container_mkv)
    print(f"  [{'通过' if container_ok else '不达标'}] "
          f"GUI 录制透传 --container（avi/mkv 随下拉切换）")
    results.append(("GUI 录制透传 --container（avi/mkv 随下拉切换）", container_ok))

    app.stereo_mode_var.set("angled")
    rec_ang = " ".join(app._record_args("10", "gui_test"))
    cal_ang = " ".join(app._calibrate_args() or [])
    mode_ok = "--stereo-mode angled" in rec_ang and "--stereo-mode angled" in cal_ang
    print(f"  [{'通过' if mode_ok else '不达标'}] 切换大角度模式后录制/标定都传 angled")
    results.append(("切换大角度模式后录制/标定都传 angled", mode_ok))
    app.stereo_mode_var.set("auto")
    app.calib_backend.set("dshow")
    rec_backend = " ".join(app._record_args("10", "gui_test"))
    app.calib_backend.set("auto")
    backend_ok = "--backend dshow" in rec_backend
    print(f"  [{'通过' if backend_ok else '不达标'}] 录制也读取相机后端（不再只影响标定）")
    results.append(("录制也读取相机后端（不再只影响标定）", backend_ok))

    # ---- 8c. 录制参数体检：模式与标定必须一致 ----
    #
    # 2026-10-28 真机：把「双目模式」从 auto 改成 angled，分辨率仍是 2560x1440，
    # 点录制后得到「标定内参基于 1718x972，当前录制请求 2560x1440」。
    # 链路没错（angled 忽略 FOV 是有意的），错在**三个独立控件**可以组合成
    # 一个从定义上不成立的方案，而判定发生在相机已开、会话目录已建之后。
    #
    # 这里全部用**合成标定**：判据不能依赖"这台机器恰好标定过"
    # （B-27 的教训：干净 checkout 上会假红）。
    print("----- 8c. 录制参数体检：模式与标定必须一致（不能等开相机才发现）-----")
    from stereo.record_plan import output_size, plan_record

    par_fov = {"reference_side": "left", "crop_side": "right",
               "crop_rect": [404, 195, 1718, 973], "target_size": [1718, 972],
               "scale": 0.664875885304807, "method": "sift",
               "inliers": 296, "quality": 0.512998266897747}
    par_rig = SimpleNamespace(stereo_mode="parallel", image_size=(1718, 972),
                              meta={"stereo_mode": "parallel", "fov": par_fov})
    ang_rig = SimpleNamespace(stereo_mode="angled", image_size=(2560, 1440),
                              meta={"stereo_mode": "angled"})

    p_auto = plan_record(par_rig, mode_choice="auto", width=2560, height=1440)
    p_par = plan_record(par_rig, mode_choice="parallel", width=2560, height=1440)
    p_ang = plan_record(par_rig, mode_choice="angled", width=2560, height=1440)
    p_1080 = plan_record(par_rig, mode_choice="auto", width=1920, height=1080)
    p_ang_ok = plan_record(ang_rig, mode_choice="angled", width=2560, height=1440)
    p_ang_par = plan_record(ang_rig, mode_choice="parallel", width=2560, height=1440)
    # 尺寸凑对了、但模式错：parallel 标定**没有 FOV**、内参就是 2560x1440，
    # 所以写盘尺寸与内参完全一致（尺寸判据放行），只有"模式必须一致"这条能拦。
    # 这才是更坏的情形：录得下来、不报错，但 R/T 用的是平行假设。
    par_raw = SimpleNamespace(stereo_mode="parallel", image_size=(2560, 1440),
                              meta={"stereo_mode": "parallel"})
    p_raw_ang = plan_record(par_raw, mode_choice="angled", width=2560, height=1440)
    ang_fixes = " ".join(p_ang.fixes)

    checks_plan = [
        ("parallel + FOV：auto 放行，写盘 = 标定裁剪后的 1718x972",
         p_auto.ok and p_auto.save_size == (1718, 972)),
        ("parallel 标定 + angled 模式：**拦下**（这次真机事故的组合）",
         (not p_ang.ok) and p_ang.mode_conflict),
        ("拦下时说清了是模式冲突（不是让人以为分辨率没调好）",
         "parallel" in p_ang.headline and "angled" in p_ang.headline),
        ("给出了可执行的一步（改回 auto 或去重标），不是只报现象",
         "auto" in ang_fixes and "重标" in ang_fixes),
        ("1080 装不下标定裁剪区 [404+1718]：拦下",
         (not p_1080.ok) and not p_1080.mode_conflict),
        ("angled 标定 + angled 录：放行（2560x1440 = 内参）",
         p_ang_ok.ok and p_ang_ok.save_size == (2560, 1440)),
        ("反向也拦：angled 标定 + parallel 模式",
         (not p_ang_par.ok) and p_ang_par.mode_conflict),
        ("尺寸凑对了但模式错也拦（写盘 2560x1440 == 内参，只有模式判据能抓）",
         (not p_raw_ang.ok) and p_raw_ang.mode_conflict
         and p_raw_ang.save_size == p_raw_ang.calib_size == (2560, 1440)),
        ("output_size 是唯一来源：angled 显式忽略 FOV，parallel 用它",
         output_size("angled", None, 2560, 1440) == (2560, 1440)
         and output_size("parallel", None, 2560, 1440) == (2560, 1440)),
        ("record_stereo 用同一个 output_size（不自己再算一份）",
         "out_w, out_h = output_size(mode, fov" in record_src),
        ("录制入口在开相机前先体检（否则相机白开、空会话白建）",
         # 锚点用**调用点**的写法：`_open_pair(args)` 这个串在 `def _open_pair(args)`
         # 那行也会匹配，拿它比顺序会永远比到函数定义处。
         "plan_record(StereoRig.load(calib_base)" in record_src
         and "cams, code = _open_pair(args)" in record_src
         and record_src.index("plan_record(StereoRig.load(calib_base)")
         < record_src.index("cams, code = _open_pair(args)")),
        ("旧窗口录制前会体检，未通过就不启动",
         "plan = self._record_plan()" in mod_src
         and "_confirm_record_plan(plan)" in mod_src),
    ]
    for label, good in checks_plan:
        print(f"  [{'通过' if good else '不达标'}] {label}")
        results.append((label, good))

    # ---- 8d. 3D 查看器能换任意会话 ----
    print("----- 8d. 3D 查看器：能换任意会话，换坏不毁当前视图 -----")
    for label, good in _check_view_3d_switch():
        print(f"  [{'通过' if good else '不达标'}] {label}")
        results.append((label, good))

    # ---- 8b. 90° 装置标定入口 / 模式标签 / 画质诊断（2026-10-24 加）----
    #
    # 守卫三件界面契约：
    # 1. `mode_label("auto")` 不能再回"未知" —— 下拉的默认项就是 auto，
    #    显示"未知"会让用户以为选项非法；
    # 2. 「标定（90° 装置）」必须**复用** `_calibrate_args` 并带上
    #    `--stereo-mode angled --skip-fov`（不是另写一份拼装）；
    # 3. 画质诊断有 GUI 入口，且走 `--together`（单开测不出总线争抢）。
    print("----- 8b. 90° 标定入口 / 模式标签 / 画质诊断 -----")
    from stereo.stereo_mode import mode_label
    app.stereo_mode_var.set("auto")
    args90 = app._calibrate_args(angled_90=True) or []
    s90 = " ".join(args90)
    args_plain = " ".join(app._calibrate_args() or [])
    checks_90 = [
        ("mode_label(auto) 不再是「未知」", mode_label("auto") == "自动判定（按标定）"),
        ("mode_label(无从识别) 仍是「未知」", mode_label("???") == "未知"),
        ("90° 入口带 --stereo-mode angled", "--stereo-mode angled" in s90),
        ("90° 入口带 --skip-fov（不白跑 FOV 对齐）", "--skip-fov" in s90),
        ("普通标定不带 --skip-fov（两个入口行为不同）", "--skip-fov" not in args_plain),
        ("90° 入口与普通标定共用同一套板参数",
         "--cols 8 --rows 5 --square 25 --target 40 --min-pairs 20" in s90),
        ("GUI 有画质诊断入口且走 --together",
         hasattr(app, "_run_camera_quality")
         and "camera_quality.py" in launcher_src and "--together" in launcher_src),
        ("工作台编号与按钮一致（①→⑤，不再写 ⑥）",
         "①→⑤" in launcher_src and "①→⑥" not in launcher_src),
        ("删除/清理只剩数据管理一个入口（工作台改为打开同一窗口）",
         "清理（在数据管理里）" in launcher_src
         and "def _wb_clean" in launcher_src
         and "from apps.cleanup_dialog import" not in launcher_src),
    ]
    for label, good in checks_90:
        print(f"  [{'通过' if good else '不达标'}] {label}")
        results.append((label, good))

    # ---- 8c. 平行装置的「本次不做 FOV 对齐」出口（2026-10-24 加）----
    # 契约：勾上只加 `--skip-fov`，**不许**偷偷把模式改成 angled（那是 90° 入口
    # 的职责）；不勾则普通标定不带这个参数。文本里必须写明后果。
    print("----- 8c. 「本次不做 FOV 对齐」勾选框 -----")
    app.stereo_mode_var.set("auto")
    app.calib_skip_fov.set(False)
    plain_fov = " ".join(app._calibrate_args() or [])
    app.calib_skip_fov.set(True)
    skip_fov_s = " ".join(app._calibrate_args() or [])
    app.calib_skip_fov.set(False)
    checks_skip_fov = [
        ("勾选后 argv 带 --skip-fov", "--skip-fov" in skip_fov_s),
        ("勾选后**不加** --stereo-mode angled（不改模式）",
         "--stereo-mode angled" not in skip_fov_s),
        ("不勾时普通标定不带 --skip-fov", "--skip-fov" not in plain_fov),
        ("界面文案写明后果（原始画面 / 纹理太少）",
         "本次不做 FOV 对齐" in launcher_src
         and "两路各自原始画面" in launcher_src
         and "共同纹理太少" in launcher_src),
        ("标定说明带上『本次不做 FOV 对齐』",
         "本次不做 FOV 对齐" in mod_src),
    ]
    for label, good in checks_skip_fov:
        print(f"  [{'通过' if good else '不达标'}] {label}")
        results.append((label, good))


    off = app._offline_stereo_args(str(ROOT / "data" / "sessions" / "demo"))
    off_s = " ".join(off)
    app.stereo_mode_var.set("angled")
    off_ang = " ".join(app._offline_stereo_args(str(ROOT / "data" / "sessions" / "demo")))
    app.stereo_mode_var.set("auto")
    vid_args = " ".join(app._common_args(for_video=True))
    checks_video = [
        ("离线视频不传 run_video 不认的 --display-width",
         "--display-width" not in vid_args),
        ("离线视频不传 run_video 不认的 --no-3d",
         "--no-3d" not in vid_args),
        ("离线视频仍传 --ball-diameter/--horizon 支持的参数",
         "--ball-diameter" in vid_args and "--horizon" in vid_args),
    ]
    for label, good in checks_video:
        print(f"  [{'通过' if good else '不达标'}] {label}")
        results.append((label, good))
    checks_off = [
        ("调用 apps/run_stereo_video.py", "run_stereo_video.py" in off_s),
        ("带会话目录", "data\\sessions\\demo" in off_s or "data/sessions/demo" in off_s),
        ("带 --method", "--method" in off_s),
        ("显式 angled 时离线也带 --stereo-mode angled", "--stereo-mode angled" in off_ang),
        ("GUI 有「点选球」入口", hasattr(app, "_pick_stereo_seed")
         and "点选球" in (ROOT / "apps/gui_main.py").read_text(encoding="utf-8")),
        ("GUI 方法列表包含 specular", "specular" in mod.METHODS),
    ]
    for label, good in checks_off:
        print(f"  [{'通过' if good else '不达标'}] {label}")
        results.append((label, good))

    # specular + 已有 seeds.json -> 离线参数必须带 --seed-json
    import tempfile
    with tempfile.TemporaryDirectory(prefix="dsh_gui_seed_") as tmp:
        sd = Path(tmp)
        (sd / "derived").mkdir()
        (sd / "session.json").write_text("{}", encoding="utf-8")
        (sd / "derived" / "seeds.json").write_text("{}", encoding="utf-8")
        old = app.method.get()
        app.method.set("specular")
        args_seed = app._offline_stereo_args(str(sd))
        app.method.set(old)
        good = "--seed-json" in args_seed
        print(f"  [{'通过' if good else '不达标'}] specular 且已有 seeds.json 时传 --seed-json")
        results.append(("specular 且已有 seeds.json 时传 --seed-json", good))

    # ---- 9. 可滚动布局（"新加的东西把原来的挤掉"是真实回归）----
    #
    # 加标定参数区之后，第 1 步那张卡片的自然宽度一度到 **1941 px**
    # （整窗 1973 px），而屏幕只有 1600 —— 原有控件被挤到窗口外。
    # 两个原因：卡片顶部与参数下方的说明**没有 wraplength**（按整行文字撑开），
    # 以及日志直接挂在窗口上、用 `expand=True` 吃掉空间。
    print("----- 9. 滚动式布局（不能被内容挤爆 / 不能右边空一片）-----")
    app.update()
    vw0 = app._canvas.winfo_width()
    pw0 = app._page.winfo_reqwidth()
    # “有没有右边空一片”看**实际页面宽度**；reqwidth 是卡片自然宽度，
    # 卡片 pack(fill=x) 后实际会被拉到视口宽，不能拿 reqwidth 当裁切判据。
    page_w, canvas_w = app._page.winfo_width(), vw0
    page_h = app._page.winfo_reqheight()
    canvas_h = app._canvas.winfo_height()
    print(f"  页面需要 {page_w}x{page_h}；滚动区可见 {canvas_w}x{canvas_h}")
    print(f"  视口与页面宽度差 {canvas_w - page_w} px（越小越说明卡片填满了）")
    checks_layout = [
        ("内容宽度不超过可视宽度（**没有横向截断**）", page_w <= canvas_w),
        # 这一条是"右边空一片 / 文字被切"的回归护栏。
        # 起因：`wraplength` 曾写死 820（按"窗口 900"估的），窗口改成 980 之后
        # 卡片自宽 873 而视口 966 —— 文字在卡片那一列里被切掉，右边还空一大片。
        # 现在换行宽度由 `_reflow` **按视口动态算**，所以差距应当很小。
        ("卡片基本填满可用宽度（差距 < 30px，不是「右边空一片」）",
         (canvas_w - page_w) < 30),
        ("内容高于视口 -> 需要竖向滚动", page_h > canvas_h),
        ("需要滚动时滚动条是显示的", bool(app._vbar.winfo_ismapped())),
        # ⚠️ **不要写死卡片数量**：第一版断言"正好 4 张"，加了「任务」卡片之后
        # 就红了 —— 而那是**测试过时**，不是代码错。数量会随功能增长，
        # 这条断言真正要守的是"卡片都挂在滚动页里，而不是挂在窗口上"。
        ("卡片都挂在滚动页里（不是挂在窗口上）",
         len(app._page.winfo_children()) >= 4
         and len(app.winfo_children()) < len(app._page.winfo_children())),
        ("日志挂到滚动页下了（否则页面滚动时它纹丝不动）",
         app.log.master.master is app._page),
    ]

    # ---- 9b. 参数页签里**不许有网格冲突** ----
    #
    # ## 这条守的是什么（真实回归，2026-10-14）
    #
    # 界面截图里"高级设置都是被遮挡的文字"：参数辅助函数内部占
    # `row*2` 与 `row*2+1` **两行**，而调用方传 1、2、3… 并**另手工放**
    # 复选框/分组标题 —— 手算的行号正好落在已被占用的行上，**文字与输入框互相盖住**。
    #
    # 根因是"手算行号"这种做法；现在改成每个 parent 一个游标、辅助函数自己分配。
    # 所以这里钉的是**没有同格重复、没有同列竖向重叠**。
    sections = list(getattr(app, "_calib_sections", ()))
    section_names = [s.header_button.cget("text") for s in sections]
    # 用 >= 2 而不是 == 2（2026-10-26）：加功能常常要在标定面板再加一个区，
    # 那是**正常演进**，不该让这条布局测试变红。真正要钉的是"前两个区是基本/高级，且默认状态正确"。
    two_ok = (len(sections) >= 2
              and "基本设置" in section_names[0]
              and "高级设置" in section_names[1])
    defaults_ok = (two_ok and sections[0].expanded and not sections[1].expanded)
    checks_layout.append((f"参数区前两个是「基本设置 / 高级设置」（共 {len(sections)} 区）", two_ok))
    checks_layout.append(("默认基本展开、高级收起", defaults_ok))
    checks_layout.append(("展开后的设置区有可见边框", all(
        int(s.content.cget("borderwidth") or 0) >= 1
        and str(s.content.cget("relief")) not in ("", "flat")
        for s in sections)))
    checks_layout.append(("窗口完整落在屏幕内（不再固定 980x780）", screen_ok))
    checks_layout.append(("头部标题与右侧操作不重叠", head_ok))
    checks_layout.append(("第 2 步单目参数默认收起且标注作用域",
                          not app._step2_advanced.expanded
                          and "单目" in app.step2_scope.cget("text")
                          and "双目" in app.step2_scope.cget("text")))
    checks_layout.append(("所有注册换行标签都有有效 wraplength（防空串崩溃）",
                          all(int(lab.cget("wraplength") or 0) > 0
                              for lab, _kind in app._wrapped)))
    for section in sections:
        section.expand()
    app.update()
    dup_any, overlap_any, too_wide_any = [], [], []
    for section in sections:
        name = section.header_button.cget("text")
        kids = section.content.winfo_children()
        cells = [(c.grid_info().get("row"), c.grid_info().get("column", 0))
                 for c in kids if c.winfo_manager() == "grid"]
        dup = [k for k, v in Counter(cells).items() if v > 1]
        # ⚠️ 必须**按列分组**再比 y 区间：同一行不同列的控件 y 区间天然重叠，
        # 那不是冲突。第一版没分组，报了一堆假阳性（「TEntry 与 目标对数 重叠」）。
        by_col: dict[int, list] = {}
        for c in kids:
            if c.winfo_manager() != "grid":
                continue
            by_col.setdefault(c.grid_info().get("column", 0), []).append(c)
        ov = []
        for _col, group in by_col.items():
            boxes = sorted((c.winfo_y(), c.winfo_y() + c.winfo_height())
                           for c in group)
            for (y1, y2), (y3, _y4) in zip(boxes, boxes[1:]):
                if y3 < y2 - 1:
                    ov.append((_col, y1, y2, y3))
        widest = max((c.winfo_x() + c.winfo_width() for c in kids), default=0)
        print(f"    {name}: {len(kids)} 个控件  同格重复={len(dup)}  "
              f"同列重叠={len(ov)}  最右={widest}（视口 {app._canvas.winfo_width()}）")
        dup_any += [(name, d) for d in dup]
        overlap_any += [(name, o) for o in ov]
        if widest > app._canvas.winfo_width():
            too_wide_any.append((name, widest))
    checks_layout.append((f"折叠区内没有同格重复（实际 {len(dup_any)} 处）", not dup_any))
    checks_layout.append((f"折叠区内没有同列竖向重叠（实际 {len(overlap_any)} 处）"
                          f"—— 这就是「文字被遮挡」的判据", not overlap_any))
    checks_layout.append((f"折叠区内容不超出视口（实际 {len(too_wide_any)} 处）",
                          not too_wide_any))
    for section in sections:
        section.collapse()
    sections[0].expand()
    app.update()
    # 窗口改宽度后，换行宽度必须跟着变（这才是"动态"的证据）
    wraps: list[tuple[int, int, int]] = []
    for w in (980, 1180, 900):
        app.geometry(f"{w}x780")
        app.update()
        cw = next((int(lab.cget("wraplength")) for lab, k in app._wrapped
                   if k == "card"), 0)
        wraps.append((w, app._canvas.winfo_width(), cw))
    print("  窗口宽 -> 视口宽 / 卡片换行宽度：")
    for w, vw, cw in wraps:
        print(f"    {w:5} -> {vw:5} / {cw:5}")
    grows = wraps[1][2] > wraps[0][2] > wraps[2][2]
    checks_layout.append(
        ("换行宽度随窗口**动态**变化（写死的话这条会红）", grows))
    app.geometry("980x780")
    app.update()

    for name, good in checks_layout:
        print(f"  [{'通过' if good else '不达标'}] {name}")
        results.append((name, good))

    # ---- 10. 日志自动滚动的约定 ----
    print("----- 10. 日志：贴底才自动滚（不许把正在读历史的用户拽走）-----")
    # 前置条件：先把日志滚到底。
    # 启动器初始化会写若干行（环境/设备），刚起来时 yview 不一定已在底部；
    # 2026-10-19 full 档偶发 at_bottom=False（实测 after2=(0.56, 0.77)）——
    # 这是**测试假设**错了，不是自动滚动坏了：用户没贴底时本来就不该被拽走。
    app.log.see("end")
    app.update()
    app._append("第一行")
    app._append("第二行")
    app.update()
    at_bottom = app.log.yview()[1] >= 0.99
    for i in range(60):
        app._append(f"填充 {i}")
    app.update()
    app.log.yview_moveto(0.0)                 # 模拟用户往上翻
    app.update()                              # 让 moveto 真正生效再读 yview
    reading_from = app.log.yview()[1] < 0.99
    before = app.log.yview()[1]
    app._append("读历史时来的新行")
    app.update()
    not_yanked = abs(app.log.yview()[1] - before) < 0.02
    print(f"  贴底时自动滚={at_bottom}；往上翻后是否被拽回底={not not_yanked}")
    checks_log = [
        ("贴底时追加会自动跟到底", at_bottom),
        ("用户往上翻之后，新行不再把他拽到底", reading_from and not_yanked),
    ]
    for name, good in checks_log:
        print(f"  [{'通过' if good else '不达标'}] {name}")
        results.append((name, good))

    panel_target = app.calib_target.get().strip()
    panel_minp = app.calib_minpairs.get().strip()
    panel_fps = app.calib_fps.get().strip()
    app.destroy()

    # ---- 8. 标定脚本自己的默认值也必须是正式档 ----
    #
    # 面板默认 40 但脚本默认 8 的话，**从命令行走的人仍然会停在试标定档** ——
    # 两处必须一致。改任何一处，这条都会红。
    #
    # ⚠️ 判据不能靠 `--help`：argparse 默认**不打印** default（除非用
    # `ArgumentDefaultsHelpFormatter`）。第一版就是这么写的，于是永远拿到
    # "(未打印 default)" 而报红 —— 那是**判据写错了**，不是代码错。
    # 改从源码里读 `add_argument("--target", ..., default=N)`。
    print("----- 8. calibrate_stereo.py / _common.py 的默认值（命令行入口）-----")
    import re as _re
    src = (ROOT / "apps" / "calibrate_stereo.py").read_text(encoding="utf-8")
    # fps 默认值改为**读函数签名**，不再用正则抠源码（2026-10-26）：
    # 默认值已收成 `video_profiles.DEFAULT_FPS` 一处来源，`_common.py` 里写的是常量名，
    # 旧正则 `fps:\s*int\s*=\s*(\d+)` 会直接失配 —— 那是判据写错了，不是代码错。
    import inspect as _inspect
    from apps import _common as _common_mod
    common_fps = "(找不到 fps 默认值)"
    for _fn in vars(_common_mod).values():
        if not _inspect.isfunction(_fn):
            continue
        _p = _inspect.signature(_fn).parameters.get("fps")
        if _p is not None and isinstance(_p.default, int):
            common_fps = str(_p.default)
            break

    def default_of(flag: str) -> str:
        """从 `add_argument("<flag>" ...)` 那一段里抠出 default 值。"""
        m = _re.search(rf'add_argument\(\s*"{_re.escape(flag)}"\s*,(.*?)\)\s*\n',
                       src, _re.S)
        if not m:
            return "(找不到该参数)"
        body = m.group(1)
        m2 = _re.search(r"default\s*=\s*([0-9.]+)", body)
        return m2.group(1) if m2 else "(没有 default)"

    tgt = default_of("--target")
    mp = default_of("--min-pairs")
    print(f"  源码里 --target 的 default = {tgt}；--min-pairs 的 default = {mp}；"
          f"_common.py 的 fps default = {common_fps}")
    print(f"  面板默认：target={panel_target} minpairs={panel_minp} fps={panel_fps}")
    checks = [
        ("--target 默认 40（与面板一致）", tgt == panel_target == "40"),
        ("--min-pairs 默认 20（与面板一致）", mp == panel_minp == "20"),
        ("--fps 默认 24（与面板一致）", common_fps == panel_fps == "24"),
        ("试标定档的文字说明还在（让人知道可以显式给 8）",
         "8" in src and "试标定" in src),
    ]
    for name, good in checks:
        print(f"  [{'通过' if good else '不达标'}] {name}")
        results.append((name, good))

    # ---- 11. 工作台会话摘要（P0 事实层）----
    print("----- 11. 工作台会话摘要（通过 / 坏帧数 / 未点选）-----")
    from apps._workbench import button_states, summarize_session

    def _make_session(base: Path, frames_l: int, frames_r: int,
                      with_seeds: bool, with_traj: bool = False,
                      n_samples: int = 2, n_matched: int = 1,
                      with_csv: bool = False, with_ann: bool = False,
                      verdict: str = "") -> Path:
        # n_samples / n_matched 默认值 == 老夹具的 2 样本 / 1 匹配，
        # 既有调用点（含目录名）行为不变；非默认才把数量缀进目录名。
        name = f"s_{frames_l}_{frames_r}_{int(with_seeds)}_{int(with_traj)}"
        if (n_samples, n_matched) != (2, 1):
            name += f"_{n_samples}_{n_matched}"
        if with_csv:
            name += "_csv"
        if with_ann:
            name += "_ann"
        if verdict:
            name += f"_v{verdict}"
        d = base / name
        (d / "derived").mkdir(parents=True)
        (d / "session.json").write_text(json.dumps({
            "schema": "sphere-stereo-session", "version": 1, "id": d.name,
            "created": "2026-10-20 10:00:00",
            "left": {"frame_count": frames_l, "fps_measured": 30.0},
            "right": {"frame_count": frames_r, "fps_measured": 30.0},
            "sync": {"method": "flash", "offset": 1, "confidence": 0.84},
            "checks": ["平均亮度：left=120.0 right=110.0"],
        }), encoding="utf-8")
        if with_seeds:
            (d / "derived" / "seeds.json").write_text("{}", encoding="utf-8")
        if with_traj:
            n_matched = max(0, min(n_matched, n_samples))
            samples = [
                ({"matched": True, "xyz": [0.1, 0.05, 0.25], "residual_px": 1.5}
                 if i < n_matched else {"matched": False})
                for i in range(n_samples)
            ]
            payload = {
                "schema": "sphere-stereo-trajectory", "version": 1,
                "samples": samples,
            }
            if verdict:
                payload["quality"] = {"verdict": verdict, "note": "测试用"}
            (d / "derived" / "traj3d.json").write_text(json.dumps(payload),
                                                       encoding="utf-8")
        if with_csv:
            # 表头照抄真实老会话（data/stereo/20261019-193809）的格式；
            # matched 在第 3 列，取值 "0"/"1"。
            n_csv_matched = max(0, min(n_matched, n_samples))
            header = ("frame,t,matched,X,Y,Z,u_left,v_left,u_right,v_right,"
                      "epipolar_px,residual_px,confident,method_left,method_right")
            rows = [header]
            for i in range(n_samples):
                t = f"{i / 30.0:.4f}"
                if i < n_csv_matched:
                    rows.append(",".join([
                        str(i), t, "1", "0.1", "0.05", "0.25", "100", "100",
                        "200", "100", "0.2", "1.5", "1", "specular", "specular"]))
                else:
                    rows.append(",".join([
                        str(i), t, "0", "", "", "", "", "", "", "",
                        "0", "0", "0", "none", "none"]))
            (d / "derived" / "traj3d.csv").write_text("\n".join(rows) + "\n",
                                                      encoding="utf-8")
        if with_ann:
            (d / "derived" / "annotated.mp4").write_bytes(b"")
        return d

    with tempfile.TemporaryDirectory(prefix="dsh_wb_") as tmp:
        base = Path(tmp)
        good = summarize_session(_make_session(base, 302, 303, True, True))
        # 2026-11-03 改：小差异（302/304、364/362）**不再算坏** —— 管线只是"取到短的那路结束"。
        # 只有比例上明显掉帧（302/380 ≈ 26%）才判死。
        small = summarize_session(_make_session(base, 364, 362, False, False))
        bad = summarize_session(_make_session(base, 302, 380, False, False))
        rec = summarize_session(_make_session(base, 302, 303, False, False))
        # 2026-11-18 用户报的 bug：seeds.json 是**输入**，不能当 ✓ 的闸门；
        # 匹配点只有一两个也不该谎称"有三维"。下面四条对应这两件事。
        # ① 三维跑好了、没点过球心 —— 仍然是 ✓（改前被判 recorded）
        traj_no_seed = summarize_session(
            _make_session(base, 300, 300, False, True, n_samples=300, n_matched=100))
        # ② 903 帧只匹配 1 个点（0.1%）—— ◑ 三维点过少
        thin_1 = summarize_session(
            _make_session(base, 903, 903, False, True, n_samples=903, n_matched=1))
        # ③ 902 帧只匹配 24 个点（2.7%），有 seeds 也一样，匹配率说了算
        thin_2 = summarize_session(
            _make_session(base, 902, 902, True, True, n_samples=902, n_matched=24))
        # ④ 跑过三维但零匹配 —— 产物在、结果空，退回 ◐ 已点选
        zero_match = summarize_session(
            _make_session(base, 300, 300, False, True, n_samples=300, n_matched=0))
        # 2026-11-18 用户："怎么可能只是这四个有问题" —— 状态不能只认
        # traj3d.json / 只认 seeds.json，要把判据本身改成"按证据强度降级"。
        # ⑤ 老会话只有 traj3d.csv（JSON 是后来的格式）：276/304 必须仍是 ✓
        csv_only = summarize_session(
            _make_session(base, 304, 304, False, False, n_samples=304, n_matched=276,
                          with_csv=True))
        # ⑥ 只有 CSV 且零匹配 —— 产物在、结果空，退回 ◐
        csv_zero = summarize_session(
            _make_session(base, 301, 301, False, False, n_samples=301, n_matched=0,
                          with_csv=True))
        # ⑦ 只剩 annotated.mp4（跑过三维，json/csv 都不在了）—— ◐ 并说明原因
        ann_only = summarize_session(
            _make_session(base, 300, 300, False, False, with_ann=True))
        # ⑧ 0 帧 + 零产物 = 录制没起来，不是数据坏了 → — empty
        empty_s = summarize_session(_make_session(base, 0, 0, False, False))
        # ⑨ 0 帧 + 有 seeds = 元数据与产物矛盾 → 仍是 ✗ bad
        empty_with_seeds = summarize_session(_make_session(base, 0, 0, True, False))
        # ⑩ 质检结论非 ok（几何与标定不一致）：匹配率 96% 也不能报 ✓
        geom_bad = summarize_session(
            _make_session(base, 300, 300, False, True, n_samples=300, n_matched=290,
                          verdict="geometry_mismatch"))
        # ⑪ verdict=ok 时照常按匹配率判 ✓
        geom_ok = summarize_session(
            _make_session(base, 300, 300, False, True, n_samples=300, n_matched=290,
                          verdict="ok"))
        # 目录 mtime 不能决定默认会话：处理旧会话会刷新 mtime。
        from apps._workbench import scan_sessions
        newer = base / "20261021-120000"
        older = base / "20261020-120000"
        for d in (newer, older):
            (d / "derived").mkdir(parents=True)
            # 2026-11-09：会话枚举改成"只认 session.json"（更早的半成品由数据管理
            # 面板用宽判据显示）。排序要验的是**会话**的排序，所以这两个夹具也
            # 得是会话；只建 derived/ 的目录已不再算会话。
            (d / "session.json").write_text(json.dumps({
                "schema": "sphere-stereo-session", "version": 1, "id": d.name,
                "created": "2026-10-21 12:00:00",
                "left": {"frame_count": 10}, "right": {"frame_count": 10},
            }), encoding="utf-8")
        now = time.time()
        os.utime(newer, (now - 1000, now - 1000))   # 故意让“更新”的会话 mtime 更旧
        os.utime(older, (now, now))
        scan_order = [s.id for s in scan_sessions(base, limit=2)]
        checks_wb = [
            ("好会话 -> traj，n_matched=1，Z=0.25",
             good.status == "traj" and good.n_matched == 1 and abs((good.z_min or 0) - 0.25) < 1e-9),
            ("帧数 302/380（差 26%）-> bad，并给出原因",
             bad.status == "bad" and any("明显掉帧" in p for p in bad.problems)),
            ("帧数 364/362（差 2 帧）-> **不算坏**，只提醒可继续",
             small.status != "bad" and any("可继续" in p for p in small.problems)),
            ("未点选 -> recorded，按钮层能据此禁用三维",
             rec.status == "recorded" and not rec.can_view),
            ("坏会话在 specular 下禁用 seed/run/view",
             not button_states(bad, True)["seed"]
             and not button_states(bad, True)["run"]
             and not button_states(bad, True)["view"]),
            ("成对导出：有轨迹会话产出 raw+motion、raw 逐字节等于源",
             _pair_check(base)),
            ("好会话在 specular 下启用 seed/run/view",
             button_states(good, True)["seed"]
             and button_states(good, True)["run"]
             and button_states(good, True)["view"]),
            ("校验失败时禁用 seed/run",
             not button_states(good, True, False)["seed"]
             and not button_states(good, True, False)["run"]),
            ("文件菜单按钮有对应状态", button_states(good, True)["files"]),
            ("默认会话按时间名排序，不按 mtime", scan_order[:1] == ["20261021-120000"]),
            # 2026-11-18 用户 bug：seeds.json 是输入，不是 ✓ 的必要条件。
            # 这条在改前必红（改前 traj_no_seed 会被判 recorded）。
            ("三维跑好但没点过球心 -> 仍是 traj ✓",
             traj_no_seed.status == "traj" and traj_no_seed.symbol == "✓"),
            ("903 帧仅 1 个匹配点（0.1%）-> traj_thin ◑，并说明匹配点过少",
             thin_1.status == "traj_thin" and thin_1.symbol == "◑"
             and any("匹配点过少" in p for p in thin_1.problems)),
            ("902 帧仅 24 个匹配点（2.7%）即使有 seeds 也判 traj_thin",
             thin_2.status == "traj_thin" and thin_2.symbol == "◑"),
            ("跑过三维但零匹配 -> seeded ◐（产物在、结果空）",
             zero_match.status == "seeded" and zero_match.symbol == "◐"),
            # 判据本身：三维产物三种证据（json > csv > 可视化/标注视频）逐级降级，
            # 不拿某个特定文件名当唯一闸门。
            ("只有 traj3d.csv 的老会话 276/304 -> traj ✓（改前误判 recorded ○）",
             csv_only.status == "traj" and csv_only.symbol == "✓"
             and csv_only.n_matched == 276 and csv_only.n_samples == 304),
            ("只有 traj3d.csv 且零匹配 -> seeded ◐",
             csv_zero.status == "seeded" and csv_zero.symbol == "◐"),
            ("只有 annotated.mp4 -> seeded ◐，且说明缺 json/csv",
             ann_only.status == "seeded"
             and any("缺 traj3d.json" in p for p in ann_only.problems)),
            ("0 帧 + 零产物 = 没录成 -> empty —（改前误判 bad ✗）",
             empty_s.status == "empty" and empty_s.symbol == "—"),
            ("0 帧但有 seeds -> 仍是 bad ✗（有产物却读不到帧数）",
             empty_with_seeds.status == "bad"),
            ("verdict=geometry_mismatch 时 290/300（96%）也判 traj_thin ◑ 并写明原因",
             geom_bad.status == "traj_thin"
             and any("geometry_mismatch" in p for p in geom_bad.problems)),
            ("verdict=ok 时照常判 traj ✓（不得因有 quality 键而误降）",
             geom_ok.status == "traj"),
            ("空壳会话禁用 seed/run/view（和坏数据一样，别让它白点）",
             not button_states(empty_s, True)["seed"]
             and not button_states(empty_s, True)["run"]
             and not button_states(empty_s, True)["view"]),
            ("既有 good 会话仍是 traj（防误伤）",
             good.status == "traj" and good.symbol == "✓"),
        ]
        for name, passed in checks_wb:
            print(f"  [{'通过' if passed else '不达标'}] {name}")
            results.append((name, passed))

    # ---- 12. Task.on_done：完成回调恰好一次（P2 基础）----
    print("----- 12. 任务完成回调（P2）-----")
    hits: list[int] = []
    app2 = mod.Launcher()
    app2.withdraw()
    task = app2.start_task(
        [sys.executable, "-c", "print('on_done_ok')"], "on_done-selftest",
        on_done=lambda t: hits.append(t.tid))
    deadline = time.time() + 8.0
    while time.time() < deadline and not hits:
        app2.update()
        time.sleep(0.03)
    checks_cb = [
        ("任务完成后 on_done 恰好调用一次",
         hits == [task.tid] and not task.running),
    ]
    for name, passed in checks_cb:
        print(f"  [{'通过' if passed else '不达标'}] {name}")
        results.append((name, passed))

    with tempfile.TemporaryDirectory(prefix="dsh_any_session_") as tmp:
        p = Path(tmp) / "20260101-120000"
        p.mkdir()
        (p / "session.json").write_text(json.dumps({
            "schema": "sphere-stereo-session", "version": 1, "id": p.name,
            "created": "2026-01-01 12:00:00",
            "left": {"frame_count": 1}, "right": {"frame_count": 1},
        }), encoding="utf-8")
        app2._wb_add_custom(p)
        app2.update()
        selected = app2._wb_selected()
        any_ok = (selected is not None and selected.path.resolve() == p.resolve()
                  and "[任意]" in app2.wb_var.get())
        print(f"  [{'通过' if any_ok else '不达标'}] 工作台可加入任意会话目录")
        results.append(("工作台可加入任意会话目录", any_ok))
    app2.destroy()

    # ---- 13. 数据导出：脱敏 + 类别 + zip ----
    print("----- 13. 数据管理导出 -----")
    from apps._data_export import (
        DEFAULT_CATEGORIES, ExportOptions, estimate_bytes, export_zip,
        normalize_output, redact_session, scan_items,
    )
    from apps.data_manager import DataManager
    with tempfile.TemporaryDirectory(prefix="dsh_export_") as tmp:
        sroot = Path(tmp) / "sessions"
        sd = sroot / "20260101-120000"
        (sd / "derived").mkdir(parents=True)
        (sd / "calib").mkdir()
        sensitive = {
            "schema": "sphere-stereo-session", "version": 1, "id": sd.name,
            "created": "2026-01-01 12:00:00", "label": "secret-label",
            "notes": "private notes",
            "stereo_mode": "angled",
            "rig": {"stereo_mode": "angled", "angle_deg": 90.0},
            "left": {"frame_count": 3, "identity": "USB\\SECRET", "name": "cam-left",
                     "port": "p1"},
            "right": {"frame_count": 3, "identity": "USB\\SECRET2", "name": "cam-right",
                      "port": "p2"},
            "environment": {"platform": "Windows-private", "python": "3.14"},
        }
        (sd / "session.json").write_text(json.dumps(sensitive), encoding="utf-8")
        (sd / "left_times.csv").write_text("frame,perf_counter_s\n0,1\n", encoding="utf-8")
        (sd / "derived" / "traj3d.json").write_text('{"samples": []}', encoding="utf-8")
        (sd / "derived" / "traj3d.png").write_bytes(b"png")
        (sd / "left.mkv").write_bytes(b"raw")
        items = scan_items(sroot)
        from apps._workbench import summarize_session as _summarize
        status_same = all(it.status == _summarize(it.path).status
                          and it.symbol == _summarize(it.path).symbol
                          for it in items)
        opts = ExportOptions(categories=frozenset({"metadata", "derived"}))
        red = redact_session(sensitive, opts)
        out_zip = Path(tmp) / "export.zip"
        res = export_zip(items, opts, out_zip)
        with zipfile.ZipFile(out_zip) as zf:
            names = zf.namelist()
            sj = [n for n in names if n.endswith("metadata/session.json")][0]
            red_zip = json.loads(zf.read(sj))
            has_raw = any("/videos/raw/" in n for n in names)
            has_manifest = any(n.endswith("/manifest.json") for n in names)

        # 默认类别 + manifest 契约
        default_opts = ExportOptions()
        default_zip = export_zip(items, default_opts, Path(tmp) / "default.zip")
        with zipfile.ZipFile(default_zip.output) as zf:
            default_names = zf.namelist()
            default_manifest = json.loads(
                zf.read([n for n in default_names if n.endswith("/manifest.json")][0]))
        default_has_raw = any("/videos/raw/" in n for n in default_names)
        default_has_annotated = any("/videos/annotated" in n for n in default_names)
        root = default_names[0].split("/")[0]
        all_under_root = all(n.startswith(root + "/") for n in default_names)
        paths = [n for n in default_names if not n.endswith("/")]
        no_abs = not any((":" in p or p.startswith("/")) for p in paths)
        unique = len(paths) == len(set(paths))
        manifest_paths = ([default_manifest.get("root", "")]
                          + [f.get("path", "") for f in default_manifest.get("files", [])]
                          + [s.get("folder", "") for s in default_manifest.get("sessions", [])]
                          + [s.get("id", "") for s in default_manifest.get("sessions", [])])
        no_abs_manifest = not any(
            re.match(r"^[A-Za-z]:[\\/]", p) or p.startswith("\\\\") or p.startswith("/")
            for p in manifest_paths if p)
        manifest_ok = (
            default_manifest.get("n_sessions") == len(items)
            and default_manifest.get("root") == root
            and set(default_manifest.get("categories") or []) == set(DEFAULT_CATEGORIES)
            and default_manifest.get("redaction", {}).get("camera_identity") is True
            and any(s.get("stereo_mode") == "angled"
                    for s in default_manifest.get("sessions", []))
            and no_abs_manifest
        )
        checks_data = [
            ("脱敏移除相机身份/名称/端口",
             all(k not in red["left"] for k in ("identity", "name", "port"))
             and all(k not in red["right"] for k in ("identity", "name", "port"))),
            ("脱敏移除 label/notes/platform",
             "label" not in red and "notes" not in red
             and "platform" not in red.get("environment", {})),
            ("默认类别不含视频", not default_has_raw and not default_has_annotated),
            ("zip 内 session.json 是脱敏后的",
             "identity" not in red_zip["left"] and red_zip.get("label") is None),
            ("zip 有 manifest.json", has_manifest),
            ("manifest/目录：单根、无绝对路径、无重复",
             all_under_root and no_abs and no_abs_manifest and unique and manifest_ok),
            ("n_files 与实际 zip 条目一致",
             default_zip.n_files == len(default_names)),
            ("estimate_bytes 非零", estimate_bytes(items, default_opts) > 0),
            ("导出结果有文件", default_zip.n_files >= 3 and default_zip.output.exists()),
        ]
        for name, passed in checks_data:
            print(f"  [{'通过' if passed else '不达标'}] {name}")
            results.append((name, passed))

        # 全部类别 + 项目自己写的 _first / brightness 文件
        (sd / "derived" / "traj3d_first1.csv").write_text("x", encoding="utf-8")
        (sd / "derived" / "traj3d_first1.png").write_bytes(b"png")
        (sd / "derived" / "annotated_first1.mp4").write_bytes(b"mp4")
        (sd / "derived" / "left_brightness.npy").write_bytes(b"npy")
        all_opts = ExportOptions(categories=frozenset({
            "metadata", "derived", "plots", "annotated", "raw", "calibration"}))
        all_zip = export_zip(items, all_opts, Path(tmp) / "all.zip")
        with zipfile.ZipFile(all_zip.output) as zf:
            all_names = zf.namelist()
        checks_extra = [
            ("_first/_brightness/annotated 变体进入包",
             any("traj3d_first1.csv" in n for n in all_names)
             and any("traj3d_first1.png" in n for n in all_names)
             and any("annotated_first1.mp4" in n for n in all_names)
             and any("left_brightness.npy" in n for n in all_names)),
        ]

        # 坏/缺 metadata 必须报错，不能静默消失
        bad_dir = sroot / "20260103-120000"
        bad_dir.mkdir()
        (bad_dir / "session.json").write_text("{", encoding="utf-8")
        bad_item = [it for it in scan_items(sroot) if it.key == bad_dir.name][0]
        non_dict_dir = sroot / "20260104-120000"
        non_dict_dir.mkdir()
        (non_dict_dir / "session.json").write_text("[]", encoding="utf-8")
        non_dict_item = [it for it in scan_items(sroot) if it.key == non_dict_dir.name][0]
        try:
            export_zip([non_dict_item], ExportOptions(categories=frozenset({"metadata"})),
                       Path(tmp) / "non_dict.zip")
            non_dict_raises = False
        except ValueError:
            non_dict_raises = True
        try:
            export_zip([bad_item], ExportOptions(categories=frozenset({"metadata"})),
                       Path(tmp) / "bad.zip")
            bad_meta_raises = False
        except ValueError:
            bad_meta_raises = True
        try:
            export_zip(items, opts, out_zip)
            overwrite_raises = False
        except FileExistsError:
            overwrite_raises = True
        try:
            normalize_output("")
            empty_raises = False
        except ValueError:
            empty_raises = True
        off = redact_session(sensitive, ExportOptions(
            redact_camera=False, redact_labels=False, redact_platform=False))
        checks_extra += [
            ("坏 metadata 明确报错", bad_meta_raises),
            ("非对象 JSON 标为坏会话且导出报错",
             non_dict_item.status == "bad" and non_dict_item.symbol == "✗"
             and non_dict_raises),
            ("数据管理与工作台状态/符号一致", status_same),
            ("重复输出报 FileExistsError", overwrite_raises),
            ("空输出路径报 ValueError", empty_raises),
            ("脱敏关闭时不谎称已移除",
             "identity" in off["left"] and "已移除" not in off["redacted"]["note"]),
        ]
        cancel_zip = Path(tmp) / "cancel.zip"
        try:
            export_zip(
                items, opts, cancel_zip,
                progress=lambda _d, _t, _n: (_ for _ in ()).throw(RuntimeError("cancel")))
            cancel_ok = False
        except RuntimeError:
            cancel_ok = not cancel_zip.exists() and not (cancel_zip.with_name(
                cancel_zip.name + ".part").exists())
        checks_extra.append(("取消/失败不留半成品", cancel_ok))

        # GUI 选择：起点→终点区间；全选/反选
        try:
            dm = DataManager(sroot)
            dm.withdraw()
            dm.update()
            rows = dm.tree.get_children()
            dm.tree.focus(rows[0])
            dm.set_anchor()
            dm.tree.focus(rows[-1])
            dm.select_to_here()
            range_ok = len(dm.selected) == len(rows)
            dm.select_none()
            dm.select_all()
            all_ok = len(dm.selected) == len(rows)
            dm.invert()
            invert_ok = len(dm.selected) == 0
            dm.select_none()
            dm._last_focus = rows[0]
            dm._anchor_id = None
            dm._select_range(rows[0], rows[-1])
            shift_ok = len(dm.selected) == len(rows)
            dm.select_none()
            warn_before = set(dm.selected)
            import apps.data_manager as dmmod
            old_warn = dmmod.messagebox.showwarning
            warnings: list = []
            dmmod.messagebox.showwarning = lambda *a, **k: warnings.append(a)
            try:
                dm.from_var.set("bogus")
                dm.to_var.set("2026-01-02")
                dm.select_time()
            finally:
                dmmod.messagebox.showwarning = old_warn
            time_bad_ok = set(dm.selected) == warn_before and bool(warnings)

            old_ask = dmmod.messagebox.askyesno
            old_popen = dmmod.subprocess.Popen
            calls: list[list[str]] = []
            dmmod.messagebox.askyesno = lambda *a, **k: True
            dmmod.subprocess.Popen = lambda args: calls.append(args)
            try:
                dm.tree.focus(sd.name)
                dm._open_viewer()
            finally:
                dmmod.messagebox.askyesno = old_ask
                dmmod.subprocess.Popen = old_popen
            viewer_ok = (bool(calls) and "view_3d.py" in calls[0][1]
                         and "--no-video" in calls[0])
            dm.destroy()
            gui_ok = (range_ok and all_ok and invert_ok and shift_ok
                      and time_bad_ok and viewer_ok)
        except Exception as exc:                            # noqa: BLE001
            print(f"    DataManager 选择测试失败：{exc}")
            gui_ok = False
        checks_extra.append(("GUI 起点→终点/全选/反选/查看3D", gui_ok))

        # 筛选 + 搜索（2026-11-03）：筛选只改显示不动勾选；全选只作用于可见项；
        # 搜索匹配会话 id 或名称 label。
        try:
            dm2 = DataManager(sroot)
            dm2.withdraw(); dm2.update()
            # 造一个带 label 的会话，验证搜索能按名称匹配
            lbl_dir = sroot / "a_lbl_1"
            (lbl_dir / "derived").mkdir(parents=True)
            (lbl_dir / "session.json").write_text(json.dumps({
                "schema": "sphere-stereo-session", "version": 1, "id": "lbl-1",
                "label": "金属球-左路虚", "created": "2026-01-03 12:00:00",
                "left": {"frame_count": 1}, "right": {"frame_count": 1},
            }), encoding="utf-8")
            dm2.refresh()
            total = len(dm2.items)
            # 先勾一个
            dm2.selected = {dm2.items[0].key}
            dm2._refresh_count()
            kept = set(dm2.selected)
            # 搜索按名称匹配
            dm2.search_var.set("金属球")
            dm2.update()
            vis = dm2._visible_items()
            search_ok = (len(vis) == 1 and vis[0].key == "a_lbl_1"
                         and dm2.selected == kept)   # 搜索不动勾选
            # 清空搜索，选状态下拉=○ 已录制
            dm2.search_var.set("")
            dm2.status_filter_var.set("○ 已录制")
            dm2.update()
            vis_rec = dm2._visible_items()
            filt_ok = all(it.status == "recorded" for it in vis_rec) and dm2.selected == kept
            # 全选只作用于可见项（此时可见=recorded 的那批），不把隐藏的也勾上
            dm2.select_none()
            dm2.select_all()
            all_visible_ok = dm2.selected == {it.key for it in vis_rec}
            dm2.destroy()
            filter_ok = search_ok and filt_ok and all_visible_ok
        except Exception as exc:                            # noqa: BLE001
            print(f"    DataManager 筛选/搜索测试失败：{exc}")
            filter_ok = False
        checks_extra.append(("GUI 筛选/搜索：只改显示不动勾选、全选只看可见", filter_ok))

        # 日期下拉有值 + 导出 README 含数据变量说明（2026-11-03）
        try:
            dm3 = DataManager(sroot); dm3.withdraw(); dm3.update(); dm3.refresh()
            date_ok = len(dm3.from_box.cget("values")) > 0
            dm3.destroy()
        except Exception as exc:                            # noqa: BLE001
            date_ok = False
        from apps._data_export import (
            ExportOptions, SessionItem, _readme as _dm_readme)
        _probe_it = SessionItem(path=sd, id="x", created="2026-01-02 12:00:00",
                                frames_left=1, frames_right=1, n_matched=0,
                                status="traj", size_bytes=1)
        readme_txt = _dm_readme([_probe_it], ExportOptions())
        readme_ok = ("数据变量说明" in readme_txt and "traj3d.csv" in readme_txt
                     and "segment" in readme_txt and "左相机系" in readme_txt)
        checks_extra.append(("日期下拉有值 + 导出 README 含数据变量说明",
                             date_ok and readme_ok))

        # 按钮文字（2026-11-03）：开始导出标注 zip、轨迹导出标注 csv、右上角合并成一个刷新按钮
        try:
            dm4 = DataManager(sroot); dm4.withdraw(); dm4.update()
            btn_ok = (dm4.export_btn.cget("text") == "开始导出 (zip)"
                      and dm4.export_pair_btn.cget("text") == "导出轨迹数据 (csv)"
                      and dm4.head_refresh.cget("text") == "刷新数据目录")
            dm4.destroy()
        except Exception as exc:                            # noqa: BLE001
            btn_ok = False
        checks_extra.append(("按钮文字：开始导出(zip)/导出轨迹数据(csv)/刷新数据目录",
                             btn_ok))

        # 分区可拖（2026-11-03）：每个 PanedWindow 至少 2 个 pane，且 sash 位置可写
        # （sashpos 设置后值变化 = 拖动机制在工作）。
        # ⚠️ 2026-11-09 改判据：原来写死 `len(_panes) == 2`（vertical + horizontal），
        # 数据管理左栏拆成「目录树 | 会话表」后变成 3 个 PanedWindow，于是这条**假红**。
        # 真正的判据是 AGENTS.md §7.2「分区必须可以拖动」—— 数量是当时的快照，不是约束。
        # 静态覆盖在 `selftest_window_mode.py` §12（它逐个窗口钉死 PanedWindow）。
        try:
            dm5 = DataManager(sroot)   # 不 withdraw：sashpos 依赖几何映射，withdraw 后恒为 0
            for _ in range(3):
                dm5.update_idletasks(); dm5.update()
            from tkinter import ttk as _ttk
            _panes = []
            def _walk(w):
                for _c in w.winfo_children():
                    if isinstance(_c, _ttk.PanedWindow): _panes.append(_c)
                    _walk(_c)
            _walk(dm5)
            _ok_struct = (len(_panes) >= 2
                          and all(len(_p.panes()) >= 2 for _p in _panes))
            _ok_drag = False
            if _ok_struct:
                _ok_drag = True
                for _p in _panes:
                    _b = _p.sashpos(0)
                    try:
                        _p.sashpos(0, max(_b + 30, 30))
                        _ok_drag = _ok_drag and (_p.sashpos(0) != _b)
                    except Exception:                        # noqa: BLE001
                        _ok_drag = False
            dm5.destroy()
            pane_ok = _ok_struct and _ok_drag
        except Exception as exc:                            # noqa: BLE001
            pane_ok = False
        checks_extra.append(("分区可拖：每个 PanedWindow ≥2 pane 且 sash 可写", pane_ok))

        # 三层结构（2026-11-03）：exports/<detailed|trajectory>/<时间戳文件夹>/…
        from apps.data_manager import DataManager as _DM
        _stub = _DM.__new__(_DM)   # 不跑 __init__（不开 Tk），只测两个纯方法
        _z = Path(_DM._default_output(_stub))
        _t = Path(_DM._default_traj_output(_stub))
        checks_extra.append(
            ("导出目录三层结构：exports/detailed/<时间戳>/zip 与 exports/trajectory/<时间戳>/",
             _z.parent.parent.name == "detailed" and _z.suffix == ".zip"
             and _z.parent.name.startswith("export-")
             and _t.parent.name == "trajectory"
             and _t.name.startswith("trajectory-")))

        for name, passed in checks_extra:
            print(f"  [{'通过' if passed else '不达标'}] {name}")
            results.append((name, passed))

        # 重复 session id + 清洗后同名目录：不能崩、不能互相覆盖
        for dn, sid in (("a_dup_1", "same-id"), ("a_dup_2", "same-id")):
            d = sroot / dn
            (d / "derived").mkdir(parents=True)
            (d / "session.json").write_text(json.dumps({
                "schema": "sphere-stereo-session", "version": 1, "id": sid,
                "created": "2026-01-02 12:00:00",
                "left": {"frame_count": 1}, "right": {"frame_count": 1},
            }), encoding="utf-8")
        dup_items = [it for it in scan_items(sroot) if it.key.startswith("a_dup_")]
        dup_zip = export_zip(dup_items, ExportOptions(categories=frozenset({"metadata"})),
                             Path(tmp) / "dup")
        with zipfile.ZipFile(dup_zip.output) as zf:
            dup_names = zf.namelist()
        folders = {n.split("/sessions/")[1].split("/")[0]
                   for n in dup_names if "/sessions/" in n}
        try:
            dm = DataManager(sroot)
            dm.withdraw()
            dm.update()
            n_rows = len(dm.tree.get_children())
            dm.destroy()
            dm_ok = n_rows == len([p for p in sroot.iterdir() if p.is_dir()])
        except Exception as exc:                            # noqa: BLE001
            print(f"    DataManager 构造失败：{exc}")
            dm_ok = False
        checks_dup = [
            ("重复 id 的会话在 zip 里各有独立目录", len(folders) == 2),
            ("无后缀输出自动补 .zip", str(dup_zip.output).endswith(".zip")),
            ("DataManager 能打开重复 id 会话", dm_ok),
        ]
        for name, passed in checks_dup:
            print(f"  [{'通过' if passed else '不达标'}] {name}")
            results.append((name, passed))

        # 视频隐私确认（2026-11-11）：默认按钮落在「否」，取消分支原来裸
        # return —— 按钮不变、status 停在「等待选择」、日志一行不写，用户以为
        # 「开始导出」坏了。这里钉住：安全默认不许改成 YES、取消要有可见反馈、
        # 正向别被护栏堵死。输出路径一律指到本 tmp，不碰 _default_output()。
        from unittest import mock
        from apps._data_export import ExportResult as _ExportResult
        import apps.data_manager as _dmmod
        privacy_checks: list[tuple[str, bool]] = []
        cancel_out = Path(tmp) / "zz_privacy_cancel.zip"
        try:
            dmc = DataManager(sroot)
            dmc.withdraw()
            dmc.update()
            dmc.out_var.set(str(cancel_out))
            dmc._cat_vars["annotated"].set(True)
            dmc._cat_vars["raw"].set(True)
            dmc.select_all()
            dmc._refresh_count()
            with mock.patch.object(_dmmod.messagebox, "askyesno",
                                   return_value=False) as ask_no:
                dmc.export_btn.invoke()
                call = ask_no.call_args
                title = call.args[0] if call else ""
                message = call.args[1] if call and len(call.args) > 1 else ""
                default = call.kwargs.get("default") if call else None
            privacy_checks += [
                ("隐私确认框被调用过（勾了视频）", bool(call)),
                ("确认框仍是安全默认 default=NO",
                 default == _dmmod.messagebox.NO),
                ("确认框标题醒目、正文含「选『是』」与勾选的类别名",
                 title == "包含视频，请确认"
                 and "选『是』" in message
                 and "标注视频" in message and "原始双目视频" in message),
            ]
            status_txt = dmc.status.cget("text")
            log_txt = dmc.log.get("1.0", "end")
            privacy_checks += [
                ("取消有反馈：status 含「已取消」、导出日志含「取消」",
                 "已取消" in status_txt and "取消" in log_txt),
                ("取消不生成 zip、_running 为 False、按钮仍 normal",
                 not cancel_out.exists()
                 and not cancel_out.with_name(
                     cancel_out.name + ".part").exists()
                 and dmc._running is False
                 and str(dmc.export_btn.cget("state")) == "normal"),
            ]
            dmc._stop_polling()
            dmc.destroy()
        except Exception as exc:                            # noqa: BLE001
            print(f"    DataManager 隐私确认测试失败：{exc}")
            privacy_checks.append(("隐私确认测试整体不抛异常", False))

        # 正向：打桩返回 True，导出真的启动（状态变「导出中」），别把正常路径堵死。
        # export_zip 也打桩，避免真写盘 / 线程残留；join 放在 patch 生效期内，
        # 防止线程调度到 with 外时调到真实函数。
        pos_out = Path(tmp) / "zz_privacy_positive.zip"
        try:
            dmp = DataManager(sroot)
            dmp.withdraw()
            dmp.update()
            dmp.out_var.set(str(pos_out))
            dmp._cat_vars["annotated"].set(True)
            dmp.select_all()
            with mock.patch.object(_dmmod.messagebox, "askyesno",
                                   return_value=True):
                with mock.patch.object(
                        _dmmod, "export_zip",
                        return_value=_ExportResult(
                            output=pos_out, n_sessions=1,
                            n_files=1, bytes_written=0)):
                    dmp.export_btn.invoke()
                    started = ("导出中" in dmp.status.cget("text")
                               and dmp._running is True)
                    if dmp._worker is not None:
                        dmp._worker.join(timeout=2)
            privacy_checks.append((
                "正向（选『是』）导出真的启动、状态变「导出中」", started))
            dmp._stop_polling()
            dmp.destroy()
        except Exception as exc:                            # noqa: BLE001
            print(f"    DataManager 隐私确认正向测试失败：{exc}")
            privacy_checks.append(("隐私确认正向测试整体不抛异常", False))

        # 2026-11-11 真机实测：DataManager 的业务方法 `_options()` 与
        # `tk.Misc._options(self, cnf, kw=None)` 撞名。`commondialog.Dialog.show()`
        # 会执行 `master._options(self.options)`，而 master 就是 `parent=self` ——
        # 于是所有 messagebox / filedialog 都抛 TypeError，弹窗根本出不来。
        # 上面两组测试把 messagebox 打桩了，恰好绕开了这条路径；这里**不打桩**，
        # 只替换 dm.tk（Tcl 调用端），让 show() 真的跑到那一行。
        try:
            import tkinter.filedialog as _fd
            import tkinter.messagebox as _mb

            class _FakeTk:
                def __init__(self, ret: str) -> None:
                    self.ret = ret

                def call(self, *_args) -> str:
                    return self.ret

                def getboolean(self, _v) -> bool:
                    return True

            dmo = DataManager(sroot)
            dmo.withdraw()
            dmo.update()
            try:
                with mock.patch.object(dmo, "tk", _FakeTk("yes")):
                    _mb.askyesno("包含视频，请确认", "测试", parent=dmo,
                                 default=_mb.NO)
                msgbox_ok = True
            except TypeError:
                msgbox_ok = False
            except Exception:                               # noqa: BLE001
                msgbox_ok = False
            try:
                with mock.patch.object(dmo, "tk", _FakeTk("")):
                    _fd.askdirectory(parent=dmo, title="测试")
                filedialog_ok = True
            except TypeError:
                filedialog_ok = False
            except Exception:                               # noqa: BLE001
                filedialog_ok = False
            # 单独求值：改名回 `_options` 时这一项也会红，但不能让它把上面
            # 两条 TypeError 判据一起吞掉（反向自证要看清红的是哪条）。
            try:
                opts_ok = "metadata" in dmo._export_options().categories
            except AttributeError:
                opts_ok = False
            privacy_checks += [
                ("DataManager 不再遮蔽 tk.Misc._options（业务方法已改名）",
                 "_options" not in DataManager.__dict__),
                ("真实 messagebox(parent=self) 不再抛 TypeError", msgbox_ok),
                ("真实 filedialog(parent=self) 不再抛 TypeError", filedialog_ok),
                ("导出选项方法 _export_options() 仍可用", opts_ok),
            ]
            dmo._stop_polling()
            dmo.destroy()
        except Exception as exc:                            # noqa: BLE001
            print(f"    Tk _options 冲突测试失败：{exc}")
            privacy_checks.append(("Tk _options 冲突测试整体不抛异常", False))

        for name, passed in privacy_checks:
            print(f"  [{'通过' if passed else '不达标'}] {name}")
            results.append((name, passed))

    # ---- 14. 清理/回收站：软删除、恢复、彻底删除、路径安全 ----
    print("----- 14. 清理与回收站 -----")
    from apps._cleanup import (
        list_trash, move_to_trash, plan_cleanup, purge_batch, restore_batch,
    )
    with tempfile.TemporaryDirectory(prefix="dsh_cleanup_") as tmp:
        root = Path(tmp) / "sessions"
        for i in range(1, 4):
            d = root / f"2026010{i}-120000"
            (d / "derived").mkdir(parents=True)
            (d / "session.json").write_text("{}", encoding="utf-8")
            (d / "left.mkv").write_bytes(b"raw")
            (d / "derived" / "traj3d.json").write_text("{}", encoding="utf-8")
        s1 = root / "20260101-120000"
        plan = plan_cleanup(root, [s1], mode="derived")
        batch = move_to_trash(root, plan)
        derived_gone = not (s1 / "derived").exists() and (s1 / "left.mkv").exists()
        restored = restore_batch(root, batch)
        derived_back = (s1 / "derived").exists()
        plan_all = plan_cleanup(root, sorted(root.iterdir()), mode="session",
                                keep_latest=1)
        keep_ok = len(plan_all.items) == 2 and len(plan_all.skipped) == 1
        outside_ok = False
        try:
            plan_cleanup(root, [Path(tmp) / "outside"], mode="session")
        except ValueError:
            outside_ok = True
        s3 = root / "20260103-120000"
        b3 = move_to_trash(root, plan_cleanup(root, [s3], mode="session"))
        purge_batch(root, b3)
        purge_ok = not s3.exists() and not any(
            b.path.name == b3.name for b in list_trash(root))
        checks_cleanup = [
            ("只清 derived 不动原始视频", derived_gone),
            ("回收站恢复 derived", restored == 1 and derived_back),
            ("保留最新 N 个", keep_ok),
            ("拒绝 sessions 根目录之外的目标", outside_ok),
            ("彻底删除只发生在回收站批次", purge_ok),
        ]
        for name, passed in checks_cleanup:
            print(f"  [{'通过' if passed else '不达标'}] {name}")
            results.append((name, passed))

    # ---- 15. 相机角色解析：采集索引 ≠ PnP 序号（2026-10-23 修的真实误报）----
    #
    # 真实缺陷：`Web Camera Left` 的**采集索引**是 2，而它的 **PnP 序号是 0**。
    # 旧代码拿采集 #2 去 `by_order` 里查 PnP 设备，命中的是 PnP #2
    # `Integrated IR Camera`，于是**外接相机被误报成「内置/红外」**，
    # 用户差点去换一台好相机。这里钉两层：有 entry 时**必须**用 entry；
    # 只有 entry 缺失（索引兜底 / 手动编号）时才按 index 查（旧行为）。
    print("----- 15. 相机角色解析：采集索引 ≠ PnP 序号（不许混用）-----")
    from sphere_tracker.camera_registry import DeviceEntry, ResolvedCamera

    pnp_left = DeviceEntry(order=0, name="Web Camera Left")
    pnp_ir = DeviceEntry(order=2, name="Integrated IR Camera")
    by_order = {pnp_left.order: pnp_left, pnp_ir.order: pnp_ir}
    resolved_real = ResolvedCamera(role="left", index=2, how="identity",
                                   entry=pnp_left)
    got_real = mod._resolved_entry(resolved_real, by_order)
    resolved_fallback = ResolvedCamera(role="left", index=2, how="index-fallback")
    got_fallback = mod._resolved_entry(resolved_fallback, {2: pnp_left})
    checks_entry = [
        ("采集 #2 且带 entry -> 返回 entry（不是 PnP #2）",
         got_real is pnp_left),
        ("名字取 entry.name（Web Camera Left，不是 Integrated IR Camera）",
         (got_real.name if got_real else "") == "Web Camera Left"
         and "Integrated" not in (got_real.name if got_real else "")),
        ("entry 缺失 -> 按 index 兜底（旧行为仍可用）",
         got_fallback is pnp_left),
        ("_confirm_stereo_cameras 已改走 helper（不再按采集索引查 PnP）",
         "by_order.get(r.index)" not in launcher_src
         and launcher_src.count("_resolved_entry(r, by_order)") >= 2),
    ]
    for name, passed in checks_entry:
        print(f"  [{'通过' if passed else '不达标'}] {name}")
        results.append((name, passed))

    print("----- 16. 容器帧率取相机协商值（--fps 只是请求提示）-----")
    from apps.record_stereo import _container_fps
    checks_container = [
        ("协商 30 + 请求 24 -> 30（容器头不许照抄请求值）",
         _container_fps(30.0, 24) == 30.0),
        ("协商 0 -> 退回请求 24", _container_fps(0.0, 24) == 24.0),
        ("协商 NaN -> 退回请求 24", _container_fps(float("nan"), 24) == 24.0),
        ("协商负数 -> 退回请求 24", _container_fps(-1.0, 24) == 24.0),
        ("保留 3 位小数", _container_fps(29.9999, 24) == 30.0
         and _container_fps(29.12349, 24) == 29.123),
    ]
    for name, passed in checks_container:
        print(f"  [{'通过' if passed else '不达标'}] {name}")
        results.append((name, passed))

    print("----- 17. 会话改名/改备注（备份 + 原子替换 + 路径安全）-----")
    import hashlib

    from apps.data_manager import latest_backup, update_session_meta
    with tempfile.TemporaryDirectory(prefix="dsh_meta_") as tmp:
        mroot = Path(tmp) / "sessions"
        md = mroot / "20260102-120000"
        md.mkdir(parents=True)
        original = {
            "schema": "sphere-stereo-session", "version": 1, "id": md.name,
            "created": "2026-01-02 12:00:00", "label": "GUI 录制",
            "notes": "GUI 录制", "left": {"frame_count": 7},
            "right": {"frame_count": 7}, "checks": ["a", "b"],
        }
        mjf = md / "session.json"
        mjf.write_text(json.dumps(original, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        md5_before = hashlib.md5(mjf.read_bytes()).hexdigest()
        bak_before = set(p.name for p in md.glob("session.json.bak-*"))

        renamed = update_session_meta(md, label="金属球-左路虚", notes="90度装置",
                                      sessions_root=mroot)
        after = json.loads(mjf.read_text(encoding="utf-8"))
        other_keys_same = all(after.get(k) == v for k, v in original.items()
                              if k not in ("label", "notes"))
        key_order_same = list(after.keys()) == list(original.keys())
        bak = latest_backup(md)
        bak_ok = (bak is not None and bak.name.startswith("session.json.bak-")
                  and bak.name not in bak_before
                  and bak.read_text(encoding="utf-8") == json.dumps(
                      original, ensure_ascii=False, indent=2))
        cleared = update_session_meta(md, notes="", sessions_root=mroot)
        notes_cleared = json.loads(mjf.read_text(encoding="utf-8")).get("notes") == ""
        empty_rejected = False
        try:
            update_session_meta(md, label="", sessions_root=mroot)
        except ValueError:
            empty_rejected = True
        broke = md.parent / "20260103-120000"
        broke.mkdir()
        (broke / "session.json").write_text('{"label": "坏了",}', encoding="utf-8")
        md5_bad_before = hashlib.md5((broke / "session.json").read_bytes()).hexdigest()
        bad_json_kept = False
        try:
            update_session_meta(broke, label="x", sessions_root=mroot)
        except ValueError:
            bad_json_kept = (hashlib.md5(
                (broke / "session.json").read_bytes()).hexdigest() == md5_bad_before)
        escape_rejected = False
        try:
            update_session_meta(mroot / ".." / "outside", label="x",
                                sessions_root=mroot)
        except ValueError:
            escape_rejected = True
        outside_dir = Path(tmp) / "outside"
        outside_dir.mkdir()
        absolute_rejected = False
        try:
            update_session_meta(outside_dir, label="x", sessions_root=mroot)
        except ValueError:
            absolute_rejected = True
        checks_meta = [
            ("改名成功且 label/notes 落盘",
             renamed.get("label") == "金属球-左路虚"
             and renamed.get("notes") == "90度装置" and after["label"] == "金属球-左路虚"),
            ("其它键逐键不变、顺序不变", other_keys_same and key_order_same),
            ("备注清空（空串）成功", notes_cleared),
            ("空 label 被拒", empty_rejected),
            ("坏 JSON 被拒且文件一字未改（md5 相同）", bad_json_kept),
            (f"备份已生成且内容=原件（{bak.name if bak else '无'}）", bak_ok),
            ("`../xxx` 路径逃逸被拒", escape_rejected),
            ("sessions 根之外绝对路径被拒", absolute_rejected),
        ]
        for name, passed in checks_meta:
            print(f"  [{'通过' if passed else '不达标'}] {name}")
            results.append((name, passed))
        print(f"    改名前 md5={md5_before}  现在 md5="
              f"{hashlib.md5(mjf.read_bytes()).hexdigest()}")
        print(f"    备注清空后 notes={json.loads(mjf.read_text(encoding='utf-8'))['notes']!r}")
        print(f"    备份文件：{[p.name for p in sorted(md.glob('session.json.bak-*'))]}")

    # ---- 18. 单板链式标定（第三条路）-----------------------------------
    #
    # 为什么要有这一段：GUI 里的链式引导区是「三段采集（参考段双路同拍）+
    # 一条 chain 标定」，而这三条路（joint / 90° / 链式）**前提不同、命令也必须不同**。
    # 拼错了不会有任何界面反馈，只会白拍半小时板。
    # 断言只碰模块级纯函数，不依赖真实 Tk 状态。
    print("----- 18. 单板链式标定：三段采集 argv / 标定 argv / 就绪判定 -----")
    chain_root = ROOT / "outputs" / "chain" / "20260101-000000"
    panel = dict(width="2560", height="1440", fps="24", backend="auto",
                 cols=8, rows=5, square=25)
    checks_chain: list[tuple[str, bool]] = []
    # 内参两段：单路 + --auto（停稳自动存）+ --target 15
    for stage, side, target in (("left", "left", "15"), ("right", "right", "15")):
        argv = mod.chain_capture_cmd(chain_root, stage, camera=side, **panel)
        out_dir = argv[argv.index("--out") + 1].replace("\\", "/")
        checks_chain.append((
            f"采 {stage}：--side {side} --target {target} --auto --out …/{stage}",
            argv[argv.index("--side") + 1] == side
            and argv[argv.index("--target") + 1] == target
            and argv[argv.index("--camera") + 1] == side
            and "--auto" in argv
            and out_dir.endswith("/" + stage)
            and any(a.endswith("capture_board.py") for a in argv)))

    # 参考段：一次双路同拍，必须 --both + 两个 --out-*，且**不得**带 --side
    ref_argv = mod.chain_capture_cmd(chain_root, "ref", camera=None, **panel)
    ref_left = ref_argv[ref_argv.index("--out-left") + 1].replace("\\", "/")
    ref_right = ref_argv[ref_argv.index("--out-right") + 1].replace("\\", "/")
    checks_chain.append((
        "采 ref：--both --out-left/--out-right --target 3 --auto（不含 --side）",
        "--both" in ref_argv
        and "--side" not in ref_argv
        and "--auto" in ref_argv
        and ref_argv[ref_argv.index("--target") + 1] == "3"
        and ref_left.endswith("/ref_left") and ref_right.endswith("/ref_right")
        and any(a.endswith("capture_board.py") for a in ref_argv)))
    checks_chain.append((
        "三段各只有一个采集入口（left / right / ref），没有单独拍 ref_right 的段",
        [s[0] for s in mod.CHAIN_STAGES] == ["left", "right", "ref"]
        and "ref_right" not in [s[0] for s in mod.CHAIN_STAGES]))

    calib_argv = mod.chain_calibrate_cmd(
        chain_root, cols=8, rows=5, square=25, width="2560", height="1440",
        fps="24", backend="auto", target=40, min_pairs=20)
    calib_joined = " ".join(calib_argv)
    calib_posix = calib_joined.replace("\\", "/")
    root_posix = chain_root.as_posix()
    checks_chain += [
        ("标定命令走 --method chain + --same-board",
         "--method chain" in calib_joined and "--same-board" in calib_argv),
        ("四个目录全传：left / right / ref_left / ref_right",
         all(f"{root_posix}/{d}" in calib_posix
             for d in ("left", "right", "ref_left", "ref_right"))),
        ("不走 90° 那条路（不得含 --stereo-mode）",
         "--stereo-mode" not in calib_argv and "angled" not in calib_argv),
        ("板参数 / 分辨率沿用同一份面板值",
         "--cols 8 --rows 5 --square 25" in calib_joined
         and "--width 2560 --height 1440 --fps 24" in calib_joined),
        ("后端 auto 时不传 --backend（落到脚本默认）",
         "--backend" not in calib_argv),
    ]

    with tempfile.TemporaryDirectory(prefix="chain_cnt_") as tmp:
        cdir = Path(tmp)
        zero = mod._count_shots(cdir)
        (cdir / "f000.png").write_bytes(b"x")
        one = mod._count_shots(cdir)
        (cdir / "notes.txt").write_text("x", encoding="utf-8")
        for i in range(1, 15):
            (cdir / f"f{i:03d}.png").write_bytes(b"x")
        fifteen = mod._count_shots(cdir)
        missing = mod._count_shots(cdir / "no_such_dir")
    checks_chain += [
        ("_count_shots：0 / 1 / 15 张，txt 不计", (zero, one, fifteen) == (0, 1, 15)),
        ("_count_shots：目录不存在按 0 张", missing == 0),
    ]

    full = {"left": 15, "right": 15, "ref_left": 3, "ref_right": 3}
    checks_chain += [
        ("_chain_ready：四目录齐 → 可点", mod._chain_ready(full) is True),
        ("_chain_ready：ref_left 只有 0 张 → 不可点（双路参考缺一路）",
         mod._chain_ready({**full, "ref_left": 0}) is False),
        ("_chain_ready：任一段 0 张 → 不可点（按钮 disabled）",
         all(mod._chain_ready({**full, k: 0}) is False for k in full)),
        ("_chain_ready：四段各 1 张也放行（参考段偏少只提示）",
         mod._chain_ready({k: 1 for k in full}) is True),
        ("缺图时状态文字点名缺哪一段",
         "拍左参考图" in mod._chain_status_text({**full, "ref_left": 0})),
    ]
    for name, passed in checks_chain:
        print(f"  [{'通过' if passed else '不达标'}] {name}")
        results.append((name, passed))
    print(f"    采集 argv（left）：{' '.join(mod.chain_capture_cmd(chain_root, 'left', camera='left', **panel))}")
    print(f"    采集 argv（right）：{' '.join(mod.chain_capture_cmd(chain_root, 'right', camera='right', **panel))}")
    print(f"    采集 argv（ref）：{' '.join(ref_argv)}")
    print(f"    标定 argv：{calib_joined}")

    # ---- 操作说明（2026-11-05）----
    # 并进这个文件而不是新建 `selftest_help.py`：`selftest_doc_facts` 里
    # `REAL_TEST_COUNT = 26` 是**有签字的预算**（注释写明"要扩得先有人签"），
    # 顺手加文件会直接把它顶红。`view_3d` 的 9 条判据当年也是这么并进来的。
    print("----- 8e. 操作说明：内容 / 搜索 / 窗口 -----")
    for _name, _passed in _check_help():
        print(f"  [{'通过' if _passed else '不达标'}] {_name}")
        results.append((_name, _passed))

    # ---- 标准窗口 + gitignore（2026-11-05）----
    # 同样并进本文件，不动 selftest_doc_facts 里有签字的 REAL_TEST_COUNT。
    print("----- 8f. 清理窗口 / gitignore ----")
    for _name, _passed in _check_cleanup_windows():
        print(f"  [{'通过' if _passed else '不达标'}] {_name}")
        results.append((_name, _passed))

    # ---- 隐私：入库文件不许带个人标识（2026-11-05）----
    print("----- 8g. 入库文件个人信息 ----")
    for _name, _passed in _check_no_personal_info():
        print(f"  [{'通过' if _passed else '不达标'}] {_name}")
        results.append((_name, _passed))

    # ---- 纯净交付包 / 白名单式 gitignore（2026-11-05）----
    print("----- 8h. 纯净交付包 ----")
    for _name, _passed in _check_clean_bundle():
        print(f"  [{'通过' if _passed else '不达标'}] {_name}")
        results.append((_name, _passed))

    # ---- 路径可移植性（2026-11-05）----
    # 并进本文件，同样不动 selftest_doc_facts 里有签字的 REAL_TEST_COUNT。
    print("----- 8i. 会话根路径可移植性 ----")
    for _name, _passed in _check_session_root_portability():
        print(f"  [{'通过' if _passed else '不达标'}] {_name}")
        results.append((_name, _passed))

    # ---- 一键环境安装器（2026-11-05）----
    print("----- 8j. 一键环境安装器 ----")
    for _name, _passed in _check_env_installer():
        print(f"  [{'通过' if _passed else '不达标'}] {_name}")
        results.append((_name, _passed))

    # ---- GUI 多语言（2026-11-05）----
    print("----- 8k. GUI 多语言 ----")
    for _name, _passed in _check_i18n():
        print(f"  [{'通过' if _passed else '不达标'}] {_name}")
        results.append((_name, _passed))

    # ---- 8l. 三个新功能：contain / 原始视频源 / 资源管理器模块（2026-11-12）----
    # 独立验证者钉的回归护栏：全部用临时会话 / 打桩，不依赖本机录过什么。
    # A=运行时证明 contain（不只 grep 源码）；B=raw 双 cap + 无三维兜底；
    # C=新模块打桩（绝不真弹资源管理器，gui_walkthrough 靠 os.startfile 属性路径）。
    print("----- 8l. 3D 查看器 contain + 原始视频源 + 资源管理器模块 -----")
    for _name, _passed in _check_viewer_raw_and_explorer():
        print(f"  [{'通过' if _passed else '不达标'}] {_name}")
        results.append((_name, _passed))

    ok = all(v for _n, v in results)
    bad = [n for n, v in results if not v]
    if bad:
        print(f"\n不达标 {len(bad)} 项：")
        for n in bad:
            print(f"  · {n}")
    print("\n判定结果：" + ("全部通过" if ok else "有不达标项"))
    return 0 if ok else 1


def _check_session_root_portability() -> list[tuple[str, bool]]:
    """会话根偏好不许把项目内目录固化成绝对路径（2026-11-05）。

    坑：单目窗关闭时把 ``mono.session_root`` 无条件写成 ``str(Path)``，
    连默认的 ``data/mono`` 也被记成 ``D:/.../data/mono``。整个仓库换个盘、
    或发给别人后，界面仍指向旧机器那个目录，静默列出一份空会话列表。
    反过来，用户自己选的**仓库外**目录必须保留绝对路径 —— 那可能只是
    移动硬盘这次没插，替他改掉才是破坏。
    """
    try:
        from apps._gui_mono import (MONO_SESSIONS_ROOT, ROOT as APP_ROOT,
                                    _restore_session_root, _store_session_root)
    except Exception as exc:                              # noqa: BLE001
        return [(f"会话根路径模块可导入（{type(exc).__name__}: {exc}）", False)]

    stored = _store_session_root(MONO_SESSIONS_ROOT)
    restored = _restore_session_root(stored)
    stale = APP_ROOT.parent / "__gone_repo__" / "data" / "mono"
    external = APP_ROOT.parent / "__external__" / "videos"
    inside = APP_ROOT / "outputs" / "custom"
    src = (ROOT / "apps" / "_gui_mono.py").read_text(encoding="utf-8")
    return [
        ("项目内默认会话根存成相对路径（不含盘符）",
         stored == "data/mono" and not re.match(r"^[A-Za-z]:", stored)),
        ("相对值还原成仓库根下的绝对路径",
         restored.is_absolute() and restored == APP_ROOT / "data" / "mono"),
        ("空值回退到默认会话根",
         _restore_session_root("") == MONO_SESSIONS_ROOT),
        ("历史遗留的失效绝对路径回退默认，不指向旧机器",
         _restore_session_root(str(stale)) == MONO_SESSIONS_ROOT),
        ("用户自定义的仓库外目录原样保留（移动硬盘没插也别替他改）",
         _restore_session_root(str(external)) == external),
        ("项目内任意子目录存/取往返一致",
         _restore_session_root(_store_session_root(inside)) == inside),
        ("保存点确实走 _store_session_root（不许退回 str(Path)）",
         'p.set("mono.session_root", _store_session_root(self._session_root))' in src),
    ]


def _check_env_installer() -> list[tuple[str, bool]]:
    """一键环境安装器（2026-11-05）：三入口认同一份指针，路径仍可移植。

    守的是这几件容易悄悄退化的事：安装器必须能在「一个第三方包都没装」
    的机器上运行；根目录不许冒出第四个脚本；bat / Go 都要优先读指针；
    指针里项目内的解释器必须是相对路径；本机状态不许混进纯净包。
    """
    import ast
    import subprocess

    installer = ROOT / "apps" / "setup_env.py"
    bat = ROOT / "run.bat"
    go = ROOT / "native" / "launcher" / "main.go"
    if not installer.is_file():
        return [("apps/setup_env.py 存在", False)]
    src = installer.read_text(encoding="utf-8")
    bat_src = bat.read_text(encoding="utf-8", errors="replace")
    go_src = go.read_text(encoding="utf-8", errors="replace")
    un = ROOT / "uninstall.bat"
    un_src = (un.read_text(encoding="utf-8", errors="replace")
              if un.is_file() else "")
    # 第二遍 :execute 段单独切出来：第一遍也有 .git / AGENTS.md 关键词，
    # 必须确保复核真的写在第二遍里，否则"只验第一遍"的退化抓不住。
    _un_lines = un_src.splitlines()
    _exec_at = next((i for i, ln in enumerate(_un_lines)
                     if ln.strip() == ":execute"), None)
    exec_part = "\n".join(_un_lines[_exec_at:]) if _exec_at is not None else ""

    def ignored(rel: str) -> bool:
        return subprocess.run(
            ["git", "check-ignore", "--no-index", "-q", rel],
            cwd=str(ROOT), capture_output=True, check=False).returncode == 0

    from apps import setup_env as se
    from apps.tools import make_clean_bundle as m

    # 用 AST 查**真正的** import：源码注释里写着 "import cv2" 是踩坑说明，
    # 字符串搜索会误报（这一条第一次就误报了）。
    imported: set[str] = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    stdlib = set(getattr(sys, "stdlib_module_names", ()))
    foreign = sorted(imported - stdlib) if stdlib else sorted(
        m for m in imported
        if m in ("cv2", "numpy", "yaml", "PIL", "sphere_tracker", "apps",
                 "stereo", "matplotlib"))

    external = se.ROOT.parent / "__external__" / "python.exe"
    rels = [f.relative_to(m.ROOT).as_posix() for f in m.collect()]
    checks: list[tuple[str, bool]] = [
        (f"安装器只用标准库（环境没装好时也得能跑；多出 {foreign}）",
         not foreign),
        ("根目录可执行文件一律英文名（AGENTS.md 规则 5）",
         all(p.name.isascii() for p in list(ROOT.glob("*.bat"))
             + list(ROOT.glob("*.py")) + list(ROOT.glob("*.exe")))),
        ("GUI / CLI 主程序已搬出根目录（根目录不再有 .py）",
         (ROOT / "apps" / "gui_main.py").is_file()
         and (ROOT / "apps" / "cli_main.py").is_file()
         and not list(ROOT.glob("*.py"))),
        ("run.bat 不再靠 glob 扫根目录找 GUI（脚本搬走后必失效）",
         "apps\\gui_main.py" in bat_src
         and "glob.glob('*.py')" not in bat_src),
        ("install.bat / uninstall.bat 都在且是英文名",
         (ROOT / "install.bat").is_file()
         and (ROOT / "uninstall.bat").is_file()),
        ("卸载器有开发仓库安全闸（.git / .workbuddy / AGENTS.md）",
         all(k in un_src for k in (".git", ".workbuddy", "AGENTS.md"))),
        ("卸载器只走回收站、需二次确认、无永久删除命令",
         "SendToRecycleBin" in un_src and "UNINSTALL" in un_src
         and "rmdir /s /q" not in un_src and "del /f /s /q" not in un_src),
        ("卸载器自我复制到 TEMP 才执行（否则删不掉自己）",
         "%TEMP%" in un_src and "--execute" in un_src),
        ("卸载目标走临时文件传参、由独立进程执行（start 会吞带空格参数）",
         "sphere_uninstall_target.txt" in un_src
         and "start " in un_src and "/D " in un_src
         and "SendToRecycleBin" in un_src),
        # 2026-11-22：installer 可靠性两处高危缺陷的回归护栏（含反向自证）。
        # ① install.bat 用 call ... install %* 原样转发（引号还在），run.bat 旧写法
        #    %~2..%~5 既剥引号又丢第 5 个之后的参数 —— 反向自证必须钉住旧串消失。
        ("run.bat 参数转发不再用写死的 %~2..%~5（反向自证）",
         "%~2 %~3 %~4 %~5" not in bat_src),
        ("run.bat 逐参数重新加引号转发、数量不限（INSTALL_ARGS + shift）",
         "INSTALL_ARGS" in bat_src and "shift" in bat_src
         and '"%BOOT%" apps\\setup_env.py %INSTALL_ARGS%' in bat_src),
        ("py launcher 兜底先验版本再收（旧 py.exe 不会在入口语法崩）",
         "sys.version_info>=(3,9)" in bat_src and "where py" in bat_src),
        ("安装完成提示不再指向纯净包里没有的 launcher.exe",
         "launcher.exe" not in next(
             (ln for ln in bat_src.splitlines() if "环境已就绪" in ln), "")),
        # ② 卸载第二遍从 %TEMP% 副本执行，必须重新验明目标身份 + 拦盘符根。
        ("卸载器第二遍(:execute)复核身份与盘符根（不是只验第一遍）",
         all(k in exec_part for k in (".git", "AGENTS.md", "%SystemDrive%"))
         and 'if not exist "%TARGET%\\run.bat"' in exec_part
         and 'if not exist "%TARGET%\\apps\\gui_main.py"' in exec_part),
        ("卸载器两遍都有 .git 安全闸（反向自证：只写一处会红）",
         un_src.count('if exist "%TARGET%\\.git"') >= 2
         and un_src.count(".git") >= 2),
        ("卸载目标传参文件带 <%RANDOM%> 后缀（并发卸载不互相覆盖）",
         "sphere_uninstall_target.txt" in un_src
         and "%RANDOM%" in un_src and "TARGETFILE" in un_src),
        ("bat 有 install 路由",
         'if /i "%MODE%"=="install" goto :do_install' in bat_src
         and ":do_install" in bat_src),
        ("bat 优先读安装器登记的环境",
         "config\\python_env.txt" in bat_src
         and "安装器登记的环境" in bat_src),
        ("Go 启动器优先读环境指针",
         "pointerPython(root)" in go_src
         and 'if p := pointerPython(root); p != ""' in go_src),
        ("Go 启动器找不到环境时会询问并自动安装",
         "runInstaller(root)" in go_src and "ask(" in go_src),
        # 2026-11-08/11-24：用户反复实测 **run.bat 能打开相机、launcher.exe 打不开**。
        # 2026-11-24 定位到根因：launcher 原来用 hideWindow()(CREATE_NO_WINDOW) 把
        # cmd→run.bat→python 整条链的控制台藏了，console=False 时 MSMF 相机全部
        # 打不开；run.bat 双击 console=True 每次都出图。修法是 showConsole()
        # 给真控制台（与 run.bat 同口径），相机优先于"无黑框"。
        ("启动器把 GUI 交给 run.bat 且给真控制台（相机可用）",
         'filepath.Join(root, "run.bat")' in go_src
         and '"/c", bat, "gui"' in go_src
         and 'showConsole(c)' in go_src
         and 'hideWindow(c)\n\t\tif err := c.Start()' not in go_src),
        ("反向：把 run.bat 那条链改回 hideWindow → 判据会红",
         'showConsole(c)' in go_src),
        ("反向：把启动器改回直接起 pythonw → 判据会红",
         not ('"/c", bat, "gui"' in
              go_src.replace('"/c", bat, "gui"', 'script'))),
        ("项目内解释器写相对路径",
         se.pointer_value(se.ROOT / ".venv" / "Scripts" / "python.exe", se.ROOT)
         == ".venv/Scripts/python.exe"),
        ("相对指针按仓库根还原成绝对路径",
         se.parse_pointer("python=.venv/Scripts/python.exe\n", se.ROOT)
         == (se.ROOT / ".venv" / "Scripts" / "python.exe").resolve()),
        ("仓库外环境保留绝对路径",
         se.pointer_value(external, se.ROOT) == str(external)
         and se.parse_pointer("python=" + str(external) + "\n", se.ROOT)
         == external.resolve()),
        ("指针里的注释行不干扰解析",
         se.parse_pointer("# comment\n\npython=data/x\n", se.ROOT)
         == (se.ROOT / "data" / "x").resolve()),
        ("AI 组 ultralytics 必须 --no-deps（否则覆盖 opencv-contrib）",
         '"ultralytics", "--no-deps"' in src
         and src.find('"torch==2.11.0"') < src.find('"ultralytics"')),
        ("安装器进纯净交付包", "apps/setup_env.py" in rels),
        ("环境指针 / venv / 日志都不进纯净包",
         not any(r.startswith((".venv/", "config/python_env", "logs/"))
                 for r in rels)),
        ("本机环境指针被 gitignore 挡住（含绝对路径，不能入库）",
         ignored("config/python_env.txt")
         and ignored(".venv/Scripts/python.exe")),
    ]

    # 安装器不能 import sphere_tracker（其 __init__ 会拉 cv2），却必须把日志
    # 写进同一个规范目录。这里拿真正的 log_dir 对账，防止两套规则悄悄漂移。
    from datetime import datetime
    from sphere_tracker.paths import log_dir
    probe_log = se.setup_log_path(datetime(2026, 11, 5, 18, 30, 0))
    checks.append(("安装日志落在规范的 logs/misc/<月>/ 分片里",
                   probe_log.parent == log_dir("misc", when=datetime(2026, 11, 5))
                   and probe_log.name == "setup_env_20261105_183000.log"))

    try:
        h = subprocess.run([sys.executable, str(installer), "--help"],
                           cwd=str(ROOT), capture_output=True, text=True,
                           timeout=60, check=False)
        d = subprocess.run([sys.executable, str(installer), "--dry-run"],
                           cwd=str(ROOT), capture_output=True, text=True,
                           timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        checks.append((f"安装器 --help / --dry-run 能跑（{exc}）", False))
        return checks
    checks.append(("--help 真解析参数（selftest_entrypoints 会扫）",
                   h.returncode == 0 and "usage:" in (h.stdout or "")))
    checks.append(("--dry-run 只打印计划、不改动环境",
                   d.returncode == 0 and "不改动任何东西" in (d.stdout or "")))
    return checks


def _unscanned_gui_files() -> set[str]:
    """界面文件里，没登记进 `i18n_extract.SCAN_NAMES` 的那些。

    只挑「一定是界面」的两类：``apps/_gui_*.py`` 与 ``apps/*_dialog.py``。
    不扫全部 `apps/*.py` —— `run_video.py` / `calibrate_stereo.py` 里的
    argparse help 本来就不该进界面词典（那会把命令行文案也拖进翻译表）。
    """
    from apps.tools import i18n_extract as ex

    apps_dir = ROOT / "apps"
    gui = {p.name for p in apps_dir.glob("_gui_*.py")}
    gui |= {p.name for p in apps_dir.glob("*_dialog.py")}
    return gui - set(ex.SCAN_NAMES)


def _check_i18n() -> list[tuple[str, bool]]:
    """GUI 多语言：运行时切换 + 新增文案自动生成对应语言版本。

    守两件容易悄悄退化的事：① 切换语言来回不能把译文当成新原文存下来
    （实测踩过，界面再也切不回中文）；② 界面上新增一句中文，双语资源
    必须自动跟上，而不是靠人记得改两份文件。
    """
    import subprocess
    import tempfile

    from apps import i18n, i18n_data, i18n_seed
    from apps.tools import i18n_extract as ex
    from apps.tools import make_clean_bundle as m

    def has_cn(s: str) -> bool:
        return any("\u4e00" <= ch <= "\u9fff" for ch in s)

    keys = ex.scan_all()
    done = sum(1 for v in i18n_data.EN_US.values() if v)

    # 2026-11-23 实测漏翻的根因回归：门厅三张卡的说明写在局部元组 `specs`
    # 里，循环解包后以变量传给 card()。旧提取器只认调用点的字面量，三句
    # 从未进词典，切到 en-US 后正文照旧中文，而当时所有门禁全绿。
    # 这里同时守"扫得到"和"翻得出"，缺任一条都会让截图里的现象复现。
    hub_keys = ex.extract_strings(ROOT / "apps" / "_gui_app.py")
    card_desc = sorted(k for k in hub_keys if k.startswith("深度靠"))
    checks: list[tuple[str, bool]] = [
        ("中英文两种语言已注册",
         "zh-CN" in i18n.LANGS and "en-US" in i18n.LANGS),
        ("种子词典 key 含中文、译文不含中文",
         all(has_cn(k) for k in i18n_seed.SEED)
         and all(not has_cn(v) for v in i18n_seed.SEED.values())),
        (f"资源覆盖全部扫描到的界面文案（{len(keys)} 条）",
         len(keys) > 300 and keys <= set(i18n_data.EN_US)),
        (f"英文资源有自动生成的译文（{done}/{len(i18n_data.EN_US)}）",
         done > len(i18n_data.EN_US) // 2),
        # `i18n_extract.SCAN_NAMES` 是白名单 —— 新加一个界面文件忘了登记时，
        # 它不会报错，只是那个窗口在英文模式下**整块照旧中文**。2026-11-07 加
        # `tools_dialog.py` 时正是这样漏的（`--check` 说"一致"）。
        # 所以这里把"记得登记"变成判据：GUI 文件必须都在白名单里。
        (f"i18n 扫描白名单覆盖全部界面文件（漏 {sorted(_unscanned_gui_files())}）",
         not _unscanned_gui_files()),
        ("提取器能扫到门厅 specs 元组里的模式卡说明（3 句）",
         len(card_desc) >= 3),
    ]

    # 2026-11-24 第二轮漏翻：相机管理窗口的失败结论写在模块级常量
    # （REASON_* / _NO_DSHOW_HELP / _OPEN_FIX_STEPS）与现拼的 f-string 里，
    # 旧提取器既扫不到常量、也不认 f-string；英文模式下相机打不开时满屏中文。
    # 这里守三件事：扫得到、翻得出、显示点确实走了 t()/tf() + forget + 重放。
    cam_keys = ex.extract_strings(ROOT / "apps" / "cameras_dialog.py")
    cam_reasons = [k for k in cam_keys if k.startswith("采集会话失效")
                   or k.startswith("打不开：") or k.startswith("打开了但没画面")]
    cam_src = (ROOT / "apps" / "cameras_dialog.py").read_text(encoding="utf-8")
    checks += [
        (f"相机失败结论常量被提取器扫到（{len(cam_reasons)} 句）",
         len(cam_reasons) >= 3),
        ("相机失败结论在英文资源里有真译文",
         bool(cam_reasons) and all(
             k in i18n_data.EN_US and i18n_data.EN_US[k]
             and not has_cn(i18n_data.EN_US[k]) for k in cam_reasons)),
        ("相机状态句在显示点翻译并清记账（tf + forget）",
         "i18n.tf(" in cam_src and "i18n.forget(" in cam_src
         and "def _i18n_refresh" in cam_src
         and "i18n.apply_children(self)" in cam_src),
        ("`#` 前缀的相机文案不再被误排除（颜色码仍排除）",
         any("出图了" in k for k in cam_keys)
         and ex._clean("#b91c1c") is False
         and ex._clean("#1 出图了") is True),
    ]

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        old_file, old_lang = i18n.LANG_FILE, i18n.language()
        old_refresh = i18n.refresh_all
        i18n.LANG_FILE = tmp / "language.txt"
        # 本文件前面建了 GUI 窗口，切换语言时不要真的去刷那棵控件树。
        i18n.refresh_all = lambda: None                 # type: ignore[assignment]
        # 用户本机可能已存 en-US；此时 set_language("en-US") 会 early-return，
        # 不写临时语言文件，下面 saved=read_text() 就抛 FileNotFoundError。
        # 先强制回到中文这个已知起点，测的才是"切换"而非"已是目标态"。
        i18n._current = "zh-CN"
        try:
            cn = i18n.t("确定")
            i18n.set_language("en-US")
            en = i18n.t("确定")
            auto = i18n.t("已选 3 个文件 / 12 个")
            # 必须在切回中文**之前**取英文结果：下面的 checks 是切回之后才构造的
            en_cards = [i18n.t(s) for s in card_desc]
            # 动态状态句的英文模板：用纯数据假 state，不依赖 GUI / 本机标定文件。
            # 截图里"就绪：已标定 fx=…"这类句子整句不在词典里，只有 tf() 模板
            # 能翻；这条守着"状态骨架在英文模式下不再漏中文"。
            from apps._gui_app import AppState as _AppState
            fake = object.__new__(_AppState)
            fake.notes = ["单目标定未绑定设备身份（老文件），换相机不会自动报警；请重标"]
            fake.infos = []
            fake.profile = type("P", (), {"name": "demo-rig"})()
            fake.rig = type("R", (), {"angle_deg": 86.5, "baseline_mm": 444.0,
                                      "meta": {"stereo_mode": "angled"}})()
            en_status = fake.status_text()[0]
            en_mono = i18n.tf("已标定 fx = {fx:.0f} px{extra}{tag}",
                              fx=1388.0, extra="", tag="")
            # 下拉框必须“显示译文、逻辑值仍是原文”：不能为了英文界面把
            # var.get() 变成英文，否则 _gui_stereo / mono 的方法匹配会静默失败。
            import tkinter as tk
            from tkinter import ttk
            probe_root = tk.Tk()
            probe_root.withdraw()
            try:
                probe_var = tk.StringVar(master=probe_root, value="镜面金属球")
                probe_cb = ttk.Combobox(
                    probe_root, values=["镜面金属球", "纯色塑料球"],
                    textvariable=probe_var)
                shown = tuple(probe_cb.cget("values"))
                probe_cb.current(1)
                combo_ok = (
                    shown == ("Mirror-finish metal ball",
                              "Solid-color plastic ball")
                    and probe_cb.get() == "纯色塑料球"
                    and probe_var.get() == "纯色塑料球")
            finally:
                probe_root.destroy()
            dynamic_ok = i18n.dynamic("共 3 条说明") == "3 help entries"
            saved = i18n.LANG_FILE.read_text(encoding="utf-8").strip()
            i18n.set_language("zh-CN")
            back = i18n.t("确定")
            checks += [
                ("中文模式是恒等映射", cn == "确定"),
                ("英文模式命中精确译文", en == "OK"),
                ("每条中文文案都有完整英文（非空且无中文）",
                 all(v and not any("\u4e00" <= c <= "\u9fff" for c in v)
                     for v in i18n_data.EN_US.values())),
                # 这句是**故意未登记**的探针，只验证"不做逐词拼接"这一行为；
                # 它不是界面真实文案。界面文案必须登记，由下面 card_desc 的
                # 断言守着 —— 别把这条读成"漏翻也可以接受"（2026-11-23 教训）。
                ("未登记长句不再逐词拼接（保持中文）",
                 auto == "已选 3 个文件 / 12 个"),
                ("门厅三句模式卡说明在 en-US 下真的翻成英文",
                 len(en_cards) == len(card_desc) >= 3
                 and all(v != s and not has_cn(v)
                         for v, s in zip(en_cards, card_desc))),
                ("动态状态句在 en-US 下翻成英文（fx/装置名等数据保留）",
                 not has_cn(en_status) and not has_cn(en_mono)
                 and "Active rig" in en_status and "Calibrated fx" in en_mono),
                ("下拉框显示英文、get() 仍返回中文逻辑值", combo_ok),
                ("动态整句模板能翻（不逐词）", dynamic_ok),
                ("语言偏好已持久化", saved == "en-US"),
                ("切回中文能还原（不会把译文当原文）", back == "确定"),
            ]
        finally:
            i18n.LANG_FILE = old_file
            i18n.refresh_all = old_refresh              # type: ignore[assignment]
            if i18n.language() != old_lang:
                i18n.set_language(old_lang)

        probe = tmp / "probe.py"
        probe.write_text(
            '"""模块说明不该被提取"""\n'
            'def f():\n'
            '    """函数说明不该被提取"""\n'
            '    tk.Label(root, text="保存新的测试")\n'
            '    log.info("内部日志 %s 不该被提取")\n',
            encoding="utf-8")
        found = ex.extract_strings(probe)
        checks += [
            ("提取器能发现新增界面文案", "保存新的测试" in found),
            ("提取器不收注释 / docstring / 内部日志",
             not any("不该被提取" in s for s in found)),
            ("新增文案不会被逐词自动拼英文（必须补整句）",
             ex.auto_translate("保存新的测试") == ""),
            ("固定短词仍走精确种子译文",
             ex.auto_translate("保存") == "Save"),
        ]

    gui_src = (ROOT / "apps" / "_gui_app.py").read_text(encoding="utf-8")
    widget_src = (ROOT / "apps" / "_gui_widgets.py").read_text(encoding="utf-8")
    help_src = (ROOT / "apps" / "help_window.py").read_text(encoding="utf-8")
    runtime_src = (ROOT / "apps" / "i18n.py").read_text(encoding="utf-8")
    checks += [
        ("主界面有语言切换入口",
         "i18n.set_language" in gui_src and "Combobox" in gui_src),
        ("状态栏 / 消息框 / 帮助窗口都接入翻译",
         # 状态栏用 forget() 清动态句的原文记账（旧版是 apply(self._label)，
         # 会把新语言文本误当旧原文，2026-11-23 实测）；HubWindow 再用
         # _i18n_refresh 重生成状态句。断言守语义，不钉某一行实现。
         "i18n.forget(self._label)" in widget_src
         and "def _i18n_refresh" in gui_src
         and "_i18n_patched" in runtime_src
         and "i18n.t(body)" in help_src),
    ]
    rels = [f.relative_to(m.ROOT).as_posix() for f in m.collect()]
    checks.append(("多语言模块与资源进纯净交付包",
                   all(x in rels for x in (
                       "apps/i18n.py", "apps/i18n_seed.py", "apps/i18n_data.py",
                       "apps/tools/i18n_extract.py"))))
    try:
        r = subprocess.run(
            [sys.executable, str(ROOT / "apps" / "tools" / "i18n_extract.py"),
             "--check"], cwd=str(ROOT), capture_output=True, text=True,
            timeout=120, check=False)
        checks.append(("提取器 --check 与源码一致", r.returncode == 0))
    except (OSError, subprocess.TimeoutExpired):
        checks.append(("提取器 --check 与源码一致", False))
    return checks


def _check_clean_bundle() -> list[tuple[str, bool]]:
    """白名单式 .gitignore + 纯净交付包（2026-11-05）。

    用户要"打包发送一个纯净版本，只有源代码和运行必须物，其他资产全部没有"。
    守三件事：
      1. 白名单规则真的按预期放行/忽略（用 `--no-index` 测规则本身，
         不用它的话已跟踪文件一律不算忽略，docs/ 会假通过）；
      2. `make_clean_bundle.collect()` 与 `.gitignore` 两份清单不漂移；
      3. 实际生成的 zip 里没有禁入目录、大二进制，也没有个人标识。
    """
    import subprocess
    import tempfile
    import zipfile

    checks: list[tuple[str, bool]] = []
    root_dir = Path(__file__).resolve().parent
    while not (root_dir / "apps").is_dir() and root_dir.parent != root_dir:
        root_dir = root_dir.parent

    def ignored(rel: str) -> bool:
        return subprocess.run(
            ["git", "check-ignore", "--no-index", "-q", rel],
            cwd=str(root_dir), capture_output=True, check=False
        ).returncode == 0

    must_pass = ("apps/view_3d.py", "apps/tools/make_clean_bundle.py",
                 "sphere_tracker/paths.py", "stereo/profiles.py",
                 "calib/methods.py", "tests/selftest_entrypoints.py",
                 "apps/gui_main.py", "requirements.txt", "README.md",
                 "assets/boards/chessboard_8x5_25mm_a4-landscape.png",
                 "config/.gitkeep",
                 # 2026-11-05：公开说明文档是要带上 GitHub 的，不是"资产"
                 "docs/GitHub-README.md", "docs/算法解析.md",
                 "docs/用户手册.md", "docs/API.md", "docs/坐标系统图.png",
                 # 原生 exe 启动器的重建源码（产物 exe 本身不进包）
                 "native/launcher/main.go", "native/launcher/build.bat")
    must_ignore = ("docs/交接文档.md", "docs/待办清单.md",
                   "docs/planbooks/2026-10-10_x.md", "docs/refs/paper.pdf",
                   "docs/new_tool.py", "legacy/README.md", "AGENTS.md",
                   "assets/images/test.jpg", "config/profiles.json",
                   "data/stereo/parallel/x/derived/visualization.json",
                   "outputs/test_results/timings.json", "logs/tasks/x.log",
                   ".workbuddy/memory/x.md", "apps/tools/tmp_debug.log",
                   "native/launcher/icon.syso", "launcher.exe")
    wrong_pass = [x for x in must_pass if ignored(x)]
    wrong_ignore = [x for x in must_ignore if not ignored(x)]
    checks.append((f"白名单放行源码/必需物（误拦 {wrong_pass[:2]}）",
                   not wrong_pass))
    checks.append((f"白名单拦住资产/产物（漏网 {wrong_ignore[:2]}）",
                   not wrong_ignore))

    from apps.tools import make_clean_bundle as m

    files = m.collect()
    rels = [f.relative_to(m.ROOT).as_posix() for f in files]
    checks.append((f"收到合理数量的源文件（{len(files)} 个）", len(files) > 100))
    checks.append(("纯净包内容校验无问题", not m.validate(files)))
    checks.append(("两份白名单不漂移（gitignore 一致性）",
                   not m.gitignore_conflicts(files)))

    forbidden = ("outputs/", "logs/", "legacy/", ".workbuddy/",
                 "data/stereo/", "data/mono/", ".git/",
                 "docs/planbooks/", "docs/refs/")
    relics = ("docs/交接文档.md", "docs/待办清单.md", "docs/决策记录.md",
              "docs/团队角色与派发记录.md")
    checks.append(("包内没有禁入目录",
                   not any(r.startswith(forbidden) for r in rels)))
    checks.append(("包内没有内部史料文档",
                   not any(r in relics for r in rels)))
    build_src = [r for r in rels if r.startswith("native/launcher/")]
    checks.append((f"包内含 exe 启动器重建源码（{len(build_src)} 个）",
                   len(build_src) == 5))
    checks.append(("编译产物（exe / syso）不进包",
                   "launcher.exe" not in rels
                   and not any(r.endswith(".syso") for r in rels)))
    need = ("apps/gui_main.py", "run.bat", "apps/cli_main.py",
            "install.bat", "uninstall.bat",
            "requirements.txt", "requirements-ai.txt", "environment.yml",
            ".gitignore", "apps/view_3d.py", "sphere_tracker/paths.py",
            "stereo/profiles.py", "calib/methods.py",
            "assets/boards/chessboard_8x5_25mm_a4-landscape.png",
            "docs/GitHub-README.md", "docs/算法解析.md",
            "docs/用户手册.md", "docs/API.md",
            "native/launcher/main.go", "native/launcher/go.mod",
            "native/launcher/build.bat", "native/launcher/icon.ico")
    checks.append((f"运行必需物齐全（缺 {[n for n in need if n not in rels][:2]}）",
                   all(n in rels for n in need)))

    mini = [f.relative_to(m.ROOT).as_posix()
            for f in m.collect(with_tests=False, with_assets=False)]
    checks.append(("--no-tests --no-assets 生效（极简包）",
                   not any(r.startswith(("tests/", "assets/")) for r in mini)
                   and len(mini) < len(rels)))

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "clean.zip"
        try:
            m.build(files, out)
            with zipfile.ZipFile(out) as z:
                names = z.namelist()
                bad_marks = []
                for n in names:
                    if (n.endswith((".py", ".md", ".txt", ".yml", ".bat"))
                            or n in (".gitignore", ".gitattributes")):
                        txt = z.read(n).decode("utf-8", "ignore")
                        if any(mark in txt for mark in m.PRIVATE_MARKS):
                            bad_marks.append(n)
            checks.append(("zip 内容恰好等于白名单", names == rels))
            checks.append((f"包内文本无个人标识（{bad_marks[:2]}）",
                           not bad_marks))
            # 发布包首页必须是面向外部的 GitHub 版，且导航链接不能指向
            # 没被打进包的文档（否则 GitHub 上一片 404）。
            import re as _re

            # 上面的 with 块已关闭，重新开一次读首页（不要复用已关闭的句柄）
            with zipfile.ZipFile(out) as z2:
                readme = z2.read("README.md").decode("utf-8", "ignore")
            checks.append(("zip 根目录 README 是 GitHub 版",
                           "## 文档导航" in readme
                           and "docs/算法解析.md" in readme))
            links = _re.findall(r"\]\((docs/[^)#]+)\)", readme)
            missing = [x for x in links if x not in names]
            checks.append((f"README 导航链接在包内可达（缺 {missing[:2]}）",
                           bool(links) and not missing))
            checks.append(("包内无 planbooks / 交接 / 待办",
                           not any(n.startswith(("docs/planbooks/", "docs/refs/"))
                                   or n in ("docs/交接文档.md", "docs/待办清单.md")
                                   for n in names)))
        except Exception as exc:                        # noqa: BLE001
            checks.append((f"能生成 zip（异常：{exc}）", False))
    return checks


def _check_no_personal_info() -> list[tuple[str, bool]]:
    """**入库**的文本文件不许出现本机用户名 / 私人目录 / 真实邮箱。

    2026-11-05 全仓库审计：8 个入库文件里写死了 `C://Users//<数字用户名>`、
    私人课程目录（含"大二上/普通物理学实验2"这类学业信息）。这些一旦推到
    公开仓库就收不回来了。本地 logs/ 与 outputs/ 里还有上万处机器生成的
    路径，但它们**已被 gitignore 挡住**，不是本检查的对象。

    标记串用拼接构造（`'391' + '63'` 等），否则这段测试源码自己就会命中，
    变成永远红的假警报。
    """
    import subprocess

    checks: list[tuple[str, bool]] = []
    root_dir = Path(__file__).resolve().parent
    while not (root_dir / "apps").is_dir() and root_dir.parent != root_dir:
        root_dir = root_dir.parent
    try:
        tracked = [x for x in subprocess.run(
            ["git", "-c", "core.quotePath=false", "ls-files", "-z"],
            cwd=str(root_dir), capture_output=True, check=False
        ).stdout.decode("utf-8").split("\0") if x]
    except Exception as exc:                            # noqa: BLE001
        checks.append((f"能列出入库文件（异常：{exc}）", False))
        return checks

    marks = [
        ("本机用户名", "391" + "63"),
        ("私人课程目录", "课程" + "相关作业资料等"),
        ("QQ 邮箱", "@qq." + "com"),
    ]
    exts = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".bat", ".cmd",
            ".sh", ".cfg", ".ini", ".toml", ".csv", ".html", ".xml"}
    hits: dict[str, list[str]] = {name: [] for name, _ in marks}
    scanned = 0
    for rel in tracked:
        f = root_dir / rel
        if f.suffix.lower() not in exts or not f.is_file():
            continue
        try:
            if f.stat().st_size > 2_000_000:
                continue
            lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:                               # noqa: BLE001
            continue
        scanned += 1
        for i, line in enumerate(lines, 1):
            for name, mark in marks:
                if mark in line:
                    hits[name].append(f"{rel}:{i} {line.strip()[:80]}")
    # 防假阴性：`ls-files -z` 的分隔符解析错了会让整个扫描静默跳过。
    checks.append((f"确实扫到了入库文本文件（{scanned} 个）", scanned >= 100))
    for name, _mark in marks:
        found = hits[name]
        checks.append((f"入库文件无{name}（{len(found)} 处：{found[:1]}）",
                       not found))
    return checks


def _check_cleanup_windows() -> list[tuple[str, bool]]:
    """清理窗口是标准窗口 + 自动产物不进 git（2026-11-05）。

    守两件事：

    1. `transient()` 在 Windows 上把 Toplevel 降成**工具窗口**，标题栏只剩一个
       ×（最小化/最大化灰掉、不进任务栏）。用户明确要"标准的三个按钮"。
       配套还去掉了 `grab_set()` —— 全局抓取会让窗口最小化后主窗口点不动；
       重复打开改由调用方单例挡住。
    2. `.gitignore` 原来按"当时见到的目录"逐条加，漏了 data/_trash/、
       散落的 *.log、根级 test_results/、.workbuddy/。这里用
       `git check-ignore --stdin` 一次批量验，同时**反向**验证源码/文档/资产
       没有被新规则误伤（比如全局忽略 *.png 会吃掉 assets/boards/）。
    """
    import subprocess

    checks: list[tuple[str, bool]] = []
    root_dir = Path(__file__).resolve().parent
    while not (root_dir / "apps").is_dir() and root_dir.parent != root_dir:
        root_dir = root_dir.parent

    # ① 源码里不许再把 transient 加回来
    for rel in ("apps/junk_dialog.py", "apps/cleanup_dialog.py"):
        src = (root_dir / rel).read_text(encoding="utf-8")
        checks.append((f"{rel} 不再用 transient（工具窗口只有 ×）",
                       "self.transient(" not in src))

    # ② 实际构造一次垃圾管理器：属性、快捷键、标题栏同步
    root = tk.Tk()
    root.withdraw()
    dlg = None
    try:
        from apps.junk_dialog import JunkDialog

        dlg = JunkDialog(root)
        root.update_idletasks()
        checks.append(("垃圾管理器是标准窗口（transient 空、可缩放）",
                       not bool(dlg.transient())
                       and dlg.resizable() in ((True, True), (1, 1))))
        checks.append(("垃圾管理器支持 F11 最大化 / Esc 关闭",
                       bool(dlg.bind("<F11>")) and bool(dlg.bind("<Escape>"))))
        dlg.state("zoomed")
        root.update()          # 必须用 update：`<Configure>` 靠事件循环派发，
                               # update_idletasks() 不会触发它
        checks.append(("标题栏最大化后窗口内按钮文字同步为「还原」",
                       str(dlg.btn_full.cget("text")) == "还原"))
        dlg._toggle_full()
        root.update()
        checks.append(("点「还原」能退回普通窗口", dlg.state() == "normal"))
    except Exception as exc:                            # noqa: BLE001
        checks.append((f"垃圾管理器能建起来（异常：{exc}）", False))
    finally:
        if dlg is not None:
            try:
                dlg.destroy()
            except Exception:                           # noqa: BLE001
                pass
        root.destroy()

    # ③ gitignore：正反向一起验
    produced = [
        "outputs/test_results/timings.json",
        "logs/tasks/x.log",
        "data/_trash/session/left.mkv",
        "data/stereo/parallel/s/derived/visualization.json",
        "config/profiles.json",
        ".workbuddy/memory/note.md",
        "apps/tools/foo.log",
        "test_results/run.json",
        "coverage.xml",
        "calib/cache.npz",
        "assets/videos/clip.mp4",
        "gui_state.json.bak-20261105",
        # 白名单式 gitignore **不放行 docs/**：文档史料不是运行必需物
        # （2026-11-05 改白名单后，这条从"应可入库"挪到"应被忽略"）。
        "docs/new.md",
    ]
    source = [
        "apps/new_tool.py", "tests/selftest_new.py",
        "sphere_tracker/new_mod.py", "assets/boards/board.png",
        "data/README.md", "config/.gitkeep", "README.md",
    ]
    try:
        # 用**命令行参数**而不是 `--stdin`：Windows 下从 Python 管道喂给
        # `check-ignore --stdin` 时，git 会把输出加引号并留 `\r`
        # （实测 `"outputs/xxx.json\r"`），解析出来全是"漏网"。
        # 直接当参数传一次即可，输出是干净的路径行。
        proc = subprocess.run(
            ["git", "check-ignore", *produced, *source], cwd=str(root_dir),
            text=True, capture_output=True, check=False)
        ignored = {line.strip().strip('"') for line in proc.stdout.splitlines()}
        miss = [x for x in produced if x not in ignored]
        hurt = [x for x in source if x in ignored]
        checks.append((f"程序自动产物全部不进 git（漏 {len(miss)}：{miss[:2]}）",
                       not miss))
        checks.append((f"源码/运行必需资产没有被误忽略（误伤 {hurt[:2]}）",
                       not hurt))
    except Exception as exc:                            # noqa: BLE001
        checks.append((f"gitignore 能被校验（异常：{exc}）", False))
    return checks


def _check_help() -> list[tuple[str, bool]]:
    """操作说明：内容结构、搜索、窗口渲染。

    守什么（2026-11-05 用户要的「说明书」）：

    1. **内容不能空**：分区/条目删光了，窗口打开是白板，用户以为程序坏了；
    2. **每条都得有正文**：只有标题没内容是占位符；
    3. **搜索要对**：单词命中、多词是「与」、搜不到不能退化成全量；
    4. **窗口真能建起来**：初始正文直接渲染（靠 `<<TreeviewSelect>>` 派发会
       出现"刚打开时右侧空白"）、搜索后变扁平列表、清空能回到分区树。
    """
    from apps.help_content import SECTIONS, iter_items, search
    from apps.help_window import HelpWindow
    from apps import i18n as _i18n

    # 本函数用中文关键词（"状态"/"视角"/"采样 坐标"）断言搜索与渲染；
    # 用户本机 config/language.txt 可能是 en-US，那时 HelpWindow 按英文渲染，
    # 中文关键词自然搜不到（不是功能坏）。这里**临时**切到中文，测完恢复，
    # 让判据与用户偏好解耦（2026-11-24）。
    _help_old_lang = _i18n.language()
    if _help_old_lang != "zh-CN":
        _i18n._current = "zh-CN"

    checks: list[tuple[str, bool]] = []

    checks.append((f"说明分区数合理（{len(SECTIONS)} 个）", len(SECTIONS) >= 5))
    ids = [sid for sid, _t, _i in SECTIONS]
    checks.append(("说明分区 id 不重复", len(ids) == len(set(ids))))

    items = list(iter_items())
    checks.append((f"说明条目数合理（{len(items)} 条）", len(items) >= 15))

    thin = [t for _s, _st, t, b in items
            if len(t.strip()) < 4 or len(b.strip()) < 20]
    checks.append((f"每条都有标题和正文（薄弱 {len(thin)} 条：{thin[:2]}）",
                   not thin))
    titles = [t for _s, _st, t, _b in items]
    dup = [t for t in set(titles) if titles.count(t) > 1]
    checks.append((f"说明标题不重复（重名：{dup[:2]}）", not dup))

    # 搜索要覆盖正文，不能只扫标题（「1920」只写在采集档那段正文里）
    checks.append(("搜索覆盖正文（「1920」只在正文里）", bool(search("1920"))))
    one = search("视角")
    checks.append((f"单词搜索有结果（「视角」{len(one)} 条）", len(one) >= 1))
    two = search("采样 坐标")
    checks.append(("多词是「与」（每条同时含两词）",
                   bool(two) and all("采样" in (t + b) and "坐标" in (t + b)
                                     for _s, _st, t, b in two)))
    checks.append(("搜不到就是空（不退化成全量）",
                   search("zzz不存在的词") == []))
    checks.append(("空关键词返回空", search("   ") == []))

    # ---- 窗口 ----
    # 自己建一个隐藏的 root：`main` 跑到这里时 Launcher 已经 destroy 了，
    # 借它的窗口会抛 "application has been destroyed"。
    root = tk.Tk()
    root.withdraw()
    win = None
    try:
        win = HelpWindow(root)
        win.update_idletasks()

        tops = win.tree.get_children()
        checks.append((f"目录分区节点数 = {len(SECTIONS)}（实际 {len(tops)}）",
                       len(tops) == len(SECTIONS)))
        empty = [win.tree.item(t, "text") for t in tops
                 if not win.tree.get_children(t)]
        checks.append((f"每个分区下都有条目（空：{empty[:2]}）", not empty))

        # 刚建好就该有正文 —— 依赖选中事件派发会留白一帧
        body = win.text.get("1.0", "end").strip()
        checks.append((f"打开即有正文（{len(body)} 字）", len(body) > 30))

        # 标准窗口：不能有 transient（Windows 上会灰掉最大化按钮）、
        # 必须可缩放、F11 全屏后 Esc 只退出全屏而不是关窗口。
        checks.append(("说明窗口非 transient（标题栏最大化按钮可用）",
                       not bool(win.transient())))
        checks.append(("说明窗口可调整大小", win.resizable() == (True, True)
                       or win.resizable() == (1, 1)))
        try:
            win._toggle_fullscreen()
            _was_full = bool(getattr(win, "_full", False))
            win._on_escape()
            checks.append(("F11 全屏 / Esc 只退出全屏不关窗口",
                           _was_full and not win._full and win.winfo_exists()))
        except Exception as exc:                        # noqa: BLE001
            checks.append((f"F11 全屏 / Esc（异常：{exc}）", False))

        hits = search("状态")
        win._query.set("状态")
        win.update_idletasks()
        flat = win.tree.get_children()
        checks.append((f"搜索后变扁平列表（命中 {len(hits)}，节点 {len(flat)}）",
                       len(flat) == len(hits) and len(flat) > 0))

        win._clear_search()
        win.update_idletasks()
        checks.append(("清空搜索回到分区树",
                       len(win.tree.get_children()) == len(SECTIONS)))

        win._query.set("zzz不存在的词")
        win.update_idletasks()
        checks.append(("搜不到时正文给出提示（不是空白/上一条）",
                       "没有匹配" in win.text.get("1.0", "end")))
        win._query.set("")
        win.update_idletasks()

        # ---- 入口 ----
        # ① `open_help` 要能接"非 widget"的对象。`Viewer3D` 是普通类（只有
        #    `.root`），传 self 会让 `master.winfo_toplevel()` 抛 AttributeError；
        #    **tkinter 回调里的异常不弹窗**，用户看到的就是"按钮点了没反应"
        #    （2026-11-05 用户截图，3D 查看器侧栏那个按钮）。
        class _FakeViewer:                 # 只保留 Viewer3D 的接口特征
            def __init__(self, root) -> None:
                self.root = root

        from apps.help_window import open_help

        r2 = tk.Tk()
        r2.withdraw()
        try:
            open_help(_FakeViewer(r2), section="viewer")
            w2 = getattr(r2, "_help_window", None)
            ok2 = w2 is not None and w2.winfo_exists()
            checks.append(("open_help 兼容非 widget（只带 .root 的对象）", ok2))
            if ok2:
                sel2 = w2.tree.selection()
                checks.append(("  且定位到 3D 查看器分区",
                               bool(sel2) and "鼠标怎么用"
                               == w2.tree.item(sel2[0], "text")))
                w2.destroy()
        except Exception as exc:                        # noqa: BLE001
            checks.append((f"open_help 兼容非 widget（异常：{exc}）", False))
        finally:
            r2.destroy()

        # ② 状态栏是四个主窗口共用的，入口挂在这里 = 每个窗口都有。
        from apps._gui_widgets import StatusBar

        r3 = tk.Tk()
        r3.withdraw()
        try:
            sb = StatusBar(r3, help_section="viewer")
            sb.pack()
            r3.update_idletasks()
            hits = [c for c in sb.winfo_children()
                    if c.winfo_class() == "TButton"
                    and "操作说明" in str(c.cget("text"))]
            checks.append((f"状态栏带「? 操作说明」按钮（{len(hits)} 个）",
                           len(hits) == 1))
        finally:
            r3.destroy()
    except Exception as exc:                            # noqa: BLE001
        checks.append((f"说明窗口能建起来（异常：{exc}）", False))
    finally:
        if win is not None:
            try:
                win.destroy()
            except Exception:                           # noqa: BLE001
                pass
        try:
            root.destroy()
        except Exception:                               # noqa: BLE001
            pass
    if _i18n.language() != _help_old_lang:
        _i18n._current = _help_old_lang
    return checks


def _pair_check(base) -> bool:
    """`export_motion_pair` 端到端：造一个 traj3d.csv，导出后 raw 与源逐字节一致。"""
    import tempfile as _tf
    from apps._data_export import export_motion_pair
    d = base / "pair_src"
    (d / "derived").mkdir(parents=True)
    (d / "session.json").write_text(
        '{"schema":"sphere-stereo-session","version":1,"id":"pair_src"}', encoding="utf-8")
    rows = ["frame,t,matched,X,Y,Z",
            "0,0.000,1,0.0,0.0,0.60",
            "1,0.033,1,0.0,0.0,0.60",
            "2,0.066,1,0.0,0.0,0.60"]
    src = d / "derived" / "traj3d.csv"
    src.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with _tf.TemporaryDirectory() as td:
        got = export_motion_pair(d, td)
        if got is None:
            return False
        raw, motion = got
        # 校验必须在 TemporaryDirectory 仍存活时做（read_bytes 此时才读得到）。
        note = Path(td) / "数据说明.md"
        return (raw.name.endswith("_raw.csv") and motion.name.endswith("_motion.csv")
                and raw.read_bytes() == src.read_bytes() and motion.stat().st_size > 0
                and note.exists()
                and "数据变量说明" in note.read_text(encoding="utf-8"))


def _check_viewer_raw_and_explorer() -> list[tuple[str, bool]]:
    """8l. 三个新功能的独立回归护栏（2026-11-12）。

    三个功能都由别人写完，这里的任务是**独立证明它们真的能用**，不是"测试
    变绿"就算完：

      A. `_render_video` 从 cover-crop 改 contain —— 运行时真起 Viewer3D，
         把 pane 尺寸替换成 744×328，量 PhotoImage 的 583×328 与居中坐标；
         反证时把 `min` 改 `max`，运行时判据必须红（不能只 grep 源码）。
      B. `--video-source raw` —— 真开左右两个 cap，量两格 shape 与像素差，
         并造"无三维 JSON"的会话证明降级视图也开得了原始视频。
      C. `apps/_open_in_explorer.py` + 数据管理接入 —— 全部打桩，绝不真弹
         资源管理器（`gui_walkthrough.py:265` 靠 `os.startfile` 属性路径）。

    全部用 `tempfile.TemporaryDirectory()`，不依赖这台机器录过什么；真起 Tk
    窗口一律 withdraw + close/destroy，否则 Windows 上临时目录删不掉
    （WinError 32）。
    """
    checks: list[tuple[str, bool]] = []

    def _run() -> None:
        import ast
        import inspect
        import unittest.mock as _mock
        from types import SimpleNamespace

        import cv2
        import numpy as np
        import tkinter as tk

        import apps._open_in_explorer as oie
        import apps.data_manager as dmmod
        from apps.view_3d import Viewer3D, load_view

        v3d_src = (ROOT / "apps" / "view_3d.py").read_text(encoding="utf-8")
        dm_src = (ROOT / "apps" / "data_manager.py").read_text(encoding="utf-8")
        oie_src = (ROOT / "apps" / "_open_in_explorer.py").read_text(
            encoding="utf-8")

        def mk_video(path: Path, fill: int) -> None:
            """合成 128×64、6 帧、mp4v 的纯色视频（两路用不同填充色）。"""
            vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 5.0, (128, 64))
            if not vw.isOpened():
                raise RuntimeError(f"视频写不了：{path}")
            for i in range(6):
                vw.write(np.full((64, 128, 3), fill, np.uint8))
            vw.release()

        def mk_viz(sess: Path, *, n: int = 3) -> Path:
            (sess / "derived").mkdir(parents=True, exist_ok=True)
            pts = [{"xyz": [0.01 * i, 0.0, 1.0], "frame": i, "t": 0.01 * i}
                   for i in range(n)]
            (sess / "derived" / "visualization.json").write_text(
                json.dumps({"schema": "sphere-stereo-visualization",
                            "version": 1, "session": sess.name,
                            "trajectory": pts, "coordinate_system":
                            "left_camera", "unit": "m",
                            "stereo_mode": "parallel"}, ensure_ascii=False),
                encoding="utf-8")
            return sess / "derived" / "visualization.json"

        # ======== A. contain（运行时证明，不能只 grep 源码）========
        rv_src = inspect.getsource(Viewer3D._render_video)
        checks.append(("A1 contain 源码用 min(w/rw,h/rh)、没有 max",
                       "scale = min(w / rw, h / rh)" in rv_src
                       and "scale = max(w / rw, h / rh)" not in rv_src))
        checks.append(("A1 源码里没有 cover-crop 的中心裁切三行",
                       "img[" not in rv_src and "[y0:" not in rv_src
                       and "[x0:" not in rv_src))
        checks.append(("A1 源码里 create_image 居中 anchor=nw",
                       'create_image(x0, y0, anchor="nw", image=photo)'
                       in rv_src
                       and "(w - nw) // 2, (h - nh) // 2" in rv_src))

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            viz = mk_viz(base / "contain_sess")
            root = tk.Tk()
            root.withdraw()
            v = None
            try:
                # video_source="none" 避免真开视频；_render_video 是纯绘制，
                # 手动灌一帧 960×540（现场源帧尺寸）。
                v = Viewer3D(root, load_view(viz), video_source="none")
                v._left_raw = np.zeros((540, 960, 3), np.uint8)
                v._right_raw = None
                v.video_error = ""
                v.mono = False
                checks.append(("A1 Canvas 背景是深色 #0f172a（当黑边）",
                               str(v.video_l.cget("bg")).lower()
                               == "#0f172a"))

                _real_w = v.video_l.winfo_width
                _real_h = v.video_l.winfo_height
                _real_ci = v.video_l.create_image
                ci_seen: list[tuple[tuple, dict]] = []

                def _spy_ci(*a, **k):
                    ci_seen.append((a, k))
                    return _real_ci(*a, **k)

                try:
                    v.video_l.winfo_width = lambda: 744
                    v.video_l.winfo_height = lambda: 328
                    ci_seen.clear()
                    with _mock.patch.object(v.video_l, "create_image",
                                            _spy_ci):
                        v._render_video()
                    w1 = int(v._left_photo.width())
                    h1 = int(v._left_photo.height())
                    checks.append((
                        f"A2 contain 744×328 → 583×328（实测 {w1}×{h1}）",
                        w1 == 583 and h1 == 328))
                    checks.append(("A2 整帧装得下（宽≤744、高≤328，没裁）",
                                   w1 <= 744 and h1 <= 328))
                    ci_ok = (len(ci_seen) == 1
                             and ci_seen[0][0][0] == (744 - 583) // 2
                             and ci_seen[0][0][1] == (328 - 328) // 2
                             and ci_seen[0][1].get("anchor") == "nw")
                    checks.append(("A2 create_image 居中 (80,0)、anchor=nw",
                                   ci_ok))

                    # 换 pane 744×780（aspect 0.954）：这次该是 Y 方向留黑边。
                    v.video_l.winfo_width = lambda: 744
                    v.video_l.winfo_height = lambda: 780
                    ci_seen.clear()
                    with _mock.patch.object(v.video_l, "create_image",
                                            _spy_ci):
                        v._render_video()
                    w2 = int(v._left_photo.width())
                    h2 = int(v._left_photo.height())
                    ci2 = ci_seen[0][0] if ci_seen else ()
                    checks.append((
                        f"A2 contain 744×780 → {w2}×{h2}（X 撑满、Y 留黑边）",
                        w2 == 744 and h2 < 780 and len(ci2) >= 2
                        and ci2[0] == 0 and ci2[1] > 0))
                    checks.append(("A2 第二 pane 整帧仍装得下（≤744×780）",
                                   w2 <= 744 and h2 <= 780))
                finally:
                    v.video_l.winfo_width = _real_w
                    v.video_l.winfo_height = _real_h
            except Exception as exc:                        # noqa: BLE001
                checks.append((f"A2 contain 运行时（异常：{exc}）", False))
            finally:
                if v is not None:
                    v.close()
                else:
                    root.destroy()

        # ======== B. raw 视频源（双 cap / 不切分 / 无三维结果兜底）========
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            sess = base / "raw_sess"
            viz = mk_viz(sess, n=3)
            mk_video(sess / "left.mp4", 40)
            mk_video(sess / "right.mp4", 200)
            root = tk.Tk()
            root.withdraw()
            v = None
            try:
                v = Viewer3D(root, load_view(viz), video_source="raw")
                checks.append(("B4 raw 真开左右两个不同的 cap",
                               v.cap_l is not None and v.cap_r is not None
                               and v.cap_l is not v.cap_r))
                checks.append(("B4 两路都记到了帧数（cap_frames_l/r>0）",
                               v.cap_frames_l > 0 and v.cap_frames_r > 0))
                lshape = getattr(v._left_raw, "shape", None)
                rshape = getattr(v._right_raw, "shape", None)
                checks.append((
                    f"B4 两格都是完整 128×64 帧（shape={lshape}/{rshape}、无 w//2 切分）",
                    lshape == (64, 128, 3) and rshape == (64, 128, 3)))
                diff = (abs(float(v._left_raw.mean())
                            - float(v._right_raw.mean()))
                        if v._left_raw is not None
                        and v._right_raw is not None else 0.0)
                checks.append((f"B4 两格来自不同 cap（平均像素差 {diff:.1f}）",
                               diff > 10.0))
            except Exception as exc:                        # noqa: BLE001
                checks.append((f"B4 raw 双 cap（异常：{exc}）", False))
            finally:
                if v is not None:
                    v.close()
                else:
                    root.destroy()

        # B5：无三维 JSON 的会话（刚录完），降级视图仍开得了原始视频。
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            sess = base / "no3d_sess"
            sess.mkdir()
            (sess / "session.json").write_text(
                json.dumps({"schema": "sphere-stereo-session", "version": 1,
                            "id": "no3d_sess", "sync": {"offset": 0}},
                           ensure_ascii=False), encoding="utf-8")
            mk_video(sess / "left.mp4", 30)
            mk_video(sess / "right.mp4", 210)
            data = load_view(sess)
            checks.append(("B5 无三维 JSON → load_view 给降级视图（n=0/degraded）",
                           bool(data.get("degraded"))
                           and not data.get("trajectory")))
            root = tk.Tk()
            root.withdraw()
            v = None
            try:
                v = Viewer3D(root, data, video_source="raw")
                rsd = getattr(v, "_raw_session_dir", None)
                checks.append((
                    f"B5 _raw_session_dir 指向该会话目录（{rsd}）",
                    rsd is not None
                    and Path(rsd).resolve() == sess.resolve()))
                checks.append(("B5 降级视图仍打开左右两个 cap",
                               v.cap_l is not None and v.cap_r is not None
                               and v.cap_l.isOpened()
                               and v.cap_r.isOpened()))
                checks.append(("B5 降级视图也解码出两格完整帧",
                               getattr(v._left_raw, "shape", None)
                               == (64, 128, 3)
                               and getattr(v._right_raw, "shape", None)
                               == (64, 128, 3)))
            except Exception as exc:                        # noqa: BLE001
                checks.append((f"B5 无三维 JSON 的 raw 兜底（异常：{exc}）",
                               False))
            finally:
                if v is not None:
                    v.close()
                else:
                    root.destroy()

        # B6：video_source="none" 等价 --no-video，三个 cap 全是 None。
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            sess = base / "none_sess"
            viz = mk_viz(sess)
            mk_video(sess / "left.mp4", 40)
            mk_video(sess / "right.mp4", 200)
            root = tk.Tk()
            root.withdraw()
            v = None
            try:
                v = Viewer3D(root, load_view(viz), video_source="none")
                checks.append(("B6 none 模式 _left_raw/_right_raw 都是 None",
                               v._left_raw is None and v._right_raw is None))
                checks.append(("B6 none 模式 cap/cap_l/cap_r 全 None",
                               v.cap is None and v.cap_l is None
                               and v.cap_r is None))
            except Exception as exc:                        # noqa: BLE001
                checks.append((f"B6 none 模式（异常：{exc}）", False))
            finally:
                if v is not None:
                    v.close()
                else:
                    root.destroy()

        # B7：close() 后三个 cap 都释放，临时目录能删掉（WinError 32 反证）。
        try:
            with tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp)
                sess = base / "close_sess"
                viz = mk_viz(sess)
                mk_video(sess / "left.mp4", 40)
                mk_video(sess / "right.mp4", 200)
                root = tk.Tk()
                root.withdraw()
                v = None
                try:
                    v = Viewer3D(root, load_view(viz), video_source="raw")
                finally:
                    if v is not None:
                        v.close()
                        checks.append(("B7 close() 后三个 cap 都置 None",
                                       v.cap is None and v.cap_l is None
                                       and v.cap_r is None))
                    else:
                        root.destroy()
            checks.append(("B7 临时目录能删掉（cap 已释放，无 WinError 32）",
                           True))
        except Exception as exc:                            # noqa: BLE001
            checks.append((f"B7 close()/删临时目录（异常：{exc}）", False))

        # B8：CLI 与 __init__ 契约（inspect.signature，不只 grep）。
        sig = inspect.signature(Viewer3D.__init__)
        p_vs = sig.parameters.get("video_source")
        checks.append(("B8 Viewer3D.__init__ 的 video_source 有默认值 annotated",
                       p_vs is not None and p_vs.default == "annotated"
                       and p_vs.default is not inspect.Parameter.empty))
        v3d_tree = ast.parse(v3d_src)
        vs_call = None
        for node in ast.walk(v3d_tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == "--video-source"):
                vs_call = node
                break
        choices = default = None
        if vs_call is not None:
            for kw in vs_call.keywords:
                if (kw.arg == "choices"
                        and isinstance(kw.value, (ast.Tuple, ast.List))):
                    choices = [e.value for e in kw.value.elts
                               if isinstance(e, ast.Constant)]
                elif (kw.arg == "default"
                      and isinstance(kw.value, ast.Constant)):
                    default = kw.value.value
        checks.append((
            f"B8 CLI --video-source choices={choices} default={default}",
            vs_call is not None
            and choices == ["annotated", "raw", "none"]
            and default == "annotated"))
        checks.append(("B8 CLI --no-video 映射成 video_source=none",
                       'source = "none" if args.no_video else args.video_source'
                       in v3d_src))

        # ======== C. 新模块 + 数据管理接入（全打桩，绝不真弹资源管理器）=====
        oie_tree = ast.parse(oie_src)
        top_mods: set[str] = set()
        for node in oie_tree.body:
            if isinstance(node, ast.Import):
                for a in node.names:
                    top_mods.add(a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_mods.add(node.module.split(".")[0])
        stdlib = {"__future__", "os", "subprocess", "sys", "pathlib"}
        checks.append((
            f"C9 新模块顶层 import 只有 stdlib（{sorted(top_mods)}）",
            top_mods <= stdlib and "tkinter" not in top_mods
            and "apps" not in top_mods))

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            allowed = base / "allowed"
            allowed.mkdir()
            outside = base / "outside"
            outside.mkdir()
            a_file = allowed / "a.txt"
            a_file.write_text("x", encoding="utf-8")

            sf: list[str] = []
            popen: list[tuple[tuple, dict]] = []

            def _popen_spy(*a, **k):
                popen.append((a, k))
                return None

            def _startfile_rec(p, *a, **k):
                sf.append(str(p))

            def _raises(fn, *a, **k):
                try:
                    fn(*a, **k)
                except oie.OpenInExplorerError:
                    return True
                except Exception:                           # noqa: BLE001
                    return False
                return False

            try:
                # C10：os.startfile 首选、Popen 未被调用（gui_walkthrough 红线）
                sf.clear()
                popen.clear()
                with _mock.patch.object(os, "startfile", _startfile_rec,
                                        create=True):
                    with _mock.patch.object(oie.subprocess, "Popen",
                                            _popen_spy):
                        ret = oie.open_in_explorer(allowed,
                                                   allowed_root=allowed)
                checks.append(("C10 open_in_explorer 返回 True", ret is True))
                checks.append(("C10 os.startfile 收到的正是该目录",
                               sf == [str(allowed.resolve())]))
                checks.append(("C10 subprocess.Popen 未被调用（打桩红线）",
                               popen == []))

                # C11：非法路径抛异常；传文件自动取 parent。
                sf.clear()
                popen.clear()
                with _mock.patch.object(os, "startfile", _startfile_rec,
                                        create=True):
                    with _mock.patch.object(oie.subprocess, "Popen",
                                            _popen_spy):
                        bad_missing = _raises(oie.open_in_explorer,
                                              allowed / "nope")
                        bad_outside = _raises(oie.open_in_explorer, outside,
                                              allowed_root=allowed)
                        sf.clear()
                        ret_file = oie.open_in_explorer(
                            a_file, allowed_root=allowed)
                checks.append((
                    "C11 不存在路径/越界路径都抛 OpenInExplorerError",
                    bad_missing and bad_outside))
                checks.append((
                    "C11 传文件自动取 parent（startfile 收到目录）",
                    ret_file is True and sf == [str(allowed.resolve())]))

                # C12：reveal_in_explorer 的 explorer /select, 参数
                popen.clear()
                with _mock.patch.object(oie.subprocess, "Popen",
                                        _popen_spy):
                    ret_reveal = oie.reveal_in_explorer(
                        a_file, allowed_root=allowed)
                a0 = popen[0][0] if popen else ()
                kw = popen[0][1] if popen else {}
                argv = a0[0] if a0 else None
                exp = f"/select,{a_file.resolve()}"
                checks.append((
                    f"C12 reveal 参数是 [explorer, /select,<绝对路径>]（{argv}）",
                    ret_reveal is True and isinstance(argv, list)
                    and len(argv) == 2 and argv[0] == "explorer"
                    and argv[1] == exp))
                checks.append((
                    "C12 /select 逗号后无空格、无 shell=True",
                    isinstance(argv, list) and len(argv) == 2
                    and argv[1].startswith("/select,")
                    and not argv[1].startswith("/select, ")
                    and "shell" not in kw))

                # C13：os.startfile 缺失 → 走 explorer Popen 降级
                popen.clear()
                with _mock.patch.object(os, "startfile", None, create=True):
                    with _mock.patch.object(oie.subprocess, "Popen",
                                            _popen_spy):
                        ret_fb = oie.open_in_explorer(
                            allowed, allowed_root=allowed)
                fb_argv = popen[0][0][0] if popen else None
                checks.append((
                    "C13 os.startfile 缺失 → 降级 [explorer, <目录>]",
                    ret_fb is True
                    and fb_argv == ["explorer", str(allowed.resolve())]))
            except Exception as exc:                        # noqa: BLE001
                checks.append((f"C10–C13 新模块打桩（异常：{exc}）", False))

        # C14：真起 DataManager，选中假会话 → 打开 it.path + allowed_root。
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            sess = base / "sess1"
            sess.mkdir()
            dm = None
            try:
                dm = dmmod.DataManager(base)
                dm.withdraw()
                dm.update()
                fake = SimpleNamespace(key="sess1", path=sess, id="sess1")
                dm.items = [fake]
                if not dm.tree.exists(fake.key):
                    dm.tree.insert("", "end", iid=fake.key, text=fake.id)
                dm.tree.focus(fake.key)

                real_open = dmmod.open_in_explorer
                seen: list[tuple[Path, object]] = []
                sf2: list[str] = []
                infos: list = []

                def _spy_open(path, *, allowed_root=None):
                    seen.append((Path(path), allowed_root))
                    return real_open(path, allowed_root=allowed_root)

                old_open = dmmod.open_in_explorer
                old_si = dmmod.messagebox.showinfo
                old_se = dmmod.messagebox.showerror
                dmmod.open_in_explorer = _spy_open
                dmmod.messagebox.showinfo = lambda *a, **k: infos.append(a)
                dmmod.messagebox.showerror = lambda *a, **k: None
                try:
                    with _mock.patch.object(
                            os, "startfile",
                            lambda p, *a, **k: sf2.append(str(p)),
                            create=True):
                        dm._open_session_in_explorer()
                        n_after = len(sf2)
                        dm.items = []
                        dm.selected.clear()
                        infos.clear()
                        dm._open_session_in_explorer()
                        no_sel = len(sf2) == n_after
                finally:
                    dmmod.open_in_explorer = old_open
                    dmmod.messagebox.showinfo = old_si
                    dmmod.messagebox.showerror = old_se
                checks.append((
                    "C14 _open_session_in_explorer 打开 it.path 且 allowed_root==root_dir",
                    len(seen) == 1
                    and seen[0][0].resolve() == sess.resolve()
                    and seen[0][1] == dm.root_dir
                    and sf2 == [str(sess.resolve())]))
                checks.append(("C14 无选中 → 弹提示、不调用 startfile",
                               bool(infos) and no_sel))
            except Exception as exc:                        # noqa: BLE001
                checks.append(
                    (f"C14 DataManager 打开会话文件夹（异常：{exc}）", False))
            finally:
                if dm is not None:
                    dm._stop_polling()
                    dm.destroy()

        # C15：_open_viewer 的 raw / 默认 / 无视频三种 argv（打桩 Popen）。
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            s_raw = base / "s_raw"
            s_raw.mkdir()
            (s_raw / "left.mkv").write_bytes(b"")
            s_ann = base / "s_ann"
            (s_ann / "derived").mkdir(parents=True)
            (s_ann / "derived" / "annotated.mp4").write_bytes(b"")
            s_bare = base / "s_bare"
            s_bare.mkdir()
            dm = None
            try:
                dm = dmmod.DataManager(base)
                dm.withdraw()
                dm.update()
                calls: list[list[str]] = []
                asks: list = []
                infos2: list = []
                old_popen = dmmod.subprocess.Popen
                old_ask = dmmod.messagebox.askyesno
                old_si = dmmod.messagebox.showinfo
                old_se = dmmod.messagebox.showerror
                dmmod.subprocess.Popen = lambda args: calls.append(args)
                dmmod.messagebox.askyesno = lambda *a, **k: (
                    asks.append(a), True)[1]
                dmmod.messagebox.showinfo = lambda *a, **k: infos2.append(a)
                dmmod.messagebox.showerror = lambda *a, **k: None

                def _focus(it_path: Path, key: str) -> None:
                    fake = SimpleNamespace(key=key, path=it_path, id=key)
                    dm.items = [fake]
                    if not dm.tree.exists(key):
                        dm.tree.insert("", "end", iid=key, text=key)
                    dm.tree.focus(key)

                def _run_viewer(mode: str, it_path: Path,
                                key: str) -> list[list[str]]:
                    _focus(it_path, key)
                    calls.clear()
                    asks.clear()
                    if mode == "raw":
                        dm._open_viewer(mode="raw")
                    else:
                        dm._open_viewer()
                    return list(calls)

                try:
                    argv_raw = _run_viewer("raw", s_raw, "s_raw")
                    argv_ann = _run_viewer("annotated", s_ann, "s_ann")
                    argv_bare = _run_viewer("annotated", s_bare, "s_bare")
                    _focus(s_bare, "s_bare")
                    calls.clear()
                    infos2.clear()
                    dm._open_viewer(mode="raw")
                    raw_bare_ok = (calls == [] and bool(infos2))
                finally:
                    dmmod.subprocess.Popen = old_popen
                    dmmod.messagebox.askyesno = old_ask
                    dmmod.messagebox.showinfo = old_si
                    dmmod.messagebox.showerror = old_se

                def _first(argv: list[list[str]]) -> list[str]:
                    return argv[0] if argv else []

                a_raw = _first(argv_raw)
                a_ann = _first(argv_ann)
                a_bare = _first(argv_bare)
                has_raw = ("--video-source" in a_raw
                           and a_raw.index("--video-source") + 1 < len(a_raw)
                           and a_raw[a_raw.index("--video-source") + 1]
                           == "raw")
                checks.append((
                    f"C15 raw + left.mkv → --video-source raw 且不含 --no-video（尾 {a_raw[3:]}）",
                    len(argv_raw) == 1 and has_raw
                    and "--no-video" not in a_raw))
                checks.append((
                    "C15 默认 + 有 annotated → 与改动前一致（无视频参数）",
                    len(argv_ann) == 1 and len(a_ann) == 3
                    and "--no-video" not in a_ann
                    and "--video-source" not in a_ann))
                checks.append((
                    "C15 默认 + 无 annotated/无 raw → askyesno + --no-video",
                    len(argv_bare) == 1
                    and "--no-video" in a_bare and bool(asks)))
                checks.append(("C15 raw + 无原始视频 → 只弹提示、不 Popen",
                               raw_bare_ok))
            except Exception as exc:                        # noqa: BLE001
                checks.append((f"C15 _open_viewer argv（异常：{exc}）", False))
            finally:
                if dm is not None:
                    dm._stop_polling()
                    dm.destroy()

        # C16：源码契约：两个新按钮 / 不叫 _options / open_output_folder 用新模块。
        checks.append(("C16 两个新按钮文案都在 data_manager.py",
                       'text="查看原始视频"' in dm_src
                       and 'text="在文件资源管理器中打开"' in dm_src))
        checks.append(("C16 DataManager 没有遮蔽 tk.Misc._options",
                       "_options" not in dmmod.DataManager.__dict__))
        checks.append((
            "C16 open_output_folder 调新模块且 allowed_root=None",
            "from apps._open_in_explorer import" in dm_src
            and "open_in_explorer(folder, allowed_root=None)" in dm_src))

        # ======== X. 跨模块契约（2026-11-12 team-lead 追加裁决）========
        # X1 `--no-video` 显式关闭优先：这个优先级的**唯一实现在 CLI 层**，
        # 不在 Viewer3D（team-lead 裁决"保持现状、不改 view_3d.py"）。
        #   apps/view_3d.py:2323  source = "none" if args.no_video else args.video_source
        #   apps/view_3d.py:2324  Viewer3D(..., auto_video=not args.no_video,
        #                                video_source=source)
        # 裁决理由（写进注释以免将来被"好心"改偏）：
        #  理由1 两个参数同时给时 CLI 先把 video_source 折成 "none" 再构造
        #        Viewer3D，所以它收到的是 none 不是 raw；`auto_video` 只管
        #        `_resolve_video` 的 annotated 自动发现路径（_apply:606），
        #        raw 分支（_apply:633）不看它是**设计**，不是漏判。
        #  理由2 若让 raw 分支也检查 auto_video 会制造两份真相（CLI 折一次、
        #        Viewer 再折一次），并打掉合法组合 `auto_video=False,
        #        video_source="raw"`（调用方想表达"别自动找 annotated.mp4，
        #        但我要显式看原始双路"）。实测该组合双 cap 打开、两格有帧、
        #        video_disabled=False，不该被 --no-video 语义误伤。
        #  理由3 别的调用点手工同时拼两个参数是**调用方的错、应在调用点修**，
        #        查看器不猜意图；优先级只在 CLI 层实现一次 = "唯一产地"规矩。
        # data_manager 的 `if mode == "raw": ... elif not annotated.exists():`
        # 保证正常路径两者互斥。下面四条判据：① 源码守 :2323 映射；② AST 守
        # --no-video 的 store_true/默认 False；③ 端到端复刻 :2323-2324 证明
        # Viewer3D 真的关掉视频；④ 直传 auto_video=False+raw 仍开双 cap，
        # 把"开关在 CLI 层、Viewer 不猜"钉成显式契约。
        checks.append((
            "X1 CLI 里 --no-video 优先于 --video-source（source 折成 none 并传给 Viewer3D）",
            'source = "none" if args.no_video else args.video_source'
            in v3d_src and "video_source=source" in v3d_src))
        novideo_call = None
        for node in ast.walk(v3d_tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == "--no-video"):
                novideo_call = node
                break
        nv_kw = ({kw.arg: kw.value for kw in novideo_call.keywords}
                 if novideo_call is not None else {})
        nv_action = nv_kw.get("action")
        nv_default = nv_kw.get("default")
        checks.append((
            "X1 --no-video 仍是 action=store_true 且默认 False（未被改偏）",
            novideo_call is not None
            and isinstance(nv_action, ast.Constant)
            and nv_action.value == "store_true"
            and (nv_default is None
                 or (isinstance(nv_default, ast.Constant)
                     and nv_default.value is False))))

        # ③ 端到端：逐字复刻 CLI :2323-2324 的映射，验证 Viewer3D 真的关掉视频。
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            sess = base / "x1_sess"
            viz = mk_viz(sess)
            mk_video(sess / "left.mp4", 40)
            mk_video(sess / "right.mp4", 200)
            root = tk.Tk()
            root.withdraw()
            v = None
            try:
                args_no_video = True
                args_video_source = "raw"
                source = ("none" if args_no_video else args_video_source)
                v = Viewer3D(root, load_view(viz),
                             auto_video=not args_no_video,
                             video_source=source)
                checks.append((
                    "X1 运行时：--no-video + --video-source raw → 三 cap 全 None、两格 None",
                    v.video_source == "none"
                    and v.cap is None and v.cap_l is None and v.cap_r is None
                    and v._left_raw is None and v._right_raw is None))
            except Exception as exc:                        # noqa: BLE001
                checks.append((f"X1 运行时（异常：{exc}）", False))
            finally:
                if v is not None:
                    v.close()
                else:
                    root.destroy()

        # ④ 契约边界：绕过 CLI 直传 auto_video=False + raw 仍开双 cap。
        # 这不是缺陷，是"查看器不猜、开关在 CLI 层"的现状契约；若有人把
        # auto_video 下沉进 raw 分支，这条会红，提醒他先改上面的裁决。
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            sess = base / "x1b_sess"
            viz = mk_viz(sess)
            mk_video(sess / "left.mp4", 40)
            mk_video(sess / "right.mp4", 200)
            root = tk.Tk()
            root.withdraw()
            v = None
            try:
                v = Viewer3D(root, load_view(viz), auto_video=False,
                             video_source="raw")
                checks.append((
                    "X1 契约边界：auto_video 只管 annotated，直传 raw 仍开双 cap（查看器不猜）",
                    v.cap_l is not None and v.cap_r is not None
                    and v._left_raw is not None and v._right_raw is not None
                    and v.video_error == ""))
            except Exception as exc:                        # noqa: BLE001
                checks.append((f"X1 契约边界（异常：{exc}）", False))
            finally:
                if v is not None:
                    v.close()
                else:
                    root.destroy()

        # X2 两端原始视频文件名集合一致。
        # 2026-11-15：文件名清单已收归 `sphere_tracker/video_files.py`（唯一产地），
        # 所以这里不再从两个模块**各自**抽元组比集合 —— 那正是"两份手写清单"的
        # 形状。改为守"两端都只 import、都不再写字面量"：集合相等由唯一产地直接
        # 保证，mkv 优先也在那一条常量里（以前这两条判据是分别抽两处元组比出来的，
        # 搬家后必须跟着搬，否则测试会替旧结构"守灵"）。
        from sphere_tracker.video_files import (                 # noqa: PLC0415
            ALL_RAW_VIDEO_NAMES, LEFT_VIDEO_NAMES, RIGHT_VIDEO_NAMES,
        )

        RAW_NAME_SET = set(ALL_RAW_VIDEO_NAMES)

        def _const_tuple(node):
            if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
                return [e.value for e in node.elts
                        if isinstance(e, ast.Constant)]
            return None

        def _find_func(tree, name):
            return next((n for n in ast.walk(tree)
                         if isinstance(n, ast.FunctionDef)
                         and n.name == name), None)

        def _literal_raw_names(func) -> list[str]:
            """函数体里手写的原始录像文件名（列表/元组/集合字面量）。"""
            if func is None:
                return ["<函数不存在>"]
            got: list[str] = []
            for node in ast.walk(func):
                for name in _const_tuple(node) or []:
                    if isinstance(name, str) and name in RAW_NAME_SET:
                        got.append(name)
            return got

        def _called_names(func) -> set[str]:
            """函数体里调用的裸函数名（`pick_left_video(...)` → `pick_left_video`）。"""
            if func is None:
                return set()
            return {n.func.id for n in ast.walk(func)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}

        dm_tree = ast.parse(dm_src)
        rv_func = _find_func(v3d_tree, "_resolve_raw_videos")
        ov_func = _find_func(dm_tree, "_open_viewer")
        viewer_lits = _literal_raw_names(rv_func)
        dm_lits = _literal_raw_names(ov_func)
        checks.append((
            f"X2 唯一产地：两端都不再手写文件名"
            f"（viewer 字面量={sorted(set(viewer_lits))} / dm 字面量={sorted(set(dm_lits))}）",
            RAW_NAME_SET == {"left.mkv", "right.mkv", "left.mp4", "right.mp4",
                             "left.avi", "right.avi"}
            and not viewer_lits and not dm_lits
            and {"pick_left_video", "pick_right_video"} <= _called_names(rv_func)
            and "has_any_raw_video" in _called_names(ov_func)))
        checks.append((
            f"X2 查看器 mkv 优先、mp4 其次、avi 兜底（LEFT={LEFT_VIDEO_NAMES} / "
            f"RIGHT={RIGHT_VIDEO_NAMES}）",
            tuple(LEFT_VIDEO_NAMES) == ("left.mkv", "left.mp4", "left.avi")
            and tuple(RIGHT_VIDEO_NAMES) == ("right.mkv", "right.mp4", "right.avi")))

        # X2 运行时：只有 right.mp4 的单路会话（按钮"至少一个存在就放行"安全）。
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            sess = base / "x2_sess"
            viz = mk_viz(sess)
            mk_video(sess / "right.mp4", 200)
            root = tk.Tk()
            root.withdraw()
            v = None
            try:
                v = Viewer3D(root, load_view(viz), video_source="raw")
                checks.append((
                    "X2 运行时：只有 right.mp4 → 不崩 / cap_l None / cap_r 打开 / 右格有帧 / 无 error",
                    v.cap_l is None and v.cap_r is not None
                    and v.cap_r.isOpened()
                    and v._left_raw is None and v._right_raw is not None
                    and v.video_error == ""))
            except Exception as exc:                        # noqa: BLE001
                checks.append((f"X2 运行时（异常：{exc}）", False))
            finally:
                if v is not None:
                    v.close()
                else:
                    root.destroy()

    # ---- 两种新标定方式（轴向大板 / 中轴 90°）在 GUI 的接入 ----
    # 这两条路各有自己的 argv 函数和窗口专属控件；判据钉三件事：
    #   ① 下拉候选按窗口模式过滤（轴向不串进大角度窗、反之亦然）；
    #   ② argv 脏数据不拼进去（"abc" 基线曾让 argparse exit 2）；
    #   ③ device90 分支真的不带 ③④ 参数（否则"省两步"是假的）。
    print("----- 新标定方式：轴向大板 / 中轴 90° 的 GUI 接入 -----")
    try:
        import ast as _ast2
        from apps import _actions as _act
        from apps._gui_stereo import (METHODS as _M2,
                                      PICKER_METHODS as _PM2,
                                      picker_methods_for as _pmf)

        _b2 = {"cols": "8", "rows": "5", "square": "25", "target": "40"}
        _ax = _act.calibrate_axial_args(root="X", board=_b2,
                                        assume_parallel=True,
                                        expected_baseline_mm="66.6")
        checks.append((
            "新标定① 轴向 argv：独立脚本 + --root + 0.05 容差 + assume_parallel",
            len(_ax) >= 2 and _ax[1].endswith("calibrate_axial.py")
            and "--root" in _ax and "X" in _ax
            and "--baseline-tolerance" in _ax and "0.05" in _ax
            and "--assume-parallel" in _ax))
        _ax_off = _act.calibrate_axial_args(root="X", board=_b2)
        checks.append((
            "新标定② 轴向默认不强制 R=I（不传 --assume-parallel）",
            "--assume-parallel" not in _ax_off))
        _d90 = _act.calibrate_device90_args(
            left_dir="L", right_dir="R", intrinsics="I.npz", board=_b2,
            rot_axis="x", rot_angle="90", z_a_mm="600",
            expected_baseline_mm="abc")
        checks.append((
            "新标定③ 中轴90° argv：独立脚本 + --rot-axis x + --z-a",
            len(_d90) >= 2 and _d90[1].endswith("calibrate_device90.py")
            and "--rot-axis" in _d90 and "x" in _d90
            and "--z-a" in _d90 and "600" in _d90))
        checks.append((
            "新标定④ 脏基线不拼进 argv（abc → 不出现 --expected-baseline-mm）",
            "--expected-baseline-mm" not in _d90))
        checks.append((
            "新标定⑤ 中轴90° 不带 ③④ 参数（ref-pair / pair-frame 都 absent）",
            not any(k in _d90 for k in
                    ("--ref-pair-left", "--ref-pair-right", "--pair-frame"))))

        checks.append((
            "新标定⑥ 下拉候选按窗口模式过滤（轴向只进平行、中轴90°只进大角度）",
            _pmf("parallel") == ("joint", "axial")
            and _pmf("angled") == ("chain", "chain_dual", "device90")))
        checks.append((
            "新标定⑦ METHODS 含两种新方法且全集 PICKER_METHODS 未改（下游 choices 不漂）",
            "axial" in _M2 and "device90" in _M2
            and tuple(_PM2) == ("joint", "chain", "chain_dual")))

        _gsrc = (Path(__file__).resolve().parent.parent
                 / "apps" / "_gui_stereo.py").read_text(encoding="utf-8")
        _gtree = _ast2.parse(_gsrc)
        _start_fn = next((n for n in _ast2.walk(_gtree)
                          if isinstance(n, _ast2.FunctionDef)
                          and n.name == "_start_calibrate"), None)
        _steps_fn = next((n for n in _ast2.walk(_gtree)
                          if isinstance(n, _ast2.FunctionDef)
                          and n.name == "_chain_steps"), None)
        _start_src = _ast2.get_source_segment(_gsrc, _start_fn) if _start_fn else ""

        _ax_branches = [_ast2.get_source_segment(_gsrc, n) or ""
                        for n in _ast2.walk(_start_fn)
                        if isinstance(n, _ast2.If) and isinstance(n.test, _ast2.Compare)
                        and isinstance(n.test.left, _ast2.Name)
                        and n.test.left.id == "method"]
        _d90_start = next((s for s in _ax_branches
                           if 'method == "device90"' in s), "")
        checks.append((
            "新标定⑧ _start_calibrate 有 axial / device90 两条分派分支",
            any('method == "axial"' in s for s in _ax_branches)
            and bool(_d90_start)))
        checks.append((
            "新标定⑨ device90 分派不引用 ③④ 目录（ref_pair / pair）",
            bool(_d90_start)
            and "ref_pair" not in _d90_start
            and "d90[\"pair\"]" not in _d90_start))
        checks.append((
            "新标定⑩ device90 分派调用了 calibrate_device90_args",
            "calibrate_device90_args" in _d90_start))
        checks.append((
            "新标定⑪ _chain_steps 有 device90 两步分支（不会走单板三步）",
            _steps_fn is not None
            and 'method_key() == "device90"' in _ast2.get_source_segment(_gsrc, _steps_fn)))
    except Exception as exc:                                # noqa: BLE001
        checks.append((f"新标定判据组整体异常（{exc}）", False))

    try:
        _run()
    except Exception as exc:                                # noqa: BLE001
        checks.append((f"8l 判据组整体异常（{exc}）", False))
    return checks


def main() -> int:
    """把跨进程账本指到临时目录再跑主体（try/finally 覆盖早退与异常）。

    本测试会真起 DataManager 的导出/查看器任务，它们走
    ``apps._task_registry.register_start/end``，写的是模块全局
    ``REGISTRY_PATH``。不注入的话，隐私测试的假文件名
    ``zz_privacy_positive.zip`` 会被写进真实 ``logs/tasks/_registry.jsonl``。

    两条注入缺一不可：**环境变量**给子进程（新解释器 import 时读到
    ``SPHERE_TASK_REGISTRY``），**模块常量**给已 import 的本进程调用方。
    临时目录随 with 退出自动清理，两者都在 finally 里还原；即使判据中途
    抛异常也不会把测试残留留在真实运行账本里。
    """
    import os

    import apps._task_registry as _reg

    with tempfile.TemporaryDirectory(prefix="launcher_params_registry_") as td:
        path = Path(td) / "_registry.jsonl"
        real = _reg.REGISTRY_PATH
        old_env = os.environ.get("SPHERE_TASK_REGISTRY")
        _reg.REGISTRY_PATH = path
        os.environ["SPHERE_TASK_REGISTRY"] = str(path)
        try:
            return _main_impl()
        finally:
            _reg.REGISTRY_PATH = real
            if old_env is None:
                os.environ.pop("SPHERE_TASK_REGISTRY", None)
            else:
                os.environ["SPHERE_TASK_REGISTRY"] = old_env
            n = 0
            if path.exists():
                try:
                    n = len(path.read_text(encoding="utf-8",
                                           errors="replace").splitlines())
                except OSError:
                    pass
            print(f"[账本隔离] 临时账本 {path.name} 共 {n} 行（真实 logs 未写入）")


if __name__ == "__main__":
    raise SystemExit(main())
