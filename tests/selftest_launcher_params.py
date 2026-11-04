"""启动器标定参数面板自测：读取、校验、以及"默认就是正式档"。

为什么需要它
------------
用户要求的"标定要收足够多对 + UI 里能设标定参数"落地时，暴露了一个**真缺陷**：

`启动程序.py` 原来把板参数**硬编码**在命令里（`--cols 8 --rows 5 --square 25`），
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

LAUNCHER = ROOT / "启动程序.py"


def _load_launcher():
    """把启动器作为模块加载（不执行 mainloop）。

    ⚠️ **必须先注册进 `sys.modules`**：启动器里有 `@dataclass`，
    而 dataclasses 解析注解时要按 `cls.__module__` 去 `sys.modules` 里找模块，
    找不到就抛 `AttributeError: 'NoneType' object has no attribute '__dict__'`。
    直接 `exec_module` 而不注册，就是这个错。
    """
    sys.argv = ["启动程序.py"]
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

    from apps.view_3d import Viewer3D, load_view, scan_sources

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

            # ---- 8d-3. 缩放（改分度值）+ 可拖动四宫格（2026-11-04）----
            # 用户："增加放大缩小（改变分度值），四宫格做成可以拖动放大缩小"。
            # 钉两件事：① 缩放真的改坐标轴范围（不是只改一个数字），且旋转
            # 与缩放解耦（缩放不碰视角）；② 四宫格是可拖的 PanedWindow。
            import numpy as _np2
            from tkinter import ttk as _ttk
            xr0 = tuple(v.ax.get_xlim())
            v._zoom_in()
            xr1 = tuple(v.ax.get_xlim())
            checks.append(("放大后坐标轴范围收窄（分度值变密）",
                           (xr1[1] - xr1[0]) < (xr0[1] - xr0[0]) and v.zoom > 1.0))
            el_before, az_before = v.ax.elev, v.ax.azim
            v._zoom_out()
            v._zoom_out()
            checks.append(("缩小后范围变宽、且旋转角没被缩放带跑",
                           (v.ax.get_xlim()[1] - v.ax.get_xlim()[0])
                           > (xr0[1] - xr0[0])
                           and abs(v.ax.elev - el_before) < 1e-6
                           and abs(v.ax.azim - az_before) < 1e-6))
            v.zoom = 50.0
            v._zoom_in()
            checks.append(("放大到上限被钳住（不会放成一条线）",
                           v.zoom <= 20.0 + 1e-6))
            v._reset_view()
            checks.append(("重置视角同时重置缩放（zoom 回 1.0）",
                           v.zoom == 1.0 and v.zoom_lbl.cget("text") == "1.0x"))
            checks.append(("四宫格外层是可拖动的 PanedWindow",
                           isinstance(v._grid, _ttk.PanedWindow)))
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


def main() -> int:
    print("=" * 70)
    print("启动器标定参数面板自测（读取 / 校验 / 默认正式档）")
    print("=" * 70)

    if tk is None:
        print("  [跳过] 这台机器没有 tkinter —— 界面测试无法进行。")
        print("         （`启动程序.bat gui` 在这种情况下会自动改用命令行界面。）")
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
    launcher_src = (ROOT / "启动程序.py").read_text(encoding="utf-8")
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
    mod_src = (ROOT / "启动程序.py").read_text(encoding="utf-8")
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
         and "点选球" in (ROOT / "启动程序.py").read_text(encoding="utf-8")),
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
                      with_seeds: bool, with_traj: bool = False) -> Path:
        d = base / f"s_{frames_l}_{frames_r}_{int(with_seeds)}_{int(with_traj)}"
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
            (d / "derived" / "traj3d.json").write_text(json.dumps({
                "schema": "sphere-stereo-trajectory", "version": 1,
                "samples": [
                    {"matched": True, "xyz": [0.1, 0.05, 0.25], "residual_px": 1.5},
                    {"matched": False},
                ],
            }), encoding="utf-8")
        return d

    with tempfile.TemporaryDirectory(prefix="dsh_wb_") as tmp:
        base = Path(tmp)
        good = summarize_session(_make_session(base, 302, 303, True, True))
        # 2026-11-03 改：小差异（302/304、364/362）**不再算坏** —— 管线只是"取到短的那路结束"。
        # 只有比例上明显掉帧（302/380 ≈ 26%）才判死。
        small = summarize_session(_make_session(base, 364, 362, False, False))
        bad = summarize_session(_make_session(base, 302, 380, False, False))
        rec = summarize_session(_make_session(base, 302, 303, False, False))
        # 目录 mtime 不能决定默认会话：处理旧会话会刷新 mtime。
        from apps._workbench import scan_sessions
        newer = base / "20261021-120000"
        older = base / "20261020-120000"
        for d in (newer, older):
            (d / "derived").mkdir(parents=True)
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

        # 三区可拖（2026-11-03）：两个 PanedWindow（vertical + horizontal）各有 2 个 pane，
        # 且 sash 位置可写（sashpos 设置后值变化 = 拖动机制在工作）。
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
            _ok_struct = (len(_panes) == 2
                          and all(len(_p.panes()) == 2 for _p in _panes))
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
        checks_extra.append(("三区可拖：vertical+horizontal 各 2 pane，sash 可写", pane_ok))

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

    ok = all(v for _n, v in results)
    bad = [n for n, v in results if not v]
    if bad:
        print(f"\n不达标 {len(bad)} 项：")
        for n in bad:
            print(f"  · {n}")
    print("\n判定结果：" + ("全部通过" if ok else "有不达标项"))
    return 0 if ok else 1


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


if __name__ == "__main__":
    raise SystemExit(main())
