"""离线 3D 轨迹查看器：读取标准 `visualization.json`，交互播放。

**单目 / 双目都能看**（2026-10-29 增）。两种数据的 schema 不同名、
尺度来源也不同，但 `trajectory` 的字段形状一致（`frame` / `t` / `xyz`），
所以画图逻辑只有一份：

    sphere-stereo-visualization   双目：深度靠**视差**（Z 是真测量）
    sphere-stereo-trajectory      双目旧格式（samples）
    sphere-mono-visualization     单目：深度靠**球径先验** Z=f·D/(2r)

单目那条必须在界面上常驻一句话说明尺度来历 —— 让人拿它当视差测量用，
比不给他这个功能更糟。所以打开单目时标题栏带 `[单目]`、图下方常驻一行
「深度由球径先验得到（D=… mm），不是视差测量」。

用法：
    python apps/view_3d.py data/stereo/20261019-221550
    python apps/view_3d.py data/mono/20261029-205522
    python apps/view_3d.py data/stereo/20261019-221550 --check

只依赖 matplotlib + Tkinter + numpy（环境已有），不依赖网络。平滑用
`stereo.motion.smooth_positions` —— **不在这里再写一份**（两处平滑 = 图与 CSV
两个结果，这个项目已经因此返工过）。scipy 缺失时它返回 `valid` 全 False，
查看器照常显示原始轨迹。

坐标系由 `visualization.json` 的 `coordinate_system` 决定：**X 右 / Y 下 / Z 前**
（单位米）；parallel 是校正后左相机系，angled 是原始左相机系，
mono-camera 是单目相机系（主点在画面中心或标定主点）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d import proj3d
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox

ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 数据根目录只有 `sphere_tracker/paths.py` 说了算：以前 `data/sessions` 这个
# 字面量散在 8 处，于是"会话"在磁盘上指了两个东西（见该模块 docstring）。
from sphere_tracker.paths import MONO_ROOT, STEREO_ROOT, stereo_roots  # noqa: E402
# 平滑**只有这一处实现**（`stereo/motion.py` 定稿）：查看器要用，就 import，
# 不许在这里再复制一份窗口/阶数的代码。默认 0.20 s 是实测扫出来的，别改。
from stereo.motion import smooth_positions, split_segments  # noqa: E402

VIS_SCHEMA = "sphere-stereo-visualization"
TRAJ_SCHEMA = "sphere-stereo-trajectory"
MONO_SCHEMA = "sphere-mono-visualization"
SCHEMAS = (VIS_SCHEMA, TRAJ_SCHEMA, MONO_SCHEMA)


def resolve_json(path: str | Path) -> Path:
    p = Path(path)
    if p.is_dir():
        for name in ("visualization.json", "traj3d.json"):
            cand = p / "derived" / name
            if cand.exists():
                return cand
        raise FileNotFoundError(f"目录下没有 derived/visualization.json：{p}")
    if not p.exists():
        raise FileNotFoundError(f"文件不存在：{p}")
    return p


def is_mono(data: dict) -> bool:
    """这份数据是不是单目的。

    认两个标记：schema 是单目那个（正常情况），或 ``mode == "mono"``
    （别人手写的文件、将来加了新 schema 但语义仍是单目时）。
    **不靠目录名猜**：单目会话可能被拷到 `data/` 下，路径不能当类型用。
    """
    return data.get("schema") == MONO_SCHEMA or data.get("mode") == "mono"


def load_view(path: str | Path) -> dict:
    p = resolve_json(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    schema = data.get("schema")
    if schema not in SCHEMAS:
        raise ValueError(f"不是 {'/'.join(SCHEMAS)}：{schema}")
    if int(data.get("version", 0)) != 1:
        raise ValueError(f"不支持的 version：{data.get('version')}")
    points = data.get("trajectory")
    if not isinstance(points, list):
        points = [s for s in (data.get("samples") or []) if s.get("matched") and s.get("xyz")]
    points = [p for p in points if isinstance(p.get("xyz"), (list, tuple)) and len(p["xyz"]) == 3]
    if not points:
        raise ValueError("没有可用的三维点（trajectory/samples 都为空）")
    data = dict(data)
    data["trajectory"] = points
    data["_path"] = str(p)
    return data


def bounds_of(points: list[dict]) -> tuple[list[float], list[float]]:
    xs = [float(p["xyz"][0]) for p in points]
    ys = [float(p["xyz"][1]) for p in points]
    zs = [float(p["xyz"][2]) for p in points]
    return [min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)]


#: 右下角俯视小图的 axes 矩形（figure 坐标）。尺寸不能动主图：
#: 主图是 `add_subplot(111)` 且 `subplots_adjust` 写死的，小图另叠一块
#: `add_axes`，**不改主图布局**（主图尺寸有别的判据盯着）。
#:
#: 为什么不是 `[0.60, 0.07, 0.36, 0.30]`，也不是 `[0.635, 0.145, 0.315, 0.255]`：
#: 3D 轴的 **Z 标签是竖着站在右侧的**（x≈0.76~0.79 / y≈0.29~0.55），
#: 它的刻度数字（0.5/0.6/0.7）就在它左边一点。实测（把 figure 只画文字、
#: 不画数据，逐像素求交）：`[0.635, 0.145, 0.315, 0.255]` 会压住
#: `Z forward (m)` 的下半截 + 0.7 刻度，默认窗口 843 个文字像素，
#: 放大窗口 986 / 1118 —— 所以不能往左放，只能落在它右边那条**空白右边距**里。
#:
#: 关键细节：小图自己的 Y 刻度数字会从 axes 左边界**再往左伸 ~16 px**，
#: 所以 axes 的 x0 必须 ≥0.81 才算真不压（按 axes 矩形算会漏掉这 16 px）。
#: 右边到 0.985、下边 y0=0.10（刻度数字下方留 29 px，> 25 px 的硬要求）。
#:
#: ⚠️ 已知边界：窗口拉到最小（900x700，3D 画布 ~1.3:1）时 3D 轴吃满宽度，
#: Z 标签右移到 ≈0.85，小图会压到它。这是"不许改主图布局"的代价 ——
#: 要根治得让主图让出右边距，那属于主图行为，不在本次范围。
#: 2026-11-05 改：**压到底部**（y 0.03~0.275），与 Z 轴标签在纵向错开。
#: 原来 y 从 0.100 到 0.435，而 `Z forward (m)` 是竖排、占 y≈0.29~0.55 ——
#: 两者本来就在同一段高度里抢地方，只是 1.0 倍缩放下恰好没压到。一旦整幅放大
#: （zoom>1.05），Z 标签右移就撞上小图。放到 y<0.28 之后，Z 标签无论怎么右移
#: 都碰不到它，主图也就能真正用满画布。
#: 2026-11-05 第三次调：往左挪一点，**右边留 5% 给刻度标签**。
#:
#: 小图的 x 轴刻度是画在 axes **外面**的（matplotlib 2D 的默认行为），
#: 右边界 0.992 时那半截数字直接跑出 figure，被裁掉（用户截图：0.2 只剩
#: 半个）。左边是空的（Y 轴刻度在 y≈0.3 一带，小图在 y<0.22，纵向不重叠），
#: 所以往左挪不挡任何东西。
INSET_RECT = (0.795, 0.018, 0.152, 0.200)

#: 曲线配色（与主图一致：灰=原始、橙=平滑/小球）。
#: 原始线用中灰而不是浅灰：浅灰在图例里几乎看不见（图例继承线条的 alpha）。
INSET_RAW_COLOR = "#64748b"
INSET_SMOOTH_COLOR = "#f97316"

#: 小图里的中文（matplotlib 默认字体没有汉字，会画成方框）。按顺序回退。
INSET_FONTS = ("Microsoft YaHei", "SimHei", "DejaVu Sans")

#: CAD 式标准视角：(elev, azim, roll)，单位度。
#:
#: 键是**用户看到的说法**，不是轴的正负号。相机系 Y 向下，所以"俯视"
#: （从头顶看）是观察者在 **-Y** —— 写成 (0, -90)，不是 (90, 0)。
#:
#: "前视"为什么是 elev=-90（实测，别凭直觉改）：
#: matplotlib 的相机位置是 (cos e·cos a, cos e·sin a, sin e)。
#: elev=-90 把眼睛放在 **-Z**（光心处）朝 +Z 看，屏幕上 X 右 / Y 下 ——
#: 与相机拍到的画面同向；elev=+90 是从场景背后回看，Y 会翻到屏幕上方。
#: 观察方向预设：(仰角 elev, 方位角 azim, 绕视线滚转 roll)，单位度。
#:
#: ── 坐标约定（2026-11-05 用户明确）──────────────────────────────
#: 相机系：**X 向右、Y 向下、Z 向前**（相机光轴朝前）。
#:
#: 所以"右视"（从右侧看过去）的屏幕必须是 **横轴 Z、纵轴 Y** ——
#: 前后方向横过来，上下方向竖着。原来写的 `(0, 0, 0)` 得到的是
#: "横轴 Y、纵轴 Z"：Z 躺成了竖轴，读起来完全不是侧视图（用户截图指出）。
#:
#: matplotlib 的 3D 里 **z 轴默认永远竖直**，光靠 elev/azim 换不过来，
#: 必须用 `roll` 绕视线转 90° 才能把 Z 放平。每个角度的屏幕效果都在
#: 注释里标了，改之前先照它验一遍（下面这些值是投影实测出来的，
#: 不是猜的）：
#:
#:   前视  -90/-90/0    X→右  Y↓下  Z=深度
#:   后视   90/ 90/0    X→左  Y↓下  Z=深度
#:   右视    0/  0/-90  Z→右  Y↓下  X=深度
#:   左视    0/180/ 90  Z→左  Y↓下  X=深度
#:   俯视    0/-90/  0  X→右  Z↑上  Y=深度
#:   仰视    0/ 90/180  X→右  Z↓下  Y=深度
#:
#: 深度轴会被 `_hide_depth_axis` 自动藏掉（它退化成一点，刻度会糊成一团）。
VIEW_PRESETS = {
    "等轴测": (35.264, -45.0, 0.0),
    "前视（相机看到的）": (-90.0, -90.0, 0.0),
    "后视（从背后）": (90.0, 90.0, 0.0),
    "右视": (0.0, 0.0, -90.0),
    "左视": (0.0, 180.0, 90.0),
    "俯视（从头顶）": (0.0, -90.0, 0.0),
    "仰视（从下方）": (0.0, 90.0, 180.0),
}
DEFAULT_VIEW = "等轴测"

#: 鼠标滚轮缩放步长（每滚一格的范围缩放倍率）。<1 = 滚一下放大一点。
ZOOM_STEP = 0.85

#: 默认整幅缩放。1.0 时 3D 立方体只占画布中间一块 —— `subplots_adjust` 已经
#: 贴到边（left=0.02/right=0.98），填不满是因为立方体在矩形内**按最紧凑方式
#: 居中**，四周留白由投影包围盒决定，只能靠 zoom 放大（2026-11-05 用户：
#: "做成可以占据全局大小的，不然图片中的 Y 方向实在是比例奇怪"）。
DEFAULT_BOX_ZOOM = 1.0

#: 右键/中键拖动时，视野中心最多可以走出数据范围多少（占数据跨度的比例）。
#:
#: mplot3d 原生 pan **没有任何边界** —— `drag_pan()` 结尾是
#: `self._set_lims_from_transformed(min+dx, max+dx, ...)`，它改的是**三轴范围**
#: 而不是相机位置，也没有 clamp。拖两下就能把整条轨迹推出画布，屏幕上只剩
#: 几根网格线（2026-11-05 用户截图："线超出被挡住"）。留 0.5 表示最多走半个
#: 数据跨度，保证轨迹始终有一部分在画面里。
PAN_MARGIN = 0.5

#: 点击拾取半径（像素）。252 个采样点投影后彼此很近 —— 8px 太挤（选不中）、
#: 20px 容易选到隔壁邻居，12px 实测最顺手。
PICK_RADIUS_PX = 12.0

#: 采样点大小（pt²）与颜色。必须小：252 个点按原尺寸画会糊成一条粗带，
#: 反而看不出"这是逐帧采到的点"。
SAMPLE_DOT_SIZE = 7.0
SAMPLE_DOT_COLOR = "#2563eb"
SAMPLE_DOT_ALPHA = 0.75

#: 「最佳视角」的搜索参数。(仰角起, 仰角止, 仰角步长)，方位角全周。
#:
#: 为什么仰角不搜 0~90：0° 是纯侧视（3D 感全没了，跟看 2D 投影一样），
#: 90° 是纯俯视（Z 轴压成一个点）。15~60° 这段既保留立体感，又能让轨迹
#: 在屏幕上尽量展开 —— 用户要的是"最适合观察"，不是"投影面积最大"。
BEST_ELEV_RANGE = (15, 60, 15)
BEST_AZIM_STEP = 15

#: 视线方向与**任一坐标轴**的点积上限（= cos(夹角)）。
#:
#: 0.75 对应夹角 ≥ 41°，即每根轴在屏幕上至少还有 41° 的倾斜，不会被压成
#: 一点。没有这条约束时，搜索会选出 elev=15°/azim=90° —— 投影面积确实最大
#: （轨迹完全展开），但视线几乎**沿着 Y 轴**，三根轴的刻度全挤到画布左边重叠
#: 成一坨，根本读不出数（2026-11-05 实测踩到）。面积最大 ≠ 能看清。
BEST_MAX_AXIS_DOT = 0.75

#: 左上角信息面板的顶部位置（axes 坐标）。
EXTREMES_TOP_Y = 0.985

#: 极值面板高度的**兜底估算**（像素）。真正定位用 `get_window_extent()`
#: 实测值，这个只在整个测量失败时用（比如 canvas 还没拿到 renderer）。
EXTREMES_PANEL_PX = 72

#: 三个坐标平面：名字 → (垂直的轴号, 颜色)。0/1/2 = X/Y/Z。
PLANE_STYLES = {
    "XY": (2, "#60a5fa"),
    "XZ": (1, "#34d399"),
    "YZ": (0, "#fbbf24"),
}
_AXIS_NAMES = ("X", "Y", "Z")


def track_arrays(points: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """点列 → ``(t, P, P_smooth, valid)``，**四个数组长度都等于 `len(points)`**。

    缺口（丢帧 / 遮挡）**不补点**：平滑是按段做的，段之间的缺口原样留着 ——
    补点就等于编造了从来没有测到的位置，而这正是这个项目最不该出的错。

    平滑参数用 `stereo/motion.py` 的默认值（0.20 s / 2 阶，实测扫出来的），
    这里不覆盖它：查看器与速度/加速度是**同一条曲线**，参数一分叉就没人信了。
    """
    t = np.array([float(p.get("t", i)) for i, p in enumerate(points)], dtype=float)
    P = np.array([p["xyz"] for p in points], dtype=float).reshape(-1, 3)
    P_smooth, valid = smooth_positions(t, P)
    return t, P, P_smooth, valid


def rough_positions(t: np.ndarray, P: np.ndarray,
                    max_gap_s: float = 0.10) -> float:
    """抖动指标：段内二阶差分的平均模长（米）。**只在段内算**，不跨缺口。

    为什么不用逐点差分：一阶差分里包含真实速度，看不出"毛刺"；
    二阶差分对真实匀加速也敏感，但短窗口的高频抖动远大于运动本身的加速度，
    所以"平滑前后同一个数变小"足以说明毛刺被压掉了（判据见
    `tests/selftest_viz_scale.py` 第 11 节）。
    """
    total, count = 0.0, 0
    for seg in split_segments(t, max_gap_s):
        if seg.size < 3:
            continue
        d2 = np.linalg.norm(np.diff(P[seg], n=2, axis=0), axis=1)
        total += float(d2.sum())
        count += d2.size
    return total / count if count else 0.0


def draw_xz_inset(ax, P_raw: np.ndarray, P_smooth: np.ndarray,
                  smooth: bool) -> None:
    """在 `ax` 上画 **X–Z 俯视投影**：原始=灰细线，平滑=橙线。

    为什么是 X–Z 而不是 X–Y：坐标系是 **X 右 / Y 下 / Z 前**，
    "从上方往下看"看到的平面就是 X–Z；X–Y 是相机正面的投影，
    看起来像"贴着地面跑"，与真实轨道不是一回事。

    `smooth=False` 时**只画原始**并在角落写"段太短，未平滑" ——
    这时 `P_smooth == P_raw`（`smooth_positions` 对短段原样返回），
    画两条重合的线等于假装平滑生效了。

    放在模块级（而不是 `Viewer3D` 的方法里）是为了能**离屏验证**：
    判据只要一张 Agg 画布就能画，不必开 Tk 窗口。
    """
    ax.clear()
    # 原始线画在平滑线**下面**、略细、半透明：两条重合时橙线盖住灰线（说明平滑
    # 没改形状）；一旦有毛刺，灰线会从橙线两侧"支棱"出来 —— 这正是用户要看的东西。
    # 0.6/0.45 实测太淡（缩到查看器实际尺寸基本看不见），改成 1.4/0.30：
    # 宽度给足、透明度再压一点，既能看到毛刺，又不会被误读成"两条轨迹"。
    ax.plot(P_raw[:, 0], P_raw[:, 2], color=INSET_RAW_COLOR,
            linewidth=1.4, alpha=0.30, label="原始")
    if smooth:
        ax.plot(P_smooth[:, 0], P_smooth[:, 2],
                color=INSET_SMOOTH_COLOR, linewidth=1.4, label="平滑")
    else:
        ax.text(0.97, 0.04, "段太短，未平滑", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6.5, color="#b45309",
                fontfamily=INSET_FONTS)
    ax.margins(0.08)
    # 等比例：俯视图的作用是看"轨迹形状对不对"，把 X 与 Z 拉成不同比例
    # 会让椭圆看起来像别的曲线。`datalim` 调范围、不动 axes 的矩形。
    ax.set_aspect("equal", adjustable="datalim")
    # 小图只有 ~1 英寸见方：刻度只留 3 个、一位小数，否则数字比曲线还宽。
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(mticker.MaxNLocator(3))
        axis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.tick_params(labelsize=6, length=1.8, pad=1)
    ax.set_title("X–Z 俯视 (m)", fontsize=6.5, pad=1.5, fontfamily=INSET_FONTS)
    if smooth:
        ax.legend(loc="lower right", frameon=False, handlelength=1.1,
                  borderaxespad=0.1,
                  prop={"family": list(INSET_FONTS), "size": 6})
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


#: 会话结果最多列几个（够选就行；下拉太长反而难找）
SOURCE_LIMIT = 40


def scan_sources(limit: int = SOURCE_LIMIT,
                 roots: tuple[Path, ...] | None = None) -> list[Path]:
    """所有**真有**三维结果的会话（单目 + 双目），新的在前。

    扫三个地方（2026-10-29 按类型分目录之后）：

    - ``stereo_roots()`` 各自的 ``*/derived/``（`data/stereo` 与旧的 `data/sessions`）
    - ``MONO_ROOT/*/derived/``（`data/mono`）
    - ``data/*/derived/``（最早期的会话直接放在 `data/` 下，那时两个子目录都没有）

    路径**一律从 `sphere_tracker/paths.py` 来**，不在这里再写 `ROOT/"data"/...`
    —— 那正是"会话"这个词有两个意思的起点。

    `roots` 只给测试用：判据要能造临时会话，不能依赖这台机器录过什么。

    **排序用目录名里的 `YYYYMMDD-HHMMSS`，不用文件 mtime**：
    跑一次离线会重写 `derived/`，把老会话的 mtime 刷成今天 —— 按 mtime 排会把
    "昨天录、今天补跑"的会话顶到第一条，默认选中就漂了。
    这个坑 `apps/_workbench.scan_sessions` 已经吃过一次，判据写在它的 docstring 里。
    """
    if roots is not None:
        bases = tuple(roots)
    else:
        # dict.fromkeys 去重保序：`data/stereo` 不存在时不多扫一遍。
        # `MONO_ROOT.parent` 就是 `data/` 本身 —— 用它而不是再写一遍字面量，
        # 这样"data 在哪"仍然只有 paths.py 一个来源。
        bases = tuple(dict.fromkeys(
            (*stereo_roots(), MONO_ROOT, MONO_ROOT.parent)))
    # 枚举会话**必须**走 `candidate_session_dirs`：会话按装置分到
    # `data/stereo/parallel/`、`data/stereo/angled/` 之后，根目录下一层已经
    # 不是会话了。原来这里自己 `base.glob("*/derived/*")`，于是**新布局的会话
    # 一个都扫不到** —— 表现为"数据源列表里没有最新那次实验"，默认会话只能
    # 落到更老的目录上（2026-11-05 用户："每次打开应该默认打开最新的会话"，
    # 查下来不是排序问题，是根本没扫到）。
    from apps._stereo_layout import candidate_session_dirs

    found: dict[Path, Path] = {}
    for base in bases:
        if not base.is_dir():
            continue
        for sess in candidate_session_dirs([base]):
            for name in ("visualization.json", "traj3d.json"):
                f = sess / "derived" / name
                if not f.exists():
                    continue
                # 同一个会话两种都写时，优先 visualization.json（它带视频叠加信息）
                if sess not in found or name == "visualization.json":
                    found[sess] = f

    def key(f: Path) -> tuple[str, float]:
        name = f.parent.parent.name
        m = re.match(r"(\d{8}-\d{6})", name)
        try:
            return (m.group(1) if m else "", f.stat().st_mtime)
        except OSError:
            return (m.group(1) if m else "", 0.0)

    return sorted(found.values(), key=key, reverse=True)[:max(1, limit)]


def source_kind(f: Path) -> str:
    """这个结果文件属于哪种数据：``"mono"`` / ``"stereo"`` / ``""``（读不出来）。

    **读文件里的 schema，不看路径**：单目会话可以被拷到 `data/` 下（早期位置），
    按目录判会把类型说错，而类型正是下拉里那个前缀要说的事。
    文件坏了就返回空串 —— 下拉里只显示名字，选它会当场报错，不静默。
    """
    try:
        schema = json.loads(Path(f).read_text(encoding="utf-8")).get("schema")
    except (OSError, ValueError):
        return ""
    return "mono" if schema == MONO_SCHEMA else ("stereo" if schema in SCHEMAS else "")


def source_label(f: Path) -> str:
    """下拉里显示的名字：``[单目]``/``[双目]`` 前缀 + 会话目录名（有 label 就接上）。

    前缀不是装饰：单目与双目的 **Z 不是一回事**（先验 vs 视差），
    混在一个列表里而看不出哪个是哪个，等于让人拿先验当测量。
    """
    sess = f.parent.parent
    try:
        tag = str(json.loads((sess / "session.json").read_text(encoding="utf-8"))
                  .get("label") or "").strip()
    except Exception:                                       # noqa: BLE001
        tag = ""
    kind = {"mono": "[单目]", "stereo": "[双目]"}.get(source_kind(f), "[?]")
    name = f"{sess.name}  {tag}" if tag else sess.name
    return f"{kind} {name}"


class Viewer3D:
    def __init__(self, root: tk.Tk, data: dict, video_path: str | Path | None = None,
                 auto_video: bool = True) -> None:
        self.root = root
        self.playing = False
        self.speed = 1.0
        self._syncing = False
        self.cap: cv2.VideoCapture | None = None
        self._left_raw = None
        self._right_raw = None
        self._sources: list[Path] = []
        # 「平滑」开关：**默认开**（用户要的就是去掉毛刺）。它是**用户的偏好**，
        # 不是会话状态 —— 换会话不清零，所以放在 `_apply` 之外建。
        # `master=root` 不能省：测试会在同一进程里连开几个 Tk 根，
        # 不指定 master 的布尔变量会绑到"默认根"上，换到第二个窗口就抛
        # `too many values to unpack` / 变量失效。
        self.use_smooth = tk.BooleanVar(master=root, value=True)
        # 视角与参考平面同样是**用户偏好**，放在 `_apply` 之外：换会话还是同一套
        # 坐标系，把视角弹回默认没有道理（用户的旋转应该留得住）。
        _e, _a, _r = VIEW_PRESETS[DEFAULT_VIEW]
        self.view = {"elev": _e, "azim": _a, "roll": _r}
        self.show_planes = {name: tk.BooleanVar(master=root, value=False)
                            for name in PLANE_STYLES}
        self.ortho = tk.BooleanVar(master=root, value=True)
        # 「等比例坐标」：默认 **关** —— 三轴各自拉满，形状便于阅读。
        # 开：按真实米数等比例（Y 只动 1.7 cm 就真的只有一小截，刻度会挤成一坨）。
        # 这是**显示**选择，不是数据选择：轨迹本身一个字都没变（2026-11-05
        # 用户："这个坐标背景图比例很奇怪…Y 方向实在是比例奇怪"）。
        self.equal_axes = tk.BooleanVar(master=root, value=False)
        # 「保持立方体比例」：默认**关** = 容器铺满画布（立方体会被宽屏拉宽）。
        # 勾上 = 容器退回正方形，形状不变形，但宽屏下左右会各留 17% 空白。
        # 这两个是几何上互斥的：等轴测投影近似正六边形，要它横向铺满，
        # 就必然横向拉伸（2026-11-05）。
        self.keep_cube = tk.BooleanVar(master=root, value=False)
        # 三轴分度值（刻度间隔）。空 = 自动（每轴最多 4 档）。
        # 用户要的是"我自己定网格多密"—— 0.05 就是每 5 cm 一根线
        # （2026-11-05："补充上手动调整三轴分度值的功能"）。
        self.tick_vars = {k: tk.StringVar(master=root, value="")
                          for k in ("x", "y", "z")}
        self._roll_drag: tuple[int, float] | None = None
        self._syncing_view = False
        self._plane_note = ""
        # 当前显示范围（三轴 lim）。None = 还没定过，首次 `_redraw` 时按数据
        # 包围盒 fit。之后 pan/zoom 只改它、`_redraw` 照它画，所以播放时
        # 用户的平移/缩放**留得住**（这是"移动的是坐标轴、一松手又弹回"的根）。
        self.lims: list[list[float]] | None = None
        # 整幅缩放（`set_box_aspect(zoom=)`）：数据、网格、刻度一起放大。
        # 与 `lims` 是两套东西 —— lims 改"看哪一段"，zoom 改"这一幅画多大"。
        self.box_zoom: float = DEFAULT_BOX_ZOOM
        # 选中的采样点下标（None = 没选）。点击拾取见 `_pick_at`。
        self.sel_idx: int | None = None
        # 采样点的屏幕像素坐标缓存（`_cache_point_px`）。点击时只做距离比较，
        # 不重投影 —— 每次点击都算 252 个点会有肉眼可见的卡顿。
        self._pt_px: "np.ndarray | None" = None
        self._press_px: tuple[int, int] | None = None
        self._pan_active: bool = False
        self._pan_button: int = 3
        self._apply(data, video_path, auto_video)
        try:
            self._build()
            self._redraw()
        except Exception:
            # 构造中途失败（多 Tk 根、matplotlib 后端、无显示…）也必须把视频
            # 句柄放掉：Windows 上不放，`annotated.mp4` 会一直被占着，
            # 现象是"临时目录删不掉 / 文件删了还在"（WinError 32）。
            self._release_cap()
            raise

    def _apply(self, data: dict, video_path: str | Path | None,
               auto_video: bool) -> None:
        """装入一份数据。**所有随会话变化的状态都必须在这里重置**。

        换会话只调它 + 刷新控件，**不重建窗口**：视频 canvas / 3D 画布 / 滑杆
        都是持久的，重建会把用户的窗口尺寸、当前视角、播放速度一起丢掉。

        `_left_raw / _right_raw` 也要清：新会话如果没有视频，`_update_video`
        会走 `_render_video`，而它还留着上一个会话的解码帧 —— 屏幕上就会出现
        "标题换了、画面还是上一条"的假象，且不报任何错。
        """
        self._release_cap()
        self._left_raw = None
        self._right_raw = None
        self.data = data
        self.points = data["trajectory"]
        self.n = len(self.points)
        self.idx = 0
        self.mono = is_mono(data)
        # 平滑曲线**在装入会话时算一次**，不在每一帧重算：0.20 s 窗口的
        # Savitzky-Golay 是 O(n)，播放时每秒要重画 30 次，重算纯属白烧 CPU。
        # 注意「平滑」开关本身不在这里重置（那是用户偏好，见 `__init__`）。
        _t, self.P_raw, self.P_smooth, self.smooth_valid = track_arrays(self.points)
        self.video_error = ""
        self.video_disabled = (video_path is None or video_path == "") and not auto_video
        self.lo, self.hi = bounds_of(self.points)
        self.video_path = self._resolve_video(video_path, auto_video)
        self.cap = cv2.VideoCapture(str(self.video_path)) if self.video_path else None
        self.cap_frames = 0
        if self.cap is not None and self.cap.isOpened():
            # 记下总帧数：轨迹点可以有很多帧，视频却可能**只有十几帧**
            # （生成中断/被覆盖）。不记的话，seek 越界只能报一句"解码失败"，
            # 用户会以为是播放器坏了（2026-11-05 现场：视频 19 帧、
            # 轨迹 301 帧，两边都显示"frame 56 解码失败"）。
            try:
                self.cap_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            except Exception:                       # noqa: BLE001
                self.cap_frames = 0
        if self.cap is not None and not self.cap.isOpened():
            self.video_error = f"视频打不开：{self.video_path}"
            self.cap.release()
            self.cap = None
        if self.cap is not None and not self.mono:
            # 只有双目才要求左右并排：单目的标注视频就是一路画面，
            # 拿 2:1 去卡它会把每一段单目会话都判成"视频不能看"。
            vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if vw > 0 and vh > 0 and vw < vh * 1.8:
                self.video_error = f"视频不是左右并排（{vw}x{vh}），按 2:1 切分会错"
                self.cap.release()
                self.cap = None
        kind = "[单目] " if self.mono else "[双目] "
        self.root.title(f"3D 轨迹 · {kind}{data.get('session', '')} · {self.n} points")
        self._sync_mono_note()

    def _sync_mono_note(self) -> None:
        """单目时把"尺度从哪来"常驻在图下方；双目时清空。

        为什么不用弹窗：每次换会话都弹一次，用户第三次就会条件反射点掉，
        等于没提示。常驻一行不打扰，但一直在。
        """
        note = getattr(self, "mono_note", None)
        if note is None:
            return
        if not self.mono:
            note.configure(text="")
            return
        d = self.data.get("ball_diameter_mm")
        d_txt = f"{float(d):g}" if isinstance(d, (int, float)) else "?"
        note.configure(
            text=f"单目深度由球径先验得到（D={d_txt} mm，Z=f·D/(2r)），"
                 f"尺度依赖这个数；不是视差测量")

    def open_source(self, path: str | Path) -> None:
        """切到另一个会话 / 文件。

        **先 `load_view` 再改状态**：加载失败（目录里没有 json、schema 不对、
        轨迹为空）时 `self` 一点没动，调用方弹个提示即可，正在看的东西不会消失。
        """
        data = load_view(path)
        self._apply(data, None, True)
        self.lims = None               # 换会话 = 新数据，范围重新 fit
        self.box_zoom = DEFAULT_BOX_ZOOM   # 整幅缩放归位（新数据量级不同）
        self.slider.configure(to=max(0, self.n - 1))
        self.slider.set(0)
        self._redraw()                       # 内含 _reset_view / _update_video / _status
        self._sync_source_row()

    def _resolve_video(self, explicit: str | Path | None, auto: bool) -> Path | None:
        if explicit:
            p = Path(explicit)
            return p if p.exists() else None
        if not auto:
            return None
        p = Path(self.data.get("_path", ""))
        if p.is_file():
            session = p.parent.parent if p.parent.name == "derived" else p.parent
            cand = session / "derived" / "annotated.mp4"
            if cand.exists():
                return cand
        return None

    # ------------------------------------------------------------------
    def _build(self) -> None:
        # 四宫格改为**可拖动**（2026-11-04 用户："四宫格做成可以拖动放大缩小"）。
        # 结构：外层竖直 PanedWindow 分上下两半；上半内再套水平 PanedWindow
        # 分左/右视频，下半同理分 3D / 视角面板。拖 sash 即调各格大小，
        # 鼠标拖动条两侧内容各自响应 resize（视频 cover-crop、3D 图随画布缩放）。
        self._grid = ttk.PanedWindow(self.root, orient="vertical")
        # 上半：左右视频
        top = ttk.PanedWindow(self._grid, orient="horizontal")
        # 下半：3D + 视角面板
        bottom = ttk.PanedWindow(self._grid, orient="horizontal")

        # Canvas 的 request size 固定为 1×1：图像只作为 item 画进去，
        # 不会反过来把容器撑大（之前用 Label + PhotoImage 会正反馈放大到全屏）。
        self.video_l = tk.Canvas(top, bg="#0f172a", highlightthickness=0,
                                 width=1, height=1)
        self.video_r = tk.Canvas(top, bg="#0f172a", highlightthickness=0,
                                 width=1, height=1)
        self._left_raw = None
        self._right_raw = None
        self._left_photo = None
        self._right_photo = None
        self.video_l.bind("<Configure>", lambda _e: self._render_video())
        self.video_r.bind("<Configure>", lambda _e: self._render_video())

        plot_frame = ttk.Frame(bottom)
        self.fig = plt.Figure(figsize=(5.5, 3.6), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")
        # 右键 = **旋转视角**，不是缩放（2026-11-05 用户："右键拖动不是我要的
        # 拖动摄像机视角，而是特么的缩放比例"）。matplotlib 3D 默认左键旋转、
        # 中键平移、右键 zoom；用户习惯右键转视角，所以把 3 也加进 rotate 并从
        # zoom 里移除。滚轮已经负责缩放，右键再缩放是重复且抢手感的。
        self._bind_mouse_buttons()
        # ── 主图**占满**画布（2026-11-05 用户："全屏不是让窗口全屏，而是让
        # 坐标轴应用于全局画面，你自己看是不是很多留白"）────────────────────
        #
        # 留白有两层来源，缺一不可：
        #
        # ① `subplots_adjust` 的边距。那是 **2D 思维** —— 2D 轴的标签在 axes
        #    **外面**，不留边距会被裁；3D 的标签是 mplot3d 摆在 axes **里面**
        #    的，边距纯属浪费。
        #
        # ② **更隐蔽**：`Axes3D.apply_aspect()` 里 `box_aspect = 1` 是**写死**
        #    的（matplotlib 3.11 源码），它无条件把 axes 收缩成**正方形**，
        #    `set_aspect('auto')` 完全不生效。于是 16:10 的画布里，3D 容器
        #    只有宽度 65%，左右各空 17.5% —— 这才是"立方体只有中间一小块"
        #    的真正原因，和边距无关。只能把 `apply_aspect` 换成不收缩。
        self.ax.apply_aspect = lambda position=None: None
        self._apply_layout()
        # 右下角俯视小图：**另叠一块 axes**，不碰主图的 subplot 布局
        # （主图的范围/视角/色标有别的判据盯着）。zorder 抬高让它盖在 3D 轴之上。
        self.ax_inset = self.fig.add_axes(INSET_RECT)
        self.ax_inset.set_zorder(10)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        # 窗口拉伸会改 figure 的物理尺寸，"保持比例"模式的方形边长得重算；
        # 铺满模式虽然恒为 [0,0,1,.96]，重算一次也无害（幂等）。
        self.canvas.mpl_connect("resize_event", lambda _e: self._apply_layout())
        # 平移/拾取自己接管：原生 pan 没有边界（见 `_on_motion` 的说明），
        # 原生 3D pick 也不可靠（`picker` 在 3D 里基本不触发），所以都走
        # press/motion/release 三个事件自己实现。
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)

        # 右下角：视角 / 参考平面。
        side = ttk.LabelFrame(bottom, text="视角 / 参考平面", padding=(10, 8))

        # 组装 paned：先 add 内容，再 add 嵌套 paned 到外层。
        # **初始比例靠 weight，不靠手动 sashpos**（2026-11-04 修）：
        # weight 会在 sash 未被拖动时按比例分配剩余空间（视频区 2/5 = 40%），
        # 一旦用户拖了，就转成绝对位置、weight 失效 —— 正好是"初始给比例、
        # 之后交给用户"的正确语义。手动 `sashpos` 在窗口首帧尺寸还没算好时
        # 会读到 winfo_height()==1，把 sash 设到 0，视频区塌成一条缝。
        top.add(self.video_l, weight=1)
        top.add(self.video_r, weight=1)
        bottom.add(plot_frame, weight=3)
        bottom.add(side, weight=1)
        self._grid.add(top, weight=2)
        self._grid.add(bottom, weight=3)
        self._top_pane = top
        self._bottom_pane = bottom
        # 首次真正布局后，把上下 sash 定位到"视频区约占 40% 高"。视频 Canvas
        # 请求尺寸是 1×1（避免正反馈撑大），所以 weight 只够给个最小值，
        # 视频区会塌成一条缝 —— 必须等 paned 有了真实高度后补一次 sashpos。
        # 只做一次（`_paned_seeded` 门），之后用户拖到哪就是哪。
        self._paned_seeded = False
        self._grid.bind("<Configure>", self._seed_video_sash)

        row = ttk.Frame(side)
        row.pack(fill="x")
        ttk.Label(row, text="观察方向").pack(side="left")
        self.view_box = ttk.Combobox(row, values=list(VIEW_PRESETS),
                                     state="readonly", width=15)
        self.view_box.set(DEFAULT_VIEW)
        self.view_box.pack(side="left", padx=(6, 0), fill="x", expand=True)
        self.view_box.bind("<<ComboboxSelected>>", self._on_view_preset)

        ttk.Label(side, text="绕视线滚转（也可 Shift + 左键拖动）",
                  style="Hint.TLabel").pack(anchor="w", pady=(10, 0))
        roll_row = ttk.Frame(side)
        roll_row.pack(fill="x")
        self.roll_scale = ttk.Scale(roll_row, from_=-180, to=180,
                                    command=self._on_roll)
        self.roll_scale.pack(side="left", fill="x", expand=True)
        self.roll_lbl = ttk.Label(roll_row, text="0°", width=5, anchor="e")
        self.roll_lbl.pack(side="left", padx=(6, 0))

        ttk.Label(side, text="参考平面（过原点）",
                  style="Hint.TLabel").pack(anchor="w", pady=(10, 2))
        for _name in PLANE_STYLES:
            ttk.Checkbutton(side, text=f"{_name} 平面",
                            variable=self.show_planes[_name],
                            command=self._on_plane_toggle).pack(anchor="w")
        self.ortho_chk = ttk.Checkbutton(
            side, text="正交投影（CAD 风格）", variable=self.ortho,
            command=self._on_proj_toggle)
        self.ortho_chk.pack(anchor="w", pady=(8, 0))
        self.equal_chk = ttk.Checkbutton(
            side, text="等比例坐标（真实几何）", variable=self.equal_axes,
            command=self._on_axes_mode)
        self.equal_chk.pack(anchor="w", pady=(8, 0))
        self.keep_chk = ttk.Checkbutton(
            side, text="保持立方体比例（会留白）", variable=self.keep_cube,
            command=self._on_layout_mode)
        self.keep_chk.pack(anchor="w", pady=(6, 0))

        ttk.Label(side, text="分度值 X / Y / Z（米，留空=自动）",
                  style="Hint.TLabel").pack(anchor="w", pady=(10, 2))
        tick_row = ttk.Frame(side)
        tick_row.pack(fill="x")
        for _key, _hint in (("x", "X"), ("y", "Y"), ("z", "Z")):
            ttk.Label(tick_row, text=_hint + " ").pack(side="left")
            _e = ttk.Entry(tick_row, textvariable=self.tick_vars[_key], width=6)
            _e.pack(side="left", padx=(0, 4))
            _e.bind("<Return>", self._on_tick_edit)
            _e.bind("<FocusOut>", self._on_tick_edit)

        self.plane_note = ttk.Label(side, text="", style="Hint.TLabel",
                                    wraplength=250, justify="left")
        self.plane_note.pack(anchor="w", pady=(8, 0))
        ttk.Label(side, text="左键拖动：旋转视角\n右键拖动：平移（人走动）",
                  style="Hint.TLabel", justify="left").pack(anchor="w", pady=(8, 0))
        # 说明都搬到「操作说明」窗口了，这里只留一键入口（2026-11-05）。
        ttk.Button(side, text="? 操作说明（3D 查看器）",
                   command=self._open_help).pack(anchor="w", pady=(6, 0))

        bar = ttk.Frame(self.root, padding=(8, 4))
        bar.pack(fill="x")
        self.play_btn = ttk.Button(bar, text="播放", command=self._toggle_play, width=8)
        self.play_btn.pack(side="left")
        ttk.Button(bar, text="重置3D视角", command=self._reset_view, width=10).pack(side="left", padx=4)
        # 「最佳视角」= 自动角度 + 自动缩放。和「重置3D视角」分开：
        # 后者回到固定预设，前者按数据形状算 —— 用户要的是后者
        # （2026-11-05："智能回到最适合的观察角度和缩放比例"）。
        ttk.Button(bar, text="最佳视角", command=self._smart_view,
                   width=9).pack(side="left", padx=(0, 4))
        # 名字必须说清是**窗口**全屏：用户以为「全屏」= 坐标轴铺满画布，
        # 于是点完发现立方体还是那么大，白失望一次（2026-11-05）。
        # 坐标轴铺满是**默认布局**，不需要按钮。
        self.full_btn = ttk.Button(bar, text="窗口全屏",
                                   command=self._toggle_fullscreen, width=9)
        self.full_btn.pack(side="left", padx=(0, 4))
        ttk.Label(bar, text="帧").pack(side="left", padx=(10, 2))
        # 逐帧步进（2026-11-05 用户："播放器那里增加前进/后退 x 帧的功能"）。
        # 放在滑杆两侧：左边退、右边进，方向和拖动一致。±1 用于精确定位，
        # ±10 用于快速扫一段 —— 拖滑杆在几百帧里根本对不准某一帧。
        for _txt, _delta in (("-10", -10), ("-1", -1)):
            ttk.Button(bar, text=_txt, width=3,
                       command=lambda d=_delta: self._step_frames(d)
                       ).pack(side="left", padx=(0, 2))
        self.slider = ttk.Scale(bar, from_=0, to=max(0, self.n - 1),
                                command=self._on_slider, orient="horizontal")
        self.slider.pack(side="left", fill="x", expand=True, padx=4)
        for _txt, _delta in (("+1", 1), ("+10", 10)):
            ttk.Button(bar, text=_txt, width=3,
                       command=lambda d=_delta: self._step_frames(d)
                       ).pack(side="left", padx=(2, 0))
        ttk.Label(bar, text="速度").pack(side="left", padx=(8, 2))
        self.speed_box = ttk.Combobox(bar, values=("0.25x", "0.5x", "1x", "2x"),
                                      width=5, state="readonly")
        self.speed_box.set("1x")
        self.speed_box.pack(side="left")
        self.speed_box.bind("<<ComboboxSelected>>", self._on_speed)
        # 「平滑」：跟现有控件同一条 bar，不另起一行（另起一行会把 3D 图挤矮）。
        # 默认开 —— 毛刺是跟踪噪声，不是运动。
        self.smooth_chk = ttk.Checkbutton(bar, text="平滑", variable=self.use_smooth,
                                          command=self._on_toggle_smooth)
        self.smooth_chk.pack(side="left", padx=(10, 0))
        # 缩放/平移提示（2026-11-04）：三键交互是 matplotlib 3D 原生的，
        # 用户看不到说明就不知道中键能平移、右键能缩放。文字放控制条右侧。
        ttk.Label(bar, text="左键旋转 · 右键平移（走动） · 滚轮缩放 · "
                            "Shift+滚轮 定点放大",
                  style="Hint.TLabel").pack(side="right", padx=(10, 4))

        self.status = ttk.Label(self.root, padding=(10, 4), anchor="w")
        self.status.pack(side="bottom", fill="x")
        # 单目尺度的常驻说明（在图/状态条之间，不弹窗）。先 pack 的在下层，
        # 所以它出现在状态条**上方** —— 与"图下方"的直觉一致。
        self.mono_note = ttk.Label(self.root, padding=(10, 2), anchor="w",
                                   foreground="#b45309")
        self.mono_note.pack(side="bottom", fill="x")

        # ---- 数据源：换会话 ----
        # 2026-10-28 用户报："打开后自动选最近会话，且不能改"。
        # 以前源只来自命令行位置参数，换一个就得关窗重开（界面那边也只能给
        # "最近一个"）。这里给一个下拉 + 目录浏览，两条路都能换，
        # **换的时候不重建窗口**（窗口尺寸/视角/播放速度都保留）。
        src = ttk.Frame(self.root, padding=(8, 2))
        src.pack(side="bottom", fill="x")     # 先 pack status 再 pack 它 → 落在 status 上方
        ttk.Label(src, text="数据源").pack(side="left")
        self.source_box = ttk.Combobox(src, width=38, state="readonly")
        self.source_box.pack(side="left", padx=(8, 4))
        self.source_box.bind("<<ComboboxSelected>>", self._on_pick_source)
        ttk.Button(src, text="打开目录…", width=10,
                   command=self._on_browse_source).pack(side="left")
        self.source_note = ttk.Label(src, text="")
        self.source_note.pack(side="left", padx=(10, 0))
        # 先给控制条/状态栏留位置，再把四宫格填满剩余空间；
        # 否则 paned 的 expand 会把后面的 pack 控件压成 1px。
        self._grid.pack(fill="both", expand=True, padx=4, pady=4)
        self.root.minsize(900, 700)
        self.root.bind("<space>", self._on_space)
        self.root.bind("<Left>", self._on_left)
        self.root.bind("<Right>", self._on_right)
        self.root.bind("<Home>", lambda _e: self._reset_view())
        # Shift + 左键拖动 = 绕视线滚转。matplotlib 原生只改 elev/azim，
        # 没有 roll；elev 又被夹在 ±90°，所以少了这一个自由度就到不了
        # "从球的正后方斜着看"这类姿态。
        self.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._refresh_sources()
        self._sync_mono_note()

    def _seed_video_sash(self, _event=None) -> None:
        """首帧布局完成后给视频区一个 40% 高的 sash，只做一次。

        不能用 `after(0)`：那时 paned 还没被映射，`winfo_height()` 是 1，
        sash 会被设到 0（就是用户报的"视频显示范围为 0"）。等到
        `<Configure>` 且高度真实（>100）再设，且 `_paned_seeded` 保证不反复
        覆盖用户后续的拖动。
        """
        if self._paned_seeded:
            return
        h = self._grid.winfo_height()
        if h < 100:                       # 还没真正布局，等下一次 Configure
            return
        self._paned_seeded = True
        try:
            self._grid.sashpos(0, int(h * 0.4))
        except Exception:                                   # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # 换数据源
    # ------------------------------------------------------------------
    def _refresh_sources(self) -> None:
        """重扫会话列表并勾住当前源。"""
        self._sources = scan_sources()
        self.source_box["values"] = [source_label(f) for f in self._sources]
        self.source_note.configure(
            text=f"共 {len(self._sources)} 个会话有结果（单目 + 双目）"
                 if self._sources else
                 "没有找到会话结果（data/stereo、data/mono 下都没有 derived/）")
        self._sync_source_row()

    def _sync_source_row(self) -> None:
        """让下拉框停在**当前真的在看的那一个**上。

        换源失败时也要调它：否则下拉会停在一个没打开的名字上，
        用户以为在看他选的那个，实际屏幕上还是上一个。
        """
        cur = str(self.data.get("_path") or "")
        for i, f in enumerate(self._sources):
            if str(f) == cur:
                self.source_box.current(i)
                return
        self.source_box.set("")

    def _on_pick_source(self, _event=None) -> None:
        i = self.source_box.current()
        if 0 <= i < len(self._sources):
            self._switch_to(self._sources[i])

    def _on_browse_source(self) -> None:
        # 起始目录优先给双目根（最常见），它不存在就退回 `data/` ——
        # 给一个不存在的目录，对话框会从"我的电脑"开起，用户得多点好几下。
        start = STEREO_ROOT if STEREO_ROOT.is_dir() else MONO_ROOT.parent
        d = filedialog.askdirectory(
            title="选择会话目录（含 derived/visualization.json）",
            initialdir=str(start), parent=self.root)
        if not d:
            return
        try:
            f = resolve_json(d)                 # 目录 -> 具体 json（会校验存在性）
        except Exception as exc:                # noqa: BLE001
            messagebox.showwarning("这里没有三维结果", f"{d}\n\n{exc}", parent=self.root)
            return
        self._switch_to(f)

    def _switch_to(self, f: Path) -> None:
        if str(f) == str(self.data.get("_path") or ""):
            return                              # 已经是它，别白重载
        try:
            self.open_source(f)
        except Exception as exc:                # noqa: BLE001
            # 换失败必须**保持原视图**（open_source 里先 load 后改状态，所以
            # 走到这里 self 还没被动过），只把下拉拉回去。
            messagebox.showwarning("这个源打不开", f"{f}\n\n{exc}", parent=self.root)
            self._sync_source_row()

    def _on_space(self, event) -> None:
        # 勾选框聚焦时按空格 = 切「平滑」；不排除它，空格会**同时**切播放
        # （两个状态一起变，用户看到的是一次乱跳）。
        if isinstance(event.widget, (ttk.Button, ttk.Checkbutton, ttk.Entry,
                                     ttk.Combobox, ttk.Scale)):
            return
        self._toggle_play()

    def _on_left(self, event) -> None:
        if isinstance(event.widget, (ttk.Entry, ttk.Combobox, ttk.Scale)):
            return
        self._step(-1)

    def _on_right(self, event) -> None:
        if isinstance(event.widget, (ttk.Entry, ttk.Combobox, ttk.Scale)):
            return
        self._step(1)

    def _release_cap(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def close(self) -> None:
        """释放视频句柄并关窗。

        单独提出来是因为**测试也会开窗**，而它必须走同一条释放路径：
        Windows 上 `cap` 不 release，`annotated.mp4` 就一直被占着 ——
        关掉查看器之后删不掉、覆盖不了，报"另一个程序正在使用此文件"。
        """
        self._release_cap()
        self.root.destroy()

    def _on_close(self) -> None:
        self.close()

    # ------------------------------------------------------------------
    def _limits(self):
        span = [max(1e-3, self.hi[i] - self.lo[i]) for i in range(3)]
        pad = [max(0.02, s * 0.12) for s in span]
        return ([self.lo[i] - pad[i] for i in range(3)],
                [self.hi[i] + pad[i] for i in range(3)])

    def _bind_mouse_buttons(self) -> None:
        """左键/右键都旋转、中键平移、右键**不**缩放。

        `ax.clear()` 不动这几个列表，但把它们收在一处，以后改键位只改这里。
        `_zoom_btn = []` 是必须的：若 3 同时出现在 rotate 和 zoom 里，
        mplot3d 的移动处理会两个都响应（拖动时一边转一边缩放）。
        """
        # 右键 = **平移**（2026-11-05 用户："假设观察者是一个人，右键不是转动
        # 视角，而是人前后左右的走"）。mplot3d 的 pan 正是这个语义：视野中心
        # 跟着鼠标在屏幕平面里移动，看起来就是人朝那个方向走。
        # 左键仍是旋转，中键也保留平移（老习惯），缩放只归滚轮。
        self.ax._rotate_btn = [1]
        # 原生 pan（`_pan_btn`）**不能用**：`drag_pan()` 改的是三轴 lim 且完全
        # 没有边界，拖两下轨迹就整体跑出画布（2026-11-05 用户截图）。
        # 改由 `_on_press/_on_motion` 手动调 `drag_pan` 并立刻 `_clamp_pan`。
        self.ax._pan_btn = []
        self.ax._zoom_btn = []

    # ── 鼠标：平移（有边界）/ 点选采样点 ────────────────────────────────
    def _on_press(self, event) -> None:
        """记下按下位置；右/中键在**主 3D 轴内**按下则开始平移。"""
        self._press_px = (event.x, event.y) if event.x is not None else None
        if event.inaxes is not self.ax or event.button not in (2, 3):
            self._pan_active = False
            return
        self._pan_active = True
        self._pan_button = int(event.button)
        try:
            self.ax.start_pan(event.x, event.y, event.button)
        except Exception:                       # noqa: BLE001
            self._pan_active = False

    def _on_motion(self, event) -> None:
        """拖动 = 平移视野，每次移动后立刻夹回数据附近。"""
        if not self._pan_active or event.x is None:
            return
        try:
            self.ax.drag_pan(self._pan_button, event.key, event.x, event.y)
        except Exception:                       # noqa: BLE001
            return
        self._clamp_pan()
        self.canvas.draw_idle()

    def _on_release(self, event) -> None:
        """松开：结束平移；左键**没拖动**才算点击（拖动是在转视角）。"""
        was_pan = self._pan_active
        self._pan_active = False
        if event.button != 1 or event.inaxes is not self.ax or was_pan:
            return
        if self._press_px is None or event.x is None:
            return
        # 4px 容差：手抖一两像素仍算点击，超过就是旋转手势
        if (abs(event.x - self._press_px[0]) > 4
                or abs(event.y - self._press_px[1]) > 4):
            return
        self._pick_at(event.x, event.y)

    def _clamp_pan(self) -> None:
        """把平移后的三轴范围夹回数据附近，防止把轨迹整体推出画布。

        `drag_pan` 的语义是**平移数据窗口**（改 lim，不是改相机），所以它没有
        任何边界。这里只限制**视野中心**不超出 `数据范围 ± PAN_MARGIN × 跨度`，
        窗口大小（缩放）不动 —— 用户"往前走"的手感保留，但走不丢。
        """
        for i, (getter, setter) in enumerate((
                (self.ax.get_xlim, self.ax.set_xlim),
                (self.ax.get_ylim, self.ax.set_ylim),
                (self.ax.get_zlim, self.ax.set_zlim))):
            a, b = getter()
            span = b - a
            if span <= 0:
                continue
            dspan = max(1e-9, float(self.hi[i] - self.lo[i]))
            center = (a + b) / 2.0
            c_lo = float(self.lo[i]) - PAN_MARGIN * dspan
            c_hi = float(self.hi[i]) + PAN_MARGIN * dspan
            fixed = min(max(center, c_lo), c_hi)
            if abs(fixed - center) > 1e-12:
                setter(fixed - span / 2.0, fixed + span / 2.0)

    def _cache_point_px(self) -> None:
        """把每个采样点投影到屏幕像素，缓存起来给点击拾取用。

        用 `proj3d.proj_transform` + `ax.transData` —— 这正是 mplot3d 自己把
        3D 点画到屏幕的同一套变换（实测与渲染像素误差 0.3 px，见 2026-11-05
        的验证）。不要在点击时现算：252 个点每次重投影会有可感的卡顿。
        """
        if self.P_raw is None or not len(self.P_raw):
            self._pt_px = None
            return
        try:
            M = self.ax.get_proj()
            x2, y2, _ = proj3d.proj_transform(
                self.P_raw[:, 0], self.P_raw[:, 1], self.P_raw[:, 2], M)
            self._pt_px = self.ax.transData.transform(
                np.column_stack([x2, y2]))
            # 记下当时的画布尺寸：窗口一拉，`transData` 就变了，缓存全作废。
            # 只靠 `resize_event` 不够（它触发时 transData 还没更新），
            # 所以在 `_pick_at` 里按 bbox 比对一次，最稳。
            self._pt_px_box = tuple(self.ax.bbox.bounds)
        except Exception:                       # noqa: BLE001
            self._pt_px = None                  # 拾取静默失效，不影响看图
            self._pt_px_box = None

    def _pick_at(self, x: float, y: float) -> None:
        """选中离 (x, y) 最近的采样点；点空处 = 取消选中。"""
        # 画布尺寸变了（用户拉过窗口 / 切过「保持立方体比例」）→ 缓存作废。
        # 不重算的话，点击会按上一个尺寸的位置命中，偏出去几十像素。
        if (self._pt_px is None
                or getattr(self, "_pt_px_box", None)
                != tuple(self.ax.bbox.bounds)):
            self._cache_point_px()
        if self._pt_px is None or not len(self._pt_px):
            return
        d = np.hypot(self._pt_px[:, 0] - x, self._pt_px[:, 1] - y)
        i = int(np.argmin(d))
        if not np.isfinite(d[i]) or d[i] > PICK_RADIUS_PX:
            if self.sel_idx is not None:
                self.sel_idx = None
                self._redraw()
            return
        self.sel_idx = i
        self._redraw()

    # ── 智能视角 / 坐标极值 ─────────────────────────────────────────────
    def _best_view_angles(self) -> tuple[float, float]:
        """搜出让轨迹在屏幕上**展开最充分**的 (elev, azim)。

        做法：对每个候选角度，把点云正交投影到屏幕平面，算投影包围盒面积，
        取面积最大的那个。面积大 = 三个方向的差异都被摊开在画面上，而不是
        被压成一条线或一坨。

        为什么不用固定等轴测：这段数据的跨度是 X 0.39 m / Y 0.14 m / Z 0.40 m，
        差异很大。固定 35°/-45° 时 Y 方向经常被压得几乎看不见 —— 用户之前
        抱怨的"Y 比例奇怪"有一半就是这个（2026-11-05）。

        这里的基向量与 mplot3d 的 `view_init(elev, azim)` 同一套约定：
        相机在 `(cos e·cos a, cos e·sin a, sin e)` 方向，屏幕右方向是
        `(-sin a, cos a, 0)`，上方向是两者叉积。所以搜出来的角度可以直接
        喂给 `view_init`。
        """
        P = getattr(self, "P_raw", None)
        if P is None or len(P) < 3:
            e, a, _r = VIEW_PRESETS[DEFAULT_VIEW]
            return float(e), float(a)
        Q = P - P.mean(axis=0)               # 以质心为原点，只看形状
        e0, e1, step = BEST_ELEV_RANGE
        best_area, best_e, best_a = -1.0, 35.264, -45.0
        for elev in range(e0, e1 + 1, step):
            er = np.radians(float(elev))
            ce, se = np.cos(er), np.sin(er)
            for azim in range(-180, 180, BEST_AZIM_STEP):
                ar = np.radians(float(azim))
                # 屏幕右方向与上方向（与相机正交）
                right = np.array([-np.sin(ar), np.cos(ar), 0.0])
                fwd = np.array([ce * np.cos(ar), ce * np.sin(ar), se])
                # 视线不能贴着任何一根轴：贴着谁，谁就在屏幕上退化成一点，
                # 它的刻度会堆到画布角落。面积再大也读不出数。
                if float(np.max(np.abs(fwd))) > BEST_MAX_AXIS_DOT:
                    continue
                up = np.cross(fwd, right)
                u = Q @ right
                v = Q @ up
                area = float((u.max() - u.min()) * (v.max() - v.min()))
                if area > best_area:
                    best_area, best_e, best_a = area, float(elev), float(azim)
        return best_e, best_a

    def _smart_view(self) -> None:
        """一键回到「最适合观察」：自动角度 + 自动缩放（按钮）。

        三步：① 搜最佳仰角/方位；② 范围回到数据包围盒；③ 按点云的实际占屏
        比例反解 zoom —— 不写死一个倍数，因为窗口大小、数据量级都会变，
        写死的值在某些会话里会顶出画布、在另一些里又缩成一小块。
        """
        elev, azim = self._best_view_angles()
        self.view = {"elev": elev, "azim": azim, "roll": 0.0}
        self.lims = None              # 让 `_redraw` 重新 fit 数据包围盒
        self.box_zoom = DEFAULT_BOX_ZOOM
        self.sel_idx = None           # 视角变了，原来的选中点位置已失效
        self._sync_roll_widget()
        self._redraw()
        self._fit_zoom_to_points()

    def _fit_zoom_to_points(self) -> None:
        """反解 `box_zoom`：让 3D 内容（含轴标签）**尽量大但不越界**。

        为什么用二分，而不是按 bbox 比例直接算：

        `ax.get_tightbbox()` 回答的是"至少需要多大才能把所有标签画下"，
        它的**下限就是 axes 本身** —— 内容一旦缩进 axes，返回值恒等于 axes
        bbox，再也看不出"还能放大多少"。按比例算会一直误判成"还超界"，
        把内容越缩越小（实测缩到 0.785 才停，比合适的值小了一大截）。

        所以只能试探：二分 zoom，看 tightbbox 是否越界。区间
        [0.35, 4.0]、4 次迭代，精度 0.23 —— 每次只改轴盒 + 一次 draw
        （约 40 ms），总共约 0.2 s，按钮点击完全可以接受。

        为什么不用点云包围盒当判据：点云占 82% 时轴标签已经被挤出画布了
        （tightbbox 要到 axes 的 1.29 倍才放得下，左下角刻度直接被切）。
        """
        # 图例 / 极值面板 / 选中信息 / 标题 / 小图都不是 3D 内容，
        # 量之前先藏起来，否则它们会把 tightbbox 撑大，算出的 zoom 偏小。
        hidden: list[tuple[object, bool]] = []
        for art in list(self.ax.texts):
            hidden.append((art, art.get_visible()))
            art.set_visible(False)
        _extras = [self.ax.get_legend(), self.ax.title]
        for art in _extras:
            if art is not None:
                hidden.append((art, art.get_visible()))
                art.set_visible(False)
        try:
            _inset_vis = self.ax_inset.get_visible()
            self.ax_inset.set_visible(False)
            try:
                lo, hi = 0.35, 4.0
                for _ in range(4):
                    mid = (lo + hi) / 2.0
                    self._set_zoom(mid)
                    self.canvas.draw()
                    try:
                        tb = self.ax.get_tightbbox(
                            self.canvas.get_renderer())
                    except Exception:                  # noqa: BLE001
                        return
                    box = self.ax.bbox
                    if tb is None or tb.width <= 1.0 or tb.height <= 1.0:
                        return
                    # 容差必须放宽：tightbbox 的**下限就是 axes 本身**，
                    # 内容全部放得下时它恰好等于 100%。用 0.995 判定会把
                    # "正好放下"也当成越界，一路缩到下限（实测缩成 0.35）。
                    # 5% 容差 = 只有真的溢出 5% 才算被裁。
                    over = (tb.width > box.width * 1.05
                            or tb.height > box.height * 1.05)
                    if over:
                        hi = mid
                    else:
                        lo = mid
                # 留 1% 余量：正好贴边时某些字体渲染会有 1px 溢出
                self._set_zoom(lo * 0.99)
            finally:
                self.ax_inset.set_visible(_inset_vis)
        finally:
            for art, vis in hidden:
                art.set_visible(vis)
            # 隐藏期间的投影不是最终状态，恢复后重算一次拾取缓存
            self.canvas.draw_idle()
            self._cache_point_px()

    def _draw_extremes(self) -> None:
        """在左上角常驻显示三轴坐标极值（自动随会话更新）。"""
        P = getattr(self, "P_raw", None)
        if P is None or not len(P):
            return
        rows = ["坐标极值 (m)"]
        for i, axis_name in enumerate(("X", "Y", "Z")):
            lo = float(P[:, i].min())
            hi = float(P[:, i].max())
            rows.append(f"{axis_name}  {lo:+.4f} ~ {hi:+.4f}   跨度 {hi - lo:.4f}")
        # 字体必须能显示中文：`monospace`(DejaVu Sans Mono) 没有汉字，
        # 会渲染成一串方框并刷 UserWarning。用项目里已有的中文回退链，
        # 数字靠 `%+.4f` 的定宽格式对齐（比例字体的数字宽度本来就接近）。
        # 存下 artist：选中面板要靠它的**实测高度**定位，不能再用常量估
        # —— 实测面板高 67.9px，常量写的 64，一乘上窗口高度就撞在一起了
        # （2026-11-05 用户截图：Z 那行被选中框盖住一半）。
        self._ext_artist = self.ax.text2D(
            0.012, EXTREMES_TOP_Y, "\n".join(rows), transform=self.ax.transAxes,
            fontsize=8.5, va="top", family=list(INSET_FONTS), color="#111827",
            linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#94a3b8", alpha=0.92),
            zorder=19)

    def _draw_selection(self) -> None:
        """把选中的采样点圈出来，并在左上角写出它的三维坐标。"""
        i = self.sel_idx
        if i is None or not (0 <= i < len(self.P_raw)):
            return
        q = self.P_raw[i]
        self.ax.scatter([q[0]], [q[1]], [q[2]], s=70, facecolors="none",
                        edgecolors="#dc2626", linewidths=1.6,
                        depthshade=False, zorder=20)
        meta = self.points[i] if i < len(self.points) else {}
        head = f"#{i}"
        if meta.get("frame") is not None:
            head = f"frame {meta['frame']}"
        t = meta.get("t")
        if isinstance(t, (int, float)):
            head += f"  t={float(t):.3f}s"
        body = (f"{head}\n"
                f"X = {q[0]:+.4f} m\n"
                f"Y = {q[1]:+.4f} m\n"
                f"Z = {q[2]:+.4f} m")
        # `text2D` 用 axes 坐标，旋转/缩放都不会跟着跑；`text` 会跟着 3D 点飞。
        # 排在极值面板**下方**。两个坑都踩过（2026-11-05）：
        # ① 基准要用面板顶部 `EXTREMES_TOP_Y`（0.985），不是 1.0 ——
        #    差那 0.015 就是 10px，正好压住 Z 那一行；
        # ② 面板高度要**实测**（`get_window_extent`），常量估不准：
        #    实际 67.9px 而常量写 64，窗口越高重叠越明显。
        _h = max(1.0, float(self.ax.bbox.height))
        _panel_h = float(EXTREMES_PANEL_PX)
        _art = getattr(self, "_ext_artist", None)
        if _art is not None:
            try:
                _panel_h = float(_art.get_window_extent(
                    self.canvas.get_renderer()).height)
            except Exception:                       # noqa: BLE001
                pass                                # 量不到就退回常量
        _y = EXTREMES_TOP_Y - (_panel_h + 8.0) / _h
        self.ax.text2D(
            0.012, _y, body, transform=self.ax.transAxes,
            fontsize=9, va="top", family="monospace", color="#111827",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#dc2626", alpha=0.94),
            zorder=21)

    def _apply_view(self) -> None:
        """把 ``self.view`` 与投影方式应用到 ax。

        `ax.clear()` 会把 `view_init` 的角度、投影类型全部打回默认，
        所以每次 `_redraw` 都必须在画完图元后重新应用一次。
        """
        try:
            self.ax.view_init(elev=self.view["elev"], azim=self.view["azim"],
                              roll=self.view["roll"])
        except TypeError:            # matplotlib < 3.6 没有 roll 参数
            self.ax.view_init(elev=self.view["elev"], azim=self.view["azim"])
        try:
            self.ax.set_proj_type("ortho" if self.ortho.get() else "persp")
        except (AttributeError, ValueError):
            pass
        self._bind_mouse_buttons()     # clear() 后重设，避免键位丢
        self._hide_depth_axis()

    def _hide_depth_axis(self) -> None:
        """视线几乎与某根轴平行时，把那根轴的刻度与标签藏掉。

        前后视（elev=±90）看的是 XY 平面、俯仰视看的是 XZ 平面 —— 此时
        深度轴在屏幕上退化成一点，matplotlib 会把它的刻度和轴名全挤在
        角落里糊成一团（实测 89.5° 也救不了）。CAD 的正视图本来也不标
        深度轴，藏掉更干净。阈值 0.995 ≈ 离轴 5.7° 以内。

        恢复靠 `_redraw` 里的 `ax.clear()`：所以切方向走的是重画，
        不是只调一次 `view_init`（否则切回等轴测时轴标签还是空的）。
        """
        e = np.radians(self.view["elev"])
        a = np.radians(self.view["azim"])
        eye = (np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e))
        k = int(np.argmax(np.abs(eye)))
        if abs(eye[k]) < 0.995:
            return
        axis = (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis)[k]
        axis.set_ticklabels([])
        axis.label.set_text("")

    def _sync_roll_widget(self) -> None:
        """把滑块/数字同步到 ``self.view["roll"]``，且不触发回调递归。"""
        self._syncing_view = True
        try:
            self.roll_scale.set(self.view["roll"])
        finally:
            self._syncing_view = False
        self.roll_lbl.configure(text=f"{self.view['roll']:.0f}°")

    def _set_roll(self, value: float) -> None:
        self.view["roll"] = max(-180.0, min(180.0, float(value)))
        self._sync_roll_widget()
        self._apply_view()
        self.canvas.draw_idle()

    def _on_roll(self, value: str) -> None:
        if self._syncing_view:
            return
        self.view["roll"] = float(value)
        self.roll_lbl.configure(text=f"{float(value):.0f}°")
        self._apply_view()
        self.canvas.draw_idle()

    def _on_view_preset(self, _event=None) -> None:
        name = self.view_box.get()
        if name not in VIEW_PRESETS:
            return
        e, a, r = VIEW_PRESETS[name]
        self.view = {"elev": e, "azim": a, "roll": r}
        self._sync_roll_widget()
        # 走重画而不是只 view_init：`_hide_depth_axis` 可能藏了某根轴，
        # 只有 `clear()` 能把它恢复回来。
        self._redraw()

    def _on_plane_toggle(self) -> None:
        # 平面是图元（surface + 网格线），不是视角属性 —— 必须整幅重画。
        self._redraw()

    def _on_proj_toggle(self) -> None:
        self._apply_view()
        self.canvas.draw_idle()

    # ---- 缩放/平移：范围持久化 + 绕点滚轮缩放 ----
    def _zoom_around(self, factor: float, px: float, py: float, pz: float) -> None:
        """把三轴范围绕**数据点 (px,py,pz)** 缩放 ``factor``（<1 放大）。

        这是"指定放大区域"的实现：滚轮时取鼠标在 3D 图上指向的数据点，
        以它为锚收缩/扩张范围 —— 鼠标指着轨迹的哪一段，那段就留在屏幕原位
        并放大，而不是绕数据中心缩（后者会把想看的地方推出画面）。
        """
        lo, hi = self._current_lims()
        for i, (loi, hii, c) in enumerate(zip(lo, hi, (px, py, pz))):
            new_lo = c - (c - loi) * factor
            new_hi = c + (hii - c) * factor
            lo[i], hi[i] = new_lo, new_hi
        self._set_lims(lo, hi)

    def _set_zoom(self, zoom: float) -> None:
        """设置整幅缩放倍数（1 = 原样，>1 放大）。"""
        # 限幅：太小看不清、太大会把内容顶出 axes 区域（用户会以为程序坏了）
        self.box_zoom = max(0.35, min(6.0, float(zoom)))
        self._apply_box()

    def _on_scroll(self, event) -> None:
        """滚轮 = **整幅缩放**；Shift+滚轮 = 定点放大（改三轴范围）。

        ## 为什么要分两种（2026-11-05 用户："缩放只有曲线缩放没有背景坐标图
        ## 缩放，很别扭"）

        原来滚轮只改 `lims`。但 3D 轴的盒子大小在画布上是**固定**的，而刻度
        数量是自动选的 —— 范围缩小 2 倍后，盒子里依旧是那几条网格线，屏幕上
        的间距几乎不变，只有曲线按新范围被放大。于是看起来像"背景一动不动"。

        真正的"缩放"应该是整幅画面变大：数据、网格线、刻度数字、轴标签一起。
        那是 `set_box_aspect(zoom=)` 的语义（matplotlib ≥3.9），不是 `lims`。

        定点放大（只改范围、把鼠标指的位置留在屏幕原位）仍然有用 —— 它是
        "凑近看某一段"，所以挪到 Shift+滚轮，不再占用默认滚轮。
        """
        if event.inaxes is not self.ax:
            return
        factor = ZOOM_STEP if event.button == "up" else 1.0 / ZOOM_STEP

        if event.key == "shift":
            # 定点放大：锚点取鼠标在轴框内的归一化位置
            lo, hi = self._current_lims()
            bbox = self.ax.get_window_extent()
            fx = 0.5
            fy = 0.5
            if event.x is not None and bbox.width > 0 and bbox.height > 0:
                fx = (event.x - bbox.x0) / bbox.width
                fy = (event.y - bbox.y0) / bbox.height
            fx = max(0.0, min(1.0, fx))
            fy = max(0.0, min(1.0, fy))
            # 屏幕 y 向上（bbox 顶部 = 数据 Y 较大侧），所以 fy 越大 → Y 越大。
            cx = lo[0] + (hi[0] - lo[0]) * fx
            cy = lo[1] + (hi[1] - lo[1]) * fy
            cz = (lo[2] + hi[2]) / 2.0
            self._zoom_around(factor, cx, cy, cz)
        else:
            # 整幅缩放：factor<1 表示放大 → zoom 要变大
            self._set_zoom(self.box_zoom / factor)
        self.canvas.draw_idle()

    def _current_lims(self) -> tuple[list[float], list[float]]:
        """当前显示范围 (lo, hi)，各 3 元素；未初始化则按数据包围盒算一次。

        与 `_limits()` 同构，方便直接解包。`self.lims` 内部存 `[lo, hi]`。
        """
        if self.lims is None:
            lo, hi = self._limits()
            self.lims = [list(lo), list(hi)]
        return self.lims[0], self.lims[1]

    def _set_lims(self, lo, hi) -> None:
        self.lims = [list(lo), list(hi)]
        self._apply_lims()

    def _apply_lims(self) -> None:
        """把 `self.lims` 写到 ax，并同步轴盒比例与整幅缩放。"""
        if self.lims is None:
            return
        lo, hi = self.lims
        self.ax.set_xlim(lo[0], hi[0])
        self.ax.set_ylim(lo[1], hi[1])
        self.ax.set_zlim(lo[2], hi[2])
        self._apply_box()

    def _on_tick_edit(self, _event=None) -> None:
        """分度值改了（回车或失焦）就立刻重设刻度。"""
        self._apply_tick_steps()
        self.canvas.draw_idle()

    def _apply_tick_steps(self) -> None:
        """按侧栏填的分度值设三轴刻度间隔；留空/非法/过密则回落到自动 4 档。

        为什么要挡"过密"：填 0.001 会让这段 3 米长的数据出现 3000 条网格线，
        matplotlib 会画上几十秒、界面看起来像卡死。超过 20 条就退回自动。
        """
        if self.lims is None:
            return
        lo, hi = self.lims
        for i, (axis, key) in enumerate(
                ((self.ax.xaxis, "x"), (self.ax.yaxis, "y"),
                 (self.ax.zaxis, "z"))):
            txt = self.tick_vars[key].get().strip()
            step = 0.0
            try:
                step = float(txt) if txt else 0.0
            except ValueError:
                step = 0.0
            span = max(1e-9, hi[i] - lo[i])
            if step > 0 and span / step <= 20:
                axis.set_major_locator(mticker.MultipleLocator(step))
                # 网格按 step 画（要密就密），但**标签自动疏**：3D 轴在屏幕上
                # 很短，12 个标签会糊成一片（实测 0.05 分度就是这个下场）。
                # 目标是最多 ~10 个标签，多出来的 tick 只画网格不写字。
                # `MultipleLocator` 会向**两端各外扩**一个刻度（-0.061 的
                # 下限会给出 -0.08），所以估算要 +2；目标是屏幕上 ~9 个标签。
                _n_est = span / step + 2
                every = max(1, int(_n_est // 9) + (1 if _n_est % 9 else 0))
                def _fmt(v, _pos=None, _s=step, _e=every):
                    """只标每 _e 个刻度；并把浮点误差的 0 写成 "0"。

                    `MultipleLocator(0.05)` 给出的"0"实际是 5.55e-17 这种值，
                    直接 `%g` 会显示成科学计数法 —— 轴上看不见一个正常的 0。
                    """
                    k = round(v / _s)
                    if abs(v / _s - k) > 1e-6 or int(k) % _e:
                        return ""
                    if abs(v) < _s * 1e-6:
                        return "0"
                    return f"{v:g}"

                axis.set_major_formatter(mticker.FuncFormatter(_fmt))
            else:
                axis.set_major_locator(mticker.MaxNLocator(4))
                axis.set_major_formatter(mticker.ScalarFormatter())

    def _toggle_fullscreen(self) -> None:
        """全屏 / 退出全屏（Esc 也能退）。

        **这个按钮只管窗口**（把窗口撑到屏幕），不改变坐标轴在画布里的占比 ——
        那件事由 `_apply_layout` 的默认"铺满"负责。名字带「窗口」就是为了
        不再和"坐标轴铺满"混淆。
        """
        self._fullscreen = not getattr(self, "_fullscreen", False)
        top = self.root.winfo_toplevel()
        try:
            top.attributes("-fullscreen", self._fullscreen)
        except tk.TclError:
            # 个别窗口管理器不认 -fullscreen：退化成最大化
            try:
                top.state("zoomed" if self._fullscreen else "normal")
            except tk.TclError:
                pass
        try:
            self.full_btn.configure(text="退出全屏" if self._fullscreen else "窗口全屏")
        except tk.TclError:
            pass
        if self._fullscreen:
            top.bind("<Escape>", self._on_escape_fullscreen)

    def _on_escape_fullscreen(self, _event=None) -> None:
        if getattr(self, "_fullscreen", False):
            self._toggle_fullscreen()

    def _apply_layout(self) -> None:
        """摆主图 axes 的盒子（铺满 or 正方形）。

        默认**铺满**画布：3D 的轴标签由 mplot3d 摆在 axes 内部，不需要外边距，
        所以容器越大、立方体越大。

        「保持立方体比例」勾上时退回正方形（等轴测投影本来就是圆的，塞进宽屏
        必然要横向拉伸；勾上就换成留白）。正方形边长取 ``min(宽, 可用高)``，
        水平居中。两个模式都能在窗口 resize 后正确重算 —— 尺寸是从
        ``fig.get_size_inches()`` 现读的，不是写死的比例。
        """
        fw, fh = self.fig.get_size_inches()
        top = 0.96                      # 顶上 4% 留给标题
        keep = getattr(self, "keep_cube", None)
        if keep is None or not keep.get():
            self.ax.set_position([0.0, 0.0, 1.0, top])
            return
        side_in = min(float(fw), float(fh) * top)
        w = side_in / float(fw)
        h = side_in / float(fh)
        self.ax.set_position([(1.0 - w) / 2.0, (top - h) / 2.0, w, h])

    def _on_layout_mode(self) -> None:
        """切换「保持立方体比例」后立刻重排 + 重画。"""
        self._apply_layout()
        self.canvas.draw_idle()

    def _on_axes_mode(self) -> None:
        """切换"等比例坐标"后立刻重画（只改显示比例，不动数据与范围）。"""
        self._apply_box()
        self.canvas.draw_idle()

    def _apply_box(self) -> None:
        """按显示比例 + 整幅缩放设置轴盒。

        `aspect` 只有**比例**意义，必须跟着 `lims` 更新：否则滚轮改了范围、
        比例却还是旧的，曲线会被非均匀拉伸，而盒子看起来纹丝不动 —— 正是
        用户说的"背景没有跟着缩放"。

        **两种比例**（2026-11-05）：
          - 默认 ``(1,1,1)``：三轴各自拉满。球在 Y 方向只走了 1.7 cm 时，等比例
            那张图里 Y 轴只剩短短一截、刻度全挤在角落（用户原话"Y 方向实在是
            比例奇怪"）。拉满后三个维度都能看清。
          - 勾了「等比例坐标」：``(dx,dy,dz)``，1 米在三个轴上等长 —— 形状是真实
            的，但短的那个轴会真的短。要做几何判断时用这个。
        """
        if self.lims is None:
            return
        lo, hi = self.lims
        if self.equal_axes.get():
            aspect = (max(1e-9, hi[0] - lo[0]),
                      max(1e-9, hi[1] - lo[1]),
                      max(1e-9, hi[2] - lo[2]))
        else:
            aspect = (1.0, 1.0, 1.0)
        try:
            self.ax.set_box_aspect(aspect, zoom=self.box_zoom)
        except TypeError:                   # matplotlib < 3.9 没有 zoom 参数
            self.ax.set_box_aspect(aspect)

    # ---- 鼠标：Shift + 左键拖动 = 绕视线滚转 ----
    def _on_mouse_press(self, event) -> None:
        if event.button == 1 and event.key == "shift":
            self._roll_drag = (event.x, self.view["roll"])
            # 关掉原生的 azim/elev 旋转，否则同一个拖动会同时改两个角
            self.ax.set_navigate(False)

    def _on_mouse_move(self, event) -> None:
        if self._roll_drag is None or event.x is None:
            return
        x0, r0 = self._roll_drag
        self._set_roll(r0 + (event.x - x0) * 0.5)

    def _on_mouse_release(self, _event) -> None:
        if self._roll_drag is not None:
            self._roll_drag = None
            self.ax.set_navigate(True)

    def _open_help(self) -> None:
        """直接打开操作说明的「3D 查看器」分区。

        **必须传 `self.root`**：`Viewer3D` 是普通类不是 widget，传 self 会在
        `open_help` 里 `master.winfo_toplevel()` 抛 AttributeError ——
        tkinter 回调里的异常不会弹窗，表现就是"按钮点了没反应"
        （2026-11-05 用户截图）。
        """
        from apps.help_window import open_help

        open_help(self.root, section="viewer")

    def _reset_view(self) -> None:
        """回到默认视角 + 默认范围（按钮 / Home 键）。"""
        e, a, r = VIEW_PRESETS[DEFAULT_VIEW]
        self.view = {"elev": e, "azim": a, "roll": r}
        self.view_box.set(DEFAULT_VIEW)
        self.lims = None               # 让下一次 `_redraw` 重新 fit 数据包围盒
        self._sync_roll_widget()
        self._redraw()

    def _draw_planes(self) -> None:
        """画勾选的坐标平面（过原点；范围取当前轴范围）。

        为什么贴边而不是死画在 0：相机系原点在光心，轨迹常在 z=1~5 m。
        若某根轴的范围不含 0，平面画在 0 就完全看不见 —— 用户勾了没反应。
        所以夹到最近的可见位置，并在面板上注明"这个平面其实在 X=…"。
        """
        self._plane_note = ""
        lo, hi = self._limits()
        notes: list[str] = []
        for name, (axis, color) in PLANE_STYLES.items():
            if not self.show_planes[name].get():
                continue
            if lo[axis] <= 0.0 <= hi[axis]:
                c = 0.0
            else:
                c = lo[axis] if 0.0 < lo[axis] else hi[axis]
                notes.append(f"{name} 在 {_AXIS_NAMES[axis]}={c:.3g}")
            other = [i for i in range(3) if i != axis]
            u0, u1 = lo[other[0]], hi[other[0]]
            v0, v1 = lo[other[1]], hi[other[1]]
            U, V = np.meshgrid([u0, u1], [v0, v1])
            C = np.full_like(U, c)
            X, Y, Z = ((C, U, V) if axis == 0 else
                       (U, C, V) if axis == 1 else (U, V, C))
            self.ax.plot_surface(X, Y, Z, color=color, alpha=0.06,
                                 linewidth=0, antialiased=False, shade=False)
            # 网格线：两个方向各 3 条，够看出是平面，又不会糊住轨迹
            for fixed in np.linspace(v0, v1, 4):
                p0 = [0.0, 0.0, 0.0]
                p1 = [0.0, 0.0, 0.0]
                p0[axis] = p1[axis] = c
                p0[other[0]], p1[other[0]] = u0, u1
                p0[other[1]] = p1[other[1]] = fixed
                self.ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                             color=color, lw=0.5, alpha=0.30)
            for fixed in np.linspace(u0, u1, 4):
                p0 = [0.0, 0.0, 0.0]
                p1 = [0.0, 0.0, 0.0]
                p0[axis] = p1[axis] = c
                p0[other[1]], p1[other[1]] = v0, v1
                p0[other[0]] = p1[other[0]] = fixed
                self.ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                             color=color, lw=0.5, alpha=0.30)
            pos = [0.0, 0.0, 0.0]
            pos[axis] = c
            pos[other[0]], pos[other[1]] = u1, v1
            self.ax.text(pos[0], pos[1], pos[2], name,
                         color=color, fontsize=8, alpha=0.85)
        self.plane_note.configure(text="；".join(notes))

    def _smooth_active(self) -> bool:
        """当前是否真的在用平滑点：勾了**且**至少有一段被平滑处理过。

        两个条件都要：段全太短时 `P_smooth == P_raw`，此时报"已平滑"是撒谎，
        小图要如实写"段太短，未平滑"。
        """
        return bool(self.use_smooth.get()) and bool(self.smooth_valid.any())

    def _capture_lims(self) -> None:
        """在 `ax.clear()` 之前，把 ax 当前范围回读进 `self.lims`。

        为什么需要：原生中键平移 / 右键缩放会**直接改 ax 的 lim**（不走
        `self.lims`），而 `ax.clear()` 会把 lim 重置回 autoscale。所以每次
        `_redraw` 在 clear 之前先回读 —— 用户刚平移/缩放过的范围就留住了，
        播放下一帧不再弹回。首次（`self.lims is None`）跳过，交给 `_current_lims`
        按数据包围盒 fit。
        """
        if self.lims is None:
            return
        self.lims = [
            [float(self.ax.get_xlim()[0]), float(self.ax.get_ylim()[0]),
             float(self.ax.get_zlim()[0])],
            [float(self.ax.get_xlim()[1]), float(self.ax.get_ylim()[1]),
             float(self.ax.get_zlim()[1])],
        ]

    def _redraw(self) -> None:
        self._capture_lims()           # clear 会重置范围，先回读用户改过的范围
        self.ax.clear()
        # 勾了「平滑」就用平滑点画主图，否则用原始点。**范围/视角不变**：
        # `_limits` 始终按原始点算（平滑不改变量级，改范围只会让两张图对不上）。
        P = self.P_smooth if self._smooth_active() else self.P_raw
        xs, ys, zs = P[:, 0], P[:, 1], P[:, 2]
        self.ax.plot(xs, ys, zs, color="#93c5fd", linewidth=1.2, alpha=0.8, label="trajectory")
        k = min(self.idx + 1, self.n)
        self.ax.plot(xs[:k], ys[:k], zs[:k], color="#2563eb", linewidth=2.0, label="trail")
        # 真实采样点：`trajectory` 里的每个点就是一次 matched 的帧
        # （252 个 matched 帧逐点对应，不是插值出来的）。画成小点，让人看出
        # "线是这些实测点连起来的"；点太小会看不见，太大会糊成一条粗带，
        # 7pt² 在 252 点的密度下刚好是"一串珠子"（2026-11-05 用户要求）。
        # **始终用 P_raw**：勾了平滑时线是平滑的，但采样点仍该是实测位置。
        self.ax.scatter(self.P_raw[:, 0], self.P_raw[:, 1], self.P_raw[:, 2],
                        s=SAMPLE_DOT_SIZE, c=SAMPLE_DOT_COLOR,
                        alpha=SAMPLE_DOT_ALPHA, depthshade=False,
                        edgecolors="none", label="samples")
        cur = self.points[self.idx]
        self.ax.scatter([cur["xyz"][0]], [cur["xyz"][1]], [cur["xyz"][2]],
                        color="#f97316", s=80, depthshade=False, label="ball")
        self.ax.set_xlabel("X right (m)")
        self.ax.set_ylabel("Y down (m)")
        self.ax.set_zlabel("Z forward (m)")
        # 刻度间隔：用户填了分度值就用它（`MultipleLocator`），否则自动 4 档。
        # 必须在 `clear()` 之后设 —— 它会重置 locator。
        self._apply_tick_steps()
        # 标题带中文（"[单目]"），默认字体 DejaVu Sans 没有汉字，会渲染成方框
        # 并刷 UserWarning —— 指定项目里的中文回退链（2026-11-05 实测单目会话）。
        self.ax.set_title(f"{'[单目] ' if self.mono else ''}"
                          f"{self.data.get('session', '')} · {self.idx + 1}/{self.n}",
                          fontfamily=list(INSET_FONTS))
        self.ax.legend(loc="upper right")
        # 顺序固定：范围（持久）→ 平面按这个范围铺 → 最后套视角/投影。
        # `clear()` 已经把视角打回默认，所以 `_apply_view` 必须在最后。
        # **范围只在首次/重置时 fit**（`_current_lims` 里判 None），
        # 之后直接用 `self.lims` —— 这样用户 pan/zoom 的范围播放时不弹回。
        lo, hi = self._current_lims()
        self.ax.set_xlim(lo[0], hi[0])
        self.ax.set_ylim(lo[1], hi[1])
        self.ax.set_zlim(lo[2], hi[2])
        self._apply_layout()       # 容器先摆好（铺满/正方形），再谈比例
        self._apply_box()          # 比例 + 整幅缩放（重画后 zoom 不能丢）
        self._draw_planes()
        self._apply_view()
        # 拾取缓存必须在 `_apply_view()` **之后** —— 视角变了投影就变了，
        # 先算会得到上一帧的位置，点哪都选不中。
        self._cache_point_px()
        self._draw_extremes()
        self._draw_selection()
        self._redraw_inset()
        self.canvas.draw_idle()
        self._update_video()
        self._status()

    def _redraw_inset(self) -> None:
        """重画右下角俯视小图（画法在模块级 `draw_xz_inset`，见那里为什么）。"""
        draw_xz_inset(self.ax_inset, self.P_raw, self.P_smooth,
                      self._smooth_active())

    def _render_video(self) -> None:
        for raw, canvas, attr, empty in (
            (self._left_raw, self.video_l, "_left_photo", ""),
            # 单目没有第二路：右格写清楚"这里本来就不该有画面"，
            # 否则用户看到空白会以为是解码失败。
            (self._right_raw, self.video_r, "_right_photo",
             "单目会话只有一路画面" if self.mono else ""),
        ):
            canvas.delete("all")
            w = max(2, canvas.winfo_width())
            h = max(2, canvas.winfo_height())
            if w <= 4 or h <= 4:
                w, h = 640, 360
            if raw is None:
                canvas.create_text(
                    w // 2, h // 2, fill="#94a3b8", anchor="center",
                    text=empty or self.video_error or (
                        "视频已关闭（--no-video）" if getattr(self, "video_disabled", False)
                        else "未找到 annotated.mp4"))
                continue
            # 填满 pane 且不拉伸：等比放大到覆盖，再中心裁剪。
            rh, rw = raw.shape[:2]
            scale = max(w / rw, h / rh)
            nw, nh = max(w, int(round(rw * scale))), max(h, int(round(rh * scale)))
            img = cv2.resize(raw, (nw, nh), interpolation=cv2.INTER_AREA)
            x0 = max(0, (nw - w) // 2)
            y0 = max(0, (nh - h) // 2)
            img = img[y0:y0 + h, x0:x0 + w]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # `master=canvas` 不能省：同一个进程里若有**另一个** Tk 根
            # （测试里先建过 App、或以后把查看器嵌进别的窗口），不带 master 的
            # PhotoImage 会绑到"默认根"上，用在另一个 Tk 的 canvas 上就是
            # `image "pyimageN" doesn't exist`。独立进程跑看不出来。
            photo = ImageTk.PhotoImage(Image.fromarray(rgb), master=canvas)
            setattr(self, attr, photo)
            canvas.create_image(0, 0, anchor="nw", image=photo)

    def _update_video(self) -> None:
        """按当前三维点的 frame 号取 annotated.mp4 的同帧，切成左右两格。"""
        if self.cap is None:
            self._render_video()
            return
        frame = int(self.points[self.idx].get("frame", self.idx))
        if self.cap_frames and frame >= self.cap_frames:
            # 视频比轨迹短 —— 不是解码器的问题，是这个 mp4 不完整。
            # 说清"差多少"，用户才知道是文件坏了而不是程序坏了。
            self.video_error = (
                f"视频只有 {self.cap_frames} 帧，当前要点第 {frame} 帧 —— "
                f"annotated.mp4 不完整（生成中断或被覆盖）")
            self._left_raw = self._right_raw = None
            self._render_video()
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = self.cap.read()
        if not ok or img is None:
            # 帧号在范围内却读不出来：真正的解码失败（文件损坏/编码不支持）。
            self.video_error = (f"frame {frame} 解码失败"
                                + (f"（视频共 {self.cap_frames} 帧）"
                                   if self.cap_frames else ""))
            self._left_raw = self._right_raw = None
            self._render_video()
            return
        self.video_error = ""
        h, w = img.shape[:2]
        if self.mono:
            # 单目视频是一路画面，整幅放进左格（右格保持 None，见 _render_video）
            self._left_raw = img
            self._right_raw = None
        else:
            self._left_raw = img[:, : w // 2].copy()
            self._right_raw = img[:, w // 2:].copy()
        self._render_video()

    def _status(self) -> None:
        p = self.points[self.idx]
        frame = p.get("frame", self.idx)
        t = p.get("t", 0.0)
        x, y, z = p["xyz"]
        mode = ("mono" if self.mono
                else self.data.get("stereo_mode", "parallel"))
        self.status.configure(
            text=f"frame {frame}   t={float(t):.3f}s   "
                 f"X={float(x):.4f}  Y={float(y):.4f}  Z={float(z):.4f} m   "
                 f"模式={mode}   "
                 f"坐标={self.data.get('coordinate_system', '?')}   "
                 f"视角 {self.view['elev']:.0f}/{self.view['azim']:.0f}/"
                 f"{self.view['roll']:.0f}°")

    # ------------------------------------------------------------------
    def _step_frames(self, delta: int) -> None:
        """前进/后退 delta 帧（按钮）。

        夹在 ``[0, n-1]`` 内，并且**顺手停掉播放** —— 播放时按步进，
        下一次 tick 会立刻把帧号推走，用户会觉得按钮没反应。
        """
        if getattr(self, "playing", False):
            self._toggle_play()
        self.idx = max(0, min(self.n - 1, self.idx + int(delta)))
        self._set_slider(self.idx)
        self._redraw()

    def _on_slider(self, value: str) -> None:
        if self._syncing:
            return
        self.idx = max(0, min(self.n - 1, int(round(float(value)))))
        self._redraw()

    def _set_slider(self, idx: int) -> None:
        self._syncing = True
        try:
            self.slider.set(idx)
        finally:
            self._syncing = False

    def _step(self, delta: int) -> None:
        self.idx = max(0, min(self.n - 1, self.idx + delta))
        self._set_slider(self.idx)
        self._redraw()

    def _on_speed(self, _event=None) -> None:
        self.speed = {"0.25x": 0.25, "0.5x": 0.5, "1x": 1.0, "2x": 2.0}.get(
            self.speed_box.get(), 1.0)

    def _on_toggle_smooth(self) -> None:
        """勾选框变化：**复用 `_redraw`**（主图 + 小图 + 状态条同一条路径）。

        新开一条"只重画小图"的分支，主图和小图就会有两条刷新时序 ——
        播放中切开关时，一张图新、一张图旧，而没有任何报错。
        """
        self._redraw()

    def _toggle_play(self) -> None:
        self.playing = not self.playing
        self.play_btn.configure(text="暂停" if self.playing else "播放")
        if self.playing:
            self._tick()

    def _tick(self) -> None:
        if not self.playing:
            return
        self.idx = (self.idx + 1) % self.n
        self._set_slider(self.idx)
        self._redraw()
        self.root.after(max(16, int(33 / max(self.speed, 1e-6))), self._tick)


def main() -> int:
    ap = argparse.ArgumentParser(description="离线 3D 轨迹查看器（读 visualization.json）")
    # 省略 source = **自动打开最新的会话**（2026-11-05 用户："每次打开应该
    # 默认打开最新的会话的结果"）。之前是必填参数，直接跑 `view_3d.py`
    # 只会报 usage，用户还得从主界面挑一次。
    ap.add_argument("source", nargs="?", default=None,
                    help="会话目录或 visualization.json / traj3d.json"
                         "（省略 = 最新的一条结果）")
    ap.add_argument("--check", action="store_true", help="只校验数据，不开窗口")
    ap.add_argument("--video", default="", help="同步视频（默认找 derived/annotated.mp4）")
    ap.add_argument("--no-video", action="store_true", help="不加载追踪视频")
    args = ap.parse_args()
    if args.source is None:
        found = scan_sources(limit=1)
        if not found:
            print("[错误] 没有任何会话有三维结果"
                  "（data/stereo、data/mono 下都没有 derived/）")
            return 1
        args.source = str(found[0])
        # 明确说一句"替你选的是哪条"，否则用户不知道看的是哪次实验
        print(f"[自动] 打开最新会话：{found[0].parent.parent.name}")
    try:
        data = load_view(args.source)
    except Exception as exc:                                # noqa: BLE001
        print(f"[错误] {exc}")
        return 1
    lo, hi = bounds_of(data["trajectory"])
    if args.check:
        mono = is_mono(data)
        print(f"文件：{data['_path']}")
        print(f"schema={data.get('schema')} version={data.get('version')} "
              f"mode={'mono' if mono else 'stereo'}")
        print(f"session={data.get('session')} points={len(data['trajectory'])}")
        print(f"coordinate_system={data.get('coordinate_system')} unit={data.get('unit')}")
        if mono:
            # 尺度来源必须进 --check 的输出：它是这份数据能不能当测量的前提
            print(f"ball_diameter_mm={data.get('ball_diameter_mm')} fx={data.get('fx')}")
        print(f"bounds min={lo} max={hi}")
        if args.no_video:
            print("video=disabled (--no-video)")
        else:
            p = Path(data["_path"])
            session = p.parent.parent if p.parent.name == "derived" else p.parent
            video = Path(args.video) if args.video else (session / "derived" / "annotated.mp4")
            print(f"video={video}")
            cap = cv2.VideoCapture(str(video)) if video.exists() else None
            if cap is None or not cap.isOpened():
                print("[错误] 视频打不开或不存在")
                if cap is not None:
                    cap.release()
                return 1
            vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            vn = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            max_frame = max(int(p.get("frame", 0)) for p in data["trajectory"])
            print(f"video_frames={vn} size={vw}x{vh} max_point_frame={max_frame}")
            if not mono and vw < vh * 1.8:
                # 只有双目要求左右并排；单目就是一路画面，卡 2:1 会把它判死
                print("[错误] 视频不是左右并排（需要约 2:1）")
                return 1
            if max_frame >= vn:
                print(f"[错误] 轨迹帧 {max_frame} 超出视频 {vn} 帧")
                return 1
        return 0
    root = tk.Tk()
    root.geometry("1500x950")
    Viewer3D(root, data, args.video or None, auto_video=not args.no_video)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
