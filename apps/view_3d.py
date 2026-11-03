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
INSET_RECT = (0.815, 0.100, 0.170, 0.335)

#: 曲线配色（与主图一致：灰=原始、橙=平滑/小球）。
#: 原始线用中灰而不是浅灰：浅灰在图例里几乎看不见（图例继承线条的 alpha）。
INSET_RAW_COLOR = "#64748b"
INSET_SMOOTH_COLOR = "#f97316"

#: 小图里的中文（matplotlib 默认字体没有汉字，会画成方框）。按顺序回退。
INSET_FONTS = ("Microsoft YaHei", "SimHei", "DejaVu Sans")


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
    found: dict[Path, Path] = {}
    for base in bases:
        if not base.is_dir():
            continue
        for name in ("visualization.json", "traj3d.json"):
            for f in base.glob(f"*/derived/{name}"):
                sess = f.parent.parent
                # 同一个会话两种都写时，优先 visualization.json（它带视频叠加信息）
                if sess not in found or f.name == "visualization.json":
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
        # 2×2：左上左视频 / 右上右视频 / 左下 3D / 右下待定
        grid = ttk.Frame(self.root)
        self._grid = grid
        grid.rowconfigure(0, weight=0)
        grid.rowconfigure(1, weight=1)
        for c in (0, 1):
            grid.columnconfigure(c, weight=1, uniform="pane")
        grid.bind("<Configure>", self._fit_video_rows)

        # Canvas 的 request size 固定为 1×1：图像只作为 item 画进去，
        # 不会反过来把 grid 撑大（之前用 Label + PhotoImage 会正反馈放大到全屏）。
        self.video_l = tk.Canvas(grid, bg="#0f172a", highlightthickness=0,
                                 width=1, height=1)
        self.video_r = tk.Canvas(grid, bg="#0f172a", highlightthickness=0,
                                 width=1, height=1)
        self.video_l.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        self.video_r.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
        self._left_raw = None
        self._right_raw = None
        self._left_photo = None
        self._right_photo = None
        self.video_l.bind("<Configure>", lambda _e: self._render_video())
        self.video_r.bind("<Configure>", lambda _e: self._render_video())

        plot_frame = ttk.Frame(grid)
        plot_frame.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
        self.fig = plt.Figure(figsize=(5.5, 3.6), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.90)
        # 右下角俯视小图：**另叠一块 axes**，不碰主图的 subplot 布局
        # （主图的范围/视角/色标有别的判据盯着）。zorder 抬高让它盖在 3D 轴之上。
        self.ax_inset = self.fig.add_axes(INSET_RECT)
        self.ax_inset.set_zorder(10)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        reserved = ttk.LabelFrame(grid, text="右下 · 待定", padding=10)
        reserved.grid(row=1, column=1, sticky="nsew", padx=2, pady=2)
        ttk.Label(reserved, text="预留区（下一阶段再定）\n\n"
                                "候选：误差曲线 / 速度 / 预测 / 多会话对比",
                  style="Hint.TLabel", justify="left").pack(anchor="nw")

        bar = ttk.Frame(self.root, padding=(8, 4))
        bar.pack(fill="x")
        self.play_btn = ttk.Button(bar, text="播放", command=self._toggle_play, width=8)
        self.play_btn.pack(side="left")
        ttk.Button(bar, text="重置3D视角", command=self._reset_view, width=10).pack(side="left", padx=4)
        ttk.Label(bar, text="帧").pack(side="left", padx=(10, 2))
        self.slider = ttk.Scale(bar, from_=0, to=max(0, self.n - 1),
                                command=self._on_slider, orient="horizontal")
        self.slider.pack(side="left", fill="x", expand=True, padx=4)
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
        # 否则 grid 的 expand 会把后面的 pack 控件压成 1px。
        grid.pack(fill="both", expand=True, padx=4, pady=4)
        self.root.after(0, self._fit_video_rows)
        self.root.minsize(900, 700)
        self.root.bind("<space>", self._on_space)
        self.root.bind("<Left>", self._on_left)
        self.root.bind("<Right>", self._on_right)
        self.root.bind("<Home>", lambda _e: self._reset_view())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._refresh_sources()
        self._sync_mono_note()

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

    def _fit_video_rows(self, _event=None) -> None:
        """顶上两格锁 16:9，避免 cover-crop 切掉左右叠加文字。"""
        width = max(100, self._grid.winfo_width() - 8)
        pane_w = max(80, width // 2)
        self._grid.rowconfigure(0, minsize=max(90, int(round(pane_w * 9 / 16))))

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

    def _reset_view(self) -> None:
        lo, hi = self._limits()
        self.ax.set_xlim(lo[0], hi[0])
        self.ax.set_ylim(lo[1], hi[1])
        self.ax.set_zlim(lo[2], hi[2])
        try:
            self.ax.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]))
        except Exception:                                   # noqa: BLE001
            pass
        self.ax.view_init(elev=22, azim=-62)
        self.canvas.draw_idle()

    def _smooth_active(self) -> bool:
        """当前是否真的在用平滑点：勾了**且**至少有一段被平滑处理过。

        两个条件都要：段全太短时 `P_smooth == P_raw`，此时报"已平滑"是撒谎，
        小图要如实写"段太短，未平滑"。
        """
        return bool(self.use_smooth.get()) and bool(self.smooth_valid.any())

    def _redraw(self) -> None:
        self.ax.clear()
        self._reset_view()
        # 勾了「平滑」就用平滑点画主图，否则用原始点。**范围/视角不变**：
        # `_limits` 始终按原始点算（平滑不改变量级，改范围只会让两张图对不上）。
        P = self.P_smooth if self._smooth_active() else self.P_raw
        xs, ys, zs = P[:, 0], P[:, 1], P[:, 2]
        self.ax.plot(xs, ys, zs, color="#93c5fd", linewidth=1.2, alpha=0.8, label="trajectory")
        k = min(self.idx + 1, self.n)
        self.ax.plot(xs[:k], ys[:k], zs[:k], color="#2563eb", linewidth=2.0, label="trail")
        cur = self.points[self.idx]
        self.ax.scatter([cur["xyz"][0]], [cur["xyz"][1]], [cur["xyz"][2]],
                        color="#f97316", s=80, depthshade=False, label="ball")
        self.ax.set_xlabel("X right (m)")
        self.ax.set_ylabel("Y down (m)")
        self.ax.set_zlabel("Z forward (m)")
        self.ax.set_title(f"{'[单目] ' if self.mono else ''}"
                          f"{self.data.get('session', '')} · {self.idx + 1}/{self.n}")
        self.ax.legend(loc="upper right")
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
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = self.cap.read()
        if not ok or img is None:
            self.video_error = f"frame {frame} 解码失败"
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
                 f"坐标={self.data.get('coordinate_system', '?')}")

    # ------------------------------------------------------------------
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
    ap.add_argument("source", help="会话目录或 visualization.json / traj3d.json")
    ap.add_argument("--check", action="store_true", help="只校验数据，不开窗口")
    ap.add_argument("--video", default="", help="同步视频（默认找 derived/annotated.mp4）")
    ap.add_argument("--no-video", action="store_true", help="不加载追踪视频")
    args = ap.parse_args()
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
