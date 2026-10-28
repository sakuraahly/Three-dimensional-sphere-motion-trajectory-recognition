"""离线 3D 轨迹查看器：读取标准 `visualization.json`，交互播放。

用法：
    python apps/view_3d.py data/sessions/20261019-221550
    python apps/view_3d.py data/sessions/20261019-221550/derived/visualization.json
    python apps/view_3d.py data/sessions/20261019-221550 --check

只依赖 matplotlib + Tkinter（环境已有），不依赖网络、不依赖 stereo 包。
坐标系由 `visualization.json` 的 `coordinate_system` 决定：**X 右 / Y 下 / Z 前**
（单位米）；parallel 是校正后左相机系，angled 是原始左相机系。
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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox

ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VIS_SCHEMA = "sphere-stereo-visualization"
TRAJ_SCHEMA = "sphere-stereo-trajectory"


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


def load_view(path: str | Path) -> dict:
    p = resolve_json(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    schema = data.get("schema")
    if schema not in (VIS_SCHEMA, TRAJ_SCHEMA):
        raise ValueError(f"不是 {VIS_SCHEMA}/{TRAJ_SCHEMA}：{schema}")
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


#: 会话结果最多列几个（够选就行；下拉太长反而难找）
SOURCE_LIMIT = 40


def scan_sources(limit: int = SOURCE_LIMIT,
                 roots: tuple[Path, ...] | None = None) -> list[Path]:
    """所有**真有**三维结果的会话，新的在前。

    扫两层：`data/sessions/<会话>/derived/` 与 `data/<会话>/derived/`
    （早期会话直接放在 `data/` 下，那时还没有 `sessions/`）。
    `roots` 只给测试用：判据要能造临时会话，不能依赖这台机器录过什么。

    **排序用目录名里的 `YYYYMMDD-HHMMSS`，不用文件 mtime**：
    跑一次离线会重写 `derived/`，把老会话的 mtime 刷成今天 —— 按 mtime 排会把
    "昨天录、今天补跑"的会话顶到第一条，默认选中就漂了。
    这个坑 `apps/_workbench.scan_sessions` 已经吃过一次，判据写在它的 docstring 里。
    """
    bases = roots if roots is not None else (ROOT / "data" / "sessions", ROOT / "data")
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


def source_label(f: Path) -> str:
    """下拉里显示的名字：会话目录名（`session.json` 有 label 就接在后面）。"""
    sess = f.parent.parent
    try:
        tag = str(json.loads((sess / "session.json").read_text(encoding="utf-8"))
                  .get("label") or "").strip()
    except Exception:                                       # noqa: BLE001
        tag = ""
    return f"{sess.name}  {tag}" if tag else sess.name


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
        self.video_error = ""
        self.video_disabled = (video_path is None or video_path == "") and not auto_video
        self.lo, self.hi = bounds_of(self.points)
        self.video_path = self._resolve_video(video_path, auto_video)
        self.cap = cv2.VideoCapture(str(self.video_path)) if self.video_path else None
        if self.cap is not None and not self.cap.isOpened():
            self.video_error = f"视频打不开：{self.video_path}"
            self.cap.release()
            self.cap = None
        if self.cap is not None:
            vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if vw > 0 and vh > 0 and vw < vh * 1.8:
                self.video_error = f"视频不是左右并排（{vw}x{vh}），按 2:1 切分会错"
                self.cap.release()
                self.cap = None
        self.root.title(f"3D 轨迹 · {data.get('session', '')} · {self.n} points")

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

        self.status = ttk.Label(self.root, padding=(10, 4), anchor="w")
        self.status.pack(side="bottom", fill="x")

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

    # ------------------------------------------------------------------
    # 换数据源
    # ------------------------------------------------------------------
    def _refresh_sources(self) -> None:
        """重扫会话列表并勾住当前源。"""
        self._sources = scan_sources()
        self.source_box["values"] = [source_label(f) for f in self._sources]
        self.source_note.configure(
            text=f"共 {len(self._sources)} 个会话有结果" if self._sources else
                 "没有找到会话结果（data/sessions/*/derived/）")
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
        d = filedialog.askdirectory(
            title="选择会话目录（含 derived/visualization.json）",
            initialdir=str(ROOT / "data" / "sessions"), parent=self.root)
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
        if isinstance(event.widget, (ttk.Button, ttk.Entry, ttk.Combobox, ttk.Scale)):
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

    def _redraw(self) -> None:
        self.ax.clear()
        self._reset_view()
        xs = [p["xyz"][0] for p in self.points]
        ys = [p["xyz"][1] for p in self.points]
        zs = [p["xyz"][2] for p in self.points]
        self.ax.plot(xs, ys, zs, color="#93c5fd", linewidth=1.2, alpha=0.8, label="trajectory")
        k = min(self.idx + 1, self.n)
        self.ax.plot(xs[:k], ys[:k], zs[:k], color="#2563eb", linewidth=2.0, label="trail")
        cur = self.points[self.idx]
        self.ax.scatter([cur["xyz"][0]], [cur["xyz"][1]], [cur["xyz"][2]],
                        color="#f97316", s=80, depthshade=False, label="ball")
        self.ax.set_xlabel("X right (m)")
        self.ax.set_ylabel("Y down (m)")
        self.ax.set_zlabel("Z forward (m)")
        self.ax.set_title(f"{self.data.get('session', '')} · {self.idx + 1}/{self.n}")
        self.ax.legend(loc="upper right")
        self.canvas.draw_idle()
        self._update_video()
        self._status()

    def _render_video(self) -> None:
        for raw, canvas, attr in (
            (self._left_raw, self.video_l, "_left_photo"),
            (self._right_raw, self.video_r, "_right_photo"),
        ):
            canvas.delete("all")
            w = max(2, canvas.winfo_width())
            h = max(2, canvas.winfo_height())
            if w <= 4 or h <= 4:
                w, h = 640, 360
            if raw is None:
                canvas.create_text(
                    w // 2, h // 2, fill="#94a3b8", anchor="center",
                    text=self.video_error or (
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
        self._left_raw = img[:, : w // 2].copy()
        self._right_raw = img[:, w // 2:].copy()
        self._render_video()

    def _status(self) -> None:
        p = self.points[self.idx]
        frame = p.get("frame", self.idx)
        t = p.get("t", 0.0)
        x, y, z = p["xyz"]
        self.status.configure(
            text=f"frame {frame}   t={float(t):.3f}s   "
                 f"X={float(x):.4f}  Y={float(y):.4f}  Z={float(z):.4f} m   "
                 f"模式={self.data.get('stereo_mode', 'parallel')}   "
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
        print(f"文件：{data['_path']}")
        print(f"schema={data.get('schema')} version={data.get('version')}")
        print(f"session={data.get('session')} points={len(data['trajectory'])}")
        print(f"coordinate_system={data.get('coordinate_system')} unit={data.get('unit')}")
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
            if vw < vh * 1.8:
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
