"""GUI 点选 seed：双目的``SeedPicker``（左右各点球心）与单目的``MonoSeedPicker``。

为什么单独成文件
----------------
`启动程序.py` 已经 1600+ 行；点选窗口是独立的交互组件，放这里更容易测
（不依赖主界面布局）。它不是命令行程序，只被 GUI import。

双目约定（``SeedPicker``）：
- 点击坐标映射回**原始 1920x1080**，不是缩放后的画布坐标；
- 半径先用点击点附近的 Hough 圆估计，估不到就用宽度/32 的保守默认；
- 保存到 `<session>/derived/seeds.json`，格式见 `stereo/visualization.py`。

单目约定（``MonoSeedPicker``，2026-10-30 增）：
- 同样把点击坐标映射回原图（纯函数 ``canvas_to_image``）；
- **半径不靠 Hough**：第一点球心、第二点球边，两点距离就是半径。
  实测这颗镀铬球 Hough 给 68~69 px，人工两点给 91 px，而
  ``Z = f·D/(2r)`` 里 r 错 24% 就是深度错 24%，且**每帧同号**（系统项，
  平均不掉）。所以半径必须由人点出来；
- 保存到 `<session>/derived/seed.json`，键为 x / y / radius / frame /
  image_size / created，供 `apps/run_video.py --seed x,y,r` 使用。
"""

from __future__ import annotations

import json
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stereo.visualization import write_seeds  # noqa: E402

PREVIEW_W, PREVIEW_H = 760, 428

#: 单目预览固定 760 宽（与双目同宽，用户不用适应两套尺度），高度按原图比例算。
MONO_PREVIEW_W = 760


# --------------------------------------------------------------------------- #
# 纯函数（无 Tk）：坐标映射 / 两点定圆 / 深度换算
# --------------------------------------------------------------------------- #
# 为什么这三个要抽成模块级函数：它们是"点在哪 → 写进 seed.json → 进命令行"这条
# 链上唯一会**静默算错**的地方（缩放比写反、半径用了直径、mm/m 搞混）。
# 它们不碰控件，所以没有显示器也能断言 —— 判据在 tests/selftest_gui_budget.py。
def canvas_to_image(x: float, y: float, scale: float,
                    offset: tuple[float, float] = (0.0, 0.0)
                    ) -> tuple[float, float]:
    """画布坐标 → 原图坐标：``(画布 - 偏移) / 缩放比``。

    ``scale`` 是"画布 1 px 对应原图多少 px"（缩放到 760 宽时 < 1）。
    先减偏移再除缩放：反过来会把偏移也缩放，平移一叠加就整片错位。
    """
    if scale <= 0:
        raise ValueError(f"scale 必须为正数，收到 {scale!r}")
    ox, oy = float(offset[0]), float(offset[1])
    return ((float(x) - ox) / scale, (float(y) - oy) / scale)


def radius_from_points(p1: tuple[float, float],
                       p2: tuple[float, float]) -> float:
    """两点定圆：第一点是圆心，两点距离就是半径。

    为什么用"球边那一点"而不是"Hough 估半径"：见模块 docstring —— 这颗镜面球
    Hough 半径系统性偏小 24%，而 Z 反比于 r，偏小就直接把深度顶大。
    """
    return float(np.hypot(float(p2[0]) - float(p1[0]),
                          float(p2[1]) - float(p1[1])))


def z_estimate_m(fx: float | None, diameter_mm: float | None,
                 radius_px: float | None) -> float | None:
    """``Z = f·D/(2r)``，返回米；任一项缺失/非正 → ``None``（不猜）。

    单位是这条链上最容易错的地方：D 在界面上是 mm、Z 要 m，所以先 /1000。
    直接照抄公式不换单位，深度会大 1000 倍 —— 而"1000 m"看起来像没测到。
    """
    try:
        f, d, r = float(fx or 0), float(diameter_mm or 0), float(radius_px or 0)
    except (TypeError, ValueError):
        return None
    if f <= 0 or d <= 0 or r <= 0:
        return None
    return f * (d / 1000.0) / (2.0 * r)


class SeedPicker(tk.Toplevel):
    """模态窗口：返回 `self.result` = seeds.json 路径，或 None。"""

    def __init__(self, parent, session_dir: str | Path, seed_frame: int = 0) -> None:
        super().__init__(parent)
        self.title("点选双目球心")
        self.transient(parent)
        self.grab_set()
        self.session_dir = Path(session_dir)
        self.frame_index = int(seed_frame)
        self.left_click: tuple[float, float] | None = None
        self.right_click: tuple[float, float] | None = None
        self.left_radius = 0.0
        self.right_radius = 0.0
        self.left_frame: np.ndarray | None = None
        self.right_frame: np.ndarray | None = None
        self._left_photo = None
        self._right_photo = None
        self.result: Path | None = None
        self._build()
        self._load()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")
        ttk.Label(top, text="在左图点球心，再在右图点球心；圆心=你点击的位置，"
                            "需要时再点『吸附Hough圆心』；半径可微调。").pack(side="left")
        ttk.Label(top, text="seed 帧").pack(side="left", padx=(18, 4))
        self.frame_var = tk.StringVar(value=str(self.frame_index))
        ttk.Spinbox(top, from_=0, to=99999, width=6,
                    textvariable=self.frame_var).pack(side="left")
        ttk.Button(top, text="载入该帧", command=self._reload_frame).pack(side="left", padx=4)

        body = ttk.Frame(self, padding=10)
        body.pack(fill="both", expand=True)
        self.left_canvas = tk.Canvas(body, width=PREVIEW_W, height=PREVIEW_H,
                                     bg="#111827", highlightthickness=0)
        self.right_canvas = tk.Canvas(body, width=PREVIEW_W, height=PREVIEW_H,
                                      bg="#111827", highlightthickness=0)
        self.left_canvas.grid(row=0, column=0, padx=(0, 8))
        self.right_canvas.grid(row=0, column=1)
        ttk.Label(body, text="左路").grid(row=1, column=0)
        ttk.Label(body, text="右路").grid(row=1, column=1)
        self.left_canvas.bind("<Button-1>", lambda e: self._on_click("left", e))
        self.right_canvas.bind("<Button-1>", lambda e: self._on_click("right", e))
        self.left_canvas.bind("<MouseWheel>", lambda e: self._on_wheel("left", e))
        self.right_canvas.bind("<MouseWheel>", lambda e: self._on_wheel("right", e))

        # 滚轮只适合粗调；直接输入半径才能精确到像素。
        rf = ttk.Frame(self, padding=(10, 0, 10, 4))
        rf.pack(fill="x")
        ttk.Label(rf, text="半径(px)：左").pack(side="left")
        self.left_r_var = tk.StringVar(value="0")
        self.right_r_var = tk.StringVar(value="0")
        lspin = ttk.Spinbox(rf, from_=8, to=300, increment=1, width=6,
                            textvariable=self.left_r_var)
        lspin.pack(side="left", padx=(2, 13))
        ttk.Label(rf, text="右").pack(side="left")
        rspin = ttk.Spinbox(rf, from_=8, to=300, increment=1, width=6,
                            textvariable=self.right_r_var)
        rspin.pack(side="left", padx=(2, 6))
        ttk.Button(rf, text="应用半径", command=self._apply_radius_vars).pack(side="left")
        ttk.Button(rf, text="吸附Hough圆心", command=self._snap_to_hough).pack(
            side="left", padx=(6, 0))
        ttk.Label(rf, text="（圆心默认=点击位置；滚轮 8–300）").pack(side="left", padx=(8, 0))
        for spin in (lspin, rspin):
            spin.bind("<Return>", lambda _e: self._apply_radius_vars())
            spin.bind("<FocusOut>", lambda _e: self._apply_radius_vars())

        bottom = ttk.Frame(self, padding=10)
        bottom.pack(fill="x")
        self.status = ttk.Label(bottom, text="等待点击")
        self.status.pack(side="left")
        ttk.Button(bottom, text="重新点", command=self._reset).pack(side="right", padx=4)
        ttk.Button(bottom, text="取消", command=self.destroy).pack(side="right", padx=4)
        ttk.Button(bottom, text="保存 seed", command=self._save).pack(side="right", padx=4)

    # ------------------------------------------------------------------
    def _read_frame(self, side: str) -> np.ndarray | None:
        path = self.session_dir / f"{side}.mkv"
        if not path.exists():
            path = self.session_dir / f"{side}.mp4"
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                return None
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index)
            ok, frame = cap.read()
            return frame if ok else None
        finally:
            cap.release()

    def _load(self) -> None:
        self.left_frame = self._read_frame("left")
        self.right_frame = self._read_frame("right")
        if self.left_frame is None or self.right_frame is None:
            messagebox.showerror("点选 seed", "读不到左右视频帧；检查会话目录。", parent=self)
            self.destroy()
            return
        self._left_photo = ImageTk.PhotoImage(
            Image.fromarray(cv2.cvtColor(cv2.resize(
                self.left_frame, (PREVIEW_W, PREVIEW_H), interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2RGB)))
        self._right_photo = ImageTk.PhotoImage(
            Image.fromarray(cv2.cvtColor(cv2.resize(
                self.right_frame, (PREVIEW_W, PREVIEW_H), interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2RGB)))
        self._redraw()

    def _reload_frame(self) -> None:
        try:
            self.frame_index = max(0, int(self.frame_var.get()))
        except ValueError:
            messagebox.showwarning("seed 帧", "seed 帧要填整数。", parent=self)
            return
        self._reset()
        self._load()

    # ------------------------------------------------------------------
    def _estimate_circle(self, frame: np.ndarray, x: float, y: float) -> tuple[float, float, float] | None:
        """在点击点附近找圆，返回 (圆心x, 圆心y, 半径)。

        默认点击只取半径，圆心保持用户点击位置；『吸附Hough圆心』按钮
        才会显式采用返回的圆心。
        """
        h, w = frame.shape[:2]
        r = 150
        x0, y0 = max(0, int(x - r)), max(0, int(y - r))
        x1, y1 = min(w, int(x + r)), min(h, int(y + r))
        roi = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        roi = cv2.GaussianBlur(roi, (7, 7), 1.5)
        circles = cv2.HoughCircles(
            roi, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
            param1=120, param2=22, minRadius=15, maxRadius=105)
        if circles is not None:
            cand = []
            for cx, cy, cr in circles[0]:
                gx, gy = float(x0 + cx), float(y0 + cy)
                d = float(np.hypot(gx - x, gy - y))
                if d <= 60.0 and 15.0 <= cr <= 105.0:
                    cand.append((float(cr), gx, gy))
            if cand:
                cr, gx, gy = max(cand, key=lambda t: t[0])
                return gx, gy, float(max(8.0, min(110.0, cr)))
        return None

    def _on_wheel(self, side: str, event) -> None:
        """滚轮粗调半径；精确值用上面的输入框。"""
        delta = 1.02 if event.delta > 0 else 0.98
        if side == "left" and self.left_click:
            self.left_radius = float(max(8, min(300, self.left_radius * delta)))
            self.left_r_var.set(f"{self.left_radius:.0f}")
            self._redraw()
        elif side == "right" and self.right_click:
            self.right_radius = float(max(8, min(300, self.right_radius * delta)))
            self.right_r_var.set(f"{self.right_radius:.0f}")
            self._redraw()

    def _apply_radius_vars(self) -> None:
        """把输入框里的精确半径写回两侧圆。"""
        try:
            lr = float(self.left_r_var.get())
            rr = float(self.right_r_var.get())
        except ValueError:
            self.status.configure(text="半径要填数字，例如 60")
            return
        self.left_radius = float(max(8, min(300, lr)))
        self.right_radius = float(max(8, min(300, rr)))
        self.left_r_var.set(f"{self.left_radius:.0f}")
        self.right_r_var.set(f"{self.right_radius:.0f}")
        self._redraw()

    def _snap_to_hough(self) -> None:
        """显式按钮：把当前左右圆吸附到附近 Hough 圆心。

        默认不自动吸附；只有用户点这个按钮才改圆心，保留“我点哪就是哪”的
        默认语义，同时不丢掉 Hough 辅助。
        """
        changed = 0
        for side in ("left", "right"):
            click = self.left_click if side == "left" else self.right_click
            frame = self.left_frame if side == "left" else self.right_frame
            if click is None or frame is None:
                continue
            circle = self._estimate_circle(frame, click[0], click[1])
            if circle is None:
                continue
            cx, cy, radius = circle
            if side == "left":
                self.left_click, self.left_radius = (cx, cy), radius
                self.left_r_var.set(f"{radius:.0f}")
            else:
                self.right_click, self.right_radius = (cx, cy), radius
                self.right_r_var.set(f"{radius:.0f}")
            changed += 1
        self._redraw()
        if changed:
            self.status.configure(text=f"已吸附 {changed} 侧到 Hough 圆心；可再手动点回")
        else:
            self.status.configure(text="附近没找到可用的 Hough 圆；保持点击位置")

    def _on_click(self, side: str, event) -> None:
        frame = self.left_frame if side == "left" else self.right_frame
        if frame is None:
            return
        h, w = frame.shape[:2]
        x = float(event.x) * w / PREVIEW_W
        y = float(event.y) * h / PREVIEW_H
        circle = self._estimate_circle(frame, x, y)
        if circle is not None:
            # 圆心必须严格等于用户点击位置，不自动吸附 Hough 圆心；
            # Hough 只用来估半径（用户明确要求“我点哪就是哪”）。
            radius = float(circle[2])
        else:
            radius = float(max(12, min(300, round(w / 32.0))))
        if side == "left":
            self.left_click, self.left_radius = (x, y), radius
            self.left_r_var.set(f"{radius:.0f}")
        else:
            self.right_click, self.right_radius = (x, y), radius
            self.right_r_var.set(f"{radius:.0f}")
        self._redraw()

    def _redraw(self) -> None:
        for canvas, photo, click, radius, side in (
            (self.left_canvas, self._left_photo, self.left_click, self.left_radius, "左"),
            (self.right_canvas, self._right_photo, self.right_click, self.right_radius, "右"),
        ):
            canvas.delete("all")
            if photo is not None:
                canvas.create_image(0, 0, anchor="nw", image=photo)
            if click is not None:
                sx = click[0] * PREVIEW_W / (self.left_frame.shape[1]
                                              if side == "左" else self.right_frame.shape[1])
                sy = click[1] * PREVIEW_H / (self.left_frame.shape[0]
                                              if side == "左" else self.right_frame.shape[0])
                sr = radius * PREVIEW_W / (self.left_frame.shape[1]
                                           if side == "左" else self.right_frame.shape[1])
                canvas.create_oval(sx - sr, sy - sr, sx + sr, sy + sr,
                                   outline="#ef4444", width=3)
                canvas.create_line(sx - 12, sy, sx + 12, sy, fill="#22c55e", width=2)
                canvas.create_line(sx, sy - 12, sx, sy + 12, fill="#22c55e", width=2)
                canvas.create_text(sx + 8, sy - 8, text=f"{side} ({click[0]:.0f},{click[1]:.0f}) r={radius:.0f}",
                                   fill="#22c55e", anchor="sw", font=("Consolas", 9))
        if self.left_click and self.right_click:
            warn = ""
            if max(self.left_radius, self.right_radius) > 90:
                warn = "  ⚠ 半径偏大：向下滚轮或直接输入更小值"
            self.status.configure(
                text=f"左 ({self.left_click[0]:.0f},{self.left_click[1]:.0f}) "
                     f"r={self.left_radius:.0f}  |  右 "
                     f"({self.right_click[0]:.0f},{self.right_click[1]:.0f}) "
                     f"r={self.right_radius:.0f}  → 可以保存{warn}")
        elif self.left_click:
            self.status.configure(text="左已点；现在点右图球心")
        else:
            self.status.configure(text="等待点击左图球心")

    def _reset(self) -> None:
        self.left_click = self.right_click = None
        self.left_radius = self.right_radius = 0.0
        self.left_r_var.set("0")
        self.right_r_var.set("0")
        self._redraw()

    def _save(self) -> None:
        if not self.left_click or not self.right_click:
            messagebox.showwarning("点选 seed", "左右两边都点完才能保存。", parent=self)
            return
        if self.left_frame is None:
            return
        try:
            path = write_seeds(
                self.session_dir, (self.left_click[0], self.left_click[1], self.left_radius),
                (self.right_click[0], self.right_click[1], self.right_radius),
                seed_frame=self.frame_index,
                image_size=(self.left_frame.shape[1], self.left_frame.shape[0]),
                source="gui_click",
                note="GUI 点选球心；圆心=点击位置，半径由 Hough 估计、可微调",
            )
        except Exception as exc:                            # noqa: BLE001
            messagebox.showerror("保存 seed", f"保存失败：{exc}", parent=self)
            return
        self.result = path
        messagebox.showinfo("保存 seed", f"已保存：\n{path}", parent=self)
        self.destroy()


class MonoSeedPicker(tk.Toplevel):
    """模态窗口：单目会话里点一下球心、再点一下球边，写 ``seed.json``。

    返回 ``self.result`` = seed.json 路径，或 None（取消）。

    为什么要人点两下（第一性原理，不是"图省事"）
    --------------------------------------------
    这颗球是镀铬镜面球，背景是宿舍（书架 / 塑料袋 / 挂着的白衣服）。
    288 帧实测：``color`` 0/288 落在球上，``hough`` 96% 有输出但只有 1.4%
    落在球上、半径中位 69.5 px；真半径约 91 px。也就是说**没有任何可靠的
    自动初始化**（`sphere_tracker/detectors/specular.py` 的 docstring 早写过
    "不假装全自动"）。而半径是 Z 的分母：69.5 对 91 会让整条轨迹近 30% 偏近，
    且是系统项，平均不掉。

    所以这里做两件事与双目版**刻意不同**：
    1. 半径来自第二点（用户点的球边），**不调 Hough**；
    2. 窗口里只显示一路图（单目只有一路，硬套双目会多出一个假的"右路"）。
    """

    def __init__(self, parent, session_dir: str | Path, *,
                 fx: float | None = None, diameter_mm: float | None = None,
                 frame_index: int = 0) -> None:
        super().__init__(parent)
        self.title("点选单目球心")
        self.transient(parent)
        self.grab_set()
        self.session_dir = Path(session_dir)
        self.frame_index = int(frame_index)
        # fx / 球径由调用方传：界面第②步那一栏就是唯一真相，
        # 在这里再读一遍配置等于造第二份口径（两边不一致时没人看得出）。
        self.fx = float(fx) if fx else None
        self.diameter_mm = float(diameter_mm) if diameter_mm else None
        self.center: tuple[float, float] | None = None      # 原图坐标
        self.edge: tuple[float, float] | None = None
        self.radius = 0.0                                   # 原图像素
        self.frame: np.ndarray | None = None
        self.scale = 1.0
        self._photo = None
        self.result: Path | None = None
        self._build()
        self._load()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")
        ttk.Label(top, text="先点球心，再点球的边缘（两点距离就是半径）。"
                            "再点一次会从头开始。半径不用估：你点哪就是哪。").pack(anchor="w")

        body = ttk.Frame(self, padding=(10, 0))
        body.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(body, width=MONO_PREVIEW_W, height=428,
                                bg="#111827", highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._on_click)

        # 三行读数：球心 / 半径 / Z。分成三行而不是一句话，
        # 是因为用户核对的是三个**独立的数**，挤一行就得自己拆。
        info = ttk.Frame(self, padding=(10, 8, 10, 0))
        info.pack(fill="x")
        self.lbl_center = ttk.Label(info, text="球心：还没点")
        self.lbl_center.pack(anchor="w")
        self.lbl_radius = ttk.Label(info, text="半径：还没点球边")
        self.lbl_radius.pack(anchor="w")
        self.lbl_z = ttk.Label(info, text="约 Z = —（点完球边才有）")
        self.lbl_z.pack(anchor="w")
        # 为什么 Z 前面写"约"：D 用的是**有效**直径（21.6 mm，反推值），
        # 不是卡尺量的名义直径 25 mm。少这个字，用户会把 0.16 m 当测量结果。
        ttk.Label(info, style="Hint.TLabel",
                  text="Z 是按球径先验反推的（不是视差测量），球径填错多少，"
                       "深度就错多少；这一路只适合看趋势。").pack(anchor="w")

        bottom = ttk.Frame(self, padding=10)
        bottom.pack(fill="x")
        self.status = ttk.Label(bottom, text="等待点击球心")
        self.status.pack(side="left")
        ttk.Button(bottom, text="保存 seed", command=self._save).pack(side="right", padx=4)
        ttk.Button(bottom, text="重新点", command=self._reset).pack(side="right", padx=4)
        ttk.Button(bottom, text="取消", command=self.destroy).pack(side="right", padx=4)

    # ------------------------------------------------------------------
    def _read_frame(self) -> np.ndarray | None:
        """读会话里那段原始视频的首帧（``clip.*``）。"""
        from apps._mono_sessions import find_clip

        clip = find_clip(self.session_dir)
        if clip is None:
            return None
        cap = cv2.VideoCapture(str(clip))
        try:
            if not cap.isOpened():
                return None
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index)
            ok, frame = cap.read()
            return frame if ok else None
        finally:
            cap.release()

    def _load(self) -> None:
        self.frame = self._read_frame()
        if self.frame is None:
            messagebox.showerror("点选球心",
                                 "读不到这段会话的视频（clip.mkv）；"
                                 "先录一段或从「历史会话…」挑一段。", parent=self)
            self.destroy()
            return
        h, w = self.frame.shape[:2]
        self.scale = MONO_PREVIEW_W / float(w)
        ch = max(1, int(round(h * self.scale)))
        # 画布尺寸必须跟着缩放比走：写死 428 时，非 16:9 的录像会被压扁，
        # 用户点到的位置和原图就对不上了（而读数看起来完全正常）。
        self.canvas.configure(width=MONO_PREVIEW_W, height=ch)
        small = cv2.resize(self.frame, (MONO_PREVIEW_W, ch),
                           interpolation=cv2.INTER_AREA)
        self._photo = ImageTk.PhotoImage(
            Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        self._redraw()

    # ------------------------------------------------------------------
    def _on_click(self, event) -> None:
        if self.frame is None:
            return
        point = canvas_to_image(event.x, event.y, self.scale)
        if self.center is None:
            self.center, self.edge, self.radius = point, None, 0.0
            self.status.configure(text="球心已定；现在点球的边缘")
        elif self.edge is None:
            self.edge = point
            self.radius = radius_from_points(self.center, point)
            if self.radius < 1.0:
                # 同一点点两次 = 半径 0：不是"很小"，是没有给出信息。
                self.edge, self.radius = None, 0.0
                self.status.configure(text="两次点在同一处（半径 0）；请点到球边")
            else:
                self.status.configure(text="半径已定；可以保存，或再点一次重来")
        else:
            # 第三次点击 = 重新开始（圆心换成这一点，半径清空）。
            # 不要求先按"重新点"：点错了最自然的动作就是直接在正确位置再点一下。
            self.center, self.edge, self.radius = point, None, 0.0
            self.status.configure(text="已重新开始：球心定在新位置，请再点球边")
        self._redraw()

    def _reset(self) -> None:
        self.center = self.edge = None
        self.radius = 0.0
        self.status.configure(text="等待点击球心")
        self._redraw()

    def _redraw(self) -> None:
        self.canvas.delete("all")
        if self._photo is not None:
            self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        self.lbl_center.configure(
            text=(f"球心：({self.center[0]:.0f}, {self.center[1]:.0f})"
                  if self.center else "球心：还没点"))
        self.lbl_radius.configure(
            text=(f"半径：{self.radius:.1f} px" if self.radius > 0
                  else "半径：还没点球边"))
        z = z_estimate_m(self.fx, self.diameter_mm, self.radius)
        if z is None:
            why = ("还没点球边" if self.radius <= 0
                   else "缺焦距或球径，先在第②步填好")
            self.lbl_z.configure(text=f"约 Z = —（{why}）")
        else:
            self.lbl_z.configure(
                text=f"约 Z = {z:.3f} m"
                     f"（按 D={self.diameter_mm:g} mm、fx={self.fx:.0f} px 换算）")
        if self.center is not None:
            cx, cy = self.center[0] * self.scale, self.center[1] * self.scale
            self.canvas.create_line(cx - 12, cy, cx + 12, cy, fill="#22c55e", width=2)
            self.canvas.create_line(cx, cy - 12, cx, cy + 12, fill="#22c55e", width=2)
            if self.radius > 0:
                sr = self.radius * self.scale
                self.canvas.create_oval(cx - sr, cy - sr, cx + sr, cy + sr,
                                        outline="#22c55e", width=2)
            if self.edge is not None:
                ex, ey = self.edge[0] * self.scale, self.edge[1] * self.scale
                self.canvas.create_oval(ex - 3, ey - 3, ex + 3, ey + 3,
                                        outline="#f59e0b", width=2)

    # ------------------------------------------------------------------
    def _save(self) -> None:
        if self.center is None or self.radius <= 0:
            messagebox.showwarning("保存 seed", "先点球心，再点球边。", parent=self)
            return
        if self.frame is None:
            return
        h, w = self.frame.shape[:2]
        payload = {
            "x": round(float(self.center[0]), 2),
            "y": round(float(self.center[1]), 2),
            "radius": round(float(self.radius), 2),
            "frame": self.frame_index,
            "image_size": [int(w), int(h)],
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        path = self.session_dir / "derived" / "seed.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("保存 seed", f"写不进 {path}：{exc}", parent=self)
            return
        # 印路径而不是只说"已保存"：这个文件就是 --seed 的来源，
        # 用户要能自己去核对/删掉它（删掉就退回自动检测）。
        print(f"seed 已保存：{path}")
        self.result = path
        messagebox.showinfo("保存 seed", f"已保存：\n{path}", parent=self)
        self.destroy()

