"""GUI 点选双目 seed：左图点球心，右图点球心，自动估半径并写 seeds.json。

为什么单独成文件
----------------
`启动程序.py` 已经 1600+ 行；点选窗口是独立的交互组件，放这里更容易测
（不依赖主界面布局）。它不是命令行程序，只被 GUI import。

约定：
- 点击坐标映射回**原始 1920x1080**，不是缩放后的画布坐标；
- 半径先用点击点附近的 Hough 圆估计，估不到就用宽度/32 的保守默认；
- 保存到 `<session>/derived/seeds.json`，格式见 `stereo/visualization.py`。
"""

from __future__ import annotations

import sys
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
                            "半径自动估、可滚轮/输入框微调。").pack(side="left")
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
        ttk.Label(rf, text="（或滚轮粗调，8–300）").pack(side="left", padx=(8, 0))
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
        """在点击点附近找圆，**只用于估半径**；圆心仍取用户点击位置。

        用户明确要求“我点哪就是哪”，所以这里虽然返回 Hough 圆心，
        调用方只使用第三个返回值（半径），不自动吸附。
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
