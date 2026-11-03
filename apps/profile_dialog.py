"""装置档案管理：列出、切换、新建当前装置（含它配的标定）。

为什么需要它
------------
同一台机器上要维护多套双目装置（近平行装置、90° 装置）。标定数据只有一份
``config/stereo_calibration.json`` 时，换装置重新标定就会**覆盖**另一套的数据
——于是"90 度装置读到了平行装置的标定"。

底层 ``stereo/profiles.py`` 支持"一套装置 = 一份档案 + 一份标定路径"
（``RigProfile.calibration_path``）。这个对话框补上整套入口：列出、切换、
**新建**档案（此前只有列表+切换，但档案列表初始为空，等于功能没闭环）。
"""

from __future__ import annotations

import sys
import tkinter as tk
from tkinter import ttk
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps._gui_widgets import BG, MUTED, ACCENT  # noqa: E402
from stereo import profiles  # noqa: E402


class ProfileDialog(tk.Toplevel):
    """装置档案管理对话框（模态）：列出 / 切换 / 新建。"""

    def __init__(self, app, on_changed=None) -> None:
        super().__init__(app)
        self.app = app
        self.on_changed = on_changed
        self.title("装置档案 · 管理当前装置")
        self.configure(bg=BG)
        self.resizable(False, False)
        self.transient(app)

        body = ttk.Frame(self, padding=14)
        body.pack(fill="both", expand=True)

        ttk.Label(body, text="每套装置配一份标定。切换后，标定/录制/分析都按新装置走。",
                  foreground=MUTED, wraplength=430).pack(anchor="w", pady=(0, 8))

        cols = ("name", "mode", "baseline", "distance", "calib")
        heads = ("装置", "模式", "期望基线", "工作距离", "标定")
        self.tree = ttk.Treeview(body, columns=cols, show="headings", height=8)
        for c, h, w in zip(cols, heads, (140, 70, 70, 70, 210)):
            self.tree.heading(c, text=h)
            self.tree.column(c, width=w, anchor="w", stretch=(c == "calib"))
        self.tree.pack(fill="both", expand=True, pady=(0, 8))
        self.tree.bind("<Double-1>", lambda _e: self._apply())

        self.status = ttk.Label(body, text="", foreground=MUTED, wraplength=430)
        self.status.pack(anchor="w", pady=(0, 8))

        bar = ttk.Frame(body)
        bar.pack(fill="x")
        ttk.Button(bar, text="新建装置…", command=self._new_profile).pack(side="left")
        ttk.Button(bar, text="设为当前", command=self._apply).pack(side="right")
        ttk.Button(bar, text="清除当前", command=self._clear).pack(side="right", padx=(0, 6))

        self._load()

    def _load(self) -> None:
        self.tree.delete(*self.tree.get_children())
        active = profiles.active_name()
        items = profiles.list_profiles()
        for p in items:
            mode = "大角度" if p.stereo_mode == "angled" else "平行"
            base = f"{p.expected_baseline_mm:.0f}" if p.expected_baseline_mm else "—"
            dist = f"{p.working_distance_m:.2f}" if p.working_distance_m else "—"
            calib = p.calibration_path or "（默认 config/stereo_calibration）"
            self.tree.insert("", "end", iid=p.name,
                             values=(p.name, mode, base, dist, calib),
                             tags=("active",) if p.name == active else ())
        self.tree.tag_configure("active", background="#dbeafe")
        if active:
            self.tree.selection_set(active)
            self.status.configure(text=f"当前装置：{active}")
        else:
            self.status.configure(text="还没有装置档案 —— 点「新建装置…」建一条"
                                       "（如「平行装置」「90度装置」），各配一份标定")

    def _selected(self) -> str | None:
        sel = self.tree.selection()
        return sel[0] if sel else None

    def _apply(self) -> None:
        name = self._selected()
        if not name:
            self.status.configure(text="先在列表里选一个装置", foreground="#b45309")
            return
        try:
            profiles.set_active(name)
        except profiles.ProfileError as exc:
            self.status.configure(text=str(exc), foreground="#b45309")
            return
        self.status.configure(text=f"已切到装置：{name}", foreground=MUTED)
        self._load()
        if self.on_changed:
            self.on_changed()

    def _clear(self) -> None:
        profiles.set_active(None)
        self.status.configure(text="已清除当前装置（回落默认标定路径）", foreground=MUTED)
        self._load()
        if self.on_changed:
            self.on_changed()

    def _new_profile(self) -> None:
        """弹一个新建档案的小表单：名称 + 模式 + 期望基线 + 工作距离。

        标定路径**自动**按装置名生成（`config/calib/<name>`），不让用户手填
        绝对路径（`_check_calibration_path` 拒绝绝对路径与 `..`）。
        """
        dlg = tk.Toplevel(self)
        dlg.title("新建装置档案")
        dlg.configure(bg=BG)
        dlg.resizable(False, False)
        dlg.transient(self)
        body = ttk.Frame(dlg, padding=14)
        body.pack(fill="both", expand=True)

        name_var = tk.StringVar()
        mode_var = tk.StringVar(value="parallel")
        base_var = tk.StringVar()
        dist_var = tk.StringVar()

        ttk.Label(body, text="名称（如「平行装置」「90度装置」）").pack(anchor="w")
        ttk.Entry(body, textvariable=name_var, width=32).pack(fill="x", pady=(2, 8))
        ttk.Label(body, text="模式").pack(anchor="w")
        ttk.Combobox(body, textvariable=mode_var, state="readonly", width=20,
                     values=("parallel（平行）", "angled（大角度）")).pack(anchor="w", pady=(2, 8))
        ttk.Label(body, text="期望基线 mm（可留空，参考值）").pack(anchor="w")
        ttk.Entry(body, textvariable=base_var, width=32).pack(fill="x", pady=(2, 8))
        ttk.Label(body, text="工作距离 m（可留空，参考值）").pack(anchor="w")
        ttk.Entry(body, textvariable=dist_var, width=32).pack(fill="x", pady=(2, 8))
        ttk.Label(body, text="标定路径自动设为 config/calib/<名称>（标定时自动写入）",
                  foreground=MUTED, wraplength=400).pack(anchor="w", pady=(0, 8))

        err = ttk.Label(body, text="", foreground="#b45309", wraplength=400)
        err.pack(anchor="w", pady=(0, 8))

        def save() -> None:
            name = name_var.get().strip()
            mode_label = mode_var.get()
            mode = "angled" if "angled" in mode_label else "parallel"
            if not name:
                err.configure(text="名称不能为空")
                return
            try:
                base = float(base_var.get()) if base_var.get().strip() else None
                dist = float(dist_var.get()) if dist_var.get().strip() else None
            except ValueError:
                err.configure(text="基线/工作距离要填数字（或留空）")
                return
            # 标定路径：仓库内相对路径，按装置名生成
            calib_path = f"config/calib/{name}"
            prof = profiles.RigProfile(
                name=name, stereo_mode=mode,
                expected_baseline_mm=base, working_distance_m=dist,
                calibration_path=calib_path)
            try:
                profiles.upsert(prof)
            except profiles.ProfileError as exc:
                err.configure(text=str(exc))
                return
            dlg.destroy()
            self._load()
            self.status.configure(text=f"已新建装置：{name}（标定路径 {calib_path}）")

        bar = ttk.Frame(body)
        bar.pack(fill="x")
        ttk.Button(bar, text="确定", command=save).pack(side="right")
        ttk.Button(bar, text="取消", command=dlg.destroy).pack(side="right", padx=(0, 6))

        dlg.grab_set()
