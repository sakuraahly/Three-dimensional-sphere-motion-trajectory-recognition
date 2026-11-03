"""装置档案管理：列出、切换当前装置（含它配的标定）。

为什么需要它
------------
同一台机器上要维护多套双目装置（近平行装置、90° 装置）。标定数据只有一份
``config/stereo_calibration.json`` 时，换装置重新标定就会**覆盖**另一套的数据
——于是"90 度装置读到了平行装置的标定"。

底层 ``stereo/profiles.py`` 早就支持"一套装置 = 一份档案 + 一份标定路径"
（``RigProfile.calibration_path``），但 GUI 一直没有切换入口。这个对话框补上：
列出所有档案、高亮当前 active、点一下切换，切换后门厅三张卡会按新装置重判就绪。

只做「列表 + 切换」，不在这里建/改档案（新建走 ``apps/tools/rig_profile.py``，
那一步人很少做，且要校验的字段多，塞进对话框反而让"切换"这个高频动作变重）。
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
    """装置档案切换对话框（模态）。选一个档案点「设为当前」，或「清除当前」。"""

    def __init__(self, app, on_changed=None) -> None:
        super().__init__(app)
        self.app = app
        self.on_changed = on_changed
        self.title("装置档案 · 切换当前装置")
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
        for c, h, w in zip(cols, heads, (140, 80, 80, 80, 200)):
            self.tree.heading(c, text=h)
            self.tree.column(c, width=w, anchor="w", stretch=(c == "calib"))
        self.tree.pack(fill="both", expand=True, pady=(0, 8))
        self.tree.bind("<Double-1>", lambda _e: self._apply())

        self.status = ttk.Label(body, text="", foreground=MUTED, wraplength=430)
        self.status.pack(anchor="w", pady=(0, 8))

        bar = ttk.Frame(body)
        bar.pack(fill="x")
        ttk.Button(bar, text="设为当前", command=self._apply).pack(side="right")
        ttk.Button(bar, text="清除当前", command=self._clear).pack(side="right", padx=(0, 6))
        ttk.Button(bar, text="刷新", command=self._load).pack(side="left")

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
            tag = "active"
            self.tree.insert("", "end", iid=p.name,
                             values=(p.name, mode, base, dist, calib),
                             tags=(tag,) if p.name == active else ())
        self.tree.tag_configure("active", background="#dbeafe")
        if active:
            self.tree.selection_set(active)
            self.status.configure(text=f"当前装置：{active}")
        else:
            self.status.configure(text="未设置当前装置（用默认标定路径）")

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


def main() -> int:
    """独立运行（不经过门厅）时也能打开，方便排查档案问题。"""
    from apps._gui_app import App
    app = App()
    app.withdraw()
    dlg = ProfileDialog(app)
    app.wait_window(dlg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
