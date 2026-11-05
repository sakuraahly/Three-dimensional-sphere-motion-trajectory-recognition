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


#: 预置的两条默认装置档案（用户第一次打开时自动建，避免空列表"没用"）。
DEFAULT_PROFILES = (
    ("平行装置", "parallel", 124.0, 0.3),
    ("90度装置", "angled", None, 0.3),
)


def ensure_default_profiles() -> int:
    """若档案列表为空，预置「平行装置」「90度装置」两条；返回新建条数。

    只建**不存在的**名字（幂等）：用户已经自己建过的同名档案不会被覆盖。
    标定路径按装置名生成 `config/calib/<名称>`。
    """
    created = 0
    existing = {p.name for p in profiles.list_profiles()}
    for name, mode, base, dist in DEFAULT_PROFILES:
        if name in existing:
            continue
        profiles.upsert(profiles.RigProfile(
            name=name, stereo_mode=mode,
            expected_baseline_mm=base, working_distance_m=dist,
            calibration_path=f"config/calib/{name}"))
        created += 1
    return created


class ProfileDialog(tk.Toplevel):
    """装置档案管理对话框（模态）：列出 / 切换 / 新建。"""

    def __init__(self, app, on_changed=None) -> None:
        super().__init__(app)
        self.on_changed = on_changed
        self.title("装置档案 · 管理当前装置")
        self.configure(bg=BG)
        self.resizable(False, False)
        self.transient(app)
        self.grab_set()

        body = ttk.Frame(self, padding=14)
        body.pack(fill="both", expand=True)

        ttk.Label(body, text="每套装置配一份标定。切换后，标定/录制/分析都按新装置走。",
                  foreground=MUTED, wraplength=430).pack(anchor="w", pady=(0, 8))

        cols = ("name", "mode", "baseline", "distance", "capture", "calib")
        heads = ("装置", "模式", "期望基线", "工作距离", "采集档", "标定")
        self.tree = ttk.Treeview(body, columns=cols, show="headings", height=8)
        for c, h, w in zip(cols, heads, (128, 62, 66, 66, 104, 190)):
            self.tree.heading(c, text=h)
            self.tree.column(c, width=w, anchor="w", stretch=(c == "calib"))
        self.tree.pack(fill="both", expand=True, pady=(0, 8))
        self.tree.bind("<Double-1>", lambda _e: self._apply())

        self.status = ttk.Label(body, text="", foreground=MUTED, wraplength=430)
        self.status.pack(anchor="w", pady=(0, 8))

        bar = ttk.Frame(body)
        bar.pack(fill="x")
        ttk.Button(bar, text="新建装置…", command=self._new_profile).pack(side="left")
        ttk.Button(bar, text="编辑…", command=self._edit_profile).pack(
            side="left", padx=(6, 0))
        ttk.Button(bar, text="设为当前", command=self._apply).pack(side="right")
        ttk.Button(bar, text="清除当前", command=self._clear).pack(side="right", padx=(0, 6))

        self._load()

    def _load(self) -> None:
        # 首次打开时预置默认档案（平行/90度各一条），否则空列表看起来"没用"。
        if ensure_default_profiles():
            self.status.configure(text="已预置默认档案：平行装置、90度装置")
        self.tree.delete(*self.tree.get_children())
        active = profiles.active_name()
        items = profiles.list_profiles()
        for p in items:
            mode = "大角度" if p.stereo_mode == "angled" else "平行"
            base = f"{p.expected_baseline_mm:.0f}" if p.expected_baseline_mm else "—"
            dist = f"{p.working_distance_m:.2f}" if p.working_distance_m else "—"
            calib = p.calibration_path or "（默认 config/stereo_calibration）"
            if p.width and p.height:
                cap = f"{p.width}x{p.height}" + (f"@{p.fps}" if p.fps else "")
            else:
                cap = "—（按标定）"
            self.tree.insert("", "end", iid=p.name,
                             values=(p.name, mode, base, dist, cap, calib),
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
        """新建档案。"""
        self._profile_form()

    def _edit_profile(self) -> None:
        """编辑选中的档案（**不能改名**）。

        名称是档案的 key，也是标定路径 ``config/calib/<名称>`` 的一部分 ——
        允许改名会让这台装置**已有的标定突然找不到**（路径变了），而档案看起来
        还在。要换名字就新建一条，标定重新做。
        """
        name = self._selected()
        if not name:
            self.status.configure(text="先在列表里选一个装置", foreground="#b45309")
            return
        self._profile_form(name)

    def _profile_form(self, name: str | None = None) -> None:
        """新建 / 编辑表单（``name`` 为空 = 新建）。

        **采集档**（宽×高@fps）在这里填。它是拍标定图与录制**共用的那一档**：
        `capture_board.py` 默认 2560x1440，而本机实测双路同时开 2560 会让一路的
        MSMF 会话失效（2026-11-05；1920x1080 两台 15/15 稳定）。不填这一档，
        拍图会照默认走 2560，标定的 capture_size 记成 2560，之后录制必然失败。
        """
        editing = name is not None
        cur = profiles.get_profile(name) if editing else None
        if editing and cur is None:
            self.status.configure(text=f"档案「{name}」读不出来", foreground="#b45309")
            return

        dlg = tk.Toplevel(self)
        dlg.title("编辑装置档案" if editing else "新建装置档案")
        dlg.configure(bg=BG)
        dlg.resizable(False, False)
        dlg.transient(self)
        body = ttk.Frame(dlg, padding=14)
        body.pack(fill="both", expand=True)

        name_var = tk.StringVar(value=cur.name if cur else "")
        mode_var = tk.StringVar(
            value="angled（大角度）" if (cur and cur.stereo_mode == "angled")
            else "parallel（平行）")
        base_var = tk.StringVar(
            value=f"{cur.expected_baseline_mm:g}"
            if (cur and cur.expected_baseline_mm) else "")
        dist_var = tk.StringVar(
            value=f"{cur.working_distance_m:g}"
            if (cur and cur.working_distance_m) else "")
        w_var = tk.StringVar(value=str(cur.width) if (cur and cur.width) else "")
        h_var = tk.StringVar(value=str(cur.height) if (cur and cur.height) else "")
        fps_var = tk.StringVar(value=str(cur.fps) if (cur and cur.fps) else "")

        ttk.Label(body, text="名称（如「平行装置」「90度装置」）").pack(anchor="w")
        name_entry = ttk.Entry(body, textvariable=name_var, width=32)
        name_entry.pack(fill="x", pady=(2, 8))
        if editing:
            # 只读：改名会改掉标定路径，见 _edit_profile 的说明
            name_entry.configure(state="readonly")

        ttk.Label(body, text="模式").pack(anchor="w")
        ttk.Combobox(body, textvariable=mode_var, state="readonly", width=20,
                     values=("parallel（平行）", "angled（大角度）")).pack(
            anchor="w", pady=(2, 8))

        ttk.Label(body, text="期望基线 mm（可留空，参考值）").pack(anchor="w")
        ttk.Entry(body, textvariable=base_var, width=32).pack(fill="x", pady=(2, 8))
        ttk.Label(body, text="工作距离 m（可留空，参考值）").pack(anchor="w")
        ttk.Entry(body, textvariable=dist_var, width=32).pack(fill="x", pady=(2, 8))

        ttk.Label(body, text="采集档 宽 × 高 @ fps（拍标定图与录制共用）").pack(anchor="w")
        cap_row = ttk.Frame(body)
        cap_row.pack(fill="x", pady=(2, 2))
        ttk.Entry(cap_row, textvariable=w_var, width=7).pack(side="left")
        ttk.Label(cap_row, text="×").pack(side="left", padx=3)
        ttk.Entry(cap_row, textvariable=h_var, width=7).pack(side="left")
        ttk.Label(cap_row, text="@").pack(side="left", padx=3)
        ttk.Entry(cap_row, textvariable=fps_var, width=5).pack(side="left")
        ttk.Label(cap_row, text="fps", foreground=MUTED).pack(side="left", padx=(4, 0))
        ttk.Label(body, text="宽高要一起填；都留空 = 按标定记录的采集档。"
                             "双路建议 1920×1080（2560 双开会让一路掉线）",
                  foreground=MUTED, wraplength=430).pack(anchor="w", pady=(0, 8))

        calib_preview = ((cur.calibration_path if editing and cur.calibration_path else "")
                         or f"config/calib/{cur.name if editing else '<名称>'}")
        ttk.Label(body, text=f"标定路径：{calib_preview}（标定时自动写入）",
                  foreground=MUTED, wraplength=430).pack(anchor="w", pady=(0, 8))

        err = ttk.Label(body, text="", foreground="#b45309", wraplength=430)
        err.pack(anchor="w", pady=(0, 8))

        def save() -> None:
            nm = name_var.get().strip()
            if not nm:
                err.configure(text="名称不能为空")
                return
            mode = "angled" if "angled" in mode_var.get() else "parallel"
            try:
                base = float(base_var.get()) if base_var.get().strip() else None
                dist = float(dist_var.get()) if dist_var.get().strip() else None
            except ValueError:
                err.configure(text="基线/工作距离要填数字（或留空）")
                return

            # 采集档：宽高必须成对 —— 只填一个会得到"宽 × 相机默认高"这种组合
            w_txt, h_txt = w_var.get().strip(), h_var.get().strip()
            f_txt = fps_var.get().strip()
            width = height = fps = None
            if w_txt or h_txt:
                if not (w_txt and h_txt):
                    err.configure(text="采集档的宽和高要一起填（或都留空）")
                    return
                try:
                    width, height = int(w_txt), int(h_txt)
                except ValueError:
                    err.configure(text="采集档的宽/高要填整数")
                    return
                if width <= 0 or height <= 0:
                    err.configure(text="采集档的宽/高要是正数")
                    return
            if f_txt:
                try:
                    fps = int(f_txt)
                except ValueError:
                    err.configure(text="fps 要填整数（或留空）")
                    return
                if fps <= 0:
                    err.configure(text="fps 要是正数")
                    return

            kwargs = {}
            if editing and cur.created:
                kwargs["created"] = cur.created       # 保留创建时间，别把编辑记成新建
            try:
                prof = profiles.RigProfile(
                    name=nm, stereo_mode=mode,
                    expected_baseline_mm=base, working_distance_m=dist,
                    width=width, height=height, fps=fps,
                    calibration_path=(
                        cur.calibration_path if editing and cur.calibration_path
                        else f"config/calib/{nm}"),
                    notes=(cur.notes if editing else ""),
                    **kwargs)
                profiles.upsert(prof)
            except profiles.ProfileError as exc:
                err.configure(text=str(exc))
                return

            dlg.destroy()
            self._load()
            self.tree.selection_set(nm)
            self.status.configure(text=f"已{'更新' if editing else '新建'}装置：{nm}",
                                  foreground=MUTED)
            # 改的若是**当前正在用**的档案，主界面的就绪判定/各窗标题必须重读，
            # 否则它还在按旧的基线/采集档做结论（这正是"切换了没生效"的来源）。
            if self.on_changed and profiles.active_name() == nm:
                self.on_changed()

        bar = ttk.Frame(body)
        bar.pack(fill="x")
        ttk.Button(bar, text="确定", command=save).pack(side="right")
        ttk.Button(bar, text="取消", command=dlg.destroy).pack(side="right", padx=(0, 6))

        dlg.grab_set()
