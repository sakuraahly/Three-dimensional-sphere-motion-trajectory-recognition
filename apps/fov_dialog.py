"""FOV 管理器：查看 / 测量 / 清除「这台机器两路相机的固有视场差」。

为什么要有这个界面
------------------
两台同型号相机装上去，出厂焦距并不一样（本机实测左路比右路放大 ~1.5 倍）。
这个差异是**装置属性**：换标定板、换场景都不会变，但它此前只活在"某一次
双目标定"的 `meta.fov` 里 —— 重新标定就没了，现场估计失败就变成 null，
90° 装置和单目更是完全看不到它。

于是同一套装置每次标定都要重新猜一遍裁剪参数，`crop_rect` 每次都差几像素，
内参跟着漂，而屏幕上一切正常。这个界面把它**从标定里拆出来**：

- 量一次（可以从任何一段双目录像取一帧）；
- 存进装置档案（跟装置走，切换装置就跟着换）；
- 之后平行标定直接复用（`--fov-from-profile`），口径不再漂。

三条边界
--------
1. **90° 装置不做裁剪**：两路光轴近乎垂直、视场几乎不重叠，单应裁剪在那种
   几何下没有物理意义（`apps/record_stereo.py` 在 angled 下明确忽略 fov）。
   这条记录在那里只是"这台机器两路差多少"的事实，供诊断与单目参考。
2. **参数是像素量纲的**：换分辨率/换相机后旧值失效。所以记录里存了测量时的
   两路分辨率，后端发现不一致会直接报错，而不是沿用。
3. **只改档案，不动标定**：这里不生成标定文件；标定仍由双目窗第②步做。

装置级「应用」开关（2026-11-15）
--------------------------------
「启用」= **把这份 FOV 应用到当前装置的两路相机画面上**，是装置级事实，不是
两个流程开关：标定与录制都服从它。本对话框的勾选框与双目窗第②步那个是**同一个
开关**（都读写档案的 ``fov_apply``），由 :func:`stereo.fov_policy.resolve_fov_apply`
统一判定后果。默认关；没有 FOV 记录时勾选框禁用。
"""

from __future__ import annotations

import dataclasses
import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

# 按**标记**上溯到含 `sphere_tracker/` 的祖先，不数层数（`selftest_entrypoints`
# 对这条是硬的：数层数在被移动/加包装目录时会静默指错）。
ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "sphere_tracker").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps import _fov_store                                 # noqa: E402
from apps._gui_widgets import BG, MUTED, WARN               # noqa: E402
from sphere_tracker.paths import STEREO_ROOT                # noqa: E402
from sphere_tracker.video_files import (                    # noqa: E402
    pick_left_video, pick_right_video,
)
from stereo import profiles                                 # noqa: E402


def _session_has_fov_record(sess: Path) -> bool:
    """这段会话有没有 ``rig.fov``（标定当时在**原始画面**上算的那份）。

    有它就该直接读，不该在录像上重新量：平行装置的录像通常已按它裁剪过，
    在裁剪后的画面上再量是"二次裁剪"的残余差（详见 `_measure_from_session`）。
    """
    try:
        data = json.loads((Path(sess) / "session.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(data, dict):
        return False
    rig = data.get("rig")
    fov = rig.get("fov") if isinstance(rig, dict) else None
    return isinstance(fov, dict) and bool(fov)


class FovDialog(tk.Toplevel):
    """当前装置档案的 FOV 记录：看 / 量 / 清。

    ``app`` 是 `apps._gui_app.App`（要它的 ``tasks`` 跑测量子进程、
    ``state`` 读当前档案）。``on_changed`` 在档案变化后调用，让门厅/窗口重判。
    """

    def __init__(self, parent, app, on_changed=None) -> None:
        super().__init__(parent)
        self.app = app
        # 这个对话框不经过 App.open_window（没有统一打标），自己登记归属键，
        # 让起任务时的 owner=self.task_owner 与模式窗口径一致。
        self.task_owner = "fov"
        # 不经 App.open_window，没有 close_window 代劳：自己接 WM_DELETE_WINDOW，
        # 按 X 时先停本窗起的任务，再销毁 —— 否则测量进程留在后台继续跑。
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.on_changed = on_changed
        self.title("FOV 管理器 · 两路固有视场差")
        self.configure(bg=BG)
        # 标准窗口：最小化 / 最大化 / 关闭（AGENTS.md §7.1）。
        # 不用 transient()（Windows 上变工具窗口、只剩 ×）、不锁尺寸、不全局 grab。
        self.resizable(True, True)

        body = ttk.Frame(self, padding=14)
        body.pack(fill="both", expand=True)

        ttk.Label(
            body, foreground=MUTED, wraplength=560,
            text="两台同型号相机的焦距常常不一样（左路可能比右路放大 1.5 倍）。"
                 "这是装置属性：量一次、存进档案，之后平行标定直接复用同一组裁剪"
                 "参数，口径不再漂。90° 装置不做裁剪（两路几乎不共视），"
                 "这条记录在那里只用于诊断。"
        ).pack(anchor="w", pady=(0, 10))

        self.lbl_rig = ttk.Label(body, text="", wraplength=560)
        self.lbl_rig.pack(anchor="w", pady=(0, 8))

        # 记录详情：左列字段名、右列值。用 grid 对齐，比一行行 Label 好读。
        grid = ttk.Frame(body)
        grid.pack(fill="x", pady=(0, 8))
        grid.columnconfigure(1, weight=1)
        self._rows: dict[str, ttk.Label] = {}
        for i, (key, label) in enumerate((
                ("scale", "尺度 left→right"),
                ("zoom", "等效放大"),
                ("crop", "裁剪矩形"),
                ("target", "共同输出尺寸"),
                ("sizes", "测量时分辨率"),
                ("quality", "估计质量"),
                ("seen", "测量时间"),
                ("source", "来源"),
        )):
            ttk.Label(grid, text=label, foreground=MUTED).grid(
                row=i, column=0, sticky="w", padx=(0, 14), pady=1)
            val = ttk.Label(grid, text="—", wraplength=380)
            val.grid(row=i, column=1, sticky="w", pady=1)
            self._rows[key] = val

        self.status = ttk.Label(body, text="", foreground=MUTED, wraplength=560)
        self.status.pack(anchor="w", pady=(6, 8))

        # 装置级「应用」开关：与双目窗第②步读写同一个档案字段 fov_apply。
        # 没有 FOV 记录时禁用（启用一个不存在的记录没有意义，_load 里控制）。
        self.var_apply = tk.BooleanVar(value=False)
        self.chk_apply = ttk.Checkbutton(
            body,
            text="把这份 FOV 应用到当前装置的左右相机（标定与录制都服从）",
            variable=self.var_apply, command=self._on_toggle_apply,
            state="disabled")
        self.chk_apply.pack(anchor="w", pady=(0, 8))

        bar = ttk.Frame(body)
        bar.pack(fill="x")
        ttk.Button(bar, text="刷新", command=self._load).pack(side="left")
        ttk.Button(bar, text="清除记录", command=self._clear).pack(side="right")
        ttk.Button(bar, text="从会话取 FOV…", command=self._measure_from_session).pack(
            side="right", padx=(0, 6))
        ttk.Button(bar, text="从当前标定读入", command=self._import_from_calibration).pack(
            side="right", padx=(0, 6))

        self._load()

    # -- 读取当前档案 ------------------------------------------------------
    def _profile(self):
        return getattr(self.app.state, "profile", None)

    def _profile_name(self) -> str:
        prof = self._profile()
        return (getattr(prof, "name", "") or "") if prof is not None else ""

    def _alive(self) -> bool:
        try:
            return bool(self.winfo_exists())
        except tk.TclError:
            return False

    def _load(self) -> None:
        prof = self._profile()
        name = self._profile_name()
        if not name:
            self.lbl_rig.configure(
                text="当前没有选中的装置档案。先在门厅「装置档案」里选一个，"
                     "FOV 记录是跟着装置走的。")
            for lbl in self._rows.values():
                lbl.configure(text="—")
            self.var_apply.set(False)
            self.chk_apply.configure(state="disabled")
            self.status.configure(text="", foreground=MUTED)
            return
        mode = ("大角度（angled）" if getattr(prof, "stereo_mode", "") == "angled"
                else "平行（parallel）")
        self.lbl_rig.configure(text=f"当前装置：{name}　模式：{mode}")

        fov = getattr(prof, "fov", None)
        applied = bool(getattr(prof, "fov_apply", False))
        # 管理器不需要标定文件：轻量逻辑直接读档案的 fov_apply 即可。
        self.var_apply.set(applied and bool(fov))
        self.chk_apply.configure(state="normal" if fov else "disabled")
        if not fov:
            for lbl in self._rows.values():
                lbl.configure(text="—")
            self.status.configure(
                text="这条档案还没有 FOV 记录。点「从会话量一次…」选一段双目录像；"
                     "标定带 FOV 时会自动记进来（不带 FOV 的标定不会写记录）。",
                foreground=WARN)
            return

        scale = fov.get("scale")
        self._rows["scale"].configure(
            text=f"{scale:.4f}" if isinstance(scale, (int, float)) else "—")
        if isinstance(scale, (int, float)) and scale > 0:
            if abs(scale - 1.0) <= 0.02:
                zoom = "两路视场基本一致"
            else:
                zoom = f"{1.0 / scale:.2f}×（{'左' if fov.get('reference_side') == 'left' else '右'}路更窄）"
        else:
            zoom = "—"
        self._rows["zoom"].configure(text=zoom)
        rect = fov.get("crop_rect") or []
        self._rows["crop"].configure(
            text=f"{tuple(rect)}（在 {fov.get('crop_side', '?')} 的像素系里）"
            if len(rect) == 4 else "—")
        target = fov.get("target_size") or []
        self._rows["target"].configure(
            text=f"{target[0]} × {target[1]}" if len(target) == 2 else "—")
        ls, rs = fov.get("left_size"), fov.get("right_size")
        self._rows["sizes"].configure(
            text=(f"左 {ls[0]}x{ls[1]} / 右 {rs[0]}x{rs[1]}"
                  if ls and rs else "未记录（换分辨率后无法核对）"))
        q, inl = fov.get("quality"), fov.get("inliers")
        self._rows["quality"].configure(
            text=(f"{fov.get('method', '?')} / inliers {inl} / quality {q:.3f}"
                  if isinstance(q, (int, float)) else "—"))
        self._rows["seen"].configure(text=fov.get("measured_at") or "—")
        self._rows["source"].configure(text=fov.get("source") or "—")
        if applied:
            self.status.configure(
                text="已应用到当前装置的左右相机：平行标定复用这份参数、"
                     "录制按它裁剪两路；90° 装置仍不裁剪。",
                foreground=MUTED)
        else:
            self.status.configure(
                text="记录已存，但未应用：标定按原始画面（--skip-fov）、"
                     "录制不裁剪。要应用请勾上面的开关。",
                foreground=MUTED)

    # -- 三个动作 ----------------------------------------------------------
    def _import_from_calibration(self) -> None:
        """从当前装置配的那份标定里读 meta.fov。"""
        prof = self._profile()
        if prof is None or not self._profile_name():
            self.status.configure(text="先选一个装置档案", foreground=WARN)
            return
        json_path = _fov_store.calibration_json_path(
            self.app.state._stereo_calib_path())
        payload = _fov_store.fov_payload_from_calibration(json_path)
        if payload is None:
            self.status.configure(
                text=f"{json_path.name} 里没有 FOV 记录（旧标定 / --skip-fov / "
                     f"估计降级成 null）。改用「从会话量一次…」。",
                foreground=WARN)
            return
        try:
            _fov_store.save_to_profile(prof, payload)
        except profiles.ProfileError as exc:
            self.status.configure(text=f"写档案失败：{exc}", foreground=WARN)
            return
        self.app.state.reload()
        self._load()
        self.status.configure(text=f"已从 {json_path.name} 读入并写进档案",
                             foreground=MUTED)
        if self.on_changed:
            self.on_changed()

    def _measure_from_session(self) -> None:
        """从一段双目会话取 FOV：**优先读记录，其次才现场量**。

        ## 为什么顺序反不得（2026-11-04 实测）

        平行装置的录像通常是**已按 FOV 裁剪过**的（`record_stereo` 用
        `fov.target_size` 写盘）。在这种画面上再估一次单应，量到的是"残余
        视场差"：数字自洽、quality 0.988 很好看，但套到原始画面上裁剪框会
        落错位置 —— 这是个**看起来特别成功**的错误结果。

        而 session.json 里的 `rig.fov` 是**标定当时在原始画面上**算的，才是
        该用的那份。所以：有记录就直接读（`--from-session`），没有才现场量
        （`measure_fov.py` 自己会拒绝已裁剪的录像）。
        """
        name = self._profile_name()
        if not name:
            self.status.configure(text="先选一个装置档案", foreground=WARN)
            return
        start = STEREO_ROOT if Path(STEREO_ROOT).is_dir() else ROOT
        picked = filedialog.askdirectory(
            parent=self, title="选一个双目会话目录（含 left/right 视频）",
            initialdir=str(start))
        if not picked:
            return
        sess = Path(picked)
        # 左/右各自独立查找（唯一产地 video_files）：不能按 left 的后缀推 right ——
        # 左 .mkv + 右 .avi 的混合格式会话是合法的，那种写法会误报"找不到 right.mkv"。
        left = pick_left_video(sess)
        if left is None:
            self.status.configure(
                text=f"{sess.name} 里没有左路视频（left.mkv/.mp4/.avi）—— "
                     f"这里要的是双目会话目录（data/stereo/<时间戳>）", foreground=WARN)
            return
        right = pick_right_video(sess)
        if right is None:
            self.status.configure(text=f"{sess.name} 里没有右路视频（right.mkv/.mp4/.avi）",
                                  foreground=WARN)
            return

        tool = ROOT / "apps" / "tools" / "measure_fov.py"
        if _session_has_fov_record(sess):
            # 记录里是**原始画面**上的裁剪参数，比在已裁剪的录像上重量可信得多
            cmd = [sys.executable, str(tool), "--from-session", str(sess),
                   "--write-profile", name]
            note = f"正在读 {sess.name} 里记录的 FOV 参数…"
        else:
            cmd = [sys.executable, str(tool),
                   "--left-video", str(left), "--right-video", str(right),
                   "--frame", "30", "--write-profile", name]
            note = (f"{sess.name} 没有 FOV 记录，正在用第 30 帧现场量…"
                    f"（若这段录像已裁剪，会被拒绝而不是给假值）")
        self.status.configure(text=note, foreground=MUTED)
        self.app.tasks.start(cmd, f"取 FOV（{sess.name}）",
                             on_done=self._on_measured,
                             owner=self.task_owner)

    def _on_measured(self, task) -> None:
        if not self._alive():
            return
        code = getattr(task, "exit_code", None)
        self.app.state.reload()
        self._load()
        if code == 0:
            self.status.configure(text="取到了，已写进档案", foreground=MUTED)
            if self.on_changed:
                self.on_changed()
        else:
            self.status.configure(
                text="没取到。常见原因：① 这段会话没记 FOV，且两路没有共同视场"
                     "（90° 装置的正常情况）；② 录像已被裁剪（会拒绝，避免二次"
                     "裁剪的假值）；③ 画面纹理不足。详细原因在运行日志里。",
                foreground=WARN)

    def _clear(self) -> None:
        prof = self._profile()
        if prof is None or not self._profile_name():
            self.status.configure(text="先选一个装置档案", foreground=WARN)
            return
        try:
            # 清了记录就不能还开着应用开关：一起置 False。
            profiles.upsert(dataclasses.replace(prof, fov=None, fov_apply=False),
                            replace_existing=True)
        except profiles.ProfileError as exc:
            self.status.configure(text=f"清除失败：{exc}", foreground=WARN)
            return
        self.app.state.reload()
        self._load()
        self.status.configure(text="已清除；应用开关已一并关闭", foreground=MUTED)
        if self.on_changed:
            self.on_changed()

    def _on_toggle_apply(self) -> None:
        """勾选/取消 = 改装置档案里那个装置级「是否应用」的开关。

        写档成功后重读 state、重绘本窗、通知门厅/窗口；写失败只动状态栏，
        并把勾选框回滚到档案里的实际值，**不抛**。
        """
        prof = self._profile()
        if prof is None or not self._profile_name():
            self.var_apply.set(False)
            return
        want = bool(self.var_apply.get())
        try:
            profiles.upsert(dataclasses.replace(prof, fov_apply=want),
                            replace_existing=True)
        except profiles.ProfileError as exc:
            self._load()                       # 把勾选框回滚到档案实际值
            self.status.configure(text=f"写应用开关失败：{exc}", foreground=WARN)
            return
        self.app.state.reload()
        self._load()
        if self.on_changed:
            self.on_changed()

    def _on_close(self) -> None:
        """按 X：先停本窗起的任务，再销毁（不经 App.close_window）。"""
        self.app.tasks.stop_by_owner(self.task_owner)
        self.destroy()


def main() -> int:
    """独立运行（不经过门厅）也能打开，方便排查档案问题。"""
    from apps._gui_app import App

    app = App()
    dlg = FovDialog(app.hub, app)
    app.wait_window(dlg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
