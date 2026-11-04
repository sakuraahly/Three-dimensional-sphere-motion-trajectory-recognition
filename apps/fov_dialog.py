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
"""

from __future__ import annotations

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
from stereo import profiles                                 # noqa: E402

#: 会话里认作"两路视频"的名字（与 `record_stereo` 的产物一致）。
_VIDEO_NAMES = ("left.mkv", "left.mp4", "left.avi")


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
        self.on_changed = on_changed
        self.title("FOV 管理器 · 两路固有视场差")
        self.configure(bg=BG)
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

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
            self.status.configure(text="", foreground=MUTED)
            return
        mode = ("大角度（angled）" if getattr(prof, "stereo_mode", "") == "angled"
                else "平行（parallel）")
        self.lbl_rig.configure(text=f"当前装置：{name}　模式：{mode}")

        fov = getattr(prof, "fov", None)
        if not fov:
            for lbl in self._rows.values():
                lbl.configure(text="—")
            self.status.configure(
                text="这条档案还没有 FOV 记录。点「从会话量一次…」选一段双目录像，"
                     "或标定一次（标定成功会自动记进来）。",
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
        self.status.configure(
            text="平行标定会复用它（不用每次重新估）；90° 装置不裁剪，"
                 "只拿它当「两路差多少」的事实。",
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
        left = next((sess / n for n in _VIDEO_NAMES if (sess / n).is_file()), None)
        if left is None:
            self.status.configure(
                text=f"{sess.name} 里没有 left.mkv —— 这里要的是双目会话目录"
                     f"（data/stereo/<时间戳>）", foreground=WARN)
            return
        right = sess / ("right" + left.suffix)
        if not right.is_file():
            self.status.configure(text=f"找不到配对的 {right.name}", foreground=WARN)
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
                             on_done=self._on_measured)

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
            profiles.upsert(_replace_fov(prof, None), replace_existing=True)
        except profiles.ProfileError as exc:
            self.status.configure(text=f"清除失败：{exc}", foreground=WARN)
            return
        self.app.state.reload()
        self._load()
        self.status.configure(text="已清除（下次标定会回到现场估计）", foreground=MUTED)
        if self.on_changed:
            self.on_changed()


def _replace_fov(profile, fov):
    import dataclasses

    return dataclasses.replace(profile, fov=fov)


def main() -> int:
    """独立运行（不经过门厅）也能打开，方便排查档案问题。"""
    from apps._gui_app import App

    app = App()
    dlg = FovDialog(app.hub, app)
    app.wait_window(dlg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
