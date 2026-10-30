"""会话工作台：在一段已录会话上按 ①–⑤ 走完离线链路。

为什么它必须是一个独立窗口（2026-10-27 计划书 §4 P4）
----------------------------------------------------
①–⑤ 原来只在经典界面的「双目工作台」里，而新界面的双目窗第 ④ 步只有
"对**这一段**会话跑一次离线三维"。两者的差别不是步骤多少，而是**对象数量**：

- 双目窗第 ④ 步：对象是"当前会话"，用户刚录完，一步到底；
- 本窗口：对象是"任意历史会话"，要能换会话、看它现在到哪一步、从中间接着做。

把它们塞进同一个窗口，就会出现"第 ④ 步里再选一次会话"的第二份真相 ——
那正是旧界面被判 FAIL 的形状（顶栏模式下拉框 vs 标定 meta）。

数据只从 ``apps/_workbench.py`` 取（计划书 §2.3 唯一来源）
--------------------------------------------------------
帧数、同步、有没有 seeds.json / traj3d.json / visualization.json / annotated.mp4、
以及"下一步该做什么"，``summarize_session`` 都已经回答了。这里**不重算一份**：
重算就会和经典界面、数据管理分叉，表现为"同一个会话在两个地方显示两种状态"。

前置条件不满足就禁用，并写清为什么
----------------------------------
旧界面用 ``messagebox.showinfo`` 解释"还没有点选球心"。弹窗的问题是：
用户点之前不知道自己会被拦。这里把判据变成按钮的 enabled 状态 + 一行原因，
点得动就是能跑，点不动就直接看得到缺什么。

跑法：从双目窗第 ④ 步「历史会话…」或门厅「会话工作台」进入。
"""

from __future__ import annotations

import os
import re
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from apps._gui_stereo import SessionPicker, pick_seeds
from apps._gui_widgets import BG, ScrollPage, StatusBar, Surface
from apps._task_runner import ROOT
from apps._workbench import SessionSummary, scan_sessions, summarize_session
from sphere_tracker.paths import stereo_roots

__all__ = ["BenchWindow", "STEPS", "step_states", "summary_text",
           "next_step_text", "offline_cmd", "visualization_path"]

#: ①–⑤ 的键。顺序即界面顺序，也即"按这个顺序做"。
STEPS = ("check", "seed", "run", "view", "export")


# --------------------------------------------------------------------------- #
# 纯逻辑：不碰 Tk，测试可以直接断言（无显示器环境也能验）
# --------------------------------------------------------------------------- #
def visualization_path(session: Path) -> Path | None:
    """会话里的三维结果文件。derived/ 优先（离线重建写在那里），顶层兼容旧会话。"""
    for rel in ("derived/visualization.json", "visualization.json"):
        cand = session / rel
        if cand.is_file():
            return cand
    return None


def offline_cmd(s: SessionSummary) -> list[str]:
    """第 ③ 步的命令行。

    有 ``seeds.json`` 就必须用 specular（点选的球心只对那个检测器有意义），
    没有就用 ``run_stereo_video`` 自己的默认检测器 —— 与经典界面
    ``_offline_stereo_args`` 同一条规则，只是不带那些面板参数。

    不传 ``--stereo-mode``：默认 auto 会读 session/标定，而 ``SessionSummary``
    把"未记录模式"折成了 parallel，照它传会把老会话按错的几何解算。
    """
    cmd = [sys.executable, str(ROOT / "apps" / "run_stereo_video.py"), str(s.path)]
    seed = s.path / "derived" / "seeds.json"
    if s.has_seeds and seed.is_file():
        cmd += ["--method", "specular", "--seed-json", str(seed)]
    return cmd


def step_states(s: SessionSummary | None) -> dict[str, tuple[bool, str]]:
    """每一步的 (可点, 不可点的原因)。纯函数，界面只套用。

    判据全部来自 ``SessionSummary`` 的既有属性，**不在这里另立规则**：
    ``can_check`` = 左右都有帧；``has_seeds`` / ``has_traj`` /
    ``has_visualization`` / ``has_annotated`` 是 derived/ 下的文件事实。
    """
    if s is None:
        missing = "先选一段会话"
        return {key: (False, missing) for key in (*STEPS, "video")}
    no_frames = "session.json 里左右帧数是 0，这段会话可能是录到一半中断的"
    broken = "这段会话被判为坏数据（" + (s.problems[0] if s.problems else "原因见摘要") + "）"
    data_ok = s.can_check and s.status != "bad"
    # 点选球心/跑三维都要能读出视频帧；坏数据（帧数不一致、几乎全黑）点了也白点
    blocked = no_frames if not s.can_check else broken
    return {
        "check": (s.can_check, no_frames),
        "seed": (data_ok, blocked),
        "run": (data_ok, blocked),
        "view": (s.has_visualization,
                 "还没有 visualization.json —— 先做第 ③ 步"),
        "video": (s.has_annotated,
                  "还没有 annotated.mp4 —— 第 ③ 步跑完会一起生成"),
        "export": (s.has_traj,
                   "还没有 traj3d.json —— 先做第 ③ 步"),
    }


def _has(name: str, flag: bool) -> str:
    return f"{name} {'有' if flag else '无'}"


def summary_text(s: SessionSummary | None) -> str:
    """会话摘要：一段话回答"这段数据现在是什么状态"。"""
    if s is None:
        return "还没有选中会话。"
    mode = "大角度" if s.stereo_mode == "angled" else "平行"
    lines = [
        f"帧数 {s.frames_left}/{s.frames_right}　"
        f"帧率 {s.fps:.1f} fps　模式 {mode}",
        f"同步 {s.sync_method}{s.sync_offset:+d}"
        f"（置信度 {s.sync_confidence:.2f}）",
        "　".join([
            _has("点选球心 seeds.json", s.has_seeds),
            _has("轨迹 traj3d.json", s.has_traj),
            _has("三维 visualization.json", s.has_visualization),
            _has("标注视频 annotated.mp4", s.has_annotated),
        ]),
    ]
    if s.has_traj:
        bits = [f"匹配 {s.n_matched}/{s.n_samples}"]
        if s.z_min is not None and s.z_max is not None:
            bits.append(f"Z {s.z_min:.3f}–{s.z_max:.3f} m")
        if s.residual_median is not None:
            bits.append(f"残差中位 {s.residual_median:.2f} px")
        lines.append("　".join(bits))
    if s.problems:
        lines.append("问题：" + "；".join(s.problems))
    return "\n".join(lines)


def next_step_text(s: SessionSummary | None) -> str:
    """一句话说"现在该点哪个"。"""
    if s is None:
        return ("还没有会话：先去双目窗录一段，或在双目窗第 ④ 步跑一次离线三维，"
                "回来这里就有历史会话可选。")
    if s.status == "bad":
        return "这段会话的数据有问题，先按上面的问题修（重录或修 session.json），别进三维。"
    if s.status == "recorded":
        return "下一步：① 校验数据；金属球（specular）要在 ② 点选球心后才能跑第 ③ 步。"
    if s.status == "seeded":
        return "下一步：③ 跑离线三维。"
    if s.status == "traj":
        return "下一步：④ 看三维；⑤ 可重新导出标准 JSON。"
    return ""


# --------------------------------------------------------------------------- #
# 窗口
# --------------------------------------------------------------------------- #
class BenchWindow(tk.Toplevel):
    """会话工作台：顶部当前会话 + 摘要，下面 ①–⑤。

    ``session`` 给了就直接用；没给且 ``auto_pick`` 为真时取最近一段会话。
    ``auto_pick=False`` 是给自测用的"空工作台"—— 否则断言会随磁盘上的数据漂。
    """

    def __init__(self, app, *, session: str | Path | None = None,
                 auto_pick: bool = True) -> None:
        super().__init__(app)
        self.app = app
        self.configure(bg=BG)
        self.geometry("880x680")
        self.minsize(780, 560)
        self.surface = Surface(self)

        self._summary: SessionSummary | None = None
        self._path: Path | None = Path(session) if session else None
        if self._path is None and auto_pick:
            self._path = _latest_session()
        self._buttons: dict[str, ttk.Button] = {}
        self._why: dict[str, ttk.Label] = {}

        # 先占底部：pack 从容器边缘往里分配，后 pack 的主内容区会盖住状态条
        self.status = StatusBar(self, on_log=app.open_console,
                                on_stop=self._stop_my_tasks)
        self.status.pack(fill="x", side="bottom")
        page = ScrollPage(self, self.surface)
        page.pack(fill="both", expand=True)
        body = page.content
        self._build_head(body)
        self._build_steps(body)
        self._reload()

    # ------------------------------------------------------------------ #
    # 顶部：当前会话 + 摘要
    # ------------------------------------------------------------------ #
    def _build_head(self, parent) -> None:
        box = self.surface.card(
            parent, "当前会话",
            "①–⑤ 只对这一段会话生效。换会话不会丢东西；每一步都进「任务管理器和运行日志」，可单独中断。")
        line = ttk.Frame(box)
        line.grid(row=self.surface._next_row(box), column=0, columnspan=9,
                  sticky="w", pady=(2, 4))
        ttk.Label(line, text="当前会话：").pack(side="left")
        self._lbl_name = ttk.Label(line, text="未选择")
        self._lbl_name.pack(side="left")
        ttk.Button(line, text="换会话…", command=self._pick_session).pack(
            side="left", padx=(12, 0))
        # 摘要用正文样式而不是 Hint：帧数/帧率是要读的数字，灰字看不清
        self._lbl_facts = self.surface.hint(box, "…", style="TLabel")
        self._lbl_next = self.surface.hint(box, "")

    # ------------------------------------------------------------------ #
    # ①–⑤
    # ------------------------------------------------------------------ #
    def _step(self, parent, key: str, title: str, hint: str):
        box = self.surface.card(parent, title, hint)
        # 说明行先占好位置：enabled 时清空、disabled 时写原因，
        # 这样按钮位置不会随状态上下跳。
        self._why[key] = self.surface.hint(box, "")
        return box, self.surface.action_row(box)

    def _build_steps(self, parent) -> None:
        _box, row = self._step(
            parent, "check", "① 校验数据",
            "先看这段会话本身有没有问题：左右帧数是否一致、同步是否可信、视频是否几乎全黑。"
            "坏数据进三维只会得到一条看起来像结果的错轨迹。")
        self._buttons["check"] = self.surface.primary(row, "校验数据", self._run_check)

        _box, row = self._step(
            parent, "seed", "② 点选球心",
            "在这一帧的左图、右图上各点一次球心，存成 derived/seeds.json。"
            "只有金属球（specular 检测）需要它；彩色球可以跳过。")
        self._buttons["seed"] = self.surface.primary(row, "点选球心", self._run_seed)

        _box, row = self._step(
            parent, "run", "③ 跑离线三维",
            "读两路视频，按标定解出三维轨迹，写出 traj3d.json 与 visualization.json。"
            "有点选球心时自动用 specular 解算。")
        self._buttons["run"] = self.surface.primary(row, "跑离线三维", self._run_offline)

        _box, row = self._step(
            parent, "view", "④ 看三维",
            "3D 查看器读的是这段会话的 visualization.json；标注视频用系统播放器打开，"
            "用来核对轨迹有没有跳到别的物体上。")
        self._buttons["view"] = self.surface.primary(row, "打开 3D 查看器",
                                                     self._open_viewer)
        self._buttons["video"] = self.surface.secondary(row, "打开标注视频",
                                                        self._open_video)

        _box, row = self._step(
            parent, "export", "⑤ 导出标准 JSON",
            "把 traj3d 写成统一的 visualization.json（含 CSV 与示意图），"
            "给别的程序读。第 ③ 步已经写过一份，这一步是重新导出。")
        self._buttons["export"] = self.surface.primary(row, "导出 JSON", self._run_export)

    # ------------------------------------------------------------------ #
    # 状态
    # ------------------------------------------------------------------ #
    def select_session(self, path: str | Path) -> None:
        """换到另一段会话。目录不合法时不换，只说明原因。"""
        p = Path(path)
        if not (p / "session.json").is_file():
            self.status.set_status(f"这不是会话目录（缺 session.json）：{p.name}", "warn")
            return
        self._path = p
        self._reload()
        self.status.set_status(f"已选会话：{p.name}", "ok")

    def _reload(self) -> None:
        s: SessionSummary | None = None
        if self._path is not None:
            if (self._path / "session.json").is_file():
                s = summarize_session(self._path)
            else:
                self.status.set_status(f"会话目录不见了：{self._path}", "warn")
        self._summary = s
        self._lbl_name.configure(text=s.id if s is not None else "未选择")
        self._lbl_facts.configure(text=summary_text(s))
        self._lbl_next.configure(text=next_step_text(s))
        self._apply_states()

    def _apply_states(self) -> None:
        states = step_states(self._summary)
        for key, btn in self._buttons.items():
            enabled, why = states[key]
            btn.configure(state="normal" if enabled else "disabled")
            lab = self._why[key]
            if enabled:
                lab.configure(text="", style="Hint.TLabel")
            else:
                lab.configure(text="还不能用：" + why, style="Warn.TLabel")

    # ------------------------------------------------------------------ #
    # 动作：每步都调现有脚本，不在这里重写算法
    # ------------------------------------------------------------------ #
    def _pick_session(self) -> None:
        # mode=None = 跨装置：工作台不按几何过滤，历史会话本来就什么都有
        SessionPicker(self, mode=None, on_pick=self._on_picked,
                      current=str(self._path or ""))

    def _on_picked(self, path: str, _why: str = "") -> None:
        self.select_session(path)

    def _run_check(self) -> None:
        s = self._summary
        if s is None:
            return
        self.app.tasks.start(
            [sys.executable, str(ROOT / "apps" / "tools" / "check_dataset.py"),
             str(s.path), "--strict"],
            f"校验数据 {s.id}", on_done=lambda _t: self._reload())
        self.status.set_status("正在校验这段会话，逐项结果会写到运行日志", "info")

    def _run_seed(self) -> None:
        """② 点选球心：进程内打开 SeedPicker（``seed_picker.py`` 没有 __main__，
        起子进程会立刻退出、什么也不发生）。点完回来刷新摘要与按钮。"""
        s = self._summary
        if s is None:
            return
        saved = pick_seeds(self, s.path)
        if saved is None:
            self.status.set_status("没有保存球心（窗口被关掉或左右没点完）", "warn")
            return
        self._reload()
        self.status.set_status(f"球心已保存：{Path(saved).name}；现在可以跑第 ③ 步",
                               "ok")

    def _run_offline(self) -> None:
        s = self._summary
        if s is None:
            return
        self.app.tasks.start(offline_cmd(s), f"离线三维 {s.id}",
                             on_done=lambda _t: self._reload())
        self.status.set_status(
            f"正在解算 {s.id}；跑完 ④ 的按钮会自动亮起", "info")

    def _open_viewer(self) -> None:
        s = self._summary
        if s is None:
            return
        src = visualization_path(s.path)
        if src is None:
            # 不盲启：view_3d 的 source 是必填位置参数，不给就 exit 2，
            # 用户看到的是"点了没反应"（B-97 的教训）
            self.status.set_status("这段会话还没有 visualization.json：先做第 ③ 步",
                                   "warn")
            return
        self.app.tasks.start(
            [sys.executable, str(ROOT / "apps" / "view_3d.py"), str(src)],
            f"3D 查看器 {s.id}")
        self.status.set_status(f"正在打开 3D 查看器：{s.id}", "info")

    def _open_video(self) -> None:
        s = self._summary
        if s is None:
            return
        video = s.path / "derived" / "annotated.mp4"
        opener = getattr(os, "startfile", None)
        if opener is None:
            self.status.set_status(f"这个平台不能直接开视频，文件在：{video}", "warn")
            return
        try:
            opener(str(video))
        except OSError as exc:
            self.status.set_status(f"打不开 {video.name}：{exc}", "warn")
            return
        self.status.set_status(f"已用系统播放器打开 {video.name}", "info")

    def _run_export(self) -> None:
        s = self._summary
        if s is None:
            return
        self.app.tasks.start(
            [sys.executable, str(ROOT / "apps" / "export_visualization.py"),
             str(s.path), "--plot"],
            f"导出标准数据 {s.id}", on_done=lambda _t: self._reload())
        self.status.set_status("正在导出 visualization.json / CSV / 示意图", "info")

    def _stop_my_tasks(self) -> None:
        ok = self.app.tasks.stop_all(graceful=True)
        self.status.set_status(f"已确认停掉 {ok} 个任务", "ok" if ok else "warn")


def _latest_session() -> Path | None:
    """最近一段会话。排序规则交给 ``_workbench.scan_sessions``（按目录名时间戳，
    不按 mtime —— 补跑一次离线会把 mtime 刷成今天，默认值就漂了）。

    旧位置也要看（``stereo_roots()``）：工作台的默认会话是**读**，迁移前后都得选得中。
    两个根各自取最新的那一段，再在这两段之间比 —— 段内排序仍由 `_workbench` 定，
    这里只比两段的目录名（`YYYYMMDD-HHMMSS` 可直接字符串比较，不规范的退回 mtime）。
    """
    rows = [r for root in stereo_roots() for r in scan_sessions(root, limit=1)]
    if not rows:
        return None

    def key(s: SessionSummary) -> tuple[str, float]:
        m = re.match(r"(\d{8}-\d{6})", s.id)
        return (m.group(1) if m else "", s.path.stat().st_mtime)

    return max(rows, key=key).path
