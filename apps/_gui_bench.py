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
from collections import namedtuple
from pathlib import Path
from tkinter import ttk

from apps._gui_stereo import SessionPicker, pick_seeds
from apps._gui_widgets import BG, ScrollPage, StatusBar, Surface, fit_to_content
from apps._collapsible import CollapsibleSection
from apps._task_runner import ROOT
from apps._workbench import SessionSummary, scan_sessions, summarize_session
from sphere_tracker.paths import stereo_roots

__all__ = ["BenchWindow", "STEP_SPECS", "BUTTON_KEYS", "step_states",
           "summary_text", "next_step_text", "offline_cmd", "visualization_path"]


# --------------------------------------------------------------------------- #
# ①–⑤ 的**唯一来源**：卡片、按钮、按钮的可用性判据全在这一张表里
# --------------------------------------------------------------------------- #
# 为什么必须是一张表（2026-10-31，缺陷 #1）
# ----------------------------------------
# 上一版把"按钮"和"按钮的说明标签"分成两份手写结构，**键的粒度还不一样**：
# ``self._buttons`` 按**按钮**建（``view`` / ``video`` 各一条），
# ``self._why`` 却按**步**建（``_step(parent, key)`` 里一句 ``self._why[key]``）。
# 第 ④ 步在同一行后来多了一个按钮「打开标注视频」（键 ``video``）：
# ``self._buttons["video"]`` 有了，而 ``self._why["video"]`` 永远不会有 ——
# ``_apply_states`` 取 ``self._why[key]`` 时 ``KeyError: 'video'``，**整窗构造失败**。
# 门厅「会话工作台」与双目④「历史会话…」两个入口都进不去，用户看到的是一张没标题的空窗口。
# （别记成「``step_states`` 少了 ``video``」：那两份里 ``states`` 是有 ``video`` 的，
#   缺的是**每按钮一条**的说明标签 —— 加 ``states.get()`` 也救不了 ``_why``。）
#
# 这不是"少写一个键"的手误，是**两份结构必然分叉**（连键的粒度都能不一致）。所以：
# 每个按钮把自己的判据一起写进同一行；``step_states`` 只是遍历这张表，
# 界面的按钮也由这张表生成 —— 想加按钮就必须同时给出判据，分叉在结构上不成立。
#
# 也别退化成 ``states.get(key)``/``self._why.get(key)``：那只是把崩溃换成
# "永远显示为空/永远不能用"，下一个人加按钮时仍然不会知道漏了什么。
_Button = namedtuple("_Button", "key text action kind predicate")
_Step = namedtuple("_Step", "key title hint buttons")

#: 判据的入参是 ``SessionSummary``（已选会话，非 None）。返回 ``(可点, 不可点的原因)``。
#: 判据全部取自 ``SessionSummary`` 的既有属性，**不在这里另立规则**。
_NO_FRAMES = "session.json 里左右帧数是 0，这段会话可能是录到一半中断的"


def _blocked(s: SessionSummary) -> str:
    """点选球心/跑三维这类"要读帧"的步骤为什么不能点。"""
    if not s.can_check:
        return _NO_FRAMES
    return "这段会话被判为坏数据（" + (s.problems[0] if s.problems else "原因见摘要") + "）"


def _data_ok(s: SessionSummary) -> bool:
    """能读出视频帧，且不是坏数据 —— 第 ②③ 步的共同前提。"""
    return s.can_check and s.status != "bad"


def _st_check(s: SessionSummary) -> tuple[bool, str]:
    return s.can_check, _NO_FRAMES


def _st_seed(s: SessionSummary) -> tuple[bool, str]:
    return _data_ok(s), _blocked(s)


def _st_run(s: SessionSummary) -> tuple[bool, str]:
    return _data_ok(s), _blocked(s)


def _st_view(s: SessionSummary) -> tuple[bool, str]:
    return s.has_visualization, "还没有 visualization.json —— 先做第 ③ 步"


def _st_video(s: SessionSummary) -> tuple[bool, str]:
    return s.has_annotated, "还没有 annotated.mp4 —— 第 ③ 步跑完会一起生成"


def _st_export(s: SessionSummary) -> tuple[bool, str]:
    return s.has_traj, "还没有 traj3d.json —— 先做第 ③ 步"


STEP_SPECS: tuple[_Step, ...] = (
    _Step("check", "① 校验数据",
          "先看这段会话本身有没有问题：左右帧数是否一致、同步是否可信、视频是否几乎全黑。"
          "坏数据进三维只会得到一条看起来像结果的错轨迹。",
          (_Button("check", "校验数据", "_run_check", "primary", _st_check),)),
    _Step("seed", "② 点选球心",
          "在这一帧的左图、右图上各点一次球心，存成 derived/seeds.json。"
          "只有金属球（specular 检测）需要它；彩色球可以跳过。",
          (_Button("seed", "点选球心", "_run_seed", "primary", _st_seed),)),
    _Step("run", "③ 跑离线三维",
          "读两路视频，按标定解出三维轨迹，写出 traj3d.json 与 visualization.json。"
          "有点选球心时自动用 specular 解算。",
          (_Button("run", "跑离线三维", "_run_offline", "primary", _st_run),)),
    _Step("view", "④ 看三维",
          "3D 查看器读的是这段会话的 visualization.json；标注视频用系统播放器打开，"
          "用来核对轨迹有没有跳到别的物体上。",
          (_Button("view", "打开 3D 查看器", "_open_viewer", "primary", _st_view),
           _Button("video", "打开标注视频", "_open_video", "secondary", _st_video))),
    _Step("export", "⑤ 导出标准 JSON",
          "把 traj3d 写成统一的 visualization.json（含 CSV 与示意图），"
          "给别的程序读。第 ③ 步已经写过一份，这一步是重新导出。",
          (_Button("export", "导出 JSON", "_run_export", "primary", _st_export),)),
)

#: 界面上**全部**按钮的键。``step_states`` 的返回值必须**恰好**覆盖它们。
BUTTON_KEYS = tuple(btn.key for step in STEP_SPECS for btn in step.buttons)


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
    """每个**按钮**的 (可点, 不可点的原因)。纯函数，界面只套用。

    返回值的键**由 ``STEP_SPECS`` 决定**（= ``BUTTON_KEYS``），不是另一份手写清单：
    加按钮就一定要在表里给它一个判据，否则这里根本不会返回它的键，界面也不会去查。
    没选会话时所有步骤同一个原因 —— 那时"缺什么"只有一个答案。
    """
    if s is None:
        return {key: (False, "先选一段会话") for key in BUTTON_KEYS}
    return {btn.key: btn.predicate(s) for step in STEP_SPECS for btn in step.buttons}


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

    #: 内容区宽度。高度按内容算（``fit_to_content``），不写死。
    WIDTH = 880

    def __init__(self, app, *, session: str | Path | None = None,
                 auto_pick: bool = True) -> None:
        super().__init__(app)
        self.app = app
        self.configure(bg=BG)
        # 高度不写死：五张步骤卡加起来多高由内容决定（`fit_to_content`）。
        # 写死 680 的后果与双目窗一样 —— ⑤ 导出被裁在窗口下沿。
        self.geometry(f"{self.WIDTH}x600")
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
        self._page = ScrollPage(self, self.surface)
        self._page.pack(fill="both", expand=True)
        body = self._page.content
        self._build_head(body)
        self._build_steps(body)
        self._build_glossary(body)
        self._reload()
        # 量内容前必须先跑完 `_reload`：五条"还不能用：…"的原因行会换行/加高，
        # 量早了算出来的高度偏小，⑤ 照样被裁。
        fit_to_content(self, self._page, self.status, width=self.WIDTH,
                       screen_h=self.app.work_area()[1])

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
    def _build_steps(self, parent) -> None:
        """按 ``STEP_SPECS`` 生成卡片与按钮 —— 界面与判据同源，不再手写第二份。

        布局是"标题 + 一行（按钮 + 为什么不能点）"，每步的说明文字统一收在
        卡片列表末尾的折叠区（见 :meth:`_build_glossary`）。这不是为了好看：
        一行说明即使空着也占 27 px，五步 + 五段说明加起来 1000+ px，
        ⑤ 导出必然掉到首屏下面（2026-10-31 实测内容 1070 px、可用 869 px）。
        """
        for step in STEP_SPECS:
            box = self.surface.card(parent, step.title)
            row = self.surface.action_row(box)
            for btn in step.buttons:
                make = (self.surface.primary if btn.kind == "primary"
                        else self.surface.secondary)
                self._buttons[btn.key] = make(row, btn.text, getattr(self, btn.action))
            # 原因文字跟按钮**同一行**：它解释的就是旁边那个按钮。
            # 键按**按钮**登记（不是按卡片）：一张卡里有两个按钮时，两个键指向
            # 同一个标签 —— 它会同时列出这两个按钮各自缺什么。
            note = self.surface.inline_note(row, "")
            for btn in step.buttons:
                self._why[btn.key] = note

    def _build_glossary(self, parent) -> None:
        """五步"在做什么"的说明，默认收起。

        文字与原来逐字一致（只是从每张卡片的说明行搬到这里），所以用户读到的
        还是同一套话；区别是**默认不占首屏**。要改这些字，改 ``STEP_SPECS``。
        """
        section = CollapsibleSection(parent, "五步分别做什么", expanded=False,
                                    summary="每步在干什么、什么时候能点")
        # 页面里的卡片是 pack 的（见 Surface.card），这里也必须 pack ——
        # 同一个容器里混用 pack 与 grid 会直接抛 TclError。
        section.pack(fill="x", padx=16, pady=(10, 0))
        for step in STEP_SPECS:
            self.surface.hint(section.content, f"{step.title}　{step.hint}")

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
        """按 ``STEP_SPECS`` 的**同一张表**写按钮状态与原因。

        遍历的是表，而不是 ``self._buttons`` —— 但两者必须一致，所以先对一遍：
        界面上的按钮若不在表里，说明有人手写了一个"表外的按钮"，而它没有判据、
        也没有原因行可写。这时**当场报出来**（不是悄悄跳过、也不是给它一个默认
        状态）：上一版就是"表外按钮"活到了运行时，以 ``KeyError: 'video'``
        的形式把整窗打死，而且那句 traceback 指不到"少写了判据"。
        """
        states = step_states(self._summary)
        stray = sorted(set(self._buttons) - set(states))
        if stray:
            raise RuntimeError(
                f"界面上的按钮 {stray} 不在 STEP_SPECS 里（因此没有可用性判据与原因行）"
                "：把它们连同判据一起写进 STEP_SPECS，别在 _build_steps 里手写按钮")
        for step in STEP_SPECS:
            off = [f"{btn.text}：{states[btn.key][1]}"
                   for btn in step.buttons if not states[btn.key][0]]
            for btn in step.buttons:
                self._buttons[btn.key].configure(
                    state="normal" if states[btn.key][0] else "disabled")
            lab = self._why[step.buttons[0].key]
            if off:
                lab.configure(text="还不能用：" + "；".join(off), style="Warn.TLabel")
            else:
                lab.configure(text="", style="Hint.TLabel")

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
            f"校验数据 {s.id}",
            on_done=self._on_done("校验", "这段会话通过了数据校验（--strict），可以进第 ③ 步",
                                  "数据校验没通过：看运行日志里它列的每一项问题"))
        self.status.set_status("正在校验这段会话，逐项结果会写到运行日志", "info")

    def _on_done(self, what: str, ok_text: str, bad_text: str):
        """任务跑完之后**一定要**回写状态条。

        为什么单独抽出来（2026-10-31 走查）：`on_done` 原来只 ``_reload()``，
        于是任务跑完（甚至失败）状态条仍停在"正在校验…"—— 用户不知道它结束
        没有，只会一直等。三个会起任务的步骤（校验/离线/导出）同一套写法，
        所以一处写对就都对了。
        """
        def _cb(task) -> None:
            self._reload()                       # 先把新的文件事实读进来
            code = getattr(task, "exit_code", None)
            if code == 0:
                self.status.set_status(ok_text, "ok")
            elif code is None:
                self.status.set_status(f"{what}被中断了，没有结论（可以再点一次）", "warn")
            else:
                self.status.set_status(
                    f"{bad_text}（返回码 {code}；完整输出在运行日志里）", "warn")
        return _cb

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
                             on_done=self._on_done(
                                 "离线三维",
                                 f"{s.id} 解算完了；④ 的两个按钮现在可用了",
                                 "离线三维没跑成：看运行日志里它的报错"))
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
            f"导出标准数据 {s.id}",
            on_done=self._on_done("导出", "导出完成：visualization.json / CSV / 示意图都写好了",
                                  "导出失败：看运行日志里它的报错"))
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
