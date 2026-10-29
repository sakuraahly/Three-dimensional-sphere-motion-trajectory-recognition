"""应用外壳：唯一 Tk 根、状态的唯一来源、窗口单例表、门厅、控制台。

窗口拓扑（2026-10-27 计划书 §2.2）
---------------------------------
::

    App(Tk root, 自身不显示内容, 只持有 AppState + TaskRunner)
      ├ HubWindow            门厅：三张模式卡 + 工具条   ← 唯一入口
      ├ MonoWindow           单目：① 内参 ② 球与量程 ③ 开始
      ├ StereoWindow(parallel / angled)  双目：① 装置 ② 标定 ③ 采集 ④ 分析
      ├ BenchWindow          会话工作台：任意一段会话的 ①–⑤
      └ ConsoleWindow        任务表 + 日志               ← 单例，随时可开

只有两种窗口关系：**门厅 → 模式窗**（关模式窗回门厅）、**任意窗 → 控制台**。
工作台由"门厅工具条"或"双目窗第 ④ 步的历史会话…"打开，是模式窗的下游：
它不反过来打开模式窗，也不复制模式窗的状态。
禁止：模式窗之间互相跳；任何窗口里出现"模式"下拉框（模式 = 你进的哪个窗口）；
任何一步要求用户"记住一个路径再去另一个窗口粘贴"。

状态归属（计划书 §2.3，**唯一来源**）
------------------------------------
============ ==========================================
概念          唯一来源
============ ==========================================
当前装置      ``stereo.profiles.get_active()`` 的档案
当前双目标定  档案 ``calibration_path``，回落 ``config/stereo_calibration``
当前单目标定  ``config/mono_calibration.json``（P0c 之后由标定入口写出）
跑着的任务    ``TaskRunner``（本模块持有唯一实例）
============ ==========================================

窗口只**读**这些来源，不得缓存成自己的属性再写回 —— 否则 ``stereo_mode``
会重新变成两份真相（旧界面那个"顶栏下拉框 vs 标定 meta"就是前车之鉴）。
"""

from __future__ import annotations

import json
import re
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from apps._gui_widgets import (ACCENT, BG, FONT_FAMILY, INK, MUTED, OK, WARN,
                               ScrollPage, StatusBar, Surface, init_style)
from apps._task_runner import ROOT, TaskRunner

__all__ = ["GuiPrefs", "AppState", "App", "HubWindow", "run"]

MONO_CALIB_PATH = ROOT / "config" / "mono_calibration.json"
STEREO_CALIB_PATH = ROOT / "config" / "stereo_calibration"


# --------------------------------------------------------------------------- #
# 状态：只读文件，不缓存写回
# --------------------------------------------------------------------------- #
class GuiPrefs:
    """记住用户改过的输入（``config/gui_state.json``）。**只存输入，不存派生**。

    为什么必须划这条线（计划书 §2.7）：基线、夹角、fx、stereo_mode、就绪度都
    能从标定与装置档案读出来。把它们抄进界面状态文件，就又多了一份会和标定
    分叉的真相 —— 旧界面"顶栏模式下拉框 vs 标定 meta"就是这类问题的现场。
    所以这里有一份 ``FORBIDDEN`` 清单，写进去**直接抛异常**，而不是靠"约定别写"。

    加载失败不报错：用户可能删了文件或手改坏了，回落默认值即可，
    这不该阻止程序启动（同 `load_mono` 的口径）。
    """

    PATH = ROOT / "config" / "gui_state.json"
    FORBIDDEN = ("baseline_mm", "angle_deg", "stereo_mode", "rms",
                 "coverage", ".fx", "fx_")

    def __init__(self) -> None:
        self.data: dict = {}
        self.load()

    def load(self) -> None:
        try:
            raw = json.loads(self.PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.data = {}
            return
        self.data = raw if isinstance(raw, dict) else {}

    def get_str(self, key: str, default: str = "") -> str:
        v = self.data.get(key)
        return default if v is None else str(v)

    def get_bool(self, key: str, default: bool = False) -> bool:
        v = self.data.get(key)
        return v if isinstance(v, bool) else default

    def set(self, key: str, value) -> None:
        if self._forbidden(key):
            raise ValueError(
                f"gui_state 不许存派生值：{key!r}（它必须从标定/装置档案读）")
        self.data[key] = value

    @classmethod
    def _forbidden(cls, key: str) -> bool:
        low = key.lower()
        return any(f in low for f in cls.FORBIDDEN)

    def save(self) -> None:
        try:
            self.PATH.parent.mkdir(parents=True, exist_ok=True)
            self.PATH.write_text(
                json.dumps(self.data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
        except OSError:
            pass                    # 存不下不该打断用户正在做的事


class AppState:
    """所有窗口共读的状态快照。``reload()`` 重新从文件读一遍。

    这里**不做任何推导**：``stereo_mode`` 来自标定的 meta，``baseline``/
``angle`` 来自 ``StereoRig``（与 ``check_stereo_setup`` 同一口径）。
    界面自己推模式 = 第二份真相，正是旧界面被判 FAIL 的那个形状。
    """

    def __init__(self) -> None:
        self.profile = None            # RigProfile | None
        self.rig = None                # StereoRig | None
        self.mono: dict | None = None  # config/mono_calibration.json 的内容
        self.notes: list[str] = []     # 读取过程中的"读不了"（不是错误，是现状）
        self.reload()

    # ---------- 读取 ----------
    def reload(self) -> None:
        self.notes.clear()
        self.profile = None
        self.rig = None
        self.mono = None
        try:
            from stereo import profiles
            self.profile = profiles.get_active()
        except Exception as exc:                            # noqa: BLE001
            self.notes.append(f"装置档案读不了：{type(exc).__name__}")
        self.rig = self._read_rig()
        self.mono = self._read_mono()

    def _stereo_calib_path(self) -> Path:
        """档案指向的标定文件优先，否则用约定的默认路径。

        档案优先是"一套装置一份标定"的落地：换装置不该还要手动换文件。
        """
        p = getattr(self.profile, "calibration_path", None)
        if p:
            cand = Path(p)
            if not cand.is_absolute():
                cand = ROOT / cand
            if cand.exists():
                return cand
        return STEREO_CALIB_PATH

    def _read_rig(self):
        path = self._stereo_calib_path()
        # StereoRig.load 接受无扩展名的基名，会自己找 .json/.npz
        if not path.exists() and not path.with_suffix(".json").exists() \
                and not path.with_suffix(".npz").exists():
            return None
        try:
            from calib.params import StereoRig
            return StereoRig.load(path)
        except Exception as exc:                            # noqa: BLE001
            self.notes.append(f"双目标定读不了：{type(exc).__name__}")
            return None

    def _read_mono(self) -> dict | None:
        if not MONO_CALIB_PATH.exists():
            return None
        try:
            raw = json.loads(MONO_CALIB_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self.notes.append(f"单目标定读不了：{type(exc).__name__}")
            return None
        return raw if isinstance(raw, dict) else None

    # ---------- 相机角色（唯一来源 config/cameras.json） ----------
    def camera_roles(self) -> dict:
        """读 ``config/cameras.json`` 的 roles 表。

        界面**只读**这张表，**不提供手打角色名的输入框** —— 角色是设备的身份，
        不是用户该记的字符串（旧界面让人填 left/right，填错就静默用错相机）。
        """
        path = ROOT / "config" / "cameras.json"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        roles = raw.get("roles") if isinstance(raw, dict) else None
        return roles if isinstance(roles, dict) else {}

    def camera_role_text(self, role: str) -> str:
        """``left`` → ``left = Integrated IR Camera，索引 2``；未固定就直说。"""
        info = self.camera_roles().get(role)
        if not isinstance(info, dict):
            return f"{role}：未固定（启动时自动挑）"
        bits = [f"{role} = {info.get('name') or '未知型号'}"]
        if info.get("index") is not None:
            bits.append(f"索引 {info.get('index')}")
        if not info.get("verified"):
            bits.append("未确认")
        return "，".join(bits)

    # ---------- 派生（只读，不写回） ----------
    def stereo_mode(self) -> str | None:
        """标定里记的模式。**唯一来源是标定 meta**，不是任何界面控件。"""
        meta = getattr(self.rig, "meta", None) or {}
        mode = str(meta.get("stereo_mode") or "").strip()
        return mode or None

    def angle_deg(self) -> float | None:
        try:
            return float(self.rig.angle_deg)
        except (AttributeError, TypeError, ValueError):
            return None

    def baseline_mm(self) -> float | None:
        try:
            return float(self.rig.baseline_mm)
        except (AttributeError, TypeError, ValueError):
            return None

    # ---------- 三张卡的就绪判定 ----------
    def mono_status(self) -> tuple[bool, str]:
        """单目：只看 ``mono_calibration.json`` 在不在、fx 合不合理。"""
        if not self.mono:
            return False, ("未标定。现在会按 60° 视场估算焦距 —— "
                           "1% 焦距误差在 0.6 m 处就是 6 mm，请先做内参标定")
        fx = self.mono.get("fx")
        if not isinstance(fx, (int, float)) or fx <= 0:
            return False, "标定文件存在但 fx 无效，请重新标定"
        rms = self.mono.get("rms_px")
        extra = f"，RMS {rms:.2f} px" if isinstance(rms, (int, float)) else ""
        return True, f"已标定 fx = {float(fx):.0f} px{extra}"

    def parallel_status(self) -> tuple[bool, str]:
        """平行双目：要求标定存在，且模式/夹角都指向"近平行"。"""
        if self.rig is None:
            return False, "没有双目标定。平行双目需要两路光轴近平行并做校正"
        mode = self.stereo_mode()
        angle = self.angle_deg()
        if mode == "angled":
            a = f"实测 {angle:.1f}°" if angle is not None else "模式记为 angled"
            return False, (f"当前标定是**大角度**（{a}）—— 这份结果请用"
                           f"「大角度双目」窗口，选这里会用错几何")
        base = self.baseline_mm()
        head = f"已标定基线 {base:.0f} mm" if base is not None else "已标定"
        if angle is not None and angle >= 15.0:
            return False, f"{head}，但实测夹角 {angle:.1f}° 已不算平行"
        return True, head + (f"，实测夹角 {angle:.1f}°" if angle is not None else "")

    def angled_status(self) -> tuple[bool, str]:
        """大角度双目：接受 joint（跳过 FOV）与 chain（单板，无共视）两种产物。

        旧界面上"单板链式标定"和"双目工作台"是两张互不相干的卡，用户看不出
        前者能不能喂给后者 —— 这里把答案直接写进卡片：**chain 产物就是 angled**，
        进这个窗口就能用。
        """
        if self.rig is None:
            return False, "没有双目标定。大角度可用「两路共视」或「单板链式」两条路"
        mode = self.stereo_mode()
        angle = self.angle_deg()
        base = self.baseline_mm()
        a = f"{angle:.1f}°" if angle is not None else "未知夹角"
        b = f"{base:.0f} mm" if base is not None else "未知基线"
        if mode == "parallel":
            return False, (f"当前标定是**平行**（{a}，基线 {b}）—— 请用"
                           f"「平行双目」窗口")
        return True, f"已标定 {a}，基线 {b}（含单板链式产物）"


# --------------------------------------------------------------------------- #
# 窗口尺寸（沿用旧界面的工作区计算，避免窗口顶出屏幕）
# --------------------------------------------------------------------------- #
def _fit_window(root: tk.Misc) -> tuple[int, int]:
    """给主窗口一个"装得下、不压任务栏"的尺寸。返回 (w, h)。"""
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes
            rect = wintypes.RECT()
            if ctypes.windll.user32.SystemParametersInfoW(
                    0x0030, 0, ctypes.byref(rect), 0):      # SPI_GETWORKAREA
                screen_w = int(rect.right - rect.left)
                screen_h = int(rect.bottom - rect.top)
        except Exception:                                   # noqa: BLE001
            pass
    w = min(980, max(760, screen_w - 120))
    h = min(760, max(560, screen_h - 140))
    return w, h


# --------------------------------------------------------------------------- #
# 应用
# --------------------------------------------------------------------------- #
class App(tk.Tk):
    """唯一 Tk 根。**自身不显示内容**（withdraw），所有界面都在 Toplevel 里。"""

    def __init__(self) -> None:
        super().__init__()
        self.title("球形物体三维运动轨迹识别")
        self.configure(bg=BG)
        init_style(self)
        w, h = _fit_window(self)
        self.geometry(f"{w}x{h}")
        self.minsize(min(820, w), min(560, h))
        self.withdraw()                                 # 门厅才是用户看到的窗口

        self.state = AppState()
        self.prefs = GuiPrefs()
        self._log_lines: list[str] = []
        self._log_listeners: list = []
        self.tasks = TaskRunner(self, on_log=self._on_task_log,
                                on_change=self._on_task_change)
        self._windows: dict[str, tk.Toplevel] = {}

        self.hub = HubWindow(self)
        self.hub.protocol("WM_DELETE_WINDOW", self.close_app)
        self.hub.focus_force()

    # ---------- 日志缓冲 ----------
    def _on_task_log(self, text: str) -> None:
        self._log_lines.append(text)
        if len(self._log_lines) > 2000:                 # 界面日志只是"最近一段"
            del self._log_lines[:500]
        for cb in list(self._log_listeners):
            try:
                cb(text)
            except Exception:                           # noqa: BLE001
                pass

    def _on_task_change(self) -> None:
        if "console" in self._windows:
            window = self._windows["console"]
            if hasattr(window, "refresh"):
                window.refresh()

    def subscribe_log(self, cb) -> None:
        self._log_listeners.append(cb)

    def unsubscribe_log(self, cb) -> None:
        if cb in self._log_listeners:
            self._log_listeners.remove(cb)

    def recent_log(self) -> list[str]:
        return list(self._log_lines)

    # ---------- 窗口单例 ----------
    def open_window(self, key: str, title: str, factory):
        """按 key 开单例窗口：已存在就抬到前面，不存在才建。

        单例是"同一功能一个入口"的落地：门厅和工作台都能进 ``console``，
        但它们指向**同一个对象**，不是两套实现。
        """
        existing = self._windows.get(key)
        if existing is not None and existing.winfo_exists():
            existing.deiconify()
            existing.lift()
            existing.focus_force()
            return existing
        win = factory(self)
        win.title(title)
        self._windows[key] = win
        win.protocol("WM_DELETE_WINDOW", lambda: self.close_window(key))
        return win

    def close_window(self, key: str) -> None:
        win = self._windows.pop(key, None)
        if win is not None and win.winfo_exists():
            if hasattr(win, "on_close"):
                try:
                    win.on_close()
                except Exception:                       # noqa: BLE001
                    pass
            win.destroy()

    def recent_visualization(self) -> Path | None:
        """最近一个**真有** ``visualization.json`` 的会话目录（按目录名里的时间戳）。

        两个用途不同、别混：
        - 这里只回答"打开查看器时先给哪一个"（一个默认值，**窗口内随时可换**）；
        - 查看器内那个下拉由 ``apps/view_3d.scan_sources`` 提供（全部会话）。

        为什么用目录名而不是 mtime：跑一次离线会重写 ``derived/``，把"昨天录、
        今天补跑"的老会话 mtime 刷成今天 —— 按 mtime 排会把它顶成"最近"，
        默认值每次都漂。这个坑 ``apps/_workbench.scan_sessions`` 已经吃过一次。

        为什么不直接 import ``view_3d.scan_sources``：那个模块会拉起 matplotlib
        （TkAgg），门厅启动完全不需要它。宁可两处各写一遍这条**一行规则**，
        也不给启动路径加一个重量级依赖；规则写在两边注释里互指。
        """
        cands: list[tuple[str, Path]] = []
        for base in (ROOT / "data" / "sessions", ROOT / "data"):
            try:
                subs = [d for d in base.iterdir() if d.is_dir()]
            except OSError:
                continue
            for d in subs:
                for rel in ("derived/visualization.json", "visualization.json"):
                    f = d / rel
                    try:
                        if f.is_file():
                            m = re.match(r"(\d{8}-\d{6})", d.name)
                            cands.append((m.group(1) if m else "", f))
                            break
                    except OSError:
                        continue
        if not cands:
            return None
        cands.sort(key=lambda x: x[0], reverse=True)
        return cands[0][1]

    def open_console(self):
        from apps._gui_console import ConsoleWindow
        return self.open_window("console", "任务管理器和运行日志", ConsoleWindow)

    # ---------- 三张卡的入口（工厂集中在 App，窗口自己不知道别人） ----------
    def open_mono(self):
        from apps._gui_mono import MonoWindow
        return self.open_window("mono", "单目 · 球径先验测深", MonoWindow)

    def open_stereo(self, mode: str):
        from apps._gui_stereo import StereoWindow
        key = f"stereo_{mode}"
        title = ("平行双目 · 视差测深" if mode == "parallel"
                 else "大角度双目 · 三角测量")
        return self.open_window(key, title,
                                lambda app: StereoWindow(app, mode=mode))

    def open_bench(self, session=None):
        """打开会话工作台（①–⑤，单例）。

        已开着时**就地换会话**，不再开第二个窗口：工作台的对象是"一段会话"，
        开两个工作台就变成两份"当前会话"，用户又得猜哪个是真的。
        """
        from apps._gui_bench import BenchWindow
        win = self.open_window("bench", "会话工作台", BenchWindow)
        if session is not None and hasattr(win, "select_session"):
            win.select_session(session)
        return win

    # ---------- 退出 ----------
    def close_app(self) -> None:
        running = self.tasks.running()
        if running:
            from tkinter import messagebox
            n = len(running)
            if not messagebox.askyesno(
                    "还有任务在跑",
                    f"有 {n} 个任务在运行，其中可能有相机或标定程序。\n"
                    f"直接关闭会让它们留在后台（相机可能一直被占着）。\n\n"
                    f"现在中断它们并退出吗？",
                    parent=self.hub):
                return
            self.tasks.shutdown(graceful=True)
        else:
            self.tasks.shutdown(graceful=False)
        self.destroy()


# --------------------------------------------------------------------------- #
# 门厅
# --------------------------------------------------------------------------- #
class HubWindow(tk.Toplevel):
    """门厅：唯一入口。三张模式卡（适用条件 + 就绪度 + 进入）+ 工具条。

    为什么是"三张卡"而不是"一个模式下拉框"：模式不是一个可以随便切的显示
    选项，而是**三条几何前提不同的链路**。卡片能写前提，下拉框只能写名字。
    """

    def __init__(self, app: App) -> None:
        super().__init__(app)
        self.app = app
        self.configure(bg=BG)
        w, h = _fit_window(self)
        self.geometry(f"{w}x{h}")
        self.minsize(min(820, w), min(520, h))
        self.surface = Surface(self)

        self._build_brand()
        # 先占底部：pack 从容器边缘往里分配，先 pack 的优先拿到空间。
        # 反过来（先 pack 主内容区、再 pack 状态条）状态条会**盖住内容** ——
        # 2026-10-28 截图里第 4 步卡片就被盖掉了一半。
        self.status = StatusBar(self, on_log=app.open_console)
        self.status.pack(fill="x", side="bottom")
        page = ScrollPage(self, self.surface)
        page.pack(fill="both", expand=True)
        body = page.content
        self._build_mode_cards(body)
        self._build_toolbar(body)
        self._refresh()

    # ---------- 品牌条 ----------
    def _build_brand(self) -> None:
        head = tk.Frame(self, bg=ACCENT)
        head.pack(fill="x")
        tk.Label(head, text="球形物体三维运动轨迹识别", bg=ACCENT, fg="white",
                 font=(FONT_FAMILY, 17, "bold")).pack(
            anchor="w", padx=20, pady=(12, 0))
        tk.Label(head, text="先选装置类型，再按窗口里的三步走；参数只在用到它的那一步出现",
                 bg=ACCENT, fg="#dbeafe", font=(FONT_FAMILY, 10)).pack(
            anchor="w", padx=20, pady=(2, 12))

    # ---------- 三张模式卡 ----------
    def _build_mode_cards(self, parent) -> None:
        self._cards: dict[str, dict] = {}
        specs = (
            ("mono", "单目 · 一台相机",
             "深度靠球径先验 Z = f·D/(2r)。需要：一台相机、一次棋盘内参标定、"
             "已知球直径。0.6 m 处误差约 ±7 mm，比双目差一个量级 —— "
             "它解决的是「只有一台相机也能测」的问题，不是替代双目。",
             lambda: self.app.open_mono()),
            ("parallel", "平行双目 · 两路光轴近平行",
             "深度靠校正后的视差。需要：两路能同时看到同一块板（joint 标定）、"
             "基线远小于工作距离。0.6 m 处半径项误差约 ±0.35 mm。",
             lambda: self.app.open_stereo("parallel")),
            ("angled", "大角度双目 · 接近直角",
             "深度靠原始像素三角测量，两路不必平行。两条标定路：两路共视"
             "（joint 跳过 FOV），或两路看不到同一区域时用单板链式"
             "（板不动、各拍一组参考图）。链式产物就是本窗口要的 angled。",
             lambda: self.app.open_stereo("angled")),
        )
        for key, title, desc, opener in specs:
            box = self.surface.card(parent, title, desc)
            # 状态行用 surface.hint 建：它会把标签登记进换行表，视口变宽时
            # 一起重排。直接 ttk.Label + 写死 row=1 会与 action_row 撞行，
            # 也会在窄窗口里被截断（2026-10-28 截图就是这么发现的）。
            status_lab = self.surface.hint(box, "检查中…")
            actions = self.surface.action_row(box)
            # 每张卡只有一个主按钮：进入。工具类动作统一放下面工具条。
            self.surface.primary(actions, "进入", opener)
            self._cards[key] = {"status": status_lab}

    # ---------- 工具条 ----------
    def _build_toolbar(self, parent) -> None:
        box = self.surface.card(
            parent, "工具",
            "这些是跨模式的东西：会话与数据、三维查看、日志。"
            "它们不属于某一种装置，所以放在门厅而不是某个模式窗里。")
        row = self.surface.action_row(box)
        # 相机设备放第一个：没有相机身份，标定/录制都没有起点，它比"数据管理"更靠前
        self.surface.secondary(row, "相机设备", self._open_cameras)
        self.surface.secondary(row, "任务管理器和运行日志", self.app.open_console)
        self.surface.secondary(row, "数据管理", self._open_data_manager)
        self.surface.secondary(row, "3D 查看器", self._open_viewer)
        # 工作台在这里出现，是因为它的对象是"任意一段历史会话"，
        # 不属于某一种装置；当前会话的 ①–⑤ 在双目窗第 ④ 步就地做。
        self.surface.secondary(row, "会话工作台", self.app.open_bench)
        self.surface.secondary(row, "重新读取标定", self._reload_state)
        # 过渡入口：旧界面还没删完，明确标注并给出它将消失的时间点，
        # 而不是让用户自己猜"两个入口哪个是真的"。
        self.surface.secondary(row, "经典界面（过渡）", self._open_legacy)

    # ---------- 动作 ----------
    def _reload_state(self) -> None:
        self.app.state.reload()
        self._refresh()
        ok, msg = True, "已重新读取装置与标定状态"
        self.status.set_status(msg, "ok" if ok else "warn")

    def _open_cameras(self) -> None:
        """打开相机设备管理（识别 / 固定 / 标无关）。

        为什么复用 ``apps/tools/cameras.py`` 而不是在新界面里重画一个：
        它已经处理了"两台各自只在不同的后端下出图"这种真机状态
        （``probe`` 逐后端×逐编号实测），重画一遍只会分叉。
        ``identify`` 是交互式的，会自己开 cv2 窗口让人逐个确认。
        """
        self.app.tasks.start(
            [sys.executable, str(ROOT / "apps" / "tools" / "cameras.py"),
             "identify"],
            "相机设备管理")
        self.status.set_status(
            "相机设备管理已启动：会逐个索引显示画面，确认哪台是左、哪台是右。"
            "改完回到门厅点「重新读取标定」刷新显示", "info")

    def _open_data_manager(self) -> None:
        # 数据管理是独立进程（apps/data_manager.py），不嵌进本进程：
        # 它自己会开 Tk 窗口，嵌进来会变成第二个 Tk 根。
        self.app.tasks.start(
            [sys.executable, str(ROOT / "apps" / "data_manager.py")],
            "数据管理")

    def _open_viewer(self) -> None:
        """打开 3D 查看器 —— **必须带源**（view_3d 的 source 是必填参数）。"""
        src = self.app.recent_visualization()
        if src is None:
            # 没有源就**不启动**：盲启的结果是进程 exit 2、用户看到"没反应"
            self.status.set_status(
                "还没有可看的三维结果。先在某个会话上跑一次「离线三维」，"
                "它会写出 visualization.json，然后这里就能打开", "warn")
            return
        self.app.tasks.start(
            [sys.executable, str(ROOT / "apps" / "view_3d.py"), str(src)],
            "3D 查看器")
        self.status.set_status(
            f"正在打开 3D 查看器：{src.parent.name}"
            f"（窗口内「数据源」可换任意会话）", "info")

    def _open_legacy(self) -> None:
        from tkinter import messagebox
        messagebox.showinfo(
            "经典界面（过渡）",
            "经典界面是本轮重做前的单窗口版本，功能完整但很长。\n\n"
            "新界面的三个模式窗还在逐步补齐采集/标定/工作台，\n"
            "补齐之前可以用它做完整流程。它会在分窗完成后删除。",
            parent=self)
        self.app.tasks.start(
            [sys.executable, str(ROOT / "启动程序.py"), "--legacy"],
            "经典界面")

    # ---------- 刷新就绪度 ----------
    def _refresh(self) -> None:
        st = self.app.state
        mapping = {
            "mono": st.mono_status(),
            "parallel": st.parallel_status(),
            "angled": st.angled_status(),
        }
        for key, (ready, text) in mapping.items():
            lab = self._cards[key]["status"]
            # 卡片上的文案**不允许**出现 Markdown 标记（B-96 的教训），
            # 状态来源里若带 ** 就换成「」
            clean = text.replace("**", "「", 1)
            while "**" in clean:
                clean = clean.replace("**", "」", 1)
            lab.configure(text=("就绪：" if ready else "还不能用：") + clean,
                          style="Ok.TLabel" if ready else "Warn.TLabel")
        notes = "；".join(st.notes) if st.notes else ""
        prof = getattr(st.profile, "name", None)
        head = f"当前装置：{prof}" if prof else "未设置装置档案"
        if st.rig is not None:
            mode = st.stereo_mode() or "未记录模式"
            head += f"（标定模式 {mode}）"
        self.status.set_status(head + (f"｜{notes}" if notes else ""),
                               "warn" if notes else "info")


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def run() -> int:
    """启动新界面。返回进程退出码。"""
    app = App()
    app.mainloop()
    return 0
