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
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from apps._gui_widgets import (ACCENT, BG, FONT_FAMILY, INK, MUTED, OK, WARN,
                               ScrollPage, StatusBar, Surface, fit_to_content,
                               init_style)
from apps._task_runner import ROOT, TaskRunner
from apps import _actions
from stereo.profiles import (calibration_files_exist, calibration_identity,
                             IDENTITY_MISMATCH, IDENTITY_UNREGISTERED)

__all__ = ["GuiPrefs", "AppState", "App", "HubWindow", "run"]

MONO_CALIB_PATH = ROOT / "config" / "mono_calibration.json"
# 2026-11-10（用户裁决 B5）：不再有 `STEREO_CALIB_PATH` 常量。
# 标定路径一律走 `apps._common.current_calibration` —— 那个"档案优先、档案没配
# 才回落顶层 config/stereo_calibration"的判断**只许有一份**。这里原来自己写了一份，
# GUI 按档案 cand 判定、CLI 按顶层判定，于是用户把标定写进
# `config/calib/<档案名>.npz` 之后，CLI 复检报"找不到标定文件"。


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
        # warn 级之外的说明（如老标定「未登记来源」）：只拼进状态条，不升成警告。
        self.infos: list[str] = []
        self.reload()

    # ---------- 读取 ----------
    def reload(self) -> None:
        self.notes.clear()
        self.infos.clear()
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
        self._collect_identity()

    def _stereo_calib_path(self) -> Path:
        """当前该用的标定基名（无扩展名）：档案优先，档案没配才回落顶层。

        判定逻辑**不在这里**：``apps._common.current_calibration`` 是唯一产地
        （2026-11-10 用户裁决 B5）。``need_existing=False`` 是因为这里要的是
        "该用哪份"，文件在不在由 ``_read_rig()`` 单独判 —— 档案指定了但文件
        不在时**仍返回档案路径**，绝不回落到别套装置的几何。
        """
        from apps._common import current_calibration

        return current_calibration(need_existing=False)[0]

    def _read_rig(self):
        path = self._stereo_calib_path()
        # StereoRig.load 接受无扩展名的基名，会自己找 .json/.npz
        if not calibration_files_exist(path):
            prof = getattr(self.profile, "name", None)
            if prof:
                self.notes.append(
                    f"装置档案「{prof}」指定的标定文件不存在：{path.name}。"
                    "门厅按「未标定」显示；录制/重建链路的 "
                    "`resolve_calibration_base()` 会直接报错拦下，"
                    "不再拿别套装置的几何顶替 —— 先给这套装置标定")
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
        if isinstance(raw, dict):
            bound = str(raw.get("device_identity") or "")
            cur = str((self.camera_roles().get("left") or {}).get("identity") or "")
            if bound and cur and bound != cur:
                self.notes.append("单目标定绑定的是另一台相机，请重做单目内参标定")
            elif not bound:
                self.notes.append("单目标定未绑定设备身份（老文件），换相机不会自动报警；请重标")
        return raw if isinstance(raw, dict) else None

    # ---------- 身份自检（C2）与状态条文案（B3） ----------
    def _collect_identity(self) -> None:
        """C2：把标定身份三态分流到 notes(warn) / infos(info)，**不阻断使用**。

        与 ``mode_conflict()``（只查档案模式 vs 标定 meta.stereo_mode）互补：
        这里查的是 meta.profile —— C1 起新标定写入的装置档案名。三态：

        - matched       -> 不提示；
        - mismatch      -> 明确警告「这份标定属于档案 X，当前是 Y」进 ``notes``；
        - unregistered  -> 老标定没有 profile，只给一句「未登记来源」的
          **说明**进 ``infos``，绝不误报成「属于别套装置」，也不阻断。

        现有 4 个档案的标定都是老文件，若把 unregistered 也当警告，用户一进
        门厅就会看到一堆假警告。``infos`` 只拼进状态条、不升成 warn 级别，
        ``mode_allowed()`` / ``mode_conflict()`` 完全不看它，所以不拦任何窗口。
        """
        if self.rig is None:
            return
        meta = getattr(self.rig, "meta", None) or {}
        cur = getattr(self.profile, "name", None)
        verdict = calibration_identity(meta, cur)
        if verdict.state == IDENTITY_MISMATCH:
            self.notes.append(verdict.message)
        elif verdict.state == IDENTITY_UNREGISTERED:
            self.infos.append(verdict.message)
        # matched：不提示

    def status_text(self) -> tuple[str, str]:
        """门厅状态条文案（B3）：``_refresh()`` 里那套 head 计算的唯一产地。

        返回 ``(文案, level)``。``level``：有 warn 级 notes（读不了 / 身份不符）
        或「有档案却没标定」时 warn，否则 info。``infos`` 是「未登记来源」这类
        说明，只拼进文案、不升成 warn。抽成纯方法后，``_reload_state()`` 与
        切换档案都能复用它，且可在测试里用假 state 纯数据验证「文案随档案变」。
        """
        notes = "；".join(self.notes) if self.notes else ""
        infos = "；".join(self.infos) if self.infos else ""
        prof = getattr(self.profile, "name", None)
        head = f"当前装置：{prof}" if prof else "未设置装置档案"
        if self.rig is not None:
            mode = self.stereo_mode() or "未记录模式"
            head += f"（标定模式 {mode}）"
        parts = [p for p in (notes, infos) if p]
        text = head + (f"｜{'；'.join(parts)}" if parts else "")
        level = "warn" if (self.notes or (prof and self.rig is None)) else "info"
        return text, level

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
        """``left`` → ``left = Integrated IR Camera，编号 0``；未固定/不在线都直说。

        ⚠️ **2026-11-08 修：不许再打印配置里存的旧索引。**
        用户报「left 和 right 是同一个相机」，查下来就是这里：他换过 USB 口之后
        right 绑定的设备已经不在线，而配置里两条记录的 `index` 都是 **0**
        （换口前 right 恰好也排在 0），于是这一行显示成
        「left…索引 0｜right…索引 0」，看起来完全是同一台。

        现在按 **identity 在线的实际设备** 报：在线就报它当前的 PnP 编号
        （不是历史值），不在线就写「不在线」——比"索引 0"还短，不会把这一行撑宽。
        """
        info = self.camera_roles().get(role)
        if not isinstance(info, dict):
            return f"{role}：未固定（启动时自动挑）"
        ident = str(info.get("identity") or "").upper()
        port = str(info.get("port") or "").lower()
        hit = None
        try:
            from sphere_tracker import camera_registry as cr
            for d in cr.list_devices():
                if ident and d.identity and d.identity.upper() == ident:
                    hit = d
                    break
                if not ident and port and d.port and port in d.port.lower():
                    hit = d
                    break
        except Exception:                                     # noqa: BLE001
            hit = None
        bits = [f"{role} = {info.get('name') or '未知型号'}"]
        if hit is None:
            bits.append("不在线")
        else:
            bits.append(f"编号 {hit.order}")
            if not info.get("verified"):
                bits.append("未确认")
        # 采集索引是 `cv2.VideoCapture` 真正用的号，**未实测就写"未实测"**，
        # 不许拿 PnP 编号冒充（那正是"显示的相机 ≠ 真实相机"的来源）。
        cap = info.get("capture_index")
        bits.append(f"采集 #{cap}" if isinstance(cap, int) else "采集索引未实测")
        return "，".join(bits)

    def camera_role_short(self, role: str) -> str:
        """双目第①步那一行用的**短标签**：设备名，不显示已废弃的“位置”。

        为什么单独一个出口：完整 `camera_role_text()` 带 PnP 位次、采集索引，
        两路一拼会超过相机行的换行边界（实测大角度窗内容高出 16 px，
        `selftest_window_mode` 红）。身份/采集状态在「换相机…」对话框里看。
        """
        info = self.camera_roles().get(role)
        if not isinstance(info, dict):
            return f"{role}：未固定"
        name = str(info.get("name") or "").strip()
        return f"{role}={name}" if name else f"{role}：未固定"

    # ---------- 派生（只读，不写回） ----------
    def stereo_mode(self) -> str | None:
        """标定里记的模式。**唯一来源是标定 meta**，不是任何界面控件。"""
        meta = getattr(self.rig, "meta", None) or {}
        mode = str(meta.get("stereo_mode") or "").strip()
        return mode or None

    def profile_mode(self) -> str | None:
        """档案声明的装置模式（parallel / angled），与标定 meta 分开读。

        为什么要有第二个来源（2026-11-08）：用户切到平行档案后，大角度卡
        照样把那份平行标定的 5.1°/66mm 念出来，像是在"读到了可用结果"。
        档案说平行 = 大角度链路**根本不适用**，不该读、也不该展示那串数字。
        两个来源不一致时按"冲突"处理，不许静默挑一个信。
        """
        mode = str(getattr(self.profile, "stereo_mode", "") or "").strip()
        return mode or None

    def mode_conflict(self) -> str | None:
        """档案模式与标定 meta 不一致时，返回一句人话；一致/缺信息返回 None。"""
        pm, cm = self.profile_mode(), self.stereo_mode()
        if pm and cm and pm != cm:
            return (f"装置档案声明「{pm}」，标定文件却记为「{cm}」—— "
                    "这很可能是别套装置的标定，请给当前装置重新标定")
        return None

    def mode_allowed(self, key: str) -> bool:
        """这张双目卡**允不允许进**。

        只拦两种：**档案模式不适用**、**档案与标定冲突**。
        「没标定」不拦 —— 平行/大角度窗口里就有标定动作，把按钮禁了等于
        把用户锁在门外、永远没法给这套装置标定（2026-11-08 自查发现）。
        """
        if key not in ("parallel", "angled"):
            return True
        if self.mode_conflict() is not None:
            return False
        pm = self.profile_mode()
        if key == "parallel":
            return pm != "angled"
        return pm != "parallel"

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
        """单目：只看全局 ``mono_calibration.json``，与双目装置档案**分开管理**。

        2026-11-08 用户："无论是哪个档案，单目都显示有标定，说明一直用着之前的
        标定结果"。这是真的，但单目内参本来就只跟**这一台相机 + 这个焦距**有关，
        跟双目装置（基线/夹角）无关 —— 所以正确做法不是把它塞进档案，
        而是把"全局、独立"这件事**写在卡片上**，别让人以为是档案没生效。
        """
        tag = "（单目内参：全局独立，与双目装置档案无关）"
        try:
            import datetime as _dt

            when = _dt.datetime.fromtimestamp(
                MONO_CALIB_PATH.stat().st_mtime).strftime("%Y-%m-%d")
            cam = self.mono.get("camera") if self.mono else None
            tag = (f"（全局单目内参，与双目档案无关；"
                   f"{when}，camera={cam or '未记录'}）")
        except (OSError, ValueError):
            pass
        if not self.mono:
            return False, ("未标定。现在会按 60° 视场估算焦距 —— "
                           "1% 焦距误差在 0.6 m 处就是 6 mm，请先做内参标定" + tag)
        fx = self.mono.get("fx")
        if not isinstance(fx, (int, float)) or fx <= 0:
            return False, "标定文件存在但 fx 无效，请重新标定" + tag
        rms = self.mono.get("rms_px")
        extra = f"，RMS {rms:.2f} px" if isinstance(rms, (int, float)) else ""
        bound = str(self.mono.get("device_identity") or "")
        cur = str((self.camera_roles().get("left") or {}).get("identity") or "")
        if bound and cur and bound != cur:
            return False, "单目标定属于另一台相机，请重做单目内参标定" + tag
        # 「未绑定」/「已核对」写进 notes（状态条），不塞进卡片：卡片文案每多
        # 一行都会把 mono 第③步顶出 13px（selftest_window_mode 直接盯着下沿）。
        return True, f"已标定 fx = {float(fx):.0f} px{extra}{tag}"

    def parallel_status(self) -> tuple[bool, str]:
        """平行双目：档案模式优先，再看标定存在与夹角。"""
        pm = self.profile_mode()
        conflict = self.mode_conflict()
        if conflict:
            return False, conflict
        if pm == "angled":
            return False, ("当前装置档案是「大角度」—— 平行双目不适用；"
                           "请切到平行装置档案（本卡不读取大角度标定）")
        if self.rig is None:
            return False, "没有双目标定。平行双目需要两路光轴近平行并做校正"
        conflict = self.mode_conflict()
        if conflict:
            return False, conflict
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
        pm = self.profile_mode()
        conflict = self.mode_conflict()
        if conflict:
            return False, conflict
        if pm == "parallel":
            return False, ("当前装置档案是「平行」—— 大角度双目不适用；"
                           "请切到大角度装置档案并给它单独标定（本卡不读取平行标定）")
        if self.rig is None:
            return False, "没有双目标定。大角度可用「两路共视」或「单板链式」两条路"
        conflict = self.mode_conflict()
        if conflict:
            return False, conflict
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
def work_area(root: tk.Misc) -> tuple[int, int]:
    """屏幕**可用工作区**（扣除任务栏）。返回 (w, h)。

    为什么单独成一个函数：门厅/模式窗/工作台的"窗口能多高"都要同一个数。
    各自 ``winfo_screenheight()`` 会连任务栏一起算进去，于是按内容算出的高度
    正好把自己压在任务栏下面 —— 用户看到的是"窗口最下面一截被挡住"，
    与写死一个太大常数是同一个后果。
    """
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
    return screen_w, screen_h


def _fit_window(root: tk.Misc) -> tuple[int, int]:
    """给主窗口一个"装得下、不压任务栏"的尺寸。返回 (w, h)。"""
    screen_w, screen_h = work_area(root)
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

        # 任何 Tk 回调里的异常都要留下痕迹：`pythonw.exe` 没有 stderr，
        # 默认行为是"静静地什么都不显示"——界面就此卡住而用户看不到原因。
        # 这个坑让 2026-11-08 的相机预览问题反复误判了好几轮。
        self.report_callback_exception = self._log_callback_exception

        self.state = AppState()
        self.prefs = GuiPrefs()
        self._work: tuple[int, int] | None = None
        self._log_lines: list[str] = []
        self._log_listeners: list = []
        self.tasks = TaskRunner(self, on_log=self._on_task_log,
                                on_change=self._on_task_change)
        self._windows: dict[str, tk.Toplevel] = {}
        #: 正在关闭中的 key。关窗期间 on_close / 回调若再触发 open_window(key)，
        #: 会拿到"还在表里但即将被销毁"的旧窗口去 deiconify —— 或者更糟，
        #: 新建第二个窗口。这个集合让 close_window 对整个流程**不可重入**。
        self._closing: set[str] = set()

        # 门厅不属于 _windows（它常驻、走 close_app 退出），但它也会起任务
        # （数据管理 / 查看器）。给一个归属键，让那些调用点能写
        # owner=self.task_owner，与模式窗的口径一致。
        self.hub = HubWindow(self)
        self.hub.task_owner = "hub"
        self.hub.protocol("WM_DELETE_WINDOW", self.close_app)
        self.hub.focus_force()

    # ---------- 屏幕可用高度（各窗口按内容算高度时的上限来源） ----------
    def work_area(self) -> tuple[int, int]:
        """可用工作区 (w, h)。进程内不变，读一次记住。"""
        if self._work is None:
            self._work = work_area(self)
        return self._work

    # ---------- 界面异常留痕 ----------
    def _log_callback_exception(self, exc_type, exc_value, tb) -> None:
        """Tk 回调里的任何异常都写进 `outputs/logs/gui_errors.log`。

        为什么必须做（2026-11-08）：`pythonw.exe` 没有 stderr，Tk 默认把回调
        异常打印到一个**没人看得见**的地方。一个 `after` 回调抛一次，整条刷新
        链就断，界面从此不再更新 —— 用户看到的是"点了没反应/一直正在打开"，
        而日志里干干净净。把异常落到文件，是让这类问题可诊断的最低成本。
        """
        import traceback
        from pathlib import Path

        text = "".join(traceback.format_exception(exc_type, exc_value, tb))
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            root = Path(__file__).resolve().parent.parent
            folder = root / "outputs" / "logs"
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / "gui_errors.log").open("a", encoding="utf-8") as fh:
                fh.write(f"\n[{stamp}] 界面回调异常（已捕获，程序继续）\n{text}\n")
        except OSError:
            pass
        # 状态条也提一句：不打断操作，但让人知道"出过错、日志在哪"。
        try:
            self.hub.status.set_status(
                "界面出过一次错（已自动恢复）：详情见 outputs/logs/gui_errors.log",
                "warn")
        except Exception:                                # noqa: BLE001
            pass

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
        # 关窗流程中（on_close / 统一结束回调里）若再有人开同 key 的窗，
        # 必须直接交还那个"即将销毁"的旧窗，不能再 new 一个 —— 旧窗随后
        # 被 destroy，新窗留在表里就成了按 X 关不掉的幽灵窗口。
        if key in self._closing:
            return self._windows.get(key)
        existing = self._windows.get(key)
        if existing is not None and existing.winfo_exists():
            existing.deiconify()
            existing.lift()
            existing.focus_force()
            return existing
        win = factory(self)
        win.title(title)
        # 统一归属：窗口关掉时 close_window 按它停自己起的任务。
        # stereo 的 key 是 stereo_parallel / stereo_angled，start 时用这个
        # 全名正好对上 close_window 的 key（不需要再按前缀猜）。
        win.task_owner = key
        self._windows[key] = win
        win.protocol("WM_DELETE_WINDOW", lambda: self.close_window(key))
        return win

    def broadcast_state_refresh(self) -> None:
        """装置档案 / 标定状态变了：通知**已打开的** mono / stereo 窗只读重绘。

        B4：``open_window`` 是单例，已开的窗口只 ``deiconify/lift``，不会重跑
        它们的 ``_refresh_state()``。于是切档案或「重新读取标定」后，双目窗顶部
        的「装置档案：X｜标定模式：Y」（``_gui_stereo.py``）、单目窗的内参状态
        （``_gui_mono.py``）都停在旧值。这里遍历 ``self._windows``，只对带
        ``_refresh_state`` 的窗口调用它 —— 不新建窗口、不重建控件、不给门厅
        卡片加行（``selftest_window_mode`` 盯着 13px 下沿）。单个窗口抛异常
        不影响其他窗口，也不打断门厅。
        """
        for _key, win in list(self._windows.items()):
            try:
                if win.winfo_exists() and hasattr(win, "_refresh_state"):
                    win._refresh_state()
            except Exception:                       # noqa: BLE001
                pass

    def close_window(self, key: str) -> None:
        """关一个子窗：先停本窗任务，再销毁，**最后**才从表里移除。

        为什么 ``pop`` 必须在 ``destroy`` 之后：旧实现是
        ``pop → on_close() → destroy()``。在 on_close 与 destroy 之间这个 key
        已经不在表里，此时任何回调调 ``open_window(key)`` 都会**新建一个窗**；
        紧接着的 ``destroy()`` 销毁的却只是旧窗，新窗留在表里 —— 用户按了 X，
        看到的却是一个"关不掉"的窗口（这正是"复苏"的结构性根因）。
        """
        win = self._windows.get(key)
        if win is None or not win.winfo_exists():
            return
        if key in self._closing:                       # 回调重入：正在关就别再关一次
            return
        self._closing.add(key)
        try:
            if hasattr(win, "on_close"):
                try:
                    win.on_close()
                except Exception:                       # noqa: BLE001
                    pass
            # 关窗口 = 停这个窗口起的任务。console 是任务管理器本身，
            # 收起它不该杀任务（明确例外）。
            if key != "console":
                self.tasks.stop_by_owner(key)
            if win.winfo_exists():
                win.destroy()
            self._windows.pop(key, None)               # ← 必须在 destroy 之后
        finally:
            self._closing.discard(key)

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
        # 延迟 import：`sphere_tracker` 的包初始化会拉起 cv2/numpy，而门厅启动不需要
        # （本项目已有先例：不为了一个默认值给启动路径加重量级依赖）。
        #
        # 枚举走 `_stereo_layout.candidate_session_dirs`：会话按装置分区后
        # （``data/stereo/parallel/<时间戳>``）只扫一层会把分区目录当成会话，
        # 于是"默认打开哪一段"永远指向 parallel/ 这种目录、直接找不到 visualization。
        #
        # 只给 `stereo_roots()`，**不**再带上 `data/` 根（2026-11-09 实测）：
        # 枚举改成递归 + 只认会话目录之后，传 `data/` 会真的走进 `data/mono/`，
        # 而 `data/mono/20261029-205522/derived/visualization.json` 确实存在 ——
        # 那会让"默认打开"变成一段**单目**会话（回归）。旧扁平布局
        # （`data/<时间戳>`）由 `data/sessions`（LEGACY_STEREO_ROOT）兜着，
        # 它已经在 `stereo_roots()` 里；`data/` 是以前"只认两层"时才顺手加的。
        from sphere_tracker.paths import stereo_roots

        from apps._stereo_layout import candidate_session_dirs

        for d in candidate_session_dirs([*stereo_roots()]):
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
                    f"是 = 中断它们并退出；否 = 保持窗口打开、任务继续运行。\n\n"
                    f"如果你只是想收起某个子窗口，请关那个子窗口，不要关这里。",
                    parent=self.hub):
                # 点"否"不是无声退出：留一行，让"我按了 X 可窗口还在"能查到原因
                _log_launch_context("退出被取消：还有任务在跑（用户选否）")
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
        self.surface = Surface(self)

        self.brand = self._build_brand()
        # 先占底部：pack 从容器边缘往里分配，先 pack 的优先拿到空间。
        # 反过来（先 pack 主内容区、再 pack 状态条）状态条会**盖住内容** ——
        # 2026-10-28 截图里第 4 步卡片就被盖掉了一半。
        self.status = StatusBar(self, on_log=app.open_console,
                                help_section="start")
        self.status.pack(fill="x", side="bottom")
        self._page = ScrollPage(self, self.surface)
        self._page.pack(fill="both", expand=True)
        body = self._page.content
        self._build_mode_cards(body)
        self._build_toolbar(body)
        self._refresh()
        # 高度也按内容算（2026-10-31）：原来用 `_fit_window` 的 760，内容要 768，
        # 于是门厅永远差 8 px —— 不多，但"一屏看全"就成了一句假话。
        # `extra` 给的是顶部那条品牌栏：它不在滚动页里，但同样占客户区。
        # 必须先 `update_idletasks` 再取它的高度 —— 否则量到的是 1（还没排版），
        # 窗口照样矮 87 px。
        self.update_idletasks()
        w, _h = _fit_window(self)
        height = fit_to_content(self, self._page, self.status, width=w,
                                screen_h=app.work_area()[1],
                                extra=self.brand.winfo_reqheight())
        self.minsize(min(820, w), min(520, height))

    # ---------- 品牌条 ----------
    def _build_brand(self) -> tk.Frame:
        head = tk.Frame(self, bg=ACCENT)
        head.pack(fill="x")

        # 语言切换固定在右侧。pack 的分配顺序就是代码顺序：右侧这块必须先占住，
        # 左侧两行标题才能拿剩下的宽度；反过来会被挤成 0 宽。
        from apps import i18n
        right = tk.Frame(head, bg=ACCENT)
        right.pack(side="right", padx=(0, 20), pady=12)
        tk.Label(right, text="Lang", bg=ACCENT, fg="white",
                 font=(FONT_FAMILY, 10)).pack(side="left")
        lang_combo = ttk.Combobox(
            right, state="readonly", width=9,
            values=[i18n.label(code) for code in i18n.LANGS])
        lang_combo.set(i18n.label())
        lang_combo.pack(side="left", padx=(6, 0))

        def _on_language(_event=None) -> None:
            idx = lang_combo.current()
            if 0 <= idx < len(i18n.LANGS):
                i18n.set_language(i18n.LANGS[idx])
                self.status.set_status("界面语言已切换为 " + i18n.label(), "ok")

        lang_combo.bind("<<ComboboxSelected>>", _on_language)

        left = tk.Frame(head, bg=ACCENT)
        left.pack(side="left", fill="both", expand=True)
        tk.Label(left, text="球形物体三维运动轨迹识别", bg=ACCENT, fg="white",
                 font=(FONT_FAMILY, 17, "bold")).pack(
            anchor="w", padx=20, pady=(12, 0))
        tk.Label(left, text="先选装置类型，再按窗口里的三步走；参数只在用到它的那一步出现",
                 bg=ACCENT, fg="#dbeafe", font=(FONT_FAMILY, 10)).pack(
            anchor="w", padx=20, pady=(2, 12))
        return head

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
             lambda: self._enter_mode("parallel")),
            ("angled", "大角度双目 · 接近直角",
             "深度靠原始像素三角测量，两路不必平行。两条标定路：两路共视"
             "（joint 跳过 FOV），或两路看不到同一区域时用单板链式"
             "（板不动、各拍一组参考图）。链式产物就是本窗口要的 angled。",
             lambda: self._enter_mode("angled")),
        )
        for key, title, desc, opener in specs:
            box = self.surface.card(parent, title, desc)
            # 状态行用 surface.hint 建：它会把标签登记进换行表，视口变宽时
            # 一起重排。直接 ttk.Label + 写死 row=1 会与 action_row 撞行，
            # 也会在窄窗口里被截断（2026-10-28 截图就是这么发现的）。
            status_lab = self.surface.hint(box, "检查中…")
            actions = self.surface.action_row(box)
            # 每张卡只有一个主按钮：进入。工具类动作统一放下面工具条。
            btn = self.surface.primary(actions, "进入", opener)
            self._cards[key] = {"status": status_lab, "enter": btn}

    def _enter_mode(self, key: str) -> None:
        """双目两张卡的「进入」闸门（2026-11-08）。

        只在卡片上写"还不能用"是**假限制**：按钮照旧能点，点进去照样拿别套
        装置的标定跑。所以按钮本身禁用，且这里再按**当下状态**复核一次
        （档案可能在窗口开着的时候被切走）。单目不设这道闸：它的内参是全局
        独立的，跟装置档案无关，进去本来就是为了做单目标定。
        """
        st = self.app.state
        if not st.mode_allowed(key):
            _ok, msg = (st.parallel_status() if key == "parallel"
                        else st.angled_status())
            self.status.set_status(f"不能进入「{key}」：{msg}", "warn")
            return
        # 「没有标定」不是拦路的理由：进去才有标定按钮。
        self.app.open_stereo(key)

    # ---------- 工具条 ----------
    def _build_toolbar(self, parent) -> None:
        box = self.surface.card(
            parent, "工具",
            "这些是跨模式的东西：会话与数据、三维查看、磁盘清理、标定与训练工具。"
            "它们不属于某一种装置，所以放门厅；任务与日志在右下角状态条。")
        # 2026-11-06：拆成两行。加「操作说明」之后这一行变成 9 个按钮，
        # 980 px 窗口里最右边的「经典界面（已废止）」被卡片边缘切掉一截
        # （走查截图 01_hub.png 可见）。action_row 是 shrink-to-fit 的 pack
        # 容器，不会自动换行 —— 不拆就只能删按钮或缩字号，那两条都更差。
        row = self.surface.action_row(box)
        # 相机设备放第一个：没有相机身份，标定/录制都没有起点，它比"数据管理"更靠前
        # 「操作说明」放第一个：它是**学习入口**，第一次用的人先看它，
        # 之后各窗口里那些挤占排版的说明文字都可以删掉挪进来（2026-11-05 用户）。
        self.surface.secondary(row, "操作说明", self._open_help)
        self.surface.secondary(row, "相机设备", self._open_cameras)
        # 「任务管理器和运行日志」**不在这里**（2026-11-04）：
        # 每个窗口右下角状态条已经有同一个入口（StatusBar 的按钮），
        # 门厅再放一个是同一个动作两个按钮，用户会以为是两件事。
        self.surface.secondary(row, "数据管理", self._open_data_manager)
        self.surface.secondary(row, "3D 查看器", self._open_viewer)
        self.surface.secondary(row, "垃圾清理", self._open_junk)

        row2 = self.surface.action_row(box)
        # 「会话工作台」入口已隐藏（2026-11-03）：它和单目/双目窗各自的
        # 「历史会话…」选择器功能重叠，属于旧架构遗留。BenchWindow 代码保留
        # （双目/单目窗的「历史会话…」仍能用），只是不再从门厅暴露这个跨装置的入口。
        self.surface.secondary(row2, "装置档案", self._open_profiles)
        self.surface.secondary(row2, "FOV 管理器", self._open_fov)
        self.surface.secondary(row2, "重新读取标定", self._reload_state)
        # 六条「只能敲命令」的离线流程收在一个对话框里（2026-11-07 盘点后补）。
        # 为什么不摊成六个按钮：它们与上面那排「每天都在点」的入口节奏不同，
        # 摊平会变成一排同级按钮；而且门厅当前只剩 19 px 高度余量，加不了第三行。
        self.surface.secondary(row2, "标定与训练…", self._open_tools)
        # 废止入口（2026-11-20）：旧界面的实现还在（多个回归测试按文本/导入引用它），
        # 但**不再能从这里启动**。按钮保留，只弹一句说明 —— 直接删按钮会让老用户
        # 以为"功能没了"，留一个只解释"去哪了"的入口更诚实。
        self.surface.secondary(row2, "经典界面（已废止）", self._open_legacy)
        # 语言切换在顶部品牌栏，不塞进这一行：action_row 是 shrink-to-fit 的
        # pack 容器，再加一块会把最右边挤出卡片。

    # ---------- 动作 ----------
    def _reload_state(self) -> None:
        self.app.state.reload()
        # B3：状态条复用 _refresh() 按当前装置算出的 head（当前装置：X（标定模式 Y）｜notes），
        # 不再用写死的「已重新读取装置与标定状态」覆盖 —— 那句永远不随档案变化，
        # 正是用户报的「切了装置再按重新读取、提示词没变」的主因。level 也跟着
        # 真实状态走（有档案却没标定 / 身份不符 -> warn）。
        head, level = self._refresh()
        self.status.set_status(head, level)
        # B4：通知已打开的 mono / stereo 窗只读重绘，别让它们的档案/标定标签停在旧值。
        self.app.broadcast_state_refresh()

    def _open_help(self) -> None:
        """打开操作说明（分区目录 + 搜索）。

        非模态：用户可以一边看说明一边操作主窗口 —— 照着做是最常见的用法。
        """
        from apps.help_window import open_help

        open_help(self, section="start")
        self.status.set_status("操作说明已打开：左栏选分区，顶部可搜索关键词", "info")

    def _open_junk(self) -> None:
        """垃圾管理器：日志 / 临时产物 / 字节码 / 回收站，按项勾选 + 按时间筛。"""
        from apps.junk_dialog import JunkDialog

        # 非模态窗口要自己挡重复打开：点几次就叠几个一模一样的窗口。
        _top = self.winfo_toplevel()
        _old = getattr(_top, "_junk_window", None)
        if _old is not None and _old.winfo_exists():
            _old.deiconify()
            _old.lift()
            _old.focus_force()
            self.status.set_status("垃圾管理器已经开着了", "info")
            return
        # 传 prefs：让「标为不动」的选择能记住（存在 config/gui_state.json）
        _top._junk_window = JunkDialog(self, prefs=self.app.prefs)
        self.status.set_status(
            "垃圾管理器已打开：只清它列出的项，data 里的录像不会被删", "info")

    def _open_profiles(self) -> None:
        """打开装置档案管理（列出 + 切换当前装置）。

        切换装置后要**重读状态**：门厅三张卡的就绪判定、每个模式窗顶部的
        "装置档案：xxx"都依赖 active profile，不重读就还是旧装置的结论。
        """
        from apps.profile_dialog import ProfileDialog

        def changed() -> None:
            self.app.state.reload()
            # B3：切换档案后状态条也复用 _refresh() 的 head（当前装置：新档案名），
            # 而不是一句写死的「已切换装置档案…」—— 用户要的是提示词随档案变化。
            head, level = self._refresh()
            self.status.set_status(head, level)
            # B4：已开的 mono / stereo 窗同步重绘。
            self.app.broadcast_state_refresh()

        # parent 传 **self（门厅，显示中的 Toplevel）**，不是 withdraw 的 self.app：
        # Toplevel 以 withdraw 的根为 parent 会继承 withdraw 状态，弹窗不出来。
        ProfileDialog(self, on_changed=changed)

    def _open_fov(self) -> None:
        """打开 FOV 管理器（两路固有视场差：看 / 量 / 清）。

        为什么单独一个入口而不是塞进装置档案：FOV 是**要量的**（拿一段录像
        取同一帧估一次），而装置档案是**填的**。两者的操作节奏不同，混在一个
        对话框里会让"切换装置"这个高频动作多背一堆用不上的控件。

        档案变了要重读状态：门厅/窗口的标定判定会跟着 active profile 走。
        """
        from apps.fov_dialog import FovDialog

        def changed() -> None:
            self.app.state.reload()
            self._refresh()
            # FOV 管理器里改了装置级应用开关，已开着的双目窗也要跟着同步
            # （遍历各窗口的 _refresh_state；方法名 broadcast_state_refresh）。
            self.app.broadcast_state_refresh()

        # parent 传 self（门厅，显示中的 Toplevel），理由同 `_open_profiles`。
        FovDialog(self, self.app, on_changed=changed)

    def _open_tools(self) -> None:
        """标定与训练工具台：生成标定板 / 取色 / 世界系 / 导出 / 数据集 / 训练。

        这六条都是「跑一次、隔很久再跑」的离线流程（2026-11-07 盘点出来的
        GUI 缺口：它们在 README 里被写成"下一步做什么"，界面上却只有命令行
        一条路）。收在一个对话框里而不是摊成六个按钮：节奏与上面那排
        "每天都在点"的入口不同，而且门厅只剩 19 px 高度余量，加不了第三行。
        """
        from apps.tools_dialog import ToolsDialog

        # 非模态窗口要自己挡重复打开：点几次就叠几个一模一样的窗口。
        _top = self.winfo_toplevel()
        _old = getattr(_top, "_tools_window", None)
        if _old is not None and _old.winfo_exists():
            _old.deiconify()
            _old.lift()
            _old.focus_force()
            self.status.set_status("标定与训练工具已经开着了", "info")
            return
        _top._tools_window = ToolsDialog(self, self.app)
        self.status.set_status(
            "标定与训练工具已打开：六条离线流程都在里面 —— "
            "标定板、点选取色、世界坐标系、导出标定、导出数据集、训练检测器", "info")

    def _open_cameras(self) -> None:
        """打开相机管理对话框（GUI 内，2026-11-07 改）。

        以前这里起一个控制台子进程去逐台开相机让人按键，用户两次报
        「平行双目相机管理打不开」。根因不在超时文案，而在那条路本身：
        没有控制台、没有按键、OpenCV 窗口不归 Tk 管、4 台各等 30 秒。
        现在改成 Tk 自己的对话框：枚举/抓帧在后台线程，绑角色按身份直接写
        ``config/cameras.json``，预览只开选中的一台、6 秒没画面就明说。
        """
        from apps.cameras_dialog import open_cameras

        # 单例在 `open_cameras()` 里统一做。原来这里查 `self.app.winfo_children()`
        # 是**无效检查**：对话框的 master 是模式窗，不是 app，查不到，
        # 于是三个入口能同时开出三个对话框、各自抢相机（2026-11-08 复盘）。
        open_cameras(self)
        self.status.set_status(
            "相机管理已打开：选一台点「预览这台」看画面，或直接点"
            "「设为左 / 设为右」按设备身份绑定（不用开相机，秒级）", "info")

    def _open_data_manager(self) -> None:
        # 数据管理是独立进程（apps/data_manager.py），不嵌进本进程：
        # 它自己会开 Tk 窗口，嵌进来会变成第二个 Tk 根。
        #
        # 状态条必须说一句（2026-10-31 走查缺陷 #2）：它是个独立进程，
        # 起得慢或起来就退出时，按钮"看起来什么都没发生"——
        # 同一排的「相机设备」「3D 查看器」都写了状态，只有这个没写。
        # 只说"看哪里"，**不去读任务对象**：状态条是给用户看的，
        # 不该因为拿不到 `task.log_path` 就整条不写。
        self.app.tasks.start(
            _actions.data_manager(),
            "数据管理", owner=self.task_owner)
        self.status.set_status(
            "已启动数据管理（它是独立窗口，首次启动要几秒）。"
            "没看到窗口就点「任务管理器和运行日志」——那里有这条任务的输出；"
            "起来了但不知道数据在哪，那个窗口自己会列 data/ 下的会话",
            "info")

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
            _actions.view_3d(src),
            "3D 查看器", owner=self.task_owner)
        # 会话名，不是 `src.parent.name` —— 离线结果在 `<会话>/derived/` 里，
        # `parent.name` 永远是 derived（与双目窗同一处口径，见 `session_name_of`）。
        from apps._gui_stereo import session_name_of
        self.status.set_status(
            f"正在打开 3D 查看器：{session_name_of(src)}"
            f"（窗口内「数据源」可换任意会话）", "info")

    def _open_legacy(self) -> None:
        """经典界面已废止：只说明去向，**不启动任何进程**。

        2026-11-20 用户裁决：新界面已覆盖完整流程，经典单窗口界面废止。
        实现保留在 `apps/gui_main.py`（入口一致性 / 启动器参数 / 任务管理器 /
        变异自证等测试仍按文本与导入引用它），但 `--legacy` 已关闭，
        `run.bat legacy` 也只打印提示。这里不调 `tasks.start(...)`：
        按钮存在的唯一意义是告诉用户功能搬去哪了，不是留一条后门。
        """
        from tkinter import messagebox
        messagebox.showinfo(
            "经典界面（已废止）",
            "经典界面已废止，不再启动。\n\n"
            "所有功能都在新界面里：门厅 → 单目 / 平行双目 / 大角度双目；\n"
            "数据管理、3D 查看器、相机设备在门厅下方工具条；\n"
            "标定板、取色、世界系、导出与训练在「标定与训练…」里。\n\n"
            "命令行 `run.bat legacy` 与 `--legacy` 也已关闭，只会打印这条说明。",
            parent=self)

    # ---------- 刷新就绪度 ----------
    def _refresh(self) -> tuple[str, str]:
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
            if key in ("parallel", "angled"):
                allowed = st.mode_allowed(key)
                self._cards[key]["enter"].configure(
                    state="normal" if allowed else "disabled")
        # 状态条文案与 head 计算收敛到 AppState.status_text()（B3 唯一产地），
        # _refresh 只负责把它显示出来，并把 (文案, level) 返回给 _reload_state 复用。
        text, level = st.status_text()
        self.status.set_status(text, level)
        return text, level


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
#: 单实例锁的名字。**整个 Windows 会话只允许一个**。
#:
#: 2026-11-08：用户双击了几次，机器上同时跑着两个「球形物体三维运动轨迹识别」——
#: 每个实例都能开相机管理、都能开预览，于是同一台相机被两三个进程抢，
#: 全部报 `-2147483638`，拔插也没用（插回来立刻被另一个实例抢走）。
#: 这不是相机的错，是**程序允许多开**。锁在这里，第二个实例激活第一个窗口后退出。
SINGLE_INSTANCE_NAME = "SphereTracker.GUI.v1"


#: 单实例锁文件（`outputs/logs/` 下：与 `launch.log` 同处，运行期不会被清理工具动）。
_LOCK_FILE = ".gui.lock"
#: 持有者的 PID，单独一个文件：**锁文件不能读**（见 `_claim_single_instance`）。
_PID_FILE = ".gui.pid"
#: 必须活到进程结束：fd 一关，文件锁就释放，锁也就没了。
_INSTANCE_LOCK = {"fd": None, "path": None}


def _raise_existing_window() -> None:
    """把已经在跑的那个主窗口叫到前面（找不到就什么也不做）。

    不能只用 ``FindWindowW(精确标题)``：旧实例可能被最小化、标题被对话框临时
    改掉，或者窗口还在但不在前台。2026-11-23 实测 launcher 第二次启动时只弹
    “已经在运行”，用户看不到旧窗口，自然以为新启动器打不开相机。
    这里优先按 ``.gui.pid`` 的进程号枚举该进程的所有顶层窗口，再回退到标题。
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        user32.FindWindowW.restype = wintypes.HWND
        pid = 0
        try:
            pid = int(_pid_path().read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            pid = 0

        handles: list[int] = []
        if pid:
            @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
            def _enum(hwnd, _lparam):
                owner = wintypes.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
                if owner.value == pid:
                    handles.append(int(hwnd))
                return True

            user32.EnumWindows(_enum, 0)

        for hwnd in handles:
            user32.ShowWindow(hwnd, 9)          # SW_RESTORE
            user32.SetForegroundWindow(hwnd)
            return

        hwnd = user32.FindWindowW(None, "球形物体三维运动轨迹识别")
        if hwnd:
            user32.ShowWindow(hwnd, 9)
            user32.SetForegroundWindow(hwnd)
    except Exception:                            # noqa: BLE001
        pass


def _lock_dir():
    from pathlib import Path

    d = Path(__file__).resolve().parents[1] / "outputs" / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _lock_path():
    return _lock_dir() / _LOCK_FILE


def _pid_path():
    return _lock_dir() / _PID_FILE


def _claim_single_instance() -> bool:
    """拿到单实例锁返回 True；已有实例返回 False（并把它的窗口叫到前面）。

    **为什么从"绑端口"换成"锁文件"（2026-11-08 14:45，用户第五次报"打不开"）。**
    绑端口的前提是"要么我绑上、要么别人绑着"，今天实测这个前提**不成立**：
    47821 在没有任何 python 进程、`netstat` 也没有监听的情况下，`bind` 连续报
    `10048 ADDRINUSE`，带 `SO_REUSEADDR` 又变成 `10013`，PowerShell 用
    `TcpListener` 复核同样失败 —— 几分钟后又自己好了。
    也就是说**端口会处于一种"绑不上但不是我们的实例"的状态**；
    旧代码把它一律当成"已经在运行"，于是 GUI 永远起不来、还报"已经开了一个"。
    这又是一次静默失败（用户看到的就是"双击没反应，自动退出"）。

    `msvcrt.locking` 是内核文件锁：不碰网络栈、进程一死内核就释放。

    **PID 另写一个文件**（2026-11-08 实测）：被 `LockFile` 锁住的文件，
    别的进程连 `open()` 都会拿到 `PermissionError 13` —— 连"只读前 12 字节"
    也不行。所以锁文件里什么都不放，PID 放 `.gui.pid`。
    """
    import os

    if sys.platform != "win32":                  # 非 Windows 不做单实例（本程序只发 Windows）
        return True
    import msvcrt

    path = _lock_path()
    fd = os.open(path, os.O_CREAT | os.O_RDWR)
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)   # 非阻塞：拿不到就抛 OSError
    except OSError:
        os.close(fd)
        _raise_existing_window()
        # 提示语统一由 `_explain_already_running()` 出（2026-11-08）：
        # 两处各打一遍会让用户看到重复的两段话。这里只负责把窗口叫到前面。
        return False
    try:
        _pid_path().write_text(str(os.getpid()), encoding="ascii")
    except OSError:
        pass                                     # 写不进去也要锁住，PID 只是诊断用
    _INSTANCE_LOCK.update(fd=fd, path=str(path))
    return True


def _log_launch_context(note: str = "") -> None:
    """把「这次是谁、怎么把我起来的」写进 `outputs/logs/launch.log`。

    为什么需要（2026-11-08）：同一台机器、同一份代码，用户实测
    **`run.bat` 能开相机预览、`launcher.exe` 不能**，前后查了 8 轮都扎在相机侧。
    入口差异必须能自己留证据：解释器是谁、有没有控制台、stdio 是否有效、
    父进程是谁、**是不是被单实例锁挡掉**（被挡掉时用户看到的是上一个旧窗口，
    于是"新启动器也打不开"——这本身就是一种可能的解释）。
    """
    try:
        import ctypes
        import os
        from pathlib import Path

        rows = [
            f"exe={sys.executable}",
            f"argv={sys.argv}",
            f"cwd={os.getcwd()}",
            f"pid={os.getpid()} ppid={os.getppid()}",
            f"stdout_none={sys.stdout is None}",
            f"stderr_none={sys.stderr is None}",
            f"console={bool(ctypes.windll.kernel32.GetConsoleWindow())}",
            f"note={note}",
        ]
        root = Path(__file__).resolve().parent.parent
        folder = root / "outputs" / "logs"
        folder.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with (folder / "launch.log").open("a", encoding="utf-8") as fh:
            fh.write(f"\n[{stamp}] " + " | ".join(rows) + "\n")
    except BaseException:                                 # noqa: BLE001
        pass


def _live_instance_hint() -> str:
    """报出"占着锁的是谁"。先读锁文件里的 PID，读不到再退回 `launch.log`。

    为什么要说清 PID（2026-11-08）：用户连续三次双击 `run.bat` 全被静默挡掉，
    因为旧代码只 return 0、一个字都不说。锁文件里的 PID 是现成的一手证据；
    进程已死就**不报**——宁可少说，也不指一个假的占用者。
    """
    try:
        import ctypes
        import re
        from pathlib import Path

        k32 = ctypes.windll.kernel32

        def alive(pid: int) -> bool:
            handle = k32.OpenProcess(0x1000, False, pid)   # QUERY_LIMITED_INFORMATION
            if not handle:
                return False
            code = ctypes.c_ulong()
            ok = bool(k32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
            k32.CloseHandle(handle)
            return ok

        raw = ""
        try:
            raw = _pid_path().read_text(encoding="ascii").strip()
        except OSError:
            raw = ""
        if raw.isdigit() and alive(int(raw)):
            return f"PID {raw}（它正开着窗口）"

        log = Path(__file__).resolve().parents[1] / "outputs" / "logs" / "launch.log"
        if not log.exists():
            return ""
        text = log.read_text(encoding="utf-8", errors="replace")
        hits = re.findall(r"\[([\d\-: ]+)\] .*?pid=(\d+).*?note=正常启动", text)
        if hits:
            when, pid = hits[-1]
            if alive(int(pid)):
                return f"PID {pid}（{when.strip()} 启动的那一次）"
        return ""
    except Exception:                                     # noqa: BLE001
        return ""


def _explain_already_running() -> None:
    """被单实例锁挡掉时，把"为什么没开窗"说成人话。

    为什么必须弹窗（2026-11-08）：从 `launcher.exe` 起的进程**没有控制台**，
    `print` 等于写进虚空 —— 用户只看到"双击了，什么都没发生"。
    有控制台时只打印：`run.bat` 会用非 0 退出码把窗口 `pause` 住，不必再弹一次。
    """
    owner = _live_instance_hint()
    msg = "已经有一个本程序在运行，本次不再开第二个窗口。"
    if owner:
        msg += f"\n\n占用者：{owner}"
    msg += ("\n\n如果看不到那个窗口（可能被最小化、或在另一块显示器上），"
            "请先在任务管理器里结束它，再重新启动。")
    print("\n" + "-" * 62 + "\n" + msg + "\n" + "-" * 62, flush=True)
    try:
        import ctypes
        import os

        if os.environ.get("SPHERE_NO_DIALOG"):   # 自动化/脚本：只要文字，不要模态框
            return
        if not ctypes.windll.kernel32.GetConsoleWindow():
            ctypes.windll.user32.MessageBoxW(
                None, msg, "已经在运行（本次没有开窗口）", 0x00000040)
    except Exception:                                     # noqa: BLE001
        pass


def run() -> int:
    """启动新界面。返回进程退出码。

    被单实例锁挡掉时返回 **2、不是 0**（2026-11-08 修）：
    返回 0 会让 `run.bat` 认为"启动成功"直接收工，控制台一闪而过 ——
    用户看到的现象就是"打不开、自动退出"，而真实原因（已经有一个实例在跑）
    一个字都没露出来。这一条被用户连续踩过多次，属于**静默失败**，必须修。
    """
    if not _claim_single_instance():
        _log_launch_context("被单实例锁挡掉，本次没有开窗口")
        _explain_already_running()
        return 2
    _log_launch_context("正常启动")
    # 退出也留一行（2026-11-08）：只有"正常启动"没有"进程退出"，
    # 才能区分"窗口关了进程还在"和"进程真的走了"—— 这次就是分不清。
    import atexit
    atexit.register(lambda: _log_launch_context("进程退出"))
    app = App()
    app.mainloop()
    return 0
