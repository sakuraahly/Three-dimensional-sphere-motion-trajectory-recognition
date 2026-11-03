"""单目模式窗：① 内参 → ② 球与量程 → ③ 开始。

为什么单目窗是第一个做出来的模式窗（2026-10-27 计划书 §4 P3）：它没有历史包袱。
双目那边的每一个控件都对应着一段踩坑史，而单目在旧界面上**只有一个"焦距 fx"
输入框**——把它做成完整三步，等于用新框架从零走一遍，能最快暴露框架本身的毛病。

三条设计约束

1. **精度必须写在脸上**。单目靠球径先验 ``Z = f·D/(2r)``，0.6 m 处误差约 ±7 mm，
   比双目差一个量级。界面给的是**算出来的误差预算**，不是"精度够用"这类软话术
   —— 让人拿单目结果去当双目用，比不给他这个功能更糟。
2. **没有标定就说没有标定**。旧界面那个「焦距fx(留空自动)」里的"自动"实际是
   ``geometry.py`` 按 60° 视场猜的（1% 焦距误差 = 6 mm @0.6 m）。这里三态分明：
   已标定 / 手动覆盖 / 未标定（会猜）。
3. **不假装有入口**。``apps/calibrate_mono.py`` 还没落地时，标定按钮明确写
   "入口待落地"并给出现阶段可用的替代品与其局限，而不是做一个点了没反应的按钮。

会话（2026-10-29 补）：录一段 = 建一个 ``data/mono/<时间戳>/`` 目录，"录了什么、
跑过检测没有"由 ``session.json`` + ``derived/`` 说清楚，与双目同构。
决策与解析都在 ``apps/_mono_sessions.py``（纯逻辑）；本窗只负责把它接上控件，
并且**不碰** ``record_mono.py`` —— 录制器只认 ``--out``，不该知道"会话"这件事。
"""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from apps._collapsible import CollapsibleSection
from apps._gui_widgets import (BALL_METAL, PRESET_NONE, BG, INK, MUTED, OK,
                               PARAM_HINT_WRAP, WARN, BallSettings, ScrollPage,
                               StatusBar, Surface)
from apps._mono_sessions import (DEFAULT_ROOT as MONO_SESSIONS_ROOT,
                                 MonoSession, create_mono_session,
                                 delete_mono_session, find_clip,
                                 scan_mono_sessions, sync_from_take)
from apps._task_runner import ROOT
from apps import _actions
from apps._actions import CALIBRATE_MONO as MONO_CALIB_APP
from apps._task_runner import Task
from sphere_tracker.config import ColorConfig
from sphere_tracker.plugins import names as plugin_names      # noqa: E402

__all__ = ["MonoWindow", "MonoSessionPicker", "preset_choices"]


MONO_CALIB_JSON = ROOT / "config" / "mono_calibration.json"

CALIB_FRAME_DIR = ROOT / "outputs" / "mono_calib_frames"

# 误差预算的假设（写在界面上，不许藏起来）
ASSUMED = {"dr_px": 0.2, "dD_mm": 0.2, "df_rel": 0.01}

#: 半径**系统**偏差的实测值（交接文档 §35.2）：specular 的亚像素半径量的是高光斑，
#: 比球的真实轮廓小，于是反推出的等效直径 21.6 mm 只有名义 25 mm 的 86.4%。
#: 这里只作**文案与默认值**的来源，绝不乘进 Z —— 把一个可复核的实测偏差
#: 做成隐式常数，就再也没人看得出 Z 为什么偏了。
MEASURED_RADIUS_SHRINK = 0.136

#: 这台球的名义（卡尺）直径与**有效**直径。两者的区别是这一整轮的重点：
#: 名义值描述"球有多大"，有效值描述"这套检测器量出来的球有多大"，
#: ``Z = f·D/(2r)`` 里的 D 必须是后者，否则公式和测量对象不自洽。
NOMINAL_DIAMETER_MM = 25.0
EFFECTIVE_DIAMETER_MM = round(NOMINAL_DIAMETER_MM * (1 - MEASURED_RADIUS_SHRINK), 1)

#: 旧版本的默认球径。窗口关闭时会把输入框无条件写回 prefs，所以磁盘上的这个值
#: 与"用户真的想要 40"无法区分 —— 迁移只认这一个字面值，且只做一次。
LEGACY_DIAMETER = "40"

#: 界面球径的默认值，**mm**（人量卡尺用的是 mm）；命令行那边要 m，见 `diameter_arg`。
DEFAULT_DIAMETER_MM = float(EFFECTIVE_DIAMETER_MM)

#: 本窗起任务时的归属标记（``TaskRunner.start(owner=...)``）。
#: 统一结束回调是**全局**的，靠它才能只认自己起的任务，不把双目窗的
#: 失败写到单目窗状态条上。按标题字符串匹配会被"标题改一个字"悄悄弄坏。
OWNER = "mono"

#: 相机枚举的进程内缓存时长（秒）。
#:
#: 为什么需要缓存：拿不到 pnputil 的机器上 ``list_devices()`` 会退到
#: PowerShell ``Get-PnpDevice``，实测约 10 秒。用户"进门厅 → 单目 → 回门厅 →
#: 再进单目"是常态，每次都等 10 秒等于把这功能做成不可用。
DEVICE_CACHE_S = 30.0
_DEVICE_CACHE: dict = {"at": 0.0, "devices": [], "note": ""}

#: 相机下拉旁边那句解释（前后两段拼起来，枚举结果会改后半段）。
#:
#: 为什么必须写"不代表能出图"（2026-10-31 走查第 4 条）：角色那一栏只做
#: **静态匹配**（身份/端口对不对得上），根本没开相机。而这台机器上有过
#: "枚举得到 ≠ 拿得到帧"（aoni + MSMF 30.8 s 后报 -2147483638）。
#: 不写这一句，用户会把这行当成"相机可用"，打不开时只会以为是自己点错了。
CAMERA_HINT_BASE = ("「#编号」是此刻插着的设备，直接用；角色对上不代表能出图"
                    "（没开过相机），要确认点「换相机…」。")


def camera_hint_text(note: str) -> str:
    return CAMERA_HINT_BASE + (note or "")


def enumerate_devices() -> tuple[list, str]:
    """真的去枚举相机；读不到就返回空表 + 一句人话，**绝不抛**。

    这个函数**很慢**（见 ``DEVICE_CACHE_S``），只许在后台线程里调 ——
    2026-10-31 之前它在 ``MonoWindow.__init__`` 的主线程里跑，于是点门厅
    「进入（单目）」之后整个门厅冻住、窗才出现。
    """
    try:
        from sphere_tracker import camera_registry as cr
        return list(cr.list_devices()), ""
    except Exception as exc:                            # noqa: BLE001
        return [], f"（设备列表读不到：{type(exc).__name__}，只能按角色选）"


def cached_devices(max_age: float = DEVICE_CACHE_S) -> tuple[list, str] | None:
    """``max_age`` 秒内的枚举结果；没有 / 过期 → ``None``（调用方决定去后台枚举）。"""
    at = float(_DEVICE_CACHE["at"])
    if at <= 0 or (time.monotonic() - at) > max_age:
        return None
    return list(_DEVICE_CACHE["devices"]), str(_DEVICE_CACHE["note"])


def store_devices_cache(devices: list, note: str) -> None:
    """记下这次的枚举结果（与窗口无关：关窗也留着，下次进窗直接秒开）。"""
    _DEVICE_CACHE["at"] = time.monotonic()
    _DEVICE_CACHE["devices"] = list(devices)
    _DEVICE_CACHE["note"] = note

#: 离线/实时识别用的检测器候选。
#:
#: **候选名一律来自注册表**（`plugin_names("detector")`），不在本文件里手写一份 ——
#: `selftest_plugins` 的 R7 就是守这条的：名字清单只准注册表一份，否则新增一个
#: 检测器时下拉里永远看不到它（本项目最贵的一类漂移）。顺序 = 注册顺序，
#: 也就是 `plugins.names()` 文档里说的"下拉框展示顺序"。
#:
#: 唯一不在注册表里的是 ``smart``：它是本窗的行为（先用 color，退出码 2 就换
#: hough 重跑），不是一个检测器。
#:
#: 为什么这个下拉必须存在（2026-10-29 现场）：这颗球在画面里颜色不突出，
#: 默认 ``color`` 40 帧里检出 **0 帧**（CSV 的 x/y 全空），而 ``hough`` 95%、
#: ``fusion`` 82.5%。没有这个下拉，用户只能撞同一面墙 —— 而墙看起来像"跑完了"。
#:
#: ``smart`` **不是**注册表里的检测器，是本窗的行为：先用 color 跑，
#: 退出码 2（0 检出）就自动换 hough 重跑一遍。实时那边没法"重跑一遍录像"，
#: 所以 smart 在实时里退化成 color（状态条会说明）。
#:
#: 为什么叫 smart 而不是 auto：`selftest_gui_budget` 有一条硬判据
#: 「窗口里不许出现 auto / parallel / angled」——那是**双目模式**的词，
#: 出现在单目窗会让人以为又能切模式。换名字比放宽判据好：
#: 判据守的是真问题，只是这个词撞了车。
SMART = "smart"


def detector_choices() -> tuple[str, ...]:
    """下拉候选：注册表里的全部检测器 + ``smart``。

    **每次调用都重读注册表**（而不是 import 时算一次）：插件是各模块 import 时
    自己登记的，算早了就会漏掉后登记的检测器 —— 那种漏法在界面上看不出来。
    """
    return tuple(plugin_names("detector")) + (SMART,)


def preset_choices() -> tuple[str, ...]:
    """颜色预设候选：**唯一来源**是 ``ColorConfig.preset_names()``。

    为什么不在界面里手抄"红/绿/蓝/黄/白"：那份清单在预设表里已经有一份，
    抄一份的下场和检测器清单一样 —— 加一种球色要改三处，漏一处就表现为
    "下拉里根本没有这个颜色"。判据（``tests/selftest_gui_budget.py``）
    断言的就是"下拉项 == 预设表"，不是"有没有这五个字"。
    """
    return tuple(ColorConfig.preset_names())

#: 界面文案（不许只写在注释里）：用户是在"要不要换检测器"的那一刻需要它。
DETECTOR_HINT = ("球颜色不突出时 color 会 0 检出（本机实测 0/40 帧），改 hough（95%）"
                 "或 fusion（82.5%）。smart = 先 color，0 检出自动换 hough 重跑一遍。")

#: 识别来源：相机实时 / 当前会话录像（离线）。合在一个下拉里是为了**留住按钮预算**
#: （第③步按钮 ≤5，判据在 tests/selftest_gui_budget.py）——
#: 实时与离线本来就是同一个动作的两种输入，放一个主操作后面比并排两个按钮更准。
SOURCE_REALTIME = "实时（相机）"
SOURCE_OFFLINE = "离线（当前会话）"
SOURCE_CHOICES = (SOURCE_OFFLINE, SOURCE_REALTIME)

#: 有种子时**强制**用的检测器名（必须是注册表里的名字，见 `detector_choices()`）。
#: 为什么强制：``--seed`` 只有 specular 认（它按点给的圆心/半径做局部跟踪），
#: 别的检测器收到种子会直接忽略 —— 命令里带着 ``--seed 100,50,91`` 却仍在
#: 全图找圆，是"看起来用了种子、实际没有"的静默失效。
SEED_DETECTOR = "specular"

#: 种子文件在会话里的位置（与其它派生结果同处 ``derived/``）。
SEED_NAME = "seed.json"


def seed_arg_from_file(path, *, image_size=None) -> tuple[str, str]:
    """读 ``seed.json`` → (``--seed`` 的值, 一句提示)；值空 = 不能用。

    为什么抽成模块级纯函数：它是"用户点的那两下 → 命令行"之间唯一会**静默失效**
    的一环。文件里有 x/y/r、命令行里却没有（或带着**另一套分辨率**的坐标），
    界面上一个字都看不出来，只有轨迹悄悄偏移。抽出来才能在没有显示器的
    机器上断言（判据在 ``tests/selftest_gui_budget.py``）。

    分辨率必须核对：种子是"点在哪"的像素坐标，换一段不同尺寸的录像后
    同一组数字指向完全不同的位置 —— 那比没有种子更糟（会跟错目标）。
    所以 **不许抛**：任何读不动/不合法都退回自动检测，并把原因交给状态条。
    """
    import json

    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return "", f"种子文件读不了（{type(exc).__name__}），退回自动检测"
    if not isinstance(raw, dict):
        return "", "种子文件不是对象，退回自动检测"
    try:
        x, y = float(raw["x"]), float(raw["y"])
        r = float(raw["radius"])
    except (KeyError, TypeError, ValueError):
        return "", "种子文件缺 x / y / radius，退回自动检测"
    if r <= 0:
        return "", "种子半径不是正数，退回自动检测"
    size = raw.get("image_size")
    if image_size and isinstance(size, (list, tuple)) and len(size) == 2:
        try:
            if (int(size[0]), int(size[1])) != (int(image_size[0]), int(image_size[1])):
                return "", (f"种子是在 {int(size[0])}x{int(size[1])} 上点的，"
                            f"这段录像是 {int(image_size[0])}x{int(image_size[1])}，"
                            f"坐标对不上，退回自动检测")
        except (TypeError, ValueError):
            pass          # 种子里的 image_size 写坏了：坐标本身还能用，不因此丢掉
    return f"{x:g},{y:g},{r:g}", ""


def diameter_mm_or_none(text: str) -> float | None:
    """框里的球径（mm）→ 正数；空 / 非数字 / 非正 → ``None``。

    为什么这里不兜底：调用方必须能分清「空」（用默认值，合理）与「填错了」
    （也得用默认值，但**必须说一句**）。两者混成一个 ``None`` 就会出现
    "用户填了 abc、界面拿 21.6 算、谁都没告诉他"（2026-10-31 走查第 5 条）。
    """
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        mm = float(raw)
    except ValueError:
        return None
    return mm if mm > 0 else None


def diameter_arg(text: str) -> str:
    """把框里的球径（**mm**）变成命令行要用的值（**米**）：空 / 非法 → ``""``（不传）。

    ⚠️ 这条 ``/1000`` 是 2026-10-31 实测出来的断链，别删：
    界面按 mm 收（人量卡尺用的是 mm），而 ``--ball-diameter`` 与
    ``AppConfig.ball_diameter_m`` 按 **米** 收。原样传 21.6 等于告诉程序
    "球有 21.6 米大"，``run_video`` 于是**一帧有效三维点都没有**。

    **2026-11-03 改：留空不再兜底成 21.6 mm。** 原因是课题的测试球会从几毫米
    到几十毫米大范围变化，"有效直径 21.6"（镜面球高光半径偏小 13.6% 的产物）
    对别的球毫无意义；而单目 ``Z = f·D/(2r)`` 对 D 是 1:1 的 —— 兜底一个数，
    就是把"我不知道"变成"深度按比例错"。现在留空 = **不传**，由 ``run_video``
    在启动前明确拒绝，界面这一层也会先拦住并说明要填什么。
    """
    mm = diameter_mm_or_none(text)
    return "" if mm is None else f"{mm / 1000.0:g}"


def diameter_text_mm(text: str) -> str:
    """给**界面文案**用的球径（mm）。与 ``diameter_arg`` 分开是为了不混单位。

    状态条里写 ``D=25 mm`` 是对的（人量的是 mm）；写 ``D=0.025`` 会让人
    以为填错了单位。所以"进命令行用米、给人看用 mm"这两件事各有一个入口。
    留空时返回空串 —— 不再编一个 21.6 出来。
    """
    mm = diameter_mm_or_none(text)
    return "" if mm is None else f"{mm:g}"


def diameter_problem(text: str) -> str:
    """这个输入会不会让单目拿不到三维？会的话返回要写进状态条的一句，否则空串。"""
    raw = (text or "").strip()
    if not raw:
        return ("球径没填：单目估深 Z = f·D/(2r) 需要它，留空就不会出三维"
                "（填毫米，比如 25 mm 的球写 25）")
    if diameter_mm_or_none(raw) is None:
        return (f"球径那一栏「{raw}」不是有效的毫米数 —— 这里填毫米（例如 25），"
                f"不是 0.025")
    return ""


def migrate_legacy_diameter(prefs) -> str:
    """把旧默认球径一次性迁移成有效直径；返回给状态条的一句话（没迁移则空串）。

    为什么允许改一次"用户存过的值"：``40`` 不是谁的选择，是旧版默认值被
    ``_save_prefs`` 无条件持久化的结果，而界面上看不出这层来历。不迁移的话，
    兜底值改成 21.6 等于没改（键一直存在，永远读不到兜底）。

    为什么要有 flag：迁移完以后用户可能**真的**想要 40（比如换了一颗大球）。
    没有 flag 的话，每次开窗都会把他填的 40 再改回来 —— 那比不迁移更糟。
    所以"做过迁移决定"这件事本身要落盘，值只在恰好等于旧默认时才动。
    """
    if prefs.get_str("mono.diameter_migrated_eff", "") == "1":
        return ""
    # flag 先落：即使本次值不是 40（新装的机器、或用户已经自己改过），
    # 也要记住"这条 profile 已经过了迁移这一关"，否则以后他填 40 又会被改。
    prefs.set("mono.diameter_migrated_eff", "1")
    if prefs.get_str("mono.diameter_mm", "").strip() != LEGACY_DIAMETER:
        prefs.save()
        return ""
    # 2026-11-03：不再"40 → 21.6"。21.6 mm 是那颗 25 mm **镜面球**的有效直径
    # （高光半径偏小 13.6% 反推出来的），对别的球毫无意义；而球径现在没有默认值，
    # 留着 21.6 等于又造了一个隐式预设。改为**清空**，让用户按手里的球重填。
    prefs.set("mono.diameter_mm", "")
    prefs.save()
    return (f"旧默认球直径 {LEGACY_DIAMETER} mm 已清空 —— 球径现在没有默认值，"
            f"请按实际球的直径（毫米）重填；单目测深 Z 对它是 1:1 的")


def error_budget(fx: float, diameter_mm: float,
                 distances_m: tuple[float, ...] = (0.3, 0.6, 1.0),
                 radius_bias: float = 0.0) -> list[tuple[float, float]]:
    """每个工作距离上的 Z 误差（mm，RSS 合成）。

    ``Z = f·D/(2r)`` 取对数微分：``dZ/Z = df/f + dD/D − dr/r``，
    而 ``r = f·D/(2Z)``，所以三项都能用已知量算出来。
    这是**解析传播**，不是实测噪声 —— 它回答"量级是多少"，
    不回答"这台机器上具体多少"（后者要静置 30 s 量 std(r)）。

    ``radius_bias`` 是半径的**相对系统偏差**（正 = 量大了），默认 0.0，
    代表"用户填的是有效直径"这一前提成立。为什么要单独给参数：前三项都是
    随机项，13.6% 的半径偏差是**全帧同号**的系统项，混进 RSS 会低估它
    （1 m 处随机项合起来才几毫米，系统项是 157 mm）。
    """
    out: list[tuple[float, float]] = []
    for z_m in distances_m:
        z = z_m * 1000.0
        r = fx * diameter_mm / (2.0 * z)                 # 像素半径
        if r <= 0:
            continue
        d_df = z * ASSUMED["df_rel"]                     # 焦距项
        d_dD = z * (ASSUMED["dD_mm"] / diameter_mm)      # 球径项
        d_dr = z * (ASSUMED["dr_px"] / r)                # 半径随机噪声项
        # 系统项：r_meas = r·(1+b) ⟹ Z 偏 −b/(1+b)，所以分子取 |b| 的分式形式
        # 而不是直接把 b 当成 Z 的偏差（填 25 实际该填 21.6 时是 +15.7%，不是 13.6%）
        d_bias = z * (abs(radius_bias) / (1.0 + radius_bias)) if radius_bias else 0.0
        rss = (d_df ** 2 + d_dD ** 2 + d_dr ** 2 + d_bias ** 2) ** 0.5
        out.append((z_m, rss))
    return out


class MonoSessionPicker(tk.Toplevel):
    """历史会话：列出磁盘上的单目会话，选中后回填成"当前会话"。

    为什么不复用 ``apps._gui_stereo.SessionPicker``：那个按**装置几何**做三态
    过滤（匹配/不匹配/未记录），单目没有这个维度；塞一个 ``mode=None`` 进去
    只会把"跨装置"的口径混进来，用户看到的是另一套说法。
    写法照抄它（Treeview + 双击 + 一个主操作），用户不用学第二个列表。

    为什么坏目录也列出来并标红：录像是不可再生的，一条"读不了"的记录必须
    看得见、并且能从这里移进回收站 —— 否则它永远占着位置又没人知道怎么处理。
    """

    COLS = (("id", "目录", 150), ("created", "创建时间", 145),
            ("camera", "相机", 80), ("size", "规格", 130),
            ("frames", "帧数", 65), ("state", "状态", 110))

    def __init__(self, parent, *, root: Path, on_pick,
                 current: Path | None = None, on_root_change=None) -> None:
        super().__init__(parent)
        self.root_dir = Path(root)
        self.on_pick = on_pick
        # 换存放目录是"会话从哪来"的问题，入口就该在列出会话的地方。
        # 2026-10-30 从第③步挪到这里：那一步的按钮预算是 5（判据在
        # tests/selftest_gui_budget.py），而「点选球心」比它更常用 ——
        # 镜面球没有可靠的自动初始化，每段录像几乎都要点一下。
        self.on_root_change = on_root_change
        self.title("历史会话 · 单目")
        self.geometry("860x520")
        self.minsize(680, 400)
        self.configure(bg=BG)
        self.surface = Surface(self)
        self._rows = scan_mono_sessions(self.root_dir)

        box = self.surface.card(
            self, "会话",
            f"这里列出 {self._short(self.root_dir)} 下的每一个目录。"
            "「已录，未检测」表示视频在、还没跑过离线检测；"
            "坏目录（红字）是元数据读不了的，可以先移进回收站再看要不要恢复。")
        self.lbl_note = self.surface.hint(box, "")

        frame = ttk.Frame(self)
        frame.pack(fill="both", expand=True, padx=16, pady=(8, 0))
        self.tree = ttk.Treeview(frame, columns=[c[0] for c in self.COLS],
                                 show="headings", selectmode="browse")
        for key, title, width in self.COLS:
            self.tree.heading(key, text=title)
            self.tree.column(key, width=width,
                             anchor="center" if key in ("frames", "size") else "w")
        bar = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=bar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        # 状态着色：正常 / 待检测 / 坏会话，一眼分得开（不靠读那列文字）
        self.tree.tag_configure("ok", foreground=INK)
        self.tree.tag_configure("todo", foreground=WARN)
        self.tree.tag_configure("bad", foreground=WARN)
        self.tree.tag_configure("idle", foreground=MUTED)
        self.tree.bind("<Double-1>", lambda _e: self._accept())
        self.tree.bind("<Return>", lambda _e: self._accept())

        actions = ttk.Frame(self)
        actions.pack(fill="x", padx=16, pady=10)
        self.surface.primary(actions, "用这个会话", self._accept)
        self.surface.secondary(actions, "移到回收站", self._delete)
        self.surface.secondary(actions, "刷新", self._rescan)
        # 兜底入口：默认 data/mono 装不下时换个盘，或把散落的录像放到新目录里。
        self.surface.secondary(actions, "换目录…", self._change_root)
        self.surface.secondary(actions, "取消", self.destroy)
        # 这句必须写在按钮下面：它刚从第③步搬过来，如果只有点开才知道是干什么的，
        # 就等于从一个坑换到另一个坑。写清"平时不用动"和"它不搬已有会话"，
        # 免得有人在会话丢了之后拿它当"修复"用。
        notes = ttk.Frame(self)
        notes.pack(fill="x", padx=16, pady=(0, 10))
        self.surface.hint(
            notes,
            "「换目录…」平时不用动：只有会话根目录被移走、或录像改放到别的磁盘时，"
            "才用它把上面的列表指到新位置。它只改「以后再录的存哪」，"
            "不搬动已有会话，也不修复已经丢了的目录。")

        self._refresh()
        if current is not None:
            iid = str(Path(current))
            try:
                self.tree.selection_set(iid)
                self.tree.see(iid)
            except tk.TclError:
                pass

    # ---------- 内部 ----------
    @staticmethod
    def _short(p: Path) -> str:
        try:
            return Path(p).resolve().relative_to(ROOT).as_posix()
        except (ValueError, OSError):
            return str(p)

    def _tag_of(self, row: MonoSession) -> str:
        if row.error:
            return "bad"
        if not row.video:
            return "idle"
        return "ok" if row.has_result else "todo"

    def _refresh(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        for row in self._rows:
            size = (f"{row.width}x{row.height} @ {row.fps:.0f} fps"
                    if row.width and row.height else "")
            self.tree.insert(
                "", "end", iid=str(row.path),
                values=(row.id, row.created, row.camera, size,
                        row.frame_count or "", row.state),
                tags=(self._tag_of(row),))
        if not self._rows:
            self.tree.insert("", "end", iid="__none__", tags=("idle",),
                             values=("（这个目录下还没有会话）", "", "", "", "", ""))

    def _selected(self) -> MonoSession | None:
        sel = self.tree.selection()
        if not sel or sel[0] == "__none__":
            return None
        return next((r for r in self._rows if str(r.path) == sel[0]), None)

    def _rescan(self) -> None:
        self._rows = scan_mono_sessions(self.root_dir)
        self._refresh()

    def _change_root(self) -> None:
        """换会话根目录（兜底）：默认自动生成 data/mono/<时间戳>/，正常不用点。

        为什么留在这里而不是第③步：「换目录」回答的是"会话从哪来"，
        与本窗口列出的东西是同一个问题；而第③步的按钮预算只有 5，
        那个位置留给了更常用的「点选球心」。
        """
        from tkinter import filedialog
        chosen = filedialog.askdirectory(
            title="选择单目会话的存放位置（每段录制在下面建一个时间戳目录）",
            initialdir=str(self.root_dir), parent=self)
        if not chosen:
            return
        self.root_dir = Path(chosen)
        # 换根后**必须重扫**：不重扫的话列表还挂着旧目录的行，
        # 点「用这个会话」就会把一个不在新根下的路径当成当前会话。
        self._rescan()
        if self.on_root_change is not None:
            self.on_root_change(self.root_dir)
        self.lbl_note.configure(
            text=f"已换到 {self._short(self.root_dir)}；这里列出的是新目录下的会话",
            style="Hint.TLabel")

    def _accept(self) -> None:
        row = self._selected()
        if row is None:
            return
        self.on_pick(str(row.path))
        self.destroy()

    def _delete(self) -> None:
        """移到回收站（**不是彻底删除**）：录像是不可再生的。"""
        from tkinter import messagebox
        row = self._selected()
        if row is None:
            return
        ok = messagebox.askyesno(
            "移到回收站",
            f"把整个会话目录 {row.id} 移到回收站吗？\n\n"
            f"{self._short(row.path)}\n\n"
            "视频和检测结果会一起移走，之后可以从回收站恢复。",
            parent=self)
        if not ok:
            return
        try:
            batch = delete_mono_session(row.path, self.root_dir)
        except (OSError, ValueError) as exc:
            self.lbl_note.configure(text=f"没移成：{exc}", style="Warn.TLabel")
            return
        self.lbl_note.configure(
            text=f"已移到回收站：{self._short(batch.parent)}/{batch.name}"
                 "（要恢复请用数据管理里的回收站）", style="Hint.TLabel")
        self._rescan()


class MonoWindow(tk.Toplevel):
    """单目三步窗。关闭即回门厅（门厅一直开着）。"""

    def __init__(self, app) -> None:
        super().__init__(app)
        self.app = app
        self.configure(bg=BG)
        # 三步加起来约 712 px 高（实测 ① 10–338、② 348–512、③ 522–712），
        # 700 的窗口会把第③步的下半截切掉——用户看到的是"方框被遮挡"。
        # 给到 820 一次看全三步；屏幕逻辑高 1000，放得下。
        # 2026-11-03：第②步多了「球种类 / 检测方法 / 颜色预设」一行（+31 px），
        # 内容实测 806 > 820 窗口的 785 视口 —— 于是抬到 860（视口 825）。
        # 不改成 `fit_to_content`：本窗三步的高度是稳定的，按内容算只会让
        # 每次开窗先跳一下；而 860 仍小于工作区 960。
        self.geometry("880x860")
        self.minsize(780, 560)
        self.surface = Surface(self)

        # 记住上次填的值（GuiPrefs 只存输入；派生值在那里会被拒绝）
        prefs = app.prefs
        # 相机候选：**角色 + 在线设备**都要给（2026-10-29 现场事实）。
        # 只列角色会让用户面对一个必然开不出画面的选项 —— 现场 `left` 指向的
        # "aoni webcam A25" 此刻枚举不到，而唯一在线的那台外接机被绑成了 `right`。
        # 设备候选 `#N 名字` 直接对应 `--camera N`（resolve_camera_arg 认纯数字），
        # 不依赖任何角色绑定，所以"角色配错了"也不会把单目堵死。
        #
        # ⚠ 枚举**不许在主线程等**：拿不到 pnputil 时它退到 PowerShell
        # ``Get-PnpDevice``（实测 ~10 s），那 10 秒里门厅整个冻住、窗才出现。
        # 这里只读 30 秒内的缓存；冷的就先用"角色"开局，后台线程补设备。
        self._devices, self._device_note = self._initial_devices()
        self._enum_queue: queue.Queue = queue.Queue()
        #: 用户还没自己动过相机下拉 → 设备枚举回来后可以替他换成更靠谱的默认值。
        self._autopick = True
        _values = self._camera_values()
        # 存过的值可能已失效（角色改名、换端口导致名字变）：先整体匹配，
        # 再按 `#N` 认编号 —— 认不出才用默认值，别把失效的串带进命令行。
        _saved = self._match_saved_camera(prefs.get_str("mono.camera", ""), _values)
        _default = self._default_camera(_values)
        self._camera_note = ""
        if _saved and not _saved.startswith("#") and self._role_online(_saved) is False:
            # 上次选的角色现在不在线（现场就有这种：left 指向的那台没插）。
            # 沿用会让人一进窗口就拿着一个开不出画面的选项；静默改掉又不解释更糟，
            # 所以改动 + 说清楚，且下拉里仍留着它（用户重新 pin 后能选回去）。
            self._camera_note = (f"上次选的相机角色 {_saved} 当前不在线，"
                                 f"已改成 {_default}（pin 好以后可以再选回来）")
            _saved = ""
        self.var_camera = tk.StringVar(value=_saved or _default)
        # 存过值 = 用户自己选过，设备枚举回来后**不许**替他改（那会让他下次
        # 又要重选一遍）；没存过才是"界面替他挑的默认值"，可以升级成在线设备。
        self._autopick = not bool(_saved)
        self.var_cols = tk.StringVar(value=prefs.get_str("board.cols", "8"))
        self.var_rows = tk.StringVar(value=prefs.get_str("board.rows", "5"))
        self.var_square = tk.StringVar(value=prefs.get_str("board.square_mm", "25"))
        self.var_target = tk.StringVar(value=prefs.get_str("mono.target", "15"))
        # **没有默认值（2026-11-03 改）**：测试球会从几毫米到几十毫米变化，
        # "有效直径 21.6" 只对那颗 25 mm 镜面球成立（它还把高光半径偏小 13.6%
        # 折进去了）。留空时界面会拦住并说明要填什么，而不是替用户猜一个数 ——
        # 猜错的话 Z 会按 1:1 比例错，而且看起来完全"正常"。
        self._diameter_note = migrate_legacy_diameter(prefs)
        self.var_diameter = tk.StringVar(
            value=prefs.get_str("mono.diameter_mm", ""))
        self.var_focal = tk.StringVar(
            value=prefs.get_str("mono.focal_override", ""))   # 空 = 用标定值
        self.var_record_secs = tk.StringVar(
            value=prefs.get_str("mono.record_seconds", "10"))
        # 检测器：默认 color（快），但**必须能换** —— 见 detector_choices() 的说明。
        # 存过的值可能已经不合法（名字改过/注册表删了某个检测器）：认不出就回落 color，
        # 别把一个不存在的名字带进命令行（那样 run_video 会在 argparse 就退出）。
        _saved_det = prefs.get_str("mono.detector", "")
        self.var_detector = tk.StringVar(
            value=_saved_det if _saved_det in detector_choices() else "color")
        # 颜色预设（2026-11-03）：只在"纯色塑料球/自定义"下才下传。
        # 存过的值可能已经不在预设表里（改过名/删掉了）：交给 BallSettings 折成
        # 「不使用」—— `--preset` 有 choices，带一个不存在的名字会直接 argparse 退出。
        self.var_preset = tk.StringVar(
            value=prefs.get_str("mono.preset", PRESET_NONE) or PRESET_NONE)
        self.var_source = tk.StringVar(
            value=(prefs.get_str("mono.source", "") or SOURCE_OFFLINE)
            if prefs.get_str("mono.source", "") in SOURCE_CHOICES else SOURCE_OFFLINE)
        # auto 链路的状态：只有 auto 才在 0 检出后自动换 hough 重跑一次
        self._auto_retry_pending = False

        # 会话：**只记用户显式换过的根目录**（默认 data/mono）。
        # 不记"上次选中的会话"—— 那是磁盘现状，属于派生值，抄进界面状态就又多
        # 一份会和磁盘分叉的真相（GuiPrefs 的 FORBIDDEN 清单守的就是这条）。
        _root_txt = prefs.get_str("mono.session_root", "")
        self._session_root = Path(_root_txt) if _root_txt else MONO_SESSIONS_ROOT
        self._session: MonoSession | None = None
        #: 第③步「会话目录」输入框里显示的文字（只读，回填自磁盘现状，不是输入）。
        self.var_session_dir = tk.StringVar(value="")
        #: 当前会话里那份种子文件（有 = 「开始识别」走人工种子的跟踪）。
        #: **每次刷新都从磁盘重取**：它不是输入，是派生事实；缓存久了就会
        #: 与磁盘分叉（用户删了 seed.json，界面还在说"会用种子"）。
        self._seed_json: Path | None = None

        # 先占底部：pack 从容器边缘往里分配，先 pack 的优先拿到空间。
        # 反过来（先 pack 主内容区、再 pack 状态条）状态条会**盖住内容** ——
        # 2026-10-28 截图里第 4 步卡片就被盖掉了一半。
        self.status = StatusBar(self, on_log=app.open_console,
                                on_stop=self._stop_my_tasks)
        self.status.pack(fill="x", side="bottom")
        page = ScrollPage(self, self.surface)
        page.pack(fill="both", expand=True)
        body = page.content
        self._build_step1(body)
        self._build_step2(body)
        self._build_step3(body)
        self._refresh_state()
        # 统一结束回调：本窗所有按钮的"结局"都从这里出去（见 _task_runner 的
        # ``add_finish_listener``）。注册放在建完控件之后 —— 回调会碰状态条。
        self.app.tasks.add_finish_listener(self._on_task_finished)
        # 相机枚举：冷的缓存才去起线程，暖的直接用（二次进窗秒开）。
        if self._enum_pending:
            self._start_device_enum()
        # 上面那几句必须最后说：_refresh_state 会把状态条改回"就绪"。
        # 几句都成立时用分号连起来，不让后一句把前一句顶掉 —— 它们都在说
        # "界面里有个值和你上次看到的不一样"，少一句用户就会以为是自己点错了。
        _notes = [n for n in (self._camera_note, self._diameter_note) if n]
        if self._enum_pending:
            # 不挡路，但要说一句：否则下拉里暂时只有角色、没有设备，用户会
            # 以为"这台机器只认角色"。约 10 秒是慢枚举的实测值。
            _notes.append("正在枚举相机（约 10 秒，不用等它，下拉会自动补上「#编号」）")
        if _notes:
            self.status.set_status("；".join(_notes), "warn")

    # ------------------------------------------------------------------ #
    # 布局辅助（首屏预算）
    # ------------------------------------------------------------------ #
    def _advanced(self, parent, title: str) -> CollapsibleSection:
        """在卡片里塞一个默认收起的折叠区。

        为什么用 grid 而不是 pack：卡片里的兄弟控件都是 grid 放的，Tk 不允许
        同一个容器混用两种几何管理器（会直接抛 TclError）。折叠区**内部**是它
        自己的 Frame，那边统一用 pack（见 ``_hint_pack``）。
        """
        sec = CollapsibleSection(parent, title, expanded=False, padding=(10, 6))
        sec.grid(row=self.surface._next_row(parent), column=0, columnspan=9,
                 sticky="ew", pady=(6, 0))
        return sec

    def _hint_pack(self, parent, text: str, style: str = "Hint.TLabel"):
        """折叠区里的说明行：pack 版 ``Surface.hint``。

        为什么不用 ``Surface.hint``：它用 grid，而折叠区内部只有 pack（见上）。
        换行宽度仍登记给 Surface —— 窗口变窄时要跟着回排，漏登记的话
        折叠区里的长句会自己撑出一条横向滚动条。
        """
        lab = ttk.Label(parent, text=text, style=style, wraplength=PARAM_HINT_WRAP)
        lab.pack(anchor="w", pady=(0, 4))
        self.surface._wrapped.append((lab, "param"))
        return lab

    # ------------------------------------------------------------------ #
    # ① 内参
    # ------------------------------------------------------------------ #
    def _build_step1(self, parent) -> None:
        box = self.surface.card(
            parent, "① 标定内参（一次就够）",
            "单目测深全靠焦距、球直径、像素半径。焦距必须标"
            "（1% 差 → 0.6 m 处 6 mm）。")
        # 状态行
        # 状态行走 surface.hint：登记换行 + 由游标分配行号（不写死 row=1）
        self.lbl_calib = self.surface.hint(box, "检查中…")
        # 相机身份放 L1（同双目窗的道理）：没有它就没法确认用的是哪台
        self.lbl_cams = self.surface.hint(box, "读取相机…")
        # 选哪台相机：值取自 config/cameras.json 的角色 **加上**此刻在线设备，
        # **不给手打**（编号会漂移，手打等于让人记住不可靠的东西）
        pick = ttk.Frame(box)
        pick.grid(row=self.surface._next_row(box), column=0, columnspan=9,
                  sticky="w", pady=(0, 4))
        ttk.Label(pick, text="用这台相机").pack(side="left")
        # width 给到 34：设备候选带名字（"#0 aoni webcam A25 left"），
        # 只给编号等于让用户靠猜 —— 名字才能当场判断"这台对不对"。
        self.cb_camera = ttk.Combobox(pick, values=self._camera_values(), width=34,
                                      state="readonly",
                                      textvariable=self.var_camera)
        self.cb_camera.pack(side="left", padx=(8, 0))
        # 用户自己动过下拉，就再也不许"替他把默认值升级成在线设备" ——
        # 那等于把他刚选的那台悄悄换掉。
        self.cb_camera.bind("<<ComboboxSelected>>",
                            lambda _e: setattr(self, "_autopick", False))
        self.lbl_devices = self.surface.hint(box,
                                             camera_hint_text(self._device_note))

        # 板参数（三个必需值）合在一行 —— 它们是一件事（这块板长什么样），
        # 拆成三行会让"8×5、25 mm"读起来不像一句话；但它们**不在首屏**：
        # 一次标定只填一次，之后几个月都不碰，却要吃掉 150 px。
        actions = self.surface.action_row(box)
        self.btn_capture = self.surface.primary(actions, "拍标定图",
                                                self._start_capture)
        self.btn_calibrate = self.surface.secondary(actions, "标定内参",
                                                    self._start_calibrate)
        if not MONO_CALIB_APP.exists():
            self.btn_calibrate.state(["disabled"])
        self.surface.secondary(actions, "换相机…", self._open_cameras)
        self.surface.secondary(actions, "打开标定图文件夹", self._open_frames_dir)
        self.btn_estimate = self.surface.secondary(
            actions, "替代：单点焦距估计", self._start_focal_estimate)

        # 为什么要折叠（2026-10-31 走查第 8 条）：首屏内容比视口高 382 px，
        # 第③步整块（含「开始识别」）在屏幕外 —— 用户的第一反应是"按钮怎么找不到"。
        # 折叠的是**一次标定只填一次**的东西；「拍标定图 / 标定内参」留在首屏。
        adv = self._advanced(box, "高级：棋盘尺寸与替代标定")
        self._hint_pack(
            adv.content,
            "拍标定图用棋盘：A4 打印一张，在画面里上下左右都摆一摆，"
            "至少 8 个不同姿态（挤在中央会让畸变系数失真）。")
        line = ttk.Frame(adv.content)
        line.pack(anchor="w", pady=(0, 4))
        ttk.Label(line, text="棋盘内角点").pack(side="left")
        ttk.Entry(line, textvariable=self.var_cols, width=5).pack(
            side="left", padx=(8, 4))
        ttk.Label(line, text="×").pack(side="left")
        ttk.Entry(line, textvariable=self.var_rows, width=5).pack(
            side="left", padx=(4, 12))
        ttk.Label(line, text="方格边长 mm").pack(side="left")
        ttk.Entry(line, textvariable=self.var_square, width=6).pack(
            side="left", padx=(8, 0))
        self._hint_pack(adv.content, "内角点 = 棋盘格内部交叉点个数，不是格子数。"
                                     "手上那张纸写着 8×5 就填 8 和 5；方格边长按实际量。")
        # 这句话必须写在按钮旁边，不能只放在点下去之后的状态条里：
        # 用户是在"要不要用这个替代方案"的那一刻需要它。
        # 旧文案（"球径误差会算进 fx、再正算深度时乘第二次"）是错的 ——
        # 两处用的是同一个 D，错的是"换了球却不改框里的值"，不是误差叠加。
        self._hint_pack(
            adv.content,
            "替代方案用的是第②步填的球直径：同一颗球、同一个 D，两处同源，"
            "所以不叠加；但换了球一定要改第②步那一栏，否则 fx 会按旧球径算错"
            "（f = 2·r·Z/D，D 错多少，fx 就错多少）。")

    # ------------------------------------------------------------------ #
    # ② 球与量程
    # ------------------------------------------------------------------ #
    def _build_step2(self, parent) -> None:
        box = self.surface.card(
            parent, "② 球与量程",
            "球直径是单目测深的第二个已知量：同一颗球在画面里占多少像素，"
            "就能反推它多远。直径量错 0.5%，深度就错 0.5%。"
            "下面的误差预算是按解析公式算的，用来判断这套方案够不够你的实验。")
        # ⚠ 走游标：写死 row=1 时游标也停在 1（card() 的说明占了 row=0），
        # 后面那句 hint 就拿到同一个 row=1 —— 说明文字直接压在参数行上，
        # 两个输入框看起来"被遮挡"（2026-10-28 用户截图）。
        #
        # 球种类 / 检测方法 / 颜色预设 / 球直径合成**一行**（2026-11-03）：
        # 它们回答的是同一件事——"这是什么球、按它的哪条线索去找"。原来的
        # 「检测器」下拉在第③步、预设根本没有，于是"录完才发现检测器不对"。
        # 这里挂了球种类之后，第③步只留「识别来源」，选项不再分成两处。
        self.ball = BallSettings(
            self.surface, box,
            methods=detector_choices(), presets=preset_choices(),
            method_var=self.var_detector, preset_var=self.var_preset,
            diameter_var=self.var_diameter,
            on_diameter_key=self._refresh_budget,
            on_change=self._ball_changed)
        # 标签仍是「球直径（有效值）」（BallSettings 的默认标签）：这个数不是
        # 卡尺量出来的名义直径，而是"填进去能让 Z 对上"的那个数。写成"球直径"
        # 会让人直接填 25，然后整体偏大 15.7%。
        row = ttk.Frame(box)
        row.grid(row=self.surface._next_row(box), column=0, columnspan=9,
                 sticky="w", pady=(4, 0))
        ttk.Label(row, text="焦距 fx（留空用标定值；手填会覆盖它）").pack(side="left")
        ent2 = ttk.Entry(row, textvariable=self.var_focal, width=8)
        ent2.pack(side="left", padx=(8, 0))
        ent2.bind("<KeyRelease>", lambda _e: self._refresh_budget())
        # 原来这里还挂一句"手动填 fx 会覆盖标定值…"的说明行（23 px）。
        # 那句话并进了输入框标签：它必须被看见，但不必占一整行 ——
        # 首屏每 20 px 都在决定「开始识别」出不出现（2026-10-31 走查第 8 条）。
        # 有效值和名义值的差距是**实测**的（交接文档 §35.2），不是保守估计，
        # 所以必须摆在输入框正下方，而不是折叠进"高级"里。
        self.surface.hint(
            box,
            f"名义 {NOMINAL_DIAMETER_MM:g} mm 的球在这套检测器下约填 "
            f"{EFFECTIVE_DIAMETER_MM:g} mm —— 高光半径偏小 "
            f"{MEASURED_RADIUS_SHRINK * 100:.1f}%，填名义值会让深度偏大 "
            f"{(1 / (1 - MEASURED_RADIUS_SHRINK) - 1) * 100:.1f}%",
            style="Warn.TLabel")

        # 误差预算行走游标（原来写死 row=20，加了控件就会撞）。
        # 初值留空：这一行的内容由 `_refresh_budget` 按"框里到底填了什么"决定。
        # 写一句"填一个球直径才能算误差预算"当占位是错的 —— 框里本来就有 21.6，
        # 那句话在开窗那一刻就在骗人（2026-10-31 走查第 5 条）。
        self.lbl_budget = self.surface.hint(box, "")
        # 系统偏差单独一行、**永远在**：预算函数没法知道用户填的是名义还是有效值，
        # 所以不猜，只把这条实测偏差摆在旁边让人自己核对。
        self.lbl_budget_sys = self.surface.hint(box, "", style="Warn.TLabel")

    # ------------------------------------------------------------------ #
    # ③ 录制与识别
    # ------------------------------------------------------------------ #
    def _build_step3(self, parent) -> None:
        box = self.surface.card(
            parent, "③ 录制与识别",
            "先录一段（只写原图，录完可换参数反复检测）：每段自动建一个会话目录，"
            "结果写进 derived/。录完点「开始识别」处理当前会话；"
            "没有内参时焦距按 60° 视场估算，那种结果只能看趋势。")
        # 检测器与识别来源合成一行：它们一起回答"用什么、跑在哪"，
        # 拆成两行只会把第③步撑长（首屏装不下三步是 2026-10-28 的问题）。
        #
        # 2026-11-03：检测器下拉搬去了第②步「球种类与检测参数」——它属于
        # "这是什么球"，不属于"录制与识别"；而两块地方各有一个下拉就是
        # 同一个值的两份控件（键一样、约束两套，迟早对不上）。这里只剩来源。
        pick = ttk.Frame(box)
        pick.grid(row=self.surface._next_row(box), column=0, columnspan=9,
                  sticky="w", pady=(4, 0))
        ttk.Label(pick, text="识别来源").pack(side="left")
        ttk.Combobox(pick, values=list(SOURCE_CHOICES), width=16,
                     state="readonly",
                     textvariable=self.var_source).pack(side="left", padx=(8, 0))

        # 会话目录行（对齐双目第④步）：只读输入框显示当前会话 + 「选择…」按钮。
        # 之前只有「历史会话…」按钮（弹窗选），现在把"当前会话"摆到首屏一眼可见，
        # 选择动作由「选择…」承担，「历史会话…」按钮撤掉（按钮预算 5 已满）。
        sess_line = ttk.Frame(box)
        sess_line.grid(row=self.surface._next_row(box), column=0, columnspan=9,
                       sticky="we", pady=(4, 0))
        ttk.Label(sess_line, text="会话目录").pack(side="left")
        ttk.Entry(sess_line, textvariable=self.var_session_dir, width=38,
                  state="readonly").pack(side="left", padx=(8, 8), fill="x",
                                         expand=True)
        ttk.Button(sess_line, text="选择…", command=self._pick_session).pack(
            side="left")

        # 会话摘要：录完 / 跑完检测后状态会变，必须一眼看出"录了但还没检测"。
        # 它在首屏（不在折叠里）—— 这是用户判断"下一步该点什么"的唯一依据。
        self.lbl_session = self.surface.hint(box, "")

        actions = self.surface.action_row(box)
        self.surface.primary(actions, "录制视频", self._start_record)
        # 实时/离线合成**一个**主操作：第③步按钮预算是 5（selftest_gui_budget 会数），
        # 而"看 3D"是单目唯一的可视化入口，必须挤进来 —— 它比多一个并列按钮值。
        self.surface.secondary(actions, "开始识别", self._start_by_source)
        self.btn_view3d = self.surface.secondary(actions, "看 3D", self._open_3d)
        # 镜面球没有可靠的自动初始化（实测 color 0/288 落在球上、hough 只有 1.4%，
        # 半径中位 69.5 px vs 真值 91 px），所以"点一下球"是这类录像的唯一正路。
        # 「历史会话…」已并入上面的「会话目录 + 选择…」行，这里不再单独占一个按钮。
        self.btn_seed = self.surface.secondary(actions, "点选球心", self._pick_seed)
        self.btn_seed.state(["disabled"])       # 没会话/没视频时不可点，见 _sync_seed_button

        # 折叠区：录制时长 + 三段落说明 + 会话路径。
        # 为什么这些可以让位（2026-10-31 走查第 8 条）：首屏比视口高 382 px，
        # 第③步整块（含「开始识别」）落在屏幕外，用户的结论是"按钮找不到"。
        # 折叠的每一项都是"默认值就够用、想调才进来"的东西；而"录多久"这种
        # 真正影响结果的数，写在折叠区标题里，收起时也看得见。
        adv = self._advanced(box, "高级：录制时长、检测器说明、会话位置")
        self._sync_record_summary(adv)
        self.adv_step3 = adv
        self.var_record_secs.trace_add(
            "write", lambda *_a: self._sync_record_summary(adv))
        line = ttk.Frame(adv.content)
        line.pack(anchor="w", pady=(0, 4))
        ttk.Label(line, text="录制时长 秒").pack(side="left")
        ttk.Entry(line, textvariable=self.var_record_secs, width=6).pack(
            side="left", padx=(8, 12))
        ttk.Label(line, text="（0 = 按 q 停止）", style="Hint.TLabel").pack(side="left")
        self._hint_pack(adv.content, DETECTOR_HINT)
        # 球种类那三档各自要注意什么（2026-11-03）：放在这里而不是第②步的
        # 首屏，是因为首屏高度已经顶到视口；而它是"换球时才需要读一遍"的内容，
        # 折叠着不影响日常使用（切档那一刻状态条还会再说一句）。
        self._hint_pack(
            adv.content,
            "球种类决定检测方法：镜面金属球 = specular，必须先在「点选球心」里"
            "人工点一次球心（没有种子多半 0 检出）；纯色塑料球 = color + 颜色预设，"
            "球色要和背景分得开；自定义 = 直接用注册表里的某个方法。")
        self._hint_pack(
            adv.content,
            "「离线」处理当前会话的录像：结果写进 derived/，跑完「看 3D」才会亮；"
            "「实时」直接开相机，不落盘，也没有 3D 结果"
            "（单目三维靠球径先验，需要落盘的轨迹）。")
        # 会话目录只显示、不手打：路径由时间戳自动生成，人没必要记它。
        sess = ttk.Frame(adv.content)
        sess.pack(anchor="w", pady=(0, 4))
        ttk.Label(sess, text="本次录制存到").pack(side="left")
        self.lbl_session_dir = ttk.Label(sess, text="", style="Hint.TLabel")
        self.lbl_session_dir.pack(side="left", padx=(8, 0))
        self._hint_pack(
            adv.content,
            "球是镜面的（镀铬、反光）时，自动检测会把书架/衣服当球：先录一段，"
            "再点「点选球心」——第一点球心、第二点球边（半径不靠自动估），"
            "之后「开始识别」用它跟踪，检测器会自动切成 specular。"
            "换会话存放目录的入口在「选择…」弹窗里。")

    def _ball_changed(self) -> None:
        """切了球种类 / 改了方法之后，把"这一档要注意什么"写进状态条。

        为什么不另起一行常驻显示：第②步的首屏高度已经顶到视口（再加一行就把
        第③步整块挤出屏幕）。说明的**常驻**部分在卡片说明里（三种球各一句），
        这里补的是"刚选完那一刻该做什么"的那一句。
        """
        text = self.ball.hint()
        if text:
            self.status.set_status(text, "warn")

    def _sync_record_summary(self, section) -> None:
        """折叠区标题里带上"现在录多久"：收起时也得看得见这个数。

        为什么盯这一项：它直接决定录出来的长度，而它已经不在首屏了；
        标题是唯一还能让人不点开就看到它的地方。
        """
        secs = self.var_record_secs.get().strip() or "10"
        try:
            section.set_summary(f"{secs} 秒"
                                if secs.isdigit() else f"「{secs}」不是数字，按 10 秒")
        except tk.TclError:
            pass                                    # 窗口已销毁：标题没人看得到

    def _start_by_source(self) -> None:
        """「开始识别」：按上面的下拉决定走实时还是离线。

        为什么不保留两个并列按钮：第③步的按钮预算 ≤5，而"看 3D"必须有位置；
        更重要的是实时与离线是**同一个动作的两种输入**（相机 / 录像），
        放在一个主操作后面才是它们的真实关系。
        """
        if self.var_source.get() == SOURCE_REALTIME:
            self._start_realtime()
        else:
            self._start_video()

    def _start_record(self) -> None:
        """录制单目视频（只写原图）：**先建会话目录，再让录制器往里写**。

        为什么目录决策留在窗口这一侧：``record_mono.py`` 只认 ``--out``，
        它不需要知道"会话"这件事。目录形态由调用方定，录制器就永远只做
        一件事（开相机、写视频、写自证元数据），两个文件的职责不会互相渗。
        """
        secs = self.var_record_secs.get().strip() or "10"
        cam_label = self.var_camera.get().strip()
        try:
            # 会话里记**下拉显示的那一串**（角色名或 "#0 设备名"）：一周后
            # 看到 "left" / "#0 aoni webcam A25 left" 才知道这段是谁录的，
            # 记 "--camera 0" 这种命令参数则要再翻一遍配置才认得出来。
            sess_dir = create_mono_session(self._session_root, camera=cam_label)
        except OSError as exc:
            self.status.set_status(
                f"建不了会话目录（{self._short(self._session_root)}）：{exc}", "warn")
            return
        out = sess_dir / "clip.mkv"
        cmd = _actions.record_mono(self._camera_arg(cam_label) or "0", secs, out)
        self._start(cmd, f"单目录制 {secs}s",
                    on_done=lambda task: self._on_recorded(sess_dir, task))
        self._select_session(sess_dir)
        self._refresh_session()
        # 角色离线就先说一句：录制任务会照跑，但它多半在"3 秒没拿到帧"那步失败，
        # 让用户当场知道"是相机不在，不是我点错了"
        warn = ""
        level = "info"
        if cam_label and not cam_label.startswith("#") \
                and self._role_online(cam_label) is False:
            warn = (f"。注意：角色 {cam_label} 指向的相机当前不在线，"
                    f"录制多半开不出画面 —— 换「#编号」那个选项更稳")
            level = "warn"
        self.status.set_status(
            f"已开始录制 {secs} 秒 → {self._short(sess_dir)}/（有倒计时，"
            f"看到 REC 才开始动）。录完点「开始识别」直接检测这段会话" + warn, level)

    def _on_recorded(self, sess_dir: Path, task=None) -> None:
        """录制任务结束：把 sidecar 的实测值补进 session.json，再刷新摘要。

        没有这一步，会话里就只有"本来要录什么"（宽高 0），而实测分辨率/帧率
        只有录制器知道 —— 那是它的自证元数据，不该让用户去别的文件里找。

        ``task`` 是统一结束回调带上来的那条任务：**失败先说话**。旧版不看返回码，
        录制进程挂掉之后只会说"会话里没有视频，八成是相机开不出图"——
        那是猜的，而真实原因是清楚的（日志尾里就写着）。
        """
        if not self._alive():
            return                      # 窗口先被关了：结果已落盘，界面不用更新
        if task is not None and not getattr(task, "ok", False):
            self._select_session(sess_dir)
            self._refresh_session()
            self.status.set_status(
                f"录制没成功（返回码 {task.exit_code}）：{task.tail_text(2)[:100]} "
                f"—— {self._log_hint(task)}", "warn")
            return
        note = ""
        try:
            if not sync_from_take(sess_dir):
                note = "（没找到录制器写的元数据，实测分辨率/帧数暂时未知）"
        except OSError as exc:
            note = f"（session.json 没写成：{exc}；视频本身不受影响）"
        self._select_session(sess_dir)
        self._refresh_session()
        cur = self._session
        if cur is None or not cur.video:
            # 这台机器上真实发生过："枚举得到 ≠ 拿得到帧"（#0 aoni 在线，
            # MSMF 30.8 s 后报 -2147483638 打不开）。所以失败提示要指向
            # identify（它会真开相机、逐台报 ✔/✘），而不是只说"去看日志"。
            self.status.set_status(
                "录制任务结束了，但会话里没有视频。八成是这台相机在线却开不出图"
                "（枚举得到不等于拿得到帧）—— 点「换相机…」跑 identify，"
                "看哪台能出图；日志里这一段也有逐台结果" + note, "warn")
            return
        self.status.set_status(
            f"录好了：{cur.summary()}{note}。点「开始识别」（来源「离线」）开始检测，"
            f"结果写进 derived/", "ok")

    # ------------------------------------------------------------------ #
    # 会话
    # ------------------------------------------------------------------ #
    def _alive(self) -> bool:
        """窗口还在不在：任务回调可能在用户关窗之后才到（那时碰控件会抛 TclError）。"""
        try:
            return bool(self.winfo_exists())
        except tk.TclError:
            return False

    @staticmethod
    def _short(p: Path) -> str:
        """仓库内的路径显示成相对路径（data/mono/…），别让人在长绝对路径里找重点。"""
        try:
            return Path(p).resolve().relative_to(ROOT).as_posix()
        except (ValueError, OSError):
            return str(p)

    def _preview_dir(self) -> str:
        """下一次录制会建在哪 —— **只是预览**：打开窗口不该在磁盘上留空目录。"""
        return self._short(self._session_root / time.strftime("%Y%m%d-%H%M%S")) + "/"

    def _find_session(self, sess_dir: Path) -> MonoSession | None:
        return next((row for row in scan_mono_sessions(sess_dir.parent)
                     if row.path.name == sess_dir.name), None)

    def _select_session(self, sess_dir: Path) -> None:
        """把"当前会话"指向 sess_dir（重扫一遍，要的是最新摘要）。

        ``record_mono`` 现在默认也落成 ``data/mono/<ts>/clip.mkv``，但**它不写
        session.json**（录制器不该知道"会话"这件事）。所以 CLI 直录的目录会以
        "只有录制器元数据"的形态出现 —— 选中它时在这里补成正式会话。
        补在"选中"这一刻而不是扫描里：扫描必须只读，用户没点开的目录
        不该因为我们看了一眼就被改。
        """
        sess_dir = Path(sess_dir)
        row = self._find_session(sess_dir)
        if row is not None and row.ok and row.note:
            try:
                if sync_from_take(sess_dir):
                    row = self._find_session(sess_dir) or row
            except OSError:
                pass          # 写不进去也不影响看视频，note 会继续显示
        self._session = row

    def _reload_session(self) -> MonoSession | None:
        """重读当前会话：录完/跑完检测后状态变了，缓存会一直显示旧的。"""
        cur = self._session
        if cur is None:
            return None
        return next((row for row in scan_mono_sessions(cur.path.parent)
                     if row.path.name == cur.path.name), None)

    def _refresh_session(self) -> None:
        if self._session is not None:
            self._session = self._reload_session()
        cur = self._session
        self._sync_view3d_button()
        self._sync_seed_button()
        # 种子路径每次都从磁盘重取（不是输入，是派生事实）：用户删掉 seed.json
        # 之后界面若还在说"会用种子跟踪"，就是第二份会和磁盘分叉的真相。
        self._seed_json = self._seed_path()
        if cur is None:
            self.lbl_session_dir.configure(text=self._preview_dir())
            self.var_session_dir.set("")   # 会话目录输入框：空 = 尚无当前会话
            self.lbl_session.configure(
                text="还没有当前会话：按「录制视频」会自动新建一个目录；"
                     "要接着看以前录的，点「选择…」挑一段。",
                style="Hint.TLabel")
            return
        self.lbl_session_dir.configure(text=self._short(cur.path) + "/")
        self.var_session_dir.set(str(cur.path))   # 回填当前会话目录（只读展示）
        if cur.error:
            self.lbl_session.configure(
                text=f"当前会话 {cur.id} 有问题：{cur.error}", style="Warn.TLabel")
        elif not cur.video:
            self.lbl_session.configure(
                text=f"当前会话 {cur.id}：目录已就绪，还没录到视频"
                     f"（录制会写 clip.mkv）", style="Warn.TLabel")
        elif cur.has_result:
            self.lbl_session.configure(
                text="当前会话 " + cur.summary() +
                     "；结果在 derived/，重新检测会覆盖同名文件", style="Ok.TLabel")
        else:
            self.lbl_session.configure(
                text="当前会话 " + cur.summary() +
                     "；点「开始识别」（来源「离线」）开始检测（结果写进 derived/）", style="Warn.TLabel")

    def _vis_path(self) -> Path | None:
        """当前会话的三维可视化文件（没有就 ``None``）。"""
        cur = self._session
        if cur is None:
            return None
        p = Path(cur.path) / "derived" / "visualization.json"
        return p if p.is_file() else None

    def _sync_view3d_button(self) -> None:
        """「看 3D」只在**真有结果**时可点。

        可点却打开就报错、或点了什么都不发生的按钮，比没有这个按钮更让人
        怀疑整个界面（"是不是坏了"）。所以这里按磁盘事实启用/禁用。
        """
        btn = getattr(self, "btn_view3d", None)
        if btn is None:                     # 第③步还没建（构造顺序）
            return
        try:
            btn.state(["!disabled"] if self._vis_path() is not None else ["disabled"])
        except tk.TclError:                 # 窗口已销毁
            pass

    # ------------------------------------------------------------------ #
    # 人工种子（点选球心）
    # ------------------------------------------------------------------ #
    def _clip_path(self) -> Path | None:
        """当前会话那段原始视频；没有/坏会话 → ``None``（别让点击落到空上）。"""
        cur = self._session
        if cur is None or cur.error:
            return None
        return find_clip(cur.path)

    def _seed_path(self) -> Path | None:
        """当前会话的 ``derived/seed.json``；不存在 → ``None``。"""
        cur = self._session
        if cur is None or cur.error:
            return None
        p = Path(cur.path) / "derived" / SEED_NAME
        return p if p.is_file() else None

    def _seed_for_command(self, clip) -> tuple[str, str]:
        """这段 clip 对应的种子 → (``--seed`` 的值, 提示)；值空 = 不传种子。

        为什么按 ``clip.parent`` 找而不是 ``self._session``：走"选文件"兜底分支时
        那段视频不在任何会话里，拿会话的种子去跟踪等于让它跟别的画面。
        """
        if clip is None:
            return "", ""
        path = Path(clip).parent / "derived" / SEED_NAME
        if not path.is_file():
            return "", ""
        return seed_arg_from_file(path, image_size=self._image_size())

    def _sync_seed_button(self) -> None:
        """「点选球心」只在**会话里真有视频**时可点。

        理由同 `_sync_view3d_button`：可点却弹一句"先录一段"的按钮，
        比灰着（旁边有说明）更让人以为界面坏了。
        """
        btn = getattr(self, "btn_seed", None)
        if btn is None:                     # 第③步还没建（构造顺序）
            return
        try:
            btn.state(["!disabled"] if self._clip_path() is not None else ["disabled"])
        except tk.TclError:                 # 窗口已销毁
            pass

    def _pick_seed(self) -> None:
        """进程内打开单目点选窗，存了种子就把检测器切成 specular。

        为什么是进程内（照 `apps/_gui_stereo.pick_seeds`）：``seed_picker.py``
        只有类、没有 ``__main__``，``python apps/seed_picker.py`` 会立刻以 0
        退出、什么窗口都不开 —— 那正是"点了没反应"的形状。
        """
        clip = self._clip_path()
        if clip is None:
            self.status.set_status(
                "当前会话里没有视频：先「录制视频」，或点「选择…」挑一段已录好的",
                "warn")
            return
        from apps.seed_picker import MonoSeedPicker

        # 这里要的是 **mm**（点选窗显示的是"球直径 21.6 mm"），而
        # ``_diameter_arg`` 已经是米了 —— 直接 float 它会把球当成 0.0216 mm，
        # 那一行 Z 估计会离谱到没人信。所以取框里的原值再解析。
        diameter = diameter_mm_or_none(self.var_diameter.get())
        picker = MonoSeedPicker(self, clip.parent, fx=self._current_fx(),
                                diameter_mm=diameter)
        self.wait_window(picker)
        if not self._alive():
            return
        if picker.result is None:
            self.status.set_status(
                "没有保存种子：这段录像仍按自动检测跑（这颗球大概率检不出）", "info")
            return
        self._refresh_session()
        value, note = seed_arg_from_file(picker.result,
                                         image_size=(self._image_size()))
        if not value:
            self.status.set_status(f"种子存了但用不上：{note}", "warn")
            return
        self.var_detector.set(SEED_DETECTOR)
        self.status.set_status(
            f"种子已存：{self._short(Path(picker.result))}（球心 x,y 半径 = {value}）。"
            f"点「开始识别」就按它跟踪，检测器已切成 {SEED_DETECTOR}"
            f"（这颗球自动检测只能认到 1.4%，别再让它自己找）", "ok")

    def _image_size(self) -> tuple[int, int] | None:
        """当前会话的宽高（用来核对种子坐标是不是同一套画布）。读不到 → None。"""
        cur = self._session
        if cur is None or not cur.width or not cur.height:
            return None
        return (int(cur.width), int(cur.height))

    def _open_3d(self) -> None:
        """打开 3D 查看器。**单目也有三维**：``Z = f·D/(2r)``。

        为什么不自己画一个小窗：查看器（``apps/view_3d.py``）已经会读单目的
        ``sphere-mono-visualization``、会切会话、会同步视频。再写一份就是第二份真相。
        """
        vis = self._vis_path()
        if vis is None:
            self.status.set_status(
                "当前会话还没有三维结果：先点「开始识别」（来源「离线」）跑一遍检测，"
                "它会把 derived/visualization.json 写出来", "warn")
            return
        self._start(
            _actions.view_3d(vis),
            f"3D 查看：{self._session.id if self._session is not None else vis.parent.name}")
        self.status.set_status(
            f"已打开 3D 查看器：{self._short(vis)}（单目深度由球径先验得到，"
            f"D={self._diameter_mm_text()} mm；不是视差测量）", "info")

    def _pick_session(self) -> None:
        """历史会话：选中后回填成当前会话，后续检测/查看都用它。"""
        MonoSessionPicker(
            self, root=self._session_root,
            current=self._session.path if self._session is not None else None,
            on_pick=self._on_session_picked,
            on_root_change=self._on_root_changed)

    def _on_session_picked(self, path: str) -> None:
        if not self._alive():
            return
        self._select_session(Path(path))
        self._refresh_session()
        cur = self._session
        if cur is None:
            self.status.set_status(
                "选中的会话读不出来（目录可能刚被移走），点「选择…」刷新一下",
                "warn")
        elif cur.error:
            self.status.set_status(f"当前会话有问题：{cur.error}", "warn")
        else:
            self.status.set_status(
                f"当前会话换成 {cur.id}（{cur.state}）；"
                f"「开始识别」（来源「离线」）会直接处理它", "info")

    def _on_root_changed(self, new_root: Path) -> None:
        """点选窗里换了存放目录：当前会话作废，只记住新根。

        为什么要作废当前会话：它不在新根下了，"当前会话"若还指着旧目录，
        接下来的检测/删除都在用户看不见的地方动 —— 换目录的本意是"以后录的
        放到这里"，不是"把手上这段搬走"。
        """
        self._session_root = Path(new_root)
        self._session = None
        self._refresh_session()
        self.status.set_status(
            f"新的会话会建在 {self._short(self._session_root)}/<时间戳>/，"
            f"以前录的还在原位置（点「选择…」只看这个目录下的）", "info")

    # ------------------------------------------------------------------ #
    # 状态
    # ------------------------------------------------------------------ #
    def _open_cameras(self) -> None:
        """打开相机设备管理（复用 apps/tools/cameras.py）。

        状态条必须写清"每台最多等 30 秒"：慢相机（aoni + MSMF）实测要 ~24 s
        才给第一帧，不说这句用户会以为卡死、把窗口关掉。
        """
        self._start(
            _actions.cameras(), "相机设备管理")
        self.status.set_status(
            "已打开相机设备管理：每台最多等 30 秒（慢相机实测 ~24 秒），"
            "日志里会逐台报进度，别关窗口。只想绑左右可直接用 pin"
            "（不开相机，秒级）", "info")

    def _refresh_state(self) -> None:
        st = self.app.state
        self.lbl_cams.configure(text="相机：" + self._camera_text())
        ready, text = st.mono_status()
        clean = text.replace("**", "")
        self.lbl_calib.configure(
            text=("内参：" if ready else "内参：") + clean,
            style="Ok.TLabel" if ready else "Warn.TLabel")
        if not MONO_CALIB_APP.exists():
            self.status.set_status(
                "标定入口 apps/calibrate_mono.py 还没落地（计划书 P0c/P1b）；"
                "现在可以先拍图，标定请用「替代：单点焦距估计」或经典界面",
                "warn")
        else:
            self.status.set_status("就绪", "ok")
        self._refresh_budget()
        self._refresh_session()

    def _current_fx(self) -> float | None:
        """焦距三态：手动覆盖 > 标定值 > None（会按 60° 视场猜）。"""
        manual = self.var_focal.get().strip()
        if manual:
            try:
                v = float(manual)
                return v if v > 0 else None
            except ValueError:
                return None
        fx = (self.app.state.mono or {}).get("fx")
        return float(fx) if isinstance(fx, (int, float)) and fx > 0 else None

    def _refresh_budget(self) -> None:
        # 这一行**先写、且无论球径填没填都在**：预算的随机项（0.2 px / 0.2 mm）
        # 加起来远小于实测的 13.6% 系统偏差，只打随机项等于骗人。
        # 界面无法判断框里填的是名义值还是有效值（两个都是合法数字），
        # 所以不猜，把这条系统项作为固定事实摆在结果旁边。
        z_bias = 1.0 / (1.0 - MEASURED_RADIUS_SHRINK) - 1.0
        self.lbl_budget_sys.configure(
            text=f"半径系统偏差：填名义直径（不是有效值）→ 深度额外 +"
                 f"{z_bias * 100:.1f}%（1 m 处约 {z_bias * 1000:.0f} mm；实测）")
        # 球径三态（2026-10-31 走查第 5 条）：**空** / **不是数字** / 可用，
        # 三者要说三种话。旧版把它们混成"填一个球直径才能算误差预算" ——
        # 用户明明填了 abc，界面却说"没填"，那是把人往错方向上推。
        raw = self.var_diameter.get().strip()
        d = diameter_mm_or_none(raw)
        if not raw:
            self.lbl_budget.configure(
                text=f"球径那一栏是空的，所以算不了误差预算 —— 填毫米数，"
                     f"这颗球填 {DEFAULT_DIAMETER_MM:g}（有效值）；"
                     f"留空也行，识别时会按 {DEFAULT_DIAMETER_MM:g} mm 算",
                style="Warn.TLabel")
            return
        if d is None:
            self.lbl_budget.configure(
                text=f"球径那一栏「{raw}」不是数字，所以算不了误差预算 —— "
                     f"这里填毫米（例如 {DEFAULT_DIAMETER_MM:g}），"
                     f"不是 {DEFAULT_DIAMETER_MM / 1000:.4f}（米是命令行那边的单位，"
                     f"界面只认 mm）。改完这一行会自动重算",
                style="Warn.TLabel")
            return
        fx = self._current_fx()
        if fx is None:
            # 60° 视场、1920 宽 → fx ≈ 1663 px（geometry.py 的同一个假设）
            fx = 1663.0
            src = "焦距未标定，按 60° 视场估（fx≈1663 px）"
            style = "Warn.TLabel"
        else:
            src = f"按 fx = {fx:.0f} px 算"
            style = "Hint.TLabel"
        parts = []
        for z_m, mm in error_budget(fx, d):
            parts.append(f"{z_m:.1f} m → ±{mm:.1f} mm")
        self.lbl_budget.configure(
            text=f"误差预算（半径噪声 0.2 px、球径误差 0.2 mm、焦距 1%；{src}）："
                 + "；".join(parts),
            style=style)

    # ------------------------------------------------------------------ #
    # 相机候选（角色 + 在线设备）
    # ------------------------------------------------------------------ #
    def _initial_devices(self) -> tuple[list, str]:
        """进窗时手里能立刻拿到的设备列表，**绝不等枚举**。

        为什么（2026-10-31 实测）：``camera_registry.list_devices()`` 在没有
        pnputil 的机器上退到 PowerShell ``Get-PnpDevice``，约 10 秒 ——
        原来它在 ``__init__`` 的主线程里跑，于是点门厅「进入（单目）」之后
        整个门厅冻住、窗才出现。现在只读 30 秒内的缓存；冷的就先只列角色，
        由后台线程补上设备（见 ``_start_device_enum``）。
        """
        cached = cached_devices()
        if cached is not None:
            self._enum_pending = False
            return cached
        self._enum_pending = True
        return [], "（正在枚举相机…约 10 秒；枚举完下拉会自动补上「#编号」）"

    def _start_device_enum(self) -> None:
        """后台枚举相机，结果回主线程填进下拉。

        tkinter 的控件只能在**主线程**碰：工作线程只把结果丢进队列，主线程用
        ``after`` 轮询取出。直接在工作线程里调 ``after`` 也能跑，但那依赖实现
        细节；队列这条路在窗口被关掉时也不会往已销毁的窗口回调。
        """
        self._enum_pending = True

        def work() -> None:
            devices, note = enumerate_devices()
            store_devices_cache(devices, note)     # 缓存与窗口无关：关窗也留着
            self._enum_queue.put((devices, note))

        threading.Thread(target=work, daemon=True,
                         name="mono-enumerate-cameras").start()
        self.after(120, self._poll_device_enum)

    def _poll_device_enum(self) -> None:
        try:
            devices, note = self._enum_queue.get_nowait()
        except queue.Empty:
            if self._enum_pending and self._alive():
                self.after(120, self._poll_device_enum)
            return
        self._apply_devices(devices, note)

    def _apply_devices(self, devices: list, note: str) -> None:
        """枚举回来了：补下拉、补说明，必要时换掉"当时只能按角色选"的默认值。"""
        self._enum_pending = False
        self._devices = list(devices)
        self._device_note = note
        if not self._alive():
            return                                  # 窗口先关了：缓存仍然有效
        values = self._camera_values()
        try:
            self.cb_camera.configure(values=values)
            self.lbl_devices.configure(text=camera_hint_text(note))
        except tk.TclError:
            return
        if self._autopick and values:
            # 进窗时设备还没枚举到，默认值是按角色挑的（可能开不出画面）；
            # 现在知道哪台在线了，替用户换上更靠谱的那个。
            self.var_camera.set(self._default_camera(values))
        self._refresh_state()
        online = sum(1 for d in self._devices if getattr(d, "present", True))
        self.status.set_status(
            f"相机枚举完成：{online} 台在线" + (f"（{note}）" if note else "")
            + "；「#编号」可以直接用，不必管角色绑定", "info")

    def _camera_values(self) -> list[str]:
        """下拉候选：角色在前、在线设备在后（``#N 名字``）。

        两组靠前缀区分（角色是裸名字、设备以 ``#`` 开头），不另做分组控件 ——
        只读下拉里插不了分隔行，而"值本身就能直接传给 ``--camera``"更重要。
        """
        values: list[str] = [str(r) for r in self.app.state.camera_roles().keys()]
        for dev in self._devices:
            if not getattr(dev, "present", True):
                continue
            label = f"#{dev.order} {getattr(dev, 'name', '') or '未命名设备'}"
            if label not in values:
                values.append(label)
        return values

    @staticmethod
    def _camera_arg(text: str) -> str:
        """下拉值 → ``--camera`` 的参数：``#3 名字`` 只传 ``3``，角色原样传。

        ``resolve_camera_arg()`` 的规则是"纯数字按索引、其余按角色名"，
        所以把 ``#0 aoni ...`` 整串传下去会被当成角色名，报"配置里没有角色「#0 ...」"。
        """
        text = (text or "").strip()
        if text.startswith("#"):
            num = text[1:].split(" ", 1)[0]
            return num if num.isdigit() else ""
        return "" if text.lower() in ("auto", "none", "自动") else text

    def _default_camera(self, values: list[str]) -> str:
        """默认选中：``main`` 角色 → 第一个在线设备 → 第一个角色。

        为什么第二个才轮到设备、而不是直接选 ``left``：``left`` 现在指向一台
        枚举不到的机器（2026-10-29 现场：绑定的是 "aoni webcam A25"，而在线的是
        "aoni webcam A25 left"）。默认选中它，等于一进窗口就递上一个开不出画面的选项。
        """
        if "main" in values:
            return "main"
        for value in values:
            if value.startswith("#"):
                return value
        return values[0] if values else ""

    @staticmethod
    def _match_saved_camera(saved: str, values: list[str]) -> str:
        """把上次存的值对回当前候选：先整体匹配，再按 ``#N`` 认编号。

        设备名字会变（换端口/改驱动），``#0`` 这个编号通常还在 ——
        认不出名字时退回编号，比把选择清空好（清空 = 又得重新想选哪台）。
        """
        saved = (saved or "").strip()
        if not saved:
            return ""
        if saved in values:
            return saved
        if saved.startswith("#"):
            num = saved[1:].split(" ", 1)[0]
            for value in values:
                if value == f"#{num}" or value.startswith(f"#{num} "):
                    return value
        return ""

    def _role_online(self, role: str) -> bool | None:
        """角色指向的那台现在在不在线；``None`` = 判断不了（别当成离线）。

        用 ``resolve_role(devices=已枚举的表)``：它只做身份/端口的静态匹配，
        **不开相机**，所以是毫秒级。``how == "index-fallback"`` 说明身份没对上、
        只是蒙了个旧编号 —— 按"不在线"报比让用户以为选对了强。
        """
        try:
            from sphere_tracker import camera_registry as cr
            resolved = cr.resolve_role(role, devices=self._devices, interactive=False)
        except Exception:                                   # noqa: BLE001
            return False if self._devices else None
        return resolved.how != "index-fallback"

    def _camera_text(self) -> str:
        """第 ① 步那行"相机：…"——**要能看出选中的这台开不开得出来**。"""
        cam = self.var_camera.get().strip()
        if not cam:
            return "自动挑（优先外接）"
        if cam.startswith("#"):
            return f"{cam}（按索引，不依赖角色绑定）"
        try:
            base = self.app.state.camera_role_text(cam)
        except Exception:                                   # noqa: BLE001
            return f"{cam}（读 config/cameras.json 失败）"
        online = self._role_online(cam)
        if online is False:
            return base + "｜这台当前不在线，换「#编号」那个选项或重新 pin"
        if online is None:
            # 设备还没枚举到：不知道在不在线，就别装作知道（也别说"在线"）
            return base + "｜设备还没枚举到"
        return base + "｜未开相机验证"

    # ------------------------------------------------------------------ #
    # 动作
    # ------------------------------------------------------------------ #
    def _start(self, cmd: list[str], title: str, on_done=None):
        """本窗起任务的**唯一入口**：打归属标记 + 接统一结束回调。

        为什么收成一个函数（2026-10-31 B3 的根子）：以前每个按钮自己决定
        传不传 ``on_done``，「拍标定图」「替代：单点焦距估计」「实时识别」
        「看 3D」「换相机」都没传 —— 点完之后状态条永远停在"已开始…"，
        失败也只写进日志。现在按钮只管拼命令，**结局**由 ``_on_task_finished``
        统一写；以后加按钮不可能再漏接回调。

        归属是起完之后**填在任务上**的（``Task.owner``），不是 ``start()`` 的
        参数：走查工具与 gui_budget 都会用自己的桩替换 ``start`` 并保持老签名
        （``start(cmd, title, on_done)``），多一个关键字参数会让它们当场
        ``TypeError`` —— 那是"修好了界面、打死了验证工具"。桩返回的不是
        ``Task``，所以这里按类型判定，不硬塞属性。
        """
        task = self.app.tasks.start(cmd, title, on_done=on_done)
        if isinstance(task, Task):
            task.owner = OWNER
        return task

    def _log_hint(self, task) -> str:
        """任务日志的**短**路径：状态条只有窗口那么宽，绝对路径会把原因挤没。

        状态条的标签不会换行（见 ``_gui_widgets.StatusBar``），实测 880 px 的窗口
        只放得下约 95 个汉字。写成 ``D:\\MY_CODING_PROGRAM\\...\\outputs\\logs\\``
        那种绝对路径时，"为什么失败"会被挤出血条外 —— 而日志文件里写的仍是
        绝对路径，两边不矛盾，只是这里显示仓库内的相对路径。
        """
        p = getattr(task, "log_path", None)
        return f"日志 {self._short(p)}" if p else "日志没落盘"

    def _on_task_finished(self, task) -> None:
        """TaskRunner 的统一出口：**成功与失败都会来**。

        三种情况分开说，因为用户要的下一句话不同：
        成功 → 结果在哪；被用户停掉 → 就说停掉了（别报成失败）；
        失败 → 返回码 + **最后几行输出** + 日志路径（"去看日志"本身不是原因）。
        有 ``on_done`` 的任务由它自己说（它知道更多上下文），这里只兜没有的。
        """
        if not self._alive() or getattr(task, "owner", "") != OWNER:
            return
        if task.ok:
            if task.on_done is None:
                self.status.set_status(
                    f"「{task.title}」完成（返回码 0）：{task.tail_text(1)[:90]}"
                    f" —— {self._log_hint(task)}", "ok")
            return
        if task.stop_reason:
            self.status.set_status(
                f"「{task.title}」{task.stop_reason}（返回码 {task.exit_code}），"
                f"没跑完就没结果 —— {self._log_hint(task)}", "warn")
            return
        self.status.set_status(
            f"「{task.title}」失败（返回码 {task.exit_code}）："
            f"{task.tail_text(2)[:120]} —— {self._log_hint(task)}", "warn")

    def _common_camera_args(self) -> list[str]:
        """``--camera`` 参数：空值必须落成具体编号，``#N`` 只传 N。

        不能传空串：``capture_board`` 的 ``resolve_camera_arg(None)`` 返回 None、
        不自动挑，会直接报"没能把 None 解析成相机索引"（2026-10-28 踩过）。
        """
        cam = self._camera_arg(self.var_camera.get())
        return ["--camera", cam or "0"]

    def _start_capture(self) -> None:
        CALIB_FRAME_DIR.mkdir(parents=True, exist_ok=True)
        cmd = _actions.capture_board(
            out=CALIB_FRAME_DIR,
            cols=self.var_cols.get().strip() or "8",
            rows=self.var_rows.get().strip() or "5",
            square=self.var_square.get().strip() or "25",
            target=self.var_target.get().strip() or "15",
            extra=self._common_camera_args())
        self._start(cmd, "拍单目标定图")
        self.status.set_status(
            f"已开始拍图：板在每个姿态停稳约 1 秒就会自动存；"
            f"存够 {self.var_target.get() or '15'} 张自动结束。图在 "
            f"{CALIB_FRAME_DIR.name}\\", "info")

    def _start_calibrate(self) -> None:
        if not MONO_CALIB_APP.exists():
            self.status.set_status(
                "标定脚本还没落地：现在可以用「替代：单点焦距估计」，"
                "但它假设主点在画面中心、不产生畸变系数，球径误差会并入 fx", "warn")
            return
        cmd = _actions.calibrate_mono(
            frames_dir=CALIB_FRAME_DIR,
            cols=self.var_cols.get().strip() or "8",
            rows=self.var_rows.get().strip() or "5",
            square=self.var_square.get().strip() or "25")
        self._start(cmd, "单目内参标定", on_done=self._on_calibrated)
        self.status.set_status("正在标定内参…", "info")

    def _on_calibrated(self, task) -> None:
        """标定结束：**只有成功才回去刷新内参**。

        旧版不看返回码，失败也 ``_refresh_state()`` —— 而它会把状态条写成
        "就绪"，等于把"标定失败"报成"一切正常"（2026-10-31 走查第 B3 条）。
        失败必须留在失败上：给返回码、日志尾、日志路径，三者缺一用户就得自己猜。
        """
        if not self._alive():
            return
        if not getattr(task, "ok", False):
            self.status.set_status(
                f"内参标定失败（返回码 {task.exit_code}）：{task.tail_text(2)[:100]} "
                f"—— 常见原因是标定图不够或姿态太单一（要 8 个以上不同角度）；"
                f"{self._log_hint(task)}", "warn")
            return
        self.app.state.reload()
        self._refresh_state()          # 这一步会把状态条写回"就绪"，所以随后立刻覆盖
        self.status.set_status("内参标定完成，已重新读取标定值（第①步那行会跟着变）", "ok")

    def _start_focal_estimate(self) -> None:
        """单点焦距估计：**球径必须传框里那个值**，不能让它吃库里的默认值。

        ``calibrate_focal`` 解的是 ``f = 2·r_px·Z/D``。用户在框里填的 D 就是这套
        检测器的有效直径；这个入口若按别的 D 算（比如库默认），解出来的 f 会
        直接错一个比例（换球后 30 / 21.6 = 1.39 倍）—— 那不是误差项叠加，
        是拿错了输入。
        """
        cmd = _actions.calibrate_focal(diameter_m=self._diameter_arg(),
                                       extra=self._common_camera_args())
        self._start(cmd, "单点焦距估计")
        note = diameter_problem(self.var_diameter.get())
        self.status.set_status(
            f"单点焦距估计会打开一个取景窗：把球放在已知距离上点一下。"
            f"它按框里的有效直径 {self._diameter_mm_text()} mm 解 fx"
            f"（命令行传的是米：{self._diameter_arg()} —— 两个数说的是同一颗球），"
            "并把这次估计的出处一起写进配置"
            + (f"。注意：{note}" if note else ""), "warn")

    def _open_frames_dir(self) -> None:
        try:
            CALIB_FRAME_DIR.mkdir(parents=True, exist_ok=True)
            import os
            os.startfile(str(CALIB_FRAME_DIR))           # noqa: S606
        except Exception as exc:                        # noqa: BLE001
            self.status.set_status(f"打不开文件夹：{exc}", "warn")

    def _diameter_arg(self) -> str:
        """框里的球径 → 命令行值（**米**；留空/写坏退回有效直径）。

        口径与界面**故意不同**：界面按 mm 收、按 mm 显示，"进命令行"这一步
        才换成米（``--ball-diameter`` 要米）。换算只在模块级 ``diameter_arg``
        里做一次，窗口这边只负责把框里的字交给它。
        """
        return diameter_arg(self.var_diameter.get())

    def _diameter_mm_text(self) -> str:
        """框里球径的 **mm** 文案（给状态条用）——和上面那个是同一颗球的两个单位。"""
        return diameter_text_mm(self.var_diameter.get())

    def _focal_arg(self) -> tuple[int | None, str]:
        """(命令行要传的 fx, 给状态条的提示)。

        三种输入分开对待：留空 → 不传（标签就写着"留空用标定值"）；
        正数 → 传；**填了但不是正数 → 不传，但要说一句**。第三种若也静默，
        用户"填了 1500"和"什么都没填"在界面上完全一样 —— 与"显示 21.6 却传 40"
        是同一类静默失效，只是这次失效的是 fx。
        """
        raw = self.var_focal.get().strip()
        if not raw:
            return None, ""
        fx = self._current_fx()
        if fx is None:
            # 说清"退回去的是哪个数"：只说"仍用标定值"在没标定的机器上等于没说
            cal = (self.app.state.mono or {}).get("fx")
            tail = (f"标定值 fx={float(cal):.0f}"
                    if isinstance(cal, (int, float)) and cal > 0
                    else "标定值（当前没有标定，会按 60° 视场估）")
            return None, f"fx 那一栏「{raw}」不是数字，已忽略，仍用{tail}"
        return int(fx), ""

    def _method_arg(self) -> str:
        """下拉里的检测器 → ``--method`` 的值。

        ``auto`` 不是注册表里的检测器（``run_video`` 的 choices 里没有它），
        所以命令行起步只能给 ``color``；真正的"自动换"由 ``_on_detected``
        看到退出码 2 之后再做（见 ``_start_video``）。
        """
        name = (self.var_detector.get() or "").strip()
        if name not in detector_choices():
            # 存过的旧值/外部改坏：回落默认，但**不许静默**（下面会写进状态条）
            return "color"
        return "color" if name == SMART else name

    def _start_realtime(self) -> None:
        # 没有球径就不启动：`run_video` 会直接拒跑（Z = f·D/(2r) 对 D 1:1），
        # 让它跑到一半再报错，用户只会看到"进程退出、没结果"。
        problem = diameter_problem(self.var_diameter.get())
        if problem:
            self.status.set_status(problem + " —— 已取消启动", "error")
            return
        # 检测器必须显式传：界面选了却不传，等于"显示 hough、实际跑 color"，
        # 与"显示 21.6 却传 40"是同一类静默失效。
        cmd = _actions.run_realtime(diameter_m=self._diameter_arg(),
                                    method=self._method_arg(),
                                    preset=self.ball.preset())
        cmd += self._common_camera_args()
        fx, fx_note = self._focal_arg()
        if fx is not None:
            cmd += ["--focal", str(fx)]                 # 只有手动覆盖才传
        if MONO_CALIB_JSON.exists():
            cmd += ["--config", str(MONO_CALIB_JSON)]
        self._start(cmd, "实时单目识别")
        auto_note = ("。auto 在实时里退化成 color（实时没有录像可以重跑）"
                     if self.var_detector.get() == SMART else "")
        d_note = diameter_problem(self.var_diameter.get())
        self.status.set_status(
            ("已启动实时识别（球径按 "
             f"{self._diameter_mm_text()} mm 传给命令行 {self._diameter_arg()} m），"
             "画面在它自己的窗口里；要停请到「任务管理器和运行日志」里中断"
             + (f"。注意：{fx_note}" if fx_note else "")
             + (f"。注意：{d_note}" if d_note else "") + auto_note),
            "warn" if (fx_note or d_note) else "info")

    def _start_video(self) -> None:
        """离线识别：有当前会话就直接处理它，没有才让用户选文件。

        为什么优先当前会话：会话管理存在的意义就是"不用再记文件在哪"。
        结果写回同一个会话的 derived/ —— 于是"有没有跑过检测"变成**看一眼
        摘要就知道**的事实，而不是靠人记。

        2026-10-29 增两件事：**显式传检测器**（界面选了 color 却跑别的、
        或反过来，都是静默失效）与 **``--vis-json``**（单目本来就有三维，
        以前只是从来没落盘，于是"主界面只能看双目"看起来像"单目没有 3D"）。

        2026-10-30 增：当前会话有 ``derived/seed.json`` 时带 ``--seed x,y,r``，
        并把检测器**强制**成 specular —— 那颗镜面球自动检测只能认到 1.4%，
        用户点了球还让它自己找，等于白点。
        """
        clip = find_clip(self._session.path) if self._session is not None else None
        # 与实时同理：没有球径就先把话说清，不要起一个注定要退出的进程。
        problem = diameter_problem(self.var_diameter.get())
        if problem:
            self.status.set_status(problem + " —— 已取消启动", "error")
            return
        # 种子在**产生命令行之前**定：因为它会改写检测器（specular），
        # 而 --method 跟着检测器走。放在后面拼，就是"带了种子却跑了别的检测器"。
        seed_value, seed_note = self._seed_for_command(clip)
        if seed_value:
            # 有种子 = 这颗球是镜面球（只有 specular 认 --seed）：球种类跟着走，
            # 否则"球种类"那一栏还显示纯色球，与命令行矛盾。
            self.ball.set_kind(BALL_METAL)
            self.var_detector.set(SEED_DETECTOR)     # 下拉框同步显示，不留两套口径
        method = self._method_arg()
        auto = self.var_detector.get() == SMART
        fx, fx_note = self._focal_arg()
        preset = self.ball.preset()
        if clip is not None:
            derived = clip.parent / "derived"
            cmd = _actions.run_video(
                clip, diameter_m=self._diameter_arg(), method=method,
                out=derived / "annotated.mp4", csv=derived / "traj.csv",
                vis_json=derived / "visualization.json", preset=preset)
            if seed_value:
                cmd += ["--seed", seed_value]
            if fx is not None:
                # 与实时那边同口径：框里填了就传，不许静默不生效。
                cmd += ["--focal", str(fx)]
            self._auto_retry_pending = auto
            self._start(cmd, f"离线单目：{clip.parent.name}（{method}）",
                        on_done=self._on_detected)
            d_note = diameter_problem(self.var_diameter.get())
            # 选了镜面球却还没点过球心：specular 没有种子 = 很可能 0 检出。
            # 提示放在"已经启动"的同一句话里，而不是拦着不让跑（用户可能是
            # 想先用 specular 试一段，或种子文件在别处）。
            metal_note = ("选了镜面金属球但这段还没有「点选球心」："
                          "specular 只在高光附近找，没有种子多半 0 检出"
                          if (self.ball.kind() == BALL_METAL and not seed_value) else "")
            self.status.set_status(
                f"已启动离线单目：{self._short(clip)}（检测器 {method}）；结果写进 "
                f"{self._short(derived)}/（跑完摘要会变成「已检测」，"
                f"那时才能点「看 3D」）"
                + ("。auto：若一帧都没检出会自动换 hough 重跑" if auto else "")
                + (f"。用你点的种子跟踪（{seed_value}），不再自动检测" if seed_value else "")
                + (f"。注意：{fx_note}" if fx_note else "")
                + (f"。注意：{d_note}" if d_note else "")
                + (f"。注意：{seed_note}" if seed_note else "")
                + (f"。注意：{metal_note}" if metal_note else ""),
                "warn" if (fx_note or seed_note or d_note or metal_note) else "info")
            return

        # 兜底走文件选择框：run_video.py 的 path 是必填位置参数，它自己不会弹
        # 选择框 —— 盲启的结果是 exit 2、点了没反应（2026-10-28 踩过）。
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="选择要处理的视频", parent=self,
            filetypes=[("视频", "*.mp4 *.avi *.mkv *.mov *.m4v"),
                       ("所有文件", "*.*")])
        if not path:
            self.status.set_status("没有选视频，已取消", "info")
            return
        cmd = _actions.run_video(path, diameter_m=self._diameter_arg(),
                                 method=method, preset=preset)
        if fx is not None:
            cmd += ["--focal", str(fx)]
        # 不在会话里的文件也给 --vis-json，落在它旁边（用户明确选了文件，
        # "看 3D"按钮指不到它，但产物本身得留下）
        side = Path(path).with_name("visualization.json")
        cmd += ["--vis-json", str(side)]
        self._auto_retry_pending = auto
        self._start(cmd, f"离线单目：{Path(path).name}（{method}）",
                    on_done=self._on_detected)
        d_note = diameter_problem(self.var_diameter.get())
        self.status.set_status(
            f"已启动离线单目：{Path(path).name}（检测器 {method}；这个文件不在"
            f"会话目录里，结果不会写进任何 derived/，可视化落在 {side.name}）"
            + (f"。注意：{fx_note}" if fx_note else "")
            + (f"。注意：{d_note}" if d_note else ""),
            "warn" if (fx_note or d_note) else "info")

    def _on_detected(self, task=None) -> None:
        """离线检测结束：按退出码说话，再重读会话摘要。

        **退出码是这段唯一的结论**：``run_video`` 在 0 检出时返回 2（并且不写
        CSV/可视化/标注视频）。以前这里不看退出码，于是"0 检出"被报成
        "derived/ 里没看到结果，去看日志" —— 用户得自己翻日志才知道
        是检测器选错了。现在当场说清楚，并（auto 时）自动换 hough 重跑。
        """
        if not self._alive():
            return
        code = getattr(task, "exit_code", None)
        if self._auto_retry_pending and code == 2:
            # auto 的全部意义就在这里：用户不知道"这颗球 color 看不见"，
            # 界面替他试一次 hough，而不是让他自己去读那段诊断。
            self._auto_retry_pending = False
            self.var_detector.set("hough")
            self.status.set_status(
                "color 一帧都没检出（退出码 2），auto 已自动换 hough 重跑一遍…", "warn")
            self._start_video()
            return
        self._auto_retry_pending = False
        self._refresh_session()
        cur = self._session
        if cur is not None and cur.has_result:
            self.status.set_status(
                f"{cur.id}：结果已写进 derived/"
                f"（annotated.mp4 + traj.csv + visualization.json）；"
                f"点「看 3D」看图", "ok")
        elif code == 2:
            self.status.set_status(
                "一帧都没检出（退出码 2）：检测器不适合这颗球。"
                "把「检测器」换成 hough（本机实测 95%）或 fusion 再跑一次；"
                "csv/可视化/标注视频都没有写，别把它们当成结果", "warn")
        elif code not in (0, None):
            # 非 0 也不是 2：程序自己报错了。**要说清原因和日志在哪** ——
            # 只写"去看日志"等于把排查的第一步也交给用户。
            self.status.set_status(
                f"离线检测失败（返回码 {code}）：{task.tail_text(2)[:110]} "
                f"—— {self._log_hint(task)}", "warn")
        else:
            self.status.set_status(
                "检测任务返回 0，但 derived/ 里没看到结果："
                "多半是这段录像在 0 检出与成功之间（去看看日志末几行），"
                "换 hough 再跑一次最省事", "warn")

    def on_close(self) -> None:
        """关窗时记住输入，并**退订统一结束回调**。"""
        # 先退订：任务跑得比窗口久（录制 10 秒、检测几十秒），关窗后回调还会来，
        # 那时碰控件就是对着已销毁的窗口调 Tcl（回调里虽有 _alive 兜着，
        # 但"该退订就退订"比"靠每个回调自己记得检查"可靠）。
        self.app.tasks.remove_finish_listener(self._on_task_finished)
        p = self.app.prefs
        p.set("mono.camera", self.var_camera.get().strip())
        p.set("board.cols", self.var_cols.get().strip())
        p.set("board.rows", self.var_rows.get().strip())
        p.set("board.square_mm", self.var_square.get().strip())
        p.set("mono.target", self.var_target.get().strip())
        p.set("mono.diameter_mm", self.var_diameter.get().strip())
        p.set("mono.focal_override", self.var_focal.get().strip())
        p.set("mono.record_seconds", self.var_record_secs.get().strip())
        # 检测器与来源是**输入**（用户的选择），要记住：这颗球 color 看不见，
        # 每次开窗都回到 color 等于每次都让他再撞一遍墙。
        p.set("mono.detector", self.var_detector.get().strip())
        p.set("mono.preset", self.var_preset.get().strip() or PRESET_NONE)
        p.set("mono.source", self.var_source.get().strip())
        # 只存"用户换过的根目录"这个输入；当前选中的会话属于磁盘现状，不存
        p.set("mono.session_root", str(self._session_root))
        p.save()

    def _stop_my_tasks(self) -> None:
        ok = self.app.tasks.stop_all(graceful=True)
        self.status.set_status(f"已确认停掉 {ok} 个任务", "ok" if ok else "warn")
