"""双目模式窗（``parallel`` / ``angled`` 共用一份代码，由 ``mode`` 参数化）。

为什么"两个窗口"不是"两套皮肤"（2026-10-27 计划书 §2.1）
--------------------------------------------------------
平行与大角度的区别不在外观，而在**几何前提与标定方法**：

======================  ====================  ==========================
                        parallel               angled
======================  ====================  ==========================
深度来源                 校正后视差            原始像素三角测量
标定方法                 joint（必须共视）      joint+跳过 FOV 或 chain
能否两路看不到同一区域    不能                   能（chain）
``--stereo-mode``        parallel               angled
======================  ====================  ==========================

所以本模块**只有一份实现**：``mode`` 只影响三处 —— 第 2 步能选的方法、
第 2 步的前提文案、第 4 步命令行里的 ``--stereo-mode``。
**窗口里没有模式下拉框**：模式 = 你进的哪个窗口（旧界面那个顶栏下拉框是
``stereo_mode`` 的第三份真相，已删除）。

用户问过的那件事（2026-10-28）：单板链式标定与双目工作台是什么关系？
答案写在 ``angled`` 窗的第 2 步：chain 的产物 meta 记 ``stereo_mode=angled``，
第 4 步直接用 —— 它们本来就是一条链路，只是旧界面把两者摆成了两张互不相干的卡。
"""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from apps import _fov_store

from apps._gui_widgets import (BALL_METAL, BALL_PLASTIC, METHOD_METAL,
                               PRESET_NONE, BG, INK, MUTED, WARN, BallSettings,
                               ScrollPage, StatusBar, Surface, fit_to_content,
                               kind_key)
from apps._collapsible import CollapsibleSection
# argv 的唯一产地在 `_actions`（见该模块 docstring）：这里只 import 名字，
# 不再自己拼命令行 —— 否则同一个动作会有两份参数清单，已经因此出过三次缺陷。
from apps._actions import (CHAIN_ROOT, board_args, calibrate_args,
                           calibrate_axial_args, calibrate_device90_args,
                           chain_dirs, check_chain_inputs, check_setup_args,
                           capture_chain_frames, capture_pair_frame,
                           current_chain_dirs, new_chain_session,
                           offline_args, record_args)
from apps import _actions
from apps._stereo_layout import (
    MODE_LABELS,
    candidate_session_dirs,
    effective_mode,
    is_legacy_flat,
    layout_mode_of,
    mode_label,
    session_mode_of,
)
from sphere_tracker.config import ColorConfig
from sphere_tracker.paths import STEREO_ROOT, stereo_mode_root, stereo_roots
from sphere_tracker.plugins import names as plugin_names

PROJECT_ROOT = Path(__file__).resolve().parents[1]

__all__ = ["StereoWindow", "SessionPicker", "scan_sessions", "pick_seeds",
           "board_args", "chain_dirs", "calibrate_args", "record_args",
           "record_out_dir", "check_setup_args", "session_name_of",
           "realtime_session_blocker",
           "detector_methods", "preset_choices", "METHODS", "PICKER_METHODS",
           "picker_methods_for"]


def detector_methods() -> tuple[str, ...]:
    """双目/工作台的检测方法候选：**注册表里的全部实现**，一个不多一个不少。

    为什么不在界面里写一份名字清单（R7）：这份清单以前在手抄的四份里漂过
    （GUI 7 项 / ``run_stereo_video`` 5 项 / ``run_all`` 4 项）。判据
    （``tests/selftest_gui_budget.py``）断言的就是"下拉项 == 注册表"，
    所以这里**只准**转发，不许过滤、不许补字面量。

    不带单目那个 ``smart``：它是单目窗自己的"0 检出换 hough 重跑"行为，
    ``run_stereo_video --method`` 的 choices 里没有它。
    """
    return tuple(plugin_names("detector"))


def preset_choices() -> tuple[str, ...]:
    """取色候选：唯一来源 ``ColorConfig.preset_choices()``（含 auto）。"""
    return tuple(ColorConfig.preset_choices())


#: 方法在界面上的说法。键与 `calibrate_stereo.py --method` 一致（chain_dual 是
#: 界面专用键，底层仍是 chain，只是双板模式：--board-b-dir + --pair-frame）。
#:
#: **每条文案必须说清"需要多少共视"**。三条路的差别就在这一点上，而它决定
#: 用户该选哪条、以及他在自己的装置上到底能不能拍出来：
#:
#:   joint      每个姿态都要共视          -> 需要**大**共视区（近平行/充分重叠）
#:   chain 单板  仅参考图那一处要共视      -> 需要**小**共视区（两路都能看到整块板）
#:   chain 双板  零共视（同框图单相机拍）   -> 真 90° 装置唯一可行的路
#:
#: 历史上 `chain` 的说明写成"90 度这类几乎看不到同一区域的装置用它"——**是错的**。
#: 它的参考段要求两路各自完整检出整块棋盘（`_ref_board_poses` 检不出就抛错），
#: 光轴垂直时一块平板从侧向只能看到边缘，那个位置根本不存在。
METHODS = {
    "joint": ("两路共视（joint）",
              "两路在**每个姿态**都要同时看到同一块板。只有两路重叠区足够大时才成立"
              "（近平行、或夹角不大）。"),
    # 仅保留给既有调用方/CLI：大角度窗的 mode 本来就是 angled，它把
    # --stereo-mode 再写死一遍，产出的 argv 与 joint **逐字相同**；平行窗又
    # 没有方法下拉。也就是说它在任何窗口都是空操作，已从下拉里移除。
    "joint_skip_fov": ("两路共视，跳过视场裁剪（joint + no FOV）",
                       "（已并入「两路共视」：angled 模式本来就跳过 FOV 对齐，"
                       "这一项与它产出的命令完全相同。）"),
    "chain": ("单板链式（chain）",
              "各拍一组内参图，再在**一处两路都能看到整块板**的位置拍参考图。"
              "仍需一小块共视区——真 90° 装置通常没有这样的位置，"
              "那种情况请用下面的双板。"),
    "chain_dual": ("双板链式（两板固定在同一支架）",
                   "① 左相机拍 A 板 ② 右相机拍 B 板（板随便挪）③ 支架摆好别动、"
                   "两路同时各拍 3 张 ④ 一台相机同时拍到两板。"
                   "**全程不需要两路看到同一块板**；第 ③ 步不能省。"),
    "axial": ("轴向大板（前后推）",
              "板正对相机、只沿光轴推进已知距离。把图按深度分目录："
              "<根目录>/z000mm/{left,right}/、<根目录>/z010mm/{left,right}/…，"
              "标定时自动从目录名解析推进量（也支持 z000/z010mm/0mm/10 等写法）。"
              "采集端不用改：现有怎么拍就怎么拍，拍完按深度归档即可。"),
    "device90": ("中轴90°装置（每相机各一次）",
                 "板 A 正对相机 1 拍一组（①），绕板的中轴转 90° 后正对相机 2 拍一组（②）。"
                 "不需要同框图（③④跳过）。需要已知内参（轴向标定产物或既有标定）和预期基线；"
                 "预期基线务必填机械实测值，装反检测只靠它。"),
}

#: 下拉里**真正列出来**的键（顺序即展示顺序）。
#: `joint_skip_fov` 不在这里：它是空操作，列出来只会让人以为两种选法有区别。
PICKER_METHODS = ("joint", "chain", "chain_dual")


def picker_methods_for(mode: str) -> tuple[str, ...]:
    """按窗口模式给方法下拉项：**轴向只进平行窗，中轴90°只进大角度窗**。

    两种新标定绑定各自的装置几何：轴向大板是平行双目（板正对、沿光轴推），
    中轴 90° 是大角度（板绕中轴转 90°）。把它们同时塞进两个窗口的下拉，
    用户就会在平行窗里选到大角度算法（或反之），拿到一份几何前提根本不成立的
    结果。所以下拉按模式过滤，全集仍是 `PICKER_METHODS`（判据可按它核对）。
    """
    if mode == "parallel":
        return ("joint", "axial")
    return ("chain", "chain_dual", "device90")


# --------------------------------------------------------------------------- #
# 命令行拼装：界面取值 → argv，**不依赖 Tk** 的纯函数
# --------------------------------------------------------------------------- #
# 为什么抽成模块级函数（2026-10-29）
# ----------------------------------
# 这两段拼装原来内联在 `_start_calibrate` / `_start_record` 里，而
# `tests/selftest_gui_budget.py` 只守控件数量与文案 —— 也就是说**新窗口真正
# 干活的那几行没有任何判据看着**："界面上选的板参数有没有正确进到
# calibrate_stereo 的命令行"，只能靠人眼看一遍。
# 老窗口（`apps/gui_main.py` 的 `_calibrate_args` / `_record_args`）里同一件事有
# 7 节判据守着，新窗口却是空的 —— 这才是必须补的缺口。
# 抽出来的函数不碰 Tk、不读控件，判据才能直接调它；**行为与抽之前逐字等价**，
# 抽的时候没有顺手"修正"任何默认值。










def record_out_dir(mode: str | None = None, root: Path | None = None) -> Path:
    """录制的 ``--out``：**该装置几何的分区根**（时间戳子目录由录制程序建）。

    为什么是根、不是 ``<根>/<时间戳>``（2026-10-31 修，缺陷 #3）
    ----------------------------------------------------------
    ``record_stereo.py`` 的 ``--out`` 是**根**：它自己调
    ``stereo.sessions.create_session(root)``，在 root 下新建
    ``YYYYMMDD-HHMMSS``（同名已存在则加 ``-01`` 后缀）。传具体会话目录
    反而会再套一层 —— 除非在这里另写一份命名与去重逻辑，那就成了
    "目录名有两处来源"。

    为什么要按 mode 分区（2026-11-04，用户报"两种装置的会话没隔离"）
    ------------------------------------------------------------
    以前所有会话都落 ``data/stereo/<时间戳>``，平行和 90° 装置物理混在一起，
    只能靠 ``session.json`` 的 ``stereo_mode`` 在列表里过滤 —— 而 29/43 段
    老会话没有这个字段，于是两个窗口都列得出来。现在按窗口的装置几何落到
    ``data/stereo/<mode>/<时间戳>``：**哪段属于哪套装置成了目录结构本身**。

    ``mode`` 不传（或非法）= 落旧扁平根 ``data/stereo``。这条兜底只给测试和
    老调用方用；界面一律传本窗的 mode（``parallel``/``angled``）。
    """
    if root is not None:
        return root
    try:
        return stereo_mode_root(str(mode or ""))
    except ValueError:
        return STEREO_ROOT


def session_name_of(visualization: Path | str) -> str:
    """一份 ``visualization.json`` 属于**哪一段会话**。

    为什么不能直接 ``path.parent.name``（2026-10-31 走查缺陷 #6/#7）：
    离线重建把结果写在 ``<会话>/derived/visualization.json``，于是
    ``parent.name`` 永远是 ``derived`` —— 状态条说"正在打开 3D 查看器：derived"，
    用户看不出打开的是哪一段。``derived`` 是会话里的**子目录**，不是会话。
    """
    p = Path(visualization).parent
    return p.parent.name if p.name == "derived" else p.name


def realtime_session_blocker(session_dir) -> str:
    """这段会话能不能跑第④步「跑离线三维」：非空 = 不能（人话原因），空 = 能跑。

    实时双目（``apps/run_realtime_stereo.py``）退出时会把这段实时跑出来的东西
    落成一段**普通会话目录**（``session.json`` + ``derived/traj3d.json`` 等），
    用户因此能在数据管理里看到它、在 3D 查看器里打开它 —— 那正是要的效果。
    但它**没有原始左右录像**（``left.mkv`` / ``right.mkv``），而「跑离线三维」
    的入口是 ``run_stereo_video.main``：它会先 ``validate_session`` 再看视频，
    于是用户点了按钮、等一次进程启动，才收到一条"缺原始视频"的报错。
    能在这里说清就不要让人白等 —— 这就是本函数的唯一职责。

    判据只读 ``<session_dir>/session.json`` 的 ``source`` 字段：
    - ``"realtime"`` → 拦；
    - 没有这个键 / 读不到 / 不是 JSON 对象 → 当普通会话放行（返回空串）。
      这里**不能**用 ``session.json`` 缺失来推断"实时" —— 坏目录也是缺文件的，
      放行后 ``run_stereo_video`` 自己会报得更具体（"缺 session.json"）。

    为什么不走 ``stereo.sessions.load_session``：它的 ``Session`` dataclass 没有
    ``source`` 字段，未知键会被静默丢掉 —— 从那里读永远是空串。
    """
    try:
        raw = json.loads((Path(session_dir) / "session.json").read_text(
            encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    if not isinstance(raw, dict) or raw.get("source") != "realtime":
        return ""
    return ("这是实时采集的会话（没有原始左右录像），不能重跑三维。"
            "要能重跑请在第③步用「开始录制」录一段。")




# --------------------------------------------------------------------------- #
# 会话选择：三态过滤（匹配 / 不匹配 / 未记录）
# --------------------------------------------------------------------------- #
def scan_sessions(base: Path | tuple[Path, ...] | None = None, *,
                  mode: str | None = None) -> list[dict]:
    """读双目会话的摘要：凡是 ``session.json`` 所在的目录都算一段，**层级不限**。

    实际会遇到的三种摆法：``data/stereo/parallel/<时间戳>``（按装置分区）、
    ``data/stereo/<时间戳>`` / ``data/sessions/<时间戳>``（2026-11-04 分区改造
    之前的扁平布局）、以及用户自己建的 ``data/stereo/实验1/梨子/<时间戳>``。
    最后一种在 2026-11-09 之前是**看不见**的（枚举只认两层），现在统一由
    ``sphere_tracker.discovery.discover_session_dirs`` 递归发现。

    只读 JSON、**不构造 ``Session`` 对象**：几十个会话也能瞬间列完，
    而且不会因为某一份坏文件把整个列表打断。

    ``base`` 不传 = 扫 ``stereo_roots()``，迁移前后都列得全；
    传一个目录/tuple 只扫那些（测试要造临时会话，不能依赖这台机器录过什么）。
    去重按**resolved 路径**（上游 ``candidate_session_dirs`` 已做，同一个物理
    目录只出现一次），**不按目录名** —— ``parallel/<时间戳>`` 与
    ``angled/<同一时间戳>`` 是两段不同的会话，按名字去重会静默丢掉一段。

    ``mode`` 传了就**只留这一种装置的会话**（`None` = 跨装置，工作台用）。
    这里做过滤而不是留给调用方：调用方（会话列表 / 状态卡）各有各的用法，
    过滤规则写两遍必然分叉 —— 2026-11-04 的隔离问题就是这么来的。

    ``mode`` 字段可能是 ``None`` —— 那表示**这份会话没有记录装置模式**。
    老会话都缺这个字段（``record_stereo`` 写入 ``stereo_mode`` 是
    ``a137766`` 才加的），所以**不能默认成 parallel**。

    每行除了原有字段，还带 ``layout``（按目录推断的模式，旧扁平层为 None）
    与 ``legacy``（是否躺在旧扁平层——会话管理器据此提示"可整理到分区"）。
    """
    if base is None:
        dirs = candidate_session_dirs()
    else:
        roots = ((Path(base),) if isinstance(base, (str, Path))
                 else tuple(Path(b) for b in base))
        dirs = candidate_session_dirs(roots)

    rows: list[dict] = []
    # **不按目录名去重**（2026-11-09 删掉原 `seen: set[str]` 那几行）：
    # `candidate_session_dirs` 已经按 resolved 路径跨根去过一次重，这里是冗余；
    # 更关键的是按 `d.name` 去重是**错的** —— 目录名是时间戳，
    # `parallel/20261021-120000` 与 `angled/20261021-120000` 是**两段不同**的
    # 会话，按名字去重会把后一段静默丢掉（审计清单第 4 条）。同一个物理目录
    # 重复出现才该去重，而那件事上游已经做了。
    # 这里**不再自己过滤**"是不是会话"（2026-11-09 删掉原
    # `if not looks_like_session(d): continue`）：`candidate_session_dirs`
    # 现在**保证**返回的都是有 `session.json` 的目录，再写一遍同一个判据
    # 就是 AGENTS.md §3 明令禁止的"调用方各写一遍过滤" —— 那种写法漏一个
    # 就看不见一段会话。要哪档判据就传 `strict=`，判据只在
    # `sphere_tracker/discovery.py` 一处定义。
    for d in dirs:
        row = _row_of(d)
        # 过滤用 `effective_mode`：分区目录里的会话即便没写字段，也知道自己是谁
        # （摆在哪就是哪）。旧扁平层里没字段的会话返回 None，——只在
        # `mode is None`（跨装置）时才留下，这正是"未记录"该有的行为。
        if mode is not None and effective_mode(d) != mode:
            continue
        rows.append(row)
    rows.sort(key=lambda r: r["id"], reverse=True)
    return rows


def _row_of(d: Path) -> dict:
    """一份会话的列表行（解析失败也不抛：坏目录要能被列出来给人处理）。"""
    try:
        raw = json.loads((d / "session.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    left = raw.get("left") if isinstance(raw.get("left"), dict) else {}
    return {
        "path": d,
        "id": d.name,
        "label": str(raw.get("label") or ""),
        "created": str(raw.get("created") or ""),
        "mode": raw.get("stereo_mode"),
        # 会话里帧数叫 frame_count（实测 20261024-173619: 301 帧），不是 frames
        "frames": left.get("frame_count") or "",
        "layout": layout_mode_of(d),
        "legacy": is_legacy_flat(d),
    }


def pick_seeds(parent, session_dir: str | Path, *, ball_kind: str = "") -> Path | None:
    """打开点选球心窗口（复用 ``apps.seed_picker.SeedPicker``），返回保存的路径或 None。

    为什么是进程内而不是 ``tasks.start`` 起子进程：``apps/seed_picker.py`` 只有类、
    **没有 ``__main__``**，`python apps/seed_picker.py <会话>` 会立刻以 0 退出、
    什么窗口都不开 —— 这正是"点了没反应"的形状。经典界面也是
    ``SeedPicker(self, ...) + wait_window``，这里沿用同一做法，不重画第二个点选界面。

    ``ball_kind`` 由调用方的「球种类」给（``kind_key`` 转成 ``metal``/``plastic``），
    写进 seeds.json：跑三维时据此决定用 specular 还是 color，**不再靠"有种子
    就是镜面球"去猜**。

    延迟 import：它依赖 numpy/cv2，而第 ①②③④ 步的其它功能不需要。
    """
    from apps.seed_picker import SeedPicker

    picker = SeedPicker(parent, str(session_dir), ball_kind=ball_kind)
    parent.wait_window(picker)
    return picker.result


class SessionPicker(tk.Toplevel):
    """选一段会话：列出**本装置分区**里的会话，按装置模式**三态**标注。

    为什么要有它：旧界面把"会话目录"做成输入框，用户得自己记住并粘贴路径 ——
    那是"该由系统做的事转嫁给用户"。

    为什么是三态而不是两态：老会话没有 ``stereo_mode`` 字段。两态（匹配/不匹配）
    会把它们当成 parallel，于是 90° 装置录的会话在平行窗里看起来"可用" ——
    **比不过滤更糟**。所以第三态叫"未记录"，如实标注、让人自己确认。

    **未记录默认不显示**（2026-11-04 改）：以前它与"匹配"一样默认列出，于是
    43 段会话里的 29 段未记录会话在两个窗口都能看到 —— 用户原话是
    "平行和大角度的会话没有完全隔离开"。未记录不是"可能匹配"，是"不知道"；
    拿不知道的几何去解算会得到系统性错的深度且不报错，所以默认藏起来。

    ``mode=None`` 表示**跨装置**（会话工作台用）：不按几何过滤、也不标不匹配 ——
    工作台要处理磁盘上任意一段会话，"只看本窗模式"在那里是错的过滤。

    「移到回收站」是**软删除**（复用 ``apps/_cleanup`` 的回收站）；
    「改装置模式…」同时改 ``session.json`` 并把目录搬进对应分区 —— 后者存在的
    理由很实际：磁盘上 29/43 段会话没有模式字段，没有这个动作它们永远归不了位。
    """

    COLS = (("id", "目录", 130), ("label", "标签", 140),
            ("created", "创建时间", 145), ("mode", "模式", 105),
            ("frames", "帧数", 60), ("where", "位置", 85))

    def __init__(self, parent, *, mode: str | None, on_pick,
                 current: str = "", on_changed=None) -> None:
        super().__init__(parent)
        self.mode = mode
        self.on_pick = on_pick
        # 删除/改模式之后父窗口要重判"当前会话还在不在" —— 否则它会继续
        # 拿着一个已经被移进回收站的路径去录制/分析。
        self.on_changed = on_changed
        self.title("选择会话 · " + {
            "parallel": "平行双目", "angled": "大角度双目",
        }.get(str(mode), "任意装置"))
        self.geometry("900x580")
        self.minsize(720, 460)
        self.configure(bg=BG)
        self.surface = Surface(self)
        # 跨装置时"显示全部"没有意义：直接置真，也不放那个复选框
        self.var_all = tk.BooleanVar(value=mode is None)
        self._rows = scan_sessions()

        if mode is None:
            intro = ("这里列出全部双目会话（新布局 data/stereo/<模式>/<时间戳>、"
                     "旧扁平层 data/stereo/<时间戳>、以及 data/sessions 都认）。"
                     "工作台不管你用哪套装置，所以不做过滤。")
        else:
            intro = (f"这里只列「{mode_label(mode)}」的会话：新布局在 "
                     f"data/stereo/{mode}/ 下；旧扁平层的会话按 session.json 的 "
                     "stereo_mode 归位。未记录的默认不显示 —— "
                     "它不是\"可能匹配\"，是\"不知道\"，"
                     "用错几何解算会得到系统性错的深度且不报错。")
        box = self.surface.card(self, "会话", intro)
        self.lbl_note = self.surface.hint(box, "")   # 统计/过滤提示
        self.lbl_sel = self.surface.hint(box, "")    # 选中那一行的说明
        if mode is not None:
            ttk.Checkbutton(
                box, text="显示不属于本窗口模式的会话（标灰；未记录的也在内，慎用）",
                variable=self.var_all,
                command=self._refresh).grid(
                row=self.surface._next_row(box), column=0, columnspan=9,
                sticky="w", pady=(2, 0))

        frame = ttk.Frame(self)
        frame.pack(fill="both", expand=True, padx=16, pady=(8, 0))
        self.tree = ttk.Treeview(frame, columns=[c[0] for c in self.COLS],
                                 show="headings", selectmode="browse")
        for key, title, width in self.COLS:
            self.tree.heading(key, text=title)
            self.tree.column(key, width=width,
                             anchor="w" if key in ("id", "label", "created") else "center")
        bar = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=bar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        for tag, colour in (("ok", INK), ("mismatch", MUTED), ("unknown", WARN)):
            self.tree.tag_configure(tag, foreground=colour)
        self.tree.bind("<Double-1>", lambda _e: self._accept())
        self.tree.bind("<Return>", lambda _e: self._accept())
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self._describe_selected())

        actions = ttk.Frame(self)
        actions.pack(fill="x", padx=16, pady=10)
        self.surface.primary(actions, "用这个会话", self._accept)
        self.surface.secondary(actions, "移到回收站", self._delete)
        self.surface.secondary(actions, "改装置模式…", self._change_mode)
        self.surface.secondary(actions, "整理旧会话…", self._tidy_legacy)
        self.surface.secondary(actions, "刷新", self._rescan)
        self.surface.secondary(actions, "取消", self.destroy)
        notes = ttk.Frame(self)
        notes.pack(fill="x", padx=16, pady=(0, 10))
        self.surface.hint(
            notes,
            "「移到回收站」是软删除：整个会话（含原始视频）进 data/_trash，"
            "之后能在数据管理里恢复。「改装置模式…」会把会话归到对应装置分区"
            "（data/stereo/parallel 或 angled）；「整理旧会话…」一次性把**已记录模式**"
            "的旧会话搬进分区（没记录的那些不会动，需要你逐条判断）。")

        self._refresh()
        if current:
            try:
                self.tree.selection_set(str(Path(current)))
                self.tree.see(str(Path(current)))
            except tk.TclError:
                pass

    # ---------- 三态 ----------
    def _state_of(self, row: dict) -> tuple[str, str]:
        if self.mode is None:
            # 跨装置：不存在"不匹配"这回事，三态退回一态
            return "匹配", ""
        rec = row.get("mode")
        if rec is None or str(rec).strip() == "":
            # 没有字段时看它摆在哪：分区目录里躺着 = 知道是哪套装置录的
            lay = row.get("layout")
            if lay:
                return ("匹配", "") if lay == self.mode else (
                    "不匹配", f"它摆在 {lay}/ 分区里，属于另一种装置")
            return "未记录", "这份会话没记录装置模式，用之前请确认是哪套装置录的"
        if str(rec) == self.mode:
            return "匹配", ""
        return "不匹配", f"这段会话是按 {rec} 几何录的，用本窗解算会得到错的深度"

    def _refresh(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        shown = 0
        hidden = 0
        for row in self._rows:
            state, _why = self._state_of(row)
            # 只有"匹配"默认可见。"不匹配"和"未记录"都要显式勾选才出现 ——
            # 未记录也算"不该默认给你"：不知道几何时用错装置解算不会报错。
            if state != "匹配" and not self.var_all.get():
                hidden += 1
                continue
            tag = {"匹配": "ok", "不匹配": "mismatch", "未记录": "unknown"}[state]
            label = row["label"] or "（无标签）"
            mode_text = mode_label(row["mode"]) if row["mode"] else "未记录"
            where = f"{row['layout']}/" if row.get("layout") else "旧扁平层"
            self.tree.insert("", "end", iid=str(row["path"]),
                             values=(row["id"], label, row["created"],
                                     mode_text, row["frames"], where),
                             tags=(tag,))
            shown += 1
        if shown == 0:
            self.tree.insert("", "end", iid="__none__",
                             values=("（没有会话）", "", "", "", "", ""),
                             tags=("mismatch",))
        if self.mode is not None:
            area = mode_label(self.mode)
            if shown == 0:
                # 空列表最容易被误读成"会话丢了"。说清是**本窗只列这套装置**，
                # 并给出下一步（勾显示全部 → 改装置模式归位）。
                msg = (f"还没有「{area}」的会话。本窗口只列这套装置的会话 —— "
                       "另一种装置录的、或没记录装置模式的会话都不在这里")
                if hidden:
                    msg += (f"（共 {hidden} 段被隐藏：勾上面的复选框就能看到，"
                            "再选中它们点「改装置模式…」归位）")
                self.lbl_note.configure(text=msg, style="Hint.TLabel")
            elif hidden:
                self.lbl_note.configure(
                    text=f"另有 {hidden} 段不是本装置的会话被隐藏"
                         "（勾上面的复选框可以看到；它们要么是另一种装置录的，"
                         "要么没记录装置模式）",
                    style="Hint.TLabel")
            else:
                self.lbl_note.configure(text="", style="Hint.TLabel")
        self._describe_selected()

    def _selected(self) -> dict | None:
        sel = self.tree.selection()
        if not sel or sel[0] == "__none__":
            return None
        return next((r for r in self._rows if str(r["path"]) == sel[0]), None)

    def _describe_selected(self) -> None:
        """选中一行时说清"它摆在哪、记录是什么"（位置与记录不符要明说）。"""
        row = self._selected()
        if row is None or self.mode is None:
            return
        rec = row.get("mode")
        lay = row.get("layout")
        if rec and lay and str(rec) != lay:
            note = (f"记录是 {mode_label(rec)}，却摆在 {lay}/ 分区里 —— "
                    "位置与记录不符，点「改装置模式…」可修正")
        elif not lay:
            note = (f"记录是 {mode_label(rec)}，但还在旧扁平目录"
                    "（data/stereo/ 下没分区）—— 点「改装置模式…」可搬进分区")
        else:
            note = f"装置：{mode_label(rec)}　位置：{lay}/"
        self.lbl_sel.configure(text=note, style="Hint.TLabel")

    def _rescan(self) -> None:
        """重扫磁盘（删完/搬完/外部新增会话后用它，不必关窗再开）。"""
        self._rows = scan_sessions()
        self._refresh()

    def _accept(self) -> None:
        row = self._selected()
        if row is None:
            return
        _state, why = self._state_of(row)
        self.on_pick(str(row["path"]), why)
        self.destroy()

    def _delete(self) -> None:
        """移到回收站（**不是彻底删除**）：录像是不可再生的。"""
        from tkinter import messagebox

        from apps._stereo_sessions import delete_stereo_session

        row = self._selected()
        if row is None:
            return
        ok = messagebox.askyesno(
            "移到回收站",
            f"把整个会话目录 {row['id']} 移到回收站吗？\n\n"
            f"{row['path']}\n\n"
            "视频和派生结果会一起移走，之后可以从数据管理里的回收站恢复。",
            parent=self)
        if not ok:
            return
        try:
            batch = delete_stereo_session(row["path"])
        except (OSError, ValueError) as exc:
            self.lbl_sel.configure(text=f"没移成：{exc}", style="Warn.TLabel")
            return
        self.lbl_sel.configure(
            text=f"已移到回收站：{batch.parent.name}/{batch.name}"
                 "（要恢复请用数据管理里的回收站）", style="Hint.TLabel")
        self._rescan()
        if self.on_changed is not None:
            self.on_changed()

    def _change_mode(self) -> None:
        """改这段会话的装置几何（同时把目录搬进对应分区）。"""
        from apps._stereo_sessions import set_session_mode

        row = self._selected()
        if row is None:
            return
        target = _ask_stereo_mode(self, row)
        if not target:
            return
        try:
            new_path = set_session_mode(row["path"], target)
        except (OSError, ValueError) as exc:
            self.lbl_sel.configure(text=f"没改成：{exc}", style="Warn.TLabel")
            return
        self.lbl_sel.configure(
            text=f"已改为{mode_label(target)}，目录现在在 "
                 f"{new_path.parent.name}/{new_path.name}", style="Hint.TLabel")
        self._rescan()
        if self.on_changed is not None:
            self.on_changed()


    def _tidy_legacy(self) -> None:
        """把**有装置模式的**旧扁平会话搬进对应分区（一次性整理）。

        为什么需要它（2026-11-04）：改动只影响**新录**的会话，而磁盘上已有
        43 段全在旧扁平层 —— 其中 14 段有 ``stereo_mode``、29 段没有。
        没有这个动作，用户升级完看到的仍是"混在一起"，等于什么都没解决。

        **只搬有字段的**：没有 ``stereo_mode`` 的会话，无法从元数据判断是哪套
        装置录的。猜（比如"没 fov 就是 angled"）会猜错 —— 而猜错的后果是用错
        几何解算得到系统性错的深度且不报错。那些留给用户逐条「改装置模式…」。
        """
        from tkinter import messagebox

        from apps._stereo_sessions import set_session_mode

        rows = scan_sessions()
        todo = [r for r in rows if r.get("mode") and r.get("layout") != r["mode"]]
        unknown = [r for r in rows if not r.get("mode") and not r.get("layout")]
        if not todo:
            self.lbl_sel.configure(
                text="没有需要整理的有模式会话"
                     + (f"；另有 {len(unknown)} 段没记录装置模式，需要逐条「改装置模式…」"
                        if unknown else ""),
                style="Hint.TLabel")
            return
        ok = messagebox.askyesno(
            "整理旧会话",
            f"把 {len(todo)} 段有装置模式的会话搬进对应分区吗？\n\n"
            "只搬 session.json 里**已经写了** stereo_mode 的那些；"
            f"另有 {len(unknown)} 段没记录的不会动（它们要你逐条判断）。\n\n"
            "搬动是移动整个目录，会话不会被删除。",
            parent=self)
        if not ok:
            return
        moved, failed = 0, []
        for r in todo:
            try:
                set_session_mode(r["path"], str(r["mode"]))
                moved += 1
            except (OSError, ValueError) as exc:
                failed.append(f"{r['id']}（{exc}）")
        note = f"已整理 {moved} 段到各自分区"
        if unknown:
            note += f"；{len(unknown)} 段没记录模式，仍需逐条「改装置模式…」"
        if failed:
            note += "；失败：" + "；".join(failed[:3])
        self.lbl_sel.configure(
            text=note, style="Hint.TLabel" if not failed else "Warn.TLabel")
        self._rescan()
        if self.on_changed is not None:
            self.on_changed()


def _ask_stereo_mode(parent, row: dict) -> str | None:
    """问「归到哪种装置」的小模态框；返回 ``parallel``/``angled`` 或 None。

    为什么不用 ``tk.Menu`` 弹出菜单：菜单要有事件循环才展开，无头测试里驱动不了，
    而且它无法写清"会把目录搬走"这句 —— 那是这个动作最需要先说清的部分。
    """
    dlg = tk.Toplevel(parent)
    dlg.title("改装置模式")
    dlg.configure(bg=BG)
    # 标准窗口：最小化 / 最大化 / 关闭（AGENTS.md §7.1）。
    # 不用 transient()（Windows 上变工具窗口、只剩 ×）、不锁尺寸、不全局 grab。
    dlg.resizable(True, True)
    body = ttk.Frame(dlg, padding=14)
    body.pack(fill="both", expand=True)

    current = str(row.get("mode") or "").strip() or str(row.get("layout") or "").strip()
    ttk.Label(body, text=f"会话 {row['id']}", font=("", 10, "bold")).pack(anchor="w")
    ttk.Label(body, text=f"当前：{mode_label(current)}",
              foreground=MUTED).pack(anchor="w", pady=(2, 8))
    ttk.Label(body, wraplength=380, foreground=MUTED,
              text="改完会把整个目录搬到 data/stereo/<模式>/ 下 —— "
                   "只改字段不改位置，隔离仍然不成立。"
                   "若目录正被别的程序打开，搬动会失败，但记录已经改好；"
                   "关掉占用后再点一次即可。").pack(anchor="w", pady=(0, 8))

    var = tk.StringVar(value=current if current in ("parallel", "angled") else "parallel")
    for value in ("parallel", "angled"):
        ttk.Radiobutton(body, text=MODE_LABELS[value], value=value,
                        variable=var).pack(anchor="w")

    out: dict[str, str | None] = {"value": None}
    bar = ttk.Frame(body)
    bar.pack(fill="x", pady=(10, 0))

    def ok() -> None:
        out["value"] = var.get()
        dlg.destroy()

    ttk.Button(bar, text="确定", command=ok).pack(side="right")
    ttk.Button(bar, text="取消", command=dlg.destroy).pack(side="right", padx=(0, 6))
    parent.wait_window(dlg)
    return out["value"]


class StereoWindow(tk.Toplevel):
    """双目四步窗。``mode`` ∈ {``parallel``, ``angled``}。"""

    #: 内容区宽度。高度**不写死**：建完之后按内容量（见 ``fit_to_content``）。
    WIDTH = 920
    WIDTH_MIN = 800

    def __init__(self, app, *, mode: str = "parallel") -> None:
        super().__init__(app)
        if mode not in ("parallel", "angled"):
            raise ValueError(f"mode 只能是 parallel/angled，收到 {mode!r}")
        self.app = app
        self.mode = mode
        self.configure(bg=BG)
        # 先给一个"先按内容量的意思"的初始尺寸：这只是建窗时的占位，
        # 真正的高度在 __init__ 末尾由 fit_to_content 按内容算出来。
        # **不要**再写死具体高度：写死的高度只代表"上次放得下"，
        # 内容一长（2026-10-31 实测 881/950）就把第 ④ 步裁在窗口下沿。
        self.geometry(f"{self.WIDTH}x600")
        self.minsize(self.WIDTH_MIN, 560)
        self.surface = Surface(self)

        # 记住上次填的值。**方法不存**：它由窗口模式决定（平行=joint、大角度=双板链式），
        # 存下来会让"同一窗口上次改过的方法"和"窗口代表哪种几何"打架。
        #
        # 大角度窗默认双板（chain_dual）而不是单板（chain）：单板链式**仍然需要
        # 一块共视区**（参考图那处两路都要看到整块板），真 90° 装置上没有这样的
        # 位置——光轴垂直时一块平板从侧向只能看到边缘。零共视的只有双板。
        prefs = app.prefs
        default_method = "joint" if mode == "parallel" else "chain_dual"
        self.var_method = tk.StringVar(value=METHODS[default_method][0])
        self.var_target = tk.StringVar(value=prefs.get_str("stereo.target", "40"))
        self.var_cols = tk.StringVar(value=prefs.get_str("board.cols", "8"))
        self.var_rows = tk.StringVar(value=prefs.get_str("board.rows", "5"))
        self.var_square = tk.StringVar(value=prefs.get_str("board.square_mm", "25"))
        self.var_stationary = tk.StringVar(
            value=prefs.get_str("stereo.stationary_px", "3"))
        self.var_sharpness = tk.StringVar(
            value=prefs.get_str("stereo.min_sharpness", "800"))
        self.var_relaxed = tk.BooleanVar(
            value=prefs.get_bool("stereo.relaxed", False))
        # FOV「应用到当前摄像头」开关（装置级事实，2026-11-15）：
        # 它就是装置档案的 fov_apply，与 FOV 管理器、录制共用同一个开关，
        # 不再读全局 UI 偏好 stereo.use_fov（装置属性不该存在 gui_state 里）。
        # 默认 False：本机四个档案都缺该字段 -> 回落 False -> 标定照旧 --skip-fov、
        # 录制照旧不裁剪，行为与改动前逐字相同。
        # 勾/取消时 _on_toggle_use_fov 立刻写回档案；_refresh_state 会从档案同步。
        self.var_use_fov = tk.BooleanVar(
            value=bool(getattr(self.app.state.profile, "fov_apply", False)))
        self.var_left = tk.StringVar(value="left")
        self.var_right = tk.StringVar(value="right")
        # 卷尺量出来的两镜头中心距（mm）。零共视链路**唯一**独立于图像之外
        # 的物理核对锚：填了后端才会把解出的基线和它比，差 >20% 在报告里明说。
        # 留空 = 不核对（不能替用户猜一个数）。2026-11-09 修 §67 A1 老缺陷。
        self.var_baseline = tk.StringVar(
            value=prefs.get_str("stereo.baseline_mm", ""))
        # 两种新标定方式的输入（轴向大板 / 中轴 90°）。都从 prefs 读，关窗写回：
        # 这些是用户的选择，不是磁盘现状；每次开窗重填 5 个字段没人会接受。
        # 预期基线复用 var_baseline（同一个物理锚：卷尺量的两镜头中心距）。
        self.var_axial_root = tk.StringVar(
            value=prefs.get_str("calib.axial.root", ""))
        self.var_axial_parallel = tk.BooleanVar(
            value=prefs.get_bool("calib.axial.assume_parallel", False))
        self.var_d90_intrinsics = tk.StringVar(
            value=prefs.get_str("calib.device90.intrinsics", ""))
        self.var_d90_axis = tk.StringVar(
            value=prefs.get_str("calib.device90.rot_axis", "y"))
        self.var_d90_angle = tk.StringVar(
            value=prefs.get_str("calib.device90.rot_angle", "90"))
        self.var_d90_za = tk.StringVar(
            value=prefs.get_str("calib.device90.z_a_mm", ""))
        self.var_d90_zb = tk.StringVar(
            value=prefs.get_str("calib.device90.z_b_mm", ""))
        self.var_session = tk.StringVar(value="")
        self._chain_step = 0     # 「采集标定板图」分步流程的游标（双板 4 步 / 单板 3 步）
        #: 关窗旗标。链式采集的 on_done 递归（run_next）会在关窗停任务后仍被
        #: 排到队列里；没有它，用户按 X 之后标定链会自己继续起下一段 ——
        #: 看起来就是"窗口关了又活过来"。
        self._closed = False
        # 可信半径锚点（默认关）：用一次「点选球心」的左右两点定出"两路半径该差多少"，
        # 据此拒掉半径离谱的假配对。开关放在这里而不是"永远开"：它要求会话里
        # **有人点过**球心，没点过就开只会得到一条报错；默认关才符合"点了才有"。
        self.var_anchor = tk.BooleanVar(value=False)
        # 球种类与检测参数（2026-11-03）：控件摆在 ④，但值是**用户的选择**，
        # 所以关窗时和板参数一起存（不存"当前会话"，那是磁盘现状）。
        self.var_preset = tk.StringVar(
            value=prefs.get_str("stereo.preset", PRESET_NONE) or PRESET_NONE)
        self.var_min_radius = tk.StringVar(
            value=prefs.get_str("stereo.min_radius", ""))
        self.var_max_radius = tk.StringVar(
            value=prefs.get_str("stereo.max_radius", ""))
        # 尺寸门自动阶梯（2026-11-08）：默认开；取消勾选会下传 --no-radius-auto。
        # 存 "" / "0" / "1" 而不是直接存 bool，沿用本窗口其它 prefs 的口径。
        self.var_radius_auto = tk.BooleanVar(
            value=prefs.get_str("stereo.radius_auto", "1") != "0")
        # 物理球直径（mm，2026-11-14）：只喂 --anchor-diameter-mm 给半径锚定门
        # 换算视半径，**不控制尺寸门**。空 = 用后端内置默认 21.6 mm。
        self.var_anchor_diameter = tk.StringVar(
            value=prefs.get_str("stereo.anchor_diameter_mm", ""))
        self.var_label = tk.StringVar(value="")
        self.var_seconds = tk.StringVar(value="10")
        #: 录制容器。老界面有 mkv/avi 下拉，新界面从重做起一直恒录 mkv；
        #: 这里补回同一能力。默认 mkv（avi 有 2 GiB 单文件上限，长录不推荐）。
        self.var_container = tk.StringVar(value="mkv")
        # 自由采集档（2026-11-21）：空 = 自动/档案档；填了就以你填的为准。
        # ② 与 ③ **共用同一组 StringVar**，不允许出现两份"当前采集档"。
        self.var_cap_width = tk.StringVar(value=prefs.get_str("stereo.cap_width", ""))
        self.var_cap_height = tk.StringVar(value=prefs.get_str("stereo.cap_height", ""))
        self.var_cap_fps = tk.StringVar(value=prefs.get_str("stereo.cap_fps", ""))

        # 先占底部：pack 从容器边缘往里分配，先 pack 的优先拿到空间。
        # 反过来（先 pack 主内容区、再 pack 状态条）状态条会**盖住内容** ——
        # 2026-10-28 截图里第 4 步卡片就被盖掉了一半。
        self.status = StatusBar(self, on_log=app.open_console,
                                on_stop=self._stop_my_tasks,
                                help_section="calib")
        self.status.pack(fill="x", side="bottom")
        self._page = ScrollPage(self, self.surface)
        self._page.pack(fill="both", expand=True)
        body = self._page.content
        self._cam_text = "相机：读取中…"
        self._build_step1(body)
        self._build_step2(body)
        self._build_step3(body)
        self._build_step4(body)
        # 左/右相机一改，「检查装置」的可用性立刻跟着变：
        # 否则用户清空一边后按钮还是亮的，点下去才被告知"填上再来"。
        for _var in (self.var_left, self.var_right):
            _var.trace_add("write", lambda *_a: self._sync_check_enabled())
        self._refresh_state()
        # 最后一步：按**内容**定高度（上限 = 屏幕可用高度）。必须在
        # `_refresh_state` 之后 —— 那一步会写状态行/原因行，量早了就是量一个
        # 还没写完的界面，算出来的高度会偏小、第 ④ 步照样被裁。
        fit_to_content(self, self._page, self.status, width=self.WIDTH,
                       screen_h=self.app.work_area()[1])

    # ------------------------------------------------------------------ #
    # ① 装置检查
    # ------------------------------------------------------------------ #
    def _build_step1(self, parent) -> None:
        premise = ("平行双目的前提：两路光轴**近平行**（夹角很小），"
                   "且基线远小于工作距离。校正后左右图的行是对齐的。"
                   if self.mode == "parallel" else
                   "大角度双目的前提：两路夹角大（接近直角），不必平行、"
                   "也不必在每个姿态都看到同一块板。深度靠原始像素三角测量。")
        box = self.surface.card(parent, "① 装置检查", premise.replace("**", ""))
        self.lbl_rig = self.surface.hint(box, "检查中…")
        # 相机身份属于 L1，**不许进折叠区**：它是装置的物理身份，不是高级参数。
        # 旧界面让人手填 left/right，填错就静默用错相机 —— 这里只显示、
        # 只能通过「换相机…」改（读 config/cameras.json）。
        self.lbl_cams = self.surface.hint(box, "读取相机…")
        actions = self.surface.action_row(box)
        self.btn_check = self.surface.primary(actions, "检查装置", self._check_setup)
        self.surface.secondary(actions, "换相机…", self._open_cameras)
        self.surface.secondary(actions, "重新读取标定", self._reload_state)

    # ------------------------------------------------------------------ #
    # ② 标定
    # ------------------------------------------------------------------ #
    def _build_step2(self, parent) -> None:
        hint = ("这一步决定第 4 步能不能用。平行装置只需要 joint；"
                "如果两路看不到同一区域，就该去「大角度双目」窗口。"
                if self.mode == "parallel" else
                "按「两路还能看到多少共同画面」选方法：还能充分共视 → 两路共视；"
                "只剩一处能同时看到整块板 → 单板链式；完全看不到同一区域 → 双板链式"
                "（要两块刚性固定的板）。链式产物都直接能在第 4 步用。")
        box = self.surface.card(parent, "② 标定", hint)
        self.lbl_calib = self.surface.hint(box, "检查中…")

        # L1：目标对数 / 分化线 + 板参数（一行）
        #
        # **两个窗口这一格不是同一个东西**（2026-11-07 用户指出"平行窗疑似和
        # 大角度用了一套代码"）：
        #   · 平行（joint）：`calibrate_stereo.py --target` 只是"收够多少对就停"。
        #     2026-11-07 用户第二次纠正：**不许再写"五个角都要拍到 / 5/5 全绿"** ——
        #     既然收够对数就停，再挂一个覆盖约束自相矛盾。后端本来就只有
        #     "这一帧够不够好"的挑帧门限（停稳 / 清晰 / 面积 / 位姿不重复），
        #     没有"五个角集齐才准收工"这一条；界面不该凭空加一个限制。
        #   · 大角度（chain）：`capture_board.py --target` 是**分化线**，拍够后
        #     覆盖标准由 5/5 放宽到 4/5，因为零共视时凑齐五个角确实很难。
        # 2026-11-05 为链式改了标签后把这个词带到了平行窗，于是平行窗写了一句
        # 自己后端根本不做的事。现在按窗口分开：平行恢复原设计，大角度原样保留。
        line = ttk.Frame(box)
        line.grid(row=self.surface._next_row(box), column=0, columnspan=9,
                  sticky="w", pady=(4, 0))
        if self.mode == "parallel":
            ttk.Label(line, text="目标对数 张").pack(side="left")
            ttk.Entry(line, textvariable=self.var_target, width=5).pack(
                side="left", padx=(8, 4))
            ttk.Label(line, text="（收够就停；板摆稳、清晰就自动收）",
                      style="Hint.TLabel").pack(side="left", padx=(0, 14))
        else:
            ttk.Label(line, text="分化线 张").pack(side="left")
            ttk.Entry(line, textvariable=self.var_target, width=5).pack(
                side="left", padx=(8, 4))
            ttk.Label(line, text="（拍够后 4/5 即可；之前要求 5/5 全绿）",
                      style="Hint.TLabel").pack(side="left", padx=(0, 14))
        ttk.Label(line, text="棋盘内角点").pack(side="left")
        ttk.Entry(line, textvariable=self.var_cols, width=4).pack(
            side="left", padx=(8, 3))
        ttk.Label(line, text="×").pack(side="left")
        ttk.Entry(line, textvariable=self.var_rows, width=4).pack(
            side="left", padx=(3, 12))
        ttk.Label(line, text="方格 mm").pack(side="left")
        ttk.Entry(line, textvariable=self.var_square, width=5).pack(
            side="left", padx=(8, 0))

        # 方法下拉：两个窗口都给，但候选按模式过滤（轴向大板只进平行窗、
        # 中轴 90° 只进大角度窗）。以前平行窗写死 joint 是因为只有一种方法，
        # 现在多了一个轴向标定，不给选就等于功能不可达。
        line2 = ttk.Frame(box)
        line2.grid(row=self.surface._next_row(box), column=0, columnspan=9,
                   sticky="w", pady=(6, 0))
        ttk.Label(line2, text="标定方法").pack(side="left")
        self._picker_keys = picker_methods_for(self.mode)
        cb = ttk.Combobox(line2, textvariable=self.var_method, width=34,
                          state="readonly",
                          values=[METHODS[k][0] for k in self._picker_keys])
        cb.pack(side="left", padx=(8, 0))
        cb.bind("<<ComboboxSelected>>", lambda _e: self._on_method_changed())
        # 必须走 surface.hint：手工 ttk.Label 既没有 wraplength、也不在
        # 换行登记表里，长说明会被 grid 单元格直接裁掉（2026-10-28 截图里
        # 「…本窗口要的 angled」末尾的 d 就是这么丢的）。
        self.lbl_method = self.surface.hint(box, "")

        # L2：高级（默认收起）
        adv = CollapsibleSection(box, "高级：采集门限与相机", expanded=False,
                                 summary="停稳 / 清晰度 / 放宽 / 相机")
        adv.grid(row=self.surface._next_row(box), column=0, columnspan=9,
                 sticky="we", pady=(8, 0))
        inner = adv.content
        # 留给方法专属控件懒建用（axial / device90）。默认方法下不创建任何
        # 东西，budget 测到的控件树与改动前一致。
        self._adv_content = inner
        self._method_rows = {}
        line3 = ttk.Frame(inner)
        line3.pack(anchor="w", pady=(4, 0))
        ttk.Label(line3, text="停稳上限 px").pack(side="left")
        ttk.Entry(line3, textvariable=self.var_stationary, width=5).pack(
            side="left", padx=(6, 12))
        ttk.Label(line3, text="清晰度下限").pack(side="left")
        ttk.Entry(line3, textvariable=self.var_sharpness, width=6).pack(
            side="left", padx=(6, 0))
        if self.mode == "angled":
            # 接在 line3 同一行：不新增行，折叠区展开高度不变（window_mode
            # 判据对展开增量有断言）。只有大角度窗需要这个锚 —— 平行双目
            # 两路共视，本身就有跨路一致性证据。
            ttk.Label(line3, text="卷尺量两镜头中心距 mm（可留空）").pack(
                side="left", padx=(12, 0))
            ttk.Entry(line3, textvariable=self.var_baseline, width=7).pack(
                side="left", padx=(6, 0))
        ttk.Checkbutton(inner, text="放宽模式（手持拍摄：只放松面积/边距/亮度四项）",
                        variable=self.var_relaxed).pack(anchor="w", pady=(6, 0))
        # FOV 装置级「应用」开关：与 FOV 管理器同一个开关（都读写档案 fov_apply）。
        # 不新增控件（只是给既有勾选框接了 command），L1 / 展开高度预算不变。
        ttk.Checkbutton(
            inner,
            text="把 FOV 应用到当前装置的左右相机（与 FOV 管理器同一个开关）",
            variable=self.var_use_fov,
            command=self._on_toggle_use_fov).pack(anchor="w", pady=(6, 0))
        self.lbl_fov = ttk.Label(inner, text="", style="Hint.TLabel")
        self.lbl_fov.pack(anchor="w")
        self._sync_fov_hint()
        ttk.Label(inner,
                  text="停稳与清晰度填了就以你填的为准：放宽模式不会覆盖这两个值。",
                  style="Hint.TLabel").pack(anchor="w")
        line4 = ttk.Frame(inner)
        line4.pack(anchor="w", pady=(6, 4))
        ttk.Label(line4, text="左相机").pack(side="left")
        ttk.Entry(line4, textvariable=self.var_left, width=10).pack(
            side="left", padx=(6, 12))
        ttk.Label(line4, text="右相机").pack(side="left")
        ttk.Entry(line4, textvariable=self.var_right, width=10).pack(
            side="left", padx=(6, 0))
        # 采集档字段与录制共用，统一放在 ③；这里只放一句不占可点预算的提示。
        ttk.Label(inner,
                  text="采集宽/高/fps 与录制共用：在 ③「采集」卡设置（空 = 自动/标定档）。",
                  style="Hint.TLabel").pack(anchor="w", pady=(4, 0))
        # 强制通过（仅链路测试，2026-11-14）：单独一个 secondary，绝不做成 primary
        # （每一步「主操作 ≤1」由 selftest_gui_budget.py 钉住）。放进「高级」折叠区
        # 而不是 action_row：既不占首屏，也不动 L1 按钮行；默认仍是「开始标定」。
        force_line = ttk.Frame(inner)
        force_line.pack(anchor="w", pady=(6, 0))
        self.surface.secondary(
            force_line, "强制通过标定（仅链路测试）…", self._force_calibrate)
        ttk.Label(force_line,
                  text="默认质量差只旁路保存、绝不覆盖正式标定路径；"
                       "此按钮需二次确认，产物标为不可用，不得用于数据结论。",
                  style="Hint.TLabel").pack(side="left", padx=(8, 0))

        actions = self.surface.action_row(box)
        self.surface.primary(actions, "开始标定", self._start_calibrate)
        if self.mode == "angled":
            # 链式标定（chain）的前置步骤：90° 装置缺的就是这两个入口 ——
            # 以前选 chain 只有「开始标定」，但后端要 `outputs/chain_frames/`
            # 下的四段图，GUI 没地方采，点了标定只能看"缺图"报错。
            self.btn_capture_chain = self.surface.secondary(
                actions, "采集标定板图…", self._capture_chain_frames)
            self.btn_reshoot_chain = self.surface.secondary(
                actions, "重拍某一步…", self._reshoot_chain_stage)
            self.btn_check_chain = self.surface.secondary(
                actions, "检查链式输入", self._check_chain_inputs)

    def _sync_fov_hint(self) -> None:
        """把档案里 FOV 的状态写清：勾了会发生什么、不勾会发生什么。

        这不是装饰。FOV 一关，两路像素尺度不一致时视差/深度会有系统偏差，
        用户必须知道代价；反过来档案里根本没量过 FOV 时，勾了也只是现场估计，
        不能让人以为"勾了就复用档案里那份"。
        """
        lbl = getattr(self, "lbl_fov", None)
        if lbl is None:
            return
        fov = getattr(self.app.state.profile, "fov", None)
        if fov:
            side = str(fov.get("crop_side") or "?")
            size = fov.get("target_size") or []
            size_txt = f"→{size[0]}×{size[1]}" if len(size) == 2 else ""
            lbl.configure(text=f"档案已记录 FOV（裁 {side}{size_txt}）；"
                               "不勾 = 按原始画面标定")
        else:
            lbl.configure(text="档案里还没量过 FOV；勾选 = 本次从启动帧现场估计"
                               "（估不出来会降级为不裁剪，并写 fov=null）")

    def _on_toggle_use_fov(self) -> None:
        """双目窗的勾选框 = 装置档案里那个装置级「是否应用」的开关。

        两者是**同一个事实**（不是两个流程开关）：写回档案后标定（第②步）
        与录制（record_stereo 自己读档案）都服从它。写失败只在状态栏说清、
        把勾选框回滚，**不抛**。
        """
        prof = self.app.state.profile
        if prof is None or not getattr(prof, "name", ""):
            self.var_use_fov.set(False)
            self.status.set_status("先选一个装置档案，再改「应用到当前摄像头」", "warn")
            return
        want = bool(self.var_use_fov.get())
        try:
            import dataclasses
            from stereo import profiles
            profiles.upsert(dataclasses.replace(prof, fov_apply=want),
                            replace_existing=True)
        except Exception as exc:                    # noqa: BLE001
            self.var_use_fov.set(bool(getattr(prof, "fov_apply", False)))
            self.status.set_status(f"写装置档案失败：{exc}", "warn")
            return
        # 必须 reload：否则 _refresh_state 会从旧 profile 把刚勾的值覆盖回去。
        self.app.state.reload()
        self._sync_fov_hint()

    # ------------------------------------------------------------------ #
    # ③ 采集
    # ------------------------------------------------------------------ #
    def _build_step3(self, parent) -> None:
        box = self.surface.card(
            parent, "③ 采集（录制一段，供第 4 步分析）",
            "录的是两路同步视频。球进入画面那一刻开始，别中途改帧率。"
            "如果实验需要同步信号，用闪光：倒计时里不要闪，听到开始再闪。")
        line = ttk.Frame(box)
        line.grid(row=self.surface._next_row(box), column=0, columnspan=9,
                  sticky="w", pady=(4, 0))
        ttk.Label(line, text="时长 秒").pack(side="left")
        ttk.Entry(line, textvariable=self.var_seconds, width=6).pack(
            side="left", padx=(8, 14))
        ttk.Label(line, text="标签").pack(side="left")
        ttk.Entry(line, textvariable=self.var_label, width=16).pack(
            side="left", padx=(8, 0))
        # 录制格式（2026-11-20）：老界面有 mkv/avi 下拉，新界面此前恒录 mkv。
        # 只给两档，**没有 mp4** —— 无损编码进 mp4 会被 FFmpeg 静默换成
        # 有损 h264（record_stereo.CODECS 的注释里有实测理由）。
        ttk.Label(line, text="格式").pack(side="left", padx=(14, 0))
        ttk.Combobox(line, textvariable=self.var_container,
                     values=("mkv", "avi"), state="readonly", width=5).pack(
            side="left", padx=(8, 0))
        # 自由采集档：空 = 录制程序按标定 capture_size 决定。填了会下传
        # --width/--height/--fps，并仍由 validate_capture_size 校验。
        # 接在**既有**那一行，不新增行（大角度 ④ 的高度预算只剩 0 px）。
        ttk.Label(line, text="宽").pack(side="left", padx=(14, 0))
        ttk.Entry(line, textvariable=self.var_cap_width, width=5).pack(
            side="left", padx=(6, 0))
        ttk.Label(line, text="高").pack(side="left", padx=(8, 0))
        ttk.Entry(line, textvariable=self.var_cap_height, width=5).pack(
            side="left", padx=(6, 0))
        ttk.Label(line, text="fps").pack(side="left", padx=(8, 0))
        ttk.Entry(line, textvariable=self.var_cap_fps, width=5).pack(
            side="left", padx=(6, 0))
        actions = self.surface.action_row(box)
        self.btn_record = self.surface.primary(actions, "开始录制", self._start_record)
        # 「实时 3D」放在这里而不是门厅（2026-11-16 用户裁决）：实时与离线是
        # **同一套装置**的两种处理方式 —— 录一段去第④步离线跑，或者直接开相机
        # 出三维。它属于装置，不属于"跨模式工具"。平行窗与大角度窗各有一份，
        # 和单目窗第③步那个「识别来源：实时（相机）/ 录像」是同一个思路。
        self.btn_realtime = self.surface.secondary(
            actions, "实时 3D（不落盘）", self._start_realtime_3d)
        # 「使用相机实际模式」放在动作行：它真开相机，属于动作而不是输入；
        # 放在 ③ 的输入行会把那一行撑高 7 px（window_mode 实测）。
        self.btn_actual_mode_record = self.surface.secondary(
            actions, "使用相机实际模式…", self._use_actual_mode)

    # ------------------------------------------------------------------ #
    # ④ 分析
    # ------------------------------------------------------------------ #
    def _build_step4(self, parent) -> None:
        box = self.surface.card(
            parent, "④ 分析（对一段已录好的会话跑三维）",
            "离线重建：读两路视频按标定解三维，强制按 "
            + ("平行" if self.mode == "parallel" else "大角度")
            + " 解算；3D 查看器同屏加载 derived/annotated.mp4，"
            + "也能用系统播放器单独打开。")
        # 球种类与检测参数（2026-11-03）：以前这一段只有一个会话输入框，
        # 检测参数全靠 `run_stereo_video` 自己的默认值 —— 换了一颗纯色球的人
        # 在界面上**没有任何地方**能说"这是红球"，只能去命令行加 `--preset`。
        # 摆在第④步而不是第②③步：它只影响"解算"，不影响标定与录制。
        self.ball = BallSettings(
            self.surface, box,
            methods=detector_methods(), presets=preset_choices(),
            kind=(BALL_METAL
                  if _actions.has_seeds(self.var_session.get().strip())
                  else BALL_PLASTIC),
            preset_var=self.var_preset,
            radius_vars=(self.var_min_radius, self.var_max_radius),
            radius_auto_var=self.var_radius_auto,
            anchor_diameter_var=self.var_anchor_diameter,
            anchor_var=self.var_anchor,
            on_change=self._refresh_ball_note)
        line = ttk.Frame(box)
        line.grid(row=self.surface._next_row(box), column=0, columnspan=9,
                  sticky="w", pady=(4, 0))
        ttk.Label(line, text="会话目录").pack(side="left")
        # 输入框宽度 46 → 38：给同一行的锚点勾选框腾地方（不新增行，见下）。
        ttk.Entry(line, textvariable=self.var_session, width=38).pack(
            side="left", padx=(8, 8))
        ttk.Button(line, text="选择…", command=self._pick_session).pack(
            side="left")
        # 半径锚点开关（默认关）：勾了才传 --anchor-gate on。
        #
        # 为什么跟会话输入框**同一行**：本窗口高度按"一屏看全"定，单起一行实测会把
        # 「大角度」顶出 5 px（内容 874 > 窗口 869）；再加一行说明则是 53 px。
        # 宁可把输入框收窄，也不裁按钮、不动控件预算。
        ttk.Checkbutton(
            line, text="半径锚点门（勾了才生效）",
            variable=self.var_anchor).pack(side="left", padx=(10, 0))
        actions = self.surface.action_row(box)
        self.surface.primary(actions, "跑离线三维", self._start_offline)
        self.surface.secondary(actions, "点选球心", self._start_seed)
        self.surface.secondary(actions, "打开 3D 查看器", self._open_viewer)
        # 「打开标注视频」**不在这里**：第④步已有 4 个动作按钮 + 「选择…」，
        # 再加就超过 L1 按钮预算（≤5）。直达入口放在 **3D 查看器工具栏**
        # （`apps/view_3d.py` 的「用系统播放器打开视频」，不在步骤卡内计数），
        # 会话工作台第④步也保留一个。模式窗点「打开 3D 查看器」即可同屏验收。
        # 第 ④ 步只管"当前会话"；要换一段历史会话从中间接着做，去工作台
        self.surface.secondary(actions, "历史会话…", self._open_bench)
        # 四个按钮**同一行**（2026-10-31，缺陷 #9/#10）：
        # 原先分成两行，多出一整行按钮 + 行距。实测四步内容为 846/915 px
        # （平行/大角度），比当时的窗口 800 高 —— 第 ④ 步的按钮就被裁在
        # 窗口下沿；合并成一行后是 799/868 px。
        # 为什么不是"把次要按钮收进折叠区"：折叠区的标题本身要 28 px，
        # 只省回一小半，却把「历史会话…」藏起来（那是用户从中间接着做的
        # 唯一入口）。合并没有藏任何东西，省得也更多。
        # 一行 4 个动作按钮 + 「选择…」仍在预算内（L1 按钮 ≤5、主操作恰好 1）。

    # ------------------------------------------------------------------ #
    # 状态
    # ------------------------------------------------------------------ #
    def _refresh_state(self) -> None:
        st = self.app.state
        # 装置级 FOV 应用开关是单一事实：FOV 管理器里改了、已经开着的双目窗
        # 也要跟着变（broadcast_state_refresh 会遍历调用本方法）。
        self.var_use_fov.set(
            bool(getattr(getattr(st, "profile", None), "fov_apply", False)))
        self._sync_fov_hint()
        prof = getattr(st.profile, "name", None)
        mode = st.stereo_mode() or "未记录"
        base = st.baseline_mm()
        angle = st.angle_deg()
        bits = [f"装置档案：{prof or '未设置'}",
                f"标定模式：{mode}"]
        if base is not None:
            bits.append(f"基线 {base:.0f} mm")
        if angle is not None:
            bits.append(f"夹角 {angle:.1f}°")
        self.lbl_rig.configure(text="｜".join(bits))

        # 第②步是**标定**（产出动作），不是消费现有标定。三种方法在本窗口都合法
        # （joint/joint_skip_fov/chain 都会产出本窗口要的 stereo_mode），所以这里
        # **不**再用 parallel_status/angled_status 的"现有标定不匹配"来拒绝 ——
        # 那套判定是给第③步录制/第④步分析用的（它们消费现有标定）。以前把
        # angled_status 用在这里，导致大角度窗在"现有标定还是上次的 parallel"时，
        # 状态栏写"请用平行窗口"，用户想选 joint 重新标定却被误判成"不匹配"。
        existing = "尚无标定" if st.rig is None else f"当前已有标定（{mode}）"
        self.lbl_calib.configure(
            text=f"标定：{existing}。选好方法点「开始标定」即可产出本窗口需要的标定",
            style="Ok.TLabel")

        # 相机身份：只读 config/cameras.json，改它只能通过「换相机…」
        try:
            # 这一行用**短标签**（位置优先）：完整身份 / PnP 位次 / 采集索引
            # 在「换相机…」对话框里。把完整状态塞进这一行会超过换行边界，
            # 大角度窗内容会高出 16 px（2026-11-20 实测，selftest_window_mode）。
            self._cam_text = ("相机：左=" + st.camera_role_short("left")
                              + "｜右=" + st.camera_role_short("right"))
        except Exception:                                   # noqa: BLE001
            self._cam_text = "相机：读取失败（config/cameras.json）"
        self._sync_check_enabled()
        self._sync_record_enabled()
        if self.lbl_method is not None:
            self._refresh_method_hint()
        # 第②步本身不再有"就绪"门槛（标定随时可做）；就绪属于第③/④步。
        # 但底部状态条是用户判断"现在能不能录"的唯一依据，不能无条件写"就绪"：
        # 大角度窗配着平行标定时，③「开始录制」点下去必然被拒（"请用平行窗口"），
        # 而底部却写着"就绪"—— 走查里用户原话是"它说就绪，点了又不让录"。
        ready_rec, why_rec = self._record_gate()
        if ready_rec:
            self.status.set_status("就绪", "ok")
        else:
            self.status.set_status(
                f"第③步录制会用不了：{why_rec.replace('**', '')}", "warn")

    def _sync_check_enabled(self) -> None:
        """「检查装置」能不能点：只看左右相机有没有填。

        拿不到左右编号就让按钮**灰着并写出原因** —— 会让用户"点了等一个永远
        不来的结果"的按钮，比灰着的按钮更伤人（2026-10-31 缺陷 #4）。
        单独成一个方法是因为它还要跟着输入框变（见 ``__init__`` 里的 trace）。
        """
        has_roles = bool(self.var_left.get().strip()) and bool(self.var_right.get().strip())
        self.btn_check.configure(state="normal" if has_roles else "disabled")
        text = self._cam_text
        if not has_roles:
            text += "｜左/右有一边没填，「检查装置」不可用：先在「高级」里填上"
        else:
            # B-149：与单目窗复用同一份 role_online 判定（延迟 import 避免顶层
            # 模块循环）。缺相机时给可执行下一步，而不是只写「不在线」。
            from apps._gui_mono import role_online
            offline = [r for r in (self.var_left.get().strip(),
                                   self.var_right.get().strip())
                       if r and not r.startswith("#") and role_online(r) is False]
            if offline:
                text += ("｜" + "、".join(offline) + " 没在 config/cameras.json "
                         "绑定：点「换相机…」跑 identify 固定；要用编号请填**实测的"
                         "采集索引**（纯数字），PnP 位次不能直接开相机")
        self.lbl_cams.configure(text=text)

    def _record_gate(self) -> tuple[bool, str]:
        """第③步"能不能录"的唯一判据。

        `_start_record`、底部状态条预警、③ 录制按钮的灰置三处共用这一份，
        不各自再写一份 parallel_status/angled_status 分支 —— 否则「重新读取
        标定」之后按钮和状态条会给出互相矛盾的原因。
        """
        st = self.app.state
        # 自由采集档的本地前置检查：填了就必须与标定同档。后端还会再验一次
        # （`validate_capture_size`），这里只是让用户别等子进程起来才被拒。
        _w, _h, _fps, _cap_err = self._cap_mode_fields()
        if _cap_err:
            return False, _cap_err
        if _w is not None and st.rig is not None:
            _size = tuple(getattr(st.rig, "image_size", ()) or ())
            if len(_size) == 2 and (_w, _h) != (_size[0], _size[1]):
                return False, (f"采集档 {_w}x{_h} 与标定 {_size[0]}x{_size[1]} "
                               "不一致；录制会被 validate_capture_size 拒，"
                               "请重标或把采集档改成标定值")
        ready, why = (st.parallel_status() if self.mode == "parallel"
                      else st.angled_status())
        # 相机角色必须实测过采集索引；否则按角色名启动会被相机层拒绝
        # （PnP 位次不能开相机）。在**进入录制/实时**之前就说清楚，别等
        # 子进程起来才报错。
        try:
            from sphere_tracker import camera_registry as cr
            reg = cr.load_registry()
            missing = []
            # 看**本窗实际会传给后端的值**，不是死看 left/right：
            # 用户显式填纯数字（或 port:/id:/name:）时，那是他自己的选择，
            # 不该被角色 capture_index 的门禁误拦。
            for role, value in (("left", self.var_left.get()),
                                ("right", self.var_right.get())):
                v = (value or "").strip()
                if not v or v.lower() in ("auto", "自动"):
                    continue
                if v.isdigit() or v.lower().startswith(("port:", "id:", "name:")):
                    continue
                spec = reg.roles.get(v)
                if spec is None:
                    missing.append(f"{v} 未绑定")
                elif spec.capture_index is None:
                    label = spec.name or v
                    missing.append(f"{v}（{label}）采集索引未实测")
            if missing:
                cam_why = ("相机：" + "；".join(missing)
                           + " —— 先跑 python apps/tools/cameras.py identify")
                return False, (why + "；" if why else "") + cam_why
        except Exception:                                    # noqa: BLE001
            pass
        return ready, why

    def _start_realtime_3d(self) -> None:
        """③「实时 3D（不落盘）」：两路相机直接出三维，不写会话。

        准入判据**复用现成的那一份**，不另写：
        - 文件在不在 → `current_calibration(need_existing=True)`。⚠️ 不许自己
          `Path(base).exists()`：标定基名是**无扩展名**的，产物是 `<base>.json` /
          `<base>.npz`，只判 `exists()` 恒为假 —— 2026-11-16 我就这么错过一次，
          用户点按钮看到「还没有可用的双目标定（实验性质平行装置1 不存在）」。
        - 这套标定配不配这个窗口 → `_record_gate()`（`parallel_status` /
          `angled_status` 的唯一产地，与「开始录制」同源）。
        """
        from apps._common import current_calibration

        _base, why = current_calibration(need_existing=True)
        if why:
            self.status.set_status(
                "还不能实时三维：" + " ".join(why.split()).replace("**", ""), "warn")
            return
        ready, why_gate = self._record_gate()
        if not ready:
            self.status.set_status(
                "还不能实时三维：" + why_gate.replace("**", ""), "warn")
            return
        # 检测方法与颜色预设**跟着第①步的球种类走**（2026-11-17 用户实测：
        # 塑料球和金属球都检不出来）。原因很具体：按钮原来只传 `--method color`
        # 且不传 `--preset`，于是永远吃 `ColorConfig()` 的默认阈值 —— 那是
        # **白/浅色、低饱和、高亮**（white_ball：S<80, V>175），彩色球必然 0 检出。
        # 单目窗一直是把 `self.ball.method()/preset()` 传下去的，这里对齐。
        method = self.ball.method() or METHOD_PLASTIC
        preset = self.ball.preset()
        if method == METHOD_METAL:
            # 镜面金属球要 `specular`，而 specular 靠 `seeds.json` 里"点选球心"
            # 的种子（只在高光附近找，不会自己初始化）—— 实时没有种子，必然 0 检出。
            # 与其让人对着黑屏猜，不如说清要走哪条路（离线第④步那条，见本文件
            # `_start_offline` 的同一道护栏）。
            self.status.set_status(
                "镜面金属球不能实时：" + "specular 靠『点选球心』的种子找高光，"
                "实时没有种子会 0 检出。请先录一段，点「点选球心」，再用第④步离线跑",
                "warn")
            return
        # 分辨率按**这份标定量的那一档**给：内参是像素量纲的，喂别的档后端会
        # exit 2 并让你加 `--width/--height`（大角度那套 2560x1440，平行 1920x1080）。
        # 与其让人去翻标定文件，不如这里直接给对。拿不到就留空（= 后端自适应）。
        _rig = getattr(self.app.state, "rig", None)
        _size = tuple(getattr(_rig, "image_size", ()) or ())
        _w, _h = ((str(_size[0]), str(_size[1])) if len(_size) == 2 else ("", ""))
        # 界面手填优先；没填才用标定 image_size。填了但不等于标定档时，
        # `_record_gate()` 已经拦住（实时也会按错内参，不能放行）。
        _cw, _ch, _cfps, _cap_err = self._cap_mode_fields()
        if _cap_err:
            self.status.set_status(
                f"还不能实时三维：{_cap_err}", "warn")
            return
        if _cw is not None:
            _w, _h = str(_cw), str(_ch)
        self.app.tasks.start(
            # 相机角色跟录制用同一份输入（右侧默认 left/right，查 config/cameras.json）。
            # 不交给后端的 `--auto-cameras` 兜底：自动挑会按"外接/避开内置"去猜，
            # 而这里已经**确定**是这套装置绑定的两台 —— 送到手上的按钮不该再猜一次。
            _actions.stereo_realtime_args(
                left=self.var_left.get(), right=self.var_right.get(),
                width=_w, height=_h,
                fps=(f"{_cfps:g}" if _cfps is not None else ""),
                method=method, preset=preset),
            "双目实时三维检测", owner=self.task_owner)
        # 底部状态条只放**一句话**（2026-11-17 用户：长句会被截断，宽度不够）。
        # 按键说明常驻在实时窗口的顶栏第二行（`REALTIME_HINT`），详细解释在「操作说明」。
        self.status.set_status(
            "已启动双目实时三维检测（OpenCV 窗口）；按键与指标说明在画面顶栏和「操作说明」",
            "info")

    def _sync_record_enabled(self) -> None:
        """③「开始录制」跟着标定状态走（由 `_refresh_state` 调用）。

        不匹配时按钮灰置并在文字里带原因；具体 why 见底部状态条（同源判据）。
        只改这一个按钮的 state/text，不新增任何控件行、不增高窗口。
        """
        btn = getattr(self, "btn_record", None)
        if btn is None:
            return
        ready, why = self._record_gate()
        # 按钮文字要指向**真正的原因**：新契约下"采集索引未实测"也会禁用录制，
        # 一律写"先解决标定"会把用户引到错误的下一步。
        if "采集索引" in why:
            short = "先确认相机"
        elif "标定" in why or "不匹配" in why:
            short = "先解决标定"
        else:
            short = "先看状态条"
        if ready:
            btn.configure(state="normal", text="开始录制")
        else:
            btn.configure(state="disabled", text=f"开始录制（{short}）")
        # 「实时 3D」用同一条判据灰置：它也要吃这套标定，没有理由一开一关。
        rt = getattr(self, "btn_realtime", None)
        if rt is not None:
            if ready:
                rt.configure(state="normal", text="实时 3D（不落盘）")
            else:
                rt.configure(state="disabled", text=f"实时 3D（{short}）")

    def _cap_mode_fields(self) -> tuple[int | None, int | None, float | None, str]:
        """解析界面上的自由采集档；返回 ``(w,h,fps,错误)``。

        空 = 不指定（自动/标定档）；只填一半 = 明确报错，**不**替用户补另一半。
        """
        w_text = self.var_cap_width.get().strip()
        h_text = self.var_cap_height.get().strip()
        f_text = self.var_cap_fps.get().strip()
        if not w_text and not h_text:
            w = h = None
        elif not w_text or not h_text:
            return None, None, None, "采集宽和高要一起填（或都留空 = 自动/标定档）"
        else:
            try:
                w, h = int(w_text), int(h_text)
            except ValueError:
                return None, None, None, "采集宽/高必须是整数"
            if w <= 0 or h <= 0:
                return None, None, None, "采集宽/高必须 > 0"
        fps = None
        if f_text:
            try:
                fps = float(f_text)
            except ValueError:
                return None, None, None, "采集 fps 必须是数字"
            if fps <= 0:
                return None, None, None, "采集 fps 必须 > 0"
        return w, h, fps, ""

    def _use_actual_mode(self) -> None:
        """「使用相机实际模式」：真开两路、逐档量实际宽高/帧率，再填回输入框。

        不自动挑一个"看起来支持"的档：两台不一致、有一台打不开、fps 没测到，
        都**不填**并在状态条说清。静默按某一台填，等于下一轮录制必被拒。
        """
        if self.app.tasks.running():
            self.status.set_status(
                "有任务正在跑（标定/录制/实时/探测）：先等它结束，"
                "避免两路同时抢相机", "warn")
            return
        left = self.var_left.get().strip()
        right = self.var_right.get().strip()
        if not left or not right:
            self.status.set_status("先把左右相机填上（角色名或实测采集编号）", "warn")
            return
        if left.lower() in ("auto", "自动") or right.lower() in ("auto", "自动"):
            self.status.set_status(
                "左右是 auto：先固定成角色名或实测采集编号，"
                "「实际模式」不能替 auto 猜是哪台", "warn")
            return
        # 只认「角色名」或「纯数字采集编号」：port:/id:/name: 需要先 pin 成角色，
        # 否则 `cameras.py modes` 会把它当角色名查不到，报错离用户的操作很远。
        try:
            from sphere_tracker import camera_registry as _cr
            _reg = _cr.load_registry()
            _roles = set(_reg.roles)
        except Exception:                                    # noqa: BLE001
            _roles = set()
        for _value in (left, right):
            if _value.isdigit() or _value in _roles:
                continue
            self.status.set_status(
                f"「{_value}」不是角色名也不是采集编号：port:/id:/name: 先用"
                "「换相机…」pin 成 left/right；实际模式只认角色或实测编号", "warn")
            return
        indices = ""
        if left.isdigit() or right.isdigit():
            if not (left.isdigit() and right.isdigit()):
                self.status.set_status(
                    "左/右一个是编号、一个是角色名：先统一（都用角色名或都用实测编号）",
                    "warn")
                return
            indices = f"{left},{right}"
            left = right = ""
        self.app.tasks.start(
            _actions.camera_modes(left=left or "left", right=right or "right",
                                  indices=indices),
            "相机实际模式探测", on_done=self._on_modes_probed,
            owner=self.task_owner)
        self.status.set_status(
            "正在逐档真开相机实测（慢相机首帧可能要几十秒；"
            "只读，不写 config）…", "info")

    def _on_modes_probed(self, task) -> None:
        """解析 `cameras.py modes --json` 的**最后一行 JSON**，两台一致才填入。"""
        if getattr(self, "_closed", False):
            return
        payload = None
        for line in reversed(list(getattr(task, "tail", []) or [])):
            text = str(line).strip()
            if not text.startswith("{"):
                continue
            try:
                payload = json.loads(text)
                break
            except ValueError:
                continue
        if not isinstance(payload, dict):
            self.status.set_status(
                f"实际模式探测没返回 JSON（退出码 {task.exit_code}）："
                f"{task.tail_text(3)[:110]} —— 开日志看完整输出", "warn")
            return
        roles = list(payload.get("roles") or [])
        errors = list(payload.get("errors") or [])
        if len(roles) != 2:
            self.status.set_status(
                "实际模式探测没有拿到两路：" + ("；".join(errors) or "roles 不足 2 条"),
                "warn")
            return
        bests = []
        for item in roles:
            best = item.get("best") or {}
            if not best.get("ok"):
                self.status.set_status(
                    f"{item.get('role') or '?'} 没有任何一档能出图："
                    + (str(best.get("error") or "见日志")[:80]), "warn")
                return
            bests.append((item.get("role") or "?", best))
        (role_a, a), (role_b, b) = bests
        size_a = (int(a.get("width") or 0), int(a.get("height") or 0))
        size_b = (int(b.get("width") or 0), int(b.get("height") or 0))
        if size_a != size_b:
            self.status.set_status(
                f"两台实际档不一致：{role_a}={size_a[0]}x{size_a[1]}，"
                f"{role_b}={size_b[0]}x{size_b[1]}；双路必须同档，未填入。"
                "请把两台设成同一个模式后再试", "warn")
            return
        if size_a[0] <= 0 or size_a[1] <= 0:
            self.status.set_status("实际模式探测返回了 0x0，未填入", "warn")
            return
        self.var_cap_width.set(str(size_a[0]))
        self.var_cap_height.set(str(size_a[1]))
        fps_a = float(a.get("fps") or 0.0)
        fps_b = float(b.get("fps") or 0.0)
        measured = bool(a.get("fps_measured")) and bool(b.get("fps_measured"))
        if measured and fps_a > 0 and fps_b > 0:
            fps = min(fps_a, fps_b)
            self.var_cap_fps.set(f"{fps:.3g}")
            fps_note = f"@{fps:.1f}fps"
        else:
            self.var_cap_fps.set("")
            fps_note = "（fps 未测到，留空由后端决定）"
        self.status.set_status(
            f"已填入 {size_a[0]}x{size_a[1]}{fps_note}：两台实测一致"
            + (f"；{len(errors)} 条提示见日志" if errors else ""), "ok")

    def _fallback_method(self) -> str:
        """下拉 label 匹配不上时的兜底键：按**本窗口模式**走，不写死。

        两处兜底以前不一致（提示用 chain、取键用 joint）——那意味着同一份
        界面状态在"显示说明"和"拼命令行"两条路上会读到不同的方法。
        """
        return "joint" if self.mode == "parallel" else "chain_dual"

    def _refresh_method_hint(self) -> None:
        if self.lbl_method is None:
            return
        label = self.var_method.get()
        key = next((k for k, v in METHODS.items() if v[0] == label),
                   self._fallback_method())
        self.lbl_method.configure(text=METHODS[key][1].replace("**", ""))

    def _on_method_changed(self) -> None:
        """切换标定方法：刷新说明 + 按需展示该方法的专属输入。

        专属控件**第一次选中时才创建**：默认方法（joint / chain_dual）下它们
        根本不存在，所以 `selftest_gui_budget` 量到的窗口结构与改动前逐字相同，
        不用为了新方法抬高那条已被反复放宽的预算上限。
        """
        self._refresh_method_hint()
        key = self._method_key()
        self._ensure_method_controls(key)
        self._show_method_controls(key)
        # device90 只有 ①② 两步：（重拍某一步 / 检查链式输入）里的 ③④ 对它
        # 没有意义，置灰避免用户走进一条必然报"缺参考对"的死路。
        self._sync_chain_buttons(key)

    def _sync_chain_buttons(self, key: str) -> None:
        for attr in ("btn_reshoot_chain", "btn_check_chain"):
            btn = getattr(self, attr, None)
            if btn is not None:
                btn.configure(state=("disabled" if key == "device90"
                                     else "normal"))

    def _ensure_method_controls(self, key: str) -> None:
        """懒建 axial / device90 的专属输入行，放进现有「高级」折叠区。"""
        if key not in ("axial", "device90"):
            return
        if getattr(self, "_method_rows", None) is None:
            self._method_rows = {}
        if key in self._method_rows:
            return
        inner = getattr(self, "_adv_content", None)
        if inner is None:
            return
        from tkinter import filedialog
        row = ttk.Frame(inner)
        if key == "axial":
            ttk.Label(row, text="采集根目录（其下 z000mm/{left,right}/…）").pack(
                side="left")
            ttk.Entry(row, textvariable=self.var_axial_root, width=24).pack(
                side="left", padx=(6, 0))
            ttk.Button(row, text="浏览…", command=lambda: self._pick_dir(
                self.var_axial_root)).pack(side="left", padx=(4, 0))
            ttk.Checkbutton(
                row, text="机械保证两光轴平行（强制 R=I）",
                variable=self.var_axial_parallel).pack(side="left", padx=(12, 0))
        else:
            ttk.Label(row, text="内参文件").pack(side="left")
            ttk.Entry(row, textvariable=self.var_d90_intrinsics, width=24).pack(
                side="left", padx=(6, 0))
            ttk.Button(row, text="浏览…", command=lambda: self._pick_file(
                self.var_d90_intrinsics)).pack(side="left", padx=(4, 0))
            row2 = ttk.Frame(inner)
            ttk.Label(row2, text="中轴").pack(side="left")
            ttk.Combobox(row2, textvariable=self.var_d90_axis, width=5,
                         state="readonly", values=("x", "y")).pack(
                             side="left", padx=(6, 12))
            ttk.Label(row2, text="旋转角 °").pack(side="left")
            ttk.Entry(row2, textvariable=self.var_d90_angle, width=6).pack(
                side="left", padx=(6, 0))
            ttk.Label(row2, text="Z_A mm（可空）").pack(side="left", padx=(12, 0))
            ttk.Entry(row2, textvariable=self.var_d90_za, width=7).pack(
                side="left", padx=(6, 0))
            ttk.Label(row2, text="Z_B mm（可空）").pack(side="left", padx=(12, 0))
            ttk.Entry(row2, textvariable=self.var_d90_zb, width=7).pack(
                side="left", padx=(6, 0))
            self._method_rows["device90_extra"] = row2
        self._method_rows[key] = row

    def _show_method_controls(self, key: str) -> None:
        """只显示当前方法的专属行；其余 pack_forget（不销毁，值保留）。

        `adv.content` 里的既有控件全是 `pack` 管理的，这里必须用同一套
        geometry manager；混用 `grid` 会被 Tk 直接拒绝（TclError）。
        """
        rows = getattr(self, "_method_rows", {})
        for k in ("axial", "device90", "device90_extra"):
            w = rows.get(k)
            if w is None:
                continue
            if k == key or (key == "device90" and k == "device90_extra"):
                w.pack(anchor="w", pady=(6, 0))
            else:
                w.pack_forget()

    @staticmethod
    def _browse_start(value: str, fallback: Path) -> str:
        """文件选择框的起点：当前路径存在就用它，否则回项目内目录。

        不传 ``initialdir`` 时 Windows 会落在“上次用过的文件夹”，可能是
        `文档` / 桌面；用户看到的就是“浏览默认怎么跑到项目外去了”。
        """
        raw = (value or "").strip()
        p = Path(raw).expanduser() if raw else None
        if p is not None:
            if p.is_file():
                return str(p.parent)
            if p.is_dir():
                return str(p)
        return str(fallback if fallback.is_dir() else PROJECT_ROOT)

    def _pick_dir(self, var: tk.StringVar) -> None:
        from tkinter import filedialog
        got = filedialog.askdirectory(
            parent=self,
            initialdir=self._browse_start(var.get(), CHAIN_ROOT))
        if got:
            var.set(got)

    def _pick_file(self, var: tk.StringVar) -> None:
        from tkinter import filedialog
        # 内参文件是 StereoRig 的 .npz；起点固定为项目的 config/calib，
        # 不跟随 Windows 的“上次目录”（否则可能在文档/桌面）。
        got = filedialog.askopenfilename(
            parent=self,
            initialdir=str(PROJECT_ROOT / "config" / "calib"),
            filetypes=[("标定文件", "*.npz *.json"), ("全部", "*.*")])
        if got:
            var.set(got)

    def _method_key(self) -> str:
        label = self.var_method.get()
        return next((k for k, v in METHODS.items() if v[0] == label),
                    self._fallback_method())

    def _reload_state(self) -> None:
        self.app.state.reload()
        self._refresh_state()
        # 档案可能换了（换装置/重新标定），FOV 提示要跟着变。
        self._sync_fov_hint()

    # ------------------------------------------------------------------ #
    # 动作
    # ------------------------------------------------------------------ #
    def _open_cameras(self) -> None:
        """打开相机管理对话框（GUI 内，2026-11-07 改）。

        原来这里起控制台子进程 + OpenCV 窗口，用户连着两次报「打不开」。
        现在改成 Tk 对话框：预览在后台线程、绑角色按设备身份直接落盘，
        不起子进程、不按键、不会等 4×30 秒。
        """
        from apps.cameras_dialog import open_cameras

        open_cameras(self, on_changed=self._refresh_state)
        self.status.set_status(
            "相机管理已打开：选一台点「预览这台」看画面；"
            "或直接「设为左 / 设为右」按身份绑定，关掉对话框即生效", "info")

    def _check_setup(self) -> None:
        """① 检查装置：带**本窗口选的**左右相机，并把结论写回状态条。

        为什么不能像上一版那样空手起进程（2026-10-31 缺陷 #4）
        --------------------------------------------------
        上一版 ``tasks.start([python, check_stereo_setup.py], "双目装置检查")``：
        一个参数都没有，``on_done`` 也没有。后果有两条，用户都看得见：

        1. 检查程序去查的相机不一定是第 ② 步要标定的那两台 —— 结论对了也是
           "另一套装置"的结论；
        2. 进程跑完（甚至退出）界面**一个字都不变**：状态条永远停在
           "正在检查装置…"，用户只能猜。

        现在：左右相机（与 ``calibrate_stereo`` 同一份取值）和棋盘参数一起
        下传，``on_done`` 按退出码回写。
        """
        left, right = self.var_left.get().strip(), self.var_right.get().strip()
        if not left or not right:
            # 正常路径下按钮这时已经是灰的（见 `_refresh_state`）；
            # 这里是"填了又清空"的兜底 —— 与其起一个必然白开的进程，不如说清原因
            self.status.set_status(
                "「检查装置」要知道检查哪两台相机：先在「高级：采集门限与相机」"
                "里填上左/右（角色名或索引）", "warn")
            return
        cmd = check_setup_args(
            left=left, right=right,
            board={"cols": self.var_cols.get(), "rows": self.var_rows.get(),
                   "square": self.var_square.get()})
        self.app.tasks.start(cmd, "双目装置检查", on_done=self._on_checked,
                             owner=self.task_owner)
        self.status.set_status(
            f"正在检查装置 {left} / {right}（只报告，不改配置）：会开两路取景窗，"
            f"举着棋盘慢慢移动，按 q 可提前结束。结论会写回这里", "info")

    def _on_checked(self, task) -> None:
        """把「检查装置」的结果写回状态条 —— 成功/失败都要写，不能只有"正在…"。"""
        code = task.exit_code
        if code == 0:
            self.status.set_status(
                "装置检查通过：两路都能稳定看到棋盘，可以做第 ② 步标定"
                "（逐项数字在运行日志里）", "ok")
        elif code == 1:
            self.status.set_status(
                "装置检查没通过：运行日志里「标定前置检查结果」会写清是哪一路没看到"
                "棋盘、重叠区多大", "warn")
        elif code is None:
            self.status.set_status("装置检查被中断了，没有结论（可以再点一次）", "warn")
        else:
            self.status.set_status(
                f"装置检查没能跑起来（退出码 {code}）：多数是相机名/索引打不开，"
                f"输出在运行日志里", "warn")

    def _start_calibrate(self, force_save: bool = False) -> None:
        method = self._method_key()
        # 标定写进「当前装置档案」的 calibration_path（若有），否则走默认
        # config/stereo_calibration。这样平行装置和 90° 装置各存各的标定，
        # 换装置重新标定不会覆盖另一套的数据。
        prof = getattr(self.app.state.profile, "calibration_path", None)
        # 2026-11-07：档案不再存"期望基线"，这里也不再自动下传 --baseline-mm。
        # 它是**这一次**卷尺量的物理核对锚，不是装置属性；需要时由 CLI 显式传。
        # FOV：**由第②步的勾选决定用不用**（2026-11-04 用户："FOV 有点影响
        # 平行双目标定了"）。勾着且档案里量过就复用档案参数；不勾就明确传
        # `--skip-fov`，两路按原始画面标定。
        # **只在档案真的存了 fov 时才给名字**——否则后端会报"档案里还没有
        # FOV 参数"而拒绝标定，等于把可选项变成了必选项。
        prof_obj = self.app.state.profile
        has_fov = bool(getattr(prof_obj, "fov", None))
        fov_prof = (getattr(prof_obj, "name", "") or "") if has_fov else ""

        # ---- 两种新标定方式：走独立 CLI，不碰 joint/chain ----
        # 它们在各自窗口的下拉里，几何前提不同（轴向=平行正对+已知推进，
        # 中轴90°=大角度+每相机各一次），所以 argv 由 _actions 的两个专用
        # 函数拼，绝不塞进 calibrate_args（那里三条分支已很密）。
        board_now = {"cols": self.var_cols.get(), "rows": self.var_rows.get(),
                     "square": self.var_square.get(),
                     "target": self.var_target.get()}
        if method == "axial":
            root = self.var_axial_root.get().strip()
            if not root or not Path(root).is_dir():
                self.status.set_status(
                    "先在「高级」里选采集根目录（其下按 z000mm/z010mm… 分深度）",
                    "warn")
                return
            cmd = calibrate_axial_args(
                root=root, board=board_now, out=(prof or ""),
                assume_parallel=bool(self.var_axial_parallel.get()),
                expected_baseline_mm=self.var_baseline.get(),
                force_save=force_save)
            self.status.set_status("正在按深度目录做轴向标定…", "info")
            self.app.tasks.start(cmd, "双目标定", on_done=self._on_calibrated,
                                 owner=self.task_owner)
            return
        if method == "device90":
            d90 = current_chain_dirs()
            if d90 is None:
                self.status.set_status(
                    "还没有标定会话：先点「采集标定板图…」拍 ①②（A 板 / B 板）",
                    "warn")
                return
            need = {"left": "① A 板", "board_b": "② B 板"}
            missing = [friendly for key, friendly in need.items()
                       if not d90[key].is_dir() or not any(d90[key].iterdir())]
            if missing:
                self.status.set_status(
                    "还差 " + "、".join(missing) + " 没拍；中轴90°模式不需要 ③④",
                    "warn")
                return
            intr = self.var_d90_intrinsics.get().strip()
            if not intr or not Path(intr).exists():
                self.status.set_status(
                    "先在「高级」里选内参文件（轴向标定产物或既有 .npz）", "warn")
                return
            cmd = calibrate_device90_args(
                left_dir=str(d90["left"]), right_dir=str(d90["board_b"]),
                intrinsics=intr, board=board_now, out=(prof or ""),
                rot_axis=self.var_d90_axis.get().strip() or "y",
                rot_angle=self.var_d90_angle.get().strip() or "90",
                z_a_mm=self.var_d90_za.get().strip(),
                z_b_mm=self.var_d90_zb.get().strip(),
                expected_baseline_mm=self.var_baseline.get(),
                force_save=force_save)
            self.status.set_status("正在按中轴 90° 装置做标定…", "info")
            self.app.tasks.start(cmd, "双目标定", on_done=self._on_calibrated,
                                 owner=self.task_owner)
            return

        # 链式标定只认「当前会话」，绝不回落读根层（2026-11-09）。根层以前
        # 堆了好几次采集的图，标定会全读进去，静默用错数据。
        d = current_chain_dirs() if method in ("chain", "chain_dual") else None
        if method in ("chain", "chain_dual") and d is None:
            self.status.set_status(
                "还没有标定会话：先点「采集标定板图…」拍图，每次采集单独存一个文件夹",
                "warn")
            return
        cmd = calibrate_args(
            method,
            {"cols": self.var_cols.get(), "rows": self.var_rows.get(),
             "square": self.var_square.get(), "target": self.var_target.get()},
            self.var_left.get(), self.var_right.get(), self.mode,
            stationary=self.var_stationary.get(),
            sharpness=self.var_sharpness.get(),
            relaxed=bool(self.var_relaxed.get()),
            out=(prof or ""), fov_from_profile=fov_prof,
            use_fov=bool(self.var_use_fov.get()),
            baseline=self.var_baseline.get(), dirs=d,
            force_save=force_save,
            width=self.var_cap_width.get(),
            height=self.var_cap_height.get(),
            fps=self.var_cap_fps.get())
        if method in ("chain", "chain_dual"):
            # 先查图在不在，缺哪段直接说，别让后端跑一半才报"缺图"。
            if method == "chain_dual":
                # 双板四步（2026-11-09 方案 A）：③ 参考对是必需输入 —— 链两端
                # 位姿必须来自同一支架摆放，缺了会静默给错外参，所以这里拦住。
                need = ("left", "board_b", "ref_pair_left", "ref_pair_right", "pair")
                friendly = {"left": "① A 板", "board_b": "② B 板",
                            "ref_pair_left": "③ 参考对·左",
                            "ref_pair_right": "③ 参考对·右", "pair": "④ 两板同框"}
            else:
                need = ("left", "right", "ref_left", "ref_right")
                friendly = {"left": "① 左内参", "right": "② 右内参",
                            "ref_left": "③ 参考·左", "ref_right": "③ 参考·右"}
            missing = [friendly[k] for k in need if not d[k].is_dir()
                       or not any(d[k].iterdir())]
            if missing:
                self.status.set_status(
                    "还差 " + "、".join(missing)
                    + " 没拍。点「采集标定板图…」按顺序补齐，再「检查链式输入」",
                    "warn")
                return
            self.status.set_status(
                f"正在标定会话 {d['left'].parent.name} 的图…", "info")
            # 拍图规格 vs 当前填写：不一致只提示、不拦（真正拦截在 CLI 侧）。
            # 放在 spawn 前的最后一条状态，免得被"正在标定…"盖掉。
            try:
                _sess = json.loads(
                    (d["left"].parent / "session.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                _sess = {}
            _rec = _sess.get("board_spec")
            if isinstance(_rec, dict):
                try:
                    _fill = (int(str(self.var_cols.get()).strip()),
                             int(str(self.var_rows.get()).strip()),
                             float(str(self.var_square.get()).strip()))
                    _same = (_fill == (int(_rec["cols"]), int(_rec["rows"]),
                                       float(_rec["square_mm"])))
                except (KeyError, TypeError, ValueError):
                    _same = True       # 填了非法值让 CLI 去报，不在这里误报
                if not _same:
                    self.status.set_status(
                        f"本次拍图规格 {_rec['cols']}x{_rec['rows']} / "
                        f"{_rec['square_mm']:g} mm，当前填写 "
                        f"{self.var_cols.get()}x{self.var_rows.get()} / "
                        f"{self.var_square.get()} mm；标定会按会话规格核对，不一致会报错",
                        "warn")
        else:
            self.status.set_status("正在标定：会打开两个取景窗，板停稳自动存图",
                                   "info")
        self.app.tasks.start(cmd, "双目标定", on_done=self._on_calibrated,
                             owner=self.task_owner)

    def _force_calibrate(self) -> None:
        """强制把「质量差」的标定写进正式路径（仅链路测试）。

        后端是子进程（_task_runner 的 Popen），GUI 只负责在 argv 带
        --force-save；旁路/正式路径的切换、forced_unusable 标记都在
        calibrate_stereo.finalize_and_save 里，这里不 import StereoRig、
        也不自己调 finalize_and_save。
        """
        from tkinter import messagebox
        # 标题/正文先落成变量再传：i18n 提取器只收 askyesno("…") 里
        # **字面量**位置的文案，变量不会被当成"新文案"要求登记（同 _start_record）。
        # 路径与判定数字是运行时值，但**标签/说明都留在字面量里**（不在变量里拼整句），
        # 这样提取器仍能从源码拿到可登记的文案片段（见 tests 的 i18n 字面量判据）。
        formal_path = str(self.app.state._stereo_calib_path())
        rot_txt, trans_txt = "未知", "未知"
        try:
            _js = _fov_store.calibration_json_path(formal_path)
            _data = json.loads(Path(_js).read_text(encoding="utf-8"))
            _cons = (_data.get("quality") or {}).get("consistency") or {}
            if _cons.get("rot_max_deg") is not None:
                rot_txt = f"{float(_cons['rot_max_deg']):.3f}"
            if _cons.get("trans_max_mm") is not None:
                trans_txt = f"{float(_cons['trans_max_mm']):.3f}"
        except Exception:                                # noqa: BLE001
            pass
        title = "强制通过标定？"
        head = (f"当前正式标定路径：{formal_path}；"
                f"本次判定 rot_max_deg={rot_txt}° / trans_max_mm={trans_txt} mm；"
                "会先把旧标定备份到同目录（_before_force_时间戳，.npz/.json 两份）再写入。")
        body = (head + "\n\n"
                "这一路会把标定结果写入上述正式标定路径；"
                "不加 --force-save 时，质量差的产物只会另存为旁路文件，"
                "激活装置档案绝不改动。\n\n"
                "产物会被标为不可用（quality.forced_unusable=True），"
                "下游录制会拒绝使用它；verdict 写明「仅链路测试，不得用于数据结论」。\n\n"
                "只用于验证标定链路跑不跑得通，不得用于数据结论。仍要继续吗？")
        if not messagebox.askyesno(title, body, parent=self):
            cancel = ("已取消：按默认旁路处理（不加 --force-save，质量差时另存旁路、"
                      "不覆盖正式标定）；要普通标定请点「开始标定」")
            self.status.set_status(cancel, "warn")
            return
        self._start_calibrate(force_save=True)

    def _chain_common(self) -> dict:
        """四步采集与「只重拍某一步」共用的参数（唯一产地）。

        抽出来是为了让全量采集与单步重拍**不会各拼一份 argv**：板参数、左右
        相机、采集档、分化线、档案名都只在这里算一次。
        """
        board_base = {"cols": self.var_cols.get(), "rows": self.var_rows.get(),
                      "square": self.var_square.get()}
        left, right = self.var_left.get(), self.var_right.get()

        # 采集档：拍标定图必须与之后录制**同一档**。`capture_board.py` 默认
        # 2560x1440，而本机实测双路同时开 2560 会让一路 MSMF 会话失效
        # （2026-11-05；1920x1080 则两台 15/15 帧稳定）。
        #
        # 优先级（2026-11-10 定稿）：**装置档案的 width/height** → 档案没填
        # 就用 1080 安全档（apps/_common.capture_size_from_profile）。
        # 不再从标定 json 的 capture_size 取：档案配了 calibration_path 但还没
        # 标定时那份 json 不存在，旧写法读不到 → cap_kw 不带宽高 →
        # capture_board 悄悄用 2560 默认档，正好踩上面的硬件坑。
        # 安全档对象取自 video_profiles（1920/1080 数字的唯一来源）。
        # 「装置档案优先」原本就是这里的意图，这次是补齐它、并把最后的兜底
        # 从「不传（等于悄悄用 2560）」改成显式的 1080 安全档。
        from apps._common import capture_size_from_profile

        cap_kw: dict[str, int | float] = {}
        _prof = getattr(self.app.state, "profile", None)
        _fps = getattr(_prof, "fps", None)
        if isinstance(_fps, int) and _fps > 0:
            cap_kw["fps"] = _fps
        _w, _h, _cap_note = capture_size_from_profile(_prof)
        cap_kw["width"], cap_kw["height"] = _w, _h
        # 界面手填优先（②/③ 共用同一组输入框）。只填一半或填错 -> 明确说，
        # 本次仍按档案/安全档拍，不静默用半套值。
        _fw, _fh, _ff, _cap_err = self._cap_mode_fields()
        if _cap_err:
            _cap_note = f"{_cap_note}；**界面采集档没采用：{_cap_err}**"
        elif _fw is not None:
            cap_kw["width"], cap_kw["height"] = _fw, _fh
            _cap_note = f"界面手填 {_fw}x{_fh}"
            if _ff is not None:
                cap_kw["fps"] = _ff
                _cap_note += f" @ {_ff:g}fps"
        # 显式提示：来源（含"没填 → 安全档"及其硬件原因）必须让用户看见，
        # 不许静默落档；文案由 helper 生成，不谎称来自标定记录。
        self.status.set_status(f"采集档：{_cap_note}", "info")

        # 档案名只用于写进 session.json，供事后追认是哪套装置拍的。
        _prof_name = getattr(getattr(self.app.state, "profile", None),
                             "name", "") or ""
        # 分化线：② 里填的「目标对数」在采集这一步的含义是——**拍够这么多张后，
        # 覆盖标准从 5/5 放宽到 4/5**。它不是"拍摄总对数"（2026-11-05 用户澄清）。
        # 空/非数字按 40（与输入框默认一致）；≤0 视为"不放宽"。
        try:
            relax_after = int(str(self.var_target.get()).strip() or "40")
        except ValueError:
            relax_after = 40
        if relax_after < 0:
            relax_after = 0
        return {"board_base": board_base, "left": left, "right": right,
                "cap_kw": cap_kw, "relax_after": relax_after,
                "prof_name": _prof_name}

    def _chain_steps(self, d, *, left, right, board_base, cap_kw, relax_after):
        """四步 / 三步的 **(标签, 目录键元组, argv)** 唯一产地。

        全量采集按它串行跑完所有步；「只重拍某一步」只取其中一条 argv、并用
        同一条里的目录键元组把该步旧图移进回收站。目录键是这段流程对
        ``chain_dirs()`` 的唯一映射，③ 的参考对必须**成对**成两个目录。
        """
        if self._method_key() == "device90":
            # 中轴 90° 装置：只拍 ①② 两段正对图（板 A 正对相机 1、绕中轴转
            # 90° 后板 B 正对相机 2）。**不需要** ③ 参考对 / ④ 同框图 ——
            # 这正是该模式比 chain_dual 省的两步，T_AB 由装置 90° 约束给出。
            # 目录键必须与 _start_calibrate 的 device90 分支一致。
            steps = [
                ("① 拍 A 板：正对相机 1（板别动，多拍几张）", ("left",),
                 capture_chain_frames(stage="left", left=left, right=right,
                                      target="999", min_cover=5,
                                      relax_after=relax_after, per_region=3,
                                      dirs=d, **board_base, **cap_kw)),
                ("② 绕中轴转 90°，拍 B 板：正对相机 2", ("board_b",),
                 capture_chain_frames(stage="dual_b", left=left, right=right,
                                      target="999", min_cover=5,
                                      relax_after=relax_after, per_region=3,
                                      dirs=d, **board_base, **cap_kw)),
            ]
        elif self._method_key() == "chain_dual":
            # 双板四步。①② 的板可以随便挪（只用来求内参）；③ 必须**摆好别动**
            # 且两路同时拍（链两端的位姿来自同一个摆放）；④ 同框图检不了单板，
            # 走 capture_board 的 --pair 模式（不检板、按空格/自动存）。
            # 步骤标签是给用户看的，一律说人话、不出现 T_AB / 姿态簇这类术语。
            steps = [
                ("① 拍 A 板：只用左相机，板随便挪", ("left",),
                 capture_chain_frames(stage="left", left=left, right=right,
                                      target="999", min_cover=5,
                                      relax_after=relax_after, per_region=3,
                                      dirs=d, **board_base, **cap_kw)),
                ("② 拍 B 板：只用右相机，板随便挪", ("board_b",),
                 capture_chain_frames(stage="dual_b", left=left, right=right,
                                      target="999", min_cover=5,
                                      relax_after=relax_after, per_region=3,
                                      dirs=d, **board_base, **cap_kw)),
                ("③ 组合装置摆好别动，两路同时各拍 3 张",
                 ("ref_pair_left", "ref_pair_right"),
                 capture_chain_frames(stage="ref_pair", left=left, right=right,
                                      target="3", dirs=d, **board_base, **cap_kw)),
                ("④ 让一台相机同时看到两块板，按空格拍 5 张（至少 2 张）",
                 ("pair",),
                 capture_pair_frame(camera=left, dirs=d, **board_base, **cap_kw)),
            ]
        else:
            steps = [
                ("内参段·左（板可移动，多姿态）", ("left",),
                 capture_chain_frames(stage="left", left=left, right=right,
                                      target="999", min_cover=5,
                                      relax_after=relax_after, per_region=3,
                                      dirs=d, **board_base, **cap_kw)),
                ("内参段·右（板可移动，多姿态）", ("right",),
                 capture_chain_frames(stage="right", left=left, right=right,
                                      target="999", min_cover=5,
                                      relax_after=relax_after, per_region=3,
                                      dirs=d, **board_base, **cap_kw)),
                ("参考段（板**固定不动**，左右成对拍）",
                 ("ref_left", "ref_right"),
                 capture_chain_frames(stage="ref", left=left, right=right,
                                      target="3", dirs=d, **board_base, **cap_kw)),
            ]
        return steps

    def _capture_chain_frames(self) -> None:
        """采集链式标定的图（按当前方法分单板/双板两套流程）。

        - 单板（chain）：① 内参左 → ② 内参右 → ③ 参考段（板不动，左右成对）；
        - 双板（chain_dual，2026-11-09 方案 A）：① 拍 A 板 → ② 拍 B 板 →
          ③ **组合装置摆好别动、两路同时各拍 3 张**（参考对）→
          ④ 一台相机同时拍到两板（同框图）。
          ③ 不能省：链两端的板位姿必须来自同一支架摆放，拿 ①② 里板移动中的
          各一帧来凑，外参会静默错到 10° / 28 mm 量级。
        每段拍完回到界面继续下一段（on_done 串起来，避免取景窗抢相机）。
        """
        c = self._chain_common()
        # 一次采集 = 一个会话目录（2026-11-09）：四段图全落进同一个文件夹，
        # 自检与标定都读根层 _current.json 指向它。以前追加到根层的
        # left/board_b，多次标定的图会混在一起被 _run_chain 全读进去 ——
        # 真机实测 board_b 344 张里混了 11-04 的 304 张和 11-08 的 40 张。
        d = new_chain_session(profile=c["prof_name"], mode=self.mode,
                              board=c["board_base"], capture=c["cap_kw"])
        for p in d.values():
            p.mkdir(parents=True, exist_ok=True)
        print(f"[标定会话] {d['left'].parent.name} → {d['left'].parent}",
              flush=True)
        # 步骤表来自唯一产地；这里只把 (标签, argv) 取出来串行跑，不改 argv。
        steps = [(lbl, cmd) for lbl, _keys, cmd in self._chain_steps(
            d, left=c["left"], right=c["right"], board_base=c["board_base"],
            cap_kw=c["cap_kw"], relax_after=c["relax_after"])]
        self._chain_step = 0

        def run_next(_task=None):
            # 关窗后不能再起下一段：on_done 递归是"上一段结束"触发的，
            # 而关窗时 close_window 停掉当前段，结束回调仍会排到队列里。
            # 没有这道旗标，用户按了 X 之后链式采集会自己一段段接着跑 ——
            # 最像"关不掉 / 自动复苏"的那个现象。
            if getattr(self, "_closed", False):
                return
            if self._chain_step >= len(steps):
                if self._method_key() == "chain_dual":
                    # 同框图也是【左相机】拍（后端拼 --pair-camera left）：
                    # A/B 的判定用的是"左相机看 A 板"的姿态簇，换右相机拍，
                    # 姿态簇坐标系就变了——两块板差 90° 时那种错误不会触发
                    # "差 <5°"护栏，会静默给出反向的 T_AB。
                    self.status.set_status(
                        "四步都拍完了。点「检查链式输入」，再点「开始标定」", "info")
                else:
                    self.status.set_status(
                        "三步都拍完了。点「检查链式输入」，再点「开始标定」", "info")
                return
            label, cmd = steps[self._chain_step]
            self._chain_step += 1
            self.status.set_status(f"[{self._chain_step}/{len(steps)}] {label}。"
                                   "取景窗里板停稳会自动存图；拍完关窗回到这里", "info")
            self.app.tasks.start(cmd, label, on_done=run_next,
                                 owner=self.task_owner)

        run_next()

    def _reshoot_chain_stage(self) -> None:
        """只重拍链式标定的某一步：该步旧图移到回收站，其余段原样保留。

        与全量采集的区别：这里**不新建会话**，``_current.json`` 指针与其余段
        的图都留着，所以旧会话里前几段的有效图对程序仍然可见。重拍完只要
        重新跑一次「检查链式输入」和「开始标定」即可 —— 标定本来就全量读
        五段，不需要为了一步没拍好把四步全部重来。
        """
        d = current_chain_dirs()
        if d is None:
            self.status.set_status(
                "还没有标定会话：先点「采集标定板图…」拍图，每次采集单独存一个文件夹",
                "warn")
            return
        c = self._chain_common()
        steps = self._chain_steps(d, left=c["left"], right=c["right"],
                                  board_base=c["board_base"], cap_kw=c["cap_kw"],
                                  relax_after=c["relax_after"])

        def preview(idx: int) -> str:
            names: list[str] = []
            n = 0
            for k in steps[idx][1]:
                p = d[k]
                n += sum(1 for x in p.iterdir() if x.is_file()) if p.is_dir() else 0
                names.append(f"{d['left'].parent.name}/{k}/")
            return (f"将把 {'、'.join(names)} 共 {n} 个文件移到 outputs/_trash/ "
                    "下的回收批次（数据管理「回收站…」不管理这类批次），"
                    f"然后只重拍第 {idx + 1} 步")

        dlg = tk.Toplevel(self)
        dlg.title("重拍某一步")
        dlg.configure(bg=BG)
        dlg.resizable(True, True)
        body = ttk.Frame(dlg, padding=14)
        body.pack(fill="both", expand=True)
        ttk.Label(body, wraplength=420, foreground=MUTED,
                  text="选一步重拍：该步旧图不删除，会移到 outputs/_trash/ 下的批次；"
                       "数据管理里的「回收站…」只管 data/_trash/，不管理这类批次。"
                       "旧图可用 apps._actions.restore_chain_stage(<批次路径>) 恢复，"
                       "一旦该步已拍了新图，恢复会被整批拒绝以免覆盖。其余段原样保留，"
                       "整条链只需重跑一次「开始标定」，不用重拍其它段。").pack(anchor="w", pady=(0, 8))
        var = tk.IntVar(value=0)
        preview_lbl = ttk.Label(body, wraplength=420, foreground=WARN)
        preview_lbl.pack(anchor="w", fill="x", pady=(8, 2))

        def sync_preview() -> None:
            preview_lbl.configure(text=preview(var.get()))

        for i, (lbl, _keys, _cmd) in enumerate(steps):
            ttk.Radiobutton(body, text=f"第 {i + 1} 步 · {lbl}", value=i,
                            variable=var, command=sync_preview).pack(anchor="w")
        sync_preview()
        out: dict[str, int | None] = {"idx": None}
        bar = ttk.Frame(body)
        bar.pack(fill="x", pady=(10, 0))

        def ok() -> None:
            out["idx"] = var.get()
            dlg.destroy()

        ttk.Button(bar, text="确定", command=ok).pack(side="right")
        ttk.Button(bar, text="取消", command=dlg.destroy).pack(side="right", padx=(0, 6))
        self.wait_window(dlg)
        idx = out["idx"]
        if idx is None:
            return
        lbl, keys, cmd = steps[idx]
        step_no = idx + 1
        info = _actions.reshoot_chain_stage(d, keys, label=f"重拍第{step_no}步")
        self.app.tasks.start(cmd, lbl, owner=self.task_owner)
        batch_txt = (str(info["batch"]) if info["batch"] is not None
                     else "（该步目录原本为空，无需移走）")
        self.status.set_status(
            f"只重拍了第 {step_no} 步，其余段保留；旧图在 outputs/_trash/ 批次 "
            f"{batch_txt}（数据管理「回收站…」不管理，可用 "
            "apps._actions.restore_chain_stage(<批次路径>) 恢复，该步已有新图则拒绝恢复）。"
            "拍完请重新点「检查链式输入」和「开始标定」", "info")

    def _check_chain_inputs(self) -> None:
        """拍完先自检：每一步的图齐不齐、板看没看全、装置有没有动。"""
        d = current_chain_dirs()
        if d is None:
            self.status.set_status(
                "还没有标定会话：先点「采集标定板图…」拍图，每次采集单独存一个文件夹",
                "warn")
            return
        dual = self._method_key() == "chain_dual"
        cmd = check_chain_inputs(
            cols=self.var_cols.get(), rows=self.var_rows.get(),
            square=self.var_square.get(), mode="dual" if dual else "single",
            dirs=d)
        if dual:
            self.status.set_status(
                "检查四步图（姿态覆盖 / 第 ③ 步参考对是否成对 / "
                "同框图能否检出两板）…", "info")
        else:
            self.status.set_status(
                "检查三步图（姿态覆盖 / 参考段板是否移动）…", "info")
        self.app.tasks.start(cmd, "链式输入自检", owner=self.task_owner)

    def _record_fov_to_profile(self) -> str:
        """把本次标定产物里的 ``meta.fov`` 记进当前装置档案（自动收尾）。

        ## 为什么自动做

        两路焦距差是**装置属性**：量一次就该锁住，之后每次标定复用同一组
        裁剪参数（口径不漂）。靠"记得去点一下保存"迟早会漏，而漏掉不会报错
        —— 只会让下一次标定又现场估一遍，`crop_rect` 悄悄变几像素，内参跟着
        变一点点，谁也不会发现。

        `meta.fov` 为 null（旧流程 / `--skip-fov` / 现场估计降级）时**不写**：
        把"没有"覆盖成"有"、或把"有"覆盖成"没有"，都会让复用逻辑拿到错前提。
        读写细节在 `apps/_fov_store.py`（FOV 管理器用的是同一份）。
        """
        prof = getattr(self.app.state, "profile", None)
        if prof is None or not (getattr(prof, "name", "") or ""):
            return ""
        json_path = _fov_store.calibration_json_path(
            self.app.state._stereo_calib_path())
        note = _fov_store.record_fov_from_calibration(prof, json_path)
        if note and not note.startswith("（"):
            self.app.state.reload()
        return note

    def _on_calibrated(self, task) -> None:
        self.app.state.reload()
        # 质量状态只记录并显示，不阻断：grade / verdict / forced_unusable
        # 都从这次落盘的标定 json 读，读不到就降级为空串。
        quality_note = self._calib_quality_note()
        # 只有真的标定成功才收 FOV：失败时 json 还是上一次的，照抄会把旧参数
        # 记成"这次的"。
        if getattr(task, "exit_code", None) == 0:
            note = self._record_fov_to_profile()
            if note:
                self._refresh_state()
                tail = f"；质量状态：{quality_note}" if quality_note else ""
                self.status.set_status(f"标定完成。{note}{tail}", "ok")
                return
        self._refresh_state()
        if quality_note:
            self.status.set_status(
                f"标定结束（质量状态：{quality_note}）；详情看运行日志", "warn")

    def _calib_quality_note(self) -> str:
        """读标定 json 的 quality.grade / verdict / forced_unusable，写成一句话。

        只是**记录并显示**，不阻断任何流程；文件还没写、旧格式、字段缺失时
        一律降级为空串，绝不抛（_on_calibrated 在每条返回路径都要能跑完）。
        """
        try:
            json_path = _fov_store.calibration_json_path(
                self.app.state._stereo_calib_path())
            data = json.loads(Path(json_path).read_text(encoding="utf-8"))
            quality = data.get("quality") or {}
            bits = []
            grade = str(quality.get("grade") or "").strip()
            if grade:
                bits.append(f"等级 {grade}")
            if bool(quality.get("forced_unusable")):
                bits.append("强制通过·不可用")
            verdict = str(quality.get("verdict") or "").strip()
            if verdict:
                bits.append(verdict)
            return " / ".join(bits)
        except Exception:                                    # noqa: BLE001
            return ""

    def _start_record(self) -> None:
        # 录制仍走 `--stereo-mode auto`（= 按标定），所以不会出现"模式选错"
        # 那种必然失败；但**标定本身与本窗模式不匹配**时录出来也是白录
        # （录的是另一套几何，第 4 步必然跑不通）。这里先拦一次，文案与
        # 第 2 步的状态栏同一来源，不另写一份判据。
        ready, why = self._record_gate()
        if not ready:
            self.status.set_status(
                f"先解决标定再录：{why.replace('**', '')}", "warn")
            return
        # 2026-11-11：**强制保存的不可用标定**默认不录。它的 verdict 带着
        # 「仅链路测试」，用它录出来的几何与现场不符（15:00 会话 299 帧 0 匹配）。
        # 二次确认后也只当"仅测试录制"，并把标记一路带到 session.json。
        forced = False
        try:
            from calib.params import StereoRig
            from stereo.record_plan import is_forced_unusable
            forced = is_forced_unusable(
                StereoRig.load(self.app.state._stereo_calib_path()))
        except Exception:                                    # noqa: BLE001
            forced = False
        if forced:
            from tkinter import messagebox
            # 标题/正文先落成变量再传：i18n 提取器只收 `askyesno("…")` 里
            # **字面量**位置的文案，变量不会被当成"新文案"要求登记。
            title = "仅测试录制？"
            body = ("当前标定是「强制保存的不可用测试标定」（verdict 标着"
                    "「仅链路测试」）。\n\n"
                    "用它录出来的会话，几何与现场不符：只能做链路测试，"
                    "不得用于数据结论。\n\n仍要继续录制吗？")
            if not messagebox.askyesno(title, body, parent=self):
                cancel_msg = ("已取消：先按现场重新标定再录；"
                              "只做链路测试才选「仅测试录制」")
                self.status.set_status(cancel_msg, "warn")
                return
        out = record_out_dir(self.mode)
        cmd = record_args(left=self.var_left.get(), right=self.var_right.get(),
                          duration=self.var_seconds.get(), out=str(out),
                          mode=self.mode,
                          label=self.var_label.get(),
                          container=self.var_container.get(),
                          width=self.var_cap_width.get(),
                          height=self.var_cap_height.get(),
                          fps=self.var_cap_fps.get())
        if forced:
            cmd = [*cmd, "--allow-forced"]
        self.app.tasks.start(cmd, "双目录制",
                             on_done=lambda t: self._on_record_done(t, forced),
                             owner=self.task_owner)
        label = self.var_label.get().strip()
        # 落点写清"哪个分区 + 新时间戳目录"：目录名由录制程序生成（它同时负责
        # 重名去重），这里不猜具体时间，但必须说清"录完在第 ④ 步能选到它" ——
        # 上一版落进 `data/stereo_session`，不在扫描范围，用户录完在列表里找不到。
        self.status.set_status(
            (f"⚠ 仅测试录制（强制保存的不可用标定）：结果不得用于数据结论；"
             if forced else "")
            + f"预览窗口已开：确认两路都有画面后按【空格】开始（再倒计时进行录制）；"
            f"录 {self.var_seconds.get() or '10'} 秒 → {out.name}/ 下的新会话目录"
            f"（标签「{label or '空'}」）",
            "warn" if forced else "info")

    def _on_record_done(self, task, forced: bool = False) -> None:
        """录制结束：放了「强制保存的不可用标定」时，状态栏必须再留一句免责。

        默认姿态仍是拒绝（``record_plan.py`` 的门禁）；只有二次确认后才带
        ``--allow-forced`` 放行。放行后的会话能跑也能出结果，但几何与现场不符，
        只能做链路测试 —— 开头那句警告会被随后的操作刷掉，所以结束时补一次。
        正常录制不改状态条（任务表那一行照旧），失败也不在这里美化。
        """
        if not forced or getattr(task, "exit_code", None) != 0:
            return
        self.status.set_status(
            "录制完成：⚠ 仅测试录制（强制保存的不可用标定），结果不得用于数据结论",
            "warn")

    def _pick_session(self) -> None:
        """打开会话选择器（只列本装置分区的会话 + 回收站/改模式）。"""
        SessionPicker(self, mode=self.mode, on_pick=self._set_session,
                      current=self.var_session.get().strip(),
                      on_changed=self._on_sessions_changed)

    def _on_sessions_changed(self) -> None:
        """会话被删掉或搬走之后：当前选中的那条若已不存在，必须清掉。

        为什么不能只是"刷新一下列表"：录制/分析都是按 ``var_session`` 的路径
        起子进程的，路径已经没了还留着，症状是"点开始才报路径不存在" ——
        而真正的原因（我刚把它移进回收站）早就不在屏幕上了。
        """
        cur = self.var_session.get().strip()
        if cur and not Path(cur).is_dir():
            self.var_session.set("")
            self.status.set_status(
                "当前会话已被移走/移动，已清空选择；用「选择…」重新挑一段", "warn")
            return
        if cur:
            # 改装置模式会换目录：按 id（时间戳目录名）在新位置里找回来。
            name = Path(cur).name
            for row in scan_sessions():
                if row["id"] == name:
                    self.var_session.set(str(row["path"]))
                    break

    def _set_session(self, path: str, why: str = "") -> None:
        """选中一个会话。``why`` 非空表示"它不属于本窗模式"，必须说出来。"""
        self.var_session.set(path)
        name = Path(path).name
        notes: list[str] = []
        if why:
            notes.append(why)
        # 会话里有人点过球心 → 按 seeds.json 记的 ball_kind 恢复球种类。
        # 旧规则是"有种子 = 镜面球"；那只对金属球成立，纯色球点选后应该留在
        # color（种子取色 + 锚半径），换成 specular 反而丢掉轮廓精度。
        # 旧文件没记 ball_kind → 按 metal 兼容（历史上点选只用于镜面球）。
        if _actions.has_seeds(path):
            kind = _actions.seed_ball_kind(path)
            want = BALL_PLASTIC if kind == "plastic" else BALL_METAL
            if self.ball.kind() != want:
                self.ball.set_kind(want)
                how = ("纯色塑料球（color 会用种子取色并锚定半径）"
                       if want == BALL_PLASTIC
                       else "镜面金属球（specular 会读那些点）")
                notes.append(f"这段会话里有点选球心，球种类已切到「{how}」")
        else:
            # 球种类是"这段会话那颗球"的属性，不是用户偏好：没有 seeds.json
            # 就是新的一段，必须回到默认纯色球。否则沿用上一段金属球，
            # 新录纯色球仍跑 specular —— 症状是静默 0 检出。
            # 说明追加到 notes，由下面唯一那条 set_status 输出（不另写状态行，
            # 免得被 set_kind 触发的 _refresh_ball_note 覆盖成半截话）。
            if self.ball.kind() != BALL_PLASTIC:
                self.ball.set_kind(BALL_PLASTIC)
            notes.append("这段会话没有点选球心，球种类已复位为纯色塑料球")
        if notes:
            self.status.set_status(f"已选 {name} —— 注意：" + "；".join(notes), "warn")
        else:
            self.status.set_status(f"已选会话：{name}", "ok")

    def _refresh_ball_note(self) -> None:
        """球种/半径改了之后的回写（只往状态条写，不新增任何行）。

        为什么不下放成一行常驻说明：第④步的下沿已经贴着视口（869 px），
        多一行就把「跑离线三维」挤出屏幕 —— 而"按钮看不见"比"少一句解释"严重。
        所以：填错 → 说清哪一项不会被采用；切档 → 说清这一档的前提。
        每档的长解释在卡片说明与 ``BallSettings`` 的 docstring 里。
        """
        problem = self.ball.problem()
        if problem:
            self.status.set_status(problem, "warn")
            return
        hint = self.ball.hint()
        if hint:
            self.status.set_status(hint, "warn")

    def _start_offline(self) -> None:
        session = self.var_session.get().strip()
        if not session:
            self.status.set_status("先选一个会话（点「选择…」）", "warn")
            return
        # 实时采集落的会话没有原始左右录像：`run_stereo_video` 会在
        # `validate_session` 直接退出，用户白等一次进程启动。拦住要说清下一步。
        # 放在**所有** `tasks.start(` 之前 —— 判据按函数体的语句顺序断言这一点。
        why = realtime_session_blocker(session)
        if why:
            self.status.set_status(why, "warn")
            return
        # 填坏了的半径范围不许静默变成"不传"：那会让用户以为限定了范围，
        # 实际是全画面在找（`run_stereo_video` 的默认是 8–200 px）。
        problem = self.ball.problem()
        if problem:
            self.status.set_status(problem, "warn")
            return
        # 勾了锚点却没点选过球心：命令行会拒绝（退出码 2），但那时用户已经等了
        # 一次进程启动。能在这里说清就不要让人等 —— 这也是"按钮看起来什么都做了"
        # 那类抱怨的来源。
        if self.var_anchor.get() and not _actions.has_seeds(session):
            self.status.set_status(
                "这个会话还没有「点选球心」，半径锚点门用不了：先点「点选球心」"
                "（左右各一次），再回来勾它；不想点就把勾去掉", "warn")
            return
        seed = Path(session) / "derived" / "seeds.json"
        method = self.ball.method()
        # 镜面球没有种子 = 必然 0 配对（specular 只在高光附近找，不会自己初始化）。
        # 与其让人等一次进程启动再看到"0 匹配"，不如在这里拦住并说清下一步。
        if self.ball.kind() == BALL_METAL and not seed.is_file():
            self.status.set_status(
                "镜面金属球要先「点选球心」（左右各点一次球心+球边）："
                "specular 靠那些点跟踪，没有它们这一段会得到 0 对匹配", "warn")
            return
        use_seed = seed.is_file() and method == METHOD_METAL
        cmd = _actions.offline_stereo(
            session, mode=self.mode, anchor=self.var_anchor.get(),
            method=method, seed_json=(seed if use_seed else ""),
            preset=self.ball.preset(),
            min_radius=self.ball.min_radius(),
            max_radius=self.ball.max_radius(),
            radius_auto=self.ball.radius_auto(),
            anchor_diameter_mm=(self.var_anchor_diameter.get()
                                if self.var_anchor.get() else ""))
        tail = "（半径锚点门已开：先读会话里的点选中心）" if self.var_anchor.get() else ""
        if use_seed:
            tail += "（用会话里的点选球心跟踪）"
        self.status.set_status(
            f"正在按 {self.mode} 解算 {Path(session).name}（检测方法 {method}）{tail}…",
            "info")
        # 质量状态必须显示（2026-11-15）：run_stereo_video 质量差也返回 0、照常写
        # 三样产物，只有 traj3d.json["quality"] 记着"这份结果可不可信"。没有
        # on_done 的话状态条会一直停在"正在解算…"，用户看不到那句"仅供参考"。
        # 会话在**点按钮那一刻**就定死（闭包传 session）：跑的过程中换选中项
        # 不该把结论写到另一段会话头上。argv 一个字不动，只加回调。
        self.app.tasks.start(cmd, "离线三维重建",
                             on_done=lambda t: self._on_offline_done(t, session),
                             owner=self.task_owner)

    def _on_offline_done(self, task, session: str = "") -> None:
        """第④步跑完：把 traj3d.json 的质量结论写回状态条（成功路径）。

        只看退出码会把"质量差但照常出结果"显示成纯成功。反过来也必须守住：
        退出码非 0 / 产物读不到时**什么都不写**（任务表那一行本来就会标失败），
        绝不凭空说一句"仅供参考"—— 那会把真错误说成"结果差点但能用"。
        """
        if getattr(task, "exit_code", None) != 0:
            return
        verdict, note = self._offline_quality(session)
        if not verdict:
            return
        if verdict == "ok":
            self.status.set_status("重建完成（质量 ok，可用于数据结论）", "ok")
        else:
            self.status.set_status(f"重建完成{note}", "warn")

    def _offline_quality(self, session: str) -> tuple[str, str]:
        """读会话产物 ``traj3d.json`` 顶层 ``quality.verdict`` → ``(verdict, 人话)``。

        verdict → 人话**只有一份**实现（``apps/run_all._QUALITY_LABELS``）：这里把
        verdict 拼回它认的 ``[质量] verdict=...`` 标记行再交给
        ``run_all.quality_note``，不另写第三份词表。读不到/写坏返回 ``("", "")``，
        调用方据此保持沉默（不误报）。
        """
        try:
            if not str(session or "").strip():
                return "", ""          # 空路径会变成 cwd，别拿仓库根的旧文件凑结论
            base = Path(session)
            for rel in ("derived/traj3d.json", "traj3d.json"):
                p = base / rel
                if p.is_file():
                    break
            else:
                return "", ""
            data = json.loads(p.read_text(encoding="utf-8"))
            verdict = str(((data.get("quality") or {}).get("verdict"))
                          or "").strip()
            if not verdict:
                return "", ""
            # 延迟导入：只在任务结束时执行一次，不给窗口启动加依赖。
            from apps.run_all import quality_note
            return verdict, quality_note(f"[质量] verdict={verdict}")
        except Exception:                                    # noqa: BLE001
            return "", ""

    def _start_seed(self) -> None:
        session = self.var_session.get().strip()
        if not session:
            self.status.set_status("先选一个会话（点「选择…」）", "warn")
            return
        saved = pick_seeds(self, session, ball_kind=kind_key(self.ball.kind()))
        if saved is None:
            self.status.set_status("没有保存球心（窗口被关掉或左右没点完）", "warn")
            return
        self.status.set_status(
            f"球心已保存：{Path(saved).name}；现在跑第 ④ 步会用当前球种类的方法解算"
            "（镜面球 specular / 纯色球 color）。想同时用「半径锚点门」，"
            "把它勾上再跑（不勾就只是普通点选）", "ok")

    def _open_bench(self) -> None:
        """打开会话工作台。带上当前会话，用户接着做 ①–⑤ 而不是再选一遍。"""
        self.app.open_bench(self.var_session.get().strip() or None)

    def _open_viewer(self) -> None:
        """打开 3D 查看器 —— 必须带源：优先当前会话，其次最近一个真有结果的门录。

        view_3d 的 source 是必填位置参数，不给就 exit 2（表现为"点了没反应"）。
        """
        session = self.var_session.get().strip()
        if session:
            cand = Path(session)
            src = next((cand / rel for rel in ("derived/visualization.json",
                                               "visualization.json")
                        if (cand / rel).is_file()), None)
            if src is None:
                self.status.set_status(
                    "这个会话还没有 visualization.json：先在上面的"
                    "「跑离线三维」跑一次，再来查看", "warn")
                return
        else:
            src = self.app.recent_visualization()
            if src is None:
                self.status.set_status(
                    "先选一个会话并跑一次「跑离线三维」，才有三维结果可看", "warn")
                return
        self.app.tasks.start(
            _actions.view_3d(src),
            "3D 查看器", owner=self.task_owner)
        # 说**会话名**而不是 `src.parent.name`（那永远是 derived）
        self.status.set_status(
            f"正在打开 3D 查看器：{session_name_of(src)}"
            f"（窗口内「数据源」可换任意会话）", "info")

    def on_close(self) -> None:
        """关窗时记住输入（会话路径与相机角色不存：前者每次新选，后者从
        ``config/cameras.json`` 读）。"""
        # 先立旗标：关窗停掉的任务其 on_done 仍会到达，链式递归必须就此止步。
        self._closed = True
        p = self.app.prefs
        p.set("stereo.target", self.var_target.get().strip())
        p.set("board.cols", self.var_cols.get().strip())
        p.set("board.rows", self.var_rows.get().strip())
        p.set("board.square_mm", self.var_square.get().strip())
        p.set("stereo.stationary_px", self.var_stationary.get().strip())
        p.set("stereo.min_sharpness", self.var_sharpness.get().strip())
        p.set("stereo.baseline_mm", self.var_baseline.get().strip())
        # 两种新标定方式的输入一起记住（关窗写回，下次开窗直接带出）。
        p.set("calib.axial.root", self.var_axial_root.get().strip())
        p.set("calib.axial.assume_parallel", bool(self.var_axial_parallel.get()))
        p.set("calib.device90.intrinsics", self.var_d90_intrinsics.get().strip())
        p.set("calib.device90.rot_axis", self.var_d90_axis.get().strip() or "y")
        p.set("calib.device90.rot_angle", self.var_d90_angle.get().strip() or "90")
        p.set("calib.device90.z_a_mm", self.var_d90_za.get().strip())
        p.set("calib.device90.z_b_mm", self.var_d90_zb.get().strip())
        p.set("stereo.relaxed", bool(self.var_relaxed.get()))
        p.set("stereo.seconds", self.var_seconds.get().strip())
        p.set("stereo.cap_width", self.var_cap_width.get().strip())
        p.set("stereo.cap_height", self.var_cap_height.get().strip())
        p.set("stereo.cap_fps", self.var_cap_fps.get().strip())
        # 球种类与检测参数是**输入**：换了一颗红球的人不该每次开窗重填一遍。
        # （球种类本身不存：它由检测方法反推，存了就是同一件事的两份值。）
        p.set("stereo.preset", self.var_preset.get().strip() or PRESET_NONE)
        p.set("stereo.min_radius", self.var_min_radius.get().strip())
        p.set("stereo.max_radius", self.var_max_radius.get().strip())
        # 只在值真的变了才落盘：默认开时若也写一遍 `config/gui_state.json`，
        # 走查器会把"只开窗、配置就被改写"记成缺陷（2026-11-08 实测）。
        want = "1" if self.var_radius_auto.get() else "0"
        if p.get_str("stereo.radius_auto", "1") != want:
            p.set("stereo.radius_auto", want)
        p.save()

    def _stop_my_tasks(self) -> None:
        # 按归属停，不再全局 stop_all：本窗的"停止"按钮不该杀掉别的窗口
        # 在后台跑的离线重建/标定。task_owner 由 open_window 打标，
        # 平行/大角度两窗各有各的 owner，互不误停。
        n = self.app.tasks.stop_by_owner(self.task_owner)
        self.status.set_status(
            f"已停掉本窗口的 {n} 个任务" if n else "本窗口没有在跑的任务",
            "ok" if n else "warn")
