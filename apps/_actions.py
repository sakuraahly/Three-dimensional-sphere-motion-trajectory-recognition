"""界面动作 → 命令行：**唯一产地**。

为什么需要这个模块（不是"代码整洁"，是缺陷率）：

同一个动作以前在两套界面里各拼一遍 argv，于是反复出同一类事 ——
经典界面传 `--preview`、新界面没传 → "录制时没有实时画面"，而且**不报错**；
单目界面按 mm 收、命令行按 m 收 → 深度差 1000 倍；
「检查装置」曾经一个参数都没传 → 必然失败但静默。
三次都是「同一个动作有两份参数清单」。

所以这道约束不是"记得传参"，而是结构性的：**任何界面文件里都不许出现
`"xxx.py"` 字面量**（`selftest_gui_budget` 用 AST 扫，出现即红）。
新加一个按钮 = 从下面挑一个动作 + 填参数，而不是再抄一遍参数名。

约定（三条，都是有意的）：

1. 每个函数返回**完整 argv**（含解释器），调用方直接交给 ``TaskRunner`` / ``Popen``；
2. 参数名与 CLI 的 ``--xxx`` 对齐；与 CLI 默认值相同的项**不在这里重复写**，
   少一处默认值就少一处漂移；
3. 只依赖标准库与本项目的 ``paths``（不 import 任何界面模块），
   避免把依赖方向拉反。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_APPS = ROOT / "apps"
_TOOLS = _APPS / "tools"
CHAIN_ROOT = ROOT / "outputs" / "chain_frames"


def _argv(script: Path, *args: str) -> list[str]:
    """解释器 + 脚本 + 参数。所有动作都从这里出去，写法只有一种。"""
    return [sys.executable, str(script), *args]


def _flag(name: str, value) -> list[str]:
    """值非空才追加该参数；空白视同"这次不填"（与界面上清空输入框一致）。"""
    text = str(value or "").strip()
    return [name, text] if text else []


# --------------------------------------------------------------------------- #
# 双目：标定 / 录制 / 分析（原在 `_gui_stereo.py`，2026-11-01 搬到这里）
# --------------------------------------------------------------------------- #

BOARD_DEFAULTS = {"cols": "8", "rows": "5", "square": "25", "target": "40"}
_BOARD_FLAGS = (("cols", "--cols"), ("rows", "--rows"),
                ("square", "--square"), ("target", "--target"))


def board_args(board: dict | None) -> list[str]:
    """棋盘参数四项（内角点列/行、方格边长、目标对数）。

    留空 = 用脚本自己的默认档（8×5 / 25mm / 40 对），**不是**非法输入：
    这几个框在界面上本来就带着默认值，用户清空只是"这一次用默认"。
    真正的坑在另一头 —— "留空也传一个空串"：`--cols ""` 会在标定程序里
    被 `int()` 直接咬死，而报错的地方离用户的操作十万八千里。
    """
    out: list[str] = []
    for key, flag in _BOARD_FLAGS:
        raw = str((board or {}).get(key) or "").strip()
        out += [flag, raw or BOARD_DEFAULTS[key]]
    return out
def chain_dirs(root: Path | None = None) -> dict[str, Path]:
    """链式标定的四个目录（左右各一组 + 参考段的双路同拍图）。"""
    base = root if root is not None else CHAIN_ROOT
    return {"left": base / "left", "right": base / "right",
            "ref_left": base / "ref_left", "ref_right": base / "ref_right"}
def calibrate_args(method: str, board: dict | None = None,
                   left: str = "left", right: str = "right",
                   mode: str = "parallel", *,
                   stationary: str = "", sharpness: str = "",
                   relaxed: bool = False, backend: str = "",
                   dirs: dict | None = None) -> list[str]:
    """双目标定的完整 argv。

    三条分支的差别**只在参数**，不在文案：

    - ``chain``：``--method chain --same-board --left-dir/--right-dir``；
      参考段真有图时才加 ``--pair-frame``（指向不存在的目录等于给后端一个
      必然失败的证据，不如不提）。
    - ``joint_skip_fov``：仍是共同求解，但**模式写死 angled** —— 按重叠视场
      裁剪正是大角度装置要跳过的那一步。
    - ``joint``：带**本窗口**的模式（平行窗 parallel / 大角度窗 angled）。

    ``--skip-fov`` 不属于这里任何一条：它只属于旧界面那个"90° 装置"专用入口。
    新窗口的等效做法是选 `joint_skip_fov`（它走 ``--stereo-mode angled``，
    后端据此不再做 FOV 对齐），把这个 flag 混进普通 joint 会同时改变裁剪与
    内参口径，属于静默改行为。

    ``backend`` 只在调用方给出**具体**后端（dshow/msmf）时才下传：
    ``auto`` 是脚本自己的默认值，替用户写死一遍等于把"没选"伪装成"选了"。
    **两台相机共用这一个参数** —— 出现两次意味着有人给左右各拼了一份，
    那会拼出"左 dshow / 右 msmf"这种直接把一路弄挂的组合。
    """
    cmd = [sys.executable, str(ROOT / "apps" / "calibrate_stereo.py")]
    if method == "chain":
        d = dirs if dirs is not None else chain_dirs()
        cmd += ["--method", "chain", "--same-board",
                "--left-dir", str(d["left"]),
                "--right-dir", str(d["right"])] + board_args(board)
        # 参考段的双路同拍图作为"同一块板"的证据
        if Path(d["ref_left"]).exists():
            cmd += ["--pair-frame", str(d["ref_left"]), "--pair-camera", "left"]
    else:
        cmd += ["--method", "joint"] + board_args(board)
        if method == "joint_skip_fov":
            cmd += ["--stereo-mode", "angled"]
        else:
            cmd += ["--stereo-mode", mode]
        cmd += ["--left", str(left or "").strip() or "left",
                "--right", str(right or "").strip() or "right"]
    # 高级项：只有用户真的填了才下传（空 = 让后端用自己的默认值）
    if str(stationary or "").strip():
        cmd += ["--stationary-px", str(stationary).strip()]
    if str(sharpness or "").strip():
        cmd += ["--min-sharpness", str(sharpness).strip()]
    if relaxed:
        cmd += ["--relaxed"]
    if str(backend or "").strip() and str(backend).strip() != "auto":
        cmd += ["--backend", str(backend).strip()]

    # `--min-pairs` **必须传**，否则"试标定档"必然失败（2026-10-29 查出的功能缺口）。
    #
    # 链条：`calibrate_stereo` 的 `--min-pairs` 默认 20；而收图一旦达到
    # 「目标对数」就停（`calib/capture.py:274` `full = count >= cfg.target_pairs`）。
    # 于是把目标改成 8 的人会看到：收 8 对 → 停 → 8 < 20 → `:278 enough` 判假
    # → 标定失败。他只想快速验证通路，却被判"对数不足"。
    #
    # 旧界面把这个值放在「高级」里可填；新窗口没有这个框，所以在这里兜住：
    # 取 `min(20, target)` —— target ≥ 20 时与脚本默认**完全一致**（不改变语义），
    # 只有试标定档才下调到 target 本身（够 target 就能过）。
    _target = str((board or {}).get("target") or "").strip() or BOARD_DEFAULTS["target"]
    try:
        _pairs = int(_target)
    except ValueError:
        _pairs = 0
    if _pairs > 0:
        cmd += ["--min-pairs", str(min(20, _pairs))]
    return cmd
def offline_stereo(session: str | Path, *, mode: str = "",
                   anchor: bool = False) -> list[str]:
    """离线重建（`run_stereo_video.py`）的 argv。

    抽成纯函数是为了**能被判据抓到**：界面上的勾选框到命令行之间断链，
    在本项目已经发生过不止一次（"检查装置"没传参、球径按 mm 传给了按米收的参数）。
    有了它，`selftest_gui_budget` 可以直接断言"勾了就必须出现 --anchor-gate on"，
    而不是靠人去点一遍。

    ``--stereo-mode`` **只出现一次**：这个窗口存在的意义就是"按本窗口的装置解算"，
    重复传会让 argparse 取最后一个，读起来却像有两处来源。
    """
    # mode 为空 = **不传** `--stereo-mode`：会话工作台那边要让脚本自己读
    # session/标定（老会话的 `stereo_mode` 常被折成 parallel，照它传会按错几何解算）。
    cmd = [sys.executable, str(ROOT / "apps" / "run_stereo_video.py"), str(session)]
    cmd += _flag("--stereo-mode", mode)
    if anchor:
        cmd += ["--anchor-gate", "on"]
    return cmd


def offline_args(session: str, mode: str, *, anchor: bool = False) -> list[str]:
    """``offline_stereo`` 的旧名字（始终显式带 ``--stereo-mode``）。

    保留它是因为判据与界面都在用这个名字，且它们的语义**必须**是"按本窗口
    强制解算"。而会话工作台那边**故意不传** mode（让脚本读 session/标定自己判断，
    否则老会话会被按错几何解算）—— 同一份实现，两种用法，靠 ``mode`` 是否为空区分。
    """
    return offline_stereo(session, mode=mode, anchor=anchor)
def record_args(*, left: str = "left", right: str = "right", duration: str = "10",
                out: str, mode: str | None = None,
                label: str = "") -> list[str]:
    """双目录制的 argv。

    **故意不传 ``--stereo-mode``**（B-107 的教训）：录制要按标定走 ``auto``，
    模式由 `stereo.record_plan.plan_record` 从标定里读。传平行/大角度都可能
    与当前标定不符 —— 后果是相机先白开、会话目录先建好，然后在检查里才报
    "模式冲突"，白等一轮；传 ``auto`` 又只是把脚本默认值写死一遍。
    所以 ``mode`` 只在调用方给出**非 auto** 的显式值时才下传，其余一律不提。

    ``--label`` 是**界面上的「标签」框**的唯一去处（2026-10-31 修）：
    它写进 ``session.json`` 的 ``label`` 字段，会话列表「标签」列读的就是它。
    上一版只拿标签去拼目录名（还拼错了根），于是录出来的每一段在列表里
    都显示"（无标签）"—— 用户分不清哪个是哪次实验。

    分辨率/帧率/后端也**不传**：录制程序会按标定校验尺寸（`validate_capture_size`），
    界面再写一份数字只会多一处漂移。要改这些，先给它们加控件 —— 那时再把
    对应的关键字参数接上。

    ``--preview --wait-key space`` 是**必传**的（2026-11-01 修）：
    老界面一直传 ``--preview``，而新界面从重做那天起就没传过 —— 于是"录制时
    看不到实时画面"成了新界面的独有现象，且**不报错**（`--preview` 是
    `store_true`，不传就是没有窗口）。用户报"标定完实时画面没了"，根因就是这一行。
    同时把起录方式改成"先出画面，按空格才开始倒计时"：程序刚起来时相机常常
    还没出第一帧，倒计时一共才 3 秒，等于把准备时间从人手里抢走。
    """
    cmd = [sys.executable, str(ROOT / "apps" / "record_stereo.py"),
           "--left", str(left or "").strip() or "left",
           "--right", str(right or "").strip() or "right",
           "--duration", str(duration or "").strip() or "10",
           "--out", str(out),
           "--preview", "--wait-key", "space"]
    # 空白标签不传：`--label ""` 与不传是同一件事，传空串只会让命令行更长
    if str(label or "").strip():
        cmd += ["--label", str(label).strip()]
    if mode and str(mode).strip() and str(mode).strip() != "auto":
        cmd += ["--stereo-mode", str(mode).strip()]
    return cmd


# --------------------------------------------------------------------------- #
# 打开类动作（不起任务、只看/管理）：相机设备、数据管理、3D 查看器、经典界面
# --------------------------------------------------------------------------- #

def cameras(sub: str = "identify") -> list[str]:
    """相机设备管理。``identify`` 是交互式的（自己开 cv2 窗口逐台确认）。

    为什么复用它而不是在新界面重画一个：它已经处理了"两台各自只在不同后端下
    出图"这种真机状态（``probe`` 逐后端×逐编号实测），重画一遍只会分叉。
    """
    return _argv(_TOOLS / "cameras.py", sub)


def data_manager() -> list[str]:
    return _argv(_APPS / "data_manager.py")


def view_3d(source: str | Path) -> list[str]:
    """3D 查看器。``source`` 是**必填**（view_3d 的第一个位置参数）。"""
    return _argv(_APPS / "view_3d.py", str(source))


def legacy_gui() -> list[str]:
    """经典界面（过渡期入口）。"""
    return _argv(ROOT / "启动程序.py", "--legacy")


# --------------------------------------------------------------------------- #
# 会话工作台：校验 / 导出
# --------------------------------------------------------------------------- #

def check_dataset(session: str | Path) -> list[str]:
    return _argv(_TOOLS / "check_dataset.py", str(session), "--strict")


def export_visualization(session: str | Path, *, plot: bool = True) -> list[str]:
    return _argv(_APPS / "export_visualization.py", str(session),
                 *(["--plot"] if plot else []))


# --------------------------------------------------------------------------- #
# 单目：录制 / 离线识别 / 实时识别 / 拍标定板
# --------------------------------------------------------------------------- #

CALIBRATE_MONO = _APPS / "calibrate_mono.py"
CALIBRATE_FOCAL = _TOOLS / "calibrate_focal.py"


def calibrate_mono(*, frames_dir: str | Path, cols: str, rows: str,
                   square: str) -> list[str]:
    """单目内参标定（棋盘图 → config/mono_calibration.json）。"""
    return _argv(CALIBRATE_MONO, "--dir", str(frames_dir),
                 "--cols", str(cols), "--rows", str(rows), "--square", str(square))


def calibrate_focal(*, diameter_m: str, extra: list[str] | None = None) -> list[str]:
    """单点焦距估计（把球放在已知距离点一下）。

    ``diameter_m`` 是**米**，与 ``run_video`` 同一个口径：这个数直接决定解出来的
    fx（``f = 2·r_px·Z/D``），按 mm 传会差 1000 倍。
    """
    return _argv(CALIBRATE_FOCAL, "--ball-diameter", str(diameter_m),
                 *(extra or []))


def record_mono(camera: str, duration: str, out: str | Path) -> list[str]:
    return _argv(_APPS / "record_mono.py", "--camera", str(camera),
                 "--duration", str(duration), "--out", str(out))


def run_video(clip: str | Path, *, diameter_m: str, method: str,
              out: str | Path | None = None, csv: str | Path | None = None,
              vis_json: str | Path | None = None) -> list[str]:
    """离线单目。

    ``diameter_m`` 一律是**米**（界面显示 mm，换算在界面那一层做完再传进来）——
    这条口径以前断过：界面按 mm 传 21.6，命令行按米收，结果一个三维点都不出。
    """
    args = [str(clip), "--ball-diameter", str(diameter_m), "--method", str(method)]
    args += _flag("--out", out)
    args += _flag("--csv", csv)
    args += _flag("--vis-json", vis_json)
    return _argv(_APPS / "run_video.py", *args)


def run_realtime(*, diameter_m: str, method: str) -> list[str]:
    return _argv(_APPS / "run_realtime.py",
                 "--ball-diameter", str(diameter_m), "--method", str(method))


def capture_board(*, out: str | Path, cols: str, rows: str, square: str,
                  target: str, extra: list[str] | None = None) -> list[str]:
    args = [str(out), "--cols", str(cols), "--rows", str(rows),
            "--square", str(square), "--target", str(target), "--auto"]
    return _argv(_TOOLS / "capture_board.py", *args, *(extra or []))

def check_setup_args(*, left: str = "left", right: str = "right",
                     board: dict | None = None) -> list[str]:
    """「检查装置」的 argv（2026-10-31 修，缺陷 #4）。

    为什么必须带参数：``check_stereo_setup.py`` 的两路画面来自
    ``apps._common.add_stereo_source_args``，**不传 ``--left/--right`` 就等于
    让它自己去猜**；它的 docstring 第一行给的就是 ``--left 0 --right 2`` 这种
    显式用法。上一版一个参数都没传，检查程序只能按 ``auto`` 自己挑相机 ——
    与第 ② 步标定用的两台不是同一对时，用户会按一个**别的装置**的结论去调
    相机，而且界面还不回写结果（状态条永远停在"正在检查装置…"）。

    相机取值可以是角色名（``left``/``right``，查 ``config/cameras.json``）
    或索引，与 ``calibrate_stereo`` 同一套解析 —— 界面里那两个框就原样下传。
    """
    cmd = [sys.executable, str(ROOT / "apps" / "tools" / "check_stereo_setup.py"),
           "--left", str(left or "").strip() or "left",
           "--right", str(right or "").strip() or "right"]
    b = board or {}
    # 棋盘参数只下传用户真填了的：留空 = 脚本默认 8×5 / 25mm（同 `board_args` 的口径）
    for key, flag in (("cols", "--cols"), ("rows", "--rows"), ("square", "--square")):
        raw = str(b.get(key) or "").strip()
        if raw:
            cmd += [flag, raw]
    return cmd
