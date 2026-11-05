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


def _positive(name: str, value) -> list[str]:
    """正数才追加该参数；空 / 非数字 / ≤0 **一律不传**。

    为什么不在界面里"取个大概"再传：``--min-radius`` 是像素门限，传 0 或负数
    等于把所有候选都放行（或全拒），而界面上那一栏看起来只是"填了个数"。
    界面负责说清"这一项没被采用"，这里负责绝不把它混进命令行。
    """
    text = str(value or "").strip()
    if not text:
        return []
    try:
        num = float(text)
    except ValueError:
        return []
    return [name, text] if num > 0 else []


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
    """链式标定的目录（单板：left/right + ref 参考段；双板：left/right + pair 同框图）。

    单板模式（`--same-board`）用 left/right/ref_left/ref_right；
    双板模式（两块刚性固定的板 A/B，各看一块）用 left/right/board_b/pair。
    两种模式共用 left/right（内参段，板可移动），区别在参考段：
    单板是"同一块板不动"（ref_left/ref_right），双板是"同框拍两板"（pair）。
    """
    base = root if root is not None else CHAIN_ROOT
    return {"left": base / "left", "right": base / "right",
            "ref_left": base / "ref_left", "ref_right": base / "ref_right",
            "board_b": base / "board_b", "pair": base / "pair"}
def calibrate_args(method: str, board: dict | None = None,
                   left: str = "left", right: str = "right",
                   mode: str = "parallel", *,
                   stationary: str = "", sharpness: str = "",
                   relaxed: bool = False, backend: str = "",
                   dirs: dict | None = None,
                   out: str = "", baseline: float | str | None = None,
                   fov_from_profile: str = "", use_fov: bool = True) -> list[str]:
    """双目标定的完整 argv。

    三条分支的差别**只在参数**，不在文案：

    - ``chain``：``--method chain --same-board --left-dir/--right-dir``；
      参考段真有图时才加 ``--pair-frame``（指向不存在的目录等于给后端一个
      必然失败的证据，不如不提）。
    - ``chain_dual``：双板模式（两块刚性固定的板），``--board-b-dir`` +
      ``--pair-frame`` 求 ``T_AB``，**不加** ``--same-board``。
    - ``joint``：带**本窗口**的模式（平行窗 parallel / 大角度窗 angled）。

    ⚠️ 历史上还有一个 ``joint_skip_fov`` 键：它把 ``--stereo-mode`` 写死
    ``angled``。但大角度窗的 ``mode`` 本来就是 angled，两条命令**完全相同**；
    而平行窗没有方法下拉（固定 joint）。也就是说它在任何窗口都是空操作，
    只会让人误以为「跳过裁剪」和「不跳过」有区别。**已从界面下拉移除**
    （分支保留给旧调用方，见 `_gui_stereo.METHODS` 的说明）。

    ``use_fov`` 是"这次标定要不要做视场对齐"（2026-11-04 用户："FOV 有点影响
    平行双目标定了"）。``False`` 时对**会做裁剪**的方法传 ``--skip-fov``：
    两路按原始画面标定，不裁像素、不重采样。默认 ``True`` = 老行为，
    旧调用方与判据不受影响。

    ``--skip-fov`` 不在 chain/chain_dual 下传：它们产出 none_raw，后端
    ``do_fov = not skip_fov and (mode != "angled" or --fov)`` 本来就为 False
    （链式是 angled），再传只是噪音。

    ``baseline`` 是**物理核对**（`--baseline-mm`）：传了它，后端才会检查
    "解出的基线 vs 卷尺量的是不是差太远"——这是 chain 双板模式下**唯一**的
    物理一致性证据（双板左看 A、右看 B，做不了跨路重合检查）。装置档案里
    存着 ``expected_baseline_mm``，GUI 必须把它传下来，否则这条护栏形同虚设。

    ``fov_from_profile`` 是"复用装置档案里已量好的视场裁剪参数"：不传就由
    后端从启动帧现场估（旧行为）。只对**做裁剪的方法**下传——chain/chain_dual
    产出 none_raw、不做 FOV 对齐（90° 无共视，单应裁剪没有物理意义），
    传了后端会报互斥错误。

    ``backend`` 只在调用方给出**具体**后端（dshow/msmf）时才下传：
    ``auto`` 是脚本自己的默认值，替用户写死一遍等于把"没选"伪装成"选了"。
    **两台相机共用这一个参数** —— 出现两次意味着有人给左右各拼了一份，
    那会拼出"左 dshow / 右 msmf"这种直接把一路弄挂的组合。
    """
    cmd = [sys.executable, str(ROOT / "apps" / "calibrate_stereo.py")]
    if method in ("chain", "chain_dual"):
        d = dirs if dirs is not None else chain_dirs()
        if method == "chain_dual":
            # 双板模式：两块刚性固定的板 A/B（如"垂直双板"装置），左看 A、右看 B，
            # 用一张「同框拍到 A、B 两板」的图求 T_AB。**不加 --same-board**，
            # 用 --board-b-dir + --pair-frame + --pair-camera（单板模式才用
            # --ref-left/--ref-right，两者互斥）。
            cmd += ["--method", "chain",
                    "--left-dir", str(d["left"]),
                    "--board-b-dir", str(d["board_b"])] + board_args(board)
            if Path(d["pair"]).exists():
                # 同框图是哪一路相机拍的（默认左），后端据此区分 A/B 板。
                cmd += ["--pair-frame", str(d["pair"]), "--pair-camera", "left"]
        else:
            cmd += ["--method", "chain", "--same-board",
                    "--left-dir", str(d["left"]),
                    "--right-dir", str(d["right"])] + board_args(board)
            # 单板模式（--same-board）的参考段是 **--ref-left / --ref-right**，
            # 不是 --pair-frame（那是「双板」模式 --board-b-dir 才用的参数，
            # calibrate_stereo.py 的 _chain_input_error 明确说单板模式不许给 pair-frame）。
            # 这里曾误传 --pair-frame —— 后端直接把它当双板模式处理，参考段参数对不上，
            # 90° 单板链式标定于是跑不通。参考段真有图才传（指向不存在的目录等于给
            # 后端一个必然失败的证据，不如不提）。
            if Path(d["ref_left"]).exists():
                cmd += ["--ref-left", str(d["ref_left"])]
            if Path(d["ref_right"]).exists():
                cmd += ["--ref-right", str(d["ref_right"])]
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
    # `--out` 只在调用方给出**具体**路径时才传：空 = 让脚本用默认
    # ``config/stereo_calibration``。GUI 会传当前装置档案的 `calibration_path`，
    # 让标定写进"这套装置专属"的路径，而不是总覆盖默认文件（否则换 90° 装置
    # 重新标定就把平行装置的标定冲掉了）。
    if str(out or "").strip():
        cmd += ["--out", str(out).strip()]
    # `--baseline-mm` 同理：空/非正数不传（等于脚本默认「不核对」）。
    # 非数字要**忽略而不是拼进去**——拼一个 "abc" 会让 argparse 直接 exit 2，
    # 用户看到的是"参数错误"而不是"档案里那一栏填错了"。
    try:
        _base = float(str(baseline).strip()) if str(baseline or "").strip() else 0.0
    except ValueError:
        _base = 0.0
    if _base > 0:
        cmd += ["--baseline-mm", f"{_base:g}"]
    # FOV 三件事**在同一个地方**判，argv 只有这一个产地：
    #   ① 关掉 → `--skip-fov`（明确不做对齐，也不做现场估计）；
    #   ② 开着且档案里有名字 → `--fov-from-profile`（复用量好的参数）；
    #   ③ 开着但没名字 → 什么都不传（后端从启动帧现场估，老行为）。
    # `--fov-from-profile` 与 `--skip-fov` 互斥：关掉时绝不带上档案名，
    # 否则后端会直接 argparse 报错（1055 行那条校验）。
    if method not in ("chain", "chain_dual"):
        if not use_fov:
            cmd += ["--skip-fov"]
        elif str(fov_from_profile or "").strip():
            cmd += ["--fov-from-profile", str(fov_from_profile).strip()]
    return cmd
def offline_stereo(session: str | Path, *, mode: str = "",
                   anchor: bool = False, method: str = "", seed_json: str = "",
                   preset: str = "", min_radius: str = "",
                   max_radius: str = "") -> list[str]:
    """离线重建（`run_stereo_video.py`）的 argv。

    抽成纯函数是为了**能被判据抓到**：界面上的勾选框到命令行之间断链，
    在本项目已经发生过不止一次（"检查装置"没传参、球径按 mm 传给了按米收的参数）。
    有了它，`selftest_gui_budget` 可以直接断言"勾了就必须出现 --anchor-gate on"，
    而不是靠人去点一遍。

    ``--stereo-mode`` **只出现一次**：这个窗口存在的意义就是"按本窗口的装置解算"，
    重复传会让 argparse 取最后一个，读起来却像有两处来源。

    ``method`` / ``preset`` / 半径范围来自「球种类与检测参数」那一块控件
    （2026-11-03 增）：纯色球用 ``color`` + 颜色预设、镜面球用 ``specular``。
    空串 = 这项不传（脚本按注册表默认 color 走），**不在这里替用户写死** ——
    写死一处就等于界面上的下拉白摆。

    ``seed_json`` 单独一个参数而不是让调用方 ``cmd += ["--seed-json", …]``：
    「跑离线三维」原来在工作台里手工拼了 ``["--method", "specular",
    "--seed-json", …]`` 三行，于是"球种类"改不动它 —— 同一段命令行有两个产地。
    """
    # mode 为空 = **不传** `--stereo-mode`：会话工作台那边要让脚本自己读
    # session/标定（老会话的 `stereo_mode` 常被折成 parallel，照它传会按错几何解算）。
    cmd = [sys.executable, str(ROOT / "apps" / "run_stereo_video.py"), str(session)]
    cmd += _flag("--stereo-mode", mode)
    if anchor:
        cmd += ["--anchor-gate", "on"]
    cmd += _flag("--method", method)
    cmd += _flag("--preset", preset)
    cmd += _positive("--min-radius", min_radius)
    cmd += _positive("--max-radius", max_radius)
    cmd += _flag("--seed-json", seed_json)
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

    ``mode`` 现在**会传**（2026-11-04 改，之前是"故意不传"）
    ---------------------------------------------------
    以前不传，是担心"传平行/大角度可能与标定不符 → 相机先白开、会话目录先建好，
    然后在检查里才报模式冲突，白等一轮"。但 ``record_stereo.py`` 的体检
    （``plan_record``）已经**前移到开相机之前**：不符时只读一个 npz 就返回 2，
    不开相机、不建目录。所以那个理由不再成立。

    而"不传"有新的代价：``auto`` 会按**标定**解析模式，于是"在 90° 窗口用
    parallel 标定录"会成功录出一段标成 ``parallel`` 的会话 —— 元数据说平行、
    装置其实是 90°，事后谁看列表都分不出，隔离形同虚设（用户实测
    ``20261103-093820`` 就是这样）。

    现在传本窗的装置几何：标定与窗口不一致时**立刻失败**并给出"先按 90° 装置
    重标一次"的出路，而不是录一段几何错的会话。会话的 ``stereo_mode`` 因此
    恒等于录它的那个窗口，分区目录与元数据不会打架。

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

def has_seeds(session: str | Path) -> bool:
    """这个会话里有没有**人点过**的球心（`derived/seeds.json`）。

    界面拿它做前置检查：勾了「半径锚点门」却没点选过，命令行那边会拒绝并退出码 2，
    但那时用户已经点了按钮、等了一次进程启动 —— 能在界面上先说清楚就不要让人等。
    """
    p = Path(session) / "derived" / "seeds.json"
    return p.is_file()


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
              vis_json: str | Path | None = None, preset: str = "") -> list[str]:
    """离线单目。

    ``diameter_m`` 一律是**米**（界面显示 mm，换算在界面那一层做完再传进来）——
    这条口径以前断过：界面按 mm 传 21.6，命令行按米收，结果一个三维点都不出。

    ``preset`` 是**颜色预设**（``--preset``，值取自
    ``ColorConfig.preset_names()``）：只有颜色分割才看它，镜面球选的那一档
    在「球种类」里已经被折成空串（见 ``apps/_gui_widgets.BallSettings.resolve``），
    所以这里只管"有就传、没有不提"。
    """
    args = [str(clip), "--ball-diameter", str(diameter_m), "--method", str(method)]
    args += _flag("--preset", preset)
    args += _flag("--out", out)
    args += _flag("--csv", csv)
    args += _flag("--vis-json", vis_json)
    return _argv(_APPS / "run_video.py", *args)


def run_realtime(*, diameter_m: str, method: str, preset: str = "") -> list[str]:
    return _argv(_APPS / "run_realtime.py",
                 "--ball-diameter", str(diameter_m), "--method", str(method),
                 *_flag("--preset", preset))


def capture_board(*, out: str | Path, cols: str, rows: str, square: str,
                  target: str, extra: list[str] | None = None) -> list[str]:
    args = [str(out), "--cols", str(cols), "--rows", str(rows),
            "--square", str(square), "--target", str(target), "--auto"]
    return _argv(_TOOLS / "capture_board.py", *args, *(extra or []))


def capture_chain_frames(*, stage: str, cols: str, rows: str, square: str,
                         target: str, left: str = "left", right: str = "right",
                         min_cover: int = 0, relax_after: int = 0,
                         per_region: int = 0,
                         dirs: dict | None = None) -> list[str]:
    """采集链式标定的某一段图（`capture_board.py`），落进 `chain_dirs()` 对应目录。

    ``stage`` ∈ {``left``, ``right``, ``ref``, ``dual_b``}：
    - ``left``/``right``：单板模式的内参段，单路（板可移动，多姿态）；
    - ``ref``：单板模式的参考段，``--both`` 双路同拍（板**固定不动**，T_AB=I）；
    - ``dual_b``：**双板模式**里右相机拍 B 板（落 `board_b` 目录），与 ``left``
      （拍 A 板）配对。同框图（pair）不能走 `capture_board.py`（它只检单板），
      需单独采集，所以这里没有 ``pair`` 段。

    参考段**不传** ``--camera-left/--camera-right``：与旧界面一致，让
    `capture_board.py` 自己按 left/right 角色取相机（角色绑定更稳，索引会漂移）。

    ``min_cover`` / ``relax_after`` 只对内参段有效（双板 A/B 与单板左右）：
    前者是**首选目标**（5=五个角全绿，最理想），后者是**分化线**——拍够这么多张
    后覆盖标准放宽到 4/5。参考段不用覆盖判据（它的规则是"板不能动"）。
    """
    d = dirs if dirs is not None else chain_dirs()
    if stage == "ref":
        return _argv(
            _TOOLS / "capture_board.py", "--both",
            "--out-left", str(d["ref_left"]),
            "--out-right", str(d["ref_right"]),
            "--cols", str(cols), "--rows", str(rows), "--square", str(square),
            "--target", str(target), "--auto")
    if stage == "dual_b":
        # 双板模式：右相机拍 B 板，落 board_b 目录。
        return _argv(
            _TOOLS / "capture_board.py", "--out", str(d["board_b"]),
            "--side", "right",
            "--camera", str(right or "").strip() or "right",
            "--cols", str(cols), "--rows", str(rows), "--square", str(square),
            "--target", str(target), "--auto",
            *(["--min-cover", str(min_cover)] if min_cover > 0 else []),
            *(["--relax-after", str(relax_after)] if relax_after > 0 else []),
            *(["--per-region", str(per_region)] if per_region > 0 else []))
    side = "left" if stage == "left" else "right"
    out = d["left"] if stage == "left" else d["right"]
    return _argv(
        _TOOLS / "capture_board.py", "--out", str(out), "--side", side,
        "--camera", str(left if stage == "left" else right or "").strip() or side,
        "--cols", str(cols), "--rows", str(rows), "--square", str(square),
        "--target", str(target), "--auto",
        *(["--min-cover", str(min_cover)] if min_cover > 0 else []),
        *(["--relax-after", str(relax_after)] if relax_after > 0 else []),
        *(["--per-region", str(per_region)] if per_region > 0 else []))


def capture_pair_frame(*, cols: str, rows: str, square: str,
                       camera: str = "left",
                       dirs: dict | None = None) -> list[str]:
    """拍双板链式的**同框图**（pair）：单相机、不做单板检测、按空格存。

    双板模式求 T_AB 需要一张"两块板同时入画"的图，`capture_board.py` 的单板
    检测器检不了它，所以走 ``--pair`` 模式（2026-11-04 前这块只能让用户手动拍）。
    """
    d = dirs if dirs is not None else chain_dirs()
    return _argv(
        _TOOLS / "capture_board.py", "--out", str(d["pair"]),
        "--camera", str(camera or "left").strip() or "left",
        "--cols", str(cols), "--rows", str(rows), "--square", str(square),
        # 5 张同框图（2026-11-05 用户：单张有风险）。T_AB 只用其中一张，
        # 但多拍几张能在"哪张两板都检全"上留选择余地，也防单张糊掉。
        "--target", "5", "--pair")


def check_chain_inputs(*, cols: str, rows: str, square: str,
                       dirs: dict | None = None,
                       mode: str = "single") -> list[str]:
    """链式标定输入自检（`check_chain_inputs.py`），拍完先跑这个。

    ``mode`` 决定查哪几段（与 `calibrate_args` 的链式分支一一对应）：
    ``single`` = 单板四段（left/right/ref_left/ref_right）；
    ``dual``   = 双板三段（left/board_b/pair）。
    以前双板只能弹一句"没有自动自检"的提示——那等于让用户拿标定去试错，
    而一次双板拍摄是十几分钟。
    """
    d = dirs if dirs is not None else chain_dirs()
    return _argv(
        _TOOLS / "check_chain_inputs.py", "--root", str(Path(d["left"]).parent),
        "--cols", str(cols), "--rows", str(rows), "--square", str(square),
        "--mode", str(mode or "single"))

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
