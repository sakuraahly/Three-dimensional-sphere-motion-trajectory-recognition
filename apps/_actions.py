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

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_APPS = ROOT / "apps"
_TOOLS = _APPS / "tools"
CHAIN_ROOT = ROOT / "outputs" / "chain_frames"
MONO_ROOT = ROOT / "outputs" / "mono_calib_frames"
#: 「当前标定会话」指针。放在根层、每次新建会话就改写，**绝不回落读根层图片**。
CURRENT_SESSION = "_current.json"
SESSION_SCHEMA = "calib-session/1"
POINTER_SCHEMA = "calib-session-pointer/1"


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

def setup_env_adopt(python: str) -> list[str]:
    """登记/切换运行解释器；校验与写指针都在 setup_env.py 内完成。"""
    return _argv(_APPS / "setup_env.py", "--adopt", str(python))


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
    """链式标定的目录。

    单板模式（`--same-board`）用 left/right（内参）+ ref_left/ref_right（参考段）；
    双板模式用 left/board_b（内参）+ **ref_pair_left/ref_pair_right（参考对）**
    + pair（同框图）。

    参考对是 2026-11-09 方案 A 加的：链上两端的板位姿必须来自**同一个支架
    摆放**，所以双板模式也要一组「组合装置摆好别动、两路同时各拍 3 张」的图。
    它由 `capture_board.py --both` 产出，两目录里**同名文件 = 同一时刻**，
    天然成对，不需要额外的配对清单。
    """
    base = root if root is not None else CHAIN_ROOT
    return {"left": base / "left", "right": base / "right",
            "ref_left": base / "ref_left", "ref_right": base / "ref_right",
            "board_b": base / "board_b", "pair": base / "pair",
            "ref_pair_left": base / "ref_pair_left",
            "ref_pair_right": base / "ref_pair_right"}


# --------------------------------------------------------------------------- #
# 标定会话：**一次采集 = 一个子目录**（2026-11-09）
# --------------------------------------------------------------------------- #
# 以前所有采集都追加到 ``outputs/chain_frames/left`` 这类固定目录，标定时
# ``_list_images(目录)`` 全读 —— 真机实测 board_b 344 张里混了 11-04 的 304 张
# 和 11-08 的 40 张，相机/分辨率还不一样；边车只记最后一批，count 与实际张数
# 差一个量级。用户明确说「相机频繁移动，这些都是负面资产」，所以：
#   * 每次点「采集标定板图…」/「拍单目标定图」新建一个带时间戳的会话目录；
#   * 四段图、自检、标定全部指向**同一个**会话（靠根层 _current.json 指针，
#     因为采集/自检/标定是不同子进程，不能只靠窗口里的临时变量）；
#   * 没有当前会话时明确报错，**绝不回落到根层**把历史图吃进去。
# --------------------------------------------------------------------------- #
def _now() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _new_session_dir(root: Path, kind: str, meta: dict) -> Path:
    from datetime import datetime

    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = root / stamp
    n = 2
    while path.exists():                       # 同一秒重复点：不覆盖、不混
        path = root / f"{stamp}-{n}"
        n += 1
    path.mkdir(parents=True)
    # 拍图时的板规格也写一份**结构化**的（2026-11-19）：`board` 里的值是
    # StringVar 转来的字符串（"8"/"25"），标定侧核对要按数值比、且要能追溯
    # 方格边长。这里统一派生 board_spec（int/int/float），老会话没有这个键
    # 时标定侧回退读 board 字符串，互不破坏。
    meta = dict(meta)
    board = meta.get("board")
    if isinstance(board, dict):
        try:
            sq_key = "square" if "square" in board else "square_mm"
            meta["board_spec"] = {
                "cols": int(str(board["cols"]).strip()),
                "rows": int(str(board["rows"]).strip()),
                "square_mm": float(str(board[sq_key]).strip()),
            }
        except (KeyError, TypeError, ValueError):
            pass                       # 板参数没填全 -> 不写，标定侧按"没记录"处理
    (path / "session.json").write_text(
        json.dumps({"schema": SESSION_SCHEMA, "kind": kind,
                    "session": path.name, "created_at": _now(), **meta},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / CURRENT_SESSION).write_text(
        json.dumps({"schema": POINTER_SCHEMA, "kind": kind,
                    "session": path.name, "created_at": _now()},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _current_session_dir(root: Path, kind: str) -> Path | None:
    """读 ``_current.json`` 指向的会话；指针缺失/会话不在 → **None**。

    返回 None 时调用方必须明说「还没有标定会话，先采集」，不许自己拿根层
    目录顶上 —— 那正是这次要消灭的混批来源。
    """
    try:
        raw = json.loads((root / CURRENT_SESSION).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if str(raw.get("kind") or "") != kind:
        return None
    name = str(raw.get("session") or "")
    if not name or name.startswith(".") or "/" in name or "\\" in name:
        return None                              # 防指针被写成根层或别的路径
    path = root / name
    return path if path.is_dir() else None


def new_chain_session(*, profile: str = "", mode: str = "",
                      board: dict | None = None, capture: dict | None = None,
                      root: Path | None = None) -> dict[str, Path]:
    """新建一次链式标定会话，返回该会话的 ``chain_dirs()``。

    ``root`` 只给自测用（默认 ``CHAIN_ROOT``），让判据能在临时目录里跑，
    不往真实 ``outputs/`` 里留空会话。
    """
    p = _new_session_dir(root or CHAIN_ROOT, "chain", {
        "profile": profile or "", "stereo_mode": mode or "",
        "board": board or {}, "capture": capture or {}})
    return chain_dirs(p)


def current_chain_dirs(root: Path | None = None) -> dict[str, Path] | None:
    """当前链式标定会话的目录；没有会话返回 None（不回落根层）。"""
    p = _current_session_dir(root or CHAIN_ROOT, "chain")
    return chain_dirs(p) if p is not None else None


def new_mono_session(*, camera: str = "", board: dict | None = None,
                     root: Path | None = None) -> Path:
    """新建一次单目标定会话，返回图片目录（图片直接落在该目录下）。"""
    return _new_session_dir(root or MONO_ROOT, "mono", {
        "camera": camera or "", "board": board or {}})


def current_mono_dir(root: Path | None = None) -> Path | None:
    """当前单目标定会话的目录；没有会话返回 None。"""
    return _current_session_dir(root or MONO_ROOT, "mono")


def calibrate_args(method: str, board: dict | None = None,
                   left: str = "left", right: str = "right",
                   mode: str = "parallel", *,
                   stationary: str = "", sharpness: str = "",
                   relaxed: bool = False, backend: str = "",
                   dirs: dict | None = None,
                   out: str = "", baseline: float | str | None = None,
                   fov_from_profile: str = "", use_fov: bool = True,
                   force_save: bool = False,
                   width: str = "", height: str = "", fps: str = "") -> list[str]:
    """双目标定的完整 argv。

    三条分支的差别**只在参数**，不在文案：

    - ``chain``：``--method chain --same-board --left-dir/--right-dir``；
      参考段真有图时才加 ``--pair-frame``（指向不存在的目录等于给后端一个
      必然失败的证据，不如不提）。
    - ``chain_dual``：双板模式（两块刚性固定的板），``--board-b-dir`` +
      ``--pair-frame`` 求 ``T_AB``，``--ref-pair-left/--ref-pair-right`` 给链上
      两端的**同一摆放**位姿，**不加** ``--same-board``。
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

    ``baseline`` 是**这一次标定的物理核对**（`--baseline-mm`）：传了它，后端才会
    检查"解出的基线 vs 卷尺量的是不是差太远"——这是 chain 双板模式下**唯一**的
    物理一致性证据（双板左看 A、右看 B，做不了跨路重合检查）。
    2026-11-07 起装置档案不再存 ``expected_baseline_mm``，GUI 也不自动下传；
    需要核对时由人用 CLI 显式给（它是单次测量，不是装置属性）。

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
        # 调用方可以只给部分目录（selftest_gui_budget 就传 left/right/
        # board_b/pair 四个）。新增的 ref_pair_* 是必需参数，不能因为字典
        # 少一个键就把旧调用方弄崩 —— 用 chain_dirs() 补齐缺省键。
        d = {**chain_dirs(Path(d["left"]).parent), **d}
        if method == "chain_dual":
            # 双板模式：两块刚性固定的板 A/B（如"垂直双板"装置），左看 A、右看 B，
            # 用一张「同框拍到 A、B 两板」的图求 T_AB。**不加 --same-board**，
            # 用 --board-b-dir + --pair-frame + --pair-camera（单板模式才用
            # --ref-left/--ref-right，两者互斥）。
            cmd += ["--method", "chain",
                    "--left-dir", str(d["left"]),
                    "--board-b-dir", str(d["board_b"])] + board_args(board)
            # 参考对是**必需**输入（2026-11-09 方案 A）。无条件下传：目录没拍
            # 图时后端会明确报「第 ③ 步没拍」，比静默不传、让人拿着一份错
            # 外参去跑重建强。链两端位姿只能来自同一摆放的这一对。
            cmd += ["--ref-pair-left", str(d["ref_pair_left"]),
                    "--ref-pair-right", str(d["ref_pair_right"])]
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
        # 自由采集档（2026-11-21）：只对**现场开相机**的 joint 分支有效；
        # chain/chain_dual 读的是图片目录，传了也没用。空/非法/<=0 不传
        # （= 后端走自适应 + 默认 fps），绝不把 "abc" 拼进 argparse。
        cmd += _positive("--width", width)
        cmd += _positive("--height", height)
        cmd += _positive("--fps", fps)
    # 高级项：只有用户真的填了才下传（空 = 让后端用自己的默认值）
    if str(stationary or "").strip():
        cmd += ["--stationary-px", str(stationary).strip()]
    if str(sharpness or "").strip():
        cmd += ["--min-sharpness", str(sharpness).strip()]
    if relaxed:
        cmd += ["--relaxed"]
    if str(backend or "").strip() and str(backend).strip() != "auto":
        cmd += ["--backend", str(backend).strip()]
    # 强制通过（仅链路测试）：质量差时把结果写进正式 --out 路径，而不是默认的
    # 旁路文件。GUI 只在二次确认通过后才置 True；argv 仍只有这一个产地。
    if force_save:
        cmd += ["--force-save"]

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


def _board_args_no_target(board: dict | None) -> list[str]:
    """只取 `--cols/--rows/--square`：`calibrate_axial.py` / `calibrate_device90.py`
    没有 `--target`（它们不靠"目标对数"收图），拼进去会 argparse 直接 exit 2。"""
    out: list[str] = []
    for key, flag in _BOARD_FLAGS:
        if key == "target":
            continue
        raw = str((board or {}).get(key) or "").strip()
        out += [flag, raw or BOARD_DEFAULTS[key]]
    return out


def calibrate_axial_args(*, root: str, board: dict | None = None, out: str = "",
                         assume_parallel: bool = False,
                         expected_baseline_mm: float | str | None = None,
                         baseline_tolerance: float = 0.05,
                         force_save: bool = False) -> list[str]:
    """轴向标定（`apps/calibrate_axial.py`）的 argv。

    与 `calibrate_args` 分开：那个函数的三条分支已经很密，轴向入口的目录语义
    （`--root` + 深度子目录）完全不同，混进去只会让两件事互相牵制。
    """
    cmd = _argv(ROOT / "apps" / "calibrate_axial.py", "--root", str(root))
    cmd += _board_args_no_target(board)
    cmd += ["--baseline-tolerance", f"{float(baseline_tolerance):g}"]
    if assume_parallel:
        cmd += ["--assume-parallel"]
    # 预期基线：空 / 非数字 / ≤0 **一律不传**（拼 "abc" 会让 argparse exit 2）
    cmd += _positive("--expected-baseline-mm", expected_baseline_mm)
    if str(out or "").strip():
        cmd += ["--out", str(out).strip()]
    if force_save:
        cmd += ["--force-save"]
    return cmd


def calibrate_device90_args(*, left_dir: str, right_dir: str, intrinsics: str,
                            board: dict | None = None, out: str = "",
                            rot_axis: str = "y", rot_angle: str | float = "90",
                            z_a_mm: str | float | None = None,
                            z_b_mm: str | float | None = None,
                            expected_baseline_mm: float | str | None = None,
                            baseline_tolerance: float = 0.05,
                            force_save: bool = False) -> list[str]:
    """中轴 90° 装置标定（`apps/calibrate_device90.py`）的 argv。

    只吃 A/B 两段正对图 + 外部内参；**不带** `--ref-pair-*` / `--pair-frame`
    （这正是该模式省掉的 ③④）。
    """
    cmd = _argv(ROOT / "apps" / "calibrate_device90.py",
                "--left-dir", str(left_dir), "--right-dir", str(right_dir),
                "--intrinsics", str(intrinsics))
    cmd += _board_args_no_target(board)
    try:
        ang = float(str(rot_angle).strip()) if str(rot_angle or "").strip() else 90.0
    except ValueError:
        ang = 90.0
    cmd += ["--rot-axis", str(rot_axis), "--rot-angle", f"{ang:g}",
            "--baseline-tolerance", f"{float(baseline_tolerance):g}"]
    cmd += _positive("--z-a", z_a_mm)
    cmd += _positive("--z-b", z_b_mm)
    cmd += _positive("--expected-baseline-mm", expected_baseline_mm)
    if str(out or "").strip():
        cmd += ["--out", str(out).strip()]
    if force_save:
        cmd += ["--force-save"]
    return cmd


def offline_stereo(session: str | Path, *, mode: str = "",
                   anchor: bool = False, method: str = "", seed_json: str = "",
                   preset: str = "", min_radius: str = "",
                   max_radius: str = "", radius_auto: bool = True,
                   anchor_diameter_mm: str = "") -> list[str]:
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
        # 物理球直径（mm）只喂给半径锚定门换算视半径，**不控制尺寸门**。
        # 只在勾了锚点门且值合法时才下传：空 / 非数字 / ≤0 一律不拼，
        # 免得 argparse 直接 exit 2（_positive 与 --min/--max-radius 同款口径）。
        cmd += _positive("--anchor-diameter-mm", anchor_diameter_mm)
    cmd += _flag("--method", method)
    cmd += _flag("--preset", preset)
    cmd += _positive("--min-radius", min_radius)
    cmd += _positive("--max-radius", max_radius)
    # 尺寸门自动阶梯（2026-11-08）：默认开；界面上取消勾选 → 显式关掉，
    # 让"界面说自动、命令行其实写死"这种断链能被判据抓到。
    if not radius_auto:
        cmd += ["--no-radius-auto"]
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
                label: str = "", container: str = "",
                width: str = "", height: str = "", fps: str = "") -> list[str]:
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

    分辨率/帧率/后端：**空 = 不传**，录制程序按标定 `capture_size` 与默认 fps
    自己决定；用户在界面填了才下传，并仍由 `validate_capture_size` 校验
    「填的档 vs 标定档」——界面不替它做第二种校验。

    ``--preview --wait-key space`` 是**必传**的（2026-11-01 修）：
    老界面一直传 ``--preview``，而新界面从重做那天起就没传过 —— 于是"录制时
    看不到实时画面"成了新界面的独有现象，且**不报错**（`--preview` 是
    `store_true`，不传就是没有窗口）。用户报"标定完实时画面没了"，根因就是这一行。
    同时把起录方式改成"先出画面，按空格才开始倒计时"：程序刚起来时相机常常
    还没出第一帧，倒计时一共才 3 秒，等于把准备时间从人手里抢走。

    ``container`` 是录制格式（``mkv`` / ``avi``），空串 = 不传、跟随后端默认。
    mp4 **不在这里的候选里**（见 ``record_stereo.CODECS``：无损编码进 mp4 会被
    FFmpeg 静默换成有损 h264）。非空就原样传给后端，由 argparse 的 choices
    给出明确报错，不在这一层再抄一份白名单。
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
    if str(container or "").strip():
        cmd += ["--container", str(container).strip().lower()]
    # 自由采集档：空/非法/<=0 不传，交给录制程序按标定 capture_size 决定。
    cmd += _positive("--width", width)
    cmd += _positive("--height", height)
    cmd += _positive("--fps", fps)
    return cmd


# --------------------------------------------------------------------------- #
# 打开类动作（不起任务、只看/管理）：相机设备、数据管理、3D 查看器
# --------------------------------------------------------------------------- #

def has_seeds(session: str | Path) -> bool:
    """这个会话里有没有**人点过**的球心（`derived/seeds.json`）。

    界面拿它做前置检查：勾了「半径锚点门」却没点选过，命令行那边会拒绝并退出码 2，
    但那时用户已经点了按钮、等了一次进程启动 —— 能在界面上先说清楚就不要让人等。
    """
    p = Path(session) / "derived" / "seeds.json"
    return p.is_file()


def seed_ball_kind(session: str | Path) -> str:
    """``seeds.json`` 里记的球种类：``metal`` / ``plastic`` / 空。

    空有三种可能：没有种子文件、旧文件没写这个字段、文件坏了。读取方要区分
    "没有种子"（`has_seeds()`）与"有种子但没记种类"（旧文件，按 metal 兼容）。
    **这里不抛异常**：界面在"选会话"那一刻调用它，一个坏 json 不该把窗口
    卡住 —— 具体坏在哪由 `run_stereo_video` 在真正开跑时给出。
    """
    p = Path(session) / "derived" / "seeds.json"
    if not p.is_file():
        return ""
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    if not isinstance(data, dict):
        return ""
    return str(data.get("ball_kind") or "")


def cameras(sub: str = "identify") -> list[str]:
    """相机设备管理。``identify`` 是交互式的（自己开 cv2 窗口逐台确认）。

    为什么复用它而不是在新界面重画一个：它已经处理了"两台各自只在不同后端下
    出图"这种真机状态（``probe`` 逐后端×逐编号实测），重画一遍只会分叉。
    """
    return _argv(_TOOLS / "cameras.py", sub)


def camera_modes(*, left: str = "left", right: str = "right",
                 indices: str = "", modes: str = "", fps: str = "",
                 backend: str = "msmf") -> list[str]:
    """「使用相机实际模式」：逐档真开、报实际宽高/实测帧率（只读）。

    走 `cameras.py modes --json`：最后一行是 JSON，界面从任务日志尾解析。
    `indices` 非空时按**实测采集编号**打开（覆盖 left/right）；否则按角色名
    解析 capture_index。空 `--left/--right` 不传（脚本有默认），空
    `--modes/--fps` 也不传（默认梯子 + 按 30 请求并实测）。整场只用一个后端。
    """
    who = (["--indices", str(indices).strip()] if str(indices or "").strip()
           else [*_flag("--left", left), *_flag("--right", right)])
    return _argv(
        _TOOLS / "cameras.py", "modes", *who,
        *_flag("--modes", modes), *_flag("--fps", fps),
        *_flag("--backend", backend), "--json")


def camera_probe(index: str) -> list[str]:
    """「试开这台相机」：只按编号测一台（``camera_quality.py``）。

    argv **只能**给 ``--indices N``：该脚本的 ``resolve_targets`` 先按
    ``--left/--right`` 取，取到了就**忽略** ``--indices``；带上 left/right
    "试开这台"会变成测两台甚至测错那台。索引是唯一产地 = 这里，界面只传编号。

    注意它打不开相机也照样 ``return 0``，只在日志尾打印「打不开：…」或
    「实测帧率 …」—— 成败得由调用方读 ``tail_text()`` 判断，**不能看退出码**。
    """
    return _argv(_TOOLS / "camera_quality.py", "--indices", str(index))


def data_manager() -> list[str]:
    return _argv(_APPS / "data_manager.py")


def run_all(session: str | Path, *, stereo_mode: str = "",
            detector: str = "") -> list[str]:
    """会话 → 校验 → 离线三维 → 标准 JSON 导出的**一条命令**（`apps/run_all.py`）。

    为什么给 GUI 用这一条而不是把三步各点一次：三步的依赖关系（缺标定跳过、
    缺轨迹跳过、退出码 3 = 全跳过）已经在 `run_all` 里有一套判据，界面再拼
    一遍就会分叉。空串 = 不传，让脚本按会话/注册表自己判断。
    """
    return _argv(_APPS / "run_all.py", "--session", str(session),
                 *_flag("--stereo-mode", stereo_mode),
                 *_flag("--detector", detector))


def view_3d(source: str | Path, video_source: str = "") -> list[str]:
    """3D 查看器。``source`` 是**必填**（view_3d 的第一个位置参数）。

    ``video_source`` ∈ ``annotated`` / ``raw`` / ``none``，空串 = 不传
    （查看器默认 ``annotated``）。老界面在数据管理里给过 raw 入口，新界面
    的查看器现在也能在窗口内切换；这里保留参数是为了让命令行/数据管理
    仍能**直接指定**打开哪一路视频。
    """
    return _argv(_APPS / "view_3d.py", str(source),
                 *_flag("--video-source", video_source))


def stereo_realtime_args(
    *,
    left: str = "",
    right: str = "",
    width: str = "",
    height: str = "",
    fps: str = "",
    detect_width: str = "960",
    display_width: str = "1600",
    max_skew_ms: str = "",
    calib: str = "",
    method: str = "color",
    preset: str = "",
) -> list[str]:
    """双目实时三维检测的 argv（`apps/run_realtime_stereo.py`）。

    **默认值就是界面上要的那一份**：`--detect-width 960` 不是可选优化 ——
    全分辨率下 `color` 单路 17.8 ms（`pipeline.py:184-203` 实测），两路串行
    35.6 ms 再加渲染，帧率会掉到 20 fps 以下，而这个功能的价值就是"实时"。

    空串一律**不拼**参数（`--left ""` 与不传在子进程里是两件事：前者会让
    角色解析报错）。检测方法只允许 `color` / `motion`（hough 4.4 s/帧、
    fusion 3.7 s/帧，见 `docs/待办清单.md` B-16）—— 这一层不校验，
    由后端给出带可用项的明确报错，避免两处各写一份白名单。
    """
    cmd = _argv(
        _APPS / "run_realtime_stereo.py",
        *_flag("--left", left),
        *_flag("--right", right),
        *_flag("--width", width),
        *_flag("--height", height),
        *_flag("--fps", fps),
        *_flag("--detect-width", detect_width),
        *_flag("--display-width", display_width),
        *_flag("--max-skew-ms", _derived_max_skew_ms(max_skew_ms, fps)),
        *_flag("--calib", calib),
        *_flag("--method", method),
        *_flag("--preset", preset),
    )
    return cmd


def _derived_max_skew_ms(max_skew_ms: str, fps: str) -> str:
    """两路时间差门限：显式给了就用，没给就按帧率派生（与后端同一条口径）。

    语义见 ``apps/run_realtime_stereo.default_max_skew_ms``：30 fps 及以下 = 60 ms，
    更高帧率按 ``60*30/fps`` 收紧、不低于 2 ms。这里**不 import 那个模块**
    （本模块只依赖标准库与 ``paths``，见文件头约定），公式与它保持同步；
    传空 fps 时退化为旧默认 60 —— 也就是"没有帧率信息 = 按 30 fps 相机处理"。

    必须在**界面层**也算一遍的原因：不传 `--max-skew-ms` 时后端会自己派生，
    但界面上显示/日志里若还写 60，用户会以为门限没跟着帧率变。
    """
    text = str(max_skew_ms or "").strip()
    if text:
        return text
    try:
        f = float(str(fps or "").strip())
    except ValueError:
        f = 0.0
    value = 60.0 if f <= 0 or f <= 30.0 else max(60.0 * 30.0 / f, 2.0)
    # `:g` 而不是裸 str()：60.0 要写成 "60"（旧 argv 逐字不变），3.6 仍是 "3.6"。
    return f"{round(value, 2):g}"


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
                         width: int = 0, height: int = 0, fps: int = 0,
                         dirs: dict | None = None) -> list[str]:
    """采集链式标定的某一段图（`capture_board.py`），落进 `chain_dirs()` 对应目录。

    ``stage`` ∈ {``left``, ``right``, ``ref``, ``dual_b``, ``ref_pair``}：
    - ``left``/``right``：单板模式的内参段，单路（板可移动，多姿态）；
    - ``ref``：单板模式的参考段，``--both`` 双路同拍（板**固定不动**，T_AB=I）；
    - ``dual_b``：**双板模式**里右相机拍 B 板（落 `board_b` 目录），与 ``left``
      （拍 A 板）配对。同框图（pair）不能走 `capture_board.py`（它只检单板），
      需单独采集，所以这里没有 ``pair`` 段；
    - ``ref_pair``：**双板模式**的参考对（2026-11-09 方案 A）。组合装置摆好
      别动，两路**同时**各拍 3 张（= 3 对，GUI 第 ③ 步默认），落
      `ref_pair_left`/`ref_pair_right`；链上两端的板位姿只从这些对里取。
      复用 ``--both``（同名文件天然成对）；``target`` **由调用方给**
      （GUI ③ 现在是 3），取景窗存够就自己关，用户只需点一次按钮。

    参考段**不传** ``--camera-left/--camera-right``：与旧界面一致，让
    `capture_board.py` 自己按 left/right 角色取相机（角色绑定更稳，索引会漂移）。

    ``min_cover`` / ``relax_after`` 只对内参段有效（双板 A/B 与单板左右）：
    前者是**首选目标**（5=五个角全绿，最理想），后者是**分化线**——拍够这么多张
    后覆盖标准放宽到 4/5。参考段不用覆盖判据（它的规则是"板不能动"）。

    ``width`` / ``height`` / ``fps`` 来自**激活装置档案**的采集档（0 = 不传，
    由 `capture_board.py` 用它自己的默认）。为什么必须由档案给：`capture_board`
    的默认是 2560x1440，而本机实测**双路同时开 2560x1440 会让其中一路的 MSMF
    会话失效**（2026-11-05，1920x1080 则两台 15/15 帧稳定）。若照默认拍标定图，
    标定的 capture_size 会被记成 2560，之后录制按它请求双开 → 必然失败。
    所以采集档要随档案走，拍图与录制用同一档。
    """
    d = dirs if dirs is not None else chain_dirs()
    # 同 calibrate_args：容忍只给部分键的自定义 dirs。
    d = {**chain_dirs(Path(d["left"]).parent), **d}
    cap: list[str] = []
    if width > 0:
        cap += ["--width", str(width)]
    if height > 0:
        cap += ["--height", str(height)]
    if fps > 0:
        cap += ["--fps", str(fps)]
    if stage == "ref":
        return _argv(
            _TOOLS / "capture_board.py", "--both",
            "--out-left", str(d["ref_left"]),
            "--out-right", str(d["ref_right"]),
            "--cols", str(cols), "--rows", str(rows), "--square", str(square),
            "--target", str(target), "--auto", *cap)
    if stage == "ref_pair":
        # 双板参考对：两路同时各一张、板绝对不动。--stage ref 让 HUD 显示
        # 「板有没有动」；target 由调用方给（GUI 第 ③ 步现在 3 对），
        # 取景窗存够就自己关 —— 这里**不许写死**张数（2026-11-11 修：
        # 曾硬编码 "--target 1"，界面说 3 对、实际只拍 1 对）。
        return _argv(
            _TOOLS / "capture_board.py", "--both",
            "--out-left", str(d["ref_pair_left"]),
            "--out-right", str(d["ref_pair_right"]),
            "--cols", str(cols), "--rows", str(rows), "--square", str(square),
            "--target", str(target), "--auto", "--stage", "ref", *cap)
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
            *(["--per-region", str(per_region)] if per_region > 0 else []),
            *cap)
    side = "left" if stage == "left" else "right"
    out = d["left"] if stage == "left" else d["right"]
    return _argv(
        _TOOLS / "capture_board.py", "--out", str(out), "--side", side,
        "--camera", str(left if stage == "left" else right or "").strip() or side,
        "--cols", str(cols), "--rows", str(rows), "--square", str(square),
        "--target", str(target), "--auto",
        *(["--min-cover", str(min_cover)] if min_cover > 0 else []),
        *(["--relax-after", str(relax_after)] if relax_after > 0 else []),
        *(["--per-region", str(per_region)] if per_region > 0 else []),
        *cap)


def capture_pair_frame(*, cols: str, rows: str, square: str,
                       camera: str = "left",
                       width: int = 0, height: int = 0, fps: int = 0,
                       dirs: dict | None = None) -> list[str]:
    """拍双板链式的**同框图**（pair）：单相机、不做单板检测、按空格存。

    双板模式求 T_AB 需要一张"两块板同时入画"的图，`capture_board.py` 的单板
    检测器检不了它，所以走 ``--pair`` 模式（2026-11-04 前这块只能让用户手动拍）。
    """
    d = dirs if dirs is not None else chain_dirs()
    # 采集档与内参段保持一致：同一次标定的所有图应当同一口径（见
    # `capture_chain_frames` 里那段说明）。0 = 不传，由 capture_board 用默认。
    cap: list[str] = []
    if width > 0:
        cap += ["--width", str(width)]
    if height > 0:
        cap += ["--height", str(height)]
    if fps > 0:
        cap += ["--fps", str(fps)]
    return _argv(
        _TOOLS / "capture_board.py", "--out", str(d["pair"]),
        "--camera", str(camera or "left").strip() or "left",
        "--cols", str(cols), "--rows", str(rows), "--square", str(square),
        # 5 张同框图（2026-11-05 用户：单张有风险；2026-11-11 用户明确 ④ 保持 5 张
        # —— 改成 3 张的是 ③ 参考对，不是这里）。T_AB 只用其中一张，
        # 但多拍几张能在"哪张两板都检全"上留选择余地，也防单张糊掉。
        "--target", "5", "--pair", *cap)


def check_chain_inputs(*, cols: str, rows: str, square: str,
                       dirs: dict | None = None,
                       mode: str = "single") -> list[str]:
    """链式标定输入自检（`check_chain_inputs.py`），拍完先跑这个。

    ``mode`` 决定查哪几段（与 `calibrate_args` 的链式分支一一对应）：
    ``single`` = 单板四段（left/right/ref_left/ref_right）；
    ``dual``   = 双板四步（left/board_b + ref_pair_left/ref_pair_right + pair）。
                 2026-11-09 方案 A 起多了一组「组合装置别动、两路同时各拍 3 张」
                 的参考对，链两端位姿只从它取。
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


# --------------------------------------------------------------------------- #
# 2026-11-07 盘点补的六条「只在命令行里能走」的流程
# --------------------------------------------------------------------------- #
# 为什么在这里而不是各窗口里各拼一份：上面那段模块注释已经说过 —— 同一个动作
# 两份参数清单必然漂移。这六条的共同点是**它们在 README 里被写成"下一步做什么"，
# 界面上却一个入口都没有**：用户走到那一步就断了，只能回头翻文档。
#
# 六条：生成标定板 / 点选取色 / 设世界坐标系 / 导出标定参数 /
#       导出训练数据集 / 训练球检测器。

def make_board(*, cols: str = "", rows: str = "", square: str = "",
               page: str = "", out: str | Path = "") -> list[str]:
    """生成可打印的棋盘标定板（PDF + PNG，`assets/boards/`）。

    留空一律落到脚本默认（8×5 / 25 mm / a4-landscape）—— 界面不替用户猜，
    但也不拦他什么都不填。
    """
    return _argv(_TOOLS / "make_calibration_board.py",
                 *_flag("--cols", cols), *_flag("--rows", rows),
                 *_flag("--square", square), *_flag("--page", page),
                 *_flag("--out", out))


def calibrate_color(*, image: str | Path = "", camera: str = "",
                    save: str | Path = "") -> list[str]:
    """点选取色：开一个带 HSV 滑块的 cv2 窗口，人拖着调。

    ``image`` 与 ``camera`` 二选一；两个都没给时返回 ``--camera 0`` 让脚本
    自己去开默认相机（比返回空 argv 好：空 argv 会让 `TaskRunner` 起一个
    立刻退出的进程，用户只看到"闪一下就没了"）。
    """
    src = _flag("--image", image) or ["--camera", str(camera).strip() or "0"]
    return _argv(_TOOLS / "calibrate_color.py", *src, *_flag("--save", save))


def set_world_frame(*, image: str | Path, cols: str = "", rows: str = "",
                    square: str = "", note: str = "", out: str | Path = "",
                    ground_axis: str = "", rig: str | Path = "",
                    dry_run: bool = False) -> list[str]:
    """用一张「棋盘平放在地面上」的左相机图求 ``T_world_cam``。

    ``image`` 是**必填**（脚本的 ``--image`` 也是 required）—— 调用方必须先
    确认它有值，否则这里的 ``str(image)`` 会把空串原样传下去，脚本报
    "参数错"而用户只看到一句 argparse 错误。

    ``ground_axis``（``z`` / ``-z``）与 ``rig``（标定基名）**留空一律不传**：
    前者交给脚本默认（``z`` = 与棋盘法向同向），后者交给
    ``current_calibration()`` 的当前装置档案。这两个参数以前界面上根本没有，
    结果世界系的 Z 朝上/朝下只能靠改配置文件 —— 而符号写错的矩阵
    "看起来完全正常"。
    """
    return _argv(_TOOLS / "set_world_frame.py", "--image", str(image),
                 *_flag("--board-cols", cols), *_flag("--board-rows", rows),
                 *_flag("--board-square", square), *_flag("--note", note),
                 *_flag("--ground-axis", ground_axis), *_flag("--rig", rig),
                 *_flag("--out", out), *(["--dry-run"] if dry_run else []))


def export_calibration(*, calib: str | Path = "", out: str | Path = "",
                       formats: str = "") -> list[str]:
    """把双目标定导成 JSON / YAML / TOML（分相机 + 双目整体）。

    只读文件、不碰相机，所以什么时候导都安全；换台机器复现实验时靠它。
    """
    return _argv(_TOOLS / "export_calibration.py",
                 *_flag("--calib", calib), *_flag("--out", out),
                 *_flag("--formats", formats))


def export_ball_dataset(*, out: str | Path = "", session: str | Path = "",
                        all_sessions: bool = True,
                        val_sessions: str = "", dry_run: bool = False) -> list[str]:
    """把「点选 → 跟踪」出来的轨迹导成 YOLO 单类检测数据集。

    这是「以后不用手动点球心」那条路的第一步：标注是跟踪结果的副产品，
    不用手画框。``session`` 给了就只导那一段（**优先于** ``all_sessions``）；
    否则按 ``all_sessions`` 决定是全导还是干脆不传（脚本会要求至少给一个）。
    """
    if session:
        src = ["--session", str(session)]
    elif all_sessions:
        src = ["--all"]
    else:
        src = []
    return _argv(_TOOLS / "export_ball_dataset.py",
                 *_flag("--out", out), *src,
                 *_flag("--val-sessions", val_sessions),
                 *(["--dry-run"] if dry_run else []))


def train_ball_detector(*, data: str | Path = "", epochs: str = "",
                        imgsz: str = "", batch: str = "", device: str = "",
                        out: str | Path = "") -> list[str]:
    """用导出的数据集自举训练一个 YOLO11 单类球检测器。

    L3 参数（``epochs``/``imgsz``/``batch``/``device``）留空就用脚本默认，
    **不在这里重复写默认值**：写一遍就多一处漂移（本模块约定第 2 条）。
    """
    return _argv(_TOOLS / "train_ball_detector.py",
                 *_flag("--data", data), *_flag("--epochs", epochs),
                 *_flag("--imgsz", imgsz), *_flag("--batch", batch),
                 *_flag("--device", device), *_flag("--out", out))


# --------------------------------------------------------------------------- #
# 链式标定：只重拍某一步（2026-11-11）
# --------------------------------------------------------------------------- #
# 以前想重拍第 ③ 步只能点「采集标定板图…」—— 它无条件新建会话并改写
# ``_current.json``，于是旧会话里 ①②④ 的有效图对程序等于不存在，四步必须
# 全部重来。这里把「把该步目录移进回收站 + 重建为空目录」抽成一个可 dry_run
# 的动作：**不新建会话、指针与其它段的图原样保留**，重拍完只需重新跑一次
# 自检与标定（标定本来就全量读五段），不需要重拍其它段。
# --------------------------------------------------------------------------- #
def reshoot_chain_stage(dirs: dict[str, Path], keys: tuple[str, ...], *,
                        label: str = "chain-stage",
                        dry_run: bool = False) -> dict:
    """只重拍链式标定的某一步：把该步目录**先移到回收站**，再重建为空目录。

    ``dirs`` 来自 ``current_chain_dirs()``（或自测里的临时会话）；``keys`` 是这
    一步对应的目录键，第 ③ 步是 ``("ref_pair_left", "ref_pair_right")`` 两个。

    图片是真机拍的、不可再生，所以绝不直接 ``rmtree``：先经
    ``plan_cleanup`` + ``move_to_trash`` 落到 ``outputs/_trash/<批次>/``
    —— 数据管理里的「回收站…」只管 ``data/_trash/``，**不管理这类批次**。
    旧图请用本模块的 :func:`restore_chain_stage` 恢复（``_cleanup.restore_batch``
    会把下面重建的空目录当成冲突而抛 ``FileExistsError``）。移走后**立刻**
    把该步各目录 ``mkdir`` 重建为空目录 —— 这是硬要求：``calibrate_args`` 只在
    ``Path(d["pair"]).exists()`` 时才下传 ``--pair-frame``，第 ④ 步目录若没
    建回来，标定 argv 会缺参数。

    ``dry_run=True`` 只返回计划，一个字节都不写（此时 ``batch`` 为 ``None``）。

    返回 ``{"batch", "moved", "n_files", "recreated", "dry_run"}``：``moved``
    是移走的目录相对 ``sessions_root`` 的路径，``recreated`` 是重建的目录。
    """
    # 纯逻辑层，不是界面模块；延迟 import，不给本模块的常规加载添依赖。
    from apps._cleanup import move_to_trash, plan_cleanup

    key_list = tuple(keys)
    if not key_list:
        raise ValueError("keys 不能为空")
    for k in key_list:
        if k not in dirs:
            raise ValueError(f"未知目录键：{k!r}")

    # 会话目录：``dirs`` 里所有路径都应落在同一个 ``<sessions_root>/<会话>/``
    # 下。用 ``resolve()`` 判归属，挡住符号链接与 ``..`` 逃逸；``plan_cleanup``
    # 的 ``_resolve_target`` 还会再挡一次（两层目录、非根、非符号链接）。
    resolved = {k: Path(p).resolve() for k, p in dirs.items()}
    parents = {p.parent for p in resolved.values()}
    if len(parents) != 1:
        raise ValueError("dirs 里的目录不在同一个会话下")
    session_dir = parents.pop()
    sessions_root = session_dir.parent
    if not session_dir.is_relative_to(sessions_root):
        raise ValueError(f"会话目录不在 sessions 根下：{session_dir}")
    for k in key_list:
        p = Path(dirs[k])
        rp = resolved[k]
        if p.is_symlink() or rp.is_symlink():
            raise ValueError(f"拒绝符号链接：{p}")
        if rp.parent != session_dir or not rp.is_relative_to(session_dir):
            raise ValueError(f"目录越出当前会话：{p}")

    targets = [str(dirs[k]) for k in key_list]
    # ``allow_nested=True`` 才能接受 ``<会话>/<段>`` 这两层；去掉它计划构造
    # 会直接抛 ValueError（判据里专门钉住这一点）。
    plan = plan_cleanup(sessions_root, targets, mode="session",
                        allow_nested=True)
    moved = [str(i.source.relative_to(sessions_root)).replace("\\", "/")
             for i in plan.items]
    n_files = plan.total_files
    recreated = [str(dirs[k]) for k in key_list]
    if dry_run:
        return {"batch": None, "moved": moved, "n_files": n_files,
                "recreated": recreated, "dry_run": True}

    batch = None
    if plan.items:
        # 某目录本就不存在时 plan 里没有 item，跳过移动但仍要建回来（见下）。
        batch = move_to_trash(sessions_root, plan, label=label)
    for k in key_list:
        Path(dirs[k]).mkdir(parents=True, exist_ok=True)
    return {"batch": batch, "moved": moved, "n_files": n_files,
            "recreated": recreated, "dry_run": False}


def restore_chain_stage(batch: str | Path | None) -> dict:
    """恢复 ``reshoot_chain_stage()`` 移到 ``outputs/_trash/`` 的那一步旧图。

    为什么需要专用入口（2026-11-11）：数据管理的「回收站…」只看
    ``data/_trash/``（``cleanup_dialog.py`` 用 ``trash_root(data_root)``），
    且它的 ``restore_batch(data_root, b.path)`` 会把图还原到
    ``data/stereo/...`` —— 位置是错的。链式重拍的批次在 ``outputs/_trash/``，
    ``reshoot_chain_stage`` 移走后又把该步目录 ``mkdir`` 重建为空目录，通用
    ``restore_batch`` 会因此抛 ``FileExistsError``。这里做三件它做不了的事：

    1. 用 manifest 里的 ``root``（= 当初的 ``sessions_root``）定位还原目标，
       不让调用方另传一个可能错的根；
    2. 目标存在且为空（正是重建的空目录）就先 ``rmdir``；存在且非空
       （用户已经重拍了新图）则**整批拒绝**，不覆盖；
    3. 先做完整预检再动手，动手阶段若有异常则回滚已恢复的目录，
       绝不留半恢复状态。

    返回 ``{"restored", "skipped", "reason", "batch"}``：整批拒绝时
    ``restored=0`` 且 ``reason`` 写明原因、批次保留；``dry_run=True`` 的
    ``batch=None`` 直接抛 ``ValueError``。
    """
    import shutil
    from apps._cleanup import trash_root

    if batch is None:
        raise ValueError("dry_run=True 的批次 batch=None，没有可恢复的内容")
    b = Path(batch).resolve()
    man_path = b / "manifest.json"
    if not man_path.is_file():
        raise ValueError(f"不是有效的链式重拍批次（缺 manifest.json）：{b}")
    man = json.loads(man_path.read_text(encoding="utf-8"))
    if man.get("schema") != "sphere-stereo-trash":
        raise ValueError(f"批次 schema 不是 sphere-stereo-trash：{man.get('schema')!r}")
    root_raw = str(man.get("root") or "")
    root = Path(root_raw)
    if not root_raw or not root.is_absolute():
        raise ValueError(f"manifest.root 不是绝对路径：{root_raw!r}")
    root = root.resolve()
    troot = trash_root(root).resolve()
    if b == troot or not b.is_relative_to(troot):
        raise ValueError(f"批次不在 {troot} 内，拒绝恢复：{b}")

    # ---- 完整预检：只读，不动任何文件（整批拒绝也在这一步，无需中途回滚）----
    plan = []                       # (src_r, dst_r, need_rmdir, raw)
    skipped: list[str] = []
    blocked: list[str] = []
    for item in man.get("items") or []:
        raw = str(item.get("path") or "")
        if not raw:
            raise ValueError("manifest 里有空 path，拒绝恢复")
        # --- 越界校验：挡 .. / 绝对路径 / 盘符 / 符号链接逃逸 ---
        # （反向自证会去掉这一整块，判据 4 必须变红）
        rel = Path(raw)
        src = b / "sessions" / raw
        dst = root / raw
        src_r, dst_r = src.resolve(), dst.resolve()
        bad = (rel.is_absolute() or ".." in rel.parts
               or raw.startswith(("/", "\\"))
               or (len(raw) > 1 and raw[1] == ":")
               or src.is_symlink() or dst.is_symlink()
               or src_r == b or not src_r.is_relative_to(b)
               or dst_r == root or not dst_r.is_relative_to(root))
        if bad:
            raise ValueError(f"manifest 路径越界（.. / 绝对路径 / 符号链接）：{raw!r}")
        if not src_r.is_dir():
            skipped.append(raw)     # 源已不在（例如被清掉），跳过但不阻塞
            continue
        need_rmdir = False
        if dst_r.exists():
            if not dst_r.is_dir():
                raise ValueError(f"恢复目标不是目录：{dst_r}")
            if any(dst_r.iterdir()):
                blocked.append(raw)  # 用户已拍新图 → 整批拒绝，绝不覆盖
            else:
                need_rmdir = True    # 正是 reshoot_chain_stage 重建的空目录
        plan.append((src_r, dst_r, need_rmdir, raw))

    if blocked:
        return {"restored": 0, "skipped": blocked,
                "reason": ("该步已有新图（" + "、".join(blocked)
                           + " 非空），恢复会覆盖，整批拒绝；旧图仍在批次 "
                           + str(b)),
                "batch": b}

    # ---- 动手：已完整预检；中途失败则把已恢复的目录搬回批次 ----
    done = []
    try:
        for src_r, dst_r, need_rmdir, raw in plan:
            if need_rmdir:
                dst_r.rmdir()
            dst_r.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_r), str(dst_r))
            done.append((src_r, dst_r))
    except Exception:
        for src_r, dst_r in reversed(done):
            if dst_r.is_dir():
                shutil.move(str(dst_r), str(src_r))
        for _src, dst_r, need_rmdir, _raw in plan:
            if need_rmdir and not dst_r.exists():
                dst_r.mkdir(parents=True, exist_ok=True)
        raise
    # 全部成功：批次已空，清掉（照 _cleanup.restore_batch 的收尾）
    shutil.rmtree(b, ignore_errors=True)
    return {"restored": len(done), "skipped": skipped, "reason": "",
            "batch": b}
