"""自动双目标定：把标定板在两台相机前挥一遍就行。

用法::

    # 1) 两台实时相机（推荐：能实时看到收帧进度）
    #    用**角色名**最省心：绑定的是设备身份，插拔 USB 口也不受影响。
    python apps/calibrate_stereo.py --left left --right right --cols 8 --rows 5 --square 25

    #    要用编号也行，但**编号的含义随后端变化**：解析用的设备顺序是 PnP 枚举，
    #    而 cv2 各后端的顺序不一样（本机实测 DSHOW 的 #1 是笔记本内置，
    #    MSMF 的 #2 才是）。本程序默认固定 MSMF，所以这里给的是 MSMF 口径的编号。
    #    不确定就先用角色名，或跑 python apps/tools/cameras.py list 看当前设备。

    # 2) 两段视频（指导文档里的流程：先分别录，再拿来标定）
    python apps/calibrate_stereo.py --left-video cam1.mp4 --right-video cam2.mp4 \
        --cols 8 --rows 5 --square 25

    # 3) 两个图片文件夹（每张都是独立位姿；此时不要求"板静止"）
    python apps/calibrate_stereo.py --left-dir imgs/left --right-dir imgs/right

程序会：
  - 自动判断每一帧该不该收（清晰度、大小、贴边、曝光、位姿多样性、覆盖均匀）
  - 收够之后自动标定，并用**极线误差**等指标判定能不能用
  - 把结果存成 config/stereo_calibration.npz + .json（含人读的质量报告）

关于同步：**标定时不需要左右视频同步**。程序只收"板停稳"的帧，
静止目标在任意时刻看起来都一样，所以时间偏移不影响标定。
（真正需要同步的是后续"拍运动的小球"，那是另一回事。）

帧源（摄像头 / 两段视频 / 两个图片文件夹）统一由 `stereo.PairSource` 提供 ——
复检工具与以后的双目实时流程共用同一份取帧逻辑，不再各写一份。
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

# --- 让脚本在任意目录深度都能找到项目根 ---
for _parent in Path(__file__).resolve().parents:
    if (_parent / "sphere_tracker").is_dir():
        sys.path.insert(0, str(_parent))
        break

import cv2
import numpy as np

from apps._window import WindowManager
from apps._actions import current_chain_dirs
from apps._common import (
    add_stereo_source_args,
    log_environment,
    open_pair_source,
    setup_logging,
)
from calib import AutoPairCollector, BoardSpec, CalibConfig, StereoRig, calibrate_stereo
from calib import detect_board, detect_two_boards
from calib.methods import plan as method_plan
from calib.methods import single_board as single_board_method
from calib.relative_pose import compose, invert, rotation_angle_deg
from calib.stereo_calib import board_pose_from_points, calibrate_mono_points
from stereo.profiles import get_active, resolve_calibration_base
from stereo.fov_align import (
    FovAlignment,
    estimate_fov_alignment,
    fov_confidence_problem,
    fov_gate_action,
)
from stereo.stereo_mode import optical_axes_angle_deg, resolve_auto
from sphere_tracker.plugins import default, describe, names
from sphere_tracker.sources import quiet_repeated_grab_warnings

WINDOW = "stereo calibration"
PREVIEW_W = 720          # 单路预览宽度的**上限**（实际按屏幕与文字条带反推，见主循环）
FOOTER_LINE_H = 19       # 文字条带每行高度（像素）
# chain（90°）产物的口径提示，必须与报告一起打印：这份参数**不能**喂给视差逻辑。
NONE_RAW_NOTE = ("未做极线校正：坐标即原始 left_camera 系；"
                 "任何依赖极线水平/视差（SGBM、y 差口径）的方法都不适用。")


def _screen_size() -> tuple[int, int]:
    """当前主屏分辨率；取不到就退回 1920x1080。

    为什么要它：面板比屏幕高的时候，``cv2.imshow`` 会把底部切掉 ——
    而被切掉的恰好是"为什么收不满"的那几条统计。所以布局必须知道屏幕多大，
    宁可把图像缩小，也不能让文字消失。

    用 ctypes 直接问系统，不引入第三方依赖；非 Windows 或调用失败时退回默认值。
    """
    try:
        import ctypes

        user32 = ctypes.windll.user32
        w, h = int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    return 1920, 1080


def _previous_calib_meta(out: str) -> dict | None:
    """读上一份标定 JSON（``StereoRig.save`` 写的是**平铺**的 meta）。

    读不到 / 不是对象就返回 ``None``：``--stereo-mode auto`` 在新装置上本来
    就没有旧文件可继承，这不是错误。
    """
    try:
        raw = json.loads(Path(out).with_suffix(".json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return raw if isinstance(raw, dict) else None


def _fov_from_profile(name: str, first_pair) -> FovAlignment:
    """从装置档案读**已量好**的 FOV 参数，跳过现场估计。

    ## 为什么要有这条路（2026-11-04 用户需求）

    两路相机的出厂焦距不同是**装置属性**：它不会因为换一块板、换一次场景而变。
    但以前的实现每次都从"启动帧"重估一次 —— 于是同一套装置标两次，得到两个
    略有差别的 `scale` / `crop_rect`，内参口径跟着漂；场景纹理不足时还会直接
    降级成 `fov=None`（按原始画面标），下一次又变回来。用户的原话是
    "两个相同摄像机打开后焦距好像不一样，但我又不能调整" —— 这种固定偏差
    应该**量一次就锁住**。

    ## 为什么必须核对分辨率

    `crop_rect` / `target_size` 是**像素量纲**。换分辨率（或换相机）之后旧数字
    直接失效，而画面看起来仍然"有输出"：裁剪框落错位置 → 内参与画面口径对不上
    → 视差/深度整体偏，但屏幕上一切正常。所以分辨率不一致时**报错**，
    不静默沿用、也不静默回退到现场估计（那会让用户以为复用生效了）。
    """
    from stereo import profiles

    try:
        prof = profiles.get_profile(name)
    except profiles.ProfileError as exc:
        raise ValueError(f"装置档案 {name!r} 的记录损坏：{exc}") from exc
    if prof is None:
        raise ValueError(
            f"装置档案 {name!r} 不存在（config/profiles.json）。"
            f"先在门厅「装置档案」里建它，或去掉 --fov-from-profile 改回现场估计。")
    raw = prof.fov
    if not raw:
        raise ValueError(
            f"装置档案 {name!r} 里还没有 FOV 参数。\n"
            f"  先量一次：python apps/tools/measure_fov.py "
            f"--left-video <会话>/left.mkv --right-video <会话>/right.mkv "
            f"--write-profile {name}\n"
            f"  或去掉 --fov-from-profile，让本次标定现场估计。")

    for key, frame, side in (("left_size", first_pair[0], "左"),
                             ("right_size", first_pair[1], "右")):
        want = raw.get(key)
        if not want:
            continue
        got = (int(frame.shape[1]), int(frame.shape[0]))
        if tuple(int(v) for v in want) != got:
            raise ValueError(
                f"装置档案 {name!r} 的 FOV 是在 "
                f"{int(want[0])}x{int(want[1])} 上量的，而现在{side}路是 "
                f"{got[0]}x{got[1]}。\n"
                f"  这些参数是像素量纲的，分辨率一变就失效。请重新量一次"
                f"（apps/tools/measure_fov.py --write-profile {name}），"
                f"或去掉 --fov-from-profile。")

    try:
        return FovAlignment.from_dict(raw)
    except ValueError as exc:
        raise ValueError(f"装置档案 {name!r} 的 FOV 记录不合法：{exc}") from exc


def _fov_or_none(estimate_fn, *, mode: str, require_fov: bool):
    """估计 FOV 对齐；**估不出来**与**估出来不可信**是两件事，这里只管前者。

    - 返回 ``(fov, None)``：估计成功。
    - 返回 ``(None, 原因)``：估计失败，调用方按**两路原始画面**继续
      （仅 ``parallel`` 且没要求 ``--require-fov``）。几何上完全合法：
      ``stereoCalibrate`` 不要求两路视场对齐。
    - 抛异常：``require_fov=True`` 或 ``mode != "parallel"`` 时保留旧的
      fail-closed 语义（要失败就明确失败，不静默降级）。

    失败原因必须带回给调用方打印 —— 降级不等于吞掉证据。
    """
    try:
        return estimate_fn(), None
    except Exception as exc:                     # noqa: BLE001 - 原因要原样带回
        if require_fov or mode != "parallel":
            raise
        return None, f"{type(exc).__name__}: {exc}"


# ---- 「宽松预设」（`--relaxed`）：手持拍摄时一次性放宽的收图门槛 ----
#
# 键 = `CaptureConfig` 的字段名；值 = 放宽后的门槛。**每一条都是"用精度换成功率"**。
# 单独放成 dict 的唯一原因：日志要说真话 —— `--stationary-px` / `--min-sharpness`
# 一旦显式给出就会**后写覆盖**这里的两项，而旧日志一律说"都放松了"，
# 用户因此以为最影响收图成功率的两项被放宽了（B-97）。
RELAXED_PRESET: dict[str, float] = {
    "stationary_max_px": 10.0,   # 允许手更抖；代价 = 两路位姿差变大
    "min_sharpness": 400.0,      # 额外绝对下限（只有高于 SB 检出下限时才真拦）
    "min_area_ratio": 0.008,     # 允许板更远；代价 = 单像素误差占比变大
    "min_margin_px": 6.0,        # 允许更贴边；代价 = 边缘畸变没被约束住
    "min_brightness": 25.0,
    "max_brightness": 245.0,
}

# 能被命令行/面板显式顶回的两项：出现在日志里要用人话（带参数名，便于对照）
_PRESET_OVERRIDE_LABELS = {
    "stationary_max_px": "停稳上限（--stationary-px）",
    "min_sharpness": "清晰度下限（--min-sharpness）",
}


def relaxed_note(stationary_px: float | None,
                 min_sharpness: float | None) -> list[str]:
    """`--relaxed` 的**如实**说明：几条真生效、哪几条被显式参数顶回。

    纯函数（不碰相机、不读 argparse 的命名空间），便于无相机自测。

    条数由实际入参算、不写死：`--stationary-px` / `--min-sharpness` 一旦显式
    给出，`cap_kw` 后写覆盖预设值，那两项就**没有**放宽。旧文案一律说
    "都放松了 / 一次放松 6 条"，与真实取值相反（B-97）。
    """
    overridden: list[str] = []
    if stationary_px is not None:
        overridden.append("stationary_max_px")
    if min_sharpness is not None:
        overridden.append("min_sharpness")
    effective = [name for name in RELAXED_PRESET if name not in overridden]
    lines = [f"⚠ 已启用 --relaxed（手持预设）：预设共 {len(RELAXED_PRESET)} 项，"
             f"生效 {len(effective)} 项、被你的显式参数覆盖 {len(overridden)} 项。"]
    if overridden:
        names = "、".join(_PRESET_OVERRIDE_LABELS.get(n, n) for n in overridden)
        lines.append(f"   被覆盖的（按你给的值，不是预设值）：{names}")
    lines.append("   这是用精度换成功率：用完必须复检极线误差。")
    lines.append("   如果标定完极线误差 > 0.3 px，或者代进去三维抖得厉害，")
    lines.append("   请处理好光照与支撑、去掉 --relaxed 重标一次。")
    return lines


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _list_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in _IMAGE_SUFFIXES else []
    if not path.is_dir():
        return []
    return sorted(p for p in path.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)


def _detect_two_boards(gray: np.ndarray, spec: BoardSpec):
    """一张图里检出**两块**同名标定板（L 形 / 垂直双板靶标）。

    实现在 `calib.board.detect_two_boards` —— 那里是唯一一份：双板标定与
    "拍完先自检"（`apps/tools/check_chain_inputs.py --mode dual`）必须用同一套
    检测逻辑，否则自检说"能检出两块"而标定说"检不出"就是必然的。
    这里**不做 A/B 语义分配**（谁是被左相机看到的 A，由 `_assign_ab` 判）。
    """
    return detect_two_boards(gray, spec)


def _pnp_reproj_rms(objp, corners, K, D) -> tuple[np.ndarray, float]:
    """`solvePnP` 求板位姿，并返回 (4x4 位姿, 重投影 RMS px)。"""
    T = board_pose_from_points(objp, corners, K, D)
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    uv, _ = cv2.projectPoints(np.asarray(objp, np.float64).reshape(-1, 3),
                              rvec, T[:3, 3].reshape(3, 1), K, D)
    err = np.asarray(corners, np.float64).reshape(-1, 2) - uv.reshape(-1, 2)
    return T, float(np.sqrt(np.mean(np.sum(err * err, axis=1))))


def _assign_ab(pose1: np.ndarray, pose2: np.ndarray,
               ref_rots: list[np.ndarray], pair_cam: str):
    """判定两板里哪一块是 A、哪一块是 B（**静默判错会直接给出反的 T_AB**）。

    依据：A 板是左相机视野里的那块，B 是右相机视野里的那块（两板刚性固定在
    同一支架上）。所以**同框图中某块板的姿态**，应当落在"该路目录里那块板的
    姿态簇"附近；另一块与它相差约两板夹角（90° 装置≈90°）。

    返回 `(T_A, T_B, d1, d2)`；若两块的姿态距离太接近（差 <5°），**报错**而不是猜。

    ⚠ 2026-11-11：双板同框图的 A/B 判定已改走 `_resolve_pair_samples`
    （整组全局指派 + 180° 等价解纠错 + 逐帧离群检查）——逐帧各自挑最近簇
    在 14:39 会话上把 5 帧里的 3 帧判反，T_AB 散布 179.597°。本函数保留
    作为"两板等距必须报错"的护栏（自测直接调用它），以及**逐帧**歧义的
    兜底说明；不要再把它接回同框图主路径。
    """
    d1 = min(rotation_angle_deg(pose1[:3, :3], r) for r in ref_rots)
    d2 = min(rotation_angle_deg(pose2[:3, :3], r) for r in ref_rots)
    if abs(d1 - d2) < 5.0:
        raise ValueError(
            f"同框图中两块板的姿态无法区分（到参考姿态簇的距离 "
            f"{d1:.1f}° vs {d2:.1f}°，差 <5°）。\n"
            f"  原因：两板夹角太小，或拍同框图时支架姿态与目录里的差异太大。\n"
            f"  下一步：让 A 板更正面朝向 --pair-camera 重拍同框图；"
            f"或把 --pair-frame 换成夹角更清楚的拍摄位置。")
    if pair_cam == "left":                    # 参考簇 = A 板
        return (pose1, pose2, d1, d2) if d1 < d2 else (pose2, pose1, d2, d1)
    # 参考簇 = B 板：更接近的那个是 B，另一块才是 A（返回顺序固定为 A, B, d_A, d_B）
    return (pose2, pose1, d2, d1) if d1 < d2 else (pose1, pose2, d1, d2)


# ---- 同框图的 180° 等价解纠错（2026-11-11）----
#
# 为什么需要：棋盘绕**板法线**转 180° 后，角点网格在几何上仍然自洽 ——
# 检测器若把起始角点/排列顺序整体翻过来，`solvePnP` 给的是 `T @ C`
# （C = 绕法线的 180° 旋转，轴过角点网格中心），它与原解**重投影几乎一样好**
# （实测：把角点倒序再 PnP，重投影 1.512 px，与正序一致），但旋转差 180°。
# 同框图里只要某块板落到这个等价解、或 A/B 指派在两帧间来回翻，`T_AB`
# 的样本就会分成两簇、相差约 180°（2026-11-11 14:39 会话实测 179.597°），
# 报告却把它读成"两板没刚性固连"——而真相是**同一个物理量的等价解选错了**。
#
# 处理分三层，**任何一层都不许把真实散布"修正"成通过**：
#   ① 板框等价解：每块板同时看 `T` 与 `T@C`，只有翻转解到参考姿态簇明显更近
#      （≥30°）才采用。只对同框相机**自己那一路**的板用这个判据 —— 另一块板
#      的参考姿态在**另一台相机**的坐标系里，直接比旋转不合法，交给层 ③。
#   ② A/B 指派：**全局**选一组（不是逐帧各自挑），用参考簇距离总和判定。
#      逐帧挑正是 14:39 的缺陷来源：两块板到参考簇只差 8~11°，检测噪声一抖
#      就有一半的帧判反，T_AB 于是 179.6° 两簇。
#   ③ 逐帧离群：与中位样本差 >45°（近 180° 的跳变）才枚举其它等价候选，
#      且必须改善 ≥30° 才接受。真实散布（验收用例故意造 10°）远低于该门限，
#      不会被"修"成一致。
PAIR_FLIP_CLEAR_DEG = 30.0     # 翻转解必须比原解至少近这么多才采用
PAIR_OUTLIER_DEG = 45.0        # 与中位样本差超过它 -> 视作等价解歧义（近 180° 跳变）
PAIR_OUTLIER_GAIN_DEG = 30.0   # 换候选必须至少改善这么多才算"修好了"
PAIR_ASSIGN_MARGIN_DEG = 10.0  # 两种全局 A/B 指派的参考距离总和差小于它 -> 判不了

# ---- 第四层（2026-11-18）：装置先验复核，堵"两板同时翻、全程一致" ----
#
# 上面三层的盲区：若 A、B 两块板在**每一张**同框图里都翻了 180°、且全程一致，
# ①只看"同框相机自己那块板"（另一块的参考位姿在另一台相机坐标系里，不能比），
# ②A/B 指派一致、③逐帧无离群 —— 三层全过，T_AB 样本却整体错 180°，
# consistency 还"漂亮"。chain 跳过 angled 极线复核（左右看不同板、无同名点），
# 因此唯一能自证的就是**装置先验**。
#
# 为什么**基线**是这一层的有效判据：`_board_180_equiv` 的 C 绕角点网格中心转，
# 平移项 `tc = 2·c = ((cols-1)·square, (rows-1)·square, 0)`（原点在第一个角点，
# 不在板心，见该函数 docstring）。两板同翻时
# `T_AB' = inv(C) · T_AB · C`，旋转被共轭（`R_AB' = Rz(pi)·R_AB·Rz(pi)`），
# **旋转测地角保持不变**（90° 翻完还是 90°），所以"两板相对角"抓不到这一例；
# 但平移多出 `Rz(pi)·(R_AB - I)·tc` 一项 —— 实测 8x5/25mm 板从 280.0 mm
# 变 373.7 mm（+33%），远超下面的 10% 门限。角度项仍在（覆盖别的错指派场景），
# 但本层主要靠基线触发。
PAIR_PRIOR_BASELINE_REL_TOL = 0.10  # 基线相对先验误差 ≤10%（卷尺读数本身有噪声）
PAIR_PRIOR_ANGLE_TOL_DEG = 15.0     # 板间角绝对门限：装置设计 90°、实测 88.55°，
                                    # ±15° 覆盖装配公差与标定噪声，仍能区分 0°/180° 错解


def _board_180_equiv(T: np.ndarray, spec: BoardSpec) -> np.ndarray:
    """棋盘角点顺序整体翻 180° 的等价板框：`T @ C`。

    `C = [Rz(pi) | c - Rz(pi)·c]`，`c` = 角点网格中心在板坐标系里的位置。
    为什么平移项不能省：`BoardSpec.object_points()` 的原点在**第一个角点**、
    不是网格中心；只乘 `Rz(pi)` 会把板心挪走（8x5、25 mm 板约 175/100 mm），
    `T_AB` 的平移凭空多出这一块，旋转对了但链还是错的。
    """
    s = spec.square_mm / 1000.0
    c = np.array([(spec.cols - 1) * s / 2.0, (spec.rows - 1) * s / 2.0, 0.0])
    Rz = np.diag([-1.0, -1.0, 1.0])
    C = np.eye(4)
    C[:3, :3] = Rz
    C[:3, 3] = c - Rz @ c
    return np.asarray(T, np.float64) @ C


def _rot_dist_to_refs(T: np.ndarray, ref_rots: list[np.ndarray]) -> float:
    """位姿 `T` 到参考姿态簇的最小旋转距离（度）。"""
    return min(rotation_angle_deg(T[:3, :3], r) for r in ref_rots)


def _pick_board_flip(T: np.ndarray, ref_rots: list[np.ndarray],
                     spec: BoardSpec) -> tuple[np.ndarray, bool, float, float]:
    """按到参考姿态簇的最小旋转距离选原解 / 180° 等价解。

    返回 `(T_used, flipped, d_orig, d_flip)`；**翻转解必须明显更近才用**
    （`PAIR_FLIP_CLEAR_DEG`）—— 宁可漏修一次，也不许把一帧本来正确的位姿
    强行翻成错的（那会制造一个 180° 假样本）。
    """
    d_orig = _rot_dist_to_refs(T, ref_rots)
    T_flip = _board_180_equiv(T, spec)
    d_flip = _rot_dist_to_refs(T_flip, ref_rots)
    if d_flip + PAIR_FLIP_CLEAR_DEG < d_orig:
        return T_flip, True, d_orig, d_flip
    return np.asarray(T, np.float64), False, d_orig, d_flip


def _resolve_pair_samples(frames, ref_rots: list[np.ndarray], spec: BoardSpec,
                          pair_cam: str):
    """把同框图的两块板位姿整理成**一致 A/B、同一等价解**后的 `T_AB` 样本。

    参数
    ----
    frames : list[dict]
        每帧 `{"name": str, "poses": [T0, T1], "errs": [e0, e1]}`；
        `T0/T1` 是 `detect_two_boards` 的原始顺序，约定 `p_cam = T @ p_board`。
    ref_rots : list[np.ndarray]
        `--pair-camera` 那一路目录里的板姿态簇（**同框相机自己那块板**的历史
        姿态）加上参考对里同一块板的姿态 —— 既用来判"这块板是不是翻了解"，
        也用来判"两块板里哪一块是它"。
    pair_cam : "left" | "right"
        同框相机是哪一路：`left` → 它自己对应的板是 A；`right` → 是 B。

    返回
    ----
    `(samples, rows, corrections, before)`：

    - `samples`：修正后每帧一个 `T_{B←A}`；
    - `rows`：逐帧展示信息（名字 / 到参考簇距离 / 用了哪些纠正）；
    - `corrections`：可机读的纠正台账（落 `quality["t_ab_corrections"]`）；
    - `before`：按**旧规则**（逐帧各自挑最近簇）会得到的样本，仅供打印
      "修正前"的数字作对照，不参与任何判定。

    判 A/B 用**整组**参考距离的总和，不用逐帧最近 —— 逐帧最近是 14:39 的
    缺陷来源（两板到参考簇只差 8~11°，一抖就判反）。两种全局指派的总和差
    不到 `PAIR_ASSIGN_MARGIN_DEG` 时直接报错：A/B 不可区分时宁可让人重拍，
    也不静默猜一个反的 `T_AB`。
    """
    if not frames:
        raise ValueError("没有可用的同框图（一张都没同时检出两块板）")
    if not ref_rots:
        raise ValueError("缺少参考姿态簇：判不了板框等价解与 A/B")

    own_is_a = pair_cam == "left"

    # ---- ② 两种全局 A/B 假设：参考簇距离总和更低的那组 ----
    totals = {}
    for h in (0, 1):
        tot = 0.0
        for fr in frames:
            T_used, flipped, d_orig, d_flip = _pick_board_flip(
                fr["poses"][h], ref_rots, spec)
            tot += d_flip if flipped else d_orig
        totals[h] = tot
    if abs(totals[0] - totals[1]) < PAIR_ASSIGN_MARGIN_DEG:
        raise ValueError(
            f"同框图中两块板到参考姿态簇的距离太接近（总和 "
            f"{totals[0]:.1f}° vs {totals[1]:.1f}°，差 <"
            f"{PAIR_ASSIGN_MARGIN_DEG:g}°），判不了哪块是 A/B。\n"
            f"  原因：两板夹角太小，或同框图拍摄姿态与内参段那批姿态差得太远。\n"
            f"  下一步：让 A 板更正面朝向 --pair-camera 重拍同框图；"
            f"或把 --pair-frame 换成夹角更清楚的拍摄位置。")
    h = 0 if totals[0] < totals[1] else 1
    a_idx = h if own_is_a else 1 - h
    b_idx = 1 - a_idx

    # ---- ① 板框等价解：只对"同框相机自己那块板"按参考簇判 ----
    entries = []
    corrections = []
    for fr in frames:
        raw = [np.asarray(fr["poses"][0], np.float64),
               np.asarray(fr["poses"][1], np.float64)]
        corrected = [raw[0].copy(), raw[1].copy()]
        flips = [False, False]
        T_used, flipped, d_orig, d_flip = _pick_board_flip(raw[h], ref_rots, spec)
        corrected[h] = T_used
        flips[h] = flipped
        if flipped:
            corrections.append({
                "file": fr["name"],
                "board": "A" if own_is_a else "B",
                "action": "板框翻 180° 等价解",
                "detail": (f"翻转解到参考簇 {d_flip:.2f}° < 原解 {d_orig:.2f}°"
                           f"（近 {d_orig - d_flip:.1f}°）"),
            })
        # 旧规则（逐帧各自挑最近参考簇）在这张图上会判成哪块当 A？与全局
        # 一致指派不同就是"这张图原本会被判反"，台账里必须点名 —— 14:39 的
        # 179.597° 正是 5 帧里 3 帧被逐帧规则判反造成的。
        d0 = _rot_dist_to_refs(raw[0], ref_rots)
        d1 = _rot_dist_to_refs(raw[1], ref_rots)
        old_own = 0 if d0 <= d1 else 1
        old_a = old_own if own_is_a else 1 - old_own
        ab_fixed = old_a != a_idx
        if ab_fixed:
            d_old = _rot_dist_to_refs(raw[old_a], ref_rots)
            d_new = _rot_dist_to_refs(raw[a_idx], ref_rots)
            corrections.append({
                "file": fr["name"],
                "board": "A/B",
                "action": "A/B 判反",
                "detail": (f"逐帧最近簇规则会把板{old_a + 1}当作 A（{d_old:.1f}°）；"
                           f"按全局一致指派纠正为板{a_idx + 1}（{d_new:.1f}°）"),
            })
        entries.append({
            "name": fr["name"],
            "raw": raw,
            "TA": corrected[a_idx],
            "TB": corrected[b_idx],
            "flip_a": flips[a_idx],
            "flip_b": flips[b_idx],
            "swap": False,
            "ab_fixed": ab_fixed,
            "errs": list(fr.get("errs", (float("nan"), float("nan")))),
        })

    def _tab(TA, TB):
        return compose(invert(TB), TA)

    samples = [_tab(e["TA"], e["TB"]) for e in entries]

    # ---- ③ 逐帧离群：与中位样本差 >45° 才试其它等价候选 ----
    if len(samples) >= 3:
        medoid = min(samples, key=lambda s: sum(
            rotation_angle_deg(s[:3, :3], t[:3, :3]) for t in samples))
        for k, e in enumerate(entries):
            d_now = rotation_angle_deg(samples[k][:3, :3], medoid[:3, :3])
            if d_now <= PAIR_OUTLIER_DEG:
                continue
            best = None
            raw = e["raw"]
            for swap in (0, 1):
                ai = b_idx if swap else a_idx
                bi = a_idx if swap else b_idx
                for fa in (0, 1):
                    for fb in (0, 1):
                        TA = _board_180_equiv(raw[ai], spec) if fa else raw[ai]
                        TB = _board_180_equiv(raw[bi], spec) if fb else raw[bi]
                        cand = _tab(TA, TB)
                        d = rotation_angle_deg(cand[:3, :3], medoid[:3, :3])
                        if best is None or d < best[0]:
                            best = (d, swap, fa, fb, TA, TB, cand)
            if best is None or best[0] + PAIR_OUTLIER_GAIN_DEG >= d_now:
                # 改善不够 -> 这是真实散布/坏帧，如实留着，让门禁去拒
                continue
            d_best, swap, fa, fb, TA, TB, cand = best
            e.update(TA=TA, TB=TB, swap=bool(swap),
                     flip_a=bool(fa), flip_b=bool(fb))
            samples[k] = cand
            if swap:
                action = "A/B 判反"
                board = "A/B"
            else:
                action = "板框翻 180° 等价解"
                board = "A" if fa else "B"
            corrections.append({
                "file": e["name"],
                "board": board,
                "action": action,
                "detail": (f"与中位样本旋转差 {d_now:.2f}° → {d_best:.2f}°"
                           f"（改善 {d_now - d_best:.1f}°）"),
            })

    rows = [{
        "name": e["name"],
        "d_a": _rot_dist_to_refs(e["TA"], ref_rots),
        "d_b": _rot_dist_to_refs(e["TB"], ref_rots),
        "flip_a": e["flip_a"], "flip_b": e["flip_b"], "swap": e["swap"],
        "ab_fixed": e["ab_fixed"],
        "errs": e["errs"],
    } for e in entries]

    # ---- 仅用于打印"修正前"：旧规则 = 每帧各自挑最近参考簇的那块当 A/B ----
    before = []
    for fr in frames:
        d0 = _rot_dist_to_refs(fr["poses"][0], ref_rots)
        d1 = _rot_dist_to_refs(fr["poses"][1], ref_rots)
        own = 0 if d0 <= d1 else 1
        ai = own if own_is_a else 1 - own
        bi = 1 - ai
        before.append(_tab(fr["poses"][ai], fr["poses"][bi]))
    return samples, rows, corrections, before


def check_pair_prior_consistency(samples, *, prior_baseline_mm=None,
                                 prior_angle_deg=None) -> dict:
    """第四层守卫：用装置先验复核 `_resolve_pair_samples` 的 T_AB 样本。

    只复核、不改样本、不抛异常、不改退出码 —— 结果由调用方写进
    ``quality.verdict`` / ``quality.warnings``，让下游按"不可用/仅链路测试"处理。

    参数
    ----
    samples : list[np.ndarray]
        三层纠错后的 `T_{B←A}` 样本。
    prior_baseline_mm : float | None
        卷尺量的两板间距（mm）。None/非正表示没先验，跳过基线项。
    prior_angle_deg : float | None
        装置设计/上次标定得到的两板相对角（度）。None 表示跳过角度项。

    返回
    ----
    dict：`ok`（是否通过）、`reasons`（不通过的人话原因列表）、`checked`
    （实际检查了哪几项）、`baseline_mm` / `angle_deg`（实测值）、以及两项相对误差。
    没给先验时 `ok=True`、`checked=[]` —— 缺先验绝不误判错解。
    """
    from calib import relative_pose               # 与 chain 调用处同源，不新造模块
    info = {
        "ok": True, "reasons": [], "checked": [],
        "baseline_mm": float("nan"), "angle_deg": float("nan"),
        "prior_baseline_mm": prior_baseline_mm,
        "prior_angle_deg": prior_angle_deg,
        "baseline_rel_err": None, "angle_err_deg": None,
    }
    if not samples:
        return info
    cons = relative_pose.consistency(samples)
    info["baseline_mm"] = float(cons["baseline_mm"])
    # 相对角取中位样本 R_AB 的测地角（对所有样本同向错的情况，中位数与均值一样）
    medoid = min(samples, key=lambda s: sum(
        rotation_angle_deg(s[:3, :3], t[:3, :3]) for t in samples))
    info["angle_deg"] = rotation_angle_deg(medoid[:3, :3], np.eye(3))

    if prior_baseline_mm is not None and float(prior_baseline_mm) > 0:
        prior_b = float(prior_baseline_mm)
        dev = abs(info["baseline_mm"] - prior_b) / prior_b
        info["baseline_rel_err"] = float(dev)
        info["checked"].append("baseline")
        if dev > PAIR_PRIOR_BASELINE_REL_TOL:
            info["ok"] = False
            info["reasons"].append(
                f"两板基线偏离装置先验 {dev * 100:.1f}%"
                f"（实测 {info['baseline_mm']:.1f} mm / 先验 {prior_b:.1f} mm，"
                f"门限 {PAIR_PRIOR_BASELINE_REL_TOL * 100:.0f}%）"
                "—— 疑似两板同时翻 180° 的等价解错解（三层纠错全过也照判）")

    if prior_angle_deg is not None:
        prior_a = float(prior_angle_deg)
        dev_a = abs(info["angle_deg"] - prior_a)
        info["angle_err_deg"] = float(dev_a)
        info["checked"].append("angle")
        if dev_a > PAIR_PRIOR_ANGLE_TOL_DEG:
            info["ok"] = False
            info["reasons"].append(
                f"两板相对角偏离装置先验 {dev_a:.1f}°"
                f"（实测 {info['angle_deg']:.1f}° / 先验 {prior_a:.1f}°，"
                f"门限 {PAIR_PRIOR_ANGLE_TOL_DEG:.0f}°）—— 疑似 A/B 错指派")
    return info


def _device_pair_priors(args) -> tuple[float | None, float | None]:
    """从**当前正式标定 json** 读装置先验（两板基线 / 相对角）。

    这不是新数据源：chain 第④步同框图左右看不同板、没有同名点，连 angled
    极线复核都被跳过（见下方 `angled_grade = "未评估"`），唯一独立于本次
    样本的证据就是上一次标定落盘时留下的卷尺基线 `quality.consistency.baseline_mm`
    与装置夹角 `angle_deg`。首次标定、旧格式、坏 JSON 都返回 (None, None)，
    第四层守卫随即跳过 —— 缺先验绝不误判。
    """
    def _pos_finite(v):
        try:
            f = float(v)
            return f if np.isfinite(f) and f > 0.0 else None
        except (TypeError, ValueError):
            return None

    try:
        out = Path(str(args.out))
        js = out.with_suffix(".json")
        if not js.is_file():
            return None, None
        data = json.loads(js.read_text(encoding="utf-8"))
        quality = data.get("quality") or {}
        cons = quality.get("consistency") or {}
        baseline = _pos_finite(cons.get("baseline_mm"))
        angle = _pos_finite(data.get("angle_deg") or quality.get("angle_deg"))
        return baseline, angle
    except Exception:                                # noqa: BLE001
        return None, None


def _chain_input_error(args) -> str | None:
    """检查 chain 的输入是否齐全；返回**可执行**的错误说明（None 表示齐全）。

    `--same-board`（单板模式）只需两路目录：两轮拍的是**同一块板**，
    公共参考系就是它自己的世界系，`T_AB` 按定义等于 `I` —— 没有"两块板"
    可测，所以同框图/`--pair-camera`/`--board-b-dir` 在这里既不需要也不接受。
    """
    same_board = bool(getattr(args, "same_board", False))
    if getattr(args, "left", None) is not None or getattr(args, "right", None) is not None \
            or getattr(args, "left_video", None) or getattr(args, "right_video", None):
        if same_board:
            return ("--method chain --same-board 本轮只支持**离线目录**：\n"
                    "  请改用：--left-dir <板目录> --right-dir <板目录>\n"
                    "  实时采集 chain 的 GUI 流程属于下一阶段（见计划书阶段 B）。")
        return ("--method chain 本轮只支持**离线目录**（链上用单目位姿，不做实时采集）。\n"
                "  请改用：--left-dir <A 板目录> --board-b-dir <B 板目录> "
                "--ref-pair-left <参考对左> --ref-pair-right <参考对右> "
                "--pair-frame <同框图目录> --pair-camera left|right\n"
                "  实时采集 chain 的 GUI 流程属于下一阶段（见计划书阶段 B）。")
    if same_board:
        missing = []
        if not getattr(args, "left_dir", None):
            missing.append("--left-dir <左相机标内参用的图像目录，板可移动>")
        if not getattr(args, "right_dir", None):
            missing.append("--right-dir <右相机标内参用的图像目录，板可移动>")
        if not getattr(args, "ref_left", None):
            missing.append("--ref-left <左相机看这块板、板**不动**的参考图目录>")
        if not getattr(args, "ref_right", None):
            missing.append("--ref-right <右相机看**同一块**板、同一个位置的参考图目录>")
        if not missing:
            return None
        why = ""
        if not getattr(args, "ref_left", None) or not getattr(args, "ref_right", None):
            why = ("  单板模式需要一组'板不动'的参考图，不能拿标内参那堆移动中的图来当外参参考：\n"
                   "  那堆图里板在移动（否则凑不出 15 个姿态），左右各挑一张就是两个不同摆放，\n"
                   "  它们的 T_AB 并不是 I，链出来的外参是错的。\n")
        return ("--same-board（单板模式）需要**两段**输入：\n"
                "  内参段：--left-dir + --right-dir（板可移动，各 ≥15 姿态）\n"
                "  外参段：--ref-left + --ref-right（板固定在同一个位置，各 1~5 张，"
                "两路都要完整看到整块板）\n"
                + why +
                "  缺少：" + "；".join(missing) + "\n"
                "  完整用法：python apps/calibrate_stereo.py --method chain --same-board "
                "--left-dir imgs/left --right-dir imgs/right "
                "--ref-left imgs/ref_left --ref-right imgs/ref_right\n"
                "  （单板模式不需要 --pair-frame / --pair-camera / --board-b-dir："
                "只有一块板，T_AB 按定义取 I）")
    missing = []
    if not args.left_dir:
        missing.append("--left-dir <左相机拍 A 板的图像目录>")
    if not (args.board_b_dir or args.right_dir):
        missing.append("--right-dir（或 --board-b-dir）<右相机拍 B 板的图像目录>")
    if not getattr(args, "ref_pair_left", None):
        missing.append("--ref-pair-left <参考对·左相机那张>")
    if not getattr(args, "ref_pair_right", None):
        missing.append("--ref-pair-right <参考对·右相机那张>")
    if not args.pair_frame:
        missing.append("--pair-frame <单相机**同时**拍到 A、B 两板的图或目录>")
    if not args.pair_camera:
        missing.append("--pair-camera left|right（同框图是哪一路拍的）")
    if missing:
        return ("chain 双板模式需要四段图（本轮只支持离线目录）：\n"
                "  ①② 内参段：左拍 A 板 --left-dir、右拍 B 板 --board-b-dir"
                "（板可以随便挪，只用来求内参）\n"
                "  ③ 参考对：--ref-pair-left + --ref-pair-right —— 组合装置摆好"
                "**别动**，左右相机同时各拍一张\n"
                "  ④ 同框图：--pair-frame + --pair-camera —— 一台相机同时看到"
                " A、B 两板（求两板相对位姿）\n"
                "  缺少：" + "；".join(missing) + "\n"
                "  为什么 ③ 不能省：链上两端的板位姿必须来自**同一个支架摆放**。"
                "拿 ①② 里各自'板移动中'的一帧来凑，摆放不同，外参会静默错到"
                " 10° / 28 mm 量级（2026-11-09 合成真值实测），程序不会自己报警。\n"
                "  完整用法：python apps/calibrate_stereo.py --method chain "
                "--left-dir imgs/A --board-b-dir imgs/B "
                "--ref-pair-left imgs/ref_pair_left --ref-pair-right imgs/ref_pair_right "
                "--pair-frame imgs/pair --pair-camera left --stereo-mode angled\n"
                "  （GUI 上就是「采集标定板图」的 4 步；没有同框图无法求 T_AB，"
                "本程序不会猜一个值）")
    return None


def _active_profile_name() -> str:
    """当前激活装置档案名，写进标定 ``meta.profile``（C1，内容级身份证据）。

    标定产物经 ``StereoRig.save()`` 把 meta 平铺写进 ``<out>.json``，所以这个
    字段在文件被拷贝 / 改名后仍可追溯来自哪套装置，与「输出路径跟随激活档案」
    互补：一个绑路径、一个绑内容。

    没有激活档案时返回空串：产物照旧能落盘，只是不带身份。读取侧
    （``apps/_gui_app.py`` 的 C2 自检）把「缺 profile」当老文件「未登记来源」，
    不误报、不阻断。只改代码，**不回写任何已有标定文件**。
    """
    try:
        prof = get_active()
    except Exception:                       # noqa: BLE001 —— 档案读不了也照常标定
        return ""
    return prof.name if prof is not None else ""


def _chain_meta(args, spec: BoardSpec, res, image_size) -> dict:
    """chain 落盘的 meta。**`stereo_mode` 由方法决定 = angled，不跟 auto。**

    单独成函数是为了让"落盘值"能被自测直接断言。以前这里写的是 `mode`
    （来自 `resolve_auto`）：新装置没有旧 JSON 时 auto 落到 parallel，于是
    `stereo_mode="parallel"` 与 `angle_deg=90` 被写进同一个文件 —— 下游按
    parallel 重建，`dy ≤ 2.5 px` 门限把**全部候选**拒掉且不报错。

    `rectification` 如实写标定产出的口径（chain -> `none_raw`）：以前这里
    写死 `"not_applicable"`，把真正的口径盖掉了，下游既分不清"能不能用视差"，
    也让历史文件里的写法与新代码对不上。
    """
    return {
        # C1：标定产物的内容级身份证据 —— 当前激活装置档案名。文件被拷贝/改名后
        # 仍可追溯；缺激活档案时为空串（读取侧按「未登记来源」处理，不误报）。
        "profile": _active_profile_name(),
        "board": spec.describe(),
        # 结构化规格（2026-11-19）：`board` 字符串保留兼容旧读，但下游要核对
        # 规格只能靠这三个数值字段 —— 人读串里抠数字既脆弱又不可靠。
        "board_spec": {"cols": int(spec.cols), "rows": int(spec.rows),
                       "square_mm": float(spec.square_mm)},
        "source": "dir",
        # 这条链式流程的方法名从注册表取（唯一声明 same_board 的那个），不写死。
        "method": single_board_method(),
        # 单板模式：内参段（板可移动）+ 外参段（板不动的参考图各一组），T_AB = I。
        "same_board": bool(getattr(args, "same_board", False)),
        "ref_left": str(getattr(args, "ref_left", "") or "") or None,
        "ref_right": str(getattr(args, "ref_right", "") or "") or None,
        # 双板模式的外参锚点（2026-11-09 方案 A）：链两端位姿来自这一对，
        # 而不是内参段各挑一帧。落盘留证，交接时能查这份外参怎么来的。
        "ref_pair_left": str(getattr(args, "ref_pair_left", "") or "") or None,
        "ref_pair_right": str(getattr(args, "ref_pair_right", "") or "") or None,
        "ref_pair_note": "两端位姿来自同一支架摆放的参考对",
        "rectification": res.rectification,
        "left": args.left_dir,
        "right": args.board_b_dir or args.right_dir,
        "pair_frame": args.pair_frame,
        "pair_camera": args.pair_camera,
        "image_size": list(image_size),
        "capture_size": list(image_size),
        # 模式 = 方法决定：none_raw 的产物只可能走 angled 重建。
        "stereo_mode": "angled",
        "angle_deg": float(rig_angle(res.R)),
    }


# --------------------------------------------------------------------------- #
# 标定板规格核对（2026-11-19）
# --------------------------------------------------------------------------- #
# 采集时把板规格写进 session.json，标定却只信 argv —— 采集后改了界面参数、
# 或换了一份档案，旧规格拍的图就会用新规格去解，而且**静默无报错**（真会话
# 20261118-140612 实测：8x5@25 拍的图按 11x8@9 解，检出 0/40）。
# 下面两个纯函数把"读会话规格 + 逐项数值比对 + 友好参数错误"收在一处，
# 不读文件、不跑标定，自测可以直接喂一个 session dict 验判定。
def _session_board_numbers(session: dict | None) -> dict | None:
    """从 session.json 的字典里取板规格数值；新会话优先 `board_spec`，老会话回退 `board`。

    老会话的 `board` 值是字符串（``{"cols": "8", "square": "25"}``），
    新会话另有数值型 `board_spec`；两者都在这里统一转成数值。任一项缺失/非法
    视为"这份会话没记录板规格"，由调用方按老办法用 argv，不硬拦。
    """
    if not isinstance(session, dict):
        return None
    for key, names in (("board_spec", ("cols", "rows", "square_mm")),
                       ("board", ("cols", "rows", "square"))):
        raw = session.get(key)
        if not isinstance(raw, dict):
            continue
        try:
            return {
                "cols": int(raw[names[0]]),
                "rows": int(raw[names[1]]),
                "square_mm": float(raw[names[2]]),
            }
        except (KeyError, TypeError, ValueError):
            continue
    return None


def resolve_board_spec(argv: dict, session: dict | None,
                       *, use_session_board: bool = False) -> tuple[dict | None, int]:
    """核对 argv 板规格与 chain 会话记录，返回 ``(生效规格, 错误码)``。

    - 会话无 `board` 记录（老会话）→ 不核对，用 argv 并提示"无法核对"；
    - `--use-session-board` → 用会话规格覆盖 argv，打一行说明；
    - 数值不一致 → 打印两边规格 + 两种修正方式，返回 ``(None, 2)``；
    - 一致 → 用 argv 规格。

    纯函数：只做数据比对与打印，不读文件、不跑标定，便于自测直接断言。
    """
    wanted = {
        "cols": int(argv["cols"]),
        "rows": int(argv["rows"]),
        "square_mm": float(argv["square"]),
    }
    recorded = _session_board_numbers(session)
    if recorded is None:
        print("[板规格] 这份会话没记录板规格，无法核对；按命令行 --cols/--rows/--square 继续。")
        return wanted, 0
    if use_session_board:
        print(f"[板规格] 用会话拍图时的规格（--use-session-board）："
              f"{recorded['cols']}x{recorded['rows']} 内角点 / {recorded['square_mm']:g} mm；"
              f"忽略命令行 --cols/--rows/--square。")
        return recorded, 0
    if recorded == wanted:
        return wanted, 0
    print("[板规格] 不一致，已停止标定（避免用错规格的图静默出结果）：")
    print(f"  会话拍图时：{recorded['cols']}x{recorded['rows']} 内角点 / "
          f"{recorded['square_mm']:g} mm")
    print(f"  命令行填写：{wanted['cols']}x{wanted['rows']} 内角点 / "
          f"{wanted['square_mm']:g} mm")
    print(f"  修正一：把界面参数改成会话那份（{recorded['cols']} / {recorded['rows']} / "
          f"{recorded['square_mm']:g}）后重试。")
    print("  修正二：确认要按会话规格标定，加 --use-session-board 重试。")
    return None, 2


def make_board_spec(values: dict) -> tuple[BoardSpec | None, int]:
    """构造 `BoardSpec`，越界时打印 `[参数错误] ...` 返回 ``(None, 2)`` 而非抛 traceback。"""
    try:
        spec = BoardSpec(cols=int(values["cols"]), rows=int(values["rows"]),
                         square_mm=float(values["square_mm"]))
    except ValueError as exc:
        print(f"[参数错误] {exc}")
        return None, 2
    return spec, 0


# 单板模式：一组"板不动"的参考图之间允许的位姿离散度。超过就不是"没动"了。
REF_ROT_WARN_DEG = 0.5
REF_TRANS_WARN_MM = 2.0


def _ref_board_poses(paths, spec, objp, K, D, label):
    """用一组"板不动"的参考图求 `T_cam←W`（单板模式唯一的外参锚点）。

    为什么必须单独一组图：内参那堆图里板是**移动**的，随便挑一张当公共参考，
    左右挑中的是**两个不同摆放**，此时 `T_AB ≠ I`，链出来的外参直接错。

    任何一张没**完整**检出整块棋盘就抛 `ValueError` —— OpenCV 会把局部角点当成
    整块板，位姿错得离谱却照样返回结果，这里不能让它静默通过。

    返回 `(T, info)`；`info["warn"]` 为真表示参考图之间位姿不一致（板动了/对焦变了）。
    """
    poses, rows, bad = [], [], []
    for p in paths:
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            bad.append(f"{p.name}（读不出来）")
            continue
        obs = detect_board(gray, spec, exhaustive=True)
        if not obs.found or obs.corners is None:
            bad.append(f"{p.name}（未完整检出整块 {spec.cols}x{spec.rows} 棋盘）")
            continue
        T, err = _pnp_reproj_rms(objp, obs.corners, K, D)
        poses.append(T)
        rows.append((p.name, err))
    if bad:
        raise ValueError(
            f"{label} 有图没检出**整块**棋盘：{'、'.join(bad)}\n"
            "  单板模式要求参考图里板要**整体入画**（四边都留白、不贴边、不过曝）。\n"
            "  只看到一角的角点会被 OpenCV 当成整块板，位姿错得很离谱却不报错。")
    k = int(np.argmin([e for _n, e in rows]))
    best = poses[k]
    rot = max(rotation_angle_deg(best[:3, :3], T[:3, :3]) for T in poses)
    tr = max(float(np.linalg.norm(T[:3, 3] - best[:3, 3])) * 1000.0 for T in poses)
    return best, {
        "n": len(poses), "best": rows[k][0], "err": rows[k][1],
        "rot_deg": float(rot), "trans_mm": float(tr), "rows": rows,
        "poses": poses,
        "warn": bool(rot > REF_ROT_WARN_DEG or tr > REF_TRANS_WARN_MM),
    }


def _ref_pair_poses(left_paths: list[Path], right_paths: list[Path], spec: BoardSpec,
                    objp, K_l, D_l, K_r, D_r,
                    size_l: tuple[int, int], size_r: tuple[int, int]):
    """双板模式的**参考对**：求链上两端的板位姿（2026-11-09 方案 A）。

    为什么非它不可：链 `T_camR_B · T_AB · inv(T_camL_A)` 里，两端位姿描述的是
    **同一个支架摆放**下 A、B 两块板各自在哪。设两次拍摄摆放为 `G_X/G_Y`，
    链会变成 `inv(C_R)·G_Y·inv(G_X)·C_L`，只有 `G_X = G_Y` 才等于真值。
    合成真值实测：摆放差 10°/2 cm → 旋转误差 10° / 平移 28 mm，且程序不会报警。
    内参段那些「板可移动」的图各挑一帧，几乎必然是不同摆放，所以**不能**用。

    参考对由 `capture_board.py --both` 产出：组合装置摆好别动，左右两路同时
    各拍一张，两个目录里**同名文件 = 同一时刻**，天然成对。
    这里用内参段求出的 K/D 做 PnP，保证外参与内参同一套口径。

    返回 `(T_camL_A, T_camR_B, info)`；任何一张没**完整**检出整块板就抛错
    （OpenCV 会把局部角点当成整块板，位姿错得离谱却不报错）。
    """
    by_l = {p.name: p for p in left_paths}
    by_r = {p.name: p for p in right_paths}
    common = sorted(set(by_l) & set(by_r))
    if not common and len(left_paths) == len(right_paths) == 1:
        # 手工各放一张、文件名不同也算一对：小白不该被文件名卡住。
        pairs = [(left_paths[0], right_paths[0])]
        pairing = "按唯一一张自动配对"
    else:
        pairs = [(by_l[n], by_r[n]) for n in common]
        pairing = "按同名文件配对"
    if not pairs:
        raise ValueError(
            "参考对左右目录里**没有成对**的图（左 "
            f"{len(left_paths)} 张 / 右 {len(right_paths)} 张，同名 {len(common)} 对）。\n"
            "  第 ③ 步要用「采集标定板图」里的两路同拍：左右各存一张、文件名相同。\n"
            "  手工整理时请把同一时刻的两张图取同一个名字，或各只放一张。")

    rows, bad = [], []
    for pl, pr in pairs:
        gl = cv2.imread(str(pl), cv2.IMREAD_GRAYSCALE)
        gr = cv2.imread(str(pr), cv2.IMREAD_GRAYSCALE)
        if gl is None or gr is None:
            bad.append(f"{pl.name}/{pr.name}（读不出来）")
            continue
        if (gl.shape[1], gl.shape[0]) != size_l or (gr.shape[1], gr.shape[0]) != size_r:
            bad.append(f"{pl.name}/{pr.name}（尺寸与内参段不一致："
                       f"{gl.shape[1]}x{gl.shape[0]} / {gr.shape[1]}x{gr.shape[0]}，"
                       f"应为 {size_l[0]}x{size_l[1]} / {size_r[0]}x{size_r[1]}）")
            continue
        obs_l = detect_board(gl, spec, exhaustive=True)
        obs_r = detect_board(gr, spec, exhaustive=True)
        if not (obs_l.found and obs_l.corners is not None
                and obs_r.found and obs_r.corners is not None):
            bad.append(f"{pl.name}/{pr.name}（有一路没完整检出整块 "
                       f"{spec.cols}x{spec.rows} 棋盘）")
            continue
        T_l, e_l = _pnp_reproj_rms(objp, obs_l.corners, K_l, D_l)
        T_r, e_r = _pnp_reproj_rms(objp, obs_r.corners, K_r, D_r)
        rows.append((pl.name, T_l, T_r, e_l, e_r))
    if bad:
        raise ValueError(
            "参考对有图不能用：" + "、".join(bad[:5])
            + ("…" if len(bad) > 5 else "") + "\n"
            "  第 ③ 步要求：组合装置摆好**别动**，左相机完整看到 A 板、"
            "右相机完整看到 B 板，两路同时各拍一张；采集档要与内参段一致。")
    if not rows:
        raise ValueError("参考对目录里没有可用的成对图（第 ③ 步没拍）。")

    # 多对时看**同一路**跨对的位姿离散：那就是「板在拍参考对时动没动」。
    # 两路看的是不同的板，不能像单板模式那样做跨路比较。
    base_l, base_r = rows[0][1], rows[0][2]
    rot = max(max(rotation_angle_deg(T_l[:3, :3], base_l[:3, :3]),
                  rotation_angle_deg(T_r[:3, :3], base_r[:3, :3]))
              for _n, T_l, T_r, _e_l, _e_r in rows)
    tr = max(max(float(np.linalg.norm(T_l[:3, 3] - base_l[:3, 3])),
                 float(np.linalg.norm(T_r[:3, 3] - base_r[:3, 3]))) * 1000.0
             for _n, T_l, T_r, _e_l, _e_r in rows)
    best = min(rows, key=lambda r: r[3] + r[4])
    return best[1], best[2], {
        "n": len(rows), "best": best[0], "err_l": best[3], "err_r": best[4],
        "pairing": pairing, "rot_deg": float(rot), "trans_mm": float(tr),
        "rows": [(n, e_l, e_r) for n, _a, _b, e_l, e_r in rows],
        "warn": bool(rot > REF_ROT_WARN_DEG or tr > REF_TRANS_WARN_MM),
        "unpaired": max(0, len(left_paths) - len(pairs))
                    + max(0, len(right_paths) - len(pairs)),
    }


def _same_board_samples(info_l, info_r) -> list[np.ndarray]:
    """单板的 `T_AB` 重复性样本：**逐张**参考图各算一次 `T_{B←A}`（≈I）。

    为什么可以这么算：所有参考图都是"同一块板、同一个摆放"，
    所以第 i 张左右参考图组成的 `T_camR_W_i · inv(T_camL_W_i)` 就是一次独立的
    `T_AB` 测量，其真值恒为 `I`；它的散布正是"板有没有动 / 对焦有没有变"的证据。
    核心层的 `consistency()` 拿到 ≥2 个样本才会评估重复性 —— 这样单板模式
    也能拿到真实的 n，而不是被当成"只测了 1 次"。

    注意配对方式：按**文件顺序** zip（第 i 张左 ↔ 第 i 张右），两侧张数不等时
    取 `min(n_l, n_r)`。这里**不要求**左右是同一时刻拍的对子 —— 板不动，任一张
    左参考图和任一张右参考图都描述同一个物理摆放，所以任意配对都成立；
    配对只影响"样本之间是否独立"，不影响正确性。
    """
    m = min(info_l["n"], info_r["n"])
    if m < 2:
        return [np.eye(4)]
    return [compose(Tr, invert(Tl))
            for Tl, Tr in zip(info_l["poses"][:m], info_r["poses"][:m])]


def _ref_stability_text(info: dict) -> str:
    """参考图的"离散度"一句话；n<2 时必须说**未评估**。

    n=1 时 info 里的 0.000°/0.00 mm 是"没算过"的默认值，印出来会被读成
    "测过而且很稳" —— 这两种含义在报告里绝不能混。
    """
    if info["n"] < 2:
        return "离散度 未评估（n<2，只有 1 张参考图，无法评估重复性）"
    return (f"离散度 旋转 {info['rot_deg']:.3f}° / "
            f"平移 {info['trans_mm']:.2f} mm")


def _chain_report(res, same_board: bool, ref_note: str = "",
                  ref_pair_note: str = "") -> str:
    """chain 的报告文本。

    单板模式下 `T_AB` 按定义是 `I`，报告里"两板间距 / T_AB 重复性"这两行
    既没有测量对象也没有意义（会印出 `0.00 mm`，读起来像"两板贴在一起"），
    所以在这里如实换成 `T_AB = I（单板）`。同理"T_AB 只测了 1 次 → 建议拍
    3~5 张同框图"也一并抹掉：单板模式压根不许给 --pair-frame，照着做会撞互斥。
    核心层报告不改 —— 双板模式的那两行是它唯一的自查证据。
    """
    text = res.report()
    if not same_board:
        # 双板：把外参锚点的来源写进报告（落日志/交接时能一眼看出这份外参
        # 是不是用了同一摆放的参考对，而不是内参段里随手挑的两帧）。
        return (text + "\n外参锚点    : " + (ref_pair_note or "—（参考对信息缺失）")
                + "\n前提        : A/B 刚性固连，参考对时装置一动不动"
                  "（程序无法从图里校验，由拍摄者保证）")
    out = []
    for line in text.splitlines():
        if "两板间距" in line or "--pair-frame" in line:
            continue
        if "T_AB 只测了 1 次" in line:
            # 单板没有同框图可补，改写成本模式能照做的建议
            line = ("  ! T_AB 重复性：参考图只有 1 张（n=1），**未评估重复性**；"
                    "建议 --ref-left/--ref-right 各再拍 1~4 张（板一动不动）")
        if line.strip().startswith("T_AB 重复性") and "参考图只有 1 张" not in line:
            # 核心层的 n/离散度就是逐张参考图算的 T_AB，语义对，只补一句来源
            line = line + "（由参考图逐张求 T_AB 得到）"
        if "刚性双板公共参考" in line:
            line = "双目标定结果（chain：独立单目 + 单块固定板）"
        out.append(line)
        if line.startswith("右目重投影 RMS"):
            out.append("T_AB          : I（单板；" + (ref_note or "—") + "）")
    return "\n".join(out)


def _bypass_calibration_out(out: str) -> str:
    """把输出前缀改到**旁路路径**：只改 stem，正式激活档案绝不改动。

    ``StereoRig.save()`` 会按这个前缀同时产出 ``.npz`` 与 ``.json``，
    两个文件一起走旁路；``_backup_existing`` 也只检查旁路路径，
    所以正式的 ``--out`` 完全不会被碰。时间戳带毫秒，避免同秒多次运行撞名。
    """
    p = Path(out)
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
    return str(p.with_name(f"{p.stem}_unusable_{stamp}{p.suffix}"))


def _backup_before_force(out: str | Path) -> list[Path]:
    """强制覆盖前，把旧标定 `.npz`/`.json` 复制到**同目录**的新备份文件。

    命名 ``<stem>_before_force_<YYYYMMDD-HHMMSS-mmm>.<ext>``，带毫秒并查重，
    **绝不覆盖/删除任何已有备份**（含 config/ 下那些历史名字）。两份都在才写；
    任何一步复制失败都抛异常，由调用方决定"不写正式路径"（保护旧标定）。
    """
    p = Path(out)
    npz, js = p.with_suffix(".npz"), p.with_suffix(".json")
    pairs = [(q, q.suffix) for q in (npz, js) if q.is_file()]
    if not pairs:
        return []
    stamp = (time.strftime("%Y%m%d-%H%M%S")
             + f"-{int(time.time() * 1000) % 1000:03d}")
    dests: list[Path] = []
    for src, ext in pairs:
        dest = p.with_name(f"{p.stem}_before_force_{stamp}{ext}")
        n = 1
        while dest.exists():                 # 撞名就换号，绝不覆盖已有备份
            dest = p.with_name(f"{p.stem}_before_force_{stamp}_{n}{ext}")
            n += 1
        shutil.copy2(src, dest)
        dests.append(dest)
    # 全部复制成功才算成功（中途抛异常由调用方兜底）
    for dest in dests:
        if not dest.is_file() or dest.stat().st_size == 0:
            raise OSError(f"备份文件校验失败：{dest}")
    return dests


def finalize_and_save(rig: StereoRig, args, *, unusable: bool,
                      log: logging.Logger) -> tuple[Path, Path, bool]:
    """统一落盘：质量差**默认旁路保存**；``--force-save`` 显式写正式路径。

    新原则「质量差也要出结果、只有错误才紧急」：``unusable`` 时不再中止，
    而是照常生成可查看、可复盘的产物，并用 ``forced_unusable`` / ``verdict``
    自证「仅链路测试，不得用于数据结论」。默认（不加 ``--force-save``）
    把 ``args.out`` 改到旁路路径，**激活装置档案绝不改动**；只有显式
    ``--force-save`` 才写正式 ``--out`` 路径。

    所有 ``not res.ok`` / angled "不可用" 的保存都必须走这里；调用方必须
    先给 ``StereoRig.from_result(..., allow_unusable=True)`` —— params.py 的
    第二道闸门（``if not res.ok: raise``）保留不动，不许静默放宽。

    返回 ``(npz, json, 是否旁路)``。
    """
    force_save = bool(getattr(args, "force_save", False))
    bypassed = False
    # A4（2026-11-18）：强制写正式路径前，先把旧 `.npz`/`.json` 复制到同目录的
    # 新备份文件（带时间戳、不覆盖任何已有备份）。**只有真的复制成功才继续写**；
    # 复制失败就打印警告并直接返回，绝不写正式路径 —— 保护那份不可再生的旧标定。
    # 放在 verdict 文案之前，避免备份失败时还打印"即将写入正式路径"误导人。
    if force_save:
        try:
            backups = _backup_before_force(args.out)
            if backups:
                print(f"  强制覆盖前已备份旧标定："
                      f"{', '.join(str(b) for b in backups)}")
                log.info("强制保存前备份旧标定：%s", [str(b) for b in backups])
        except Exception as exc:                     # noqa: BLE001
            npz_p = Path(str(args.out)).with_suffix(".npz")
            js_p = Path(str(args.out)).with_suffix(".json")
            msg = (f"强制保存前备份旧标定失败（{type(exc).__name__}: {exc}）；"
                   f"为保护旧标定，本次不写正式路径 {args.out}")
            print(f"\n[错误] {msg}")
            log.error(msg)
            rig.quality["force_backup_failed"] = True
            rig.quality.setdefault("warnings", []).append(msg)
            return npz_p, js_p, True
    if unusable:
        verdict = str(rig.quality.get("verdict") or "不可用")
        # 产物自证：forced_unusable 可机读，verdict 追加人话；grade 保持判定原值。
        rig.quality["forced_unusable"] = True
        rig.quality["verdict"] = f"{verdict} 【仅链路测试，不得用于数据结论】"
        if force_save:
            print("\n" + "!" * 66)
            print("⚠ 用户强制保存不可用标定（仅链路测试，写入正式 --out 路径）")
            print(f"  判定：{verdict}")
            print("  这份参数不得用于任何数据结论；下游据此应拒绝出数。")
            print("!" * 66 + "\n")
            log.warning("标定质量不合格，给了 --force-save：写入正式路径（仅链路测试）")
        else:
            args.out = _bypass_calibration_out(str(args.out))
            bypassed = True
            print("\n" + "!" * 66)
            print("⚠ 标定质量不合格（仅质量差，非程序错误，不中止）")
            print(f"  结果另存为旁路文件供检查：{args.out}")
            print("  激活装置档案未改动（正式标定路径保持原样）。")
            print(f"  判定：{verdict}")
            print("!" * 66 + "\n")
            log.warning("标定质量不合格：另存旁路 %s（正式档案未改动）", args.out)
    npz, js = rig.save(args.out)
    return npz, js, bypassed


def _run_chain(args, spec: BoardSpec, log: logging.Logger) -> int:
    """chain 的离线流程：三份图 → 单目内参/位姿 → T_AB → 链式外参。

    `--same-board` 时只读两份目录，`T_AB = I`：板没动，A/B 是同一块板的
    同一个世界系，链式外参退化成 `T_{C_R←W} · T_{W←C_L}`。
    """
    same_board = bool(getattr(args, "same_board", False))
    problem = _chain_input_error(args)
    if problem:
        print(f"\n[错误] {problem}")
        log.error("chain 输入不全：%s", problem.replace("\n", " "))
        return 1

    from calib import relative_pose

    left_paths = _list_images(Path(args.left_dir))
    right_paths = _list_images(Path(args.board_b_dir or args.right_dir))
    right_label = "--right-dir" if same_board else "B 板目录"
    pair_paths = [] if same_board else _list_images(Path(args.pair_frame))
    ref_left_paths = _list_images(Path(args.ref_left)) if same_board else []
    ref_right_paths = _list_images(Path(args.ref_right)) if same_board else []
    # 双板模式的参考对（方案 A）：`_chain_input_error` 已保证两个参数都在，
    # 这里仍判一次空值，避免 Path("") 变成当前目录把无关图片全扫进来。
    ref_pair_l_paths = (_list_images(Path(args.ref_pair_left))
                        if not same_board and getattr(args, "ref_pair_left", None) else [])
    ref_pair_r_paths = (_list_images(Path(args.ref_pair_right))
                        if not same_board and getattr(args, "ref_pair_right", None) else [])
    required = [("--left-dir", left_paths), (right_label, right_paths)]
    if same_board:
        # 外参段：两路各一组"板不动"的参考图
        required += [("--ref-left", ref_left_paths),
                     ("--ref-right", ref_right_paths)]
    else:
        required += [("--ref-pair-left", ref_pair_l_paths),
                     ("--ref-pair-right", ref_pair_r_paths),
                     ("--pair-frame", pair_paths)]
    for label, paths in required:
        if not paths:
            print(f"\n[错误] {label} 里没有图像（支持 "
                  f"{'/'.join(sorted(_IMAGE_SUFFIXES))}）。")
            log.error("chain：%s 无图", label)
            return 1

    def _read_corners(paths: list[Path], label: str):
        corners, size, failed = [], None, 0
        for p in paths:
            gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                failed += 1
                continue
            if size is None:
                size = (gray.shape[1], gray.shape[0])
            elif (gray.shape[1], gray.shape[0]) != size:
                print(f"\n[错误] {label} 内图像尺寸不一致：{p.name} 是 "
                      f"{gray.shape[1]}x{gray.shape[0]}，之前是 {size[0]}x{size[1]}。")
                log.error("chain：%s 尺寸不一致", label)
                return None, None, 0
            obs = detect_board(gray, spec, exhaustive=True)
            if obs.found and obs.corners is not None:
                corners.append(obs.corners.reshape(-1, 2).astype(np.float32))
            else:
                failed += 1
        return corners, size, failed

    c_left, size_left, fail_l = _read_corners(left_paths, "--left-dir")
    c_right, size_right, fail_r = _read_corners(right_paths, right_label)
    if c_left is None or c_right is None:
        return 1
    print(f"[chain] 检出：左 {len(c_left)}/{len(left_paths)}（{fail_l} 张未检出），"
          f"右 {len(c_right)}/{len(right_paths)}（{fail_r} 张未检出）；"
          f"尺寸 左 {size_left[0]}x{size_left[1]} / 右 {size_right[0]}x{size_right[1]}")
    if not same_board:
        # 前提必须打在**结果之前**：这份外参的正确性完全押在"参考对是同一个
        # 支架摆放"上。程序无法从图里验证 A/B 是否刚性固连、装置有没有被挪过，
        # 所以必须先把这句话说清楚，而不是等出了错数再回头猜。
        print("[chain] 双板模式：①② 只求内参；③ 参考对定外参锚点；④ 同框图求 T_AB")
        print(f"        参考对：左 {len(ref_pair_l_paths)} 张 / "
              f"右 {len(ref_pair_r_paths)} 张（组合装置摆好别动、两路同时拍）")
        print("        前提：A/B 两板必须**刚性固连在同一支架**上，且拍参考对时"
              "支架一动不动。板动 1 mm，外参就错 1 mm —— 这条程序无法替你校验。")
        log.info("chain 双板模式：外参锚点来自参考对（同一摆放）")
    if same_board:
        # 前提必须打在**结果之前**：这份外参的正确性完全押在"板没动"上，
        # 事后才发现这块板被动过，全部数字都得重来。
        print("[chain] 单板模式：T_AB = I（同一块固定板）")
        print("        内参段：--left-dir / --right-dir（板可移动，各 ≥15 姿态）")
        print(f"        外参段：--ref-left {len(ref_left_paths)} 张 / "
              f"--ref-right {len(ref_right_paths)} 张（板固定不动）")
        print("        前提：两段之间、以及左右参考图之间**板必须一动不动**。"
              "板动 1 mm，外参就错 1 mm。")
        print("        每路都要**完整**检出整块棋盘（OpenCV 不认局部角点）。")
        log.warning("chain 单板模式：T_AB=I，前提是参考图期间板未移动")
    if len(c_left) < 3 or len(c_right) < 3:
        print("\n[错误] chain 每路至少需要 3 个姿态才能标内参；"
              "建议各 ≥15 个并覆盖画面四角。")
        log.error("chain：姿态太少 左%d 右%d", len(c_left), len(c_right))
        return 1

    cfg_mono = CalibConfig(rational_model=bool(getattr(args, "rational_model", False)),
                           fix_principal_point=bool(getattr(args, "fix_principal_point", False)))
    objp = spec.object_points()
    try:
        mono_l = calibrate_mono_points(objp, c_left, size_left, cfg_mono)
        mono_r = calibrate_mono_points(objp, c_right, size_right, cfg_mono)
    except Exception as exc:                          # noqa: BLE001
        print(f"\n[错误] 单目标定失败：{type(exc).__name__}: {exc}")
        log.error("chain 单目标定失败：%s", exc)
        return 1
    print(f"[chain] 单目 RMS：左 {mono_l['rms']:.3f} px / 右 {mono_r['rms']:.3f} px")

    # 跨路一致性的输入：单板模式下两路看的是**同一块板的同一位置**，逐张参考图
    # 配成 (T_camL_B, T_camR_B) 交给核心层做"搬坐标系看是否重合"的检查。
    # 双板模式没有这个前提（左看 A、右看 B），保持 None —— 核心层会如实写"无法计算"。
    cross_poses = None
    # 双板 180° 纠错台账（单板模式不适用，保持空表）：由
    # `_resolve_pair_samples` 产出，随后写进 `quality["t_ab_corrections"]` 落盘 ——
    # 过几天回看时能知道这组 T_AB 被纠正过什么、凭什么（否则和"没纠过"长得一样）。
    t_ab_corrections: list[dict] = []
    t_ab_before: list[np.ndarray] = []
    # 第四层（装置先验复核）结果：只对双板模式有意义，单板保持 None。
    pair_prior: dict | None = None
    ref_pair_note = ""
    if same_board:
        # 单板：A 与 B 是同一块板的同一个世界系，T_AB 按定义就是单位阵 ——
        # 它不是"估"出来的，也不该被当成可优化的量。
        samples, pair_rows = [np.eye(4)], []
        ref_note = ""
        try:
            T_ref_l, info_l = _ref_board_poses(ref_left_paths, spec, objp,
                                               mono_l["K"], mono_l["D"], "--ref-left")
            T_ref_r, info_r = _ref_board_poses(ref_right_paths, spec, objp,
                                               mono_r["K"], mono_r["D"], "--ref-right")
        except ValueError as exc:
            print(f"\n[错误] {exc}")
            log.error("chain 单板参考图不可用：%s", exc)
            return 1
        for lbl, info in (("--ref-left", info_l), ("--ref-right", info_r)):
            print(f"[chain] {lbl} 参考位姿：n={info['n']}（取 {info['best']}，"
                  f"重投影 {info['err']:.2f} px）  {_ref_stability_text(info)}")
            if info["warn"]:
                print(f"[警告] 参考图之间位姿不一致 → 板在拍参考图时动了或对焦变了"
                      f"（{lbl}：旋转 {info['rot_deg']:.2f}° / 平移 {info['trans_mm']:.2f} mm，"
                      f"门限 {REF_ROT_WARN_DEG}° / {REF_TRANS_WARN_MM} mm）。外参按这张算，"
                      "但请重拍参考图后再定稿。")
                log.warning("chain 单板：%s 参考图位姿离散 旋转 %.2f° 平移 %.2f mm",
                            lbl, info["rot_deg"], info["trans_mm"])
        T_camL_A, T_camR_B = T_ref_l, T_ref_r
        ref_note = f"参考图 左 {info_l['n']} / 右 {info_r['n']}"
        # 重复性样本：逐张参考图各算一次 T_AB（真值恒为 I）。
        # 参考图 ≥2 张时核心层就能拿到真实 n，单板模式也能到「可用」以上；
        # 只有 1 张时如实退回 [I]（n=1，未评估重复性），绝不伪造样本。
        samples = _same_board_samples(info_l, info_r)
        # 跨路一致性：逐张参考图配对（同一物理摆放 -> 可逐对比较）。
        # 张数不等时取 min（与 `_same_board_samples` 同一配对口径）。
        m = min(int(info_l["n"]), int(info_r["n"]))
        if m >= 1:
            cross_poses = list(zip(info_l["poses"][:m], info_r["poses"][:m]))
        print(f"[chain] T_AB：I（单板；{ref_note}）  "
              f"重复性 n={len(samples)}"
              + ("（n=1，未评估重复性）" if len(samples) < 2 else ""))
    else:
        # 参考对（第 ③ 步）**先算**：它既是链上的外参锚点，也是判 A/B 的
        # "同一块板的参考姿态"之一。挪到同框图之前不改变语义，只让判 A/B 有
        # 两份独立证据（内参段姿态簇 + 参考对锚点），而不是只靠那簇被移动过的板。
        try:
            T_camL_A, T_camR_B, rp_info = _ref_pair_poses(
                ref_pair_l_paths, ref_pair_r_paths, spec, objp,
                mono_l["K"], mono_l["D"], mono_r["K"], mono_r["D"],
                size_left, size_right)
        except ValueError as exc:
            print(f"\n[错误] 参考对（第 ③ 步）不可用：{exc}")
            log.error("chain 双板参考对不可用：%s", exc)
            return 1
        print(f"[chain] 外参锚点：参考对 {rp_info['n']} 对"
              f"（{rp_info['pairing']}），取 {rp_info['best']}，"
              f"重投影 左 {rp_info['err_l']:.2f} / 右 {rp_info['err_r']:.2f} px")
        if rp_info["n"] >= 2:
            print(f"        参考对离散度：旋转 {rp_info['rot_deg']:.3f}° / "
                  f"平移 {rp_info['trans_mm']:.2f} mm"
                  + ("  ⚠ 装置在拍参考对时动过，请重拍第 ③ 步"
                     if rp_info["warn"] else "（装置没动）"))
        else:
            print("        只有 1 对，未评估重复性（多拍 1~2 对更稳，非必需）")
        if rp_info["unpaired"]:
            print(f"        [警告] 有 {rp_info['unpaired']} 张落单的图没法配对，已忽略")
        ref_pair_note = (f"参考对 {rp_info['n']} 对，取 {rp_info['best']}，"
                         f"重投影 {rp_info['err_l']:.2f}/{rp_info['err_r']:.2f} px")

        # 同框图的参考姿态簇：--pair-camera 那一路目录里的板姿态簇（同框相机
        # 自己那块板）+ 参考对里同一块板的姿态。另一块板的参考位姿在**另一台
        # 相机**的坐标系里，直接比旋转不合法（见 `_resolve_pair_samples`）。
        ref_mono = mono_l if args.pair_camera == "left" else mono_r
        ref_rots = [cv2.Rodrigues(r)[0] for r in ref_mono["rvecs"]]
        own_anchor = T_camL_A if args.pair_camera == "left" else T_camR_B
        ref_rots.append(own_anchor[:3, :3].copy())
        K_pair, D_pair = ref_mono["K"], ref_mono["D"]
        pair_cam_size = size_left if args.pair_camera == "left" else size_right

        frames = []
        for p in pair_paths:
            gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            if (gray.shape[1], gray.shape[0]) != pair_cam_size:
                print(f"\n[错误] 同框图 {p.name} 尺寸 {gray.shape[1]}x{gray.shape[0]} "
                      f"与 --pair-camera {args.pair_camera} 的标定尺寸 {pair_cam_size} 不一致。")
                log.error("chain：同框图尺寸不一致 %s", p)
                return 1
            found = _detect_two_boards(gray, spec)
            if found is None:
                print(f"  [跳过] {p.name}：没有同时检出两块板")
                continue
            T1, e1 = _pnp_reproj_rms(objp, found[0].corners, K_pair, D_pair)
            T2, e2 = _pnp_reproj_rms(objp, found[1].corners, K_pair, D_pair)
            frames.append({"name": p.name, "poses": [T1, T2], "errs": [e1, e2]})

        if not frames:
            print("\n[错误] --pair-frame 里没有一张图能**同时**检出 A、B 两块板。\n"
                  "  下一步：把双板靶标放到 --pair-camera 那一路的画面中央、光照均匀、"
                  "两块板都完整入画，重拍 3~5 张。")
            log.error("chain：没有可用的同框图")
            return 1
        try:
            samples, pair_rows, t_ab_corrections, t_ab_before = _resolve_pair_samples(
                frames, ref_rots, spec, args.pair_camera)
        except ValueError as exc:
            print(f"\n[错误] --pair-frame 里无法判定 A/B 板：{exc}")
            log.error("chain：A/B 分配失败 %s", exc)
            return 1
        for row in pair_rows:
            tags = []
            if row["swap"]:
                tags.append("A/B 判反（逐帧离群）→已纠正")
            elif row["ab_fixed"]:
                tags.append("A/B 判反→已纠正")
            if row["flip_a"]:
                tags.append("A 板框 180°→已纠正")
            if row["flip_b"]:
                tags.append("B 板框 180°→已纠正")
            print(f"  同框图 {row['name']}: 重投影 {row['errs'][0]:.2f}/"
                  f"{row['errs'][1]:.2f} px，到参考姿态簇 "
                  f"{row['d_a']:.1f}°/{row['d_b']:.1f}°"
                  + (f"  [{'、'.join(tags)}]" if tags else ""))
        if len(t_ab_before) >= 2:
            cb = relative_pose.consistency(t_ab_before)
            print(f"[chain] T_AB 180° 纠错前：旋转最大偏离 {cb['rot_max_deg']:.3f}°  "
                  f"平移最大偏离 {cb['trans_max_mm']:.3f} mm（逐帧各自挑最近参考簇）")
        n_flip = sum(1 for c in t_ab_corrections if c["action"].startswith("板框"))
        n_swap = sum(1 for c in t_ab_corrections if c["action"] == "A/B 判反")
        detail = "；".join(f"{c['file']} {c['board']} {c['action']}（{c['detail']}）"
                          for c in t_ab_corrections)
        print(f"[chain] T_AB 180° 纠错：板框等价解 {n_flip} 处 / A/B 判反 {n_swap} 处"
              + (f"；{detail}" if detail else "（无需纠正）"))
        cons = relative_pose.consistency(samples)
        print(f"[chain] T_AB：n={cons['n']}  旋转最大偏离 {cons['rot_max_deg']:.3f}°  "
              f"平移最大偏离 {cons['trans_max_mm']:.3f} mm  两板间距 "
              f"{cons['baseline_mm']:.1f} mm")
        # ---- 第四层：装置先验复核（三层全过也照判的错解）----
        pri_b, pri_a = _device_pair_priors(args)
        pair_prior = check_pair_prior_consistency(
            samples, prior_baseline_mm=pri_b, prior_angle_deg=pri_a)
        print(f"[chain] T_AB 装置先验复核：基线实测 {pair_prior['baseline_mm']:.1f} mm"
              + (f" / 先验 {pri_b:.1f} mm（偏离 {pair_prior['baseline_rel_err'] * 100:.1f}%）"
                 if pri_b is not None else " / 先验缺省")
              + f"；相对角实测 {pair_prior['angle_deg']:.1f}°"
              + (f" / 先验 {pri_a:.1f}°" if pri_a is not None else " / 先验缺省")
              + f" -> {'通过' if pair_prior['ok'] else '错解：' + '；'.join(pair_prior['reasons'])}")

    # 链上用的两路板位姿（2026-11-09 方案 A）。
    # 双板：**只从参考对取**。以前这里写的是"A/B 板框随板走，各挑重投影最小的
    # 一帧不影响链式外参" —— 那句是错的。链 = inv(C_R)·G_Y·inv(G_X)·C_L，
    # 两端位姿取自不同摆放时多出 G_Y·inv(G_X) 一项（合成真值实测：差 10°/2 cm
    # → 旋转 10° / 平移 28 mm），而双板模式 cross_poses=None、GUI 不传
    # --baseline-mm，完全不会报警。内参段那些移动中的图只用来求 K/D。
    # 单板：参考图本来就是固定板，上面已单独算好，这里不能再覆盖。
    # （双板的外参锚点参考对已在上面的 else 分支里先算好了，此处不再重复。）

    # chain 的产物是 none_raw（未做极线校正）的链式外参，只可能用 angled 重建。
    # 以前这里跟 auto 走：新装置没有旧 JSON 时 auto 落到 parallel，于是把
    # meta["stereo_mode"] 写成 parallel，而本函数下面已算出 angle_deg=90 ——
    # 下游按 parallel 跑，dy≤2.5px 门限把**全部候选**拒掉且不报错。
    mode = "angled"
    print("[模式] chain → stereo_mode=angled（由方法决定，不跟 auto）")
    log.info("chain -> stereo_mode=angled（由方法决定，不跟 auto）")
    cfg = CalibConfig(method=args.method, expected_baseline_mm=args.baseline_mm,
                      rational_model=bool(getattr(args, "rational_model", False)),
                      fix_principal_point=bool(getattr(args, "fix_principal_point", False)))
    try:
        res = calibrate_stereo(
            objp, c_left, c_right, size_left, cfg, spec, stereo_mode=mode,
            T_camL_A=T_camL_A, T_AB=np.eye(4) if same_board else samples[0],
            T_camR_B=T_camR_B, T_AB_samples=samples, cross_poses=cross_poses)
    except ValueError as exc:
        print(f"\n[错误] {exc}")
        log.error("chain 标定被拒绝：%s", exc)
        return 1

    report = _chain_report(res, same_board, ref_note if same_board else "",
                          ref_pair_note)
    print(report)
    log.info("chain 结果：\n%s", report)
    if same_board and rig_angle(res.R) > 60.0:
        print("[警告] 注意：单板模式在 >60° 的装置上要求两路各自都能看到**整块**板；"
              "请确认每路检出的是完整棋盘，而不是只看到一角。")
        log.warning("单板模式 + 夹角 %.1f° > 60°：需确认每路检出完整棋盘", rig_angle(res.R))
    # 新原则：质量差也要出结果、只有错误才紧急。不可用时不再 return 1，
    # 而是继续构造 rig —— 但必须显式给 allow_unusable=True（params.py 第二道
    # 闸门保留）。落盘一律走 finalize_and_save：默认旁路（不污染激活档案），
    # 只有显式 --force-save 才写正式 --out 路径，两条轨道都带「仅链路测试」标记。
    rig = StereoRig.from_result(res, meta=_chain_meta(args, spec, res, size_left),
                                allow_unusable=not res.ok)
    rig.quality["method"] = res.method
    rig.quality["rectification"] = res.rectification
    rig.quality["consistency"] = res.consistency
    if not same_board:
        rig.quality["t_ab_mm"] = res.consistency.get("baseline_mm", float("nan"))
        # 180° 等价解纠错台账：哪张图、哪块板、原解/翻转解、修正前后散布。
        # 不落盘的话，这组"被纠正过"的 T_AB 和"一次就测对"的落盘文件长得一样。
        rig.quality["t_ab_corrections"] = {
            "n": len(t_ab_corrections),
            "flips": int(sum(1 for c in t_ab_corrections
                             if c["action"].startswith("板框"))),
            "ab_swaps": int(sum(1 for c in t_ab_corrections
                                if c["action"] == "A/B 判反")),
            "items": t_ab_corrections,
        }
    else:
        # 参考图的稳定性必须**落盘**。只打印警告的话，过几天回看，
        # 这份标定和"参考图确实没动"的那份长得一模一样——而外参的可信度不同。
        def _ref_stab(info: dict) -> dict:
            if info["n"] < 2:
                return {"n": info["n"], "rot_deg": None, "trans_mm": None,
                        "note": "n<2：未评估重复性"}
            return {"n": info["n"], "rot_deg": float(info["rot_deg"]),
                    "trans_mm": float(info["trans_mm"])}

        rig.quality["same_board_ref"] = {
            "left": _ref_stab(info_l),
            "right": _ref_stab(info_r),
            "warn": bool(info_l["warn"] or info_r["warn"]),
            "warn_threshold": {"rot_deg": REF_ROT_WARN_DEG,
                               "trans_mm": REF_TRANS_WARN_MM},
            "best": [str(info_l["best"]), str(info_r["best"])],
        }
        if rig.quality["same_board_ref"]["warn"]:
            rig.quality["same_board_ref"]["note"] = (
                "参考图之间位姿不一致：板在拍参考图时动了或对焦变了；"
                "外参按其中一张算，建议重拍参考图后再定稿")
    # 90° 下 F 极线指标需要"同名点"，而 chain 的左右角点**不是对应点**
    # （各自看各自的板），所以这里不做 angled 质量评估，并写清原因。
    rig.quality["angled_grade"] = "未评估"
    # verdict 先落原始判定；不可用时的「仅链路测试」标记由 finalize_and_save 统一加。
    rig.quality["verdict"] = res.verdict
    # 第四层错解：不改样本、不 raise、不改退出码，只把判定写成不可用并落 warnings，
    # 由 finalize_and_save 统一追加「仅链路测试」并走旁路（显式 --force-save 才写正式）。
    prior_bad = bool(pair_prior is not None and not pair_prior["ok"])
    if pair_prior is not None:
        rig.quality["t_ab_prior_guard"] = pair_prior
    if prior_bad:
        reason_text = "；".join(pair_prior["reasons"])
        rig.quality["verdict"] = (
            f"{rig.quality['verdict']} 【装置先验复核判错解：{reason_text}】")
        rig.quality.setdefault("warnings", []).append(
            "T_AB 装置先验复核判错解：" + reason_text)
        rig.quality["grade"] = "不可用"
        log.warning("chain T_AB 装置先验复核判错解：%s", reason_text)
    print(f"[chain] 实测光轴夹角 {rig.angle_deg:.2f}°（由链式 R 计算，不写死 90°）")
    if rig.rectification == "none_raw":
        # 必须在报告里明说：这份参数**不能**拿去喂视差逻辑。
        # 不说的话，下游会看到 R1/R2/P1/P2 齐全就以为"已校正"，
        # 而 none_raw 下 y 差/视差都没有几何意义（见 calib/params.py 的说明）。
        print(f"[chain] {NONE_RAW_NOTE}")
        log.warning("chain 产物为 none_raw：%s", NONE_RAW_NOTE)
    _unusable = (not res.ok) or prior_bad
    npz, js, _bypassed = finalize_and_save(rig, args, unusable=_unusable, log=log)
    _saved = Path(str(npz)).is_file() if npz else False
    print(f"已保存：{npz}" if _saved else f"未写入正式路径（旧标定已保护）：{npz}")
    print(f"        {js}")
    print("注意：chain 的 R1/R2=I、P1/P2 是「只去畸变」的投影矩阵（未做极线校正），"
          "三角测量可直接用；但校正图像/视差接口不适用。")
    return 0


# 交叉验证的最少输入对数：4 折每折至少 3 对训练/1 对测试才有意义；
# 12 对是"每折 9 训 3 测"的下限，再少就只是在测噪声。
CV_MIN_PAIRS = 12


def _cross_validate_joint(objp, ip1, ip2, size, cfg: CalibConfig, spec: BoardSpec,
                          mode: str, k: int = 4) -> dict:
    """joint 的 k 折交叉验证：每折用 k-1 折重标，剩下 1 折**只打分**。

    ## 为什么只能对 joint 做（chain 不行）

    chain 的单目内参是在**全量**数据上标好后冻结的；在子集上"重跑"只会得到另一份
    内参，检验不到外参能不能泛化 —— 那不是交叉验证，那是重标一次。joint 的
    `stereoCalibrate` 每折真的重新解 `(R,T)`，所以测试折的 RMS 是**没参与拟合**
    的数据给出的，才有信息量。

    ## 打分口径

    测试折用 `reprojection_rms_px`（参数焊死、只放开板的摆放），不用
    `stereoCalibrate` 的返回值 —— 后者对测试折来说根本没被计算过。

    折内**关掉离群剔除**（`reject_outliers=False`）：否则每折剔掉的集合不同，
    训练/测试比值就混进了"剔了多少对"这个变量，无法比较。
    """
    from calib.validation import cross_validate
    from calib.stereo_calib import reprojection_rms_px

    cfg_cv = CalibConfig(
        expected_baseline_mm=cfg.expected_baseline_mm,
        rational_model=cfg.rational_model,
        fix_principal_point=cfg.fix_principal_point,
        reject_outliers=False,
        alpha=cfg.alpha,
    )

    def fit(train_idx, test_idx):
        tr = calibrate_stereo(objp, [ip1[i] for i in train_idx],
                              [ip2[i] for i in train_idx], size, cfg_cv, spec,
                              stereo_mode=mode)
        if not tr.ok:
            raise RuntimeError(f"训练折本身判定为「{tr.grade}」：{tr.verdict}")
        test = reprojection_rms_px(objp, [ip1[i] for i in test_idx],
                                   [ip2[i] for i in test_idx], tr.K1, tr.D1,
                                   tr.K2, tr.D2, tr.R, tr.T)
        # 判外参泛化**只看右目**：左目残差与 (R,T) 无关，合并会把它当常数底、
        # 把比值压向 1（真机 25 对上实测 1.45 vs 只算右目 1.56，差一个判定档）。
        params = (tr.K1, tr.D1, tr.K2, tr.D2, tr.R, tr.T)
        test_right = reprojection_rms_px(
            objp, [ip1[i] for i in test_idx], [ip2[i] for i in test_idx], *params,
            camera="right")
        test_left = reprojection_rms_px(
            objp, [ip1[i] for i in test_idx], [ip2[i] for i in test_idx], *params,
            camera="left")
        train_right = reprojection_rms_px(
            objp, [ip1[i] for i in train_idx], [ip2[i] for i in train_idx], *params,
            camera="right")
        train_left = reprojection_rms_px(
            objp, [ip1[i] for i in train_idx], [ip2[i] for i in train_idx], *params,
            camera="left")
        train_both = reprojection_rms_px(
            objp, [ip1[i] for i in train_idx], [ip2[i] for i in train_idx], *params)

        # ---- 数值自洽守卫（不依赖任何自报字段）----
        # 两目角点数相同、等权 pooling，恒有 rms_both² = (rms_R² + rms_L²)/2。
        # 这条恒等式能同时抓住两类真实 bug：残差被广播/拼错、只算了一目却标成两目。
        # 自报字段（used_train_idx / rms_estimator）可以说谎，算术说不谎。
        for tag, both, right, left in (("训练折", train_both, train_right, train_left),
                                       ("测试折", test, test_right, test_left)):
            expect = (right ** 2 + left ** 2) / 2.0
            if both > 0 and abs(both ** 2 - expect) / both ** 2 > 0.05:
                raise RuntimeError(
                    f"交叉验证打分自相矛盾（{tag}）：两目 {both:.4f}² 与 "
                    f"√((右 {right:.4f}² + 左 {left:.4f}²)/2)={expect ** 0.5:.4f}² 不符"
                    f"（>5%）。角点数应相等，出现偏差说明残差拼装或口径有 bug。"
                )
        return {"train_rms": float(train_right), "test_rms": float(test_right),
                # 声明估计量：两者必须相同，否则比值无意义（见 calib/validation.py 的守卫）
                "train_rms_estimator": "reprojection_right_px",
                "test_rms_estimator": "reprojection_right_px",
                "test_rms_both": float(test), "test_rms_left": float(test_left),
                # 左目的 train/test 必须与右目同口径：左目与 (R,T) 无关，
                # 两者本该几乎相等（差得多说明打分口径混了，别拿去当结论）
                "train_rms_left": float(train_left), "train_rms_both": float(train_both),
                # 供 cross_validate 的泄漏守卫核对：真正参与拟合的帧
                "used_train_idx": list(train_idx)}

    rep = cross_validate(fit, n_items=len(ip1), k=k, seed=0)
    # 两目合并的比值只作参考：从每折里取中位数，不参与判定
    import statistics as _st
    ok_folds = [f for f in rep.get("per_fold", []) if f.get("error") is None]

    def _med(key):
        vals = [f[key] for f in ok_folds if isinstance(f.get(key), float)]
        return float(_st.median(vals)) if vals else None

    rep["test_rms_both_median"] = _med("test_rms_both")
    rep["train_rms_both_median"] = _med("train_rms_both")
    # 左目自检：与 (R,T) 无关，train/test 应当接近 1（差太多 = 口径混了）
    rep["test_rms_left_median"] = _med("test_rms_left")
    rep["train_rms_left_median"] = _med("train_rms_left")
    return rep


def rig_angle(R) -> float:
    """两光轴夹角（度），只用 R；与 `StereoRig.angle_deg` 同一公式。"""
    z2 = np.asarray(R, np.float64).T @ np.array([0.0, 0.0, 1.0])
    n = float(np.linalg.norm(z2))
    if n <= 0:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(z2[2] / n, -1.0, 1.0))))


# --------------------------------------------------------------------------
def build_report_figure(rig: StereoRig, pairs, spec: BoardSpec, out_path: Path) -> None:
    """画一张验证图：把左右两路的角点放到同一张画布上（左半 / 右半）。

    极线误差是个抽象像素数，这张图让它变得可见：
    校正正确时，**左右对应的角点应该落在同一条水平线上**，
    所以这里把每一对对应角点用横线连起来 —— 线是水平的就说明校正对了。
    """
    if not pairs:
        return
    panels = []
    for k, pair in enumerate(pairs[:3]):
        pl, pr = rig.rectify_points(pair.corners_left, pair.corners_right)
        h, w = rig.image_size[1], rig.image_size[0]
        canvas = np.zeros((h, w * 2 + 10, 3), np.uint8)
        for y in range(0, h, 60):
            cv2.line(canvas, (0, y), (canvas.shape[1], y), (60, 60, 66), 1)
        n = min(len(pl), len(pr))
        max_dy = 0.0
        for i in range(n):
            x1, y1 = pl[i]
            x2, y2 = pr[i]
            max_dy = max(max_dy, abs(y1 - y2))
            cv2.line(canvas, (int(x1), int(y1)), (int(x2) + w + 10, int(y2)), (90, 200, 90), 1)
        for pts, off in ((pl, 0), (pr, w + 10)):
            for x, y in pts:
                cv2.circle(canvas, (int(x) + off, int(y)), 3, (0, 220, 255), -1)
        cv2.putText(canvas, f"pair {k + 1}: left | right   max |dy| = {max_dy:.3f} px",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1, cv2.LINE_AA)
        panels.append(canvas)
    fig = np.vstack(panels) if len(panels) > 1 else panels[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), fig)
    print(f"验证图已保存：{out_path}（左右对应角点连成的线越水平，校正越准）")


def main() -> int:
    quiet_repeated_grab_warnings()
    ap = argparse.ArgumentParser(description="自动双目标定")
    add_stereo_source_args(ap)
    ap.add_argument("--cols", type=int, default=8, help="棋盘格内角点列数")
    ap.add_argument("--rows", type=int, default=5, help="棋盘格内角点行数")
    ap.add_argument("--square", type=float, default=25.0, help="方格实际边长（毫米）")
    ap.add_argument("--use-session-board", action="store_true",
                    help="标定板规格用当前 chain 会话 session.json 记录的那份，"
                         "忽略 --cols/--rows/--square。会话没记录板规格时本开关无效")
    # 默认值 = **试标定档（8 对）**，由用户 2026-10-10 指定，用于先跑通通路。
    #
    # 代价必须说清：`CalibConfig` 的注释写着"默认值按 20-30 对图像调好的"，
    # 重投影/极线门槛（good 0.35 px、ok 0.80 px）也是按那个量级定的。
    # 8 对能解出外参（内参是固定的），但：
    #   · 离群剔除会让剩余低于 8 时**停止剔除**（stereo_calib.py 的门槛），
    #     所以坏数据可能留在里面，RMS 偏高；
    #   · 姿态覆盖必然不足，边角区域的畸变约束不住 —— 三维在画面边角会偏。
    # **拿它验证"通路能不能跑通"可以，拿它出正式三维结果不行。**
    #
    # ⚠️ **默认值已从 8/8 -> 25/10 -> 40/20**（2026-10-19，按真机标定经验）。
    # 原来的 8/8 是"试标定档"，但用户从 GUI 走时根本不会去加参数，
    # 于是**永远停在试标定档**却以为在做正式标定 —— 那比没有默认值更糟。
    # 40 对是**输入**对数：真机约 1/3 会因移动/位姿重复被剔除，
    # 留出余量后仍能剩 20+ 对参与求解。只想验证通路请显式给 `--target 8`。
    ap.add_argument("--target", type=int, default=40,
                    help="目标收多少对（默认 40 = 正式；只想验证通路可给 8）")
    ap.add_argument("--min-pairs", type=int, default=20,
                    help="最少收多少对才尝试标定（默认 20）")
    ap.add_argument("--baseline-mm", type=float, default=None, help="预期基线（毫米），用于核对")
    # ---- 内参解法的两个开关（2026-10-26 接上 CLI）----
    #
    # 这两个参数在 `CalibConfig` 里一直存在、`stereo_calib` 也一直在读，但**全仓
    # 没有任何入口**：用户永远只能拿到 5 参数、放开主点的解。参数在代码里而入口
    # 不存在，等于告诉使用者"这条路可以走"却把门锁上。
    ap.add_argument("--rational-model", action="store_true",
                    help="畸变用 8 参数有理模型（k1..k6）而不是 5 参数。"
                         "**什么时候该开**：视场很广/畸变很大（广角短焦、鱼眼、画面边缘直线"
                         "明显弯成弧）。**开了会怎样**：待解参数从 5 个涨到 8 个，需要更多、"
                         "更铺满画面边角的姿态；姿态不足时它会过拟合（重投影变小、"
                         "而外参与深度反而更差）。默认关 = 与本项目历史结果同口径")
    ap.add_argument("--fix-principal-point", action="store_true",
                    help="把主点 (cx,cy) 固定在图像中心、不参与优化。"
                         "**什么时候该开**：姿态很少（<约 10 个）或板始终只在画面中央 —— "
                         "这时主点与焦距/畸变会互相抵消而漂掉。**开了会怎样**：主点若真有"
                         "装配偏差，这个偏差会被强制带进所有结果，且报告不会报警。默认关")
    ap.add_argument("--no-cv", action="store_true",
                    help="跳过 4 折交叉验证（joint 且 n_pairs≥12 时默认会跑）。"
                         "**它做什么**：把图像分 4 折，每次用 3 折重标、剩 1 折只打分，"
                         "看训练/测试 RMS 的比值 —— 这是唯一能发现「过拟合」的指标。"
                         "**代价**：多花几秒。只想先看一眼结果时可以关")
    ap.add_argument("--stereo-mode", choices=("auto", "parallel", "angled"),
                    default="auto",
                    help="双目模式：auto=沿用旧输出模式（无旧文件则 parallel）；"
                         "angled=大角度（近似直角）。实际夹角由标定 R 计算，不写死 90°")
    # 选项与默认值都来自标定方法注册表（`calib/methods.py`，R7）——
    # 这里不再写第二份方法名清单，说明文字直接用注册表里的用户视角 summary。
    ap.add_argument("--method", choices=names("calib_method"),
                    default=default("calib_method"),
                    help="外参求解方法（默认 %(default)s）：" + "；".join(
                        f"{d['name']}={d['summary']}" for d in describe("calib_method")))
    # ---- chain 的三样输入（本轮只支持离线目录）----
    ap.add_argument("--board-b-dir", default=None,
                    help="chain：右路那块 B 板的图像目录（省略则用 --right-dir）")
    ap.add_argument("--pair-frame", default=None,
                    help="chain：**单相机同时拍到 A、B 两板**的图（文件或目录），"
                         "用来求两板刚性相对位姿 T_AB")
    ap.add_argument("--pair-camera", choices=("left", "right"), default=None,
                    help="chain：--pair-frame 是哪一路相机拍的（left/right）")
    ap.add_argument("--same-board", action="store_true",
                    help="同一块固定板：板在两次拍摄之间不得移动；"
                         "左/右各拍一轮（--left-dir/--right-dir），"
                         "无需双板靶标与同框图。此时 T_AB 按定义取 I")
    ap.add_argument("--force-save", action="store_true",
                    help="⚠ 仅链路测试，不得用于数据结论：质量差时把结果写入正式 --out 路径。"
                         "默认（不加此开关）不中止程序，但会把结果另存为 "
                         "*_unusable_<时间戳>.npz/.json 旁路文件供检查，"
                         "激活装置档案绝不改动；加此开关才写正式路径。"
                         "两种产物都标 forced_unusable /「仅链路测试」，下游据此拒绝出数")
    ap.add_argument("--ref-left", default=None,
                    help="单板模式的外参锚点：左相机看这块板、**板不动**的参考图"
                         "（文件或目录，1~5 张）。板要整体入画")
    ap.add_argument("--ref-right", default=None,
                    help="单板模式的外参锚点：右相机看**同一块板同一个位置**的参考图"
                         "（文件或目录，1~5 张）。内参仍取 --right-dir 的结果")
    ap.add_argument("--ref-pair-left", default=None,
                    help="双板模式的外参锚点（2026-11-09 方案 A）：组合装置摆好别动，"
                         "左相机**同时**拍 A 板那张（文件或目录）")
    ap.add_argument("--ref-pair-right", default=None,
                    help="双板模式的外参锚点：与 --ref-pair-left **同一时刻**右相机"
                         "拍 B 板那张（文件或目录）；两目录同名文件自动成对")
    fov = ap.add_mutually_exclusive_group()
    fov.add_argument("--skip-fov", action="store_true",
                     help="跳过两路视场（FOV）对齐：不估单应、不裁剪不缩放，"
                          "按两路各自原始画面标定。angled 默认就这样")
    fov.add_argument("--fov", action="store_true",
                     help="即使 --stereo-mode angled 也强行做 FOV 对齐（单应/裁剪的前提"
                          "只对近平行成立；大角度下是否可用由你自己判断）")
    ap.add_argument("--require-fov", action="store_true",
                    help="视场对齐估计失败就退出（旧的 fail-closed 语义）。"
                         "默认不要求：parallel 下估不出来时降级为不裁剪、"
                         "按两路原始画面继续标定，并在 meta 里写 fov=null")
    ap.add_argument("--out", default=None,
                    help="输出前缀；缺省写到激活装置档案的标定路径，再回落 "
                         "config/stereo_calibration")
    ap.add_argument("--save-pairs", default=None, help="把收下的图像对另存到这个目录（便于复查）")
    ap.add_argument("--report-png", default=None, help="输出验证图路径")
    ap.add_argument("--no-show", action="store_true", help="不弹窗（批处理）")
    ap.add_argument("--max-frames", type=int, default=0, help=">0 时最多处理这么多帧")
    ap.add_argument("--stationary-px", type=float, default=None,
                    help="判定'板停稳'的帧间位移上限（像素，默认 3）。"
                         "手持板够不着就调大，但别超过 ~12：这条门限同时替代了左右同步，"
                         "放太松会让两路拍到不同位姿，解出的 R/T 会系统性偏")
    ap.add_argument("--min-sharpness", type=float, default=None,
                    help="额外绝对下限（focus 单位）。默认 800——低于 "
                         "findChessboardCornersSB 的可检出下限（约 3200），"
                         "实际不会触发；真正生效的是「低于本组中位数 40%%」"
                         "的相对清晰度判据。要更严请调大，但会成片拒收。")
    ap.add_argument("--relaxed", action="store_true",
                    help="手持拍摄的宽松预设：停稳上限 10 px、额外绝对下限 400、"
                         "允许更小的板。**换来过检风险**，精度不满意就退回默认值重标")
    ap.add_argument("--fov-from-profile", default="", metavar="NAME",
                    help="复用指定装置档案里已量好的 FOV 裁剪参数，跳过现场估计。"
                         "两路焦距不同是装置属性，量一次锁住可保证每次标定口径一致"
                         "（量法：apps/tools/measure_fov.py --write-profile NAME）。"
                         "分辨率与档案记录不一致会直接报错，不静默沿用")
    # 标定输出路径：显式 --out 优先，否则跟随激活装置档案，再回落顶层。
    # 2026-11-04 前这里硬编码顶层，档案"切换装置"是假动作。
    args = ap.parse_args()
    if args.out is None:
        # 这是**输出**路径（新标定要写到哪），不要求它已经存在 → strict=False。
        args.out = str(resolve_calibration_base(strict=False))
    if args.skip_fov and args.require_fov:
        ap.error("--skip-fov 与 --require-fov 矛盾：前者不做 FOV 对齐，后者要求"
                 "「估不出来就失败」")
    if args.skip_fov and args.fov_from_profile:
        ap.error("--skip-fov 与 --fov-from-profile 矛盾：前者明确不做 FOV 对齐，"
                 "后者要求复用它。要复用就删掉 --skip-fov")
    # chain 产出的是 none_raw（未做极线校正）的链式外参；parallel 重建依赖
    # 校正后的极线与视差（`stereo/triangulate.py` 的 dy 门限/视差范围），90° 下
    # 两者都不成立。允许标成 parallel = 让下游静默 0 配对，所以入口就拒绝。
    # （下面三处校验是**业务规则**，不是分派逻辑；方法的属性从注册表取，不比较名字。）
    method = method_plan(args.method)
    if args.fov_from_profile and method.rectification == "none_raw":
        ap.error(
            f"--fov-from-profile 只对会做视场裁剪的方法有意义：{args.method} 产出"
            f"未校正标定（none_raw）、不做 FOV 对齐 —— 90° 装置两路几乎不共视，"
            f"单应裁剪没有物理意义。档案里的 FOV 参数仍会记在装置档案里供查，"
            f"但不参与这条标定路。")
    if method.rectification == "none_raw" and args.stereo_mode == "parallel":
        ap.error("chain 产出的是未校正标定（none_raw），不能标成 parallel："
                 "parallel 重建依赖校正后的极线与视差，90° 下不成立。")
    # 单板模式没有"两块板"：同框图/同框相机/B 板目录都失去测量对象。
    # 静默忽略它们最坏 —— 用户会以为自己给的同框图被用上了（T_AB 其实仍是 I）。
    if args.same_board and (args.pair_frame or args.pair_camera or args.board_b_dir):
        ap.error("--same-board 与 --pair-frame/--pair-camera/--board-b-dir 互斥："
                 "单板模式没有\"两块板\"可测，T_AB 直接取 I。")
    if args.same_board and not method.same_board:
        ap.error(f"--same-board 只对需要它的方法有意义（{args.method} 要求两路共视）；"
                 f"请改用 --method {' / '.join(n for n in names('calib_method') if method_plan(n).same_board)}。")

    # 日志要在**做任何事之前**起来：标定失败时最有价值的证据（拒绝了什么、
    # 各拒了多少帧）现在只在内存里，进程一退就没了。
    log_path = setup_logging("calibrate")
    log = logging.getLogger("calibrate")
    print(f"本次日志：{log_path}")
    log.info("=== 双目标定开始 ===")
    _size_note = ("自适应（逐档协商）" if (args.width <= 0 or args.height <= 0)
                  else f"{args.width}x{args.height}")
    log.info("参数：left=%s right=%s backend=%s 采集档=%s@%d 目标 %d 对（最少 %d）",
             args.left or args.left_video or args.left_dir,
             args.right or args.right_video or args.right_dir,
             args.backend, _size_note, args.fps,
             args.target, args.min_pairs)
    log_environment(log)
    if args.relaxed:
        log.warning("启用了 --relaxed：门限放松，用精度换成功率")

    # ---- 板规格：chain/offline 路径先核对拍图会话里记录的规格 ----
    # 采集后改界面参数、或换一份档案再标定，旧规格拍的图会被新规格静默解出
    # （真会话实测：8x5@25 的图按 11x8@9 解 -> 0/40 检出，且不报错）。
    # 非 chain 路径（相机/视频）没有会话，下面整段跳过，行为完全不变。
    spec_values = {"cols": args.cols, "rows": args.rows, "square": args.square}
    if method.offline_only:
        session = None
        dirs = current_chain_dirs()
        if dirs is not None:
            sjson = Path(dirs["left"]).parent / "session.json"
            try:
                session = json.loads(sjson.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                print(f"[板规格] 会话记录读不出来（{exc}），无法核对；按命令行规格继续。")
        spec_values, code = resolve_board_spec(
            spec_values, session, use_session_board=args.use_session_board)
        if code != 0:
            return code
    spec, code = make_board_spec(spec_values)
    if code != 0:
        return code
    print(f"标定板：{spec.describe()}")
    if args.target < 15:
        # 低目标数必须**显式可见**，不能悄悄用：否则以后看到一份差标定，
        # 没人知道是因为只收了几对。
        print(f"⚠ 试标定档：只收 {args.target} 对（默认/正式档 40 对）。")
        print("   内参是固定的，少几对也能解出外参，但姿态覆盖不足、离群剔除会被下限卡住 ——")
        print("   **够验证通路，不够出正式三维结果**。要正式标定去掉 --target（默认 40）")
        print()
        log.warning("试标定档：target=%d（正式档 40）", args.target)
    log.info("标定板：%s", spec.describe())
    # ---- chain：离线目录流程（不打开相机/视频，也不做 FOV 对齐）----
    # 分派按注册表声明（`offline_only`），不比较方法名。
    if method.offline_only:
        if args.same_board:
            print(f"方法：{args.method} + 单板（独立单目 + 同一块固定板公共参考，T_AB=I）")
        else:
            print(f"方法：{args.method}（独立单目 + 刚性双板公共参考，不需要两路共视）")
        log.info("method=%s same_board=%s", args.method, args.same_board)
        return _run_chain(args, spec, log)

    print("提示：把板举到**两路都能看到**的位置，慢慢移动，每个姿势停一下（程序只收停稳的帧）")
    print()

    src = open_pair_source(args)
    if src is None:
        log.error("取帧源装配失败，退出")
        return 1
    print(src.describe())
    log.info("取帧源：%s", src.describe())
    src.report_warnings()
    for w in getattr(src, "warnings", []) or []:
        log.warning("取帧源告警：%s", w)

    # ---- 双目模式：入口层唯一解析点 ----
    # auto 必须在这里落定；核心层 normalize_mode() 收到 auto 会直接报错。
    # 旧输出 JSON 的 meta 是**平铺**的（StereoRig.save 不做嵌套），所以整份
    # JSON 直接当 calibration 传进去，不要包成 {"meta": old}（那样永远查不到）。
    prev_meta = _previous_calib_meta(args.out)
    mode = resolve_auto(args.stereo_mode, calibration=prev_meta)
    if args.stereo_mode == "auto":
        note = (f"沿用旧输出 {Path(args.out).with_suffix('.json').name}"
                if prev_meta else "无旧输出，默认 parallel")
        print(f"[模式] {mode}（auto：{note}）")
        log.info("auto -> %s（%s）", mode, note)
    else:
        print(f"[模式] {mode}（显式）")
        log.info("模式：%s（显式）", mode)

    # FOV 对齐的前提是两光轴**近平行**：两路看到同一片纹理，单应 H 才能把
    # 较广一路裁到较窄一路的视场里。大角度（~90°）时两路视场几乎不重叠，
    # H 没有物理意义，裁出来的画面会错位 —— 所以 angled 默认不做对齐，
    # 两路按各自原始画面标定（坐标系仍是 left_camera，见 数据标准.md §1.5）。
    # --fov 可以强行打开（是否真的近共视由用户判断）。
    if args.require_fov and mode == "angled" and not args.fov:
        # 静默失效是最坏的：用户要求"估不出来就失败"，但 angled 压根不做对齐，
        # 这个要求会被悄悄忽略。宁可报错让他选清楚。
        ap.error("--require-fov 在 angled 模式下无意义（该模式不做 FOV 对齐）。"
                 "要强制对齐请加 --fov；要严格的失败语义请用 --stereo-mode parallel。")
    do_fov = not args.skip_fov and (mode != "angled" or args.fov)
    if mode == "angled" and not do_fov:
        print("[FOV] 跳过视场对齐（angled）：单应/裁剪的前提只对近平行成立；"
              "两路按各自原始画面标定。")
        print("      如需强行对齐（仅当你确认两路其实近共视）请加 --fov。")
        log.info("angled 模式跳过 FOV 对齐（可用 --fov 强制）")
    elif args.skip_fov:
        print("[FOV] 已按 --skip-fov 跳过视场对齐，两路按原始画面标定。")
        log.info("--skip-fov：跳过 FOV 对齐")

    # ---- 启动时一次性估计 FOV 对齐（之后逐帧复用，绝不重估） ----
    #
    # 两种失败要分开对待：
    #   · **估不出来**（特征太少 / 纯色墙）→ 默认在 parallel 下降级：不裁剪，
    #     按两路原始画面继续标定。`stereoCalibrate` 不要求两路视场对齐，
    #     这在几何上完全合法；只是两路像素尺度不一致时视差/深度会有系统偏差，
    #     所以必须**明确打印**这次没有对齐，并写进 meta（fov=null）。
    #     要恢复旧的 fail-closed，用 --require-fov。
    #   · **估出来但不可信**（quality 低）→ 仍然由下面的 fov_gate_action()
    #     拦住，那条是"不可信还硬用"，不在降级范围内。
    #
    # 相机刚打开的前几帧可能是黑帧/坏帧，这里最多读 30 帧或约 5 秒；
    # 一旦拿到成对帧就**只估计一次**，并把这帧对留给主循环（不丢弃）。
    # 跳过 FOV 时仍然要读这一帧 —— 主循环拿它当首帧，不浪费。
    fov = None
    first_pair = None
    read_attempts = 0
    probe_deadline = time.perf_counter() + 5.0
    while read_attempts < 30 and time.perf_counter() < probe_deadline:
        ok, left0, right0 = src.read()
        if not ok or left0 is None or right0 is None:
            if not src.is_live:
                break
            read_attempts += 1
            time.sleep(0.05)
            continue
        first_pair = (left0, right0)
        break

    if first_pair is None:
        print("\n[错误] 启动后读不到可用的成对画面，无法开始采集。")
        if src.is_live:
            # ⚠️ 这里**不许**再写"先确认两路相机都能出图 → 跑 doctor"：旧 doctor
            # 是**逐台串行**检查（同一时刻只开一台），两台分开都正常、**同时**取帧
            # 却失败时它会说"两台都好" —— 用户照它重试就永远走不出来（死循环）。
            # 现在指向真问题：doctor 的**「双路同时取帧」**那段（与双目同工况），
            # 以及设备级失效只能物理拔插。
            print("       这是**取帧失败**（相机没给出成对画面），与**标定算法拒收**"
                  "是两回事：")
            print("       拒收发生在画面正常、只是棋盘没停稳/没完整出现在两路里的时候。")
            print("       取帧失败按代价从小到大排查：")
            print("         ① 跑 `python apps/tools/cameras.py doctor`，看**「双路同时取帧」**"
                  "那段结论")
            print("            —— 它是与双目同工况地同时开两路；逐台检查会说'两台都好'，"
                  "不作数。")
            print("         ② 把那路失败的相机**物理拔插**一次，并换一个**直插机身**的 "
                  "USB 口")
            print("            （别走 HUB / 扩展坞）；设备级失效（-2147483638）软件重开"
                  "救不回来。")
            print("         ③ 两路分辨率必须一致：若某台相机只支持某个档，用 "
                  "`--width/--height` 指定**两台都支持**的档。")
            print("       本次不会在没有画面的情况下继续标定。")
        else:
            print("       两路视频/图片序列没有同时有效的帧：检查左右文件是否成对、"
                  "长度是否一致。")
        log.error("启动阶段取不到成对帧，退出（%s）", src.kind)
        src.release()
        return 1

    if do_fov:
        # 两条来源：复用装置档案里**已量好**的参数（推荐，口径稳定），
        # 或者现场从启动帧估一次（旧行为，纹理不足会失败/降级）。
        fov_fail_reason: str | None = None
        if args.fov_from_profile:
            try:
                fov = _fov_from_profile(args.fov_from_profile, first_pair)
            except ValueError as exc:
                print(f"\n[错误] 无法复用装置档案里的 FOV 参数：{exc}")
                log.error("复用档案 FOV 失败：%s", exc)
                src.release()
                return 2
            print(f"[FOV] 复用装置档案「{args.fov_from_profile}」里已量好的参数"
                  f"（不重新估计，口径与上次一致）")
            print(f"[FOV] scale={fov.scale:.4f}  inliers={fov.inliers}  "
                  f"quality={fov.quality:.3f}  crop_rect={list(fov.crop_rect)}  "
                  f"target_size={list(fov.target_size)}")
            log.info("FOV 复用档案 %s：%s", args.fov_from_profile, fov.to_dict())
        else:
            print("[FOV] 正在用启动帧估计两路视场对齐（只做一次，约数秒）…", flush=True)
            try:
                fov, fov_fail_reason = _fov_or_none(
                    lambda: estimate_fov_alignment(first_pair[0], first_pair[1]),
                    mode=mode, require_fov=args.require_fov)
            except Exception as exc:                 # noqa: BLE001 - 原始异常原样呈现
                print("\n[错误] 无法自动估计两路 FOV 对齐，已停止（不会按原生分辨率继续）。")
                print(f"       原因：{type(exc).__name__}: {exc}")
                print("       请把**有纹理的物体或标定板**同时放进两路画面、光照均匀、"
                      "对焦清晰，然后重试。")
                print("       空白墙/纯色桌面/严重欠曝都会让特征匹配失败。")
                print("       本次是 --require-fov 或非 parallel 模式，所以不降级。")
                print("       若这台装置已经量过 FOV，可用 --fov-from-profile <档案名> 复用。")
                log.error("FOV 估计失败，退出（--require-fov / 非 parallel）：%s", exc)
                src.release()
                return 1

        if fov is None:
            print(f"\n[警告] 两路视场对齐估计失败（{fov_fail_reason}）："
                  "本次不做裁剪，按两路原始画面继续标定。")
            print("       注意：两路像素尺度可能不一致，视差/深度会有系统偏差；"
                  "标定 meta 里 fov=null。")
            print("       下一步：若这是大角度/90° 装置，下次请直接选『90° 装置』"
                  "或加 --stereo-mode angled --skip-fov；若只是画面纹理太少，"
                  "可换有纹理的场景重标。")
            log.warning("FOV 估计失败，降级为不裁剪（fov=None）：%s", fov_fail_reason)
            print()
        else:
            # 这五个数不管哪种模式都并列打印出来：判据交给人，但数字必须是明面上的。
            print(f"[FOV] scale={fov.scale:.4f}  inliers={fov.inliers}  "
                  f"quality={fov.quality:.3f}  crop_rect={list(fov.crop_rect)}  "
                  f"target_size={list(fov.target_size)}")
            if mode == "parallel":
                print("      平行基线参照（本机现役装置，实测夹角 1.4°）：scale=0.6649  "
                      "inliers=296  quality=0.513  crop_rect=[404,195,1718,973]  "
                      "target_size=[1718,972]")
                # 只有"这次估计真的会被用来裁剪"才拦；不裁剪时低置信度不会污染
                # 内参，中止它等于让一个与结果无关的条件决定能不能干活。
                action = fov_gate_action(fov)
                if action == "block":
                    print(f"\n[错误] FOV 对齐不可信（{fov_confidence_problem(fov)}）："
                          "拒绝用这次估计裁剪标定图。")
                    print("       下一步：换有纹理的场景（标定板 / 纸箱 / 桌面杂物）重拍，"
                          "别对着白墙或纯色桌面；")
                    print("       或者用 --skip-fov 让两路各用原始画面"
                          "（大角度装置请加 --stereo-mode angled）。")
                    log.error("FOV 对齐不可信（%s），退出：%s",
                              fov_confidence_problem(fov), fov.to_dict())
                    src.release()
                    return 1
                if action == "warn":
                    print(f"[警告] FOV 估计置信度低（{fov_confidence_problem(fov)}），"
                          "本次不裁剪，按原始画面标定；内参不受影响。")
                    log.warning("FOV 估计置信度低但不裁剪（%s）：%s",
                                fov_confidence_problem(fov), fov.to_dict())
            else:
                print("      注：angled 不按下限判据 —— 大角度下的失败模式是"
                      "「在近处某个平面上匹配得很好」，")
                print("          inliers 变多不是好消息；请自行核对 scale 是否偏离物理预期。")

            mag = max(fov.scale, 1.0 / fov.scale)
            print("[FOV] 自动视场对齐（启动时只估计一次，之后逐帧复用）")
            print(f"      参考（较窄；必要时降采样，不裁剪）：{fov.reference_side}    "
                  f"裁剪：{fov.crop_side}")
            print(f"      裁剪区 (x, y, w, h) = {fov.crop_rect}    "
                  f"归一化尺寸 {fov.target_size[0]}x{fov.target_size[1]}")
            print(f"      较窄一侧相对较广一侧放大 ~{mag:.2f}x    "
                  f"尺度 left->right={fov.scale:.4f}    "
                  f"质量 {fov.quality:.2f}（{fov.method}，内点 {fov.inliers}）")
            if fov.active:
                wide_frame = first_pair[0] if fov.crop_side == "left" else first_pair[1]
                wide_h, wide_w = wide_frame.shape[:2]
                x, y, w, h = fov.crop_rect
                keep = (w * h) / float(max(1, wide_w * wide_h))
                print(f"      ⚠ 较广的一侧（{fov.crop_side}）会损失视场：只保留 {keep:.0%} "
                      f"原画面（{w}x{h} / {wide_w}x{wide_h}），其余会被裁掉；")
                print(f"        录制的两路视频会统一为 {fov.target_size[0]}x"
                      f"{fov.target_size[1]} 并沿用这份对齐参数。")
            else:
                print("      两路视场基本一致：不裁剪、不缩放（crop_side=none）。")
            log.info("FOV 对齐：参考 %s / 裁剪 %s / 裁剪区 %s / 放大 %.3fx / "
                     "质量 %.3f（%s，内点 %d）",
                     fov.reference_side, fov.crop_side, fov.crop_rect, mag,
                     fov.quality, fov.method, fov.inliers)
            print()

    from calib import CaptureConfig
    cap_kw: dict = dict(
        target_pairs=args.target,
        min_pairs=args.min_pairs,
        # 独立静帧场景不要求"板静止"（帧间本来就是不同位姿）
        require_stationary=(src.kind != "dir"),
    )
    if args.relaxed:
        # 手持预设：先整组套上，再用显式参数覆盖（顺序不变 = 行为不变）。
        cap_kw.update(RELAXED_PRESET)
        for line in relaxed_note(args.stationary_px, args.min_sharpness):
            print(line)
        print()
    if args.stationary_px is not None:
        cap_kw["stationary_max_px"] = float(args.stationary_px)
    if args.min_sharpness is not None:
        cap_kw["min_sharpness"] = float(args.min_sharpness)
    cap_cfg = CaptureConfig(**cap_kw)
    collector = AutoPairCollector(spec, cap_cfg)
    # 全屏/窗口尺寸统一由 apps/_window.py 管（各程序口径一致）
    wins = WindowManager()

    cov_cache = None
    cov_cache_n = -1
    det_ms = 0.0
    t_prev = time.perf_counter()
    fps_frames = 0
    pairs_dir: Path | None = Path(args.save_pairs) if args.save_pairs else None
    if pairs_dir:
        (pairs_dir / "left").mkdir(parents=True, exist_ok=True)
        (pairs_dir / "right").mkdir(parents=True, exist_ok=True)

    frames = 0
    # ⚠️ **读不到帧必须能自己收场**（2026-10-14 修的真缺陷）。
    #
    # 原来（`if src.is_live: continue`）在相机掉线时会**永远转下去**，
    # 而且**一个字都不打印** —— 用户看到的就是"标定程序卡死了"，
    # 只能去关启动器（而关窗口并不会杀掉子进程）。
    #
    # 相机掉线的信号很明确（`-2147483638` = MF_E_VIDEO_RECORDING_DEVICE_INVALIDATED，
    # 实测每 10 秒一次），所以：先报，再等，超时就带着已经收到的对数退出，
    # 让**已经收下的图像对不白收**。
    miss = 0
    miss_reported = False
    # 启动时估计 FOV 用的那帧对**不能丢**：第一轮直接用它喂 collector。
    pending_pair = first_pair
    first_pair = None
    try:
        while True:
            if pending_pair is not None:
                left, right = pending_pair
                pending_pair = None
                ok = True
            else:
                ok, left, right = src.read()
            if not ok or left is None or right is None:
                if not src.is_live:
                    break
                miss += 1
                if miss == 30 and not miss_reported:
                    miss_reported = True
                    why = ""
                    health = getattr(src, "health_line", None)
                    if callable(health):
                        why = f"（{health()}）"
                    print(f"\n⚠ 读不到画面了{why}。已收 {collector.count} 对。", flush=True)
                    print("   常见原因：相机被别的程序占用 / USB 掉线 / 掉电。"
                          "拔插一次 USB 往往就好。", flush=True)
                    print("   再等 5 秒；还是不行就保存已收的对并退出。", flush=True)
                if miss >= 150:                      # 约 5 秒（30fps）
                    print(f"\n✘ 连续 5 秒读不到画面，停止采集。"
                          f"已收 {collector.count} 对（目标 {args.target}）。", flush=True)
                    break
                time.sleep(0.02)
                continue
            miss = 0
            miss_reported = False
            frames += 1
            if args.max_frames and frames > args.max_frames:
                break

            # 逐帧复用启动时得到的同一份 FOV 参数（不重估）；两路归一化后再
            # 送进 collector / 存图，所以 collector.image_size 就是归一化尺寸。
            # 跳过对齐（angled / --skip-fov）时两路保持原始画面。
            if fov is not None:
                left, right = fov.apply(left, right)

            t0 = time.perf_counter()
            gl = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            gr = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
            accepted, reason = collector.offer(gl, gr)
            det_ms = 0.9 * det_ms + 0.1 * (time.perf_counter() - t0) * 1000.0
            fps_frames += 1

            if accepted and pairs_dir:
                i = collector.count
                cv2.imwrite(str(pairs_dir / "left" / f"{i:03d}.png"), left)
                cv2.imwrite(str(pairs_dir / "right" / f"{i:03d}.png"), right)

            if not args.no_show:
                # 直接复用粗检结果画框 —— 之前这里又检测了一遍，
                # 每帧 4 次全图精确检测（约 1.2 秒）就是卡顿的根源。
                #
                # ---- 布局：先算文字，图像用剩下的空间 ----
                #
                # 为什么这么排：原来状态文字压在图像上（逐路状态在右上角、
                # 拒绝统计在左下角），而且面板比窗口高，底部那几条**被裁掉** ——
                # 被裁掉的恰好是"为什么收不满"的答案。
                # 所以现在把两路图像 + 覆盖率图放上面，所有文字放**下面独立的条带**里，
                # 并且反过来让图像去适应文字，而不是让文字去挤图像。
                footer_lines = (
                    collector.status_lines()
                    + collector.side_lines()
                    + [f"detect {det_ms:.0f} ms/frame  ({1000.0 / max(det_ms, 1e-3):.0f} fps)"
                       f"   [q] 退出  [r] 清空重收  [f] 全屏（Esc 退出全屏）"]
                )
                footer_h = len(footer_lines) * FOOTER_LINE_H + 10

                sw, sh = _screen_size()
                cov_w = 260          # 覆盖率面板宽度：要放得下宫格占用数字与棋盘四边形
                per_cam_w = max(240, (sw - 60 - cov_w) // 2)
                # 取两路里更"瘦"的宽高比，保证两路都放得下（两路分辨率可能不同）
                ar = min(left.shape[1] / left.shape[0], right.shape[1] / right.shape[0])
                img_h = int(min(PREVIEW_W / ar, per_cam_w / ar, sh - 90 - footer_h))
                img_h = max(200, img_h)

                def _prep(img, target_h):
                    s = target_h / float(img.shape[0])
                    out = cv2.resize(
                        img,
                        (max(1, int(round(img.shape[1] * s))), target_h),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    return out, s

                a, scale_l = _prep(left, img_h)
                b, scale_r = _prep(right, img_h)
                # 宽高比不同时补黑边，而不是拉伸（拉伸会让角点坐标与图上对不上）
                for img in (a, b):
                    if img.shape[0] != img_h:
                        img = np.vstack([img, np.zeros((img_h - img.shape[0], img.shape[1], 3), np.uint8)])
                for img, obs, sc in ((a, collector.last_obs[0], scale_l),
                                     (b, collector.last_obs[1], scale_r)):
                    if obs is None or not obs.found or obs.corners is None:
                        continue
                    colour = (0, 220, 0) if collector.last_accepted else (0, 140, 255)
                    quad = obs.quad()
                    if quad is not None:
                        cv2.polylines(img, [(quad * sc).astype(np.int32)], True, colour, 2)
                    pts = obs.corners.reshape(-1, 2) * sc
                    for x, y in pts:
                        cv2.circle(img, (int(x), int(y)), 2, (0, 255, 255), -1)

                # 在两半上各打一个 LEFT / RIGHT 标记（含指定的相机编号/角色名）。
                # 为什么要标：这个窗口是用户**判断左右有没有接反**的唯一现场依据 ——
                # 他会在某台摄像头前晃手，看哪一半跟着动。没有标记就只能靠猜
                # "左半边到底是不是 left"。
                for img, tag in ((a, f"LEFT  ({args.left})"), (b, f"RIGHT ({args.right})")):
                    (tw, th), _base = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(img, (0, 0), (tw + 16, th + 14), (0, 0, 0), -1)
                    cv2.putText(img, tag, (8, th + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 255, 0), 2, cv2.LINE_AA)

                # 覆盖率图只在收帧数变化时重画（原来每帧都重建一次画布）
                if collector.count != cov_cache_n:
                    # **给输出画布尺寸**（不是源图尺寸）—— 这样位姿一定会被缩放并
                    # 居中画进面板里。曾经把源尺寸当成输出尺寸传，导致姿态画到画布外，
                    # 面板看起来是空的（用户："看不到已经标定的姿态和位置"）。
                    cov_cache = collector.coverage_image(out_size=(cov_w, img_h))
                    cov_cache_n = collector.count
                cov = cov_cache
                if cov is None or cov.shape[:2] != (img_h, cov_w):
                    cov = collector.coverage_image(out_size=(cov_w, img_h))
                row = np.hstack([a, b, cov])

                # 文字条带：整幅宽度，画在图像**下方**，永远不会压住画面
                footer = np.zeros((footer_h, row.shape[1], 3), np.uint8)
                for i, line in enumerate(footer_lines):
                    cv2.putText(footer, line, (10, 18 + i * FOOTER_LINE_H),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)
                panel = np.vstack([row, footer])
                wins.show(WINDOW, panel)
                key = cv2.waitKey(1) & 0xFF
                if wins.handle_key(WINDOW, key, panel):
                    continue
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    collector.reset()
                    print("已清空重新收集")

            if collector.full:
                print(f"已收满 {collector.count} 对，开始标定…")
                break
            if src.is_live and frames % 60 == 0:
                now = time.perf_counter()
                fps = fps_frames / max(now - t_prev, 1e-6)
                t_prev = now
                fps_frames = 0
                print(f"  已收 {collector.count}/{args.target}  {fps:5.1f} fps  "
                      f"检测 {det_ms:.0f} ms/帧  ({collector.last_reason})")
    finally:
        src.release()
        if not args.no_show:
            cv2.destroyAllWindows()

    print()
    print(f"共处理 {frames} 帧，收下 {collector.count} 对")
    # 结束时的**完整**统计：不再只印前三条，也不再只留在内存里。
    # 用户卡在"收了 8 对就不动"时，能看到的证据就是这里。
    print()
    for line in collector.summary_lines():
        print(line)
    collector.log_summary()
    log.info("采集结束：%d 对（目标 %d，最少 %d）", collector.count, args.target, args.min_pairs)

    if not collector.enough:
        hint = collector.next_step_hint()
        print(f"\n[失败] 只收到 {collector.count} 对，少于 {args.min_pairs} 对，无法标定。")
        if hint:
            print(f"下一步：{hint}")
        print(f"（完整逐帧记录见日志：{log_path}）")
        return 1

    objp = spec.object_points()
    ip1 = [p.corners_left for p in collector.pairs]
    ip2 = [p.corners_right for p in collector.pairs]
    img_size = collector.image_size
    # image_size 必须是 FOV 归一化后的尺寸（录制端会拿它核对分辨率）；
    # 不一致说明采集中途分辨率变了，继续标定只会得到错误内参。
    if fov is not None and fov.active and tuple(img_size) != tuple(fov.target_size):
        print(f"\n[错误] 采集帧尺寸 {img_size} 与 FOV 归一化尺寸 "
              f"{fov.target_size} 不一致，停止标定。")
        print("       通常是相机中途重连 / 分辨率被改变；请重新标定。")
        log.error("image_size %s 与 fov.target_size %s 不一致", img_size,
                  fov.target_size)
        return 1

    cfg = CalibConfig(expected_baseline_mm=args.baseline_mm,
                      rational_model=args.rational_model,
                      fix_principal_point=args.fix_principal_point)
    try:
        res = calibrate_stereo(objp, ip1, ip2, img_size, cfg, spec, stereo_mode=mode)
    except ValueError as exc:
        # 护栏的原文必须原样打出来：它包含"下一步怎么做"，
        # 被包装成一句"标定失败"就等于把最有用的信息扔了。
        print(f"\n[错误] {exc}")
        log.error("标定被拒绝：%s", exc)
        return 1

    # ---- 交叉验证（2026-10-26）----
    #
    # 必须在 `res.report()` **之前**算：报告里的那行要印它的结论。
    # 默认只在 joint 且有效对 ≥12 时跑（4 折、每折一次完整重标，几秒量级）。
    #
    # ⚠️ 只用**已通过离群剔除**的那批图：标定流程本身已经认定那 8 对是坏数据，
    #    再把它们塞进测试折，测出来的就是"坏数据有多坏"，而不是"外参能不能泛化"。
    cv_idx = list(res.used_index) if res.used_index else list(range(len(ip1)))
    cv_ip1 = [ip1[i] for i in cv_idx]
    cv_ip2 = [ip2[i] for i in cv_idx]
    if args.no_cv:
        res.cv = {"skipped": "--no-cv"}
    elif len(cv_idx) < CV_MIN_PAIRS:
        res.cv = {"skipped": f"有效对 n={len(cv_idx)} < {CV_MIN_PAIRS}，4 折拆不开"
                              f"（每折至少留 1 对只打分）"}
    else:
        t_cv = time.perf_counter()
        try:
            res.cv = _cross_validate_joint(objp, cv_ip1, cv_ip2, img_size, cfg, spec, mode)
            print(f"[CV] 4 折交叉验证完成（{time.perf_counter() - t_cv:.1f} s）")
        except Exception as exc:                          # noqa: BLE001 - 原文要保留
            res.cv = {"error": f"{type(exc).__name__}: {exc}"}
            log.warning("交叉验证失败：%s", exc)
        # 失败的折**逐条印原因**：只写"3/4 折失败"等于把唯一能修的线索藏起来。
        for fold in (res.cv.get("failed_folds") or []):
            print(f"  [CV] 折 {fold['fold']} 失败（训 {fold['n_train']} / "
                  f"测 {fold['n_test']}）：{fold['error']}")
    print(res.report())
    log.info("标定结果：\n%s", res.report())

    if args.report_png:
        try:
            rig_tmp = StereoRig.from_result(res)
            build_report_figure(rig_tmp, collector.pairs, spec, Path(args.report_png))
        except ValueError:
            pass

    # 新原则：质量差也要出结果、只有错误才紧急。res.ok=False 不再 return 1，
    # 继续构造 + 落盘（默认旁路 / --force-save 显式写正式路径）；判定逻辑一字不改。
    quality_bad = not res.ok

    # 模式在启动前就由 resolve_auto() 落定了（入口层唯一解析点），这里只做
    # "推断值与实测夹角是否自相矛盾"的复核；不再先 normalize 再分叉。
    measured_angle = float(optical_axes_angle_deg(res.R))
    if args.stereo_mode == "auto" and (
            (mode == "parallel" and measured_angle >= 40.0)
            or (mode == "angled" and measured_angle < 15.0)):
        print(f"[错误] auto 沿用的模式 {mode} 与本次实测夹角 {measured_angle:.1f}° 矛盾；"
              "请显式 --stereo-mode parallel 或 angled 后重标。")
        log.warning("auto 模式与实测夹角矛盾：%s / %.1f°", mode, measured_angle)
        return 1
    if mode == "angled" and measured_angle < 15.0:
        print(f"[警告] 标为 angled，但实测夹角只有 {measured_angle:.1f}°，更像平行双目。")
    # ---- 采集档：录制 / 复检 / 前置检查要复现**同一档**才能对齐 ----------------
    # `capture_size` 是相机的**实际采集分辨率**（FOV 裁剪**之前**），不是标定
    # `image_size`（裁剪**之后**）。录制端读它来按同一档开相机（见
    # `apps/_common.py` 的 `calib_capture_size`）；`capture` 子 dict 记录本次
    # 自适应协商的完整过程（请求了什么 / 定案成什么 / 两路实际值 / 成对帧率），
    # 供"标定 1080p、录制却换了档"这类问题回查。video/dir 源没有协商，置 None。
    if src.kind == "camera":
        _req = src.requested_size
        capture_meta = {
            "requested": list(_req) if isinstance(_req, tuple) else _req,
            "resolved": list(src.resolved_size) if src.resolved_size else None,
            "actual_left": list(src.actual_sizes[0]) if src.actual_sizes else None,
            "actual_right": (list(src.actual_sizes[1])
                             if len(src.actual_sizes) > 1 else None),
            "pair_fps": (round(src.pair_fps, 2)
                         if src.pair_fps is not None else None),
        }
        capture_size = list(src.image_size)
    else:
        capture_meta = None
        capture_size = None
    rig = StereoRig.from_result(res, meta={
        # C1：与 _chain_meta 同一份内容级身份证据（见那里的注释）。
        "profile": _active_profile_name(),
        "board": spec.describe(),
        # 与 _chain_meta 同一份结构化规格；`board` 字符串保留兼容旧读。
        "board_spec": {"cols": int(spec.cols), "rows": int(spec.rows),
                       "square_mm": float(spec.square_mm)},
        "source": src.kind,
        "left": args.left if args.left is not None else (args.left_video or args.left_dir),
        "right": args.right if args.right is not None else (args.right_video or args.right_dir),
        "image_size": list(img_size),
        "capture_size": capture_size,
        "capture": capture_meta,
        "fov": fov.to_dict() if fov is not None else None,
        "stereo_mode": mode,
        "angle_deg": measured_angle,
    }, allow_unusable=not res.ok)
    print(f"[标定] 实测光轴夹角 {measured_angle:.2f}°（模式 {mode}）")
    if mode == "angled":
        try:
            from calib.angled_quality import evaluate_angled, grade_angled
            used = [collector.pairs[i] for i in getattr(res, "used_index", [])
                    if 0 <= i < len(collector.pairs)]
            if not used:
                used = collector.pairs
            if used:
                left_px = np.vstack([p.corners_left.reshape(-1, 2) for p in used])
                right_px = np.vstack([p.corners_right.reshape(-1, 2) for p in used])
            else:
                left_px = np.zeros((0, 2), np.float64)
                right_px = np.zeros((0, 2), np.float64)
            q = evaluate_angled(rig.K1, rig.D1, rig.K2, rig.D2, rig.R, rig.T,
                                left_px, right_px, reproj_rms_px=res.rms)
            grade, verdict = grade_angled(q)
            rig.quality["angled"] = q.as_dict()
            rig.quality["angled_grade"] = grade
            rig.quality["angle_deg"] = float(optical_axes_angle_deg(rig.R))
            # 进入录制/数据管理的 verdict 必须是 angled 的判定，不能沿用平行门限
            rig.quality["verdict"] = verdict
            print(f"[angled] 标定质量：{grade} —— {verdict}")
            for warning in q.warnings:
                print(f"  [警告] {warning}")
            log.info("angled 质量：%s / %s / %s", grade, verdict, q.as_dict())
            if grade == "不可用":
                # 质量差：不再 return 1，交给 finalize_and_save 旁路保存 + 标记。
                quality_bad = True
        except Exception as exc:                            # noqa: BLE001
            rig.quality["angled_grade"] = "未评估"
            rig.quality["verdict"] = f"angled 质量评估失败，未验证：{exc}"
            print(f"[错误] angled 质量评估失败，未保存：{exc}")
            log.warning("angled 质量评估失败，未保存：%s", exc)
            return 1
    npz, js, _bypassed = finalize_and_save(rig, args, unusable=quality_bad, log=log)
    _saved = Path(str(npz)).is_file() if npz else False
    print(f"已保存：{npz}" if _saved else f"未写入正式路径（旧标定已保护）：{npz}")
    print(f"        {js}")
    print()
    print("下一步：")
    print(f"  - 三角测量时用 calib.StereoRig.load('{args.out}') 读取参数")
    print("  - 装置如果被碰动，用 apps/verify_stereo.py 复检极线误差")
    log.info("标定成功，已保存 %s", npz)
    print(f"（本次日志：{log_path}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
