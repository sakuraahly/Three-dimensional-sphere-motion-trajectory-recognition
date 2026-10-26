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
from apps._common import (
    add_stereo_source_args,
    log_environment,
    open_pair_source,
    setup_logging,
)
from calib import AutoPairCollector, BoardSpec, CalibConfig, StereoRig, calibrate_stereo
from calib import detect_board
from calib.relative_pose import compose, invert, rotation_angle_deg
from calib.stereo_calib import board_pose_from_points, calibrate_mono_points
from stereo.fov_align import (
    estimate_fov_alignment,
    fov_confidence_problem,
    fov_gate_action,
)
from stereo.stereo_mode import optical_axes_angle_deg, resolve_auto
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


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _list_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in _IMAGE_SUFFIXES else []
    if not path.is_dir():
        return []
    return sorted(p for p in path.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)


def _detect_two_boards(gray: np.ndarray, spec: BoardSpec):
    """一张图里检出**两块**同名标定板（L 形双板靶标）。

    做法：先检第一块，把它占的凸包**填白**（棋盘的黑格被抹掉，检测器就不会
    在同一位置再报一次），再检第二块。返回 `(obs_a, obs_b)` 或 `None`。
    这里**不做 A/B 语义分配** —— 谁是被左相机看到的 A，由调用方用几何一致性判定。
    """
    first = detect_board(gray, spec, exhaustive=True)
    if not first.found or first.corners is None:
        return None
    masked = gray.copy()
    pts = first.corners.reshape(-1, 2).astype(np.float32)
    center = pts.mean(axis=0)
    grown = ((pts - center) * 1.12 + center).astype(np.int32)   # 外扩一点，避免留下边
    cv2.fillConvexPoly(masked, cv2.convexHull(grown), 255)
    second = detect_board(masked, spec, exhaustive=True)
    if not second.found or second.corners is None:
        return None
    if float(np.linalg.norm(first.corners.reshape(-1, 2)
                            - second.corners.reshape(-1, 2))) < 1.0:
        return None                                    # 同一块被检了两次
    return first, second


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
                "  请改用：--left-dir <A 板目录> --right-dir <B 板目录> "
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
    if not args.pair_frame:
        missing.append("--pair-frame <单相机**同时**拍到 A、B 两板的图或目录>")
    if not args.pair_camera:
        missing.append("--pair-camera left|right（同框图是哪一路拍的）")
    if missing:
        return ("chain 需要三样输入（本轮只支持离线目录）：\n"
                "  1) 左相机看 A 板的图像  -> --left-dir\n"
                "  2) 右相机看 B 板的图像  -> --right-dir（或 --board-b-dir）\n"
                "  3) 一台相机同时看到 A、B 两板的同框图 -> --pair-frame + --pair-camera\n"
                "  缺少：" + "；".join(missing) + "\n"
                "  完整用法：python apps/calibrate_stereo.py --method chain "
                "--left-dir imgs/A --right-dir imgs/B --pair-frame imgs/pair "
                "--pair-camera left --stereo-mode angled\n"
                "  （没有同框图就无法求 T_AB，外参在数学上不可解——本程序不会猜一个值）")
    return None


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
        "board": spec.describe(),
        "source": "dir",
        "method": "chain",
        # 单板模式：内参段（板可移动）+ 外参段（板不动的参考图各一组），T_AB = I。
        "same_board": bool(getattr(args, "same_board", False)),
        "ref_left": str(getattr(args, "ref_left", "") or "") or None,
        "ref_right": str(getattr(args, "ref_right", "") or "") or None,
        "rectification": res.rectification,
        "left": args.left_dir,
        "right": args.board_b_dir or args.right_dir,
        "pair_frame": args.pair_frame,
        "pair_camera": args.pair_camera,
        "image_size": list(image_size),
        # 模式 = 方法决定：none_raw 的产物只可能走 angled 重建。
        "stereo_mode": "angled",
        "angle_deg": float(rig_angle(res.R)),
    }


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


def _chain_report(res, same_board: bool, ref_note: str = "") -> str:
    """chain 的报告文本。

    单板模式下 `T_AB` 按定义是 `I`，报告里"两板间距 / T_AB 重复性"这两行
    既没有测量对象也没有意义（会印出 `0.00 mm`，读起来像"两板贴在一起"），
    所以在这里如实换成 `T_AB = I（单板）`。同理"T_AB 只测了 1 次 → 建议拍
    3~5 张同框图"也一并抹掉：单板模式压根不许给 --pair-frame，照着做会撞互斥。
    核心层报告不改 —— 双板模式的那两行是它唯一的自查证据。
    """
    text = res.report()
    if not same_board:
        return text
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
    required = [("--left-dir", left_paths), (right_label, right_paths)]
    if same_board:
        # 外参段：两路各一组"板不动"的参考图
        required += [("--ref-left", ref_left_paths),
                     ("--ref-right", ref_right_paths)]
    else:
        required.append(("--pair-frame", pair_paths))
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
        # 同框图的参考姿态簇：由 --pair-camera 那一路的目录决定
        ref_mono = mono_l if args.pair_camera == "left" else mono_r
        ref_rots = [cv2.Rodrigues(r)[0] for r in ref_mono["rvecs"]]
        K_pair, D_pair = ref_mono["K"], ref_mono["D"]
        pair_cam_size = size_left if args.pair_camera == "left" else size_right

        samples, pair_rows = [], []
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
            try:
                T1, e1 = _pnp_reproj_rms(objp, found[0].corners, K_pair, D_pair)
                T2, e2 = _pnp_reproj_rms(objp, found[1].corners, K_pair, D_pair)
                T_A, T_B, d_a, d_b = _assign_ab(T1, T2, ref_rots, args.pair_camera)
            except ValueError as exc:
                print(f"\n[错误] 同框图 {p.name} 无法分配 A/B 板：{exc}")
                log.error("chain：A/B 分配失败 %s", exc)
                return 1
            T_ab = relative_pose.board_to_board_from_single_camera(T_A, T_B)
            samples.append(T_ab)
            pair_rows.append((p.name, e1, e2, d_a, d_b))

        if not samples:
            print("\n[错误] --pair-frame 里没有一张图能**同时**检出 A、B 两块板。\n"
                  "  下一步：把双板靶标放到 --pair-camera 那一路的画面中央、光照均匀、"
                  "两块板都完整入画，重拍 3~5 张。")
            log.error("chain：没有可用的同框图")
            return 1
        for name, e1, e2, d_a, d_b in pair_rows:
            print(f"  同框图 {name}: 重投影 {e1:.2f}/{e2:.2f} px，"
                  f"到参考姿态簇 {d_a:.1f}°/{d_b:.1f}°")
        cons = relative_pose.consistency(samples)
        print(f"[chain] T_AB：n={cons['n']}  旋转最大偏离 {cons['rot_max_deg']:.3f}°  "
              f"平移最大偏离 {cons['trans_max_mm']:.3f} mm  两板间距 "
              f"{cons['baseline_mm']:.1f} mm")

    # 链上用的两路板位姿。
    # 双板：A/B 是**两块不同的板**，板框随板走，所以挑各自重投影最小的那一帧
    # 都还是"该板在某一位姿下"的位姿，链式外参不受影响。
    # 单板：两路必须用**同一个物理摆放**（--ref-*），绝不能各挑一张移动中的图，
    # 否则 T_AB ≠ I，链出来的外参是错的 —— 上面已单独算好，这里不能再覆盖。
    if not same_board:
        i = int(np.argmin(mono_l["per_view"]))
        j = int(np.argmin(mono_r["per_view"]))
        T_camL_A = relative_pose.rt_to_T(cv2.Rodrigues(mono_l["rvecs"][i])[0],
                                         mono_l["tvecs"][i])
        T_camR_B = relative_pose.rt_to_T(cv2.Rodrigues(mono_r["rvecs"][j])[0],
                                         mono_r["tvecs"][j])

    # chain 的产物是 none_raw（未做极线校正）的链式外参，只可能用 angled 重建。
    # 以前这里跟 auto 走：新装置没有旧 JSON 时 auto 落到 parallel，于是把
    # meta["stereo_mode"] 写成 parallel，而本函数下面已算出 angle_deg=90 ——
    # 下游按 parallel 跑，dy≤2.5px 门限把**全部候选**拒掉且不报错。
    mode = "angled"
    print("[模式] chain → stereo_mode=angled（由方法决定，不跟 auto）")
    log.info("chain -> stereo_mode=angled（由方法决定，不跟 auto）")
    cfg = CalibConfig(method="chain", expected_baseline_mm=args.baseline_mm,
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

    report = _chain_report(res, same_board, ref_note if same_board else "")
    print(report)
    log.info("chain 结果：\n%s", report)
    if same_board and rig_angle(res.R) > 60.0:
        print("[警告] 注意：单板模式在 >60° 的装置上要求两路各自都能看到**整块**板；"
              "请确认每路检出的是完整棋盘，而不是只看到一角。")
        log.warning("单板模式 + 夹角 %.1f° > 60°：需确认每路检出完整棋盘", rig_angle(res.R))
    if not res.ok:
        print("标定未通过质量判定，未保存结果。")
        log.warning("chain 标定未通过质量判定，未保存")
        return 1

    rig = StereoRig.from_result(res, meta=_chain_meta(args, spec, res, size_left))
    rig.quality["method"] = "chain"
    rig.quality["rectification"] = res.rectification
    rig.quality["consistency"] = res.consistency
    if not same_board:
        rig.quality["t_ab_mm"] = res.consistency.get("baseline_mm", float("nan"))
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
    rig.quality["verdict"] = res.verdict
    print(f"[chain] 实测光轴夹角 {rig.angle_deg:.2f}°（由链式 R 计算，不写死 90°）")
    if rig.rectification == "none_raw":
        # 必须在报告里明说：这份参数**不能**拿去喂视差逻辑。
        # 不说的话，下游会看到 R1/R2/P1/P2 齐全就以为"已校正"，
        # 而 none_raw 下 y 差/视差都没有几何意义（见 calib/params.py 的说明）。
        print(f"[chain] {NONE_RAW_NOTE}")
        log.warning("chain 产物为 none_raw：%s", NONE_RAW_NOTE)
    npz, js = rig.save(args.out)
    print(f"已保存：{npz}")
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
    ap.add_argument("--method", choices=("joint", "chain"), default="joint",
                    help="外参求解方法：joint=两路联合求解（需充分共视，平行装置用）；"
                         "chain=独立单目 + 刚性双板公共参考（90° 用，不需要共视）。"
                         "angled + joint 且共视<8 对会被拒绝出数")
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
    ap.add_argument("--ref-left", default=None,
                    help="单板模式的外参锚点：左相机看这块板、**板不动**的参考图"
                         "（文件或目录，1~5 张）。板要整体入画")
    ap.add_argument("--ref-right", default=None,
                    help="单板模式的外参锚点：右相机看**同一块板同一个位置**的参考图"
                         "（文件或目录，1~5 张）。内参仍取 --right-dir 的结果")
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
    ap.add_argument("--out", default="config/stereo_calibration", help="输出前缀")
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
    args = ap.parse_args()
    if args.skip_fov and args.require_fov:
        ap.error("--skip-fov 与 --require-fov 矛盾：前者不做 FOV 对齐，后者要求"
                 "「估不出来就失败」")
    # chain 产出的是 none_raw（未做极线校正）的链式外参；parallel 重建依赖
    # 校正后的极线与视差（`stereo/triangulate.py` 的 dy 门限/视差范围），90° 下
    # 两者都不成立。允许标成 parallel = 让下游静默 0 配对，所以入口就拒绝。
    if args.method == "chain" and args.stereo_mode == "parallel":
        ap.error("chain 产出的是未校正标定（none_raw），不能标成 parallel："
                 "parallel 重建依赖校正后的极线与视差，90° 下不成立。")
    # 单板模式没有"两块板"：同框图/同框相机/B 板目录都失去测量对象。
    # 静默忽略它们最坏 —— 用户会以为自己给的同框图被用上了（T_AB 其实仍是 I）。
    if args.same_board and (args.pair_frame or args.pair_camera or args.board_b_dir):
        ap.error("--same-board 与 --pair-frame/--pair-camera/--board-b-dir 互斥："
                 "单板模式没有\"两块板\"可测，T_AB 直接取 I。")
    if args.same_board and args.method != "chain":
        ap.error("--same-board 只对 --method chain 有意义（joint 需要两路共视）；"
                 "请加 --method chain。")

    # 日志要在**做任何事之前**起来：标定失败时最有价值的证据（拒绝了什么、
    # 各拒了多少帧）现在只在内存里，进程一退就没了。
    log_path = setup_logging("calibrate")
    log = logging.getLogger("calibrate")
    print(f"本次日志：{log_path}")
    log.info("=== 双目标定开始 ===")
    log.info("参数：left=%s right=%s backend=%s %dx%d@%d 目标 %d 对（最少 %d）",
             args.left or args.left_video or args.left_dir,
             args.right or args.right_video or args.right_dir,
             args.backend, args.width, args.height, args.fps,
             args.target, args.min_pairs)
    log_environment(log)
    if args.relaxed:
        log.warning("启用了 --relaxed：门限放松，用精度换成功率")

    spec = BoardSpec(cols=args.cols, rows=args.rows, square_mm=args.square)
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
    if args.method == "chain":
        if args.same_board:
            print("方法：chain + 单板（独立单目 + 同一块固定板公共参考，T_AB=I）")
        else:
            print("方法：chain（独立单目 + 刚性双板公共参考，不需要两路共视）")
        log.info("method=chain same_board=%s", args.same_board)
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
            print("       请先确认两路相机都能出图："
                  "python apps/tools/cameras.py doctor")
            print("       然后重试；本次不会在没有画面的情况下继续标定。")
        else:
            print("       两路视频/图片序列没有同时有效的帧：检查左右文件是否成对、"
                  "长度是否一致。")
        log.error("启动阶段取不到成对帧，退出（%s）", src.kind)
        src.release()
        return 1

    if do_fov:
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
        # 手持预设。**每一条都是"用精度换成功率"**，所以逐条说清代价：
        cap_kw.update(
            stationary_max_px=10.0,   # 允许手更抖；代价=两路位姿差变大
            min_sharpness=400.0,      # 额外的绝对下限（默认值高于 SB 检出下限时才会真拦）
            min_area_ratio=0.008,     # 允许板更远；代价=单像素误差占比变大
            min_margin_px=6.0,        # 允许更贴边；代价=边缘畸变没被约束住
            min_brightness=25.0,
            max_brightness=245.0,
        )
        print("⚠ 已启用 --relaxed（手持预设）：门限都放松了，**这是用精度换成功率**。")
        print("   如果标定完 极线误差 > 0.3 px，或者代进去三维抖得厉害，")
        print("   请处理好光照与支撑、去掉 --relaxed 重标一次。")
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

    if not res.ok:
        print("标定未通过质量判定，未保存结果。请按上面的提示重拍。")
        log.warning("标定未通过质量判定，未保存")
        return 1

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
    rig = StereoRig.from_result(res, meta={
        "board": spec.describe(),
        "source": src.kind,
        "left": args.left if args.left is not None else (args.left_video or args.left_dir),
        "right": args.right if args.right is not None else (args.right_video or args.right_dir),
        "image_size": list(img_size),
        "fov": fov.to_dict() if fov is not None else None,
        "stereo_mode": mode,
        "angle_deg": measured_angle,
    })
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
                print("大角度质量判定为『不可用』，未保存结果；请调整夹角/标定板后重拍。")
                log.warning("angled 质量不可用，未保存")
                return 1
        except Exception as exc:                            # noqa: BLE001
            rig.quality["angled_grade"] = "未评估"
            rig.quality["verdict"] = f"angled 质量评估失败，未验证：{exc}"
            print(f"[错误] angled 质量评估失败，未保存：{exc}")
            log.warning("angled 质量评估失败，未保存：%s", exc)
            return 1
    npz, js = rig.save(args.out)
    print(f"已保存：{npz}")
    print(f"        {js}")
    print()
    print("下一步：")
    print("  - 三角测量时用 calib.StereoRig.load('config/stereo_calibration') 读取参数")
    print("  - 装置如果被碰动，用 apps/verify_stereo.py 复检极线误差")
    log.info("标定成功，已保存 %s", npz)
    print(f"（本次日志：{log_path}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
