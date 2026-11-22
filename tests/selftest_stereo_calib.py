"""双目标定自测：用"已知真值"的合成双目装置验证整条标定链路。

为什么要这个
------------
标定是双目方案的入口，错了后面全错，而且**在真实相机上很难发现**：
重投影误差看起来正常，三维却系统性偏。所以这里构造一个真值完全已知的
双目装置（内参、畸变、基线、相对旋转都已知），渲染出真实的标定板图像，
再让标定程序去算，最后比对：

    内参误差 / 基线误差 / 相对旋转误差 / 极线误差
    + 端到端三维精度（用标定结果三角测量已知三维点）

最后一项才是"能不能投入生产"的真正判据。

用法::

    python tests/selftest_stereo_calib.py
    python tests/selftest_stereo_calib.py --views 25 --noise 0.3 --distortion
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# --- 让脚本在任意目录深度都能找到项目根 ---
ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from calib import AutoPairCollector, BoardSpec, CalibConfig, CaptureConfig, calibrate_stereo
from calib.params import StereoRig


# --------------------------------------------------------------------------
# 合成双目装置
# --------------------------------------------------------------------------
def _K(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


# 畸变映射缓存（2026-10-27）。
#
# 这个映射**只依赖 (K, D, size, iters)，与图像内容无关**，而本测试会用同一台"相机"
# 渲染几十帧。原先每帧重算一遍 1280x720x12 次迭代 —— profile 显示
# `_forward_distort_map` 被调 242 次、**独占 172.6s**（占整个测试 211s 的 81%），
# 而真正做标定的 `calibrate_stereo` 连前 14 名都没进。
#
# 记忆化后结果**按位相同**（同一份映射表复用，不是近似），因此所有数值断言不受影响。
# 容量上限是为了防止"参数组合很多"时无界占内存（一张表约 7MB）。
_DISTORT_CACHE: dict[tuple, tuple] = {}
_DISTORT_CACHE_MAX = 12


def _forward_distort_map(K, D, size, iters: int = 12):
    """构造"把无畸变图变成畸变图"的 remap 映射（OpenCV 只提供反方向）。

    做法：对每个目标像素 q 迭代求解 p，使 forward(p) = q。
    径向畸变单调时几步就收敛。

    结果按 `(K, D, size, iters)` 缓存 —— 调用方**不得原地修改**返回的数组
    （它们是共享的；`cv2.remap` 只读，没问题）。
    """
    key = (
        np.asarray(K, dtype=np.float64).tobytes(),
        np.asarray(D, dtype=np.float64).ravel().tobytes(),
        tuple(size),
        int(iters),
    )
    cached = _DISTORT_CACHE.get(key)
    if cached is not None:
        return cached

    w, h = size
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    qx, qy = (u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1]
    d = np.zeros(5, dtype=np.float64)
    d[: min(5, len(D.ravel()))] = D.ravel()[:5]
    k1, k2, p1, p2, k3 = d
    px, py = qx.copy(), qy.copy()
    for _ in range(iters):
        r2 = px * px + py * py
        radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        fx_ = px * radial + 2 * p1 * px * py + p2 * (r2 + 2 * px * px)
        fy_ = py * radial + p1 * (r2 + 2 * py * py) + 2 * p2 * px * py
        px += qx - fx_
        py += qy - fy_
    out = ((px * K[0, 0] + K[0, 2]).astype(np.float32),
           (py * K[1, 1] + K[1, 2]).astype(np.float32))
    if len(_DISTORT_CACHE) >= _DISTORT_CACHE_MAX:      # 简单 FIFO 淘汰
        _DISTORT_CACHE.pop(next(iter(_DISTORT_CACHE)))
    _DISTORT_CACHE[key] = out
    return out


def make_board_image(
    spec: BoardSpec, px_per_m: float = 900.0
) -> tuple[np.ndarray, float]:
    """画一张标准棋盘格图像（作为贴图源）。

    注意棋盘格的方格数 = 内角点 + 1。黑色方格画成黑，白色画成白，
    并且四周留一点白边，避免 warp 之后边缘出现人工边界。
    """
    sq = spec.square_mm / 1000.0 * px_per_m
    cols, rows = spec.cols + 1, spec.rows + 1
    pad = int(sq)
    w = int(cols * sq) + 2 * pad
    h = int(rows * sq) + 2 * pad
    img = np.full((h, w), 255, np.uint8)
    for i in range(cols):
        for j in range(rows):
            if (i + j) % 2 == 0:
                x0 = int(pad + i * sq)
                y0 = int(pad + j * sq)
                cv2.rectangle(img, (x0, y0), (int(x0 + sq), int(y0 + sq)), 0, -1)
    return img, pad


def render_view(
    board_img,
    pad,
    spec: BoardSpec,
    px_per_m: float,
    rvec,
    tvec,
    K,
    D,
    size,
    noise: float,
    blur: float,
) -> np.ndarray:
    """把棋盘按给定位姿渲染到该相机视角（平面 -> 图像 是精确单应）。"""
    R, _ = cv2.Rodrigues(rvec)
    sq_m = spec.square_mm / 1000.0
    # 棋盘格第一格左上角在"板坐标系"里是 (-sq, -sq)（因为贴图留了 1 格白边）
    origin = np.array([-sq_m, -sq_m, 0.0])
    r1, r2 = R[:, 0], R[:, 1]
    t = np.asarray(tvec, np.float64).ravel()
    # 板米坐标 -> 无畸变图像
    Hm = K @ np.column_stack([r1, r2, R @ origin + t])
    S = np.diag([1.0 / px_per_m, 1.0 / px_per_m, 1.0])
    H = Hm @ S
    und = cv2.warpPerspective(
        board_img,
        H,
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    und = cv2.cvtColor(und, cv2.COLOR_GRAY2BGR)
    # 板上黑格之外是"背景"，warp 之后边界外是 0（黑），会把棋盘外的区域也变成黑，
    # 这里用亮的背景填充，更接近真实拍摄
    mask = cv2.warpPerspective(
        np.full_like(board_img, 255),
        H,
        size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    bg = 150
    und[mask == 0] = bg
    if D is not None and np.any(np.abs(D) > 1e-12):
        mx, my = _forward_distort_map(K, D, size)
        und = cv2.remap(
            und,
            mx,
            my,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(bg, bg, bg),
        )
    img = und.astype(np.float32)
    if blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    if noise > 0:
        img += np.random.default_rng(7).normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_poses(n: int, rng) -> list[tuple[np.ndarray, np.ndarray]]:
    """生成 n 个覆盖良好的标定板位姿（在左相机坐标系下）。"""
    poses = []
    for i in range(n):
        # 深度、横向、纵向按分层采样，保证全画面覆盖
        z = 0.45 + 0.35 * ((i % 5) / 4.0)
        x = -0.13 + 0.26 * (((i // 5) % 3) / 2.0)
        y = -0.10 + 0.20 * (((i // 15) % 2))
        tvec = np.array([x, y, z], np.float64) + rng.normal(0, 0.008, 3)
        # 倾斜：绕 x/y 各 ±22 度，绕 z 旋转 ±25 度
        ang = np.radians(
            [
                rng.uniform(-22, 22),
                rng.uniform(-22, 22),
                rng.uniform(-25, 25),
            ]
        )
        poses.append((ang, tvec))
    rng.shuffle(poses)
    return poses


# --------------------------------------------------------------------------
def run_case(
    views: int,
    noise: float,
    use_distortion: bool,
    blur: float,
    verbose: bool,
    stills: bool = False,
) -> bool:
    rng = np.random.default_rng(20240930)
    size = (1280, 720)
    spec = BoardSpec(cols=8, rows=5, square_mm=25.0)

    # ---- 真值 ----
    fx1, fy1 = 900.0, 900.0
    fx2, fy2 = 880.0, 882.0  # 两台相机故意不同（不同型号更接近现实）
    K1 = _K(fx1, fy1, 655.0, 372.0)
    K2 = _K(fx2, fy2, 628.0, 365.0)
    if use_distortion:
        D1 = np.array([-0.11, 0.045, 0.0012, -0.0009, 0.0])
        D2 = np.array([-0.09, 0.030, -0.0010, 0.0011, 0.0])
    else:
        D1 = np.zeros(5)
        D2 = np.zeros(5)
    baseline = 0.100  # 100 mm
    R_true, _ = cv2.Rodrigues(np.radians([0.8, -1.2, 0.5]))
    T_true = np.array([-baseline, 0.0015, 0.002], np.float64).reshape(3, 1)

    board_img, pad = make_board_image(spec)
    objp = spec.object_points()

    # ---- 渲染 + 自动挑帧 ----
    from calib import CaptureConfig

    # 独立静帧：帧间本来就是不同位姿，不该要求"静止"；
    # 视频：模拟"手持板走一段、停一下"，让静止门限去挑停稳的那几帧。
    cfg_cap = CaptureConfig(require_stationary=not stills)
    collector = AutoPairCollector(spec, cfg_cap)
    poses = make_poses(views, rng)

    # 渲染序列：每个目标位姿前面插几帧过渡（板在移动），最后两帧完全相同（停稳）
    seq: list[tuple[np.ndarray, np.ndarray, bool]] = []
    for i, (ang, tvec) in enumerate(poses):
        if not stills and i > 0:
            pa, pt = poses[i - 1]
            for f in (0.25, 0.5, 0.75):
                seq.append((pa + (ang - pa) * f, pt + (tvec - pt) * f, False))
            seq.append((ang, tvec, False))
        seq.append((ang, tvec, True))
    if stills:
        seq = [(a, t, True) for a, t, _ in seq]

    for i, (ang, tvec, _settled) in enumerate(seq):
        rvec1 = ang.reshape(3, 1)
        img1 = render_view(
            board_img, pad, spec, 900.0, rvec1, tvec, K1, D1, size, noise, blur
        )
        # 同一个板在相机 2 里的位姿：X2 = R X1 + T
        R_b1, _ = cv2.Rodrigues(rvec1)
        R_b2 = R_true @ R_b1
        t_b2 = (R_true @ tvec.reshape(3, 1) + T_true).reshape(3, 1)
        rvec2, _ = cv2.Rodrigues(R_b2)
        # 注意：px_per_m 是"棋盘贴图的分辨率"，两台相机必须用同一个值，
        # 它决定的是贴图比例而不是成像大小（成像大小由 K 和位姿决定）。
        # 写错会让另一台相机看到一个"尺寸不对"的板，导致焦距/基线系统性偏移。
        img2 = render_view(
            board_img, pad, spec, 900.0, rvec2, t_b2, K2, D2, size, noise, blur
        )

        collector.offer(
            cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY),
        )
        if verbose:
            print(
                f"    帧 {i + 1:3d}: 已收 {collector.count}  ({collector.last_reason})"
            )

    print(
        f"  自动挑帧：收下 {collector.count}/{views} 对"
        + (
            f"，主要剔除原因：{max(collector.rejected.items(), key=lambda kv: kv[1])[0]}"
            if collector.rejected
            else ""
        )
    )
    if collector.count < 8:
        print("  [失败] 自动挑帧收得太少，无法标定")
        return False

    objpoints = [objp] * collector.count
    ip1 = [p.corners_left for p in collector.pairs]
    ip2 = [p.corners_right for p in collector.pairs]

    # ---- 标定 ----
    cfg = CalibConfig(expected_baseline_mm=baseline * 1000.0)
    res = calibrate_stereo(objpoints, ip1, ip2, size, cfg, spec)
    print(res.report())

    # ---- 与真值比对 ----
    fx1_err = abs(res.K1[0, 0] - fx1) / fx1 * 100
    fy1_err = abs(res.K1[1, 1] - fy1) / fy1 * 100
    fx2_err = abs(res.K2[0, 0] - fx2) / fx2 * 100
    base_err = abs(res.baseline_mm - baseline * 1000.0) / (baseline * 1000.0) * 100
    # 两个旋转之间的测地角
    R_err = math.degrees(
        math.acos(float(np.clip((np.trace(res.R @ R_true.T) - 1) / 2, -1, 1)))
    )

    print("与真值对比：")
    print(f"  焦距 fx1 {res.K1[0, 0]:8.2f} (真值 {fx1:.2f}, 误差 {fx1_err:.3f}%)")
    print(f"  焦距 fy1 {res.K1[1, 1]:8.2f} (真值 {fy1:.2f}, 误差 {fy1_err:.3f}%)")
    print(f"  焦距 fx2 {res.K2[0, 0]:8.2f} (真值 {fx2:.2f}, 误差 {fx2_err:.3f}%)")
    print(
        f"  基线     {res.baseline_mm:8.3f} mm (真值 {baseline * 1000:.3f} mm, 误差 {base_err:.3f}%)"
    )
    print(f"  相对旋转误差 {R_err:.4f} 度")

    # ---- 端到端三维精度（最关键的一项）----
    # 用真值装置把一组已知三维点投影到两路，加检测噪声，
    # 再用【标定出来的】参数三角测量，看恢复出的三维误差。
    pts3d = np.array(
        [
            [0.00, 0.00, 0.60],
            [0.06, 0.02, 0.55],
            [-0.05, 0.03, 0.70],
            [0.02, -0.04, 0.50],
            [-0.07, -0.02, 0.65],
            [0.04, 0.05, 0.75],
        ],
        np.float64,
    )
    rng2 = np.random.default_rng(11)
    uv1 = cv2.projectPoints(pts3d, np.zeros(3), np.zeros(3), K1, D1)[0].reshape(-1, 2)
    uv2 = cv2.projectPoints(pts3d, cv2.Rodrigues(R_true)[0], T_true, K2, D2)[0].reshape(
        -1, 2
    )
    uv1 += rng2.normal(0, 0.2, uv1.shape)
    uv2 += rng2.normal(0, 0.2, uv2.shape)

    a = cv2.undistortPoints(uv1.reshape(-1, 1, 2), K1, D1, R=res.R1, P=res.P1).reshape(
        -1, 2
    )
    b = cv2.undistortPoints(uv2.reshape(-1, 1, 2), K2, D2, R=res.R2, P=res.P2).reshape(
        -1, 2
    )
    pts4d = cv2.triangulatePoints(res.P1, res.P2, a.T, b.T)
    got = (pts4d[:3] / pts4d[3]).T
    # 关键：三角测量得到的点在【校正后】的左相机坐标系里，
    # 而真值是在【原始】左相机系 —— 两者相差立体校正的旋转 R1。
    # 忘了这一步会看到一个几十毫米的"系统性误差"，其实是坐标系没换算。
    truth_rect = (res.R1 @ pts3d.T).T
    err = np.linalg.norm(got - truth_rect, axis=1)
    dist = np.linalg.norm(pts3d, axis=1)
    print(
        f"  端到端三维误差：均值 {err.mean() * 1000:.3f} mm，最大 {err.max() * 1000:.3f} mm "
        f"（目标距离 {dist.min():.2f}-{dist.max():.2f} m）"
    )

    # ---- 判定 ----
    checks = [
        (f"极线误差 <= 0.30 px", res.epipolar_rms <= 0.30),
        (f"重投影 RMS <= 0.30 px", res.rms <= 0.30),
        (f"焦距误差 <= 0.5%", max(fx1_err, fy1_err, fx2_err) <= 0.5),
        (f"基线误差 <= 1.0%", base_err <= 1.0),
        (f"相对旋转误差 <= 0.15 度", R_err <= 0.15),
        (f"三维误差均值 <= 8 mm", err.mean() * 1000 <= 8.0),
    ]
    print()
    ok = True
    for name, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {name}")
        ok = ok and passed
    return ok


def test_T_sign_convention() -> bool:
    """`T` 的符号约定：**右相机物理在右侧 ⟹ `T[0] < 0`**（合成真值判据）。

    ## 守的是一个"每台正常装置都会报假警报"的真缺陷（2026-10-18 修）

    原实现 `baseline_side()` 认为 `T[0] > 0` 才是"右相机在右边"，于是
    `StereoRig.describe()` 对**每一台正常装置**都打印
    「[警告] 标定时 --left/--right 可能接反了」。假警报会训练人忽略真警报。

    ## 为什么必须走"真标定求解器"而不是只测 `baseline_side()`

    只测 `baseline_side()` 只能证明它和当前实现自洽 —— 而错误恰恰在
    "这个符号跟 OpenCV 的约定对不对得上"。所以这里**造真值**：
    右相机放在左相机 `+X`（物理右侧），用 `cv2.projectPoints` 生成角点，
    再交给 `calibrate_stereo()` 反解，看它解出的 `T[0]` 是正还是负。

    判据全部对照**真值**，不引用被测函数的中间量。
    """
    print("\n----- T 符号约定（合成真值） -----")
    W, H = 1280, 720
    f = 1000.0
    B = 0.065
    K = _K(f, f, W / 2, H / 2)
    D = np.zeros(5)
    spec = BoardSpec(square_mm=25.0)
    objp = spec.object_points().astype(np.float64)

    rng = np.random.default_rng(7)
    img1, img2 = [], []
    for i in range(14):
        rv = np.array([rng.uniform(-0.30, 0.30), rng.uniform(-0.30, 0.30),
                       rng.uniform(-0.25, 0.25)])
        tv = np.array([rng.uniform(-0.10, 0.10), rng.uniform(-0.07, 0.07),
                       0.45 + 0.05 * (i % 5)])
        Rm, _ = cv2.Rodrigues(rv)
        P_l = (Rm @ objp.T).T + tv.reshape(1, 3)
        # 真值：右光心在左相机系里的位置 C = (+B, 0, 0)（物理右侧）
        P_r = P_l - np.array([B, 0.0, 0.0])
        p1, _ = cv2.projectPoints(P_l, np.zeros(3), np.zeros(3), K, D)
        p2, _ = cv2.projectPoints(P_r, np.zeros(3), np.zeros(3), K, D)
        img1.append(p1.astype(np.float32))
        img2.append(p2.astype(np.float32))

    res = calibrate_stereo(objp, img1, img2, (W, H), CalibConfig())
    t = np.ravel(res.T)
    print(f"  真值：右光心在左相机 +X（物理右侧），B={B * 1000:.0f} mm")
    print(f"  解出：T = [{t[0]:+.6f} {t[1]:+.6f} {t[2]:+.6f}]   "
          f"基线 {np.linalg.norm(t) * 1000:.2f} mm  rms {res.rms:.4f} px")
    rig = StereoRig.from_result(res)
    side, sign = rig.baseline_side(), rig.disparity_sign()
    print(f"  baseline_side()={side!r}  disparity_sign()={sign:+d}")

    # 真值视差符号（无畸变、R≈I，直接看 x 差）
    d_true = float((img1[0][:, 0, 0] - img2[0][:, 0, 0]).mean())
    print(f"  真值视差 x_left-x_right = {d_true:+.2f} px")

    checks = [
        ("解出的 T[0] 为**负**（符合 X_right = R·X_left + T）", float(t[0]) < 0),
        ("基线长度与真值一致（±2%）", abs(np.linalg.norm(t) - B) / B < 0.02),
        ("baseline_side() 判为 'right'（物理右侧）", side == "right"),
        ("disparity_sign() 为 +1", sign == 1),
        ("与真值视差符号一致", np.sign(d_true) == sign),
        ("describe() **不再**对正常装置报「可能接反了」",
         "可能接反" not in rig.describe()),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")

    # 镜像 check：把左右对调，必须**报**接反
    img1b, img2b = img2, img1
    res_b = calibrate_stereo(objp, img1b, img2b, (W, H), CalibConfig())
    rig_b = StereoRig.from_result(res_b)
    hit = "可能接反" in rig_b.describe()
    print(f"  镜像（左右对调）：baseline_side()={rig_b.baseline_side()!r}  "
          f"describe 报警={hit}")
    checks.append(("左右对调后 describe() **确实**报警（不是永远不报）", hit))
    checks.append(("左右对调后 disparity_sign() 变为 -1", rig_b.disparity_sign() == -1))

    return all(ok for _n, ok in checks)


def test_board_area_caps_grade() -> bool:
    """板占比必须能**卡住**等级，且不确定度要真的算出来（审查 A5）。

    ## 守的是什么（2026-10-18）

    合成真值 + 0.30 px 亚像素噪声，**同样的 RMS**：

        板占 1.9%（只在中央）：RMS 0.414 px   Z 误差 σ 5.76 mm  最大 13.7 mm
        板占 7.9%（铺满画面）：RMS 0.413 px   Z 误差 σ 1.40 mm  最大  2.8 mm

    **RMS 对"板太小"几乎免疫，而深度精度差 4 倍。**
    所以"RMS/极线都达标"**不足以**判"优秀"，必须再卡板占比；
    同时要把 `stdDeviationsIntrinsics`（原来被丢弃）接出来，
    让"参数有多可信"有数字可看。

    ## 判据

    1. 板小 + 指标完美 → **不得判"优秀"**（最高"可用"），且给出原因；
    2. 板大 + 指标完美 → 判"优秀"（不是把"优秀"这一档废掉）；
    3. `std1` 非空、`fx_sigma()` 有限且 > 0；
    4. `depth_sigma_mm()` 随距离增大而增大（单调），且在 0.6 m 处量级合理。
    """
    print("\n----- 板占比对判级的约束（审查 A5） -----")
    W, H = 1280, 720
    f = 1000.0
    B = 0.065
    K = _K(f, f, W / 2, H / 2)
    D = np.zeros(5)
    spec = BoardSpec(square_mm=25.0)
    objp = spec.object_points().astype(np.float64)

    def make(z: float, n: int = 14, seed: int = 3):
        rng = np.random.default_rng(seed)
        img1, img2 = [], []
        for i in range(n):
            rv = np.array([rng.uniform(-0.25, 0.25), rng.uniform(-0.25, 0.25),
                           rng.uniform(-0.20, 0.20)])
            tv = np.array([rng.uniform(-0.05, 0.05), rng.uniform(-0.04, 0.04),
                           z + 0.03 * (i % 4)])
            Rm, _ = cv2.Rodrigues(rv)
            P_l = (Rm @ objp.T).T + tv.reshape(1, 3)
            P_r = P_l - np.array([B, 0.0, 0.0])
            p1, _ = cv2.projectPoints(P_l, np.zeros(3), np.zeros(3), K, D)
            p2, _ = cv2.projectPoints(P_r, np.zeros(3), np.zeros(3), K, D)
            img1.append(p1.astype(np.float32))
            img2.append(p2.astype(np.float32))
        return img1, img2

    img1_s, img2_s = make(z=1.05)      # 板小
    img1_b, img2_b = make(z=0.38)      # 板大
    # 第三组：**带噪声**。零噪声下不确定度恒为 0（物理上正确），
    # 所以"量级是否合理"必须用有噪声的数据测，否则判据本身不成立。
    img1_n, img2_n = make(z=0.75, seed=11)
    rng = np.random.default_rng(2024)
    img1_n = [(p + rng.normal(0, 0.3, p.shape)).astype(np.float32) for p in img1_n]
    img2_n = [(p + rng.normal(0, 0.3, p.shape)).astype(np.float32) for p in img2_n]
    cfg = CalibConfig()
    res_s = calibrate_stereo(objp, img1_s, img2_s, (W, H), cfg)
    res_b = calibrate_stereo(objp, img1_b, img2_b, (W, H), cfg)
    res_n = calibrate_stereo(objp, img1_n, img2_n, (W, H), cfg)

    print(f"  小：板占 {res_s.board_area * 100:.1f}%  rms {res_s.rms:.4f} px  "
          f"判决 {res_s.grade}")
    print(f"  大：板占 {res_b.board_area * 100:.1f}%  rms {res_b.rms:.4f} px  "
          f"判决 {res_b.grade}")

    print(f"  带噪声：rms {res_n.rms:.4f} px  σ(fx)={res_n.fx_sigma():.2f} px  "
          f"σ_Z(0.6m)={res_n.depth_sigma_mm(0.6):.2f} mm  "
          f"σ_Z(1.2m)={res_n.depth_sigma_mm(1.2):.2f} mm")

    checks = [
        ("小板确实落在门限之下（测试前提成立）",
         res_s.board_area < cfg.min_board_area_for_good),
        ("大板确实落在门限之上（测试前提成立）",
         res_b.board_area >= cfg.min_board_area_for_good),
        ("小板 + 指标完美 **不得判优秀**",
         res_s.grade != "优秀"),
        ("小板给出「最高只判可用」的原因说明",
         any("最高只判" in w for w in res_s.warnings)),
        ("大板 + 指标完美 **仍判优秀**（不是把这一档废掉）",
         res_b.grade == "优秀"),
        ("std1 被保留下来（原来被丢弃）",
         res_n.std1 is not None and np.size(res_n.std1) >= 4),
        ("fx_sigma() 有限且 > 0（带噪声数据）",
         np.isfinite(res_n.fx_sigma()) and res_n.fx_sigma() > 0),
        ("depth_sigma_mm() 随距离单调增大",
         res_n.depth_sigma_mm(0.6) < res_n.depth_sigma_mm(1.2)),
        ("0.6 m 处深度不确定度在 0.5~50 mm 量级（合理性）",
         0.5 < res_n.depth_sigma_mm(0.6) < 50.0),
        ("零噪声时不确定度趋于 0（不虚报）",
         not np.isfinite(res_s.depth_sigma_mm(0.6))
         or res_s.depth_sigma_mm(0.6) < 1.0),
        ("报告里出现了「深度不确定度」这一行",
         "深度不确定度" in res_n.report()),
        ("报告里出现了「板占画面」这一行", "板占画面" in res_n.report()),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)



def test_epipolar_rejects_mismatch() -> bool:
    """左右角点数不一致时必须**报错**，不许静默截断（审查 A8）。

    旧代码 
 = min(len(p1), len(p2)) 后截断再比 —— 等于拿**错配的点对**算极线误差，
    把一个真实故障（角点漏检/配错对）掩盖成"误差稍微大一点"。
    """
    print("\n----- 极线误差：角点数不一致必须报错 -----")
    from calib.stereo_calib import _epipolar_errors

    K = _K(1000.0, 1000.0, 640.0, 360.0)
    D = np.zeros(5)
    R = np.eye(3)
    P = np.hstack([K, np.zeros((3, 1))])
    rng = np.random.default_rng(0)
    a = (rng.normal(size=(10, 2)) * 100 + 600).astype(np.float32)
    b = (rng.normal(size=(8, 2)) * 100 + 600).astype(np.float32)

    checks = []
    try:
        out = _epipolar_errors([a], [a], K, D, K, D, R, R, P, P)
        checks.append(("点数一致时正常算出误差", out.shape == (1,) and np.isfinite(out[0])))
    except Exception as exc:                              # noqa: BLE001
        checks.append((f"点数一致时抛了 {type(exc).__name__}: {exc}", False))

    try:
        _epipolar_errors([a], [b], K, D, K, D, R, R, P, P)
        checks.append(("点数不一致 -> **必须报错**（不许静默截断）", False))
    except ValueError as exc:
        ok = "不一致" in str(exc)
        print(f"  报错：{str(exc).splitlines()[0][:78]}")
        checks.append(("点数不一致 -> **必须报错**（不许静默截断）", ok))
    except Exception as exc:                              # noqa: BLE001
        checks.append((f"点数不一致 -> 抛了非 ValueError：{type(exc).__name__}", False))

    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_fov_or_none() -> bool:
    """FOV **估不出来** 与 **估出来不可信** 必须分开处理（2026-10-24）。

    真实故障：parallel 标定在启动帧上 SIFT/ORB 都匹配不够（纯色墙/共同纹理太少），
    旧代码直接 `return 1`，用户看到的就是"标定打不开"。但 `stereoCalibrate`
    并不要求两路视场对齐 —— 估不出来应当**降级为不裁剪、按原始画面继续**，
    同时把原始原因带出来打印，不能吞掉。
    """
    print("\n----- FOV 估计失败：默认降级 / --require-fov 仍失败 -----")
    from unittest import mock

    import apps.calibrate_stereo as cs

    reason_text = "sift: SIFT 特征 2474/5000，ratio+交叉校验后匹配 14 < min_inliers=30"
    boom = mock.Mock(side_effect=ValueError(reason_text))
    checks = []

    fov, reason = cs._fov_or_none(boom, mode="parallel", require_fov=False)
    print(f"  parallel 默认：fov={fov!r}  reason={reason}")
    checks.append(("parallel 默认降级为 fov=None（不抛异常）", fov is None))
    checks.append(("降级时带回**原始原因**（ValueError + 细节）",
                   isinstance(reason, str) and reason_text in reason
                   and "ValueError" in reason))

    for label, kw in (
        ("--require-fov 时仍然失败（不降级）", dict(mode="parallel", require_fov=True)),
        ("非 parallel（angled）不静默降级", dict(mode="angled", require_fov=False)),
    ):
        try:
            cs._fov_or_none(boom, **kw)
            checks.append((label, False))
        except ValueError as exc:
            checks.append((f"{label}（原因：{str(exc)[:40]}…）", reason_text in str(exc)))
        except Exception as exc:                          # noqa: BLE001
            checks.append((f"{label}（抛了非 ValueError：{type(exc).__name__}）", False))

    sentinel = object()
    got, why = cs._fov_or_none(lambda: sentinel, mode="parallel", require_fov=True)
    checks.append(("估计成功时原样返回 (fov, None)",
                   got is sentinel and why is None))

    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_chain_extrinsics_90deg() -> bool:
    """**证伪实验**：90° 装置用"独立单目 + 刚性双板公共参考"恢复外参。

    为什么必须是数值实验（§11.7 Q3）
    --------------------------------
    `stereoCalibrate` 的前提是两路共视。90° 装置共视几乎为零，联合求解会
    给出**好看但无意义**的解。这里构造真值已知的 90° 装置 + L 形双板：

        T_{C_R←C_L} = T_{C_R←B} · T_{B←A} · T_{A←C_L}

    用**带噪声的单目位姿**走链，与真值比对；再把同一批数据喂给
    `method="joint"` 且只有 3 对共视 → 必须被护栏**拒绝**（抛 ValueError），
    证明"少数据联合求解"这条路确实被堵住，而不是悄悄给出数。

    另外做两件"反证测试本身能证伪"的事：
      1. 把 `T_AB` 取逆（方向错误）→ 误差必须爆掉（否则门限形同虚设）；
      2. 8 对共视时护栏**不**拦截（否则它只是无差别禁止）。
    """
    print("\n----- 90° 链式外参（合成真值 + 证伪） -----")
    import calib.relative_pose as rp
    from calib.stereo_calib import (
        MIN_COVISIBLE_PAIRS_FOR_JOINT_ANGLED,
        calibrate_mono_points,
    )

    W, H = 1280, 720
    K_L = _K(900.0, 900.0, 640.0, 360.0)
    K_R = _K(980.0, 982.0, 610.0, 352.0)
    D = np.zeros(5)
    spec = BoardSpec(cols=8, rows=5, square_mm=25.0)
    objp = spec.object_points()

    # ---- 真值：90° 装置 ----
    # 右相机在 (0.5, 0, 0.5)，光轴沿 -X；左相机在原点，光轴 +Z。
    R_true = cv2.Rodrigues(np.radians([0.0, 90.0, 0.0]))[0]
    C = np.array([0.50, 0.0, 0.50])
    T_true = (-R_true @ C).reshape(3, 1)
    T_cr_cl_true = rp.rt_to_T(R_true, T_true)

    # ---- L 形双板：A 面对左相机（+Z），B 在 A 右侧转 90°（面朝 +X）----
    T_wA = rp.rt_to_T(np.eye(3), [0.0, 0.0, 0.50])
    R_wB = cv2.Rodrigues(np.radians([0.0, 90.0, 0.0]))[0]
    T_wB = rp.rt_to_T(R_wB, [0.20, 0.0, 0.50])
    T_BA_true = rp.compose(rp.invert(T_wB), T_wA)      # A 系 -> B 系

    rng = np.random.default_rng(4242)
    rng_px = np.random.default_rng(77)      # 角点像素噪声（0.3 px）
    n_views = 12
    img_l, img_r = [], []
    T_cl_A, T_cr_B = [], []
    for i in range(n_views):
        dR = cv2.Rodrigues(np.radians(rng.uniform(-7, 7, 3)))[0]
        dt = rng.uniform(-0.012, 0.012, 3)
        T_cl_A_i = rp.rt_to_T(dR @ T_wA[:3, :3], T_wA[:3, 3] + dt)
        T_cr_B_i = rp.compose(T_cr_cl_true, T_cl_A_i, rp.invert(T_BA_true))
        T_cl_A.append(T_cl_A_i)
        T_cr_B.append(T_cr_B_i)

        r_l = cv2.Rodrigues(T_cl_A_i[:3, :3])[0]
        p_l, _ = cv2.projectPoints(objp, r_l, T_cl_A_i[:3, 3], K_L, D)
        r_r = cv2.Rodrigues(T_cr_B_i[:3, :3])[0]
        p_r, _ = cv2.projectPoints(objp, r_r, T_cr_B_i[:3, 3], K_R, D)
        p_l = p_l + rng_px.normal(0, 0.3, p_l.shape)
        p_r = p_r + rng_px.normal(0, 0.3, p_r.shape)
        img_l.append(p_l.astype(np.float32))
        img_r.append(p_r.astype(np.float32))

    # ---- 单目位姿（带 0.03° / 0.2 mm 噪声）→ 链式恢复 ----
    def _perturb(T, r):
        dR = cv2.Rodrigues(np.radians(r.normal(0, 0.03, 3)))[0]
        out = T.copy()
        out[:3, :3] = dR @ T[:3, :3]
        out[:3, 3] = T[:3, 3] + r.normal(0, 0.0002, 3)
        return out

    rot_err, trans_err = [], []
    for trial in range(20):
        r = np.random.default_rng(900 + trial)
        TclA = _perturb(T_cl_A[0], r)
        TcrB = _perturb(T_cr_B[0], r)
        TcA = _perturb(T_cl_A[1], r)
        TcB = _perturb(rp.compose(T_cl_A[1], rp.invert(T_BA_true)), r)
        T_ab = rp.board_to_board_from_single_camera(TcA, TcB)
        R_est, T_est, _ = rp.relative_extrinsics(TclA, T_ab, TcrB)
        rot_err.append(rp.rotation_angle_deg(R_est, R_true))
        trans_err.append(float(np.linalg.norm(T_est.ravel() - T_true.ravel())) * 1000.0)
    print(f"  20 次链式恢复：旋转误差 最大 {max(rot_err):.4f}° / 均值 "
          f"{np.mean(rot_err):.4f}°；平移误差 最大 {max(trans_err):.3f} mm / 均值 "
          f"{np.mean(trans_err):.3f} mm")
    print(f"  真值基线 {np.linalg.norm(T_true) * 1000:.2f} mm")

    # 反证 1：T_AB 方向取逆 -> 误差必须比正确链大一个量级
    r = np.random.default_rng(1)
    T_ab_bad = rp.invert(rp.compose(T_cl_A[1], rp.invert(T_BA_true)))
    R_bad, T_bad, _ = rp.relative_extrinsics(T_cl_A[0], T_ab_bad, T_cr_B[0])
    bad_rot = rp.rotation_angle_deg(R_bad, R_true)
    bad_trans = float(np.linalg.norm(T_bad.ravel() - T_true.ravel())) * 1000.0
    print(f"  反证（T_AB 取逆）：旋转误差 {bad_rot:.2f}°  平移误差 {bad_trans:.1f} mm"
          f"（正确链为 {max(rot_err):.2f}° / {max(trans_err):.1f} mm）")

    # ---- 用真实角点跑一遍 chain，验证产物形状/字段 ----
    cfg_chain = CalibConfig(method="chain")
    mono_l = calibrate_mono_points(objp, img_l, (W, H), CalibConfig())
    mono_r = calibrate_mono_points(objp, img_r, (W, H), CalibConfig())
    T_camL_A = rp.rt_to_T(cv2.Rodrigues(mono_l["rvecs"][0])[0], mono_l["tvecs"][0])
    T_camR_B = rp.rt_to_T(cv2.Rodrigues(mono_r["rvecs"][0])[0], mono_r["tvecs"][0])
    T_ab_real = rp.board_to_board_from_single_camera(
        _perturb(T_cl_A[0], np.random.default_rng(5)),
        _perturb(T_cl_A[0] @ rp.invert(T_BA_true), np.random.default_rng(6)))
    Tab_2 = _perturb(T_ab_real, np.random.default_rng(7))
    res = calibrate_stereo(objp, img_l, img_r, (W, H), cfg_chain, spec,
                           stereo_mode="angled", T_camL_A=T_camL_A,
                           T_AB=T_ab_real, T_camR_B=T_camR_B,
                           T_AB_samples=[T_ab_real, Tab_2])
    R_e2e = rp.rotation_angle_deg(res.R, R_true)
    base_e2e = abs(res.baseline_mm - np.linalg.norm(T_true) * 1000.0)
    print(f"  chain 端到端：旋转误差 {R_e2e:.4f}°  基线误差 {base_e2e:.3f} mm  "
          f"判定 {res.grade}  rectification={res.rectification}")
    print(f"  T_AB 重复性：n={res.consistency['n']} "
          f"旋转 {res.consistency['rot_max_deg']:.4f}° "
          f"平移 {res.consistency['trans_max_mm']:.4f} mm")

    # ---- 2026-11-09：双板模式只测 1 次 T_AB 必须拒绝出数 ----
    # 零共视链路里，T_AB 的重复性是「A/B 是否真的刚性固连」唯一的证据。
    # 只拍 1 张同框图时以前仍 ok=True，等于把"没验过"当成"验过了"。
    res_one = calibrate_stereo(objp, img_l, img_r, (W, H), cfg_chain, spec,
                               stereo_mode="angled", T_camL_A=T_camL_A,
                               T_AB=T_ab_real, T_camR_B=T_camR_B,
                               T_AB_samples=[T_ab_real])
    print(f"  单张同框图：判定 {res_one.grade} / ok={res_one.ok}"
          f"（{res_one.verdict.splitlines()[0][:46]}…）")

    # ---- 反证 2：3 对共视 + angled + joint -> 必须被护栏拒绝 ----
    guard_msg, guard_ok = "", False
    try:
        calibrate_stereo(objp, img_l[:3], img_r[:3], (W, H), CalibConfig(), spec,
                         stereo_mode="angled")
    except ValueError as exc:
        guard_msg = str(exc)
        guard_ok = ("拒绝出数" in guard_msg and "chain" in guard_msg
                    and str(MIN_COVISIBLE_PAIRS_FOR_JOINT_ANGLED) in guard_msg)
        print(f"  护栏原文：{guard_msg.splitlines()[0]}")
    except Exception as exc:                              # noqa: BLE001
        print(f"  护栏抛了非 ValueError：{type(exc).__name__}: {exc}")

    # 护栏不是"无差别禁止"：8 对共视时应当继续走到求解（不抛护栏异常）
    not_blocked = True
    try:
        calibrate_stereo(objp, img_l[:8], img_r[:8], (W, H), CalibConfig(), spec,
                         stereo_mode="angled")
    except ValueError as exc:
        not_blocked = "拒绝出数" not in str(exc)
        print(f"  8 对共视：非护栏的 ValueError（{str(exc).splitlines()[0][:50]}）")
    print(f"  8 对共视：未被护栏拦截 = {not_blocked}")

    # ---- 2026-11-09 方案 A：链两端位姿必须来自**同一个支架摆放** ----
    # 内参段里板是移动的：`T_cl_A[i]` 与 `T_cr_B[j]`（i≠j）描述的是两个
    # 不同摆放。本合成数据里 `T_cr_B[j]` 正是由 `T_cl_A[j]` 推出来的，
    # 所以能定量测出"错位"有多大 —— 这就是参考对不能省的理由。
    # 旧 GUI 双板流程正是各取 `argmin(per_view)` 一帧（几乎必然 i≠j），
    # 而且 cross_poses=None、不传 --baseline-mm，**完全不会报警**。
    mix_rot, mix_tr = [], []
    for j in (1, 2, 3):
        R_m, T_m, _ = rp.relative_extrinsics(T_cl_A[0], T_BA_true, T_cr_B[j])
        mix_rot.append(rp.rotation_angle_deg(R_m, R_true))
        mix_tr.append(float(np.linalg.norm(T_m.ravel() - T_true.ravel())) * 1000.0)
    print(f"  反证（两端位姿取自不同摆放）：旋转误差最大 {max(mix_rot):.2f}° / "
          f"平移最大 {max(mix_tr):.1f} mm"
          f"（同摆放为 {max(rot_err):.2f}° / {max(trans_err):.1f} mm）")

    # ---- 缺 T_AB 必须报错（不许猜、不许回退 joint）----
    no_tab_msg = ""
    try:
        calibrate_stereo(objp, img_l, img_r, (W, H), cfg_chain, spec,
                         stereo_mode="angled")
    except ValueError as exc:
        no_tab_msg = str(exc)
    print(f"  缺 T_AB：{'被拒绝（' + no_tab_msg.splitlines()[0][:46] + '…）' if no_tab_msg else '**没有报错**'}")

    checks = [
        ("链式恢复旋转误差 < 0.5°（20 次最大值）", max(rot_err) < 0.5),
        ("链式恢复平移误差 < 2 mm（20 次最大值）", max(trans_err) < 2.0),
        ("反证：T_AB 取逆后误差 ≥10 倍（测试本身能证伪）",
         bad_rot > 1.0 and bad_rot > 10.0 * max(rot_err)),
        ("反证：两端位姿取自不同摆放 → 旋转误差 >2°（参考对不能省）",
         max(mix_rot) > 2.0),
        ("反证：两端位姿取自不同摆放 → 平移误差 >5 mm",
         max(mix_tr) > 5.0),
        ("同摆放仍准（判据没把正确链路一起判死）",
         max(rot_err) < 0.5 and max(trans_err) < 2.0),
        ("chain 产物：rectification == 'none_raw'",
         res.rectification == "none_raw"),
        ("chain 产物：R1=R2=I（未做极线校正，只去畸变）",
         np.allclose(res.R1, np.eye(3)) and np.allclose(res.R2, np.eye(3))),
        ("chain 产物：P1/P2 是 3x4 且 P1 = K1·[I|0]",
         res.P1.shape == (3, 4) and res.P2.shape == (3, 4)
         and np.allclose(res.P1, res.K1 @ np.hstack([np.eye(3), np.zeros((3, 1))]))),
        ("chain 产物：P2 = K2·[R|T]（与 R/T 自洽，三角化才可用）",
         np.allclose(res.P2, res.K2 @ np.hstack([res.R, np.asarray(res.T).reshape(3, 1)]))),
        ("chain 产物：Q 为 None（未校正无视差图）", res.Q is None),
        ("chain 端到端旋转误差 < 0.5°", R_e2e < 0.5),
        ("chain 端到端基线误差 < 2 mm", base_e2e < 2.0),
        ("chain 判定可用（ok=True）", bool(res.ok)),
        ("双板：2 次 T_AB 且一致 → 仍可用（没把正确链路一起判死）",
         bool(res.ok) and res.consistency["n"] == 2),
        ("双板：只测 1 次 T_AB → 拒绝出数（A/B 刚性没证据）",
         (not res_one.ok) and "重复性证据" in res_one.verdict),
        ("consistency：n 等于样本数", res.consistency.get("n") == 2),
        ("3 对共视 + angled + joint -> 抛 ValueError 拒绝出数", guard_ok),
        ("8 对共视时护栏不拦截（不是无差别禁止）", not_blocked),
        ("chain 缺 T_AB -> 抛 ValueError（不猜、不回退）",
         "T_AB" in no_tab_msg and "不可解" in no_tab_msg),
        ("非法 method 立即报错", _raises_value_error(lambda: CalibConfig(method="chna"))),
        ("报告里不出现「极线误差 RMS」这一行（90° 无此指标）",
         "极线误差 RMS" not in res.report()),
        ("报告里有 ||T|| 与 T_AB 重复性",
         "基线 ||T||" in res.report() and "T_AB 重复性" in res.report()),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_chain_frame_algebra() -> bool:
    """**证伪实验**：相机系与支架摆放都能在链里约掉，但两端必须同一摆放。

    用户 2026-11-09 问：「最后拍组合装置也是用一台相机，怎么保证算出两台
    相机的相对位置？」答案分两半，以前只写在文档里、没有判据守着：

    ① ``T_AB = inv(T_cam_B) · T_cam_A`` 里，**观测相机的坐标系被夹在中间
       约掉了**。第 ④ 步那台相机只是临时的一把尺子，换相机、换摆放、
       换时间都不影响 ``T_AB`` —— 所以「只用一台相机」不矛盾。
    ② 链 ``inv(C_R) · G_Y · inv(G_X) · C_L`` 里，**支架摆放只在
       ``G_X = G_Y`` 时约掉**。摆放本身不进结果，但两端必须来自同一摆放
       —— 这就是第 ③ 步参考对不能省的理由。

    顺带反证：随便少取一个逆、或两块板不同一次测量，结果都会爆掉，
    说明这些判据不是恒真的装饰。
    """
    print("\n----- 链式坐标系代数（相机系 / 摆放被约掉） -----")
    import calib.relative_pose as rp

    def T(R, t):
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = t
        return M

    def Ry(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def Rz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def ang(R):
        return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))

    # 支架系里 A、B 两板（B 相对 A 转 90°）：与相机、与摆放都无关的刚性常量
    T_wA0 = T(np.eye(3), np.zeros(3))
    T_wB0 = T(Ry(np.radians(90)), np.array([0.20, 0.0, 0.0]))
    T_AB_true = rp.compose(rp.invert(T_wB0), T_wA0)

    # ---- ① 三个完全不同的观测相机位姿 ----
    views = [T(np.eye(3), np.array([0.0, 0.0, -1.0])),
             T(Rz(np.radians(25)) @ Ry(np.radians(15)),
               np.array([0.5, 0.1, -1.3])),
             T(Ry(np.radians(-40)), np.array([-0.3, 0.2, -0.9]))]
    cam_errs, last = [], None
    for T_wC in views:
        T_cam_A = rp.compose(rp.invert(T_wC), T_wA0)
        T_cam_B = rp.compose(rp.invert(T_wC), T_wB0)
        last = (T_cam_A, T_cam_B)
        got = rp.board_to_board_from_single_camera(T_cam_A, T_cam_B)
        cam_errs.append(float(np.abs(got - T_AB_true).max()))
    print(f"  换 3 个观测相机位姿：|T_AB - 真值| 最大 {max(cam_errs):.2e}"
          f"（浮点舍入量级）")

    # 反证：少取一个逆（两块板不是同一次测量的正确组合）→ 结果必须爆掉
    wrong = rp.compose(last[1], last[0])
    wrong_err = float(np.abs(wrong - T_AB_true).max())
    print(f"  反证（少取逆 / 两块板不同一次测量）：偏差 {wrong_err:.3f}")

    # ---- ② 两台固定相机 + 支架摆放 G ----
    T_wCL = T(np.eye(3), np.zeros(3))
    T_wCR = T(Ry(np.radians(90)), np.array([0.5, 0.0, 0.0]))
    TRUE = rp.compose(rp.invert(T_wCR), T_wCL)

    def chain(G_X, G_Y, invert_left=True):
        T_camL_A = rp.compose(rp.invert(T_wCL), rp.compose(G_X, T_wA0))
        T_camR_B = rp.compose(rp.invert(T_wCR), rp.compose(G_Y, T_wB0))
        if invert_left:
            # 走真实核心层（它内部对 T_camL_A 取逆）
            R, t, _ = rp.relative_extrinsics(T_camL_A, T_AB_true, T_camR_B)
            got = rp.rt_to_T(R, t.ravel())
        else:
            # 反证：故意不取逆，看链会不会爆掉
            got = rp.compose(T_camR_B, T_AB_true, T_camL_A)
        return (ang(got[:3, :3] @ TRUE[:3, :3].T),
                float(np.linalg.norm(got[:3, 3] - TRUE[:3, 3])) * 1000.0)

    G1 = T(Rz(np.radians(12)), np.array([0.02, 0.0, 0.60]))
    G2 = T(Rz(np.radians(22)), np.array([0.05, 0.0, 0.62]))   # 差 10° / ~3.6 cm
    G3 = T(Rz(np.radians(-30)), np.array([-0.1, 0.0, 0.8]))  # 完全不同的摆放
    e_same = chain(G1, G1)
    e_other = chain(G3, G3)
    e_diff = chain(G1, G2)
    e_noinv = chain(G1, G1, invert_left=False)
    print(f"  两路同一摆放 G1：{e_same[0]:.4f}° / {e_same[1]:.3f} mm；"
          f"换成 G3：{e_other[0]:.4f}° / {e_other[1]:.3f} mm"
          f"（摆放本身不进结果）")
    print(f"  反证：左用 G1、右用 G2 → {e_diff[0]:.4f}° / {e_diff[1]:.3f} mm；"
          f"故意不取逆 → {e_noinv[0]:.2f}° / {e_noinv[1]:.1f} mm")

    checks = [
        ("换 3 个观测相机位姿，T_AB 不变（相机系被约掉）",
         max(cam_errs) < 1e-10),
        ("反证：少取逆 → T_AB 明显变掉（判据不是恒真）",
         wrong_err > 1e-3),
        ("两路同一摆放 → 外参误差 ≈ 0（G 被约掉）",
         e_same[0] < 1e-6 and e_same[1] < 1e-3),
        ("换一个完全不同的摆放 G3，外参仍 ≈ 0（摆放不进结果）",
         e_other[0] < 1e-6 and e_other[1] < 1e-3),
        ("反证：两端不同摆放 → 旋转误差 > 2° / 平移 > 5 mm",
         e_diff[0] > 2.0 and e_diff[1] > 5.0),
        ("反证：链上少取逆 → 外参爆掉（方向不是装饰）",
         e_noinv[0] > 10.0 or e_noinv[1] > 50.0),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_pair_frame_T_AB_roundtrip() -> bool:
    """**从一张"同时看到两块板"的同框图求 T_AB**（2026-11-06，补 §11.7 的缺口）。

    ## 守的是什么

    `test_chain_extrinsics_90deg` 把 `T_AB` **真值**直接喂进 `calibrate_stereo`，
    于是「`detect_two_boards` 检出两块板 → `solvePnP` 出位姿 → `_assign_ab`
    判定谁是 A/B → `board_to_board_from_single_camera` 求 T_AB」这一整段
    **没有任何测试覆盖**。真机上 A/B 判反、或第二块板根本检不出，旧测试照样全绿。

    ## 真值怎么造（为什么不是直接拿物理位姿比）

    棋盘**内角点网格**在 180° 旋转下自对称，`render_view` 又因四周留白使板坐标系
    与贴图错开一格；两者合成一个**常量**刚体变换 G，使
    `detect_board` + `solvePnP` 给出的板坐标系 = 物理板坐标系 ∘ G。
    G 对两块同规格板完全相同，**在链式外参里整段抵消**
    （`T_camR_B · T_AB · inv(T_camL_A)` 里 G 与 inv(G) 相消），所以真实链路不受
    影响；但拿"物理板坐标系"直接比对会稳定差 180°（下面会打印出来当证据）。

    因此本测试的真值取**同一渲染器/检测器下两块板各自单张成像**测到的相对位姿
    —— 这正是真机上 `--left-dir` / `--right-dir` 两段目录的语义，且是两次
    **独立**的成像与检测，不是把同框图的结果反过来当答案。
    """
    print("\n----- 同框图求 T_AB（检测链路闭环 + A/B 判反护栏） -----")
    import apps.calibrate_stereo as cs
    import calib.relative_pose as rp
    from calib import detect_board, detect_two_boards

    W, H = 1280, 720
    K = _K(900.0, 900.0, 640.0, 360.0)
    D = np.zeros(5)
    spec = BoardSpec(cols=8, rows=5, square_mm=25.0)
    objp = spec.object_points()
    ppm = 900.0
    board_img, pad = make_board_image(spec, ppm)
    solid = np.full_like(board_img, 255)          # 只用来取"板覆盖了哪些像素"

    # ---- 刚性双板真值：A 绕 Y +45°、B 绕 Y -45°，两板夹角正好 90° ----
    # 板距（各自中心到相机）：A 0.75 m、B 0.75 m；横向 ±0.14 m —— 一块在画面左、
    # 一块在右，投影后互不重叠（下面的 bbox 会打印出来）。1280x720 / f=900。
    R_A = cv2.Rodrigues(np.radians([0.0, 45.0, 0.0]))[0]
    R_B = cv2.Rodrigues(np.radians([0.0, -45.0, 0.0]))[0]
    t_A = np.array([-0.14, 0.0, 0.75])
    t_B = np.array([0.14, 0.0, 0.75])
    rv_A, rv_B = cv2.Rodrigues(R_A)[0], cv2.Rodrigues(R_B)[0]

    def _render(rv, t, noise=0.4, blur=0.5):
        return render_view(board_img, pad, spec, ppm, rv, t, K, D, (W, H), noise, blur)

    def _mask(rv, t):
        return render_view(solid, pad, spec, ppm, rv, t, K, D, (W, H), 0.0, 0.0)[:, :, 0] > 200

    m_a, m_b = _mask(rv_A, t_A), _mask(rv_B, t_B)
    img_a, img_b = _render(rv_A, t_A), _render(rv_B, t_B)
    pair = np.full((H, W, 3), 150, np.uint8)      # 与 render_view 的背景同色
    pair[m_a] = img_a[m_a]
    pair[m_b] = img_b[m_b]
    for name, m in (("A", m_a), ("B", m_b)):
        ys, xs = np.where(m)
        print(f"  合成板 {name}：bbox=({xs.min()},{ys.min()})-({xs.max()},{ys.max()})")
    print(f"  两板像素重叠 = {int((m_a & m_b).sum())} px（必须为 0，否则检板会互相遮盖）")

    # ---- 真实链路第 1 环：一张图里检出两块板 ----
    gray = cv2.cvtColor(pair, cv2.COLOR_BGR2GRAY)
    found = detect_two_boards(gray, spec)
    n_pts = [] if found is None else [len(o.corners) for o in found]
    print(f"  detect_two_boards -> {'None（检不出）' if found is None else n_pts}")
    if found is None:
        print("  [不达标] 合成同框图上检不出两块板 —— 是**构造参数**问题，"
              "必须修参数，不许 skip / 直接返回 True")
        return False

    T1, e1 = cs._pnp_reproj_rms(objp, found[0].corners, K, D)
    T2, e2 = cs._pnp_reproj_rms(objp, found[1].corners, K, D)
    print(f"  同框图位姿重投影：{e1:.3f} / {e2:.3f} px")

    # ---- 独立真值：两块板各自单张成像（对应真机的 --left-dir / --right-dir）----
    ref = {}
    for name, rv, t in (("A", rv_A, t_A), ("B", rv_B, t_B)):
        one = _render(rv, t)
        obs = detect_board(cv2.cvtColor(one, cv2.COLOR_BGR2GRAY), spec, exhaustive=True)
        if not obs.found or obs.corners is None:
            print(f"  [不达标] 单板 {name} 的对照图检不出板")
            return False
        ref[name], e_one = cs._pnp_reproj_rms(objp, obs.corners, K, D)
        print(f"  单板真值 {name}：重投影 {e_one:.3f} px  t={ref[name][:3, 3].round(4)}")
    T_AB_true = rp.board_to_board_from_single_camera(ref["A"], ref["B"])

    # ---- 真实链路第 2 环：A/B 分配（参考簇 = 左路目录里 A 板的姿态）----
    T_A, T_B, d_a, d_b = cs._assign_ab(T1, T2, [ref["A"][:3, :3]], "left")
    dA_A = rp.rotation_angle_deg(T_A[:3, :3], ref["A"][:3, :3])
    dB_B = rp.rotation_angle_deg(T_B[:3, :3], ref["B"][:3, :3])
    print(f"  到参考姿态簇：{d_a:.3f}° / {d_b:.3f}°（差 {abs(d_a - d_b):.2f}°）")
    print(f"  分配结果：T_A 到 A 板真值 {dA_A:.3f}°，T_B 到 B 板真值 {dB_B:.3f}°")

    # ---- 真实链路第 3 环：由一次共视求刚性相对位姿 ----
    T_AB = rp.board_to_board_from_single_camera(T_A, T_B)
    rot_err = rp.rotation_angle_deg(T_AB[:3, :3], T_AB_true[:3, :3])
    trans_err = float(np.linalg.norm(T_AB[:3, 3] - T_AB_true[:3, 3])) * 1000.0
    print(f"  T_AB（同框图）vs 单板真值：旋转 {rot_err:.4f}°  平移 {trans_err:.4f} mm")

    # 证据：与"物理板坐标系"直接比对稳定差 180° + 一格平移（G，链上会抵消）
    T_AB_phys = rp.compose(rp.invert(rp.rt_to_T(R_B, t_B)), rp.rt_to_T(R_A, t_A))
    print(f"  物理板坐标系直接比对：旋转 {rp.rotation_angle_deg(T_AB[:3, :3], T_AB_phys[:3, :3]):.2f}°"
          f" / {np.linalg.norm(T_AB[:3, 3] - T_AB_phys[:3, 3]) * 1000.0:.1f} mm"
          f"（常量 G=180°+一格，链式里与 inv(G) 相消，**不是**链路误差）")

    # ---- A/B 判反护栏：两板到参考簇等距 -> 必须报错，不许猜 ----
    def _mid_rot(Ra, Rb):
        """两个旋转的测地中点（参考簇放在这里，两块板距离必然相等）。"""
        half = cv2.Rodrigues(cv2.Rodrigues(Ra.T @ Rb)[0].ravel() * 0.5)[0]
        return Ra @ half

    R_mid = _mid_rot(ref["A"][:3, :3], ref["B"][:3, :3])
    guard_msg, guard_ok = "", False
    try:
        cs._assign_ab(T1, T2, [R_mid], "left")
        print("  [不达标] 两板等距时 _assign_ab **没有报错**（静默判反）")
    except ValueError as exc:
        guard_msg = str(exc)
        guard_ok = "无法区分" in guard_msg
        print(f"  等距护栏原文：{guard_msg.splitlines()[0]}")

    # 对称性：参考簇换成右路的 B 板，返回顺序必须仍然是 (A, B)
    T_A2, T_B2, _d_a2, _d_b2 = cs._assign_ab(T1, T2, [ref["B"][:3, :3]], "right")
    sym_ok = (rp.rotation_angle_deg(T_A2[:3, :3], ref["A"][:3, :3]) < 1.0
              and rp.rotation_angle_deg(T_B2[:3, :3], ref["B"][:3, :3]) < 1.0)

    # ---- 2026-11-11：同框图的 180° 等价解纠错 --------------------------------
    #
    # 真实事故：14:39 会话的 5 张同框图里，旧规则（逐帧各自挑最近参考簇）有
    # 3 帧把 A/B 判反 -> T_AB 样本两簇差 179.597°。下面两条把纠错机制的两端
    # 都钉住：**能修的必须修对**（板框等价解）、**真实散布不许修**。
    from calib.stereo_calib import CHAIN_ROT_OK_DEG

    # 场景 ①：单板角点顺序整体翻 180°（等价于 PnP 给出 `T @ C`）。
    # 把同框图里 A 板的 40 个角点倒序再 PnP —— 这正是真机上"起始角点翻了"
    # 时 `solvePnP` 拿到的东西，重投影几乎不变、旋转差 180°。
    T1_rev, e1_rev = cs._pnp_reproj_rms(
        objp, np.asarray(found[0].corners).reshape(-1, 2)[::-1], K, D)
    frames_flip = [
        {"name": "synth_flip_01.png", "poses": [T1_rev, T2], "errs": [e1_rev, e2]},
        {"name": "synth_flip_02.png", "poses": [T1, T2], "errs": [e1, e2]},
    ]
    samples_flip, _rows_flip, corr_flip, _before_flip = cs._resolve_pair_samples(
        frames_flip, [ref["A"][:3, :3]], spec, "left")
    rot_err_flip = max(rp.rotation_angle_deg(s[:3, :3], T_AB_true[:3, :3])
                       for s in samples_flip)
    trans_err_flip = max(float(np.linalg.norm(s[:3, 3] - T_AB_true[:3, 3])) * 1000.0
                         for s in samples_flip)
    flip_fixed = any(c["action"].startswith("板框") for c in corr_flip)
    print(f"  合成「单板角点倒序」同框图：纠正后 T_AB 旋转误差 {rot_err_flip:.3f}° / "
          f"平移 {trans_err_flip:.3f} mm；板框翻转台账 {int(flip_fixed)} 条")

    # 场景 ②：故意 10° 真实散布 —— 纠错**不许**把它"修"成一致。
    # A 板姿态恒为单位阵；B 板绕 Z 轴 -110°/-115°/-120°，样本正好散开 10°。
    # 与中位样本差 5/10° 远低于逐帧离群门限（45°），纠错必须一条都不做。
    def _Rz_deg(deg):
        return cv2.Rodrigues(np.radians([0.0, 0.0, deg]))[0]

    frames_spread = [{
        "name": f"spread_{k}.png",
        "poses": [np.eye(4),
                  rp.rt_to_T(_Rz_deg(-110.0 - kdeg), np.array([0.30, 0.0, 0.0]))],
        "errs": [0.0, 0.0],
    } for k, kdeg in enumerate((0.0, 5.0, 10.0))]
    samples_spread, _rows_sp, corr_sp, _before_sp = cs._resolve_pair_samples(
        frames_spread, [np.eye(3)], spec, "left")
    spread = rp.consistency(samples_spread)
    spread_kept = (spread["rot_max_deg"] > CHAIN_ROT_OK_DEG and not corr_sp)
    print(f"  故意 10° 真实散布：旋转最大偏离 {spread['rot_max_deg']:.3f}° "
          f"（门限 {CHAIN_ROT_OK_DEG}°），纠正台账 {len(corr_sp)} 条 -> "
          f"{'仍判不一致（未被修成通过）' if spread_kept else '[不达标] 被修成一致了'}")

    # 场景 ③（2026-11-11，task-25）：**一帧**额外转了近 180°（B 板框等价解），
    # 其余三帧一致 —— >45° 的逐帧离群块必须把它纠正，最终 T_AB 一致、台账点名该帧。
    # 这条单独存在的原因（task-21 Critic）：10° 用例走的是"一致性门 + 候选改善"，
    # **碰不到** >45° 那段；当时删掉逐帧离群块，10° 用例照样绿（测试缺口）。
    TA_ok = np.eye(4)
    TB_ok = rp.rt_to_T(_Rz_deg(-100.0), np.array([0.30, 0.0, 0.0]))
    _sq = spec.square_mm / 1000.0
    _c = np.array([(spec.cols - 1) * _sq / 2.0, (spec.rows - 1) * _sq / 2.0, 0.0])
    _Rz180 = np.diag([-1.0, -1.0, 1.0])
    _C = np.eye(4)
    _C[:3, :3] = _Rz180
    _C[:3, 3] = _c - _Rz180 @ _c            # 绕**角点网格中心**翻 180°，与实现同源
    frames_one_outlier = [
        {"name": "ok_1.png", "poses": [TA_ok, TB_ok], "errs": [0.0, 0.0]},
        {"name": "ok_2.png", "poses": [TA_ok, TB_ok], "errs": [0.0, 0.0]},
        {"name": "ok_3.png", "poses": [TA_ok, TB_ok], "errs": [0.0, 0.0]},
        {"name": "bad_flip_180.png", "poses": [TA_ok, TB_ok @ _C],
         "errs": [0.0, 0.0]},
    ]
    samples_ol, _rows_ol, corr_ol, _before_ol = cs._resolve_pair_samples(
        frames_one_outlier, [np.eye(3)], spec, "left")
    cons_ol = rp.consistency(samples_ol)
    ol_fixed = (cons_ol["rot_max_deg"] < 0.1 and cons_ol["trans_max_mm"] < 1.0
                and any(c["file"] == "bad_flip_180.png"
                        and c["action"].startswith("板框") for c in corr_ol))
    print(f"  一帧 >45°（B 板框翻 180°）：纠正后旋转最大偏离 "
          f"{cons_ol['rot_max_deg']:.3f}° / 平移 {cons_ol['trans_max_mm']:.3f} mm，"
          f"台账 {len(corr_ol)} 条（{corr_ol[0]['file'] if corr_ol else '—'}）")

    # 场景 ④：**两帧**都 >45° 且互相不一致（不是等价解，修不回来）→
    # 不许"修"成通过；最终一致性仍超门限，标定会拒绝出数。
    _D1 = rp.rt_to_T(_Rz_deg(70.0), np.zeros(3))
    _D2 = rp.rt_to_T(cv2.Rodrigues(np.radians([0.0, 80.0, 0.0]))[0], np.zeros(3))
    frames_two_outliers = [
        {"name": "ok_1.png", "poses": [TA_ok, TB_ok], "errs": [0.0, 0.0]},
        {"name": "ok_2.png", "poses": [TA_ok, TB_ok], "errs": [0.0, 0.0]},
        {"name": "bad_1.png", "poses": [TA_ok, TB_ok @ _D1], "errs": [0.0, 0.0]},
        {"name": "bad_2.png", "poses": [TA_ok, TB_ok @ _D2], "errs": [0.0, 0.0]},
    ]
    samples_two, _rows_two, _corr_two, _before_two = cs._resolve_pair_samples(
        frames_two_outliers, [np.eye(3)], spec, "left")
    cons_two = rp.consistency(samples_two)
    two_rejected = cons_two["rot_max_deg"] > cs.PAIR_OUTLIER_DEG
    print(f"  两帧 >45° 且互不一致：纠正后旋转最大偏离 "
          f"{cons_two['rot_max_deg']:.3f}°（一致性门 {CHAIN_ROT_OK_DEG}°）-> "
          f"{'仍判不一致（没被修成通过）' if two_rejected else '[不达标] 被修成通过'}")

    checks = [
        ("同框图中检出两块板（各 40 个角点）", found is not None and n_pts == [40, 40]),
        ("同框图位姿重投影 < 1 px", max(e1, e2) < 1.0),
        ("A/B 分配正确：T_A 落在 A 板真值 1° 内", dA_A < 1.0),
        ("A/B 分配正确：T_B 落在 B 板真值 1° 内", dB_B < 1.0),
        ("参考簇距离有区分度（两板差 > 20°，不是碰巧过）",
         abs(d_a - d_b) > 20.0),
        ("T_AB 旋转误差 < 1.0°（同框图 vs 独立单板真值）", rot_err < 1.0),
        ("T_AB 平移误差 < 3.0 mm", trans_err < 3.0),
        ("两板到参考簇等距（<5°）时抛 ValueError 且含「无法区分」", guard_ok),
        ("参考簇取右路 B 板时返回顺序仍为 (A, B)", sym_ok),
        ("合成「单板角点倒序」同框图：180° 等价解被纠正，T_AB 旋转 < 1.0° / 平移 < 3.0 mm",
         rot_err_flip < 1.0 and trans_err_flip < 3.0 and flip_fixed),
        ("故意 10° 真实散布：仍判不一致（> 1.5°）且纠错台账为空", spread_kept),
        ("一帧 >45°（B 板框翻 180°）：逐帧离群纠正后 T_AB 一致（<0.1°）且台账点名该帧",
         ol_fixed),
        ("两帧 >45° 互相不一致：仍判不一致（>45°），不许修成通过", two_rejected),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_pair_prior_guard_double_flip() -> bool:
    """第四层守卫：两板同框图里**都**翻 180°、全程一致 → 三层全过但守卫判错解。

    真机没有可复核的同名点（chain 左右看不同板、angled 评估被跳过），
    唯一独立证据是装置先验：`config/calib/实验性质大角度1.json` 里
    `quality.consistency.baseline_mm=242.15`、`angle_deg=88.55`。
    合成装置刻意把两板中心距做成 242.15 mm，让正常样本正好对上真机先验。
    """
    print("\n----- 第四层：两板同时翻 + 全程一致（装置先验复核） -----")
    import apps.calibrate_stereo as cs
    import calib.relative_pose as rp

    spec = BoardSpec(cols=8, rows=5, square_mm=25.0)
    PRIOR_B, PRIOR_A = 242.15, 88.55

    def _Ry(deg):
        return cv2.Rodrigues(np.radians([0.0, deg, 0.0]))[0]

    # A +45° / B -45°（两板夹角 90°）；中心距 242.15 mm → 正常 T_AB 基线=先验
    t_a = np.array([-0.121075, 0.0, 0.75])
    t_b = np.array([0.121075, 0.0, 0.75])
    T_A = rp.rt_to_T(_Ry(45.0), t_a)
    T_B = rp.rt_to_T(_Ry(-45.0), t_b)
    # 板框 180° 等价解（与 _board_180_equiv 同源：绕角点网格中心翻）
    s = spec.square_mm / 1000.0
    c = np.array([(spec.cols - 1) * s / 2.0, (spec.rows - 1) * s / 2.0, 0.0])
    rz = np.diag([-1.0, -1.0, 1.0])
    C = np.eye(4)
    C[:3, :3] = rz
    C[:3, 3] = c - rz @ c
    T_AF, T_BF = T_A @ C, T_B @ C

    def _frames(flip_both: bool):
        out = []
        for k in range(5):
            jit = rp.rt_to_T(_Ry(0.02 * k), np.zeros(3))   # 微小真实抖动
            ta, tb = (jit @ T_AF, jit @ T_BF) if flip_both else (jit @ T_A, jit @ T_B)
            out.append({"name": f"pair_{k}.png", "poses": [ta, tb],
                        "errs": [0.0, 0.0]})
        return out

    checks = []
    # ---- ① 正常 5 帧：第四层不误拒 ----
    samples_n, _rows_n, _corr_n, _before_n = cs._resolve_pair_samples(
        _frames(False), [T_A[:3, :3]], spec, "left")
    g_n = cs.check_pair_prior_consistency(
        samples_n, prior_baseline_mm=PRIOR_B, prior_angle_deg=PRIOR_A)
    cons_n = rp.consistency(samples_n)
    print(f"  正常 5 帧：基线 {cons_n['baseline_mm']:.2f} mm / 相对角 "
          f"{g_n['angle_deg']:.2f}° -> 第四层 {'通过' if g_n['ok'] else '误拒'}"
          f"（{g_n['reasons']}）")
    checks.append(("正常 5 帧：第四层不误拒（ok=True，无 reasons）",
                   g_n["ok"] is True and not g_n["reasons"]))

    # ---- ② 两板同时翻、5 帧全程一致：三层全过，第四层判错解 ----
    samples_f, _rows_f, corr_f, _before_f = cs._resolve_pair_samples(
        _frames(True), [T_AF[:3, :3]], spec, "left")
    cons_f = rp.consistency(samples_f)
    # 三层证据：②/① 没报错也没纠正（ref 取翻后 A 板，模拟内参段与同框图同一等价解）；
    # ③ 逐帧无离群。若三层任一介入，corr_f 就非空。
    three_layers_pass = (not corr_f) and cons_f["rot_max_deg"] < 1.0
    g_f = cs.check_pair_prior_consistency(
        samples_f, prior_baseline_mm=PRIOR_B, prior_angle_deg=PRIOR_A)
    print(f"  两板全翻 5 帧（三层证据）：A/B 指派未报错、纠正台账 {len(corr_f)} 条、"
          f"逐帧旋转最大偏离 {cons_f['rot_max_deg']:.3f}° -> 三层{'全过' if three_layers_pass else '有介入'}")
    brel = g_f.get("baseline_rel_err") or 0.0   # 守卫被改空时也要能干净判红，不抛异常
    bdeg = g_f.get("angle_deg")
    print(f"  第四层：基线 {g_f['baseline_mm']:.2f} mm vs 先验 {PRIOR_B:.2f} mm"
          f"（相对误差 {brel * 100:.1f}%）；"
          f"相对角 {bdeg:.2f}° vs {PRIOR_A:.2f}° -> "
          f"{'错解' if not g_f['ok'] else '漏判'}")
    for r in g_f["reasons"]:
        print(f"    原因：{r}")
    checks.append(("两板同时翻且全程一致：三层全过（0 纠正 / rot_max<1°）",
                   three_layers_pass))
    checks.append(("第四层判错解：ok=False、reasons 写明基线与先验偏离及疑似双翻",
                   g_f["ok"] is False
                   and any("基线" in r and "180" in r for r in g_f["reasons"])))
    # 相对角共轭不变（90°→90°），这一例靠基线触发；角度项仍在字段里可查
    checks.append(("双翻例由基线触发（相对角共轭不变，未被误写成角度触发）",
                   bdeg is not None and abs(bdeg - 90.0) < 0.5
                   and brel > cs.PAIR_PRIOR_BASELINE_REL_TOL))

    # ---- ③ 只翻 1 帧：③ 离群层负责，第四层不误判为"全程一致错解" ----
    frames_one = _frames(False)
    frames_one[-1] = {"name": "bad.png", "poses": [T_AF, T_BF], "errs": [0.0, 0.0]}
    samples_o, _rows_o, corr_o, _before_o = cs._resolve_pair_samples(
        frames_one, [T_A[:3, :3]], spec, "left")
    cons_o = rp.consistency(samples_o)
    g_o = cs.check_pair_prior_consistency(
        samples_o, prior_baseline_mm=PRIOR_B, prior_angle_deg=PRIOR_A)
    print(f"  单帧两板翻：③ 纠正台账 {len(corr_o)} 条、纠正后 rot_max "
          f"{cons_o['rot_max_deg']:.3f}° -> 第四层 "
          f"{'不误判' if g_o['ok'] else '误判'}")
    checks.append(("单帧翻由③处理：纠正后第四层不误判为全程一致错解（ok=True）",
                   g_o["ok"] is True and bool(corr_o)))
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_force_save_backup_and_dialog() -> bool:
    """A4：强制保存先把旧标定备份到同目录、失败则不写；弹窗说清路径与判定数字。"""
    print("\n----- A4：强制保存先备份 + 弹窗说清 -----")
    import hashlib
    import inspect
    import json
    import logging
    import tempfile
    from types import SimpleNamespace

    import apps.calibrate_stereo as cs
    from apps import _gui_stereo

    def _sha(p: Path) -> str:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    checks = []
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        formal = root / "forced_calibration.npz"
        js_formal = formal.with_suffix(".json")
        old_npz = b"OLD-FORMAL-NPZ-DO-NOT-LOSE-1234567890"
        old_json = '{"old": true, "tag": "before-force"}'
        formal.write_bytes(old_npz)
        js_formal.write_text(old_json, encoding="utf-8")
        # 历史备份越界不改：放一个带老名字的文件在旁边当哨兵
        hist = root / "forced_calibration_before_identity-fix.json"
        hist.write_text("KEEP-HISTORICAL-BACKUP", encoding="utf-8")
        hist_sha = _sha(hist)

        rig = StereoRig(
            K1=np.eye(3), D1=np.zeros(5), K2=np.eye(3), D2=np.zeros(5),
            R=np.eye(3), T=np.array([[0.24], [0.0], [0.0]]),
            image_size=(1280, 720),
            quality={"verdict": "测试"}, meta={"profile": "测试"})
        args = SimpleNamespace(out=str(formal), force_save=True)
        npz_new, js_new, bypassed = cs.finalize_and_save(
            rig, args, unusable=True, log=logging.getLogger("selftest_a4"))

        bak_npz = sorted(root.glob("forced_calibration_before_force_*.npz"))
        bak_json = sorted(root.glob("forced_calibration_before_force_*.json"))
        bak_ok = (len(bak_npz) == 1 and len(bak_json) == 1)
        same_old = (bak_ok and bak_npz[0].read_bytes() == old_npz
                    and bak_json[0].read_text(encoding="utf-8") == old_json)
        new_is_new = (Path(npz_new) == formal and formal.read_bytes() != old_npz
                      and Path(js_new).is_file() and "quality" in
                      json.loads(js_new.read_text(encoding="utf-8")))
        hist_ok = _sha(hist) == hist_sha
        print(f"  旧 npz sha256   {hashlib.sha256(old_npz).hexdigest()}")
        print(f"  旧 json sha256  {hashlib.sha256(old_json.encode()).hexdigest()}")
        if bak_ok:
            print(f"  备份 npz sha256 {_sha(bak_npz[0])}  <- {bak_npz[0].name}")
            print(f"  备份 json sha256 {_sha(bak_json[0])} <- {bak_json[0].name}")
        print(f"  新 npz sha256   {_sha(npz_new)}（bypassed={bypassed}）")
        print(f"  历史备份未改动={hist_ok}")
        checks.append(("force_save 先备份旧 .npz/.json 两份到同目录（新文件名带时间戳）",
                       bak_ok and same_old))
        checks.append(("备份内容逐字节 == 旧文件；新正式路径是新内容；历史备份未动",
                       same_old and new_is_new and hist_ok))

        # ② 备份失败：正式路径必须原样保留（反向保护）
        formal2 = root / "fail_case.npz"
        js2 = formal2.with_suffix(".json")
        formal2.write_bytes(old_npz)
        js2.write_text(old_json, encoding="utf-8")
        args2 = SimpleNamespace(out=str(formal2), force_save=True)
        orig_backup = cs._backup_before_force
        try:
            cs._backup_before_force = lambda _p: (_ for _ in ()).throw(
                OSError("模拟复制失败"))
            n2, j2, bp2 = cs.finalize_and_save(
                rig, args2, unusable=True, log=logging.getLogger("selftest_a4"))
        finally:
            cs._backup_before_force = orig_backup
        protected = (formal2.read_bytes() == old_npz
                     and js2.read_text(encoding="utf-8") == old_json
                     and not list(root.glob("fail_case_before_force_*"))
                     and bool(rig.quality.get("force_backup_failed")))
        print(f"  备份失败时正式路径逐字节保留={protected}（返回值 {Path(n2).name} / "
              f"{Path(j2).name} 仅为占位，未落盘）")
        checks.append(("备份失败 -> 打印警告且不写正式路径（旧标定受保护）", protected))

    # ③ 弹窗文案：i18n 字面量可提取（路径占位 + rot/trans 字样都在源码字面量里）
    src = (Path(__file__).resolve().parents[1] / "apps" / "_gui_stereo.py").read_text(
        encoding="utf-8")
    start = src.index("    def _force_calibrate(self)")
    end = src.index("    def _chain_common", start)
    block = src[start:end]
    text_ok = ("当前正式标定路径" in block and "rot_max_deg" in block
               and "trans_max_mm" in block and "备份" in block
               and '"未知"' in block and "askyesno(title, body" in block)
    print(f"  弹窗字面量：路径占位={'当前正式标定路径' in block} / "
          f"rot={'rot_max_deg' in block} / trans={'trans_max_mm' in block} / "
          f"备份={'备份' in block} / askyesno 变量传参={'askyesno(title, body' in block}")
    checks.append(("弹窗正文含路径占位、rot_max_deg/trans_max_mm、备份字样且走变量传参",
                   text_ok))
    # inspect 能拿到函数源码 = 文案确实落在这个函数里（不是别处拼的）
    try:
        fn_src = inspect.getsource(_gui_stereo.StereoWindow._force_calibrate)
    except Exception:                                    # noqa: BLE001
        fn_src = block
    checks.append(("文案位于 StereoWindow._force_calibrate（inspect 可见）",
                   "rot_max_deg" in fn_src and "当前正式标定路径" in fn_src))

    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_chain_triangulation_closure() -> bool:
    """chain 的 `P1/P2` 必须**真的能三角化**，且确实对应"未校正坐标系"（2026-10-24）。

    ## 守的是什么

    上一版 chain 把 `R1/R2/P1/P2` 置 `None`：标定"成功"、报告"优秀"，
    但 `StereoRig.rectify_maps()/triangulate()` 一读就越界 ——
    **标出来了却三角化用不了**。这个缺口在真实装置上要等到跑三维才发现。

    这里做三件事，缺一不可：

    1. **正证**：真值 3D 点 →（真值 K/R/T）投影到两路像素 → 用 chain 产出的
       `P1/P2` 三角化 → 回到 3D，误差必须 < 1 mm（P 矩阵与 R/T 自洽）；
    2. **不重复去畸变**：`D=0` 时 `rectify_points()` 的输出必须**就是**原始像素
       （`R=I, P=K` 下 `undistortPoints` 是恒等），否则说明畸变被扣了两次；
    3. **反证**：把 `stereoRectify` 的 `R1/P1` 校正结果喂给 `none_raw` 的
       `P1/P2`（校正/未校正混用），误差必须**显著变大** —— 这证明这套矩阵
       真的是"原始 left_camera 系"，而不是碰巧自洽。
    """
    print("\n----- chain 三角化闭环（none_raw 投影矩阵） -----")
    import calib.relative_pose as rp

    W, H = 1280, 720
    K_L = _K(900.0, 900.0, 640.0, 360.0)
    K_R = _K(980.0, 982.0, 610.0, 352.0)
    D = np.zeros(5)
    spec = BoardSpec(cols=8, rows=5, square_mm=25.0)
    objp = spec.object_points()

    # 90° 真值装置：右相机在 (0.5, 0, 0.5)，光轴沿 -X；左相机在原点，光轴 +Z
    R_true = cv2.Rodrigues(np.radians([0.0, 90.0, 0.0]))[0]
    C_true = np.array([0.50, 0.0, 0.50])
    T_true = (-R_true @ C_true).reshape(3, 1)
    T_cr_cl_true = rp.rt_to_T(R_true, T_true)

    # L 形双板：A 面朝左相机，B 转 90° 面朝 +X
    T_wA = rp.rt_to_T(np.eye(3), [0.0, 0.0, 0.50])
    R_wB = cv2.Rodrigues(np.radians([0.0, 90.0, 0.0]))[0]
    T_wB = rp.rt_to_T(R_wB, [0.20, 0.0, 0.50])
    T_BA_true = rp.compose(rp.invert(T_wB), T_wA)

    # ---- 合成两路视图（板 A 给左、板 B 给右），保证两路都完整入画 ----
    rng = np.random.default_rng(20261024)
    T_cl_A: list[np.ndarray] = []
    img_l, img_r = [], []
    tries = 0
    # 位姿要**跨距离**（0.35~0.75 m）+ 大角度倾斜：焦距与深度、畸变互相耦合，
    # 全在同一个深度上拍，内参会明显漂（实测 fx 偏 0.28%，闭环就破 1 mm）。
    while len(T_cl_A) < 20 and tries < 600:
        tries += 1
        rv = rng.uniform(-0.40, 0.40, 3)
        tv = np.array([rng.uniform(-0.07, 0.07), rng.uniform(-0.06, 0.06),
                       0.35 + 0.40 * rng.uniform(0, 1)])
        T_a = rp.rt_to_T(cv2.Rodrigues(rv)[0], tv)
        T_b = rp.compose(T_cr_cl_true, T_a, rp.invert(T_BA_true))
        p_l, _ = cv2.projectPoints(objp, cv2.Rodrigues(T_a[:3, :3])[0], T_a[:3, 3], K_L, D)
        p_r, _ = cv2.projectPoints(objp, cv2.Rodrigues(T_b[:3, :3])[0], T_b[:3, 3], K_R, D)
        q_l, q_r = p_l.reshape(-1, 2), p_r.reshape(-1, 2)
        inside = (lambda q: (q[:, 0] > 8).all() and (q[:, 0] < W - 8).all()
                  and (q[:, 1] > 8).all() and (q[:, 1] < H - 8).all())
        if not (inside(q_l) and inside(q_r)):
            continue
        T_cl_A.append(T_a)
        img_l.append((q_l + rng.normal(0, 0.1, q_l.shape)).astype(np.float32))
        img_r.append((q_r + rng.normal(0, 0.1, q_r.shape)).astype(np.float32))
    print(f"  合成视图：{len(T_cl_A)} 个（尝试 {tries} 次）")

    # ---- 跑 chain（T_camL_A / T_AB / T_camR_B 用真值，R/T 应精确复现真值）----
    T_camR_B = rp.compose(T_cr_cl_true, T_cl_A[0], rp.invert(T_BA_true))
    res = calibrate_stereo(
        objp, img_l, img_r, (W, H), CalibConfig(method="chain"), spec,
        stereo_mode="angled", T_camL_A=T_cl_A[0], T_AB=T_BA_true, T_camR_B=T_camR_B,
        T_AB_samples=[T_BA_true, T_BA_true])
    rig = StereoRig.from_result(res, meta={
        "rectification": res.rectification, "stereo_mode": "angled", "method": "chain"})
    print(f"  chain：rectification={res.rectification}  K1.fx={res.K1[0, 0]:.2f}"
          f"（真值 {K_L[0, 0]:.0f}）  R 误差 "
          f"{rp.rotation_angle_deg(res.R, R_true):.4f}°")

    # ---- 正证：真值 3D -> 真值投影 -> chain 的 P1/P2 反演 ----
    pts3d = np.array([[0.00, 0.00, 0.55], [0.05, 0.03, 0.60], [-0.04, -0.02, 0.50],
                      [0.02, 0.04, 0.65], [-0.06, 0.01, 0.58]], np.float64)
    uv1, _ = cv2.projectPoints(pts3d, np.zeros(3), np.zeros(3), K_L, D)
    uv2, _ = cv2.projectPoints(pts3d, cv2.Rodrigues(R_true)[0], T_true, K_R, D)
    uv1, uv2 = uv1.reshape(-1, 2), uv2.reshape(-1, 2)

    got = rig.triangulate(uv1, uv2)          # 内部走 rectify_points(R=I,P=K) -> P1/P2
    err = np.linalg.norm(got - pts3d, axis=1) * 1000.0
    print(f"  闭环：均值 {err.mean():.4f} mm  最大 {err.max():.4f} mm（目标 < 1 mm）")

    # ---- 不重复去畸变：已去畸变的点再走一次 rectify_points，必须几乎不动 ----
    #
    # 这也是 none_raw 的正确性证据：`P=K`（而不是 `P=K2·[R|T]`）时，
    # 右路只做去畸变、不被 R 多转一次；若把 P2 当 newCameraMatrix 用，
    # 这一项会差到几千像素（实测 828 px -> -9155 px）。
    p1r, p2r = rig.rectify_points(uv1, uv2)
    p1r2, p2r2 = rig.rectify_points(p1r, p2r)
    dd = max(float(np.abs(p1r2 - p1r).max()), float(np.abs(p2r2 - p2r).max()))
    print(f"  rectify_points 幂等性（重复去畸变）最大漂移：{dd:.3e} px")

    # ---- 反证：校正/未校正混用 -> 误差必须显著变大 ----
    rect = cv2.stereoRectify(
        K_L, D, K_R, D, (W, H), res.R, np.asarray(res.T).reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.0)
    R1r, R2r, P1r, P2r = rect[0], rect[1], rect[2], rect[3]   # OpenCV 顺序：R1,R2,P1,P2
    a = cv2.undistortPoints(uv1.reshape(-1, 1, 2), K_L, D, R=R1r, P=P1r).reshape(-1, 2)
    b = cv2.undistortPoints(uv2.reshape(-1, 1, 2), K_R, D, R=R2r, P=P2r).reshape(-1, 2)
    mix4 = cv2.triangulatePoints(res.P1, res.P2, a.T, b.T)
    mix = (mix4[:3] / mix4[3]).T
    err_mix = np.linalg.norm(mix - pts3d, axis=1) * 1000.0
    print(f"  反证（校正点 + none_raw 的 P1/P2）：均值 {err_mix.mean():.2f} mm  "
          f"最大 {err_mix.max():.2f} mm（正确闭环 {err.mean():.4f} mm）")

    # ---- epipolar_error 在 none_raw 下必须拒绝 ----
    guard = ""
    try:
        rig.epipolar_error(uv1, uv2)
    except ValueError as exc:
        guard = str(exc)
    print(f"  epipolar_error 护栏：{guard.splitlines()[0][:70] if guard else '**没有报错**'}")

    checks = [
        ("三角化闭环误差均值 < 1 mm", err.mean() < 1.0),
        ("三角化闭环误差最大 < 1 mm", err.max() < 1.0),
        # 门限 0.1 px：undistortPoints 的反解是迭代的，重复调用有 ~0.04 px 数值残差；
        # 真"扣两次畸变"或"右路多转一次 R"会是几百~几千 px，不是这个量级。
        ("rectify_points 幂等（不重复去畸变、右路不多转一次 R）", dd < 0.1),
        ("反证：校正/未校正混用 -> 均值误差 ≥ 10 mm 且 ≥ 20 倍",
         err_mix.mean() >= 10.0 and err_mix.mean() >= 20.0 * max(err.mean(), 1e-6)),
        ("none_raw 下 epipolar_error 抛 ValueError",
         "未做极线校正" in guard and "epipolar_error_by_mode" in guard),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def _legacy_angled_midpoint(ag, K, D, R, T, p_left, p_right):
    """复刻改造前的"两射线最近点中点 + 半间距"，只用于判据的参照值。"""
    ray1 = ag.normalized_ray(K, D, np.asarray(p_left, dtype=np.float64))
    ray2 = ag.normalized_ray(K, D, np.asarray(p_right, dtype=np.float64))
    d1 = ray1
    d2 = R.T @ ray2
    origin2 = -R.T @ T
    a = float(d1 @ d1)
    b = float(d1 @ d2)
    c = float(d2 @ d2)
    det = a * c - b * b
    dv = float(d1 @ origin2)
    ev = float(d2 @ origin2)
    s = (c * dv - b * ev) / det
    t = (b * dv - a * ev) / det
    p1 = s * d1
    p2 = origin2 + t * d2
    return np.asarray(0.5 * (p1 + p2), dtype=np.float64), 0.5 * float(
        np.linalg.norm(p1 - p2)
    )


def test_angled_weighted_triangulation() -> bool:
    """B3：大角度三角化的加权 ML 必须真的抗噪，且等权时退回旧中点。

    本机没有真机同步采集条件，所以只能证明**合成抗噪提升**，不能证明
    真机精度提升。这里刻意把噪声做成各向异性并给右路加一点系统性偏置
    （模拟模糊球心的阈值化偏差），这正是中点法最吃亏、加权 ML 最受益
    的场景；seed 固定，所有数字可复现。

    判据：抗噪 RMS 每组降幅 ≥20%、均值偏差降到旧法一半以下；等权逐位
    等于旧中点；1°/0° 返回 None 而不是 NaN；模块自测仍全绿。
    """
    print("\n----- 大角度三角化：加权 ML vs 旧中点（合成抗噪） -----")
    from stereo import angled_geometry as ag

    K = _K(600.0, 600.0, 320.0, 240.0)
    D = np.zeros(5, dtype=np.float64)
    right_center_in_left = np.array([0.4, 0.0, 0.0], dtype=np.float64)
    # 完全复用 angled_geometry._self_test 的合成点云与交会角区间。
    cases = (
        (
            60.0,
            np.array(
                [[0.00, 0.00, 0.30], [-0.05, 0.05, 0.36], [0.06, -0.04, 0.42]],
                dtype=np.float64,
            ),
        ),
        (
            90.0,
            np.array(
                [[0.04, 0.00, 0.14], [0.06, 0.03, 0.16], [0.02, -0.02, 0.18]],
                dtype=np.float64,
            ),
        ),
    )
    sigma_left = np.array([0.3, 0.3])   # 左路各向同性 0.3 px
    sigma_right = np.array([0.3, 1.0])  # 右路各向异性：x 0.3 px / y 1.0 px
    bias_right = np.array([0.0, 0.3])   # 右路 y 方向 0.3 px 系统性偏置
    n_trials = 800
    rng = np.random.default_rng(20261118)

    checks = []
    for angle_deg, points in cases:
        R = cv2.Rodrigues(np.radians([0.0, angle_deg, 0.0]))[0]
        T = -R @ right_center_in_left
        err_old, err_new = [], []
        ray_angles = []
        for X in points:
            X_right = R @ X + T
            p0 = cv2.projectPoints(X.reshape(1, 1, 3), np.zeros(3), np.zeros(3),
                                   K, D)[0].reshape(2)
            q0 = cv2.projectPoints(X_right.reshape(1, 1, 3), np.zeros(3),
                                   np.zeros(3), K, D)[0].reshape(2)
            nl = rng.normal(0.0, sigma_left, (n_trials, 2))
            nr = rng.normal(0.0, sigma_right, (n_trials, 2))
            for k in range(n_trials):
                p = p0 + nl[k]
                q = q0 + nr[k] + bias_right
                old = ag.triangulate_pixels(K, D, K, D, R, T, p, q)
                new = ag.triangulate_pixels(
                    K, D, K, D, R, T, p, q,
                    sigma_left=sigma_left, sigma_right=sigma_right,
                )
                if old is None or new is None:
                    continue
                ray_angles.append(new.angle_deg)
                err_old.append(old.xyz - X)
                err_new.append(new.xyz - X)
        err_old = np.asarray(err_old)
        err_new = np.asarray(err_new)
        rms_old = float(np.sqrt(np.mean(err_old ** 2)))
        rms_new = float(np.sqrt(np.mean(err_new ** 2)))
        bias_old = float(np.linalg.norm(err_old.mean(axis=0)))
        bias_new = float(np.linalg.norm(err_new.mean(axis=0)))
        drop = 100.0 * (1.0 - rms_new / rms_old)
        print(
            f"  交会角 {min(ray_angles):.1f}-{max(ray_angles):.1f}° "
            f"(相机旋转 {angle_deg:.0f}°): 旧 RMS={rms_old * 1000:.4f} mm, "
            f"新 RMS={rms_new * 1000:.4f} mm, 降幅={drop:.1f}%; "
            f"均值偏差 {bias_old * 1000:.4f} -> {bias_new * 1000:.4f} mm "
            f"({bias_new / max(bias_old, 1e-15):.2f}x)"
        )
        checks.append((f"{angle_deg:.0f}° 抗噪 RMS 降幅 ≥20%", drop >= 20.0))
        checks.append((
            f"{angle_deg:.0f}° 不系统偏（均值偏差 < 旧法 1/2）",
            bias_new < 0.5 * bias_old,
        ))

    # 等权退化：没有不确定度时，新实现必须逐位等于旧中点，且 residual_m 不变。
    max_equal_delta = 0.0
    max_residual_delta = 0.0
    for angle_deg, points in cases:
        R = cv2.Rodrigues(np.radians([0.0, angle_deg, 0.0]))[0]
        T = -R @ right_center_in_left
        for X in points:
            p0 = cv2.projectPoints(X.reshape(1, 1, 3), np.zeros(3), np.zeros(3),
                                   K, D)[0].reshape(2)
            q0 = cv2.projectPoints((R @ X + T).reshape(1, 1, 3), np.zeros(3),
                                   np.zeros(3), K, D)[0].reshape(2)
            for k in range(20):
                p = p0 + rng.normal(0.0, 0.5, 2)
                q = q0 + rng.normal(0.0, 0.5, 2)
                got = ag.triangulate_pixels(K, D, K, D, R, T, p, q)
                if got is None:
                    continue
                legacy_xyz, legacy_res = _legacy_angled_midpoint(
                    ag, K, D, R, T, p, q)
                max_equal_delta = max(max_equal_delta,
                                      float(np.max(np.abs(got.xyz - legacy_xyz))))
                max_residual_delta = max(max_residual_delta,
                                         abs(got.residual_m - legacy_res))
                # 加权路径也不能改 residual_m 的语义。
                weighted = ag.triangulate_pixels(
                    K, D, K, D, R, T, p, q,
                    sigma_left=0.3, sigma_right=[0.3, 1.0])
                if weighted is not None:
                    max_residual_delta = max(
                        max_residual_delta, abs(weighted.residual_m - legacy_res))
    print(f"  等权退化：与旧中点最大差 {max_equal_delta:.3e} m; "
          f"residual_m 最大差 {max_residual_delta:.3e} m")
    checks.append(("等权逐位等于旧中点（浮点误差内）", max_equal_delta < 1e-12))
    checks.append(("residual_m 语义不变（含加权路径）", max_residual_delta < 1e-12))

    # 数值稳定：交会角（两条视线的夹角，不是相机旋转角）0°/≈1° 时必须
    # 返回 None，绝不把 NaN 或病态大深度当有效点抛出。
    stable = True
    R0 = np.eye(3, dtype=np.float64)
    T0 = -right_center_in_left
    center = np.array([320.0, 240.0], dtype=np.float64)
    far = np.array([0.0, 0.0, 25.0], dtype=np.float64)  # 基线 0.4 m -> 视线夹角 ≈0.92°
    degenerate = (
        ("0°（两视线完全平行）",
         R0, T0, center, center),
        ("≈1°（远点时两视线近平行）",
         R0, T0,
         cv2.projectPoints(far.reshape(1, 1, 3), np.zeros(3), np.zeros(3),
                           K, D)[0].reshape(2),
         cv2.projectPoints((far - right_center_in_left).reshape(1, 1, 3),
                           np.zeros(3), np.zeros(3), K, D)[0].reshape(2)),
    )
    for label, R, T, p, q in degenerate:
        for kw in ({}, {"sigma_left": 0.3, "sigma_right": [0.3, 1.0]}):
            got = ag.triangulate_pixels(K, D, K, D, R, T, p, q, **kw)
            stable = stable and got is None
        print(f"  {label}: 返回 None（等权+加权）")
    print(f"  数值稳定：近平行均明确无效 = {stable}")
    checks.append(("0°/1° 不抛异常且明确无效", stable))

    # 旧自测入口仍全绿（无噪声时加权路径不是必须的，旧结果不能被破坏）。
    ag._self_test()
    checks.append(("angled_geometry 模块自测全绿", True))

    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_chain_mode_guard() -> bool:
    """chain 落盘必须是 angled；显式 parallel 必须被 argparse 拒绝（2026-10-24）。

    守的断点：`_run_chain` 原来用 `resolve_auto(...)` —— 新装置没有旧
    `stereo_calibration.json` 时 auto 落到 parallel，于是把
    `meta["stereo_mode"]="parallel"` 与 `angle_deg=90` 一起写盘。下游按
    parallel 重建，`dy ≤ 2.5 px` 门限把**全部候选**拒掉、匹配率 0，且不报错。
    所以必须两条同时成立：落盘值由**方法**决定；显式给 parallel 时入口就拒。
    """
    print("\n----- chain 双目模式护栏 -----")
    import io
    from contextlib import redirect_stderr
    from types import SimpleNamespace

    import apps.calibrate_stereo as app

    args = SimpleNamespace(left_dir="imgs/A", right_dir="imgs/B",
                           board_b_dir=None, pair_frame="imgs/pair",
                           pair_camera="left", stereo_mode="auto")
    # 90° 的 R：夹角由 rig_angle 从 R 算出，不写死 90
    fake_res = SimpleNamespace(
        R=cv2.Rodrigues(np.radians([0.0, 90.0, 0.0]))[0],
        rectification="none_raw", consistency={})
    meta = app._chain_meta(args, BoardSpec(cols=8, rows=5, square_mm=25.0),
                           fake_res, (1280, 720))
    print(f"  chain 落盘 meta：stereo_mode={meta['stereo_mode']!r}  "
          f"angle_deg={meta['angle_deg']:.2f}  "
          f"rectification={meta['rectification']!r}")

    # 显式 --method chain --stereo-mode parallel：argparse 必须报错、退出码 2
    code, buf = None, io.StringIO()
    argv = sys.argv
    try:
        sys.argv = ["calibrate_stereo.py", "--method", "chain",
                    "--stereo-mode", "parallel"]
        with redirect_stderr(buf):
            app.main()
    except SystemExit as exc:
        code = exc.code
    finally:
        sys.argv = argv
    err = buf.getvalue()
    first = next((ln.strip() for ln in err.splitlines() if "none_raw" in ln), err.strip()[:70])
    print(f"  显式 parallel -> 退出码 {code}；报错：{first}")

    checks = [
        ("chain 落盘 stereo_mode 恒为 'angled'（不跟 auto）",
         meta["stereo_mode"] == "angled"),
        ("chain 落盘 rectification 如实为 'none_raw'",
         meta["rectification"] == "none_raw"),
        ("--method chain --stereo-mode parallel -> argparse 报错、退出码 2",
         code == 2 and "none_raw" in err and "parallel" in err),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_same_board_mode() -> bool:
    """**单板模式**：一块固定板，左右各拍一轮 ⇒ `T_AB = I`（2026-10-25）。

    用户实际只有一块板：左相机拍一轮、右相机拍一轮，两轮看的是同一块板的
    同一个世界系。数学上就是 `T_AB = I` 的特例：

        T_{C_R←C_L} = T_{C_R←W} · I · T_{W←C_L}

    这里守六件事：
      1. 链式公式在 `T_AB=I` 下必须**精确**等于真值（<1e-9，不是"差不多"）；
      2. 3 张同摆放的合成参考图 -> 离散度在门限内、链式外参 <1mm/0.5°；
      2b. 参考图里板动了 -> 必须报离散度警告（不能静默出数）；
      2c. 缺 `--right-dir`/`--ref-right` 时必须说清怎么补（不许猜 T_AB）；
      3. `--same-board` 与双板专属参数同时给 -> argparse 退出码 2（不静默忽略）。
    """
    print("\n----- 单板模式（T_AB = I，参考图定外参） -----")
    import subprocess
    import tempfile
    from types import SimpleNamespace

    import apps.calibrate_stereo as app
    import calib.relative_pose as rp

    # ---- 1) T_AB = I 的链式外参 ----
    T_camL_W = rp.rt_to_T(cv2.Rodrigues(np.radians([2.0, -3.0, 1.5]))[0],
                          [0.02, -0.01, 0.45])
    T_camR_W = rp.rt_to_T(cv2.Rodrigues(np.radians([88.0, -2.0, 90.5]))[0],
                          [0.48, 0.03, 0.52])
    T_true = rp.compose(T_camR_W, rp.invert(T_camL_W))
    R_e, T_e, T_e_mat = rp.relative_extrinsics(T_camL_W, np.eye(4), T_camR_W)
    d_mat = float(np.abs(T_e_mat - T_true).max())
    d_rt = max(float(np.abs(R_e - T_true[:3, :3]).max()),
               float(np.abs(np.asarray(T_e).reshape(3) - T_true[:3, 3]).max()))
    print(f"  T_AB=I 链式：|T_camR_camL - 真值|max = {d_mat:.2e}  "
          f"（R/T 分量 {d_rt:.2e}）  基线 {np.linalg.norm(T_true[:3, 3]) * 1000:.1f} mm")

    # ---- 2) 3 张合成参考图：板同一位置 + 小噪声（90° 装置，板朝两路角平分线）----
    # 为什么必须单独一组：内参那堆图里板在移动，左右各挑一张 = 两个不同摆放，
    # 那时 T_AB 不是 I，链出来的外参就错了。这里把"参考图"当独立一份来验。
    spec = BoardSpec(cols=8, rows=5, square_mm=25.0)
    K = _K(900.0, 900.0, 640.0, 360.0)
    D = np.zeros(5)
    size = (1280, 720)

    def _rotY(deg: float) -> np.ndarray:
        a = math.radians(deg)
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)

    T_RL = np.eye(4)                      # 右相机 ← 左相机（90° 装置的真值）
    T_RL[:3, :3] = _rotY(90.0)
    T_RL[:3, 3] = [-0.5, 0.0, 0.6]
    T_bW = np.eye(4)                      # 板在左相机系里的固定摆放
    T_bW[:3, :3] = _rotY(140.0)           # 法向朝两光轴角平分线，两路都看正面
    T_bW[:3, 3] = [0.05, 0.0, 0.45]

    board_img, pad = make_board_image(spec, px_per_m=1200.0)
    rng = np.random.default_rng(7)

    def _write_refs(dirp: Path, T_cam_W: np.ndarray, jitter: float,
                    prefix: str = "ref") -> None:
        dirp.mkdir(parents=True, exist_ok=True)
        for k in range(3):
            Rj = cv2.Rodrigues(np.radians(rng.normal(0, jitter, 3)))[0]
            rv, _ = cv2.Rodrigues(T_cam_W[:3, :3] @ Rj)
            tv = T_cam_W[:3, 3] + rng.normal(0, jitter * 0.004, 3)
            img = render_view(board_img, pad, spec, 1200.0, rv, tv, K, D, size,
                              0.1, 0.3)
            cv2.imwrite(str(dirp / f"{prefix}{k}.png"), img)

    objp = spec.object_points()
    with tempfile.TemporaryDirectory() as td:
        dl, dr = Path(td) / "refL", Path(td) / "refR"
        _write_refs(dl, T_bW, 0.03)                     # 0.03° 抖动 ≈ 手不抖
        _write_refs(dr, rp.compose(T_RL, T_bW), 0.03)
        T_l, info_l = app._ref_board_poses(sorted(dl.glob("*.png")), spec, objp,
                                           K, D, "--ref-left")
        T_r, info_r = app._ref_board_poses(sorted(dr.glob("*.png")), spec, objp,
                                           K, D, "--ref-right")
        est = rp.relative_extrinsics(T_l, np.eye(4), T_r)[2]
        d_rot = rp.rotation_angle_deg(est[:3, :3], T_RL[:3, :3])
        d_tr = float(np.linalg.norm(est[:3, 3] - T_RL[:3, 3])) * 1000.0
        print(f"  3 张参考图：左 n={info_l['n']} 离散 {info_l['rot_deg']:.3f}°/"
              f"{info_l['trans_mm']:.2f} mm；右 n={info_r['n']} 离散 "
              f"{info_r['rot_deg']:.3f}°/{info_r['trans_mm']:.2f} mm")
        print(f"  链式外参 vs 真值：{d_rot:.3f}° / {d_tr:.3f} mm（门限 0.5° / 1 mm）")
        quiet = not info_l["warn"] and not info_r["warn"]
        accurate = d_rot < 0.5 and d_tr < 1.0

        # ---- 2b) 重复性样本 = 逐张参考图各算一次 T_AB（真值恒为 I）----
        smp = app._same_board_samples(info_l, info_r)
        cs = rp.consistency(smp)
        rep_ok = (len(smp) == 3 and cs["n"] == 3
                  and cs["rot_max_deg"] < 0.5 and cs["trans_max_mm"] < 2.0)
        # 只有 1 张参考图时必须如实退回 n=1，不许伪造样本
        one = app._same_board_samples({"n": 1, "poses": [np.eye(4)]},
                                      {"n": 1, "poses": [np.eye(4)]})
        rep_one = len(one) == 1 and np.allclose(one[0], np.eye(4))
        # 只有 1 张参考图时不能印 0.000°/0.00 mm（会被读成"测过且很稳"）
        n1_txt = app._ref_stability_text(
            {"n": 1, "rot_deg": 0.0, "trans_mm": 0.0})
        n3_txt = app._ref_stability_text(
            {"n": 3, "rot_deg": 0.106, "trans_mm": 0.29})
        n1_ok = "未评估" in n1_txt and "0.000" not in n1_txt
        n3_ok = "0.106" in n3_txt and "0.29" in n3_txt
        # ---- 2c) 报告行格式（用假 report 文本验过滤/改写）----
        fake = "\n".join([
            "双目标定结果（chain：独立单目 + 刚性双板公共参考）",
            "右目重投影 RMS: 0.10 px",
            "T_AB 重复性   : n=3  旋转最大偏离 0.106°  平移最大偏离 0.29 mm",
            "两板间距      : 0.00 mm（||T_AB||，用卷尺核对）",
            "  ! T_AB 只测了 1 次：**无法评估重复性**",
            "  → --pair-frame 给一个含 3~5 张同框图的目录",
        ])
        rep = app._chain_report(SimpleNamespace(report=lambda: fake), True,
                                "参考图 左 3 / 右 3")
        fmt_ok = "T_AB          : I（单板；参考图 左 3 / 右 3）" in rep \
            and "由参考图逐张求 T_AB" in rep \
            and "两板间距" not in rep and "--pair-frame" not in rep

        # ---- 3) 参考图里板动了（8° / 40 mm）-> 必须警告或拒绝，不能静默 ----
        dm = Path(td) / "moved"
        _write_refs(dm, T_bW, 0.03)
        moved = T_bW.copy()
        moved[:3, :3] = _rotY(148.0)                    # 板转了 8°
        moved[:3, 3] = [0.07, 0.0, 0.485]               # 且挪了 40 mm
        _write_refs(dm, moved, 0.03, prefix="mv")       # 追加 3 张"动了"的
        try:
            _Tm, info_m = app._ref_board_poses(sorted(dm.glob("*.png")), spec,
                                               objp, K, D, "--ref-left")
            moved_flagged = bool(info_m["warn"])
            how = (f"警告（离散 {info_m['rot_deg']:.2f}°/"
                   f"{info_m['trans_mm']:.1f} mm）")
        except ValueError as exc:
            moved_flagged, how = True, f"拒绝：{str(exc).splitlines()[0][:40]}"
        print(f"  参考图混入板动了的 3 张 -> {how}")

    # ---- 4) 缺 --right-dir / --ref-right 的说明 ----
    msg2 = app._chain_input_error(SimpleNamespace(
        same_board=True, left_dir=str(ROOT), right_dir=str(ROOT),
        ref_left=str(ROOT), ref_right=None))
    has_ref = bool(msg2) and "--ref-right" in msg2 \
        and "不能拿标内参那堆移动中的图" in msg2
    print(f"  缺 --ref-right：{(msg2 or '').splitlines()[-3][:56]}")

    msg = app._chain_input_error(SimpleNamespace(
        same_board=True, left_dir=str(ROOT), right_dir=None,
        ref_left=str(ROOT), ref_right=str(ROOT)))
    print(f"  缺 --right-dir：{(msg or '').splitlines()[0][:56]}")
    has_right = bool(msg) and "--right-dir" in msg and "缺少" in msg

    # ---- 3) 与双板参数互斥 -> exit 2 ----
    proc = subprocess.run(
        [sys.executable, str(ROOT / "apps" / "calibrate_stereo.py"),
         "--same-board", "--pair-frame", "x"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    err = (proc.stderr or "") + (proc.stdout or "")
    mutual = "互斥" in err and "T_AB" in err
    line = next((ln.strip() for ln in err.splitlines() if "互斥" in ln), err.strip()[:70])
    print(f"  --same-board + --pair-frame -> 退出码 {proc.returncode}；{line[:96]}")

    # ---- 4) 帮助里必须有这个开关（否则用户找不到这条路） ----
    help_proc = subprocess.run(
        [sys.executable, str(ROOT / "apps" / "calibrate_stereo.py"), "--help"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    help_txt = (help_proc.stdout or "") + (help_proc.stderr or "")
    has_help = help_proc.returncode == 0 and "--same-board" in help_txt \
        and "--ref-left" in help_txt and "同一块固定板" in help_txt
    print(f"  --help 含 --same-board/--ref-left：{has_help}（退出码 {help_proc.returncode}）")

    checks = [
        ("T_AB=I 时链式外参 = 真值（< 1e-9）", d_mat < 1e-9),
        ("返回的 R/T 与 4x4 自洽（< 1e-9）", d_rt < 1e-9),
        ("3 张同摆放参考图：离散度在门限内（不误报）", quiet),
        ("3 张参考图链式外参 < 0.5° / 1 mm", accurate),
        ("重复性样本 n=3、离散在门限内", rep_ok),
        ("参考图仅 1 张 -> 如实 n=1（不伪造样本）", rep_one),
        ("n=1 印「未评估」而非 0.000°/0.00 mm", n1_ok),
        ("n≥2 印真实离散度", n3_ok),
        ("报告写 T_AB=I（单板；参考图 左 3 / 右 3），无两板专属行", fmt_ok),
        ("参考图里板动了 -> 警告或拒绝（不静默出数）", moved_flagged),
        ("缺 --right-dir 时说明该怎么给", has_right),
        ("缺 --ref-right 时说明「不能用移动中的图」", has_ref),
        ("--same-board + --pair-frame -> 退出码 2 且报「互斥」", proc.returncode == 2 and mutual),
        ("--help 列出 --same-board / --ref-* 及其说明", has_help),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_average_T() -> bool:
    """`average_T`：SE(3) 鲁棒平均（2026-10-26）。

    ## 守的是什么

    `T_AB` 是 chain 上的**乘法因子**，它的误差原样进入外参（不衰减）。以前多帧
    同框图只拿来算离散度、**真正进链的是调用方给的那一帧** —— n 帧的信息被丢成
    1 帧。这里守四件事：

      1. 10 帧（其中 2 帧故意偏 5°/20 mm）聚合后 < 0.5°/2 mm，且离群正好 2 个；
      2. 单帧 -> 原样返回且散布为 0；0 帧 -> `ValueError`；
      3. **对照实验**：同一份数据下"取第一帧"误差 > 2°（证明聚合不是摆设）——
         所以故意让第一帧就是坏帧之一；
      4. 干净数据（0.3°/1 mm 抖动）**不许误杀** —— 否则"聚合"会变成"随机丢帧"。
         这一条卡的是 σ 估计：对"距离"做 MAD 会系统性低估（距离恒正、分布偏斜），
         实测能把 σ=3 mm 的干净 8 帧剔掉 6 帧。
    """
    print("\n----- T_AB 多帧鲁棒聚合（average_T） -----")
    import calib.relative_pose as rp

    T_true = rp.rt_to_T(cv2.Rodrigues(np.radians([12.0, -30.0, 5.0]))[0],
                        [0.21, -0.05, 0.13])

    def _perturb(T, dr_deg: float, dt_mm: float, r) -> np.ndarray:
        out = T.copy()
        out[:3, :3] = cv2.Rodrigues(np.radians(r.normal(0.0, dr_deg, 3)))[0] @ T[:3, :3]
        out[:3, 3] = T[:3, 3] + r.normal(0.0, dt_mm / 1000.0, 3)
        return out

    rng = np.random.default_rng(2026)
    samples = []
    for i in range(10):
        s = _perturb(T_true, 0.1, 0.5, rng)
        if i < 2:                              # 前两帧坏：偏 5° / 20 mm
            s[:3, :3] = cv2.Rodrigues(np.radians([0.0, 5.0, 0.0]))[0] @ s[:3, :3]
            s[:3, 3] = s[:3, 3] + np.array([0.0, 0.0, 0.020])
        samples.append(s)

    T_mean, st = rp.average_T(samples)
    agg_rot = rp.rotation_angle_deg(T_mean[:3, :3], T_true[:3, :3])
    agg_tr = float(np.linalg.norm(T_mean[:3, 3] - T_true[:3, 3])) * 1000.0
    first_rot = rp.rotation_angle_deg(samples[0][:3, :3], T_true[:3, :3])
    first_tr = float(np.linalg.norm(samples[0][:3, 3] - T_true[:3, 3])) * 1000.0
    print(f"  聚合：误差 {agg_rot:.4f}° / {agg_tr:.3f} mm "
          f"（n={st['n']}，使用 {st['used']}，剔除 {st['n_outliers']}："
          f"{st['outlier_index']}）")
    print(f"  对照「取第一帧」：误差 {first_rot:.4f}° / {first_tr:.3f} mm")
    print(f"  散布：旋转 {st['rot_spread_deg']:.3f}° / 平移 {st['trans_spread_mm']:.2f} mm"
          f"（含坏帧）  门限 {st['threshold_rot_deg']:.3f}° / "
          f"{st['threshold_trans_mm']:.2f} mm  σ {st['sigma_rot_deg']:.3f}° / "
          f"{st['sigma_trans_mm']:.2f} mm")

    # ---- 边界：单帧原样返回 / 0 帧必须报错 ----
    T_one, st_one = rp.average_T([T_true])
    one_ok = (np.array_equal(T_one, T_true) and st_one["rot_spread_deg"] == 0.0
              and st_one["trans_spread_mm"] == 0.0 and st_one["n_outliers"] == 0)
    zero_raises = _raises_value_error(lambda: rp.average_T([]))

    # ---- 干净数据不许误杀 ----
    clean = [_perturb(T_true, 0.3, 1.0, rng) for _ in range(6)]
    T_clean, st_clean = rp.average_T(clean)
    clean_err = rp.rotation_angle_deg(T_clean[:3, :3], T_true[:3, :3])
    clean_tr = float(np.linalg.norm(T_clean[:3, 3] - T_true[:3, 3])) * 1000.0
    print(f"  干净 6 帧（0.3°/1 mm 抖动）：误差 {clean_err:.4f}° / {clean_tr:.3f} mm，"
          f"误判离群 {st_clean['n_outliers']} 个")

    checks = [
        ("10 帧（2 坏）聚合误差 < 0.5°", agg_rot < 0.5),
        ("10 帧（2 坏）聚合误差 < 2 mm", agg_tr < 2.0),
        ("离群正好 2 个且就是那两帧", st["n_outliers"] == 2
         and sorted(st["outlier_index"]) == [0, 1] and st["used"] == 8),
        ("对照：取第一帧误差 > 2°（聚合确实有用）", first_rot > 2.0),
        ("单帧 -> 原样返回、散布 0、无离群", one_ok),
        ("0 帧 -> ValueError", zero_raises),
        ("干净 6 帧不误杀离群（σ 估计不偏小）", st_clean["n_outliers"] == 0),
        ("干净 6 帧聚合误差 < 0.5° / 2 mm", clean_err < 0.5 and clean_tr < 2.0),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_scale_self_check() -> bool:
    """外参**尺度**自检（2026-10-26）：重投影 RMS 抓不到的那一类错。

    ## 为什么必须单独验

    `stereoCalibrate`/链式外参的重投影 RMS 只约束"射线方向一致"，**不约束尺度**：
    把基线整体放大 1.1 倍，重投影残差一模一样。所以这里构造真值已知的合成装置，
    再把 `T_{C_R←C_L}` 的平移乘 1.1，看检查能不能抓到。

    守五件事：
      1. 无噪 -> `median_mm < 0.5`；
      2. 尺度 ×1.1 -> `median_mm` 显著变大，且 ≈ `0.1·‖T‖`（随基线放大）；
      3. 同一组 ×1.1 数据下**旋转残差仍为 0** —— 证明只查旋转抓不到尺度；
      4. `square_size_check` 在无噪下量出 25 mm、×1.1 时量出 27.5 mm；
      5. 单目入口如实地把自己标成 `tautological=True`（循环论证，不能当验收）。
    """
    print("\n----- 外参尺度自检（stereo_scale_consistency / square_size_check） -----")
    import calib.relative_pose as rp
    from calib import metric_check

    R_true = cv2.Rodrigues(np.radians([0.0, 90.0, 0.0]))[0]
    T_true = rp.rt_to_T(R_true, [-0.5, 0.0, 0.6])        # T_{C_R←C_L}，基线 781 mm
    rng = np.random.default_rng(5)
    poses_l = [rp.rt_to_T(cv2.Rodrigues(np.radians(rng.uniform(-15, 15, 3)))[0],
                          [0.03, -0.02, 0.45 + 0.02 * i]) for i in range(8)]
    poses_r = [rp.compose(T_true, TL) for TL in poses_l]

    good = metric_check.stereo_scale_consistency(T_true, poses_l, poses_r)
    bad_T = T_true.copy()
    bad_T[:3, 3] *= 1.1
    bad = metric_check.stereo_scale_consistency(bad_T, poses_l, poses_r)
    expect_mm = 0.1 * float(np.linalg.norm(T_true[:3, 3])) * 1000.0
    print(f"  真值外参：median {good['median_mm']:.6f} mm  旋转 {good['rotation_deg']:.6f}°"
          f"  -> {good['verdict']}")
    print(f"  平移 ×1.1：median {bad['median_mm']:.3f} mm（理论 0.1·||T|| = "
          f"{expect_mm:.1f} mm）  旋转 {bad['rotation_deg']:.6f}°  -> {bad['verdict']}")

    # ---- 方格边长回代（三角化后量绝对尺寸）----
    spec = BoardSpec(cols=8, rows=5, square_mm=25.0)
    objp = spec.object_points()
    K = _K(900.0, 900.0, 640.0, 360.0)
    D = np.zeros(5)
    ang = math.radians(140.0)
    R_b = np.array([[math.cos(ang), 0, math.sin(ang)], [0, 1, 0],
                    [-math.sin(ang), 0, math.cos(ang)]])
    T_CL_B = rp.rt_to_T(R_b, [0.05, 0.0, 0.45])

    def _proj(T) -> np.ndarray:
        rvec, _ = cv2.Rodrigues(T[:3, :3])
        pts, _ = cv2.projectPoints(objp, rvec, T[:3, 3].reshape(3, 1), K, D)
        return pts.reshape(-1, 2)

    sq_good = metric_check.square_size_check(
        T_true, objp, _proj(T_CL_B), _proj(rp.compose(T_true, T_CL_B)),
        spec, K, D, K, D)
    sq_bad = metric_check.square_size_check(
        bad_T, objp, _proj(T_CL_B), _proj(rp.compose(T_true, T_CL_B)),
        spec, K, D, K, D)
    print(f"  方格边长：真值外参 {sq_good['square_mm']:.4f} mm"
          f"（{sq_good['error_pct']:+.4f}%） / ×1.1 {sq_bad['square_mm']:.4f} mm"
          f"（{sq_bad['error_pct']:+.2f}%）")

    mono = metric_check.board_scale_error(np.eye(4), objp, [_proj(T_CL_B)], spec, K, D)
    print(f"  单目入口：reproj {mono['reproj_rms_px']:.4f} px  "
          f"tautological={mono['tautological']}  verdict={mono['verdict'][:22]}…")

    checks = [
        ("真值外参：median < 0.5 mm", good["median_mm"] < 0.5),
        ("真值外参：旋转残差 < 0.05°", good["rotation_deg"] < 0.05),
        ("尺度 ×1.1：median > 5 mm", bad["median_mm"] > 5.0),
        ("尺度 ×1.1：median ≈ 0.1·||T||（±1 mm，随基线放大）",
         abs(bad["median_mm"] - expect_mm) < 1.0),
        ("尺度 ×1.1：旋转残差仍 ≈ 0（只查旋转抓不到尺度）",
         bad["rotation_deg"] < 0.05),
        ("square_size_check：无噪量出 25 mm（< 0.5%）",
         abs(sq_good["error_pct"]) < 0.5),
        ("square_size_check：×1.1 量出 27.5 mm（±1%）",
         abs(sq_bad["square_mm"] - 27.5) < 0.275),
        ("单目入口标成 tautological（不冒充尺度证据）",
         mono["tautological"] is True and "无效" in mono["verdict"]),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def _focus_board(board_w: int, sigma: float, hi: int = 255, lo: int = 0) -> np.ndarray:
    """画一块 8x5 内角点的棋盘（板宽精确等于 ``board_w`` px），再按 σ 高斯模糊。

    为什么板宽要"精确等于"：本节要证明 `focus` **不随板的大小漂移**，
    所以三档必须只有尺寸不同，其余（方格比例、边界、对比度）完全一致。
    """
    spec = BoardSpec()
    sq = board_w / float(spec.cols + 1)          # 内角点 8 -> 9 个方格
    pad = int(round(sq * 1.5))
    w = int(round((spec.cols + 1) * sq)) + 2 * pad
    h = int(round((spec.rows + 1) * sq)) + 2 * pad
    img = np.full((h, w), hi, np.uint8)
    for i in range(spec.cols + 1):
        for j in range(spec.rows + 1):
            if (i + j) % 2 == 0:
                x0 = int(round(pad + i * sq))
                y0 = int(round(pad + j * sq))
                img[y0:y0 + int(round(sq)), x0:x0 + int(round(sq))] = lo
    if sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), sigma)
    # 白边：SB 需要棋盘外侧有一圈安静区，否则边缘那圈角点检不稳
    return cv2.copyMakeBorder(img, 80, 80, 80, 80, cv2.BORDER_REPLICATE)


def _focus_of(gray: np.ndarray):
    from calib.board import detect_board as _db
    return _db(gray, BoardSpec(), exhaustive=True)


def test_focus_is_scale_invariant() -> bool:
    """focus = Laplacian 方差 × 格子像素数 应当**不随棋盘大小变化**。

    这是本次换指标的全部理由：旧指标 `sharpness` 随棋盘占的像素数强烈变化，
    板一大就被判成"糊" —— 实测出现"格点已经稳稳检出、却判图模糊拒收"。
    所以这里同时钉两件事：focus 三档相对偏差 ≤15%，而 sharpness 必须差 ≥2 倍
    （后者证明"旧指标确实有尺度偏差"，不是我们凭空换的口径）。
    """
    print("----- focus 尺度不变性（同一模糊 σ=0.8，板宽 240/480/960 px） -----")
    print(f"  {'板宽px':>7} {'格子px':>8} {'sharpness':>10} {'focus':>10}")
    rows = []
    for bw in (240, 480, 960):
        obs = _focus_of(_focus_board(bw, 0.8))
        rows.append((bw, obs.square_px, obs.sharpness, obs.focus))
        print(f"  {bw:>7} {obs.square_px:>8.1f} {obs.sharpness:>10.0f} {obs.focus:>10.0f}")
    bad = [r for r in rows if not r[3]]
    if bad:
        print(f"  [不达标] 这些档没算出 focus（没检出板？）：{[r[0] for r in bad]}")
        return False
    foc = [r[3] for r in rows]
    shr = [r[2] for r in rows]
    foc_spread = (max(foc) - min(foc)) / float(np.mean(foc))
    shr_ratio = max(shr) / max(min(shr), 1e-9)
    checks = [
        (f"focus 三档相对偏差 {foc_spread * 100:.1f}% ≤ 15%", foc_spread <= 0.15),
        (f"sharpness 三档最大/最小 = {shr_ratio:.2f} ≥ 2（证明旧指标有尺度偏差）",
         shr_ratio >= 2.0),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def _gt_corners(board_w: int) -> np.ndarray:
    """同一块合成棋盘的**真值内角点**（像素坐标，(N,1,2) float32）。

    为什么要它：`findChessboardCornersSB` 在 σ≳2.0 时**直接检不出**板
    （实测 240/480/960/1440/1920/2400 六档板宽都是同一个截止点，与板大小无关）。
    而"单调性"要覆盖到 σ=2.5 —— 那一档只能用真值角点算 ROI 指标，
    否则测的就不是"指标随模糊的变化"，而是"SB 的检出上限"。
    """
    spec = BoardSpec()
    sq = board_w / float(spec.cols + 1)
    pad = int(round(sq * 1.5))
    xs = 80 + pad + np.arange(spec.cols, dtype=np.float32) * sq
    ys = 80 + pad + np.arange(spec.rows, dtype=np.float32) * sq
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=1).reshape(-1, 1, 2).astype(np.float32)


def test_focus_is_monotonic_in_blur() -> bool:
    """越糊 focus 越小（σ=0 -> 0.8 -> 1.6 -> 2.5）。

    只要求**单调下降**，不钉具体数值 —— 数值随棋盘图案/插值方式变，
    钉死会让这条测试变成"改一行渲染就红"的脆弱测试。

    σ=2.5 那一档 SB 已经检不出板，用真值角点算指标（见 ``_gt_corners``）。
    """
    print("----- focus 单调性（板宽 960 px，逐级加模糊） -----")
    from calib.board import BoardObservation, _finish

    sigmas = (0.0, 0.8, 1.6, 2.5)
    vals = []
    for s in sigmas:
        gray = _focus_board(960, s)
        obs = _focus_of(gray)
        src = "检出的角点"
        if not obs.found:
            obs = _finish(BoardObservation(), _gt_corners(960), gray, BoardSpec(),
                          precise=True, detector="gt")
            src = "真值角点（SB 已检不出）"
        vals.append(obs.focus)
        print(f"  σ={s:.1f}  {src:<18} 格子 {obs.square_px:5.1f}px  "
              f"sharpness {obs.sharpness:7.1f}  focus {obs.focus:>9.0f}")
    checks = [("每一档都算出了 focus", all(v > 0 for v in vals))]
    mono = all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))
    checks.append(("σ 递增时 focus 严格递减", mono))
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_focus_threshold() -> bool:
    """绝对下限 800（focus）既不能误杀好帧，也要能拦下垃圾帧。

    两个样本都走**真正的 AutoPairCollector**（不是直接比数字）——
    换指标最容易出的事就是"指标换了、判据还连着旧字段"，只有走完整条链路才测得出来。

    * 清晰帧（σ=2.0 以内）：必须收下 —— 这正是用户报的场景"格点已经检出来了却被判糊"；
    * 糊到只剩低对比度的帧（focus≈200）：必须被拒，且原因里带 focus 与阈值。

    ⚠️ 这条**不是**主要的清晰度闸门：任何被检出的帧 focus 都远高于 800，
    所以它实际不触发。真正的闸门是相对判据（见 ``test_focus_relative_gate``）。
    """
    print("----- 绝对下限 800（额外下限，走完整 AutoPairCollector） -----")
    cfg = CaptureConfig()
    cfg.require_stationary = False
    abs_min = float(cfg.min_sharpness)

    def _offer(gray):
        col = AutoPairCollector(BoardSpec(), cfg)
        return col, col.offer(gray, gray)

    sharp_img = _focus_board(960, 1.6)
    o_sharp = _focus_of(sharp_img)
    col1, (ok1, why1) = _offer(sharp_img)
    print(f"  清晰帧 σ=1.6：focus={o_sharp.focus:.0f} 格子 {o_sharp.square_px:.1f}px"
          f"  -> ok={ok1}  {why1}")

    # 糊帧：把黑白压到 150/200（对比度 50）。SB 仍能检出，但棋盘区域几乎没有
    # 高频能量 —— focus 掉到 ~200，正是"糊到不该用来标定"的水平。
    blur_img = _focus_board(960, 1.6, hi=200, lo=150)
    o_blur = _focus_of(blur_img)
    col2, (ok2, why2) = _offer(blur_img)
    print(f"  糊帧（对比度 50）：focus={o_blur.focus:.0f}"
          f" 格子 {o_blur.square_px:.1f}px  -> ok={ok2}  {why2}")

    checks = [
        ("清晰帧 focus 高于绝对下限 800", o_sharp.focus >= abs_min),
        ("清晰帧被收下（旧口径在这里会误杀）", ok1 is True and col1.count == 1),
        ("糊帧 focus 低于绝对下限 800", 0 < o_blur.focus < abs_min),
        ("糊帧被拒，且原因里点明「糊」", ok2 is False and "糊" in why2),
        ("拒收原因带出两个数（focus 与阈值），能照着判断",
         "focus" in why2 and f"{abs_min:.0f}" in why2),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def _focus_stub(focus: float):
    """造一个"检出且清晰度已知"的观测（只给判据需要的字段）。"""
    from calib.board import BoardObservation

    grid = np.stack(np.meshgrid(np.linspace(100, 500, 8),
                                np.linspace(100, 340, 5)), axis=-1)
    corners = grid.reshape(-1, 1, 2).astype(np.float32)
    obs = BoardObservation()
    obs.found = True
    obs.corners = corners
    obs.homography = np.array([[400, 0, 100], [0, 240, 100], [0, 0, 1]], np.float64)
    obs.focus = float(focus)
    obs.sharpness = float(focus) / 20.0
    obs.square_px = 20.0
    obs.area_ratio = 0.30
    obs.margin_px = 100.0
    obs.brightness = 128.0
    obs.detector = "sb"
    return obs


def test_focus_relative_gate() -> bool:
    """相对判据：**闸门必须真的会触发**，同时不误杀整组偏软的帧。

    为什么必须证伪：绝对阈值（旧值 800）在任何被检出的帧上都不触发 ——
    表面有闸、实际全放行。下面第一条检查就是"明显糊的帧确实被拒"，
    如果哪天判据退回绝对阈值，这条会立刻变红。
    """
    print("----- 相对清晰度判据（绝对阈值永不触发 -> 改判相对） -----")
    from calib.capture import focus_gate

    cfg = CaptureConfig()
    kw = dict(min_ratio=cfg.min_focus_ratio, floor=cfg.focus_floor,
              min_samples=cfg.focus_min_samples, abs_min=None)
    hist = [5000.0, 5200.0, 4800.0, 5000.0, 5100.0]
    med = float(np.median(hist))
    n = len(hist)

    ok_low, why_low = focus_gate(1200.0, label="左", median_focus=med,
                                 n_samples=n, **kw)
    print(f"  本组中位数 {med:.0f}；来一帧 focus 1200 -> ok={ok_low}  {why_low}")
    ok_mid, why_mid = focus_gate(2600.0, label="左", median_focus=med,
                                 n_samples=n, **kw)
    print(f"  再来一帧 focus 2600（52%） -> ok={ok_mid}  {why_mid or '收下'}")
    ok_floor, why_floor = focus_gate(180.0, label="左", median_focus=med,
                                     n_samples=n, **kw)
    print(f"  地板：focus 180 -> ok={ok_floor}  {why_floor}")
    ok_cold, why_cold = focus_gate(900.0, label="左", median_focus=5000.0,
                                   n_samples=3, **kw)
    print(f"  冷启动（样本 3<{cfg.focus_min_samples}）：focus 900 -> ok={ok_cold}"
          f"  {why_cold or '收下（没有基准，不做相对判断）'}")

    # 整组都软：5 帧 focus 都是 3200 -> 全部收下，结束时提示一次
    import contextlib
    import io
    import calib.capture as _cap
    cfg_soft = CaptureConfig()
    cfg_soft.require_stationary = False
    cfg_soft.target_pairs = 5
    cfg_soft.diversity_min = 0.0
    col = AutoPairCollector(BoardSpec(), cfg_soft)
    real_detect, real_roi = _cap.detect_board, _cap.detect_board_in_roi
    _cap.detect_board = lambda *a, **k: _focus_stub(3200.0)
    _cap.detect_board_in_roi = lambda *a, **k: _focus_stub(3200.0)
    img = np.zeros((480, 640), np.uint8)
    buf = io.StringIO()
    accepted = []
    try:
        with contextlib.redirect_stdout(buf):
            for _ in range(5):
                accepted.append(col.offer(img, img)[0])
    finally:
        _cap.detect_board, _cap.detect_board_in_roi = real_detect, real_roi
    out = buf.getvalue().strip()
    print(f"  整组 focus 全 3200：收下 {col.count} 对（{accepted}）")
    print(f"  组末提示：{out or '（没有提示）'}")

    checks = [
        ("明显偏糊（24% < 40%）的帧**被拒** —— 闸门真的会触发",
         ok_low is False and "偏糊" in why_low),
        ("拒收文案自证：带 focus / 中位数 / 阈值三个数",
         f"{1200:.0f}" in why_low and f"{med:.0f}" in why_low and "40%" in why_low),
        ("52% 的帧仍被收（不误杀正常帧）", ok_mid is True and why_mid == ""),
        ("绝对地板拦得住垃圾帧，文案点明地板值",
         ok_floor is False and "地板" in why_floor and "300" in why_floor),
        ("样本不足（3<5）时不做相对拒绝（避免冷启动误杀）",
         ok_cold is True and why_cold == ""),
        ("整组偏软（focus 全 3200）全部收下 —— 估计值不否决测量",
         all(accepted) and col.count == 5),
        ("组末打印一次「整体偏软」提示，且不拒收",
         "整体偏软" in out and "3200" in out and col.reject_total == 0),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_pending_metrics_honesty() -> bool:
    """**没算出来/不适用**的指标必须写明原因，不许印 0，也不许省略（2026-10-26）。

    守三条具体的历史谎言：

    1. `depth_sigma_mm` 用 `Z=f·B/d`，而 90°/未校正下"视差"不存在 ——
       以前照旧代公式，**给出一个对不上任何物理量的数**。现在必须 `nan`，
       且报告写"不适用"而不是空着。
    2. `extrinsic_sigma` 失败时返回 `{"error": ...}`，报告必须**原样**打印；
       印 0 或略过都等于说"外参是准的"。
    3. 交叉验证跳过时必须写**为什么**跳过（"--no-cv" / 样本不足 / chain 不适用）。

    这些都是"输出口径"的断言，构造 `StereoCalibrationResult` 直接验，不必跑标定。
    """
    print("\n----- 未算出的指标：必须写原因（不允许 0 / 省略） -----")
    from calib.stereo_calib import StereoCalibrationResult

    K = _K(900.0, 900.0, 640.0, 360.0)
    T = np.array([-0.065, 0.0, 0.0])
    R = np.eye(3)

    ang = StereoCalibrationResult(
        K1=K, K2=K, D1=np.zeros(5), D2=np.zeros(5), R=R, T=T,
        std1=np.array([1.0, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0]),
        rms=0.4, method="chain", stereo_mode="angled", rectification="none_raw")
    ang_text = ang.report()
    dz_ang = ang.depth_sigma_mm(0.6)
    ang_ok = (not np.isfinite(dz_ang)) and "深度不确定度  : 不适用" in ang_text \
        and "Z=fB/d" in ang_text
    print(f"  angled：σ_Z(0.6m)={dz_ang}  报告行="
          f"{next((l for l in ang_text.splitlines() if '深度不确定度' in l), '（缺）')[:60]}")

    # 同一份参数标成 parallel + 已校正 -> 公式仍要能算（不是把这一档废掉）
    par = StereoCalibrationResult(
        K1=K, K2=K, D1=np.zeros(5), D2=np.zeros(5), R=R, T=T,
        std1=np.array([1.0, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0]),
        rms=0.4, method="joint", stereo_mode="parallel",
        rectification="stereo_rectify")
    dz_par = par.depth_sigma_mm(0.6)
    par_ok = np.isfinite(dz_par) and dz_par > 0
    print(f"  parallel：σ_Z(0.6m)={dz_par:.2f} mm（同一份 K/T，公式仍可用）")

    # 外参不确定度：失败 -> 原文打印；成功 -> 两行（含基线项）+ 新旧对照
    err_text = "雅可比奇异：数据不足以确定 6 个自由度"
    ang.extrinsic_sigma = {"error": err_text}
    err_ok = err_text in ang.report()
    print(f"  extrinsic_sigma 失败 -> 报告含原文：{err_ok}")

    sig = {"sigma_baseline_mm": 0.3, "sigma_angle_deg": 0.02,
           "baseline_mm": 65.0, "depth_sigma_mm_at": lambda z: 12.0}
    par.extrinsic_sigma = sig
    t2 = par.report()
    sig_ok = ("σ_B = 0.300 mm" in t2 and "1 m 处 σ_Z ≈ 12.0 mm" in t2
              and "σ_Z(1 m) 对照" in t2 and "无基线项" in t2)
    print(f"  外参不确定度行 + 新旧对照行：{sig_ok}")

    # 交叉验证：跳过必须带原因；成功必须带 k/中位/比值
    par.cv = {"skipped": "--no-cv"}
    skip_ok = "交叉验证      : 跳过（--no-cv）" in par.report()
    par.cv = {"k": 4, "train_rms_median": 0.5, "test_rms_median": 0.7,
              "ratio": 1.4, "verdict": "泛化良好"}
    run_ok = "4 折" in par.report() and "1.40" in par.report()

    # 跨路一致性：n=1 的 0.00 是构造出来的，必须标成"无法评估"而不是证据
    ang.cross_check = {"insufficient": "n=1：外参正是由这一对位姿构造的，残差恒为 0"}
    insuff_ok = "跨路一致性    : 无法评估" in ang.report()
    ang.cross_check = None
    none_ok = "跨路一致性    : 无法计算（缺少两路同板位姿）" in ang.report()
    ang.cross_check = {"median_mm": 0.8, "p95_mm": 1.2, "rot_deg": 0.1, "n": 4}
    num_ok = "中位 0.80 mm / P95 1.20 mm / 0.100 °（n=4）" in ang.report()
    print(f"  跨路一致性：n=1 标未评估={insuff_ok}  缺失写原因={none_ok}  有数印数={num_ok}")

    checks = [
        ("angled/none_raw：σ_Z 为 nan 且报告写「不适用」", ang_ok),
        ("parallel/已校正：同一公式仍能算（没把这一档废掉）", par_ok),
        ("extrinsic_sigma 的错误原文被原样打印", err_ok),
        ("有 σ_B 时印 σ_B/σ_θ/σ_Z(1 m) 且并列新旧口径", sig_ok),
        ("交叉验证跳过时印「跳过（原因）」", skip_ok),
        ("交叉验证成功时印折数/中位/比值", run_ok),
        ("跨路一致性 n=1 -> 标「无法评估」（残差恒 0 不是证据）", insuff_ok),
        ("跨路一致性缺位姿 -> 印「无法计算 + 原因」", none_ok),
        ("跨路一致性有数 -> 印中位/P95/旋转/n", num_ok),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_chain_verdict_180_reasons() -> bool:
    """verdict 不许再用单项式藏住 180° 真因（2026-11-11）。

    14:57 那份「不可用」的大角度标定，真因是同框图棋盘角点顺序整体倒序 /
    A-B 指派帧间翻转 → solvePnP 的绕板法线 180° 等价解 / 反向 T_AB
    （rot_max_deg≈179.60、trans_max_mm≈348.66），但旧 verdict 用
    ``if bad_mono ... elif bad_rep ...`` 让「单目重投影超门限」抢先，把真正
    致命的 180° 完全盖住 —— 用户看到「不可用」却不知道为什么转了 180°。

    这里直接构造一份假结果调 ``_apply_chain_verdict``（生产代码本身），
    钉四件事：并列列出所有命中原因、不误报 180°、grade/ok 与门限不变、
    提示里带的是 ``c`` 里的**实测数值**而不是硬编码。
    """
    print("\n----- chain verdict：不藏 180° 真因 -----")
    from types import SimpleNamespace

    from calib import stereo_calib as sc

    cfg = CalibConfig()

    def verdict_of(err_left, err_right, rot, trans, n=3):
        c = {"n": n, "rot_max_deg": rot, "trans_max_mm": trans, "n_outliers": 2}
        res = SimpleNamespace(
            err_left=err_left, err_right=err_right, consistency=c,
            grade=None, verdict=None, ok=None)
        sc._apply_chain_verdict(res, cfg, c, same_board_TAB=False)
        return res

    checks = []
    # ① 同时 bad_mono + bad_rep：两句短句都要在，且带 180° 机制提示
    r = verdict_of(1.899293205368028, 0.5, 179.59687, 348.66034)
    checks.append(("同时命中 bad_mono+bad_rep：verdict 同时含「单目重投影」与「T_AB 重复性」",
                   "单目重投影" in r.verdict and "T_AB 重复性" in r.verdict))
    checks.append(("180° 专项提示点名机制：同框图 / A-B 指派 / 重拍 / 180",
                   "180" in r.verdict and "同框图" in r.verdict
                   and "A-B 指派" in r.verdict and "重拍" in r.verdict))
    checks.append(("180° 提示带 c 里的实测数值（179.60 / 348.66）",
                   "179.60" in r.verdict and "348.66" in r.verdict))

    # ② 只有 bad_mono（rot 正常）→ 绝不误报 180°
    r_mono = verdict_of(1.9, 0.5, 0.8, 2.0)
    checks.append(("只有 bad_mono、rot 正常时不出现 180° 提示（别误报）",
                   "180" not in r_mono.verdict
                   and "关键真因" not in r_mono.verdict))

    # ③ grade / ok 与三条门限常量值未变（防顺手改判定）
    checks.append(("grade == 「不可用」且 ok is False",
                   r.grade == "不可用" and r.ok is False))
    checks.append(("门限常量未变：CHAIN_ROT_OK_DEG=1.5 / CHAIN_TRANS_OK_MM=5.0 / "
                   "ok_rms=0.80 / good_rms=0.35",
                   sc.CHAIN_ROT_OK_DEG == 1.5 and sc.CHAIN_TRANS_OK_MM == 5.0
                   and cfg.ok_rms == 0.80 and cfg.good_rms == 0.35))

    # ④ 数值取自 c：rot 改成 162.3 -> 应出现 162.30，而不是硬编码的 179.60
    r_alt = verdict_of(1.9, 0.5, 162.3, 348.66)
    checks.append(("rot 改成 162.3 -> verdict 出现 162.30 而非 179.60（不硬编码）",
                   "162.30" in r_alt.verdict and "179.60" not in r_alt.verdict))

    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_board_spec_validity() -> bool:
    """标定板规格切换有效性（2026-11-19）：会话核对 + 结构化落盘 + 友好错误 + GUI 提示。

    七条判据：
      1. `_actions.board_args` / `calibrate_args` 带 --cols/--rows/--square 且随 board 变；
      2. 会话 board_spec={8,5,25} vs argv={11,8,9} -> 明确错误码 2 + 两种修正方式；
      3. 同会话 + --use-session-board -> 用会话规格、不报错；
      4. 老会话（无 board 字段）+ 任意 argv -> 不报错、提示无法核对；
      5. 产物 meta 有结构化 board_spec（int/int/float），board 字符串仍在；
      6. 越界 BoardSpec(cols=31) 在新路径下是友好错误、不是 traceback；
      7. AST 扫 `_gui_stereo._start_calibrate` 里有会话规格比对（非脆弱字符串匹配）。
    """
    print("\n----- 标定板规格核对（chain 会话 vs argv）-----")
    import ast
    import io
    import json
    import tempfile
    from contextlib import redirect_stdout
    from types import SimpleNamespace

    import apps._actions as actions
    import apps.calibrate_stereo as app

    checks: list[tuple[str, bool]] = []

    # ---- 1. argv 带三项板参数，且值随 board 变化 ----
    with tempfile.TemporaryDirectory(prefix="bsv_argv_") as td:
        root = Path(td)
        d_a = actions.new_chain_session(
            board={"cols": "11", "rows": "8", "square": "9"}, root=root)
        d_b = actions.new_chain_session(
            board={"cols": "8", "rows": "5", "square": "25"}, root=root)
        argv_a = actions.calibrate_args(
            "chain", {"cols": "11", "rows": "8", "square": "9"}, dirs=d_a)
        argv_b = actions.calibrate_args(
            "chain", {"cols": "8", "rows": "5", "square": "25"}, dirs=d_b)

        def _flag_value(argv, flag):
            return argv[argv.index(flag) + 1] if flag in argv else None

        got_a = tuple(_flag_value(argv_a, f) for f in ("--cols", "--rows", "--square"))
        got_b = tuple(_flag_value(argv_b, f) for f in ("--cols", "--rows", "--square"))
        board_a = actions.board_args({"cols": "11", "rows": "8", "square": "9"})
        board_b = actions.board_args({"cols": "8", "rows": "5", "square": "25"})
        print(f"  calibrate_args board={{11,8,9}} -> {got_a}；{{8,5,25}} -> {got_b}")
        print(f"  board_args board={{11,8,9}} -> {' '.join(board_a)}")
        checks.append((
            "calibrate_args/board_args 带 --cols/--rows/--square，且值随 board 变化",
            got_a == ("11", "8", "9") and got_b == ("8", "5", "25")
            and " ".join(board_a) == "--cols 11 --rows 8 --square 9 --target 40"
            and " ".join(board_b) == "--cols 8 --rows 5 --square 25 --target 40"))

    # 2/3/4：纯函数核对，喂真实结构的 session dict（不跑标定）
    with tempfile.TemporaryDirectory(prefix="bsv_sess_") as td:
        root = Path(td)
        dirs = actions.new_chain_session(
            board={"cols": "8", "rows": "5", "square": "25"}, root=root)
        sess = json.loads((dirs["left"].parent / "session.json").read_text(encoding="utf-8"))
        print(f"  临时会话 board={sess['board']}  board_spec={sess['board_spec']}")
        argv = {"cols": 11, "rows": 8, "square": 9}

        buf = io.StringIO()
        with redirect_stdout(buf):
            spec_a, code_a = app.resolve_board_spec(argv, sess)
        out_a = buf.getvalue()
        print(f"  核对(无开关): code={code_a} spec={spec_a} -> "
              f"{out_a.strip().splitlines()[0] if out_a.strip() else ''}")
        checks.append((
            "会话规格≠argv -> 明确返回 2，打印两边规格 + 两种修正方式",
            code_a == 2 and spec_a is None
            and all(k in out_a for k in ("8x5", "11x8", "25 mm", "9 mm",
                                          "修正一", "修正二", "--use-session-board"))))

        buf = io.StringIO()
        with redirect_stdout(buf):
            spec_b, code_b = app.resolve_board_spec(argv, sess, use_session_board=True)
        out_b = buf.getvalue()
        print(f"  核对(--use-session-board): code={code_b} spec={spec_b}")
        checks.append((
            "--use-session-board -> 用会话规格 {8,5,25}，不报错",
            code_b == 0 and spec_b == {"cols": 8, "rows": 5, "square_mm": 25.0}
            and "--use-session-board" in out_b))

        old = {"kind": "chain", "session": "legacy-no-board"}
        buf = io.StringIO()
        with redirect_stdout(buf):
            spec_c, code_c = app.resolve_board_spec(argv, old)
        out_c = buf.getvalue()
        print(f"  老会话(无 board 字段): code={code_c} spec={spec_c} -> {out_c.strip()}")
        checks.append((
            "老会话（无 board 字段）-> 不核对、用 argv、提示无法核对",
            code_c == 0 and spec_c == {"cols": 11, "rows": 8, "square_mm": 9.0}
            and "无法核对" in out_c))

        # 老会话但有 board 字符串值（无 board_spec）-> 回退读字符串、按数值比对
        legacy_board = {"board": {"cols": "8", "rows": "5", "square": "25"}}
        buf = io.StringIO()
        with redirect_stdout(buf):
            spec_l, code_l = app.resolve_board_spec(argv, legacy_board)
        print(f"  老会话(仅 board 字符串): code={code_l} -> "
              f"{buf.getvalue().strip().splitlines()[0]}")
        checks.append((
            "老会话仅 board 字符串值 -> 回退读取、数值比对仍判不一致",
            code_l == 2 and spec_l is None))

    # 5. 产物 meta 结构化 board_spec + 保留 board 字符串
    fake_res = SimpleNamespace(
        R=cv2.Rodrigues(np.radians([0.0, 90.0, 0.0]))[0],
        rectification="none_raw", consistency={})
    meta = app._chain_meta(
        SimpleNamespace(left_dir="imgs/A", board_b_dir=None, right_dir="imgs/B",
                        pair_frame=None, pair_camera=None),
        BoardSpec(cols=8, rows=5, square_mm=25.0), fake_res, (1280, 720))
    bs = meta.get("board_spec")
    print(f"  _chain_meta board={meta.get('board')!r}  board_spec={bs!r}")
    checks.append((
        "产物 meta 有结构化 board_spec(int/int/float)，board 字符串仍在",
        isinstance(bs, dict)
        and type(bs.get("cols")) is int and type(bs.get("rows")) is int
        and type(bs.get("square_mm")) is float
        and bs == {"cols": 8, "rows": 5, "square_mm": 25.0}
        and isinstance(meta.get("board"), str) and "8x5" in meta["board"]))

    # 6. 越界输入是友好错误而非 traceback
    buf = io.StringIO()
    with redirect_stdout(buf):
        spec_d, code_d = app.make_board_spec({"cols": 31, "rows": 5, "square_mm": 25})
    out_d = buf.getvalue()
    print(f"  越界 cols=31: code={code_d} spec={spec_d} -> {out_d.strip()}")
    checks.append((
        "越界 BoardSpec(cols=31) -> [参数错误] + 返回 2，不抛 traceback",
        code_d == 2 and spec_d is None
        and "[参数错误]" in out_d and "31" in out_d and "Traceback" not in out_d))

    # 7. AST 扫 GUI 标定回调里的会话规格比对（不是脆弱字符串 includes）
    gui_src = (ROOT / "apps" / "_gui_stereo.py").read_text(encoding="utf-8")
    fn = next((n for n in ast.walk(ast.parse(gui_src))
               if isinstance(n, ast.FunctionDef) and n.name == "_start_calibrate"), None)
    consts = ([n.value for n in ast.walk(fn) if isinstance(n, ast.Constant)]
              if fn is not None else [])
    attrs = ([n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)]
             if fn is not None else [])
    print(f"  AST _start_calibrate: board_spec={('board_spec' in consts)} "
          f"var_cols/rows/square={all(v in attrs for v in ('var_cols','var_rows','var_square'))}")
    checks.append((
        "GUI 标定回调 AST 含 board_spec 比对 + var_cols/rows/square + set_status",
        fn is not None and "board_spec" in consts
        and all(v in attrs for v in ("var_cols", "var_rows", "var_square", "set_status"))))

    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def _raises_value_error(fn) -> bool:
    try:
        fn()
    except ValueError:
        return True
    except Exception:                                     # noqa: BLE001
        return False
    return False


def test_axial_calibration_synthetic() -> bool:
    """模式 A 轴向标定：板正对 + 已知 ΔZ/L 的线性闭合（含退化与反向自证）。

    验证点（对应实现 `calib/axial_calib.py`）：
      1. 无噪声下 fx/fy/Z0 误差 <1%、主点误差 <2 px；
      2. 0.3 px 角点噪声下仍 <1%；
      3. ΔZ 全相同 / 帧数<3 -> 明确 `AxialCalibError`（不出假数）；
      4. 推进方向取反 -> 斜率变负 -> 必须报错（符号敏感）；
      5. 把 fx 公式写成 `L/b`（量纲错）-> 必须远离真值（证明 L 的位置不是可有可无）；
      6. 锁定内参 stereoCalibrate 的基线误差 <5%（R 在纯 Z 推进下弱可观测，不设硬门限）。
    """
    print("\n----- 模式 A 轴向标定（合成真值 + 退化 + 反向自证） -----")
    import calib.relative_pose as rp
    from calib.axial_calib import (
        AxialCalibError,
        solve_axial_intrinsics,
        solve_axial_stereo,
    )

    spec = BoardSpec(cols=9, rows=6, square_mm=40.0)
    objp = spec.object_points()
    L = spec.square_mm / 1000.0
    K = _K(1650.0, 1655.0, 960.0, 540.0)
    D = np.zeros(5)
    Z0 = 0.80
    dZs = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.30])

    def _render(Kx, Dx, delta, tx=0.03, ty=-0.02, noise=0.0, seed=1):
        rng = np.random.default_rng(seed)
        out = []
        for dz in delta:
            pts, _ = cv2.projectPoints(objp, np.zeros(3),
                                       np.array([tx, ty, Z0 + dz]), Kx, Dx)
            pts = pts.reshape(-1, 2)
            if noise:
                pts = pts + rng.normal(0, noise, pts.shape)
            out.append(pts)
        return out

    imgs = _render(K, D, dZs)
    intr = solve_axial_intrinsics(imgs, dZs, spec)
    efx = abs(intr.fx - K[0, 0]) / K[0, 0]
    efy = abs(intr.fy - K[1, 1]) / K[1, 1]
    ez0 = abs(intr.Z0 - Z0) / Z0
    ecx, ecy = abs(intr.cx - K[0, 2]), abs(intr.cy - K[1, 2])
    print(f"  理想：fx {efx * 100:.4f}%  fy {efy * 100:.4f}%  Z0 {ez0 * 100:.4f}%  "
          f"主点 ({ecx:.2f}, {ecy:.2f}) px  rms {intr.rms_px:.4f} px")

    intr_n = solve_axial_intrinsics(_render(K, D, dZs, noise=0.3, seed=7), dZs, spec)
    nfx = abs(intr_n.fx - K[0, 0]) / K[0, 0]
    nfy = abs(intr_n.fy - K[1, 1]) / K[1, 1]
    nz0 = abs(intr_n.Z0 - Z0) / Z0
    print(f"  0.3px 噪声：fx {nfx * 100:.3f}%  fy {nfy * 100:.3f}%  Z0 {nz0 * 100:.3f}%")

    deg_ok = True
    for name, bad_imgs, bad_dz in [
        ("ΔZ 全相同", imgs, np.full(len(imgs), 0.1)),
        ("帧数<3", imgs[:2], dZs[:2]),
    ]:
        try:
            solve_axial_intrinsics(bad_imgs, bad_dz, spec)
            deg_ok = False
            print(f"  退化[{name}]：**没有报错**")
        except AxialCalibError:
            print(f"  退化[{name}]：正确报错")

    rev_ok = False
    try:
        solve_axial_intrinsics(imgs, -dZs, spec)
        print("  反向自证[ΔZ 取反]：**没有报错**")
    except AxialCalibError:
        rev_ok = True
        print("  反向自证[ΔZ 取反]：正确报错（符号敏感）")

    # 反向自证：fx 公式写成 L/b（量纲错）应远离真值
    g = np.array([im.reshape(spec.rows, spec.cols, 2) for im in imgs])
    gaps_all = np.linalg.norm(g[:, :, 1:] - g[:, :, :-1], axis=3).reshape(len(g), -1)
    pitches = np.median(gaps_all, axis=1)
    b_x, _a_x = np.polyfit(dZs, 1.0 / pitches, 1)
    fx_wrong = L / b_x
    wrong_far = abs(fx_wrong - K[0, 0]) / K[0, 0] > 0.5
    print(f"  反向自证[fx=L/b]：错值 {fx_wrong:.1f} vs 真值 {K[0, 0]:.1f}"
          f"（{'远离' if wrong_far else '仍然接近'}）")

    # ---- 外参：锁定内参 stereoCalibrate 的基线 ----
    K1, K2 = K, _K(1610.0, 1618.0, 980.0, 552.0)
    Rcb = cv2.Rodrigues(np.radians([0.6, 0.4, -0.3]))[0]
    c_b = np.array([0.12, -0.005, 0.0])
    T_true = rp.invert(rp.rt_to_T(Rcb, c_b))
    cl, cr = [], []
    for dz in dZs:
        Tl = rp.rt_to_T(np.eye(3), [0.02, -0.01, Z0 + dz])
        Tr = rp.compose(T_true, Tl)
        p1, _ = cv2.projectPoints(objp, np.zeros(3), Tl[:3, 3], K1, D)
        p2, _ = cv2.projectPoints(objp, cv2.Rodrigues(Tr[:3, :3])[0], Tr[:3, 3], K2, D)
        cl.append(p1.reshape(-1, 2))
        cr.append(p2.reshape(-1, 2))
    res = solve_axial_stereo(cl, cr, dZs, spec, (1920, 1080))
    base_true = np.linalg.norm(T_true[:3, 3]) * 1000.0
    base_err = abs(res.baseline_mm - base_true)
    print(f"  外参：基线 {res.baseline_mm:.2f}/{base_true:.2f} mm（误差 {base_err:.3f} mm，"
          f"{base_err / base_true * 100:.2f}%）R 弱可观测性见 warnings；判定 {res.grade}")

    checks = [
        ("理想：fx 误差 <1%", efx < 0.01),
        ("理想：fy 误差 <1%", efy < 0.01),
        ("理想：Z0 误差 <1%", ez0 < 0.01),
        ("理想：主点误差 <2 px", ecx < 2.0 and ecy < 2.0),
        ("0.3px 噪声：fx/fy/Z0 误差均 <1%", nfx < 0.01 and nfy < 0.01 and nz0 < 0.01),
        ("退化：ΔZ 全相同 / 帧数<3 -> AxialCalibError（不出假数）", deg_ok),
        ("反向自证：ΔZ 取反 -> 报错（符号敏感）", rev_ok),
        ("反向自证：fx=L/b（量纲错）-> 明显远离真值", wrong_far),
        ("外参：基线误差 <5%", base_err / base_true < 0.05),
        ("外参：产物 ok=True", bool(res.ok)),
        ("外参：rectification 合法（stereo_rectify 或降级 none_raw）",
         res.rectification in ("stereo_rectify", "none_raw")),
        ("外参：P1/P2 是 3x4 投影矩阵",
         res.P1.shape == (3, 4) and res.P2.shape == (3, 4)),
        ("外参：none_raw 时 P2=K2·[R|T] 与 R/T 自洽（stereo_rectify 由 stereoRectify 产出，不套此式）",
         res.rectification != "none_raw" or np.allclose(
             res.P2, res.K2 @ np.hstack([res.R, np.asarray(res.T).reshape(3, 1)]))),
        ("诚实标注：D 全 0 且 distortion_calibrated=False",
         (not intr.distortion_calibrated) and np.allclose(
             intr.matrix_K_D()[1], 0.0)),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_device90_axis_alignment() -> bool:
    """模式 B 中轴 90°：两板中心重合 + 纯旋转 T_AB 的链式恢复（含反向自证）。

    验证点（对应实现 `calib/device90_calib.py`）：
      1. R 误差 <1°、T 误差 <5%（基线量级）；
      2. 把 90° 改成 80° -> 结果明显变化（证明角度真的进入计算、非硬编码）；
      3. 把 Z_B 改掉 -> T 的对应分量随之变化（证明 Z 真的进入结果）；
      4. 三角化闭环（P1/P2 喂 triangulate_pixels 能还原已知点）；
      5. 反向自证：R_AB 取逆 -> 旋转误差爆；
      6. 反向自证：用角点原点位姿却忘掉 t_AB=(I−R_AB)·c -> 平移误差爆
         （证明"中心原点平移到 0"与"角点原点补平移"必须二选一）。
    """
    print("\n----- 模式 B 中轴 90° 装置（合成真值 + 构型 + 反向自证） -----")
    import calib.relative_pose as rp
    from calib.device90_calib import (
        Device90Diagnostics,
        R_AB_matrix,
        solve_device90,
    )
    from stereo.angled_geometry import triangulate_pixels

    spec = BoardSpec(cols=7, rows=5, square_mm=30.0)     # 7x5：中心正好是角点
    objp = spec.object_points()
    c_board = (objp.min(0) + objp.max(0)) / 2.0
    objc = objp - c_board
    K1, K2 = _K(900.0, 902.0, 640.0, 360.0), _K(980.0, 978.0, 612.0, 352.0)
    D = np.zeros(5)

    R_CB = cv2.Rodrigues(np.radians([0.0, -90.0, 0.0]))[0]   # 相机B光轴=世界-X
    c_B = np.array([0.50, 0.0, 0.50])
    C_B = rp.rt_to_T(R_CB, c_B)
    T_true = rp.invert(C_B)

    Q = np.array([0.03, -0.02, 0.60])
    R_wA = np.eye(3)
    R_AB = R_AB_matrix("y", 90.0)
    R_wB = R_wA @ R_AB.T                                   # 保证 T_AB=[R_AB|0] 精确
    T_Aw = rp.rt_to_T(R_wA, Q)
    T_cB = rp.compose(rp.invert(C_B), rp.rt_to_T(R_wB, Q))
    pA, _ = cv2.projectPoints(objc, np.zeros(3), T_Aw[:3, 3], K1, D)
    pB, _ = cv2.projectPoints(objc, cv2.Rodrigues(T_cB[:3, :3])[0], T_cB[:3, 3], K2, D)
    pA, pB = pA.reshape(-1, 2), pB.reshape(-1, 2)

    diag = Device90Diagnostics()
    res = solve_device90(pA, pB, K1, D, K2, D, spec, (1280, 720),
                         axis="y", angle_deg=90.0, diagnostics=diag)
    base_true = np.linalg.norm(T_true[:3, 3]) * 1000.0
    r_err = rp.rotation_angle_deg(res.R, T_true[:3, :3])
    t_err = np.linalg.norm(res.T.ravel() - T_true[:3, 3]) * 1000.0
    print(f"  90°：R 误差 {r_err:.5f}°  T 误差 {t_err:.4f} mm（基线 {base_true:.1f} mm）"
          f"  Z_A={diag.Z_A:.4f} Z_B={diag.Z_B:.4f}  判定 {res.grade}")

    res80 = solve_device90(pA, pB, K1, D, K2, D, spec, (1280, 720),
                           axis="y", angle_deg=80.0)
    r_err80 = rp.rotation_angle_deg(res80.R, T_true[:3, :3])
    t_err80 = np.linalg.norm(res80.T.ravel() - T_true[:3, 3]) * 1000.0
    print(f"  改 80°：R 误差 {r_err80:.3f}°  T 误差 {t_err80:.2f} mm"
          f"（相对 90° 明显变化即证明角度进入计算）")

    resZ = solve_device90(pA, pB, K1, D, K2, D, spec, (1280, 720),
                          axis="y", angle_deg=90.0, Z_B=diag.Z_B + 0.05)
    dz_mm = (resZ.T.ravel()[2] - res.T.ravel()[2]) * 1000.0
    print(f"  改 Z_B+50mm：T 的 z 分量变化 {dz_mm:.2f} mm（应≈50）")

    # 三角化闭环
    Qw = np.array([0.02, 0.01, 0.55])
    QB = rp.compose(T_true, rp.rt_to_T(np.eye(3), Qw))[:3, 3]
    pa, _ = cv2.projectPoints(Qw.reshape(1, 1, 3), np.zeros(3), np.zeros(3), K1, D)
    pb, _ = cv2.projectPoints(QB.reshape(1, 1, 3), np.zeros(3), np.zeros(3), K2, D)
    tri = triangulate_pixels(K1, D, K2, D, res.R, res.T.ravel(),
                             pa.reshape(2), pb.reshape(2))
    tri_err = float("inf") if tri is None else float(np.linalg.norm(tri.xyz - Qw) * 1000.0)
    print(f"  三角化闭环：误差 {tri_err:.4f} mm")

    def _pnp(objp_, pts, Kx, Dx):
        ok, rv, tv = cv2.solvePnP(objp_.astype(np.float64), pts.reshape(-1, 1, 2),
                                  Kx, Dx, flags=cv2.SOLVEPNP_ITERATIVE)
        assert ok
        return rp.rt_to_T(cv2.Rodrigues(rv)[0], tv.ravel())

    TA_c, TB_c = _pnp(objc, pA, K1, D), _pnp(objc, pB, K2, D)
    # 反向自证 1：R_AB 取逆
    T_bad = rp.compose(TB_c, rp.rt_to_T(np.asarray(R_AB).T, np.zeros(3)),
                       rp.invert(TA_c))
    bad_r = rp.rotation_angle_deg(T_bad[:3, :3], T_true[:3, :3])
    print(f"  反向自证[R_AB 取逆]：R 误差 {bad_r:.2f}°（应爆掉）")
    # 反向自证 2：角点原点位姿 + 忘掉 t_AB
    TA_o, TB_o = _pnp(objp, pA, K1, D), _pnp(objp, pB, K2, D)
    T_bad2 = rp.compose(TB_o, rp.rt_to_T(R_AB, np.zeros(3)), rp.invert(TA_o))
    bad2_t = np.linalg.norm(T_bad2[:3, 3] - T_true[:3, 3]) * 1000.0
    print(f"  反向自证[角点原点忘 t_AB]：T 误差 {bad2_t:.1f} mm（应 >5% 基线）")

    checks = [
        ("90°：R 误差 <1°", r_err < 1.0),
        ("90°：T 误差 <5% 基线", t_err < 0.05 * base_true),
        ("改 80°：结果明显变化（R 误差 >5° 或 T 误差 >10% 基线）",
         (r_err80 > 5.0) or (t_err80 > 0.10 * base_true)),
        ("改 Z_B+50mm：T 的 z 分量变化≈50 mm（Z 真的进入结果）",
         abs(dz_mm - 50.0) < 2.0),
        ("三角化闭环误差 <1 mm", tri_err < 1.0),
        ("反向自证：R_AB 取逆 -> 旋转误差 >90°", bad_r > 90.0),
        ("反向自证：角点原点忘 t_AB -> 平移误差 >5% 基线",
         bad2_t > 0.05 * base_true),
        ("产物：rectification == none_raw（不校正）", res.rectification == "none_raw"),
        ("产物：R1=R2=I 且 P2=K2·[R|T] 自洽",
         np.allclose(res.R1, np.eye(3)) and np.allclose(res.R2, np.eye(3))
         and np.allclose(res.P2, res.K2 @ np.hstack([res.R, np.asarray(res.T).reshape(3, 1)]))),
        ("产物：stereo_mode == angled", res.stereo_mode == "angled"),
        ("产物：ok=True 且 T_AB 平移为 0（中心重合）",
         bool(res.ok) and np.allclose(res.T_AB[:3, 3], 0.0, atol=1e-9)),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_device90_baseline_guard_and_assume_parallel() -> bool:
    """基线降级护栏 + 模式 A `assume_parallel` 强制 R=I 的回归判据。

    覆盖两处本轮新增行为：
      1. device90：装错 axis/angle（板仍正对、残差仍小）但给了
         `expected_baseline_mm` 且基线量级偏离 -> **必须 ok=False / 不可用**
         （证明"只靠残差会误判可用"，护栏是必要的）。
      2. axial `assume_parallel=True` -> `res.R` 严格 I、`P2=K2·[I|T]`、
         `Q=None`、基线仍接近真值；`assume_parallel=False` 仍走 stereo_rectify
         （默认路径未变）。
      3. axial 给错 `expected_baseline_mm` 同样强制降级。
    """
    print("\n----- 基线降级护栏 + assume_parallel（回归 + 反向自证） -----")
    import calib.relative_pose as rp
    from calib.device90_calib import R_AB_matrix, solve_device90
    from calib.axial_calib import solve_axial_stereo

    # ---- device90 装置（合法 axis=y, +90）----
    spec = BoardSpec(cols=7, rows=5, square_mm=30.0)
    objp = spec.object_points()
    c_board = (objp.min(0) + objp.max(0)) / 2.0
    objc = objp - c_board
    K1, K2 = _K(900.0, 902.0, 640.0, 360.0), _K(980.0, 978.0, 612.0, 352.0)
    D = np.zeros(5)
    R_CB = cv2.Rodrigues(np.radians([0.0, -90.0, 0.0]))[0]
    c_B = np.array([0.50, 0.0, 0.50])
    C_B = rp.rt_to_T(R_CB, c_B)
    T_true = rp.invert(C_B)
    Q = np.array([0.03, -0.02, 0.60])
    R_AB = R_AB_matrix("y", 90.0)
    T_Aw = rp.rt_to_T(np.eye(3), Q)
    T_cB = rp.compose(rp.invert(C_B), rp.rt_to_T(R_AB.T, Q))
    pA, _ = cv2.projectPoints(objc, np.zeros(3), T_Aw[:3, 3], K1, D)
    pB, _ = cv2.projectPoints(objc, cv2.Rodrigues(T_cB[:3, :3])[0], T_cB[:3, 3], K2, D)
    pA, pB = pA.reshape(-1, 2), pB.reshape(-1, 2)
    base_true = float(np.linalg.norm(T_true[:3, 3])) * 1000.0

    # 装错 axis/angle 实测只让基线偏 7.8% / 16.9%，**默认 20% 门限抓不住**。
    # 这里显式用 5% 证明"偏离 > tolerance 就降级"的逻辑本身生效；现场应把
    # baseline_tolerance 调到 5%~10%（CalibConfig 默认 20% 是给常规 joint/chain 的，
    # 不应为这两个新模式改动全局默认）。
    cfg_true = CalibConfig(expected_baseline_mm=base_true, baseline_tolerance=0.05)
    res_ok = solve_device90(pA, pB, K1, D, K2, D, spec, (1280, 720),
                            axis="y", angle_deg=90.0, cfg=cfg_true)
    print(f"  合法 axis=y,+90：基线 {res_ok.baseline_mm:.2f}/{base_true:.2f} mm "
          f"ok={res_ok.ok} grade={res_ok.grade}")

    res_x = solve_device90(pA, pB, K1, D, K2, D, spec, (1280, 720),
                           axis="x", angle_deg=90.0, cfg=cfg_true)
    res_rev = solve_device90(pA, pB, K1, D, K2, D, spec, (1280, 720),
                             axis="y", angle_deg=-90.0, cfg=cfg_true)
    dev_x = abs(res_x.baseline_mm - base_true) / base_true * 100
    dev_rev = abs(res_rev.baseline_mm - base_true) / base_true * 100
    print(f"  装错 axis=x：基线 {res_x.baseline_mm:.2f} mm（偏离 {dev_x:.1f}%）"
          f" ok={res_x.ok} grade={res_x.grade}（残差 {res_x.err_left:.2f}/{res_x.err_right:.2f} px）")
    print(f"  装错 angle=-90：基线 {res_rev.baseline_mm:.2f} mm（偏离 {dev_rev:.1f}%）"
          f" ok={res_rev.ok} grade={res_rev.grade}")

    # 反向自证：装错轴时残差**仍然小** -> 只靠残差会误判可用；护栏必须生效
    resid_small = (res_x.err_left <= 0.80 and res_x.err_right <= 0.80)
    print(f"  装错轴的残差仍 <=0.80px = {resid_small}（证明只靠残差会误判 -> 护栏必要）")

    # ---- axial 平行装置数据 ----
    aspec = BoardSpec(cols=9, rows=6, square_mm=40.0)
    aobj = aspec.object_points()
    aL = aspec.square_mm / 1000.0
    aK1 = _K(1650.0, 1655.0, 960.0, 540.0)
    aK2 = _K(1610.0, 1618.0, 980.0, 552.0)
    Z0 = 0.80
    dZs = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.30])
    Rcb = cv2.Rodrigues(np.radians([0.6, 0.4, -0.3]))[0]
    cb = np.array([0.12, -0.005, 0.0])
    T_true_a = rp.invert(rp.rt_to_T(Rcb, cb))
    cl, cr = [], []
    for dz in dZs:
        Tl = rp.rt_to_T(np.eye(3), [0.02, -0.01, Z0 + dz])
        Tr = rp.compose(T_true_a, Tl)
        q1, _ = cv2.projectPoints(aobj, np.zeros(3), Tl[:3, 3], aK1, D)
        q2, _ = cv2.projectPoints(aobj, cv2.Rodrigues(Tr[:3, :3])[0], Tr[:3, 3], aK2, D)
        cl.append(q1.reshape(-1, 2))
        cr.append(q2.reshape(-1, 2))
    base_a = float(np.linalg.norm(T_true_a[:3, 3])) * 1000.0

    res_def = solve_axial_stereo(cl, cr, dZs, aspec, (1920, 1080), assume_parallel=False)
    res_par = solve_axial_stereo(cl, cr, dZs, aspec, (1920, 1080), assume_parallel=True)
    r_is_I = bool(np.allclose(res_par.R, np.eye(3), atol=1e-12))
    p2_ok = bool(np.allclose(
        res_par.P2, res_par.K2 @ np.hstack([np.eye(3), np.asarray(res_par.T).reshape(3, 1)])))
    base_par_err = abs(res_par.baseline_mm - base_a) / base_a
    print(f"  axial assume_parallel=False：rect={res_def.rectification} "
          f"Q is None={res_def.Q is None} 基线 {res_def.baseline_mm:.2f} mm")
    print(f"  axial assume_parallel=True ：R==I {r_is_I}  rect={res_par.rectification} "
          f"Q is None={res_par.Q is None} 基线 {res_par.baseline_mm:.2f}/{base_a:.2f} mm "
          f"（{base_par_err * 100:.2f}%）")

    cfg_a_true = CalibConfig(expected_baseline_mm=base_a, baseline_tolerance=0.20)
    cfg_a_bad = CalibConfig(expected_baseline_mm=base_a * 2.0, baseline_tolerance=0.20)
    res_a_ok = solve_axial_stereo(cl, cr, dZs, aspec, (1920, 1080), cfg_a_true)
    res_a_bad = solve_axial_stereo(cl, cr, dZs, aspec, (1920, 1080), cfg_a_bad)
    print(f"  axial 预期基线正确：ok={res_a_ok.ok} grade={res_a_ok.grade}；"
          f"预期基线错一倍：ok={res_a_bad.ok} grade={res_a_bad.grade}")

    checks = [
        ("合法装轴 + 预期基线正确 -> 不被误杀（ok=True）", bool(res_ok.ok)),
        ("装错 axis=x + 预期基线 -> 强制降级 ok=False / 不可用",
         (not res_x.ok) and res_x.grade == "不可用"
         and "基线" in res_x.verdict and "axis/angle" in res_x.verdict),
        ("装错 angle=-90 + 预期基线 -> 强制降级", (not res_rev.ok) and res_rev.grade == "不可用"),
        ("反向自证：装错轴时残差仍小（只靠残差会误判，护栏必要）", resid_small),
        ("未给 expected_baseline 时不误杀（默认 None 不受影响）",
         bool(solve_device90(pA, pB, K1, D, K2, D, spec, (1280, 720),
                             axis="y", angle_deg=90.0).ok)),
        ("assume_parallel=False 默认仍走 stereo_rectify（回归）",
         res_def.rectification == "stereo_rectify" and res_def.Q is not None),
        ("assume_parallel=True：R 严格等于 I（atol=1e-12）", r_is_I),
        ("assume_parallel=True：P2=K2·[I|T] 且 Q=None",
         p2_ok and res_par.Q is None and res_par.rectification == "none_raw"),
        ("assume_parallel=True：基线仍接近真值（<5%）", base_par_err < 0.05),
        ("assume_parallel=True：warnings 说明已强制 R=I",
         any("assume_parallel" in w for w in res_par.warnings)),
        ("axial 预期基线正确 -> ok=True", bool(res_a_ok.ok)),
        ("axial 预期基线错一倍 -> 强制降级 ok=False",
         (not res_a_bad.ok) and res_a_bad.grade == "不可用"),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_cli_axial_device90_parsing() -> bool:
    """CLI 接线判据（不真读相机）：ΔZ 解析/扫描、argparse 默认、坏 intrinsics。

    覆盖两个新独立入口：
      - `apps/calibrate_axial.py`：目录名 ΔZ 解析、显式 --delta-z、重复/数量/解析失败；
      - `apps/calibrate_device90.py`：内参读取失败要明确报错（不 D=0 兜底）；
      - 两个入口默认 `baseline_tolerance` 必须是 **0.05**（而不是 joint/chain 的 0.20）。
    """
    print("\n----- CLI: 轴向 / 中轴90° 入口（解析 + 默认值 + 反向自证） -----")
    import shutil
    import tempfile
    from pathlib import Path as _P
    from apps.calibrate_axial import (
        _parse_dz, _scan_depth_dirs, build_parser as axial_parser)
    from apps.calibrate_device90 import _load_intrinsics, build_parser as d90_parser

    # 1. ΔZ 解析
    cases = {"z000": 0.0, "z010mm": 0.010, "z-05mm": -0.005, "0mm": 0.0,
             "10": 0.010, "z1.5cm": 0.015, "run2_z030mm": 0.030}
    parsed_ok = True
    for name, want in cases.items():
        got = _parse_dz(name)
        if got is None or abs(got - want) > 1e-9:
            parsed_ok = False
            print(f"  [ΔZ] {name!r} -> {got}（应 {want}）")
    print(f"  ΔZ 解析样例 {len(cases)} 个：{'全对' if parsed_ok else '有错'}")

    # 2. 三种目录命名 -> 相同 ΔZ 序列
    def _make_tree(names):
        r = _P(tempfile.mkdtemp(prefix="dz_"))
        for n in names:
            (r / n / "left").mkdir(parents=True)
            (r / n / "right").mkdir()
        return r

    seqs, roots = [], []
    for names in (["z000", "z010", "z020", "z030"],
                  ["0mm", "10mm", "20mm", "30mm"],
                  ["z000mm", "z010mm", "z020mm", "z030mm"]):
        r = _make_tree(names)
        roots.append(r)
        items = _scan_depth_dirs(r, "left", "right", None)
        seqs.append([round(it["dz_m"] * 1000, 6) for it in items])
    same_seq = (seqs[0] == seqs[1] == seqs[2] == [0, 10, 20, 30])
    print(f"  三种命名序列：{seqs} -> 一致={same_seq}")

    # 3. 显式 --delta-z 与目录名解析一致
    explicit_seq = [round(it["dz_m"] * 1000, 6) for it in
                    _scan_depth_dirs(roots[0], "left", "right", [0.0, 0.01, 0.02, 0.03])]
    print(f"  显式 --delta-z 序列：{explicit_seq}")

    # 4. 重复 / 数量<3 / 解析失败 -> ValueError
    def _bad(names, dz=None):
        r = _make_tree(names)
        try:
            _scan_depth_dirs(r, "left", "right", dz)
            return None
        except ValueError as exc:
            return str(exc)
        finally:
            shutil.rmtree(r, ignore_errors=True)

    dup_msg = _bad(["z000", "z000b", "z010", "z020"])
    few_msg = _bad(["z000", "z010"])
    badname_msg = _bad(["aaa", "bbb", "ccc"])
    print(f"  重复：{'报错' if dup_msg else '未报错'}；数量<3：{'报错' if few_msg else '未报错'}；"
          f"解析失败：{'报错' if badname_msg else '未报错'}")

    # 5. argparse 默认值
    a = axial_parser().parse_args(["--root", "x"])
    d = d90_parser().parse_args(["--left-dir", "a", "--right-dir", "b", "--intrinsics", "c"])
    print(f"  argparse 默认：axial tol={a.baseline_tolerance} assume_parallel={a.assume_parallel}；"
          f"d90 tol={d.baseline_tolerance} axis={d.rot_axis} angle={d.rot_angle}")

    # 6. 坏 intrinsics -> (None, 原因)
    loaded, reason = _load_intrinsics("__definitely_not_exist__")
    bad_intr_ok = (loaded is None and bool(reason))
    print(f"  坏 intrinsics：{'明确报错' if bad_intr_ok else '未报错'}（{reason[:34]}…）")

    for r in roots:
        shutil.rmtree(r, ignore_errors=True)

    checks = [
        ("ΔZ 解析：z000/z010mm/z-05mm/0mm/10/1.5cm/run2_z030mm 全对", parsed_ok),
        ("三种目录命名 -> 相同 ΔZ 序列 [0,10,20,30] mm", same_seq),
        ("显式 --delta-z 与目录名解析得到相同序列", explicit_seq == [0, 10, 20, 30]),
        ("重复深度（z000/z000b）-> ValueError", dup_msg is not None),
        ("深度数量<3 -> ValueError", few_msg is not None),
        ("目录名解析失败 -> ValueError（不静默跳过）", badname_msg is not None),
        ("CLI 默认 baseline_tolerance == 0.05（不是全局 0.20）",
         abs(a.baseline_tolerance - 0.05) < 1e-12
         and abs(d.baseline_tolerance - 0.05) < 1e-12),
        ("device90 CLI 默认 rot_axis=y / rot_angle=90",
         d.rot_axis == "y" and abs(d.rot_angle - 90.0) < 1e-12),
        ("bad intrinsics -> (None, 原因)（不静默 D=0 兜底）", bad_intr_ok),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def test_device90_intrinsics_sanity() -> bool:
    """device90 内参 sanity：硬错误 exit 2、软告警不误杀几何、CLI 退出码。

    真实产物实测 K1 fx=2304 / K2 fx=1541（差 33%），而平行装置只差 0.7%/0.26%。
    device90 吃外部内参，内参错外参必错，但**内参可疑不等于几何错**：软告警只进
    warnings/meta，不得把 res.ok 判 False。
    """
    print("\n----- device90 内参 sanity（硬错误 + 软告警 + 隔离几何） -----")
    import shutil
    import tempfile
    from pathlib import Path as _P
    from calib.device90_calib import R_AB_matrix, solve_device90
    import calib.relative_pose as rp
    from apps.calibrate_device90 import (
        _check_intrinsics_hard,
        _intrinsics_sanity_reasons,
        main as d90_main,
    )

    def K_of(a, b, c, d):
        return np.array([[a, 0, c], [0, b, d], [0, 0, 1.0]])

    zero5 = np.zeros(5)
    Kgood = K_of(1650, 1655, 960, 540)
    bad_fx = _check_intrinsics_hard(K_of(0, 1655, 960, 540), zero5, Kgood, zero5)
    bad_nan = _check_intrinsics_hard(np.full((3, 3), np.nan), zero5, Kgood, zero5)
    bad_shape = _check_intrinsics_hard(np.eye(2), zero5, Kgood, zero5)
    good = _check_intrinsics_hard(Kgood, zero5, Kgood, zero5)
    hard_ok = bool(bad_fx) and bool(bad_nan) and bool(bad_shape) and (good == "")
    print(f"  硬错误：fx=0->{'报' if bad_fx else '漏'} NaN->{'报' if bad_nan else '漏'} "
          f"非3x3->{'报' if bad_shape else '漏'} 合法->{'空' if not good else '误报'}")

    # 软告警
    K1 = K_of(2304.11, 2304.5, 960, 540)
    K2 = K_of(1540.79, 1541.0, 955, 545)
    reasons = _intrinsics_sanity_reasons(K1, zero5, K2, zero5, width=1920)
    fx_warn = any("fx 相差" in r for r in reasons)
    d_warn = any("畸变" in r for r in _intrinsics_sanity_reasons(
        K1, np.array([1.5, 0, 0, 0, 0]), K2, zero5, None))
    c_warn = any("主点" in r for r in _intrinsics_sanity_reasons(
        K_of(1650, 1655, 5000, 540), zero5, K_of(1650, 1655, 5000, 540), zero5, 1920))
    print(f"  软告警：fx差33%={fx_warn} |D|>1.0={d_warn} cx越界={c_warn}")

    # 几何不误杀：用同一对"33% 内参"生成自洽角点，solve 仍应 ok=True
    spec = BoardSpec(cols=7, rows=5, square_mm=30.0)
    objp = spec.object_points()
    c_board = (objp.min(0) + objp.max(0)) / 2.0
    objc = objp - c_board
    C_B = rp.rt_to_T(cv2.Rodrigues(np.radians([0.0, -90.0, 0.0]))[0],
                     np.array([0.50, 0.0, 0.50]))
    Q = np.array([0.03, -0.02, 0.60])
    R_AB = R_AB_matrix("y", 90.0)
    T_Aw = rp.rt_to_T(np.eye(3), Q)
    T_cB = rp.compose(rp.invert(C_B), rp.rt_to_T(R_AB.T, Q))
    pA, _ = cv2.projectPoints(objc, np.zeros(3), T_Aw[:3, 3], K1, zero5)
    pB, _ = cv2.projectPoints(objc, cv2.Rodrigues(T_cB[:3, :3])[0], T_cB[:3, 3], K2, zero5)
    res = solve_device90(pA.reshape(-1, 2), pB.reshape(-1, 2), K1, zero5, K2, zero5,
                         spec, (1280, 720), axis="y", angle_deg=90.0)
    geo_ok = bool(res.ok) and fx_warn
    print(f"  几何不误杀：res.ok={res.ok} grade={res.grade}（内参可疑但角点自洽）")

    # CLI 端到端：fx=0 的内参文件 -> exit 2（硬检查在读图前触发）
    tmp = _P(tempfile.mkdtemp(prefix="intr_bad_"))
    rc = None
    try:
        np.savez(str(tmp / "bad.npz"), K1=np.zeros((3, 3)), D1=zero5,
                 K2=Kgood, D2=zero5, R=np.eye(3), T=np.zeros((3, 1)),
                 image_size=np.array([1920, 1080]))
        rc = d90_main(["--left-dir", "x", "--right-dir", "y",
                       "--intrinsics", str(tmp / "bad")])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"  CLI fx=0 内参 -> 退出码 {rc}")

    checks = [
        ("硬错误：fx=0 / NaN / 非3x3 都报错，合法通过", hard_ok),
        ("软告警：左右 fx 差 33% 命中", fx_warn),
        ("软告警：|D|>1.0 命中", d_warn),
        ("软告警：主点越界命中", c_warn),
        ("内参可疑但几何自洽 -> res.ok 仍 True（不误杀）", geo_ok),
        ("CLI fx=0 内参 -> exit 2", rc == 2),
    ]
    for name, ok in checks:
        print(f"  [{'通过' if ok else '不达标'}] {name}")
    return all(ok for _n, ok in checks)


def main() -> int:
    ap = argparse.ArgumentParser(description="双目标定自测（合成真值）")
    ap.add_argument("--views", type=int, default=25,
                    help="渲染多少个位姿（少于 min_pairs=10 对无法标定）")
    ap.add_argument("--noise", type=float, default=0.6, help="图像噪声标准差")
    ap.add_argument("--blur", type=float, default=0.8, help="高斯模糊 sigma")
    ap.add_argument("--distortion", action="store_true", help="带镜头畸变")
    ap.add_argument(
        "--stills",
        action="store_true",
        help="按独立静帧测试（不要求板静止）；默认按视频测试",
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print("=" * 62)
    print("双目标定自测（合成真值）")
    print("=" * 62)
    cases = [
        ("理想：无畸变无噪声", dict(noise=0.0, use_distortion=False, blur=0.0)),
        (
            "现实：有畸变 + 噪声 + 轻微模糊",
            dict(noise=args.noise, use_distortion=True, blur=args.blur),
        ),
    ]
    if args.distortion:
        cases = [cases[1]]
    all_ok = True
    for name, kw in cases:
        print(f"\n----- {name} -----")
        ok = run_case(
            args.views,
            kw["noise"],
            kw["use_distortion"],
            kw["blur"],
            args.verbose,
            stills=args.stills,
        )
        all_ok = all_ok and ok
    all_ok = test_T_sign_convention() and all_ok
    all_ok = test_board_area_caps_grade() and all_ok
    all_ok = test_epipolar_rejects_mismatch() and all_ok
    all_ok = test_fov_or_none() and all_ok
    all_ok = test_chain_extrinsics_90deg() and all_ok
    all_ok = test_chain_frame_algebra() and all_ok
    all_ok = test_pair_frame_T_AB_roundtrip() and all_ok
    all_ok = test_pair_prior_guard_double_flip() and all_ok
    all_ok = test_force_save_backup_and_dialog() and all_ok
    all_ok = test_chain_triangulation_closure() and all_ok
    all_ok = test_angled_weighted_triangulation() and all_ok
    all_ok = test_chain_mode_guard() and all_ok
    all_ok = test_same_board_mode() and all_ok
    all_ok = test_average_T() and all_ok
    all_ok = test_scale_self_check() and all_ok
    all_ok = test_pending_metrics_honesty() and all_ok
    all_ok = test_focus_is_scale_invariant() and all_ok
    all_ok = test_focus_is_monotonic_in_blur() and all_ok
    all_ok = test_focus_threshold() and all_ok
    all_ok = test_focus_relative_gate() and all_ok
    all_ok = test_chain_verdict_180_reasons() and all_ok
    all_ok = test_board_spec_validity() and all_ok
    all_ok = test_axial_calibration_synthetic() and all_ok
    all_ok = test_device90_axis_alignment() and all_ok
    all_ok = test_device90_baseline_guard_and_assume_parallel() and all_ok
    all_ok = test_cli_axial_device90_parsing() and all_ok
    all_ok = test_device90_intrinsics_sanity() and all_ok
    print()
    print("判定结果：" + ("全部通过" if all_ok else "有不达标项"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
