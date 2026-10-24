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

from calib import AutoPairCollector, BoardSpec, CalibConfig, calibrate_stereo
from calib.params import StereoRig


# --------------------------------------------------------------------------
# 合成双目装置
# --------------------------------------------------------------------------
def _K(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _forward_distort_map(K, D, size, iters: int = 12):
    """构造"把无畸变图变成畸变图"的 remap 映射（OpenCV 只提供反方向）。

    做法：对每个目标像素 q 迭代求解 p，使 forward(p) = q。
    径向畸变单调时几步就收敛。
    """
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
    return (px * K[0, 0] + K[0, 2]).astype(np.float32), (py * K[1, 1] + K[1, 2]).astype(
        np.float32
    )


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


def _raises_value_error(fn) -> bool:
    try:
        fn()
    except ValueError:
        return True
    except Exception:                                     # noqa: BLE001
        return False
    return False


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
    all_ok = test_chain_triangulation_closure() and all_ok
    all_ok = test_chain_mode_guard() and all_ok
    print()
    print("判定结果：" + ("全部通过" if all_ok else "有不达标项"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
