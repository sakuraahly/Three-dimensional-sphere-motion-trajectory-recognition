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
    print()
    print("判定结果：" + ("全部通过" if all_ok else "有不达标项"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
