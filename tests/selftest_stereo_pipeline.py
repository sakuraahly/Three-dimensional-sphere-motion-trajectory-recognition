"""双目链路自测：同步 + 极线配对 + 三角测量。

用已知真值的合成双目装置验证 stereo/ 包：
  1. 同步：造一段带闪光的亮度序列，把其中一路人为平移 k 帧，看能不能找回来；
  2. 配对 + 三角测量：把已知三维点投到两路，再让程序反算回去，看误差。

    python tests/selftest_stereo_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from calib import StereoRig
from stereo import (
    Kalman3D,
    MatchConfig,
    StereoReconstructor,
    SyncConfig,
    find_offset,
    fuse_depth,
    match_epipolar,
    mono_depth,
    savgol_trajectory,
    stereo_depth,
    timestamp_offset,
)
from sphere_tracker.types import Detection


def _K(fx, fy, cx, cy):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)


def make_rig(size=(1280, 720)):
    K1 = _K(900.0, 900.0, 640.0, 360.0)
    K2 = _K(880.0, 880.0, 620.0, 355.0)
    D1 = np.zeros(5)
    D2 = np.zeros(5)
    R, _ = cv2.Rodrigues(np.radians([0.5, -1.0, 0.3]))
    T = np.array([-0.100, 0.0015, 0.002], np.float64).reshape(3, 1)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.0
    )
    return StereoRig(K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T, image_size=size,
                     R1=R1, R2=R2, P1=P1, P2=P2, Q=Q), (K1, D1, K2, D2, R, T)


# --------------------------------------------------------------------------
def test_sync() -> bool:
    print("----- 视频同步 -----")
    rng = np.random.default_rng(7)
    n = 240
    base = 120.0 + rng.normal(0, 0.8, n)
    for f in (40, 95, 160):          # 三次闪光
        base[f] += 45.0
    ok = True
    for k in (0, 8, -11, 37):
        left = np.concatenate([np.full(k, 120.0), base])[:n] if k > 0 else \
               np.concatenate([base[-k:], np.full(-k, 120.0)])[:n] if k < 0 else base.copy()
        res = find_offset(left, base, SyncConfig(max_offset=100))
        # left[i] = base[i-k]  =>  left[i] 对应 right[i-k]，所以偏移应是 -k
        good = (res.offset == -k)
        ok = ok and good
        print(f"  人为平移 {k:+3d} 帧 -> 检出 {res.offset:+3d} 帧  "
              f"置信度 {res.confidence:.3f}  {'通过' if good else '不达标'}")
    # 闪光只在一路明显时也要数得到（另一路曝光不同）
    right_only = np.full(n, 120.0)
    right_only[40] += 45.0
    res = find_offset(np.full(n, 120.0), right_only, SyncConfig(max_offset=100))
    good = res.peak_count >= 1
    ok = ok and good
    print(f"  仅右路有闪光 -> peak_count={res.peak_count}  "
          f"{'通过' if good else '不达标'}")

    # 没有闪光时的自动兜底：右路整体晚 3 帧（时间戳晚 3/fps），
    # left[i] 应配 right[i-3]，即 offset=-3。
    t0 = 1000.0
    left_t = t0 + np.arange(n) / 30.0
    right_t = t0 + 3.0 / 30.0 + np.arange(n) / 30.0 + rng.normal(0, 0.001, n)
    ts = timestamp_offset(left_t, right_t, 30.0)
    good = ts.offset == -3 and ts.usable
    ok = ok and good
    print(f"  时间戳兜底：右路晚 3 帧 -> offset={ts.offset:+d}，"
          f"中位差 {ts.median_abs_ms:.2f} ms  {'通过' if good else '不达标'}")
    return ok


def test_triangulate() -> bool:
    print("----- 极线配对 + 三角测量 -----")
    rig, (K1, D1, K2, D2, R, T) = make_rig()
    rvec_R, _ = cv2.Rodrigues(R)
    rng = np.random.default_rng(11)

    pts3d = np.array([
        [0.00, 0.00, 0.60], [0.06, 0.02, 0.55], [-0.05, 0.03, 0.70],
        [0.02, -0.04, 0.50], [-0.07, -0.02, 0.65], [0.04, 0.05, 0.75],
        [0.10, -0.06, 0.45], [-0.10, 0.07, 0.80],
    ], np.float64)
    uv1 = cv2.projectPoints(pts3d, np.zeros(3), np.zeros(3), K1, D1)[0].reshape(-1, 2)
    uv2 = cv2.projectPoints(pts3d, rvec_R, T, K2, D2)[0].reshape(-1, 2)
    # 检测噪声（真实检测器就是亚像素量级）
    uv1 += rng.normal(0, 0.2, uv1.shape)
    uv2 += rng.normal(0, 0.2, uv2.shape)

    rec = StereoReconstructor(rig, MatchConfig(max_epipolar_px=3.0))
    errs = []
    for i in range(len(pts3d)):
        left = Detection(x=float(uv1[i, 0]), y=float(uv1[i, 1]), radius=20.0, score=1.0)
        # 右图给 3 个候选：正确的那个 + 两个干扰
        cands = [
            Detection(x=float(uv2[i, 0] + rng.normal(0, 40)), y=float(uv2[i, 1] + rng.normal(0, 40)),
                      radius=20.0, score=0.8),
            Detection(x=float(uv2[i, 0]), y=float(uv2[i, 1]), radius=20.0, score=0.9),
            Detection(x=float(uv2[i, 0] + rng.normal(0, 30)), y=float(uv2[i, 1] + rng.normal(0, 30)),
                      radius=20.0, score=0.7),
        ]
        pt = rec.update(t=i * 0.01, left=left, right_dets=cands)
        if pt is None:
            print(f"  点 {i}: 没配上")
            continue
        # 三角测量结果在【校正后】左相机系，真值要乘 R1 换算
        truth = rig.R1 @ pts3d[i]
        errs.append(float(np.linalg.norm(pt.xyz - truth)))

    print(f"  {rec.summary()}")
    if not errs:
        return False
    mean_mm = float(np.mean(errs)) * 1000
    max_mm = float(np.max(errs)) * 1000
    print(f"  三维误差：均值 {mean_mm:.3f} mm，最大 {max_mm:.3f} mm")
    checks = [
        ("配对率 100%", rec.match_rate >= 0.999),
        ("三维误差均值 <= 5 mm", mean_mm <= 5.0),
        ("三维误差最大 <= 10 mm", max_mm <= 10.0),
    ]
    ok = True
    for name, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {name}")
        ok = ok and passed
    return ok


def test_trajectory_fusion() -> bool:
    """第 3 节：三维平滑 / Kalman / 深度融合的纯数学行为。"""
    print("----- 三维平滑 / Kalman / 深度融合 -----")
    rng = np.random.default_rng(7)
    n, fps = 60, 30.0
    t = np.arange(n) / fps
    truth = np.column_stack([0.20 * t, 0.10 + 0.05 * t, 1.00 + 0.30 * t])
    noisy = truth + rng.normal(0.0, 0.004, truth.shape)

    smooth = savgol_trajectory(t, noisy, window=11, polyorder=2)
    rmse_before = float(np.sqrt(np.mean((noisy - truth) ** 2)))
    rmse_after = float(np.sqrt(np.mean((smooth - truth) ** 2)))

    kf = Kalman3D(process_var=1e-5, measure_var=2e-5)
    for ti, p in zip(t, noisy):
        kf.update(float(ti), p)
    t_ahead = t[-1] + 0.2
    pred = kf.predict(float(t_ahead))
    truth_ahead = np.array([0.20 * t_ahead, 0.10 + 0.05 * t_ahead, 1.00 + 0.30 * t_ahead])
    pred_err = float(np.linalg.norm(pred - truth_ahead)) if pred is not None else 1e9

    z_s = stereo_depth(90.0, 900.0, 0.100)          # 900*0.1/90 = 1.000
    z_m = mono_depth(18.0, 900.0, 0.040)            # 900*0.04/(2*18) = 1.000
    z_f, sig_f = fuse_depth(z_s, 0.010, z_m, 0.050)
    z_bad, sig_bad = fuse_depth(float("inf"), 0.0, 1.2, 0.05)

    checks = [
        (f"Savitzky-Golay 降噪（RMSE {rmse_before:.4f} -> {rmse_after:.4f}）",
         rmse_after < rmse_before * 0.8),
        (f"Kalman 外推 0.2 s 误差 {pred_err*1000:.1f} mm", pred_err < 0.05),
        (f"双目深度 Z={z_s:.3f} m / 单目 Z={z_m:.3f} m",
         abs(z_s - 1.0) < 1e-6 and abs(z_m - 1.0) < 1e-6),
        (f"逆方差融合 Z={z_f:.3f}，σ={sig_f*1000:.1f} mm",
         abs(z_f - 1.0) < 0.02 and sig_f < 0.01),
        ("单目无效时退化为双目", abs(z_bad - 1.2) < 1e-9 and abs(sig_bad - 0.05) < 1e-9),
    ]
    ok = True
    for name, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {name}")
        ok = ok and passed
    return ok


def main() -> int:
    print("=" * 62)
    print("双目链路自测（合成真值）")
    print("=" * 62)
    a = test_sync()
    print()
    b = test_triangulate()
    print()
    c = test_trajectory_fusion()
    print()
    print("判定结果：" + ("全部通过" if (a and b and c) else "有不达标项"))
    return 0 if (a and b and c) else 1


if __name__ == "__main__":
    raise SystemExit(main())
