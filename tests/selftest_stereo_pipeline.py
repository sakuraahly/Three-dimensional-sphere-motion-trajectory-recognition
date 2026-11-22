"""双目链路自测：同步 + 极线配对 + 三角测量。

用已知真值的合成双目装置验证 stereo/ 包：
  1. 同步：造一段带闪光的亮度序列，把其中一路人为平移 k 帧，看能不能找回来；
  2. 配对 + 三角测量：把已知三维点投到两路，再让程序反算回去，看误差；
  3. FOV 自动对齐：1.5x 合成视场差、同视场、空白帧、序列化往返。

    python tests/selftest_stereo_pipeline.py
"""

from __future__ import annotations

import json
import sys
import tempfile
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
    SMOOTH_METHODS,
    Kalman3D,
    MatchConfig,
    PredictConfig,
    StereoReconstructor,
    SyncConfig,
    detect_bounce,
    find_offset,
    fit_ballistic,
    fuse_depth,
    jitter_rms,
    match_epipolar,
    mono_depth,
    predict,
    savgol_trajectory,
    smooth_trajectory,
    stereo_depth,
    timestamp_offset,
)
from stereo.fov_align import FovAlignment, estimate_fov_alignment
from stereo.joint_refine import describe_result, refine_pair
from stereo.triangulate import (anchor_accepts, anchor_problem,
                                build_radius_anchor, epipolar_gate_px,
                                rectify_point_left, rectify_point_right)
from stereo.visualization import (
    build_trajectory_payload, export_session_visualization, load_seeds, write_seeds,
)
from sphere_tracker.geometry import refine_circle_subpixel_checked
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
    # ---- 使用点护栏：none_raw（链式/未校正）不能走 parallel 重建 ----
    # 为什么要测：历史标定文件把 chain 记成 stereo_mode="parallel"，而
    # dy ≤ 2.5 px / 视差范围都建立在校正图上 —— 90° 原始像素 |dy| 可达几百 px，
    # 结果是全部候选被拒、配对率 0、**不报错**。护栏必须在构造点就拒绝。
    bad_rig, _ = make_rig()
    bad_rig.meta["rectification"] = "none_raw"
    guard_text, guard_ok = "", False
    try:
        StereoReconstructor(bad_rig)
    except ValueError as exc:
        guard_text = str(exc)
        guard_ok = "none_raw" in guard_text and "angled" in guard_text
    print(f"  none_raw 护栏："
          f"{guard_text.splitlines()[0][:62] if guard_text else '**没有报错**'}")
    checks.append(("none_raw 标定构造 StereoReconstructor -> ValueError（拒绝 parallel 重建）",
                   guard_ok))
    # 护栏不能误伤正常标定：stereo_rectify 的 rig 仍可构造
    checks.append(("stereo_rectify 标定不受护栏影响",
                   StereoReconstructor(rig).n_frames == 0))

    # ---- 自适应极线门限 epipolar_gate_px：纯函数，默认行为不变 ----
    # 本机自报极线 RMS 0.584 px；4×RMS ≈ 2.336，非回退值 2.5（证明真按 RMS 算）。
    gate_0584 = epipolar_gate_px({"epipolar_rms": 0.584})
    checks.append((f"0.584 px RMS -> 门限 {gate_0584:.4f}（≈4×RMS 且 != 2.5，"
                   f"确非偷偷回退）",
                   abs(gate_0584 - 2.336) < 0.01 and gate_0584 != 2.5))
    checks.append(("0.30 px RMS -> 门限 1.2（比 2.5 更紧）",
                   abs(epipolar_gate_px({"epipolar_rms": 0.30}) - 1.2) < 1e-9))
    checks.append(("3.0 px RMS -> 门限被上限夹在 6.0",
                   epipolar_gate_px({"epipolar_rms": 3.0}) == 6.0))
    checks.append(("0.05 px RMS -> 门限被下限夹在 1.0",
                   epipolar_gate_px({"epipolar_rms": 0.05}) == 1.0))
    _fb_cases = [None, {}, {"epipolar_rms": float("nan")},
                 {"epipolar_rms": float("inf")}, {"epipolar_rms": 0.0},
                 {"epipolar_rms": "0.6"}]
    _fb_vals = [epipolar_gate_px(c) for c in _fb_cases]
    checks.append((f"回退分支全覆盖（None/空/NaN/inf/0.0/字符串）-> {_fb_vals}"
                   f"（全部 = 2.5）", all(v == 2.5 for v in _fb_vals)))
    _sweep = [epipolar_gate_px({"epipolar_rms": float(r)})
              for r in np.arange(0.1, 2.0 + 1e-9, 0.05)]
    _mono_ok = all(b >= a for a, b in zip(_sweep, _sweep[1:]))
    _bnd_ok = all(1.0 <= v <= 6.0 for v in _sweep)
    # 两端必须真的碰到夹子（0.1×4=0.4→下限 1.0；2.0×4=8→上限 6.0）：
    # 少了这一条，"k 写错成永远回退"这类改动会给出恒定 2.5，仍满足单调+在界内，
    # 判据就空转了。
    _clip_ok = abs(_sweep[0] - 1.0) < 1e-9 and abs(_sweep[-1] - 6.0) < 1e-9
    checks.append((f"单调性/边界：RMS 0.1→2.0 门限非降={_mono_ok}，"
                   f"全程落在 [{min(_sweep):.1f}, {max(_sweep):.1f}] 内={_bnd_ok}，"
                   f"两端触及夹子={_clip_ok}",
                   _mono_ok and _bnd_ok and _clip_ok))
    # 两路等价：自适应路径与显式同值路径必须给出完全相同的匹配
    rig.quality = {"epipolar_rms": 0.584}
    auto_px = epipolar_gate_px(rig.quality)
    explicit_px = 4.0 * 0.584                      # 人手写死的同一个数
    left_d = Detection(x=float(uv1[0, 0]), y=float(uv1[0, 1]), radius=20.0, score=1.0)

    def _cands():
        return [Detection(x=float(uv2[0, 0]), y=float(uv2[0, 1]), radius=20.0, score=0.9),
                Detection(x=float(uv2[0, 0]) + 30.0, y=float(uv2[0, 1]) + 30.0,
                          radius=20.0, score=0.6)]

    b_auto, dy_auto = match_epipolar(rig, left_d, _cands(),
                                     MatchConfig(max_epipolar_px=auto_px))
    b_exp, dy_exp = match_epipolar(rig, left_d, _cands(),
                                   MatchConfig(max_epipolar_px=explicit_px))
    _same = (abs(auto_px - explicit_px) < 1e-9 and dy_auto == dy_exp
             and b_auto is not None and b_exp is not None
             and b_auto.x == b_exp.x and b_auto.y == b_exp.y)
    checks.append((f"自适应 {auto_px:.4f} 与显式 {explicit_px:.4f} 同值 -> 匹配一致={_same}",
                   _same))

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

    # ---- M4：抖动指标 + 平滑分派（savgol / kalman）----
    # 抖动 = 二阶中心差分 RMS；这里是"同一序列平滑前后"的相对量（真值线性，故只有噪声）。
    jit_before = jitter_rms(noisy)
    jit_after = jitter_rms(smooth)
    jit_red = (1.0 - jit_after / jit_before) * 100.0 if jit_before > 0 else 0.0
    sm_kal, sm_kal_params = smooth_trajectory(t, noisy, method="kalman")
    sm_sg, sm_sg_params = smooth_trajectory(t, noisy, method="savgol")
    bad_method = False
    try:
        smooth_trajectory(t, noisy, method="nope")
    except ValueError:
        bad_method = True

    z_s = stereo_depth(90.0, 900.0, 0.100)          # 900*0.1/90 = 1.000
    z_m = mono_depth(18.0, 900.0, 0.040)            # 900*0.04/(2*18) = 1.000
    z_f, sig_f = fuse_depth(z_s, 0.010, z_m, 0.050)
    z_bad, sig_bad = fuse_depth(float("inf"), 0.0, 1.2, 0.05)

    checks = [
        (f"Savitzky-Golay 降噪（RMSE {rmse_before:.4f} -> {rmse_after:.4f}）",
         rmse_after < rmse_before * 0.8),
        (f"抖动 RMS（二阶差分）{jit_before * 1e3:.2f} -> {jit_after * 1e3:.2f} mm，"
         f"下降 {jit_red:.1f}%（M4 验收 ≥50%）",
         jit_before > 0.0 and jit_red >= 50.0),
        (f"smooth_trajectory 分派：savgol {sm_sg_params} / kalman {sm_kal_params}；"
         f"形状 {sm_sg.shape}/{sm_kal.shape}；未知方法报错={bad_method}",
         sm_sg.shape == noisy.shape and sm_kal.shape == noisy.shape
         and bool(np.all(np.isfinite(sm_sg))) and bool(np.all(np.isfinite(sm_kal)))
         and bad_method and set(SMOOTH_METHODS) == {"savgol", "kalman"}),
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


def test_predict3d() -> bool:
    """第 7 节：三维弹道预测（重力是已知常向量）+ 反弹重初始化。"""
    print("----- 三维弹道预测（匀速 + 已知重力） -----")
    cfg = PredictConfig()
    g = np.asarray(cfg.g, dtype=np.float64)
    # 默认重力必须跟本项目的坐标约定一致（Y 向下 = +9.81）；写回 -9.81 要红。
    default_g_ok = tuple(cfg.g) == (0.0, 9.81, 0.0)
    p_origin = np.array([0.0, 1.0, 0.0])
    v_origin = np.array([2.0, 3.0, 0.0])
    t = np.arange(0.0, 0.5 + 1e-9, 1.0 / 30.0)
    pts = p_origin + v_origin * t[:, None] + 0.5 * g * (t ** 2)[:, None]

    # ---- 1. 解析解：无噪点必须被精确恢复 ----
    # 注意 state 的时基是 t_ref = 最后一个样本（"现在"），所以 v0 对应 v(t_ref)，
    # 不是 t=0 的初速度 —— 这正是预测该用的量。
    st = fit_ballistic(t, pts, cfg)
    t_ref = float(t[-1])
    p_ref = p_origin + v_origin * t_ref + 0.5 * g * t_ref ** 2
    v_ref = v_origin + g * t_ref
    err_p = float(np.max(np.abs(st.p0 - p_ref))) if st is not None else 1e9
    err_v = float(np.max(np.abs(st.v0 - v_ref))) if st is not None else 1e9
    pred = predict(st, cfg.horizons) if st is not None else {}
    truth_03 = p_origin + v_origin * (t_ref + 0.3) + 0.5 * g * (t_ref + 0.3) ** 2
    err_03 = float(np.linalg.norm(pred[0.3] - truth_03)) if 0.3 in pred else 1e9

    # 重复时间戳：去重 + 记账，不能把 p0/v0 拉偏
    st_dup = fit_ballistic(np.concatenate([t, t[-1:]]),
                           np.concatenate([pts, pts[-1:]]), cfg)

    # ---- 2. 噪声鲁棒：1 mm 高斯噪声 ----
    rng = np.random.default_rng(5)
    noisy = pts + rng.normal(0.0, 1e-3, pts.shape)
    st_n = fit_ballistic(t, noisy, cfg)
    err_03_n = (float(np.linalg.norm(predict(st_n, [0.3])[0.3] - truth_03))
                if st_n is not None else 1e9)

    # ---- 3. 样本不足：n=3 必须拒绝并说明原因 ----
    why_short: list[str] = []
    st_short = fit_ballistic(t[:3], pts[:3], cfg, why_short)

    # ---- 4. 机动检测：0.4 s 落地反弹（速度沿重力方向由同向变反向）----
    # 反弹后的速度方向必须跟着 g 的符号走：默认 Y 向下（g[1]>0）时下落速度为正、
    # 反弹后向上（负）；Y 向上的数学系则相反。写死一个符号会让合成数据与
    # `detect_bounce` 的判据互相矛盾。
    tb = 0.4
    tt = np.arange(0.0, 0.8 + 1e-9, 1.0 / 30.0)
    y_b = 1.0 - 2.0 * tb + 0.5 * g[1] * tb ** 2
    v_bounce = -1.5 if g[1] > 0 else 1.5
    y = np.where(tt <= tb, 1.0 - 2.0 * tt + 0.5 * g[1] * tt ** 2,
                 y_b + v_bounce * (tt - tb) + 0.5 * g[1] * (tt - tb) ** 2)
    bounce_pts = np.column_stack([0.5 * tt, y, 1.0 + 0.1 * tt])
    truth_idx = int(np.argmax(tt > tb))
    why_b: list[str] = []
    idx = detect_bounce(tt, bounce_pts, cfg, why_b)
    idx_err = abs(idx - truth_idx) if idx is not None else 10 ** 6

    # ---- 4b. 负 g（Y 向上的数学系）也要能检出反弹：判据是 v·g，不是写死符号 ----
    cfg_up = PredictConfig(g=(0.0, -9.81, 0.0))
    g_up = np.asarray(cfg_up.g, dtype=np.float64)
    y_bu = 1.0 - 2.0 * tb + 0.5 * g_up[1] * tb ** 2
    v_bu = -1.5 if g_up[1] > 0 else 1.5
    y_up = np.where(tt <= tb, 1.0 - 2.0 * tt + 0.5 * g_up[1] * tt ** 2,
                    y_bu + v_bu * (tt - tb) + 0.5 * g_up[1] * (tt - tb) ** 2)
    bounce_up = np.column_stack([0.5 * tt, y_up, 1.0 + 0.1 * tt])
    idx_up = detect_bounce(tt, bounce_up, cfg_up, [])

    # 关键证伪：纯噪声（同一条弹道 + 2 mm 噪声，无反弹）必须返回 None
    tn = np.arange(0.0, 0.5 + 1e-9, 1.0 / 30.0)
    pn = (p_origin + v_origin * tn[:, None] + 0.5 * g * (tn ** 2)[:, None]
          + rng.normal(0.0, 2e-3, (tn.size, 3)))
    why_n: list[str] = []
    idx_noise = detect_bounce(tn, pn, cfg, why_n)

    # ---- 5. 平滑不得移动反弹拐点（M4 验收：≤2 帧）----
    # 放在 pn 之后算，避免多吃 rng 状态影响上面"纯噪声"那条判据。
    bounce_noisy = bounce_pts + rng.normal(0.0, 3e-3, bounce_pts.shape)
    bounce_sg = savgol_trajectory(tt, bounce_noisy, window=11, polyorder=2)
    why_s: list[str] = []
    idx_s = detect_bounce(tt, bounce_sg, cfg, why_s)
    idx_s_err = abs(idx_s - truth_idx) if idx_s is not None else 10 ** 6

    checks = [
        (f"解析解：n={getattr(st, 'n', 0)}，v0 误差 {err_v:.1e} m/s，"
         f"p(t_ref) 误差 {err_p:.1e} m，rms={getattr(st, 'rms_m', float('nan')):.1e}",
         st is not None and err_v < 1e-9 and err_p < 1e-9 and st.rms_m < 1e-9),
        (f"predict 四档 {0.1}/{0.2}/{0.3}/{0.5} s 键齐全，0.3 s 误差 {err_03:.1e} m",
         set(pred) == {0.1, 0.2, 0.3, 0.5} and err_03 < 1e-9
         and predict(None, cfg.horizons) == {}),
        (f"重复时间戳去重：dropped={getattr(st_dup, 'dropped', -1)}，n={getattr(st_dup, 'n', 0)}",
         st_dup is not None and st_dup.dropped == 1 and st_dup.n == t.size
         and float(np.max(np.abs(st_dup.v0 - v_ref))) < 1e-9),
        (f"1 mm 噪声：rms={st_n.rms_m * 1e3 if st_n else float('nan'):.2f} mm"
         f"（最小二乘吸收约 0.8x，故略低于 1 mm），0.3 s 误差 {err_03_n * 1e3:.2f} mm",
         st_n is not None and 0.5e-3 <= st_n.rms_m <= 3e-3 and err_03_n < 0.010),
        (f"n=3 -> None，原因「{why_short[0] if why_short else '无'}」",
         st_short is None and bool(why_short) and "样本" in why_short[0]),
        (f"0.4 s 反弹：检出索引 {idx}（真值 {truth_idx}，{len(tt)} 点，"
         f"±2 帧内）；{why_b[0][:46] if why_b else '无'}",
         idx is not None and idx_err <= 2),
        (f"平滑后反弹拐点：S-G 检出 {idx_s}（真值 {truth_idx}，误差 {idx_s_err} 帧，"
         f"≤2 帧）；{why_s[0][:40] if why_s else '无'}",
         idx_s is not None and idx_s_err <= 2),
        (f"纯噪声（无反弹）-> {idx_noise}（必须为 None）；"
         f"{why_n[0][:34] if why_n else '无'}",
         idx_noise is None),
        (f"默认 g 是 Y 向下（+9.81），不是 Y 向上", default_g_ok),
        (f"负 g（Y 向上）也能检出反弹：{idx_up}（真值 {truth_idx}）",
         idx_up is not None and abs(idx_up - truth_idx) <= 2),
    ]
    ok = True
    for name, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {name}")
        ok = ok and passed
    return ok


def test_angled() -> bool:
    """大角度模式：F 匹配 + 原始像素射线交会，与平行模式隔离。"""
    print("----- 大角度（近似直角）合成真值 -----")
    from types import SimpleNamespace
    from sphere_tracker.types import Detection
    from stereo.angled import AngledMatchConfig, AngledReconstructor
    from stereo.stereo_mode import geometry_advice, infer_mode, normalize_mode
    import cv2
    import numpy as np

    K = np.array([[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1.0]])
    D = np.zeros(5)
    R = cv2.Rodrigues(np.array([0.0, np.pi / 2, 0.0]))[0]
    # 右相机在左相机 +X 0.4 m：X_right = R X_left + T
    T = -R @ np.array([0.4, 0.0, 0.0])
    rig = SimpleNamespace(K1=K, D1=D, K2=K, D2=D, R=R, T=T, F=None,
                          stereo_mode="angled")
    pts = np.array([[0.10, 0.02, 1.00],
                    [0.20, -0.05, 1.10],
                    [0.15, 0.08, 0.95]], np.float64)
    rvec = cv2.Rodrigues(R)[0]
    uv_l, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), K, D)
    uv_r, _ = cv2.projectPoints(pts, rvec, T, K, D)
    rec = AngledReconstructor(rig, AngledMatchConfig(max_epipolar_px=3.0))
    errors = []
    for i, p in enumerate(pts):
        dl = Detection(x=float(uv_l[i, 0, 0]), y=float(uv_l[i, 0, 1]), radius=20)
        dr = Detection(x=float(uv_r[i, 0, 0]), y=float(uv_r[i, 0, 1]), radius=20)
        out = rec.update(float(i), dl, [dr])
        if out is not None:
            errors.append(float(np.linalg.norm(out.xyz - p)))
    checks = [
        ("F 匹配 + 射线交会，3/3 点误差 <1e-5 m",
         len(errors) == len(pts) and max(errors) < 1e-5),
        ("模式别名/旧数据默认 parallel",
         normalize_mode("orthogonal") == "angled"
         and infer_mode({}) == "parallel"),
        ("旧会话无字段 + angled 标定 -> auto=angled",
         infer_mode(session={}, calibration={"meta": {"stereo_mode": "angled"}})
         == "angled"),
        ("显式 session=parallel 不被标定覆盖",
         infer_mode(session={"stereo_mode": "parallel"},
                    calibration={"meta": {"stereo_mode": "angled"}})
         == "parallel"),
        ("B=600mm/D=0.3m 估算约 90°",
         abs(geometry_advice(600.0, 0.30)["angle_deg"] - 90.0) < 1e-6),
        ("大角度统计计数器",
         rec.n_matched == len(pts) and rec.match_rate > 0.99),
    ]
    from calib.angled_quality import evaluate_angled, grade_angled
    q = evaluate_angled(K, D, K, D, R, T,
                        uv_l.reshape(-1, 2), uv_r.reshape(-1, 2))
    grade, verdict = grade_angled(q, min_pairs=3, min_angle=5)
    checks.append(("F 线距离/交会角质量评估",
                   q.line_rms_px < 1e-5 and "极线" in verdict
                   and q.n_pairs == len(pts)))
    # run_stereo_video 的旧数据兼容：parallel + none_raw 必须改判 angled，
    # 否则 90° 标定会走进 dy 门限为 0 配对的 parallel 路径。纯函数，可直接断言。
    from apps.run_stereo_video import _mode_guard
    checks.append(("run_video：parallel + none_raw -> 自动改判 angled",
                   _mode_guard("parallel", "none_raw") == ("angled", True)))
    checks.append(("run_video：已校正的 parallel / 显式 angled 不被改判",
                   _mode_guard("parallel", "stereo_rectify") == ("parallel", False)
                   and _mode_guard("angled", "none_raw") == ("angled", False)))
    ok = True
    for name, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {name}")
        ok = ok and passed
    return ok


def test_visualization() -> bool:
    """标准数据：seeds.json / traj3d.json / visualization.json 的契约。"""
    print("----- 标准数据导出 -----")
    ok = True
    with tempfile.TemporaryDirectory(prefix="dsh_stereo_vis_") as tmp:
        d = Path(tmp)
        (d / "session.json").write_text(json.dumps({
            "schema": "sphere-stereo-session", "version": 1, "id": "synthetic",
            "left": {"width": 1920, "height": 1080},
            "sync": {"method": "flash", "offset": 0, "confidence": 0.9},
            "stereo_mode": "angled",
            "rig": {"stereo_mode": "angled", "angle_deg": 90.0},
            "calibration": {"baseline_mm": 65.0, "created": "x", "verdict": "ok"},
        }), encoding="utf-8")
        (d / "derived").mkdir()
        (d / "derived" / "traj3d.csv").write_text(
            "frame,t,matched,X,Y,Z,u_left,v_left,u_right,v_right,epipolar_px,residual_px,confident,method_left,method_right\n"
            "0,0.0,1,0.1,0.05,0.3,100,200,90,200,1.0,0.5,1,track,track\n"
            "1,0.03,0,0,,,,,,,,,0,track,track\n", encoding="utf-8")
        s = write_seeds(d, (100, 200, 30), (90, 200, 30), seed_frame=0)
        loaded = load_seeds(s)
        paths = export_session_visualization(d)
        traj = json.loads(paths["trajectory"].read_text(encoding="utf-8"))
        vis = json.loads(paths["visualization"].read_text(encoding="utf-8"))
        checks = [
            ("seeds schema/坐标", loaded["schema"] == "sphere-stereo-seeds"
             and loaded["left"]["x"] == 100),
            ("traj3d.json schema/单位", traj["schema"] == "sphere-stereo-trajectory"
             and traj["unit"] == "m" and traj["n_matched"] == 1),
            ("短行（未匹配）不破坏解析", len(traj["samples"]) == 2
             and traj["samples"][1]["matched"] is False),
            ("visualization.json 点数/bounds",
             vis["schema"] == "sphere-stereo-visualization"
             and vis["n_points"] == 1 and vis["bounds"]["max"][2] == 0.3),
            ("大角度轨迹坐标/模式透传",
             traj["stereo_mode"] == "angled"
             and traj["coordinate_system"] == "left_camera"
             and vis["stereo_mode"] == "angled"),
            ("显式 parallel 覆盖不会沿用旧 session 的 angled",
             build_trajectory_payload(d, stereo_mode="parallel")["coordinate_system"]
             == "rectified_left_camera"),
        ]
        for name, passed in checks:
            print(f"  [{'通过' if passed else '不达标'}] {name}")
            ok = ok and passed
    return ok


def _fov_texture(seed: int = 42):
    """合成随机纹理：低频色块 + 几何/文字标记，保证 SIFT 有足够稳定特征。"""
    rng = np.random.default_rng(seed)
    h, w = 1350, 2400
    low = rng.normal(0.0, 1.0, (h // 16 + 2, w // 16 + 2, 3)).astype(np.float32)
    base = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)
    base = cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    for _ in range(900):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        radius = int(rng.integers(4, 50))
        color = tuple(int(v) for v in rng.integers(0, 256, 3))
        kind = int(rng.integers(0, 4))
        if kind == 0:
            cv2.circle(base, (x, y), radius, color, -1)
        elif kind == 1:
            cv2.rectangle(base, (x, y), (x + radius, y + radius), color, 2)
        elif kind == 2:
            cv2.line(base, (x, y),
                     (x + int(rng.integers(-80, 80)), y + int(rng.integers(-80, 80))),
                     color, int(rng.integers(1, 4)))
        else:
            cv2.putText(base, str(int(rng.integers(0, 1000))), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, float(rng.uniform(0.4, 1.1)),
                        color, 2, cv2.LINE_AA)
    return base


def _fov_pair(size=(2560, 1440)):
    """左=纹理中央 1600x900（放大 1.5x），右=整幅 2400x1350。"""
    master = _fov_texture()
    width, height = size
    left = cv2.resize(master[225:1125, 400:2000], (width, height),
                      interpolation=cv2.INTER_LINEAR)
    right = cv2.resize(master, (width, height), interpolation=cv2.INTER_LINEAR)
    return left, right


def test_fov_alignment() -> bool:
    """FOV 自动对齐：已知 1.5x 合成视场差、共同不放大目标、同视场、空白帧、序列化。"""
    print("----- 双目 FOV 自动对齐 -----")
    cv2.setRNGSeed(20261024)
    left, right = _fov_pair()
    checks = []
    align = None

    try:
        align = estimate_fov_alignment(left, right)
        aligned_left, aligned_right = align.apply(left, right)
        x, y, w, h = align.crop_rect
        # 共同输出尺寸 = 参考帧与裁剪区逐轴取小、再向下取偶（无损 MKV 要偶数）。
        want_w, want_h = min(2560, w), min(1440, h)
        want_target = (want_w - want_w % 2, want_h - want_h % 2)
        checks.extend([
            (f"1.5x 视场差 -> 参考 {align.reference_side} / 裁剪 {align.crop_side}，"
             f"尺度 {align.scale:.3f}",
             align.active and align.reference_side == "left"
             and align.crop_side == "right"
             and abs(align.scale - 1.0 / 1.5) <= 0.05),
            (f"裁剪区 {align.crop_rect}（内点 {align.inliers}、"
             f"质量 {align.quality:.2f}、方法 {align.method}）",
             align.method == "sift" and align.inliers >= 30
             and 0.0 <= align.quality <= 1.0
             and 0 <= x and 0 <= y and x + w <= 2560 and y + h <= 1440
             and abs(x - 426) <= 25 and abs(y - 240) <= 25
             and abs(w - 1707) <= 25 and abs(h - 960) <= 25),
            (f"target_size {align.target_size} 取共同（不放大）尺寸并取偶，"
             f"不是参考帧 2560x1440",
             align.target_size == want_target
             and align.target_size[0] % 2 == 0 and align.target_size[1] % 2 == 0
             and align.target_size[0] <= w and align.target_size[1] <= h
             and align.target_size[0] <= 2560 and align.target_size[1] <= 1440
             and align.target_size != (2560, 1440)),
            ("两路都缩到共同 target_size；参考路下采样、广角裁剪区不放大",
             aligned_left.shape == aligned_right.shape
             == (want_target[1], want_target[0], 3)
             and not np.shares_memory(aligned_left, left)
             and not np.shares_memory(aligned_right, right)),
        ])
        ref_only = align.apply_side(left, "left")
        crop_only = align.apply_side(right, "right")
        checks.append((
            "apply_side：参考路与裁剪路都到 target_size，单路结果与成对 apply 一致",
            ref_only.shape == crop_only.shape == aligned_left.shape
            and np.array_equal(ref_only, aligned_left)
            and np.array_equal(crop_only, aligned_right),
        ))
        post = estimate_fov_alignment(aligned_left, aligned_right)
        checks.append((f"对齐后残余尺度 {post.scale:.4f}（目标 1±0.1），target 为原生尺寸",
                       abs(post.scale - 1.0) <= 0.10
                       and post.target_size == (want_target[0], want_target[1])))
    except ValueError as exc:
        checks.append((f"1.5x 合成帧对估计失败：{exc}", False))

    # 根因守卫：target_size 大于裁剪区 / 参考帧时必须拒绝，绝不静默放大
    # （旧实现把 1685x979 放大到 2560x1440，Laplacian 清晰度塌到 ~20）。
    try:
        bad = FovAlignment(reference_side="left", crop_side="right",
                           crop_rect=(0, 0, 160, 90), target_size=(320, 180),
                           scale=0.5, method="sift", inliers=40, quality=0.9)
        rejected = []
        for side in ("right", "left"):
            try:
                bad.apply_side(np.zeros((90, 160, 3), np.uint8), side)
            except ValueError:
                rejected.append(True)
            else:
                rejected.append(False)
        checks.append(("target_size 大于裁剪区 / 参考帧 -> 拒绝放大",
                       rejected == [True, True]))
    except Exception as exc:                                # noqa: BLE001
        checks.append((f"放大守卫异常：{exc}", False))

    try:
        mirror = estimate_fov_alignment(right, left)
        mw, mh = mirror.crop_rect[2], mirror.crop_rect[3]
        checks.append((f"交换左右（右更窄）-> 参考 {mirror.reference_side} / "
                       f"裁剪 {mirror.crop_side}，尺度 {mirror.scale:.3f}",
                       mirror.reference_side == "right"
                       and mirror.crop_side == "left"
                       and abs(mirror.scale - 1.5) <= 0.075
                       and mirror.target_size[0] <= min(2560, mw)
                       and mirror.target_size[1] <= min(1440, mh)
                       and mirror.target_size[0] % 2 == 0
                       and mirror.target_size[1] % 2 == 0))
    except ValueError as exc:
        checks.append((f"右窄左宽估计失败：{exc}", False))

    try:
        same = estimate_fov_alignment(left, left.copy())
        same_l, same_r = same.apply(left, left.copy())
        checks.append(("相同画面 -> crop_side='none'，target=参考原生尺寸，不缩放",
                       same.crop_side == "none" and not same.active
                       and same.target_size == (2560, 1440)
                       and np.array_equal(same_l, left)
                       and np.array_equal(same_r, left)))
    except ValueError as exc:
        checks.append((f"相同画面被误判为需要对齐：{exc}", False))

    blank = np.full((480, 640, 3), 127, np.uint8)
    try:
        estimate_fov_alignment(blank, blank.copy())
    except ValueError as exc:
        checks.append((f"空白无特征 -> ValueError（{str(exc)[:32]}…）", True))
    else:
        checks.append(("空白无特征必须报错，不能静默返回对齐", False))

    # 参考帧四角越界的判定：少量越界（<=16 px）容忍轻微旋转/取整；
    # 大幅越界即共同视场不足。
    try:
        from stereo.fov_align import _mapped_rect

        corners = np.float32(
            [[0, 0], [100, 0], [100, 100], [0, 100]]).reshape(-1, 1, 2)
        h_tol = np.array([[1.0, 0.0, -0.5], [0.0, 1.0, -0.5], [0.0, 0.0, 1.0]])
        rect = _mapped_rect(corners, h_tol, 100, 100)
        h_bad = np.array([[1.0, 0.0, -20.5], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        try:
            _mapped_rect(corners, h_bad, 100, 100)
        except ValueError:
            bad_raised = True
        else:
            bad_raised = False
        checks.append(("四角少量越界容忍；>16 px 报共同视场不足",
                       rect == (0, 0, 100, 100) and bad_raised))
    except Exception as exc:                                # noqa: BLE001
        checks.append((f"_mapped_rect 越界判定异常：{exc}", False))

    # 参考帧没有完整落在广角帧内（共同视场不足）：不能把越界角裁掉、
    # 拿“部分重叠”继续做主动对齐。
    try:
        partial_right = cv2.resize(right[0:1080, 0:1920], (2560, 1440),
                                   interpolation=cv2.INTER_LINEAR)
        estimate_fov_alignment(left, partial_right)
    except ValueError as exc:
        text = str(exc)
        checks.append((f"参考帧未完整落在广角帧内 -> ValueError（{text[:32]}…）",
                       "共同视场不足" in text and "完整" in text))
    else:
        checks.append(("参考帧未完整落在广角帧内必须报错，不能被静默裁剪", False))

    # 损坏的 fov 记录必须被当成“损坏”，而不是被“空值 = 旧标定”静默跳过。
    try:
        from apps.record_stereo import _load_calib_fov

        bad_rejected = True
        for bad in ({}, [], "fov", 3):
            try:
                _load_calib_fov(bad)
            except ValueError:
                continue
            bad_rejected = False
        valid_ok = align is None or _load_calib_fov(align.to_dict()) == align
        checks.append(("record：fov 缺失/None=旧标定，{} / [] / 非 mapping=损坏",
                       _load_calib_fov(None) is None and bad_rejected and valid_ok))
    except Exception as exc:                                # noqa: BLE001
        checks.append((f"record 的 fov 读取接线异常：{exc}", False))

    # verify：视频/图片首对帧的一次性“是否已归一化”判定（用桩替代 SIFT）。
    try:
        import io
        from contextlib import redirect_stdout
        from types import SimpleNamespace

        import apps.verify_stereo as verify_app

        def _probe(active, scale):
            return SimpleNamespace(active=active, scale=scale,
                                   crop_side="right" if active else "none",
                                   crop_rect=(10, 20, 30, 40))

        def _raise(*_a, **_k):
            raise ValueError("特征不足")

        probe_cases = [
            ("原生画面（scale 0.667）-> 退出 2",
             lambda *_a, **_k: _probe(True, 0.667), 2, False),
            ("已归一化（无需对齐）-> 放行",
             lambda *_a, **_k: _probe(False, 1.0), 0, False),
            ("低纹理估不出来 -> 退出 2（不静默假定）", _raise, 2, False),
            ("--fov-input-normalized -> 跳过校验", _raise, 0, True),
            ("仍需要对齐但尺度在 0.9–1.1 -> 退出 2（fail closed）",
             lambda *_a, **_k: _probe(True, 0.98), 2, False),
        ]
        original = verify_app.estimate_fov_alignment
        verdicts = []
        try:
            with redirect_stdout(io.StringIO()):
                for name, stub, want, assume in probe_cases:
                    verify_app.estimate_fov_alignment = stub
                    got = verify_app._check_normalized_input(
                        blank, blank.copy(), assume_normalized=assume)
                    verdicts.append((name, got == want))
        finally:
            verify_app.estimate_fov_alignment = original
        checks.extend(verdicts)
    except Exception as exc:                                # noqa: BLE001
        checks.append((f"verify 的输入归一化校验接线异常：{exc}", False))

    roundtrip_ok = False
    if align is not None:
        try:
            payload = json.loads(json.dumps(align.to_dict()))
            roundtrip_ok = FovAlignment.from_dict(payload) == align
        except (TypeError, ValueError) as exc:
            print(f"  序列化往返异常：{exc}")
    checks.append(("to_dict/from_dict 往返一致", roundtrip_ok))

    ok = True
    for name, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {name}")
        ok = ok and passed
    return ok


def test_pipeline_blocks() -> bool:
    """主流程（run_stereo_video）的两段纯函数：世界系块 / 预测块。

    不启动相机、不读真实会话 —— 只喂数组，验证"该给的给、该拒的拒"。
    """
    print("----- 主流程纯函数（世界系 / 预测块） -----")
    from calib.world_frame import WorldFrame
    from apps.run_stereo_video import _predict_block, _world_block

    # ---- 1. 没有世界系：坐标系由调用处给定，必须如实写，不改写 ----
    arr = np.array([[0.0, 0.1, 0.2, 0.3],
                    [0.1, 0.2, 0.3, 0.4]], np.float64)
    w_par, j_par = _world_block(None, arr, "rectified_left_camera")
    w_ang, j_ang = _world_block(None, arr, "left_camera")
    # ---- 2. 已知世界系（平移 +10 mm）：点整体平移，时间列不受影响 ----
    T = np.eye(4)
    T[:3, 3] = [0.010, 0.010, 0.010]
    wf = WorldFrame(T_world_cam=T, source="selftest", note="平移 10 mm")
    w_out, j_out = _world_block(wf, arr, "left_camera")
    w_out_p, j_out_p = _world_block(wf, arr, "rectified_left_camera")
    shift_err = (float(np.max(np.abs(w_out[:, 1:4] - arr[:, 1:4] - 0.010)))
                 if w_out is not None else float("inf"))
    # ---- 3. 解析抛物线（无噪）：0.3 s 预测必须精确复现 ----
    cfg = PredictConfig()
    g = np.asarray(cfg.g, np.float64)
    p0 = np.array([0.05, 1.20, 0.80])
    v0 = np.array([1.5, 0.8, -0.4])
    t = np.arange(0.0, 0.5 + 1e-9, 1.0 / 30.0)
    pts = p0 + v0 * t[:, None] + 0.5 * g * (t ** 2)[:, None]
    rows, block = _predict_block(np.column_stack([t, pts]), cfg)
    t_ref = float(t[-1])
    truth_03 = p0 + v0 * (t_ref + 0.3) + 0.5 * g * (t_ref + 0.3) ** 2
    row_03 = next((r for r in rows if abs(r["horizon_s"] - 0.3) < 1e-12), None)
    err_03 = (float(np.max(np.abs(np.array([row_03["x"], row_03["y"], row_03["z"]])
                                 - truth_03)))
              if row_03 is not None else 1e9)
    # ---- 4. 样本不足（3 点）：必须拒绝并给出"样本"原因 ----
    rows_short, block_short = _predict_block(np.column_stack([t[:3], pts[:3]]), cfg)

    checks = [
        ("parallel 无世界系 -> rectified_left_camera / world_frame=None",
         w_par is None and j_par["coordinate_system"] == "rectified_left_camera"
         and j_par["world_frame"] is None),
        ("angled 无世界系 -> left_camera / world_frame=None",
         w_ang is None and j_ang["coordinate_system"] == "left_camera"
         and j_ang["world_frame"] is None),
        (f"世界系平移 10 mm -> 最大偏差 {shift_err:.1e} m",
         w_out is not None and j_out["coordinate_system"] == "world"
         and shift_err < 1e-12
         and float(np.max(np.abs(w_out[:, 0] - arr[:, 0]))) == 0.0),
        ("有世界系：两种基础系都记 world（不再暴露 base_frame）",
         j_out_p["coordinate_system"] == "world"
         and w_out_p is not None
         and float(np.max(np.abs(w_out_p - w_out))) == 0.0),
        (f"抛物线 0.3 s 预测误差 {err_03:.1e} m（{len(rows)} 档）",
         len(rows) == len(cfg.horizons) and err_03 < 1e-9
         and block["state"]["n"] == t.size),
        (f"3 点 -> rows 空，原因「{block_short.get('skipped', '无')}」",
         rows_short == [] and "样本" in str(block_short.get("skipped", ""))),
    ]
    ok = True
    for name, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {name}")
        ok = ok and passed
    return ok


def test_radius_anchor() -> bool:
    """人工锚点：一次左右点选 → 期望半径比 → 用它拒掉假目标。

    必须证明两件事，缺一不算：
      ① **数学对**（Z、两路各自的期望半径、比值 = f_L/f_R）；
      ② **这道门真的改变了配对结果** —— 所以故意让"错的那个候选"极线偏差更小：
         不开门它会被选中，开门它被拒。只断言"过滤掉了一个坏候选"是不够的，
         因为那可能只是一次本来就不会发生的选择。
    """
    print("----- 人工锚点（左右半径比）-----")
    size = (1920, 1080)
    # 两路焦距差 40%：镜面球那种"左右一样粗"的假设不成立的极端情形。
    K1 = _K(1000.0, 1000.0, 960.0, 540.0)
    K2 = _K(1400.0, 1400.0, 960.0, 540.0)
    D1, D2 = np.zeros(5), np.zeros(5)
    R = np.eye(3)
    T = np.array([[-0.1], [0.0], [0.0]], np.float64)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.0)
    rig = StereoRig(K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T, image_size=size,
                    R1=R1, R2=R2, P1=P1, P2=P2, Q=Q)

    # 真值：球在 Z=1.0 m、直径 40 mm。T=(-0.1,0,0) 的约定下，右光心在 x 的**负**方向
    # （OpenCV：T 是"左→右"的平移），所以右图横坐标 = 960 − f2·0.1/1.0 = 820。
    # 这个符号写反会让三角化给出 Z=−1（我第一版就写反了，测试当场抓住）。
    dia = 0.04
    anchor = build_radius_anchor(rig, (960.0, 540.0), (820.0, 540.0), dia)
    r_l_expect, r_r_expect = 1000.0 * dia / 2.0, 1400.0 * dia / 2.0   # 20 / 28 px

    # 候选：对的（半径 28、y 偏 2px）与错的（半径 40、y 恰好对上）。
    det_l = Detection(x=960.0, y=540.0, radius=r_l_expect)
    good = Detection(x=820.0, y=543.0, radius=r_r_expect)
    bad = Detection(x=820.0, y=540.0, radius=r_r_expect * 1.6)

    cfg_off = MatchConfig(max_epipolar_px=4.0)
    cfg_on = MatchConfig(max_epipolar_px=4.0,
                         radius_ratio_expected=anchor.radius_ratio,
                         radius_ratio_band=anchor.band)
    pick_off, _ = match_epipolar(rig, det_l, [good, bad], cfg_off)
    pick_on, _ = match_epipolar(rig, det_l, [good, bad], cfg_on)

    ok_pair = anchor_accepts(anchor, r_l_expect, r_r_expect)[0]
    bad_pair = anchor_accepts(anchor, r_l_expect, r_r_expect * 1.6)[0]
    no_r = anchor_accepts(anchor, r_l_expect, 0.0)[0]
    # 锚点自身的几何判据：y 差超限必须给理由（用 dataclasses.replace 造一个坏锚点）
    import dataclasses
    # 12px：远超"人工点高光斑"的 6px 系统偏移，说明这次点选真的对不上
    bad_anchor = dataclasses.replace(anchor, dy_px=12.0)
    bad_why = anchor_problem(bad_anchor)

    checks = [
        (f"锚点三角化 Z={anchor.z_m:.4f} m（真值 1.0）", abs(anchor.z_m - 1.0) < 1e-6),
        (f"左右期望半径 {anchor.r_left_px:.2f} / {anchor.r_right_px:.2f} px"
         f"（真值 {r_l_expect:.0f} / {r_r_expect:.0f}）",
         abs(anchor.r_left_px - r_l_expect) < 0.05
         and abs(anchor.r_right_px - r_r_expect) < 0.05),
        (f"期望比值 {anchor.radius_ratio:.3f} = f_L/f_R（不是 1）",
         abs(anchor.radius_ratio - 1000.0 / 1400.0) < 1e-6),
        ("不开门：被『半径离谱但极线更准』的候选骗走（这就是要修的那个错）",
         pick_off is bad),
        ("开门：选中半径比正确的那个", pick_on is good),
        ("半径比符合 → 接受；离谱 → 拒绝；缺半径 → 拒绝（不静默放行）",
         ok_pair and not bad_pair and not no_r),
        ("锚点自身 y 差超限 → 给得出原因（不是只返回 False）",
         bool(bad_why) and "y 差" in str(bad_why)),
        # 真实会话（20261101-205709）人点出来是 6.4px：那是高光斑偏移，不是几何坏。
        # 门限若沿用检测器口径的 3px，会把合法点选判成"标定坏了"。
        ("6.4px（人工点高光斑）→ 放行；12px（真的对不上）→ 拒绝",
         anchor_problem(dataclasses.replace(anchor, dy_px=6.43)) is None
         and anchor_problem(dataclasses.replace(anchor, dy_px=12.0)) is not None),
    ]
    ok = True
    for name, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {name}")
        ok = ok and passed
    return ok


def test_warmup_no_false_alarm() -> bool:
    """种子落在第 N 帧时，前 N 帧的 0 检出是**设计如此**，不能判成检测器坏了。

    真实踩到过：用户在 GUI 第 80 帧点选球心，跑离线三维时 `seed_frame=80` 被正确读入，
    但零配对门禁在第 20 帧就抛错，界面看起来像"点选没用"。
    """
    print("----- 预热期不误杀（seed_frame>0）-----")
    rec = StereoReconstructor(rig=None, warmup_frames=20)
    for _ in range(25):
        rec.update(0.0, None, [])          # 左路一直没检测 = 预热期的样子
    warm_ok = rec.n_frames == 25 and rec.match_rate == 0.0

    control_raised = False
    rec2 = StereoReconstructor(rig=None)     # 没有预热期：同样 25 帧应该判死
    try:
        for _ in range(25):
            rec2.update(0.0, None, [])
    except RuntimeError as exc:
        control_raised = "0 个配对" in str(exc)

    checks = [
        ("有 warmup=20：25 帧全空不抛错（预热期不参与判据）", warm_ok),
        ("对照：没有 warmup 时同样 25 帧必须抛『0 个配对』", control_raised),
    ]
    ok = True
    for name, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {name}")
        ok = ok and passed
    return ok


def _joint_rig(size=(640, 480)) -> StereoRig:
    """给跨视图联合细化用的平行装置（R=I、T=(-0.1,0,0)）。

    特意让 R=I：`stereoRectify` 出来的校正映射就是恒等映射，于是"校正后的 v"
    与原图 y 是同一个数，测试可以直接拿真值比，不必再换算一层。
    T 的符号按 OpenCV 约定（T 是"左→右"的平移），右相机在 x 的**负**方向 ——
    写反了三角化会给出 Z<0（`test_radius_anchor` 里踩过这个坑）。
    焦距按画幅的 0.8 倍取，640/1080p 两档都落在"正常镜头"的视场里。
    """
    fx = 0.8 * float(size[0])
    K = _K(fx, fx, size[0] / 2.0, size[1] / 2.0)
    D = np.zeros(5)
    R = np.eye(3)
    T = np.array([[-0.1], [0.0], [0.0]], np.float64)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K, D, K, D, size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.0)
    return StereoRig(K1=K, D1=D, K2=K, D2=D, R=R, T=T, image_size=size,
                     R1=R1, R2=R2, P1=P1, P2=P2, Q=Q)


def _draw_arc_ball(size, center, radius, a0, a1, highlight_dy, seed, noise=4.0):
    """画一颗"只有 (a1-a0) 这段圆周上有轮廓"的镜面球，外加一个偏离球心的亮斑。

    为什么用**填充扇形**而不是粗弧线：粗线会造出第二条圆边（内沿），环核会把它
    当成半径更小的假轮廓，圆心被拉走 6 px（第一版就踩了）。扇形只有一条阶跃边，
    正好落在 radius 上；两条辐条是径向边，梯度是切向的，既不计入轮廓支撑
    （支撑只看径向分量），也不随圆心平移变化，构不成竞争峰。

    高光用实心小圆（偏离球心 8 px）：镜面球的高光在左右视图落在球面不同位置，
    这正是检测框中心会往两边各偏一点的来由。
    """
    img = np.full((size[1], size[0]), 30, np.uint8)
    cx, cy = int(round(center[0])), int(round(center[1]))
    r = int(round(radius))
    cv2.ellipse(img, (cx, cy), (r, r), 0, a0, a1, 200, -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy + int(round(highlight_dy))), 5, 255, -1, cv2.LINE_AA)
    rng = np.random.default_rng(seed)
    out = img.astype(np.float32) + rng.normal(0.0, noise, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def test_joint_refine() -> bool:
    """跨视图联合细化：共享极线 v，把两路圆心**一起**解出来（B-166）。

    必须证明三件事，缺一不算：
      ① **问题真实存在**：两路各自独立细化出的圆心，校正后 v 差是像素级起步的；
      ② **联合细化解得对**：v 差落进 1 px，且两路圆心各自离真值 < 6 px
         （说明解出来的是**球心**，不是被高光拉偏的框中心）；
      ③ **诚实降级**：没轮廓时不采纳、坐标原样返回；也不许动半径。
    """
    print("----- 跨视图联合细化（共享极线 v） -----")
    import dataclasses
    import time

    size = (640, 480)
    rig = _joint_rig(size)
    truth_l, truth_r, rad = (300.0, 240.0), (280.0, 240.0), 40.0
    # 可见弧各 108 度 = 整圈 30%（`B-130` 实测 25%~35%）。
    # 左弧在 3 点钟（切线竖直：v 病态、u 清楚），右弧在 12 点钟（反过来）。
    # 故意的：任意一路单独都定不死圆心，只能靠对面那一路补信息。
    gray_l = _draw_arc_ball(size, truth_l, rad, -54, 54, -8.0, seed=3)
    gray_r = _draw_arc_ball(size, truth_r, rad, -144, -36, 8.0, seed=4)

    # 检测框中心被高光拉偏（CSRT 框就是这么偏的）：左偏上、右偏下各 6 px
    det_l = Detection(x=truth_l[0], y=truth_l[1] - 6.0, radius=rad)
    det_r = Detection(x=truth_r[0], y=truth_r[1] + 6.0, radius=rad)

    # ① 现有做法：两路各解各的（`refine_circle_subpixel_checked`，检测器用的就是它）
    ind = []
    for gray, det in ((gray_l, det_l), (gray_r, det_r)):
        x, y, _r, ok = refine_circle_subpixel_checked(gray, det.x, det.y, det.radius)
        ind.append((x, y, ok))
        print(f"  独立细化 {'左' if gray is gray_l else '右'}路：ok={ok} "
              f"({x:.2f}, {y:.2f})，相对检测框 dy={y - det.y:+.2f} px")
    vl = rectify_point_left(rig, ind[0][0], ind[0][1])
    vr = rectify_point_right(rig, ind[1][0], ind[1][1])
    dv_ind = abs(vl[1] - vr[1])
    print(f"  独立细化后两路 v 差 {dv_ind:.2f} px（真值应该差 0：同一行的同名点）")

    # ② 联合细化：输入就是上一步的结果（真实接线：检测 -> 独立细化 -> 配对）
    det_l2 = Detection(x=ind[0][0], y=ind[0][1], radius=rad)
    det_r2 = Detection(x=ind[1][0], y=ind[1][1], radius=rad)
    res = refine_pair(gray_l, gray_r, det_l2, det_r2, rig)
    print("  " + describe_result(res))
    err_l = float(np.hypot(res.left[0] - truth_l[0], res.left[1] - truth_l[1]))
    err_r = float(np.hypot(res.right[0] - truth_r[0], res.right[1] - truth_r[1]))
    print(f"  圆心误差：左 {err_l:.2f} px，右 {err_r:.2f} px；"
          f"轮廓支撑 {res.support_left:.2f} / {res.support_right:.2f}")

    # ③-1 没有任何轮廓（纯噪声）时必须拒绝，而且坐标逐位不变
    rng = np.random.default_rng(9)
    noise_l = np.clip(rng.normal(120, 8, (size[1], size[0])), 0, 255).astype(np.uint8)
    noise_r = np.clip(rng.normal(120, 8, (size[1], size[0])), 0, 255).astype(np.uint8)
    res_noise = refine_pair(noise_l, noise_r, det_l2, det_r2, rig)
    print("  " + describe_result(res_noise))
    noise_same = (res_noise.left == (det_l2.x, det_l2.y)
                  and res_noise.right == (det_r2.x, det_r2.y))

    # ③-2 v 差本来就在门限内（同一行基本对上了）-> 不值得动，也别白动
    gray_lf = _draw_arc_ball(size, truth_l, rad, 0, 360, -8.0, seed=13)
    gray_rf = _draw_arc_ball(size, truth_r, rad, 0, 360, 8.0, seed=14)
    det_lf = Detection(x=truth_l[0], y=truth_l[1], radius=rad)
    det_rf = Detection(x=truth_r[0], y=truth_r[1], radius=rad)
    res_full = refine_pair(gray_lf, gray_rf, det_lf, det_rf, rig)
    print("  " + describe_result(res_full))
    full_same = (res_full.left == (det_lf.x, det_lf.y)
                 and res_full.right == (det_rf.x, det_rf.y))

    # ③-3 未做极线校正的标定：没有"同一行"可言，必须直接说清
    rig_raw = dataclasses.replace(rig, meta={"rectification": "none_raw"})
    res_raw = refine_pair(gray_l, gray_r, det_l2, det_r2, rig_raw)
    print("  " + describe_result(res_raw))

    # ③-4 两路 v 差远超搜索窗：必须**拒绝**，不能"取个平均"把左边一路拽走。
    #
    # 这条是线上实测倒逼出来的：右路跟丢时两路 v 差能到 200+ px，而候选 v 的窗口
    # 若只围绕两路均值展开，就会跟着错误一起平移 —— 实测把左路圆心从球上拽到了
    # 130 px 外的背景墙上，然后"成功采纳"。所以夹逼必须同时用两侧原始 v。
    far = dataclasses.replace(det_l2, y=det_l2.y + 120.0)
    res_far = refine_pair(gray_l, gray_r, far, det_r2, rig)
    print("  " + describe_result(res_far))

    # ④ 耗时：1080p、r=70（真机量级），20 次取均值
    big = (1920, 1080)
    rig_big = _joint_rig(big)
    gl_b = _draw_arc_ball(big, (1000.0, 540.0), 70.0, -54, 54, -10.0, seed=5)
    gr_b = _draw_arc_ball(big, (925.0, 540.0), 70.0, -144, -36, 10.0, seed=6)
    dl_b = Detection(x=1000.0, y=530.0, radius=70.0)
    dr_b = Detection(x=925.0, y=550.0, radius=70.0)
    refine_pair(gl_b, gr_b, dl_b, dr_b, rig_big)          # 预热（建映射/环核）
    t0 = time.perf_counter()
    for _ in range(20):
        res_big = refine_pair(gl_b, gr_b, dl_b, dr_b, rig_big)
    ms_pair = (time.perf_counter() - t0) / 20.0 * 1000.0
    print(f"  1080p（r=70）单帧耗时 {ms_pair:.2f} ms/对；"
          f"|v 差| {res_big.dv_before:.2f} -> {res_big.dv_after:.2f} px")

    checks = [
        (f"独立细化后两路 v 差 {dv_ind:.2f} px（> 2：两路各偏一边，问题真实存在）",
         dv_ind > 2.0),
        (f"联合细化：已采纳且 |v 差| {res.dv_after:.2f} px（< 1.0，落得进极线门）",
         res.applied and res.dv_after < 1.0),
        (f"共享 v 的契约：两路校正后 y 相等，残差 {res.dv_after:.4f} px（< 0.05，"
         f"只剩数值往返误差）", res.dv_after < 0.05),
        (f"支撑保留率可查：own {res.support_left_own:.2f} / {res.support_right_own:.2f}"
         f" -> 共享 {res.support_left:.2f} / {res.support_right:.2f}（都不掉，"
         f"证明这两处是同一条轮廓）",
         res.support_left_own > 0.0 and res.support_right_own > 0.0
         and res.support_left >= 0.85 * res.support_left_own
         and res.support_right >= 0.85 * res.support_right_own),
        (f"联合细化的两路圆心离真值 {err_l:.2f} / {err_r:.2f} px（各 < 6：解的是球心）",
         err_l < 6.0 and err_r < 6.0),
        ("reason 非空且摘要带 [已采纳]（用户能看懂为什么动）",
         bool(res.reason.strip()) and "[已采纳]" in describe_result(res)),
        (f"纯噪声：不采纳且坐标原样（支撑 {res_noise.support_left:.2f} / "
         f"{res_noise.support_right:.2f}）", (not res_noise.applied) and noise_same),
        ("v 差本来就在 0.5 px 内：不采纳且坐标原样（不值得动就不动）",
         (not res_full.applied) and full_same),
        ("未做极线校正：不采纳并说明原因（不假装能算）",
         (not res_raw.applied) and "极线校正" in res_raw.reason
         and res_raw.left == (det_l2.x, det_l2.y)),
        (f"两路 v 差 {res_far.dv_before:.0f}px 远超搜索窗：必须拒绝（不许取平均把"
         f"圆心拽走），且原样返回",
         (not res_far.applied) and "同一颗球" in res_far.reason
         and res_far.left == (far.x, far.y)),
        ("半径不被改（镜面球半径量不准，只解圆心）",
         det_l2.radius == rad and det_r2.radius == rad),
        (f"1080p 一对 {ms_pair:.2f} ms（< 20，防退化成逐点采样）", ms_pair < 20.0),
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
    d = test_visualization()
    print()
    e = test_angled()
    print()
    f = test_fov_alignment()
    print()
    p3 = test_predict3d()
    print()
    g = test_pipeline_blocks()
    print()
    h = test_radius_anchor()
    print()
    w = test_warmup_no_false_alarm()
    print()
    j = test_joint_refine()
    print()
    results = [a, b, c, d, e, f, p3, g, h, w, j]
    allok = all(results)
    print(f"合计 {len(results)} 项：通过 {sum(1 for r in results if r)}，"
          f"不达标 {sum(1 for r in results if not r)}")
    print("判定结果：" + ("全部通过" if allok else "有不达标项"))
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
