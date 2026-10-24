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
from stereo.fov_align import FovAlignment, estimate_fov_alignment
from stereo.visualization import (
    build_trajectory_payload, export_session_visualization, load_seeds, write_seeds,
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
    """FOV 自动对齐：已知 1.5x 合成视场差、同视场、空白帧、序列化。"""
    print("----- 双目 FOV 自动对齐 -----")
    cv2.setRNGSeed(20261024)
    left, right = _fov_pair()
    checks = []
    align = None

    try:
        align = estimate_fov_alignment(left, right)
        aligned_left, aligned_right = align.apply(left, right)
        x, y, w, h = align.crop_rect
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
             and align.target_size == (2560, 1440)
             and 0 <= x and 0 <= y and x + w <= 2560 and y + h <= 1440
             and abs(x - 426) <= 25 and abs(y - 240) <= 25
             and abs(w - 1707) <= 25 and abs(h - 960) <= 25),
            ("参考帧原样保留；广角帧裁剪后缩放到参考尺寸",
             np.array_equal(aligned_left, left)
             and aligned_right.shape == (1440, 2560, 3)
             and not np.shares_memory(aligned_right, right)),
        ])
        ref_only = align.apply_side(left, "left")
        crop_only = align.apply_side(right, "right")
        checks.append((
            "apply_side：只归一化裁剪路，参考路原样、两路尺寸一致",
            ref_only is left and ref_only.shape == left.shape
            and crop_only.shape == aligned_right.shape
            and np.array_equal(crop_only, aligned_right),
        ))
        post = estimate_fov_alignment(left, aligned_right)
        checks.append((f"对齐后残余尺度 {post.scale:.4f}（目标 1±0.1）",
                       abs(post.scale - 1.0) <= 0.10))
    except ValueError as exc:
        checks.append((f"1.5x 合成帧对估计失败：{exc}", False))

    try:
        mirror = estimate_fov_alignment(right, left)
        checks.append((f"交换左右（右更窄）-> 参考 {mirror.reference_side} / "
                       f"裁剪 {mirror.crop_side}，尺度 {mirror.scale:.3f}",
                       mirror.reference_side == "right"
                       and mirror.crop_side == "left"
                       and abs(mirror.scale - 1.5) <= 0.075))
    except ValueError as exc:
        checks.append((f"右窄左宽估计失败：{exc}", False))

    try:
        same = estimate_fov_alignment(left, left.copy())
        same_l, same_r = same.apply(left, left.copy())
        checks.append(("相同画面 -> crop_side='none'，不缩放",
                       same.crop_side == "none" and not same.active
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

    # 参考帧四角越界的判定：<=1 px 只容忍取整/浮点误差；>1 px 即共同视场不足。
    try:
        from stereo.fov_align import _mapped_rect

        corners = np.float32(
            [[0, 0], [100, 0], [100, 100], [0, 100]]).reshape(-1, 1, 2)
        h_tol = np.array([[1.0, 0.0, -0.5], [0.0, 1.0, -0.5], [0.0, 0.0, 1.0]])
        rect = _mapped_rect(corners, h_tol, 100, 100)
        h_bad = np.array([[1.0, 0.0, -1.5], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        try:
            _mapped_rect(corners, h_bad, 100, 100)
        except ValueError:
            bad_raised = True
        else:
            bad_raised = False
        checks.append(("四角越界 <=1 px 容忍取整；>1 px 报共同视场不足",
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
    print("判定结果：" + ("全部通过" if (a and b and c and d and e and f) else "有不达标项"))
    return 0 if (a and b and c and d and e and f) else 1


if __name__ == "__main__":
    raise SystemExit(main())
