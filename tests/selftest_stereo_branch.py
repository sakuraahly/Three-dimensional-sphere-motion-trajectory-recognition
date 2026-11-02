"""双目链路端到端自测（合成真值）：两路 2D 检测 -> 同步取帧 -> 配对 -> 三维。

和另外两个双目测试的分工（别搞混）：

    selftest_stereo_calib.py     只测**标定**：合成标定板 -> 内参/基线/极线误差
    selftest_stereo_pipeline.py  只测**算法**：直接把真值像素喂进配对与三角测量
    selftest_stereo_branch.py    测**整条链路接起来**：本文件
                                 —— 渲染真实双目画面（噪声/运动模糊/干扰物）
                                 -> 两路各自跑 SpherePipeline（检测+跟踪）
                                 -> StereoReconstructor 配成三维 -> 比真值

为什么它不能被前两个替代
------------------------
前两个测试都假设"像素检测是准的"。而实际误差的大头往往在**检测**这一侧：
球是白色圆，背景里还有同色的白方块和同形的橙色圆；运动模糊会把圆边拖成月牙。
这些只有把两路画面真的渲染出来、真的跑一遍检测器，才能暴露。

顺带钉死两件事：
  1. **同步到底值不值**：把右路检测人为错开 1/2/4 帧，看三维误差涨多少。
     这是"必须做同步"的量化证据，也是 sync_offset 接对了没有的回归防线；
  2. **双目比单目强多少**：同一个球，单目靠像素半径推深度（Z=fR/r），
     双目靠三角测量。两条路都比一遍，避免"以为用了双目其实还在靠半径"。

另外钉两条**入口层让用户少踩坑**的判据（2026-10-30 补，都是真实现场翻车过的）：
  3. 会话里有 `derived/seeds.json` 且没显式给 `--method` -> 自动用 specular；
  4. `parallel` 0 配对的报错要**按分项**给建议（"没检出"不能引向 angled）；
  5. 配对率低时要把"极线偏差 vs 标定自报"量出来——几何对不上时说"重新标定"，
     而不是让人继续调 `--max-epipolar`；
  6. 双目圆心取**轮廓**（不是高光斑）、跑满 N 对先判一次几何（`B-132` / `B-131`）。

    python tests/selftest_stereo_branch.py
    python tests/selftest_stereo_branch.py --scenario clean
    python tests/selftest_stereo_branch.py --write-video
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
for p in (str(ROOT), str(ROOT / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2
import numpy as np

from calib import StereoRig
from selftest_synthetic import Scene, make_background, render_frame
from sphere_tracker.config import AppConfig, ColorConfig
from sphere_tracker.geometry import CameraModel, refine_circle_subpixel, \
    refine_circle_subpixel_checked, estimate_radius_px
from sphere_tracker.pipeline import SpherePipeline
from sphere_tracker.types import Detection
from stereo import MatchConfig, StereoReconstructor, epipolar_dy, geometry_verdict
from stereo.triangulate import rectify_point_left, rectify_point_right

# --------------------------------------------------------------------------
# 双目装置真值
# --------------------------------------------------------------------------
WIDTH, HEIGHT = 960, 540
FPS = 60.0
DURATION = 3.0
BALL_DIAMETER_M = 0.22
BASELINE_M = 0.150                 # 150 mm：比标定自测的 100 mm 略宽，
                                   # 视差更大、深度更稳，也更接近真实支架

# 两路内参故意不同（真实的两台相机不会一样）
K1 = np.array([[850.0, 0.0, 480.0], [0.0, 850.0, 270.0], [0.0, 0.0, 1.0]])
K2 = np.array([[838.0, 0.0, 466.0], [0.0, 838.0, 264.0], [0.0, 0.0, 1.0]])
D1 = np.zeros(5)
D2 = np.zeros(5)
R_REL, _ = cv2.Rodrigues(np.radians([0.4, -0.8, 0.25]))   # 右相机相对左相机的旋转
T_REL = np.array([BASELINE_M, 0.002, 0.003]).reshape(3, 1)  # 右光心在左相机系里的位置


def build_rig(size: tuple[int, int] = (WIDTH, HEIGHT)) -> StereoRig:
    """用真值搭一个 StereoRig（**不跑标定** —— 标定精度归 selftest_stereo_calib）。

    这样测出来的误差就只归因于"检测 + 配对 + 三角测量"这一段，
    标定误差另算，出问题时能立刻分清是哪一段的锅。
    """
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, size, R_REL, T_REL,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.0,
    )
    return StereoRig(
        K1=K1, D1=D1, K2=K2, D2=D2, R=R_REL, T=T_REL, image_size=size,
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        quality={"epipolar_rms": 0.0, "rms": 0.0},
        meta={"board": "合成真值（未跑标定）", "baseline_mm": BASELINE_M * 1000},
    )


def make_pipeline(K: np.ndarray) -> SpherePipeline:
    """一路单目流水线。内参用真值 —— 要测的是"检测+跟踪+配对"，不是内参估计。"""
    cfg = AppConfig()
    cfg.detector.method = "color"
    cfg.detector.color = ColorConfig.preset("white_ball")
    cfg.predict.enable = False                  # 本测试不测预测
    cfg.camera.hfov_deg = math.degrees(2.0 * math.atan((WIDTH / 2.0) / K[0, 0]))
    cfg.camera.fx = float(K[0, 0])
    cfg.camera.ball_diameter_m = BALL_DIAMETER_M
    pipe = SpherePipeline(cfg)
    # 直接把真值相机模型塞进去：fx/fy/cx/cy 都是标定出来的值。
    # （否则 SpherePipeline 会用 hfov 估算 fx，单目测深会整体偏掉，
    #   那属于"没标定"，不是我们这里要测的东西。）
    pipe.camera = CameraModel(
        width=WIDTH, height=HEIGHT,
        fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]),
        ball_diameter_m=BALL_DIAMETER_M,
    )
    return pipe


# --------------------------------------------------------------------------
# 一轮采集：渲染两路画面，各自跑流水线，记录检测与跟踪结果
# --------------------------------------------------------------------------
class FrameLog:
    """一帧的两路结果。"""

    __slots__ = ("t", "det_l", "det_r", "trk_l", "trk_r", "gt3", "visible")

    def __init__(self, t, det_l, det_r, trk_l, trk_r, gt3, visible):
        self.t = t
        self.det_l = det_l
        self.det_r = det_r
        self.trk_l = trk_l
        self.trk_r = trk_r
        self.gt3 = gt3
        self.visible = visible


def _as_detection(state, fallback: Detection | None) -> Detection | None:
    """把跟踪状态转成 Detection 喂给配对器。

    用跟踪后的圆心：跟踪器滤掉了逐帧抖动，配对更稳；
    半径仍取原始观测（跟踪后的半径有滞后，而半径只用于深度兜底）。
    """
    if state is None or state.coasting:
        return fallback
    r = fallback.radius if fallback is not None else state.radius
    return Detection(x=state.x, y=state.y, radius=r, score=1.0, method="track")


def collect(scene: Scene, rig: StereoRig, rng: np.random.Generator,
            write_video: bool = False) -> list[FrameLog]:
    bg = make_background(scene, rng)
    pipe_l = make_pipeline(K1)
    pipe_r = make_pipeline(K2)

    writer = None
    if write_video:
        out_dir = ROOT / "outputs" / "videos"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "synthetic_stereo_side_by_side.mp4"
        # 两路并排；两路分辨率相同（本项目里不同分辨率要各自缩放，见踩坑 7）
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"),
                                 scene.fps, (WIDTH * 2, HEIGHT))

    logs: list[FrameLog] = []
    cam_r = {"side": "right", "R": R_REL, "T": T_REL, "K": K2, "D": D2}
    for i in range(scene.n):
        t = i / scene.fps
        # 两路各自渲染自己看到的画面（同一物理时刻）
        frame_l = render_frame(scene, bg, t, rng, K=K1, D=D1)
        frame_r = render_frame(scene, bg, t, rng, cam=cam_r)

        res_l = pipe_l.process(frame_l, t)
        res_r = pipe_r.process(frame_r, t)

        gt3 = scene.center3d(t)
        vis_l = scene.fully_visible(t, K=K1, D=D1, side="left")
        vis_r = scene.fully_visible(t, **cam_r)
        logs.append(FrameLog(
            t=t, det_l=res_l.detection, det_r=res_r.detection,
            trk_l=res_l.track, trk_r=res_r.track, gt3=gt3,
            visible=(vis_l and vis_r),
        ))

        if writer is not None:
            vis_l = pipe_l.render(frame_l, res_l)
            vis_r = pipe_r.render(frame_r, res_r)
            writer.write(np.hstack([vis_l, vis_r]))
    if writer is not None:
        writer.release()
        print(f"  已导出并排视频：outputs/videos/synthetic_stereo_side_by_side.mp4")
    return logs


# --------------------------------------------------------------------------
# 一轮评估
# --------------------------------------------------------------------------
def depth_error_coeff(rig: StereoRig) -> float:
    """1 px 视差误差会变成多少米深度误差：εZ = Z² / (f · B)。

    这是双目精度的**物理定律**，不是实现好坏：深度误差随距离**平方**增长。
    所以"1 m 处 ≤ 10 mm"和"4 m 处 ≤ 10 mm"是完全不同的要求
    （后者要求的检测精度是前者的 16 倍）。
    本测试的球在 3.0–4.2 m，绝对阈值必须按这条定律折算，否则门槛毫无意义。
    """
    f = float(rig.P1[0, 0])
    b = float(np.linalg.norm(rig.T))
    return 1.0 / (f * b)


def mm_at_1m(rig: StereoRig, rmse_mm: float, z_ref: float) -> float:
    """把"在 z_ref 处测到的三维 RMSE"折算到 1 m 处（εZ ∝ Z²）。"""
    if z_ref <= 0:
        return float("nan")
    return rmse_mm * (1.0 / z_ref) ** 2


def evaluate(logs: list[FrameLog], rig: StereoRig, scene: Scene,
             right_lag: int = 0) -> dict:
    """把两路结果配成三维，统计误差。right_lag>0 表示右路检测人为滞后若干帧。"""
    rec = StereoReconstructor(rig, MatchConfig(max_epipolar_px=3.0))
    err3d: list[float] = []
    err_xy: list[float] = []
    err_z: list[float] = []
    mono_l: list[float] = []
    mono_r: list[float] = []
    det_l_err: list[float] = []
    det_r_err: list[float] = []
    trk_l_err: list[float] = []
    trk_r_err: list[float] = []
    det_l_hits = det_r_hits = 0
    both_det = 0
    frames_seen = 0

    cam_r = {"side": "right", "R": R_REL, "T": T_REL, "K": K2, "D": D2}
    for i, log in enumerate(logs):
        t = log.t
        uv_l = scene_uv(scene, t, "left", K1, D1)
        uv_r = scene_uv(scene, t, **cam_r)

        if log.det_l is not None:
            det_l_hits += 1
            det_l_err.append(float(np.linalg.norm(
                np.array([log.det_l.x, log.det_l.y]) - np.array(uv_l))))
        if log.det_r is not None:
            det_r_hits += 1
            det_r_err.append(float(np.linalg.norm(
                np.array([log.det_r.x, log.det_r.y]) - np.array(uv_r))))
        if log.trk_l is not None and not log.trk_l.coasting:
            trk_l_err.append(float(np.linalg.norm(
                np.array([log.trk_l.x, log.trk_l.y]) - np.array(uv_l))))
        if log.trk_r is not None and not log.trk_r.coasting:
            trk_r_err.append(float(np.linalg.norm(
                np.array([log.trk_r.x, log.trk_r.y]) - np.array(uv_r))))
        if log.det_l is not None and log.det_r is not None:
            both_det += 1
        if log.visible:
            frames_seen += 1

        left = _as_detection(log.trk_l, log.det_l)
        j = i - right_lag
        right_log = logs[j] if j >= 0 else None
        right = _as_detection(right_log.trk_r, right_log.det_r) if right_log else None
        cands = [right] if right is not None else []

        pt = rec.update(t, left, cands)
        if pt is None:
            continue
        truth_rect = rig.R1 @ log.gt3          # 三角测量结果在【校正后】左相机系
        d = pt.xyz - truth_rect
        err3d.append(float(np.linalg.norm(d)))
        err_xy.append(float(np.hypot(d[0], d[1])))
        err_z.append(abs(float(d[2])))

        # 单目对照：Z = f * R / r（用的是这一路自己的像素半径）
        for det, K, D, store in ((log.det_l, K1, D1, mono_l),
                                 (log.det_r, K2, D2, mono_r)):
            if det is None or det.radius <= 1e-6:
                continue
            f = 0.5 * (float(K[0, 0]) + float(K[1, 1]))
            z_est = f * (BALL_DIAMETER_M / 2.0) / det.radius
            store.append(abs(z_est - float(log.gt3[2])) / float(log.gt3[2]))

    def stats(xs: list[float]) -> tuple[float, float, float]:
        if not xs:
            return float("nan"), float("nan"), float("nan")
        a = np.asarray(xs, np.float64)
        return float(a.mean()), float(np.sqrt(np.mean(a * a))), float(a.max())

    mean3, rmse3, max3 = stats(err3d)
    z_ref = float(np.mean([log.gt3[2] for log in logs])) if logs else float("nan")
    return {
        "frames": len(logs),
        "visible_frames": frames_seen,
        "det_l_rate": det_l_hits / max(len(logs), 1),
        "det_r_rate": det_r_hits / max(len(logs), 1),
        "det_l_rmse_px": stats(det_l_err)[1],
        "det_r_rmse_px": stats(det_r_err)[1],
        "trk_l_med_px": float(np.median(trk_l_err)) if trk_l_err else float("nan"),
        "trk_r_med_px": float(np.median(trk_r_err)) if trk_r_err else float("nan"),
        "both_det_frames": both_det,
        "match_rate": rec.n_matched / max(both_det, 1),
        "rejected_frames": rec.n_rejected,
        "disparity_sign": rec.cfg.disparity_sign,
        "residual_med_px": (float(np.median([p.residual_px for p in rec.trajectory]))
                            if rec.trajectory else float("nan")),
        "err3d_mean_mm": mean3 * 1000,
        "err3d_rmse_mm": rmse3 * 1000,
        "err3d_max_mm": max3 * 1000,
        "err_xy_rmse_mm": stats(err_xy)[1] * 1000,
        "err_z_mean_mm": stats(err_z)[0] * 1000,
        "z_ref_m": z_ref,
        "coeff_m_per_px": depth_error_coeff(rig),
        "rmse_at_1m_mm": mm_at_1m(rig, rmse3 * 1000, z_ref),
        "mono_l_depth_pct": stats(mono_l)[0] * 100,
        "mono_r_depth_pct": stats(mono_r)[0] * 100,
        "summary": rec.summary(),
    }


# 真值投影的小包装：把 project() 的 (u, v, r) 收成 (u, v)
def scene_uv(scene: Scene, t: float, side: str, K: np.ndarray, D: np.ndarray,
             **cam) -> tuple[float, float]:
    u, v, _r = scene.project(t, side=side, K=K, D=D, **cam)
    return u, v


# --------------------------------------------------------------------------
def run_scenario(scenario: str, write_video: bool = False) -> dict:
    realistic = scenario == "realistic"
    scene = Scene(
        width=WIDTH, height=HEIGHT, fps=FPS, duration=DURATION,
        ball_diameter_m=BALL_DIAMETER_M,
        motion_blur=realistic, noise=realistic, distractors=True,
        # 深度范围 1.8–4.2 m：原单目自测用 2.4–6.4 m，6 m 外球的像素半径
        # 只有 15 px 左右；双目这里没必要那么极端，收紧后"检测精度 -> 三维精度"
        # 的传递关系看得更清楚。
        #
        # z 初速度从 0.14 提到 0.55 m/s：默认那点 z 速度在 3 秒里只走 0.4 m，
        # 深度几乎不变 —— 而**深度方向的误差恰恰是双目最该被检验的一项**，
        # 轨迹不动就等于没测。0.55 m/s 让深度在 3 秒里走完大半个范围。
        z_range=(1.8, 4.2),
        velocity=(0.95, -2.0, 0.55),
    )
    rig = build_rig((scene.width, scene.height))
    rng = np.random.default_rng(20241005)

    print(f"  基线 {rig.baseline_mm:.1f} mm   深度范围 "
          f"{scene.pos[:, 2].min():.2f}-{scene.pos[:, 2].max():.2f} m   "
          f"球直径 {BALL_DIAMETER_M * 100:.0f} cm   {scene.n} 帧")
    logs = collect(scene, rig, rng, write_video=write_video)

    out = evaluate(logs, rig, scene, right_lag=0)
    out["scenario"] = scenario

    print(f"  两路检出率 左 {out['det_l_rate'] * 100:.1f}% / 右 {out['det_r_rate'] * 100:.1f}%"
          f"    检测 RMSE 左 {out['det_l_rmse_px']:.2f} px / 右 {out['det_r_rmse_px']:.2f} px"
          f"    跟踪中位 左 {out['trk_l_med_px']:.2f} px / 右 {out['trk_r_med_px']:.2f} px")
    print(f"  配对：{out['summary']}    配对率 {out['match_rate'] * 100:.1f}%"
          f"（装置朝向 {rig.baseline_side()}，视差符号 {out['disparity_sign']:+d}）")
    print(f"  三维误差：均值 {out['err3d_mean_mm']:.2f} mm / RMSE {out['err3d_rmse_mm']:.2f} mm"
          f" / 最大 {out['err3d_max_mm']:.2f} mm"
          f"   （横向 {out['err_xy_rmse_mm']:.2f} mm，深度 {out['err_z_mean_mm']:.2f} mm）")
    print(f"  折算到 1 m 处：{out['rmse_at_1m_mm']:.2f} mm"
          f"（εZ = Z²/(f·B) = {out['coeff_m_per_px'] * 1000:.2f} mm 深度误差 / 1 px 视差误差，"
          f"Z≈{out['z_ref_m']:.2f} m；项目验收口径为 1 m 处 ≤ 10 mm）")
    print(f"  单目对照（只靠像素半径推深度）：左 {out['mono_l_depth_pct']:.2f}% / "
          f"右 {out['mono_r_depth_pct']:.2f}% 深度相对误差")

    # ---- 同步敏感性：右路检测人为滞后若干帧 ----
    print("  同步敏感性（右路检测人为滞后 -> 三维误差）：")
    sync_rows = []
    for lag in (0, 1, 2, 4):
        r = out if lag == 0 else evaluate(logs, rig, scene, right_lag=lag)
        sync_rows.append((lag, r["err3d_mean_mm"], r["err3d_rmse_mm"]))
        print(f"      滞后 {lag} 帧（{lag / FPS * 1000:4.0f} ms）："
              f"均值 {r['err3d_mean_mm']:7.2f} mm   RMSE {r['err3d_rmse_mm']:7.2f} mm")
    out["sync_rows"] = sync_rows
    return out


def test_baseline_sign() -> bool:
    """左右装反时，配对不能静默归零。

    这是写这条测试时**真的抓到的一个隐患**，值得单独钉住：

    真实场景里"左右装反"很常见 —— 两台同型号相机重新插拔/重启后索引互换，
    画面完全正常，标定也能过（基线长度、重投影误差都好看），
    但 `T[0]` 反号 -> 校正后视差整体反号。

    原来的 `match_epipolar` 把"视差必须为正"写死在范围检查里
    （`min_disparity_px <= disparity <= max_disparity_px`），于是装反时
    **每一个点都被静默拒绝**：配对率 0%，报错只有"配不上"，
    完全指不到原因。三角测量本身没错（P1/P2 自洽），所以更隐蔽。

    修法：范围检查改用 |视差|，符号单独判，并按装置实际朝向自动定符号。

    验证方式上踩过一次坑，记在这里：**不要自己手写"校正后投影"**。
    第一版测试用 `P[0,0]*x/z + P[0,2]` 算校正后像素，视差算出来是 343 px
    （真值 43 px）；改成 `projectPoints` 又把**已经校正过**的像素喂给
    `rectify_point_left`（它内部还会再校正一次），dy 变成 6.1 px 配不上。
    校正量必须"一路配套"（踩坑 8）。所以这里直接用真值场景 + 真值投影，
    让整条链路自己跑出视差 —— 既避开了手工投影，也顺带验证了配套关系。
    """
    print("----- 左右装反（基线反号）时的配对行为 -----")
    K1 = np.array([[850.0, 0.0, 480.0], [0.0, 850.0, 270.0], [0.0, 0.0, 1.0]])
    K2 = np.array([[838.0, 0.0, 466.0], [0.0, 838.0, 264.0], [0.0, 0.0, 1.0]])
    D = np.zeros(5)
    R_rel = np.eye(3)

    def rig_with(tx: float) -> StereoRig:
        T = np.array([tx, 0.0, 0.0]).reshape(3, 1)
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            K1, D, K2, D, (WIDTH, HEIGHT), R_rel, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.0,
        )
        return StereoRig(K1=K1, D1=D, K2=K2, D2=D, R=R_rel, T=T,
                         image_size=(WIDTH, HEIGHT), R1=R1, R2=R2,
                         P1=P1, P2=P2, Q=Q, quality={}, meta={})

    # 场景：球静止在 (0.05, 0.02, 3.0)，两路都是真值投影
    scene = Scene(width=WIDTH, height=HEIGHT, fps=FPS, duration=0.2,
                  ball_diameter_m=BALL_DIAMETER_M, motion_blur=False, noise=False,
                  distractors=False, z_range=(2.9, 3.1), velocity=(0.0, 0.0, 0.0))
    truth_pt = None

    checks: list[tuple[str, bool]] = []
    # ⚠️ 2026-10-18 修正了这里的**标签**（数值断言原本就对）：
    # `T` 是"左光心在右相机系里的位置"（`T = -R·C`，`C` = 右光心在左相机系里的位置），
    # 所以 **`T[0] < 0` 才对应"右相机在右边"**。原来的标签恰好标反了
    # （把 `tx=+0.15` 叫"正常"），与 `baseline_side()` 的错误实现互相印证。
    # 真值依据：`outputs/_handoff_read/audit_T_sign.py`。
    for name, tx, want_side, want_sign in (
        ("右相机在右边（正常，T[0]<0）", -0.15, "right", +1),
        ("右相机在左边（左右装反，T[0]>0）", +0.15, "left", -1),
    ):
        rig = rig_with(tx)
        side, sign = rig.baseline_side(), rig.disparity_sign()
        checks.append((f"{name}：baseline_side={side}（应为 {want_side}）", side == want_side))
        checks.append((f"{name}：视差符号 {sign:+d}（应为 {want_sign:+d}）", sign == want_sign))

        # 把 `P2[0,3] = f·C = -f·T[0]` 这个关系钉住 —— 符号判定的依据就是它
        implied = -float(rig.P2[0, 3]) / float(rig.P2[0, 0])
        checks.append((f"{name}：P2 第 4 列反推基线 {implied:+.4f} 与 T[0]={tx:+.2f} 反号",
                       abs(implied + tx) < 1e-6))

        cam_r = {"side": "right", "R": R_rel, "T": T_of(tx), "K": K2, "D": D}
        t = 0.0
        ul, vl, rl = scene.project(t, side="left", K=K1, D=D)
        ur, vr, rr = scene.project(t, **cam_r)
        truth_pt = scene.center3d(t)

        # 先确认视差的实际符号确实与判定一致
        hl = rectify_point_left(rig, ul, vl)
        hr = rectify_point_right(rig, ur, vr)
        disp = hl[0] - hr[0]
        checks.append((f"{name}：实测视差 {disp:+.2f} px 与判定符号 {sign:+d} 一致",
                       math.copysign(1.0, disp) == sign))

        rec = StereoReconstructor(rig, MatchConfig(max_epipolar_px=3.0))
        got = rec.update(t, Detection(x=ul, y=vl, radius=rl),
                         [Detection(x=ur, y=vr, radius=rr)])
        ok = got is not None
        checks.append((f"{name}：能配上（自动取符号 {rec.cfg.disparity_sign:+d}）", ok))
        if ok:
            err_mm = float(np.linalg.norm(got.xyz - rig.R1 @ truth_pt)) * 1000
            checks.append((f"{name}：三角测量误差 {err_mm:.3f} mm < 0.5 mm", err_mm < 0.5))

    # 装反时三维**依然正确**（只是视差反号）—— 这正是它难被发现的原因
    ok_all = True
    for label, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {label}")
        ok_all = ok_all and passed
    return ok_all


def T_of(tx: float) -> np.ndarray:
    return np.array([tx, 0.0, 0.0]).reshape(3, 1)


# --------------------------------------------------------------------------
def test_seed_auto_detector() -> bool:
    """会话里有种子时该用哪个检测器 —— 判据是"用户有没有显式 --method"。

    现场（2026-10-30）：GUI 点「离线三维重建」跑 `data/stereo/20261021-151144`，
    3 秒后抛"0 配对，装置夹角可能远大于 parallel 的假设"。真实原因是默认检测器
    `color` 对镜面球恒为 0 检出（交接文档 §39.3），而正确做法所需的一切
    （用 specular + 会话里那份 seeds.json）都已经躺在会话目录里，只是没人用。

    这里不跑视频，只钉 `_pick_detector` 的**判据**，因为它最容易写错成
    "值等于默认值就换" —— 那样显式 `--method color` 会被悄悄改掉，
    用户下次就不敢相信自己敲的命令了。
    """
    print("----- 有种子 / 没种子 / 显式 --method 时，检测器怎么选 -----")
    import json as _json
    import tempfile

    from apps.run_stereo_video import _pick_detector
    from stereo.visualization import SEEDS_SCHEMA, VERSION

    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        good = tmp / "seeds.json"
        good.write_text(_json.dumps({
            "schema": SEEDS_SCHEMA, "version": VERSION, "seed_frame": 0,
            "left": {"x": 100.0, "y": 200.0, "radius": 20.0},
            "right": {"x": 90.0, "y": 210.0, "radius": 19.0},
        }, ensure_ascii=False), encoding="utf-8")
        broken = tmp / "broken.json"
        broken.write_text("{ 这不是 json", encoding="utf-8")
        missing = tmp / "missing.json"

        got, note = _pick_detector("color", False, good, 0)
        checks.append(("没给 --method + 有种子 -> 用 specular", got == "specular"))
        checks.append(("自动切换必须打印原因（含『种子』）", "种子" in note))

        got, note = _pick_detector("color", True, good, 0)
        checks.append(("显式 --method color -> 不许被改，且不打印",
                       got == "color" and not note))

        got, _ = _pick_detector("hough", True, good, 0)
        checks.append(("显式 --method hough -> 不许被改", got == "hough"))

        got, note = _pick_detector("color", False, broken, 0)
        checks.append(("种子文件坏了 -> 退回 color，不抛且留警告",
                       got == "color" and "警告" in note))

        got, note = _pick_detector("color", False, missing, 0)
        checks.append(("没有种子文件 -> 保持 color、不啰嗦", got == "color" and not note))

        got, _ = _pick_detector("color", False, good, 960)
        checks.append(("给了 --detect-width -> 不换（种子是原图坐标，缩放会错位）",
                       got == "color"))

    ok = True
    for label, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {label}")
        ok = ok and passed
    return ok


def test_zero_match_message() -> bool:
    """0 配对的报错必须区分"检测器没检出"与"有候选但配不上"。

    现场那句话原来把三种原因揉成一句、建议永远指向 `--stereo-mode angled`：
    左路 20 帧全空（检测问题）也被指去改几何，照着做只会再失败一次。

    判据依据：`_match_frame` 每帧只会落进三个计数器之一，所以
    ``左路缺 + 右路无候选 + 被拒 == 总帧数`` 恒成立 —— 因此
    "纯检测问题"等价于"被拒 == 0"，"纯配对问题"等价于两个缺帧计数都是 0。
    """
    print("----- 0 配对报错：分项不同，建议必须不同 -----")
    rig = build_rig()                       # 合成平行装置，光轴夹角约 0.9°
    # 90° 装置：只有这种几何才该被建议"试 angled"（判据见 `_angled_suggestion`）
    rig_angled = build_rig()
    rig_angled.R = cv2.Rodrigues(np.radians([0.0, 90.0, 0.0]))[0]

    def make(*, r=None, **counts) -> "StereoReconstructor":
        rec = StereoReconstructor(r if r is not None else rig,
                                  MatchConfig(max_epipolar_px=2.5))
        for k, v in counts.items():
            setattr(rec, k, v)
        return rec

    only_det = make(n_frames=20, n_no_left=20, n_no_right=0, n_rejected=0)
    m_det = only_det._zero_match_message()                      # noqa: SLF001
    only_rej = make(r=rig_angled, n_frames=20, n_no_left=0, n_no_right=0, n_rejected=20)
    m_rej = only_rej._zero_match_message()                      # noqa: SLF001
    flat_rej = make(n_frames=20, n_no_left=0, n_no_right=0, n_rejected=20)
    m_flat = flat_rej._zero_match_message()                     # noqa: SLF001
    mixed = make(r=rig_angled, n_frames=20, n_no_left=13, n_no_right=0, n_rejected=7)
    m_mix = mixed._zero_match_message()                         # noqa: SLF001
    hinted = make(n_frames=20, n_no_left=20, n_rejected=0)
    hinted.source_hint = "data/stereo/20261021-151144"
    m_hint = hinted._zero_match_message()                       # noqa: SLF001

    checks = [
        ("全是没检出 -> 建议 specular + 种子", "specular" in m_det and "seeds.json" in m_det),
        ("全是没检出 -> **不许**再提 angled", "angled" not in m_det),
        ("全是配对被拒（90° 装置）-> 才建议 angled", "angled" in m_rej),
        ("全是配对被拒 -> 不提 specular（那不是这里的原因）", "specular" not in m_rej),
        ("全是配对被拒（夹角 0.9° 的平行装置）-> 不许提 angled",
         "angled" not in m_flat and "标定" in m_flat),
        ("两种都有 -> 两段建议都在", "specular" in m_mix and "angled" in m_mix),
        ("给了会话路径 -> 报错里就是可复制的完整路径",
         "data/stereo/20261021-151144/derived/seeds.json" in m_hint),
    ]

    # 光测文案不够：真实抛出来的 RuntimeError 也必须带着新文案
    rec = StereoReconstructor(rig, MatchConfig(max_epipolar_px=2.5))
    raised = ""
    try:
        for i in range(20):
            rec.update(i / 30.0, None, [])
    except RuntimeError as exc:
        raised = str(exc)
    checks.append(("20 帧全没检出时抛的 RuntimeError 指向检测器",
                   "specular" in raised and "angled" not in raised))

    ok = True
    for label, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {label}")
        ok = ok and passed
    return ok


# --------------------------------------------------------------------------
def test_angled_self_tests() -> bool:
    """90° 链路的三个模块**自带**自测，但一直只在 `__main__` 里跑。

    `stereo/angled.py` / `stereo/angled_geometry.py` / `calib/angled_quality.py`
    各有一个 `_self_test()`：造 90° 装置（`R` = 绕 Y 转 90°、`T=(0.4,0,0)`）、
    把三个真值点投影到两路、故意塞一个"极线外的干扰点"、断言选对候选，
    且 3D 误差 < 1e-6 m、极线距离 < 1e-9 px、重投影残差 < 1e-6 px。

    它们写得比这里能补的任何东西都细（连 `distance_m`、`radius_left`、`t`
    是否原样拷贝都验），所以**不重写**，只把它们接进回归防线 ——
    在此之前 `run_tests.py` 从不执行它们，等于"验证存在、但没人跑"。

    边界要说清：这条只验**几何**（真值投影 -> 匹配 -> 三角测量）。
    "渲染 + 检测"那一环由本文件的 parallel 场景覆盖（检测器是同一套代码，
    装置不同不影响它），所以 angled 没有单独的渲染测试 —— 这是有意省略，
    不是忘了。真机 90° 一轮仍然是 `B-08`（要相机）。
    """
    print("----- 90° 三模块的自带自测（几何：真值投影 -> 匹配 -> 三角） -----")
    import contextlib
    import importlib
    import io

    ok = True
    for name in ("stereo.angled", "stereo.angled_geometry", "calib.angled_quality"):
        mod = importlib.import_module(name)
        buf = io.StringIO()
        try:
            # 吞掉它们自己的 PASS 行（每模块十几行），失败时再把尾部打出来
            with contextlib.redirect_stdout(buf):
                mod._self_test()                       # noqa: SLF001 —— 它就是这个模块的验证入口
        except Exception as exc:                       # noqa: BLE001
            print(f"  [不达标] {name}._self_test 抛出 {type(exc).__name__}: {exc}")
            for line in buf.getvalue().splitlines()[-3:]:
                print(f"           {line}")
            ok = False
            continue
        n = len([ln for ln in buf.getvalue().splitlines() if ln.strip()])
        print(f"  [通过] {name}._self_test（{n} 条断言全过）")
    return ok


def test_geometry_diagnostic() -> bool:
    """低配对率时必须由程序自己说出"录像和标定对不上"，而不是让人去调门限。

    现场数据（2026-10-30，`data/stereo/20261021-151144`）：实测极线偏差中位
    12.4 px，而该会话标定自报 `epipolar_rms` 0.56 px —— 差 22 倍。往
    `--max-epipolar` 上调能"跑通"，但配上的点残差 3.9~7.2 px、交会角 4.3°，
    本来就是错的：这是几何问题，代码只能把它**说清楚**，不能修。

    判据要的是方向性：坏几何必须出现"重新标定"、好几何**不许**出现；
    样本不足时宁可不说话。反向自证：把 `stereo/triangulate.py` 的
    `LOW_MATCH_RATE_REPORT` 从 0.10 改成 0.0（等于永不出诊断），第 1 条立刻红。
    """
    print("----- 低配对率诊断：极线偏差 vs 标定自报 -----")
    rig = build_rig()
    # build_rig() 的 quality 是 0.0（合成真值、没跑标定），这里换成该会话的真实值
    rig.quality = {"epipolar_rms": 0.56, "epipolar_max": 1.53}

    def make(median_dy: float, n: int = 60, matched: int = 2) -> StereoReconstructor:
        rec = StereoReconstructor(rig, MatchConfig())
        rec.n_frames = 60
        rec.n_matched = matched
        rec.dy_samples = [median_dy] * n
        return rec

    bad = "\n".join(make(12.4).low_match_diagnostic())
    # 低配对率 + 几何相符：这是"另找原因"的另一种组合（球被挡、检测丢失等）
    good = "\n".join(make(0.8).low_match_diagnostic())

    checks = [
        ("dy 中位 12.4 px（标定自报 0.56）-> 建议重新标定", "重新标定" in bad),
        ("同一句里要给出标定自报值作对照", "标定自报" in bad and "0.56" in bad),
        ("同一句里要有实测中位数", "12.4" in bad),
        ("dy 中位 0.8 px -> 不许说重新标定", "重新标定" not in good),
        ("dy 中位 0.8 px -> 明确说几何没问题", "与标定相符" in good),
        ("样本 < 20 帧 -> 一句话都不说（中位数会误导）",
         make(12.4, n=10).low_match_diagnostic() == []),
        ("配对率不低（58/60）-> 不插话",
         make(12.4, matched=58).low_match_diagnostic() == []),
    ]

    # 标定里没有 epipolar_rms 时：省略那半句，但不许编一个比值
    bare = build_rig()                       # quality 里 rms=0.0 -> 视为"没有自报值"
    rec_bare = StereoReconstructor(bare, MatchConfig())
    rec_bare.n_frames, rec_bare.n_matched = 60, 2
    rec_bare.dy_samples = [12.4] * 60
    text_bare = "\n".join(rec_bare.low_match_diagnostic())
    checks.append(("没有标定自报值 -> 不提『标定自报』", "标定自报" not in text_bare))
    checks.append(("没有标定自报值 -> 仍能凭绝对值判可疑", "重新标定" in text_bare))

    # 0 配对走的是异常那条路，诊断也必须带上
    rec0 = StereoReconstructor(rig, MatchConfig(max_epipolar_px=2.5))
    rec0.n_frames, rec0.n_matched, rec0.n_rejected = 60, 0, 60
    rec0.dy_samples = [12.4] * 60
    m0 = rec0._zero_match_message()                            # noqa: SLF001
    checks.append(("0 配对的报错里也要带几何诊断",
                   "左右圆心的 y 差实测" in m0 and "重新标定" in m0))

    ok = True
    for label, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {label}")
        ok = ok and passed
    return ok


def test_low_rate_advice() -> bool:
    """运行期那条"配对率极低"的告警，不许再无条件把人引向 angled。

    现场（2026-10-30）：第一行日志是
    「…若装置是大角度，请用 --stereo-mode angled 重跑…」，而同一段数据
    交会角只有 3.6~4.8°（< angled 门限 5.0°）、残差 3.9~7.2 px（> 3.0 px），
    照做 100% 被拒；更糟的是它和结束时的 [诊断]（"几何与标定不符"）直接矛盾。

    所以：branches 用语 `dominant_cause()`（与 0 配对报错同源），
    angled 只在"parallel 且标称夹角 > 2°"时才提，末尾一律指向 [诊断]。
    """
    print("----- 运行期低配对率告警：不许无条件引导 angled -----")
    rig = build_rig()                       # 平行装置，夹角约 0.9°
    rig_angled = build_rig()
    rig_angled.R = cv2.Rodrigues(np.radians([0.0, 90.0, 0.0]))[0]

    def make(r, **counts) -> "StereoReconstructor":
        rec = StereoReconstructor(r, MatchConfig(max_epipolar_px=8.0))
        for k, v in counts.items():
            setattr(rec, k, v)
        return rec

    # 低配对率 + 有候选但被拒（本项目的真实现场：1/30）
    flat = make(rig, n_frames=30, n_matched=1, n_rejected=29).      \
        _low_rate_advice()                                          # noqa: SLF001
    wide = make(rig_angled, n_frames=30, n_matched=1, n_rejected=29). \
        _low_rate_advice()                                          # noqa: SLF001
    # 低配对率 + 主要是没检出
    miss = make(rig, n_frames=30, n_matched=1, n_no_left=29, n_rejected=0). \
        _low_rate_advice()                                          # noqa: SLF001
    # 低配对率 + 混合（有帧缺、也有帧被拒）：不许谎称"左右都检出了"
    mixed = make(rig, n_frames=30, n_matched=1, n_no_left=7, n_rejected=22). \
        _low_rate_advice()                                          # noqa: SLF001

    checks = [
        ("夹角 0.9° 的平行装置 -> 告警里不许出现 angled", "angled" not in flat),
        ("同时要保留『检查标定/左右是否接反』", "标定" in flat and "接反" in flat),
        ("90° 装置 -> 才允许提 angled", "angled" in wide),
        ("主要是没检出 -> 说检测器没给出圆心", "检测器没给出圆心" in miss),
        ("主要是没检出 -> 指向种子", "specular" in miss and "seeds.json" in miss),
        ("混合型 -> 不许说『左右都检出了』", "左右都检出了" not in mixed),
        ("混合型 -> 仍指出先补检测", "specular" in mixed),
        ("四种都指向结束时的 [诊断]（统一括号收尾）",
         all("（详见结束时的 [诊断]）" in m for m in (flat, wide, miss, mixed))),
        ("都带上门限，便于对照",
         all("max_epipolar=8px" in m for m in (flat, wide, miss, mixed))),
    ]

    ok = True
    for label, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {label}")
        ok = ok and passed
    return ok


# --------------------------------------------------------------------------
def test_refine_center_and_geometry_guard() -> bool:
    """S2a/S2c：双目圆心取**轮廓**（不是高光斑），且几何对不上要早停。

    三组都带真值、都不依赖 CSRT，所以是确定性的（可反复跑出同一结果）。

    ① **证伪"用高光斑当跨视图对应点"**（`B-132`）：同一颗球，左高光偏 12 px、
       右高光偏 6 px（实测左右高光错位约 6 px、半径差 40%）。用**轮廓圆心**
       配对，三角测量误差 0.000 mm；用**高光**配对，误差 487 mm —— 数字"算得
       出来、看着正常"，实际整条轨迹整体歪掉。这就是 S2a 必须存在的原因。
       （`B-130` 已单独证伪"从边缘量镜面球半径"：整圈只有 25%~35% 有梯度。）
    ② **证伪"门限放宽就能救"**（`B-131`）：判读必须随 dy 单调分档，且
       "比值够大"与"绝对值够大"**两个条件都要**——只满足一个不算可疑。
       否则把 `--max-epipolar` 调大就能把一条错轨迹跑成"看起来正常"。
    ③ **证伪"CSRT 框中心就是球心"**：合成球图（球 200 / 背景 50 / 偏心高光 255），
       框中心被高光带偏 3.6 px。细化必须把它拉回真圆心（< 1 px），且**不许**
       被吸到高光上。反向自证：把 `refine_center` 关掉，那 3.6 px 就留在出口。
    ④ **守卫的归因**：圆心没钉到轮廓上（`center_source=box`）时的 y 差 ≈6 px 与
       "几何坏掉"长得一样，但修法相反。得到这个占比就**必须先说圆心口径**；
       占比 0 / 未报告 / 恰好 50% 都不能出现这条提示（不误伤真几何问题）。
    """
    print("----- 圆心的出处（轮廓 vs 高光）+ 几何判读的判据 -----")
    from sphere_tracker.config import DetectorConfig
    from sphere_tracker.detectors.specular import SpecularBallDetector

    checks: list[tuple[str, bool]] = []

    # ---- ① 高光当对应点 -> 好看但无意义的三角测量 ----
    rig = build_rig()
    P = np.array([0.05, -0.02, 3.0])
    rv, _ = cv2.Rodrigues(R_REL)
    ul, vl = cv2.projectPoints(P.reshape(1, 3), np.zeros((3, 1)), np.zeros(3),
                               K1, D1)[0].reshape(-1, 2)[0]
    ur, vr = cv2.projectPoints(P.reshape(1, 3), rv, T_REL, K2, D2)[0].reshape(-1, 2)[0]
    xyz_c = rig.triangulate(np.array([[ul, vl]]), np.array([[ur, vr]]))[0]
    err_contour_mm = float(np.linalg.norm(xyz_c - rig.R1 @ P)) * 1000
    xyz_h = rig.triangulate(np.array([[ul + 12.0, vl + 2.0]]),
                            np.array([[ur + 6.0, vr + 1.0]]))[0]
    err_hl_mm = float(np.linalg.norm(xyz_h - rig.R1 @ P)) * 1000
    checks += [
        (f"① 轮廓圆心配对 -> 误差 {err_contour_mm:.4f} mm（< 1 mm）",
         err_contour_mm < 1.0),
        (f"① 高光配对 -> 误差 {err_hl_mm:.0f} mm（> 100 mm，是『1 m 处 ≤10 mm』"
         "验收线的 10 倍以上，却照样算得出数）", err_hl_mm > 100.0),
        (f"① 同一装置上：轮廓 {err_contour_mm:.4f} mm vs 高光 {err_hl_mm:.0f} mm ——"
         "圆心出处错了，几何再准也没用", err_hl_mm > 100.0 * err_contour_mm),
    ]

    # ---- ② 几何判读：单调分档 + 两个条件都要 ----
    q = {"epipolar_rms": 0.56, "epipolar_max": 1.53}   # 该会话标定自报（B-131）
    v = {d: geometry_verdict([d] * 60, q) for d in (1.0, 2.9, 3.0, 5.0, 12.4)}
    checks += [
        ("② dy 1.0 px（比标定自报 1.8 倍）-> 不可疑", not v[1.0].suspect),
        ("② dy 2.9 px：比值 5.2 已超 4，但中位 < 3 px -> **仍不可疑**（两个都要）",
         not v[2.9].suspect),
        ("② dy 3.0 px：两个条件同时满足 -> 可疑", v[3.0].suspect),
        ("② dy 5.0 / 12.4 px -> 可疑（分档单调）", v[5.0].suspect and v[12.4].suspect),
        ("② 判读值随 dy 单调不减",
         v[1.0].med <= v[2.9].med <= v[3.0].med <= v[5.0].med <= v[12.4].med),
        ("② 样本 10 帧（< 20）-> 返回 None，宁可不说话",
         geometry_verdict([12.4] * 10, q) is None),
        ("② 标定没自报值（q={}）时按绝对值判：dy 4 px 可疑 / 2 px 不可疑",
         geometry_verdict([4.0] * 60, {}).suspect
         and not geometry_verdict([2.0] * 60, {}).suspect),
    ]
    dl = Detection(x=float(ul), y=float(vl), radius=10)
    near = Detection(x=float(ur), y=float(vr), radius=10)
    far = Detection(x=float(ur), y=float(vr) + 40.0, radius=10)
    checks += [
        ("② epipolar_dy 取候选里**最小**的 v 差（不是『配对成功那个』）",
         epipolar_dy(rig, dl, [far, near]) < 1.0 < epipolar_dy(rig, dl, [far])),
        ("② epipolar_dy 缺一边 -> None（不编数）",
         epipolar_dy(rig, None, [near]) is None and epipolar_dy(rig, dl, []) is None),
    ]

    # ---- ③ 框中心被高光带偏：细化必须拉回轮廓 ----
    def sphere_gray(cx, cy, r, hl_dx, hl_dy, hl_r=7, size=400):
        """合成球图：球 200 / 背景 50 / 偏心高光 255（硬边 + 一点离焦）。"""
        img = np.full((size, size), 50, np.uint8)
        cv2.circle(img, (cx, cy), r, 200, -1)
        cv2.circle(img, (cx + hl_dx, cy + hl_dy), hl_r, 255, -1)
        return cv2.GaussianBlur(img, (3, 3), 0.8)

    gray = sphere_gray(200, 200, 25, 10, 8)
    det = SpecularBallDetector(DetectorConfig())
    bx, by = 203.0, 202.0                      # 框中心被高光带偏，半径还是对的
    err_box = math.hypot(bx - 200.0, by - 200.0)
    nx, ny, src, shift = det._refine_center(gray, bx, by, 25.0)      # noqa: SLF001
    err_ref = math.hypot(nx - 200.0, ny - 200.0)
    to_hl = math.hypot(nx - 210.0, ny - 208.0)
    checks += [
        (f"③ 细化后圆心误差 {err_ref:.3f} px < 1 px（框中心本身差 {err_box:.2f} px）",
         err_ref < 1.0),
        (f"③ 来源记为 contour、位移 {shift:.2f} px ≈ 框的偏差",
         src == "contour" and abs(shift - err_box) < 0.5),
        (f"③ 没被高光吸走：到高光中心 {to_hl:.1f} px 远大于到真圆心 {err_ref:.3f} px",
         to_hl > 5.0 * max(err_ref, 1e-9)),
    ]
    # 反向自证：关掉细化 -> 框中心那 3.6 px 偏差原样留在出口
    det.cfg.specular.refine_center = False
    ex, ey, esrc, eshift = det._refine_center(gray, bx, by, 25.0)    # noqa: SLF001
    checks.append(("③ 关掉 refine_center -> 原样返回框中心、来源 box（偏差留在出口）",
                   esrc == "box" and eshift == 0.0 and (ex, ey) == (bx, by)))
    # 只有高光斑、没有球轮廓：搜索窗围绕给定半径(25)，够不到 7 px 的高光边
    # -> 找不到弧、原样退回（不会把高光本身当成球）
    hl_only = np.full((400, 400), 50, np.uint8)
    cv2.circle(hl_only, (200, 200), 7, 255, -1)
    ox, oy, _r_o, ook = refine_circle_subpixel_checked(hl_only, 203.0, 202.0, 25.0, search=8.0)
    checks.append(("③ 只有高光斑（半径 7）时不许把高光当球：ok=False、原样返回",
                   ook is False and (ox, oy) == (203.0, 202.0)))
    flat = np.full((100, 100), 128, np.uint8)
    fok = refine_circle_subpixel_checked(flat, 50.0, 50.0, 20.0, search=5.0)
    checks.append(("③ 全平图 -> ok=False；且非 _checked 版的『原样返回』约定成立",
                   fok == (50.0, 50.0, 20.0, False)
                   and refine_circle_subpixel(flat, 50.0, 50.0, 20.0, search=5.0)
                   == (50.0, 50.0, 20.0)))

    # ---- ④ 守卫的**归因**：圆心没钉到轮廓上时，不许把锅甩给"标定" ----
    # `B-132`：左右高光在球面上落在不同位置，圆心退回框中心时 y 差本就错开 ≈6 px，
    # **与"几何坏掉"长得一样**，但修法相反。程序得自己分清，别把用户引去重标定。
    rec = StereoReconstructor(rig, MatchConfig(max_epipolar_px=2.5))
    rec.dy_samples = [12.4] * 60              # 与 `B-131` 实测同量级
    m_box = rec.geometry_failure_message(center_box_share=1.0)
    m_ok = rec.geometry_failure_message(center_box_share=0.0)
    m_mid = rec.geometry_failure_message(center_box_share=0.5)
    m_none = rec.geometry_failure_message()
    checks += [
        ("④ 多数帧没钉到轮廓（box）-> 文案先指向**圆心口径**，不是重标定",
         "没钉到轮廓上" in m_box and "center_source=box" in m_box),
        ("④ 但『第 1/2/3 步』仍在（加了提示不等于少了指导）",
         all(f"第 {i} 步" in m_box for i in (1, 2, 3))),
        ("④ 圆心正常（box 占比 0）-> **不出现**这条提示（不误伤真几何问题）",
         "没钉到轮廓上" not in m_ok),
        ("④ 边界：恰好 50% 不提（门槛是『多数帧』，> 0.5）",
         "没钉到轮廓上" not in m_mid),
        ("④ 调用方没报占比（None）-> 保持原样，不瞎猜",
         "没钉到轮廓上" not in m_none and "第 1 步" in m_none),
    ]

    ok = True
    for label, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {label}")
        ok = ok and passed
    return ok


def test_estimate_radius() -> bool:
    """B-171：半径从图上估，但**不许**被更强的内部反射带跑。

    为什么这组判据必须存在
    ----------------------
    半径是 `Z = f·D/(2r)` 的分母，偏 10% 就是整条轨迹系统偏 10%。可是镜面球
    有一个反直觉的性质：**内部反射的径向梯度比外轮廓还强**（实测真球处最强峰
    落在 0.75~0.80 r0）。任何"取全区间最大梯度/最大分"的实现都会锁到反射上，
    估出的半径系统性偏小 20%。

    所以这里钉两条：
      ① 30% 弧 + 偏心高光的球，已知半径 40 -> 40±3；
      ② 再画一圈**比轮廓亮一倍**、覆盖 38% 的内部反射环 -> 结果**仍然** 40±3。
         这一条是"取更靠外的局部峰"的唯一证据：把选择规则改回"直接取最大分"，
         它必须变红（实测会掉到 33）。
      ③ 纯噪声 -> score < 0.2（调用方据此不采信）。
      ④ r0 故意给偏（50 vs 真值 40）也要能拉回来 —— 证明不是贴着输入走。
    """
    print("----- 半径从图上估（内部反射更强时也不许偏）-----")

    def sphere(r: float = 40.0, size: int = 200, arc_frac: float = 0.30,
               ring: float = 0.0, ring_frac: float = 0.38,
               ring_value: int = 255, ball_value: int = 120
               ) -> tuple[np.ndarray, int, int]:
        """合成一颗"只有 30% 弧可见"的球；``ring>0`` 时再加一圈内部反射。"""
        img = np.full((size, size), 30, np.float32)
        cx = cy = size // 2
        disk = np.zeros((size, size), np.uint8)
        cv2.circle(disk, (cx, cy), int(r), 255, -1)
        sector = np.zeros((size, size), np.uint8)
        cv2.ellipse(sector, (cx, cy), (int(r) + 2, int(r) + 2), 0,
                    200, 200 + 360 * arc_frac, 255, -1)
        img[(disk > 0) & (sector > 0)] = ball_value
        cv2.circle(img, (cx + 9, cy - 7), 5, 255, -1)      # 偏离球心的高光
        if ring > 0:
            # 内部反射：0.80r，亮度 255（轮廓只有 120，**梯度确实更强**），
            # 覆盖 38% 的角 —— 做得比轮廓的 30% 略宽，正是"按分数会赢"的形态。
            ring_mask = np.zeros((size, size), np.uint8)
            cv2.ellipse(ring_mask, (cx, cy), (int(ring * r), int(ring * r)),
                        0, 10, 10 + 360 * ring_frac, 255, 1)
            img[ring_mask > 0] = ring_value
        img = np.clip(img, 0, 255).astype(np.uint8)
        return cv2.GaussianBlur(img, (3, 3), 0.8), cx, cy

    checks: list[tuple[str, bool]] = []

    gray, cx, cy = sphere()
    r_plain, s_plain = estimate_radius_px(gray, cx, cy, 40.0)
    checks.append((f"① 30% 弧 + 偏心高光：r={r_plain:.2f} px（真值 40±3），"
                   f"score={s_plain:.2f}", abs(r_plain - 40.0) <= 3.0 and s_plain >= 0.2))

    gray_ring, cx, cy = sphere(ring=0.75)
    r_ring, s_ring = estimate_radius_px(gray_ring, cx, cy, 40.0)
    checks.append((f"② 内部反射环（0.80r、亮度 2 倍、38% 弧）仍在 "
                   f"r={r_ring:.2f} px（真值 40±3）",
                   abs(r_ring - 40.0) <= 3.0))
    checks.append((f"② 同图 score={s_ring:.2f}（≥0.2，调用方敢用）", s_ring >= 0.2))

    r_far, s_far = estimate_radius_px(gray_ring, cx, cy, 50.0)
    checks.append((f"④ 输入 r0=50（故意偏 25%）仍估到 {r_far:.2f} px —— "
                   "不是贴着输入走", abs(r_far - 40.0) <= 3.0))

    rng = np.random.default_rng(3)
    noise_img = np.clip(rng.normal(120.0, 30.0, (200, 200)), 0, 255).astype(np.uint8)
    r_noise, s_noise = estimate_radius_px(noise_img, 100, 100, 40.0)
    checks.append((f"③ 纯噪声：score={s_noise:.3f} < 0.2，原样返回 r0="
                   f"{r_noise:.1f}（不猜）", s_noise < 0.2 and r_noise == 40.0))

    flat = np.full((120, 120), 128, np.uint8)
    _, s_flat = estimate_radius_px(flat, 60, 60, 20.0)
    checks.append((f"③ 全平图：score={s_flat:.3f}（没有边就没有分）", s_flat == 0.0))

    ok = True
    for label, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {label}")
        ok = ok and passed
    return ok


def test_guard_step() -> bool:
    """几何守卫的一步判定抽成纯函数后，语义不许变（`B-131` 的验收缺口②）。

    原来这段判定内联在 `run_stereo_video.main()` 的循环里，只有真跑一段坏录像
    （跑到 `--geometry-guard N` 对）才能碰到，所以它一直不被任何测试覆盖 ——
    这正是独立 Critic 指出的验收缺口。抽成 `_guard_step` 后可以只喂输入、看输出。

    这里钉的是**状态机语义**，尤其是那个最容易被写错、又最难在真机上发现的点：
    "跑够了 N 对，但两路同时检出的样本还不够 `GEOM_MIN_SAMPLES`" 时，必须返回
    `(True, None)` —— **继续等、不消耗这次机会**。若写成 `(False, None)`（原来的
    `n_pairs == guard_at` 只判一次就是这个效果），丢帧多的录像会**一次也判不成、
    然后永久放弃**：几何明明坏掉，守卫却静默跳过，用户以为"守过了"。
    """
    print("----- 几何守卫的一步判定（纯函数 _guard_step）-----")
    from apps.run_stereo_video import _guard_step
    from stereo.triangulate import GEOM_MIN_SAMPLES

    q = {"epipolar_rms": 0.56, "epipolar_max": 1.53}   # 该会话标定自报（B-131）

    checks: list[tuple[str, bool]] = []
    # ① 没武装 -> 保持原状、不判读
    checks.append(("① armed=False -> (False, None)",
                   _guard_step(False, 30, 100, [12.0] * 30, q) == (False, None)))
    # ② 还没跑够 N 对 -> 继续等，不消耗机会
    armed, v = _guard_step(True, 30, 10, None, q)
    checks.append(("② n_pairs < guard_at -> (True, None)", armed and v is None))
    # ③ 跑够了，但样本不足 -> (True, None)：不消耗这次机会（关键判据）
    armed, v = _guard_step(True, 30, 30, [0.3] * 5, q)
    checks.append((f"③ 样本只有 5 个（< GEOM_MIN_SAMPLES={GEOM_MIN_SAMPLES}）"
                   "-> (True, None)，下次接着看", armed and v is None))
    # ④ 样本够且一致（与标定自报相符）-> (False, v) 且不可疑
    armed, v = _guard_step(True, 30, 40, [0.3] * 30, q)
    checks.append(("④ 30 个 ≈0.3 px（与标定 0.56 相符）-> (False, 不可疑)",
                   (not armed) and v is not None and not v.suspect))
    # ⑤ 样本够但明显不一致 -> (False, v) 且可疑
    armed, v = _guard_step(True, 30, 40, [12.0] * 30, q)
    checks.append(("⑤ 30 个 ≈12 px（远超标定自报）-> (False, 可疑)",
                   (not armed) and v is not None and v.suspect))

    ok = True
    for label, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {label}")
        ok = ok and passed
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="双目链路端到端自测（合成真值）")
    ap.add_argument("--scenario", default="all", choices=["clean", "realistic", "all"])
    ap.add_argument("--write-video", action="store_true",
                    help="导出两路并排视频（带标注）便于肉眼检查")
    args = ap.parse_args()

    print("=" * 70)
    print("双目链路端到端自测（合成真值）")
    print("=" * 70)
    print("两路真实渲染 -> 各自 2D 检测/跟踪 -> 极线配对 -> 三角测量 -> 比真值")
    print()

    sign_ok = test_baseline_sign()
    print()

    angled_ok = test_angled_self_tests()
    print()

    seed_ok = test_seed_auto_detector()
    print()

    msg_ok = test_zero_match_message()
    print()

    geom_ok = test_geometry_diagnostic()
    print()

    advice_ok = test_low_rate_advice()
    print()

    center_ok = test_refine_center_and_geometry_guard()
    print()

    radius_ok = test_estimate_radius()
    print()

    guard_ok = test_guard_step()
    print()

    scenarios = ["clean", "realistic"] if args.scenario == "all" else [args.scenario]

    # 门槛怎么定的（不是凑出来的，按 εZ = Z²/(f·B) 反推）：
    #
    # 本装置 f≈889 px、B=0.150 m -> 1 px 视差误差 = 13.4 mm 深度误差 @ 4.2 m。
    # 实测两路检测/跟踪残差约 0.5–1.0 px（单目自测同量级），
    # 于是 4.2 m 处的三维 RMSE 落在 15–35 mm 是**这条定律的必然结果**，不是实现差。
    #
    # 项目验收口径是"1 m 处 ≤ 10 mm"，所以这里同时打印折算值——
    # 换算到 1 m 后 clean ≈ 0.8 mm，比验收线宽十几倍。
    # 绝对阈值只用来防"实现退化"（回归防线），并随场景放宽。
    limits = {
        "match_rate": (0.99, 0.98),        # 两路都检出却配不上，只能是真 bug
        "rmse_at_1m_mm": (2.0, 3.0),       # 折算到 1 m，直接对齐验收口径
        "err3d_rmse_mm": (20.0, 40.0),     # 4 m 处的绝对量级
        "err3d_max_mm": (60.0, 110.0),     # 最大值受个别模糊帧影响
        "det_l_rate": (0.99, 0.98),
        "det_r_rate": (0.99, 0.98),
    }
    labels = {
        "match_rate": "配对率", "rmse_at_1m_mm": "折算1m三维RMSE",
        "err3d_rmse_mm": "三维RMSE", "err3d_max_mm": "三维最大误差",
        "det_l_rate": "左路检出率", "det_r_rate": "右路检出率",
    }

    ok = True
    results = []
    for scenario in scenarios:
        print(f"===== 场景：{scenario} "
              f"({'无噪声无模糊' if scenario == 'clean' else '噪声 + 运动模糊 + 干扰物'}) =====")
        r = run_scenario(scenario, write_video=args.write_video)
        results.append(r)
        print()

        idx = 0 if scenario == "clean" else 1
        bad = []
        for key, lim in limits.items():
            v = r[key]
            if math.isnan(v):
                bad.append(f"{labels[key]}:无样本")
            elif key.endswith("_rate"):
                if v < lim[idx]:
                    bad.append(f"{labels[key]} {v * 100:.1f}% < {lim[idx] * 100:.0f}%")
            elif v > lim[idx]:
                bad.append(f"{labels[key]} {v:.2f} > {lim[idx]}")
        if bad:
            ok = False
            print(f"  [不达标] " + "; ".join(bad))
        else:
            print(f"  [通过]")

        # 同步的价值：滞后 1 帧就该明显变差
        lag1 = dict((lag, mean) for lag, mean, _rmse in r["sync_rows"]).get(1, float("nan"))
        gain = lag1 / max(r["err3d_mean_mm"], 1e-9)
        print(f"  [参考] 滞后 1 帧让三维均值误差变成 {gain:.1f} 倍"
              f"（{'>' if gain > 1.5 else '≤'} 1.5，说明同步{'确实必要' if gain > 1.5 else '影响有限'}）")
        print()

    all_ok = (ok and sign_ok and angled_ok and seed_ok and msg_ok
              and geom_ok and advice_ok and center_ok and radius_ok and guard_ok)
    print("判定结果：" + ("全部通过" if all_ok else "有不达标项"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
