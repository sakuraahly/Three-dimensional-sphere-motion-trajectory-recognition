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
  7. 质量结论**两个产物各带一份**（`traj3d.json` 与 `visualization.json` 的
     `quality` 逐键一致，verdict 与守卫判定一致）——2026-11-15 补。GUI / 3D
     查看器只读 visualization.json，少了它用户就看不到「仅供参考」。
  8. 重导出（`apps/export_visualization.py`）**不许冲掉** `quality` /
     `predictions` / `smoothing` / `world_frame`：导出是重建，这几个键是
     `run_stereo_video.py` 事后补的。轨迹没变就沿用（并写 `carried_over`
     标记），轨迹变了就丢 —— 2026-11-15 补。
  9. **往返一致性** `round_trip`（`stereo/round_trip.py`）：去程/回程在同一个 X
     处的 (Y,Z) 偏移，四档结论 `retrace` / `ellipse` / `scattered` / `no_pairs`。
     它是**轨迹的几何属性**，每次导出重算，不参与 `carried_over` —— 2026-11-15 补。
 10. **装置级 `fov_apply`**（`stereo/fov_policy.py`）：「启用」= 应用到当前摄像头，
     标定与录制都服从；九个判据 + 反向自证（删掉第 2 条 -> ②⑧ 变红，字节级
     还原后 sha256 一致）—— 2026-11-15 补。
 11. **双目实时入口**（`apps/run_realtime_stereo.py`）：`skew_correct` /
     `should_drop` / `compose_hud` / `compose_canvas` 四个纯函数 + 注入假帧源的
     端到端（60 对、8 px/帧）。核心是 P3 那条：**同一组数据注入一帧 skew，
     补偿后 RMS 必须显著小于补偿前**（实测 665 -> 32 mm @3.6 m），
     并把加速段（5 m/s²）的残差如实量出来 —— 2026-11-16 补。

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
from stereo.angled_geometry import (  # B4 refined F 的合成真值与量测
    epipolar_line, fundamental_from_pose, point_line_distance,
)

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


def test_guard_classification() -> bool:
    """几何守卫把「检出质量差」与「几何不一致」分开报（2026-11-13 事故）。

    钉住四件事：
      1. 真几何不一致（dy 大、gates 全 0、两路检出率 100%）-> geometry_mismatch，
         原文案仍含「重新标定」、``geometry_verdict(...).suspect is True``（判定式没被改）；
      2. 检测质量差（形状门 / 分数门拒了大量候选、检出率低）-> detection_quality，
         新文案**不含**「重新标定」、含形状门 / 预设方向；
      3. 保守边界：只有少量拒绝、检出率 100%、或只有 3/4 弱证据 -> 仍
         geometry_mismatch（不许把真几何问题降级）；
      4. 不可疑 -> ok；center_box_share>0.5 与 biggest 命中门占比>0.5 是强证据。

    反向自证：把 ``stereo/triangulate.classify_guard`` 短路成永远返回
    ``"detection_quality"``，第 1 条立刻红。
    """
    print("----- 几何守卫：检出质量差 / 几何不一致 分开报 -----")
    from stereo.triangulate import classify_guard

    rig = build_rig()
    q = {"epipolar_rms": 0.56, "epipolar_max": 1.53}

    def rec_with(dy: float, n: int, matched: int) -> StereoReconstructor:
        r = StereoReconstructor(rig, MatchConfig())
        r.n_frames, r.n_matched, r.dy_samples = 60, matched, [dy] * n
        return r

    checks: list[tuple[str, bool]] = []

    # ---- 1) 真几何不一致 ----
    v_bad = geometry_verdict([12.4] * 60, q)
    sig_bad = {"gates": {"radius": 0, "circ": 0, "fill": 0, "radial": 0, "score": 0},
               "candidates": 120, "biggest_gate_share": 0.0,
               "center_box_share": None, "both_detected_rate": 1.0,
               "seed_anchor_share": 0.0}
    m_bad = rec_with(12.4, 60, 2).geometry_failure_message(v_bad)
    checks += [
        ("① 真几何不一致 -> geometry_mismatch",
         classify_guard(v_bad, sig_bad) == "geometry_mismatch"),
        ("① 原文案仍含「重新标定」与「第 1 步」",
         "重新标定" in m_bad and "第 1 步" in m_bad),
        ("① geometry_verdict(...).suspect is True（判定式没被改）",
         v_bad.suspect is True),
    ]

    # ---- 2) 检测质量差 ----
    v_qual = geometry_verdict([6.8] * 30, q)
    sig_qual = {"gates": {"radius": 0, "circ": 18, "fill": 9, "radial": 0, "score": 7},
                "both_detected_rate": 0.3}
    m_qual = rec_with(6.8, 30, 9).geometry_failure_message(
        v_qual, cause="detection_quality", sig=sig_qual)
    checks += [
        ("② 检测质量差 -> detection_quality",
         classify_guard(v_qual, sig_qual) == "detection_quality"),
        ("② 新文案不含「重新标定」", "重新标定" not in m_qual),
        ("② 新文案含形状门 / 预设方向",
         "形状门" in m_qual and ("预设" in m_qual or "specular" in m_qual)),
        ("② 新文案给出 gates 汇总与两路检出率",
         "circ=18" in m_qual and "30%" in m_qual),
    ]

    # ---- 3) 保守边界：不许把真几何问题降级 ----
    sig_edge = {"gates": {"radius": 0, "circ": 2, "fill": 0, "radial": 0, "score": 0},
                "candidates": 100, "both_detected_rate": 1.0}
    checks += [
        ("③ 少量拒绝（占比<0.5）+ 检出率 100% -> 仍 geometry_mismatch",
         classify_guard(v_qual, sig_edge) == "geometry_mismatch"),
        ("③ 不可疑 verdict -> ok",
         classify_guard(geometry_verdict([0.3] * 30, q), sig_bad) == "ok"),
        ("③ 3/4 弱证据（检出率低 / seed 锚点多）单独不许降级",
         classify_guard(v_bad, {"gates": {}, "both_detected_rate": 0.3,
                                "seed_anchor_share": 0.9}) == "geometry_mismatch"),
        ("③ center_box_share>0.5 是强证据 -> detection_quality",
         classify_guard(v_bad, {"gates": {}, "center_box_share": 0.8})
         == "detection_quality"),
        ("③ biggest 命中门占比>0.5 是强证据 -> detection_quality",
         classify_guard(v_bad, {"gates": {}, "biggest_gate_share": 0.7})
         == "detection_quality"),
    ]

    ok = True
    for label, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {label}")
        ok = ok and passed
    return ok


def test_guard_no_early_stop() -> bool:
    """端到端：**两套早停**都不中止，四种产物齐、quality 块正确、CSV 15 列。

    造最小会话（``tempfile.mkdtemp`` + ``finally: shutil.rmtree``），monkeypatch「输入侧」
    依赖，但让 ``main()`` 的**真实控制流**跑完：守卫块 / 零配对 -> 不中止 -> CSV 写盘 ->
    ``os.replace(tmp -> 正式)`` -> export -> ``payload["quality"]`` -> ``return 0``。

    两条路径各跑一次：
      A 几何守卫：mock 重建器让 n_matched>0 + 强制 suspect + 检测质量差信号
                  -> verdict=detection_quality（证明逐帧 gates 累加真接线）；
      B 零配对门禁：用**真实 StereoReconstructor**（入口 allow_zero_match=True 放行
                  门禁），40 帧全空 -> verdict=zero_match、return 0、产物齐、warnings 非空。

    反向自证（4 条）：
      - classify_guard 恒 detection_quality -> A 的 geometry_mismatch 判据红；
      - 守卫块放回 ``return 2`` -> A 产物完整性红；
      - 逐帧 gates 累加恒为空 -> A 的 detection_quality 强断言红；
      - 入口 ``allow_zero_match=True`` 改回 False -> B「不中止 + 产物齐」红。
    """
    print("----- 端到端：两套早停都不中止 + 产物齐 -----")
    import contextlib as _contextlib
    import csv as _csv
    import io as _io
    import json as _json
    import shutil as _shutil
    import sys as _sys
    import tempfile as _tempfile
    import types as _types
    from unittest import mock as _mock

    import apps.run_stereo_video as rsv
    from stereo.triangulate import geometry_verdict as _gv

    _frame = np.zeros((540, 960, 3), np.uint8)

    class _FakePairSource:
        image_size = (960, 540)

        def __init__(self, *a, **k):
            self.n = 0

        def read(self):
            if self.n >= 40:
                return (False, None, None)
            self.n += 1
            return (True, _frame, _frame)

        def report_warnings(self):
            pass

        def describe(self):
            return "fake-index"

        def release(self):
            pass

    class _FakeDetector:
        def debug_snapshot(self):
            # 形状门 / 分数门拒了大量候选 + 最大候选死在 circ：逐帧累加后
            # 能把 classify_guard 判成 detection_quality（证明信号真接线）。
            return {"gates": {"radius": 0, "circ": 2, "fill": 1,
                              "radial": 0, "score": 1},
                    "biggest": {"gate": "circ"}}

    class _FakePipeline:
        def __init__(self):
            self.detector = _FakeDetector()

        def process(self, frame, t):
            return _types.SimpleNamespace(track=None, detection=None)

    class _FakePoint:
        residual_px = 0.5

    class _FakeRec:
        def __init__(self, *a, **k):
            self.n_matched = 0
            self.n_frames = 0
            self.dy_samples = [6.8] * 30
            self.n_no_left = self.n_no_right = self.n_rejected = 0
            self.trajectory = [_FakePoint()]
            self.warmup_frames = 0
            self.warnings = []

        def update(self, t, det_l, det_r):
            self.n_matched += 1
            self.n_frames += 1
            return _types.SimpleNamespace(
                xyz=(0.1, 0.05, 3.0), u_left=(100.0, 200.0),
                u_right=(90.0, 200.0), epipolar_px=0.5,
                residual_px=0.5, confident=1)

        @property
        def match_rate(self):
            return 1.0

        def as_array(self):
            return np.array([[0.0, 0.1, 0.05, 3.0]], np.float64)

        def low_match_diagnostic(self):
            return []

        def geometry_failure_message(self, *a, **k):
            return "[占位] 检出质量差 / 几何不一致 分开报文案"

    class _FakeWriter:
        def __init__(self, path, fourcc, fps, size):
            self._path = Path(path)

        def isOpened(self):
            return True

        def write(self, frame):
            pass

        def release(self):
            # 模拟 moov atom 落盘：让 annotated.tmp.mp4 真实存在，
            # os.replace 才能把它改名成 annotated.mp4。
            self._path.write_bytes(b"fake-mp4")

    suspect_v = _gv([6.8] * 30, {"epipolar_rms": 0.56, "epipolar_max": 1.53})
    cam = _types.SimpleNamespace(video="left.mkv", fps_measured=30.0,
                                 fps_requested=30.0)
    sess = _types.SimpleNamespace(
        left=cam, right=cam, calibration=_types.SimpleNamespace(),
        stereo_mode="parallel", rig={},
        sync=_types.SimpleNamespace(offset=0, method="flash"))

    def _run(zero_match: bool):
        """跑一次 main()，返回 (rc, stdout, exc_text, out_dir, tmp)。

        zero_match=False -> A 几何守卫路径（mock 重建器 / 强制 suspect）；
        zero_match=True  -> B 零配对门禁路径（**真实** StereoReconstructor，
                            不 mock 重建器与 _guard_step，靠 allow_zero_match 放行）。
        """
        tmp = _tempfile.mkdtemp(prefix="dsh_guard_e2e_")
        session_dir = Path(tmp) / "session"
        out_dir = Path(tmp) / "out"
        session_dir.mkdir()
        (session_dir / "derived").mkdir()
        fake_traj = Path(tmp) / "exported_traj3d.json"
        fake_vis = Path(tmp) / "exported_visualization.json"
        fake_traj.write_text(_json.dumps({
            "schema": "sphere-stereo-trajectory", "unit": "m",
            "n_matched": 40, "samples": []}), encoding="utf-8")
        fake_vis.write_text(_json.dumps({
            "schema": "sphere-stereo-visualization", "n_points": 0,
            "bounds": {"min": [0, 0, 0], "max": [0, 0, 0]}}), encoding="utf-8")
        argv = ["run_stereo_video.py", str(session_dir),
                "--out", str(out_dir), "--geometry-guard", "30",
                "--pairing", "index", "--no-plot", "--no-predict",
                "--no-world", "--no-radius-auto", "--method", "color"]
        rig = build_rig()      # 完整合成 StereoRig，供真实重建器 / _resolve_max_epipolar
        base = [
            _mock.patch.object(rsv, "load_session", return_value=(session_dir, sess)),
            _mock.patch.object(rsv, "validate_session", return_value=([], [])),
            _mock.patch.object(rsv, "session_calibration_file",
                               return_value=session_dir / "calib.npz"),
            _mock.patch.object(rsv.StereoRig, "load", return_value=rig),
            _mock.patch.object(rsv, "resolve_auto", return_value="parallel"),
            _mock.patch.object(rsv, "_mode_guard", return_value=("parallel", False)),
            _mock.patch.object(rsv, "optical_axes_angle_deg", return_value=0.0),
            _mock.patch.object(rsv, "_pick_detector", return_value=("color", "")),
            _mock.patch.object(rsv, "PairSource", _FakePairSource),
            _mock.patch.object(rsv, "_make_pipeline",
                               side_effect=lambda cfg, r, side: _FakePipeline()),
            _mock.patch.object(rsv, "_draw", side_effect=lambda *a, **k: _frame),
            _mock.patch.object(rsv, "export_session_visualization",
                               return_value={"trajectory": fake_traj,
                                             "visualization": fake_vis}),
            _mock.patch.object(rsv.cv2, "VideoWriter", _FakeWriter),
        ]
        if zero_match:
            # B 路径：**不** mock StereoReconstructor / _guard_step —— 用真实重建器，
            # 40 帧全空，验证库层 allow_zero_match=True 放行门禁后仍跑完出数。
            patches = base
        else:
            # A 路径：mock 重建器让 n_matched>0、强制 suspect 触发几何守卫。
            patches = base + [
                _mock.patch.object(rsv, "StereoReconstructor", _FakeRec),
                _mock.patch.object(rsv, "_guard_step", return_value=(False, suspect_v)),
            ]
        rc, stdout, exc_text = None, "", ""
        buf = _io.StringIO()
        try:
            for p in patches:
                p.start()
            old_argv = _sys.argv
            _sys.argv = argv
            try:
                with _contextlib.redirect_stdout(buf):
                    rc = rsv.main()
            finally:
                _sys.argv = old_argv
            stdout = buf.getvalue()
        except Exception as exc:                                # noqa: BLE001
            import traceback as _tb
            exc_text = _tb.format_exc()
            print(f"  [不达标] main() 抛异常：{exc!r}")
            print(exc_text)
        finally:
            for p in reversed(patches):
                p.stop()
        return rc, stdout, exc_text, out_dir, tmp
    checks: list[tuple[str, bool]] = []
    for zero_match, want, tag in ((False, "detection_quality", "A几何守卫"),
                                  (True, "zero_match", "B零配对门禁")):
        rc, stdout, exc_text, out_dir, tmp = _run(zero_match)
        csv_path = out_dir / "traj3d.csv"
        traj_path = out_dir / "traj3d.json"
        vis_path = out_dir / "visualization.json"
        video_path = out_dir / "annotated.mp4"
        tmp_video = out_dir / "annotated.tmp.mp4"
        checks.append((f"[{tag}] main() 返回 0（实际 {rc}）",
                       rc == 0 and not exc_text))
        for name, p in (("traj3d.csv", csv_path), ("traj3d.json", traj_path),
                        ("visualization.json", vis_path), ("annotated.mp4", video_path)):
            checks.append((f"[{tag}] {name} 存在", p.exists()))
        checks.append((f"[{tag}] annotated.tmp.mp4 已被 os.replace 改名（不存在）",
                       not tmp_video.exists()))
        checks.append((f"[{tag}] stdout 有结构化标记行 [质量] verdict=...",
                       "[质量] verdict=" in stdout and "结果仅供参考" in stdout))
        try:
            payload = _json.loads(traj_path.read_text(encoding="utf-8"))
            q = payload.get("quality", {})
            checks.append((f"[{tag}] quality.verdict == {want}",
                           q.get("verdict") == want))
            checks.append((f"[{tag}] quality 含通用字段 grade/degraded/warnings",
                           all(k in q for k in ("grade", "degraded",
                                                "degraded_reason", "warnings",
                                                "dy_med_px", "gate_rejects", "note"))))
            checks.append((f"[{tag}] grade=degraded / degraded=True",
                           q.get("grade") == "degraded" and q.get("degraded") is True))
            if not zero_match:
                # A 强断言：检测质量差靠逐帧 gates 累加，不是靠 dy 猜。
                checks.append((f"[{tag}] gate_rejects 真汇总到逐帧计数（circ>0）",
                               (q.get("gate_rejects") or {}).get("circ", 0) > 0))
            else:
                # B：库层低配对率 warning 必须保留（不中止但也不静默）。
                checks.append((f"[{tag}] zero_match 的 warnings 非空（低配对告警保留）",
                               len(q.get("warnings") or []) > 0))
        except Exception as exc:                                # noqa: BLE001
            checks.append((f"[{tag}] 读 JSON quality：{exc}", False))
        # 2026-11-15：quality 必须**两个产物各带一份**。GUI / 3D 查看器只读
        # visualization.json，少了这个键，用户拿着"质量差"的结果却看不到标记。
        # 两份必须**同一个结论**（同一个 dict），否则"以哪个为准"又成了第二份真相。
        try:
            tq = _json.loads(traj_path.read_text(encoding="utf-8")).get("quality") or {}
            vq = _json.loads(vis_path.read_text(encoding="utf-8")).get("quality") or {}
            checks.append((f"[{tag}] visualization.json 也带 quality.verdict == {want}",
                           vq.get("verdict") == want))
            checks.append((f"[{tag}] traj3d.json 与 visualization.json 的 quality 逐键一致",
                           bool(vq) and vq == tq))
        except Exception as exc:                                # noqa: BLE001
            checks.append((f"[{tag}] 读 visualization.json 的 quality：{exc}", False))
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                rd = _csv.DictReader(f)
                header = rd.fieldnames
                rows = list(rd)
            checks.append((f"[{tag}] CSV 仍是 15 列（表头整串不变）",
                           header == list(rsv.CSV_HEADER)))
            checks.append((f"[{tag}] CSV DictReader 不错位",
                           all(None not in r and all(v is not None for v in r.values())
                               for r in rows)))
        except Exception as exc:                                # noqa: BLE001
            checks.append((f"[{tag}] 读 CSV：{exc}", False))
        _shutil.rmtree(tmp, ignore_errors=True)

    ok = True
    for label, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {label}")
        ok = ok and passed
    return ok


def test_export_carryover() -> bool:
    """重导出不许冲掉 `quality` / `predictions` / `smoothing` / `world_frame`。

    根因（2026-11-15 已核实）：`build_trajectory_payload()` /
    `build_visualization_payload()` 是**从头重建**，返回字典里根本没有 `quality`；
    这些字段一直是 `apps/run_stereo_video.py` 在导出**之后**补写的。而
    `apps/export_visualization.py` 以前只调 `export_session_visualization()`、
    之后什么都不补 —— 于是用户只要重导一次，「质量仅供参考」的标记就凭空消失，
    直接违背「质量差也要出结果，但质量状态必须记录并显示」。

    这里钉住 `carry_over_runtime_extras()` 的两条边界：

    - 轨迹**没变** -> 沿用上一版，并在 `carried_over` 里写明"不是本次算的"；
    - 轨迹**变了** -> 一律丢掉旧结论，`notes` 写明"已丢弃、请重跑重建"
      （宁可丢，也不许把旧结论挂到新轨迹上）。

    所有输入输出都在 `tempfile.mkdtemp()` 里，`finally: shutil.rmtree`
    —— 绝不碰 `data/`（用户真实录制数据）。
    """
    print("----- 重导出：轨迹没变沿用结论 / 轨迹变了必须丢 -----")
    import json as _json
    import os as _os
    import shutil as _shutil
    import subprocess as _subprocess
    import tempfile as _tempfile

    from apps.export_visualization import _trajectory_changed
    from stereo.visualization import carry_over_runtime_extras

    script = ROOT / "apps" / "export_visualization.py"
    created_old = "2026-01-01 00:00:00"
    header = ("frame,t,matched,X,Y,Z,u_left,v_left,u_right,v_right,"
              "epipolar_px,residual_px,confident,method_left,method_right")

    def _write_csv(path: Path, n_rows: int) -> None:
        lines = [header]
        for i in range(1, n_rows + 1):
            lines.append(f"{i},{i * 0.02:.3f},1,{0.10 + i * 0.01:.4f},0.05,"
                         f"{2.0 + i * 0.005:.4f},480,270,470,270,0.3,0.4,1,"
                         f"color,color")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _make_session(root: Path, n_rows: int = 5) -> Path:
        session = root / "session"
        (session / "derived").mkdir(parents=True)
        (session / "session.json").write_text(_json.dumps({
            "id": "carryover_probe",
            "left": {"width": 960, "height": 540},
            "right": {"width": 960, "height": 540},
            "stereo_mode": "angled",
            "calibration": {"baseline_mm": 150.0, "created": created_old,
                            "verdict": "ok"},
            "sync": {"offset": 0, "method": "flash"},
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_csv(session / "derived" / "traj3d.csv", n_rows)
        return session

    def _read(path: Path) -> dict:
        return _json.loads(path.read_text(encoding="utf-8"))

    def _run_cli(session: Path) -> tuple[int, str]:
        env = dict(_os.environ, PYTHONIOENCODING="utf-8")
        proc = _subprocess.run([sys.executable, str(script), str(session)],
                               cwd=str(ROOT), capture_output=True, text=True,
                               encoding="utf-8", errors="replace", env=env,
                               timeout=180)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")

    quality_old = {"verdict": "detection_quality", "grade": "degraded",
                   "degraded": True, "degraded_reason": "检测质量差",
                   "warnings": ["逐帧 gates 拒绝较多"], "note": "上一版算的"}
    # 旧 JSON 里**故意不写 bounds**：这台机器上有没有 config/world_frame.json
    # 会改变新导出的坐标，拿它比 bounds 就成了环境断言。bounds 判据直接测
    # `_trajectory_changed()`（纯函数），见后面那三条。
    traj_old = {
        "schema": "sphere-stereo-trajectory", "version": 1,
        "session": "carryover_probe", "created": created_old,
        "unit": "m", "n_samples": 5, "n_matched": 5,
        # `coordinate_system` 与 `world_frame` 必须**成对**出现，否则不该搬
        "coordinate_system": "world",
        "quality": quality_old,
        "predictions": {"t_ref": 1.5, "state": [0.1, 0.05, 2.0, 0.0],
                        "points": [[1.8, 0.11, 0.05, 2.02]]},
        "smoothing": {"method": "savgol", "window": 7, "points": []},
        "world_frame": {"source": "board", "note": "fixture",
                        "created_at": created_old},
        "notes": ["上一版轨迹说明"],
    }
    vis_old = {
        "schema": "sphere-stereo-visualization", "version": 1,
        "session": "carryover_probe", "created": created_old,
        "n_points": 5,
        "coordinate_system": "world",
        "world_frame": {"source": "board", "note": "fixture",
                        "created_at": created_old},
        "quality": quality_old,
        # 世界系跑出来的 vis 就是这么写的：一句"已应用世界系"
        "notes": ["已应用世界系：config/world_frame.json", "上一版可视化说明"],
    }
    carry_keys = ("quality", "predictions", "smoothing", "world_frame")

    checks: list[tuple[str, bool]] = []
    tmp = _tempfile.mkdtemp(prefix="export_carry_")
    try:
        session = _make_session(Path(tmp))
        derived = session / "derived"
        (derived / "traj3d.json").write_text(
            _json.dumps(traj_old, ensure_ascii=False, indent=2), encoding="utf-8")
        (derived / "visualization.json").write_text(
            _json.dumps(vis_old, ensure_ascii=False, indent=2), encoding="utf-8")

        # ---- 判据 1/2/4/6：轨迹没变 -> 沿用，且标记显眼（走真 CLI 子进程） ----
        rc, out = _run_cli(session)
        checks.append((f"①⑥ 轨迹未变：CLI 退出码 0（实际 {rc}）", rc == 0))
        checks.append(("① stdout 明说「沿用上一版」（[保留]）", "[保留]" in out))

        traj_new = _read(derived / "traj3d.json")
        carried = traj_new.get("carried_over") or {}
        keys = list(carried.get("keys") or [])
        checks.append(("① quality / predictions / smoothing / world_frame 四项都还在",
                       all(k in traj_new for k in carry_keys)))
        checks.append(("① quality.verdict 仍是 detection_quality（没被重建冲掉）",
                       (traj_new.get("quality") or {}).get("verdict")
                       == "detection_quality"))
        checks.append(("① carried_over.keys 含那四项（读者能看出不是本次算的）",
                       all(k in keys for k in carry_keys)))
        # 判据 1（成对搬）：旧文件写着 world + world_frame -> 两个必须**一起**回来。
        # 只搬 world_frame 会得到"标签说相机系、数据是世界系"的静默错配。
        checks.append((f"① 成对搬：coordinate_system 跟 world_frame 一起变成 world"
                       f"（实际 {traj_new.get('coordinate_system')!r}）",
                       traj_new.get("coordinate_system") == "world"
                       and "coordinate_system" in keys and "world_frame" in keys))
        checks.append(("① carried_over.reason 交代了坐标系是沿用的",
                       "坐标系沿用上一版" in (carried.get("reason") or "")))
        checks.append((f"② carried_over.from_created == 旧 created"
                       f"（{carried.get('from_created')!r}）",
                       carried.get("from_created") == created_old))
        checks.append(("② carried_over.reason 非空", bool((carried.get("reason") or "").strip())))
        checks.append(("① notes 是**合并**（上一版的说明还在）",
                       "上一版轨迹说明" in (traj_new.get("notes") or [])))

        vis_new = _read(derived / "visualization.json")
        checks.append(("④ visualization.json 也保住 quality（GUI/3D 查看器只读它）",
                       (vis_new.get("quality") or {}).get("verdict")
                       == "detection_quality"))
        checks.append(("④ visualization.json 也带 carried_over 标记",
                       bool(vis_new.get("carried_over"))))
        # 判据 4：notes 不许自相矛盾。新文件由 `_apply_world_frame` 生成，本机没有
        # config/world_frame.json 时它会写"当前坐标为左相机系"——搬了 world 标签后
        # 那句已经与新标签打脸，必须换成"坐标系沿用上一版 world"，且**不许删**
        # `_apply_world_frame` 写的其它说明（"已应用世界系…"仍然要留着）。
        vis_notes = [str(n) for n in (vis_new.get("notes") or [])]
        checks.append(("④ vis 的 notes 不再同时说「已应用世界系」和「未设置 world_frame.json」",
                       not (any("已应用世界系" in n for n in vis_notes)
                            and any("未设置 world_frame.json" in n for n in vis_notes))))
        checks.append(("④ vis 的 notes 写明「坐标系沿用上一版 world」",
                       any("坐标系沿用上一版 world" in n for n in vis_notes)))
        checks.append(("④ vis 的坐标标签与那句 note 一致（都是 world）",
                       vis_new.get("coordinate_system") == "world"))

        # ---- 判据 3：轨迹变了（多一行 -> n_matched 5 -> 6）必须丢 ----
        _write_csv(derived / "traj3d.csv", 6)
        rc2, out2 = _run_cli(session)
        traj_changed = _read(derived / "traj3d.json")
        vis_changed = _read(derived / "visualization.json")
        checks.append((f"③ 轨迹已变：CLI 退出码 0（实际 {rc2}）", rc2 == 0))
        checks.append(("③ n_matched 真的变成 6（确认判据看的是计数）",
                       traj_changed.get("n_matched") == 6))
        checks.append(("③ quality 不在（旧结论没挂到新轨迹上）",
                       "quality" not in traj_changed))
        checks.append(("③ predictions 不在", "predictions" not in traj_changed))
        checks.append(("③ notes 写明「轨迹已变…已丢弃」",
                       any("轨迹已变" in str(n) and "已丢弃" in str(n)
                           for n in (traj_changed.get("notes") or []))))
        checks.append(("③ visualization.json 的 quality 同样丢弃",
                       "quality" not in vis_changed))
        checks.append(("③ 丢弃时 stdout 有 [丢弃] 摘要", "[丢弃]" in out2))
        # 判据 3：轨迹变了连坐标标签也不许沿用 —— 丢弃分支里不动坐标系。
        checks.append((f"③ coordinate_system 回到重建值（实际"
                       f" {traj_changed.get('coordinate_system')!r}）",
                       traj_changed.get("coordinate_system") == "left_camera"))
        checks.append(("③ world_frame 不在（旧世界系连同新轨迹一起丢）",
                       "world_frame" not in traj_changed))
        checks.append(("③ vis 的 coordinate_system 也回到重建值",
                       vis_changed.get("coordinate_system") == "left_camera"))

        # ---- 纯函数边界：previous 不是 dict / 轨迹变了 ----
        base_payload = {"session": "x"}
        untouched = carry_over_runtime_extras(dict(base_payload), None,
                                              trajectory_changed=False)
        checks.append(("纯函数：previous=None -> 原样返回（首次导出不多事）",
                       untouched == base_payload))
        dropped = carry_over_runtime_extras(dict(base_payload), traj_old,
                                            trajectory_changed=True)
        checks.append(("纯函数：轨迹变了 -> 一个结论都不搬 + 坐标系标签也不动",
                       not any(k in dropped for k in carry_keys)
                       and "coordinate_system" not in dropped
                       and any("已丢弃" in str(n) for n in dropped.get("notes") or [])))

        # ---- 判据：`_trajectory_changed()` 的计数 / bounds 容差 ----
        base = {"n_matched": 5, "n_samples": 5,
                "bounds": {"min": [0.0, 0.0, 2.0], "max": [1.0, 1.0, 3.0]}}
        jitter = _json.loads(_json.dumps(base))
        jitter["bounds"]["min"][0] += 5e-9          # 浮点抖动 << 1e-6
        checks.append(("判据：计数相同 + bounds 抖动 5e-9 -> 未变（允许沿用）",
                       not _trajectory_changed(base, jitter)))
        bumped = _json.loads(_json.dumps(base))
        bumped["n_matched"] = 6
        checks.append(("判据：n_matched 5->6 -> 已变", _trajectory_changed(base, bumped)))
        far = _json.loads(_json.dumps(base))
        far["bounds"]["max"][2] += 1e-3             # 1e-3 > 1e-6
        checks.append(("判据：bounds 差 1e-3 > 1e-6 -> 已变",
                       _trajectory_changed(base, far)))
    finally:
        _shutil.rmtree(tmp, ignore_errors=True)

    # ---- 判据 2：不成对不搬（旧文件是相机系，却鬼使神差留了个 world_frame）----
    tmp_lone = _tempfile.mkdtemp(prefix="export_carry_lone_")
    try:
        session_c = _make_session(Path(tmp_lone))
        lone = dict(traj_old)
        lone["coordinate_system"] = "rectified_left_camera"   # 相机系（脏数据）
        lone.pop("notes", None)
        (session_c / "derived" / "traj3d.json").write_text(
            _json.dumps(lone, ensure_ascii=False, indent=2), encoding="utf-8")
        rc4, out4 = _run_cli(session_c)
        lone_new = _read(session_c / "derived" / "traj3d.json")
        lone_keys = list((lone_new.get("carried_over") or {}).get("keys") or [])
        checks.append((f"② 不成对：旧 coordinate_system 是相机系 -> 不搬 world_frame"
                       f"（rc={rc4}）",
                       rc4 == 0 and "world_frame" not in lone_new))
        checks.append((f"② 不成对：coordinate_system 保持重建值"
                       f"（实际 {lone_new.get('coordinate_system')!r}）",
                       lone_new.get("coordinate_system") == "left_camera"))
        checks.append(("② 不成对：carried_over.keys 里也没有坐标系两项",
                       "coordinate_system" not in lone_keys
                       and "world_frame" not in lone_keys))
        checks.append(("② 不成对：quality 照旧沿用（不因坐标问题一起丢）",
                       (lone_new.get("quality") or {}).get("verdict")
                       == "detection_quality"))
        checks.append((f"② 不成对：stdout 不说「坐标系沿用」", "坐标系沿用" not in out4))
    finally:
        _shutil.rmtree(tmp_lone, ignore_errors=True)

    # ---- 判据 5：旧文件损坏 -> 导出仍成功，只警告、不搬、不抛 ----
    tmp_broken = _tempfile.mkdtemp(prefix="export_carry_broken_")
    try:
        session_b = _make_session(Path(tmp_broken))
        (session_b / "derived" / "traj3d.json").write_text(
            "{ 这不是 json", encoding="utf-8")
        rc3, out3 = _run_cli(session_b)
        broken = _read(session_b / "derived" / "traj3d.json")
        checks.append((f"⑤ 旧 traj3d.json 是非法 JSON：导出仍成功（rc={rc3}）",
                       rc3 == 0 and "Traceback" not in out3))
        checks.append(("⑤ 只打一行 [警告]，不抛", "[警告]" in out3))
        checks.append(("⑤ 坏文件不当「上一版」：产物里没有 quality / carried_over",
                       "quality" not in broken and "carried_over" not in broken))
    except Exception as exc:                                # noqa: BLE001
        checks.append((f"⑤ 损坏文件路径不该抛异常：{exc!r}", False))
    finally:
        _shutil.rmtree(tmp_broken, ignore_errors=True)

    ok = True
    for label, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {label}")
        ok = ok and passed
    return ok


def test_world_frame_in_traj_payload() -> bool:
    """`traj3d.json` 的 xyz 必须真的换到 `coordinate_system` 说的那个系（B-134）。

    问题起点：`run_stereo_video.py` 有世界系时把 `coordinate_system` 改写成 "world"、
    并写上 `world_frame`，但 `samples[*]["xyz"]` 仍是 `traj3d.csv` 的**相机系**坐标
    —— **标签与数据不一致**：下游按标签解释就得到错数，而且看不出来。世界系坐标
    其实早在 `_world_block()` 里算好了（另存进 `traj3d_world.csv`），只是没写进 JSON。

    这里跑三段**真实的 `main()` 控制流**（只 mock 输入侧：视频源/检测/重建/写盘，
    与 `test_guard_no_early_stop` 同一套做法），输入输出全在 `tempfile.mkdtemp()` 里：

    - ① `--no-world`：主路径不变 —— `coordinate_system` 是 `base_frame`、
      `world_frame is None`、matched 的 xyz 与 `traj3d.csv` 同一行逐个相同（1e-12）；
    - ② `--world-frame <非单位阵>`：matched 的 xyz 逐点 == `world.apply(相机系)`
      （1e-9）、`coordinate_system == "world"`、`world_frame` 三字段齐全；
    - ③ 同一次跑出来的 `visualization.json` 与 `traj3d.json` 数值一致
      （同一段轨迹，两个文件不许各说一套）；
    - ④ 世界系行数与 matched 样本数不一致（人为截断）：**不宣称 world**、
      `world_skipped` 有理由、stdout 有 `[警告]`、进程仍 `return 0`。

    反向自证：把 `_apply_world_to_trajectory_payload()` 里换 xyz 的那几行注掉
    （只留标签）→ ② 与 ③ 必须变红；字节级还原后 `sha256` 与改前一致。
    """
    print("----- 世界系：traj3d.json 的 xyz 必须真的换系 -----")
    import contextlib as _contextlib
    import io as _io
    import json as _json
    import shutil as _shutil
    import sys as _sys
    import tempfile as _tempfile
    import types as _types
    from unittest import mock as _mock

    import apps.run_stereo_video as rsv
    from calib.world_frame import WorldFrame
    from stereo.visualization import read_trajectory_csv

    n_frames = 24
    # 世界系必须是**非单位阵**（绕 X 转 30° + 平移）：单位阵下"没换系"也是对的，
    # 测不出缺陷 —— 这条是判据②的前提，不是可选项。
    ang = math.radians(30.0)
    T_wc = np.eye(4)
    T_wc[:3, :3] = np.array([[1.0, 0.0, 0.0],
                             [0.0, math.cos(ang), -math.sin(ang)],
                             [0.0, math.sin(ang), math.cos(ang)]])
    T_wc[:3, 3] = [0.31, -0.12, 0.47]
    world_fixture = WorldFrame(T_world_cam=T_wc, source="fixture",
                               note="非单位阵（绕 X 30°+平移）",
                               created_at="2026-01-01 00:00:00")

    holder: dict[str, object] = {}
    frame = np.zeros((540, 960, 3), np.uint8)

    class _FakePairSource:
        image_size = (960, 540)

        def __init__(self, *a, **k):
            self.n = 0

        def read(self):
            if self.n >= n_frames:
                return (False, None, None)
            self.n += 1
            return (True, frame, frame)

        def report_warnings(self):
            pass

        def describe(self):
            return "fake-index"

        def release(self):
            pass

    class _FakePipeline:
        def __init__(self):
            self.detector = _types.SimpleNamespace(debug_snapshot=lambda: {})

        def process(self, frame, t):
            return _types.SimpleNamespace(track=None, detection=None)

    class _FakePoint:
        residual_px = 0.5

    class _FakeRec:
        """假重建器：每帧都配一对，并把**相机系**坐标记下来供判据比对。"""

        def __init__(self, *a, **k):
            holder["rec"] = self
            self.n_matched = 0
            self.n_frames = 0
            self.dy_samples = [0.4] * 30
            self.n_no_left = self.n_no_right = self.n_rejected = 0
            self.trajectory = [_FakePoint()]
            self.warmup_frames = 0
            self.warnings = []
            self.pts: list[tuple[float, float, float]] = []
            self.ts: list[float] = []

        def update(self, t, det_l, det_r):
            i = len(self.pts)
            self.ts.append(float(i) * 0.033)
            self.pts.append((0.10 + 0.005 * i, 0.20 - 0.003 * i, 3.0 + 0.01 * i))
            self.n_matched += 1
            self.n_frames += 1
            return _types.SimpleNamespace(
                xyz=self.pts[-1], u_left=(100.0, 200.0), u_right=(90.0, 200.0),
                epipolar_px=0.5, residual_px=0.5, confident=1)

        @property
        def match_rate(self):
            return 1.0

        def as_array(self):
            return np.array([[self.ts[i], *p] for i, p in enumerate(self.pts)],
                            np.float64).reshape(-1, 4)

        def low_match_diagnostic(self):
            return []

        def geometry_failure_message(self, *a, **k):
            return "[占位] 检出质量差 / 几何不一致 分开报文案"

    class _FakeWriter:
        def __init__(self, path, fourcc, fps, size):
            self._path = Path(path)

        def isOpened(self):
            return True

        def write(self, frame):
            pass

        def release(self):
            self._path.write_bytes(b"fake-mp4")

    cam = _types.SimpleNamespace(video="left.mkv", fps_measured=30.0,
                                 fps_requested=30.0)
    sess = _types.SimpleNamespace(
        left=cam, right=cam, calibration=_types.SimpleNamespace(),
        stereo_mode="angled", rig={},
        sync=_types.SimpleNamespace(offset=0, method="flash"))

    def _run(*, world: bool, truncate_world: bool = False):
        """跑一次真实 main()；返回 (rc, stdout, out_dir, tmp)。"""
        tmp = _tempfile.mkdtemp(prefix="world_traj_")
        session_dir = Path(tmp) / "session"
        out_dir = Path(tmp) / "out"
        session_dir.mkdir()
        (session_dir / "derived").mkdir()
        # session.json 必须是**真实文件**：export 内部会读它重建 payload。
        (session_dir / "session.json").write_text(_json.dumps({
            "id": "world_probe",
            "left": {"width": 960, "height": 540},
            "right": {"width": 960, "height": 540},
            "stereo_mode": "angled",
            "calibration": {"baseline_mm": 150.0, "verdict": "ok",
                            "created": "2026-01-01 00:00:00"},
            "sync": {"offset": 0, "method": "flash"},
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        wf_path = Path(tmp) / "world_frame.json"
        world_fixture.save(wf_path)
        argv = ["run_stereo_video.py", str(session_dir), "--out", str(out_dir),
                "--pairing", "index", "--no-plot", "--no-predict",
                "--no-radius-auto", "--method", "color", "--stereo-mode", "angled",
                "--geometry-guard", "0"]
        argv += ["--no-world"] if not world else ["--world-frame", str(wf_path)]
        rig = build_rig()
        patches = [
            _mock.patch.object(rsv, "load_session", return_value=(session_dir, sess)),
            _mock.patch.object(rsv, "validate_session", return_value=([], [])),
            _mock.patch.object(rsv, "session_calibration_file",
                               return_value=session_dir / "calib.npz"),
            _mock.patch.object(rsv.StereoRig, "load", return_value=rig),
            _mock.patch.object(rsv, "resolve_auto", return_value="angled"),
            _mock.patch.object(rsv, "_mode_guard", return_value=("angled", False)),
            _mock.patch.object(rsv, "optical_axes_angle_deg", return_value=88.0),
            _mock.patch.object(rsv, "_pick_detector", return_value=("color", "")),
            _mock.patch.object(rsv, "PairSource", _FakePairSource),
            _mock.patch.object(rsv, "_make_pipeline",
                               side_effect=lambda cfg, r, side: _FakePipeline()),
            _mock.patch.object(rsv, "_draw", side_effect=lambda *a, **k: frame),
            _mock.patch.object(rsv.cv2, "VideoWriter", _FakeWriter),
            _mock.patch("stereo.angled.AngledReconstructor", _FakeRec),
        ]
        real_block = rsv._world_block
        if truncate_world:
            def _short(world_obj, arr, base_frame):
                w_arr, block = real_block(world_obj, arr, base_frame)
                if w_arr is None or len(w_arr) == 0:
                    return w_arr, block
                return w_arr[:-1], block
            patches.append(_mock.patch.object(rsv, "_world_block", _short))
        rc, stdout = None, ""
        buf = _io.StringIO()
        try:
            for p in patches:
                p.start()
            old_argv = _sys.argv
            _sys.argv = argv
            try:
                with _contextlib.redirect_stdout(buf):
                    rc = rsv.main()
            finally:
                _sys.argv = old_argv
            stdout = buf.getvalue()
        finally:
            for p in reversed(patches):
                p.stop()
        if rc != 0:
            print(f"  [调试] main() 返回 {rc}，stdout 如下：\n{stdout}")
        return rc, stdout, out_dir, tmp

    def _matched(payload: dict) -> list[list[float]]:
        return [s["xyz"] for s in payload.get("samples") or []
                if s.get("matched") and s.get("xyz")]

    checks: list[tuple[str, bool]] = []

    # ---- ① 没开世界系：主路径不变 ----
    tmp_a = None
    try:
        rc, out, out_dir, tmp_a = _run(world=False)
        traj = _json.loads((out_dir / "traj3d.json").read_text(encoding="utf-8"))
        csv_samples = read_trajectory_csv(out_dir / "traj3d.csv")
        csv_xyz = [s["xyz"] for s in csv_samples if s["matched"] and s["xyz"]]
        got = _matched(traj)
        same = (len(got) == len(csv_xyz) and len(csv_xyz) == n_frames
                and all(abs(a[k] - b[k]) <= 1e-12
                        for a, b in zip(got, csv_xyz) for k in range(3)))
        checks.append((f"① --no-world：main() 返回 0（实际 {rc}）", rc == 0))
        checks.append((f"① --no-world：coordinate_system == base_frame（实际"
                       f" {traj.get('coordinate_system')!r}）",
                       traj.get("coordinate_system") == "left_camera"))
        checks.append(("① --no-world：world_frame is None", traj.get("world_frame") is None))
        checks.append((f"① --no-world：{len(got)} 个 matched 点的 xyz 与 CSV 逐个相同（1e-12）",
                       same))
        checks.append(("① --no-world：没有 world_skipped（主路径不多写键）",
                       "world_skipped" not in traj))
    finally:
        if tmp_a:
            _shutil.rmtree(tmp_a, ignore_errors=True)

    # ---- ②③ 给一个非单位阵世界系：xyz 必须真的换系，两个文件必须一致 ----
    tmp_b = None
    try:
        rc, out, out_dir, tmp_b = _run(world=True)
        rec = holder.get("rec")
        cam_pts = np.array(getattr(rec, "pts", []), np.float64).reshape(-1, 3)
        expect = world_fixture.apply(cam_pts)
        traj = _json.loads((out_dir / "traj3d.json").read_text(encoding="utf-8"))
        got = np.array(_matched(traj), np.float64).reshape(-1, 3)
        checks.append((f"② --world：main() 返回 0（实际 {rc}）", rc == 0))
        checks.append((f"② --world：coordinate_system == 'world'（实际"
                       f" {traj.get('coordinate_system')!r}）",
                       traj.get("coordinate_system") == "world"))
        wf = traj.get("world_frame") or {}
        checks.append(("② --world：world_frame 的 source/note/created_at 齐全",
                       bool(wf.get("source")) and "note" in wf
                       and bool(wf.get("created_at"))))
        checks.append(("② --world：world_csv 指向 traj3d_world.csv",
                       (traj.get("world_csv") or "").endswith("traj3d_world.csv")))
        close = (got.shape == expect.shape and got.size > 0
                 and float(np.max(np.abs(got - expect))) <= 1e-9)
        checks.append((f"② --world：{len(got)} 个 matched 点 xyz == world.apply(相机系)"
                       f"（1e-9，最大偏差 "
                       f"{0.0 if not got.size else float(np.max(np.abs(got - expect))):.2e}）",
                       close))
        checks.append(("② --world：不是照抄相机系坐标（与 CSV 明显不同）",
                       got.size > 0 and float(np.max(np.abs(got - cam_pts))) > 1e-3))
        # bounds 必须跟着换（口径同 build_trajectory_payload）
        b_expect = {"min": [float(expect[:, k].min()) for k in range(3)],
                    "max": [float(expect[:, k].max()) for k in range(3)]}
        b_got = traj.get("bounds") or {}
        checks.append(("② --world：bounds 按换系后的点重算",
                       bool(b_got) and all(
                           abs(b_got[e][k] - b_expect[e][k]) <= 1e-9
                           for e in ("min", "max") for k in range(3))))

        vis = _json.loads((out_dir / "visualization.json").read_text(encoding="utf-8"))
        vis_pts = np.array([p["xyz"] for p in (vis.get("trajectory") or [])],
                           np.float64)
        # 允许 2e-5：traj 的 xyz 来自未取整的 world_arr，vis 走的是 CSV（6 位小数）。
        checks.append((f"③ 同一次 --world：visualization.json 的 {len(vis_pts)} 个点与"
                       f" traj3d.json 一致（≤2e-5）",
                       vis_pts.shape == got.shape and got.size > 0
                       and float(np.max(np.abs(vis_pts - got))) <= 2e-5))
        checks.append((f"③ 同一次 --world：visualization.json 也自称 world（实际"
                       f" {vis.get('coordinate_system')!r}）",
                       vis.get("coordinate_system") == "world"))
    finally:
        if tmp_b:
            _shutil.rmtree(tmp_b, ignore_errors=True)

    # ---- ④ 世界系行数与 matched 样本数不一致：不许宣称 world，但也不许崩 ----
    tmp_c = None
    try:
        rc, out, out_dir, tmp_c = _run(world=True, truncate_world=True)
        traj = _json.loads((out_dir / "traj3d.json").read_text(encoding="utf-8"))
        csv_xyz = [s["xyz"] for s in read_trajectory_csv(out_dir / "traj3d.csv")
                   if s["matched"] and s["xyz"]]
        got = _matched(traj)
        checks.append((f"④ 数量不一致：进程仍返回 0（实际 {rc}）", rc == 0))
        checks.append(("④ 数量不一致：stdout 有 [警告] 说明未换系", "[警告]" in out))
        checks.append((f"④ 数量不一致：不宣称 world（实际"
                       f" {traj.get('coordinate_system')!r}）",
                       traj.get("coordinate_system") == "left_camera"))
        checks.append(("④ 数量不一致：world_skipped 写了原因",
                       "不一致" in (traj.get("world_skipped") or "")))
        checks.append(("④ 数量不一致：world_frame 不写（不给半套世界系）",
                       traj.get("world_frame") is None))
        checks.append(("④ 数量不一致：xyz 保持 CSV 原值（没换系也不许涂层）",
                       len(got) == len(csv_xyz) and all(
                           abs(a[k] - b[k]) <= 1e-12
                           for a, b in zip(got, csv_xyz) for k in range(3))
                       and float(np.max(np.abs(
                           np.array(got) - np.array(csv_xyz)))) <= 1e-12))
    finally:
        if tmp_c:
            _shutil.rmtree(tmp_c, ignore_errors=True)

    ok = True
    for label, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {label}")
        ok = ok and passed
    return ok


def test_round_trip_metric() -> bool:
    """往返一致性：去程 / 回程在**同一个 X** 处差多少（`traj3d.json["round_trip"]`）。

    为什么需要这个指标：双目只在极线方向上受硬约束，竖直 Y 与深度 Z 全靠三角
    测量。球来回摆时，同一个 X 处若去程与回程的 (Y,Z) **系统性**错开，说明这条
    轨迹本身不是一条线（窄椭圆 / 空回差）；若偏移方向忽正忽负，那更像逐帧检测
    抖动。这两种情况的处置完全不同，靠眼睛看三维图分辨不出来。

    全部**合成**数据（不读 `data/`），钉七件事：

    - ① 原路返回（Y/Z 完全重合）-> `retrace`，且 `sep_median_mm < 0.1`；
    - ② 窄椭圆两个来回 -> `ellipse`、`n_pairs == 2`、`sign_consistent`、
      `dz_median_mm == +18 mm`（`Δ = 回程 - 去程`，去程 Z=-9、回程 Z=+9）；
    - ③ 两段来回偏移方向相反（+8 / -8 mm）-> `scattered`；
    - ④ 只有一条腿（X 单调到底）-> `no_pairs`、`pairs == []`，**不抛**；
    - ⑤ 空数组 / 全 NaN / 只有 2 点 -> 返回 `None` 或 `no_pairs`，**不抛**
      （明确规定：空输入返回 `None`，调用方写成 JSON `null`）；
    - ⑥ 符号口径钉死：② 的 `dz_median_mm` 必须落在 [+17, +19]。**反向自证**：
      把 `stereo/round_trip.py` 里 `dy = y_ret - y_out` / `dz = z_ret - z_out`
      改成 `out - ret` -> ② 立刻变红（-18）；字节级还原后 `sha256` 与改前一致；
    - ⑦ 不污染既有产物：`build_trajectory_payload()` 的 `n_matched` / `samples` /
      `unit` / `bounds` 与手算值逐项相同，`round_trip` 也只是**新增**键、
      不进 `RUNTIME_EXTRA_KEYS`（几何属性，每次重算，不靠搬运）。

    所有输入输出都在 `tempfile.mkdtemp()` 里，`finally: shutil.rmtree`
    —— 绝不碰 `data/`（用户真实录制数据）。
    """
    print("----- 往返一致性：去程 / 回程在同一个 X 处差多少（round_trip）-----")
    import json as _json
    import shutil as _shutil
    import tempfile as _tempfile

    from stereo.round_trip import analyze_round_trip
    from stereo.visualization import RUNTIME_EXTRA_KEYS, build_trajectory_payload

    def _trip(leg_spec: list[tuple[float, float, float, float]],
              *, n: int = 50, dt: float = 0.033) -> tuple[np.ndarray, np.ndarray]:
        """按腿拼一条轨迹。leg_spec 每项是 (x起, x止, y, z)，腿内线性。

        相邻腿共用端点（上一腿的止点 == 下一腿的起点）—— 顶点那一帧差分恰好
        为 0，正好练习「零差分沿用前一个方向、不把一条腿劈成两段」。
        """
        xs, ys, zs = [], [], []
        for x0, x1, y, z in leg_spec:
            xs.append(np.linspace(x0, x1, n))
            ys.append(np.full(n, float(y)))
            zs.append(np.full(n, float(z)))
        X = np.concatenate(xs)
        t = np.arange(X.size) * dt
        return t, np.column_stack([X, np.concatenate(ys), np.concatenate(zs)])

    checks: list[tuple[str, bool]] = []

    # ---- ① 原路返回 ----
    t1, P1 = _trip([(0.0, 0.4, 0.10, 2.0), (0.4, 0.0, 0.10, 2.0)])
    r1 = analyze_round_trip(t1, P1)
    checks.append((f"① 原路返回 -> verdict=retrace（实际 {r1 and r1['verdict']}）",
                   bool(r1) and r1["verdict"] == "retrace"))
    checks.append((f"① sep_median_mm {r1 and r1['sep_median_mm']} < 0.1 mm",
                   bool(r1) and r1["sep_median_mm"] is not None
                   and r1["sep_median_mm"] < 0.1))
    checks.append((f"① n_pairs == 1（实际 {r1 and r1['n_pairs']}）",
                   bool(r1) and r1["n_pairs"] == 1))

    # ---- ② 窄椭圆：两个来回，去程 Z=-9 mm、回程 Z=+9 mm ----
    t2, P2 = _trip([(0.0, 0.4, 0.10, -0.009), (0.4, 0.0, 0.10, 0.009),
                    (0.0, 0.4, 0.10, -0.009), (0.4, 0.0, 0.10, 0.009)])
    r2 = analyze_round_trip(t2, P2)
    print(f"  ② 实际返回：{_json.dumps(r2, ensure_ascii=False)}")
    checks.append((f"② 窄椭圆 -> verdict=ellipse（实际 {r2 and r2['verdict']}）",
                   bool(r2) and r2["verdict"] == "ellipse"))
    checks.append((f"② n_pairs == 2（实际 {r2 and r2['n_pairs']}）",
                   bool(r2) and r2["n_pairs"] == 2))
    checks.append((f"② sign_consistent is True（实际 {r2 and r2['sign_consistent']}）",
                   bool(r2) and r2["sign_consistent"] is True))
    checks.append((f"② dz_median_mm == +18 ± 1（实际 {r2 and r2['dz_median_mm']}）",
                   bool(r2) and r2["dz_median_mm"] is not None
                   and 17.0 <= r2["dz_median_mm"] <= 19.0))
    checks.append((f"② dy_median_mm ≈ 0（实际 {r2 and r2['dy_median_mm']}）",
                   bool(r2) and r2["dy_median_mm"] is not None
                   and abs(r2["dy_median_mm"]) < 0.01))
    checks.append((f"② sep_median_mm ≈ 18（|Δ| 就是间距，实际"
                   f" {r2 and r2['sep_median_mm']}）",
                   bool(r2) and r2["sep_median_mm"] is not None
                   and 17.0 <= r2["sep_median_mm"] <= 19.0))
    checks.append((f"② 每对都有速度（实际 "
                   f"{r2 and [(p['speed_out_mm_s'], p['speed_back_mm_s']) for p in r2['pairs']]}）",
                   bool(r2) and all(p["speed_out_mm_s"] is not None
                                    and p["speed_back_mm_s"] is not None
                                    and p["speed_out_mm_s"] > 0
                                    for p in r2["pairs"])))
    checks.append((f"② x_overlap_m 落在 [0.4] 之内（实际"
                   f" {r2 and r2['pairs'][0]['x_overlap_m']}）",
                   bool(r2) and 0.0 < r2["pairs"][0]["x_overlap_m"][0]
                   < r2["pairs"][0]["x_overlap_m"][1] < 0.4))

    # ---- ③ 符号乱：第一段来回 Z 偏 +8 mm、第二段 -8 mm ----
    t3, P3 = _trip([(0.0, 0.4, 0.10, 0.0), (0.4, 0.0, 0.10, 0.008),
                    (0.0, 0.4, 0.10, 0.0), (0.4, 0.0, 0.10, -0.008)])
    r3 = analyze_round_trip(t3, P3)
    checks.append((f"③ 偏移方向相反 -> verdict=scattered（实际 {r3 and r3['verdict']}）",
                   bool(r3) and r3["verdict"] == "scattered"))
    checks.append((f"③ sign_consistent is False（实际 {r3 and r3['sign_consistent']}）",
                   bool(r3) and r3["sign_consistent"] is False))
    checks.append((f"③ 两对的 dz 反号（实际 "
                   f"{r3 and [p['dz_median_mm'] for p in r3['pairs']]}）",
                   bool(r3) and {np.sign(p["dz_median_mm"]) for p in r3["pairs"]}
                   == {1.0, -1.0}))

    # ---- ④ 只有一条腿 ----
    t4, P4 = _trip([(0.0, 0.4, 0.10, 2.0)], n=100)
    r4 = analyze_round_trip(t4, P4)
    checks.append((f"④ 只有一条腿 -> verdict=no_pairs（实际 {r4 and r4['verdict']}）",
                   bool(r4) and r4["verdict"] == "no_pairs"))
    checks.append((f"④ pairs == []（实际 {r4 and r4['pairs']}）",
                   bool(r4) and r4["pairs"] == []))
    checks.append((f"④ n_pairs == 0 且 note 非空（实际 {r4 and r4['n_pairs']}）",
                   bool(r4) and r4["n_pairs"] == 0 and bool(r4["note"])))

    # ---- ⑤ 边界输入：不抛，返回 None 或 no_pairs ----
    def _safe(t, P) -> tuple[bool, object]:
        try:
            return True, analyze_round_trip(t, P)
        except Exception as exc:                    # noqa: BLE001
            return False, exc

    empty_ok, empty_val = _safe(np.array([]), np.zeros((0, 3)))
    nan_ok, _ = _safe(np.arange(20.0), np.full((20, 3), np.nan))
    two_ok, two_val = _safe(np.array([0.0, 0.033]),
                            np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]))
    still_ok, still_val = _safe(np.arange(30.0), np.full((30, 3), 0.0))
    checks.append((f"⑤ 空数组：不抛，返回 None（实际 {empty_val!r}）",
                   empty_ok and empty_val is None))
    checks.append(("⑤ 全 NaN：不抛，返回 None 或 no_pairs（实际 "
                   f"{nan_ok}）", nan_ok))
    checks.append((f"⑤ 只有 2 点：不抛，返回 None（实际 {two_val!r}）",
                   two_ok and two_val is None))
    checks.append((f"⑤ 完全不动（30 点同一点）：不抛，no_pairs（实际"
                   f" {(still_val or {}).get('verdict')!r}）",
                   still_ok and still_val is not None
                   and still_val["verdict"] == "no_pairs"))

    # ---- ⑥ 符号口径：② 的 dz 必须是 +18（改成 out-ret 会变成 -18） ----
    checks.append((f"⑥ 符号口径：Δ = 回程 - 去程（dz_median_mm={r2 and r2['dz_median_mm']}，"
                   "改成 out-ret 会变 -18 即本条变红）",
                   bool(r2) and 17.0 <= r2["dz_median_mm"] <= 19.0
                   and r2["pairs"][0]["dz_median_mm"] > 0))

    # ---- ⑦ 不污染既有产物 ----
    header = ("frame,t,matched,X,Y,Z,u_left,v_left,u_right,v_right,"
              "epipolar_px,residual_px,confident,method_left,method_right")
    rows = []
    xs = list(np.linspace(0.0, 0.4, 20)) + list(np.linspace(0.4, 0.0, 20))
    for i, x in enumerate(xs):
        rows.append(f"{i},{i * 0.02:.4f},1,{x:.6f},0.050000,2.000000,"
                    f"480,270,470,270,0.3,0.4,1,color,color")
    rows.insert(3, "999,0.0000,0,,,,,,,,,0,0,,")   # 一行 matched=0：不许参与
    tmp_rt = None
    try:
        session = Path(_tempfile.mkdtemp(prefix="round_trip_probe_"))
        tmp_rt = str(session)
        (session / "derived").mkdir(parents=True)
        (session / "session.json").write_text(_json.dumps({
            "id": "round_trip_probe",
            "left": {"width": 960, "height": 540},
            "right": {"width": 960, "height": 540},
            "stereo_mode": "parallel",
            "calibration": {"baseline_mm": 150.0, "created": "2026-01-01 00:00:00",
                            "verdict": "ok"},
            "sync": {"offset": 0, "method": "flash"},
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        (session / "derived" / "traj3d.csv").write_text(
            "\n".join([header, *rows]) + "\n", encoding="utf-8")

        payload = build_trajectory_payload(session)
        checks.append((f"⑦ n_samples == 41（40 matched + 1 未匹配，实际"
                       f" {payload['n_samples']}）", payload["n_samples"] == 41))
        checks.append((f"⑦ n_matched == 40（实际 {payload['n_matched']}）",
                       payload["n_matched"] == 40))
        checks.append((f"⑦ unit == 'm'（实际 {payload['unit']!r}）",
                       payload["unit"] == "m"))
        checks.append((f"⑦ bounds 与手算一致（实际 {payload['bounds']}）",
                       payload["bounds"] == {"min": [0.0, 0.05, 2.0],
                                             "max": [0.4, 0.05, 2.0]}))
        keys = {"frame", "t", "matched", "xyz", "uv_left", "uv_right",
                "epipolar_px", "residual_px", "confident",
                "method_left", "method_right"}
        checks.append(("⑦ samples 每条仍是原来那 11 个键（没加字段）",
                       len(payload["samples"]) == 41
                       and all(set(s) == keys for s in payload["samples"])))
        checks.append(("⑦ samples[0].xyz 仍是 CSV 原值 [0, 0.05, 2]",
                       payload["samples"][0]["xyz"] == [0.0, 0.05, 2.0]))
        checks.append((f"⑦ round_trip 是**新增**键且算出来了（verdict="
                       f"{(payload.get('round_trip') or {}).get('verdict')!r}）",
                       payload.get("round_trip") is not None
                       and payload["round_trip"]["verdict"] == "retrace"))
        checks.append(("⑦ round_trip 不在 RUNTIME_EXTRA_KEYS（几何属性，每次重算，"
                       "不参与 carried_over）",
                       "round_trip" not in RUNTIME_EXTRA_KEYS))
        checks.append((f"⑦ CSV 仍是 15 列（实际 {len(header.split(','))}）",
                       len(header.split(",")) == 15))
    finally:
        if tmp_rt:
            _shutil.rmtree(tmp_rt, ignore_errors=True)

    ok = True
    for label, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {label}")
        ok = ok and passed
    return ok


def test_fov_apply() -> bool:
    """装置级 fov_apply：数据模型往返 + 六个判定分支 + 端到端 + 反向自证。

    概念（用户原话）：「启用」= 把这份 FOV 参数应用到当前装置的两路相机画面上，
    是**装置级**事实，不是「标定时用不用」「录制时用不用」两个流程开关。标定与
    录制只是它的下游后果，都服从 :func:`stereo.fov_policy.resolve_fov_apply`。

    下列判据（全部临时目录 + 显式路径，绝不写真 config/data，也不起相机）：
      ① RigProfile 往返 / 缺字段 / 字符串 "false" 回落 False；
      ② 未启用+标定 fov=null -> apply=False 且无 legacy_follow/无 warning；
      ③ 启用+档案无 fov+标定也没 fov -> 无 error（benign）；
      ③b 启用+档案无 fov+标定带 fov -> **有 error**（内参/画面静默错配）；
      ④ 启用+parallel+标定 fov=null -> 有 error、不许 apply；
      ⑤ 启用+parallel+标定 fov 与档案 scale/crop_rect/target_size 任一不同 -> error；
      ⑥ 启用+parallel+一致 -> apply=True 且返回档案那份 fov(dict)；
      ⑦ 启用+angled -> apply=False、理由含「不共视」、无 error；
      ⑧ 关+带 fov 旧标定 -> legacy_follow=True、warning 含「沿用标定」，端到端
         决策链真跑最终 fov 仍是标定读出的对象（未置 None）；关+标定 fov=null 无提示；
      ⑨ 反向自证：删掉 resolve_fov_apply 第 2 条整段 -> ②⑧ 变红；
         字节级还原后 sha256 与改前一致，且还原后 ②⑧ 复绿。
     ⑩ angled 忽略警告抽成纯函数 angled_ignore_message（真调，含「忽略」），
         并钉住 angled+默认关+带 fov 标定 -> apply=False 且无 error（不拒绝录制）。
    """
    print("----- 装置级 fov_apply：唯一判定点 resolve_fov_apply -----")
    import hashlib as _hashlib
    import importlib as _importlib
    import json as _json
    import shutil as _shutil
    import tempfile as _tempfile

    from stereo import fov_policy, profiles
    import apps.record_stereo as _rs

    FOV = {
        "reference_side": "left", "crop_side": "right",
        "crop_rect": [404, 195, 1718, 973], "target_size": [1718, 972],
        "scale": 0.664875885304807, "method": "sift",
        "inliers": 296, "quality": 0.513,
    }

    def _prof(apply, fov=FOV):
        return profiles.RigProfile(name="probe", fov_apply=apply, fov=fov)

    checks: list[tuple[str, bool]] = []

    # ① 数据模型
    _d = _prof(True).to_dict()
    checks.append(("① to_dict 带 fov_apply=True", _d.get("fov_apply") is True))
    checks.append(("① from_dict 读回仍是 True",
                   profiles.RigProfile.from_dict(_d).fov_apply is True))
    checks.append(("① 缺字段的老档案读成 False",
                   profiles.RigProfile.from_dict({"name": "old"}).fov_apply is False))
    checks.append(('① 字符串 "false" 也读成 False（不能 bool("false")==True）',
                   profiles.RigProfile.from_dict(
                       {"name": "x", "fov_apply": "false"}).fov_apply is False))
    checks.append(("① 非 bool 的数字 1 也回落 False",
                   profiles.RigProfile.from_dict(
                       {"name": "x", "fov_apply": 1}).fov_apply is False))
    _td = _tempfile.mkdtemp(prefix="fov_apply_probe_")
    try:
        _pp = Path(_td) / "profiles.json"
        profiles.upsert(_prof(True), replace_existing=True, path=_pp)
        checks.append(("① upsert 往返仍是 True",
                       profiles.get_profile("probe", path=_pp).fov_apply is True))
    finally:
        _shutil.rmtree(_td, ignore_errors=True)

    # ② 未启用 + 标定 fov=null -> 两路按原始画面（无 legacy_follow / 无 warning）
    def _case2():
        r = fov_policy.resolve_fov_apply(_prof(False), mode="parallel",
                                         calib_fov=None)
        return (r["apply"] is False and r["fov"] is None and r["error"] is None
                and r.get("legacy_follow") is False
                and r.get("warning", "") == "")

    checks.append(("② 未启用+标定 fov=null -> apply=False 且无 legacy_follow/无 warning",
                   _case2()))

    # ③ 启用但档案无 fov，且标定也没用 FOV -> benign（无 error）
    r3 = fov_policy.resolve_fov_apply(_prof(True, fov=None), mode="parallel",
                                      calib_fov=None)
    checks.append(("③ 启用+档案无 fov+标定 fov=null -> apply=False 且无 error",
                   r3["apply"] is False and r3["error"] is None
                   and "没有 FOV 记录" in r3["reason"]))

    # ③b 启用但档案无 fov，标定却带 fov -> 必须报错（内参/画面口径静默错配）
    r3b = fov_policy.resolve_fov_apply(_prof(True, fov=None), mode="parallel",
                                       calib_fov=FOV)
    checks.append(("③b 启用+档案无 fov+标定带 fov -> apply=False 且有 error",
                   r3b["apply"] is False and bool(r3b["error"])))
    checks.append(("③b error 点明「补进档案 / 关开关并重新标定」两条出路",
                   bool(r3b["error"]) and "从当前标定读入" in r3b["error"]
                   and "重新标定" in r3b["error"]))

    # ④ 启用 + parallel + 标定 fov null
    r4 = fov_policy.resolve_fov_apply(_prof(True), mode="parallel", calib_fov=None)
    checks.append(("④ 启用+parallel+标定 fov=null -> 有 error、不许 apply",
                   r4["apply"] is False and bool(r4["error"])))

    # ⑤ 启用 + parallel + 标定 fov 与档案不一致（三个字段逐一）
    r5a = fov_policy.resolve_fov_apply(_prof(True), mode="parallel",
                                       calib_fov=dict(FOV, scale=0.5))
    r5b = fov_policy.resolve_fov_apply(_prof(True), mode="parallel",
                                       calib_fov=dict(FOV, crop_rect=[1, 2, 3, 4]))
    r5c = fov_policy.resolve_fov_apply(_prof(True), mode="parallel",
                                       calib_fov=dict(FOV, target_size=[100, 100]))
    checks.append(("⑤ scale 不一致 -> 有 error",
                   r5a["apply"] is False and bool(r5a["error"])))
    checks.append(("⑤ crop_rect / target_size 任一不一致也 -> 有 error",
                   (r5b["apply"] is False and bool(r5b["error"]))
                   and (r5c["apply"] is False and bool(r5c["error"]))))

    # ⑥ 启用 + parallel + 一致
    r6 = fov_policy.resolve_fov_apply(_prof(True), mode="parallel", calib_fov=FOV)
    checks.append(("⑥ 启用+parallel+一致 -> apply=True 且返回档案那份 fov(dict)",
                   r6["apply"] is True and r6["error"] is None
                   and isinstance(r6["fov"], dict) and r6["fov"] == FOV))

    # ⑦ 启用 + angled
    r7 = fov_policy.resolve_fov_apply(_prof(True), mode="angled", calib_fov=FOV)
    checks.append(("⑦ 启用+angled -> apply=False、理由含「不共视」、无 error",
                   r7["apply"] is False and "不共视" in r7["reason"]
                   and r7["error"] is None))

    # ⑧ 关 + 带 fov 旧标定 -> 沿用标定自带的 FOV（用户裁决：旧流程行为不变）
    def _case8():
        with _tempfile.TemporaryDirectory(prefix="fov_apply_e2e_") as td:
            calib = Path(td) / "stereo_calibration.json"
            calib.write_text(_json.dumps(
                {"fov": FOV, "stereo_mode": "parallel"}, ensure_ascii=False),
                encoding="utf-8")
            raw = _json.loads(calib.read_text(encoding="utf-8"))["fov"]
            dec = _rs.decide_record_fov("parallel", raw, _prof(False))
            ok_flag = (dec["apply"] is False and dec["error"] is None
                       and dec.get("legacy_follow") is True
                       and "沿用标定" in dec.get("warning", ""))
            # 端到端决策链：真调 _load_calib_fov 拿到标定 fov 对象，再按 main
            # 决策块走一遍（legacy_follow 分支不置 None），不起相机。
            fov = _rs._load_calib_fov(raw)
            if dec["error"]:
                final_fov = None
            elif dec.get("legacy_follow"):
                final_fov = fov            # 沿用：保持标定读出来的对象，照旧裁剪
            elif not dec["apply"]:
                final_fov = None
            else:
                final_fov = fov
            return ok_flag and final_fov is fov

    r8 = fov_policy.resolve_fov_apply(_prof(False), mode="parallel",
                                      calib_fov=FOV)
    checks.append(("⑧-1 关+带 fov 标定 -> legacy_follow=True 且 warning 含「沿用标定」",
                   r8.get("legacy_follow") is True
                   and "沿用标定" in r8.get("warning", "")))
    checks.append(("⑧-2 端到端决策链：最终 fov 仍是标定读出的对象、没被置 None",
                   _case8()))
    r8n = fov_policy.resolve_fov_apply(_prof(False), mode="parallel",
                                       calib_fov=None)
    checks.append(("⑧-3 关+标定 fov=null -> 无 legacy_follow、无 warning",
                   r8n.get("legacy_follow") is False
                   and r8n.get("warning", "") == ""))
    checks.append(("⑧-4 legacy_follow=True 时 error is None（不拒绝、照旧录）",
                   r8.get("legacy_follow") is True and r8["error"] is None))

    # ⑨ 反向自证：删掉第 2 条整段 -> ②⑧ 变红；字节级还原 sha256 一致
    _src = ROOT / "stereo" / "fov_policy.py"
    _orig = _src.read_bytes()
    _sha_before = _hashlib.sha256(_orig).hexdigest()
    _nl = b"\r\n" if b"\r\n" in _orig else b"\n"
    _text_lf = (_orig.decode("utf-8").replace("\r\n", "\n")
                if _nl == b"\r\n" else _orig.decode("utf-8"))
    # 第 2 条拆成两支后整段变长，改用起止边界定位「第 2 条整段」更稳：
    _start_mark = "    # 2. 未启用 —— 反向自证锚点：删掉这一整段，判据 2 / 8 必须变红"
    _end_mark = "    # 3. 启用但档案没有 FOV 记录"
    _i0 = _text_lf.find(_start_mark)
    _i1 = _text_lf.find(_end_mark, _i0 + 1) if _i0 >= 0 else -1
    _hit = 1 if (_i0 >= 0 and _i1 > _i0) else 0
    checks.append(("⑨ 反向自证锚点（第 2 条整段）命中", _hit == 1))
    _sha_restored = False
    if _hit == 1:
        _mut_lf = _text_lf[:_i0] + _text_lf[_i1:]
        _mut_bytes = (_mut_lf.replace("\n", "\r\n") if _nl == b"\r\n"
                      else _mut_lf).encode("utf-8")
        checks.append(("⑨ 变异注入后源码字节确实变了", _mut_bytes != _orig))
        try:
            _src.write_bytes(_mut_bytes)
            _importlib.reload(fov_policy)
            # 删掉第 2 条后：② 落到第 5 条（有 error）、⑧ 落到第 7 条
            # （apply=True、无 legacy_follow），两条都变红。
            checks.append(("⑨ 删掉第 2 条后 ② 变红（不再无 error）",
                           not _case2()))
            checks.append(("⑨ 删掉第 2 条后 ⑧ 变红（不再 legacy_follow）",
                           not _case8()))
        finally:
            _src.write_bytes(_orig)                 # 字节级还原
            _importlib.reload(fov_policy)
            _sha_after = _hashlib.sha256(_src.read_bytes()).hexdigest()
            _sha_restored = (_sha_after == _sha_before)
    checks.append(("⑨ 字节级还原后 sha256 一致", _sha_restored))
    checks.append(("⑨ 还原后 ② 复绿", _case2()))
    checks.append(("⑨ 还原后 ⑧ 复绿", _case8()))

    # ⑩ angled 忽略警告抽成纯函数：真调（不许用源码字符串匹配替代）
    from types import SimpleNamespace as _NS
    _fake_active = _NS(active=True, to_dict=lambda: {"crop_side": "right"})
    _fake_inactive = _NS(active=False, to_dict=lambda: {"crop_side": "none"})
    _ignore = _rs.angled_ignore_message(_fake_active)
    checks.append(("⑩ angled_ignore_message(active=True) 含「忽略」",
                   isinstance(_ignore, str) and "忽略" in _ignore))
    checks.append(("⑩ angled_ignore_message(None) == 空串",
                   _rs.angled_ignore_message(None) == ""))
    checks.append(("⑩ angled_ignore_message(active=False) == 空串",
                   _rs.angled_ignore_message(_fake_inactive) == ""))
    # 配套：angled+默认关+带 fov 标定 -> 决策为「未启用、无 error」，决策块进
    # apply=False 分支但（改后）不提前置 fov=None，原 angled 分支才拿得到带
    # active 的 fov 并打出「忽略」警告。
    r_ang = fov_policy.resolve_fov_apply(_prof(False), mode="angled", calib_fov=FOV)
    checks.append(("⑩ angled+默认关+带 fov 标定 -> apply=False 且无 error（不拒绝录制）",
                   r_ang["apply"] is False and r_ang["error"] is None))

    ok = True
    for label, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {label}")
        ok = ok and passed
    return ok


# --------------------------------------------------------------------------
# 11. 双目实时入口（apps/run_realtime_stereo.py）：纯函数 + 注入式端到端
# --------------------------------------------------------------------------
RT_DT = 1.0 / 30.0        # 虚拟帧间隔（33.3 ms）——"一帧"就是判据里的这个数
RT_Z = 3.6                # 球所在深度（米）：f=850、B=150 mm 时 1 px ≈ 102 mm
RT_SPEED_PX = 8.0         # 要求：球在左图匀速右移 8 px/帧
#: 对应的三维横移速度：u 每帧 8 px => dX = 8·Z/f，再折成 米/秒
RT_V0 = RT_SPEED_PX * (1.0 / RT_DT) * RT_Z / float(K1[0, 0])


def _rt_xyz(t, *, v0=RT_V0, a=0.0, x0=-1.0):
    """真值三维位置。**同一个时钟**贯穿渲染与判据（t=0 是第 0 对帧）。"""
    return np.array([x0 + v0 * t + 0.5 * a * t * t, 0.0, RT_Z], np.float64)


def _rt_project(xyz, side):
    """真值 -> 某一路像素 (u, v, r)，与 `selftest_stereo_branch` 的装置真值同源。"""
    K = K1 if side == "left" else K2
    p = np.asarray(xyz, np.float64)
    if side == "right":
        p = R_REL @ p + T_REL.ravel()
    u = K[0, 2] + K[0, 0] * p[0] / p[2]
    v = K[1, 2] + K[1, 1] * p[1] / p[2]
    r = K[0, 0] * (BALL_DIAMETER_M / 2.0) / p[2]
    return float(u), float(v), float(r)


def _rt_frame(u, v, r):
    """白圆 + 深灰底。用合成画面而不是真相机：判据要能自动跑、且真值已知。"""
    img = np.full((HEIGHT, WIDTH, 3), 60, np.uint8)
    cv2.circle(img, (int(round(u)), int(round(v))), int(round(r)),
               (255, 255, 255), -1, cv2.LINE_AA)
    return img


def _rt_pairs(n, *, skew_ms=0.0, a=0.0, v0=RT_V0, x0=-1.0):
    """造 n 对帧。右路拍的是 ``t + skew`` 时刻的**真实场景**，时间戳也如实带上。

    这一点是判据 ⑤ 成立的前提：如果右路还按 t 渲染、却把时间戳写成 t+skew，
    那"补偿"只是在跟一个不存在的偏差较劲。
    """
    out = []
    for i in range(n):
        t = i * RT_DT
        t_l = 100.0 + t                     # 绝对值不重要：两路同钟、单调递增即可
        t_r = t_l + skew_ms / 1000.0
        ul, vl, rl = _rt_project(_rt_xyz(t, v0=v0, a=a, x0=x0), "left")
        ur, vr, rr = _rt_project(
            _rt_xyz(t + skew_ms / 1000.0, v0=v0, a=a, x0=x0), "right")
        out.append((_rt_frame(ul, vl, rl), _rt_frame(ur, vr, rr),
                    t_l, t_r, skew_ms))
    return out


class _RTPairSource:
    """假帧源：只实现 `run_loop` 真正用到的那几个方法。

    判据不碰相机的关键就在这里 —— 主循环跑的是**生产代码**，只有帧源是假的。
    """

    kind = "camera"

    def __init__(self, pairs, stalled=()):
        self._pairs, self._i = list(pairs), 0
        self._stalled = set(stalled)
        self.dropped_pairs = 0

    def read_pair(self, max_skew_ms=None):
        if self._i >= len(self._pairs):
            return False, None, None, 0.0, 0.0, None, 0, 0
        fl, fr, t_l, t_r, sk = self._pairs[self._i]
        self._i += 1
        return True, fl, fr, t_l, t_r, sk, self._i, self._i

    def stalled_sides(self, timeout=1.0):
        # read_pair 里已 +1，这里问的是**刚返回那一对**
        return (self._i - 1 in self._stalled, False)

    def release(self):
        pass


def _rt_run(pairs, *, skew_fix=True, stalled=(), max_skew_ms=60.0,
            a=0.0, v0=RT_V0, x0=-1.0, shift_s=0.0,
            session_root=None, out: dict | None = None):
    """跑一遍生产主循环，返回 ``(stats, rec, rms_mm, errs_mm)``。

    ``shift_s``：真值取哪个时刻。三角化按约定标注在**较晚**时刻，但补偿之后
    这个点描述的是**较早**时刻的位置，所以 skew 用例传 ``-skew/1000``。
    前 5 对不进 RMS：tracker 起步期 `hits<3` 按设计不补偿（门禁本身在判据 ②/⑦）。

    ``session_root`` 给了就**走真正的落盘链路**（`_SessionRecorder` + `save`），
    并把 ``{"dir": 会话目录或 None, "recorder": ...}`` 填进 ``out``。
    """
    import apps.run_realtime_stereo as rts

    pipe_l, pipe_r = make_pipeline(K1), make_pipeline(K2)
    rig = build_rig()
    rec = StereoReconstructor(rig, MatchConfig(max_epipolar_px=2.5),
                              allow_zero_match=True)
    recorder = None
    if session_root is not None:
        recorder = rts._SessionRecorder(
            root=session_root, mode="parallel", image_size=(WIDTH, HEIGHT),
            fps=30.0, calib_base="<test>", rig=rig)
    # 传的是**零参工厂**（`source_factory`），不是实例：同一进程要跑好几轮，
    # 每轮必须拿到一对新帧源。顺带把"工厂这条路"也真跑一遍。
    stats = rts.run_loop(lambda: _RTPairSource(pairs, stalled=stalled),
                         pipe_l, pipe_r, rec, skew_fix=skew_fix,
                         max_skew_ms=max_skew_ms, max_frames=len(pairs),
                         display_width=800, panel_on=False, log_every=0,
                         frame_sink=(recorder.write_frame if recorder
                                     else None),
                         collect_rows=recorder is not None)
    if out is not None:
        out["stats"] = stats
        if recorder is not None:
            out["recorder"] = recorder
            out["dir"] = recorder.save(stats)
    elif recorder is not None:
        recorder.save(stats)
    errs = []
    for pt in rec.trajectory:
        if pt.t < 5 * RT_DT:
            continue
        truth = _rt_xyz(pt.t + shift_s, v0=v0, a=a, x0=x0)
        errs.append(float(np.linalg.norm(pt.xyz - truth)) * 1000.0)
    rms = float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")
    return stats, rec, rms, errs


def test_realtime_stereo() -> bool:
    """双目实时入口（`apps/run_realtime_stereo.py`）的判决逻辑与端到端。

    为什么这一段必须存在：这个入口是**唯一**会被真机反复使用的双目程序，
    而它过去不存在 —— 单目实时跑得再顺，也证明不了"两路 + 时间差 + 三角测量"
    这条链在实时节奏下成立。下面每条都**真跑**（不碰相机、不做源码字符串匹配）：

      ① `skew_correct` / `skew_dt_for_side`：匀速位移正好 3.3 px、dt=0 原样、
         负 dt 反向、**只有较晚那一路**有位移；
      ② `should_drop`：不超限不丢 / 超 skew 丢且原因含 skew / 任一路 stalled
         丢且原因含 stalled；
      ③ `compose_hud` 同时含 match/res/disp/skew/drop 五个键（取不到写 n/a）；
      ④ 注入式端到端：60 对、33.3 ms/帧 —— 三维点 > 50 且不崩；
      ⑤ 补偿有效性：同一组数据右路延后一帧，开/关补偿各跑一遍，
         **补偿后 RMS 必须显著小于补偿前**（两个数都打出来）；
      ⑥ 加速边界：5 m/s²、同样一帧 skew，量出补偿后的残差（如实报，不粉饰）；
      ⑦ stalled / |skew| 超限：该对不产出三维点，且 dropped 计数增加；
      ⑧ 反向自证：删掉 `should_drop` 的 stalled 分支 -> ⑦ 的 stalled 半边变红；
         字节级还原后 sha256 与改前一致，且还原后复绿。
      ⑩ 分发：`make_reconstructor` 按标定模式选重建器（六个组合逐个真调）；
      ⑪ `none_raw + parallel` 仍被拒（文案含「重标」「angled」）；
      ⑫ 大角度重建器真能出点（合成真值 3 点、误差 < 1 mm、match_rate 有值）；
      ⑬ 大角度 HUD 的 `disp` 写 n/a，收尾用「极线距离 / 交会角」；
         `run_loop` 的模式是从重建器类型读出来的（不是写死 parallel）；
      ⑭ 反向自证：删掉 `make_reconstructor` 的 angled 分支 -> ⑩ 变红，
         字节级还原后 sha256 一致、复绿。
      ⑮ 退出落会话：60 对 -> 目录 + `session.json`（`load_session` 可读、
         `left.frame_count > 0`）；
      ⑯ `traj3d.csv` 表头 == `CSV_HEADER`、行数 == 三维点数；
      ⑰ `derived/seeds.json` 的 `source == "realtime"`；
      ⑱ `derived/visualization.json` 的 schema/version/trajectory 合规；
      ⑲ `view_3d.load_view` 能加载、数据管理判为 `traj`（✓）；
      ⑳ 三维点 < 30 不落盘（目录都不留）；
      ㉑ CSV 的 `frame` 与 `annotated.mp4` 的帧序号一一对应（含被丢弃的空洞）；
      ㉒ 反向自证：把 `frame_index += 1` 挪进 `if pt is not None` ->
         ㉑ 变红；字节级还原后 sha256 一致并复绿。
    """
    print("----- 11. 双目实时入口：纯函数 + 注入式端到端 -----")
    import hashlib as _hashlib
    import importlib as _importlib

    import apps.run_realtime_stereo as rts

    checks: list[tuple[str, bool]] = []

    # ---- ① skew_correct / skew_dt_for_side ----
    u2, v2 = rts.skew_correct(100.0, 200.0, 100.0, 0.0, 0.033)
    checks.append(("① 匀速 vx=100px/s、dt=0.033s -> 位移正好 3.3 px",
                   abs(u2 - 103.3) < 1e-6 and abs(v2 - 200.0) < 1e-9))
    zero = rts.skew_correct(12.5, -3.0, 999.0, -999.0, 0.0)
    checks.append(("① dt=0 原样返回（一个浮点噪声都不加）", zero == (12.5, -3.0)))
    u3, v3 = rts.skew_correct(100.0, 200.0, 100.0, 50.0, -0.033)
    checks.append(("① 负 dt 反向（-3.3 px / -1.65 px）",
                   abs(u3 - 96.7) < 1e-6 and abs(v3 - 198.35) < 1e-6))
    checks.append(("① skew>0（右路更晚）：右路 dt=-12 ms、左路 0",
                   abs(rts.skew_dt_for_side(12.0, "right") + 0.012) < 1e-12
                   and rts.skew_dt_for_side(12.0, "left") == 0.0))
    checks.append(("① skew<0（左路更晚）：左路 dt=-12 ms、右路 0",
                   abs(rts.skew_dt_for_side(-12.0, "left") + 0.012) < 1e-12
                   and rts.skew_dt_for_side(-12.0, "right") == 0.0))

    # ---- ② should_drop 三支 ----
    d0, r0 = rts.should_drop(12.3, False, False, 60.0)
    d1, r1 = rts.should_drop(100.0, False, False, 60.0)
    d2, r2 = rts.should_drop(12.3, True, False, 60.0)
    d3, r3 = rts.should_drop(12.3, False, True, 60.0)
    checks.append(("② |skew|<=上限且不 stalled -> 不丢（原因空串）",
                   d0 is False and r0 == ""))
    checks.append(("② 超限 -> 丢，原因含 skew", d1 is True and "skew" in r1))
    checks.append(("② 任一路 stalled -> 丢，原因含 stalled（左右都算）",
                   d2 is True and "stalled" in r2 and d3 is True and "stalled" in r3))

    # ---- ③ compose_hud 五个键 ----
    line = rts.compose_hud(match_rate=0.87, residual_px=0.8, disparity_px=42.1,
                           skew_ms=12.3, dropped=3)
    keys = ("match", "res", "disp", "skew", "drop")
    checks.append(("③ compose_hud 含 match/res/disp/skew/drop 五个键",
                   all(k in line for k in keys)
                   and "87%" in line and "42.1" in line and "+12.3" in line))
    empty = rts.compose_hud()
    checks.append(("③ 取不到真值时写 n/a（不拿 0 冒充测量值）",
                   all(k in empty for k in keys) and "n/a" in empty))

    # ---- ④ 注入式端到端：60 对、8 px/帧 ----
    stats, rec, rms_clean, _ = _rt_run(_rt_pairs(60))
    print(f"     ④ 60 对无 skew：三维点 {stats.points}，配对 {rec.n_matched}/"
          f"{rec.n_frames}，RMS {rms_clean:.1f} mm（{RT_Z} m 处）")
    checks.append((f"④ 注入 60 对 -> 三维点 {stats.points} > 50 且不崩",
                   stats.points > 50 and rec.n_matched > 50))
    checks.append(("④ HUD 的 skew/drop 与真实统计一致（不是写死的）",
                   "skew" in rts.compose_hud(match_rate=rec.match_rate,
                                            skew_ms=stats.skews[-1],
                                            dropped=stats.dropped)
                   and stats.dropped == 0))

    # ---- ⑤ 补偿有效性：右路整体延后一帧 ----
    skew_ms = 1000.0 * RT_DT
    st_off, _, rms_off, _ = _rt_run(_rt_pairs(60, skew_ms=skew_ms), skew_fix=False,
                                    shift_s=-skew_ms / 1000.0)
    st_on, _, rms_on, _ = _rt_run(_rt_pairs(60, skew_ms=skew_ms), skew_fix=True,
                                  shift_s=-skew_ms / 1000.0)
    print(f"     ⑤ skew +{skew_ms:.1f} ms：关补偿 RMS {rms_off:.1f} mm"
          f" -> 开补偿 {rms_on:.1f} mm（{rms_off / max(rms_on, 1e-9):.1f}×）")
    checks.append((f"⑤ 补偿后 RMS {rms_on:.1f} mm 显著小于补偿前 {rms_off:.1f} mm",
                   rms_on * 3.0 < rms_off and st_on.points > 50))

    # ---- ⑥ 加速边界（5 m/s²）----
    acc_kw = dict(a=5.0, v0=0.0, x0=-1.2)
    _, _, rms_acc_off, _ = _rt_run(_rt_pairs(30, skew_ms=skew_ms, **acc_kw),
                                   skew_fix=False, shift_s=-skew_ms / 1000.0,
                                   **acc_kw)
    st_acc, _, rms_acc, _ = _rt_run(_rt_pairs(30, skew_ms=skew_ms, **acc_kw),
                                    skew_fix=True, shift_s=-skew_ms / 1000.0,
                                    **acc_kw)
    print(f"     ⑥ 5 m/s² + skew {skew_ms:.1f} ms：关补偿 {rms_acc_off:.1f} mm"
          f" -> 开补偿 {rms_acc:.1f} mm（点 {st_acc.points}）")
    # 阈值按实测设（本次 30 对实测约 65 mm，留 ~4 倍余量）：这条只防退化，
    # 不假装"加速段也精确" —— 一阶补偿的残差本来就随加速度线性增长。
    checks.append((f"⑥ 加速段补偿后残余 {rms_acc:.1f} mm < 250 mm（一阶补偿的边界）",
                   rms_acc < 250.0 and rms_acc < rms_acc_off))

    # ---- ⑦ stalled / skew 超限都不产出三维点 ----
    n = 30
    st_st, rec_st, _, _ = _rt_run(_rt_pairs(n), stalled={10, 11})
    print(f"     ⑦ stalled 两对：dropped {st_st.dropped} "
          f"{dict(st_st.drop_reasons)}，三维点 {st_st.points}/{n}")
    checks.append(("⑦ 任一路 stalled -> 该对不出三维点、dropped+2",
                   st_st.dropped == 2 and st_st.points == n - 2
                   and all("stalled" in k for k in st_st.drop_reasons)))
    pairs_bad = _rt_pairs(n)
    pairs_bad[5] = (pairs_bad[5][0], pairs_bad[5][1],
                    pairs_bad[5][2], pairs_bad[5][3], 100.0)
    st_bad, _, _, _ = _rt_run(pairs_bad, max_skew_ms=60.0)
    print(f"     ⑦ |skew| 100 ms 一对：dropped {st_bad.dropped} "
          f"{dict(st_bad.drop_reasons)}，三维点 {st_bad.points}/{n}")
    checks.append(("⑦ |skew| 超限 -> 该对不出三维点、dropped+1",
                   st_bad.dropped == 1 and st_bad.points == n - 1
                   and any("skew" in k for k in st_bad.drop_reasons)))

    # ---- ⑧ 反向自证：删掉 should_drop 的 stalled 分支 ----
    src_path = ROOT / "apps" / "run_realtime_stereo.py"
    orig = src_path.read_bytes()
    sha_before = _hashlib.sha256(orig).hexdigest()
    nl = b"\r\n" if b"\r\n" in orig else b"\n"
    text = orig.decode("utf-8")
    start_mark = "    if stalled_l or stalled_r:"
    end_mark = "    if skew_ms is None:"
    i0 = text.find(start_mark)
    i1 = text.find(end_mark, i0 + 1) if i0 >= 0 else -1
    hit = i0 >= 0 and i1 > i0
    checks.append(("⑧ 反向自证锚点（should_drop 的 stalled 分支）命中", hit))
    restored = False
    if hit:
        mutated = (text[:i0] + text[i1:]).encode("utf-8")
        checks.append(("⑧ 变异注入后源码字节确实变了", mutated != orig))
        try:
            src_path.write_bytes(mutated)
            _importlib.reload(rts)
            d_mut, r_mut = rts.should_drop(12.3, True, False, 60.0)
            checks.append(("⑧ 删掉 stalled 分支后 ⑦ 的 stalled 判据变红",
                           d_mut is False and "stalled" not in r_mut))
            st_mut, _, _, _ = _rt_run(_rt_pairs(n), stalled={10, 11})
            checks.append(("⑧ 删掉后 stalled 两对不再被丢（dropped 0）",
                           st_mut.dropped == 0 and st_mut.points == n))
        finally:
            src_path.write_bytes(orig)              # 字节级还原
            _importlib.reload(rts)
            restored = (_hashlib.sha256(src_path.read_bytes()).hexdigest()
                        == sha_before)
    checks.append(("⑧ 字节级还原后 sha256 与改前一致", restored))
    checks.append(("⑧ 还原后 ⑦ 的 stalled 判据复绿",
                   rts.should_drop(12.3, True, False, 60.0)[0] is True))

    # ---- ⑨ 并排画布：HUD 独占顶栏，小窗**独占右侧一列**（不自我遮挡）----
    vl = np.zeros((120, 160, 3), np.uint8)
    vr = np.zeros((120, 160, 3), np.uint8)
    panel = np.full((40, 60, 3), 7, np.uint8)
    canvas = rts.compose_canvas(vl, vr, "stereo  match 100%", panel=panel)
    # 2026-11-17 用户截图：「画面自己遮挡自己」—— 旧版把小窗贴在 `w-pw`，
    # 正好压在右相机画面上。现在宽度必须是**两路之和 + 小窗宽**。
    checks.append(("⑨ compose_canvas：宽 = 两路之和 + 小窗宽（小窗独占一列）",
                   canvas.shape[1] == 320 + panel.shape[1]
                   and canvas.shape[0] > 120))
    pasted = int(((canvas == 7).all(axis=2)).sum())
    checks.append((f"⑨ 小窗真的贴进了画布（贴入 {pasted} 像素 = 40x60）",
                   pasted >= 40 * 60))
    # 画面区域（顶栏以下、前 320 列）不许出现小窗像素 —— 这才是"不遮挡"的定义。
    body_area = canvas[canvas.shape[0] - vl.shape[0]:, :320]
    leaked = int(((body_area == 7).all(axis=2)).sum())
    checks.append((f"⑨ 左右画面区域内没有小窗像素（泄漏 {leaked}）", leaked == 0))
    # 提示与指标**同一行**（2026-11-17 用户：录制帧的回显会把第二行顶掉）。
    with_hint = rts.compose_canvas(vl, vr, "hud", panel=panel, hint="q 退出 · 空格 暂停")
    checks.append(("⑨ 给了 hint 不增高（始终单行，提示右对齐）",
                   with_hint.shape == canvas.shape))
    # 提示必须画在**右半边**：左半边是指标，挤在一起就等于没分开。
    top = with_hint[:with_hint.shape[0] - vl.shape[0]]
    right_half = top[:, canvas.shape[1] // 2:]
    left_half = top[:, :canvas.shape[1] // 2]
    lit_r = int((right_half != 24).any(axis=2).sum())
    checks.append((f"⑨ 提示真的画在顶栏右半边（右侧亮点 {lit_r}）", lit_r > 20))
    # 角标（瞬时回显）也不许改画布尺寸。
    with_badge = rts.compose_canvas(vl, vr, "hud", panel=panel, badge="▸ 已暂停")
    checks.append(("⑨ 角标不改变画布尺寸（不占顶栏）",
                   with_badge.shape == canvas.shape))
    body_zone = with_badge[with_badge.shape[0] - vl.shape[0]:, :vl.shape[1]]
    checks.append(("⑨ 角标画在画面左上角（画面区有角标像素）",
                   int((body_zone != 0).any(axis=2).sum()) > 50))

    # ---- ⑨b 按键必须有**画面回显**（2026-11-17 用户："按下了没有提示很可恶"）----
    ui = object.__new__(rts._Ui)
    ui._toast = ""
    ui._toast_until = 0.0
    ui._rec_frames = 0
    idle_txt, idle_badge = ui.hint_line(), ui.badge_line()
    ui.toast("已暂停")
    hot_badge = ui.badge_line()
    hot_hint = ui.hint_line()
    ui.toast("这条已过期", ttl=0.0)
    exp_badge = ui.badge_line()
    checks.append(("⑨b 没有回显时角标为空、提示照常在",
                   idle_badge == "" and idle_txt == rts.REALTIME_HINT))
    checks.append((f"⑨b toast 之后角标出现（{hot_badge!r}）",
                   hot_badge.startswith("▸ ") and "已暂停" in hot_badge))
    checks.append(("⑨b 回显**不顶掉**按键提示（两者同时存在）",
                   hot_hint == rts.REALTIME_HINT))
    checks.append(("⑨b ttl 到期后角标消失（不会一直挂着）", exp_badge == ""))
    # 结构：六个会改变状态的键都要回显（p/r 这类"看不见结果"的键最容易漏）。
    _src = rts.__file__
    _body = open(_src, encoding="utf-8").read()
    _i = _body.index("def _handle_key")
    _j = _body.index("def _cycle_method", _i)
    _seg = _body[_i:_j]
    missing = [k for k in ('ord(" ")', 'ord("r")', 'ord("3")') if k not in _seg]
    checks.append((f"⑨b 空格/r/3 三个键都调了 toast（缺 {missing}）",
                   not missing and _seg.count("self.toast(") >= 3))
    checks.append(("⑨b c/s/v 三条路都有 toast",
                   all("self.toast(" in _body[_body.index(f"def {fn}"):]
                       for fn in ("_cycle_method", "_toggle_record", "_snapshot"))))

    # ---- ⑨c 主程序**启动路径**冒烟（不需要相机）----
    # 2026-11-17 用户点「实时 3D」→ 直接 `UnboundLocalError: recorder`：
    # 我把 `ui.recorder = recorder` 写在了 `recorder = None` 之前。
    # ⑨/⑨b 全绿也没用 —— 那些判据只跑 `run_loop`，压根没进 `main()`。
    # 这条把 `resolve_rig` / `open_pair_source` / `make_pipeline` / `_Ui` /
    # `run_loop` 全换成桩，真调一次 `main()`：**启动路径有人走过**。
    import contextlib as _ctx
    import io as _io
    import types as _types

    _fake_src = _types.SimpleNamespace(
        is_live=True, pair_fps=30.0, resolved_size=(1920, 1080),
        image_size=(1920, 1080), report_warnings=lambda: None,
        release=lambda: None,
        read_pair=lambda **k: (False, None, None, 0.0, 0.0, None, 0, 0))
    _fake_rig = _types.SimpleNamespace(
        image_size=(1920, 1080), stereo_mode="parallel",
        rectification="stereo_rectify", baseline_mm=66.5, angle_deg=4.5,
        quality=None)
    _names = ("resolve_rig", "open_pair_source", "make_pipeline", "_Ui", "run_loop")
    _orig = {k: getattr(rts, k) for k in _names}
    _rc, _err = None, ""
    try:
        rts.resolve_rig = lambda calib=None: (_fake_rig, "")
        rts.open_pair_source = lambda args: _fake_src
        rts.make_pipeline = lambda cfg, rig, side: _types.SimpleNamespace(
            set_detect_width=lambda w: None, reset=lambda: None, detector=None)
        rts._Ui = lambda *a, **k: _types.SimpleNamespace(
            recorder=None, panel_on=True, hint_line=lambda: "h",
            badge_line=lambda: "", show=lambda *a: True,
            write_frame=lambda *a: None, toast=lambda *a, **kw: None,
            close=lambda: None)
        rts.run_loop = lambda *a, **kw: rts.LoopStats()
        with _ctx.redirect_stdout(_io.StringIO()):
            _rc = rts.main(["--left", "left", "--right", "right", "--no-3d"])
    except BaseException as _exc:                        # noqa: BLE001
        _err = f"{type(_exc).__name__}: {_exc}"
    finally:
        for _k, _v in _orig.items():
            setattr(rts, _k, _v)
    checks.append((f"⑨c main() 启动路径能走完（rc={_rc}，err={_err or '无'}）",
                   _rc == 0 and not _err))

    # ---- ⑨d 两档录制：`v` 只出检测画面，`b` 额外出两路**原始**画面 ----
    # 2026-11-17 用户要求："按下 v 只有检测视频，再加一个键，同时录检测视频和原始视频"。
    import pathlib as _pl
    import tempfile as _tf2

    ui2 = object.__new__(rts._Ui)
    ui2.args = _types.SimpleNamespace(record="")
    ui2.fps = 10.0
    ui2.record_wanted = False
    ui2.record_mode = ""
    ui2.record_path = None
    ui2.writer = None
    ui2.raw_writers = []
    ui2.raw_paths = ()
    ui2._rec_frames = 0
    ui2.recorder = None
    ui2._toast = ""
    ui2._toast_until = 0.0
    ui2.toast = lambda *a, **k: None

    def _frames_of(path) -> tuple[int, tuple[int, int]]:
        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
        return n, size

    with _tf2.TemporaryDirectory() as _td:
        canvas_f = np.zeros((60, 160, 3), np.uint8)
        fake_stats = _types.SimpleNamespace(
            last_left=np.zeros((120, 160, 3), np.uint8),
            last_right=np.zeros((120, 160, 3), np.uint8))

        # 两档各用**自己的**基名（命令行 --record 固定时，第二档会覆盖同名文件，
        # 那是 `--record` 的既有语义，不是这条判据要测的东西）。
        ui2.args.record = str(_pl.Path(_td) / "rec_v.mp4")
        ui2._toggle_record("canvas")
        for _ in range(3):
            ui2.write_frame(canvas_f, fake_stats)
        ui2._toggle_record("canvas")
        after_v = sorted(x.name for x in _pl.Path(_td).iterdir())

        ui2.args.record = str(_pl.Path(_td) / "rec_b.mp4")
        ui2._toggle_record("both")
        for _ in range(3):
            ui2.write_frame(canvas_f, fake_stats)
        ui2._toggle_record("both")
        after_b = sorted(x.name for x in _pl.Path(_td).iterdir())

        n_left, sz_left = _frames_of(_pl.Path(_td) / "rec_b_raw_left.mp4")
        n_right, sz_right = _frames_of(_pl.Path(_td) / "rec_b_raw_right.mp4")
        n_canvas, sz_canvas = _frames_of(_pl.Path(_td) / "rec_b.mp4")

    checks.append((f"⑨d `v` 只写检测画面（v 后文件 {after_v}）",
                   after_v == ["rec_v.mp4"]))
    checks.append((f"⑨d `b` 额外写出两路原始（b 后 {len(after_b)} 个文件）",
                   {"rec_b.mp4", "rec_b_raw_left.mp4", "rec_b_raw_right.mp4"}
                   <= set(after_b)))
    checks.append((f"⑨d 原始两路是真 120x160（left={sz_left} right={sz_right}，"
                   f"检测画面={sz_canvas}）",
                   sz_left == (160, 120) and sz_right == (160, 120)
                   and sz_canvas == (160, 60)))
    checks.append((f"⑨d 三路帧数一致且都是 3（canvas={n_canvas} L={n_left} R={n_right}）",
                   n_canvas == n_left == n_right == 3))

    # ---------------------------------------------------------------- ⑩-⑬
    # 大角度装置（2026-11-16 用户裁决："平行和大角度各有一个实时 3D"）。
    # 这里不重跑检测链路，只钉三件事：**分发正确**、**错标定仍拒绝**、
    # **大角度重建器真能出点**。检测->跟踪->三角化在平行那段已覆盖。
    from types import SimpleNamespace as _NS

    from stereo.angled import AngledReconstructor as _AngledRec
    from stereo.triangulate import StereoReconstructor as _ParRec

    def _fake_rig(mode, rect):
        """假装置：K=I、R=+90°(绕 Y)、T=(0.4,0,0) —— 与 `stereo/angled.py`
        的 `_self_test()` 完全同一套几何（那边已经自测过它是共视的）。"""
        K = np.eye(3, dtype=np.float64)
        D = np.zeros(5, dtype=np.float64)
        R = cv2.Rodrigues(np.array([0.0, np.deg2rad(90.0), 0.0]))[0]
        T = np.array([0.4, 0.0, 0.0])
        return _NS(K1=K, D1=D, K2=K, D2=D, R=R, T=T, F=None,
                   stereo_mode=mode, rectification=rect,
                   quality={"epipolar_rms": 0.62},
                   disparity_sign=lambda: 1,
                   meta={"stereo_mode": mode, "rectification": rect})

    # ⑩ 六个分发组合
    combos = [
        ("angled + none_raw", "angled", "none_raw", _AngledRec),
        ("angled + stereo_rectify", "angled", "stereo_rectify", _AngledRec),
        ("parallel + stereo_rectify", "parallel", "stereo_rectify", _ParRec),
        ("parallel + 缺 rectification 字段", "parallel", "", _ParRec),
    ]
    combo_ok = True
    for label, mode, rect, want in combos:
        got = rts.make_reconstructor(_fake_rig(mode, rect))
        combo_ok = combo_ok and isinstance(got, want)
        print(f"     ⑩ {label:34s} -> {type(got).__name__}")
    # 第五种：只有 meta 里写 angled（对象没有 stereo_mode 属性）
    meta_only = _fake_rig("angled", "none_raw")
    del meta_only.stereo_mode
    got_meta = rts.make_reconstructor(meta_only)
    combo_ok = combo_ok and isinstance(got_meta, _AngledRec)
    print(f"     ⑩ meta 里才是 angled{' '*22} -> {type(got_meta).__name__}")
    checks.append(("⑩ angled -> AngledReconstructor；parallel -> StereoReconstructor"
                   "（含 meta-only）", combo_ok))

    # ⑪ none_raw + parallel 仍拒绝，且文案给了「重标 / angled」两条出路
    refused = None
    try:
        rts.make_reconstructor(_fake_rig("parallel", "none_raw"))
    except Exception as exc:                        # noqa: BLE001  判据要的是"拒绝"
        refused = exc
    text = str(refused or "")
    print(f"     ⑪ none_raw + parallel -> {type(refused).__name__}："
          f"{text.splitlines()[0][:52]}…")
    checks.append(("⑪ none_raw + parallel 被拒（ReconstructionModeError），"
                   "文案含「重标」与「angled」",
                   isinstance(refused, rts.ReconstructionModeError)
                   and "重标" in text and "angled" in text))

    # ⑫ 大角度重建器真能出点（合成一对检测，真值已知）
    a_rig = _fake_rig("angled", "none_raw")
    a_rec = rts.make_reconstructor(a_rig)
    K_a = np.array(a_rig.K1, np.float64)
    D_a = np.array(a_rig.D1, np.float64)

    def _a_project(xyz):
        pts, _ = cv2.projectPoints(np.asarray(xyz, np.float64).reshape(1, 1, 3),
                                   np.zeros(3), np.zeros(3), K_a, D_a)
        return np.asarray(pts, np.float64).reshape(2)

    truth3 = [[-0.20, 0.00, 0.30], [-0.10, 0.06, 0.35], [-0.24, -0.05, 0.25]]
    a_err = 0.0
    for i, p in enumerate(truth3):
        p = np.asarray(p, np.float64)
        pl, pr = _a_project(p), _a_project(a_rig.R @ p + a_rig.T)
        pt = a_rec.update(0.1 * i,
                          Detection(x=float(pl[0]), y=float(pl[1]), radius=5.0),
                          [Detection(x=float(pr[0]), y=float(pr[1]), radius=4.5)])
        if pt is not None:
            a_err = max(a_err, float(np.linalg.norm(pt.xyz - p)))
    rate_a = a_rec.match_rate                 # 是 **property**，不是方法
    print(f"     ⑫ 大角度合成 3 点：配对 {a_rec.n_matched}/{a_rec.n_frames}"
          f"（match_rate {rate_a * 100:.0f}%），最大三维误差 {a_err * 1000:.3f} mm")
    checks.append(("⑫ 大角度重建器至少出 1 个点、match_rate 有值、不抛",
                   a_rec.n_matched >= 1 and 0.0 < rate_a <= 1.0))
    checks.append((f"⑫ 大角度三维误差 {a_err * 1000:.3f} mm < 1 mm（合成无噪声）",
                   a_err < 1e-3))

    # ⑬ HUD / 统计在大角度下**不把角度量塞进 disp**
    hud_a = rts.compose_hud(match_rate=0.87, residual_px=0.8, disparity_px=None,
                            skew_ms=12.3, dropped=3, note="epi 0.9 px  ang 88°")
    print(f"     ⑬ 大角度 HUD：{hud_a}")
    # 断言按**语义**取"disp 后面那个值"，不按空格数匹配（排版一改就假红/假绿）
    disp_field = hud_a.split("disp", 1)[1].split("skew", 1)[0].strip()
    checks.append((f"⑬ 大角度 HUD 的 disp 写 n/a（实测 {disp_field!r}），"
                   "且仍含 match/skew/drop",
                   disp_field == "n/a" and all(
                       k in hud_a for k in ("match", "skew", "drop"))
                   and "epi" in hud_a and "88°" in hud_a))
    st_a = rts.LoopStats(mode="angled", points=3, pairs=3, match_rate=1.0,
                         epipolars=[0.5, 0.9], angles=[86.0, 88.5],
                         skews=[1.0], process_ms=[12.0])
    lines_a = "\n".join(st_a.summary_lines())
    checks.append(("⑬ 大角度收尾用「极线距离 / 交会角」，不出现「视差」",
                   "极线距离" in lines_a and "交会角" in lines_a
                   and "视差" not in lines_a and "重建模式 angled" in lines_a))
    # 空源跑一遍主循环：确认 `run_loop` 真的从 rec 的类型读出模式（不是写死的）
    st_mode = rts.run_loop(lambda: _RTPairSource([]), make_pipeline(K1),
                           make_pipeline(K2), a_rec, max_frames=1,
                           display_width=0, panel_on=False, log_every=0)
    checks.append(("⑬ run_loop 从重建器类型读出 mode=angled（空源也成立）",
                   st_mode.mode == "angled" and st_mode.pairs == 0))

    # ⑬b GUI 的「实时 3D」按钮**不带任何相机参数**（`stereo_realtime_args()` 全默认）。
    # 后端必须能把它解释成"自动成对挑"，否则按钮点一次只得到一句参数错误 =
    # 又一个假入口。这里直接吞下 GUI 真产出的 argv（去掉解释器与脚本两项）。
    import apps._actions as _actions
    gui_argv = _actions.stereo_realtime_args()[2:]
    gui_args = rts.parse_args(gui_argv)
    print(f"     ⑬ GUI 默认 argv {len(gui_argv)} 项 -> auto_cameras="
          f"{gui_args.auto_cameras}, left={gui_args.left!r}, "
          f"detect_width={gui_args.detect_width}")
    checks.append(("⑬ GUI 全默认 argv -> 自动成对挑相机（不再报「请指定 --left」）",
                   gui_args.auto_cameras is True and not gui_args.left
                   and not gui_args.right))

    # ⑭ 反向自证：删掉 make_reconstructor 的 angled 分支
    src_path = ROOT / "apps" / "run_realtime_stereo.py"
    orig = src_path.read_bytes()
    sha_before = _hashlib.sha256(orig).hexdigest()
    text = orig.decode("utf-8")
    start_mark = '    if mode == "angled":'
    end_mark = "    if uncorrected:"
    j0 = text.find(start_mark)
    j1 = text.find(end_mark, j0 + 1) if j0 >= 0 else -1
    hit2 = j0 >= 0 and j1 > j0
    checks.append(("⑭ 反向自证锚点（make_reconstructor 的 angled 分支）命中", hit2))
    restored2 = False
    if hit2:
        mutated = (text[:j0] + text[j1:]).encode("utf-8")
        checks.append(("⑭ 变异注入后源码字节确实变了", mutated != orig))
        try:
            src_path.write_bytes(mutated)
            _importlib.reload(rts)
            try:
                rts.make_reconstructor(_fake_rig("angled", "none_raw"))
                red = False                        # 还能拿到重建器 = 判据没红
            except Exception:                      # noqa: BLE001
                red = True
            checks.append(("⑭ 删掉 angled 分支后 ⑩ 的分发判据变红（拿不到大角度重建器）",
                           red))
            try:
                rts.make_reconstructor(_fake_rig("parallel", "stereo_rectify"))
                still_ok = True
            except Exception:                      # noqa: BLE001
                still_ok = False
            checks.append(("⑭ 删掉后 parallel 仍然正常（只红该红的那条）", still_ok))
        finally:
            src_path.write_bytes(orig)
            _importlib.reload(rts)
            restored2 = (_hashlib.sha256(src_path.read_bytes()).hexdigest()
                         == sha_before)
    checks.append(("⑭ 字节级还原后 sha256 与改前一致", restored2))
    checks.append(("⑭ 还原后 ⑩ 的分发复绿",
                   isinstance(rts.make_reconstructor(_fake_rig("angled", "none_raw")),
                              _AngledRec)))

    # ---------------------------------------------------------------- ⑮-㉒
    # 退出落会话（2026-11-17 用户）：实时跑出来的轨迹要能在「数据管理」里看见、
    # 在「3D 查看器」里打开。这一段走**真正的落盘链路**（`_SessionRecorder` +
    # `save`），会话根用临时目录 —— 不往 `data/` 里塞东西，测完删干净。
    import csv as _csv
    import json as _json
    import shutil as _shutil
    import tempfile as _tempfile

    from apps.run_stereo_video import CSV_HEADER as _CSV_HEADER
    from apps.view_3d import load_view as _load_view
    from apps._workbench import summarize_session as _summarize
    from stereo.sessions import load_session as _load_session
    from stereo.visualization import load_seeds as _load_seeds

    def _frames_of(sdir) -> list[int]:
        """读 CSV 的 frame 列（判据 ㉑ 用）。"""
        with open(Path(sdir) / "derived" / "traj3d.csv", encoding="utf-8",
                  newline="") as f:
            return [int(r["frame"]) for r in _csv.DictReader(f)]

    def _video_frames(sdir) -> int:
        cap = cv2.VideoCapture(str(Path(sdir) / "derived" / "annotated.mp4"))
        try:
            return int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else -1
        finally:
            cap.release()

    sess_root = Path(_tempfile.mkdtemp(prefix="rt_session_probe_"))
    try:
        # ⑮ 60 对（全配对）-> 会话目录 + load_session 能读
        box: dict = {}
        st_sess, _, _, _ = _rt_run(_rt_pairs(60), session_root=sess_root, out=box)
        s_dir = box.get("dir")
        loaded = None
        if s_dir is not None:
            loaded = _load_session(s_dir)[1]
        print(f"     ⑮ 会话 {s_dir.name if s_dir else None}：三维点 {st_sess.points}，"
              f"left.frame_count={getattr(getattr(loaded, 'left', None), 'frame_count', None)}")
        checks.append(("⑮ 跑完 60 对 -> 会话目录存在、load_session 可读、"
                       "left.frame_count > 0",
                       s_dir is not None and Path(s_dir).is_dir() and loaded is not None
                       and loaded.left is not None
                       and loaded.left.frame_count == 60
                       and loaded.right.frame_count == 60))

        # ⑯ CSV 表头（唯一产地）+ 行数 == 三维点数
        csv_path = Path(s_dir) / "derived" / "traj3d.csv"
        with open(csv_path, encoding="utf-8", newline="") as f:
            hdr = next(_csv.reader(f))
            n_rows = sum(1 for _ in f)
        print(f"     ⑯ CSV {n_rows} 行；表头 {hdr[0]}…{hdr[-1]}")
        checks.append((f"⑯ CSV 表头 == CSV_HEADER（{len(_CSV_HEADER)} 列）、"
                       f"行数 {n_rows} == 三维点 {st_sess.points}",
                       hdr == _CSV_HEADER and n_rows == st_sess.points > 0))

        # ⑰ seeds：实时自动取的第一对匹配点
        seeds = _load_seeds(Path(s_dir) / "derived" / "seeds.json")
        print(f"     ⑰ seeds source={seeds['source']!r} seed_frame={seeds['seed_frame']}")
        checks.append(("⑰ derived/seeds.json 存在且 source == 'realtime'",
                       seeds.get("source") == "realtime"
                       and int(seeds.get("seed_frame", -1)) >= 0))

        # ⑱ visualization.json：schema / version / 至少 1 个点
        vis = _json.loads((Path(s_dir) / "derived" / "visualization.json")
                          .read_text(encoding="utf-8"))
        print(f"     ⑱ visualization {vis.get('schema')} v{vis.get('version')}"
              f" points={len(vis.get('trajectory') or [])}")
        checks.append(("⑱ visualization.json schema/version/trajectory 合规",
                       vis.get("schema") == "sphere-stereo-visualization"
                       and int(vis.get("version", 0)) == 1
                       and len(vis.get("trajectory") or []) >= 1))

        # ⑲ 查看器 + 数据管理两处都能"看得见"（只调用，不开窗）
        view = _load_view(s_dir)
        summ = _summarize(s_dir)
        print(f"     ⑲ load_view points={len(view['trajectory'])}，"
              f"workbench status={summ.status!r} matched={summ.n_matched}")
        checks.append(("⑲ 3D 查看器能加载（不开窗）且数据管理判为有三维（status=traj）",
                       len(view["trajectory"]) >= 1 and summ.status == "traj"))

        # ⑳ 点数 < 30 不落盘（目录都不留）
        tiny_root = Path(_tempfile.mkdtemp(prefix="rt_session_tiny_"))
        try:
            box2: dict = {}
            st_tiny, _, _, _ = _rt_run(_rt_pairs(10), session_root=tiny_root,
                                       out=box2)
            left_over = [p.name for p in tiny_root.iterdir()]
            print(f"     ⑳ 10 对（点 {st_tiny.points} < 30）-> dir={box2.get('dir')}，"
                  f"根目录残留 {left_over}")
            checks.append((f"⑳ 三维点 {st_tiny.points} < 30 -> 不落盘、不留目录",
                           box2.get("dir") is None and not left_over))
        finally:
            _shutil.rmtree(tiny_root, ignore_errors=True)

        # ㉑ frame 与视频帧**一一对应**（含被丢弃的空洞）
        gap_root = Path(_tempfile.mkdtemp(prefix="rt_session_gap_"))
        try:
            box3: dict = {}
            st_gap, _, _, _ = _rt_run(_rt_pairs(40), stalled={10, 11},
                                      session_root=gap_root, out=box3)
            gap_dir = box3.get("dir")
            frames = _frames_of(gap_dir) if gap_dir else []
            n_video = _video_frames(gap_dir) if gap_dir else -1
            want = [i for i in range(40) if i not in (10, 11)]
            print(f"     ㉑ 40 对里 stalled 两对：点 {st_gap.points}、视频 {n_video} 帧、"
                  f"frame {frames[:3]}…{frames[-2:]}")
            checks.append((
                f"㉑ CSV frame == 帧序号（跳过被丢的 {[10, 11]}），"
                f"视频 {n_video} 帧 == 40",
                frames == want and n_video == 40))
        finally:
            _shutil.rmtree(gap_root, ignore_errors=True)
    finally:
        _shutil.rmtree(sess_root, ignore_errors=True)

    # ㉒ 反向自证：把 `frame_index += 1` 从"每对"挪进 `if pt is not None`
    src_path = ROOT / "apps" / "run_realtime_stereo.py"
    orig = src_path.read_bytes()
    sha_before = _hashlib.sha256(orig).hexdigest()
    text = orig.decode("utf-8")
    # 行尾按文件实际（仓库里是 CRLF）：用 "\n" 拼锚点会匹配不上，
    # 那会让"锚点命中"这条自己变红 —— 而它红了就等于后面的变异全没跑。
    _nl = "\r\n" if b"\r\n" in orig else "\n"
    inc = f"        stats.frame_index += 1{_nl}"
    anchor = f"                stats.points += 1{_nl}"
    hit4 = inc in text and anchor in text
    checks.append(("㉒ 反向自证锚点（frame_index += 1 / points += 1）命中", hit4))
    restored4 = False
    if hit4:
        # 挪进 `if pt is not None:` 里 —— 缩进必须是 16 空格（同一层），
        # 否则删掉外面那行后 `if collect_rows:` 会顶到一层，直接 IndentationError。
        moved = f"                stats.frame_index += 1{_nl}"
        mutated = text.replace(inc, "", 1).replace(anchor, anchor + moved, 1)
        checks.append(("㉒ 变异注入后源码字节确实变了",
                       mutated.encode("utf-8") != orig))
        try:
            src_path.write_bytes(mutated.encode("utf-8"))
            _importlib.reload(rts)
            mut_root = Path(_tempfile.mkdtemp(prefix="rt_session_mut_"))
            try:
                box4: dict = {}
                _rt_run(_rt_pairs(40), stalled={10, 11},
                        session_root=mut_root, out=box4)
                mut_dir = box4.get("dir")
                mut_frames = _frames_of(mut_dir) if mut_dir else []
                print(f"     ㉒ 变异后 frame {mut_frames[:3]}…{mut_frames[-2:]}"
                      f"（max={max(mut_frames) if mut_frames else None}，应为 39）")
                checks.append(("㉒ 挪进 if pt 后 ㉑ 变红（frame 按点计数、"
                               "与视频帧错位）",
                               bool(mut_frames) and max(mut_frames) != 39))
            finally:
                _shutil.rmtree(mut_root, ignore_errors=True)
        finally:
            src_path.write_bytes(orig)
            _importlib.reload(rts)
            restored4 = (_hashlib.sha256(src_path.read_bytes()).hexdigest()
                         == sha_before)
    checks.append(("㉒ 字节级还原后 sha256 与改前一致", restored4))
    # 还原后再真跑一轮（40 对、丢 1 对 -> 39 点 >= 30 才会落盘），
    # 确认对齐闸门回来了 —— 不是只看源码字节。
    back_root = Path(_tempfile.mkdtemp(prefix="rt_session_back_"))
    try:
        box5: dict = {}
        _rt_run(_rt_pairs(40), stalled={5}, session_root=back_root, out=box5)
        back_dir = box5.get("dir")
        back_frames = _frames_of(back_dir) if back_dir else []
        checks.append(("㉒ 还原后 ㉑ 的对齐闸门复绿（frame 跳过第 5 对、max=39）",
                       back_frames == [i for i in range(40) if i != 5]))
    finally:
        _shutil.rmtree(back_root, ignore_errors=True)

    ok = True
    for label, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {label}")
        ok = ok and passed
    return ok


# ---------------------------------------------------------------------------
# A1/A2/A5/B4：离线产物可归因 + 强制标定标记 + 阈值 CLI + refined F
# ---------------------------------------------------------------------------
_PROBE_COUNTERS = {"n_no_left": 2, "n_no_right": 3, "n_rejected": 4}


def _offline_probe(*, forced: bool = False,
                   min_angle: float | None = None,
                   max_reproj: float | None = None) -> dict:
    """让 `run_stereo_video.main()` 的**真实控制流**跑一段 angled 离线链。

    所有输入都 mock 掉；唯一放行的是被 A1/A2/A5 改动的真实代码路径：
    重建器构造（拿 cfg）、门限打印、quality 写入、强制标定横幅。
    返回的 dict 里有 rc / stdout / quality / 完整 payload / cfg 实际收到值 /
    三产物是否齐。反向自证在子进程里复用本函数，因此不要依赖外层局部状态。
    """
    import contextlib as _contextlib
    import io as _io
    import json as _json
    import shutil as _shutil
    import sys as _sys
    import tempfile as _tempfile
    import types as _types
    from unittest import mock as _mock

    import apps.run_stereo_video as rsv
    import stereo.angled as angled_mod

    tmp = _tempfile.mkdtemp(prefix="dsh_attr_")
    result: dict = {}
    try:
        session_dir = Path(tmp) / "session"
        out_dir = Path(tmp) / "out"
        session_dir.mkdir()
        (session_dir / "derived").mkdir()
        fake_traj = Path(tmp) / "exported_traj3d.json"
        fake_vis = Path(tmp) / "exported_visualization.json"
        fake_traj.write_text(_json.dumps({
            "schema": "sphere-stereo-trajectory", "unit": "m", "n_matched": 40,
            "samples": [],
            # 另一个 worker 刚加的键：A1 只补 quality 子键，绝不能删它。
            "round_trip": {"verdict": "ellipse", "n_pairs": 3,
                           "dy_median_mm": 1.0, "dz_median_mm": 2.0,
                           "sign_consistent": True},
        }), encoding="utf-8")
        fake_vis.write_text(_json.dumps({
            "schema": "sphere-stereo-visualization", "n_points": 0,
            "bounds": {"min": [0, 0, 0], "max": [0, 0, 0]},
        }), encoding="utf-8")

        rig = build_rig()
        calib_path = Path(tmp) / ("forced_calib" if forced else "plain_calib")
        if forced:
            # A2：真的写一份临时 json（不碰 config/ / 真实标定）。
            rig.quality["forced_unusable"] = True
        rig.save(calib_path)

        cam = _types.SimpleNamespace(video="left.mkv", fps_measured=30.0,
                                     fps_requested=30.0)
        sess = _types.SimpleNamespace(
            left=cam, right=cam, calibration=_types.SimpleNamespace(),
            stereo_mode="angled", rig={},
            sync=_types.SimpleNamespace(offset=0, method="flash"))
        frame = np.zeros((540, 960, 3), np.uint8)

        class FakePairSource:
            image_size = (960, 540)

            def __init__(self, *a, **k):
                self.n = 0

            def read(self):
                if self.n >= 6:
                    return (False, None, None)
                self.n += 1
                return (True, frame, frame)

            def report_warnings(self):
                pass

            def describe(self):
                return "fake-index"

            def release(self):
                pass

        class FakePipeline:
            def __init__(self):
                self.detector = _types.SimpleNamespace(debug_snapshot=lambda: {})

            def process(self, f, t):
                return _types.SimpleNamespace(track=None, detection=None)

        captured: dict = {}

        class FakeRec:
            def __init__(self, rig_arg, cfg):
                self.n_matched = 7
                self.n_no_left = _PROBE_COUNTERS["n_no_left"]
                self.n_no_right = _PROBE_COUNTERS["n_no_right"]
                self.n_rejected = _PROBE_COUNTERS["n_rejected"]
                self.n_frames = 16
                self.trajectory = []
                self.warmup_frames = 0
                self.warnings = []
                self.dy_samples = []
                self.f_source = "pose"
                self.cfg = cfg
                captured["min_angle_deg"] = getattr(cfg, "min_angle_deg", None)
                captured["max_reproj_px"] = getattr(cfg, "max_reproj_px", None)
                captured["max_epipolar_px"] = getattr(cfg, "max_epipolar_px", None)

            @property
            def match_rate(self):
                return 7.0 / 16.0

            def update(self, t, det_l, dets):
                return None

            def as_array(self):
                return np.zeros((0, 4), np.float64)

            def low_match_diagnostic(self):
                return []

        class FakeWriter:
            def __init__(self, path, fourcc, fps, size):
                self._p = Path(path)

            def isOpened(self):
                return True

            def write(self, f):
                pass

            def release(self):
                self._p.write_bytes(b"fake-mp4")

        argv = ["run_stereo_video.py", str(session_dir), "--out", str(out_dir),
                "--stereo-mode", "angled", "--pairing", "index", "--no-plot",
                "--no-predict", "--no-world", "--no-radius-auto",
                "--method", "color"]
        if min_angle is not None:
            argv += ["--min-angle-deg", str(min_angle)]
        if max_reproj is not None:
            argv += ["--max-reproj-px", str(max_reproj)]
        patches = [
            _mock.patch.object(rsv, "load_session", return_value=(session_dir, sess)),
            _mock.patch.object(rsv, "validate_session", return_value=([], [])),
            _mock.patch.object(rsv, "session_calibration_file", return_value=calib_path),
            _mock.patch.object(rsv, "resolve_auto", return_value="angled"),
            _mock.patch.object(rsv, "_mode_guard", return_value=("angled", False)),
            _mock.patch.object(rsv, "optical_axes_angle_deg", return_value=60.0),
            _mock.patch.object(rsv, "_pick_detector", return_value=("color", "")),
            _mock.patch.object(rsv, "PairSource", FakePairSource),
            _mock.patch.object(rsv, "_make_pipeline",
                               side_effect=lambda cfg, r, side: FakePipeline()),
            _mock.patch.object(rsv, "_draw", side_effect=lambda *a, **k: frame),
            _mock.patch.object(rsv, "export_session_visualization",
                               return_value={"trajectory": fake_traj,
                                             "visualization": fake_vis}),
            _mock.patch.object(rsv.cv2, "VideoWriter", FakeWriter),
            _mock.patch.object(angled_mod, "AngledReconstructor", FakeRec),
        ]
        rc, stdout, exc_text = None, "", ""
        buf = _io.StringIO()
        try:
            for p in patches:
                p.start()
            old_argv = _sys.argv
            _sys.argv = argv
            try:
                with _contextlib.redirect_stdout(buf):
                    rc = rsv.main()
            finally:
                _sys.argv = old_argv
            stdout = buf.getvalue()
        except Exception as exc:                            # noqa: BLE001
            import traceback as _tb
            exc_text = _tb.format_exc()
            stdout = buf.getvalue()
        finally:
            for p in reversed(patches):
                p.stop()
        payload = {}
        traj_path = out_dir / "traj3d.json"
        if traj_path.exists():
            try:
                payload = _json.loads(traj_path.read_text(encoding="utf-8"))
            except Exception:                                # noqa: BLE001
                pass
        result = {
            "rc": rc, "stdout": stdout, "exc_text": exc_text,
            "quality": payload.get("quality") or {}, "payload": payload,
            "captured": captured,
            "artifacts": all((out_dir / n).exists()
                             for n in ("traj3d.csv", "traj3d.json", "annotated.mp4")),
        }
    finally:
        _shutil.rmtree(tmp, ignore_errors=True)
    return result


def _probe_A1() -> bool:
    """子进程里复用的 A1 判据（反向自证用）。"""
    r = _offline_probe()
    q = r["quality"]
    want = {
        "n_no_left": _PROBE_COUNTERS["n_no_left"],
        "n_no_right": _PROBE_COUNTERS["n_no_right"],
        "n_rejected": _PROBE_COUNTERS["n_rejected"],
        "f_source": "pose", "max_epipolar_px": 8.0,
        "min_angle_deg": 5.0, "max_reproj_px": 3.0,
    }
    return (r["rc"] == 0 and not r["exc_text"]
            and all(q.get(k) == v for k, v in want.items())
            and "round_trip" in r["payload"])


def _probe_A2() -> bool:
    """子进程里复用的 A2 判据（反向自证用）。"""
    r = _offline_probe(forced=True)
    return (r["quality"].get("calibration_forced_unusable") is True
            and "仅链路测试" in r["stdout"]
            and r["rc"] == 0 and r["artifacts"])


def _probe_A5() -> bool:
    """子进程里复用的 A5 判据（反向自证用）。"""
    r = _offline_probe(min_angle=30.0, max_reproj=5.0)
    d = _offline_probe()
    c = r["captured"]
    return (c.get("min_angle_deg") == 30.0
            and c.get("max_reproj_px") == 5.0
            and c.get("max_epipolar_px") == 8.0
            and "极线 8px" in r["stdout"]
            and "交会角 30°" in r["stdout"]
            and "重投影 5px" in r["stdout"]
            and d["captured"].get("min_angle_deg") == 5.0
            and d["captured"].get("max_reproj_px") == 3.0)


def test_offline_attribution() -> bool:
    """判据 1（A1）：quality 分项 == 重建器真值；round_trip 不被删。"""
    print("----- A1：离线产物可归因 -----")
    r = _offline_probe()
    q = r["quality"]
    checks: list[tuple[str, bool]] = []
    if r["exc_text"]:
        print(r["exc_text"][-800:])
    for key, want in (
        ("n_no_left", _PROBE_COUNTERS["n_no_left"]),
        ("n_no_right", _PROBE_COUNTERS["n_no_right"]),
        ("n_rejected", _PROBE_COUNTERS["n_rejected"]),
        ("f_source", "pose"),
        ("max_epipolar_px", 8.0),
        ("min_angle_deg", 5.0),
        ("max_reproj_px", 3.0),
    ):
        got = q.get(key)
        checks.append((f"A1 quality.{key}={got!r}（真值 {want!r}）", got == want))
    checks.append(("A1 round_trip 键没被删", "round_trip" in r["payload"]))
    checks.append(("A1 quality.verdict 没被删", "verdict" in q))
    checks.append(("A1 stdout 有 [归因] 分项行",
                   "[归因]" in r["stdout"]
                   and f"n_no_left={_PROBE_COUNTERS['n_no_left']}" in r["stdout"]))
    checks.append(("A1 main() 返回 0", r["rc"] == 0 and not r["exc_text"]))
    ok = True
    for label, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {label}")
        ok = ok and passed
    print(f"  [实测] quality={q}")
    return ok


def test_offline_forced_calibration() -> bool:
    """判据 2（A2）：坏标定只标记不拦，rc=0 且三产物齐。"""
    print("----- A2：离线侧标记强制标定（不拦） -----")
    r = _offline_probe(forced=True)
    q = r["quality"]
    if r["exc_text"]:
        print(r["exc_text"][-800:])
    checks: list[tuple[str, bool]] = [
        ("A2 quality.calibration_forced_unusable=True（实际 "
         f"{q.get('calibration_forced_unusable')!r}）",
         q.get("calibration_forced_unusable") is True),
        ("A2 stdout 含『仅链路测试』", "仅链路测试" in r["stdout"]),
        ("A2 退出码仍是 0（实际 " + str(r["rc"]) + "）", r["rc"] == 0),
        ("A2 三产物（traj3d.csv/json + annotated.mp4）都在", r["artifacts"]),
    ]
    ok = True
    for label, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {label}")
        ok = ok and passed
    print(f"  [实测] rc={r['rc']}, artifacts={r['artifacts']}, "
          f"calibration_forced_unusable={q.get('calibration_forced_unusable')}")
    return ok


def test_cli_thresholds() -> bool:
    """判据 3（A5）：新 CLI 透传到 AngledMatchConfig；不传仍是 5.0/3.0。"""
    print("----- A5：阈值暴露到 CLI -----")
    r = _offline_probe(min_angle=30.0, max_reproj=5.0)
    d = _offline_probe()
    checks: list[tuple[str, bool]] = []
    if r["exc_text"] or d["exc_text"]:
        print((r["exc_text"] + d["exc_text"])[-800:])
    for key, want in (("min_angle_deg", 30.0), ("max_reproj_px", 5.0),
                      ("max_epipolar_px", 8.0)):
        got = r["captured"].get(key)
        checks.append((f"A5 显式传参 -> 重建器 cfg.{key}={got!r}（真值 {want!r}）",
                       got == want))
    for token in ("极线 8px", "交会角 30°", "重投影 5px"):
        checks.append((f"A5 日志含『{token}』", token in r["stdout"]))
    checks.append(("A5 不传时 min_angle_deg 仍是 5.0（实际 "
                   f"{d['captured'].get('min_angle_deg')!r}）",
                   d["captured"].get("min_angle_deg") == 5.0))
    checks.append(("A5 不传时 max_reproj_px 仍是 3.0（实际 "
                   f"{d['captured'].get('max_reproj_px')!r}）",
                   d["captured"].get("max_reproj_px") == 3.0))
    ok = True
    for label, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {label}")
        ok = ok and passed
    print(f"  [实测] 显式 cfg={r['captured']}；默认 cfg={d['captured']}")
    return ok


def _synthetic_correspondences(angle_deg: float = 60.0, noise_px: float = 0.3,
                               n: int = 60, n_outliers: int = 0,
                               seed: int = 7):
    """按给定夹角/噪声造左右对应点，返回 (left, right, pose_F, K, R, T)。"""
    K = np.array([[800.0, 0.0, 480.0],
                  [0.0, 800.0, 270.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    D = np.zeros(5, dtype=np.float64)
    R = cv2.Rodrigues(np.array([0.0, np.radians(angle_deg), 0.0],
                               dtype=np.float64))[0]
    T = np.array([0.3, 0.0, 0.0], dtype=np.float64)
    F = fundamental_from_pose(K, D, K, D, R, T)
    rng = np.random.default_rng(seed)
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    while len(left) < n:
        X = np.array([rng.uniform(-0.6, 0.6), rng.uniform(-0.3, 0.3),
                      rng.uniform(1.5, 4.0)], dtype=np.float64)
        X2 = R @ X + T
        if X[2] <= 0 or X2[2] <= 0:
            continue
        # 不限像素落在 0..W/H：60° 夹角下两机视野重叠很窄，强加画面边界会让
        # 采样几乎不可能收敛。F 与极线距离对任意有限像素都有定义，这里只要求
        # 两点都在各自相机前方（z>0），这已足够检验 Hartley+RANSAC 的几何。
        pl = (K @ X / X[2])[:2]
        pr = (K @ X2 / X2[2])[:2]
        left.append(pl + rng.normal(0.0, noise_px, 2))
        right.append(pr + rng.normal(0.0, noise_px, 2))
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    if n_outliers:
        idx = rng.choice(n, size=n_outliers, replace=False)
        right_arr[idx] = rng.uniform(0, 960, size=(n_outliers, 2))
    return left_arr, right_arr, F, K, R, T


def _median_epipolar(F: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    ds = []
    for pl, pr in zip(left, right):
        line = epipolar_line(F, (float(pl[0]), float(pl[1])))
        d = point_line_distance(line, (float(pr[0]), float(pr[1])))
        if math.isfinite(d):
            ds.append(float(d))
    return float(np.median(ds)) if ds else float("inf")


def test_refined_f() -> bool:
    """判据 4（B4）：Hartley+RANSAC 重估 F 不劣于 pose-F；不足点静默回退。"""
    print("----- B4：refined F（默认关闭，显式启用才试） -----")
    from stereo.angled import AngledReconstructor, refine_fundamental

    checks: list[tuple[str, bool]] = []
    left, right, F_true, K, R, T = _synthetic_correspondences()
    # 模拟 2026-11-11 的事故：对应点来自真实几何，但标定（pose）漂了，
    # 于是 pose-F 是错的。refined-F 的价值就在这里 —— 若拿"完美 pose-F"当基线，
    # 8 点拟合的几何残差本来就不可能低于真值，那条比较没有意义。
    D = np.zeros(5)
    R_bad = R @ cv2.Rodrigues(
        np.array([0.0, np.radians(2.0), 0.0], dtype=np.float64))[0]
    T_bad = T * 1.15
    F_pose = fundamental_from_pose(K, D, K, D, R_bad, T_bad)
    med_pose = _median_epipolar(F_pose, left, right)
    result = refine_fundamental(F_pose, left, right)
    checks.append(("B4 60° + 0.3px 噪声：refine 返回 F（点数 "
                   f"{len(left)}）", result is not None))
    med_ref = float("inf")
    if result is not None:
        F_ref, info = result
        med_ref = _median_epipolar(F_ref, left, right)
        checks.append((f"B4 极线中位 refined {med_ref:.4f}px ≤ pose {med_pose:.4f}px",
                       med_ref <= med_pose + 1e-6))
        checks.append((f"B4 内点占比 refined {info['inlier_ratio'] * 100:.1f}% "
                       f"≥ pose {info['base_inlier_ratio'] * 100:.1f}%",
                       info["inlier_ratio"] >= info["base_inlier_ratio"] - 1e-9))

    left2, right2, F_true2, K2, R2, T2 = _synthetic_correspondences(
        n_outliers=6, seed=11)
    R_bad2 = R2 @ cv2.Rodrigues(
        np.array([0.0, np.radians(2.0), 0.0], dtype=np.float64))[0]
    T_bad2 = T2 * 1.15
    F_pose2 = fundamental_from_pose(K2, D, K2, D, R_bad2, T_bad2)
    med_pose2 = _median_epipolar(F_pose2, left2, right2)
    res2 = refine_fundamental(F_pose2, left2, right2)
    checks.append(("B4 10% 外点：refine 仍返回 F", res2 is not None))
    if res2 is not None:
        F_ref2, info2 = res2
        med_ref2 = _median_epipolar(F_ref2, left2, right2)
        checks.append((f"B4 10% 外点：极线中位 refined {med_ref2:.4f}px "
                       f"≤ pose {med_pose2:.4f}px（不外劣）",
                       med_ref2 <= med_pose2 + 1e-6))
        checks.append((f"B4 10% 外点：内点数足够（{info2['n_points']} 点，"
                       f"内点占比 {info2['inlier_ratio'] * 100:.0f}%）",
                       info2["n_points"] >= 8 and info2["inlier_ratio"] >= 0.8))

    # 点数 <8：静默回退，不抛异常，f_source 保持 pose。
    short_l, short_r, F_true3, K3, R3, T3 = _synthetic_correspondences(n=6, seed=13)
    F_pose3 = fundamental_from_pose(
        K3, D, K3, D,
        R3 @ cv2.Rodrigues(np.array([0.0, np.radians(2.0), 0.0]))[0], T3 * 1.15)
    checks.append(("B4 点数<8：refine 返回 None（不抛）",
                   refine_fundamental(F_pose3, short_l, short_r) is None))

    class FakeRig:
        def __init__(self):
            self.K1, self.K2 = K, K
            self.D1, self.D2 = np.zeros(5), np.zeros(5)
            self.R, self.T, self.F = R_bad, T_bad, None

    try:
        rec_short = AngledReconstructor(
            FakeRig(), f_source="refined",
            correspondences=(short_l, short_r))
        checks.append(("B4 点数<8：构造不抛异常，f_source 回退 pose（实际 "
                       f"{rec_short.f_source!r}）", rec_short.f_source == "pose"))
        rec_none = AngledReconstructor(FakeRig(), f_source="refined",
                                       correspondences=None)
        checks.append(("B4 无对应点：f_source 回退 pose（实际 "
                       f"{rec_none.f_source!r}）", rec_none.f_source == "pose"))
        rec_ok = AngledReconstructor(FakeRig(), f_source="refined",
                                     correspondences=(left, right))
        checks.append(("B4 好数据：f_source 采纳 refined（实际 "
                       f"{rec_ok.f_source!r}）", rec_ok.f_source == "refined"))
    except Exception as exc:                                # noqa: BLE001
        checks.append((f"B4 构造异常 {type(exc).__name__}: {exc}", False))

    # 默认不启用：不传 f_source 时仍是 pose（即使给了好对应点也不碰）。
    rec_default = AngledReconstructor(FakeRig(), correspondences=(left, right))
    checks.append(("B4 默认关闭：不传 f_source 时仍为 pose（实际 "
                   f"{rec_default.f_source!r}）", rec_default.f_source == "pose"))

    ok = True
    for label, passed in checks:
        print(f"  [{'通过' if passed else '不达标'}] {label}")
        ok = ok and passed
    print(f"  [实测] 极线中位：pose {med_pose:.4f}px -> refined {med_ref:.4f}px；"
          f"10%外点 pose {med_pose2:.4f}px -> refined "
          f"{_median_epipolar(res2[0], left2, right2) if res2 else float('inf'):.4f}px")
    return ok


def test_reverse_selfcert() -> bool:
    """判据 5（反向自证）：A1/A2/A5 写入改空操作 -> 对应判据红；还原 sha256 一致。"""
    print("----- 反向自证：写入空操作 -> 变红 + 字节级还原 -----")
    import hashlib as _hashlib
    import subprocess as _subprocess

    root = Path(__file__).resolve().parent
    while not (root / "sphere_tracker").is_dir() and root.parent != root:
        root = root.parent
    target = root / "apps" / "run_stereo_video.py"
    original = target.read_bytes()
    orig_sha = _hashlib.sha256(original).hexdigest()
    mutations = [
        ("A1", b'            _q["max_epipolar_px"] = float(epi)\r\n',
         b'            pass  # reverse-A1-noop\r\n', "_probe_A1"),
        ("A2",
         b'            _q["calibration_forced_unusable"] = forced_unusable\r\n',
         b'            pass  # reverse-A2-noop\r\n', "_probe_A2"),
        ("A5",
         b'        angled_cfg = AngledMatchConfig(max_epipolar_px=epi,\r\n'
         b'                                       min_angle_deg=_min_angle,\r\n'
         b'                                       max_reproj_px=_max_reproj)\r\n',
         b'        angled_cfg = AngledMatchConfig(max_epipolar_px=epi)\r\n',
         "_probe_A5"),
    ]
    checks: list[tuple[str, bool]] = []
    code = (
        "import sys; sys.path[:0] = [{root!r}, {tests!r}]; "
        "import selftest_stereo_branch as m; "
        "print('RED' if not m.{probe}() else 'GREEN')"
    )
    for name, old, new, probe in mutations:
        found = original.count(old)
        checks.append((f"反向自证 {name}：源码有唯一可改点（找到 {found} 处）",
                       found == 1))
        if found != 1:
            continue
        try:
            target.write_bytes(original.replace(old, new, 1))
            proc = _subprocess.run(
                [sys.executable, "-c",
                 code.format(root=str(root), tests=str(root / "tests"),
                             probe=probe)],
                capture_output=True, text=True, timeout=300,
                cwd=str(root),
            )
            out = (proc.stdout or "").strip()
            checks.append((f"反向自证 {name}：写入改空操作 -> 判据变红"
                           f"（子进程 {out or proc.returncode}）",
                           "RED" in out))
            if proc.returncode != 0 and "RED" not in out:
                print((proc.stderr or "")[-600:])
        except Exception as exc:                            # noqa: BLE001
            checks.append((f"反向自证 {name}：子进程异常 {exc}", False))
        finally:
            target.write_bytes(original)
        restored_sha = _hashlib.sha256(target.read_bytes()).hexdigest()
        checks.append((f"反向自证 {name}：字节级还原后 sha256 一致",
                       restored_sha == orig_sha))
    print(f"  [实测] sha256(original)={orig_sha}")
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

    classify_ok = test_guard_classification()
    print()

    no_early_stop_ok = test_guard_no_early_stop()
    print()

    carry_ok = test_export_carryover()
    print()

    world_traj_ok = test_world_frame_in_traj_payload()
    print()

    round_trip_ok = test_round_trip_metric()
    print()

    fov_apply_ok = test_fov_apply()
    print()

    realtime_ok = test_realtime_stereo()
    print()

    attribution_ok = test_offline_attribution()
    print()

    forced_ok = test_offline_forced_calibration()
    print()

    cli_ok = test_cli_thresholds()
    print()

    refined_ok = test_refined_f()
    print()

    reverse_ok = test_reverse_selfcert()
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
              and geom_ok and advice_ok and center_ok and radius_ok and guard_ok
              and classify_ok and no_early_stop_ok and carry_ok and world_traj_ok
              and round_trip_ok and fov_apply_ok and realtime_ok
              and attribution_ok and forced_ok and cli_ok and refined_ok
              and reverse_ok)
    print("判定结果：" + ("全部通过" if all_ok else "有不达标项"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
