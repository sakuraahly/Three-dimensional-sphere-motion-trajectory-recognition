"""单帧总流程：检测 -> 跟踪 -> 三维 -> 预测 -> 画图。

一站式用法::

    from sphere_tracker import AppConfig, SpherePipeline, open_source  # open_source 在 sources 里
    pipe = SpherePipeline(AppConfig())
    src = open_source(camera=0)
    while True:
        ok, frame, t = src.read()
        if not ok:
            break
        result = pipe.process(frame, t)
        vis = pipe.render(frame, result)
        cv2.imshow("ball", vis)
"""

from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np

from .camera_motion import GlobalMotionEstimator
from .config import AppConfig
from .detect import build_detector
from .geometry import CameraModel
from .predict import TrajectoryPredictor
from .track import BallTracker
from .types import FrameContext, FrameResult, TrailPoint
from .viz import (
    draw_3d_panel,
    draw_detection,
    draw_hud,
    draw_prediction,
    draw_trail,
    hud_lines,
    paste_mask_preview,
)


class SpherePipeline:
    """把各个模块串起来的编排器。"""

    def __init__(self, cfg: AppConfig | None = None) -> None:
        self.cfg = cfg or AppConfig()
        self.detector = build_detector(self.cfg.detector)
        self.tracker = BallTracker(self.cfg.tracker)
        self.predictor = TrajectoryPredictor(self.cfg.predict)
        self.camera: CameraModel | None = None
        self.history3d: deque[TrailPoint] = deque(maxlen=self.cfg.tracker.trail_length)
        self._frame_index = 0
        self._prev_t: float | None = None
        self._fps_ema = 0.0
        self._t0: float | None = None
        self._detector_error: str | None = None
        # 相机自运动估计（手持/云台/平移拍摄时用它区分"画面在动"与"球在动"）
        self.motion_estimator = GlobalMotionEstimator(
            work_width=self.cfg.detector.motion_work_width,
            threshold_px=self.cfg.detector.camera_motion_threshold_px,
            feature_fallback=self.cfg.detector.motion_feature_fallback,
        )
        self._prev_gray: np.ndarray | None = None
        self._motion_recent: deque[float] = deque(maxlen=6)   # 最近几帧的全局位移幅值

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """清空轨迹（换目标/球飞出画面后手动调用）。"""
        self.tracker.reset()
        self.history3d.clear()
        self.detector.reset()
        self.motion_estimator.reset()
        self._prev_gray = None
        self._frame_index = 0
        self._prev_t = None

    def _ensure_camera(self, frame: np.ndarray) -> None:
        if self.camera is not None:
            return
        h, w = frame.shape[:2]
        c = self.cfg.camera
        self.camera = CameraModel(
            width=w, height=h, fx=c.fx, fy=c.fy, cx=c.cx, cy=c.cy,
            hfov_deg=c.hfov_deg, ball_diameter_m=c.ball_diameter_m,
        )

    # ------------------------------------------------------------------
    def process(self, frame: np.ndarray, t: float | None = None) -> FrameResult:
        """处理一帧。t 是秒（视频文件用帧号/fps，摄像头用 perf_counter）。"""
        now = time.perf_counter()
        if t is None:
            if self._t0 is None:
                self._t0 = now
            t = now - self._t0

        if self._prev_t is not None and t > self._prev_t:
            inst = 1.0 / (t - self._prev_t)
            self._fps_ema = inst if self._fps_ema == 0.0 else 0.9 * self._fps_ema + 0.1 * inst
        self._prev_t = t

        self._ensure_camera(frame)

        # ---- 每帧只算一次：灰度图 + 相机自运动 ----
        # 灰度图给 color/hough/亚像素细化共用；相机自运动用来区分
        # "画面在动"到底是因为相机在动，还是因为球在动。
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion = self.motion_estimator.estimate(self._prev_gray, gray)
        camera_moving = self.motion_estimator.is_moving(motion)
        self._prev_gray = gray
        self._motion_recent.append(float(motion.magnitude))
        # 区分"匀速移动"和"抖动/加速"：
        #   匀速平移只是给图像叠加一个常数速度，不引入额外的表观加速度，
        #   所以"已知像素重力"这一招依然成立；
        #   而抖动/加减速会污染加速度项，必须退回自由二次拟合。
        steady = False
        if camera_moving and len(self._motion_recent) >= 3:
            recent = list(self._motion_recent)[-5:]
            lo, hi = min(recent), max(recent)
            # 容差要能容忍"亚像素位移被取整"造成的 1 px 跳动，
            # 又要能把真正的抖动/加减速区分出来
            steady = (hi - lo) <= max(1.0, 0.5 * hi)
        ctx = FrameContext(t=t, gray=gray, motion=motion, camera_moving=camera_moving)

        t_det = time.perf_counter()
        try:
            detections = self.detector.detect(frame, ctx)
        except RuntimeError as exc:  # AI 后端未安装时给出清晰提示
            detections = []
            if self._detector_error is None:
                self._detector_error = str(exc)
                print(f"[检测器不可用] {exc}")
        detect_ms = (time.perf_counter() - t_det) * 1000.0

        best = detections[0] if detections else None
        track = self.tracker.update(detections, t)
        state = track.state() if track is not None else None

        result = FrameResult(
            t=t, frame_index=self._frame_index, detection=best, track=state,
            fps=self._fps_ema, detect_ms=detect_ms,
            camera_motion=motion, camera_moving=camera_moving,
            camera_steady=steady,
        )
        self._frame_index += 1

        # ---- 三维：像素 + 由球半径反推的深度 -------------------------
        if (
            self.cfg.camera.enable_3d
            and self.camera is not None
            and state is not None
            and best is not None
            and best.radius > 1e-3
        ):
            # 用"原始观测"的圆心与半径做三维重建：半径直接决定深度，
            # 用滤波后的半径会引入滞后，深度就会跟着飘。
            z = self.camera.depth_from_radius(best.radius)
            if np.isfinite(z) and 0.01 < z < 100.0:
                pos = self.camera.backproject(best.x, best.y, z)
                state.z = float(z)
                if self.history3d:
                    prev = self.history3d[-1]
                    dt = state.t - prev.t
                    if dt > 1e-6 and prev.z is not None:
                        state.vz = float((z - prev.z) / dt)
                result.position_3d = pos
                self.history3d.append(
                    TrailPoint(t=state.t, x=float(pos[0]), y=float(pos[1]), z=float(pos[2]),
                               radius=float(best.radius), score=1.0)
                )

        # ---- 轨迹拖尾（画图用） -------------------------------------
        result.trail = list(track.trail) if track is not None else []

        # ---- 预测 ----------------------------------------------------
        if self.cfg.predict.enable and track is not None and not track.coasting:
            dims = 2
            if self.cfg.predict.use_gravity and len(self.history3d) >= self.cfg.predict.min_points:
                dims = 3
            if dims == 3:
                # 三维拟合的残差容差要换算成米：tol_m = tol_px * Z / f
                z_now = state.z if state.z else 1.0
                f_now = 0.5 * (self.camera.fx + self.camera.fy) if self.camera else 1.0
                tol_m = self.cfg.predict.residual_tol * z_now / max(f_now, 1e-6)
                pred = self.predictor.predict(
                    list(self.history3d), state, dims=3, residual_tol=tol_m, t_now=state.t,
                    t_break=state.last_maneuver_t,
                )
                if pred is not None:
                    result.prediction_3d = pred
                    projected = np.array([self.camera.project(p) for p in pred], dtype=np.float64)
                    result.prediction = projected
            if result.prediction is None:
                # 二维像素重力：g_px = fy * g / Z（Z 由球的像素半径反推）
                g_px = None
                if (
                    self.cfg.predict.use_pixel_gravity
                    and self.camera is not None
                    and state.z is not None
                    # 相机在动时不能用"像素重力"：球在图像里的表观加速度 =
                    # 球的重力 + 相机自身的运动加速度，两者混在一起没法分离，
                    # 硬套 g_px 反而会给出错误的抛物线。此时退回自由二次拟合。
                    and (not camera_moving or steady)
                ):
                    g_px = self.camera.fy * self.cfg.predict.gravity / max(state.z, 1e-6)
                result.prediction = self.predictor.predict(
                    result.trail, state, dims=2, t_now=state.t, gravity_value=g_px,
                    t_break=state.last_maneuver_t,
                )

        return result

    # ------------------------------------------------------------------
    def render(self, frame: np.ndarray, result: FrameResult,
               health=None, display_width: int | None = None) -> np.ndarray:
        """把结果画到帧上，返回新图（不修改输入）。

        `display_width`：**只影响显示用的输出图宽度**（像素）。给 960 就返回
        宽 960 的图，叠加层位置与线宽按同一倍率缩好 —— 也就是"看起来一样，
        只是一张更小的图"。

        ## 为什么要有这个参数（2026-10-14 实测）

        `cv2.imshow` 把一个 1920x1080 的图**缩到窗口里**显示，代价高得离谱：

            imshow 1920x1080 窗口 1080p 图    17.74 ms
            imshow  960x540 窗口 1080p 图      6.86 ms     （让 imshow 自己缩）
            先把图 resize 到 960x540，再 imshow   1.84 ms   ← 便宜 9.6 倍

        也就是说每秒最多 56 帧是**花在缩放上**的，而这件事我们自己做只要
        0.55 ms。显示从 17.7 ms 降到 1.84 ms，直接決定"顺不顺"。

        `health`：可选的 `stats.Health`。**为什么由调用方传、而不是 pipeline 自己持有** ——
        pipeline 是编排层，不该持有一个"随运行不断变化的累加器"：

        - 采集侧的数据**只有 app 拿得到**（它持有 `LatestFrameCamera`）；
        - 同一份 `Health` 还要被序列化落盘（P1-4 的会话记录），那是 app 的事；
        - pipeline 若自己累加，就会出现"pipeline 的和 app 的各算一遍"，
          正是 `stats.py` 刻意消除的那种"同一个量两处各算"。

        不传时 HUD 退化成一行明确的 `loop ...`（不再有裸 `FPS`）。
        """
        viz = self.cfg.viz
        scale = viz.scale if viz.scale and viz.scale > 0 else 1.0
        # 显示倍率：`display_width` 与 `viz.scale` 是**两件事**，
        # 前者是"输出图多宽"（通常 <1，为了显示便宜），
        # 后者是"把整张图放大多少"（通常 =1，为了在小窗口里看清叠加层）。
        # 两者相乘得到最终倍率，于是既能缩小显示、又能保持放大语义。
        if display_width and frame.shape[1] > 0:
            scale *= float(display_width) / float(frame.shape[1])
        if abs(scale - 1.0) > 1e-6:
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            img = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=interp)
        else:
            img = frame.copy()

        if viz.trail_enabled and result.trail:
            draw_trail(img, result.trail, fade=viz.trail_fade,
                       thickness=max(1, int(2 * scale)), scale=scale)

        if viz.show_circle and result.detection is not None:
            label = f"r={result.detection.radius:.1f}"
            draw_detection(img, result.detection, color=(0, 255, 0), label=label,
                           contour=True, scale=scale)

        if viz.show_prediction and result.prediction is not None and len(result.prediction) > 1:
            tag = "pred 3D+g" if result.prediction_3d is not None else "pred 2D"
            # 不在这里预乘 scale —— 交给 draw_prediction，让"画布缩放"只有一个负责处
            draw_prediction(img, result.prediction, color=(0, 140, 255), label=tag,
                            thickness=max(1, int(2 * scale)), scale=scale)

        if viz.show_mask and result.detection is not None and result.detection.mask is not None:
            paste_mask_preview(img, result.detection.mask, width=int(200 * scale))

        if viz.show_hud:
            draw_hud(img, hud_lines(result, self.cfg, health=health), scale=scale)

        if viz.show_axes_3d:
            hist = np.array([[p.x, p.y, p.z] for p in self.history3d if p.z is not None], dtype=np.float64)
            pred3d = result.prediction_3d
            panel = draw_3d_panel(
                size=(int(320 * scale), int(240 * scale)),
                history=hist if len(hist) else None,
                prediction=pred3d,
                # 面板内部的字号/线宽也要跟 scale，否则放大后 3D 内容相对变小
                scale=scale,
            )
            ph, pw = panel.shape[:2]
            h, w = img.shape[:2]
            if ph < h and pw < w:
                img[10 : 10 + ph, w - pw - 10 : w - 10] = panel
        return img

    # ------------------------------------------------------------------
    def process_and_render(self, frame: np.ndarray, t: float | None = None,
                           health=None,
                           display_width: int | None = None) -> tuple[np.ndarray, FrameResult]:
        result = self.process(frame, t)
        return self.render(frame, result, health=health,
                           display_width=display_width), result
