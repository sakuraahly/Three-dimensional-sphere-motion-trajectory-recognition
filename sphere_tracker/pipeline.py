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

from .config import AppConfig
from .detect import build_detector
from .geometry import CameraModel
from .predict import TrajectoryPredictor
from .track import BallTracker
from .types import FrameResult, TrailPoint
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

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """清空轨迹（换目标/球飞出画面后手动调用）。"""
        self.tracker.reset()
        self.history3d.clear()
        self.detector.reset()
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

        t_det = time.perf_counter()
        try:
            detections = self.detector.detect(frame)
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
                ):
                    g_px = self.camera.fy * self.cfg.predict.gravity / max(state.z, 1e-6)
                result.prediction = self.predictor.predict(
                    result.trail, state, dims=2, t_now=state.t, gravity_value=g_px,
                    t_break=state.last_maneuver_t,
                )

        return result

    # ------------------------------------------------------------------
    def render(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        """把结果画到帧上，返回新图（不修改输入）。"""
        viz = self.cfg.viz
        scale = viz.scale if viz.scale and viz.scale > 0 else 1.0
        if abs(scale - 1.0) > 1e-6:
            img = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        else:
            img = frame.copy()

        if viz.trail_enabled and result.trail:
            draw_trail(img, result.trail, fade=viz.trail_fade, thickness=max(1, int(2 * scale)))

        if viz.show_circle and result.detection is not None:
            label = f"r={result.detection.radius:.1f}"
            draw_detection(img, result.detection, color=(0, 255, 0), label=label, contour=True)

        if viz.show_prediction and result.prediction is not None and len(result.prediction) > 1:
            pts = result.prediction * scale if abs(scale - 1.0) > 1e-6 else result.prediction
            tag = "pred 3D+g" if result.prediction_3d is not None else "pred 2D"
            draw_prediction(img, pts, color=(0, 140, 255), label=tag, thickness=max(1, int(2 * scale)))

        if viz.show_mask and result.detection is not None and result.detection.mask is not None:
            paste_mask_preview(img, result.detection.mask, width=int(200 * scale))

        if viz.show_hud:
            draw_hud(img, hud_lines(result, self.cfg))

        if viz.show_axes_3d:
            hist = np.array([[p.x, p.y, p.z] for p in self.history3d if p.z is not None], dtype=np.float64)
            pred3d = result.prediction_3d
            panel = draw_3d_panel(
                size=(int(320 * scale), int(240 * scale)),
                history=hist if len(hist) else None,
                prediction=pred3d,
            )
            ph, pw = panel.shape[:2]
            h, w = img.shape[:2]
            if ph < h and pw < w:
                img[10 : 10 + ph, w - pw - 10 : w - 10] = panel
        return img

    # ------------------------------------------------------------------
    def process_and_render(self, frame: np.ndarray, t: float | None = None) -> tuple[np.ndarray, FrameResult]:
        result = self.process(frame, t)
        return self.render(frame, result), result
