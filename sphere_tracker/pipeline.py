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
        # 检测输入宽度（None = 用原始分辨率）。见 set_detect_width。
        self._detect_width: int | None = None
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

    def set_detect_width(self, width: int | None) -> None:
        """让检测跑在**缩小的图**上（坐标会自动换算回原图）。

        None / 0 = 不缩（默认，保持既有精度）。
        见 process() 里那段长注释：真实纹理下 hough 在 1080p 上要 4.4 秒一帧，
        降到 960 宽是唯一有效的杠杆。
        """
        self._detect_width = int(width) if width else None

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
        # ---- 检测可以在**更小的图**上跑（2026-10-14 加）----
        #
        # ## 为什么这是必需的，不是优化
        #
        # 实测（1920x1080，**有真实纹理**的画面 —— 书架/床/杂物那种）：
        #
        #     color    17.8 ms  ->  56 fps
        #     motion   15.5 ms  ->  64 fps
        #     hough  4384.9 ms  -> 0.2 fps   ← **4.4 秒一帧**
        #     fusion 3740.5 ms  -> 0.3 fps   ← **3.7 秒一帧**
        #
        # 而同样的画面如果是"纯色底 + 一个圆"，`hough` 只要 132 ms。
        # 差别在于 `HoughCircles` 要先做 Canny，**纹理一多边缘就爆炸**，
        # 累加器搜索量随边缘数暴涨。真实房间**永远**是纹理丰富的，
        # 所以"合成图上测得挺快"完全没有参考价值 —— 这是本项目的老教训。
        #
        # 降分辨率是这里**唯一有效**的杠杆：像素少 4 倍，边缘与累加器都少一个量级。
        # 代价是小目标的定位精度下降，所以：
        #   · 默认**不降**（保持既有行为，避免悄悄改变精度）；
        #   · 由 app 通过 `--detect-width` 显式打开，并在启用时打印代价。
        detect_frame = frame
        detect_scale = 1.0
        if self._detect_width and 0 < self._detect_width < frame.shape[1]:
            detect_scale = float(self._detect_width) / float(frame.shape[1])
            detect_frame = cv2.resize(
                frame, None, fx=detect_scale, fy=detect_scale,
                interpolation=cv2.INTER_AREA)
        try:
            detections = self.detector.detect(detect_frame, ctx)
        except RuntimeError as exc:  # AI 后端未安装时给出清晰提示
            detections = []
            if self._detector_error is None:
                self._detector_error = str(exc)
                print(f"[检测器不可用] {exc}")
        detect_ms = (time.perf_counter() - t_det) * 1000.0

        # 检测是在缩小的图上做的 -> **坐标必须换算回原图**。
        # 不换算的话，圈会画在错位置、三维反投影也会全错
        # （半径直接决定深度，半径错一倍深度就错一倍）。
        if detect_scale < 1.0 and detections:
            detections = [d.scaled(1.0 / detect_scale) for d in detections]

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
               health=None, display_width: int | None = None,
               render_scale: float = 1.0) -> np.ndarray:
        """把结果画到帧上，返回新图（不修改输入）。

        ## 两个缩放参数**不是一回事**，别混

        | 参数 | 管什么 | 什么时候用 |
        | ---- | ------ | ---------- |
        | `display_width` | **先缩图再画叠加**：把输出图缩到指定宽度 | 实时显示（省 imshow 的缩放开销） |
        | `render_scale` | **先放大再画叠加**：整张画布（含叠加）等比放大 | 想把叠加层画大些看细节 |

        ## ⚠️ 顺序：**先缩图/放大，再画叠加**（2026-10-14 修的真回归）

        我第一版把 `display_width` 折进同一个 `scale`，于是变成
        **先 resize、再用原尺寸字号画 HUD** —— `cv2.putText` 在 1920 的画布上
        写完字，整张图（连字一起）被压到 960，**字就糊了**。
        用户的原话是"连参数文字也变得模糊了"，这是对的。

        正确做法是**先定画布尺寸、再在上面画**：字在最终分辨率上渲染，
        永远清晰。所以 `display_width` 走"先 resize 再作画"这条路，
        而 `render_scale` 走"先放大再作画"。

        `health`：可选的 `stats.Health`。**为什么由调用方传、而不是 pipeline 自己持有** ——
        pipeline 是编排层，不该持有一个"随运行不断变化的累加器"：

        - 采集侧的数据**只有 app 拿得到**（它持有 `LatestFrameCamera`）；
        - 同一份 `Health` 还要被序列化落盘（P1-4 的会话记录），那是 app 的事；
        - pipeline 若自己累加，就会出现"pipeline 的和 app 的各算一遍"，
          正是 `stats.py` 刻意消除的那种"同一个量两处各算"。

        不传时 HUD 退化成一行明确的 `loop ...`（不再有裸 `FPS`）。
        """
        viz = self.cfg.viz
        # `render_scale` 是"放大画布"的语义（用户显式要的）；
        # `display_width` 是"把输出图控制到多宽"（显示用，通常缩小）。
        scale = render_scale * (viz.scale if viz.scale and viz.scale > 0 else 1.0)
        if display_width and frame.shape[1] > 0:
            scale *= float(display_width) / float(frame.shape[1])
        if abs(scale - 1.0) > 1e-6:
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            img = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=interp)
        else:
            img = frame.copy()

        # ---- 以下所有叠加都画在 `img`（已是最终尺寸）上 ----
        # 字号/线宽乘的是**最终相对原始帧的倍率**，所以：
        #   · 缩小显示时，字按小画布的正常字号渲染 → **清晰**；
        #   · 放大显示时，字跟着变大 → 与被放大的画面相称。
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
                           health=None, display_width: int | None = None,
                           render_scale: float = 1.0) -> tuple[np.ndarray, FrameResult]:
        result = self.process(frame, t)
        return self.render(frame, result, health=health,
                           display_width=display_width,
                           render_scale=render_scale), result
