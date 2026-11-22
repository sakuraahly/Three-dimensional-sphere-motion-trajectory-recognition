"""合成数据自测：用已知真值的运动球体检验整条流水线。

为什么需要它：真实摄像头前你永远不知道"正确答案"，也就无法判断
自己的检测到底差多少。这里程序生成一段"真值完全已知"的球体运动视频：

    三维空间里按真实重力 g=9.80665 运动 -> 透视投影到像素
    （所以像素轨迹并不是标准抛物线，这正是真实情况）

场景包含：斜抛 + 落地反弹（恢复系数 0.8）+ 前后/左右碰壁 + 深度变化 +
          逐帧噪声 + 按曝光时间模拟的运动模糊 +
          两个"陷阱"干扰物（一个和球同色的白色方块、一个和球同形状的橙色圆）

然后逐帧比较：检出率 / 圆心误差 / 半径误差 / 三维误差 / 预测误差。

用法::

    python tests/selftest_synthetic.py                 # 全部用例
    python tests/selftest_synthetic.py --method color
    python tests/selftest_synthetic.py --predict kalman
    python tests/selftest_synthetic.py --write-video    # 顺便导出 mp4
    python tests/selftest_synthetic.py --check-ai       # 检查 AI 依赖是否就绪
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

# --- 让脚本在任意目录深度都能找到项目根 ---
ROOT = Path(__file__).resolve().parent
while not (ROOT / "sphere_tracker").is_dir() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from sphere_tracker.config import AppConfig, ColorConfig
from sphere_tracker.geometry import CameraModel
from sphere_tracker.pipeline import SpherePipeline
from sphere_tracker.plugins import names as method_names  # 预测器清单的唯一来源（R7）
from stereo.motion import kinematics, smooth_positions, split_segments

GRAVITY = 9.80665


# --------------------------------------------------------------------------
# 场景真值（三维 + 透视投影）
# --------------------------------------------------------------------------
class Scene:
    """一段真值已知的球体运动。

    相机在原点，看向 +Z，Y 轴向下（和图像 v 轴同向）。
    球在三维空间里做斜抛，落地反弹、碰壁反射，半径恒定（真实球体）。
    图像里的半径 r = f*R/Z 会随深度自然变化，深度也就"天然"可知。
    """

    def __init__(
        self,
        width: int = 960,
        height: int = 540,
        fps: float = 60.0,
        duration: float = 3.0,
        ball_diameter_m: float = 0.22,
        hfov_deg: float = 60.0,
        restitution: float = 0.85,
        motion_blur: bool = True,
        noise: bool = True,
        distractors: bool = True,
        pan_speed: float = 0.0,
        z_range: tuple[float, float] = (2.4, 6.4),
        velocity: tuple[float, float, float] | None = None,
    ) -> None:
        self.width, self.height = width, height
        # 相机平移速度（像素/秒，全分辨率）。>0 表示相机在匀速横移，
        # 用来检验"相机不固定"时检测是否还成立。
        self.pan_speed = float(pan_speed)
        self.pan_max = abs(pan_speed) * (duration + 0.5) + 40.0
        self.fps = fps
        self.duration = duration
        self.ball_diameter_m = ball_diameter_m
        self.restitution = restitution
        self.motion_blur = motion_blur
        self.noise = noise
        self.distractors = distractors

        self.f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        self.cx = width / 2.0
        self.cy = height / 2.0
        self.hfov_deg = hfov_deg
        self.n = int(duration * fps)
        self.dt = 1.0 / fps

        # 三维初值（米）：从左上方向右下抛出，同时在远离相机
        pos = np.array([-1.05, -0.12, 3.7], dtype=np.float64)
        vel = np.array([0.95, -2.0, 0.14] if velocity is None else velocity,
                       dtype=np.float64)
        R = ball_diameter_m / 2.0
        floor_y = 0.82            # 地面平面（相机下方 0.82 m）
        wall_x = 1.30             # 左右"墙"，只为把球留在画面里
        z_min, z_max = z_range

        sub = 0.001
        # 多仿真 2 秒"视频之后"的时间：评估 t+horizon 处的预测精度时要用到
        steps = int((duration + 2.0) / sub)
        self.sub = sub
        self.pos = np.empty((steps, 3), dtype=np.float64)
        self.bounces: list[float] = []
        for i in range(steps):
            vel[1] += GRAVITY * sub
            pos = pos + vel * sub
            if pos[1] > floor_y - R:
                pos[1] = floor_y - R
                vel[1] = -abs(vel[1]) * restitution
                self.bounces.append(i * sub)
            if pos[0] < -wall_x + R:
                pos[0] = -wall_x + R
                vel[0] = abs(vel[0])
                self.bounces.append(i * sub)
            if pos[0] > wall_x - R:
                pos[0] = wall_x - R
                vel[0] = -abs(vel[0])
                self.bounces.append(i * sub)
            if pos[2] < z_min:
                pos[2] = z_min
                vel[2] = abs(vel[2])
                self.bounces.append(i * sub)
            if pos[2] > z_max:
                pos[2] = z_max
                vel[2] = -abs(vel[2])
                self.bounces.append(i * sub)
            self.pos[i] = pos

    # -- 查询 ----------------------------------------------------------
    def _idx(self, t: float) -> np.ndarray:
        return np.clip((np.asarray(t) / self.sub).astype(np.int64), 0, self.pos.shape[0] - 1)

    def center3d(self, t: float) -> np.ndarray:
        return self.pos[self._idx(t)].astype(np.float64)

    def pan_offset(self, t: float) -> float:
        """相机横移量（像素）。相机向右移 -> 画面内容向左移。"""
        return self.pan_speed * t

    def project(self, t: float) -> tuple[float, float, float]:
        """三维真值 -> 像素 (u, v, r)。含相机自运动带来的偏移。"""
        p = self.center3d(t)
        z = float(p[2])
        r = self.f * (self.ball_diameter_m / 2.0) / z
        u = self.cx + self.f * float(p[0]) / z - self.pan_offset(t)
        v = self.cy + self.f * float(p[1]) / z
        return u, v, r

    def camera_params(
        self, side: str = "left", K: np.ndarray | None = None, D: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """取某一路的内参/畸变；不给就退化成"本场景的针孔相机、无畸变"。

        K=None 时用 scene.f / cx / cy 拼一个标准针孔内参 —— 这样老的调用方
        （单目自测）行为完全不变，而双目自测可以把自己的真实内参传进来。
        """
        if K is not None:
            return np.asarray(K, np.float64), (None if D is None else np.asarray(D, np.float64))
        K_std = np.array([[self.f, 0.0, self.cx], [0.0, self.f, self.cy], [0.0, 0.0, 1.0]])
        return K_std, (None if D is None else np.asarray(D, np.float64))

    def project(
        self,
        t: float,
        side: str = "left",
        R: np.ndarray | None = None,
        T: np.ndarray | None = None,
        K: np.ndarray | None = None,
        D: np.ndarray | None = None,
    ) -> tuple[float, float, float]:
        """三维真值 -> 某一路的像素 (u, v, r)。

        坐标系约定（与 calib.StereoRig 一致）：
          - 世界系 = 左相机系（原点在左光心，Z 向前，Y 向下，与图像 v 同向）；
          - R, T 描述**右相机相对左相机**的位姿（X_right = R @ X_left + T）。
        所以 side="right" 时先换算到右相机系再投影。

        u 里含相机自运动带来的偏移（-pan_offset），r 是针孔下的 f*R/Z
        （小孔成像下球轮廓的角半径，忽略透视椭圆化 —— 见 _ball_coverage 的说明）。
        """
        p = self.center3d(t)
        if side == "right":
            if R is None or T is None:
                raise ValueError('side="right" 必须给 R 和 T')
            p = np.asarray(R, np.float64) @ p + np.asarray(T, np.float64).ravel()
        K_use, D_use = self.camera_params(side, K, D)
        r = float(K_use[0, 0]) * (self.ball_diameter_m / 2.0) / float(p[2])
        if D_use is not None and np.any(np.abs(D_use) > 1e-12):
            obj = np.asarray(p, np.float64).reshape(1, 1, 3)
            u, v = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), K_use, D_use)[0].reshape(2)
        else:
            u = float(K_use[0, 2]) + float(K_use[0, 0]) * float(p[0]) / float(p[2])
            v = float(K_use[1, 2]) + float(K_use[1, 1]) * float(p[1]) / float(p[2])
        return float(u) - self.pan_offset(t), float(v), r

    def center(self, t: float) -> tuple[float, float]:
        u, v, _ = self.project(t)
        return u, v

    def radius(self, t: float) -> float:
        return self.project(t)[2]

    def bounce_near(self, t0: float, t1: float) -> bool:
        return any(t0 <= b <= t1 for b in self.bounces)

    def fully_visible(self, t: float, margin: float = 1.0, **cam) -> bool:
        """球的整个圆是否都在画面内。

        半个球在画面外时，轮廓被裁掉、圆拟合必然失准，
        这时候检测不到是"正确行为"，不该算进检出率里。

        `**cam` 透传给 project()，双目里可以传真值内参（`K=` / `D=` / `side=`），
        否则"可见性"判定用的内参和渲染用的内参不是同一个，边界帧会算错。
        """
        u, v, r = self.project(t, **cam)
        return (
            u - r > margin and v - r > margin
            and u + r < self.width - margin and v + r < self.height - margin
        )


# --------------------------------------------------------------------------
# 画面渲染
# --------------------------------------------------------------------------
def make_background(scene: Scene, rng: np.random.Generator) -> np.ndarray:
    """静态背景 + 两个"陷阱"干扰物。

      - 亮白色方块：颜色和球完全一样（白），只有形状指标能筛掉它；
      - 橙色圆：形状和球完全一样（圆），只有颜色指标能筛掉它。
    两个陷阱都躲过去，才说明检测器真的有判别力，而不是"恰好看对了"。
    """
    w, h = scene.width, scene.height
    # 相机平移时背景要画得比画面宽，后面从里面滑窗取景；
    # 不平移时必须严格等于画面宽，否则整个场景就变了（干扰物位置、相机内参全跟着变）。
    ww = int(w + scene.pan_max) if scene.pan_speed else w
    grad = np.linspace(38, 118, h, dtype=np.float32)[:, None]
    img = np.dstack([np.repeat(grad, ww, axis=1)] * 3).astype(np.uint8)
    if scene.distractors:
        # 坐标按"画面宽 w"放，保证不平移时与原来完全一致；平移时靠滑窗自然进出画面
        cv2.rectangle(img, (int(w * 0.70), int(h * 0.10)), (int(w * 0.88), int(h * 0.30)), (238, 238, 238), -1)
        cv2.circle(img, (int(w * 0.83), int(h * 0.78)), 36, (40, 130, 240), -1, cv2.LINE_AA)
        for i in range(7):
            y = int(h * (0.08 + 0.13 * i))
            cv2.line(img, (0, y), (ww, y), (int(58 + 9 * i), int(54 + 7 * i), int(50 + 5 * i)), 2)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    if not scene.noise:
        return img
    return np.clip(img.astype(np.float32) + rng.normal(0, 2.5, img.shape), 0, 255).astype(np.uint8)


def _ball_coverage(
    bbox: tuple[int, int, int, int],
    samples: list[tuple[float, float, float]],
    supersample: int = 4,
) -> np.ndarray:
    """在给定 ROI 内计算球体的覆盖率（0-1），用 4 倍超采样保证亚像素精度。

    为什么不能用 cv2.circle 直接画：它的半径参数是整数，会把半径量化到
    0.5 px 以内，而我们要检验的亚像素细化能力是 0.05 px 量级 ——
    渲染精度不够，测试就失去了意义。
    """
    x0, y0, x1, y1 = bbox
    h, w = y1 - y0, x1 - x0
    ss = supersample
    big = np.zeros((h * ss, w * ss), dtype=np.uint8)
    for u, v, r in samples:
        cv2.circle(
            big,
            (int(round((u - x0) * ss)), int(round((v - y0) * ss))),
            max(1, int(round(r * ss))),
            255, -1, cv2.LINE_8,
        )
    return cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def render_frame(
    scene: Scene,
    bg: np.ndarray,
    t: float,
    rng: np.random.Generator | None = None,
    K: np.ndarray | None = None,
    D: np.ndarray | None = None,
    cam: dict | None = None,
) -> np.ndarray:
    """渲染一帧。运动模糊按"曝光时间 x 速度"积分，和真实相机一致。

    相机平移时，从加宽的背景里滑窗取景 —— 取景窗右移等价于画面内容左移，
    与 Scene.project() 里的 -pan_offset 保持一致。

    `K` / `D` / `cam` 是给双目用的：把"哪一路相机"的参数透传给 project()，
    于是同一套渲染代码能画出左路和右路的画面（`cam` 里放 side/R/T）。
    """
    cam = dict(cam or {})
    if K is not None:
        cam["K"] = K
    if D is not None:
        cam["D"] = D
    if scene.pan_speed:
        W0 = scene.width
        off = scene.pan_offset(t)
        x0 = max(0, min(int(math.floor(off)), bg.shape[1] - W0 - 1))
        img = bg[:, x0:x0 + W0].copy()
        # 亚像素平移：先按整数取窗，再把小数部分用 warpAffine 补上。
        # 只按整数取窗会让相机运动变成"阶梯"，给球的位置凭空加上 ±0.5 px 抖动，
        # 那会污染预测精度的测量（看起来像"相机运动让预测变差"）。
        frac = off - x0
        if abs(frac) > 1e-6:
            img = cv2.warpAffine(
                img, np.float32([[1, 0, -frac], [0, 1, 0]]), (W0, scene.height),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
            )
    else:
        img = bg.copy()
    u, v, r = scene.project(t, **cam)
    u_prev, v_prev, _ = scene.project(t - scene.dt, **cam)
    speed = math.hypot(u - u_prev, v - v_prev) * scene.fps
    exposure = 1.0 / (2.0 * scene.fps)      # 假设快门时间是帧间隔的一半
    smear = speed * exposure
    steps = 1
    if scene.motion_blur and smear > 1.5:
        steps = int(min(9, max(2, round(smear / 1.5))))

    samples = []
    for k in range(steps):
        tk = t - exposure + 2.0 * exposure * (k + 0.5) / steps
        samples.append(scene.project(tk, **cam))

    H, W = img.shape[:2]
    pad = 3.0
    x0 = max(0, int(math.floor(min(s[0] - s[2] for s in samples) - pad)))
    x1 = min(W, int(math.ceil(max(s[0] + s[2] for s in samples) + pad)))
    y0 = max(0, int(math.floor(min(s[1] - s[2] for s in samples) - pad)))
    y1 = min(H, int(math.ceil(max(s[1] + s[2] for s in samples) + pad)))
    if x1 > x0 and y1 > y0:
        coverage = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
        weight = 1.0 / len(samples)
        for s in samples:
            coverage += _ball_coverage((x0, y0, x1, y1), [s]) * weight
        roi = img[y0:y1, x0:x1].astype(np.float32)
        img[y0:y1, x0:x1] = np.clip(roi * (1.0 - coverage[..., None]) + 255.0 * coverage[..., None], 0, 255).astype(np.uint8)

    # 球内一点高光（不影响边界）。
    # 注意不要沿边界再画一圈"暗边"：那会在半径 round(r) 处做出第二条
    # 量化过的硬边缘，亚像素细化会锁到那条假边上去。
    hl_r = max(2.0, r * 0.20)
    hb = (int(u - r * 0.3), int(v - r * 0.35))
    cv2.circle(img, hb, int(round(hl_r)), (255, 255, 255), -1, cv2.LINE_AA)
    if not scene.noise:
        return img
    if rng is None:
        rng = np.random.default_rng(0)
    return np.clip(img.astype(np.float32) + rng.normal(0, 2.0, img.shape), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# 单个用例
# --------------------------------------------------------------------------
HORIZONS = (0.1, 0.2, 0.3)


def run_case(
    method: str = "fusion",
    preset: str = "white_ball",
    predict: str = "quadratic",
    gravity_fit: bool = False,
    write_video: bool = False,
    scenario: str = "realistic",
    verbose: bool = False,
) -> dict:
    realistic = scenario in ("realistic", "panning")
    panning = scenario == "panning"
    scene = Scene(
        motion_blur=realistic, noise=realistic, distractors=True,
        # 相机匀速横移 90 px/s（1080p 下相当于手持慢摇/云台慢速平移）
        pan_speed=90.0 if panning else 0.0,
    )
    rng = np.random.default_rng(20240925)
    bg = make_background(scene, rng)

    cfg = AppConfig()
    # motion_raw = 背景差分但不做相机运动补偿，用来对照"补偿到底有没有用"
    cfg.detector.method = "motion" if method == "motion_raw" else method
    if method == "motion_raw":
        cfg.detector.motion_compensation = "off"
    cfg.detector.color = ColorConfig.preset(preset)
    cfg.predict.method = predict
    cfg.predict.use_gravity = gravity_fit
    cfg.predict.horizon_s = 0.4
    cfg.predict.min_points = 6
    cam = CameraModel(
        width=scene.width, height=scene.height, hfov_deg=scene.hfov_deg,
        ball_diameter_m=scene.ball_diameter_m,
    )
    cfg.camera.hfov_deg = scene.hfov_deg
    cfg.camera.fx = cam.fx
    cfg.camera.ball_diameter_m = scene.ball_diameter_m

    pipe = SpherePipeline(cfg)
    pipe.camera = cam

    # 导出两个视频：
    #   synthetic_ball.mp4            原始画面（没有任何标注）—— 喂给 run_video.py 做端到端测试
    #   synthetic_ball_annotated.mp4  带标注的画面 —— 直接肉眼检查效果
    writer = None
    writer_ann = None
    if write_video:
        out_dir = ROOT / "outputs" / "videos"
        out_dir.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_dir / "synthetic_ball.mp4"), fourcc, scene.fps,
                                 (scene.width, scene.height))
        writer_ann = cv2.VideoWriter(str(out_dir / "synthetic_ball_annotated.mp4"), fourcc, scene.fps,
                                     (scene.width, scene.height))

    err_det: list[float] = []
    err_trk: list[float] = []
    err_r: list[float] = []
    err_3d: list[float] = []
    err_z_rel: list[float] = []
    # 预测误差分两档统计：
    #   steady   —— 距上次反弹 >0.15 s，运动规律稳定，这才是"正常水平"
    #   recovery —— 距上次反弹 <=0.15 s，刚被打断，只有两三个点可用，
    #               精度天然差一个量级（物理上不可能更好：速度突变要两次
    #               观测才能测出来）
    preds: dict[float, list[float]] = {h: [] for h in HORIZONS}
    preds_rec: dict[float, list[float]] = {h: [] for h in HORIZONS}
    det_hits = 0
    det_expected = 0
    t_pipeline = 0.0

    for i in range(scene.n):
        t = i / scene.fps
        frame = render_frame(scene, bg, t, rng)
        t0 = time.perf_counter()
        result = pipe.process(frame, t)
        t_pipeline += time.perf_counter() - t0      # 只统计流水线耗时，不含合成渲染
        gt_uv = np.array(scene.center(t))
        gt_r = scene.radius(t)

        visible = scene.fully_visible(t)
        if visible:
            det_expected += 1
        if result.detection is not None:
            det_hits += 1
            err_det.append(float(np.linalg.norm(np.array([result.detection.x, result.detection.y]) - gt_uv)))
        if result.track is not None and not result.track.coasting:
            err_trk.append(float(np.linalg.norm(np.array([result.track.x, result.track.y]) - gt_uv)))
            err_r.append(abs(result.track.radius - gt_r))
            # 三维指标只在相机不动时统计：本场景的"平移"是纯图像平移，
            # 不对应一致的相机三维运动，此时重建出的三维坐标是在"当前相机系"下，
            # 与固定世界系的真值没有可比性 —— 报出来只会误导。
            if result.position_3d is not None and not panning:
                gt3 = scene.center3d(t)
                err_3d.append(float(np.linalg.norm(result.position_3d - gt3)))
                err_z_rel.append(abs(result.position_3d[2] - gt3[2]) / gt3[2])

        # 预测精度：只统计"预测窗口内没有反弹/击打"的帧（撞了以后谁也不可能预测对）
        if result.prediction is not None and len(result.prediction) > 1:
            for h in HORIZONS:
                if t + h > scene.duration or scene.bounce_near(t, t + h + 0.02):
                    continue
                # 在预测点列上按时间线性插值到精确的 t+h，
                # 否则"取最近的那个离散点"会引入零点几帧的系统偏差
                idx = h / cfg.predict.horizon_s * (len(result.prediction) - 1)
                idx = max(0.0, min(idx, len(result.prediction) - 1.0))
                lo, hi = int(math.floor(idx)), int(math.ceil(idx))
                frac = idx - lo
                pt = result.prediction[lo][:2] * (1.0 - frac) + result.prediction[hi][:2] * frac
                gt = np.array(scene.center(t + h))
                err = float(np.linalg.norm(pt - gt))
                last_b = max([b for b in scene.bounces if b <= t], default=-99.0)
                (preds_rec if (t - last_b) <= 0.15 else preds)[h].append(err)

        if writer is not None:
            writer.write(frame)                    # 原始帧，不加任何标注
        if writer_ann is not None:
            writer_ann.write(pipe.render(frame, result))

    for w in (writer, writer_ann):
        if w is not None:
            w.release()
    elapsed = t_pipeline

    def mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    def rmse(xs: list[float]) -> float:
        return float(np.sqrt(np.mean(np.square(xs)))) if xs else float("nan")

    out = {
        "method": method,
        "preset": preset,
        "scenario": scenario,
        "detect_rate": det_hits / max(det_expected, 1),
        "visible_frames": det_expected,
        "det_rmse_px": rmse(err_det),
        "det_median_px": float(np.median(err_det)) if err_det else float("nan"),
        "center_rmse_px": rmse(err_trk),
        "center_median_px": float(np.median(err_trk)) if err_trk else float("nan"),
        "radius_rmse_px": rmse(err_r),
        "pos3d_rmse_m": rmse(err_3d),
        "depth_rel_err": mean(err_z_rel),
        "fps_processed": scene.n / elapsed,
        "ms_per_frame": 1000.0 * elapsed / scene.n,
    }
    for h in HORIZONS:
        out[f"pred_{h}s_px"] = mean(preds[h])
        out[f"pred_{h}s_median"] = float(np.median(preds[h])) if preds[h] else float("nan")
        out[f"pred_{h}s_n"] = len(preds[h])
        out[f"rec_{h}s_px"] = mean(preds_rec[h])
        out[f"rec_{h}s_median"] = float(np.median(preds_rec[h])) if preds_rec[h] else float("nan")
    if verbose:
        print(out)
    return out


# --------------------------------------------------------------------------
# 运动学判据（`stereo/motion.py`）：真值已知，所以误差可以直接量
# --------------------------------------------------------------------------
# 为什么放在这个文件里：上面那些用例检验"能不能看见球"，这一段检验"看见之后
# 算出来的速度/加速度对不对"。两者是同一套思路 —— 用合成真值，而不是肉眼看图。
#
# 这四条判据各自钉死一件事（都对应下游真会踩的坑）：
#
#   1. 匀速直线      差分本身不能引入误差（误差 < 1e-9）—— 否则滤波的收益无法归因；
#   2. 丢帧          缺口两侧不许"连着算"（差分和平滑都不许）—— 跨缺口一次就能
#                    把 |a| 抬到几十 m/s²，卡尔曼会当成真观测吃进去；
#   3. 噪声          平滑必须真的降抖动（≥20%）—— 不然"平滑"只是个空动作；
#   4. 短段          凑不出窗口就必须标 `valid=False` —— 不许装成"已平滑"。
MOTION_FPS = 30.0
MOTION_SPEED = 0.5          # m/s，匀速直线的真值速度
MOTION_GAP_S = 0.10         # 与 stereo.motion.DEFAULT_MAX_GAP_S 同口径


def _straight(t: np.ndarray, speed: float = MOTION_SPEED) -> np.ndarray:
    """沿 +X 匀速的直线，返回 (N,3) 米。"""
    return np.column_stack([speed * t, np.zeros_like(t), np.zeros_like(t)])


def _jitter(xyz: np.ndarray) -> float:
    """抖动指标：二阶差分模长的**中位数**（稳态下只剩噪声，不含真实加速度）。"""
    return float(np.median(np.linalg.norm(np.diff(xyz, n=2, axis=0), axis=1)))


def test_motion_uniform_velocity() -> bool:
    """匀速直线：速度必须逐点等于真值（中心/单边差分都要精确）。"""
    t = np.arange(0.0, 3.0, 1.0 / MOTION_FPS)
    kin = kinematics(t, _straight(t), smooth=False)
    err = float(np.max(np.abs(kin["speed"] - MOTION_SPEED)))
    ok = err < 1e-9
    print(f"[{'通过' if ok else '不达标'}] 匀速直线 {MOTION_SPEED} m/s："
          f"速度最大误差 {err:.1e} m/s（阈值 1e-9，{t.size} 点）")
    return ok


def test_motion_gap_does_not_leak_velocity() -> bool:
    """丢帧：切成 2 段，且差分与平滑都不许跨过缺口。

    **缺口处刻意放一个折角**（前段走 +X、后段走 +Y）。理由：如果数据本身是
    一条直线，跨缺口插值出来的也还是那条直线 —— 错误**不可观测**。有了折角，
    "整条当一段"就会把折角抹圆，位置偏差是毫米级、速度直接掉到 0.05 m/s。
    """
    t = np.arange(0.0, 3.0, 1.0 / MOTION_FPS)
    keep = np.ones(t.size, dtype=bool)
    keep[40:50] = False                     # 中间挖掉 10 帧（0.33 s 缺口）
    t2 = t[keep]

    x0 = MOTION_SPEED * t[39]               # 折角：缺口前走 +X，缺口后走 +Y
    before = t2 <= t[39] + 1e-9
    P2 = np.zeros((t2.size, 3))
    P2[before, 0] = MOTION_SPEED * t2[before]
    P2[~before, 0] = x0
    P2[~before, 1] = MOTION_SPEED * (t2[~before] - t[50])

    segs = split_segments(t2, MOTION_GAP_S)
    n_seg = len(segs)

    # (a) 原始差分：跨缺口算一步，速度会从 0.5 掉到 ~0.05（位置连续、时间跳 10 帧）
    kin = kinematics(t2, P2, smooth=False)
    speed_err = float(np.max(np.abs(kin["speed"] - MOTION_SPEED)))
    seg_ids = sorted(set(kin["segment"].tolist()))

    # (b) 平滑：每段都是精确直线，所以平滑结果必须仍在各自直线上
    Ps, valid = smooth_positions(t2, P2, window_s=0.20, poly=2)
    ref = np.zeros_like(P2)
    ref[before, 0] = MOTION_SPEED * t2[before]
    ref[~before, 0] = x0
    ref[~before, 1] = MOTION_SPEED * (t2[~before] - t[50])
    bend = float(np.max(np.linalg.norm(Ps - ref, axis=1)))

    ok = (n_seg == 2 and speed_err < 1e-9 and bend < 1e-6
          and seg_ids == [0, 1] and bool(valid.all()))
    print(f"[{'通过' if ok else '不达标'}] 丢帧（挖掉 10 帧）：段数 {n_seg}（应为 2）、"
          f"段号 {seg_ids}；原始速度误差 {speed_err:.1e} m/s（阈值 1e-9）；"
          f"平滑后偏离本段直线 {bend * 1000:.3f} mm（阈值 1e-6 m）")
    return ok


def test_motion_smoothing_cuts_jitter() -> bool:
    """噪声：σ=2 mm 高斯噪声下，平滑后的抖动必须比原始小 ≥20%。"""
    rng = np.random.default_rng(7)
    t = np.arange(0.0, 3.0, 1.0 / MOTION_FPS)
    P = np.column_stack([0.5 * t, 0.2 * t, np.zeros_like(t)])
    P = P + rng.normal(0.0, 0.002, P.shape)
    Ps, valid = smooth_positions(t, P, window_s=0.20, poly=2)
    raw, sm = _jitter(P), _jitter(Ps)
    drop = 1.0 - sm / raw
    ok = bool(valid.all()) and drop >= 0.20
    print(f"[{'通过' if ok else '不达标'}] 噪声 σ=2 mm：抖动 {raw * 1000:.2f} → "
          f"{sm * 1000:.2f} mm（降 {drop * 100:.0f}%，要求 ≥20%）；"
          f"全部点参与平滑={bool(valid.all())}")
    return ok


def test_motion_short_segment_is_marked() -> bool:
    """短段：2 点段必须 `valid=False`；单点段的速度必须是 nan（不是 0）。"""
    t = np.array([0.0, 1.0 / MOTION_FPS, 1.5])
    P = np.array([[0.0, 0.0, 0.0], [MOTION_SPEED / MOTION_FPS, 0.0, 0.0], [0.1, 0.0, 0.0]])

    _Ps, valid = smooth_positions(t, P)
    no_fake_smooth = not bool(valid.any())          # 2 点段 < poly+2，不许自称已平滑

    kin = kinematics(t, P, smooth=False)
    ok_seg = kin["segment"].tolist() == [0, 0, -1]
    ok_nan = bool(np.isnan(kin["V"][2]).all() and np.isnan(kin["speed"][2]))
    ok_acc = bool(np.isnan(kin["A"][:2]).all())     # 2 点凑不出加速度
    ok = no_fake_smooth and ok_seg and ok_nan and ok_acc
    print(f"[{'通过' if ok else '不达标'}] 短段：2 点段 valid={bool(valid.any())}"
          f"（应为 False）、段号 {kin['segment'].tolist()}（单点段应为 -1）、"
          f"单点段速度为 nan={ok_nan}、2 点段加速度为 nan={ok_acc}")
    return ok


def run_motion_checks() -> bool:
    """跑完四条运动学判据；任一不达标 → False。"""
    print("===== 运动学判据（stereo/motion.py：差分 / 丢帧 / 平滑 / 短段） =====")
    results = []
    for fn in (test_motion_uniform_velocity,
               test_motion_gap_does_not_leak_velocity,
               test_motion_smoothing_cuts_jitter,
               test_motion_short_segment_is_marked):
        results.append(fn())
    print()
    return all(results)


# --------------------------------------------------------------------------
# 通用球（近圆水果）合成场景：不依赖颜色的几何路径 + Q3 数值证伪
# --------------------------------------------------------------------------
# 为什么单独一个渲染分支：既有 render_frame():349 把球画成**纯白实心圆**，
# clean/realistic/panning 的真值、达标门限（:446/:723/:761）全钉在这个着色上。
# 改它一个像素就会动那三套数字。所以水果另起一套渲染：真值独立、门限独立，
# 只新增、不碰既有场景；也不新增 selftest_*.py 文件（26 个的计数判据）。
#
# Q3 的正确方向（§2/P11、§11.7，Leader 用真实代码预检后修正）：**不是**
# 「去掉新路径会退化」的消融，也**不是**「单球上高置信却偏心」——形状门会把
# 碎掩膜直接拦掉，那种场景不存在。真正的证伪是「沿用旧颜色范式给出好看却
# 无意义的数」：带斑水果 + **同色纯圆干扰物**同框，旧 HSV 把纯色干扰物排到
# detections[0]，自报高置信、last_error 为空，心却离真值 200 px 以上。
FRUIT_W, FRUIT_H = 640, 480
FRUIT_BG = 205                          # 灰底（BGR 205，S≈0，不进橙预设）
FRUIT_COLOR = (30, 110, 210)            # 水果与干扰物同色（orange_ball 预设内）
FRUIT_SPOT_COLOR = (150, 60, 60)        # 异色斑（蓝，不进橙预设 -> 挖洞）
FRUIT_DISTRACTOR = (150.0, 130.0, 30.0)   # 纯色完美圆（cx, cy, r）
FRUIT_N_FRAMES = 8
FRUIT_GATE_PX = 3.0             # 达标门限：圆心/半径误差 ≤3 px
FRUIT_Q3_SCORE_MIN = 0.80       # 旧路径「好看」：自报置信下限
FRUIT_Q3_MIN_ERR_PX = 100.0     # 「无意义」的错靶线，远高于 3 px 门限


def fruit_ground_truth() -> list[tuple[float, float, float]]:
    """N 帧真值 (cx, cy, r)。

    第 0 帧就是 Leader 预检的构造坐标 (320, 240)、等效半径 42；后续小幅平移
    （≤20 px）以形成可算 RMSE 的多帧序列。干扰物固定在 (150,130)，不动。
    seed 只取第 0 帧一次，不是逐帧喂真值。
    """
    out = []
    for i in range(FRUIT_N_FRAMES):
        t = i / (FRUIT_N_FRAMES - 1)
        out.append((320.0 + 18.0 * t, 240.0 + 8.0 * t, 42.0))
    return out


def render_fruit_frame(cx: float, cy: float, r: float,
                       rng: np.random.Generator) -> np.ndarray:
    """一帧：纯色圆干扰物 + 带 22 个异色斑的**同色**椭圆水果（Q3 构造）。

    * 干扰物与水果同为 BGR(30,110,210)，都落进 orange_ball 的 HSV 区间；
      干扰物是纯色完美圆、掩膜最完整，旧 color 因此把它排到 detections[0]。
    * 水果是近圆椭圆（长短轴 43:41，比值 0.953），球面 22 个异色斑
      (150,60,60) 在橙掩膜上挖洞；斑点全限制在 0.72 半径内，**不破坏闭合
      外轮廓**，几何路径仍能测到真实球缘（合成图的诚实约束，不是喂真值）。
    * 灰底 205（S≈0）与蓝色斑点都不进橙预设，只在水果掩膜里留洞。
    """
    img = np.full((FRUIT_H, FRUIT_W, 3), FRUIT_BG, np.uint8)
    dx, dy, dr = FRUIT_DISTRACTOR
    cv2.circle(img, (int(round(dx)), int(round(dy))), int(round(dr)),
               FRUIT_COLOR, -1)
    ax, ay = r * 43.0 / 42.0, r * 41.0 / 42.0    # 长短轴比 0.953
    cv2.ellipse(img, (int(round(cx)), int(round(cy))),
                (int(round(ax)), int(round(ay))), 0, 0, 360,
                FRUIT_COLOR, -1, cv2.LINE_AA)
    for _ in range(22):
        rr = int(rng.integers(4, 10))
        ang = rng.uniform(0.0, 2.0 * math.pi)
        rad = 0.72 * math.sqrt(rng.uniform(0.0, 1.0))
        px = int(round(cx + ax * rad * math.cos(ang)))
        py = int(round(cy + ay * rad * math.sin(ang)))
        cv2.circle(img, (px, py), rr, FRUIT_SPOT_COLOR, -1)
    return img


def render_fruit_frame_broken(cx: float, cy: float, r: float,
                              rng: np.random.Generator) -> np.ndarray:
    """「斑点啃到轮廓」的水果 + 同色纯圆干扰物 —— 种子门的 P0 现场。

    与 ``render_fruit_frame`` 只差**斑点参数**（Leader 扫参扫出来的，不是随手挑
    的）：斑点半径 5~13 px、半径占比放到 1.02，于是部分斑点压在外缘上，把 Canny
    的闭合外轮廓啃断 —— ``generic`` 对水果本体 **0 候选**，而旁边那个纯色干扰物
    照样检出（score≈0.96）。这正是 2026-11-07 Critic 75b 抓出的 P0 现场：
    门内空、门外有货，旧代码会默默把干扰物当结果返回。

    诚实约束：这里改的只是「果子表面有多脏」，真值圆心/半径仍是
    ``fruit_ground_truth()``，没有动真值去迁就检测器；判据里另有一条
    ``meta_ok`` 在验「门内确实空、门外确实有」，防止这道检查变成碰巧过。
    """
    img = np.full((FRUIT_H, FRUIT_W, 3), FRUIT_BG, np.uint8)
    dx, dy, dr = FRUIT_DISTRACTOR
    cv2.circle(img, (int(round(dx)), int(round(dy))), int(round(dr)),
               FRUIT_COLOR, -1)
    ax, ay = r * 43.0 / 42.0, r * 41.0 / 42.0
    cv2.ellipse(img, (int(round(cx)), int(round(cy))),
                (int(round(ax)), int(round(ay))), 0, 0, 360,
                FRUIT_COLOR, -1, cv2.LINE_AA)
    for _ in range(22):
        rr = int(rng.integers(5, 14))
        ang = rng.uniform(0.0, 2.0 * math.pi)
        rad = 1.02 * math.sqrt(rng.uniform(0.0, 1.0))
        px = int(round(cx + ax * rad * math.cos(ang)))
        py = int(round(cy + ay * rad * math.sin(ang)))
        cv2.circle(img, (px, py), rr, (0, 0, 0), -1)
    return img


def run_fruit_checks() -> bool:
    """水果场景：真值 RMSE + Q3 数值证伪 + 圆形干扰物不误检（同一场景）。"""
    from sphere_tracker.config import ColorConfig, DetectorConfig
    from sphere_tracker.detectors import build_single_detector
    from sphere_tracker.detectors.color import ColorBlobDetector

    print("===== 场景：fruit（带斑水果 + 同色纯圆干扰物）=====")
    rng = np.random.default_rng(20261106)
    gts = fruit_ground_truth()
    # 只点选第 0 帧一次；后续帧水果平移 ≤20 px，仍在空间门（gate=2*r=84）内，
    # 不是逐帧把真值喂给检测器。
    cfg = DetectorConfig(color=ColorConfig.preset("orange_ball"))
    geom_plain = build_single_detector("generic", cfg)     # 无 seed：诚实多候选
    geom_seeded = build_single_detector("generic", cfg)
    geom_seeded.seed(*gts[0])

    e_c: list[float] = []
    e_g: list[float] = []
    er_c: list[float] = []
    er_g: list[float] = []
    scores_c: list[float] = []
    last_errors: list[str] = []
    first_frame: np.ndarray | None = None
    for i, (cx, cy, r) in enumerate(gts):
        frame = render_fruit_frame(cx, cy, r, rng)
        if i == 0:
            first_frame = frame
        # 每帧新建 color：Q3 是「范式」的单帧行为，不让 _rank 的帧间重排
        # 把干扰物 / 水果的先后顺序搅成另一套结论。
        color_det = ColorBlobDetector(cfg)
        dc = color_det.detect(frame)
        dg = geom_seeded.detect(frame)
        if not dc or not dg:
            print(f"  [不达标] 帧({cx:.0f},{cy:.0f}) 检出缺失："
                  f"color={len(dc)} generic={len(dg)}")
            return False
        top_c = dc[0]                     # 下游默认消费方式：取第一个
        top_g = dg[0]
        scores_c.append(top_c.score)
        last_errors.append(str(getattr(color_det, "last_error", "")))
        e_c.append(math.hypot(top_c.x - cx, top_c.y - cy))
        e_g.append(math.hypot(top_g.x - cx, top_g.y - cy))
        er_c.append(abs(top_c.radius - r))
        er_g.append(abs(top_g.radius - r))

    def rmse(xs: list[float]) -> float:
        return float(np.sqrt(np.mean(np.square(xs))))

    c_rmse, g_rmse = rmse(e_c), rmse(e_g)
    cr_rmse, gr_rmse = rmse(er_c), rmse(er_g)
    min_score_c = min(scores_c)
    diag_empty = all(s == "" for s in last_errors)
    dx, dy, dr = FRUIT_DISTRACTOR
    print(f"  真值 {len(gts)} 帧（圆心 (320→338, 240→248)，等效半径 42 px）；"
          f"干扰物纯橙圆 ({dx:.0f},{dy:.0f}) r={dr:.0f}")
    print(f"  旧 HSV(color, orange_ball)：score 最小 {min_score_c:.3f}（门 ≥0.8）、"
          f"last_error 全空={diag_empty}、圆心 RMSE {c_rmse:.2f}px、"
          f"半径 RMSE {cr_rmse:.2f}px")
    print(f"        └ 第 0 帧：detections[0] 落在干扰物，score {scores_c[0]:.3f}、"
          f"距真值 {e_c[0]:.1f}px（门 {FRUIT_GATE_PX:.0f}px）")
    print(f"  新几何(generic, 一次 seed)：圆心 RMSE {g_rmse:.2f}px、"
          f"半径 RMSE {gr_rmse:.2f}px")
    # Q3：沿用旧颜色范式 -> 高置信 + 空诊断 + 远超门限的错靶；新几何路径在
    # 同一张图上命中真值。方向不是「去掉新路径会退化」的消融（§2/P11）。
    q3 = (min_score_c >= FRUIT_Q3_SCORE_MIN
          and diag_empty
          and c_rmse > FRUIT_GATE_PX
          and c_rmse > FRUIT_Q3_MIN_ERR_PX
          and g_rmse <= FRUIT_GATE_PX
          and gr_rmse <= FRUIT_GATE_PX)
    print(f"  [{'通过' if q3 else '不达标'}] Q3 数值证伪：旧 HSV 自报高置信、"
          f"诊断为空，却错靶 {c_rmse:.1f}px；几何路径误差达标")

    # ---- 同一场景的「圆形干扰物不误检」+ 判据（第 6 条与 Q3 合并）----
    cx, cy, r = gts[0]
    assert first_frame is not None
    ds = geom_seeded.detect(first_frame)
    dp = geom_plain.detect(first_frame)
    near_ball = sum(1 for d in ds
                    if math.hypot(d.x - cx, d.y - cy) <= FRUIT_GATE_PX)
    # (a) 给了 seed：全部候选都在真球上，干扰物不占 detections[0]，也不标歧义
    seeded_ok = (len(ds) >= 1 and len(ds) == near_ball
                 and all(not d.extra.get("geometry_ambiguous") for d in ds))
    # (b) 不给 seed：如实输出多个候选并给判据，绝不默默只给一个答案
    plain_ok = (len(dp) >= 2
                and all(d.extra.get("geometry_ambiguous") for d in dp)
                and all(d.extra.get("geometry_n_candidates") == len(dp) for d in dp))
    print("  圆形干扰物判据（不给 seed 时全部候选的几何量，纯几何下两者都合格、不可区分）：")
    for d in dp:
        p = d.parts
        tag = "水果" if math.hypot(d.x - cx, d.y - cy) <= FRUIT_GATE_PX else "干扰物"
        print(f"    {tag} c=({d.x:.0f},{d.y:.0f}) r={d.radius:.1f} "
              f"圆度={p['circularity']:.2f} 径向={p['radial']:.2f} "
              f"梯度支持={p['gradient_support']:.2f} 符号一致={p['sign_consistency']:.2f} "
              f"边缘支持={p['edge_support']:.2f}")
    print(f"  [{'通过' if (seeded_ok and plain_ok) else '不达标'}] 不误检："
          f"seed 后 {len(ds)} 个候选中 {near_ball} 个在真球；"
          f"无 seed 时 {len(dp)} 个候选全部标注 geometry_ambiguous（判据随候选给出）")

    # ---- (c) 种子门内**一个候选都没有**时，必须给锚点，绝不回落门外候选 ----
    # 2026-11-07 Critic 抓出的真实缺陷：原版 `if near: out = near`，
    # near 为空时保留 `out`（门外候选）照常返回 —— 同色圆干扰物会**默默占住
    # detections[0]**，而 last_error 还是空。现在要求：返回 1 个 seed_anchor，
    # 位置等于种子本身，且 last_error 说清原因。
    #
    # Critic 75b 复评指出第一版不合格：它用画面左上角 (5,5) 的**假种子**，
    # 那是「画面里压根没有候选」的角落退化场景，不是原始复现。真正的复现必须
    # 同时成立两条 —— 门内（水果本体）0 候选、门外（同色干扰物）**有**候选。
    # 下面 meta_ok 就是在证这两条，防止这道检查退化成碰巧过。
    broken = render_fruit_frame_broken(gts[0][0], gts[0][1], gts[0][2],
                                       np.random.default_rng(77))
    probe = build_single_detector("generic", geom_seeded.cfg)
    probe_out = probe.detect(broken)
    in_gate = [d for d in probe_out
               if math.hypot(d.x - gts[0][0], d.y - gts[0][1]) <= FRUIT_GATE_PX]
    out_gate = [d for d in probe_out
                if math.hypot(d.x - FRUIT_DISTRACTOR[0],
                              d.y - FRUIT_DISTRACTOR[1]) <= 6.0]
    meta_ok = (not in_gate) and bool(out_gate)
    anchor = build_single_detector("generic", geom_seeded.cfg)
    anchor.seed(*gts[0])
    dc = anchor.detect(broken)
    anchor_ok = (len(dc) == 1
                 and dc[0].extra.get("seed_anchor") is True
                 and math.hypot(dc[0].x - gts[0][0], dc[0].y - gts[0][1]) < 1e-6
                 and bool(getattr(anchor, "last_error", "")))
    print(f"  [{'通过' if (meta_ok and anchor_ok) else '不达标'}] "
          f"seed 门内无候选 -> 返回点选锚点"
          f"（门内 {len(in_gate)} 个 / 门外干扰物 {len(out_gate)} 个候选，"
          f"现场成立={meta_ok}；返回 {len(dc)} 个，"
          f"seed_anchor={dc[0].extra.get('seed_anchor') if dc else None}，"
          f"离种子 {min((math.hypot(d.x - gts[0][0], d.y - gts[0][1]) for d in dc), default=float('nan')):.2f}px）"
          f"；last_error 非空={bool(getattr(anchor, 'last_error', ''))}")

    # ---- (d) 0 检出不静默：全空场景必须给出可执行的 last_error ----
    blank = build_single_detector("generic", geom_seeded.cfg)
    db = blank.detect(np.full_like(first_frame, 205, dtype=np.uint8))
    blank_ok = (not db) and bool(getattr(blank, "last_error", ""))
    print(f"  [{'通过' if blank_ok else '不达标'}] 纯色空场景 0 检出但 last_error 非空"
          f"（静默失效检查）：{str(getattr(blank, 'last_error', ''))[:60]}…")

    # ---- (e) last_error 必须**逐帧清空**：上一帧的原因不许带到这一帧 ----
    # Critic 75b 的 exp4：把 detect() 开头那行 `self.last_error = ""` 删掉，
    # 第二帧明明有检出却仍带着第一帧的原因。而 (d) 用**新** detector 实例，
    # 删掉那行照样绿 —— 那行等于没人守。这里用**同一个实例**连打两帧。
    seq = build_single_detector("generic", geom_seeded.cfg)
    d1 = seq.detect(np.full_like(first_frame, 205, dtype=np.uint8))
    err1 = str(getattr(seq, "last_error", ""))
    d2 = seq.detect(first_frame)
    err2 = str(getattr(seq, "last_error", ""))
    reset_ok = (not d1) and bool(err1) and bool(d2) and err2 == ""
    print(f"  [{'通过' if reset_ok else '不达标'}] last_error 逐帧清空："
          f"第 1 帧 0 检出且给了原因={bool(err1)}，"
          f"第 2 帧检出 {len(d2)} 个、原因已清空={err2 == ''}")

    # ---- (f) 锚点诊断必须走到**消费端**，且**四个检测器**都要走得通 ----
    # 75b 抓的是 pipeline 只在 `best is None` 时读 last_error；(f) 当时只测了
    # generic，结果 75c 复评抓出另外三路各写各的：
    #   · color   —— 有 `seed_anchor` 但 `_anchor_detection` **不写** last_error；
    #   · fusion  —— 透传成员的 extra（含 seed_anchor），却在成功分支把 last_error 清空；
    #   · specular—— 兜底只写 `state="seed-anchor"`，**根本没有** `seed_anchor` 键。
    # 后两条都让 pipeline 连分支都进不去。现在 pipeline 在「是锚点却没给原因」时
    # 自己补一句，所以判据钉两条：① 四路都带 `extra["seed_anchor"]`；
    # ② pipeline 对四路都给出非空 `detector_note`。
    from sphere_tracker.pipeline import SpherePipeline

    blank_scene = np.full((300, 400, 3), 128, dtype=np.uint8)
    # specular 用的亮暗相间球：故意让 CSRT 起不来，逼它走锚点兜底
    spec_img = np.full((300, 400, 3), 90, dtype=np.uint8)
    _yy, _xx = np.ogrid[:300, :400]
    _inside = (_xx - 200) ** 2 + (_yy - 150) ** 2 <= 20 ** 2
    spec_img[_inside & (((_xx // 4 + _yy // 4) % 2) == 0)] = (190, 190, 190)

    def _anchor_case(name: str, frame: np.ndarray, seed: tuple[float, float, float]):
        """让某个检测器走一遍锚点兜底，返回 (检测器, 帧)。"""
        d = build_single_detector(name, DetectorConfig(
            color=ColorConfig.preset("red_ball")))
        d.seed(*seed)
        if name == "specular":
            d._make_tracker = lambda: None      # type: ignore[assignment]
        return d, frame

    cases = [
        ("generic", broken, gts[0]),
        ("color", blank_scene, (200.0, 150.0, 30.0)),
        ("specular", spec_img, (200.0, 150.0, 20.0)),
        ("fusion", blank_scene, (200.0, 150.0, 30.0)),
    ]
    per_det: dict[str, tuple[bool, bool]] = {}
    for name, frame, seed in cases:
        d, fr = _anchor_case(name, frame, seed)
        pipe = SpherePipeline(AppConfig(), detector=d)
        res = pipe.process(fr, 0.0)
        marked = bool(res.detection is not None
                      and (res.detection.extra or {}).get("seed_anchor") is True)
        spoken = bool(res.detector_note)
        per_det[name] = (marked, spoken)
        print(f"    {name:<9} 标记={marked}  诊断非空={spoken}"
              f"  ← {str(res.detector_note)[:44]}…")
    note_ok = all(m and s for m, s in per_det.values())
    print(f"  [{'通过' if note_ok else '不达标'}] 锚点诊断到达消费端（四路）："
          f"{sum(1 for m, s in per_det.values() if m and s)}/4 路"
          f"「有标记 + 有话说」")

    print()
    return bool(q3 and seeded_ok and plain_ok and anchor_ok and blank_ok
                and reset_ok and note_ok)


def check_ai() -> None:
    print("\n=== AI 依赖检查 ===")
    for mod in ("torch", "ultralytics", "onnxruntime"):
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            print(f"  [已安装] {mod:12s} {ver}")
            if mod == "torch":
                print(f"           CUDA 可用: {m.cuda.is_available()}")
                if m.cuda.is_available():
                    cap = m.cuda.get_device_capability(0)
                    print(f"           设备: {m.cuda.get_device_name(0)}  算力 sm_{cap[0]}{cap[1]}")
        except ImportError:
            pkg = "onnxruntime-gpu" if mod == "onnxruntime" else mod
            print(f"  [未安装] {mod:12s} —— pip install {pkg}")
    print("  装上 ultralytics 后，可以用 --method yolo / yolo_sam 直接对比效果。")


def main() -> int:
    p = argparse.ArgumentParser(description="合成数据自测")
    p.add_argument("--method", default=None, help="只测某一种方法")
    p.add_argument("--preset", default="white_ball")
    p.add_argument("--predict", default="quadratic", choices=method_names("predictor"))
    p.add_argument("--gravity-fit", action="store_true", help="三维预测时用已知重力约束")
    p.add_argument("--write-video", action="store_true")
    p.add_argument("--check-ai", action="store_true")
    p.add_argument("--scenario", default="all",
                   choices=["clean", "realistic", "panning", "fruit", "all"],
                   help="panning = 相机匀速横移；fruit = 近圆水果（Q3 数值证伪 + 圆形干扰物）")
    p.add_argument("--gate", default="color,fusion", help="必须达标的方法（逗号分隔）")
    p.add_argument("--motion", action="store_true",
                   help="只跑运动学判据（stereo/motion.py），不跑合成视频流水线（秒级）")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.check_ai:
        check_ai()
        return 0

    motion_ok = run_motion_checks()
    if args.motion:
        print("判定结果：" + ("全部通过" if motion_ok else "有不达标项"))
        return 0 if motion_ok else 1

    scenarios = (["clean", "realistic", "panning", "fruit"]
                 if args.scenario == "all" else [args.scenario])
    # 平移场景额外加一个"不补偿"的对照组，直接看出补偿值不值。
    # fruit 走独立渲染 + 独立真值/门限（run_fruit_checks），不进 run_case 的循环：
    # 它的着色与既有纯白圆不同，混进同一套 limits/gate 会污染既有三场景的基线。
    methods_by_scenario = {
        "clean": ["color", "hough", "motion", "fusion"],
        "realistic": ["color", "hough", "motion", "fusion"],
        "panning": ["color", "motion_raw", "motion", "fusion"],
        "fruit": ["color", "generic"],
    }
    def methods_for(scenario: str) -> list[str]:
        if args.method:
            return [args.method]
        return methods_by_scenario.get(scenario, ["color", "fusion"])
    gate = {s.strip() for s in args.gate.split(",") if s.strip()}
    probe = Scene()
    print(f"合成场景：{probe.width}x{probe.height} @ {probe.fps:.0f} fps, {probe.n} 帧, "
          f"焦距 {probe.f:.0f} px, 球直径 {probe.ball_diameter_m * 100:.0f} cm")
    print(f"  真实三维运动 + 透视投影（像素轨迹不是标准抛物线）")
    print(f"  含：斜抛 / 落地反弹(恢复系数 {probe.restitution}) / 前后左右碰壁 / 深度变化 "
          f"{probe.pos[:, 2].min():.1f}-{probe.pos[:, 2].max():.1f} m")
    print(f"  陷阱干扰物：同色白方块 + 同形橙色圆")
    print(f"  clean = 无噪声无模糊（验几何）   realistic = 噪声 + 运动模糊（验鲁棒性）")
    print(f"预测方法：{args.predict}  重力约束：{args.gravity_fit}  必须达标：{sorted(gate)}")
    print(f"评估视界：{[f'{h}s' for h in HORIZONS]}\n")

    limits = {
        "detect_rate": (0.995, 0.90),
        "center_rmse_px": (2.0, 3.0),      # RMSE 会被"刚反弹"的几帧拉高，属正常
        "center_median_px": (0.80, 1.10),  # 中位数才是典型精度
        "radius_rmse_px": (0.30, 0.90),
        "pred_0.1s_px": (2.0, 4.0),
        "pred_0.2s_px": (5.0, 10.0),
        "pred_0.3s_px": (12.0, 22.0),
    }
    labels = {
        "detect_rate": "检出率", "center_rmse_px": "跟踪RMSE", "center_median_px": "跟踪中位数",
        "radius_rmse_px": "半径RMSE",
        "pred_0.1s_px": "稳定段0.1s", "pred_0.2s_px": "稳定段0.2s", "pred_0.3s_px": "稳定段0.3s",
    }

    ok = motion_ok
    for scenario in scenarios:
        if scenario == "fruit":
            ok = bool(ok and run_fruit_checks())
            continue
        print(f"===== 场景：{scenario} =====")
        print(f"{'方法':<9}{'检出率':>8}{'检测RMSE':>10}{'跟踪中位':>10}{'半径RMSE':>10}"
              f"{'三维RMSE':>10}{'深度误差':>9}"
              f"{'0.1s':>8}{'0.2s':>8}{'0.3s':>8}{'0.2s中':>8}{'恢复0.2s':>10}{'速度':>9}")
        print("   （预测误差单位 px；'0.x s' 为稳定段均值；'恢复0.2s' 为反弹后 0.15 s 内的均值）")
        print("-" * 135)
        scenario_methods = methods_for(scenario)
        for m in scenario_methods:
            r = run_case(
                m, args.preset, args.predict, args.gravity_fit, verbose=args.verbose,
                write_video=args.write_video and m == scenario_methods[-1] and scenario == scenarios[-1],
                scenario=scenario,
            )
            print(f"{r['method']:<9}{r['detect_rate'] * 100:7.1f}%{r['det_rmse_px']:9.2f}px"
                  f"{r['center_median_px']:9.2f}px{r['radius_rmse_px']:9.2f}px"
                  f"{r['pos3d_rmse_m'] * 1000:8.1f}mm{r['depth_rel_err'] * 100:8.2f}%"
                  f"{r['pred_0.1s_px']:8.2f}{r['pred_0.2s_px']:8.2f}{r['pred_0.3s_px']:8.2f}"
                  f"{r['pred_0.2s_median']:8.2f}{r['rec_0.2s_px']:10.2f}"
                  f"{r['fps_processed']:7.1f}fps")
            if m not in gate:
                continue
            idx = 0 if scenario == "clean" else 1
            bad = []
            for key, lim in limits.items():
                value = r[key]
                if math.isnan(value):
                    bad.append(f"{labels[key]}:无样本")
                elif key == "detect_rate":
                    if value < lim[idx]:
                        bad.append(f"{labels[key]} {value * 100:.1f}% < {lim[idx] * 100:.0f}%")
                elif value > lim[idx]:
                    bad.append(f"{labels[key]} {value:.2f} > {lim[idx]}")
            if bad:
                ok = False
                print(f"    [不达标] {m}: " + "; ".join(bad))
            else:
                print(f"    [通过]   {m}")
        print()

    if args.write_video:
        print("已导出：outputs/videos/synthetic_ball.mp4（原始画面，可喂给 apps/run_video.py）")
        print("        outputs/videos/synthetic_ball_annotated.mp4（带标注，直接看效果）")
    print("判定结果：" + ("全部通过" if ok else "有不达标项"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
