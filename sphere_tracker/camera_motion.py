"""相机自运动估计：判断"画面在动"是相机在动，还是球在动。

为什么必须做这件事
------------------
  - 背景差分（MOG2）假设背景静止。相机一平移，整幅画面都变成"前景"，
    误检瞬间爆炸 —— 这是"相机不固定"时第一个崩掉的环节。
  - 卡尔曼滤波在图像坐标系里建模。相机在动时，球在图像里的运动 =
    球自身运动 + 相机运动，不区分两者，预测就会把相机运动也外推出去。
  - 单目测深 Z = fR/r 逐帧成立，与相机是否运动无关，不受影响。

为什么主力是"特征点 + RANSAC"而不是相位相关
--------------------------------------------
最初用 cv2.phaseCorrelate 做主力，结果在**完全静止**的合成场景里
报出 6-16 px 的位移 —— 因为背景不变时，帧间唯一的变化就是那颗移动的球，
相位相关的峰值自然锁到了球上，测出来的是"球的位移"而不是"相机的位移"。

正确的做法是让"多数派"说话：在背景上撒几百个特征点，
球只占其中很少一部分，RANSAC 会把球那些点当外点剔除，
剩下的多数派运动就是相机的运动。相机静止 -> 结果≈0；相机平移 -> 结果≈平移量。
相位相关只作为"画面没纹理、特征点太少"时的兜底。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MotionEstimate:
    """相邻两帧之间的全局运动（= 相机自运动）。"""

    dx: float = 0.0                 # 全分辨率下的水平位移（像素）
    dy: float = 0.0                 # 垂直位移
    rotation_deg: float = 0.0       # 旋转角（度）
    scale: float = 1.0              # 缩放
    response: float = 0.0           # 置信度：特征点法的内点比例 / 相位相关的响应
    inliers: int = 0                # 参与估计的内点数
    method: str = "none"            # "features" | "phase" | "none"
    matrix: np.ndarray | None = None  # 2x3（全分辨率），把"上一帧"对齐到"当前帧"

    @property
    def magnitude(self) -> float:
        return float(math.hypot(self.dx, self.dy))

    @property
    def has_transform(self) -> bool:
        return self.matrix is not None


class GlobalMotionEstimator:
    """估计相机自身运动（相邻两帧的全局运动）。

    参数
    ----
    work_width : 估计用的缩放宽度。320 足够，缩小后速度约 1-3 ms。
    threshold_px : 判定"相机在动"的位移阈值（全分辨率像素/帧）。
        静止相机实测约 0.0-0.2 px；手持轻微抖动 0.3-1.5 px；
        明显平移 5-50 px。
    hysteresis : 退出"移动"状态时用 threshold * hysteresis（避免状态抖动）。
    min_inliers : 特征点法的内点下限，低于此认为估计不可信。
    """

    def __init__(
        self,
        work_width: int = 320,
        threshold_px: float = 1.2,
        feature_fallback: bool = True,
        max_features: int = 250,
        min_inliers: int = 20,
        hysteresis: float = 0.5,
    ) -> None:
        self.work_width = int(work_width)
        self.threshold_px = float(threshold_px)
        self.feature_fallback = bool(feature_fallback)
        self.max_features = int(max_features)
        self.min_inliers = int(min_inliers)
        self.hysteresis = float(hysteresis)
        self._prev_small: np.ndarray | None = None
        self._scale = 1.0
        self._moving = False

    def reset(self) -> None:
        self._prev_small = None
        self._moving = False

    # ------------------------------------------------------------------
    def _prepare(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape[:2]
        self._scale = self.work_width / float(w)
        small = cv2.resize(
            gray, (self.work_width, max(1, int(round(h * self._scale)))),
            interpolation=cv2.INTER_AREA,
        )
        return small

    def estimate(self, prev_gray: np.ndarray | None, gray: np.ndarray) -> MotionEstimate:
        """估计 prev_gray -> gray 的全局运动（即相机相对场景的位移）。"""
        small = self._prepare(gray)
        prev_small = self._prev_small
        self._prev_small = small
        if prev_gray is None or prev_small is None or prev_small.shape != small.shape:
            return MotionEstimate(method="none")

        # 三级阶梯，越往下越"退而求其次"：
        #   1) 角点 + RANSAC 相似变换：最准，还能给出旋转/缩放（手持）；
        #   2) 网格点 + LK + 中位数：画面没角点时用，中位数天然忽略少数派的运动物体；
        #   3) 相位相关：最后兜底，但它会被高对比度运动物体带偏，只作参考。
        est = self._estimate_features(prev_small, small)
        if est is None:
            est = self._estimate_grid(prev_small, small)
        if est is None and self.feature_fallback:
            est = self._estimate_phase(prev_small, small)
        if est is None:
            return MotionEstimate(method="none")
        return est

    # ------------------------------------------------------------------
    def _estimate_grid(self, prev_small: np.ndarray, small: np.ndarray):
        """网格点 + LK 光流 + 中位数位移。

        为什么用中位数：画面里绝大多数点是背景，球只占很小一部分。
        中位数对"少数派"免疫，所以球怎么动都不会影响相机运动的估计。
        这也正是相位相关做不到的地方 —— 它会被高对比度的球整个带偏。
        """
        h, w = prev_small.shape[:2]
        step = max(8, int(round(min(h, w) / 12.0)))
        ys, xs = np.mgrid[step:h - step:step, step:w - step:step]
        if xs.size < 12:
            return None
        pts = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32).reshape(-1, 1, 2)
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_small, small, pts, None, winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if nxt is None or status is None:
            return None
        ok = status.reshape(-1).astype(bool)
        if int(ok.sum()) < 12:
            return None
        p0 = pts.reshape(-1, 2)[ok]
        p1 = nxt.reshape(-1, 2)[ok]
        delta = p1 - p0
        med = np.median(delta, axis=0)                 # 中位数：抗运动物体
        mad = np.median(np.abs(delta - med), axis=0)   # 离散度，用来估置信度
        spread = float(np.hypot(mad[0], mad[1]))
        n = int(ok.sum())
        # 一致性好（离散度小）说明确实是全局运动
        confidence = float(np.clip(1.0 - spread / 3.0, 0.0, 1.0)) * (n / float(n + 8))
        s = max(self._scale, 1e-6)
        dx, dy = float(med[0]) / s, float(med[1]) / s
        return MotionEstimate(
            dx=dx, dy=dy, response=confidence, inliers=n, method="grid",
            matrix=np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32),
        )

    # ------------------------------------------------------------------
    def _estimate_features(self, prev_small: np.ndarray, small: np.ndarray):
        """特征点 + LK 光流 + RANSAC 相似变换。多数派（背景）决定结果。"""
        pts = cv2.goodFeaturesToTrack(
            prev_small, maxCorners=self.max_features, qualityLevel=0.01, minDistance=8
        )
        if pts is None or len(pts) < self.min_inliers:
            return None
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev_small, small, pts, None)
        if nxt is None or status is None:
            return None
        ok = status.reshape(-1).astype(bool)
        if int(ok.sum()) < self.min_inliers:
            return None
        p0 = pts.reshape(-1, 2)[ok]      # 上一帧里的位置
        p1 = nxt.reshape(-1, 2)[ok]      # 当前帧里的位置
        # estimateAffinePartial2D(from=p0, to=p1) 给出把"上一帧坐标"映射到
        # "当前帧坐标"的相似变换；warpAffine(prev, M) 正好把上一帧对齐到当前帧。
        M_small, inliers = cv2.estimateAffinePartial2D(
            p0, p1, method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=2000,
            confidence=0.995, refineIters=10,
        )
        if M_small is None:
            return None
        n_in = int(inliers.sum()) if inliers is not None else 0
        if n_in < self.min_inliers:
            return None

        # 小图尺度 -> 全分辨率尺度
        s = max(self._scale, 1e-6)
        S = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=np.float64)
        Si = np.array([[1 / s, 0, 0], [0, 1 / s, 0], [0, 0, 1]], dtype=np.float64)
        M_full = (S @ np.vstack([M_small, [0.0, 0.0, 1.0]]) @ Si)[:2]

        dx, dy = float(M_full[0, 2]), float(M_full[1, 2])
        rot = math.degrees(math.atan2(float(M_full[1, 0]), float(M_full[0, 0])))
        scale = math.sqrt(abs(float(np.linalg.det(M_full[:, :2]))))
        return MotionEstimate(
            dx=dx, dy=dy, rotation_deg=rot, scale=scale,
            response=n_in / float(max(len(p0), 1)), inliers=n_in,
            method="features", matrix=M_full.astype(np.float32),
        )

    # ------------------------------------------------------------------
    def _estimate_phase(self, prev_small: np.ndarray, small: np.ndarray):
        """兜底：相位相关（纯平移）。画面没纹理、特征点太少时用。

        注意：场景里如果有高对比度的运动物体，它可能主导结果 ——
        所以只作为兜底，并且把置信度原样报出去供上层参考。
        """
        p = prev_small.astype(np.float32)
        c = small.astype(np.float32)
        hann = cv2.createHanningWindow((c.shape[1], c.shape[0]), cv2.CV_32F)
        try:
            (sx, sy), response = cv2.phaseCorrelate(p, c, hann)
        except cv2.error:
            return None
        s = max(self._scale, 1e-6)
        dx, dy = sx / s, sy / s
        return MotionEstimate(
            dx=dx, dy=dy, response=float(response), method="phase",
            matrix=np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32),
        )

    # ------------------------------------------------------------------
    def is_moving(self, motion: MotionEstimate | None) -> bool:
        """带迟滞的"相机是否在动"判定，避免在阈值附近来回抖动。

        进入"移动"的门限是 `threshold_px`，退出要等位移掉到
        `threshold_px * hysteresis`（默认 0.5 倍）以下 —— 两个门限分开，
        位移在门限附近抖动时状态才不会一帧一变。

        **这里必须把状态写回 `self._moving`**：判定结果同时决定
        "要不要用补偿差分"（`detectors/motion.py`）和
        "像素重力约束还成不成立"（`pipeline.py`），都不希望逐帧翻转。
        估计失败（`method == "none"`）时沿用上一帧状态，不要凭空复位。
        """
        if motion is None or motion.method == "none":
            return self._moving
        # 相位相关会被运动物体带偏，判"在动"时对它要求更高的位移，
        # 宁可漏判（退回 MOG2）也不要误判（用错模型补偿）。
        scale = 2.0 if motion.method == "phase" else 1.0
        enter_px = self.threshold_px * scale
        exit_px = self.threshold_px * self.hysteresis * scale
        if self._moving:
            if motion.magnitude < exit_px:
                self._moving = False
        elif motion.magnitude > enter_px:
            self._moving = True
        return self._moving


def warp_to_current(prev_frame: np.ndarray, motion: MotionEstimate | None) -> np.ndarray:
    """把上一帧按估计出的全局运动对齐到当前帧。"""
    if motion is None or motion.matrix is None:
        return prev_frame
    h, w = prev_frame.shape[:2]
    return cv2.warpAffine(
        prev_frame, motion.matrix, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
