"""极线约束配对 + 三角测量。

分工：
  - calib.StereoRig 负责数学部分（去畸变 → 立体校正 → 三角测量）；
  - 本模块负责**流程**：怎么把左图的检测点配到右图的候选上、怎么给出置信度。

核心是**极线约束**：校正之后，左图上的点在右图中的对应点必定在**同一条水平线**上。
这一下就把二维搜索降成了一维，是双目能实时跑起来的关键。

用法::

    from calib import StereoRig
    from stereo import StereoReconstructor

    rig = StereoRig.load("config/stereo_calibration")
    rec = StereoReconstructor(rig)
    pt = rec.update(t=1.23, left=det_left, right_dets=dets_right)
    if pt is not None:
        print(pt.xyz, pt.residual_px)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from calib import StereoRig
from sphere_tracker.types import Detection


@dataclass
class MatchConfig:
    max_epipolar_px: float = 2.5   # 允许的极线偏差（校正后 y 之差）
    min_disparity_px: float = 0.5  # 视差小于这个 -> 点在无穷远或配错了
    max_disparity_px: float = 600.0
    require_radius_ratio: bool = False  # 是否要求左右半径接近（同型号相机才合理）
    radius_ratio_tol: float = 0.35


@dataclass
class StereoPoint:
    """一个时刻的三维点及其置信度。"""

    t: float
    xyz: np.ndarray                # (3,) 米，**校正后左相机系**
    u_left: tuple[float, float]
    u_right: tuple[float, float]
    epipolar_px: float             # 配对时的极线偏差
    residual_px: float             # 三角测量后重投影回两路的误差（交会残差）
    radius_left: float = float("nan")

    @property
    def distance_m(self) -> float:
        return float(np.linalg.norm(self.xyz))

    @property
    def confident(self) -> bool:
        return self.residual_px <= 2.0


def rectify_point_left(rig: StereoRig, x: float, y: float) -> tuple[float, float]:
    """把一个左图像素点映射到"校正后的左相机"坐标系。"""
    p = cv2.undistortPoints(np.array([[[x, y]]], np.float64), rig.K1, rig.D1,
                            R=rig.R1, P=rig.P1)
    return float(p[0, 0, 0]), float(p[0, 0, 1])


def rectify_point_right(rig: StereoRig, x: float, y: float) -> tuple[float, float]:
    """把一个右图像素点映射到"校正后的右相机"坐标系。"""
    p = cv2.undistortPoints(np.array([[[x, y]]], np.float64), rig.K2, rig.D2,
                            R=rig.R2, P=rig.P2)
    return float(p[0, 0, 0]), float(p[0, 0, 1])


def match_epipolar(
    rig: StereoRig,
    det_left: Detection,
    candidates_right: list[Detection],
    cfg: MatchConfig | None = None,
) -> tuple[Detection | None, float]:
    """在右图候选里挑出与左图检测点对应的那一个。

    返回 (最佳候选, 极线偏差)。找不到合适的返回 (None, inf)。
    """
    cfg = cfg or MatchConfig()
    if not candidates_right:
        return None, float("inf")

    x1, y1 = rectify_point_left(rig, det_left.x, det_left.y)

    best, best_cost, best_dy = None, float("inf"), float("inf")
    for cand in candidates_right:
        x2, y2 = rectify_point_right(rig, cand.x, cand.y)
        dy = abs(y1 - y2)
        if dy > cfg.max_epipolar_px:
            continue
        disparity = x1 - x2
        if not (cfg.min_disparity_px <= disparity <= cfg.max_disparity_px):
            continue
        if cfg.require_radius_ratio and det_left.radius > 0 and cand.radius > 0:
            ratio = abs(cand.radius - det_left.radius) / det_left.radius
            if ratio > cfg.radius_ratio_tol:
                continue
        cost = dy
        if cost < best_cost:
            best, best_cost, best_dy = cand, cost, dy
    return best, (best_dy if best is not None else float("inf"))


def triangulate_one(
    rig: StereoRig,
    det_left: Detection,
    det_right: Detection,
    t: float,
    epipolar_px: float = float("nan"),
) -> StereoPoint:
    """单点三角测量 + 交会残差。"""
    pts_l = np.array([[det_left.x, det_left.y]], np.float64)
    pts_r = np.array([[det_right.x, det_right.y]], np.float64)
    xyz = rig.triangulate(pts_l, pts_r)[0]

    # 交会残差：把三维点投回两路，看和用于三角测量的像素差多少。
    #
    # 注意这里必须用**校正后**的量：xyz 是在校正后左相机系下的，
    # 对应的像素也是校正后的像素。用原始 K1/K2 和原始像素去比，
    # 会得到一个几十像素的假残差（立体校正的旋转 R1 没算进去）。
    p1 = np.array(rectify_point_left(rig, det_left.x, det_left.y))
    p2 = np.array(rectify_point_right(rig, det_right.x, det_right.y))
    e_l = float(np.linalg.norm(_project(rig.P1, xyz) - p1))
    e_r = float(np.linalg.norm(_project(rig.P2, xyz) - p2))
    residual = 0.5 * (e_l + e_r)

    return StereoPoint(
        t=float(t), xyz=np.asarray(xyz, np.float64),
        u_left=(det_left.x, det_left.y), u_right=(det_right.x, det_right.y),
        epipolar_px=float(epipolar_px), residual_px=residual,
        radius_left=float(det_left.radius),
    )


def _project(P: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """用 3x4 投影矩阵把（校正后左相机系下的）三维点投影到像素。"""
    h = np.asarray(P, np.float64) @ np.append(np.asarray(xyz, np.float64), 1.0)
    return h[:2] / h[2]


@dataclass
class StereoReconstructor:
    """把逐帧的左右检测配成一条三维轨迹。"""

    rig: StereoRig
    cfg: MatchConfig = field(default_factory=MatchConfig)
    trajectory: list[StereoPoint] = field(default_factory=list)

    # 统计
    n_frames: int = 0
    n_matched: int = 0

    def update(
        self,
        t: float,
        left: Detection | None,
        right_dets: list[Detection],
    ) -> StereoPoint | None:
        """喂一帧的左右检测，返回三维点（配不上则返回 None）。"""
        self.n_frames += 1
        if left is None or not right_dets:
            return None
        matched, dy = match_epipolar(self.rig, left, right_dets, self.cfg)
        if matched is None:
            return None
        pt = triangulate_one(self.rig, left, matched, t, dy)
        self.n_matched += 1
        self.trajectory.append(pt)
        return pt

    @property
    def match_rate(self) -> float:
        return self.n_matched / self.n_frames if self.n_frames else 0.0

    def as_array(self) -> np.ndarray:
        """(N,4) 的 [t, X, Y, Z]，方便直接喂给平滑/预测。"""
        if not self.trajectory:
            return np.zeros((0, 4))
        return np.array([[p.t, *p.xyz] for p in self.trajectory], np.float64)

    def summary(self) -> str:
        if not self.trajectory:
            return "没有任何配对成功的三维点"
        res = np.array([p.residual_px for p in self.trajectory], np.float64)
        res = res[np.isfinite(res)]
        z = np.array([p.xyz[2] for p in self.trajectory], np.float64)
        return (f"配对率 {self.match_rate * 100:.1f}%  ({self.n_matched}/{self.n_frames})  "
                f"交会残差 中位 {np.median(res):.3f} px / 最大 {res.max():.3f} px  "
                f"深度 {z.min():.2f}–{z.max():.2f} m")
