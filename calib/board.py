"""标定板定义与检测。

设计目标：为"自动标定"提供**可量化的质量指标**，让程序能自己判断
"这一帧该不该收"。只回答"检测到/没检测到"是不够的 ——
模糊的、贴边的、太小的、过曝的帧都会把标定结果拖坏。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class BoardSpec:
    """标定板规格。

    cols/rows 是**内角点**数量（不是方格数）。例如常见的 9x6 棋盘
    （10 列 9 行方格）内角点是 9x8。

    注意 findChessboardCorners 的参数顺序是 (列, 行)，
    即 pattern_size = (cols, rows)。
    """

    cols: int = 8                 # 内角点列数
    rows: int = 5                 # 内角点行数
    square_mm: float = 25.0       # 方格实际边长（毫米）

    @property
    def n_points(self) -> int:
        return self.cols * self.rows

    @property
    def pattern_size(self) -> tuple[int, int]:
        return (self.cols, self.rows)

    def object_points(self) -> np.ndarray:
        """棋盘格角点在"板坐标系"下的三维坐标（Z=0），单位与 square_mm 一致。

        用米做单位，方便标定结果直接和物理量对应。
        """
        s = self.square_mm / 1000.0
        objp = np.zeros((self.n_points, 3), np.float32)
        objp[:, :2] = np.mgrid[0 : self.cols, 0 : self.rows].T.reshape(-1, 2) * s
        return objp

    def unit_square_points(self) -> np.ndarray:
        """板平面上的归一化采样点（0..1），用来做位姿多样性度量。"""
        g = np.linspace(0.0, 1.0, 4)
        xs, ys = np.meshgrid(g, g)
        return np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)

    def describe(self) -> str:
        return f"棋盘格 {self.cols}x{self.rows} 内角点，方格 {self.square_mm:g} mm"


@dataclass
class BoardObservation:
    """一帧里对标定板的观测，附带质量指标。"""

    found: bool = False
    corners: np.ndarray | None = None       # (N,1,2) float32
    homography: np.ndarray | None = None    # 板平面(0..1)->图像 的 3x3 单应
    sharpness: float = 0.0                  # 棋盘区域的 Laplacian 方差
    area_ratio: float = 0.0                 # 棋盘凸包面积 / 图像面积
    margin_px: float = 0.0                  # 角点到图像边界的最小距离
    brightness: float = 0.0
    note: str = ""                          # 未通过时的原因

    def quad(self) -> np.ndarray | None:
        """把板平面四角映射到图像，得到棋盘的四边形轮廓。"""
        if self.homography is None:
            return None
        unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(unit, self.homography).reshape(-1, 2)

    def pose_points(self) -> np.ndarray | None:
        """把归一化采样点映射到图像 —— 位姿多样性/覆盖率的统一表示。"""
        if self.homography is None:
            return None
        unit = BoardSpec().unit_square_points().reshape(-1, 1, 2)
        return cv2.perspectiveTransform(unit, self.homography).reshape(-1, 2)


def _sharpness(gray: np.ndarray, corners: np.ndarray | None) -> float:
    """棋盘区域的清晰度：Laplacian 方差。越小越模糊（运动模糊/失焦）。"""
    if corners is None or len(corners) == 0:
        return 0.0
    x, y, w, h = cv2.boundingRect(corners.reshape(-1, 1, 2).astype(np.int32))
    pad = 4
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
    roi = gray[y0:y1, x0:x1]
    if roi.size < 100:
        return 0.0
    return float(cv2.Laplacian(roi, cv2.CV_64F).var())


def detect_board(
    gray: np.ndarray,
    spec: BoardSpec,
    use_sb: bool = True,
    normalize: bool = True,
) -> BoardObservation:
    """检测标定板并给出质量指标。

    优先用 findChessboardCornersSB（OpenCV 自带亚像素与鲁棒化搜索，
    精度明显好于老接口 + cornerSubPix），失败再回退到经典接口 + cornerSubPix。
    """
    obs = BoardObservation()
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    corners = None
    flags = 0
    if use_sb and hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        if normalize:
            flags |= cv2.CALIB_CB_NORMALIZE_IMAGE
        ok, c = cv2.findChessboardCornersSB(gray, spec.pattern_size, flags=flags)
        if ok:
            corners = c.astype(np.float32)

    if corners is None:
        flags2 = cv2.CALIB_CB_ADAPTIVE_THRESH
        if normalize:
            flags2 |= cv2.CALIB_CB_NORMALIZE_IMAGE
        flags2 |= cv2.CALIB_CB_FAST_CHECK
        ok, c = cv2.findChessboardCorners(gray, spec.pattern_size, flags=flags2)
        if not ok:
            obs.note = "未检测到棋盘"
            return obs
        # 经典接口需要自己细化到亚像素
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        corners = cv2.cornerSubPix(gray, c, (5, 5), (-1, -1), criteria)

    obs.found = True
    obs.corners = corners.reshape(-1, 1, 2).astype(np.float32)

    # 板平面(归一化 0..1) -> 图像 的单应，作为"位姿"的统一表示
    obj_xy = spec.object_points()[:, :2].astype(np.float32)
    obj_xy = obj_xy / np.array([spec.cols - 1, spec.rows - 1], dtype=np.float32)
    H, _ = cv2.findHomography(obj_xy, obs.corners.reshape(-1, 2), 0)
    obs.homography = H

    pts = obs.corners.reshape(-1, 2)
    hull = cv2.convexHull(pts.astype(np.float32))
    obs.area_ratio = float(cv2.contourArea(hull) / float(w * h))
    obs.margin_px = float(min(pts[:, 0].min(), pts[:, 1].min(),
                              w - 1 - pts[:, 0].max(), h - 1 - pts[:, 1].max()))
    obs.sharpness = _sharpness(gray, obs.corners)
    if obs.corners is not None:
        xs = np.clip(pts[:, 0].astype(np.int32), 0, w - 1)
        ys = np.clip(pts[:, 1].astype(np.int32), 0, h - 1)
        obs.brightness = float(gray[ys, xs].mean())
    return obs


def pose_distance(h1: np.ndarray | None, h2: np.ndarray | None, image_diag: float) -> float:
    """两个位姿的差异（归一化到图像对角线）。

    做法：把板平面上的一组采样点分别用两个单应映射到图像，
    取对应点的均方根距离。它同时包含了平移、缩放、旋转、倾斜，
    比单独比某一个量更能代表"这两个视角差得够不够远"。
    """
    if h1 is None or h2 is None:
        return 0.0
    unit = BoardSpec().unit_square_points().reshape(-1, 1, 2)
    p1 = cv2.perspectiveTransform(unit, h1).reshape(-1, 2)
    p2 = cv2.perspectiveTransform(unit, h2).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum((p1 - p2) ** 2, axis=1))) / max(image_diag, 1.0))
