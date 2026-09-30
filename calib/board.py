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
    precise: bool = True                    # 角点是否来自全分辨率精确检测
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


def _finish(obs: BoardObservation, corners: np.ndarray, gray: np.ndarray,
            spec: BoardSpec, precise: bool) -> BoardObservation:
    """由角点补齐单应、面积、边距、清晰度、亮度等指标。"""
    obs.found = True
    obs.precise = precise
    obs.corners = np.asarray(corners, np.float32).reshape(-1, 1, 2)
    h, w = gray.shape[:2]

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
    xs = np.clip(pts[:, 0].astype(np.int32), 0, w - 1)
    ys = np.clip(pts[:, 1].astype(np.int32), 0, h - 1)
    obs.brightness = float(gray[ys, xs].mean())
    return obs


def detect_board(
    gray: np.ndarray,
    spec: BoardSpec,
    use_sb: bool = True,
    normalize: bool = True,
    work_width: int | None = None,
    exhaustive: bool = True,
) -> BoardObservation:
    """检测标定板并给出质量指标。

    性能提示（实测 1920x1080）：
        SB + EXHAUSTIVE + ACCURACY   约 300 ms   <- 精确但慢，只能用在"要收的那一帧"
        SB 不开 EXHAUSTIVE           约 93 ms
        缩到 640 宽再检测             约 12 ms    <- 适合每帧跑的粗筛
    EXHAUSTIVE 在**找不到板**时会穷举搜索，所以绝不能每帧开。
    要每帧跑就用 work_width 降分辨率 + exhaustive=False，拿到粗位姿做判断，
    真正收帧时再用 detect_board_in_roi() 在棋盘附近做精确检测
    （裁剪不损失分辨率，只是缩小搜索范围）。
    """
    obs = BoardObservation()
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    full = gray
    scale = 1.0
    if work_width and gray.shape[1] > work_width:
        scale = work_width / float(gray.shape[1])
        gray = cv2.resize(gray, (work_width, max(1, int(round(gray.shape[0] * scale)))),
                          interpolation=cv2.INTER_AREA)

    corners = None
    if use_sb and hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE if normalize else 0
        if exhaustive:
            flags |= cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        ok, c = cv2.findChessboardCornersSB(gray, spec.pattern_size, flags=flags)
        if ok:
            corners = np.asarray(c, np.float32)

    if corners is None and exhaustive:
        flags2 = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK
        if normalize:
            flags2 |= cv2.CALIB_CB_NORMALIZE_IMAGE
        ok, c = cv2.findChessboardCorners(gray, spec.pattern_size, flags=flags2)
        if not ok:
            obs.note = "未检测到棋盘"
            return obs
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        corners = cv2.cornerSubPix(gray, c, (5, 5), (-1, -1), criteria)

    if corners is None:
        obs.note = "未检测到棋盘"
        return obs
    if scale != 1.0:
        corners = corners / scale          # 换算回全分辨率坐标
    # 指标要在全分辨率图上算（清晰度、亮度才可比）
    return _finish(obs, corners, full, spec, precise=(scale == 1.0 and exhaustive))


def detect_board_in_roi(
    gray: np.ndarray,
    spec: BoardSpec,
    hint: np.ndarray,
    margin_ratio: float = 0.5,
    margin_px: int = 28,
) -> BoardObservation | None:
    """在"粗检到的棋盘"附近裁剪出一小块，做全分辨率精确检测。

    为什么要这一步：每帧都做全图精确检测要 300 ms，撑不住实时；
    但直接拿降分辨率粗检的角点去标定，精度又差 3 倍（640 宽下 0.5 px
    ≈ 全分辨率 1.5 px）。裁剪既不损失分辨率，又把搜索范围缩小到棋盘附近，
    实测一次约 40 ms —— 只在"真要收这一帧"时做，完全感觉不到。
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    pts = np.asarray(hint, np.float32).reshape(-1, 2)
    if len(pts) < 4:
        return None
    x0, y0, w, h = cv2.boundingRect(pts.astype(np.int32))
    pad_x = int(max(margin_px, w * margin_ratio))
    pad_y = int(max(margin_px, h * margin_ratio))
    gx0 = max(0, x0 - pad_x); gy0 = max(0, y0 - pad_y)
    gx1 = min(gray.shape[1], x0 + w + pad_x); gy1 = min(gray.shape[0], y0 + h + pad_y)
    crop = gray[gy0:gy1, gx0:gx1]
    if crop.size < 4000:
        return None
    obs = detect_board(crop, spec, exhaustive=True)
    if not obs.found or obs.corners is None:
        return None
    obs.corners = (obs.corners.reshape(-1, 2) + np.array([gx0, gy0], np.float32)).reshape(-1, 1, 2)
    # 指标换成在全分辨率整图上算
    return _finish(BoardObservation(), obs.corners, gray, spec, precise=True)


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
