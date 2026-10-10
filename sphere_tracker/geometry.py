"""几何工具：亚像素圆拟合、相机模型、单目测深。"""

from __future__ import annotations

import math

import cv2
import numpy as np


# --------------------------------------------------------------------------
# 圆的拟合
# --------------------------------------------------------------------------
def fit_circle_taubin(points: np.ndarray) -> tuple[float, float, float]:
    """Taubin 代数圆拟合，返回 (cx, cy, r)。

    比 minEnclosingCircle 精度高得多（亚像素级），是做轨迹预测的关键：
    圆心抖动 1 px，速度就会被放大成很大的噪声。
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 3:
        raise ValueError("圆拟合至少需要 3 个点")
    x, y = pts[:, 0], pts[:, 1]
    x_m, y_m = x.mean(), y.mean()
    u, v = x - x_m, y - y_m
    Suu = float((u * u).sum())
    Svv = float((v * v).sum())
    Suv = float((u * v).sum())
    Suuu = float((u * u * u).sum())
    Svvv = float((v * v * v).sum())
    Suvv = float((u * v * v).sum())
    Svuu = float((v * u * u).sum())
    A = np.array([[Suu, Suv], [Suv, Svv]], dtype=np.float64)
    b = np.array([(Suuu + Suvv) / 2.0, (Svvv + Svuu) / 2.0], dtype=np.float64)
    try:
        uc, vc = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        uc = vc = 0.0
    r = math.sqrt(max(uc * uc + vc * vc + (Suu + Svv) / pts.shape[0], 0.0))
    return float(uc + x_m), float(vc + y_m), float(r)


def fit_circle_robust(
    points: np.ndarray,
    iterations: int = 3,
    trim_sigma: float = 2.0,
) -> tuple[float, float, float]:
    """带离群点剔除的圆拟合。

    轮廓点里常混有粘连的噪声点，直接用最小二乘会被拉偏；
    这里反复拟合并把残差超过 trim_sigma 倍标准差的点丢掉。
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 5:
        return fit_circle_taubin(pts)
    work = pts
    cx = cy = r = 0.0
    for _ in range(iterations):
        try:
            cx, cy, r = fit_circle_taubin(work)
        except ValueError:
            break
        d = np.abs(np.hypot(work[:, 0] - cx, work[:, 1] - cy) - r)
        sigma = float(d.std())
        if sigma < 1e-6:
            break
        keep = d <= trim_sigma * sigma + 1e-6
        if keep.sum() < 5 or keep.all():
            break
        work = work[keep]
    return float(cx), float(cy), float(r)


def _bilinear(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """双线性采样（xs/ys 是浮点坐标）。"""
    h, w = image.shape[:2]
    x = np.clip(xs, 0.0, w - 1.001)
    y = np.clip(ys, 0.0, h - 1.001)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = x - x0
    fy = y - y0
    return (
        image[y0, x0] * (1 - fx) * (1 - fy)
        + image[y0, x1] * fx * (1 - fy)
        + image[y1, x0] * (1 - fx) * fy
        + image[y1, x1] * fx * fy
    )


def refine_circle_subpixel(
    gray: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    n_rays: int = 72,
    search: float = 2.5,
    step: float = 0.25,
    min_gradient: float = 15.0,
) -> tuple[float, float, float]:
    """亚像素级圆弧细化：沿径向射线找梯度峰值，再重新拟合圆。

    这是"单目测深"最关键的一步。深度 Z = f*R/r 完全由像素半径 r 决定：
      - 阈值分割得到的边缘，误差受二值化阈值影响，典型 0.3-0.5 px；
      - 梯度峰值位置（配合抛物线插值）能到 0.05-0.1 px。
    半径精度提高 4 倍，深度精度也就提高 4 倍。

    做法：从粗略圆心向 72 个方向发射射线，在半径附近采样梯度幅值，
    取峰值并用抛物线插值求亚像素位置，得到 72 个精确边缘点，最后做稳健圆拟合。
    """
    if radius < 3.0 or gray.size == 0:
        return cx, cy, radius
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    n_samples = int(2 * search / step) + 1
    offsets = radius - search + step * np.arange(n_samples)
    if offsets[0] <= 1.0:
        return cx, cy, radius
    angles = np.linspace(0.0, 2.0 * math.pi, n_rays, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    # 一次性把所有射线、所有采样点都算出来（向量化，比逐条射线快一个量级）
    xs = cx + np.outer(cos_a, offsets)
    ys = cy + np.outer(sin_a, offsets)
    prof = _bilinear(mag, xs, ys)                      # (n_rays, n_samples)
    peaks = np.argmax(prof, axis=1)
    rows = np.arange(n_rays)
    valid = (peaks > 0) & (peaks < n_samples - 1) & (prof[rows, peaks] >= min_gradient)
    if int(valid.sum()) < max(8, n_rays // 3):
        return cx, cy, radius
    rows_v = rows[valid]
    k_v = peaks[valid]
    y0 = prof[rows_v, k_v - 1]
    y1 = prof[rows_v, k_v]
    y2 = prof[rows_v, k_v + 1]
    denom = y0 - 2.0 * y1 + y2
    delta = np.where(np.abs(denom) > 1e-6, 0.5 * (y0 - y2) / np.where(np.abs(denom) > 1e-6, denom, 1.0), 0.0)
    delta = np.clip(delta, -1.0, 1.0)
    s = offsets[k_v] + delta * step
    edge = np.stack([cx + s * cos_a[valid], cy + s * sin_a[valid]], axis=1)
    return fit_circle_robust(edge)


def contour_circularity(contour: np.ndarray) -> float:
    area = cv2.contourArea(contour)
    peri = cv2.arcLength(contour, True)
    if peri <= 1e-6 or area <= 0:
        return 0.0
    return float(min(4.0 * math.pi * area / (peri * peri), 1.0))


def radial_consistency(points: np.ndarray, cx: float, cy: float, radius: float) -> float:
    """轮廓点到圆心的距离有多"一致"（0-1，1 = 完美圆）。

    这是区分"圆"和"方块/胶囊/人形"最有效的一个指标：
    圆上所有点到圆心的距离几乎相同（std 很小），方块的顶点比边中点远 41%，
    所以方块的径向一致性明显偏低，而圆度、填充率都可能骗过去。
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 3 or radius <= 1e-6:
        return 0.0
    d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    return float(max(0.0, 1.0 - float(d.std()) / radius))


def circle_radial_gradient_stats(
    gray: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    samples: int = 96,
    min_magnitude: float = 40.0,
    min_alignment: float = 0.65,
) -> tuple[float, float]:
    """返回 (沿半径方向梯度的比例, 梯度符号一致性)。

    符号一致性是识别"真实物体边界"的关键：球体边缘要么全是
    "由暗到亮"、要么全是"由亮到暗"，符号统一；
    而霍夫圆在纹理、直线、角点上凑出来的假圆，梯度方向正负混杂，
    符号一致性会很低。这一项能把霍夫的乱检压下去一大截。
    """
    if radius <= 1.5 or gray.size == 0:
        return 0.0, 0.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    h, w = gray.shape[:2]
    angles = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    hit = 0
    valid = 0
    signs: list[int] = []
    for a in angles:
        x = int(round(cx + radius * math.cos(a)))
        y = int(round(cy + radius * math.sin(a)))
        if x < 1 or y < 1 or x >= w - 1 or y >= h - 1:
            continue
        valid += 1
        dx = float(gx[y, x])
        dy = float(gy[y, x])
        mag = math.hypot(dx, dy)
        if mag < min_magnitude:
            continue
        radial = dx * math.cos(a) + dy * math.sin(a)
        if abs(radial) / (mag + 1e-6) >= min_alignment:
            hit += 1
            signs.append(1 if radial > 0 else -1)
    if valid == 0:
        return 0.0, 0.0
    support = hit / valid
    if not signs:
        return support, 0.0
    arr = np.asarray(signs, dtype=np.float64)
    consistency = float(abs(arr.mean()))
    return support, consistency


def circle_gradient_support(
    gray: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    samples: int = 96,
    min_magnitude: float = 40.0,
    min_alignment: float = 0.65,
) -> float:
    """圆上梯度是否"沿半径方向"（0-1）。

    真实球体的边缘梯度一定指向圆心；而霍夫圆经常在纹理/直线上凑出一个圆，
    那里的梯度方向和半径方向不一致。这一项能显著减少霍夫的乱检。
    """
    support, _ = circle_radial_gradient_stats(
        gray, cx, cy, radius, samples, min_magnitude, min_alignment
    )
    return support


def circle_edge_support(edges: np.ndarray, cx: float, cy: float, radius: float, samples: int = 72) -> float:
    """圆上有多少比例的点真的落在边缘图上（0-1）。用于给霍夫圆打分。"""
    if radius <= 1.0 or edges.size == 0:
        return 0.0
    h, w = edges.shape[:2]
    angles = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    xs = np.rint(cx + radius * np.cos(angles)).astype(np.int32)
    ys = np.rint(cy + radius * np.sin(angles)).astype(np.int32)
    hit = 0
    for x, y in zip(xs, ys):
        if x < 1 or y < 1 or x >= w - 1 or y >= h - 1:
            continue
        if edges[y - 1 : y + 2, x - 1 : x + 2].any():
            hit += 1
    return hit / samples


# --------------------------------------------------------------------------
# 相机模型
# --------------------------------------------------------------------------
class CameraModel:
    """针孔相机模型。

    标定与否都能用：
      - 有 fx/fy/cx/cy：直接使用；
      - 没有：用水平视场角 hfov 与图像宽度估算 fx = (w/2) / tan(hfov/2)。
        这个估算对"单目测深"的绝对尺度有影响，但轨迹形状不受影响。
    """

    def __init__(
        self,
        width: int,
        height: int,
        fx: float = 0.0,
        fy: float = 0.0,
        cx: float = -1.0,
        cy: float = -1.0,
        hfov_deg: float = 60.0,
        ball_diameter_m: float = 0.04,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.ball_diameter_m = float(ball_diameter_m)
        if fx and fx > 0:
            self.fx = float(fx)
        else:
            self.fx = (self.width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        self.fy = float(fy) if fy and fy > 0 else self.fx
        self.cx = float(cx) if cx is not None and cx >= 0 else self.width / 2.0
        self.cy = float(cy) if cy is not None and cy >= 0 else self.height / 2.0
        self.calibrated = bool(fx and fx > 0)

    # -- 单目测深：球在图像里的像素半径与距离成反比 ------------------------
    def depth_from_radius(self, radius_px: float) -> float:
        """由像素半径反推球心到相机的距离（米）。半径 -> 0 时返回 inf。"""
        if radius_px <= 1e-6:
            return float("inf")
        f = 0.5 * (self.fx + self.fy)
        return f * (self.ball_diameter_m / 2.0) / radius_px

    def backproject(self, u: float, v: float, z: float) -> np.ndarray:
        """像素 + 深度 -> 相机坐标系下的三维点 (X, Y, Z)，单位米，Z 向前。"""
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return np.array([x, y, z], dtype=np.float64)

    def project(self, point3d: np.ndarray) -> tuple[float, float]:
        """三维点 -> 像素坐标。"""
        x, y, z = (float(v) for v in point3d[:3])
        if abs(z) < 1e-9:
            z = 1e-9
        return self.cx + self.fx * x / z, self.cy + self.fy * y / z


def estimate_focal_from_known_ball(
    measured_radius_px: float, distance_m: float, ball_diameter_m: float
) -> float:
    """已知球的实际直径和它到相机的距离，用一次测量反算焦距。

    这是最省事的"半标定"：把球放在量好的距离上，量出像素半径即可。
    返回像素焦距 f。
    """
    if measured_radius_px <= 1e-6:
        raise ValueError("像素半径必须大于 0")
    return measured_radius_px * distance_m / (ball_diameter_m / 2.0)
