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

    ⚠️ **失败时不抛异常，原样返回输入**（`radius < 3.0`、采样起点越界、
    有效射线不足）。要区分"真的拟合出来了"与"原样退回"，用
    :func:`refine_circle_subpixel_checked`。
    """
    x, y, r, _ = refine_circle_subpixel_checked(
        gray, cx, cy, radius, n_rays=n_rays, search=search, step=step,
        min_gradient=min_gradient)
    return x, y, r


def refine_circle_subpixel_checked(
    gray: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    n_rays: int = 72,
    search: float = 2.5,
    step: float = 0.25,
    min_gradient: float = 15.0,
) -> tuple[float, float, float, bool]:
    """:func:`refine_circle_subpixel` 的实现，末位多一个 **是否真的细化成功**。

    为什么要这个布尔量：原函数的约定是"失败就把输入原样返回"（`docs/API.md`）。
    调用方若想**只取圆心、不要它估的半径**（典型用途：双目要跨视图对应的
    *轮廓圆心*，而半径另有出处），就必须能分辨"真的拟合出了一个圆"与
    "压根没找到足够梯度"。靠比对"输出是否与输入逐位相同"来分辨，是把语义
    压在浮点相等上；这里直接给出判据。
    """
    if radius < 3.0 or gray.size == 0:
        return cx, cy, radius, False
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    n_samples = int(2 * search / step) + 1
    offsets = radius - search + step * np.arange(n_samples)
    if offsets[0] <= 1.0:
        return cx, cy, radius, False
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
        return cx, cy, radius, False
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
    fx, fy, fr = fit_circle_robust(edge)
    return fx, fy, fr, True


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
# 极线几何（parallel / angled 两种模式共用）
# --------------------------------------------------------------------------
#
# 为什么这些公式放在这里而不是 `stereo/angled_geometry.py`：
# `calib/` 在依赖方向上**低于** `stereo/`（见 AGENTS.md §5），
# 所以 `calib/params.py` 不能反向 import `stereo.angled_geometry`。
# angled 模式的极线判据（F 矩阵 + 点到极线距离）既是 stereo 的配对依据，
# 也是 calib 复检 90° 装置时唯一有意义的指标，因此共享公式必须下沉到本模块，
# `stereo/angled_geometry.py` 从这里导入并 re-export，两边不会各写一份。


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """向量 `v` 的反对称矩阵 `[v]_x`（满足 `[v]_x @ x == cross(v, x)`）。"""
    x, y, z = (float(v[0]), float(v[1]), float(v[2]))
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )


def fundamental_matrix_from_pose(
    K1: np.ndarray,
    K2: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    """由位姿构造基础矩阵：`F = K2^-T @ [T]_x @ R @ K1^-1`。

    约定 `X_right = R @ X_left + T`（OpenCV 的 `stereoCalibrate` 输出）。
    输入像素必须是**去畸变后、仍在各自 K 像素系**的坐标
    （`cv2.undistortPoints(..., P=K)`）；畸变是非线性的，无法塞进 3x3 的 F。
    不做任何有效性判断（rank-2、T 是否为零）—— 由调用方决定如何处置。
    """
    K1_arr = np.asarray(K1, dtype=np.float64).reshape(3, 3)
    K2_arr = np.asarray(K2, dtype=np.float64).reshape(3, 3)
    R_arr = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T_arr = np.asarray(T, dtype=np.float64).reshape(3)
    F = np.linalg.inv(K2_arr).T @ skew_symmetric(T_arr) @ R_arr @ np.linalg.inv(K1_arr)
    return np.asarray(F, dtype=np.float64)


def epipolar_line(F: np.ndarray, pixel_left: np.ndarray) -> np.ndarray:
    """左图点对应的右图极线 `(a, b, c)`，法向量已归一化（`a²+b²==1`）。

    退化（法向量长度为 0，例如极线在无穷远）时返回未归一化的系数，
    由 :func:`point_line_distance` 决定是否可用。
    """
    F_arr = np.asarray(F, dtype=np.float64)
    if F_arr.shape != (3, 3):
        raise ValueError(f"F must have shape (3, 3), got {F_arr.shape}")
    p = np.asarray(pixel_left, dtype=np.float64).reshape(-1)
    if p.size != 2:
        raise ValueError(f"pixel_left must contain 2 values, got shape {p.shape}")
    if not np.all(np.isfinite(p)):
        raise ValueError("pixel_left values must be finite")

    line = F_arr @ np.array([p[0], p[1], 1.0], dtype=np.float64)
    normal_length = float(np.hypot(line[0], line[1]))
    if np.isfinite(normal_length) and normal_length > 0.0:
        normalized = line / normal_length
        if np.all(np.isfinite(normalized)):
            line = normalized
    return np.asarray(line, dtype=np.float64).reshape(3)


def point_line_distance(line: np.ndarray, point: np.ndarray) -> float:
    """点到直线 `(a, b, c)` 的垂直距离（像素）：`|a x + b y + c| / hypot(a, b)`。

    直线退化（`a == b == 0`）时抛 `ValueError`，不返回 0 ——
    0 会被误读成"完美对齐"，而实际上这条极线根本没有定义。
    """
    line_arr = np.asarray(line, dtype=np.float64).reshape(-1)
    if line_arr.size != 3:
        raise ValueError(f"line must contain 3 values, got shape {line_arr.shape}")
    p = np.asarray(point, dtype=np.float64).reshape(-1)
    if p.size != 2:
        raise ValueError(f"point must contain 2 values, got shape {p.shape}")
    if not (np.all(np.isfinite(line_arr)) and np.all(np.isfinite(p))):
        raise ValueError("line and point values must be finite")

    a, b, c = (float(line_arr[0]), float(line_arr[1]), float(line_arr[2]))
    normal_length = float(np.hypot(a, b))
    if normal_length <= 0.0:
        raise ValueError("cannot measure distance to a degenerate line (a=b=0)")
    return float(abs(a * p[0] + b * p[1] + c) / normal_length)


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
        ball_diameter_m: float = -1.0,
        distortion: Sequence[float] | None = None,
    ) -> None:
        # 球径**没有默认值**，必须由调用方显式给出。原因：它曾有三份默认值
        # （这里 0.04、`AppConfig` 0.04、GUI 40 mm），而单目测深 `Z=f·D/(2r)`
        # 对它 1:1 敏感 —— 三个数不一致时，同一条视频换个入口跑就换一个答案。
        # 现在唯一口径是**有效直径**（名义 25 mm 的球 → 21.6 mm，检测器半径
        # 系统性偏小 13.6%），依据见交接文档 §35.2。传 -1 等于没传，当场报错，
        # 而不是悄悄用一个谁都记不住的旧默认值。
        if ball_diameter_m is None or float(ball_diameter_m) <= 0:
            raise ValueError(
                "CameraModel 必须显式给出 ball_diameter_m（有效直径：名义 25 mm 的球"
                " 约 21.6 mm，见交接文档 §35.2）—— 这个参数没有安全的默认值")
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
        # 畸变系数 (k1, k2, p1, p2, k3)，缺省为空 = 没有畸变模型。
        # **无畸变时下面所有去畸变路径都直接短路**，所以不标定的用户零开销。
        d = np.asarray([] if distortion is None else distortion, np.float64).ravel()
        self.distortion: tuple[float, ...] = (
            tuple(float(x) for x in d[:5]) + (0.0,) * max(0, 5 - d[:5].size)
            if d.size else ())
        self.calibrated = bool(fx and fx > 0)

    # -- 畸变 -----------------------------------------------------------------
    @property
    def has_distortion(self) -> bool:
        """有没有可用的畸变模型。全 0 等同于"没有"。"""
        return bool(self.distortion) and any(abs(x) > 1e-12 for x in self.distortion)

    @property
    def _K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    @property
    def _D(self) -> np.ndarray:
        return np.asarray(self.distortion or (0.0,) * 5, np.float64)

    def undistort_points(self, uv) -> np.ndarray:
        """畸变像素坐标 -> 无畸变像素坐标。**没有畸变模型时原样返回**。

        为什么必须做在这里而不是"每帧 undistort 整张图"：整帧去畸变在 1080p
        下每帧要 5–10 ms，而检测是在降分辨率上跑的（``--detect-width``），
        在错误的尺度上去畸变还会把两件事搅在一起。点级校正只在你真正要
        三维坐标的那几步付代价，且与检测分辨率无关。
        """
        pts = np.asarray(uv, np.float64).reshape(-1, 2)
        if not self.has_distortion:
            return pts
        out = cv2.undistortPoints(pts.reshape(-1, 1, 2), self._K, self._D,
                                  P=self._K)
        return out.reshape(-1, 2)

    def distort_points(self, uv) -> np.ndarray:
        """无畸变像素坐标 -> 畸变像素坐标（画图用；没有畸变时原样返回）。"""
        pts = np.asarray(uv, np.float64).reshape(-1, 2)
        if not self.has_distortion:
            return pts
        x = (pts[:, 0] - self.cx) / self.fx
        y = (pts[:, 1] - self.cy) / self.fy
        obj = np.stack([x, y, np.ones_like(x)], axis=1).reshape(-1, 1, 3)
        out, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), self._K, self._D)
        return out.reshape(-1, 2)

    def undistort_radius(self, u: float, v: float, r_px: float) -> float:
        """畸变图上的像素半径 -> 无畸变半径（8 个方向取中位）。

        为什么取中位而不是直接量水平半径：球在画面边缘时畸变会把它压成椭圆，
        长轴与短轴差几个百分点，取任一方向都会引入方向性偏差。
        """
        if not self.has_distortion or r_px <= 0:
            return float(r_px)
        ang = np.arange(8) * (math.pi / 4.0)
        ring = np.stack([u + r_px * np.cos(ang),
                         v + r_px * np.sin(ang)], axis=1)
        centre = self.undistort_points(np.array([[u, v]]))[0]
        pts = self.undistort_points(ring)
        return float(np.median(np.linalg.norm(pts - centre, axis=1)))

    # -- 单目测深：球在图像里的像素半径与距离成反比 ------------------------
    def depth_from_radius(self, radius_px: float,
                          u: float | None = None,
                          v: float | None = None) -> float:
        """由像素半径反推球心到相机的距离（米）。半径 -> 0 时返回 inf。

        给了 ``u, v`` 且有畸变模型时，先把半径校正到无畸变尺度再算 ——
        否则同一个球在画面边缘会比在中心"看起来更大"，深度随之偏近。
        """
        if radius_px <= 1e-6:
            return float("inf")
        r = radius_px if (u is None or v is None) else self.undistort_radius(u, v, radius_px)
        f = 0.5 * (self.fx + self.fy)
        return f * (self.ball_diameter_m / 2.0) / r

    def backproject(self, u: float, v: float, z: float) -> np.ndarray:
        """像素 + 深度 -> 相机坐标系下的三维点 (X, Y, Z)，单位米，Z 向前。

        有畸变模型时**先把像素校正到无畸变坐标**——反投影公式只对无畸变的
        针孔成像成立，直接拿畸变坐标代入会在画面边缘偏出好几厘米。
        """
        if self.has_distortion:
            u, v = (float(x) for x in self.undistort_points(np.array([[u, v]]))[0])
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return np.array([x, y, z], dtype=np.float64)

    def project(self, point3d: np.ndarray) -> tuple[float, float]:
        """三维点 -> 像素坐标（有畸变模型时把结果投回畸变图，画图才对得上）。"""
        x, y, z = (float(v) for v in point3d[:3])
        if abs(z) < 1e-9:
            z = 1e-9
        u = self.cx + self.fx * x / z
        v = self.cy + self.fy * y / z
        if self.has_distortion:
            u, v = (float(a) for a in self.distort_points(np.array([[u, v]]))[0])
        return u, v


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
