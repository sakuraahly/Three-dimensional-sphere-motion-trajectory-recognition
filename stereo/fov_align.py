"""自动 FOV 对齐：把两路不同出厂视场的画面归一到同一虚拟视场。

硬件背景
--------
当前两台相机的出厂视场不同：左相机约比右相机放大 1.5 倍（左视场更窄）。
直接拿两路画面做极线配对 / 三角测量，像素尺度不一致会污染视差与深度。

模块职责（已接入标定 / 录制 / 复检）
------------------------------------
1. SIFT 优先、ORB 兜底；Lowe ratio test + 双向交叉校验；
2. RANSAC 估计 homography（失败退而求仿射），由仿射拟合得到左->右尺度 ``s``；
3. ``s < 1``：左相机更窄 -> 以左帧为参考，把左帧四角通过 homography 映射到
   右帧，取轴对齐包围盒作为裁剪区，再把裁剪区缩放到左帧尺寸；
   ``s > 1``：右相机更窄 -> 镜像处理；**参考帧必须完整落在更广的那一路里**
   （四角越界 > 16 px 即共同视场不足，报错而不是裁掉越界部分）；
4. 尺度接近 1 且平移 / 重叠都表明视场基本一致 -> ``crop_side="none"``，不裁不缩；
5. 匹配 / 内点 / 重叠 / 共同视场不足一律抛 ``ValueError``，
   **绝不静默退化成“不对齐”或“部分重叠也对齐”**。

坐标约定
--------
- ``crop_rect = (x, y, w, h)``，坐标在被裁剪那一路的像素系里；
- ``target_size = (width, height)``，两路共同输出尺寸；取参考帧与裁剪区的较小尺寸，
  **不把广角裁剪区上采样**（当前硬件约 1685x979）；
- ``scale`` 是估计出的 left->right 像素尺度（右图像素 / 左图像素）：
  1.0 表示两路视场基本一致，当前硬件约为 1/1.5 ≈ 0.667。
- **参考帧永远不缩放**；只裁剪较广的一路，再用 ``INTER_LINEAR`` 缩放到参考帧尺寸。

本模块是纯函数 / 纯数据模块：不读配置、不碰相机、不写文件。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import cv2
import numpy as np

__all__ = ["FovAlignment", "estimate_fov_alignment"]

# 匹配与估计参数集中在这里，换机器时只调一处。
_SIFT_MAX_FEATURES = 5000
_ORB_MAX_FEATURES = 5000
_SIFT_RATIO = 0.75
_ORB_RATIO = 0.80
_RANSAC_PX = 3.0
_RANSAC_ITERS = 5000
_RANSAC_CONFIDENCE = 0.999

# “几乎同一视场”的判据：尺度接近 1 + 参考帧中心偏移小 + 参考帧可见比例高。
_NEAR_UNITY = 0.05          # |s - 1| <= 5%
_NEAR_CENTER_FRAC = 0.10    # 中心偏移 <= 短边的 10%
_NEAR_OVERLAP = 0.90        # 参考帧可见面积 >= 90%

# 广角帧里必须裁出有意义的一块，否则说明估计不可信。
_MIN_CROP_AREA_RATIO = 0.20

# 参考帧四角映射到广角帧后允许的越界量（像素）：只容忍浮点 / 取整误差。
# 超过它就是“参考帧没有完整落在广角帧内”，共同视场不足，必须报错。
_MAPPED_OUTSIDE_TOL_PX = 16.0


@dataclass(frozen=True)
class FovAlignment:
    """一次 FOV 估计的结果，以及把它应用到帧对上的方法。

    字段
    ----
    reference_side : ``"left"`` / ``"right"``，作为尺度参考（较窄、较放大）的一路。
    crop_side      : 被裁剪的一路（较广的一路），或 ``"none"`` 表示无需对齐。
    crop_rect      : ``(x, y, w, h)``，在 ``crop_side`` 那一路的像素系里；
                     ``crop_side == "none"`` 时记参考帧全幅 ``(0, 0, w, h)``，仅作占位。
    target_size    : ``(width, height)``，两路共同输出尺寸；不放大裁剪区，通常小于参考帧。
    scale          : 估计出的 left->right 像素尺度。
    method         : ``"sift"`` / ``"orb"``，实际使用的特征检测器。
    inliers        : 几何模型（homography / 仿射）的 RANSAC 内点数。
    quality        : ``[0, 1]``，由内点数量与内点比例综合得到。
    """

    reference_side: str
    crop_side: str
    crop_rect: tuple[int, int, int, int]
    target_size: tuple[int, int]
    scale: float
    method: str
    inliers: int
    quality: float

    def __post_init__(self) -> None:
        reference = str(self.reference_side)
        if reference not in ("left", "right"):
            raise ValueError(
                f"reference_side 必须是 'left' 或 'right'，收到 {self.reference_side!r}")
        crop = str(self.crop_side)
        if crop not in ("left", "right", "none"):
            raise ValueError(
                f"crop_side 必须是 'left'/'right'/'none'，收到 {self.crop_side!r}")
        if crop != "none" and crop == reference:
            raise ValueError("crop_side 不能与 reference_side 相同（参考帧不允许被裁剪）")

        try:
            rect = tuple(int(v) for v in self.crop_rect)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"crop_rect 必须是 4 个整数，收到 {self.crop_rect!r}") from exc
        if len(rect) != 4:
            raise ValueError(f"crop_rect 必须是 (x, y, w, h)，收到 {self.crop_rect!r}")
        x, y, w, h = rect
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            raise ValueError(
                f"crop_rect 必须满足 x>=0, y>=0, w>0, h>0，收到 {rect}")

        try:
            size = tuple(int(v) for v in self.target_size)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"target_size 必须是 2 个正整数，收到 {self.target_size!r}") from exc
        if len(size) != 2 or size[0] <= 0 or size[1] <= 0:
            raise ValueError(f"target_size 必须是 (w>0, h>0)，收到 {self.target_size!r}")

        try:
            scale = float(self.scale)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"scale 必须是正数，收到 {self.scale!r}") from exc
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"scale 必须是有限正数，收到 {self.scale!r}")

        try:
            inliers = int(self.inliers)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"inliers 必须是整数，收到 {self.inliers!r}") from exc
        if inliers < 0:
            raise ValueError(f"inliers 不能为负，收到 {inliers}")

        try:
            quality = float(self.quality)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"quality 必须是 0..1，收到 {self.quality!r}") from exc
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError(f"quality 必须在 [0, 1] 内，收到 {self.quality!r}")

        object.__setattr__(self, "reference_side", reference)
        object.__setattr__(self, "crop_side", crop)
        object.__setattr__(self, "crop_rect", rect)
        object.__setattr__(self, "target_size", size)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "method", str(self.method))
        object.__setattr__(self, "inliers", inliers)
        object.__setattr__(self, "quality", quality)

    @property
    def active(self) -> bool:
        """是否真的发生了裁剪 / 缩放。"""
        return self.crop_side != "none"

    def apply_side(self, frame: np.ndarray, side: str) -> np.ndarray:
        """归一化指定的单路帧。

        - 裁剪路：按 crop_rect 裁剪，再缩放到 target_size；
        - 参考路：不裁剪，但若 target_size 比参考原生尺寸小，必须**降采样**到
          target_size（不放大裁剪区的代价就是参考路也要降采样）；
        - crop_side == "none"：原样返回。
        """
        if side not in ("left", "right"):
            raise ValueError(f"side 必须是 'left' 或 'right'，收到 {side!r}")
        if self.crop_side == "none":
            return frame
        if not isinstance(frame, np.ndarray):
            raise ValueError(f"apply_side 需要 {side} 帧是 numpy.ndarray，"
                             f"收到 {type(frame).__name__}")

        target_w, target_h = self.target_size
        if side == self.reference_side:
            if frame.shape[:2] == (target_h, target_w):
                return frame
            # 参考帧只降采样，不裁剪；若传入比 target 还小的帧则拒绝，不放大。
            if frame.shape[0] < target_h or frame.shape[1] < target_w:
                raise ValueError(
                    f"参考帧尺寸 {(frame.shape[1], frame.shape[0])} 小于 "
                    f"target_size {(target_w, target_h)}，拒绝放大参考帧")
            return cv2.resize(frame, (target_w, target_h),
                              interpolation=cv2.INTER_AREA)

        if side != self.crop_side:
            return frame
        x, y, w, h = self.crop_rect
        crop_h, crop_w = frame.shape[:2]
        if x < 0 or y < 0 or x + w > crop_w or y + h > crop_h:
            raise ValueError(
                f"crop_rect {(x, y, w, h)} 超出 {self.crop_side} 帧范围 "
                f"{(crop_w, crop_h)}")

        region = frame[y:y + h, x:x + w]
        # 裁剪区可能略大于 target（AABB / 旋转）；缩到 target 是允许的，
        # 但不会把更小的裁剪区放大。
        if w < target_w or h < target_h:
            raise ValueError(
                f"crop_rect {(x, y, w, h)} 小于 target_size "
                f"{(target_w, target_h)}，拒绝放大裁剪区")
        aligned = cv2.resize(region, (target_w, target_h),
                             interpolation=cv2.INTER_AREA)
        if aligned.shape[0] != target_h or aligned.shape[1] != target_w:
            raise ValueError(
                f"缩放后尺寸 {aligned.shape[:2]} 不等于 target_size "
                f"{(target_h, target_w)}")
        return aligned

    def apply(self, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """按本次估计归一化帧对，两路都输出 target_size。"""
        if self.crop_side == "none":
            return left, right
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            raise ValueError("apply 需要 left/right 都是 numpy.ndarray")
        return (self.apply_side(left, "left"), self.apply_side(right, "right"))

    def to_dict(self) -> dict[str, Any]:
        """JSON 友好（list 而不是 tuple）的字典。"""
        return {
            "reference_side": self.reference_side,
            "crop_side": self.crop_side,
            "crop_rect": list(self.crop_rect),
            "target_size": list(self.target_size),
            "scale": self.scale,
            "method": self.method,
            "inliers": self.inliers,
            "quality": self.quality,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FovAlignment":
        """从 :meth:`to_dict` 的结果（含 JSON 往返后的 list）重建。"""
        if not isinstance(data, Mapping):
            raise ValueError(
                f"from_dict 需要 mapping，收到 {type(data).__name__}")
        try:
            payload = {field.name: data[field.name] for field in fields(cls)}
        except KeyError as exc:
            raise ValueError(f"FovAlignment 字典缺少字段 {exc.args[0]!r}") from exc
        return cls(**payload)


# --------------------------------------------------------------------------
# 图像 / 特征
# --------------------------------------------------------------------------
def _as_gray(image: np.ndarray, side: str) -> np.ndarray:
    """转灰度并统一到 uint8；输入非法时给出带 side 的明确错误。"""
    if not isinstance(image, np.ndarray):
        raise ValueError(f"{side} 帧必须是 numpy.ndarray，收到 {type(image).__name__}")
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 1:
        gray = image[:, :, 0]
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"{side} 帧形状不支持：{image.shape}")
    if gray.size == 0:
        raise ValueError(f"{side} 帧为空")

    if gray.dtype != np.uint8:
        data = gray.astype(np.float32)
        if not np.all(np.isfinite(data)):
            raise ValueError(f"{side} 帧含 NaN/Inf，无法提取特征")
        if float(data.max()) <= 1.0:
            data = data * 255.0
        gray = np.clip(data, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(gray)


def _detect(gray: np.ndarray, method: str):
    """SIFT / ORB 检测；特征太少时明确报错（由上层决定是否换检测器）。"""
    if method == "sift":
        detector = cv2.SIFT_create(nfeatures=_SIFT_MAX_FEATURES)
        norm = cv2.NORM_L2
    else:
        detector = cv2.ORB_create(nfeatures=_ORB_MAX_FEATURES)
        norm = cv2.NORM_HAMMING
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    if descriptors is None or len(keypoints) < 2:
        raise ValueError(f"{method.upper()} 只检出 {len(keypoints)} 个特征点，无法匹配")
    return keypoints, descriptors, norm


def _ratio_matches(matcher, desc_a: np.ndarray, desc_b: np.ndarray,
                   ratio: float) -> list:
    """Lowe ratio test（k=2）；匹配器异常时退回空列表，由上层判“不足”。"""
    if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
        return []
    try:
        pairs = matcher.knnMatch(desc_a, desc_b, k=2)
    except cv2.error:
        return []
    good = []
    for pair in pairs:
        if len(pair) >= 2 and pair[0].distance < ratio * pair[1].distance:
            good.append(pair[0])
    return good


def _match_features(gray_l: np.ndarray, gray_r: np.ndarray, method: str):
    """返回 (左特征数, 右特征数, 交叉校验后的匹配数, 左点, 右点)。"""
    kp_l, desc_l, norm = _detect(gray_l, method)
    kp_r, desc_r, _ = _detect(gray_r, method)

    if norm == cv2.NORM_L2:
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4),
                                        dict(checks=64))
        ratio = _SIFT_RATIO
    else:
        matcher = cv2.BFMatcher(norm)
        ratio = _ORB_RATIO

    forward = _ratio_matches(matcher, desc_l, desc_r, ratio)
    reverse = _ratio_matches(matcher, desc_r, desc_l, ratio)
    # 交叉校验：右->左也算“最佳”的那些，才保留。
    reverse_set = {(m.trainIdx, m.queryIdx) for m in reverse}
    mutual = [m for m in forward if (m.queryIdx, m.trainIdx) in reverse_set]

    pts_l = np.float32([kp_l[m.queryIdx].pt for m in mutual])
    pts_r = np.float32([kp_r[m.trainIdx].pt for m in mutual])
    return len(kp_l), len(kp_r), len(mutual), pts_l, pts_r


# --------------------------------------------------------------------------
# 几何模型
# --------------------------------------------------------------------------
def _find_homography(pts_l: np.ndarray, pts_r: np.ndarray):
    """RANSAC homography；返回 (H, 内点布尔掩码, 内点数)，失败给 (None, None, 0)。"""
    if len(pts_l) < 4:
        return None, None, 0
    try:
        homography, mask = cv2.findHomography(
            pts_l, pts_r, cv2.RANSAC, _RANSAC_PX,
            maxIters=_RANSAC_ITERS, confidence=_RANSAC_CONFIDENCE)
    except cv2.error:
        return None, None, 0
    if homography is None or mask is None or not np.all(np.isfinite(homography)):
        return None, None, 0
    if abs(float(homography[2, 2])) < 1e-12:
        return None, None, 0
    homography = homography / homography[2, 2]
    mask = mask.ravel().astype(bool)
    return homography, mask, int(mask.sum())


def _find_affine(pts_l: np.ndarray, pts_r: np.ndarray):
    """RANSAC 仿射回退；返回 (嵌入 3x3 的 H, 掩码, 内点数)。"""
    if len(pts_l) < 3:
        return None, None, 0
    try:
        affine, mask = cv2.estimateAffine2D(
            pts_l, pts_r, method=cv2.RANSAC,
            ransacReprojThreshold=_RANSAC_PX,
            maxIters=_RANSAC_ITERS, confidence=_RANSAC_CONFIDENCE)
    except cv2.error:
        return None, None, 0
    if affine is None or mask is None or not np.all(np.isfinite(affine)):
        return None, None, 0
    mask = mask.ravel().astype(bool)
    n_inliers = int(mask.sum())
    homography = np.eye(3, dtype=np.float64)
    homography[:2] = affine
    return homography, mask, n_inliers


def _similarity_scale(pts_l: np.ndarray, pts_r: np.ndarray):
    """用相似变换拟合内点，返回 left->right 尺度；失败返回 None。"""
    if len(pts_l) < 3:
        return None
    try:
        affine, _ = cv2.estimateAffinePartial2D(pts_l, pts_r, method=cv2.LMEDS)
    except cv2.error:
        return None
    if affine is None or not np.all(np.isfinite(affine)):
        return None
    scale = float(np.hypot(float(affine[0, 0]), float(affine[1, 0])))
    if not math.isfinite(scale) or scale <= 0.0:
        return None
    return scale


def _homography_scale(homography: np.ndarray, x: float, y: float):
    """在 (x, y) 处取 homography 的局部面积尺度（Jacobian 行列式的平方根）。"""
    if abs(float(homography[2, 2])) < 1e-12:
        return None
    h = homography / homography[2, 2]
    w = h[2, 0] * x + h[2, 1] * y + h[2, 2]
    if abs(float(w)) < 1e-12:
        return None
    u = (h[0, 0] * x + h[0, 1] * y + h[0, 2]) / w
    v = (h[1, 0] * x + h[1, 1] * y + h[1, 2]) / w
    j00 = (h[0, 0] - u * h[2, 0]) / w
    j01 = (h[0, 1] - u * h[2, 1]) / w
    j10 = (h[1, 0] - v * h[2, 0]) / w
    j11 = (h[1, 1] - v * h[2, 1]) / w
    determinant = j00 * j11 - j01 * j10
    if not math.isfinite(float(determinant)) or determinant <= 0.0:
        return None
    return math.sqrt(float(determinant))


def _scale_sane(scale: float, max_scale: float) -> bool:
    """左->右尺度是否在 [1/max_scale, max_scale] 内。"""
    return math.isfinite(scale) and (1.0 / max_scale) <= scale <= max_scale


def _quality(n_inliers: int, n_matches: int, min_inliers: int) -> float:
    """质量 = 内点比例 × 相对 min_inliers 的支撑度，落在 [0, 1]。"""
    if n_matches <= 0:
        return 0.0
    inlier_ratio = n_inliers / float(n_matches)
    support = min(1.0, n_inliers / max(1.0, 2.0 * float(min_inliers)))
    return float(min(1.0, max(0.0, inlier_ratio * support)))


def _mapped_rect(corners: np.ndarray, homography: np.ndarray,
                 wide_w: int, wide_h: int) -> tuple[int, int, int, int]:
    """把参考帧四角映射到广角帧，要求**完整落在广角帧内**，再取包围盒。

    如果任一映射角越出广角帧超过 ``_MAPPED_OUTSIDE_TOL_PX``（只容忍浮点 /
    取整误差），说明参考帧并没有完整落在更广的那一路里 —— 两路共同视场不足。
    这时必须报错：把越界角裁到边界再继续，等于拿“部分重叠”做主动对齐，
    像素尺度与视差会被静默污染。
    """
    try:
        mapped = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    except cv2.error as exc:
        raise ValueError(f"参考帧四角映射失败：{exc}") from exc
    if not np.all(np.isfinite(mapped)):
        raise ValueError("参考帧四角映射出了非有限坐标")

    # 每个角到广角帧矩形的外越量（px）：x / y 各自取正的外越部分。
    overshoot = np.maximum(
        np.maximum(-mapped[:, 0], mapped[:, 0] - float(wide_w)), 0.0)
    overshoot = np.maximum(
        overshoot,
        np.maximum(np.maximum(-mapped[:, 1], mapped[:, 1] - float(wide_h)), 0.0))
    n_out = int(np.sum(overshoot > _MAPPED_OUTSIDE_TOL_PX))
    if n_out:
        worst = int(np.argmax(overshoot))
        raise ValueError(
            "共同视场不足（reference frame not fully inside the wider frame）："
            f"参考帧四角中有 {n_out} 个映射到广角帧后越界超过 "
            f"{_MAPPED_OUTSIDE_TOL_PX:.0f} px（最大 "
            f"{float(overshoot[worst]):.1f} px，角点 "
            f"({float(mapped[worst, 0]):.1f}, {float(mapped[worst, 1]):.1f})，"
            f"广角帧 {wide_w}x{wide_h}）。参考帧没有完整落在更广那一路的视场里，"
            "拒绝按部分重叠做主动对齐；请调整相机安装 / 重新标定，"
            "或让两路对准同一片共同视场"
        )

    # 越界量在容差内：按取整落到帧内，避免 floor/ceil 产生 -1 或 >宽 的坐标。
    xs = np.clip(mapped[:, 0], 0.0, float(wide_w))
    ys = np.clip(mapped[:, 1], 0.0, float(wide_h))
    x0 = int(np.floor(float(xs.min())))
    y0 = int(np.floor(float(ys.min())))
    x1 = int(np.ceil(float(xs.max())))
    y1 = int(np.ceil(float(ys.max())))
    return x0, y0, x1 - x0, y1 - y0


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def _estimate_with(method: str, gray_l: np.ndarray, gray_r: np.ndarray,
                   min_inliers: int, max_scale: float) -> FovAlignment:
    n_l, n_r, n_matches, pts_l, pts_r = _match_features(gray_l, gray_r, method)
    if n_matches < min_inliers:
        raise ValueError(
            f"{method.upper()} 特征 {n_l}/{n_r}，ratio+交叉校验后匹配 "
            f"{n_matches} < min_inliers={min_inliers}")

    notes: list[str] = []
    model = None
    homography, mask, n_inliers = _find_homography(pts_l, pts_r)
    if homography is not None and n_inliers >= min_inliers:
        inlier_l = pts_l[mask]
        inlier_r = pts_r[mask]
        scale = _similarity_scale(inlier_l, inlier_r)
        if scale is None:
            scale = _homography_scale(
                homography, gray_l.shape[1] / 2.0, gray_l.shape[0] / 2.0)
        if scale is not None and _scale_sane(scale, max_scale):
            model = (homography, n_inliers, scale)
        else:
            notes.append(
                f"homography 尺度 {scale!r} 超出 [{1.0 / max_scale:.3f}, "
                f"{max_scale:.3f}]")
    else:
        notes.append(f"homography 内点 {n_inliers} < {min_inliers}")

    if model is None:
        affine, _affine_mask, n_affine = _find_affine(pts_l, pts_r)
        if affine is not None and n_affine >= min_inliers:
            scale = math.sqrt(abs(float(np.linalg.det(affine[:2, :2]))))
            if _scale_sane(scale, max_scale):
                model = (affine, n_affine, scale)
            else:
                notes.append(
                    f"仿射回退尺度 {scale:.3f} 超出 [{1.0 / max_scale:.3f}, "
                    f"{max_scale:.3f}]")
        else:
            notes.append(f"仿射回退内点 {n_affine} < {min_inliers}")

    if model is None:
        raise ValueError(
            f"{method.upper()} 几何估计不可用（内点/尺度不足）：" + "；".join(notes))

    homography, n_inliers, scale = model

    # 哪一路更窄，就以哪一路为参考；裁剪另一路。
    if scale > 1.0:
        reference_side, crop_side = "right", "left"
    else:
        reference_side, crop_side = "left", "right"

    if reference_side == "left":
        ref_h, ref_w = gray_l.shape[:2]
        wide_h, wide_w = gray_r.shape[:2]
        homography_ref_to_wide = homography
    else:
        ref_h, ref_w = gray_r.shape[:2]
        wide_h, wide_w = gray_l.shape[:2]
        try:
            homography_ref_to_wide = np.linalg.inv(homography)
        except np.linalg.LinAlgError as exc:
            raise ValueError("homography 不可逆，无法把参考帧映射到广角帧") from exc
        if not np.all(np.isfinite(homography_ref_to_wide)):
            raise ValueError("homography 的逆出现非有限数值")

    corners = np.float32([
        [0.0, 0.0], [float(ref_w), 0.0], [float(ref_w), float(ref_h)],
        [0.0, float(ref_h)],
    ]).reshape(-1, 1, 2)
    x, y, w, h = _mapped_rect(corners, homography_ref_to_wide, wide_w, wide_h)
    if w < 2 or h < 2:
        raise ValueError(
            f"参考帧映射到 {crop_side} 帧后包围盒退化：{(x, y, w, h)}")

    area_ratio = (w * h) / float(wide_w * wide_h)
    if area_ratio < _MIN_CROP_AREA_RATIO:
        raise ValueError(
            f"参考帧在 {crop_side} 帧内只占 {area_ratio:.1%} 面积 "
            f"(< {_MIN_CROP_AREA_RATIO:.0%})，重叠不足")

    quality = _quality(n_inliers, n_matches, min_inliers)
    # 共同输出分辨率取“不放大”的交集：参考帧与裁剪区都不被 upscale。
    # 只把较窄/参考帧 downscale 到广角裁剪区的原生像素规模；若参考更小则不放大宽帧。
    # 这样既保持同一视场，又不把 1685x979 的裁剪硬拉到 2560x1440 导致模糊。
    target_size = (min(ref_w, w), min(ref_h, h))

    # 尺度接近 1 且中心偏移小、参考帧几乎完整可见 -> 视场基本一致，不裁不缩。
    if abs(scale - 1.0) <= _NEAR_UNITY:
        center = np.float32([[[ref_w / 2.0, ref_h / 2.0]]])
        mapped_center = cv2.perspectiveTransform(
            center, homography_ref_to_wide)[0, 0]
        offset = float(np.hypot(
            float(mapped_center[0]) - wide_w / 2.0,
            float(mapped_center[1]) - wide_h / 2.0))
        overlap = min(1.0, (w * h) / float(ref_w * ref_h))
        if (offset <= _NEAR_CENTER_FRAC * min(wide_w, wide_h)
                and overlap >= _NEAR_OVERLAP):
            return FovAlignment(
                reference_side=reference_side,
                crop_side="none",
                crop_rect=(0, 0, ref_w, ref_h),
                target_size=target_size,
                scale=float(scale),
                method=method,
                inliers=n_inliers,
                quality=quality,
            )

    return FovAlignment(
        reference_side=reference_side,
        crop_side=crop_side,
        crop_rect=(x, y, w, h),
        target_size=target_size,
        scale=float(scale),
        method=method,
        inliers=n_inliers,
        quality=quality,
    )


def estimate_fov_alignment(left: np.ndarray, right: np.ndarray, *,
                           min_inliers: int = 30,
                           max_scale: float = 2.5) -> FovAlignment:
    """估计两路画面的 FOV 对齐关系。

    参数
    ----
    left, right : BGR / BGRA / 灰度 ``np.ndarray``（uint8 最稳）。
    min_inliers : 几何模型所需的最少 RANSAC 内点数，至少 4。
    max_scale   : 允许的最大视场尺度比（左->右尺度须在 ``[1/max_scale, max_scale]``）。

    返回
    ----
    :class:`FovAlignment`。``active`` 为 False 时两路视场基本一致。

    异常
    ----
    ``ValueError``：特征 / 匹配 / 内点 / 重叠不足，或估计出的尺度/裁剪区不可信。
    **不会**静默退化成“不对齐”。
    """
    try:
        min_inliers = int(min_inliers)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"min_inliers 必须是整数，收到 {min_inliers!r}") from exc
    if min_inliers < 4:
        raise ValueError("min_inliers 至少为 4（homography 需要 4 对点）")

    try:
        max_scale = float(max_scale)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"max_scale 必须是正数，收到 {max_scale!r}") from exc
    if not math.isfinite(max_scale) or max_scale <= 1.0:
        raise ValueError("max_scale 必须 > 1.0")

    gray_l = _as_gray(left, "left")
    gray_r = _as_gray(right, "right")
    if min(gray_l.shape) < 16 or min(gray_r.shape) < 16:
        raise ValueError(
            f"帧太小（left {gray_l.shape[1]}x{gray_l.shape[0]}，"
            f"right {gray_r.shape[1]}x{gray_r.shape[0]}），无法估计 FOV")

    failures: list[str] = []
    for method in ("sift", "orb"):
        try:
            return _estimate_with(method, gray_l, gray_r, min_inliers, max_scale)
        except (ValueError, cv2.error, AttributeError) as exc:
            # AttributeError：某些 OpenCV 构建没有 SIFT/ORB 工厂函数。
            failures.append(f"{method}: {exc}")

    raise ValueError(
        "FOV 自动对齐失败（SIFT/ORB 都未得到可用估计）：" + "；".join(failures))
