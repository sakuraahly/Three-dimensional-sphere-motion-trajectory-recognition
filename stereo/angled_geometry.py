"""Geometry for wide-angle ("angled") stereo without image rectification.

The parallel pipeline rectifies both images and triangulates in the
rectified frames.  This module is the independent geometry kernel for the
angled mode: it uses the original pixels plus ``K/D/R/T`` directly.

Conventions
-----------
* OpenCV camera frame: +X right, +Y down, +Z forward.
* ``X_right = R @ X_left + T``; ``R`` is 3x3 and ``T`` is in metres.
* A ray returned here is homogeneous ``[x, y, 1]`` in normalised camera
  coordinates (the distortion-free pinhole plane), not a rectified pixel.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

_EPS = 1e-12
# Relative threshold on sin^2(theta) between the two rays.  This only
# rejects numerically parallel rays; useful-angle policy belongs upstream.
_PARALLEL_SIN2_EPS = 1e-12
_OPTICAL_AXIS = np.array([0.0, 0.0, 1.0], dtype=np.float64)


@dataclass(frozen=True)
class AngledTriangulation:
    """Result of one non-rectified two-view triangulation.

    Attributes
    ----------
    xyz:
        Midpoint of the closest points on the two rays, expressed in the
        left camera frame, in metres, shape ``(3,)``.
    angle_deg:
        Angle between the left viewing ray and the right viewing ray
        after both are expressed in the left camera frame.
    residual_m:
        Half the distance between the two rays' closest points, in metres.
    reproj_left_px:
        Euclidean reprojection error in the original left image.
    reproj_right_px:
        Euclidean reprojection error in the original right image.
    """

    xyz: np.ndarray
    angle_deg: float
    residual_m: float
    reproj_left_px: float
    reproj_right_px: float


def _as_K(K: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(K, dtype=np.float64)
    if arr.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {arr.shape}")
    return arr


def _as_dist_coeffs(D: np.ndarray | None) -> np.ndarray | None:
    if D is None:
        return None
    arr = np.asarray(D, dtype=np.float64).reshape(-1)
    # OpenCV accepts None for "no distortion"; an empty array is not
    # equally portable across versions.
    return arr if arr.size else None


def _as_R_T(R: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R_arr = np.asarray(R, dtype=np.float64)
    if R_arr.shape != (3, 3):
        raise ValueError(f"R must have shape (3, 3), got {R_arr.shape}")
    T_arr = np.asarray(T, dtype=np.float64).reshape(-1)
    if T_arr.size != 3:
        raise ValueError(f"T must contain 3 values, got shape {T_arr.shape}")
    return R_arr, T_arr


def _as_pixel(pixel: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(pixel, dtype=np.float64).reshape(-1)
    if arr.size != 2:
        raise ValueError(f"{name} must contain 2 values, got shape {arr.shape}")
    return arr


def _finite(*arrays: np.ndarray) -> bool:
    return all(bool(np.all(np.isfinite(arr))) for arr in arrays)


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = (float(v[0]), float(v[1]), float(v[2]))
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )


def _project_pixel(
    xyz: np.ndarray,
    K: np.ndarray,
    D: np.ndarray | None,
) -> np.ndarray:
    """Project a point in one camera frame with that camera's K/D."""
    image_points, _ = cv2.projectPoints(
        np.asarray(xyz, dtype=np.float64).reshape(1, 1, 3),
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        K,
        D,
    )
    return np.asarray(image_points, dtype=np.float64).reshape(2)


def normalized_ray(
    K: np.ndarray,
    D: np.ndarray | None,
    pixel: np.ndarray,
) -> np.ndarray:
    """Undistort one original pixel and return its normalised ray.

    ``cv2.undistortPoints`` is called **without** a ``P`` argument, so the
    result is in the normalized pinhole plane rather than in a rectified or
    new-camera pixel plane.  The returned homogeneous vector is
    ``[x, y, 1]`` with dtype ``float64`` and shape ``(3,)``.
    """
    K_arr = _as_K(K, "K")
    D_arr = _as_dist_coeffs(D)
    pixel_arr = _as_pixel(pixel, "pixel")
    if not _finite(pixel_arr):
        raise ValueError("pixel values must be finite")

    undistorted = cv2.undistortPoints(
        pixel_arr.reshape(1, 1, 2),
        K_arr,
        D_arr,
    )
    x, y = np.asarray(undistorted, dtype=np.float64).reshape(2)
    return np.array([x, y, 1.0], dtype=np.float64)


def fundamental_from_pose(
    K1: np.ndarray,
    D1: np.ndarray | None,
    K2: np.ndarray,
    D2: np.ndarray | None,
    R: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    """Return the 3x3 fundamental matrix for corresponding camera pixels.

    The pose follows ``X_right = R @ X_left + T``.  For ideal pinhole
    pixels this is::

        F = K2^-T @ skew(T) @ R @ K1^-1

    Distortion coefficients are accepted only to keep the call-site
    signature parallel to :func:`triangulate_pixels`; they are deliberately
    **not used**.  Lens distortion is nonlinear and cannot be represented
    by a 3x3 fundamental matrix.  For distorted images, first undistort the
    pixels (for example with ``cv2.undistortPoints(..., P=K)``, which keeps
    them in the K pixel frame) and then apply ``F``.  Applied directly to
    raw distorted pixels, this matrix is only approximate; the metric path
    :func:`triangulate_pixels` handles raw pixels internally.
    """
    K1_arr = _as_K(K1, "K1")
    K2_arr = _as_K(K2, "K2")
    R_arr, T_arr = _as_R_T(R, T)

    # Explicit no-op reference: omitting D is intentional, not forgotten.
    _ = (D1, D2)

    K1_inv = np.linalg.inv(K1_arr)
    K2_inv_t = np.linalg.inv(K2_arr).T
    F = K2_inv_t @ _skew(T_arr) @ R_arr @ K1_inv
    return np.asarray(F, dtype=np.float64)


def epipolar_line(F: np.ndarray, pixel_left: np.ndarray) -> np.ndarray:
    """Return ``(a, b, c)`` for the epipolar line in the right image.

    The line is normalized so ``a**2 + b**2 == 1`` whenever the
    homogeneous line has a finite nonzero normal.  If it does not (for
    example a line at infinity), the unnormalized coefficients are
    returned instead.
    """
    F_arr = np.asarray(F, dtype=np.float64)
    if F_arr.shape != (3, 3):
        raise ValueError(f"F must have shape (3, 3), got {F_arr.shape}")
    p = _as_pixel(pixel_left, "pixel_left")
    if not _finite(p):
        raise ValueError("pixel_left values must be finite")

    line = F_arr @ np.array([p[0], p[1], 1.0], dtype=np.float64)
    normal_length = float(np.hypot(line[0], line[1]))
    if np.isfinite(normal_length) and normal_length > 0.0:
        normalized = line / normal_length
        if _finite(normalized):
            line = normalized
    return np.asarray(line, dtype=np.float64).reshape(3)


def point_line_distance(line: np.ndarray, pixel_right: np.ndarray) -> float:
    """Return the perpendicular pixel distance from a point to a line."""
    line_arr = np.asarray(line, dtype=np.float64).reshape(-1)
    if line_arr.size != 3:
        raise ValueError(f"line must contain 3 values, got shape {line_arr.shape}")
    p = _as_pixel(pixel_right, "pixel_right")
    if not _finite(line_arr, p):
        raise ValueError("line and pixel_right values must be finite")

    a, b, c = (float(line_arr[0]), float(line_arr[1]), float(line_arr[2]))
    normal_length = float(np.hypot(a, b))
    if normal_length <= 0.0:
        raise ValueError("cannot measure distance to a degenerate line (a=b=0)")
    return float(abs(a * p[0] + b * p[1] + c) / normal_length)


def triangulate_pixels(
    K1: np.ndarray,
    D1: np.ndarray | None,
    K2: np.ndarray,
    D2: np.ndarray | None,
    R: np.ndarray,
    T: np.ndarray,
    p_left: np.ndarray,
    p_right: np.ndarray,
) -> AngledTriangulation | None:
    """Triangulate original left/right pixels without rectification.

    Both pixels are undistorted into normalized rays.  The right ray is
    expressed in the left camera frame with ``R.T``; its origin there is
    ``-R.T @ T``.  The closest points on the two rays are found, and their
    midpoint is returned in the left camera frame.

    ``None`` is returned for parallel/degenerate rays, non-finite values,
    or a solution that is not in front of both cameras.  It is a normal
    rejection, not an exception, so callers can use it per detection.
    """
    K1_arr = _as_K(K1, "K1")
    K2_arr = _as_K(K2, "K2")
    R_arr, T_arr = _as_R_T(R, T)
    p1 = _as_pixel(p_left, "p_left")
    p2 = _as_pixel(p_right, "p_right")
    D1_arr = _as_dist_coeffs(D1)
    D2_arr = _as_dist_coeffs(D2)

    if not _finite(K1_arr, K2_arr, R_arr, T_arr, p1, p2):
        return None
    if D1_arr is not None and not _finite(D1_arr):
        return None
    if D2_arr is not None and not _finite(D2_arr):
        return None

    ray1 = normalized_ray(K1_arr, D1_arr, p1)
    ray2 = normalized_ray(K2_arr, D2_arr, p2)
    if not _finite(ray1, ray2):
        return None

    # Left ray: origin 0, direction ray1.
    # Right ray in left coordinates: origin -R.T @ T, direction R.T @ ray2.
    d1 = ray1
    d2 = R_arr.T @ ray2
    origin2 = -R_arr.T @ T_arr
    if not _finite(d1, d2, origin2):
        return None

    a = float(d1 @ d1)
    b = float(d1 @ d2)
    c = float(d2 @ d2)
    if a <= _EPS or c <= _EPS:
        return None

    # sin^2(angle) * |d1|^2 * |d2|^2.  Reject both parallel and
    # anti-parallel rays, where the closest-point solve is singular.
    determinant = a * c - b * b
    if not np.isfinite(determinant) or determinant <= _PARALLEL_SIN2_EPS * a * c:
        return None

    # Closest points P1 = s*d1 and P2 = origin2 + t*d2 solve
    # [a -b; b -c] [s, t]^T = [d1.origin2, d2.origin2]^T.
    d = float(d1 @ origin2)
    e = float(d2 @ origin2)
    s = (c * d - b * e) / determinant
    t = (b * d - a * e) / determinant
    if not np.isfinite(s) or not np.isfinite(t):
        return None

    # With normalized rays z=1, s and t are exactly the left/right
    # camera-space depths of the closest points.
    if s <= 0.0 or t <= 0.0:
        return None

    point_left = s * d1
    point_right = origin2 + t * d2
    xyz = np.asarray(0.5 * (point_left + point_right), dtype=np.float64)
    residual = 0.5 * float(np.linalg.norm(point_left - point_right))
    if not _finite(xyz) or not np.isfinite(residual):
        return None

    xyz_right = R_arr @ xyz + T_arr
    if xyz[2] <= 0.0 or xyz_right[2] <= 0.0:
        return None

    cos_angle = b / float(np.sqrt(a * c))
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cos_angle)))

    projected_left = _project_pixel(xyz, K1_arr, D1_arr)
    projected_right = _project_pixel(xyz_right, K2_arr, D2_arr)
    reproj_left = float(np.linalg.norm(projected_left - p1))
    reproj_right = float(np.linalg.norm(projected_right - p2))

    if not _finite(np.array([angle_deg, residual, reproj_left, reproj_right])):
        return None

    return AngledTriangulation(
        xyz=xyz,
        angle_deg=angle_deg,
        residual_m=residual,
        reproj_left_px=reproj_left,
        reproj_right_px=reproj_right,
    )


def optical_axes_angle_deg(R: np.ndarray) -> float:
    """Angle between +Z (left axis) and the right axis in the left frame.

    The right optical axis is ``R.T @ [0, 0, 1]`` under the module's pose
    convention.  The result is in ``[0, 180]`` degrees.
    """
    R_arr = np.asarray(R, dtype=np.float64)
    if R_arr.shape != (3, 3):
        raise ValueError(f"R must have shape (3, 3), got {R_arr.shape}")

    right_axis_in_left = R_arr.T @ _OPTICAL_AXIS
    norm = float(np.linalg.norm(right_axis_in_left))
    if not np.isfinite(norm) or norm <= _EPS:
        raise ValueError("R produces a degenerate right optical axis")

    cos_angle = float(right_axis_in_left[2] / norm)
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def _self_test() -> None:
    """Synthetic checks with known 3D points; no calibration files needed."""
    K = np.array(
        [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    D = np.zeros(5, dtype=np.float64)
    baseline_m = 0.4
    right_center_in_left = np.array([baseline_m, 0.0, 0.0], dtype=np.float64)

    cases = (
        (
            60.0,
            np.array(
                [
                    [0.00, 0.00, 0.30],
                    [-0.05, 0.05, 0.36],
                    [0.06, -0.04, 0.42],
                ],
                dtype=np.float64,
            ),
        ),
        (
            90.0,
            np.array(
                [
                    [0.04, 0.00, 0.14],
                    [0.06, 0.03, 0.16],
                    [0.02, -0.02, 0.18],
                ],
                dtype=np.float64,
            ),
        ),
    )

    axis_errors = []
    axis_values = []
    normalized_checked = False
    for angle_deg, points in cases:
        R = cv2.Rodrigues(
            np.array([0.0, np.deg2rad(angle_deg), 0.0], dtype=np.float64)
        )[0]
        # The right camera centre is +X in the left frame; the pose
        # parameter is the translation in X_right = R @ X_left + T.
        T = -R @ right_center_in_left

        axis_angle = optical_axes_angle_deg(R)
        axis_errors.append(abs(axis_angle - angle_deg))
        axis_values.append(axis_angle)
        F = fundamental_from_pose(K, D, K, D, R, T)
        assert F.shape == (3, 3)

        max_3d_error = 0.0
        max_epipolar_px = 0.0
        ray_angles = []
        for X in points:
            X_right = R @ X + T
            p_left = _project_pixel(X, K, D)
            p_right = _project_pixel(X_right, K, D)

            result = triangulate_pixels(K, D, K, D, R, T, p_left, p_right)
            assert result is not None, f"triangulation returned None at {angle_deg} deg"
            assert result.xyz.shape == (3,)
            error_m = float(np.linalg.norm(result.xyz - X))
            max_3d_error = max(max_3d_error, error_m)
            assert error_m < 1e-6, f"3D error {error_m} m at {angle_deg} deg"
            assert 10.0 < result.angle_deg < 120.0, (
                f"implausible ray angle {result.angle_deg} deg at {angle_deg} deg"
            )
            assert result.reproj_left_px < 1e-6
            assert result.reproj_right_px < 1e-6
            ray_angles.append(result.angle_deg)

            line = epipolar_line(F, p_left)
            assert abs(float(np.hypot(line[0], line[1])) - 1.0) < 1e-12
            epi_px = point_line_distance(line, p_right)
            max_epipolar_px = max(max_epipolar_px, epi_px)

            if not normalized_checked:
                ray = normalized_ray(K, D, p_left)
                assert ray.shape == (3,) and ray.dtype == np.float64
                np.testing.assert_allclose(
                    ray,
                    X / X[2],
                    rtol=0.0,
                    atol=1e-12,
                )
                normalized_checked = True

        assert max_epipolar_px < 1e-6, (
            f"epipolar distance {max_epipolar_px} px at {angle_deg} deg"
        )
        print(
            f"PASS triangulation @ {angle_deg:.0f} deg rotation: "
            f"3 points, max 3D error={max_3d_error:.3e} m, "
            f"ray angle={min(ray_angles):.3f}-{max(ray_angles):.3f} deg"
        )
        print(
            f"PASS epipolar @ {angle_deg:.0f} deg rotation: "
            f"max pixel distance={max_epipolar_px:.3e} px"
        )

    assert normalized_checked
    print("PASS normalized_ray: shape=(3,), dtype=float64, normalized x/y")
    assert max(axis_errors) < 1e-9
    print(
        "PASS optical_axes_angle_deg: "
        f"{axis_values[0]:.6f} / {axis_values[1]:.6f} deg"
    )

    identity = np.eye(3, dtype=np.float64)
    zero_translation = np.zeros(3, dtype=np.float64)
    parallel = triangulate_pixels(
        K,
        D,
        K,
        D,
        identity,
        zero_translation,
        np.array([320.0, 240.0]),
        np.array([320.0, 240.0]),
    )
    assert parallel is None
    print("PASS degenerate parallel rays rejected with None")
    print("PASS angled_geometry self-test")


if __name__ == "__main__":
    _self_test()
