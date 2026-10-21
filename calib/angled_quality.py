"""Calibration-quality metrics for angled ("approximately 90°") stereo.

The parallel pipeline judges a calibration by the residual vertical
disparity left after rectification.  An angled rig cannot use that metric:
the two optical axes meet at a large angle, so rectification either fails
or badly distorts the overlap.  This module therefore measures quality on
the **original pixels**:

* ``line_rms_px`` / ``line_max_px``
      Distance from each right pixel to the epipolar line induced by its
      left partner through ``F``.  ``F`` is computed once from the pose.
* ``median_angle_deg`` / ``min_angle_deg``
      Angle between the two viewing rays of each triangulated point,
      expressed in the left camera frame.  No rectification is involved.
* ``reproj_rms_px``
      Optional echo of the caller's reprojection RMS (for example from
      ``cv2.stereoCalibrate``).  It is **not** recomputed here: the caller
      may have no per-pair reprojection values, and the definition of that
      number belongs to the calibration path.  NaN means "not supplied".

Conventions
-----------
* OpenCV camera frame: +X right, +Y down, +Z forward.
* ``X_right = R @ X_left + T``; ``R`` is 3x3 and ``T`` is in metres.
* ``K1/D1`` and ``K2/D2`` are the original (unrectified) intrinsics.

Distortion caveat: a 3x3 fundamental matrix cannot represent lens
distortion, so the epipolar line distance is exact only for pinhole /
undistorted pixels.  :func:`stereo.angled_geometry.triangulate_pixels`
does undistort internally, so the crossing-angle metrics stay metric even
when ``D`` is strongly non-zero.  With strong distortion, expect the line
distance to include some model error.

Input handling
--------------
``left_pixels`` / ``right_pixels`` are arrays or lists of ``(u, v)``.
Their lengths must match.  Rows containing NaN/Inf are ignored and the
skipped count is reported in ``warnings``.  If nothing usable remains,
:func:`evaluate_angled` does **not** raise: it returns ``n_pairs=0``,
undefined (NaN) metrics and an explanatory warning.  Malformed pixel
shapes and unequal lengths do raise :class:`ValueError` -- those are
programming errors, not "bad data".

The module is pure: no camera, GUI, app or file-system dependency.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    # Allow ``python calib/angled_quality.py``: the script directory, not
    # the project root, is sys.path[0] in that mode.
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

__all__ = ["AngledQuality", "evaluate_angled", "grade_angled"]

_NAN = float("nan")

# A line error beyond this multiple of the usable ceiling is treated as a
# physically wrong / diverged calibration, not merely a precision miss.
# Matches the "diverged" spirit of ``calib.stereo_calib`` (which calls a
# reprojection RMS above 5 px diverged) without inventing a second user
# parameter: the caller already controls the scale through ``ok_line``.
_LINE_DIVERGENCE_FACTOR = 10.0


@dataclass(frozen=True)
class AngledQuality:
    """Aggregated quality of an angled stereo calibration.

    Attributes
    ----------
    n_pairs:
        Number of pixel pairs left after dropping non-finite rows.  Line
        and angle statistics can use fewer values when an individual pair
        is degenerate; ``warnings`` says so.
    line_rms_px:
        Root-mean-square point-to-epipolar-line distance, pixels.  NaN when
        no pair yielded a finite distance.
    line_max_px:
        Largest point-to-epipolar-line distance, pixels.  NaN when no pair
        yielded a finite distance.
    median_angle_deg:
        Median ray crossing angle over successfully triangulated pairs.
        NaN when none were triangulated.
    min_angle_deg:
        Smallest ray crossing angle over successfully triangulated pairs.
        NaN when none were triangulated.
    reproj_rms_px:
        Caller-supplied reprojection RMS, echoed unchanged; NaN if absent
        or non-finite.
    warnings:
        Human-readable Chinese notes about skipped / degenerate pairs.
    """

    n_pairs: int
    line_rms_px: float
    line_max_px: float
    median_angle_deg: float
    min_angle_deg: float
    reproj_rms_px: float
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        # A frozen dataclass with a list field is only shallowly frozen;
        # normalise so callers can rely on the declared tuple type.
        object.__setattr__(self, "warnings", tuple(self.warnings))

    def as_dict(self) -> dict:
        """Return a JSON-friendly dictionary (``warnings`` as a list)."""
        return {
            "n_pairs": int(self.n_pairs),
            "line_rms_px": float(self.line_rms_px),
            "line_max_px": float(self.line_max_px),
            "median_angle_deg": float(self.median_angle_deg),
            "min_angle_deg": float(self.min_angle_deg),
            "reproj_rms_px": float(self.reproj_rms_px),
            "warnings": list(self.warnings),
        }


def _import_geometry():
    """Import the geometry kernel lazily.

    ``stereo/__init__.py`` imports :mod:`calib.params`.  Importing that
    package while ``calib`` itself is still initialising (for example if
    this module is re-exported from ``calib/__init__.py``) can produce a
    circular import.  Deferring the import until evaluation keeps
    ``import calib.angled_quality`` safe in any import order.
    """
    from stereo.angled_geometry import (
        epipolar_line,
        fundamental_from_pose,
        point_line_distance,
        triangulate_pixels,
    )

    return (
        epipolar_line,
        fundamental_from_pose,
        point_line_distance,
        triangulate_pixels,
    )


def _as_pixel_array(points: object, name: str) -> np.ndarray:
    """Coerce a list/array of ``(u, v)`` to ``float64`` shape ``(N, 2)``."""
    arr = np.asarray(points, dtype=np.float64)
    if arr.size == 0:
        # Empty input is valid "no data", whatever its exact shape.
        return np.empty((0, 2), dtype=np.float64)
    if arr.ndim == 1:
        if arr.size != 2:
            raise ValueError(f"{name} must have shape (N, 2); got {arr.shape}")
        return arr.reshape(1, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2); got {arr.shape}")
    return arr


def _as_reproj(value: object) -> float:
    """Normalise the optional reprojection RMS to a float or NaN."""
    if value is None:
        return _NAN
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"reproj_rms_px must be a number or None; got {value!r}"
        ) from None
    return number if math.isfinite(number) else _NAN


def evaluate_angled(
    K1: np.ndarray,
    D1: np.ndarray | None,
    K2: np.ndarray,
    D2: np.ndarray | None,
    R: np.ndarray,
    T: np.ndarray,
    left_pixels: object,
    right_pixels: object,
    reproj_rms_px: float = _NAN,
) -> AngledQuality:
    """Measure an angled calibration from corresponding original pixels.

    The fundamental matrix is built once with
    :func:`stereo.angled_geometry.fundamental_from_pose`; every pair then
    contributes one epipolar point-line distance (left pixel -> right
    image) and, if triangulation succeeds, one ray crossing angle from
    :func:`stereo.angled_geometry.triangulate_pixels`.

    Parameters
    ----------
    K1, D1, K2, D2:
        Original left/right intrinsics and distortion.  ``D`` may be
        ``None`` or empty.
    R, T:
        Pose with ``X_right = R @ X_left + T``.
    left_pixels, right_pixels:
        Array-like of shape ``(N, 2)`` (a single ``(u, v)`` is accepted as
        one pair).  Lengths must match.
    reproj_rms_px:
        Echoed into the result when finite; NaN otherwise.  It is **not**
        recomputed, so callers can pass a calibration-wide value or leave
        it unset.

    Returns
    -------
    AngledQuality
        Never raises for empty / all-non-finite pixels: the result then
        has ``n_pairs=0``, NaN line and angle metrics, and a warning.
        Malformed pixel shapes or mismatched lengths raise ``ValueError``.
    """
    left = _as_pixel_array(left_pixels, "left_pixels")
    right = _as_pixel_array(right_pixels, "right_pixels")
    if left.shape[0] != right.shape[0]:
        raise ValueError(
            "left_pixels and right_pixels must have the same length; "
            f"got {left.shape[0]} and {right.shape[0]}"
        )

    warnings: list[str] = []
    finite_rows = np.all(np.isfinite(left), axis=1) & np.all(
        np.isfinite(right), axis=1
    )
    skipped = int(np.count_nonzero(~finite_rows))
    if skipped:
        warnings.append(f"跳过了 {skipped} 对含非有限坐标（NaN/Inf）的像素")
    left = left[finite_rows]
    right = right[finite_rows]
    reproj = _as_reproj(reproj_rms_px)

    if left.shape[0] == 0:
        warnings.append(
            "没有可用的像素对（输入为空或全部为非有限值），"
            "无法评估大角度标定质量"
        )
        return AngledQuality(
            n_pairs=0,
            line_rms_px=_NAN,
            line_max_px=_NAN,
            median_angle_deg=_NAN,
            min_angle_deg=_NAN,
            reproj_rms_px=reproj,
            warnings=tuple(warnings),
        )

    (
        epipolar_line,
        fundamental_from_pose,
        point_line_distance,
        triangulate_pixels,
    ) = _import_geometry()

    F = fundamental_from_pose(K1, D1, K2, D2, R, T)

    line_distances: list[float] = []
    line_degenerate = 0
    angles: list[float] = []
    triangulation_failed = 0

    for p_left, p_right in zip(left, right):
        try:
            line = epipolar_line(F, p_left)
            distance = point_line_distance(line, p_right)
        except ValueError:
            # Degenerate line (a=b=0), or non-finite F from a bad pose.
            line_degenerate += 1
        else:
            if math.isfinite(distance):
                line_distances.append(float(distance))
            else:
                line_degenerate += 1

        triangulation = triangulate_pixels(
            K1, D1, K2, D2, R, T, p_left, p_right
        )
        if triangulation is None:
            triangulation_failed += 1
        else:
            angles.append(float(triangulation.angle_deg))

    if line_degenerate:
        warnings.append(
            f"{line_degenerate} 对的极线线距离无法计算（F 或线退化），"
            "未计入线误差统计"
        )
    if triangulation_failed:
        warnings.append(
            f"{triangulation_failed} 对无法三角测量"
            "（射线平行或交点不在两台相机前方），未计入交会角统计"
        )

    if line_distances:
        line_arr = np.asarray(line_distances, dtype=np.float64)
        line_rms = float(np.sqrt(np.mean(line_arr * line_arr)))
        line_max = float(np.max(line_arr))
    else:
        line_rms = _NAN
        line_max = _NAN

    finite_angles = [a for a in angles if math.isfinite(a)]
    if finite_angles:
        median_angle = float(np.median(finite_angles))
        min_angle = float(np.min(finite_angles))
    else:
        median_angle = _NAN
        min_angle = _NAN

    return AngledQuality(
        n_pairs=int(left.shape[0]),
        line_rms_px=line_rms,
        line_max_px=line_max,
        median_angle_deg=median_angle,
        min_angle_deg=min_angle,
        reproj_rms_px=reproj,
        warnings=tuple(warnings),
    )


def grade_angled(
    q: AngledQuality,
    min_pairs: int = 25,
    good_line: float = 0.5,
    ok_line: float = 1.0,
    good_angle: tuple[float, float] = (60.0, 100.0),
    min_angle: float = 30.0,
) -> tuple[str, str]:
    """Turn measured quality into a Chinese grade and verdict.

    Criteria (defaults in parentheses)
    --------------------------------
    优秀
        ``n_pairs >= min_pairs`` (25), ``line_rms <= good_line`` (0.5 px),
        every pair within ``ok_line`` (1.0 px), median crossing angle inside
        ``good_angle`` (60°..100°), and the worst angle at least
        ``min_angle`` (30°).
    可用
        ``n_pairs >= min_pairs``, ``line_rms <= ok_line``, and every pair
        at least ``min_angle``; this covers rigs that miss the "excellent"
        band (for example median 50°) but are still geometrically usable.
    勉强可用
        At least one usable pair and no hard failure, but one or more of:
        too few pairs, ``line_rms > ok_line``, or some pair below
        ``min_angle``.  It can add 3D points, but the numbers are below the
        project target.
    不可用
        No usable pair; line metric undefined or above
        ``10 * ok_line`` (diverged); no pair could be triangulated; or the
        median crossing angle is below ``min_angle`` (depth is nearly
        unconstrained).

    Too few pairs is deliberately **not** an automatic "不可用": mirroring
    ``calib.stereo_calib``, that hard label is reserved for physically
    wrong results.  A short but consistent capture is reported as
    "勉强可用" so the caller can see how far off it is.

    ``reproj_rms_px`` is reported in :class:`AngledQuality` but does not
    gate the grade: it is optional (NaN) and its pixel definition depends
    on the caller's calibration path.  Line distance and crossing angle are
    the criteria specific to angled stereo.

    Raises
    ------
    ValueError
        For nonsensical thresholds (``min_pairs < 1``,
        ``good_line <= 0``, ``good_line > ok_line``, ``min_angle < 0``,
        non-finite or non-positive angle band).
    """
    if min_pairs < 1:
        raise ValueError(f"min_pairs must be >= 1; got {min_pairs}")
    if not (math.isfinite(good_line) and good_line > 0.0):
        raise ValueError(f"good_line must be finite and > 0; got {good_line}")
    if not (math.isfinite(ok_line) and ok_line > 0.0):
        raise ValueError(f"ok_line must be finite and > 0; got {ok_line}")
    if good_line > ok_line:
        raise ValueError(
            f"good_line ({good_line}) must not exceed ok_line ({ok_line})"
        )
    if not (math.isfinite(min_angle) and min_angle >= 0.0):
        raise ValueError(f"min_angle must be finite and >= 0; got {min_angle}")
    if len(good_angle) != 2:
        raise ValueError(f"good_angle must contain 2 values; got {good_angle!r}")
    angle_lo, angle_hi = (float(good_angle[0]), float(good_angle[1]))
    if not all(math.isfinite(v) and v > 0.0 for v in (angle_lo, angle_hi)):
        raise ValueError(
            f"good_angle values must be finite and > 0; got {good_angle!r}"
        )
    if angle_lo > angle_hi:
        angle_lo, angle_hi = angle_hi, angle_lo

    n_pairs = int(q.n_pairs)
    line_rms = float(q.line_rms_px)
    line_max = float(q.line_max_px)
    median_angle = float(q.median_angle_deg)
    worst_angle = float(q.min_angle_deg)
    hard_line = _LINE_DIVERGENCE_FACTOR * ok_line

    if n_pairs <= 0:
        return (
            "不可用",
            "不可用 —— 没有可用的像素对，无法评估大角度标定质量；"
            "请检查输入或重新采集标定图像",
        )

    if not math.isfinite(line_rms):
        return (
            "不可用",
            "不可用 —— 极线线距离无法计算（F 或所有极线退化），"
            "请检查 R/T、基线与左右图顺序",
        )

    if line_rms > hard_line:
        return (
            "不可用",
            f"不可用 —— 极线线距离 RMS {line_rms:.1f} px 已发散"
            f"（> {hard_line:.1f} px），标定数值不可信；请重拍重标",
        )

    if not math.isfinite(median_angle):
        return (
            "不可用",
            "不可用 —— 没有任何像素对能完成三角测量"
            "（射线平行或交点在相机后方），请检查标定与左右图顺序",
        )

    if median_angle < min_angle:
        return (
            "不可用",
            f"不可用 —— 中位交会角 {median_angle:.1f}° 低于 {min_angle:.0f}°，"
            "深度方向近乎无约束；请调整两台相机的夹角或基线",
        )

    excellent = (
        n_pairs >= min_pairs
        and line_rms <= good_line
        and math.isfinite(line_max)
        and line_max <= ok_line
        and angle_lo <= median_angle <= angle_hi
        and math.isfinite(worst_angle)
        and worst_angle >= min_angle
    )
    if excellent:
        return (
            "优秀",
            f"优秀 —— 极线线距离 RMS {line_rms:.2f} px"
            f"（最大 {line_max:.2f} px），中位交会角 {median_angle:.1f}°"
            f"（最小 {worst_angle:.1f}°），{n_pairs} 个像素对；"
            "可直接用于大角度三角测量",
        )

    usable = (
        n_pairs >= min_pairs
        and line_rms <= ok_line
        and math.isfinite(worst_angle)
        and worst_angle >= min_angle
    )
    if usable:
        if line_rms > good_line:
            why = f"线距离 RMS 高于优秀档 {good_line:.2f} px"
        elif not (angle_lo <= median_angle <= angle_hi):
            why = (
                f"中位交会角不在目标 {angle_lo:.0f}–{angle_hi:.0f}° 内"
            )
        elif math.isfinite(line_max) and line_max > ok_line:
            why = (
                f"最大线距离 {line_max:.2f} px 超过 {ok_line:.2f} px"
                "（存在离群对）"
            )
        else:
            why = "未全部达到优秀档"
        return (
            "可用",
            f"可用 —— 极线线距离 RMS {line_rms:.2f} px、"
            f"中位交会角 {median_angle:.1f}°、{n_pairs} 个像素对；"
            f"{why}，但仍在可用门限内，可用于三角测量",
        )

    reasons: list[str] = []
    if n_pairs < min_pairs:
        reasons.append(
            f"有效像素对只有 {n_pairs} 个（目标 ≥{min_pairs} 个）"
        )
    if line_rms > ok_line:
        reasons.append(
            f"极线线距离 RMS {line_rms:.2f} px 高于可用门限 {ok_line:.2f} px"
        )
    if not math.isfinite(worst_angle):
        reasons.append("最小交会角无法计算")
    elif worst_angle < min_angle:
        reasons.append(
            f"最小交会角 {worst_angle:.1f}° 低于 {min_angle:.0f}°"
        )
    if not reasons:
        reasons.append("部分指标未达到可用档")

    return (
        "勉强可用",
        "勉强可用 —— "
        + "；".join(reasons)
        + "。可用于三角测量，但精度/可信度低于目标，"
        "建议补拍并重标后再做正式测量",
    )


def _self_test() -> None:
    """Noise-free synthetic 90° rig + edge cases; no calibration files."""
    image_width, image_height = 640, 480
    K = np.array(
        [
            [250.0, 0.0, image_width / 2.0],
            [0.0, 250.0, image_height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    D = np.zeros(5, dtype=np.float64)

    # +90° about Y: the right camera sits at +X and looks back along -X.
    # R is exactly orthogonal; R.T @ [0, 0, 1] = [-1, 0, 0].
    R = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    baseline_m = 0.4
    # Right camera centre is +X in the left frame; pose is X_r = R X_l + T.
    T = np.array([0.0, 0.0, baseline_m], dtype=np.float64)

    right_axis = R.T @ np.array([0.0, 0.0, 1.0])
    cos_axis = float(
        np.clip(right_axis[2] / np.linalg.norm(right_axis), -1.0, 1.0)
    )
    axis_angle_deg = float(np.degrees(np.arccos(cos_axis)))
    assert abs(axis_angle_deg - 90.0) < 1e-9, axis_angle_deg

    xs = np.linspace(0.17, 0.23, 5)
    ys = np.linspace(-0.02, 0.02, 3)
    zs = np.linspace(0.19, 0.21, 2)
    points = np.array(
        [[x, y, z] for x in xs for y in ys for z in zs],
        dtype=np.float64,
    )

    def project(xyz: np.ndarray, *, right: bool) -> np.ndarray:
        xyz_cam = R @ xyz + T if right else xyz
        return K[:2, :2] @ (xyz_cam[:2] / xyz_cam[2]) + K[:2, 2]

    left_pixels = np.array([project(X, right=False) for X in points])
    right_pixels = np.array([project(X, right=True) for X in points])
    for pixels, label in ((left_pixels, "left"), (right_pixels, "right")):
        assert np.all(np.isfinite(pixels)), label
        assert np.all(pixels[:, 0] >= 0.0), label
        assert np.all(pixels[:, 0] < image_width), label
        assert np.all(pixels[:, 1] >= 0.0), label
        assert np.all(pixels[:, 1] < image_height), label
    assert len(points) == 30
    print(
        "PASS synthetic 90° pose: "
        f"optical-axis angle={axis_angle_deg:.3f}°, "
        f"{len(points)} point pairs inside {image_width}x{image_height}"
    )

    q = evaluate_angled(
        K, D, K, D, R, T, left_pixels, right_pixels, reproj_rms_px=0.42
    )
    assert q.n_pairs == 30, q.n_pairs
    assert q.line_rms_px < 1e-6, q.line_rms_px
    assert q.line_max_px < 1e-6, q.line_max_px
    assert 60.0 <= q.median_angle_deg <= 100.0, q.median_angle_deg
    assert q.min_angle_deg >= 30.0, q.min_angle_deg
    assert q.reproj_rms_px == 0.42, q.reproj_rms_px
    assert q.warnings == (), q.warnings
    print(
        "PASS evaluate_angled: "
        f"n_pairs={q.n_pairs}, line RMS={q.line_rms_px:.3e} px, "
        f"line max={q.line_max_px:.3e} px, "
        f"median angle={q.median_angle_deg:.3f}°, "
        f"min angle={q.min_angle_deg:.3f}°"
    )

    grade, verdict = grade_angled(q)
    assert grade != "不可用", verdict
    print(f"PASS grade_angled: {verdict}")

    payload = q.as_dict()
    assert isinstance(payload, dict)
    assert payload["n_pairs"] == 30 and isinstance(payload["n_pairs"], int)
    assert payload["warnings"] == [] and isinstance(payload["warnings"], list)
    assert payload["reproj_rms_px"] == 0.42
    print("PASS as_dict: int n_pairs, list warnings, all fields present")

    empty = evaluate_angled(K, D, K, D, R, T, [], [])
    assert empty.n_pairs == 0
    assert math.isnan(empty.line_rms_px)
    assert math.isnan(empty.line_max_px)
    assert math.isnan(empty.median_angle_deg)
    assert math.isnan(empty.min_angle_deg)
    assert math.isnan(empty.reproj_rms_px)
    assert any("没有可用的像素对" in w for w in empty.warnings)
    assert grade_angled(empty)[0] == "不可用"
    print(
        "PASS empty input: n_pairs=0, metrics=NaN, "
        f"{len(empty.warnings)} warning(s), grade=不可用"
    )

    left_with_nan = left_pixels.copy()
    right_with_inf = right_pixels.copy()
    left_with_nan[0, 0] = np.nan
    right_with_inf[1, 1] = np.inf
    q_skipped = evaluate_angled(
        K, D, K, D, R, T, left_with_nan, right_with_inf
    )
    assert q_skipped.n_pairs == 28, q_skipped.n_pairs
    assert any(
        "2" in warning and "非有限" in warning
        for warning in q_skipped.warnings
    ), q_skipped.warnings
    print(
        "PASS non-finite rows: 2 skipped, "
        f"n_pairs={q_skipped.n_pairs}, warning present"
    )

    def expect_value_error(func, label: str) -> None:
        try:
            func()
        except ValueError:
            print(f"PASS invalid input: {label} -> ValueError")
            return
        raise AssertionError(f"{label}: expected ValueError")

    expect_value_error(
        lambda: evaluate_angled(
            K, D, K, D, R, T, np.zeros((5, 3)), np.zeros((5, 3))
        ),
        "pixel array with 3 columns",
    )
    expect_value_error(
        lambda: evaluate_angled(
            K, D, K, D, R, T, np.zeros((5, 2)), np.zeros((4, 2))
        ),
        "unequal pixel counts",
    )
    expect_value_error(
        lambda: evaluate_angled(
            np.eye(2), D, K, D, R, T, left_pixels, right_pixels
        ),
        "K1 not 3x3",
    )

    print("PASS angled_quality self-test")


if __name__ == "__main__":
    _self_test()
