"""Angled stereo: fundamental-matrix matching + unrectified ray triangulation.

The parallel pipeline (``stereo.triangulate.StereoReconstructor``) matches on
rectified images, where epipolar lines are horizontal.  When the two optical
axes are far from parallel (the "angled" mode, roughly a right angle),
rectification crops/rotates most of the overlap, so this module matches the
**original** pixels with the fundamental matrix and triangulates the two rays
directly.

Geometry is delegated to :mod:`stereo.angled_geometry`; this module only owns
the per-frame policy:

* pick the right candidate with the smallest distance to the left point's
  epipolar line (:func:`stereo.angled_geometry.epipolar_line`);
* reject pairs whose epipolar distance, ray-intersection angle or reprojection
  error is outside :class:`AngledMatchConfig`;
* accumulate accepted points into a trajectory and expose frame counters.

Conventions (identical to ``angled_geometry``)
----------------------------------------------
* OpenCV camera frame: +X right, +Y down, +Z forward.
* ``X_right = R @ X_left + T``; ``T`` is in metres.
* ``AngledPoint.xyz`` is in the **original** left camera frame (metres), not
  in the rectified frame that ``StereoPoint.xyz`` uses.

Counter / match-rate contract
-----------------------------
Every :meth:`AngledReconstructor.update` call increments exactly one of
``n_matched`` / ``n_no_left`` / ``n_no_right`` / ``n_rejected``.  Hence
``n_frames`` (their sum) equals the number of calls and
``match_rate == n_matched / n_frames``.

Candidates that exist but fail the epipolar gate are counted in
``n_rejected``, **not** in ``n_no_right``: "the right camera supplied no
candidates" and "the candidates did not match" are different failures and
need different fixes (right-camera coverage vs. gate/calibration).  The same
counter covers a degenerate epipolar line, a failed triangulation, a
too-small ray angle and a too-large reprojection error.

Distortion caveat
-----------------
``fundamental_from_pose`` builds ``F`` from ``K``/``R``/``T`` only; a 3x3
matrix cannot represent lens distortion (that helper documents this
explicitly).  The epipolar gate below is therefore exact for distortion-free
pixels and approximate for strongly distorted ones.  Metric triangulation
(:func:`stereo.angled_geometry.triangulate_pixels`) undistorts internally and
is not affected.  The looser default ``max_epipolar_px=8.0`` (parallel mode
uses 2.5 px) leaves room for that approximation.

Typical use::

    from stereo.angled import AngledReconstructor

    rec = AngledReconstructor(rig)     # rig.K1/D1/K2/D2/R/T (and optional F)
    pt = rec.update(t, det_left, dets_right)
    if pt is not None:
        print(pt.xyz, pt.epipolar_px, pt.angle_deg, pt.residual_px)

Run ``python stereo/angled.py`` for a self-test (synthetic rig, no cameras).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if __package__:
    from .angled_geometry import (
        epipolar_line,
        fundamental_from_pose,
        point_line_distance,
        triangulate_pixels,
    )
else:  # `python stereo/angled.py`: no package context, run the self-test.
    _ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / "sphere_tracker").is_dir())
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from angled_geometry import (  # noqa: E402  (path bootstrapped just above)
        epipolar_line,
        fundamental_from_pose,
        point_line_distance,
        triangulate_pixels,
    )

if TYPE_CHECKING:
    from sphere_tracker.types import Detection

    from .triangulate import StereoPoint

__all__ = ["AngledMatchConfig", "AngledPoint", "AngledReconstructor"]

# Same "trust this measurement" threshold as ``StereoPoint.confident``.  A
# data object cannot reach the reconstructor's config, so this is a fixed
# heuristic; callers that need the configured gate can use ``rec.cfg``.
_CONFIDENT_RESIDUAL_PX = 2.0


@dataclass(frozen=True)
class AngledMatchConfig:
    """Thresholds for one frame of angled matching.

    Values exactly on a limit are accepted (rejection is strict: ``>`` for
    distances/errors, ``<`` for the angle).

    Attributes
    ----------
    max_epipolar_px:
        Maximum distance (pixels) from a right candidate to the epipolar line
        of the left detection.
    min_angle_deg:
        Minimum angle between the two viewing rays.  Small angles make depth
        ill-conditioned (far points / nearly parallel rays).
    max_reproj_px:
        Maximum of the **two** reprojection errors after triangulation (the
        larger one, not the mean).  Catches "on the epipolar line but at the
        wrong place" mismatches that a first-order epipolar distance cannot
        see.
    """

    max_epipolar_px: float = 8.0
    min_angle_deg: float = 5.0
    max_reproj_px: float = 3.0


@dataclass
class AngledPoint:
    """One reconstructed 3D point in the original left camera frame.

    Attributes
    ----------
    t:
        Timestamp (seconds), copied from the caller.
    xyz:
        ``(3,)`` metres, original left camera frame (not rectified).
    u_left / u_right:
        Original (distorted) pixel coordinates used for the triangulation.
    epipolar_px:
        Distance of ``u_right`` from the left point's epipolar line.
    residual_px:
        Mean of the left/right reprojection errors, the same convention as
        ``StereoPoint.residual_px``.  The acceptance gate uses the **max** of
        the two; rejected points are never stored.
    angle_deg:
        Angle between the two viewing rays (small = poor depth).
    radius_left:
        Left detection radius, for mono/stereo consistency checks.
    """

    t: float
    xyz: np.ndarray
    u_left: tuple[float, float]
    u_right: tuple[float, float]
    epipolar_px: float
    residual_px: float
    angle_deg: float
    radius_left: float = float("nan")

    @property
    def distance_m(self) -> float:
        """Distance from the left camera centre, in metres."""
        return float(np.linalg.norm(self.xyz))

    @property
    def confident(self) -> bool:
        """Whether the reprojection residual alone looks trustworthy.

        The configured thresholds are not reachable from a data object, so
        this mirrors ``StereoPoint.confident`` and only checks
        ``residual_px <= 2.0``.  ``angle_deg`` is kept for callers that want
        to apply their own angle policy.
        """
        return self.residual_px <= _CONFIDENT_RESIDUAL_PX

    def as_stereo_point(self) -> "StereoPoint":
        """Return the equivalent ``stereo.triangulate.StereoPoint``.

        This is an adapter for code that already consumes ``StereoPoint``
        (smoothed trajectories, export).  It is **lossy and frame-shifted**:

        * ``StereoPoint.xyz`` is documented in the *rectified* left camera
          frame, while this point is in the *original* left camera frame.
          Do not mix the two frames in one trajectory.
        * ``angle_deg`` has no field in ``StereoPoint`` and is dropped.

        ``StereoPoint`` is imported here, not at module import time, because
        ``stereo.triangulate`` pulls in ``calib`` + ``sphere_tracker`` (and so
        cv2), which this module's geometry path does not need.
        """
        if __package__:
            from .triangulate import StereoPoint
        else:  # `python stereo/angled.py` (self-test) bootstrapped the root.
            from stereo.triangulate import StereoPoint

        return StereoPoint(
            t=self.t,
            xyz=np.asarray(self.xyz, dtype=np.float64),
            u_left=self.u_left,
            u_right=self.u_right,
            epipolar_px=self.epipolar_px,
            residual_px=self.residual_px,
            radius_left=self.radius_left,
        )


def _as_fundamental(F: object) -> np.ndarray | None:
    """Return a usable 3x3 fundamental matrix, or ``None`` if unusable."""
    if F is None:
        return None
    try:
        arr = np.asarray(F, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.shape != (3, 3) or not bool(np.all(np.isfinite(arr))):
        return None
    try:
        # A fundamental matrix is rank 2 by definition.  Rank 0/1 cannot even
        # define a line for every left pixel (all-zero F gives a == b == 0),
        # and rank 3 is not a fundamental matrix at all.  A real calibrated F
        # passes this easily (measured sigma3/sigma1 ~ 1e-18); rebuilding from
        # the pose is safer than using a malformed matrix.
        if np.linalg.matrix_rank(arr) != 2:
            return None
    except np.linalg.LinAlgError:
        return None
    return arr


def _epipolar_line_or_none(
    F: np.ndarray, pixel: tuple[float, float]
) -> np.ndarray | None:
    """Return the epipolar line, or ``None`` for an unusable pixel."""
    try:
        return epipolar_line(F, pixel)
    except (TypeError, ValueError):
        return None


def _epipolar_distance_or_none(
    line: np.ndarray, pixel: tuple[float, float]
) -> float | None:
    """Return the pixel distance to ``line``, or ``None`` when unusable."""
    try:
        return point_line_distance(line, pixel)
    except (TypeError, ValueError):
        return None


def _resolve_fundamental(rig: object) -> tuple[np.ndarray, str]:
    """Use ``rig.F`` when it is valid, otherwise rebuild it from the pose.

    The rig object is never mutated (``rig.F`` is left exactly as supplied);
    the matrix actually used is returned together with its provenance
    (``"rig"`` or ``"pose"``).
    """
    given = _as_fundamental(getattr(rig, "F", None))
    if given is not None:
        return given, "rig"

    try:
        computed = fundamental_from_pose(
            rig.K1, rig.D1, rig.K2, rig.D2, rig.R, rig.T,
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "rig.F 不可用，且 K1/K2 奇异，无法由 K/D/R/T 构造 F"
        ) from exc
    resolved = _as_fundamental(computed)
    if resolved is None:
        raise ValueError(
            "rig.F 不可用，且由 K1/D1/K2/D2/R/T 构造出的 F 不是有效的 rank-2 矩阵"
            "（检查 T 是否为零、K/R 是否有效）"
        )
    return resolved, "pose"


class AngledReconstructor:
    """Match left/right detections with ``F`` and triangulate raw pixels.

    Parameters
    ----------
    rig:
        Anything exposing ``K1, D1, K2, D2, R, T`` (e.g. ``calib.StereoRig``)
        and optionally ``F``.  A valid ``rig.F`` is used as-is; otherwise
        ``F`` is computed from the pose once and cached in :attr:`F`.
    cfg:
        Thresholds; defaults to ``AngledMatchConfig()``.

    Attributes
    ----------
    F:
        The 3x3 fundamental matrix actually used (left pixels -> right
        epipolar lines).
    f_source:
        ``"rig"`` when ``rig.F`` was used, ``"pose"`` when :attr:`F` was
        derived from ``K/D/R/T``.  Useful when a calibration file and the
        pose disagree.
    trajectory:
        Accepted :class:`AngledPoint` values, in call order.
    n_matched / n_no_left / n_no_right / n_rejected:
        Mutually exclusive per-call counters (see the module docstring).
    """

    def __init__(self, rig: object, cfg: AngledMatchConfig | None = None) -> None:
        if rig is None:
            raise ValueError(
                "AngledReconstructor 需要一个 rig（含 K1/D1/K2/D2/R/T）"
            )
        self.rig = rig
        self.cfg = cfg if cfg is not None else AngledMatchConfig()
        self.F, self.f_source = _resolve_fundamental(rig)

        self.trajectory: list[AngledPoint] = []
        self.n_matched = 0
        self.n_no_left = 0
        self.n_no_right = 0
        self.n_rejected = 0

    @property
    def n_frames(self) -> int:
        """Number of ``update`` calls (= sum of the four counters)."""
        return self.n_matched + self.n_no_left + self.n_no_right + self.n_rejected

    @property
    def match_rate(self) -> float:
        """Fraction of ``update`` calls that produced a 3D point."""
        frames = self.n_frames
        return self.n_matched / frames if frames else 0.0

    def update(
        self,
        t: float,
        det_left: Detection | None,
        right_dets: list[Detection],
    ) -> AngledPoint | None:
        """Consume one frame and return the reconstructed point, or ``None``.

        Exactly one counter is incremented per call.  ``det_left is None``
        means the left camera saw nothing; an empty ``right_dets`` means the
        right camera had no candidates.  Every other failure (degenerate
        epipolar line, candidates outside the epipolar gate, triangulation
        returning ``None``, too-small ray angle, too-large reprojection error)
        increments ``n_rejected``.
        """
        if det_left is None:
            self.n_no_left += 1
            return None
        if not right_dets:
            self.n_no_right += 1
            return None

        line = _epipolar_line_or_none(self.F, (det_left.x, det_left.y))
        if line is None:
            # A malformed left pixel (e.g. NaN) gives no line at all.  Count
            # the frame instead of raising, so one bad detection cannot kill
            # a whole offline run.
            self.n_rejected += 1
            return None

        best: Detection | None = None
        best_epipolar_px = float("inf")
        for cand in right_dets:
            distance = _epipolar_distance_or_none(line, (cand.x, cand.y))
            if distance is None:
                # Degenerate line (left pixel at the epipole) or a non-finite
                # candidate: it cannot be matched, skip it.  If every
                # candidate is like this, the frame ends up in ``n_rejected``.
                continue
            if distance < best_epipolar_px:
                best, best_epipolar_px = cand, distance

        if best is None:
            self.n_rejected += 1
            return None
        if best_epipolar_px > self.cfg.max_epipolar_px:
            self.n_rejected += 1
            return None

        p_left = (float(det_left.x), float(det_left.y))
        p_right = (float(best.x), float(best.y))
        tri = triangulate_pixels(
            self.rig.K1, self.rig.D1, self.rig.K2, self.rig.D2,
            self.rig.R, self.rig.T, p_left, p_right,
        )
        if tri is None:
            self.n_rejected += 1
            return None

        reproj_max = max(tri.reproj_left_px, tri.reproj_right_px)
        if (
            tri.angle_deg < self.cfg.min_angle_deg
            or reproj_max > self.cfg.max_reproj_px
        ):
            self.n_rejected += 1
            return None

        point = AngledPoint(
            t=float(t),
            xyz=np.asarray(tri.xyz, dtype=np.float64),
            u_left=p_left,
            u_right=p_right,
            epipolar_px=float(best_epipolar_px),
            residual_px=0.5 * (tri.reproj_left_px + tri.reproj_right_px),
            angle_deg=float(tri.angle_deg),
            radius_left=float(getattr(det_left, "radius", float("nan"))),
        )
        self.trajectory.append(point)
        self.n_matched += 1
        return point

    def low_match_diagnostic(self, low_rate: float = 0.10) -> list[str]:
        """配对率低于 ``low_rate`` 时给出几何一致性诊断，否则返回空列表。

        与 ``StereoReconstructor.low_match_diagnostic`` **同名同形**，因为
        ``run_stereo_video`` 对两种重建器都无条件调用它（2026-11-04 之前
        angled 没有这个方法，跑完直接 AttributeError）。
        返回几行"该看什么"，不做自动放宽 —— 拒了多少就是多少，
        那是"这段录像和它的标定对不上"的结论，不是待修的 bug。
        """
        if self.n_frames == 0 or self.match_rate >= low_rate:
            return []
        lines = [
            f"[诊断] 配对率 {self.match_rate * 100:.1f}%"
            f"（{self.n_matched}/{self.n_frames}）："
            f"无左 {self.n_no_left} / 无右 {self.n_no_right} / 被拒 {self.n_rejected}",
            f"  本质几何来自 {self.f_source}（F 的来源决定拒收口径）",
        ]
        if self.n_rejected and self.n_matched == 0:
            lines.append(
                "  拒绝集中在极线/交会角/重投影：录制时的实际几何与标定不一致"
                "（相机被碰动/改对焦/换装置），先重新标定，别调 --max-epipolar")
        elif self.n_rejected:
            lines.append(
                f"  {self.n_rejected} 帧被极线/交会角/重投影拒掉；"
                f"先看标定自报的极线误差，与这里对不上就是几何漂了")
        return lines

    def as_array(self) -> np.ndarray:
        """``(N, 4)`` float64 ``[t, X, Y, Z]``, same layout as StereoReconstructor."""
        if not self.trajectory:
            return np.zeros((0, 4), dtype=np.float64)
        return np.array(
            [[p.t, *p.xyz] for p in self.trajectory], dtype=np.float64
        )

    def summary(self) -> str:
        """One-line human-readable status (never contains a newline)."""
        frames = self.n_frames
        if frames == 0:
            return "大角度重建：还没有处理任何帧"
        head = (
            f"大角度配对率 {self.match_rate * 100:.1f}%  "
            f"({self.n_matched}/{frames})"
        )
        if not self.trajectory:
            return (
                f"{head}  左路缺 {self.n_no_left} 帧 / "
                f"右路无候选 {self.n_no_right} 帧 / 被拒 {self.n_rejected} 帧"
            )

        epipolar = np.array(
            [p.epipolar_px for p in self.trajectory], dtype=np.float64
        )
        residual = np.array(
            [p.residual_px for p in self.trajectory], dtype=np.float64
        )
        angle = np.array([p.angle_deg for p in self.trajectory], dtype=np.float64)
        depth = np.array([p.xyz[2] for p in self.trajectory], dtype=np.float64)
        return (
            f"{head}  "
            f"极线 中位 {np.median(epipolar):.2f} px / 最大 {epipolar.max():.2f} px  "
            f"交会角 {angle.min():.1f}–{angle.max():.1f}°  "
            f"重投影 中位 {np.median(residual):.3f} px  "
            f"深度 {depth.min():.2f}–{depth.max():.2f} m"
        )


# ---------------------------------------------------------------------------
# Self-test: `python stereo/angled.py` (no cameras, no calibration files).
# ---------------------------------------------------------------------------
def _require(condition: bool, message: str) -> None:
    """Raise ``AssertionError`` independently of ``python -O``."""
    if not condition:
        raise AssertionError(message)


def _self_test() -> None:
    """Synthetic end-to-end checks with a known 3D point cloud."""
    import cv2

    from sphere_tracker.types import Detection

    @dataclass
    class FakeRig:
        """Minimal stand-in for ``calib.StereoRig`` (no calibration needed)."""

        K1: np.ndarray
        D1: np.ndarray
        K2: np.ndarray
        D2: np.ndarray
        R: np.ndarray
        T: np.ndarray
        F: np.ndarray | None = None

    # "Zero" fake rig: K = I (unit focal, zero principal point) and D = 0
    # (no distortion).  A literally all-zero K is singular, so the identity is
    # the smallest pinhole matrix that still defines projection/undistortion.
    # R = +90 deg about Y, T = (0.4, 0, 0): the right camera centre is at
    # (0, 0, -0.4) in the left frame, so points need x < 0.4 to be seen by both.
    K = np.eye(3, dtype=np.float64)
    D = np.zeros(5, dtype=np.float64)
    R = cv2.Rodrigues(
        np.array([0.0, np.deg2rad(90.0), 0.0], dtype=np.float64)
    )[0]
    T = np.array([0.4, 0.0, 0.0], dtype=np.float64)
    F_expected = fundamental_from_pose(K, D, K, D, R, T)

    def project(xyz: np.ndarray) -> np.ndarray:
        """Project a point with the rig's K/D into original pixels."""
        image_points, _ = cv2.projectPoints(
            np.asarray(xyz, dtype=np.float64).reshape(1, 1, 3),
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            K,
            D,
        )
        return np.asarray(image_points, dtype=np.float64).reshape(2)

    ball_radius_px = 5.0

    def detection(pixel: np.ndarray, radius: float = ball_radius_px) -> Detection:
        return Detection(
            x=float(pixel[0]), y=float(pixel[1]), radius=float(radius)
        )

    def unit_normal(line: np.ndarray) -> np.ndarray:
        normal = np.asarray(line[:2], dtype=np.float64)
        unit_tol = 1e-12
        _require(
            abs(float(np.linalg.norm(normal)) - 1.0) < unit_tol,
            "epipolar line normal is not unit length",
        )
        return normal

    rig = FakeRig(K1=K, D1=D, K2=K, D2=D, R=R, T=T)

    default_cfg = AngledMatchConfig()
    _require(
        (default_cfg.max_epipolar_px, default_cfg.min_angle_deg,
         default_cfg.max_reproj_px) == (8.0, 5.0, 3.0),
        "AngledMatchConfig defaults changed",
    )
    print("PASS config: max_epipolar_px=8.0, min_angle_deg=5.0, max_reproj_px=3.0")

    # 1) Match + triangulate known points, with an off-line distractor first.
    truth = np.array(
        [
            [-0.20, 0.00, 0.30],
            [-0.10, 0.06, 0.35],
            [-0.24, -0.05, 0.25],
        ],
        dtype=np.float64,
    )
    rec = AngledReconstructor(rig)
    _require(rec.f_source == "pose", "missing rig.F should be rebuilt from pose")
    np.testing.assert_array_equal(rec.F, F_expected)
    _require(rec.as_array().shape == (0, 4), "empty as_array shape")
    _require("\n" not in rec.summary(), "empty summary must be one line")

    max_error = 0.0
    for index, xyz_true in enumerate(truth):
        pixel_left = project(xyz_true)
        pixel_right = project(R @ xyz_true + T)
        normal = unit_normal(epipolar_line(F_expected, pixel_left))
        distractor = pixel_right + 20.0 * normal
        point = rec.update(
            t=0.1 * index,
            det_left=detection(pixel_left),
            right_dets=[
                detection(distractor),
                detection(pixel_right, radius=4.5),
            ],
        )
        _require(point is not None, f"point {index} was not reconstructed")
        error = float(np.linalg.norm(point.xyz - xyz_true))
        max_error = max(max_error, error)
        _require(error < 1e-6, f"3D error {error} m for point {index}")
        _require(
            point.u_right == (float(pixel_right[0]), float(pixel_right[1])),
            f"point {index} matched the wrong candidate",
        )
        _require(
            point.u_left == (float(pixel_left[0]), float(pixel_left[1])),
            f"u_left changed for point {index}",
        )
        _require(
            point.epipolar_px < 1e-9,
            f"epipolar distance {point.epipolar_px} px for point {index}",
        )
        _require(
            point.residual_px < 1e-6,
            f"reprojection residual {point.residual_px} px for point {index}",
        )
        _require(
            point.angle_deg > default_cfg.min_angle_deg,
            f"angle {point.angle_deg} deg for point {index}",
        )
        _require(point.confident, "synthetic point should be confident")
        _require(
            abs(point.distance_m - float(np.linalg.norm(xyz_true))) < 1e-9,
            f"distance_m wrong for point {index}",
        )
        _require(
            point.radius_left == ball_radius_px, "radius_left not copied"
        )
        _require(point.t == 0.1 * index, "t not copied")

    _require(rec.n_matched == 3, f"n_matched={rec.n_matched}, expected 3")
    _require(rec.n_no_left == 0 and rec.n_no_right == 0, "unexpected side errors")
    _require(rec.n_rejected == 0, f"n_rejected={rec.n_rejected}, expected 0")
    _require(rec.n_frames == 3, f"n_frames={rec.n_frames}, expected 3")
    _require(rec.match_rate == 1.0, "match_rate should be 1.0")
    array = rec.as_array()
    _require(array.shape == (3, 4), f"as_array shape {array.shape}")
    _require(array.dtype == np.float64, "as_array dtype")
    expected_array = np.column_stack(
        [0.1 * np.arange(3, dtype=np.float64), truth]
    )
    np.testing.assert_allclose(array, expected_array, rtol=0.0, atol=1e-9)
    _require("\n" not in rec.summary(), "summary must be one line")
    print(
        f"PASS match+triangulate: 3/3 points, max 3D error={max_error:.2e} m, "
        "counters clean"
    )

    # 2) Adapter to the parallel pipeline's data type.
    held = rec.trajectory[0]
    stereo = held.as_stereo_point()
    _require(type(stereo).__name__ == "StereoPoint", "as_stereo_point type")
    _require(
        stereo.t == held.t and stereo.epipolar_px == held.epipolar_px,
        "as_stereo_point scalars",
    )
    _require(
        stereo.u_left == held.u_left and stereo.u_right == held.u_right,
        "as_stereo_point pixels",
    )
    _require(
        stereo.residual_px == held.residual_px
        and stereo.radius_left == held.radius_left,
        "as_stereo_point residual/radius",
    )
    np.testing.assert_array_equal(stereo.xyz, held.xyz)
    print("PASS as_stereo_point: StereoPoint fields copied (angle_deg dropped)")

    # 3) `confident` only looks at the residual (the config is not reachable).
    from dataclasses import replace

    _require(replace(held, residual_px=2.0).confident, "boundary 2.0 should pass")
    _require(
        not replace(held, residual_px=2.0001).confident,
        "2.0001 px should not be confident",
    )
    _require(
        not replace(held, residual_px=float("nan")).confident,
        "NaN residual should not be confident",
    )
    print("PASS confident: residual <= 2.0 px, NaN is not confident")

    # 4) Missing left / missing right counters.
    pixel_left = project(truth[0])
    pixel_right = project(R @ truth[0] + T)
    det_left = detection(pixel_left)
    rec_missing = AngledReconstructor(rig)
    _require(
        rec_missing.update(0.0, None, [detection(pixel_right)]) is None,
        "missing left should return None",
    )
    _require(
        rec_missing.n_no_left == 1 and rec_missing.n_frames == 1,
        "n_no_left counter",
    )
    _require(
        rec_missing.update(0.1, det_left, []) is None,
        "missing right should return None",
    )
    _require(
        rec_missing.n_no_right == 1 and rec_missing.n_frames == 2,
        "n_no_right counter",
    )
    _require(rec_missing.match_rate == 0.0, "missing-side match_rate")
    _require("\n" not in rec_missing.summary(), "missing summary one line")
    print("PASS missing side: None -> n_no_left, [] -> n_no_right, match_rate=0")

    # 5) Epipolar gate: 20 px off the line > 8 px -> n_rejected.
    line = epipolar_line(F_expected, pixel_left)
    normal = unit_normal(line)
    off_line = pixel_right + 20.0 * normal
    off_line_px = point_line_distance(line, off_line)
    _require(
        off_line_px > default_cfg.max_epipolar_px,
        f"candidate distance {off_line_px} px should exceed the gate",
    )
    rec_epipolar = AngledReconstructor(rig)
    _require(
        rec_epipolar.update(0.0, det_left, [detection(off_line)]) is None,
        "epipolar reject should return None",
    )
    _require(
        rec_epipolar.n_rejected == 1 and rec_epipolar.n_matched == 0,
        "epipolar reject counter",
    )
    _require(rec_epipolar.n_frames == 1, "epipolar reject frame count")
    print(
        f"PASS epipolar reject: {off_line_px:.1f} px > "
        f"{default_cfg.max_epipolar_px:.1f} px -> n_rejected"
    )

    # 6) Angle gate: a 10 m point is on the epipolar line and reprojects well,
    # but the rays are nearly parallel (0.002 deg), so only the angle can fail.
    far_xyz = np.array([-0.01, 0.0, 10.0], dtype=np.float64)
    far_left = project(far_xyz)
    far_right = project(R @ far_xyz + T)
    far_line = epipolar_line(F_expected, far_left)
    far_tri = triangulate_pixels(K, D, K, D, R, T, far_left, far_right)
    _require(far_tri is not None, "far point should still triangulate")
    _require(
        far_tri.angle_deg < default_cfg.min_angle_deg,
        f"far angle {far_tri.angle_deg} deg should be below the gate",
    )
    _require(
        max(far_tri.reproj_left_px, far_tri.reproj_right_px)
        <= default_cfg.max_reproj_px,
        "far point reprojection should pass (angle must be the reject reason)",
    )
    _require(
        point_line_distance(far_line, far_right)
        <= default_cfg.max_epipolar_px,
        "far point epipolar distance should pass",
    )
    rec_angle = AngledReconstructor(rig)
    _require(
        rec_angle.update(0.0, detection(far_left), [detection(far_right)])
        is None,
        "angle reject should return None",
    )
    _require(rec_angle.n_rejected == 1, "angle reject counter")
    print(
        f"PASS angle reject: {far_tri.angle_deg:.4f} deg < "
        f"{default_cfg.min_angle_deg:.1f} deg -> n_rejected"
    )

    # 7) Reproj gate: 1 px off the line passes the epipolar gate, but the
    # triangulated midpoint reprojects 5.9 px away -> only reproj can fail.
    decoy = pixel_right + 1.0 * normal
    decoy_px = point_line_distance(line, decoy)
    decoy_tri = triangulate_pixels(K, D, K, D, R, T, pixel_left, decoy)
    _require(decoy_tri is not None, "decoy should still triangulate")
    decoy_reproj = max(decoy_tri.reproj_left_px, decoy_tri.reproj_right_px)
    _require(
        decoy_px <= default_cfg.max_epipolar_px,
        f"decoy epipolar {decoy_px} px should pass",
    )
    _require(
        decoy_tri.angle_deg >= default_cfg.min_angle_deg,
        f"decoy angle {decoy_tri.angle_deg} deg should pass",
    )
    _require(
        decoy_reproj > default_cfg.max_reproj_px,
        f"decoy reprojection {decoy_reproj} px should fail",
    )
    rec_reproj = AngledReconstructor(rig)
    _require(
        rec_reproj.update(0.0, det_left, [detection(decoy)]) is None,
        "reproj reject should return None",
    )
    _require(
        rec_reproj.n_rejected == 1 and rec_reproj.n_matched == 0,
        "reproj reject counter",
    )
    print(
        f"PASS reproj reject: epipolar {decoy_px:.2f} px ok, "
        f"reproj {decoy_reproj:.2f} px > {default_cfg.max_reproj_px:.1f} px "
        "-> n_rejected"
    )

    # 8) F provenance: valid rig.F is used as-is; unusable ones are rebuilt.
    for bad_f in (
        None,
        np.zeros((3, 3), dtype=np.float64),
        np.eye(3, dtype=np.float64),
        np.full((3, 3), np.nan),
        np.zeros((2, 2), dtype=np.float64),
    ):
        rebuilt = AngledReconstructor(
            FakeRig(K1=K, D1=D, K2=K, D2=D, R=R, T=T, F=bad_f)
        )
        _require(rebuilt.f_source == "pose", "unusable rig.F should be rebuilt")
        np.testing.assert_array_equal(rebuilt.F, F_expected)
    scaled = 2.5 * F_expected
    kept = AngledReconstructor(
        FakeRig(K1=K, D1=D, K2=K, D2=D, R=R, T=T, F=scaled)
    )
    _require(kept.f_source == "rig", "valid rig.F should be used")
    np.testing.assert_array_equal(kept.F, scaled)
    _require(rig.F is None, "constructor must not mutate rig.F")
    print(
        "PASS F resolution: rank-2 rig.F kept; None/zeros/rank-3/NaN/"
        "wrong-shape rebuilt from pose"
    )

    print("PASS angled self-test")


if __name__ == "__main__":
    _self_test()
    raise SystemExit(0)
