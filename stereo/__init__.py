"""双目：同步、成对取帧、极线配对、三角测量。

本包里"能算的"都已经能跑了（同步、配对、三角测量），
"还没接"的是把它们串进实时主流程 —— 见 docs/计划书.md 的 M3。

    sync.py         双路视频同步（闪光/拍手对齐），算出帧偏移
    pair_source.py  成对帧源：按 sync_offset 取到"同一时刻"的左右两帧
    triangulate.py  极线约束配对 + 三角测量 + 三维轨迹
    visualization.py 标准数据：seeds / traj3d.json / visualization.json
    sessions.py     采集会话：session.json 的读写与校验（data/ 的契约）

分工说明：
    calib/    负责"标定出参数"（离线、一次）
    sphere_tracker/  负责"单目 2D 检测与跟踪"（逐帧）
    stereo/   负责"把两路合成三维"（逐帧，依赖上面两者）
"""

from __future__ import annotations

from .pair_source import PairSource, from_args
from .sessions import (
    CameraInfo,
    CalibrationInfo,
    Session,
    SessionError,
    SyncInfo,
    create_session,
    list_sessions,
    load_session,
    save_session,
    sha256_file,
    validate_session,
)
from .sync import (
    SyncConfig,
    SyncResult,
    TimestampSync,
    apply_offset,
    brightness_jumps,
    brightness_series,
    find_offset,
    timestamp_offset,
)
from .timestamp_pair import (
    TimestampPairPlan,
    canary_safe,
    plan_timestamp_pairs,
    read_time_sidecar,
    sidecar_for_video,
)
from .trajectory import Kalman3D, savgol_trajectory
from .depth_fusion import fuse_depth, mono_depth, stereo_depth
from .triangulate import (
    MatchConfig,
    StereoPoint,
    StereoReconstructor,
    match_epipolar,
    triangulate_one,
)
from .angled import AngledMatchConfig, AngledPoint, AngledReconstructor
from .angled_geometry import AngledTriangulation, triangulate_pixels
from .stereo_mode import (
    MODES,
    geometry_advice,
    infer_mode,
    mode_label,
    normalize_mode,
    optical_axes_angle_deg,
)
from .visualization import (
    build_trajectory_payload,
    build_visualization_payload,
    export_session_visualization,
    load_seeds,
    read_trajectory_csv,
    write_seeds,
)

__all__ = [
    "PairSource",
    "from_args",
    "SyncConfig",
    "SyncResult",
    "find_offset",
    "brightness_series",
    "brightness_jumps",
    "timestamp_offset",
    "TimestampSync",
    "TimestampPairPlan",
    "plan_timestamp_pairs",
    "canary_safe",
    "read_time_sidecar",
    "sidecar_for_video",
    "apply_offset",
    "Kalman3D",
    "savgol_trajectory",
    "fuse_depth",
    "mono_depth",
    "stereo_depth",
    "MatchConfig",
    "StereoPoint",
    "StereoReconstructor",
    "match_epipolar",
    "triangulate_one",
    "AngledMatchConfig",
    "AngledPoint",
    "AngledReconstructor",
    "AngledTriangulation",
    "triangulate_pixels",
    "MODES",
    "infer_mode",
    "normalize_mode",
    "mode_label",
    "optical_axes_angle_deg",
    "geometry_advice",
    "write_seeds",
    "load_seeds",
    "read_trajectory_csv",
    "build_trajectory_payload",
    "build_visualization_payload",
    "export_session_visualization",
    "CameraInfo",
    "CalibrationInfo",
    "Session",
    "SessionError",
    "SyncInfo",
    "create_session",
    "save_session",
    "load_session",
    "validate_session",
    "list_sessions",
    "sha256_file",
]
