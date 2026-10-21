"""双目相机标定：自动挑帧 -> 标定 -> 稳健化 -> 质量验证 -> 保存。"""

from .board import BoardObservation, BoardSpec, detect_board, pose_distance
from .capture import AutoPairCollector, CaptureConfig, CapturedPair
from .export import build_camera, build_export, build_rig, write_export
from .params import StereoRig, load_stereo_rig, save_stereo_rig
from .angled_quality import AngledQuality, evaluate_angled, grade_angled
from .stereo_calib import CalibConfig, StereoCalibrationResult, calibrate_stereo

__all__ = [
    "BoardSpec",
    "BoardObservation",
    "detect_board",
    "pose_distance",
    "AutoPairCollector",
    "CaptureConfig",
    "CapturedPair",
    "CalibConfig",
    "StereoCalibrationResult",
    "calibrate_stereo",
    "StereoRig",
    "load_stereo_rig",
    "save_stereo_rig",
    "AngledQuality",
    "evaluate_angled",
    "grade_angled",
    "build_camera",
    "build_rig",
    "build_export",
    "write_export",
]
