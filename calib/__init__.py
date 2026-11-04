"""双目相机标定：自动挑帧 -> 标定 -> 稳健化 -> 质量验证 -> 保存。"""

from .board import (BoardObservation, BoardSpec, detect_board, detect_two_boards,
                    pose_distance)
from .capture import AutoPairCollector, CaptureConfig, CapturedPair
from .export import build_camera, build_export, build_rig, write_export
from .params import StereoRig, load_stereo_rig, save_stereo_rig
from .world_frame import WorldFrame
from .angled_quality import AngledQuality, evaluate_angled, grade_angled
from .mono_calib import (MonoCalib, MonoCalibError, calibrate_mono, load_mono,
                         save_mono)
from .stereo_calib import CalibConfig, StereoCalibrationResult, calibrate_stereo

__all__ = [
    "BoardSpec",
    "BoardObservation",
    "detect_board",
    "detect_two_boards",
    "pose_distance",
    "AutoPairCollector",
    "CaptureConfig",
    "CapturedPair",
    "CalibConfig",
    "StereoCalibrationResult",
    "calibrate_stereo",
    "StereoRig",
    "WorldFrame",
    "load_stereo_rig",
    "save_stereo_rig",
    "AngledQuality",
    "evaluate_angled",
    "grade_angled",
    "build_camera",
    "build_rig",
    "build_export",
    "write_export",
    # 单目内参（2026-10-28 加）：放在这里才算"对外契约"，
    # 否则它只是 calib 目录里一个谁都不知道存在的文件（B-100 的原病）。
    "MonoCalib",
    "MonoCalibError",
    "calibrate_mono",
    "load_mono",
    "save_mono",
]
