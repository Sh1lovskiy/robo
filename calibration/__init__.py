"""Robotics calibration package."""

from .handeye import (
    calibrate_opencv,
    calibrate_svd,
    calibrate_svd_points,
    HandEyeResult,
)
from .detector import (
    CheckerboardConfig,
    CharucoBoardConfig,
    Detection,
    pose_from_detection,
    detect_charuco,
    find_checkerboard,
)
from .extractor import load_depth, board_center_from_depth
from geometry import (
    load_extrinsics,
    map_rgb_corners_to_depth,
    estimate_board_points_3d,
)
from .runner import run_handeye

__all__ = [
    "HandEyeResult",
    "calibrate_opencv",
    "calibrate_svd",
    "calibrate_svd_points",
    "CheckerboardConfig",
    "CharucoBoardConfig",
    "Detection",
    "pose_from_detection",
    "detect_charuco",
    "find_checkerboard",
    "load_depth",
    "board_center_from_depth",
    "load_extrinsics",
    "map_rgb_corners_to_depth",
    "estimate_board_points_3d",
    "run_handeye",
]
