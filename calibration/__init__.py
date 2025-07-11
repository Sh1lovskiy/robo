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
from .extractor import load_depth, board_center_from_depth, board_points_from_depth
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
    "board_points_from_depth",
    "run_handeye",
]
