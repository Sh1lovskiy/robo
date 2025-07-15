"""Robotics calibration package."""

from .detector import (
    CheckerboardConfig,
    CharucoBoardConfig,
    ArucoBoardConfig,
    Detection,
    pose_from_detection,
    detect_charuco,
    find_checkerboard,
    find_aruco,
    draw_markers,
)
from .pattern import ArucoPattern
from .handeye import (
    calibrate_opencv,
    calibrate_svd,
    calibrate_svd_points,
    HandEyeResult,
)
from utils.cloud_utils import load_depth
from utils.geometry import (
    load_extrinsics,
    map_rgb_corners_to_depth,
    estimate_board_points_3d,
    board_center_from_depth,
)


__all__ = [
    "CharucoBoardConfig",
    "ArucoBoardConfig",
    "ArucoPattern",
    "find_aruco",
    "draw_markers",
    "HandEyeResult",
    "calibrate_opencv",
    "calibrate_svd",
    "calibrate_svd_points",
    "CheckerboardConfig",
    "Detection",
    "pose_from_detection",
    "detect_charuco",
    "find_checkerboard",
    "load_depth",
    "board_center_from_depth",
    "load_extrinsics",
    "map_rgb_corners_to_depth",
    "estimate_board_points_3d",
]
