"""Robotics calibration package."""

from .aruco import (
    ArucoBoardConfig,
    ArucoPattern,
    detect_markers,
    draw_markers,
)
from .charuco import (
    CharucoBoardConfig,
    detect_charuco_corners,
    draw_corners,
)
from .detector import (
    CheckerboardConfig,
    CharucoBoardConfig,
    Detection,
    pose_from_detection,
    detect_charuco,
    find_checkerboard,
)
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
    "detect_charuco_corners",
    "detect_markers",
    "draw_corners",
    "draw_markers",
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
]
