"""Robotics calibration package."""

from .detector import (
    CheckerboardConfig,
    CharucoBoardConfig,
    ArucoBoardConfig,
    Detection,
    pose_from_detection,
    detect_charuco,
    find_aruco,
    draw_markers,
)
from .pattern import ArucoPattern
from .comparison import HandEyeComparison
from utils.cloud_utils import load_depth
from utils.geometry import (
    load_extrinsics,
)


__all__ = [
    "CharucoBoardConfig",
    "ArucoBoardConfig",
    "ArucoPattern",
    "find_aruco",
    "draw_markers",
    "HandEyeComparison",
    "CheckerboardConfig",
    "Detection",
    "pose_from_detection",
    "detect_charuco",
    "load_depth",
    "load_extrinsics",
]
