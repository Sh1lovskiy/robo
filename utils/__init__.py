"""Shared helper modules used across the project.

The :mod:`utils` package contains lightweight helpers for logging, CLI
dispatching and simple file I/O. These utilities are used by most other
packages and avoid additional dependencies.
"""

from .logger import Logger, LoggerType
from .settings import (
    DEPTH_SCALE,
    D415_Cfg,
    GridCalibCfg,
    HAND_EYE_METHODS,
    HAND_EYE_MAP,
    HandEyeCfg,
    IMAGE_EXT,
    DEPTH_EXT,
    DEFAULT_INTERACTIVE,
    paths,
    robot,
    camera,
    handeye,
    charuco,
    checkerboard,
    aruco,
    logging,
    grid_calib,
    cloud,
)
from .io import (
    JSONPoseLoader,
    read_image,
    write_image,
    load_json,
    save_json,
    load_npy,
    save_npy,
    load_camera_params,
    save_camera_params_xml,
    save_camera_params_txt,
)
from .transform import TransformUtils
from .math_utils import (
    euler_to_matrix,
    make_transform,
    decompose_transform,
    invert_transform,
)

__all__ = [
    "DEPTH_SCALE",
    "Logger",
    "LoggerType",
    "D415_Cfg",
    "GridCalibCfg",
    "HAND_EYE_METHODS",
    "HAND_EYE_MAP",
    "HandEyeCfg",
    "IMAGE_EXT",
    "DEPTH_EXT",
    "DEFAULT_INTERACTIVE",
    "JSONPoseLoader",
    "paths",
    "robot",
    "camera",
    "handeye",
    "charuco",
    "checkerboard",
    "aruco",
    "logging",
    "grid_calib",
    "cloud",
    "read_image",
    "write_image",
    "load_json",
    "save_json",
    "load_npy",
    "save_npy",
    "load_camera_params",
    "save_camera_params_xml",
    "save_camera_params_txt",
    "TransformUtils",
    "euler_to_matrix",
    "make_transform",
    "decompose_transform",
    "invert_transform",
]
