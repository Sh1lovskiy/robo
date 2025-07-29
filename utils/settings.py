"""Project wide configuration dataclasses and default values."""

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Root dir
BASE_DIR = Path(__file__).resolve().parent.parent

# Common file name extensions for project data (frames saving from RealSense2 cam)
IMAGE_EXT = ".png"
DEPTH_EXT = ".npy"

# Global flag: use interactive plotting by default
DEFAULT_INTERACTIVE = True

# List of supported hand-eye calibration methods.
HAND_EYE_METHODS = [
    (cv2.CALIB_HAND_EYE_TSAI, "tsai"),
    (cv2.CALIB_HAND_EYE_PARK, "park"),
    (cv2.CALIB_HAND_EYE_HORAUD, "horaud"),
    (cv2.CALIB_HAND_EYE_ANDREFF, "andreff"),
    (cv2.CALIB_HAND_EYE_DANIILIDIS, "daniilidis"),
    ("svd", "svd"),
]

# Dictionary: string name → OpenCV constant identifier
HAND_EYE_MAP = {name: method for method, name in HAND_EYE_METHODS}


@dataclass(frozen=True)
class Paths:
    """
    Dataclass aggregating all important filesystem paths used in the project.
    These paths are used for organizing captures, results, logs, etc.
    """

    CAPTURES_EXTR_DIR: Path = BASE_DIR / "calib"
    CAPTURES_DIR: Path = CAPTURES_EXTR_DIR / "imgs"
    # RESULTS_DIR: Path = BASE_DIR / "calibration" / "results1"
    RESULTS_DIR: Path = CAPTURES_EXTR_DIR / "calib_res"
    VIZ_DIR: Path = CAPTURES_EXTR_DIR / "calib_viz"
    CLOUD_DIR: Path = BASE_DIR / ".clouds"
    LOG_DIR: Path = BASE_DIR / ".logs"
    CAMERA_INTR: Path = BASE_DIR / ".data" / "results1980"


paths = Paths()


@dataclass(frozen=True)
class LoggingCfg:
    """
    Logging configuration for the project.

    - level: Log level ("INFO", "DEBUG", etc.)
    - json: Enable/disable structured JSON logging.
    - log_dir: Directory where log files are stored.
    - log_format: Console log output format.
    - log_file_format: File log output format.
    - progress_bar_format: TQDM progress bar format.
    """

    level: str = "INFO"
    json: bool = True
    log_dir: Path = Path(".logs")
    log_format: str = (
        "<green>{time:MM-DD HH:mm:ss}</green>"
        "[<level>{level:.3}</level>]"
        "[<cyan>{extra[module]:.16}</cyan>:<cyan>{line:<3}</cyan>]"
        "<level>{message}</level>"
    )
    log_file_format: str = "{time:YYYY-MM-DD HH:mm:ss}[{level}][{file}:{line}]{message}"
    progress_bar_format: str = (
        "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    )


logging = LoggingCfg()


@dataclass(frozen=True)
class RobotCfg:
    """
    Robot connection and operation parameters.
    Includes IP, tool and frame identifiers, speed, and safety timeouts.
    """

    ip: str = "192.168.58.2"
    tool_id: int = 0
    user_frame_id: int = 0
    velocity: float = 35.0  # % of max speed: [0, 100]
    restart_delay: float = 0.5  # sec


robot = RobotCfg()


@dataclass(frozen=True)
class ArucoDefaults:
    """
    Default parameters for ArUco marker grid boards.
    """

    marker_length: float = 0.05  # Marker side in meters
    dictionary: int = cv2.aruco.DICT_5X5_100


aruco = ArucoDefaults()


@dataclass(frozen=True)
class CharucoDefaults:
    """
    Default parameters for Charuco calibration boards (hybrid ArUco+checkerboard).
    - squares: (columns, rows)
    - square_size: length of each square (meters)
    - marker_size: side length of ArUco markers (meters)
    - dictionary: OpenCV dictionary constant
    """

    squares: tuple[int, int] = (9, 6)
    square_size: float = 0.031
    marker_size: float = 0.023
    dictionary: int = cv2.aruco.DICT_5X5_100


charuco = CharucoDefaults()


# TODO mb add CharucoDefaults class and ArucoDefaults, clarify refrences
@dataclass(frozen=True)
class HandEyeCfg:
    """
    Hand-eye calibration configuration parameters.
    Includes Charuco board parameters, allowed outliers,
    min corners, input/output paths, etc.
    """

    square_numbers: tuple[int, int] = (9, 6)
    square_length: float = 0.031
    marker_length: float = 0.023
    CHARUCO_DICT_MAP = {
        "5X5_50": cv2.aruco.DICT_5X5_50,
        "5X5_100": cv2.aruco.DICT_5X5_100,
    }
    aruco_dict: str = "5X5_100"
    min_corners: int = 4
    outlier_std: float = 2.0
    method: str = "ALL"
    analyze_corners: bool = False
    visualize: bool = False
    robot_poses_file = paths.CAPTURES_EXTR_DIR.glob("*.json")  # 'calib/*.json'
    images_dir: str = str(paths.CAPTURES_DIR)
    charuco_xml: str = str(paths.CAMERA_INTR / "charuco_cam.xml")
    charuco_txt: str = str(paths.CAMERA_INTR / "charuco_cam.txt")
    calib_output_dir: str = str(paths.RESULTS_DIR)


handeye = HandEyeCfg()


@dataclass(frozen=True)
class GridCalibCfg:
    """
    Grid-based workspace sampling for hand-eye calibration.
    Defines the limits, grid step, orientation, and output.
    """

    calibration_type: str = "EYE_IN_HAND"
    workspace_limits: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float]
    ] = (
        (-70.0, 50.0),  # X, mm
        (-250.0, -130.0),  # Y, mm
        (300.0, 400.0),  # Z, mm
    )
    grid_step: float = 40.0
    tool_orientation: tuple[float, float, float] = (180.0, 0.0, 180.0)
    charuco_xml: str = str(paths.CAMERA_INTR / "charuco_cam.xml")
    calib_output_dir: str = str(paths.RESULTS_DIR)


grid_calib = GridCalibCfg()


@dataclass(frozen=True)
class CameraIntrinsics:
    """
    Intrinsic camera parameters for pinhole/radial models.
    """

    width: int
    height: int
    ppx: float  # principal point X (cx)
    ppy: float  # principal point Y (cy)
    fx: float  # focal length X
    fy: float  # focal length Y
    model: str  # distortion model name
    coeffs: tuple[float, float, float, float, float]  # distortion coeffs


@dataclass(frozen=True)
class CameraExtrinsics:
    """
    Extrinsic calibration parameters (rotation and translation)
    Rotation: 3x3 matrix, translation: 3-vector (usually meters).
    """

    rotation: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    translation: tuple[float, float, float]


# Default RealSense depth scale (meters per unit)
DEPTH_SCALE = 0.00010000000474974513


@dataclass(frozen=True)
class D415_Cfg:
    """
    Intel RealSense D415 camera configuration:
    - frame size (color/depth)
    - frame rate
    - depth scale
    - alignment mode
    """

    rgb_width: int = 1920
    rgb_height: int = 1080
    # rgb_width: int = 1280
    # rgb_height: int = 720
    depth_width: int = 1280
    depth_height: int = 720
    fps: int = 30
    # TODO clarify depth_scale import at all python pakages
    # and del DEPTH_SCALE anywhere
    depth_scale: float = DEPTH_SCALE
    align_to_color: bool = True


# RealSense D415 configuration
camera = D415_Cfg()


# Intrinsics for depth stream
INTRINSICS_DEPTH = CameraIntrinsics(
    width=1280,
    height=720,
    ppx=632.6915893554688,
    ppy=383.97265625,
    fx=900.0674438476562,
    fy=900.0674438476562,
    model="distortion.brown_conrady",
    coeffs=(0.0, 0.0, 0.0, 0.0, 0.0),
)

# Intrinsics for color stream
INTRINSICS_COLOR = CameraIntrinsics(
    width=1280,
    height=720,
    fx=922.1192656114683,
    fy=926.5733034002656,
    ppx=635.5098688405669,
    ppy=339.22998506754635,
    model="distortion.inverse_brown_conrady",
    coeffs=(
        0.14276977476919803,
        -0.37253190781644513,
        -0.002400175306122351,
        0.005829250084910286,
        0.23582308984644337,
    ),
)

# 3x3 Camera intrinsic matrices for depth and color streams
# TODO think about the format
INTRINSICS_DEPTH_MATRIX = np.array(
    [
        [INTRINSICS_DEPTH.fx, 0.0, INTRINSICS_DEPTH.ppx],
        [0.0, INTRINSICS_DEPTH.fy, INTRINSICS_DEPTH.ppy],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

INTRINSICS_COLOR_MATRIX = np.array(
    [
        [INTRINSICS_COLOR.fx, 0.0, INTRINSICS_COLOR.ppx],
        [0.0, INTRINSICS_COLOR.fy, INTRINSICS_COLOR.ppy],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# R|t for depth-to-color and color-to-depth (extrinsics)
EXTR_DEPTH_TO_COLOR_ROT = (
    (0.9999984502792358, 0.0017436681082472205, -3.4469805541448295e-05),
    (-0.001743800356052816, 0.9999892115592957, -0.004307812545448542),
    (2.6958035959978588e-05, 0.004307866096496582, 0.999990701675415),
)
EXTR_DEPTH_TO_COLOR_TRANS = (
    0.015152666717767715,
    0.00015625852392986417,
    -0.0004899608320556581,
)

EXTR_COLOR_TO_DEPTH_ROT = (
    (0.9999984502792358, -0.001743800356052816, 2.6958035959978588e-05),
    (0.0017436681082472205, 0.9999892115592957, 0.004307866096496582),
    (-3.4469805541448295e-05, -0.004307812545448542, 0.999990701675415),
)
EXTR_COLOR_TO_DEPTH_TRANS = (
    -0.015152933076024055,
    -0.00013194428174756467,
    0.0004888746771030128,
)


@dataclass(frozen=True)
class CloudCfg:
    """
    Configuration for point cloud region-of-interest cropping.
    Each axis can have (min, max) limits in meters.
    """

    roi_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (-0.6, -0.1),
            "y": (-0.2, 0.1),
            "z": (-0.009, 0.07),
        }
    )


cloud = CloudCfg()

__all__ = [
    "Paths",
    "LoggingCfg",
    "RobotCfg",
    "ArucoDefaults",
    "CharucoDefaults",
    "HandEyeCfg",
    "GridCalibCfg",
    "D415_Cfg",
    "CameraIntrinsics",
    "CameraExtrinsics",
    "DEPTH_SCALE",
    "paths",
    "logging",
    "robot",
    "aruco",
    "charuco",
    "handeye",
    "grid_calib",
    "camera",
    "cloud",
    "IMAGE_EXT",
    "DEPTH_EXT",
    "INTRINSICS_DEPTH_MATRIX",
    "INTRINSICS_COLOR_MATRIX",
    "EXTR_DEPTH_TO_COLOR_ROT",
    "EXTR_DEPTH_TO_COLOR_TRANS",
    "EXTR_COLOR_TO_DEPTH_ROT",
    "EXTR_COLOR_TO_DEPTH_TRANS",
]
