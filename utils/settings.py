"""Project wide configuration dataclasses and default values."""

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

# Common file name extensions
IMAGE_EXT = "_rgb.png"
DEPTH_EXT = "_depth.npy"

# Default plotting behaviour
DEFAULT_INTERACTIVE = True

HAND_EYE_METHODS = [
    (cv2.CALIB_HAND_EYE_TSAI, "tsai"),
    (cv2.CALIB_HAND_EYE_PARK, "park"),
    (cv2.CALIB_HAND_EYE_HORAUD, "horaud"),
    (cv2.CALIB_HAND_EYE_ANDREFF, "andreff"),
    (cv2.CALIB_HAND_EYE_DANIILIDIS, "daniilidis"),
    ("svd", "svd"),
]

HAND_EYE_MAP = {name: method for method, name in HAND_EYE_METHODS}


@dataclass(frozen=True)
class Paths:
    """Convenient bundle of frequently used project paths."""

    CAPTURES_DIR: Path = BASE_DIR / "calib"
    RESULTS_DIR: Path = BASE_DIR / "calibration" / "results1"
    CLOUD_DIR: Path = BASE_DIR / "clouds"
    LOG_DIR: Path = BASE_DIR / "logs"
    CAMERA_INTR: Path = BASE_DIR / "data" / "results1980"


paths = Paths()


@dataclass(frozen=True)
class RobotCfg:
    ip: str = "192.168.58.2"
    tool_id: int = 0
    user_frame_id: int = 0
    velocity: float = 35.0
    emergency_delay: float = 0.5
    restart_delay: float = 0.5


robot = RobotCfg()


@dataclass(frozen=True)
class D415_Cfg:
    """Resolution and frame rate parameters."""

    rgb_width: int = 1920
    rgb_height: int = 1080
    depth_width: int = 1280
    depth_height: int = 720
    fps: int = 30
    depth_scale: float = 0.001
    align_to_color: bool = True


camera = D415_Cfg()


@dataclass(frozen=True)
class CheckerboardDefaults:
    size: tuple[int, int] = (7, 6)
    square_size: float = 0.02


checkerboard = CheckerboardDefaults()


@dataclass(frozen=True)
class CharucoDefaults:
    squares: tuple[int, int] = (8, 5)
    square_size: float = 0.035
    marker_size: float = 0.026
    dictionary: int = cv2.aruco.DICT_5X5_100


charuco = CharucoDefaults()


@dataclass(frozen=True)
class ArucoDefaults:
    marker_length: float = 0.5
    dictionary: int = cv2.aruco.DICT_5X5_100


aruco = ArucoDefaults()


@dataclass(frozen=True)
class HandEyeCfg:
    square_numbers: tuple[int, int] = (5, 8)
    square_length: float = 0.035
    marker_length: float = 0.026
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
    robot_poses_file: str = str(paths.CAPTURES_DIR / "poses.json")
    images_dir: str = str(paths.CAPTURES_DIR)
    charuco_xml: str = str(paths.CAMERA_INTR / "charuco_cam.xml")
    charuco_txt: str = str(paths.CAMERA_INTR / "charuco_cam.txt")
    calib_output_dir: str = str(paths.RESULTS_DIR)


handeye = HandEyeCfg()


@dataclass(frozen=True)
class LoggingCfg:
    level: str = "INFO"
    json: bool = True


logging = LoggingCfg()


@dataclass(frozen=True)
class GridCalibCfg:
    """Workspace sampling parameters for explicit hand-eye calibration."""

    calibration_type: str = "EYE_IN_HAND"
    workspace_limits: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float]
    ] = (
        (-250.0, -240.0),
        (-50.0, -40.0),
        (350.0, 360.0),
    )
    grid_step: float = 0.1
    reference_point_offset: tuple[float, float, float, float] = (
        0.7,
        0.0,
        0.05,
        1.0,
    )
    tool_orientation: tuple[float, float, float] = (180.0, 0.0, 0.0)
    charuco_xml: str = str(paths.CAMERA_INTR / "charuco_cam.xml")
    calib_output_dir: str = str(paths.RESULTS_DIR)


grid_calib = GridCalibCfg()

DEPTH_SCALE = 0.001

# Default depth camera intrinsics
DEFAULT_DEPTH_INTRINSICS = np.array(
    [[616.365, 0.0, 318.268], [0.0, 616.202, 243.215], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)


@dataclass(frozen=True)
class CloudCfg:
    roi_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (-0.6, -0.1),
            "y": (-0.2, 0.1),
            "z": (-0.009, 0.07),
        }
    )


cloud = CloudCfg()
