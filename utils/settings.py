"""Project wide configuration dataclasses and default values."""

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

# Common file name extensions
IMAGE_EXT = ".png"
DEPTH_EXT = ".npy"

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

    CAPTURES_EXTR_DIR: Path = BASE_DIR / "calib"
    CAPTURES_DIR: Path = CAPTURES_EXTR_DIR / "imgs"
    # RESULTS_DIR: Path = BASE_DIR / "calibration" / "results1"
    RESULTS_DIR: Path = CAPTURES_EXTR_DIR / "calib_res"
    VIZ_DIR: Path = CAPTURES_EXTR_DIR / "calib_viz"
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
    marker_length: float = 0.05
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
    robot_poses_file = paths.CAPTURES_EXTR_DIR.glob("*.json")
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
        (-70.0, 50.0),
        (-250.0, -130.0),
        (300.0, 400.0),
    )
    grid_step: float = 50.0
    reference_point_offset: tuple[float, float, float, float] = (
        0.7,
        0.0,
        0.05,
        1.0,
    )
    tool_orientation: tuple[float, float, float] = (180.0, 0.0, 180.0)
    charuco_xml: str = str(paths.CAMERA_INTR / "charuco_cam.xml")
    calib_output_dir: str = str(paths.RESULTS_DIR)


grid_calib = GridCalibCfg()


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    ppx: float
    ppy: float
    fx: float
    fy: float
    model: str
    coeffs: tuple[float, float, float, float, float]


@dataclass(frozen=True)
class CameraExtrinsics:
    rotation: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    translation: tuple[float, float, float]


DEPTH_SCALE = 0.0010000000474974513


@dataclass(frozen=True)
class D415_Cfg:
    """Resolution and frame rate parameters."""

    rgb_width: int = 1280
    rgb_height: int = 720
    depth_width: int = 1280
    depth_height: int = 720
    fps: int = 30
    depth_scale: float = DEPTH_SCALE
    align_to_color: bool = True


camera = D415_Cfg()
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

INTRINSICS_COLOR = CameraIntrinsics(
    width=1280,
    height=720,
    ppx=636.4371337890625,
    ppy=363.9315490722656,
    fx=911.01806640625,
    fy=909.2081909179688,
    model="distortion.inverse_brown_conrady",
    coeffs=(0.0, 0.0, 0.0, 0.0, 0.0),
)

INTRINSICS_DEPTH_MATRIX = (
    (INTRINSICS_DEPTH.fx, 0.0, INTRINSICS_DEPTH.ppx),
    (0.0, INTRINSICS_DEPTH.fy, INTRINSICS_DEPTH.ppy),
    (0.0, 0.0, 1.0),
)

INTRINSICS_COLOR_MATRIX = (
    (INTRINSICS_COLOR.fx, 0.0, INTRINSICS_COLOR.ppx),
    (0.0, INTRINSICS_COLOR.fy, INTRINSICS_COLOR.ppy),
    (0.0, 0.0, 1.0),
)

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

DEFAULT_DEPTH_INTRINSICS = np.array(INTRINSICS_DEPTH_MATRIX, dtype=np.float32)


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
