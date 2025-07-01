"""Project settings and paths"""

from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Paths:
    """Convenient bundle of frequently used project paths."""

    CAPTURES_DIR: Path = BASE_DIR / "captures"
    RESULTS_DIR: Path = BASE_DIR / "calibration" / "results"
    CLOUD_DIR: Path = BASE_DIR / "clouds"
    LOG_DIR: Path = BASE_DIR / "logs"


paths = Paths()


@dataclass(frozen=True)
class RobotSettings:
    ip: str = "192.168.58.2"
    tool_id: int = 0
    user_frame_id: int = 0
    velocity: float = 20.0
    emergency_delay: float = 0.5
    restart_delay: float = 1.0


robot = RobotSettings()


@dataclass(frozen=True)
class VisionSettings:
    realsense_width: int = 1920
    realsense_height: int = 1080
    realsense_fps: int = 30


vision = VisionSettings()


@dataclass(frozen=True)
class CharucoSettings:
    squares_x: int = 5
    squares_y: int = 8
    square_length: float = 0.035
    marker_length: float = 0.026
    aruco_dict: str = "5X5_100"
    min_corners: int = 4
    outlier_std: float = 2.0
    analyze_corners: bool = False
    visualize: bool = False
    calib_output_dir: str = str(paths.RESULTS_DIR)
    xml_file: str = "charuco_cam.xml"


charuco = CharucoSettings()


@dataclass(frozen=True)
class HandEyeSettings:
    method: str = "ALL"
    min_corners: int = 4
    outlier_std: float = 2.0
    visualize: bool = False
    robot_poses_file: str = str(paths.CAPTURES_DIR / "poses.json")
    images_dir: str = str(paths.CAPTURES_DIR)
    charuco_xml: str = str(paths.RESULTS_DIR / charuco.xml_file)
    calib_output_dir: str = str(paths.RESULTS_DIR)


handeye = HandEyeSettings()


@dataclass(frozen=True)
class ValidationSettings:
    board_lt_base: tuple[float, float, float] = (-0.165, -0.365, 0.0)
    board_rb_base: tuple[float, float, float] = (-0.4, -0.53, 0.0)


validation = ValidationSettings()


@dataclass(frozen=True)
class LoggingSettings:
    level: str = "INFO"
    json: bool = True


logging = LoggingSettings()

# RealSense depth unit to meters conversion
DEPTH_SCALE: float = 0.0001
