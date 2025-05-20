"""
Project-wide constants and configuration settings.
"""

import cv2
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from enum import Enum
from typing import Dict, List


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
CALIBRATION_DATA_DIR = os.path.join(PROJECT_ROOT, "calibration", "calibration_data")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CALIBRATION_DATA_DIR, exist_ok=True)
# Output image paths
OUTPUT_PATHS = {
    "top": "images/top.jpg",
    "side": "images/side.jpg",
    "front": "images/front.jpg",
}

# Workspace segment of interest (in mm)
WORKSPACE_SEGMENT = {
    "x_min": -900,
    "x_max": 0,
    "y_min": -900,
    "y_max": 0,
    "z_min": 150,
    "z_max": 900,
}


# Grid density for mapping (points per axis)
# Higher values = more detailed mapping but longer execution time
class GridDensity:
    LOW = 10  # Quick preview
    MEDIUM = 20  # Standard analysis
    HIGH = 30  # Detailed analysis
    ULTRA = 50  # Very detailed analysis (time-consuming)


# Default tool and user frame
TOOL_ID = 0
USER_FRAME_ID = 0


# Tool orientations to test
# These quaternions represent different tool orientations to test at each position
# Format: [x, y, z, w]
def generate_orientations(num_orientations: int = 8) -> List[np.ndarray]:
    """Generate a set of tool orientations to test as quaternions."""
    orientations = []

    # Default downward orientation (pointing down along Z-axis)
    default_quat = R.from_euler("xyz", [180, 0, 0], degrees=True).as_quat()
    orientations.append(default_quat)

    # Add orientations tilted around Y-axis
    for angle in np.linspace(-60, 60, num_orientations // 2):
        quat = R.from_euler("xyz", [180, angle, 0], degrees=True).as_quat()
        orientations.append(quat)

    # Add orientations tilted around X-axis
    for angle in np.linspace(-60, 60, num_orientations // 2):
        quat = R.from_euler("xyz", [180 + angle, 0, 0], degrees=True).as_quat()
        orientations.append(quat)

    # Add orientations tilted around Z-axis
    for angle in np.linspace(0, 90, num_orientations // 4):
        quat = R.from_euler("xyz", [180, 0, angle], degrees=True).as_quat()
        orientations.append(quat)

    return orientations


# Default orientations
DEFAULT_ORIENTATIONS = generate_orientations(8)


# Visualization settings
class VisualizationConfig:
    # Point sizes and colors
    POINT_SIZE = 25
    REACHABLE_COLOR = "#2980b9"  # blue
    UNREACHABLE_COLOR = "#e74c3c"  # red
    AXES_COLOR = "#2c3e50"
    HIGHLIGHT_COLOR = "#f39c12"  # orange

    # Orientation arrows
    ARROW_LENGTH = 50  # Length of orientation arrows in mm
    ARROW_WIDTH = 3  # Width of orientation arrows
    ARROW_COLORS = {
        "x": "#e74c3c",  # red
        "y": "#2ecc71",  # green
        "z": "#3498db",  # blue
    }

    # Colormap for orientation availability
    ORIENTATION_COLORMAP = "viridis"

    # Figure size and DPI
    FIGURE_SIZE = (12, 10)
    FIGURE_DPI = 100

    # Slice thickness (mm) for 2D slice views
    SLICE_THICKNESS = 50

    # Output file formats
    SAVE_FORMATS = ["png", "html"]


# Analysis settings
class AnalysisConfig:
    # Number of points to process before saving intermediate results
    CHECKPOINT_INTERVAL = 1000

    # Minimum percentage of tested orientations that must be reachable
    # for a point to be considered "fully reachable"
    ORIENTATION_THRESHOLD = 0.7  # 70%

    # Minimum singular value threshold for detecting singularities
    SINGULARITY_THRESHOLD = 1e-5

    # IK solution strategy
    MAX_IK_ITERATIONS = 100
    IK_TOLERANCE = 1e-4

# Camera and calibration parameters
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
FRAME_SIZE = (CAMERA_WIDTH // 2, CAMERA_HEIGHT)  # Size of a single camera frame
CHESSBOARD_SIZE = (9, 6)  # Number of inner corners
SQUARE_SIZE_MM = 23  # Size of chessboard square (mm)

# Default camera index
DEFAULT_CAMERA_INDEX = 0
CAMERA_SHIFT = np.array([0, -50, 130])  # mm

# Delay in seconds between captures (optional)
CAPTURE_DELAY = 1.0
SAVE_PATH = Path("captures")
IMAGE_NAMES = ["top.png", "front.png", "side.png"]
# [x, y, z, Rx, Ry, Rz] for top, front, and side views
ORTHOGONAL_POSES = [
    [-290.0, -240.0, 530.0, -168.0, 12.0, 70.0],  # top view
    [-450.0, 0.0, 200.0, -90.0, -180.0, 0.0],  # front view
    [0.0, -330.0, 200.0, -90.0, -90.0, 0.0],  # side view
]
# Chessboard square size (mm)
SQUARE_SIZE = 23

# ArUco marker parameters
ARUCO_DICT_TYPE = cv2.aruco.DICT_5X5_50
MARKER_LENGTH = 50  # (mm)
MARKER_IDS = [0, 1]  # Target marker IDs to detect

# RAG configuration
# RAG_PATTERNS_DIR = os.path.join(DATA_DIR, "surface_patterns")
# RAG_EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
RAG_MODEL_NAME = "microsoft/codereviewer"
RAG_VECTOR_DIM = 768  # Embedding dimension for the model
RAG_TOP_K = 5  # Number of results to return

DEFAULT_IP = "192.168.58.2"
NORMAL_VELOCITY = 15.0
EMERGENCY_DELAY = 0.5  # sec


# Motion parameters
class MotionParameters:
    DEFAULT_SPEED = 20.0  # % of maximum speed
    DEFAULT_ACCELERATION = 50.0  # % of maximum acceleration
    MOTION_TIMEOUT = 10.0  # seconds
    MIN_SINGULAR_VALUE = 1e-6  # threshold for singularity detection


# Reference frames for jogging
class ReferenceFrame(Enum):
    JOINT = 0
    BASE = 2
    TOOL = 4
    WORKPIECE = 8


# Coordinate systems
class CoordinateSystem:
    DEFAULT_TOOL = 0
    DEFAULT_USER_FRAME = 0


# Robot communication constants
class CommunicationConstants:
    DEFAULT_IP = "192.168.58.2"
    COMMAND_PORT = 20003
    STATUS_PORT = 20004
    BUFFER_SIZE = 1024 * 8
    CONNECTION_TIMEOUT = 2.0  # seconds


# Robot DH parameters
class DHParameters:
    PARAMS: List[Dict[str, float]] = [
        {"theta": 0, "d": 140, "a": 0, "alpha": np.pi / 2},
        {"theta": 0, "d": 0, "a": -280, "alpha": 0},
        {"theta": 0, "d": 0, "a": -240, "alpha": 0},
        {"theta": 0, "d": 102, "a": 0, "alpha": np.pi / 2},
        {"theta": 0, "d": 102, "a": 0, "alpha": -np.pi / 2},
        {"theta": 0, "d": 100, "a": 0, "alpha": 0},
    ]
    JOINT_COUNT = len(PARAMS)


# Joint limits
class JointLimits:
    SOFT_LIMITS_DEG = np.array(
        [[-175, 175], [-265, 85], [-150, 150], [-265, 85], [-175, 175], [-175, 175]]
    )

    HARD_LIMITS_DEG = np.array(
        [[-179, 179], [-269, 89], [-152, 152], [-269, 89], [-179, 179], [-179, 179]]
    )

    @staticmethod
    def get_soft_limits_rad() -> np.ndarray:
        return np.radians(JointLimits.SOFT_LIMITS_DEG)

    @staticmethod
    def get_hard_limits_rad() -> np.ndarray:
        return np.radians(JointLimits.HARD_LIMITS_DEG)

    @staticmethod
    def get_soft_limits_deg() -> np.ndarray:
        return JointLimits.SOFT_LIMITS_DEG

    @staticmethod
    def get_hard_limits_def() -> np.ndarray:
        return JointLimits.HARD_LIMITS_DEG
