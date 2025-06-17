# utils/constants.py
"""Common constants used across the project."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"
DEFAULT_LOG_DIR = REPO_ROOT / "logs"
DEFAULT_CLOUD_DIR = REPO_ROOT / "clouds"
DEFAULT_ROBOT_IP = "192.168.58.2"

# Mapping of Charuco dictionary names to OpenCV constants
CHARUCO_DICT_MAP = {
    "4X4_50": 0,
    "4X4_100": 1,
    "5X5_50": 8,
    "5X5_100": 9,
    "6X6_50": 16,
    "6X6_100": 17,
}


