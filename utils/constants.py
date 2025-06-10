# utils/constants.py

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
CLOUD_OUTPUT_DIR = os.path.join(BASE_DIR, "clouds")

DEFAULT_IP = "192.168.1.10"
TOOL_ID = 0
USER_FRAME_ID = 0
NORMAL_VELOCITY = 35.0
EMERGENCY_DELAY = 0.5

POSES_FILE = "poses.txt"
PATH_FILE = "path.txt"
HAND_EYE_FILE = "handeye.npz"
CHARUCO_CALIB_FILE = "charuco_cam.xml"
