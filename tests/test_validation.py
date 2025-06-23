import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.append(str(Path(__file__).resolve().parents[1]))

from validation.marker_detection import MarkerDetector
from validation.handeye_validation import HandEyeValidator
from validation.workflows import StubCamera, StubRobot


def test_validate_pose() -> None:
    img_gray = np.full((480, 640), 255, np.uint8)
    dict5 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    marker = cv2.aruco.generateImageMarker(dict5, 0, 100)
    img_gray[190:290, 270:370] = marker
    img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=float)
    dist = np.zeros(5)
    camera = StubCamera(img, K, dist)
    robot = StubRobot([])
    detector = MarkerDetector("5X5_100")
    validator = HandEyeValidator(camera, robot, np.eye(4), K, dist, detector)
    pose = np.eye(4)
    pose[:3, 3] = [0.0, 0.0, 1.0]
    result = validator.validate_pose(pose)
    assert result["pixel_error"] < 1.0
