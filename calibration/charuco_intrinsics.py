"""Charuco board intrinsic calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import yaml

import cv2
import numpy as np

from utils.logger import Logger

logger = Logger.get_logger("calibration.charuco_intrinsics")


DICT_MAP = {
    "4X4_50": cv2.aruco.DICT_4X4_50,
    "4X4_100": cv2.aruco.DICT_4X4_100,
    "5X5_50": cv2.aruco.DICT_5X5_50,
    "5X5_100": cv2.aruco.DICT_5X5_100,
    "6X6_50": cv2.aruco.DICT_6X6_50,
    "6X6_100": cv2.aruco.DICT_6X6_100,
}


@dataclass
class CalibrationResult:
    """Holds intrinsic calibration outputs."""

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rms: float


def save_camera_params(
    filename: Path,
    image_size: tuple[int, int],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    total_avg_err: float,
) -> None:
    """Save camera parameters and reprojection error to YAML."""

    calibration_data = {
        "image_width": image_size[0],
        "image_height": image_size[1],
        "camera_matrix": {
            "rows": camera_matrix.shape[0],
            "cols": camera_matrix.shape[1],
            "dt": "d",
            "data": camera_matrix.tolist(),
        },
        "distortion_coefficients": {
            "rows": dist_coeffs.shape[0],
            "cols": dist_coeffs.shape[1] if dist_coeffs.ndim > 1 else 1,
            "dt": "d",
            "data": dist_coeffs.flatten().tolist(),
        },
        "avg_reprojection_error": float(total_avg_err),
    }
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        yaml.dump(calibration_data, f)
    logger.info(f"Saved calibration parameters to {filename}")


class CharucoCalibrator:
    """Collects frames and performs Charuco intrinsic calibration."""

    def __init__(
        self, board: cv2.aruco.CharucoBoard, dictionary: cv2.aruco_Dictionary
    ) -> None:
        self.board = board
        self.dictionary = dictionary
        self.all_corners: list[np.ndarray] = []
        self.all_ids: list[np.ndarray] = []
        self.image_size: Optional[tuple[int, int]] = None

    def add_frame(self, img: np.ndarray) -> bool:
        """Detect Charuco corners and add them for calibration."""

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.image_size = (gray.shape[1], gray.shape[0])
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary)
        if ids is None or len(corners) == 0:
            logger.warning("No ArUco markers detected")
            return False
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, self.board
        )
        if retval < 10 or charuco_ids is None or len(charuco_ids) < 10:
            logger.warning("Not enough Charuco corners")
            return False
        self.all_corners.append(charuco_corners)
        self.all_ids.append(charuco_ids)
        return True

    def calibrate(self) -> CalibrationResult:
        """Run OpenCV Charuco calibration."""

        if len(self.all_corners) < 3 or self.image_size is None:
            raise RuntimeError("Not enough frames for calibration")
        rms, camera_matrix, dist_coeffs, *_ = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=self.all_corners,
            charucoIds=self.all_ids,
            board=self.board,
            imageSize=self.image_size,
            cameraMatrix=None,
            distCoeffs=None,
        )
        return CalibrationResult(camera_matrix, dist_coeffs, rms)
