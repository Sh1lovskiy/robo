"""Charuco board calibration pattern."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from utils.logger import Logger

from .base import CalibrationPattern

log = Logger.get_logger("calibrate.charuco")


class CharucoPattern(CalibrationPattern):
    """Calibration using Charuco board corners."""

    def __init__(
        self,
        board_size: Tuple[int, int],
        square_length: float,
        aruco_dict: str,
    ) -> None:
        super().__init__(board_size, square_length)
        dict_obj = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_dict))
        marker_len = square_length * 0.7
        self.board = cv2.aruco.CharucoBoard(board_size, square_length, marker_len, dict_obj)
        self.dict = dict_obj

    def detect(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dict)
        if len(corners) == 0 or ids is None:
            return None
        ret, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, self.board
        )
        if not ret or ch_corners is None or ch_ids is None or len(ch_corners) < 4:
            return None
        overlay = image.copy()
        cv2.aruco.drawDetectedCornersCharuco(overlay, ch_corners, ch_ids, (0, 255, 0))
        obj_pts = self.board.getChessboardCorners()[ch_ids.flatten()].reshape(-1, 3)
        return obj_pts, ch_corners.reshape(-1, 2), overlay
