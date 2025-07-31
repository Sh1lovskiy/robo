"""Chessboard calibration pattern."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from utils.logger import Logger

from .base import CalibrationPattern

log = Logger.get_logger("calibrate.chessboard")


class ChessboardPattern(CalibrationPattern):
    """Calibration pattern based on a regular chessboard."""

    def __init__(self, board_size: Tuple[int, int], square_length: float) -> None:
        super().__init__(board_size, square_length)
        objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2)
        self.obj_points = objp * square_length

    def detect(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, self.board_size)
        if not found:
            return None
        cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 0.0001),
        )
        overlay = image.copy()
        cv2.drawChessboardCorners(overlay, self.board_size, corners, found)
        return self.obj_points, corners.reshape(-1, 2), overlay
