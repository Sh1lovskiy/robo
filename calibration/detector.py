from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, List

import cv2
import numpy as np


@dataclass(frozen=True)
class CheckerboardConfig:
    """Checkerboard pattern configuration."""

    size: Tuple[int, int]
    square_size: float


@dataclass(frozen=True)
class CharucoBoardConfig:
    """Charuco board pattern configuration."""

    squares: Tuple[int, int]
    square_size: float
    marker_size: float
    dictionary: cv2.aruco_Dictionary

    def create(self) -> cv2.aruco_CharucoBoard:
        """Create an OpenCV Charuco board."""

        return cv2.aruco.CharucoBoard(
            self.squares,
            self.square_size,
            self.marker_size,
            self.dictionary,
        )


@dataclass(frozen=True)
class ArucoConfig:
    """Single ArUco marker configuration."""

    marker_length: float
    dictionary: cv2.aruco_Dictionary


def find_checkerboard(
    img: np.ndarray, cfg: CheckerboardConfig
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, cfg.size, None)
    if not ret:
        return None
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
    objp = np.zeros((cfg.size[0] * cfg.size[1], 3), np.float32)
    grid = np.mgrid[0 : cfg.size[0], 0 : cfg.size[1]]
    objp[:, :2] = grid.T.reshape(-1, 2)
    objp *= cfg.square_size
    return corners, objp


def find_charuco(
    img: np.ndarray,
    cfg: CharucoBoardConfig,
    K: Optional[np.ndarray] = None,
    dist: Optional[np.ndarray] = None,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    board = cfg.create()
    detector_params = cv2.aruco.CharucoParameters()
    detector_params.minMarkers = 0
    detector_params.tryRefineMarkers = True
    charucodetector = cv2.aruco.CharucoDetector(board, detector_params)
    charucodetector.setBoard(board)
    charuco_corners, charuco_ids, marker_corners, marker_ids = (
        charucodetector.detectBoard(gray)
    )
    return charuco_corners, charuco_ids, marker_corners, marker_ids


def find_aruco(
    img: np.ndarray, cfg: ArucoConfig
) -> Optional[tuple[List[np.ndarray], np.ndarray]]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    params = (
        cv2.aruco.DetectorParameters()
        if hasattr(cv2.aruco, "DetectorParameters")
        else cv2.aruco.DetectorParameters_create()
    )
    detector = cv2.aruco.ArucoDetector(cfg.dictionary, params)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None
    return corners, ids
