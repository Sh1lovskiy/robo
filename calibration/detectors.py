from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class Checkerboard:
    size: Tuple[int, int]
    square_size: float


@dataclass
class CharucoBoardCfg:
    squares: Tuple[int, int]
    square_size: float
    marker_size: float
    dictionary: cv2.aruco_Dictionary

    def create(self) -> cv2.aruco_CharucoBoard:
        return cv2.aruco.CharucoBoard(
            self.squares,
            self.square_size,
            self.marker_size,
            self.dictionary,
        )


def find_checkerboard(
    img: np.ndarray, board: Checkerboard
) -> Tuple[np.ndarray, np.ndarray] | None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, board.size, None)
    if not ret:
        return None
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
    objp = np.zeros((board.size[0] * board.size[1], 3), np.float32)
    grid = np.mgrid[0 : board.size[0], 0 : board.size[1]]
    objp[:, :2] = grid.T.reshape(-1, 2)
    objp *= board.square_size
    return corners, objp


def find_charuco(
    img: np.ndarray, board: CharucoBoardCfg
) -> Tuple[np.ndarray, np.ndarray] | None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dictionary = board.dictionary
    board_inst = board.create()
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    if ids is None or len(ids) == 0:
        return None
    _, char_corners, char_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, board_inst
    )
    if char_corners is None or char_ids is None or len(char_ids) < 4:
        return None
    return char_corners, char_ids
