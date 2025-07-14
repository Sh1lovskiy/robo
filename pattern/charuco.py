"""Charuco board detection utilities supporting multiple OpenCV versions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

__all__ = [
    "CharucoBoardConfig",
    "detect_charuco_corners",
    "draw_corners",
]


@dataclass(frozen=True)
class CharucoBoardConfig:
    """Configuration for generating a Charuco board."""

    squares: Tuple[int, int]
    square_size: float
    marker_size: float
    dictionary: cv2.aruco_Dictionary

    def create(self) -> cv2.aruco_CharucoBoard:
        """Return an OpenCV Charuco board."""
        return cv2.aruco.CharucoBoard(
            self.squares, self.square_size, self.marker_size, self.dictionary
        )


def _detect_opencv_lt_4_7(gray: np.ndarray, board: cv2.aruco_CharucoBoard):
    """Detection routine for OpenCV < 4.7.0."""
    corners, ids, _ = cv2.aruco.detectMarkers(gray, board.dictionary)
    if ids is None or len(ids) == 0:
        return None, None
    cv2.aruco.refineDetectedMarkers(gray, board, corners, ids, rejectedCorners=None)
    char_corners, char_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, board
    )
    if char_ids is None or len(char_ids) == 0:
        return None, None
    return char_corners, char_ids


def _detect_opencv_ge_4_7(gray: np.ndarray, board: cv2.aruco_CharucoBoard):
    """Detection routine for OpenCV >= 4.7.0."""
    detector = cv2.aruco.CharucoDetector(board, cv2.aruco.CharucoParameters())
    char_corners, char_ids, _, _ = detector.detectBoard(gray)
    return char_corners, char_ids


def detect_charuco_corners(
    img: np.ndarray,
    board: cv2.aruco_CharucoBoard,
    *,
    reorder: bool = True,
    visualize: bool = False,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Return detected Charuco corners and ids.

    Parameters
    ----------
    img:
        Input BGR or grayscale image.
    board:
        Charuco board model defining square/marker layout.
    reorder:
        If ``True`` reorder IDs to follow bottom-left to top-right order.
    visualize:
        When ``True`` an OpenCV window with detected corners is shown.

    Returns
    -------
    Optional[tuple[np.ndarray, np.ndarray]]
        ``(corners, ids)`` on success or ``None`` when detection fails.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if hasattr(cv2.aruco, "CharucoDetector"):
        corners, ids = _detect_opencv_ge_4_7(gray, board)
    else:
        corners, ids = _detect_opencv_lt_4_7(gray, board)
    if corners is None or ids is None:
        return None

    if reorder:
        num_w, num_h = board.getChessboardSize()
        num_int = (num_w - 1) * (num_h - 1)
        corr = np.arange(num_int)
        first_y = corners[0][0][1]
        last_y = corners[-1][0][1]
        if first_y < last_y:
            for row_a in range((num_h - 1) // 2):
                row_b = (num_h - 2) - row_a
                sa = slice(row_a * (num_w - 1), (row_a + 1) * (num_w - 1))
                sb = slice(row_b * (num_w - 1), (row_b + 1) * (num_w - 1))
                corr[sa], corr[sb] = corr[sb].copy(), corr[sa].copy()
        ids = np.array([[corr[i[0]]] for i in ids], dtype=np.int32)

    if visualize:
        vis = draw_corners(img, corners, ids)
        cv2.imshow("charuco", vis)
        cv2.waitKey(0)
        cv2.destroyWindow("charuco")
    return corners, ids


def draw_corners(
    image: np.ndarray, corners: np.ndarray, ids: np.ndarray | None
) -> np.ndarray:
    """Return ``image`` overlaid with ``corners`` and ``ids`` for inspection."""
    vis = image.copy()
    for i, pt in enumerate(corners.reshape(-1, 2)):
        pos = tuple(int(x) for x in pt)
        cv2.circle(vis, pos, 3, (0, 255, 0), -1)
        if ids is not None:
            cv2.putText(
                vis,
                str(int(ids[i])),
                pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )
    return vis
