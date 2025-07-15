from __future__ import annotations

"""Aruco marker utilities and calibration pattern implementation."""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class ArucoBoardConfig:
    """Definition of a single-marker board."""

    marker_length: float
    dictionary: cv2.aruco_Dictionary

    def create(self) -> cv2.aruco_Board:
        """Return an OpenCV GridBoard representing the marker."""
        return cv2.aruco.GridBoard(
            (1, 1), self.marker_length, self.marker_length * 0.5, self.dictionary
        )


def detect_markers(
    image: np.ndarray, cfg: ArucoBoardConfig
) -> Optional[tuple[list[np.ndarray], np.ndarray]]:
    """Return detected marker corners and ids from ``image``."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    det = cv2.aruco.ArucoDetector(cfg.dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None
    return corners, ids


def draw_markers(
    image: np.ndarray, corners: list[np.ndarray], ids: np.ndarray
) -> np.ndarray:
    """Return ``image`` overlaid with marker corners and ids."""
    vis = image.copy()
    cv2.aruco.drawDetectedMarkers(vis, corners, ids)
    return vis


class ArucoPattern:
    """Calibration pattern for a single ArUco marker."""

    def __init__(self, config: ArucoBoardConfig) -> None:
        self.config = config
        self.board = self.config.create()
        self.detector = cv2.aruco.ArucoDetector(
            self.config.dictionary, cv2.aruco.DetectorParameters()
        )
        self.detections: list[tuple[list[np.ndarray], np.ndarray]] = []

    def detect(
        self, image: np.ndarray
    ) -> Optional[tuple[list[np.ndarray], np.ndarray]]:
        """Detect markers in ``image`` and store the result."""
        result = detect_markers(image, self.config)
        if result is not None:
            self.detections.append(result)
        return result

    def clear(self) -> None:
        """Remove stored detections."""
        self.detections.clear()

    def calibrate_camera(
        self,
        image_size: Tuple[int, int],
        K: Optional[np.ndarray] = None,
        dist: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """Return camera parameters estimated from detected markers."""
        corners = [c for c, _ in self.detections]
        ids = [i for _, i in self.detections]
        flags = cv2.CALIB_USE_INTRINSIC_GUESS if K is not None else 0
        ret, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraAruco(
            corners, ids, self.board, image_size, K, dist, flags=flags
        )
        errors = []
        obj_points = self.board.objPoints
        for c_set, id_set, rv, tv in zip(corners, ids, rvecs, tvecs):
            pts = []
            img = []
            for c, mid in zip(c_set, id_set.flatten()):
                pts.extend(obj_points[mid][0])
                img.extend(c.reshape(4, 2))
            pts = np.asarray(pts, dtype=np.float32)
            img = np.asarray(img, dtype=np.float32)
            proj, _ = cv2.projectPoints(pts, rv, tv, K, dist)
            diff = np.linalg.norm(img - proj.reshape(-1, 2), axis=1)
            errors.append(float(np.sqrt(np.mean(np.square(diff)))))
        self.clear()
        return K, dist, ret, np.asarray(errors)

    def estimate_pose(
        self,
        corners: list[np.ndarray],
        ids: np.ndarray,
        K: np.ndarray,
        dist: np.ndarray,
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Estimate board pose from detected corners and ids."""
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.config.marker_length, K, dist
        )
        if rvec is None or tvec is None:
            return None
        R, _ = cv2.Rodrigues(rvec[0])
        return R, tvec[0].reshape(3)
