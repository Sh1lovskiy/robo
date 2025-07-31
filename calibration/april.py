"""AprilTag grid calibration pattern."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from utils.logger import Logger

from .base import CalibrationPattern

log = Logger.get_logger("calibrate.april")


class AprilTagPattern(CalibrationPattern):
    """Calibration using an AprilTag grid board."""

    def __init__(
        self,
        board_size: Tuple[int, int],
        tag_size: float,
        aruco_dict: str,
    ) -> None:
        super().__init__(board_size, tag_size)
        dict_obj = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_dict))
        self.board = cv2.aruco.GridBoard(board_size, tag_size, tag_size * 0.3, dict_obj)
        self.dict = dict_obj

    def detect(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detector = cv2.aruco.ArucoDetector(self.dict)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None or len(corners) < 4:
            return None
        tag_centers = np.array([c.mean(axis=1).flatten() for c in corners], dtype=np.float32)
        obj = self.board.getObjPoints()
        obj_ids = self.board.getIds().flatten()
        id2center = {int(i): np.mean(obj[idx], axis=0) for idx, i in enumerate(obj_ids)}
        obj_pts = np.array([id2center[int(i)] for i in ids.flatten() if int(i) in id2center])
        if obj_pts.shape[0] != tag_centers.shape[0]:
            return None
        overlay = image.copy()
        for pt, idx in zip(tag_centers, ids.flatten()):
            cv2.circle(overlay, tuple(np.int32(pt)), 5, (0, 255, 0), 2)
            cv2.putText(
                overlay,
                str(int(idx)),
                tuple(np.int32(pt)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )
        return obj_pts.astype(np.float32), tag_centers, overlay
