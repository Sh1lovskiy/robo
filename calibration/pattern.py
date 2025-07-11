from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import cv2
import numpy as np

from .detector import (
    CheckerboardConfig,
    CharucoBoardConfig,
    ArucoConfig,
    find_checkerboard,
    find_charuco,
    find_aruco,
)
from utils.logger import Logger, LoggerType


@dataclass
class PatternDetection:
    """Container for detection results."""

    corners: np.ndarray
    ids: np.ndarray | None = None
    object_points: np.ndarray | None = None


@dataclass
class CalibrationPattern:
    """Base class for calibration target patterns."""

    logger: LoggerType = field(
        default_factory=lambda: Logger.get_logger("calibration.pattern"), init=False
    )
    detections: List[PatternDetection] = field(default_factory=list, init=False)

    def add_detection(self, detection: PatternDetection) -> None:
        """Store a single detection result."""

        self.detections.append(detection)

    def clear(self) -> None:
        """Reset stored detections."""

        self.detections.clear()

    def detect(self, image: np.ndarray) -> Optional[PatternDetection]:
        raise NotImplementedError

    def calibrate_camera(
        self, image_size: Tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray, float]:
        raise NotImplementedError

    def estimate_pose(
        self, detection: PatternDetection, K: np.ndarray, dist: np.ndarray
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError


class CheckerboardPattern(CalibrationPattern):
    """Checkerboard calibration target."""

    config: CheckerboardConfig = CheckerboardConfig((7, 6), 0.02)

    def detect(self, image: np.ndarray) -> Optional[PatternDetection]:
        result = find_checkerboard(image, self.config)
        if result is None:
            return None
        corners, objp = result
        det = PatternDetection(corners, None, objp)
        self.add_detection(det)
        return det

    def calibrate_camera(
        self, image_size: Tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray, float]:
        obj_points = [
            d.object_points for d in self.detections if d.object_points is not None
        ]
        img_points = [d.corners for d in self.detections]
        ret, K, dist, _, _ = cv2.calibrateCamera(
            obj_points, img_points, image_size, None, None
        )
        self.clear()
        return K, dist, ret

    def estimate_pose(
        self, detection: PatternDetection, K: np.ndarray, dist: np.ndarray
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        assert detection.object_points is not None
        ok, rvec, tvec = cv2.solvePnP(
            detection.object_points, detection.corners, K, dist
        )
        if not ok:
            return None
        R, _ = cv2.Rodrigues(rvec)
        return R, tvec.reshape(3)


@dataclass
class CharucoPattern(CalibrationPattern):
    """Charuco board calibration target."""

    config: CharucoBoardConfig

    def __post_init__(self) -> None:
        self.board = self.config.create()

    def detect(self, image: np.ndarray) -> Optional[PatternDetection]:
        result = find_charuco(image, self.config)
        if result is None:
            return None
        corners, ids = result
        objp = self.board.chessboardCorners[ids.flatten()].copy()
        det = PatternDetection(corners, ids, objp)
        self.add_detection(det)
        return det

    def calibrate_camera(
        self, image_size: Tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray, float]:
        corners = [d.corners for d in self.detections]
        ids = [d.ids for d in self.detections]
        ret, K, dist, _, _ = cv2.aruco.calibrateCameraCharuco(
            corners, ids, self.board, image_size, None, None
        )
        self.clear()
        return K, dist, ret

    def estimate_pose(
        self, detection: PatternDetection, K: np.ndarray, dist: np.ndarray
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        assert detection.ids is not None
        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            detection.corners, detection.ids, self.board, K, dist
        )
        if not ok:
            return None
        R, _ = cv2.Rodrigues(rvec)
        return R, tvec.reshape(3)


class ArucoPattern(CalibrationPattern):
    """Single ArUco marker calibration target."""

    config: ArucoConfig

    def __post_init__(self) -> None:
        self.board = cv2.aruco.GridBoard_create(
            1, 1, self.config.marker_length, 0.0, self.config.dictionary
        )

    def detect(self, image: np.ndarray) -> Optional[PatternDetection]:
        result = find_aruco(image, self.config)
        if result is None:
            return None
        corners, ids = result
        det = PatternDetection(corners, ids)
        self.add_detection(det)
        return det

    def calibrate_camera(
        self, image_size: Tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray, float]:
        corners = [d.corners for d in self.detections]
        ids = [d.ids for d in self.detections]
        ret, K, dist, _, _ = cv2.aruco.calibrateCameraAruco(
            corners, ids, self.board, image_size, None, None
        )
        self.clear()
        return K, dist, ret

    def estimate_pose(
        self, detection: PatternDetection, K: np.ndarray, dist: np.ndarray
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            detection.corners, self.config.marker_length, K, dist
        )
        if rvec is None or tvec is None:
            return None
        R, _ = cv2.Rodrigues(rvec[0])
        return R, tvec[0].reshape(3)


def create_pattern(name: str) -> CalibrationPattern:
    """Factory for ``CalibrationPattern`` implementations."""

    name = name.lower()
    if name == "chess":
        return CheckerboardPattern()
    if name == "charuco":
        cfg = CharucoBoardConfig(
            squares=(5, 8),
            square_size=0.035,
            marker_size=0.026,
            dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100),
        )
        return CharucoPattern(cfg)
    if name == "aruco":
        cfg = ArucoConfig(
            marker_length=0.035,
            dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100),
        )
        return ArucoPattern(cfg)
    raise ValueError(f"Unknown pattern: {name}")
