from __future__ import annotations

"""Calibration pattern implementations and factory helpers."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np

from .detector import (
    CheckerboardConfig,
    CharucoBoardConfig,
    detect_charuco,
    find_checkerboard,
    create_checkerboard_points,
)
from utils.logger import Logger, LoggerType
from utils.error_tracker import ErrorTracker
import utils
from calibration.aruco import ArucoBoardConfig, ArucoPattern


def get_dictionary_name(dictionary: cv2.aruco_Dictionary) -> str:
    """Return OpenCV predefined dictionary name for ``dictionary``."""
    if hasattr(cv2.aruco, "getPredefinedDictionaryName"):
        try:
            return cv2.aruco.getPredefinedDictionaryName(dictionary)
        except Exception:  # pragma: no cover - fallback for older OpenCV
            pass
    for name in dir(cv2.aruco):
        if name.startswith("DICT_"):
            dict_id = getattr(cv2.aruco, name)
            if (
                dictionary.bytesList.shape[0]
                == cv2.aruco.getPredefinedDictionary(dict_id).bytesList.shape[0]
            ):
                return name
    return "UNKNOWN_DICTIONARY"


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
        """Store a detection result."""
        self.logger.debug("Adding detection")
        self.detections.append(detection)

    def clear(self) -> None:
        """Reset stored detections."""
        self.logger.debug("Clearing detections")
        self.detections.clear()

    def detect(self, image: np.ndarray) -> Optional[PatternDetection]:
        raise NotImplementedError

    def calibrate_camera(
        self, image_size: Tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        raise NotImplementedError

    def estimate_pose(
        self, detection: PatternDetection, K: np.ndarray, dist: np.ndarray
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError


@dataclass
class CheckerboardPattern(CalibrationPattern):
    """Checkerboard calibration target."""

    def __post_init__(self) -> None:
        self.logger = Logger.get_logger("calibration.checkerboard")

    def detect(self, image: np.ndarray) -> Optional[PatternDetection]:
        """Detect checkerboard corners and store the result."""
        self.logger.info("Detecting checkerboard")
        try:
            result = find_checkerboard(image, self.config)
            if result is None:
                self.logger.warning("Checkerboard not found")
                return None
            corners, objp = result
            det = PatternDetection(corners, None, objp)
            self.add_detection(det)
            return det
        except Exception as exc:
            self.logger.error(f"Detection failed: {exc}")
            ErrorTracker.report(exc)
            return None

    def calibrate_camera(
        self, image_size: Tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """Calibrate camera using stored checkerboard detections."""
        self.logger.info("Calibrating from checkerboard detections")
        try:
            obj_points = [self.obj_points] * len(self.detections)
            img_points = [d.corners for d in self.detections]
            ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                obj_points,
                img_points,
                image_size,
                None,
                None,
            )
            errors: List[float] = []
            for objp, imgp, rv, tv in zip(obj_points, img_points, rvecs, tvecs):
                proj, _ = cv2.projectPoints(objp, rv, tv, K, dist)
                diff = np.linalg.norm(imgp.squeeze() - proj.squeeze(), axis=1)
                errors.append(float(np.sqrt(np.mean(np.square(diff)))))
            self.logger.info(f"Calibration RMS: {ret:.6f}")
            return K, dist, ret, np.asarray(errors)
        except Exception as exc:
            self.logger.error(f"Calibration failed: {exc}")
            ErrorTracker.report(exc)
            raise
        finally:
            self.clear()

    def estimate_pose(
        self, detection: PatternDetection, K: np.ndarray, dist: np.ndarray
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Estimate pose of the checkerboard from a detection."""
        self.logger.debug("Estimating pose from checkerboard")
        try:
            assert detection.object_points is not None
            ok, rvec, tvec = cv2.solvePnP(
                detection.object_points, detection.corners, K, dist
            )
            if not ok:
                self.logger.warning("Pose estimation failed")
                return None
            R, _ = cv2.Rodrigues(rvec)
            return R, tvec.reshape(3)
        except Exception as exc:
            self.logger.error(f"Pose estimation error: {exc}")
            ErrorTracker.report(exc)
            return None


class CharucoPattern(CalibrationPattern):
    """
    Returns all corners, ids, marker corners/ids, and supports enforced ascending IDs.
    """

    config: CharucoBoardConfig

    def __init__(self, config: CharucoBoardConfig):
        """
        Args:
            board: cv2.aruco_CharucoBoard instance for detection.
            enforce_ascending_ids: If True, reorders detected IDs bottom-left to top-right.
        """
        super().__init__()
        self.config = config
        self.board = self.config.create()
        self.logger = Logger.get_logger("calibration.charuco")
        self.enforce_ascending_ids = True
        self._correlation_map = None
        self.detector_params = cv2.aruco.CharucoParameters()
        self.detector_params.minMarkers = 0
        self.detector_params.tryRefineMarkers = True
        self.detector = cv2.aruco.CharucoDetector(self.board, self.detector_params)
        self.detector.setBoard(self.board)
        name = get_dictionary_name(self.config.dictionary)
        self.logger.info(
            "CharucoPattern initialized with parameters:\n"
            f"  - Board size: {self.config.squares[0]} x {self.config.squares[1]}\n"
            f"  - Square size: {self.config.square_size:.4f} m\n"
            f"  - Marker size: {self.config.marker_size:.4f} m\n"
            f"  - Dictionary: {name}"
        )

    def detect(
        self, image: np.ndarray, visualize: bool = True
    ) -> Optional[PatternDetection]:
        """
        Detects charuco corners, ids, marker corners and ids.
        Optionally reorders charuco_ids to be ascending from the bottom left.
        Returns PatternDetection or None.
        """
        result = detect_charuco(
            image,
            self.config,
            enforce_ascending_ids=self.enforce_ascending_ids,
            visualize=visualize,
        )
        if result is None:
            return None
        det = PatternDetection(
            corners=result.corners,
            ids=result.ids,
            object_points=result.obj_points,
        )
        self.add_detection(det)
        return det

    def calibrate_camera(
        self, image_size: Tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Calibrate camera using stored Charuco detections."""
        self.logger.info("Calibrating from Charuco detections")
        try:
            corners = [d.corners for d in self.detections]
            ids = [d.ids for d in self.detections]
            ret, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                corners,
                ids,
                self.board,
                image_size,
                None,
                None,
            )
            errors: List[float] = []
            obj_points = [d.object_points for d in self.detections]
            for objp, imgp, rv, tv in zip(obj_points, corners, rvecs, tvecs):
                proj, _ = cv2.projectPoints(objp, rv, tv, K, dist)
                diff = np.linalg.norm(imgp.squeeze() - proj.squeeze(), axis=1)
                errors.append(float(np.sqrt(np.mean(np.square(diff)))))
            self.logger.info(f"   RMS: {ret:.6f} px")
            return K, dist, ret, np.asarray(errors)
        except Exception as exc:
            self.logger.error(f"Calibration failed: {exc}")
            ErrorTracker.report(exc)
            raise
        finally:
            self.clear()

    def estimate_pose(
        self, detection: PatternDetection, K: np.ndarray, dist: np.ndarray
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Estimate pose of the Charuco board."""
        self.logger.debug("Estimating pose from Charuco")
        try:
            assert detection.ids is not None
            ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                detection.corners,
                detection.ids,
                self.board,
                K,
                dist,
                rvec=None,
                tvec=None,
            )
            if not ok:
                self.logger.warning("Pose estimation failed")
                return None
            R, _ = cv2.Rodrigues(rvec)
            return R, tvec.reshape(3)
        except Exception as exc:
            self.logger.error(f"Pose estimation error: {exc}")
            ErrorTracker.report(exc)
            return None


def create_pattern(name: str) -> CalibrationPattern:
    """Return a calibration pattern instance by name."""

    name = name.lower()
    if name == "chess":
        cfg = CheckerboardConfig(
            utils.checkerboard.size,
            utils.checkerboard.square_size,
        )
        return CheckerboardPattern(cfg)
    if name == "charuco":
        cfg = CharucoBoardConfig(
            squares=utils.charuco.squares,
            square_size=utils.charuco.square_size,
            marker_size=utils.charuco.marker_size,
            dictionary=cv2.aruco.getPredefinedDictionary(utils.charuco.dictionary),
        )
        return CharucoPattern(cfg)
    if name == "aruco":
        cfg = ArucoBoardConfig(
            marker_length=utils.aruco.marker_length,
            dictionary=cv2.aruco.getPredefinedDictionary(utils.aruco.dictionary),
        )
        return ArucoPattern(cfg)
    raise ValueError(f"Unknown pattern: {name}")
