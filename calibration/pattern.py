from __future__ import annotations

"""Calibration pattern implementations and factory helpers."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import cv2
import numpy as np

from .detector import (
    CharucoBoardConfig,
    ArucoBoardConfig,
    detect_charuco,
    find_aruco,
    draw_markers,
)
from utils.logger import Logger, LoggerType
from utils.error_tracker import ErrorTracker
import utils


def get_dictionary_name(dictionary: cv2.aruco_Dictionary) -> str:
    """Return OpenCV predefined dictionary name for ``dictionary``."""
    if hasattr(cv2.aruco, "getPredefinedDictionaryName"):
        try:
            return cv2.aruco.getPredefinedDictionaryName(dictionary)
        except Exception:
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


class ArucoPattern(CalibrationPattern):
    """Single ArUco marker pattern used for basic pose estimation."""

    config: ArucoBoardConfig

    def __init__(self, config: ArucoBoardConfig) -> None:
        """Initialize with the given configuration."""
        super().__init__()
        self.config = config
        self.logger = Logger.get_logger("calibration.aruco")
        params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.config.dictionary, params)

    def detect(
        self, image: np.ndarray, visualize: bool = False
    ) -> Optional[PatternDetection]:
        """
        Detect ArUco marker(s) in the image and optionally draw them.

        Args:
            image (np.ndarray): Input BGR image.
            visualize (bool): If True, draw detected markers on the image window.

        Returns:
            Optional[PatternDetection]: Detection object if markers found, else None.
        """
        result = find_aruco(image, self.config)
        if result is None:
            self.logger.debug("No ArUco markers detected in image")
            return None

        corners, ids, obj_points, marker_corners = result

        self.logger.debug(
            f"Aruco detection corners sample: {corners.reshape(-1, 2)[:2].tolist()}"
        )

        if visualize:
            vis_image = image.copy()
            draw_markers(vis_image, marker_corners, ids)

        det = PatternDetection(np.asarray(corners, dtype=np.float32), ids, obj_points)
        self.add_detection(det)
        return det

    def calibrate_camera(
        self, image_size: Tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """Calibrate from stored ArUco detections."""
        corners = [d.corners for d in self.detections]
        ids = [d.ids for d in self.detections]
        ret, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraAruco(
            corners, ids, self.board, image_size, None, None
        )
        errors: List[float] = []
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
        self, detection: PatternDetection, K: np.ndarray, dist: np.ndarray
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Return marker pose relative to the camera."""
        if detection.ids is None:
            return None
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            detection.corners,
            self.config.marker_length,
            K,
            dist,
        )
        if rvec is None or tvec is None:
            return None
        R, _ = cv2.Rodrigues(rvec[0])
        if detection.object_points is not None:
            pts = (R @ detection.object_points[:2].T).T + tvec[0].reshape(3)
            self.logger.debug(f"Aruco transformed points sample: {pts.tolist()}")
        return R, tvec[0].reshape(3)


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
        self.logger.debug(
            f"Charuco detection corners sample: {result.corners.reshape(-1, 2)[:2].tolist()}"
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
            if not corners or not ids:
                raise RuntimeError("No valid Charuco detections collected")
            K_init = np.eye(3, dtype=np.float64)
            dist_init = np.zeros((5, 1), dtype=np.float64)
            ret, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                charucoCorners=corners,
                charucoIds=ids,
                board=self.board,
                imageSize=image_size,
                cameraMatrix=K_init,
                distCoeffs=dist_init,
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
            if detection.object_points is not None:
                pts = (R @ detection.object_points.T).T + tvec.reshape(3)
                self.logger.debug(
                    f"Charuco transformed points sample: {pts[:2].tolist()}"
                )
            return R, tvec.reshape(3)
        except Exception as exc:
            self.logger.error(f"Pose estimation error: {exc}")
            ErrorTracker.report(exc)
            return None


def create_pattern(name: str) -> CalibrationPattern:
    """Return a calibration pattern instance by name."""

    name = name.lower()
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
