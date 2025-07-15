from __future__ import annotations

"""Calibration pattern detectors and pose estimation helpers."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from utils.error_tracker import ErrorTracker
from utils.keyboard import GlobalKeyListener
from utils.logger import Logger

logger = Logger.get_logger("calibration.detector")


@dataclass(frozen=True)
class CheckerboardConfig:
    """Checkerboard pattern configuration."""

    size: Tuple[int, int]
    square_size: float


@dataclass(frozen=True)
class CharucoBoardConfig:
    """Charuco board configuration."""

    squares: Tuple[int, int]
    square_size: float
    marker_size: float
    dictionary: cv2.aruco_Dictionary

    def create(self) -> cv2.aruco_CharucoBoard:
        """Return an OpenCV Charuco board."""
        return cv2.aruco.CharucoBoard(
            self.squares, self.square_size, self.marker_size, self.dictionary
        )


@dataclass(frozen=True)
class ArucoBoardConfig:
    """Single ArUco marker configuration."""

    marker_length: float
    dictionary: cv2.aruco_Dictionary


def create_checkerboard_points(cfg: CheckerboardConfig) -> np.ndarray:
    """Return 3D object points for the checkerboard."""
    objp = np.zeros((cfg.size[0] * cfg.size[1], 3), np.float32)
    grid = np.mgrid[0 : cfg.size[0], 0 : cfg.size[1]]
    objp[:, :2] = grid.T.reshape(-1, 2)
    objp *= cfg.square_size
    return objp


def find_checkerboard(
    img: np.ndarray, cfg: CheckerboardConfig
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Detect a checkerboard and return 2D corners and 3D object points."""
    logger.info("Detecting checkerboard")
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, cfg.size, None)
        if not ret:
            logger.warning("Checkerboard not found")
            return None
        term = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
        objp = create_checkerboard_points(cfg)
        logger.info("Checkerboard detected")
        return corners, objp
    except Exception as exc:
        logger.error(f"Checkerboard detection failed: {exc}")
        ErrorTracker.report(exc)
        return None


def find_aruco(
    img: np.ndarray, cfg: ArucoBoardConfig
) -> Optional[tuple[List[np.ndarray], np.ndarray]]:
    """Detect ArUco markers and return their corners and ids."""
    logger.debug("Detecting ArUco markers")
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detector = cv2.aruco.ArucoDetector(
            cfg.dictionary, cv2.aruco.DetectorParameters()
        )
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            logger.warning("No ArUco markers found")
            return None
        logger.debug(f"Detected {len(ids)} markers")
        return corners, ids
    except Exception as exc:
        logger.error(f"Aruco detection failed: {exc}")
        ErrorTracker.report(exc)
        return None


def draw_markers(
    image: np.ndarray, corners: list[np.ndarray], ids: np.ndarray
) -> np.ndarray:
    """Return ``image`` overlaid with detected ArUco markers."""
    vis = image.copy()
    cv2.aruco.drawDetectedMarkers(vis, corners, ids)
    return vis


@dataclass
class Detection:
    """Detected pattern result with 2D-3D correspondences."""

    corners: np.ndarray
    obj_points: np.ndarray
    ids: np.ndarray | None = None


def pose_from_detection(
    detection: Detection, K: np.ndarray, dist: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate pose (R, t) of the detected pattern relative to the camera."""
    ok, rvec, tvec = cv2.solvePnP(detection.obj_points, detection.corners, K, dist)
    if not ok:
        raise RuntimeError("solvePnP failed")
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3)


_CHARUCO_CACHE: dict[tuple, np.ndarray] = {}


def _compute_corr_map(
    board: cv2.aruco_CharucoBoard, detector: cv2.aruco.CharucoDetector
) -> np.ndarray:
    """Compute ID reordering map for a Charuco board.

    The OpenCV detector may return corner IDs in an orientation that does not
    match the conventional bottom-left to top-right ordering.  A synthetic board
    image is generated and passed through the detector to determine if the
    detected ordering needs to be flipped along the vertical axis.  The returned
    array maps detected IDs to the desired ascending order.
    """
    try:
        num_w, num_h = board.getChessboardSize()
        num_int = (num_w - 1) * (num_h - 1)
        corr = np.arange(num_int)

        img = board.generateImage(outSize=(1000, 1000))
        corners, ids, marker_corners, marker_ids = detector.detectBoard(img)

        if corners is None or ids is None or len(ids) < 4:
            logger.warning("Charuco correlation map: insufficient synthetic detection.")
            return corr

        # Compute using Y-axis flip (compare first and last point)
        first_y = corners[0][0][1]
        last_y = corners[-1][0][1]

        if first_y < last_y:
            for row_a in range((num_h - 1) // 2):
                row_b = (num_h - 2) - row_a
                sa = slice(row_a * (num_w - 1), (row_a + 1) * (num_w - 1))
                sb = slice(row_b * (num_w - 1), (row_b + 1) * (num_w - 1))
                corr[sa], corr[sb] = corr[sb].copy(), corr[sa].copy()

        logger.debug(
            f"Computed Charuco correlation map:\n"
            f" - Shape: {corr.shape},\n"
            f" - First remapped IDs: {corr[:min(28, len(corr))].tolist()}"
        )
        return corr

    except Exception as exc:
        logger.error(f"Failed to compute Charuco correlation map: {exc}")
        ErrorTracker.report(exc)
        return np.arange(
            (board.getChessboardSize()[0] - 1) * (board.getChessboardSize()[1] - 1)
        )


def detect_charuco(
    img: np.ndarray,
    cfg: CharucoBoardConfig,
    *,
    enforce_ascending_ids: bool = True,
    visualize: bool = False,
) -> Optional[Detection]:
    """
    Detect Charuco board corners with optional ID remapping and visualization.

    Returns:
        Detection: corners, object points, remapped IDs
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        board = cfg.create()
        detector_params = cv2.aruco.CharucoParameters()
        detector = cv2.aruco.CharucoDetector(board, detector_params)
        corners, ids, marker_corners, marker_ids = detector.detectBoard(gray)

        if corners is None or ids is None or len(ids) < 4:
            logger.warning("Charuco board not found")
            return None

        key = (
            cfg.squares,
            cfg.square_size,
            cfg.marker_size,
            tuple(cfg.dictionary.bytesList.flatten()),
        )

        if key not in _CHARUCO_CACHE:
            corr = _compute_corr_map(board, detector)
            _CHARUCO_CACHE[key] = corr
        else:
            corr = _CHARUCO_CACHE[key]

        ids_remapped = np.array([[corr[idx[0]]] for idx in ids], dtype=np.int32)

        logger.debug(
            f"Remapping Charuco IDs: {ids.flatten().tolist()} → {ids_remapped.flatten().tolist()}"
        )

        obj_points = board.getChessboardCorners()[ids_remapped.flatten()].copy()

        if visualize:
            _show_charuco(img, corners, ids_remapped, marker_corners, marker_ids)

        return Detection(
            corners, obj_points, ids_remapped if enforce_ascending_ids else ids
        )

    except Exception as exc:
        logger.error(f"Charuco detection failed: {exc}")
        ErrorTracker.report(exc)
        return None


def _show_charuco(
    img: np.ndarray,
    corners: np.ndarray,
    ids: np.ndarray,
    marker_corners: np.ndarray | None,
    marker_ids: np.ndarray | None,
) -> None:
    """Visualize Charuco detection in an OpenCV window."""
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    if marker_corners is not None and marker_ids is not None:
        vis = cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)
    for i, pt in enumerate(corners):
        pos = tuple(int(x) for x in pt.ravel())
        cv2.circle(vis, pos, 3, (0, 255, 0), -1)
        cv2.putText(
            vis, str(ids[i][0]), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
        )
    window_name = "Charuco Detection"
    cv2.imshow(window_name, vis)
    closed = False

    def on_quit():
        nonlocal closed
        closed = True

    listener = GlobalKeyListener({"q": on_quit})
    listener.start()

    try:
        while not closed:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            key = cv2.waitKey(50)
    finally:
        listener.stop()
        cv2.destroyWindow(window_name)
