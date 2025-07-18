from __future__ import annotations

"""Calibration pattern detectors and pose estimation helpers."""

from dataclasses import dataclass
from typing import Optional, Tuple

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
        board = cv2.aruco.CharucoBoard(
            self.squares, self.square_size, self.marker_size, self.dictionary
        )
        return board


def draw_markers(image, corners, ids):
    """
    Draw ArUco markers on the image for visualization.
    corners: list of N (4,2) arrays (as returned by detectMarkers, not vstack)
    ids:     (N,1) or (N,) array
    """
    vis = image.copy()
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)
    else:
        cv2.aruco.drawDetectedMarkers(vis, corners)
    cv2.namedWindow("Aruco Detection", cv2.WINDOW_NORMAL)
    cv2.imshow("Aruco Detection", vis)
    return vis


def draw_charuco(
    image: np.ndarray,
    corners: np.ndarray,
    ids: np.ndarray,
    marker_corners: np.ndarray | None = None,
    marker_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Return ``image`` with detected Charuco corners and marker outlines."""
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    if marker_corners is not None and marker_ids is not None:
        vis = cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)
    for i, pt in enumerate(corners):
        pos = tuple(int(x) for x in pt.ravel())
        cv2.circle(vis, pos, 3, (0, 255, 0), -1)
        cv2.putText(
            vis,
            str(ids[i][0]),
            pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )
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


def find_aruco(
    img: np.ndarray, cfg: ArucoBoardConfig
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, list]]:
    """
    Detect ArUco markers and compute their object points.
    Returns:
        corners: (N*4, 2) — all marker corners (as 2D points, order: marker0-pt0..pt3, marker1-pt0..pt3...)
        ids:     (N, 1)   — marker ids
        obj_pts: (N*4, 3) — corresponding 3D object points for each corner
    """
    logger.debug("Detecting ArUco markers")
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detector = cv2.aruco.ArucoDetector(
            cfg.dictionary, cv2.aruco.DetectorParameters()
        )
        marker_corners, ids, _ = detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            logger.debug("No ArUco markers found")
            return None

        # For each detected marker, get its 3D corner coordinates in the board frame
        all_corners = []
        all_obj_pts = []
        for i, id_ in enumerate(ids.flatten()):
            # By convention: top-left, top-right, bottom-right, bottom-left
            # For 1x1 GridBoard, marker at (0, 0)
            # obj is the coordinates of the four marker corners in the pattern coord's system
            obj = np.array(
                [
                    [0, 0, 0],
                    [cfg.marker_length, 0, 0],
                    [cfg.marker_length, cfg.marker_length, 0],
                    [0, cfg.marker_length, 0],
                ],
                dtype=np.float32,
            )
            all_obj_pts.append(obj)
            all_corners.append(marker_corners[i].reshape(4, 2))
        # Stack everything for solvePnP
        corners = np.vstack(all_corners)  # (N*4,2)
        obj_points = np.vstack(all_obj_pts)  # (N*4,3)
        logger.debug(f"Aruco corners sample: {corners[:2].tolist()}")
        # Log first correspondence for traceability
        logger.info(
            f"[find_aruco] Corner #0 pixel={corners[0].tolist()}, object={obj_points[0].tolist()}"
        )
        return corners, ids, obj_points, marker_corners
    except Exception as exc:
        logger.error(f"Aruco detection failed: {exc}")
        ErrorTracker.report(exc)
        return None


_CHARUCO_CACHE: dict[tuple, np.ndarray] = {}


def _compute_corr_map(board: cv2.aruco.CharucoBoard) -> dict[int, int]:
    """
    Compute mapping between charuco ID order and expected index order.
    """
    dictionary = board.getDictionary()
    img = cv2.aruco.drawPlanarBoard(board, (1000, 1000), marginSize=10, borderBits=1)
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector_params = cv2.aruco.DetectorParameters()
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
        gray, dictionary, parameters=detector_params
    )

    if marker_ids is None or len(marker_ids) == 0:
        raise ValueError("Cannot detect markers on rendered board")

    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        markerCorners=marker_corners,
        markerIds=marker_ids,
        image=gray,
        board=board,
    )

    if not retval or charuco_corners is None or charuco_ids is None:
        raise ValueError("Cannot interpolate charuco corners on rendered board")

    corr = {}
    ref_corners = board.getChessboardCorners()
    ref_corners_2d = ref_corners[:, :2]
    # Match detected ids to expected corner order
    for idx, id_ in enumerate(charuco_ids.flatten()):
        detected_corner = charuco_corners[idx].flatten()
        distances = np.linalg.norm(ref_corners_2d - detected_corner, axis=1)
        ref_id = int(np.argmin(distances))
        corr[id_] = ref_id

    return corr


def detect_charuco(
    img: np.ndarray,
    cfg: CharucoBoardConfig,
    *,
    enforce_ascending_ids: bool = True,
    visualize: bool = False,
) -> Optional[Detection]:
    """
    Detect Charuco board corners using detectMarkers + interpolateCornersCharuco.
    Optionally remaps IDs and shows visualization.
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        board = cfg.create()
        dictionary = cfg.dictionary

        # Step 1: Detect ArUco markers
        detector_params = cv2.aruco.DetectorParameters()
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, dictionary, parameters=detector_params
        )

        if ids is None or len(ids) == 0:
            logger.debug("No ArUco markers detected")
            return None

        # Step 2: Interpolate Charuco corners
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=board,
        )

        if (
            not retval
            or charuco_corners is None
            or charuco_ids is None
            or len(charuco_corners) < 4
        ):
            logger.debug("Charuco board interpolation failed or not enough corners")
            return None

        # Step 3: Remap IDs (if requested)
        key = (
            cfg.squares,
            cfg.square_size,
            cfg.marker_size,
            tuple(dictionary.bytesList.flatten()),
        )

        if key not in _CHARUCO_CACHE:
            corr = _compute_corr_map(board)
            _CHARUCO_CACHE[key] = corr
        else:
            corr = _CHARUCO_CACHE[key]

        ids_remapped = np.array([[corr[idx[0]]] for idx in charuco_ids], dtype=np.int32)

        obj_points = board.getChessboardCorners()[ids_remapped.flatten()].copy()

        logger.debug(f"Charuco IDs: {charuco_ids.ravel().tolist()}")
        logger.debug(f"Remapped IDs: {ids_remapped.ravel().tolist()}")
        logger.debug(
            f"Charuco corners sample: {charuco_corners.reshape(-1, 2)[:2].tolist()}"
        )

        if visualize:
            vis_img = cv2.aruco.drawDetectedMarkers(img.copy(), corners, ids)
            vis_img = cv2.aruco.drawDetectedCornersCharuco(
                vis_img, charuco_corners, charuco_ids
            )
            cv2.imshow("Charuco Detection", cv2.resize(vis_img, (1280, 720)))
            cv2.waitKey(0)

        return Detection(
            corners=charuco_corners,
            obj_points=obj_points,
            ids=charuco_ids,
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
    vis = draw_charuco(img, corners, ids, marker_corners, marker_ids)
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
