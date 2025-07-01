"""
Charuco board calibration utilities.

This module wraps OpenCV's Charuco detection and calibration routines.  It
provides helpers for loading board definitions, accumulating corner detections
across images and computing camera intrinsics.  The functions are intended for
offline calibration workflows and may be reused programmatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import List, Mapping

import cv2
import numpy as np

from utils.logger import Logger, LoggerType
from utils.lmdb_storage import LmdbStorage
from .helpers.validation_utils import euler_to_matrix

CHARUCO_DICT_MAP = {
    "4X4_100": cv2.aruco.DICT_4X4_100,
    "5X5_50": cv2.aruco.DICT_5X5_50,
    "5X5_100": cv2.aruco.DICT_5X5_100,
}


def load_board(
    cfg: Mapping[str, float | str],
) -> tuple[cv2.aruco_CharucoBoard, cv2.aruco_Dictionary]:
    """Create a Charuco board from configuration."""
    dict_name = str(cfg.get("aruco_dict", "5X5_100"))
    if dict_name not in CHARUCO_DICT_MAP:
        raise ValueError(f"Unknown ArUco dictionary: {dict_name}")
    squares_x = int(cfg.get("squares_x", 5))
    squares_y = int(cfg.get("squares_y", 8))
    square_len = float(cfg.get("square_length", 0.035))
    marker_len = float(cfg.get("marker_length", 0.026))
    dictionary = cv2.aruco.getPredefinedDictionary(CHARUCO_DICT_MAP[dict_name])
    board = cv2.aruco.CharucoBoard(
        (squares_y, squares_x), square_len, marker_len, dictionary
    )
    return board, dictionary


@dataclass
class CharucoCalibrator:
    """Accumulate Charuco detections and solve for camera intrinsics."""

    board: cv2.aruco_CharucoBoard
    dictionary: cv2.aruco_Dictionary
    logger: LoggerType = field(
        default_factory=lambda: Logger.get_logger("calibration.charuco")
    )
    all_corners: List[np.ndarray] = field(default_factory=list, init=False)
    all_ids: List[np.ndarray] = field(default_factory=list, init=False)
    img_size: tuple[int, int] | None = field(default=None, init=False)

    def add_frame(self, img: np.ndarray) -> bool:
        """Detect markers in an image and store the corners if successful."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        res = cv2.aruco.detectMarkers(gray, self.dictionary)
        if len(res[0]) > 0:
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                res[0], res[1], gray, self.board
            )
            if (
                charuco_corners is not None
                and charuco_ids is not None
                and len(charuco_corners) > 3
            ):
                self.all_corners.append(charuco_corners)
                self.all_ids.append(charuco_ids)
                self.img_size = gray.shape[::-1]
                self.logger.debug(f"Frame added, ids found: {len(charuco_ids)}")
                return True
        self.logger.warning("No Charuco corners found in frame")
        return False

    def calibrate(self) -> dict[str, np.ndarray | float]:
        """Return calibration results after multiple calls to :meth:`add_frame`."""
        assert self.img_size is not None, "No frames added."
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = (
            cv2.aruco.calibrateCameraCharuco(
                self.all_corners, self.all_ids, self.board, self.img_size, None, None
            )
        )
        self.logger.info(f"Charuco calibration RMS: {ret:.6f}")
        return dict(
            rms=ret,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            rvecs=rvecs,
            tvecs=tvecs,
        )


@dataclass
class ExtractionParams:
    """
    Parameters controlling Charuco pose extraction.

    Attributes:
        min_corners: Minimum number of detected corners required to accept a
            frame.
        visualize: If ``True``, show detections as they are processed.
        analyze_corners: Enable detailed corner statistics gathering.
        outlier_std: Z-score threshold for outlier rejection when analyzing
            corners.
    """

    min_corners: int = 4
    visualize: bool = False
    analyze_corners: bool = False
    outlier_std: float = 2.0


@dataclass
class ExtractionResult:
    """Container for pose extraction results."""

    rotations: List[np.ndarray]
    translations: List[np.ndarray]
    valid_paths: List[str]
    all_paths: List[str]
    stats: dict[str, dict[str, np.ndarray]]
    outliers: List[int]


def load_camera_params(filename: str) -> tuple[np.ndarray, np.ndarray]:
    """Load camera matrix and distortion coefficients from OpenCV XML/YAML."""
    fs = cv2.FileStorage(str(filename), cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("dist_coeffs").mat()
    fs.release()
    return camera_matrix, dist_coeffs


def save_camera_params_xml(
    filename: str, camera_matrix: np.ndarray, dist_coeffs: np.ndarray
) -> None:
    """Save camera calibration to an XML/YAML file."""
    fs = cv2.FileStorage(str(filename), cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", camera_matrix)
    fs.write("dist_coeffs", dist_coeffs)
    fs.release()


def save_camera_params_txt(
    filename: str,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rms: float | None = None,
) -> None:
    """Save camera calibration to a plain text file."""
    with open(filename, "w") as f:
        if rms is not None:
            f.write(f"RMS Error: {rms:.6f}\n")
        f.write("camera_matrix =\n")
        np.savetxt(f, camera_matrix, fmt="%.10f")
        f.write("dist_coeffs =\n")
        np.savetxt(f, dist_coeffs.reshape(1, -1), fmt="%.10f")


# Pose extraction helpers ---------------------------------------------------


def _estimate_pose(
    img: np.ndarray,
    board: cv2.aruco_CharucoBoard,
    dictionary: cv2.aruco_Dictionary,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    params: ExtractionParams,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Internal helper to estimate board pose from an image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    if ids is None or len(ids) < max(6, params.min_corners):
        return None
    _, char_corners, char_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, board
    )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000000, 0.00001)
    corners = cv2.cornerSubPix(gray, char_corners, (15, 15), (-1, -1), criteria)
    if (
        corners is None
        or char_ids is None
        or len(char_ids) < max(6, params.min_corners)
    ):
        return None
    rvec_init = np.zeros((3, 1), dtype=np.float64)
    tvec_init = np.zeros((3, 1), dtype=np.float64)
    retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        corners,
        char_ids,
        board,
        camera_matrix,
        dist_coeffs,
        rvec_init,
        tvec_init,
    )
    if not retval:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.flatten()


def extract_charuco_poses(
    images_dir: str,
    board: cv2.aruco_CharucoBoard,
    dictionary: cv2.aruco_Dictionary,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    *,
    logger: LoggerType | None = None,
    params: ExtractionParams | None = None,
) -> ExtractionResult:
    """Process a directory of images and estimate board poses for each."""
    params = params or ExtractionParams()
    logger = logger or Logger.get_logger("calibration.pose_extractor")
    image_paths = sorted(
        [
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )
    Rs: List[np.ndarray] = []
    ts: List[np.ndarray] = []
    valid_paths: List[str] = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            logger.warning("Cannot read image: %s", img_path)
            continue
        pose = _estimate_pose(
            img, board, dictionary, camera_matrix, dist_coeffs, params
        )
        if pose is None:
            logger.warning(
                "Charuco pose not found for image: %s", os.path.basename(img_path)
            )
            continue
        R, t = pose
        Rs.append(R)
        ts.append(t)
        valid_paths.append(img_path)
        if params.visualize:
            cv2.aruco.drawDetectedMarkers(img, None, None)
            cv2.imshow("charuco pose", img)
            cv2.waitKey(50)
    if params.visualize:
        cv2.destroyAllWindows()
    stats = {}
    outliers: List[int] = []
    return ExtractionResult(Rs, ts, valid_paths, image_paths, stats, outliers)
