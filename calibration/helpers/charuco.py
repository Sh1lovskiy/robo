# calibration/charuco.py
"""Charuco board calibration utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Mapping

import cv2
import numpy as np

from utils.logger import Logger, LoggerType
from calibration.helpers.pose_utils import (
    ExtractionParams,
    ExtractionResult,
    extract_charuco_poses as _extract_charuco_poses,
)

CHARUCO_DICT_MAP = {
    "4X4_100": cv2.aruco.DICT_4X4_100,
    "5X5_50": cv2.aruco.DICT_5X5_50,
    "5X5_100": cv2.aruco.DICT_5X5_100,
}


class CalibrationSaver:
    """
    Strategy interface for saving calibration results.

    Implementations write camera matrix and distortion coefficients to a
    destination such as an XML file or plain text file.
    """

    def save(
        self, filename: str, camera_matrix: np.ndarray, dist_coeffs: np.ndarray
    ) -> None:
        raise NotImplementedError


class OpenCVXmlSaver(CalibrationSaver):
    def save(
        self, filename: str, camera_matrix: np.ndarray, dist_coeffs: np.ndarray
    ) -> None:
        """
        Write calibration parameters to an OpenCV XML/YAML file.

        Args:
            filename: Destination path for the file.
            camera_matrix: 3x3 intrinsic matrix ``K``.
            dist_coeffs: Distortion coefficients ``(k1, k2, p1, p2, k3)``.

        The file is created along with any missing parent directories using
        :func:`cv2.FileStorage` for compatibility with OpenCV tools.
        """
        dir_ = os.path.dirname(filename)
        if dir_ and not os.path.exists(dir_):
            os.makedirs(dir_, exist_ok=True)
        fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
        fs.write("camera_matrix", camera_matrix)
        fs.write("dist_coeffs", dist_coeffs)
        fs.release()


class TextSaver(CalibrationSaver):
    def save(
        self, filename: str, camera_matrix: np.ndarray, dist_coeffs: np.ndarray
    ) -> None:
        """
        Save calibration to a plain text format.

        Args:
            filename: Text file destination.
            camera_matrix: Camera intrinsic matrix to write.
            dist_coeffs: Distortion coefficients vector.

        The arrays are saved using :func:`numpy.savetxt` so they can easily be
        inspected or imported from other tools.
        """
        dir_ = os.path.dirname(filename)
        if dir_ and not os.path.exists(dir_):
            os.makedirs(dir_, exist_ok=True)
        with open(filename, "w") as f:
            np.savetxt(f, camera_matrix, fmt="%.8f", header="camera_matrix")
            np.savetxt(f, dist_coeffs, fmt="%.8f", header="dist_coeffs")


def load_board(
    cfg: Mapping[str, float | str],
) -> tuple[cv2.aruco_CharucoBoard, cv2.aruco_Dictionary]:
    """
    Create a Charuco board from configuration.

    Args:
        cfg: Mapping with keys ``squares_x``, ``squares_y``, ``square_length``
            and ``marker_length`` describing the board geometry. ``aruco_dict``
            selects the ArUco dictionary name.

    Returns:
        Tuple of the OpenCV Charuco board object and its dictionary.
    """
    dict_name = str(cfg.get("aruco_dict", "5X5_100"))
    if dict_name not in CHARUCO_DICT_MAP:
        raise ValueError(f"Unknown ArUco dictionary: {dict_name}")
    squares_x = int(cfg.get("squares_x", 5))
    squares_y = int(cfg.get("squares_y", 7))
    square_len = float(cfg.get("square_length", 0.033))
    marker_len = float(cfg.get("marker_length", 0.025))
    dictionary = cv2.aruco.getPredefinedDictionary(CHARUCO_DICT_MAP[dict_name])
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y), square_len, marker_len, dictionary
    )
    return board, dictionary


def extract_charuco_poses(
    images_dir: str,
    board: cv2.aruco_CharucoBoard,
    dictionary: cv2.aruco_Dictionary,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    *,
    min_corners: int,
    visualize: bool,
    analyze_corners: bool,
    outlier_std: float,
    logger: LoggerType | None = None,
) -> tuple[
    tuple[list[np.ndarray], list[np.ndarray], list[str], list[str]],
    tuple[dict[str, dict[str, np.ndarray]], list[int]],
]:
    """
    Extract board poses from a folder of images.

    Args:
        images_dir: Directory containing image files named ``*_rgb.png`` or
            similar.
        board: Charuco board instance describing marker layout.
        dictionary: ArUco dictionary used for marker detection.
        camera_matrix: Intrinsic camera matrix.
        dist_coeffs: Distortion coefficients.
        min_corners: Minimum required corners to accept a detection.
        visualize: If ``True`` show intermediate OpenCV windows while
            processing images.
        analyze_corners: Compute statistics of detected corners when ``True``.
        outlier_std: Frames with corner position deviation beyond this many
            standard deviations are discarded.
        logger: Optional project logger.

    Returns:
        Tuple ``(poses, (stats, outliers))`` where ``poses`` contains lists of
        rotation matrices, translation vectors and image file paths. ``stats`` is
        a dictionary of corner statistics and ``outliers`` lists the indices of
        rejected frames.
    """
    params = ExtractionParams(
        min_corners=min_corners,
        visualize=visualize,
        analyze_corners=analyze_corners,
        outlier_std=outlier_std,
    )
    result: ExtractionResult = _extract_charuco_poses(
        images_dir,
        board,
        dictionary,
        camera_matrix,
        dist_coeffs,
        logger=logger,
        params=params,
    )
    poses = (
        result.rotations,
        result.translations,
        result.valid_paths,
        result.all_paths,
    )
    stats = result.stats
    return poses, (stats, result.outliers)


@dataclass
class CharucoCalibrator:
    """
    Charuco board calibration using OpenCV.

    The class accumulates detected corners from multiple images and then calls
    :func:`cv2.aruco.calibrateCameraCharuco` to compute the camera matrix and
    distortion coefficients.
    """

    board: cv2.aruco_CharucoBoard
    dictionary: cv2.aruco_Dictionary
    logger: LoggerType = field(
        default_factory=lambda: Logger.get_logger("calibration.charuco")
    )
    all_corners: List[np.ndarray] = field(default_factory=list, init=False)
    all_ids: List[np.ndarray] = field(default_factory=list, init=False)
    img_size: tuple[int, int] | None = field(default=None, init=False)

    def add_frame(self, img: np.ndarray) -> bool:
        """
        Detect Charuco markers in ``img`` and store them for calibration.

        Args:
            img: BGR image from which to detect the board.

        Returns:
            ``True`` if enough corners were found and saved, ``False`` otherwise.
        """
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
        """
        Run OpenCV Charuco calibration on the collected frames.

        Returns:
            Dictionary with RMS reprojection error, camera matrix, distortion
            coefficients and per-frame vectors ``rvecs`` and ``tvecs``.
        """
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

    def save(
        self,
        saver: CalibrationSaver,
        filename: str,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> None:
        """
        Persist calibration with the provided ``saver`` strategy.

        Args:
            saver: Object implementing :class:`CalibrationSaver`.
            filename: Path where calibration will be written.
            camera_matrix: Camera intrinsics to save.
            dist_coeffs: Distortion coefficients to save.
        """
        saver.save(filename, camera_matrix, dist_coeffs)
        self.logger.info(
            f"Calibration saved with {saver.__class__.__name__} to {filename}"
        )
