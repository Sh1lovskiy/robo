# calibration/charuco.py
"""Charuco board calibration utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Mapping

import cv2
import numpy as np

from utils.logger import Logger, LoggerType
from .pose_extractor import (
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
    """Strategy interface for saving calibration results."""

    def save(
        self, filename: str, camera_matrix: np.ndarray, dist_coeffs: np.ndarray
    ) -> None:
        raise NotImplementedError


class OpenCVXmlSaver(CalibrationSaver):
    def save(
        self, filename: str, camera_matrix: np.ndarray, dist_coeffs: np.ndarray
    ) -> None:
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
        dir_ = os.path.dirname(filename)
        if dir_ and not os.path.exists(dir_):
            os.makedirs(dir_, exist_ok=True)
        with open(filename, "w") as f:
            np.savetxt(f, camera_matrix, fmt="%.8f", header="camera_matrix")
            np.savetxt(f, dist_coeffs, fmt="%.8f", header="dist_coeffs")


def load_board(
    cfg: Mapping[str, float | str],
) -> tuple[cv2.aruco_CharucoBoard, cv2.aruco_Dictionary]:
    """Create a Charuco board from configuration."""

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
    """Backward compatible wrapper for :func:`pose_extractor.extract_charuco_poses`."""

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
    """Charuco board calibration using OpenCV."""

    board: cv2.aruco_CharucoBoard
    dictionary: cv2.aruco_Dictionary
    logger: LoggerType = field(
        default_factory=lambda: Logger.get_logger("calibration.charuco")
    )
    all_corners: List[np.ndarray] = field(default_factory=list, init=False)
    all_ids: List[np.ndarray] = field(default_factory=list, init=False)
    img_size: tuple[int, int] | None = field(default=None, init=False)

    def add_frame(self, img: np.ndarray) -> bool:
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
        saver.save(filename, camera_matrix, dist_coeffs)
        self.logger.info(
            f"Calibration saved with {saver.__class__.__name__} to {filename}"
        )
