"""Hand-eye calibration validation and error analysis."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils.config import Config
from utils.logger import Logger, LoggerType
from utils.io import load_camera_params
from vision.transform import TransformUtils

from .marker_detection import MarkerDetector


class CameraInterface:
    """Camera interface for a color image."""

    def capture(self) -> np.ndarray:
        raise NotImplementedError

    def get_intrinsics(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class RobotInterface:
    """Robot interface exposing pose control."""

    def move_to(self, pose: np.ndarray) -> None:
        raise NotImplementedError


@dataclass
class ValidationConfig:
    results_dir: Path = Path("validation/results")
    aruco_dict: str = "DICT_4X4_100"
    squares_x: int = 5
    squares_y: int = 7
    square_length: float = 0.035
    marker_length: float = 0.026
    pixel_threshold: float = 5.0


@dataclass
class HandEyeValidator:
    camera: CameraInterface
    robot: RobotInterface
    T_base_cam: np.ndarray
    K: np.ndarray
    dist: np.ndarray
    detector: MarkerDetector
    logger: LoggerType = field(
        default_factory=lambda: Logger.get_logger("validation.validator")
    )

    def _project_world_point(self, point: np.ndarray) -> np.ndarray:
        Tu = TransformUtils(self.logger)
        point_cam = Tu.world_to_camera(point[None, :], self.T_base_cam)[0]
        rvec = np.zeros((3, 1))
        tvec = np.zeros((3, 1))
        pts_2d, _ = cv2.projectPoints(point_cam, rvec, tvec, self.K, self.dist)
        return pts_2d.reshape(-1)

    def validate_pose(self, pose: np.ndarray) -> dict[str, float]:
        self.robot.move_to(pose)
        img = self.camera.capture()
        detected, _ = self.detector.detect(img)
        if detected is None:
            return {"pixel_error": float("nan")}
        proj = self._project_world_point(pose[:3, 3])
        err = float(np.linalg.norm(proj - detected))
        self.logger.info(f"Pixel error: {err:.2f}")
        return {"pixel_error": err}


@dataclass
class DatasetAnalyzer:
    validator: HandEyeValidator
    poses: Iterable[np.ndarray]
    pixel_threshold: float = 5.0
    out_dir: Path = Path("validation/results")

    def run(self) -> list[float]:
        errors: list[float] = []
        for pose in self.poses:
            res = self.validator.validate_pose(pose)
            if not np.isnan(res["pixel_error"]):
                errors.append(res["pixel_error"])
        if not errors:
            self.validator.logger.warning("No valid frames for analysis")
            return []
        self._plot(errors)
        self._save_outliers(errors)
        return errors

    def _plot(self, errors: list[float]) -> None:
        sns.histplot(errors, kde=True)
        plt.xlabel("Pixel error")
        plt.title("Hand-Eye Validation Error")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.out_dir / "error_hist.png")
        plt.close()

    def _save_outliers(self, errors: list[float]) -> None:
        arr = np.array(errors)
        mask = arr > self.pixel_threshold
        outliers = np.nonzero(mask)[0]
        np.savetxt(self.out_dir / "outliers.txt", outliers, fmt="%d")
        self.validator.logger.info(f"Outliers: {outliers.tolist()}")


def get_board_and_detector() -> tuple[cv2.aruco_CharucoBoard, MarkerDetector]:
    """Build CharucoBoard and detector from config."""
    cfg = Config.get("validation")
    squares_x = cfg.get("squares_x", 5)
    squares_y = cfg.get("squares_y", 7)
    square_length = cfg.get("square_length", 0.035)
    marker_length = cfg.get("marker_length", 0.026)
    dict_name = cfg.get("aruco_dict", "4X4_100")
    if not dict_name.startswith("DICT_"):
        dict_attr = f"DICT_{dict_name}"
    else:
        dict_attr = dict_name
    aruco_id = getattr(cv2.aruco, dict_attr)
    dictionary = cv2.aruco.getPredefinedDictionary(aruco_id)
    detector = MarkerDetector(dict_name)
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y), square_length, marker_length, dictionary
    )

    return board, detector


def load_default_validator(
    camera: CameraInterface, robot: RobotInterface
) -> HandEyeValidator:
    """Create HandEyeValidator using config and calibration files."""
    Config.load()
    cfg = Config.get("validation")
    results_dir = Path(cfg.get("results_dir", "validation/results"))
    marker_length = float(cfg.get("marker_length", 0.026))
    charuco_xml = Config.get("charuco.xml_file", "calibration/results/charuco_cam.xml")
    handeye_npz = os.path.join(
        Config.get("handeye.calib_output_dir", "calibration/results"),
        "handeye_TSAI.npz",
    )
    K, dist = load_camera_params(charuco_xml)
    if os.path.isfile(handeye_npz):
        data = np.load(handeye_npz)
        R, t = data["R"], data["t"]
    else:
        R, t = np.eye(3), np.zeros(3)
    Tu = TransformUtils()
    T_base_cam = Tu.build_transform(R, t)
    _, detector = get_board_and_detector()
    return HandEyeValidator(camera, robot, T_base_cam, K, dist, detector)


__all__ = [
    "ValidationConfig",
    "HandEyeValidator",
    "DatasetAnalyzer",
    "get_board_and_detector",
    "load_default_validator",
]
