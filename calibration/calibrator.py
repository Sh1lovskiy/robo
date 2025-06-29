from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Mapping

import cv2
import numpy as np

from utils.logger import Logger, LoggerType
from utils.lmdb_storage import LmdbStorage

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


class HandEyeSaver:
    """Strategy interface for saving Hand-Eye calibration results."""

    def save(self, filename: str, R: np.ndarray, t: np.ndarray) -> None:
        raise NotImplementedError


class NPZHandEyeSaver(HandEyeSaver):
    def save(self, filename: str, R: np.ndarray, t: np.ndarray) -> None:
        np.savez(filename, R=R, t=t)


class TxtHandEyeSaver(HandEyeSaver):
    def save(self, filename: str, R: np.ndarray, t: np.ndarray) -> None:
        with open(filename, "w") as f:
            f.write("R =\n")
            np.savetxt(f, R, fmt="%.8f")
            f.write("t =\n")
            np.savetxt(f, t.reshape(1, -1), fmt="%.8f")


class DBHandEyeSaver(HandEyeSaver):
    """Store calibration results in an ``LmdbStorage`` backend."""

    def __init__(self, storage: LmdbStorage) -> None:
        self.storage = storage

    def save(self, filename: str, R: np.ndarray, t: np.ndarray) -> None:
        self.storage.put_array(f"{filename}:R", R)
        self.storage.put_array(f"{filename}:t", t)


class HandEyeCalibrator:
    """Hand-Eye calibration wrapper using OpenCV algorithms."""

    METHODS = {
        name.replace("CALIB_HAND_EYE_", ""): getattr(cv2, name)
        for name in dir(cv2)
        if name.startswith("CALIB_HAND_EYE_")
    }

    def __init__(self, logger: LoggerType | None = None) -> None:
        self.R_gripper2base: list[np.ndarray] = []
        self.t_gripper2base: list[np.ndarray] = []
        self.R_target2cam: list[np.ndarray] = []
        self.t_target2cam: list[np.ndarray] = []
        self.logger = logger or Logger.get_logger("calibration.handeye")

    def add_sample(
        self,
        R_gripper: np.ndarray,
        t_gripper: np.ndarray,
        R_target: np.ndarray,
        t_target: np.ndarray,
    ) -> None:
        self.R_gripper2base.append(R_gripper)
        self.t_gripper2base.append(t_gripper)
        self.R_target2cam.append(R_target)
        self.t_target2cam.append(t_target)
        self.logger.debug("Added Hand-Eye sample")

    def calibrate(self, method: str = "TSAI") -> tuple[np.ndarray, np.ndarray]:
        """Calibrate using the specified method."""
        assert self.R_gripper2base, "No samples for calibration"
        key = method.upper()
        if key not in self.METHODS:
            raise ValueError(
                f"Unknown method '{method}'. Available: {list(self.METHODS.keys())}"
            )
        R, t = cv2.calibrateHandEye(
            self.R_gripper2base,
            self.t_gripper2base,
            self.R_target2cam,
            self.t_target2cam,
            method=self.METHODS[key],
        )
        self.logger.info(f"Hand-Eye calibration ({method}) done")
        return R, t

    def calibrate_all(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Run calibration with all available methods."""
        results = {}
        for name, code in self.METHODS.items():
            try:
                R, t = cv2.calibrateHandEye(
                    self.R_gripper2base,
                    self.t_gripper2base,
                    self.R_target2cam,
                    self.t_target2cam,
                    method=code,
                )
                results[name] = (R, t)
                self.logger.info(f"Hand-Eye calibration ({name}) done")
            except Exception as e:
                self.logger.error(f"Hand-Eye calibration failed for {name}: {e}")
        return results


__all__ = [
    "CHARUCO_DICT_MAP",
    "load_board",
    "CharucoCalibrator",
    "HandEyeCalibrator",
    "HandEyeSaver",
    "NPZHandEyeSaver",
    "TxtHandEyeSaver",
    "DBHandEyeSaver",
]

