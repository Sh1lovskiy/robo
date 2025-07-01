"""Hand-eye calibration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from utils.logger import Logger, LoggerType
from utils.lmdb_storage import LmdbStorage


class HandEyeSaver:
    """Strategy interface for saving hand-eye calibration results."""

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


@dataclass
class HandEyeCalibrator:
    """Wrapper over OpenCV hand-eye algorithms."""

    logger: LoggerType | None = None

    def __post_init__(self) -> None:
        self.logger = self.logger or Logger.get_logger("calibration.handeye")
        self.R_gripper2base: List[np.ndarray] = []
        self.t_gripper2base: List[np.ndarray] = []
        self.R_target2cam: List[np.ndarray] = []
        self.t_target2cam: List[np.ndarray] = []

    METHODS = {
        name.replace("CALIB_HAND_EYE_", ""): getattr(cv2, name)
        for name in dir(cv2)
        if name.startswith("CALIB_HAND_EYE_")
    }

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
        """Run calibration using the specified method."""
        assert self.R_gripper2base, "No samples for calibration"
        key = method.upper()
        if key not in self.METHODS:
            raise ValueError(f"Unknown method '{method}'. Available: {list(self.METHODS.keys())}")
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
        """Run all available calibration methods."""
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

