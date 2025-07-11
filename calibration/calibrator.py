from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from utils import JSONPoseLoader, load_camera_params, handeye as handeye_cfg, paths
from utils.logger import Logger, LoggerType

from .pattern import CalibrationPattern
from .utils import save_camera_params, save_transform, timestamp


@dataclass
class IntrinsicResult:
    """Result of an intrinsic calibration."""

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    output_base: Path


@dataclass
class IntrinsicCalibrator:
    """Compute camera intrinsics using a generic calibration pattern."""

    logger: LoggerType = field(
        default_factory=lambda: Logger.get_logger("calibration.intrinsic")
    )

    def calibrate(
        self, images: List[Path], pattern: CalibrationPattern
    ) -> IntrinsicResult:
        """Calibrate camera intrinsics from ``images`` using ``pattern``."""

        img_size: Tuple[int, int] | None = None
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                self.logger.error(f"Failed to read {img_path}")
                continue
            img_size = (img.shape[1], img.shape[0])
            if pattern.detect(img) is None:
                self.logger.warning(f"Pattern not detected in {img_path}")
        if not pattern.detections or img_size is None:
            raise RuntimeError("No valid detections for intrinsic calibration")
        K, dist, rms = pattern.calibrate_camera(img_size)
        out_base = paths.RESULTS_DIR / f"camera_{timestamp()}"
        save_camera_params(out_base, K, dist, rms)
        self.logger.info(f"Intrinsics saved to {out_base} (RMS={rms:.6f})")
        return IntrinsicResult(K, dist, out_base)


@dataclass
class HandEyeCalibrator:
    """Solve the robot-to-camera transform using any pattern."""

    logger: LoggerType = field(
        default_factory=lambda: Logger.get_logger("calibration.handeye")
    )

    def calibrate(
        self,
        poses_file: Path,
        images: List[Path],
        pattern: CalibrationPattern,
        intrinsics: tuple[np.ndarray, np.ndarray],
    ) -> Path:
        """Compute hand-eye transformation from robot poses and images."""

        K, dist = intrinsics
        robot_Rs, robot_ts = JSONPoseLoader.load_poses(str(poses_file))
        target_Rs: List[np.ndarray] = []
        target_ts: List[np.ndarray] = []
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                self.logger.error(f"Failed to read {img_path}")
                continue
            detection = pattern.detect(img)
            if detection is None:
                self.logger.warning(f"Pattern not detected in {img_path}")
                continue
            pose = pattern.estimate_pose(detection, K, dist)
            if pose is None:
                self.logger.warning(f"Pose estimation failed for {img_path}")
                continue
            R, t = pose
            target_Rs.append(R)
            target_ts.append(t)
        if not target_Rs:
            raise RuntimeError("No valid detections for hand-eye calibration")
        method = getattr(cv2, handeye_cfg.method, cv2.CALIB_HAND_EYE_TSAI)
        R_cam2tool, t_cam2tool = cv2.calibrateHandEye(
            robot_Rs, robot_ts, target_Rs, target_ts, method=method
        )
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_cam2tool
        T[:3, 3] = t_cam2tool
        out_base = paths.RESULTS_DIR / f"handeye_{timestamp()}"
        save_transform(out_base, T)
        self.logger.info(f"Hand-eye result saved to {out_base}")
        pattern.clear()
        return out_base
