from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from utils import (
    JSONPoseLoader,
    load_camera_params,
    handeye as handeye_cfg,
    paths,
)
from utils.logger import Logger, LoggerType
from .detectors import Checkerboard, CharucoBoardCfg, find_checkerboard, find_charuco
from .utils import save_camera_params, save_transform, timestamp


@dataclass
class HandEyeCalibrator:
    """Solve the robot to camera transform."""

    logger: LoggerType = field(
        default_factory=lambda: Logger.get_logger("calibration.handeye")
    )

    def calibrate(self, poses_file: Path, images: List[Path]) -> Path:
        board_cfg = CharucoBoardCfg(
            squares=handeye_cfg.square_numbers,
            square_size=handeye_cfg.square_length,
            marker_size=handeye_cfg.marker_length,
            dictionary=cv2.aruco.getPredefinedDictionary(
                handeye_cfg.CHARUCO_DICT_MAP[handeye_cfg.aruco_dict]
            ),
        )
        K, dist = load_camera_params(handeye_cfg.charuco_xml)
        robot_Rs, robot_ts = JSONPoseLoader.load_poses(str(poses_file))
        target_Rs: List[np.ndarray] = []
        target_ts: List[np.ndarray] = []
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                self.logger.error(f"Failed to read {img_path}")
                continue
            res = find_charuco(img, board_cfg)
            if res is None:
                self.logger.warning(f"Charuco not detected in {img_path}")
                continue
            corners, ids = res
            board = board_cfg.create()
            ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                corners, ids, board, K, dist
            )
            if not ok:
                self.logger.warning(f"Pose estimation failed for {img_path}")
                continue
            R, _ = cv2.Rodrigues(rvec)
            target_Rs.append(R)
            target_ts.append(tvec.reshape(3))
        if not target_Rs:
            raise RuntimeError("No valid detections for hand-eye calibration")
        R_cam2tool, t_cam2tool = cv2.calibrateHandEye(
            robot_Rs,
            robot_ts,
            target_Rs,
            target_ts,
            method=getattr(cv2, handeye_cfg.method, cv2.CALIB_HAND_EYE_TSAI),
        )
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_cam2tool
        T[:3, 3] = t_cam2tool
        out_base = paths.RESULTS_DIR / f"handeye_{timestamp()}"
        save_transform(out_base, T)
        self.logger.info(f"Hand-eye result saved to {out_base}")
        return out_base


@dataclass
class IntrinsicCalibrator:
    """Compute camera intrinsic parameters from checkerboard images."""

    board_size: Tuple[int, int] = (7, 6)
    square_size: float = 0.02
    logger: LoggerType = field(
        default_factory=lambda: Logger.get_logger("calibration.intrinsic")
    )

    def calibrate(self, images: List[Path]) -> Path:
        board = Checkerboard(self.board_size, self.square_size)
        obj_points: List[np.ndarray] = []
        img_points: List[np.ndarray] = []
        img_shape: Tuple[int, int] | None = None
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                self.logger.error(f"Failed to read {img_path}")
                continue
            res = find_checkerboard(img, board)
            if res is None:
                self.logger.warning(f"Checkerboard not found in {img_path}")
                continue
            corners, objp = res
            obj_points.append(objp)
            img_points.append(corners)
            img_shape = img.shape[:2]
        if not obj_points or img_shape is None:
            raise RuntimeError("No valid checkerboard detections")
        ret, K, dist, _, _ = cv2.calibrateCamera(
            obj_points,
            img_points,
            img_shape[::-1],
            None,
            None,
        )
        out_base = paths.RESULTS_DIR / f"camera_{timestamp()}"
        save_camera_params(out_base, K, dist, ret)
        self.logger.info(f"Intrinsics saved to {out_base}")
        return out_base


@dataclass
class CharucoCalibrator:
    """Calibrate camera using Charuco board images."""

    logger: LoggerType = field(
        default_factory=lambda: Logger.get_logger("calibration.charuco")
    )

    def calibrate(self, images: List[Path]) -> Path:
        board_cfg = CharucoBoardCfg(
            squares=handeye_cfg.square_numbers,
            square_size=handeye_cfg.square_length,
            marker_size=handeye_cfg.marker_length,
            dictionary=cv2.aruco.getPredefinedDictionary(
                handeye_cfg.CHARUCO_DICT_MAP[handeye_cfg.aruco_dict]
            ),
        )
        board = board_cfg.create()
        all_corners: List[np.ndarray] = []
        all_ids: List[np.ndarray] = []
        img_size: Tuple[int, int] | None = None
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                self.logger.error(f"Failed to read {img_path}")
                continue
            res = find_charuco(img, board_cfg)
            if res is None:
                self.logger.warning(f"Charuco not found in {img_path}")
                continue
            corners, ids = res
            all_corners.append(corners)
            all_ids.append(ids)
            img_size = img.shape[1], img.shape[0]
        if img_size is None or not all_corners:
            raise RuntimeError("No valid Charuco detections")
        ret, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            all_corners, all_ids, board, img_size, None, None
        )
        out_base = paths.RESULTS_DIR / f"charuco_{timestamp()}"
        save_camera_params(out_base, K, dist, ret)
        self.logger.info(f"Charuco calibration saved to {out_base}")
        return out_base
