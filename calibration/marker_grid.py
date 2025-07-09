"""Hand-eye calibration using a grid of ArUco marker observations."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs

from calibration.aruco_utils import marker_center_from_depth
from calibration.helpers.validation_utils import euler_to_matrix
from utils.logger import Logger, LoggerType
from utils.settings import marker_grid, MarkerGridSettings
from robot.controller import RobotController
from vision.camera import CameraBase, RealSenseD415


def get_rigid_transform(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute optimal rigid transform with SVD."""
    assert len(A) == len(B)
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.T @ BB
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = -R @ centroid_A + centroid_B
    return R, t


def rigid_transform_with_scale(
    A: np.ndarray, B: np.ndarray, *, optimize: bool = True
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Solve for ``R`` and ``t`` optionally optimizing depth scale."""

    def error(scale: float) -> float:
        R, t = get_rigid_transform(A, B * scale)
        pred = (R @ A.T).T + t
        return float(np.sqrt(np.mean(np.sum((pred - B * scale) ** 2, axis=1))))

    scale = 1.0
    if optimize:
        from scipy.optimize import minimize_scalar

        res = minimize_scalar(error, bounds=(0.9, 1.1), method="bounded")
        scale = float(res.x)
    err = error(scale)
    R, t = get_rigid_transform(A, B * scale)
    return R, t, scale, err


def pose_to_matrix(pose: List[float]) -> np.ndarray:
    R = euler_to_matrix(*pose[3:], degrees=True)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array(pose[:3], dtype=np.float64)
    return T


@dataclass
class MarkerGridCalibrator:
    """Collect samples and compute camera pose via SVD."""

    robot: RobotController = field(default_factory=RobotController)
    camera: CameraBase = field(default_factory=RealSenseD415)
    cfg: MarkerGridSettings = marker_grid
    logger: LoggerType = field(
        default_factory=lambda: Logger.get_logger("calibration.marker_grid")
    )

    def _generate_grid(self) -> np.ndarray:
        cfg = self.cfg
        xs = np.arange(
            cfg.workspace_limits[0][0],
            cfg.workspace_limits[0][1] + cfg.grid_step,
            cfg.grid_step,
        )
        ys = np.arange(
            cfg.workspace_limits[1][0],
            cfg.workspace_limits[1][1] + cfg.grid_step,
            cfg.grid_step,
        )
        zs = np.arange(
            cfg.workspace_limits[2][0],
            cfg.workspace_limits[2][1] + cfg.grid_step,
            cfg.grid_step,
        )
        mesh = np.meshgrid(xs, ys, zs, indexing="ij")
        return np.stack([m.ravel() for m in mesh], axis=1)

    def _marker_center(
        self, color: np.ndarray, depth: np.ndarray, K: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[int, int]] | Tuple[None, None]:
        return marker_center_from_depth(
            color,
            depth,
            self.dictionary,
            K,
            marker_id=self.cfg.marker_id,
            depth_scale=self.camera.depth_scale,
            logger=self.logger,
        )

    def _measured_point(self) -> np.ndarray:
        pose = self.robot.get_tcp_pose()
        if pose is None:
            raise RuntimeError("Robot pose unavailable")
        T = pose_to_matrix(pose)
        invT = np.linalg.inv(T)
        offset = np.array(self.cfg.reference_point_offset).reshape(4, 1)
        tool_pt = invT @ offset
        return tool_pt[:3, 0]

    def _collect_samples(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, List[List[float]], List[Tuple[int, int]]]:
        grid_pts = self._generate_grid()
        measured, observed, poses, pixels = [], [], [], []
        self.camera.start()
        stream = self.camera.profile.get_stream(rs.stream.color)
        intr = stream.as_video_stream_profile().get_intrinsics()
        K = np.array(
            [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]],
            dtype=np.float64,
        )
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        for idx, pt in enumerate(grid_pts):
            pose_cmd = [
                float(pt[0]),
                float(pt[1]),
                float(pt[2]),
                *self.cfg.tool_orientation,
            ]
            if not self.robot.move_linear(pose_cmd):
                self.logger.error(f"Move failed for {idx}")
                continue
            self.robot.wait_motion_done()
            color, depth = self.camera.get_frames()
            if color is None or depth is None:
                self.logger.warning(f"No frame at {idx}")
                continue
            center, px = self._marker_center(color, depth, K)
            if center is None:
                self.logger.warning(f"Marker not detected at {idx}")
                continue
            measured.append(self._measured_point())
            observed.append(center)
            poses.append(pose_cmd)
            pixels.append(px)
        self.camera.stop()
        return (
            np.asarray(measured, dtype=float),
            np.asarray(observed, dtype=float),
            poses,
            pixels,
        )

    def calibrate(self) -> Tuple[np.ndarray, np.ndarray, float, float]:
        meas, obs, poses, pixels = self._collect_samples()
        if len(meas) < 3:
            raise RuntimeError("Insufficient samples")
        R, t, scale, err = rigid_transform_with_scale(meas, obs, optimize=True)
        out_dir = self.cfg.calib_output_dir
        os.makedirs(out_dir, exist_ok=True)
        np.savez(
            os.path.join(out_dir, "marker_handeye.npz"),
            R=R,
            t=t,
            depth_scale=scale,
            rmse=err,
            robot_poses=np.asarray(poses),
            pixels=np.asarray(pixels),
            observed=obs,
            measured=meas,
        )
        self.logger.info(
            f"Calibration complete using {len(meas)} samples, RMSE {err:.6f}"
        )
        return R, t, scale, err
