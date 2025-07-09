"""Explicit hand-eye calibration via grid sampling."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2

import numpy as np

from calibration.charuco import (
    load_board,
    load_camera_params,
    board_center_from_depth,
)
from calibration.pose_loader import JSONPoseLoader
from calibration.helpers.validation_utils import euler_to_matrix
from robot.controller import RobotController
from utils.logger import Logger, LoggerType
from utils.settings import paths, charuco, grid_calib, handeye, DEPTH_SCALE
from utils.cloud_utils import load_depth, get_image_pairs
from vision.camera import CameraBase, RealSenseD415
import matplotlib.pyplot as plt


def get_rigid_transform(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert len(A) == len(B)
    N = A.shape[0]
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))
    H = np.dot(AA.T, BB)
    U, _, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    t = np.dot(-R, centroid_A.T) + centroid_B.T
    return R, t


def pose_to_matrix(pose: List[float]) -> np.ndarray:
    R = euler_to_matrix(*pose[3:], degrees=True)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array(pose[:3], dtype=np.float64)
    return T


@dataclass
class GridCalibrator:
    """Move robot in a grid to compute camera pose using SVD."""

    robot: RobotController = field(default_factory=RobotController)
    camera: CameraBase = field(default_factory=RealSenseD415)
    cfg: grid_calib.__class__ = grid_calib
    logger: LoggerType = field(
        default_factory=lambda: Logger.get_logger("calibration.grid")
    )

    def __post_init__(self) -> None:
        board_cfg = dict(
            squares_x=charuco.squares_x,
            squares_y=charuco.squares_y,
            square_length=charuco.square_length,
            marker_length=charuco.marker_length,
            aruco_dict=charuco.aruco_dict,
        )
        self.board, self.dictionary = load_board(board_cfg)
        xml_path = self.cfg.charuco_xml
        if not os.path.isabs(xml_path):
            xml_path = os.path.join(paths.RESULTS_DIR, xml_path)
        self.camera_matrix, self.dist_coeffs = load_camera_params(xml_path)

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
        pts = np.stack([m.ravel() for m in mesh], axis=1)
        return pts

    def _board_center(self, color: np.ndarray, depth: np.ndarray) -> np.ndarray | None:
        return board_center_from_depth(
            color,
            depth,
            self.board,
            self.dictionary,
            self.camera_matrix,
            self.dist_coeffs,
            depth_scale=self.camera.depth_scale,
            min_corners=charuco.min_corners,
        )

    def _measured_point(self) -> np.ndarray:
        pose = self.robot.get_tcp_pose()
        if pose is None:
            raise RuntimeError("Robot pose unavailable")
        if self.cfg.calibration_type.upper() == "EYE_IN_HAND":
            T_be = pose_to_matrix(pose)
            invT = np.linalg.inv(T_be)
            offset = np.array(self.cfg.reference_point_offset).reshape(4, 1)
            tool_point = invT @ offset
            return tool_point[:3, 0]
        pos = np.array(pose[:3], dtype=np.float64)
        off = np.array(self.cfg.reference_point_offset[:3])
        return pos + off

    def _collect_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        grid_pts = self._generate_grid()
        measured, observed = [], []
        for idx, pt in enumerate(grid_pts):
            ori = list(self.cfg.tool_orientation)
            pose = [float(pt[0]), float(pt[1]), float(pt[2]), *ori]
            if not self.robot.move_linear(pose):
                self.logger.error(f"Move failed for point {idx}")
                continue
            self.robot.wait_motion_done()
            color, depth = self.camera.get_frames()
            if color is None or depth is None:
                self.logger.warning(f"No camera frame at {idx}")
                continue
            center_cam = self._board_center(color, depth)
            if center_cam is None:
                self.logger.warning(f"Board not detected at {idx}")
                continue
            measured.append(self._measured_point())
            observed.append(center_cam)
        return np.asarray(measured), np.asarray(observed)

    def calibrate(self) -> Tuple[np.ndarray, np.ndarray]:
        meas, obs = self._collect_samples()
        if len(meas) < 3:
            raise RuntimeError("Insufficient samples")
        R, t = get_rigid_transform(meas, obs)
        out_dir = self.cfg.calib_output_dir
        os.makedirs(out_dir, exist_ok=True)
        np.savez(os.path.join(out_dir, "handeye_svd.npz"), R=R, t=t)
        self.logger.info(f"Calibration complete using {len(meas)} samples")
        return R, t


@dataclass
class OfflineCalibrator:
    """Compute camera pose from saved Charuco images and robot poses."""

    hcfg: handeye.__class__ = handeye
    gcfg: grid_calib.__class__ = grid_calib
    logger: LoggerType = field(
        default_factory=lambda: Logger.get_logger("calibration.grid.offline")
    )

    def __post_init__(self) -> None:
        board_cfg = dict(
            squares_x=charuco.squares_x,
            squares_y=charuco.squares_y,
            square_length=charuco.square_length,
            marker_length=charuco.marker_length,
            aruco_dict=charuco.aruco_dict,
        )
        self.board, self.dictionary = load_board(board_cfg)
        xml_path = self.gcfg.charuco_xml
        if not os.path.isabs(xml_path):
            xml_path = os.path.join(paths.RESULTS_DIR, xml_path)
        self.camera_matrix, self.dist_coeffs = load_camera_params(xml_path)

    @staticmethod
    def _filter_poses(
        Rs: List[np.ndarray],
        ts: List[np.ndarray],
        valid_paths: List[str],
        all_paths: List[str],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        def idx(fname: str) -> str:
            return os.path.splitext(os.path.basename(fname))[0].split("_")[0]

        pose_map = {idx(p): (R, t) for p, R, t in zip(all_paths, Rs, ts)}
        f_Rs, f_ts = [], []
        for p in valid_paths:
            key = idx(p)
            if key in pose_map:
                R, t = pose_map[key]
                f_Rs.append(R)
                f_ts.append(t)
        return f_Rs, f_ts

    def _point_from_pose(self, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        if self.gcfg.calibration_type.upper() == "EYE_IN_HAND":
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            invT = np.linalg.inv(T)
            off = np.array(self.gcfg.reference_point_offset).reshape(4, 1)
            tool = invT @ off
            return tool[:3, 0]
        off = np.array(self.gcfg.reference_point_offset[:3])
        return t + off

    def _load_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        Rs, ts = JSONPoseLoader.load_poses(self.hcfg.robot_poses_file)
        img_pairs = get_image_pairs(self.hcfg.images_dir)
        measured, observed = [], []
        for (rgb_path, depth_path), (R_tcp, t_tcp) in zip(img_pairs, zip(Rs, ts)):
            color = cv2.imread(rgb_path)
            depth = load_depth(depth_path)
            if color is None or depth is None:
                self.logger.warning(f"Failed to load {rgb_path} or {depth_path}")
                continue
            center = board_center_from_depth(
                color,
                depth,
                self.board,
                self.dictionary,
                self.camera_matrix,
                self.dist_coeffs,
                depth_scale=DEPTH_SCALE,
                min_corners=charuco.min_corners,
            )
            if center is None:
                self.logger.warning(f"Board not found in {os.path.basename(rgb_path)}")
                continue
            measured.append(self._point_from_pose(R_tcp, t_tcp))
            observed.append(center)
        return np.vstack(measured), np.vstack(observed)

    def _plot(
        self, meas: np.ndarray, obs: np.ndarray, R: np.ndarray, t: np.ndarray
    ) -> None:
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        cam2base = np.linalg.inv(T)
        obs_base = (cam2base[:3, :3] @ obs.T + cam2base[:3, 3:]).T
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(meas[:, 0], meas[:, 1], meas[:, 2], c="blue", label="robot")
        ax.scatter(
            obs_base[:, 0], obs_base[:, 1], obs_base[:, 2], c="red", label="camera"
        )
        ax.legend()
        plt.tight_layout()
        out = os.path.join(self.gcfg.calib_output_dir, "handeye_svd_plot.png")
        plt.savefig(out)
        plt.close()

    def calibrate_from_files(self) -> Tuple[np.ndarray, np.ndarray]:
        meas, obs = self._load_samples()
        if len(meas) < 3:
            raise RuntimeError("Insufficient samples")
        R, t = get_rigid_transform(meas, obs)
        pred = (R @ meas.T).T + t
        err = np.sqrt(np.mean(np.sum((pred - obs) ** 2, axis=1)))
        out_dir = self.gcfg.calib_output_dir
        os.makedirs(out_dir, exist_ok=True)
        np.savez(os.path.join(out_dir, "handeye_svd_offline.npz"), R=R, t=t, rmse=err)
        self._plot(meas, obs, R, t)
        self.logger.info(
            f"Offline calibration complete using {len(meas)} samples, RMSE {err:.6f}"
        )
        return R, t
