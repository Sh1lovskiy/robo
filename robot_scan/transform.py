"""Transformation utilities for point clouds and robot poses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from utils.logger import Logger

logger = Logger.get_logger("robot_scan.transform")


def load_handeye(path: str | Path) -> np.ndarray:
    """Load a 4x4 hand-eye transformation matrix from ``.npy`` or JSON."""
    path = Path(path)
    if path.suffix == ".npy":
        mat = np.load(path)
    else:
        with open(path, "r", encoding="utf-8") as f:
            mat = np.array(json.load(f))
    if mat.shape != (4, 4):
        raise ValueError("Hand-eye matrix must be 4x4")
    logger.info("Loaded hand-eye matrix from %s", path)
    return mat


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert [x,y,z,rx,ry,rz] pose (mm, deg) to 4x4 matrix."""
    T = np.eye(4)
    T[:3, 3] = pose[:3] / 1000.0
    T[:3, :3] = R.from_euler("xyz", pose[3:], degrees=True).as_matrix()
    return T


def transform_cloud(
    cloud: o3d.geometry.PointCloud, handeye: np.ndarray, tcp_pose: np.ndarray
) -> o3d.geometry.PointCloud:
    """Transform a point cloud from camera to robot base frame.

    Parameters
    ----------
    cloud : o3d.geometry.PointCloud
        Input point cloud in camera coordinates.
    handeye : np.ndarray
        4x4 camera-to-TCP transform.
    tcp_pose : np.ndarray
        Current robot TCP pose [x,y,z,rx,ry,rz] in base frame.
    """
    cam2tcp = handeye
    tcp2base = pose_to_matrix(tcp_pose)
    pts = np.asarray(cloud.points)
    ones = np.ones((len(pts), 1))
    pts_h = np.hstack((pts, ones))
    pts_base = (tcp2base @ cam2tcp @ pts_h.T).T[:, :3]
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts_base)
    out.colors = cloud.colors
    return out

def compute_tcp_pose(point: np.ndarray, main_axis: np.ndarray, normal: np.ndarray, offset: tuple[float, float, float] = (0, 0, 0)) -> np.ndarray:
    """Compute TCP pose from target point and plane orientation."""
    x = -main_axis / np.linalg.norm(main_axis)
    z = -normal / np.linalg.norm(normal)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    R_tcp = np.column_stack((x, y, z))
    pos = point + R_tcp @ np.array(offset)
    angles = R.from_matrix(R_tcp).as_euler("xyz", degrees=True)
    return np.concatenate((pos * 1000, angles))
