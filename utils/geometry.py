from __future__ import annotations

"""Geometric helper functions for depth and RGB coordinate operations."""

from pathlib import Path
from scipy.spatial.transform import Rotation
from typing import Optional, Tuple
import json

import cv2
from dataclasses import dataclass
import numpy as np

from .logger import Logger

__all__ = [
    "load_extrinsics",
    "pixel_to_camera",
    "pixel_to_camera_depth",
    "rgb_to_depth_pixel",
    "map_rgb_corners_to_depth",
    "board_center_from_depth",
    "board_points_from_depth",
    "board_points_from_depth_with_extrinsics",
    "decompose_transform",
    "estimate_board_points_3d",
    "euler_to_matrix",
    "invert_transform",
    "make_transform",
    "TransformUtils",
]


def euler_to_matrix(
    rx: float, ry: float, rz: float, *, degrees: bool = True
) -> np.ndarray:
    """Return a rotation matrix from Euler angles."""
    return Rotation.from_euler("xyz", [rx, ry, rz], degrees=degrees).as_matrix()


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a homogeneous transform from ``R`` and ``t``."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T


def decompose_transform(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return rotation matrix and translation vector from a transform."""
    return T[:3, :3], T[:3, 3]


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Return the inverse of a homogeneous transform."""
    R, t = decompose_transform(T)
    R_inv = R.T
    t_inv = -R_inv @ t
    return make_transform(R_inv, t_inv)


def load_extrinsics(
    json_path: Path, from_key: str, to_key: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Return rotation matrix and translation vector from ``json_path``."""
    data = json.loads(Path(json_path).read_text())
    key = f"{from_key}_to_{to_key}"
    R = np.asarray(data[key]["R"], dtype=np.float64)
    t = np.asarray(data[key]["t"], dtype=np.float64)
    return R, t


def pixel_to_camera(pixel: np.ndarray, depth: float, K: np.ndarray) -> np.ndarray:
    """Project ``pixel`` with ``depth`` into camera coordinates."""
    x = (pixel[0] - K[0, 2]) * depth / K[0, 0]
    y = (pixel[1] - K[1, 2]) * depth / K[1, 1]
    return np.array([x, y, depth], dtype=np.float64)


def pixel_to_camera_depth(
    pixel: np.ndarray, depth: float, K_depth: np.ndarray
) -> np.ndarray:
    """Project depth pixel ``pixel`` into the depth camera frame."""
    x = (pixel[0] - K_depth[0, 2]) * depth / K_depth[0, 0]
    y = (pixel[1] - K_depth[1, 2]) * depth / K_depth[1, 1]
    return np.array([x, y, depth], dtype=np.float64)


def rgb_to_depth_pixel(
    x_rgb: float,
    y_rgb: float,
    K_rgb: np.ndarray,
    K_depth: np.ndarray,
    R_depth2rgb: np.ndarray,
    t_depth2rgb: np.ndarray,
) -> Tuple[float, float]:
    """Return depth pixel coordinates corresponding to ``(x_rgb, y_rgb)``."""
    pt_rgb = np.array([x_rgb, y_rgb, 1.0])
    ray_rgb = np.linalg.inv(K_rgb) @ pt_rgb
    ray_depth = R_depth2rgb.T @ (ray_rgb - t_depth2rgb)
    pt_depth = K_depth @ (ray_depth / ray_depth[2])
    return float(pt_depth[0]), float(pt_depth[1])


def board_center_from_depth(
    corners: np.ndarray, depth: np.ndarray, K: np.ndarray
) -> np.ndarray:
    """Return board center in camera coordinates using ``depth`` map."""
    center = corners.reshape(-1, 2).mean(axis=0)
    z = float(depth[int(round(center[1])), int(round(center[0]))])
    return pixel_to_camera(center, z, K)


def board_points_from_depth_with_extrinsics(
    corners_rgb: np.ndarray,
    depth_map: np.ndarray,
    K_depth: np.ndarray,
    K_rgb: np.ndarray,
    R_depth2rgb: np.ndarray,
    t_depth2rgb: np.ndarray,
    depth_scale: float,
) -> Optional[np.ndarray]:
    """Map RGB ``corners_rgb`` to 3-D points using depth map and extrinsics."""
    pts_rgb = []
    for x_rgb, y_rgb in corners_rgb.reshape(-1, 2):
        x_depth, y_depth = rgb_to_depth_pixel(
            x_rgb, y_rgb, K_rgb, K_depth, R_depth2rgb, t_depth2rgb
        )
        ix, iy = int(round(x_depth)), int(round(y_depth))
        if iy < 0 or iy >= depth_map.shape[0] or ix < 0 or ix >= depth_map.shape[1]:
            return None
        z = float(depth_map[iy, ix]) * depth_scale
        if not np.isfinite(z) or z <= 0:
            return None
        pt_depth = pixel_to_camera_depth(np.array([x_depth, y_depth]), z, K_depth)
        pt_rgb = R_depth2rgb @ pt_depth + t_depth2rgb
        pts_rgb.append(pt_rgb)
    return np.asarray(pts_rgb)


def board_points_from_depth(
    corners: np.ndarray, depth: np.ndarray, K: np.ndarray
) -> Optional[np.ndarray]:
    """Return 3-D coordinates of ``corners`` using ``depth`` map."""
    pts_3d = []
    for x, y in corners.reshape(-1, 2):
        ix, iy = int(round(x)), int(round(y))
        if iy < 0 or iy >= depth.shape[0] or ix < 0 or ix >= depth.shape[1]:
            return None
        z = float(depth[iy, ix])
        if not np.isfinite(z) or z <= 0:
            return None
        pts_3d.append(pixel_to_camera(np.array([x, y]), z, K))
    return np.asarray(pts_3d)


def map_rgb_corners_to_depth(
    corners_rgb: np.ndarray,
    depth_map: np.ndarray,
    K_rgb: np.ndarray,
    K_depth: np.ndarray,
    R_depth2rgb: np.ndarray,
    t_depth2rgb: np.ndarray,
    depth_scale: float = 0.001,
) -> Optional[np.ndarray]:
    """Convert RGB corner pixels to 3-D points using ``depth_map``."""
    pts_rgb = []
    for x_rgb, y_rgb in corners_rgb.reshape(-1, 2):
        x_d, y_d = rgb_to_depth_pixel(
            x_rgb, y_rgb, K_rgb, K_depth, R_depth2rgb, t_depth2rgb
        )
        ix, iy = int(round(x_d)), int(round(y_d))
        if iy < 0 or iy >= depth_map.shape[0] or ix < 0 or ix >= depth_map.shape[1]:
            return None
        z = float(depth_map[iy, ix]) * depth_scale
        if not np.isfinite(z) or z <= 0:
            return None
        pt_depth = pixel_to_camera_depth(np.array([x_d, y_d]), z, K_depth)
        pt_rgb = R_depth2rgb @ pt_depth + t_depth2rgb
        pts_rgb.append(pt_rgb)
    return np.asarray(pts_rgb)


def estimate_board_points_3d(
    charuco_corners: np.ndarray,
    depth_map: np.ndarray,
    object_points: np.ndarray,
    K_rgb: np.ndarray,
    dist_rgb: np.ndarray,
    K_depth: np.ndarray,
    R_depth2rgb: np.ndarray,
    t_depth2rgb: np.ndarray,
    depth_scale: float = 0.001,
) -> Optional[np.ndarray]:
    """Return 3-D board points using the depth map or fall back to PnP."""
    pts_rgb = map_rgb_corners_to_depth(
        charuco_corners,
        depth_map,
        K_rgb,
        K_depth,
        R_depth2rgb,
        t_depth2rgb,
        depth_scale,
    )
    if pts_rgb is not None:
        return pts_rgb
    ok, rvec, tvec = cv2.solvePnP(object_points, charuco_corners, K_rgb, dist_rgb)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return (R @ object_points.T).T + tvec.reshape(3)


@dataclass
class TransformUtils:
    """Utility class for chaining and applying transformations."""

    logger: Logger | None = None

    def __post_init__(self) -> None:  # noqa: D401
        self.logger = self.logger or Logger.get_logger("utils.transform")

    @staticmethod
    def build_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
        return make_transform(R, t)

    @staticmethod
    def apply_transform(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        T = make_transform(R, t)
        points_h = np.hstack([points, np.ones((points.shape[0], 1))])
        return (T @ points_h.T).T[:, :3]

    @staticmethod
    def chain_transforms(*Ts: np.ndarray) -> np.ndarray:
        T_out = np.eye(4)
        for T in Ts:
            T_out = T_out @ T
        return T_out

    def transform_points(self, points: np.ndarray, T: np.ndarray) -> np.ndarray:
        self.logger.debug(f"Applying transform to {points.shape[0]} points")
        points_h = np.hstack([points, np.ones((points.shape[0], 1))])
        return (T @ points_h.T).T[:, :3]

    def base_to_tcp(
        self, tcp_pose: np.ndarray | tuple[np.ndarray, np.ndarray]
    ) -> np.ndarray:
        if isinstance(tcp_pose, np.ndarray) and tcp_pose.shape == (4, 4):
            return tcp_pose
        if isinstance(tcp_pose, (tuple, list)) and len(tcp_pose) == 2:
            R, t = tcp_pose
            return make_transform(R, t)
        raise ValueError("tcp_pose must be SE(3) 4x4 or (R, t)")

    def tool_to_tcp(self, tcp_offset: np.ndarray | None) -> np.ndarray:
        if tcp_offset is None or np.allclose(tcp_offset, 0):
            return np.eye(4)
        x, y, z, rx, ry, rz = tcp_offset
        rot = euler_to_matrix(rx, ry, rz, degrees=True)
        return make_transform(rot, np.array([x, y, z]))

    def tcp_to_camera(self, R_handeye: np.ndarray, t_handeye: np.ndarray) -> np.ndarray:
        return make_transform(R_handeye, t_handeye)

    def get_base_to_camera(
        self,
        tcp_pose: np.ndarray | tuple[np.ndarray, np.ndarray],
        tcp_offset: np.ndarray | None,
        R_handeye: np.ndarray,
        t_handeye: np.ndarray,
    ) -> np.ndarray:
        T_base_tcp = self.base_to_tcp(tcp_pose)
        T_tcp_tool = self.tool_to_tcp(tcp_offset)
        T_tool_cam = self.tcp_to_camera(R_handeye, t_handeye)
        T_base_cam = self.chain_transforms(T_base_tcp, T_tcp_tool, T_tool_cam)
        self.logger.info("Computed T_base→camera")
        return T_base_cam

    def camera_to_world(
        self, points_cam: np.ndarray, T_base_cam: np.ndarray
    ) -> np.ndarray:
        self.logger.info("Transforming points: camera → world")
        return self.transform_points(points_cam, T_base_cam)

    def world_to_camera(
        self, points_world: np.ndarray, T_base_cam: np.ndarray
    ) -> np.ndarray:
        self.logger.info("Transforming points: world → camera")
        return self.transform_points(points_world, invert_transform(T_base_cam))

    def camera_to_tcp(
        self, points_cam: np.ndarray, R_handeye: np.ndarray, t_handeye: np.ndarray
    ) -> np.ndarray:
        T = self.tcp_to_camera(R_handeye, t_handeye)
        return self.transform_points(points_cam, invert_transform(T))

    def tcp_to_camera_points(
        self, points_tcp: np.ndarray, R_handeye: np.ndarray, t_handeye: np.ndarray
    ) -> np.ndarray:
        T = self.tcp_to_camera(R_handeye, t_handeye)
        return self.transform_points(points_tcp, T)
