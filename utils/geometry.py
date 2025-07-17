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
    "TransformUtils",
    "load_extrinsics",
    "pixel_to_camera",
    "pixel_to_camera_depth",
    "rgb_to_depth_pixel",
    "decompose_transform",
    "euler_to_matrix",
    "invert_transform",
    "make_transform",
    "rotation_angle",
]


def rotation_angle(R: np.ndarray) -> float:
    """Return the angle of rotation represented by ``R`` in degrees."""
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2.0, -1.0, 1.0))
    return float(np.degrees(angle))


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
    """
    Back-project a pixel with depth value into 3D coordinates in the depth camera frame.

    Parameters:
        pixel:     (x, y) pixel coordinates in the depth image (can be float or int).
        depth:     Depth value (in meters or millimeters, depending on camera calibration).
        K_depth:   3x3 intrinsic matrix of the depth camera.

    Returns:
        np.ndarray: 3D point [X, Y, Z] in the depth camera coordinate system.
    """
    # Unpack pixel coordinates
    u, v = pixel

    # Calculate 3D X, Y coordinates using the pinhole camera model
    # x = (u - cx) * depth / fx
    # y = (v - cy) * depth / fy
    x = (u - K_depth[0, 2]) * depth / K_depth[0, 0]
    y = (v - K_depth[1, 2]) * depth / K_depth[1, 1]

    # Return the 3D point in the depth camera coordinate system
    return np.array([x, y, depth], dtype=np.float64)


def rgb_to_depth_pixel(
    x_rgb: float,
    y_rgb: float,
    K_rgb: np.ndarray,
    K_depth: np.ndarray,
    R_depth2rgb: np.ndarray,
    t_depth2rgb: np.ndarray,
) -> tuple[float, float]:
    """
    Convert a pixel from the RGB image to the corresponding pixel in the depth image.

    Parameters:
        x_rgb, y_rgb: Pixel coordinates in the RGB image.
        K_rgb:       3x3 intrinsic matrix of the RGB camera.
        K_depth:     3x3 intrinsic matrix of the depth camera.
        R_depth2rgb: 3x3 rotation matrix from depth to RGB camera.
        t_depth2rgb: 3x1 translation vector from depth to RGB camera.

    Returns:
        (x_depth, y_depth): Corresponding subpixel coordinates in the depth image.
    """
    # Convert the RGB pixel to homogeneous coordinates
    pt_rgb = np.array([x_rgb, y_rgb, 1.0])

    # Back-project to a normalized 3D ray in the RGB camera frame
    ray_rgb = np.linalg.inv(K_rgb) @ pt_rgb

    # Transform the ray into the depth camera frame
    # (Apply rotation and translation from depth to RGB, in reverse)
    # Apply extrinsics/offset depth
    ray_depth = R_depth2rgb.T @ (ray_rgb - t_depth2rgb)

    # Project the point onto the depth image plane (intrinsic transform)
    # Normalize so that the third (z) coordinate is 1
    pt_depth = K_depth @ (ray_depth / ray_depth[2])

    # Return the (x, y) pixel coordinates in the depth image as floats
    return float(pt_depth[0]), float(pt_depth[1])




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
        transformed = (T @ points_h.T).T[:, :3]
        self.logger.debug(
            f"Transformed points sample: {transformed[:2].tolist()}"
        )
        return transformed

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
