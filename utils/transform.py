from __future__ import annotations

"""Rigid transformation utilities."""

from dataclasses import dataclass
import numpy as np

from .logger import Logger
from .math_utils import (
    euler_to_matrix,
    make_transform,
    decompose_transform,
    invert_transform,
)


__all__ = [
    "euler_to_matrix",
    "make_transform",
    "decompose_transform",
    "invert_transform",
    "TransformUtils",
]


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
