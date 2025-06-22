# vision/transform.py

from __future__ import annotations

import numpy as np
from utils.logger import Logger
from utils.geometry import euler_to_matrix


class TransformUtils:
    """
    Utilities for 3D rigid transformations between coordinate frames
    (robot base, TCP, camera, point cloud).
    Explicitly handles eye-in-hand/eye-to-hand conventions and tool (TCP) offset.
    """

    def __init__(self, logger: Logger | None = None) -> None:
        self.logger = logger or Logger.get_logger("vision.transform")

    @staticmethod
    def build_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Build 4x4 homogeneous SE(3) transform from rotation and translation.
        """
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    @staticmethod
    def apply_transform(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Apply rotation and translation to Nx3 points."""
        T = TransformUtils.build_transform(R, t)
        points_h = np.hstack([points, np.ones((points.shape[0], 1))])
        return (T @ points_h.T).T[:, :3]

    @staticmethod
    def decompose_transform(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Decompose 4x4 transform into (R, t).
        """
        R = T[:3, :3]
        t = T[:3, 3]
        return R, t

    def transform_points(self, points: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Apply 4x4 transform to Nx3 points.
        """
        self.logger.debug(f"Applying transform to {points.shape[0]} points.")
        points_h = np.hstack([points, np.ones((points.shape[0], 1))])
        out = (T @ points_h.T).T[:, :3]
        return out

    def chain_transforms(self, *Ts: np.ndarray) -> np.ndarray:
        """
        Multiply (chain) an arbitrary number of 4x4 transforms.
        """
        T_out = np.eye(4)
        for T in Ts:
            T_out = T_out @ T
        return T_out

    def base_to_tcp(
        self, tcp_pose: np.ndarray | tuple[np.ndarray, np.ndarray]
    ) -> np.ndarray:
        """
        Convert TCP (tool center point) pose to 4x4 base→TCP transform.
        Accepts:
        - (4,4) numpy array (SE(3))
        - (R, t) tuple
        """
        if isinstance(tcp_pose, np.ndarray) and tcp_pose.shape == (4, 4):
            return tcp_pose
        elif isinstance(tcp_pose, (tuple, list)) and len(tcp_pose) == 2:
            R, t = tcp_pose
            return self.build_transform(R, t)
        else:
            raise ValueError("tcp_pose must be SE(3) 4x4 or (R, t)")

    def tool_to_tcp(self, tcp_offset: np.ndarray | None) -> np.ndarray:
        """
        Create transform from tool flange to TCP (tool offset, usually
        from config or robot teach pendant).
        tcp_offset: 6dof (x, y, z, rx, ry, rz) in robot tool convention.
        """
        # If no offset, just identity
        if tcp_offset is None or np.allclose(tcp_offset, 0):
            return np.eye(4)
        x, y, z, rx, ry, rz = tcp_offset
        from utils.geometry import euler_to_matrix

        rot = euler_to_matrix(rx, ry, rz, degrees=True)
        return self.build_transform(rot, np.array([x, y, z]))

    def tcp_to_camera(self, R_handeye: np.ndarray, t_handeye: np.ndarray) -> np.ndarray:
        """
        Transform from TCP to camera (hand-eye calibration result).
        """
        return self.build_transform(R_handeye, t_handeye)

    def get_base_to_camera(
        self,
        tcp_pose: np.ndarray | tuple[np.ndarray, np.ndarray],
        tcp_offset: np.ndarray | None,
        R_handeye: np.ndarray,
        t_handeye: np.ndarray,
    ) -> np.ndarray:
        """
        Compute SE(3) from robot base to camera using all links:
        base→TCP→(tool offset)→camera
        """
        T_base_tcp = self.base_to_tcp(tcp_pose)
        T_tcp_tool = self.tool_to_tcp(tcp_offset)
        T_tool_cam = self.tcp_to_camera(R_handeye, t_handeye)
        T_base_cam = self.chain_transforms(T_base_tcp, T_tcp_tool, T_tool_cam)
        self.logger.info("Computed T_base→camera.")
        return T_base_cam

    def camera_to_world(
        self, points_cam: np.ndarray, T_base_cam: np.ndarray
    ) -> np.ndarray:
        """
        Project points from camera to world (base) coordinates.
        """
        self.logger.info("Transforming points: camera → world")
        return self.transform_points(points_cam, T_base_cam)

    def world_to_camera(
        self, points_world: np.ndarray, T_base_cam: np.ndarray
    ) -> np.ndarray:
        """
        Project points from world (base) to camera coordinates.
        """
        self.logger.info("Transforming points: world → camera")
        return self.transform_points(points_world, np.linalg.inv(T_base_cam))

    def camera_to_tcp(
        self, points_cam: np.ndarray, R_handeye: np.ndarray, t_handeye: np.ndarray
    ) -> np.ndarray:
        """
        Camera frame to TCP frame.
        """
        T = self.tcp_to_camera(R_handeye, t_handeye)
        return self.transform_points(points_cam, np.linalg.inv(T))

    def tcp_to_camera_points(
        self, points_tcp: np.ndarray, R_handeye: np.ndarray, t_handeye: np.ndarray
    ) -> np.ndarray:
        """
        TCP frame to camera frame.
        """
        T = self.tcp_to_camera(R_handeye, t_handeye)
        return self.transform_points(points_tcp, T)


def _test_transform_utils():
    from scipy.spatial.transform import Rotation as R

    logger = Logger.get_logger("vision.transform.test")
    tu = TransformUtils(logger)

    # Identity test (no offset, no rotation)
    tcp_pose = (np.eye(3), np.zeros(3))
    tcp_offset = np.zeros(6)
    R_handeye = np.eye(3)
    t_handeye = np.zeros(3)
    T_base_cam = tu.get_base_to_camera(tcp_pose, tcp_offset, R_handeye, t_handeye)
    assert np.allclose(T_base_cam, np.eye(4))

    # Simple translation
    tcp_pose = (np.eye(3), np.array([1, 2, 3]))
    T_base_cam = tu.get_base_to_camera(tcp_pose, tcp_offset, R_handeye, t_handeye)
    assert np.allclose(T_base_cam[:3, 3], [1, 2, 3])

    # Offset and handeye rotation
    tcp_offset = np.array([0.1, 0.2, 0.3, 0, 0, np.pi / 2])
    R_handeye = euler_to_matrix(0, 0, 90)
    t_handeye = np.array([0.05, 0, 0])
    T_base_cam = tu.get_base_to_camera(tcp_pose, tcp_offset, R_handeye, t_handeye)
    assert T_base_cam.shape == (4, 4)

    # Cloud transform: camera→world→camera round-trip
    cloud = np.random.randn(10, 3)
    world = tu.camera_to_world(cloud, T_base_cam)
    cam2 = tu.world_to_camera(world, T_base_cam)
    assert np.allclose(cloud, cam2, atol=1e-8)
    logger.info("All TransformUtils tests passed.")


if __name__ == "__main__":
    _test_transform_utils()
