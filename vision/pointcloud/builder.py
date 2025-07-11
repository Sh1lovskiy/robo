"""Point cloud acquisition in the robot base frame."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Tuple

import numpy as np
import open3d as o3d
import pyrealsense2 as rs

from utils.math_utils import euler_to_matrix
from utils.logger import Logger, LoggerType
from ..transform import TransformUtils
from vision.camera import CameraBase
from .generator import PointCloudGenerator


@dataclass
class PointCloudBuilder:
    """Capture and process clouds using a calibrated camera."""

    camera: CameraBase
    handeye: Tuple[np.ndarray, np.ndarray]
    logger: LoggerType = field(
        default_factory=lambda: Logger.get_logger("vision.pointcloud.builder")
    )
    transformer: TransformUtils = field(default_factory=TransformUtils)
    generator: PointCloudGenerator = field(default_factory=PointCloudGenerator)

    def capture(self, robot_pose: Iterable[float]) -> o3d.geometry.PointCloud:
        color, depth = self.camera.get_frames()
        stream = self.camera.profile.get_stream(rs.stream.depth)
        intr = stream.as_video_stream_profile().get_intrinsics()
        K = {"fx": intr.fx, "fy": intr.fy, "ppx": intr.ppx, "ppy": intr.ppy}
        depth_m = depth.astype(np.float32) * self.camera.depth_scale
        points, colors = self.generator.depth_to_cloud(depth_m, K, color)
        Rb, tb = self._robot_pose_to_rt(robot_pose)
        Rc, tc = self.handeye
        T = self.transformer.get_base_to_camera((Rb, tb), None, Rc, tc)
        pts_base = self.transformer.transform_points(points, T)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_base)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd = pcd.voxel_down_sample(voxel_size=0.002)
        pcd, _ = pcd.remove_statistical_outlier(20, 1.0)
        self.logger.info(
            f"Cloud: {len(pcd.points)} points, mean Z {np.mean(points[:,2]):.3f} m"
        )
        return pcd

    def _robot_pose_to_rt(self, pose: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
        arr = list(pose)
        R = euler_to_matrix(arr[3], arr[4], arr[5], degrees=True)
        t = np.array(arr[:3], dtype=float) / 1000.0
        return R, t

    def save(self, pcd: o3d.geometry.PointCloud, path: str) -> None:
        o3d.io.write_point_cloud(path, pcd)
        self.logger.info(f"Saved cloud to {path}")

    def load(self, path: str) -> o3d.geometry.PointCloud:
        pcd = o3d.io.read_point_cloud(path)
        self.logger.info(f"Loaded cloud from {path}")
        return pcd

    def remove_plane(
        self, pcd: o3d.geometry.PointCloud, dist: float = 0.003
    ) -> o3d.geometry.PointCloud:
        plane_model, inliers = pcd.segment_plane(dist, 3, 1000)
        self.logger.debug(
            f"Plane: {np.round(plane_model,4).tolist()} with {len(inliers)} inliers"
        )
        return pcd.select_by_index(inliers, invert=True)
