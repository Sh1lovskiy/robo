# vision/cloud/generator.py
"""Point cloud utilities built around Open3D."""

from __future__ import annotations

import numpy as np
import open3d as o3d
from calibration.helpers.pose_utils import euler_to_matrix


class PointCloudGenerator:
    """Utility class for creating and handling point clouds."""

    @staticmethod
    def pose_to_transform(
        pose: list[float],
        angles_in_deg: bool = True,
    ) -> np.ndarray:
        """Convert a 6-DOF pose [x,y,z,rx,ry,rz] to a 4x4 transform."""

        x, y, z, rx, ry, rz = pose
        x, y, z = x / 1000.0, y / 1000.0, z / 1000.0
        rot = euler_to_matrix(rx, ry, rz, degrees=angles_in_deg)
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = [x, y, z]
        return T

    @staticmethod
    def depth_to_cloud(
        depth: np.ndarray,
        intr: dict,
        rgb: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Convert a depth map (and optional RGB) to XYZ point cloud."""

        h, w = depth.shape
        fx, fy = intr["fx"], intr["fy"]
        cx, cy = intr["ppx"], intr["ppy"]
        mask = (depth > 0.1) & (depth < 1.0)
        ys, xs = np.where(mask)
        zs = depth[ys, xs]
        xs_ = (xs - cx) * zs / fx
        ys_ = (ys - cy) * zs / fy
        points = np.stack((xs_, ys_, zs), axis=1)
        if rgb is not None:
            if rgb.shape[2] == 4:
                rgb = rgb[..., :3]
            colors = rgb[ys, xs][:, ::-1] / 255.0
            return points, colors
        return points, None

    @staticmethod
    def save_ply(
        filename: str,
        points: np.ndarray,
        colors: np.ndarray | None = None,
    ) -> None:
        """Save point cloud as a PLY file using Open3D."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(filename, pcd)

    @staticmethod
    def load_ply(filename: str) -> tuple[np.ndarray, np.ndarray | None]:
        """Load points and colors from a PLY file."""
        pcd = o3d.io.read_point_cloud(filename)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        return points, colors

    @staticmethod
    def downsample_cloud(
        pcd: o3d.geometry.PointCloud, voxel_size: float = 0.005
    ) -> o3d.geometry.PointCloud:
        """Voxel downsample the cloud to reduce point count."""
        return pcd.voxel_down_sample(voxel_size)

    @staticmethod
    def icp_pairwise_align(
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        threshold: float = 0.02,
    ) -> np.ndarray:
        """Perform point-to-point ICP and return the transform matrix."""
        reg = o3d.pipelines.registration.registration_icp(
            source,
            target,
            threshold,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )
        return reg.transformation

    @staticmethod
    def merge_clouds(
        clouds: list[o3d.geometry.PointCloud],
        voxel_size: float = 0.003,
    ) -> o3d.geometry.PointCloud:
        """Combine multiple clouds and optionally downsample."""
        merged = o3d.geometry.PointCloud()
        for pcd in clouds:
            merged += pcd
        if voxel_size:
            merged = merged.voxel_down_sample(voxel_size)
        return merged

    @staticmethod
    def filter_cloud(
        pcd: o3d.geometry.PointCloud,
        nb_neighbors: int = 20,
        std_ratio: float = 2.0,
    ) -> o3d.geometry.PointCloud:
        """Remove outliers using statistical filtering."""
        cl, _ = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )
        return cl

    @staticmethod
    def visualize(pcd: o3d.geometry.PointCloud) -> None:
        """Display the given cloud using Open3D GUI."""
        o3d.visualization.draw_geometries([pcd])
