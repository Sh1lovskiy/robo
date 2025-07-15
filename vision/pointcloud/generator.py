# vision/pointcloud/generator.py
"""Point cloud utilities built around Open3D."""

from __future__ import annotations

import numpy as np
import open3d as o3d
from utils.geometry import euler_to_matrix


class PointCloudGenerator:
    """Utility class for creating and handling point clouds."""

    @staticmethod
    def pose_to_transform(
        pose: list[float] | np.ndarray,
        *,
        angles_in_deg: bool = True,
    ) -> np.ndarray:
        """
        Convert [x, y, z, rx, ry, rz] pose to a 4x4 homogeneous transformation matrix.

        Args:
            pose: 6-DOF pose as [x, y, z, rx, ry, rz].
                x, y, z in millimeters.
                rx, ry, rz are Euler angles (degrees or radians).
            angles_in_deg: Whether the input angles are in degrees (default: True).

        Returns:
            4x4 SE(3) transformation matrix (numpy.ndarray)
        """
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (6,):
            raise ValueError(
                "Pose must be a sequence of 6 floats: [x, y, z, rx, ry, rz]"
            )

        position = pose[:3] / 1000.0  # mm → m
        rotation = euler_to_matrix(*pose[3:], degrees=angles_in_deg)  # 3x3

        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = position
        return T

    @staticmethod
    def depth_to_cloud(
        depth: np.ndarray,
        intr: dict,
        rgb: np.ndarray | None = None,
        depth_range: tuple[float, float] = (0.1, 1.0),
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Convert a depth map (and optionally an RGB image) to a 3D point cloud.

        Args:
            depth: HxW float array of depth (meters).
            intr: Dict with camera intrinsics: fx, fy, ppx (cx), ppy (cy).
            rgb: Optional HxWx3 or HxWx4 uint8 image for colors.
            depth_range: Valid depth range (min, max), in meters.

        Returns:
            points: (N, 3) float32 array of 3D points.
            colors: (N, 3) float32 array of RGB colors in [0, 1], or None.
        """
        h, w = depth.shape
        fx, fy = intr["fx"], intr["fy"]
        cx, cy = intr["ppx"], intr["ppy"]
        mask = (depth > 0.1) & (depth < 1.0)
        # np.where return y, x
        ys, xs = np.where(mask)
        zs = depth[ys, xs]
        # pinhole formula
        xs_ = (xs - cx) * zs / fx
        ys_ = (ys - cy) * zs / fy
        # stack in cloud Nx3
        points = np.stack((xs_, ys_, zs), axis=1)
        colors = None
        if rgb is not None:
            rgb = rgb[..., :3]  # Discard alpha if present
            # rgb[ys, xs] — colors for valid points
            colors = rgb[ys, xs]
            if colors.shape[1] == 3:
                colors = colors[:, ::-1]
            colors = colors.astype(np.float32) / 255.0  # [:, ::-1] RGB → BGR for opencv
            return points, colors
        return points, None

    @staticmethod
    def save_ply(
        filename: str,
        points: np.ndarray,
        colors: np.ndarray | None = None,
    ) -> None:
        """
        Save a point cloud to a PLY file using Open3D.

        Args:
            filename: Output path.
            points: (N, 3) array.
            colors: (N, 3) array in [0, 1] or None.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(filename, pcd)

    @staticmethod
    def load_ply(filename: str) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Load a point cloud from a PLY file.

        Returns:
            points: (N, 3) array.
            colors: (N, 3) array or None.
        """
        pcd = o3d.io.read_point_cloud(filename)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        return points, colors

    @staticmethod
    def downsample_cloud(
        pcd: o3d.geometry.PointCloud, voxel_size: float = 0.005
    ) -> o3d.geometry.PointCloud:
        """
        Downsample a point cloud using voxel grid filtering.
        """
        return pcd.voxel_down_sample(voxel_size)

    @staticmethod
    def icp_pairwise_align(
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        threshold: float = 0.02,
    ) -> np.ndarray:
        """
        Perform pairwise ICP registration (point-to-point).

        Returns:
            4x4 transformation matrix.
        """
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
        """
        Merge a list of point clouds, with optional voxel downsampling.
        """
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
        """
        Remove outliers from a point cloud using statistical filtering.
        """
        cl, _ = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )
        return cl

    @staticmethod
    def visualize(pcd: o3d.geometry.PointCloud) -> None:
        """
        Visualize a point cloud using Open3D's viewer.
        """
        o3d.visualization.draw_geometries([pcd])
