# vision/pointcloud.py

import numpy as np
import open3d as o3d
from utils.logger import Logger


class PointCloudGenerator:
    """
    Generate, save, and load 3D point clouds.
    """

    def __init__(self, logger=None):
        self.logger = logger or Logger.get_logger("vision.pointcloud")

    def depth_to_cloud(
        self, depth: np.ndarray, intrinsics: dict, rgb: np.ndarray = None
    ):
        """
        Convert depth image (+optional RGB) to Nx3 (or Nx6) cloud.
        """
        h, w = depth.shape
        fx, fy = intrinsics["fx"], intrinsics["fy"]
        cx, cy = intrinsics["ppx"], intrinsics["ppy"]

        mask = depth > 0
        ys, xs = np.where(mask)
        zs = depth[ys, xs]
        xs = (xs - cx) * zs / fx
        ys = (ys - cy) * zs / fy
        points = np.stack((xs, ys, zs), axis=1)
        if rgb is not None:
            colors = rgb[ys.astype(np.int32), xs.astype(np.int32)] / 255.0
            return points, colors
        return points, None

    def save_ply(self, filename, points, colors=None):
        """
        Save point cloud to a PLY file.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(filename, pcd)
        self.logger.info(f"Point cloud saved to {filename}")

    def load_ply(self, filename):
        """
        Load point cloud from PLY file.
        """
        pcd = o3d.io.read_point_cloud(filename)
        self.logger.info(f"Loaded point cloud: {filename}")
        return np.asarray(pcd.points), np.asarray(pcd.colors) if pcd.colors else None
