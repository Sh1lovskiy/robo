# vision/pointcloud.py
"""Point cloud utilities built around Open3D."""

import os
import numpy as np
import open3d as o3d
import cv2
from scipy.spatial.transform import Rotation as R


class PointCloudGenerator:
    """Utility class for creating and handling point clouds."""

    @staticmethod
    def load_camera_params(xml_path):
        fs = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_READ)
        camera_matrix = fs.getNode("camera_matrix").mat()
        dist_coeffs = fs.getNode("dist_coeffs").mat()
        fs.release()
        return camera_matrix, dist_coeffs

    @staticmethod
    def pose_to_transform(pose, angles_in_deg=True):
        x, y, z, rx, ry, rz = pose
        x, y, z = x / 1000.0, y / 1000.0, z / 1000.0
        if angles_in_deg:
            rx, ry, rz = np.deg2rad([rx, ry, rz])
        rot = R.from_euler("xyz", [rx, ry, rz]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = [x, y, z]
        return T

    @staticmethod
    def depth_to_cloud(depth, intr, rgb=None):
        h, w = depth.shape
        fx, fy = intr["fx"], intr["fy"]
        cx, cy = intr["ppx"], intr["ppy"]
        mask = (depth > 0.1) & (depth < 2.0)
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
    def save_ply(filename, points, colors=None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(filename, pcd)

    @staticmethod
    def load_ply(filename):
        pcd = o3d.io.read_point_cloud(filename)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        return points, colors

    @staticmethod
    def downsample_cloud(pcd, voxel_size=0.005):
        return pcd.voxel_down_sample(voxel_size)

    @staticmethod
    def icp_pairwise_align(source, target, threshold=0.02):
        reg = o3d.pipelines.registration.registration_icp(
            source,
            target,
            threshold,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )
        return reg.transformation

    @staticmethod
    def merge_clouds(clouds, voxel_size=0.003):
        merged = o3d.geometry.PointCloud()
        for pcd in clouds:
            merged += pcd
        if voxel_size:
            merged = merged.voxel_down_sample(voxel_size)
        return merged

    @staticmethod
    def filter_cloud(pcd, nb_neighbors=20, std_ratio=2.0):
        cl, _ = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )
        return cl

    @staticmethod
    def visualize(pcd):
        o3d.visualization.draw_geometries([pcd])

