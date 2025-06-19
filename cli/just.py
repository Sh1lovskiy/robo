#!/usr/bin/env python3
"""
Detects object on a table via depth, finds its top face,
and computes a trajectory for a marker mounted on a robot.
Works with Intel RealSense D415 (pyrealsense2).
"""

import os
import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import json

from scipy.spatial.transform import Rotation as R

from vision.opencv_utils import OpenCVUtils

# ----------- CONFIG -------------
CAM_CALIB_PATH = "calibration/results/charuco_cam.xml"
HANDEYE_PATH = "calibration/results/handeye_DANIILIDIS.npz"
MARKER_LENGTH_MM = 210.0

# --------- UTILITY CLASSES ----------


class RealSenseDepthCapture:
    """Acquires RGB and depth images and computes 3D point cloud with scale."""

    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.profile = None
        self.depth_scale = None

    def start(self):
        self.profile = self.pipeline.start(self.config)
        self.depth_scale = (
            self.profile.get_device().first_depth_sensor().get_depth_scale()
        )
        print(f"Depth scale: {self.depth_scale}")

    def stop(self):
        self.pipeline.stop()

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())
        return color, depth

    def get_pointcloud(self, depth, intrinsics, color=None):
        h, w = depth.shape
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        i, j = np.meshgrid(np.arange(w), np.arange(h))
        valid = depth > 0
        z = depth[valid] * self.depth_scale
        x = (i[valid] - cx) * z / fx
        y = (j[valid] - cy) * z / fy
        points = np.stack((x, y, z), axis=-1)
        if color is not None:
            colors = color[valid].reshape(-1, 3) / 255.0
            return points, colors
        return points, None


class CameraCalibLoader:
    """Loads camera intrinsics from XML."""

    def __init__(self, path):
        self.K, self.dist = self._load(path)

    def _load(self, xml_path):
        fs = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_READ)
        K = fs.getNode("camera_matrix").mat()
        dist = fs.getNode("dist_coeffs").mat()
        fs.release()
        return K, dist


class HandEyeCalibLoader:
    """Loads hand-eye calibration R, t."""

    def __init__(self, path):
        arr = np.load(path)
        self.R = arr["R"]
        self.t = arr["t"]


class TableObjectSegmenter:
    """Clusters point cloud and extracts the object above the table."""

    def __init__(self, min_table_height=0.02, min_obj_height=0.02):
        self.min_table_height = min_table_height
        self.min_obj_height = min_obj_height

    def segment(self, cloud):
        # Filter very low points (assume table at z~min, object above)
        z = cloud[:, 2]
        table_z = np.percentile(z, 2)
        obj_mask = z > (table_z + self.min_obj_height)
        obj_cloud = cloud[obj_mask]
        # DBSCAN for further split (optional)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_cloud)
        labels = np.array(
            pcd.cluster_dbscan(eps=0.008, min_points=50, print_progress=False)
        )
        if labels.max() < 0:
            raise RuntimeError("No object found!")
        biggest = np.bincount(labels[labels >= 0]).argmax()
        object_indices = np.where(labels == biggest)[0]
        object_points = obj_cloud[object_indices]
        return object_points


class TopFaceFinder:
    """Finds points on the top face of the object."""

    def __init__(self, z_tol=0.01):
        self.z_tol = z_tol

    def find(self, cloud):
        max_z = np.max(cloud[:, 2])
        mask = np.abs(cloud[:, 2] - max_z) < self.z_tol
        top_points = cloud[mask]
        return top_points


class CenterTrajectoryPlanner:
    """Plans a centerline trajectory along the top face, considering marker length."""

    def __init__(self, marker_length_mm=210):
        self.marker_length = marker_length_mm / 1000.0

    def plan(self, top_points):
        centroid = np.mean(top_points, axis=0)
        centered = top_points - centroid
        u, s, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0]
        start_proj = np.min(centered @ direction)
        end_proj = np.max(centered @ direction)
        start = centroid + start_proj * direction
        end = centroid + end_proj * direction
        # Shift down by marker length (Z)
        offset = np.array([0, 0, -self.marker_length])
        traj_start = start + offset
        traj_end = end + offset
        return traj_start, traj_end


class HandEyeTransformer:
    """Applies hand-eye calibration to points (camera to TCP/world)."""

    def __init__(self, R, t):
        self.R = R
        self.t = t.reshape(-1)

    def transform(self, points):
        return (self.R @ points.T).T + self.t


# --------- MAIN PIPELINE ------------


def main():
    # 1. Load calibrations
    cam_calib = CameraCalibLoader(CAM_CALIB_PATH)
    handeye = HandEyeCalibLoader(HANDEYE_PATH)
    print("Loaded calibrations.")

    # 2. Get frame and cloud
    rs_cap = RealSenseDepthCapture()
    rs_cap.start()
    print("Move object to FOV and press any key in image window...")
    color, depth = rs_cap.get_frames()
    cv2.imshow("Color", color)
    OpenCVUtils.show_depth(depth)
    cv2.waitKey(0)
    points, colors = rs_cap.get_pointcloud(depth, cam_calib.K, color)
    rs_cap.stop()
    print(f"Captured {points.shape[0]} valid depth points")

    he = HandEyeTransformer(handeye.R, handeye.t)
    points_robot = he.transform(points)

    segmenter = TableObjectSegmenter()
    object_points = segmenter.segment(points_robot)
    print(f"Object points: {object_points.shape[0]}")

    top_finder = TopFaceFinder()
    top_points = top_finder.find(object_points)
    print(f"Top face points: {top_points.shape[0]}")

    planner = CenterTrajectoryPlanner(MARKER_LENGTH_MM)
    traj_start, traj_end = planner.plan(top_points)
    print("Marker trajectory:")
    print("  Start:", np.round(traj_start, 4))
    print("  End:  ", np.round(traj_end, 4))

    pcd_full = o3d.geometry.PointCloud()
    pcd_full.points = o3d.utility.Vector3dVector(points_robot)
    pcd_full.paint_uniform_color([0.6, 0.6, 0.6])
    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(object_points)
    obj_pcd.paint_uniform_color([0.2, 0.8, 0.2])
    top_pcd = o3d.geometry.PointCloud()
    top_pcd.points = o3d.utility.Vector3dVector(top_points)
    top_pcd.paint_uniform_color([1, 1, 0])
    traj = o3d.geometry.LineSet()
    traj.points = o3d.utility.Vector3dVector([traj_start, traj_end])
    traj.lines = o3d.utility.Vector2iVector([[0, 1]])
    traj.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
    o3d.visualization.draw_geometries([pcd_full, obj_pcd, top_pcd, traj])

    poses = {"start": traj_start.tolist(), "end": traj_end.tolist()}
    with open("marker_trajectory.json", "w") as f:
        json.dump(poses, f, indent=4)
    print("Saved marker trajectory to marker_trajectory.json")


if __name__ == "__main__":
    main()
