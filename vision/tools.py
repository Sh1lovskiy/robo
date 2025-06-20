# vision/tools.py
"""Utility functions for camera debug and point cloud operations."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import cv2
import numpy as np
import open3d as o3d

from utils.logger import Logger
from utils.config import Config
from vision.realsense import RealSenseCamera
from vision.pointcloud import PointCloudGenerator
from vision.transform import TransformUtils


# ---------------------------------------------------------------------------
# Camera utilities
# ---------------------------------------------------------------------------

@dataclass
class IntrinsicsPrinter:
    """Print camera intrinsics."""

    camera: RealSenseCamera = RealSenseCamera()
    logger: Logger = Logger.get_logger("vision.tools.intrinsics")

    def run(self) -> None:
        self.camera.start()
        intr = self.camera.get_intrinsics()
        self.logger.info(f"Intrinsics: {intr}")
        for k, v in intr.items():
            print(f"{k}: {v}")
        self.camera.stop()


@dataclass
class DepthChecker:
    """Display live depth map with distance readout."""

    camera: RealSenseCamera = RealSenseCamera()
    logger: Logger = Logger.get_logger("vision.tools.depth")

    def run(self) -> None:
        self.camera.start()
        depth_scale = self.camera.get_depth_scale()
        self.logger.info(f"Depth scale: {depth_scale:.6f} m")
        try:
            while True:
                color, depth = self.camera.get_frames()
                h, w = depth.shape
                x, y = w // 2, h // 2
                dist_mm = int(depth[y, x] * depth_scale * 1000)
                depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                cv2.circle(depth_vis, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(depth_vis, f"{dist_mm} mm", (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow("Depth", depth_vis)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        finally:
            self.camera.stop()
            cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Point cloud utilities
# ---------------------------------------------------------------------------

def capture_cloud(output: str) -> None:
    logger = Logger.get_logger("vision.tools.capture")
    cam = RealSenseCamera()
    cam.start()
    color, depth = cam.get_frames()
    intr = cam.get_intrinsics()
    points, colors = PointCloudGenerator.depth_to_cloud(depth, intr, rgb=color)
    PointCloudGenerator.save_ply(output, points, colors)
    logger.info(f"Saved cloud to {output}")
    cam.stop()


def transform_cloud(input_ply: str, calib_npz: str, output: str) -> None:
    logger = Logger.get_logger("vision.tools.transform")
    cloud_gen = PointCloudGenerator()
    points, colors = cloud_gen.load_ply(input_ply)
    calib = np.load(calib_npz)
    R, t = calib["R"], calib["t"]
    points_t = TransformUtils.apply_transform(points, R, t)
    cloud_gen.save_ply(output, points_t, colors)
    logger.info(f"Transformed cloud saved to {output}")


def view_cloud(input_ply: str) -> None:
    logger = Logger.get_logger("vision.tools.view")
    pcd = o3d.io.read_point_cloud(input_ply)
    logger.info(f"Loaded point cloud {input_ply}")
    o3d.visualization.draw_geometries([pcd])


# CLI entry helpers ---------------------------------------------------------

def main_capture() -> None:
    parser = argparse.ArgumentParser(description="Capture point cloud")
    Config.load()
    out_dir = Config.get("cloud.output_dir", "clouds")
    parser.add_argument("--output", default=f"{out_dir}/cloud.ply")
    args = parser.parse_args()
    capture_cloud(args.output)


def main_transform() -> None:
    parser = argparse.ArgumentParser(description="Transform point cloud")
    parser.add_argument("--input", required=True)
    parser.add_argument("--calib", required=True)
    Config.load()
    out_dir = Config.get("cloud.output_dir", "clouds")
    parser.add_argument("--output", default=f"{out_dir}/cloud_world.ply")
    args = parser.parse_args()
    transform_cloud(args.input, args.calib, args.output)


def main_view() -> None:
    parser = argparse.ArgumentParser(description="View point cloud")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    view_cloud(args.input)


def main_intrinsics() -> None:
    IntrinsicsPrinter().run()


def main_depth() -> None:
    DepthChecker().run()


# ---------------------------------------------------------------------------
# Point cloud analysis pipeline
# ---------------------------------------------------------------------------


def _roi_limits():
    Config.load()
    return Config.get(
        "cloud_pipeline.roi_limits",
        {"x": (-0.3, -0.1), "y": (-0.2, 0.2), "z": (0.02, 0.1)},
    )


class PointCloudDenoiser:
    """Remove outliers from a point cloud."""

    def __init__(self, nb_neighbors=30, std_ratio=5.0, logger=None):
        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio
        self.logger = logger or Logger.get_logger("vision.pipeline.denoiser")

    def denoise(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        cl, ind = pcd.remove_statistical_outlier(
            nb_neighbors=self.nb_neighbors, std_ratio=self.std_ratio
        )
        denoised = pcd.select_by_index(ind)
        self.logger.info(
            f"Removed {len(pcd.points) - len(denoised.points)} outliers"
        )
        return denoised


class PointCloudCropper:
    """Crop cloud to a region of interest."""

    def __init__(self, limits=None, logger=None):
        self.limits = limits or _roi_limits()
        self.logger = logger or Logger.get_logger("vision.pipeline.cropper")

    def crop(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        points = np.asarray(pcd.points)
        mask = (
            (points[:, 0] >= self.limits["x"][0])
            & (points[:, 0] <= self.limits["x"][1])
            & (points[:, 1] >= self.limits["y"][0])
            & (points[:, 1] <= self.limits["y"][1])
            & (points[:, 2] >= self.limits["z"][0])
            & (points[:, 2] <= self.limits["z"][1])
        )
        cropped = pcd.select_by_index(np.where(mask)[0])
        self.logger.info(
            f"Cropped to ROI: {len(cropped.points)} of {len(points)} points"
        )
        return cropped


class PointCloudClusterer:
    """Cluster cloud using DBSCAN."""

    def __init__(self, eps=0.05, min_points=100, logger=None):
        self.eps = eps
        self.min_points = min_points
        self.logger = logger or Logger.get_logger("vision.pipeline.cluster")

    def extract_object(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        labels = np.array(
            pcd.cluster_dbscan(
                eps=self.eps, min_points=self.min_points, print_progress=True
            )
        )
        n_clusters = labels.max() + 1
        self.logger.info(f"Found {n_clusters} clusters")
        if n_clusters == 0:
            raise RuntimeError("No clusters found")
        largest_label = np.bincount(labels[labels >= 0]).argmax()
        obj_idx = np.where(labels == largest_label)[0]
        return pcd.select_by_index(obj_idx)


class ObjectSurfaceAnalyzer:
    """Compute bounding box of the object."""

    def __init__(self, logger=None):
        self.logger = logger or Logger.get_logger("vision.pipeline.analyzer")

    def get_bounding_box(self, pcd: o3d.geometry.PointCloud):
        aabb = pcd.get_axis_aligned_bounding_box()
        aabb.color = (0, 1, 0)
        return aabb, np.asarray(aabb.get_box_points())


class TopFaceFinder:
    """Find points belonging to the top face."""

    def __init__(self, z_tol=0.01, logger=None):
        self.z_tol = z_tol
        self.logger = logger or Logger.get_logger("vision.pipeline.topface")

    def find_top_face(self, pcd: o3d.geometry.PointCloud) -> np.ndarray:
        points = np.asarray(pcd.points)
        max_z = np.max(points[:, 2])
        mask = np.abs(points[:, 2] - max_z) < self.z_tol
        top_points = points[mask]
        self.logger.info(f"Top face points: {len(top_points)} at Z={max_z:.5f}")
        return top_points


class TopFaceTrajectoryPlanner:
    """Plan trajectory along the top face center line."""

    def __init__(self, marker_length_mm=210, logger=None):
        self.marker_length = marker_length_mm / 1000.0
        self.logger = logger or Logger.get_logger("vision.pipeline.traj")

    def plan_center_trajectory(self, top_points: np.ndarray):
        centroid = np.mean(top_points, axis=0)
        centered = top_points - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0]
        start_proj = np.min(centered @ direction)
        end_proj = np.max(centered @ direction)
        start = centroid + start_proj * direction
        end = centroid + end_proj * direction
        offset = np.array([0, 0, self.marker_length])
        traj_start = start + offset
        traj_end = end + offset
        self.logger.info(
            f"Trajectory from {np.round(traj_start,5)} to {np.round(traj_end,5)}"
        )
        return traj_start, traj_end


class CloudPipeline:
    """Full visualization and analysis pipeline."""

    def __init__(self, logger=None):
        self.logger = logger or Logger.get_logger("vision.pipeline")
        self.denoiser = PointCloudDenoiser(logger=self.logger)
        self.cropper = PointCloudCropper(logger=self.logger)
        self.clusterer = PointCloudClusterer(logger=self.logger)
        self.analyzer = ObjectSurfaceAnalyzer(logger=self.logger)
        self.topfinder = TopFaceFinder(logger=self.logger)
        self.trajectory_planner = TopFaceTrajectoryPlanner(logger=self.logger)

    def run(self, input_ply: str) -> None:
        pcd = o3d.io.read_point_cloud(input_ply)
        self.logger.info(f"Loaded {input_ply}, {len(pcd.points)} points")
        o3d.visualization.draw_geometries([pcd], window_name="Raw cloud")
        cropped = self.cropper.crop(pcd)
        o3d.visualization.draw_geometries([cropped], window_name="Cropped cloud")
        clean = self.denoiser.denoise(cropped)
        o3d.visualization.draw_geometries([clean], window_name="Denoised cloud")
        obj = self.clusterer.extract_object(clean)
        o3d.visualization.draw_geometries([obj], window_name="Clustered object")
        aabb, _ = self.analyzer.get_bounding_box(obj)
        o3d.visualization.draw_geometries([obj, aabb], window_name="Bounding box")
        top_points = self.topfinder.find_top_face(obj)
        if len(top_points) > 0:
            top_pcd = o3d.geometry.PointCloud()
            top_pcd.points = o3d.utility.Vector3dVector(top_points)
            top_pcd.paint_uniform_color([1, 1, 0])
            o3d.visualization.draw_geometries(
                [obj, top_pcd], window_name="Top face points"
            )
            traj_start, traj_end = self.trajectory_planner.plan_center_trajectory(top_points)
            traj_line = o3d.geometry.LineSet()
            traj_line.points = o3d.utility.Vector3dVector([traj_start, traj_end])
            traj_line.lines = o3d.utility.Vector2iVector([[0, 1]])
            traj_line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
            o3d.visualization.draw_geometries(
                [obj, top_pcd, traj_line], window_name="Marker Trajectory"
            )


def main_pipeline() -> None:
    parser = argparse.ArgumentParser(
        description="Point cloud denoise/cluster/trajectory pipeline"
    )
    parser.add_argument(
        "--input",
        default="captures/cloud_aggregated.ply",
        help="Input PLY file",
    )
    args = parser.parse_args()
    CloudPipeline().run(args.input)

