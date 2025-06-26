from __future__ import annotations

import argparse
import numpy as np
import open3d as o3d
from utils.cli import Command, CommandDispatcher
from utils.logger import Logger, LoggerType

"""Point cloud processing pipeline utilities."""


ROI_LIMITS = {"x": (-0.6, -0.1), "y": (-0.2, 0.2), "z": (0.001, 0.06)}


class PointCloudDenoiser:
    """Remove outliers from a point cloud."""

    def __init__(
        self,
        nb_neighbors: int = 30,
        std_ratio: float = 5.0,
        logger: LoggerType | None = None,
    ) -> None:
        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio
        self.logger = logger or Logger.get_logger("vision.pipeline.denoiser")

    def denoise(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        cl, ind = pcd.remove_statistical_outlier(
            nb_neighbors=self.nb_neighbors, std_ratio=self.std_ratio
        )
        denoised = pcd.select_by_index(ind)
        self.logger.info(f"Removed {len(pcd.points) - len(denoised.points)} outliers")
        return denoised


class PointCloudCropper:
    """Crop cloud to a region of interest."""

    def __init__(
        self,
        limits: dict[str, tuple[float, float]] = ROI_LIMITS,
        logger: LoggerType | None = None,
    ) -> None:
        self.limits = limits
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

    def __init__(
        self,
        eps: float = 0.05,
        min_points: int = 100,
        logger: LoggerType | None = None,
    ) -> None:
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

    def __init__(self, logger: LoggerType | None = None) -> None:
        self.logger = logger or Logger.get_logger("vision.pipeline.analyzer")

    def get_bounding_box(
        self, pcd: o3d.geometry.PointCloud
    ) -> tuple[o3d.geometry.AxisAlignedBoundingBox, np.ndarray]:
        aabb = pcd.get_axis_aligned_bounding_box()
        aabb.color = (0, 1, 0)
        return aabb, np.asarray(aabb.get_box_points())


class TopFaceFinder:
    """Find points belonging to the top face."""

    def __init__(self, z_tol: float = 0.01, logger: LoggerType | None = None) -> None:
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

    def __init__(
        self, marker_length_mm: float = 195.0, logger: LoggerType | None = None
    ) -> None:
        self.marker_length = marker_length_mm / 1000.0
        self.logger = logger or Logger.get_logger("vision.pipeline.traj")

    def plan_center_trajectory(
        self, top_points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
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


class CloudAnalyzer:
    """Full visualization and analysis pipeline."""

    def __init__(self, logger: LoggerType | None = None) -> None:
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
        o3d.visualization.draw_geometries_with_editing([pcd], window_name="Raw cloud")

        cropped = self.cropper.crop(pcd)
        # o3d.visualization.draw_geometries_with_editing(
        #     [cropped], window_name="Cropped cloud"
        # )

        clean = self.denoiser.denoise(cropped)
        o3d.visualization.draw_geometries_with_editing(
            [clean], window_name="Denoised cloud"
        )

        obj = self.clusterer.extract_object(clean)
        # o3d.visualization.draw_geometries([obj], window_name="Clustered object")

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
            traj_start, traj_end = self.trajectory_planner.plan_center_trajectory(
                top_points
            )
            traj_line = o3d.geometry.LineSet()
            traj_line.points = o3d.utility.Vector3dVector([traj_start, traj_end])
            traj_line.lines = o3d.utility.Vector2iVector([[0, 1]])
            traj_line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
            o3d.visualization.draw_geometries(
                [obj, top_pcd, traj_line], window_name="Marker Trajectory"
            )


def _add_cloud_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input_ply",
        default="cloud_new_1/cloud_aggregated.ply",
        help="Path to input PLY point cloud file",
    )


def _run_cloud(args: argparse.Namespace) -> None:
    pipeline = CloudAnalyzer()
    pipeline.run(args.input_ply)


if __name__ == "__main__":
    dispatcher = CommandDispatcher(
        description="Point cloud processing pipeline",
        commands=[
            Command("cloud", _run_cloud, _add_cloud_args, "Run Point Cloud pipeline"),
        ],
    )
    dispatcher.run()
