# cli/vis_pointcloud.py
"""Point cloud pipeline: denoise, crop, cluster, find top face & bounding box, plan marker trajectory, visualize."""

import open3d as o3d
import numpy as np
from utils.logger import Logger

ROI_LIMITS = {
    "x": (-0.3, -0.1),
    "y": (-0.2, 0.2),
    "z": (0.02, 0.1),
}


class PointCloudDenoiser:
    """Removes outliers and noise from the input cloud."""

    def __init__(self, nb_neighbors=30, std_ratio=5.0, logger=None):
        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio
        self.logger = logger or Logger.get_logger("cli.vis_pointcloud.denoiser")

    def denoise(self, pcd):
        cl, ind = pcd.remove_statistical_outlier(
            nb_neighbors=self.nb_neighbors, std_ratio=self.std_ratio
        )
        denoised = pcd.select_by_index(ind)
        n_removed = len(pcd.points) - len(denoised.points)
        self.logger.info(
            f"Removed {n_removed} outliers; remaining: {len(denoised.points)} points"
        )
        return denoised


class PointCloudCropper:
    """ROI-based cloud cropping."""

    def __init__(self, limits=ROI_LIMITS, logger=None):
        self.limits = limits
        self.logger = logger or Logger.get_logger("cli.vis_pointcloud.cropper")

    def crop(self, pcd):
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
    """Clusters cloud using DBSCAN, returns the largest cluster (object)."""

    def __init__(self, eps=0.05, min_points=100, logger=None):
        self.eps = eps
        self.min_points = min_points
        self.logger = logger or Logger.get_logger("cli.vis_pointcloud.clusterer")

    def extract_object(self, pcd):
        labels = np.array(
            pcd.cluster_dbscan(
                eps=self.eps, min_points=self.min_points, print_progress=True
            )
        )
        n_clusters = labels.max() + 1
        self.logger.info(f"Found {n_clusters} clusters (label -1 = noise)")
        if n_clusters == 0:
            raise RuntimeError("No clusters found!")
        largest_label = np.bincount(labels[labels >= 0]).argmax()
        object_indices = np.where(labels == largest_label)[0]
        object_pcd = pcd.select_by_index(object_indices)
        self.logger.info(f"Object cluster: {len(object_pcd.points)} points")
        return object_pcd


class ObjectSurfaceAnalyzer:
    """Finds bounding box of the object."""

    def __init__(self, logger=None):
        self.logger = logger or Logger.get_logger("cli.vis_pointcloud.analyzer")

    def get_bounding_box(self, pcd):
        aabb = pcd.get_axis_aligned_bounding_box()
        aabb.color = (0, 1, 0)
        corners = np.asarray(aabb.get_box_points())
        for idx, pt in enumerate(corners):
            self.logger.info(f"Bounding box corner {idx}: {pt}")
        return aabb, corners


class TopFaceFinder:
    """
    Finds points belonging to the top face of the clustered object.
    The top face is defined as the set of points with Z close to the max Z.
    """

    def __init__(self, z_tol=0.01, logger=None):
        self.z_tol = z_tol  # tolerance in meters
        self.logger = logger or Logger.get_logger("cli.vis_pointcloud.topface")

    def find_top_face(self, pcd):
        points = np.asarray(pcd.points)
        max_z = np.max(points[:, 2])
        mask = np.abs(points[:, 2] - max_z) < self.z_tol
        top_points = points[mask]
        self.logger.info(f"Top face: found {len(top_points)} points at Z={max_z:.5f}")
        if len(top_points) < 4:
            self.logger.warning("Too few points detected on top face.")
        return top_points


class TopFaceTrajectoryPlanner:
    """
    Plans a trajectory along the center of the top face, along its main axis,
    considering the marker length offset (marker at tool tip).
    """

    def __init__(self, marker_length_mm=210, logger=None):
        self.marker_length = marker_length_mm / 1000.0  # to meters
        self.logger = logger or Logger.get_logger("cli.vis_pointcloud.trajectory")

    def plan_center_trajectory(self, top_points):
        # Use PCA to find the main direction of the top face (assume rectangle/elongated).
        centroid = np.mean(top_points, axis=0)
        centered = top_points - centroid
        u, s, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0]  # Main axis
        length = np.max(centered @ direction) - np.min(centered @ direction)
        start_proj = np.min(centered @ direction)
        end_proj = np.max(centered @ direction)
        start = centroid + start_proj * direction
        end = centroid + end_proj * direction

        offset = np.array([0, 0, self.marker_length])
        traj_start = start + offset
        traj_end = end + offset

        self.logger.info(
            f"Planned marker trajectory from {np.round(traj_start, 5)} to {np.round(traj_end, 5)} (center of top face)"
        )
        return traj_start, traj_end


class CloudPipeline:
    """Full cloud denoise, object isolation, top face analysis, trajectory planning, visualization."""

    def __init__(self, logger=None):
        self.logger = logger or Logger.get_logger("cli.vis_pointcloud.pipeline")
        self.denoiser = PointCloudDenoiser(logger=self.logger)
        self.cropper = PointCloudCropper(logger=self.logger)
        self.clusterer = PointCloudClusterer(logger=self.logger)
        self.analyzer = ObjectSurfaceAnalyzer(logger=self.logger)
        self.topfinder = TopFaceFinder(logger=self.logger)
        self.trajectory_planner = TopFaceTrajectoryPlanner(logger=self.logger)

    def run(self, input_ply):
        pcd = o3d.io.read_point_cloud(input_ply)
        self.logger.info(f"Loaded {input_ply}, {len(pcd.points)} points")
        o3d.visualization.draw_geometries([pcd], window_name="Raw cloud")
        cropped = self.cropper.crop(pcd)
        o3d.visualization.draw_geometries([cropped], window_name="Cropped cloud")
        clean = self.denoiser.denoise(cropped)
        o3d.visualization.draw_geometries([clean], window_name="Denoised cloud")
        obj = self.clusterer.extract_object(clean)
        o3d.visualization.draw_geometries([obj], window_name="Clustered object")
        aabb, corners = self.analyzer.get_bounding_box(obj)
        o3d.visualization.draw_geometries([obj, aabb], window_name="Bounding box")
        top_points = self.topfinder.find_top_face(obj)
        if len(top_points) > 0:
            top_pcd = o3d.geometry.PointCloud()
            top_pcd.points = o3d.utility.Vector3dVector(top_points)
            top_pcd.paint_uniform_color([1, 1, 0])
            o3d.visualization.draw_geometries(
                [obj, top_pcd], window_name="Top face points"
            )
            centroid = np.mean(top_points, axis=0)
            print("Centroid of top face:", np.round(centroid, 5))
            traj_start, traj_end = self.trajectory_planner.plan_center_trajectory(
                top_points
            )
            print("Marker trajectory (tool tip should move):")
            print("  Start:", np.round(traj_start, 5))
            print("  End:  ", np.round(traj_end, 5))
            traj_line = o3d.geometry.LineSet()
            traj_line.points = o3d.utility.Vector3dVector([traj_start, traj_end])
            traj_line.lines = o3d.utility.Vector2iVector([[0, 1]])
            traj_line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # red
            o3d.visualization.draw_geometries(
                [obj, top_pcd, traj_line], window_name="Marker Trajectory"
            )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Point cloud denoise/ROI/cluster/topface/bbox/trajectory/visualize pipeline"
    )
    parser.add_argument(
        "--input",
        default="captures/cloud_aggregated.ply",
        help="Input PLY file",
    )
    args = parser.parse_args()
    pipeline = CloudPipeline()
    pipeline.run(args.input)


if __name__ == "__main__":
    main()
