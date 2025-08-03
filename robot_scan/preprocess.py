"""Point cloud preprocessing utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import open3d as o3d

from utils.logger import Logger

logger = Logger.get_logger("robot_scan.preprocess")


def downsample_cloud(
    cloud: o3d.geometry.PointCloud, voxel: float = 0.002
) -> o3d.geometry.PointCloud:
    """Voxel down-sample the point cloud."""
    return cloud.voxel_down_sample(voxel)


def crop_cloud(
    cloud: o3d.geometry.PointCloud, bbox_points: np.ndarray
) -> o3d.geometry.PointCloud:
    """Crop cloud with axis-aligned bounding box defined by ``bbox_points``."""
    bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(bbox_points)
    )
    return cloud.crop(bbox)


def segment_plane(
    cloud: o3d.geometry.PointCloud,
    threshold: float = 0.004,
    ransac_n: int = 3,
    iterations: int = 1000,
) -> Tuple[o3d.geometry.PointCloud, np.ndarray, np.ndarray]:
    """Segment the dominant plane from the cloud.

    Returns
    -------
    plane : o3d.geometry.PointCloud
        Points belonging to the plane.
    normal : np.ndarray
        Plane normal vector.
    center : np.ndarray
        Centroid of the plane.
    """
    plane_model, inliers = cloud.segment_plane(
        distance_threshold=threshold, ransac_n=ransac_n, num_iterations=iterations
    )
    plane = cloud.select_by_index(inliers)
    normal = np.array(plane_model[:3])
    center = np.asarray(plane.points).mean(axis=0)
    logger.info(f"Plane segmented: {len(plane.points)} points")
    return plane, normal, center


def compute_plane_axes(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return principal axis and normal for given plane points."""
    center = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - center)
    main_axis = vt[0]
    normal = vt[2]
    return main_axis, normal
