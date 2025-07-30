import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree

from farm import HAND_EYE_R, HAND_EYE_t
from save_rotate_clouds import TARGET_POSE, transform_cloud_to_tcp


def load_point_cloud(
    pcd_path: str, voxel_size: float = 0.002
) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    return pcd_down


def filter_bbox(
    pcd: o3d.geometry.PointCloud, bbox_points: np.ndarray
) -> o3d.geometry.PointCloud:
    bbox_pcd = o3d.geometry.PointCloud()
    bbox_pcd.points = o3d.utility.Vector3dVector(bbox_points)
    bbox = bbox_pcd.get_axis_aligned_bounding_box()
    return pcd.crop(bbox)


def segment_dbscan(
    pcd: o3d.geometry.PointCloud, eps: float = 0.01, min_points: int = 50
) -> list[o3d.geometry.PointCloud]:
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    clusters = []
    for label in np.unique(labels):
        if label < 0:
            continue
        cluster = pcd.select_by_index(np.where(labels == label)[0])
        clusters.append(cluster)
    return clusters


def get_main_axis(cluster: o3d.geometry.PointCloud) -> np.ndarray:
    points = np.asarray(cluster.points)
    pca = PCA(n_components=3)
    pca.fit(points)
    return pca.components_[0]


def get_obb(cluster: o3d.geometry.PointCloud) -> o3d.geometry.OrientedBoundingBox:
    return cluster.get_oriented_bounding_box()


def obb_aabb_intersect(
    obb1: o3d.geometry.OrientedBoundingBox, obb2: o3d.geometry.OrientedBoundingBox
) -> bool:
    aabb1 = obb1.get_axis_aligned_bounding_box()
    aabb2 = obb2.get_axis_aligned_bounding_box()
    min1 = np.asarray(aabb1.get_min_bound())
    max1 = np.asarray(aabb1.get_max_bound())
    min2 = np.asarray(aabb2.get_min_bound())
    max2 = np.asarray(aabb2.get_max_bound())
    # Intersection if all intervals overlap in all axes
    return np.all(max1 >= min2) and np.all(max2 >= min1)


def intersect_clusters_obb(
    obbs: list[o3d.geometry.OrientedBoundingBox],
) -> list[tuple[int, int]]:
    intersections = []
    for i, obb1 in enumerate(obbs):
        for j, obb2 in enumerate(obbs[i + 1 :], start=i + 1):
            if obb_aabb_intersect(obb1, obb2):
                intersections.append((i, j))
    return intersections


def visualize_clusters_with_obbs(clusters, obbs):
    geometries = []
    for cluster, obb in zip(clusters, obbs):
        cluster.paint_uniform_color(np.random.rand(3))
        obb.color = (0, 0, 0)
        geometries.extend([cluster, obb])
    o3d.visualization.draw_geometries(geometries)


def find_main_plane(
    pcd: o3d.geometry.PointCloud,
    distance_threshold=0.004,
    ransac_n=3,
    num_iterations=1000,
):
    """
    Fit main supporting plane using RANSAC.
    Returns: plane_model, inliers (indices), plane_cloud, rest_cloud
    """
    plane_model, inliers = pcd.segment_plane(
        distance_threshold, ransac_n, num_iterations
    )
    plane_cloud = pcd.select_by_index(inliers)
    rest_cloud = pcd.select_by_index(inliers, invert=True)
    return plane_model, inliers, plane_cloud, rest_cloud


def filter_rectangular_clusters(clusters, min_aspect_ratio=2.0, min_length=0.05):
    filtered = []
    for cluster in clusters:
        obb = cluster.get_oriented_bounding_box()
        sizes = np.sort(obb.extent)
        # Keep only clusters with pronounced length
        aspect = sizes[2] / sizes[1] if sizes[1] > 0 else 0
        if aspect > min_aspect_ratio and sizes[2] > min_length:
            filtered.append(cluster)
    return filtered


# High-level function to run the complete pipeline quickly
def run_truss_segmentation_pipeline(pcd_path: str, bbox_points: np.ndarray):
    pcd = load_point_cloud(pcd_path)
    pcd = transform_cloud_to_tcp(pcd, HAND_EYE_R, HAND_EYE_t, TARGET_POSE)
    pcd_filtered = filter_bbox(pcd, bbox_points)

    # NEW: find main plane and get points "над" ней
    plane_model, inliers, plane_cloud, rest_cloud = find_main_plane(pcd_filtered)
    print(f"Main plane equation: {plane_model}, points on plane: {len(inliers)}")

    # Кластеризация только по тем, что вне плоскости (или рядом с ней)
    clusters = segment_dbscan(plane_cloud, eps=0.05, min_points=400)
    # clusters = filter_rectangular_clusters(clusters)
    obbs = [get_obb(cluster) for cluster in clusters]
    main_axes = [get_main_axis(cluster) for cluster in clusters]
    intersections = intersect_clusters_obb(obbs)
    print("Clusters found:", len(clusters))
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i}: {len(cluster.points)} points")
    # Визуализируем всё: плоскость (основу) + кластеры
    geometries = [plane_cloud]
    for cluster, obb in zip(clusters, obbs):
        cluster.paint_uniform_color(np.random.rand(3))
        obb.color = (0, 0, 0)
        geometries.extend([cluster, obb])
    o3d.visualization.draw_geometries(geometries)

    return {
        "plane_cloud": plane_cloud,
        "clusters": clusters,
        "obbs": obbs,
        "main_axes": main_axes,
        "intersections": intersections,
    }


# Example usage (can integrate into existing farm.py logic)
if __name__ == "__main__":
    BBOX_POINTS = np.array(
        [
            [-0.57, -0.34, 0.46],
            [-0.57, -0.05, 0.27],
            [-0.38, -0.05, 0.27],
            [-0.38, -0.05, 0.46],
        ]
    )
    result = run_truss_segmentation_pipeline(
        ".data_clouds/farm_20250730_143651.ply", BBOX_POINTS
    )
    for idx, axis in enumerate(result["main_axes"]):
        print(f"Cluster {idx} main axis: {axis}")
    print("Intersections:", result["intersections"])
