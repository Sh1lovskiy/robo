# bounding_box_graph.py
import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA

from farm import HAND_EYE_R, TARGET_POSE, HAND_EYE_t
from farm1 import filter_bbox, load_point_cloud
from save_rotate_clouds import transform_cloud_to_tcp


def visualize_bbox_graph(clusters, obbs, edges):
    """
    Visualize clusters with their bounding boxes and connecting lines for intersecting pairs.
    """
    geometries = []
    for cluster, obb in zip(clusters, obbs):
        cluster.paint_uniform_color(np.random.rand(3))
        obb.color = (0, 0, 0)
        geometries.extend([cluster, obb])

    centers = [obb.get_center() for obb in obbs]
    edge_lines = []
    edge_points = o3d.utility.Vector3dVector(centers)
    for i, j in edges:
        edge_lines.append([i, j])
    if edge_lines:
        line_set = o3d.geometry.LineSet(
            points=edge_points, lines=o3d.utility.Vector2iVector(edge_lines)
        )
        line_set.paint_uniform_color([1, 0, 0])
        geometries.append(line_set)

    o3d.visualization.draw_geometries(geometries)


def segment_clusters(pcd, eps=0.006, min_points=40):
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    clusters = []
    for label in np.unique(labels):
        if label >= 0:
            clusters.append(pcd.select_by_index(np.where(labels == label)[0]))
    return clusters


def compute_bounding_boxes(clusters):
    obbs = []
    for cluster in clusters:
        try:
            obb = cluster.get_oriented_bounding_box()
            # Проверка: если extent по 2 осям очень мал, пропускать
            if np.sum(np.array(obb.extent) < 1e-4) >= 2:
                continue
            obbs.append(obb)
        except Exception as e:
            print("OBB error:", e)
            continue
    return obbs


def filter_parallelepiped_clusters(clusters, min_dim=0.0001, max_aspect_ratio=4.0):
    """Оставляет только похожие на параллелепипед кластеры."""
    filtered = []
    for cluster in clusters:
        obb = cluster.get_oriented_bounding_box()
        sizes = np.array(obb.extent)
        min_side = np.min(sizes)
        max_side = np.max(sizes)
        aspect = max_side / (min_side + 1e-6)
        # Не берем слишком вытянутые или слишком тонкие
        if min_side > min_dim and aspect < max_aspect_ratio:
            filtered.append(cluster)
    return filtered


def obb_intersect(obb1, obb2):
    aabb1, aabb2 = (
        obb1.get_axis_aligned_bounding_box(),
        obb2.get_axis_aligned_bounding_box(),
    )
    min1, max1 = aabb1.get_min_bound(), aabb1.get_max_bound()
    min2, max2 = aabb2.get_min_bound(), aabb2.get_max_bound()
    return np.all(max1 >= min2) and np.all(max2 >= min1)


def build_bbox_graph(obbs):
    edges = []
    for i, obb1 in enumerate(obbs):
        for j, obb2 in enumerate(obbs[i + 1 :], start=i + 1):
            if obb_intersect(obb1, obb2):
                edges.append((i, j))
    return edges


# clustering_ransac_features.py
import numpy as np
import open3d as o3d


def segment_clusters_dbscan(pcd, eps=0.008, min_points=50):
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    clusters = []
    for label in np.unique(labels):
        if label >= 0:
            clusters.append(pcd.select_by_index(np.where(labels == label)[0]))
    return clusters


def fit_ransac_primitives(
    cluster, distance_threshold=0.002, ransac_n=3, num_iterations=500
):
    points = np.asarray(cluster.points)
    plane_model, inliers = cluster.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    plane_cloud = cluster.select_by_index(inliers)
    remaining_cloud = cluster.select_by_index(inliers, invert=True)
    return plane_model, plane_cloud, remaining_cloud


def extract_features_from_clusters(clusters):
    cluster_features = []
    for cluster in clusters:
        plane_model, plane_cloud, remaining_cloud = fit_ransac_primitives(cluster)
        cluster_features.append(
            {
                "plane_model": plane_model,
                "plane_cloud": plane_cloud,
                "remaining_cloud": remaining_cloud,
            }
        )
    return cluster_features


# Example usage (integration)
if __name__ == "__main__":
    BBOX_POINTS = np.array(
        [
            [-0.57, -0.34, 0.46],
            [-0.57, -0.05, 0.27],
            [-0.38, -0.05, 0.27],
            [-0.38, -0.05, 0.46],
        ]
    )
    pcd_path = ".data_clouds/farm_20250730_143651.ply"
    pcd = load_point_cloud(pcd_path)
    pcd = transform_cloud_to_tcp(pcd, HAND_EYE_R, HAND_EYE_t, TARGET_POSE)
    pcd_filtered = filter_bbox(pcd, BBOX_POINTS)
    clusters = segment_clusters(pcd_filtered)
    clusters = filter_parallelepiped_clusters(clusters)
    # Bounding box graph
    obbs = [cluster.get_oriented_bounding_box() for cluster in clusters]

    bbox_edges = build_bbox_graph(obbs)
    print("BBox Graph edges:", bbox_edges)
    visualize_bbox_graph(clusters, obbs, bbox_edges)

    # RANSAC feature extraction
    ransac_features = extract_features_from_clusters(clusters)
    for idx, feature in enumerate(ransac_features):
        print(f"Cluster {idx}: Plane model {feature['plane_model']}")
