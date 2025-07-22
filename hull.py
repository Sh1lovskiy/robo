import open3d as o3d
import numpy as np
import re
from scipy.spatial.transform import Rotation as R
from utils.logger import Logger


def parse_pose_from_filename(filename):
    name = filename[:-4] if filename.endswith(".ply") else filename
    match = re.search(
        r"x([-\d.]+)_y([-\d.]+)_z([-\d.]+)_rx([-\d.]+)_ry([-\d.]+)_rz([-\d.]+)", name
    )
    if not match:
        raise ValueError(f"Filename '{filename}' does not match pose pattern!")
    return map(float, match.groups())


def get_transform(x, y, z, rx, ry, rz):
    t = np.array([x, y, z], dtype=np.float64) / 1000.0
    Rmat = R.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = Rmat
    T[:3, 3] = t
    return T


def load_and_transform_cloud(filename):
    x, y, z, rx, ry, rz = parse_pose_from_filename(filename)
    pcd = o3d.io.read_point_cloud(filename)
    T = get_transform(x, y, z, rx, ry, rz)
    pcd.transform(T)
    return pcd


def refine_with_icp(clouds, logger):
    ref = clouds[0]
    refined_clouds = [ref]
    for pc in clouds[1:]:
        pc_down = pc.voxel_down_sample(voxel_size=0.005)
        ref_down = ref.voxel_down_sample(voxel_size=0.005)
        reg = o3d.pipelines.registration.registration_icp(
            pc_down,
            ref_down,
            max_correspondence_distance=0.02,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            init=np.eye(4),
        )
        logger.info(f"ICP fitness: {reg.fitness:.5f}  rmse: {reg.inlier_rmse:.6f}")
        pc.transform(reg.transformation)
        refined_clouds.append(pc)
    merged = refined_clouds[0]
    for pc in refined_clouds[1:]:
        merged += pc
    return merged


def smooth_point_cloud_knn(pcd, k=10):
    points = np.asarray(pcd.points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    new_points = []
    for i in range(points.shape[0]):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(points[i], k)
        mean_pt = points[idx, :].mean(axis=0)
        new_points.append(mean_pt)
    pcd_smooth = o3d.geometry.PointCloud()
    pcd_smooth.points = o3d.utility.Vector3dVector(np.array(new_points))
    if pcd.has_colors():
        pcd_smooth.colors = pcd.colors
    return pcd_smooth


def segment_largest_component(mesh, min_triangles=50, logger=None):
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    big_cluster_id = np.argmax(cluster_n_triangles)
    idx = np.where(triangle_clusters == big_cluster_id)[0]
    mesh_largest = mesh.select_by_index(idx.tolist(), cleanup=True)
    mesh_largest.compute_vertex_normals()
    if logger:
        logger.info(f"Largest component: triangles={len(mesh_largest.triangles)}")
    return mesh_largest


def main():
    logger = Logger.get_logger("hull")
    filenames = [
        "x-156.9_y-53.8_z586.8_rx-141.3_ry4.8_rz112.0.ply",
        "x-170.5_y85.5_z566.6_rx-132.5_ry1.1_rz130.8.ply",
        "x-203.5_y220.4_z499.0_rx-119.1_ry17.4_rz138.6.ply",
    ]
    logger.info("Loading and aligning clouds...")
    clouds = [load_and_transform_cloud(fn) for fn in filenames]
    logger.info("Refining alignment via ICP...")
    cloud_merged = refine_with_icp(clouds, logger)

    logger.info("Downsampling and smoothing merged cloud...")
    cloud_merged = cloud_merged.voxel_down_sample(voxel_size=0.005)
    cloud_merged = smooth_point_cloud_knn(cloud_merged, k=10)
    o3d.visualization.draw_geometries([cloud_merged])

    logger.info("Estimating normals...")
    cloud_merged.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30)
    )

    logger.info("Defining bounding box...")
    bbox_points = np.array(
        [
            [-0.42, -0.093, 0.3],
            [-0.39, -0.3, 0.29],
            [-0.48, -0.3, 0.36],
            [-0.32, -0.3, 0.19],
        ]
    )
    pcd_sel = o3d.geometry.PointCloud()
    pcd_sel.points = o3d.utility.Vector3dVector(bbox_points)
    aabb = pcd_sel.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)

    logger.info("Cropping cloud to bounding box...")
    cloud_cropped = cloud_merged.crop(aabb)
    cloud_cropped = cloud_cropped.voxel_down_sample(voxel_size=0.003)
    cloud_cropped.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=50)
    )

    logger.info("Alpha shape surface (alpha=0.02)...")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        cloud_cropped, alpha=0.02
    )
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0, 0, 1])

    largest_face = segment_largest_component(mesh, logger=logger)
    largest_face.paint_uniform_color([1, 0, 0])

    triangles = np.asarray(largest_face.triangles)
    vertices = np.asarray(largest_face.vertices)
    edge_set = set()
    for tri in triangles:
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            e = tuple(sorted([tri[a], tri[b]]))
            edge_set.add(e)
    edge_list = list(edge_set)
    lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(vertices),
        lines=o3d.utility.Vector2iVector(edge_list),
    )
    lineset.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in edge_list])

    o3d.visualization.draw_geometries(
        [cloud_cropped, mesh, largest_face, lineset, aabb]
    )
    o3d.visualization.draw_geometries(
        [cloud_merged, aabb, lineset],
        width=1280,
        height=900,
    )


if __name__ == "__main__":
    main()
