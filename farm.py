"""
farm.py

Automated cycle:
 - Acquire point cloud from camera
 - Transform to TCP frame
 - Build skeleton graph (via skelet.py)
 - Visualize target points at nodes with 20cm offset along normal
 - Move robot to each point, enable laser
 - Rotate part with stepper
 - Proceed to next iteration
"""

import os
import random
import time
import numpy as np
import open3d as o3d
import networkx as nx
import threading
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from esp32.control import ESP32Controller
from utils.logger import Logger
from skelet import run_pipeline, run_visualization
from rectangle import (
    robot_connect,
    robot_movej,
    get_camera_cloud,
    robot_movel,
    save_cloud_timestamped,
    transform_cloud_to_tcp,
)


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


logger = Logger.get_logger("farm")

ROBOT_IP = "192.168.58.2"
STEPS_PER_ITER = [50, 50, 50, 50]
LASER_ON_TIME = 1.0
MOVE_DELAY = 0.5

HAND_EYE_R = np.array(
    [
        [0.999048, 0.02428, -0.03625],
        [-0.02706, 0.99658, -0.07804],
        [0.03423, 0.07895, 0.99629],
    ]
)
HAND_EYE_t = np.array([-0.03424, -0.07905, 0.00128]).reshape(3, 1)
TARGET_POSE = np.array([3.63, -103.5, 540.2, -120.2, -1.13, 103.5])
BBOX_POINTS = np.array(
    [
        [-0.57, -0.34, 0.46],
        [-0.57, -0.05, 0.27],
        [-0.38, -0.05, 0.27],
        [-0.38, -0.05, 0.46],
    ]
)


def get_plane_main_axes(plane):
    """
    Return fixed canonical axes for the plane:
    - X axis: always upward (world +Z)
    - Z axis: toward the camera (opposite of plane normal)
    - Y axis: leftward (completes right-handed system)
    """
    pts = np.asarray(plane.points)
    center = np.mean(pts, axis=0)
    _, _, Vt = np.linalg.svd(pts - center, full_matrices=False)

    normal = Vt[2]
    if normal[2] < 0:
        normal = -normal  # Z always toward camera

    x_axis = Vt[0]  # fixed
    y_axis = np.cross(normal, x_axis)  # leftward
    y_axis /= np.linalg.norm(y_axis)

    x_axis = np.cross(y_axis, normal)  # re-orthogonalize
    x_axis /= np.linalg.norm(x_axis)

    return x_axis, normal


def align_arrow(mesh, direction):
    """Align an O3D mesh arrow to a direction vector."""
    src = np.array([0, 0, 1], dtype=np.float64)
    dst = direction / np.linalg.norm(direction)
    v = np.cross(src, dst)
    c = np.dot(src, dst)
    if np.allclose(v, 0):
        R_ = (
            np.eye(3)
            if c > 0
            else o3d.geometry.get_rotation_matrix_from_axis_angle(
                np.pi * np.array([1, 0, 0])
            )
        )
    else:
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R_ = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s**2))
    mesh.rotate(R_, center=np.zeros(3))
    return mesh


def compute_tcp_poses(
    target_points: np.ndarray,
    main_axis: np.ndarray,
    plane_normal: np.ndarray,
    offset_xyz: tuple[float, float, float] = (0.03, 0.017, -0.35),
    frame_size: float = 0.05,
):
    """
    Build TCP poses ([x, y, z, rx, ry, rz]) and O3D frames for each target point.
    X: main_axis, Z: plane_normal, Y: right-hand rule.
    """
    x_axis = main_axis / np.linalg.norm(main_axis)
    z_axis = -plane_normal / np.linalg.norm(plane_normal)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    R_tcp = np.column_stack((x_axis, y_axis, z_axis))
    offset_vec = R_tcp @ np.array(offset_xyz, dtype=np.float64)
    poses, frames = [], []
    for pt in target_points:
        T = np.eye(4)
        T[:3, :3] = R_tcp
        T[:3, 3] = pt + offset_vec
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
        mesh.transform(T)
        frames.append(mesh)
        pose = np.concatenate(
            [T[:3, 3] * 1000, R.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)]
        )
        poses.append(pose)
    return poses, frames


def cluster_nodes_2d(
    raw_points: np.ndarray, radius: float = 0.03, tree: cKDTree | None = None
) -> tuple[np.ndarray, dict[int, int]]:
    """
    Cluster nearby 2D points (XY projection) within a given radius.
    Each cluster is averaged into a single node. Returns the merged nodes
    and a mapping from original point indices to cluster indices.
    """
    if tree is None:
        tree = cKDTree(raw_points[:, :2])

    visited = np.zeros(len(raw_points), dtype=bool)
    merged = []
    index_map = {}

    for i in range(len(raw_points)):
        if visited[i]:
            continue
        # find all neighbors in radius in XY space
        idxs = tree.query_ball_point(raw_points[i], radius)
        cluster = raw_points[idxs]
        new_point = np.mean(cluster, axis=0)
        new_idx = len(merged)
        merged.append(new_point)
        # map all clustered points to new index
        for j in idxs:
            index_map[j] = new_idx
        visited[idxs] = True

    return np.array(merged), index_map


def rebuild_edges(
    branches_xyz: list[np.ndarray],
    raw_points: np.ndarray,
    index_map: dict[int, int],
    tree: cKDTree,
) -> list[list[int]]:
    """
    Convert original 3D branches into a set of unique undirected edges between merged nodes.
    Each branch is reduced to its start and end point and mapped through the clustering.
    """
    edges = set()
    for branch in branches_xyz:
        # find closest raw point index to branch endpoints
        i_raw = tree.query(branch[0])[1]
        j_raw = tree.query(branch[-1])[1]
        # map to merged node index
        i = index_map.get(i_raw)
        j = index_map.get(j_raw)
        if i is not None and j is not None and i != j:
            edges.add(tuple(sorted((i, j))))
    return [list(e) for e in edges]


def build_graph(nodes: np.ndarray, edges: list[list[int]]) -> nx.Graph:
    """
    Build an undirected graph from 3D node positions and edges (by index).
    Each edge is weighted by Euclidean distance.
    """
    G = nx.Graph()
    for i, coord in enumerate(nodes):
        G.add_node(i, pos=coord)
    for i, j in edges:
        dist = np.linalg.norm(nodes[i] - nodes[j])
        G.add_edge(i, j, weight=dist)
    return G


def compute_pose_order(G: nx.Graph, poses: list[np.ndarray]) -> list[int]:
    """
    Given a graph and TCP poses, returns a greedy ordered list of node indices
    based on Dijkstra shortest paths between successive nodes.
    """
    nodes = np.array([p[:3] for p in poses])
    current_pos = poses[0][:3]
    # start from closest node to current TCP pose
    start_idx = np.argmin(np.linalg.norm(nodes - current_pos, axis=1))
    reachable = set(nx.node_connected_component(G, start_idx))
    pool = set(range(len(poses))) & reachable
    order, current = [], start_idx
    while pool:
        # find next closest node in graph distance
        next_idx = min(
            pool, key=lambda t: nx.shortest_path_length(G, current, t, weight="weight")
        )
        order.append(next_idx)
        pool.remove(next_idx)
        current = next_idx
    return order


def render_path_visualization(
    nodes: np.ndarray,
    edges: list[list[int]],
    o3d_branches: list[list[int]],
    o3d_nodes: list[int],
    order: list[int],
    plane: o3d.geometry.PointCloud,
    main_axis: np.ndarray,
    plane_normal: np.ndarray,
    frames: list[o3d.geometry.TriangleMesh],
):
    """
    Visualize clustered nodes, graph edges, planned route and TCP poses in Open3D.
    Also adds orientation arrows for main axis and surface normal.
    """
    # point cloud of nodes
    node_pcd = o3d.geometry.PointCloud()
    node_pcd.points = o3d.utility.Vector3dVector(nodes)
    node_pcd.paint_uniform_color([1, 0.4, 0])

    # Dijkstra route
    route_lines = [[order[i], order[i + 1]] for i in range(len(order) - 1)]
    route_ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(nodes),
        lines=o3d.utility.Vector2iVector(route_lines),
    )
    route_ls.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(route_lines))

    # all edges in graph
    graph_ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(nodes),
        lines=o3d.utility.Vector2iVector(edges),
    )
    graph_ls.colors = o3d.utility.Vector3dVector([[0, 0, 0]] * len(edges))

    center = np.mean(nodes, axis=0)
    arrow_len = 0.025

    # main axis arrow
    arrow_main = align_arrow(
        o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.001,
            cone_radius=0.003,
            cylinder_height=arrow_len * 0.8,
            cone_height=arrow_len * 0.2,
        ),
        main_axis,
    )
    arrow_main.paint_uniform_color([1, 0.1, 0.1])
    arrow_main.translate(center)

    # normal vector arrow
    arrow_normal = align_arrow(
        o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.001,
            cone_radius=0.003,
            cylinder_height=arrow_len * 0.8,
            cone_height=arrow_len * 0.2,
        ),
        plane_normal,
    )
    arrow_normal.paint_uniform_color([0.1, 0.2, 1])
    arrow_normal.translate(center)

    run_visualization(
        plane=plane,
        o3d_branches=o3d_branches,
        o3d_nodes=o3d_nodes,
        graph_lineset=graph_ls,
        arrow_main=arrow_main,
        arrow_normal=arrow_normal,
        frames=frames,
    )


def prepare_and_visualize_tcp_path(
    node_coords_3d: np.ndarray,
    branches_3d: list[np.ndarray],
    o3d_branches,
    o3d_nodes,
    plane: o3d.geometry.PointCloud,
    main_axis: np.ndarray,
    plane_normal: np.ndarray,
    offset_xyz=(0.0, 0.0, -0.2),
    radius=0.02,
    frame_size=0.04,
) -> list[np.ndarray]:
    """
    Complete pipeline:
    - Cluster node positions in 2D
    - Map branches to edges
    - Build graph
    - Compute TCP poses and sort them greedily by distance
    - Visualize result
    """
    raw = np.asarray(node_coords_3d)
    tree = cKDTree(raw)
    merged, index_map = cluster_nodes_2d(raw, radius, tree)
    edges = rebuild_edges(branches_3d, raw, index_map, tree)
    poses, frames = compute_tcp_poses(
        merged, main_axis, plane_normal, offset_xyz=offset_xyz, frame_size=frame_size
    )
    G = build_graph(merged, edges)
    order = compute_pose_order(G, poses)
    poses_in_path = [poses[i] for i in order]
    render_path_visualization(
        merged,
        edges,
        o3d_branches,
        o3d_nodes,
        order,
        plane,
        main_axis,
        plane_normal,
        frames,
    )
    return poses_in_path


def move_motor_threaded(esp32, steps, direction=1, delay_us=5000):
    """Run stepper movement in a thread with a small delay."""
    time.sleep(0.5)
    t = threading.Thread(target=esp32.move_motor, args=(steps, direction, delay_us))
    t.start()
    t.join()


def move_robot_to_targets(rpc, esp32, poses, laser_on_time, vel=35):
    """Move robot to each TCP pose, blink laser, handle exceptions."""
    for idx, pose in enumerate(poses):
        logger.info(f"MoveJ to target {idx+1}/{len(poses)}: {pose[:3]} mm")
        try:
            robot_movej(rpc, pose, vel=vel)
        except Exception as e:
            logger.error(f"Move error at {idx+1}: {e}")
            continue
        esp32.laser_on()
        time.sleep(laser_on_time)
    esp32.laser_off()


def single_farm_iteration(i, rpc, esp32):
    """One complete farm pass: acquire, process, visualize, move, rotate."""
    logger.info(f"--- Iteration {i + 1} ---")
    pcd = get_camera_cloud()
    save_cloud_timestamped(pcd)
    pcd_tcp, _, _ = transform_cloud_to_tcp(pcd, HAND_EYE_R, HAND_EYE_t, TARGET_POSE)
    graph = run_pipeline(
        cloud_path=None,
        bbox_points=BBOX_POINTS,
        cloud_obj=pcd_tcp,
        voxel_size=0.0001,
    )

    o3d.visualization.draw_geometries(graph["o3d_branches"] + graph["o3d_nodes"])
    main_axis, plane_normal = get_plane_main_axes(graph["plane"])
    if main_axis[0] < 0:
        main_axis = -main_axis
    CAMERA_DIR = np.array([0, 0, 1])
    if np.dot(plane_normal, CAMERA_DIR) < 0:
        plane_normal = -plane_normal
    logger.info("Visualizing target nodes")
    target_poses = prepare_and_visualize_tcp_path(graph, main_axis, plane_normal)
    logger.info(target_poses)
    input("Check visualization. Press Enter to start movement...")
    move_robot_to_targets(rpc, esp32, target_poses, LASER_ON_TIME)
    time.sleep(MOVE_DELAY)


def farm_cycle():
    """Main robot farm logic."""
    logger.info("=== ROBOT FARM CYCLE START ===")
    rpc = robot_connect(ROBOT_IP, safety_mode=False)
    esp32 = ESP32Controller()
    esp32.laser_off()
    robot_movej(rpc, TARGET_POSE, vel=40)
    for i, steps in enumerate(STEPS_PER_ITER):
        single_farm_iteration(i, rpc, esp32)
        if i < len(STEPS_PER_ITER) - 1:
            move_motor_threaded(esp32, steps)
            robot_movej(rpc, TARGET_POSE, vel=40)
            time.sleep(1)
    robot_movej(rpc, TARGET_POSE, vel=40)
    logger.success("=== ALL NODES PASSED ===")


def load_point_cloud(path):
    pcd = o3d.io.read_point_cloud(path)
    pcd.estimate_normals()
    logger.info(f"Loaded point cloud: {path} ({len(pcd.points)} points)")
    return pcd


def repair_surface(
    pcd: o3d.geometry.PointCloud, method="laplacian"
) -> o3d.geometry.PointCloud:
    """
    Repair and smooth a point cloud with missing regions or noisy borders.
    Supported methods:
    - "laplacian": Laplacian smoothing + outlier removal
    - "rolling": Open3D mesh + rolling-ball-like smoothing
    - "tsdf": TSDF voxel volume fusion (no color)
    """
    pcd = pcd.voxel_down_sample(0.002)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(50)

    if method == "laplacian":
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd, alpha=0.001
        )
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=5)
        mesh.compute_vertex_normals()
        return mesh.sample_points_poisson_disk(300000)

    else:
        raise ValueError(f"Unknown repair method: {method}")


def run_from_file(pcd_path):
    logger.info("=== RUNNING FROM FILE ===")
    pcd = load_point_cloud(pcd_path)
    pcd_tcp, _, _ = transform_cloud_to_tcp(pcd, HAND_EYE_R, HAND_EYE_t, TARGET_POSE)

    bbox = o3d.geometry.PointCloud()
    bbox.points = o3d.utility.Vector3dVector(BBOX_POINTS)
    cropped = pcd_tcp.crop(bbox.get_axis_aligned_bounding_box())

    methods = ["laplacian"]
    for method in methods:
        logger.info(f"=== TESTING REPAIR METHOD: {method.upper()} ===")
        repaired = repair_surface(cropped, method=method)
        o3d.visualization.draw_geometries([repaired], window_name=f"REPAIRED: {method}")

        try:
            graph = run_pipeline(
                cloud_path=None,
                bbox_points=BBOX_POINTS,
                cloud_obj=repaired,
                voxel_size=0.0001,
            )
        except Exception as e:
            logger.error(f"Pipeline failed for method '{method}': {e}")
            continue

        try:
            main_axis, plane_normal = get_plane_main_axes(graph["plane"])
            logger.info(f"Visualizing target nodes ({method})")
            target_poses = prepare_and_visualize_tcp_path(
                graph["node_coords_3d"],
                graph["branches_3d"],
                graph["o3d_branches"],
                graph["o3d_nodes"],
                graph["plane"],
                main_axis,
                plane_normal,
            )
            logger.info("TCP poses:")
            for pose in target_poses:
                logger.info(np.round(pose, 2))
        except Exception as e:
            logger.error(f"Visualization failed for method '{method}': {e}")
            continue

    logger.success("=== FILE MODE COMPLETE ===")


if __name__ == "__main__":
    # farm_cycle()
    run_from_file(".data_clouds/farm_20250730_143651.ply")
