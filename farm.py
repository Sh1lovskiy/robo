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
    """Estimate main axis and normal from O3D plane point cloud."""
    pts = np.asarray(plane.points)
    center = np.mean(pts, axis=0)
    _, _, Vt = np.linalg.svd(pts - center, full_matrices=False)
    return Vt[0], Vt[2]  # main_axis, normal


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
    offset_xyz: tuple[float, float, float] = (0.03, 0.017, -0.25),
    frame_size: float = 0.05,
):
    """
    Build TCP poses ([x, y, z, rx, ry, rz]) and O3D frames for each target point.
    X: main_axis, Z: plane_normal, Y: right-hand rule.
    """
    x_axis = -main_axis / np.linalg.norm(main_axis)
    z_axis = plane_normal / np.linalg.norm(plane_normal)
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


def build_graph(nodes, branches):
    """Build networkx graph from node coords and branch index pairs."""
    # undirected graph
    G = nx.Graph()
    for i, coord in enumerate(nodes):
        # add each node with its 3D position as attribute
        G.add_node(i, pos=coord)

    for branch in branches:
        # use only the endpoints of the branch as graph edges
        # i: first node in the branch, j: last node
        i, j = branch[0], branch[-1]
        # euclidean distance between endpoints
        dist = np.linalg.norm(nodes[i] - nodes[j])
        # add edge with distance as weight
        G.add_edge(i, j, weight=dist)
    return G


def merge_close_points_2d(points: np.ndarray, radius: float = 0.2) -> np.ndarray:
    """
    Merge 2D points that are closer than radius (in XY).
    Each cluster is replaced by its average.
    """
    tree = cKDTree(points[:, :2])
    visited = np.zeros(len(points), dtype=bool)
    merged = []
    for i in range(len(points)):
        if visited[i]:
            continue
        idxs = tree.query_ball_point(points[i, :2], radius)
        cluster = points[idxs]
        merged_point = np.mean(cluster, axis=0)
        merged.append(merged_point)
        visited[idxs] = True
    return np.array(merged)


def rebuild_edges_from_coords(
    nodes: np.ndarray, branches_xyz: list[np.ndarray], tol=1e-6
):
    """Map 3D branches to new node indices after merging."""
    tree = cKDTree(nodes)
    edges = []
    for branch in branches_xyz:
        start, end = branch[0], branch[-1]
        i = tree.query(start)[1]
        j = tree.query(end)[1]
        if i != j and [i, j] not in edges and [j, i] not in edges:
            edges.append([i, j])
    return edges


def shortest_path_order(G, start_idx, targets):
    """Order all targets by shortest path from start, greedy."""
    # get the connected component that contains the starting node
    component = set(nx.node_connected_component(G, start_idx))
    # not reachable from start
    filtered = [t for t in targets if t in component]
    # path order list, current node, and set of unvisited targets
    order, current, pool = [], start_idx, set(filtered)
    while pool:
        next_idx = min(
            pool, key=lambda t: nx.shortest_path_length(G, current, t, weight="weight")
        )
        order.append(next_idx)
        current = next_idx
        pool.remove(next_idx)
    return order


def visualize_targets_on_graph(graph, main_axis, plane_normal, frame_size=0.04):
    """Visualize nodes, graph, Dijkstra route and TCP frames. Return ordered TCP poses."""
    nodes = merge_close_points_2d(np.array(graph["node_coords_3d"]), radius=0.02)
    edges = rebuild_edges_from_coords(nodes, graph["branches_3d"])
    target_points = np.array(nodes)

    # Compute poses and frames
    poses, frames = compute_tcp_poses(
        target_points,
        main_axis,
        plane_normal,
        offset_xyz=(0.0, 0.0, -0.2),
        frame_size=frame_size,
    )

    # Build graph
    G = build_graph(nodes, edges)
    current_pos = poses[0][:3]
    start_idx = np.argmin(np.linalg.norm(nodes - current_pos, axis=1))
    order = shortest_path_order(G, start_idx, list(range(len(nodes))))
    poses_in_path = [poses[i] for i in order]

    # Geometries
    node_pcd = o3d.geometry.PointCloud()
    node_pcd.points = o3d.utility.Vector3dVector(nodes)
    node_pcd.paint_uniform_color([1, 0.4, 0])

    route_lines = [[order[i], order[i + 1]] for i in range(len(order) - 1)]
    route_ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(nodes),
        lines=o3d.utility.Vector2iVector(route_lines),
    )
    route_ls.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(route_lines))

    graph_ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(nodes),
        lines=o3d.utility.Vector2iVector(edges),
    )
    graph_ls.colors = o3d.utility.Vector3dVector([[0, 0, 0]] * len(edges))

    # Direction arrows
    center = np.mean(nodes, axis=0)
    arrow_len = 0.025

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

    arrow_normal = align_arrow(
        o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.001,
            cone_radius=0.003,
            cylinder_height=arrow_len * 0.8,
            cone_height=arrow_len * 0.2,
        ),
        -plane_normal,
    )
    arrow_normal.paint_uniform_color([0.1, 0.2, 1])
    arrow_normal.translate(center)

    run_visualization(
        plane=graph["plane"],
        o3d_branches=graph["o3d_branches"],
        o3d_nodes=graph["o3d_nodes"],
        graph_lineset=graph_ls,
        arrow_main=arrow_main,
        arrow_normal=arrow_normal,
        frames=frames,
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
    o3d.visualization.draw_geometries(
        [graph["plane"]] + graph["o3d_branches"] + graph["o3d_nodes"]
    )
    main_axis, plane_normal = get_plane_main_axes(graph["plane"])
    logger.info("Visualizing target nodes")
    target_poses = visualize_targets_on_graph(graph, main_axis, plane_normal)
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


def run_from_file(pcd_path):
    logger.info("=== RUNNING FROM FILE ===")
    pcd = load_point_cloud(pcd_path)
    pcd_tcp, _, _ = transform_cloud_to_tcp(pcd, HAND_EYE_R, HAND_EYE_t, TARGET_POSE)
    graph = run_pipeline(
        cloud_path=None,
        bbox_points=BBOX_POINTS,
        cloud_obj=pcd_tcp,
        voxel_size=0.0001,
    )
    # o3d.visualization.draw_geometries(
    #     [graph["plane"]] + graph["o3d_branches"] + graph["o3d_nodes"]
    # )
    main_axis, plane_normal = get_plane_main_axes(graph["plane"])
    logger.info("Visualizing target nodes")
    target_poses = visualize_targets_on_graph(graph, main_axis, plane_normal)
    logger.info("TCP poses:")
    for pose in target_poses:
        logger.info(np.round(pose, 2))
    logger.success("=== FILE MODE COMPLETE ===")


if __name__ == "__main__":
    # farm_cycle()
    run_from_file(".data_clouds/farm_20250729_145942.ply")
