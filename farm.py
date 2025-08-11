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
import cv2
import pyrealsense2 as rs
import numpy as np
import glob
import open3d as o3d
import networkx as nx
import sys
import threading
from utils.settings import camera as cam_cfg
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from esp32.control import ESP32Controller
from utils.logger import Logger
from skelet import run_pipeline, run_visualization
from rectangle import (
    robot_connect,
    robot_movej,
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
        [0.999048, 0.00428, -0.00625],
        [-0.00706, 0.99658, -0.00804],
        [0.00423, 0.00895, 0.99629],
    ]
)
HAND_EYE_t = np.array([-0.036, -0.078, 0.006]).reshape(3, 1)
TARGET_POSE = np.array([3.63, -103.5, 540.2, -120.2, -1.13, 103.5])
BBOX_POINTS = np.array(
    [
        [-0.57, -0.4, 0.46],
        [-0.57, 0.2, 0.2],
        [-0.3, 0.2, 0.2],
        [-0.3, 0.2, 0.46],
    ]
)


def load_point_cloud(
    pcd_path: str, voxel_size: float = 0.002
) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    return pcd_down


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
    offset_xyz: tuple[float, float, float] = (0.043, -0.005, -0.27),
    frame_size: float = 0.05,
):
    """
    Build TCP poses ([x, y, z, rx, ry, rz]) and O3D frames for each target point.
    X: main_axis, Z: plane_normal, Y: right-hand rule.
    """
    x_axis = -main_axis / np.linalg.norm(main_axis)
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
    raw_points: np.ndarray,
    plane_points: np.ndarray,
    radius: float = 0.012,
    refine_radius: float = 0.0,
) -> tuple[np.ndarray, dict[int, int]]:
    tree = cKDTree(raw_points[:, :2])
    plane_tree = cKDTree(plane_points[:, :2])
    visited = np.zeros(len(raw_points), dtype=bool)
    clusters, index_map = [], {}

    for idx, pt in enumerate(raw_points):
        if visited[idx]:
            continue
        neighbor_idxs = tree.query_ball_point(pt[:2], radius)
        visited[neighbor_idxs] = True
        cluster_pts = raw_points[neighbor_idxs]
        cluster_center = np.mean(cluster_pts, axis=0)
        clusters.append(cluster_center)
        cluster_id = len(clusters) - 1
        for ni in neighbor_idxs:
            index_map[ni] = cluster_id

    refined_nodes = []
    for node in clusters:
        nearby_plane_idxs = plane_tree.query_ball_point(node[:2], refine_radius)
        if nearby_plane_idxs:
            refined_center = np.mean(plane_points[nearby_plane_idxs], axis=0)
            node[:2] = refined_center[:2]
            node[2] = refined_center[2]
        refined_nodes.append(node)

    return np.array(refined_nodes), index_map


def rebuild_edges(
    branches_xyz: list[np.ndarray],
    raw_points: np.ndarray,
    index_map: dict[int, int],
) -> list[list[int]]:
    """
    Convert original 3D branches into a set of unique undirected edges between merged nodes.
    Each branch is reduced to its start and end point and mapped through the clustering.
    """
    tree = cKDTree(raw_points)
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
    nodes = np.array([p[:3] for p in poses])
    current_pos = poses[0][:3]
    start_idx = np.argmin(np.linalg.norm(nodes - current_pos, axis=1))
    order = list(nx.dfs_preorder_nodes(G, source=start_idx))
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


def edge_dfs_path(G, start):
    visited_edges = set()
    path = []

    def dfs(u):
        for v in G.neighbors(u):
            eid = tuple(sorted((u, v)))
            if eid not in visited_edges:
                visited_edges.add(eid)
                dfs(v)
        path.append(u)

    dfs(start)
    return path[::-1]


def ant_colony_like_path(G, start=0, n_ants=10, n_iters=20, timeout=2.0):
    return list(G.nodes)
    edges = set(G.edges())
    best_path = None
    best_len = float("inf")
    t0 = time.time()
    for _ in range(n_iters):
        if time.time() - t0 > timeout:
            return list(G.nodes)
        for _ in range(n_ants):
            path = [start]
            visited_edges = set()
            current = start
            while len(visited_edges) < len(edges):
                neighbors = [
                    n
                    for n in G.neighbors(current)
                    if tuple(sorted((current, n))) not in visited_edges
                ]
                if not neighbors:
                    neighbors = list(G.neighbors(current))
                nxt = random.choice(neighbors)
                eid = tuple(sorted((current, nxt)))
                visited_edges.add(eid)
                path.append(nxt)
                current = nxt
            path_len = sum(G[u][v]["weight"] for u, v in zip(path[:-1], path[1:]))
            if path_len < best_len:
                best_len = path_len
                best_path = path
    return best_path


def prepare_and_visualize_tcp_path(
    graph,
    main_axis: np.ndarray,
    plane_normal: np.ndarray,
) -> list[np.ndarray]:
    """
    Complete pipeline:
    - Cluster node positions in 2D
    - Map branches to edges
    - Build graph
    - Compute TCP poses and sort them greedily by distance
    - Visualize result
    """
    try:
        logger.info("Extracting geometry from graph...")
        plane = graph["plane"]
        node_coords_3d = graph["node_coords_3d"]
        branches_3d = graph["branches_3d"]
        o3d_branches = graph["o3d_branches"]
        o3d_nodes = graph["o3d_nodes"]
    except Exception as e:
        logger.error(f"Error extracting geometry: {e}")
        raise

    try:
        logger.info("Preparing raw node points for clustering...")
        raw = np.asarray(node_coords_3d)
        plane_pts = np.asarray(plane.points)
        raw_points = np.concatenate([branch[[0, -1]] for branch in branches_3d], axis=0)
    except Exception as e:
        logger.error(f"Error preparing raw points: {e}")
        raise

    try:
        logger.info("Clustering node positions in 2D...")
        merged, index_map = cluster_nodes_2d(raw_points, plane_pts)
        logger.success(f"Clustered nodes: {len(merged)}")
    except Exception as e:
        logger.error(f"Error clustering nodes: {e}")
        raise

    try:
        logger.info("Rebuilding graph edges...")
        edges = []
        for branch in branches_3d:
            tree = cKDTree(raw_points)
            i_raw = tree.query(branch[0])[1]
            j_raw = tree.query(branch[-1])[1]
            i = index_map[i_raw]
            j = index_map[j_raw]
            if i != j:
                edges.append([i, j])
        logger.success(f"Edges rebuilt: {len(edges)}")
    except Exception as e:
        logger.error(f"Error rebuilding edges: {e}")
        raise

    try:
        logger.info("Building graph structure...")
        G = build_graph(merged, edges)
        logger.success(
            f"Graph has {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
        )
    except Exception as e:
        logger.error(f"Error building graph: {e}")
        raise

    try:
        logger.info("Computing TCP poses...")
        poses, frames = compute_tcp_poses(merged, main_axis, plane_normal)
        logger.success("TCP poses computed")
    except Exception as e:
        logger.error(f"Error computing TCP poses: {e}")
        raise

    try:
        logger.info("Building traversal order for TCP path...")
        order = ant_colony_like_path(G, start=0, timeout=2.0)
        if isinstance(order, list) and all(isinstance(x, int) for x in order):
            poses_in_path = [poses[i] for i in order if 0 <= i < len(poses)]
        else:
            poses_in_path = poses
        logger.success(f"Order computed: {order}")
    except Exception as e:
        logger.error(f"Error computing traversal order: {e}")
        raise

    try:
        logger.info("Filtering poses along traversal path...")
        poses_in_path = [poses[i] for i in order if 0 <= i < len(poses)]
        logger.success(f"Prepared {len(poses_in_path)} poses in path")
    except Exception as e:
        logger.error(f"Error filtering poses: {e}")
        raise

    try:
        logger.info("Rendering path visualization in Open3D...")
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
        logger.success("Visualization rendered")
    except Exception as e:
        logger.error(f"Error rendering visualization: {e}")
        raise

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


def get_camera_cloud() -> o3d.geometry.PointCloud:
    """Capture and return filtered point cloud from RealSense."""
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(
        rs.stream.color,
        cam_cfg.rgb_width,
        cam_cfg.rgb_height,
        rs.format.bgr8,
        cam_cfg.fps,
    )
    cfg.enable_stream(
        rs.stream.depth,
        cam_cfg.depth_width,
        cam_cfg.depth_height,
        rs.format.z16,
        cam_cfg.fps,
    )
    profile = pipeline.start(cfg)
    align = rs.align(rs.stream.color)
    for _ in range(10):
        frames = align.process(pipeline.wait_for_frames())
    color = np.asanyarray(frames.get_color_frame().get_data())
    depth = np.asanyarray(frames.get_depth_frame().get_data())
    scale = profile.get_device().first_depth_sensor().get_depth_scale()
    depth = depth.astype(np.float32) * scale
    intr = frames.get_depth_frame().profile.as_video_stream_profile().get_intrinsics()
    pinhole = o3d.camera.PinholeCameraIntrinsic(
        intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy
    )
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(cv2.cvtColor(color, cv2.COLOR_BGR2RGB)),
        o3d.geometry.Image(depth),
        depth_scale=1,
        convert_rgb_to_intensity=False,
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole)
    pts = np.asarray(pcd.points)
    mask = (pts[:, 2] > 0.2) & (pts[:, 2] < 2.0)
    pcd = pcd.select_by_index(np.where(mask)[0])
    pipeline.stop()
    return pcd


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

    # o3d.visualization.draw_geometries(graph["o3d_branches"] + graph["o3d_nodes"])
    main_axis, plane_normal = get_plane_main_axes(graph["plane"])
    if main_axis[0] < 0:
        main_axis = -main_axis
    CAMERA_DIR = np.array([0, 0, 1])
    if np.dot(plane_normal, CAMERA_DIR) < 0:
        plane_normal = -plane_normal
    logger.info("Visualizing target nodes")
    target_poses = prepare_and_visualize_tcp_path(graph, main_axis, plane_normal)
    logger.info(target_poses)
    input("Press Enter to start movement...")
    move_robot_to_targets(rpc, esp32, target_poses, LASER_ON_TIME)
    time.sleep(MOVE_DELAY)


def from_file():
    """Main robot farm logic."""
    logger.info("=== ROBOT FARM CYCLE START ===")
    pcd = load_point_cloud(".data_clouds/farm_20250729_143818.ply")
    pcd_tcp, _, _ = transform_cloud_to_tcp(pcd, HAND_EYE_R, HAND_EYE_t, TARGET_POSE)
    result, result_serializable = run_pipeline(
        cloud_path=None,
        bbox_points=BBOX_POINTS,
        cloud_obj=pcd_tcp,
        voxel_size=0.0001,
    )
    main_axis, plane_normal = get_plane_main_axes(result["plane"])
    if main_axis[0] < 0:
        main_axis = -main_axis
    CAMERA_DIR = np.array([0, 0, 1])
    if np.dot(plane_normal, CAMERA_DIR) < 0:
        plane_normal = -plane_normal
    logger.info("Visualizing target nodes")
    target_poses = prepare_and_visualize_tcp_path(result, main_axis, plane_normal)
    logger.info(target_poses)

    save_graph = input("Save graph.npy? [y/n]: ").strip().lower()
    if save_graph == "y":
        np.save("graph.npy", result_serializable, allow_pickle=True)
        print("Graph saved to graph.npy")
    else:
        print("Graph NOT saved.")


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


def load_cloud_from_npy_png(
    folder: str,
    depth_file: str = "frame.npy",
    color_file: str = "frame.png",
    intrinsics: o3d.camera.PinholeCameraIntrinsic = None,
    depth_scale: float = 1.0,
) -> o3d.geometry.PointCloud:
    """
    Load point cloud from depth npy and color png in a folder.
    """

    depth = np.load(os.path.join(folder, depth_file)).astype(np.float32)
    color = cv2.imread(os.path.join(folder, color_file), cv2.IMREAD_COLOR)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

    depth_o3d = o3d.geometry.Image(depth)
    color_o3d = o3d.geometry.Image(color)

    if intrinsics is None:
        width, height = 1280, 720
        fx, fy = 894.19386173964381, 900.3243830458739
        ppx, ppy = 635.18707006349337, 358.16277611314541
        intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, ppx, ppy)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1.0,
        convert_rgb_to_intensity=False,
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
    return pcd


def farm_from_captures_dir(
    captures_dir: str,
    hand_eye_R: np.ndarray,
    hand_eye_t: np.ndarray,
    target_pose: np.ndarray,
    bbox_points: np.ndarray,
    voxel_size: float = 0.001,
):
    views = sorted(glob.glob(os.path.join(captures_dir, "view_*")))
    logger.info(f"Found {len(views)} views in {captures_dir}")

    for i, view_folder in enumerate(views):
        logger.info(f"Processing {view_folder}...")
        pcd = load_cloud_from_npy_png(view_folder)

        ply_path = os.path.join(view_folder, "frame.ply")
        o3d.io.write_point_cloud(ply_path, pcd)
        logger.info(f"Saved {ply_path}")

        pcd_tcp, _, _ = transform_cloud_to_tcp(pcd, hand_eye_R, hand_eye_t, target_pose)
        result, result_serializable = run_pipeline(
            cloud_path=None,
            bbox_points=bbox_points,
            cloud_obj=pcd_tcp,
            voxel_size=voxel_size,
        )

        main_axis, plane_normal = get_plane_main_axes(result["plane"])
        if main_axis[0] < 0:
            main_axis = -main_axis
        CAMERA_DIR = np.array([0, 0, 1])
        if np.dot(plane_normal, CAMERA_DIR) < 0:
            plane_normal = -plane_normal

        target_poses = prepare_and_visualize_tcp_path(result, main_axis, plane_normal)
        # logger.info(target_poses)


if __name__ == "__main__":
    # farm_cycle()
    # from_file()
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        # python farm.py captures/20250806_142752
        farm_from_captures_dir(
            sys.argv[1],
            HAND_EYE_R,
            HAND_EYE_t,
            TARGET_POSE,
            BBOX_POINTS,
        )
    else:
        farm_cycle()
