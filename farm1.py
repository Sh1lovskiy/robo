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
    pts = np.asarray(plane.points)
    center = np.mean(pts, axis=0)
    _, _, Vt = np.linalg.svd(pts - center, full_matrices=False)
    return Vt[0], Vt[2]


def compute_tcp_poses(points, main_axis, plane_normal, offset=(0.035, 0.015, -0.3)):
    x = -main_axis / np.linalg.norm(main_axis)
    z = -plane_normal / np.linalg.norm(plane_normal)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    R_tcp = np.column_stack((x, y, z))
    offset_vec = R_tcp @ np.array(offset)
    poses = []
    for pt in points:
        T = np.eye(4)
        T[:3, :3] = R_tcp
        T[:3, 3] = pt + offset_vec
        pose = np.concatenate(
            [T[:3, 3] * 1000, R.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)]
        )
        poses.append(pose)
    return poses


def merge_close_points_2d(points, radius=0.03):
    tree = cKDTree(points[:, :2])
    visited = np.zeros(len(points), dtype=bool)
    merged = []
    for i in range(len(points)):
        if visited[i]:
            continue
        idxs = tree.query_ball_point(points[i, :2], radius)
        cluster = points[idxs]
        merged.append(np.mean(cluster, axis=0))
        visited[idxs] = True
    return np.array(merged)


def sample_branch_nodes(branches_3d):
    nodes = []
    for b in branches_3d:
        b = np.asarray(b)
        if b.shape[0] == 0:
            continue
        if len(b) == 1:
            nodes.append(b[0])
        else:
            nodes.extend([b[0], b[-1]])
    return np.array(nodes)


def move_robot_to_targets(rpc, esp32, poses, vel=35, laser_on_time=0.2):
    for i, pose in enumerate(poses):
        logger.info(f"[{i+1}/{len(poses)}] MoveJ to {np.round(pose[:3], 1)} mm")
        try:
            robot_movej(rpc, pose, vel=vel)
            esp32.laser_on()
            time.sleep(laser_on_time)
        except Exception as e:
            logger.error(f"MoveJ failed at {i+1}: {e}")
    esp32.laser_off()


def visualize_and_prepare_poses(branches_3d, main_axis, plane_normal):
    nodes = sample_branch_nodes(branches_3d)
    if nodes.size == 0:
        logger.error("No node points extracted")
        return []
    merged_nodes = merge_close_points_2d(nodes, radius=0.02)
    poses = compute_tcp_poses(merged_nodes, main_axis, plane_normal)
    frames = [
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.04).translate(
            p[:3] / 1000
        )
        for p in poses
    ]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_nodes)
    pcd.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([pcd] + frames)
    return poses


def single_farm_iteration(i, rpc, esp32):
    logger.info(f"=== Iteration {i+1} ===")
    pcd = get_camera_cloud()
    save_cloud_timestamped(pcd)
    pcd_tcp, _, _ = transform_cloud_to_tcp(pcd, HAND_EYE_R, HAND_EYE_t, TARGET_POSE)
    graph = run_pipeline(
        cloud_path=None, bbox_points=BBOX_POINTS, cloud_obj=pcd_tcp, voxel_size=0.0001
    )
    main_axis, plane_normal = get_plane_main_axes(graph["plane"])
    if main_axis[0] < 0:
        main_axis = -main_axis
    if np.dot(plane_normal, [0, 0, 1]) < 0:
        plane_normal = -plane_normal
    poses = visualize_and_prepare_poses(graph["branches_3d"], main_axis, plane_normal)
    input("Check visualization. Press Enter to start robot movement...")
    move_robot_to_targets(rpc, esp32, poses, vel=35, laser_on_time=LASER_ON_TIME)
    time.sleep(MOVE_DELAY)


def move_motor_threaded(esp32, steps, direction=1, delay_us=5000):
    time.sleep(0.5)
    t = threading.Thread(target=esp32.move_motor, args=(steps, direction, delay_us))
    t.start()
    t.join()


def farm_cycle():
    logger.info("=== ROBOT FARM START ===")
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
    logger.success("=== ALL ITERATIONS COMPLETE ===")


def run_from_file(pcd_path):
    logger.info("=== RUN FROM FILE ===")
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd.estimate_normals()
    logger.info(f"Loaded point cloud: {len(pcd.points)} points")
    pcd_tcp, _, _ = transform_cloud_to_tcp(pcd, HAND_EYE_R, HAND_EYE_t, TARGET_POSE)
    graph = run_pipeline(
        cloud_path=None, bbox_points=BBOX_POINTS, cloud_obj=pcd_tcp, voxel_size=0.0001
    )
    main_axis, plane_normal = get_plane_main_axes(graph["plane"])
    poses = visualize_and_prepare_poses(graph["branches_3d"], main_axis, plane_normal)
    logger.info("TCP poses:")
    for pose in poses:
        logger.info(np.round(pose, 2))
    logger.success("=== FILE MODE COMPLETE ===")


if __name__ == "__main__":
    # farm_cycle()
    run_from_file(".data_clouds/farm_20250729_145942.ply")
