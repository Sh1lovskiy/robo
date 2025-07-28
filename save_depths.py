from __future__ import annotations
import time
import numpy as np
import open3d as o3d
import cv2
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R
from pynput import keyboard
import threading

import signal
from esp32.control import ESP32Controller
from robot.rpc import RPC
from utils.logger import Logger
from utils.settings import camera as cam_cfg
from utils.error_tracker import ErrorTracker
import os


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


logger = Logger.get_logger("task3")

HAND_EYE_R = np.array(
    [
        [0.9990477302446203, 0.024283853200164546, -0.036248133287455686],
        [-0.02705558992019083, 0.9965827079525681, -0.07804422633351543],
        [0.03422904829443468, 0.07895062180679385, 0.9962907063549165],
    ]
)
HAND_EYE_t = np.array(
    [-0.034243512723116474, -0.07904859731388036, 0.00127890214613602]
).reshape(3, 1)
TARGET_POSE = np.array([3.63, -103.5, 540.2, -120.2, -1.13, 103.5])
BBOX_POINTS = np.array(
    [
        [-0.57, -0.34, 0.46],
        [-0.57, -0.24, 0.27],
        [-0.38, -0.34, 0.27],
        [-0.38, -0.34, 0.46],
    ]
)

np.set_printoptions(precision=3, suppress=True, linewidth=120)

running = True
rpc_global: RPC | None = None


def shutdown_handler(sig, frame):
    global running
    logger.warning("Interrupted by user (Ctrl+C)")
    running = False
    if rpc_global:
        try:
            logger.info("Stopping robot safely...")
            rpc_global.StopMotion()
        except Exception as e:
            logger.error(f"Shutdown error: {e}")
    exit(0)


signal.signal(signal.SIGINT, shutdown_handler)
ErrorTracker.install_signal_handlers()


def robot_connect(ip: str, safety_mode: bool = False) -> RPC:
    logger.info(f"Connecting to robot at {ip}")
    rpc = RPC(ip=ip)
    if rpc.RobotEnable(1) != 0:
        logger.error("Failed to enable robot")
        raise RuntimeError("Failed to enable robot")
    if safety_mode and rpc.GetSafetyCode() != 0:
        logger.error("Robot not in a safe state")
        raise RuntimeError("Robot not in a safe state")
    return rpc


def robot_movej(
    rpc: RPC, pose: list | np.ndarray, vel=30.0, acc=30.0, reset_last=False
):
    pose = pose.tolist() if isinstance(pose, np.ndarray) else pose
    result = rpc.GetInverseKin(0, pose)
    if isinstance(result, tuple) and len(result) == 2:
        code, joints = result
    else:
        logger.error(f"GetInverseKin failed: {result}")
        raise RuntimeError(f"GetInverseKin failed: {result}")
    if reset_last:
        joints[5] = 0.0
    if rpc.MoveJ(joint_pos=joints, tool=0, user=0, vel=vel, acc=acc) != 0:
        logger.error("MoveJ failed")
        raise RuntimeError("MoveJ failed")
    logger.info(f"MoveJ to {np.array(pose)} completed")
    time.sleep(0.5)


def robot_movel(
    rpc: RPC, pose: list | np.ndarray, vel=30.0, acc=30.0, reset_last=False
):
    pose = pose.tolist() if isinstance(pose, np.ndarray) else pose
    result = rpc.GetInverseKin(0, pose)
    if isinstance(result, tuple) and len(result) == 2:
        code, joints = result
    else:
        logger.error(f"GetInverseKin failed: {result}")
        raise RuntimeError(f"GetInverseKin failed: {result}")
    if reset_last:
        joints[5] = 0.0
    if (
        rpc.MoveL(desc_pos=pose, tool=0, user=0, joint_pos=joints, vel=vel, acc=acc)
        != 0
    ):
        logger.error("MoveL failed")
        raise RuntimeError("MoveL failed")
    logger.info(f"MoveL to {np.array(pose)} completed")
    time.sleep(0.5)


def get_camera_cloud() -> o3d.geometry.PointCloud:
    logger.info("Capturing point cloud from camera")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.color,
        cam_cfg.rgb_width,
        cam_cfg.rgb_height,
        rs.format.bgr8,
        cam_cfg.fps,
    )
    config.enable_stream(
        rs.stream.depth,
        cam_cfg.depth_width,
        cam_cfg.depth_height,
        rs.format.z16,
        cam_cfg.fps,
    )
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    for _ in range(5):  # сократил до 5 для ускорения
        align.process(pipeline.wait_for_frames())
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
        depth_scale=1.0,
        convert_rgb_to_intensity=False,
    )
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole)
    mask = np.logical_and(
        0.2 < np.asarray(cloud.points)[:, 2], np.asarray(cloud.points)[:, 2] < 2.0
    )
    cloud = cloud.select_by_index(np.where(mask)[0])
    pipeline.stop()
    logger.info(f"Captured cloud with {len(cloud.points)} points")
    return cloud, color, depth


def pose_to_transform(pose: list | np.ndarray) -> np.ndarray:
    pose = np.asarray(pose)
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", pose[3:], degrees=True).as_matrix()
    T[:3, 3] = pose[:3] / 1000.0
    logger.debug(f"Computed transform from pose: {pose}")
    return T


def transform_cloud_to_tcp(pcd, handeye_R, handeye_t, tcp_pose):
    T_cam2base = np.eye(4)
    T_cam2base[:3, :3] = handeye_R
    T_cam2base[:3, 3] = handeye_t.flatten()
    T_base2tcp = pose_to_transform(tcp_pose)
    T_cam2tcp = T_base2tcp @ T_cam2base
    pts = np.asarray(pcd.points)
    pts_h = np.empty((4, pts.shape[0]), dtype=np.float64)
    pts_h[:3, :] = pts.T
    pts_h[3, :] = 1.0
    pts_tcp = (T_cam2tcp @ pts_h)[:3, :].T
    pts_base = (T_cam2base @ pts_h)[:3, :].T
    pcd_tcp = o3d.geometry.PointCloud()
    pcd_tcp.points = o3d.utility.Vector3dVector(pts_tcp)
    if pcd.has_colors():
        pcd_tcp.colors = pcd.colors
    pcd_base = o3d.geometry.PointCloud()
    pcd_base.points = o3d.utility.Vector3dVector(pts_base)
    if pcd.has_colors():
        pcd_base.colors = pcd.colors
    logger.info(f"Transformed cloud to TCP/base")
    return pcd_tcp, pcd_base, T_cam2tcp


def get_bbox_crop(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    box = o3d.geometry.PointCloud()
    box.points = o3d.utility.Vector3dVector(BBOX_POINTS)
    cropped = pcd.crop(box.get_axis_aligned_bounding_box())
    logger.info(f"Cropped cloud to bbox: {len(cropped.points)} pts")
    return cropped


def get_main_plane_and_pca_line(pcd: o3d.geometry.PointCloud):
    if len(pcd.points) < 4:
        logger.error("Too few points for PCA")
        raise ValueError("Too few points for PCA")
    pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
    plane_model, inliers = pcd_clean.segment_plane(
        distance_threshold=0.004, ransac_n=3, num_iterations=500
    )
    plane = pcd_clean.select_by_index(inliers)
    pts = np.asarray(plane.points)
    center = np.mean(pts, axis=0)
    _, _, Vt = np.linalg.svd(pts - center, full_matrices=False)
    direction, normal = Vt[0], Vt[2]
    projections = (pts - center) @ direction
    pt1 = center + direction * projections.min()
    pt2 = center + direction * projections.max()
    logger.info(f"Extracted main plane and line: p1={pt1}, p2={pt2}")
    return plane, (pt1, pt2), center, direction, normal


def compute_tcp_frames(
    pt1: np.ndarray,
    pt2: np.ndarray,
    normal: np.ndarray,
    robot_base: np.ndarray = np.array([0.0, 0.0, 0.0]),
    offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    plane_center: np.ndarray = None,
) -> tuple[list[np.ndarray], list[o3d.geometry.TriangleMesh]]:
    x_axis = (pt1 - pt2) / np.linalg.norm(pt2 - pt1)
    z_axis = -normal / np.linalg.norm(normal)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    R_tcp = np.column_stack((x_axis, y_axis, z_axis))
    det = np.linalg.det(R_tcp)
    logger.info(f"det(R_tcp) = {det:.3f} (should be +1)")

    frames = []
    poses = []
    offset_xyz_vec = R_tcp @ np.array(offset_xyz, dtype=np.float64)
    for pt in (pt1, pt2):
        T = np.eye(4)
        T[:3, :3] = R_tcp
        T[:3, 3] = pt + offset_xyz_vec
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        mesh.transform(T)
        frames.append(mesh)
        pose = np.concatenate(
            [(T[:3, 3] * 1000), R.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)]
        )
        poses.append(pose)
    lines = "\n".join(
        np.array2string(p, precision=3, suppress_small=True) for p in poses
    )
    logger.info(f"TCP frames computed:\n{lines}")
    return poses, frames


def wait_for_n_key(logger=None):
    logger = logger or print
    logger("Press 'n' to continue...")
    result = []

    def on_press(key):
        try:
            if key.char == "n":
                result.append(True)
                return False
        except Exception:
            pass

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


def visualize_with_quit(geoms: list[o3d.geometry.Geometry]):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Scene", width=1280, height=720)
    for g in geoms:
        vis.add_geometry(g)

    def quit_callback(vis_obj):
        logger.info("Pressed 'q' in Open3D window")
        vis_obj.close()

    vis.register_key_callback(ord("Q"), quit_callback)
    vis.run()
    vis.destroy_window()
    logger.info("Closed Open3D visualization window")


def move_motor_async(esp32, steps, direction, delay_us):
    thread = threading.Thread(
        target=esp32.move_motor, args=(steps, direction, delay_us)
    )
    thread.start()
    return thread


def main():
    # 90, 45, 30
    steps_per_iter = [50, 25, 17]
    # steps_per_iter = [50, 50, 50]
    farm_dir = "box1"
    ensure_dir(farm_dir)
    global rpc_global
    rpc = robot_connect("192.168.58.2", safety_mode=False)
    rpc_global = rpc
    esp32 = ESP32Controller()
    esp32.laser_off()
    robot_movej(rpc, TARGET_POSE, vel=40)
    for i in range(4):
        logger.info(f"--- Iter {i+1}/4 ---")
        # try:
        # ======== Съем данных ========
        pcd, color, depth = get_camera_cloud()
        pcd_tcp, _, _ = transform_cloud_to_tcp(pcd, HAND_EYE_R, HAND_EYE_t, TARGET_POSE)
        # --- Сохраняем облако в farm ---
        o3d.io.write_point_cloud(
            os.path.join(farm_dir, f"cloud3d_iter{i+1}.ply"), pcd_tcp
        )
        logger.info(f"Saved 3D point cloud: cloud3d_iter{i+1}.ply")
        # Сохраняем RGB
        rgb_path = os.path.join(farm_dir, f"rgb_iter{i+1}.png")
        cv2.imwrite(rgb_path, color)
        logger.info(f"Saved RGB image: rgb_iter{i+1}.png")
        # Сохраняем DEPTH
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_norm.astype(np.uint8)
        depth_path = os.path.join(farm_dir, f"depth_iter{i+1}.png")
        cv2.imwrite(depth_path, depth_uint8)
        logger.info(f"Saved depth image: depth_iter{i+1}.png")
        # --- Дополнительно, если нужен align/depth в npy или tiff ---
        np.save(os.path.join(farm_dir, f"depth_iter{i+1}.npy"), depth)
        # ========== Дальнейший код ==========

        #     pcd_tcp, _, _ = transform_cloud_to_tcp(
        #         pcd, HAND_EYE_R, HAND_EYE_t, TARGET_POSE
        #     )
        #     o3d.io.write_point_cloud("task_robot.ply", pcd_tcp)
        #     cropped = get_bbox_crop(pcd_tcp)
        #     plane, (pt1, pt2), _, _, normal = get_main_plane_and_pca_line(cropped)
        #     plane.paint_uniform_color([0.6, 0.6, 0.6])

        #     line = o3d.geometry.LineSet(
        #         points=o3d.utility.Vector3dVector([pt1, pt2]),
        #         lines=o3d.utility.Vector2iVector([[0, 1]]),
        #     )
        #     line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
        #     base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        #     poses, frames = compute_tcp_frames(
        #         pt1, pt2, normal, offset_xyz=(0.03, 0.017, -0.25)
        #     )
        #     # visualize_with_quit([plane, line, base] + frames)
        #     try:
        #         robot_movej(rpc, poses[0], vel=45)
        #     except RuntimeError as err:
        #         if "112" in str(err):
        #             logger.error(f"MoveJ error 112 on edge {i+1}, continue")
        #             motor_thread = move_motor_async(
        #                 esp32, steps=steps_per_iter[i], direction=1, delay_us=5000
        #             )
        #             continue
        #         else:
        #             raise
        #     esp32.laser_on()
        #     time.sleep(1)
        #     try:
        #         robot_movel(rpc, poses[1], vel=45)
        #     except RuntimeError as err:
        #         if "112" in str(err):
        #             logger.error(f"MoveL error 112 on {i+1} edge, continue")
        #             esp32.laser_off()
        #             continue
        #         else:
        #             esp32.laser_off()
        #             raise
        #     time.sleep(0.5)
        #     esp32.laser_off()
        # except Exception as e:
        #     logger.error(f"Iter {i+1} stop: {e}")
        #     continue
        if i < 3:
            motor_thread = move_motor_async(
                esp32, steps=steps_per_iter[i], direction=1, delay_us=5000
            )
            robot_movej(rpc, TARGET_POSE, vel=45)
            time.sleep(1)
    robot_movej(rpc, TARGET_POSE, vel=40)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
