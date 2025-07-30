from __future__ import annotations
import time
import os
import numpy as np
import open3d as o3d
import cv2
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R
import threading
import signal

from esp32.control import ESP32Controller
from robot.rpc import RPC
from utils.logger import Logger, SuppressO3DInfo
from utils.settings import camera as cam_cfg
from utils.error_tracker import ErrorTracker

np.set_printoptions(precision=3, suppress=True, linewidth=120)
logger = Logger.get_logger("scan_full_pipeline")


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
        [-0.57, -0.24, 0.27],
        [-0.38, -0.34, 0.27],
        [-0.38, -0.34, 0.46],
    ]
)
CLOUD_DIR = "box1"
os.makedirs(CLOUD_DIR, exist_ok=True)

running = True
rpc_global: RPC | None = None


def shutdown_handler(sig, frame):
    """Handles Ctrl+C and stops robot safely."""
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


def ensure_dir(path: str):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def robot_connect(ip: str, safety_mode: bool = False) -> RPC:
    """Connects to robot, checks safety, returns RPC instance."""
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
    """Moves robot to joint pose using MoveJ."""
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


def move_motor_async(esp32, steps, direction, delay_us):
    """Runs stepper motor in separate thread for non-blocking rotation."""
    thread = threading.Thread(
        target=esp32.move_motor, args=(steps, direction, delay_us)
    )
    thread.start()
    return thread


def get_realsense_cloud(
    align_to_color: bool = True,
    min_depth: float = 0.2,
    max_depth: float = 2.0,
):
    """
    Captures point cloud, RGB, and depth images from RealSense.
    Args:
        align_to_color: align depth to color (for proper RGBD), otherwise use depth optical frame.
        min_depth, max_depth: depth filtering in meters.
    Returns:
        cloud: Open3D point cloud (filtered)
        color_img: (H,W,3) np.uint8
        depth_img: (H,W) np.float32, meters
    """
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
    align = rs.align(rs.stream.color) if align_to_color else None
    for _ in range(10):
        frames = pipeline.wait_for_frames()
        if align:
            frames = align.process(frames)
    frames = pipeline.wait_for_frames()
    if align:
        frames = align.process(frames)
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if not color_frame or not depth_frame:
        logger.error("Frames not found from RealSense")
        pipeline.stop()
        raise RuntimeError("Frames not found")
    color_img = np.asanyarray(color_frame.get_data())
    depth_img = np.asanyarray(depth_frame.get_data())
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    logger.info(f"Depth scale: {depth_scale:.5f} meters/unit")
    depth_meter = depth_img.astype(np.float32) * depth_scale
    depth_intr = depth_frame.profile.as_video_stream_profile().get_intrinsics()
    pinhole = o3d.camera.PinholeCameraIntrinsic(
        depth_intr.width,
        depth_intr.height,
        depth_intr.fx,
        depth_intr.fy,
        depth_intr.ppx,
        depth_intr.ppy,
    )
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)),
        o3d.geometry.Image(depth_meter),
        depth_scale=1.0,
        depth_trunc=max_depth,
        convert_rgb_to_intensity=False,
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole)
    mask = (np.asarray(pcd.points)[:, 2] > min_depth) & (
        np.asarray(pcd.points)[:, 2] < max_depth
    )
    pcd = pcd.select_by_index(np.where(mask)[0])
    pipeline.stop()
    logger.info(f"Captured aligned cloud with {len(pcd.points)} points")
    return pcd, color_img, depth_meter


def pose_to_transform(pose: list | np.ndarray) -> np.ndarray:
    """Converts [x,y,z,rx,ry,rz] pose to 4x4 transform (mm->m)."""
    pose = np.asarray(pose)
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", pose[3:], degrees=True).as_matrix()
    T[:3, 3] = pose[:3] / 1000.0
    logger.debug(f"Computed transform from pose: {pose}")
    return T


def transform_cloud_to_tcp(
    pcd: o3d.geometry.PointCloud,
    handeye_R: np.ndarray,
    handeye_t: np.ndarray,
    tcp_pose: np.ndarray,
):
    """Transforms point cloud from camera to TCP and base frame."""
    T_cam2base = np.eye(4)
    T_cam2base[:3, :3] = handeye_R
    T_cam2base[:3, 3] = handeye_t.flatten()
    T_base2tcp = np.eye(4)
    T_base2tcp[:3, 3] = tcp_pose.flatten()
    T_cam2tcp = T_base2tcp @ T_cam2base
    pts = np.asarray(pcd.points)
    pts_h = np.empty((4, pts.shape[0]), dtype=np.float64)
    pts_h[:3, :] = pts.T
    pts_h[3, :] = 1.0
    pts_tcp = (T_cam2tcp @ pts_h)[:3, :].T
    pcd_tcp = o3d.geometry.PointCloud()
    pcd_tcp.points = o3d.utility.Vector3dVector(pts_tcp)
    if pcd.has_colors():
        pcd_tcp.colors = pcd.colors
    logger.info(f"Transformed cloud to TCP frame")
    return pcd_tcp


def save_cloud_and_images(
    cloud: o3d.geometry.PointCloud,
    color: np.ndarray,
    depth: np.ndarray,
    out_dir: str,
    prefix: str,
):
    """Saves point cloud (.ply), RGB (.png), depth (.png, .npy) to out_dir with prefix."""
    ensure_dir(out_dir)
    pcd_path = os.path.join(out_dir, f"{prefix}_cloud.ply")
    o3d.io.write_point_cloud(pcd_path, cloud)
    rgb_path = os.path.join(out_dir, f"{prefix}_rgb.png")
    cv2.imwrite(rgb_path, color)
    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_norm.astype(np.uint8)
    depth_png_path = os.path.join(out_dir, f"{prefix}_depth.png")
    cv2.imwrite(depth_png_path, depth_uint8)
    np.save(os.path.join(out_dir, f"{prefix}_depth.npy"), depth)
    logger.info(f"Saved: {pcd_path}, {rgb_path}, {depth_png_path}, and npy")


def main():
    steps_per_iter = [50, 25, 17]
    out_dir = CLOUD_DIR
    global rpc_global
    rpc = robot_connect("192.168.58.2", safety_mode=False)
    rpc_global = rpc
    esp32 = ESP32Controller()
    esp32.laser_off()
    robot_movej(rpc, TARGET_POSE, vel=40)
    for i in range(4):
        logger.info(f"--- Iter {i+1}/4 ---")
        # Capture aligned cloud
        cloud_al, rgb_al, depth_al = get_realsense_cloud(align_to_color=True)
        cloud_tcp = transform_cloud_to_tcp(
            cloud_al, HAND_EYE_R, HAND_EYE_t, TARGET_POSE
        )
        save_cloud_and_images(cloud_tcp, rgb_al, depth_al, out_dir, f"iter{i+1}_align")
        # Capture NOT aligned (depth optical frame)
        cloud_na, rgb_na, depth_na = get_realsense_cloud(align_to_color=False)
        cloud_tcp_na = transform_cloud_to_tcp(
            cloud_na, HAND_EYE_R, HAND_EYE_t, TARGET_POSE
        )
        save_cloud_and_images(
            cloud_tcp_na, rgb_na, depth_na, out_dir, f"iter{i+1}_noalign"
        )
        if i < 3:
            move_motor_async(esp32, steps=steps_per_iter[i], direction=1, delay_us=5000)
            robot_movej(rpc, TARGET_POSE, vel=45)
            time.sleep(1)
    robot_movej(rpc, TARGET_POSE, vel=40)
    logger.success("Scan pipeline finished!")


if __name__ == "__main__":
    try:
        with SuppressO3DInfo():
            main()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
