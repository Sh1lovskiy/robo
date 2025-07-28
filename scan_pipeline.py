import time
import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R

from esp32.control import ESP32Controller
from robot.rpc import RPC
from utils.logger import Logger
from utils.settings import camera as cam_cfg


def robot_connect(ip: str, safety_mode: bool = True) -> RPC:
    logger = Logger.get_logger(__name__)
    logger.info(f"Connecting to robot at {ip}")
    rpc = RPC(ip=ip)
    if rpc.RobotEnable(1) != 0:
        raise RuntimeError("Failed to enable robot")
    if safety_mode and rpc.GetSafetyCode() != 0:
        raise RuntimeError("Robot not in a safe state")
    logger.info("Robot ready")
    return rpc


def robot_movej(rpc: RPC, pose: np.ndarray, vel: float) -> None:
    logger = Logger.get_logger(__name__)
    target = pose.tolist() if isinstance(pose, np.ndarray) else pose
    logger.info(f"MoveJ to {target}")
    code, joints = rpc.GetInverseKin(0, target)
    if code != 0 or not joints:
        raise RuntimeError(f"Inverse kinematics failed: code={code}")
    code = rpc.MoveJ(joint_pos=joints, tool=0, user=0, vel=vel)
    if code != 0:
        raise RuntimeError(f"MoveJ failed: code={code}")
    time.sleep(1)


def robot_movel(rpc: RPC, pose: list, vel: float) -> None:
    logger = Logger.get_logger(__name__)
    logger.info(f"MoveL to {pose}")
    code, joints = rpc.GetInverseKin(0, pose)
    if code != 0 or not joints:
        raise RuntimeError(f"Inverse kinematics failed: code={code}")
    code = rpc.MoveL(desc_pos=pose, tool=0, user=0, joint_pos=joints, vel=vel)
    if code != 0:
        raise RuntimeError(f"MoveL failed: code={code}")
    time.sleep(1)


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


def pose_to_transform(pose: np.ndarray) -> np.ndarray:
    """Convert [x,y,z,rx,ry,rz] pose to 4x4 transform matrix."""
    pos = pose[:3] / 1000.0
    rot = R.from_euler("xyz", pose[3:], degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return T


def transform_cloud(
    pcd: o3d.geometry.PointCloud,
    handeye_R: np.ndarray,
    handeye_t: np.ndarray,
    tcp_pose: np.ndarray,
) -> o3d.geometry.PointCloud:
    cam2tcp = np.eye(4)
    cam2tcp[:3, :3] = handeye_R
    cam2tcp[:3, 3] = handeye_t.flatten()
    tcp2base = pose_to_transform(tcp_pose)
    pts = np.asarray(pcd.points)
    ones = np.ones((pts.shape[0], 1))
    pts_base = (tcp2base @ cam2tcp @ np.hstack((pts, ones)).T)[:3].T
    pcd_base = o3d.geometry.PointCloud()
    pcd_base.points = o3d.utility.Vector3dVector(pts_base)
    return pcd_base


def crop_point_cloud(
    pcd: o3d.geometry.PointCloud, bbox_pts: np.ndarray
) -> o3d.geometry.PointCloud:
    bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(bbox_pts)
    )
    return pcd.crop(bbox)


def fit_plane_and_pca_line(pcd: o3d.geometry.PointCloud):
    """Return face cloud, endpoints (pt1, pt2), normal, and PCA axis."""
    filt, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
    _, inliers = filt.segment_plane(
        distance_threshold=0.004, ransac_n=3, num_iterations=1000
    )
    face = filt.select_by_index(inliers)
    pts = np.asarray(face.points)
    center = pts.mean(0)
    U, S, Vt = np.linalg.svd(pts - center)
    axis = Vt[0]
    proj = (pts - center) @ axis
    pt1 = center + proj.min() * axis
    pt2 = center + proj.max() * axis
    normal = Vt[2]
    return face, pt1, pt2, normal, axis


def compute_tcp_orientation(pt1: np.ndarray, pt2: np.ndarray, normal: np.ndarray):
    x = pt1 - pt2
    x /= np.linalg.norm(x)
    z = normal / np.linalg.norm(normal)
    z = z - (z @ x) * x  # enforce perpendicularity
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    R_tcp = np.column_stack((x, y, z))
    if np.linalg.det(R_tcp) < 0:
        R_tcp[:, 1] *= -1
    angles = R.from_matrix(R_tcp).as_euler("xyz", degrees=True)
    return R_tcp, angles


def visualize_line_and_poses(
    face: o3d.geometry.PointCloud,
    pt1: np.ndarray,
    pt2: np.ndarray,
    normal: np.ndarray,
    axis: np.ndarray,
    R_tcp: np.ndarray,
    distance: float = 0.25,
):
    """
    Visualize the detected face, robot base frame, and two TCP poses at offset distance.
    """
    frames = []
    # Face cloud
    face.paint_uniform_color([0.7, 0.7, 0.7])
    frames.append(face)
    # Robot base frame
    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    frames.append(base_frame)
    # Two TCP frames at pt1 and pt2 with offset along normal
    for pt in (pt1, pt2):
        pos = pt + normal / np.linalg.norm(normal) * distance
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        T = np.eye(4)
        T[:3, :3] = R_tcp
        T[:3, 3] = pos
        mesh.transform(T)
        frames.append(mesh)
    o3d.visualization.draw_geometries(frames)


def move_along_line(
    rpc: RPC,
    pt1: np.ndarray,
    pt2: np.ndarray,
    normal: np.ndarray,
    vel: float,
    offset: float,
    shift: float,
):
    home = rpc.GetActualJointPosDegree()[1]
    R_tcp, (rx, ry, rz) = compute_tcp_orientation(pt1, pt2, normal)
    approach = (
        pt1
        + normal * np.sign(normal[2]) * offset
        + (pt2 - pt1) / np.linalg.norm(pt2 - pt1) * shift
    )
    pose1 = np.hstack((approach * 1000, [rx, ry, rz]))
    robot_movej(rpc, pose1, vel)
    esp = ESP32Controller()
    esp.laser_on()
    time.sleep(1)
    end = pt2 + normal * np.sign(normal[2]) * offset
    pose2 = np.hstack((end * 1000, [rx, ry, rz]))
    robot_movel(rpc, pose2.tolist(), vel)
    esp.laser_off()
    rpc.MoveJ(joint_pos=home, tool=0, user=0, vel=vel)


def main():
    # Constants
    HAND_EYE_R = np.array(
        [
            [0.99891689, 0.02255943, -0.04069543],
            [-0.02603024, 0.99587872, -0.08687929],
            [0.03856776, 0.0878445, 0.9953873],
        ]
    )
    HAND_EYE_T = np.array([-0.03511529, -0.07953357, 0.00013575]).reshape(3, 1)
    TARGET_POSE = np.array([-66.2, -135.7, 504.3, -119.2, 4.5, 110.8])
    BBOX_POINTS = np.array(
        [
            [-0.52, -0.36, 0.49],
            [-0.43, -0.36, 0.25],
            [-0.52, -0.2, 0.49],
            [-0.43, -0.2, 0.25],
        ]
    )
    VEL_J, VEL_L = 30, 40
    OFFSET, SHIFT = 0.3, -0.022

    logger = Logger.get_logger(__name__)
    rpc = robot_connect("192.168.58.2")
    robot_movej(rpc, TARGET_POSE, VEL_J)

    for i in range(4):
        cloud = get_camera_cloud()
        base_cloud = transform_cloud(cloud, HAND_EYE_R, HAND_EYE_T, TARGET_POSE)
        o3d.io.write_point_cloud(f"cloud_face_{i}.ply", base_cloud)
        face, pt1, pt2, normal, axis = fit_plane_and_pca_line(
            crop_point_cloud(base_cloud, BBOX_POINTS)
        )
        # visualize face, line points, TCP poses, and base
        R_tcp, _ = compute_tcp_orientation(pt1, pt2, normal)
        visualize_line_and_poses(face, pt1, pt2, normal, axis, R_tcp)
        move_along_line(rpc, pt1, pt2, normal, VEL_L, OFFSET, SHIFT)
        if i < 3:
            ESP32Controller().move_motor(steps=50, direction=1, delay_us=5000)
            time.sleep(2)


if __name__ == "__main__":
    main()
