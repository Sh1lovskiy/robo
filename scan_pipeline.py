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

# Hand-eye calibration matrix (camera to robot base)
HAND_EYE_R = np.array(
    [
        [0.9989168906402133, 0.022559426924731108, -0.040695427882205995],
        [-0.026030236545789346, 0.9958787154587994, -0.08687928914017393],
        [0.03856776346974718, 0.08784450098304053, 0.9953872971200619],
    ]
)
HAND_EYE_t = np.array(
    [-0.035115287744637756, -0.0795335678934103, 0.00023574569552313]
).reshape((3, 1))

# Target robot pose for scanning [x,y,z,rx,ry,rz] (mm and degrees)
TARGET_POSE = np.array([-66.2, -135.7, 504.3, -119.2, 4.5, 110.8])

# Bounding box points for cropping (in meters)
BBOX_POINTS = np.array(
    [
        [-0.57, -0.34, 0.46],
        [-0.57, -0.24, 0.27],
        [-0.38, -0.34, 0.27],
        [-0.38, -0.34, 0.46],
    ]
)


def robot_connect(ip):
    """Connect to robot controller and verify safety"""
    logger = Logger.get_logger("task")
    logger.info(f"Connecting to robot at {ip}")
    rpc = RPC(ip=ip)
    code_enable = rpc.RobotEnable(1)
    if code_enable != 0:
        logger.error("Failed to enable robot!")
        raise RuntimeError("Failed to enable robot!")
    safety = rpc.GetSafetyCode()
    if safety != 0:
        logger.error("Robot is not in a safe state!")
        raise RuntimeError("Robot is not in a safe state!")
    logger.info("Robot ready to move")
    return rpc


def robot_movej(rpc, target_pose, vel=30.0, acc=30.0, reset_last_joint=False):
    """Move robot to joint position"""
    logger = Logger.get_logger("task")
    if isinstance(target_pose, np.ndarray):
        target_pose = target_pose.tolist()
    result = rpc.GetInverseKin(0, target_pose)
    if isinstance(result, tuple) and len(result) == 2:
        code, joints = result
    else:
        raise RuntimeError(f"GetInverseKin failed: {result}")

    if code == 0 and joints:
        if reset_last_joint:
            joints = list(joints)
            joints[5] = 0.0
        code_j = rpc.MoveJ(joint_pos=joints, tool=0, user=0, vel=vel, acc=acc)
        if code_j != 0:
            raise RuntimeError(f"MoveJ failed, code={code_j}")
    else:
        raise RuntimeError(f"GetInverseKin failed, code={code}, joints={joints}")
    time.sleep(1)


def get_camera_cloud():
    """Capture point cloud from RealSense camera"""
    logger = Logger.get_logger("task")
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

    # Skip first few frames for auto-exposure
    for _ in range(10):
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

    # Process frames
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    color_img = np.asanyarray(color_frame.get_data())
    depth_img = np.asanyarray(depth_frame.get_data())
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    # Convert to Open3D format
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

    # Create RGBD image and point cloud
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)),
        o3d.geometry.Image(depth_meter),
        depth_scale=1,
        convert_rgb_to_intensity=False,
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole)

    # Filter by depth range
    points = np.asarray(pcd.points)
    mask = (points[:, 2] > 0.2) & (points[:, 2] < 2.0)
    pcd = pcd.select_by_index(np.where(mask)[0])

    pipeline.stop()
    return pcd


def pose_to_transform(pose, angles_in_deg=True):
    """Convert 6D pose [x,y,z,rx,ry,rz] to 4x4 transformation matrix"""
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (6,):
        raise ValueError("Pose must be 6 floats: [x,y,z,rx,ry,rz]")

    position = pose[:3] / 1000.0  # mm to meters
    rotation = R.from_euler("xyz", pose[3:], degrees=angles_in_deg).as_matrix()

    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = position
    return T


def transform_cloud_to_tcp(pcd, handeye_R, handeye_t, tcp_pose):
    """
    Transform point cloud from camera to robot TCP frame
    Returns:
        - pcd_tcp: cloud in TCP frame
        - pcd_base: cloud in robot base frame
        - T_cam2tcp: transformation matrix
    """
    logger = Logger.get_logger("task")

    # Camera to base transformation
    T_cam2base = np.eye(4)
    T_cam2base[:3, :3] = handeye_R
    T_cam2base[:3, 3] = handeye_t.flatten()

    # Base to TCP transformation
    T_base2tcp = pose_to_transform(tcp_pose)

    # Combined camera to TCP transformation
    T_cam2tcp = T_base2tcp @ T_cam2base

    # Transform points
    pts = np.asarray(pcd.points)
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))]).T  # Homogeneous coords

    pts_tcp_h = T_cam2tcp @ pts_h
    pts_tcp = pts_tcp_h[:3, :].T

    pts_base_h = T_cam2base @ pts_h
    pts_base = pts_base_h[:3, :].T

    # Create output clouds
    pcd_tcp = o3d.geometry.PointCloud()
    pcd_tcp.points = o3d.utility.Vector3dVector(pts_tcp)
    if pcd.has_colors():
        pcd_tcp.colors = pcd.colors

    pcd_base = o3d.geometry.PointCloud()
    pcd_base.points = o3d.utility.Vector3dVector(pts_base)
    if pcd.has_colors():
        pcd_base.colors = pcd.colors

    return pcd_tcp, pcd_base, T_cam2tcp


def get_bbox_crop(pcd):
    """Crop point cloud to predefined bounding box"""
    logger = Logger.get_logger("task")
    bbox_pcd = o3d.geometry.PointCloud()
    bbox_pcd.points = o3d.utility.Vector3dVector(BBOX_POINTS)
    aabb = bbox_pcd.get_axis_aligned_bounding_box()
    cropped_pcd = pcd.crop(aabb)
    return cropped_pcd


def get_main_plane_and_pca_line(cropped_pcd):
    """Extract dominant plane and principal axis from cropped cloud"""
    logger = Logger.get_logger("task")

    # 1. Remove outliers
    pcd_filt, _ = cropped_pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)

    # 2. Find dominant plane using RANSAC
    plane_model, inliers = pcd_filt.segment_plane(
        distance_threshold=0.004, ransac_n=3, num_iterations=1000
    )
    plane_cloud = pcd_filt.select_by_index(inliers)

    # 3. Compute PCA on plane points
    pts = np.asarray(plane_cloud.points)
    center = pts.mean(axis=0)
    pts_c = pts - center
    U, S, Vt = np.linalg.svd(pts_c, full_matrices=False)

    # Principal direction and normal
    direction = Vt[0]  # First principal component
    normal = Vt[2]  # Plane normal

    # Line endpoints along principal axis
    projections = pts_c @ direction
    t_min, t_max = projections.min(), projections.max()
    pt1 = center + t_min * direction
    pt2 = center + t_max * direction

    return plane_cloud, (pt1, pt2), center, direction, normal


def robot_movel(rpc, desc_pos, vel=30.0, acc=30.0, reset_last_joint=False):
    """Linear robot motion to Cartesian position"""
    logger = Logger.get_logger("task")
    if isinstance(desc_pos, np.ndarray):
        desc_pos = desc_pos.tolist()

    result = rpc.GetInverseKin(0, desc_pos)
    if isinstance(result, tuple) and len(result) == 2:
        code, joints = result
    else:
        raise RuntimeError(f"GetInverseKin failed: {result}")

    if code == 0 and joints:
        if reset_last_joint:
            joints = list(joints)
            joints[5] = 0.0
        code_l = rpc.MoveL(
            desc_pos=desc_pos, tool=0, user=0, joint_pos=joints, vel=vel, acc=30.0
        )
        if code_l != 0:
            raise RuntimeError(f"MoveL failed, code={code_l}")
    else:
        raise RuntimeError(f"GetInverseKin failed, code={code}, joints={joints}")
    time.sleep(1)


def compute_tcp_euler(pt_from, pt_to, normal_prefer):
    """Compute TCP orientation from line and normal"""
    x_axis = pt_to - pt_from
    x_axis = x_axis / np.linalg.norm(x_axis)
    x_axis = -x_axis  # Flip for correct orientation

    z_axis = normal_prefer / np.linalg.norm(normal_prefer)
    z_axis = -z_axis  # Flip for correct orientation

    # Ensure orthogonality
    x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    # Build rotation matrix
    R_tcp = np.column_stack((x_axis, y_axis, z_axis))
    if np.linalg.det(R_tcp) < 0:  # Ensure right-handed system
        y_axis = -y_axis
        R_tcp = np.column_stack((x_axis, y_axis, z_axis))

    r = R.from_matrix(R_tcp)
    rx, ry, rz = r.as_euler("xyz", degrees=True)
    return R_tcp, (rx, ry, rz), x_axis, y_axis, z_axis


def move_robot_along_line(
    rpc, pt1, pt2, normal, tcp_shift=0.25, offset=[-0.22, 0.0, -1.2], vel=30.0
):
    """Move robot along detected line while maintaining orientation"""
    logger = Logger.get_logger("task")

    # Save home position for return
    code, home_joints = rpc.GetActualJointPosDegree()

    # Compute TCP orientation
    R_tcp, (rx, ry, rz), x_axis, y_axis, z_axis = compute_tcp_euler(pt1, pt2, normal)

    # Calculate approach points with offset
    approach_vec = normal * offset
    approach_point = pt1 + approach_vec + x_axis * tcp_shift

    # Create pose array [x,y,z,rx,ry,rz] in mm/degrees
    pose1 = np.concatenate([approach_point * 1000, [rx, ry, rz]])

    # Initialize laser controller
    esp32 = ESP32Controller()

    # Move to start point
    robot_movej(rpc, pose1, vel=vel, acc=30.0)
    time.sleep(0.5)
    esp32.laser_on()
    time.sleep(0.5)

    # Move to end point
    approach_point2 = (pt2 + approach_vec) * 1000
    pose2 = np.concatenate([approach_point2, [rx, ry, rz]])
    robot_movel(rpc, pose2, vel=vel, acc=30.0)

    # Finish operation
    time.sleep(0.5)
    esp32.laser_off()
    time.sleep(0.5)
    rpc.MoveJ(joint_pos=home_joints, tool=0, user=0, vel=vel)


def main():
    """Main scanning and processing routine"""
    logger = Logger.get_logger("task")
    rpc = robot_connect(ip="192.168.58.2")
    robot_movej(rpc, TARGET_POSE, vel=40.0)
    esp32 = ESP32Controller()
    esp32.laser_off()
    for i in range(4):
        logger.info(f"===== FACE {i+1} OF 4 =====")

        # 1. Capture point cloud
        pcd = get_camera_cloud()

        # 2. Transform to robot coordinates
        pcd_tcp, pcd_base, T_cam2tcp = transform_cloud_to_tcp(
            pcd, HAND_EYE_R, HAND_EYE_t, TARGET_POSE
        )

        # Save cloud with pose info
        filename = (
            f"task_{i}_x{TARGET_POSE[0]:.1f}_y{TARGET_POSE[1]:.1f}_z{TARGET_POSE[2]:.1f}_"
            f"rx{TARGET_POSE[3]:.1f}_ry{TARGET_POSE[4]:.1f}_rz{TARGET_POSE[5]:.1f}_robot.ply"
        )
        o3d.io.write_point_cloud(filename, pcd_tcp)

        # 3. Process cloud - crop, find plane and line
        cropped_pcd = get_bbox_crop(pcd_tcp)
        plane_cloud, (pt1, pt2), center, direction, normal = (
            get_main_plane_and_pca_line(cropped_pcd)
        )
        # line_set = o3d.geometry.LineSet(
        #     points=o3d.utility.Vector3dVector([pt1, pt2]),
        #     lines=o3d.utility.Vector2iVector([[0, 1]]),
        # )
        # line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
        # o3d.visualization.draw_geometries([plane_cloud, line_set])
        # 4. Move robot along detected line
        move_robot_along_line(
            rpc, pt1, pt2, normal, tcp_shift=-0.022, offset=0.3, vel=30
        )

        # Rotate part for next face (except last iteration)
        if i < 3:
            esp32.move_motor(steps=50, direction=1, delay_us=5000)
            time.sleep(2)


if __name__ == "__main__":
    main()
