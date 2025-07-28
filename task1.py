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

# Constants
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

TARGET_POSE = np.array([-66.2, -135.7, 504.3, -119.2, 4.5, 110.8])
BBOX_POINTS = np.array(
    [
        [-0.57, -0.34, 0.46],
        [-0.57, -0.24, 0.27],
        [-0.38, -0.34, 0.27],
        [-0.38, -0.34, 0.46],
    ]
)


def robot_connect(ip):
    logger = Logger.get_logger("task")
    logger.info(f"Connecting to robot at {ip}")
    rpc = RPC(ip=ip)
    code_enable = rpc.RobotEnable(1)
    logger.info(f"RobotEnable(1): {code_enable}")
    if code_enable != 0:
        logger.error("Failed to enable robot!")
        raise RuntimeError("Failed to enable robot!")
    safety = rpc.GetSafetyCode()
    logger.info(f"GetSafetyCode: {safety}")
    if safety != 0:
        logger.error("Robot is not in a safe state!")
        raise RuntimeError("Robot is not in a safe state!")
    logger.info("Robot ready to move")
    return rpc


def robot_movej(rpc, target_pose, vel=30, reset_last_joint=False):
    logger = Logger.get_logger("task")
    if isinstance(target_pose, np.ndarray):
        target_pose = target_pose.tolist()
    logger.info(f"Planning MoveJ to: {target_pose}")
    result = rpc.GetInverseKin(0, target_pose)
    logger.info(f"GetInverseKin result: {result}")
    if isinstance(result, tuple) and len(result) == 2:
        code, joints = result
    else:
        logger.error(f"GetInverseKin failed: {result}")
        raise RuntimeError(f"GetInverseKin failed: {result}")
    if code == 0 and joints:
        if reset_last_joint:
            joints = list(joints)
            joints[5] = 0.0
            logger.info(f"Set last joint to zero: {joints}")
        code_j = rpc.MoveJ(joint_pos=joints, tool=0, user=0, vel=vel)
        logger.info(f"MoveJ: {code_j}")
        if code_j != 0:
            logger.error(f"MoveJ failed, code={code_j}")
            raise RuntimeError(f"MoveJ failed, code={code_j}")
    else:
        logger.error(f"GetInverseKin failed, code={code}, joints={joints}")
        raise RuntimeError(f"GetInverseKin failed, code={code}, joints={joints}")
    logger.info("Motion complete, waiting 1 sec...")
    time.sleep(1)


def get_camera_cloud():
    logger = Logger.get_logger("task")
    logger.info("Starting RealSense pipeline and capturing cloud...")
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
    min_depth = 0.2
    max_depth = 2.0
    for _ in range(10):
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    color_img = np.asanyarray(color_frame.get_data())
    depth_img = np.asanyarray(depth_frame.get_data())
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    logger.info(f"Depth scale: {depth_scale}")

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
        depth_scale=1,
        # depth_trunc=2.0,
        convert_rgb_to_intensity=False,
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole)
    points = np.asarray(pcd.points)
    mask = (points[:, 2] > min_depth) & (points[:, 2] < max_depth)
    pcd = pcd.select_by_index(np.where(mask)[0])
    logger.info(f"Cloud has {len(pcd.points)} points")
    pipeline.stop()
    logger.info("Pipeline stopped")
    return pcd


def pose_to_transform(pose, angles_in_deg=True):
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (6,):
        raise ValueError("Pose must be a sequence of 6 floats: [x, y, z, rx, ry, rz]")
    position = pose[:3] / 1000.0  # mm → m
    rotation = R.from_euler("xyz", pose[3:], degrees=angles_in_deg).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = position
    return T


def transform_cloud_to_tcp(pcd, handeye_R, handeye_t, tcp_pose):
    """
    Трансформирует облако точек из камеры в базу робота и далее в tcp через одну матрицу.
    Возвращает облако уже в tcp, матрицу T_cam2tcp и облако в базе (если нужно).
    """
    logger = Logger.get_logger("task")
    # HandEye в одну SE(3) матрицу (camera→base)
    T_cam2base = np.eye(4)
    T_cam2base[:3, :3] = handeye_R
    T_cam2base[:3, 3] = handeye_t.flatten()
    logger.info(f"T_cam2base:\n{T_cam2base}")

    # TCP как SE(3): (base→tcp)
    T_base2tcp = pose_to_transform(tcp_pose)
    logger.info(f"T_base2tcp:\n{T_base2tcp}")

    # Итоговое camera→tcp: T_cam2tcp = T_base2tcp @ T_cam2base
    T_cam2tcp = T_base2tcp @ T_cam2base
    logger.info(f"T_cam2tcp (camera→tcp):\n{T_cam2tcp}")

    # Трансформируем все точки
    pts = np.asarray(pcd.points)
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))]).T  # (4, N)
    pts_tcp_h = T_cam2tcp @ pts_h
    pts_tcp = pts_tcp_h[:3, :].T

    # Для сохранения можно вернуть и pts_base (после handeye)
    pts_base_h = T_cam2base @ pts_h
    pts_base = pts_base_h[:3, :].T

    # Создаем облака
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
    logger = Logger.get_logger("task")
    bbox_pcd = o3d.geometry.PointCloud()
    bbox_pcd.points = o3d.utility.Vector3dVector(BBOX_POINTS)
    aabb = bbox_pcd.get_axis_aligned_bounding_box()
    logger.info(f"Cropping cloud to bbox: {BBOX_POINTS.tolist()}")
    cropped_pcd = pcd.crop(aabb)
    logger.info(f"Cropped cloud points: {len(cropped_pcd.points)}")
    return cropped_pcd


def get_main_plane_and_pca_line(cropped_pcd):
    logger = Logger.get_logger("task")
    if len(cropped_pcd.points) < 4:
        logger.error("Not enough points to compute a face.")
        raise RuntimeError("Not enough points to compute a face.")
    # 1. Удаление выбросов
    pcd_filt, _ = cropped_pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
    logger.info(f"Filtered outliers, remaining points: {len(pcd_filt.points)}")
    # 2. Поиск самой большой плоскости через RANSAC
    plane_model, inliers = pcd_filt.segment_plane(
        distance_threshold=0.004, ransac_n=3, num_iterations=1000
    )
    plane_cloud = pcd_filt.select_by_index(inliers)
    logger.info(f"Plane inliers: {len(inliers)}")
    if len(inliers) < 4:
        logger.error("Not enough inliers on main plane.")
        raise RuntimeError("Not enough inliers on main plane.")
    # 3. PCA только по этой плоскости
    pts = np.asarray(plane_cloud.points)
    center = pts.mean(axis=0)
    pts_c = pts - center
    U, S, Vt = np.linalg.svd(pts_c, full_matrices=False)
    direction = Vt[0]
    # Для визуализации: два конца линии по границе точек
    projections = pts_c @ direction
    t_min, t_max = projections.min(), projections.max()
    pt1 = center + t_min * direction
    pt2 = center + t_max * direction
    # Нормаль к плоскости
    normal = Vt[2]
    logger.info(f"Main plane center: {center}")
    logger.info(f"PCA axis: {direction}")
    logger.info(f"Plane normal: {normal}")
    logger.info(f"PCA line: pt1={pt1}, pt2={pt2}")
    return plane_cloud, (pt1, pt2), center, direction, normal


def principal_axis_line(face: o3d.geometry.TriangleMesh, return_extents=False):
    """
    По точкам largest face строит PCA и возвращает:
    - точку центра (среднюю)
    - главный вектор (основное направление поверхности)
    - если return_extents=True: координаты концов линии по границе проекций
    """
    # Получить точки грани
    pts = np.asarray(face.vertices)
    if pts.shape[0] < 2:
        raise ValueError("Not enough points for PCA")
    center = pts.mean(axis=0)
    # PCA через SVD
    pts_c = pts - center
    U, S, Vt = np.linalg.svd(pts_c, full_matrices=False)
    direction = Vt[0]  # главный компонент (ось максимального разброса)
    # Для наглядности — концы линии по проекции на эту ось
    if return_extents:
        projections = pts_c @ direction
        t_min, t_max = projections.min(), projections.max()
        pt1 = center + t_min * direction
        pt2 = center + t_max * direction
        return center, direction, (pt1, pt2)
    return center, direction


def robot_movel(rpc, desc_pos, vel=30, reset_last_joint=False):
    logger = Logger.get_logger("task")
    if isinstance(desc_pos, np.ndarray):
        desc_pos = desc_pos.tolist()
    logger.info(f"Planning MoveL to: {desc_pos}")
    # Получим IK (опционально)
    result = rpc.GetInverseKin(0, desc_pos)
    logger.info(f"GetInverseKin for MoveL: {result}")
    if isinstance(result, tuple) and len(result) == 2:
        code, joints = result
    else:
        logger.error(f"GetInverseKin failed: {result}")
        raise RuntimeError(f"GetInverseKin failed: {result}")
    if code == 0 and joints:
        if reset_last_joint:
            joints = list(joints)
            joints[5] = 0.0
            logger.info(f"Set last joint to zero for MoveL: {joints}")
        # MoveL c этим joint_pos
        code_l = rpc.MoveL(desc_pos=desc_pos, tool=0, user=0, joint_pos=joints, vel=vel)
        logger.info(f"MoveL: {code_l}")
        if code_l != 0:
            logger.error(f"MoveL failed, code={code_l}")
            raise RuntimeError(f"MoveL failed, code={code_l}")
    else:
        logger.error(f"GetInverseKin failed for MoveL, code={code}, joints={joints}")
        raise RuntimeError(
            f"GetInverseKin failed for MoveL, code={code}, joints={joints}"
        )
    logger.info("Linear motion complete, waiting 1 sec...")
    time.sleep(1)


def compute_tcp_euler(pt_from, pt_to, normal_prefer):
    x_axis = pt_to - pt_from
    x_axis = x_axis / np.linalg.norm(x_axis)
    x_axis = -x_axis

    z_axis = normal_prefer / np.linalg.norm(normal_prefer)
    z_axis = -z_axis

    x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    R_tcp = np.column_stack((x_axis, y_axis, z_axis))

    if np.linalg.det(R_tcp) < 0:
        y_axis = -y_axis
        R_tcp = np.column_stack((x_axis, y_axis, z_axis))

    r = R.from_matrix(R_tcp)
    rx, ry, rz = r.as_euler("xyz", degrees=True)
    return R_tcp, (rx, ry, rz), x_axis, y_axis, z_axis


def move_robot_along_line(
    rpc, pt1, pt2, normal, tcp_shift=0.0, offset=[-0.22, 0.0, -0.12], vel=40
):
    """
    Двигает робота:
    - сначала подводит его на расстояние offset от точки pt1 по нормали к грани (перпендикулярно),
    - с учетом смещения tcp_shift по x в системе TCP (например, -0.022 м, если лазер сдвинут относительно TCP)
    - после 5 сек перемещается аналогично к pt2 (конец линии),
    - потом возвращается обратно в исходную позицию.

    Все точки подаются в метрах, normal должна быть направлена "наружу" от детали.
    """
    logger = Logger.get_logger("task")
    # 1. Текущая позиция для возврата
    code, home_joints = rpc.GetActualJointPosDegree()
    logger.info(f"Current joint pos (for return): {home_joints}")

    # 2. Определить угол ориентации TCP, чтобы ось Z TCP была вдоль normal, а X — вдоль линии (pt2-pt1)
    R_tcp, (rx, ry, rz), x_axis, y_axis, z_axis = compute_tcp_euler(pt1, pt2, normal)

    approach_vec = normal / np.linalg.norm(normal) * offset
    approach_point = pt1 + approach_vec + x_axis * tcp_shift  # теперь x_axis определён

    pose1 = np.concatenate([approach_point * 1000, [rx, ry, rz]])
    code = rpc.GetInverseKin(0, pose1)

    logger.info(f"Move orientation: R=\n{R_tcp}\nEuler angles: {rx}, {ry}, {rz}")

    # 3. Рассчитываем точку для подвода (от pt1 по нормали outwards + смещение TCP)

    # Смещение TCP по X (в локальной системе TCP): сдвигаем на -22 мм вдоль x_axis

    logger.info(f"Approach pose to pt1: {pose1}")

    esp32 = ESP32Controller()
    # Движение к pt1
    robot_movej(rpc, pose1, vel=vel, reset_last_joint=False)

    logger.info("Arrived to pt1 (start of line), holding for 5 sec...")
    time.sleep(1)
    esp32.laser_on()
    time.sleep(1)

    # Аналогично для второй точки (pt2)
    approach_point2 = pt2 + approach_vec
    pose2 = np.concatenate([approach_point2 * 1000, [rx, ry, rz]])
    logger.info(f"Approach pose to pt2: {pose2}")
    # Движение к pt2
    robot_movel(rpc, pose2, vel=vel, reset_last_joint=False)

    logger.info("Arrived to pt2 (end of line), holding for 2 sec...")
    time.sleep(1)
    esp32.laser_off()
    time.sleep(1)
    # Возврат домой
    code = rpc.MoveJ(joint_pos=home_joints, tool=0, user=0, vel=vel)
    logger.info(f"Returned to initial position, code={code}")


def main():
    logger = Logger.get_logger("task")

    rpc = robot_connect(ip="192.168.58.2")
    robot_movej(rpc, TARGET_POSE)
    esp32 = ESP32Controller()

    for i in range(4):
        logger.info(f"===== ГРАНЬ {i+1} ИЗ 4 =====")
        # 1. Сканируем облако
        pcd = get_camera_cloud()

        # 2. Переводим в tcp
        pcd_tcp, pcd_base, T_cam2tcp = transform_cloud_to_tcp(
            pcd, HAND_EYE_R, HAND_EYE_t, TARGET_POSE
        )
        filename = (
            f"task_{i}_x{TARGET_POSE[0]:.1f}_y{TARGET_POSE[1]:.1f}_z{TARGET_POSE[2]:.1f}_"
            f"rx{TARGET_POSE[3]:.1f}_ry{TARGET_POSE[4]:.1f}_rz{TARGET_POSE[5]:.1f}_robot.ply"
        )
        logger.info("Visualizing transform cloud...")
        o3d.io.write_point_cloud(filename, pcd_tcp)
        logger.info(f"Point cloud (robot coords) saved: {filename}")

        # 3. Обработка — bbox, поиск грани, линия и нормаль
        cropped_pcd = get_bbox_crop(pcd_tcp)
        logger.info("Visualizing cropped cloud (robot coords)...")
        plane_cloud, (pt1, pt2), center, direction, normal = (
            get_main_plane_and_pca_line(cropped_pcd)
        )
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector([pt1, pt2]),
            lines=o3d.utility.Vector2iVector([[0, 1]]),
        )
        line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
        # o3d.visualization.draw_geometries([plane_cloud, line_set])

        # 4. Робот обходит линию
        move_robot_along_line(
            rpc, pt1, pt2, normal, tcp_shift=-0.022, offset=0.3, vel=30
        )

        if i < 3:
            logger.info(f"Поворачиваем деталь для следующей грани...")
            esp32.move_motor(steps=50, direction=1, delay_us=5000)
            # Время на поворот детали (или ожидание стабилизации)
            time.sleep(2)


if __name__ == "__main__":
    main()
