import numpy as np
import open3d as o3d

from save_rotate_clouds import transform_cloud_to_tcp


def load_point_cloud(path):
    return o3d.io.read_point_cloud(path)


def filter_bbox(pcd, bbox_points):
    bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(bbox_points)
    )
    return pcd.crop(bbox)


def rotate_about_axis(pcd, angle_deg, axis="z"):
    angle_rad = np.radians(angle_deg)
    if axis == "z":
        R = pcd.get_rotation_matrix_from_axis_angle([0, 0, angle_rad])
    elif axis == "x":
        R = pcd.get_rotation_matrix_from_axis_angle([angle_rad, 0, 0])
    elif axis == "y":
        R = pcd.get_rotation_matrix_from_axis_angle([0, angle_rad, 0])
    pcd.rotate(R, center=(0, 0, 0))
    return pcd


def main():
    HAND_EYE_R = np.array(
        [
            [0.999048, 0.02428, -0.03625],
            [-0.02706, 0.99658, -0.07804],
            [0.03423, 0.07895, 0.99629],
        ]
    )
    HAND_EYE_t = np.array([-0.03424, -0.07905, 0.00128]).reshape(3, 1)

    TARGET_POSE = np.array([-66.2, -135.7, 504.3])
    BBOX_POINTS = np.array(
        [
            [-0.57, -0.34, 0.46],
            [-0.57, -0.05, 0.27],
            [-0.38, -0.05, 0.27],
            [-0.38, -0.05, 0.46],
        ]
    )

    angles = [0, 90, 180, 270]
    clouds = []
    for i, angle in enumerate(angles):
        pcd_path = f".data_clouds/cloud_base_face{i}_x-66.2_y-135.7_z504.3.ply"
        pcd = load_point_cloud(pcd_path)
        pcd = transform_cloud_to_tcp(pcd, HAND_EYE_R, HAND_EYE_t, TARGET_POSE)
        pcd = filter_bbox(pcd, BBOX_POINTS)
        o3d.visualization.draw_geometries([pcd])
        if angle != 0:
            pcd = rotate_about_axis(pcd, angle, axis="z")
        clouds.append(pcd)

    combined_pcd = clouds[0]
    for cloud in clouds[1:]:
        combined_pcd += cloud

    o3d.visualization.draw_geometries([combined_pcd])


if __name__ == "__main__":
    main()
