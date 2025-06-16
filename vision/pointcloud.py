#!/usr/bin/env python3
"""
Build, ICP-align, and merge all clouds (with RGB color) in cloud_dir into a single base-frame point cloud.
Allows picking any point to see its base-frame coordinates.
"""

import argparse
import os
import glob
import numpy as np
import open3d as o3d
import json
import cv2
from scipy.spatial.transform import Rotation as R


def load_camera_params(xml_path):
    """Load camera matrix and distortion coefficients from OpenCV XML file."""
    fs = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("dist_coeffs").mat()
    fs.release()
    return camera_matrix, dist_coeffs


def pose_to_transform(pose, angles_in_deg=True):
    """
    Convert TCP pose [x, y, z, Rx, Ry, Rz] to SE(3) 4x4 matrix.
    Angles can be in degrees (default) or radians.
    """
    x, y, z, rx, ry, rz = pose
    x, y, z = x / 1000.0, y / 1000.0, z / 1000.0
    if angles_in_deg:
        rx, ry, rz = np.deg2rad([rx, ry, rz])
    rot = R.from_euler("xyz", [rx, ry, rz]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = [x, y, z]
    return T


def depth_to_cloud(depth, intr, rgb=None):
    h, w = depth.shape
    fx, fy = intr["fx"], intr["fy"]
    cx, cy = intr["ppx"], intr["ppy"]
    mask = (depth > 0.1) & (depth < 2.0)
    ys, xs = np.where(mask)
    zs = depth[ys, xs]
    xs_ = (xs - cx) * zs / fx
    ys_ = (ys - cy) * zs / fy
    points = np.stack((xs_, ys_, zs), axis=1)
    if rgb is not None:
        if rgb.shape[2] == 4:
            rgb = rgb[..., :3]
        colors = rgb[ys, xs][:, ::-1] / 255.0
        return points, colors
    return points, None


def downsample_cloud(pcd, voxel_size=0.005):
    """Downsample point cloud for faster registration."""
    return pcd.voxel_down_sample(voxel_size)


def icp_pairwise_align(source, target, threshold=0.02):
    """Align source cloud to target cloud using ICP (point-to-point)."""
    reg = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    print(f"ICP fitness: {reg.fitness:.3f}, RMSE: {reg.inlier_rmse:.4f}")
    return reg.transformation


def merge_clouds(clouds, voxel_size=0.003):
    """Merge list of clouds into one, with optional voxel downsample."""
    merged = o3d.geometry.PointCloud()
    for pcd in clouds:
        merged += pcd
    if voxel_size:
        merged = merged.voxel_down_sample(voxel_size)
    return merged


def filter_cloud(pcd, nb_neighbors=20, std_ratio=2.0):
    """Remove outliers from point cloud using statistical filter."""
    cl, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    print(f"Filtered: {len(cl.points)} points remain out of {len(pcd.points)}")
    return cl


def visualize_and_print_coords(pcd, picked_file="picked_points.txt"):
    print("INSTRUCTIONS:")
    print("  - Left-click to pick points (marked in red).")
    print("  - Press 'Q' to close the window.")
    print(f"  - After closing the window, indices from {picked_file} will be printed.")
    o3d.visualization.draw_geometries_with_editing([pcd])


def main():
    parser = argparse.ArgumentParser(
        description="Build, ICP-align, merge, filter and view colored base-frame point cloud from depth+png+poses."
    )
    parser.add_argument(
        "--cloud_dir",
        required=True,
        help="Folder with *_depth.npy, *_rgb.png, poses.json, etc.",
    )
    parser.add_argument("--handeye", required=True, help="Hand-eye calibration .npz")
    parser.add_argument("--camxml", required=True, help="Camera intrinsics XML")
    parser.add_argument("--output", required=True, help="Output merged PLY file")
    parser.add_argument(
        "--voxel",
        type=float,
        default=0.005,
        help="Voxel size for downsample before registration (m)",
    )
    parser.add_argument(
        "--merge_voxel", type=float, default=0.003, help="Voxel size for merge (m)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.02, help="ICP threshold (m)"
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="Apply noise/statistical filter before viewing/saving",
    )
    parser.add_argument(
        "--nb_neighbors", type=int, default=20, help="Neighbors for outlier removal"
    )
    parser.add_argument(
        "--std_ratio", type=float, default=2.0, help="Std ratio for outlier removal"
    )
    parser.add_argument(
        "--rgb_suffix",
        type=str,
        default="_rgb.png",
        help="Suffix for color images (default: _rgb.png)",
    )
    args = parser.parse_args()

    # Load camera intrinsics and hand-eye
    camera_matrix, _ = load_camera_params(args.camxml)
    handeye = np.load(args.handeye)
    R_handeye = handeye["R"]
    t_handeye = handeye["t"].flatten()

    # Load TCP poses
    with open(os.path.join(args.cloud_dir, "poses.json")) as f:
        tcp_poses = json.load(f)

    # Build all clouds with RGB
    depth_files = sorted(
        [f for f in os.listdir(args.cloud_dir) if f.endswith("_depth.npy")]
    )
    clouds = []
    for depth_file in depth_files:
        key = depth_file.split("_")[0]
        pose = tcp_poses[key]["tcp_coords"]
        T_base_tcp = pose_to_transform(pose, angles_in_deg=True)
        T_tcp_cam = np.eye(4)
        T_tcp_cam[:3, :3] = R_handeye
        T_tcp_cam[:3, 3] = t_handeye
        T_base_cam = T_base_tcp @ T_tcp_cam

        depth = (
            np.load(os.path.join(args.cloud_dir, depth_file)).astype(np.float32) * 0.001
        )  # meters

        rgb_file = os.path.join(args.cloud_dir, f"{key}{args.rgb_suffix}")
        if not os.path.exists(rgb_file):
            print(
                f"WARNING: RGB file not found for {depth_file} -> {rgb_file}. Skipping color."
            )
            rgb = None
        else:
            rgb = cv2.imread(rgb_file, cv2.IMREAD_UNCHANGED)
            if rgb is None:
                print(f"Failed to load color image: {rgb_file}")
        intr = {
            "fx": camera_matrix[0, 0],
            "fy": camera_matrix[1, 1],
            "ppx": camera_matrix[0, 2],
            "ppy": camera_matrix[1, 2],
            "width": depth.shape[1],
            "height": depth.shape[0],
        }
        points, colors = depth_to_cloud(depth, intr, rgb=rgb)
        print(f"cloud {depth_file}: points shape = {points.shape}")

        # To Open3D pointcloud (in camera frame)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.transform(T_base_cam)
        clouds.append(pcd)

    if not clouds or len(clouds) < 2:
        print("Need at least 2 point clouds to merge!")
        return

    # Registration and merge
    clouds_down = [downsample_cloud(c, args.voxel) for c in clouds]
    ref = clouds_down[0]
    transformed_clouds = [clouds[0]]
    for i in range(1, len(clouds)):
        print(f"Registering cloud {i} to reference...")
        transformation = icp_pairwise_align(
            clouds_down[i], ref, threshold=args.threshold
        )
        clouds[i].transform(transformation)
        transformed_clouds.append(clouds[i])

    merged = merge_clouds(transformed_clouds, voxel_size=args.merge_voxel)
    if args.filter:
        merged = filter_cloud(
            merged, nb_neighbors=args.nb_neighbors, std_ratio=args.std_ratio
        )

    o3d.io.write_point_cloud(args.output, merged)
    print(f"Saved merged cloud: {args.output}")
    visualize_and_print_coords(merged)


if __name__ == "__main__":
    main()
