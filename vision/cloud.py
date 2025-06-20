#!/usr/bin/env python3
# vision/cloud.py
"""
Build and aggregate a 3D point cloud from RGB, depth maps, and robot poses,
using camera and hand-eye calibration. Optionally adds ICP alignment for each frame.
"""

import os
import glob
import cv2
import json
import numpy as np
import open3d as o3d
import argparse
from utils.logger import Logger
from utils.io import load_camera_params
from calibration.pose_loader import JSONPoseLoader
from vision.pointcloud import PointCloudGenerator
from vision.transform import TransformUtils

# DATA_DIR = "cloud"
DATA_DIR = "captures"
CAM_CALIB_PATH = "calibration/results/charuco_cam.xml"
HANDEYE_PATH = "calibration/results/handeye_PARK.txt"
POSES_PATH = "captures/poses.json"
OUTPUT_PLY_ICP = os.path.join(DATA_DIR, "cloud_aggregated_icp.ply")
OUTPUT_PLY_NOICP = os.path.join(DATA_DIR, "cloud_aggregated.ply")
DEPTH_SCALE = 0.001


def load_handeye_txt(path):
    with open(path, "r") as f:
        lines = f.readlines()
    R = []
    t = []
    for line in lines:
        if line.startswith("R"):
            continue
        elif line.startswith("t"):
            continue
        vals = [float(x) for x in line.strip().split()]
        if len(vals) == 3:
            if len(R) < 3:
                R.append(vals)
            else:
                t = np.array(vals)
    R = np.array(R)
    t = np.array(t)
    return R, t




def load_depth(depth_path):
    depth = np.load(depth_path)
    if np.issubdtype(depth.dtype, np.integer):
        depth = depth.astype(np.float32) * DEPTH_SCALE
    return depth


def get_image_pairs(data_dir):
    rgb_list = sorted(glob.glob(os.path.join(data_dir, "*_rgb.*")))
    depth_list = sorted(glob.glob(os.path.join(data_dir, "*_depth.*")))
    assert len(rgb_list) == len(depth_list), "RGB and depth image count mismatch."
    return list(zip(rgb_list, depth_list))

class PointCloudAggregator:
    def __init__(self, logger=None):
        self.logger = logger or Logger.get_logger("cloud.aggregator")
        self.cloud_gen = PointCloudGenerator()
        self.transformer = TransformUtils()

    def aggregate(
        self,
        img_pairs,
        rotations,
        translations,
        intrinsics,
        R_handeye,
        t_handeye,
        use_icp=False,
    ):
        all_points, all_colors = [], []
        base_pcd = None

        for idx, ((rgb_path, depth_path), (R_tcp, t_tcp)) in enumerate(
            zip(img_pairs, zip(rotations, translations))
        ):
            self.logger.info(f"Processing frame {idx}: {rgb_path}, {depth_path}")

            rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
            depth = load_depth(depth_path)

            R_base_tcp = R_tcp
            t_base_tcp = t_tcp

            cam_intr = {
                "fx": intrinsics[0, 0],
                "fy": intrinsics[1, 1],
                "ppx": intrinsics[0, 2],
                "ppy": intrinsics[1, 2],
            }
            points, colors = self.cloud_gen.depth_to_cloud(depth, cam_intr, rgb)
            self.logger.info(f"Frame {idx}: {points.shape[0]} points generated.")

            T_base_tcp = self.transformer.build_transform(R_base_tcp, t_base_tcp)
            T_tcp_cam = self.transformer.build_transform(R_handeye, t_handeye)
            T_base_cam = T_base_tcp @ T_tcp_cam
            points_world = self.transformer.transform_points(points, T_base_cam)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_world)
            if colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(colors)

            if idx == 0:
                base_pcd = pcd
                self.logger.info("First cloud set as base.")
            else:
                if use_icp:
                    threshold = 0.003  # meters
                    reg = o3d.pipelines.registration.registration_icp(
                        pcd,
                        base_pcd,
                        threshold,
                        np.eye(4),
                        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    )
                    self.logger.info(
                        f"ICP frame {idx}: RMSE={reg.inlier_rmse:.5f}, iterations={len(reg.correspondence_set)}"
                    )
                    pcd.transform(reg.transformation)
                base_pcd += pcd
                base_pcd = base_pcd.voxel_down_sample(voxel_size=0.003)

        all_points = np.asarray(base_pcd.points)
        all_colors = np.asarray(base_pcd.colors) if base_pcd.has_colors() else None
        return all_points, all_colors

    def save_cloud(self, points, colors, out_path):
        self.cloud_gen.save_ply(out_path, points, colors)
        self.logger.info(f"Aggregated cloud saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--icp", action="store_true", help="Enable ICP alignment between frames"
    )
    args = parser.parse_args()

    logger = Logger.get_logger("cloud.pipeline", console_output=True)

    K, _ = load_camera_params(CAM_CALIB_PATH)
    logger.info("Camera intrinsics loaded.")
    R_handeye, t_handeye = load_handeye_txt(HANDEYE_PATH)
    logger.info("Hand-eye calibration loaded.")
    Rs, ts = JSONPoseLoader.load_poses(POSES_PATH)
    logger.info(f"{len(Rs)} poses loaded.")
    img_pairs = get_image_pairs(DATA_DIR)
    logger.info(f"Found {len(img_pairs)} RGB/depth image pairs.")

    aggregator = PointCloudAggregator(logger)
    points, colors = aggregator.aggregate(
        img_pairs, Rs, ts, K, R_handeye, t_handeye, use_icp=args.icp
    )
    output_path = OUTPUT_PLY_ICP if args.icp else OUTPUT_PLY_NOICP
    aggregator.save_cloud(points, colors, output_path)

    logger.info("Point cloud aggregation completed. ICP: %s", args.icp)


if __name__ == "__main__":
    main()
