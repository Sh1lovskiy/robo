#!/usr/bin/env python3
"""Point cloud aggregation utilities."""

from __future__ import annotations

import os
import argparse
from typing import Any

import cv2
import numpy as np
import open3d as o3d

from utils.logger import Logger, LoggerType
from utils.cli import Command, CommandDispatcher
from utils.cloud_utils import (
    load_depth,
    load_extrinsics_json,
    load_handeye_txt,
    get_image_pairs,
)
from calibration.charuco import load_camera_params
from calibration.pose_loader import JSONPoseLoader
from utils.settings import paths, handeye
from vision.pointcloud.generator import PointCloudGenerator
from vision.transform import TransformUtils


class RGBDAggregator:
    def __init__(self, logger: LoggerType | None = None) -> None:
        self.logger = logger or Logger.get_logger("cloud.aggregator")
        self.cloud_gen = PointCloudGenerator()
        self.transformer = TransformUtils()

    def aggregate(
        self,
        img_pairs: list[tuple[str, str]],
        rotations: list[np.ndarray],
        translations: list[np.ndarray],
        intrinsics: np.ndarray,
        R_handeye: np.ndarray,
        t_handeye: np.ndarray,
        R_depth2rgb: np.ndarray,
        t_depth2rgb: np.ndarray,
        use_icp: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        base_pcd: o3d.geometry.PointCloud | None = None
        for idx, ((rgb_path, depth_path), (R_tcp, t_tcp)) in enumerate(
            zip(img_pairs, zip(rotations, translations))
        ):
            self.logger.info(f"Processing frame {idx}: {rgb_path}, {depth_path}")
            rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
            depth = load_depth(depth_path)
            cam_intr = {
                "fx": intrinsics[0, 0],
                "fy": intrinsics[1, 1],
                "ppx": intrinsics[0, 2],
                "ppy": intrinsics[1, 2],
            }
            points, colors = self.cloud_gen.depth_to_cloud(depth, cam_intr, rgb)
            T_rgb_ir = self.transformer.build_transform(R_depth2rgb, t_depth2rgb)
            T_tcp_cam = self.transformer.build_transform(R_handeye, t_handeye)
            T_tcp_rgb = T_tcp_cam @ T_rgb_ir
            self.logger.info(f"Frame {idx}: {points.shape[0]} points (in RGB frame).")
            T_base_tcp = self.transformer.build_transform(R_tcp, t_tcp)
            T_base_cam = T_base_tcp @ T_tcp_rgb
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
                    threshold = 0.003
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
        assert base_pcd is not None
        all_points = np.asarray(base_pcd.points)
        all_colors = np.asarray(base_pcd.colors) if base_pcd.has_colors() else None
        return all_points, all_colors

    def save_cloud(
        self, points: np.ndarray, colors: np.ndarray | None, out_path: str
    ) -> None:
        self.cloud_gen.save_ply(out_path, points, colors)
        self.logger.info(f"Aggregated cloud saved: {out_path}")


def _add_aggregate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data_dir", default=str(paths.CAPTURES_DIR))
    parser.add_argument("--extrinsics_json", default="realsense_extrinsics.json")
    parser.add_argument("--charuco_xml", default=str(handeye.charuco_xml))
    parser.add_argument("--handeye_txt", default="calibration/results/handeye_TSAI.txt")
    parser.add_argument(
        "--icp", action="store_true", help="Enable ICP alignment between frames"
    )


def _run_aggregate(args: argparse.Namespace) -> None:
    logger = Logger.get_logger("cloud.pipeline")
    data_dir = args.data_dir
    extrinsics_txt = args.extrinsics_json
    charuco_xml = args.charuco_xml
    handeye_txt = args.handeye_txt

    K, _ = load_camera_params(charuco_xml)
    logger.info(f"Camera intrinsics loaded from {charuco_xml}.")
    R_handeye, t_handeye = load_handeye_txt(handeye_txt)
    logger.info(f"Hand-eye calibration loaded from {handeye_txt}.")
    Rs, ts = JSONPoseLoader.load_poses(os.path.join(data_dir, "poses.json"))
    logger.info(f"{len(Rs)} poses loaded from {data_dir}/poses.json.")
    img_pairs = get_image_pairs(data_dir)
    logger.info(f"Found {len(img_pairs)} RGB/depth image pairs in {data_dir}.")
    R_depth2rgb, t_depth2rgb = load_extrinsics_json(extrinsics_txt, logger)
    aggregator = RGBDAggregator(logger)
    points, colors = aggregator.aggregate(
        img_pairs,
        Rs,
        ts,
        K,
        R_handeye,
        t_handeye,
        R_depth2rgb,
        t_depth2rgb,
        use_icp=args.icp,
    )
    output_path = (
        os.path.join(data_dir, "cloud_aggregated_icp.ply")
        if args.icp
        else os.path.join(data_dir, "cloud_aggregated.ply")
    )
    aggregator.save_cloud(points, colors, output_path)
    logger.info(f"Point cloud aggregation completed. ICP: {args.icp}")


def create_cli() -> CommandDispatcher:
    return CommandDispatcher(
        "Point cloud aggregation utilities",
        [
            Command(
                "aggregate",
                _run_aggregate,
                _add_aggregate_args,
                "Aggregate captured clouds (with extrinsics)",
            )
        ],
    )


def main() -> None:
    logger = Logger.get_logger("vision.aggregator")
    create_cli().run(logger=logger)


if __name__ == "__main__":
    main()
