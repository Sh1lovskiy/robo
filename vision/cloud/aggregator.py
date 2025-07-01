#!/usr/bin/env python3
"""Point cloud aggregation utilities."""

from __future__ import annotations

import os
import glob
import argparse
import json
from typing import Any

import cv2
import numpy as np
import open3d as o3d

from utils.logger import Logger, LoggerType
from utils.cli import Command, CommandDispatcher
from utils.config import Config
from calibration.helpers.pose_utils import load_camera_params, JSONPoseLoader
from vision.cloud.generator import PointCloudGenerator
from vision.transform import TransformUtils

DEPTH_SCALE = 0.0001


def load_handeye_txt(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "r") as f:
        lines = f.readlines()
    R: list[list[float]] = []
    t: np.ndarray = np.empty(3)
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


def load_depth(depth_path: str) -> np.ndarray:
    depth = np.load(depth_path)
    if np.issubdtype(depth.dtype, np.integer):
        depth = depth.astype(np.float32) * DEPTH_SCALE
    return depth


def get_image_pairs(data_dir: str) -> list[tuple[str, str]]:
    rgb_list = sorted(glob.glob(os.path.join(data_dir, "*_rgb.*")))
    depth_list = sorted(glob.glob(os.path.join(data_dir, "*_depth.*")))
    assert len(rgb_list) == len(depth_list), "RGB and depth image count mismatch."
    return list(zip(rgb_list, depth_list))


def load_extrinsics_json(
    json_path: str, logger: LoggerType
) -> tuple[np.ndarray, np.ndarray]:
    with open(json_path, "r") as f:
        data = json.load(f)
    R = np.array(data["depth_to_rgb"]["rotation"])
    t = np.array(data["depth_to_rgb"]["translation"])
    logger.info("Extrinsics loaded from %s", json_path)
    return R, t


import re


def load_extrinsics_txt(path: str, logger: LoggerType) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "r") as f:
        lines = f.readlines()
    matrix = []
    tvec = None
    parsing_matrix = False
    for line in lines:
        if "Rotation:" in line:
            parsing_matrix = True
            matrix = []
            continue
        if parsing_matrix:
            if "]" in line:
                parsing_matrix = False
            nums = re.findall(
                r"[-+]?[0-9]*\.?[0-9]+e[-+]?\d+|[-+]?[0-9]*\.?[0-9]+", line
            )
            if nums:
                matrix.append([float(n) for n in nums])
            continue
        if "Translation:" in line:
            nums = re.findall(
                r"[-+]?[0-9]*\.?[0-9]+e[-+]?\d+|[-+]?[0-9]*\.?[0-9]+", line
            )
            if nums:
                tvec = np.array([float(n) for n in nums])
    R = np.array(matrix)
    t = np.array(tvec)
    logger.info("Extrinsics loaded from %s", path)
    return R, t


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
            points = (R_depth2rgb @ points.T).T + t_depth2rgb
            self.logger.info(f"Frame {idx}: {points.shape[0]} points (in RGB frame).")
            T_base_tcp = self.transformer.build_transform(R_tcp, t_tcp)
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
                    threshold = 0.003
                    reg = o3d.pipelines.registration.registration_icp(
                        pcd,
                        base_pcd,
                        threshold,
                        np.eye(4),
                        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    )
                    self.logger.info(
                        f"ICP frame {idx}: RMSE={reg.inlier_rmse:.5f}, \
iterations={len(reg.correspondence_set)}"
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
    Config.load()
    cfg = Config.get("aggregator")
    default_data_dir = (
        cfg.get("data_dir", "captures") if cfg is not None else "captures"
    )
    parser.add_argument("--data_dir", default=default_data_dir)
    parser.add_argument(
        "--extrinsics_json", required=True, help="JSON with depth_to_rgb extrinsics"
    )
    parser.add_argument(
        "--icp", action="store_true", help="Enable ICP alignment between frames"
    )


def _add_aggregate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data_dir", default="data/1920x0180_5_cloud")
    parser.add_argument("--extrinsics_txt", default="realsense_extrinsics.json")
    parser.add_argument(
        "--charuco_xml", default="calibration/results1980/charuco_cam.xml"
    )
    parser.add_argument(
        "--handeye_txt", default="data/5x5_100_15imgs_top1/results3/handeye_ANDREFF.txt"
    )
    parser.add_argument(
        "--icp", action="store_true", help="Enable ICP alignment between frames"
    )


def _run_aggregate(args: argparse.Namespace) -> None:
    logger = Logger.get_logger("cloud.pipeline")
    Config.load()
    data_dir = "captures"
    extrinsics_txt = "realsense_extrinsics.json"
    charuco_xml = "data/old/results1980/charuco_cam.xml"
    handeye_txt = "calib/results3/handeye_ANDREFF.txt"

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
