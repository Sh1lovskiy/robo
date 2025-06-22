"""CLI entry points for camera and point cloud utilities."""
from __future__ import annotations

import argparse

import numpy as np
import open3d as o3d

from utils.logger import Logger
from utils.config import Config
from vision.camera_utils import IntrinsicsPrinter, DepthChecker
from vision.realsense import RealSenseCamera
from vision.cloud.generator import PointCloudGenerator
from vision.cloud.pipeline import CloudPipeline
from vision.transform import TransformUtils

# ---------------------------------------------------------------------------
# Basic point cloud helpers
# ---------------------------------------------------------------------------

def capture_cloud(output: str) -> None:
    logger = Logger.get_logger("vision.tools.capture")
    cam = RealSenseCamera()
    cam.start()
    color, depth = cam.get_frames()
    intr = cam.get_intrinsics()
    points, colors = PointCloudGenerator.depth_to_cloud(depth, intr, rgb=color)
    PointCloudGenerator.save_ply(output, points, colors)
    logger.info(f"Saved cloud to {output}")
    cam.stop()


def transform_cloud(input_ply: str, calib_npz: str, output: str) -> None:
    logger = Logger.get_logger("vision.tools.transform")
    cloud_gen = PointCloudGenerator()
    points, colors = cloud_gen.load_ply(input_ply)
    calib = np.load(calib_npz)
    R, t = calib["R"], calib["t"]
    points_t = TransformUtils.apply_transform(points, R, t)
    cloud_gen.save_ply(output, points_t, colors)
    logger.info(f"Transformed cloud saved to {output}")


def view_cloud(input_ply: str) -> None:
    logger = Logger.get_logger("vision.tools.view")
    pcd = o3d.io.read_point_cloud(input_ply)
    logger.info(f"Loaded point cloud {input_ply}")
    o3d.visualization.draw_geometries([pcd])


# CLI helpers ---------------------------------------------------------------

def main_capture() -> None:
    parser = argparse.ArgumentParser(description="Capture point cloud")
    Config.load()
    out_dir = Config.get("cloud.output_dir", "clouds")
    parser.add_argument("--output", default=f"{out_dir}/cloud.ply")
    args = parser.parse_args()
    capture_cloud(args.output)


def main_transform() -> None:
    parser = argparse.ArgumentParser(description="Transform point cloud")
    parser.add_argument("--input", required=True)
    parser.add_argument("--calib", required=True)
    Config.load()
    out_dir = Config.get("cloud.output_dir", "clouds")
    parser.add_argument("--output", default=f"{out_dir}/cloud_world.ply")
    args = parser.parse_args()
    transform_cloud(args.input, args.calib, args.output)


def main_view() -> None:
    parser = argparse.ArgumentParser(description="View point cloud")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    view_cloud(args.input)


def main_intrinsics() -> None:
    IntrinsicsPrinter().run()


def main_depth() -> None:
    DepthChecker().run()


def main_pipeline() -> None:
    parser = argparse.ArgumentParser(
        description="Point cloud denoise/cluster/trajectory pipeline"
    )
    parser.add_argument(
        "--input",
        default="captures/cloud_aggregated.ply",
        help="Input PLY file",
    )
    args = parser.parse_args()
    CloudPipeline().run(args.input)
