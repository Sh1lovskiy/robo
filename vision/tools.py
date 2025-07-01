"""CLI entry points for camera and point cloud utilities.

TODO: add CI badges for build and coverage.
"""

from __future__ import annotations

import argparse

import numpy as np
import open3d as o3d

from utils.cli import Command, CommandDispatcher
from utils.config import Config
from utils.error_tracker import CameraConnectionError
from utils.logger import Logger
from vision.camera_utils import DepthChecker, IntrinsicsPrinter
from vision.cloud.generator import PointCloudGenerator
from vision.cloud.analyzer import CloudAnalyzer
from vision.transform import TransformUtils

# ---------------------------------------------------------------------------
# Basic point cloud helpers
# ---------------------------------------------------------------------------


def capture_cloud(output: str) -> None:
    logger = Logger.get_logger("vision.tools.capture")
    cam = RealSenseCamera(RealSenseConfig())
    try:
        cam.start()
    except CameraConnectionError as e:
        logger.error(f"Camera connection failed: {e}")
        return

    try:
        color, depth = cam.get_frames()
        intr = cam.get_intrinsics()
        points, colors = PointCloudGenerator.depth_to_cloud(depth, intr, rgb=color)
        PointCloudGenerator.save_ply(output, points, colors)
        logger.info(f"Saved cloud to {output}")
    finally:
        cam.stop()


def transform_cloud(input_ply: str, calib_npz: str, output: str) -> None:
    logger = Logger.get_logger("vision.tools.transform")
    cloud_gen = PointCloudGenerator()
    try:
        points, colors = cloud_gen.load_ply(input_ply)
        calib = np.load(calib_npz)
    except Exception as e:
        logger.error(f"Failed to load input files: {e}")
        return

    R, t = calib["R"], calib["t"]
    points_t = TransformUtils.apply_transform(points, R, t)
    cloud_gen.save_ply(output, points_t, colors)
    logger.info(f"Transformed cloud saved to {output}")


def view_cloud(input_ply: str) -> None:
    logger = Logger.get_logger("vision.tools.view")
    try:
        pcd = o3d.io.read_point_cloud(input_ply)
    except Exception as e:
        logger.error(f"Failed to load point cloud {input_ply}: {e}")
        return

    logger.info(f"Loaded point cloud {input_ply}")
    o3d.visualization.draw_geometries([pcd])


# CLI helpers


def _add_capture_args(parser: argparse.ArgumentParser) -> None:
    Config.load()
    out_dir = Config.get("cloud.output_dir", "clouds")
    parser.add_argument(
        "--output",
        default=f"{out_dir}/cloud.ply",
        help="Output PLY file path",
    )


def _run_capture(args: argparse.Namespace) -> None:
    capture_cloud(args.output)


def main_capture() -> None:
    parser = argparse.ArgumentParser(description="Capture point cloud")
    _add_capture_args(parser)
    args = parser.parse_args()
    _run_capture(args)


def _add_transform_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="Input PLY file")
    parser.add_argument("--calib", required=True, help="Calibration npz file")
    Config.load()
    out_dir = Config.get("cloud.output_dir", "clouds")
    parser.add_argument(
        "--output",
        default=f"{out_dir}/cloud_world.ply",
        help="Transformed output file",
    )


def _run_transform(args: argparse.Namespace) -> None:
    transform_cloud(args.input, args.calib, args.output)


def main_transform() -> None:
    parser = argparse.ArgumentParser(description="Transform point cloud")
    _add_transform_args(parser)
    args = parser.parse_args()
    _run_transform(args)


def _add_view_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="PLY file to view")


def _run_view(args: argparse.Namespace) -> None:
    view_cloud(args.input)


def main_view() -> None:
    parser = argparse.ArgumentParser(description="View point cloud")
    _add_view_args(parser)
    args = parser.parse_args()
    _run_view(args)


def _run_intrinsics(args: argparse.Namespace) -> None:
    IntrinsicsPrinter().run()


def main_intrinsics() -> None:
    _run_intrinsics(argparse.Namespace())


def _run_depth(args: argparse.Namespace) -> None:
    DepthChecker().run()


def main_depth() -> None:
    _run_depth(argparse.Namespace())


def _add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        default="captures/cloud_aggregated.ply",
        help="Input PLY file",
    )


def _run_pipeline(args: argparse.Namespace) -> None:
    CloudAnalyzer().run(args.input)


def main_pipeline() -> None:
    parser = argparse.ArgumentParser(
        description="Point cloud denoise/cluster/trajectory pipeline"
    )
    _add_pipeline_args(parser)
    args = parser.parse_args()
    _run_pipeline(args)


def create_cli() -> CommandDispatcher:
    commands = [
        Command("capture", _run_capture, _add_capture_args, "Capture point cloud"),
        Command(
            "transform", _run_transform, _add_transform_args, "Transform point cloud"
        ),
        Command("view", _run_view, _add_view_args, "View point cloud"),
        Command("intrinsics", _run_intrinsics, help="Print camera intrinsics"),
        Command("depth", _run_depth, help="Check depth frames"),
        Command("pipeline", _run_pipeline, _add_pipeline_args, "Run cloud pipeline"),
    ]
    return CommandDispatcher("Vision utilities", commands)


def main() -> None:
    logger = Logger.get_logger("vision.tools")
    create_cli().run(logger=logger)


if __name__ == "__main__":
    main()
