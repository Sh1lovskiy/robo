"""Entry point for the robot scanning pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from utils.error_tracker import ErrorTracker
from utils.logger import Logger
from utils.settings import CAM_PARAMS_PATH

from .capture import capture_rgbd
from .graph import build_graph
from .motion import connect_robot, move_l
from .preprocess import (
    compute_plane_axes,
    downsample_cloud,
    segment_plane,
)
from .save import create_run_dir, save_cloud, save_metadata, save_rgbd
from .skeleton import skeletonize_plane
from .transform import compute_tcp_pose, load_handeye, transform_cloud

logger = Logger.get_logger("robot_scan.main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robot scanning pipeline")
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--handeye", type=str, required=True, help="path to handeye matrix")
    parser.add_argument(
        "--intrinsics",
        type=str,
        default=str(CAM_PARAMS_PATH),
        help="Camera intrinsics",
    )
    parser.add_argument("--mode", choices=["auto", "interactive"], default="interactive")
    parser.add_argument("--robot-ip", type=str, default=None)
    parser.add_argument("--target-strategy", type=str, default="first")
    return parser.parse_args()


def main() -> None:
    ErrorTracker.install_signal_handlers()
    np.set_printoptions(precision=6, suppress=True)
    args = parse_args()
    handeye = load_handeye(args.handeye)
    rpc = None
    tcp_pose = np.zeros(6)
    if args.robot_ip:
        rpc = connect_robot(args.robot_ip)
        code, pose = rpc.GetActualTCPPose(0)
        if code == 0:
            tcp_pose = np.array(pose)
        logger.info("Robot TCP pose: %s", tcp_pose)
    frame = capture_rgbd()
    cloud = frame.cloud
    cloud = downsample_cloud(cloud)
    cloud_base = transform_cloud(cloud, handeye, tcp_pose)
    plane, normal, _ = segment_plane(cloud_base)
    main_axis, plane_normal = compute_plane_axes(np.asarray(plane.points))
    nodes, branches = skeletonize_plane(plane)
    graph = build_graph(nodes, branches)
    if graph.number_of_nodes() == 0:
        logger.error("No graph nodes detected")
        return
    target_idx = list(graph.nodes)[0]
    target_point = graph.nodes[target_idx]["pos"]
    pose = compute_tcp_pose(target_point, main_axis, plane_normal)
    if rpc:
        move_l(rpc, pose)
    if args.save_data:
        run_dir = create_run_dir()
        save_rgbd(run_dir, 0, frame.color, frame.depth)
        save_cloud(run_dir, 0, cloud_base)
        meta = {
            "tcp_pose": tcp_pose,
            "handeye": handeye,
            "target": target_point,
        }
        if args.intrinsics and Path(args.intrinsics).exists():
            meta["intrinsics"] = Path(args.intrinsics).read_text()
        save_metadata(run_dir, meta)


if __name__ == "__main__":
    main()
