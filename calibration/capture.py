"""Capture RGB-D frames and robot poses with robot motion."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List

import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from robot.controller import RobotController
from utils.error_tracker import ErrorTracker
from utils.logger import Logger
from robot_scan.capture import capture_rgbd
from .utils import ImagePair, save_image_pair

log = Logger.get_logger("calibrate.capture")


def generate_grid(
    workspace: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    step: float = 0.05,
    rx_base=180.0,
    ry_base=0.0,
    rz_base=180.0,
    rx_range=15.0,
    ry_range=15.0,
    rz_range=15.0,
    batch_size: int = 128,
    seed: int = 42,
) -> list[list[float]]:
    """Generate a list of [x, y, z, rx, ry, rz] grid poses."""
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = workspace
    x_vals = np.arange(x_min, x_max, step)
    y_vals = np.arange(y_min, y_max, step)
    z_vals = np.arange(z_min, z_max, step)

    rng = np.random.default_rng(seed)
    poses = []

    with tqdm(
        total=len(x_vals) * len(y_vals) * len(z_vals), desc="Generating robot poses"
    ) as pbar:
        for x in x_vals:
            for y in y_vals:
                for z in z_vals:
                    found = False
                    while not found:
                        rx = rx_base + rng.uniform(-rx_range, rx_range, size=batch_size)
                        ry = ry_base + rng.uniform(-ry_range, ry_range, size=batch_size)
                        rz = rz_base + rng.uniform(-rz_range, rz_range, size=batch_size)

                        angles = np.stack([rx, ry, rz], axis=-1)
                        rots = R.from_euler("xyz", angles, degrees=True)
                        Rmats = rots.as_matrix()

                        dets = np.linalg.det(Rmats)
                        ortho = np.linalg.norm(
                            np.matmul(np.transpose(Rmats, (0, 2, 1)), Rmats)
                            - np.eye(3),
                            axis=(1, 2),
                        )
                        traces = np.trace(Rmats, axis1=1, axis2=2)
                        valid = (dets > 0.99) & (ortho < 1e-6) & (traces > -1.0)
                        idx = np.flatnonzero(valid)

                        if idx.size > 0:
                            i = idx[0]
                            poses.append(
                                [
                                    float(x),
                                    float(y),
                                    float(z),
                                    float(rx[i]),
                                    float(ry[i]),
                                    float(rz[i]),
                                ]
                            )
                            found = True
                    pbar.update(1)

    log.info(f"Generated {len(poses)} poses in grid")
    return poses


def capture_dataset(
    out_dir: Path,
    *,
    workspace: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    grid_step: float = 0.05,
    rx_base: float = 180.0,
    ry_base: float = 0.0,
    rz_base: float = 180.0,
    interactive: bool = False,
    max_frames: int | None = None,
) -> List[ImagePair]:
    """Move robot to grid poses, capture RGB-D frames and save poses/images."""

    robot = RobotController()
    robot.connect(safety_check=False)
    ErrorTracker.register_cleanup(robot.shutdown)

    grid = generate_grid(
        workspace=workspace,
        step=grid_step,
        rx_base=rx_base,
        ry_base=ry_base,
        rz_base=rz_base,
    )
    if max_frames:
        grid = grid[:max_frames]

    pairs: List[ImagePair] = []
    poses: dict[str, dict[str, float]] = {}

    with tqdm(total=len(grid), desc="Capture") as pbar:
        for idx, pose in enumerate(grid):
            if not robot.move_linear(pose):
                log.error(f"Failed to move to pose {idx}")
                continue

            time.sleep(1.0)  # Allow robot to settle
            frame = capture_rgbd()
            pair = save_image_pair(frame.color, frame.depth, out_dir, idx)
            pairs.append(pair)

            tcp = robot.get_tcp_pose()
            if tcp is None:
                log.warning(f"No robot pose for frame {idx}")
            else:
                poses[f"{idx:03d}"] = {
                    "x": round(float(tcp[0]), 6),
                    "y": round(float(tcp[1]), 6),
                    "z": round(float(tcp[2]), 6),
                    "Rx": round(float(tcp[3]), 6),
                    "Ry": round(float(tcp[4]), 6),
                    "Rz": round(float(tcp[5]), 6),
                }
            pbar.update(1)

    with open(out_dir / "poses.json", "w", encoding="utf-8") as f:
        json.dump(poses, f, indent=2)
    log.info(f"Saved {len(poses)} poses")

    return pairs
