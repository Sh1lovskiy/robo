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
from robot_scan.capture import RealSenseGrabber, capture_rgbd
from .utils import ImagePair, save_image_pair

log = Logger.get_logger("calibrate.capture")


def generate_grid(
    workspace: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    step: float = 0.05,
    rx_base: float = 180.0,
    ry_base: float = 0.0,
    rz_base: float = 180.0,
    orient_mode: str = "fixed",
    rx_range: float = 15.0,
    ry_range: float = 15.0,
    rz_range: float = 15.0,
    seed: int = 42,
    snake: bool = True,
) -> list[list[float]]:
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = workspace
    x_vals = np.arange(x_min, x_max + 1e-9, step)
    y_vals = np.arange(y_min, y_max + 1e-9, step)
    z_vals = np.arange(z_min, z_max + 1e-9, step)

    rng = np.random.default_rng(seed)
    poses: list[list[float]] = []

    for zi, z in enumerate(z_vals):
        y_iter = y_vals[::-1] if (snake and (zi % 2 == 1)) else y_vals
        for yi, y in enumerate(y_iter):
            x_iter = x_vals[::-1] if (snake and (yi % 2 == 1)) else x_vals
            for x in x_iter:
                if orient_mode == "random":
                    rx = float(rx_base + rng.uniform(-rx_range, rx_range))
                    ry = float(ry_base + rng.uniform(-ry_range, ry_range))
                    rz = float(rz_base + rng.uniform(-rz_range, rz_range))
                else:
                    rx, ry, rz = float(rx_base), float(ry_base), float(rz_base)
                poses.append([float(x), float(y), float(z), rx, ry, rz])

    log.info(f"Generated {len(poses)} poses in grid (step={step}, snake={snake})")
    return poses


def capture_dataset(
    out_dir: Path,
    *,
    workspace: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    grid_step: float = 0.05,
    rx_base: float = 180.0,
    ry_base: float = 0.0,
    rz_base: float = 180.0,
    orient_mode: str = "fixed",
    interactive: bool = False,
    max_frames: int | None = None,
    settle_sec: float = 0.8,
) -> List[ImagePair]:
    out_dir.mkdir(parents=True, exist_ok=True)

    robot = RobotController()
    robot.connect(safety_check=False)
    ErrorTracker.register_cleanup(robot.shutdown)

    grid = generate_grid(
        workspace=workspace,
        step=grid_step,
        rx_base=rx_base,
        ry_base=ry_base,
        rz_base=rz_base,
        orient_mode=orient_mode,
    )
    if max_frames is not None:
        grid = grid[:max_frames]

    pairs: List[ImagePair] = []
    poses: dict[str, dict[str, float]] = {}

    with RealSenseGrabber() as cam, tqdm(total=len(grid), desc="Capture") as pbar:
        for idx, pose in enumerate(grid):
            if interactive:
                log.info(f"[{idx}] Visualizing pose (interactive mode)")
            ok = robot.move_linear(pose)
            if not ok:
                log.error(f"[{idx}] Failed to move to pose {pose}")
                pbar.update(1)
                continue

            time.sleep(settle_sec)  # стабилизация
            frame = cam.grab()
            pair = save_image_pair(frame.color, frame.depth, out_dir, idx)
            pairs.append(pair)

            tcp = robot.get_tcp_pose()
            if tcp is None:
                log.warning(f"[{idx}] No TCP pose reported")
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
        json.dump(poses, f, indent=2, ensure_ascii=False)
    log.info(f"Saved {len(poses)} poses to {out_dir/'poses.json'}")
    return pairs
