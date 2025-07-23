from __future__ import annotations

import time
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from robot.controller import RobotController
from utils.logger import Logger
from utils.settings import (
    grid_calib,
    paths,
    IMAGE_EXT,
    DEPTH_EXT,
    camera as cam_cfg,
    robot as robot_cfg,
)


logger = Logger.get_logger("capture")

# CONSTANTS
OUT_DIR = paths.CAPTURES_DIR
POSE_DIR = paths.CAPTURES_EXTR_DIR
GRID_LIMITS = grid_calib.workspace_limits
STEP = grid_calib.grid_step
TOOL_ORIENT = grid_calib.tool_orientation


def camera_start(pipeline, config):
    pipeline.start(config)
    logger.info("Camera started")


def camera_stop(pipeline):
    pipeline.stop()
    logger.info("Camera stopped")


def camera_get_frames(pipeline) -> Tuple[np.ndarray, np.ndarray]:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        logger.warning("Failed to capture frames")
        return None, None
    depth_scale = np.array(cam_cfg.depth_scale, np.float32)
    color_img = np.asanyarray(color_frame.get_data(), np.uint8)
    depth_img = (
        np.asanyarray(depth_frame.get_data(), np.uint16).astype(float) * depth_scale
    )
    return color_img, depth_img


def wrap_angle_deg(angle):
    return (angle + 180) % 360 - 180


def generate_grid(
    rx_base=180.0,
    ry_base=0.0,
    rz_base=180.0,
    rx_range=15.0,
    ry_range=15.0,
    rz_range=15.0,
    randomize=True,
    seed=42,
    batch_size=128,
) -> list[list[float]]:
    """
    For each grid point, randomly search in batches for a valid orientation.
    All angles in degrees, positions in millimeters.
    Returns: list of [x, y, z, rx, ry, rz]
    """
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = grid_calib.workspace_limits
    step = grid_calib.grid_step

    rng = np.random.default_rng(seed)
    poses = []
    filtered_total = 0

    x_vals = np.arange(x_min, x_max, step)
    y_vals = np.arange(y_min, y_max, step)
    z_vals = np.arange(z_min, z_max, step)

    total_points = len(x_vals) * len(y_vals) * len(z_vals)

    with tqdm(total=total_points, desc="Generating grid poses") as pbar:
        for x in x_vals:
            for y in y_vals:
                for z in z_vals:
                    found = False
                    filtered = 0
                    while not found:
                        if randomize:
                            rx = rx_base + rng.uniform(
                                -rx_range, rx_range, size=batch_size
                            )
                            ry = ry_base + rng.uniform(
                                -ry_range, ry_range, size=batch_size
                            )
                            rz = rz_base + rng.uniform(
                                -rz_range, rz_range, size=batch_size
                            )
                        else:
                            rx = np.full(batch_size, rx_base)
                            ry = np.full(batch_size, ry_base)
                            rz = np.full(batch_size, rz_base)

                        angles = np.stack([rx, ry, rz], axis=-1)
                        rots = R.from_euler("xyz", angles, degrees=True)
                        Rmats = rots.as_matrix()

                        # checks
                        dets = np.linalg.det(Rmats)
                        ortho = np.linalg.norm(
                            np.matmul(np.transpose(Rmats, (0, 2, 1)), Rmats)
                            - np.eye(3),
                            axis=(1, 2),
                        )
                        traces = np.trace(Rmats, axis1=1, axis2=2)

                        # mask
                        valid = (dets > 0.99) & (ortho < 1e-6) & (traces > -1.0)
                        idx = np.flatnonzero(valid)
                        filtered += batch_size - valid.sum()

                        if idx.size > 0:
                            i = idx[0]
                            poses.append(
                                [
                                    float(x),
                                    float(y),
                                    float(z),
                                    float(wrap_angle_deg(rx[i])),
                                    float(wrap_angle_deg(ry[i])),
                                    float(wrap_angle_deg(rz[i])),
                                ]
                            )
                            found = True
                            filtered_total += filtered
                    pbar.update(1)

    logger.info(
        f"Generated {len(poses)} grid poses\n"
        f"base=({rx_base}, {ry_base}, {rz_base}), range=({rx_range}, {ry_range}, {rz_range})"
    )
    return poses


def capture_image_pose(
    robot,
    idx: int,
    pose: List[float],
    pipeline,
    out_dir: Path,
    image_paths: List[Path],
    collected_poses: List[List[float]],
) -> None:
    if not robot.move_linear(pose):
        logger.error(f"Failed to move to pose {idx}")
        return
    # if not robot.wait_motion_done(timeout=3.0):
    #     logger.error(f"Motion not done at pose {idx}")
    #     return
    time.sleep(1.0)
    color, depth = camera_get_frames(pipeline)
    if color is None:
        logger.warning(f"No image captured at pose {idx}")
        return

    base = out_dir / f"frame_{idx:03d}"
    cv2.imwrite(str(base.with_suffix(IMAGE_EXT)), color)
    if depth is not None:
        np.save(str(base.with_suffix(DEPTH_EXT)), depth)
    image_paths.append(base.with_suffix(IMAGE_EXT))
    tcp_pose = robot.get_tcp_pose()
    collected_poses.append(tcp_pose)
    logger.info(f"Captured image and pose at index {idx}")


def save_poses(poses: List[List[float]], out_dir: Path) -> Path:
    file = out_dir / f"poses_{int(time.time())}.json"
    data = {str(i): {"tcp_coords": list(map(float, p))} for i, p in enumerate(poses)}
    with open(file, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved poses to {file}")
    return file


def main() -> None:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.depth,
        cam_cfg.depth_width,
        cam_cfg.depth_height,
        rs.format.z16,
        cam_cfg.fps,
    )
    config.enable_stream(
        rs.stream.color,
        cam_cfg.rgb_width,
        cam_cfg.rgb_height,
        rs.format.bgr8,
        cam_cfg.fps,
    )
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = generate_grid()

    image_paths: List[Path] = []
    collected_poses: List[List[float]] = []

    robot = RobotController(cfg=robot_cfg)
    robot.connect()
    camera_start(pipeline, config)
    try:
        for idx, pose in enumerate(grid):
            capture_image_pose(
                robot, idx, pose, pipeline, out_dir, image_paths, collected_poses
            )
    except Exception as e:
        logger.error(f"Exception in main loop: {e}")
        raise
    finally:
        camera_stop(pipeline)
        robot.shutdown()

    if collected_poses:
        save_poses(collected_poses, POSE_DIR)


if __name__ == "__main__":
    main()
