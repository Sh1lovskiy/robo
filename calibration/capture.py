"""Capture RGB-D frames and robot poses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from tqdm import tqdm

from robot.controller import RobotController
from utils.error_tracker import ErrorTracker
from utils.logger import Logger

from .utils import ImagePair, confirm, save_image_pair

log = Logger.get_logger("calibrate.capture")


def capture_dataset(
    out_dir: Path,
    *,
    max_frames: int = 20,
    interactive: bool = False,
) -> List[ImagePair]:
    """Capture images and robot poses into ``out_dir``.

    Parameters
    ----------
    out_dir:
        Destination directory where images and ``poses.json`` are stored.
    max_frames:
        Maximum number of frames to capture.
    interactive:
        If ``True`` confirm each capture via keyboard.

    Returns
    -------
    list of :class:`ImagePair`
        Saved image pairs.
    """

    from vision.camera.realsense_d415 import RealSenseD415

    cam = RealSenseD415()
    cam.start()
    ErrorTracker.register_cleanup(cam.stop)

    robot = RobotController()
    robot.connect()
    ErrorTracker.register_cleanup(robot.shutdown)

    pairs: List[ImagePair] = []
    poses: dict[str, dict[str, float]] = {}

    with tqdm(total=max_frames, desc="Capture") as pbar:
        for idx in range(max_frames):
            if interactive and not confirm(f"Capture frame {idx}?"):
                break
            rgb, depth = cam.get_frames()
            if rgb is None or depth is None:
                log.warning("Failed to capture frame %d", idx)
                continue
            pair = save_image_pair(rgb, depth, out_dir, idx)
            pairs.append(pair)
            tcp = robot.get_tcp_pose()
            if tcp is None:
                log.warning("No robot pose for frame %d", idx)
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
    log.info("Saved %d poses", len(poses))

    return pairs
