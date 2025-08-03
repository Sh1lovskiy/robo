# utils/cloud_utils.py — Point cloud helper functions
"""Utility helpers for cloud aggregation and calibration files."""

from __future__ import annotations

import glob
import json
import os
from typing import Tuple
import numpy as np

from utils.logger import Logger, LoggerType
from utils.settings import DEPTH_SCALE, IMAGE_EXT, DEPTH_EXT


def load_handeye_txt(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load hand-eye calibration from a simple text format."""
    with open(path, "r") as f:
        lines = f.readlines()
    R: list[list[float]] = []
    t: np.ndarray = np.empty(3)
    for line in lines:
        if line.startswith("R") or line.startswith("t"):
            continue
        values = [float(x) for x in line.strip().split()]
        if len(values) == 3:
            if len(R) < 3:
                R.append(values)
            else:
                t = np.array(values)
    return np.array(R), t


def load_depth(depth_path: str) -> np.ndarray:
    """Load depth map converting integer arrays to meters."""
    depth = np.load(depth_path)
    if np.issubdtype(depth.dtype, np.integer):
        depth = depth.astype(np.uint16)
    return depth


def get_image_pairs(data_dir: str) -> list[tuple[str, str]]:
    """Return matched RGB/depth image file pairs in ``data_dir``."""
    rgb_list = sorted(
        f
        for f in glob.glob(os.path.join(data_dir, f"*{IMAGE_EXT}"))
        if f.endswith(IMAGE_EXT)
    )
    depth_list = sorted(
        f
        for f in glob.glob(os.path.join(data_dir, f"*{DEPTH_EXT}"))
        if f.endswith(DEPTH_EXT)
    )
    if len(rgb_list) != len(depth_list):
        raise RuntimeError("RGB and depth image count mismatch")
    return list(zip(rgb_list, depth_list))


def load_extrinsics_json(
    json_path: str, logger: LoggerType | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Read depth-to-RGB extrinsics from a JSON file."""
    logger = logger or Logger.get_logger("utils.cloud")
    with open(json_path, "r") as f:
        data = json.load(f)
    ext = data.get("depth_to_rgb", data)
    if "rotation" in ext:
        R = np.array(ext["rotation"])
    elif "R" in ext:
        R = np.array(ext["R"])
    else:
        raise KeyError("rotation matrix not found in extrinsics JSON")

    if "translation" in ext:
        t = np.array(ext["translation"])
    elif "t" in ext:
        t = np.array(ext["t"])
    else:
        raise KeyError("translation vector not found in extrinsics JSON")
    logger.info(f"Extrinsics loaded from {json_path}")
    return R, t
