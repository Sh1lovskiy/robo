"""Data saving utilities."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import open3d as o3d

from utils.logger import Logger

logger = Logger.get_logger("robot_scan.save")

BASE_DIR = Path(".data_clouds")


def create_run_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = BASE_DIR / ts
    path.mkdir(parents=True, exist_ok=True)
    logger.info("Created data directory %s", path)
    return path


def save_rgbd(path: Path, idx: int, color: np.ndarray, depth: np.ndarray) -> None:
    import cv2

    rgb_path = path / f"{idx:03d}_rgb.png"
    depth_path = path / f"{idx:03d}_depth.npy"
    cv2.imwrite(str(rgb_path), color)
    np.save(depth_path, depth)
    logger.info("Saved RGB-D images for frame %03d", idx)


def save_cloud(path: Path, idx: int, cloud: o3d.geometry.PointCloud) -> None:
    pcd_path = path / f"{idx:03d}_cloud.pcd"
    o3d.io.write_point_cloud(str(pcd_path), cloud)
    logger.info("Saved point cloud %s", pcd_path)


def save_metadata(path: Path, data: Dict[str, object]) -> None:
    meta_path = path / "metadata.txt"
    with open(meta_path, "w", encoding="utf-8") as f:
        for key, val in data.items():
            if isinstance(val, np.ndarray):
                val = np.array2string(val, precision=6, suppress_small=True)
            f.write(f"{key}: {val}\n")
    logger.info("Saved metadata to %s", meta_path)
