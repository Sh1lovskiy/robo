"""Depth helpers converting image pixels to 3D coordinates."""

from __future__ import annotations

from pathlib import Path
import numpy as np


def load_depth(image_path: Path) -> np.ndarray:
    """Return depth array matching ``image_path``."""
    depth_path = image_path.parent / f"{image_path.stem.replace('_rgb', '')}_depth.npy"
    return np.load(depth_path)


def pixel_to_camera(pixel: np.ndarray, depth: float, K: np.ndarray) -> np.ndarray:
    """Project ``pixel`` with ``depth`` to the camera frame."""
    x = (pixel[0] - K[0, 2]) * depth / K[0, 0]
    y = (pixel[1] - K[1, 2]) * depth / K[1, 1]
    return np.array([x, y, depth], dtype=np.float64)


def board_center_from_depth(
    corners: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """Return board center point in the camera frame using ``depth``."""
    center = corners.reshape(-1, 2).mean(axis=0)
    z = float(depth[int(round(center[1])), int(round(center[0]))])
    return pixel_to_camera(center, z, K)


def board_points_from_depth(
    corners: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
) -> np.ndarray | None:
    """Return 3D points of each detected corner using ``depth``."""
    pts_3d = []
    for x, y in corners.reshape(-1, 2):
        ix, iy = int(round(x)), int(round(y))
        if iy < 0 or iy >= depth.shape[0] or ix < 0 or ix >= depth.shape[1]:
            return None
        z = float(depth[iy, ix])
        if not np.isfinite(z) or z <= 0:
            return None
        pts_3d.append(pixel_to_camera(np.array([x, y]), z, K))
    return np.asarray(pts_3d)
