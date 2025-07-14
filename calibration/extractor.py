"""Depth helpers converting image pixels to 3D coordinates."""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np


def load_extrinsics(json_path: Path, from_key: str, to_key: str):
    """Load extrinsics from JSON: returns (R, t) for from_key_to_key."""
    with open(json_path, "r") as f:
        data = json.load(f)
    # example: from_key = 'depth', to_key = 'rgb', key = 'depth_to_rgb'
    key = f"{from_key}_to_{to_key}"
    R = np.array(data[key]["R"])
    t = np.array(data[key]["t"])
    return R, t


def load_depth(image_path: Path) -> np.ndarray:
    """Return depth array matching ``image_path``."""
    depth_path = image_path.parent / f"{image_path.stem.replace('_rgb', '')}_depth.npy"
    return np.load(depth_path)


def pixel_to_camera(pixel: np.ndarray, depth: float, K: np.ndarray) -> np.ndarray:
    """Project ``pixel`` with a given ``depth`` to 3-D camera coordinates."""
    x = (pixel[0] - K[0, 2]) * depth / K[0, 0]
    y = (pixel[1] - K[1, 2]) * depth / K[1, 1]
    return np.array([x, y, depth], dtype=np.float64)


def pixel_to_camera_depth_ir(
    pixel: np.ndarray, depth: float, K_depth: np.ndarray
) -> np.ndarray:
    """Project depth pixel with depth value to depth camera coordinates."""
    x = (pixel[0] - K_depth[0, 2]) * depth / K_depth[0, 0]
    y = (pixel[1] - K_depth[1, 2]) * depth / K_depth[1, 1]
    return np.array([x, y, depth], dtype=np.float64)


def rgb_to_depth_pixel(x_rgb, y_rgb, K_rgb, K_depth, R_depth2rgb, t_depth2rgb):
    pt_rgb = np.array([x_rgb, y_rgb, 1.0])
    pt_rgb_norm = np.linalg.inv(K_rgb) @ pt_rgb
    ray_rgb = pt_rgb_norm
    ray_depth = R_depth2rgb.T @ (ray_rgb - t_depth2rgb)
    pt_depth = K_depth @ (ray_depth / ray_depth[2])
    return pt_depth[0], pt_depth[1]


def board_center_from_depth(
    corners: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """Return the board center in camera coordinates using the depth image."""
    center = corners.reshape(-1, 2).mean(axis=0)
    z = float(depth[int(round(center[1])), int(round(center[0]))])
    return pixel_to_camera(center, z, K)


def board_points_from_depth_with_extrinsics(
    corners_rgb, depth_map, K_depth, K_rgb, R_depth2rgb, t_depth2rgb, depth_scale
):
    pts_3d_rgb = []
    for x_rgb, y_rgb in corners_rgb.reshape(-1, 2):
        x_depth, y_depth = rgb_to_depth_pixel(
            x_rgb, y_rgb, K_rgb, K_depth, R_depth2rgb, t_depth2rgb
        )
        ix, iy = int(round(x_depth)), int(round(y_depth))
        if iy < 0 or iy >= depth_map.shape[0] or ix < 0 or ix >= depth_map.shape[1]:
            return None
        z = float(depth_map[iy, ix]) * depth_scale
        if not np.isfinite(z) or z <= 0:
            return None
        pt_depth = pixel_to_camera_depth_ir(np.array([x_depth, y_depth]), z, K_depth)
        pt_rgb = R_depth2rgb @ pt_depth + t_depth2rgb
        pts_3d_rgb.append(pt_rgb)
    return np.asarray(pts_3d_rgb)


def board_points_from_depth(
    corners: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
) -> np.ndarray | None:
    """Return 3-D coordinates of detected corners using the depth map.

    Returns ``None`` if any corner falls outside the depth image or has an
    invalid/non-positive depth value.
    """
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
