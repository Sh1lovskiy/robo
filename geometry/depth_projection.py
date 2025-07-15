"""Depth/RGB coordinate projection helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import json

import numpy as np
import cv2

__all__ = [
    "load_extrinsics",
    "pixel_to_camera",
    "rgb_to_depth_pixel",
    "map_rgb_corners_to_depth",
    "estimate_board_points_3d",
]


def load_extrinsics(
    json_path: Path, from_key: str, to_key: str
) -> Tuple[np.ndarray, np.ndarray]:

    with open(json_path, "r") as f:
        data = json.load(f)
    key = f"{from_key}_to_{to_key}"
    R = np.asarray(data[key]["R"], dtype=np.float64)
    t = np.asarray(data[key]["t"], dtype=np.float64)
    return R, t


def pixel_to_camera(pixel: np.ndarray, depth: float, K: np.ndarray) -> np.ndarray:
    x = np.multiply((pixel[0] - K[0, 2]), depth / K[0, 0])
    y = np.multiply((pixel[1] - K[1, 2]), depth / K[1, 1])
    return np.array([x, y, depth], dtype=np.float64)


def rgb_to_depth_pixel(
    x_rgb: float,
    y_rgb: float,
    K_rgb: np.ndarray,
    K_depth: np.ndarray,
    R_depth2rgb: np.ndarray,
    t_depth2rgb: np.ndarray,
) -> Tuple[float, float]:
    pt_rgb = np.array([x_rgb, y_rgb, 1.0])
    ray_rgb = np.linalg.inv(K_rgb) @ pt_rgb
    ray_depth = R_depth2rgb.T @ (ray_rgb - t_depth2rgb)
    pt_depth = K_depth @ (ray_depth / ray_depth[2])
    return float(pt_depth[0]), float(pt_depth[1])


def map_rgb_corners_to_depth(
    corners_rgb: np.ndarray,
    depth_map: np.ndarray,
    K_rgb: np.ndarray,
    K_depth: np.ndarray,
    R_depth2rgb: np.ndarray,
    t_depth2rgb: np.ndarray,
    depth_scale: float = 0.001,
) -> Optional[np.ndarray]:
    pts_rgb = []
    for x_rgb, y_rgb in corners_rgb.reshape(-1, 2):
        x_d = x_rgb + 0.01515267
        y_d = y_rgb
        ix, iy = int(round(x_d)), int(round(y_d))
        if iy < 0 or iy >= depth_map.shape[0] or ix < 0 or ix >= depth_map.shape[1]:
            return None
        z = float(depth_map[iy, ix]) * depth_scale
        if not np.isfinite(z) or z <= 0:
            return None
        pt_depth = pixel_to_camera(np.array([x_d, y_d]), z, K_depth)
        pt_rgb = R_depth2rgb @ pt_depth + t_depth2rgb
        pts_rgb.append(pt_rgb)
    return np.asarray(pts_rgb)


def estimate_board_points_3d(
    charuco_corners: np.ndarray,
    depth_map: np.ndarray,
    object_points: np.ndarray,
    K_rgb: np.ndarray,
    dist_rgb: np.ndarray,
    K_depth: np.ndarray,
    R_depth2rgb: np.ndarray,
    t_depth2rgb: np.ndarray,
    depth_scale: float = 0.001,
) -> Optional[np.ndarray]:

    pts_rgb = map_rgb_corners_to_depth(
        charuco_corners,
        depth_map,
        K_rgb,
        K_depth,
        R_depth2rgb,
        t_depth2rgb,
        depth_scale,
    )
    if pts_rgb is not None:
        return pts_rgb

    ok, rvec, tvec = cv2.solvePnP(object_points, charuco_corners, K_rgb, dist_rgb)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return (R @ object_points.T).T + tvec.reshape(3)
