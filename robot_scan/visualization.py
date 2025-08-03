"""Visualization utilities using Open3D."""

from __future__ import annotations

import open3d as o3d

from utils.logger import Logger
from .transform import pose_to_matrix

logger = Logger.get_logger("robot_scan.visualization")


def visualize_tcp_target(pose) -> None:
    """Render a coordinate frame at the target TCP pose."""
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    mesh.transform(pose_to_matrix(pose))
    logger.info(f"Visualizing pose {pose}")
    o3d.visualization.draw_geometries([mesh])
