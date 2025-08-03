"""Data saving utilities."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import cv2
import networkx as nx
import numpy as np
import open3d as o3d
import plotly.graph_objects as go

from utils.error_tracker import ErrorTracker
from utils.logger import Logger
from utils.settings import (
    ARTIFACTS_DIR,
    CLOUD_TXT_PATH,
    GRAPH_PATH,
    GRAPH_TXT_PATH,
)

logger = Logger.get_logger("robot_scan.save")


BASE_DIR = ARTIFACTS_DIR


def create_run_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = BASE_DIR / ts
    path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created data directory {path}")
    return path


def save_rgbd(path: Path, idx: int, color: np.ndarray, depth: np.ndarray) -> None:
    rgb_path = path / f"{idx:03d}_rgb.png"
    depth_path = path / f"{idx:03d}_depth.npy"
    cv2.imwrite(str(rgb_path), color)
    np.save(depth_path, depth)
    logger.info("Saved RGB-D images for frame %03d", idx)


def save_cloud(path: Path, idx: int, cloud: o3d.geometry.PointCloud) -> None:
    pcd_path = path / f"{idx:03d}_cloud.pcd"
    o3d.io.write_point_cloud(str(pcd_path), cloud)
    logger.info(f"Saved point cloud {pcd_path}")


def save_cloud_txt(cloud: o3d.geometry.PointCloud, path: Path = CLOUD_TXT_PATH) -> Path:
    """Save point cloud to a text file with XYZ coordinates.

    Parameters
    ----------
    cloud:
        Open3D point cloud object.
    path:
        Output file path. Defaults to ``data/artifacts/cloud.txt``.

    Returns
    -------
    Path
        Path to the saved file.
    """

    try:
        pts = np.asarray(cloud.points)
        if cloud.has_normals():
            normals = np.asarray(cloud.normals)
            data = np.hstack([pts, normals])
        else:
            data = pts
        np.savetxt(path, data, fmt="%.6f")
        logger.info(f"Saved cloud TXT {path}")
    except Exception as exc:
        logger.error("Failed saving cloud TXT")
        ErrorTracker.report(exc)
    return path


def save_metadata(path: Path, data: Dict[str, object]) -> None:
    meta_path = path / "metadata.txt"
    with open(meta_path, "w", encoding="utf-8") as f:
        for key, val in data.items():
            if isinstance(val, np.ndarray):
                val = np.array2string(val, precision=6, suppress_small=True)
            f.write(f"{key}: {val}\n")
    logger.info(f"Saved metadata to {meta_path}")


def save_depth_txt(
    path: Path, idx: int, depth: np.ndarray, *, scale: float = 1.0
) -> None:
    """Save depth map as a text file with Z values in meters."""

    txt_path = path / f"{idx:03d}_depth.txt"
    try:
        np.savetxt(txt_path, depth.astype(np.float64) * scale, fmt="%.6f")
        logger.info(f"Saved depth TXT {txt_path}")
    except Exception as exc:
        logger.error("Failed saving depth TXT")
        ErrorTracker.report(exc)


def save_graph_txt(graph: nx.Graph, path: Path = GRAPH_TXT_PATH) -> Path:
    """Save a graph's nodes and edges to a text file.

    Parameters
    ----------
    graph:
        Graph with nodes having ``pos`` attribute.
    path:
        Output file path. Defaults to ``data/artifacts/graph.txt``.

    Returns
    -------
    Path
        Path to the saved file.
    """

    try:
        with open(path, "w", encoding="utf-8") as f:
            for node, data in graph.nodes(data=True):
                pos = data.get("pos", (0.0, 0.0, 0.0))
                f.write(f"node {node} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
            for u, v in graph.edges():
                f.write(f"edge {u} {v}\n")
        logger.info(f"Saved graph TXT {path}")
    except Exception as exc:
        logger.error("Failed saving graph TXT")
        ErrorTracker.report(exc)
    return path


def save_graph_npy(graph: nx.Graph, path: Path = GRAPH_PATH) -> Path:
    """Save a graph's adjacency matrix to a NumPy ``.npy`` file.

    Parameters
    ----------
    graph:
        Graph to serialize.
    path:
        Output file path. Defaults to ``data/artifacts/graph.npy``.

    Returns
    -------
    Path
        Path to the saved file.
    """

    try:
        np.save(path, nx.to_numpy_array(graph))
        logger.info(f"Saved graph NPY {path}")
    except Exception as exc:
        logger.error("Failed saving graph NPY")
        ErrorTracker.report(exc)
    return path


def save_graph_html(
    path: Path,
    graph: nx.Graph,
    cloud: Optional[o3d.geometry.PointCloud] = None,
    name: str = "graph",
) -> Path:
    """Export an interactive 3D Plotly visualization of a graph."""

    html_path = path / f"{name}.html"
    try:
        fig = go.Figure()
        if cloud is not None:
            pts = np.asarray(cloud.points)
            fig.add_trace(
                go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    mode="markers",
                    marker=dict(size=1, color="lightgray"),
                    name="cloud",
                )
            )
        if graph.number_of_nodes() > 0:
            xs, ys, zs = [], [], []
            for u, v in graph.edges():
                xu, yu, zu = graph.nodes[u]["pos"]
                xv, yv, zv = graph.nodes[v]["pos"]
                xs += [xu, xv, None]
                ys += [yu, yv, None]
                zs += [zu, zv, None]
            fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name="edges"))
            nx_pts = np.array([data["pos"] for _, data in graph.nodes(data=True)])
            fig.add_trace(
                go.Scatter3d(
                    x=nx_pts[:, 0],
                    y=nx_pts[:, 1],
                    z=nx_pts[:, 2],
                    mode="markers",
                    marker=dict(size=3, color="red"),
                    name="nodes",
                )
            )
        fig.update_layout(scene_aspectmode="data")
        fig.write_html(str(html_path))
        logger.info(f"Saved graph HTML {html_path}")
    except Exception as exc:
        logger.error("Failed saving graph HTML")
        ErrorTracker.report(exc)
    return html_path
