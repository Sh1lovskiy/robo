"""Skeletonization utilities for planar point clouds."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import open3d as o3d
from skimage.morphology import skeletonize

from utils.logger import Logger

logger = Logger.get_logger("robot_scan.skeleton")


def _project_to_mask(points: np.ndarray, img_res: int = 1024):
    center = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - center)
    basis = vt[:2]
    coords2d = (points - center) @ basis.T
    min_xy = coords2d.min(axis=0)
    max_xy = coords2d.max(axis=0)
    norm = (coords2d - min_xy) / (max_xy - min_xy + 1e-9)
    pix = np.clip((norm * (img_res - 1)).astype(int), 0, img_res - 1)
    mask = np.zeros((img_res, img_res), np.uint8)
    mask[pix[:, 1], pix[:, 0]] = 1
    return mask, center, basis, min_xy, max_xy


def _pixel_to_3d(
    pix: Tuple[int, int],
    center: np.ndarray,
    basis: np.ndarray,
    min_xy: np.ndarray,
    max_xy: np.ndarray,
    img_res: int,
) -> np.ndarray:
    xy = np.array([pix[1], pix[0]], dtype=np.float32)
    norm = xy / (img_res - 1)
    coord = norm * (max_xy - min_xy) + min_xy
    pt = center + coord @ basis
    return pt


def _extract_nodes(skel: np.ndarray) -> List[Tuple[int, int]]:
    h, w = skel.shape
    offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    nodes = []
    for y in range(h):
        for x in range(w):
            if skel[y, x]:
                cnt = 0
                for dy, dx in offsets:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and skel[ny, nx]:
                        cnt += 1
                if cnt != 2:
                    nodes.append((y, x))
    return nodes


def _skeleton_branches(skel: np.ndarray, nodes: List[Tuple[int, int]]):
    node_map = {tuple(p): i for i, p in enumerate(nodes)}
    visited = np.zeros_like(skel, dtype=bool)
    h, w = skel.shape
    offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    node_mask = np.zeros_like(skel, dtype=bool)
    for y, x in nodes:
        node_mask[y, x] = True
    branches = []
    for (y0, x0) in nodes:
        for dy, dx in offsets:
            y, x = y0 + dy, x0 + dx
            if 0 <= y < h and 0 <= x < w and skel[y, x] and not node_mask[y, x]:
                if visited[y, x]:
                    continue
                branch = [(y0, x0)]
                cy, cx = y, x
                prev = (y0, x0)
                while True:
                    branch.append((cy, cx))
                    visited[cy, cx] = True
                    nbrs = []
                    for ddy, ddx in offsets:
                        ny, nx = cy + ddy, cx + ddx
                        if 0 <= ny < h and 0 <= nx < w and skel[ny, nx]:
                            if (ny, nx) != prev:
                                nbrs.append((ny, nx))
                    if len(nbrs) != 1:
                        break
                    prev = (cy, cx)
                    cy, cx = nbrs[0]
                    if node_mask[cy, cx]:
                        branch.append((cy, cx))
                        break
                if len(branch) > 1:
                    branches.append(branch)
    return branches


def skeletonize_plane(
    plane: o3d.geometry.PointCloud, img_res: int = 1024
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Compute skeleton of a planar cloud.

    Parameters
    ----------
    plane : o3d.geometry.PointCloud
        Planar region of interest.
    img_res : int
        Resolution of intermediate mask.

    Returns
    -------
    nodes : np.ndarray
        3D coordinates of skeleton junction/endpoint nodes.
    branches : list of np.ndarray
        Lists of 3D points for each branch between nodes.
    """
    pts = np.asarray(plane.points)
    mask, center, basis, min_xy, max_xy = _project_to_mask(pts, img_res)
    skel = skeletonize(mask > 0).astype(np.uint8)
    nodes_px = _extract_nodes(skel)
    branches_px = _skeleton_branches(skel, nodes_px)
    nodes = np.array(
        [_pixel_to_3d(p, center, basis, min_xy, max_xy, img_res) for p in nodes_px]
    )
    branches = [
        np.array(
            [_pixel_to_3d(p, center, basis, min_xy, max_xy, img_res) for p in br]
        )
        for br in branches_px
    ]
    logger.info("Skeleton extracted: %s nodes, %s branches", len(nodes), len(branches))
    return nodes, branches
