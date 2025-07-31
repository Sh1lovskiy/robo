"""Graph construction utilities from skeleton data."""

from __future__ import annotations

from typing import List, Tuple

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from utils.logger import Logger

logger = Logger.get_logger("robot_scan.graph")


def merge_close_points(points: np.ndarray, radius: float):
    tree = cKDTree(points)
    visited = np.zeros(len(points), dtype=bool)
    merged, mapping = [], np.full(len(points), -1)
    for i in range(len(points)):
        if visited[i]:
            continue
        group = tree.query_ball_point(points[i], radius)
        merged_point = np.mean(points[group], axis=0)
        idx = len(merged)
        merged.append(merged_point)
        mapping[group] = idx
        visited[group] = True
    return np.array(merged), mapping


def build_graph(nodes: np.ndarray, branches: List[np.ndarray], radius: float = 0.02):
    """Build a graph from skeleton nodes and branch polylines."""
    merged, node_map = merge_close_points(nodes, radius)
    edges = set()
    for br in branches:
        if len(br) < 2:
            continue
        a = np.argmin(np.linalg.norm(nodes - br[0], axis=1))
        b = np.argmin(np.linalg.norm(nodes - br[-1], axis=1))
        i, j = node_map[a], node_map[b]
        if i != j:
            edges.add(tuple(sorted((i, j))))
    G = nx.Graph()
    for idx, pt in enumerate(merged):
        G.add_node(idx, pos=pt)
    for i, j in edges:
        dist = np.linalg.norm(merged[i] - merged[j])
        G.add_edge(i, j, weight=dist)
    logger.info("Graph built: %s nodes, %s edges", G.number_of_nodes(), G.number_of_edges())
    return G
