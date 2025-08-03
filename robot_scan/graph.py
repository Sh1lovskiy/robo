"""Graph construction utilities from skeleton data."""

from __future__ import annotations

from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
import heapq
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
    logger.info(
        f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
    )
    return G


def _first_edge(graph: nx.Graph, start: int) -> Optional[Tuple[int, int]]:
    """Return the smallest indexed edge from ``start``."""

    neighbors = sorted(graph.neighbors(start))
    return (start, neighbors[0]) if neighbors else None


def backtracking(graph: nx.Graph, start: int) -> List[Tuple[int, int]]:
    """Traverse the graph using backtracking starting from ``start``."""

    g = graph.copy()
    path: List[Tuple[int, int]] = []
    stack = [start]
    edge = _first_edge(g, start)
    if edge:
        path.append(edge)
        g.remove_edge(*edge)
        stack.append(edge[1])
    while stack:
        v = stack[-1]
        nbrs = list(g.neighbors(v))
        if nbrs:
            u = nbrs[0]
            g.remove_edge(v, u)
            path.append((v, u))
            stack.append(u)
        else:
            stack.pop()
    return path


def dfs(graph: nx.Graph, start: int) -> List[Tuple[int, int]]:
    """Depth-first traversal returning visited edges."""

    path: List[Tuple[int, int]] = []
    visited: set[Tuple[int, int]] = set()
    stack = [start]
    edge = _first_edge(graph, start)
    if edge:
        visited.add(tuple(sorted(edge)))
        path.append(edge)
        stack.append(edge[1])
    while stack:
        v = stack.pop()
        for u in graph.neighbors(v):
            e = tuple(sorted((v, u)))
            if e not in visited:
                visited.add(e)
                path.append((v, u))
                stack.append(u)
    return path


def dijkstra_walk(graph: nx.Graph, start: int) -> List[Tuple[int, int]]:
    """Traverse using repeated Dijkstra shortest paths from ``start``."""

    visited: set[Tuple[int, int]] = set()
    path: List[Tuple[int, int]] = []
    current = start
    edge = _first_edge(graph, start)
    if edge:
        visited.add(tuple(sorted(edge)))
        path.append(edge)
        current = edge[1]

    def next_route(v: int) -> Optional[List[int]]:
        heap = [(0.0, [v])]
        seen = set()
        while heap:
            cost, route = heapq.heappop(heap)
            node = route[-1]
            if node in seen:
                continue
            seen.add(node)
            for n in graph.neighbors(node):
                e = tuple(sorted((node, n)))
                if e not in visited:
                    return route + [n]
                heapq.heappush(heap, (cost + graph[node][n]["weight"], route + [n]))
        return None

    while len(visited) < graph.number_of_edges():
        route = next_route(current)
        if not route:
            break
        for a, b in zip(route, route[1:]):
            e = tuple(sorted((a, b)))
            if e not in visited:
                visited.add(e)
                path.append((a, b))
        current = route[-1]
    return path


def greedy(graph: nx.Graph, start: int) -> List[Tuple[int, int]]:
    """Greedy traversal following nearest unvisited edges."""

    visited: set[Tuple[int, int]] = set()
    path: List[Tuple[int, int]] = []
    current = start
    edge = _first_edge(graph, start)
    if edge:
        visited.add(tuple(sorted(edge)))
        path.append(edge)
        current = edge[1]
    while len(visited) < graph.number_of_edges():
        candidates = [
            u
            for u in graph.neighbors(current)
            if tuple(sorted((current, u))) not in visited
        ]
        if candidates:
            u = min(
                candidates,
                key=lambda x: np.linalg.norm(
                    np.array(graph.nodes[current]["pos"])
                    - np.array(graph.nodes[x]["pos"])
                ),
            )
            e = tuple(sorted((current, u)))
            visited.add(e)
            path.append((current, u))
            current = u
        else:
            unvisited = [e for e in graph.edges if tuple(sorted(e)) not in visited]
            if not unvisited:
                break
            current = unvisited[0][0]
    return path
