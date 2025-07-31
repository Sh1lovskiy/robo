# graph_traversal_visualizer.py

"""
Graph traversal visualizer for 3D branch-node graphs.

- Merges spatially close 3D points
- Constructs graph edges from segmented branches
- Supports interactive traversal visualization
"""

import os
import heapq
import numpy as np
import open3d as o3d
import networkx as nx
from scipy.spatial import cKDTree
from utils.logger import Logger

logger = Logger.get_logger("graph")
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"

# =============================================================================
# Graph Construction
# =============================================================================


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


def build_graph(nodes: np.ndarray, branches: list, radius: float = 0.02):
    merged, node_map = merge_close_points(nodes, radius)
    edges = set()

    for branch in branches:
        if len(branch) < 2:
            continue
        a = np.argmin(np.linalg.norm(nodes - branch[0], axis=1))
        b = np.argmin(np.linalg.norm(nodes - branch[-1], axis=1))
        i, j = node_map[a], node_map[b]
        if i != j:
            edges.add(tuple(sorted((i, j))))

    G = nx.Graph()
    for idx, pt in enumerate(merged):
        G.add_node(idx, pos=pt)
    for i, j in edges:
        dist = np.linalg.norm(merged[i] - merged[j])
        G.add_edge(i, j, weight=dist)

    logger.info(f"Graph has {len(G.nodes)} nodes and {len(G.edges)} edges.")
    return G, merged


# =============================================================================
# Graph Traversal Algorithms
# =============================================================================


def get_first_edge(G, start):
    neighbors = sorted(G.neighbors(start))
    return (start, neighbors[0]) if neighbors else None


def backtracking(G, start):
    G_copy = G.copy()
    path, stack = [], [start]

    edge = get_first_edge(G, start)
    if edge:
        path.append(edge)
        G_copy.remove_edge(*edge)
        stack.append(edge[1])

    while stack:
        v = stack[-1]
        nbrs = list(G_copy.neighbors(v))
        if nbrs:
            u = nbrs[0]
            G_copy.remove_edge(v, u)
            path.append((v, u))
            stack.append(u)
        else:
            stack.pop()

    return path


def dfs(G, start):
    path, visited = [], set()
    stack = [start]

    edge = get_first_edge(G, start)
    if edge:
        visited.add(tuple(sorted(edge)))
        path.append(edge)
        stack.append(edge[1])

    while stack:
        v = stack.pop()
        for u in G.neighbors(v):
            e = tuple(sorted((v, u)))
            if e not in visited:
                visited.add(e)
                path.append((v, u))
                stack.append(u)

    return path


def dijkstra_walk(G, start):
    visited, path = set(), []
    current = start

    edge = get_first_edge(G, start)
    if edge:
        visited.add(tuple(sorted(edge)))
        path.append(edge)
        current = edge[1]

    def next_path(v):
        heap = [(0, [v])]
        seen = set()
        while heap:
            cost, route = heapq.heappop(heap)
            u = route[-1]
            if u in seen:
                continue
            seen.add(u)
            for n in G.neighbors(u):
                e = tuple(sorted((u, n)))
                if e not in visited:
                    return route + [n]
                heapq.heappush(heap, (cost + G[u][n]["weight"], route + [n]))
        return None

    while len(visited) < G.number_of_edges():
        route = next_path(current)
        if not route:
            break
        for a, b in zip(route, route[1:]):
            e = tuple(sorted((a, b)))
            if e not in visited:
                visited.add(e)
                path.append((a, b))
        current = route[-1]

    return path


def greedy(G, start):
    visited, path = set(), []
    current = start

    edge = get_first_edge(G, start)
    if edge:
        visited.add(tuple(sorted(edge)))
        path.append(edge)
        current = edge[1]

    while len(visited) < G.number_of_edges():
        candidates = [
            u
            for u in G.neighbors(current)
            if tuple(sorted((current, u))) not in visited
        ]
        if candidates:
            u = min(
                candidates,
                key=lambda x: np.linalg.norm(
                    np.array(G.nodes[current]["pos"]) - np.array(G.nodes[x]["pos"])
                ),
            )
            e = tuple(sorted((current, u)))
            visited.add(e)
            path.append((current, u))
            current = u
        else:
            unvisited = [e for e in G.edges if tuple(sorted(e)) not in visited]
            if not unvisited:
                break
            current = unvisited[0][0]

    return path


ALGO_LIST = [
    ("Backtracking", backtracking),
    ("DFS", dfs),
    ("Dijkstra", dijkstra_walk),
    ("Greedy", greedy),
]


# =============================================================================
# Open3D Viewer
# =============================================================================


def visualize(graph: dict, start_node: int):
    if (
        graph["node_coords_3d"] is None
        or len(graph["node_coords_3d"]) == 0
        or not any(len(b) >= 2 for b in graph["branches_3d"])
    ):
        logger.error("Graph data missing or invalid.")
        return

    nodes = np.array(graph["node_coords_3d"])
    branches = graph["branches_3d"]
    G, points = build_graph(nodes, branches)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Graph Viewer")

    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pc.paint_uniform_color([1, 0, 0])

    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(points)
    lines.lines = o3d.utility.Vector2iVector(list(G.edges))
    lines.colors = o3d.utility.Vector3dVector([[0.4, 0.4, 0.4]] * len(G.edges))

    trace = o3d.geometry.LineSet()
    trace.points = o3d.utility.Vector3dVector([[0, 0, 0]])
    trace.lines = o3d.utility.Vector2iVector([[0, 0]])
    trace.colors = o3d.utility.Vector3dVector([[0, 1, 0]])

    algo_idx, step_idx = {"i": 0}, {"i": 1}
    trace_lines = []

    def build_trace():
        name, func = ALGO_LIST[algo_idx["i"]]
        logger.info(f"Switch algo to: {name}")
        route = func(G, start_node)
        if not route:
            return [[0, 0]]
        pts, lines = [], []
        for a, b in route:
            pts.extend([points[a], points[b]])
            lines.append([len(pts) - 2, len(pts) - 1])
        trace.points = o3d.utility.Vector3dVector(np.array(pts))
        trace.lines = o3d.utility.Vector2iVector([lines[0]])
        trace.colors = o3d.utility.Vector3dVector([[0, 1, 0]])
        step_idx["i"] = 1
        return lines

    def update():
        lines = trace_lines[: step_idx["i"]] or [[0, 0]]
        colors = np.zeros((len(lines), 3))
        colors[:, 1] = np.linspace(0.2, 1, len(colors))
        colors[:, 2] = np.linspace(1, 0, len(colors))
        trace.lines = o3d.utility.Vector2iVector(lines)
        trace.colors = o3d.utility.Vector3dVector(colors)
        vis.update_geometry(trace)
        vis.update_renderer()

    def next_step(vis_):
        step_idx["i"] += 1
        update()
        return False

    def prev_step(vis_):
        step_idx["i"] = max(1, step_idx["i"] - 1)
        update()
        return False

    def switch_algo(vis_):
        algo_idx["i"] = (algo_idx["i"] + 1) % len(ALGO_LIST)
        nonlocal trace_lines
        trace_lines = build_trace()
        update()
        return False

    vis.add_geometry(pc)
    vis.add_geometry(lines)
    vis.add_geometry(trace)
    vis.register_key_callback(ord("D"), next_step)
    vis.register_key_callback(ord("A"), prev_step)
    vis.register_key_callback(ord("S"), switch_algo)

    logger.info("[A] <- prev │ [D] -> next │ [S] switch algo")
    trace_lines = build_trace()
    update()
    vis.run()
    vis.destroy_window()


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    data = np.load("graph_20250731_200614.npy", allow_pickle=True).item()
    logger.info(
        f"Loaded {len(data['node_coords_3d'])} nodes "
        f"and {len(data['branches_3d'])} branches."
    )
    visualize(data, start_node=2)
