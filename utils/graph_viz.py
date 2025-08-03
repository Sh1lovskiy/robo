"""Plotly visualization helpers for graph on point cloud."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import networkx as nx
from scipy.linalg import svd
import plotly.graph_objects as go


def _load_point_cloud(path: Path) -> np.ndarray:
    """Load XYZ coordinates from a text file.

    Parameters
    ----------
    path: Path
        Path to ``cloud.txt`` containing rows of ``x y z`` values.

    Returns
    -------
    np.ndarray
        Array of shape (N, 3) with point coordinates.
    """

    return np.loadtxt(path, usecols=(0, 1, 2))


def _load_graph(path: Path) -> tuple[nx.Graph, list[np.ndarray]]:
    """Load a graph and optional branch polylines.

    The serialized object may be an :class:`networkx.Graph` pickled via
    ``numpy.save`` or a dictionary with ``"nodes"`` and ``"edges"`` keys.

    Parameters
    ----------
    path: Path
        Path to ``graph.npy``.

    Returns
    -------
    tuple[nx.Graph, list[np.ndarray]]
        The constructed graph and a list of branch polylines for curved
        visualization.
    """

    data = np.load(path, allow_pickle=True)
    obj = data.item()

    branches: list[np.ndarray] = []
    if isinstance(obj, nx.Graph):
        graph = obj
    else:
        graph = nx.Graph()
        coords = obj.get("node_coords_3d")
        if coords is not None:
            for i, coord in enumerate(coords):
                graph.add_node(int(i), xyz=np.asarray(coord))
        for edge in obj.get("edges", []):
            graph.add_edge(int(edge[0]), int(edge[1]))
        branches = [np.asarray(b) for b in obj.get("branches_3d", [])]
    return graph, branches


def _estimate_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate best-fit plane using SVD/PCA.

    Parameters
    ----------
    points: np.ndarray
        Point cloud array of shape (N, 3).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of centroid and normal vector.
    """

    centroid = points.mean(axis=0)
    _, _, vh = svd(points - centroid)
    normal = vh[-1]
    return centroid, normal


def _plane_mesh(centroid: np.ndarray, normal: np.ndarray, scale: float) -> go.Mesh3d:
    """Create a mesh representing a plane."""

    normal = normal / np.linalg.norm(normal)
    # Choose arbitrary vector not parallel to normal
    if np.allclose(normal, [1.0, 0.0, 0.0]):
        v = np.array([0.0, 1.0, 0.0])
    else:
        v = np.array([1.0, 0.0, 0.0])
    basis1 = np.cross(normal, v)
    basis1 /= np.linalg.norm(basis1)
    basis2 = np.cross(normal, basis1)

    s = scale
    corners = np.array(
        [
            centroid + s * (basis1 + basis2),
            centroid + s * (-basis1 + basis2),
            centroid + s * (-basis1 - basis2),
            centroid + s * (basis1 - basis2),
        ]
    )
    # Two triangles (0,1,2) and (0,2,3)
    i, j, k = [0, 0], [1, 2], [2, 3]
    return go.Mesh3d(
        x=corners[:, 0],
        y=corners[:, 1],
        z=corners[:, 2],
        i=i,
        j=j,
        k=k,
        color="lightgrey",
        opacity=0.5,
        name="plane",
        showscale=False,
    )


def visualize_graph_on_cloud(
    cloud_path: Path,
    graph_path: Path,
    output_path: Path,
    show_plane: bool = True,
    show_edges: bool = True,
    show_nodes: bool = True,
    use_straight_edges: bool = True,
) -> None:
    """Generate interactive visualization of graph on a point cloud.

    Parameters
    ----------
    cloud_path: Path
        Path to point cloud text file.
    graph_path: Path
        Path to graph ``.npy`` file.
    output_path: Path
        Location where the HTML visualization is written.
    show_plane: bool, default=True
        Whether to render the estimated plane initially.
    show_edges: bool, default=True
        Whether to render graph edges initially.
    show_nodes: bool, default=True
        Whether to render graph nodes initially.
    use_straight_edges: bool, default=True
        Use straight line segments for edges; otherwise use curved
        polylines if available.
    """

    points = _load_point_cloud(cloud_path)
    graph, branches = _load_graph(graph_path)

    # Prepare traces
    cloud_trace = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker=dict(size=2, color="rgba(30, 136, 229, 0.5)"),
        name="cloud",
    )

    node_coords = np.array([graph.nodes[n]["xyz"] for n in graph.nodes])
    node_trace = go.Scatter3d(
        x=node_coords[:, 0],
        y=node_coords[:, 1],
        z=node_coords[:, 2],
        mode="markers+text",
        marker=dict(size=5, color="red"),
        text=[str(n) for n in graph.nodes],
        name="nodes",
    )

    # Straight edge trace
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    for u, v in graph.edges:
        p, q = graph.nodes[u]["xyz"], graph.nodes[v]["xyz"]
        xs.extend([p[0], q[0], None])
        ys.extend([p[1], q[1], None])
        zs.extend([p[2], q[2], None])
    straight_trace = go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line=dict(color="darkgrey", width=4),
        name="edges",
    )

    # Curved edge traces
    curved_traces = [
        go.Scatter3d(
            x=b[:, 0],
            y=b[:, 1],
            z=b[:, 2],
            mode="lines",
            line=dict(color="darkred", width=4),
            showlegend=False,
            name="curve",
        )
        for b in branches
    ]

    traces: list[go.Trace] = [cloud_trace]

    # Plane
    plane_trace = None
    if show_plane:
        centroid, normal = _estimate_plane(points)
        scale = np.linalg.norm(points - centroid, axis=1).max()
        plane_trace = _plane_mesh(centroid, normal, scale)
    traces.append(plane_trace if plane_trace else go.Scatter3d())

    traces.append(node_trace)

    # Append curved traces
    for t in curved_traces:
        traces.append(t)

    traces.append(straight_trace)

    fig = go.Figure(data=[t for t in traces if t is not None])

    n_curved = len(curved_traces)
    base = [True, show_plane, show_nodes]
    for i, trace in enumerate(fig.data):
        if i == 1:
            trace.visible = show_plane
        elif i == 2:
            trace.visible = show_nodes
        elif 3 <= i < 3 + n_curved:
            trace.visible = show_edges and not use_straight_edges
        elif i == 3 + n_curved:
            trace.visible = show_edges and use_straight_edges

    vis_curved = base + [True] * n_curved + [False]
    vis_straight = base + [False] * n_curved + [True]
    vis_none = base + [False] * (n_curved + 1)

    updatemenus = [
        dict(
            buttons=[
                dict(label="Curved", method="update", args=[{"visible": vis_curved}]),
                dict(
                    label="Straight", method="update", args=[{"visible": vis_straight}]
                ),
                dict(label="No edges", method="update", args=[{"visible": vis_none}]),
            ],
            direction="left",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.0,
            y=1.15,
            xanchor="left",
            yanchor="top",
        ),
        dict(
            buttons=[
                dict(label="Plane on", method="restyle", args=[{"visible": True}, [1]]),
                dict(
                    label="Plane off", method="restyle", args=[{"visible": False}, [1]]
                ),
            ],
            direction="left",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.0,
            y=1.05,
            xanchor="left",
            yanchor="top",
        ),
        dict(
            buttons=[
                dict(label="Cloud on", method="restyle", args=[{"visible": True}, [0]]),
                dict(
                    label="Cloud off", method="restyle", args=[{"visible": False}, [0]]
                ),
            ],
            direction="left",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.0,
            y=0.95,
            xanchor="left",
            yanchor="top",
        ),
        dict(
            buttons=[
                dict(label="Nodes on", method="restyle", args=[{"visible": True}, [2]]),
                dict(
                    label="Nodes off", method="restyle", args=[{"visible": False}, [2]]
                ),
            ],
            direction="left",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.0,
            y=0.85,
            xanchor="left",
            yanchor="top",
        ),
    ]

    fig.update_layout(
        scene=dict(aspectmode="data"),
        updatemenus=updatemenus,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))


if __name__ == "__main__":
    from utils.settings import CLOUD_TXT_PATH, GRAPH_PATH, NOTEBOOKS_DIR

    visualize_graph_on_cloud(
        CLOUD_TXT_PATH, GRAPH_PATH, NOTEBOOKS_DIR / "graph_scene.html"
    )
