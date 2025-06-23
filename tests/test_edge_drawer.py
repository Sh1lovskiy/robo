import sys
from pathlib import Path

import numpy as np
import open3d as o3d  # type: ignore

sys.path.append(str(Path(__file__).resolve().parents[1]))

from edge_drawer import _find_top_edge, run, AppConfig


def rectangle_cloud(width: float, height: float) -> o3d.geometry.PointCloud:
    xs = np.linspace(0.0, width, 5)
    ys = np.linspace(0.0, height, 5)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    return pcd


def test_extract_edge() -> None:
    pcd = rectangle_cloud(0.1, 0.05)
    edge, _ = _find_top_edge(pcd, 0.01)
    assert np.allclose(edge[:, 1], 0.05)
    assert np.isclose(edge[0, 0], 0.1)
    assert np.isclose(edge[-1, 0], 0.0)


def test_main_smoke() -> None:
    cfg = AppConfig()
    run(cfg, dry_run=True)
