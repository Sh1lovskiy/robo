"""
laser.py — Compact laser footprint visualizer with auto skeleton graph (Open3D).
What it does:
- Loads a PLY captured in the CAMERA frame -> transforms to BASE using TARGET_POSE and hand-eye.
- Crops to a fixed AABB (BASE).
- Finds the dominant PCA plane of the cropped cloud ("the truss face").
- Skeletonizes ONLY points within a small distance to that plane (no environment), builds branches+nodes.
- Animates a soft laser footprint (circle by default) along the longest branch.
- No beam line; only the spot on the surface.
"""

from __future__ import annotations

import os, sys, time, math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np

# -------------------------- Configuration (edit me) ---------------------------
FORCE_CPU = True  # Set CPU renderer + EGL surfaceless before importing Open3D.
CLOUD_PATH = ".data_clouds/farm_20250729_143818.ply"  # PLY in CAMERA frame.
GRAPH_NPY = "graph.npy"  # Will be overwritten by skeleton built from the plane.

# Visual look
MODE = "segment"  # "circle" or "segment"
STROKE_SIGMA_MM = 2.0  # Gaussian sigma in mm (circle radius-scale / segment thickness)
SEGMENT_LENGTH_MM = (
    55.0  # Length for "segment" mode; width = STROKE_SIGMA_MM via Gaussian
)
LIGHT_RGB = (1.0, 1.0, 0.1)  # Soft yellow
SELECTION_PLANE_TOL_M = 0.008  # Only points within |d| to the local plane are affected

# Animation
SHOW_GRAPH = True
STEP_MM = 3.0  # Resample step for branch playback
TICK_MS = 30.0  # Animation tick (ms)

# Transforms (BASE<-TCP pose; Euler XYZ deg, XYZ mm). Hand-eye is known (CAM<->TCP).
TARGET_POSE = np.array([3.63, -103.5, 540.2, -120.2, -1.13, 103.5], dtype=np.float64)
HAND_EYE_R = np.array(
    [
        [0.999048, 0.00428, -0.00625],
        [-0.00706, 0.99658, -0.00804],
        [0.00423, 0.00895, 0.99629],
    ],
    dtype=np.float64,
)
HAND_EYE_T = np.array([-0.036, -0.078, 0.006], dtype=np.float64).reshape(3, 1)  # meters

# Crop AABB (BASE, meters). Tune to include only the truss face region you care about.
BBOX_POINTS = np.array(
    [[-0.57, -0.2, 0.46], [-0.57, 0.2, 0.2], [-0.3, 0.2, 0.05], [-0.3, 0.2, 0.46]],
    dtype=np.float64,
)

# Optics are not used directly here (we use STROKE_SIGMA_MM to keep the circle clearly visible),
# but keep for completeness.
WAVELENGTH_NM, FOCAL_MM, INCIDENT_DIAM_MM, M2 = 1064.0, 210.0, 10.0, 1.6
LIGHT_PLANE_MODE = "knn"  # "knn" for bends; "global" to stick to the PCA plane


# ------------------------- Early renderer env setup ---------------------------
if FORCE_CPU:
    os.environ["OPEN3D_CPU_RENDERING"] = "true"
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")

try:
    import open3d as o3d
except Exception as e:
    print(f"[ERR] Open3D init failed: {e}\nHint: set FORCE_CPU=True.", file=sys.stderr)
    raise


# ------------------------------ Small math utils ------------------------------
def euler_xyz_deg_to_R(rx: float, ry: float, rz: float) -> np.ndarray:
    x, y, z = np.deg2rad([rx, ry, rz])
    Rx = np.array(
        [[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]]
    )
    Ry = np.array(
        [[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]]
    )
    Rz = np.array(
        [[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]]
    )
    return Rz @ Ry @ Rx


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    P = np.hstack([pts, np.ones((len(pts), 1), dtype=pts.dtype)])
    Q = (T @ P.T).T
    return Q[:, :3]


def pca_plane(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    c = pts.mean(axis=0)
    X = pts - c
    C = (X.T @ X) / max(len(pts) - 1, 1)
    w, V = np.linalg.eigh(C)
    V = V[:, np.argsort(w)[::-1]]
    u, v, n = V[:, 0], V[:, 1], V[:, 2]
    if np.dot(np.cross(u, v), n) < 0:
        v = -v
    return c, -n / np.linalg.norm(n), u / np.linalg.norm(u), v / np.linalg.norm(v)


def knn_plane_at(
    points: np.ndarray, query: np.ndarray, k: int = 200
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    kdt = o3d.geometry.KDTreeFlann(pc)
    k = min(max(3, k), len(points))
    _, idx, _ = kdt.search_knn_vector_3d(query, k)
    neigh = points[np.asarray(idx, dtype=np.int64)]
    return pca_plane(neigh)


def resample_polyline(poly: np.ndarray, step_m: float) -> np.ndarray:
    if len(poly) < 2:
        return poly.copy()
    seg = np.diff(poly, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    L = float(seglen.sum())
    if L <= 0:
        return poly[:1].copy()
    n = max(1, int(L / step_m))
    s = np.hstack([[0.0], np.cumsum(seglen)])
    s_new = np.linspace(0.0, L, n + 1)
    res = []
    j = 0
    for sn in s_new:
        while j < len(s) - 1 and s[j + 1] < sn:
            j += 1
        t = 0.0 if s[j + 1] == s[j] else (sn - s[j]) / (s[j + 1] - s[j])
        res.append(poly[j] * (1 - t) + poly[j + 1] * t)
    return np.asarray(res)


# ------------------------------ Core container --------------------------------
@dataclass
class BranchSet:
    branches: List[np.ndarray]
    nodes: Optional[np.ndarray]
    edges: Optional[List[Tuple[int, int, int]]]  # (u, v, branch_idx)


class LaserSim:
    """Self-contained renderer and graph builder."""

    def __init__(self) -> None:
        # --- transforms
        rx, ry, rz = TARGET_POSE[3:6]
        R_tcp = euler_xyz_deg_to_R(rx, ry, rz)
        t_tcp = TARGET_POSE[:3] * 1e-3
        self.T_b_tcp = make_T(R_tcp, t_tcp)
        T_he = make_T(HAND_EYE_R.copy(), HAND_EYE_T.copy())
        self.T_b_cam = self.T_b_tcp @ np.linalg.inv(T_he)  # BASE<-TCP  TCP<-CAM

        # --- load & crop
        pcd_cam = o3d.io.read_point_cloud(CLOUD_PATH)
        if pcd_cam.is_empty():
            raise RuntimeError(f"No points in cloud: {CLOUD_PATH}")
        pts_base = transform_points(self.T_b_cam, np.asarray(pcd_cam.points))
        self.cloud = o3d.geometry.PointCloud()
        self.cloud.points = o3d.utility.Vector3dVector(pts_base)
        self.cloud.colors = (
            pcd_cam.colors
            if pcd_cam.has_colors()
            else o3d.utility.Vector3dVector(
                np.full((len(pts_base), 3), 0.55, dtype=np.float64)
            )
        )
        mn, mx = BBOX_POINTS.min(axis=0), BBOX_POINTS.max(axis=0)
        self.crop = self.cloud.crop(o3d.geometry.AxisAlignedBoundingBox(mn, mx))
        if self.crop.is_empty():
            raise RuntimeError("Cropped cloud is empty; check BBOX/transforms.")
        self.P = np.asarray(self.crop.points)
        self.colors0 = np.asarray(self.crop.colors, dtype=np.float64).copy()
        self.kdt = o3d.geometry.KDTreeFlann(self.crop)

        # --- main/global plane (truss face)
        self.c_glob, self.n_glob, self.u_glob, self.v_glob = pca_plane(self.P)

        # --- restrict to the plane (segmentation)
        dplane = np.abs((self.P - self.c_glob) @ self.n_glob)
        plane_idx = np.where(dplane <= SELECTION_PLANE_TOL_M)[0]
        self.P_plane = self.P[plane_idx]
        if len(self.P_plane) < 50:
            print("[WARN] Few plane points; falling back to whole crop for skeleton.")
            self.P_plane = self.P

        # --- graph: always rebuild from plane points, save to GRAPH_NPY
        self.branchset = self._build_skeleton_from_plane(self.P_plane)
        try:
            out = {"branches_3d": np.array(self.branchset.branches, dtype=object)}
            if self.branchset.nodes is not None:
                out["node_coords_3d"] = self.branchset.nodes
            if self.branchset.edges is not None:
                out["edges"] = np.array(self.branchset.edges, dtype=np.int32)
            np.save(GRAPH_NPY, out, allow_pickle=True)
            print(f"[GRAPH] saved -> {GRAPH_NPY}")
        except Exception as e:
            print(f"[GRAPH] save failed: {e}")

        # --- choose branch & prepare playback
        self.step_m = max(1e-6, STEP_MM * 1e-3)
        self.branch_idx = self._longest_branch_index()
        self.path = (
            resample_polyline(self.branchset.branches[self.branch_idx], self.step_m)
            if self.branchset.branches
            else np.array([self.c_glob])
        )
        self.i = 0
        self.center = self.path[0] if len(self.path) else self.c_glob

        # --- visual params
        self.show_graph = SHOW_GRAPH
        self.show_cloud = True
        self.laser_on = True
        self.ms_per_tick = float(TICK_MS)
        self.last = time.time()

        # Circle/segment sigma (visible, independent of optics)
        sigma_m = max(1e-6, STROKE_SIGMA_MM * 1e-3)
        self.sigma_circle = sigma_m
        self.sigma_across = sigma_m
        print(f"[OPTICS] sigma_circle={self.sigma_circle*1e3:.2f} mm   mode={MODE}")

        # --- visuals
        self.aabb_ls = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
            o3d.geometry.AxisAlignedBoundingBox(mn, mx)
        )
        self.aabb_ls.paint_uniform_color([0.95, 0.9, 0.1])

    # --------------------------- skeletonization ---------------------------
    def _build_skeleton_from_plane(self, P: np.ndarray) -> BranchSet:
        """Downsample -> linearity filter -> DBSCAN -> PCA-ordered polylines -> node clustering."""
        # Downsample for robustness vs noise
        voxel = max(0.0025, float(self._median_nn(P) * 2.0))
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(P)
        pc = pc.voxel_down_sample(voxel)
        Q = np.asarray(pc.points)
        if len(Q) < 30:
            return BranchSet([], None, None)

        # Linearity via local PCA
        kdt = o3d.geometry.KDTreeFlann(pc)
        k = min(30, len(Q))
        scores = np.zeros(len(Q))
        for i in range(len(Q)):
            _, idx, _ = kdt.search_knn_vector_3d(Q[i], k)
            N = Q[np.asarray(idx, dtype=np.int64)]
            c, n, u, v = pca_plane(N)
            X = N - c
            C = (X.T @ X) / max(len(N) - 1, 1)
            vals, _ = np.linalg.eigh(C)
            vals = np.sort(vals)[::-1]
            scores[i] = (vals[0] - vals[1]) / (vals[0] + 1e-12)
        mask = scores >= np.quantile(scores, 0.5)  # keep top 50% "line-like"
        Ql = Q[mask]
        if len(Ql) < 20:
            Ql = Q

        # Cluster by DBSCAN in the plane
        eps = max(0.01, float(self._median_nn(Ql) * 4.0))
        labels = np.array(
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Ql)).cluster_dbscan(
                eps=eps, min_points=12, print_progress=False
            )
        )
        ncl = int(labels.max()) + 1 if labels.size else 0

        branches: List[np.ndarray] = []
        endpoints: List[np.ndarray] = []

        for ci in range(ncl):
            I = np.where(labels == ci)[0]
            if I.size < 8:
                continue
            pts = Ql[I]
            # Project order along principal axis u
            c, n, u, v = pca_plane(pts)
            t = (pts - c) @ u
            order = np.argsort(t)
            poly = pts[order]
            if len(poly) >= 2:
                branches.append(poly)
                endpoints.append(poly[0])
                endpoints.append(poly[-1])

        if not branches:
            return BranchSet([], None, None)

        # Nodes from endpoint clustering
        E = np.vstack(endpoints)
        node_eps = max(0.008, float(self._median_nn(E) * 3.0))
        nlabels = np.array(
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(E)).cluster_dbscan(
                eps=node_eps, min_points=1, print_progress=False
            )
        )
        ncount = int(nlabels.max()) + 1
        nodes = np.zeros((ncount, 3), dtype=np.float64)
        for i in range(ncount):
            nodes[i] = E[nlabels == i].mean(axis=0)

        # Map branch endpoints to nodes
        kdt_nodes = o3d.geometry.KDTreeFlann(
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(nodes))
        )
        edges: List[Tuple[int, int, int]] = []
        for bi, poly in enumerate(branches):
            _, i0, _ = kdt_nodes.search_knn_vector_3d(poly[0], 1)
            _, i1, _ = kdt_nodes.search_knn_vector_3d(poly[-1], 1)
            u_id, v_id = int(i0[0]), int(i1[0])
            if u_id != v_id:
                edges.append((u_id, v_id, bi))

        print(
            f"[GRAPH] plane-only: nodes={len(nodes)} branches={len(branches)} edges={len(edges)} (eps={eps:.3f}m)"
        )
        return BranchSet(branches, nodes, edges)

    @staticmethod
    def _median_nn(P: np.ndarray) -> float:
        if len(P) < 2:
            return 0.002
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(P)
        kdt = o3d.geometry.KDTreeFlann(pc)
        d = []
        step = max(1, len(P) // 1500)
        for i in range(0, len(P), step):
            _, idx, _ = kdt.search_knn_vector_3d(P[i], 2)
            if len(idx) == 2:
                d.append(np.linalg.norm(P[i] - P[idx[1]]))
        return float(np.median(d)) if d else 0.002

    def _longest_branch_index(self) -> int:
        if not self.branchset.branches:
            return 0
        lens = [
            float(np.linalg.norm(np.diff(b, axis=0), axis=1).sum())
            for b in self.branchset.branches
        ]
        return int(np.argmax(lens))

    # ------------------------------ Rendering ------------------------------
    def _local_plane(
        self, center: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if LIGHT_PLANE_MODE == "global":
            c, n, u, v = self.c_glob, self.n_glob, self.u_glob, self.v_glob
        else:
            c, n, u, v = knn_plane_at(self.P, center, 200)
        tcp_minus_z = -self.T_b_tcp[:3, 2]
        if np.dot(n, tcp_minus_z) < 0.0:
            n = -n
            v = -v
        return c, n / np.linalg.norm(n), u / np.linalg.norm(u), v / np.linalg.norm(v)

    def _weights(self, center: np.ndarray) -> np.ndarray:
        _, n, u, v = self._local_plane(center)
        d = self.P - center
        mask = (np.abs(d @ n) <= SELECTION_PLANE_TOL_M).astype(np.float64)
        du, dv = d @ u, d @ v
        if MODE == "circle":
            r2 = du * du + dv * dv
            w = np.exp(-0.5 * (r2 / max(self.sigma_circle * self.sigma_circle, 1e-12)))
        else:
            half = max(1e-6, 0.5 * SEGMENT_LENGTH_MM * 1e-3)
            rect = (np.abs(dv) <= half).astype(np.float64)
            w = (
                np.exp(
                    -0.5 * (du * du) / max(self.sigma_across * self.sigma_across, 1e-12)
                )
                * rect
            )
        return w * mask

    def update_colors(self) -> None:
        if not self.laser_on:
            self.crop.colors = o3d.utility.Vector3dVector(self.colors0)
            return
        a = np.clip(self._weights(self.center), 0.0, 1.0).reshape(-1, 1)
        yellow = np.array(LIGHT_RGB, dtype=np.float64).reshape(1, 3)
        colors = self.colors0 * (1.0 - a) + yellow * a
        self.crop.colors = o3d.utility.Vector3dVector(colors)

    # ------------------------------ Viewer loop ------------------------------
    def run(self) -> None:
        pcd = self.crop
        voxel_size = 0.003
        normal_length = 0.02
        pcd = pcd.voxel_down_sample(voxel_size)

        pcd.normalize_normals()

        pcd.normals = o3d.utility.Vector3dVector(
            np.asarray(pcd.normals) * normal_length
        )
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 2.5, max_nn=30
            )
        )
        o3d.visualization.draw_geometries([pcd])

        # vis = o3d.visualization.VisualizerWithKeyCallback()
        # vis.create_window(
        #     "Laser (plane-only skeleton)", width=1280, height=800, visible=True
        # )
        # vis.add_geometry(self.crop)
        # vis.add_geometry(self.aabb_ls)

        # # graph geoms
        # graph_ls = o3d.geometry.LineSet()
        # nodes_pc = o3d.geometry.PointCloud()
        # self._refresh_graph_geoms(graph_ls, nodes_pc)
        # if SHOW_GRAPH and len(self.branchset.branches):
        #     vis.add_geometry(graph_ls)
        #     if len(nodes_pc.points):
        #         vis.add_geometry(nodes_pc)

        # # keys
        # vis.register_key_callback(ord(" "), lambda v: self._k_space(vis))
        # vis.register_key_callback(ord("L"), lambda v: self._k_l(vis))
        # vis.register_key_callback(
        #     ord("G"), lambda v: self._k_g(vis, graph_ls, nodes_pc)
        # )
        # vis.register_key_callback(ord("C"), lambda v: self._k_c(vis))
        # vis.register_key_callback(ord("H"), lambda v: self._k_h(vis))
        # vis.register_key_callback(ord("q"), lambda v: self._k_quit(vis))
        # vis.register_key_callback(ord("Q"), lambda v: self._k_quit(vis))
        # vis.register_key_callback(256, lambda v: self._k_quit(vis))
        # print("[INFO] Keys: SPACE, L, G, C, H, ESC/q")

        # self.update_colors()
        # playing = True if len(self.path) > 1 else False
        # while True:
        #     if playing and (time.time() - self.last) * 1000.0 >= self.ms_per_tick:
        #         self.last = time.time()
        #         self.i = (self.i + 1) % len(self.path)
        #         self.center = self.path[self.i]
        #         self.update_colors()
        #         vis.update_geometry(self.crop)
        #     vis.update_renderer()
        #     if not vis.poll_events():
        #         break

    def _refresh_graph_geoms(
        self, graph_ls: o3d.geometry.LineSet, nodes_pc: o3d.geometry.PointCloud
    ) -> None:
        B, N = self.branchset.branches, self.branchset.nodes
        if not B:
            return
        pts_all, lines, base = [], [], 0
        for b in B:
            pts_all.append(b)
            lines += [[base + i, base + i + 1] for i in range(len(b) - 1)]
            base += len(b)
        pts = np.vstack(pts_all)
        graph_ls.points = o3d.utility.Vector3dVector(pts)
        graph_ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
        graph_ls.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([[0.8, 0.8, 0.8]]), (len(lines), 1))
        )
        if N is not None and len(N):
            nodes_pc.points = o3d.utility.Vector3dVector(N)
            nodes_pc.paint_uniform_color([0.9, 0.2, 0.2])

    # ---- key callbacks
    def _k_space(self, vis) -> bool:
        self.ms_per_tick = 1e9 if self.ms_per_tick < 1e8 else float(TICK_MS)
        print(f"[SPACE] {'pause' if self.ms_per_tick>1e8 else 'play'}")
        return True

    def _k_l(self, vis) -> bool:
        self.laser_on = not self.laser_on
        self.update_colors()
        vis.update_geometry(self.crop)
        print(f"[L] laser_on={self.laser_on}")
        return True

    def _k_g(self, vis, graph_ls, nodes_pc) -> bool:
        self.show_graph = not self.show_graph
        if self.show_graph and len(self.branchset.branches):
            self._refresh_graph_geoms(graph_ls, nodes_pc)
            vis.add_geometry(graph_ls)
            if len(nodes_pc.points):
                vis.add_geometry(nodes_pc)
        else:
            vis.remove_geometry(graph_ls, reset_bounding_box=False)
            vis.remove_geometry(nodes_pc, reset_bounding_box=False)
        print(f"[G] graph={self.show_graph}")
        return True

    def _k_c(self, vis) -> bool:
        self.show_cloud = not self.show_cloud
        if self.show_cloud:
            vis.add_geometry(self.crop, reset_bounding_box=False)
        else:
            vis.remove_geometry(self.crop, reset_bounding_box=False)
        print(f"[C] cloud={self.show_cloud}")
        return True

    def _k_h(self, vis) -> bool:
        self.i = 0
        if len(self.path):
            self.center = self.path[0]
            self.update_colors()
            vis.update_geometry(self.crop)
        print("[H] restart")
        return True

    def _k_quit(self, vis) -> bool:
        vis.close()
        return True


def main() -> None:
    print(
        f"[INIT] cloud={CLOUD_PATH} mode={MODE} sigma={STROKE_SIGMA_MM}mm tol={SELECTION_PLANE_TOL_M}m"
    )
    LaserSim().run()


if __name__ == "__main__" or __name__ == "laser":
    main()
