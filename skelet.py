"""
Extracts graph structure from a 3D point cloud plane via 2D skeletonization.
Includes logging via project logger, time measurement, and progress bars.
"""

import time
from functools import partial

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from scipy.ndimage import distance_transform_edt, label as nd_label
from scipy.spatial import cKDTree
from skimage.morphology import thin
from skimage.segmentation import watershed, find_boundaries

from utils.logger import Logger, SuppressO3DInfo


logger = Logger.get_logger("skelet")


def show_mask(mask, title=""):
    """Visualizes a 2D mask using matplotlib."""
    plt.figure(figsize=(4, 4))
    plt.imshow(mask, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


def load_point_cloud(path):
    """
    Loads a point cloud from the specified path.
    Returns Open3D PointCloud with estimated normals.
    """
    pcd = o3d.io.read_point_cloud(path)
    pcd.estimate_normals()
    logger.info(f"Loaded point cloud: {path} ({len(pcd.points)} points)")
    return pcd


def preprocess_mask(points, center, basis, img_res=1024):
    """
    Projects 3D plane points to 2D via PCA and rasterizes to a binary mask.

    points: (N, 3) array of plane points
    center: mean 3D point
    basis: (2, 3) PCA basis (first two right-singular vectors)
    img_res: output mask resolution

    Returns:
        mask: binary mask (img_res, img_res)
        img_xy: (N, 2) int pixel coordinates for each 3D point
    """
    coords_2d = (points - center) @ basis.T
    min_xy = coords_2d.min(axis=0)
    max_xy = coords_2d.max(axis=0)
    norm_coords = (coords_2d - min_xy) / (max_xy - min_xy + 1e-9)
    img_xy = np.clip((norm_coords * (img_res - 1)).astype(np.int32), 0, img_res - 1)
    mask2d = np.zeros((img_res, img_res), np.uint8)
    mask2d[img_xy[:, 1], img_xy[:, 0]] = 1
    morph_kernel = 45
    kernel_close = np.ones((morph_kernel // 2, morph_kernel // 2), np.uint8)
    kernel_open = np.ones((morph_kernel, morph_kernel), np.uint8)
    mask = cv2.morphologyEx(mask2d * 255, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.GaussianBlur(mask, (7, 7), 1.5)
    mask = (mask > 32).astype(np.uint8)
    logger.info(f"Mask projection: img_res={img_res}, points={len(points)}")
    return mask, img_xy


def extract_nodes(skel):
    """
    Detects skeleton graph nodes: endpoints and junctions.

    Node = pixel with number of neighbors != 2.
    """
    h, w = skel.shape
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
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
    logger.info(f"Extracted {len(nodes)} graph nodes from skeleton")
    return nodes


def skeleton_branches(skel, nodes):
    """
    Traces all branches in skeleton between nodes.

    For each node, traces all 8-connected skeleton paths to another node,
    yielding a list of branches (list of pixels from node to node)
    and adjacency dictionary.

    Returns:
        branches: list of [pixel, pixel, ...] for each branch
        neighbors: dict node_index -> [neighbor_indices...]
    """
    from collections import defaultdict

    node_map = {tuple(p): i for i, p in enumerate(nodes)}
    visited = np.zeros_like(skel, dtype=bool)
    h, w = skel.shape
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    unique_edges = set()
    branches = []
    neighbors = defaultdict(list)
    node_mask = np.zeros_like(skel, dtype=bool)
    for y, x in nodes:
        node_mask[y, x] = True
    for idx, (y0, x0) in enumerate(nodes):
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
                    nb = [
                        (cy + dy, cx + dx)
                        for dy, dx in offsets
                        if 0 <= cy + dy < h
                        and 0 <= cx + dx < w
                        and skel[cy + dy, cx + dx]
                        and (cy + dy, cx + dx) != prev
                    ]
                    if not nb:
                        break
                    next_nb = nb[0]
                    if node_mask[next_nb]:
                        branch.append(next_nb)
                        break
                    prev = (cy, cx)
                    cy, cx = next_nb
                u = node_map[branch[0]]
                v = node_map[branch[-1]]
                if u != v and (u, v) not in unique_edges and (v, u) not in unique_edges:
                    branches.append(branch)
                    unique_edges.add((u, v))
                neighbors[u].append(v)
    logger.info(f"Extracted {len(branches)} branches between nodes")
    return branches, neighbors


def nodes2d_to_3d(nodes, img_xy, points):
    """
    Maps each node pixel to nearest 3D point on the plane.
    Uses KD-tree for robust assignment.

    nodes: [(y, x), ...]
    img_xy: (N, 2) array of (y, x) for each 3D point
    points: (N, 3) array
    """
    tree = cKDTree(img_xy)
    node_arr = np.array(nodes)
    dists, idxs = tree.query(node_arr, k=1)
    coords_3d = points[idxs]
    logger.info(f"Mapped {len(nodes)} nodes to 3D coordinates")
    return coords_3d


def skeleton_branches_to_3d(branches, img_xy, points_3d):
    """
    Maps each skeleton branch (list of pixels) to 3D points via (y, x) → 3D assignment.

    Returns:
        branches_3d: list of (M, 3) arrays
    """
    pix_map = {}
    for i, (y, x) in enumerate(img_xy):
        pix_map.setdefault((y, x), []).append(i)
    branches_3d = []
    for branch in Logger.progress(branches, desc="Mapping branches to 3D"):
        pts = []
        for y, x in branch:
            inds = pix_map.get((y, x), [])
            pts.extend(points_3d[inds])
        if pts:
            branches_3d.append(np.array(pts))
    logger.info(f"Mapped {len(branches_3d)} branches to 3D")
    return branches_3d


def make_o3d_lineset(branches_3d, node_coords_3d):
    """
    Builds Open3D geometries for all branches and node spheres.

    branches_3d: list of (N, 3) arrays
    node_coords_3d: (M, 3) array of node positions
    """
    pts, lines, colors = [], [], []
    idx = 0
    cmap = plt.colormaps["tab20"](np.linspace(0, 1, len(branches_3d)))[:, :3]
    for i, b3d in enumerate(branches_3d):
        if len(b3d) < 2:
            continue
        b3d = np.array(b3d)
        pts.extend(b3d)
        lines.extend([[idx + j, idx + j + 1] for j in range(len(b3d) - 1)])
        colors.extend([cmap[i]] * (len(b3d) - 1))
        idx += len(b3d)
    lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines),
    )
    lineset.colors = o3d.utility.Vector3dVector(colors)
    spheres = [
        o3d.geometry.TriangleMesh.create_sphere(0.002)
        .translate(c)
        .paint_uniform_color([1, 0.2, 0])
        for c in node_coords_3d
    ]
    logger.info(
        f"Prepared Open3D geometry: {len(pts)} points, {len(lines)} lines, {len(spheres)} nodes"
    )
    return [lineset], spheres


def make_graph_lineset(node_coords_3d, branches, nodes, eps=3):
    node_coords_2d = np.array(nodes)
    node_tree = cKDTree(node_coords_2d)
    pairs = set()
    for branch in branches:
        i0 = node_tree.query(branch[0], k=1)[1]
        i1 = node_tree.query(branch[-1], k=1)[1]
        if i0 != i1:
            pairs.add(tuple(sorted((i0, i1))))
    lines = [list(pair) for pair in pairs]
    points = np.array(node_coords_3d)
    color = np.array([[0.0, 0.0, 0.0] for _ in lines])
    graph_lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    graph_lineset.colors = o3d.utility.Vector3dVector(color)
    return graph_lineset


def toggle_plane(vis, plane, show_plane):
    if show_plane["visible"]:
        vis.remove_geometry(plane, reset_bounding_box=False)
    else:
        vis.add_geometry(plane, reset_bounding_box=False)
    show_plane["visible"] = not show_plane["visible"]
    return False


def toggle_graph(vis, graph_lineset, show_graph):
    if show_graph["visible"]:
        vis.remove_geometry(graph_lineset, reset_bounding_box=False)
    else:
        vis.add_geometry(graph_lineset, reset_bounding_box=False)
    show_graph["visible"] = not show_graph["visible"]
    return False


def toggle_nodes(vis, o3d_nodes, show_nodes):
    if show_nodes["visible"]:
        for node in o3d_nodes:
            vis.remove_geometry(node, reset_bounding_box=False)
    else:
        for node in o3d_nodes:
            vis.add_geometry(node, reset_bounding_box=False)
    show_nodes["visible"] = not show_nodes["visible"]
    return False


def toggle_skeletons(vis, o3d_branches, show_skeleton):
    if show_skeleton["visible"]:
        for ls in o3d_branches:
            vis.remove_geometry(ls, reset_bounding_box=False)
    else:
        for ls in o3d_branches:
            vis.add_geometry(ls, reset_bounding_box=False)
    show_skeleton["visible"] = not show_skeleton["visible"]
    return False


def geodesic_skeletonization(mask):
    # Step 1: find domain edges (design contour)
    contour = find_boundaries(mask, mode="outer").astype(np.uint8)
    # Step 2: find domain boundary (no load edges in OT, usually just boundary)
    edges = cv2.Canny(mask.astype(np.uint8) * 255, 100, 200) // 255
    # Step 3: geodesic distance from boundary
    geo_dist = np.full_like(mask, np.inf, dtype=np.float32)
    geo_dist[mask.astype(bool)] = distance_transform_edt(~edges)[mask.astype(bool)]
    # Step 4: Add design contour (watershed "lake filling")
    map2 = geo_dist + contour.astype(np.float32)
    markers, _ = nd_label(mask)
    map3 = watershed(map2, markers=markers, mask=mask)
    # Step 5: Skeleton = watershed ridges (map3==0) minus outer contour
    skeleton = thin((1 - map3 == 0).astype(np.uint8))
    return skeleton, edges, contour, geo_dist, map2, map3


def main():
    """
    Full pipeline: loads cloud, extracts main plane, projects to 2D,
    skeletonizes, builds graph and restores all in 3D.
    Logs all major steps and runtime.
    """
    t0 = time.time()
    cloud_path = ".data_clouds/farm1/cloud3d_iter4.ply"
    logger.info(f"Loading point cloud: {cloud_path}")
    pcd_tcp = load_point_cloud(cloud_path)

    # Bounding box crop (adjust as needed)
    bbox_points = np.array(
        [
            [-0.57, -0.34, 0.46],
            [-0.57, -0.24, 0.27],
            [-0.38, -0.34, 0.27],
            [-0.38, -0.34, 0.46],
        ]
    )
    bbox_points[1:4, 1] += 0.2
    voxel_size = 0.001
    nb_neighbors = 30
    std_ratio = 2.0
    distance_threshold = 0.004
    ransac_n = 3

    pcd_tcp_down = pcd_tcp.voxel_down_sample(voxel_size=voxel_size)
    logger.info(f"Downsampled to {len(pcd_tcp_down.points)} points")
    box = o3d.geometry.PointCloud()
    box.points = o3d.utility.Vector3dVector(bbox_points)
    cropped = pcd_tcp_down.crop(box.get_axis_aligned_bounding_box())
    logger.info(f"Cropped to bbox: {len(cropped.points)} points")
    pcd_clean, _ = cropped.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    logger.info(f"After outlier removal: {len(pcd_clean.points)} points")

    # Plane segmentation (RANSAC)
    _, inliers = pcd_clean.segment_plane(
        distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=500
    )
    plane = pcd_clean.select_by_index(inliers)
    logger.info(f"Main plane: {len(plane.points)} points")
    points = np.asarray(plane.points)
    center = np.mean(points, axis=0)
    _, _, vt = np.linalg.svd(points - center, full_matrices=False)
    basis = vt[:2]  # (2, 3) PCA basis

    # 2D mask and skeletonization
    mask, img_xy = preprocess_mask(points, center, basis)

    skel, *_ = geodesic_skeletonization(mask)
    logger.info("Skeletonization complete")
    nodes = extract_nodes(skel)
    branches, node_neighbors = skeleton_branches(skel, nodes)

    # Map to 3D (branch and node positions)
    img_xy_pix = np.stack([img_xy[:, 1], img_xy[:, 0]], axis=1)
    branches_3d = skeleton_branches_to_3d(branches, img_xy_pix, points)
    node_coords_3d = nodes2d_to_3d(nodes, img_xy_pix, points)
    o3d_branches, o3d_nodes = make_o3d_lineset(branches_3d, node_coords_3d)
    graph_lineset = make_graph_lineset(node_coords_3d, branches, nodes)

    t1 = time.time()
    logger.success(f"Finished in {t1-t0:.2f}s")

    # Visualize only after all computation
    with SuppressO3DInfo():
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
        vis.add_geometry(plane)
        for node in o3d_nodes:
            vis.add_geometry(node)
        for ls in o3d_branches:
            vis.add_geometry(ls)
        show_plane = {"visible": True}
        show_graph = {"visible": False}
        show_nodes = {"visible": True}
        show_skeleton = {"visible": True}
        vis.register_key_callback(
            ord("G"), partial(toggle_plane, plane=plane, show_plane=show_plane)
        )
        vis.register_key_callback(
            ord("H"),
            partial(toggle_graph, graph_lineset=graph_lineset, show_graph=show_graph),
        )
        vis.register_key_callback(
            ord("N"), partial(toggle_nodes, o3d_nodes=o3d_nodes, show_nodes=show_nodes)
        )
        vis.register_key_callback(
            ord("J"),
            partial(
                toggle_skeletons, o3d_branches=o3d_branches, show_skeleton=show_skeleton
            ),
        )
        vis.run()
        vis.destroy_window()


if __name__ == "__main__":
    main()
