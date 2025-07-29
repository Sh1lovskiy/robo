"""
Modular pipeline for extracting graph structure from 3D point cloud planes
via 2D skeletonization. All steps are split into independent methods.
"""

import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from functools import partial
from scipy.ndimage import distance_transform_edt, label as nd_label
from scipy.spatial import cKDTree
from skimage.morphology import thin
from skimage.segmentation import watershed, find_boundaries

from utils.logger import Logger, SuppressO3DInfo

logger = Logger.get_logger("skelet")


# 1. --- IO, visualization, helpers --- #


def show_mask(mask, title=""):
    plt.figure(figsize=(4, 4))
    plt.imshow(mask, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


def load_point_cloud(path):
    pcd = o3d.io.read_point_cloud(path)
    pcd.estimate_normals()
    logger.info(f"Loaded point cloud: {path} ({len(pcd.points)} points)")
    return pcd


def crop_and_downsample(
    pcd, bbox_points, voxel_size=0.001, nb_neighbors=30, std_ratio=2.0
):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    box = o3d.geometry.PointCloud()
    box.points = o3d.utility.Vector3dVector(bbox_points)
    cropped = pcd_down.crop(box.get_axis_aligned_bounding_box())
    logger.info(f"Cropped to bbox: {len(cropped.points)} points")
    pcd_clean, _ = cropped.remove_statistical_outlier(nb_neighbors, std_ratio)
    logger.info(f"After outlier removal: {len(pcd_clean.points)} points")
    return pcd_clean


def get_main_plane(pcd, distance_threshold=0.004, ransac_n=3, num_iterations=500):
    _, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    plane = pcd.select_by_index(inliers)
    logger.info(f"Main plane: {len(plane.points)} points")
    return plane


# 2. --- Projections, skeletonization, graph extraction --- #


def get_pca_basis(points):
    center = np.mean(points, axis=0)
    _, _, vt = np.linalg.svd(points - center, full_matrices=False)
    basis = vt[:2]
    return center, basis


def preprocess_mask(points, center, basis, img_res=1024):
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


def geodesic_skeletonization(mask):
    contour = find_boundaries(mask, mode="outer").astype(np.uint8)
    edges = cv2.Canny(mask.astype(np.uint8) * 255, 100, 200) // 255
    geo_dist = np.full_like(mask, np.inf, dtype=np.float32)
    geo_dist[mask.astype(bool)] = distance_transform_edt(~edges)[mask.astype(bool)]
    map2 = geo_dist + contour.astype(np.float32)
    markers, _ = nd_label(mask)
    map3 = watershed(map2, markers=markers, mask=mask)
    skeleton = thin((1 - map3 == 0).astype(np.uint8))
    return skeleton


def extract_nodes(skel):
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


# 3. --- Mapping 2D nodes and branches to 3D --- #


def nodes2d_to_3d(nodes, img_xy, points):
    tree = cKDTree(img_xy)
    node_arr = np.array(nodes)
    dists, idxs = tree.query(node_arr, k=1)
    coords_3d = points[idxs]
    logger.info(f"Mapped {len(nodes)} nodes to 3D coordinates")
    return coords_3d


def skeleton_branches_to_3d(branches, img_xy, points_3d):
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


# 4. --- Visualization builders --- #


def make_o3d_lineset(branches_3d, node_coords_3d):
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


# 5. --- Key callbacks --- #


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


# 6. --- Pipeline steps as methods (main logic building blocks) --- #


def run_pipeline(
    cloud_path,
    bbox_points,
    voxel_size=0.001,
    nb_neighbors=30,
    std_ratio=2.0,
    distance_threshold=0.004,
    ransac_n=3,
    img_res=1024,
):
    logger.info(f"Loading point cloud: {cloud_path}")
    pcd_tcp = load_point_cloud(cloud_path)
    pcd_clean = crop_and_downsample(
        pcd_tcp, bbox_points, voxel_size, nb_neighbors, std_ratio
    )
    plane = get_main_plane(pcd_clean, distance_threshold, ransac_n)
    points = np.asarray(plane.points)
    center, basis = get_pca_basis(points)
    mask, img_xy = preprocess_mask(points, center, basis, img_res)
    skel = geodesic_skeletonization(mask)
    nodes = extract_nodes(skel)
    branches, node_neighbors = skeleton_branches(skel, nodes)
    img_xy_pix = np.stack([img_xy[:, 1], img_xy[:, 0]], axis=1)
    branches_3d = skeleton_branches_to_3d(branches, img_xy_pix, points)
    node_coords_3d = nodes2d_to_3d(nodes, img_xy_pix, points)
    o3d_branches, o3d_nodes = make_o3d_lineset(branches_3d, node_coords_3d)
    graph_lineset = make_graph_lineset(node_coords_3d, branches, nodes)
    logger.success("Pipeline finished.")
    return {
        "plane": plane,
        "branches_3d": branches_3d,
        "node_coords_3d": node_coords_3d,
        "o3d_branches": o3d_branches,
        "o3d_nodes": o3d_nodes,
        "graph_lineset": graph_lineset,
        "mask": mask,
        "skel": skel,
        "nodes": nodes,
        "branches": branches,
        "node_neighbors": node_neighbors,
    }


def run_visualization(plane, o3d_branches, o3d_nodes, graph_lineset):
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


# 7. --- Main entry --- #


def main():
    t0 = time.time()
    cloud_path = ".data_clouds/farm1/cloud3d_iter4.ply"
    bbox_points = np.array(
        [
            [-0.57, -0.34, 0.46],
            [-0.57, -0.24, 0.27],
            [-0.38, -0.34, 0.27],
            [-0.38, -0.34, 0.46],
        ]
    )
    bbox_points[1:4, 1] += 0.2

    results = run_pipeline(cloud_path, bbox_points)
    t1 = time.time()
    logger.success(f"Finished in {t1-t0:.2f}s")

    run_visualization(
        results["plane"],
        results["o3d_branches"],
        results["o3d_nodes"],
        results["graph_lineset"],
    )


if __name__ == "__main__":
    main()
