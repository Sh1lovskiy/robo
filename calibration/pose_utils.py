"""Helper functions for Charuco pose estimation using depth."""

import numpy as np
import cv2

from utils.logger import Logger

logger = Logger.get_logger("pose_utils.debug")


def rgb_to_depth_pixel(
    x: float,
    y: float,
    K_rgb: np.ndarray,
    K_depth: np.ndarray,
    R_c2d: np.ndarray,
    t_c2d: np.ndarray,
    *,
    idx: int | None = None,
    logger: Logger | None = None,
) -> tuple[float, float]:
    """Project an RGB pixel to depth image coordinates."""
    log = logger or globals()["logger"]
    if idx == 0:
        log.debug(f"[rgb_to_depth_pixel] K_rgb:\n{K_rgb}")
        log.debug(f"[rgb_to_depth_pixel] K_depth:\n{K_depth}")
        log.debug(f"[rgb_to_depth_pixel] R_c2d:\n{R_c2d}\n t_c2d:{t_c2d}")
    log.debug(f"RGB pixel: (x={x}, y={y})")
    # [x, y] → [x, y, 1]^t
    pt_rgb = np.array([x, y, 1.0])
    log.debug(f"pt_rgb (homogeneous): {pt_rgb}")

    ray_rgb = np.linalg.inv(K_rgb) @ pt_rgb
    log.debug(f"Ray in RGB cam to (x, y) pixel normalized: {ray_rgb}")

    # pt_rgb from colot->depth
    ray_depth = R_c2d @ ray_rgb + t_c2d
    log.debug(f"Ray in depth normalized: {ray_depth}")
    # [x, y, z] to pixel like (X, Y) with normalize K @ (X/Z, Y/Z, 1)
    pt_depth = K_depth @ (ray_depth / ray_depth[2])
    log.debug(f"Projected to depth pixel: {pt_depth}")

    return float(pt_depth[0]), float(pt_depth[1])


def get_3d_points_from_depth(
    corners: np.ndarray,
    depth: np.ndarray,
    K_rgb: np.ndarray,
    K_depth: np.ndarray,
    R_c2d: np.ndarray,
    R_d2c: np.ndarray,
    t_c2d: np.ndarray,
    t_d2c: np.ndarray,
    depth_scale: float = 0.001,
    logger=None,
) -> np.ndarray:
    logger.debug(f"[get_3d_points_from_depth] K_rgb:\n{K_rgb}")
    logger.debug(f"[get_3d_points_from_depth] K_depth:\n{K_depth}")
    logger.debug(f"[get_3d_points_from_depth] R_c2d:\n{R_c2d}")
    logger.debug(f"[get_3d_points_from_depth] t_c2d: {t_c2d.flatten()}")
    points = []
    # (u, v) -> (x, y, z), z = 1 
    for i, pt in enumerate(corners.reshape(-1, 2)):
        u_rgb, v_rgb = pt
        if i == 0:
            logger.debug(
                f"[get_3d_points_from_depth] #{i}: pixel RGB=({u_rgb:.2f}, {v_rgb:.2f})"
            )
        px_d, py_d = rgb_to_depth_pixel(
            u_rgb,
            v_rgb,
            K_rgb,
            K_depth,
            R_c2d,
            t_c2d,
            idx=i,
            logger=logger,
        )
        xi, yi = int(px_d), int(py_d)
        if i == 0:
            logger.debug(
                f"    Projected to depth pixel=({px_d:.2f}, {py_d:.2f}), integer=({xi},{yi})"
            )
        if 0 <= yi < depth.shape[0] and 0 <= xi < depth.shape[1]:
            d = depth[yi, xi]
        else:
            d = 0
        if i == 0:
            logger.debug(f"    Depth at ({xi},{yi}) = {d}")
        if d == 0:
            points.append([np.nan, np.nan, np.nan])
            if i == 0:
                logger.debug("    Invalid depth, skip")
            continue
        Z = float(d) * depth_scale + 0.0001
        X = (px_d - K_depth[0, 2]) * Z / K_depth[0, 0]
        Y = (py_d - K_depth[1, 2]) * Z / K_depth[1, 1]
        P_depth = np.array([X, Y, Z], dtype=np.float64)
        if i == 0:
            logger.debug(f"    P_depth={P_depth}")
        P_rgb = R_d2c @ P_depth + t_d2c.flatten()
        if i == 0:
            logger.debug(f"    P_rgb={P_rgb}")
        points.append(P_rgb)
    return np.array(points)


def rigid_transform_3D(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return rigid transform aligning `A` to `B`."""
    logger.debug(
        f"---[rigid_transform_3D] Input A shape: {A.shape}, B shape: {B.shape}"
    )
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    logger.debug(
        f"[rigid_transform_3D] centroid_A: {centroid_A}, centroid_B: {centroid_B}"
    )
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.T @ BB
    logger.debug(f"[rigid_transform_3D] H matrix:\n{H}")
    U, S, Vt = np.linalg.svd(H)
    logger.debug(f"[rigid_transform_3D] SVD U:\n{U}\nS: {S}\nVt:\n{Vt}")
    R_ = Vt.T @ U.T
    det_R = np.linalg.det(R_)
    logger.debug(f"[rigid_transform_3D] Determinant of R: {det_R}")
    if det_R < 0:
        logger.debug(
            "[rigid_transform_3D] Reflection detected. Correcting for right-handedness."
        )
        Vt[-1, :] *= -1
        R_ = Vt.T @ U.T
    t_ = centroid_B - R_ @ centroid_A
    logger.debug(f"[rigid_transform_3D] Resulting R:\n{R_}")
    logger.debug(f"[rigid_transform_3D] Resulting t: {t_}")
    return R_, t_


def solve_pnp_obj_to_img(
    obj_pts: np.ndarray,
    img_pts: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Solve PnP from object points to image points."""
    logger.debug(
        f"---[solve_pnp_obj_to_img] obj_pts shape: {obj_pts.shape}, img_pts shape: {img_pts.shape}"
    )
    logger.debug(f"[solve_pnp_obj_to_img] K:\n{K}\ndist: {dist}")
    if obj_pts.size > 0:
        logger.debug(
            f"[solve_pnp_obj_to_img] First correspondence obj={obj_pts[0].tolist()}, img={img_pts[0].tolist()}"
        )
    retval, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )
    logger.debug(f"[solve_pnp_obj_to_img] solvePnP returned: {retval}")
    if not retval:
        logger.debug("[solve_pnp_obj_to_img] PnP failed!")
        return None, None
    logger.debug(
        f"[solve_pnp_obj_to_img] rvec: {rvec.flatten()}, tvec: {tvec.flatten()}"
    )
    R_ = cv2.Rodrigues(rvec)[0]
    logger.debug(f"[solve_pnp_obj_to_img] Rotation matrix R:\n{R_}")
    t_ = tvec.flatten()
    logger.debug(f"[solve_pnp_obj_to_img] Translation t: {t_}")
    return R_, t_
