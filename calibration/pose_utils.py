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
    R_d2c: np.ndarray,
    t_d2c: np.ndarray,
) -> tuple[float, float]:
    """Project an RGB pixel to depth image coordinates."""
    logger.info(f"RGB pixel: (x={x}, y={y})")
    pt_rgb = np.array([x, y, 1.0])
    logger.info(f"pt_rgb (homogeneous): {pt_rgb}")

    ray_rgb = np.linalg.inv(K_rgb) @ pt_rgb
    logger.info(f"Ray in RGB normalized: {ray_rgb}")

    # !!! R_d2c = depth->color (или color->depth? Проверь смысл!)
    ray_depth = R_d2c @ ray_rgb + t_d2c
    logger.info(f"Ray in depth normalized: {ray_depth}")

    pt_depth = K_depth @ (ray_depth / ray_depth[2])
    logger.info(f"Projected to depth pixel: {pt_depth}")

    return float(pt_depth[0]), float(pt_depth[1])


def get_3d_points_from_depth(
    corners: np.ndarray,
    depth: np.ndarray,
    K_rgb: np.ndarray,
    K_depth: np.ndarray,
    R_d2c: np.ndarray,
    t_d2c: np.ndarray,
    depth_scale: float = 0.001,
    logger=None,
) -> np.ndarray:
    points = []
    for i, pt in enumerate(corners.reshape(-1, 2)):
        u_rgb, v_rgb = pt
        if logger and i < 2:  # лог только по первым двум точкам (или убрать i<2)
            logger.info(
                f"[get_3d_points_from_depth] #{i}: pixel RGB=({u_rgb:.2f}, {v_rgb:.2f})"
            )
        # Проекция в depth-координаты
        px_d, py_d = rgb_to_depth_pixel(u_rgb, v_rgb, K_rgb, K_depth, R_d2c, t_d2c)
        xi, yi = int(round(px_d)), int(round(py_d))
        if logger and i < 2:
            logger.info(
                f"    Projected to depth pixel=({px_d:.2f}, {py_d:.2f}), integer=({xi},{yi})"
            )
        if 0 <= yi < depth.shape[0] and 0 <= xi < depth.shape[1]:
            d = depth[yi, xi]
        else:
            d = 0
        if logger and i < 2:
            logger.info(f"    Depth at ({xi},{yi}) = {d}")
        if d == 0:
            points.append([np.nan, np.nan, np.nan])
            if logger and i < 2:
                logger.info("    Invalid depth, skip")
            continue
        Z = float(d) * depth_scale + 0.0001
        X = (px_d - K_depth[0, 2]) * Z / K_depth[0, 0]
        Y = (py_d - K_depth[1, 2]) * Z / K_depth[1, 1]
        P_depth = np.array([X, Y, Z], dtype=np.float64)
        if logger and i < 2:
            logger.info(f"    P_depth={P_depth}")
        # Переводим в RGB-камеру:
        P_rgb = R_d2c @ P_depth + t_d2c.flatten()
        if logger and i < 2:
            logger.info(f"    P_rgb={P_rgb}")
        points.append(P_rgb)
    return np.array(points)


def rigid_transform_3D(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return rigid transform aligning `A` to `B`."""
    logger.info(f"---[rigid_transform_3D] Input A shape: {A.shape}, B shape: {B.shape}")
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    logger.info(
        f"[rigid_transform_3D] centroid_A: {centroid_A}, centroid_B: {centroid_B}"
    )
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.T @ BB
    logger.info(f"[rigid_transform_3D] H matrix:\n{H}")
    U, S, Vt = np.linalg.svd(H)
    logger.info(f"[rigid_transform_3D] SVD U:\n{U}\nS: {S}\nVt:\n{Vt}")
    R_ = Vt.T @ U.T
    det_R = np.linalg.det(R_)
    logger.info(f"[rigid_transform_3D] Determinant of R: {det_R}")
    if det_R < 0:
        logger.info(
            "[rigid_transform_3D] Reflection detected. Correcting for right-handedness."
        )
        Vt[-1, :] *= -1
        R_ = Vt.T @ U.T
    t_ = centroid_B - R_ @ centroid_A
    logger.info(f"[rigid_transform_3D] Resulting R:\n{R_}")
    logger.info(f"[rigid_transform_3D] Resulting t: {t_}")
    return R_, t_


def solve_pnp_obj_to_img(
    obj_pts: np.ndarray,
    img_pts: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Solve PnP from object points to image points."""
    logger.info(
        f"---[solve_pnp_obj_to_img] obj_pts shape: {obj_pts.shape}, img_pts shape: {img_pts.shape}"
    )
    logger.info(f"[solve_pnp_obj_to_img] K:\n{K}\ndist: {dist}")
    retval, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )
    logger.info(f"[solve_pnp_obj_to_img] solvePnP returned: {retval}")
    if not retval:
        logger.info("[solve_pnp_obj_to_img] PnP failed!")
        return None, None
    logger.info(
        f"[solve_pnp_obj_to_img] rvec: {rvec.flatten()}, tvec: {tvec.flatten()}"
    )
    R_ = cv2.Rodrigues(rvec)[0]
    logger.info(f"[solve_pnp_obj_to_img] Rotation matrix R:\n{R_}")
    t_ = tvec.flatten()
    logger.info(f"[solve_pnp_obj_to_img] Translation t: {t_}")
    return R_, t_
