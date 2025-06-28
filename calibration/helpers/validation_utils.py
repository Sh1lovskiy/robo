from __future__ import annotations

import os
import json
import shutil
import threading
import time
from typing import Iterable, Tuple
from scipy.spatial.transform import Rotation as R

import cv2
import numpy as np

from utils.keyboard import GlobalKeyListener, TerminalEchoSuppressor
from utils.logger import Logger, LoggerType


def load_image_paths(images_dir: str) -> list[str]:
    """Return sorted list of image paths inside ``images_dir``."""
    return sorted(
        [
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )


def euler_to_matrix(
    rx: float, ry: float, rz: float, degrees: bool = True
) -> np.ndarray:
    """Convert Euler angles to rotation matrix."""
    if degrees:
        angles = np.deg2rad([rx, ry, rz])
    else:
        angles = [rx, ry, rz]
    return R.from_euler("xyz", angles).as_matrix()


def detect_board_corners(
    img_path: str,
    board: cv2.aruco_CharucoBoard,
    dictionary: cv2.aruco_Dictionary,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    min_corners: int = 4,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Detect Charuco board corners and return left-top/right-bottom points.

    Args:
        img_path: Image file containing the printed board.
        board: Charuco board model used for pose estimation.
        dictionary: ArUco dictionary for marker detection.
        camera_matrix: Intrinsic matrix ``K``.
        dist_coeffs: Distortion coefficients.
        min_corners: Minimum number of detected corners required.

    Returns:
        A tuple ``(lt, rb)`` with the corner positions in camera frame if the
        board was detected, otherwise ``(None, None)``.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    if ids is None or len(ids) < min_corners:
        return None, None
    _, char_corners, char_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, board
    )
    if (
        char_corners is None
        or char_ids is None
        or len(char_ids) < min_corners
        or len(char_corners) != len(char_ids)
    ):
        return None, None

    char_corners = char_corners.squeeze(1)
    char_ids = char_ids.flatten()
    all_obj_pts = board.getChessboardCorners()
    obj_pts = all_obj_pts[char_ids]
    img_pts = char_corners

    if len(obj_pts) < 4 or len(img_pts) < 4:
        return None, None

    method = cv2.SOLVEPNP_IPPE_SQUARE if len(obj_pts) == 4 else cv2.SOLVEPNP_ITERATIVE

    retval, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, camera_matrix, dist_coeffs, flags=method
    )

    R, _ = cv2.Rodrigues(rvec)
    lt = R @ obj_pts[0].reshape(3, 1) + tvec
    rb = R @ obj_pts[-1].reshape(3, 1) + tvec
    return lt.flatten(), rb.flatten()


def analyze_handeye_residuals(
    cam_corners: Iterable[Tuple[np.ndarray, np.ndarray]],
    R_cam2base: np.ndarray,
    t_cam2base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project detected corners to base frame using hand-eye result.

    Args:
        cam_corners: Iterable of ``(lt, rb)`` corner tuples in camera frame.
        R_cam2base: Rotation from camera to robot base.
        t_cam2base: Translation from camera to robot base.

    Returns:
        Arrays ``(lt_base_pred, rb_base_pred)`` with predicted base frame
        coordinates for each detected corner.
    """
    lt_base_pred, rb_base_pred = [], []
    for lt_cam, rb_cam in cam_corners:
        lt_base = R_cam2base @ lt_cam + t_cam2base
        rb_base = R_cam2base @ rb_cam + t_cam2base
        lt_base_pred.append(lt_base)
        rb_base_pred.append(rb_base)
    return np.stack(lt_base_pred), np.stack(rb_base_pred)


def error_vs_reference(
    lt_pred: np.ndarray,
    rb_pred: np.ndarray,
    ref_lt: Iterable[float],
    ref_rb: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute distance errors versus reference positions."""
    lt_errs = np.linalg.norm(lt_pred - ref_lt, axis=1)
    rb_errs = np.linalg.norm(rb_pred - ref_rb, axis=1)
    return lt_errs, rb_errs


def error_vs_mean(
    lt_pred: np.ndarray, rb_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return errors relative to mean position and the means themselves."""
    lt_mean = lt_pred.mean(axis=0)
    rb_mean = rb_pred.mean(axis=0)
    lt_errs = np.linalg.norm(lt_pred - lt_mean, axis=1)
    rb_errs = np.linalg.norm(rb_pred - rb_mean, axis=1)
    return lt_errs, rb_errs, lt_mean, rb_mean


def filter_by_percentile(
    errors: np.ndarray,
    img_paths: list[str],
    *,
    percentile: float = 80,
    logger: LoggerType | None = None,
    title: str = "Filter by percentile",
) -> tuple[list[str], list[str]]:
    """Split image paths by error percentile threshold."""
    threshold = np.percentile(errors, percentile)
    keep_idx = [i for i, e in enumerate(errors) if e > threshold]
    drop_idx = [i for i, e in enumerate(errors) if e <= threshold]
    keep_paths = [img_paths[i] for i in keep_idx]
    drop_paths = [img_paths[i] for i in drop_idx]
    if logger:
        logger.info(
            f"{title}: keeping {len(keep_paths)} frames (<= {percentile}th perc., "
            f"thr={threshold:.5f} m)"
        )
        logger.info("KEPT:")
        for p in keep_paths:
            logger.info(f" {os.path.basename(p)}")
        logger.info(f"TO DROP ({len(drop_paths)} frames):")
        for p in drop_paths:
            logger.info(f" {os.path.basename(p)}")
    return keep_paths, drop_paths


def move_images(
    img_paths: Iterable[str],
    images_dir: str,
    drop_dir: str,
    logger: LoggerType,
) -> None:
    """Move files to a ``drop_imgs`` subfolder, skipping existing ones."""
    os.makedirs(drop_dir, exist_ok=True)
    for p in img_paths:
        fname = os.path.basename(p)
        dst = os.path.join(drop_dir, fname)
        if os.path.exists(dst):
            logger.warning(f"File already exists in drop_imgs: {dst}, skipping")
            continue
        shutil.move(p, dst)
        logger.info(f"Moved {fname} → drop_imgs/")


def move_poses_for_dropped_images(
    drop_img_paths: Iterable[str],
    images_dir: str,
    logger: LoggerType,
) -> None:
    """Move corresponding robot poses for dropped images."""
    poses_json = os.path.join(images_dir, "poses.json")
    drop_dir = os.path.join(images_dir, "drop_imgs")
    drop_poses_json = os.path.join(drop_dir, "poses.json")
    if not os.path.isfile(poses_json):
        logger.warning(f"No poses.json found in {images_dir}, skipping poses move.")
        return
    with open(poses_json, "r") as f:
        all_poses = json.load(f)
    drop_filenames = {
        os.path.splitext(os.path.basename(p))[0].split("_")[0] for p in drop_img_paths
    }
    drop_poses = {k: v for k, v in all_poses.items() if k in drop_filenames}
    keep_poses = {k: v for k, v in all_poses.items() if k not in drop_filenames}
    with open(poses_json, "w") as f:
        json.dump(keep_poses, f, indent=2)
    logger.info(f"Removed {len(drop_poses)} poses from {poses_json}")
    if os.path.isfile(drop_poses_json):
        with open(drop_poses_json, "r") as f:
            drop_file_poses = json.load(f)
    else:
        drop_file_poses = {}
    drop_file_poses.update(drop_poses)
    with open(drop_poses_json, "w") as f:
        json.dump(drop_file_poses, f, indent=2)
    logger.info(f"Added {len(drop_poses)} poses to {drop_poses_json}")


def ask_confirm_keyboard(logger: LoggerType, msg: str) -> bool:
    """Prompt user via hotkeys and return True if confirmed."""
    confirmed = {"value": False}
    done = threading.Event()

    def on_yes() -> None:
        """Set confirmation flag and stop waiting."""
        confirmed["value"] = True
        done.set()

    def on_any() -> None:
        """Stop waiting without confirming."""
        done.set()

    hotkeys = {
        "y": on_yes,
        "n": on_any,
        "<enter>": on_any,
        "<esc>": on_any,
        "<space>": on_any,
    }
    logger.info(msg + " [y=confirm, any other=skip]")
    suppressor = TerminalEchoSuppressor()
    suppressor.start()
    listener = GlobalKeyListener(hotkeys, suppress=True)
    listener.start()
    try:
        for _ in range(120):
            if done.is_set():
                break
            time.sleep(0.1)
    finally:
        listener.stop()
        suppressor.stop()
    return confirmed["value"]


def plot_errors(
    errs1: np.ndarray,
    errs2: np.ndarray,
    label1: str,
    label2: str,
    fname: str,
) -> None:
    """Save a histogram of residuals for visual inspection."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.hist(errs1, bins=15, alpha=0.6, label="LT", color="blue")
    plt.hist(errs2, bins=15, alpha=0.6, label="RB", color="red")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title(f"Hand-Eye validation: {label1} vs {label2}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def project_board_via_handeye(
    board_pts_base: np.ndarray,
    R_base2cam: np.ndarray,
    t_base2cam: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    """Project board model into the image using hand-eye transform."""
    pts_cam = (R_base2cam.T @ (board_pts_base.T - t_base2cam[:, None])).T
    img_points, _ = cv2.projectPoints(
        pts_cam, np.zeros((3, 1)), np.zeros((3, 1)), camera_matrix, dist_coeffs
    )
    return img_points


def validate_handeye_calibration(
    board_pts_base: np.ndarray,
    R_base2cam: np.ndarray,
    t_base2cam: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    detected_corners: np.ndarray,
    logger: LoggerType,
) -> float:
    """Compute reprojection error for hand-eye calibration."""
    projected_corners = project_board_via_handeye(
        board_pts_base, R_base2cam, t_base2cam, camera_matrix, dist_coeffs
    )
    if detected_corners.shape[1] == 3:
        detected_corners_2d, _ = cv2.projectPoints(
            detected_corners,
            np.zeros((3, 1)),
            np.zeros((3, 1)),
            camera_matrix,
            dist_coeffs,
        )
        detected_corners_2d = detected_corners_2d.squeeze()
    else:
        detected_corners_2d = detected_corners
    if detected_corners_2d.shape != projected_corners.shape:
        logger.warning(
            f"Mismatch in number of corners: detected {detected_corners_2d.shape[0]}, projected {projected_corners.shape[0]}"
        )
        detected_corners_2d = detected_corners_2d[: projected_corners.shape[0]]
        projected_corners = projected_corners[: detected_corners_2d.shape[0]]
    error = np.linalg.norm(detected_corners_2d - projected_corners.squeeze(), axis=1)
    mean_error = float(error.mean())
    logger.info(f"Mean reprojection error: {mean_error:.3f} pixels")
    return mean_error
