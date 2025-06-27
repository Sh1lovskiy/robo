# calibration/helpers/pose_utils.py
"""Helper to load robot poses I/O."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import json
import matplotlib.pyplot as plt
import numpy as np

from utils.config import Config
from utils.logger import Logger, LoggerType
from .validation_utils import euler_to_matrix


def load_camera_params(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load camera matrix and distortion coefficients from OpenCV XML/YAML.

    Args:
        filename: Path to ``.xml`` or ``.yml`` file created by OpenCV
            calibration routines.

    Returns:
        ``(camera_matrix, dist_coeffs)`` as NumPy arrays.
    """
    fs = cv2.FileStorage(str(filename), cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("dist_coeffs").mat()
    fs.release()
    return camera_matrix, dist_coeffs


def save_camera_params_xml(
    filename: str,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> None:
    """Save camera calibration to an OpenCV XML/YAML file.

    Args:
        filename: Output XML file.
        camera_matrix: 3x3 intrinsic matrix.
        dist_coeffs: Distortion coefficients vector.
    """
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    fs = cv2.FileStorage(str(filename), cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", camera_matrix)
    fs.write("dist_coeffs", dist_coeffs)
    fs.release()


def save_camera_params_txt(
    filename: str,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rms: float | None = None,
) -> None:
    """Save camera calibration to a plain text file.

    Args:
        filename: Output text path.
        camera_matrix: Intrinsic matrix to store.
        dist_coeffs: Distortion coefficients to store.
        rms: Optional RMS error value written as a header.
    """
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as f:
        if rms is not None:
            f.write(f"RMS Error: {rms:.6f}\n")
        f.write("camera_matrix =\n")
        np.savetxt(f, camera_matrix, fmt="%.10f")
        f.write("dist_coeffs =\n")
        np.savetxt(f, dist_coeffs.reshape(1, -1), fmt="%.10f")


class JSONPoseLoader:
    """
    Loads robot poses for hand-eye calibration from a JSON file.
    Expects keys:
        - "robot_tcp_pose": [x, y, z, rx, ry, rz] (angles in degrees or radians)
    """

    @staticmethod
    def load_poses(json_file: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Return rotation and translation lists from ``json_file``.

        The JSON is expected to map IDs to ``{"tcp_coords": [x, y, z, rx, ry, rz]}``.

        Args:
            json_file: Path to the recorded poses file.

        Returns:
            Tuple ``(rotations, translations)`` where each is a list of NumPy
            arrays describing the gripper pose relative to the robot base.
        """
        with open(json_file, "r") as f:
            data = json.load(f)

        Rs, ts = [], []
        for pose in data.values():
            tcp_pose = pose["tcp_coords"]  # [x, y, z, rx, ry, rz]
            t = np.array(tcp_pose[:3], dtype=np.float64) / 1000.0  # mm → m
            rx, ry, rz = tcp_pose[3:]
            R_mat = euler_to_matrix(rx, ry, rz, degrees=True)
            Rs.append(R_mat)
            ts.append(t)
        return Rs, ts


@dataclass
class ExtractionParams:
    """Thresholds and flags controlling pose extraction."""

    min_corners: int = 4
    visualize: bool = False
    analyze_corners: bool = False
    outlier_std: float = 2.0


@dataclass
class ExtractionResult:
    """Results returned by :func:`extract_charuco_poses`."""

    rotations: List[np.ndarray]
    translations: List[np.ndarray]
    valid_paths: List[str]
    all_paths: List[str]
    stats: dict[str, dict[str, np.ndarray]]
    outliers: List[int]


def _load_params() -> ExtractionParams:
    """Load :class:`ExtractionParams` from global config."""

    cfg = Config.get("charuco")
    return ExtractionParams(
        min_corners=cfg.get("min_corners", 4),
        visualize=cfg.get("visualize", False),
        analyze_corners=cfg.get("analyze_corners", False),
        outlier_std=float(cfg.get("outlier_std", 2.0)),
    )


def _list_images(images_dir: str) -> List[str]:
    """Return sorted image file paths from ``images_dir``."""
    return sorted(
        [
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )


def _estimate_pose(
    img: np.ndarray,
    board: cv2.aruco_CharucoBoard,
    dictionary: cv2.aruco_Dictionary,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    params: ExtractionParams,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Estimate board pose in the camera frame using OpenCV."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    if ids is None or len(ids) < max(6, params.min_corners):
        return None
    _, char_corners, char_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, board
    )
    if (
        char_corners is None
        or char_ids is None
        or len(char_ids) < max(6, params.min_corners)
    ):
        return None

    cc = np.ascontiguousarray(char_corners.astype(np.float32))
    ci = np.ascontiguousarray(char_ids.astype(np.int32))
    if cc.shape[0] != ci.shape[0]:
        return None
    rvec_init = np.zeros((3, 1), dtype=np.float64)
    tvec_init = np.zeros((3, 1), dtype=np.float64)
    retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        cc,
        ci,
        board,
        camera_matrix,
        dist_coeffs,
        rvec_init,
        tvec_init,
    )
    if not retval:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.flatten()


def _collect_corner_stats(
    Rs: List[np.ndarray],
    ts: List[np.ndarray],
    board: cv2.aruco_CharucoBoard,
    params: ExtractionParams,
    logger: LoggerType | None,
) -> tuple[dict[str, dict[str, np.ndarray]], List[int]]:
    """Compute corner statistics and remove outlier frames."""
    if not params.analyze_corners:
        return {}, []
    obj_pts = board.getChessboardCorners()
    lt = []
    rb = []
    for R, t in zip(Rs, ts):
        lt.append((R @ obj_pts[0].reshape(3, 1) + t.reshape(3, 1)).flatten())
        rb.append((R @ obj_pts[-1].reshape(3, 1) + t.reshape(3, 1)).flatten())
    lt_arr = np.stack(lt)
    rb_arr = np.stack(rb)
    stats = {
        "lt": {"mean": lt_arr.mean(axis=0), "std": lt_arr.std(axis=0)},
        "rb": {"mean": rb_arr.mean(axis=0), "std": rb_arr.std(axis=0)},
    }
    for name, stat in stats.items():
        if logger:
            logger.info(
                f"{name.upper()} mean: {stat['mean'].round(4)} std: {stat['std'].round(4)}"
            )
    mask = np.ones(len(Rs), dtype=bool)
    for arr, stat in [(lt_arr, stats["lt"]), (rb_arr, stats["rb"])]:
        mask &= np.all(
            np.abs(arr - stat["mean"]) <= params.outlier_std * stat["std"], axis=1
        )
    outliers = [i for i, good in enumerate(mask) if not good]
    Rs[:] = [R for i, R in enumerate(Rs) if mask[i]]
    ts[:] = [t for i, t in enumerate(ts) if mask[i]]
    if params.visualize:
        plt.figure(figsize=(8, 6))
        plt.scatter(lt_arr[:, 0], lt_arr[:, 1], c="blue", label="Left Top")
        plt.scatter(rb_arr[:, 0], rb_arr[:, 1], c="red", label="Right Bottom")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("charuco_corners_distribution.png")
        plt.close()
    return stats, outliers


def extract_charuco_poses(
    images_dir: str,
    board: cv2.aruco_CharucoBoard,
    dictionary: cv2.aruco_Dictionary,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    *,
    logger: LoggerType | None = None,
    params: ExtractionParams | None = None,
) -> ExtractionResult:
    """Run pose extraction for all images in a directory."""
    params = params or _load_params()
    logger = logger or Logger.get_logger("calibration.pose_extractor")
    image_paths = _list_images(images_dir)
    Rs: List[np.ndarray] = []
    ts: List[np.ndarray] = []
    valid_paths: List[str] = []
    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"Cannot read image: {img_path}")
            continue
        pose = _estimate_pose(
            img, board, dictionary, camera_matrix, dist_coeffs, params
        )
        if pose is None:
            logger.warning(
                f"Charuco pose not found for image: {os.path.basename(img_path)}"
            )
            continue
        R, t = pose
        Rs.append(R)
        ts.append(t)
        valid_paths.append(img_path)
        if params.visualize:
            cv2.aruco.drawDetectedMarkers(img, None, None)
            cv2.imshow("charuco pose", img)
            cv2.waitKey(200)
    if params.visualize:
        cv2.destroyAllWindows()
    stats, outliers = _collect_corner_stats(Rs, ts, board, params, logger)
    logger.info(f"Extracted {len(Rs)} poses after filtering")
    return ExtractionResult(Rs, ts, valid_paths, image_paths, stats, outliers)
