from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils.config import Config
from utils.logger import Logger, LoggerType


@dataclass
class ExtractionParams:
    min_corners: int = 4
    visualize: bool = False
    analyze_corners: bool = False
    outlier_std: float = 2.0


@dataclass
class ExtractionResult:
    rotations: List[np.ndarray]
    translations: List[np.ndarray]
    valid_paths: List[str]
    all_paths: List[str]
    stats: dict[str, dict[str, np.ndarray]]
    outliers: List[int]


def _load_params() -> ExtractionParams:
    cfg = Config.get("charuco")
    return ExtractionParams(
        min_corners=cfg.get("min_corners", 4),
        visualize=cfg.get("visualize", False),
        analyze_corners=cfg.get("analyze_corners", False),
        outlier_std=float(cfg.get("outlier_std", 2.0)),
    )


def _list_images(images_dir: str) -> List[str]:
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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    if ids is None or len(ids) < params.min_corners:
        return None
    _, char_corners, char_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, board
    )
    if char_corners is None or char_ids is None or len(char_ids) < params.min_corners:
        return None
    retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        char_corners,
        char_ids,
        board,
        camera_matrix,
        dist_coeffs,
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
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
    params = params or _load_params()
    logger = logger or Logger.get_logger("calibration.pose_extractor")
    image_paths = _list_images(images_dir)
    Rs: List[np.ndarray] = []
    ts: List[np.ndarray] = []
    valid_paths: List[str] = []
    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        if img is None:
            logger.warning("Cannot read image: %s", img_path)
            continue
        pose = _estimate_pose(
            img, board, dictionary, camera_matrix, dist_coeffs, params
        )
        if pose is None:
            continue
        R, t = pose
        Rs.append(R)
        ts.append(t)
        valid_paths.append(img_path)
        if params.visualize:
            cv2.aruco.drawDetectedMarkers(img, None, None)
            cv2.imshow("charuco pose", img)
            cv2.waitKey(50)
    if params.visualize:
        cv2.destroyAllWindows()
    stats, outliers = _collect_corner_stats(Rs, ts, board, params, logger)
    logger.info("Extracted %d poses after filtering", len(Rs))
    return ExtractionResult(Rs, ts, valid_paths, image_paths, stats, outliers)
