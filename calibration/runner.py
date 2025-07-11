"""Command line interface orchestrating calibration workflows."""

from __future__ import annotations

from pathlib import Path
import argparse
from typing import List

import cv2
import numpy as np

from utils.io import JSONPoseLoader
from utils.settings import handeye as cfg, IMAGE_EXT, DEPTH_EXT
from utils.logger import Logger

from .detector import CharucoBoardConfig, detect_charuco, pose_from_detection
from .extractor import load_depth, board_center_from_depth
from .handeye import calibrate_opencv, calibrate_svd, calibrate_svd_points


LOGGER = Logger.get_logger("calibration.runner")


def _load_images(directory: Path) -> List[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix == IMAGE_EXT)


def _default_board() -> CharucoBoardConfig:
    return CharucoBoardConfig(
        squares=cfg.square_numbers,
        square_size=cfg.square_length,
        marker_size=cfg.marker_length,
        dictionary=cv2.aruco.getPredefinedDictionary(
            cfg.CHARUCO_DICT_MAP[cfg.aruco_dict]
        ),
    )


def run_handeye(
    poses_file: Path,
    images_dir: Path,
    K: np.ndarray,
    dist: np.ndarray,
    method: str = "opencv",
) -> None:
    images = _load_images(images_dir)
    board_cfg = _default_board()
    robot_Rs, robot_ts = JSONPoseLoader.load_poses(str(poses_file))
    target_Rs, target_ts, measured, observed, pix = [], [], [], [], []
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        det = detect_charuco(img, board_cfg)
        if det is None:
            continue
        depth = load_depth(img_path)
        center = board_center_from_depth(det.corners, depth, K)
        R, t = pose_from_detection(det, K, dist)
        target_Rs.append(R)
        target_ts.append(t)
        measured.append(center)
        observed.append(center)
        pix.append(det.corners.reshape(-1, 2).mean(axis=0))
    if method == "opencv":
        result = calibrate_opencv(robot_Rs, robot_ts, target_Rs, target_ts)
    elif method == "svd":
        result = calibrate_svd(robot_Rs, robot_ts, target_Rs, target_ts)
    else:
        measured = np.asarray(measured)
        observed = np.asarray(observed)
        pix = np.asarray(pix)
        result = calibrate_svd_points(measured, observed, pix, K)
    cam_Rs, cam_ts = [], []
    T = np.eye(4)
    T[:3, :3] = result.rotation
    T[:3, 3] = result.translation
    for Rg, tg in zip(robot_Rs, robot_ts):
        T_base = np.eye(4)
        T_base[:3, :3] = Rg
        T_base[:3, 3] = tg
        T_cam = T_base @ T
        cam_Rs.append(T_cam[:3, :3])
        cam_ts.append(T_cam[:3, 3])
    out_base = Path(cfg.calib_output_dir) / f"handeye_{method}"
    out_base.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_base.with_suffix(".txt"), T, fmt="%.8f")
    LOGGER.info("Result saved to %s", out_base)


def main() -> None:
    p = argparse.ArgumentParser(description="hand-eye calibration runner")
    p.add_argument("poses", type=Path)
    p.add_argument("images", type=Path)
    p.add_argument("intrinsics", type=Path)
    p.add_argument("method", choices=["opencv", "svd", "points"], default="opencv")
    args = p.parse_args()
    fs = cv2.FileStorage(str(args.intrinsics), cv2.FILE_STORAGE_READ)
    K = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("dist_coeffs").mat()
    fs.release()
    run_handeye(args.poses, args.images, K, dist, args.method)


if __name__ == "__main__":
    main()
