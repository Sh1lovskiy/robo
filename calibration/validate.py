"""Calibration diagnostics and validation for Charuco/Hand-Eye."""

import os
import argparse
from dataclasses import dataclass

import cv2
import numpy as np
import matplotlib.pyplot as plt
import threading
import time
import shutil
import json

from calibration.charuco import CharucoCalibrator, CHARUCO_DICT_MAP
from calibration.handeye import HandEyeCalibrator
from calibration.pose_loader import JSONPoseLoader
from utils.cli import Command, CommandDispatcher
from utils.config import Config
from utils.io import load_camera_params
from utils.keyboard import GlobalKeyListener, TerminalEchoSuppressor
from utils.logger import Logger, LoggerType

import open3d as o3d

BOARD_LT_BASE = np.array([-0.165, -0.365, 0.0])
BOARD_RB_BASE = np.array([-0.4, -0.53, 0.0])


def plot_handeye_reconstruction_o3d(
    lt_base_pred, rb_base_pred, gt_lt=None, gt_rb=None, mean_lt=None, mean_rb=None
):
    points = []
    colors = []

    # Predicted LT (blue)
    for p in lt_base_pred:
        points.append(p)
        colors.append([0.1, 0.1, 1.0])
    # Predicted RB (red)
    for p in rb_base_pred:
        points.append(p)
        colors.append([1.0, 0.2, 0.2])
    # GT LT (big blue dot)
    if gt_lt is not None:
        points.append(gt_lt)
        colors.append([0.0, 0.0, 0.0])
    # GT RB (big red dot)
    if gt_rb is not None:
        points.append(gt_rb)
        colors.append([0.0, 0.0, 0.0])
    # Mean LT (big cyan dot)
    if mean_lt is not None:
        points.append(mean_lt)
        colors.append([0.3, 1.0, 1.0])
    # Mean RB (large raspberry dot)
    if mean_rb is not None:
        points.append(mean_rb)
        colors.append([1.0, 0.2, 1.0])

    points = np.array(points)
    colors = np.array(colors)

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.colors = o3d.utility.Vector3dVector(colors)

    size = [5.0] * (len(lt_base_pred) + len(rb_base_pred))
    if gt_lt is not None:
        size.append(50.0)
    if gt_rb is not None:
        size.append(50.0)
    if mean_lt is not None:
        size.append(15.0)
    if mean_rb is not None:
        size.append(15.0)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Charuco hand-eye 3D overlay")
    vis.add_geometry(pc)

    render_option = vis.get_render_option()
    render_option.point_size = 7.0

    vis.run()
    vis.destroy_window()


def load_image_paths(images_dir):
    return sorted(
        [
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )


def detect_board_corners(
    img_path, board, dictionary, camera_matrix, dist_coeffs, min_corners=4
):
    img = cv2.imread(img_path)
    if img is None:
        return None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    if ids is None or len(ids) < min_corners:
        return None, None
    _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, board
    )
    if charuco_corners is None or charuco_ids is None or len(charuco_ids) < min_corners:
        return None, None
    rvec_init = np.zeros((3, 1), dtype=np.float64)
    tvec_init = np.zeros((3, 1), dtype=np.float64)
    retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        charuco_corners,
        charuco_ids,
        board,
        camera_matrix,
        dist_coeffs,
        rvec_init,
        tvec_init,
    )
    if not retval:
        return None, None
    R, _ = cv2.Rodrigues(rvec)
    obj_pts = board.getChessboardCorners()
    lt = R @ obj_pts[0].reshape(3, 1) + tvec
    rb = R @ obj_pts[-1].reshape(3, 1) + tvec
    return lt.flatten(), rb.flatten()


def analyze_handeye_residuals(cam_corners, R_cam2base, t_cam2base):
    lt_base_pred, rb_base_pred = [], []
    for lt_cam, rb_cam in cam_corners:
        lt_base = R_cam2base @ lt_cam + t_cam2base
        rb_base = R_cam2base @ rb_cam + t_cam2base
        lt_base_pred.append(lt_base)
        rb_base_pred.append(rb_base)
    return np.stack(lt_base_pred), np.stack(rb_base_pred)


def error_vs_reference(lt_pred, rb_pred, ref_lt, ref_rb):
    lt_errs = np.linalg.norm(lt_pred - ref_lt, axis=1)
    rb_errs = np.linalg.norm(rb_pred - ref_rb, axis=1)
    return lt_errs, rb_errs


def error_vs_mean(lt_pred, rb_pred):
    lt_mean = lt_pred.mean(axis=0)
    rb_mean = rb_pred.mean(axis=0)
    lt_errs = np.linalg.norm(lt_pred - lt_mean, axis=1)
    rb_errs = np.linalg.norm(rb_pred - rb_mean, axis=1)
    return lt_errs, rb_errs, lt_mean, rb_mean


def filter_by_percentile(
    errors, img_paths, percentile=80, logger=None, title="Filter by percentile"
):
    threshold = np.percentile(errors, percentile)
    keep_idx = [i for i, e in enumerate(errors) if e <= threshold]
    drop_idx = [i for i, e in enumerate(errors) if e > threshold]
    keep_paths = [img_paths[i] for i in keep_idx]
    drop_paths = [img_paths[i] for i in drop_idx]
    if logger:
        logger.info(
            f"{title}: keeping {len(keep_paths)} frames (<= {percentile}th perc., thr={threshold:.5f} m)"
        )
        logger.info("KEPT:")
        for p in keep_paths:
            logger.info(f"  {os.path.basename(p)}")
        logger.info(f"TO DROP ({len(drop_paths)} frames):")
        for p in drop_paths:
            logger.info(f"  {os.path.basename(p)}")
    return keep_paths, drop_paths


def move_images(img_paths, images_dir, drop_dir, logger):
    os.makedirs(drop_dir, exist_ok=True)
    for p in img_paths:
        fname = os.path.basename(p)
        dst = os.path.join(drop_dir, fname)
        if os.path.exists(dst):
            logger.warning(f"File already exists in drop_imgs: {dst}, skipping")
            continue
        shutil.move(p, dst)
        logger.info(f"Moved {fname} → drop_imgs/")


def move_poses_for_dropped_images(drop_img_paths, images_dir, logger):
    poses_json = os.path.join(images_dir, "poses.json")
    drop_dir = os.path.join(images_dir, "drop_imgs")
    drop_poses_json = os.path.join(drop_dir, "poses.json")

    if not os.path.isfile(poses_json):
        logger.warning("No poses.json found in %s, skipping poses move.", images_dir)
        return

    with open(poses_json, "r") as f:
        all_poses = json.load(f)

    drop_filenames = set(
        os.path.splitext(os.path.basename(p))[0].split("_")[0] for p in drop_img_paths
    )

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


def ask_confirm_keyboard(logger, msg="Press [y] to move, any other to skip: "):
    """
    Use GlobalKeyListener and TerminalEchoSuppressor to ask user for a single key.
    Returns True if 'y' was pressed, else False.
    """
    confirmed = {"value": False}
    done = threading.Event()

    def on_yes():
        confirmed["value"] = True
        done.set()

    def on_any():
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


def show_charuco_overlay(img_path, board, dictionary, camera_matrix, dist_coeffs):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read {img_path}")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    out = img.copy()
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(out, corners, ids)
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, board
        )
        if charuco_corners is not None and charuco_ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(
                out, charuco_corners, charuco_ids, (0, 255, 0)
            )
        else:
            print(f"No Charuco corners found for {img_path}")
    else:
        print(f"No ArUco markers found for {img_path}")
    cv2.imshow(f"Charuco overlay: {img_path}", out)
    cv2.waitKey(500)
    cv2.destroyWindow(f"Charuco overlay: {img_path}")


def plot_errors(errs1, errs2, label1, label2, fname):
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


def contribution_table(errors, img_paths, logger, title="Frame contribution"):
    sorted_idx = np.argsort(errors)[::-1]
    logger.info(f"{title}:")
    logger.info(f"{'Frame':>20} | {'Error [m]':>10} | {'Percent of Total':>15}")
    total = errors.sum()
    for idx in sorted_idx:
        fname = os.path.basename(img_paths[idx])
        perc = 100.0 * errors[idx] / total if total > 0 else 0.0
        logger.info(f"{fname:>20} | {errors[idx]:10.5f} | {perc:15.2f} %")
    logger.info("-" * 54)


@dataclass
class HandEyeValidationWorkflow:
    """Validation of hand-eye calibration by board corner analysis."""

    logger: LoggerType = Logger.get_logger("calibration.workflow.handeyeval")

    def run(self):
        Config.load()
        cfg = Config.get("handeye")
        images_dir = cfg.get("images_dir", "captures")
        charuco_xml = cfg.get("charuco_xml", "calibration/results/charuco_cam.xml")
        out_dir = cfg.get("calib_output_dir", "calibration/results")
        squares_x = cfg.get("squares_x", 5)
        squares_y = cfg.get("squares_y", 7)
        square_length = cfg.get("square_length", 0.033)
        marker_length = cfg.get("marker_length", 0.025)
        dict_name = cfg.get("aruco_dict", "5X5_100")
        dictionary = cv2.aruco.getPredefinedDictionary(CHARUCO_DICT_MAP[dict_name])
        board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y), square_length, marker_length, dictionary
        )
        camera_matrix, dist_coeffs = load_camera_params(charuco_xml)
        img_paths = load_image_paths(images_dir)

        npz_file = os.path.join(out_dir, "handeye_PARK.npz")
        handeye = np.load(npz_file)
        R_cam2base, t_cam2base = handeye["R"], handeye["t"].flatten()

        cam_corners = []
        val_img_paths = []
        bad_img_paths = []
        for idx, path in enumerate(img_paths):
            lt, rb = detect_board_corners(
                path, board, dictionary, camera_matrix, dist_coeffs
            )
            show_charuco_overlay(path, board, dictionary, camera_matrix, dist_coeffs)
            if lt is not None and rb is not None:
                cam_corners.append((lt, rb))
                val_img_paths.append(path)
            else:
                bad_img_paths.append(path)

        num_good = len(val_img_paths)
        num_bad = len(bad_img_paths)
        total = len(img_paths)
        self.logger.info(
            f"Charuco detection: good={num_good}, bad={num_bad}, total={total}"
        )
        if num_bad > 0:
            self.logger.warning(
                f"Bad frames: {', '.join(os.path.basename(p) for p in bad_img_paths)}"
            )

        if not cam_corners:
            self.logger.error("No valid board detections.")
            return

        lt_base_pred, rb_base_pred = analyze_handeye_residuals(
            cam_corners, R_cam2base, t_cam2base
        )

        # 1. Error vs ground truth
        lt_errs_gt, rb_errs_gt = error_vs_reference(
            lt_base_pred, rb_base_pred, BOARD_LT_BASE, BOARD_RB_BASE
        )
        self.logger.info("=== Error relative to GROUND TRUTH board coordinates ===")
        for label, errs in [("LT/GT", lt_errs_gt), ("RB/GT", rb_errs_gt)]:
            self.logger.info(
                f"{label}: mean={errs.mean():.5f}, std={errs.std():.5f}, "
                f"min={errs.min():.5f}, max={errs.max():.5f}, "
                f"median={np.median(errs):.5f}, "
                f"90th={np.percentile(errs,90):.5f}"
            )

        plot_errors(
            lt_errs_gt,
            rb_errs_gt,
            "LT/RB",
            "GT",
            "handeye_corner_errors_gt.png",
        )

        # 2. Error vs mean value
        lt_errs_mean, rb_errs_mean, lt_mean, rb_mean = error_vs_mean(
            lt_base_pred, rb_base_pred
        )
        self.logger.info("=== Error relative to MEAN board position ===")
        for label, errs in [("LT/MEAN", lt_errs_mean), ("RB/MEAN", rb_errs_mean)]:
            self.logger.info(
                f"{label}: mean={errs.mean():.5f}, std={errs.std():.5f}, "
                f"min={errs.min():.5f}, max={errs.max():.5f}, "
                f"90th={np.percentile(errs,90):.5f}"
            )
        plot_errors(
            lt_errs_mean,
            rb_errs_mean,
            "LT/RB",
            "mean",
            "handeye_corner_errors_mean.png",
        )

        self.logger.info("=== Filtering by 80th percentile (LT/GT error) ===")
        keep_paths, drop_paths = filter_by_percentile(
            lt_errs_gt,
            val_img_paths,
            percentile=80,
            logger=self.logger,
            title="LT/GT error filtering",
        )

        if drop_paths:
            if ask_confirm_keyboard(
                self.logger, f"\nMove {len(drop_paths)} images to 'drop_imgs'?"
            ):
                drop_dir = os.path.join(images_dir, "drop_imgs")
                move_images(drop_paths, images_dir, drop_dir, self.logger)
                move_poses_for_dropped_images(drop_paths, images_dir, self.logger)
            else:
                self.logger.info("Skipping move to drop_imgs.")
        else:
            self.logger.info("No images to drop.")

        plot_handeye_reconstruction_o3d(
            lt_base_pred,
            rb_base_pred,
            gt_lt=BOARD_LT_BASE,
            gt_rb=BOARD_RB_BASE,
            mean_lt=lt_mean,
            mean_rb=rb_mean,
        )

        # Show contribution table
        # self.logger.info("=== GT error ===")
        # contribution_table(lt_errs_gt, val_img_paths, self.logger, "LT (vs GT)")
        # contribution_table(rb_errs_gt, val_img_paths, self.logger, "RB (vs GT)")
        # self.logger.info("=== Mean error ===")
        # contribution_table(lt_errs_mean, val_img_paths, self.logger, "LT (vs mean)")
        # contribution_table(rb_errs_mean, val_img_paths, self.logger, "RB (vs mean)")


def _add_validate_args(parser: argparse.ArgumentParser):
    Config.load()
    cfg = Config.get("handeye")
    parser.add_argument(
        "--images_dir",
        default=cfg.get("images_dir", "cloud"),
        help="Directory with Charuco images",
    )
    parser.add_argument(
        "--charuco_xml",
        default=cfg.get("charuco_xml", "calibration/results/charuco_cam.xml"),
        help="Camera calibration XML file",
    )


def _run_validate(args: argparse.Namespace):
    HandEyeValidationWorkflow().run()


def create_cli() -> CommandDispatcher:
    return CommandDispatcher(
        "Calibration validation",
        [
            Command(
                "validate",
                _run_validate,
                _add_validate_args,
                "Validate Hand-Eye calibration",
            ),
        ],
    )


def main() -> None:
    logger = Logger.get_logger("calibration.handeyeval")
    create_cli().run(logger=logger)


if __name__ == "__main__":
    main()
