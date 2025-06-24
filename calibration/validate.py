"""Calibration diagnostics and validation for Charuco/Hand-Eye."""

import os
import argparse
from dataclasses import dataclass

import cv2
import numpy as np
import matplotlib.pyplot as plt

from calibration.charuco import CharucoCalibrator, CHARUCO_DICT_MAP
from calibration.handeye import HandEyeCalibrator
from calibration.pose_loader import JSONPoseLoader
from utils.config import Config
from utils.io import load_camera_params
from utils.logger import Logger, LoggerType
from utils.cli import Command, CommandDispatcher

BOARD_LT_BASE = np.array([-0.165, -0.365, 0.0])
BOARD_RB_BASE = np.array([-0.4, -0.53, 0.0])


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


def plot_errors(errs1, errs2, label1, label2, fname):
    plt.figure(figsize=(8, 6))
    plt.hist(errs1, bins=15, alpha=0.6, label=label1, color="blue")
    plt.hist(errs2, bins=15, alpha=0.6, label=label2, color="red")
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
            "LT vs GT",
            "RB vs GT",
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
                f"median={np.median(errs):.5f},"
                f"90th={np.percentile(errs,90):.5f}"
            )
        plot_errors(
            lt_errs_mean,
            rb_errs_mean,
            "LT vs mean",
            "RB vs mean",
            "handeye_corner_errors_mean.png",
        )

        # Show contribution table
        self.logger.info("=== GT error ===")
        contribution_table(lt_errs_gt, val_img_paths, self.logger, "LT (vs GT)")
        contribution_table(rb_errs_gt, val_img_paths, self.logger, "RB (vs GT)")
        self.logger.info("=== Mean error ===")
        contribution_table(lt_errs_mean, val_img_paths, self.logger, "LT (vs mean)")
        contribution_table(rb_errs_mean, val_img_paths, self.logger, "RB (vs mean)")


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
