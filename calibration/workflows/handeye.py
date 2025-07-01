"""Workflow for solving hand-eye calibration from saved images and poses."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import cv2
import numpy as np

from calibration.charuco import (
    load_board,
    extract_charuco_poses,
    load_camera_params,
    ExtractionParams,
)
from calibration.pose_loader import LmdbPoseLoader
from calibration.handeye import (
    HandEyeCalibrator,
    NPZHandEyeSaver,
    TxtHandEyeSaver,
    DBHandEyeSaver,
)
from utils.logger import Logger, LoggerType
from utils.lmdb_storage import LmdbStorage
from utils.cli import Command
from utils.settings import paths, handeye, charuco, HandEyeSettings


@dataclass
class HandEyeCalibrationWorkflow:
    """Run hand-eye calibration using saved poses and Charuco frames.

    Attributes:
        cfg: Configuration with paths and algorithm settings.
        logger: Optional logger instance for status updates.
    """

    cfg: HandEyeSettings = handeye
    logger: LoggerType = Logger.get_logger("calibration.workflow.handeye")

    def _load_config(self) -> tuple[
        dict,
        str,
        str,
        str,
        str,
        str,
        cv2.aruco_CharucoBoard,
        cv2.aruco_Dictionary,
    ]:
        """Gather calibration inputs and board configuration."""
        cfg = self.cfg
        out_dir = cfg.calib_output_dir
        os.makedirs(out_dir, exist_ok=True)
        charuco_xml = cfg.charuco_xml
        robot_file = cfg.robot_poses_file
        images_dir = cfg.images_dir
        method = cfg.method.upper()
        board_cfg = dict(
            squares_x=charuco.squares_x,
            squares_y=charuco.squares_y,
            square_length=charuco.square_length,
            marker_length=charuco.marker_length,
            aruco_dict=charuco.aruco_dict,
        )
        board, dictionary = load_board(board_cfg)
        return (
            cfg,
            out_dir,
            charuco_xml,
            robot_file,
            images_dir,
            method,
            board,
            dictionary,
        )

    def _filter_robot_poses(
        self,
        Rs: list[np.ndarray],
        ts: list[np.ndarray],
        valid_paths: list[str],
        all_paths: list[str],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Match robot poses to successfully processed images."""

        def extract_index(fname: str) -> str:
            return os.path.splitext(os.path.basename(fname))[0].split("_")[0]

        pose_map = {extract_index(p): (R, t) for p, R, t in zip(all_paths, Rs, ts)}
        filtered_Rs, filtered_ts = [], []
        for p in valid_paths:
            idx = extract_index(p)
            if idx in pose_map:
                R, t = pose_map[idx]
                filtered_Rs.append(R)
                filtered_ts.append(t)
            else:
                self.logger.warning("No robot pose for image %s (index %s)", p, idx)
        return filtered_Rs, filtered_ts

    def _save(
        self,
        calibrator: HandEyeCalibrator,
        out_dir: str,
        method: str,
        results: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> None:
        """Persist computed hand-eye matrices in multiple formats."""
        if results is None:
            R, t = calibrator.calibrate(method)
            results = {method: (R, t)}
        db = LmdbStorage(os.path.join(out_dir, "calibration.lmdb"))
        for name, (R, t) in results.items():
            npz_file = os.path.join(out_dir, f"handeye_{name}.npz")
            txt_file = os.path.join(out_dir, f"handeye_{name}.txt")
            calibrator.save(NPZHandEyeSaver(), npz_file, R, t)
            calibrator.save(TxtHandEyeSaver(), txt_file, R, t)
            calibrator.save(DBHandEyeSaver(db), name, R, t)
            self.logger.info(f"Saved {name} calibration to {npz_file}")

    def run(self) -> None:
        """Execute the hand-eye calibration workflow."""
        cfg, out_dir, charuco_xml, robot_file, images_dir, method, board, dictionary = (
            self._load_config()
        )
        if not os.path.isfile(charuco_xml):
            self.logger.error(f"Charuco file {charuco_xml} not found")
            return
        if not os.path.isfile(robot_file):
            self.logger.error(f"Robot poses file {robot_file} not found")
            return
        if not os.path.isdir(images_dir):
            self.logger.error(f"Images directory {images_dir} not found")
            return
        camera_matrix, dist_coeffs = load_camera_params(charuco_xml)
        Rs_g2b, ts_g2b = LmdbPoseLoader.load_poses(robot_file)
        params = ExtractionParams(
            min_corners=cfg.min_corners,
            visualize=cfg.visualize,
            analyze_corners=True,
            outlier_std=cfg.outlier_std,
        )
        extraction = extract_charuco_poses(
            images_dir,
            board,
            dictionary,
            camera_matrix,
            dist_coeffs,
            logger=self.logger,
            params=params,
        )
        Rs_t2c = extraction.rotations
        ts_t2c = extraction.translations
        valid_paths = extraction.valid_paths[: len(Rs_t2c)]
        all_paths = extraction.all_paths
        Rs_g2b_f, ts_g2b_f = self._filter_robot_poses(
            Rs_g2b, ts_g2b, valid_paths, all_paths
        )
        if not Rs_t2c or len(Rs_t2c) != len(Rs_g2b_f):
            self.logger.error(
                f"Pose data mismatch after filtering: {len(Rs_t2c)} camera poses, {len(Rs_g2b_f)} robot poses"
            )
            return
        calibrator = HandEyeCalibrator(self.logger)
        for Rg, tg, Rc, tc in Logger.progress(
            list(zip(Rs_g2b_f, ts_g2b_f, Rs_t2c, ts_t2c)),
            desc="HandEye samples",
        ):
            calibrator.add_sample(Rg, tg, Rc, tc)
        if method == "ALL":
            results = calibrator.calibrate_all()
            self._save(calibrator, out_dir, method, results)
        else:
            self._save(calibrator, out_dir, method)


def add_handeye_args(parser: argparse.ArgumentParser) -> None:
    """Add command line arguments for the ``handeye`` workflow."""
    cfg = handeye
    out_dir = cfg.calib_output_dir
    parser.add_argument(
        "--images_dir",
        default=cfg.images_dir,
        help="Directory with Charuco images",
    )
    parser.add_argument(
        "--charuco_xml",
        default=cfg.charuco_xml,
        help="Camera calibration XML file",
    )
    parser.add_argument(
        "--robot_poses_file",
        default=cfg.robot_poses_file,
        help="JSON file with robot poses",
    )
    parser.add_argument(
        "--method",
        default=cfg.method,
        help="Calibration method or ALL",
    )
    parser.add_argument(
        "--analyze-charuco",
        action="store_true",
        help="Analyze Charuco board positions and reject outlier images",
    )


def run_handeye(args: argparse.Namespace) -> None:
    """Entry point for hand-eye calibration from the CLI."""
    cfg = HandEyeSettings(
        images_dir=args.images_dir,
        charuco_xml=args.charuco_xml,
        robot_poses_file=args.robot_poses_file,
        method=args.method,
        min_corners=handeye.min_corners,
        outlier_std=handeye.outlier_std,
        visualize=handeye.visualize,
        calib_output_dir=handeye.calib_output_dir,
    )
    HandEyeCalibrationWorkflow(cfg).run()
