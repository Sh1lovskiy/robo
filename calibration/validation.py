"""Calibration diagnostics and hand-eye validation."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import cv2
import numpy as np

from calibration.charuco import CHARUCO_DICT_MAP, load_camera_params
from utils.cli import Command, CommandDispatcher
from utils.logger import Logger, LoggerType
from utils.settings import handeye, validation, charuco
from .helpers.validation_utils import (
    load_image_paths,
    detect_board_corners,
    analyze_handeye_residuals,
    error_vs_reference,
    error_vs_mean,
    filter_by_percentile,
    move_images,
    move_poses_for_dropped_images,
    ask_confirm_keyboard,
    plot_errors,
    validate_handeye_calibration,
)


@dataclass
class HandEyeValidationWorkflow:
    """Validate hand-eye calibration by analyzing board corners."""

    logger: LoggerType = Logger.get_logger("calibration.workflow.handeyeval")

    def _load_config(self) -> tuple[
        str,
        str,
        cv2.aruco_CharucoBoard,
        cv2.aruco_Dictionary,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        cfg = handeye
        val_cfg = validation
        images_dir = cfg.images_dir
        charuco_xml = cfg.charuco_xml
        out_dir = cfg.calib_output_dir
        squares_x = cfg.squares_x
        squares_y = cfg.squares_y
        square_length = cfg.square_length
        marker_length = cfg.marker_length
        dict_name = cfg.aruco_dict
        dictionary = cv2.aruco.getPredefinedDictionary(CHARUCO_DICT_MAP[dict_name])
        board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y), square_length, marker_length, dictionary
        )
        camera_matrix, dist_coeffs = load_camera_params(charuco_xml)
        img_paths = load_image_paths(images_dir)
        npz_file = os.path.join(out_dir, "handeye_PARK.npz")
        data = np.load(npz_file)
        R_cam2base, t_cam2base = data["R"], data["t"].flatten()
        lt_ref = np.array(val_cfg.board_lt_base)
        rb_ref = np.array(val_cfg.board_rb_base)
        return (
            images_dir,
            out_dir,
            board,
            dictionary,
            camera_matrix,
            dist_coeffs,
            R_cam2base,
            t_cam2base,
            lt_ref,
            rb_ref,
        )

    def _gather_corners(
        self,
        img_paths: list[str],
        board: cv2.aruco_CharucoBoard,
        dictionary: cv2.aruco_Dictionary,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[str], list[str]]:
        cam_corners: list[tuple[np.ndarray, np.ndarray]] = []
        good_paths: list[str] = []
        bad_paths: list[str] = []
        for path in img_paths:
            lt, rb = detect_board_corners(
                path, board, dictionary, camera_matrix, dist_coeffs
            )
            if lt is not None and rb is not None:
                cam_corners.append((lt, rb))
                good_paths.append(path)
            else:
                bad_paths.append(path)
        return cam_corners, good_paths, bad_paths

    def _analyze(
        self,
        cam_corners: list[tuple[np.ndarray, np.ndarray]],
        good_paths: list[str],
        board: cv2.aruco_CharucoBoard,
        R_cam2base: np.ndarray,
        t_cam2base: np.ndarray,
        lt_ref: np.ndarray,
        rb_ref: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        images_dir: str,
    ) -> None:
        lt_pred, rb_pred = analyze_handeye_residuals(
            cam_corners, R_cam2base, t_cam2base
        )
        lt_errs_gt, rb_errs_gt = error_vs_reference(lt_pred, rb_pred, lt_ref, rb_ref)
        self.logger.info(
            f"Ground truth error: LT mean {lt_errs_gt.mean():.5f} m, RB mean {rb_errs_gt.mean():.5f} m"
        )
        board_pts_base = board.getChessboardCorners()
        for path, (lt, rb) in zip(good_paths, cam_corners):
            validate_handeye_calibration(
                board_pts_base,
                R_cam2base,
                t_cam2base,
                camera_matrix,
                dist_coeffs,
                np.array([lt, rb]),
                self.logger,
            )
        plot_errors(lt_errs_gt, rb_errs_gt, "LT/RB", "GT", "handeye_gt.png")
        lt_errs_mean, rb_errs_mean, lt_mean, rb_mean = error_vs_mean(lt_pred, rb_pred)
        self.logger.info(
            f"Mean position error: LT mean {lt_errs_mean.mean():.5f} m, RB mean {rb_errs_mean.mean():.5f} m"
        )
        plot_errors(lt_errs_mean, rb_errs_mean, "LT/RB", "mean", "handeye_mean.png")
        keep_paths, drop_paths = filter_by_percentile(
            lt_errs_mean,
            good_paths,
            percentile=50,
            logger=self.logger,
            title="LT/GT error filtering",
        )
        if drop_paths and ask_confirm_keyboard(
            self.logger,
            f"\nMove {len(drop_paths)} images to 'drop_imgs'?",
        ):
            drop_dir = os.path.join(images_dir, "drop_imgs")
            move_images(drop_paths, images_dir, drop_dir, self.logger)
            move_poses_for_dropped_images(drop_paths, images_dir, self.logger)
        else:
            self.logger.info("No images to drop.")

    def run(self) -> None:
        (
            images_dir,
            _out_dir,
            board,
            dictionary,
            camera_matrix,
            dist_coeffs,
            R_cam2base,
            t_cam2base,
            lt_ref,
            rb_ref,
        ) = self._load_config()
        img_paths = load_image_paths(images_dir)
        cam_corners, good_paths, bad_paths = self._gather_corners(
            img_paths, board, dictionary, camera_matrix, dist_coeffs
        )
        self.logger.info(
            f"Charuco detection: good={len(good_paths)}, bad={len(bad_paths)}, total={len(img_paths)}"
        )
        if not cam_corners:
            self.logger.error("No valid board detections.")
            return
        self._analyze(
            cam_corners,
            good_paths,
            board,
            R_cam2base,
            t_cam2base,
            lt_ref,
            rb_ref,
            camera_matrix,
            dist_coeffs,
            images_dir,
        )


def _add_validate_args(parser: argparse.ArgumentParser) -> None:
    cfg = handeye
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


def _run_validate(_: argparse.Namespace) -> None:
    HandEyeValidationWorkflow().run()
