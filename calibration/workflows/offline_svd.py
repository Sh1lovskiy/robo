"""Workflow for offline SVD-based hand-eye calibration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from calibration.grid_calibrator import OfflineCalibrator
from utils.logger import Logger, LoggerType
from utils.settings import (
    HandEyeSettings,
    GridCalibrationSettings,
    handeye,
    grid_calib,
)


@dataclass
class OfflineSVDWorkflow:
    """Run the :class:`OfflineCalibrator` with configurable settings."""

    hcfg: HandEyeSettings = handeye
    gcfg: GridCalibrationSettings = grid_calib
    logger: LoggerType = Logger.get_logger("calibration.workflow.offline")

    def run(self) -> None:
        """Execute calibration and report results."""
        calibrator = OfflineCalibrator(self.hcfg, self.gcfg, self.logger)
        calibrator.calibrate_from_files()


def add_offline_args(parser: argparse.ArgumentParser) -> None:
    """CLI options for offline calibration."""
    parser.add_argument(
        "--images_dir",
        default=handeye.images_dir,
        help="Directory with Charuco images",
    )
    parser.add_argument(
        "--robot_poses_file",
        default=handeye.robot_poses_file,
        help="JSON file with robot poses",
    )
    parser.add_argument(
        "--charuco_xml",
        default=grid_calib.charuco_xml,
        help="Camera calibration XML file",
    )
    parser.add_argument(
        "--calib_output_dir",
        default=grid_calib.calib_output_dir,
        help="Directory to save results",
    )
    parser.add_argument(
        "--calibration_type",
        default=grid_calib.calibration_type,
        help="EYE_IN_HAND or EYE_TO_HAND",
    )


def run_offline(args: argparse.Namespace) -> None:
    """Entry point for the ``offline-svd`` CLI command."""
    hcfg = HandEyeSettings(
        robot_poses_file=args.robot_poses_file,
        images_dir=args.images_dir,
        charuco_xml=handeye.charuco_xml,
        method=handeye.method,
        min_corners=handeye.min_corners,
        outlier_std=handeye.outlier_std,
        visualize=handeye.visualize,
        calib_output_dir=args.calib_output_dir,
    )
    gcfg = GridCalibrationSettings(
        calibration_type=args.calibration_type,
        workspace_limits=grid_calib.workspace_limits,
        grid_step=grid_calib.grid_step,
        reference_point_offset=grid_calib.reference_point_offset,
        tool_orientation=grid_calib.tool_orientation,
        charuco_xml=args.charuco_xml,
        calib_output_dir=args.calib_output_dir,
    )
    OfflineSVDWorkflow(hcfg, gcfg).run()
