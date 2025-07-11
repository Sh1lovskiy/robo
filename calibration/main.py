from __future__ import annotations

"""Command line interface for camera and hand-eye calibration."""

import argparse
from pathlib import Path
from typing import List

from utils import paths, handeye, logging, load_camera_params, IMAGE_EXT
from utils.logger import Logger
from utils.error_tracker import ErrorTracker

from .pattern import create_pattern
from .data_collector import DataCollector
from .calibrator import IntrinsicCalibrator, HandEyeCalibrator
from .robot_runner import RobotRunner


import argparse


def _parse_args() -> argparse.Namespace:
    """Return parsed command line arguments with beautiful help."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Robotics calibration CLI\n\n"
            "Example: python -m calibration.main --mode both --collect --calib --visualize"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        usage=(
            "main.py [-h] --mode [intr|handeye|both]\n"
            "       [--pattern charuco|aruco|chess]\n"
            "       [--collect]\n"
            "       [--calib]\n"
            "       [--method svd|tsai|park|horaud|andreff|daniilidis|all]\n"
            "       [--visualize]\n"
            "       [--count COUNT]\n"
            "       [--dataset PATH]"
        ),
    )
    parser.add_argument(
        "--mode",
        required=True,
        metavar="[intr|handeye|both]",
        help="Calibration mode",
    )
    parser.add_argument(
        "--pattern",
        metavar="[charuco|aruco|chess]",
        default="charuco",
        help="Calibration pattern type (default: charuco)",
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Capture new images/poses before calibration",
    )
    parser.add_argument(
        "--calib",
        action="store_true",
        help="Run calibration after collection or on existing data",
    )
    parser.add_argument(
        "--method",
        choices=["svd", "tsai", "park", "horaud", "andreff", "daniilidis", "all"],
        default="all",
        help="Hand-eye calibration method to use",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Plot robot and camera poses after calibration",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        metavar="COUNT",
        help="Number of images to capture (default: 20)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Optional directory of existing images",
    )
    return parser.parse_args()


def _load_images(directory: Path) -> List[Path]:
    """Return all images in ``directory`` sorted by name."""
    return sorted(directory.glob(f"*{IMAGE_EXT}"))


def main() -> None:
    """Entry point for the calibration command line interface."""
    args = _parse_args()
    logger = Logger.get_logger("calibration.cli")
    logger.info("Robotics calibration CLI")
    Logger.configure(logging.level, paths.LOG_DIR, logging.json)
    ErrorTracker.install_excepthook()
    ErrorTracker.install_signal_handlers()
    ErrorTracker.install_keyboard_listener()
    if args.collect:
        robot = RobotRunner()
    else:
        robot = None
    collector = DataCollector(robot=robot)
    pattern = create_pattern(args.pattern)
    images: List[Path] = []
    poses_file: Path | None = None

    if args.collect:
        if args.mode == "intr":
            images = collector.collect_images(args.count)
        else:
            images, poses_file = collector.collect_handeye()
    else:
        if args.mode == "intr":
            images = _load_images(args.dataset or paths.CAPTURES_DIR)
        else:
            images = _load_images(args.dataset or Path(handeye.images_dir))
            poses_file = Path(handeye.robot_poses_file)

    if args.mode in ("intr", "both") and args.calib:
        intr = IntrinsicCalibrator()
        intr_result = intr.calibrate(images, pattern)
        K, dist = intr_result.camera_matrix, intr_result.dist_coeffs
    else:
        K, dist = load_camera_params(handeye.charuco_xml)

    if args.mode in ("handeye", "both") and args.calib:
        assert poses_file is not None
        he = HandEyeCalibrator(method=args.method, visualize=args.visualize)
        he.calibrate(poses_file, images, pattern, (K, dist))


if __name__ == "__main__":
    main()
