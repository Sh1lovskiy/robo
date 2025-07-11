from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

from utils.logger import Logger
from utils.settings import paths, handeye as handeye
from utils import load_camera_params

from .calibrator import HandEyeCalibrator, IntrinsicCalibrator
from .data_collector import DataCollector
from .robot_runner import RobotRunner
from .pattern import create_pattern


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Camera and robot calibration package")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["intr", "handeye", "both"],
        metavar="[intr|handeye|both]",
        help="Calibration workflow mode",
    )
    parser.add_argument(
        "--pattern",
        required=True,
        choices=["charuco", "aruco", "chess"],
        metavar="[charuco|aruco|chess]",
        help="Calibration target pattern",
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Collect images (and poses for hand-eye)",
    )
    parser.add_argument(
        "--calib",
        action="store_true",
        help="Run calibration using available data",
    )
    return parser.parse_args()


def latest_file(pattern: str) -> Path | None:
    files = sorted(paths.CAPTURES_DIR.glob(pattern))
    return files[-1] if files else None


def load_images(prefix: str) -> List[Path]:
    return sorted(paths.CAPTURES_DIR.glob(f"*.png"))


def main() -> None:
    args = parse_args()
    logger = Logger.get_logger("calibration.main")
    pattern = create_pattern(args.pattern)

    images: List[Path] = []
    poses_file: Path | None = None

    if args.collect:
        if args.mode in ("handeye", "both"):
            collector = DataCollector(robot=RobotRunner())
            images, poses_file = collector.collect_handeye()
        else:
            collector = DataCollector()
            images = collector.collect_images(20)
    if args.calib:
        if not images:
            prefix = "frame_" if args.mode in ("handeye", "both") else "img_"
            images = load_images(prefix)
        if args.mode in ("handeye", "both") and poses_file is None:
            poses_file = latest_file("poses.json")
        if not images:
            logger.error("No images available for calibration")
            return

        results: List[str] = []
        intrinsics: tuple[np.ndarray, np.ndarray] | None = None
        if args.mode in ("intr", "both"):
            try:
                result = IntrinsicCalibrator().calibrate(images, pattern)
            except Exception as exc:
                logger.error(str(exc))
                return
            intrinsics = (result.camera_matrix, result.dist_coeffs)
            results.append(str(result.output_base))
        if args.mode in ("handeye", "both"):
            if intrinsics is None:
                intrinsics = load_camera_params(handeye.charuco_xml)
            if poses_file is None:
                logger.error("No pose file found for hand-eye calibration")
                return
            out = HandEyeCalibrator().calibrate(poses_file, images, pattern, intrinsics)
            results.append(str(out))
        if results:
            logger.info("Calibration complete: " + ", ".join(results))

    if not args.collect and not args.calibrate:
        logger.warning("Nothing to do. Use --collect and/or --calib")


if __name__ == "__main__":
    main()
