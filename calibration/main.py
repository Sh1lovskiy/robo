from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from .calibrator import (
    HandEyeCalibrator,
    IntrinsicCalibrator,
    CharucoCalibrator,
)
from .data_collector import DataCollector
from .robot_runner import RobotRunner
from utils.logger import Logger
from utils.settings import paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Camera and robot calibration package")
    parser.add_argument(
        "--type",
        required=True,
        choices=["handeye", "camera", "charuco"],
        help=(
            "Calibration type: 'handeye' (robot-camera), 'camera' (intrinsic),"
            " 'charuco' (marker/Charuco)"
        ),
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Run robot grid scan and collect data using current settings.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help=(
            "Run the selected calibration on previously collected data and "
            "save results."
        ),
    )
    return parser.parse_args()


def latest_file(pattern: str) -> Path | None:
    files = sorted(paths.CAPTURES_DIR.glob(pattern))
    return files[-1] if files else None


def load_images(prefix: str) -> List[Path]:
    return sorted(paths.CAPTURES_DIR.glob(f"{prefix}*.png"))


def main() -> None:
    args = parse_args()
    logger = Logger.get_logger("calibration.main")

    images: List[Path] = []
    poses_file: Path | None = None

    if args.collect:
        if args.type == "handeye":
            collector = DataCollector(robot=RobotRunner())
            images, poses_file = collector.collect_handeye()
        else:
            collector = DataCollector()
            images = collector.collect_images(20)

    if args.calibrate:
        if not images:
            prefix = "frame_" if args.type == "handeye" else "img_"
            images = load_images(prefix)
        if args.type == "handeye" and poses_file is None:
            poses_file = latest_file("poses_*.json")
        if not images:
            logger.error("No images available for calibration")
            return
        if args.type == "handeye":
            if poses_file is None:
                logger.error("No pose file found for hand-eye calibration")
                return
            calibrator = HandEyeCalibrator()
            out = calibrator.calibrate(poses_file, images)
        elif args.type == "camera":
            calibrator = IntrinsicCalibrator()
            out = calibrator.calibrate(images)
        else:
            calibrator = CharucoCalibrator()
            out = calibrator.calibrate(images)
        logger.info(f"Calibration complete, results saved to {out}")

    if not args.collect and not args.calibrate:
        logger.warning("Nothing to do. Use --collect and/or --calibrate")


if __name__ == "__main__":
    main()
