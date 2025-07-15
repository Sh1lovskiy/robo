from __future__ import annotations

"""Command line interface for camera and hand-eye calibration."""

import argparse
from pathlib import Path
from typing import List

from utils import paths, handeye, logging, load_camera_params, IMAGE_EXT
import utils.settings as settings
from utils.logger import Logger
from utils.error_tracker import ErrorTracker

from .pattern import create_pattern
from .data_collector import DataCollector
from .calibrator import IntrinsicCalibrator, HandEyeCalibrator
from .robot_runner import RobotRunner


def _parse_args() -> argparse.Namespace:
    """Return parsed command line arguments."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        formatter_class=argparse.RawTextHelpFormatter,
        usage=(
            "main.py [OPTIONS]\n"
            "\n"
            "Arguments:\n"
            "  --mode MODE           Calibration stages to run\n"
            "  --pattern PATTERN     Calibration pattern type\n"
            "  --collect             Capture new data before calibration\n"
            "  --method METHOD       Hand-eye calibration method\n"
            "  --visualize FORMAT    Output visualization format\n"
            "  --count COUNT         Number of images to capture\n"
            "  --dataset PATH        Optional directory of existing images\n"
            "  -h, --help            Show this help message and exit\n"
        ),
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["intr", "handeye", "both", "none"],
        default="both",
        metavar="MODE",
        help=(
            "Calibration stages to run:\n"
            "  'intr'    : intrinsic calibration only\n"
            "  'handeye' : hand-eye calibration only\n"
            "  'both'    : run both intrinsic and hand-eye stages (default)\n"
            "  'none'    : only collect data, skip calibration"
        ),
    )
    parser.add_argument(
        "--pattern",
        choices=["charuco", "aruco", "chess"],
        default="charuco",
        metavar="PATTERN",
        help=(
            "Calibration pattern type:\n"
            "  'charuco' : Charuco board (default)\n"
            "  'aruco'   : ArUco marker grid\n"
            "  'chess'   : classic chessboard"
        ),
    )
    parser.add_argument(
        "--collect", action="store_true", help="Capture new data before calibration"
    )
    parser.add_argument(
        "--method",
        choices=["svd", "tsai", "park", "horaud", "andreff", "daniilidis", "all"],
        default="all",
        metavar="METHOD",
        help=(
            "Hand-eye calibration method:\n"
            "  'svd'        : SVD-based closed-form method (AX=XB)\n"
            "  'tsai'       : Tsai-Lenz method\n"
            "  'park'       : Park-Martin method\n"
            "  'horaud'     : Horaud's method\n"
            "  'andreff'    : Andreff's iterative method\n"
            "  'daniilidis' : Daniilidis' dual quaternion method\n"
            "  'all'        : Run all available methods (default)"
        ),
    )
    parser.add_argument(
        "--visualize",
        choices=["none", "html", "corners", "full"],
        default="none",
        metavar="FORMAT",
        help=(
            "Output visualization format:\n"
            "  'html'    : save plots as interactive HTML with Plotly\n"
            "  'corners' : show detected Charuco/Aruco corners on one image\n"
            "  'full'    : enable both HTML plots and corner overlays\n"
            "  'none'    : disable both HTML plots and corner overlays (default)"
        ),
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        metavar="COUNT",
        help="Number of images to capture",
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
    html_out = args.visualize in ("html", "full")
    show_corners = args.visualize in ("corners", "full")
    settings.DEFAULT_INTERACTIVE = html_out

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

    run_intr = args.mode in ("intr", "both")
    run_handeye = args.mode in ("handeye", "both")

    if args.mode == "none":
        return

    if run_intr:
        intr = IntrinsicCalibrator()
        intr_result = intr.calibrate(images, pattern)
        K, dist = intr_result.camera_matrix, intr_result.dist_coeffs
    else:
        K, dist = load_camera_params(handeye.charuco_xml)

    if run_handeye:
        assert poses_file is not None
        he = HandEyeCalibrator(method=args.method, visualize=show_corners)
        he.calibrate(poses_file, images, pattern, (K, dist))


if __name__ == "__main__":
    main()
