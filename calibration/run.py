"""Command line entry point for calibration workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.error_tracker import ErrorTracker
from utils.logger import Logger

from .april import AprilTagPattern
from .base import Calibrator
from .capture import capture_dataset
from .charuco import CharucoPattern
from .chessboard import ChessboardPattern
from .utils import (
    confirm,
    create_output_dir,
    load_intrinsics_yml,
    parse_board_size,
    save_intrinsics,
)

log = Logger.get_logger("calibrate.run")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Camera calibration tool")
    parser.add_argument("--pattern", choices=["charuco", "chessboard", "april"], required=True)
    parser.add_argument("--board-size", required=True, help="Board size WxH")
    parser.add_argument("--square-length", type=float, required=True, help="Square side length [m]")
    parser.add_argument("--aruco-dict", default="DICT_5X5_100")
    parser.add_argument("--image-dir", type=Path, help="Offline image directory")
    parser.add_argument("--intrinsics", type=Path, default=Path("cam_params.yml"))
    parser.add_argument("--capture-poses", action="store_true", help="Capture images and robot poses")
    parser.add_argument("--save-images", action="store_true", help="Save overlay images")
    parser.add_argument("--interactive", action="store_true", help="Confirm steps via keyboard")
    parser.add_argument("--max-frames", type=int, default=20, help="Maximum frames to capture")
    return parser


def main(argv: list[str] | None = None) -> None:
    ErrorTracker.install_excepthook()
    ErrorTracker.install_signal_handlers()
    parser = build_parser()
    args = parser.parse_args(argv)

    board_size = parse_board_size(args.board_size)
    K, dist = load_intrinsics_yml(args.intrinsics)

    if args.pattern == "chessboard":
        pattern = ChessboardPattern(board_size, args.square_length)
    elif args.pattern == "charuco":
        pattern = CharucoPattern(board_size, args.square_length, args.aruco_dict)
    else:
        pattern = AprilTagPattern(board_size, args.square_length, args.aruco_dict)

    calibrator = Calibrator(pattern, K, dist, save_images=args.save_images)

    if args.capture_poses:
        if not confirm("Start pose/image capture?"):
            log.info("Aborted before capture")
            return
        out_dir = create_output_dir(args.pattern, board_size, args.square_length)
        save_intrinsics(K, dist, out_dir / "intrinsics.json")
        capture_dataset(out_dir, max_frames=args.max_frames, interactive=args.interactive)
        image_dir = out_dir
        if args.interactive and not confirm("Continue to calibration?"):
            log.info("Aborted before calibration")
            return
    elif args.image_dir:
        image_dir = args.image_dir
    else:
        parser.error("--image-dir required when not capturing poses")
        return

    calibrator.run(image_dir)


if __name__ == "__main__":
    main()
