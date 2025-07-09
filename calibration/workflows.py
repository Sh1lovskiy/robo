"""CLI wrappers for common calibration workflows."""

from __future__ import annotations

from utils.logger import Logger
from utils.cli import Command, CommandDispatcher

from .workflows.charuco import add_camera_calib_args, run_camera_calib
from .workflows.handeye import add_handeye_args, run_handeye
from .workflows.offline_svd import add_offline_args, run_offline
from .workflows.marker_grid import add_marker_he_args, run_marker_he


def create_cli() -> CommandDispatcher:
    """Build the argument dispatcher for calibration commands."""
    return CommandDispatcher(
        "Calibration workflows",
        [
            Command(
                "camera_calib",
                run_camera_calib,
                add_camera_calib_args,
                "Camera calibration",
            ),
            Command(
                "handeye_calib",
                run_handeye,
                add_handeye_args,
                "Hand-Eye (HE) calibration",
            ),
            Command(
                "marker_he",
                run_marker_he,
                add_marker_he_args,
                "ArUco HE calibration",
            ),
            Command(
                "offline_svd",
                run_offline,
                add_offline_args,
                "Offline SVD HE calibration",
            ),
        ],
    )


def main() -> None:
    """Entry point used by calibration script."""
    logger = Logger.get_logger("calibration.workflows")
    create_cli().run(logger=logger)


if __name__ == "__main__":
    main()
