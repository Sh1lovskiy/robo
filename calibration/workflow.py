"""CLI wrappers for common calibration workflows."""

from __future__ import annotations

from utils.logger import Logger
from utils.cli import Command, CommandDispatcher

from .validation import _add_validate_args, _run_validate
from .workflows.charuco import add_charuco_args, run_charuco
from .workflows.handeye import add_handeye_args, run_handeye


def create_cli() -> CommandDispatcher:
    """Build the argument dispatcher for calibration commands."""
    return CommandDispatcher(
        "Calibration workflows",
        [
            Command(
                "charuco", run_charuco, add_charuco_args, "Run Charuco calibration"
            ),
            Command(
                "handeye", run_handeye, add_handeye_args, "Run Hand-Eye calibration"
            ),
            Command(
                "validate",
                _run_validate,
                _add_validate_args,
                "Validate Hand-Eye calibration",
            ),
        ],
    )


def main() -> None:
    """Entry point used by ``calibration-cli`` script."""
    logger = Logger.get_logger("calibration.workflows")
    create_cli().run(logger=logger)


if __name__ == "__main__":
    main()
