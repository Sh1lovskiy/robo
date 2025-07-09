"""Calibration command line interface."""

from __future__ import annotations

import argparse

from utils.logger import Logger
from utils.settings import logging

from calibration.calibrator import Calibrator


def parse_args() -> argparse.Namespace:
    """Return parsed CLI arguments."""

    parser = argparse.ArgumentParser(description="Calibration CLI")
    parser.add_argument("--run_grid", action="store_true", help="Move robot on grid and capture data")
    parser.add_argument("--calibrate", action="store_true", help="Run calibration after data collection")
    parser.add_argument("--visualize", action="store_true", help="Visualize collected data")
    parser.add_argument("--config_file", type=str, help="Optional config file")
    return parser.parse_args()


def main() -> None:
    """Entry point for the calibration CLI."""
    args = parse_args()
    Logger.configure(level=logging.level, json_format=logging.json)
    logger = Logger.get_logger("main")
    calibrator = Calibrator(visualize=args.visualize)
    if args.run_grid or args.calibrate:
        calibrator.run()
    else:
        logger.info("No action specified")


if __name__ == "__main__":
    main()
