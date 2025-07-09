"""CLI workflow for ArUco based hand-eye calibration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from calibration.marker_grid import MarkerGridCalibrator
from utils.logger import Logger, LoggerType
from utils.settings import marker_grid, MarkerGridSettings


@dataclass
class MarkerGridWorkflow:
    """Run the :class:`MarkerGridCalibrator` with CLI settings."""

    cfg: MarkerGridSettings = marker_grid
    logger: LoggerType = Logger.get_logger("calibration.workflow.marker_grid")

    def run(self) -> None:
        calibrator = MarkerGridCalibrator(cfg=self.cfg, logger=self.logger)
        calibrator.calibrate()


def add_marker_grid_args(parser: argparse.ArgumentParser) -> None:
    """CLI options for ``marker-grid`` calibration."""
    parser.add_argument(
        "--calib_output_dir",
        default=marker_grid.calib_output_dir,
        help="Directory to store calibration files",
    )


def run_marker_grid(args: argparse.Namespace) -> None:
    """Entry point for the ``marker-grid`` command."""
    cfg = MarkerGridSettings(calib_output_dir=args.calib_output_dir)
    MarkerGridWorkflow(cfg).run()
