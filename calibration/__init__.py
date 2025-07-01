"""Camera and robot calibration utilities.

This package bundles helpers for Charuco-based camera calibration and hand-eye
calibration of a robot-mounted camera.  High level workflow classes are
provided for command line tools while the core algorithms remain accessible as
simple classes.

Public entry points:
    - :class:`CharucoCalibrator` for intrinsic calibration using Charuco boards.
    - :class:`HandEyeCalibrator` for solving ``AX = XB`` hand-eye problems.
    - Workflow classes ``CharucoCalibrationWorkflow`` and
      ``HandEyeCalibrationWorkflow`` for CLI usage.
"""

from .charuco import (
    CharucoCalibrator,
    CHARUCO_DICT_MAP,
    load_board,
    ExtractionParams,
    ExtractionResult,
    extract_charuco_poses,
)
from .handeye import HandEyeCalibrator, NPZHandEyeSaver, TxtHandEyeSaver, DBHandEyeSaver
from calibration.workflows.handeye import HandEyeCalibrationWorkflow
from calibration.workflows.charuco import CharucoCalibrationWorkflow

__all__ = [
    "CharucoCalibrator",
    "HandEyeCalibrator",
    "NPZHandEyeSaver",
    "TxtHandEyeSaver",
    "DBHandEyeSaver",
    "CharucoCalibrationWorkflow",
    "HandEyeCalibrationWorkflow",
    "CHARUCO_DICT_MAP",
    "load_board",
    "ExtractionParams",
    "ExtractionResult",
    "extract_charuco_poses",
]
