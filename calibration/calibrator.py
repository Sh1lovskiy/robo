from __future__ import annotations

"""Core calibration routines for intrinsic camera calibration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from utils import paths, IMAGE_EXT
from utils.logger import Logger, LoggerType
from utils.error_tracker import ErrorTracker

from .pattern import CalibrationPattern
from .utils import save_camera_params, timestamp
from .visualizer import plot_reprojection_errors
import utils.settings as settings


@dataclass
class IntrinsicResult:
    """Result of an intrinsic calibration."""

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    output_base: Path


@dataclass
class IntrinsicCalibrator:
    """Compute camera intrinsics using a generic calibration pattern."""

    logger: LoggerType = field(
        default_factory=lambda: Logger.get_logger("calibration.intrinsic")
    )

    def calibrate(
        self, images: List[Path], pattern: CalibrationPattern
    ) -> IntrinsicResult:
        """
        Calibrate camera intrinsics from a set of images.

        The provided ``pattern`` is used to detect 2-D/3-D correspondences in
        all input images which are then passed to OpenCV's camera calibration
        routines.  Per-image reprojection RMSE values are computed for
        diagnostics and the resulting parameters are stored on disk.
        """
        self.logger.info("Starting intrinsic calibration")
        try:
            img_paths = [str(p) for p in images]
            corners, ids, img_size = pattern.detect_many(img_paths)

            if not pattern.detections or img_size is None:
                raise RuntimeError("No valid detections for intrinsic calibration")

            K, dist, rms, per_view = pattern.calibrate_camera(img_size)
            out_base = paths.RESULTS_DIR / f"camera_{timestamp()}"
            save_camera_params(out_base, K, dist, rms)
            intr_viz_file = paths.VIZ_DIR / f"{out_base.stem}_reproj{IMAGE_EXT}"

            plot_reprojection_errors(
                per_view, intr_viz_file, interactive=settings.DEFAULT_INTERACTIVE
            )
            self.logger.info(
                f"Intrinsics saved to {out_base.relative_to(Path.cwd())} (RMS={rms:.6f})"
            )
            return IntrinsicResult(K, dist, out_base)
        except Exception as exc:
            self.logger.error(f"Intrinsic calibration failed: {exc}")
            ErrorTracker.report(exc)
            raise
