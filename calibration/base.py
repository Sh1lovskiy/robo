"""Calibration core classes and interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from utils.logger import Logger

from .utils import ImagePair, estimate_pose_pnp, load_image_pairs, save_poses

log = Logger.get_logger("calibrate.base")


class CalibrationPattern(ABC):
    """Abstract base class for calibration patterns."""

    def __init__(self, board_size: Tuple[int, int], square_length: float) -> None:
        self.board_size = board_size
        self.square_length = square_length

    @abstractmethod
    def detect(
        self, image: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Detect pattern points in ``image``.

        Returns
        -------
        tuple of ``(object_points, image_points, overlay)`` or ``None`` if the
        pattern is not found.
        """


class Calibrator:
    """Solve camera-to-target poses using a specific pattern."""

    def __init__(
        self,
        pattern: CalibrationPattern,
        K: np.ndarray,
        dist: np.ndarray,
        *,
        save_images: bool = False,
    ) -> None:
        self.pattern = pattern
        self.K = K
        self.dist = dist
        self.save_images = save_images

    # ------------------------------------------------------------------
    def run(self, image_dir: Path) -> None:
        """Run pose estimation on images within ``image_dir``.

        Parameters
        ----------
        image_dir:
            Directory containing ``*_rgb.png`` and ``*_depth.npy`` pairs.
        """

        pairs = load_image_pairs(image_dir)
        poses: List[np.ndarray] = []
        for idx, pair in enumerate(tqdm(pairs, desc="Frames")):
            rgb = cv2.imread(str(pair.rgb))
            if rgb is None:
                log.warning(f"Skipping {pair.rgb.name}: missing RGB image")
                continue
            det = self.pattern.detect(rgb)
            if det is None:
                log.warning("Pattern not detected in frame %d", idx)
                continue
            obj_pts, img_pts, overlay = det
            pose, err = estimate_pose_pnp(obj_pts, img_pts, self.K, self.dist)
            poses.append(pose)
            if self.save_images and overlay is not None:
                cv2.imwrite(str(image_dir / f"{idx:03d}_overlay.png"), overlay)
            log.debug("Frame %d reprojection error %.3f", idx, err)

        if not poses:
            log.error("No valid poses computed")
            return

        out_json = image_dir / "camera_poses.json"
        save_poses(poses, out_json)
        log.info(f"Calibration finished -> {out_json}")
