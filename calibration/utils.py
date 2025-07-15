from __future__ import annotations

"""I/O utilities for calibration results and timestamps."""

from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from utils.logger import Logger
from utils.error_tracker import ErrorTracker
from utils.io import save_json as _save_json, save_camera_params_xml, save_camera_params_txt

logger = Logger.get_logger("calibration.utils")


def timestamp() -> str:
    """Return current timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_text(path: Path, text: str) -> None:
    """Write plain text to ``path`` creating parent directories."""
    logger.debug(f"Saving text to {path}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(text)
    except Exception as exc:
        logger.error(f"Failed to save text: {exc}")
        ErrorTracker.report(exc)


def save_json(path: Path, data: Any) -> None:
    """Write JSON data to ``path`` creating parent directories."""
    logger.debug(f"Saving JSON to {path}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _save_json(str(path), data)
    except Exception as exc:
        logger.error(f"Failed to save JSON: {exc}")
        ErrorTracker.report(exc)


def save_transform(base: Path, matrix: np.ndarray) -> None:
    """Save a transformation matrix to ``base`` with txt and json."""
    logger.debug(f"Saving transform to {base}")
    try:
        save_text(base.with_suffix(".txt"), np.array2string(matrix, precision=8))
        save_json(base.with_suffix(".json"), matrix.tolist())
    except Exception as exc:
        logger.error(f"Failed to save transform: {exc}")
        ErrorTracker.report(exc)


def save_camera_params(base: Path, K: np.ndarray, dist: np.ndarray, rms: float) -> None:
    """Save camera intrinsics to ``base`` (.txt, .json and .xml)."""
    logger.info(f"Saving camera parameters to {base}")
    try:
        txt = base.with_suffix(".txt")
        xml = base.with_suffix(".xml")
        save_camera_params_txt(str(txt), K, dist, rms)
        save_camera_params_xml(str(xml), K, dist)
    except Exception as exc:
        logger.error(f"Failed to save camera params: {exc}")
        ErrorTracker.report(exc)
