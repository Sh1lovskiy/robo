# utils/io.py
"""I/O helpers for calibration files."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def load_camera_params(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load camera matrix and distortion coefficients from OpenCV XML/YAML."""
    fs = cv2.FileStorage(str(filename), cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("dist_coeffs").mat()
    fs.release()
    return camera_matrix, dist_coeffs


def save_camera_params_xml(
    filename: str,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> None:
    """Save camera calibration to an OpenCV XML/YAML file."""
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    fs = cv2.FileStorage(str(filename), cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", camera_matrix)
    fs.write("dist_coeffs", dist_coeffs)
    fs.release()


def save_camera_params_txt(
    filename: str,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rms: float | None = None,
) -> None:
    """Save camera calibration to a plain text file."""
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as f:
        if rms is not None:
            f.write(f"RMS Error: {rms:.6f}\n")
        f.write("camera_matrix =\n")
        np.savetxt(f, camera_matrix, fmt="%.10f")
        f.write("dist_coeffs =\n")
        np.savetxt(f, dist_coeffs.reshape(1, -1), fmt="%.10f")
