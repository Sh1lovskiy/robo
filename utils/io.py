"""File I/O helpers for images and calibration data."""

from __future__ import annotations

import json
from typing import Any, List, Tuple

import cv2
import numpy as np

from scipy.spatial.transform import Rotation


class JSONPoseLoader:
    """Load robot poses for hand-eye calibration from a JSON file."""

    @staticmethod
    def load_poses(json_file: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Return rotation and translation lists from ``json_file``."""
        data = load_json(json_file)
        Rs, ts = [], []
        for pose in data.values():
            tcp_pose = pose["tcp_coords"]
            t = np.array(tcp_pose[:3], dtype=np.float64) / 1000.0
            rx, ry, rz = tcp_pose[3:]
            R_mat = Rotation.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()
            Rs.append(R_mat)
            ts.append(t)
        return Rs, ts


def read_image(path: str) -> np.ndarray | None:
    """Return an image from ``path`` or ``None`` if loading fails."""
    return cv2.imread(path)


def write_image(path: str, img: np.ndarray) -> None:
    """Save an image to disk."""
    cv2.imwrite(path, img)


def load_json(path: str) -> Any:
    """Load JSON data from ``path``."""
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str, data: Any) -> None:
    """Write data as JSON to ``path``."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_npy(path: str) -> np.ndarray:
    """Load an ``.npy`` array."""
    return np.load(path)


def save_npy(path: str, arr: np.ndarray) -> None:
    """Save an array to an ``.npy`` file."""
    np.save(path, arr)


def load_camera_params(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read camera matrix and distortion coefficients from OpenCV XML/YAML."""
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("dist_coeffs").mat()
    fs.release()
    return camera_matrix, dist_coeffs


def save_camera_params_xml(path: str, K: np.ndarray, dist: np.ndarray) -> None:
    """Write camera parameters to an XML/YAML file."""
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", K)
    fs.write("dist_coeffs", dist)
    fs.release()


def save_camera_params_txt(
    path: str, K: np.ndarray, dist: np.ndarray, rms: float | None = None
) -> None:
    """Write camera parameters to a text file."""
    with open(path, "w") as f:
        if rms is not None:
            f.write(f"RMS Error: {rms:.6f}\n")
        f.write("camera_matrix =\n")
        np.savetxt(f, K, fmt="%.10f")
        f.write("dist_coeffs =\n")
        np.savetxt(f, dist.reshape(1, -1), fmt="%.10f")
