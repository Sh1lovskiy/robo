# utils/geometry.py
"""Helper math utilities used across modules."""

from scipy.spatial.transform import Rotation as R
import numpy as np


def euler_to_matrix(rx: float, ry: float, rz: float, degrees: bool = True) -> np.ndarray:
    """Convert Euler angles to rotation matrix."""
    if degrees:
        angles = np.deg2rad([rx, ry, rz])
    else:
        angles = [rx, ry, rz]
    return R.from_euler("xyz", angles).as_matrix()
