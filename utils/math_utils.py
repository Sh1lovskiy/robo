from __future__ import annotations
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation

__all__ = [
    "euler_to_matrix",
    "make_transform",
    "decompose_transform",
    "invert_transform",
]


def euler_to_matrix(
    rx: float, ry: float, rz: float, *, degrees: bool = True
) -> np.ndarray:
    """Return a rotation matrix from Euler angles."""
    return Rotation.from_euler("xyz", [rx, ry, rz], degrees=degrees).as_matrix()


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build a homogeneous transform from ``R`` and ``t``."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T


def decompose_transform(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return rotation matrix and translation vector from a transform."""
    return T[:3, :3], T[:3, 3]


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Return the inverse of a homogeneous transform."""
    R, t = decompose_transform(T)
    R_inv = R.T
    t_inv = -R_inv @ t
    return make_transform(R_inv, t_inv)
