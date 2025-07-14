"""Hand-eye solvers using OpenCV and SVD variants."""

from __future__ import annotations

from typing import Iterable, List, Tuple
from dataclasses import dataclass

import cv2
import numpy as np

from .metrics import optimize_z_scale, svd_transform


@dataclass
class HandEyeResult:
    """Container for a solved camera-to-tool transform."""

    rotation: np.ndarray
    translation: np.ndarray
    z_scale: float | None = None


def calibrate_opencv(
    robot_Rs: Iterable[np.ndarray],
    robot_ts: Iterable[np.ndarray],
    target_Rs: Iterable[np.ndarray],
    target_ts: Iterable[np.ndarray],
    method: int = cv2.CALIB_HAND_EYE_TSAI,
) -> HandEyeResult:
    """Solve ``AX = XB`` using OpenCV's hand-eye routines."""
    R, t = cv2.calibrateHandEye(
        list(robot_Rs), list(robot_ts), list(target_Rs), list(target_ts), method=method
    )
    return HandEyeResult(R, t.flatten())


def calibrate_svd_points(
    measured_pts: np.ndarray,
    observed_pts: np.ndarray,
    observed_pix: np.ndarray,
    K: np.ndarray,
) -> HandEyeResult:
    """Estimate camera pose by aligning 3D points measured by the robot and depth."""
    z_scale = optimize_z_scale(measured_pts, observed_pts, observed_pix, K)
    z = observed_pts[:, 2] * z_scale
    x = (observed_pix[:, 0] - K[0, 2]) * z / K[0, 0]
    y = (observed_pix[:, 1] - K[1, 2]) * z / K[1, 1]
    pts = np.column_stack([x, y, z])
    R, t = svd_transform(measured_pts, pts)
    return HandEyeResult(R, t, z_scale)


def _skew(v: np.ndarray) -> np.ndarray:
    """Return the skew-symmetric matrix of vector ``v``."""
    return np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ],
        dtype=np.float64,
    )


def calibrate_handeye_svd(
    robot_Rs: List[np.ndarray],
    robot_ts: List[np.ndarray],
    target_Rs: List[np.ndarray],
    target_ts: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve ``AX = XB`` using the Tsai–Lenz algorithm.

    Parameters
    ----------
    robot_Rs, robot_ts
        Absolute poses of the robot tool with respect to the base frame.
    target_Rs, target_ts
        Poses of the calibration pattern with respect to the camera frame.

    Returns
    -------
    np.ndarray, np.ndarray
        Rotation and translation from the camera frame to the robot tool frame.

    Notes
    -----
    The method first solves for the rotation using a linear system derived from
    relative motions and then estimates the translation via least squares as
    described in Tsai and Lenz, "A New Technique for Fully Autonomous and
    Efficient 3D Robotics Hand/Eye Calibration" (1989).
    """
    pairs = len(robot_Rs)
    A_list, b_list = [], []
    for i in range(pairs - 1):
        Ra_rel = robot_Rs[i + 1] @ robot_Rs[i].T
        Rb_rel = target_Rs[i + 1] @ target_Rs[i].T
        ra, _ = cv2.Rodrigues(Ra_rel)
        rb, _ = cv2.Rodrigues(Rb_rel)
        M = _skew(ra.ravel() + rb.ravel())
        b = (rb - ra).ravel()
        A_list.append(M)
        b_list.append(b)
    A = np.vstack(A_list)
    b = np.hstack(b_list)
    rvec, *_ = np.linalg.lstsq(A, b, rcond=None)
    Rx, _ = cv2.Rodrigues(rvec)
    C_list, d_list = [], []
    for Rg, tg, Rc, tc in zip(robot_Rs, robot_ts, target_Rs, target_ts):
        C_list.append(Rg - np.eye(3))
        d_list.append(Rx @ tc - tg)
    C = np.vstack(C_list)
    d = np.hstack(d_list)
    tvec, *_ = np.linalg.lstsq(C, d, rcond=None)
    return Rx, tvec


def calibrate_svd(
    robot_Rs: Iterable[np.ndarray],
    robot_ts: Iterable[np.ndarray],
    target_Rs: Iterable[np.ndarray],
    target_ts: Iterable[np.ndarray],
) -> HandEyeResult:
    """Solve ``AX = XB`` using the SVD approach."""

    R, t = calibrate_handeye_svd(
        list(robot_Rs), list(robot_ts), list(target_Rs), list(target_ts)
    )
    return HandEyeResult(R, t.flatten())
