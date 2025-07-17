from __future__ import annotations

"""Error metrics and optimization helpers for calibration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

from utils.logger import Logger, LoggerType
from utils.error_tracker import ErrorTracker

logger: LoggerType = Logger.get_logger("calibration.metrics")


@dataclass
class RegistrationError:
    translation_rmse: float
    rotation_rmse: float


def pose_rmse(
    Rs: Iterable[np.ndarray],
    ts: Iterable[np.ndarray],
    R_ref: np.ndarray,
    t_ref: np.ndarray,
) -> RegistrationError:
    """Compute RMSE of pose differences to a reference pose."""
    trans_err = []
    rot_err = []
    for R, t in zip(Rs, ts):
        dR = R_ref.T @ R
        angle = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1.0, 1.0))
        rot_err.append(np.degrees(angle))
        trans_err.append(np.linalg.norm(t - t_ref))
    return RegistrationError(
        float(np.sqrt(np.mean(np.square(trans_err)))),
        float(np.sqrt(np.mean(np.square(rot_err)))),
    )


def plot_errors(errors: RegistrationError, file: Path, logger: LoggerType) -> None:
    """Save a simple bar plot of translation/rotation RMSE."""
    try:
        fig, ax = plt.subplots()
        ax.bar(["trans", "rot"], [errors.translation_rmse, errors.rotation_rmse])
        ax.set_ylabel("RMSE")
        file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(file)
        plt.close(fig)
        logger.info(f"Error plot saved to {file}")
    except Exception as exc:
        logger.error(f"Plotting error metrics failed: {exc}")
        raise


def optimize_z_scale(
    measured_pts: np.ndarray,
    observed_pts: np.ndarray,
    observed_pix: np.ndarray,
    K: np.ndarray,
    logger: LoggerType | None = None,
) -> float:
    """Optimize Z scaling factor to minimize registration error."""
    logger = logger or Logger.get_logger("calibration.metrics")

    def _error(scale: float) -> float:
        z = observed_pts[:, 2] * scale
        x = (observed_pix[:, 0] - K[0, 2]) * z / K[0, 0]
        y = (observed_pix[:, 1] - K[1, 2]) * z / K[1, 1]
        pts = np.column_stack([x, y, z])
        R, t = svd_transform(measured_pts, pts)
        reg = (R @ measured_pts.T).T + t
        return float(np.sqrt(np.mean(np.sum((reg - pts) ** 2, axis=1))))

    res = optimize.minimize(
        lambda s: _error(float(s)), np.array([1.0]), method="Nelder-Mead"
    )
    logger.info(f"Optimized z-scale: {res.x[0]:.6f}")
    return float(res.x[0])


def svd_transform(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return rigid transform from ``A`` to ``B`` using SVD."""
    try:
        assert A.shape == B.shape
        dim = A.shape[1]
        centroid_A = A.mean(axis=0)
        centroid_B = B.mean(axis=0)
        # centroid_A = centroid_A.reshape(-1, dim)
        # centroid_B = centroid_B.reshape(-1, dim)
        AA = A - centroid_A
        BB = B - centroid_B
        H = AA.T @ BB
        rank = np.linalg.matrix_rank(H)
        # find rotation
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        det = np.linalg.det(R)
        if det < 0:
            print(f"det(R) = {det}, reflection detected!, correcting for it ...")
            S = np.eye(dim)
            S[-1, -1] = -1
            R = Vt.T @ S @ U.T
        t = R @ centroid_A.T + centroid_B.T

        transformed = (R @ A[:2].T).T + t
        logger.debug(f"SVD transform sample: {transformed.tolist()}")
        return R, t
    except Exception as exc:
        logger.error(f"SVD transform failed: {exc}")
        ErrorTracker.report(exc)
        raise


def rotation_angle(R: np.ndarray) -> float:
    """
    Returns the rotation angle (in degrees) from a rotation matrix.
    """
    trace = np.trace(R)
    angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    return np.degrees(angle)


def handeye_errors(
    robot_Rs: List[np.ndarray],
    robot_ts: List[np.ndarray],
    target_Rs: List[np.ndarray],
    target_ts: List[np.ndarray],
    R_cam2tool: np.ndarray,
    t_cam2tool: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-sample hand-eye calibration errors:
      - Translational error (meters)
      - Rotational error (degrees)

    For each frame:
        T_robot * T_cam2tool ≈ T_target

    Returns:
        trans_errors: np.ndarray, shape (N,)
        rot_errors: np.ndarray, shape (N,)
    """
    N = len(robot_Rs)
    trans_errors = []
    rot_errors = []

    # Build camera-to-tool transformation
    T_cam2tool = np.eye(4)
    T_cam2tool[:3, :3] = R_cam2tool
    T_cam2tool[:3, 3] = t_cam2tool.ravel()

    for Rr, tr, Rt, tt in zip(robot_Rs, robot_ts, target_Rs, target_ts):
        # Robot base to tool
        T_robot = np.eye(4)
        T_robot[:3, :3] = Rr
        T_robot[:3, 3] = tr.ravel()

        # Target/base to pattern
        T_target = np.eye(4)
        T_target[:3, :3] = Rt
        T_target[:3, 3] = tt.ravel()

        # Predicted camera pose in robot base: T_pred = T_robot @ T_cam2tool
        T_pred = T_robot @ T_cam2tool

        # Compare predicted to target
        delta_T = np.linalg.inv(T_target) @ T_pred

        # Translational error
        trans_err = np.linalg.norm(delta_T[:3, 3])
        trans_errors.append(trans_err)

        # Rotational error (degrees)
        rot_err = rotation_angle(delta_T[:3, :3])
        rot_errors.append(rot_err)

    return np.array(trans_errors), np.array(rot_errors)
