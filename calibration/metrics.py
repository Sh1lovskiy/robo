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
        centroid_A = A.mean(axis=0)
        centroid_B = B.mean(axis=0)
        AA = A - centroid_A
        BB = B - centroid_B
        H = AA.T @ BB
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T
        t = -R @ centroid_A + centroid_B
        return R, t
    except Exception as exc:
        logger.error(f"SVD transform failed: {exc}")
        ErrorTracker.report(exc)
        raise


def handeye_errors(
    robot_Rs: List[np.ndarray],
    robot_ts: List[np.ndarray],
    target_Rs: List[np.ndarray],
    target_ts: List[np.ndarray],
    R_cam2tool: np.ndarray,
    t_cam2tool: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-pair rotation [deg] and translation errors."""
    errors_rot: List[float] = []
    errors_trans: List[float] = []
    X = np.eye(4)
    X[:3, :3] = R_cam2tool
    X[:3, 3] = t_cam2tool.flatten()
    for i in range(len(robot_Rs) - 1):
        A = np.eye(4)
        A[:3, :3] = robot_Rs[i + 1] @ robot_Rs[i].T
        A[:3, 3] = robot_ts[i + 1] - A[:3, :3] @ robot_ts[i]
        B = np.eye(4)
        B[:3, :3] = target_Rs[i + 1] @ target_Rs[i].T
        B[:3, 3] = target_ts[i + 1] - B[:3, :3] @ target_ts[i]
        left = A @ X
        right = X @ B
        dR = left[:3, :3] @ right[:3, :3].T
        angle = np.arccos(np.clip((np.trace(dR) - 1) / 2.0, -1.0, 1.0))
        dt = np.linalg.norm(left[:3, 3] - right[:3, 3])
        errors_rot.append(np.degrees(angle))
        errors_trans.append(dt)
    return np.array(errors_rot), np.array(errors_trans)
