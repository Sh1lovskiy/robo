from __future__ import annotations

"""Error metrics and plotting helpers for calibration."""

from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

from utils.logger import Logger, LoggerType
from utils.error_tracker import ErrorTracker
from utils.geometry import TransformUtils

logger = Logger.get_logger("calibration.evaluate")


def transformation_error(
    A_R: Iterable[np.ndarray],
    A_t: Iterable[np.ndarray],
    B_R: Iterable[np.ndarray],
    B_t: Iterable[np.ndarray],
    X_R: np.ndarray,
    X_t: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Compute residual translation error for the hand-eye equation.

    The function evaluates how well a transformation ``X`` satisfies
    ``A_i X = X B_i`` for each pair of robot (``A``) and camera (``B``) poses.
    The returned values are translation differences for each pair and their
    root mean square error.
    """
    errors = []
    try:
        T_x = TransformUtils.build_transform(X_R, X_t)
        inv_tx = np.linalg.inv(T_x)
        for Ra, ta, Rb, tb in zip(A_R, A_t, B_R, B_t):
            T_a = TransformUtils.build_transform(Ra, ta)
            T_b = TransformUtils.build_transform(Rb, tb)
            left = T_a @ T_x
            right = T_x @ T_b
            diff = np.linalg.inv(right) @ left
            errors.append(np.linalg.norm(diff[:3, 3]))
        errs = np.array(errors)
        rmse = float(np.sqrt(np.mean(errs**2)))
        return errs, rmse
    except Exception as exc:
        logger.error(f"Error computation failed: {exc}")
        ErrorTracker.report(exc)
        return np.array([]), 0.0


def optimize_z_offset(
    target_ts: list[np.ndarray],
    initial: float,
    compute_error: callable[[float], float],
) -> float:
    """Optimize z-scale for pose translations."""
    try:
        res = optimize.minimize(
            lambda z: compute_error(float(z)), np.asarray(initial), method="Nelder-Mead"
        )
        return float(res.x)
    except Exception as exc:
        logger.error(f"Z optimization failed: {exc}")
        ErrorTracker.report(exc)
        return initial


def plot_registration(
    measured: np.ndarray,
    observed: np.ndarray,
    file: Path,
    title: str = "registration",
) -> None:
    """Save 3D scatter plot comparing measured and observed points."""
    try:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            measured[:, 0], measured[:, 1], measured[:, 2], c="b", label="measured"
        )
        ax.scatter(
            observed[:, 0], observed[:, 1], observed[:, 2], c="r", label="observed"
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend(loc="best")
        file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(file)
        plt.close(fig)
        logger.info(f"Plot saved to {file}")
    except Exception as exc:
        logger.error(f"Plotting failed: {exc}")
        ErrorTracker.report(exc)
