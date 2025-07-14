from __future__ import annotations

"""Plot utilities for calibration results in world coordinates (TCP-aligned)."""

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from utils.logger import Logger, LoggerType
from utils.error_tracker import ErrorTracker
from utils.settings import DEFAULT_INTERACTIVE
from utils.transform import TransformUtils

logger: LoggerType = Logger.get_logger("calibration.visualizer")


def _rotation_angle(R: np.ndarray) -> float:
    """Return the angle of rotation represented by ``R`` in degrees."""
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2.0, -1.0, 1.0))
    return float(np.degrees(angle))


def _plot_interactive(robot_ts: np.ndarray, cam_ts: np.ndarray, file: Path) -> None:
    """Render an interactive Plotly figure."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=robot_ts[:, 0],
            y=robot_ts[:, 1],
            z=robot_ts[:, 2],
            mode="markers+text",
            name="robot TCP",
            marker=dict(size=5, color="red"),
            text=[f"R{i}" for i in range(len(robot_ts))],
            textposition="top center",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=cam_ts[:, 0],
            y=cam_ts[:, 1],
            z=cam_ts[:, 2],
            mode="markers+text",
            name="camera",
            marker=dict(size=5, color="blue"),
            text=[f"C{i}" for i in range(len(cam_ts))],
            textposition="top center",
        )
    )
    for rt, ct in zip(robot_ts, cam_ts):
        fig.add_trace(
            go.Scatter3d(
                x=[rt[0], ct[0]],
                y=[rt[1], ct[1]],
                z=[rt[2], ct[2]],
                mode="lines",
                line=dict(color="gray", width=2, dash="dash"),
                showlegend=False,
            )
        )
    fig.update_layout(
        title="Robot vs Camera Poses",
        scene=dict(
            xaxis_title="X [m]",
            yaxis_title="Y [m]",
            zaxis_title="Z [m]",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=30),
    )
    file = file.with_suffix(".html")
    file.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(file))
    logger.info(f"Interactive pose plot saved to {file.relative_to(file.cwd())}")


def _plot_static(robot_ts: np.ndarray, cam_ts: np.ndarray, file: Path) -> None:
    """Render a static Matplotlib figure."""
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        robot_ts[:, 0],
        robot_ts[:, 1],
        robot_ts[:, 2],
        c="tab:red",
        label="robot TCP",
        s=50,
    )
    ax.scatter(
        cam_ts[:, 0], cam_ts[:, 1], cam_ts[:, 2], c="tab:blue", label="camera", s=50
    )
    for rt, ct in zip(robot_ts, cam_ts):
        ax.plot(
            [rt[0], ct[0]],
            [rt[1], ct[1]],
            [rt[2], ct[2]],
            color="gray",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
        )
    ax.set_title("Robot vs Camera Poses (Board centers)")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend(loc="upper right")
    ax.set_box_aspect([1, 1, 1])
    file = file.with_suffix(".png")
    file.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(file)
    plt.close(fig)
    logger.info(f"Static pose plot saved to {file.relative_to(file.cwd())}")


def plot_reprojection_errors(errors: Sequence[float], file: Path) -> None:
    """Plot per-frame reprojection RMS error."""
    logger.info("Plotting intrinsic calibration reprojection errors...")
    fig, ax = plt.subplots()
    ax.plot(
        range(1, len(errors) + 1), errors, marker="o", color="tab:blue", label="error"
    )
    ax.set_xlabel("Frame")
    ax.set_ylabel("RMS reprojection error [px]")
    ax.set_title("Reprojection Error per Frame")
    ax.legend()
    file.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(file)
    plt.close(fig)
    logger.info(f"Saved reprojection error plot to {file.relative_to(file.cwd())}")


def plot_poses(
    robot_Rs: Iterable[np.ndarray],
    robot_ts: Iterable[np.ndarray],
    cam_Rs: Iterable[np.ndarray],
    cam_ts: Iterable[np.ndarray],
    file: Path,
    interactive: bool = DEFAULT_INTERACTIVE,
) -> None:
    """
    Plot robot and camera poses as 3D points (spheres) and compute RMSE.
    Red: robot, Blue: camera. Dashed lines connect corresponding poses.
    Also logs translational and rotational RMSE.
    """
    try:
        robot_ts = np.asarray(robot_ts)
        cam_ts = np.asarray(cam_ts)

        if len(robot_ts) != len(cam_ts):
            logger.warning(
                f"Cannot compute RMSE: pose count mismatch "
                f"(robot={len(robot_ts)}, camera={len(cam_ts)})"
            )
            return

        trans_errors = np.linalg.norm(robot_ts - cam_ts, axis=1)
        rot_errors = np.array(
            [_rotation_angle(Rr.T @ Rc) for Rr, Rc in zip(robot_Rs, cam_Rs)]
        )
        trans_rmse = float(np.sqrt(np.mean(trans_errors**2)))
        rot_rmse = float(np.sqrt(np.mean(rot_errors**2)))

        if interactive:
            _plot_interactive(robot_ts, cam_ts, file)
        else:
            _plot_static(robot_ts, cam_ts, file)

    except Exception as exc:
        logger.error(f"Plotting poses failed: {exc}")
        ErrorTracker.report(exc)
