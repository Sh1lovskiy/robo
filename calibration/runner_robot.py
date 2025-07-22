from __future__ import annotations

"""Move the robot on a grid and record poses for calibration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import numpy as np
import random
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from robot.controller import RobotController
from utils.logger import Logger, LoggerType
from utils.settings import grid_calib, paths
from .utils import timestamp, save_json


def wrap_angle_deg(angle):
    """Wrap angle(s) in degrees to [-180, 180]."""
    return (angle + 180) % 360 - 180


@dataclass
class RobotRunner:
    """Move the robot on a calibration grid and record poses."""

    controller: RobotController = field(default_factory=RobotController)
    logger: LoggerType = field(
        default_factory=lambda: Logger.get_logger("calibration.robot_runner")
    )

    def generate_grid(self) -> list[list[float]]:
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = grid_calib.workspace_limits
        step = grid_calib.grid_step
        poses = []
        for x in np.arange(x_min, x_max, step):
            for y in np.arange(y_min, y_max, step):
                for z in np.arange(z_min, z_max, step):
                    base_orient = list(grid_calib.tool_orientation)
                    n_axes = random.choice([1, 2, 3])
                    axes = random.sample([0, 1, 2], k=n_axes)
                    orient = base_orient.copy()
                    for axis in axes:
                        angle = random.uniform(-25, 25)
                        orient[axis] += float(angle)
                    poses.append([float(x), float(y), float(z), *orient])
        self.logger.debug(
            f"Generated {len(poses)} grid poses with random multi-axis tilt"
        )
        return poses

    def generate_oriented_grid(
        self,
        rx_base=180.0,
        ry_base=0.0,
        rz_base=180.0,
        rx_range=30.0,
        ry_range=30.0,
        rz_range=30.0,
        randomize=True,
        seed=42,
        batch_size=128,
    ) -> list[list[float]]:
        """
        For each grid point, randomly search in batches for a valid orientation.
        All angles in degrees, positions in millimeters.
        Returns: list of [x, y, z, rx, ry, rz]
        """
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = grid_calib.workspace_limits
        step = grid_calib.grid_step

        rng = np.random.default_rng(seed)
        poses = []
        filtered_total = 0

        x_vals = np.arange(x_min, x_max, step)
        y_vals = np.arange(y_min, y_max, step)
        z_vals = np.arange(z_min, z_max, step)

        total_points = len(x_vals) * len(y_vals) * len(z_vals)

        with tqdm(total=total_points, desc="Generating grid poses") as pbar:
            for x in x_vals:
                for y in y_vals:
                    for z in z_vals:
                        found = False
                        filtered = 0
                        while not found:
                            if randomize:
                                rx = rx_base + rng.uniform(
                                    -rx_range, rx_range, size=batch_size
                                )
                                ry = ry_base + rng.uniform(
                                    -ry_range, ry_range, size=batch_size
                                )
                                rz = rz_base + rng.uniform(
                                    -rz_range, rz_range, size=batch_size
                                )
                            else:
                                rx = np.full(batch_size, rx_base)
                                ry = np.full(batch_size, ry_base)
                                rz = np.full(batch_size, rz_base)

                            angles = np.stack([rx, ry, rz], axis=-1)
                            rots = R.from_euler("xyz", angles, degrees=True)
                            Rmats = rots.as_matrix()

                            # checks
                            dets = np.linalg.det(Rmats)
                            ortho = np.linalg.norm(
                                np.matmul(np.transpose(Rmats, (0, 2, 1)), Rmats)
                                - np.eye(3),
                                axis=(1, 2),
                            )
                            traces = np.trace(Rmats, axis1=1, axis2=2)

                            # mask
                            valid = (dets > 0.99) & (ortho < 1e-6) & (traces > -1.0)
                            idx = np.flatnonzero(valid)
                            filtered += batch_size - valid.sum()

                            if idx.size > 0:
                                i = idx[0]
                                poses.append(
                                    [
                                        float(x),
                                        float(y),
                                        float(z),
                                        float(wrap_angle_deg(rx[i])),
                                        float(wrap_angle_deg(ry[i])),
                                        float(wrap_angle_deg(rz[i])),
                                    ]
                                )
                                found = True
                                filtered_total += filtered
                        pbar.update(1)

        self.logger.info(
            f"Generated {len(poses)} grid poses\n"
            f"base=({rx_base}, {ry_base}, {rz_base}), range=({rx_range}, {ry_range}, {rz_range})"
        )
        return poses

    def save_poses(self, poses: List[List[float]]) -> Path:
        """Save recorded poses to a timestamped JSON file."""
        out_dir = paths.CAPTURES_EXTR_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        file = out_dir / f"poses_{timestamp()}.json"
        data = {
            str(i): {"tcp_coords": np.asarray(p).tolist()} for i, p in enumerate(poses)
        }
        save_json(file, data)
        self.logger.info(f"Poses saved to {file}")
        return file
