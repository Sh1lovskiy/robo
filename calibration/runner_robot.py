from __future__ import annotations

"""Move the robot on a grid and record poses for calibration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Iterator
import numpy as np
import random

from robot.controller import RobotController
from utils.logger import Logger, LoggerType
from utils.error_tracker import ErrorTracker
from utils.settings import grid_calib, paths
from .utils import timestamp, save_json


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
                    pose = [float(x), float(y), float(z), *orient]
                    poses.append(pose)
        self.logger.debug(
            f"Generated {len(poses)} grid poses with random multi-axis tilt"
        )
        return poses

    def generate_oriented_grid(
        self,
        n_poses_per_point: int = 3,
        rx_base: float = 180.0,
        ry_base: float = 0.0,
        rz_base: float = 180.0,
        rx_range: float = 20.0,
        ry_range: float = 30.0,
        rz_range: float = 30.0,
        randomize: bool = True,
        seed: int = 42,
    ) -> list[list[float]]:
        """Generate grid points with randomized orientations around a base."""
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = grid_calib.workspace_limits
        step = grid_calib.grid_step

        rng = np.random.default_rng(seed)
        poses = []
        for x in np.arange(x_min, x_max, step):
            for y in np.arange(y_min, y_max, step):
                for z in np.arange(z_min, z_max, step):
                    for _ in range(n_poses_per_point):
                        if randomize:
                            rx = rx_base + rng.uniform(-rx_range, rx_range)
                            ry = ry_base + rng.uniform(-ry_range, ry_range)
                            rz = rz_base + rng.uniform(-rz_range, rz_range)
                        else:
                            rx, ry, rz = rx_base, ry_base, rz_base
                        pose = [float(x), float(y), float(z), rx, ry, rz]
                        poses.append(pose)
        self.logger.debug(
            f"Generated {len(poses)} grid poses with randomized orientations: "
            f"base=({rx_base},{ry_base},{rz_base}), "
            f"range=({rx_range},{ry_range},{rz_range})"
        )
        return poses

    def save_poses(self, poses: List[List[float]]) -> Path:
        """Save recorded poses to a timestamped JSON file."""
        out_dir = paths.CAPTURES_EXTR_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            file = out_dir / f"poses_{timestamp()}.json"
            data = {
                str(i): {"tcp_coords": np.asarray(p).tolist()}
                for i, p in enumerate(poses)
            }
            save_json(file, data)
            self.logger.info(f"Poses saved to {file}")
            return file
        except Exception as exc:
            self.logger.error(f"Failed to save poses: {exc}")
            # ErrorTracker.report(exc)
            raise
