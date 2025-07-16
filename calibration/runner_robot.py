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
                    axis = random.randint(0, 2)
                    angle = random.uniform(-25, 25)
                    orient = base_orient.copy()
                    orient[axis] += float(angle)
                    pose = [float(x), float(y), float(z), *orient]
                    poses.append(pose)
        self.logger.debug(f"Generated {len(poses)} grid poses with random tilt")
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
