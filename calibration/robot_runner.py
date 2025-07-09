from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

from robot.controller import RobotController
from utils.logger import Logger, LoggerType
from utils.settings import grid_calib, paths
from .utils import timestamp, save_json


@dataclass
class RobotRunner:
    """Move the robot on a calibration grid and record poses."""

    controller: RobotController = field(default_factory=RobotController)
    logger: LoggerType = field(
        default_factory=lambda: Logger.get_logger("calibration.robot_runner")
    )

    def generate_grid(self) -> List[List[float]]:
        """Return workspace grid poses relative to the reference point."""
        limits = grid_calib.workspace_limits
        step = grid_calib.grid_step
        x = np.arange(limits[0][0], limits[0][1] + 1e-9, step)
        y = np.arange(limits[1][0], limits[1][1] + 1e-9, step)
        z = np.arange(limits[2][0], limits[2][1] + 1e-9, step)
        base = np.array(grid_calib.reference_point_offset[:3])
        orient = list(grid_calib.tool_orientation)
        poses = []
        for xi in x:
            for yi in y:
                for zi in z:
                    pos = base + np.array([xi, yi, zi])
                    poses.append([float(pos[0]), float(pos[1]), float(pos[2]), *orient])
        self.logger.debug(f"Generated {len(poses)} grid poses")
        return poses

    def run_grid(self) -> List[List[float]]:
        """Move through the grid and record TCP poses."""
        self.logger.info("Starting grid motion")
        poses: List[List[float]] = []
        for target in self.generate_grid():
            if not self.controller.move_linear(target):
                self.logger.error("Move failed, aborting grid run")
                break
            self.controller.wait_motion_done()
            pose = self.controller.get_tcp_pose()
            if pose is None:
                self.logger.error("Pose read failed")
                continue
            poses.append(pose)
        self.logger.info(f"Grid motion finished, {len(poses)} poses recorded")
        return poses

    def save_poses(self, poses: List[List[float]]) -> Path:
        """Save recorded poses to a timestamped JSON file."""
        out_dir = paths.CAPTURES_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        file = out_dir / f"poses_{timestamp()}.json"
        data = {str(i): {"tcp_coords": p} for i, p in enumerate(poses)}
        save_json(file, data)
        self.logger.info(f"Poses saved to {file}")
        return file
