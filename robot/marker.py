"""Utility for executing a simple linear marker trajectory."""

from __future__ import annotations

import numpy as np
from robot.controller import RobotController
from utils.logger import Logger


class MarkerPathRunner:
    """Execute a simple approach-move-retreat trajectory."""

    def __init__(
        self, controller: RobotController, logger: Logger | None = None
    ) -> None:
        """Create the runner using an existing :class:`RobotController`."""
        self.controller = controller
        self.logger = logger or Logger.get_logger("marker.path_runner")

    def move_linear(self, pos: list[float]) -> bool:
        """Move the robot linearly to ``pos``."""
        self.logger.info(f"Moving linear to: {pos}")
        return self.controller.move_linear(list(pos))

    def execute_marker_path(
        self,
        start: list[float],
        end: list[float],
        approach_dz: float = 50,
    ) -> bool:
        """Run a short path from ``start`` to ``end`` with an approach height."""
        above_start = list(start[:3])
        above_start[2] += approach_dz
        above_pose = above_start + list(start[3:])
        if not self.move_linear(above_pose):
            self.logger.error("Failed to move above start point.")
            return False
        if not self.move_linear(start):
            self.logger.error("Failed to move to start point.")
            return False
        if not self.move_linear(end):
            self.logger.error("Failed to move to end point.")
            return False
        above_end = list(end[:3])
        above_end[2] += approach_dz
        above_end_pose = above_end + list(end[3:])
        if not self.move_linear(above_end_pose):
            self.logger.error("Failed to move above end point.")
            return False
        self.logger.info("Marker path executed successfully.")
        return True


def main():
    """Simple test routine when running this module directly."""
    Rx, Ry, Rz = 180.0, 0.0, 0.0
    start = [-293.4, -6.83, 277.57, Rx, Ry, Rz]
    end = [-293.4, -6.83, 267.57, Rx, Ry, Rz]
    controller = RobotController()
    runner = MarkerPathRunner(controller)
    runner.execute_marker_path(start, end)


if __name__ == "__main__":
    main()
