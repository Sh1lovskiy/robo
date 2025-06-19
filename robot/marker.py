# marker_path.py
import numpy as np
from robot.controller import RobotController
from utils.logger import Logger


class MarkerPathRunner:
    def __init__(self, controller, logger=None):
        self.controller = controller
        self.logger = logger or Logger.get_logger("marker.path_runner")

    def move_linear(self, pos):
        self.logger.info(f"Moving linear to: {pos}")
        return self.controller.move_linear(list(pos))

    def execute_marker_path(self, start, end, approach_dz=50):
        # Approach above start
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
    Rx, Ry, Rz = 180.0, 0.0, 0.0
    start = [-196.28, 93.6, 275.7, Rx, Ry, Rz]
    end = [-200.05, -37.89, 280.39, Rx, Ry, Rz]
    controller = RobotController()
    runner = MarkerPathRunner(controller)
    runner.execute_marker_path(start, end)


if __name__ == "__main__":
    main()
