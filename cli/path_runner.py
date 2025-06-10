# cli/path_runner.py

import numpy as np
from robot.controller import RobotController
from utils.logger import Logger


class PathRunner:
    """
    Executes a robot path from file using RobotController.
    """

    def __init__(self, controller=None, logger=None):
        self.controller = controller or RobotController()
        self.logger = logger or Logger.get_logger("cli.path_runner")

    def run(self, path_file="path.txt"):
        self.controller.initialize()
        path = np.loadtxt(path_file)
        for i, pose in enumerate(path):
            self.logger.info(f"Moving to pose {i}: {pose}")
            if not self.controller.move_linear(pose):
                self.logger.error(f"Movement to pose {i} failed.")
                break
        self.controller.shutdown()
        print("Path execution complete.")


def main():
    runner = PathRunner()
    runner.run()


if __name__ == "__main__":
    main()
