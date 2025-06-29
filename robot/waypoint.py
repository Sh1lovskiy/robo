from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, List

from robot.controller import RobotController
from utils.keyboard import GlobalKeyListener, TerminalEchoSuppressor
from utils.logger import Logger, LoggerType


@dataclass
class Waypoint:
    x: float
    y: float
    z: float

    def __post_init__(self):
        self.x *= 1000.0
        self.y *= 1000.0
        self.z *= 1000.0


class WaypointRunner:
    """Move through base frame points with a fixed downward orientation."""

    def __init__(
        self,
        controller: RobotController,
        points: Iterable[Waypoint],
        *,
        rx: float = 180.0,
        ry: float = 0.0,
        rz: float = 0.0,
        height_offset: float = 0.0,
        logger: LoggerType | None = None,
    ) -> None:
        self.controller = controller
        self.points = list(points)
        self.rx = rx
        self.ry = ry
        self.rz = rz
        self.offset = height_offset
        self.logger = logger or Logger.get_logger("robot.waypoint_runner")

    def _make_pose(self, wp: Waypoint) -> List[float]:
        pose = [wp.x, wp.y, wp.z + self.offset, self.rx, self.ry, self.rz]
        return pose

    def _wait_for_user(self, listener: GlobalKeyListener, stop: dict) -> bool:
        listener.start()
        self.logger.info("Press 'n' to continue or 'q' to quit.")
        while not stop.get("next"):
            if stop.get("quit"):
                return False
            time.sleep(0.1)
        return not stop.get("quit")

    def run(self) -> None:
        if not self.controller.connect():
            return
        self.controller.enable()
        stop: dict = {}
        hot = {
            "n": lambda: stop.update(next=True),
            "q": lambda: stop.update(quit=True, next=True),
        }
        listener = GlobalKeyListener(hot, suppress=True)
        suppressor = TerminalEchoSuppressor()
        suppressor.start()
        listener.start()
        try:
            for idx, wp in enumerate(self.points):
                pose = self._make_pose(wp)
                self.logger.info(f"Moving to waypoint {idx}: {pose}")
                if not self.controller.move_linear(pose):
                    break
                self.controller.wait_motion_done()
                stop.clear()
                self.logger.info("Press 'n' to continue or 'q' to quit.")
                while not stop.get("next"):
                    if stop.get("quit"):
                        break
                    time.sleep(0.1)
                if stop.get("quit"):
                    break
        finally:
            listener.stop()
            suppressor.stop()
            self.logger.info("Waypoint run finished")


def main() -> None:
    points = [
        Waypoint(-0.30328, -0.07911, 0.27024),
        Waypoint(-0.30328, -0.07911, 0.26024),
        Waypoint(-0.30328, -0.07911, 0.25024),
        Waypoint(-0.30759, 0.07771, 0.2506),
        Waypoint(-0.30759, 0.07771, 0.2606),
        Waypoint(-0.30759, 0.07771, 0.2706),
    ]
    runner = WaypointRunner(RobotController(), points)
    runner.run()


if __name__ == "__main__":
    main()
