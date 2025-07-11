"""Robot client abstraction for calibration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from utils.logger import Logger, LoggerType
from utils.settings import RobotCfg, robot

from .controller import RobotController


@dataclass
class RobotClient:
    """High level robot wrapper exposing minimal API."""

    cfg: RobotCfg = robot
    logger: LoggerType = field(
        default_factory=lambda: Logger.get_logger("robot.client")
    )

    def __post_init__(self) -> None:
        self.controller = RobotController(cfg=self.cfg, logger=self.logger)
        self.controller.connect()

    def connect(self) -> bool:
        """Ensure the robot connection is valid."""
        return self.controller.connect()

    def move_to(self, position: Sequence[float], orientation: Sequence[float]) -> bool:
        """Move robot linearly to the given pose."""
        pose = [
            float(position[0]),
            float(position[1]),
            float(position[2]),
            *orientation,
        ]
        return self.controller.move_linear(pose)

    def get_pose(self) -> np.ndarray | None:
        """Return current TCP pose."""
        pose = self.controller.get_tcp_pose()
        return np.asarray(pose) if pose is not None else None

    def go_home(self) -> None:
        """Return robot to previously recorded home pose."""
        self.controller.return_to_home()

    def activate_safe_mode(self) -> None:
        """Enable robot motors."""
        self.controller.enable()
