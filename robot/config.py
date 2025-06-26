from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RobotConfig:
    """Configuration for :class:`RobotController`."""

    ip: str = "192.168.58.2"
    tool_id: int = 0
    user_frame_id: int = 0
    velocity: float = 20.0
    restart_delay: float = 1.0
