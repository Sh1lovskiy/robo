"""Workflow helpers for typical robot tasks.

This package contains utilities used by the ``robot-cli`` entry point: pose
recording, trajectory playback and camera management.  The API is intentionally
lightweight so it can be reused programmatically as well as via the command
line.
"""

from .record import CameraManager, PoseRecorder
from .path import PathRunner
from .cli import create_cli, main

__all__ = [
    "CameraManager",
    "PoseRecorder",
    "PathRunner",
    "create_cli",
    "main",
]
