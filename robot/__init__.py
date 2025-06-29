"""High-level robot API exports with lazy loading of submodules."""

from importlib import import_module
from typing import TYPE_CHECKING

from .controller import RobotController

__all__ = [
    "RobotController",
    "PoseRecorder",
    "PathRunner",
    "CameraManager",
    "MarkerPathRunner",
    "WaypointRunner",
]

if TYPE_CHECKING:  # pragma: no cover - for static type checking only
    from .workflows import PoseRecorder, PathRunner, CameraManager
    from .marker import MarkerPathRunner
    from .waypoint import WaypointRunner


def __getattr__(name: str):
    if name in {"PoseRecorder", "PathRunner", "CameraManager"}:
        module = import_module(".workflows", __name__)
        return getattr(module, name)
    if name == "MarkerPathRunner":
        module = import_module(".marker", __name__)
        return getattr(module, name)
    if name == "WaypointRunner":
        module = import_module(".waypoint_runner", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name}")
