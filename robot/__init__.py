# """Robot control API and workflow helpers.

# The package exposes :class:`RobotController` as the main programmatic interface
# to the robot along with higher level workflow classes for common tasks such as
# trajectory execution and pose recording.  Optional classes are imported lazily
# to keep import time low on systems without full dependencies.
# """

# from importlib import import_module
# from typing import TYPE_CHECKING

# from .controller import RobotController

# __all__ = [
#     "RobotController",
#     "PoseRecorder",
#     "PathRunner",
#     "CameraManager",
#     "MarkerPathRunner",
#     "WaypointRunner",
# ]

# if TYPE_CHECKING:  # pragma: no cover - for static type checking only
#     from .workflows import PoseRecorder, PathRunner, CameraManager
#     from .marker import MarkerPathRunner
#     from .waypoint import WaypointRunner


# def __getattr__(name: str):
#     if name in {"PoseRecorder", "PathRunner", "CameraManager"}:
#         module = import_module(".workflows", __name__)
#         return getattr(module, name)
#     if name == "MarkerPathRunner":
#         module = import_module(".marker", __name__)
#         return getattr(module, name)
#     if name == "WaypointRunner":
#         module = import_module(".waypoint_runner", __name__)
#         return getattr(module, name)
#     raise AttributeError(f"module {__name__!r} has no attribute {name}")
