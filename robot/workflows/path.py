"""Run robot trajectories while capturing frames."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Callable, List

from robot.controller import RobotController
from utils.logger import Logger, LoggerType

from .record import CameraManager, FrameSaver


@dataclass
class PathRunner:
    """Execute a trajectory while saving camera frames."""

    controller: RobotController
    camera_mgr: CameraManager
    frame_saver: FrameSaver
    traj_file: str | None = None
    progress_cb: Callable[[int, List[float]], None] | None = None
    logger: LoggerType = Logger.get_logger("robot.workflow.path")

    def run(self) -> None:
        """Synchronously execute the loaded trajectory."""
        path = self._load_path()
        if not path:
            return
        try:
            self.controller.enable()
        except Exception as e:  # pragma: no cover - hardware error
            self.logger.error(f"Failed to enable robot: {e}")
            return
        if not self.camera_mgr.start():
            self.logger.error("Camera not available. Aborting path run.")
            return
        for idx, pose in Logger.progress(list(enumerate(path)), desc="Path"):
            self.logger.info(f"Moving to {pose}")
            if not self.controller.move_linear(pose):
                self.logger.error(f"Movement failed at {idx}")
                break
            time.sleep(0.5)
            color, depth = self.camera_mgr.get_frames()
            self.frame_saver.save(idx, color, depth)
            if self.progress_cb:
                self.progress_cb(idx, pose)
        self.camera_mgr.stop()
        self.logger.info("Path execution finished")

    async def run_async(self) -> None:
        """Asynchronous variant of :meth:`run`."""
        path = self._load_path()
        if not path:
            return
        if not await asyncio.to_thread(self.camera_mgr.start):
            self.logger.error("Camera not available. Aborting path run.")
            return
        for idx, pose in enumerate(path):
            self.logger.info(f"Moving to {pose}")
            ok = await asyncio.to_thread(self.controller.move_linear, pose)
            if not ok:
                self.logger.error(f"Movement failed at {idx}")
                break
            await asyncio.sleep(0.5)
            color, depth = await asyncio.to_thread(self.camera_mgr.get_frames)
            await asyncio.to_thread(self.frame_saver.save, idx, color, depth)
            if self.progress_cb:
                self.progress_cb(idx, pose)
        await asyncio.to_thread(self.camera_mgr.stop)
        self.logger.info("Path execution finished")
