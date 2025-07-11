from __future__ import annotations

"""Capture synchronized robot poses and camera frames for calibration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import cv2
import json
import numpy as np

from utils.logger import Logger, LoggerType
from utils.error_tracker import ErrorTracker
from utils.settings import paths, IMAGE_EXT, DEPTH_EXT
from .camera_runner import CameraRunner
from .robot_runner import RobotRunner
from .utils import timestamp


@dataclass
class DataCollector:
    """Synchronize robot and camera data capture."""

    robot: RobotRunner | None = None
    camera: CameraRunner = field(default_factory=CameraRunner)
    logger: LoggerType = field(
        default_factory=lambda: Logger.get_logger("calibration.collector")
    )

    def collect_handeye(self) -> Tuple[List[Path], Path]:
        """Capture synchronized robot poses and images."""
        assert self.robot is not None, "RobotRunner required for hand-eye"
        self.logger.info("Collecting hand-eye data")

        # Generate and save grid
        grid_poses = self.robot.generate_grid()
        grid_file = self.robot.save_poses(grid_poses)

        # Read saved poses for execution
        with open(grid_file) as f:
            all_targets = [v["tcp_coords"] for v in json.load(f).values()]

        if not all_targets:
            self.logger.warning("No target poses found in saved grid file")
            return [], Path()

        images: List[Path] = []
        collected_poses: List[List[float]] = []
        out_dir = paths.CAPTURES_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.camera.camera.start()

            for idx, target in enumerate(Logger.progress(all_targets, desc="capture")):
                self._capture_pose(idx, target, out_dir, images, collected_poses)

            if not collected_poses:
                self.logger.warning("No valid robot poses collected!")

            self.logger.info("Hand-eye data collection finished")
            poses_file = self.robot.save_poses(collected_poses)
            return images, poses_file

        except Exception as exc:
            self.logger.error(f"Data collection failed: {exc}")
            ErrorTracker.report(exc)
            return images, Path()

        finally:
            self.camera.camera.stop()

    def _capture_pose(
        self,
        idx: int,
        target: list[float],
        out_dir: Path,
        images: List[Path],
        collected_poses: List[List[float]],
    ) -> None:
        """Move robot to ``target`` and capture one frame."""
        target_np = np.array(target, dtype=np.float64)
        self.logger.info(f"[{idx}] Moving to pose: {target}")
        self.robot.controller.enable()
        self.robot.controller.connect()
        if not self.robot.controller.move_linear(target_np):
            self.logger.error(f"Move failed to: {target}")
            return
        self.robot.controller.wait_motion_done()
        pose = self.robot.controller.get_tcp_pose()
        if pose is None:
            self.logger.error("Pose read failed")
            return
        color, depth = self.camera.camera.get_frames()
        if color is None:
            self.logger.error("Image capture failed")
            return
        base = out_dir / f"frame_{timestamp()}_{idx:04d}"
        cv2.imwrite(str(base.with_suffix(IMAGE_EXT)), color)
        if depth is not None:
            cv2.imwrite(str(base.with_suffix(DEPTH_EXT)), depth)
        self.logger.info(f"Saved image: {base.with_suffix(IMAGE_EXT)}")
        images.append(base.with_suffix(IMAGE_EXT))
        collected_poses.append(pose)

    def collect_images(self, count: int) -> List[Path]:
        """Capture ``count`` images without robot."""
        return self.camera.capture(count)
