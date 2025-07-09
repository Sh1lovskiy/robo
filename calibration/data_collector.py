from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import cv2

from utils.logger import Logger, LoggerType
from utils.settings import paths
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
        assert self.robot is not None, "RobotRunner required for hand-eye"
        self.camera.camera.start()
        poses: List[List[float]] = []
        images: List[Path] = []
        out_dir = paths.CAPTURES_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, target in enumerate(self.robot.generate_grid()):
            if not self.robot.controller.move_linear(target):
                self.logger.error("Move failed")
                break
            self.robot.controller.wait_motion_done()
            pose = self.robot.controller.get_tcp_pose()
            if pose is None:
                self.logger.error("Pose read failed")
                continue
            color, depth = self.camera.camera.get_frames()
            if color is None:
                self.logger.error("Image capture failed")
                continue
            base = out_dir / f"frame_{timestamp()}_{idx:04d}"
            cv2.imwrite(str(base.with_suffix(".png")), color)
            if depth is not None:
                cv2.imwrite(str(base.with_suffix("_depth.png")), depth)
            images.append(base.with_suffix(".png"))
            poses.append(pose)
        self.camera.camera.stop()
        poses_file = self.robot.save_poses(poses)
        return images, poses_file

    def collect_images(self, count: int) -> List[Path]:
        return self.camera.capture(count)
