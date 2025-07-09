from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import cv2

from utils.logger import Logger, LoggerType
from utils.settings import paths
from vision.camera.realsense_d415 import RealSenseD415
from .utils import timestamp


@dataclass
class CameraRunner:
    """Acquire and save frames from a RealSense camera."""

    camera: RealSenseD415 = field(default_factory=RealSenseD415)
    logger: LoggerType = field(
        default_factory=lambda: Logger.get_logger("calibration.camera_runner")
    )

    def capture(self, count: int) -> List[Path]:
        """Capture ``count`` frames and store them in ``captures`` directory."""
        out_dir = paths.CAPTURES_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        self.camera.start()
        paths_list: List[Path] = []
        for i in range(count):
            color, depth = self.camera.get_frames()
            if color is None:
                self.logger.error("Failed to get color frame")
                continue
            base = out_dir / f"img_{timestamp()}_{i:04d}"
            cv2.imwrite(str(base.with_suffix(".png")), color)
            if depth is not None:
                cv2.imwrite(str(base.with_suffix("_depth.png")), depth)
            paths_list.append(base.with_suffix(".png"))
            self.logger.debug(f"Frame saved: {base.with_suffix('.png')}")
        self.camera.stop()
        self.logger.info(f"Captured {len(paths_list)} frames")
        return paths_list
