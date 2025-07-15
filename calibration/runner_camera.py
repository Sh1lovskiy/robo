from __future__ import annotations

"""Utility for capturing frames from a RealSense camera."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import cv2

from utils.logger import Logger, LoggerType
from utils.error_tracker import ErrorTracker
from utils.settings import paths, IMAGE_EXT, DEPTH_EXT
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
        """Capture ``count`` frames and save them to disk."""
        self.logger.info(f"Capturing {count} frames")
        out_dir = paths.CAPTURES_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        paths_list: List[Path] = []
        try:
            self.camera.start()
            for i in range(count):
                color, depth = self.camera.get_frames()
                if color is None:
                    self.logger.error("Failed to get color frame")
                    continue
                base = out_dir / f"img_{timestamp()}_{i:04d}"
                cv2.imwrite(str(base.with_suffix(IMAGE_EXT)), color)
                if depth is not None:
                    cv2.imwrite(str(base.with_suffix(DEPTH_EXT)), depth)
                paths_list.append(base.with_suffix(IMAGE_EXT))
                self.logger.debug(f"Frame saved: {base.with_suffix(IMAGE_EXT)}")
            self.logger.info(f"Captured {len(paths_list)} frames")
            return paths_list
        except Exception as exc:
            self.logger.error(f"Capture failed: {exc}")
            ErrorTracker.report(exc)
            return paths_list
        finally:
            self.camera.stop()
