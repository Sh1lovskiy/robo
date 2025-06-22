"""Small utilities for camera debugging."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from utils.error_tracker import CameraError
from utils.logger import Logger
from vision.realsense import RealSenseCamera


@dataclass
class IntrinsicsPrinter:
    """Print camera intrinsics to stdout."""

    camera: RealSenseCamera = RealSenseCamera()
    logger: Logger = Logger.get_logger("vision.tools.intrinsics")

    def run(self) -> None:
        try:
            self.camera.start()
        except CameraError as e:
            self.logger.error(f"Failed to start camera: {e}")
            return

        try:
            intr = self.camera.get_intrinsics()
            self.logger.info(f"Intrinsics: {intr}")
            for k, v in intr.items():
                print(f"{k}: {v}")
        finally:
            self.camera.stop()


@dataclass
class DepthChecker:
    """Display live depth map with distance overlay."""

    camera: RealSenseCamera = RealSenseCamera()
    logger: Logger = Logger.get_logger("vision.tools.depth")

    def run(self) -> None:
        try:
            self.camera.start()
        except CameraError as e:
            self.logger.error(f"Failed to start camera: {e}")
            return

        depth_scale = self.camera.get_depth_scale()
        self.logger.info(f"Depth scale: {depth_scale:.6f} m")
        try:
            while True:
                color, depth = self.camera.get_frames()
                h, w = depth.shape
                x, y = w // 2, h // 2
                dist_mm = int(depth[y, x] * depth_scale * 1000)
                depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(
                    np.uint8
                )
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                cv2.circle(depth_vis, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(
                    depth_vis,
                    f"{dist_mm} mm",
                    (x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow("Depth", depth_vis)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        finally:
            self.camera.stop()
            cv2.destroyAllWindows()
