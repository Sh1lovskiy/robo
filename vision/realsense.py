# vision/realsense.py

from __future__ import annotations

import cv2
import numpy as np
import pyrealsense2 as rs
from utils.logger import Logger


class RealSenseCamera:
    """
    Unified interface for Intel RealSense camera.
    Provides methods for streaming, frame alignment, and getting intrinsics.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        enable_color: bool = True,
        enable_depth: bool = True,
        align_to_color: bool = True,
        logger: Logger | None = None,
    ) -> None:
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.enable_color = enable_color
        self.enable_depth = enable_depth
        self.align_to_color = align_to_color
        self.logger = logger or Logger.get_logger("vision.realsense")
        if enable_depth:
            self.config.enable_stream(
                rs.stream.depth, width, height, rs.format.z16, fps
            )
        if enable_color:
            self.config.enable_stream(
                rs.stream.color, width, height, rs.format.bgr8, fps
            )
        self.profile = None
        self.depth_scale = None
        self.align = None
        self.started = False

    def start(self) -> None:
        """
        Start streaming from the camera.
        """
        try:
            self.profile = self.pipeline.start(self.config)
            if self.enable_depth:
                depth_sensor = self.profile.get_device().first_depth_sensor()
                self.depth_scale = depth_sensor.get_depth_scale()
                self.logger.info(f"Depth scale: {self.depth_scale:.6f} meters per unit")
            if self.align_to_color:
                self.align = rs.align(rs.stream.color)
            self.started = True
        except Exception as e:
            self.logger.error(f"Failed to start RealSense pipeline: {e}")
            raise

    def stop(self) -> None:
        """
        Stop streaming and release resources.
        """
        if self.started:
            self.pipeline.stop()
            self.started = False

    def get_frames(
        self, aligned: bool = True
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Returns (color_img, depth_img) as numpy arrays (None if not enabled).
        If aligned=True and both streams enabled, aligns depth to color.
        """
        assert self.started, "Camera must be started before getting frames."
        frames = self.pipeline.wait_for_frames()
        if self.align_to_color and aligned and self.enable_color and self.enable_depth:
            frames = self.align.process(frames)
        color_frame = frames.get_color_frame() if self.enable_color else None
        depth_frame = frames.get_depth_frame() if self.enable_depth else None
        color_img = np.asanyarray(color_frame.get_data()) if color_frame else None
        depth_img = np.asanyarray(depth_frame.get_data()) if depth_frame else None
        return color_img, depth_img

    def get_intrinsics(self) -> dict:
        """
        Get depth camera intrinsics as a dictionary.
        """
        assert self.started, "Camera must be started before getting intrinsics."
        depth_stream = self.profile.get_stream(
            rs.stream.depth
        ).as_video_stream_profile()
        intr = depth_stream.get_intrinsics()
        return {
            "width": intr.width,
            "height": intr.height,
            "ppx": intr.ppx,
            "ppy": intr.ppy,
            "fx": intr.fx,
            "fy": intr.fy,
            "model": str(intr.model),
            "coeffs": list(intr.coeffs),
        }

    def get_depth_scale(self) -> float:
        """
        Get depth scale in meters per unit.
        """
        assert self.depth_scale is not None, "Depth scale not available."
        return self.depth_scale


if __name__ == "__main__":
    cam = RealSenseCamera()
    cam.start()
    intr = cam.get_intrinsics()
    print("Intrinsics:", intr)
    assert "width" in intr and intr["width"] > 0

    for _ in range(10):
        color, depth = cam.get_frames()
        assert color is not None and depth is not None
        print("Frame shapes:", color.shape, depth.shape)
        cv2.imshow("Color", color)
        cv2.imshow("Depth", (depth / depth.max() * 255).astype(np.uint8))
        if cv2.waitKey(1) == 27:
            break
    cam.stop()
    cv2.destroyAllWindows()
    print("RealSense camera self-test OK")
