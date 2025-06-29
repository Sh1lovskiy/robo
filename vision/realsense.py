# vision/realsense.py

"""RealSense camera wrapper with custom controls.

Streams color (1920x1080) and depth (1280x720), disables projector,
sets depth units to 0.0001, disables auto exposure and sets exposure for RGB sensor.
All frames are displayed at 640x480 for UI preview.

TODO: add CI badges for build and coverage.
"""

from __future__ import annotations

import cv2
import numpy as np
import pyrealsense2 as rs
from dataclasses import dataclass
from typing import cast
from utils.error_tracker import CameraConnectionError, ErrorTracker
from utils.logger import Logger, LoggerType
from .camera_base import Camera


@dataclass
class RealSenseConfig:
    color_width: int = 1920
    color_height: int = 1080
    depth_width: int = 1280
    depth_height: int = 720
    fps: int = 30
    enable_color: bool = True
    enable_depth: bool = True
    align_to_color: bool = True
    display_width: int = 640
    display_height: int = 480


class RealSenseCamera(Camera):  # type: ignore[misc]
    """
    Unified interface for Intel RealSense camera.
    Streams color and depth, sets depth units, disables emitter,
    and allows exposure control.
    """

    def __init__(self, cfg: RealSenseConfig, logger: LoggerType | None = None) -> None:
        self.cfg = cfg
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.logger = logger or Logger.get_logger("vision.realsense")

        if cfg.enable_depth:
            self.config.enable_stream(
                rs.stream.depth,
                cfg.depth_width,
                cfg.depth_height,
                rs.format.z16,
                cfg.fps,
            )
        if cfg.enable_color:
            self.config.enable_stream(
                rs.stream.color,
                cfg.color_width,
                cfg.color_height,
                rs.format.bgr8,
                cfg.fps,
            )
        self.enable_color = cfg.enable_color
        self.enable_depth = cfg.enable_depth
        self.align_to_color = cfg.align_to_color
        self.profile: rs.pipeline_profile | None = None
        self.depth_scale: float | None = None
        self.align: rs.align | None = None
        self.started = False

    def start(self) -> None:
        """
        Start streaming and apply sensor configuration.
        """
        try:
            self.profile = self.pipeline.start(self.config)
            dev = self.profile.get_device()
            # Depth sensor config
            if self.enable_depth:
                depth_sensor = dev.first_depth_sensor()
                try:
                    depth_sensor.set_option(rs.option.depth_units, 0.0001)
                    self.logger.info("Depth units set to 0.0001 meters")
                except Exception as e:
                    self.logger.warning(f"Failed to set depth units: {e}")
                self.depth_scale = depth_sensor.get_option(rs.option.depth_units)
                if depth_sensor.supports(rs.option.emitter_enabled):
                    depth_sensor.set_option(rs.option.emitter_enabled, 0)
                    self.logger.info("Depth projector (emitter) disabled")
            # RGB sensor config
            color_sensor = None
            for s in dev.sensors:
                if s.get_info(rs.camera_info.name) == "RGB Camera":
                    color_sensor = s
                    break
            if color_sensor:
                if color_sensor.supports(rs.option.enable_auto_exposure):
                    color_sensor.set_option(rs.option.enable_auto_exposure, 0)
                    self.logger.info("RGB auto exposure disabled")
                if color_sensor.supports(rs.option.exposure):
                    color_sensor.set_option(rs.option.exposure, 50)
                    self.logger.info("RGB exposure set to 50")
            if self.align_to_color and self.enable_color and self.enable_depth:
                self.align = rs.align(rs.stream.color)
            self.started = True
            ErrorTracker.register_cleanup(self.stop)
        except Exception as e:
            self.logger.error(f"Failed to start RealSense pipeline: {e}")
            raise CameraConnectionError(str(e)) from e

    def stop(self) -> None:
        """Stop streaming and release resources."""
        if self.started:
            self.pipeline.stop()
            self.started = False

    def get_frames(
        self, aligned: bool = True
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Returns (color_img, depth_img) as numpy arrays.
        If aligned=True and both streams enabled, aligns depth to color.
        """
        assert self.started, "Camera must be started before getting frames."
        frames = self.pipeline.wait_for_frames()
        if self.align_to_color and aligned and self.enable_color and self.enable_depth:
            assert self.align is not None
            frames = self.align.process(frames)
        color_frame = frames.get_color_frame() if self.enable_color else None
        depth_frame = frames.get_depth_frame() if self.enable_depth else None
        color_img = np.asanyarray(color_frame.get_data()) if color_frame else None
        depth_img = np.asanyarray(depth_frame.get_data()) if depth_frame else None
        return color_img, depth_img

    def get_intrinsics(self) -> dict[str, float | list[float] | str]:
        """
        Get depth camera intrinsics as a dictionary.
        """
        assert self.started, "Camera must be started before getting intrinsics."
        assert self.profile is not None
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
        """Get depth scale in meters per unit."""
        assert self.depth_scale is not None, "Depth scale not available."
        return self.depth_scale


if __name__ == "__main__":
    cam = RealSenseCamera(RealSenseConfig())
    cam.start()
    intr = cam.get_intrinsics()
    print("Intrinsics:", intr)
    width = cast(float, intr["width"])
    assert width > 0

    for _ in range(10):
        color, depth = cam.get_frames()
        assert color is not None and depth is not None
        # Resize for display
        color_disp = cv2.resize(
            color,
            (cam.cfg.display_width, cam.cfg.display_height),
            interpolation=cv2.INTER_AREA,
        )
        # Normalize depth for visualization and resize
        depth_vis = cv2.convertScaleAbs(depth, alpha=0.03)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        depth_disp = cv2.resize(
            depth_vis,
            (cam.cfg.display_width, cam.cfg.display_height),
            interpolation=cv2.INTER_AREA,
        )
        cv2.imshow("Color", color_disp)
        cv2.imshow("Depth", depth_disp)
        if cv2.waitKey(1) == 27:
            break
    cam.stop()
    cv2.destroyAllWindows()
    print("RealSense camera self-test OK")
