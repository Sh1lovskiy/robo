"""RealSense D415 configuration and wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pyrealsense2 as rs

from utils.logger import Logger, LoggerType
from utils.settings import D415_Cfg, camera
from .camera_base import CameraBase


@dataclass
class D415CameraSettings:
    """Manual exposure and laser settings."""

    ir_exposure: int = 100
    ir_gain: int = 16
    rgb_exposure: int = 100
    rgb_gain: int = 64
    projector_power: int = 0
    max_projector_power: int = 360


@dataclass
class D415FilterConfig:
    """RealSense post-processing options."""

    decimation: int = 100
    spatial_alpha: float = 0.5
    spatial_delta: int = 20
    temporal_alpha: float = 0.4
    temporal_delta: int = 20
    hole_filling: int = 1


class RealSenseD415(CameraBase):
    """RealSense D415 RGB-D camera driver."""

    def __init__(
        self,
        stream_cfg: D415_Cfg | None = None,
        settings: D415CameraSettings | None = None,
        filters: D415FilterConfig | None = None,
        logger: LoggerType | None = None,
    ) -> None:
        """Create a new camera instance.

        Parameters
        ----------
        stream_cfg:
            Optional streaming configuration.  If ``None`` the global
            :data:`utils.settings.camera` is used.
        settings:
            Manual exposure and laser options.
        filters:
            Post-processing filter configuration.
        logger:
            Logger to use for all messages.
        """

        self.stream_cfg = stream_cfg or camera
        self.settings = settings or D415CameraSettings()
        self.filters = filters or D415FilterConfig()
        self.logger = logger or Logger.get_logger("vision.d415")
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.started = False
        self.profile: rs.pipeline_profile | None = None
        self.align: rs.align | None = None
        self.depth_scale: float = 1.0
        self._init_config()

    def _init_config(self) -> None:
        """Pre-configure the pipeline with the desired streams."""

        cfg = self.stream_cfg
        self.config.enable_stream(
            rs.stream.depth,
            cfg.depth_width,
            cfg.depth_height,
            rs.format.z16,
            cfg.fps,
        )
        self.config.enable_stream(
            rs.stream.color,
            cfg.rgb_width,
            cfg.rgb_height,
            rs.format.bgr8,
            cfg.fps,
        )

    def start(self) -> None:
        """Start streaming from the camera."""

        config = rs.config()
        config.enable_stream(
            rs.stream.depth,
            self.stream_cfg.depth_width,
            self.stream_cfg.depth_height,
            rs.format.z16,
            self.stream_cfg.fps,
        )
        config.enable_stream(
            rs.stream.color,
            self.stream_cfg.rgb_width,
            self.stream_cfg.rgb_height,
            rs.format.bgr8,
            self.stream_cfg.fps,
        )
        self.profile = self.pipeline.start(config)
        device = self.profile.get_device()
        sensors = {s.get_info(rs.camera_info.name): s for s in device.sensors}
        self.depth_sensor = sensors.get("Stereo Module")
        self.rgb_sensor = sensors.get("RGB Camera")
        if self.depth_sensor is None or self.rgb_sensor is None:
            raise RuntimeError("Required sensors not found")
        self._apply_settings()
        self.depth_scale = float(self.depth_sensor.get_depth_scale())
        self.logger.info(f"Depth scale: {self.depth_scale:.6f} m/unit")
        if self.stream_cfg.align_to_color:
            self.align = rs.align(rs.stream.color)
        self._log_device_info(device)
        self._setup_filters()
        self.started = True

    def _apply_settings(self) -> None:
        s = self.settings
        self.depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
        power = float(s.projector_power)
        self.depth_sensor.set_option(rs.option.laser_power, power)
        self.rgb_sensor.set_option(rs.option.enable_auto_exposure, 0)
        self.rgb_sensor.set_option(rs.option.exposure, float(s.rgb_exposure))
        self.rgb_sensor.set_option(rs.option.gain, float(s.rgb_gain))

    def _setup_filters(self) -> None:
        """Initialize depth post-processing filters."""

        cfg = self.filters
        self.decimation = rs.decimation_filter()
        self.decimation.set_option(rs.option.filter_magnitude, cfg.decimation)

        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_smooth_alpha, cfg.spatial_alpha)
        self.spatial.set_option(rs.option.filter_smooth_delta, cfg.spatial_delta)

        self.temporal = rs.temporal_filter()
        self.temporal.set_option(rs.option.filter_smooth_alpha, cfg.temporal_alpha)
        self.temporal.set_option(rs.option.filter_smooth_delta, cfg.temporal_delta)

        self.hole_filling = rs.hole_filling_filter(cfg.hole_filling)

    def _log_device_info(self, device: rs.device) -> None:
        name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)
        self.logger.info(f"Device: {name} SN:{serial}")
        depth_stream = self.profile.get_stream(
            rs.stream.depth
        ).as_video_stream_profile()
        color_stream = self.profile.get_stream(
            rs.stream.color
        ).as_video_stream_profile()
        extr = depth_stream.get_extrinsics_to(color_stream)
        R = np.array(extr.rotation).reshape(3, 3)
        t = np.array(extr.translation)
        self.logger.debug(f"Extrinsics depth→color R={R.tolist()} t={t.tolist()}")

    def stop(self) -> None:
        """Stop camera streaming."""

        if self.started:
            self.pipeline.stop()
            self.started = False

    def set_projector(self, enable: bool) -> None:
        """Enable or disable the infrared projector."""

        power = self.settings.max_projector_power if enable else 0
        self.depth_sensor.set_option(rs.option.laser_power, float(power))

    def _process_depth(self, frame: rs.frame) -> rs.frame:
        """Apply post-processing filters to a depth frame."""
        frame = self.decimation.process(frame)
        frame = self.spatial.process(frame)
        frame = self.temporal.process(frame)
        frame = self.hole_filling.process(frame)
        return frame

    def get_frames(
        self, aligned: bool = True
    ) -> Tuple[np.ndarray | None, np.ndarray | None]:
        """Return the next color and depth frame pair."""

        assert self.started, "Camera not started"
        frames = self.pipeline.wait_for_frames()
        if aligned and self.align:
            frames = self.align.process(frames)
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if depth:
            depth = self._process_depth(depth)
        color_img = np.asanyarray(color.get_data()) if color else None
        depth_img = np.asanyarray(depth.get_data()) if depth else None
        return color_img, depth_img


def load_depth_intrinsics_from_camera() -> np.ndarray:
    """Return depth intrinsics as a 3x3 matrix from a temporary pipeline."""

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth)
    cfg.enable_stream(rs.stream.color)
    profile = pipeline.start(cfg)
    try:
        depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        intr = depth_stream.get_intrinsics()
        K = np.array(
            [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return K
    finally:
        pipeline.stop()
