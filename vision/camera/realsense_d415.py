"""RealSense D415 configuration and wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pyrealsense2 as rs

from utils.logger import Logger, LoggerType
from utils.settings import DEPTH_SCALE
from .camera_base import CameraBase


@dataclass
class D415StreamConfig:
    """Resolution and frame rate parameters."""

    depth_width: int = 1280
    depth_height: int = 720
    color_width: int = 1920
    color_height: int = 1080
    fps: int = 30
    align_to_color: bool = True


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
    """Wrapper around ``pyrealsense2`` exposing tuned settings."""

    def __init__(
        self,
        stream_cfg: D415StreamConfig | None = None,
        settings: D415CameraSettings | None = None,
        filters: D415FilterConfig | None = None,
        logger: LoggerType | None = None,
    ) -> None:
        self.stream_cfg = stream_cfg or D415StreamConfig()
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
            cfg.color_width,
            cfg.color_height,
            rs.format.bgr8,
            cfg.fps,
        )

    def start(self) -> None:
        self.profile = self.pipeline.start(self.config)
        device = self.profile.get_device()
        sensors = {s.get_info(rs.camera_info.name): s for s in device.sensors}
        self.depth_sensor = sensors.get("Stereo Module")
        self.rgb_sensor = sensors.get("RGB Camera")
        if self.depth_sensor is None or self.rgb_sensor is None:
            raise RuntimeError("Required sensors not found")
        self._apply_settings()
        self.depth_scale = DEPTH_SCALE
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
        pass

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
        if self.started:
            self.pipeline.stop()
            self.started = False

    def set_projector(self, enable: bool) -> None:
        power = self.settings.max_projector_power if enable else 0
        self.depth_sensor.set_option(rs.option.laser_power, float(power))

    def _process_depth(self, frame: rs.frame) -> rs.frame:
        return frame

    def get_frames(
        self, aligned: bool = True
    ) -> Tuple[np.ndarray | None, np.ndarray | None]:
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
