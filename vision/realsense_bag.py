"""RealSense frame recording and filtered playback module."""

from __future__ import annotations

from typing import List

import cv2
import numpy as np
import pyrealsense2 as rs

from utils.logger import Logger, LoggerType
from vision.camera import (
    D415CameraSettings,
    D415FilterConfig,
    D415StreamConfig,
)


class FrameRecorder:
    """Record individual RealSense frames as separate ``.bag`` files."""

    def __init__(
        self,
        output_dir: str = "bags",
        stream_cfg: D415StreamConfig | None = None,
        settings: D415CameraSettings | None = None,
        logger: LoggerType | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stream_cfg = stream_cfg or D415StreamConfig()
        self.settings = settings or D415CameraSettings()
        self.logger = logger or Logger.get_logger("vision.bag_recorder")
        self.pipeline = rs.pipeline()

    # ------------------------------------------------------------------
    def _configure_streams(self, cfg: rs.config) -> None:
        sc = self.stream_cfg
        cfg.enable_stream(
            rs.stream.depth,
            sc.depth_width,
            sc.depth_height,
            rs.format.z16,
            sc.fps,
        )
        cfg.enable_stream(
            rs.stream.color,
            sc.color_width,
            sc.color_height,
            rs.format.bgr8,
            sc.fps,
        )

    def _apply_settings(self, profile: rs.pipeline_profile) -> None:
        device = profile.get_device()
        sensors = {s.get_info(rs.camera_info.name): s for s in device.sensors}
        depth = sensors.get("Stereo Module")
        rgb = sensors.get("RGB Camera")
        if depth is None or rgb is None:
            raise RuntimeError("Required sensors not found")
        depth.set_option(rs.option.enable_auto_exposure, 0)
        depth.set_option(rs.option.exposure, float(self.settings.ir_exposure))
        depth.set_option(rs.option.gain, float(self.settings.ir_gain))
        depth.set_option(rs.option.laser_power, float(self.settings.projector_power))
        rgb.set_option(rs.option.enable_auto_exposure, 0)
        rgb.set_option(rs.option.exposure, float(self.settings.rgb_exposure))
        rgb.set_option(rs.option.gain, float(self.settings.rgb_gain))

    def _start(self, bag: str | None = None) -> rs.pipeline_profile:
        cfg = rs.config()
        self._configure_streams(cfg)
        if bag:
            cfg.enable_record_to_file(str(bag))
        profile = self.pipeline.start(cfg)
        self._apply_settings(profile)
        return profile

    def _stop(self) -> None:
        self.pipeline.stop()

    # ------------------------------------------------------------------
    def record_frames(
        self, num_frames: int = 3, delay: float = 5.0, interval: float = 3.0
    ) -> List[str]:
        """Record ``num_frames`` bags after warmup."""

        self.logger.info("Starting camera warmup")
        profile = self._start()
        start = time.time()
        while time.time() - start < delay:
            self.pipeline.wait_for_frames()
        self._stop()

        bags: List[str] = []
        for i in range(num_frames):
            bag_path = self.output_dir / f"frame_{i:03d}.bag"
            self.logger.info(f"Recording {bag_path}")
            self._start(str(bag_path))
            self.pipeline.wait_for_frames()
            self._stop()
            bags.append(str(bag_path))
            if i < num_frames - 1:
                time.sleep(interval)
        return bags


class BagFilterVisualizer:
    """Play back a bag and show filter stages in separate windows."""

    def __init__(
        self,
        filters: D415FilterConfig | None = None,
        window_size: tuple[int, int] = (640, 480),
        logger: LoggerType | None = None,
    ) -> None:
        self.logger = logger or Logger.get_logger("vision.bag_vis")
        self.filters = filters or D415FilterConfig()
        self.window_size = window_size
        self._init_filters()

    def _init_filters(self) -> None:
        f = self.filters
        self.decimation = rs.decimation_filter()
        self.decimation.set_option(rs.option.filter_magnitude, f.decimation)
        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_smooth_alpha, f.spatial_alpha)
        self.spatial.set_option(rs.option.filter_smooth_delta, f.spatial_delta)
        self.temporal = rs.temporal_filter()
        self.temporal.set_option(rs.option.filter_smooth_alpha, f.temporal_alpha)
        self.temporal.set_option(rs.option.filter_smooth_delta, f.temporal_delta)
        self.holes = rs.hole_filling_filter(f.hole_filling)

    def _frame_to_img(self, frame: rs.depth_frame) -> np.ndarray:
        arr = np.asanyarray(frame.get_data())
        disp = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)
        color = cv2.applyColorMap(disp.astype(np.uint8), cv2.COLORMAP_JET)
        return cv2.resize(color, self.window_size, interpolation=cv2.INTER_AREA)

    def _process(self, depth: rs.depth_frame) -> list[np.ndarray]:
        stages = [self._frame_to_img(depth)]
        d1 = self.decimation.process(depth)
        stages.append(self._frame_to_img(d1))
        d2 = self.spatial.process(d1)
        stages.append(self._frame_to_img(d2))
        d3 = self.temporal.process(d2)
        stages.append(self._frame_to_img(d3))
        d4 = self.holes.process(d3)
        stages.append(self._frame_to_img(d4))
        return stages

    # ------------------------------------------------------------------
    def visualize(self, bag_path: str) -> None:
        """Replay ``bag_path`` and show all filter outputs."""

        cfg = rs.config()
        cfg.enable_device_from_file(bag_path, repeat_playback=False)
        pipe = rs.pipeline()
        profile = pipe.start(cfg)
        align = rs.align(rs.stream.color)
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)

        try:
            while True:
                if playback.current_status() == rs.playback_status.stopped:
                    break
                frames = pipe.wait_for_frames()
                frames = align.process(frames)
                depth = frames.get_depth_frame()
                color = frames.get_color_frame()
                if not depth:
                    continue
                images = self._process(depth)
                if color:
                    color_img = cv2.resize(
                        np.asanyarray(color.get_data()),
                        self.window_size,
                        interpolation=cv2.INTER_AREA,
                    )
                    cv2.imshow("Color", color_img)
                for idx, img in enumerate(images):
                    cv2.imshow(f"Step {idx}", img)
                if cv2.waitKey(1) == 27:
                    break
        finally:
            pipe.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    rec = FrameRecorder()
    bags = rec.record_frames()
    vis = BagFilterVisualizer()
    for b in bags:
        vis.visualize(b)
