"""Camera client wrapper for calibration workflows."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from utils.logger import Logger, LoggerType
from utils.settings import D415_Cfg, camera

from .camera import RealSenseD415


@dataclass
class CameraClient:
    """High level camera wrapper using :class:`RealSenseD415`."""

    cfg: D415_Cfg = camera
    logger: LoggerType = field(
        default_factory=lambda: Logger.get_logger("vision.camera_client")
    )

    def __post_init__(self) -> None:
        self.device = RealSenseD415(stream_cfg=self.cfg, logger=self.logger)

    def connect(self) -> None:
        """Start the camera pipeline."""
        self.device.start()

    def get_color_image(self) -> np.ndarray | None:
        """Return the latest color image."""
        color, _ = self.device.get_frames()
        return color

    def get_depth_image(self) -> np.ndarray | None:
        """Return the latest depth image."""
        _, depth = self.device.get_frames()
        return depth

    def get_intrinsics(self) -> np.ndarray:
        """Return 3x3 camera intrinsic matrix."""
        profile = self.device.profile
        if profile is None:
            raise RuntimeError("Camera not started")
        stream = profile.get_stream(2)  # rs.stream.color
        intr = stream.as_video_stream_profile().get_intrinsics()
        K = np.array(
            [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=float
        )
        return K

    def get_state(self) -> dict:
        """Return simple state dictionary."""
        return {"started": self.device.started}

    @property
    def depth_scale(self) -> float:
        """Depth scale in meters per unit."""
        return float(self.device.depth_scale)

    def disconnect(self) -> None:
        """Stop the camera pipeline."""
        self.device.stop()
