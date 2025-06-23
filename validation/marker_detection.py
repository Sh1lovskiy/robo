"""Utilities for ArUco marker detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import cv2
import numpy as np

from utils.logger import Logger, LoggerType


ARUCO_DICT_MAP = {
    "4X4_50": cv2.aruco.DICT_4X4_50,
    "4X4_100": cv2.aruco.DICT_4X4_100,
    "5X5_100": cv2.aruco.DICT_5X5_100,
    "6X6_250": cv2.aruco.DICT_6X6_250,
}


@dataclass
class MarkerDetector:
    """Detect ArUco markers and return center and size."""

    dict_name: str = "5X5_100"
    logger: LoggerType = field(
        default_factory=lambda: Logger.get_logger("validation.marker")
    )

    def __post_init__(self) -> None:
        if self.dict_name not in ARUCO_DICT_MAP:
            raise ValueError(f"Unknown ArUco dictionary: {self.dict_name}")
        self.dictionary = cv2.aruco.getPredefinedDictionary(
            ARUCO_DICT_MAP[self.dict_name]
        )

    def detect(self, image: np.ndarray) -> tuple[np.ndarray | None, float | None]:
        """Return center pixel coordinate and marker side length in pixels."""
        corners, ids, _ = cv2.aruco.detectMarkers(image, self.dictionary)
        if ids is None or len(ids) == 0:
            self.logger.warning("No ArUco markers detected")
            return None, None
        c = corners[0].reshape(-1, 2)
        center = c.mean(axis=0)
        size = float(np.linalg.norm(c[0] - c[2]))
        self.logger.debug(f"Marker detected at {center} size {size}")
        return center, size
