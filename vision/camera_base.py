from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class Camera(ABC):
    """Abstract camera interface for vision modules."""

    @abstractmethod
    def start(self) -> None:
        """Open the camera and start streaming."""
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """Stop streaming and release resources."""
        raise NotImplementedError

    @abstractmethod
    def get_frames(
        self, aligned: bool = True
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return color and depth frames."""
        raise NotImplementedError

    @abstractmethod
    def get_intrinsics(self) -> dict:
        """Return intrinsic parameters as a dictionary."""
        raise NotImplementedError

    @abstractmethod
    def get_depth_scale(self) -> float:
        """Return depth scale in meters per unit."""
        raise NotImplementedError
