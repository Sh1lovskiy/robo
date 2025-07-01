"""Abstract camera interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class CameraBase(ABC):
    """Minimal camera control API."""

    @abstractmethod
    def start(self) -> None:
        """Start streaming."""

    @abstractmethod
    def stop(self) -> None:
        """Stop streaming."""

    @abstractmethod
    def get_frames(
        self, aligned: bool = True
    ) -> Tuple[np.ndarray | None, np.ndarray | None]:
        """Return color and depth frames."""
