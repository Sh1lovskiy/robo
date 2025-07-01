"""Common OpenCV helper functions used for visualization."""

from __future__ import annotations

import cv2
import numpy as np
from calibration.charuco import load_camera_params


class OpenCVUtils:
    """Collection of OpenCV helper functions."""

    def __init__(self, display_width: int = 640, display_height: int = 480):
        """Set display frame size for visualization."""
        self.display_width = display_width
        self.display_height = display_height

    @staticmethod
    def draw_text(
        img: np.ndarray,
        text: str,
        pos: tuple[int, int] = (10, 30),
        color: tuple[int, int, int] = (0, 255, 0),
        font_scale: float = 0.8,
        thickness: int = 2,
    ) -> None:
        """Draw text on the image at the specified position."""
        cv2.putText(
            img,
            text,
            pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def normalize_depth(depth: np.ndarray) -> np.ndarray:
        """Normalize a depth map to 0–255 uint8 for visualization."""
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        return depth_norm.astype(np.uint8)

    @staticmethod
    def apply_colormap(depth_uint8: np.ndarray) -> np.ndarray:
        """Apply the JET colormap to an 8-bit depth image."""
        return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)

    def show_depth(self, depth: np.ndarray, window: str = "Depth Colormap") -> None:
        """Display a depth image with a JET colormap and resize to target window."""
        depth_uint8 = self.normalize_depth(depth)
        depth_color = self.apply_colormap(depth_uint8)
        depth_disp = cv2.resize(
            depth_color,
            (self.display_width, self.display_height),
            interpolation=cv2.INTER_AREA,
        )
        cv2.imshow(window, depth_disp)

    @staticmethod
    def load_camera_calib_from_xml(
        filename: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load camera matrix and distortion coefficients from file."""
        return load_camera_params(filename)
