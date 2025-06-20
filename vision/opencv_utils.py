# vision/opencv_utils.py

import cv2
import numpy as np
from utils.io import load_camera_params


class OpenCVUtils:
    """Collection of OpenCV helper functions."""

    @staticmethod
    def draw_text(img, text, pos=(10, 30), color=(0, 255, 0), font_scale=0.8, thickness=2):
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

    @staticmethod
    def show_depth(depth: np.ndarray, window: str = "Depth Colormap") -> None:
        """Display a depth image with a JET colormap."""
        depth_uint8 = OpenCVUtils.normalize_depth(depth)
        depth_color = OpenCVUtils.apply_colormap(depth_uint8)
        cv2.imshow(window, depth_color)

    @staticmethod
    def load_camera_calib_from_xml(filename):
        """Load camera matrix and distortion coefficients from file."""
        return load_camera_params(filename)
