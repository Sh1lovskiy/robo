"""Interactive RealSense depth filter visualizer."""

import numpy as np
import pyrealsense2 as rs
import cv2
from utils.logger import Logger, LoggerType


class DepthFilterVisualizer:
    """
    Apply RS depth filters step-by-step to .npy map and show all results in OpenCV.
    """

    def __init__(
        self,
        logger: LoggerType = None,
        decimation: int = 2,
        spatial_alpha: float = 0.5,
        spatial_delta: int = 20,
        temporal_alpha: float = 0.4,
        temporal_delta: int = 20,
        hole_filling: int = 1,
        use_disparity: bool = False,
    ):
        self.logger = logger or Logger.get_logger("vision.depth_filter_vis")
        self.decimation = rs.decimation_filter()
        self.decimation.set_option(rs.option.filter_magnitude, decimation)
        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_smooth_alpha, spatial_alpha)
        self.spatial.set_option(rs.option.filter_smooth_delta, spatial_delta)
        self.temporal = rs.temporal_filter()
        self.temporal.set_option(rs.option.filter_smooth_alpha, temporal_alpha)
        self.temporal.set_option(rs.option.filter_smooth_delta, temporal_delta)
        self.holes = rs.hole_filling_filter(hole_filling)
        self.use_disparity = use_disparity
        if use_disparity:
            self.to_disp = rs.disparity_transform(True)
            self.from_disp = rs.disparity_transform(False)
            self.logger.info("Disparity transform will be used.")
        else:
            self.to_disp = None
            self.from_disp = None

    def npy_to_frame(self, depth: np.ndarray) -> rs.frame:
        if depth.dtype != np.uint16:
            raise ValueError("Input depth must be uint16 (z16).")
        h, w = depth.shape
        dev = rs.software_device()
        sensor = dev.add_sensor("Depth")
        stream = sensor.add_video_stream(rs.stream.depth, 0, 0, w, h, rs.format.z16, 30)
        dev.create_matcher(rs.matchers.default)
        sensor.open(stream)
        sensor.start(None)
        frame = rs.py_realsense2.py_depth_frame_from_buffer(
            depth.tobytes(), w, h, rs.format.z16
        )
        sensor.on_video_frame(frame)
        for f in dev.poll_for_frames():
            if f.is_depth_frame():
                return f
        raise RuntimeError("No depth frame returned from software device.")

    def frame_to_np(self, frame: rs.frame) -> np.ndarray:
        return np.asanyarray(frame.get_data())

    def visualize_pipeline(self, depth: np.ndarray) -> None:
        h, w = depth.shape
        orig = self.frame_to_np(self.npy_to_frame(depth))
        images = [self._resize_colormap(orig, "Original")]

        f = self.npy_to_frame(depth)
        f1 = self.decimation.process(f)
        images.append(self._resize_colormap(self.frame_to_np(f1), "Decimation"))

        if self.use_disparity:
            f2 = self.to_disp.process(f1)
            images.append(self._resize_colormap(self.frame_to_np(f2), "To Disparity"))
        else:
            f2 = f1

        f3 = self.spatial.process(f2)
        images.append(self._resize_colormap(self.frame_to_np(f3), "Spatial"))

        f4 = self.temporal.process(f3)
        images.append(self._resize_colormap(self.frame_to_np(f4), "Temporal"))

        if self.use_disparity:
            f5 = self.from_disp.process(f4)
            images.append(self._resize_colormap(self.frame_to_np(f5), "From Disparity"))
        else:
            f5 = f4

        f6 = self.holes.process(f5)
        images.append(self._resize_colormap(self.frame_to_np(f6), "Hole Fill"))

        self._show_all(images)

    def _resize_colormap(self, depth: np.ndarray, label: str) -> np.ndarray:
        d8 = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cimg = cv2.applyColorMap(d8, cv2.COLORMAP_JET)
        cimg = cv2.resize(cimg, (640, 480), interpolation=cv2.INTER_AREA)
        cv2.putText(
            cimg, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )
        return cimg

    def _show_all(self, images: list[np.ndarray]) -> None:
        n = len(images)
        rows = 2
        cols = (n + 1) // 2
        h, w, _ = images[0].shape
        canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
        for idx, img in enumerate(images):
            r, c = divmod(idx, cols)
            canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = img
        cv2.imshow("Depth Filtering Pipeline", canvas)
        self.logger.info("Showing all processing stages. Press any key to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys

    npy_file = sys.argv[1]
    arr = np.load(npy_file)
    vis = DepthFilterVisualizer()
    vis.visualize_pipeline(arr)
