"""Offline depth filtering utilities built with OpenCV."""

import numpy as np
import cv2
from utils.logger import Logger, LoggerType


class DepthOfflineFilters:
    """
    Intel-like depth map filters for .npy files using OpenCV/numpy.
    """

    def __init__(self, logger: LoggerType = None):
        self.logger = logger or Logger.get_logger("vision.depth_offline")

    def decimation(self, depth: np.ndarray, factor: int = 2) -> np.ndarray:
        result = cv2.resize(
            depth,
            (depth.shape[1] // factor, depth.shape[0] // factor),
            interpolation=cv2.INTER_NEAREST,
        )
        self.logger.info("Decimation filter applied.")
        return result

    def spatial(self, depth: np.ndarray, d: int = 5, sigma: float = 50) -> np.ndarray:
        depth32 = depth.astype(np.float32)
        result = cv2.bilateralFilter(depth32, d, sigma, sigma)
        self.logger.info("Spatial(bilateral) filter applied.")
        return result.astype(depth.dtype)

    def temporal(self, depths: list[np.ndarray], alpha: float = 0.4) -> np.ndarray:
        avg = np.copy(depths[0]).astype(np.float32)
        for d in depths[1:]:
            avg = alpha * d.astype(np.float32) + (1 - alpha) * avg
        result = avg.astype(depths[0].dtype)
        self.logger.info("Temporal filter applied.")
        return result

    def hole_filling(self, depth: np.ndarray, method: str = "median") -> np.ndarray:
        mask = (depth == 0).astype(np.uint8)
        if method == "median":
            result = cv2.medianBlur(depth, 5)
            depth[mask == 1] = result[mask == 1]
            self.logger.info("Median hole filling applied.")
            return depth
        if method == "inpaint":
            depth8 = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(
                np.uint8
            )
            result = cv2.inpaint(depth8, mask, 3, cv2.INPAINT_NS)
            self.logger.info("Inpaint hole filling applied.")
            return result.astype(depth.dtype)
        raise ValueError("Unknown method for hole filling.")

    def pipeline(self, depth: np.ndarray) -> list[np.ndarray]:
        steps = []
        steps.append(depth.copy())
        d1 = self.decimation(steps[-1])
        steps.append(d1)
        d2 = self.spatial(d1)
        steps.append(d2)
        d3 = self.hole_filling(d2)
        steps.append(d3)
        return steps

    def show_all(self, steps: list[np.ndarray], labels=None) -> None:
        if labels is None:
            labels = ["Original", "Decimation", "Spatial", "Hole Fill"]
        imgs = []
        for img, lbl in zip(steps, labels):
            norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
            color = cv2.resize(color, (640, 480), interpolation=cv2.INTER_AREA)
            cv2.putText(
                color, lbl, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
            )
            imgs.append(color)
        h, w, _ = imgs[0].shape
        out = np.zeros((2 * h, 2 * w, 3), np.uint8)
        for idx, img in enumerate(imgs):
            r, c = divmod(idx, 2)
            out[r * h : (r + 1) * h, c * w : (c + 1) * w] = img
        cv2.imshow("Offline Depth Pipeline", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys

    arr = np.load(sys.argv[1])
    f = DepthOfflineFilters()
    steps = f.pipeline(arr)
    f.show_all(steps)
