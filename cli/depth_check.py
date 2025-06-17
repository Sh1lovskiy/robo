# cli/depth_check.py
"""Display live depth stream for camera tuning."""

import cv2
import numpy as np
from vision.realsense import RealSenseCamera
from utils.logger import Logger


class DepthCheckCLI:
    """Displays depth map and distance at image center from RealSense."""

    def __init__(self, camera=None, logger=None):
        self.camera = camera or RealSenseCamera()
        self.logger = logger or Logger.get_logger("cli.depth_check")

    def run(self):
        self.camera.start()
        depth_scale = self.camera.get_depth_scale()
        self.logger.info(f"Depth scale: {depth_scale:.6f} meters per unit")

        try:
            while True:
                color, depth = self.camera.get_frames()
                h, w = depth.shape
                x, y = w // 2, h // 2
                distance_m = depth[y, x] * depth_scale
                distance_mm = int(distance_m * 1000)

                depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(
                    np.uint8
                )
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                cv2.circle(depth_vis, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(
                    depth_vis,
                    f"{distance_mm} mm",
                    (x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow("Depth", depth_vis)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        finally:
            self.camera.stop()
            cv2.destroyAllWindows()


def main():
    DepthCheckCLI().run()


if __name__ == "__main__":
    main()
