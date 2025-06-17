# cli/path_runner.py
"""Execute stored robot trajectories and capture camera frames."""

import os
import time
import json
import numpy as np
import cv2
from robot.controller import RobotController
from utils.logger import Logger
from utils.config import Config
from vision.realsense import RealSenseCamera


class PathRunner:
    """
    Executes a robot path from file using RobotController.
    Captures RGB (PNG) and depth (NPY) images at each waypoint using RealSenseCamera.
    """

    def __init__(
        self,
        controller=None,
        logger=None,
        out_dir=None,
        path_file=None,
        ip_address=None,
    ):
        Config.load()
        robot_cfg = Config.get("robot")
        path_cfg = Config.get("path_runner", {})

        self.ip_address = ip_address or robot_cfg.get("ip", "192.168.58.2")
        self.path_file = path_file or path_cfg.get("path_file", "poses.json")
        self.out_dir = out_dir or path_cfg.get("out_dir", "captures")

        self.controller = controller or RobotController(rpc=self.ip_address)
        self.logger = logger or Logger.get_logger("cli.path_runner")
        self.cam = RealSenseCamera()
        os.makedirs(self.out_dir, exist_ok=True)

    def _warmup_camera(self, n=10):
        """
        Warm up the RealSense camera pipeline by grabbing and dropping n frames.
        """
        for _ in range(n):
            try:
                self.cam.get_frames()
            except Exception:
                pass
            time.sleep(0.05)

    def _get_valid_frames(
        self, retries=10, expected_shape=(480, 640, 3), expected_depth_shape=(480, 640)
    ):
        """
        Try to get valid RGB and depth frames. Ensures frames have expected shape.
        """
        for attempt in range(retries):
            try:
                color_img, depth_img = self.cam.get_frames()
                # Проверка размеров, чтобы не было обрезанных снимков
                if (
                    color_img is not None
                    and color_img.shape == expected_shape
                    and depth_img is not None
                    and depth_img.shape == expected_depth_shape
                ):
                    return color_img, depth_img
                else:
                    self.logger.warning(
                        f"Got frames of unexpected shape: "
                        f"color={getattr(color_img, 'shape', None)}, "
                        f"depth={getattr(depth_img, 'shape', None)} (attempt {attempt+1})"
                    )
            except Exception as e:
                self.logger.warning(
                    f"Camera frame not ready (attempt {attempt+1}): {e}"
                )
            time.sleep(0.2)
        raise RuntimeError("Failed to get valid camera frames after retries")

    def run(self, path_file=None):
        path_file = path_file or self.path_file
        self.cam.start()
        self._warmup_camera(n=10)
        with open(path_file, "r") as f:
            data = json.load(f)
        pose_ids = sorted(data.keys(), key=lambda x: int(x))
        path = [data[k]["tcp_coords"] for k in pose_ids]

        for i, pose in enumerate(path):
            self.logger.info(f"Moving to pose {i}: {pose}")
            if not self.controller.move_linear(pose):
                self.logger.error(f"Movement to pose {i} failed.")
                break

            # Задержка для стабилизации камеры и позы
            time.sleep(0.5)

            try:
                color_img, depth_img = self._get_valid_frames()
            except RuntimeError as e:
                self.logger.error(str(e))
                continue

            rgb_path = os.path.join(self.out_dir, f"{i:03d}_rgb.png")
            depth_npy_path = os.path.join(self.out_dir, f"{i:03d}_depth.npy")

            cv2.imwrite(rgb_path, color_img)
            np.save(depth_npy_path, depth_img)
            self.logger.info(f"Saved RGB to {rgb_path}, depth to {depth_npy_path}")

        self.cam.stop()
        self.controller.shutdown()
        print("Path execution complete.")


def main():
    import sys

    ip = sys.argv[1] if len(sys.argv) > 1 else None
    path_file = sys.argv[2] if len(sys.argv) > 2 else None
    out_dir = sys.argv[3] if len(sys.argv) > 3 else None
    runner = PathRunner(ip_address=ip, path_file=path_file, out_dir=out_dir)
    runner.run()


if __name__ == "__main__":
    main()
