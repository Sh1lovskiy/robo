#!/usr/bin/env python3
"""
Execute robot trajectory and capture synchronized RGB/depth images at each waypoint.
Saves RGB (PNG) and depth (NPY) to the specified output directory.
"""

import os
import time
import json
import numpy as np
import cv2
from robot.controller import RobotController
from utils.logger import Logger
from utils.config import Config
from vision.realsense import RealSenseCamera


class FrameSaver:
    """
    Handles saving of RGB and depth images to disk.
    """

    def __init__(self, out_dir, logger=None):
        self.out_dir = out_dir
        self.logger = logger or Logger.get_logger("cli.path_runner.framesaver")
        os.makedirs(self.out_dir, exist_ok=True)

    def save(self, idx, color_img, depth_img):
        rgb_path = os.path.join(self.out_dir, f"{idx:03d}_rgb.png")
        depth_path = os.path.join(self.out_dir, f"{idx:03d}_depth.npy")
        cv2.imwrite(rgb_path, color_img)
        np.save(depth_path, depth_img)
        self.logger.info(f"Saved RGB to {rgb_path}, depth to {depth_path}")


class CameraManager:
    """
    Wraps RealSenseCamera: start, stop, warmup, and robust frame retrieval.
    """

    def __init__(
        self, expected_shape=(480, 640, 3), expected_depth_shape=(480, 640), logger=None
    ):
        self.cam = RealSenseCamera()
        self.logger = logger or Logger.get_logger("cli.path_runner.camera")
        self.expected_shape = expected_shape
        self.expected_depth_shape = expected_depth_shape

    def start(self):
        self.cam.start()
        self._warmup(n=10)

    def stop(self):
        self.cam.stop()

    def _warmup(self, n=10):
        """
        Grab and discard n frames to stabilize camera auto-exposure etc.
        """
        for _ in range(n):
            try:
                self.cam.get_frames()
            except Exception:
                pass
            time.sleep(0.05)

    def get_valid_frames(self, retries=10):
        """
        Robustly fetch valid color and depth frames with expected shapes.
        """
        for attempt in range(retries):
            try:
                color_img, depth_img = self.cam.get_frames()
                if (
                    color_img is not None
                    and color_img.shape == self.expected_shape
                    and depth_img is not None
                    and depth_img.shape == self.expected_depth_shape
                ):
                    return color_img, depth_img
                else:
                    self.logger.warning(
                        f"Unexpected frame shape: color={getattr(color_img, 'shape', None)}, "
                        f"depth={getattr(depth_img, 'shape', None)} (attempt {attempt+1})"
                    )
            except Exception as e:
                self.logger.warning(
                    f"Camera frame not ready (attempt {attempt+1}): {e}"
                )
            time.sleep(0.2)
        raise RuntimeError("Failed to get valid camera frames after retries")


class TrajectoryLoader:
    """
    Loads trajectory/waypoints from a JSON file.
    """

    def __init__(self, path_file, logger=None):
        self.path_file = path_file
        self.logger = logger or Logger.get_logger("cli.path_runner.trajloader")

    def load(self):
        with open(self.path_file, "r") as f:
            data = json.load(f)
        pose_ids = sorted(data.keys(), key=lambda x: int(x))
        path = [data[k]["tcp_coords"] for k in pose_ids]
        self.logger.info(f"Loaded {len(path)} waypoints from {self.path_file}")
        return path


class PathRunner:
    """
    Executes a robot path and captures RGB/depth images at each waypoint.
    """

    def __init__(self, controller, camera_mgr, frame_saver, traj_loader, logger=None):
        self.controller = controller
        self.camera_mgr = camera_mgr
        self.frame_saver = frame_saver
        self.traj_loader = traj_loader
        self.logger = logger or Logger.get_logger("cli.path_runner")

    def run(self):
        path = self.traj_loader.load()
        self.camera_mgr.start()

        for idx, pose in enumerate(path):
            self.logger.info(f"Moving to pose {idx}: {pose}")
            if not self.controller.move_linear(pose):
                self.logger.error(f"Movement to pose {idx} failed.")
                break

            time.sleep(0.5)  # Stabilize before capturing

            try:
                color_img, depth_img = self.camera_mgr.get_valid_frames()
            except RuntimeError as e:
                self.logger.error(str(e))
                continue

            self.frame_saver.save(idx, color_img, depth_img)

        self.camera_mgr.stop()
        self.controller.shutdown()
        self.logger.info("Path execution complete.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Execute robot path and save captures")
    parser.add_argument("--ip", type=str, help="Robot controller IP address")
    parser.add_argument(
        "--path_file",
        type=str,
        default="captures/poses.json",
        help="Path to poses JSON",
    )
    parser.add_argument(
        "--out_dir", type=str, default="captures", help="Output directory for captures"
    )
    args = parser.parse_args()

    Config.load()
    robot_cfg = Config.get("robot", {})
    ip_address = args.ip or robot_cfg.get("ip", "192.168.58.2")

    controller = RobotController(rpc=ip_address)
    camera_mgr = CameraManager()
    frame_saver = FrameSaver(args.out_dir)
    traj_loader = TrajectoryLoader(args.path_file)

    runner = PathRunner(controller, camera_mgr, frame_saver, traj_loader)
    runner.run()


def main1():
    from robot.controller import RobotController

    controller = RobotController()
    pose1 = [320, -300, 320, -40, -10, -150]
    pose2 = [330, -310, 320, -40, -10, -150]
    controller.move_linear(pose1)
    controller.move_linear(pose2)


if __name__ == "__main__":
    main1()
