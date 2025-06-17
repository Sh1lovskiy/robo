# cli/poses_saver.py
"""Interactive tool to capture and save robot poses with depth images."""

import os
import json
import numpy as np
import cv2
from robot.controller import RobotController
from utils.logger import Logger
from utils.config import Config
from vision.realsense import RealSenseCamera
from vision.opencv_utils import OpenCVUtils


class PoseSaver:
    """Abstract base class for saving poses."""

    def save(self, filename: str, pose_id: str, pose: list):
        raise NotImplementedError


class JsonPoseSaver(PoseSaver):
    """Save poses to a JSON file."""

    def save(self, filename: str, pose_id: str, pose: list):
        try:
            if os.path.exists(filename):
                with open(filename, "r") as f:
                    data = json.load(f)
            else:
                data = {}
        except Exception:
            data = {}
        data[pose_id] = {"tcp_coords": pose}
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)


class PoseSaverCLI:
    """Interactive CLI for saving robot poses with synchronized images."""

    def __init__(self, controller=None, saver=None, logger=None, filename=None, ip_address=None):
        Config.load()
        self.captures_dir = Config.get("path_saver.captures_dir", "cloud")
        self.filename = filename or os.path.join(self.captures_dir, "poses.json")
        ip = ip_address or Config.get("robot.ip")
        self.controller = controller or RobotController(rpc=ip)
        self.saver = saver or JsonPoseSaver()
        self.logger = logger or Logger.get_logger("cli.poses_saver")

    def run(self):
        cam = RealSenseCamera()
        cam.start()
        os.makedirs(self.captures_dir, exist_ok=True)
        pose_count = 0
        print("Press ENTER to save current pose. Press 'q' to quit.")

        while True:
            color, depth = cam.get_frames()
            if depth is not None:
                OpenCVUtils.show_depth(depth)
            if color is not None:
                cv2.imshow("RGB Camera Stream", color)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print(f"All poses saved to {self.filename}")
                self.logger.info("Exit requested by user")
                break
            elif key in (13, 10):
                pose = self.controller.get_tcp_pose()
                if pose is not None:
                    pose_id = f"{pose_count:03d}"
                    self.saver.save(self.filename, pose_id, pose)
                    self.logger.info(f"Pose {pose_id} saved")
                    rgb_path = os.path.join(self.captures_dir, f"{pose_id}_rgb.png")
                    depth_path = os.path.join(self.captures_dir, f"{pose_id}_depth.npy")
                    cv2.imwrite(rgb_path, color)
                    np.save(depth_path, depth)
                    pose_count += 1
                else:
                    self.logger.error("Failed to get pose")

        cam.stop()
        cv2.destroyAllWindows()
        self.controller.shutdown()


def main():
    PoseSaverCLI().run()


if __name__ == "__main__":
    main()
