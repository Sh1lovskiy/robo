# cli/poses_saver.py

import os
import time
import json
import numpy as np
import cv2
from robot.controller import RobotController
from utils.logger import Logger
from utils.config import Config
from vision.realsense import RealSenseCamera




class PoseSaver:
    """Abstract base class for saving poses."""

    def save(self, filename, pose_id, pose):
        raise NotImplementedError


class JsonPoseSaver(PoseSaver):
    """
    Saves poses in JSON format.
    Structure: {pose_id: {"tcp_coords": [x, y, z, rx, ry, rz]}}
    """

    def save(self, filename, pose_id, pose):
        try:
            dir_ = os.path.dirname(filename)
            if dir_ and not os.path.exists(dir_):
                os.makedirs(dir_, exist_ok=True)
            with open(filename, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {}
        data[pose_id] = {"tcp_coords": pose}
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)


def show_depth(depth):
    depth_vis = np.clip(depth, 0, np.percentile(depth, 99))  # обрежь выбросы
    depth_vis = (depth_vis / depth_vis.max() * 255).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    cv2.imshow("Depth Colormap", depth_colormap)


def main(
    controller: RobotController = None,
    saver: PoseSaver = None,
    logger=None,
    filename: str | None = None,
    ip_address: str | None = None,
):
    """
    Main function to save robot poses using key presses.
    - Shows RGB camera stream.
    - Press ENTER to save current pose & RGB/depth images.
    - Press 'q' to quit.
    """
    Config.load("config.yaml")
    captures_dir = Config.get("path_saver.captures_dir", "cloud")
    filename = filename or os.path.join(captures_dir, "poses.json")
    if ip_address is None:
        ip_address = Config.get("robot.ip", default="192.168.1.10")

    controller = controller or RobotController(rpc=ip_address)
    saver = saver or JsonPoseSaver()
    logger = logger or Logger.get_logger("cli.poses_saver")

    if hasattr(controller, "initialize"):
        controller.initialize()

    # === Init camera ===
    cam = RealSenseCamera()
    cam.start()

    # === Prepare capture dir ===
    os.makedirs(captures_dir, exist_ok=True)

    pose_count = 0
    print("Press ENTER to save current pose. Press 'q' to quit.")

    while True:
        color, depth = cam.get_frames()
        if depth is not None:
            show_depth(depth)
        if color is not None:
            cv2.imshow("RGB Camera Stream", color)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print(f"All poses saved to {filename}")
            logger.info(f"Exit requested by user, saved to {filename}")
            break
        elif key == 13 or key == 10:  # ENTER (Windows/Linux)
            pose = controller.get_tcp_pose()
            if pose is not None:
                pose_id = f"{pose_count:03d}"
                saver.save(filename, pose_id, pose)
                logger.info(f"Pose {pose_id} saved: {pose}")
                print(f"Saved pose {pose_id}: {pose}")

                # --- Save RGB & depth images ---
                rgb_path = os.path.join(captures_dir, f"{pose_id}_rgb.png")
                depth_path = os.path.join(captures_dir, f"{pose_id}_depth.npy")
                cv2.imwrite(rgb_path, color)
                np.save(depth_path, depth)
                logger.info(f"RGB saved: {rgb_path}; Depth saved: {depth_path}")

                pose_count += 1
            else:
                logger.error("Failed to get pose.")

    cam.stop()
    cv2.destroyAllWindows()
    controller.shutdown()


if __name__ == "__main__":
    main()
