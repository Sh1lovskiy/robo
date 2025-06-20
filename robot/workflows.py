# robot/workflows.py
"""Robot-related workflows: pose recording and path execution."""

from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from robot.controller import RobotController
from utils.config import Config
from utils.logger import Logger
from vision.realsense import RealSenseCamera
from vision.opencv_utils import OpenCVUtils


# --- Pose Recording ---------------------------------------------------------

class PoseSaver:
    """Strategy interface for saving poses."""

    def save(self, filename: str, pose_id: str, pose: List[float]) -> None:
        raise NotImplementedError


class JsonPoseSaver(PoseSaver):
    """Save poses to a JSON file."""

    def save(self, filename: str, pose_id: str, pose: List[float]) -> None:
        if os.path.exists(filename):
            with open(filename, "r") as f:
                data = json.load(f)
        else:
            data = {}
        data[pose_id] = {"tcp_coords": pose}
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)


@dataclass
class IRFrameSaver:
    """Save infrared frames to disk."""

    out_dir: str
    logger: Logger = Logger.get_logger("robot.workflow.ir")

    def save(self, idx: str, ir_img: np.ndarray) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(self.out_dir, f"{idx}_ir.png"), ir_img)
        self.logger.info(f"Saved IR for {idx}")


@dataclass
class PoseRecorder:
    """Interactive pose recorder with synchronized camera frames."""

    controller: RobotController
    saver: PoseSaver
    captures_dir: str
    logger: Logger = Logger.get_logger("robot.workflow.record")

    def run(self) -> None:
        cam = RealSenseCamera()
        cam.start()
        ir_saver = IRFrameSaver(self.captures_dir)
        pose_count = 0
        print("Press ENTER to save current pose. Press 'q' to quit.")
        while True:
            color, depth, ir = self._get_frames(cam)
            if depth is not None:
                OpenCVUtils.show_depth(depth)
            if color is not None:
                cv2.imshow("RGB", color)
            if ir is not None:
                ir_vis = ir if ir.dtype == np.uint8 else cv2.convertScaleAbs(ir, alpha=255.0 / np.max(ir))
                cv2.imshow("IR", ir_vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in (13, 10):
                pose = self.controller.get_tcp_pose()
                if pose:
                    idx = f"{pose_count:03d}"
                    self.saver.save(os.path.join(self.captures_dir, "poses.json"), idx, pose)
                    self._save_frames(idx, color, depth)
                    if ir is not None:
                        ir_saver.save(idx, ir_vis)
                    pose_count += 1
        cam.stop()
        cv2.destroyAllWindows()
        self.controller.shutdown()

    def _get_frames(self, cam: RealSenseCamera):
        result = cam.get_frames()
        if isinstance(result, tuple) and len(result) == 3:
            return result
        color, depth = result
        return color, depth, None

    def _save_frames(self, idx: str, color: np.ndarray, depth: np.ndarray) -> None:
        rgb_path = os.path.join(self.captures_dir, f"{idx}_rgb.png")
        depth_path = os.path.join(self.captures_dir, f"{idx}_depth.npy")
        cv2.imwrite(rgb_path, color)
        np.save(depth_path, depth)
        self.logger.info(f"Saved frames for {idx}")


# --- Path Execution --------------------------------------------------------

@dataclass
class FrameSaver:
    """Save RGB and depth frames to disk."""

    out_dir: str
    logger: Logger = Logger.get_logger("robot.workflow.frames")

    def save(self, idx: int, color: np.ndarray, depth: np.ndarray) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        rgb_path = os.path.join(self.out_dir, f"{idx:03d}_rgb.png")
        depth_path = os.path.join(self.out_dir, f"{idx:03d}_depth.npy")
        cv2.imwrite(rgb_path, color)
        np.save(depth_path, depth)
        self.logger.info(f"Saved {rgb_path} and {depth_path}")


class CameraManager:
    """Wrap RealSense camera for reliable frame acquisition."""

    def __init__(self, logger: Optional[Logger] = None):
        self.cam = RealSenseCamera()
        self.logger = logger or Logger.get_logger("robot.workflow.camera")

    def start(self) -> None:
        self.cam.start()
        for _ in range(10):
            try:
                self.cam.get_frames()
            except Exception:
                pass
            time.sleep(0.05)

    def stop(self) -> None:
        self.cam.stop()

    def get_frames(self) -> tuple[np.ndarray, np.ndarray]:
        for attempt in range(10):
            color, depth = self.cam.get_frames()
            if color is not None and depth is not None:
                return color, depth
            self.logger.debug(f"Frame not ready (attempt {attempt})")
            time.sleep(0.2)
        raise RuntimeError("Failed to get frames")


def load_trajectory(path_file: str) -> List[List[float]]:
    with open(path_file, "r") as f:
        data = json.load(f)
    keys = sorted(data.keys(), key=lambda x: int(x))
    return [data[k]["tcp_coords"] for k in keys]


@dataclass
class PathRunner:
    """Run robot path and capture frames."""

    controller: RobotController
    camera_mgr: CameraManager
    frame_saver: FrameSaver
    traj_file: str
    logger: Logger = Logger.get_logger("robot.workflow.path")

    def run(self) -> None:
        path = load_trajectory(self.traj_file)
        self.camera_mgr.start()
        for idx, pose in enumerate(path):
            self.logger.info(f"Moving to {pose}")
            if not self.controller.move_linear(pose):
                self.logger.error(f"Movement failed at {idx}")
                break
            time.sleep(0.5)
            color, depth = self.camera_mgr.get_frames()
            self.frame_saver.save(idx, color, depth)
        self.camera_mgr.stop()
        self.controller.shutdown()
        self.logger.info("Path execution finished")


def main_record_poses() -> None:
    """CLI entry for pose recording."""
    Config.load()
    captures_dir = Config.get("path_saver.captures_dir", "cloud")
    ip = Config.get("robot.ip")
    recorder = PoseRecorder(RobotController(rpc=ip), JsonPoseSaver(), captures_dir)
    recorder.run()


def main_run_path() -> None:
    """CLI entry for path execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Execute robot path and save frames")
    parser.add_argument("--ip", type=str, help="Robot IP")
    parser.add_argument("--path_file", type=str, default="captures/poses.json")
    parser.add_argument("--out_dir", type=str, default="captures")
    args = parser.parse_args()

    Config.load()
    ip = args.ip or Config.get("robot.ip")
    runner = PathRunner(
        controller=RobotController(rpc=ip),
        camera_mgr=CameraManager(),
        frame_saver=FrameSaver(args.out_dir),
        traj_file=args.path_file,
    )
    runner.run()

