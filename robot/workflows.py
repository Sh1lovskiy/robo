# robot/workflows.py
"""Robot-related workflows: pose recording and path execution."""

from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass
from typing import List, Optional
import argparse

import cv2
import numpy as np

from robot.controller import RobotController
from utils.cli import Command, CommandDispatcher
from utils.config import Config
from utils.error_tracker import CameraError
from utils.logger import Logger
from vision.realsense import RealSenseCamera
from vision.camera_base import Camera
from vision.opencv_utils import OpenCVUtils


# --- Pose Recording ---


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
        camera_mgr = CameraManager()
        if not camera_mgr.start():
            self.logger.error("Camera not available. Exiting pose recorder.")
            return
        ir_saver = IRFrameSaver(self.captures_dir)
        pose_count = 0
        print("Press ENTER to save current pose. Press 'q' to quit.")
        while True:
            color, depth = camera_mgr.get_frames()
            ir = None
            if depth is not None:
                OpenCVUtils.show_depth(depth)
            if color is not None:
                cv2.imshow("RGB", color)
            if ir is not None:
                ir_vis = (
                    ir
                    if ir.dtype == np.uint8
                    else cv2.convertScaleAbs(ir, alpha=255.0 / np.max(ir))
                )
                cv2.imshow("IR", ir_vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in (13, 10):
                pose = self.controller.get_tcp_pose()
                if pose:
                    idx = f"{pose_count:03d}"
                    self.saver.save(
                        os.path.join(self.captures_dir, "poses.json"), idx, pose
                    )
                    self._save_frames(idx, color, depth)
                    if ir is not None:
                        ir_saver.save(idx, ir_vis)
                    pose_count += 1
        camera_mgr.stop()
        cv2.destroyAllWindows()
        self.controller.shutdown()

    def _save_frames(self, idx: str, color: np.ndarray, depth: np.ndarray) -> None:
        rgb_path = os.path.join(self.captures_dir, f"{idx}_rgb.png")
        depth_path = os.path.join(self.captures_dir, f"{idx}_depth.npy")
        cv2.imwrite(rgb_path, color)
        np.save(depth_path, depth)
        self.logger.info(f"Saved frames for {idx}")


# --- Path Execution ---


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
    """Wrap a camera instance for reliable frame acquisition."""

    def __init__(self, camera: Camera | None = None, logger: Optional[Logger] = None):
        self.cam = camera or RealSenseCamera()
        self.logger = logger or Logger.get_logger("robot.workflow.camera")

    def start(self) -> bool:
        try:
            self.cam.start()
        except CameraError as e:
            self.logger.error(f"Camera start failed: {e}")
            return False
        for _ in Logger.progress(range(10), desc="Warmup"):
            try:
                self.cam.get_frames()
            except Exception:
                pass
            time.sleep(0.05)
        return True

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
        if not self.camera_mgr.start():
            self.logger.error("Camera not available. Aborting path run.")
            self.controller.shutdown()
            return
        for idx, pose in Logger.progress(list(enumerate(path)), desc="Path"):
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


def _add_record_args(parser: argparse.ArgumentParser) -> None:
    Config.load()
    parser.add_argument(
        "--ip",
        default=Config.get("robot.ip"),
        help="Robot IP address",
    )
    parser.add_argument(
        "--captures_dir",
        default=Config.get("path_saver.captures_dir", "cloud"),
        help="Directory for saved poses",
    )


def _run_record(args: argparse.Namespace) -> None:
    recorder = PoseRecorder(
        RobotController(rpc=args.ip), JsonPoseSaver(), args.captures_dir
    )
    recorder.run()


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    Config.load()
    parser.add_argument("--ip", default=Config.get("robot.ip"), help="Robot IP")
    parser.add_argument(
        "--path_file",
        default="captures/poses.json",
        help="JSON file with path poses",
    )
    parser.add_argument(
        "--out_dir",
        default="captures",
        help="Directory to save captured frames",
    )


def _run_path(args: argparse.Namespace) -> None:
    runner = PathRunner(
        controller=RobotController(rpc=args.ip),
        camera_mgr=CameraManager(),
        frame_saver=FrameSaver(args.out_dir),
        traj_file=args.path_file,
    )
    runner.run()


def _add_restart_args(parser: argparse.ArgumentParser) -> None:
    Config.load()
    parser.add_argument("--ip", default=Config.get("robot.ip"), help="Robot IP")
    parser.add_argument(
        "--delay", type=float, default=3.0, help="Seconds between reconnects"
    )
    parser.add_argument(
        "--attempts", type=int, default=3, help="Number of reconnect attempts"
    )


def _run_restart(args: argparse.Namespace) -> None:
    controller = RobotController(rpc=args.ip)
    ok = controller.restart(
        ip_address=args.ip, delay=args.delay, attempts=args.attempts
    )
    if ok:
        controller.logger.info("Robot restart completed successfully")
    else:
        controller.logger.error("Failed to restart robot")
    controller.shutdown()


def create_cli() -> CommandDispatcher:
    return CommandDispatcher(
        "Robot workflows",
        [
            Command("record", _run_record, _add_record_args, "Record robot poses"),
            Command("run", _run_path, _add_run_args, "Execute path and capture"),
            Command(
                "restart",
                _run_restart,
                _add_restart_args,
                "Restart robot connection",
            ),
        ],
    )


def main() -> None:
    logger = Logger.get_logger("robot.workflows")
    create_cli().run(logger=logger)


if __name__ == "__main__":
    main()
