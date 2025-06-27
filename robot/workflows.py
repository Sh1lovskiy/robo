# robot/workflows.py
"""Robot-related workflows: pose recording and path execution.

TODO: add CI badges for build and coverage.
"""

from __future__ import annotations

import os
import json
import re
import time
from dataclasses import dataclass
from typing import List
import argparse

import cv2
import numpy as np

from calibration.helpers.pose_utils import load_camera_params
from robot.controller import RobotController
from utils.cli import Command, CommandDispatcher
from utils.config import Config
from utils.error_tracker import CameraError
from utils.keyboard import GlobalKeyListener
from utils.logger import Logger, LoggerType
from vision.realsense import RealSenseCamera, RealSenseConfig
from vision.camera_base import Camera
from vision.opencv_utils import OpenCVUtils


# --- Pose Recording ---


class PoseSaver:
    """Strategy interface for saving poses."""

    def save(self, filename: str, pose_id: str, pose: List[float]) -> None:
        """Persist ``pose`` identified by ``pose_id`` into ``filename``."""
        raise NotImplementedError


class JsonPoseSaver(PoseSaver):
    """Save poses to a JSON file."""

    def save(self, filename: str, pose_id: str, pose: List[float]) -> None:
        """Append ``pose`` to ``filename`` creating the JSON if needed."""
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
    logger: LoggerType = Logger.get_logger("robot.workflow.ir")

    def save(self, idx: str, ir_img: np.ndarray) -> None:
        """Write an infrared PNG for frame ``idx``."""
        os.makedirs(self.out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(self.out_dir, f"{idx}_ir.png"), ir_img)
        self.logger.info(f"Saved IR for {idx}")


@dataclass
class PoseRecorder:
    """Interactive pose recorder with synchronized camera frames."""

    controller: RobotController
    saver: PoseSaver
    captures_dir: str
    drag: bool = False
    logger: LoggerType = Logger.get_logger("robot.workflow.record")

    def run(self) -> None:
        """Interactively save robot poses with synchronized camera frames."""
        camera_mgr = CameraManager()
        try:
            self.controller.enable()
        except Exception as e:
            self.logger.error(f"Failed to enable robot: {e}")
            return

        if not camera_mgr.start():
            self.logger.error("Camera not available. Exiting pose recorder.")
            return
        Config.load()
        char_cfg = Config.get("charuco")
        squares_x = char_cfg.get("squares_x", 5)
        squares_y = char_cfg.get("squares_y", 7)
        square_length = char_cfg.get("square_length", 0.035)
        marker_length = char_cfg.get("marker_length", 0.026)
        dict_name = char_cfg.get("aruco_dict", "4X4_100")
        dict_attr = f"DICT_{dict_name}"
        aruco_id = getattr(cv2.aruco, dict_attr, cv2.aruco.DICT_4X4_100)
        dictionary = cv2.aruco.getPredefinedDictionary(aruco_id)
        board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y), square_length, marker_length, dictionary
        )

        xml_path = os.path.join(
            char_cfg.get("calib_output_dir", "calibration/results"),
            char_cfg.get("xml_file", "charuco_cam.xml"),
        )
        camera_matrix = None
        dist_coeffs = None
        if os.path.isfile(xml_path):
            camera_matrix, dist_coeffs = load_camera_params(xml_path)

        os.makedirs(self.captures_dir, exist_ok=True)
        if self.drag:
            self.controller.enable()
            try:
                self.controller.rpc.DragTeachSwitch(1)
            except Exception as e:
                self.logger.error(f"Failed to enter drag mode: {e}")
                camera_mgr.stop()
                return
        ir_saver = IRFrameSaver(self.captures_dir)
        poses_path = os.path.join(self.captures_dir, "poses.json")
        if os.path.exists(poses_path):
            with open(poses_path, "r") as f:
                poses_data = json.load(f)
            existing_ids = [
                int(k) for k in poses_data.keys() if re.fullmatch(r"\d+", k)
            ]
            pose_count = max(existing_ids) + 1 if existing_ids else 0
        else:
            pose_count = 0
        print("Press ENTER to save current pose. Press 'q' to quit.")

        should_exit = [False]

        def on_save():
            nonlocal pose_count
            color, depth = camera_mgr.get_frames()
            pose = self.controller.get_tcp_pose()
            if pose:
                idx = f"{pose_count:03d}"
                self.saver.save(poses_path, idx, pose)
                self._save_frames(idx, color, depth)
                print(f"Saved pose {idx}")
                pose_count += 1

        def on_exit():
            should_exit[0] = True
            print("Exit requested by hotkey!")

        hotkeys = {
            "<enter>": on_save,
            "<ctrl>+s": on_save,
            "q": on_exit,
            "<ctrl>+q": on_exit,
        }

        listener = GlobalKeyListener(hotkeys)
        listener.start()

        while not should_exit[0]:
            color, depth = camera_mgr.get_frames()
            if depth is not None:
                OpenCVUtils.show_depth(depth)
            if color is not None:
                gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
                if ids is not None and len(ids) > 0:
                    cv2.aruco.drawDetectedMarkers(color, corners, ids)
                    ret, char_corners, char_ids = cv2.aruco.interpolateCornersCharuco(
                        corners, ids, gray, board
                    )
                    if (
                        camera_matrix is not None
                        and dist_coeffs is not None
                        and char_ids is not None
                        and len(char_ids) >= 4
                    ):
                        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                            char_corners,
                            char_ids,
                            board,
                            camera_matrix,
                            dist_coeffs,
                            np.zeros((3, 1), dtype=np.float64),
                            np.zeros((3, 1), dtype=np.float64),
                        )
                        if ok:
                            cv2.drawFrameAxes(
                                color, camera_matrix, dist_coeffs, rvec, tvec, 0.05
                            )
                            tx, ty, tz = tvec.flatten()
                            OpenCVUtils.draw_text(
                                color,
                                f"Charuco t: {tx:.3f}, {ty:.3f}, {tz:.3f}",
                                (10, 30),
                            )
                cv2.imshow("RGB", color)
            if cv2.waitKey(50) == 27:
                break
        listener.stop()
        camera_mgr.stop()
        cv2.destroyAllWindows()
        if self.drag:
            try:
                self.controller.rpc.DragTeachSwitch(0)
            except Exception as e:
                self.logger.error(f"Failed to exit drag mode: {e}")

    def _save_frames(self, idx: str, color: np.ndarray, depth: np.ndarray) -> None:
        """Store RGB and depth frames for pose ``idx`` on disk."""
        rgb_path = os.path.join(self.captures_dir, f"{idx}_rgb.png")
        depth_path = os.path.join(self.captures_dir, f"{idx}_depth.npy")
        os.makedirs(self.captures_dir, exist_ok=True)
        cv2.imwrite(rgb_path, color)
        np.save(depth_path, depth)
        self.logger.info(f"Saved frames for {idx}")


# --- Path Execution ---


@dataclass
class FrameSaver:
    """Save RGB and depth frames to disk."""

    out_dir: str
    logger: LoggerType = Logger.get_logger("robot.workflow.frames")

    def save(self, idx: int, color: np.ndarray, depth: np.ndarray) -> None:
        """Write color/depth pair for frame ``idx``."""
        os.makedirs(self.out_dir, exist_ok=True)
        rgb_path = os.path.join(self.out_dir, f"{idx:03d}_rgb.png")
        depth_path = os.path.join(self.out_dir, f"{idx:03d}_depth.npy")
        cv2.imwrite(rgb_path, color)
        np.save(depth_path, depth)
        self.logger.info(f"Saved {rgb_path} and {depth_path}")


class CameraManager:
    """Wrap a camera instance for reliable frame acquisition."""

    def __init__(self, camera: Camera | None = None, logger: LoggerType | None = None):
        """Initialize with an optional :class:`Camera` implementation."""

        self.cam = camera or RealSenseCamera(RealSenseConfig())
        self.logger = logger or Logger.get_logger("robot.workflow.camera")

    def start(self) -> bool:
        """Start the underlying camera and perform a short warmup."""
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
        """Stop the underlying camera stream."""

        self.cam.stop()

    def get_frames(self) -> tuple[np.ndarray, np.ndarray]:
        """Retry grabbing frames until both color and depth are valid."""
        for attempt in range(10):
            color, depth = self.cam.get_frames()
            if color is not None and depth is not None:
                return color, depth
            self.logger.debug(f"Frame not ready (attempt {attempt})")
            time.sleep(0.2)
        raise RuntimeError("Failed to get frames")


def load_trajectory(path_file: str) -> List[List[float]]:
    """Load a list of TCP poses from a JSON path file."""
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
    logger: LoggerType = Logger.get_logger("robot.workflow.path")

    def run(self) -> None:
        """Execute the trajectory while recording frames."""
        path = load_trajectory(self.traj_file)
        try:
            self.controller.enable()
        except Exception as e:
            self.logger.error(f"Failed to enable robot: {e}")
            return
        if not self.camera_mgr.start():
            self.logger.error("Camera not available. Aborting path run.")
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
        self.logger.info("Path execution finished")


def _add_record_args(parser: argparse.ArgumentParser) -> None:
    """Arguments for the ``record`` sub-command."""
    Config.load()
    parser.add_argument(
        "--ip",
        default=Config.get("robot.ip"),
        help="Robot IP address",
    )
    parser.add_argument(
        "--captures_dir",
        default=Config.get("path_saver.captures_dir", "captures"),
        help="Directory for saved poses",
    )
    parser.add_argument(
        "--drag",
        action="store_true",
        help="Enable drag teaching mode while recording",
    )


def _run_record(args: argparse.Namespace) -> None:
    """CLI hook to record robot poses.

    Args:
        args: Parsed command line namespace with ``ip``, ``captures_dir`` and
            ``drag`` options supplied by :func:`_add_record_args`.
    """
    recorder = PoseRecorder(
        RobotController(rpc=args.ip), JsonPoseSaver(), args.captures_dir, drag=args.drag
    )
    recorder.run()


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    """Arguments for the ``run`` sub-command."""
    Config.load()
    parser.add_argument("--ip", default=Config.get("robot.ip"), help="Robot IP")
    parser.add_argument(
        "--path_file",
        default="/home/sha/Documents/work/robo/clouds/cloud_new/poses.json",
        help="JSON file with path poses",
    )
    parser.add_argument(
        "--out_dir",
        default="captures",
        help="Directory to save captured frames",
    )


def _run_path(args: argparse.Namespace) -> None:
    """Execute a recorded path while capturing frames.

    Args:
        args: Namespace with ``ip``, ``path_file`` and ``out_dir`` attributes
            provided by :func:`_add_run_args`.
    """
    runner = PathRunner(
        controller=RobotController(rpc=args.ip),
        camera_mgr=CameraManager(),
        frame_saver=FrameSaver(args.out_dir),
        traj_file=args.path_file,
    )
    runner.run()


def _add_restart_args(parser: argparse.ArgumentParser) -> None:
    """Arguments for ``restart`` command."""
    Config.load()
    parser.add_argument("--ip", default=Config.get("robot.ip"), help="Robot IP")
    parser.add_argument(
        "--delay", type=float, default=3.0, help="Seconds between reconnects"
    )
    parser.add_argument(
        "--attempts", type=int, default=3, help="Number of reconnect attempts"
    )


def _run_restart(args: argparse.Namespace) -> None:
    """Restart robot connection via CLI."""
    controller = RobotController(rpc=args.ip)
    ok = controller.restart(
        ip_address=args.ip, delay=args.delay, attempts=args.attempts
    )
    if ok:
        controller.logger.info("Robot restart completed successfully")
    else:
        controller.logger.error("Failed to restart robot")


def create_cli() -> CommandDispatcher:
    """Assemble the command dispatcher for robot workflows."""

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
    """Entry point for ``robot.workflows`` when used as a module."""

    logger = Logger.get_logger("robot.workflows")
    create_cli().run(logger=logger)


if __name__ == "__main__":
    main()
