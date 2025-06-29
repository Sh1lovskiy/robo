# robot/workflows.py
"""Robot-related workflows: pose recording and path execution.

TODO: add CI badges for build and coverage.
"""

from __future__ import annotations

import os
import json
import re
import time
import asyncio
from dataclasses import dataclass
from typing import List
from typing import List, Callable
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
from utils.lmdb_storage import IStorage, LmdbStorage

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


class DBPoseSaver(PoseSaver):
    """Store poses in a :class:`LmdbStorage` backend."""

    def __init__(self, storage: LmdbStorage, prefix: str = "poses") -> None:
        self.storage = storage
        self.prefix = prefix

    def save(self, filename: str, pose_id: str, pose: List[float]) -> None:
        key = f"{self.prefix}:{pose_id}"
        self.storage.put_json(key, {"tcp_coords": pose})


@dataclass
class IRFrameSaver:
    """Save infrared frames to disk."""

    out_dir: str
    logger: LoggerType = Logger.get_logger("robot.workflow.ir")

    def save(self, idx: str, ir_img: np.ndarray) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(self.out_dir, f"{idx}_ir.png"), ir_img)
        self.logger.info(f"Saved IR for {idx}")


class DBIRFrameSaver(IRFrameSaver):
    """Store IR frames in an :class:`LmdbStorage` backend."""

    def __init__(self, storage: LmdbStorage, prefix: str = "ir") -> None:
        self.storage = storage
        self.prefix = prefix
        self.logger = Logger.get_logger("robot.workflow.dbir")

    def save(self, idx: str, ir_img: np.ndarray) -> None:
        key = f"{self.prefix}:{idx}"
        self.storage.put_image(key, ir_img)
        self.logger.info(f"Saved IR for {idx}")


from vision.depth_filtering import RealSenseDepthFilter


@dataclass
class PoseRecorder:
    """Interactive pose recorder with synchronized camera frames."""

    controller: RobotController
    saver: PoseSaver
    captures_dir: str
    drag: bool = False
    progress_cb: Callable[[str, List[float]], None] | None = None
    logger: LoggerType = Logger.get_logger("robot.workflow.record")
    depth_filter = RealSenseDepthFilter(
        decimation=2,
        spatial_alpha=0.5,
        spatial_delta=20,
        temporal_alpha=0.4,
        temporal_delta=20,
        hole_filling=1,
        use_disparity=False,
    )

    def run(self) -> None:
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
        squares_y = char_cfg.get("squares_y", 8)
        square_length = char_cfg.get("square_length", 0.035)
        marker_length = char_cfg.get("marker_length", 0.026)
        dict_name = char_cfg.get("aruco_dict", "5X5_100")
        dict_attr = f"DICT_{dict_name}"
        aruco_id = getattr(cv2.aruco, dict_attr, cv2.aruco.DICT_5X5_100)
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
                if self.progress_cb:
                    self.progress_cb(idx, pose)

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
        opencv_utils = OpenCVUtils(display_width=640, display_height=480)

        while not should_exit[0]:
            color, depth = camera_mgr.get_frames()
            if depth is not None:
                opencv_utils.show_depth(depth)
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
                color_disp = cv2.resize(
                    color,
                    (opencv_utils.display_width, opencv_utils.display_height),
                    interpolation=cv2.INTER_AREA,
                )
                cv2.imshow("RGB", color_disp)
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
        filtered_depth = self.depth_filter.filter(depth)
        rgb_path = os.path.join(self.captures_dir, f"{idx}_rgb.png")
        depth_path = os.path.join(self.captures_dir, f"{idx}_depth.npy")
        os.makedirs(self.captures_dir, exist_ok=True)
        cv2.imwrite(rgb_path, color)
        np.save(depth_path, filtered_depth)
        self.logger.info(f"Saved FILTERED frames for {idx}")

    async def run_async(self) -> None:
        """Async version of :meth:`run`."""
        camera_mgr = CameraManager()
        try:
            await asyncio.to_thread(self.controller.enable)
        except Exception as e:
            self.logger.error(f"Failed to enable robot: {e}")
            return

        if not await asyncio.to_thread(camera_mgr.start):
            self.logger.error("Camera not available. Exiting pose recorder.")
            return
        if self.drag:
            await asyncio.to_thread(self.controller.enable)
            try:
                await asyncio.to_thread(self.controller.rpc.DragTeachSwitch, 1)
            except Exception as e:
                self.logger.error(f"Failed to enter drag mode: {e}")
                await asyncio.to_thread(camera_mgr.stop)
                return
        pose_count = 0
        print("Press ENTER to save current pose. Press 'q' to quit.")

        should_exit = False

        def on_save() -> None:
            nonlocal pose_count
            color, depth = camera_mgr.get_frames()
            pose = self.controller.get_tcp_pose()
            if pose:
                idx = f"{pose_count:03d}"
                if isinstance(self.saver, DBPoseSaver):
                    self.saver.save("", idx, pose)
                else:
                    path = os.path.join(self.captures_dir, "poses.json")
                    self.saver.save(path, idx, pose)
                self._save_frames(idx, color, depth)
                pose_count += 1
                if self.progress_cb:
                    self.progress_cb(idx, pose)

        def on_exit() -> None:
            nonlocal should_exit
            should_exit = True

        listener = GlobalKeyListener({"<enter>": on_save, "q": on_exit})
        listener.start()
        while not should_exit:
            await asyncio.sleep(0.1)
        listener.stop()
        await asyncio.to_thread(camera_mgr.stop)
        if self.drag:
            try:
                await asyncio.to_thread(self.controller.rpc.DragTeachSwitch, 0)
            except Exception as e:
                self.logger.error(f"Failed to exit drag mode: {e}")


# --- Path Execution ---


@dataclass
class FrameSaver:
    """Save RGB and depth frames to disk."""

    out_dir: str
    logger: LoggerType = Logger.get_logger("robot.workflow.frames")

    def save(self, idx: int, color: np.ndarray, depth: np.ndarray) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        rgb_path = os.path.join(self.out_dir, f"{idx:03d}_rgb.png")
        depth_path = os.path.join(self.out_dir, f"{idx:03d}_depth.npy")
        cv2.imwrite(rgb_path, color)
        np.save(depth_path, depth)
        self.logger.info(f"Saved {rgb_path} and {depth_path}")


class DBFrameSaver(FrameSaver):
    """Store RGB and depth frames in :class:`LmdbStorage`."""

    def __init__(self, storage: LmdbStorage, prefix: str = "frame") -> None:
        self.storage = storage
        self.prefix = prefix
        self.logger = Logger.get_logger("robot.workflow.dbframes")

    def save(self, idx: int, color: np.ndarray, depth: np.ndarray) -> None:
        key = f"{self.prefix}:{idx:03d}"
        with self.storage.batch() as b:
            b.put_image(f"{key}:rgb", color)
            b.put_array(f"{key}:depth", depth)
        self.logger.info(f"Stored frame {idx}")


class CameraManager:
    """Wrap a camera instance for reliable frame acquisition."""

    def __init__(self, camera: Camera | None = None, logger: LoggerType | None = None):
        self.cam = camera or RealSenseCamera(RealSenseConfig())
        self.logger = logger or Logger.get_logger("robot.workflow.camera")

    def start(self) -> bool:
        try:
            self.cam.start()
        except CameraError as e:
            self.logger.error(f"Camera start failed: {e}")
            return False
        # --- Extrinsics inspection block ---
        try:
            import pyrealsense2 as rs

            rs_device = getattr(self.cam, "device", None)
            if rs_device is None and hasattr(self.cam, "pipeline"):
                pipeline_profile = self.cam.pipeline.get_active_profile()
                rs_device = pipeline_profile.get_device()
            if rs_device is not None:
                sensors = rs_device.query_sensors()
                sensor_names = [s.get_info(rs.camera_info.name) for s in sensors]
                extrinsics_report = []
                for i, s_from in enumerate(sensors):
                    for j, s_to in enumerate(sensors):
                        if i == j:
                            continue
                        try:
                            stream_from = s_from.get_stream_profiles()[0]
                            stream_to = s_to.get_stream_profiles()[0]
                            extr = stream_from.get_extrinsics_to(stream_to)
                            txt = (
                                f"Extrinsics FROM [{sensor_names[i]}] TO [{sensor_names[j]}]:\n"
                                f"  Rotation: {np.array(extr.rotation).reshape(3,3)}\n"
                                f"  Translation: {np.array(extr.translation)}\n"
                            )
                            print(txt)
                            extrinsics_report.append(txt)
                        except Exception as e:
                            self.logger.warning(
                                f"Extrinsics {sensor_names[i]} -> {sensor_names[j]}: {e}"
                            )
                out_dir = getattr(self, "captures_dir", "captures")
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, "realsense_extrinsics.txt"), "w") as f:
                    f.write("\n".join(extrinsics_report))
                self.logger.info(
                    "RealSense extrinsics saved to realsense_extrinsics.txt"
                )
            else:
                self.logger.warning("No RealSense device found for extrinsics export.")
        except Exception as e:
            self.logger.warning(f"Failed to dump RealSense extrinsics: {e}")
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


def load_trajectory(db_path: str, prefix: str = "poses") -> List[List[float]]:
    """Load a list of TCP poses from an LMDB database."""
    store = LmdbStorage(db_path, readonly=True)
    keys = sorted(store.iter_keys(f"{prefix}:"), key=lambda k: int(k.split(":")[1]))
    return [store.get_json(k)["tcp_coords"] for k in keys]


def load_trajectory_db(storage: IStorage, prefix: str = "poses") -> List[List[float]]:
    """Retrieve trajectory poses from an :class:`IStorage` backend."""
    keys = sorted(storage.iter_prefix(f"{prefix}:"), key=lambda k: int(k.split(":")[1]))
    poses: List[List[float]] = []
    for k in keys:
        data = storage.get_json(k)
        if data is not None:
            poses.append(data["tcp_coords"])
    return poses


@dataclass
class PathRunner:
    """Run robot path and capture frames."""

    controller: RobotController
    camera_mgr: CameraManager
    frame_saver: FrameSaver
    traj_file: str | None = None
    storage: IStorage | None = None
    progress_cb: Callable[[int, List[float]], None] | None = None

    def run(self) -> None:
        """Execute the trajectory while recording frames."""
        if self.storage is not None:
            path = load_trajectory_db(self.storage)
        elif self.traj_file is not None:
            path = load_trajectory(self.traj_file)
        else:
            self.logger.error("No trajectory source provided")
            return
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
            if self.progress_cb:
                self.progress_cb(idx, pose)
        self.camera_mgr.stop()
        self.logger.info("Path execution finished")

    async def run_async(self) -> None:
        """Async variant of :meth:`run`."""
        if self.storage is not None:
            keys = await self.storage.async_iter_prefix("poses:")
            keys = sorted(keys, key=lambda k: int(k.split(":")[1]))
            path = []
            for k in keys:
                data = await self.storage.async_get_json(k)
                if data:
                    path.append(data["tcp_coords"])
        elif self.traj_file is not None:
            path = load_trajectory(self.traj_file)
        else:
            self.logger.error("No trajectory source provided")
            return
        await asyncio.to_thread(self.controller.enable)
        if not await asyncio.to_thread(self.camera_mgr.start):
            self.logger.error("Camera not available. Aborting path run.")
            return
        for idx, pose in enumerate(path):
            self.logger.info(f"Moving to {pose}")
            ok = await asyncio.to_thread(self.controller.move_linear, pose)
            if not ok:
                self.logger.error(f"Movement failed at {idx}")
                break
            await asyncio.sleep(0.5)
            color, depth = await asyncio.to_thread(self.camera_mgr.get_frames)
            await asyncio.to_thread(self.frame_saver.save, idx, color, depth)
            if self.progress_cb:
                self.progress_cb(idx, pose)
        await asyncio.to_thread(self.camera_mgr.stop)
        self.logger.info("Path execution finished")


def _add_record_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for record mode."""
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
    parser.add_argument(
        "--db_path",
        default="robot_data.lmdb",
        help="LMDB database path",
    )


def _run_record(args: argparse.Namespace) -> None:
    """
    CLI hook to record robot poses using LMDB storage.
    args: Parsed command line options from :func:`_add_record_args`.
    """
    storage = LmdbStorage(args.db_path)
    recorder = PoseRecorder(
        RobotController(rpc=args.ip),
        JsonPoseSaver(),
        DBPoseSaver(storage),
        args.captures_dir,
        DBFrameSaver(storage),
        drag=args.drag,
    )
    recorder.run()


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for path run mode."""
    Config.load()
    parser.add_argument("--ip", default=Config.get("robot.ip"), help="Robot IP")
    parser.add_argument(
        "--db_path",
        default="robot_data.lmdb",
        help="LMDB database with poses",
    )


def _run_path(args: argparse.Namespace) -> None:
    """Run robot path execution workflow."""
    storage = LmdbStorage(args.db_path)
    runner = PathRunner(
        controller=RobotController(rpc=args.ip),
        camera_mgr=CameraManager(),
        frame_saver=DBFrameSaver(storage),
        storage=storage,
        traj_file=None if args.path_file is None else args.path_file,
    )
    runner.run()


def _add_restart_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for restart mode."""
    Config.load()
    parser.add_argument("--ip", default=Config.get("robot.ip"), help="Robot IP")
    parser.add_argument(
        "--delay", type=float, default=3.0, help="Seconds between reconnects"
    )
    parser.add_argument(
        "--attempts", type=int, default=3, help="Number of reconnect attempts"
    )


def _run_restart(args: argparse.Namespace) -> None:
    """Run robot restart workflow."""
    controller = RobotController(rpc=args.ip)
    ok = controller.restart(
        ip_address=args.ip, delay=args.delay, attempts=args.attempts
    )
    if ok:
        controller.logger.info("Robot restart completed successfully")
    else:
        controller.logger.error("Failed to restart robot")


def create_cli() -> CommandDispatcher:
    """Create command line interface dispatcher."""
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
    """Entry point for CLI."""
    logger = Logger.get_logger("robot.workflows")
    create_cli().run(logger=logger)


if __name__ == "__main__":
    main()
