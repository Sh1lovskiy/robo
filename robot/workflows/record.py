"""Pose recording helpers and camera manager."""

from __future__ import annotations

import os
import re
import json
import time
from dataclasses import dataclass, field
from typing import Callable, List

import cv2
import numpy as np

from calibration.charuco import load_camera_params
from robot.controller import RobotController
from utils.error_tracker import CameraError
from utils.keyboard import GlobalKeyListener
from utils.logger import Logger, LoggerType
from utils.settings import handeye
from vision.opencv_utils import OpenCVUtils
from vision.camera import (
    CameraBase,
    RealSenseD415,
    D415CameraSettings,
    D415FilterConfig,
)


class PoseSaver:
    """Abstract pose persistence interface."""

    def save(self, filename: str, pose_id: str, pose: List[float]) -> None:
        raise NotImplementedError


class JsonPoseSaver(PoseSaver):
    def save(self, filename: str, pose_id: str, pose: List[float]) -> None:
        """Append pose data to ``filename`` in JSON format."""
        data: dict = {}
        if os.path.exists(filename):
            with open(filename, "r") as f:
                data = json.load(f)
        data[pose_id] = {"tcp_coords": pose}
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)


@dataclass
class FrameSaver:
    out_dir: str
    logger: LoggerType = Logger.get_logger("robot.workflow.frames")

    def save(self, idx: int, color: np.ndarray, depth: np.ndarray) -> None:
        """Persist color and depth images to ``out_dir``."""
        os.makedirs(self.out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(self.out_dir, f"{idx:03d}_rgb.png"), color)
        np.save(os.path.join(self.out_dir, f"{idx:03d}_depth.npy"), depth)
        self.logger.info(f"Saved frame {idx}")


class CameraManager:
    """Safe wrapper around camera start/stop and frame retrieval."""

    def __init__(
        self,
        camera: CameraBase | None = None,
        *,
        logger: LoggerType | None = None,
        camera_class: type[CameraBase] | None = None,
        camera_kwargs: dict | None = None,
    ) -> None:
        """Create and optionally configure the underlying camera object."""
        if camera is not None:
            self.cam = camera
        else:
            camera_class = camera_class or RealSenseD415
            camera_kwargs = camera_kwargs or dict(
                settings=D415CameraSettings(rgb_exposure=50),
                filters=D415FilterConfig(
                    decimation=2,
                    spatial_alpha=0.5,
                    spatial_delta=20,
                    temporal_alpha=0.4,
                    temporal_delta=20,
                    hole_filling=1,
                ),
            )
            self.cam = camera_class(**camera_kwargs)
        self.logger = logger or Logger.get_logger("robot.workflow.camera")

    def start(self) -> bool:
        """Start the camera and warm up the stream."""
        try:
            self.cam.start()
        except CameraError as exc:
            self.logger.error(f"Camera start failed: {exc}")
            return False
        for _ in Logger.progress(range(10), desc="Warmup"):
            try:
                self.cam.get_frames()
            except Exception:
                pass
            time.sleep(0.05)
        return True

    def stop(self) -> None:
        """Stop streaming."""
        self.cam.stop()

    def get_frames(self) -> tuple[np.ndarray, np.ndarray]:
        """Return a color/depth frame pair, retrying until available."""
        for attempt in range(10):
            color, depth = self.cam.get_frames()
            if color is not None and depth is not None:
                return color, depth
            self.logger.debug(f"Frame not ready (attempt {attempt})")
            time.sleep(0.2)
        raise RuntimeError("Failed to get frames")


@dataclass
class PoseRecorder:
    """Interactively record TCP poses with optional frame capture."""

    controller: RobotController
    saver: PoseSaver
    captures_dir: str
    frame_saver: FrameSaver | None = None
    progress_cb: Callable[[str, List[float]], None] | None = None
    logger: LoggerType = Logger.get_logger("robot.workflow.record")
    drag: bool = False
    depth_filter: any = field(default=None)

    def run(self) -> None:
        """Main interactive pose recording loop."""
        camera_mgr = CameraManager()
        if not self._enable_robot(camera_mgr):
            return

        board, camera_matrix, dist_coeffs = self._setup_calibration()
        pose_count, poses_path = self._init_pose_storage(camera_mgr)

        self._interactive_loop(
            camera_mgr, board, camera_matrix, dist_coeffs, poses_path, pose_count
        )
        camera_mgr.stop()
        if self.drag:
            self._exit_drag()

    def _enable_robot(self, camera_mgr: CameraManager) -> bool:
        """Enable robot and start the camera if available."""
        try:
            self.controller.enable()
        except Exception as e:  # pragma: no cover - hardware error
            self.logger.error(f"Failed to enable robot: {e}")
            return False
        if not camera_mgr.start():
            self.logger.error("Camera not available. Exiting pose recorder.")
            return False
        return True

    def _setup_calibration(
        self,
    ) -> tuple[cv2.aruco_CharucoBoard, np.ndarray | None, np.ndarray | None]:
        """Prepare Charuco board and load camera intrinsics if present."""
        char_cfg = handeye
        board = cv2.aruco.CharucoBoard(
            char_cfg.square_numbers,
            char_cfg.square_length,
            char_cfg.marker_length,
            cv2.aruco.getPredefinedDictionary(
                getattr(cv2.aruco, f"DICT_{char_cfg.aruco_dict}")
            ),
        )
        xml_path = os.path.join(char_cfg.calib_output_dir, char_cfg.xml_file)
        camera_matrix, dist_coeffs = (None, None)
        if os.path.isfile(xml_path):
            camera_matrix, dist_coeffs = load_camera_params(xml_path)
        return board, camera_matrix, dist_coeffs

    def _init_pose_storage(self, camera_mgr: CameraManager) -> tuple[int, str]:
        """Initialize pose storage and handle drag-teaching mode."""
        os.makedirs(self.captures_dir, exist_ok=True)
        if self.drag:
            try:
                self.controller.rpc.DragTeachSwitch(1)
            except Exception as e:  # pragma: no cover - hardware error
                self.logger.error(f"Failed to enter drag mode: {e}")
                camera_mgr.stop()
                raise
        poses_path = os.path.join(self.captures_dir, "poses.json")
        if os.path.exists(poses_path):
            with open(poses_path, "r") as f:
                poses_data = json.load(f)
            existing = [int(k) for k in poses_data if re.fullmatch(r"\d+", k)]
            pose_count = max(existing) + 1 if existing else 0
        else:
            pose_count = 0
        print("Press ENTER to save current pose. Press 'q' to quit.")
        return pose_count, poses_path

    def _interactive_loop(
        self,
        camera_mgr: CameraManager,
        board: cv2.aruco_CharucoBoard,
        camera_matrix: np.ndarray | None,
        dist_coeffs: np.ndarray | None,
        poses_path: str,
        pose_count: int,
    ) -> None:
        """Handle user input and frame capture until exit."""
        should_exit = False

        def on_save() -> None:
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

        def on_exit() -> None:
            nonlocal should_exit
            should_exit = True
            print("Exit requested by hotkey!")

        listener = GlobalKeyListener(
            {
                "<enter>": on_save,
                "<ctrl>+s": on_save,
                "q": on_exit,
                "<ctrl>+q": on_exit,
            }
        )
        listener.start()
        opencv_utils = OpenCVUtils(display_width=640, display_height=480)
        while not should_exit:
            color, depth = camera_mgr.get_frames()
            self._handle_frame(
                color, depth, board, camera_matrix, dist_coeffs, opencv_utils
            )
            if cv2.waitKey(50) == 27:
                break
        listener.stop()
        cv2.destroyAllWindows()

    def _handle_frame(
        self,
        color: np.ndarray | None,
        depth: np.ndarray | None,
        board: cv2.aruco_CharucoBoard,
        camera_matrix: np.ndarray | None,
        dist_coeffs: np.ndarray | None,
        opencv_utils: OpenCVUtils,
    ) -> None:
        """Display RGB and depth images with optional corner overlays."""
        if depth is not None:
            opencv_utils.show_depth(depth)
        if color is None:
            return
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, board.getDictionary())
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
                        color, f"Charuco t: {tx:.3f}, {ty:.3f}, {tz:.3f}", (10, 30)
                    )
        color_disp = cv2.resize(
            color,
            (opencv_utils.display_width, opencv_utils.display_height),
            interpolation=cv2.INTER_AREA,
        )
        cv2.imshow("RGB", color_disp)

    def _exit_drag(self) -> None:
        """Disable drag mode on the robot if it was enabled."""
        try:
            self.controller.rpc.DragTeachSwitch(0)
        except Exception as e:  # pragma: no cover - hardware error
            self.logger.error(f"Failed to exit drag mode: {e}")

    def _save_frames(self, idx: str, color: np.ndarray, depth: np.ndarray) -> None:
        """Store captured frames on disk and via :class:`FrameSaver`."""
        filtered = self.depth_filter.filter(depth) if self.depth_filter else depth
        if self.frame_saver:
            self.frame_saver.save(int(idx), color, filtered)
        rgb_path = os.path.join(self.captures_dir, f"{idx}_rgb.png")
        depth_path = os.path.join(self.captures_dir, f"{idx}_depth.npy")
        os.makedirs(self.captures_dir, exist_ok=True)
        cv2.imwrite(rgb_path, color)
        np.save(depth_path, filtered)
        self.logger.info(f"Saved frames for {idx}")
