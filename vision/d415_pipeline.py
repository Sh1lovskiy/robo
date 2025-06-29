"""D415 eye-in-hand calibration and point cloud utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Tuple

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs

from calibration.calibrator import load_board, HandEyeCalibrator, NPZHandEyeSaver
from calibration.helpers.pose_utils import (
    ExtractionParams,
    load_camera_params,
)
from calibration.helpers.validation_utils import euler_to_matrix
from robot.controller import RobotController
from utils.logger import Logger, LoggerType
from vision.cloud.generator import PointCloudGenerator
from vision.transform import TransformUtils


@dataclass
class D415StreamConfig:
    """Resolution and frame rate parameters."""

    depth_width: int = 1280
    depth_height: int = 720
    color_width: int = 1920
    color_height: int = 1080
    fps: int = 30
    align_to_color: bool = True


@dataclass
class D415CameraSettings:
    """Manual exposure and laser settings."""

    ir_exposure: int = 8000
    ir_gain: int = 16
    rgb_exposure: int = 200
    rgb_gain: int = 64
    projector_power: int = 0
    max_projector_power: int = 360


@dataclass
class D415FilterConfig:
    """RealSense post-processing options."""

    decimation: int = 2
    spatial_alpha: float = 0.5
    spatial_delta: int = 20
    temporal_alpha: float = 0.4
    temporal_delta: int = 20
    hole_filling: int = 1


class RealSenseD415:
    """Wrapper around ``pyrealsense2`` exposing tuned settings."""

    def __init__(
        self,
        stream_cfg: D415StreamConfig | None = None,
        settings: D415CameraSettings | None = None,
        filters: D415FilterConfig | None = None,
        logger: LoggerType | None = None,
    ) -> None:
        self.stream_cfg = stream_cfg or D415StreamConfig()
        self.settings = settings or D415CameraSettings()
        self.filters = filters or D415FilterConfig()
        self.logger = logger or Logger.get_logger("vision.d415")
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.started = False
        self.profile: rs.pipeline_profile | None = None
        self.align: rs.align | None = None
        self.depth_scale: float = 1.0
        self._init_config()

    # ---------------------------------------------------------------
    def _init_config(self) -> None:
        cfg = self.stream_cfg
        self.config.enable_stream(
            rs.stream.depth,
            cfg.depth_width,
            cfg.depth_height,
            rs.format.z16,
            cfg.fps,
        )
        self.config.enable_stream(
            rs.stream.color,
            cfg.color_width,
            cfg.color_height,
            rs.format.bgr8,
            cfg.fps,
        )

    # ---------------------------------------------------------------
    def start(self) -> None:
        self.profile = self.pipeline.start(self.config)
        device = self.profile.get_device()
        sensors = {s.get_info(rs.camera_info.name): s for s in device.sensors}
        self.depth_sensor = sensors.get("Stereo Module")
        self.rgb_sensor = sensors.get("RGB Camera")
        if self.depth_sensor is None or self.rgb_sensor is None:
            raise RuntimeError("Required sensors not found")
        self._apply_settings()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        self.logger.info(f"Depth scale: {self.depth_scale:.6f} m/unit")
        if self.stream_cfg.align_to_color:
            self.align = rs.align(rs.stream.color)
        self._log_device_info(device)
        self._setup_filters()
        self.started = True

    # ---------------------------------------------------------------
    def _apply_settings(self) -> None:
        s = self.settings
        self.depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
        self.depth_sensor.set_option(rs.option.exposure, float(s.ir_exposure))
        self.depth_sensor.set_option(rs.option.gain, float(s.ir_gain))
        power = float(s.projector_power)
        self.depth_sensor.set_option(rs.option.laser_power, power)
        self.rgb_sensor.set_option(rs.option.enable_auto_exposure, 0)
        self.rgb_sensor.set_option(rs.option.exposure, float(s.rgb_exposure))
        self.rgb_sensor.set_option(rs.option.gain, float(s.rgb_gain))

    # ---------------------------------------------------------------
    def _setup_filters(self) -> None:
        f = self.filters
        self.decimation = rs.decimation_filter()
        self.decimation.set_option(rs.option.filter_magnitude, f.decimation)
        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_smooth_alpha, f.spatial_alpha)
        self.spatial.set_option(rs.option.filter_smooth_delta, f.spatial_delta)
        self.temporal = rs.temporal_filter()
        self.temporal.set_option(rs.option.filter_smooth_alpha, f.temporal_alpha)
        self.temporal.set_option(rs.option.filter_smooth_delta, f.temporal_delta)
        self.holes = rs.hole_filling_filter(f.hole_filling)

    # ---------------------------------------------------------------
    def _log_device_info(self, device: rs.device) -> None:
        name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)
        self.logger.info(f"Device: {name} SN:{serial}")
        depth_stream = self.profile.get_stream(rs.stream.depth)
        depth_stream = depth_stream.as_video_stream_profile()
        color_stream = self.profile.get_stream(rs.stream.color)
        color_stream = color_stream.as_video_stream_profile()
        extr = depth_stream.get_extrinsics_to(color_stream)
        R = np.array(extr.rotation).reshape(3, 3)
        t = np.array(extr.translation)
        self.logger.debug(f"Extrinsics depth→color R={R.tolist()} t={t.tolist()}")

    # ---------------------------------------------------------------
    def stop(self) -> None:
        if self.started:
            self.pipeline.stop()
            self.started = False

    # ---------------------------------------------------------------
    def set_projector(self, enable: bool) -> None:
        power = self.settings.max_projector_power if enable else 0
        self.depth_sensor.set_option(rs.option.laser_power, float(power))

    # ---------------------------------------------------------------
    def _process_depth(self, frame: rs.frame) -> rs.frame:
        frame = self.decimation.process(frame)
        frame = self.spatial.process(frame)
        frame = self.temporal.process(frame)
        frame = self.holes.process(frame)
        return frame

    # ---------------------------------------------------------------
    def get_frames(
        self, aligned: bool = True
    ) -> Tuple[np.ndarray | None, np.ndarray | None]:
        assert self.started, "Camera not started"
        frames = self.pipeline.wait_for_frames()
        if aligned and self.align:
            frames = self.align.process(frames)
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if depth:
            depth = self._process_depth(depth)
        color_img = np.asanyarray(color.get_data()) if color else None
        depth_img = np.asanyarray(depth.get_data()) if depth else None
        return color_img, depth_img


@dataclass
class EyeInHandCalibrator:
    """Collect samples and compute hand-eye transform."""

    camera: RealSenseD415
    robot: RobotController
    board_cfg: Mapping[str, float | str]
    charuco_xml: str
    use_rgb: bool = True
    logger: LoggerType = field(
        default_factory=lambda: Logger.get_logger("vision.d415.calib")
    )

    def __post_init__(self) -> None:
        self.board, self.dictionary = load_board(self.board_cfg)
        self.K, self.dist = load_camera_params(self.charuco_xml)
        self.calibrator = HandEyeCalibrator(self.logger)
        self.params = ExtractionParams(
            min_corners=6,
            visualize=False,
            analyze_corners=False,
        )

    # -----------------------------------------------------------
    def _pose_from_robot(self) -> Tuple[np.ndarray, np.ndarray] | None:
        pose = self.robot.get_tcp_pose()
        if pose is None:
            return None
        R = euler_to_matrix(pose[3], pose[4], pose[5], degrees=True)
        t = np.array(pose[:3], dtype=float) / 1000.0
        return R, t

    # -----------------------------------------------------------
    def _detect_charuco(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray] | None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary)
        if ids is None or len(ids) < self.params.min_corners:
            return None
        for c in corners:
            cv2.cornerSubPix(
                gray,
                c,
                (3, 3),
                (-1, -1),
                (
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001,
                ),
            )
        _, char_corners, char_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, self.board
        )
        if char_corners is None or char_ids is None:
            return None
        if len(char_ids) < self.params.min_corners:
            return None
        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            char_corners,
            char_ids,
            self.board,
            self.K,
            self.dist,
        )
        if not ok:
            return None
        R, _ = cv2.Rodrigues(rvec)
        err = self._reprojection_error(char_corners, char_ids, rvec, tvec)
        self.logger.info(f"Charuco corners: {len(char_ids)} reproj_err: {err:.3f} px")
        return R, tvec.flatten()

    # -----------------------------------------------------------
    def _reprojection_error(
        self,
        corners: np.ndarray,
        ids: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> float:
        obj_pts = self.board.chessboardCorners[ids.flatten()]
        img_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, self.K, self.dist)
        img_pts = img_pts.squeeze(1)
        err = np.linalg.norm(img_pts - corners.squeeze(1), axis=1).mean()
        return float(err)

    # -----------------------------------------------------------
    def collect(self, num_samples: int = 10) -> None:
        self.camera.set_projector(False)
        while len(self.calibrator.R_gripper2base) < num_samples:
            color, depth = self.camera.get_frames()
            frame = color if self.use_rgb else cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
            pose_c = self._detect_charuco(frame)
            pose_r = self._pose_from_robot()
            if pose_c is None or pose_r is None:
                continue
            Rg, tg = pose_r
            Rc, tc = pose_c
            self.calibrator.add_sample(Rg, tg, Rc, tc)
            self.logger.info(f"Sample {len(self.calibrator.R_gripper2base)} captured")
        if len(self.calibrator.R_gripper2base) < num_samples:
            self.logger.warning("Not enough samples collected")

    # -----------------------------------------------------------
    def compute(
        self, method: str = "DANIILIDIS", out_file: str | None = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        R, t = self.calibrator.calibrate(method)
        if out_file:
            saver = NPZHandEyeSaver()
            Path(out_file).parent.mkdir(parents=True, exist_ok=True)
            self.calibrator.save(saver, out_file, R, t)
        return R, t


@dataclass
class PointCloudBuilder:
    """Acquire and process point clouds in the robot base frame."""

    camera: RealSenseD415
    handeye: Tuple[np.ndarray, np.ndarray]
    logger: LoggerType = field(
        default_factory=lambda: Logger.get_logger("vision.d415.cloud")
    )
    transformer: TransformUtils = field(default_factory=TransformUtils)
    generator: PointCloudGenerator = field(default_factory=PointCloudGenerator)

    # -----------------------------------------------------------
    def capture(self, robot_pose: Iterable[float]) -> o3d.geometry.PointCloud:
        color, depth = self.camera.get_frames()
        stream = self.camera.profile.get_stream(rs.stream.depth)
        intr = stream.as_video_stream_profile().get_intrinsics()
        K = {
            "fx": intr.fx,
            "fy": intr.fy,
            "ppx": intr.ppx,
            "ppy": intr.ppy,
        }
        depth_m = depth.astype(np.float32) * self.camera.depth_scale
        points, colors = self.generator.depth_to_cloud(depth_m, K, color)
        Rb, tb = self._robot_pose_to_rt(robot_pose)
        Rc, tc = self.handeye
        T = self.transformer.get_base_to_camera((Rb, tb), None, Rc, tc)
        pts_base = self.transformer.transform_points(points, T)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_base)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd = pcd.voxel_down_sample(voxel_size=0.002)
        pcd, _ = pcd.remove_statistical_outlier(20, 1.0)
        self.logger.info(
            f"Cloud: {len(pcd.points)} points, mean Z {np.mean(points[:,2]):.3f} m"
        )
        return pcd

    # -----------------------------------------------------------
    def _robot_pose_to_rt(self, pose: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
        arr = list(pose)
        R = euler_to_matrix(arr[3], arr[4], arr[5], degrees=True)
        t = np.array(arr[:3], dtype=float) / 1000.0
        return R, t

    # -----------------------------------------------------------
    def save(self, pcd: o3d.geometry.PointCloud, path: str) -> None:
        o3d.io.write_point_cloud(path, pcd)
        self.logger.info(f"Saved cloud to {path}")

    # -----------------------------------------------------------
    def load(self, path: str) -> o3d.geometry.PointCloud:
        pcd = o3d.io.read_point_cloud(path)
        self.logger.info(f"Loaded cloud from {path}")
        return pcd

    # -----------------------------------------------------------
    def remove_plane(
        self, pcd: o3d.geometry.PointCloud, dist: float = 0.003
    ) -> o3d.geometry.PointCloud:
        plane_model, inliers = pcd.segment_plane(dist, 3, 1000)
        self.logger.debug(
            f"Plane: {np.round(plane_model,4).tolist()} with {len(inliers)} inliers"
        )
        return pcd.select_by_index(inliers, invert=True)
