"""Draw along the top edge of a part using a Fairino FR3 robot."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from robot.config import RobotConfig as RobotConfigBase

import numpy as np
import open3d as o3d  # type: ignore
from loguru import logger
from omegaconf import DictConfig, OmegaConf
import yaml  # type: ignore
from scipy.spatial import ConvexHull  # type: ignore
import sys
from typing import Any, TYPE_CHECKING, cast

try:
    import pyrealsense2 as rs  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    rs = None

import hydra

DEFAULT_YAML = """
camera:
  width: 640
  height: 480
  decimation: 2
  filter_sigma: 1.0
robot:
  ip: "192.168.58.2"
  velocity: 20.0
edge:
  hull_alpha: 0.03
  sample_resolution: 0.0001
  speed: 0.03
  transform_path: "handeye.yaml"
  force_limit: 2.0
logging:
  level: INFO
"""

CONF_DIR = Path(__file__).with_name("conf")
CONF_PATH = CONF_DIR / "default.yaml"
if not CONF_PATH.exists():
    CONF_DIR.mkdir(exist_ok=True)
    CONF_PATH.write_text(DEFAULT_YAML)


class EdgeDetectionError(Exception):
    """Raised when edge detection fails."""


class TrajectoryError(Exception):
    """Raised when trajectory execution fails."""


class CameraInterface(ABC):
    """Abstract camera interface."""

    @abstractmethod
    def capture_pointcloud(self) -> o3d.geometry.PointCloud:
        """Return filtered point cloud in camera frame."""


class RobotInterface(ABC):
    """Abstract robot interface."""

    @abstractmethod
    def stream_line(self, poses: Iterable[np.ndarray], speed: float) -> None:
        """Stream poses to robot at given speed."""

    @abstractmethod
    def stop_motion(self) -> None:
        """Stop robot motion."""

    @abstractmethod
    def force_exceeded(self, limit: float) -> bool:
        """Return True if force torque exceeds ``limit``."""


@dataclass
class CameraConfig:
    width: int = 640
    height: int = 480
    decimation: int = 2
    filter_sigma: float = 1.0


@dataclass
class RobotConfig(RobotConfigBase):
    pass


@dataclass
class EdgeConfig:
    hull_alpha: float = 0.03
    sample_resolution: float = 0.0001
    speed: float = 0.03
    transform_path: str = "handeye.yaml"
    force_limit: float = 2.0


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    edge: EdgeConfig = field(default_factory=EdgeConfig)
    logging: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(cfg: dict[str, Any]) -> "AppConfig":
        camera = CameraConfig(**cfg.get("camera", {}))
        robot = RobotConfig(**cfg.get("robot", {}))
        edge = EdgeConfig(**cfg.get("edge", {}))
        logging_cfg = cfg.get("logging", {})
        return AppConfig(camera=camera, robot=robot, edge=edge, logging=logging_cfg)


class StubCamera(CameraInterface):
    """Synthetic camera for dry runs."""

    def capture_pointcloud(self) -> o3d.geometry.PointCloud:
        xs = np.linspace(0.0, 0.1, 5)
        ys = np.linspace(0.0, 0.05, 5)
        X, Y = np.meshgrid(xs, ys)
        Z_obj = 0.02 * np.ones_like(X)
        obj_pts = np.stack([X.ravel(), Y.ravel(), Z_obj.ravel()], axis=1)
        Z_table = np.zeros_like(X)
        table_pts = np.stack([X.ravel(), Y.ravel(), Z_table.ravel()], axis=1)
        pts = np.vstack([obj_pts, table_pts])
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        return pcd


class RealSenseCamera(CameraInterface):
    """Intel RealSense D415 camera wrapper."""

    def __init__(self, cfg: CameraConfig) -> None:
        if rs is None:
            raise RuntimeError("pyrealsense2 not available")
        self.cfg = cfg
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, cfg.width, cfg.height, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, cfg.width, cfg.height, rs.format.rgb8, 30)
        self.pipeline.start(config)
        self.decimate = rs.decimation_filter()
        self.decimate.set_option(rs.option.filter_magnitude, cfg.decimation)
        self.filter = rs.disparity_transform(True)

    def capture_pointcloud(self) -> o3d.geometry.PointCloud:
        frames = self.pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        depth = self.decimate.process(depth)
        depth = self.filter.process(depth)
        depth_img = o3d.geometry.Image(np.asanyarray(depth.get_data()))
        color_img = o3d.geometry.Image(np.asanyarray(color.get_data()))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img,
            depth_img,
            convert_rgb_to_intensity=False,
        )
        intr = o3d.camera.PinholeCameraIntrinsic(
            self.cfg.width,
            self.cfg.height,
            600,
            600,
            self.cfg.width / 2,
            self.cfg.height / 2,
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)
        return pcd


class StubRobot(RobotInterface):
    """Robot stub that prints poses."""

    def stream_line(self, poses: Iterable[np.ndarray], speed: float) -> None:
        for p in poses:
            logger.info(f"Pose: {p.tolist()}")

    def stop_motion(self) -> None:
        logger.info("Motion stopped")

    def force_exceeded(self, limit: float) -> bool:  # pragma: no cover
        return False


# Real robot implementation omitted for brevity; this project focuses on stubs
class FR3Robot(StubRobot):
    """Placeholder for the real robot SDK wrapper."""


# --- Edge detection utilities ---


def _segment_table(
    pcd: o3d.geometry.PointCloud,
) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    plane_model, inliers = pcd.segment_plane(0.003, 3, 1000)
    object_cloud = pcd.select_by_index(inliers, invert=True)
    return object_cloud, np.array(plane_model)


def _find_top_edge(
    pcd: o3d.geometry.PointCloud, resolution: float
) -> tuple[np.ndarray, o3d.geometry.PointCloud]:
    points = np.asarray(pcd.points)
    if points.size == 0:
        raise EdgeDetectionError("Empty point cloud")
    xy = points[:, :2]
    hull = ConvexHull(xy)
    hull_pts = xy[hull.vertices]
    n = hull_pts.shape[0]
    y_max = hull_pts[:, 1].max()
    best_len = 0.0
    best = None
    for i in range(n):
        p1 = hull_pts[i]
        p2 = hull_pts[(i + 1) % n]
        y_avg = (p1[1] + p2[1]) / 2
        if abs(y_avg - y_max) > 1e-3:
            continue
        length = float(np.linalg.norm(p1 - p2))
        if length > best_len:
            best_len = length
            best = (p1, p2)
    if best is None:
        raise EdgeDetectionError("Top edge not found")
    p1, p2 = best
    steps = max(int(np.ceil(best_len / resolution)) + 1, 2)
    t = np.linspace(0.0, 1.0, steps)
    line_xy = (1 - t)[:, None] * p1 + t[:, None] * p2
    # Approximate Z as max z of object points with same y
    z_vals = []
    for xy_i in line_xy:
        mask = np.linalg.norm(xy - xy_i, axis=1) < 0.005
        if np.any(mask):
            z_vals.append(float(points[mask, 2].max()))
        else:
            z_vals.append(float(points[:, 2].max()))
    line = np.column_stack([line_xy, np.array(z_vals)])
    return line, pcd


def load_transform(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise EdgeDetectionError(f"Transform file not found: {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return np.array(data["transform"], dtype=float)


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1))
    pts_h = np.hstack([points, ones])
    out = (T @ pts_h.T).T
    return cast(np.ndarray, out[:, :3])


def generate_trajectory(
    points: np.ndarray, orientation: np.ndarray, speed: float
) -> list[np.ndarray]:
    poses = []
    for p in points:
        pose = np.hstack([p, orientation])
        poses.append(pose)
    return poses


def save_preview(pcd: o3d.geometry.PointCloud, edge: np.ndarray, out: Path) -> None:
    import matplotlib.pyplot as plt

    pts = np.asarray(pcd.points)
    plt.figure()
    plt.scatter(pts[:, 0], pts[:, 1], s=1)
    plt.plot(edge[:, 0], edge[:, 1], "r-", linewidth=2)
    plt.axis("equal")
    plt.savefig(out)
    plt.close()


def run(cfg: AppConfig, *, dry_run: bool = False) -> None:

    logger.remove()
    logger.add(sys.stdout, level=cfg.logging.get("level", "INFO"))

    camera: CameraInterface = StubCamera() if dry_run else RealSenseCamera(cfg.camera)
    robot: RobotInterface = StubRobot() if dry_run else FR3Robot()

    pcd = camera.capture_pointcloud()
    obj_cloud, _ = _segment_table(pcd)
    edge, obj_cloud = _find_top_edge(obj_cloud, cfg.edge.sample_resolution)

    save_preview(obj_cloud, edge, Path("edge_preview.png"))

    T = np.eye(4)
    if os.path.exists(cfg.edge.transform_path):
        T = load_transform(cfg.edge.transform_path)
    pts_base = transform_points(edge, T)
    traj = generate_trajectory(pts_base, np.array([0, 0, 0]), cfg.edge.speed)

    if dry_run:
        for p in traj:
            logger.info(f"Planned pose: {p.tolist()}")
        return

    robot.stream_line(traj, cfg.edge.speed)


@hydra.main(version_base=None, config_path=str(CONF_DIR), config_name="default")
def main(cfg: DictConfig) -> None:
    cfg_dict = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    app_cfg = AppConfig.from_dict(cfg_dict)
    run(app_cfg)


if __name__ == "__main__":
    main()
