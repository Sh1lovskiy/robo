#!/usr/bin/env python3
"""
Capture RGB-D frames from RealSense D415, rotate a stepper, and save:
- color image (.png), depth (.npy), optional cloud (.ply)
- robot TCP pose (.txt) if a robot is reachable

Notes:
- Uses your capture_rgbd() to get a frame with .color, .depth, .cloud
- Robot RPC is optional; TCP pose is saved as "NO TCP DATA" if missing
- ESP32 stepper is rotated between views
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np
import open3d as o3d

from robot_scan.capture import capture_rgbd
from robot.rpc import RPC
from esp32.control import ESP32Controller
from utils.error_tracker import ErrorTracker
from utils.logger import Logger

# ============================== CONSTANTS ====================================

IMAGE_EXT = ".png"
DEPTH_EXT = ".npy"
CLOUD_EXT = ".ply"

DEFAULT_CAP_DIR = Path("captures")
ROBOT_IP = "192.168.58.2"

NUM_VIEWS = 4
MOTOR_STEPS = 50
MOTOR_DIR = 1
MOTOR_DELAY_US = 5000
MOTOR_SLEEP_S = 2.0

LOG = Logger.get_logger("cap")


# ============================== DATA TYPES ===================================


@dataclass(frozen=True)
class SavePaths:
    color: Path
    depth: Path
    cloud: Path
    tcp_txt: Path


# ============================== UTILITIES ====================================


def create_run_dir(base: Path = DEFAULT_CAP_DIR) -> Path:
    """Create a timestamped run dir."""
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = base / now
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def connect_robot(ip: str) -> Optional[RPC]:
    """Try to connect to the robot; return None on failure."""
    try:
        rpc = RPC(ip=ip)
        LOG.info("Connected to robot at %s", ip)
        return rpc
    except Exception as exc:
        LOG.warning("Robot not connected, TCP will be None: %s", exc)
        return None


def build_esp() -> ESP32Controller:
    """Create ESP32 controller."""
    return ESP32Controller()


def get_tcp_pose(robot: Optional[RPC]) -> Optional[List[float]]:
    """Return TCP pose [x,y,z,rx,ry,rz] or None."""
    if robot is None:
        return None
    try:
        ret = robot.GetActualTCPPose()
    except Exception as exc:
        LOG.error("GetActualTCPPose exception: %s", exc)
        _safe_report(exc)
        return None

    code, pose = _parse_tcp_response(ret)
    if code == 0 and pose is not None and len(pose) == 6:
        try:
            return [float(x) for x in pose]
        except Exception as exc:
            LOG.error("TCP cast error: %s", exc)
            _safe_report(exc)
            return None

    LOG.error("GetActualTCPPose failed: code=%s, pose=%s", code, pose)
    return None


def _parse_tcp_response(ret: object) -> Tuple[int, Optional[List[float]]]:
    """Support common RPC return shapes."""
    if isinstance(ret, (list, tuple)):
        if len(ret) == 2:
            code, pose = ret  # type: ignore[misc]
            pose_list = list(pose) if isinstance(pose, (list, tuple)) else None
            return int(code), pose_list
        if len(ret) == 7:
            code, *pose = ret  # type: ignore[misc]
            return int(code), list(pose)
    LOG.error("Unexpected GetActualTCPPose format: %r", ret)
    return 1, None


def _safe_report(exc: Exception) -> None:
    try:
        ErrorTracker.report(exc)
    except Exception:
        pass


def make_paths(view_dir: Path, stem: str = "frame") -> SavePaths:
    """Build output file paths for a view."""
    base = view_dir / stem
    return SavePaths(
        color=base.with_suffix(IMAGE_EXT),
        depth=base.with_suffix(DEPTH_EXT),
        cloud=base.with_suffix(CLOUD_EXT),
        tcp_txt=base.with_suffix(".txt"),
    )


def save_color(path: Path, img: np.ndarray) -> None:
    """Save BGR image; raise on failure."""
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise IOError(f"cv2.imwrite failed: {path}")


def save_depth(path: Path, depth: np.ndarray) -> None:
    """Save depth as .npy."""
    np.save(str(path), depth)


def save_cloud_if_any(path: Path, frame) -> None:
    """Save cloud if the frame exposes a valid Open3D point cloud."""
    cloud = getattr(frame, "cloud", None)
    if not isinstance(cloud, o3d.geometry.PointCloud):
        return
    # Uncomment if you need normals for downstream tasks.
    # cloud.estimate_normals()
    # cloud.orient_normals_consistent_tangent_plane(10)
    o3d.io.write_point_cloud(str(path), cloud, write_ascii=False)


def save_tcp_txt(path: Path, tcp: Optional[List[float]]) -> None:
    """Save TCP pose or a placeholder."""
    with open(path, "w", encoding="utf-8") as f:
        if tcp is None:
            f.write("NO TCP DATA\n")
            return
        f.write(" ".join(f"{x:.6f}" for x in tcp))


def rotate_motor(esp: ESP32Controller) -> None:
    """Rotate stepper and wait for motion to settle."""
    esp.move_motor(steps=MOTOR_STEPS, direction=MOTOR_DIR, delay_us=MOTOR_DELAY_US)
    time.sleep(MOTOR_SLEEP_S)


# ============================== MAIN FLOW ====================================


def capture_view(
    view_idx: int, total: int, outdir: Path, robot: Optional[RPC], esp: ESP32Controller
) -> None:
    """Capture one view: save data and rotate if not last."""
    LOG.info("Capturing view %d/%d...", view_idx + 1, total)
    view_dir = outdir / f"view_{view_idx}"
    view_dir.mkdir(parents=True, exist_ok=True)
    paths = make_paths(view_dir)

    frame = capture_rgbd()
    save_color(paths.color, frame.color)
    save_depth(paths.depth, frame.depth)
    save_cloud_if_any(paths.cloud, frame)

    tcp_pose = get_tcp_pose(robot)
    save_tcp_txt(paths.tcp_txt, tcp_pose)

    LOG.info(
        "Saved:\n  %s\n  %s\n  %s\n  %s",
        paths.color,
        paths.depth,
        paths.cloud,
        paths.tcp_txt,
    )

    if view_idx < total - 1:
        LOG.info("Rotating motor...")
        rotate_motor(esp)


def main() -> None:
    """Entry point."""
    ErrorTracker.install_excepthook()
    ErrorTracker.install_signal_handlers()

    outdir = create_run_dir(DEFAULT_CAP_DIR)
    esp = build_esp()
    robot = connect_robot(ROBOT_IP)

    try:
        for idx in range(NUM_VIEWS):
            capture_view(idx, NUM_VIEWS, outdir, robot, esp)
        LOG.info("All data saved in %s", outdir)
    except KeyboardInterrupt:
        LOG.info("Interrupted by user.")
    except Exception as exc:
        LOG.error("Unhandled error: %s", exc)
        _safe_report(exc)


if __name__ == "__main__":
    main()
