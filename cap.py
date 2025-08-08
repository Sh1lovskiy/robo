from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Optional, List

import numpy as np
import cv2
import open3d as o3d

from robot_scan.capture import capture_rgbd
from robot.rpc import RPC
from esp32.control import ESP32Controller
from utils.error_tracker import ErrorTracker
from utils.logger import Logger
from pathlib import Path

IMAGE_EXT = ".png"
DEPTH_EXT = ".npy"
CLOUD_EXT = ".ply"

logger = Logger.get_logger("cap")


def get_tcp_pose(robot: Optional[RPC]) -> Optional[List[float]]:
    """Return current TCP pose as [x, y, z, rx, ry, rz], or None on error."""
    if robot is None:
        return None
    try:
        ret = robot.GetActualTCPPose()
        if isinstance(ret, (list, tuple)):
            if len(ret) == 2:
                code, pose = ret
            elif len(ret) == 7:
                code, *pose = ret
                pose = list(pose)
            else:
                logger.error(f"Unexpected GetActualTCPPose format: {ret}")
                return None
        else:
            logger.error(f"Unexpected GetActualTCPPose type: {type(ret)}")
            return None

        if code == 0 and isinstance(pose, (list, tuple)) and len(pose) == 6:
            return [float(x) for x in pose]
        else:
            logger.error(f"GetActualTCPPose failed: code={code}, pose={pose}")
            return None
    except Exception as exc:
        logger.error(f"get_tcp_pose exception: {exc}")
        try:
            ErrorTracker.report(exc)
        except Exception:
            pass
        return None


def create_run_dir(base="captures") -> str:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(base, now)
    os.makedirs(outdir, exist_ok=True)
    return outdir


def save_tcp(tcp: Optional[List[float]], base: Path) -> None:
    path = str(base.with_suffix(".txt"))
    with open(path, "w") as f:
        if tcp is not None:
            f.write(" ".join(f"{x:.6f}" for x in tcp))
        else:
            f.write("NO TCP DATA\n")


def main() -> None:
    outdir = create_run_dir("captures")
    esp = ESP32Controller()

    try:
        rpc = RPC(ip="192.168.58.2")
        logger.info("Connected to robot at 192.168.58.2")
    except Exception as e:
        logger.warning(f"Robot not connected, TCP will be None: {e}")
        rpc = None

    for idx in range(4):
        logger.info(f"Capturing view {idx + 1}/4...")
        subdir = Path(outdir) / f"view_{idx}"
        subdir.mkdir(parents=True, exist_ok=True)
        base = subdir / "frame"

        frame = capture_rgbd()
        # save
        cv2.imwrite(str(base.with_suffix(IMAGE_EXT)), frame.color)
        np.save(str(base.with_suffix(DEPTH_EXT)), frame.depth)
        # frame.cloud.estimate_normals()
        # frame.cloud.orient_normals_consistent_tangent_plane(10)
        # o3d.io.write_point_cloud(str(base.with_suffix(CLOUD_EXT)), frame.cloud)

        tcp_pose = get_tcp_pose(rpc)
        save_tcp(tcp_pose, base)
        logger.info(
            f"Saved: {base.with_suffix(IMAGE_EXT)}\n       {base.with_suffix(DEPTH_EXT)}"
            f"\n       {base.with_suffix(CLOUD_EXT)}\n       {base.with_suffix('.txt')}"
        )

        if idx < 3:
            logger.info("Rotating motor 90 degrees...")
            esp.move_motor(steps=50, direction=1, delay_us=5000)
            time.sleep(2.0)

    logger.info(f"All data saved in {outdir}")


if __name__ == "__main__":
    main()
