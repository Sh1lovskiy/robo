"""RGB-D capture utilities using Intel RealSense."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs

from utils.logger import Logger
from utils.settings import camera as cam_cfg

logger = Logger.get_logger("robot_scan.capture")


@dataclass
class RGBDFrame:
    """Container for captured RGB-D data."""

    color: np.ndarray
    depth: np.ndarray
    cloud: o3d.geometry.PointCloud


def capture_rgbd() -> RGBDFrame:
    """Capture a single RGB-D frame and convert to a point cloud.

    Returns
    -------
    RGBDFrame
        Captured color image, depth image (in meters) and the corresponding
        point cloud filtered for valid depth range.
    """
    logger.info("Starting RealSense pipeline for capture")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.color,
        cam_cfg.rgb_width,
        cam_cfg.rgb_height,
        rs.format.bgr8,
        cam_cfg.fps,
    )
    config.enable_stream(
        rs.stream.depth,
        cam_cfg.depth_width,
        cam_cfg.depth_height,
        rs.format.z16,
        cam_cfg.fps,
    )
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    for _ in range(5):
        align.process(pipeline.wait_for_frames())
    frames = align.process(pipeline.wait_for_frames())

    color = np.asanyarray(frames.get_color_frame().get_data())
    depth_raw = frames.get_depth_frame()
    depth = np.asanyarray(depth_raw.get_data()).astype(np.float32)
    scale = profile.get_device().first_depth_sensor().get_depth_scale()
    depth *= scale

    intr = depth_raw.profile.as_video_stream_profile().get_intrinsics()
    pinhole = o3d.camera.PinholeCameraIntrinsic(
        intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy
    )
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(cv2.cvtColor(color, cv2.COLOR_BGR2RGB)),
        o3d.geometry.Image(depth),
        depth_scale=1.0,
        convert_rgb_to_intensity=False,
    )
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole)
    mask = np.logical_and(
        0.2 < np.asarray(cloud.points)[:, 2], np.asarray(cloud.points)[:, 2] < 2.0
    )
    cloud = cloud.select_by_index(np.where(mask)[0])
    pipeline.stop()
    logger.info(f"Captured frame: {len(cloud.points)} points")
    return RGBDFrame(color=color, depth=depth, cloud=cloud)
