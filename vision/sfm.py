"""RGB-D Structure from Motion utilities."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
import open3d as o3d
import pyrealsense2 as rs

from utils.logger import Logger, LoggerType
from utils.settings import paths, vision as vision_settings
from vision.camera.realsense_d415 import RealSenseD415, D415StreamConfig
from vision.pointcloud.generator import PointCloudGenerator
from vision.opencv_utils import OpenCVUtils


@dataclass
class SFMConfig:
    """Parameters controlling the reconstruction volume."""

    voxel_length: float = 0.004
    sdf_trunc: float = 0.04
    depth_trunc: float = 2.0
    visualize: bool = False
    show_frames: bool = False
    output: Path = field(default_factory=lambda: paths.CLOUD_DIR / "sfm_map.ply")


class RGBDSFM:
    """Incremental RGB-D reconstruction using Open3D TSDF volume."""

    def __init__(
        self,
        camera: RealSenseD415,
        config: SFMConfig | None = None,
        logger: LoggerType | None = None,
    ) -> None:
        self.camera = camera
        self.config = config or SFMConfig()
        self.logger = logger or Logger.get_logger("vision.sfm")
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.config.voxel_length,
            sdf_trunc=self.config.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )
        self.pcd_gen = PointCloudGenerator()
        self.prev_rgbd: o3d.geometry.RGBDImage | None = None
        self.pose = np.eye(4)
        self.intrinsic: o3d.camera.PinholeCameraIntrinsic | None = None

    def _update_intrinsics(self) -> None:
        if not self.camera.profile:
            raise RuntimeError("Camera not started")
        stream = self.camera.profile.get_stream(rs.stream.color)
        intr = stream.as_video_stream_profile().get_intrinsics()
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy
        )
        self.logger.info("Camera intrinsics updated")

    def _to_rgbd(self, color: np.ndarray, depth: np.ndarray) -> o3d.geometry.RGBDImage:
        depth_m = depth.astype(np.float32) * self.camera.depth_scale
        color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        d_img = o3d.geometry.Image(depth_m)
        c_img = o3d.geometry.Image(color_rgb)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            c_img,
            d_img,
            depth_scale=0.001,
            depth_trunc=self.config.depth_trunc,
            convert_rgb_to_intensity=False,
        )
        return rgbd

    def process_frame(self, color: np.ndarray, depth: np.ndarray) -> None:
        if self.intrinsic is None:
            self._update_intrinsics()
        rgbd = self._to_rgbd(color, depth)
        if self.prev_rgbd is None:
            self.volume.integrate(rgbd, self.intrinsic, np.linalg.inv(self.pose))
            self.prev_rgbd = rgbd
            return
        option = o3d.pipelines.odometry.OdometryOption()
        odo_init = np.eye(4)
        success, trans, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
            rgbd,
            self.prev_rgbd,
            self.intrinsic,
            odo_init,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
            option,
        )
        if success:
            self.pose = self.pose @ trans
        else:
            self.logger.warning("Odometry failed for frame")
        self.volume.integrate(rgbd, self.intrinsic, np.linalg.inv(self.pose))
        self.prev_rgbd = rgbd

    def run(self, num_frames: int = 0) -> o3d.geometry.PointCloud:
        self.camera.start()
        self._update_intrinsics()
        frame_count = 0
        vis: o3d.visualization.Visualizer | None = None
        if self.config.visualize:
            vis = o3d.visualization.Visualizer()
            vis.create_window("RGB-D SFM")
        cv_utils = OpenCVUtils() if self.config.show_frames else None
        try:
            while True:
                color, depth = self.camera.get_frames()
                if color is None or depth is None:
                    self.logger.warning("Invalid frame skipped")
                    continue
                self.process_frame(color, depth)
                if cv_utils:
                    rgb_disp = cv2.resize(
                        color,
                        (cv_utils.display_width, cv_utils.display_height),
                        interpolation=cv2.INTER_AREA,
                    )
                    cv2.imshow("RGB", rgb_disp)
                    cv_utils.show_depth(depth, "Depth")
                if vis:
                    pcd = self.volume.extract_point_cloud()
                    pcd = pcd.voxel_down_sample(self.config.voxel_length)
                    vis.clear_geometries()
                    vis.add_geometry(pcd)
                    vis.poll_events()
                    vis.update_renderer()
                frame_count += 1
                if num_frames and frame_count >= num_frames:
                    break
                key = cv2.waitKey(1) & 0xFF if cv_utils else 0
                if key == ord("q"):
                    break
        finally:
            if vis:
                vis.destroy_window()
            if cv_utils:
                cv2.destroyAllWindows()
            self.camera.stop()
        pcd = self.volume.extract_point_cloud()
        return pcd.voxel_down_sample(self.config.voxel_length)

    def save(self, pcd: o3d.geometry.PointCloud) -> None:
        self.config.output.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(self.config.output), pcd)
        self.logger.info(f"Saved reconstruction to {self.config.output}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RGB-D SFM")
    parser.add_argument("--frames", type=int, default=0, help="Capture limit")
    parser.add_argument("--voxel", type=float, default=0.004, help="Voxel size")
    parser.add_argument("--trunc", type=float, default=0.04, help="SDF truncation")
    parser.add_argument("--depth_trunc", type=float, default=2.0, help="Depth trunc")
    parser.add_argument("--visualize", action="store_true", help="Show live map")
    parser.add_argument(
        "--show-frames", action="store_true", help="Display RGB and depth windows"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=paths.CLOUD_DIR / "sfm_map.ply",
        help="Output PLY path",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logger = Logger.get_logger("vision.sfm")
    stream_cfg = D415StreamConfig(
        depth_width=vision_settings.realsense_width,
        depth_height=vision_settings.realsense_height,
        color_width=vision_settings.realsense_width,
        color_height=vision_settings.realsense_height,
        fps=vision_settings.realsense_fps,
    )
    cam = RealSenseD415(stream_cfg=stream_cfg, logger=logger)
    cfg = SFMConfig(
        voxel_length=args.voxel,
        sdf_trunc=args.trunc,
        depth_trunc=args.depth_trunc,
        visualize=args.visualize,
        show_frames=args.show_frames,
        output=args.output,
    )
    sfm = RGBDSFM(cam, cfg, logger)
    pcd = sfm.run(num_frames=args.frames)
    sfm.save(pcd)


if __name__ == "__main__":
    main()
