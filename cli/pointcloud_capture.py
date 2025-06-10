# cli/pointcloud_capture.py

import argparse
from vision.realsense import RealSenseCamera
from vision.pointcloud import PointCloudGenerator
from utils.logger import Logger
from utils.constants import CLOUD_OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Capture and save a point cloud from RealSense."
    )
    parser.add_argument(
        "--output", default=f"{CLOUD_OUTPUT_DIR}/cloud.ply", help="Output PLY file"
    )
    args = parser.parse_args()

    logger = Logger.get_logger("cli.pointcloud_capture")
    cam = RealSenseCamera()
    cloud_gen = PointCloudGenerator(logger)
    cam.start()
    color, depth = cam.get_frames()
    intr = cam.get_intrinsics()
    points, colors = cloud_gen.depth_to_cloud(depth, intr, rgb=color)
    cloud_gen.save_ply(args.output, points, colors)
    logger.info(f"Saved point cloud: {args.output}")
    cam.stop()


if __name__ == "__main__":
    main()
