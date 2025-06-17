# cli/pointcloud_transform.py

import argparse
import numpy as np
from vision.pointcloud import PointCloudGenerator
from vision.transform import TransformUtils
from utils.logger import Logger
from utils.config import Config


def main():
    parser = argparse.ArgumentParser(
        description="Transform a point cloud using calibration."
    )
    parser.add_argument("--input", required=True, help="Input PLY file")
    parser.add_argument("--calib", required=True, help="Calibration .npz file")
    Config.load("config.yaml")
    out_dir = Config.get("cloud.output_dir", "clouds")
    parser.add_argument(
        "--output",
        default=f"{out_dir}/cloud_world.ply",
        help="Output PLY file",
    )
    args = parser.parse_args()

    logger = Logger.get_logger("cli.pointcloud_transform")
    cloud_gen = PointCloudGenerator(logger)
    points, colors = cloud_gen.load_ply(args.input)

    calib = np.load(args.calib)
    R, t = calib["R"], calib["t"]

    points_t = TransformUtils.apply_transform(points, R, t)
    cloud_gen.save_ply(args.output, points_t, colors)
    logger.info(f"Transformed cloud saved: {args.output}")


if __name__ == "__main__":
    main()
