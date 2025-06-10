# cli/vis_pointcloud.py

import open3d as o3d
from utils.logger import Logger


def view_pointcloud(filename, logger=None):
    logger = logger or Logger.get_logger("cli.vis_pointcloud")
    pcd = o3d.io.read_point_cloud(filename)
    logger.info(f"Loaded {filename}, {len(pcd.points)} points")
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    view_pointcloud(args.input)
