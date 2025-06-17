# cli/pointcloud_view.py
"""Visualize a PLY point cloud with Open3D."""
import argparse
import open3d as o3d
from utils.logger import Logger


def main():
    parser = argparse.ArgumentParser(description="View a point cloud PLY file.")
    parser.add_argument("--input", required=True, help="Input PLY file")
    args = parser.parse_args()

    logger = Logger.get_logger("cli.pointcloud_view")
    pcd = o3d.io.read_point_cloud(args.input)
    logger.info(f"Loaded point cloud: {args.input}")
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
