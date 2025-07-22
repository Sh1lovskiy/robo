#!/usr/bin/env python3

import open3d as o3d
import numpy as np
import pyrealsense2 as rs
import cv2

from robot.controller import RobotController


def show_realsense_pointcloud(
    robot_controller, min_depth=0.2, max_depth=2.0, align_to_color=True
):

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color) if align_to_color else None

    for _ in range(10):
        frames = pipeline.wait_for_frames()
        if align:
            frames = align.process(frames)
    frames = pipeline.wait_for_frames()
    if align:
        frames = align.process(frames)

    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if not color_frame or not depth_frame:
        print("Frames not found")
        pipeline.stop()
        return

    color_img = np.asanyarray(color_frame.get_data())
    depth_img = np.asanyarray(depth_frame.get_data())
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    print(f"Depth scale: {depth_scale:.5f} meters per unit")

    depth_meter = depth_img.astype(np.float32) * depth_scale

    depth_intr = depth_frame.profile.as_video_stream_profile().get_intrinsics()
    pinhole = o3d.camera.PinholeCameraIntrinsic(
        depth_intr.width,
        depth_intr.height,
        depth_intr.fx,
        depth_intr.fy,
        depth_intr.ppx,
        depth_intr.ppy,
    )

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)),
        o3d.geometry.Image(depth_meter),
        depth_scale=1.0,
        depth_trunc=max_depth,
        convert_rgb_to_intensity=False,
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole)
    points = np.asarray(pcd.points)
    mask = (points[:, 2] > min_depth) & (points[:, 2] < max_depth)
    pcd = pcd.select_by_index(np.where(mask)[0])

    print(f"Cloud has {len(pcd.points)} points")
    pipeline.stop()

    tcp_pose = robot_controller.get_tcp_pose()
    if tcp_pose is None:
        print("TCP pose not available, saving as cloud_unknown.ply")
        out_file = "cloud_unknown.ply"
    else:
        out_file = (
            f"cloud_x{tcp_pose[0]:.1f}_y{tcp_pose[1]:.1f}_z{tcp_pose[2]:.1f}_"
            f"rx{tcp_pose[3]:.1f}_ry{tcp_pose[4]:.1f}_rz{tcp_pose[5]:.1f}.ply"
        )
    o3d.io.write_point_cloud(out_file, pcd)
    print(f"Saved point cloud to: {out_file}")

    o3d.visualization.draw_geometries_with_editing(
        [pcd], window_name="RealSense PointCloud"
    )


if __name__ == "__main__":
    robot_controller = RobotController()
    show_realsense_pointcloud(robot_controller)
