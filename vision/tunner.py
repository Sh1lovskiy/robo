import cv2
import numpy as np
import json
from vision.d415_pipeline import (
    RealSenseD415,
    D415CameraSettings,
    D415FilterConfig,
)
from vision.opencv_utils import OpenCVUtils
from robot.workflows import CameraManager
import pyrealsense2 as rs


def nothing(x):
    pass


def update_filters(
    cam,
    decimation,
    spatial_alpha,
    spatial_delta,
    temporal_alpha,
    temporal_delta,
    hole_filling,
):
    cam.decimation.set_option(rs.option.filter_magnitude, float(decimation))
    cam.spatial.set_option(rs.option.filter_smooth_alpha, float(spatial_alpha))
    cam.spatial.set_option(rs.option.filter_smooth_delta, float(spatial_delta))
    cam.temporal.set_option(rs.option.filter_smooth_alpha, float(temporal_alpha))
    cam.temporal.set_option(rs.option.filter_smooth_delta, float(temporal_delta))
    cam.holes = type(cam.holes)(hole_filling)


def main():
    target_w, target_h = 960, 540
    cam = RealSenseD415(
        settings=D415CameraSettings(
            ir_exposure=8000, ir_gain=16, rgb_exposure=50, rgb_gain=64
        ),
        filters=D415FilterConfig(
            decimation=2,
            spatial_alpha=0.5,
            spatial_delta=20,
            temporal_alpha=0.4,
            temporal_delta=20,
            hole_filling=1,
        ),
    )
    camera_mgr = CameraManager(camera=cam)
    if not camera_mgr.start():
        print("Camera not available. Exiting tuner.")
        return
    opencv_utils = OpenCVUtils(display_width=target_w, display_height=target_h)
    cv2.namedWindow("Tuner", cv2.WINDOW_NORMAL)
    cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("decimation", "Tuner", 2, 8, nothing)
    cv2.createTrackbar("spatial_alpha x100", "Tuner", 50, 100, nothing)
    cv2.createTrackbar("spatial_delta", "Tuner", 20, 50, nothing)
    cv2.createTrackbar("temporal_alpha x100", "Tuner", 40, 100, nothing)
    cv2.createTrackbar("temporal_delta", "Tuner", 20, 50, nothing)
    cv2.createTrackbar("hole_filling", "Tuner", 1, 2, nothing)
    cv2.createTrackbar("IR exposure", "Tuner", 8000, 16000, nothing)
    cv2.createTrackbar("IR gain", "Tuner", 16, 100, nothing)
    cv2.createTrackbar("RGB exposure", "Tuner", 50, 1000, nothing)
    cv2.createTrackbar("RGB gain", "Tuner", 64, 100, nothing)
    cv2.createTrackbar("Projector", "Tuner", 0, 360, nothing)

    print("Use sliders to tune RealSense. ESC — save & exit.")
    last_rgb, last_depth = None, None

    while True:
        decimation = max(1, cv2.getTrackbarPos("decimation", "Tuner"))
        spatial_alpha = cv2.getTrackbarPos("spatial_alpha x100", "Tuner") / 100.0
        spatial_delta = cv2.getTrackbarPos("spatial_delta", "Tuner")
        temporal_alpha = cv2.getTrackbarPos("temporal_alpha x100", "Tuner") / 100.0
        temporal_delta = cv2.getTrackbarPos("temporal_delta", "Tuner")
        hole_filling = cv2.getTrackbarPos("hole_filling", "Tuner")
        ir_exp = cv2.getTrackbarPos("IR exposure", "Tuner")
        ir_gain = cv2.getTrackbarPos("IR gain", "Tuner")
        rgb_exp = cv2.getTrackbarPos("RGB exposure", "Tuner")
        rgb_gain = cv2.getTrackbarPos("RGB gain", "Tuner")
        projector = cv2.getTrackbarPos("Projector", "Tuner")

        cam.decimation.set_option(rs.option.filter_magnitude, float(decimation))
        cam.spatial.set_option(rs.option.filter_smooth_alpha, float(spatial_alpha))
        cam.spatial.set_option(rs.option.filter_smooth_delta, float(spatial_delta))
        cam.temporal.set_option(rs.option.filter_smooth_alpha, float(temporal_alpha))
        cam.temporal.set_option(rs.option.filter_smooth_delta, float(temporal_delta))
        cam.holes = rs.hole_filling_filter(hole_filling)
        cam.depth_sensor.set_option(rs.option.exposure, float(ir_exp))
        cam.depth_sensor.set_option(rs.option.gain, float(ir_gain))
        cam.rgb_sensor.set_option(rs.option.exposure, float(rgb_exp))
        cam.rgb_sensor.set_option(rs.option.gain, float(rgb_gain))
        cam.depth_sensor.set_option(rs.option.laser_power, float(projector))

        color, depth = camera_mgr.get_frames()
        if color is not None and color.size > 0:
            rgb_show = cv2.resize(color, (target_w, target_h))
            last_rgb = rgb_show
        else:
            rgb_show = (
                last_rgb
                if last_rgb is not None
                else np.zeros((target_h, target_w, 3), dtype=np.uint8)
            )
        cv2.imshow("RGB", rgb_show)

        if depth is not None and depth.size > 0:
            depth_norm = opencv_utils.normalize_depth(depth)
            depth_show = opencv_utils.apply_colormap(depth_norm)
            depth_show = cv2.resize(depth_show, (target_w, target_h))
            last_depth = depth_show
        else:
            depth_show = (
                last_depth
                if last_depth is not None
                else np.zeros((target_h, target_w, 3), dtype=np.uint8)
            )
        cv2.imshow("Depth", depth_show)

        display = np.vstack([rgb_show, depth_show])
        cv2.imshow("Tuner", display)

        key = cv2.waitKey(10)
        if key == 27:
            params = dict(
                decimation=decimation,
                spatial_alpha=spatial_alpha,
                spatial_delta=spatial_delta,
                temporal_alpha=temporal_alpha,
                temporal_delta=temporal_delta,
                hole_filling=hole_filling,
                ir_exposure=ir_exp,
                ir_gain=ir_gain,
                rgb_exposure=rgb_exp,
                rgb_gain=rgb_gain,
                projector_power=projector,
            )
            with open("d415_last_tune.json", "w") as f:
                json.dump(params, f, indent=2)
            print("Params saved to d415_last_tune.json. Exiting...")
            break

    camera_mgr.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
