import pyrealsense2 as rs
import json


def get_intrinsics_as_dict(intrinsics):
    return {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "ppx": intrinsics.ppx,
        "ppy": intrinsics.ppy,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "model": str(intrinsics.model),
        "coeffs": list(intrinsics.coeffs),
    }


def main(out_json="realsense_params.json"):
    # 1. run pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth)
    config.enable_stream(rs.stream.color)
    profile = pipeline.start(config)

    try:
        # 2. camera warm up
        for _ in range(10):
            pipeline.wait_for_frames()

        # 3. streams
        depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()

        # 4. intr's
        intr_depth = depth_profile.get_intrinsics()
        intr_color = color_profile.get_intrinsics()

        # 5. depth_to_color and color_to_depth
        extr_depth2color = depth_profile.get_extrinsics_to(color_profile)
        extr_color2depth = color_profile.get_extrinsics_to(depth_profile)

        # 6. depth scale
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        out = {
            "depth_scale": depth_scale,
            "intrinsics": {
                "depth": get_intrinsics_as_dict(intr_depth),
                "color": get_intrinsics_as_dict(intr_color),
            },
            "extrinsics": {
                "depth_to_color": {
                    "rotation": [
                        list(row) for row in zip(*[iter(extr_depth2color.rotation)] * 3)
                    ],
                    "translation": list(extr_depth2color.translation),
                },
                "color_to_depth": {
                    "rotation": [
                        list(row) for row in zip(*[iter(extr_color2depth.rotation)] * 3)
                    ],
                    "translation": list(extr_color2depth.translation),
                },
            },
        }

        with open(out_json, "w") as f:
            json.dump(out, f, indent=4)
        print(f"Saved to {out_json}")

    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
