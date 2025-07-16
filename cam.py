import pyrealsense2 as rs
import json


def main():
    # Подключаем камеру
    ctx = rs.context()
    if len(ctx.devices) == 0:
        print("Нет подключённых RealSense камер")
        return

    # Настраиваем пайплайн
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline_profile = pipeline.start(config)

    # Получаем профили стримов
    depth_stream = pipeline_profile.get_stream(
        rs.stream.depth
    ).as_video_stream_profile()
    color_stream = pipeline_profile.get_stream(
        rs.stream.color
    ).as_video_stream_profile()

    # Получаем интринсики глубинной камеры
    depth_intr = depth_stream.get_intrinsics()
    depth_intr_dict = {
        "width": depth_intr.width,
        "height": depth_intr.height,
        "fx": depth_intr.fx,
        "fy": depth_intr.fy,
        "ppx": depth_intr.ppx,
        "ppy": depth_intr.ppy,
        "model": str(depth_intr.model),
        "coeffs": list(depth_intr.coeffs),
    }

    # Экстринсики: из глубины в цвет
    extr = depth_stream.get_extrinsics_to(color_stream)
    extr_dict = {
        "rotation": [
            list(extr.rotation[0:3]),
            list(extr.rotation[3:6]),
            list(extr.rotation[6:9]),
        ],
        "translation": list(extr.translation),
    }

    # Интринсики цветной (если надо)
    color_intr = color_stream.get_intrinsics()
    color_intr_dict = {
        "width": color_intr.width,
        "height": color_intr.height,
        "fx": color_intr.fx,
        "fy": color_intr.fy,
        "ppx": color_intr.ppx,
        "ppy": color_intr.ppy,
        "model": str(color_intr.model),
        "coeffs": list(color_intr.coeffs),
    }

    data = {
        "depth_intrinsics": depth_intr_dict,
        "color_intrinsics": color_intr_dict,
        "depth_to_color_extrinsics": extr_dict,
    }

    with open("realsense_params.json", "w") as f:
        json.dump(data, f, indent=2)

    print("Параметры камеры сохранены в realsense_params.json")
    pipeline.stop()


if __name__ == "__main__":
    main()
