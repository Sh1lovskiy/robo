import os
from PIL import Image


def convert_dng_to_jpg_alternative(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    all_files = os.listdir(input_folder)
    dng_files = [
        os.path.join(input_folder, f) for f in all_files if f.lower().endswith(".dng")
    ]

    for dng_file in dng_files:
        try:
            filename = os.path.basename(dng_file)
            name, _ = os.path.splitext(filename)
            output_path = os.path.join(output_folder, f"{name}.jpg")

            with Image.open(dng_file) as img:
                img.save(output_path, "JPEG")
                print(f"Success to convert {dng_file} -> {output_path}")
        except Exception as e:
            print(f"Convert error {dng_file}: {str(e)}")


input_folder = "funetune/orig"
output_folder = "funetune/orig_jpg"

convert_dng_to_jpg_alternative(input_folder, output_folder)
