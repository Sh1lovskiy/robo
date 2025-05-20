import os
import cv2
from pathlib import Path


input_dir = "vision/imgs_fix"
output_dir = "vision/imgs_fix/left_imgs"

Path(output_dir).mkdir(parents=True, exist_ok=True)

for img_name in os.listdir(input_dir):
    if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)

        if img is not None:
            height, width = img.shape[:2]
            left_half = img[:, : width // 2]
            output_path = os.path.join(output_dir, img_name)
            cv2.imwrite(output_path, left_half)
