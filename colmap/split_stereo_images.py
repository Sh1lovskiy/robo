import os
import cv2
import argparse

def split_stereo_images(input_dir, left_dir, right_dir):
    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)

    for img_name in os.listdir(input_dir):
        if not img_name.endswith('.png'):
            continue
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load {img_path}")
            continue

        height, width = img.shape[:2]
        if width % 2 != 0:
            print(f"Warning: Image {img_name} has odd width {width}, expected even")
        
        half_width = width // 2
        left_img = img[:, :half_width]
        right_img = img[:, half_width:]

        base_name = img_name.replace('.png', '')
        left_path = os.path.join(left_dir, f"{base_name}.png")
        right_path = os.path.join(right_dir, f"{base_name}.png")

        cv2.imwrite(left_path, left_img)
        cv2.imwrite(right_path, right_img)
        print(f"Processed: {img_name} -> {left_path}, {right_path}")

        # Debug: Check output image sizes
        left_loaded = cv2.imread(left_path)
        right_loaded = cv2.imread(right_path)
        print(f"Left image size: {left_loaded.shape[:2]}, Right image size: {right_loaded.shape[:2]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split stereo images into left and right.")
    parser.add_argument("--input_dir", required=True, help="Input directory with stereo images")
    parser.add_argument("--left_dir", required=True, help="Output directory for left images")
    parser.add_argument("--right_dir", required=True, help="Output directory for right images")
    args = parser.parse_args()

    split_stereo_images(args.input_dir, args.left_dir, args.right_dir)