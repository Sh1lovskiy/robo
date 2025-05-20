import os
import struct


def collect_pairs(image_list):
    pairs = []
    for i in range(len(image_list)):
        for j in range(i + 1, min(i + 30, len(image_list))):  # N nearest
            pairs.append((image_list[i], image_list[j]))
    return pairs


def generate_stereo_pairs():
    LEFT_DIR = "/home/sha/Documents/work/robohand_v2/colmap/data1/images/devide/left"
    RIGHT_DIR = "/home/sha/Documents/work/robohand_v2/colmap/data1/images/devide/right"
    OUTPUT_TXT = "/home/sha/Documents/work/robohand_v2/colmap/data1/stereo_pairs.txt"
    OUTPUT_BIN = "/home/sha/Documents/work/robohand_v2/colmap/data1/stereo_pairs.bin"

    # Generate pairs and save to text file
    left_images = sorted(os.listdir(LEFT_DIR))
    right_images = sorted(os.listdir(RIGHT_DIR))

    left_images = [f"left/{img}" for img in left_images if img.endswith(".png")]
    right_images = [f"right/{img}" for img in right_images if img.endswith(".png")]

    pairs = collect_pairs(left_images) + collect_pairs(right_images)

    with open(OUTPUT_TXT, "w") as f:
        for a, b in pairs:
            f.write(f"{a} {b}\n")

    print(f"Saved {len(pairs)} stereo pairs to {OUTPUT_TXT}")

    # Convert text pairs to indices and save to binary file
    # First create a mapping from image names to indices
    all_images = left_images + right_images
    image_to_index = {img: idx for idx, img in enumerate(all_images)}

    # Convert pairs to indices
    index_pairs = []
    for img1, img2 in pairs:
        i1 = image_to_index[img1]
        i2 = image_to_index[img2]
        index_pairs.append((i1, i2))

    # Save to binary file
    with open(OUTPUT_BIN, "wb") as f:
        f.write(struct.pack("Q", len(index_pairs)))
        for i1, i2 in index_pairs:
            f.write(struct.pack("ii", i1, i2))

    print(f"Saved {len(index_pairs)} stereo pairs to {OUTPUT_BIN} in binary format")


def generate_true_stereo_pairs():
    LEFT_DIR = "/home/sha/Documents/work/robohand_v2/colmap/data1/images/devide/left"
    RIGHT_DIR = "/home/sha/Documents/work/robohand_v2/colmap/data1/images/devide/right"
    OUTPUT_TXT = "/home/sha/Documents/work/robohand_v2/colmap/data1/stereo_pairs.txt"
    OUTPUT_BIN = "/home/sha/Documents/work/robohand_v2/colmap/data1/stereo_pairs.bin"

    left_images = sorted([img for img in os.listdir(LEFT_DIR) if img.endswith(".png")])
    right_images = sorted(
        [img for img in os.listdir(RIGHT_DIR) if img.endswith(".png")]
    )

    assert len(left_images) == len(
        right_images
    ), "Mismatch in number of left/right images"

    pairs = []
    all_images = []

    for left_img, right_img in zip(left_images, right_images):
        left_path = f"left/{left_img}"
        right_path = f"right/{right_img}"
        pairs.append((left_path, right_path))
        all_images.append(left_path)
        all_images.append(right_path)

    with open(OUTPUT_TXT, "w") as f:
        for a, b in pairs:
            f.write(f"{a} {b}\n")

    print(f"Saved {len(pairs)} stereo pairs to {OUTPUT_TXT}")

    # Map image names to indices
    image_to_index = {img: idx for idx, img in enumerate(all_images)}

    with open(OUTPUT_BIN, "wb") as f:
        f.write(struct.pack("Q", len(pairs)))
        for img1, img2 in pairs:
            i1 = image_to_index[img1]
            i2 = image_to_index[img2]
            f.write(struct.pack("ii", i1, i2))

    print(f"Saved {len(pairs)} stereo pairs to {OUTPUT_BIN} in binary format")


if __name__ == "__main__":
    generate_true_stereo_pairs()
