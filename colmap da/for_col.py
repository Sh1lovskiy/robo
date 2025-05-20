import json
import os
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import shutil


# === Paths ===
colmap_data_dir = "/home/sha/Documents/work/robohand_v2/colmap/data1"
camera_poses_json = os.path.join(colmap_data_dir, "camera_poses.json")
hand_results_json = os.path.join(colmap_data_dir, "hand_results.json")
calibration_xml = os.path.join(colmap_data_dir, "stereo_calibration.xml")

left_images_folder = os.path.join(colmap_data_dir, "images", "devide", "left")
right_images_folder = os.path.join(colmap_data_dir, "images", "devide", "right")
dense_images_dir = os.path.join(colmap_data_dir, "dense", "images")

cameras_txt_path = os.path.join(colmap_data_dir, "cameras.txt")
images_txt_path = os.path.join(colmap_data_dir, "images.txt")
points3d_txt_path = os.path.join(colmap_data_dir, "points3D.txt")
left_params_txt = os.path.join(colmap_data_dir, "left_camera_params.txt")
right_params_txt = os.path.join(colmap_data_dir, "right_camera_params.txt")


# === Load hand-eye calibration ===
def load_T_cam2ee(path):
    with open(path, "r") as f:
        data = json.load(f)
    for entry in data["results"]:
        if entry["method"] == "Andreff":
            return np.array(entry["T_cam2ee"]), entry["mean_translation_error_mm"]
    raise ValueError("No 'Andreff' method found in hand_results.json")


T_cam2ee, _ = load_T_cam2ee(hand_results_json)
T_ee2cam = np.linalg.inv(T_cam2ee)

# === Load camera poses ===
with open(camera_poses_json, "r") as f:
    camera_poses = json.load(f)
print(f"Loaded {len(camera_poses)} camera poses from {camera_poses_json}")

# === Load stereo calibration ===
fs = cv2.FileStorage(calibration_xml, cv2.FILE_STORAGE_READ)
left_K = fs.getNode("left_camera_matrix").mat()
right_K = fs.getNode("right_camera_matrix").mat()
R_rel = fs.getNode("R").mat()
T_rel = fs.getNode("T").mat()
fs.release()

# === Save intrinsic parameters for COLMAP CLI ===
left_params_str = f"{left_K[0,0]},{left_K[1,1]},{left_K[0,2]},{left_K[1,2]}"
right_params_str = f"{right_K[0,0]},{right_K[1,1]},{right_K[0,2]},{right_K[1,2]}"
with open(left_params_txt, "w") as f:
    f.write(left_params_str)
with open(right_params_txt, "w") as f:
    f.write(right_params_str)

print("Left camera params:", left_params_str)
print("Right camera params:", right_params_str)

# === Clean dense image output folder ===
if os.path.exists(dense_images_dir):
    shutil.rmtree(dense_images_dir)
os.makedirs(dense_images_dir, exist_ok=True)

# === Get image size from example frame ===
left_images = [f for f in os.listdir(left_images_folder) if f.endswith(".png")]
if not left_images:
    raise FileNotFoundError(f"No images found in {left_images_folder}")
example_img_path = os.path.join(left_images_folder, left_images[0])
img = cv2.imread(example_img_path)
if img is None:
    raise FileNotFoundError(f"Cannot load example image: {example_img_path}")
img_h, img_w = img.shape[:2]
print(f"Image size: width={img_w}, height={img_h}")

# === Write cameras.txt with ONE CAMERA ===
with open(cameras_txt_path, "w") as f:
    f.write("# Camera list with one line of data per camera:\n")
    f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
    f.write(
        f"1 PINHOLE {img_w} {img_h} {left_K[0,0]} {left_K[1,1]} {left_K[0,2]} {left_K[1,2]}\n"
    )


# === Pose conversion helpers ===
def euler_to_matrix(rx, ry, rz, tx, ty, tz):
    r = Rotation.from_euler("xyz", [rx, ry, rz], degrees=True)
    R = r.as_matrix()
    if np.linalg.det(R) < 0:
        R = -R
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def matrix_to_quat_T(T):
    R = T[:3, :3]
    if np.linalg.det(R) < 0:
        R = -R
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    t = T[:3, 3]
    q = Rotation.from_matrix(R).as_quat()  # [x, y, z, w]
    q = np.roll(q, 1)  # [w, x, y, z]
    return q, t


# === Load camera poses (dictionary with id keys) ===
with open(camera_poses_json, "r") as f:
    camera_poses = json.load(f)
print(f"Loaded {len(camera_poses)} camera poses from {camera_poses_json}")

# === Validate images ===
image_count = 0
for id in camera_poses:
    for folder, suffix in [(left_images_folder, ".png"), (right_images_folder, ".png")]:
        img_path = os.path.join(folder, f"{id}{suffix}")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing image: {img_path}")
        img_check = cv2.imread(img_path)
        if img_check.shape[:2] != (img_h, img_w):
            raise ValueError(
                f"Image {img_path} has unexpected size {img_check.shape[:2]}"
            )
        image_count += 1
print(f"Total images validated: {image_count}")

# === Write images.txt with CAMERA_ID = 1 for ALL ===
with open(images_txt_path, "w") as f:
    f.write("# Image list with two lines of data per image:\n")
    f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
    f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")

    for idx, id in enumerate(sorted(camera_poses.keys(), key=lambda x: int(x))):
        pose = camera_poses[id]
        rx, ry, rz = pose["tcp_pose"][:3]
        tx, ty, tz = pose["tcp_pose"][3:]
        T_ee2world = euler_to_matrix(rx, ry, rz, tx, ty, tz)
        T_ee2world[:3, 2] = -T_ee2world[:3, 2]

        # === Left camera
        T_left2world = T_ee2world @ T_cam2ee
        q_left, t_left = matrix_to_quat_T(T_left2world)
        left_filename = f"{id}.png"
        left_filename = os.path.join("left", os.path.basename(left_filename))
        f.write(
            f"{idx*2+1} {q_left[0]} {q_left[1]} {q_left[2]} {q_left[3]} "
            f"{t_left[0]} {t_left[1]} {t_left[2]} 1 {left_filename}\n\n"
        )

        # === Right camera (same CAMERA_ID = 1)
        R_left = T_left2world[:3, :3]
        t_left_vec = T_left2world[:3, 3]
        R_right = R_left @ R_rel
        t_right_vec = t_left_vec + R_left @ T_rel[:, 0]
        T_right2world = np.eye(4)
        T_right2world[:3, :3] = R_right
        T_right2world[:3, 3] = t_right_vec
        q_right, t_right = matrix_to_quat_T(T_right2world)
        right_filename = f"{id}.png"
        right_filename = os.path.join("right", os.path.basename(right_filename))
        f.write(
            f"{idx*2+2} {q_right[0]} {q_right[1]} {q_right[2]} {q_right[3]} "
            f"{t_right[0]} {t_right[1]} {t_right[2]} 1 {right_filename}\n\n"
        )

# # === Empty points3D.txt ===
# open(points3d_txt_path, "w").close()
# print("COLMAP input files generated.")
