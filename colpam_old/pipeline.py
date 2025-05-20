#!/usr/bin/env python3

import os
import json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from pathlib import Path
import open3d as o3d
import subprocess
import sys

# Configuration
PROJECT_DIR = Path("/home/sha/Documents/work/robohand_v2/colmap/data")
CAMERA_POSES_FILE = PROJECT_DIR / "camera_poses.json"
HAND_RESULTS_FILE = PROJECT_DIR / "hand_results.json"
CALIBRATION_FILE = PROJECT_DIR / "stereo_calibration.xml"
IMAGES_DIR = PROJECT_DIR / "images"
INPUT_IMAGES = IMAGES_DIR / "input_imgs"
LEFT_IMAGES = IMAGES_DIR / "left"
RIGHT_IMAGES = IMAGES_DIR / "right"
DATABASE_PATH = PROJECT_DIR / "database.db"
SPARSE_PATH = PROJECT_DIR / "sparse"
DENSE_PATH = PROJECT_DIR / "dense"
POINT_CLOUD_PATH = PROJECT_DIR / "point_cloud.ply"
CAMERAS_TXT = PROJECT_DIR / "cameras.txt"
IMAGES_TXT = PROJECT_DIR / "images.txt"
POINTS3D_TXT = PROJECT_DIR / "points3D.txt"

# Ensure directories exist
for path in [LEFT_IMAGES, RIGHT_IMAGES, SPARSE_PATH, DENSE_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# Helper Functions
def parse_hand_results(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    for method_data in data.get("results", []):
        if method_data.get("method") == "Park":
            T_cam2ee = np.array(method_data["T_cam2ee"])
            mean_error = method_data["mean_translation_error_mm"]
            return T_cam2ee, mean_error
    raise ValueError("Park method not found in hand calibration results")

def euler_to_matrix(rx, ry, rz, tx, ty, tz):
    r = Rotation.from_euler("xyz", [rx, ry, rz], degrees=True)
    R_mat = r.as_matrix()
    if np.linalg.det(R_mat) < 0:
        R_mat = -R_mat
    T_mat = np.eye(4)
    T_mat[:3, :3] = R_mat
    T_mat[:3, 3] = [tx, ty, tz]
    return T_mat

def matrix_to_quaternion_and_translation(T_mat):
    R_mat = T_mat[:3, :3]
    if np.linalg.det(R_mat) < 0:
        R_mat = -R_mat
    if not np.allclose(R_mat @ R_mat.T, np.eye(3), atol=1e-6):
        U, _, Vt = np.linalg.svd(R_mat)
        R_mat = U @ Vt
    T_vec = T_mat[:3, 3]
    r = Rotation.from_matrix(R_mat)
    quat = r.as_quat()  # [x, y, z, w]
    return np.roll(quat, 1), T_vec  # [w, x, y, z]

def split_images():
    for image_file in INPUT_IMAGES.glob("*.png"):
        img = cv2.imread(str(image_file))
        if img is None:
            print(f"Failed to read {image_file}")
            continue
        h, w = img.shape[:2]
        left_img = img[:, :w//2]
        right_img = img[:, w//2:]
        base_name = image_file.stem
        left_path = LEFT_IMAGES / f"{base_name}_left.png"
        right_path = RIGHT_IMAGES / f"{base_name}_right.png"
        cv2.imwrite(str(left_path), left_img)
        cv2.imwrite(str(right_path), right_img)
        print(f"Saved {base_name}_left.png and {base_name}_right.png")

def write_colmap_files(camera_poses, left_camera_matrix, right_camera_matrix, T_ee2cam, R_rel, T_rel, image_width, image_height):
    # Write cameras.txt
    with open(CAMERAS_TXT, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(
            f"1 PINHOLE {image_width} {image_height} {left_camera_matrix[0, 0]} {left_camera_matrix[1, 1]} "
            f"{left_camera_matrix[0, 2]} {left_camera_matrix[1, 2]}\n"
        )
        f.write(
            f"2 PINHOLE {image_width} {image_height} {right_camera_matrix[0, 0]} {right_camera_matrix[1, 1]} "
            f"{right_camera_matrix[0, 2]} {right_camera_matrix[1, 2]}\n"
        )

    # Write images.txt
    with open(IMAGES_TXT, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for idx, item in enumerate(camera_poses):
            base_name = item["frame"][:-4]
            left_image = f"left/{base_name}_left.png"
            right_image = f"right/{base_name}_right.png"

            # Left camera pose
            world_pos = np.array(item["camera_pose"]["world_position"])
            world_ori = np.array(item["camera_pose"]["world_orientation"])
            T_ee2world = euler_to_matrix(*world_ori, *world_pos)
            T_left2world = T_ee2cam @ T_ee2world
            left_quat, left_tvec = matrix_to_quaternion_and_translation(T_left2world)
            f.write(
                f"{idx*2+1} {left_quat[0]} {left_quat[1]} {left_quat[2]} {left_quat[3]} "
                f"{left_tvec[0]} {left_tvec[1]} {left_tvec[2]} 1 {left_image}\n"
            )
            f.write("\n")

            # Right camera pose
            left_R = T_left2world[:3, :3]
            left_T = T_left2world[:3, 3]
            right_R = left_R @ R_rel
            right_T = left_T + (left_R @ T_rel.flatten())
            T_right2world = np.vstack((np.hstack((right_R, right_T[:, np.newaxis])), [0, 0, 0, 1]))
            right_quat, right_tvec = matrix_to_quaternion_and_translation(T_right2world)
            f.write(
                f"{idx*2+2} {right_quat[0]} {right_quat[1]} {right_quat[2]} {right_quat[3]} "
                f"{right_tvec[0]} {left_tvec[1]} {right_tvec[2]} 2 {right_image}\n"
            )
            f.write("\n")

    # Write empty points3D.txt
    open(POINTS3D_TXT, "w").close()

def run_colmap_command(args):
    env = os.environ.copy()
    env["QT_LOGGING_RULES"] = "qt5ct.debug=false"
    env["QT_QPA_PLATFORM"] = "offscreen"
    
    cmd = ["colmap"] + args
    print(f"Running COLMAP command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)
        print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running COLMAP command: {' '.join(cmd)}")
        print(e.stderr)
        raise
    except FileNotFoundError:
        print("COLMAP executable not found. Ensure COLMAP is installed and in PATH.")
        sys.exit(1)

# Main Pipeline
def main():
    # Check for display availability
    if "DISPLAY" not in os.environ and os.environ.get("XDG_SESSION_TYPE") != "wayland":
        print("Warning: No display detected. Using QT_QPA_PLATFORM=offscreen to bypass GUI requirements.")

    # Step 1: Load Calibration and Poses
    fs = cv2.FileStorage(str(CALIBRATION_FILE), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Failed to open calibration file: {CALIBRATION_FILE}")
    left_camera_matrix = fs.getNode("left_camera_matrix").mat()
    right_camera_matrix = fs.getNode("right_camera_matrix").mat()
    R_rel = fs.getNode("R").mat()
    T_rel = fs.getNode("T").mat()
    fs.release()

    T_cam2ee, park_error = parse_hand_results(HAND_RESULTS_FILE)
    T_ee2cam = np.linalg.inv(T_cam2ee)

    with open(CAMERA_POSES_FILE, "r") as f:
        camera_poses = json.load(f)

    # Get image dimensions
    example_image = next(LEFT_IMAGES.glob("*.png"), None)
    if example_image:
        image = cv2.imread(str(example_image))
        image_height, image_width = image.shape[:2]
    else:
        print("No images found in left folder, splitting images first...")
        split_images()
        example_image = next(LEFT_IMAGES.glob("*.png"), None)
        if not example_image:
            raise FileNotFoundError("No images found after splitting")
        image = cv2.imread(str(example_image))
        image_height, image_width = image.shape[:2]

    # Step 2: Split Images
    print("Splitting images...")
    split_images()

    # Step 3: Write COLMAP Input Files
    print("Generating COLMAP input files...")
    write_colmap_files(camera_poses, left_camera_matrix, right_camera_matrix, T_ee2cam, R_rel, T_rel, image_width, image_height)

    # Step 4: Feature Extraction for Left Camera
    print("Extracting features for left camera...")
    left_params = f"{left_camera_matrix[0,0]},{left_camera_matrix[1,1]},{left_camera_matrix[0,2]},{left_camera_matrix[1,2]}"
    run_colmap_command([
        "feature_extractor",
        "--database_path", str(DATABASE_PATH),
        "--image_path", str(LEFT_IMAGES),
        "--ImageReader.camera_model", "PINHOLE",
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_params", left_params,
        "--SiftExtraction.peak_threshold", "0.01",
        "--SiftExtraction.first_octave", "0",
        "--SiftExtraction.max_image_size", "7000",
        "--General.verbosity", "5"  # Increased verbosity for debugging
    ])

    # Step 5: Feature Extraction for Right Camera
    print("Extracting features for right camera...")
    right_params = f"{right_camera_matrix[0,0]},{right_camera_matrix[1,1]},{right_camera_matrix[0,2]},{right_camera_matrix[1,2]}"
    run_colmap_command([
        "feature_extractor",
        "--database_path", str(DATABASE_PATH),
        "--image_path", str(RIGHT_IMAGES),
        "--ImageReader.camera_model", "PINHOLE",
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_params", right_params,
        "--SiftExtraction.peak_threshold", "0.01",
        "--SiftExtraction.first_octave", "0",
        "--SiftExtraction.max_image_size", "7000",
        "--General.verbosity", "5"
    ])

    # Step 6: Feature Matching
    print("Matching features...")
    run_colmap_command([
        "exhaustive_matcher",
        "--database_path", str(DATABASE_PATH),
        "--SiftMatching.guided_matching", "1",
        "--SiftMatching.max_num_matches", "32768",
        "--General.verbosity", "5"
    ])

    # Step 7: Sparse Reconstruction
    print("Running sparse reconstruction...")
    run_colmap_command([
        "point_triangulator",
        "--database_path", str(DATABASE_PATH),
        "--image_path", str(IMAGES_DIR),
        "--input_path", str(PROJECT_DIR),
        "--output_path", str(SPARSE_PATH),
        "--Mapper.tri_ignore_two_view_tracks", "0",
        "--General.verbosity", "5"
    ])

    # Step 8: Convert to Point Cloud
    print("Converting to point cloud...")
    run_colmap_command([
        "model_converter",
        "--input_path", str(SPARSE_PATH),
        "--output_path", str(POINT_CLOUD_PATH),
        "--output_type", "PLY",
        "--General.verbosity", "5"
    ])

    # Step 9: Dense Reconstruction (Optional)
    print("Running dense reconstruction...")
    run_colmap_command([
        "image_undistorter",
        "--image_path", str(IMAGES_DIR),
        "--input_path", str(SPARSE_PATH),
        "--output_path", str(DENSE_PATH),
        "--General.verbosity", "5"
    ])

    run_colmap_command([
        "patch_match_stereo",
        "--workspace_path", str(DENSE_PATH),
        "--PatchMatchStereo.max_image_size", "4096",
        "--PatchMatchStereo.window_radius", "5",
        "--General.verbosity", "5"
    ])

    run_colmap_command([
        "stereo_fusion",
        "--workspace_path", str(DENSE_PATH),
        "--output_path", str(DENSE_PATH / "fused.ply"),
        "--StereoFusion.max_image_size", "4096",
        "--General.verbosity", "5"
    ])

    # Step 10: Visualization
    print("Visualizing point cloud...")
    pcd = o3d.io.read_point_cloud(str(POINT_CLOUD_PATH))
    o3d.visualization.draw_geometries([pcd])

    print("COLMAP stereo reconstruction complete.")

if __name__ == "__main__":
    main()