"""
Hand-eye calibration module for robotic systems.
Performs eye-in-hand calibration using a chessboard pattern to compute the transformation
between the camera and the robot's gripper. Loads stereo calibration parameters and
processes image-pose pairs to estimate the camera-to-gripper transformation using multiple
calibration methods. Includes logging for key steps and timing measurements.
"""

import os
import glob
import json
import cv2
import numpy as np
from pathlib import Path
from calib.utils import matrix_from_rtvec
from misc.logger import Logger, Timer

# --- Logger ---
logger = Logger.get_logger("HandEyeCalibration")

# --- Settings ---
CALIB_DIR = (
    "/home/sha/Documents/work/robohand_v2/calib_hand/world_calib/20250430_150219"
)
STEREO_XML = (
    "/home/sha/Documents/work/robohand_v2/calib_hand/output/stereo_calibration.xml"
)
RESULT_TXT = (
    "/home/sha/Documents/work/robohand_v2/calib_hand/output/eye_in_hand_results.txt"
)
CHESSBOARD_SIZE = (9, 6)  # Number of inner corners (columns, rows)
SQUARE_SIZE = 23.0  # mm
CALIBRATION_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


@Logger.silent_log_function(logger)
def load_stereo_calibration(stereo_xml):
    """Load stereo calibration parameters from XML file."""
    logger.info(f"Loading stereo calibration from {stereo_xml}")
    fs = cv2.FileStorage(stereo_xml, cv2.FILE_STORAGE_READ)
    K_left = fs.getNode("left_camera_matrix").mat()
    D_left = fs.getNode("left_distortion").mat()
    fs.release()
    return K_left, D_left


@Logger.silent_log_function(logger)
def prepare_object_points(chessboard_size, square_size):
    """Prepare 3D object points for chessboard pattern."""
    objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
    objp[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)
    objp *= square_size
    return objp


@Logger.silent_log_function(logger)
def process_image_pose_pairs(calib_dir, chessboard_size, K_left, D_left, objp):
    """Process image-pose pairs to extract hand and camera poses."""
    image_files = sorted(glob.glob(os.path.join(calib_dir, "frame_*.png")))
    hand_poses = []  # Robot poses (gripper wrt base)
    cam_poses = []  # Camera poses (chessboard wrt camera)

    logger.info(f"Found {len(image_files)} images for processing")

    with Timer("ImagePoseProcessing", logger):
        for img_path in image_files:
            idx = Path(img_path).stem.split("_")[-1]
            pose_path_json = os.path.join(calib_dir, f"pose_{idx}.json")
            pose_path_txt = os.path.join(calib_dir, f"pose_{idx}.txt")

            # Load and split stereo image
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Failed to load image {img_path}, skipping")
                continue

            h, w = img.shape[:2]
            left_img = img[:, : w // 2]

            # Find chessboard corners
            gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            if not ret:
                logger.warning(f"Chessboard not found in {img_path}, skipping")
                continue

            # Refine corners
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria=CALIBRATION_CRITERIA
            )

            # SolvePnP: find board pose wrt camera
            success, rvec, tvec = cv2.solvePnP(objp, corners, K_left, D_left)
            if not success:
                logger.warning(f"SolvePnP failed for {img_path}, skipping")
                continue

            # Load robot pose (eye-in-hand = gripper wrt base)
            if os.path.exists(pose_path_json):
                with open(pose_path_json, "r") as f:
                    pose_data = json.load(f)["robot_pose"]
            elif os.path.exists(pose_path_txt):
                with open(pose_path_txt, "r") as f:
                    for line in f:
                        if "Robot pose:" in line:
                            pose_data = eval(line.strip().split(":", 1)[1])
                            break
                    else:
                        logger.warning(
                            f"Robot pose not found in {pose_path_txt}, skipping"
                        )
                        continue
            else:
                logger.warning(f"Pose file not found for frame {idx}, skipping")
                continue

            T_gripper2base = matrix_from_rtvec(
                np.deg2rad(pose_data[3:]), np.array(pose_data[:3])
            )

            # Save poses
            hand_poses.append(T_gripper2base)
            cam_poses.append(matrix_from_rtvec(rvec, tvec))

    logger.info(f"Successfully processed {len(hand_poses)} image-pose pairs")
    return hand_poses, cam_poses


def extract_rt(pose_list):
    """Extract rotation and translation components from a list of transformation matrices."""
    R_list = [T[:3, :3] for T in pose_list]
    t_list = [T[:3, 3] for T in pose_list]
    return R_list, t_list


@Logger.silent_log_function(logger)
def perform_hand_eye_calibration(hand_poses, cam_poses, result_txt):
    """Perform hand-eye calibration using multiple methods and save results."""
    R_base2gripper, t_base2gripper = extract_rt(hand_poses)
    R_target2cam, t_target2cam = extract_rt(cam_poses)

    methods = {
        "Tsai": cv2.CALIB_HAND_EYE_TSAI,
        "Park": cv2.CALIB_HAND_EYE_PARK,
        "Horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "Andreff": cv2.CALIB_HAND_EYE_ANDREFF,
        "Daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }

    logger.info(f"Saving hand-eye calibration results to {result_txt}")
    with Timer("HandEyeCalibration", logger):
        with open(result_txt, "w") as f:
            for name, method in methods.items():
                R_eye2hand, t_eye2hand = cv2.calibrateHandEye(
                    R_base2gripper,
                    t_base2gripper,
                    R_target2cam,
                    t_target2cam,
                    method=method,
                )
                T_eye2hand = np.eye(4)
                T_eye2hand[:3, :3] = R_eye2hand
                T_eye2hand[:3, 3] = t_eye2hand.squeeze()

                # Compute error based on T_base2gripper * T_eye2hand = T_target2cam
                errors = []
                for T_b2g, T_t2c in zip(hand_poses, cam_poses):
                    T_pred = T_b2g @ T_eye2hand
                    delta_t = T_pred[:3, 3] - T_t2c[:3, 3]
                    errors.append(np.linalg.norm(delta_t))
                mean_error = np.mean(errors)

                # Save result
                f.write(f"Method: {name}\n")
                f.write(f"T_cam2ee (Eye to Hand):\n{T_eye2hand}\n")
                f.write(f"Mean translation error: {mean_error:.4f} mm\n\n")
                logger.info(
                    f"Completed {name} method, mean_err: {mean_error:.4f} mm"
                )

    logger.info(f"Hand-eye calibration results saved successfully")


def main():
    """Main function to execute hand-eye calibration pipeline."""
    logger.info("Starting hand-eye calibration pipeline")

    with Timer("FullHandEyeCalibration", logger):
        # Load stereo calibration
        K_left, D_left = load_stereo_calibration(STEREO_XML)

        # Prepare object points
        objp = prepare_object_points(CHESSBOARD_SIZE, SQUARE_SIZE)

        # Process image-pose pairs
        hand_poses, cam_poses = process_image_pose_pairs(
            CALIB_DIR, CHESSBOARD_SIZE, K_left, D_left, objp
        )

        # Perform hand-eye calibration
        perform_hand_eye_calibration(hand_poses, cam_poses, RESULT_TXT)

    logger.info("Hand-eye calibration pipeline completed successfully")


if __name__ == "__main__":
    main()
