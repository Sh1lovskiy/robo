"""
Stereo camera calibration module using ChArUco board for both stereo and hand-eye calibration.
Performs sequential stereo calibration followed by hand-eye calibration, saving minimal required parameters.
Uses a ChArUco board with specified dimensions and dictionary.
"""

import cv2
import numpy as np
import glob
import os
import json
import argparse
from pathlib import Path
from calib.logger import Logger, Timer

# --- Logger ---
logger = Logger.get_logger("StereoCharucoCalibration")

# --- Constants and Variables ---
CHARUCO_SQUARE_LENGTH = 35.0  # Checker Width [mm]
CHARUCO_MARKER_LENGTH = 26.0  # Marker Size [mm]
CHARUCO_BOARD_SIZE = (7, 5)  # Columns, Rows
CHARUCO_DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
CAMERA_IMAGE_EXTENSIONS = (".jpg", ".png")
CALIBRATION_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
MIN_CALIBRATION_IMAGES = 5
DEFAULT_IMAGES_FOLDER = "calib_hand/images"
DEFAULT_STEREO_OUTPUT = "calib_hand/ouput/stereo_calib.xml"
DEFAULT_HAND_EYE_OUTPUT = "calib_hand/output/hand_results.txt"


def matrix_from_rtvec(rvec, tvec):
    """Convert rotation vector and translation vector to 4x4 transformation matrix."""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.ravel()
    return T


@Logger.silent_log_function(logger)
def create_charuco_board(square_length, marker_length, board_size):
    """Create ChArUco board with specified dimensions and dictionary."""
    board = cv2.aruco.CharucoBoard(
        board_size,
        square_length,
        marker_length,
        CHARUCO_DICTIONARY,
    )
    return board, CHARUCO_DICTIONARY


@Logger.silent_log_function(logger)
def calibrate_stereo_camera(images_folder, board, dictionary):
    """Calibrate stereo camera using left_*.png and right_*.png pairs."""
    objpoints = []
    imgpoints_left = []
    imgpoints_right = []
    image_shape = None

    left_files = sorted(glob.glob(os.path.join(images_folder, "left_*.png")))
    right_files = sorted(glob.glob(os.path.join(images_folder, "right_*.png")))

    if not left_files or not right_files:
        raise ValueError(f"No left/right images found in {images_folder}")
    if len(left_files) != len(right_files):
        raise ValueError("Mismatch in number of left and right images")

    logger.info(f"Found {len(left_files)} stereo image pairs")

    with Timer("CharucoDetection", logger):
        for l_file, r_file in zip(left_files, right_files):
            img_left = cv2.imread(l_file)
            img_right = cv2.imread(r_file)

            if img_left is None or img_right is None:
                logger.warning(f"Failed to load image pair: {l_file} / {r_file}, skipping")
                continue

            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

            if image_shape is None:
                h, w = gray_left.shape
                image_shape = (w, h)

            corners_left, ids_left, _ = cv2.aruco.detectMarkers(gray_left, dictionary)
            corners_right, ids_right, _ = cv2.aruco.detectMarkers(gray_right, dictionary)

            if ids_left is not None and ids_right is not None:
                ret_left, charuco_corners_left, charuco_ids_left = (
                    cv2.aruco.interpolateCornersCharuco(
                        corners_left, ids_left, gray_left, board
                    )
                )
                ret_right, charuco_corners_right, charuco_ids_right = (
                    cv2.aruco.interpolateCornersCharuco(
                        corners_right, ids_right, gray_right, board
                    )
                )

                if ret_left > 0 and ret_right > 0:
                    all_board_corners = board.getChessboardCorners()
                    if len(charuco_corners_left) == len(all_board_corners) and len(charuco_corners_right) == len(all_board_corners):
                        objpoints.append(all_board_corners)
                        imgpoints_left.append(charuco_corners_left)
                        imgpoints_right.append(charuco_corners_right)
                    else:
                        logger.warning(f"Corner count mismatch in pair: {l_file} / {r_file}")
                else:
                    logger.warning(f"Could not find enough ChArUco corners in pair: {l_file} / {r_file}")
            else:
                logger.warning(f"Could not detect ArUco markers in pair: {l_file} / {r_file}")

    logger.info(f"Successfully detected patterns in {len(objpoints)} image pairs")

    if len(objpoints) < MIN_CALIBRATION_IMAGES:
        raise ValueError(
            f"Not enough valid calibration image pairs (need at least {MIN_CALIBRATION_IMAGES})"
        )

    logger.info("Calibrating left camera...")
    ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
        objpoints, imgpoints_left, gray_left.shape[::-1], None, None
    )
    logger.info("Calibrating right camera...")
    ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
        objpoints, imgpoints_right, gray_right.shape[::-1], None, None
    )

    logger.info("Performing stereo calibration...")
    ret, mtx_left, dist_left, mtx_right, dist_right, R, T, _, _ = cv2.stereoCalibrate(
        objpoints,
        imgpoints_left,
        imgpoints_right,
        mtx_left, dist_left,
        mtx_right, dist_right,
        gray_left.shape[::-1],
        criteria=CALIBRATION_CRITERIA,
        flags=0
    )

    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
        mtx_left, dist_left, mtx_right, dist_right, gray_left.shape[::-1], R, T
    )

    map_left_x, map_left_y = cv2.initUndistortRectifyMap(
        mtx_left, dist_left, R1, P1, gray_left.shape[::-1], cv2.CV_32FC1
    )
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(
        mtx_right, dist_right, R2, P2, gray_right.shape[::-1], cv2.CV_32FC1
    )

    calibration_result = {
        "left_camera_matrix": mtx_left,
        "left_distortion": dist_left,
        "right_camera_matrix": mtx_right,
        "right_distortion": dist_right,
        "R1": R1,
        "R2": R2,
        "P1": P1,
        "P2": P2,
        "Q": Q,
        "map_left_x": map_left_x,
        "map_left_y": map_left_y,
        "map_right_x": map_right_x,
        "map_right_y": map_right_y,
        "roi_left": roi_left,
        "roi_right": roi_right,
        "image_shape": image_shape,
    }

    logger.info(f"Stereo calibration completed with RMS error: {ret}")
    return calibration_result


@Logger.silent_log_function(logger)
def save_calibration(calibration_result, output_file):
    """Save minimal calibration parameters to a file."""
    logger.info(f"Saving calibration parameters to {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    fs = cv2.FileStorage(output_file, cv2.FILE_STORAGE_WRITE)
    for key, value in calibration_result.items():
        fs.write(key, value)
    fs.release()
    logger.info("Calibration parameters saved successfully")


@Logger.silent_log_function(logger)
def calibrate_hand_eye(calib_dir, stereo_xml, board, result_txt):
    """Perform hand-eye calibration using ChArUco board and data from poses.json."""
    # Load stereo calibration
    fs = cv2.FileStorage(stereo_xml, cv2.FILE_STORAGE_READ)
    K_left = fs.getNode("left_camera_matrix").mat()
    D_left = fs.getNode("left_distortion").mat()
    fs.release()

    # Load robot poses from poses.json
    pose_file = os.path.join(calib_dir, "poses.json")
    if not os.path.exists(pose_file):
        raise FileNotFoundError(f"Pose file {pose_file} not found")

    with open(pose_file, "r") as f:
        all_poses = json.load(f)

    logger.info(f"Loaded {len(all_poses)} poses from JSON")

    hand_poses = []
    cam_poses = []

    with Timer("HandEyeProcessing", logger):
        for key in sorted(all_poses.keys()):
            idx = key.split("_")[2]
            left_img_path = os.path.join(calib_dir, f"left_{idx}.png")
            right_img_path = os.path.join(calib_dir, f"right_{idx}.png")

            if not (os.path.exists(left_img_path) and os.path.exists(right_img_path)):
                logger.warning(f"Missing image pair for {key}, skipping")
                continue

            left_img = cv2.imread(left_img_path)
            gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)

            corners, ids, _ = cv2.aruco.detectMarkers(gray, CHARUCO_DICTIONARY)
            if ids is None:
                logger.warning(f"No ArUco markers detected in {left_img_path}, skipping")
                continue

            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board
            )
            if ret < 4:
                logger.warning(f"Insufficient ChArUco corners in {left_img_path}, skipping")
                continue

            success, rvec, tvec = cv2.solvePnP(
                board.getChessboardCorners(), charuco_corners, K_left, D_left
            )
            if not success:
                logger.warning(f"SolvePnP failed for {left_img_path}, skipping")
                continue

            try:
                robot_pose = np.array(all_poses[key]["robot_tcp_pose"])
                T_gripper2base = matrix_from_rtvec(
                    np.deg2rad(robot_pose[3:]), robot_pose[:3]
                )
                hand_poses.append(T_gripper2base)
                cam_poses.append(matrix_from_rtvec(rvec, tvec))
            except KeyError as e:
                logger.warning(f"Missing robot pose data in {key}: {e}")
                continue

    logger.info(f"Successfully processed {len(hand_poses)} image-pose pairs")

    if len(hand_poses) < 3:
        logger.warning(f"Not enough measurements ({len(hand_poses)}) for hand-eye calibration")
        return

    # Extract R and t for calibration
    def extract_rt(pose_list):
        R_list = [T[:3, :3] for T in pose_list]
        t_list = [T[:3, 3] for T in pose_list]
        return R_list, t_list

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
    with Timer("HandEyeCalibration", logger), open(result_txt, "w") as f:
        for name, method in methods.items():
            R_eye2hand, t_eye2hand = cv2.calibrateHandEye(
                R_base2gripper, t_base2gripper,
                R_target2cam, t_target2cam,
                method=method
            )

            T_eye2hand = np.eye(4)
            T_eye2hand[:3, :3] = R_eye2hand
            T_eye2hand[:3, 3] = t_eye2hand.ravel()

            # Compute error
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
                f"Completed calibration with {name} method, mean error: {mean_error:.4f} mm"
            )

    logger.info(f"Hand-eye calibration results saved successfully")


def main():
    # Local constants for argparse defaults
    DEFAULT_SQUARE_LENGTH = 35.0
    DEFAULT_MARKER_LENGTH = 26.0
    DEFAULT_PATTERN_WIDTH = 5
    DEFAULT_PATTERN_HEIGHT = 7

    parser = argparse.ArgumentParser(
        description="Stereo and hand-eye calibration with ChArUco board"
    )
    parser.add_argument(
        "--images",
        default=DEFAULT_IMAGES_FOLDER,
        help="Path to folder containing calibration images",
    )
    parser.add_argument(
        "--calib_dir",
        default=DEFAULT_IMAGES_FOLDER,
        help="Path to folder containing hand-eye calibration data",
    )
    parser.add_argument(
        "--stereo_output",
        default=DEFAULT_STEREO_OUTPUT,
        help="Output stereo calibration file",
    )
    parser.add_argument(
        "--hand_eye_output",
        default=DEFAULT_HAND_EYE_OUTPUT,
        help="Output hand-eye calibration results",
    )
    parser.add_argument(
        "--pattern_width",
        type=int,
        default=DEFAULT_PATTERN_WIDTH,
        help="ChArUco board columns",
    )
    parser.add_argument(
        "--pattern_height",
        type=int,
        default=DEFAULT_PATTERN_HEIGHT,
        help="ChArUco board rows",
    )
    parser.add_argument(
        "--square_size",
        type=float,
        default=DEFAULT_SQUARE_LENGTH,
        help="Size of ChArUco squares in mm",
    )
    parser.add_argument(
        "--marker_size",
        type=float,
        default=DEFAULT_MARKER_LENGTH,
        help="Size of ArUco markers in mm",
    )

    args = parser.parse_args()

    # Update global variables
    global CHARUCO_SQUARE_LENGTH, CHARUCO_MARKER_LENGTH, CHARUCO_BOARD_SIZE
    CHARUCO_SQUARE_LENGTH = args.square_size
    CHARUCO_MARKER_LENGTH = args.marker_size
    CHARUCO_BOARD_SIZE = (args.pattern_width, args.pattern_height)

    logger.info(
        f"Starting calibration with ChArUco board "
        f"size {CHARUCO_BOARD_SIZE}, square size {args.square_size} mm, marker size {args.marker_size} mm"
    )

    with Timer("FullCalibration", logger):
        # Create ChArUco board
        board, dictionary = create_charuco_board(
            CHARUCO_SQUARE_LENGTH, CHARUCO_MARKER_LENGTH, CHARUCO_BOARD_SIZE
        )

        # Stereo calibration
        calibration_result = calibrate_stereo_camera(args.images, board, dictionary)
        save_calibration(calibration_result, args.stereo_output)

        # Hand-eye calibration
        calibrate_hand_eye(
            args.calib_dir, args.stereo_output, board, args.hand_eye_output
        )

    logger.info("Calibration pipeline completed successfully")


if __name__ == "__main__":
    main()
