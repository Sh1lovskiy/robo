"""
Charuco Stereo Calibration Module for RGB-Depth Camera Systems
"""

import os
import cv2
import yaml
import numpy as np
from typing import Optional, List, Tuple
from utils.logger import Logger


# 1. ARUCO DICTIONARY MAP


DICT_MAP = {
    "4X4_50": cv2.aruco.DICT_4X4_50,
    "4X4_100": cv2.aruco.DICT_4X4_100,
    "5X5_50": cv2.aruco.DICT_5X5_50,
    "5X5_100": cv2.aruco.DICT_5X5_100,
    "6X6_50": cv2.aruco.DICT_6X6_50,
    "6X6_100": cv2.aruco.DICT_6X6_100,
}


# 2. SAVING UTILITIES FOR CALIBRATION OUTPUT


def save_camera_params(filename, image_size, camera_matrix, dist_coeffs, total_avg_err):
    """
    Save camera intrinsic parameters and reprojection error in OpenCV YAML format.

    Args:
        filename: Output path (yml or yaml).
        image_size: (width, height) tuple, pixels.
        camera_matrix: 3x3 numpy array (intrinsics, OpenCV).
        dist_coeffs: distortion coefficients, shape Nx1 or N.
        total_avg_err: mean reprojection error.
    """
    calibration_data = {
        "image_width": image_size[0],
        "image_height": image_size[1],
        "camera_matrix": {
            "rows": camera_matrix.shape[0],
            "cols": camera_matrix.shape[1],
            "dt": "d",
            "data": camera_matrix.tolist(),
        },
        "distortion_coefficients": {
            "rows": dist_coeffs.shape[0],
            "cols": dist_coeffs.shape[1] if dist_coeffs.ndim > 1 else 1,
            "dt": "d",
            "data": dist_coeffs.flatten().tolist(),
        },
        "avg_reprojection_error": float(total_avg_err),
    }
    dir_ = os.path.dirname(filename)
    if dir_:
        os.makedirs(dir_, exist_ok=True)
    with open(filename, "w") as f:
        yaml.dump(calibration_data, f)
    print(f"Saved calibration to {filename}")


# 3. CHARUCO CALIBRATOR: MONOCULAR INTRINSICS


class CharucoCalibrator:
    """
    Performs intrinsic calibration of a single camera (RGB or Depth) using a Charuco board.

    Each image added must have visible, correctly detected Charuco corners.
    After adding frames, call `calibrate()` to estimate camera intrinsics.
    """

    def __init__(self, board, dictionary, logger=None):
        """
        Args:
            board: cv2.aruco.CharucoBoard object.
            dictionary: cv2.aruco_Dictionary instance.
            logger: optional Logger.
        """
        self.board = board
        self.dictionary = dictionary
        self.all_corners = []
        self.all_ids = []
        self.image_size = None
        self.logger = logger or Logger.get_logger("charuco_calibrator")

    def add_frame(self, img: np.ndarray) -> bool:
        """
        Detect Charuco corners and add for calibration.
        Args:
            img: BGR image (OpenCV).
        Returns:
            True if enough corners detected, else False.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.image_size = (gray.shape[1], gray.shape[0])
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary)
        if ids is None or len(corners) == 0:
            self.logger.warning("No ArUco markers detected.")
            return False
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, self.board
        )
        if retval < 10 or charuco_ids is None or len(charuco_ids) < 10:
            self.logger.warning(
                f"Not enough Charuco corners: {len(charuco_ids) if charuco_ids is not None else 0}"
            )
            return False
        self.all_corners.append(charuco_corners)
        self.all_ids.append(charuco_ids)
        return True

    def calibrate(self) -> dict:
        """
        Run OpenCV Charuco intrinsic calibration.

        Returns:
            Dictionary with keys: rms, camera_matrix, dist_coeffs, mean_rmse.
        """
        if len(self.all_corners) < 3:
            raise RuntimeError("Not enough valid frames for calibration.")
        flags = 0

        rms, camera_matrix, dist_coeffs, rvecs, tvecs = (
            cv2.aruco.calibrateCameraCharuco(
                charucoCorners=self.all_corners,
                charucoIds=self.all_ids,
                board=self.board,
                imageSize=self.image_size,
                cameraMatrix=None,
                distCoeffs=None,
                flags=flags,
                criteria=(
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    100,
                    1e-6,
                ),
            )
        )
        all_errors = []
        for i in range(len(rvecs)):
            img_points_reproj, _ = cv2.projectPoints(
                self.board.getChessboardCorners(),
                rvecs[i],
                tvecs[i],
                camera_matrix,
                dist_coeffs,
            )
            img_points_detected = self.all_corners[i]
            n = min(len(img_points_reproj), len(img_points_detected))
            diffs = img_points_reproj[:n].reshape(-1, 2) - img_points_detected[
                :n
            ].reshape(-1, 2)
            errs = np.linalg.norm(diffs, axis=1)
            all_errors.append(errs)
        if all_errors:
            all_errors = np.concatenate(all_errors)
            mean_rmse = float(np.sqrt(np.mean(all_errors**2)))
        else:
            mean_rmse = float("nan")
        return {
            "rms": rms,
            "camera_matrix": camera_matrix,
            "dist_coeffs": dist_coeffs,
            "mean_rmse": mean_rmse,
        }

    def save(self, saver, filename, camera_matrix, dist_coeffs):
        saver.save(filename, self.image_size, camera_matrix, dist_coeffs)


# 4. FILE SAVERS FOR OUTPUT


class OpenCVXmlSaver:
    """Saves camera parameters to OpenCV XML format."""

    def save(self, filename, image_size, camera_matrix, dist_coeffs):
        dir_ = os.path.dirname(filename)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
        if not fs.isOpened():
            raise IOError(f"Failed to open {filename} for writing.")
        fs.write("image_width", image_size[0])
        fs.write("image_height", image_size[1])
        fs.write("camera_matrix", camera_matrix)
        fs.write("distortion_coefficients", dist_coeffs)
        fs.release()
        print(f"Saved calibration to {filename} (OpenCV XML)")


class TextSaver:
    """Saves camera parameters to a plain text file."""

    def save(self, filename, image_size, camera_matrix, dist_coeffs):
        with open(filename, "w") as f:
            f.write(f"# Image size: {image_size}\n")
            f.write("camera_matrix:\n")
            for row in camera_matrix:
                f.write("  " + " ".join(f"{v:.8f}" for v in row) + "\n")
            f.write("distortion_coefficients:\n")
            f.write("  " + " ".join(f"{v:.8f}" for v in dist_coeffs.flatten()) + "\n")
        print(f"Saved calibration to {filename} (TXT)")


# 5. CHARUCO STEREO: CORRESPONDENCES FOR STEREO CALIBRATION (EXTRINSICS)


def find_stereo_correspondences(
    rgb_files: List[str],
    depth_files: List[str],
    board,
    dictionary,
    logger,
    min_points=10,
) -> Tuple[list, list, list, tuple]:
    """
    Find 3D-2D correspondences for stereo calibration.

    For each Charuco frame, detects corners in both RGB and depth images,
    matches by ID, and returns object/image point lists for stereo calibration.

    Returns:
        objpoints_all: List of 3D board points (world, meters)
        imgpoints_rgb: List of 2D points in RGB images
        imgpoints_depth: List of 2D points in depth images
        image_size: (width, height) of images (assumes same for both cameras)
    """
    objpoints_all = []
    imgpoints_rgb = []
    imgpoints_depth = []
    image_size = None

    for rgb_file, depth_file in zip(rgb_files, depth_files):
        if not os.path.exists(rgb_file) or not os.path.exists(depth_file):
            logger.warning(f"File(s) not found: {rgb_file} or {depth_file}")
            continue

        img_rgb = cv2.imread(rgb_file)
        img_depth = cv2.imread(
            depth_file
        )  # If depth is .png IR image (not .npy metric!)
        if img_rgb is None or img_depth is None:
            logger.warning(f"Failed to read {rgb_file} or {depth_file}")
            continue

        gray_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        gray_depth = cv2.cvtColor(img_depth, cv2.COLOR_BGR2GRAY)
        image_size = (gray_rgb.shape[1], gray_rgb.shape[0])

        # Detect markers in both images
        corners_rgb, ids_rgb, _ = cv2.aruco.detectMarkers(gray_rgb, dictionary)
        corners_depth, ids_depth, _ = cv2.aruco.detectMarkers(gray_depth, dictionary)
        if ids_rgb is None or ids_depth is None:
            logger.warning(f"ArUco not detected in {rgb_file} or {depth_file}")
            continue

        # Interpolate Charuco corners
        ret_rgb, charuco_corners_rgb, charuco_ids_rgb = (
            cv2.aruco.interpolateCornersCharuco(corners_rgb, ids_rgb, gray_rgb, board)
        )
        ret_depth, charuco_corners_depth, charuco_ids_depth = (
            cv2.aruco.interpolateCornersCharuco(
                corners_depth, ids_depth, gray_depth, board
            )
        )
        if (
            ret_rgb < min_points
            or ret_depth < min_points
            or charuco_ids_rgb is None
            or charuco_ids_depth is None
            or len(charuco_ids_rgb) < min_points
            or len(charuco_ids_depth) < min_points
        ):
            logger.warning(f"Not enough Charuco corners in {rgb_file} or {depth_file}")
            continue

        # Match corners by board ID (integer IDs assigned by OpenCV)
        # This gives (N,3) objpoints, (N,2) rgb image points, (N,2) depth image points
        board_corners_3d = board.getChessboardCorners()
        idx_map_rgb = {int(i): j for j, i in enumerate(charuco_ids_rgb.flatten())}
        idx_map_depth = {int(i): j for j, i in enumerate(charuco_ids_depth.flatten())}
        common_ids = sorted(set(idx_map_rgb.keys()) & set(idx_map_depth.keys()))
        if len(common_ids) < min_points:
            logger.warning(f"Too few matching Charuco IDs in {rgb_file}")
            continue

        matched_obj_points = []
        matched_rgb_points = []
        matched_depth_points = []
        for cid in common_ids:
            matched_obj_points.append(board_corners_3d[cid])
            matched_rgb_points.append(charuco_corners_rgb[idx_map_rgb[cid], 0, :])
            matched_depth_points.append(charuco_corners_depth[idx_map_depth[cid], 0, :])

        objpoints_all.append(np.array(matched_obj_points, dtype=np.float32))
        imgpoints_rgb.append(np.array(matched_rgb_points, dtype=np.float32))
        imgpoints_depth.append(np.array(matched_depth_points, dtype=np.float32))

    return objpoints_all, imgpoints_rgb, imgpoints_depth, image_size


# 6. STEREO CALIBRATION: RGB-DEPTH EXTRINSICS


def stereo_calibrate(
    objpoints: List[np.ndarray],
    imgpoints1: List[np.ndarray],
    imgpoints2: List[np.ndarray],
    K1: np.ndarray,
    dist1: np.ndarray,
    K2: np.ndarray,
    dist2: np.ndarray,
    image_size,
    logger=None,
) -> dict:
    """
    Perform stereo calibration to estimate the extrinsics (R, t) between two cameras.

    Returns:
        dict with fields: rms, R, T, E, F
        R: 3x3 rotation matrix (from cam2 to cam1)
        T: 3x1 translation (from cam2 to cam1, meters)
        E, F: essential, fundamental matrices (OpenCV)
    """
    logger = logger or Logger.get_logger("stereo_calib")
    flags = 0  # No fixed intrinsics, optimize all
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    ret, K1_out, dist1_out, K2_out, dist2_out, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints1,
        imgpoints2,
        K1,
        dist1,
        K2,
        dist2,
        image_size,
        criteria=criteria,
        flags=flags,
    )
    logger.info(f"Stereo calibration RMS: {ret:.6f}\n" f"R:\n{R}\nT (m): {T.ravel()}\n")
    return {
        "rms": ret,
        "R": R,
        "T": T,
        "K1": K1_out,
        "dist1": dist1_out,
        "K2": K2_out,
        "dist2": dist2_out,
        "E": E,
        "F": F,
    }


# 7. MAIN PIPELINE: FULL INTRINSICS + EXTRINSICS CALIBRATION


def run_calibration(mode: str = "both"):
    """
    Run the full calibration pipeline: intrinsics for RGB/depth, plus stereo extrinsics.
    Saves all outputs to calib/camera_intrs.

    Args:
        mode: "rgb", "depth", "stereo", or "both".
    """
    logger = Logger.get_logger("cam_calib")
    folder = "calib/imgs"
    output_dir = "calib/camera_intrs"
    os.makedirs(output_dir, exist_ok=True)
    squares_x = 8
    squares_y = 5
    square_length = 0.035
    marker_length = 0.026
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard(
        size=(squares_x, squares_y),
        squareLength=square_length,
        markerLength=marker_length,
        dictionary=dictionary,
    )

    # File sorting: expects "frame_xxx_color.png" and "frame_xxx_depth.png"
    files = sorted(
        [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(".png")
        ]
    )
    rgb_files = [f for f in files if "color" in f or "png" in f]
    depth_files = [f for f in files if "depth" in f or "npy" in f]
    logger.info(
        f"Found {len(rgb_files)} RGB and {len(depth_files)} depth images in {folder}"
    )

    # ---- Intrinsic calibration RGB ----
    K_rgb = dist_rgb = None
    K_depth = dist_depth = None

    if mode.lower() in ["rgb", "both"]:
        logger.info("==== RGB calibration ====")
        calibrator = CharucoCalibrator(board, dictionary, logger=logger)
        n_good = 0
        for fname in rgb_files:
            img = cv2.imread(fname)
            if img is None:
                logger.warning(f"Failed to read {fname}")
                continue
            if calibrator.add_frame(img):
                n_good += 1
                logger.info(f"Frame added: {fname}")
            else:
                logger.warning(f"Frame rejected: {fname}")
        if n_good == 0:
            logger.error("No valid charuco frames found. RGB calibration aborted.")
        else:
            result = calibrator.calibrate()
            xml_file = os.path.join(output_dir, "charuco_cam.xml")
            txt_file = os.path.join(output_dir, "charuco_cam.txt")
            yaml_file = os.path.join(output_dir, "charuco_cam.yml")
            calibrator.save(
                OpenCVXmlSaver(),
                xml_file,
                result["camera_matrix"],
                result["dist_coeffs"],
            )
            calibrator.save(
                TextSaver(), txt_file, result["camera_matrix"], result["dist_coeffs"]
            )
            save_camera_params(
                yaml_file,
                calibrator.image_size,
                result["camera_matrix"],
                result["dist_coeffs"],
                result["rms"],
            )
            logger.info(
                f"Charuco RGB calibration complete. RMS: {result['rms']:.6f}  RMSE: {result['mean_rmse']:.6f}"
            )
            K_rgb, dist_rgb = result["camera_matrix"], result["dist_coeffs"]
    else:
        K_rgb = dist_rgb = None

    # ---- Intrinsic calibration Depth ----
    if mode.lower() in ["depth", "both"]:
        logger.info("==== Depth calibration ====")
        calibrator = CharucoCalibrator(board, dictionary, logger=logger)
        n_good = 0
        for fname in depth_files:
            img = np.load(fname)
            if img is None:
                logger.warning(f"Failed to read {fname}")
                continue
            if calibrator.add_frame(img):
                n_good += 1
                logger.info(f"Frame added: {fname}")
            else:
                logger.warning(f"Frame rejected: {fname}")
        if n_good == 0:
            logger.error("No valid charuco frames found. Depth calibration aborted.")
        else:
            result = calibrator.calibrate()
            xml_file = os.path.join(output_dir, "depth_cam.xml")
            txt_file = os.path.join(output_dir, "depth_cam.txt")
            yaml_file = os.path.join(output_dir, "depth_cam.yml")
            calibrator.save(
                OpenCVXmlSaver(),
                xml_file,
                result["camera_matrix"],
                result["dist_coeffs"],
            )
            calibrator.save(
                TextSaver(), txt_file, result["camera_matrix"], result["dist_coeffs"]
            )
            save_camera_params(
                yaml_file,
                calibrator.image_size,
                result["camera_matrix"],
                result["dist_coeffs"],
                result["rms"],
            )
            logger.info(
                f"Charuco depth calibration complete. RMS: {result['rms']:.6f}  RMSE: {result['mean_rmse']:.6f}"
            )
            K_depth, dist_depth = result["camera_matrix"], result["dist_coeffs"]
    else:
        K_depth = dist_depth = None

    # ---- Stereo calibration (extrinsics) ----
    if mode.lower() in ["stereo", "both"] and K_rgb is not None and K_depth is not None:
        logger.info("==== Stereo (extrinsic) calibration ====")
        objpoints, imgpoints_rgb, imgpoints_depth, image_size = (
            find_stereo_correspondences(
                rgb_files, depth_files, board, dictionary, logger
            )
        )
        if len(objpoints) == 0 or image_size is None:
            logger.error(
                "No valid stereo charuco frames found. Stereo calibration aborted."
            )
            return
        result = stereo_calibrate(
            objpoints,
            imgpoints_rgb,
            imgpoints_depth,
            K_rgb,
            dist_rgb,
            K_depth,
            dist_depth,
            image_size,
            logger=logger,
        )
        yaml_file = os.path.join(output_dir, "rgb_depth_extrinsics.yml")
        extr_data = {
            "rotation": result["R"].tolist(),
            "translation": result["T"].flatten().tolist(),
            "rms": float(result["rms"]),
        }
        with open(yaml_file, "w") as f:
            yaml.safe_dump(extr_data, f)
        logger.info(
            f"Stereo extrinsics saved to {yaml_file}.\nR:\n{result['R']}\nT: {result['T'].flatten()}\nRMS: {result['rms']:.6f}"
        )


if __name__ == "__main__":
    run_calibration("both")
