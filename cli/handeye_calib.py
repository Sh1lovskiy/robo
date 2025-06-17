# cli/handeye_calib.py
"""Command-line interface for hand-eye calibration."""

import os
import numpy as np
import cv2
import json
from calibration.handeye import HandEyeCalibrator, NPZHandEyeSaver, TxtHandEyeSaver
from utils.config import Config
from utils.logger import Logger


def load_camera_params(xml_path):
    fs = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("dist_coeffs").mat()
    fs.release()
    return camera_matrix, dist_coeffs


def extract_charuco_poses(
    images_dir, board, dictionary, camera_matrix, dist_coeffs, logger
):
    image_paths = sorted(
        [
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )
    Rs, ts = [], []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"Cannot read image: {img_path}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
        if ids is not None and len(ids) > 0:
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board
            )
            if (
                charuco_corners is not None
                and charuco_ids is not None
                and len(charuco_corners) >= 4
            ):
                # Под opencv-contrib-python>=4.7.x допускается (N,2) float32 и (N,1) int32
                cc = np.ascontiguousarray(
                    charuco_corners.reshape(-1, 2).astype(np.float32)
                )
                ci = np.ascontiguousarray(charuco_ids.reshape(-1, 1).astype(np.int32))
                if cc.shape[0] != ci.shape[0]:
                    logger.error(
                        f"Charuco count mismatch in {os.path.basename(img_path)}: corners={cc.shape[0]}, ids={ci.shape[0]}"
                    )
                    continue
                rvec_init = np.zeros((3, 1), dtype=np.float64)
                tvec_init = np.zeros((3, 1), dtype=np.float64)
                try:
                    retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                        cc, ci, board, camera_matrix, dist_coeffs, rvec_init, tvec_init
                    )
                except Exception as e:
                    logger.error(
                        f"estimatePoseCharucoBoard exception on {os.path.basename(img_path)}: {repr(e)}"
                    )
                    logger.error(
                        f"charuco_corners shape: {charuco_corners.shape}, cc shape: {cc.shape}, "
                        f"charuco_ids shape: {charuco_ids.shape}, ci shape: {ci.shape}"
                    )
                    continue
                if retval:
                    R, _ = cv2.Rodrigues(rvec)
                    Rs.append(R)
                    ts.append(tvec.flatten())
                    logger.info(f"Charuco pose found: {os.path.basename(img_path)}")
                else:
                    logger.warning(f"Pose NOT found: {os.path.basename(img_path)}")
            else:
                logger.warning(
                    f"Invalid Charuco data in {os.path.basename(img_path)} "
                    f"(corners={None if charuco_corners is None else charuco_corners.shape}, "
                    f"ids={None if charuco_ids is None else charuco_ids.shape})"
                )
        else:
            logger.warning(f"No ArUco markers detected: {os.path.basename(img_path)}")
    return Rs, ts


def load_robot_poses_from_json(filename):
    """
    Load robot poses from JSON file in format:
    { "000": {"tcp_coords": [x, y, z, rx, ry, rz]}, ... }
    Returns lists of R (3x3) and t (3,)
    """
    with open(filename, "r") as f:
        data = json.load(f)
    Rs, ts = [], []
    for key in sorted(data.keys()):
        tcp = data[key]["tcp_coords"]
        t = np.array(tcp[:3], dtype=np.float64)
        angles = np.deg2rad(np.array(tcp[3:6], dtype=np.float64))
        from scipy.spatial.transform import Rotation as R

        rot = R.from_euler("xyz", angles).as_matrix()
        Rs.append(rot)
        ts.append(t)
    return Rs, ts


class HandEyeCalibrationCLI:
    """
    Command-line tool for hand-eye calibration.
    Loads robot and charuco poses, runs calibration, saves results.
    """

    def __init__(self, logger=None):
        Config.load()
        self.cfg = Config.get("handeye")
        self.logger = logger or Logger.get_logger("cli.handeye_calib")
        self.output_dir = self.cfg.get("calib_output_dir", "calibration/results")
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        images_dir = self.cfg.get("images_dir", "cloud")
        robot_poses_file = self.cfg.get("robot_poses_file", "poses.json")
        charuco_xml = self.cfg.get(
            "charuco_xml", os.path.join(self.output_dir, "charuco_cam.xml")
        )
        method = self.cfg.get("method", "ALL").upper()

        # Charuco board config
        squares_x = self.cfg.get("squares_x", 5)
        squares_y = self.cfg.get("squares_y", 7)
        square_length = self.cfg.get("square_length", 0.035)
        marker_length = self.cfg.get("marker_length", 0.026)
        dict_name = self.cfg.get("aruco_dict", "5X5_100")
        DICT_MAP = {
            "4X4_50": cv2.aruco.DICT_4X4_50,
            "4X4_100": cv2.aruco.DICT_4X4_100,
            "5X5_50": cv2.aruco.DICT_5X5_50,
            "5X5_100": cv2.aruco.DICT_5X5_100,
        }
        if dict_name not in DICT_MAP:
            raise ValueError(f"Unknown ArUco dictionary: {dict_name}")
        dictionary = cv2.aruco.getPredefinedDictionary(DICT_MAP[dict_name])
        board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y), square_length, marker_length, dictionary
        )

        camera_matrix, dist_coeffs = load_camera_params(charuco_xml)

        # Load poses
        Rs_g2b, ts_g2b = load_robot_poses_from_json(robot_poses_file)
        Rs_t2c, ts_t2c = extract_charuco_poses(
            images_dir, board, dictionary, camera_matrix, dist_coeffs, self.logger
        )
        self.logger.info(f"Robot poses: {len(Rs_g2b)}, Charuco poses: {len(Rs_t2c)}")

        if not Rs_t2c:
            self.logger.error("No valid Charuco poses found! Calibration aborted.")
            return
        if len(Rs_g2b) != len(Rs_t2c):
            self.logger.error(
                f"Pose count mismatch: robot={len(Rs_g2b)}, charuco={len(Rs_t2c)}"
            )
            return

        calibrator = HandEyeCalibrator(logger=self.logger)
        for Rg, tg, Rc, tc in zip(Rs_g2b, ts_g2b, Rs_t2c, ts_t2c):
            calibrator.add_sample(Rg, tg, Rc, tc)

        if method == "ALL":
            results = calibrator.calibrate_all()
            for meth, (R, t) in results.items():
                npz_file = os.path.join(self.output_dir, f"handeye_{meth}.npz")
                txt_file = os.path.join(self.output_dir, f"handeye_{meth}.txt")
                calibrator.save(NPZHandEyeSaver(), npz_file, R, t)
                calibrator.save(TxtHandEyeSaver(), txt_file, R, t)
                self.logger.info(f"Saved {meth} results to {npz_file}, {txt_file}")
                print(f"{meth}: saved to {npz_file}, {txt_file}")
        else:
            R, t = calibrator.calibrate(method)
            npz_file = os.path.join(self.output_dir, "handeye.npz")
            txt_file = os.path.join(self.output_dir, "handeye.txt")
            calibrator.save(NPZHandEyeSaver(), npz_file, R, t)
            calibrator.save(TxtHandEyeSaver(), txt_file, R, t)
            self.logger.info(f"Hand-eye calibration saved to {npz_file}, {txt_file}")
            print(f"Hand-eye calibration saved to {npz_file}, {txt_file}")


def main():
    cli = HandEyeCalibrationCLI()
    cli.run()


if __name__ == "__main__":
    main()
