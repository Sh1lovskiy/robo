# calibration/workflows.py
"""High-level calibration routines for Charuco and hand-eye workflows."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from calibration.charuco import CharucoCalibrator
from calibration.handeye import HandEyeCalibrator, NPZHandEyeSaver, TxtHandEyeSaver
from calibration.pose_loader import JSONPoseLoader
from utils.config import Config
from utils.io import load_camera_params, save_camera_params_xml, save_camera_params_txt
from utils.logger import Logger

# Mapping of Charuco dictionary names to OpenCV constants
CHARUCO_DICT_MAP = {"5X5_50": 8, "5X5_100": 9}


@dataclass
class CharucoCalibrationWorkflow:
    """Run Charuco calibration on a folder of images."""

    visualize: bool = True
    logger: Logger = Logger.get_logger("calibration.workflow.charuco")

    def run(self) -> None:
        Config.load()
        cfg = Config.get("charuco")
        folder = cfg.get("images_dir", "cloud")
        out_dir = cfg.get("calib_output_dir", "calibration/results")
        os.makedirs(out_dir, exist_ok=True)
        xml_file = os.path.join(out_dir, cfg.get("xml_file", "charuco_cam.xml"))
        txt_file = os.path.join(out_dir, cfg.get("txt_file", "charuco_cam.txt"))

        squares_x = cfg.get("squares_x", 5)
        squares_y = cfg.get("squares_y", 7)
        square_length = cfg.get("square_length", 0.035)
        marker_length = cfg.get("marker_length", 0.026)
        dict_name = cfg.get("aruco_dict", "5X5_100")
        if dict_name not in CHARUCO_DICT_MAP:
            raise ValueError(f"Unknown ArUco dictionary: {dict_name}")
        dictionary = cv2.aruco.getPredefinedDictionary(CHARUCO_DICT_MAP[dict_name])
        board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, dictionary)

        calibrator = CharucoCalibrator(board, dictionary, self.logger)

        images = [
            os.path.join(folder, f)
            for f in sorted(os.listdir(folder))
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        self.logger.info(f"Found {len(images)} images in {folder}")
        for img_path in Logger.progress(images, desc="Charuco frames"):
            img = cv2.imread(img_path)
            if img is None:
                self.logger.warning(f"Cannot read {img_path}")
                continue
            if calibrator.add_frame(img) and self.visualize:
                cv2.imshow("detected", img)
                cv2.waitKey(50)
        cv2.destroyAllWindows()

        if not calibrator.all_corners:
            self.logger.error("No valid frames for calibration")
            return
        result = calibrator.calibrate()
        save_camera_params_xml(xml_file, result["camera_matrix"], result["dist_coeffs"])
        save_camera_params_txt(txt_file, result["camera_matrix"], result["dist_coeffs"], rms=result["rms"])
        self.logger.info(f"Calibration RMS: {result['rms']:.6f}")


@dataclass
class HandEyeCalibrationWorkflow:
    """Run hand-eye calibration using saved poses and Charuco frames."""

    logger: Logger = Logger.get_logger("calibration.workflow.handeye")

    def run(self) -> None:
        Config.load()
        cfg = Config.get("handeye")
        out_dir = cfg.get("calib_output_dir", "calibration/results")
        os.makedirs(out_dir, exist_ok=True)
        charuco_xml = cfg.get("charuco_xml", os.path.join(out_dir, "charuco_cam.xml"))
        robot_poses_file = cfg.get("robot_poses_file", "cloud/poses.json")
        method = cfg.get("method", "ALL").upper()
        images_dir = cfg.get("images_dir", "cloud")

        squares_x = cfg.get("squares_x", 5)
        squares_y = cfg.get("squares_y", 7)
        square_length = cfg.get("square_length", 0.035)
        marker_length = cfg.get("marker_length", 0.026)
        dict_name = cfg.get("aruco_dict", "5X5_100")
        if dict_name not in CHARUCO_DICT_MAP:
            raise ValueError(f"Unknown ArUco dictionary: {dict_name}")
        dictionary = cv2.aruco.getPredefinedDictionary(CHARUCO_DICT_MAP[dict_name])
        board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, dictionary)

        camera_matrix, dist_coeffs = load_camera_params(charuco_xml)
        Rs_g2b, ts_g2b = JSONPoseLoader.load_poses(robot_poses_file)

        Rs_t2c, ts_t2c = self._extract_charuco_poses(images_dir, board, dictionary, camera_matrix, dist_coeffs)
        if not Rs_t2c or len(Rs_t2c) != len(Rs_g2b):
            self.logger.error("Pose data mismatch")
            return

        calibrator = HandEyeCalibrator(self.logger)
        for Rg, tg, Rc, tc in Logger.progress(
            list(zip(Rs_g2b, ts_g2b, Rs_t2c, ts_t2c)),
            desc="HandEye samples",
        ):
            calibrator.add_sample(Rg, tg, Rc, tc)

        if method == "ALL":
            results = calibrator.calibrate_all()
            for name, (R, t) in Logger.progress(
                results.items(), desc="Saving results", total=len(results)
            ):
                npz_file = os.path.join(out_dir, f"handeye_{name}.npz")
                txt_file = os.path.join(out_dir, f"handeye_{name}.txt")
                calibrator.save(NPZHandEyeSaver(), npz_file, R, t)
                calibrator.save(TxtHandEyeSaver(), txt_file, R, t)
                self.logger.info(f"Saved {name} calibration to {npz_file}")
        else:
            R, t = calibrator.calibrate(method)
            npz_file = os.path.join(out_dir, "handeye.npz")
            txt_file = os.path.join(out_dir, "handeye.txt")
            calibrator.save(NPZHandEyeSaver(), npz_file, R, t)
            calibrator.save(TxtHandEyeSaver(), txt_file, R, t)
            self.logger.info(f"Calibration saved to {npz_file}")

    def _extract_charuco_poses(
        self,
        images_dir: str,
        board: cv2.aruco_CharucoBoard,
        dictionary: cv2.aruco_Dictionary,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        images = [
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        Rs, ts = [], []
        for img_path in Logger.progress(images, desc="Extract poses"):
            img = cv2.imread(img_path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
            if ids is None or len(ids) == 0:
                continue
            _, char_corners, char_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
            if char_ids is None or len(char_ids) < 4:
                continue
            retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(char_corners, char_ids, board, camera_matrix, dist_coeffs)
            if retval:
                R, _ = cv2.Rodrigues(rvec)
                Rs.append(R)
                ts.append(tvec.flatten())
        self.logger.info(f"Extracted {len(Rs)} Charuco poses")
        return Rs, ts


def main_charuco() -> None:
    """CLI entry for Charuco calibration."""
    CharucoCalibrationWorkflow().run()


def main_handeye() -> None:
    """CLI entry for Hand-Eye calibration."""
    HandEyeCalibrationWorkflow().run()
