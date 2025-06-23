# calibration/workflows.py
"""High-level calibration routines for Charuco and hand-eye workflows."""

from __future__ import annotations

import os
from dataclasses import dataclass

import argparse
import cv2
import numpy as np

from calibration.charuco import CharucoCalibrator
from calibration.handeye import HandEyeCalibrator, NPZHandEyeSaver, TxtHandEyeSaver
from calibration.pose_loader import JSONPoseLoader
from utils.config import Config
from utils.io import load_camera_params, save_camera_params_xml, save_camera_params_txt
from utils.logger import Logger, LoggerType
from utils.cli import Command, CommandDispatcher

CHARUCO_DICT_MAP = {"5X5_50": 8, "5X5_100": 9}


@dataclass
class CharucoCalibrationWorkflow:
    """Run Charuco calibration on a folder of images."""

    visualize: bool = True
    logger: LoggerType = Logger.get_logger("calibration.workflow.charuco")

    def run(self) -> None:
        if Config._data is None:
            Config.load()
        cfg = Config.get("charuco")
        folder = cfg.get("images_dir", "cloud")
        if not os.path.isdir(folder):
            self.logger.error(f"Images directory '{folder}' not found")
            return
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
        board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y), square_length, marker_length, dictionary
        )

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
        save_camera_params_txt(
            txt_file, result["camera_matrix"], result["dist_coeffs"], rms=result["rms"]
        )
        self.logger.info(f"Calibration RMS: {result['rms']:.6f}")


@dataclass
class HandEyeCalibrationWorkflow:
    """Run hand-eye calibration using saved poses and Charuco frames."""

    logger: LoggerType = Logger.get_logger("calibration.workflow.handeye")

    def run(self) -> None:
        if Config._data is None:
            Config.load()
        cfg = Config.get("handeye")
        out_dir = cfg.get("calib_output_dir", "calibration/results")
        os.makedirs(out_dir, exist_ok=True)
        charuco_xml = cfg.get("charuco_xml", os.path.join(out_dir, "charuco_cam.xml"))
        robot_poses_file = cfg.get("robot_poses_file", "cloud/poses.json")
        method = cfg.get("method", "ALL").upper()
        images_dir = cfg.get("images_dir", "cloud")

        if not os.path.isfile(charuco_xml):
            self.logger.error(f"Charuco file '{charuco_xml}' not found")
            return
        if not os.path.isfile(robot_poses_file):
            self.logger.error(f"Robot poses file '{robot_poses_file}' not found")
            return
        if not os.path.isdir(images_dir):
            self.logger.error(f"Images directory '{images_dir}' not found")
            return

        squares_x = cfg.get("squares_x", 5)
        squares_y = cfg.get("squares_y", 7)
        square_length = cfg.get("square_length", 0.035)
        marker_length = cfg.get("marker_length", 0.026)
        dict_name = cfg.get("aruco_dict", "5X5_100")
        if dict_name not in CHARUCO_DICT_MAP:
            raise ValueError(f"Unknown ArUco dictionary: {dict_name}")
        dictionary = cv2.aruco.getPredefinedDictionary(CHARUCO_DICT_MAP[dict_name])
        board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y), square_length, marker_length, dictionary
        )

        camera_matrix, dist_coeffs = load_camera_params(charuco_xml)
        Rs_g2b, ts_g2b = JSONPoseLoader.load_poses(robot_poses_file)

        Rs_t2c, ts_t2c = self._extract_charuco_poses(
            images_dir, board, dictionary, camera_matrix, dist_coeffs
        )
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
            _, char_corners, char_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board
            )
            if char_ids is None or len(char_ids) < 4:
                continue
            retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                char_corners, char_ids, board, camera_matrix, dist_coeffs
            )
            if retval:
                R, _ = cv2.Rodrigues(rvec)
                Rs.append(R)
                ts.append(tvec.flatten())
        self.logger.info(f"Extracted {len(Rs)} Charuco poses")
        return Rs, ts


def _add_charuco_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--no_viz",
        action="store_true",
        help="Disable frame visualization",
    )


def _run_charuco(args: argparse.Namespace) -> None:
    CharucoCalibrationWorkflow(not args.no_viz).run()


def _add_handeye_args(parser: argparse.ArgumentParser) -> None:
    Config.load()
    cfg = Config.get("handeye")
    out_dir = cfg.get("calib_output_dir", "calibration/results")
    parser.add_argument(
        "--images_dir",
        default=cfg.get("images_dir", "cloud"),
        help="Directory with Charuco images",
    )
    parser.add_argument(
        "--charuco_xml",
        default=cfg.get("charuco_xml", os.path.join(out_dir, "charuco_cam.xml")),
        help="Camera calibration XML file",
    )
    parser.add_argument(
        "--robot_poses_file",
        default=cfg.get("robot_poses_file", "cloud/poses.json"),
        help="JSON file with robot poses",
    )
    parser.add_argument(
        "--method",
        default=cfg.get("method", "ALL"),
        help="Calibration method or ALL",
    )


def _run_handeye(args: argparse.Namespace) -> None:
    Config.load()
    cfg = Config.get("handeye")
    cfg["images_dir"] = args.images_dir
    cfg["charuco_xml"] = args.charuco_xml
    cfg["robot_poses_file"] = args.robot_poses_file
    cfg["method"] = args.method
    HandEyeCalibrationWorkflow().run()


def create_cli() -> CommandDispatcher:
    return CommandDispatcher(
        "Calibration workflows",
        [
            Command(
                "charuco", _run_charuco, _add_charuco_args, "Run Charuco calibration"
            ),
            Command(
                "handeye", _run_handeye, _add_handeye_args, "Run Hand-Eye calibration"
            ),
        ],
    )


def main() -> None:
    logger = Logger.get_logger("calibration.workflows")
    create_cli().run(logger=logger)


if __name__ == "__main__":
    main()
