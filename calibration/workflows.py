"""High-level calibration routines for Charuco and hand-eye workflows."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import cv2
import matplotlib.pyplot as plt
import numpy as np

from calibration.charuco import CharucoCalibrator
from calibration.charuco import (
    CHARUCO_DICT_MAP,
    CharucoCalibrator,
    extract_charuco_poses,
    load_board,
)
from calibration.handeye import HandEyeCalibrator, NPZHandEyeSaver, TxtHandEyeSaver
from calibration.pose_loader import JSONPoseLoader
from calibration.pose_extractor import extract_charuco_poses, ExtractionParams
from utils.config import Config
from utils.io import load_camera_params, save_camera_params_xml, save_camera_params_txt
from utils.logger import Logger, LoggerType
from utils.cli import Command, CommandDispatcher


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
        board, dictionary = load_board(cfg)
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

    def _load_config(
        self,
    ) -> tuple[
        dict,
        str,
        str,
        str,
        str,
        str,
        cv2.aruco_CharucoBoard,
        cv2.aruco_Dictionary,
    ]:
        if Config._data is None:
            Config.load()
        cfg = Config.get("handeye")
        out_dir = cfg.get("calib_output_dir", "calibration/results")
        os.makedirs(out_dir, exist_ok=True)
        charuco_xml = cfg.get("charuco_xml", os.path.join(out_dir, "charuco_cam.xml"))
        robot_file = cfg.get("robot_poses_file", "cloud/poses.json")
        images_dir = cfg.get("images_dir", "cloud")
        method = cfg.get("method", "ALL").upper()
        board, dictionary = load_board(cfg)
        return (
            cfg,
            out_dir,
            charuco_xml,
            robot_file,
            images_dir,
            method,
            board,
            dictionary,
        )

    def _filter_robot_poses(
        self,
        Rs: list[np.ndarray],
        ts: list[np.ndarray],
        valid_paths: list[str],
        all_paths: list[str],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        name2idx = {os.path.basename(p): i for i, p in enumerate(all_paths)}
        indices = [name2idx[os.path.basename(p)] for p in valid_paths]
        return [Rs[i] for i in indices], [ts[i] for i in indices]

    def _save(
        self,
        calibrator: HandEyeCalibrator,
        out_dir: str,
        method: str,
        results: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> None:
        if results is None:
            R, t = calibrator.calibrate(method)
            results = {method: (R, t)}
        for name, (R, t) in results.items():
            npz_file = os.path.join(out_dir, f"handeye_{name}.npz")
            txt_file = os.path.join(out_dir, f"handeye_{name}.txt")
            calibrator.save(NPZHandEyeSaver(), npz_file, R, t)
            calibrator.save(TxtHandEyeSaver(), txt_file, R, t)
            self.logger.info("Saved %s calibration to %s", name, npz_file)

    def run(self) -> None:
        cfg, out_dir, charuco_xml, robot_file, images_dir, method, board, dictionary = (
            self._load_config()
        )
        if not os.path.isfile(charuco_xml):
            self.logger.error("Charuco file '%s' not found", charuco_xml)
            return
        if not os.path.isfile(robot_file):
            self.logger.error("Robot poses file '%s' not found", robot_file)
            return
        if not os.path.isdir(images_dir):
            self.logger.error("Images directory '%s' not found", images_dir)
            return

        camera_matrix, dist_coeffs = load_camera_params(charuco_xml)
        Rs_g2b, ts_g2b = JSONPoseLoader.load_poses(robot_file)

        poses, _ = extract_charuco_poses(
            images_dir,
            board,
            dictionary,
            camera_matrix,
            dist_coeffs,
            min_corners=cfg.get("min_corners", 4),
            visualize=cfg.get("visualize", False),
            analyze_corners=True,
            outlier_std=cfg.get("outlier_std", 2.0),
            logger=self.logger,
        )
        Rs_t2c, ts_t2c, valid_paths, all_paths = poses
        Rs_g2b_f, ts_g2b_f = self._filter_robot_poses(
            Rs_g2b, ts_g2b, valid_paths, all_paths
        )
        if not Rs_t2c or len(Rs_t2c) != len(Rs_g2b_f):
            self.logger.error("Pose data mismatch after filtering")
            return

        calibrator = HandEyeCalibrator(self.logger)
        for Rg, tg, Rc, tc in Logger.progress(
            list(zip(Rs_g2b_f, ts_g2b_f, Rs_t2c, ts_t2c)),
            desc="HandEye samples",
        ):
            calibrator.add_sample(Rg, tg, Rc, tc)
        if method == "ALL":
            results = calibrator.calibrate_all()
            self._save(calibrator, out_dir, method, results)
        else:
            self._save(calibrator, out_dir, method)


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
        default=cfg.get("images_dir", "captures"),
        help="Directory with Charuco images",
    )
    parser.add_argument(
        "--charuco_xml",
        default=cfg.get("charuco_xml", os.path.join(out_dir, "charuco_cam.xml")),
        help="Camera calibration XML file",
    )
    parser.add_argument(
        "--robot_poses_file",
        default=cfg.get("robot_poses_file", "captures/poses.json"),
        help="JSON file with robot poses",
    )
    parser.add_argument(
        "--method",
        default=cfg.get("method", "ALL"),
        help="Calibration method or ALL",
    )
    parser.add_argument(
        "--analyze-charuco",
        action="store_true",
        help="Analyze Charuco board positions and reject outlier images",
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
