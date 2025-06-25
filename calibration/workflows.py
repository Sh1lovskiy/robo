"""High-level calibration routines for Charuco and hand-eye workflows."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import cv2
import numpy as np

from calibration.charuco import CharucoCalibrator, load_board
from calibration.handeye import HandEyeCalibrator, NPZHandEyeSaver, TxtHandEyeSaver
from calibration.pose_loader import JSONPoseLoader
from calibration.pose_extractor import extract_charuco_poses, ExtractionParams
from utils.config import Config
from utils.io import load_camera_params, save_camera_params_xml, save_camera_params_txt
from utils.logger import Logger, LoggerType
from utils.cli import Command, CommandDispatcher
from utils.settings import paths


@dataclass
class CharucoCalibrationWorkflow:
    """Run Charuco calibration on a folder of images."""

    visualize: bool = True
    logger: LoggerType = Logger.get_logger("calibration.workflow.charuco")

    def _load_config(
        self,
    ) -> tuple[
        dict, str, str, str, list[str], cv2.aruco_CharucoBoard, cv2.aruco_Dictionary
    ]:
        if Config._data is None:
            Config.load()
        cfg = Config.get("charuco")
        folder = cfg.get("images_dir", "captures") or str(paths.CAPTURES_DIR)
        if not os.path.isdir(folder):
            self.logger.error("Images directory '%s' not found", folder)
            raise FileNotFoundError(folder)
        out_dir = cfg.get("calib_output_dir", "calibration/results") or str(
            paths.RESULTS_DIR
        )
        os.makedirs(out_dir, exist_ok=True)
        xml_file = os.path.join(out_dir, cfg.get("xml_file", "charuco_cam.xml")) or str(
            paths.RESULTS_DIR / "charuco_cam.xml"
        )
        txt_file = os.path.join(out_dir, cfg.get("txt_file", "charuco_cam.txt")) or str(
            paths.RESULTS_DIR / "charuco_cam.txt"
        )
        board, dictionary = load_board(cfg)
        images = [
            os.path.join(folder, f)
            for f in sorted(os.listdir(folder))
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        self.logger.info(f"Found {len(images)} images in {folder}")
        return cfg, out_dir, xml_file, txt_file, images, board, dictionary

    def _process_images(self, calibrator: CharucoCalibrator, images: list[str]) -> None:
        for img_path in Logger.progress(images, desc="Charuco frames"):
            img = cv2.imread(img_path)
            if img is None:
                self.logger.warning("Cannot read %s", img_path)
                continue
            if calibrator.add_frame(img) and self.visualize:
                cv2.imshow("detected", img)
                cv2.waitKey(50)
        cv2.destroyAllWindows()
        if self.visualize:
            cv2.destroyAllWindows()

    def _save_results(
        self,
        xml_file: str,
        txt_file: str,
        result: dict[str, np.ndarray | float],
    ) -> None:
        save_camera_params_xml(xml_file, result["camera_matrix"], result["dist_coeffs"])
        save_camera_params_txt(
            txt_file,
            result["camera_matrix"],
            result["dist_coeffs"],
            rms=result.get("rms"),
        )
        self.logger.info("Calibration RMS: %.6f", float(result["rms"]))

    def run(self) -> None:
        try:
            cfg, out_dir, xml_file, txt_file, images, board, dictionary = (
                self._load_config()
            )
        except FileNotFoundError:
            return
        calibrator = CharucoCalibrator(board, dictionary, self.logger)
        self.logger.info("Found %d images in %s", len(images), cfg.get("images_dir"))
        self._process_images(calibrator, images)
        if not calibrator.all_corners:
            self.logger.error("No valid frames for calibration")
            return
        result = calibrator.calibrate()
        self._save_results(xml_file, txt_file, result)


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
        out_dir = cfg.get("calib_output_dir", "calibration/results") or str(
            paths.RESULTS_DIR
        )
        os.makedirs(out_dir, exist_ok=True)
        charuco_xml = cfg.get(
            "charuco_xml", os.path.join(out_dir, "charuco_cam.xml")
        ) or str(paths.RESULTS_DIR / "charuco_cam.xml")
        robot_file = cfg.get("robot_poses_file", "cloud/poses.json") or str(
            paths.CAPTURES_DIR / "poses.json"
        )
        images_dir = cfg.get("images_dir", "cloud") or str(paths.CAPTURES_DIR)
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
        def extract_index(fname):
            return os.path.splitext(os.path.basename(fname))[0].split("_")[0]

        idx_map = {extract_index(p): i for i, p in enumerate(all_paths)}
        indices = []
        for p in valid_paths:
            idx = extract_index(p)
            if idx in idx_map:
                indices.append(idx_map[idx])
            else:
                self.logger.warning(f"No robot pose for image {p} (index {idx})")
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
            self.logger.info(f"Saved {name} calibration to {npz_file}")

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

        params = ExtractionParams(
            min_corners=cfg.get("min_corners", 4),
            visualize=cfg.get("visualize", False),
            analyze_corners=True,
            outlier_std=float(cfg.get("outlier_std", 2.0)),
        )
        extraction = extract_charuco_poses(
            images_dir,
            board,
            dictionary,
            camera_matrix,
            dist_coeffs,
            logger=self.logger,
            params=params,
        )
        self.logger.info(f"len(extraction.rotations): {len(extraction.rotations)}")
        self.logger.info(f"len(extraction.valid_paths): {len(extraction.valid_paths)}")
        for p in extraction.valid_paths:
            self.logger.info(f"Valid: {p}")
        Rs_t2c = extraction.rotations
        ts_t2c = extraction.translations
        valid_paths = extraction.valid_paths[: len(Rs_t2c)]
        all_paths = extraction.all_paths
        self.logger.info(f"Total camera poses: {len(Rs_t2c)}")
        self.logger.info(f"Total robot poses: {len(Rs_g2b)}")
        self.logger.info(f"Valid image paths: {valid_paths}")
        self.logger.info(f"All image paths: {all_paths}")
        Rs_g2b_f, ts_g2b_f = self._filter_robot_poses(
            Rs_g2b, ts_g2b, valid_paths, all_paths
        )
        self.logger.info(
            f"After robot pose filtering: {len(Rs_g2b_f)} robot poses for {len(valid_paths)} valid paths"
        )
        self.logger.info(f"len(Rs_t2c): {len(Rs_t2c)}")
        if not Rs_t2c or len(Rs_t2c) != len(Rs_g2b_f):
            self.logger.error(
                f"Pose data mismatch after filtering: {len(Rs_t2c)} camera poses, {len(Rs_g2b_f)} robot poses"
            )
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
