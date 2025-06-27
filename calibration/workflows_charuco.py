from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import cv2
import numpy as np

from calibration.helpers.charuco import CharucoCalibrator, load_board
from calibration.helpers.pose_utils import (
    load_camera_params,
    save_camera_params_xml,
    save_camera_params_txt,
)
from utils.config import Config
from utils.logger import Logger, LoggerType
from utils.cli import Command
from utils.settings import paths


@dataclass
class CharucoCalibrationWorkflow:
    """Run Charuco calibration on a folder of images."""

    visualize: bool = True
    logger: LoggerType = Logger.get_logger("calibration.workflow.charuco")

    def _load_config(self) -> tuple[
        dict,
        str,
        str,
        str,
        list[str],
        cv2.aruco_CharucoBoard,
        cv2.aruco_Dictionary,
    ]:
        """Read configuration and build board/dictionary objects."""
        if Config._data is None:
            Config.load()
        cfg = Config.get("charuco")
        folder = str(paths.CAPTURES_DIR)
        if not os.path.isdir(folder):
            self.logger.error(f"Images directory {folder} not found")
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
        """Feed all images to ``calibrator`` with optional visualization."""
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
        """Write calibration outputs in XML and TXT formats."""
        save_camera_params_xml(xml_file, result["camera_matrix"], result["dist_coeffs"])
        save_camera_params_txt(
            txt_file,
            result["camera_matrix"],
            result["dist_coeffs"],
            rms=result.get("rms"),
        )
        # self.logger.info(f"Calibration RMS: {float(result['rms'])}")

    def run(self) -> None:
        """Perform full Charuco calibration workflow."""
        try:
            cfg, out_dir, xml_file, txt_file, images, board, dictionary = (
                self._load_config()
            )
        except FileNotFoundError:
            return
        calibrator = CharucoCalibrator(board, dictionary, self.logger)
        self._process_images(calibrator, images)
        if not calibrator.all_corners:
            self.logger.error("No valid frames for calibration")
            return
        result = calibrator.calibrate()
        self._save_results(xml_file, txt_file, result)


def add_charuco_args(parser: argparse.ArgumentParser) -> None:
    """CLI arguments for the Charuco calibration command."""
    parser.add_argument(
        "--no_viz",
        action="store_true",
        help="Disable frame visualization",
    )


def run_charuco(args: argparse.Namespace) -> None:
    """Entry point used by :mod:`argparse`."""
    CharucoCalibrationWorkflow(not args.no_viz).run()
