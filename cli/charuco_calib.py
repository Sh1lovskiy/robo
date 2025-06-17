# cli/charuco_calib.py
"""Batch Charuco calibration from a folder of images."""

import os
import cv2
from calibration.charuco import CharucoCalibrator, OpenCVXmlSaver, TextSaver
from utils.config import Config
from utils.logger import Logger
from utils.constants import CHARUCO_DICT_MAP


class CharucoCalibrationCLI:
    """Command-line tool for batch Charuco board calibration."""

    def __init__(self, logger=None):
        Config.load()
        self.cfg = Config.get("charuco")
        self.logger = logger or Logger.get_logger("cli.charuco_calib")

    def run(self):
        folder = self.cfg.get("images_dir", "calib_images")
        output_dir = self.cfg.get("calib_output_dir", "calibration/results")
        xml_file = os.path.join(output_dir, self.cfg.get("xml_file", "charuco_cam.xml"))
        txt_file = os.path.join(output_dir, self.cfg.get("txt_file", "charuco_cam.txt"))

        squares_x = self.cfg.get("squares_x", 5)
        squares_y = self.cfg.get("squares_y", 7)
        square_length = self.cfg.get("square_length", 0.035)
        marker_length = self.cfg.get("marker_length", 0.026)
        dict_name = self.cfg.get("aruco_dict", "5X5_100")
        if dict_name not in CHARUCO_DICT_MAP:
            raise ValueError(f"Unknown ArUco dictionary: {dict_name}")
        dictionary = cv2.aruco.getPredefinedDictionary(CHARUCO_DICT_MAP[dict_name])
        board = cv2.aruco.CharucoBoard_create(
            squares_x, squares_y, square_length, marker_length, dictionary
        )
        calibrator = CharucoCalibrator(board, dictionary, logger=self.logger)

        files = [
            os.path.join(folder, f)
            for f in sorted(os.listdir(folder))
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        self.logger.info(f"Found {len(files)} images in {folder}")
        if not files:
            self.logger.error("No images found.")
            return

        n_good = 0
        for fname in files:
            img = cv2.imread(fname)
            if img is None:
                self.logger.warning(f"Failed to read {fname}")
                continue
            if calibrator.add_frame(img):
                n_good += 1
                self.logger.info(f"Frame added: {fname}")
            else:
                self.logger.warning(f"Frame rejected: {fname}")

        if n_good == 0:
            self.logger.error("No valid charuco frames found. Calibration aborted.")
            return

        result = calibrator.calibrate()
        calibrator.save(OpenCVXmlSaver(), xml_file, result["camera_matrix"], result["dist_coeffs"])
        calibrator.save(TextSaver(), txt_file, result["camera_matrix"], result["dist_coeffs"])
        self.logger.info(f"Charuco calibration complete. RMS: {result['rms']:.5f}")


def main():
    CharucoCalibrationCLI().run()


if __name__ == "__main__":
    main()
