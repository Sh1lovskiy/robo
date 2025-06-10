# cli/charuco_calib.py

import os
import cv2
from calibration.charuco import CharucoCalibrator, OpenCVXmlSaver, TextSaver
from utils.config import Config
from utils.logger import Logger


DICT_MAP = {
    "4X4_50": cv2.aruco.DICT_4X4_50,
    "4X4_100": cv2.aruco.DICT_4X4_100,
    "5X5_50": cv2.aruco.DICT_5X5_50,
    "5X5_100": cv2.aruco.DICT_5X5_100,
    "6X6_50": cv2.aruco.DICT_6X6_50,
    "6X6_100": cv2.aruco.DICT_6X6_100,
}


def calibrate_from_folder():
    Config.load("config.yaml")
    cfg = Config.get("charuco")

    logger = Logger.get_logger("cli.charuco_batch_calib")

    folder = cfg.get("images_dir", "calib_images")
    output_dir = cfg.get("calib_output_dir", "calibration/results")
    xml_file = os.path.join(output_dir, cfg.get("xml_file", "charuco_cam.xml"))
    txt_file = os.path.join(output_dir, cfg.get("txt_file", "charuco_cam.txt"))

    squares_x = cfg.get("squares_x", 5)
    squares_y = cfg.get("squares_y", 7)
    square_length = cfg.get("square_length", 0.035)
    marker_length = cfg.get("marker_length", 0.026)
    dict_name = cfg.get("aruco_dict", "5X5_100")
    dictionary = cv2.aruco.getPredefinedDictionary(DICT_MAP[dict_name])

    board = cv2.aruco.CharucoBoard_create(
        squares_x, squares_y, square_length, marker_length, dictionary
    )
    calibrator = CharucoCalibrator(board, dictionary, logger=logger)

    files = sorted(
        [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ]
    )
    logger.info(f"Found {len(files)} images in {folder}")
    if not files:
        logger.error("No images found.")
        return

    n_good = 0
    for fname in files:
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
        logger.error("No valid charuco frames found. Calibration aborted.")
        return

    result = calibrator.calibrate()
    calibrator.save(
        OpenCVXmlSaver(), xml_file, result["camera_matrix"], result["dist_coeffs"]
    )
    calibrator.save(
        TextSaver(), txt_file, result["camera_matrix"], result["dist_coeffs"]
    )
    logger.info(f"Charuco calibration complete. RMS: {result['rms']:.5f}")


if __name__ == "__main__":
    calibrate_from_folder()
