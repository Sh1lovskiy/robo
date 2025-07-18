import os
import cv2
import yaml
from utils.logger import Logger


DICT_MAP = {
    "4X4_50": cv2.aruco.DICT_4X4_50,
    "4X4_100": cv2.aruco.DICT_4X4_100,
    "5X5_50": cv2.aruco.DICT_5X5_50,
    "5X5_100": cv2.aruco.DICT_5X5_100,
    "6X6_50": cv2.aruco.DICT_6X6_50,
    "6X6_100": cv2.aruco.DICT_6X6_100,
}


def save_camera_params(filename, image_size, camera_matrix, dist_coeffs, total_avg_err):
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
            "cols": dist_coeffs.shape[1],
            "dt": "d",
            "data": dist_coeffs.tolist(),
        },
        "avg_reprojection_error": float(total_avg_err),
    }

    dir_ = os.path.dirname(filename)
    if dir_:
        os.makedirs(dir_, exist_ok=True)

    with open(filename, "w") as f:
        yaml.dump(calibration_data, f)
    print(f"Saved calibration to {filename}")


def calibrate_from_folder():
    logger = Logger.get_logger("cli.charuco_batch_calib")

    folder = "calib/imgs"
    output_dir = "calib/calib_res"
    xml_file = "charuco_cam.xml"
    txt_file = "charuco_cam.txt"

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
    # img = cv2.aruco.drawPlanarBoard(board, (600, 800), 10, 1)
    # cv2.imwrite("debug_board.png", img)

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
