# cli/charuco_calib.py
"""Batch Charuco calibration (OpenCV 4.11+). Saves XML, TXT (with RMSE), visualizes detection."""

import os
import cv2
import numpy as np
from utils.config import Config
from utils.logger import Logger
from utils.constants import CHARUCO_DICT_MAP


class CharucoCalibrator:
    """
    Charuco batch calibration with optional marker detection visualization (OpenCV 4.11+).
    """

    def __init__(self, board, dictionary, logger=None, visualize=False):
        self.board = board
        self.dictionary = dictionary
        self.logger = logger or Logger.get_logger("cli.charuco_calib")
        self.visualize = visualize
        self.all_corners = []
        self.all_ids = []
        self.img_shape = None

        # DetectorParameters is a class object in OpenCV 4.11
        self.params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.params)

    def add_frame(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect markers with ArucoDetector
        corners, ids, _ = self.detector.detectMarkers(gray)

        if self.visualize:
            vis_img = img.copy()
            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(vis_img, corners, ids)
            cv2.imshow("Detected Markers", vis_img)
            cv2.waitKey(50)  # short pause for visualization

        # Must have at least 4 detected markers for charuco detection
        if ids is None or len(ids) < 4:
            return False

        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners, markerIds=ids, image=gray, board=self.board
        )
        # Need at least 4 valid charuco corners for calibration
        if charuco_ids is None or len(charuco_ids) < 4:
            return False

        self.all_corners.append(charuco_corners)
        self.all_ids.append(charuco_ids)
        self.img_shape = gray.shape[::-1]  # (width, height)
        return True

    def calibrate(self):
        ret, camera_matrix, dist_coeffs, _, _ = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=self.all_corners,
            charucoIds=self.all_ids,
            board=self.board,
            imageSize=self.img_shape,
            cameraMatrix=None,
            distCoeffs=None,
            flags=0,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
        )
        return {"rms": ret, "camera_matrix": camera_matrix, "dist_coeffs": dist_coeffs}

    def save_xml(self, filename, camera_matrix, dist_coeffs):
        fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
        fs.write("camera_matrix", camera_matrix)
        fs.write("dist_coeffs", dist_coeffs)
        fs.release()

    def save_txt(self, filename, camera_matrix, dist_coeffs, rms):
        with open(filename, "w") as f:
            f.write(f"RMS Error: {rms:.6f}\n")
            f.write("camera_matrix =\n")
            np.savetxt(f, camera_matrix, fmt="%.10f")
            f.write("dist_coeffs =\n")
            np.savetxt(f, dist_coeffs.reshape(1, -1), fmt="%.10f")


class CharucoCalibrationCLI:
    """Batch Charuco calibration CLI."""

    def __init__(self, logger=None, visualize=True):
        Config.load()
        self.cfg = Config.get("charuco")
        self.logger = logger or Logger.get_logger("cli.charuco_calib")
        self.visualize = visualize

    def run(self):
        folder = self.cfg.get("images_dir", "cloud")
        output_dir = self.cfg.get("calib_output_dir", "calibration/results")
        os.makedirs(output_dir, exist_ok=True)
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
        board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y),
            square_length,
            marker_length,
            dictionary,
        )
        calibrator = CharucoCalibrator(
            board, dictionary, logger=self.logger, visualize=self.visualize
        )

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

        cv2.destroyAllWindows()

        if n_good == 0:
            self.logger.error("No valid charuco frames found. Calibration aborted.")
            return

        result = calibrator.calibrate()
        calibrator.save_xml(xml_file, result["camera_matrix"], result["dist_coeffs"])
        calibrator.save_txt(
            txt_file, result["camera_matrix"], result["dist_coeffs"], result["rms"]
        )
        self.logger.info(f"Charuco calibration complete. RMS: {result['rms']:.6f}")


def main():
    CharucoCalibrationCLI().run()


if __name__ == "__main__":
    main()
