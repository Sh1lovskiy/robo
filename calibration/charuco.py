# calibration/charuco.py

import os
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List
from utils.logger import Logger


class CalibrationSaver:
    """Strategy interface for saving calibration results."""

    def save(self, filename, camera_matrix, dist_coeffs):
        raise NotImplementedError


class OpenCVXmlSaver(CalibrationSaver):
    def save(self, filename, camera_matrix, dist_coeffs):
        dir_ = os.path.dirname(filename)
        if dir_ and not os.path.exists(dir_):
            os.makedirs(dir_, exist_ok=True)
        fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
        fs.write("camera_matrix", camera_matrix)
        fs.write("dist_coeffs", dist_coeffs)
        fs.release()


class TextSaver(CalibrationSaver):
    def save(self, filename, camera_matrix, dist_coeffs):
        dir_ = os.path.dirname(filename)
        if dir_ and not os.path.exists(dir_):
            os.makedirs(dir_, exist_ok=True)
        with open(filename, "w") as f:
            np.savetxt(f, camera_matrix, fmt="%.8f", header="camera_matrix")
            np.savetxt(f, dist_coeffs, fmt="%.8f", header="dist_coeffs")


@dataclass
class CharucoCalibrator:
    """Charuco board calibration using OpenCV."""

    board: any
    dictionary: any
    logger: any = field(default_factory=lambda: Logger.get_logger("calibration.charuco"))
    all_corners: List[np.ndarray] = field(default_factory=list, init=False)
    all_ids: List[np.ndarray] = field(default_factory=list, init=False)
    img_size: any = field(default=None, init=False)

    def add_frame(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        res = cv2.aruco.detectMarkers(gray, self.dictionary)
        if len(res[0]) > 0:
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                res[0], res[1], gray, self.board
            )
            if (
                charuco_corners is not None
                and charuco_ids is not None
                and len(charuco_corners) > 3
            ):
                self.all_corners.append(charuco_corners)
                self.all_ids.append(charuco_ids)
                self.img_size = gray.shape[::-1]
                self.logger.debug(f"Frame added, ids found: {len(charuco_ids)}")
                return True
        self.logger.warning("No Charuco corners found in frame")
        return False

    def calibrate(self):
        assert self.img_size is not None, "No frames added."
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = (
            cv2.aruco.calibrateCameraCharuco(
                self.all_corners, self.all_ids, self.board, self.img_size, None, None
            )
        )
        self.logger.info(f"Charuco calibration RMS: {ret:.6f}")
        return dict(
            rms=ret,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            rvecs=rvecs,
            tvecs=tvecs,
        )

    def save(self, saver: CalibrationSaver, filename, camera_matrix, dist_coeffs):
        saver.save(filename, camera_matrix, dist_coeffs)
        self.logger.info(
            f"Calibration saved with {saver.__class__.__name__} to {filename}"
        )


if __name__ == "__main__":
    board = cv2.aruco.CharucoBoard_create(
        5, 7, 0.04, 0.02, cv2.aruco.getPredefinedDictionary(0)
    )
    calib = CharucoCalibrator(board, board.dictionary)
    img = np.full((480, 640, 3), 255, np.uint8)
    assert not calib.add_frame(img)
    print("CharucoCalibrator minimal test OK")
