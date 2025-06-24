# calibration/charuco.py
"""Charuco board calibration utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Mapping

import cv2
import numpy as np

from utils.logger import Logger, LoggerType

CHARUCO_DICT_MAP = {
    "4X4_100": cv2.aruco.DICT_4X4_100,
    "5X5_50": cv2.aruco.DICT_5X5_50,
    "5X5_100": cv2.aruco.DICT_5X5_100,
}


class CalibrationSaver:
    """Strategy interface for saving calibration results."""

    def save(
        self, filename: str, camera_matrix: np.ndarray, dist_coeffs: np.ndarray
    ) -> None:
        raise NotImplementedError


class OpenCVXmlSaver(CalibrationSaver):
    def save(
        self, filename: str, camera_matrix: np.ndarray, dist_coeffs: np.ndarray
    ) -> None:
        dir_ = os.path.dirname(filename)
        if dir_ and not os.path.exists(dir_):
            os.makedirs(dir_, exist_ok=True)
        fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
        fs.write("camera_matrix", camera_matrix)
        fs.write("dist_coeffs", dist_coeffs)
        fs.release()


class TextSaver(CalibrationSaver):
    def save(
        self, filename: str, camera_matrix: np.ndarray, dist_coeffs: np.ndarray
    ) -> None:
        dir_ = os.path.dirname(filename)
        if dir_ and not os.path.exists(dir_):
            os.makedirs(dir_, exist_ok=True)
        with open(filename, "w") as f:
            np.savetxt(f, camera_matrix, fmt="%.8f", header="camera_matrix")
            np.savetxt(f, dist_coeffs, fmt="%.8f", header="dist_coeffs")


def load_board(
    cfg: Mapping[str, float | str],
) -> tuple[cv2.aruco_CharucoBoard, cv2.aruco_Dictionary]:
    """Create a Charuco board from configuration."""

    dict_name = str(cfg.get("aruco_dict", "5X5_100"))
    if dict_name not in CHARUCO_DICT_MAP:
        raise ValueError(f"Unknown ArUco dictionary: {dict_name}")
    squares_x = int(cfg.get("squares_x", 5))
    squares_y = int(cfg.get("squares_y", 7))
    square_len = float(cfg.get("square_length", 0.033))
    marker_len = float(cfg.get("marker_length", 0.025))
    dictionary = cv2.aruco.getPredefinedDictionary(CHARUCO_DICT_MAP[dict_name])
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y), square_len, marker_len, dictionary
    )
    return board, dictionary


def iter_image_paths(folder: str) -> Iterable[str]:
    """Yield image file paths sorted alphabetically."""

    for path in sorted(Path(folder).iterdir()):
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            yield str(path)


def _detect_charuco_pose(
    img_path: str,
    board: cv2.aruco_CharucoBoard,
    dictionary: cv2.aruco_Dictionary,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    *,
    min_corners: int,
    visualize: bool,
    logger: LoggerType | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    img = cv2.imread(img_path)
    if img is None:
        if logger:
            logger.warning("Cannot read image: %s", img_path)
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    if ids is None or len(ids) < min_corners:
        return None
    _, c_corners, c_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
    if c_corners is None or c_ids is None or len(c_ids) < min_corners:
        return None
    ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        c_corners, c_ids, board, camera_matrix, dist_coeffs
    )
    if not ret:
        return None
    R, _ = cv2.Rodrigues(rvec)
    obj_pts = board.getChessboardCorners()
    lt = (R @ obj_pts[0].reshape(3, 1) + tvec).flatten()
    rb = (R @ obj_pts[-1].reshape(3, 1) + tvec).flatten()
    if visualize:
        vis = img.copy()
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)
        cv2.aruco.drawDetectedCornersCharuco(vis, c_corners, c_ids)
        cv2.drawFrameAxes(vis, camera_matrix, dist_coeffs, rvec, tvec, 0.05)
        cv2.imshow("charuco pose", vis)
        cv2.waitKey(100)
    return R, tvec.flatten(), lt, rb


def _corner_stats(
    lt: np.ndarray, rb: np.ndarray, outlier_std: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stats = np.array([lt.mean(axis=0), lt.std(axis=0), rb.mean(axis=0), rb.std(axis=0)])
    mask_lt = np.all(np.abs(lt - stats[0]) <= outlier_std * stats[1], axis=1)
    mask_rb = np.all(np.abs(rb - stats[2]) <= outlier_std * stats[3], axis=1)
    mask = mask_lt & mask_rb
    return stats, mask, np.where(~mask)[0]


def _plot_corner_distribution(lt: np.ndarray, rb: np.ndarray, visualize: bool) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.scatter(lt[:, 0], lt[:, 1], c="blue", label="Left Top", alpha=0.7)
    plt.scatter(rb[:, 0], rb[:, 1], c="red", label="Right Bottom", alpha=0.7)
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("Charuco Board Corner Positions")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("charuco_corners_distribution.png")
    if visualize:
        plt.show()
    plt.close()


def extract_charuco_poses(
    images_dir: str,
    board: cv2.aruco_CharucoBoard,
    dictionary: cv2.aruco_Dictionary,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    *,
    min_corners: int,
    visualize: bool,
    analyze_corners: bool,
    outlier_std: float,
    logger: LoggerType | None = None,
) -> tuple[
    tuple[list[np.ndarray], list[np.ndarray], list[str], list[str]],
    tuple[dict[str, dict[str, np.ndarray]], list[int]],
]:
    Rs: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    valid_paths: list[str] = []
    image_paths = list(iter_image_paths(images_dir))
    lt_list: list[np.ndarray] = []
    rb_list: list[np.ndarray] = []
    indices: list[int] = []
    for idx, path in enumerate(image_paths):
        pose = _detect_charuco_pose(
            path,
            board,
            dictionary,
            camera_matrix,
            dist_coeffs,
            min_corners=min_corners,
            visualize=visualize,
            logger=logger,
        )
        if pose is None:
            continue
        R, t, lt, rb = pose
        Rs.append(R)
        ts.append(t)
        valid_paths.append(path)
        lt_list.append(lt)
        rb_list.append(rb)
        indices.append(idx)
    if visualize:
        cv2.destroyAllWindows()
    stats: dict[str, dict[str, np.ndarray]] = {}
    outliers: list[int] = []
    if analyze_corners and lt_list:
        lt_arr = np.stack(lt_list)
        rb_arr = np.stack(rb_list)
        s, mask, outliers = _corner_stats(lt_arr, rb_arr, outlier_std)
        stats = {
            "lt": {"mean": s[0], "std": s[1]},
            "rb": {"mean": s[2], "std": s[3]},
        }
        Rs = [R for R, m in zip(Rs, mask) if m]
        ts = [t for t, m in zip(ts, mask) if m]
        valid_paths = [p for p, m in zip(valid_paths, mask) if m]
        lt_arr = lt_arr[mask]
        rb_arr = rb_arr[mask]
        _plot_corner_distribution(lt_arr, rb_arr, visualize)
    if logger:
        logger.info("Extracted %d Charuco poses", len(Rs))
    return (Rs, ts, valid_paths, image_paths), (stats, outliers)


@dataclass
class CharucoCalibrator:
    """Charuco board calibration using OpenCV."""

    board: cv2.aruco_CharucoBoard
    dictionary: cv2.aruco_Dictionary
    logger: LoggerType = field(
        default_factory=lambda: Logger.get_logger("calibration.charuco")
    )
    all_corners: List[np.ndarray] = field(default_factory=list, init=False)
    all_ids: List[np.ndarray] = field(default_factory=list, init=False)
    img_size: tuple[int, int] | None = field(default=None, init=False)

    def add_frame(self, img: np.ndarray) -> bool:
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

    def calibrate(self) -> dict[str, np.ndarray | float]:
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

    def save(
        self,
        saver: CalibrationSaver,
        filename: str,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> None:
        saver.save(filename, camera_matrix, dist_coeffs)
        self.logger.info(
            f"Calibration saved with {saver.__class__.__name__} to {filename}"
        )
