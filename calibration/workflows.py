"""High-level calibration routines for Charuco and hand-eye workflows."""

from __future__ import annotations
import os
from dataclasses import dataclass
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

from calibration.charuco import CharucoCalibrator
from calibration.handeye import HandEyeCalibrator, NPZHandEyeSaver, TxtHandEyeSaver
from calibration.pose_loader import JSONPoseLoader
from utils.config import Config
from utils.io import load_camera_params, save_camera_params_xml, save_camera_params_txt
from utils.logger import Logger, LoggerType
from utils.cli import Command, CommandDispatcher

CHARUCO_DICT_MAP = {"4X4_100": 1, "5X5_50": 4, "5X5_100": 5}


def extract_charuco_poses(
    images_dir,
    board,
    dictionary,
    camera_matrix,
    dist_coeffs,
    logger=None,
    min_corners=4,
    visualize=False,
    debug=False,
    analyze_corners=False,
    outlier_std=2.0,
):
    """
    Extracts Charuco board poses from images.
    If analyze_corners is True, computes stats for left-top and right-bottom board corners (meters),
    visualizes their distribution, and excludes outlier frames (>outlier_std*std from mean).
    Returns: (Rs, ts, valid_img_paths, all_img_paths), (stats, outlier_indices)
    """
    image_paths = sorted(
        [
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )
    Rs, ts, valid_paths = [], [], []
    all_lt, all_rb, all_indices = [], [], []

    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        if img is None:
            if logger:
                logger.warning(f"Cannot read image: {img_path}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
        if ids is None or len(ids) < min_corners:
            continue
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, board
        )
        if (
            charuco_corners is None
            or charuco_ids is None
            or len(charuco_ids) < min_corners
        ):
            continue
        rvec_init = np.zeros((3, 1), dtype=np.float64)
        tvec_init = np.zeros((3, 1), dtype=np.float64)
        try:
            retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners,
                charuco_ids,
                board,
                camera_matrix,
                dist_coeffs,
                rvec_init,
                tvec_init,
            )
        except Exception as e:
            if logger:
                logger.error(f"estimatePoseCharucoBoard error on {img_path}: {repr(e)}")
            continue
        if not retval:
            continue
        R, _ = cv2.Rodrigues(rvec)
        Rs.append(R)
        ts.append(tvec.flatten())
        valid_paths.append(img_path)
        obj_pts = board.getChessboardCorners()
        pos_lt = R @ obj_pts[0].reshape(3, 1) + tvec
        pos_rb = R @ obj_pts[-1].reshape(3, 1) + tvec
        all_lt.append(pos_lt.flatten())
        all_rb.append(pos_rb.flatten())
        all_indices.append(idx)
        if visualize:
            vis = img.copy()
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)
            cv2.drawFrameAxes(vis, camera_matrix, dist_coeffs, rvec, tvec, 0.05)
            cv2.imshow("charuco pose", vis)
            cv2.waitKey(100)
    if visualize:
        cv2.destroyAllWindows()

    corners_stats, outlier_indices = {}, []
    mask_good = np.ones(len(Rs), dtype=bool)

    if analyze_corners and all_lt and all_rb:
        lt = np.stack(all_lt)
        rb = np.stack(all_rb)
        for name, arr in [("lt", lt), ("rb", rb)]:
            corners_stats[name] = {"mean": arr.mean(axis=0), "std": arr.std(axis=0)}
            if logger:
                logger.info(
                    f"{name.upper()} mean [m]: {corners_stats[name]['mean'].round(4)}, std: {corners_stats[name]['std'].round(4)}"
                )
        for arr, stats in [(lt, corners_stats["lt"]), (rb, corners_stats["rb"])]:
            mask_good &= np.all(
                np.abs(arr - stats["mean"]) <= outlier_std * stats["std"], axis=1
            )
        outlier_indices = [
            all_indices[i] for i, good in enumerate(mask_good) if not good
        ]
        if logger and outlier_indices:
            logger.warning(
                f"Frames excluded as outliers: {[os.path.basename(image_paths[i]) for i in outlier_indices]}"
            )
        Rs = [R for i, R in enumerate(Rs) if mask_good[i]]
        ts = [t for i, t in enumerate(ts) if mask_good[i]]
        valid_paths = [p for i, p in enumerate(valid_paths) if mask_good[i]]
        lt = lt[mask_good]
        rb = rb[mask_good]
        plt.figure(figsize=(8, 6))
        plt.scatter(lt[:, 0], lt[:, 1], c="blue", label="Left Top [0]", alpha=0.7)
        plt.scatter(rb[:, 0], rb[:, 1], c="red", label="Right Bottom [-1]", alpha=0.7)
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.title("Charuco Board Corner Positions (Camera frame)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("charuco_corners_distribution.png")
        if visualize:
            plt.show()
        plt.close()
    if logger:
        logger.info(f"Extracted {len(Rs)} Charuco poses after outlier removal")
    return (Rs, ts, valid_paths, image_paths), (corners_stats, outlier_indices)


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
        square_length = cfg.get("square_length", 0.033)
        marker_length = cfg.get("marker_length", 0.025)
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
        square_length = cfg.get("square_length", 0.033)
        marker_length = cfg.get("marker_length", 0.025)
        dict_name = cfg.get("aruco_dict", "5X5_100")
        if dict_name not in CHARUCO_DICT_MAP:
            raise ValueError(f"Unknown ArUco dictionary: {dict_name}")
        dictionary = cv2.aruco.getPredefinedDictionary(CHARUCO_DICT_MAP[dict_name])
        board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y), square_length, marker_length, dictionary
        )

        camera_matrix, dist_coeffs = load_camera_params(charuco_xml)
        Rs_g2b, ts_g2b = JSONPoseLoader.load_poses(robot_poses_file)

        (Rs_t2c, ts_t2c, valid_img_paths, all_img_paths), (stats, outliers) = (
            extract_charuco_poses(
                images_dir,
                board,
                dictionary,
                camera_matrix,
                dist_coeffs,
                logger=self.logger,
                analyze_corners=True,
            )
        )

        # Filter robot poses to only those images that passed outlier rejection
        name2idx = {os.path.basename(p): i for i, p in enumerate(all_img_paths)}
        indices = [name2idx[os.path.basename(p)] for p in valid_img_paths]
        Rs_g2b_f = [Rs_g2b[i] for i in indices]
        ts_g2b_f = [ts_g2b[i] for i in indices]

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
