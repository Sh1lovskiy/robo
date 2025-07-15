from __future__ import annotations
import random

"""Core calibration routines for intrinsics and hand-eye."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from utils import (
    HAND_EYE_METHODS,
    HAND_EYE_MAP,
    JSONPoseLoader,
    handeye as handeye_cfg,
    paths,
    IMAGE_EXT,
    DEPTH_SCALE,
)
from .handeye import calibrate_handeye_svd
from utils.logger import Logger, LoggerType
from utils.error_tracker import ErrorTracker
from utils.transform import TransformUtils
from .metrics import handeye_errors, svd_transform

from .pattern import CalibrationPattern
from .utils import save_camera_params, save_transform, timestamp
from utils.cloud_utils import load_depth
from utils.geometry import load_extrinsics, estimate_board_points_3d
from .visualizer import plot_poses, plot_reprojection_errors, _rotation_angle


@dataclass
class IntrinsicResult:
    """Result of an intrinsic calibration."""

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    output_base: Path


@dataclass
class IntrinsicCalibrator:
    """Compute camera intrinsics using a generic calibration pattern."""

    logger: LoggerType = field(
        default_factory=lambda: Logger.get_logger("calibration.intrinsic")
    )

    def calibrate(
        self, images: List[Path], pattern: CalibrationPattern
    ) -> IntrinsicResult:
        """
        Calibrate camera intrinsics from a set of images.

        The provided ``pattern`` is used to detect 2-D/3-D correspondences in
        all input images which are then passed to OpenCV's camera calibration
        routines.  Per-image reprojection RMSE values are computed for
        diagnostics and the resulting parameters are stored on disk.
        """
        self.logger.info("Starting intrinsic calibration")
        try:
            img_size: Tuple[int, int] | None = None
            for img_path in Logger.progress(images, desc="intrinsic"):
                img = cv2.imread(str(img_path))
                if img is None:
                    self.logger.error(f"Failed to read {img_path}")
                    continue
                img_size = (img.shape[1], img.shape[0])
                if pattern.detect(img) is None:
                    self.logger.warning(f"Pattern not detected in {img_path}")
            if not pattern.detections or img_size is None:
                return ErrorTracker.report(
                    RuntimeError("No valid detections for intrinsic calibration")
                )
            K, dist, rms, per_view = pattern.calibrate_camera(img_size)
            out_base = paths.RESULTS_DIR / f"camera_{timestamp()}"
            save_camera_params(out_base, K, dist, rms)
            viz_file = paths.CAPTURES_DIR / "viz" / f"{out_base.stem}_reproj{IMAGE_EXT}"

            plot_reprojection_errors(per_view, viz_file)
            self.logger.info(
                f"Intrinsics saved to {out_base.relative_to(Path.cwd())} (RMS={rms:.6f})"
            )
            return IntrinsicResult(K, dist, out_base)
        except Exception as exc:
            self.logger.error(f"Intrinsic calibration failed: {exc}")
            ErrorTracker.report(exc)


class HandEyeCalibrator:
    """Solve the robot-to-camera transform using selected method."""

    def __init__(
        self,
        method: str = "svd",
        visualize: bool = False,
        logger: LoggerType | None = None,
    ) -> None:
        self.method = method
        self.visualize = visualize
        self.logger = logger or Logger.get_logger("calibration.handeye")

    def _gather_target_poses(
        self,
        images: List[Path],
        pattern: CalibrationPattern,
        intrinsics_rgb: tuple[np.ndarray, np.ndarray],
        K_depth: np.ndarray | None = None,
        use_extrinsics: bool = True,
    ) -> tuple[List[np.ndarray], List[np.ndarray]]:
        K_rgb, dist = intrinsics_rgb
        targets_R: List[np.ndarray] = []
        targets_t: List[np.ndarray] = []
        visualize_path = random.choice(images)

        if use_extrinsics:
            R_d2rgb, t_d2rgb = load_extrinsics(Path("d415_extr.json"), "depth", "rgb")

        for img_path in Logger.progress(images, desc="hand-eye"):
            img = cv2.imread(str(img_path))
            if img is None:
                self.logger.error(f"Failed to read {img_path}")
                continue

            detection = pattern.detect(img)
            if detection is None:
                self.logger.warning(f"Pattern not detected in {img_path}")
                continue

            depth_path = img_path.parent / f"{img_path.stem.replace('_rgb', '')}_depth.npy"
            depth = load_depth(str(depth_path))

            if use_extrinsics:
                pts_cam = estimate_board_points_3d(
                    detection.corners,
                    depth,
                    detection.object_points,
                    K_rgb,
                    dist,
                    K_depth,
                    R_d2rgb,
                    t_d2rgb,
                    depth_scale=DEPTH_SCALE,
                )
            else:
                pts_cam = None

            if pts_cam is not None:
                try:
                    R, t = svd_transform(detection.object_points, pts_cam)
                except Exception as exc:
                    self.logger.warning(f"SVD failed: {exc}; falling back to PnP")
                    pose = pattern.estimate_pose(detection, K_rgb, dist)
                    if pose is None:
                        continue
                    R, t = pose
            else:
                pose = pattern.estimate_pose(detection, K_rgb, dist)
                if pose is None:
                    self.logger.warning(f"Pose estimation failed for {img_path}")
                    continue
                R, t = pose

            targets_R.append(R)
            targets_t.append(t)

        return targets_R, targets_t

    def _compute_camera_poses(
        self,
        robot_Rs: List[np.ndarray],
        robot_ts: List[np.ndarray],
        R_cam2tool: np.ndarray,
        t_cam2tool: np.ndarray,
    ) -> tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Return camera poses in the robot base frame.

        Parameters
        ----------
        robot_Rs, robot_ts
            Absolute robot poses :math:`\{{^{b}R_{g_i}},\ {^{b}t_{g_i}}\}` with
            respect to the robot base.
        R_cam2tool, t_cam2tool
            Transformation from the camera frame to the robot tool frame
            obtained from hand-eye calibration.

        Returns
        -------
        List[np.ndarray], List[np.ndarray]
            Rotations and translations of the camera expressed in the robot
            base frame for each measurement.
        """
        cam_Rs: List[np.ndarray] = []
        cam_ts: List[np.ndarray] = []
        T_tool = TransformUtils.build_transform(R_cam2tool, t_cam2tool)
        for Rg, tg in zip(robot_Rs, robot_ts):
            T_base = TransformUtils.build_transform(Rg, tg)
            T_cam = T_base @ np.linalg.inv(T_tool)
            cam_Rs.append(T_cam[:3, :3])
            cam_ts.append(T_cam[:3, 3])
        return cam_Rs, cam_ts

    def _compute_camera_poses_inv(
        self,
        robot_Rs: List[np.ndarray],
        robot_ts: List[np.ndarray],
        R_cam2tool: np.ndarray,
        t_cam2tool: np.ndarray,
    ) -> tuple[List[np.ndarray], List[np.ndarray]]:
        """Return robot tool poses expressed in the camera frame."""
        cam_Rs: List[np.ndarray] = []
        cam_ts: List[np.ndarray] = []
        T_tool = TransformUtils.build_transform(R_cam2tool, t_cam2tool)
        for Rg, tg in zip(robot_Rs, robot_ts):
            T_base = TransformUtils.build_transform(Rg, tg)
            T_cam = T_tool @ T_base
            cam_Rs.append(T_cam[:3, :3])
            cam_ts.append(T_cam[:3, 3])
        return cam_Rs, cam_ts

    def calibrate(
        self,
        poses_file: Path,
        images: List[Path],
        pattern: CalibrationPattern,
        intrinsics: tuple[np.ndarray, np.ndarray],
    ) -> Path:
        """Compute hand-eye transformation from poses and images."""
        self.logger.info("Starting hand-eye calibration")
        try:
            robot_Rs, robot_ts = JSONPoseLoader.load_poses(str(poses_file))
            K_rgb, dist = intrinsics
            K_depth = np.array(
                [[616.365, 0, 318.268], [0, 616.202, 243.215], [0, 0, 1]]
            )  # TODO import from  camera
            target_Rs, target_ts = self._gather_target_poses(
                images, pattern, (K_rgb, dist), K_depth=K_depth, use_extrinsics=True
            )
            if not target_Rs:
                exc = RuntimeError("No valid detections for hand-eye calibration")
                ErrorTracker.report(exc)
                return

            viz_dir = paths.CAPTURES_DIR / "viz"
            out_base = paths.RESULTS_DIR / f"handeye_{timestamp()}"

            if self.method == "all":
                methods_to_run = [*HAND_EYE_METHODS]
            else:
                method_const = HAND_EYE_MAP.get(self.method)
                if method_const is None:
                    raise ValueError(f"Unknown hand-eye method: {self.method}")
                methods_to_run = [(method_const, self.method)]
            summary = []
            for method, method_name in methods_to_run:
                self.logger.info(f"Running hand-eye method: {method_name.upper()}")

                if method_name == "svd":
                    R_cam2tool, t_cam2tool = calibrate_handeye_svd(
                        robot_Rs, robot_ts, target_Rs, target_ts
                    )
                else:
                    R_cam2tool, t_cam2tool = cv2.calibrateHandEye(
                        robot_Rs, robot_ts, target_Rs, target_ts, method=method
                    )

                base_stem = f"{out_base.stem}_{method_name}"
                base_out = out_base.with_stem(base_stem)

                T = TransformUtils.build_transform(R_cam2tool, t_cam2tool)
                save_transform(base_out, T)

                # cam_Rs, cam_ts = self._compute_camera_poses(
                #     target_Rs, target_ts, R_cam2tool, t_cam2tool
                # )
                cam_Rs, cam_ts = self._compute_camera_poses(
                    robot_Rs, robot_ts, R_cam2tool, t_cam2tool
                )
                robot_Rss, robot_tss = self._compute_camera_poses_inv(
                    robot_Rs, robot_ts, R_cam2tool, t_cam2tool
                )

                rot_err, trans_err = handeye_errors(
                    robot_Rs, robot_ts, target_Rs, target_ts, R_cam2tool, t_cam2tool
                )

                t_rmse = float(np.sqrt(np.mean(trans_err**2)))
                r_rmse = float(np.sqrt(np.mean(rot_err**2)))

                trans_errors = np.linalg.norm(
                    np.array(robot_tss) - np.array(cam_ts), axis=1
                )
                rot_errors = np.array(
                    [_rotation_angle(Rr.T @ Rc) for Rr, Rc in zip(robot_Rss, cam_Rs)]
                )
                align_t_rmse = float(np.sqrt(np.mean(trans_errors**2)))
                align_r_rmse = float(np.sqrt(np.mean(rot_errors**2)))

                summary.append(
                    (method_name, t_rmse, r_rmse, align_t_rmse, align_r_rmse)
                )

                if self.visualize:
                    plot_file = viz_dir / f"{base_stem}_poses{IMAGE_EXT}"
                    plot_poses(robot_Rs, robot_ts, cam_Rs, cam_ts, plot_file)

            self.logger.info(
                "Summary of hand-eye calibration and pose alignment errors:"
            )
            self.logger.info(
                "Method     | T. RMSE [m] | R. RMSE [deg] | Align T. RMSE [m] | Align R. RMSE [deg]"
            )
            self.logger.info(
                "-----------|-------------|---------------|-------------------|--------------------"
            )
            for name, t_rmse, r_rmse, a_t_rmse, a_r_rmse in summary:
                self.logger.info(
                    f"{name:<10} |  {t_rmse:>9.6f}  |  {r_rmse:>11.4f}  |  {a_t_rmse:>15.6f}  |  {a_r_rmse:>17.4f}"
                )

            self.logger.info(
                f"Hand-eye results saved to {out_base.relative_to(Path.cwd())}"
            )
            return out_base
        except Exception as exc:
            self.logger.error(f"Hand-eye calibration failed: {exc}")
            ErrorTracker.report(exc)
        finally:
            pattern.clear()
