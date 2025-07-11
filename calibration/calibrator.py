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
)
from .handeye import calibrate_handeye_svd
from utils.logger import Logger, LoggerType
from utils.error_tracker import ErrorTracker
from utils.transform import TransformUtils
from .metrics import handeye_errors, svd_transform

from .pattern import CalibrationPattern
from .utils import save_camera_params, save_transform, timestamp
from .extractor import load_depth, board_points_from_depth
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
        """Calibrate camera intrinsics from images."""
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
        K: np.ndarray,
        dist: np.ndarray,
    ) -> tuple[List[np.ndarray], List[np.ndarray]]:
        targets_R: List[np.ndarray] = []
        targets_t: List[np.ndarray] = []

        # Visualize only one random image
        visualize_path = random.choice(images)
        for img_path in Logger.progress(images, desc="hand-eye"):
            img = cv2.imread(str(img_path))
            if img is None:
                self.logger.error(f"Failed to read {img_path}")
                continue
            detection = pattern.detect(img, visualize=(img_path == visualize_path))
            if detection is None:
                self.logger.warning(f"Pattern not detected in {img_path}")
                continue
            depth = load_depth(img_path)
            pts_cam = board_points_from_depth(detection.corners, depth, K)
            if pts_cam is not None:
                try:
                    R, t = svd_transform(detection.object_points, pts_cam)
                except Exception as exc:
                    self.logger.warning(
                        f"Depth pose failed for {img_path}: {exc}; using PnP"
                    )
                    pose = pattern.estimate_pose(detection, K, dist)
                    if pose is None:
                        continue
                    R, t = pose
            else:
                pose = pattern.estimate_pose(detection, K, dist)
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
        cam_Rs: List[np.ndarray] = []
        cam_ts: List[np.ndarray] = []
        T_tool = TransformUtils.build_transform(R_cam2tool, t_cam2tool)
        for Rg, tg in zip(robot_Rs, robot_ts):
            T_base = TransformUtils.build_transform(Rg, tg)
            T_cam = T_base @ T_tool
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
            K, dist = intrinsics
            robot_Rs, robot_ts = JSONPoseLoader.load_poses(str(poses_file))
            target_Rs, target_ts = self._gather_target_poses(images, pattern, K, dist)
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

                cam_Rs, cam_ts = self._compute_camera_poses(
                    robot_Rs, robot_ts, R_cam2tool, t_cam2tool
                )

                rot_err, trans_err = handeye_errors(
                    robot_Rs, robot_ts, target_Rs, target_ts, R_cam2tool, t_cam2tool
                )

                t_rmse = float(np.sqrt(np.mean(trans_err**2)))
                r_rmse = float(np.sqrt(np.mean(rot_err**2)))

                trans_errors = np.linalg.norm(
                    np.array(robot_ts) - np.array(cam_ts), axis=1
                )
                rot_errors = np.array(
                    [_rotation_angle(Rr.T @ Rc) for Rr, Rc in zip(robot_Rs, cam_Rs)]
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
