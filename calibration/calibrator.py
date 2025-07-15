from __future__ import annotations
import contextlib
import io
import random

"""Core calibration routines for intrinsics and hand-eye."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from utils import (
    HAND_EYE_METHODS,
    HAND_EYE_MAP,
    JSONPoseLoader,
    paths,
    IMAGE_EXT,
    DEPTH_SCALE,
)
from .handeye import calibrate_handeye_svd
from utils.logger import CaptureStderrToLogger, Logger, LoggerType
from utils.error_tracker import ErrorTracker
from .metrics import handeye_errors, svd_transform

from .pattern import CalibrationPattern, PatternDetection
from .utils import save_camera_params, save_transform, timestamp
from utils.cloud_utils import load_depth
from utils.geometry import (
    TransformUtils,
    load_extrinsics,
    estimate_board_points_3d,
)
from .visualizer import plot_poses, plot_reprojection_errors
from utils.geometry import rotation_angle
from utils.settings import DEFAULT_DEPTH_INTRINSICS, DEPTH_EXT
import utils.settings as settings


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
            intr_viz_file = paths.VIZ_DIR / f"{out_base.stem}_reproj{IMAGE_EXT}"

            plot_reprojection_errors(
                per_view, intr_viz_file, interactive=settings.DEFAULT_INTERACTIVE
            )
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

    def _estimate_target_pose(
        self,
        pattern: CalibrationPattern,
        detection: PatternDetection,
        depth_map: np.ndarray,
        K_rgb: np.ndarray,
        dist: np.ndarray,
        K_depth: np.ndarray | None,
        R_d2rgb: np.ndarray | None,
        t_d2rgb: np.ndarray | None,
        use_extrinsics: bool,
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """
        Estimate the pose of the pattern (any type), prioritizing 3D reconstruction from depth.
        Fallback to 2D PnP only if necessary.
        """
        pts_cam = None
        # Try to estimate 3D points using depth, if possible
        if use_extrinsics and R_d2rgb is not None and t_d2rgb is not None:
            pts_cam = estimate_board_points_3d(
                detection.corners,
                depth_map,
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

        if pts_cam is not None and len(pts_cam) >= 4:
            try:
                return svd_transform(detection.object_points, pts_cam)
            except Exception as exc:
                self.logger.warning(f"SVD failed: {exc}; falling back to PnP")
        # Fallback to PnP if 3D reconstruction fails
        return pattern.estimate_pose(detection, K_rgb, dist)

    def _depth_for_image(self, img_path: Path) -> Path | None:
        """Return associated depth file for ``img_path`` or ``None``."""
        base = img_path.stem.replace("_rgb", "")
        candidates = list(img_path.parent.glob(f"{base}*_depth{DEPTH_EXT}"))
        if candidates:
            return candidates[0]
        fallback = img_path.parent / f"{base}_depth{DEPTH_EXT}"
        return fallback if fallback.exists() else None

    def _gather_target_poses(
        self,
        images: List[Path],
        pattern: CalibrationPattern,
        intrinsics_rgb: tuple[np.ndarray, np.ndarray],
        robot_Rs_all: List[np.ndarray],
        robot_ts_all: List[np.ndarray],
        K_depth: np.ndarray | None = None,
        use_extrinsics: bool = True,
    ) -> tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Collect board and robot poses for successful detections.
        Returns also: warnings, detected count, not_detected count
        """
        K_rgb, dist = intrinsics_rgb
        filtered_robot_Rs: List[np.ndarray] = []
        filtered_robot_ts: List[np.ndarray] = []
        targets_R: List[np.ndarray] = []
        targets_t: List[np.ndarray] = []

        if use_extrinsics:
            R_d2rgb, t_d2rgb = load_extrinsics(Path("d415_extr.json"), "depth", "rgb")
        visualize_path = random.choice(images) if self.visualize else None

        warnings = []
        detected = 0
        not_detected = 0
        for idx, img_path in enumerate(Logger.progress(images, desc="hand-eye")):
            if idx >= len(robot_Rs_all):
                break
            relative_path = Path(img_path).resolve().relative_to(Path.cwd())
            img = cv2.imread(str(img_path))
            if img is None:
                warnings.append(f"Failed to read {img_path}")
                not_detected += 1
                continue

            detection = pattern.detect(img, visualize=(img_path == visualize_path))
            if detection is None:
                warnings.append(f"Pattern not detected in {relative_path}")
                not_detected += 1
                continue

            depth_path = self._depth_for_image(img_path)
            if depth_path is None:
                warnings.append(f"No depth file found for {relative_path}")
                not_detected += 1
                continue
            depth = load_depth(str(depth_path))

            pose = self._estimate_target_pose(
                pattern,
                detection,
                depth,
                K_rgb,
                dist,
                K_depth,
                R_d2rgb if use_extrinsics else None,
                t_d2rgb if use_extrinsics else None,
                use_extrinsics,
            )
            if pose is None:
                warnings.append(f"Pose estimation failed for {img_path}")
                not_detected += 1
                continue
            R, t = pose

            filtered_robot_Rs.append(robot_Rs_all[idx])
            filtered_robot_ts.append(robot_ts_all[idx])
            targets_R.append(R)
            targets_t.append(t)
            detected += 1
        total = detected + not_detected
        self.logger.info(
            f"Detected pattern in {detected}/{total} images "
            f"({not_detected} not detected, {detected/total:.1%} success rate)"
        )
        if warnings:
            self.logger.info("--- Detection Warnings ---")
            for w in warnings:
                self.logger.warning(w)
        return filtered_robot_Rs, filtered_robot_ts, targets_R, targets_t

    def _camera_poses(
        self,
        robot_Rs: List[np.ndarray],
        robot_ts: List[np.ndarray],
        R_cam2tool: np.ndarray,
        t_cam2tool: np.ndarray,
        *,
        invert: bool = False,
    ) -> tuple[List[np.ndarray], List[np.ndarray]]:
        """Return camera or tool poses depending on ``invert`` flag."""
        cam_Rs: List[np.ndarray] = []
        cam_ts: List[np.ndarray] = []
        T_tool = TransformUtils.build_transform(R_cam2tool, t_cam2tool)
        T_tool_inv = np.linalg.inv(T_tool)
        for Rg, tg in zip(robot_Rs, robot_ts):
            T_base = TransformUtils.build_transform(Rg, tg)
            T = T_tool @ T_base if invert else T_base @ T_tool_inv
            cam_Rs.append(T[:3, :3])
            cam_ts.append(T[:3, 3])
        return cam_Rs, cam_ts

    def _run_method(
        self,
        method: int | str,
        name: str,
        robot_Rs: List[np.ndarray],
        robot_ts: List[np.ndarray],
        target_Rs: List[np.ndarray],
        target_ts: List[np.ndarray],
        out_base: Path,
        viz_dir: Path,
    ) -> tuple[str, float, float, float, float]:
        """Execute one hand-eye solver and compute error metrics."""

        if name == "svd":
            R_cam2tool, t_cam2tool = calibrate_handeye_svd(
                robot_Rs, robot_ts, target_Rs, target_ts
            )
        else:
            with CaptureStderrToLogger(self.logger):
                R_cam2tool, t_cam2tool = cv2.calibrateHandEye(
                    robot_Rs, robot_ts, target_Rs, target_ts, method=method
                )
        base_stem = f"{out_base.stem}_{name}"
        base_out = out_base.with_stem(base_stem)
        T = TransformUtils.build_transform(R_cam2tool, t_cam2tool)
        save_transform(base_out, T)
        cam_Rs, cam_ts = self._camera_poses(robot_Rs, robot_ts, R_cam2tool, t_cam2tool)
        robot_Rss, robot_tss = self._camera_poses(
            robot_Rs, robot_ts, R_cam2tool, t_cam2tool, invert=True
        )
        rot_err, trans_err = handeye_errors(
            robot_Rs, robot_ts, target_Rs, target_ts, R_cam2tool, t_cam2tool
        )
        t_rmse = float(np.sqrt(np.mean(trans_err**2)))
        r_rmse = float(np.sqrt(np.mean(rot_err**2)))
        trans_errors = np.linalg.norm(np.array(robot_tss) - np.array(cam_ts), axis=1)
        rot_errors = np.array(
            [rotation_angle(Rr.T @ Rc) for Rr, Rc in zip(robot_Rss, cam_Rs)]
        )
        align_t_rmse = float(np.sqrt(np.mean(trans_errors**2)))
        align_r_rmse = float(np.sqrt(np.mean(rot_errors**2)))
        if self.visualize:
            plot_file = viz_dir / f"{base_stem}_poses{IMAGE_EXT}"
            plot_poses(robot_Rs, robot_ts, cam_Rs, cam_ts, plot_file)
        return name, t_rmse, r_rmse, align_t_rmse, align_r_rmse

    def _log_summary(
        self, summary: list[tuple[str, float, float, float, float]], out_base: Path
    ) -> None:
        """Log results for all methods."""
        self.logger.info("Summary of hand-eye calibration and pose alignment errors:")
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

    def _run_all_methods(
        self,
        methods_to_run: list[tuple[int | str, str]],
        robot_Rs: List[np.ndarray],
        robot_ts: List[np.ndarray],
        target_Rs: List[np.ndarray],
        target_ts: List[np.ndarray],
        out_base: Path,
        viz_dir: Path,
    ) -> list[tuple[str, float, float, float, float]]:
        """Execute each calibration method and collect results."""
        summary = []
        for method, method_name in methods_to_run:
            self.logger.info(f"Running hand-eye method: {method_name.upper()}")
            self.logger.debug(
                f"robot_Rs: {len(robot_Rs)}, robot_ts: {len(robot_ts)}, "
                f"target_Rs: {len(target_Rs)}, target_ts: {len(target_ts)}"
            )
            result = self._run_method(
                method,
                method_name,
                robot_Rs,
                robot_ts,
                target_Rs,
                target_ts,
                out_base,
                viz_dir,
            )
            summary.append(result)
        return summary

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
            robot_Rs_all, robot_ts_all = JSONPoseLoader.load_poses(str(poses_file))
            K_rgb, dist = intrinsics
            K_depth = DEFAULT_DEPTH_INTRINSICS
            robot_Rs, robot_ts, target_Rs, target_ts = self._gather_target_poses(
                images,
                pattern,
                (K_rgb, dist),
                robot_Rs_all,
                robot_ts_all,
                K_depth=K_depth,
                use_extrinsics=True,
            )
            if not target_Rs:
                exc = RuntimeError("No valid detections for hand-eye calibration")
                ErrorTracker.report(exc)
                return

            handeye_viz_dir = paths.VIZ_DIR
            out_base = paths.RESULTS_DIR / f"handeye_{timestamp()}"

            if self.method == "all":
                methods_to_run = [*HAND_EYE_METHODS]
            else:
                method_const = HAND_EYE_MAP.get(self.method)
                if method_const is None:
                    raise ValueError(f"Unknown hand-eye method: {self.method}")
                methods_to_run = [(method_const, self.method)]
            summary = self._run_all_methods(
                methods_to_run,
                robot_Rs,
                robot_ts,
                target_Rs,
                target_ts,
                out_base,
                handeye_viz_dir,
            )
            self._log_summary(summary, out_base)
            return out_base
        except Exception as exc:
            self.logger.error(f"Hand-eye calibration failed: {exc}")
            ErrorTracker.report(exc)
        finally:
            pattern.clear()
