from __future__ import annotations

"""Hand-eye calibration pipeline comparing multiple pose strategies."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from utils import HAND_EYE_METHODS, JSONPoseLoader, paths
from utils.logger import Logger, LoggerType, CaptureStderrToLogger
from utils.error_tracker import ErrorTracker
from utils.settings import (
    EXTR_COLOR_TO_DEPTH_ROT,
    EXTR_COLOR_TO_DEPTH_TRANS,
    INTRINSICS_DEPTH_MATRIX,
    EXTR_DEPTH_TO_COLOR_ROT,
    EXTR_DEPTH_TO_COLOR_TRANS,
    DEPTH_SCALE,
    DEPTH_EXT,
)
from utils.geometry import TransformUtils, rotation_angle
from .pose_utils import (
    get_3d_points_from_depth,
    solve_pnp_obj_to_img,
    rigid_transform_3D,
)
from utils.cloud_utils import load_depth
from .metrics import handeye_errors
from .pattern import CalibrationPattern
from .utils import save_transform, timestamp
from .visualizer import plot_poses


def rigid_transform_3D(A, B):
    # A, B: Nx3 (correspondence)
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R_ = Vt.T @ U.T
    if np.linalg.det(R_) < 0:
        Vt[-1, :] *= -1
        R_ = Vt.T @ U.T
    t_ = centroid_B - R_ @ centroid_A
    return R_, t_


def solve_pnp_obj_to_3d(model_points, points_3d):
    # 3D-3D (Procrustes)
    mask = ~np.isnan(points_3d).any(axis=1)
    if np.sum(mask) < 4:
        return None, None
    R_, t_ = rigid_transform_3D(model_points[mask], points_3d[mask])
    return R_, t_


@dataclass
class HandEyeComparison:
    """Run hand-eye calibration for several pose estimation variants."""

    visualize: bool = False
    logger: LoggerType = field(
        default_factory=lambda: Logger.get_logger("calibration.comparison")
    )

    # ------------------------------------------------------------------
    # Pose estimation helpers
    # ------------------------------------------------------------------
    def _pose_pnp(
        self, obj: np.ndarray, img: np.ndarray, K: np.ndarray, dist: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Estimate pose using OpenCV PnP."""
        return solve_pnp_obj_to_img(obj, img, K, dist)

    def _pose_svd(
        self, model: np.ndarray, meas: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Estimate pose by aligning two 3-D point sets."""
        try:
            return rigid_transform_3D(model, meas)
        except Exception as exc:  # pragma: no cover - numerical issues
            self.logger.warning(f"SVD failed: {exc}")
            return None

    def _depth_points(
        self,
        corners: np.ndarray,
        depth: np.ndarray,
        K_rgb: np.ndarray,
        K_depth: np.ndarray,
    ) -> np.ndarray:
        R = np.asarray(EXTR_COLOR_TO_DEPTH_ROT, dtype=np.float64)
        t = np.asarray(EXTR_COLOR_TO_DEPTH_TRANS, dtype=np.float64)
        self.logger.info(
            f"[_depth_points] Input: corners shape={corners.shape}, depth shape={depth.shape}"
        )
        pts_3d = get_3d_points_from_depth(
            corners,
            depth,
            K_rgb,
            K_depth,
            R,
            t,
            depth_scale=DEPTH_SCALE,
            logger=self.logger,
        )
        self.logger.info(
            f"[_depth_points] Output: 3D points shape={pts_3d.shape}, "
            f"sample: {pts_3d[:2]}"
        )
        return pts_3d

    def _camera_poses(
        self,
        robot_Rs: List[np.ndarray],
        robot_ts: List[np.ndarray],
        R: np.ndarray,
        t: np.ndarray,
        invert: bool = False,
    ) -> tuple[List[np.ndarray], List[np.ndarray]]:
        cam_Rs, cam_ts = [], []
        T_tool = TransformUtils.build_transform(R, t)
        T_inv = np.linalg.inv(T_tool)
        for Rg, tg in zip(robot_Rs, robot_ts):
            T_base = TransformUtils.build_transform(Rg, tg)
            T = T_tool @ T_base if invert else T_base @ T_inv
            cam_Rs.append(T[:3, :3])
            cam_ts.append(T[:3, 3])
        return cam_Rs, cam_ts

    def _run_method(
        self,
        method: int,
        name: str,
        robot_Rs: List[np.ndarray],
        robot_ts: List[np.ndarray],
        target_Rs: List[np.ndarray],
        target_ts: List[np.ndarray],
        out_base: Path,
    ) -> tuple[str, float, float, float, float]:
        if len(robot_Rs) < 3 or len(target_Rs) < 3:
            self.logger.warning(
                f"Skipping {name} — too few pose pairs: {len(robot_Rs)}"
            )
            return name, np.nan, np.nan, np.nan, np.nan

        for i, R in enumerate(target_Rs):
            det = np.linalg.det(R)
            if abs(det) < 1e-4:
                self.logger.warning(
                    f"target_R[{i}] has near-zero determinant: {det:.6f}"
                )

        with CaptureStderrToLogger(self.logger):
            try:
                R_ct, t_ct = cv2.calibrateHandEye(
                    robot_Rs, robot_ts, target_Rs, target_ts, method=method
                )
            except cv2.error as e:
                self.logger.warning(f"{name} failed: {e}")
                return name, np.nan, np.nan, np.nan, np.nan  # gracefully skip

        out_base.mkdir(parents=True, exist_ok=True)
        save_transform(out_base / name, TransformUtils.build_transform(R_ct, t_ct))
        cam_Rs, cam_ts = self._camera_poses(robot_Rs, robot_ts, R_ct, t_ct)
        robot_Rss, robot_tss = self._camera_poses(robot_Rs, robot_ts, R_ct, t_ct, True)

        rot_err, trans_err = handeye_errors(
            robot_Rs, robot_ts, target_Rs, target_ts, R_ct, t_ct
        )
        t_rmse = float(np.sqrt(np.mean(trans_err**2)))
        r_rmse = float(np.sqrt(np.mean(rot_err**2)))
        trans = np.linalg.norm(np.array(robot_tss) - np.array(cam_ts), axis=1)
        rot = [rotation_angle(Rr.T @ Rc) for Rr, Rc in zip(robot_Rss, cam_Rs)]
        a_t_rmse = float(np.sqrt(np.mean(trans**2)))
        a_r_rmse = float(np.sqrt(np.mean(np.square(rot))))

        if self.visualize:
            plot_file = paths.VIZ_DIR / f"{out_base.stem}_{name}_poses.png"
            plot_poses(robot_Rs, robot_ts, cam_Rs, cam_ts, plot_file)

        return name, t_rmse, r_rmse, a_t_rmse, a_r_rmse

    def _log_summary(
        self, summary: List[tuple[str, float, float, float, float]], out_dir: Path
    ) -> None:
        lines = [
            "Summary of hand-eye calibration and pose alignment errors:",
            "Method     | T. RMSE [m] | R. RMSE [deg] | Align T. RMSE [m] | Align R. RMSE [deg]",
            "-----------|-------------|---------------|-------------------|-------------------",
        ]
        for n, t1, r1, t2, r2 in summary:
            lines.append(
                f"{n:<10} |  {t1:>9.6f}  |  {r1:>11.4f}  |  {t2:>15.6f}  |  {r2:>17.4f}"
            )
        lines.append(f"Results saved to {out_dir.relative_to(Path.cwd())}")
        self.logger.info("\n".join(lines))

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------
    def _collect(
        self,
        images: List[Path],
        pattern: CalibrationPattern,
        intrinsics: tuple[np.ndarray, np.ndarray],
        robot_Rs: List[np.ndarray],
        robot_ts: List[np.ndarray],
    ) -> Dict[str, Dict[str, List[np.ndarray]]]:
        K_rgb, dist = intrinsics
        K_depth = np.asarray(INTRINSICS_DEPTH_MATRIX, dtype=np.float64)
        variants = {
            "pnp": {"robot_Rs": [], "robot_ts": [], "R": [], "t": []},
            "svd": {"robot_Rs": [], "robot_ts": [], "R": [], "t": []},
            "pnp_depth": {"robot_Rs": [], "robot_ts": [], "R": [], "t": []},
        }
        for idx, img_path in enumerate(Logger.progress(images, desc="detect")):
            if idx >= len(robot_Rs):
                break
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            detection = pattern.detect(img, visualize=False)
            if detection is None:
                continue
            corners = detection.corners  # Nx2
            object_points = detection.object_points
            depth_file = img_path.with_suffix(DEPTH_EXT)
            if not depth_file.exists():
                continue
            depth = load_depth(str(depth_file))
            # Получаем 3D точки — порядок совпадает с detection.corners/object_points
            depth_pts = self._depth_points(detection.corners, depth, K_rgb, K_depth)
            mask = ~np.isnan(depth_pts).any(axis=1)
            if np.sum(mask) < 4:
                continue
            # Согласованные пары:
            corners_valid = corners[mask]  # Kx2
            self.logger.info(
                f"[{img_path.name}] First valid point: corner RGB={corners_valid[0]}, "
                f"object_point={object_points_valid[0]}, depth_3d={depth_pts_valid[0]}"
            )
            object_points_valid = object_points[mask]  # Kx3
            depth_pts_valid = depth_pts[mask]  # Kx3
            if (
                corners_valid.shape[0] != object_points_valid.shape[0]
                or corners_valid.shape[0] != depth_pts_valid.shape[0]
            ):
                self.logger.warning("Corner/depth correspondence mismatch")
                continue
            self.logger.debug(
                f"[{img_path.name}] valid points: {corners_valid.shape[0]}"
            )

            # 2D-3D PnP
            R_pnp2d, t_pnp2d = self._pose_pnp(
                object_points_valid, corners_valid, K_rgb, dist
            )
            if R_pnp2d is not None and t_pnp2d is not None:
                object_points_in_rgb = (
                    R_pnp2d @ object_points_valid.T
                ).T + t_pnp2d.flatten()
                variants["pnp"]["robot_Rs"].append(robot_Rs[idx])
                variants["pnp"]["robot_ts"].append(robot_ts[idx])
                variants["pnp"]["R"].append(R_pnp2d)
                variants["pnp"]["t"].append(t_pnp2d)

                # SVD 3D-3D: сопоставь object_points_in_rgb и depth_pts_valid, mask уже учтен!
                pose_svd = rigid_transform_3D(object_points_in_rgb, depth_pts_valid)
                if (
                    pose_svd is not None
                    and pose_svd[0] is not None
                    and pose_svd[1] is not None
                ):
                    variants["svd"]["robot_Rs"].append(robot_Rs[idx])
                    variants["svd"]["robot_ts"].append(robot_ts[idx])
                    variants["svd"]["R"].append(pose_svd[0])
                    variants["svd"]["t"].append(pose_svd[1])

                # Procrustes "наоборот" (pnp_depth)
                pose_pd = solve_pnp_obj_to_3d(depth_pts_valid, object_points_valid)
                if (
                    pose_pd is not None
                    and pose_pd[0] is not None
                    and pose_pd[1] is not None
                ):
                    variants["pnp_depth"]["robot_Rs"].append(robot_Rs[idx])
                    variants["pnp_depth"]["robot_ts"].append(robot_ts[idx])
                    variants["pnp_depth"]["R"].append(pose_pd[0])
                    variants["pnp_depth"]["t"].append(pose_pd[1])
        return variants

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def calibrate(
        self,
        poses_file: Path,
        images: List[Path],
        pattern: CalibrationPattern,
        intrinsics: tuple[np.ndarray, np.ndarray],
    ) -> None:
        self.logger.info("Starting hand-eye comparison")
        try:
            robot_Rs, robot_ts = JSONPoseLoader.load_poses(str(poses_file))
            data = self._collect(images, pattern, intrinsics, robot_Rs, robot_ts)
            for variant, vals in data.items():
                if not vals["R"]:
                    self.logger.warning(f"No valid poses for {variant}")
                    continue
                out_base = paths.RESULTS_DIR / f"handeye_{variant}_{timestamp()}"
                summary = []
                for m, name in HAND_EYE_METHODS[:-1]:
                    self.logger.info(f"Method {name.upper()} on {variant}")
                    res = self._run_method(
                        m,
                        name,
                        vals["robot_Rs"],
                        vals["robot_ts"],
                        vals["R"],
                        vals["t"],
                        out_base,
                    )
                    summary.append(res)
                self._log_summary(summary, out_base)
        except Exception as exc:
            self.logger.error(f"Hand-eye comparison failed: {exc}")
            ErrorTracker.report(exc)
        finally:
            pattern.clear()
