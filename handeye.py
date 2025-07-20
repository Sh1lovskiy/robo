"""Transparent, modular hand-eye calibration pipeline with full geometric logging and YAML audit."""

from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import json
import sys
import yaml

import cv2
from cv2 import aruco
from scipy.spatial.transform import Rotation as R

from utils.logger import Logger
from utils.settings import EXTR_COLOR_TO_DEPTH_ROT, EXTR_COLOR_TO_DEPTH_TRANS

# =============================================================================
# 1. CONFIGURATION CONSTANTS
# =============================================================================

# --- Paths & Board ---
INTR_FILE = Path("cam_params.yml")
INTRD_FILE = Path("dcam_params.yml")
IMG_DIR = Path("calib/imgs")
POSES_JSON_PATTERN = "*.json"

# --- Charuco Board ---
CHARUCO_DICT = "DICT_5X5_100"
BOARD_SIZE = (8, 5)
SQUARE_LEN = 0.035
MARKER_LEN = 0.026

# --- Depth & Overlay ---
DEPTH_SCALE = 0.0010000000474974513
WINDOW_SIZE = 66
DISABLE_DEPTH_Z = False
OVERLAY_ENABLED = True

# --- Logging & Numeric ---
VERBOSE = True
POSE_ORTHO_TOL = 1e-2
POSE_DET_TOL = 1e-2
MIN_ROT_ELEM = 1e-20
CORNER_RADIUS = 7
OVERLAY_FONT_SCALE = 0.4

np.set_printoptions(suppress=True, precision=6, linewidth=200)

log = Logger.get_logger("handeye")
DICT_NAMES = {
    getattr(aruco, name): name for name in dir(aruco) if name.startswith("DICT_")
}

# =============================================================================
# 2. UTILITIES: VERSIONS, DICTIONARY, YAML OUTPUT
# =============================================================================


def log_versions_and_config(board: aruco.CharucoBoard) -> None:
    import numpy

    log.info(f"Python: {sys.version}")
    log.info(f"OpenCV: {cv2.__version__}, numpy: {numpy.__version__}")
    log.info(
        f"BOARD: {board.getChessboardSize()} squares, "
        f"square_len={board.getSquareLength()}, marker_len={board.getMarkerLength()}, "
        f"dict={get_dict_name(board.getDictionary())}"
    )
    log.info(
        f"PIPELINE: DISABLE_DEPTH_Z={DISABLE_DEPTH_Z}, DEPTH_SCALE={DEPTH_SCALE}, "
        f"OVERLAY_ENABLED={OVERLAY_ENABLED}, VERBOSE={VERBOSE}"
    )


def get_dict_name(dictionary: aruco.Dictionary) -> str:
    for key, name in DICT_NAMES.items():
        if (
            dictionary.bytesList.shape
            == aruco.getPredefinedDictionary(key).bytesList.shape
        ):
            return name
    return "UNKNOWN"


def save_yaml(out_file: Path, data: dict) -> None:
    with open(out_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)
    log.info(f"Saved YAML results to {out_file}")


# =============================================================================
# 3. I/O & CONFIGURATION
# =============================================================================


def load_intrinsics(yaml_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    log.info(f"Loading intrinsics from {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if "camera_matrix" in data:
        K = np.array(data["camera_matrix"]["data"], dtype=float).reshape(3, 3)
        dist = np.array(data["distortion_coefficients"]["data"], dtype=float)
    else:
        fx, fy, cx, cy = data["fx"], data["fy"], data["cx"], data["cy"]
        dist = np.array(data.get("dist", []), dtype=float)
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
    if VERBOSE:
        log.info(f"[intrinsics] K:\n{K}\ndist:{dist}")
    return K, dist


def load_image_pairs(directory: Path) -> List[Tuple[Path, Path]]:
    rgb_files = sorted(directory.glob("frame_*.png"))
    pairs = []
    missing = []
    for rgb in rgb_files:
        dpth = rgb.with_suffix(".npy")
        if dpth.exists():
            pairs.append((rgb, dpth))
        else:
            missing.append(rgb)
    assert not missing, f"Missing depth files for: {[f.name for f in missing]}"
    if VERBOSE:
        log.info(f"[image_pairs] {len(pairs)} pairs, missing: {len(missing)}")
    return pairs


def load_robot_poses(json_path: Path) -> List[np.ndarray]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    poses = []
    for v in data.values():
        x, y, z, rx, ry, rz = v["tcp_coords"]
        t = np.array([x, y, z], dtype=float) / 1000.0
        Rmat = R.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = Rmat
        T[:3, 3] = t
        poses.append(T)
        if VERBOSE:
            log.info(f"[robot_pose] t: {t}, R:\n{Rmat}")
    log.info(f"Loaded {len(poses)} robot poses")
    return poses


def configure_charuco_board(
    board_size: Tuple[int, int], square_len: float, marker_len: float, dict_name: str
) -> Tuple[aruco.CharucoBoard, aruco.ArucoDetector]:
    dict_id = getattr(aruco, dict_name)
    aruco_dict = aruco.getPredefinedDictionary(dict_id)
    board = aruco.CharucoBoard(board_size, square_len, marker_len, aruco_dict)
    params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, params)
    if VERBOSE:
        log.info(
            f"[charuco] Created board {board_size}, sq={square_len}, marker={marker_len}"
        )
    return board, detector


# =============================================================================
# 4. CHARUCO DETECTION
# =============================================================================


def detect_charuco_corners(
    img: np.ndarray, board: aruco.CharucoBoard, detector: aruco.ArucoDetector
) -> Tuple[np.ndarray, np.ndarray] | None:
    marker_corners, marker_ids, _ = detector.detectMarkers(img)
    if marker_ids is None or len(marker_ids) == 0:
        if VERBOSE:
            log.info("[detect_charuco] No Aruco markers detected.")
        return None
    ok, char_corners, char_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners, marker_ids, img, board
    )
    if not ok or char_ids is None or len(char_ids) < 4:
        if VERBOSE:
            log.info(f"[detect_charuco] Charuco failed. ok={ok}, ids={char_ids}")
        return None
    if VERBOSE:
        log.info(f"[detect_charuco] Found {len(char_ids)} corners/ids.")
    return char_corners.squeeze(1), char_ids.flatten()


# =============================================================================
# 5. GEOMETRY & TRANSFORMATIONS
# =============================================================================


def project_rgb_to_depth(
    corners: np.ndarray,
    K_rgb: np.ndarray,
    K_depth: np.ndarray,
    rgb_to_depth: np.ndarray,
) -> np.ndarray:
    K_rgb_inv = np.linalg.inv(K_rgb)
    R = rgb_to_depth[:3, :3]
    t = rgb_to_depth[:3, 3]
    out = []
    for i, (u, v) in enumerate(corners):
        pt_rgb = np.array([u, v, 1.0], dtype=np.float64)
        ray_rgb = K_rgb_inv @ pt_rgb
        if VERBOSE:
            log.info(f"[proj][pt#{i}] RGB pix: {pt_rgb} -> cam ray: {ray_rgb}")
        ray_depth = R @ ray_rgb + t
        ray_depth /= ray_depth[2]
        pt_depth = K_depth @ ray_depth
        if VERBOSE:
            log.info(
                f"[proj][pt#{i}] Depth cam ray: {ray_depth} -> depth pix: {pt_depth}"
            )
        out.append(pt_depth[:2])
    return np.array(out, dtype=np.float64)


def backproject_rgb_corners_to_3d(
    corners: np.ndarray,
    depth_map: np.ndarray,
    K_rgb: np.ndarray,
    K_depth: np.ndarray,
    rgb_to_depth: np.ndarray,
    depth_scale: float,
    window: int = WINDOW_SIZE,
) -> np.ndarray:
    K_rgb_inv = np.linalg.inv(K_rgb)
    R = rgb_to_depth[:3, :3]
    t = rgb_to_depth[:3, 3]
    points_3d = []
    half = window // 2
    h, w = depth_map.shape[:2]
    for i, (u, v) in enumerate(corners):
        pt_rgb = np.array([u, v, 1.0], dtype=np.float64)
        ray_rgb = K_rgb_inv @ pt_rgb
        ray_depth = R @ ray_rgb + t
        ray_depth /= ray_depth[2]
        pt_depth = K_depth @ ray_depth
        u_d, v_d = int(pt_depth[0]), int(pt_depth[1])
        u1, u2 = max(0, u_d - half), min(w, u_d + half + 1)
        v1, v2 = max(0, v_d - half), min(h, v_d + half + 1)
        window_vals = depth_map[v1:v2, u1:u2]
        valid = window_vals[(window_vals > 0) & np.isfinite(window_vals)]
        if valid.size == 0:
            if VERBOSE:
                log.info(f"[backproj][pt#{i}] NO DEPTH: ({u_d},{v_d})")
            continue
        depth_raw = float(np.median(valid))
        Z = depth_raw * depth_scale
        X = (pt_depth[0] - K_depth[0, 2]) * Z / K_depth[0, 0]
        Y = (pt_depth[1] - K_depth[1, 2]) * Z / K_depth[1, 1]
        pt_depth_3d = np.array([X, Y, Z], dtype=np.float64)
        pt_rgb_3d = R.T @ (pt_depth_3d - t)
        if VERBOSE:
            log.debug(
                f"[backproj][pt#{i}] RGB ({u:.1f},{v:.1f}) -> D ({u_d},{v_d}), Z={Z:.3f}, D3D={pt_depth_3d}, RGB3D={pt_rgb_3d}"
            )
        points_3d.append(pt_rgb_3d)
    return np.array(points_3d, dtype=np.float64)


# =============================================================================
# 6. VISUALIZATION & OVERLAY
# =============================================================================


def overlay_corners(
    image: np.ndarray | None,
    corners: np.ndarray,
    ids: np.ndarray,
    prefix: str,
    out_file: Path,
    is_depth: bool = False,
    depth_map: np.ndarray | None = None,
) -> None:
    if not OVERLAY_ENABLED:
        return
    out = (
        colorize_depth_with_legend(depth_map)
        if is_depth and depth_map is not None
        else image.copy()
    )
    color = (0, 255, 0) if is_depth else (0, 0, 255)
    text_color = (255, 255, 255) if is_depth else (0, 255, 255)
    for i, (pt, id_) in enumerate(zip(corners, ids)):
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(out, (x, y), CORNER_RADIUS, color, -1)
        cv2.putText(
            out,
            f"{prefix}#{i:02d}:({x},{y})",
            (x + CORNER_RADIUS, y - CORNER_RADIUS),
            cv2.FONT_HERSHEY_SIMPLEX,
            OVERLAY_FONT_SCALE,
            text_color,
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(out_file), out)
    if VERBOSE:
        log.info(f"[overlay] Saved {out_file}")


def colorize_depth_with_legend(depth: np.ndarray) -> np.ndarray:
    d = depth.astype(np.float32)
    mask = d > 0
    vmin, vmax = (
        (np.percentile(d[mask], 1), np.percentile(d[mask], 99))
        if np.any(mask)
        else (0, 1)
    )
    d_norm = np.clip((d - vmin) / (vmax - vmin + 1e-8), 0, 1)
    d_img = (d_norm * 255).astype(np.uint8)
    color = cv2.applyColorMap(d_img, cv2.COLORMAP_JET)
    h, w = color.shape[:2]
    bar = np.linspace(vmax, vmin, h).reshape(h, 1)
    bar_img = (255 * (bar - vmin) / (vmax - vmin + 1e-8)).astype(np.uint8)
    bar_color = cv2.applyColorMap(bar_img, cv2.COLORMAP_JET)
    bar_color = cv2.putText(
        bar_color,
        f"{vmax:.2f}m",
        (2, 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        OVERLAY_FONT_SCALE,
        (255, 255, 255),
        1,
    )
    bar_color = cv2.putText(
        bar_color,
        f"{vmin:.2f}m",
        (2, h - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        OVERLAY_FONT_SCALE,
        (255, 255, 255),
        1,
    )
    out = np.hstack([color, bar_color])
    return out


# =============================================================================
# 7. POSE ESTIMATION & HAND-EYE CALIBRATION
# =============================================================================


def estimate_pose_pnp(
    obj_pts: np.ndarray, img_pts: np.ndarray, K: np.ndarray, dist: np.ndarray
) -> np.ndarray | None:
    if obj_pts.shape[0] < 4:
        if VERBOSE:
            log.info(f"[PnP] Not enough points: {obj_pts.shape[0]}")
        return None
    flags = (
        cv2.SOLVEPNP_IPPE_SQUARE if obj_pts.shape[0] == 4 else cv2.SOLVEPNP_ITERATIVE
    )
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=flags)
    if not ok:
        if VERBOSE:
            log.info("[PnP] PnP failed.")
        return None
    Rmat, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = Rmat
    T[:3, 3] = tvec.flatten()
    if VERBOSE:
        log.info(f"[PnP] rvec:{rvec.flatten()}, tvec:{tvec.flatten()}, R:\n{Rmat}")
    return T


def run_handeye_calibration(
    robot_T: List[np.ndarray], target_T: List[np.ndarray]
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    methods = {
        "tsai": cv2.CALIB_HAND_EYE_TSAI,
        "park": cv2.CALIB_HAND_EYE_PARK,
        "horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
        "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    robot_R = [T[:3, :3] for T in robot_T]
    robot_t = [T[:3, 3] for T in robot_T]
    target_R = [T[:3, :3] for T in target_T]
    target_t = [T[:3, 3] for T in target_T]
    results = {}
    for name, method in methods.items():
        R, t = cv2.calibrateHandEye(robot_R, robot_t, target_R, target_t, method=method)
        if VERBOSE:
            log.info(f"[handeye][{name}] R:\n{R}\nt:{t.flatten()}")
        results[name] = (R, t)
    return results


def compute_handeye_errors(
    robot_T: List[np.ndarray],
    target_T: List[np.ndarray],
    solutions: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> List[Tuple[str, float, float]]:
    summary = []
    for name, (R_he, t_he) in solutions.items():
        T_he = np.eye(4)
        T_he[:3, :3] = R_he
        T_he[:3, 3] = t_he.flatten()
        pos_err, rot_err = [], []
        for i, (T_r, T_t) in enumerate(zip(robot_T, target_T)):
            cam_T = T_r @ T_he
            diff = np.linalg.inv(cam_T) @ T_t
            pos_err.append(np.linalg.norm(diff[:3, 3]))
            angle = np.arccos(np.clip((np.trace(diff[:3, :3]) - 1) / 2, -1.0, 1.0))
            rot_err.append(np.degrees(angle))
            if VERBOSE and i == 0:
                log.debug(
                    f"[handeye][{name}][pair0] T_r:\n{T_r}\nT_t:\n{T_t}\ncam_T:\n{cam_T}\ndiff:\n{diff}"
                )
        summary.append((name, float(np.mean(pos_err)), float(np.mean(rot_err))))
    return summary


def compute_mean_reprojection_error(
    poses: List[np.ndarray],
    obj_pts_list: List[np.ndarray],
    img_pts_list: List[np.ndarray],
    K: np.ndarray,
    dist: np.ndarray,
) -> float:
    total_err = 0.0
    total_pts = 0
    for T, obj_pts, img_pts in zip(poses, obj_pts_list, img_pts_list):
        if T is None or obj_pts.shape[0] == 0:
            continue
        R_cv, _ = cv2.Rodrigues(cv2.Rodrigues(T[:3, :3])[0])
        tvec = T[:3, 3].reshape(-1, 1)
        proj, _ = cv2.projectPoints(obj_pts, R_cv, tvec, K, dist)
        err = np.linalg.norm(proj.squeeze(1) - img_pts, axis=1)
        total_err += err.sum()
        total_pts += err.size
    return float(total_err / total_pts) if total_pts > 0 else float("nan")


def is_valid_pose(T: np.ndarray) -> bool:
    R = T[:3, :3]
    det = np.linalg.det(R)
    ortho = np.allclose(R @ R.T, np.eye(3), atol=POSE_ORTHO_TOL)
    return np.isfinite(det) and abs(det - 1.0) < POSE_DET_TOL and ortho


def is_valid_rotation(R: np.ndarray) -> bool:
    return (
        np.isfinite(R).all()
        and abs(np.linalg.det(R) - 1) < POSE_DET_TOL
        and np.allclose(R @ R.T, np.eye(3), atol=POSE_ORTHO_TOL)
        and np.min(np.abs(R)) > MIN_ROT_ELEM
    )


def filter_pose_pairs(
    robot_T: List[np.ndarray], target_T: List[np.ndarray]
) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, int]]:
    reason_stats: Dict[str, int] = {}
    robot_T_final, target_T_final = [], []
    for i, (T_r, T_t) in enumerate(zip(robot_T, target_T)):
        reasons = []
        if not is_valid_rotation(T_r[:3, :3]):
            reasons.append("robot_R not valid")
        if not is_valid_rotation(T_t[:3, :3]):
            reasons.append("target_R not valid")
        if not np.all(np.isfinite(T_r)):
            reasons.append("robot_T NaN/Inf")
        if not np.all(np.isfinite(T_t)):
            reasons.append("target_T NaN/Inf")
        if reasons:
            for r in reasons:
                reason_stats[r] = reason_stats.get(r, 0) + 1
            if VERBOSE:
                log.info(
                    f"[filter][pair#{i}] Drop: {reasons}, T_r:\n{T_r}\nT_t:\n{T_t}"
                )
        else:
            robot_T_final.append(T_r)
            target_T_final.append(T_t)
            if VERBOSE:
                log.info(f"[filter][pair#{i}] VALID.\nT_r:\n{T_r}\nT_t:\n{T_t}")
    return robot_T_final, target_T_final, reason_stats


# =============================================================================
# 8. MAIN EXECUTION
# =============================================================================
# 1. Load calibration/intrinsics, images, robot poses
# 2. Configure marker detector (Charuco/Aruco)
# 3. Detect markers on all RGB images
# 4. Project detected corners to depth frame using intrinsics + extrinsics
# 5. Backproject to 3D using depth, convert back to RGB sensor coords
# 6. SolvePnP for board pose in each frame
# 7. Filter all invalid pose pairs (det, ortho, nan)
# 8. Hand-eye calibration (all methods)
# 9. Error analysis and YAML report

if __name__ == "__main__":
    # =============================
    # [1] LOGGER & INPUTS
    # =============================
    Logger.configure(level="DEBUG" if VERBOSE else "INFO")
    pose_json_files = list(IMG_DIR.parent.glob(POSES_JSON_PATTERN))
    if not pose_json_files:
        raise FileNotFoundError(f"No JSON files found in {IMG_DIR}")
    POSES_FILE = pose_json_files[0]

    # =============================
    # [2] LOAD CALIBRATION DATA
    # =============================
    K_rgb, dist_rgb = load_intrinsics(INTR_FILE)
    K_depth, dist_depth = load_intrinsics(INTRD_FILE)
    rgb_to_depth = np.eye(4)
    rgb_to_depth[:3, :3] = np.array(EXTR_COLOR_TO_DEPTH_ROT)
    rgb_to_depth[:3, 3] = np.array(EXTR_COLOR_TO_DEPTH_TRANS)
    board, detector = configure_charuco_board(
        BOARD_SIZE, SQUARE_LEN, MARKER_LEN, CHARUCO_DICT
    )
    log_versions_and_config(board)

    # =============================
    # [3] LOAD DATASET PAIRS
    # =============================
    pairs = load_image_pairs(IMG_DIR)
    robot_T = load_robot_poses(POSES_FILE)
    overlay_dir = IMG_DIR.parent / "over"
    overlay_dir.mkdir(exist_ok=True)

    # =============================
    # [4] DETECTION, PROJECTION, PNP
    # =============================
    n_mismatch = n_too_few = n_pnp_fail = n_rot_bad = 0
    target_T: List[np.ndarray] = []
    obj_pts_list: List[np.ndarray] = []
    img_pts_list: List[np.ndarray] = []

    for idx, (rgb_file, depth_file) in enumerate(pairs):
        img = cv2.imread(str(rgb_file))
        depth = np.load(depth_file)
        detection = detect_charuco_corners(img, board, detector)
        if detection is None:
            if VERBOSE:
                log.info(f"[main][{rgb_file.name}] Charuco detection failed.")
            continue
        img_pts, ids = detection
        obj_pts = board.getChessboardCorners()[ids]
        if DISABLE_DEPTH_Z:
            pts_3d = obj_pts
        else:
            overlay_corners(
                img,
                img_pts,
                ids,
                "crn",
                overlay_dir / f"{rgb_file.stem}_rgb.png",
                is_depth=False,
            )
            proj_pts = project_rgb_to_depth(img_pts, K_rgb, K_depth, rgb_to_depth)
            overlay_corners(
                None,
                proj_pts,
                ids,
                "crn",
                overlay_dir / f"{rgb_file.stem}_depth.png",
                is_depth=True,
                depth_map=depth,
            )
            pts_3d = backproject_rgb_corners_to_3d(
                img_pts, depth, K_rgb, K_depth, rgb_to_depth, DEPTH_SCALE
            )
        if pts_3d.shape[0] != obj_pts.shape[0]:
            n_mismatch += 1
            if VERBOSE:
                log.info(
                    f"[main][{rgb_file.name}] Invalid 3D/2D: {pts_3d.shape[0]} vs {obj_pts.shape[0]}"
                )
            continue
        if pts_3d.shape[0] < 6:
            n_too_few += 1
            if VERBOSE:
                log.info(f"[main][{rgb_file.name}] Too few points: {pts_3d.shape[0]}")
            continue
        pose = estimate_pose_pnp(obj_pts, img_pts, K_rgb, dist_rgb)
        if pose is None:
            n_pnp_fail += 1
            continue
        if not is_valid_pose(pose):
            n_rot_bad += 1
            if VERBOSE:
                log.info(f"[main][{rgb_file.name}] Bad rotation in pose.")
            continue
        target_T.append(pose)
        obj_pts_list.append(obj_pts)
        img_pts_list.append(img_pts)

    # =============================
    # [5] FILTER POSE PAIRS
    # =============================
    robot_T_final, target_T_final, reason_stats = filter_pose_pairs(
        robot_T[: len(target_T)], target_T
    )
    n_total = len(robot_T[: len(target_T)])
    n_valid = len(robot_T_final)
    n_dropped = n_total - n_valid
    log.info(
        f"Final filtered: {n_valid} valid pairs out of {n_total} ({n_valid / n_total * 100:.1f}%)"
    )
    if n_dropped > 0:
        log.info("Drop reasons:")
        for reason, count in sorted(reason_stats.items(), key=lambda x: -x[1]):
            percent = count / n_dropped * 100 if n_dropped else 0
            log.info(f"  - {reason:<18}: {count:3d} ({percent:5.1f}%)")

    # =============================
    # [6] METRICS & QUALITY
    # =============================
    mean_repr_err = compute_mean_reprojection_error(
        target_T_final, obj_pts_list[:n_valid], img_pts_list[:n_valid], K_rgb, dist_rgb
    )
    log.info(f"Mean reprojection error (PnP): {mean_repr_err:.4f} px")
    results: Dict[str, Any] = {
        "num_valid_pairs": n_valid,
        "num_total_pairs": n_total,
        "mean_reprojection_error_px": mean_repr_err,
    }

    # =============================
    # [7] HAND-EYE CALIBRATION
    # =============================
    if n_valid >= 3:
        solutions = run_handeye_calibration(robot_T_final, target_T_final)
        summary = compute_handeye_errors(robot_T_final, target_T_final, solutions)
        log.info("Summary of hand-eye calibration errors:")
        log.info("Method     |  T. RMSE [m]  |  R. RMSE [deg]")
        log.info("-----------|---------------|----------------")
        for n, t_err, r_err in summary:
            log.info(f"{n:<10} |  {t_err:>10.6f}   |  {r_err:>13.4f}")
        results["handeye_solutions"] = {
            n: {
                "R": solutions[n][0].tolist(),
                "t": solutions[n][1].flatten().tolist(),
                "t_rmse_m": t_err,
                "r_rmse_deg": r_err,
            }
            for n, t_err, r_err in summary
        }
    else:
        log.error("Not enough valid pairs for hand-eye calibration.")

    # =============================
    # [8] SAVE FINAL YAML OUTPUT
    # =============================
    save_yaml(IMG_DIR.parent / "calibration_results.yaml", results)
