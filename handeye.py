"""Function-based, mathematically-correct calibration pipeline with clear overlays and robust logging."""

from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import json
import numpy as np
import yaml
from cv2 import aruco

from utils.logger import Logger
from utils.settings import EXTR_COLOR_TO_DEPTH_ROT, EXTR_COLOR_TO_DEPTH_TRANS


log = Logger.get_logger("handeye")
np.set_printoptions(suppress=True, precision=6, linewidth=200)
DICT_NAMES = {
    getattr(aruco, name): name for name in dir(aruco) if name.startswith("DICT_")
}


def get_dict_name(dictionary: aruco.Dictionary) -> str:
    """Return a human-readable name for a cv2.aruco.Dictionary instance."""
    for key, name in DICT_NAMES.items():
        if (
            dictionary.bytesList.shape
            == aruco.getPredefinedDictionary(key).bytesList.shape
        ):
            return name
    return "UNKNOWN"


# ===============================
# I/O and configuration functions
# ===============================


def load_intrinsics(yaml_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load camera matrix and distortion coefficients from YAML file."""
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
    log.debug(f"Loaded intrinsics: K={K}, dist={dist}")
    return K, dist


def load_image_pairs(directory: Path) -> List[Tuple[Path, Path]]:
    """Return list of (rgb, depth) image pairs by filename stem."""
    rgb_files = sorted(directory.glob("frame_*.png"))
    pairs = []
    for rgb in rgb_files:
        dpth = rgb.with_suffix(".npy")
        if dpth.exists():
            pairs.append((rgb, dpth))
        else:
            log.warning(f"Missing depth file for {rgb.name}")
    return pairs


def load_robot_poses(json_path: Path) -> List[np.ndarray]:
    """Load robot poses from JSON file, each as 4x4 T matrix (meters, radians)."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    from scipy.spatial.transform import Rotation as R

    poses = []
    for v in data.values():
        x, y, z, rx, ry, rz = v["tcp_coords"]
        t = np.array([x, y, z], dtype=float) / 1000.0
        Rmat = R.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = Rmat
        T[:3, 3] = t
        poses.append(T)
    log.info(f"Loaded {len(poses)} robot poses")
    return poses


def configure_charuco_board(
    board_size: Tuple[int, int], square_len: float, marker_len: float, dict_name: str
) -> Tuple[cv2.aruco.CharucoBoard, cv2.aruco.ArucoDetector]:
    """Create a Charuco board and its detector."""
    dict_id = getattr(cv2.aruco, dict_name)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    board = cv2.aruco.CharucoBoard(board_size, square_len, marker_len, aruco_dict)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    return board, detector


# ================
# Detection blocks
# ================


def detect_charuco_corners(
    img: np.ndarray, board, detector
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Detect Charuco corners and their IDs. Returns (corners, ids) or None."""
    marker_corners, marker_ids, _ = detector.detectMarkers(img)
    if marker_ids is None or len(marker_ids) == 0:
        return None
    ok, char_corners, char_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners, marker_ids, img, board
    )
    if not ok or char_ids is None or len(char_ids) < 4:
        return None
    return char_corners.squeeze(1), char_ids.flatten()


# ====================
# Geometry + Backproj
# ====================


def project_rgb_to_depth(
    corners: np.ndarray,
    K_rgb: np.ndarray,
    K_depth: np.ndarray,
    rgb_to_depth: np.ndarray,
) -> np.ndarray:
    """
    Project 2D points in RGB image onto the depth image using full extrinsic/intrinsic chain.
    Returns (N,2) array of projected 2D points in depth image coordinates (float).
    """
    K_rgb_inv = np.linalg.inv(K_rgb)
    R = rgb_to_depth[:3, :3]
    t = rgb_to_depth[:3, 3]
    out = []
    for i, (u, v) in enumerate(corners):
        pt_rgb = np.array([u, v, 1.0], dtype=np.float64)
        ray_rgb = K_rgb_inv @ pt_rgb
        ray_depth = R @ ray_rgb + t
        ray_depth /= ray_depth[2]
        pt_depth = K_depth @ ray_depth
        out.append(pt_depth[:2])
    return np.array(out, dtype=np.float64)


def backproject_rgb_corners_to_3d(
    corners: np.ndarray,
    depth_map: np.ndarray,
    K_rgb: np.ndarray,
    K_depth: np.ndarray,
    rgb_to_depth: np.ndarray,
    depth_scale: float,
) -> np.ndarray:
    """
    Backproject RGB image corners to 3D points in the RGB camera frame using the depth map and full extrinsics/intrinsics.
    Returns: (N, 3) array (some points may be skipped if depth invalid).
    """
    K_rgb_inv = np.linalg.inv(K_rgb)
    R = rgb_to_depth[:3, :3]
    t = rgb_to_depth[:3, 3]
    points_3d = []
    for i, (u, v) in enumerate(corners):
        # Project to depth image
        pt_rgb = np.array([u, v, 1.0], dtype=np.float64)
        ray_rgb = K_rgb_inv @ pt_rgb
        ray_depth = R @ ray_rgb + t
        ray_depth /= ray_depth[2]
        pt_depth = K_depth @ ray_depth
        u_d, v_d = int(round(pt_depth[0])), int(round(pt_depth[1]))
        # Check bounds
        if not (0 <= u_d < depth_map.shape[1] and 0 <= v_d < depth_map.shape[0]):
            log.debug(f"[crn#{i:02d}] Out of bounds in depth image: ({u_d},{v_d})")
            continue
        depth_raw = float(depth_map[v_d, u_d])
        if not np.isfinite(depth_raw) or depth_raw <= 0:
            log.debug(f"[crn#{i:02d}] Invalid depth at ({u_d},{v_d}): {depth_raw}")
            continue
        Z = depth_raw * depth_scale
        X = (pt_depth[0] - K_depth[0, 2]) * Z / K_depth[0, 0]
        Y = (pt_depth[1] - K_depth[1, 2]) * Z / K_depth[1, 1]
        pt_depth_3d = np.array([X, Y, Z], dtype=np.float64)
        # Transform back to RGB cam frame
        pt_rgb_3d = R.T @ (pt_depth_3d - t)
        log.debug(
            f"[crn#{i:02d}] RGB ({u:.1f},{v:.1f}) → depth ({u_d},{v_d}) → 3D {pt_rgb_3d.round(4)}"
        )
        points_3d.append(pt_rgb_3d)
    return np.array(points_3d, dtype=np.float64)


# ======================
# Overlay / Visualization
# ======================


def overlay_corners_on_rgb(
    image: np.ndarray, corners: np.ndarray, ids: np.ndarray, prefix: str, out_file: Path
):
    """Overlay Charuco corners on an RGB image, label with index and coordinates."""
    out = image.copy()
    for i, (pt, id_) in enumerate(zip(corners, ids)):
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(out, (x, y), 7, (0, 0, 255), -1)
        cv2.putText(
            out,
            f"{prefix}#{i:02d}:({x},{y})",
            (x + 7, y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(out_file), out)
    log.info(f"Saved overlay: {out_file}")


def overlay_corners_on_depth(
    depth_map: np.ndarray,
    proj_corners: np.ndarray,
    ids: np.ndarray,
    prefix: str,
    out_file: Path,
):
    """Overlay projected corners on depth image, label with index and coordinates."""
    img_norm = colorize_depth_with_legend(depth_map)
    out = img_norm.copy()
    for i, (pt, id_) in enumerate(zip(proj_corners, ids)):
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(out, (x, y), 7, (0, 255, 0), -1)
        cv2.putText(
            out,
            f"{prefix}#{i:02d}:({x},{y})",
            (x + 7, y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(out_file), out)
    log.info(f"Saved overlay: {out_file}")


def colorize_depth_with_legend(depth: np.ndarray) -> np.ndarray:
    """Normalize depth map to [1,99] percentiles, colormap, add legend bar at right."""
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
    # add colorbar legend
    h, w = color.shape[:2]
    bar = np.linspace(vmax, vmin, h).reshape(h, 1)
    bar_img = (255 * (bar - vmin) / (vmax - vmin + 1e-8)).astype(np.uint8)
    bar_color = cv2.applyColorMap(bar_img, cv2.COLORMAP_JET)
    bar_color = cv2.putText(
        bar_color,
        f"{vmax:.2f}m",
        (2, 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
    )
    bar_color = cv2.putText(
        bar_color,
        f"{vmin:.2f}m",
        (2, h - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
    )
    out = np.hstack([color, bar_color])
    return out


# ===================
# PnP + hand-eye code
# ===================


def estimate_pose_pnp(
    obj_pts: np.ndarray, img_pts: np.ndarray, K: np.ndarray, dist: np.ndarray
) -> Optional[np.ndarray]:
    """Estimate camera pose from 3D-2D correspondences (OpenCV PnP). Returns 4x4 T or None."""
    if obj_pts.shape[0] < 4:
        return None
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if not ok:
        return None
    Rmat, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = Rmat
    T[:3, 3] = tvec.flatten()
    return T


def run_handeye_calibration(
    robot_T: List[np.ndarray], target_T: List[np.ndarray]
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Run hand-eye calibration for all supported OpenCV methods."""
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
        results[name] = (R, t)
    return results


def compute_handeye_errors(
    robot_T: List[np.ndarray],
    target_T: List[np.ndarray],
    solutions: Dict[str, Tuple[np.ndarray, np.ndarray]],
):
    """Compute mean translation/rotation error for each solution."""
    for name, (R_he, t_he) in solutions.items():
        T_he = np.eye(4)
        T_he[:3, :3] = R_he
        T_he[:3, 3] = t_he
        pos_err, rot_err = [], []
        for T_r, T_t in zip(robot_T, target_T):
            cam_T = T_r @ T_he
            diff = np.linalg.inv(cam_T) @ T_t
            pos_err.append(np.linalg.norm(diff[:3, 3]))
            angle = np.arccos(np.clip((np.trace(diff[:3, :3]) - 1) / 2, -1.0, 1.0))
            rot_err.append(np.degrees(angle))
        log.info(
            f"{name}: translational {np.mean(pos_err):.6f} m, rotational {np.mean(rot_err):.4f} deg"
        )


# ================
# Main pipeline
# ================

if __name__ == "__main__":
    Logger.configure(level="DEBUG")
    INTR_FILE = Path("cam_params.yml")
    INTRD_FILE = Path("dcam_params.yml")
    IMG_DIR = Path("calib/imgs")
    POSES_FILE = Path("calib/poses_20250717_151804.json")
    CHARUCO_DICT = "DICT_5X5_100"
    SQUARE_LEN = 0.035
    MARKER_LEN = 0.026
    DEPTH_SCALE = 0.0001
    BOARD_SIZE = (8, 5)

    log.info("Loading camera intrinsics...")
    K_rgb, dist_rgb = load_intrinsics(INTR_FILE)
    K_depth, dist_depth = load_intrinsics(INTRD_FILE)

    log.debug(
        "CAMERA INTRINSICS\n--- RGB Camera ---\n"
        f"Camera mtx:\n{K_rgb}\nDistCoeffs:\n{dist_rgb}\n\n"
        "--- Depth Camera ---\n"
        f"Camera mtx:\n{K_depth}\nDistCoeffs:\n{dist_depth}"
    )

    # Build RGB→Depth transform as a full 4x4
    rgb_to_depth = np.eye(4)
    rgb_to_depth[:3, :3] = np.array(EXTR_COLOR_TO_DEPTH_ROT)
    rgb_to_depth[:3, 3] = np.array(EXTR_COLOR_TO_DEPTH_TRANS)
    log.debug(f"EXTR_COLOR_TO_DEPTH\n{rgb_to_depth}")

    log.info("Configuring Charuco board...")
    board, detector = configure_charuco_board(
        BOARD_SIZE, SQUARE_LEN, MARKER_LEN, CHARUCO_DICT
    )
    dict_name = get_dict_name(board.getDictionary())
    log.debug(
        "CHARUCO BOARD CONFIG\n"
        f"  Size          : {board.getChessboardSize()}\n"
        f"  Square length : {board.getSquareLength():.16f} m\n"
        f"  Marker length : {board.getMarkerLength():.16f} m\n"
        f"  Dictionary    : {dict_name}"
    )

    log.info("Loading RGB-Depth image pairs...")
    pairs = load_image_pairs(IMG_DIR)
    log.debug(f"Found {len(pairs)} image pairs")

    log.info("Loading robot poses...")
    robot_T = load_robot_poses(POSES_FILE)
    log.debug(f"Loaded {len(robot_T)} robot poses")

    target_T = []
    for idx, (rgb_file, depth_file) in enumerate(pairs[:1]):
        log.info(f"[img#{idx:02d}] {rgb_file.name} / {depth_file.name}")
        img = cv2.imread(str(rgb_file))
        if img is None:
            log.error(f"[img#{idx:02d}] Can't load image: {rgb_file}")
            continue
        depth = np.load(depth_file)
        detection = detect_charuco_corners(img, board, detector)
        if detection is None:
            log.warning(f"[img#{idx:02d}] Charuco detection failed.")
            continue
        img_pts, ids = detection
        obj_pts = board.getChessboardCorners()[ids]
        overlay_dir = rgb_file.parent / "over"
        overlay_dir.mkdir(exist_ok=True)
        overlay_path_rgb = overlay_dir / f"{rgb_file.stem}_overlay_rgb.png"
        overlay_path_depth = overlay_dir / f"{rgb_file.stem}_overlay_depth.png"

        overlay_corners_on_rgb(img, img_pts, ids, "crn", overlay_path_rgb)
        proj_pts = project_rgb_to_depth(img_pts, K_rgb, K_depth, rgb_to_depth)
        overlay_corners_on_depth(depth, proj_pts, ids, "crn", overlay_path_depth)

        pts_3d = backproject_rgb_corners_to_3d(
            img_pts, depth, K_rgb, K_depth, rgb_to_depth, DEPTH_SCALE
        )
        if pts_3d.shape[0] != obj_pts.shape[0]:
            log.warning(
                f"[img#{idx:02d}] Mismatch 3D/2D count: {pts_3d.shape[0]} vs {obj_pts.shape[0]}"
            )
            continue

        pose = estimate_pose_pnp(obj_pts, img_pts, K_rgb, dist_rgb)
        if pose is None:
            log.warning(f"[img#{idx:02d}] PnP failed.")
            continue
        target_T.append(pose)

    if len(robot_T) < len(target_T):
        target_T = target_T[: len(robot_T)]
    if len(robot_T) >= 3 and len(target_T) >= 3:
        solutions = run_handeye_calibration(robot_T[: len(target_T)], target_T)
        compute_handeye_errors(robot_T[: len(target_T)], target_T, solutions)
    else:
        log.error("Not enough valid pairs for hand-eye calibration.")
