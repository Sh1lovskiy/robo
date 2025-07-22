"""
Hand-Eye Calibration Pipeline for Vision-Guided Robotics
========================================================
"""

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
# 1. CONFIGURATION CONSTANTS AND SYSTEM PARAMETERS
# =============================================================================

# --- File and Dataset Paths ---
INTR_FILE = Path("cam_params.yml")  # YAML with RGB camera intrinsics/distortion
INTRD_FILE = Path("dcam_params.yml")  # YAML with depth camera intrinsics/distortion
IMG_DIR = Path("calib/imgs")  # directory containing image/depth pairs
POSES_JSON_PATTERN = "*.json"  # pattern for robot TCP pose logs (per image set)

# --- Charuco Board Geometry ---
CHARUCO_DICT = "DICT_5X5_100"  # Charuco dictionary name for ArUco detection
BOARD_SIZE = (8, 5)  # (columns, rows) of internal chessboard
SQUARE_LEN = 0.035  # length of board squares [meters]
MARKER_LEN = 0.026  # length of ArUco markers [meters]

# --- Depth/Overlay/Logging Parameters ---
VERBOSE = True
OVERLAY_ENABLED = True  # corner viz
DISABLE_DEPTH_Z = True  # if True, disables use of depth for 3D
DEPTH_SCALE = 0.0010000000474974513  # multiply raw depth values to get meters
WINDOW_SIZE = 1  # median window for local depth filtering [pxls]

POSE_ORTHO_TOL = 1e-7  # tolerance for orthogonality in rotation validation
POSE_DET_TOL = 1e-7  # tolerance for determinant in rotation validation
MIN_ROT_ELEM = 1e-20  # min abs value for elements in a rotation validation

CORNER_RADIUS = 7  # Overlay circle radius [pxls]
OVERLAY_FONT_SCALE = 0.4  # Overlay text scale

np.set_printoptions(suppress=True, precision=6, linewidth=200)

log = Logger.get_logger("handeye")
DICT_NAMES = {
    getattr(aruco, name): name for name in dir(aruco) if name.startswith("DICT_")
}

# =============================================================================
# 2. SYSTEM INFORMATION AND UTILITY OUTPUT
# =============================================================================


def log_versions_and_config(board: aruco.CharucoBoard) -> None:
    """
    Log system library versions, Charuco board parameters, and pipeline configuration.

    Args:
        board: CharucoBoard instance used for detection.
    """
    log.info(
        f"""System/Calibration Configuration
    Python: {sys.version}
    OpenCV: {cv2.__version__}
    NumPy: {np.__version__}
    Pipeline config:
        - DISABLE_DEPTH_Z: {DISABLE_DEPTH_Z}
        - DEPTH_SCALE: {DEPTH_SCALE}
        - OVERLAY_ENABLED: {OVERLAY_ENABLED}
        - VERBOSE: {VERBOSE}
    Charuco BOARD:
        - size: {board.getChessboardSize()} squares
        - square_len: {board.getSquareLength()} m
        - marker_len: {board.getMarkerLength()} m
        - dictionary: {get_dict_name(board.getDictionary())}
    """
    )


def get_dict_name(dictionary: aruco.Dictionary) -> str:
    """Return the string name of an OpenCV ArUco dictionary."""
    for key, name in DICT_NAMES.items():
        if (
            dictionary.bytesList.shape
            == aruco.getPredefinedDictionary(key).bytesList.shape
        ):
            return name
    return "UNKNOWN"


def save_yaml(out_file: Path, data: dict) -> None:
    """Save a Python dictionary as YAML for reproducible calibration logging."""
    with open(out_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)
    log.info(f"Saved YAML calibration results to: {out_file}")


# =============================================================================
# 3. CAMERA/ROBOT DATA LOADING AND CONFIGURATION
# =============================================================================


def load_intrinsics(yaml_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load camera intrinsic matrix and distortion coefficients from a YAML file.

    Returns:
        K (3x3 ndarray): Camera intrinsic matrix (OpenCV pinhole convention, pixels).
        dist (1D ndarray): Distortion coefficients, OpenCV order (k1, k2, p1, p2, ...).
    """
    log.info(f"Loading intrinsics from: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if "camera_matrix" in data:
        K = np.array(data["camera_matrix"]["data"], dtype=float).reshape(3, 3)
        dist = np.array(data["distortion_coefficients"]["data"], dtype=float)
    else:
        # Fallback: Custom format (fx, fy, cx, cy, possibly 'dist')
        fx, fy, cx, cy = data["fx"], data["fy"], data["cx"], data["cy"]
        dist = np.array(data.get("dist", []), dtype=float)
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
    if VERBOSE:
        log.info(f"[intrinsics]\nK (pixels):\n{K}\nDistortion: {dist}")
    return K, dist


def load_image_pairs(directory: Path) -> List[Tuple[Path, Path]]:
    """
    Load (RGB, depth) image pairs from the specified directory.
    Each RGB frame should have a corresponding .npy depth file.
    Returns:
        List of (rgb_image_path, depth_npy_path) pairs.
    """
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
        log.info(
            f"[image_pairs] {len(pairs)} RGB/depth pairs loaded, missing: {len(missing)}"
        )
    return pairs


def load_robot_poses(json_path: Path) -> List[np.ndarray]:
    """
    Load a list of robot TCP (tool center point) poses from a JSON file.

    Each pose must include fields: 'tcp_coords': [x(mm), y, z, rx(deg), ry, rz]
    Returns:
        List of 4x4 T_base_tcp transforms (robot base -> TCP, right-handed, meters).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    poses = []
    for idx, v in enumerate(data.values()):
        x, y, z, rx, ry, rz = v["tcp_coords"]  # Position (mm), orientation (deg)
        t = np.array([x, y, z], dtype=float) / 1000.0  # Convert mm -> meters
        # Convert extrinsic Euler angles (xyz, degrees) to rotation matrix
        Rmat = R.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()
        # Compose 4x4 homogeneous transform: T_base_tcp
        T = np.eye(4)
        T[:3, :3] = Rmat
        T[:3, 3] = t
        poses.append(T)
        if VERBOSE:
            log.info(
                f"[robot_pose][#{idx}] T_base_tcp\n"
                f"  t_base_tcp: {t} [m]\n"
                f"  R_base_tcp:\n{Rmat}"
            )
    log.info(f"Loaded {len(poses)} robot TCP poses from {json_path}")
    return poses


def configure_charuco_board(
    board_size: Tuple[int, int], square_len: float, marker_len: float, dict_name: str
) -> Tuple[aruco.CharucoBoard, aruco.ArucoDetector]:
    """
    Configure Charuco board and ArUco detector.

    Returns:
        - CharucoBoard instance.
        - ArucoDetector instance.
    """
    dict_id = getattr(aruco, dict_name)
    aruco_dict = aruco.getPredefinedDictionary(dict_id)
    board = aruco.CharucoBoard(board_size, square_len, marker_len, aruco_dict)
    params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, params)
    if VERBOSE:
        log.info(
            f"[charuco] Created Charuco board {board_size} (cols,rows), "
            f"square_len={square_len} m, marker_len={marker_len} m"
        )
    return board, detector


# =============================================================================
# 4. CHARUCO CORNER DETECTION (IN RGB CAMERA FRAME)
# =============================================================================


def detect_charuco_corners(
    img: np.ndarray, board: aruco.CharucoBoard, detector: aruco.ArucoDetector
) -> Tuple[np.ndarray, np.ndarray] | None:
    """
    Detect Charuco corners in an RGB image (uint8, BGR).
    Returns:
        - charuco_corners: (N, 2) pixel coordinates (float32)
        - charuco_ids:     (N, ) marker-corner integer IDs
    If detection fails or < 4 corners, returns None.
    """
    marker_corners, marker_ids, _ = detector.detectMarkers(img)
    if marker_ids is None or len(marker_ids) == 0:
        if VERBOSE:
            log.info("[detect_charuco] No ArUco markers detected.")
        return None
    ok, char_corners, char_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners, marker_ids, img, board
    )
    if not ok or char_ids is None or len(char_ids) < 4:
        if VERBOSE:
            log.info(f"[detect_charuco] Charuco failed: ok={ok}, ids={char_ids}")
        return None
    if VERBOSE:
        log.info(f"[detect_charuco] {len(char_ids)} Charuco corners found.")
    return char_corners.squeeze(1), char_ids.flatten()


# =============================================================================
# 5. GEOMETRIC PROJECTIONS AND FRAME CONVERSIONS
# =============================================================================


def project_rgb_to_depth(
    corners: np.ndarray,
    K_rgb: np.ndarray,
    K_depth: np.ndarray,
    rgb_to_depth: np.ndarray,
) -> np.ndarray:
    """
    Project 2D pixel points from RGB camera to depth camera image.
    Follows OpenCV frame conventions.

    Args:
        corners: (N, 2) pixel locations in RGB image.
        K_rgb: (3, 3) intrinsics of RGB camera.
        K_depth: (3, 3) intrinsics of depth camera.
        rgb_to_depth: (4, 4) T_depth_rgb. R, t maps points in RGB camera frame into depth frame.

    Returns:
        (N, 2) projected pixel coordinates in depth camera image.
    """
    K_rgb_inv = np.linalg.inv(K_rgb)
    R = rgb_to_depth[:3, :3]
    t = rgb_to_depth[:3, 3]
    out = []
    for i, (u, v) in enumerate(corners):
        pt_rgb = np.array([u, v, 1.0], dtype=np.float64)
        ray_rgb = K_rgb_inv @ pt_rgb  # normalized ray (RGB camera frame)
        ray_depth = R @ ray_rgb + t  # ray mapped to depth camera frame
        ray_depth /= ray_depth[2]  # normalize z=1 for projection
        pt_depth = K_depth @ ray_depth  # project to depth image pxls
        if VERBOSE:
            log.info(
                f"[proj][pt#{i}] RGB px: {pt_rgb} -> cam ray: {ray_rgb}, "
                f"-> depth cam ray: {ray_depth} -> depth px: {pt_depth[:2]}"
            )
        out.append(pt_depth[:2])
    return np.array(out, dtype=np.float64)


def backproject_rgb_corners_to_3d_aligned(
    corners: np.ndarray,
    depth_map: np.ndarray,
    K_rgb: np.ndarray,
    depth_scale: float,
    window: int = 1,
) -> np.ndarray:
    """
    Backproject Charuco corners to 3D using aligned depth (RGB and depth are pixel-wise aligned).
    Returns (M,3) points in RGB camera frame (meters).
    """
    K_inv = np.linalg.inv(K_rgb)
    points_3d = []
    half = window // 2
    h, w = depth_map.shape[:2]
    for i, (u, v) in enumerate(corners):
        u_int, v_int = int(round(u)), int(round(v))
        u1, u2 = max(0, u_int - half), min(w, u_int + half + 1)
        v1, v2 = max(0, v_int - half), min(h, v_int + half + 1)
        win = depth_map[v1:v2, u1:u2]
        vals = win[(win > 0) & np.isfinite(win)]
        if vals.size == 0:
            continue
        z = float(np.median(vals)) * depth_scale
        pt_h = np.array([u, v, 1.0], dtype=np.float64)
        xyz = z * K_inv @ pt_h
        points_3d.append(xyz)
    return np.array(points_3d, dtype=np.float64)


def backproject_rgb_corners_to_3d(
    corners: np.ndarray,
    depth_map: np.ndarray,
    K_rgb: np.ndarray,
    K_depth: np.ndarray,
    rgb_to_depth: np.ndarray,
    depth_scale: float,
    window: int = WINDOW_SIZE,
) -> np.ndarray:
    """
    For each 2D Charuco corner in the RGB image, map to depth camera, read depth,
    and backproject to a 3D point in the RGB camera frame (right-handed, meters).

    Math/steps:
      1. RGB px (u, v) -> normalized ray in RGB cam frame (using K_rgb^(-1))
      2. Transform ray to depth camera: X_depth = R * ray_rgb + t
      3. Project to depth image, get (u_d, v_d) px
      4. Median depth in window -> Z [meters]
      5. Backproject (u_d, v_d, Z) to point in depth camera frame
      6. Transform 3D point back to RGB camera frame using inverse extrinsics

    Returns:
        (M, 3) 3D points in RGB camera coordinate frame (meters)
        M <= N: points where valid depth was found
    """
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
        ray_rgb /= ray_rgb[2]
        pt_depth = K_depth @ ray_rgb
        u_d, v_d = int(pt_depth[0]), int(pt_depth[1])
        # ensure window is inside image
        u1, u2 = max(0, u_d - half), min(w, u_d + half + 1)
        v1, v2 = max(0, v_d - half), min(h, v_d + half + 1)
        window_vals = depth_map[v1:v2, u1:u2]
        valid = window_vals[(window_vals > 0) & np.isfinite(window_vals)]
        if valid.size == 0:
            if VERBOSE:
                log.info(f"[backproj][pt#{i}] No depth: ({u_d},{v_d})")
            continue
        depth_raw = float(np.median(valid))
        Z = depth_raw * depth_scale
        # backproject depth image pixel to 3D (in depth camera frame)
        X = (pt_depth[0] - K_depth[0, 2]) * Z / K_depth[0, 0]
        Y = (pt_depth[1] - K_depth[1, 2]) * Z / K_depth[1, 1]
        pt_rgb_3d = np.array([X, Y, Z], dtype=np.float64)
        # transform back to RGB camera frame (inverse extrinsics)
        # pt_rgb_3d = R.T @ (pt_depth_3d - t)
        if VERBOSE:
            log.debug(
                f"[backproj][pt#{i}] RGB ({u:.1f},{v:.1f}) -> Depth px ({u_d},{v_d}), Z={Z:.3f} m, "
                # f"Depth3D: {pt_depth_3d}, RGB3D: {pt_rgb_3d}"
            )
        points_3d.append(pt_rgb_3d)
    return np.array(points_3d, dtype=np.float64)


# =============================================================================
# 6. VISUALIZATION AND OVERLAY UTILS
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
    """
    Overlay detected corners and their indices/IDs on either a color or depth image.

    Args:
        image: Input BGR image (if not depth).
        corners: (N,2) pixel coordinates to plot.
        ids:     (N,) corner IDs (or indices).
        prefix:  String prefix for text overlay.
        out_file: Output filename.
        is_depth: If True, renders on depth colormap.
        depth_map: Optional; if is_depth, must be provided.
    """
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
        log.info(f"[overlay] Saved corner overlay to: {out_file}")


def colorize_depth_with_legend(depth: np.ndarray) -> np.ndarray:
    """
    Render a depth map as a colored image with a value legend.
    Returns: Color image (BGR, uint8).
    """
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
# 7. POSE ESTIMATION AND HAND-EYE CALIBRATION
# =============================================================================


def estimate_pose_pnp(
    obj_pts: np.ndarray, img_pts: np.ndarray, K: np.ndarray, dist: np.ndarray
) -> np.ndarray | None:
    """
    Estimate the Charuco board pose in the RGB camera frame using OpenCV PnP.

    Args:
        obj_pts: (N,3) Charuco object points (meters, board frame).
        img_pts: (N,2) 2D image points (pixels, OpenCV).
        K: (3,3) RGB camera intrinsics.
        dist: (N,) distortion coefficients.
    Returns:
        T_cam_board: (4x4) Homogeneous transform. Maps board points to camera frame:
            X_cam = T_cam_board * X_board_homog
        Returns None if PnP fails or too few points.
    """
    if obj_pts.shape[0] < 4:
        if VERBOSE:
            log.info(f"[PnP] Not enough points for PnP: {obj_pts.shape[0]}")
        return None
    flags = (
        cv2.SOLVEPNP_IPPE_SQUARE if obj_pts.shape[0] == 4 else cv2.SOLVEPNP_ITERATIVE
    )
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=flags)
    if not ok:
        if VERBOSE:
            log.info("[PnP] PnP solve failed.")
        return None
    Rmat, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = Rmat
    T[:3, 3] = tvec.flatten()
    if VERBOSE:
        log.info(
            f"[PnP] T_cam_board: rvec (rad):{rvec.flatten()}, tvec (m):{tvec.flatten()},\nR:\n{Rmat}"
        )
    return T


def run_handeye_calibration(
    robot_T: List[np.ndarray], target_T: List[np.ndarray]
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Solve AX=XB hand-eye calibration for camera→TCP using OpenCV's algorithms.

    Args:
        robot_T: List of T_base_tcp, robot poses (4x4, meters).
        target_T: List of T_cam_board, board poses in camera frame (4x4, meters).
    Returns:
        Dict {method_name: (R_cam2tcp, t_cam2tcp)}.
        All transforms are right-handed, OpenCV, units: meters.
        Result is T_cam2tcp such that: X_tcp = T_cam2tcp * X_cam
    """
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
        R_he, t_he = cv2.calibrateHandEye(
            robot_R, robot_t, target_R, target_t, method=method
        )
        if VERBOSE:
            log.info(
                f"[handeye][{name}]\nR_cam2tcp:\n{R_he}\nt_cam2tcp (m)\n{t_he.flatten()}"
            )
        results[name] = (R_he, t_he)
    return results


def compute_handeye_errors(
    robot_T: List[np.ndarray],
    target_T: List[np.ndarray],
    solutions: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> List[Tuple[str, float, float]]:
    """
    Compute root-mean-square error (RMSE) for translation and rotation
    for each hand-eye calibration solution.

    For each (robot_T, target_T) pair:
      - Computes predicted camera pose in base frame: T_base_cam = T_base_tcp @ T_cam2tcp
      - Compares predicted camera-to-board transform to ground truth (T_cam_board)
      - Translation error: Euclidean distance between translation vectors [meters]
      - Rotation error: Angle between orientations [degrees] (via rotation matrix trace)
      - Both errors are aggregated as RMSE (root-mean-square error) over all pairs.

    Args:
        robot_T: List of T_base_tcp (4x4), robot TCP poses in base frame [meters]
        target_T: List of T_cam_board (4x4), board poses in camera frame [meters]
        solutions: Dict of {method_name: (R, t)} where R: (3,3), t: (3,)

    Returns:
        List of (method_name, translation_RMSE [meters], rotation_RMSE [degrees])
    """
    summary = []
    for name, (R_he, t_he) in solutions.items():
        T_he = np.eye(4)
        T_he[:3, :3] = R_he
        T_he[:3, 3] = t_he.flatten()
        pos_err, rot_err = [], []
        for i, (T_r, T_t) in enumerate(zip(robot_T, target_T)):
            # Predicted camera pose in base frame: T_base_cam = T_base_tcp @ T_cam2tcp
            cam_T = T_r @ T_he
            det = np.linalg.det(cam_T)
            if abs(det) < 1e-6 or not np.isfinite(det):
                log.error(
                    f"[handeye][{name}][pair{i}] T_base_cam is singular or nearly singular! det={det}"
                )
                continue
            # Error transform: difference between predicted and measured camera-to-board
            diff = np.linalg.inv(cam_T) @ T_t
            # Translation error: Euclidean distance in meters
            pos_err.append(np.linalg.norm(diff[:3, 3]))
            # Rotation error: angle in degrees between diff rotation and identity
            angle = np.arccos(np.clip((np.trace(diff[:3, :3]) - 1) / 2, -1.0, 1.0))
            rot_err.append(np.degrees(angle))
            if VERBOSE and i == 0:
                log.debug(
                    f"[handeye][{name}]\nT_base_tcp:\n{T_r}\nT_cam_board:\n{T_t}\n"
                    f"T_base_cam:\n{cam_T}\nT_error:\n{diff}"
                )
        # Root mean square error over all pose pairs
        translation_rmse = (
            float(np.sqrt(np.mean(np.square(pos_err)))) if pos_err else float("nan")
        )
        rotation_rmse = (
            float(np.sqrt(np.mean(np.square(rot_err)))) if rot_err else float("nan")
        )
        summary.append((name, translation_rmse, rotation_rmse))
    return summary


def compute_mean_reprojection_error(
    poses: List[np.ndarray],
    obj_pts_list: List[np.ndarray],
    img_pts_list: List[np.ndarray],
    K: np.ndarray,
    dist: np.ndarray,
) -> float:
    """
    Compute mean reprojection error (pixels) over all PnP-estimated board poses.

    Args:
        poses: List of T_cam_board (4x4) estimated by PnP
        obj_pts_list: List of 3D object points (board frame, meters)
        img_pts_list: List of 2D image points (pixels, OpenCV)
        K, dist: RGB camera intrinsics and distortion

    Returns:
        Mean reprojection error in pixels.
    """
    all_errs = []
    for T, obj_pts, img_pts in zip(poses, obj_pts_list, img_pts_list):
        if T is None or obj_pts.shape[0] == 0:
            continue
        rvec, _ = cv2.Rodrigues(T[:3, :3])
        tvec = T[:3, 3].reshape(-1, 1)
        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
        err = np.linalg.norm(proj.squeeze(1) - img_pts, axis=1)
        all_errs.append(err)
    if all_errs:
        all_errs = np.concatenate(all_errs)
        return float(np.sqrt(np.mean(all_errs**2)))
    return float("nan")


def is_valid_pose(T: np.ndarray) -> bool:
    """
    Validate a pose matrix (4x4): checks orthogonality, determinant, and finite values.
    """
    R_ = T[:3, :3]
    det = np.linalg.det(R_)
    ortho = np.allclose(R_ @ R_.T, np.eye(3), atol=POSE_ORTHO_TOL)
    return np.isfinite(det) and abs(det - 1.0) < POSE_DET_TOL and ortho


def is_valid_rotation(R_: np.ndarray) -> bool:
    """
    Validate a 3x3 rotation matrix for hand-eye calibration.
    """
    return (
        np.isfinite(R_).all()
        and abs(np.linalg.det(R_) - 1) < POSE_DET_TOL
        and np.allclose(R_ @ R_.T, np.eye(3), atol=POSE_ORTHO_TOL)
        and np.min(np.abs(R_)) > MIN_ROT_ELEM
    )


def filter_pose_pairs(
    robot_T: List[np.ndarray], target_T: List[np.ndarray]
) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, int]]:
    """
    Remove pose pairs with invalid matrices (rotation/NaN).
    Returns valid pairs and stats on reasons for dropping.
    """
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
                    f"[filter][pair#{i}] Dropped: {reasons}\nT_base_tcp:\n{T_r}\nT_cam_board:\n{T_t}"
                )
        else:
            robot_T_final.append(T_r)
            target_T_final.append(T_t)
            if VERBOSE:
                log.info(
                    f"[filter][pair#{i}] VALID\nT_base_tcp:\n{T_r}\nT_cam_board:\n{T_t}"
                )
    return robot_T_final, target_T_final, reason_stats


def analyze_board_surface(obj_pts: np.ndarray, pts_3d: np.ndarray) -> dict:
    """
    Оценка качества облака точек доски:
    - RMS-отклонение от наилучшей плоскости
    - Максимальное отклонение
    - Средняя глубина (по Z)
    """
    if pts_3d.shape[0] < 3:
        return {
            "rms_plane": float("nan"),
            "max_dev": float("nan"),
            "mean_z": float("nan"),
        }
    # центрируем
    xyz = pts_3d - np.mean(pts_3d, axis=0)
    _, _, vh = np.linalg.svd(xyz)
    normal = vh[-1]
    dists = np.dot(xyz, normal)
    rms_plane = np.sqrt(np.mean(dists**2))
    max_dev = np.max(np.abs(dists))
    return {
        "rms_plane": float(rms_plane),
        "max_dev": float(max_dev),
        "mean_z": float(np.mean(pts_3d[:, 2])),
    }


# =============================================================================
# 8. MAIN EXECUTION PIPELINE
# =============================================================================

if __name__ == "__main__":
    # [1] Logger setup and file search
    Logger.configure(level="DEBUG" if VERBOSE else "INFO")
    pose_json_files = list(IMG_DIR.parent.glob(POSES_JSON_PATTERN))
    if not pose_json_files:
        raise FileNotFoundError(f"No JSON files found in {IMG_DIR}")
    POSES_FILE = pose_json_files[0]

    # [2] Calibration parameter loading
    K_rgb, dist_rgb = load_intrinsics(INTR_FILE)
    K_depth, dist_depth = load_intrinsics(INTRD_FILE)
    rgb_to_depth = np.eye(4)
    rgb_to_depth[:3, :3] = np.array(EXTR_COLOR_TO_DEPTH_ROT)
    rgb_to_depth[:3, 3] = np.array(EXTR_COLOR_TO_DEPTH_TRANS)
    board, detector = configure_charuco_board(
        BOARD_SIZE, SQUARE_LEN, MARKER_LEN, CHARUCO_DICT
    )
    log_versions_and_config(board)
    import matplotlib.pyplot as plt

    depth_dir = IMG_DIR.parent / "depth"
    depth_dir.mkdir(exist_ok=True)

    # [3] Dataset loading (image/pose pairs)
    pairs = load_image_pairs(IMG_DIR)
    robot_T = load_robot_poses(POSES_FILE)
    overlay_dir = IMG_DIR.parent / "over"
    overlay_dir.mkdir(exist_ok=True)

    # [4] Corner detection, 3D/2D pairing, pose estimation
    n_mismatch = n_too_few = n_pnp_fail = n_rot_bad = 0
    target_T, obj_pts_list, img_pts_list = [], [], []
    surf_stats_list = []
    for idx, (rgb_file, depth_file) in enumerate(pairs):
        img = cv2.imread(str(rgb_file))
        depth = np.load(depth_file)
        detection = detect_charuco_corners(img, board, detector)
        if detection is None:
            continue
        img_pts, ids = detection
        obj_pts = board.getChessboardCorners()[ids]  # (N,3) — board frame
        if DISABLE_DEPTH_Z:
            pts_3d = obj_pts
        else:
            pts_3d = backproject_rgb_corners_to_3d_aligned(
                corners=img_pts, depth_map=depth, K_rgb=K_rgb, depth_scale=DEPTH_SCALE
            )
        if pts_3d.shape[0] != obj_pts.shape[0] or pts_3d.shape[0] < 6:
            continue
        # PnP: object (board frame), image pts, K, dist
        stats = analyze_board_surface(obj_pts, pts_3d)
        surf_stats_list.append(stats)
        log.info(
            f"[surface][{rgb_file.name}] RMS_plane={stats['rms_plane']:.5f} m, "
            f"max_dev={stats['max_dev']:.5f} m, mean_z={stats['mean_z']:.5f} m"
        )
        pose = estimate_pose_pnp(obj_pts, img_pts, K_rgb, dist_rgb)
        if pose is None or not is_valid_pose(pose):
            continue
        target_T.append(pose)
        obj_pts_list.append(obj_pts)
        img_pts_list.append(img_pts)

    robot_T_final, target_T_final, reason_stats = filter_pose_pairs(
        robot_T[: len(target_T)], target_T
    )
    obj_pts_list_final = obj_pts_list[: len(target_T_final)]
    img_pts_list_final = img_pts_list[: len(target_T_final)]

    mean_repr_err = compute_mean_reprojection_error(
        target_T_final, obj_pts_list_final, img_pts_list_final, K_rgb, dist_rgb
    )
    log.info(f"Mean PnP reprojection error: {mean_repr_err:.4f} px")

    results: Dict[str, Any] = {
        # "num_valid_pairs": n_valid,
        # "num_total_pairs": n_total,
        "mean_reprojection_error_px": mean_repr_err,
    }
    assert (
        len(robot_T_final)
        == len(target_T_final)
        == len(obj_pts_list_final)
        == len(img_pts_list_final)
    )
    if len(target_T_final) < 3:
        log.error("Not enough valid pairs for hand-eye calibration")
        sys.exit(1)
    for i, (T_r, T_t) in enumerate(zip(robot_T_final, target_T_final)):
        d_r = np.linalg.det(T_r[:3, :3])
        d_t = np.linalg.det(T_t[:3, :3])
        print(
            f"{i}: det robot={d_r:.6f}, det target={d_t:.6f}, finite robot={np.isfinite(T_r).all()}, finite target={np.isfinite(T_t).all()}"
        )

    # [7] Hand-eye calibration and error summary
    # if n_valid >= 3:
    solutions = run_handeye_calibration(robot_T_final, target_T_final)
    summary = compute_handeye_errors(robot_T_final, target_T_final, solutions)
    log.info("Hand-eye calibration error summary (all methods):")
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
    # else:
    #     log.error("Insufficient valid pose pairs for hand-eye calibration (need >2).")

    # [8] Output YAML
    save_yaml(IMG_DIR.parent / "handeye_res.yaml", results)

    if len(target_T_final) < 3:
        log.error("Not enough valid pairs for hand-eye calibration")
        sys.exit(1)

    mean_repr_err = compute_mean_reprojection_error(
        target_T_final, obj_pts_list_final, img_pts_list_final, K_rgb, dist_rgb
    )
    log.info(f"Mean PnP reprojection error: {mean_repr_err:.4f} px")

    results: Dict[str, Any] = {
        # "num_valid_pairs": n_valid,
        # "num_total_pairs": n_total,
        "mean_reprojection_error_px": mean_repr_err,
    }

    # [7] Hand-eye calibration and error summary
    # if n_valid >= 3:
    solutions = run_handeye_calibration(robot_T_final, target_T_final)
    summary = compute_handeye_errors(robot_T_final, target_T_final, solutions)
    log.info("Hand-eye calibration error summary (all methods):")
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
    # else:
    #     log.error("Insufficient valid pose pairs for hand-eye calibration (need >2).")

    # [8] Output YAML
    save_yaml(IMG_DIR.parent / "handeye_res.yaml", results)
