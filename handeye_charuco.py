"""
Hand-Eye Calibration Pipeline for Vision-Guided Robotics (Charuco version, с использованием 3D-углов Charuco по depth)
"""

from pathlib import Path
import numpy as np
import cv2
import yaml
import json
from scipy.spatial.transform import Rotation as R
from utils.logger import Logger

# =========================== CONSTANTS & CONFIG =============================

INTR_FILE = Path("cam_params.yml")
IMG_DIR = Path("calib/imgs")
POSES_JSON_PATTERN = "*.json"

CHARUCO_DICT = "DICT_5X5_100"
BOARD_SIZE = (8, 5)  # (squares_x, squares_y)
SQUARE_LEN = 0.035  # [m]
MARKER_LEN = 0.026  # [m]
OVERLAY_ENABLED = True
REPROJ_ERROR_THRESH = 1.5  # px (not used for 3D, only for overlay)
VERBOSE = True

np.set_printoptions(suppress=True, precision=6, linewidth=200)
log = Logger.get_logger("handeye_charuco")

# =========================== UTILITY FUNCTIONS ==============================


def get_charuco_board(board_size, square_len, marker_len, dict_name):
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    board = cv2.aruco.CharucoBoard(board_size, square_len, marker_len, aruco_dict)
    return board, aruco_dict


def detect_charuco_corners(img, board, aruco_dict):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
    if len(corners) == 0 or ids is None:
        return None, None
    ret, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, board
    )
    if not ret or ch_corners is None or ch_ids is None or len(ch_corners) < 4:
        return None, None
    return ch_corners.reshape(-1, 2), ch_ids.flatten()


def overlay_charuco(img, ch_corners, ch_ids, board, out_file):
    if not OVERLAY_ENABLED:
        return
    if (
        ch_corners is None
        or ch_ids is None
        or len(ch_corners) != len(ch_ids)
        or len(ch_corners) == 0
    ):
        log.debug("overlay_charuco: skip overlay (invalid corners/ids)")
        return
    vis = img.copy()
    cc = ch_corners.reshape(-1, 1, 2).astype(np.float32)
    ci = ch_ids.reshape(-1, 1).astype(np.int32)
    cv2.aruco.drawDetectedCornersCharuco(vis, cc, ci, (0, 255, 0))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_file), vis)
    log.debug(f"Overlay saved: {out_file}")


def get_board_obj_points(ch_ids, board):
    obj_pts = [board.getChessboardCorners()[i] for i in ch_ids]
    return np.array(obj_pts, dtype=np.float32)


def load_intrinsics_yml(path):
    with open(path) as f:
        d = yaml.safe_load(f)
    K = np.array(d["camera_matrix"]["data"]).reshape(3, 3)
    dist = np.array(d["distortion_coefficients"]["data"])
    return K, dist


def load_image_pairs(img_dir):
    imgs = sorted(img_dir.glob("frame_*.png"))
    pairs = [
        (img, img.with_suffix(".npy"))
        for img in imgs
        if img.with_suffix(".npy").exists()
    ]
    log.debug(f"Loaded {len(pairs)} image/depth pairs from {img_dir}")
    return pairs


def load_depth_map(path):
    return np.load(path)


def load_robot_poses(json_file):
    with open(json_file) as f:
        data = json.load(f)
    poses = []
    for v in data.values():
        xyz = np.array(v["tcp_coords"][:3]) / 1000.0
        rpy = v["tcp_coords"][3:]
        Rmat = R.from_euler("xyz", rpy, degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = Rmat
        T[:3, 3] = xyz
        poses.append(T)
    log.debug(f"Loaded {len(poses)} robot poses from {json_file}")
    return np.array(poses)


def deproject_pixel(K, pt2d, depth):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x, y = pt2d
    X = (x - cx) * depth / fx
    Y = (y - cy) * depth / fy
    Z = depth
    return np.array([X, Y, Z], dtype=np.float32)


def get_depth_in_window(depth_map, x, y, win=5):
    H, W = depth_map.shape
    x0, x1 = max(0, x - win), min(W, x + win + 1)
    y0, y1 = max(0, y - win), min(H, y + win + 1)
    patch = depth_map[y0:y1, x0:x1]
    patch = patch[np.isfinite(patch) & (patch > 0.1) & (patch < 5.0)]
    if patch.size == 0:
        return np.nan
    return np.median(patch)


def get_charuco_3d_from_depth(ch_corners, depth_map, K, win=2, outlier_thr=None):
    pts3d = []
    H, W = depth_map.shape
    total = len(ch_corners)
    valid = 0

    for pt in ch_corners:
        x, y = int(round(pt[0])), int(round(pt[1]))
        if x < 0 or x >= W or y < 0 or y >= H:
            pts3d.append([np.nan, np.nan, np.nan])
            continue

        x0, x1 = int(max(0, x - win)), int(min(W, x + win + 1))
        y0, y1 = int(max(0, y - win)), int(min(H, y + win + 1))
        patch = depth_map[y0:y1, x0:x1]
        patch = patch[
            (patch > 0.1)
            & (patch < 5.0)
            & (np.abs(patch - 65.535) > 1e-3)
            & np.isfinite(patch)
        ]
        if patch.size == 0:
            pts3d.append([np.nan, np.nan, np.nan])
            continue

        d = np.median(patch)
        if np.isnan(d) or d < 0.1 or d > 5.0 or abs(d - 65.535) < 1e-3:
            pts3d.append([np.nan, np.nan, np.nan])
            continue

        pt3d = deproject_pixel(K, pt, d)
        pts3d.append(pt3d)
        valid += 1

    pts3d = np.array(pts3d, dtype=np.float32)

    # Для дебага:
    print(f"Total corners: {total}, valid 3D points: {valid}")

    if outlier_thr is not None and valid > 4:
        med = np.nanmedian(pts3d, axis=0)
        dists = np.linalg.norm(pts3d - med, axis=1)
        mask = (dists < outlier_thr) & np.all(np.isfinite(pts3d), axis=1)
        pts3d[~mask] = np.nan

    return pts3d


def svd_rigid_transform(A, B):
    assert len(A) == len(B) and len(A) >= 3
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    Rmat = Vt.T @ U.T
    if np.linalg.det(Rmat) < 0:
        Vt[-1, :] *= -1
        Rmat = Vt.T @ U.T
    tvec = centroid_B - Rmat @ centroid_A
    T = np.eye(4)
    T[:3, :3] = Rmat
    T[:3, 3] = tvec
    return T


def calibrate_handeye(robot_T, target_T):
    methods = {
        "tsai": cv2.CALIB_HAND_EYE_TSAI,
        "park": cv2.CALIB_HAND_EYE_PARK,
        "horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
        "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    out = {}
    for n, m in methods.items():
        R_he, t_he = cv2.calibrateHandEye(
            [T[:3, :3] for T in robot_T],
            [T[:3, 3] for T in robot_T],
            [T[:3, :3] for T in target_T],
            [T[:3, 3] for T in target_T],
            method=m,
        )
        out[n] = (R_he, t_he)
    return out


# =============================== MAIN BLOCK =================================


def main():
    Logger.configure(level="DEBUG" if VERBOSE else "INFO")
    K_rgb, dist_rgb = load_intrinsics_yml(INTR_FILE)
    pairs = load_image_pairs(IMG_DIR)
    pose_file = next(IMG_DIR.parent.glob(POSES_JSON_PATTERN))
    robot_T = load_robot_poses(pose_file)
    board, aruco_dict = get_charuco_board(
        BOARD_SIZE, SQUARE_LEN, MARKER_LEN, CHARUCO_DICT
    )

    log.info(f"Total image pairs found: {len(pairs)}")
    log.info(f"Total robot poses found: {len(robot_T)}")

    target_T, robot_Tf = [], []
    num_corners = BOARD_SIZE[0] * BOARD_SIZE[1]
    detected_per_img = np.zeros(len(pairs), dtype=int)
    depth_scale = 1.0

    for idx, (rgb_file, depth_file) in enumerate(pairs):
        img = cv2.imread(str(rgb_file))
        depth = load_depth_map(depth_file)
        ch_corners, ch_ids = detect_charuco_corners(img, board, aruco_dict)
        if ch_corners is None or len(ch_corners) < 4:
            log.debug(f"[idx={idx}] Charuco: not found or <4 corners.")
            continue
        obj_pts = get_board_obj_points(ch_ids, board)
        pts3d = get_charuco_3d_from_depth(ch_corners, depth, K_rgb, depth_scale)
        mask = ~np.isnan(pts3d).any(axis=1)
        if np.count_nonzero(mask) < 4:
            log.debug(f"[idx={idx}] Charuco: <4 valid 3D points from depth.")
            continue
        obj_pts_valid = obj_pts[mask]
        pts3d_valid = pts3d[mask]
        if obj_pts_valid.shape[0] != pts3d_valid.shape[0]:
            log.debug(f"[idx={idx}] Charuco: point count mismatch after filtering.")
            continue

        T_obj_to_cam = svd_rigid_transform(obj_pts_valid, pts3d_valid)
        det = np.linalg.det(T_obj_to_cam[:3, :3])
        if det < 0.8:
            log.debug(f"[idx={idx}] Skipped: Bad rot (det={det:.3f})")
            continue

        pts3d_proj = (T_obj_to_cam[:3, :3] @ obj_pts_valid.T).T + T_obj_to_cam[:3, 3]
        err = np.linalg.norm(pts3d_proj - pts3d_valid, axis=1)
        mean_err = np.mean(err)

        target_T.append(T_obj_to_cam)
        robot_Tf.append(robot_T[idx])
        detected_per_img[idx] = 1

        overlay_path = IMG_DIR.parent / "over" / f"overlay_{idx:03d}.png"
        overlay_charuco(img, ch_corners, ch_ids, board, overlay_path)

        log.debug(
            f"[idx={idx}] {obj_pts_valid.shape[0]} points, mean 3D error = {mean_err:.4f} m"
        )

    num_detected = detected_per_img.sum()
    log.debug(f"Detected Charuco board in {num_detected} of {len(pairs)} images.")
    log.debug(f"Max possible corners per board: {num_corners}")

    if len(target_T) == 0:
        log.error("No valid Charuco board poses for calibration!")
        return
    if len(robot_Tf) < 3:
        log.error("Not enough valid pose pairs for hand-eye calibration.")
        return

    solutions = calibrate_handeye(robot_Tf, target_T)
    table_lines = [
        "Hand-eye calibration results",
        f"{'Method':<11}|{'Translation [m]':^36}|{'Rotation [Euler, deg]':^27}",
        f"{'-'*11}|{'-'*36}|{'-'*26}",
    ]
    out_summary = {}
    for n, (R_he, t_he) in solutions.items():
        try:
            angles = R.from_matrix(R_he).as_euler("xyz", degrees=True)
        except Exception as e:
            angles = "ERROR"
            log.error(f"[handeye][{n}] Euler conversion failed: {e}")
        t_str = "[{: 10.6f} {: 10.6f} {: 10.6f}]".format(*t_he.ravel())
        angles_str = (
            "[{: 7.4f} {: 7.4f} {: 7.4f}]".format(*angles)
            if isinstance(angles, np.ndarray)
            else f"{angles:>27}"
        )
        table_lines.append(f"{n:<10} | {t_str} | {angles_str}")
        out_summary[n] = {"R": R_he.tolist(), "t": t_he.flatten().tolist()}

    log.info("\n".join(table_lines))

    out_yaml = {
        "handeye_solutions": out_summary,
        "charuco_detected": int(num_detected),
        "total_images": int(len(pairs)),
        "max_corners_per_board": int(num_corners),
    }
    out_path = IMG_DIR.parent / "handeye_res.yaml"
    with open(out_path, "w") as f:
        yaml.safe_dump(out_yaml, f, sort_keys=False)
    log.info(f"Saved hand-eye calibration YAML to: {out_path}")


if __name__ == "__main__":
    main()
