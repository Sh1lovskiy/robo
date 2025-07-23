"""
Hand-Eye Calibration Pipeline for Vision-Guided Robotics (modular, optimized)
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

BOARD_SIZE = (5, 8)
SQUARE_LEN = 0.03
OVERLAY_ENABLED = True
REPROJ_ERROR_THRESH = 1.5  # px, chessboard PnP outlier filter
VERBOSE = True

np.set_printoptions(suppress=True, precision=6, linewidth=200)
log = Logger.get_logger("handeye_chess")

# =========================== UTILITY FUNCTIONS ==============================


def get_chessboard_objpoints(board_size, square_len):
    """Generate chessboard 3D object points."""
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2)
    return objp * square_len


def detect_chessboard_corners(img, board_size):
    """Detect and refine chessboard corners in image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, board_size, None)
    if not found:
        return None
    refined = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 0.0001),
    )
    return refined.squeeze(1)


def overlay_corners(img, corners, board_size, out_file):
    """Draw and save chessboard corners overlay."""
    if not OVERLAY_ENABLED:
        return
    vis = img.copy()
    cv2.drawChessboardCorners(vis, board_size, corners.reshape(-1, 1, 2), True)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_file), vis)
    log.debug(f"Overlay saved: {out_file}")


def load_intrinsics_yml(path):
    """Load camera matrix and distortion from YAML."""
    with open(path) as f:
        d = yaml.safe_load(f)
    K = np.array(d["camera_matrix"]["data"]).reshape(3, 3)
    dist = np.array(d["distortion_coefficients"]["data"])
    return K, dist


def load_image_pairs(img_dir):
    """Return sorted pairs: (image.png, image.npy) if depth exists."""
    imgs = sorted(img_dir.glob("frame_*.png"))
    pairs = [
        (img, img.with_suffix(".npy"))
        for img in imgs
        if img.with_suffix(".npy").exists()
    ]
    log.debug(f"Loaded {len(pairs)} image/depth pairs from {img_dir}")
    return pairs


def load_robot_poses(json_file):
    """Load robot poses as SE(3) matrices from JSON log."""
    with open(json_file) as f:
        data = json.load(f)
    poses = []
    for v in data.values():
        xyz = np.array(v["tcp_coords"][:3]) / 1000.0  # mm->m
        rpy = v["tcp_coords"][3:]
        Rmat = R.from_euler("xyz", rpy, degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = Rmat
        T[:3, 3] = xyz
        poses.append(T)
    log.debug(f"Loaded {len(poses)} robot poses from {json_file}")
    return np.array(poses)


def estimate_pose_pnp(obj_pts, img_pts, K, dist):
    """Estimate pose via solvePnP and return (SE3, reproj error)."""
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist)
    if not ok:
        return None, np.inf
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    error = np.linalg.norm(proj.squeeze(1) - img_pts, axis=1)
    T = np.eye(4)
    T[:3, :3], _ = cv2.Rodrigues(rvec)
    T[:3, 3] = tvec.ravel()
    return T, np.mean(error)


def calibrate_handeye(robot_T, target_T):
    """Run all OpenCV hand-eye calibration methods."""
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

    # --- Load all calibration data ---
    K_rgb, dist_rgb = load_intrinsics_yml(INTR_FILE)
    pairs = load_image_pairs(IMG_DIR)
    pose_file = next(IMG_DIR.parent.glob(POSES_JSON_PATTERN))
    robot_T = load_robot_poses(pose_file)
    obj_pts_board = get_chessboard_objpoints(BOARD_SIZE, SQUARE_LEN)

    log.info(f"Total image pairs found: {len(pairs)}")
    log.info(f"Total robot poses found: {len(robot_T)}")

    # --- Chessboard detection & filtering loop ---
    target_T, reproj_errors, robot_Tf = [], [], []
    detected_per_img = np.zeros(len(pairs), dtype=int)
    num_corners = obj_pts_board.shape[0]

    for idx, (rgb_file, _) in enumerate(pairs):
        img = cv2.imread(str(rgb_file))
        img_pts = detect_chessboard_corners(img, BOARD_SIZE)
        if img_pts is None or img_pts.shape[0] != num_corners:
            log.debug(f"[idx={idx}] Chessboard not found or wrong corner count.")
            continue

        pose, err = estimate_pose_pnp(obj_pts_board, img_pts, K_rgb, dist_rgb)
        det = np.linalg.det(pose[:3, :3]) if pose is not None else 0
        valid = pose is not None and err < REPROJ_ERROR_THRESH and det > 0.8

        if valid:
            target_T.append(pose)
            robot_Tf.append(robot_T[idx])
            reproj_errors.append(err)
            detected_per_img[idx] = 1
            overlay_path = IMG_DIR.parent / "over" / f"overlay_{idx:03d}.png"
            overlay_corners(img, img_pts, BOARD_SIZE, overlay_path)
        else:
            log.debug(
                f"[idx={idx}] Skipped: "
                f"{'SolvePnP failed' if pose is None else ''} "
                f"{'High error' if err >= REPROJ_ERROR_THRESH else ''} "
                f"{'Bad rot (det={:.3f})'.format(det) if det <= 0.8 else ''}"
            )

    # --- Logging statistics for debug ---
    num_detected = detected_per_img.sum()
    log.debug(f"Detected chessboard in {num_detected} of {len(pairs)} images.")
    log.debug(f"Expected corners per board: {num_corners}")

    if len(reproj_errors) == 0:
        log.error("No valid chessboard poses for calibration!")
        return
    mean_reproj = float(np.mean(reproj_errors))
    log.info(f"Mean reprojection error: {mean_reproj:.3f} px")
    if len(robot_Tf) < 3:
        log.error("Not enough valid pose pairs for hand-eye calibration.")
        return

    # --- Run and report hand-eye calibration results ---
    solutions = calibrate_handeye(robot_Tf, target_T)
    table_lines = [
        "Hand-eye calibration results",
        f"{'Method':<11}|{'Translation [m]':^36}|{'Rotation [Euler, deg]':^27}",
        f"{'-'*11}|{'-'*36}|{'-'*26}",
    ]

    out_summary = {}
    for n, (R_he, t_he) in solutions.items():
        det = np.linalg.det(R_he)
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

    # --- Save calibration result as YAML ---
    out_yaml = {
        "handeye_solutions": out_summary,
        "mean_reprojection_error_px": mean_reproj,
        "chessboards_detected": int(num_detected),
        "total_images": int(len(pairs)),
        "corners_per_board": int(num_corners),
    }
    out_path = IMG_DIR.parent / "handeye_res.yaml"
    with open(out_path, "w") as f:
        yaml.safe_dump(out_yaml, f, sort_keys=False)
    log.info(f"Saved hand-eye calibration YAML to: {out_path}")


if __name__ == "__main__":
    main()
