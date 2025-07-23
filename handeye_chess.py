"""
Hand-Eye Calibration Pipeline (3D chessboard, depth-based)
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
    if not OVERLAY_ENABLED:
        return
    vis = img.copy()
    if corners is not None:
        cv2.drawChessboardCorners(vis, board_size, corners.reshape(-1, 1, 2), True)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_file), vis)
    log.debug(f"Overlay saved: {out_file}")


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


def load_robot_poses(json_file):
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


def deproject_pixel(K, pt, d):
    x, y = pt
    X = (x - K[0, 2]) * d / K[0, 0]
    Y = (y - K[1, 2]) * d / K[1, 1]
    return np.array([X, Y, d], dtype=np.float32)


def show_depth_hist_cv(depth, bins=200, winname="Depth Histogram"):
    vals = depth[np.isfinite(depth) & (depth > 0)]
    if vals.size == 0:
        return
    hist, bin_edges = np.histogram(vals, bins=bins)
    hist_img_h, hist_img_w = 300, 600
    img = np.full((hist_img_h, hist_img_w, 3), 255, np.uint8)

    hist = hist.astype(np.float32)
    hist /= hist.max()
    hist *= hist_img_h - 20
    bin_w = hist_img_w // bins

    for i in range(bins - 1):
        pt1 = (i * bin_w, hist_img_h - int(hist[i]))
        pt2 = ((i + 1) * bin_w, hist_img_h)
        cv2.rectangle(img, pt1, pt2, (0, 150, 255), -1)

    min_val, max_val = float(vals.min()), float(vals.max())
    cv2.putText(
        img,
        f"min={min_val:.2f}m max={max_val:.2f}m",
        (5, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (40, 40, 40),
        2,
    )
    cv2.putText(
        img,
        "Depth histogram",
        (5, hist_img_h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
    )
    cv2.imshow(winname, img)
    cv2.waitKey(0)
    cv2.destroyWindow(winname)


def get_corners_3d_from_depth(corners, depth, K, win=5, thr=0.01):

    print("Depth map info:")
    print("  shape:", depth.shape)
    print("  dtype:", depth.dtype)
    print("  min:", np.nanmin(depth))
    print("  max:", np.nanmax(depth))
    print("  mean:", np.nanmean(depth))
    print("  median:", np.nanmedian(depth))
    if np.issubdtype(depth.dtype, np.integer):
        vals, counts = np.unique(depth, return_counts=True)
        print(f"  unique vals: {len(vals)}  (top-10: {vals[:10]})")
    else:
        finite_mask = np.isfinite(depth)
        print("  finite min:", np.min(depth[finite_mask]))
        print("  finite max:", np.max(depth[finite_mask]))
        print("  num nans:", np.sum(np.isnan(depth)))
        print("  num +inf:", np.sum(np.isposinf(depth)))
        print("  num -inf:", np.sum(np.isneginf(depth)))
        print("  num <= 0:", np.sum(depth <= 0))
    show_depth_hist_cv(depth)

    pts3d = []
    for pt in corners:
        x, y = int(round(pt[0])), int(round(pt[1]))
        H, W = depth.shape
        if x < 0 or x >= W or y < 0 or y >= H:
            pts3d.append([np.nan, np.nan, np.nan])
            continue
        x0, x1 = max(0, x - win), min(W, x + win + 1)
        y0, y1 = max(0, y - win), min(H, y + win + 1)
        patch = depth[y0:y1, x0:x1]
        patch = patch[(patch > thr) & (patch < 5.0) & np.isfinite(patch)]
        if patch.size == 0:
            pts3d.append([np.nan, np.nan, np.nan])
            continue
        d = np.median(patch)
        if np.isnan(d) or d < thr or d > 5.0:
            pts3d.append([np.nan, np.nan, np.nan])
            continue
        pt3d = deproject_pixel(K, pt, d)
        pts3d.append(pt3d)
    return np.array(pts3d, dtype=np.float32)


def umeyama_align(src, dst, with_scale=False):
    """Rigid (Umeyama) transform, dst = R @ src + t"""
    src = src[~np.isnan(src).any(axis=1)]
    dst = dst[~np.isnan(dst).any(axis=1)]
    if src.shape[0] < 4 or dst.shape[0] < 4 or src.shape[0] != dst.shape[0]:
        return None, np.inf
    mu_src = np.mean(src, axis=0)
    mu_dst = np.mean(dst, axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    H = src_c.T @ dst_c
    U, S, Vt = np.linalg.svd(H)
    R_ = Vt.T @ U.T
    if np.linalg.det(R_) < 0:
        Vt[-1, :] *= -1
        R_ = Vt.T @ U.T
    t_ = mu_dst - R_ @ mu_src
    T = np.eye(4)
    T[:3, :3] = R_
    T[:3, 3] = t_
    # error as mean distance
    src_aligned = (R_ @ src.T).T + t_
    err = np.mean(np.linalg.norm(src_aligned - dst, axis=1))
    return T, err


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
    obj_pts_board = get_chessboard_objpoints(BOARD_SIZE, SQUARE_LEN)

    log.info(f"Total image pairs found: {len(pairs)}")
    log.info(f"Total robot poses found: {len(robot_T)}")

    # --- Chessboard detection & filtering loop ---
    target_T, errors_3d, robot_Tf = [], [], []
    detected_per_img = np.zeros(len(pairs), dtype=int)
    num_corners = obj_pts_board.shape[0]

    for idx, (rgb_file, depth_file) in enumerate(pairs):
        img = cv2.imread(str(rgb_file))
        depth = np.load(depth_file)
        img_pts = detect_chessboard_corners(img, BOARD_SIZE)
        if img_pts is None or img_pts.shape[0] != num_corners:
            log.debug(f"[idx={idx}] Chessboard not found or wrong corner count.")
            continue
        # 3D extraction from depth
        pts3d = get_corners_3d_from_depth(img_pts, depth, K_rgb)
        valid_mask = ~np.isnan(pts3d[:, 0])
        n_valid = np.sum(valid_mask)
        print(f"Total corners: {num_corners}, valid 3D points: {n_valid}")
        if n_valid < 4:
            log.debug(f"[idx={idx}] Chessboard: <4 valid 3D points from depth.")
            continue
        pose, err = umeyama_align(obj_pts_board[valid_mask], pts3d[valid_mask])
        det = np.linalg.det(pose[:3, :3]) if pose is not None else 0
        valid = pose is not None and n_valid >= 4 and det > 0.8

        if valid:
            target_T.append(pose)
            robot_Tf.append(robot_T[idx])
            errors_3d.append(err)
            detected_per_img[idx] = 1
            overlay_path = IMG_DIR.parent / "over" / f"overlay_{idx:03d}.png"
            overlay_corners(img, img_pts, BOARD_SIZE, overlay_path)
            log.debug(f"[idx={idx}] {n_valid} points, mean 3D error = {err:.4f} m")
        else:
            log.debug(
                f"[idx={idx}] Skipped: "
                f"{'Umeyama failed' if pose is None else ''} "
                f"{'Too few 3D points' if n_valid < 4 else ''} "
                f"{'Bad rot (det={:.3f})'.format(det) if det <= 0.8 else ''}"
            )

    # --- Logging statistics for debug ---
    num_detected = detected_per_img.sum()
    log.debug(f"Detected chessboard in {num_detected} of {len(pairs)} images.")
    log.debug(f"Expected corners per board: {num_corners}")

    if len(errors_3d) == 0:
        log.error("No valid chessboard poses for calibration!")
        return
    mean_error = float(np.mean(errors_3d))
    log.info(f"Mean 3D alignment error: {mean_error:.3f} m")
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
        "mean_3d_alignment_error_m": mean_error,
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
