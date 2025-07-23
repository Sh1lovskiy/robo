"""
Hand-Eye Calibration Pipeline for Vision-Guided Robotics (AprilTag version)
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

APRIL_DICT = "DICT_APRILTAG_36H11"
BOARD_SIZE = (7, 5)
TAG_SIZE = 0.029
OVERLAY_ENABLED = True
REPROJ_ERROR_THRESH = 1000
VERBOSE = True

np.set_printoptions(suppress=True, precision=6, linewidth=200)
log = Logger.get_logger("handeye_april")

# =========================== UTILITY FUNCTIONS ==============================


def get_april_grid(board_size, tag_size, dict_name, marker_sep=None):
    marker_sep = 0.0087
    april_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    board = cv2.aruco.GridBoard(
        (board_size[0], board_size[1]), tag_size, marker_sep, april_dict
    )
    return board, april_dict


def detect_april_tags(img, board, april_dict):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(april_dict, params)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) < 4:
        return None, None

    tag_centers = np.array(
        [c.mean(axis=1).flatten() for c in corners], dtype=np.float32
    )
    return tag_centers, ids.flatten()


def overlay_april(img, tag_centers, tag_ids, board, out_file):
    if not OVERLAY_ENABLED:
        return
    if (
        tag_centers is None
        or tag_ids is None
        or len(tag_centers) != len(tag_ids)
        or len(tag_centers) == 0
    ):
        log.debug("overlay_april: skip overlay (invalid centers/ids)")
        return
    vis = img.copy()
    for pt, idx in zip(tag_centers, tag_ids):
        cv2.circle(vis, tuple(np.int32(pt)), 7, (0, 255, 0), 2)
        cv2.putText(
            vis,
            str(idx),
            tuple(np.int32(pt)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_file), vis)
    log.debug(f"Overlay saved: {out_file}")


def get_board_obj_points(tag_ids, board):
    obj_points = board.getObjPoints()
    obj_ids = board.getIds().flatten()
    id2center = {
        int(id_): np.mean(obj_points[i], axis=0) for i, id_ in enumerate(obj_ids)
    }
    obj_pts = [id2center[int(idx)] for idx in tag_ids if int(idx) in id2center]
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


def estimate_pose_pnp(obj_pts, img_pts, K, dist):
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
    board, april_dict = get_april_grid(BOARD_SIZE, TAG_SIZE, APRIL_DICT)

    log.info(f"Total image pairs found: {len(pairs)}")
    log.info(f"Total robot poses found: {len(robot_T)}")

    target_T, reproj_errors, robot_Tf = [], [], []
    detected_per_img = np.zeros(len(pairs), dtype=int)
    num_tags = BOARD_SIZE[0] * BOARD_SIZE[1]

    for idx, (rgb_file, _) in enumerate(pairs):
        img = cv2.imread(str(rgb_file))
        tag_centers, tag_ids = detect_april_tags(img, board, april_dict)
        if tag_centers is None or len(tag_centers) < 4:
            log.debug(f"[idx={idx}] April: not found or <4 tags.")
            continue
        if tag_ids is None or len(tag_centers) != len(tag_ids):
            log.debug(f"[idx={idx}] April: invalid id/center count.")
            continue
        obj_pts = get_board_obj_points(tag_ids, board)
        if obj_pts.shape[0] != tag_centers.shape[0]:
            log.debug(f"[idx={idx}] April: tag count mismatch.")
            continue
        pose, err = estimate_pose_pnp(obj_pts, tag_centers, K_rgb, dist_rgb)
        det = np.linalg.det(pose[:3, :3]) if pose is not None else 0
        valid = pose is not None and err < REPROJ_ERROR_THRESH and det > 0.8
        if valid:
            target_T.append(pose)
            robot_Tf.append(robot_T[idx])
            reproj_errors.append(err)
            detected_per_img[idx] = 1
            overlay_path = IMG_DIR.parent / "over" / f"overlay_{idx:03d}.png"
            overlay_april(img, tag_centers, tag_ids, board, overlay_path)
        else:
            log.debug(
                f"[idx={idx}] Skipped: "
                f"{'SolvePnP failed' if pose is None else ''} "
                f"{'High reproj error' if err >= REPROJ_ERROR_THRESH else ''} "
                f"{'Bad rot (det={:.3f})'.format(det) if det <= 0.8 else ''}"
            )

    num_detected = detected_per_img.sum()
    log.debug(f"Detected April board in {num_detected} of {len(pairs)} images.")
    log.debug(f"Max possible tags per board: {num_tags}")

    if len(reproj_errors) == 0:
        log.error("No valid AprilTag board poses for calibration!")
        return
    mean_reproj = float(np.mean(reproj_errors))
    log.info(f"Mean reprojection error: {mean_reproj:.3f} px")
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
        "mean_reprojection_error_px": mean_reproj,
        "april_detected": int(num_detected),
        "total_images": int(len(pairs)),
        "max_tags_per_board": int(num_tags),
    }
    out_path = IMG_DIR.parent / "handeye_res.yaml"
    with open(out_path, "w") as f:
        yaml.safe_dump(out_yaml, f, sort_keys=False)
    log.info(f"Saved hand-eye calibration YAML to: {out_path}")


if __name__ == "__main__":
    main()
