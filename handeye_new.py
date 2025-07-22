from pathlib import Path
import numpy as np
import json, sys, yaml, cv2
from cv2 import aruco
from scipy.spatial.transform import Rotation as R
from utils.logger import Logger
from utils.settings import EXTR_COLOR_TO_DEPTH_ROT, EXTR_COLOR_TO_DEPTH_TRANS

INTR_FILE = Path("cam_params.yml")
INTRD_FILE = Path("dcam_params.yml")
IMG_DIR = Path("calib/imgs")
CHARUCO_DICT = "DICT_5X5_100"
BOARD_SIZE = (8, 5)
SQUARE_LEN = 0.035
MARKER_LEN = 0.026
DISABLE_DEPTH_Z = True
DEPTH_SCALE = 0.0010000000474974513
VERBOSE = True

log = Logger.get_logger("handeye")
np.set_printoptions(suppress=True, precision=6, linewidth=200)


def load_intrinsics(yaml_path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    if "camera_matrix" in data:
        K = np.array(data["camera_matrix"]["data"], float).reshape(3, 3)
        dist = np.array(data["distortion_coefficients"]["data"], float)
    else:
        fx, fy, cx, cy = data["fx"], data["fy"], data["cx"], data["cy"]
        dist = np.array(data.get("dist", []), float)
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
    return K, dist


def load_image_pairs(directory):
    rgb_files = sorted(directory.glob("frame_*.png"))
    pairs = [
        (rgb, rgb.with_suffix(".npy"))
        for rgb in rgb_files
        if rgb.with_suffix(".npy").exists()
    ]
    return pairs


def load_robot_poses(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    poses = []
    for v in data.values():
        x, y, z, rx, ry, rz = v["tcp_coords"]
        t = np.array([x, y, z], float) / 1000.0
        Rmat = R.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = Rmat
        T[:3, 3] = t
        poses.append(T)
    return poses


def configure_charuco_board(board_size, square_len, marker_len, dict_name):
    dict_id = getattr(aruco, dict_name)
    aruco_dict = aruco.getPredefinedDictionary(dict_id)
    board = aruco.CharucoBoard(board_size, square_len, marker_len, aruco_dict)
    detector = aruco.ArucoDetector(aruco_dict, aruco.DetectorParameters())
    return board, detector


def detect_charuco_corners(img, board, detector):
    marker_corners, marker_ids, _ = detector.detectMarkers(img)
    if marker_ids is None or len(marker_ids) == 0:
        return None
    ok, char_corners, char_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners, marker_ids, img, board
    )
    if not ok or char_ids is None or len(char_ids) < 4:
        return None
    return char_corners.squeeze(1), char_ids.flatten()


def backproject_rgb_corners_to_3d_aligned(
    corners, depth_map, K_rgb, depth_scale, window=1
):
    K_inv = np.linalg.inv(K_rgb)
    points_3d = []
    half = window // 2
    h, w = depth_map.shape[:2]
    for u, v in corners:
        u_int, v_int = int(round(u)), int(round(v))
        u1, u2 = max(0, u_int - half), min(w, u_int + half + 1)
        v1, v2 = max(0, v_int - half), min(h, v_int + half + 1)
        win = depth_map[v1:v2, u1:u2]
        vals = win[(win > 0) & np.isfinite(win)]
        if vals.size == 0:
            continue
        z = float(np.median(vals)) * depth_scale
        pt_h = np.array([u, v, 1.0], float)
        xyz = z * K_inv @ pt_h
        points_3d.append(xyz)
    return np.array(points_3d, float)


def estimate_pose_pnp(obj_pts, img_pts, K, dist):
    if obj_pts.shape[0] < 4:
        return None
    flags = (
        cv2.SOLVEPNP_IPPE_SQUARE if obj_pts.shape[0] == 4 else cv2.SOLVEPNP_ITERATIVE
    )
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=flags)
    if not ok:
        return None
    Rmat, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = Rmat
    T[:3, 3] = tvec.flatten()
    return T


def is_good_rotation(R):
    tr = np.trace(R)
    return np.linalg.det(R) > 0.99 and tr > -1.0


def run_handeye_calibration(robot_T, target_T):
    methods = {
        "tsai": cv2.CALIB_HAND_EYE_TSAI,
        "park": cv2.CALIB_HAND_EYE_PARK,
        "horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
        "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }

    robot_T = np.asarray(robot_T)
    target_T = np.asarray(target_T)

    robot_R = np.array([T[:3, :3] for T in robot_T])
    robot_t = np.array([T[:3, 3] for T in robot_T])
    target_R = np.array([T[:3, :3] for T in target_T])
    target_t = np.array([T[:3, 3] for T in target_T])

    for i, (R1, R2) in enumerate(zip(robot_R, target_R)):
        d1 = np.linalg.det(R1)
        d2 = np.linalg.det(R2)
        if not (np.isfinite(d1) and np.isfinite(d2)):
            print(f"[BAD] Pair {i}: NaN or Inf in determinant")
        if abs(d1 - 1) > 1e-3 or abs(d2 - 1) > 1e-3:
            print(f"[BAD] Pair {i}: det(robot_R)={d1:.4f}, det(target_R)={d2:.4f}")
        if abs(d1) < 1e-6 or abs(d2) < 1e-6:
            print(f"[BAD] Pair {i}: det(robot_R)={d1:.4e}, det(target_R)={d2:.4e}")
        if (
            np.isnan(R1).any()
            or np.isnan(R2).any()
            or np.isnan(robot_t[i]).any()
            or np.isnan(target_t[i]).any()
        ):
            print(f"[BAD] Pair {i}: NaN found in R or t")
        if (
            np.isinf(R1).any()
            or np.isinf(R2).any()
            or np.isinf(robot_t[i]).any()
            or np.isinf(target_t[i]).any()
        ):
            print(f"[BAD] Pair {i}: Inf found in R or t")

    def is_valid(R, t):
        d = np.linalg.det(R)
        return (
            np.isfinite(d)
            and abs(d - 1.0) < 1e-6
            and np.all(np.isfinite(R))
            and np.all(np.isfinite(t))
        )

    print("==== Checking all determinants robot_R ====")
    for i, R in enumerate(robot_R):
        d = np.linalg.det(R)
        print(f"robot_R[{i}]: det = {d:.6f}")

    print("==== Checking all determinants target_R ====")
    for i, R in enumerate(target_R):
        d = np.linalg.det(R)
    print(f"target_R[{i}]: det = {d:.6f}")
    print("nan in robot_R:", np.isnan(robot_R).any())
    print("nan in robot_t:", np.isnan(robot_t).any())
    print("nan in target_R:", np.isnan(target_R).any())
    print("nan in target_t:", np.isnan(target_t).any())
    print("inf in robot_R:", np.isinf(robot_R).any())
    print("inf in robot_t:", np.isinf(robot_t).any())
    print("inf in target_R:", np.isinf(target_R).any())
    print("inf in target_t:", np.isinf(target_t).any())

    print("robot_R shape:", robot_R.shape)
    print("robot_t shape:", robot_t.shape)
    print("target_R shape:", target_R.shape)
    print("target_t shape:", target_t.shape)

    print("Sample robot_t:", robot_t)
    print("Sample target_t:", target_t)
    print("Sample robot_R:", robot_R)
    print("Sample target_R:", target_R)
    robot_t = np.asarray(robot_t, dtype=np.float64).reshape(-1, 3, 1)
    target_t = np.asarray(target_t, dtype=np.float64).reshape(-1, 3, 1)
    robot_R = np.asarray(robot_R, dtype=np.float64)
    target_R = np.asarray(target_R, dtype=np.float64)
    good_idx = [
        i
        for i, (R1, t1, R2, t2) in enumerate(zip(robot_R, robot_t, target_R, target_t))
        if is_valid(R1, t1) and is_valid(R2, t2)
    ]
    if len(good_idx) < len(robot_R):
        print(
            f"Filtered out {len(robot_R) - len(good_idx)} invalid pose pairs due to det(R)!=1 or NaN/Inf."
        )

    robot_R = robot_R[good_idx]
    robot_t = robot_t[good_idx]
    target_R = target_R[good_idx]
    target_t = target_t[good_idx]

    print("robot_R", robot_R.shape)
    print("robot_t", robot_t.shape)
    print("target_R", target_R.shape)
    print("target_t", target_t.shape)
    if len(robot_R) < 3:
        raise RuntimeError(
            "Not enough valid pose pairs for hand-eye calibration (need >=3)"
        )

    return {
        name: cv2.calibrateHandEye(robot_R, robot_t, target_R, target_t, method=method)
        for name, method in methods.items()
    }


def compute_handeye_errors(robot_T, target_T, solutions):
    summary = []
    for name, (R_he, t_he) in solutions.items():
        T_he = np.eye(4)
        T_he[:3, :3] = R_he
        T_he[:3, 3] = t_he.flatten()
        pos_err, rot_err = [], []
        for T_r, T_t in zip(robot_T, target_T):
            cam_T = T_r @ T_he
            diff = np.linalg.inv(cam_T) @ T_t
            pos_err.append(np.linalg.norm(diff[:3, 3]))
            angle = np.arccos(np.clip((np.trace(diff[:3, :3]) - 1) / 2, -1.0, 1.0))
            rot_err.append(np.degrees(angle))
        translation_rmse = (
            float(np.sqrt(np.mean(np.square(pos_err)))) if pos_err else float("nan")
        )
        rotation_rmse = (
            float(np.sqrt(np.mean(np.square(rot_err)))) if rot_err else float("nan")
        )
        summary.append((name, translation_rmse, rotation_rmse))
    return summary


def compute_mean_reprojection_error(poses, obj_pts_list, img_pts_list, K, dist):
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


def is_valid_pose(T):
    R_ = T[:3, :3]
    det = np.linalg.det(R_)
    ortho = np.allclose(R_ @ R_.T, np.eye(3), atol=1e-6)
    return np.isfinite(det) and abs(det - 1.0) < 1e-6 and ortho


def filter_pose_pairs(robot_T, target_T):
    robot_T_final, target_T_final = [], []
    for T_r, T_t in zip(robot_T, target_T):
        if (
            is_valid_pose(T_r)
            and is_valid_pose(T_t)
            and np.any(np.isfinite(T_r))
            and np.any(np.isfinite(T_t))
        ):
            robot_T_final.append(T_r)
            target_T_final.append(T_t)
    return robot_T_final, target_T_final, {}


def analyze_board_surface(obj_pts, pts_3d):
    if pts_3d.shape[0] < 3:
        return {
            "rms_plane": float("nan"),
            "max_dev": float("nan"),
            "mean_z": float("nan"),
        }
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


def save_yaml(out_file, data):
    with open(out_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)


def is_valid_so3(Rmat, tol=1e-3):
    det = np.linalg.det(Rmat)
    return (
        np.isfinite(Rmat).all()
        and np.isclose(det, 1.0, atol=tol)
        and np.allclose(Rmat @ Rmat.T, np.eye(3), atol=tol)
    )


def is_valid_pose(T, tol=1e-3):
    return (
        T is not None
        and T.shape == (4, 4)
        and np.isfinite(T).all()
        and is_valid_so3(T[:3, :3], tol)
    )


def filter_pose_pairs(robot_T, target_T):
    robot_T_final, target_T_final = [], []
    for i, (T_r, T_t) in enumerate(zip(robot_T, target_T)):
        valid_r = is_valid_pose(T_r)
        valid_t = is_valid_pose(T_t)
        if valid_r and valid_t:
            robot_T_final.append(T_r)
            target_T_final.append(T_t)
        else:
            print(f"Drop pair {i}: valid_r={valid_r}, valid_t={valid_t}")
    return robot_T_final, target_T_final


if __name__ == "__main__":
    Logger.configure(level="DEBUG" if VERBOSE else "INFO")
    pose_json_files = list(IMG_DIR.parent.glob("*.json"))
    if not pose_json_files:
        raise FileNotFoundError(f"No JSON in {IMG_DIR}")
    POSES_FILE = pose_json_files[0]
    K_rgb, dist_rgb = load_intrinsics(INTR_FILE)
    K_depth, dist_depth = load_intrinsics(INTRD_FILE)
    rgb_to_depth = np.eye(4)
    rgb_to_depth[:3, :3] = np.array(EXTR_COLOR_TO_DEPTH_ROT)
    rgb_to_depth[:3, 3] = np.array(EXTR_COLOR_TO_DEPTH_TRANS)
    board, detector = configure_charuco_board(
        BOARD_SIZE, SQUARE_LEN, MARKER_LEN, CHARUCO_DICT
    )
    pairs = load_image_pairs(IMG_DIR)
    robot_T = load_robot_poses(POSES_FILE)
    target_T, obj_pts_list, img_pts_list = [], [], []
    for rgb_file, depth_file in pairs:
        img = cv2.imread(str(rgb_file))
        depth = np.load(depth_file)
        detection = detect_charuco_corners(img, board, detector)
        if detection is None:
            continue
        img_pts, ids = detection
        obj_pts = board.getChessboardCorners()[ids]
        if DISABLE_DEPTH_Z:
            pts_3d = obj_pts
        else:
            pts_3d = backproject_rgb_corners_to_3d_aligned(
                img_pts, depth, K_rgb, DEPTH_SCALE
            )
        if pts_3d.shape[0] != obj_pts.shape[0] or pts_3d.shape[0] < 6:
            continue
        target_T.append(estimate_pose_pnp(obj_pts, img_pts, K_rgb, dist_rgb))
        obj_pts_list.append(obj_pts)
        img_pts_list.append(img_pts)
    robot_T_final, target_T_final = [], []
    for i, (Tr, Tt) in enumerate(zip(robot_T[: len(target_T)], target_T)):
        if is_valid_pose(Tr) and is_valid_pose(Tt):
            robot_T_final.append(Tr)
            target_T_final.append(Tt)
        else:
            print(
                f"Drop pair {i}: robot det={np.linalg.det(Tr[:3,:3])}, target det={np.linalg.det(Tt[:3,:3])}"
            )
    print(f"Total valid pose pairs: {len(robot_T_final)}")

    obj_pts_list_final = obj_pts_list[: len(target_T_final)]
    img_pts_list_final = img_pts_list[: len(target_T_final)]
    mean_repr_err = compute_mean_reprojection_error(
        target_T_final, obj_pts_list_final, img_pts_list_final, K_rgb, dist_rgb
    )
    results = {"mean_reprojection_error_px": mean_repr_err}
    if len(target_T_final) < 3:
        sys.exit(1)
    robot_T_valid = []
    target_T_valid = []
    for Tr, Tt in zip(robot_T, target_T):
        if is_valid_pose(Tr) and is_valid_pose(Tt):
            robot_T_valid.append(Tr)
            target_T_valid.append(Tt)
    for i, (Tr, Tt) in enumerate(zip(robot_T_final, target_T_final)):
        Rr = Tr[:3, :3]
        Rt = Tt[:3, :3]
        if not (np.isfinite(Rr).all() and np.isfinite(Rt).all()):
            print(f"[ERROR] NaN/Inf in matrix! Pair {i}")
        det_r = np.linalg.det(Rr)
        det_t = np.linalg.det(Rt)
        if abs(det_r - 1.0) > 1e-2 or abs(det_t - 1.0) > 1e-2:
            print(f"[ERROR] Bad determinant! Pair {i}: det_r={det_r}, det_t={det_t}")
        if not np.allclose(Rr @ Rr.T, np.eye(3), atol=1e-2):
            print(f"[ERROR] Not orthogonal Rr! Pair {i}")
        if not np.allclose(Rt @ Rt.T, np.eye(3), atol=1e-2):
            print(f"[ERROR] Not orthogonal Rt! Pair {i}")

    solutions = run_handeye_calibration(robot_T_final, target_T_final)
    summary = compute_handeye_errors(robot_T_final, target_T_final, solutions)
    results["handeye_solutions"] = {
        n: {
            "R": solutions[n][0].tolist(),
            "t": solutions[n][1].flatten().tolist(),
            "t_rmse_m": t_err,
            "r_rmse_deg": r_err,
        }
        for n, t_err, r_err in summary
    }
    save_yaml(IMG_DIR.parent / "handeye_res.yaml", results)
