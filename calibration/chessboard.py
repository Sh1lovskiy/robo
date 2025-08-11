#!/usr/bin/env python3
"""
Hand-eye calibration from chessboard datasets.

This script takes a folder with images like:
  calib/<TIMESTAMP>/{000_rgb.png, 001_rgb.png, ...} and a poses.json
(poses are robot TCP poses per frame index), detects a chessboard,
solves PnP per frame, and runs OpenCV hand-eye calibration for
a camera mounted on the robot end-effector (eye-in-hand).

Keyboard:
- Press 'q' or 'ESC' during processing to abort early.

Outputs:
- handeye.json (or handeye_<method>.json if --method all)
- Optional corner overlays saved to <dataset>/overlays/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from utils.logger import Logger, CaptureStderrToLogger
from utils.error_tracker import ErrorTracker
from utils.keyboard import GlobalKeyListener, TerminalEchoSuppressor

# =============================================================================
# defaults
# =============================================================================

# Paths & data layout
DEFAULT_DATASET_DIR = Path("calib/20250808_163328")
DEFAULT_INTRINSICS_PATH = Path(".data/params/realsense_params.json")
DEFAULT_STREAM = "color"

# Chessboard
DEFAULT_BOARD_SPEC = "8x5"
DEFAULT_SQUARE_SIZE_M = 0.03

# Hand-eye
DEFAULT_METHOD = "daniilidis"  # tsai|park|horaud|andreff|daniilidis|all
DEFAULT_EULER_ORDER = "XYZ"

# Overlays & verbosity
DEFAULT_SAVE_OVERLAYS = True
DEFAULT_OVERLAY_SUBDIR = "overlays"
DEFAULT_MAX_FRAMES: Optional[int] = None
DEFAULT_VERBOSE = False

# OpenCV constants
CORNER_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 1e-3)
CHESSBOARD_FLAGS = (
    cv2.CALIB_CB_ADAPTIVE_THRESH
    | cv2.CALIB_CB_NORMALIZE_IMAGE
    | cv2.CALIB_CB_FAST_CHECK
)
PNP_FLAGS = cv2.SOLVEPNP_ITERATIVE

METHOD_MAP = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}

KEYS_ABORT = "q"
OVERLAY_SUFFIX = "_overlay.png"

log = Logger.get_logger("chessboard")


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class CalibConfig:
    dataset_dir: Path
    intrinsics_path: Path
    stream: str
    board_rows: int
    board_cols: int
    square_size_m: float
    method: str
    save_overlays: bool
    overlay_dir: Path
    max_frames: Optional[int] = None
    euler_order: str = DEFAULT_EULER_ORDER
    verbose: bool = DEFAULT_VERBOSE

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "CalibConfig":
        rows, cols = parse_board(args.board or DEFAULT_BOARD_SPEC)

        dataset_dir = Path(args.dataset or DEFAULT_DATASET_DIR)
        intrinsics_path = Path(args.intrinsics or DEFAULT_INTRINSICS_PATH)
        stream = (args.stream or DEFAULT_STREAM).lower()
        square = float(args.square or DEFAULT_SQUARE_SIZE_M)
        method = (args.method or DEFAULT_METHOD).lower()
        save_ov = bool(args.save_overlays or DEFAULT_SAVE_OVERLAYS)
        overlay_dir = Path(args.overlay_dir or DEFAULT_OVERLAY_SUBDIR)
        if not overlay_dir.is_absolute():
            overlay_dir = dataset_dir / overlay_dir

        return cls(
            dataset_dir=dataset_dir,
            intrinsics_path=intrinsics_path,
            stream=stream,
            board_rows=rows,
            board_cols=cols,
            square_size_m=square,
            method=method,
            save_overlays=save_ov,
            overlay_dir=overlay_dir,
            max_frames=(
                args.max_frames if args.max_frames is not None else DEFAULT_MAX_FRAMES
            ),
            euler_order=args.euler_order or DEFAULT_EULER_ORDER,
            verbose=bool(args.verbose or DEFAULT_VERBOSE),
        )


# =============================================================================
# Helpers
# =============================================================================


def parse_board(spec: str) -> Tuple[int, int]:
    """Parse 'RxC' board specification into (rows, cols)."""
    parts = spec.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid board spec: {spec}")
    rows, cols = int(parts[0]), int(parts[1])
    if rows < 2 or cols < 2:
        raise ValueError(f"Board must be at least 2x2; got {rows}x{cols}")
    return rows, cols


def make_objpoints(rows: int, cols: int, square: float) -> np.ndarray:
    """Create planar chessboard object points with Z=0."""
    obj = np.zeros((rows * cols, 3), np.float32)
    xs, ys = np.meshgrid(np.arange(cols), np.arange(rows))
    obj[:, 0] = xs.reshape(-1) * square
    obj[:, 1] = ys.reshape(-1) * square
    return obj


def load_intrinsics(path: Path, stream: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load camera intrinsics from:
    RealSense-style JSON: { intrinsics: { <stream>: {fx, fy, ppx|cx, ppy|cy, coeffs[]} } }
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Intrinsics file not found: {path}")

    if path.suffix.lower() in {".yml", ".yaml"}:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        K = np.array(data["camera_matrix"]["data"], float).reshape(3, 3)
        dist = np.array(data["distortion_coefficients"]["data"], float).ravel()
        return K, dist

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "camera_matrix" in data and "distortion_coefficients" in data:
        K = np.array(data["camera_matrix"], float).reshape(3, 3)
        dist = np.array(data["distortion_coefficients"], float).ravel()
        return K, dist

    intr = data.get("intrinsics", {}).get(stream)
    if intr is None:
        raise KeyError(
            f"Stream '{stream}' not found in intrinsics {list(data.get('intrinsics', {}).keys())}"
        )
    fx, fy = float(intr["fx"]), float(intr["fy"])
    cx = float(intr.get("ppx", intr.get("cx")))
    cy = float(intr.get("ppy", intr.get("cy")))
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], float)
    coeffs = intr.get("coeffs") or [0, 0, 0, 0, 0]
    dist = np.array(coeffs, float).ravel()
    return K, dist


def load_dataset_pairs(dataset_dir: Path) -> List[Tuple[str, Path]]:
    """Return sorted list of (index, rgb_path) pairs from <dataset>/*_rgb.(png|jpg|jpeg)."""
    exts = (".png", ".jpg", ".jpeg")
    files = sorted(
        [
            p
            for p in dataset_dir.iterdir()
            if p.suffix.lower() in exts and p.name.endswith("_rgb" + p.suffix)
        ]
    )
    pairs: List[Tuple[str, Path]] = []
    for p in files:
        stem = p.stem
        idx = stem.split("_")[0]
        if idx.isdigit():
            pairs.append((idx, p))
    return pairs


def load_poses(path: Path) -> Dict[str, Dict[str, float]]:
    """Load TCP poses from JSON: { "000": {"x":..,"y":..,"z":..,"Rx":..,"Ry":..,"Rz":..}, ... }"""
    if not path.exists():
        raise FileNotFoundError(f"poses.json not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: {kk: float(vv) for kk, vv in v.items()} for k, v in data.items()}


def euler_deg_to_R(rx: float, ry: float, rz: float, order: str) -> np.ndarray:
    """Convert Euler angles in degrees to rotation matrix with given order."""
    return R.from_euler(order.lower(), [rx, ry, rz], degrees=True).as_matrix()


def invert_RT(Rm: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Invert rotation+translation transform."""
    R_inv = Rm.T
    t_inv = -R_inv @ t.reshape(3, 1)
    return R_inv, t_inv


def solve_pnp(
    K: np.ndarray, dist: np.ndarray, obj_pts: np.ndarray, img_pts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve PnP and return (R_tc, t_tc)."""
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts,
        img_pts.reshape(-1, 1, 2),
        K,
        dist,
        flags=PNP_FLAGS,
    )
    if not ok:
        raise cv2.error("solvePnP failed")
    Rm, _ = cv2.Rodrigues(rvec)
    return Rm, tvec.reshape(3, 1)


def detect_chessboard(
    gray: np.ndarray, rows: int, cols: int
) -> Tuple[bool, np.ndarray]:
    """Find chessboard corners and refine them."""
    found, corners = cv2.findChessboardCorners(
        gray, (cols, rows), flags=CHESSBOARD_FLAGS
    )
    if not found:
        return False, corners
    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CORNER_CRITERIA)
    return True, corners.reshape(-1, 2)


def project_rmse(
    K: np.ndarray,
    dist: np.ndarray,
    Rm: np.ndarray,
    t: np.ndarray,
    obj_pts: np.ndarray,
    img_pts: np.ndarray,
) -> float:
    """Compute per-frame reprojection RMSE."""
    rvec, _ = cv2.Rodrigues(Rm)
    proj, _ = cv2.projectPoints(obj_pts, rvec, t, K, dist)
    err = proj.reshape(-1, 2) - img_pts.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(err * err, axis=1))))


def try_autodetect_board(
    sample_paths: List[Path],
    candidates: List[Tuple[int, int]],
    max_trials: int = 8,
) -> Optional[Tuple[int, int]]:
    """
    Try a few images and a set of candidate (rows, cols) to guess the board size.
    Returns the first working (rows, cols), or None if none worked.
    """
    trials = sample_paths[:max_trials]
    for rows, cols in candidates:
        ok_count = 0
        for p in trials:
            img = cv2.imread(str(p))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ok, _ = detect_chessboard(gray, rows, cols)
            if ok:
                ok_count += 1
                if ok_count >= 2:
                    log.info(f"Autodetected board as {rows}x{cols} on path {p.parent}")
                    return rows, cols
    return None


# =============================================================================
# Frame processing
# =============================================================================


def process_frame(
    idx: str,
    path: Path,
    poses: Dict[str, Dict[str, float]],
    obj_pts: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    cfg: CalibConfig,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]]:
    """Process one frame and return (R_gb, t_gb, R_tc, t_tc, rmse) or None if failed."""
    img = cv2.imread(str(path))
    if img is None:
        log.warning(f"Failed to read image: {path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok, corners = detect_chessboard(gray, cfg.board_rows, cfg.board_cols)
    if not ok:
        log.debug(f"No chessboard in frame {idx} ({cfg.board_rows}x{cfg.board_cols})")
        return None

    try:
        R_tc, t_tc = solve_pnp(K, dist, obj_pts, corners)
    except cv2.error as e:
        log.debug(f"PnP failed for frame {idx}: {e}")
        return None

    pose = poses.get(idx)
    if pose is None:
        log.debug(f"No pose for frame index {idx}, skipping")
        return None

    R_bg = euler_deg_to_R(pose["Rx"], pose["Ry"], pose["Rz"], cfg.euler_order)
    t_bg = np.array([pose["x"], pose["y"], pose["z"]], float).reshape(3, 1)
    R_gb, t_gb = invert_RT(R_bg, t_bg)

    rmse = project_rmse(K, dist, R_tc, t_tc, obj_pts, corners)

    if cfg.save_overlays:
        try:
            overlay = img.copy()
            cv2.drawChessboardCorners(
                overlay,
                (cfg.board_cols, cfg.board_rows),
                corners.reshape(-1, 1, 2),
                True,
            )
            out_path = cfg.overlay_dir / f"{idx}{OVERLAY_SUFFIX}"
            cv2.imwrite(str(out_path), overlay)
        except Exception as e:  # keep calibration going even if overlay save fails
            log.warning(f"Failed to save overlay for {idx}: {e}")

    return R_gb, t_gb, R_tc, t_tc, rmse


def collect_calib_pairs(
    pairs: List[Tuple[str, Path]],
    poses: Dict[str, Dict[str, float]],
    obj_pts: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    cfg: CalibConfig,
) -> Tuple[
    int,
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[float],
]:
    """Process frames and gather inputs for cv2.calibrateHandEye."""
    R_g2b: List[np.ndarray] = []
    t_g2b: List[np.ndarray] = []
    R_t2c: List[np.ndarray] = []
    t_t2c: List[np.ndarray] = []
    rmses: List[float] = []

    used = 0
    abort = False

    def _on_abort() -> None:
        nonlocal abort
        abort = True
        log.info("Abort requested by user input")

    listener = None
    echo = None
    try:
        keymap = {k: _on_abort for k in KEYS_ABORT}
        listener = GlobalKeyListener(keymap)
        listener.start()
        echo = TerminalEchoSuppressor()
        echo.start()
    except Exception as e:
        log.warning(f"Keyboard listener not available: {e}")

    # Progress iterator (utils.logger or fallback to plain for-loop)
    progress_fn = getattr(Logger, "progress", None)
    iterator = progress_fn(pairs, desc="frames") if callable(progress_fn) else pairs

    try:
        for idx, path in iterator:
            if abort:
                break
            res = process_frame(idx, path, poses, obj_pts, K, dist, cfg)
            if res is None:
                continue
            R_gb, t_gb, R_tc, t_tc, rmse = res
            R_g2b.append(R_gb)
            t_g2b.append(t_gb)
            R_t2c.append(R_tc)
            t_t2c.append(t_tc)
            rmses.append(rmse)
            used += 1
    finally:
        try:
            if listener:
                listener.stop()
            if echo:
                echo.stop()
        except Exception:
            pass

    return used, R_g2b, t_g2b, R_t2c, t_t2c, rmses


# =============================================================================
# Pipeline
# =============================================================================


def prepare_data(
    cfg: CalibConfig,
) -> Optional[
    Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Dict[str, Dict[str, float]],
        List[Tuple[str, Path]],
    ]
]:
    """Load intrinsics, poses and image pairs. Optionally auto-detect board."""
    if cfg.verbose:
        Logger.get_logger("root").debug("Verbose mode on")

    if not cfg.dataset_dir.exists():
        log.error(f"Dataset folder not found: {cfg.dataset_dir}")
        return None

    K, dist = load_intrinsics(cfg.intrinsics_path, cfg.stream)
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        log.error("Invalid intrinsics: non-positive focal length")
        return None

    poses_path = cfg.dataset_dir / "poses.json"
    try:
        poses = load_poses(poses_path)
    except Exception as e:
        log.error(f"Failed to load poses: {e}")
        return None
    img_path = cfg.dataset_dir / "imgs"
    pairs = load_dataset_pairs(img_path)
    pairs = [p for p in pairs if p[0] in poses]
    pairs.sort(key=lambda x: int(x[0]) if x[0].isdigit() else x[0])

    if cfg.max_frames:
        pairs = pairs[: cfg.max_frames]

    if not pairs:
        log.error("No frames with both RGB and pose found")
        return None

    # Object points
    obj_pts = make_objpoints(cfg.board_rows, cfg.board_cols, cfg.square_size_m)

    # Optional board autodetection if the given spec fails on first few frames
    # (useful when DEFAULT_BOARD_SPEC is wrong)
    if not test_board_on_sample(pairs, cfg.board_rows, cfg.board_cols):
        candidates = [
            (7, 10),
            (10, 7),
            (6, 9),
            (9, 6),
            (5, 7),
            (7, 5),
            (8, 11),
            (11, 8),
        ]
        detected = try_autodetect_board([p for _, p in pairs], candidates, max_trials=8)
        if detected:
            cfg.board_rows, cfg.board_cols = detected
            obj_pts = make_objpoints(cfg.board_rows, cfg.board_cols, cfg.square_size_m)
            log.info(f"Using autodetected board {cfg.board_rows}x{cfg.board_cols}")
        else:
            log.warning(
                f"Given board {cfg.board_rows}x{cfg.board_cols} not detected in sample frames; proceeding anyway"
            )

    return K, dist, obj_pts, poses, pairs


def test_board_on_sample(
    pairs: List[Tuple[str, Path]], rows: int, cols: int, trials: int = 3
) -> bool:
    """Try a few frames to see if (rows, cols) board is detectable."""
    for _, p in pairs[:trials]:
        img = cv2.imread(str(p))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, _ = detect_chessboard(gray, rows, cols)
        if ok:
            return True
    return False


def ensure_proper_rotation(Rm: np.ndarray, tag: str = "") -> np.ndarray:
    """
    Project R onto SO(3): orthonormalize via SVD and enforce det=+1.
    Returns the closest proper rotation (Frobenius norm).
    """
    det_before = float(np.linalg.det(Rm))
    U, _, Vt = np.linalg.svd(Rm)
    R_hat = U @ Vt
    if np.linalg.det(R_hat) < 0:
        # Flip the smallest singular direction
        U[:, -1] *= -1.0
        R_hat = U @ Vt
    det_after = float(np.linalg.det(R_hat))
    # log.debug(f"{tag} det_before={det_before:.6f} -> det_after={det_after:.6f}")
    return R_hat


def calibrate_and_save(
    cfg: CalibConfig,
    Rg2b: List[np.ndarray],
    tg2b: List[np.ndarray],
    Rt2c: List[np.ndarray],
    tt2c: List[np.ndarray],
    rmses: List[float],
    used: int,
    total: int,
) -> None:
    """Run cv2.calibrateHandEye and save ALL results in one file handeye.json."""
    if used == 0:
        log.error("No valid calibration pairs")
        return

    methods = list(METHOD_MAP.keys()) if cfg.method == "all" else [cfg.method]
    rmse_mean = float(np.mean(rmses)) if rmses else float("nan")

    results = {
        "metadata": {
            "dataset_dir": str(cfg.dataset_dir),
            "intrinsics_path": str(cfg.intrinsics_path),
            "stream": cfg.stream,
            "board": [cfg.board_rows, cfg.board_cols],
            "square_m": cfg.square_size_m,
            "euler_order": cfg.euler_order,
            "pairs_used": used,
            "pairs_total": total,
            "rmse_mean": rmse_mean,
        },
        "methods": {},
    }

    for m in methods:
        flag = METHOD_MAP.get(m)
        if flag is None:
            log.error(f"Unknown hand-eye method: {m}")
            continue

        # Solve hand-eye
        Rcg_raw, tcg = cv2.calibrateHandEye(Rg2b, tg2b, Rt2c, tt2c, method=flag)

        # Sanitize rotation to det=+1
        Rcg = ensure_proper_rotation(Rcg_raw, tag=f"Rcg[{m}]")

        # Builds
        Tcg = np.eye(4)
        Tcg[:3, :3] = Rcg
        Tcg[:3, 3] = tcg.ravel()
        Rgc, tgc = invert_RT(Rcg, tcg)
        Tgc = np.eye(4)
        Tgc[:3, :3] = Rgc
        Tgc[:3, 3] = tgc.ravel()

        r_cg = R.from_matrix(Rcg)
        quat_xyzw = r_cg.as_quat().tolist()
        euler_deg = r_cg.as_euler(cfg.euler_order.lower(), degrees=True).tolist()

        results["methods"][m] = {
            "T_cam_gripper": Tcg.tolist(),
            "T_gripper_cam": Tgc.tolist(),
            "R_cam_gripper_quat_xyzw": quat_xyzw,
            "R_cam_gripper_euler": {
                "order": cfg.euler_order,
                "degrees": euler_deg,
            },
            "translation_cam_gripper_m": tcg.ravel().tolist(),
            "diagnostics": {
                "det_Rcg_raw": float(np.linalg.det(Rcg_raw)),
                "det_Rcg_sanitized": float(np.linalg.det(Rcg)),
            },
            "rmse_mean": rmse_mean,
        }

    out = cfg.dataset_dir / "handeye.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved aggregated results to {out}")
    log.info(f"Processed {used}/{total} frames")
    # log.info(f"rmse_mean={rmse_mean:.4f}")


def run(cfg: CalibConfig) -> int:
    """Main entry: run calibration pipeline."""
    if cfg.verbose:
        Logger.configure(level="DEBUG")

    ErrorTracker.install_excepthook()
    ErrorTracker.install_signal_handlers()

    if cfg.save_overlays:
        try:
            cfg.overlay_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            log.warning(f"Failed to create overlay dir {cfg.overlay_dir}: {e}")

    with CaptureStderrToLogger(log):
        data = prepare_data(cfg)
        if data is None:
            return 1

        K, dist, obj_pts, poses, pairs = data
        used, Rg2b, tg2b, Rt2c, tt2c, rmses = collect_calib_pairs(
            pairs, poses, obj_pts, K, dist, cfg
        )
        if used == 0:
            log.error("No valid calibration pairs after processing")
            return 1

        calibrate_and_save(cfg, Rg2b, tg2b, Rt2c, tt2c, rmses, used, len(pairs))
    return 0


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    """Create argument parser with CLI options."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        type=Path,
        help=f"Path to dataset folder (default: {DEFAULT_DATASET_DIR})",
    )
    p.add_argument(
        "--intrinsics",
        type=Path,
        help=f"Path to camera intrinsics (default: {DEFAULT_INTRINSICS_PATH})",
    )
    p.add_argument(
        "--stream",
        choices=["color", "depth", "ir1", "ir2"],
        help=f"Stream key (default: {DEFAULT_STREAM})",
    )
    p.add_argument(
        "--board", help=f"Chessboard as 'rowsxcols' (default: {DEFAULT_BOARD_SPEC})"
    )
    p.add_argument(
        "--square",
        type=float,
        help=f"Square size in meters (default: {DEFAULT_SQUARE_SIZE_M})",
    )
    p.add_argument(
        "--method",
        help=f"Hand-eye method: {list(METHOD_MAP.keys())} or 'all' (default: {DEFAULT_METHOD})",
    )
    p.add_argument(
        "--save-overlays",
        action="store_true",
        help=f"Save corner overlays (default: {DEFAULT_SAVE_OVERLAYS})",
    )
    p.add_argument(
        "--overlay-dir",
        type=Path,
        help=f"Overlay dir (default: <dataset>/{DEFAULT_OVERLAY_SUBDIR})",
    )
    p.add_argument("--max-frames", type=int, help="Limit number of frames")
    p.add_argument(
        "--euler-order",
        default=DEFAULT_EULER_ORDER,
        help=f"Euler order for Rx,Ry,Rz in poses.json (default: {DEFAULT_EULER_ORDER})",
    )
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for command line execution."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cfg = CalibConfig.from_args(args)
        return run(cfg)
    except Exception as e:
        ErrorTracker.report(e)
        log.error(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
