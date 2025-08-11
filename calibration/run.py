# calibration/run.py
"""Command line entry point for calibration & grid capture with RPC.

Features:
- Saves RGB + depth and TCP poses in calib/<TIMESTAMP>/ in real-time.
- Optional randomized rotation per grid point: up to ±15° on 1/2/3 axes
  (enable with --orient-mode random).
- Loads intrinsics from RealSense JSON (e.g., realsense_params.json) or OpenCV YAML/JSON.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import yaml

# --- your utils ---
from utils.error_tracker import ErrorTracker
from utils.logger import Logger

from .april import AprilTagPattern
from .base import Calibrator
from .charuco import CharucoPattern
from .chessboard import ChessboardPattern
from .utils import (
    confirm,
    parse_board_size,
    save_intrinsics,
    ImagePair,
    save_image_pair,
)
from robot_scan.capture import capture_rgbd  # uses RealSense

# --- robot SDK (direct) ---
from robot.rpc import RPC  # your RPC wrapper

log = Logger.get_logger("calibration.run")

# Default workspace (IN METERS) converted from your mm limits
DEFAULT_WORKSPACE_M = (
    (-0.070, 0.050),  # X: -70..50 mm
    (-0.250, -0.130),  # Y: -250..-130 mm
    (0.300, 0.400),  # Z: 300..400 mm
)


# ---------------------------------------------------------------------
# Intrinsics loader that supports both RealSense JSON and OpenCV YAML/JSON
# ---------------------------------------------------------------------
def load_intrinsics_any(
    path: Path, stream: str = "color"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read camera intrinsics/distortion from:
      1) OpenCV YAML/JSON:
         {
           "camera_matrix": {"data":[...]},
           "distortion_coefficients":{"data":[...]}
         }
         OR OpenCV-style JSON:
         {"camera_matrix":[...], "distortion_coefficients":[...]}
      2) RealSense JSON (like your realsense_params.json):
         {
           "depth_scale": ...,
           "intrinsics": {
              "color": {"fx","fy","ppx","ppy","coeffs":[...], "width","height","model"},
              "depth": {...}, "ir1": {...}, "ir2": {...}
           },
           "extrinsics": {...}
         }
    Returns: (K: 3x3, dist: (N,))
    """
    path = Path(path)

    # OpenCV YAML
    if path.suffix.lower() in {".yml", ".yaml"}:
        with open(path, "r", encoding="utf-8") as f:
            yml = yaml.safe_load(f)
        K = np.array(yml["camera_matrix"]["data"], dtype=float).reshape(3, 3)
        dist = np.array(yml["distortion_coefficients"]["data"], dtype=float).ravel()
        log.info(f"Loaded intrinsics (OpenCV YAML) from {path}")
        return K, dist

    # JSON variants
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # OpenCV-style JSON
    if "camera_matrix" in data and "distortion_coefficients" in data:
        K = np.array(data["camera_matrix"], dtype=float).reshape(3, 3)
        dist = np.array(data["distortion_coefficients"], dtype=float).ravel()
        log.info(f"Loaded intrinsics (OpenCV JSON) from {path}")
        return K, dist

    # RealSense-style JSON
    if "intrinsics" in data:
        intr_all = data["intrinsics"]
        if stream not in intr_all:
            raise KeyError(
                f"Stream '{stream}' not found in JSON 'intrinsics' (has: {list(intr_all.keys())})"
            )
        intr = intr_all[stream]
        fx = float(intr["fx"])
        fy = float(intr["fy"])
        cx = float(intr.get("ppx", intr.get("cx")))
        cy = float(intr.get("ppy", intr.get("cy")))
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)
        coeffs = intr.get("coeffs")
        dist = np.array(
            coeffs if coeffs is not None else [0, 0, 0, 0, 0], dtype=float
        ).ravel()
        log.info(f"Loaded intrinsics (RealSense JSON, stream '{stream}') from {path}")
        return K, dist

    raise ValueError(f"Unsupported intrinsics format in {path}")


# ------------------------- robot helpers (RPC) ------------------------- #
def rpc_connect(ip: str, auto_mode: int = 0) -> RPC:
    """Connect to robot controller and enable it."""
    rpc = RPC(ip=ip)
    # auto/manual mode (0/1), clear errors, enable
    try:
        rpc.Mode(auto_mode)
    except Exception:
        pass
    try:
        rpc.ResetAllError()
    except Exception:
        pass
    err = rpc.RobotEnable(1)
    if err != 0:
        raise RuntimeError(f"RobotEnable failed: {err}")
    return rpc


def rpc_close(rpc: RPC) -> None:
    try:
        rpc.RobotEnable(0)
    except Exception:
        pass
    try:
        rpc.CloseRPC()
    except Exception:
        pass


def rpc_move_l(
    rpc: RPC, pose_mm_deg: Iterable[float], tool: int, user: int, vel: float
) -> bool:
    """Move linearly in Cartesian pose [x,y,z,rx,ry,rz]; x,y,z in mm, angles in deg."""
    desc = list(map(float, pose_mm_deg))
    err = rpc.MoveL(desc_pos=desc, tool=tool, user=user, vel=vel)
    return err == 0


def rpc_get_tcp_mm_deg(rpc: RPC) -> List[float] | None:
    """Read current TCP pose [mm,deg]. Tries XML-RPC, falls back to realtime packet."""
    try:
        res = rpc.GetActualTCPPose(1)  # non-blocking
        if isinstance(res, tuple) and len(res) == 2 and res[0] == 0:
            return list(map(float, res[1]))
    except Exception:
        pass
    try:
        pkg = rpc.robot_state_pkg
        return [
            float(pkg.tl_cur_pos[0]),
            float(pkg.tl_cur_pos[1]),
            float(pkg.tl_cur_pos[2]),
            float(pkg.tl_cur_pos[3]),
            float(pkg.tl_cur_pos[4]),
            float(pkg.tl_cur_pos[5]),
        ]
    except Exception:
        return None


# ------------------------- grid generation ------------------------- #
def _rand_rot_multi_axes(
    rng: np.random.Generator,
    rx_base: float,
    ry_base: float,
    rz_base: float,
    max_abs_deg: float = 15.0,
) -> Tuple[float, float, float]:
    """Randomize 1, 2, or 3 axes (uniformly), each jitter in [-max_abs_deg, +max_abs_deg]."""
    k = int(rng.integers(1, 4))  # choose {1,2,3} axes
    axes = [0, 1, 2]
    rng.shuffle(axes)
    chosen = set(axes[:k])

    def jit() -> float:
        return float(rng.uniform(-max_abs_deg, max_abs_deg))

    rx = rx_base + (jit() if 0 in chosen else 0.0)
    ry = ry_base + (jit() if 1 in chosen else 0.0)
    rz = rz_base + (jit() if 2 in chosen else 0.0)

    # sanity check; if numeric oddities, fallback to base
    Rm = R.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()
    if (np.linalg.det(Rm) < 0.99) or (np.linalg.norm(Rm.T @ Rm - np.eye(3)) > 1e-5):
        rx, ry, rz = rx_base, ry_base, rz_base
    return float(rx), float(ry), float(rz)


def build_grid(
    workspace_m: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    step_m: float,
    orient_mode: str,
    rx_base: float,
    ry_base: float,
    rz_base: float,
    seed: int = 42,
) -> List[List[float]]:
    """Return list of poses [x,y,z,rx,ry,rz] in meters/deg."""
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = workspace_m
    xs = np.arange(x_min, x_max + 1e-9, step_m)
    ys = np.arange(y_min, y_max + 1e-9, step_m)
    zs = np.arange(z_min, z_max + 1e-9, max(step_m, 1e-6))
    rng = np.random.default_rng(seed)

    poses: List[List[float]] = []
    for x in xs:
        for y in ys:
            for z in zs:
                if orient_mode == "random":
                    rx, ry, rz = _rand_rot_multi_axes(
                        rng, rx_base, ry_base, rz_base, max_abs_deg=15.0
                    )
                else:
                    rx, ry, rz = rx_base, ry_base, rz_base
                poses.append([float(x), float(y), float(z), rx, ry, rz])
    return poses


def parse_workspace_arg(
    workspace: str,
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """Parse 'xmin:xmax,ymin:ymax,zmin:zmax' in meters; allow zmin==zmax."""
    try:
        parts = workspace.split(",")
        xr = tuple(map(float, parts[0].split(":")))
        yr = tuple(map(float, parts[1].split(":")))
        zr = tuple(map(float, parts[2].split(":")))
        assert len(xr) == len(yr) == len(zr) == 2
        return (xr[0], xr[1]), (yr[0], yr[1]), (zr[0], zr[1])
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Bad --workspace format: {workspace}") from e


def _make_calib_timestamp_dir(base: Path | str = "calib") -> Path:
    """Create calib/<TIMESTAMP>/ and return the path."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = Path(base) / ts
    out.mkdir(parents=True, exist_ok=True)
    log.info(f"Output dir: {out}")
    return out


# ------------------------- capture loop ------------------------- #
def capture_dataset_rpc(
    out_dir: Path,
    *,
    robot_ip: str,
    tool: int,
    user: int,
    workspace_m: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    grid_step_m: float,
    rx_base: float,
    ry_base: float,
    rz_base: float,
    orient_mode: str,
    settle_sec: float,
    max_frames: int | None,
    vel_percent: float,
) -> List[ImagePair]:
    """Move robot over grid, capture RGB-D, save images and poses.json in out_dir."""

    rpc = rpc_connect(robot_ip, auto_mode=0)
    try:
        grid = build_grid(
            workspace_m,
            grid_step_m,
            orient_mode,
            rx_base=rx_base,
            ry_base=ry_base,
            rz_base=rz_base,
        )
        if max_frames:
            grid = grid[:max_frames]

        out_dir.mkdir(parents=True, exist_ok=True)
        poses_json: dict[str, dict[str, float]] = {}
        pairs: List[ImagePair] = []

        with tqdm(total=len(grid), desc="Capture") as pbar:
            for idx, pose_m_deg in enumerate(grid):
                x_m, y_m, z_m, rx, ry, rz = pose_m_deg
                # RPC expects mm + deg
                pose_mm_deg = [1000.0 * x_m, 1000.0 * y_m, 1000.0 * z_m, rx, ry, rz]
                ok = rpc_move_l(rpc, pose_mm_deg, tool=tool, user=user, vel=vel_percent)
                if not ok:
                    log.error(f"MoveL failed at idx={idx} to {pose_mm_deg}")
                    pbar.update(1)
                    continue

                time.sleep(settle_sec)

                # Capture RGB-D and save immediately (real-time)
                frame = capture_rgbd()
                pair = save_image_pair(frame.color, frame.depth, out_dir, idx)
                pairs.append(pair)

                # Read TCP after move (mm,deg) -> save in meters + deg, rounded
                tcp = rpc_get_tcp_mm_deg(rpc)
                if tcp:
                    poses_json[f"{idx:03d}"] = {
                        "x": round(tcp[0] / 1000.0, 6),
                        "y": round(tcp[1] / 1000.0, 6),
                        "z": round(tcp[2] / 1000.0, 6),
                        "Rx": round(tcp[3], 6),
                        "Ry": round(tcp[4], 6),
                        "Rz": round(tcp[5], 6),
                    }

                # Flush poses to disk frequently (real-time JSON update)
                with open(out_dir / "poses.json", "w", encoding="utf-8") as f:
                    json.dump(poses_json, f, indent=2)

                pbar.update(1)

        # Final write (ensures consistency)
        with open(out_dir / "poses.json", "w", encoding="utf-8") as f:
            json.dump(poses_json, f, indent=2)
        log.info(f"Saved {len(poses_json)} poses to {out_dir/'poses.json'}")

        return pairs
    finally:
        rpc_close(rpc)


# ------------------------- CLI ------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Camera calibration tool (with RPC grid capture)"
    )
    p.add_argument(
        "--pattern", choices=["charuco", "chessboard", "april"], required=True
    )
    p.add_argument("--board-size", required=True, help="Board size WxH, e.g. 8x5")
    p.add_argument(
        "--square-length", type=float, required=True, help="Square side length [m]"
    )
    p.add_argument("--aruco-dict", default="DICT_5X5_100")
    p.add_argument("--image-dir", type=Path, help="Offline image directory")

    # Intrinsics
    p.add_argument(
        "--intrinsics",
        type=Path,
        default=Path("realsense_params.json"),
        help="Path to intrinsics: RealSense JSON (e.g., realsense_params.json) or OpenCV YAML/JSON",
    )
    p.add_argument(
        "--intrinsics-stream",
        choices=["color", "depth", "ir1", "ir2"],
        default="color",
        help="Which stream's intrinsics to use when loading RealSense-style JSON",
    )

    # Capture options
    p.add_argument(
        "--capture-poses", action="store_true", help="Capture images and robot poses"
    )
    p.add_argument("--robot-ip", default="192.168.58.2", help="Robot controller IP")
    p.add_argument("--tool", type=int, default=0)
    p.add_argument("--user", type=int, default=0)

    # Workspace: optional; defaults to fixed mm limits converted to meters
    p.add_argument(
        "--workspace",
        help=(
            'Optional workspace in meters "xmin:xmax,ymin:ymax,zmin:zmax". '
            "If omitted, defaults to X[-70,50]mm, Y[-250,-130]mm, Z[300,400]mm."
        ),
        default=None,
    )

    p.add_argument("--grid-step", type=float, default=0.025, help="Grid step [m]")
    p.add_argument("--rx-base", type=float, default=180.0)
    p.add_argument("--ry-base", type=float, default=0.0)
    p.add_argument("--rz-base", type=float, default=180.0)
    p.add_argument(
        "--orient-mode",
        choices=["fixed", "random"],
        default="fixed",
        help='Use "random" to add up to ±15° on 1/2/3 axes per point',
    )
    p.add_argument("--settle-sec", type=float, default=0.8)
    p.add_argument(
        "--vel",
        type=float,
        default=35,
        help="Move speed percent [0..100] for RPC.MoveL",
    )
    p.add_argument(
        "--max-frames", type=int, default=100, help="Maximum frames to capture"
    )
    p.add_argument(
        "--save-images",
        action="store_true",
        help="Save overlay images during calibration",
    )
    p.add_argument(
        "--interactive", action="store_true", help="Confirm steps via keyboard"
    )
    return p


def main(argv: list[str] | None = None) -> None:
    ErrorTracker.install_excepthook()
    ErrorTracker.install_signal_handlers()

    parser = build_parser()
    args = parser.parse_args(argv)

    # Load intrinsics (supports RealSense JSON & OpenCV formats)
    K, dist = load_intrinsics_any(args.intrinsics, stream=args.intrinsics_stream)

    # Build pattern
    board_size = parse_board_size(args.board_size)
    if args.pattern == "chessboard":
        pattern = ChessboardPattern(board_size, args.square_length)
    elif args.pattern == "charuco":
        pattern = CharucoPattern(board_size, args.square_length, args.aruco_dict)
    else:
        pattern = AprilTagPattern(board_size, args.square_length, args.aruco_dict)

    calibrator = Calibrator(pattern, K, dist, save_images=args.save_images)

    # Decide capture or offline
    if args.capture_poses:
        if args.interactive and not confirm("Start pose/image capture?"):
            log.info("Aborted before capture")
            return

        # Force output into calib/<TIMESTAMP>/
        out_dir = _make_calib_timestamp_dir("calib")
        # Save intrinsics we actually used (for traceability)
        save_intrinsics(K, dist, out_dir / "intrinsics.json")

        # Workspace choice
        if args.workspace:
            workspace_m = parse_workspace_arg(args.workspace)
        else:
            workspace_m = DEFAULT_WORKSPACE_M
            log.info(
                "Using default workspace: X[-70,50]mm, Y[-250,-130]mm, Z[300,400]mm"
            )

        # Capture
        capture_dataset_rpc(
            out_dir=out_dir,
            robot_ip=args.robot_ip,
            tool=args.tool,
            user=args.user,
            workspace_m=workspace_m,
            grid_step_m=args.grid_step,
            rx_base=args.rx_base,
            ry_base=args.ry_base,
            rz_base=args.rz_base,
            orient_mode=args.orient_mode,
            settle_sec=args.settle_sec,
            max_frames=args.max_frames,
            vel_percent=args.vel,
        )
        image_dir = out_dir

        if args.interactive and not confirm("Continue to calibration?"):
            log.info("Aborted before calibration")
            return

    elif args.image_dir:
        image_dir = args.image_dir
    else:
        parser.error("--image-dir required when not capturing poses")
        return

    # Run calibration (estimates camera-to-pattern poses from captured RGB)
    calibrator.run(image_dir)


if __name__ == "__main__":
    main()
