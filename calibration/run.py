# calibration/run.py
"""
Grid capture (RPC) + Hand-Eye calibration (OpenCV) for an eye-in-hand camera.

- Intrinsics are read from realsense_params.json (not re-saved in other formats).
- Saves RGB (.png), depth (.npy, meters), and TCP poses into calib/<TIMESTAMP>/.
- Images go to calib/<TIMESTAMP>/imgs with indices: 000.png / 000.npy, etc.
- TCP is written as-is from SDK: x,y,z in millimeters; Rx,Ry,Rz in degrees.
- Grid size is auto-derived from a single number of photos.
- Progress bar only for the move+shoot loop (not for grid generation).
- Hand-Eye uses OpenCV (all methods). Report and reprojection errors are saved
  to calib/<TIMESTAMP>/handeye_report.txt and JSON to handeye_results.json.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from utils.error_tracker import ErrorTracker
from utils.logger import Logger
from robot.rpc import RPC

log = Logger.get_logger("run")

# ========================== CONSTANTS (edit) ==========================
ROBOT_IP: str = "192.168.58.2"
TOOL_ID: int = 0
USER_ID: int = 0
VEL_PERCENT: float = 35.0
SETTLE_SEC: float = 0.8

# Base orientation (deg). Use "fixed" or "random" below.
RX_BASE: float = 180.0
RY_BASE: float = 0.0
RZ_BASE: float = 180.0
ORIENT_MODE: str = "random"  # "fixed" | "random" (±15° jitter on 1-3 axes)

# Workspace in meters
WORKSPACE_M: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
    (-0.070, 0.050),  # X
    (-0.250, -0.130),  # Y
    (0.400, 0.450),  # Z
)

# Intrinsics source (RealSense JSON)
REALSENSE_PARAMS: Path = Path("realsense_params.json")
USE_STREAM: str = "color"  # "color" or "depth"

# Chessboard (fixed)
CB_SIZE: Tuple[int, int] = (8, 5)  # (cols, rows) inner corners
CB_SQUARE: float = 0.03  # meters

SAVE_OVERLAY_IMAGES: bool = True  # save chessboard corner overlays
# =====================================================================


@dataclass
class CapturePaths:
    color: Path
    depth: Path  # .npy (meters)


# ------------------------- RealSense intrinsics -----------------------
def load_realsense_intrinsics(
    path: Path, stream: str = "color"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read K and dist from RealSense-style JSON.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    intr = data["intrinsics"][stream]
    fx, fy = float(intr["fx"]), float(intr["fy"])
    cx = float(intr.get("ppx", intr.get("cx")))
    cy = float(intr.get("ppy", intr.get("cy")))
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array(intr.get("coeffs", [0, 0, 0, 0, 0]), dtype=np.float64).ravel()
    log.info(f"Intrinsics loaded from {path} [{stream}]")
    return K, dist


# ------------------------------ RPC utils -----------------------------
def rpc_connect(ip: str, auto_mode: int = 0) -> RPC:
    rpc = RPC(ip=ip)
    try:
        rpc.Mode(auto_mode)
    except Exception as e:
        log.warning(f"Mode set failed (ignored): {e}")
    try:
        rpc.ResetAllError()
    except Exception as e:
        log.warning(f"ResetAllError failed (ignored): {e}")
    err = rpc.RobotEnable(1)
    if err != 0:
        raise RuntimeError(f"RobotEnable failed: {err}")
    log.info("RPC connected and robot enabled.")
    return rpc


def rpc_close(rpc: RPC) -> None:
    try:
        rpc.RobotEnable(0)
    except Exception as e:
        log.warning(f"RobotEnable(0) failed (ignored): {e}")
    try:
        rpc.CloseRPC()
    except Exception as e:
        log.warning(f"CloseRPC failed (ignored): {e}")
    log.info("RPC closed.")


def rpc_move_l(
    rpc: RPC, pose_mm_deg: Iterable[float], tool: int, user: int, vel: float
) -> bool:
    err = rpc.MoveL(
        desc_pos=list(map(float, pose_mm_deg)), tool=tool, user=user, vel=vel
    )
    if err != 0:
        log.error(f"MoveL error={err} pose={pose_mm_deg}")
    return err == 0


def rpc_get_tcp_mm_deg(rpc: RPC) -> List[float] | None:
    """
    Returns [x(mm), y(mm), z(mm), Rx(deg), Ry(deg), Rz(deg)] or None.
    """
    try:
        res = rpc.GetActualTCPPose(1)
        if isinstance(res, tuple) and len(res) == 2 and res[0] == 0:
            vals = [float(x) for x in res[1][:6]]
            log.info(f"TCP SDK: {vals}")
            return vals
    except Exception as e:
        log.warning(f"GetActualTCPPose failed: {e}")
    try:
        pkg = rpc.robot_state_pkg
        vals = [float(x) for x in pkg.tl_cur_pos[:6]]
        log.info(f"TCP (state_pkg): {vals}")
        return vals
    except Exception as e:
        log.error(f"TCP read failed: {e}")
        return None


# ----------------------- Grid generation --------------------
def _rand_rot_multi_axes(
    rng: np.random.Generator, rx: float, ry: float, rz: float, max_abs: float = 10.0
):
    k = int(rng.integers(1, 4))  # {1,2,3}
    axes = [0, 1, 2]
    rng.shuffle(axes)
    chosen = axes[:k]

    def jitter() -> float:
        mag = float(rng.uniform(0.0, max_abs))
        sgn = float(rng.choice([-1.0, 1.0]))
        return sgn * mag

    rx2, ry2, rz2 = rx, ry, rz
    if 0 in chosen:
        rx2 = rx + jitter()
    if 1 in chosen:
        ry2 = ry + jitter()
    if 2 in chosen:
        rz2 = rz + jitter()
    return rx2, ry2, rz2


def _counts_for_total(
    workspace: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    total: int,
) -> Tuple[int, int, int]:
    """
    Choose (nx, ny, nz) - proportional to spans, with nx*ny*nz >= total.
    No infinite loops. Greedy increment on largest fractional deficit.
    """
    assert total >= 1
    (x0, x1), (y0, y1), (z0, z1) = workspace
    spans = np.array([abs(x1 - x0), abs(y1 - y0), abs(z1 - z0)], dtype=float)
    spans = np.maximum(spans, 1e-9)

    # Normalize spans by geometric mean so that prod(weights) ~ 1
    g = float(np.prod(spans)) ** (1.0 / 3.0)
    w = spans / g

    # Target per-axis “base” resolution ~ cube root of total
    base = float(np.cbrt(total))
    nf = base * w  # continuous target counts

    counts = np.maximum(1, np.floor(nf).astype(int))
    prod = int(np.prod(counts))

    # Greedily bump axes with the largest fractional remainder until product >= total
    rema = nf - counts
    while prod < total:
        idx = int(np.argmax(rema))
        counts[idx] += 1
        prod = int(np.prod(counts))
        # update remainder for that axis only
        rema[idx] = (base * w[idx]) - counts[idx]

    return int(counts[0]), int(counts[1]), int(counts[2])


def build_grid_for_count(
    workspace: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    total: int,
    orient_mode: str,
    rx_base: float,
    ry_base: float,
    rz_base: float,
    seed: int = 42,
) -> List[List[float]]:
    (x0, x1), (y0, y1), (z0, z1) = workspace
    nx, ny, nz = _counts_for_total(workspace, total)
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    zs = np.linspace(z0, z1, nz)

    rng = np.random.default_rng(seed)
    poses: List[List[float]] = []
    total_cells = int(nx * ny * nz)

    log.info(
        f"Building grid: nx={nx}, ny={ny}, nz={nz}, cells={total_cells}, requested={total}"
    )

    with tqdm(total=total_cells, desc="Generate grid", leave=False) as pbar:
        for x in xs:
            for y in ys:
                for z in zs:
                    if orient_mode == "random":
                        rx, ry, rz = _rand_rot_multi_axes(
                            rng, rx_base, ry_base, rz_base
                        )
                    else:
                        rx, ry, rz = rx_base, ry_base, rz_base
                    poses.append([float(x), float(y), float(z), rx, ry, rz])
                    pbar.update(1)

    # trim/pad to exact requested count
    if len(poses) > total:
        poses = poses[:total]
    elif len(poses) < total:
        poses += poses[-1:] * (total - len(poses))

    log.info(f"Grid ready: produced={total_cells}, used={len(poses)}")
    return poses


# ------------------------------ I/O utils -----------------------------
def _make_calib_timestamp_dir(base: Path | str = "calib") -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = Path(base) / ts
    (out / "imgs").mkdir(parents=True, exist_ok=True)
    log.info(f"Output dir: {out}")
    return out


# -------------------------- RealSense capture -------------------------
class RealSense:
    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, fps)
        self.cfg.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, fps)
        self.profile = self.pipe.start(self.cfg)
        self.align = rs.align(rs.stream.color)
        self.scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        log.info(f"RealSense started. Depth scale={self.scale}")

    def get_aligned_frames(self):
        frames = self.pipe.wait_for_frames()
        frames = self.align.process(frames)
        return frames

    def stop(self):
        self.pipe.stop()
        log.info("RealSense stopped.")


def save_one_rgbd(rs_cam: RealSense, out_imgs: Path, idx: int) -> CapturePaths:
    """
    Save color as PNG and depth as NPY (meters). Filenames: 000.png / 000.npy.
    """
    frames = rs_cam.get_aligned_frames()
    depth_raw = frames.get_depth_frame()
    color_raw = frames.get_color_frame()

    if not depth_raw or not color_raw:
        raise RuntimeError("No frames from RealSense.")

    color = np.asanyarray(color_raw.get_data()).copy()
    depth = np.asanyarray(depth_raw.get_data()).astype(np.float32)
    depth *= float(rs_cam.scale)

    color_path = out_imgs / f"{idx:03d}.png"
    depth_path = out_imgs / f"{idx:03d}.npy"

    ok = cv2.imwrite(str(color_path), color)
    if not ok:
        raise IOError(f"Failed to write color image: {color_path}")
    np.save(str(depth_path), depth)

    log.info(f"Saved RGBD idx={idx:03d}: {color_path.name}, {depth_path.name}")
    return CapturePaths(color=color_path, depth=depth_path)


# ----------------------------- Capture loop ---------------------------
def capture_dataset_rpc(
    out_dir: Path,
    robot_ip: str,
    tool: int,
    user: int,
    poses_m_deg: List[List[float]],
    vel_percent: float,
    settle_sec: float,
) -> List[int]:
    rpc = rpc_connect(robot_ip, auto_mode=0)
    rs_cam = RealSense(width=1280, height=720, fps=30)

    out_imgs = out_dir / "imgs"
    poses_json: Dict[str, Dict[str, float]] = {}
    indices: List[int] = []

    try:
        with tqdm(total=len(poses_m_deg), desc="Move+Shoot") as pbar:
            for idx, (x, y, z, rx, ry, rz) in enumerate(poses_m_deg):
                pose_mm_deg = [1000.0 * x, 1000.0 * y, 1000.0 * z, rx, ry, rz]
                if not rpc_move_l(
                    rpc, pose_mm_deg, tool=tool, user=user, vel=vel_percent
                ):
                    pbar.update(1)
                    continue

                time.sleep(settle_sec)

                paths = save_one_rgbd(rs_cam, out_imgs, idx)
                indices.append(idx)

                tcp = rpc_get_tcp_mm_deg(rpc)
                if tcp:
                    poses_json[f"{idx:03d}"] = {
                        "x": round(tcp[0], 6),
                        "y": round(tcp[1], 6),
                        "z": round(tcp[2], 6),
                        "Rx": round(tcp[3], 6),
                        "Ry": round(tcp[4], 6),
                        "Rz": round(tcp[5], 6),
                    }

                with open(out_dir / "poses.json", "w", encoding="utf-8") as f:
                    json.dump(poses_json, f, indent=2)

                pbar.update(1)

        with open(out_dir / "poses.json", "w", encoding="utf-8") as f:
            json.dump(poses_json, f, indent=2)
        log.info(f"Saved {len(poses_json)} poses -> {out_dir/'poses.json'}")
        return indices
    finally:
        try:
            rs_cam.stop()
        except Exception as e:
            log.warning(f"RealSense stop failed (ignored): {e}")
        rpc_close(rpc)


# ------------------------- Chessboard + PnP ---------------------------
def _chessboard_objpoints(cols: int, rows: int, square: float) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(cols), np.arange(rows))
    obj = (
        np.stack([xs, ys, np.zeros_like(xs)], axis=-1).reshape(-1, 3).astype(np.float32)
    )
    obj[:, :2] *= float(square)
    return obj


@dataclass
class PnPResult:
    idx: int
    R_tc: np.ndarray
    t_tc: np.ndarray
    rmse_px: float
    n_corners: int


def _detect_chessboard_corners_any(
    img_bgr: np.ndarray,
    prefer_size: Tuple[int, int],
) -> tuple[np.ndarray, Tuple[int, int]] | None:
    """
    Try to detect chessboard corners with either (cols,rows) or (rows,cols).
    Returns (corners[N,1,2] float32, (cols,rows)) or None.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sizes = (prefer_size, (prefer_size[1], prefer_size[0]))

    # flags for classic detector
    classic_flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_EXHAUSTIVE
    )

    for sz in sizes:
        cols, rows = sz
        N = cols * rows

        # SB detector
        try:
            found, corners = cv2.findChessboardCornersSB(gray, (cols, rows), flags=0)
            if found and corners is not None and corners.shape[0] == N:
                return corners.astype(np.float32), (cols, rows)
        except Exception:
            pass

        # Classic + subpix
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), classic_flags)
        if not found or corners is None or corners.shape[0] != N:
            continue
        corners = cv2.cornerSubPix(
            gray,
            corners,
            winSize=(11, 11),
            zeroZone=(-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-4),
        )
        return corners.astype(np.float32), (cols, rows)

    return None


def pnp_for_folder(imgs_dir: Path, K: np.ndarray, dist: np.ndarray) -> List[PnPResult]:
    results: List[PnPResult] = []
    color_paths = sorted(imgs_dir.glob("*.png"))

    for cp in color_paths:
        try:
            idx = int(cp.stem)
        except ValueError:
            log.warning(f"Skip non-index file: {cp.name}")
            continue

        img = cv2.imread(str(cp), cv2.IMREAD_COLOR)
        if img is None:
            log.warning(f"Read fail: {cp}")
            continue

        det = _detect_chessboard_corners_any(img, CB_SIZE)
        if det is None:
            log.warning(f"Chessboard not found: {cp.name}")
            continue
        corners, dims = det  # dims = (cols, rows)

        obj = _chessboard_objpoints(dims[0], dims[1], CB_SQUARE)

        ok, rvec, tvec = cv2.solvePnP(
            obj, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            log.warning(f"solvePnP failed: {cp.name}")
            continue

        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        proj = proj.reshape(-1, 2)
        obs = corners.reshape(-1, 2)
        rmse = float(np.sqrt(np.mean(np.sum((proj - obs) ** 2, axis=1))))

        if SAVE_OVERLAY_IMAGES:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, dims, corners, True)
            cv2.putText(
                vis,
                f"RMSE={rmse:.3f}px",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
            cv2.imwrite(str(cp.with_name(f"{cp.stem}_cb.png")), vis)

        R_tc, _ = cv2.Rodrigues(rvec)
        t_tc = tvec.reshape(3, 1).astype(np.float64)
        results.append(
            PnPResult(
                idx=idx,
                R_tc=R_tc.astype(np.float64),
                t_tc=t_tc,
                rmse_px=rmse,
                n_corners=int(obj.shape[0]),
            )
        )
        log.info(f"[PnP] idx={idx:03d} rmse={rmse:.3f}px  dims={dims[0]}x{dims[1]}")

    results.sort(key=lambda r: r.idx)
    if len(results) < 3:
        log.error("PnP succeeded on <3 frames — hand-eye may be unstable.")
    else:
        log.info(f"PnP successful on {len(results)} frames.")
    return results


def run_handeye_all_methods(
    pnp_results: List[PnPResult], poses_json_path: Path
) -> dict:
    with open(poses_json_path, "r", encoding="utf-8") as f:
        poses_dict = json.load(f)

    idxs = sorted(
        set(int(k) for k in poses_dict.keys()) & set(r.idx for r in pnp_results)
    )
    if not idxs:
        raise RuntimeError("No overlapping indices between poses and PnP results.")

    R_gripper2base, t_gripper2base = [], []
    R_target2cam, t_target2cam = [], []

    for idx in idxs:
        pose = poses_dict[f"{idx:03d}"]
        R_bg = R.from_euler(
            "xyz", [pose["Rx"], pose["Ry"], pose["Rz"]], degrees=True
        ).as_matrix()
        t_bg = np.array(
            [[pose["x"] / 1000.0], [pose["y"] / 1000.0], [pose["z"] / 1000.0]],
            dtype=np.float64,
        )  # meters
        R_gripper2base.append(R_bg.astype(np.float64))
        t_gripper2base.append(t_bg.astype(np.float64))

        pr = next(r for r in pnp_results if r.idx == idx)
        R_target2cam.append(pr.R_tc)
        t_target2cam.append(pr.t_tc)

    methods = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }

    results = {"indices_used": [int(i) for i in idxs], "methods": {}}
    for name, mcode in methods.items():
        R_c2g, t_c2g = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, method=mcode
        )
        euler = R.from_matrix(R_c2g).as_euler("xyz", degrees=True).tolist()
        t_list = np.asarray(t_c2g).reshape(3).tolist()
        results["methods"][name] = {
            "R_cam2gripper": R_c2g.tolist(),
            "t_cam2gripper_m": t_list,
            "euler_xyz_deg": euler,
        }
        log.info(f"[{name}] t(m)={t_list}  euler(deg)={euler}")
    return results


def write_report_txt(
    out_dir: Path,
    pnp_results: List[PnPResult],
    handeye_results: dict,
) -> None:
    pnp_rmses = [r.rmse_px for r in pnp_results]
    mean_rmse = float(np.mean(pnp_rmses)) if pnp_rmses else float("nan")
    med_rmse = float(np.median(pnp_rmses)) if pnp_rmses else float("nan")
    min_rmse = float(np.min(pnp_rmses)) if pnp_rmses else float("nan")
    max_rmse = float(np.max(pnp_rmses)) if pnp_rmses else float("nan")

    lines = []
    lines.append("=== PnP (chessboard) reprojection error (pixels) ===")
    lines.append(f"frames_ok={len(pnp_results)}")
    lines.append(
        f"rmse_mean={mean_rmse:.4f}  rmse_median={med_rmse:.4f}  min={min_rmse:.4f}  max={max_rmse:.4f}"
    )
    lines.append("")

    lines.append("=== Hand-Eye results (cam->gripper) ===")
    for name, res in handeye_results["methods"].items():
        t = res["t_cam2gripper_m"]
        e = res["euler_xyz_deg"]
        lines.append(f"[{name}] t(m) = [{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}]")
        lines.append(f"[{name}] euler_xyz_deg = [{e[0]:.3f}, {e[1]:.3f}, {e[2]:.3f}]")
        lines.append("")
    lines.append(f"indices_used = {handeye_results.get('indices_used', [])}")

    report_path = out_dir / "handeye_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info(f"Report written: {report_path}")


def run(num_photos: int = 35) -> Path:
    """
    End-to-end capture + Hand-Eye. Returns the output directory.
    """
    ErrorTracker.install_excepthook()
    ErrorTracker.install_signal_handlers()

    # Prepare output
    out_dir = _make_calib_timestamp_dir("calib")

    # Copy original RealSense params into the dataset folder
    try:
        shutil.copy2(REALSENSE_PARAMS, out_dir / REALSENSE_PARAMS.name)
        log.info(f"Copied {REALSENSE_PARAMS} -> {out_dir / REALSENSE_PARAMS.name}")
    except Exception as e:
        log.warning(f"Failed to copy realsense params: {e}")

    # Load intrinsics (for PnP)
    K, dist = load_realsense_intrinsics(REALSENSE_PARAMS, stream=USE_STREAM)

    # Build grid
    grid = build_grid_for_count(
        WORKSPACE_M,
        total=int(num_photos),
        orient_mode=ORIENT_MODE,
        rx_base=RX_BASE,
        ry_base=RY_BASE,
        rz_base=RZ_BASE,
    )

    # Capture
    indices = capture_dataset_rpc(
        out_dir=out_dir,
        robot_ip=ROBOT_IP,
        tool=TOOL_ID,
        user=USER_ID,
        poses_m_deg=grid,
        vel_percent=VEL_PERCENT,
        settle_sec=SETTLE_SEC,
    )

    # PnP on captured set
    pnp_results = pnp_for_folder(out_dir / "imgs", K, dist)

    # Hand-Eye
    he_results = run_handeye_all_methods(
        pnp_results=pnp_results,
        poses_json_path=out_dir / "poses.json",
    )

    # Save JSON + human-readable report + reprojection errs
    with open(out_dir / "handeye_results.json", "w", encoding="utf-8") as f:
        json.dump(he_results, f, indent=2)
    write_report_txt(out_dir, pnp_results, he_results)

    log.info(f"Done. Output: {out_dir}")
    return out_dir


if __name__ == "__main__":
    run(num_photos=21)
