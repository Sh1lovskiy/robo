#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
capture_circle_target_movej.py — Fixed circle around a given center; each shot looks at LOOK_AT_XYZ.
- Center position is fixed by CENTER_POS_XYZ (orientation is recomputed to "look at" point).
- Then N points on a circle (radius = CIRCLE_RADIUS_MM) in the plane ⟂ (center→LOOK_AT_XYZ).
- Each pose is executed via MoveJ after solving IK; failures are skipped (logged) without aborting the run.
- RGB saved as PNG; Depth saved as NPY in meters (float32).
- TCP poses recorded to poses.json (x,y,z in mm; rx,ry,rz in deg from SDK).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

from utils.logger import Logger
from utils.error_tracker import ErrorTracker
from robot.rpc import RPC

# ========================== CONFIG ==========================
ROBOT_IP = "192.168.58.2"
TOOL_ID = 0
WOBJ_ID = 0

SPEED_PCT = 20
VEL_PCT = 25.0
ACC_PCT = 0.0
OVL_PCT = 100.0
SETTLE_SEC = 0.35

# Look-at target (only XYZ used)
LOOK_AT_XYZ = np.array([-318.559, -209.009, 405.3], dtype=float)

# Circle center (absolute position). Orientation here is ignored; recomputed as "look-at".
CENTER_POS_XYZ = np.array([8.331, -106.449, -106.449], dtype=float)
CAPTURE_CENTER = True

# Circle parameters
CIRCLE_COUNT = 8  # number of photos on the circle
CIRCLE_RADIUS_MM = 100.0  # 10 cm
CIRCLE_START_DEG = 0.0  # 0° = "up" direction in the view plane

# RealSense
RS_WIDTH = 1280
RS_HEIGHT = 720
RS_FPS = 30

# Output (None -> captures/{timestamp})
OUT_DIR = None
# ===========================================================

WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=float)
log = Logger.get_logger("add_photo")


# ---------------------- RealSense ----------------------
class RealSense:
    """RGB/Depth capture. RGB -> PNG (uint8 BGR). Depth -> NPY in meters (float32)."""

    def __init__(self, width=1280, height=720, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.align = None
        self.profile = None
        self.rs = None

    def start(self):
        import pyrealsense2 as rs

        self.rs = rs
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
        )
        cfg.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
        )
        self.align = rs.align(rs.stream.color)
        self.profile = self.pipeline.start(cfg)
        # warmup
        for _ in range(10):
            self.pipeline.wait_for_frames()
        log.info("RealSense pipeline started")

    def stop(self):
        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None
            log.info("RealSense pipeline stopped")

    def capture_pair(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (color_bgr uint8 HxWx3, depth_m float32 HxW)."""
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        color_raw = frames.get_color_frame()
        depth_raw = frames.get_depth_frame()
        if not color_raw or not depth_raw:
            raise RuntimeError("RealSense: incomplete frames")

        color = np.asanyarray(color_raw.get_data())
        depth = np.asanyarray(depth_raw.get_data()).astype(np.float32)  # z16 -> float32
        # scale to meters
        active_profile = self.pipeline.get_active_profile()
        scale = active_profile.get_device().first_depth_sensor().get_depth_scale()
        depth *= float(scale)
        return color, depth


# ---------------------- Math helpers ----------------------
def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def euler_xyz_from_look_at(
    src_xyz: np.ndarray, tgt_xyz: np.ndarray, world_up=WORLD_UP
) -> Tuple[float, float, float]:
    """Return Euler XYZ (deg) so tool -Z looks at tgt."""
    view = normalize(tgt_xyz - src_xyz)
    z_axis = -view
    x_axis = normalize(np.cross(world_up, z_axis))
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = normalize(np.cross(np.array([1.0, 0.0, 0.0]), z_axis))
    y_axis = normalize(np.cross(z_axis, x_axis))
    Rm = np.column_stack([x_axis, y_axis, z_axis])
    rx, ry, rz = R.from_matrix(Rm).as_euler("xyz", degrees=True)
    return float(rx), float(ry), float(rz)


def fmt_pose(p: List[float]) -> str:
    return f"[{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}, {p[3]:.3f}, {p[4]:.3f}, {p[5]:.3f}]"


def ask_yes_no(prompt: str) -> bool:
    ans = input(f"{prompt} [y/N]: ").strip().lower()
    return ans in ("y", "yes")


def circle_points(
    center_xyz: np.ndarray,
    target_xyz: np.ndarray,
    radius_mm: float,
    count: int,
    start_deg: float,
) -> List[np.ndarray]:
    """Points on a circle in the plane ⟂ (center→target). 0° = 'up'."""
    view = normalize(target_xyz - center_xyz)
    right = normalize(np.cross(view, WORLD_UP))
    if np.linalg.norm(right) < 1e-6:
        right = normalize(np.cross(view, np.array([1.0, 0.0, 0.0])))
    up = normalize(np.cross(right, view))

    pts = []
    theta0 = np.deg2rad(start_deg)
    for k in range(count):
        theta = theta0 + 2.0 * np.pi * (k / float(count))
        p_xyz = center_xyz + radius_mm * (np.sin(theta) * right + np.cos(theta) * up)
        pts.append(p_xyz.astype(float))
    return pts


# ---------------------- MoveJ helpers (robust/skip-on-failure) ----------------------
def try_ik_solve_joints(rpc: RPC, desc_pose: List[float]) -> Optional[List[float]]:
    """
    Robust IK call. Returns 6 joint angles (deg) or None if unsolvable/unexpected.
    Handles SDK returning either:
      - (err, [j1..j6])
      - (err, j1, j2, j3, j4, j5, j6)
      - err_only (int)  -> treat as failure if err != 0
    """
    try:
        res = rpc.GetInverseKin(0, desc_pose, -1)
    except Exception as e:
        log.error(f"GetInverseKin RPC exception for pose {fmt_pose(desc_pose)}: {e}")
        return None

    # Normalize result
    if isinstance(res, (list, tuple)):
        if not res:
            log.warning(
                f"GetInverseKin empty return for pose {fmt_pose(desc_pose)}; skip"
            )
            return None
        err = int(res[0])
        if err != 0:
            log.warning(
                f"GetInverseKin failed (err={err}) for pose {fmt_pose(desc_pose)}; skip"
            )
            return None
        # success path
        if len(res) >= 2 and isinstance(res[1], (list, tuple)):
            joints = list(map(float, res[1]))
        elif len(res) >= 7:
            joints = list(map(float, res[1:7]))
        else:
            log.warning(f"GetInverseKin unexpected success form: {res}; skip")
            return None
    elif isinstance(res, (int, np.integer, float)):
        err = int(res)
        if err != 0:
            log.warning(
                f"GetInverseKin failed (err={err}) for pose {fmt_pose(desc_pose)}; skip"
            )
            return None
        log.warning(
            f"GetInverseKin returned 0 but no joints for pose {fmt_pose(desc_pose)}; skip"
        )
        return None
    else:
        log.warning(f"GetInverseKin unknown return type {type(res)}: {res}; skip")
        return None

    if len(joints) != 6:
        log.warning(f"IK returned {len(joints)} joints (expected 6): {joints}; skip")
        return None
    return joints


def try_movej_to_pose(rpc: RPC, pose6: List[float]) -> bool:
    """
    Solve IK and MoveJ to 'pose6'. Returns True on success; otherwise logs and returns False.
    """
    joints = try_ik_solve_joints(rpc, pose6)
    if joints is None:
        return False

    try:
        err = rpc.MoveJ(
            joint_pos=joints,
            tool=TOOL_ID,
            user=WOBJ_ID,
            desc_pos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vel=VEL_PCT,
            acc=ACC_PCT,
            ovl=OVL_PCT,
            exaxis_pos=[0.0, 0.0, 0.0, 0.0],
            blendT=-1.0,
            offset_flag=0,
            offset_pos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
    except Exception as e:
        log.error(f"MoveJ RPC exception to pose {fmt_pose(pose6)}: {e}")
        return False

    if isinstance(err, (list, tuple)):
        # some SDKs might return composite; try first element as code
        code = int(err[0]) if err else -999
    else:
        code = int(err)

    if code != 0:
        log.warning(f"MoveJ failed (err={code}) to pose {fmt_pose(pose6)}; skip")
        return False
    return True


# ---------------------- Capture session ----------------------
@dataclass
class PoseRec:
    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float


def run() -> None:
    ErrorTracker.install_excepthook()
    ErrorTracker.install_signal_handlers()

    ts = time.strftime("%Y%m%d_%H%M%S")
    outdir = Path(OUT_DIR) if OUT_DIR else Path("captures") / ts
    imgdir = outdir / "imgs"
    outdir.mkdir(parents=True, exist_ok=True)
    imgdir.mkdir(parents=True, exist_ok=True)
    poses_path = outdir / "poses.json"
    poses: Dict[str, Dict[str, float]] = {}

    # --- Connect & prepare robot
    log.info(f"Connecting to robot at {ROBOT_IP} ...")
    rpc = RPC(ip=ROBOT_IP)
    log.info("Connected. Configuring robot ...")
    rpc.SetSpeed(SPEED_PCT)
    rpc.RobotEnable(1)

    # --- Start RealSense
    cam = RealSense(RS_WIDTH, RS_HEIGHT, RS_FPS)
    cam.start()

    try:
        # Info: current TCP (for logs only)
        err, tcp = rpc.GetActualTCPPose(1)
        if err == 0 and isinstance(tcp, (list, tuple)) and len(tcp) == 6:
            log.info(f"Current TCP (SDK): {fmt_pose([float(v) for v in tcp])}")

        # -------- Go to CENTER (position fixed; orientation = look-at) --------
        rx, ry, rz = euler_xyz_from_look_at(CENTER_POS_XYZ, LOOK_AT_XYZ)
        center_pose_cmd = [
            float(CENTER_POS_XYZ[0]),
            float(CENTER_POS_XYZ[1]),
            float(CENTER_POS_XYZ[2]),
            rx,
            ry,
            rz,
        ]
        log.info(f"Center pose (look-at): {fmt_pose(center_pose_cmd)}")
        print(f"NEXT MOVE (CENTER): {fmt_pose(center_pose_cmd)}")
        if not ask_yes_no("Proceed with CENTER move?"):
            log.warning("Aborted by user at CENTER move.")
            return

        if not try_movej_to_pose(rpc, center_pose_cmd):
            log.warning(
                "CENTER move skipped due to IK/MoveJ failure; continuing with circle planning."
            )
        else:
            time.sleep(SETTLE_SEC)
            # Ground-truth arrived center (for logging; circle is planned from fixed CENTER_POS_XYZ anyway)
            err, atcp = rpc.GetActualTCPPose(1)
            if err == 0 and isinstance(atcp, (list, tuple)) and len(atcp) == 6:
                log.info(f"Arrived (CENTER) TCP: {fmt_pose([float(v) for v in atcp])}")
            if CAPTURE_CENTER:
                idx = f"{0:03d}"
                color, depth_m = cam.capture_pair()
                cv2.imwrite(str(imgdir / f"{idx}_rgb.png"), color)
                np.save(str(imgdir / f"{idx}_depth.npy"), depth_m.astype(np.float32))
                # Save actual pose if available
                if err == 0 and isinstance(atcp, (list, tuple)) and len(atcp) == 6:
                    pose6 = [float(v) for v in atcp]
                else:
                    pose6 = center_pose_cmd  # fallback to commanded
                poses[idx] = dict(
                    x=pose6[0],
                    y=pose6[1],
                    z=pose6[2],
                    rx=pose6[3],
                    ry=pose6[4],
                    rz=pose6[5],
                )
                with open(poses_path, "w", encoding="utf-8") as f:
                    json.dump(poses, f, indent=2, ensure_ascii=False)

        next_index = 1 if CAPTURE_CENTER else 0

        # -------- Circle around center (always looking at LOOK_AT_XYZ) --------
        circle_xyz = circle_points(
            CENTER_POS_XYZ,
            LOOK_AT_XYZ,
            CIRCLE_RADIUS_MM,
            CIRCLE_COUNT,
            CIRCLE_START_DEG,
        )

        for k, p_xyz in enumerate(circle_xyz):
            rx, ry, rz = euler_xyz_from_look_at(p_xyz, LOOK_AT_XYZ)
            pose6 = [float(p_xyz[0]), float(p_xyz[1]), float(p_xyz[2]), rx, ry, rz]
            idx = f"{next_index:03d}"
            log.info(
                f"CIRCLE[{k+1}/{CIRCLE_COUNT}] pose: {fmt_pose(pose6)}  (R={CIRCLE_RADIUS_MM} mm)"
            )
            print(f"NEXT MOVE (CIRCLE {k+1}/{CIRCLE_COUNT}): {fmt_pose(pose6)}")
            if not ask_yes_no("Proceed with this move?"):
                log.warning(f"Skipped by user: circle index {k}")
                next_index += 1
                continue

            if not try_movej_to_pose(rpc, pose6):
                log.warning(
                    f"Skipping capture at circle index {k} due to IK/MoveJ failure."
                )
                next_index += 1
                continue

            time.sleep(SETTLE_SEC)

            color, depth_m = cam.capture_pair()
            cv2.imwrite(str(imgdir / f"{idx}_rgb.png"), color)
            np.save(str(imgdir / f"{idx}_depth.npy"), depth_m.astype(np.float32))

            err, atcp = rpc.GetActualTCPPose(1)
            if err == 0 and isinstance(atcp, (list, tuple)) and len(atcp) == 6:
                pose6 = [float(v) for v in atcp]  # use ground-truth from SDK

            poses[idx] = dict(
                x=pose6[0],
                y=pose6[1],
                z=pose6[2],
                rx=pose6[3],
                ry=pose6[4],
                rz=pose6[5],
            )

            with open(poses_path, "w", encoding="utf-8") as f:
                json.dump(poses, f, indent=2, ensure_ascii=False)

            next_index += 1

        log.info(f"Saved images to: {imgdir}")
        log.info(f"Saved poses JSON: {poses_path}")

    finally:
        try:
            cam.stop()
        except Exception:
            pass
        try:
            rpc.CloseRPC()
        except Exception:
            pass
        log.info("Done.")


# ---------------------- Entrypoint ----------------------
if __name__ == "__main__":
    run()
