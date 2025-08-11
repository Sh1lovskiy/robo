#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
manual_capture.py — Manual RGB/Depth capture with TCP logging + real-time depth preview.

Flow:
- Connect to robot RPC (read-only) — only to query TCP right after user confirms capture.
- Start RealSense + background preview loop that shows RGB and depth (meters) in OpenCV windows.
- Loop:
    * Ask: "Capture now? [y/N/q]".
    * If 'y': read TCP now, print it, snapshot latest RGB/Depth from preview loop, save, append poses.json.
    * If 'q': exit. Otherwise: skip and continue.

Files:
  captures/{timestamp}/imgs/{idx}_rgb.png     (uint8, BGR)
  captures/{timestamp}/imgs/{idx}_depth.npy   (float32 meters)
  captures/{timestamp}/poses.json             (x,y,z mm; rx,ry,rz deg)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import threading

import cv2
import numpy as np

from utils.logger import Logger
from utils.error_tracker import ErrorTracker
from robot.rpc import RPC

# ========================== CONFIG ==========================
ROBOT_IP = "192.168.58.2"  # Robot controller IP (read-only)
RS_WIDTH = 1280
RS_HEIGHT = 720
RS_FPS = 30
WARMUP_FRAMES = 10

# Preview settings
SHOW_PREVIEW = True
PREVIEW_WINDOW_RGB = "RGB"
PREVIEW_WINDOW_DEPTH = "Depth (m)"
# If None -> auto range per frame (percentile-based). Or set e.g. 0.2..2.0
DEPTH_VIZ_MIN_M = None
DEPTH_VIZ_MAX_M = None
# ===========================================================

log = Logger.get_logger("manual.capture")


# ---------------------- RealSense helper with preview thread ----------------------
class RealSense:
    """RGB/Depth capture with background preview. RGB -> PNG; Depth -> NPY (meters)."""

    def __init__(self, width=1280, height=720, fps=30, warmup=10, show_preview=True):
        self.width = width
        self.height = height
        self.fps = fps
        self.warmup = warmup
        self.show_preview = show_preview

        self.pipeline = None
        self.align = None
        self.depth_scale = None
        self.rs = None

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last_color: Optional[np.ndarray] = None
        self._last_depth_m: Optional[np.ndarray] = None

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
        profile = self.pipeline.start(cfg)
        self.depth_scale = float(
            profile.get_device().first_depth_sensor().get_depth_scale()
        )

        # warmup buffers
        for _ in range(self.warmup):
            self.pipeline.wait_for_frames()
        log.info("RealSense pipeline started")

        # start preview loop
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._preview_loop, name="rs-preview", daemon=True
        )
        self._thread.start()

    def stop(self):
        # stop thread first (so it stops using OpenCV windows and pipeline)
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

        # then stop pipeline
        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None

        # close windows
        if self.show_preview:
            try:
                cv2.destroyWindow(PREVIEW_WINDOW_RGB)
                cv2.destroyWindow(PREVIEW_WINDOW_DEPTH)
            except Exception:
                pass

        log.info("RealSense pipeline stopped")

    def _preview_loop(self):
        if self.show_preview:
            cv2.namedWindow(PREVIEW_WINDOW_RGB, cv2.WINDOW_NORMAL | cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow(
                PREVIEW_WINDOW_DEPTH, cv2.WINDOW_NORMAL | cv2.WINDOW_AUTOSIZE
            )

        while not self._stop.is_set():
            try:
                frames = self.pipeline.wait_for_frames()
                frames = self.align.process(frames)
                color_raw = frames.get_color_frame()
                depth_raw = frames.get_depth_frame()
                if not color_raw or not depth_raw:
                    continue

                color = np.asanyarray(color_raw.get_data())
                depth_m = (
                    np.asanyarray(depth_raw.get_data()).astype(np.float32)
                    * self.depth_scale
                )

                with self._lock:
                    self._last_color = color
                    self._last_depth_m = depth_m

                if self.show_preview:
                    cv2.imshow(PREVIEW_WINDOW_RGB, color)
                    cv2.imshow(PREVIEW_WINDOW_DEPTH, self._depth_to_viz(depth_m))
                    cv2.waitKey(1)
            except Exception as e:
                log.warning(f"Preview loop warning: {e}")
                time.sleep(0.01)

    def _depth_to_viz(self, depth_m: np.ndarray) -> np.ndarray:
        d = depth_m.copy()
        # mask zeros
        mask = d > 0
        if DEPTH_VIZ_MIN_M is None or DEPTH_VIZ_MAX_M is None:
            if np.any(mask):
                vmin = (
                    np.percentile(d[mask], 2.0)
                    if DEPTH_VIZ_MIN_M is None
                    else DEPTH_VIZ_MIN_M
                )
                vmax = (
                    np.percentile(d[mask], 98.0)
                    if DEPTH_VIZ_MAX_M is None
                    else DEPTH_VIZ_MAX_M
                )
            else:
                vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = float(DEPTH_VIZ_MIN_M), float(DEPTH_VIZ_MAX_M)
        if vmax <= vmin:
            vmax = vmin + 1e-3

        d = np.clip((d - vmin) / (vmax - vmin), 0.0, 1.0)
        d = (d * 255.0).astype(np.uint8)
        d_color = cv2.applyColorMap(d, cv2.COLORMAP_JET)
        # set invalid (zero) to black
        d_color[~mask] = (0, 0, 0)
        return d_color

    def snapshot(self, timeout_s: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return copies of the latest (color, depth_m). Waits up to timeout_s for first frames.
        """
        t0 = time.time()
        while True:
            with self._lock:
                if self._last_color is not None and self._last_depth_m is not None:
                    return self._last_color.copy(), self._last_depth_m.copy()
            if time.time() - t0 > timeout_s:
                raise RuntimeError(
                    "RealSense snapshot timeout: no frames available yet"
                )
            time.sleep(0.01)


# ---------------------- Small utils ----------------------
def fmt_pose(p: List[float]) -> str:
    return f"[{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}, {p[3]:.3f}, {p[4]:.3f}, {p[5]:.3f}]"


def ask(prompt: str) -> str:
    return input(prompt).strip().lower()


# ---------------------- Main capture loop ----------------------
def run() -> None:
    ErrorTracker.install_excepthook()
    ErrorTracker.install_signal_handlers()

    ts = time.strftime("%Y%m%d_%H%M%S")
    outdir = Path("captures") / ts
    imgdir = outdir / "imgs"
    outdir.mkdir(parents=True, exist_ok=True)
    imgdir.mkdir(parents=True, exist_ok=True)
    poses_path = outdir / "poses.json"
    poses: Dict[str, Dict[str, float]] = {}

    # Connect to robot RPC (read-only)
    log.info(f"Connecting to robot at {ROBOT_IP} (read-only)...")
    rpc = RPC(ip=ROBOT_IP)
    log.info("Connected.")

    # Start camera (+ preview thread)
    cam = RealSense(
        RS_WIDTH, RS_HEIGHT, RS_FPS, warmup=WARMUP_FRAMES, show_preview=SHOW_PREVIEW
    )
    cam.start()

    try:
        index = 0
        while True:
            ans = ask("Capture now? [y/N/q]: ")
            if ans == "q":
                log.info("Quit requested by user.")
                break
            if ans not in ("y", "yes"):
                log.info("Capture skipped by user.")
                continue

            # Read TCP ONLY after confirmation
            err, tcp = rpc.GetActualTCPPose(1)
            if err != 0 or not isinstance(tcp, (list, tuple)) or len(tcp) != 6:
                log.error(
                    f"GetActualTCPPose failed right before capture: err={err}, tcp={tcp}"
                )
                retry = ask("TCP unavailable. Retry capture? [y/N]: ")
                if retry in ("y", "yes"):
                    continue
                else:
                    log.info("Capture aborted due to TCP read failure.")
                    continue

            pose6 = [float(v) for v in tcp]
            print(f"TCP (at capture): {fmt_pose(pose6)}")
            log.info(f"TCP (at capture): {fmt_pose(pose6)}")

            # Snapshot latest frames from preview loop
            color, depth_m = cam.snapshot()

            idx = f"{index:03d}"
            rgb_path = imgdir / f"{idx}_rgb.png"
            dpt_path = imgdir / f"{idx}_depth.npy"

            cv2.imwrite(str(rgb_path), color)
            np.save(str(dpt_path), depth_m.astype(np.float32))

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

            log.info(f"Saved: {rgb_path.name}, {dpt_path.name}; poses.json updated.")
            index += 1

        log.info(f"Session dir: {outdir}")
        log.info(f"Images: {imgdir}")
        log.info(f"Poses JSON: {poses_path}")

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


if __name__ == "__main__":
    run()
