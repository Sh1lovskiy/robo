"""Utility helpers for calibration workflows."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
import yaml

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from utils.logger import Logger

log = Logger.get_logger("calibrate.utils")

np.set_printoptions(suppress=True, precision=6)


@dataclass
class ImagePair:
    """Container for synchronized RGB/depth file paths."""

    rgb: Path
    depth: Path


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def create_output_dir(
    pattern: str, board_size: Tuple[int, int], square_length: float
) -> Path:
    """Return a timestamped directory for storing calibration data."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{timestamp}_{pattern}_{board_size[0]}x{board_size[1]}_{square_length:.3f}"
    out_dir = Path(".data_calib") / name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_image_pair(
    rgb: np.ndarray, depth: np.ndarray, out_dir: Path, index: int
) -> ImagePair:
    """Save RGB and depth images to ``out_dir`` using zero padded index."""

    rgb_path = out_dir / f"{index:03d}_rgb.png"
    depth_path = out_dir / f"{index:03d}_depth.npy"
    cv2.imwrite(str(rgb_path), rgb)
    np.save(depth_path, depth)
    log.debug(f"Saved image pair {rgb_path.stem}")
    return ImagePair(rgb_path, depth_path)


def load_image_pairs(image_dir: Path) -> List[ImagePair]:
    """Return all RGB/depth pairs from ``image_dir``.

    The function expects files named ``*_rgb.png`` with matching
    ``*_depth.npy`` files.
    """

    pairs: List[ImagePair] = []
    for rgb_path in sorted(image_dir.glob("*_rgb.png")):
        depth_path = rgb_path.with_name(
            rgb_path.stem.replace("_rgb", "_depth") + ".npy"
        )
        if depth_path.exists():
            pairs.append(ImagePair(rgb_path, depth_path))
        else:
            log.warning(f"Missing depth file for {rgb_path.name}")
    log.info(f"Loaded {len(pairs)} image pairs from {image_dir}")
    return pairs


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------


def load_intrinsics_yml(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read camera intrinsic matrix and distortion coefficients from YAML."""

    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        K = np.array(data["camera_matrix"]).reshape(3, 3)
        dist = np.array(data["distortion_coefficients"])
        return K, dist

    with open(path) as f:
        yml = yaml.safe_load(f)
    K = np.array(yml["camera_matrix"]["data"]).reshape(3, 3)
    dist = np.array(yml["distortion_coefficients"]["data"])
    return K, dist


def save_intrinsics(K: np.ndarray, dist: np.ndarray, out_file: Path) -> None:
    """Persist intrinsics and distortion coefficients to JSON."""

    data = {
        "camera_matrix": K.tolist(),
        "distortion_coefficients": dist.tolist(),
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    log.info(f"Saved intrinsics to {out_file}")


def estimate_pose_pnp(
    obj_pts: np.ndarray, img_pts: np.ndarray, K: np.ndarray, dist: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Estimate pose using ``cv2.solvePnP`` returning transformation matrix and error."""

    success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist)
    if not success:
        raise RuntimeError("PnP failed")
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    err = float(np.linalg.norm(proj.squeeze(1) - img_pts, axis=1).mean())
    Rmat, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = Rmat
    T[:3, 3] = tvec.ravel()
    return T, err


def rotation_matrix_to_xyz(rot: np.ndarray) -> Tuple[float, float, float]:
    """Convert rotation matrix to rotation-vector components."""

    vec = R.from_matrix(rot).as_rotvec()
    return float(vec[0]), float(vec[1]), float(vec[2])


def save_poses(poses: List[np.ndarray], out_file: Path) -> None:
    """Persist pose matrices in the required JSON format."""

    results = {}
    for idx, T in enumerate(poses):
        x, y, z = T[:3, 3]
        rx, ry, rz = rotation_matrix_to_xyz(T[:3, :3])
        results[f"{idx:03d}"] = {
            "x": round(float(x), 6),
            "y": round(float(y), 6),
            "z": round(float(z), 6),
            "Rx": round(rx, 6),
            "Ry": round(ry, 6),
            "Rz": round(rz, 6),
        }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved {len(poses)} poses to {out_file}")


# ---------------------------------------------------------------------------
# User interaction
# ---------------------------------------------------------------------------


def confirm(prompt: str, default: bool = True) -> bool:
    """Ask the user to confirm an action."""
    try:
        from utils.keyboard import TerminalEchoSuppressor
    except Exception:  # pragma: no cover - fallback when keyboard unavailable

        class TerminalEchoSuppressor:  # type: ignore
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

    yn = "Y/n" if default else "y/N"
    prompt = f"{prompt} [{yn}]: "
    with TerminalEchoSuppressor():
        ans = input(prompt).strip().lower()
    if not ans:
        return default
    return ans.startswith("y")


def parse_board_size(text: str) -> Tuple[int, int]:
    """Parse a ``WxH`` board size string."""

    try:
        w, h = text.lower().split("x")
        return int(w), int(h)
    except Exception as exc:  # pragma: no cover - user error path
        raise argparse.ArgumentTypeError(f"Invalid board size '{text}'") from exc
