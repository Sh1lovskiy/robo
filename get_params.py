#!/usr/bin/env python3
"""
Dump Intel RealSense (D415) intrinsics/extrinsics/scale to a JSON file.

- Starts color+depth streams, warms up, reads intrinsics and extrinsics,
  reads depth scale, and writes a compact JSON.
- Keeps functions short, single-purpose, and unit-test friendly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pyrealsense2 as rs

# =============================== CONSTANTS ===================================

WARMUP_FRAMES: int = 10
DEFAULT_OUT_PATH: Path = Path("realsense_params.json")

# =============================== HELPERS =====================================


def _reshape_3x3(flat: List[float]) -> List[List[float]]:
    """Convert a flat 9-element list into a 3x3 row-major matrix."""
    if len(flat) != 9:
        raise ValueError("Rotation must have 9 elements.")
    return [flat[0:3], flat[3:6], flat[6:9]]


def intrinsics_to_dict(intr: Any) -> Dict[str, Any]:
    """Serialize rs.intrinsics."""
    return {
        "width": int(intr.width),
        "height": int(intr.height),
        "ppx": float(intr.ppx),
        "ppy": float(intr.ppy),
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "model": str(intr.model),
        "coeffs": [float(c) for c in intr.coeffs],
    }


def extrinsics_to_dict(extr: Any) -> Dict[str, Any]:
    """Serialize rs.extrinsics to rotation (3x3) and translation (3)."""
    rot = [float(x) for x in extr.rotation]
    t = [float(x) for x in extr.translation]
    return {"rotation": _reshape_3x3(rot), "translation": t}


def start_pipeline() -> Tuple[rs.pipeline, rs.pipeline_profile]:
    """Start depth+color with default device profiles."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth)
    config.enable_stream(rs.stream.color)
    profile = pipeline.start(config)
    return pipeline, profile


def warmup(pipeline: rs.pipeline, n_frames: int = WARMUP_FRAMES) -> None:
    """Wait for a few frames so auto-exposure settles."""
    for _ in range(max(0, n_frames)):
        pipeline.wait_for_frames()


def collect_params(profile: rs.pipeline_profile) -> Dict[str, Any]:
    """Read intrinsics, extrinsics, and depth scale."""
    d_prof = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    c_prof = profile.get_stream(rs.stream.color).as_video_stream_profile()

    intr_d = d_prof.get_intrinsics()
    intr_c = c_prof.get_intrinsics()
    extr_d2c = d_prof.get_extrinsics_to(c_prof)
    extr_c2d = c_prof.get_extrinsics_to(d_prof)

    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())

    return {
        "depth_scale": depth_scale,
        "intrinsics": {
            "depth": intrinsics_to_dict(intr_d),
            "color": intrinsics_to_dict(intr_c),
        },
        "extrinsics": {
            "depth_to_color": extrinsics_to_dict(extr_d2c),
            "color_to_depth": extrinsics_to_dict(extr_c2d),
        },
    }


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Write JSON with indentation and stable key order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# ================================== MAIN =====================================


def main(out_json: str | Path = DEFAULT_OUT_PATH) -> int:
    """Entry point. Returns 0 on success, non-zero on failure."""
    out_path = Path(out_json)
    pipeline, profile = start_pipeline()
    try:
        warmup(pipeline, WARMUP_FRAMES)
        params = collect_params(profile)
        save_json(params, out_path)
        print(f"Saved to {out_path}")
        return 0
    finally:
        pipeline.stop()


if __name__ == "__main__":
    raise SystemExit(main())
