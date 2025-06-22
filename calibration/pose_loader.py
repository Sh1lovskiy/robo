# calibration/pose_loader.py
"""Helper to load robot poses from JSON files."""

from __future__ import annotations

import json
from typing import List, Tuple

import numpy as np

from utils.geometry import euler_to_matrix


class JSONPoseLoader:
    """
    Loads robot poses for hand-eye calibration from a JSON file.
    Expects keys:
        - "robot_tcp_pose": [x, y, z, rx, ry, rz] (angles in degrees or radians)
    """

    @staticmethod
    def load_poses(json_file: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        with open(json_file, "r") as f:
            data = json.load(f)

        Rs, ts = [], []
        for pose in data.values():
            tcp_pose = pose["tcp_coords"]  # [x, y, z, rx, ry, rz]
            t = np.array(tcp_pose[:3], dtype=np.float64) / 1000.0  # mm → m
            rx, ry, rz = tcp_pose[3:]
            R_mat = euler_to_matrix(rx, ry, rz, degrees=True)
            Rs.append(R_mat)
            ts.append(t)
        return Rs, ts
