"""Utilities for loading robot poses."""

from __future__ import annotations

import json
from typing import List, Tuple

import numpy as np

from utils.lmdb_storage import LmdbStorage
from calibration.helpers.validation_utils import euler_to_matrix


class JSONPoseLoader:
    """Load robot poses for hand-eye calibration from a JSON file."""

    @staticmethod
    def load_poses(json_file: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        with open(json_file, "r") as f:
            data = json.load(f)
        Rs, ts = [], []
        for pose in data.values():
            tcp_pose = pose["tcp_coords"]
            t = np.array(tcp_pose[:3], dtype=np.float64) / 1000.0
            rx, ry, rz = tcp_pose[3:]
            R_mat = euler_to_matrix(rx, ry, rz, degrees=True)
            Rs.append(R_mat)
            ts.append(t)
        return Rs, ts


class LmdbPoseLoader:
    """Load robot poses from an LMDB database."""

    @staticmethod
    def load_poses(db_path: str, prefix: str = "poses") -> Tuple[List[np.ndarray], List[np.ndarray]]:
        store = LmdbStorage(db_path, readonly=True)
        keys = sorted(store.iter_keys(f"{prefix}:"), key=lambda k: int(k.split(":")[1]))
        Rs: List[np.ndarray] = []
        ts: List[np.ndarray] = []
        for k in keys:
            pose = store.get_json(k)
            tcp_pose = pose["tcp_coords"]
            t = np.array(tcp_pose[:3], dtype=np.float64) / 1000.0
            rx, ry, rz = tcp_pose[3:]
            R_mat = euler_to_matrix(rx, ry, rz, degrees=True)
            Rs.append(R_mat)
            ts.append(t)
        return Rs, ts

