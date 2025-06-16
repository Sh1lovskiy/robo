# calibration/pose_loader.py

import json
import numpy as np
from scipy.spatial.transform import Rotation as R


class JSONPoseLoader:
    """
    Loads robot poses for hand-eye calibration from a JSON file.
    Expects keys:
        - "robot_tcp_pose": [x, y, z, rx, ry, rz] (angles in degrees or radians, обсудим ниже)
    """

    @staticmethod
    def load_poses(json_file):
        with open(json_file, "r") as f:
            data = json.load(f)

        Rs, ts = [], []
        for pose in data.values():
            tcp_pose = pose["tcp_coords"]  # [x, y, z, rx, ry, rz]
            t = np.array(tcp_pose[:3], dtype=np.float64) / 1000.0  # mm → meters
            rx, ry, rz = np.deg2rad(tcp_pose[3:])  # если углы в градусах!
            rot = R.from_euler("xyz", [rx, ry, rz])
            R_mat = rot.as_matrix()
            Rs.append(R_mat)
            ts.append(t)
        return Rs, ts
