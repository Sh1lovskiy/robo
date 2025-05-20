import cv2
import numpy as np
from math import cos, sin


def load_camera_matrix(file_path):
    """Load camera matrix from file."""
    with open(file_path, "r") as file:
        return np.array(
            [list(map(float, line.strip().split(","))) for line in file],
            dtype=np.float32,
        )


def load_distortion_coefficients(file_path):
    """Load distortion coefficients from file."""
    with open(file_path, "r") as file:
        return np.array(
            list(map(float, file.read().strip().split(","))), dtype=np.float32
        )


def dh_transform(theta, d, a, alpha):
    return np.array(
        [
            [
                cos(theta),
                -sin(theta) * cos(alpha),
                sin(theta) * sin(alpha),
                a * cos(theta),
            ],
            [
                sin(theta),
                cos(theta) * cos(alpha),
                -cos(theta) * sin(alpha),
                a * sin(theta),
            ],
            [0, sin(alpha), cos(alpha), d],
            [0, 0, 0, 1],
        ]
    )


def rotation_matrix(angle, axis):
    c, s = cos(angle), sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rotation_vector_to_euler(rvec):
    """Convert rotation vector to Euler angles (yaw, pitch, roll) in degrees.

    Args:
        rvec: Rotation vector (3,) from Rodrigues format

    Returns:
        Tuple: (yaw, pitch, roll) in degrees
    """
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        yaw = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        yaw = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        roll = 0

    yaw = np.degrees(yaw)
    pitch = np.degrees(pitch)
    roll = np.degrees(roll)
    return yaw, pitch, roll


def pose_to_transform(x, y, z, yaw, pitch, roll):
    """Transform coordinates from XYZ + Euler angles to 4x4 matrix"""
    yaw = np.deg2rad(yaw)
    pitch = np.deg2rad(pitch)
    roll = np.deg2rad(roll)

    Rz = np.array([[cos(yaw), -sin(yaw), 0], [sin(yaw), cos(yaw), 0], [0, 0, 1]])

    Ry = np.array(
        [[cos(pitch), 0, sin(pitch)], [0, 1, 0], [-sin(pitch), 0, cos(pitch)]]
    )

    Rx = np.array([[1, 0, 0], [0, cos(roll), -sin(roll)], [0, sin(roll), cos(roll)]])

    R = Rz @ Ry @ Rx
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T
