"""
Utility module for 3D transformation operations.
Provides functions to create and manipulate 4x4 transformation matrices,
convert between rotation vectors and matrices, and apply transformations to points.
"""

import cv2
import numpy as np


def matrix_from_rtvec(rvec, tvec):
    """Convert rotation vector and translation vector to 4x4 transformation matrix."""
    (R, jac) = cv2.Rodrigues(rvec)  # ignore the jacobian
    M = np.eye(4)
    M[0:3, 0:3] = R
    M[0:3, 3] = tvec.squeeze()  # 1-D vector, row vector, column vector, whatever
    return M


def rtvec_from_matrix(M):
    """Convert 4x4 transformation matrix to rotation and translation vectors."""
    (rvec, jac) = cv2.Rodrigues(M[0:3, 0:3])  # ignore the jacobian
    tvec = M[0:3, 3]
    assert M[3] == [0, 0, 0, 1], M  # sanity check
    return (rvec, tvec)


def vec(vec):
    """Convert input to numpy array."""
    return np.asarray(vec)


def normalize(vec):
    """Normalize a vector to unit length."""
    return np.asarray(vec) / np.linalg.norm(vec)


def translate(dx=0, dy=0, dz=0):
    """Create a 4x4 translation matrix."""
    M = np.eye(4)
    M[0:3, 3] = (dx, dy, dz)
    return M


def rotate(axis, angle=None):
    """Create a 4x4 rotation matrix from axis and optional angle."""
    if angle is None:
        rvec = vec(axis)
    else:
        rvec = normalize(axis) * angle

    (R, jac) = cv2.Rodrigues(rvec)
    M = np.eye(4)
    M[0:3, 0:3] = R
    return M


def scale(s=1, sx=1, sy=1, sz=1):
    """Create a 4x4 scaling matrix."""
    M = np.diag([s * sx, s * sy, s * sz, 1])
    return M


def apply_to_rowvec(T, data):
    """Apply 4x4 transformation matrix to row vectors (3D or 4D homogeneous coordinates)."""
    (n, k) = data.shape

    if k == 4:
        pass
    elif k == 3:
        data = np.hstack([data, np.ones((n, 1))])
    else:
        assert False, k

    # Transform data using matrix multiplication
    data = data @ T.T
    return data[:, :k]
