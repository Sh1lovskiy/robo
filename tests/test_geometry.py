import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from utils.geometry import euler_to_matrix


def test_euler_to_matrix_identity():
    R = euler_to_matrix(0, 0, 0)
    assert np.allclose(R, np.eye(3))
