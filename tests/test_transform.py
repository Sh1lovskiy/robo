import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from vision.transform import TransformUtils


def test_apply_transform_identity():
    points = np.random.rand(10, 3)
    R = np.eye(3)
    t = np.zeros(3)
    out = TransformUtils.apply_transform(points, R, t)
    assert np.allclose(points, out)
