import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from utils.io import save_camera_params_xml, load_camera_params
import tempfile
import os

def test_save_and_load_camera_params(tmp_path):
    K = np.eye(3)
    dist = np.zeros(5)
    xml = tmp_path / "cam.xml"
    save_camera_params_xml(xml, K, dist)
    K2, dist2 = load_camera_params(xml)
    assert np.allclose(K, K2)
    assert np.allclose(dist, dist2)
