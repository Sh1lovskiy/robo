# vision: 3D Vision & Point Cloud Module

A robust, modular set of tools for 3D vision tasks: point cloud generation from depth, coordinate transformations, camera-robot calibration (hand-eye), and cloud visualization.

---

## Features

* Convert depth/RGB to dense point clouds in camera, TCP (gripper/tool), or world (robot base) coordinates
* Rigid SE(3) transformation utilities between all frames: camera, TCP, tool offset, world
* Full support for hand-eye calibration (base→cam, TCP→cam, tool→TCP)
* Handles gripper/tool offset (TCP ≠ flange) for correct scene geometry
* Build and apply transform chains for any robot/camera mounting
* Save/load point clouds (PLY, XYZ, npz)
* All steps and transforms logged via `utils.logger`
* Easy to test and extend in production pipelines
* `vision.opencv_utils.show_depth` for quick depth visualization

---

## 📌 Coordinate Systems and Transform Chain

**Coordinate frames:**

* **World (Robot Base):**  Origin at robot base. All absolute poses are here.
* **TCP (Tool Center Point):**  Offset from base by forward kinematics.
* **Tool/Flange:**  Optionally, apply further tool offset (for custom end-effectors).
* **Camera:**  Mounted rigidly, pose given by hand-eye calibration relative to TCP/tool.
* **Point cloud:**  By default, generated in camera frame; transform as needed.

### Full transform chain (with tool offset):

```
T_base_cam = T_base_tcp @ T_tcp_tool @ T_tool_cam
```

* If hand-eye is TCP→camera (OpenCV convention):
  Use robot TCP pose and chain: T\_base\_tcp (from robot) → T\_tcp\_cam (from calibration)
* If mounting is not at TCP (e.g., flange):
  Insert T\_tcp\_tool (from robot/tool config) into the chain

---

## 🚀 Typical Workflow

1. Capture depth + color in camera frame
2. Build point cloud with `vision/pointcloud.py` (Nx3 cloud)
3. Transform to TCP or world coordinates using `vision/transform.py`:

   * Load hand-eye calibration (`handeye.npz`, R, t)
   * Get TCP pose from robot (`RobotController.get_current_pose()`)
   * Build full transform chain with tool offset if needed
4. Save/export/visualize the cloud (Open3D, PLY, etc)

---

## 🧮 Code Examples

### Build full transform chain (base→TCP→tool→camera):

```python
from vision.transform import TransformUtils
import numpy as np

# Load robot TCP pose (from robot state)
tcp_R, tcp_t = ...  # from RobotController or forward kinematics
# Tool offset: [x, y, z, rx, ry, rz] (from config, or zero)
tcp_offset = ...    # from config.yaml or teach pendant
# Hand-eye calibration (TCP→camera)
R_handeye, t_handeye = ...  # from handeye.npz

T_base_tcp = TransformUtils.build_transform(tcp_R, tcp_t)
T_tcp_tool = TransformUtils.tool_to_tcp(tcp_offset)
T_tool_cam = TransformUtils.build_transform(R_handeye, t_handeye)
T_base_cam = TransformUtils.chain_transforms(T_base_tcp, T_tcp_tool, T_tool_cam)
```

### Transform cloud to world:

```python
cloud_cam = ...  # Nx3 from vision.pointcloud
cloud_world = TransformUtils().camera_to_world(cloud_cam, T_base_cam)
```

---

## 🧑‍💻 API Summary

* `TransformUtils.build_transform(R, t)` — Build SE(3) 4x4 from R, t
* `TransformUtils.decompose_transform(T)` — Extract (R, t) from 4x4
* `TransformUtils.transform_points(points, T)` — Apply 4x4 to Nx3 points
* `TransformUtils.world_to_camera(points, T_base_cam)` — World → camera
* `TransformUtils.camera_to_world(points, T_base_cam)` — Camera → world
* `TransformUtils.chain_transforms(*Ts)` — Compose multiple transforms
* `TransformUtils.tool_to_tcp(offset)` — 6DoF tool offset to SE(3) (if needed)

---

## 🧪 Test

```python
def test_transform_utils():
    import numpy as np
    from vision.transform import TransformUtils
    R = np.eye(3)
    t = np.array([1, 2, 3])
    T = TransformUtils.build_transform(R, t)
    points = np.array([[0, 0, 0], [1, 0, 0]])
    points_w = TransformUtils().camera_to_world(points, T)
    assert np.allclose(points_w, points + t)
    points_c = TransformUtils().world_to_camera(points_w, T)
    assert np.allclose(points_c, points)
```

---

## 🔬 References

* [OpenCV Hand-Eye Calibration](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga34d5c42c2290c1c2137be7fddbc75b5d)
* [OpenCV Forum: Eye-to-Hand/Hand-Eye conventions](https://forum.opencv.org/t/eye-to-hand-calibration/5690/2)
* [arXiv: On Hand-Eye Calibration](https://arxiv.org/pdf/2311.12655)
