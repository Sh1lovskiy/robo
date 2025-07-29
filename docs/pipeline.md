# Hand‑Eye Calibration Pipeline

This section summarizes the calibration workflow implemented in `handeye_chess.py` and `handeye_charuco.py`.

## Overview

1. **Load camera parameters and robot poses.** Intrinsics `K` and distortion `d` are read from `cam_params.yml`. Depth‑to‑RGB extrinsics `R_ext`, `t_ext` provide sensor alignment.
2. **Detect the calibration pattern.** Chess boards use `cv2.findChessboardCorners`; Charuco boards use `cv2.aruco.detectMarkers` followed by `cv2.aruco.interpolateCornersCharuco`.
3. **Filter detections** by reprojection error and board pose quality.
4. **Estimate the board pose** via `solvePnP` to obtain rotation `R` and translation `t` for each frame.
5. **Assemble camera↔robot pairs** and run `cv2.calibrateHandEye` using OpenCV’s Tsai/Park/Horaud/Andreff/Daniilidis methods.
6. **Save results and diagnostics.** The final hand–eye matrix is written to `handeye_res.yaml` together with per‑frame logs and overlay images.

```
RGB/Depth frames ──► detect pattern ──► solvePnP ──► hand‑eye calibrate ──► YAML
```

All measurements use meters and degrees. Input images are loaded from `calib/imgs/frame_*.png` with matching depth maps `frame_*.npy`. Robot poses are stored as `{ "tcp_coords": [x, y, z, Rx, Ry, Rz] }`.
