# calibration package

Utilities for camera calibration and hand‑eye calibration. These modules implement the mathematical routines for estimating camera intrinsics and the transform between the robot tool and the camera.

- `charuco.py` - Charuco board routines and saving helpers
- `handeye.py` - Hand‑eye calibration using OpenCV.  Solves the equation `AX = XB` for the unknown transform `X` by collecting pairs of robot poses and detected Charuco target poses.  Supports Tsai, Daniilidis, Horaud and Park formulations.
- `pose_loader.py` - Load robot pose samples from JSON files.  Each pose is a six‑element vector `[x, y, z, rx, ry, rz]` stored in millimetres and degrees.
- `workflows.py` - High-level calibration workflows and CLI entry points. These functions orchestrate Charuco image capture and the hand‑eye procedure while keeping logging and I/O separated from computation.

Both calibrators keep a single responsibility. The classes accept dependency injected loggers and file savers so they can be reused in scripts or unit tests.

