# calibration package

Utilities for camera calibration and hand‑eye calibration. These modules implement the mathematical routines for estimating camera intrinsics and the transform between the robot tool and the camera.

- `charuco.py` - Charuco board routines and saving helpers
- `handeye.py` - Hand‑eye calibration using OpenCV. Solves `AX = XB` for the tool→camera transform with Tsai, Daniilidis, Horaud and Park methods.
- `pose_loader.py` - Load robot pose samples from JSON files.
- `workflows.py` - High-level calibration workflows and CLI entry points built with `CommandDispatcher`.

Both calibrators keep a single responsibility and accept dependency injected loggers and file savers so they can be reused in scripts or unit tests.

