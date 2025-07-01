# calibration package

Calibration utilities for recovering camera intrinsics and the rigid transform between a robot and a camera.

- `charuco.py` - Charuco board routines and saving helpers
- `handeye.py` - Hand‑eye calibration using OpenCV. Solves `AX = XB` for the tool→camera transform with Tsai, Daniilidis, Horaud and Park methods.
- `pose_loader.py` - Load robot pose samples from JSON files.
- `workflows.py` - High-level calibration workflows and CLI entry points built with `CommandDispatcher`.

The core classes (:class:`CharucoCalibrator` and :class:`HandEyeCalibrator`) are decoupled from I/O so they can be reused in scripts or unit tests.  Command line workflows are exposed via ``calibration-cli``.

