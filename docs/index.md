# Robotics Vision & Calibration Suite

Welcome to the project documentation. The repository groups several robotics and computer vision utilities under a single umbrella. The core packages are:

- **`calibration/`** – algorithms for intrinsic camera calibration, hand‑eye estimation and data collection.
- **`robot/`** – a minimal robot control API with pose recording workflows.
- **`vision/`** – camera drivers, point cloud utilities and RGB‑D pipelines.
- **`utils/`** – shared helpers for logging, CLI dispatch and geometry math.

The [README](../README.md) describes the directory layout in detail. Below is an excerpt:

```text
project-root/
│
├── calibration/          # Calibration algorithms & workflows
│   ├── pattern.py              # Unified calibration patterns
│   ├── handeye.py              # Hand-eye calibration helpers
│   ├── pose_loader.py          # Load robot poses from JSON
│   ├── workflows.py            # High-level calibration routines & CLI
│   └── README.md               # Package overview
│
├── robot/                # Robot API & workflows
│   ├── controller.py           # Main control class
│   ├── workflows.py            # Pose recorder and path runner
│   ├── marker.py               # Simple marker utilities
│   ├── Robot.py                # Cython RPC bindings
│   └── README.md               # Package overview
│
├── utils/                # Common utilities and storage
│   ├── logger.py               # Centralized, JSON-capable logger
│   ├── error_tracker.py        # Global exception and signal handling
│   ├── cli.py                  # CommandDispatcher helper
│   ├── keyboard.py             # Global hotkey listener
│   ├── io.py                   # File and camera parameter I/O
│   └── README.md               # Package overview
│
├── vision/               # Vision, cloud, and camera utils
│   ├── camera_base.py          # Abstract camera interface
│   ├── realsense.py            # RealSense camera implementation
│   ├── camera_utils.py         # Depth/intrinsic debug helpers
│   ├── opencv_utils.py         # OpenCV helper class
│   ├── cloud/                  # Point cloud subpackage
│   │   ├── generator.py            # PointCloudGenerator class
│   │   ├── aggregator.py           # Multi-frame cloud builder
│   │   └── pipeline.py             # Filtering/analysis helpers
│   ├── mapping/               # PatchMatch & point set registration
│   ├── tools.py                # Camera and cloud helper CLI
│   ├── transform.py            # 3D transformation utilities
│   └── README.md               # Explanation of transform chain
```

Use the navigation on the left to explore usage examples, pipeline descriptions and the full Python API.
