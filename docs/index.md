# Robotics Vision & Calibration Suite

Welcome to the project documentation. The repository groups several robotics and computer vision utilities under a single umbrella. The core packages are:

- **`calibration/`** – algorithms for intrinsic camera calibration, hand‑eye estimation and data collection.
- **`robot/`** – a minimal robot control API with pose recording workflows.
- **`robot_scan/`** – RGB‑D capture, point‑cloud processing and robot‑guided scanning.
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
├── robot_scan/           # Robot-guided scanning pipeline
│   ├── capture.py             # RealSense RGB‑D frame capture
│   ├── graph.py               # Skeleton graph construction
│   ├── main.py                # End-to-end CLI pipeline
│   ├── motion.py              # Robot motion helpers
│   ├── preprocess.py          # Cloud filtering and plane detection
│   ├── save.py                # Data export utilities
│   ├── skeleton.py            # 2D skeletonization helpers
│   └── visualization.py       # Plotly-based inspection tools
```

Use the navigation on the left to explore usage examples, pipeline descriptions and the full Python API.
