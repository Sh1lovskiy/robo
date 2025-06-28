# Robotics Vision & Calibration Suite

A modular, production-grade Python toolkit for robotics calibration, vision, 3D point cloud processing, and automated pose/trajectory collection.

This repository groups several focused modules under a single project umbrella. Each individual components remain testable and extensible. The sections below describe the directory layout and how the pieces fit together.

**Supports:**

* Robot high-level API (move, home, restart, state, RPC)
* Pluggable camera interface with a RealSense implementation
* Automated Charuco and hand-eye calibration (OpenCV)
* 3D point cloud capture, filtering and visualization
* CLI tools built with a `CommandDispatcher` for calibration, robot control and cloud ops
* Centralized logger and `ErrorTracker` with optional JSON output
* RocksDB-based storage backend for poses, images and metrics
* LMDB-based storage backend for poses, images and metrics

---

## 📂 Project Structure (Navigator)

```
project-root/
│
├── calibration/          # Calibration algorithms & workflows
│   ├── charuco.py              # Charuco board calibration helpers
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
├── utils/                # Common utilities
│   ├── config.py               # Config loading/abstraction
│   ├── logger.py               # Centralized, JSON-capable logger
│   ├── error_tracker.py        # Global exception and signal handling
│   ├── cli.py                  # CommandDispatcher helper
│   ├── keyboard.py             # Global hotkey listener
│   ├── io.py                   # Camera calibration I/O
│   ├── geometry.py             # Math helpers
│   └── README.md               # Package overview
│
├── storage/             # LMDB storage layer
│   ├── interface.py          # IStorage definitions
│   ├── lmdb_storage.py       # LMDB implementation
│   └── README.md             # Package overview
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
│   ├── tools.py                # Camera and cloud helper CLI
│   ├── transform.py            # 3D transformation utilities
│   └── README.md               # Explanation of transform chain
│
├── conf/app.yaml         # Main configuration file
├── pyproject.toml        # Project metadata & dependencies
└── README.md             # You are here
```

---

## Overview

The project is organized into four main packages that mirror typical robotics layers:

* **calibration/** – Charuco board detection and hand‑eye calibration. The mathematical background of the pose estimation and the classical `AX = XB` formulation are explained in [calibration/README.md](calibration/README.md).
* **robot/** – Robot connection logic and high‑level motion API. Communication is decoupled from workflows so hardware can be replaced without touching the algorithm code (Dependency Inversion Principle).
* **vision/** – Camera interfaces, point cloud tools, and coordinate transforms. Transform chains and coordinate conventions appear in [vision/README.md](vision/README.md).
* **utils/** – Shared helpers for configuration, logging and geometry calculations.
* **storage/** – LMDB based persistence layer for images and metadata.

Every component keeps a single responsibility and exposes a minimal interface. New robots or cameras can be integrated by implementing the same interfaces without modifying existing modules.

### Design Principles

* **Single Responsibility** – each module focuses on one task only.
* **Open/Closed** – functionality is extended via abstractions (e.g. `Camera` base class).
* **Liskov Substitution** – alternate controllers or cameras can drop in without breaking workflows.
* **Interface Segregation** – small APIs are preferred for ease of testing.
* **Dependency Inversion** – high level workflows depend on abstract interfaces, not implementations.

---

## 🚀 Quickstart

### 1. Build & Install

Use `pyproject.toml` together with [uv](https://github.com/astral-sh/uv) for reproducible environments/builds:

```bash
uv venv .venv -p 3.12
uv pip install -e .
```

All dependencies are defined in `pyproject.toml`.

### 2. Configure

Edit `conf/app.yaml` for your robot/camera IP, tool, velocity, logging, and point cloud settings. Configuration is managed with [Hydra](https://github.com/facebookresearch/hydra), so any value can be overridden via the command line:

```bash
python some_module.py robot.ip=192.168.1.10 logging.level=DEBUG
```

### 3. Run CLI Tools
CLI modules are thin wrappers calling workflow helpers under
`calibration/`, `robot/`, and `vision/`.

* Save a pose:

  ```bash
  poses-saver
  ```
* Run a trajectory from file:

  ```bash
  path-runner
  ```
* Calibrate camera (Charuco):

  ```bash
  charuco-calib
  ```
* Capture point cloud:

  ```bash
  pointcloud-capture --output clouds/cloud.ply
  ```

---

## 📑 Documentation

### Configuration (`conf/app.yaml`)

* `robot:` — IP, tool/user frame, velocity, emergency delay
* `vision:` — RealSense stream parameters, cloud parameters
* `logging:` — log directory, level, JSON output
* `cloud:` — point cloud setting, (resolution, voxel size, output dir, ...)
* Default paths, robot IP and Charuco dictionary mapping live in `conf/app.yaml`

### Logger (`utils/logger.py`)

* Central logger: file/console, JSON/text, progress bars
* Decorators for function/exception logging
* Integrated `ErrorTracker` for uncaught exceptions
* Use in all modules:

  ```python
  from utils.logger import Logger
  logger = Logger.get_logger("my.module", json_format=True)
  ```

### Error Tracker (`utils/error_tracker.py`)

* Global exception and signal handler
* Executes registered cleanup functions on failures
* Optional hotkeys via `utils.keyboard`

### Storage (`storage/lmdb_storage.py`)

* `LmdbStorage` — default persistence backend implementing `IStorage`

```python
from storage import LmdbStorage
store = LmdbStorage("app.lmdb")
store.put_json("pose:0", {"tcp_coords": [0, 0, 0, 0, 0, 0]})
img = store.get_image("rgb:0")
```

### Robot API (`robot/controller.py`)

* `RobotController` — High-level API: connect, move, restart, record, return home, shutdown
* Uses `Config` and `Logger` instances

### Calibration Modules

* `CharucoCalibrator` — Board-based camera calibration (OpenCV ArUco)
* `HandEyeCalibrator` — Tsai/Daniilidis methods, modular savers for results

### Vision

* `Camera` base class — abstract interface for camera drivers
* `RealSenseCamera` — Start/stop, get frames, intrinsics, depth scale
* `camera_utils.py` — intrinsics printer and depth checker
* `opencv_utils.py` — draw\_text, normalize\_depth, apply\_colormap

### 3D Point Cloud Modules

* `cloud/generator.py` — Depth to point cloud conversion, save/load (PLY, XYZ, npz)
* `cloud/aggregator.py` — Multi-frame cloud assembly with optional ICP
* `cloud/pipeline.py` — Filtering, clustering, and trajectory analysis
* `transform.py` — Rigid transformations between camera/robot/world coordinates (uses calibration, see vision/README.md)
* CLI scripts provided via entry points (`pointcloud-capture`, `pointcloud-transform`, `pointcloud-view`) and built with `utils.cli.CommandDispatcher`

**See [vision/README.md](./vision/README.md) for detailed usage, coordinate system details, and all math for 3D transforms.**

**Typical workflow:**

1. Capture depth + color → generate cloud (`pointcloud-capture`)
2. (Optional) Filter, merge, or transform clouds (robot <-> camera <-> world)
3. Save or visualize result (`pointcloud-view`)

### Hand-Eye Validation

Run `validation-cli` to compute projection errors and analyze datasets. Example:

```bash
validation-cli validate-pose
validation-cli analyze-dataset
```

## Mathematical Background

Camera calibration uses OpenCV's implementation of Zhang's algorithm.  Given a set of world coordinates \(X_i\) and their observed pixel locations \(x_i\), the intrinsic matrix \(K\) and distortion parameters are solved by minimizing

$$
\sum_i \left\| x_i - \pi( K [R \; t] X_i ) \right\|^2,
$$

where $\pi$ is the perspective projection.  Hand‑eye calibration solves the classic $AX = XB$ equation using multiple poses from the robot and the target marker. The unknown transform $X$ (camera with respect to tool) is recovered via methods such as Tsai–Lenz or Daniilidis. Once $X$ is known, point clouds can be transformed to the robot base with

$$
T_{base \leftarrow cam} = T_{base \leftarrow tcp} \times T_{tcp \leftarrow tool} \times T_{tool \leftarrow cam}.
$$

Rigid motions are represented as 4×4 matrices in $SE(3)$. Composition uses matrix multiplication and inverse transforms use the transpose for rotation with the negative translated vector.

### CLI Tools

Entry points defined in `pyproject.toml` expose the common workflows:
`poses-saver`, `path-runner`, `charuco-calib`, etc. The underlying logic lives within the respective packages.

### Extensibility/Testing
* Logger, config, robot, camera: all support dependency injection for unit tests or swapping implementations.
* Add new data savers, control strategies, vision pipelines, or point cloud processors with minimal edits.

---

## 🧰 Troubleshooting

* All logs are stored in `logs/` (JSON if enabled)
* Use `logging.level` and `logging.json` in `conf/app.yaml` to control verbosity/format
* CLI tools print progress, errors, and file names
* For RealSense errors: check camera connection, permissions, drivers, and stream config
* For robot errors: verify IP, cable, firewall, and physical enable state
* For point cloud issues: check calibration files, cloud parameters, and output formats

---
