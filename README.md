# Robotics Vision & Calibration Suite

A modular, production-grade Python toolkit for robotics calibration, vision, 3D point cloud processing, and automated pose/trajectory collection.

**Supports:**

* Robot high-level API (move, home, state, RPC)
* RealSense 3D camera management
* Automated Charuco and hand-eye calibration (OpenCV)
* 3D point cloud generation, transformation, and visualization
* CLI tools for pose recording, trajectory execution, calibration, camera debug, and cloud ops
* Flexible, extensible logger with JSON/file support

---

## 📂 Project Structure (Navigator)

```
project-root/
│
├── calibration/          # Calibration algorithms & data savers
│   ├── charuco.py        # Charuco board calibration
│   └── handeye.py        # Hand-eye calibration
│
├── cli/                  # CLI utilities (ready-to-run scripts)
│   ├── poses_saver.py           # Save robot poses to file
│   ├── path_runner.py           # Run robot path from file
│   ├── charuco_calib.py         # Charuco calibration wizard
│   ├── handeye_calib.py         # Hand-eye calibration wizard
│   ├── get_intrinsics.py        # Print RealSense camera intrinsics
│   ├── depth_check.py           # Real-time depth stream debug
│   ├── pointcloud_capture.py    # Capture and save point cloud
│   ├── pointcloud_transform.py  # Transform point clouds (hand-eye/world)
│   └── pointcloud_view.py       # View/inspect point clouds
│
├── robot/                # Robot high-level and low-level logic
│   ├── controller.py     # Main control class
│   └── rpc.py            # RPC protocol and low-level driver
│
├── utils/                # Utilities: configuration, logging, constants
│   ├── config.py         # Config loading/abstraction
│   ├── logger.py         # Centralized, JSON-capable logger
│   └── constants.py      # Shared paths and defaults
│
├── vision/               # Vision, cloud, and camera utils
│   ├── opencv_utils.py   # OpenCV helper class
│   ├── realsense.py      # RealSense camera wrapper
│   ├── pointcloud.py     # PointCloudGenerator class and utilities
│   ├── transform.py      # 3D transformation utilities (see vision/README.md)
│   └── README.md         # Explanation of transform chain
|
├── config.yaml           # Main configuration file
├── pyproject.toml        # Project metadata & dependencies
└── README.md             # You are here
```

---

## 🚀 Quickstart

### 1. Clone & Install

```bash
git clone ...
cd project-root
uv venv .venv -p 3.11
source .venv/bin/activate
uv pip install -e .
```

### 2. Configure

Edit `config.yaml` for your robot/camera IP, tool, velocity, logging, and point cloud settings.

### 3. Run CLI Tools

* Save a pose:

  ```bash
  .venv/bin/python -m cli.poses_saver
  ```
* Run a trajectory from file:

  ```bash
  .venv/bin/python -m cli.path_runner
  ```
* Calibrate camera (Charuco):

  ```bash
  .venv/bin/python -m cli.charuco_calib
  ```
* Capture point cloud:

  ```bash
  .venv/bin/python -m cli.pointcloud_capture --output clouds/cloud.ply
  ```

### 4. Build & Install

Use `pyproject.toml` together with [uv](https://github.com/astral-sh/uv) for reproducible environments/builds:

```bash
uv venv .venv -p 3.12
uv pip install -e .
```

All runtime dependencies live in `pyproject.toml`. If you need a classic
`requirements.txt` file, generate it with `uv pip freeze > requirements.txt`.

---

## 📑 Documentation

### Configuration (`config.yaml`)

* `robot:` — IP, tool/user frame, velocity, emergency delay
* `vision:` — RealSense stream parameters, cloud parameters
* `logging:` — log directory, level, JSON output
* `cloud:` — point cloud settings (resolution, voxel size, output dir, ...)
* Defaults for paths, robot IP, and Charuco dictionary mapping are defined in `utils/constants.py`

### Logger (`utils/logger.py`)

* Central logger: file/console, JSON/text, auto-timestamped logs
* Decorators for function/exception logging
* Use in all modules:

  ```python
  from utils.logger import Logger
  logger = Logger.get_logger("my.module", json_format=True)
  ```

### Robot API (`robot/controller.py`)

* `RobotController` — High-level API: connect, move, record, return home, shutdown
* Uses config + logger

### Calibration Modules

* `CharucoCalibrator` — Board-based camera calibration (OpenCV ArUco)
* `HandEyeCalibrator` — Tsai/Daniilidis methods, modular savers for results

### Vision

* `RealSenseCamera` — Start/stop, get frames, intrinsics, depth scale
* `opencv_utils.py` — draw\_text, normalize\_depth, apply\_colormap

### 3D Point Cloud Modules

* `pointcloud.py` — Depth to point cloud conversion, save/load (PLY, XYZ, npz), filtering, merging
* `transform.py` — Rigid transformations between camera/robot/world coordinates (uses calibration, see vision/README.md)
* CLI: `pointcloud_capture.py`, `pointcloud_transform.py`, `pointcloud_view.py`

**See [vision/README.md](./vision/README.md) for detailed usage, coordinate system details, and all math for 3D transforms.**

**Typical workflow:**

1. Capture depth + color → generate cloud (`cli/pointcloud_capture.py`)
2. (Optional) Filter, merge, or transform clouds (robot <-> camera <-> world)
3. Save or visualize result (`cli/pointcloud_view.py`)

### CLI Tools

* All logic in `cli/` is ready for direct call and piped scripting
* Consistent logger and error output

### Extensibility/Testing

* Logger, config, robot, camera: all support dependency injection for unit tests or swapping implementations.
* Add new data savers, control strategies, vision pipelines, or point cloud processors with minimal edits.

---

## 🧰 Troubleshooting

* All logs are stored in `logs/` (JSON if enabled)
* Use `logging.level` and `logging.json` in config.yaml to control verbosity/format
* CLI tools print progress, errors, and file names
* For RealSense errors: check camera connection, permissions, drivers, and stream config
* For robot errors: verify IP, cable, firewall, and physical enable state
* For point cloud issues: check calibration files, cloud parameters, and output formats

---
