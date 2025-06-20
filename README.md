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
├── calibration/          # Calibration algorithms & workflows
│   ├── charuco.py              # Charuco board calibration helpers
│   ├── handeye.py              # Hand-eye calibration helpers
│   ├── pose_loader.py          # Load robot poses from JSON
│   ├── workflows.py            # High-level calibration routines
│   └── README.md               # Package overview
│
├── robot/                # Robot API & workflows
│   ├── controller.py          # Main control class
│   ├── workflows.py          # Pose recorder and path runner
│   ├── marker.py             # Simple marker utilities
│   ├── Robot.py              # Cython RPC bindings
│   └── README.md             # Package overview
│
├── utils/                # Common utilities
│   ├── config.py         # Config loading/abstraction
│   ├── logger.py         # Centralized, JSON-capable logger
│   ├── io.py             # Camera calibration I/O
│   ├── geometry.py       # Math helpers
│   └── README.md         # Package overview
│
├── vision/               # Vision, cloud, and camera utils
│   ├── opencv_utils.py        # OpenCV helper class
│   ├── realsense.py           # RealSense camera wrapper
│   ├── pointcloud.py          # PointCloudGenerator class and utilities
│   ├── tools.py               # Camera and cloud helper routines
│   ├── transform.py           # 3D transformation utilities
│   └── README.md              # Explanation of transform chain
│
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
* Restart robot connection:

  ```bash
  robot-restart
  ```

### 4. Build & Install

Use `pyproject.toml` together with [uv](https://github.com/astral-sh/uv) for reproducible environments/builds:

```bash
uv venv .venv -p 3.12
uv pip install -e .
```

All dependencies are defined in `pyproject.toml`. To create a traditional
`requirements.txt` snapshot run `uv pip freeze > requirements.txt`.

---

## 📑 Documentation

### Configuration (`config.yaml`)

* `robot:` — IP, tool/user frame, velocity, emergency delay
* `vision:` — RealSense stream parameters, cloud parameters
* `logging:` — log directory, level, JSON output
* `cloud:` — point cloud settings (output dir)
* `cloud_pipeline:` — dataset paths, depth scale, ROI limits
* `gpt:` — API keys and endpoints for path generation
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
* CLI scripts provided via entry points (`pointcloud-capture`, `pointcloud-transform`, `pointcloud-view`)

**See [vision/README.md](./vision/README.md) for detailed usage, coordinate system details, and all math for 3D transforms.**

**Typical workflow:**

1. Capture depth + color → generate cloud (`pointcloud-capture`)
2. (Optional) Filter, merge, or transform clouds (robot <-> camera <-> world)
3. Save or visualize result (`pointcloud-view`)

### CLI Tools

Entry points defined in `pyproject.toml` expose the common workflows:
`poses-saver`, `path-runner`, `charuco-calib`, etc. The underlying logic lives
within the respective packages.

### Extensibility/Testing
* Run tests with `pytest`.

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
