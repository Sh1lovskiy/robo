# Robotics Vision & Calibration Suite

Minimal toolkit for controlling a robot, calibrating its camera and executing RGB‑D scanning tasks.

## Installation

```bash
uv venv .venv -p 3.12
uv pip install -e .
```

## Project Structure

- `robot/` – robot controller and RPC wrapper
- `calibration/` – hand‑eye and intrinsic calibration utilities
- `robot_scan/` – RGB‑D capture, graph extraction and motion pipeline
- `utils/` – logging, settings and helpers

## Usage

Run a scan:

```bash
python -m robot_scan.main --handeye path/to/handeye.npy
```

Calibrate the camera:

```bash
python -m calibration.run --help
```

Visualize a saved graph:

```bash
python -m robot_scan.visualization path/to/graph.npy
```

## Development

Activate the environment and open notebooks or save data using paths defined in `utils/settings.py`. See [docs/handeye.md](docs/handeye.md) for a detailed explanation of the math and processing pipeline.
