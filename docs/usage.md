# Usage Guide

## Calibration Workflow

Calibration utilities operate on captured RGB/D frames and recorded robot poses. A typical session:

1. **Capture data:** `python -m calibration.runner_camera --count 20`
2. **Move the robot on a grid:** `python -m calibration.runner_robot`
3. **Run hand–eye calibration:** `python handeye_charuco.py`

Results are stored under `calibration/results/`.

## Robot CLI

The `robot` package exposes a small command line interface for common tasks:

```bash
python -m robot.cli record --ip <robot_ip> --captures_dir ./captures
```

Arguments are defined in the `robot.cli` module:

```python
    def _add_record_args(parser: argparse.ArgumentParser) -> None:
        """Arguments for the ``record`` subcommand."""
        parser.add_argument("--ip", default=robot.ip, help="Robot IP")
        parser.add_argument(
            "--captures_dir",
            default=str(paths.CAPTURES_DIR),
            help="Directory for saved poses",
        )
        parser.add_argument("--drag", action="store_true", help="Enable drag teaching mode")
        parser.add_argument(
            "--db_path", default="robot_data.lmdb", help="LMDB database path"
        )
        parser.add_argument("--use-db", action="store_true", help="Use LMDB storage")
```

Use `robot.cli restart` to reconnect to the controller if the connection drops.
