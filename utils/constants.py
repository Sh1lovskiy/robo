# utils/constants.py
"""Common constants used across the project."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"
DEFAULT_LOG_DIR = REPO_ROOT / "logs"
DEFAULT_CLOUD_DIR = REPO_ROOT / "clouds"

