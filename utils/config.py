# utils/config.py
"""Configuration loader with YAML backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, cast

import yaml
from omegaconf import OmegaConf

from utils.logger import Logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = PROJECT_ROOT / "conf"
DEFAULT_CONFIG_PATH = CONF_DIR / "app.yaml"

DEFAULT_YAML = """
robot:
  ip: "192.168.58.2"
  tool_id: 0
  user_frame_id: 0
  velocity: 20.0
  emergency_delay: 0.5
path_saver:
  captures_dir: cloud
  path_file: cloud/poses.json
vision:
  realsense_width: 640
  realsense_height: 480
  realsense_fps: 30
logging:
  log_dir: "./logs"
  level: "INFO"
  json: true
cloud:
  output_dir: clouds
charuco:
  squares_x: 5
  squares_y: 7
  square_length: 0.035
  marker_length: 0.026
  aruco_dict: 5X5_100
  calib_output_dir: calibration/results2
  xml_file: charuco_cam.xml
  txt_file: charuco_cam.txt
  images_dir: cloud
handeye:
  images_dir: cloud
  calib_output_dir: calibration/results
  npz_file: handeye.npz
  txt_file: handeye.txt
  method: all
"""

if not DEFAULT_CONFIG_PATH.exists():
    CONF_DIR.mkdir(exist_ok=True)
    DEFAULT_CONFIG_PATH.write_text(DEFAULT_YAML)


class ConfigLoader:
    """Strategy interface for config loading."""

    def load(self, filename: str) -> Dict[str, Any]:
        raise NotImplementedError


class YamlConfigLoader(ConfigLoader):
    def load(self, filename: str) -> Dict[str, Any]:
        cfg = OmegaConf.load(filename)
        return cast(Dict[str, Any], OmegaConf.to_container(cfg, resolve=True))


class Config:
    _data: Dict[str, Any] | None = None
    _loader = YamlConfigLoader()
    _logger = Logger.get_logger("utils.config")

    @classmethod
    def load(
        cls, filename: Path | str = DEFAULT_CONFIG_PATH, force_reload: bool = False
    ) -> None:
        """Load configuration from ``filename`` unless already loaded."""

        if cls._data is not None and not force_reload:
            return

        try:
            cls._data = cls._loader.load(filename)
            cls._logger.info(f"Config loaded from {filename}")
            logging_cfg = cls._data.get("logging", {})
            Logger.configure(
                level=logging_cfg.get("level", "INFO"),
                log_dir=logging_cfg.get("log_dir", "logs"),
                json_format=logging_cfg.get("json", True),
            )
        except Exception as e:
            cls._logger.error(f"Failed to load config: {e}")
            raise

    @classmethod
    def get(cls, path: str, default: Any | None = None) -> Any:
        if cls._data is None:
            cls.load()
        value = cls._data
        for key in path.split("."):
            if not isinstance(value, dict):
                cls._logger.warning(f"Key {key} not found in path {path}")
                return default
            value = value.get(key, None)
            if value is None:
                cls._logger.warning(f"Key {key} not found in path {path}")
                return default
        return value

    @classmethod
    def set_loader(cls, loader: ConfigLoader) -> None:
        cls._loader = loader
        cls._logger.info(f"Config loader set to {loader.__class__.__name__}")
