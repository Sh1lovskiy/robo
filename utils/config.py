# utils/config.py
"""Configuration loader with YAML backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from utils.logger import Logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class ConfigLoader:
    """Strategy interface for config loading."""

    def load(self, filename: str) -> Dict[str, Any]:
        raise NotImplementedError


class YamlConfigLoader(ConfigLoader):
    def load(self, filename: str) -> Dict[str, Any]:
        with open(filename, "r") as f:
            return yaml.safe_load(f)


class Config:
    _data: Dict[str, Any] | None = None
    _loader = YamlConfigLoader()
    _logger = Logger.get_logger("utils.config")

    @classmethod
    def load(cls, filename: Path | str = DEFAULT_CONFIG_PATH) -> None:
        try:
            cls._data = cls._loader.load(filename)
            cls._logger.info(f"Config loaded from {filename}")
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
