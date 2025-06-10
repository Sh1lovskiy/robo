# utils/config.py

import yaml
from pathlib import Path
from utils.logger import Logger


class ConfigLoader:
    """Strategy interface for config loading."""

    def load(self, filename):
        raise NotImplementedError


class YamlConfigLoader(ConfigLoader):
    def load(self, filename):
        with open(filename, "r") as f:
            return yaml.safe_load(f)


class Config:
    _data = None
    _loader = YamlConfigLoader()
    _logger = Logger.get_logger("utils.config")

    @classmethod
    def load(cls, filename="config.yaml"):
        try:
            cls._data = cls._loader.load(filename)
            cls._logger.info(f"Config loaded from {filename}")
        except Exception as e:
            cls._logger.error(f"Failed to load config: {e}")
            raise

    @classmethod
    def get(cls, path, default=None):
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
    def set_loader(cls, loader: ConfigLoader):
        cls._loader = loader
        cls._logger.info(f"Config loader set to {loader.__class__.__name__}")
