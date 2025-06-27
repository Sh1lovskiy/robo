# utils/config.py
"""Configuration loader with YAML backend."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, cast
from omegaconf import OmegaConf
from utils.logger import Logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = PROJECT_ROOT / "conf"
DEFAULT_CONFIG_PATH = CONF_DIR / "app.yaml"


class ConfigLoader:
    """Strategy interface for config loading."""

    def load(self, filename: str) -> Dict[str, Any]:
        raise NotImplementedError


class YamlConfigLoader(ConfigLoader):
    def load(self, filename: str) -> Dict[str, Any]:
        """Load YAML file and return plain ``dict`` data."""

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
        """Retrieve value from dotted ``path`` or return ``default``."""
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
        """Replace the config loader strategy (useful for testing)."""

        cls._loader = loader
        cls._logger.info(f"Config loader set to {loader.__class__.__name__}")
