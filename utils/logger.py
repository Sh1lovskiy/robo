"""Logging helpers built on top of loguru."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, TypeVar, cast

from loguru import logger as _logger
from loguru._logger import Logger as LoguruLogger
from tqdm.auto import tqdm

LoggerType = LoguruLogger

T = TypeVar("T")

_is_configured = False
_log_dir = Path("logs")
_log_file = None
PROGRESS_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"


class Logger:
    """Project-wide logger wrapper using loguru."""

    @staticmethod
    def _configure(level: str, json_format: bool) -> None:
        """Configure log sinks on first use."""
        global _is_configured, _log_file
        _logger.remove()
        os.makedirs(_log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _log_file = _log_dir / f"{timestamp}.log.json"
        LOG_FORMAT = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> "
            "[<level>{level: <5}</level>] "
            "[<cyan>{name}</cyan>:<cyan>{line}</cyan>]"
            "[PID:{process}] - <level>{message}</level>"
        )
        _logger.add(sys.stdout, level=level, serialize=False, format=LOG_FORMAT)
        _logger.add(
            _log_file,
            level=level,
            serialize=json_format,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} [{level}] [{name}:{line}][PID:{process}] - {message}",
        )
        _is_configured = True

    def get_logger(
        name: str, level: str = "INFO", json_format: bool = True
    ) -> LoguruLogger:
        """Return a configured loguru logger bound to ``name``."""
        global _is_configured
        if not _is_configured:
            Logger._configure(level, json_format)
        return _logger.bind(module=name)

    @staticmethod
    def progress(
        iterable: Iterable[T],
        desc: str | None = None,
        total: int | None = None,
    ) -> Iterable[T]:
        """Return a tqdm iterator with unified style."""
        return cast(
            Iterable[T],
            tqdm(
                iterable,
                desc=desc,
                total=total,
                leave=False,
                bar_format=PROGRESS_BAR_FORMAT,
            ),
        )

    @staticmethod
    def configure_root_logger(level: str = "WARNING") -> None:
        """Configure the root logger for third-party libraries."""
        global _is_configured
        _logger.remove()
        _logger.add(sys.stdout, level=level)
        _is_configured = True

    @staticmethod
    def configure(
        level: str = "INFO", log_dir: str | Path = "logs", json_format: bool = True
    ) -> None:
        """Manually configure the logger with given settings."""
        global _log_dir
        _log_dir = Path(log_dir)
        Logger._configure(level, json_format)
