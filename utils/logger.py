"""Logging helpers built on top of loguru."""

from __future__ import annotations

import os
import sys
import threading
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
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green>"
            "[<level>{level}</level>]"
            "[<cyan>{file}</cyan>:<cyan>{line}</cyan>]"
            # "[PID:{process}] - <level>{message}</level>"
            "<level>{message}</level>"
        )
        _logger.add(sys.stdout, level=level, serialize=False, format=LOG_FORMAT)
        _logger.add(
            _log_file,
            level=level,
            serialize=json_format,
            format="{time:YYYY-MM-DD HH:mm:ss}[{level}][{file}:{line}]{message}",
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


class CaptureStderrToLogger:
    """
    Context manager: перехватывает C/C++ stderr (fd=2) и пишет всё в твой логгер.
    """

    def __init__(self, logger):
        self.logger = logger
        self.pipe_read = None
        self.pipe_write = None
        self.thread = None
        self._old_stderr_fd = None

    def _reader(self):
        with os.fdopen(self.pipe_read, "r", errors="replace") as f:
            for line in f:
                line = line.rstrip()
                if line:
                    self.logger.warning(f"[STDERR] {line}")

    def __enter__(self):
        # Save original stderr fd
        self._old_stderr_fd = os.dup(2)
        # Create pipe
        self.pipe_read, self.pipe_write = os.pipe()
        # Redirect stderr to pipe
        os.dup2(self.pipe_write, 2)
        # Start thread that reads from pipe and writes to logger
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Flush and restore stderr
        sys.stderr.flush()
        os.dup2(self._old_stderr_fd, 2)
        os.close(self.pipe_write)
        os.close(self._old_stderr_fd)
        # Let thread finish reading
        if self.thread:
            self.thread.join(timeout=0.2)
