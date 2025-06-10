# utils/logger.py

"""
Centralized logging system for the robot controller.
Provides logging utilities with support for console and file output,
JSON formatting, and function timing. Includes a silent logging decorator
to suppress function call and return logs.
"""

import os
import logging
import numpy as np
import sys
import json
import functools
from datetime import datetime
from typing import Optional, Union, Dict, Any
from utils.constants import LOG_DIR
import traceback
from pythonjsonlogger import jsonlogger


class Logger:
    """
    Centralized logging system for the robot controller.
    Preserves the original logger functionality with additional features.
    """

    _LOGGERS = {}

    DEFAULT_LOG_LEVEL = logging.INFO

    DEFAULT_FORMAT = (
        "%(asctime)s [%(name)s][%(filename)s:%(lineno)d][%(levelname)s] %(message)s"
    )

    @staticmethod
    def setup_logger(
        name: str,
        log_dir: str = LOG_DIR,
        level: int = DEFAULT_LOG_LEVEL,
        console_output: bool = True,
        file_output: bool = True,
        format_str: Optional[str] = None,
        json_format: bool = False,
        json_fields: Optional[Dict[str, Any]] = None,
    ) -> logging.Logger:
        """
        Setup a logger with the given parameters.

        Args:
            name: Name of the logger
            log_dir: Directory to store log files
            level: Logging level
            console_output: Whether to output logs to console
            file_output: Whether to output logs to file
            format_str: Format string for regular logging
            json_format: Whether to use JSON format for file output
            json_fields: Additional fields to add to each JSON log entry

        Returns:
            Configured logger
        """
        if name in Logger._LOGGERS:
            return Logger._LOGGERS[name]

        if format_str is None:
            format_str = Logger.DEFAULT_FORMAT

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False

        if logger.hasHandlers():
            logger.handlers.clear()

        console_formatter = logging.Formatter(format_str)

        if json_format:

            class CustomJsonFormatter(jsonlogger.JsonFormatter):
                def __init__(self, *args, **kwargs):
                    self.json_fields = kwargs.pop("json_fields", {}) or {}
                    super().__init__(*args, **kwargs)

                def add_fields(self, log_record, record, message_dict):
                    super().add_fields(log_record, record, message_dict)

                    for field, value in self.json_fields.items():
                        log_record[field] = value

                    if "asctime" in log_record:
                        log_record["timestamp"] = log_record.pop("asctime")

            json_format_str = "%(timestamp)s %(name)s %(filename)s %(lineno)d %(levelname)s %(message)s"
            file_formatter = CustomJsonFormatter(
                json_format_str, json_fields=json_fields
            )
        else:
            file_formatter = logging.Formatter(format_str)

        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        if file_output:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            extension = "json" if json_format else "log"
            log_file = os.path.join(log_dir, f"{name}_{timestamp}.{extension}")

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        if json_fields:
            setattr(logger, "_json_fields", json_fields)

        Logger._LOGGERS[name] = logger
        return logger

    @staticmethod
    def get_logger(
        name: str,
        log_dir: str = LOG_DIR,
        level: int = DEFAULT_LOG_LEVEL,
        console_output: bool = True,
        file_output: bool = True,
        json_format: bool = False,
        json_fields: Optional[Dict[str, Any]] = None,
    ) -> logging.Logger:
        """
        Get a logger by name. Creates a new logger if it doesn't exist.

        Args:
            name: Name of the logger
            log_dir: Directory to store log files
            level: Logging level
            console_output: Whether to output logs to console
            file_output: Whether to output logs to file
            json_format: Whether to use JSON format for file output
            json_fields: Additional fields to add to each JSON log entry

        Returns:
            Configured logger
        """
        if name in Logger._LOGGERS:
            return Logger._LOGGERS[name]

        return Logger.setup_logger(
            name,
            log_dir,
            level,
            console_output,
            file_output,
            json_format=json_format,
            json_fields=json_fields,
        )

    @staticmethod
    def configure_root_logger(level: int = logging.WARNING):
        """Configure the root logger for third-party libraries"""
        logging.basicConfig(
            level=level,
            format=Logger.DEFAULT_FORMAT,
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    @staticmethod
    def format_value(value):
        """
        Format non-integer values to 3 decimal places.
        Recursively formats lists, tuples, and numpy arrays.
        Skips logging for dictionaries and lists.
        """
        if isinstance(value, (dict, list)):
            return "<Omitted Object>"

        if isinstance(value, float):
            return round(value, 3)
        elif isinstance(value, (list, tuple)):
            return type(value)(Logger.format_value(v) for v in value)
        elif isinstance(value, np.ndarray):
            return np.round(value, 3).tolist()

        return value

    @staticmethod
    def log_data(
        logger: logging.Logger,
        level: int,
        message: str,
        *args,
        extra_fields: Optional[Dict[str, Any]] = None,
    ):
        """
        Log data with formatted non-integer values.

        Args:
            logger: Logger instance
            level: Logging level
            message: Message format string
            *args: Values for message formatting
            extra_fields: Additional JSON fields to include in this log entry
        """
        filtered_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                filtered_args.append(f"Array of shape {arg.shape}")
            else:
                filtered_args.append(Logger.format_value(arg))

        extra = {}
        if extra_fields:
            extra.update(extra_fields)

        if hasattr(logger, "_json_fields"):
            extra.update(getattr(logger, "_json_fields"))

        if extra:
            getattr(logger, logging.getLevelName(level).lower())(
                message, *filtered_args, extra=extra
            )
        else:
            getattr(logger, logging.getLevelName(level).lower())(
                message, *filtered_args
            )

    @staticmethod
    def log_json(logger: logging.Logger, level: int, **fields):
        """
        Log data directly as JSON fields.

        Args:
            logger: Logger instance
            level: Logging level
            **fields: Fields to include in the JSON log
        """
        formatted_fields = {k: Logger.format_value(v) for k, v in fields.items()}
        msg_parts = []
        for key, value in formatted_fields.items():
            if isinstance(value, (dict, list)):
                msg_parts.append(f"{key}={json.dumps(value)}")
            else:
                msg_parts.append(f"{key}={value}")

        message = ", ".join(msg_parts)

        extra = {}
        if hasattr(logger, "_json_fields"):
            extra.update(getattr(logger, "_json_fields"))
        extra.update(formatted_fields)
        getattr(logger, logging.getLevelName(level).lower())(message, extra=extra)

    @staticmethod
    def log_function(logger: logging.Logger, level=logging.INFO):
        """Decorator that logs function calls, arguments, returns, and exceptions."""

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                Logger.log_data(
                    logger,
                    level,
                    "Calling %s with args=%s, kwargs=%s",
                    func.__name__,
                    args,
                    kwargs,
                )
                try:
                    result = func(*args, **kwargs)
                    Logger.log_data(
                        logger, level, "Function %s returned %s", func.__name__, result
                    )
                    return result
                except Exception as e:
                    Logger.log_data(
                        logger,
                        logging.ERROR,
                        "Exception in %s: %s\n%s",
                        func.__name__,
                        str(e),
                        traceback.format_exc(),
                    )
                    raise

            return wrapper

        return decorator

    @staticmethod
    def silent_log_function(logger: logging.Logger, level=logging.INFO):
        """Decorator that logs only exceptions, suppressing function call and return logs."""

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    Logger.log_data(
                        logger,
                        logging.ERROR,
                        "Exception in %s: %s\n%s",
                        func.__name__,
                        str(e),
                        traceback.format_exc(),
                    )
                    raise

            return wrapper

        return decorator


class Timer:
    """Simple timer utility to measure elapsed time"""

    def __init__(self, name: str = None, logger: Union[logging.Logger, None] = None):
        self.name = name or "Timer"
        self.logger = logger
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

        if self.logger:
            elapsed = self.elapsed()

            if hasattr(self.logger, "_json_fields"):
                Logger.log_json(
                    self.logger, logging.INFO, operation=self.name, elapsed_time=elapsed
                )
            else:
                self.logger.info(f"{self.name} took {elapsed:.4f} seconds")

    def start(self):
        """Start the timer"""
        self.start_time = datetime.now()
        return self

    def stop(self):
        """Stop the timer"""
        self.end_time = datetime.now()
        return self

    def elapsed(self) -> float:
        """Get elapsed time in seconds"""
        if self.start_time is None:
            return 0.0

        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()

    def reset(self):
        """Reset the timer"""
        self.start_time = None
        self.end_time = None
        return self


if __name__ == "__main__":
    json_fields = {
        "app_version": "1.0.0",
        "environment": "development",
        "robot_id": "R2D2",
    }

    logger = Logger.setup_logger(
        "ExampleLogger", json_format=True, json_fields=json_fields
    )

    array = np.array([1.23456789, 2.3456789, 3.456789])
    float_value = 3.1415926535
    list_value = [0.123456, 0.987654]

    Logger.log_data(logger, logging.INFO, "Logging formatted data: %s", array)
    Logger.log_data(logger, logging.INFO, "Float value: %s", float_value)

    Logger.log_data(
        logger,
        logging.INFO,
        "List value: %s",
        list_value,
        extra_fields={"operation": "array_processing", "count": len(list_value)},
    )

    Logger.log_json(
        logger,
        logging.INFO,
        event_type="measurement",
        sensor_id="S123",
        values=[1.234, 5.678, 9.012],
        status="ok",
    )

    with Timer("DatabaseQuery", logger):
        import time

        time.sleep(0.5)
