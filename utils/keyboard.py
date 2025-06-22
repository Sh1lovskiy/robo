from __future__ import annotations

import sys
from typing import Callable, Dict, List, Optional

from pynput import keyboard
from utils.logger import Logger

KeyAction = Callable[[], None]


class GlobalKeyListener:
    """Listen for global hotkeys and suppress terminal output."""

    def __init__(self, hotkeys: Dict[str, KeyAction], *, suppress: bool = True) -> None:
        self.logger = Logger.get_logger("utils.keyboard")
        self.hotkeys = hotkeys
        self.listener = keyboard.GlobalHotKeys(hotkeys, suppress=suppress)
        self.listener.daemon = True

    def start(self) -> None:
        """Start listening for configured hotkeys."""
        self.listener.start()
        self.logger.debug("GlobalKeyListener started with keys: %s", list(self.hotkeys))

    def stop(self) -> None:
        """Stop the hotkey listener."""
        try:
            self.listener.stop()
        finally:
            self.logger.debug("GlobalKeyListener stopped")


class TerminalEchoSuppressor:
    """Disable terminal echo to hide typed hotkeys."""

    def __init__(self) -> None:
        self.logger = Logger.get_logger("utils.terminal")
        self.fd = sys.stdin.fileno()
        self.enabled = False
        self._orig_attrs: Optional[List[int]] = None

    def start(self) -> None:
        """Disable echo if running in a TTY."""
        if self.enabled or not sys.stdin.isatty():
            return
        try:
            import termios

            self._orig_attrs = termios.tcgetattr(self.fd)
            new_attrs = termios.tcgetattr(self.fd)
            new_attrs[3] &= ~(termios.ECHO | termios.ICANON)
            termios.tcsetattr(self.fd, termios.TCSADRAIN, new_attrs)
            self.enabled = True
            self.logger.debug("Terminal echo disabled")
        except Exception as e:
            self.logger.error("Failed to disable terminal echo: %s", e)

    def stop(self) -> None:
        """Restore echo settings."""
        if not self.enabled or self._orig_attrs is None:
            return
        try:
            import termios

            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._orig_attrs)
        finally:
            self.enabled = False
            self.logger.debug("Terminal echo restored")
