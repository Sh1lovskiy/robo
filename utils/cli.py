"""Generic CLI dispatcher utilities."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from utils.error_tracker import ErrorTracker
from utils.logger import Logger


@dataclass
class Command:
    """Represents a single CLI command."""

    name: str
    handler: Callable[[argparse.Namespace], None]
    add_arguments: Optional[Callable[[argparse.ArgumentParser], None]] = None
    help: str | None = None


@dataclass
class CommandDispatcher:
    """Register and execute subcommands using ``argparse``."""

    description: str
    commands: Iterable[Command] = field(default_factory=list)

    def _build_parser(self) -> argparse.ArgumentParser:
        """Create the :class:`argparse.ArgumentParser` for this dispatcher."""

        parser = argparse.ArgumentParser(description=self.description)
        subparsers = parser.add_subparsers(dest="command")
        for cmd in self.commands:
            sp = subparsers.add_parser(cmd.name, help=cmd.help)
            if cmd.add_arguments:
                cmd.add_arguments(sp)
            sp.set_defaults(func=cmd.handler)
        return parser

    def run(
        self,
        args: Optional[list[str]] = None,
        *,
        logger: Optional[Logger] = None,
        track_exceptions: bool = True,
    ) -> None:
        """
        Parse arguments and dispatch the selected command.

        Any ``SystemExit`` raised by ``argparse`` is logged before re-raising so
        callers can track CLI usage issues across the project. Optionally
        installs the global :class:`ErrorTracker` for uncaught exceptions.
        """

        if logger is None:
            logger = Logger.get_logger("utils.cli")

        if track_exceptions:
            ErrorTracker.install_excepthook()
            ErrorTracker.install_signal_handlers()
            ErrorTracker.install_keyboard_listener()

        parser = self._build_parser()

        try:
            ns = parser.parse_args(args)
        except SystemExit as exc:  # argparse calls sys.exit() on error
            logger.error(f"Argument parsing failed: {exc}")
            raise

        if hasattr(ns, "func"):
            ns.func(ns)
        else:
            parser.print_help()
