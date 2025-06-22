"""Generic CLI dispatcher utilities."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Optional


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
        parser = argparse.ArgumentParser(description=self.description)
        subparsers = parser.add_subparsers(dest="command")
        for cmd in self.commands:
            sp = subparsers.add_parser(cmd.name, help=cmd.help)
            if cmd.add_arguments:
                cmd.add_arguments(sp)
            sp.set_defaults(func=cmd.handler)
        return parser

    def run(self, args: Optional[list[str]] = None) -> None:
        parser = self._build_parser()
        ns = parser.parse_args(args)
        if hasattr(ns, "func"):
            ns.func(ns)
        else:
            parser.print_help()
