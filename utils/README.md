# utils package

Shared utilities for configuration, logging, CLI handling and math helpers. These modules are referenced throughout the project to avoid code duplication. They are intentionally lightweight so that higher level packages depend only on stable abstractions.

- `config.py` - Global config loader. Reads `config.yaml` once and provides accessors for nested keys.
- `logger.py` - Configurable logger using the standard library plus JSON formatting. It exposes decorators for automatic function tracing.
- `error_tracker.py` - Global exception hook and signal handlers with optional hotkeys.
- `cli.py` - Small `CommandDispatcher` used by all CLI entry points.
- `keyboard.py` - Global hotkey listener used by `ErrorTracker`.
- `io.py` - Camera parameter read/write helpers.
- `geometry.py` - Small math helpers, e.g. Euler angle conversions and homogeneous transform builders.

Using these helpers keeps each module focused on a single task.
