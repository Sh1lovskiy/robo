# utils package

Shared utilities for configuration, logging, I/O and math helpers. These modules are referenced throughout the project to avoid code duplication. They are intentionally lightweight so that higher level packages depend only on stable abstractions.

- `config.py` - Global config loader. Reads `config.yaml` once and provides accessors for nested keys.
- `logger.py` - Configurable logger using the standard library plus JSON formatting. It exposes decorators for automatic function tracing.
- `io.py` - Camera parameter read/write helpers.
- `geometry.py` - Small math helpers, e.g. Euler angle conversions and homogeneous transform builders.

Using these helpers promotes reuse (Open Closed) and keeps each module focused on a single task.
