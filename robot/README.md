# robot package

Provides :class:`RobotController` and workflow helpers for pose recording and trajectory execution.  The Cython RPC bindings are wrapped in a clean Python API so that implementations can be swapped or mocked for testing.

- `controller.py` - `RobotController` wrapping the Cython `Robot` RPC bindings. Provides `move_*` helpers, restart logic and detailed logging.
- `workflows.py` - Pose recorder and path runner implementations.  These scripts orchestrate the controller and file I/O while remaining open for extension via composition.
- `workflows.py` also exposes a CLI via `CommandDispatcher`.
- `marker.py` - Simple marker representation used to store named poses or features.
- `Robot.py` - Cython RPC bindings (do not edit)

Dependency inversion allows injecting a different controller (e.g., a simulator) without changing consumers. This decouples robot specifics from business logic, illustrating the Open/Closed Principle. All classes keep a single responsibility for clean, testable code.

