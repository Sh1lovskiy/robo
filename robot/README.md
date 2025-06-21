# robot package

High-level controller and helper workflows for the robot. The code abstracts the low-level RPC interface provided by `Robot.py` into a Pythonic API that can be mocked or replaced.  Each class is designed to obey the Single Responsibility Principle: the controller only exposes motion commands and state queries, while workflow helpers deal with user interaction and data storage.

- `controller.py` - `RobotController` wrapping the Cython `Robot` RPC bindings. It provides methods such as `move_linear` and `get_current_pose` and logs all calls for traceability.
- `workflows.py` - Pose recorder and path runner implementations.  These scripts orchestrate the controller and file I/O while remaining open for extension via composition.
- `marker.py` - Simple marker representation used to store named poses or features.
- `Robot.py` - Cython RPC bindings (do not edit)

Dependency inversion allows injecting a different controller (e.g., a simulator)without changing consumers. This decouples robot specifics from business logic, illustrating the Open/Closed Principle.

