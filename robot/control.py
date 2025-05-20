"""
Robot Control Module

This module provides functionality for controlling a robot via an SDK.
It includes solutions for common issues such as handling socket errors
and terminating processes that block communication.

Troubleshooting:
----------------
1. **SDKFailed to read real-time data of the robot [WinError 10038]:**
   - This error typically occurs when there is an attempt to perform an operation
     on an invalid socket (e.g., the connection was closed improperly).

   Solution:
   - Terminate all Python processes to release sockets and resources:
     ```
     taskkill /IM python.exe /F
     ```

2. **Default Robot Connection Details:**
   - The default IP address and port for the robot are:
     ```
     192.168.58.2:9999
     ```

3. **Identify and Kill Processes Using a Specific Port:**
   - To check which process is using port 9999:
     ```
     netstat -ano | findstr 9999
     ```
   - Locate the PID (Process ID) from the output.

   - To terminate the process with the identified PID:
     ```
     taskkill /PID <PID> /F
     ```
     Replace `<PID>` with the actual Process ID.

4. **Restart the System (Optional):**
   - If the above steps do not resolve the issue, restarting the system ensures
     all resources are released.

Usage:
------
Run the script to initialize robot control. Use hotkeys or physical buttons
to manage robot operations.

Hotkeys:
- `ESC`: Emergency stop
- `H`: Return to home (init) position
- `C`: Resume motion
"""

import threading
import time
import sys
import keyboard
from typing import List
from fairino.Robot import RPC
from misc.logger import Logger
from config.constants import (
    DEFAULT_IP,
    TOOL_ID,
    USER_FRAME_ID,
    NORMAL_VELOCITY,
    EMERGENCY_DELAY,
)


class RobotController:
    """
    Integrated robot control with direct method access.

    Provides methods for movement, safety functions, and system control.
    Ensures proper cleanup of resources upon shutdown.
    """

    def __init__(self, ip_address: str = DEFAULT_IP):
        """
        Initialize the robot controller.

        Args:
            ip_address (str): IP address of the robot (default: 192.168.58.2).
        """
        self.logger = Logger.get_logger("robot_controller", json_format=True)
        self.robot = RPC(ip=ip_address)
        self.connected = bool(self.robot.is_conect)
        self.initial_pose = None
        self.is_active = False

        if not self.connected:
            self.logger.error(f"Connection failed to {ip_address}")
            sys.exit(1)

    def initialize(self) -> bool:
        """
        Initialize the robot control system.

        Returns:
            bool: True if initialization succeeds, False otherwise.
        """
        if not self.connected:
            return False

        self._setup_key_controls()
        self._record_home_position()
        self.is_active = True
        threading.Thread(target=self._control_loop, daemon=True).start()
        return True

    def _setup_key_controls(self):
        """Configure keyboard bindings."""
        keyboard.add_hotkey("esc", self.emergency_stop)
        keyboard.add_hotkey("h", self.return_to_home)
        keyboard.add_hotkey("c", self.resume_motion)
        self.logger.info("Keyboard controls configured.")

    def _record_home_position(self):
        """Cache the current position as the home position."""
        pose_data = self.robot.GetActualTCPPose()
        if pose_data[0] == 0:
            self.initial_pose = pose_data[1]
            self.logger.info(f"Home position recorded: {self.initial_pose}")
        else:
            self.logger.error("Failed to record home position.")

    def get_current_pose(self) -> List[float]:
        """Retrieve current TCP pose."""
        try:
            if not self.connected:
                self.logger.error("Coordinate request on disconnected robot")
                return None

            pose_data = self.robot.GetActualTCPPose()
            if pose_data[0] != 0:
                self.logger.error(f"Pose request failed with code {pose_data[0]}")
                return None

            self.logger.debug(f"Current pose: {pose_data[1]}")
            return pose_data[1]
        except Exception as e:
            self.logger.error(f"Socket error while retrieving pose: {e}")
            return None

    # Core Motion Commands
    def move_joints(self, positions: List[float]) -> bool:
        """
        Perform direct joint movement.

        Args:
            positions (List[float]): Target joint positions.

        Returns:
            bool: True if the command succeeds, False otherwise.
        """
        if not self.connected:
            self.logger.error("Robot is not connected.")
            return False

        result = self.robot.MoveJ(
            desc_pos=positions, tool=TOOL_ID, user=USER_FRAME_ID, vel=NORMAL_VELOCITY
        )
        return self._check_result(result, "Joint movement")

    def move_linear(self, coordinates: List[float]) -> bool:
        """
        Perform straight-line Cartesian movement.

        Args:
            coordinates (List[float]): Target Cartesian coordinates.

        Returns:
            bool: True if the command succeeds, False otherwise.
        """
        if not self.connected:
            self.logger.error("Robot is not connected.")
            return False

        result = self.robot.MoveL(
            desc_pos=coordinates, tool=TOOL_ID, user=USER_FRAME_ID, vel=NORMAL_VELOCITY
        )
        return self._check_result(result, "Linear movement")

    def enable_drag_teach_mode(self, enable: bool) -> bool:
        """
        Enable or disable drag teaching mode.

        This method handles both enabling and disabling drag teaching mode.
        According to the robot documentation, it requires:
        1. Setting the robot to Manual mode (1) or Automatic mode (0)
        2. Enabling (1) or disabling (0) the drag teaching switch

        Args:
            enable (bool): True to enable drag teaching mode, False to disable.

        Returns:
            bool: True if the operation succeeds, False otherwise.
        """
        if not self.connected:
            self.logger.error("Robot is not connected.")
            return False

        # Step 1: Set robot to Manual mode (1) or Automatic mode (0)
        mode_state = 1 if enable else 0
        mode_result = self.robot.Mode(mode_state)
        if mode_result != 0:
            self.logger.error(
                f"Failed to set robot mode to {'Manual' if enable else 'Automatic'}. Error code: {mode_result}"
            )
            return False

        # Step 2: Enable (1) or disable (0) drag teaching mode
        drag_state = 1 if enable else 0
        drag_result = self.robot.DragTeachSwitch(drag_state)
        if drag_result != 0:
            self.logger.error(
                f"Failed to {'enable' if enable else 'disable'} drag teaching mode. Error code: {drag_result}"
            )
            return False

        self.logger.info(
            f"Drag teaching mode {'enabled' if enable else 'disabled'} successfully."
        )
        return True

    def is_in_drag_teach_mode(self) -> tuple[bool, bool]:
        """
        Check if the robot is currently in drag teaching mode.

        Returns:
            tuple[bool, bool]:
                - First value: True if the query succeeds, False otherwise.
                - Second value: True if the robot is in drag teaching mode, False otherwise.
        """
        if not self.connected:
            self.logger.error("Robot is not connected.")
            return False, False

        result = self.robot.IsInDragTeach()
        if isinstance(result, tuple) and len(result) == 2:
            error_code, status = result
            if error_code == 0:
                self.logger.info(
                    f"Robot is {'in' if status == 1 else 'not in'} drag teaching mode."
                )
                return True, status == 1
            else:
                self.logger.error(
                    f"Failed to check drag teaching mode status. Error code: {error_code}"
                )
                return False, False
        else:
            self.logger.error(
                f"Failed to check drag teaching mode status. Invalid response format: {result}"
            )
            return False, False

    def _check_result(self, code: int, operation: str) -> bool:
        """
        Validate the result of a robot command.

        Args:
            code (int): Error code returned by the robot API.
            operation (str): Description of the operation being validated.

        Returns:
            bool: True if the operation succeeds, False otherwise.
        """
        if code != 0:
            self.logger.error(f"{operation} failed with code {code}")
            return False
        return True

    # Safety Functions
    def emergency_stop(self):
        """Immediate motion halt."""
        self.logger.warning("EMERGENCY STOP TRIGGERED")
        self.robot.StopMotion()

    def return_to_home(self):
        """Return to the initial position."""
        if not self.initial_pose:
            self.logger.error("No home position recorded.")
            return

        self.robot.StopMotion()
        time.sleep(EMERGENCY_DELAY)
        self.move_linear(self.initial_pose)

    def resume_motion(self):
        """Continue after pause."""
        self.robot.ResumeMotion()
        self.logger.info("Resuming motion.")

    # System Control
    def _control_loop(self):
        """Main control thread for periodic tasks."""
        while self.is_active:
            time.sleep(0.1)

    def shutdown(self):
        """Gracefully shut down the robot controller."""
        self.logger.info("Shutting down robot controller...")
        self.is_active = False
        keyboard.unhook_all()
        try:
            if self.connected:
                self.robot.RobotEnable(0)  # Disable the robot
                self.robot.CloseRPC()  # Close the RPC connection
                self.logger.info("Robot disabled and RPC connection closed.")

            self.logger.info("Robot controller shutdown complete.")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
        finally:
            self.logger.info("System shutdown complete.")


if __name__ == "__main__":
    try:
        bot = RobotController()
        if bot.connected and bot.initialize():
            print("Robot initialized. Press 'Q' to quit.")
            while not keyboard.is_pressed("q"):
                time.sleep(0.1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        bot.shutdown()
