# robot/controller.py

"""High-level robot control interface.

``RobotController`` operates on a :class:`RobotInterface` allowing different
communication backends.  The default adapter wraps the Fairino SDK RPC but
tests may inject a mock implementation.
"""

from __future__ import annotations
import time

from typing import List, Optional, Union, cast

from utils.logger import Logger, LoggerType
from utils.error_tracker import ErrorTracker

from robot.interfaces import RobotInterface, FairinoRPC
from utils.settings import RobotSettings


class RobotController:
    """
    High-level robot controller for movement, pose, and state management.
    Uses :mod:`utils.settings` for defaults.
    """

    def __init__(
        self,
        cfg: RobotSettings = RobotSettings(),
        robot: Optional[Union[str, RobotInterface]] = None,
        logger: Optional[LoggerType] = None,
    ) -> None:
        """Initialize the robot controller."""

        self.cfg = cfg
        self.ip_address = cfg.ip
        self.tool_id = cfg.tool_id
        self.user_frame_id = cfg.user_frame_id
        self.velocity = cfg.velocity
        self.logger = logger or Logger.get_logger("robot.controller")
        self.initial_pose: List[float] | None = None

        self.robot = self._init_robot(robot)

        ErrorTracker.register_cleanup(self.shutdown)

        self.logger.info(f"RobotController initialized with IP {self.ip_address}")

    def _init_robot(
        self, robot: Optional[Union[str, RobotInterface]]
    ) -> RobotInterface:
        """Resolve robot communication interface from input."""
        if robot is None:
            self.logger.info("Creating default RPC adapter from config IP")
            return FairinoRPC(ip=self.ip_address)
        if isinstance(robot, str):
            self.logger.info("Creating RPC adapter from provided IP")
            return FairinoRPC(ip=robot)
        if isinstance(robot, RobotInterface):
            self.logger.info("Using provided robot interface instance")
            return robot
        raise TypeError(f"Invalid robot argument: {type(robot)}")

    def connect(self) -> bool:
        """
        Check connection to the robot.

        Returns:
            bool: True if connected, False otherwise.
        """
        if not self.robot.is_connected():
            self.logger.error(f"Cannot connect to robot at {self.ip_address}")
            return False
        self.logger.info("Robot connected")
        return True

    def get_tcp_pose(self) -> Optional[List[float]]:
        """
        Get current TCP pose.

        Returns:
            Optional[List[float]]: [x, y, z, Rx, Ry, Rz] or None on error.
        """
        res = self.robot.GetActualTCPPose()
        if res[0] == 0:
            pose = cast(List[float], res[1])
            self.logger.debug(f"Current pose: {pose}")
            return pose
        self.logger.error(f"GetActualTCPPose failed with code {res[0]}")
        return None

    def move_joints(self, joints: List[float]) -> bool:
        """
        Move robot to specified joint angles.

        Args:
            joints (List[float]): Target joint positions (degrees).

        Returns:
            bool: True on success, False on failure.
        """
        code = self.robot.MoveJ(
            joint_pos=joints,
            tool=self.tool_id,
            user=self.user_frame_id,
            vel=self.velocity,
        )
        if code != 0:
            self.logger.error(f"MoveJ failed with code {code}")
            return False
        self.logger.info(f"MoveJ success: {joints}")
        return True

    def move_linear(self, pose: List[float]) -> bool:
        """
        Move robot in a straight line to the specified TCP pose.

        Args:
            pose (List[float]): Target TCP pose [x, y, z, Rx, Ry, Rz].

        Returns:
            bool: True on success, False on failure.
        """
        code = self.robot.MoveL(
            desc_pos=pose, tool=self.tool_id, user=self.user_frame_id, vel=self.velocity
        )
        if code != 0:
            self.logger.error(f"MoveL failed with code {code}")
            return False
        self.logger.info(f"MoveL success: {pose}")
        return True

    def record_home(self) -> None:
        """
        Record current pose as 'home'.
        """
        pose = self.get_tcp_pose()
        if pose is None:
            self.logger.error("Failed to record home position")
            raise RuntimeError("Failed to record home position")
        self.initial_pose = pose
        self.logger.info(f"Home position recorded: {pose}")

    def return_to_home(self) -> None:
        """
        Move robot to previously recorded 'home' pose.
        """
        if self.initial_pose is not None:
            self.move_linear(self.initial_pose)
            self.logger.info("Returned to home position")
        else:
            self.logger.warning("Home position not recorded")

    def shutdown(self) -> None:
        """
        Disable and disconnect robot.
        """
        self.disable()
        self.robot.CloseRPC()
        self.logger.info("Robot shutdown complete")

    def enable(self) -> None:
        """Enable robot motors."""
        self.robot.RobotEnable(1)
        self.logger.info("Robot enabled")

    def disable(self) -> None:
        """Disable robot motors."""
        self.robot.RobotEnable(0)
        self.logger.info("Robot disabled")

    # --- Additional SOLID helpers ---
    def wait_motion_done(self, timeout_sec: float = 20) -> bool:
        """
        Wait until robot finishes motion (with timeout).

        Args:
            timeout_sec (float): Maximum time to wait.

        Returns:
            bool: True if motion finished, False if timeout.
        """

        start = time.time()
        while time.time() - start < timeout_sec:
            code, done = self.robot.GetRobotMotionDone()
            if code == 0 and done == 1:
                return True
            time.sleep(0.05)
        self.logger.warning("Wait for motion done: timeout")
        return False

    def get_joints(self) -> Optional[List[float]]:
        """
        Get current joint positions.

        Returns:
            Optional[List[float]]: Joint angles in degrees or None.
        """
        res = self.robot.GetActualJointPosDegree()
        if res[0] == 0:
            return cast(List[float], res[1])
        self.logger.error("GetActualJointPosDegree failed")
        return None

    def restart(
        self,
        ip_address: str | None = None,
        *,
        delay: float | None = None,
        attempts: int = 3,
    ) -> bool:
        """Restart the controller.

        Disable motors, close RPC, and attempt to reconnect ``attempts`` times
        (waiting ``delay`` seconds between tries). ``ip_address`` overrides the
        configured IP.

        Returns ``True`` on success.
        """

        delay = delay if delay is not None else self.cfg.restart_delay
        ip = ip_address or self.ip_address
        self.logger.info(
            f"Restarting robot at {ip} with delay {delay}s and {attempts} attempts"
        )

        if self.robot.RobotEnable(0) != 0:
            self.logger.error("Disable failed")
            return False
        self.logger.info("Robot disabled")

        self.robot.CloseRPC()
        self.logger.debug("RPC connection closed")

        for attempt in range(attempts):
            time.sleep(delay)
            self.logger.info(f"Reconnect attempt {attempt + 1} of {attempts}")
            self.robot = self._init_robot(ip)
            if not self.connect():
                self.logger.warning("Reconnect failed")
                continue
            if self.robot.RobotEnable(1) != 0:
                self.logger.error("Enable failed")
                return False
            safety_code = self.robot.GetSafetyCode()
            if safety_code != 0:
                self.logger.error(f"Safety state prevents operation: {safety_code}")
                return False
            self.logger.info("Robot restart successful")
            return True

        self.logger.error("All reconnect attempts failed")
        return False
