from __future__ import annotations
import time
from typing import List, Optional, Union, cast

from utils.logger import Logger, LoggerType
from utils.error_tracker import ErrorTracker
from robot.interfaces import RobotInterface, FairinoRPC
from utils.settings import robot as RobotSettings, RobotCfg


class RobotController:
    """
    High-level robot controller that wraps the robot SDK.
    Provides safe motion commands, logging, and state introspection.
    """

    def __init__(
        self,
        cfg: RobotCfg = RobotSettings,
        robot: Optional[Union[str, RobotInterface]] = None,
        logger: Optional[LoggerType] = None,
    ) -> None:
        """Initialize controller with configuration and backend."""
        self.cfg = cfg
        self.ip_address = cfg.ip
        self.tool_id = cfg.tool_id
        self.user_frame_id = cfg.user_frame_id
        self.velocity = cfg.velocity
        self.logger = logger or Logger.get_logger("robot.controller")
        self.initial_pose: Optional[List[float]] = None

        try:
            self.robot = self._resolve_interface(robot)
            ErrorTracker.register_cleanup(self.shutdown)
            self.logger.info(f"RobotController initialized with IP {self.ip_address}")
        except Exception as exc:
            self.logger.exception("Failed to initialize RobotController")
            ErrorTracker.report(exc)
            raise

    def _resolve_interface(
        self, robot: Optional[Union[str, RobotInterface]]
    ) -> RobotInterface:
        """Determine and initialize the robot communication interface."""
        try:
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
        except Exception as exc:
            self.logger.exception("Failed to initialize robot interface")
            ErrorTracker.report(exc)
            raise

    def connect(self) -> bool:
        """Attempt connection to the robot and log outcome."""
        try:
            if not self.robot.is_connected():
                self.logger.error(f"Cannot connect to robot at {self.ip_address}")
                return False
            self.logger.info("Robot connected")
            return True
        except Exception as exc:
            self.logger.exception("Connection check failed")
            ErrorTracker.report(exc)
            return False

    def get_tcp_pose(self) -> Optional[List[float]]:
        """Retrieve current TCP pose from robot."""
        try:
            code, pose = self.robot.GetActualTCPPose()
            if code == 0:
                self.logger.debug(f"Current TCP pose: {pose}")
                return cast(List[float], pose)
            self.logger.error(f"GetActualTCPPose failed with code {code}")
            return None
        except Exception as exc:
            self.logger.exception("get_tcp_pose failed")
            ErrorTracker.report(exc)
            return None

    def move_linear(self, pose: List[float]) -> bool:
        """Send linear motion command to robot and verify actual movement."""
        try:
            current = self.get_tcp_pose()
            self.logger.info(f"Sending MoveL to: {pose}")
            code = self.robot.MoveL(
                pose, self.tool_id, self.user_frame_id, self.velocity
            )
            if code != 0:
                self.logger.error(f"MoveL failed with code {code} for pose: {pose}")
                return False

            if not self.wait_motion_done():
                self.logger.warning("MoveL issued but motion never finished")
                return False

            after = self.get_tcp_pose()
            if (
                current
                and after
                and all(abs(a - b) < 1e-2 for a, b in zip(current, after))
            ):
                self.logger.warning("Pose unchanged after MoveL; check robot state")
                return False

            self.logger.info(f"MoveL succeeded to: {pose}")
            return True
        except Exception as exc:
            self.logger.exception("move_linear failed")
            ErrorTracker.report(exc)
            return False

    def enable(self) -> None:
        """Enable robot motors."""
        try:
            code = self.robot.RobotEnable(1)
            if code != 0:
                self.logger.error(f"RobotEnable failed with code {code}")
            else:
                self.logger.info("Robot enabled")
        except Exception as exc:
            self.logger.exception("enable failed")
            ErrorTracker.report(exc)

    def disable(self) -> None:
        """Disable robot motors."""
        try:
            code = self.robot.RobotEnable(0)
            if code != 0:
                self.logger.error(f"Robot disable failed with code {code}")
            else:
                self.logger.info("Robot disabled")
        except Exception as exc:
            self.logger.exception("disable failed")
            ErrorTracker.report(exc)

    def wait_motion_done(self, timeout_sec: float = 20.0) -> bool:
        """Poll motion state until robot stops or timeout expires."""
        try:
            start = time.time()
            while time.time() - start < timeout_sec:
                code, done = self.robot.GetRobotMotionDone()
                self.logger.debug(f"Motion status: code={code}, done={done}")
                if code == 0 and done == 1:
                    return True
                time.sleep(0.05)
            self.logger.warning("Wait for motion done: timeout")
            return False
        except Exception as exc:
            self.logger.exception("wait_motion_done failed")
            ErrorTracker.report(exc)
            return False

    def check_safety(self) -> bool:
        """Check robot safety state and return True if all OK."""
        try:
            code = self.robot.GetSafetyCode()
            if code == 0:
                self.logger.info("Safety state OK")
                return True
            self.logger.error(f"Safety lock active: code {code}")
            return False
        except Exception as exc:
            self.logger.exception("check_safety failed")
            ErrorTracker.report(exc)
            return False

    def shutdown(self) -> None:
        """Cleanly shutdown the robot connection."""
        try:
            self.disable()
            self.robot.CloseRPC()
            self.logger.info("Robot shutdown complete")
        except Exception as exc:
            self.logger.exception("shutdown failed")
            ErrorTracker.report(exc)
