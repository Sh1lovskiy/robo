# robot/controller.py

"""
High-level robot control interface using Fairino Robot SDK.
SOLID principles, dependency injection, robust logging.
"""

from typing import List, Optional, Union
from utils.logger import Logger
from utils.config import Config
from utils.constants import DEFAULT_ROBOT_IP

from robot.Robot import RPC


class RobotController:
    """
    High-level robot controller for movement, pose, and state management.
    Loads configuration from config.yaml unless overridden.
    """

    def __init__(
        self,
        rpc: Optional[Union[str, RPC]] = None,
        logger: Optional[Logger] = None,
        config: Optional[Config] = None,
    ):
        """
        Initialize the robot controller.

        Args:
            rpc (Optional[Union[str, RPC]]): RPC object or IP address as string.
            logger (Optional[Logger]): Logger instance.
            config (Optional[Config]): Config object.
        """
        self.config = config or Config
        self.ip_address = self.config.get("robot.ip", default=DEFAULT_ROBOT_IP)
        self.tool_id = self.config.get("robot.tool_id", default=0)
        self.user_frame_id = self.config.get("robot.user_frame_id", default=0)
        self.velocity = self.config.get("robot.velocity", default=20.0)
        self.logger = logger or Logger.get_logger("robot.controller", json_format=True)
        self.initial_pose = None

        if rpc is None:
            self.rpc = RPC(ip=self.ip_address)
            self.logger.info(
                f"RobotController initialized with IP {self.ip_address} (default from config)"
            )
        elif isinstance(rpc, str):
            self.rpc = RPC(ip=rpc)
            self.logger.info(f"RobotController initialized with IP {rpc} (from string)")
        elif isinstance(rpc, RPC):
            self.rpc = rpc
            self.logger.info(
                f"RobotController initialized with external RPC (IP: {self.ip_address})"
            )
        else:
            raise TypeError(f"Invalid rpc argument: {type(rpc)}")

        self.logger.info(f"RobotController initialized with IP {self.ip_address}")

    def connect(self) -> bool:
        """
        Check connection to the robot.

        Returns:
            bool: True if connected, False otherwise.
        """
        if not self.rpc.is_conect:
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
        res = self.rpc.GetActualTCPPose()
        if res[0] == 0:
            self.logger.debug(f"Current pose: {res[1]}")
            return res[1]
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
        code = self.rpc.MoveJ(
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
        code = self.rpc.MoveL(
            desc_pos=pose, tool=self.tool_id, user=self.user_frame_id, vel=self.velocity
        )
        if code != 0:
            self.logger.error(f"MoveL failed with code {code}")
            return False
        self.logger.info(f"MoveL success: {pose}")
        return True

    def record_home(self):
        """
        Record current pose as 'home'.
        """
        pose = self.get_tcp_pose()
        if pose is None:
            self.logger.error("Failed to record home position")
            raise RuntimeError("Failed to record home position")
        self.initial_pose = pose
        self.logger.info(f"Home position recorded: {pose}")

    def return_to_home(self):
        """
        Move robot to previously recorded 'home' pose.
        """
        if self.initial_pose is not None:
            self.move_linear(self.initial_pose)
            self.logger.info("Returned to home position")
        else:
            self.logger.warning("Home position not recorded")

    def shutdown(self):
        """
        Disable and disconnect robot.
        """
        self.rpc.RobotEnable(0)
        self.rpc.closeRPC_state = True
        self.logger.info("Robot shutdown complete")

    # --- Примеры дополнительных методов с SOLID ---
    def wait_motion_done(self, timeout_sec=20):
        """
        Wait until robot finishes motion (with timeout).

        Args:
            timeout_sec (float): Maximum time to wait.

        Returns:
            bool: True if motion finished, False if timeout.
        """
        import time

        start = time.time()
        while time.time() - start < timeout_sec:
            code, done = self.rpc.GetRobotMotionDone()
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
        res = self.rpc.GetActualJointPosDegree()
        if res[0] == 0:
            return res[1]
        self.logger.error("GetActualJointPosDegree failed")
        return None
