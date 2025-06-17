# robot/controller.py
"""High-level robot control interface."""

from typing import List, Optional, Union
from robot.rpc import RPC
from utils.logger import Logger
from utils.config import Config


class RobotController:
    """
    High-level robot controller for movement, pose, and state management.
    All configuration is loaded from config.yaml unless overridden.
    """

    def __init__(
        self,
        rpc: Optional[Union[str, RPC]] = None,
        logger=None,
        config: Optional[Config] = None,
    ):
        self.config = config or Config
        self.ip_address = self.config.get("robot.ip", default="192.168.1.10")
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
        if not self.rpc.is_conect:
            self.logger.error(f"Cannot connect to robot at {self.ip_address}")
            return False
        self.logger.info("Robot connected")
        return True

    def get_tcp_pose(self) -> Optional[List[float]]:
        res = self.rpc.GetActualTCPPose()
        if res[0] == 0:
            self.logger.debug(f"Current pose: {res[1]}")
            return res[1]
        self.logger.error(f"GetActualTCPPose failed with code {res[0]}")
        return None

    def move_joints(self, joints: List[float]) -> bool:
        code = self.rpc.MoveJ(
            desc_pos=joints,
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
        code = self.rpc.MoveL(
            desc_pos=pose, tool=self.tool_id, user=self.user_frame_id, vel=self.velocity
        )
        if code != 0:
            self.logger.error(f"MoveL failed with code {code}")
            return False
        self.logger.info(f"MoveL success: {pose}")
        return True

    def record_home(self):
        pose = self.get_current_pose()
        if pose is None:
            self.logger.error("Failed to record home position")
            raise RuntimeError("Failed to record home position")
        self.initial_pose = pose
        self.logger.info(f"Home position recorded: {pose}")

    def return_to_home(self):
        if self.initial_pose is not None:
            self.move_linear(self.initial_pose)
            self.logger.info("Returned to home position")
        else:
            self.logger.warning("Home position not recorded")

    def shutdown(self):
        self.rpc.RobotEnable(0)
        self.rpc.CloseRPC()
        self.logger.info("Robot shutdown complete")
