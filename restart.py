# restart.py
"""Simple robot restart utility using RPC."""

from __future__ import annotations

import time
from robot.controller import RobotController
from robot.Robot import RPC
from utils.logger import Logger
from utils.config import Config

logger = Logger.get_logger("robot_controller")


def restart_robot(ip_address: str | None = None) -> bool:
    """
    Restart robot by disabling, closing connection,
    reconnecting and enabling.
    """
    Config.load()
    ip = ip_address or Config.get("robot.ip", "192.168.58.2")
    rpc = RPC(ip=ip)
    robot = RobotController(rpc=rpc, logger=logger)
    logger.info("Connection to robot established")

    ret = robot.rpc.RobotEnable(0)
    if ret == 0:
        logger.info("Robot successfully disabled")
    else:
        logger.error(f"Error disabling robot: {ret}")
        return False

    robot.rpc.CloseRPC()
    logger.info("Connection to robot closed")

    time.sleep(3)

    rpc = RPC(ip=ip)
    robot = RobotController(rpc=rpc, logger=logger)
    logger.info("Reconnection to robot established")

    ret = robot.rpc.RobotEnable(1)
    if ret == 0:
        logger.info("Robot successfully enabled")
    else:
        logger.error(f"Error enabling robot: {ret}")
        return False

    safety_code = robot.rpc.GetSafetyCode()
    if safety_code == 0:
        logger.info("Robot is ready to operate")
    else:
        logger.error(f"Safety state prevents operation: {safety_code}")
        return False
    return True


if __name__ == "__main__":
    success = restart_robot()
    if success:
        logger.info("Robot restart completed successfully")
    else:
        logger.error("Failed to restart robot")
