# restart.py
"""Utility to restart the robot via the low level RPC interface."""

import time
from robot.Robot import RPC
from utils.logger import Logger
from utils.config import Config
from utils.constants import DEFAULT_ROBOT_IP

logger = Logger.get_logger("robot.restart")


def restart_robot(ip_address: str | None = None):
    """Restart robot by disabling, closing connection, reconnecting and enabling."""
    Config.load()
    ip = ip_address or Config.get("robot.ip", DEFAULT_ROBOT_IP)
    rpc = RPC(ip=ip)
    logger.info("Connection to robot established")

    ret = rpc.RobotEnable(0)
    if ret == 0:
        logger.info("Robot successfully disabled")
    else:
        logger.error(f"Error disabling robot: {ret}")
        return False

    rpc.CloseRPC()
    logger.info("Connection to robot closed")

    time.sleep(3)

    rpc = RPC(ip=ip)
    logger.info("Reconnection to robot established")

    ret = rpc.RobotEnable(1)
    if ret == 0:
        logger.info("Robot successfully enabled")
    else:
        logger.error(f"Error enabling robot: {ret}")
        return False

    safety_code = rpc.GetSafetyCode()
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
