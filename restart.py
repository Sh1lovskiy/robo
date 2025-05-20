import time
from fairino import Robot
from misc.logger import Logger


logger = Logger.get_logger("robot_controller")


def restart_robot(ip_address="192.168.58.2"):
    """Restarts the robot by disabling it, closing the connection, and reconnecting."""
    robot = Robot.RPC(ip_address)
    logger.info("Connection to robot established")

    ret = robot.RobotEnable(0)
    if ret == 0:
        logger.info("Robot successfully disabled")
    else:
        logger.error(f"Error disabling robot: {ret}")
        return False

    robot.CloseRPC()
    logger.info("Connection to robot closed")

    time.sleep(3)

    robot = Robot.RPC(ip_address)
    logger.info("Reconnection to robot established")

    ret = robot.RobotEnable(1)
    if ret == 0:
        logger.info("Robot successfully enabled")
    else:
        logger.error(f"Error enabling robot: {ret}")
        return False

    safety_code = robot.GetSafetyCode()
    if safety_code == 0:
        logger.info("Robot is ready to operate")
    else:
        logger.error(f"Safety state prevents operation: {safety_code}")
        return False
    return True


if __name__ == "__main__":
    success = restart_robot("192.168.58.2")
    if success:
        logger.info("Robot restart completed successfully")
    else:
        logger.error("Failed to restart robot")
