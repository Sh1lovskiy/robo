"""Robot motion helpers with user confirmation."""

from __future__ import annotations

import time
from typing import Iterable

import numpy as np
from robot.rpc import RPC

from utils.keyboard import TerminalEchoSuppressor
from utils.logger import Logger
from utils.settings import robot as robot_cfg

from .visualization import visualize_tcp_target

logger = Logger.get_logger("robot_scan.motion")


def connect_robot(ip: str) -> RPC:
    """Connect to robot controller."""
    logger.info(f"Connecting to robot at {ip}")
    rpc = RPC(ip=ip)
    if rpc.RobotEnable(1) != 0:
        raise RuntimeError("Failed to enable robot")
    logger.info("Robot connected")
    return rpc


def _confirm(prompt: str) -> bool:
    tes = TerminalEchoSuppressor()
    tes.start()
    try:
        ans = input(f"{prompt} [y/N]: ").strip().lower() == "y"
    finally:
        tes.stop()
    return ans


def move_l(rpc: RPC, pose: Iterable[float], vel: float | None = None) -> None:
    """Move robot linearly to ``pose`` after visualization and confirmation."""
    pose_arr = np.asarray(list(pose), dtype=float)
    visualize_tcp_target(pose_arr)
    if not _confirm("Move robot to this pose?"):
        logger.info("Move canceled by user")
        return
    vel = vel if vel is not None else robot_cfg.velocity
    code, joints = rpc.GetInverseKin(0, pose_arr.tolist())
    if code != 0 or joints is None:
        raise RuntimeError(f"Inverse kinematics failed: code={code}")
    code = rpc.MoveL(
        desc_pos=pose_arr.tolist(), tool=0, user=0, joint_pos=joints, vel=vel
    )
    if code != 0:
        raise RuntimeError(f"MoveL failed: code={code}")
    logger.info(f"MoveL executed to {pose_arr}")
    time.sleep(robot_cfg.restart_delay)
