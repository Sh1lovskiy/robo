from __future__ import annotations
import time
from typing import List, Optional

import numpy as np

from utils.logger import Logger, LoggerType
from utils.error_tracker import ErrorTracker
from utils.settings import robot as RobotSettings, RobotCfg

from robot.rpc import RPC


def calc_pose_errors(target: List[float], actual: List[float]) -> tuple[float, float]:
    t, a = np.array(target), np.array(actual)
    pos_err = float(np.linalg.norm(t[:3] - a[:3]))
    rot_err = float(np.linalg.norm((t[3:] - a[3:] + 180) % 360 - 180))
    return pos_err, rot_err


class RobotController:
    """High-level, safe, and minimal robot controller (direct RPC)."""

    def __init__(
        self,
        cfg: RobotCfg = RobotSettings,
        ip: Optional[str] = None,
        logger: Optional[LoggerType] = None,
    ) -> None:
        self.cfg = cfg
        self.ip = ip or cfg.ip
        self.tool_id = cfg.tool_id
        self.user_frame_id = cfg.user_frame_id
        self.velocity = cfg.velocity
        self.logger = logger or Logger.get_logger("robot.controller")
        self.robot = RPC(ip=self.ip)
        ErrorTracker.register_cleanup(self.shutdown)

    def shutdown(self) -> None:
        self.logger.info("Shutting down robot...")
        try:
            try:
                self.robot.StopMotion()
            except Exception:
                self.logger.warning("StopMotion not available")
            self.disable()
            self.robot.CloseRPC()
            self.logger.info("Robot shutdown complete")
        except Exception as exc:
            self.logger.error("Shutdown failed")
            ErrorTracker.report(exc)

    def connect(self, attempts: int = 3, delay: float = 0.2) -> bool:
        for attempt in range(1, attempts + 1):
            try:
                self.logger.info(f"Connect attempt {attempt} to {self.ip}")
                self.robot = RPC(ip=self.ip)
                if self.enable():
                    safety = self.robot.GetSafetyCode()
                    if safety == 0:
                        self.logger.info("Robot connected and in safe state")
                        return True
                    self.logger.warning(f"Safety check failed: {safety}")
            except Exception as exc:
                self.logger.warning(f"Connect error: {exc}")
                ErrorTracker.report(exc)
            time.sleep(delay)
        self.logger.error("Robot connection failed after retries")
        return False

    def enable(self) -> bool:
        try:
            code = self.robot.RobotEnable(1)
            if code == 0:
                self.logger.info("Robot enabled")
                return True
            self.logger.error(f"RobotEnable(1) failed: code {code}")
            return False
        except Exception as exc:
            self.logger.error("Enable failed")
            ErrorTracker.report(exc)
            return False

    def disable(self) -> bool:
        try:
            code = self.robot.RobotEnable(0)
            if code == 0:
                self.logger.info("Robot disabled")
                return True
            self.logger.error(f"RobotEnable(0) failed: code {code}")
            return False
        except Exception as exc:
            self.logger.error("Disable failed")
            ErrorTracker.report(exc)
            return False

    def move_linear(
        self, pose: List[float], pos_tol: float = 1.0, rot_tol: float = 2.0
    ) -> bool:
        try:
            if self.robot.GetSafetyCode() != 0:
                self.logger.error("Safety lock active!")
                return False
            code = self.robot.MoveL(
                desc_pos=pose,
                tool=self.tool_id,
                user=self.user_frame_id,
                vel=self.velocity,
            )
            if code != 0:
                self.logger.error(f"MoveL failed: code {code}")
                return False
            # actual = self.get_tcp_pose()
            # if actual is None:
            #     self.logger.error("No pose after motion")
            #     return False
            # pos_err, rot_err = calc_pose_errors(pose, actual)
            # if pos_err > pos_tol or rot_err > rot_tol:
            #     self.logger.warning(f"Pose mismatch after motion: pos_err={pos_err:.2f}, rot_err={rot_err:.2f}")
            #     return False
            return True
        except Exception as exc:
            self.logger.error("move_linear failed")
            ErrorTracker.report(exc)
            return False

    def wait_motion_done(self, timeout: float = 20.0) -> bool:
        start = time.time()
        try:
            while time.time() - start < timeout:
                code, done = self.robot.GetRobotMotionDone()
                if code == 0 and done == 1:
                    return True
                time.sleep(0.05)
            self.logger.warning("wait_motion_done: timeout")
            return False
        except Exception as exc:
            self.logger.error("wait_motion_done failed")
            ErrorTracker.report(exc)
            return False

    def get_tcp_pose(self) -> Optional[List[float]]:
        code, pose = self.robot.GetActualTCPPose()
        print(pose)
        try:
            if code == 0 and isinstance(pose, (list, tuple)) and len(pose) == 6:
                return [float(x) for x in pose]
            else:
                self.logger.error(f"GetActualTCPPose failed: code={code}, pose={pose}")
                return None
        except Exception as exc:
            self.logger.error("get_tcp_pose failed")
            ErrorTracker.report(exc)
            return None

    def restart(self, delay: float = 0.2, attempts: int = 3) -> bool:
        self.logger.info(f"Restarting robot at {self.ip}")
        try:
            self.disable()
            self.robot.CloseRPC()
            time.sleep(delay)
            return self.connect(attempts=attempts, delay=delay)
        except Exception as exc:
            self.logger.error("Restart failed")
            ErrorTracker.report(exc)
            return False
