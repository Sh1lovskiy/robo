from __future__ import annotations

"""High-level, safe, and minimal robot controller."""

import time
from typing import List, Optional

import numpy as np

from utils.error_tracker import ErrorTracker
from utils.logger import Logger, LoggerType
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
        self.logger = logger or Logger.get_logger("controller")
        self.robot = RPC(ip=self.ip)
        ErrorTracker.register_cleanup(self.shutdown)

    def shutdown(self) -> None:
        self.logger.info("Shutting down robot...")
        try:
            try:
                self.robot.StopMotion()
            except Exception:
                self.logger.warning("StopMotion not available")
            # self.disable()
            self.robot.CloseRPC()
            self.logger.info("Robot shutdown complete")
        except Exception as exc:
            self.logger.error("Shutdown failed")
            ErrorTracker.report(exc)

    def connect(
        self,
        attempts: int = 3,
        delay: float = 0.2,
        safety_check: bool = False,
    ) -> bool:
        for attempt in range(1, attempts + 1):
            try:
                self.logger.info(f"Connect attempt {attempt} to {self.ip}")
                self.robot = RPC(ip=self.ip)

                if self.enable():
                    if not safety_check:
                        self.logger.info("Robot connected (safety check skipped)")
                        return True

                    try:
                        start = time.time()
                        while time.time() - start < 2.0:
                            safety = self.robot.GetSafetyCode()
                            if safety == 0:
                                self.logger.info("Robot connected and in safe state")
                                return True
                            else:
                                self.logger.warning(
                                    f"Safety check failed: code {safety}"
                                )
                                time.sleep(0.2)
                        self.logger.error("Timeout waiting for safe state")
                    except Exception as e:
                        self.logger.warning(f"GetSafetyCode failed: {e}")
                        ErrorTracker.report(e)

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
        if not self.robot or getattr(self.robot, "robot", None) is None:
            self.logger.warning("RPC not initialized or already closed")
            return False
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

    def move_linear(self, pose: List[float]) -> bool:
        """Move TCP linearly to ``pose``."""

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
            return True
        except Exception as exc:
            self.logger.error("move_linear failed")
            ErrorTracker.report(exc)
            return False

    def move_joints(self, joints: List[float]) -> bool:
        """Move robot joints to ``joints`` configuration."""

        try:
            code = self.robot.MoveJ(
                joint=joints,
                tool=self.tool_id,
                user=self.user_frame_id,
                vel=self.velocity,
            )
            if code != 0:
                self.logger.error(f"MoveJ failed: code {code}")
                return False
            return True
        except Exception as exc:
            self.logger.error("move_joints failed")
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
        """Return current TCP pose or ``None`` on error."""

        code, pose = self.robot.GetActualTCPPose()
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

    def restart(self, delay: float = 0.05, attempts: int = 3) -> bool:
        """Restart RPC connection."""

        self.logger.info(f"Restarting robot at {self.ip}")
        try:
            self.disable()
            self.robot.CloseRPC()
            time.sleep(delay)
            self.robot = RPC(ip=self.ip)
            return self.connect(attempts=attempts, delay=delay)
        except Exception as exc:
            self.logger.error("Restart failed")
            ErrorTracker.report(exc)
            return False

    def stop(self) -> None:
        """Emergency stop current motion."""

        try:
            self.robot.StopMotion()
        except Exception as exc:
            self.logger.error("StopMotion failed")
            ErrorTracker.report(exc)

    def get_status(self) -> Optional[dict]:
        """Return basic robot status information."""

        try:
            safety = self.robot.GetSafetyCode()
            code, motion_done = self.robot.GetRobotMotionDone()
            return {
                "safety_code": int(safety),
                "motion_done": int(motion_done),
                "error_code": int(code),
            }
        except Exception as exc:
            self.logger.error("get_status failed")
            ErrorTracker.report(exc)
            return None
