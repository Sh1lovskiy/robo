from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple


class RobotInterface(ABC):
    """Minimal robot command set used by :class:`RobotController`."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Return ``True`` if the connection is active."""

    @abstractmethod
    def GetActualTCPPose(self) -> Tuple[int, List[float]]:
        """Return TCP pose as ``(err, pose)``."""

    @abstractmethod
    def MoveJ(
        self, joint_pos: List[float], tool: int, user: int, vel: float
    ) -> int:
        """Execute a joint move."""

    @abstractmethod
    def MoveL(
        self, desc_pos: List[float], tool: int, user: int, vel: float
    ) -> int:
        """Execute a linear move."""

    @abstractmethod
    def RobotEnable(self, flag: int) -> int:
        """Enable (1) or disable (0) robot power."""

    @abstractmethod
    def GetRobotMotionDone(self) -> Tuple[int, int]:
        """Return motion done status."""

    @abstractmethod
    def GetActualJointPosDegree(self) -> Tuple[int, List[float]]:
        """Return current joint positions."""

    @abstractmethod
    def GetSafetyCode(self) -> int:
        """Return current safety state code."""

    @abstractmethod
    def CloseRPC(self) -> None:
        """Close underlying connection."""


class FairinoRPC(RobotInterface):
    """Adapter around the SDK ``RPC`` class implementing :class:`RobotInterface`."""

    def __init__(self, ip: str) -> None:
        from .rpc import RPC

        self._rpc = RPC(ip=ip)

    def is_connected(self) -> bool:
        return self._rpc.is_conect

    def GetActualTCPPose(self) -> Tuple[int, List[float]]:
        return self._rpc.GetActualTCPPose()

    def MoveJ(
        self, joint_pos: List[float], tool: int, user: int, vel: float
    ) -> int:
        return self._rpc.MoveJ(joint_pos, tool, user, vel=vel)

    def MoveL(
        self, desc_pos: List[float], tool: int, user: int, vel: float
    ) -> int:
        return self._rpc.MoveL(desc_pos=desc_pos, tool=tool, user=user, vel=vel)

    def RobotEnable(self, flag: int) -> int:
        return self._rpc.RobotEnable(flag)

    def GetRobotMotionDone(self) -> Tuple[int, int]:
        return self._rpc.GetRobotMotionDone()

    def GetActualJointPosDegree(self) -> Tuple[int, List[float]]:
        return self._rpc.GetActualJointPosDegree()

    def GetSafetyCode(self) -> int:
        return self._rpc.GetSafetyCode()

    def CloseRPC(self) -> None:
        self._rpc.CloseRPC()
__all__ = ['RobotInterface', 'FairinoRPC']
