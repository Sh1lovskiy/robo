import time
from typing import List, Tuple
from pymodbus.client.sync import ModbusTcpClient
from pymodbus.payload import BinaryPayloadBuilder, Endian


BORUNTE_REGISTERS = {
    "mode": 0x0889,
    "set_mode": 0x0888,
    "alarm": 0x095C,
    "movement": 0x09A6,
    "joints": 0x08DC,
    "tcp_pose": 0x091C,
    "remote_len_clear": 0x7535,
    "remote_data": 0x7537,
    "clear_alarm_continue": 0x4E26,
    "servo_enable": 0x003E,
    "clear_alarm_all": 0x0046,
}

BORUNTE_ALARM_CODES = {
    0x0000: "No error",
    0x0001: "Emergency stop pressed",
    0x0002: "Overcurrent on servo",
    0x0003: "Servo drive overheating",
    0x0325: "Joint limit reached",
    0x0400: "Program aborted",
    0x0410: "Servo not ready",
    0x1000: "Invalid motion command",
}


class BorunteRobot:
    def __init__(self, ip="192.168.4.4", port=9760, unit=1, verbose=True):
        self.client = ModbusTcpClient(ip, port=port)
        self.unit = unit
        self.verbose = verbose

    def connect(self):
        ok = self.client.connect()
        self._log("Connected." if ok else "Connection failed.")
        return ok

    def close(self):
        self.client.close()
        self._log("Connection closed.")

    def _log(self, msg: str):
        if self.verbose:
            print("[BORUNTE]", msg)

    def _read(self, addr: int, count: int) -> List[int]:
        r = self.client.read_holding_registers(addr, count, unit=self.unit)
        if r.isError():
            raise RuntimeError(f"Read failed at {hex(addr)}")
        self._log(f"Read {count} registers from {hex(addr)}")
        return r.registers

    def _write(self, addr: int, values: List[int]):
        r = self.client.write_registers(addr, values, unit=self.unit)
        if r.isError():
            raise RuntimeError(f"Write failed at {hex(addr)}")
        self._log(f"Wrote {len(values)} registers at {hex(addr)}")

    def _write_single(self, addr: int, value: int):
        r = self.client.write_register(addr, value, unit=self.unit)
        if r.isError():
            raise RuntimeError(f"Write failed at {hex(addr)}")
        self._log(f"Wrote value {value} at {hex(addr)}")

    def get_mode(self) -> int:
        mode = self._read(BORUNTE_REGISTERS["mode"], 1)[0]
        self._log(f"Mode: {mode}")
        return mode

    def get_alarm(self) -> Tuple[int, str]:
        code = self._read(BORUNTE_REGISTERS["alarm"], 1)[0]
        message = BORUNTE_ALARM_CODES.get(code, "Unknown error code")
        return code, message

    def clear_alarm(self):
        self._write_single(BORUNTE_REGISTERS["clear_alarm_continue"], 1)
        self._log("Cleared alarm.")

    def is_moving(self) -> bool:
        moving = self._read(BORUNTE_REGISTERS["movement"], 1)[0] == 1
        self._log(f"Moving: {'yes' if moving else 'no'}")
        return moving

    def get_joint_angles(self) -> List[float]:
        regs = self._read(BORUNTE_REGISTERS["joints"], 12)
        return self._decode_float32(regs, 6)

    def get_tcp_pose(self) -> List[float]:
        regs = self._read(BORUNTE_REGISTERS["tcp_pose"], 12)
        return self._decode_float32(regs, 6)

    def move_to_pose(self, pose: List[float], wait=True, timeout=10.0):
        if len(pose) != 6:
            raise ValueError("Pose must have 6 elements.")
        regs = self._encode_float32(pose)
        self._write(BORUNTE_REGISTERS["remote_len_clear"], [len(regs), 1])
        self._write(BORUNTE_REGISTERS["remote_data"], regs)
        self._log(f"Sent TCP pose: {pose}")
        if wait:
            self._wait_for_idle(timeout)

    def clear_all_alarms(self, broadcast: bool = False):
        """
        Clears alarms either using unit-based or broadcast method.
        """
        addr = BORUNTE_REGISTERS["clear_alarm_all"]
        if broadcast:
            r = self.client.write_register(addr, 1, unit=0)
            if r.isError():
                raise RuntimeError("Broadcast failed.")
            self._log("Broadcasted alarm clear (unit=0).")
        else:
            self._write_single(addr, 1)
            self._log("Cleared alarms for this robot (unit mode).")

    def enable_servo(self):
        self._write_single(BORUNTE_REGISTERS["servo_enable"], 1)
        self._log("Sent command to enable servo motors.")

    def set_auto_mode(self):
        self._write_single(BORUNTE_REGISTERS["set_mode"], 2)
        self._log("Set robot mode to AUTO.")

    def move_tcp_delta(self, dx=0, dy=0, dz=0, drx=0, dry=0, drz=0, wait=True):
        pose = self.get_tcp_pose()
        target = [
            pose[0] + dx,
            pose[1] + dy,
            pose[2] + dz,
            pose[3] + drx,
            pose[4] + dry,
            pose[5] + drz,
        ]
        self.move_to_pose(target, wait=wait)

    def _wait_for_idle(self, timeout=10.0):
        start = time.time()
        while time.time() - start < timeout:
            if not self.is_moving():
                self._log("Robot is idle.")
                return
            time.sleep(0.1)
        self._log("Timeout waiting for motion to complete.")

    @staticmethod
    def _encode_float32(values: List[float]) -> List[int]:
        builder = BinaryPayloadBuilder(byteorder=Endian.Big, wordorder=Endian.Big)
        for v in values:
            vi = int(v * 1000)
            builder.add_16bit_uint((vi >> 16) & 0xFFFF)
            builder.add_16bit_uint(vi & 0xFFFF)
        return builder.to_registers()

    @staticmethod
    def _decode_float32(regs: List[int], count: int) -> List[float]:
        result = []
        for i in range(count):
            high, low = regs[2 * i], regs[2 * i + 1]
            raw = (high << 16) | low
            if raw >= 0x80000000:
                raw -= 0x100000000
            result.append(raw / 1000.0)
        return result


if __name__ == "__main__":
    bot = BorunteRobot(ip="192.168.4.4")
    if not bot.connect():
        exit(1)

    try:
        bot.clear_all_alarms()
        bot.set_auto_mode()
        time.sleep(0.3)
        bot.enable_servo()
        time.sleep(0.5)

        code, msg = bot.get_alarm()
        print(f"ALARM: {code} ({msg})")
        print("JOINTS:", bot.get_joint_angles())
        print("TCP:", bot.get_tcp_pose())

        if code != 0:
            bot.clear_alarm()

        bot.move_tcp_delta(dz=-10.0)
        code, msg = bot.get_alarm()
        print(f"ALARM: {code} ({msg})")
        bot.clear_all_alarms()
        print("TCP after move:", bot.get_tcp_pose())

    finally:
        bot.close()
