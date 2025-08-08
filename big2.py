import logging
import time
from typing import List

from pymodbus.client.sync import ModbusTcpClient
from pymodbus.payload import BinaryPayloadBuilder, BinaryPayloadDecoder
from pymodbus.constants import Endian


# Borunte Modbus register map
REG_TCP_POSE = 0x091C  # TCP pose (XYZ + RxRyRz), 6 float32 = 12 registers
REG_REMOTE_LEN = 0x7535  # Remote command length and trigger
REG_REMOTE_DATA = 0x7537  # Remote command buffer (write TCP target)
REG_MOVEMENT = 0x09A6  # Motion flag (1 = moving, 0 = idle)


class BorunteTCPModbusClient:
    """
    Modbus TCP client for Borunte industrial robot.

    Supports:
      - Reading TCP pose (in mm and degrees)
      - Writing new TCP pose
      - Triggering MoveL-type motion
    """

    def __init__(self, ip="192.168.4.4", port=9760, unit=1, timeout=3.0):
        self.ip = ip
        self.port = port
        self.unit = unit
        self.timeout = timeout
        self.client = ModbusTcpClient(ip, port=port, timeout=timeout)
        self.logger = logging.getLogger("BorunteModbus")
        logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

    def connect(self):
        if not self.client.connect():
            raise ConnectionError("Failed to connect to robot.")
        self.logger.info("Connected to Borunte.")

    def disconnect(self):
        self.client.close()
        self.logger.info("Connection closed.")

    def read_tcp_pose(self) -> List[float]:
        """
        Read current TCP pose (XYZ + RxRyRz) from robot.
        Values are in millimeters and degrees.
        """
        result = self.client.read_holding_registers(REG_TCP_POSE, 12, unit=self.unit)
        if result.isError():
            raise RuntimeError("Failed to read TCP pose.")
        decoder = BinaryPayloadDecoder.fromRegisters(
            result.registers, byteorder=Endian.Big, wordorder=Endian.Big
        )
        pose = [decoder.decode_32bit_float() for _ in range(6)]
        self.logger.info(f"Current TCP pose: {pose}")
        return pose

    def write_tcp_pose_and_move(self, pose: List[float]):
        """
        Send target TCP pose and trigger robot movement.
        Pose must be 6 float values: [X, Y, Z, Rx, Ry, Rz]
        """
        if len(pose) != 6:
            raise ValueError("Pose must have 6 float values.")
        builder = BinaryPayloadBuilder(byteorder=Endian.Big, wordorder=Endian.Big)
        for value in pose:
            builder.add_32bit_float(value)
        regs = builder.to_registers()

        # Step 1: write command length (12 registers for 6 floats) and trigger flag
        self.client.write_registers(REG_REMOTE_LEN, [12, 1], unit=self.unit)

        # Step 2: write TCP pose to data buffer
        self.client.write_registers(REG_REMOTE_DATA, regs, unit=self.unit)

        self.logger.info(f"Target pose written: {pose}")

    def wait_for_motion_done(self, timeout=10.0):
        """
        Wait until motion flag becomes 0 (idle).
        """
        start = time.time()
        while time.time() - start < timeout:
            result = self.client.read_holding_registers(REG_MOVEMENT, 1, unit=self.unit)
            if result.isError():
                raise RuntimeError("Failed to read movement status.")
            moving = result.registers[0] == 1
            if not moving:
                self.logger.info("Motion completed.")
                return
            time.sleep(0.2)
        self.logger.warning("Timeout waiting for motion to complete.")

    def move_z_down(self, dz: float):
        """
        Move TCP Z position by dz (mm).
        Negative dz means moving down.
        """
        pose = self.read_tcp_pose()
        pose[2] += dz
        self.write_tcp_pose_and_move(pose)
        self.wait_for_motion_done()


def main():
    bot = BorunteTCPModbusClient()

    try:
        bot.connect()
        bot.move_z_down(-200.0)
        final_pose = bot.read_tcp_pose()
        print("Final TCP pose:", final_pose)

    except Exception as e:
        bot.logger.exception(f"Error: {e}")

    finally:
        bot.disconnect()


if __name__ == "__main__":
    main()
