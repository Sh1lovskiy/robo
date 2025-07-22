import subprocess


class ESP32Controller:
    def __init__(self, port="/dev/ttyUSB0", module="robo"):
        self.port = port
        self.module = module

    def exec(self, code: str):
        """Run any code on ESP32 plate"""
        full_code = f"import {self.module}; {code}"
        subprocess.run(
            ["mpremote", "connect", self.port, "exec", full_code], check=True
        )

    def move_motor(self, steps=200, direction=1, delay_us=500):
        self.exec(
            f"{self.module}.motor.move(steps={steps}, direction={direction}, delay_us={delay_us})"
        )

    def laser_on(self):
        self.exec(f"{self.module}.laser.on()")

    def laser_off(self):
        self.exec(f"{self.module}.laser.off()")

    def laser_blink(self, count=3, interval=0.5):
        self.exec(f"{self.module}.laser.blink(count={count}, interval={interval})")


if __name__ == "__main__":
    esp = ESP32Controller()

    esp.laser_blink(5, 0.5)
    esp.move_motor(steps=100, direction=1, delay_us=5000)
    esp.laser_off()
