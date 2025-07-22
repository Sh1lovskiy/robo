from machine import Pin
from time import sleep, sleep_us


PIN_PUL = 17
PIN_DIR = 16
PIN_LASER = 15


class LaserController:
    """Controls a laser connected to a digital output pin."""

    def __init__(self, pin: int = PIN_LASER):
        self._laser = Pin(pin, Pin.OUT)

    def on(self):
        """Turn the laser on."""
        self._laser.value(1)

    def off(self):
        """Turn the laser off."""
        self._laser.value(0)

    def blink(self, count: int = 5, interval: float = 0.3):
        """
        Blink the laser a number of times.

        :param count: Number of on/off cycles
        :param interval: Time in seconds between blinks
        """
        for _ in range(count):
            self.on()
            sleep(interval)
            self.off()
            sleep(interval)


class MotorController:
    """Controls a stepper motor using step/direction pins."""

    def __init__(self, pul_pin: int = PIN_PUL, dir_pin: int = PIN_DIR):
        self._pul = Pin(pul_pin, Pin.OUT)
        self._dir = Pin(dir_pin, Pin.OUT)

    def move(
        self,
        steps: int = 100,
        direction: int = 1,
        delay_us: int = 500,
        invert_dir: bool = False,
        pause_us: int = 0,
        **kwargs,
    ):
        """
        Move the stepper motor a number of steps.

        :param steps: Number of steps to move
        :param direction: 1 = forward, 0 = backward
        :param delay_us: Delay between step transitions
        :param invert_dir: If True, inverts direction logic
        :param pause_us: Optional delay between each pulse cycle
        :param kwargs: Future extensibility (acceleration, etc.)
        """
        actual_dir = direction ^ invert_dir
        self._dir.value(actual_dir)

        for _ in range(steps):
            self._pul.value(1)
            sleep_us(delay_us)
            self._pul.value(0)
            sleep_us(delay_us)
            if pause_us > 0:
                sleep_us(pause_us)


# Instantiate shared instances for use in REPL or remote exec
laser = LaserController()
motor = MotorController()
