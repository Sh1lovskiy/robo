# cli/get_intrinsics.py

from vision.realsense import RealSenseCamera
from utils.logger import Logger


class IntrinsicsCLI:
    """CLI utility for printing RealSense camera intrinsics."""

    def __init__(self, camera=None, logger=None):
        self.camera = camera or RealSenseCamera()
        self.logger = logger or Logger.get_logger("cli.get_intrinsics")

    def run(self):
        self.camera.start()
        intr = self.camera.get_intrinsics()
        self.logger.info(f"Intrinsics: {intr}")
        print("Depth Camera Intrinsics:")
        for k, v in intr.items():
            print(f"{k}: {v}")
        self.camera.stop()


def main():
    IntrinsicsCLI().run()


if __name__ == "__main__":
    main()
