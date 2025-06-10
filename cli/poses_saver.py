# cli/poses_saver.py

import time
import numpy as np
from robot.controller import RobotController
from utils.logger import Logger


class PoseSaver:
    def save(self, filename, pose):
        raise NotImplementedError


class TxtPoseSaver(PoseSaver):
    def save(self, filename, pose):
        with open(filename, "a") as f:
            f.write(" ".join(f"{v:.8f}" for v in pose) + "\n")


def main(
    controller: RobotController = None,
    saver: PoseSaver = None,
    logger=None,
    filename: str = "poses.txt",
):
    controller = controller or RobotController()
    saver = saver or TxtPoseSaver()
    logger = logger or Logger.get_logger("cli.poses_saver")

    if hasattr(controller, "initialize"):
        controller.initialize()
    print("Press ENTER to save current pose. Type 'q' + ENTER to quit.")

    while True:
        inp = input("> ")
        if inp.strip().lower() == "q":
            break
        pose = controller.get_current_pose()
        if pose is not None:
            saver.save(filename, pose)
            logger.info(f"Pose saved: {pose}")
            print(f"Saved: {pose}")
        else:
            logger.error("Failed to get pose.")
        time.sleep(0.2)
    controller.shutdown()
    print(f"All poses saved to {filename}")


def _test_save_pose():
    import tempfile, os

    arr = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
    saver = TxtPoseSaver()
    with tempfile.NamedTemporaryFile(delete=False) as f:
        fname = f.name
    saver.save(fname, arr)
    arr2 = np.loadtxt(fname)
    assert np.allclose(arr, arr2), "Saved pose not equal to loaded pose"
    os.remove(fname)
    print("save_pose test OK")


if __name__ == "__main__":
    main()
