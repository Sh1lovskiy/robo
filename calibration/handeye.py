# calibration/handeye.py
"""Hand-eye calibration helpers using OpenCV."""

import numpy as np
import cv2
from utils.logger import Logger


class HandEyeSaver:
    """Strategy interface for saving Hand-Eye calibration results."""

    def save(self, filename, R, t):
        raise NotImplementedError


class NPZHandEyeSaver(HandEyeSaver):
    def save(self, filename, R, t):
        np.savez(filename, R=R, t=t)


class TxtHandEyeSaver(HandEyeSaver):
    def save(self, filename, R, t):
        with open(filename, "w") as f:
            f.write("R =\n")
            np.savetxt(f, R, fmt="%.8f")
            f.write("t =\n")
            np.savetxt(f, t.reshape(1, -1), fmt="%.8f")


class HandEyeCalibrator:
    """
    Hand-Eye calibration using all OpenCV methods automatically.
    Uses SOLID principles and centralized logging.
    """

    METHODS = {
        name.replace("CALIB_HAND_EYE_", ""): getattr(cv2, name)
        for name in dir(cv2)
        if name.startswith("CALIB_HAND_EYE_")
    }

    def __init__(self, logger=None):
        self.R_gripper2base = []
        self.t_gripper2base = []
        self.R_target2cam = []
        self.t_target2cam = []
        self.logger = logger or Logger.get_logger("calibration.handeye")

    def add_sample(self, R_gripper, t_gripper, R_target, t_target):
        self.R_gripper2base.append(R_gripper)
        self.t_gripper2base.append(t_gripper)
        self.R_target2cam.append(R_target)
        self.t_target2cam.append(t_target)
        self.logger.debug("Added Hand-Eye sample")

    def calibrate(self, method="TSAI"):
        """
        Calibrate using the specified method (case-insensitive).
        """
        assert self.R_gripper2base, "No samples for calibration"
        key = method.upper()
        if key not in self.METHODS:
            raise ValueError(
                f"Unknown method '{method}'. Available: {list(self.METHODS.keys())}"
            )
        R, t = cv2.calibrateHandEye(
            self.R_gripper2base,
            self.t_gripper2base,
            self.R_target2cam,
            self.t_target2cam,
            method=self.METHODS[key],
        )
        self.logger.info(f"Hand-Eye calibration ({method}) done")
        return R, t

    def calibrate_all(self):
        """
        Run calibration with all available methods.
        Returns: dict {method_name: (R, t)}
        """
        results = {}
        for name, code in self.METHODS.items():
            try:
                R, t = cv2.calibrateHandEye(
                    self.R_gripper2base,
                    self.t_gripper2base,
                    self.R_target2cam,
                    self.t_target2cam,
                    method=code,
                )
                results[name] = (R, t)
                self.logger.info(f"Hand-Eye calibration ({name}) done")
            except Exception as e:
                self.logger.error(f"Hand-Eye calibration failed for {name}: {e}")
        return results

    def save(self, saver: HandEyeSaver, filename, R, t):
        saver.save(filename, R, t)
        self.logger.info(
            f"Calibration saved with {saver.__class__.__name__} to {filename}"
        )


def simple_handeye_test():
    R_g2b = []
    t_g2b = []
    R_t2c = []
    t_t2c = []

    angles = [0, np.pi / 8, -np.pi / 8]
    translations = [
        [0, 0, 0],
        [0.035, 0.000, 0.000],
        [0.000, 0.035, 0.000],
    ]

    for ang, t in zip(angles, translations):
        R = cv2.Rodrigues(np.array([0, 0, ang], dtype=np.float64))[0]
        R_g2b.append(R)
        t_g2b.append(np.array(t, dtype=np.float64))
        Rt = cv2.Rodrigues(np.array([0, ang / 2, 0], dtype=np.float64))[0]
        tt = np.array([0.01, 0.01, 0.5], dtype=np.float64) + np.random.normal(
            0, 0.001, 3
        )
        R_t2c.append(Rt)
        t_t2c.append(tt)

    R, t = cv2.calibrateHandEye(
        R_g2b, t_g2b, R_t2c, t_t2c, method=cv2.CALIB_HAND_EYE_TSAI
    )

    print("Hand-Eye calibration (TSAI) result:")
    print("R =\n", R)
    print("t =\n", t)
    print("Norm t (meters):", np.linalg.norm(t))


if __name__ == "__main__":
    simple_handeye_test()
