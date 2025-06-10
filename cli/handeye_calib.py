# cli/handeye_calib.py

import os
import numpy as np
from calibration.handeye import HandEyeCalibrator, NPZHandEyeSaver, TxtHandEyeSaver
from utils.config import Config
from utils.logger import Logger
from calibration.pose_loader import JSONPoseLoader


class PoseLoader:
    """
    Loads (R, t) pose pairs from a text file.
    Each line: tx ty tz r11 r12 ... r33
    """

    @staticmethod
    def load_poses(filename):
        arr = np.loadtxt(filename)
        Rs, ts = [], []
        for row in arr:
            t = row[:3]
            R = row[3:].reshape(3, 3)
            Rs.append(R)
            ts.append(t)
        return Rs, ts


class HandEyeCalibrationIO:
    """
    Handles file and directory naming for Hand-Eye calibration.
    """

    def __init__(self, config):
        self.cfg = config
        self.output_dir = config.get("calib_output_dir", "calibration/results")
        self.npz_file = os.path.join(
            self.output_dir, config.get("npz_file", "handeye.npz")
        )
        self.txt_file = os.path.join(
            self.output_dir, config.get("txt_file", "handeye.txt")
        )

    def method_files(self, method):
        base = os.path.join(self.output_dir, f"handeye_{method}")
        return f"{base}.npz", f"{base}.txt"


class HandEyeCalibrationCLI:
    """
    CLI utility for Hand-Eye calibration using config and centralized logger.
    """

    def __init__(self, logger=None):
        Config.load("config.yaml")
        self.cfg = Config.get("handeye")
        self.io = HandEyeCalibrationIO(self.cfg)
        self.logger = logger or Logger.get_logger("cli.handeye_calib")

    def run(self):
        images_dir = self.cfg.get("images_dir", "calibration/handeye_data")
        json_file = os.path.join(images_dir, "poses.json")
        method = self.cfg.get("method", "ALL").upper()

        Rs_g2b, ts_g2b = JSONPoseLoader.load_poses(json_file)
        Rs_t2c, ts_t2c = JSONPoseLoader.load_poses(json_file)

        calibrator = HandEyeCalibrator(logger=self.logger)
        for Rg, tg, Rc, tc in zip(Rs_g2b, ts_g2b, Rs_t2c, ts_t2c):
            calibrator.add_sample(Rg, tg, Rc, tc)

        if method == "ALL":
            results = calibrator.calibrate_all()
            for meth, (R, t) in results.items():
                npz_path, txt_path = self.io.method_files(meth)
                calibrator.save(NPZHandEyeSaver(), npz_path, R, t)
                calibrator.save(TxtHandEyeSaver(), txt_path, R, t)
                self.logger.info(f"Saved {meth} results to {npz_path}, {txt_path}")
                print(f"{meth}: saved to {npz_path}, {txt_path}")
        else:
            R, t = calibrator.calibrate(method)
            calibrator.save(NPZHandEyeSaver(), self.io.npz_file, R, t)
            calibrator.save(TxtHandEyeSaver(), self.io.txt_file, R, t)
            self.logger.info(
                f"Hand-Eye calibration saved to {self.io.npz_file} and {self.io.txt_file}"
            )
            print(
                f"Hand-Eye calibration saved to {self.io.npz_file} and {self.io.txt_file}"
            )


def main():
    cli = HandEyeCalibrationCLI()
    cli.run()


if __name__ == "__main__":
    main()
