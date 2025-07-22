import json
import numpy as np
import pprint
from scipy.spatial.transform import Rotation as R
from utils.logger import Logger


def check_poses_for_handeye(json_path, logger=None, verbose=True):
    with open(json_path, "r") as f:
        data = json.load(f)

    fails = []
    nan_fail = []
    det_fail = []
    ortho_fail = []
    reflection_fail = []
    reflection_like_fail = []
    for idx, v in data.items():
        coords = v["tcp_coords"]
        x, y, z, rx, ry, rz = coords

        for a, name in zip([rx, ry, rz], "rx ry rz".split()):
            if not -180 <= a <= 180:
                fails.append((idx, f"{name}={a} out of [-180, 180]"))
                logger.warning(f"Pose {idx}: {name}={a:.6f} out of [-180, 180]")

        t = np.array([x, y, z], float) / 1000.0
        try:
            Rmat = R.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()
        except Exception as ex:
            fails.append((idx, f"Bad Euler angles: {ex}"))
            logger.error(f"Pose {idx}: Bad Euler angles: {ex}")
            continue

        if np.any(~np.isfinite(Rmat)):
            nan_fail.append(idx)
            fails.append((idx, "Rotation matrix has NaN or inf"))
            logger.error(f"Pose {idx}: Rotation matrix has NaN or inf")
        det = np.linalg.det(Rmat)
        trace = np.trace(Rmat)
        ortho_error = np.linalg.norm(Rmat.T @ Rmat - np.eye(3))

        logger.info(
            f"Pose {idx}: trace={trace:.6f}, det={det:.6f}, ortho_error={ortho_error:.2e}"
        )

        if det < 0:
            # orthogonal but NOT rotations
            reflection_fail.append((idx, det))
            fails.append((idx, f"reflection detected: determinant={det}"))
            logger.error(f"Pose {idx}: REFLECTION detected!")
        elif np.isclose(det, 1.0, atol=1e-6) and trace < -0.999:
            # rotation exactly 180 degrees around some axis
            reflection_like_fail.append((idx, trace))
            fails.append((idx, f"reflection-like rotation: trace={trace}"))
            logger.error(f"Pose {idx}: REFLECTION-LIKE rotation!")
        elif not np.isclose(det, 1.0, atol=1e-6):
            # for a correct rotation matrix from SO(3) it is necessary
            # that the determinant must be equal to 1
            det_fail.append((idx, det))
            fails.append((idx, f"determinant={det}"))
            logger.error(f"Pose {idx}: determinant={det:.8f} (should be ~1.0)")

        if ortho_error > 1e-6:
            ortho_fail.append((idx, ortho_error))
            fails.append((idx, f"ortho_error={ortho_error}"))
            logger.error(f"Pose {idx}: ortho_error={ortho_error:.2e} (should be <1e-6)")

    if verbose:
        if fails:
            logger.warning("Not all poses are valid")
        else:
            logger.info("All poses are valid for HandEye calib")

    return {
        "fails": fails,
        "det_fail": det_fail,
        "reflection_fail": reflection_fail,
        "reflection_like_fail": reflection_like_fail,
        "ortho_fail": ortho_fail,
        "nan_fail": nan_fail,
    }


logger = Logger.get_logger("calibration.pose_check")
result = check_poses_for_handeye("calib/poses_20250721_170522.json", logger)
logger.info(f"error results:\n{pprint.pformat(result)}")
