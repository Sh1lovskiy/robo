"""
Orthogonal Projection Capture Script
Moves the robot to three predefined positions (top, front, side), captures an image at each position using OpenCV, and saves them to disk.
"""

import time
import cv2
import logging
import traceback
from pathlib import Path
from robot.control import RobotController
from misc.logger import Logger, Timer
from config.constants import (
    ORTHOGONAL_POSES,
    IMAGE_NAMES,
    SAVE_PATH,
    CAPTURE_DELAY,
)

logger = Logger.setup_logger(
    "OrthogonalCapture",
    json_format=True,
    json_fields={
        "component": "imaging",
        "operation_type": "orthogonal_capture",
    },
)


def capture_image(cam_index: int = 0):
    """Capture a single frame from the default camera."""
    with Timer("ImageCapture", logger):
        Logger.log_data(
            logger,
            logging.INFO,
            "Attempting to capture image from camera index %s",
            cam_index,
        )
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            Logger.log_data(
                logger,
                logging.ERROR,
                "Camera failed to open",
                extra_fields={"camera_index": cam_index},
            )
            return None
        ret, frame = cap.read()
        cap.release()

        if ret:
            Logger.log_json(
                logger,
                logging.INFO,
                event="image_captured",
                camera_index=cam_index,
                frame_shape=frame.shape if frame is not None else None,
            )
            return frame
        else:
            Logger.log_data(
                logger,
                logging.ERROR,
                "Failed to read frame from camera",
                extra_fields={"camera_index": cam_index},
            )
            return None


def capture_at_pose(robot: RobotController, pose: list, filename: Path) -> bool:
    """
    Move to a pose and capture an image.
    Args:
        robot (RobotController): Robot interface instance.
        pose (list): TCP pose [x, y, z, Rx, Ry, Rz].
        filename (Path): Path to save the captured image.
    Returns:
        bool: True if success, False otherwise.
    """
    with Timer("CaptureAtPose", logger):
        Logger.log_json(
            logger,
            logging.INFO,
            action="move_to_pose",
            pose=pose,
            target_file=str(filename),
        )
        print(pose)
        if not robot.move_linear(pose):
            Logger.log_json(
                logger,
                logging.ERROR,
                action="move_failed",
                pose=pose,
                error="Linear movement failed",
            )
            return False

        Logger.log_data(
            logger, logging.INFO, "Waiting for %s seconds before capture", CAPTURE_DELAY
        )
        time.sleep(CAPTURE_DELAY)

        Logger.log_data(logger, logging.INFO, "Initiating image capture")
        frame = capture_image()
        if frame is None:
            Logger.log_json(
                logger,
                logging.ERROR,
                action="capture_failed",
                error="Image capture returned None",
            )
            return False
        Logger.log_data(logger, logging.INFO, "Saving image to: %s", filename)
        cv2.imwrite(str(filename), frame)
        Logger.log_json(
            logger,
            logging.INFO,
            action="image_saved",
            filename=str(filename),
            image_dimensions=frame.shape if frame is not None else None,
        )

        Logger.log_json(
            logger,
            logging.INFO,
            action="capture_completed",
            pose=pose,
            filename=str(filename),
        )
        return True


def main():
    with Timer("OrthogonalCaptureProcess", logger):
        Logger.log_json(
            logger,
            logging.INFO,
            event="process_started",
            description="Starting Orthogonal Projection Capture Process",
        )

        Logger.log_data(
            logger, logging.INFO, "Creating save directory at %s", SAVE_PATH
        )
        SAVE_PATH.mkdir(parents=True, exist_ok=True)

        Logger.log_data(logger, logging.INFO, "Initializing robot controller")
        robot = RobotController()

        if not robot.connected or not robot.initialize():
            Logger.log_json(
                logger,
                logging.ERROR,
                event="initialization_failed",
                error="Failed to connect to robot",
            )
            return

        try:
            for i, pose in enumerate(ORTHOGONAL_POSES):
                filename = SAVE_PATH / IMAGE_NAMES[i]

                Logger.log_json(
                    logger,
                    logging.INFO,
                    step=i + 1,
                    pose_name=IMAGE_NAMES[i],
                    action="starting_capture",
                )

                # Use timer for each step
                with Timer(f"Step{i+1}", logger):
                    result = capture_at_pose(robot, pose, filename)

                if result:
                    Logger.log_json(
                        logger,
                        logging.INFO,
                        step=i + 1,
                        status="success",
                        pose_name=IMAGE_NAMES[i],
                    )
                else:
                    Logger.log_json(
                        logger,
                        logging.WARNING,
                        step=i + 1,
                        status="issue_encountered",
                        pose_name=IMAGE_NAMES[i],
                    )
        except Exception as e:
            Logger.log_json(logger, logging.ERROR, event="process_error", error=str(e))
            Logger.log_data(
                logger, logging.ERROR, "Error traceback: %s", traceback.format_exc()
            )
        finally:
            Logger.log_data(logger, logging.INFO, "Shutting down robot controller")
            robot.shutdown()
            Logger.log_json(
                logger,
                logging.INFO,
                event="process_completed",
                description="Orthogonal projection capture process complete",
            )


if __name__ == "__main__":
    import logging

    Logger.configure_root_logger(level=logging.WARNING)

    try:
        main()
    except Exception as e:
        Logger.log_json(
            logger,
            logging.CRITICAL,
            event="unhandled_exception",
            error=str(e),
            error_type=type(e).__name__,
        )
        raise
