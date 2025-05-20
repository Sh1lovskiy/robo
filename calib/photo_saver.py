"""
Robot pose and image capture module with Charuco board.
"""

import cv2
import numpy as np
import os
import json
import keyboard
import asyncio
from core.control import RobotController
from misc.logger import Logger
import glob 


# --- Charuco Board Parameters ---
CHARUCO_SQUARE_LENGTH = 35.0  # Checker Width [mm]
CHARUCO_MARKER_LENGTH = 26.0  # Marker Size [mm]
CHARUCO_BOARD_SIZE = (13, 19)  # Columns, Rows
CHARUCO_DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)

# --- Logger ---
logger = Logger.get_logger("RobotCalibration")


class PhotoSaver:
    def __init__(
        self,
        robot_controller,
        camera_id=0,
        output_folder="calib_hand/images",
        json_file="poses.json",
    ):
        self.robot = robot_controller
        self.camera_id = camera_id
        self.output_folder = output_folder
        self.json_file = os.path.join(output_folder, json_file)
        
        self.board = cv2.aruco.CharucoBoard(
            size=CHARUCO_BOARD_SIZE,
            squareLength=CHARUCO_SQUARE_LENGTH,
            markerLength=CHARUCO_MARKER_LENGTH,
            dictionary=CHARUCO_DICTIONARY
        )
        self.detector_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(CHARUCO_DICTIONARY, self.detector_params)

        self.cap = None
        self.frame_count = 0
        self.capture_requested = False
        self.exit_requested = False

        os.makedirs(output_folder, exist_ok=True)

        self.start_index = self.find_next_available_index()
        self.frame_count = self.start_index

        logger.info(f"Initialized with Charuco board size {CHARUCO_BOARD_SIZE}")
    def find_next_available_index(self):
        """Ищет максимальный индекс среди существующих файлов и записей в JSON."""
        left_files = glob.glob(os.path.join(self.output_folder, "left_*.png"))
        right_files = glob.glob(os.path.join(self.output_folder, "right_*.png"))

        existing_indices = set()

        for f in left_files + right_files:
            base_name = os.path.basename(f)
            if "_" in base_name:
                try:
                    idx = int(base_name.split("_")[1].split(".")[0])
                    existing_indices.add(idx)
                except (ValueError, IndexError):
                    continue

        if os.path.exists(self.json_file):
            try:
                with open(self.json_file, "r") as f:
                    data = json.load(f)
                    for key in data.keys():
                        if "stereo_pose_" in key:
                            try:
                                idx = int(key.split("_")[2])
                                existing_indices.add(idx)
                            except (ValueError, IndexError):
                                continue
            except json.JSONDecodeError:
                logger.warning("JSON file is corrupted or empty")

        if existing_indices:
            next_idx = max(existing_indices) + 1
            logger.info(f"Found existing data up to index {max(existing_indices)}, starting from {next_idx}")
            return next_idx
        else:
            logger.info("No previous data found. Starting from index 0")
            return 0
    async def start_camera(self):
        """Initialize camera with basic parameters."""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            logger.error(f"Could not open camera {self.camera_id}")
            raise ValueError(f"Camera {self.camera_id} not available")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        logger.info("Camera started")
        return True

    def setup_key_controls(self):
        keyboard.add_hotkey("space", self.request_capture)
        keyboard.add_hotkey("esc", self.request_exit)
        logger.info("Controls: SPACE to capture, ESC to exit")

    def request_capture(self):
        self.capture_requested = True

    def request_exit(self):
        self.exit_requested = True

    def detect_charuco(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        focal_length = width * 0.8
        camera_matrix = np.array([
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1]
        ], dtype=np.float32)
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)

        corners, ids, rejected = self.aruco_detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, CHARUCO_MARKER_LENGTH, camera_matrix, dist_coeffs
            )

            for i in range(len(rvecs)):
                cv2.drawFrameAxes(
                    frame, camera_matrix, dist_coeffs,
                    rvecs[i], tvecs[i], CHARUCO_MARKER_LENGTH
                )

            try:
                ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    corners, ids, gray, self.board
                )
                return ret, charuco_corners, charuco_ids
            except cv2.error as e:
                logger.warning(f"Charuco interpolation failed: {e}")
                return False, None, None

        return False, None, None

    async def capture_and_process(self, frame_left, frame_right):
        current_tcp_pose = self.robot.get_current_pose()
        current_joint_pose = self.robot.get_current_joint_pose()

        if current_tcp_pose is None:
            logger.error("No robot TCP pose")
            return False

        left_filename = os.path.join(self.output_folder, f"left_{self.frame_count:03d}.png")
        right_filename = os.path.join(self.output_folder, f"right_{self.frame_count:03d}.png")

        cv2.imwrite(left_filename, frame_left)
        cv2.imwrite(right_filename, frame_right)

        pose_data = {
            f"stereo_pose_{self.frame_count:03d}": {
                "robot_tcp_pose": current_tcp_pose.tolist() if hasattr(current_tcp_pose, 'tolist') else current_tcp_pose,
                "robot_joints_pose": current_joint_pose.tolist() if hasattr(current_joint_pose, 'tolist') else current_joint_pose,
            }
        }

        existing_data = {}
        if os.path.exists(self.json_file):
            try:
                with open(self.json_file, "r") as f:
                    data = f.read().strip()
                    if data:
                        existing_data = json.loads(data)
            except (json.JSONDecodeError, FileNotFoundError):
                logger.warning("Failed to read JSON file — starting fresh")

        existing_data.update(pose_data)

        with open(self.json_file, "w") as f:
            json.dump(existing_data, f, indent=2)

        logger.info(f"Captured stereo pair {self.frame_count}")
        self.frame_count += 1

        return True
    async def run_capture_ui(self):
        if not await self.start_camera():
            return False

        self.setup_key_controls()
        logger.info("Capture UI ready - SPACE to capture, ESC to exit")

        try:
            while not self.exit_requested:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Frame capture failed")
                    await asyncio.sleep(0.1)
                    continue

                height, width = frame.shape[:2]
                half_width = width // 2
                frame_left = frame[:, :half_width]
                frame_right = frame[:, half_width:]

                display_frame_left = frame_left.copy()
                ret_left, corners_left, ids_left = self.detect_charuco(display_frame_left)

                display_frame_right = frame_right.copy()
                ret_right, corners_right, ids_right = self.detect_charuco(display_frame_right)

                combined_display = np.hstack((display_frame_left, display_frame_right))
                
                current_tcp_pose = self.robot.get_current_pose()
                tcp_text = f"TCP: {np.round(current_tcp_pose, 3)}" if current_tcp_pose is not None else "TCP: N/A"
                
                cv2.putText(combined_display, f"Frames: {self.frame_count}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(combined_display, tcp_text, (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(combined_display, "SPACE: Capture | ESC: Exit", 
                            (10, combined_display.shape[0]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                status_left = "DETECTED" if ret_left else "NOT DETECTED"
                status_right = "DETECTED" if ret_right else "NOT DETECTED"
                color_left = (0, 255, 0) if ret_left else (0, 0, 255)
                color_right = (0, 255, 0) if ret_right else (0, 0, 255)

                cv2.putText(combined_display, f"L: {status_left}", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_left, 2)
                cv2.putText(combined_display, f"R: {status_right}", (half_width + 10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_right, 2)

                cv2.imshow("Stereo Calibration", combined_display)
                cv2.waitKey(1)

                if self.capture_requested:
                    self.capture_requested = False

                    await self.capture_and_process(frame_left, frame_right)
                    # else:
                    #     logger.warning("Cannot capture - no detection in either view")

                await asyncio.sleep(0.03)

        finally:
            if self.cap and self.cap.isOpened():
                self.cap.release()
            cv2.destroyAllWindows()
            keyboard.unhook_all()
            logger.info(f"Finished with {self.frame_count} stereo poses")

async def main():
    robot = RobotController()
    system = PhotoSaver(robot_controller=robot)
    await system.run_capture_ui()


if __name__ == "__main__":
    asyncio.run(main())