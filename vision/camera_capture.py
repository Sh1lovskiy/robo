import cv2
import json
import os
from datetime import datetime
from typing import Dict, Tuple
from core.control import RobotController
from misc.logger import Logger
from config.constants import CAMERA_SHIFT


class CameraCapture:
    """Robot-camera synchronization tool for capturing images with corresponding robot coordinates.

    Features:
    - Real-time camera preview
    - Space-triggered image capture
    - Automatic coordinate logging with timestamps
    - Non-destructive JSON data appending
    - Drag teach mode control via keyboard
    """

    def __init__(self):
        """Initialize camera, robot connection, and data storage."""
        self.logger = Logger.get_logger("camera_capture", json_format=True)
        self.camera = cv2.VideoCapture(0)
        self.robot = RobotController()
        self.output_dir = "vision/imgs_fix"
        self.coords_file = "vision/imgs_fix/coordinates.json"

        os.makedirs(self.output_dir, exist_ok=True)
        self._init_coords_file()
        self.logger.info("System initialized")

    def _init_coords_file(self) -> None:
        """Ensure coordinates file exists with proper structure."""
        if not os.path.exists(self.coords_file):
            with open(self.coords_file, "w") as f:
                json.dump({}, f)
            self.logger.debug("Created new coordinates file")

    def _generate_timestamp(self) -> str:
        """Create unique filename timestamp.

        Returns:
            str: Formatted timestamp (YYYYMMDD_HHMMSS_µs)
        """
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    def capture_image(self) -> Tuple[cv2.Mat, str]:
        """Capture single frame from camera.

        Returns:
            Tuple[Mat, str]: (image matrix, timestamp)
        Raises:
            RuntimeError: On camera failure
        """
        ret, frame = self.camera.read()
        if not ret:
            self.logger.error("Camera frame capture failed")
            raise RuntimeError("Camera acquisition error")
        return frame, self._generate_timestamp()

    def _load_existing_coords(self) -> Dict:
        """Read current coordinate database.

        Returns:
            Dict: Existing coordinate records
        """
        with open(self.coords_file, "r") as f:
            return json.load(f)

    def save_data(self, img: cv2.Mat, timestamp: str, tcp_coords: list) -> None:
        """Persist image and coordinates to storage.

        Args:
            img: OpenCV image matrix
            timestamp: Unique identifier
            coords: Robot position data
        """
        camera_coords = tcp_coords.copy()
        camera_coords[0] += CAMERA_SHIFT[0]
        camera_coords[1] += CAMERA_SHIFT[1]
        camera_coords[2] += CAMERA_SHIFT[2]
        # Save image
        img_path = os.path.join(self.output_dir, f"img_{timestamp}.png")
        cv2.imwrite(img_path, img)

        # Update coordinates
        data = self._load_existing_coords()
        data[f"img_{timestamp}.png"] = {
            "tcp_coords": tcp_coords,
            "camera_coords": camera_coords,
        }

        with open(self.coords_file, "w") as f:
            json.dump(data, f, indent=4)

        self.logger.info(
            f"Saved {img_path} with TCP coordinates {tcp_coords} and camera coordinates {camera_coords}"
        )

    def run(self) -> None:
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    self.logger.error("Camera feed interrupted")
                    break

                cv2.imshow("Camera Feed", frame)

                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    break
                elif key == 32:  # SPACE
                    self._process_capture()
        finally:
            self._shutdown()

    def _process_capture(self) -> None:
        """Handle single capture sequence."""
        try:
            # Ensure drag teach mode is disabled before capturing
            if not self._disable_drag_teach_mode():
                self.logger.warning(
                    "Failed to disable drag teach mode. Capture aborted."
                )
                return

            if (coords := self.robot.get_current_pose()) is None:
                return

            img, timestamp = self.capture_image()
            self.save_data(img, timestamp, coords)

        except Exception as e:
            self.logger.error(f"Capture failed: {str(e)}")

    def _enable_drag_teach_mode(self) -> bool:
        """Enable drag teaching mode."""
        if self.robot.enable_drag_teach_mode(True):
            self.logger.info("Drag teaching mode enabled.")
            return True
        else:
            self.logger.error("Failed to enable drag teaching mode.")
            return False

    def _disable_drag_teach_mode(self) -> bool:
        """Disable drag teaching mode."""
        if self.robot.enable_drag_teach_mode(False):
            self.logger.info("Drag teaching mode disabled.")
            return True
        else:
            self.logger.error("Failed to disable drag teaching mode.")
            return False

    def _shutdown(self) -> None:
        """Release system resources."""
        self.camera.release()
        cv2.destroyAllWindows()
        self.robot.shutdown()
        self.logger.info("System shutdown complete")


if __name__ == "__main__":
    try:
        CameraCapture().run()
    except Exception as e:
        Logger.get_logger("camera_capture").critical(f"Fatal error: {str(e)}")
