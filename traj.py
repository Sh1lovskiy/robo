"""
Robot Motion Controller with Object Tracking

This script moves a robot from a starting position to a target position
while ensuring a part remains in the stereo camera frame.

It uses:
- RobotController for robot motion control
- YOLO model for object detection
- OpenCV for camera feed processing
- Incremental movements with visual feedback
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Dict, Optional

# Import our robot controller
from misc.logger import Logger
from config.constants import DEFAULT_IP
from core.control import RobotController

# Set up logging
logger = Logger.get_logger("tracking_motion", json_format=True)

# Define positions
START_POSITION = [-105, -490, 150, -120, 10, 51]  # Current position
TARGET_POSITION = [-518, 111, 170, -131, 5, -163]  # Target position

# Camera and detection parameters
CAMERA_INDEX = 0  # Update as needed
CONFIDENCE_THRESHOLD = 0.5
YOLO_WEIGHTS_PATH = (
    r"C:\Users\Алексей\Documents\kinematics\funetune\runs\detect\train2\weights\best.pt"
)
FRAME_WIDTH = 1280  # Total width for stereo camera
FRAME_HEIGHT = 720
STEREO_SPLIT_X = FRAME_WIDTH // 2  # Split point for stereo images

# Motion parameters
MAX_STEP_SIZE = 10.0  # Maximum step size in mm or degrees
MIN_STEP_SIZE = 2.0  # Minimum step size in mm or degrees
MOTION_DELAY = 0.5  # Delay between movements in seconds
MAX_ITERATIONS = 200  # Safety limit for iterations


class VisualGuidedMotion:
    def __init__(
        self, start_pos: List[float], target_pos: List[float], ip: str = DEFAULT_IP
    ):
        """
        Initialize the visual guided motion controller.

        Args:
            start_pos: Starting position [x, y, z, Rx, Ry, Rz]
            target_pos: Target position [x, y, z, Rx, Ry, Rz]
            ip: Robot IP address
        """
        # Get the actual current position from the robot
        self.robot = RobotController(ip_address=ip)
        if not self.robot.connected or not self.robot.initialize():
            raise ConnectionError("Failed to connect to the robot")

        # Update start position with actual robot position
        self.actual_start_pos = self.robot.get_current_pose()
        if self.actual_start_pos:
            logger.info(
                f"Using actual robot position as start: {self.actual_start_pos}"
            )
            self.start_pos = self.actual_start_pos
        else:
            logger.warning(
                f"Could not get actual robot position, using provided start: {start_pos}"
            )
            self.start_pos = start_pos

        self.target_pos = target_pos
        self.current_pos = self.start_pos.copy()
        self.distances = self._calculate_distances()

        # Initialize camera
        self.camera = cv2.VideoCapture(CAMERA_INDEX)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        # Load YOLO model
        self.model = self._load_yolo_model()

        # Status tracking
        self.part_visible = False
        self.last_detection = None
        self.iterations = 0

        # Visualization window
        self.show_visualization = True

        logger.info("Visual guided motion controller initialized")

    def _load_yolo_model(self):
        """Load the YOLO model with custom weights."""
        # Import necessary libraries locally to avoid scoping issues
        import torch
        import cv2
        import pandas as pd

        try:
            logger.info(f"Attempting to load YOLO model from: {YOLO_WEIGHTS_PATH}")

            # Check if CUDA is available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {device}")

            # Try the most reliable method first - using ultralytics YOLO
            try:
                from ultralytics import YOLO

                model = YOLO(YOLO_WEIGHTS_PATH)
                logger.info("YOLO model loaded successfully using ultralytics")
                return model
            except Exception as e1:
                logger.warning(f"Failed to load with ultralytics YOLO: {e1}")

            # Try with torch.load and weights_only=False (but be careful with this)
            try:
                logger.info("Attempting to load with torch.load and weights_only=False")
                model = torch.load(
                    YOLO_WEIGHTS_PATH, map_location=device, weights_only=False
                )
                logger.info("YOLO model loaded successfully with torch.load")
                return model
            except Exception as e2:
                logger.warning(f"Failed to load with torch.load: {e2}")

            # Try with safe_globals context manager
            try:
                logger.info("Attempting to load with safe_globals context manager")
                # This might fail if ultralytics is not installed or the class is not found
                try:
                    from ultralytics.nn.tasks import DetectionModel
                    import torch.serialization

                    with torch.serialization.safe_globals([DetectionModel]):
                        model = torch.load(YOLO_WEIGHTS_PATH, map_location=device)
                        logger.info("YOLO model loaded successfully with safe_globals")
                    return model
                except ImportError:
                    logger.warning("Could not import DetectionModel from ultralytics")
            except Exception as e3:
                logger.warning(f"Failed to load with safe_globals: {e3}")

            # If all methods fail, fallback to a simpler approach - OpenCV DNN
            logger.info(
                "All YOLO loading methods failed. Switching to OpenCV DNN detector"
            )

            # Create a simplified detector using OpenCV
            class SimpleDetector:
                def __init__(self):
                    self.conf = CONFIDENCE_THRESHOLD
                    logger.info("Initialized SimpleDetector with OpenCV")

                def __call__(self, frame):
                    # Use OpenCV for basic object detection
                    # Convert BGR to grayscale
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Apply Gaussian blur
                    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

                    # Use adaptive thresholding
                    thresh = cv2.adaptiveThreshold(
                        blurred,
                        255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY_INV,
                        11,
                        2,
                    )

                    # Find contours
                    contours, _ = cv2.findContours(
                        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )

                    # Filter contours by area and create a mock YOLO result
                    min_area = 500  # Minimum contour area to consider
                    results = []

                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area > min_area:
                            x, y, w, h = cv2.boundingRect(contour)
                            results.append(
                                {
                                    "xmin": float(x),
                                    "ymin": float(y),
                                    "xmax": float(x + w),
                                    "ymax": float(y + h),
                                    "confidence": 1.0,
                                    "class": 0,
                                }
                            )

                    # Mock the YOLO results structure
                    class MockResults:
                        def __init__(self, detections):
                            self.detections = detections

                        def pandas(self):
                            class MockPandasResult:
                                def __init__(self, detections):
                                    self.xyxy = [pd.DataFrame(detections)]

                            return MockPandasResult(self.detections)

                    return MockResults(results)

            return SimpleDetector()

        except Exception as e:
            logger.error(f"All detection methods failed: {e}")

            # Last resort - return a simple detector that always returns nothing
            class NoOpDetector:
                def __init__(self):
                    self.conf = CONFIDENCE_THRESHOLD
                    logger.info(
                        "Initialized NoOpDetector (always returns empty results)"
                    )

                def __call__(self, frame):
                    class EmptyResults:
                        def pandas(self):
                            class EmptyPandasResult:
                                def __init__(self):
                                    self.xyxy = [pd.DataFrame([])]

                            return EmptyPandasResult()

                    return EmptyResults()

            return NoOpDetector()

    def _calculate_distances(self) -> List[float]:
        """Calculate the total distance to move for each component."""
        return [t - s for s, t in zip(self.start_pos, self.target_pos)]

    def _calculate_step_sizes(self) -> List[float]:
        """Calculate step sizes for each component based on remaining distance."""
        # Calculate remaining distances
        remaining = [t - c for c, t in zip(self.current_pos, self.target_pos)]

        # Find the maximum distance component
        max_distance = max(abs(d) for d in remaining)
        if max_distance < MIN_STEP_SIZE:
            # We're close enough to the target
            return [0] * len(remaining)

        # Scale step sizes so the maximum is MAX_STEP_SIZE
        scale_factor = min(MAX_STEP_SIZE / max_distance, 1.0) if max_distance > 0 else 0
        steps = [d * scale_factor for d in remaining]

        # Ensure no step is smaller than minimum (unless it's zero)
        for i, step in enumerate(steps):
            if 0 < abs(step) < MIN_STEP_SIZE:
                steps[i] = MIN_STEP_SIZE if step > 0 else -MIN_STEP_SIZE

        return steps

    def process_stereo_frame(self, frame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split the stereo camera frame into left and right images.

        Args:
            frame: The combined stereo frame

        Returns:
            Tuple of (left_image, right_image)
        """
        left_img = frame[:, :STEREO_SPLIT_X]
        right_img = frame[:, STEREO_SPLIT_X:]
        return left_img, right_img

    def detect_part(self, frame) -> Optional[Dict]:
        """
        Detect the part in the frame using YOLO.

        Args:
            frame: Camera frame

        Returns:
            Detection info or None if not detected
        """
        try:
            # Run inference
            results = self.model(frame)

            # Process ultralytics YOLO results format
            try:
                # Check if there are any detections
                if len(results) > 0 and hasattr(results[0], "boxes"):
                    boxes = results[0].boxes

                    # Check if we have any boxes
                    if (
                        len(boxes) > 0
                        and boxes.xyxy is not None
                        and len(boxes.xyxy) > 0
                    ):
                        # Get the highest confidence detection
                        confidences = boxes.conf.cpu().numpy()
                        best_idx = confidences.argmax()
                        best_box = boxes.xyxy[best_idx].cpu().numpy()
                        confidence = float(confidences[best_idx])

                        # Skip if confidence is too low
                        if confidence < CONFIDENCE_THRESHOLD:
                            logger.debug(f"Detection confidence too low: {confidence}")
                            return None

                        # Extract box coordinates
                        x1, y1, x2, y2 = best_box
                        class_id = (
                            int(boxes.cls[best_idx].item())
                            if hasattr(boxes, "cls")
                            else 0
                        )

                        # Calculate center point and size
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1

                        detection_info = {
                            "bbox": (x1, y1, x2, y2),
                            "center": (center_x, center_y),
                            "size": (width, height),
                            "confidence": confidence,
                            "class": class_id,
                        }

                        logger.debug(f"Part detected: {detection_info}")
                        return detection_info

                logger.debug("No part detected in frame")
                return None

            except Exception as e:
                logger.error(f"Error processing detection results: {e}")
                logger.error(f"Results type: {type(results)}")
                if hasattr(results, "__len__"):
                    logger.error(f"Results length: {len(results)}")
                if len(results) > 0:
                    logger.error(f"First result type: {type(results[0])}")
                    if hasattr(results[0], "boxes"):
                        logger.error(f"Boxes attribute exists")
                        if hasattr(results[0].boxes, "xyxy"):
                            logger.error(
                                f"Boxes has xyxy attribute with shape: {results[0].boxes.xyxy.shape if hasattr(results[0].boxes.xyxy, 'shape') else 'unknown'}"
                            )
                return None

        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return None

    def check_part_visibility(self) -> bool:
        """
        Check if the part is visible in the camera frame.

        Returns:
            True if part is visible, False otherwise
        """
        ret, frame = self.camera.read()
        if not ret:
            logger.error("Failed to capture frame from camera")
            return False

        # Split stereo frame
        left_img, right_img = self.process_stereo_frame(frame)

        # Try to detect in both left and right frames
        left_detection = self.detect_part(left_img)
        right_detection = self.detect_part(right_img)

        # Visualize the camera feed and detections
        if self.show_visualization:
            try:
                # Create visualization images (copies to draw on)
                left_viz = left_img.copy()
                right_viz = right_img.copy()

                # Draw detection box on left image if detected
                if left_detection:
                    x1, y1, x2, y2 = left_detection["bbox"]
                    confidence = left_detection["confidence"]
                    # Green box for detection
                    cv2.rectangle(
                        left_viz, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
                    )
                    # Display confidence
                    cv2.putText(
                        left_viz,
                        f"Conf: {confidence:.2f}",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

                # Draw detection box on right image if detected
                if right_detection:
                    x1, y1, x2, y2 = right_detection["bbox"]
                    confidence = right_detection["confidence"]
                    # Green box for detection
                    cv2.rectangle(
                        right_viz,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 0),
                        2,
                    )
                    # Display confidence
                    cv2.putText(
                        right_viz,
                        f"Conf: {confidence:.2f}",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

                # Add motion information
                # Calculate target motion vector
                current_xyz = self.current_pos[:3]
                target_xyz = self.target_pos[:3]

                # Display positions and progress
                info_text = [
                    f"Current: [{', '.join([f'{p:.1f}' for p in self.current_pos[:3]])}]",
                    f"Target: [{', '.join([f'{p:.1f}' for p in self.target_pos[:3]])}]",
                    f"Iteration: {self.iterations}",
                ]

                # Calculate remaining distance and progress percentage
                remaining_distance = sum(
                    abs(t - c) for c, t in zip(self.current_pos, self.target_pos)
                )
                total_distance = sum(abs(d) for d in self.distances)
                progress = (
                    (1 - remaining_distance / total_distance) * 100
                    if total_distance > 0
                    else 100
                )

                info_text.append(f"Progress: {progress:.1f}%")

                # Add info text to left image
                for i, text in enumerate(info_text):
                    cv2.putText(
                        left_viz,
                        text,
                        (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )

                # Draw detection status
                status_text = "Part Status: " + (
                    "DETECTED"
                    if (left_detection or right_detection)
                    else "NOT DETECTED"
                )
                cv2.putText(
                    right_viz,
                    status_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0) if (left_detection or right_detection) else (0, 0, 255),
                    2,
                )

                # Combine left and right images for display
                # Ensure both images have the same height
                if left_viz.shape[0] != right_viz.shape[0]:
                    # Resize to match heights
                    h = min(left_viz.shape[0], right_viz.shape[0])
                    left_viz = cv2.resize(
                        left_viz, (int(left_viz.shape[1] * h / left_viz.shape[0]), h)
                    )
                    right_viz = cv2.resize(
                        right_viz, (int(right_viz.shape[1] * h / right_viz.shape[0]), h)
                    )

                # Combine the images
                combined_img = np.hstack((left_viz, right_viz))

                # Resize if too large for screen
                screen_width = 1600  # Max reasonable screen width
                if combined_img.shape[1] > screen_width:
                    scale = screen_width / combined_img.shape[1]
                    combined_img = cv2.resize(combined_img, None, fx=scale, fy=scale)

                # Show the combined image
                cv2.namedWindow("Robot Camera View", cv2.WINDOW_NORMAL)
                cv2.imshow("Robot Camera View", combined_img)
                cv2.waitKey(1)  # Display for at least 1ms

                logger.debug("Visualization updated")
            except Exception as e:
                logger.error(f"Error in visualization: {e}")

        # Update last detection
        if left_detection:
            self.last_detection = ("left", left_detection)
            return True
        elif right_detection:
            self.last_detection = ("right", right_detection)
            return True
        else:
            return False

    def move_incrementally(self) -> bool:
        """
        Move the robot incrementally toward the target position.

        Returns:
            True if the movement was successful, False otherwise
        """
        # Save current position before moving
        previous_position = self.current_pos.copy()

        # Check if part is visible
        part_visible_before = self.check_part_visibility()
        if part_visible_before:
            logger.info("Part is visible before movement")
            if self.last_detection:
                camera_side, detection = self.last_detection
                # Store detection location for comparison after move
                self.previous_detection = self.last_detection
        else:
            logger.info("Part is not visible before movement")

        # Calculate step sizes toward target
        step_sizes = self._calculate_step_sizes()

        # If all step sizes are zero, we've reached the target
        if all(abs(step) < 0.001 for step in step_sizes):
            logger.info("Reached target position")
            return False

        # Modify the step based on part visibility
        if part_visible_before and self.last_detection:
            camera_side, detection = self.last_detection
            center_x, center_y = detection["center"]

            # Get image dimensions
            if camera_side == "left":
                img_width = STEREO_SPLIT_X
            else:
                img_width = STEREO_SPLIT_X
            img_height = FRAME_HEIGHT

            # Calculate normalized position (-1 to 1, where 0 is center)
            norm_x = (center_x / img_width) * 2 - 1
            norm_y = (center_y / img_height) * 2 - 1

            logger.info(
                f"Part center normalized position: ({norm_x:.2f}, {norm_y:.2f})"
            )

            # Adjust vertical position (Z-axis) to keep part centered
            z_adjustment = -norm_y * 3.0  # Scale factor for adjustment

            # Calculate area coverage of part in frame
            width, height = detection["size"]
            area_ratio = (width * height) / (img_width * img_height)
            logger.info(f"Part area coverage: {area_ratio:.2%}")

            # Add adjustments to step - prioritize keeping part in frame
            step_sizes[2] += z_adjustment

            # If part is close to edge of frame, slow down movement in that direction
            if abs(norm_x) > 0.6:  # Part is near left/right edge
                # Reduce x and y movement to keep part in frame
                x_reduction = 0.5  # Reduce to 50% of original step
                step_sizes[0] *= x_reduction
                step_sizes[1] *= x_reduction
                logger.info(f"Reduced X/Y movement (edge of frame)")

            logger.info(f"Adjusted step sizes: {step_sizes}")

        # Calculate new position
        new_position = [c + s for c, s in zip(self.current_pos, step_sizes)]

        # Move the robot
        success = self.robot.move_linear(new_position)
        if not success:
            logger.error("Failed to move robot")
            return False

        # Update current position
        self.current_pos = new_position
        logger.info(f"Moved to {new_position}")

        # Check if part is still visible after move
        time.sleep(MOTION_DELAY)  # Wait for motion to complete
        part_visible_after = self.check_part_visibility()

        # Handle case where part is lost from view
        if part_visible_before and not part_visible_after:
            logger.warning("Part lost from view after movement")

            # Try different recovery strategies
            recovery_successful = self._try_recovery_strategies(previous_position)
            if not recovery_successful:
                logger.warning(
                    "All recovery strategies failed, continuing toward target"
                )

            return recovery_successful

        # If we found the part after not having it before, that's great!
        if not part_visible_before and part_visible_after:
            logger.info("Part came into view after movement - success!")

        return True

    def _try_recovery_strategies(self, previous_position):
        """
        Try different strategies to recover the part when it's lost from view.

        Args:
            previous_position: The position before the part was lost

        Returns:
            bool: True if recovery was successful, False otherwise
        """
        logger.info("Attempting recovery strategies to find part again")

        # Strategy 1: Return to previous position
        logger.info("Strategy 1: Return to previous position")
        if not self.robot.move_linear(previous_position):
            logger.error("Failed to return to previous position")
            return False

        self.current_pos = previous_position.copy()
        time.sleep(MOTION_DELAY)

        # Check if part is visible after returning
        if self.check_part_visibility():
            logger.info("Part found after returning to previous position")

            # Strategy 2: Try moving up a bit
            logger.info("Strategy 2: Adjusting height to keep part in view")
            test_positions = [
                # Try moving up
                [p + (5 if i == 2 else 0) for i, p in enumerate(previous_position)],
                # Try moving down
                [p + (-5 if i == 2 else 0) for i, p in enumerate(previous_position)],
                # Try adjusting roll (Rx)
                [p + (5 if i == 3 else 0) for i, p in enumerate(previous_position)],
                # Try adjusting pitch (Ry)
                [p + (5 if i == 4 else 0) for i, p in enumerate(previous_position)],
            ]

            # Try each test position
            for i, test_pos in enumerate(test_positions):
                logger.info(f"Testing recovery position {i+1}: {test_pos}")

                if not self.robot.move_linear(test_pos):
                    logger.error(f"Failed to move to test position {i+1}")
                    continue

                self.current_pos = test_pos.copy()
                time.sleep(MOTION_DELAY)

                # Check if part is visible
                if self.check_part_visibility():
                    logger.info(f"Part found after trying test position {i+1}")
                    return True

                # Return to previous position if part not found
                if not self.robot.move_linear(previous_position):
                    logger.error("Failed to return to previous position")
                    return False

                self.current_pos = previous_position.copy()
                time.sleep(MOTION_DELAY)

            # If we get here, none of the test positions worked
            # Try Strategy 3: Make a smaller move toward the target
            logger.info("Strategy 3: Making a smaller move toward target")

            # Calculate a smaller step (25% of original)
            small_step_sizes = [
                (t - p) * 0.25 for p, t in zip(previous_position, self.target_pos)
            ]
            small_step_pos = [
                p + s for p, s in zip(previous_position, small_step_sizes)
            ]

            if not self.robot.move_linear(small_step_pos):
                logger.error("Failed to make smaller step")
                return False

            self.current_pos = small_step_pos.copy()
            time.sleep(MOTION_DELAY)

            # Check if part is visible
            if self.check_part_visibility():
                logger.info("Part found after making smaller step")
                return True

            # Return to previous position if part not found
            if not self.robot.move_linear(previous_position):
                logger.error("Failed to return to previous position")
                return False

            self.current_pos = previous_position.copy()

            # As a last resort, try a completely different approach angle
            logger.info("Strategy 4: Trying different approach angle")

            # Calculate position with modified orientation
            angle_pos = previous_position.copy()
            # Modify Rx, Ry by 10 degrees in alternating directions
            angle_pos[3] += 10  # Rx
            angle_pos[4] -= 10  # Ry

            if not self.robot.move_linear(angle_pos):
                logger.error("Failed to move to different angle")
                return False

            self.current_pos = angle_pos.copy()
            time.sleep(MOTION_DELAY)

            # Check if part is visible
            if self.check_part_visibility():
                logger.info("Part found after trying different angle")
                return True

            # Return to previous position
            if not self.robot.move_linear(previous_position):
                logger.error("Failed to return to previous position")
                return False

            self.current_pos = previous_position.copy()

        logger.warning("All recovery strategies failed")
        return False

    def run(self):
        """
        Main control loop to move from start to target position.
        """
        logger.info(f"Starting motion from {self.start_pos} to {self.target_pos}")

        # Initial check for part - but we'll proceed even if not visible
        self.check_part_visibility()
        logger.info("Beginning motion sequence - press 'q' to quit at any time")

        # Main motion loop
        while True:
            self.iterations += 1

            # Check if we've reached the maximum iterations (safety limit)
            if self.iterations >= MAX_ITERATIONS:
                logger.warning(f"Reached maximum iterations ({MAX_ITERATIONS})")
                break

            # Check if we've reached the target
            remaining_distance = sum(
                abs(t - c) for c, t in zip(self.current_pos, self.target_pos)
            )
            if remaining_distance < 0.1:  # Threshold for considering target reached
                logger.info("Target position reached")
                break

            # Move incrementally
            success = self.move_incrementally()
            if not success:
                logger.warning("Movement unsuccessful, trying a smaller step")
                # Reduce step size and try again?
                global MAX_STEP_SIZE
                MAX_STEP_SIZE *= 0.8
                if MAX_STEP_SIZE < MIN_STEP_SIZE:
                    logger.error("Step size too small, cannot proceed")
                    break

            # Display status
            progress = (
                1 - remaining_distance / sum(abs(d) for d in self.distances)
            ) * 100
            logger.info(f"Progress: {progress:.1f}% - Current pos: {self.current_pos}")

            # Check for manual exit command (press 'q' to quit)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("User requested exit")
                break

        # Keep the window open until user presses a key after completion
        if self.show_visualization:
            logger.info("Motion complete. Press any key to exit...")
            while True:
                if cv2.waitKey(100) != -1:
                    break

        return True

    def shutdown(self):
        """Clean up resources."""
        logger.info("Shutting down visual guided motion controller")
        if self.show_visualization:
            cv2.destroyAllWindows()
        self.camera.release()
        self.robot.shutdown()


def main():
    try:
        # Initialize the visual guided motion controller
        # Define start and target positions
        START_POSITION = [-105, -490, 150, -120, 10, 51]  # Initial position
        TARGET_POSITION = [-518, 111, 170, -131, 5, -163]  # Target position

        # Test camera and visualization first
        try:
            # Open a simple window to verify OpenCV is working
            cap = cv2.VideoCapture(CAMERA_INDEX)
            if not cap.isOpened():
                logger.error("Cannot open camera")
                return

            # Read a test frame
            ret, frame = cap.read()
            if not ret:
                logger.error("Cannot read from camera")
                cap.release()
                return

            # Try displaying a test window
            cv2.namedWindow("Camera Test", cv2.WINDOW_NORMAL)
            cv2.imshow("Camera Test", frame)
            cv2.waitKey(1000)  # Show for 1 second
            logger.info("Camera and display test successful")
            cap.release()
            cv2.destroyAllWindows()
        except Exception as e:
            logger.error(f"Camera display test failed: {e}")
            # Continue anyway

        # Initialize the motion controller
        motion_controller = VisualGuidedMotion(START_POSITION, TARGET_POSITION)

        # Run the motion sequence
        motion_controller.run()

    except Exception as e:
        logger.error(f"Error in visual guided motion: {e}")
        import traceback

        logger.error(traceback.format_exc())
    finally:
        # Clean up
        if "motion_controller" in locals():
            motion_controller.shutdown()
        # Make sure all OpenCV windows are closed
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
