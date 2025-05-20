"""Part scanning trajectory generation and execution

This module implements a scanning trajectory to capture stereo images
of a part from multiple viewpoints for 3D reconstruction.
"""

import numpy as np
import time
import os
from datetime import datetime
from typing import List, Tuple

from misc.logger import Logger, Timer
from robot.kinematics import QuaternionKinematics
from robot.robot_api import RobotAPI
from config.constants import MotionParameters, CoordinateSystem


class PartScanner:
    """Generate and execute scanning trajectories for 3D part reconstruction"""

    def __init__(self, robot_api: RobotAPI, kinematics: QuaternionKinematics):
        self.logger = Logger.get_logger("part_scanner", json_format=True)
        self.robot_api = robot_api
        self.kinematics = kinematics

        # Camera parameters relative to the end effector
        # Camera is mounted 50mm behind, 130mm above the end effector
        self.camera_offset = np.array([-50, 0, 130])  # mm, in end effector frame

        # Output directories
        self.output_dir = os.path.join(os.getcwd(), "scan_data")
        self.image_dir = os.path.join(self.output_dir, "images")
        self.pose_dir = os.path.join(self.output_dir, "poses")

        # Create output directories if they don't exist
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.pose_dir, exist_ok=True)

        self.logger.info(
            "Part scanner initialized with camera offset: %s", self.camera_offset
        )

    def generate_scanning_poses(
        self,
        part_center: List[float],
        scanning_height: float = 350,
        orbit_radius: float = 50,
        num_points: int = 8,
        min_height: float = 100,
        max_height: float = 600,
    ) -> List[List[float]]:
        """Generate scanning positions around part center

        Args:
            part_center: [x, y, z] center coordinates of the part
            scanning_height: Fixed height for the end effector (mm)
            orbit_radius: Radius around the center for scanning positions (mm)
            num_points: Number of points around the orbit
            min_height: Minimum allowed Z-coordinate (mm)
            max_height: Maximum allowed Z-coordinate (mm)

        Returns:
            List of cartesian poses [x,y,z,rx,ry,rz] ready for robot motion
        """
        self.logger.info(
            "Generating scan positions around center [%s] at height %.1f mm",
            part_center,
            scanning_height,
        )

        part_center = np.array(part_center)
        poses = []

        # Generate evenly spaced angles around the part
        angles = np.linspace(0, 360, num_points, endpoint=False)

        for angle_deg in angles:
            angle_rad = np.radians(angle_deg)

            # XY position on the circular orbit
            x = part_center[0] + orbit_radius * np.cos(angle_rad)
            y = part_center[1] + orbit_radius * np.sin(angle_rad)

            # Fixed height Z position
            z = scanning_height

            # Ensure Z is within allowed range
            if z < min_height:
                self.logger.warning(
                    f"Position at angle {angle_deg}° has Z={z:.1f}mm below minimum. Adjusting to Z={min_height}mm"
                )
                z = min_height
            elif z > max_height:
                self.logger.warning(
                    f"Position at angle {angle_deg}° has Z={z:.1f}mm above maximum. Adjusting to Z={max_height}mm"
                )
                z = max_height

            # Camera position
            camera_pos = np.array([x, y, z])

            # Use quaternion-based "look at" function to calculate orientation
            # This replaces the manual calculation of Euler angles
            quaternion = self.kinematics.look_at_quaternion(camera_pos, part_center)

            # Convert quaternion to Euler angles
            euler_angles = self.kinematics.quaternion_to_euler(quaternion)

            # Create cartesian pose [x,y,z,rx,ry,rz]
            cartesian_pose = list(camera_pos) + list(euler_angles)
            poses.append(cartesian_pose)

            self.logger.debug(
                f"Added scanning pose at angle {angle_deg}°: {cartesian_pose}"
            )
            self.logger.debug(f"  Position: {camera_pos}, Quaternion: {quaternion}")

        self.logger.info(f"Generated {len(poses)} scanning poses")

        return poses

    def calculate_camera_pose(self, robot_pose: List[float]) -> List[float]:
        """Calculate camera pose from robot pose

        Args:
            robot_pose: Robot pose [x,y,z,rx,ry,rz]

        Returns:
            Camera pose [x,y,z,rx,ry,rz]
        """
        # Convert robot pose to transform
        robot_transform = self.kinematics.euler_pose_to_transform(robot_pose)

        # Extract rotation matrix and position
        R = robot_transform[:3, :3]
        t = robot_transform[:3, 3]

        # Apply camera offset in robot frame
        camera_pos = t + R @ self.camera_offset

        # Create camera transform
        camera_transform = np.eye(4)
        camera_transform[:3, :3] = R  # Same rotation as robot
        camera_transform[:3, 3] = camera_pos

        # Convert back to pose format
        camera_pose = self.kinematics.transform_to_euler_pose(camera_transform)

        return camera_pose

    def execute_scanning_trajectory(
        self,
        poses: List[List[float]],
        tool: int = CoordinateSystem.DEFAULT_TOOL,
        user: int = CoordinateSystem.DEFAULT_USER_FRAME,
        velocity: float = MotionParameters.DEFAULT_SPEED * 0.5,
        capture_delay: float = 0.5,
        simulation: bool = True,
    ) -> List[Tuple[str, np.ndarray]]:
        """Execute the scanning trajectory and capture images

        Args:
            poses: List of cartesian poses [x,y,z,rx,ry,rz]
            tool: Tool frame ID
            user: User frame ID
            velocity: Motion velocity (fraction of maximum)
            capture_delay: Delay after motion to allow camera stabilization
            simulation: Run in simulation mode without robot movement

        Returns:
            List of (image_path, camera_pose) tuples
        """
        self.logger.info(
            "Executing scanning trajectory with %d poses%s",
            len(poses),
            " (SIMULATION MODE)" if simulation else "",
        )

        # Store image paths and corresponding camera poses
        capture_data = []

        # Create timestamp for this scan session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(self.image_dir, timestamp)
        os.makedirs(session_dir, exist_ok=True)

        # Create a file to store all poses
        poses_file = os.path.join(self.pose_dir, f"poses_{timestamp}.txt")

        try:
            for i, robot_pose in enumerate(poses):
                pose_name = f"pose_{i:03d}"

                self.logger.info("Pose %d: %s", i, robot_pose)

                success = True
                if not simulation:
                    # Execute the motion with cartesian pose directly
                    self.logger.info("Moving to scanning pose %d", i)
                    success = self.robot_api.move_linear(
                        robot_pose, tool, user, velocity
                    )

                    if not success:
                        self.logger.error("Failed to move to pose %d", i)
                        continue

                    # Wait for the robot to stabilize and motion to complete
                    self.robot_api.wait_for_motion_done()
                    time.sleep(capture_delay)
                else:
                    # In simulation mode, just log the pose
                    self.logger.info("SIMULATION: Would move to pose %d", i)

                # Get the actual robot pose to calculate camera pose
                actual_robot_pose = None
                if not simulation:
                    # Get actual TCP pose from robot API
                    actual_robot_pose = self.robot_api.get_cartesian_pose()
                    if actual_robot_pose is None or len(actual_robot_pose) != 6:
                        self.logger.warning(
                            "Could not get actual TCP pose, using planned pose"
                        )
                        actual_robot_pose = robot_pose
                else:
                    actual_robot_pose = robot_pose

                # Calculate camera pose from robot pose
                camera_pose = self.calculate_camera_pose(actual_robot_pose)

                # Capture image (this would call the camera API)
                image_path = os.path.join(session_dir, f"{pose_name}.png")

                if not simulation:
                    self.logger.info("Capturing image at pose %d", i)
                    # TODO: Replace with actual camera capture function
                    # self.camera.capture_image(image_path)
                else:
                    self.logger.info("SIMULATION: Would capture image at pose %d", i)

                # Simulate image capture
                with open(image_path, "w") as f:
                    f.write(f"Simulated image for pose {i}")

                # Convert camera pose to transform for return value
                camera_transform = self.kinematics.euler_pose_to_transform(camera_pose)

                # Append to capture data
                capture_data.append((image_path, camera_transform))

                # Append to poses file
                with open(poses_file, "a") as f:
                    f.write(f"{pose_name}.png: {camera_pose}\n")

                self.logger.info("Completed capture %d/%d", i + 1, len(poses))

            self.logger.info("Scanning trajectory completed successfully")
            self.logger.info("Captured %d images", len(capture_data))
            self.logger.info("Image data saved to: %s", session_dir)
            self.logger.info("Pose data saved to: %s", poses_file)

            return capture_data

        except Exception as e:
            self.logger.error("Error executing scanning trajectory: %s", str(e))
            return capture_data

    def scan_part(
        self,
        part_center: List[float],
        part_dimensions: List[float],
        tool: int = CoordinateSystem.DEFAULT_TOOL,
        user: int = CoordinateSystem.DEFAULT_USER_FRAME,
        simulation: bool = True,
        velocity: float = 0.3,
        capture_delay: float = 0.5,
    ) -> List[Tuple[str, np.ndarray]]:
        """Perform a complete scan of a part

        Args:
            part_center: [x, y, z] center of the part in robot base frame
            part_dimensions: [length, width, height] of the part
            tool: Tool frame ID
            user: User frame ID
            simulation: Run in simulation mode without robot movement
            velocity: Motion velocity (fraction of maximum)
            capture_delay: Delay after motion to allow camera stabilization

        Returns:
            List of (image_path, camera_pose) tuples
        """
        with Timer("scan_part", self.logger):
            self.logger.info(
                "Starting part scan at position %s with dimensions %s%s",
                part_center,
                part_dimensions,
                " (SIMULATION MODE)" if simulation else "",
            )

            # Generate scanning poses directly in cartesian format
            scan_params = {
                "part_center": part_center,
                "scanning_height": 350,
                "orbit_radius": 100,
                "num_points": 8,
            }

            poses = self.generate_scanning_poses(**scan_params)

            # Execute trajectory with direct cartesian poses
            capture_data = self.execute_scanning_trajectory(
                poses,
                tool,
                user,
                velocity=velocity,
                capture_delay=capture_delay,
                simulation=simulation,
            )

            return capture_data


if __name__ == "__main__":
    try:
        # Initialize robot components
        kinematics = QuaternionKinematics()

        # Use actual robot or simulation
        simulation_mode = False

        # If using real robot
        robot_api = None
        if not simulation_mode:
            try:
                # Try to connect to the robot
                robot_api = RobotAPI()

                # Enable the robot
                success = robot_api.enable_robot()
                if not success:
                    print("Failed to enable robot. Switching to simulation mode.")
                    simulation_mode = True

            except Exception as e:
                print(f"Error connecting to robot: {str(e)}")
                print("Switching to simulation mode.")
                simulation_mode = True

        # If simulation mode, create dummy robot API
        if simulation_mode or robot_api is None:
            print("Running in simulation mode")
            robot_api = RobotAPI()  # Will create a dummy instance if connection fails

        # Initialize part scanner
        scanner = PartScanner(robot_api, kinematics)

        # Define part parameters
        part_center = [-435, -300, 30]  # Object center in robot base frame
        part_dimensions = [50, 60, 60]  # Length, width, height in mm

        # Customize scanning parameters
        scan_params = {"velocity": 15, "simulation": simulation_mode}

        # Perform the scan
        capture_data = scanner.scan_part(part_center, part_dimensions, **scan_params)

        # Print results
        print(f"Scan completed with {len(capture_data)} images")

    except Exception as e:
        print(f"Error in main program: {str(e)}")
        import traceback

        traceback.print_exc()
    finally:
        # Clean up and disconnect if using real robot
        if not simulation_mode and "robot_api" in locals() and robot_api is not None:
            print("Disconnecting from robot")
            robot_api.disconnect()
