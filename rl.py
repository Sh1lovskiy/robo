"""
Reinforcement Learning for Robot Motion with Object Tracking

This script implements a Gym environment and a reinforcement learning agent
to control a 6-DOF robot arm while keeping a part in the camera view.
"""

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from gym import spaces
import time
import cv2
from ultralytics import YOLO
from core.control import RobotController
from typing import List, Tuple, Dict, Optional
from misc.logger import Logger
from config.constants import DEFAULT_IP

logger = Logger.get_logger("rl_robot_controller", json_format=True)

CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
STEREO_SPLIT_X = FRAME_WIDTH // 2
CONFIDENCE_THRESHOLD = 0.5
YOLO_WEIGHTS_PATH = (
    r"C:\Users\Алексей\Documents\kinematics\funetune\runs\detect\train2\weights\best.pt"
)


class RobotEnv(gym.Env):
    """
    Robot Control Environment for reinforcement learning.

    State:
    - Current robot position (6 values)
    - Target position (6 values)
    - Part detection info (5 values - visibility, x, y, width, height)

    Action:
    - Continuous values for the 6 joints [-1, 1], scaled to appropriate step sizes
    """

    def __init__(self, robot_ip=DEFAULT_IP, target_position=None):
        super(RobotEnv, self).__init__()
        self.robot = RobotController(ip_address=robot_ip)
        if not self.robot.connected or not self.robot.initialize():
            raise ConnectionError("Failed to connect to the robot")

        self.camera = cv2.VideoCapture(CAMERA_INDEX)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        self.model = self._load_yolo_model()

        self.home_position = self.robot.get_current_pose()
        self.current_position = self.home_position.copy()

        self.target_position = target_position or [-518, 111, 170, -131, 5, -163]

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # Observations:
        # - Current position (6)
        # - Target position (6)
        # - Part visibility (1)
        # - Part position in frame (2) - normalized x, y
        # - Part size in frame (2) - normalized width, height
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32
        )

        self.max_steps = 100
        self.current_step = 0
        self.max_distance = self._calculate_distance(
            self.home_position, self.target_position
        )
        self.last_distance = self.max_distance
        self.last_detection = None
        self.episode_reward = 0
        self.visualization_enabled = True

    def _calculate_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two positions."""
        return np.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(pos1[:3], pos2[:3])))

    def _load_yolo_model(self):
        """Load the YOLO object detection model."""
        import torch

        try:
            logger.info(f"Attempting to load YOLO model from: {YOLO_WEIGHTS_PATH}")

            try:
                model = YOLO(YOLO_WEIGHTS_PATH)
                logger.info("YOLO model loaded successfully using ultralytics")
                return model
            except Exception as e1:
                logger.warning(f"Failed to load with ultralytics YOLO: {e1}")

            logger.info("Using fallback detector")

            class SimpleDetector:
                def __init__(self):
                    self.conf = CONFIDENCE_THRESHOLD

                def __call__(self, frame):
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
                    thresh = cv2.adaptiveThreshold(
                        blurred,
                        255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY_INV,
                        11,
                        2,
                    )

                    class MockResults:
                        def __init__(self, boxes=None):
                            self.boxes = boxes or []

                    class MockBoxes:
                        def __init__(self, xyxy=None, conf=None, cls=None):
                            self.xyxy = xyxy or []
                            self.conf = conf or []
                            self.cls = cls or []

                    contours, _ = cv2.findContours(
                        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )

                    # Filter contours by area
                    min_area = 500
                    boxes = []
                    confs = []
                    classes = []

                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area > min_area:
                            x, y, w, h = cv2.boundingRect(contour)
                            boxes.append([x, y, x + w, y + h])
                            confs.append(0.9)
                            classes.append(0)

                    if boxes:
                        boxes_tensor = torch.tensor(boxes)
                        confs_tensor = torch.tensor(confs)
                        classes_tensor = torch.tensor(classes)

                        mock_boxes = MockBoxes(
                            xyxy=boxes_tensor, conf=confs_tensor, cls=classes_tensor
                        )

                        return [MockResults([mock_boxes])]

                    return [MockResults()]

            return SimpleDetector()

        except Exception as e:
            logger.error(f"All detection methods failed: {e}")
            raise

    def detect_part(self, frame):
        """Detect the part in the frame using YOLO."""
        try:
            results = self.model(frame)

            try:
                if len(results) > 0 and hasattr(results[0], "boxes"):
                    boxes = results[0].boxes

                    if (
                        len(boxes) > 0
                        and hasattr(boxes, "xyxy")
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
                            "normalized_center": (
                                (center_x / STEREO_SPLIT_X) * 2 - 1,
                                (center_y / FRAME_HEIGHT) * 2 - 1,
                            ),
                            "normalized_size": (
                                width / STEREO_SPLIT_X,
                                height / FRAME_HEIGHT,
                            ),
                        }

                        return detection_info

                return None

            except Exception as e:
                logger.error(f"Error processing detection results: {e}")
                return None

        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return None

    def process_stereo_frame(self, frame):
        """Split the stereo camera frame into left and right images."""
        left_img = frame[:, :STEREO_SPLIT_X]
        right_img = frame[:, STEREO_SPLIT_X:]
        return left_img, right_img

    def check_part_visibility(self):
        """Check if the part is visible in the camera frame."""
        ret, frame = self.camera.read()
        if not ret:
            logger.error("Failed to capture frame from camera")
            return None

        # Split stereo frame
        left_img, right_img = self.process_stereo_frame(frame)

        # Try to detect in both left and right frames
        left_detection = self.detect_part(left_img)
        right_detection = self.detect_part(right_img)

        if self.visualization_enabled:
            self._visualize_frame(frame, left_detection, right_detection)

        if left_detection and right_detection:
            if left_detection.get("confidence", 0) > right_detection.get(
                "confidence", 0
            ):
                return ("left", left_detection)
            else:
                return ("right", right_detection)
        elif left_detection:
            return ("left", left_detection)
        elif right_detection:
            return ("right", right_detection)
        else:
            return None

    def _visualize_frame(self, frame, left_detection, right_detection):
        """Visualize the camera feed with detections and robot state."""
        try:
            left_img, right_img = self.process_stereo_frame(frame)

            left_viz = left_img.copy()
            right_viz = right_img.copy()

            if left_detection:
                x1, y1, x2, y2 = left_detection["bbox"]
                confidence = left_detection["confidence"]
                cv2.rectangle(
                    left_viz, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
                )
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
                cv2.rectangle(
                    right_viz, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
                )
                cv2.putText(
                    right_viz,
                    f"Conf: {confidence:.2f}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            # Add robot state information
            info_text = [
                f"Step: {self.current_step}/{self.max_steps}",
                f"Distance to target: {self._calculate_distance(self.current_position, self.target_position):.1f}",
                f"Reward: {self.episode_reward:.1f}",
            ]

            # Add current position info
            pos_info = [
                f"Pos: {', '.join([f'{p:.1f}' for p in self.current_position[:3]])}"
            ]

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

            # Add position info
            for i, text in enumerate(pos_info):
                cv2.putText(
                    right_viz,
                    text,
                    (10, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            # Combine left and right images for display
            # Ensure both images have the same height
            if left_viz.shape[0] != right_viz.shape[0]:
                h = min(left_viz.shape[0], right_viz.shape[0])
                left_viz = cv2.resize(
                    left_viz, (int(left_viz.shape[1] * h / left_viz.shape[0]), h)
                )
                right_viz = cv2.resize(
                    right_viz, (int(right_viz.shape[1] * h / right_viz.shape[0]), h)
                )

            combined_img = np.hstack((left_viz, right_viz))

            # Resize if too large for screen
            screen_width = 1600
            if combined_img.shape[1] > screen_width:
                scale = screen_width / combined_img.shape[1]
                combined_img = cv2.resize(combined_img, None, fx=scale, fy=scale)

            # Show the combined image
            cv2.namedWindow("Robot RL View", cv2.WINDOW_NORMAL)
            cv2.imshow("Robot RL View", combined_img)
            cv2.waitKey(1)

        except Exception as e:
            logger.error(f"Error in visualization: {e}")

    def _get_observation(self):
        """Get the current observation (state)."""
        detection = self.check_part_visibility()

        # Default values if part not detected
        visible = 0.0
        norm_center_x, norm_center_y = 0.0, 0.0
        norm_width, norm_height = 0.0, 0.0

        if detection:
            camera_side, det_info = detection
            visible = 1.0
            norm_center_x, norm_center_y = det_info["normalized_center"]
            norm_width, norm_height = det_info["normalized_size"]
            self.last_detection = detection

        # Create the observation vector
        obs = np.array(
            self.current_position  # Current position (6)
            + self.target_position  # Target position (6)
            + [visible]  # Part visibility (1)
            + [norm_center_x, norm_center_y]  # Normalized center (2)
            + [norm_width, norm_height],  # Normalized size (2)
            dtype=np.float32,
        )

        return obs

    def _calculate_reward(self):
        """Calculate the reward for the current state."""
        # Get the current detection
        detection = self.last_detection

        # Calculate distance to target
        current_distance = self._calculate_distance(
            self.current_position, self.target_position
        )

        # Base reward for moving closer to target (reduced importance)
        distance_reward = (self.last_distance - current_distance) * 5.0
        self.last_distance = current_distance

        # Primary reward focused on part visibility and confidence
        visibility_reward = 0.0

        if detection:
            camera_side, det_info = detection
            confidence = det_info.get("confidence", 0.0)

            # Strong reward for part detection confidence
            confidence_reward = confidence * 10.0
            visibility_reward += confidence_reward

            # Check if the entire part is in the frame
            # Use normalized coordinates to check if part is fully in frame
            norm_x, norm_y = det_info["normalized_center"]
            norm_width, norm_height = det_info["normalized_size"]

            # Calculate how much of the part is in frame (estimate)
            # Parts closer to edge are more likely to be partially out of frame
            edge_distance_x = 1.0 - abs(norm_x) - norm_width / 2
            edge_distance_y = 1.0 - abs(norm_y) - norm_height / 2

            in_frame_score = min(max(edge_distance_x, 0), 1) * min(
                max(edge_distance_y, 0), 1
            )
            frame_reward = in_frame_score * 8.0
            visibility_reward += frame_reward

            # Penalize being too close to the part
            # Ideal part size would be 20-40% of frame
            ideal_min_size = 0.2
            ideal_max_size = 0.4
            part_size_ratio = (norm_width + norm_height) / 2

            if part_size_ratio > ideal_max_size:
                # Too close to part
                too_close_penalty = (
                    -5.0 * (part_size_ratio - ideal_max_size) / (1.0 - ideal_max_size)
                )
                visibility_reward += too_close_penalty
                logger.debug(f"Too close to part penalty: {too_close_penalty:.2f}")
            elif part_size_ratio < ideal_min_size:
                # Too far from part
                too_far_penalty = (
                    -2.0 * (ideal_min_size - part_size_ratio) / ideal_min_size
                )
                visibility_reward += too_far_penalty
                logger.debug(f"Too far from part penalty: {too_far_penalty:.2f}")
            else:
                # Ideal distance - bonus
                ideal_distance_bonus = 3.0 * (
                    1.0
                    - abs(part_size_ratio - (ideal_min_size + ideal_max_size) / 2)
                    / ((ideal_max_size - ideal_min_size) / 2)
                )
                visibility_reward += ideal_distance_bonus
                logger.debug(f"Ideal distance bonus: {ideal_distance_bonus:.2f}")

            # Log detailed reward components
            logger.debug(
                f"Detection confidence: {confidence:.2f}, reward: {confidence_reward:.2f}"
            )
            logger.debug(
                f"In-frame score: {in_frame_score:.2f}, reward: {frame_reward:.2f}"
            )
            logger.debug(f"Part size ratio: {part_size_ratio:.2f}")
        else:
            # Strong penalty for losing sight of part
            visibility_reward -= 8.0

        # Combine rewards
        total_reward = distance_reward + visibility_reward

        # Log reward components
        logger.debug(
            f"Rewards - Distance: {distance_reward:.2f}, Visibility: {visibility_reward:.2f}"
        )

        return total_reward, 0.0  # Return total reward and action penalty

    def step(self, action):
        """Take an action and return the new state, reward, done flag, and info."""
        self.current_step += 1

        # Scale actions from [-1, 1] to appropriate step sizes - REDUCED SPEEDS
        max_xyz_step = 5.0  # mm (reduced from 10.0)
        max_angle_step = 2.5  # degrees (reduced from 5.0)

        scaled_action = np.zeros(6)
        scaled_action[:3] = action[:3] * max_xyz_step
        scaled_action[3:] = action[3:] * max_angle_step

        # Penalty for large actions (smoothness)
        action_magnitude = np.sum(np.abs(action))
        action_penalty = -0.1 * action_magnitude

        # Apply action
        new_position = [c + a for c, a in zip(self.current_position, scaled_action)]

        # Apply constraint: Z coordinate cannot go below 150mm
        if new_position[2] < 150.0:
            logger.warning(
                f"Z-coordinate constraint violated: {new_position[2]}mm < 150mm"
            )
            new_position[2] = 150.0  # Enforce minimum Z height
            action_penalty -= 2.0  # Additional penalty for trying to violate constraint

        # Move robot
        success = self.robot.move_linear(new_position)
        if not success:
            logger.warning("Failed to move robot to requested position")
            reward = -10.0  # Penalty for invalid move
            self.episode_reward += reward
            return self._get_observation(), reward, True, {"success": False}

        # Update current position
        self.current_position = new_position

        # Small delay to allow robot to complete movement
        time.sleep(0.5)

        # Get observation and calculate reward
        observation = self._get_observation()
        reward, _ = self._calculate_reward()
        reward += action_penalty  # Add the action penalty

        # Check if we're done
        done = False

        # Done if reached target (within threshold)
        target_distance = self._calculate_distance(
            self.current_position, self.target_position
        )
        if target_distance < 20.0:  # 20mm threshold
            done = True
            reward += 50.0  # Bonus for reaching target

        # Done if lost part for too long
        if not self.last_detection and self.current_step > 10:
            done = True
            reward -= 20.0  # Penalty for losing part

        # Done if max steps reached
        if self.current_step >= self.max_steps:
            done = True

        # Update cumulative reward
        self.episode_reward += reward

        return observation, reward, done, {"success": True}

    def reset(self):
        """Reset the environment to initial state."""
        # Move robot back to home position
        self.robot.move_linear(self.home_position)
        self.current_position = self.home_position.copy()

        # Reset episode variables
        self.current_step = 0
        self.last_distance = self._calculate_distance(
            self.home_position, self.target_position
        )
        self.last_detection = None
        self.episode_reward = 0.0

        # Wait for robot to reach home position
        time.sleep(1.0)

        return self._get_observation()

    def render(self, mode="human"):
        """Render the environment (already handled in check_part_visibility)."""
        pass

    def close(self):
        """Clean up resources."""
        if self.camera is not None:
            self.camera.release()
        if hasattr(self, "robot") and self.robot is not None:
            self.robot.shutdown()
        cv2.destroyAllWindows()


class Actor(nn.Module):
    """Neural network for the actor in PPO."""

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Mean and standard deviation for each action dimension
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = self.network(state)
        mean = self.mean(x)
        std = self.log_std.exp()

        return mean, std


class Critic(nn.Module):
    """Neural network for the critic in PPO."""

    def __init__(self, state_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.network(state)


class PPO:
    """Proximal Policy Optimization algorithm."""

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=64,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        target_kl=0.01,
        max_grad_norm=0.5,
    ):
        # Initialize actor and critic networks
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, hidden_dim)

        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.max_grad_norm = max_grad_norm

    def get_action(self, state):
        """Sample an action from the policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            mean, std = self.actor(state_tensor)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=1)

        return action.squeeze().numpy(), log_prob.item()

    def get_value(self, state):
        """Get value estimate from the critic."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            value = self.critic(state_tensor)

        return value.item()

    def update(self, batch):
        """Update the actor and critic networks using PPO."""
        # Extract batch data
        states = torch.FloatTensor(batch["states"])
        actions = torch.FloatTensor(batch["actions"])
        old_log_probs = torch.FloatTensor(batch["log_probs"])
        returns = torch.FloatTensor(batch["returns"])
        advantages = torch.FloatTensor(batch["advantages"])

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO actor update
        mean, std = self.actor(states)
        dist = Normal(mean, std)
        new_log_probs = dist.log_prob(actions).sum(dim=1)

        # Calculate ratio and clipped objective
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
            * advantages
        )
        actor_loss = -torch.min(surr1, surr2).mean()

        # Calculate approximate KL divergence
        kl = (old_log_probs - new_log_probs).mean().item()

        # PPO critic update
        value_preds = self.critic(states).squeeze()
        critic_loss = nn.MSELoss()(value_preds, returns)

        # Perform updates
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "kl": kl,
        }


def compute_returns_and_advantages(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """Compute returns and advantages using Generalized Advantage Estimation."""
    returns = []
    advantages = []
    gae = 0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
            next_done = 1
        else:
            next_value = values[t + 1]
            next_done = dones[t + 1]

        delta = rewards[t] + gamma * next_value * (1 - next_done) - values[t]
        gae = delta + gamma * gae_lambda * (1 - next_done) * gae

        returns.insert(0, gae + values[t])
        advantages.insert(0, gae)

    return returns, advantages


def train_robot_rl(episodes=100, steps_per_episode=50, update_steps=10):
    """Train a reinforcement learning agent to control the robot."""
    try:
        # Create environment
        env = RobotEnv()

        # Get state and action dimensions
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        # Initialize PPO agent
        agent = PPO(state_dim, action_dim)

        # Training loop
        logger.info(f"Starting training for {episodes} episodes")

        for episode in range(1, episodes + 1):
            # Reset environment
            state = env.reset()
            done = False
            episode_reward = 0

            # Storage for PPO update
            batch = {
                "states": [],
                "actions": [],
                "log_probs": [],
                "rewards": [],
                "values": [],
                "dones": [],
            }

            # Episode loop
            for t in range(steps_per_episode):
                # Get action from agent
                action, log_prob = agent.get_action(state)
                value = agent.get_value(state)

                # Take action in environment
                next_state, reward, done, info = env.step(action)

                # Store data for update
                batch["states"].append(state)
                batch["actions"].append(action)
                batch["log_probs"].append(log_prob)
                batch["rewards"].append(reward)
                batch["values"].append(value)
                batch["dones"].append(done)

                # Update state and accumulate reward
                state = next_state
                episode_reward += reward

                # Check for early termination
                if done:
                    break

                # Check for user quit
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("User requested exit")
                    done = True
                    break

            # Compute returns and advantages for the episode
            values = batch["values"] + [0]  # Add final value estimate
            returns, advantages = compute_returns_and_advantages(
                batch["rewards"], values, batch["dones"]
            )

            # Update batch with computed returns and advantages
            batch["returns"] = returns
            batch["advantages"] = advantages

            # Update the agent
            if episode % update_steps == 0:
                update_info = agent.update(batch)
                logger.info(
                    f"Update - Actor Loss: {update_info['actor_loss']:.4f}, "
                    f"Critic Loss: {update_info['critic_loss']:.4f}, "
                    f"KL: {update_info['kl']:.4f}"
                )

            # Log episode results
            logger.info(
                f"Episode {episode}/{episodes} - "
                f"Reward: {episode_reward:.2f}, "
                f"Steps: {t+1}"
            )

            # Save model periodically
            if episode % 10 == 0:
                torch.save(
                    {
                        "actor": agent.actor.state_dict(),
                        "critic": agent.critic.state_dict(),
                    },
                    f"robot_rl_model_ep{episode}.pt",
                )

        # Final save
        torch.save(
            {"actor": agent.actor.state_dict(), "critic": agent.critic.state_dict()},
            "robot_rl_model_final.pt",
        )

        logger.info("Training complete")

    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback

        logger.error(traceback.format_exc())
    finally:
        # Clean up
        if "env" in locals():
            env.close()
        cv2.destroyAllWindows()


def test_trained_agent(model_path, episodes=5):
    """Test a trained agent on the robot control task."""
    try:
        # Create environment
        env = RobotEnv()

        # Get state and action dimensions
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        # Initialize PPO agent
        agent = PPO(state_dim, action_dim)

        # Load trained model
        checkpoint = torch.load(model_path)
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.critic.load_state_dict(checkpoint["critic"])

        logger.info(f"Testing trained agent for {episodes} episodes")

        for episode in range(1, episodes + 1):
            # Reset environment
            state = env.reset()
            done = False
            episode_reward = 0

            # Episode loop
            step = 0
            while not done:
                step += 1

                # Get action from agent (deterministic)
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    mean, _ = agent.actor(state_tensor)
                    action = mean.squeeze().numpy()  # Use mean action (no sampling)

                # Take action in environment
                next_state, reward, done, info = env.step(action)

                # Update state and accumulate reward
                state = next_state
                episode_reward += reward

                # Check for user quit
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("User requested exit")
                    done = True
                    break

                # Optional: slow down execution for better visualization
                time.sleep(0.1)

            # Log episode results
            logger.info(
                f"Test Episode {episode}/{episodes} - "
                f"Reward: {episode_reward:.2f}, "
                f"Steps: {step}"
            )

    except Exception as e:
        logger.error(f"Error during testing: {e}")
        import traceback

        logger.error(traceback.format_exc())
    finally:
        # Clean up
        if "env" in locals():
            env.close()
        cv2.destroyAllWindows()


def main():
    """Main function to run training or testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Robot RL Control")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Mode: train or test",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="robot_rl_model_final.pt",
        help="Path to model file for testing",
    )
    parser.add_argument(
        "--episodes", type=int, default=100, help="Number of episodes to train/test"
    )

    args = parser.parse_args()

    if args.mode == "train":
        train_robot_rl(episodes=args.episodes)
    else:
        test_trained_agent(args.model, episodes=args.episodes)


if __name__ == "__main__":
    main()
