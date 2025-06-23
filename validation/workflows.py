"""CLI workflows for hand-eye calibration validation."""

from __future__ import annotations

import argparse
import os
import cv2
import numpy as np
from dataclasses import dataclass

from utils.cli import Command, CommandDispatcher
from utils.config import Config
from utils.io import load_camera_params
from utils.logger import Logger

from .handeye_validation import (
    CameraInterface,
    RobotInterface,
    DatasetAnalyzer,
    HandEyeValidator,
    load_default_validator,
)


@dataclass
class StubCamera(CameraInterface):
    """Camera stub returning a pre-generated image."""

    image: np.ndarray
    K: np.ndarray
    dist: np.ndarray

    def capture(self) -> np.ndarray:
        return self.image.copy()

    def get_intrinsics(self) -> tuple[np.ndarray, np.ndarray]:
        return self.K, self.dist


@dataclass
class StubRobot(RobotInterface):
    """Robot stub that records commanded poses."""

    poses: list[np.ndarray]

    def move_to(self, pose: np.ndarray) -> None:
        self.poses.append(pose)


def _validate_pose(_: argparse.Namespace) -> None:
    Config.load()
    char_cfg = Config.get("charuco")
    xml_path = os.path.join(
        char_cfg.get("calib_output_dir", "calibration/results"),
        char_cfg.get("xml_file", "charuco_cam.xml"),
    )
    camera_matrix = None
    dist_coeffs = None
    if os.path.isfile(xml_path):
        camera_matrix, dist_coeffs = load_camera_params(xml_path)

    images_dir = char_cfg.get("images_dir", "captures")
    image_name = char_cfg.get("image_name", "0_rgb.png")
    image_path = os.path.join(images_dir, image_name)

    if not os.path.isfile(image_path):
        Logger.get_logger("validation.cli").error(f"Image not found: {image_path}")
        return

    img = cv2.imread(image_path)
    if img is None:
        Logger.get_logger("validation.cli").error(f"Failed to load image: {image_path}")
        return

    camera = StubCamera(img, camera_matrix, dist_coeffs)
    robot = StubRobot([])
    validator = load_default_validator(camera, robot)
    pose = np.eye(4)
    pose[:3, 3] = [0.0, 0.0, 1.0]
    result = validator.validate_pose(pose)
    Logger.get_logger("validation.cli").info(f"Result: {result}")


def _analyze_dataset(_: argparse.Namespace) -> None:
    import json

    Config.load()
    char_cfg = Config.get("charuco")
    xml_path = os.path.join(
        char_cfg.get("calib_output_dir", "calibration/results"),
        char_cfg.get("xml_file", "charuco_cam.xml"),
    )

    if os.path.isfile(xml_path):
        camera_matrix, dist_coeffs = load_camera_params(xml_path)
    else:
        width = Config.get("vision.realsense_width", 640)
        height = Config.get("vision.realsense_height", 480)
        fx = char_cfg.get("fx", 600)
        fy = char_cfg.get("fy", 600)
        cx = char_cfg.get("cx", width // 2)
        cy = char_cfg.get("cy", height // 2)
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
        dist_coeffs = np.zeros(5)

    images_dir = char_cfg.get("images_dir", "captures")
    image_files = sorted(
        [
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )
    if not image_files:
        Logger.get_logger("validation.cli").error(f"No images found in {images_dir}")
        return

    poses_path = os.path.join(images_dir, "poses.json")
    if not os.path.isfile(poses_path):
        Logger.get_logger("validation.cli").error(
            f"poses.json not found in {images_dir}"
        )
        return

    with open(poses_path, "r") as f:
        poses_json = json.load(f)

    image_pose_pairs = []
    for img_path in image_files:
        fname = os.path.basename(img_path)
        idx = os.path.splitext(fname)[0].split("_")[0]  # "007_rgb.png" -> "007"
        pose_entry = poses_json.get(idx)
        if pose_entry is not None:
            pose = np.eye(4)
            coords = pose_entry["tcp_coords"]
            pose[:3, 3] = coords[:3]
            image_pose_pairs.append((img_path, pose))
        else:
            Logger.get_logger("validation.cli").warning(
                f"No pose found for image: {fname} (index: {idx})"
            )

    if not image_pose_pairs:
        Logger.get_logger("validation.cli").error("No valid image-pose pairs found")
        return

    imgs = []
    poses = []
    for img_path, pose in image_pose_pairs:
        img = cv2.imread(img_path)
        if img is not None:
            imgs.append(img)
            poses.append(pose)
        else:
            Logger.get_logger("validation.cli").warning(
                f"Skipped corrupted image: {img_path}"
            )

    if not imgs or not poses:
        Logger.get_logger("validation.cli").error("No valid images/poses for analysis")
        return

    class MultiImageCamera(StubCamera):
        def __init__(self, images, K, dist):
            super().__init__(images[0], K, dist)
            self.images = images
            self.idx = 0

        def capture(self):
            img = self.images[self.idx]
            self.idx += 1
            if self.idx >= len(self.images):
                self.idx = 0
            return img

    camera = MultiImageCamera(imgs, camera_matrix, dist_coeffs)
    robot = StubRobot([])
    validator = load_default_validator(camera, robot)
    analyzer = DatasetAnalyzer(validator, poses)
    analyzer.run()


def create_cli() -> CommandDispatcher:
    return CommandDispatcher(
        "Validation workflows",
        [
            Command("validate-pose", _validate_pose, None, "Validate single pose"),
            Command("analyze-dataset", _analyze_dataset, None, "Analyze dataset"),
        ],
    )


def main() -> None:
    create_cli().run(logger=Logger.get_logger("validation.cli"))


if __name__ == "__main__":
    main()
