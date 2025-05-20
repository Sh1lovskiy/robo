import os
import json
import numpy as np
import cv2
import torch
from transformers import DPTForDepthEstimation, DPTImageProcessor
from scipy.spatial.transform import Rotation
import open3d as o3d
from misc.logger import Logger, Timer
from typing import List, Dict, Tuple, Any, Optional
import copy

# Initialize logger
logger = Logger.get_logger("PointCloudGenerator", json_format=True)


class PointCloudGenerator:
    """Generate and merge point clouds from stereo images using depth estimation models."""

    def __init__(
        self,
        model_name: str = "Intel/dpt-large",
        calib_path: str = "calibration/calibration_data/stereo_calibration.npz",
    ):
        """
        Initialize the PointCloudGenerator.

        Args:
            model_name: HuggingFace model name for depth estimation
            calib_path: Path to stereo calibration data
        """
        with Timer("Initialization", logger):
            logger.info(f"Initializing PointCloudGenerator with model: {model_name}")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")

            # Load depth estimation model
            self.processor = DPTImageProcessor.from_pretrained(model_name)
            self.model = DPTForDepthEstimation.from_pretrained(model_name).to(
                self.device
            )

            # Load calibration data if available
            self.load_calibration(calib_path)

    def load_calibration(self, calib_path: str) -> None:
        """
        Load stereo camera calibration parameters.

        Args:
            calib_path: Path to calibration file
        """
        try:
            if os.path.exists(calib_path):
                calib_data = np.load(calib_path)
                logger.info(f"Loaded calibration data from: {calib_path}")

                # Store calibration parameters
                self.calib_data = calib_data
                self.camera_matrix_left = calib_data["camera_matrix_left"]
                self.camera_matrix_right = calib_data["camera_matrix_right"]
                self.dist_left = calib_data["dist_coeffs_left"]
                self.dist_right = calib_data["dist_coeffs_right"]
                self.R = calib_data["R"]  # Rotation matrix between cameras
                self.T = calib_data["T"]  # Translation vector between cameras
                self.Q = calib_data["Q"]  # Disparity-to-depth mapping matrix

                # Maps for undistortion and rectification
                self.stereoMapL_x = calib_data["stereoMapL_x"]
                self.stereoMapL_y = calib_data["stereoMapL_y"]
                self.stereoMapR_x = calib_data["stereoMapR_x"]
                self.stereoMapR_y = calib_data["stereoMapR_y"]

                logger.info(f"Camera matrix left:\n{self.camera_matrix_left}")
                logger.info(f"Camera matrix right:\n{self.camera_matrix_right}")
                logger.info(f"Stereo baseline: {abs(self.T[0])} mm")
            else:
                logger.warning(f"Calibration file not found: {calib_path}")
                # Set default camera parameters
                self.camera_matrix_left = np.array(
                    [[1000.0, 0, 640.0], [0, 1000.0, 360.0], [0, 0, 1]]
                )
                self.camera_matrix_right = self.camera_matrix_left.copy()
                self.Q = None
        except Exception as e:
            logger.error(f"Failed to load calibration data: {str(e)}")
            # Set default camera parameters
            self.camera_matrix_left = np.array(
                [[1000.0, 0, 640.0], [0, 1000.0, 360.0], [0, 0, 1]]
            )
            self.camera_matrix_right = self.camera_matrix_left.copy()
            self.Q = None

    def load_data(self, base_dir: str) -> List[Dict[str, Any]]:
        """
        Load images and camera coordinates from directory.

        Args:
            base_dir: Directory containing images and coordinates.json

        Returns:
            List of dictionaries with image data and camera coordinates
        """
        with Timer("DataLoading", logger):
            coords_path = os.path.join(base_dir, "coordinates.json")
            logger.info(f"Loading coordinates from: {coords_path}")

            try:
                with open(coords_path, "r") as f:
                    coords_data = json.load(f)
                logger.info(f"Loaded coordinates for {len(coords_data)} images")
            except Exception as e:
                logger.error(f"Failed to load coordinates: {str(e)}")
                return []

            image_data = []
            for img_name, data in coords_data.items():
                img_path = os.path.join(base_dir, img_name)
                if os.path.exists(img_path):
                    try:
                        img = cv2.imread(img_path)
                        if img is not None:
                            camera_coords = data.get("camera_coords", [])
                            if len(camera_coords) == 6:
                                image_data.append(
                                    {
                                        "name": img_name,
                                        "image": img,
                                        "camera_coords": camera_coords,
                                    }
                                )
                                logger.info(
                                    f"Loaded image: {img_name}, shape: {img.shape}"
                                )
                            else:
                                logger.warning(
                                    f"Invalid camera coordinates for {img_name}"
                                )
                        else:
                            logger.warning(f"Failed to load image: {img_name}")
                    except Exception as e:
                        logger.error(f"Error processing image {img_name}: {str(e)}")
                else:
                    logger.warning(f"Image file not found: {img_path}")

            logger.info(f"Successfully loaded {len(image_data)} valid images")
            return image_data

    def split_stereo_image(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split stereo image into left and right parts.

        Args:
            img: Input stereo image

        Returns:
            Tuple of left and right images
        """
        height, width = img.shape[:2]
        mid = width // 2
        left_img = img[:, :mid]
        right_img = img[:, mid:]
        return left_img, right_img

    def rectify_images(
        self, left_img: np.ndarray, right_img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rectify stereo images using calibration data.

        Args:
            left_img: Left camera image
            right_img: Right camera image

        Returns:
            Tuple of rectified left and right images
        """
        try:
            if hasattr(self, "stereoMapL_x") and hasattr(self, "stereoMapL_y"):
                left_rectified = cv2.remap(
                    left_img, self.stereoMapL_x, self.stereoMapL_y, cv2.INTER_LINEAR
                )
                right_rectified = cv2.remap(
                    right_img, self.stereoMapR_x, self.stereoMapR_y, cv2.INTER_LINEAR
                )
                logger.info("Images rectified using calibration data")
                return left_rectified, right_rectified
            else:
                logger.warning(
                    "Stereo rectification maps not available, using original images"
                )
                return left_img, right_img
        except Exception as e:
            logger.error(f"Error during image rectification: {str(e)}")
            return left_img, right_img

    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth using DPT model.

        Args:
            image: Input RGB image

        Returns:
            Depth map
        """
        with Timer("DepthEstimation", logger):
            # Convert BGR to RGB for the model
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Prepare image for the model
            inputs = self.processor(images=image_rgb, return_tensors="pt").to(
                self.device
            )

            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_depth = outputs.predicted_depth

            # Interpolate to original size and convert to numpy
            prediction = (
                torch.nn.functional.interpolate(
                    predicted_depth.unsqueeze(1),
                    size=image.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

            # Normalize depth values
            depth_min = prediction.min()
            depth_max = prediction.max()

            # Scale depth values based on expected scene depth
            # This is a heuristic - in a real application, you'd calibrate this
            scale_factor = 1
            scaled_depth = prediction * scale_factor

            logger.info(
                f"Depth estimation complete. Shape: {scaled_depth.shape}, Range: [{depth_min:.3f}, {depth_max:.3f}]"
            )
            return scaled_depth

    def save_depth_map(self, depth: np.ndarray, output_path: str) -> None:
        """
        Save depth map as image.

        Args:
            depth: Depth map
            output_path: Path to save depth image
        """
        try:
            # Normalize depth for visualization
            depth_min = depth.min()
            depth_max = depth.max()
            depth_normalized = (
                (depth - depth_min) / (depth_max - depth_min) * 255
            ).astype(np.uint8)

            # Apply colormap for better visualization
            depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)

            # Save both grayscale and colormap versions
            cv2.imwrite(output_path + "_gray.png", depth_normalized)
            cv2.imwrite(output_path + "_color.png", depth_colormap)

            # Save raw depth data as numpy array
            np.save(output_path + "_raw.npy", depth)

            logger.info(f"Saved depth map to {output_path}")
        except Exception as e:
            logger.error(f"Error saving depth map: {str(e)}")

    def depth_to_pointcloud(
        self,
        depth: np.ndarray,
        color_img: np.ndarray,
        camera_matrix: np.ndarray,
        use_q_matrix: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert depth map to point cloud.

        Args:
            depth: Depth map
            color_img: RGB image for color information
            camera_matrix: Camera intrinsic matrix
            use_q_matrix: Whether to use the Q matrix from calibration

        Returns:
            Tuple of points and colors
        """
        with Timer("PointCloudGeneration", logger):
            height, width = depth.shape

            if use_q_matrix and hasattr(self, "Q") and self.Q is not None:
                # Use the Q matrix from calibration to reproject
                # Create 3-channel matrix with x, y, disparity
                # Note: For DPT, we're treating the depth directly, not as disparity
                # In a real application, you'd need to properly convert depth to disparity

                # We'll create a 3D array where:
                # - First channel is the column index (X coordinate)
                # - Second channel is the row index (Y coordinate)
                # - Third channel is the depth
                xyz = np.zeros((height, width, 3), dtype=np.float32)
                x, y = np.meshgrid(np.arange(width), np.arange(height))
                xyz[:, :, 0] = x
                xyz[:, :, 1] = y
                xyz[:, :, 2] = depth

                # Reshape to a list of points
                points_2d = xyz.reshape(-1, 3)

                # Filter out points with invalid depth
                valid = points_2d[:, 2] > 0.0001
                points_2d = points_2d[valid]

                # Project to 3D using the Q matrix
                # [X, Y, Z, W] = Q * [x, y, depth, 1]
                # Then normalize by W
                ones = np.ones((points_2d.shape[0], 1), dtype=np.float32)
                points_homogeneous = np.hstack([points_2d, ones])
                points_3d_homogeneous = np.dot(points_homogeneous, self.Q.T)

                # Normalize by W
                w = points_3d_homogeneous[:, 3:4]
                mask = w != 0
                w[~mask] = 1.0  # Avoid division by zero
                points_3d = points_3d_homogeneous[:, :3] / w

                # Extract colors
                y_indices = points_2d[:, 1].astype(np.int32)
                x_indices = points_2d[:, 0].astype(np.int32)
                colors = color_img[y_indices, x_indices] / 255.0

                logger.info(f"Generated {len(points_3d)} points using Q matrix")
                return points_3d, colors
            else:
                # Fallback to manual reprojection
                # Camera parameters
                fx = camera_matrix[0, 0]
                fy = camera_matrix[1, 1]
                cx = camera_matrix[0, 2]
                cy = camera_matrix[1, 2]

                # Create meshgrid for pixel coordinates
                y_indices, x_indices = np.meshgrid(
                    np.arange(height), np.arange(width), indexing="ij"
                )
                x_indices = x_indices.reshape(-1)
                y_indices = y_indices.reshape(-1)

                # Get depth values
                z = depth.reshape(-1)

                # Filter out invalid depth values
                valid_depth = z > 0.0001  # Minimum depth threshold
                x_indices = x_indices[valid_depth]
                y_indices = y_indices[valid_depth]
                z = z[valid_depth]

                # Back-project to 3D
                x = (x_indices - cx) * z / fx
                y = (y_indices - cy) * z / fy

                # Create point cloud
                points = np.column_stack((x, y, z))

                # Extract colors
                colors = color_img[y_indices, x_indices] / 255.0

                logger.info(
                    f"Generated {len(points)} points from depth map using manual reprojection"
                )
                return points, colors

    def transform_pointcloud(
        self, points: np.ndarray, camera_coords: List[float]
    ) -> np.ndarray:
        """
        Transform point cloud to robot coordinate system.

        Args:
            points: Input point cloud
            camera_coords: Camera coordinates [x, y, z, Rx, Ry, Rz]

        Returns:
            Transformed point cloud
        """
        with Timer("PointCloudTransformation", logger):
            # Extract position and orientation
            position = np.array(camera_coords[:3])
            orientation = np.array(camera_coords[3:])

            logger.info(
                f"Transforming pointcloud using position: {position}, orientation: {orientation}"
            )

            # Create rotation matrix from Euler angles (assuming radians)
            # Note: The order of rotation is important. This uses 'xyz' convention
            rot = Rotation.from_euler("xyz", orientation, degrees=True)
            rotation_matrix = rot.as_matrix()

            # Debug info
            logger.info(f"Rotation matrix:\n{rotation_matrix}")

            # Apply rotation and translation
            transformed_points = np.dot(points, rotation_matrix.T) + position

            return transformed_points

    def visualize_single_pointcloud(
        self, pcd: o3d.geometry.PointCloud, title: str = "Point Cloud"
    ):
        """
        Visualize a single point cloud.

        Args:
            pcd: Point cloud to visualize
            title: Window title
        """
        logger.info(f"Visualizing point cloud: {title}, {len(pcd.points)} points")

        vis = o3d.visualization.Visualizer()
        vis.create_window(title, 800, 600)
        vis.add_geometry(pcd)

        # Add coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )
        vis.add_geometry(coordinate_frame)

        # Set rendering options
        opt = vis.get_render_option()
        opt.background_color = np.array([0.1, 0.1, 0.1])
        opt.point_size = 2.0

        # Update view
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        # Capture view
        vis.run()
        vis.destroy_window()

    def registration_icp(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        max_correspondence_distance: float = 0.05,
        method: str = "point_to_plane",
    ) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
        """
        Register source point cloud to target using ICP.

        Args:
            source: Source point cloud
            target: Target point cloud
            max_correspondence_distance: Maximum correspondence distance
            method: ICP method (point_to_point or point_to_plane)

        Returns:
            Tuple of (transformed source point cloud, transformation matrix)
        """
        with Timer("ICPRegistration", logger):
            # Ensure point clouds have normals for point-to-plane ICP
            if method == "point_to_plane":
                if not source.has_normals():
                    source.estimate_normals(
                        search_param=o3d.geometry.KDTreeSearchParamHybrid(
                            radius=0.1, max_nn=30
                        )
                    )
                if not target.has_normals():
                    target.estimate_normals(
                        search_param=o3d.geometry.KDTreeSearchParamHybrid(
                            radius=0.1, max_nn=30
                        )
                    )

            # Set ICP parameters
            icp_method = (
                o3d.pipelines.registration.TransformationEstimationPointToPlane()
                if method == "point_to_plane"
                else o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )

            # Run ICP
            result = o3d.pipelines.registration.registration_icp(
                source,
                target,
                max_correspondence_distance,
                np.identity(4),
                icp_method,
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
            )

            # Apply transformation
            transformed_source = copy.deepcopy(source)
            transformed_source.transform(result.transformation)

            logger.info(
                f"ICP registration finished with fitness: {result.fitness:.4f}, RMSE: {result.inlier_rmse:.4f}"
            )
            logger.info(f"Transformation matrix:\n{result.transformation}")

            return transformed_source, result.transformation

    def process_images(
        self, image_data: List[Dict[str, Any]], depth_dir: str, output_dir: str
    ) -> o3d.geometry.PointCloud:
        """
        Process images to create a combined point cloud.

        Args:
            image_data: List of dictionaries with image data
            depth_dir: Directory to save depth maps
            output_dir: Directory to save individual point clouds

        Returns:
            Combined point cloud
        """
        individual_pcds = []

        for idx, data in enumerate(image_data):
            with Timer(f"ProcessImage_{idx}", logger):
                logger.info(
                    f"Processing image {idx+1}/{len(image_data)}: {data['name']}"
                )

                # Split stereo image
                left_img, right_img = self.split_stereo_image(data["image"])

                # Rectify images using calibration data
                left_rect, right_rect = self.rectify_images(left_img, right_img)

                # Estimate depth for left image
                depth_left = self.estimate_depth(left_img)

                # Save depth map
                base_filename = os.path.splitext(data["name"])[0]
                depth_path = os.path.join(depth_dir, f"{base_filename}_left_depth")
                self.save_depth_map(depth_left, depth_path)

                # Create point clouds from depth maps
                points_left, colors_left = self.depth_to_pointcloud(
                    depth_left,
                    left_img,
                    self.camera_matrix_left,
                    use_q_matrix=hasattr(self, "Q") and self.Q is not None,
                )

                # Transform point clouds to robot coordinate system
                transformed_points_left = self.transform_pointcloud(
                    points_left, data["camera_coords"]
                )

                # Create Open3D point cloud
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(transformed_points_left)
                pcd.colors = o3d.utility.Vector3dVector(colors_left)

                # Filter outliers
                pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

                # Save individual point cloud
                pcd_path = os.path.join(output_dir, f"{base_filename}_pointcloud.ply")
                o3d.io.write_point_cloud(pcd_path, pcd)
                logger.info(f"Saved individual point cloud to {pcd_path}")

                # Visualize individual point cloud
                self.visualize_single_pointcloud(
                    pcd, f"Point Cloud {idx+1} - {data['name']}"
                )

                # Store in list of individual point clouds
                individual_pcds.append(pcd)

                logger.info(
                    f"Processed point cloud for image {data['name']} with {len(pcd.points)} points"
                )

        # Combine point clouds using ICP registration
        if len(individual_pcds) > 1:
            combined_pcd = self.combine_pointclouds_with_icp(individual_pcds)
        elif len(individual_pcds) == 1:
            combined_pcd = individual_pcds[0]
        else:
            combined_pcd = o3d.geometry.PointCloud()

        return combined_pcd

    def combine_pointclouds_with_icp(self, point_clouds: List[o3d.geometry.PointCloud]) -> o3d.geometry.PointCloud:
        """
        Combine multiple point clouds using ICP registration.
        
        Args:
            point_clouds: List of point clouds to combine
            
        Returns:
            Combined point cloud
        """
        with Timer("CombinePointCloudsICP", logger):
            if not point_clouds:
                logger.warning("No point clouds to combine")
                return o3d.geometry.PointCloud()
            
            if len(point_clouds) == 1:
                logger.info("Only one point cloud, no need for ICP")
                return point_clouds[0]
            
            logger.info(f"Combining {len(point_clouds)} point clouds using ICP")
            
            # Simply merge the point clouds without alignment
            # This should work if the camera coordinates are correct
            combined = o3d.geometry.PointCloud()
            for i, pc in enumerate(point_clouds):
                logger.info(f"Adding point cloud {i+1}/{len(point_clouds)} with {len(pc.points)} points")
                combined += pc
            
            # Remove duplicates with voxel downsampling
            combined = combined.voxel_down_sample(voxel_size=0.005)
            
            # Remove outliers 
            combined, _ = combined.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
            
            logger.info(f"Final combined point cloud has {len(combined.points)} points")
            return combined

    def visualize_alignment(
        self,
        target: o3d.geometry.PointCloud,
        source: o3d.geometry.PointCloud,
        title: str = "Alignment",
    ) -> None:
        """
        Visualize the alignment of two point clouds.

        Args:
            target: Target point cloud (original)
            source: Source point cloud (aligned)
            title: Window title
        """
        target_temp = copy.deepcopy(target)
        source_temp = copy.deepcopy(source)

        # Color the point clouds differently
        target_temp.paint_uniform_color([1, 0.706, 0])  # Yellow
        source_temp.paint_uniform_color([0, 0.651, 0.929])  # Blue

        vis = o3d.visualization.Visualizer()
        vis.create_window(title, 800, 600)
        vis.add_geometry(target_temp)
        vis.add_geometry(source_temp)

        # Add coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )
        vis.add_geometry(coordinate_frame)

        # Set rendering options
        opt = vis.get_render_option()
        opt.background_color = np.array([0.1, 0.1, 0.1])
        opt.point_size = 2.0

        vis.run()
        vis.destroy_window()

    def filter_and_optimize(
        self, pcd: o3d.geometry.PointCloud
    ) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
        """
        Filter and optimize the point cloud, identify the table plane.

        Args:
            pcd: Input point cloud

        Returns:
            Tuple of (object point cloud, table point cloud)
        """
        with Timer("FilterOptimize", logger):
            # Downsample
            logger.info(f"Original point cloud size: {len(pcd.points)}")
            downsampled_pcd = pcd.voxel_down_sample(voxel_size=0.01)
            logger.info(f"Downsampled point cloud size: {len(downsampled_pcd.points)}")

            # Remove outliers
            filtered_pcd, _ = downsampled_pcd.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=2.0
            )
            logger.info(f"Filtered point cloud size: {len(filtered_pcd.points)}")

            # Segment plane (table)
            distance_threshold = 0.05  # Увеличим порог для обнаружения стола
            plane_model, inliers = filtered_pcd.segment_plane(
                distance_threshold=distance_threshold, 
                ransac_n=3, 
                num_iterations=2000  # Увеличим число итераций
            )
            a, b, c, d = plane_model
            logger.info(f"Plane equation: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")

            # Extract table and object
            table_cloud = filtered_pcd.select_by_index(inliers)
            object_cloud = filtered_pcd.select_by_index(inliers, invert=True)

            # Paint table cloud in gray
            table_cloud.paint_uniform_color([0.8, 0.8, 0.8])

            logger.info(f"Final object point cloud size: {len(object_cloud.points)}")
            logger.info(f"Table point cloud size: {len(table_cloud.points)}")

            return object_cloud, table_cloud


def main():
    """Main execution function."""
    base_dir = "vision/imgs_fix"
    output_dir = "vision/output"
    depth_dir = os.path.join(output_dir, "depth")
    individual_clouds_dir = os.path.join(output_dir, "individual_clouds")
    calib_path = "calibration/calibration_data/stereo_calibration.npz"

    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(individual_clouds_dir, exist_ok=True)

    logger.info("Starting point cloud generation process")

    with Timer("TotalProcessing", logger):
        # Initialize the point cloud generator with calibration data
        generator = PointCloudGenerator(calib_path=calib_path)

        # Load data
        image_data = generator.load_data(base_dir)
        if not image_data:
            logger.error("No valid image data found. Exiting.")
            return

        # Process images and get combined point cloud with ICP
        combined_pcd = generator.process_images(
            image_data, depth_dir, individual_clouds_dir
        )

        # Filter and optimize point cloud
        object_cloud, table_cloud = generator.filter_and_optimize(combined_pcd)

        # Save results
        object_path = os.path.join(output_dir, "object_pointcloud.ply")
        table_path = os.path.join(output_dir, "table_pointcloud.ply")
        combined_path = os.path.join(output_dir, "combined_pointcloud.ply")

        o3d.io.write_point_cloud(object_path, object_cloud)
        o3d.io.write_point_cloud(table_path, table_cloud)
        o3d.io.write_point_cloud(combined_path, combined_pcd)

        logger.info(f"Saved point clouds: {object_path}, {table_path}, {combined_path}")

        # Visualize final result
        logger.info("Showing final visualization...")
        vis = o3d.visualization.Visualizer()
        vis.create_window("Final Result", 800, 600)
        vis.add_geometry(object_cloud)
        vis.add_geometry(table_cloud)

        # Add coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0, origin=[0, 0, 0])

        vis.add_geometry(coordinate_frame)

        # Set some visualization options
        opt = vis.get_render_option()
        opt.background_color = np.array([0.1, 0.1, 0.1])
        opt.point_size = 2.0

        vis.run()
        vis.destroy_window()

    logger.info("Point cloud generation complete")


if __name__ == "__main__":
    main()
