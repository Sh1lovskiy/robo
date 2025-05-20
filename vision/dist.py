"""
Detection of upper surface boundaries using stereo images
Processes a stereo image split into left and right parts,
creates an enhanced disparity map, depth map, and identifies the top
surface of an object as the area closest to the camera.
"""

import cv2
import numpy as np
import os
from pathlib import Path
import time
from misc.logger import Logger, Timer

# Logger setup with JSON formatting
logger = Logger.setup_logger(
    "TopSurfaceDetector",
    json_format=True,
    json_fields={
        "component": "vision",
        "operation_type": "top_surface_detection",
    },
)


class TopSurfaceDetector:
    def __init__(
        self, calibration_path="calibration/calibration_data/stereo_calibration.npz"
    ):
        """
        Initialize the top surface detector

        Args:
            calibration_path: Path to stereo camera calibration file
        """
        import logging

        with Timer("Initialization", logger):
            self.calibration_data = self.load_stereo_calibration(calibration_path)
            self.init_stereo_matcher()

            # Performance tracking
            self.processing_time = 0

    def load_stereo_calibration(self, calibration_path):
        """Load stereo camera calibration data from NPZ file."""
        import logging

        with Timer("LoadCalibration", logger):
            try:
                Logger.log_data(
                    logger,
                    logging.INFO,
                    "Loading calibration data from %s",
                    calibration_path,
                )

                if not os.path.exists(calibration_path):
                    Logger.log_json(
                        logger,
                        logging.ERROR,
                        event="file_not_found",
                        error_message=f"Calibration file not found: {calibration_path}",
                    )
                    return None

                calibration_data = np.load(calibration_path)

                # Log successful loading with some calibration info
                Logger.log_json(
                    logger,
                    logging.INFO,
                    event="calibration_loaded",
                    frame_size=(
                        calibration_data["frame_size"].tolist()
                        if "frame_size" in calibration_data
                        else None
                    ),
                )

                return calibration_data
            except Exception as e:
                Logger.log_json(
                    logger,
                    logging.ERROR,
                    event="calibration_load_error",
                    error_message=str(e),
                )
                return None

    def init_stereo_matcher(self):
        """Initialize stereo matching algorithm"""
        import logging

        with Timer("InitStereoMatcher", logger):
            # Create StereoSGBM matcher with improved parameters
            window_size = 5
            min_disp = 0
            num_disp = 16 * 10 - min_disp  # Must be divisible by 16

            self.stereo = cv2.StereoSGBM_create(
                minDisparity=min_disp,
                numDisparities=num_disp,
                blockSize=window_size,
                P1=8 * 3 * window_size**2,  # Controls disparity smoothness
                P2=32 * 3 * window_size**2,  # Controls disparity smoothness
                disp12MaxDiff=1,
                uniquenessRatio=15,
                speckleWindowSize=100,
                speckleRange=32,
                preFilterCap=63,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
            )

            # Create WLS filter for post-processing
            self.right_matcher = cv2.ximgproc.createRightMatcher(self.stereo)
            self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(
                matcher_left=self.stereo
            )
            self.wls_filter.setLambda(8000)
            self.wls_filter.setSigmaColor(1.5)

            Logger.log_json(
                logger,
                logging.INFO,
                event="stereo_matcher_initialized",
                window_size=window_size,
                num_disparities=num_disp,
            )

    def split_stereo_image(self, image_path):
        """Split side-by-side stereo image into left and right frames."""
        import logging

        with Timer("SplitStereoImage", logger):
            try:
                Logger.log_data(
                    logger,
                    logging.INFO,
                    "Loading stereo image from %s",
                    image_path,
                )

                if not os.path.exists(image_path):
                    Logger.log_json(
                        logger,
                        logging.ERROR,
                        event="file_not_found",
                        error_message=f"Image file not found: {image_path}",
                    )
                    return None, None

                # Read stereo image
                stereo_img = cv2.imread(image_path)

                if stereo_img is None:
                    Logger.log_json(
                        logger,
                        logging.ERROR,
                        event="image_read_error",
                        error_message="Failed to read image",
                    )
                    return None, None

                # Get dimensions
                height, width = stereo_img.shape[:2]
                mid_point = width // 2

                # Split image into left and right frames
                left_img = stereo_img[:, :mid_point]
                right_img = stereo_img[:, mid_point:]

                Logger.log_json(
                    logger,
                    logging.INFO,
                    event="image_split",
                    original_size=[width, height],
                    left_size=left_img.shape[:2],
                    right_size=right_img.shape[:2],
                )

                return left_img, right_img
            except Exception as e:
                Logger.log_json(
                    logger,
                    logging.ERROR,
                    event="image_split_error",
                    error_message=str(e),
                )
                return None, None

    def enhance_contrast(self, image):
        """Enhance image contrast for better disparity detection."""
        import logging

        with Timer("EnhanceContrast", logger):
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Apply CLAHE for improved local contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            Logger.log_json(
                logger,
                logging.INFO,
                event="contrast_enhanced",
                image_shape=enhanced.shape,
            )

            return enhanced

    def compute_disparity(self, left_img, right_img):
        """
        Compute disparity map from stereo image pair using enhanced
        filtering methods.

        Args:
            left_img: Left image
            right_img: Right image

        Returns:
            tuple: Color-coded disparity, normalized disparity,
                  filtered disparity
        """
        import logging

        with Timer("ComputeDisparity", logger):
            try:
                if left_img is None or right_img is None:
                    Logger.log_json(
                        logger,
                        logging.ERROR,
                        event="disparity_error",
                        error_message="Input images are None",
                    )
                    return None, None, None

                # Convert to grayscale and enhance contrast
                gray_left = self.enhance_contrast(left_img)
                gray_right = self.enhance_contrast(right_img)

                # Compute disparity maps (left and right)
                left_disp = self.stereo.compute(gray_left, gray_right)
                right_disp = self.right_matcher.compute(gray_right, gray_left)

                # Apply WLS filter to improve quality
                filtered_disp = self.wls_filter.filter(
                    left_disp, gray_left, None, right_disp
                )

                # Normalize for visualization
                # Scale disparity (divide by 16 for SGBM)
                disparity_float = filtered_disp.astype(np.float32) / 16.0

                # Filter out negative values and outliers
                min_disp = np.percentile(disparity_float[disparity_float > 0], 5)
                max_disp = np.percentile(disparity_float[disparity_float > 0], 95)

                # Normalize for visualization accounting for outliers
                disparity_valid = np.clip(disparity_float, min_disp, max_disp)
                norm_disp = cv2.normalize(
                    disparity_valid, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
                )

                # Apply morphological filtering to improve the map
                kernel = np.ones((3, 3), np.uint8)
                filtered_norm_disp = cv2.morphologyEx(
                    norm_disp, cv2.MORPH_CLOSE, kernel
                )

                # Apply color map for better visualization
                disp_color = cv2.applyColorMap(filtered_norm_disp, cv2.COLORMAP_JET)

                # Add legend
                disp_color = self.add_disparity_legend(disp_color)

                Logger.log_json(
                    logger,
                    logging.INFO,
                    event="disparity_computed",
                    min_disp=float(min_disp),
                    max_disp=float(max_disp),
                    disparity_shape=disparity_float.shape,
                )

                return disp_color, filtered_norm_disp, disparity_float
            except Exception as e:
                Logger.log_json(
                    logger,
                    logging.ERROR,
                    event="disparity_computation_error",
                    error_message=str(e),
                )
                import traceback

                Logger.log_data(
                    logger, logging.ERROR, "Error traceback: %s", traceback.format_exc()
                )
                return None, None, None

    def add_disparity_legend(self, disparity_image):
        """
        Add legend to disparity map explaining the colors

        Args:
            disparity_image: Disparity map image

        Returns:
            numpy.ndarray: Disparity map with legend
        """
        # Create image copy
        result = disparity_image.copy()

        # Add gradient bar
        h, w = result.shape[:2]
        bar_height = 20
        bar_margin = 10
        y_pos = h - bar_height - bar_margin
        x_start = w // 4
        x_end = w - w // 4
        bar_width = x_end - x_start

        # Draw gradient bar
        for i in range(bar_width):
            x = x_start + i
            # Create gradient color from blue to red (like COLORMAP_JET)
            if i < bar_width / 3:
                # Blue to cyan
                g = int(255 * (i / (bar_width / 3)))
                color = (255, g, 0)
            elif i < 2 * bar_width / 3:
                # Cyan to yellow
                r = int(255 * ((i - bar_width / 3) / (bar_width / 3)))
                color = (255 - r, 255, r)
            else:
                # Yellow to red
                b = int(255 * ((i - 2 * bar_width / 3) / (bar_width / 3)))
                color = (0, 255 - b, 255)

            cv2.line(result, (x, y_pos), (x, y_pos + bar_height), color, 1)

        # Add labels
        cv2.putText(
            result,
            "Far",
            (x_start - 40, y_pos + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            result,
            "Near",
            (x_end + 5, y_pos + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Add title
        title = "Disparity Map (blue=far, red=near)"
        cv2.putText(
            result,
            title,
            (w // 2 - 180, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        return result

    def compute_depth_map(self, disparity_float):
        """
        Compute depth map from disparity map using Q matrix.

        Args:
            disparity_float: Disparity map (float32)

        Returns:
            tuple: Depth map, normalized depth map and valid values mask
        """
        import logging

        with Timer("ComputeDepth", logger):
            try:
                if disparity_float is None:
                    Logger.log_json(
                        logger,
                        logging.ERROR,
                        event="depth_error",
                        error_message="Input disparity is None",
                    )
                    return None, None, None, None

                # Get Q matrix from calibration data
                Q = self.calibration_data["Q"]

                # Reproject to 3D points
                points_3d = cv2.reprojectImageTo3D(disparity_float, Q)

                # Extract Z-coordinate (depth)
                depth_map = points_3d[:, :, 2]

                # Filter invalid depths (too close, too far, or outliers)
                valid_mask = (
                    (depth_map > 0)
                    & (depth_map < 10000)
                    & (~np.isinf(depth_map))
                    & (~np.isnan(depth_map))
                )

                # Create mask for valid depth values
                valid_depth_mask = np.zeros_like(depth_map, dtype=np.uint8)
                valid_depth_mask[valid_mask] = 255

                # Discard outliers using percentiles
                valid_depths = depth_map[valid_mask]
                if valid_depths.size > 0:
                    min_depth = np.percentile(valid_depths, 5)  # 5th percentile
                    max_depth = np.percentile(valid_depths, 95)  # 95th percentile

                    # Limit depth values to remove outliers
                    clipped_depth = np.clip(depth_map, min_depth, max_depth)
                    clipped_depth[~valid_mask] = 0

                    # Normalize for visualization (invert to highlight near objects)
                    # Nearest objects will be brighter (whiter)
                    depth_normalized = np.zeros_like(clipped_depth, dtype=np.uint8)
                    normalized_values = 255 - cv2.normalize(
                        clipped_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                    )
                    depth_normalized[valid_mask] = normalized_values[valid_mask]

                    # Apply color map for visualization
                    depth_color = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

                    # Add depth information
                    cv2.putText(
                        depth_color,
                        f"Depth range: {min_depth:.2f}m - {max_depth:.2f}m",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                    Logger.log_json(
                        logger,
                        logging.INFO,
                        event="depth_computed",
                        depth_shape=depth_map.shape,
                        min_depth=float(min_depth),
                        max_depth=float(max_depth),
                        valid_points_percent=float(
                            np.sum(valid_mask) / valid_mask.size * 100
                        ),
                    )
                else:
                    # If no valid points
                    depth_color = np.zeros_like(disparity_float, dtype=np.uint8)
                    depth_normalized = depth_color.copy()

                    Logger.log_json(
                        logger,
                        logging.WARNING,
                        event="no_valid_depth_points",
                        error_message="No valid depth points",
                    )

                return depth_map, depth_normalized, valid_depth_mask, depth_color
            except Exception as e:
                Logger.log_json(
                    logger,
                    logging.ERROR,
                    event="depth_computation_error",
                    error_message=str(e),
                )
                import traceback

                Logger.log_data(
                    logger, logging.ERROR, "Error traceback: %s", traceback.format_exc()
                )
                return None, None, None, None

    def detect_top_surface(self, depth_normalized, original_image, percentage=20):
        """
        Detect the top surface of an object based on depth map.
        The top surface is the points closest to the camera.

        Args:
            depth_normalized: Normalized depth map (0-255)
            original_image: Original image
            percentage: Percentage of nearest points to consider as top surface

        Returns:
            tuple: Image with boundaries, top surface mask, contours,
                 top surface image, final result
        """
        import logging

        with Timer("DetectTopSurface", logger):
            try:
                if depth_normalized is None or original_image is None:
                    Logger.log_json(
                        logger,
                        logging.ERROR,
                        event="surface_detection_error",
                        error_message="Input data is None",
                    )
                    return None, None, None, None, None

                # Define threshold for top surface detection
                # Filter out black pixels (invalid)
                valid_pixels = depth_normalized[depth_normalized > 0]

                if valid_pixels.size == 0:
                    Logger.log_json(
                        logger,
                        logging.WARNING,
                        event="no_valid_depth_pixels",
                        error_message="No valid depth pixels",
                    )
                    return None, None, None, None, None

                # Threshold to select top percentage% of points
                # (closest to camera, i.e., brightest on inverted map)
                threshold = np.percentile(valid_pixels, 100 - percentage)

                # Binarize to extract top surface
                top_mask = np.zeros_like(depth_normalized, dtype=np.uint8)
                top_mask[depth_normalized > threshold] = 255

                # Morphological operations to improve mask
                kernel = np.ones((5, 5), np.uint8)
                top_mask = cv2.morphologyEx(top_mask, cv2.MORPH_CLOSE, kernel)
                top_mask = cv2.morphologyEx(top_mask, cv2.MORPH_OPEN, kernel)

                # Find top surface contours
                contours, _ = cv2.findContours(
                    top_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                # Filter contours by size
                min_contour_area = 200  # Adjust based on object size
                significant_contours = [
                    c for c in contours if cv2.contourArea(c) > min_contour_area
                ]

                # Sort contours by area (largest to smallest)
                significant_contours = sorted(
                    significant_contours, key=cv2.contourArea, reverse=True
                )

                # Create top surface mask with only significant contours
                refined_mask = np.zeros_like(top_mask)
                cv2.drawContours(refined_mask, significant_contours, -1, 255, -1)

                # Create result image with contours
                contour_img = original_image.copy()
                cv2.drawContours(contour_img, significant_contours, -1, (0, 255, 0), 2)

                # Apply mask to original image to highlight top surface
                top_surface = cv2.bitwise_and(
                    original_image, original_image, mask=refined_mask
                )

                # Create visually appealing result with semi-transparent overlay
                alpha = 0.6  # Transparency
                overlay = original_image.copy()
                cv2.drawContours(overlay, significant_contours, -1, (0, 255, 0), -1)
                final_result = cv2.addWeighted(
                    overlay, alpha, original_image, 1 - alpha, 0
                )
                cv2.drawContours(final_result, significant_contours, -1, (0, 255, 0), 2)

                Logger.log_json(
                    logger,
                    logging.INFO,
                    event="top_surface_detected",
                    threshold_value=int(threshold),
                    total_contours=len(contours),
                    significant_contours=len(significant_contours),
                )

                return (
                    contour_img,
                    refined_mask,
                    significant_contours,
                    top_surface,
                    final_result,
                )
            except Exception as e:
                Logger.log_json(
                    logger,
                    logging.ERROR,
                    event="surface_detection_error",
                    error_message=str(e),
                )
                import traceback

                Logger.log_data(
                    logger, logging.ERROR, "Error traceback: %s", traceback.format_exc()
                )
                return None, None, None, None, None

    def process_image(self, image_path, output_dir="output"):
        """
        Process stereo image to detect the top surface of an object.

        Args:
            image_path: Path to stereo image
            output_dir: Directory to save results

        Returns:
            bool: True on success, False on error
        """
        import logging

        overall_timer = Timer("ProcessImage", logger)
        overall_timer.start()

        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Split stereo image
            left_img, right_img = self.split_stereo_image(image_path)
            if left_img is None or right_img is None:
                return False

            # Save split images
            cv2.imwrite(os.path.join(output_dir, "left.png"), left_img)
            cv2.imwrite(os.path.join(output_dir, "right.png"), right_img)

            # Compute disparity map
            disp_color, disp_norm, disparity_float = self.compute_disparity(
                left_img, right_img
            )
            if disparity_float is None:
                return False

            # Save disparity map
            cv2.imwrite(os.path.join(output_dir, "disparity.png"), disp_color)

            # Compute depth map
            depth_map, depth_normalized, valid_depth_mask, depth_color = (
                self.compute_depth_map(disparity_float)
            )
            if depth_map is None:
                return False

            # Save depth map
            cv2.imwrite(os.path.join(output_dir, "depth.png"), depth_color)
            cv2.imwrite(
                os.path.join(output_dir, "depth_valid_mask.png"), valid_depth_mask
            )

            # Detect top surface based on depth map
            contour_img, top_mask, contours, top_surface, final_result = (
                self.detect_top_surface(depth_normalized, left_img, percentage=20)
            )
            if contour_img is None:
                return False

            # Save results
            cv2.imwrite(
                os.path.join(output_dir, "top_surface_contours.png"), contour_img
            )
            cv2.imwrite(os.path.join(output_dir, "top_surface_mask.png"), top_mask)
            cv2.imwrite(os.path.join(output_dir, "top_surface.png"), top_surface)
            cv2.imwrite(os.path.join(output_dir, "final_result.png"), final_result)

            # Combine original image, disparity, depth and result for comparison
            # Resize images for compatibility
            h, w = left_img.shape[:2]
            disp_color_resized = cv2.resize(disp_color, (w, h))
            depth_color_resized = cv2.resize(depth_color, (w, h))

            # First row: original and disparity
            top_row = np.hstack((left_img, disp_color_resized))
            # Second row: depth and result
            bottom_row = np.hstack((depth_color_resized, final_result))
            # Combine rows
            composite = np.vstack((top_row, bottom_row))

            cv2.imwrite(os.path.join(output_dir, "composite_result.png"), composite)

            # Save contour data in JSON format
            contour_data = []
            for i, contour in enumerate(contours):
                # Simplify contour for better visualization
                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Calculate center and area
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = 0, 0

                # Convert contour to list for saving
                contour_points = approx.reshape(-1, 2).tolist()

                contour_data.append(
                    {
                        "id": i,
                        "area": float(cv2.contourArea(contour)),
                        "center": [cx, cy],
                        "points": contour_points,
                    }
                )

            # Save contour data in JSON format
            import json

            with open(
                os.path.join(output_dir, "top_surface_contours.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(contour_data, f, indent=2, ensure_ascii=False)

            overall_timer.stop()
            self.processing_time = overall_timer.elapsed()

            Logger.log_json(
                logger,
                logging.INFO,
                event="processing_complete",
                output_dir=output_dir,
                processing_time=self.processing_time,
                contours_found=len(contours),
            )

            return True
        except Exception as e:
            overall_timer.stop()

            Logger.log_json(
                logger, logging.ERROR, event="processing_error", error_message=str(e)
            )
            import traceback

            Logger.log_data(
                logger, logging.ERROR, "Error traceback: %s", traceback.format_exc()
            )
            return False


def main():
    """Main function for processing stereo images."""
    import logging
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Process stereo images to detect the top surface of an object."
    )
    parser.add_argument(
        "--image", default="captures/top.png", help="Path to stereo image"
    )
    parser.add_argument(
        "--calibration",
        default="calibration/calibration_data/stereo_calibration.npz",
        help="Path to stereo camera calibration data",
    )
    parser.add_argument("--output", default="output", help="Output directory")
    args = parser.parse_args()

    # Configure root logger
    Logger.configure_root_logger(level=logging.WARNING)

    Logger.log_json(
        logger,
        logging.INFO,
        event="process_started",
        image_path=args.image,
        calibration_path=args.calibration,
        output_dir=args.output,
    )

    # Create top surface detector
    detector = TopSurfaceDetector(args.calibration)

    # Process stereo image
    start_time = time.time()
    success = detector.process_image(args.image, args.output)
    elapsed = time.time() - start_time

    if success:
        Logger.log_json(
            logger,
            logging.INFO,
            event="process_successful",
            status_text=f"Processing completed successfully in {elapsed:.2f} sec.",
        )
    else:
        Logger.log_json(
            logger,
            logging.ERROR,
            event="process_failed",
            error_message="Stereo image processing failed.",
        )

    return 0 if success else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        exit(exit_code)
    except Exception as e:
        import logging
        import traceback

        # Log unhandled exception
        Logger.log_json(
            logger,
            logging.CRITICAL,
            event="unhandled_exception",
            error_message=str(e),
            error_type=type(e).__name__,
        )

        # Log full call stack
        Logger.log_data(
            logger,
            logging.CRITICAL,
            "Unhandled exception traceback: %s",
            traceback.format_exc(),
        )

        # Exit with error code
        exit(1)
