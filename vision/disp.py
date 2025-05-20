"""
Stereo vision-based top surface detection for 3D objects.
Processes stereo images to generate disparity maps, depth maps,
and identifies the closest surface points.
"""

import cv2
import numpy as np
import os
import json
import argparse
from misc.logger import Logger, Timer

logger = Logger.setup_logger(
    "TopSurfaceDetector",
    json_format=True,
    json_fields={
        "component": "vision",
        "operation_type": "top_surface_detection",
    },
)


class TopSurfaceDetector:
    def __init__(self, calibration_path="calibration/stereo_calibration.npz"):
        """Initialize detector with stereo calibration data."""
        with Timer("Initialization", logger):
            self.calibration_data = self._load_calibration(calibration_path)
            self._init_matcher()
            self.processing_time = 0

    def _load_calibration(self, path):
        """Load stereo camera calibration parameters."""
        with Timer("LoadCalibration", logger):
            if not os.path.exists(path):
                logger.error(f"Calibration file not found: {path}")
                return None

            try:
                data = np.load(path)
                logger.info("Calibration loaded successfully")
                return data
            except Exception as e:
                logger.error(f"Calibration load error: {str(e)}")
                return None

    def _init_matcher(self):
        """Initialize stereo matching algorithm."""
        with Timer("InitStereoMatcher", logger):
            window_size = 5
            min_disp = 0
            num_disp = 16 * 10 - min_disp

            self.stereo = cv2.StereoSGBM_create(
                minDisparity=min_disp,
                numDisparities=num_disp,
                blockSize=window_size,
                P1=8 * 3 * window_size**2,
                P2=32 * 3 * window_size**2,
                disp12MaxDiff=1,
                uniquenessRatio=15,
                speckleWindowSize=100,
                speckleRange=32,
                preFilterCap=63,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
            )

            self.right_matcher = cv2.ximgproc.createRightMatcher(self.stereo)
            self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.stereo)
            self.wls_filter.setLambda(8000)
            self.wls_filter.setSigmaColor(1.5)

    def _split_image(self, image_path):
        """Split stereo image into left/right frames."""
        with Timer("SplitStereoImage", logger):
            if not os.path.exists(image_path):
                logger.error(f"Image not found: {image_path}")
                return None, None

            img = cv2.imread(image_path)
            if img is None:
                logger.error("Failed to read image")
                return None, None

            h, w = img.shape[:2]
            return img[:, : w // 2], img[:, w // 2 :]

    def _enhance_contrast(self, image):
        """Apply CLAHE for better feature detection."""
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if len(image.shape) == 3
            else image.copy()
        )
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def _compute_disparity(self, left, right):
        """Generate disparity map from stereo pair."""
        with Timer("ComputeDisparity", logger):
            gray_left = self._enhance_contrast(left)
            gray_right = self._enhance_contrast(right)

            left_disp = self.stereo.compute(gray_left, gray_right)
            right_disp = self.right_matcher.compute(gray_right, gray_left)
            filtered_disp = self.wls_filter.filter(
                left_disp, gray_left, None, right_disp
            )

            disp_float = filtered_disp.astype(np.float32) / 16.0
            valid_disp = disp_float[disp_float > 0]

            if valid_disp.size == 0:
                return None, None, None

            min_disp = np.percentile(valid_disp, 5)
            max_disp = np.percentile(valid_disp, 95)
            norm_disp = cv2.normalize(
                np.clip(disp_float, min_disp, max_disp),
                None,
                0,
                255,
                cv2.NORM_MINMAX,
                cv2.CV_8U,
            )
            return cv2.applyColorMap(norm_disp, cv2.COLORMAP_JET), norm_disp, disp_float

    def _compute_depth(self, disparity):
        """Convert disparity map to depth map."""
        with Timer("ComputeDepth", logger):
            if disparity is None or "Q" not in self.calibration_data:
                return None, None, None, None

            points_3d = cv2.reprojectImageTo3D(disparity, self.calibration_data["Q"])
            depth = points_3d[:, :, 2]
            valid_mask = (
                (depth > 0) & (depth < 10000) & (~np.isinf(depth)) & (~np.isnan(depth))
            )

            if not np.any(valid_mask):
                return None, None, None, None

            valid_depths = depth[valid_mask]
            min_depth = np.percentile(valid_depths, 5)
            max_depth = np.percentile(valid_depths, 95)

            clipped = np.clip(depth, min_depth, max_depth)
            norm_depth = 255 - cv2.normalize(
                clipped, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
            )
            return (
                depth,
                norm_depth,
                valid_mask.astype(np.uint8) * 255,
                cv2.applyColorMap(norm_depth, cv2.COLORMAP_JET),
            )

    def _detect_surface(self, depth_norm, image, percent=20):
        """Identify closest surface points from depth map."""
        with Timer("DetectTopSurface", logger):
            valid_pixels = depth_norm[depth_norm > 0]
            if valid_pixels.size == 0:
                return None, None, None, None, None

            threshold = np.percentile(valid_pixels, 100 - percent)
            mask = np.zeros_like(depth_norm, dtype=np.uint8)
            mask[depth_norm > threshold] = 255

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contours = [c for c in contours if cv2.contourArea(c) > 200]
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            refined = np.zeros_like(mask)
            cv2.drawContours(refined, contours, -1, 255, -1)

            contour_img = image.copy()
            cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

            surface = cv2.bitwise_and(image, image, mask=refined)
            result = cv2.addWeighted(
                cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), -1),
                0.6,
                image,
                0.4,
                0,
            )
            cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

            return contour_img, refined, contours, surface, result

    def process_image(self, image_path, output_dir="output"):
        """Full processing pipeline for stereo image."""
        overall_timer = Timer("ProcessImage", logger)
        overall_timer.start()

        try:
            os.makedirs(output_dir, exist_ok=True)
            left, right = self._split_image(image_path)
            if left is None:
                return False

            disp_color, disp_norm, disp_float = self._compute_disparity(left, right)
            if disp_float is None:
                return False

            depth_map, depth_norm, depth_mask, depth_color = self._compute_depth(
                disp_float
            )
            if depth_map is None:
                return False

            contour_img, top_mask, contours, surface, result = self._detect_surface(
                depth_norm, left
            )
            if contour_img is None:
                return False

            # Save all outputs
            outputs = {
                "left.png": left,
                "disparity.png": disp_color,
                "depth.png": depth_color,
                "depth_mask.png": depth_mask,
                "surface_contours.png": contour_img,
                "surface_mask.png": top_mask,
                "surface.png": surface,
                "result.png": result,
            }

            for name, img in outputs.items():
                cv2.imwrite(os.path.join(output_dir, name), img)

            # Save contour data
            contour_data = []
            for i, cnt in enumerate(contours):
                M = cv2.moments(cnt)
                cx = int(M["m10"] / M["m00"]) if M["m00"] else 0
                cy = int(M["m01"] / M["m00"]) if M["m00"] else 0

                contour_data.append(
                    {
                        "id": i,
                        "area": float(cv2.contourArea(cnt)),
                        "center": [cx, cy],
                        "points": cnt.squeeze().tolist(),
                    }
                )

            with open(os.path.join(output_dir, "contours.json"), "w") as f:
                json.dump(contour_data, f)

            overall_timer.stop()
            self.processing_time = overall_timer.elapsed()
            return True

        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            return False


def main():
    """Command line interface for surface detection."""
    parser = argparse.ArgumentParser(
        description="Top surface detection from stereo images"
    )
    parser.add_argument(
        "--image", default="captures/stereo.png", help="Input stereo image"
    )
    parser.add_argument(
        "--calibration", default="calibration/stereo.npz", help="Calibration file"
    )
    parser.add_argument("--output", default="output", help="Output directory")
    args = parser.parse_args()

    detector = TopSurfaceDetector(args.calibration)
    success = detector.process_image(args.image, args.output)

    if success:
        logger.info(f"Processing completed in {detector.processing_time:.2f}s")
    else:
        logger.error("Processing failed")

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
