"""Camera interfaces and 3D vision helpers.

The vision package contains camera drivers, point cloud tools and transformation
utilities used throughout the robotics stack.  Open3D and OpenCV are leveraged
to convert depth images to point clouds, apply filters and perform coordinate
transforms.  Subpackages provide reusable components that can be combined in
custom pipelines or accessed through the provided CLI tools.
"""

from .camera import RealSenseD415
from .transform import TransformUtils
from .handeye import EyeInHandCalibrator
from .pointcloud.generator import PointCloudGenerator
from .pointcloud.aggregator import RGBDAggregator
from .pointcloud.analyzer import CloudAnalyzer
from .pointcloud.builder import PointCloudBuilder
from .realsense_bag import FrameRecorder, BagFilterVisualizer


__all__ = [
    "RealSenseCamera",
    "RealSenseD415",
    "TransformUtils",
    "EyeInHandCalibrator",
    "PointCloudGenerator",
    "RGBDAggregator",
    "CloudAnalyzer",
    "PointCloudBuilder",
    "D415StreamConfig",
    "D415CameraSettings",
    "D415FilterConfig",
    "FrameRecorder",
    "BagFilterVisualizer",
]
