from pathlib import Path
from dataclasses import dataclass

BASE_DIR = Path(__file__).parent.resolve()


@dataclass(frozen=True)
class Paths:
    CAPTURES_DIR: Path = BASE_DIR / "../calibration/captures_1"
    RESULTS_DIR: Path = BASE_DIR / "calibration" / "results2"
    CLOUD_DIR: Path = BASE_DIR / "clouds" / "captures_3"

    LOG_DIR: Path = BASE_DIR / "logs"
    VALIDATION_RESULTS_DIR: Path = BASE_DIR / "calibration" / "results2"

    # Aggregator-specific paths
    AGGREGATOR_DATA_DIR: Path = BASE_DIR / "clouds" / "captures_3"
    CHARUCO_XML: Path = RESULTS_DIR / "charuco_cam.xml"
    HANDEYE_TXT: Path = RESULTS_DIR / "handeye_TSAI.txt"
    POSES_FILE: Path = AGGREGATOR_DATA_DIR / "poses.json"
    PLY_ICP: Path = AGGREGATOR_DATA_DIR / "cloud_aggregated_icp.ply"
    PLY_NOICP: Path = AGGREGATOR_DATA_DIR / "cloud_aggregated.ply"


paths = Paths()
