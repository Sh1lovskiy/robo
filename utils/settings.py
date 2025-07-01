from pathlib import Path
from dataclasses import dataclass

BASE_DIR = Path(__file__).parent.resolve()


@dataclass(frozen=True)
class Paths:
    """Convenient bundle of frequently used project paths."""
    CAPTURES_DIR: Path = BASE_DIR / "../calibration/captures_1"
    RESULTS_DIR: Path = BASE_DIR / "calibration" / "results2"
    CLOUD_DIR: Path = BASE_DIR / "clouds" / "captures_3"

    LOG_DIR: Path = BASE_DIR / "logs"
    VALIDATION_RESULTS_DIR: Path = BASE_DIR / "calibration" / "results2"


paths = Paths()

# RealSense depth unit to meters conversion
DEPTH_SCALE: float = 0.0001
