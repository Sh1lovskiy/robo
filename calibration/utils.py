from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import json
import numpy as np


def timestamp() -> str:
    """Return current timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_text(path: Path, text: str) -> None:
    """Write plain text to ``path`` creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def save_json(path: Path, data: Any) -> None:
    """Write JSON data to ``path`` creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_transform(base: Path, matrix: np.ndarray) -> None:
    """Save a transformation matrix to ``base`` with txt and json."""
    save_text(base.with_suffix(".txt"), np.array2string(matrix, precision=8))
    save_json(base.with_suffix(".json"), matrix.tolist())


def save_camera_params(base: Path, K: np.ndarray, dist: np.ndarray, rms: float) -> None:
    """Save camera intrinsics to ``base`` (.txt, .json and .xml)."""
    txt_lines = [f"RMS Error: {rms:.6f}", "camera_matrix:"]
    txt_lines.extend(" ".join(f"{v:.8f}" for v in row) for row in K)
    txt_lines.append("dist_coeffs:")
    txt_lines.append(" ".join(f"{v:.8f}" for v in dist.ravel()))
    save_text(base.with_suffix(".txt"), "\n".join(txt_lines))
    save_json(
        base.with_suffix(".json"),
        {"rms": rms, "camera_matrix": K.tolist(), "dist_coeffs": dist.tolist()},
    )
    fs = cv2.FileStorage(str(base.with_suffix(".xml")), cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", K)
    fs.write("dist_coeffs", dist)
    fs.release()
