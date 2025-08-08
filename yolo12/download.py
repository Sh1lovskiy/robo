#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download all available YOLO12 weights into ./models

- Tries GitHub "latest" assets first:
    https://github.com/ultralytics/assets/releases/latest/download/<filename>
- Falls back to Ultralytics auto-download via `YOLO("<filename>")`
  and copies the cached .pt into ./models.

Usage:
  python download_yolo12_weights.py
  python download_yolo12_weights.py --tasks detect seg pose cls obb --scales n s m l x
"""

import sys
import os
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple
import requests
from tqdm import tqdm

# Optional: only import ultralytics if needed in fallback
_ULTRA_OK = True
try:
    from ultralytics import YOLO  # noqa: F401
except Exception:
    _ULTRA_OK = False

ASSETS_BASE = "https://github.com/ultralytics/assets/releases/latest/download/"
OUT_DIR = Path("yolo12/models")

# Task suffixes follow the Ultralytics naming convention:
#  detect: yolo12n.pt
#  seg:    yolo12n-seg.pt
#  pose:   yolo12n-pose.pt
#  cls:    yolo12n-cls.pt
#  obb:    yolo12n-obb.pt
TASK_SUFFIX = {
    "detect": "",
    "seg": "-seg",
    "pose": "-pose",
    "cls": "-cls",
    "obb": "-obb",
}

DEFAULT_SCALES = ["n", "s", "m", "l", "x"]
DEFAULT_TASKS = ["detect", "seg", "pose", "cls", "obb"]


def filenames(tasks: List[str], scales: List[str]) -> List[str]:
    names = []
    for t in tasks:
        suf = TASK_SUFFIX[t]
        for s in scales:
            names.append(f"yolo12{s}{suf}.pt")
    return names


def _download_stream(url: str, dst: Path) -> bool:
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            if r.status_code != 200:
                return False
            total = int(r.headers.get("content-length", 0))
            tmp = dst.with_suffix(".part")
            with (
                open(tmp, "wb") as f,
                tqdm(
                    total=total if total > 0 else None,
                    unit="B",
                    unit_scale=True,
                    desc=dst.name,
                ) as pbar,
            ):
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        if total > 0:
                            pbar.update(len(chunk))
            tmp.replace(dst)
        return True
    except Exception:
        return False


def _fallback_ultralytics_pull(name: str, dst: Path) -> bool:
    """Use Ultralytics to auto-download and then copy .pt to dst."""
    if not _ULTRA_OK:
        return False
    try:
        # Trigger internal download; this returns a model object, but download is side-effect.
        from ultralytics import YOLO

        _ = YOLO(name)  # downloads into Ultralytics cache if missing

        # Find file in common cache paths
        # Ultralytics typically places weights under ~/.cache/ultralytics or torch hub paths.
        candidates = [
            Path.home() / ".cache" / "ultralytics" / name,
            Path.home() / ".cache" / "torch" / "hub" / name,
        ]
        for c in candidates:
            if c.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(c, dst)
                return True

        # If YOLO(name) downloaded into current working dir
        local = Path(name)
        if local.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(local), dst)
            return True
    except Exception:
        return False
    return False


def fetch_one(name: str) -> Tuple[str, bool, str]:
    out = OUT_DIR / name
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        return name, True, "exists"

    # Try GitHub latest assets
    url = ASSETS_BASE + name
    if _download_stream(url, out):
        return name, True, "downloaded:latest"

    # Fallback: use ultralytics auto-download
    if _fallback_ultralytics_pull(name, out):
        return name, True, "downloaded:ultralytics"

    # Clean up partial
    if out.exists() and out.stat().st_size == 0:
        out.unlink(missing_ok=True)

    return name, False, "missing"


def main():
    parser = argparse.ArgumentParser(
        description="Download YOLO12 weights into ./models"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
        choices=list(TASK_SUFFIX.keys()),
        help="Which task heads to fetch",
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        default=DEFAULT_SCALES,
        choices=DEFAULT_SCALES,
        help="Which model scales to fetch",
    )
    args = parser.parse_args()

    names = filenames(args.tasks, args.scales)
    ok, fail = [], []
    for n in names:
        name, success, how = fetch_one(n)
        (ok if success else fail).append((name, how))
        status = "OK" if success else "FAIL"
        print(f"[{status}] {name} ({how})")

    print("\nSummary:")
    print("  downloaded:", len(ok))
    print("  missing   :", len(fail))
    if fail:
        print("Missing files:")
        for n, how in fail:
            print(f"  - {n} ({how})")


if __name__ == "__main__":
    sys.exit(main())
