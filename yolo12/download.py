#!/usr/bin/env python3
"""
Download YOLO12 weights into ./yolo12/models.

Strategy:
1) Try GitHub 'latest' assets:
   https://github.com/ultralytics/assets/releases/latest/download/<file>
2) If missing, trigger Ultralytics auto-download via YOLO("<file>") and copy
   from common caches into ./yolo12/models.

Usage:
  python download_yolo12_weights.py
  python download_yolo12_weights.py --tasks detect seg pose cls obb \
      --scales n s m l x
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import requests
from tqdm import tqdm

# ------------------------------ CONSTANTS ------------------------------------

ASSETS_BASE: str = "https://github.com/ultralytics/assets/releases/latest/download/"
OUT_DIR: Path = Path("yolo12/models")
CHUNK_SIZE: int = 1024 * 1024  # 1 MiB
HTTP_TIMEOUT: int = 30  # seconds

# Map task -> filename suffix.
TASK_SUFFIX: Dict[str, str] = {
    "detect": "",
    "seg": "-seg",
    "pose": "-pose",
    "cls": "-cls",
    "obb": "-obb",
}

DEFAULT_SCALES: List[str] = ["n", "s", "m", "l", "x"]
DEFAULT_TASKS: List[str] = ["detect", "seg", "pose", "cls", "obb"]

# Candidate locations where Ultralytics may cache weights.
CACHE_CANDIDATES: List[Path] = [
    Path(os.environ.get("YOLO_WEIGHTS_CACHE", "")),
    Path.home() / ".cache" / "ultralytics",
    Path.home() / ".cache" / "torch" / "hub",
]

ULTRALYTICS_AVAILABLE: bool = find_spec("ultralytics") is not None

# ------------------------------ UTILITIES ------------------------------------


def build_filenames(tasks: Iterable[str], scales: Iterable[str]) -> List[str]:
    """Return list of weight filenames from tasks and scales."""
    out: List[str] = []
    for task in tasks:
        suf = TASK_SUFFIX[task]
        for sc in scales:
            out.append(f"yolo12{sc}{suf}.pt")
    return out


def _copy_if_exists(src: Path, dst: Path) -> bool:
    """Copy src to dst if present."""
    if not src.exists() or not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _find_in_caches(name: str) -> Path | None:
    """Search known caches for a file with given name."""
    for base in CACHE_CANDIDATES:
        if not base:
            continue
        candidate = base / name
        if candidate.exists():
            return candidate
        # Walk one level deeper (common subfolders).
        if base.exists():
            for sub in base.iterdir():
                cand = sub / name
                if cand.exists():
                    return cand
    return None


def download_stream(url: str, dst: Path) -> bool:
    """
    Stream-download URL to dst atomically.

    Writes into dst.with_suffix('.part') and renames on success.
    """
    try:
        with requests.get(url, stream=True, timeout=HTTP_TIMEOUT) as r:
            if r.status_code != 200:
                return False
            total = int(r.headers.get("content-length", 0))
            tmp = dst.with_suffix(".part")
            dst.parent.mkdir(parents=True, exist_ok=True)
            with (
                open(tmp, "wb") as f,
                tqdm(
                    total=total if total > 0 else None,
                    unit="B",
                    unit_scale=True,
                    desc=dst.name,
                    dynamic_ncols=True,
                ) as pbar,
            ):
                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    f.write(chunk)
                    if total > 0:
                        pbar.update(len(chunk))
            tmp.replace(dst)
        return True
    except Exception:
        return False


def fallback_ultralytics_pull(name: str, dst: Path) -> bool:
    """
    Trigger Ultralytics auto-download and copy weight into dst.

    Returns True if the file ends up at dst.
    """
    if not ULTRALYTICS_AVAILABLE:
        return False
    try:
        from ultralytics import YOLO

        _ = YOLO(name)  # side effect: ensure weight is downloaded
    except Exception:
        return False

    # Try known caches and CWD.
    found = _find_in_caches(name) or Path(name)
    if found and found.exists():
        return _copy_if_exists(found, dst)
    return False


def fetch_one(name: str) -> Tuple[str, bool, str]:
    """
    Ensure a single weight file is present in OUT_DIR.

    Returns (name, success, how).
    """
    out = OUT_DIR / name
    if out.exists() and out.stat().st_size > 0:
        return name, True, "exists"

    url = ASSETS_BASE + name
    if download_stream(url, out):
        return name, True, "downloaded:latest"

    if fallback_ultralytics_pull(name, out):
        return name, True, "downloaded:ultralytics"

    if out.exists() and out.stat().st_size == 0:
        out.unlink(missing_ok=True)
    return name, False, "missing"


def parse_args(argv: List[str]) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Download YOLO12 weights into ./yolo12/models"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
        choices=list(TASK_SUFFIX.keys()),
        help="Task heads to fetch",
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        default=DEFAULT_SCALES,
        choices=DEFAULT_SCALES,
        help="Model scales to fetch",
    )
    return parser.parse_args(argv)


def print_summary(ok: List[Tuple[str, str]], fail: List[Tuple[str, str]]) -> None:
    """Print a human-friendly summary."""
    print("\nSummary:")
    print(f"  downloaded: {len(ok)}")
    print(f"  missing   : {len(fail)}")
    if fail:
        print("Missing files:")
        for n, how in fail:
            print(f"  - {n} ({how})")


# --------------------------------- MAIN --------------------------------------


def main(argv: List[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv or sys.argv[1:])
    names = build_filenames(args.tasks, args.scales)

    ok: List[Tuple[str, str]] = []
    fail: List[Tuple[str, str]] = []

    for n in names:
        name, success, how = fetch_one(n)
        (ok if success else fail).append((name, how))
        status = "OK" if success else "FAIL"
        print(f"[{status}] {name} ({how})")

    print_summary(ok, fail)
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
