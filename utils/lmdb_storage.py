from __future__ import annotations

import io
import json
import os
from typing import Any, Iterable, Iterator

import cv2
import lmdb
import numpy as np

from utils.logger import Logger, LoggerType


class LmdbBatch:
    """Context manager for atomic LMDB writes."""

    def __init__(self, env: lmdb.Environment) -> None:
        self._env = env
        self._txn = env.begin(write=True)

    def put(self, key: str, value: bytes) -> None:
        self._txn.put(key.encode(), value)

    def delete(self, key: str) -> None:
        self._txn.delete(key.encode())

    def commit(self) -> None:
        if self._txn:
            self._txn.commit()
            self._txn = None

    def __enter__(self) -> "LmdbBatch":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._txn is None:
            return
        if exc_type is None:
            self._txn.commit()
        else:
            self._txn.abort()
        self._txn = None


class LmdbStorage:
    """Simple LMDB storage for images, arrays and metadata."""

    def __init__(
        self, path: str, map_size: int = 1 << 30, readonly: bool = False
    ) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.env = lmdb.open(
            path, map_size=map_size, readonly=readonly, lock=not readonly, subdir=False
        )
        self.logger: LoggerType = Logger.get_logger("utils.lmdb_storage")
        self.logger.info(f"Opened LMDB at {path}")

    # --- basic primitives -------------------------------------------------
    def put(self, key: str, value: bytes) -> None:
        with self.env.begin(write=True) as txn:
            txn.put(key.encode(), value)

    def get(self, key: str) -> bytes | None:
        with self.env.begin() as txn:
            data = txn.get(key.encode())
            return bytes(data) if data else None

    def delete(self, key: str) -> None:
        with self.env.begin(write=True) as txn:
            txn.delete(key.encode())

    def iter_keys(self, prefix: str = "") -> Iterator[str]:
        pref = prefix.encode()
        with self.env.begin() as txn:
            cur = txn.cursor()
            if not cur.set_range(pref):
                return
            for k, _ in cur:
                if not k.startswith(pref):
                    break
                yield k.decode()

    def batch(self) -> LmdbBatch:
        return LmdbBatch(self.env)

    # --- helpers for common datatypes ------------------------------------
    def put_json(self, key: str, obj: Any) -> None:
        self.put(key, json.dumps(obj).encode())

    def get_json(self, key: str) -> Any | None:
        data = self.get(key)
        return json.loads(data.decode()) if data else None

    def put_array(self, key: str, array: np.ndarray) -> None:
        buf = io.BytesIO()
        np.save(buf, array)
        self.put(key, buf.getvalue())

    def get_array(self, key: str) -> np.ndarray | None:
        data = self.get(key)
        if data is None:
            return None
        buf = io.BytesIO(data)
        buf.seek(0)
        return np.load(buf)

    def put_image(self, key: str, img: np.ndarray) -> None:
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            raise ValueError("Image encode failed")
        self.put(key, buf.tobytes())

    def get_image(self, key: str) -> np.ndarray | None:
        data = self.get(key)
        if data is None:
            return None
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)


import sys
from utils.logger import Logger
from utils.lmdb_storage import LmdbStorage


def show_lmdb_keys(path: str, prefix: str = ""):
    store = LmdbStorage(path, readonly=True)
    print(f"LMDB at {path} keys with prefix '{prefix}':")
    for k in store.iter_keys(prefix):
        print(" -", k)


def show_lmdb_value(path: str, key: str):
    store = LmdbStorage(path, readonly=True)
    val = store.get(key)
    if val is None:
        print(f"Key '{key}' not found")
        return
    try:
        print(f"{key}: {val[:200]}... (raw bytes)")
        try:
            # Try as JSON
            import json

            print("As JSON:", store.get_json(key))
        except Exception:
            pass
        try:
            import numpy as np

            arr = store.get_array(key)
            if arr is not None:
                print("As numpy array, shape:", arr.shape, "dtype:", arr.dtype)
        except Exception:
            pass
    except Exception as e:
        print(f"Error decoding {key}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("LMDB Inspector")
    parser.add_argument("lmdb_path", help="Path to .lmdb")
    parser.add_argument("--list", action="store_true", help="List all keys")
    parser.add_argument("--prefix", default="", help="Filter keys by prefix")
    parser.add_argument("--show", help="Show value for key")
    args = parser.parse_args()

    if args.list:
        show_lmdb_keys(args.lmdb_path, args.prefix)
    elif args.show:
        show_lmdb_value(args.lmdb_path, args.show)
    else:
        print("Use --list or --show <key>")
