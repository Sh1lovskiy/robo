from __future__ import annotations

import io
import json
from abc import ABC, abstractmethod
from typing import Any, Iterable

import numpy as np
import cv2


class IStorageBatch(ABC):
    """Batch interface for atomic writes."""

    @abstractmethod
    def put(self, key: str, value: bytes) -> None:
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        pass

    @abstractmethod
    def commit(self) -> None:
        pass

    def put_json(self, key: str, obj: Any) -> None:
        self.put(key, json.dumps(obj).encode())

    def put_array(self, key: str, array: np.ndarray) -> None:
        buf = io.BytesIO()
        np.save(buf, array)
        self.put(key, buf.getvalue())

    def put_image(self, key: str, img: np.ndarray) -> None:
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            raise ValueError("Image encode failed")
        self.put(key, buf.tobytes())


class IStorage(ABC):
    """Abstract storage backend."""

    @abstractmethod
    def put(self, key: str, value: bytes) -> None:
        pass

    @abstractmethod
    def get(self, key: str) -> bytes | None:
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        pass

    @abstractmethod
    def batch(self) -> IStorageBatch:
        pass

    @abstractmethod
    def iter_prefix(self, prefix: str) -> Iterable[str]:
        pass

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
