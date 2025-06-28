from __future__ import annotations

import os
from typing import Iterable

import lmdb

from utils.logger import Logger

from .interface import IStorage, IStorageBatch


class LmdbBatch(IStorageBatch):
    """Context manager around :class:`lmdb.Transaction`."""

    def __init__(self, env: lmdb.Environment) -> None:
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
        if self._txn:
            if exc_type is None:
                self._txn.commit()
            else:
                self._txn.abort()
            self._txn = None


class LmdbStorage(IStorage):
    """LMDB based implementation of :class:`IStorage`."""

    def __init__(self, db_path: str, map_size: int = 1 << 30) -> None:
        os.makedirs(db_path, exist_ok=True)
        self.env = lmdb.open(db_path, map_size=map_size)
        self.logger = Logger.get_logger("storage.lmdb")
        self.logger.info(f"Opened LMDB at {db_path}")

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

    def batch(self) -> LmdbBatch:
        return LmdbBatch(self.env)

    def iter_prefix(self, prefix: str) -> Iterable[str]:
        pref = prefix.encode()
        with self.env.begin() as txn:
            cur = txn.cursor()
            if not cur.set_range(pref):
                return
            for k, _ in cur:
                if not k.startswith(pref):
                    break
                yield k.decode()
