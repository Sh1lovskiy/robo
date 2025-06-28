from __future__ import annotations

from typing import Iterable

import rocksdb

from utils.logger import Logger

from .interface import IStorage, IStorageBatch


class RocksDBBatch(IStorageBatch):
    """Context manager around :class:`rocksdb.WriteBatch`."""

    def __init__(self, db: rocksdb.DB) -> None:
        self.db = db
        self.batch = rocksdb.WriteBatch()

    def put(self, key: str, value: bytes) -> None:
        self.batch.put(key.encode(), value)

    def delete(self, key: str) -> None:
        self.batch.delete(key.encode())

    def commit(self) -> None:
        self.db.write(self.batch)

    def __enter__(self) -> "RocksDBBatch":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.commit()


class RocksDBStorage(IStorage):
    """RocksDB based implementation of :class:`IStorage`."""

    def __init__(self, db_path: str) -> None:
        opts = rocksdb.Options()
        opts.create_if_missing = True
        self.db = rocksdb.DB(db_path, opts)
        self.logger = Logger.get_logger("storage.rocksdb")
        self.logger.info(f"Opened RocksDB at {db_path}")

    def put(self, key: str, value: bytes) -> None:
        self.db.put(key.encode(), value)

    def get(self, key: str) -> bytes | None:
        return self.db.get(key.encode())

    def delete(self, key: str) -> None:
        self.db.delete(key.encode())

    def batch(self) -> RocksDBBatch:
        return RocksDBBatch(self.db)

    def iter_prefix(self, prefix: str) -> Iterable[str]:
        it = self.db.iterkeys()
        pref = prefix.encode()
        it.seek(pref)
        for k in it:
            if not k.startswith(pref):
                break
            yield k.decode()
