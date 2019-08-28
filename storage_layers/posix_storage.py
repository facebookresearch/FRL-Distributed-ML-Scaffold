#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import os
from multiprocessing import synchronize
from typing import Iterator, List, Tuple

import numpy as np

from .base_path import StoragePath
from .base_storage import StorageLayer
from .dataset import ChunkData, IndexedDatasetReader
from .posix_path import PosixStorageType


class PosixStorageLayer(StorageLayer):
    scheme = PosixStorageType.scheme

    def makedirs(self, exist_ok: bool, ttl: int) -> None:
        original_umask: int
        try:
            original_umask = os.umask(0)
            os.makedirs(self._storage_path.path, 0o777, exist_ok)
        finally:
            os.umask(original_umask)

    def open(self, mode: str):
        return open(self._storage_path.path, mode)

    def getsize(self) -> int:
        return os.path.getsize(self._storage_path.path)

    def getmtime(self) -> float:
        return os.path.getmtime(self._storage_path.path)

    def exists(self) -> bool:
        return os.path.exists(self._storage_path.path)

    def walk(self) -> Iterator[Tuple[str, List[str], List[str]]]:
        yield from os.walk(self._storage_path.path, followlinks=True)


class PosixIndexedDatasetReader(IndexedDatasetReader):

    scheme = PosixStorageLayer.scheme

    def __init__(self, *, idxfile: StoragePath, binfile: StoragePath) -> None:
        self.datafilename = binfile.path

        idx = np.fromfile(str(idxfile.path), dtype="int64")
        self._init_from_index_data(idx)
        self.data = np.memmap(str(self.datafilename), dtype=self.dtype, mode="r")

    def __getitem__(self, index: int):
        assert index >= 0 and index < self.N, "index out of range"

        data = self.data[self.framesize * index : self.framesize * (index + 1)]
        # return a cloned version of original data
        return data.reshape(self.size).copy()

    def __setstate__(self, state):
        state["data"] = self.data = np.memmap(
            state["datafilename"], dtype=state["dtype"], mode="r"
        )
        self.__dict__.update(state)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["data"]
        return state

    def get_chunk_lock(self, frame_idx: int) -> synchronize.Lock:
        raise NotImplementedError("Not supported for posix")

    def get_chunk_data(self, frame_idx: int) -> ChunkData:
        raise NotImplementedError("Not supported for posix")
