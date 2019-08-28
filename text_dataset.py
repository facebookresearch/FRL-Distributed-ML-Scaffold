#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import logging
import time
from multiprocessing import Pool, RawArray
from typing import Any, Dict, List, NamedTuple, Sequence, Tuple

import numpy as np
import torch

from .storage import StoragePath, sizeof_fmt, storage_open, storage_os
from .storage_layers.dataset import (
    CachedDatasetAccessor,
    DatasetField,
    MultifieldDataset,
)
from .transform import MultifieldTransform
from .types import Split


logger = logging.getLogger(__name__)


N_CHUNK_BYTES = 100 * 1024 * 1024


class ChunkTask(NamedTuple):
    offset: int
    size: int


class ChunkLoader:
    path: StoragePath
    buff: RawArray
    split_char: str

    @classmethod
    def init(
        cls,
        path: StoragePath,
        # pyre-ignore RawData exists
        buff: RawArray,
        split_char: str,
    ) -> None:
        cls.path = path
        cls.buff = buff
        cls.split_char = split_char

    @classmethod
    def load_chunk(cls, task: ChunkTask) -> Tuple[ChunkTask, List[int]]:
        assert len(cls.split_char) == 1
        split_char_bytes = cls.split_char.encode()
        indices: List[int] = []
        with storage_open(cls.path, "rb") as f:
            f.seek(task.offset)
            chunk_data = f.read(task.size)
            cls.buff[task.offset : task.offset + task.size] = chunk_data
            search_start = 0
            while True:
                try:
                    idx = chunk_data.index(split_char_bytes, search_start)
                    indices.append(task.offset + idx + 1)
                    search_start = idx + 1
                except ValueError:
                    break
        return (task, indices)


class TextDataset(MultifieldDataset):
    FIELD_KEY = "line"
    PAD = 0

    def __init__(
        self,
        data_type: Split,
        txt_file_path: StoragePath,
        transform: MultifieldTransform,
        seq_len: int,
    ) -> None:
        # We assume text files will be small-ish (< 240 GB) and we can just
        # hold them in memory.
        n_bytes = storage_os.path.getsize(txt_file_path)

        # pyre-ignore RawArray exists
        self._data = RawArray("c", n_bytes)
        self._sample_indices: List[int] = []
        chunks = [
            ChunkTask(offset, min(n_bytes - offset, N_CHUNK_BYTES))
            for offset in range(0, n_bytes, N_CHUNK_BYTES)
        ]
        logger.info(
            "Loading %s (%s in %d chunks)",
            txt_file_path,
            sizeof_fmt(n_bytes),
            len(chunks),
        )
        tic = time.time()
        progress_bytes = 0
        with Pool(
            32, initializer=ChunkLoader.init, initargs=(txt_file_path, self._data, "\n")
        ) as p:
            for chunk, indices in p.imap_unordered(ChunkLoader.load_chunk, chunks):
                self._sample_indices += indices
                progress_bytes += chunk.size
                if (time.time() - tic) > 15:
                    logger.info("Loaded %d of %d bytes", progress_bytes, n_bytes)
                    tic = time.time()
        logger.info("Loaded %s, sorting indices", txt_file_path)
        self._sample_indices.append(0)
        self._sample_indices = sorted(self._sample_indices)

        if self._sample_indices[-1] != n_bytes - 1:
            self._sample_indices.append(n_bytes - 1)
        logger.info("Found a total of %d samples", len(self._sample_indices) - 1)
        self._seq_len = seq_len + 1

        self._np_data = np.frombuffer(self._data, dtype=np.uint8)
        self.data_type = data_type
        self._transform = transform

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._np_data = np.frombuffer(self._data, dtype=np.uint8)

    def __getstate__(self) -> Dict[str, Any]:
        """
        When pickling we need to avoid trying to pickle numpy arrays backed by
        anything fancy (memmap, or shared memory in this case), because numpy's
        pickling logic is naive and will attempt to serialize the entire
        contents of any np.ndarrays.
        """
        state = self.__dict__.copy()
        del state["_np_data"]
        return state

    def __len__(self) -> int:
        return len(self._sample_indices) - 1

    def get_raw_item(self, idx: int) -> Dict[DatasetField, np.ndarray]:
        full_sample = self._np_data[
            self._sample_indices[idx] : self._sample_indices[idx + 1] - 1
        ]
        sample_len = min(self._seq_len, full_sample.shape[0])
        packed_sample = np.ndarray((self._seq_len,), dtype=full_sample.dtype)
        packed_sample.fill(self.PAD)
        packed_sample[:sample_len] = full_sample[:sample_len]
        return {self.FIELD_KEY: packed_sample}

    def __getitem__(
        self, idx: int
    ) -> Tuple[Sequence[torch.Tensor], Sequence[torch.Tensor], Dict[str, Any]]:
        data = self.get_raw_item(idx)
        return self._transform(data, split=self.data_type)

    def set_accessor(self, accessor: CachedDatasetAccessor) -> None:
        pass
