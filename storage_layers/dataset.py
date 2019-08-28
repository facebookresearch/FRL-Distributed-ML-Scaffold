#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import bisect
import logging
import os
from abc import ABC, abstractmethod
from enum import IntEnum
from functools import reduce
from io import BytesIO
from itertools import chain
from math import ceil
from multiprocessing import Lock, RawArray, synchronize
from operator import mul
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Sized, Tuple

import numpy as np
import psutil
import torch
from torch.utils.data import ConcatDataset, Dataset, Subset

from ..types import Split


logger = logging.getLogger(__name__)


class ChunkData(NamedTuple):
    first_frame_idx: int
    data: bytes
    bytes_per_frame: int

    def get_frame(self, frame_idx: int) -> bytes:
        intra_chunk_offset = (frame_idx - self.first_frame_idx) * self.bytes_per_frame
        return self.data[intra_chunk_offset : intra_chunk_offset + self.bytes_per_frame]


class DataStorage(ABC):
    """
    Abstract class which can load chunks of data from a data source.
    """

    @abstractmethod
    def get_chunk_lock(self, frame_idx: int) -> synchronize.Lock:
        """
        Determinstically fetch a lock for a specific chunk of data.
        """
        pass

    @abstractmethod
    def get_chunk_data(self, frame_idx: int) -> ChunkData:
        """
        Get a chunk of data containing a specific video frame from a dataset.
        This is likely more than a single frame of data in order to cut down on
        the overhead of network calls. The number of bytes in a chunk must be an
        integer multiple of the size of a single frame of data.
        """
        pass


"""
DatasetField is an identifier for a specific type of data in a dataset. Each item
in a dataset will be a dictionary with keys (DatasetField) and values (np.ndarray).

The set of fields will be identical for every item in the dataset. The np.ndarray
for a specific field will always have the same shape/type for every item. The field
"mono" might always point to one frame of monochrome data, for example, and another
field "sitstand" might always point to a single element np.ndarray with a binary
label
"""
DatasetField = str


class FrameDesc(NamedTuple):
    dims: Tuple[int, ...]
    num_bytes: int
    dtype: np.dtype


class SharedMemoryFrames(NamedTuple):
    data: RawArray
    statuses: RawArray
    indices: RawArray


class SharedMemoryDatasetCache:
    """
    With DistributedDataParllel we have 1 process per GPU, each with N worker
    threads loading frames of data. With high latency data sources we want to
    fetch a chunk of data at once to ammortize this. When a thread loads a chunk,
    it should extract not only the frame it was trying to load, but also future
    frames that are expected to be loaded by other threads.

    In order for processes to share data and minimize calls to the storage layer
    we have to use shared memory buffers for the cache. This class maintains
    those buffers and facilitates sharing large amounts of data across processes.
    """

    TARGET_MEMORY_UTILIZATION = 0.85
    MIN_ALLOWED_CACHE_SIZE = 1440  # in cache lines, not in bytes

    def __init__(
        self,
        dataset: "MultifieldDataset",
        local_worker_count: int,
        total_worker_count: int,
        memory_quota: int,
    ) -> None:
        assert len(dataset) > 0, "Cannot build cache for empty dataset"

        # all available physical memory on the machine according to docs
        total_physical_memory: int = psutil.virtual_memory().total

        logger.info(f"Setting up shared dataset cache:")
        logger.info(f"\tshared cache pid: {os.getpid()}")
        logger.info(f"\tdataset size: {len(dataset)} objects")
        logger.info(f"\tlocal worker process count: {local_worker_count}")
        logger.info(f"\tcross-machine worker process count: {total_worker_count}")
        logger.info(f"\tmemory quota: {memory_quota} bytes")
        logger.info(f"\ttotal physical memory: {total_physical_memory} bytes")

        if memory_quota == 0:
            memory_quota = total_physical_memory
        logger.info(f"\teffective memory quota: {memory_quota} bytes")

        memory_quota = ceil(memory_quota * self.TARGET_MEMORY_UTILIZATION)
        logger.info(f"\ttarget memory budget: {memory_quota} bytes")

        self.local_worker_count = local_worker_count

        # In order to build a cache for a MultifieldIndexedDataset we need to
        # know what fields exist and what the type/size of each field is. By
        # design every frame of a dataset will have the same fields and the same
        # size data, so we pick an arbitrary frame (the 0th frame) to measure
        # these properties.
        canonical_sample: Dict[DatasetField, np.ndarray] = dataset.get_raw_item(0)

        # We generate more locks than there are processes, by some arbitrary
        # factor. Too many locks and it gets expensive to pass them around,
        # too few and we will contend all the time.
        self.chunk_locks = [Lock() for _ in range(local_worker_count * 8)]

        self.frame_desc: Dict[DatasetField, FrameDesc] = {
            multifield_dataset_field: FrameDesc(
                dims=data.shape,
                num_bytes=int(reduce(mul, data.shape, 1) * data.itemsize),
                dtype=data.dtype,
            )
            for multifield_dataset_field, data in canonical_sample.items()
        }
        self.sequential_idx_to_frame_idx: Optional[torch.LongTensor] = None
        self.sorted_frame_idx: Optional[torch.LongTensor] = None
        self.sorted_frame_idx_to_sequential_idx: Optional[torch.LongTensor] = None

        cache_line_size: int = sum(d.num_bytes for d in self.frame_desc.values())
        logger.info(f"\tcache line will be of size: {cache_line_size} bytes")

        num_cache_lines = memory_quota // cache_line_size
        logger.info(f"\tmax cache lines for memory quota: {num_cache_lines} objects")

        self.cache_lines_per_worker = num_cache_lines // local_worker_count
        logger.info(
            f"\tmax cache lines per worker: {self.cache_lines_per_worker} objects"
        )

        self.cache_lines_per_worker = min(
            self.cache_lines_per_worker, ceil(len(dataset) / local_worker_count)
        )
        logger.info(
            f"\tactual cache lines per worker: {self.cache_lines_per_worker} objects"
        )

        self.cache_lines = self.cache_lines_per_worker * local_worker_count
        logger.info(
            f"final shared cache size: {self.cache_lines} objects or {self.cache_lines * cache_line_size} bytes"
        )

        # Shared memory buffers
        self.smem_frames: Dict[DatasetField, SharedMemoryFrames] = {
            multifield_dataset_field: SharedMemoryFrames(
                # pyre-ignore RawArray exists
                data=RawArray("c", self.cache_lines * desc.num_bytes),
                # pyre-ignore RawArray exists
                statuses=RawArray("i", self.cache_lines),
                # pyre-ignore RawArray exists
                indices=RawArray("l", self.cache_lines),
            )
            for multifield_dataset_field, desc in self.frame_desc.items()
        }

        # Numpy ndarray pre-bound to shared memory buffer
        self.np_ndarray_smem_frame_data = self._get_frame_data_as_np_ndarray()

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.np_ndarray_smem_frame_data = self._get_frame_data_as_np_ndarray()

    def __getstate__(self) -> Dict[str, Any]:
        """
        When pickling we need to avoid trying to pickle numpy arrays backed by
        anything fancy (memmap, or shared memory in this case), because numpy's
        pickling logic is naive and will attempt to serialize the entire
        contents of any np.ndarrays.
        """
        state = self.__dict__.copy()
        del state["np_ndarray_smem_frame_data"]
        return state

    def _get_frame_data_as_np_ndarray(self) -> Dict[DatasetField, np.ndarray]:
        return {
            multifield_dataset_field: np.frombuffer(
                self.smem_frames[multifield_dataset_field].data, dtype=desc.dtype
            ).reshape(-1, *desc.dims)
            for multifield_dataset_field, desc in self.frame_desc.items()
        }

    def set_sequence_indices(self, process_idx: int, frame_indices: List[int]) -> None:
        """
        For a particular process, set the planned access pattern for frames in the
        dataset. This may not include all frames in the dataset if we are running
        with DistributedDataParallel. This is required for prefetching.

        All processes participate in data fetching, and when any process loads a
        chunk of data it will cache data both for itself and for other processes
        based on this shared record of planned frame access.
        """
        # pyre-ignore share_memory_ exists
        self.sequential_idx_to_frame_idx = torch.LongTensor(
            frame_indices
        ).share_memory_()
        sort_result = self.sequential_idx_to_frame_idx.sort()
        self.sorted_frame_idx = sort_result.values.share_memory_()
        self.sorted_frame_idx_to_sequential_idx = sort_result.indices.share_memory_()

        sequence_size = min(len(frame_indices), self.cache_lines_per_worker)
        s = slice(
            process_idx,
            process_idx + self.local_worker_count * sequence_size,
            self.local_worker_count,
        )
        for field_frames in self.smem_frames.values():
            field_frames.indices[s] = frame_indices[: self.cache_lines_per_worker]
            field_frames.statuses[s] = [FrameStatus.NOT_LOADED] * sequence_size

    def _read_item(
        self,
        multifield_dataset_field: DatasetField,
        *,
        sequential_idx: int,
        buffer_idx: int,
    ) -> np.ndarray:
        """
        Each item in the cache is read only once per epoch. This handles reading
        an item and recycling the cache.
        """

        assert self.sequential_idx_to_frame_idx is not None

        field_frames = self.smem_frames[multifield_dataset_field]
        field_frames.statuses[buffer_idx] = FrameStatus.NOT_LOADED

        next_sequential_idx = sequential_idx + self.cache_lines_per_worker
        next_frame_idx = (
            # pyre-fixme[16]: `Optional` has no attribute `__getitem__`.
            self.sequential_idx_to_frame_idx[next_sequential_idx].item()
            # pyre-fixme[6]: Expected `Sized` for 1st param but got
            #  `Optional[LongTensor]`.
            if next_sequential_idx < len(self.sequential_idx_to_frame_idx)
            else -1
        )
        field_frames.indices[buffer_idx] = next_frame_idx

        # make sure to copy resulting slice to prevent pinning a larger frame
        # in memory
        return self.np_ndarray_smem_frame_data[multifield_dataset_field][
            buffer_idx
        ].copy()

    def read_frame_through_cache(
        self,
        multifield_dataset_field: DatasetField,
        data_storage: DataStorage,
        *,
        process_idx: int,
        dataset_local_frame_idx: int,
        dataset_global_offset: int,
    ) -> np.ndarray:
        """
        Reads a single frame of data out of the cache if it exists, or from the
        storage layer otherwise. If the frame is read from the storage layer this
        will also handle caching the requested frame and other
        nearby/soon-to-be-read frames fetched as part of the same contiguous data
        chunk.
        """

        # The dataset being read could be one dataset out of many that are
        # concatenated together. Since the shared memory cache is common for all
        # processes and datasets we need to transform back into a global index in
        # top level (concatenated) dataset to read from the cache.
        global_frame_idx = dataset_global_offset + dataset_local_frame_idx

        assert self.sorted_frame_idx is not None
        assert self.sorted_frame_idx_to_sequential_idx is not None
        sorted_frame_idx = bisect.bisect_left(
            # pyre-fixme[16]: `Optional` has no attribute `numpy`.
            self.sorted_frame_idx.numpy(),
            global_frame_idx,
        )
        # pyre-fixme[16]: `Optional` has no attribute `__getitem__`.
        sequential_idx = self.sorted_frame_idx_to_sequential_idx[
            sorted_frame_idx
        ].item()

        # We interleave frame caches for each process, so we need to take this
        # into account when we index into the cache to get the right data subset.
        strided_buffer_idx: Optional[int] = None

        strided_buffer_idx_candidate = (
            process_idx + sequential_idx * self.local_worker_count
        ) % self.cache_lines
        field_frames = self.smem_frames[multifield_dataset_field]
        if global_frame_idx == field_frames.indices[strided_buffer_idx_candidate]:
            strided_buffer_idx = strided_buffer_idx_candidate

            if field_frames.statuses[strided_buffer_idx] == FrameStatus.LOADED:
                return self._read_item(
                    multifield_dataset_field,
                    sequential_idx=sequential_idx,
                    buffer_idx=strided_buffer_idx,
                )

        # Item did not exist in cache so we will read it from storage

        # We lock to prevent multiple processes from querying storage for
        # the same chunk simultaneously (we would either contend here or
        # later when handling the chunk data, this is more efficient)
        with data_storage.get_chunk_lock(dataset_local_frame_idx):
            if strided_buffer_idx is not None:
                # We check again because another process might have just loaded
                # the frame we want while we were waiting for the lock.
                if field_frames.statuses[strided_buffer_idx] == FrameStatus.LOADED:
                    return self._read_item(
                        multifield_dataset_field,
                        sequential_idx=sequential_idx,
                        buffer_idx=strided_buffer_idx,
                    )

            return self._load_chunk_frame_cache(
                multifield_dataset_field=multifield_dataset_field,
                data_storage=data_storage,
                dataset_local_frame_idx=dataset_local_frame_idx,
                dataset_global_offset=dataset_global_offset,
                sequential_idx=sequential_idx,
                buffer_idx=strided_buffer_idx,
            )

    def _load_chunk_frame_cache(
        self,
        multifield_dataset_field: DatasetField,
        data_storage: DataStorage,
        *,
        dataset_local_frame_idx: int,
        dataset_global_offset: int,
        sequential_idx: int,
        buffer_idx: Optional[int],
    ) -> np.ndarray:
        """
        Reads a chunk of data containing a specific frame from data_storage,
        populates the shared memory cache from the loaded data, and then returns
        the specifically requested frame.
        """
        chunk_data = data_storage.get_chunk_data(dataset_local_frame_idx)

        desc = self.frame_desc[multifield_dataset_field]
        frames_loaded = len(chunk_data.data) / desc.num_bytes
        field_frames = self.smem_frames[multifield_dataset_field]

        # We don't yet have a straightforward way to figure out which frames from the
        # current chunk of data are frames we should be holding on to. This is
        # brute force, and we can improve this later if it ends up being a bottleneck.
        for i in range(self.cache_lines):
            if field_frames.statuses[i] == FrameStatus.NOT_LOADED:
                global_frame_idx = field_frames.indices[i]
                chunk_global_start_idx = (
                    chunk_data.first_frame_idx + dataset_global_offset
                )
                if (
                    global_frame_idx >= chunk_global_start_idx
                    and global_frame_idx < chunk_global_start_idx + frames_loaded
                ):
                    field_frames.statuses[i] = FrameStatus.LOADED
                    frame_offset = i * desc.num_bytes
                    field_frames.data[
                        frame_offset : frame_offset + desc.num_bytes
                    ] = chunk_data.get_frame(
                        frame_idx=global_frame_idx - dataset_global_offset
                    )

        if buffer_idx is not None:
            return self._read_item(
                multifield_dataset_field,
                sequential_idx=sequential_idx,
                buffer_idx=buffer_idx,
            )
        else:
            raw_bytes = chunk_data.get_frame(frame_idx=dataset_local_frame_idx)
            # make sure to copy resulting slice to prevent pinning a larger frame
            # in memory
            return np.frombuffer(raw_bytes, dtype=desc.dtype).reshape(desc.dims).copy()


class CachedDatasetAccessor:
    """
    We use a single shared memory cache (see SharedMemoryDatasetCache for details)
    to minimize calls to a high latency network storage layer. During training
    we expect to have multiple machines, each with multiple processes. Each of
    those processes is reading a disjoint subset of the total dataset, a dataset
    which is composed of many individual datasets concatenated together. The
    individual datasets have multiple fields which each have a different data
    frame format/shape.

    Every dataset (including the total concatenated dataset and all the individual
    datasets) will have an CachedDatasetAccessor which handles reading the correct
    subset of data from the shared memory cache. This requires keeping track of
    the process "ID" (a 0-based sequential index per participating machine),
    the offset of the current dataset inside the total concatenated dataset, and
    the multifield_dataset_field of the particular field being cached.
    """

    def __init__(
        self,
        shared_mem_dataset_cache: SharedMemoryDatasetCache,
        *,
        process_idx: int,
        dataset_global_offset: int = 0,
        multifield_dataset_field: Optional[DatasetField] = None,
    ) -> None:
        if process_idx >= shared_mem_dataset_cache.local_worker_count:
            raise RuntimeError(
                "Process idx cannot be larger than the number of processes the "
                + "cache was built for"
            )
        self.process_idx = process_idx
        self.cache = shared_mem_dataset_cache
        self.dataset_global_offset = dataset_global_offset
        self.multifield_dataset_field = multifield_dataset_field

    def with_dataset_global_offset(
        self, dataset_global_offset: int
    ) -> "CachedDatasetAccessor":
        return CachedDatasetAccessor(
            self.cache,
            process_idx=self.process_idx,
            dataset_global_offset=dataset_global_offset,
            multifield_dataset_field=self.multifield_dataset_field,
        )

    def with_multifield_dataset_field(
        self, multifield_dataset_field: DatasetField
    ) -> "CachedDatasetAccessor":
        return CachedDatasetAccessor(
            self.cache,
            process_idx=self.process_idx,
            dataset_global_offset=self.dataset_global_offset,
            multifield_dataset_field=multifield_dataset_field,
        )

    def set_sequence_indices(self, frame_indices: List[int]) -> None:
        self.cache.set_sequence_indices(self.process_idx, frame_indices)

    def get_lock_pool(self) -> List[synchronize.Lock]:
        return self.cache.chunk_locks

    def get_frame_data(
        self, dataset_local_frame_idx: int, storage: DataStorage
    ) -> np.ndarray:
        assert (
            self.multifield_dataset_field is not None
        ), "multifield_dataset_field must be set before reading frame data"
        return self.cache.read_frame_through_cache(
            process_idx=self.process_idx,
            multifield_dataset_field=self.multifield_dataset_field,
            dataset_local_frame_idx=dataset_local_frame_idx,
            dataset_global_offset=self.dataset_global_offset,
            data_storage=storage,
        )


class MultifieldDataset(Dataset, Sized):
    """
    A prefetched dataset has one shared memory cache shared across processes.
    """

    data_type: Split

    @abstractmethod
    def set_accessor(self, accessor: CachedDatasetAccessor) -> None:
        """
        Must be called on every MultifieldDataset to initialize an accessor to
        the shared cache. This accessor takes care of reading only the subset of
        data in the shared cache relevant to this instance of this dataset. See
        CachedDatasetAccessor for more details on the purpose of an accessor.
        """
        pass

    @abstractmethod
    def get_raw_item(self, idx: int) -> Dict[DatasetField, np.ndarray]:
        """
        Get a raw record from the dataset. This is the data we would cache in
        order to avoid hitting the storage layer. This might not be the same
        as the final record format, because for example the final record might
        be transformed/augmented on the fly during access.
        """
        pass


class ConcatMultifieldDataset(MultifieldDataset, ConcatDataset):
    def __init__(self, datasets: Sequence[MultifieldDataset]):
        ConcatDataset.__init__(self, datasets=datasets)

    def set_accessor(self, accessor: CachedDatasetAccessor) -> None:
        for idx, dataset in enumerate(self.datasets):
            offset = self.cumulative_sizes[idx - 1] if idx > 0 else 0
            dataset.set_accessor(accessor.with_dataset_global_offset(offset))

    def get_raw_item(self, idx: int) -> Dict[DatasetField, np.ndarray]:
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].get_raw_item(sample_idx)


class SubsetMultifieldDataset(MultifieldDataset, Subset):
    def __init__(self, dataset: MultifieldDataset, indices: Sequence[int]):
        Subset.__init__(self, dataset=dataset, indices=indices)

    def set_accessor(self, accessor: CachedDatasetAccessor) -> None:
        self.dataset.set_accessor(accessor)

    def get_raw_item(self, idx: int) -> Dict[DatasetField, np.ndarray]:
        return self.dataset.get_raw_item(idx)


class FrameStatus(IntEnum):
    NOT_LOADED = 0
    LOADED = 1


np_types = ["uint8", "int8", "int16", "int32", "int64", "float32", "float64", None]


class IndexedDatasetReader(DataStorage, Sized, ABC):

    scheme = ""

    dtype: np.dtype
    N: int
    S: int
    ndim: int
    size: np.ndarray
    framesize: int

    def _init_from_index_data(self, idx: np.ndarray) -> None:
        # assert idx[0] == 0x584449544E54, "unrecognized index format"
        # assert idx[1] == 1, "unsupported format version"
        code = idx[2]

        np_type = np_types[code - 1]
        assert np_type, "unrecognized type"
        self.dtype = np.dtype(np_type)
        assert self.dtype.itemsize == idx[3]

        self.N = int(idx[4])
        self.S = int(idx[5])

        ofs = 6
        dimoffsets = idx[ofs : ofs + self.N + 1]
        ofs += self.N + 1
        datoffsets = idx[ofs : ofs + self.N + 1]
        ofs += self.N + 1
        sizes = idx[ofs : ofs + self.S]

        # We already assume that every frame has the same size for the caching layer
        # so we will assume that here too.
        self.ndim = dimoffsets[1] - dimoffsets[0]
        so = dimoffsets[0]
        self.size = sizes[so : so + self.ndim]
        assert (
            datoffsets[0] == 0
        ), "first data frame must be at the start of the .bin file"
        self.framesize = datoffsets[1] - datoffsets[0]

    def __len__(self) -> int:
        return self.N

    def set_accessor(self, accessor: CachedDatasetAccessor) -> None:
        return

    @abstractmethod
    def __getitem__(self, index: int):
        pass


class IndexedDatasetWriter:
    def __init__(self, *, idxfile: BytesIO, binfile: BytesIO) -> None:
        self._idxfile = idxfile
        self._binfile = binfile
        self._sizes: List[Tuple[int, ...]] = []
        self._dtype = None

    def _generate_idx(self) -> np.ndarray:
        N = len(self._sizes)
        assert N > 0, "Cannot write empty dataset"
        S = sum(len(s) for s in self._sizes)

        dtype = self._dtype
        # pyre-fixme[25]: Assertion will always fail.
        assert dtype is not None

        idx = np.empty(6 + (N + 1) * 2 + S, dtype="int64")
        idx[0] = 0
        idx[1] = 0
        idx[2] = np_types.index(dtype.name) + 1
        idx[3] = dtype.itemsize
        idx[4] = N
        idx[5] = S
        ofs = 6
        idx[ofs : ofs + N + 1] = np.insert(
            np.cumsum([len(s) for s in self._sizes]), 0, 0
        )
        ofs += N + 1
        idx[ofs : ofs + N + 1] = np.insert(
            np.cumsum([np.prod(s) for s in self._sizes]), 0, 0
        )
        ofs += N + 1
        idx[ofs : ofs + S] = list(chain(*self._sizes))
        return idx

    def flush(self) -> None:
        idx = self._generate_idx()
        self._idxfile.write(idx.tobytes())

    def push_back(self, frame: np.ndarray) -> None:
        self._sizes.append(frame.shape)
        if self._dtype is not None:
            assert self._dtype == frame.dtype, "Frames must all have same dtype"
        else:
            self._dtype = frame.dtype
        self._binfile.write(frame.tobytes())
