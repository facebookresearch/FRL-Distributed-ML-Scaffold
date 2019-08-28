#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import itertools
import multiprocessing
import os
import unittest
from contextlib import ExitStack
from functools import reduce
from operator import mul
from pathlib import PurePath
from tempfile import TemporaryDirectory
from typing import Callable, Dict, List, NamedTuple, Union

import numpy as np
from mock import patch
from torch.utils.data import Dataset

from ..indexed_dataset import IndexedDatasetWriterFactory, MultifieldIndexedDataset
from ..storage import StoragePath
from ..storage_layers.dataset import IndexedDatasetReader, np_types
from ..storage_layers.posix_storage import PosixIndexedDatasetReader


FIELD = "TEST"
FILE_BASE = "TEST_FILE"
FILE2_BASE = "TEST_FILE2"


class IndexedDatasetContent(NamedTuple):
    index_content: np.ndarray
    data_content: np.ndarray


def get_test_indexed_dataset_content(
    *,
    type_name: str,
    num_frames: int,
    single_frame_dims: List[int],
    start_value: int = 0,
) -> IndexedDatasetContent:
    """
    We only use fixed size frames in practice so that's all we test here for now.
    Torch indexed datasets support arbitrary frame data and don't technically
    require this consistency.
    """
    magic = 0
    version = 1

    type_code = np_types.index(type_name) + 1

    dtype = np.dtype(type_name)
    type_itemsize = dtype.itemsize

    frame_dims = [single_frame_dims] * num_frames
    frame_ndims = [0] + [len(dim) for dim in frame_dims]
    frame_sizes = [0] + [reduce(mul, dim, 1) for dim in frame_dims]

    sum_num_dimensions_over_frames = sum(frame_ndims)
    frame_dim_offsets = np.cumsum(frame_ndims)
    frame_data_offsets = np.cumsum(frame_sizes)

    index_content = np.array(
        [
            magic,
            version,
            type_code,
            type_itemsize,
            num_frames,
            sum_num_dimensions_over_frames,
            *frame_dim_offsets,
            *frame_data_offsets,
            *itertools.chain.from_iterable(frame_dims),
        ],
        dtype="int64",
    )

    data_content = np.arange(0, frame_data_offsets[-1], dtype=dtype) + start_value

    return IndexedDatasetContent(index_content=index_content, data_content=data_content)


def mock_np_content(content: IndexedDatasetContent) -> Callable[[str], np.ndarray]:
    def fake_np_content(filename: Union[str, PurePath], *args, **kwargs) -> np.ndarray:
        if str(filename) == FILE_BASE + ".idx":
            return content.index_content
        elif str(filename) == FILE_BASE + ".bin":
            return content.data_content
        else:
            raise ValueError(f"Unhandled filename {filename}")

    return fake_np_content


def check_frame_equality(
    dataset: Dataset, frame_idx: int, single_frame_dims: List[int]
) -> bool:
    frame_element_count = reduce(mul, single_frame_dims, 1)
    frame_contents = dataset[frame_idx][FIELD]
    expected_frame_contents = frame_idx * frame_element_count + np.arange(
        0, frame_element_count, dtype=frame_contents.dtype
    ).reshape(single_frame_dims)
    return np.array_equal(frame_contents, expected_frame_contents)


class TestPosixIndexedDatasetReader(unittest.TestCase):
    def test_reading_element(self):
        num_frames = 10
        single_frame_dims = (128, 128)
        content = get_test_indexed_dataset_content(
            type_name="float64",
            num_frames=num_frames,
            single_frame_dims=single_frame_dims,
        )
        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "frldistml.scaffold.storage_layers.posix_storage.np.memmap",
                    side_effect=mock_np_content(content),
                )
            )
            stack.enter_context(
                patch(
                    "frldistml.scaffold.storage_layers.posix_storage.np.fromfile",
                    side_effect=mock_np_content(content),
                )
            )
            dataset = MultifieldIndexedDataset(
                folder_path=StoragePath(""), fields=[FIELD], filenames=[FILE_BASE]
            )
            self.assertEqual(len(dataset), num_frames)
            self.assertEqual(dataset[0][FIELD].shape, single_frame_dims)
            self.assertEqual(dataset[num_frames - 1][FIELD].shape, single_frame_dims)
            self.assertTrue(check_frame_equality(dataset, 0, single_frame_dims))

    def test_multiprocessing(self):
        num_frames = 10
        content = get_test_indexed_dataset_content(
            type_name="float64", num_frames=num_frames, single_frame_dims=(128, 128)
        )
        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "frldistml.scaffold.storage_layers.posix_storage.np.memmap",
                    side_effect=mock_np_content(content),
                )
            )
            stack.enter_context(
                patch(
                    "frldistml.scaffold.storage_layers.posix_storage.np.fromfile",
                    side_effect=mock_np_content(content),
                )
            )
            dataset = PosixIndexedDatasetReader(
                idxfile=StoragePath(FILE_BASE + ".idx"),
                binfile=StoragePath(FILE_BASE + ".bin"),
            )
            frames = {idx: dataset[idx] for idx in [0, len(dataset) - 2]}
            p = multiprocessing.Process(
                target=mp_dataset_worker, args=(dataset, len(dataset), frames)
            )
            p.start()
            p.join()
            self.assertEqual(p.exitcode, 0)


class TestPosixIndexedDatasetWriter(unittest.TestCase):
    def test_write_read(self):
        data1 = np.arange(100).reshape(10, 5, 2)
        data2 = data1 + 1000
        with TemporaryDirectory() as d:
            idxpath = StoragePath(os.path.join(d, "dataset.idx"))
            binpath = StoragePath(os.path.join(d, "dataset.bin"))
            with IndexedDatasetWriterFactory.get(
                idxfile=idxpath, binfile=binpath
            ) as writer:
                writer.push_back(data1)
                writer.push_back(data2)

            reader = PosixIndexedDatasetReader(idxfile=idxpath, binfile=binpath)
            self.assertTrue(np.array_equal(data1, reader[0]))
            self.assertTrue(np.array_equal(data2, reader[1]))


def mp_dataset_worker(
    dataset: IndexedDatasetReader, orig_len: int, frames: Dict[int, np.ndarray]
) -> None:
    assert orig_len == len(dataset), "Dataset size must match original"
    for idx, frame_data in frames.items():
        assert np.array_equal(dataset[idx], frame_data)
