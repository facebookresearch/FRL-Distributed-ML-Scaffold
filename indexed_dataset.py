#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import logging
import os
from contextlib import contextmanager
from functools import partial
from multiprocessing import Pool
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
from torch import Tensor

from .config import Config
from .configurables import CONFIG_NUM_IO_WORKERS
from .storage import StoragePath, storage_open, storage_os
from .storage_layers.dataset import (
    CachedDatasetAccessor,
    ConcatMultifieldDataset,
    DatasetField,
    IndexedDatasetReader,
    IndexedDatasetWriter,
    MultifieldDataset,
)
from .storage_layers.posix_storage import PosixIndexedDatasetReader
from .transform import MultifieldTransform
from .types import Split
from .utils import mp_process


logger = logging.getLogger(__name__)


RawDatasets = Dict[DatasetField, IndexedDatasetReader]


class IndexedDatasetReaderFactory:
    @staticmethod
    def get(idxfile: StoragePath, binfile: StoragePath) -> IndexedDatasetReader:
        assert idxfile.scheme == binfile.scheme
        for cls in IndexedDatasetReader.__subclasses__():
            if cls.scheme == idxfile.scheme:
                # pyre-fixme[28]: Unexpected keyword argument `idxfile`.
                return cls(idxfile=idxfile, binfile=binfile)
        return PosixIndexedDatasetReader(idxfile=idxfile, binfile=binfile)


class IndexedDatasetWriterFactory:
    @staticmethod
    @contextmanager
    def get(
        idxfile: StoragePath, binfile: StoragePath
    ) -> Iterator[IndexedDatasetWriter]:
        assert idxfile.scheme == binfile.scheme
        with storage_open(idxfile, "wb") as idx_f, storage_open(binfile, "wb") as bin_f:
            writer = IndexedDatasetWriter(idxfile=idx_f, binfile=bin_f)
            yield writer
            writer.flush()


class MultifieldIndexedDataset(MultifieldDataset):
    """
    Dataset that merge several IndexedDatasets
    deeplearning/projects/faireq-py/fairseq/indexed_dataset.py
    corresponding to different fields.

    Description:
        If there are multiple torchnet IndexedDataset files describing different
        properties of same data sample, we sample at the same index from all dataset and
        save the data in a dictionary indexed by different field names.
    Parameters:
        folder_path: folder that contains multiple tnt IndexDataset files, each
        one named by the corresponding field
        fields: names use to represent different properties in the dictionary
        filenames: corresponding filenames for different properties
    """

    def __init__(
        self,
        folder_path: StoragePath,
        *,
        fields: List[DatasetField],
        filenames: List[str],
    ) -> None:
        # first create one IndexedDataset for each field
        assert len(fields) == len(
            filenames
        ), "Number of properties should equal number of filenames"
        self.datasets: RawDatasets = {}
        logger.info("Create dataset from %s", folder_path)
        logger.info("Files %s", " ".join(filenames))
        for i in range(len(fields)):
            self.datasets[fields[i]] = self.get_dataset(folder_path, filenames[i])

        self.length = len(self.datasets[fields[0]])

        arbitrary_dataset_length = len(self.datasets[fields[0]])
        for field, dataset in self.datasets.items():
            assert len(dataset) == arbitrary_dataset_length, (
                "dataset %s should have same number of samples as %s (%d vs %d)"
                % (field, fields[0], len(dataset), arbitrary_dataset_length)
            )

    def __len__(self) -> int:
        return self.length

    def get_raw_item(self, idx: int) -> Dict[DatasetField, np.ndarray]:
        return self[idx]

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        return {field: dataset[index] for field, dataset in self.datasets.items()}

    def get_dataset(
        self, folder_path: StoragePath, filename: str
    ) -> IndexedDatasetReader:
        bin_path = folder_path / (filename + ".bin")
        idx_path = folder_path / (filename + ".idx")
        return IndexedDatasetReaderFactory.get(idxfile=idx_path, binfile=bin_path)

    def set_accessor(self, accessor: CachedDatasetAccessor) -> None:
        for key, dataset in self.datasets.items():
            dataset.set_accessor(accessor.with_multifield_dataset_field(key))


def find_split_folders(
    root_path: StoragePath, splits: List[Split], filenames: List[str]
) -> Dict[Split, List[StoragePath]]:
    split_strs = [s.value for s in splits]
    data_folders: Dict[Split, List[StoragePath]] = {split: [] for split in splits}
    for root, _dirs, files in storage_os.walk(root_path):
        # select any folder that has this name, e.g. training/testing/heldOut
        folder_name = os.path.basename(root)
        if folder_name in split_strs and len(files) > 0:
            if all(f in files for f in filenames):
                data_folders[Split(folder_name)].append(StoragePath(root))
            elif any(f in files for f in filenames):
                logger.info(
                    "Skipping data directory "
                    + root
                    + " because the following dataset files were missing: "
                    + ", ".join(f for f in filenames if f not in files)
                )
            else:
                logger.info(
                    "Skipping data directory "
                    + root
                    + " because couldn't find at least one dataset file "
                )
    return data_folders


@mp_process
def _make_multifield_dataset(
    fields: List[DatasetField], filenames: List[str], folder: StoragePath
) -> MultifieldIndexedDataset:
    return MultifieldIndexedDataset(folder, fields=fields, filenames=filenames)


class MultiFolderDataset(MultifieldDataset):
    """
    Traverse a folder to find all subfolders for a specific datatype,
    and create a dataset for it.

    Details:
    Assume that the data folder are always named by datatype (e.g. training,
    testing, heldout), traverse the root folder to find all data folders, and
    generate MultifieldIndexedDataset for each data folder. The final dataset
    is a concatenation of all data folder datasets.
    """

    def __init__(
        self,
        leaf_folders: List[StoragePath],
        data_type: Split,
        transform: Optional[MultifieldTransform],
        *,
        fields: List[DatasetField],
        filenames: List[str],
    ) -> None:
        make_dataset = partial(_make_multifield_dataset, fields, filenames)
        with Pool(Config.get().read_key(CONFIG_NUM_IO_WORKERS)) as p:  # pyre-ignore
            datasets = p.map(make_dataset, leaf_folders)
        # by default it's an empty list
        self._fields = fields.copy()
        self._dataset: Optional[MultifieldDataset] = None
        # ConcatDataset fatals if given an empty list of datasets to concat, so
        # we special case handling when there are no datasets found.
        if len(datasets) > 0:
            self._dataset = ConcatMultifieldDataset(datasets)
        self.data_type = data_type
        self.transform = transform

    def set_accessor(self, accessor: CachedDatasetAccessor) -> None:
        if self._dataset:
            # pyre-fixme[16]: `Optional` has no attribute `set_accessor`.
            self._dataset.set_accessor(accessor)

    def __len__(self):
        return 0 if self._dataset is None else len(self._dataset)

    def get_raw_item(self, idx: int) -> Dict[DatasetField, np.ndarray]:
        if self._dataset is None:
            raise RuntimeError("Cannot get element from empty dataset")
        # pyre-fixme[16]: `Optional` has no attribute `__getitem__`.
        return self._dataset[idx]

    def __getitem__(
        self, idx: int
    ) -> Tuple[Tensor, Optional[List[Tensor]], Optional[Any]]:
        data = self.get_raw_item(idx)

        target, meta = None, None
        if self.transform is not None:
            # pyre-fixme[29]: `Optional[MultifieldTransform[Any]]` is not a function.
            data, target, meta = self.transform(data, split=self.data_type)

        return data, target, meta

    @staticmethod
    def generateSplitDatasets(
        root: str,
        splits: List[Split],
        transform: Optional[MultifieldTransform],
        *,
        fields: List[DatasetField],
        filenames: List[str],
    ) -> Dict[Split, "MultiFolderDataset"]:
        storage_path = StoragePath(root)
        logger.info("Loading all splits")
        leaf_folders = find_split_folders(
            storage_path, splits, [f + ".bin" for f in filenames]
        )
        out: Dict[Split, MultiFolderDataset] = {}
        for split, folders in leaf_folders.items():
            logger.info("Found %d leaf %s data folders" % (len(folders), split))
            out[split] = MultiFolderDataset(
                folders, split, transform, fields=fields, filenames=filenames
            )
        return out
