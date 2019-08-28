#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from contextlib import contextmanager
from typing import Iterator, List, Tuple

from .storage_layers import *  # noqa F403
from .storage_layers.base_path import StoragePath
from .storage_layers.base_storage import StorageLayer
from .storage_layers.posix_storage import PosixStorageLayer


def sizeof_fmt(n_bytes: int) -> str:
    n_scale: float = n_bytes
    scales = ["", "Ki", "Mi", "Gi", "Ti", "Pi"]
    for scale in scales:
        if abs(n_scale) < 1024.0:
            return "%3.1f%sB" % (n_scale, scale)
        n_scale /= 1024.0
    return "%.1f%sB" % (n_scale, scales[-1])


@contextmanager
def storage_open(storage_path: StoragePath, mode: str):
    with StorageLayerFactory.get(storage_path).open(mode) as f:
        yield f


class StorageLayerFactory:
    @staticmethod
    def get(path: StoragePath) -> StorageLayer:
        for cls in StorageLayer.__subclasses__():
            if cls.scheme == path.scheme:
                return cls(path)
        if path.scheme:
            raise ValueError(
                "Unsupported scheme %s for path %s, supported schemes: %Ls"
                % (path.scheme, path, [c.scheme for c in StorageLayer.__subclasses__()])
            )
        return PosixStorageLayer(path)


class storage_os:
    @staticmethod
    def makedirs(
        storage_path: StoragePath, *, exist_ok: bool = False, ttl: int = 0
    ) -> None:
        StorageLayerFactory.get(storage_path).makedirs(exist_ok=exist_ok, ttl=ttl)

    @staticmethod
    def walk(storage_path: StoragePath) -> Iterator[Tuple[str, List[str], List[str]]]:
        yield from StorageLayerFactory.get(storage_path).walk()

    class path:
        @staticmethod
        def getsize(storage_path: StoragePath) -> int:
            return StorageLayerFactory.get(storage_path).getsize()

        @staticmethod
        def getmtime(storage_path: StoragePath) -> float:
            return StorageLayerFactory.get(storage_path).getmtime()

        @staticmethod
        def exists(storage_path: StoragePath) -> bool:
            return StorageLayerFactory.get(storage_path).exists()
