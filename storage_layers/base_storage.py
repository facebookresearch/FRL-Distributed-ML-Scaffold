#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from abc import abstractmethod
from typing import Iterator, List, Tuple

from .base_path import StoragePath


class StorageLayer:
    scheme: str = ""

    def __init__(self, storage_path: StoragePath) -> None:
        self._storage_path = storage_path

    @abstractmethod
    def makedirs(self, exist_ok: bool, ttl: int) -> None:
        ...

    @abstractmethod
    def open(self, mode: str):
        ...

    @abstractmethod
    def getsize(self) -> int:
        ...

    @abstractmethod
    def getmtime(self) -> float:
        ...

    @abstractmethod
    def exists(self) -> bool:
        ...

    @abstractmethod
    def walk(self) -> Iterator[Tuple[str, List[str], List[str]]]:
        ...
