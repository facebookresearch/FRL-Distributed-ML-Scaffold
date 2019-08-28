#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from pathlib import PurePosixPath
from typing import Tuple


SCHEME_SEPARATOR = "://"


class StorageType:
    scheme = ""


def decode_scheme_path(path: str) -> Tuple[str, PurePosixPath]:
    if SCHEME_SEPARATOR in path:
        components = path.split(SCHEME_SEPARATOR)
        return (components[0], PurePosixPath(components[1]))
    return ("file", PurePosixPath(path))


def encode_scheme_path(scheme: str, path: PurePosixPath) -> str:
    return "%s://%s" % (scheme, path)


class StoragePath:
    def __init__(self, path: str) -> None:
        self._scheme, self._storage_path = decode_scheme_path(path)

    def format(self, path: PurePosixPath) -> str:
        return encode_scheme_path(self._scheme, path)

    def __str__(self) -> str:
        return self.format(self._storage_path)

    def with_suffix(self, suffix: str) -> "StoragePath":
        return self.__class__(self.format(self._storage_path.with_suffix(suffix)))

    def __truediv__(self, other: str) -> "StoragePath":
        return self.__class__(self.format(self._storage_path / other))

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return (
            self._storage_path == other._storage_path and self._scheme == other._scheme
        )

    @property
    def scheme(self) -> str:
        return self._scheme

    @property
    def path(self) -> PurePosixPath:
        return self._storage_path
