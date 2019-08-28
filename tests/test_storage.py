#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import unittest
from pathlib import PurePath

from frldistml.scaffold.storage import (
    StorageLayerFactory,
    StoragePath,
    storage_open,
    storage_os,
)
from frldistml.scaffold.storage_layers.posix_path import PosixStorageType
from mock import mock_open, patch


class TestPaths(unittest.TestCase):
    posix_implicit_path = "blah"
    posix_explicit_path = "file://blah"
    invalid_explicit_path = "invalid_storage://blah"

    def test_posix_explicit(self):
        str_path = "test"
        storage_path = StoragePath("file://" + str_path)
        self.assertEqual(storage_path.scheme, PosixStorageType.scheme)
        self.assertEqual(storage_path.path, PurePath(str_path))

    def test_posix_implicit(self):
        str_path = "test"
        storage_path = StoragePath(str_path)
        self.assertEqual(storage_path.scheme, PosixStorageType.scheme)
        self.assertEqual(storage_path.path, PurePath(str_path))

    def test_invalid_explicit(self):
        storage_path = StoragePath(self.invalid_explicit_path)
        with self.assertRaises(ValueError):
            StorageLayerFactory.get(storage_path)

    # open
    def test_storage_open_posix_implicit(self):
        with patch("builtins.open", mock_open()) as open_mock:
            with storage_open(StoragePath(self.posix_implicit_path), "rb"):
                open_mock.assert_called_once()

    def test_storage_open_posix_explicit(self):
        with patch("builtins.open", mock_open()) as open_mock:
            with storage_open(StoragePath(self.posix_explicit_path), "rb"):
                open_mock.assert_called_once()

    # os.makedirs
    def test_storage_os_makedirs_posix_implicit(self):
        with patch(
            "frldistml.scaffold.storage_layers.posix_storage.PosixStorageLayer.makedirs"
        ) as os_makedirs:
            storage_os.makedirs(StoragePath(self.posix_implicit_path))
            os_makedirs.assert_called_once()

    def test_storage_os_makedirs_posix_explicit(self):
        with patch(
            "frldistml.scaffold.storage_layers.posix_storage.PosixStorageLayer.makedirs"
        ) as os_makedirs:
            storage_os.makedirs(StoragePath(self.posix_explicit_path))
            os_makedirs.assert_called_once()

    # os.path.exists
    def test_storage_os_path_exists_posix_implicit(self):
        with patch(
            "frldistml.scaffold.storage_layers.posix_storage.PosixStorageLayer.exists"
        ) as os_path_exists:
            storage_os.path.exists(StoragePath(self.posix_implicit_path))
            os_path_exists.assert_called_once()

    def test_storage_os_path_exists_posix_explicit(self):
        with patch(
            "frldistml.scaffold.storage_layers.posix_storage.PosixStorageLayer.exists"
        ) as os_path_exists:
            storage_os.path.exists(StoragePath(self.posix_explicit_path))
            os_path_exists.assert_called_once()

    # os.path.getmtime
    def test_storage_os_path_getmtime_posix_implicit(self):
        with patch(
            "frldistml.scaffold.storage_layers.posix_storage.PosixStorageLayer.getmtime"
        ) as os_path_getmtime:
            storage_os.path.getmtime(StoragePath(self.posix_implicit_path))
            os_path_getmtime.assert_called_once()

    def test_storage_os_path_getmtime_posix_explicit(self):
        with patch(
            "frldistml.scaffold.storage_layers.posix_storage.PosixStorageLayer.getmtime"
        ) as os_path_getmtime:
            storage_os.path.getmtime(StoragePath(self.posix_explicit_path))
            os_path_getmtime.assert_called_once()

    # os.path.getsize
    def test_storage_os_path_getsize_posix_implicit(self):
        with patch(
            "frldistml.scaffold.storage_layers.posix_storage.PosixStorageLayer.getsize"
        ) as os_path_getsize:
            storage_os.path.getsize(StoragePath(self.posix_implicit_path))
            os_path_getsize.assert_called_once()

    def test_storage_os_path_getsize_posix_explicit(self):
        with patch(
            "frldistml.scaffold.storage_layers.posix_storage.PosixStorageLayer.getsize"
        ) as os_path_getsize:
            storage_os.path.getsize(StoragePath(self.posix_explicit_path))
            os_path_getsize.assert_called_once()

    # StoragePath serialization

    def test_posix_serialize_deserialize(self):
        orig = StoragePath(self.posix_explicit_path)
        recreated = StoragePath(str(orig))
        self.assertEqual(orig.path, recreated.path)
