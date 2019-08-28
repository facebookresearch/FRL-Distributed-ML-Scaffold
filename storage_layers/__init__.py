#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import glob
from os.path import basename, dirname, isfile


py_modules = glob.glob(dirname(__file__) + "/*.py")
pyc_modules = glob.glob(dirname(__file__) + "/*.pyc")
__all__ = [
    basename(f)[:-3] for f in py_modules if isfile(f) and not f.endswith("__init__.py")
] + [
    basename(f)[:-4]
    for f in pyc_modules
    if isfile(f) and not f.endswith("__init__.pyc")
]
