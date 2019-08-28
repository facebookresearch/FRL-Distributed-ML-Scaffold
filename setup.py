#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from setuptools import setup


setup(
    name="frldistml",
    version="0.1",
    url="https://github.com/facebookresearch/FRL-Distributed-ML-Scaffold",
    license="MIT",
    packages=["frldistml"],
    install_requires=["opencv-python", "plotly", "nbformat"],
)
