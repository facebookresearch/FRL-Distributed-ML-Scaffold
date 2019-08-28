#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import unittest

import torch
from frldistml.scaffold.sampler import per_node_randperm


class TestSampler(unittest.TestCase):
    def test_shuffle_12_4_replicas(self) -> None:
        n = 12
        g = torch.Generator()
        shuffle = []
        for i in range(4):
            shuffle += per_node_randperm(max=n, node_idx=i, node_count=4, generator=g)
        self.assertCountEqual(shuffle, list(range(n)))

    def test_shuffle_11_4_replicas(self) -> None:
        n = 11
        g = torch.Generator()
        shuffle = []
        for i in range(4):
            shuffle += per_node_randperm(max=n, node_idx=i, node_count=4, generator=g)
        self.assertCountEqual(shuffle[:-1], list(range(n)))

    def test_shuffle_12_1_replica(self) -> None:
        n = 12
        g = torch.Generator()
        shuffle = per_node_randperm(max=n, node_idx=0, node_count=1, generator=g)
        self.assertCountEqual(shuffle, list(range(n)))

    def test_shuffle_11_1_replica(self) -> None:
        n = 11
        g = torch.Generator()
        shuffle = per_node_randperm(max=n, node_idx=0, node_count=1, generator=g)
        self.assertCountEqual(shuffle, list(range(n)))
