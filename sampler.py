#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import math
from typing import Iterator, List

import torch

from .storage_layers.dataset import MultifieldDataset
from .types import ShuffleType


def per_node_randperm(
    max: int, *, node_idx: int, node_count: int, generator: torch.Generator
) -> List[int]:
    target_chunk_size = math.ceil(max / node_count)
    start_idx = node_idx * target_chunk_size
    actual_chunk_size = min(max - start_idx, target_chunk_size)
    indices = (
        torch.randperm(actual_chunk_size, generator=generator)
        + target_chunk_size * node_idx
    )
    indices = indices.tolist()

    # Every GPU needs an identical quantity of work for pytorch DDP. Since the
    # total number of frames might not be evenly divisible by the number of
    # participating GPUs, the last GPU might end up with a short amount of work.
    # We pad this GPU's work by recycling a few frames to even things out.
    indices += indices[: (target_chunk_size - actual_chunk_size)]
    return indices


class ScaffoldSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(
        self,
        dataset: MultifieldDataset,
        *,
        shuffle_type: ShuffleType,
        node_idx: int,
        node_count: int,
    ) -> None:
        super().__init__(
            dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
        )
        self._shuffle_type = shuffle_type
        self._node_idx = node_idx
        self._node_count = node_count

    def __iter__(self) -> Iterator[int]:
        # deterministically shuffle based on epoch
        g = torch.Generator()
        # pyre-ignore This exists
        g.manual_seed(self.epoch)
        if self._shuffle_type == ShuffleType.PER_NODE_RANDPERM:
            node_size = self.num_replicas // self._node_count
            local_rank = self.rank % node_size
            return iter(
                per_node_randperm(
                    len(self.dataset),
                    node_idx=self._node_idx,
                    node_count=self._node_count,
                    generator=g,
                )[local_rank::node_size]
            )
        elif self._shuffle_type == ShuffleType.RANDPERM:
            if self.shuffle:
                indices = torch.randperm(len(self.dataset), generator=g).tolist()
            else:
                indices = list(range(len(self.dataset)))
        else:
            raise ValueError("Unhandled shuffle type %s", self._shuffle_type)

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)
