#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from abc import abstractmethod
from typing import (
    Dict,
    Generic,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
)

import torch
import torch.nn as nn
import torch.nn.modules.loss as L

from .problem import BatchMetrics, EpochMetrics, Ordering
from .types import SampleSummary


SampleMetaT = TypeVar("SampleMetaT")
BatchMetaT = TypeVar("BatchMetaT")
TransformT = TypeVar("TransformT", bound=NamedTuple)


class Task(Generic[TransformT, SampleMetaT, BatchMetaT]):
    @property
    @abstractmethod
    def network_head(self) -> nn.Module:
        ...

    @property
    @abstractmethod
    def criterion(self) -> L._Loss:
        ...

    @property
    @abstractmethod
    def criterion_weight(self) -> float:
        ...

    @abstractmethod
    def get_target(
        self, tensors: Dict[str, torch.Tensor], transform: TransformT
    ) -> Tuple[Sequence[torch.Tensor], SampleMetaT]:
        ...

    @abstractmethod
    def compute_batch_metrics(
        self, meta: BatchMetaT, target: Tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> BatchMetrics:
        ...

    @property
    @abstractmethod
    def rankable_metrics(self) -> Set[Tuple[str, Ordering]]:
        ...

    @abstractmethod
    def summarize_epoch_metrics(self, batch_metrics: BatchMetrics) -> EpochMetrics:
        ...

    @abstractmethod
    def summarize_epoch_samples(
        self,
        data: List[torch.Tensor],
        target: Tuple[torch.Tensor, ...],
        meta: BatchMetaT,
        output: torch.Tensor,
        metric: Optional[BatchMetrics],
    ) -> List[SampleSummary]:
        ...
