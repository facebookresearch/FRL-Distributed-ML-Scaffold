#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from abc import abstractmethod
from itertools import chain
from typing import (
    Any,
    Dict,
    Generic,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
)

import torch

from .model import MultiTaskModel
from .problem import BatchMetrics, EpochMetrics, Ordering, Problem
from .task import Task
from .transform import MultifieldTransform, Sample
from .types import SampleSummary, Split


SampleMetaT = TypeVar("SampleMetaT", bound=NamedTuple)
BatchMetaT = TypeVar("BatchMetaT")
AnnoParamT = TypeVar("AnnoParamT")
TransformT = TypeVar("TransformT", bound=NamedTuple)


class MultiTaskTransform(
    Generic[TransformT, SampleMetaT], MultifieldTransform[SampleMetaT]
):
    SampleMetaType: Type[SampleMetaT]
    _tasks: Sequence[Task]

    def __init__(
        self, tasks: Sequence[Task], SampleMetaType: Type[SampleMetaT]
    ) -> None:
        self._tasks = tasks
        self.SampleMetaType = SampleMetaType

    @abstractmethod
    def transform_source_data(
        self, tensors: Dict[str, torch.Tensor], split: Split
    ) -> Tuple[Sequence[torch.Tensor], TransformT]:
        ...

    def transform(
        self, data: Dict[str, Any], split: Split
    ) -> Tuple[Sample, SampleMetaT]:
        tensors = {k: torch.from_numpy(v) for k, v in data.items()}
        source_data, transform = self.transform_source_data(tensors, split)

        target_data = []
        meta_data = {}

        for t in self._tasks:
            target, meta = t.get_target(tensors, transform)
            target_data.append(target)
            meta_data.update(meta._asdict())

        sample = Sample(data=source_data, target=target_data)
        return (sample, self.SampleMetaType(**meta_data))


class MultiTaskProblem(
    Generic[BatchMetaT, AnnoParamT], Problem[BatchMetaT, AnnoParamT]
):
    _tasks: Sequence[Task]
    BatchMetaType: Type[BatchMetaT]

    def get_model(self) -> torch.nn.Module:
        return MultiTaskModel(
            model_base=self.get_model_base(),
            additional_layers=[t.network_head for t in self._tasks],
        )

    @abstractmethod
    def get_model_base(self) -> torch.nn.Module:
        ...

    def compute_batch_metrics(
        self,
        meta: BatchMetaT,
        target: List[Tuple[torch.Tensor, ...]],
        output: List[torch.Tensor],
        device: torch.device,
    ) -> BatchMetrics:
        metrics = {}
        for idx, t in enumerate(self._tasks):
            metrics.update(t.compute_batch_metrics(meta, target[idx], output[idx]))
        return metrics

    def get_rankable_metric(self) -> Tuple[str, Ordering]:
        metrics = list(chain(*[t.rankable_metrics for t in self._tasks]))
        return metrics[0]

    def summarize_epoch_metrics(self, batch_metrics: BatchMetrics) -> EpochMetrics:
        epoch_metrics = {}
        for t in self._tasks:
            epoch_metrics.update(t.summarize_epoch_metrics(batch_metrics))
        return epoch_metrics

    def summarize_epoch_samples(
        self,
        data: List[torch.Tensor],
        target: List[Tuple[torch.Tensor, ...]],
        meta: BatchMetaT,
        output: List[torch.Tensor],
        metric: Optional[BatchMetrics] = None,
    ) -> List[SampleSummary]:
        return list(
            chain(
                *[
                    t.summarize_epoch_samples(
                        data, target[idx], meta, output[idx], metric
                    )
                    for idx, t in enumerate(self._tasks)
                ]
            )
        )

    def refine_batch_meta(self, meta: Dict[str, Any]) -> BatchMetaT:
        return self.BatchMetaType(**meta)
