#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Generic, List, NamedTuple, Optional, Tuple, TypeVar

import numpy as np
import torch

from .cli import get_repro_args
from .criteria import BaseParallelCriterion
from .storage_layers.dataset import MultifieldDataset
from .types import SampleSummary


BatchMetrics = Dict[str, np.ndarray]
EpochMetrics = Dict[str, float]


class Ordering(Enum):
    ASC = "asc"
    DESC = "desc"


BatchMetaT = TypeVar("BatchMetaT")
AnnoParamT = TypeVar("AnnoParamT")

logger = logging.getLogger(__name__)


TNamedTuple = TypeVar("TNamedTuple", bound=NamedTuple)


class Problem(ABC, Generic[BatchMetaT, AnnoParamT]):
    """
    This defines the problem we are trying to solve.
    """

    @abstractmethod
    def __init__(self, *opts: TNamedTuple) -> None:
        target = self.get_solver_buck_target()
        if not target:
            logger.info(
                "Buck solver target not specified in problem class, unable to "
                + "suggest repro command"
            )
        else:
            repro_cmd = "buck run @mode/dev-nosan %s -- %s" % (
                target,
                get_repro_args(*opts),
            )
            logger.info("To repro this run use: %s", repro_cmd)

    @property
    @abstractmethod
    def datasets(self) -> List[MultifieldDataset]:
        """
        Datasets that read data and generate augmented data samples
        """
        pass

    @property
    @abstractmethod
    def save_dir(self) -> str:
        """
        The output location for checkpoint data, images, and the final trained model
        """
        pass

    @property
    @abstractmethod
    def anno_param(self) -> Optional[AnnoParamT]:
        pass

    @abstractmethod
    def get_model(self) -> torch.nn.Module:
        """
        Generate a model to be trained or evaluated. This could be pretrained or
        randomly initialized.

        The criterion and the model are typically closely coupled.
        """
        pass

    @abstractmethod
    def get_criterion(self) -> BaseParallelCriterion:
        """
        The criterion used to train the model returned from get_model().

        The criterion and the model are typically closely coupled.
        """
        pass

    @abstractmethod
    def refine_batch_meta(self, meta: Dict[str, Any]) -> BatchMetaT:
        """
        Given a string-keyed dictionary, generate a stronger typed NamedTuple
        for the metadata associated with samples in a minibatch.

        Necessary since python3 does not support accessing generic types at
        runtime.
        """
        pass

    @staticmethod
    def get_solver_buck_target() -> Optional[str]:
        """
        Return the buck-target of a python binary that trains this specific problem
        ("//path/to/solver:specialized_solver").

        If specified, when the solver is run it will dump out a command to stdout
        to re-run the same job. Useful for debugging.
        """
        return None

    @abstractmethod
    def compute_batch_metrics(
        self,
        meta: BatchMetaT,
        target: List[Tuple[torch.Tensor, ...]],
        output: List[torch.Tensor],
        device: torch.device,  # suggested device
    ) -> BatchMetrics:
        """
        For a minibatch of data, generate a set of metrics (or data necessary to
        compute metrics, such as true positive/true negative counts). This is called
        for every batch during an epoch, and the aggregated results will be later
        passed to summarize_epoch_metrics().

        This function is responsible for moving necessary data to the same device.
        """
        pass

    @abstractmethod
    def get_rankable_metric(self) -> Tuple[str, Ordering]:
        """
        In order to pick worst samples during an epoch we use the metric returned
        here (with a corresponding ordering, asc/desc) to figure out which samples
        were outliers.
        """
        pass

    @abstractmethod
    def summarize_epoch_samples(
        self,
        data: List[torch.Tensor],
        target: List[Tuple[torch.Tensor, ...]],
        meta: BatchMetaT,
        output: List[torch.Tensor],
        metric: Optional[BatchMetrics] = None,
    ) -> List[SampleSummary]:
        """
        Given a bunch of network input/output and target samples, optionally produce
        an image and text summary that illustrate the performance on these samples.
        The samples might be good, bad, or random, the summary is only illustrative.
        """
        pass

    @abstractmethod
    def summarize_epoch_metrics(self, batch_metrics: BatchMetrics) -> EpochMetrics:
        """
        Given metric data computed for every minibatch in the epoch, produce a final
        aggregated set of metrics for the entire epoch to allow tracking training
        performance or evaluating pretrained models.
        """
        pass
