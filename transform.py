#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, NamedTuple, Sequence, Tuple, TypeVar, Union

import numpy as np
from torch import Tensor

from .types import Split


SampleMetaT = TypeVar("SampleMetaT", bound=NamedTuple)


class Sample(NamedTuple):
    data: Sequence[Tensor]
    target: Sequence[Union[Tensor, Tuple[Tensor, ...]]]


class MultifieldTransform(ABC, Generic[SampleMetaT]):
    def __call__(
        self, data: Dict[str, np.ndarray], split: Split
    ) -> Tuple[Sequence[Tensor], Sequence[Tensor], Dict[str, Any]]:
        sample, meta = self.transform(data, split)
        # pytorch dataset loaders can only aggregate dicts, lists, and tensors
        # pyre-fixme[7]: Expected `Tuple[Sequence[Tensor], Sequence[Tensor],
        #  Dict[str, Any]]` but got `Tuple[Sequence[Tensor], Sequence[Union[Tensor,
        #  Tuple[Tensor, ...]]], Dict[str, Any]]`.
        return (
            sample.data,
            sample.target,
            {k: v for k, v in meta._asdict().items() if v is not None},
        )

    @abstractmethod
    def transform(
        self, data: Dict[str, np.ndarray], split: Split
    ) -> Tuple[Sample, SampleMetaT]:
        pass
