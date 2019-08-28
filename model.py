#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from queue import Queue
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class MultiTaskModel(nn.Module):
    def __init__(
        self,
        model_base: nn.Module,
        additional_layers: List[nn.Module],
        additional_layer_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.model_base = model_base
        self.additional_layers = nn.ModuleList(additional_layers)
        if additional_layer_names is not None:
            assert len(additional_layer_names) == len(additional_layers)
        else:
            additional_layer_names = list(range(len(additional_layers)))
        self.additional_layer_names: List[str] = additional_layer_names

    def final_shared_params(self, outputs: List[torch.Tensor]) -> torch.Tensor:
        trunk_params = list(self.model_base.parameters())

        # All of the heads should be connected to the model_base so without loss
        # of generality we can just traverse the graph from the first head.
        output = outputs[0]

        candidates = Queue()
        # pyre-ignore
        candidates.put(output.grad_fn)
        while not candidates.empty():
            fn = candidates.get()
            for candidate_fn, _ in fn.next_functions:
                if hasattr(candidate_fn, "variable"):
                    if any(candidate_fn.variable is p for p in trunk_params):
                        return candidate_fn.variable
                if candidate_fn is not None:
                    candidates.put(candidate_fn)
        raise RuntimeError("Unable to find any shared parameters in the model")

    def forward(self, x):
        out = self.model_base(x)
        out = [l(out) for l in self.additional_layers]
        return out


class View(nn.Module):
    def __init__(self, dims: Tuple[int, ...]) -> None:
        super(View, self).__init__()
        self.dims = dims

    def forward(self, *input) -> torch.Tensor:
        x: torch.Tensor = input[0]
        y = x.view(self.dims)
        return y


class MulConstant(nn.Module):
    def __init__(self, constant: float) -> None:
        super(MulConstant, self).__init__()
        self.constant = constant

    def forward(self, *input) -> torch.Tensor:
        x: torch.Tensor = input[0]
        y = x * self.constant
        return y


class ListSelect(nn.Module):
    def __init__(self, *, sel_index: int, num_elements: int):
        super(ListSelect, self).__init__()
        self._sel_index = sel_index
        self._num_elements = num_elements

    def forward(self, input):
        assert len(input) == self._num_elements, "number of elements does not match!"
        return input[self._sel_index]
