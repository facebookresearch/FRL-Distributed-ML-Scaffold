#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.nn.modules.loss as L

from .types import LossType


class BaseParallelCriterion(nn.Module, ABC):
    @abstractmethod
    def forward(self, *input) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        ...

    @property
    @abstractmethod
    def loss_names(self) -> List[str]:
        ...


class ParallelCriterion(BaseParallelCriterion):
    def __init__(self, loss_modules, loss_weights, loss_names=None) -> None:
        super().__init__()
        self.loss_modules = nn.ModuleList(loss_modules)
        self.loss_weights = loss_weights
        self._loss_names = loss_names

    @property
    def loss_names(self) -> List[str]:
        return self._loss_names

    def compute_split_loss(
        self, input: List[torch.Tensor], target: List[Tuple[torch.Tensor, ...]]
    ) -> Dict[str, torch.Tensor]:

        split_loss = {}
        for i, (loss, weight, name) in enumerate(
            zip(self.loss_modules, self.loss_weights, self.loss_names)
        ):
            split_loss[name] = weight * loss.forward(input[i], *target[i])
        return split_loss

    def forward(self, *input) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        output: List[torch.Tensor]
        target: List[Tuple[torch.Tensor, ...]]
        output, target = input
        split_loss = self.compute_split_loss(output, target)
        # pyre-ignore this is a tensor
        total_loss: torch.Tensor = sum(split_loss.values())

        return total_loss, split_loss


class UncertaintyWeightedCriterion(BaseParallelCriterion):
    """
    This implements the task-uncertainty weighted multi-task loss as described
    in https://arxiv.org/abs/1705.07115. It currently only supports regression
    loss (LossType.MSE) or cross-entropy loss (LossType.CrossEntropy).
    Note that according to this paper, the weight of each objective is a function
    of the task noise sigma, and it's re-paramatrized as log(sigma^2) during
    training.
    """

    def __init__(self, loss_modules, loss_types, loss_names, initial_weights) -> None:
        super().__init__()
        assert (
            len(loss_types)
            == len(loss_modules)
            == len(loss_names)
            == len(initial_weights)
        )
        self.loss_modules = nn.ModuleList(loss_modules)
        self.loss_types = loss_types
        # instead of learning the \sigma, we learn log variance (log \sigma^2)
        # according to the paper
        self.log_variance = nn.Parameter(torch.Tensor(len(loss_modules)))
        initial_log_variance = []
        for i, w in enumerate(initial_weights):
            if loss_types[i] == LossType.MSE:
                initial_log_variance.append(np.log(1 / (2 * w)))
            elif loss_types[i] == LossType.CrossEntropy:
                initial_log_variance.append(np.log(1 / w))
            else:
                raise RuntimeError(
                    "Loss type other than MSE or CrossEntropy is not supported now."
                )
        self.log_variance.data.copy_(torch.tensor(initial_log_variance))
        self._loss_names = loss_names
        for loss_type in loss_types:
            assert (
                loss_type == LossType.MSE or loss_type == LossType.CrossEntropy
            ), "currently only support MSE or CrossEntropy loss"

    @property
    def loss_names(self) -> List[str]:
        return self._loss_names

    def compute_split_loss(
        self, input: List[torch.Tensor], target: List[Tuple[torch.Tensor, ...]]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:

        split_loss = {}
        uncertainty_costs = []
        for i, (loss, loss_type, name) in enumerate(
            zip(self.loss_modules, self.loss_types, self.loss_names)
        ):
            if loss_type == LossType.MSE:
                split_loss[name] = (
                    1.0
                    / (2.0 * torch.exp(self.log_variance[i]))
                    * loss.forward(input[i], *target[i])
                )
                uncertainty_costs.append(0.5 * self.log_variance[i])
            elif loss_type == LossType.CrossEntropy:
                split_loss[name] = (
                    1.0
                    / torch.exp(self.log_variance[i])
                    * loss.forward(input[i], *target[i])
                )
                uncertainty_costs.append(0.5 * self.log_variance[i])
            else:
                raise RuntimeError(
                    "Loss type other than MSE or CrossEntropy is not supported now."
                )
        # pyre-fixme[7]: Expected `Tuple[Dict[str, Tensor], Tensor]` but got
        #  `Tuple[Dict[Any, Any], Union[Any, int]]`.
        return split_loss, sum(uncertainty_costs)

    def forward(self, *input) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        output: List[torch.Tensor]
        target: List[Tuple[torch.Tensor, ...]]
        output, target = input

        split_loss, uncertainty_cost = self.compute_split_loss(output, target)

        total_loss = sum(split_loss.values()) + uncertainty_cost

        return total_loss, split_loss


class GradNormWeightedCriterion(BaseParallelCriterion):
    def __init__(
        self,
        loss_modules: List[L._Loss],
        loss_names: List[str],
        alpha: float,
        base_weights: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        assert len(loss_modules) == len(loss_names)
        self._loss_modules = nn.ModuleList(loss_modules)
        self._loss_names = loss_names

        assert alpha > 0, "alpha must be >0"
        self._alpha = alpha
        self._num_tasks = len(loss_modules)
        self._weight_factors = nn.Parameter(torch.zeros(self._num_tasks))
        self._baseline_loss: Optional[List[float]] = None
        self._shared_params: Optional[torch.Tensor] = None
        self._base_weights = base_weights or [1] * self._num_tasks

    def set_shared_params(self, shared_params: torch.Tensor) -> None:
        self._shared_params = shared_params

    @property
    def loss_names(self) -> List[str]:
        return self._loss_names

    def forward(self, *input) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        output: List[torch.Tensor]
        target: List[Tuple[torch.Tensor, ...]]
        output, target = input

        unweighted_task_losses = [
            self._base_weights[i] * loss.forward(output[i], *target[i])
            for i, loss in enumerate(self._loss_modules)
        ]

        if self._baseline_loss is None:
            self._baseline_loss = [l.item() for l in unweighted_task_losses]

        assert self._baseline_loss is not None

        inv_training_rates = [
            unweighted_task_losses[i] / self._baseline_loss[i]
            for i in range(self._num_tasks)
        ]

        mean_inv_training_rate = sum(inv_training_rates) / len(inv_training_rates)
        normalized_inv_training_rates = [
            inv_rate / mean_inv_training_rate for inv_rate in inv_training_rates
        ]

        # We detach the gradients just before the loss function so that when we
        # backprop the grad_loss below we don't invoke the second derivative of
        # the loss functions, which isn't actually needed to compute the
        # grad_loss gradients w.r.t. the task weights.
        detached_unweighted_loss_grads = [
            g.detach()
            for g in torch.autograd.grad(
                unweighted_task_losses, output, retain_graph=True
            )
        ]

        # This is not in the original paper, but I found that naively
        # parameterized weights can easily become negative. Since the weights
        # should sum to the number of tasks, a natural reparameterization is to
        # softmax the original weight factors.
        # pyre-ignore
        weights = self._weight_factors.softmax(0) * self._num_tasks

        # Each task loss is backproped separately and the gradient contributions
        # are measured at the last shared parameters of the model trunk.
        assert self._shared_params is not None
        grad_norms = [
            torch.autograd.grad(
                output[task_idx],
                self._shared_params,
                weights[task_idx] * detached_unweighted_loss_grads[task_idx],
                retain_graph=True,
                create_graph=True,
            )[0].norm()
            for task_idx in range(self._num_tasks)
        ]

        # The loss function encourages gradient contributions to be similar from
        # each task, but also inversely proportional to the convergence rate of
        # each task. The hyperparameter alpha trades off these two goals.
        mean_grad_norm = sum(grad_norms) / len(grad_norms)
        target_grad_norms = [
            mean_grad_norm * (inv_rate ** self._alpha)
            for inv_rate in normalized_inv_training_rates
        ]
        grad_loss = sum(
            # pyre-fixme[16]: `float` has no attribute `detach`.
            f.l1_loss(grad_norm, target_grad_norm.detach())
            for grad_norm, target_grad_norm in zip(grad_norms, target_grad_norms)
        )

        # We detach the weights here because the individual task losses should not
        # be affecting the weighting directly, or else the weights would just
        # converge to zero. The weights should only be affected by grad_loss.
        weighted_task_losses = [
            weights[task_idx].detach() * unweighted_task_losses[task_idx]
            for task_idx in range(self._num_tasks)
        ]
        # pyre-fixme[9]: total_loss has type `Tensor`; used as `int`.
        total_loss: torch.Tensor = sum(weighted_task_losses) + grad_loss

        return total_loss, dict(zip(self._loss_names, unweighted_task_losses))


# This is a container loss layer. It's only meaningful when the user supplies
# the loss layer to be used. The mask should be a boolean tensor to control
# which entries in the input and target are used to compute the final loss.
# This is useful when we don't have supervision for certain entries
class MaskedLoss(L._Loss):
    def __init__(self, loss_layer, reduction: str = "mean") -> None:
        super(MaskedLoss, self).__init__(reduction=reduction)
        self.loss_layer = loss_layer

    def forward(self, *input) -> torch.Tensor:
        output: torch.Tensor
        target: torch.Tensor
        mask: torch.Tensor
        output, target, mask = input
        assert not target.requires_grad
        assert not mask.requires_grad

        # TODO:T43357373: do not forward if mask.sum() == 0 to prevent undefined
        # behaviour
        if mask.sum() == 0:
            return self.loss_layer.forward(output - output, target - target)
        else:
            mask = mask.bool()
            final_loss = self.loss_layer.forward(output[mask], target[mask])
            return final_loss
