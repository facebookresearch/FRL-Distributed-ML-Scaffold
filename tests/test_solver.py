#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import unittest

import torch

from ..solver import create_lr_scheduler
from ..types import Mode, OptAlgorithm, OptimOpts, RunOpts


class TestSolver(unittest.TestCase):
    def test_create_lr_scheduler_T47589710(self):
        lr: float = 0.01

        optim_opts = OptimOpts(lr=lr, algo=OptAlgorithm.ADAM)
        run_opts = RunOpts(nEpochs=75, mode=Mode.TRAIN, batchSize=16, optim=optim_opts)
        checkpoint_epoch: int = 60

        optimizer: torch.optim.Optimizer = torch.optim.Adam(
            {torch.Tensor()}, lr=lr, weight_decay=0.0001, eps=1e-8, amsgrad=False
        )
        optimizer.param_groups[0]["initial_lr"] = lr

        # we had a bug (T47589710) where create_lr_scheduler did not correctly
        # initialize the scheduler, causing wrong learning rate to be passed to
        # the optimizer during the first epoch after a checkpoint
        create_lr_scheduler(run_opts, optimizer, checkpoint_epoch)

        self.assertEqual(optimizer.param_groups[0]["lr"], 0.001)
