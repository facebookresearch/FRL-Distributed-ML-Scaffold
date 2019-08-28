#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import functools
import logging
import os
from typing import Dict, List, Optional

import cv2
import torch

from . import configurables
from .config import Config
from .transform import Sample
from .types import RunOpts


def mp_process(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)

        console = logging.StreamHandler()
        root.addHandler(console)

        o = f(*args, **kwargs)

        root.handlers.remove(console)

        return o

    return wrapper


def dict_to_str(d: Dict[str, float]) -> str:
    return "".join("{}: {:.3f}\t".format(k, v) for k, v in d.items())


def load_color_backgrounds(backgrounds_path, rgb=True):
    backgrounds = []
    fnames = os.listdir(backgrounds_path)
    print("Loading {} background images...".format(len(fnames)))
    for f in fnames:
        bg = cv2.imread(os.path.join(backgrounds_path, f))
        bg = bg.transpose((2, 0, 1)) / 255.0
        if rgb is False:
            bg = 0.21 * bg[0, :] + 0.72 * bg[1, :] + 0.07 * bg[2, :]
        backgrounds.append(bg)
        if len(backgrounds) % 100 == 0:
            print("Loaded {} / {}...".format(len(backgrounds), len(fnames)))
    return backgrounds


def stack_optional_tensors(
    data_stack: List[Optional[torch.Tensor]], dim: int = 0
) -> Optional[torch.Tensor]:
    if len(data_stack) == 0:
        raise RuntimeError("the data stack is empty! Nothing to stack together!")
    filtered_stack = [t for t in data_stack if t is not None]
    if len(filtered_stack) == 0:
        return None
    elif len(filtered_stack) != len(data_stack):
        raise RuntimeError(
            "all tensors in this stack have to be either all torch.Tensor or all None"
        )
    return torch.cat(filtered_stack, dim=dim)


def consolidate_sample_stack(data_stack: List[Sample], dim: int = 0) -> Sample:
    if len(data_stack) == 0:
        raise RuntimeError("the data stack is empty! Nothing to consolidate!")

    consolidated_data = [
        torch.cat([sample.data[i] for sample in data_stack], dim=dim)
        for i in range(len(data_stack[0].data))
    ]

    consolidated_target = []
    sample_target = data_stack[0].target
    for i, t in enumerate(sample_target):
        consolidated_target.append(
            tuple(
                torch.cat([sample.target[i][j] for sample in data_stack], dim=dim)
                for j in range(len(t))
            )
        )
    return Sample(data=consolidated_data, target=consolidated_target)


def import_config(run_opts: RunOpts) -> None:
    Config.get().configure(
        configurables.CONFIG_MINIBATCH_TIMEOUT_MS, run_opts.minibatchTimeoutMs
    )
    Config.get().configure(configurables.CONFIG_NUM_IO_WORKERS, run_opts.numIOThreads)
